//! `SearChain` — Search-in-the-Chain (Xu et al. 2024, "Search-in-the-Chain:
//! Interactively Enhancing Large Language Models with Search for
//! Knowledge-intensive Tasks").
//!
//! `SearChain` is a **two-phase, backtracking** architecture, distinct from
//! its neighbours in this crate:
//!
//! * `knowledge_conflict` detects contradictions *between already-retrieved
//!   passages* — there is no reasoning chain and no backtracking.
//! * `query_planning` executes a DAG of steps *as it goes*, one dependency
//!   layer at a time — there is no upfront chain-of-query and no
//!   verify-then-backtrack pass.
//!
//! `SearChain`, by contrast:
//!
//! 1. **Phase 1 — Chain-of-Query construction.** A [`ChainGenerator`] builds
//!    the *complete* global reasoning chain in one pass, upfront, before any
//!    retrieval happens: a sequence of [`ChainNode`]s, each with a
//!    sub-query, a tentative (parametric-knowledge-only) answer, and the ids
//!    of earlier nodes it depends on.
//! 2. **Phase 2 — Interactive Reasoning-Verification (IRV).** The chain is
//!    walked node by node. A [`Retriever`] fetches evidence for each node's
//!    sub-query; the evidence is compared against the node's current answer
//!    to produce a [`NodeVerdict`]:
//!    - [`NodeVerdict::Verified`] — evidence supports the answer; it is
//!      kept.
//!    - [`NodeVerdict::Unverified`] — evidence is insufficient or
//!      inconclusive; the answer is replaced with the evidence's best
//!      candidate.
//!    - [`NodeVerdict::Conflicting`] — evidence directly contradicts the
//!      answer. The answer is revised from evidence, **and** every node that
//!      causally depends on it is explicitly re-verified — a genuine,
//!      dependency-aware backtracking pass, not a one-shot linear walk. A
//!      corrected premise cascades transitively through the chain.
//! 3. **Phase 3 — Final-answer assembly.** A [`Generator`] synthesises the
//!    final answer from the verified chain.
//!
//! Backtracking is bounded by
//! [`SearChainConfig::max_backtrack_iterations`]. If the budget is
//! exhausted before every cascade settles, [`SearChainEngine::verify_chain`]
//! does **not** silently return a possibly-wrong answer or mask the
//! situation as an error: it returns an honest
//! [`SearChainStatus::PartialBacktrackExhausted`] naming every node left
//! unresolved.
//!
//! [`MockChainGenerator`], [`MockSearchainRetriever`], and
//! [`MockSearchainGenerator`] provide deterministic implementations of the
//! three pluggable traits, for use in tests.
//!
//! # Example
//!
//! ```
//! use oxirag::searchain::{
//!     AnswerAssemblyStrategy, ChainNode, MockChainGenerator, MockSearchainGenerator,
//!     MockSearchainRetriever, NodeVerdict, SearChainConfig, SearChainEngine,
//! };
//!
//! // Phase 1 (upfront, no retrieval): node 1's tentative answer was written
//! // assuming node 0's tentative answer — a premise that Phase 2 will show
//! // is wrong.
//! let chain = vec![
//!     ChainNode::new(0, "Was the treaty ratified?", "The treaty was ratified."),
//!     ChainNode::new(
//!         1,
//!         "Did trade resume after the treaty?",
//!         "Trade resumed because The treaty was ratified.",
//!     )
//!     .with_depends_on(vec![0]),
//! ];
//! let chain_generator = MockChainGenerator::new(vec![("trade".to_string(), chain)]);
//!
//! // Phase 2: evidence contradicts node 0's tentative answer, which must
//! // cascade into an explicit re-verification of node 1.
//! let retriever = MockSearchainRetriever::new(vec![
//!     (
//!         "Was the treaty ratified".to_string(),
//!         vec!["The treaty was not ratified.".to_string()],
//!     ),
//!     (
//!         "Did trade resume".to_string(),
//!         vec!["Trade resumed because the treaty was not ratified after all.".to_string()],
//!     ),
//! ]);
//!
//! let generator = MockSearchainGenerator::new(AnswerAssemblyStrategy::LastNode);
//! let engine = SearChainEngine::new(SearChainConfig::default());
//! let result = engine
//!     .run(
//!         "Did trade resume after the treaty was ratified?",
//!         &chain_generator,
//!         &retriever,
//!         &generator,
//!     )
//!     .expect("run should succeed");
//!
//! assert!(result.is_complete());
//! assert_eq!(result.chain[0].verdict, Some(NodeVerdict::Conflicting));
//! assert!(result.chain[0].revised);
//! // Node 1 was explicitly re-verified as a consequence of node 0's
//! // backtrack cascade.
//! assert!(result.backtrack_events >= 1);
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::SearChainEngine;
pub use types::{
    AnswerAssemblyStrategy, ChainGenerator, ChainNode, Generator, MockChainGenerator,
    MockSearchainGenerator, MockSearchainRetriever, NodeVerdict, Retriever, SearChainConfig,
    SearChainError, SearChainResult, SearChainStatus,
};
