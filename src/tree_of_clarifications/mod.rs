//! Tree of Clarifications (`ToC`; Kim et al., EMNLP 2023, "Tree of Clarifications:
//! Answering Ambiguous Questions with Retrieval-Augmented Large Language
//! Models").
//!
//! Many questions posed to a RAG system are genuinely *ambiguous* — they admit
//! several valid readings (`"What is the capital?"` could mean the capital of
//! a country, or the financial capital of a company). Answering under a single
//! guessed interpretation risks silently answering the wrong question. `ToC`
//! instead:
//!
//! 1. **Disambiguates** the question into candidate readings via a
//!    [`Disambiguator`] (an empty result means the question is already
//!    unambiguous, producing a single root-equals-leaf node).
//! 2. **Recurses**: each candidate reading may itself still be ambiguous, so it
//!    is disambiguated again, up to [`ToCConfig::max_depth`] — this builds a
//!    genuine [`ClarificationTree`], not a flat list of readings.
//! 3. **Retrieves and answers** every node's own question via a
//!    [`ClarificationRetriever`] and [`ClarificationAnswerer`], and scores its
//!    `relevance_score` from the lexical overlap between the retrieved
//!    passages and the question (a retrieval-support-based confidence).
//! 4. **Recursively prunes** branches whose `relevance_score` falls below
//!    [`ToCConfig::prune_threshold`]. Because every node — not just leaves —
//!    already carries its own precomputed answer, a node that loses *all* of
//!    its children simply reverts to being a leaf and its own best-effort
//!    answer stands in for the pruned subtree.
//! 5. **Aggregates** every surviving leaf's answer into one long-form answer,
//!    attributed by its disambiguated question.
//!
//! # Distinct from related modules
//!
//! | Module | Structure | Purpose |
//! |--------|-----------|---------|
//! | [`crate::tree_of_thought`] | `ThoughtTree` explored by BFS/DFS search | Branches *reasoning steps* toward a solution, not question readings |
//! | [`crate::graph_of_thought`] | `ThoughtGraph`, a DAG with generate/aggregate/refine/`KeepBest` operations | Merges independent thought branches; not disambiguation-driven |
//! | [`crate::self_ask`] | Linear sequence of follow-up questions | Sequential composition, not a branching disambiguation tree with pruning |
//! | **`tree_of_clarifications`** | `ClarificationTree`, a forest of disambiguated readings | Branches on *question ambiguity*, then prunes weakly supported readings |
//!
//! # Example
//!
//! ```
//! use oxirag::tree_of_clarifications::{
//!     MockClarificationAnswerer, MockClarificationRetriever, MockDisambiguator, ToCConfig,
//!     ToCEngine,
//! };
//!
//! // "What is the capital?" is ambiguous between a country and a company reading.
//! let disambiguator = MockDisambiguator::new(vec![(
//!     "capital".to_string(),
//!     vec![
//!         "What is the capital of France?".to_string(),
//!         "What is the capital of a company?".to_string(),
//!     ],
//! )]);
//! let retriever = MockClarificationRetriever::new(vec![
//!     (
//!         "capital of france".to_string(),
//!         vec!["Paris is the capital of France.".to_string()],
//!     ),
//!     (
//!         "capital of a company".to_string(),
//!         vec!["The capital of a company is called shareholder capital.".to_string()],
//!     ),
//! ]);
//! let answerer = MockClarificationAnswerer;
//!
//! // A single disambiguation round is enough for this example.
//! let config = ToCConfig::default().with_max_depth(1);
//! let engine = ToCEngine::new(config, disambiguator, retriever, answerer);
//!
//! let (tree, answer) = engine.run("What is the capital?").unwrap();
//!
//! assert_eq!(tree.root_questions.len(), 2);
//! assert_eq!(tree.leaves().len(), 2);
//! assert!(answer.contains("Regarding What is the capital of France?"));
//! assert!(answer.contains("Regarding What is the capital of a company?"));
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod tree;
pub mod types;

pub use engine::{ToCEngine, aggregate};
pub use tree::{ClarificationNode, ClarificationTree};
pub use types::{
    ClarificationAnswerer, ClarificationRetriever, Disambiguator, MockClarificationAnswerer,
    MockClarificationRetriever, MockDisambiguator, ToCConfig, ToCError,
};
