//! RQ-RAG: Learning to Refine Queries for Retrieval-Augmented Generation
//! (Chan et al., 2024).
//!
//! RQ-RAG's central idea is that a retrieval pipeline should not treat every
//! incoming query identically. Instead, it first *explicitly classifies* the
//! query into one of four **refinement actions** and then applies the
//! matching strategy to produce the query text(s) that are actually sent to
//! the retriever:
//!
//! - [`RefinementAction::Rewrite`] — the query is a reasonable search target
//!   but its phrasing is ambiguous, colloquial, or otherwise poorly suited to
//!   retrieval (e.g. filler words). Produces **one** rewritten query.
//! - [`RefinementAction::Decompose`] — the query bundles several independent
//!   questions (a compound / multi-part question). Produces **multiple**
//!   sub-queries, one per part.
//! - [`RefinementAction::Disambiguate`] — the query contains an ambiguous
//!   entity or term with multiple plausible readings (e.g. "bank", "mercury",
//!   "python"). Produces **multiple** disambiguated variants, one per reading.
//! - [`RefinementAction::Respond`] — the query is already clear and atomic;
//!   no refinement is necessary, so it passes through unchanged.
//!
//! Each of the (possibly several) resulting queries is intended to be
//! retrieved independently and the retrieved evidence later recombined by the
//! caller — this module owns the *classify-then-refine* step, not the
//! retrieval or recombination step.
//!
//! # Distinct from `query_router`, `query_decomposition`, `self_query`, and
//! `adaptive_rag`
//!
//! | Module | Classifies | Produces |
//! |--------|------------|----------|
//! | `query_router` | Semantic *intent* (factual / comparative / navigational / …) | A retrieval *modality* (vector / hybrid / graph search) — the query **text** itself is left untouched |
//! | `adaptive_rag` | Structural *complexity* (straightforward / single-step / multi-step) | A retrieval *depth* (no retrieval / one pass / iterative) — the query **text** itself is left untouched |
//! | `self_query` | Structured *metadata filters* embedded in the query | A residual semantic query plus extracted filter conditions — never rewrites, decomposes, or disambiguates the query text |
//! | `query_decomposition` | Nothing (always decomposes unconditionally) | Sub-questions only — there is no rewrite/disambiguate branch and no explicit action classification |
//! | **`rq_rag`** (this module) | One of four explicit **refinement actions** | The refined **query text(s)** themselves, ready to be retrieved |
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|-----------------|
//! | [`RefinementAction`] | The four-way action taxonomy |
//! | [`RefinementPlan`] | Classification result + refined query text(s) |
//! | [`QueryRefiner`] | Trait: classify + rewrite + decompose + disambiguate |
//! | [`MockRefiner`] | Deterministic, signal-based default [`QueryRefiner`] |
//! | [`RqRagConfig`] | Decompose threshold + ambiguity/colloquial marker tables |
//! | [`QueryRefinementEngine`] | Orchestrates classify → dispatch → [`RefinementPlan`] |
//!
//! # Quick start
//!
//! ```
//! use oxirag::rq_rag::{MockRefiner, QueryRefinementEngine, RefinementAction, RqRagConfig};
//!
//! let engine = QueryRefinementEngine::new(RqRagConfig::default(), MockRefiner::default());
//!
//! // A compound question is decomposed into multiple independent sub-queries.
//! let plan = engine.run("What is Rust and what is Python?").unwrap();
//! assert_eq!(plan.action, RefinementAction::Decompose);
//! assert!(plan.refined_queries.len() >= 2);
//!
//! // An ambiguous entity is disambiguated into multiple readings.
//! let plan = engine.run("Tell me about mercury").unwrap();
//! assert_eq!(plan.action, RefinementAction::Disambiguate);
//! assert!(plan.refined_queries.len() >= 2);
//!
//! // Colloquial phrasing is rewritten into a single cleaner query.
//! let plan = engine.run("um, like, what is Rust basically").unwrap();
//! assert_eq!(plan.action, RefinementAction::Rewrite);
//! assert_eq!(plan.refined_queries.len(), 1);
//!
//! // A clear, atomic question passes through unchanged.
//! let plan = engine.run("What is Rust?").unwrap();
//! assert_eq!(plan.action, RefinementAction::Respond);
//! assert_eq!(plan.refined_queries, vec!["What is Rust?".to_string()]);
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::{MockRefiner, QueryRefinementEngine, QueryRefiner};
pub use types::{
    AMBIGUITY_SENSE_TABLE, DEFAULT_COLLOQUIAL_MARKERS, RefinementAction, RefinementPlan,
    RqRagConfig, RqRagError,
};
