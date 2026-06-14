//! Nugget-based answer evaluation (TREC-style).
//!
//! This module evaluates a system answer against a gold/reference answer by
//! decomposing the reference into atomic information **nuggets** — short factual
//! units — grading each as [`NuggetImportance::Vital`] or
//! [`NuggetImportance::Okay`], then measuring the weighted fraction of nuggets
//! the system answer **covers**. Coverage is decided lexically via token
//! overlap (a deterministic approximation of lexical entailment).
//!
//! The approach is distinct from the RAGAS-style metrics in the `evaluation`
//! module (which score relevance/faithfulness over retrieved context) and from
//! the `llm_judge` module (which grades against a rubric): here the unit of
//! credit is the individual nugget, and the score reflects how many of the
//! reference's facts survive into the answer.
//!
//! # Pipeline
//!
//! 1. A [`NuggetExtractor`] (default [`HeuristicNuggetExtractor`]) splits the
//!    reference into clauses and emits one [`Nugget`] per qualifying clause.
//! 2. A [`NuggetScorer`] checks, for each nugget, whether enough of its tokens
//!    appear in the answer ([`NuggetScorer::is_covered`]).
//! 3. The result is a [`NuggetScore`] with overall, vital-only, and
//!    importance-weighted coverage plus the covered / missed nugget indices.
//!
//! # Example
//!
//! ```rust
//! use oxirag::nugget_eval::{NuggetConfig, NuggetScorer};
//!
//! let scorer = NuggetScorer::new(NuggetConfig::default());
//! let reference = "Rust was created by Mozilla, and it guarantees memory safety.";
//! let answer = "Rust, made at Mozilla, guarantees memory safety.";
//!
//! let score = scorer.score(reference, answer).expect("non-empty reference");
//! assert!(score.coverage > 0.0);
//! ```

pub mod extractor;
pub mod scorer;
pub mod types;

#[cfg(test)]
mod tests;

pub use extractor::{HeuristicNuggetExtractor, NuggetExtractor};
pub use scorer::NuggetScorer;
pub use types::{Nugget, NuggetConfig, NuggetEvalError, NuggetImportance, NuggetScore};
