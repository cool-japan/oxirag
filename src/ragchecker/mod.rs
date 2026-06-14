//! `RAGChecker` — claim-level RAG diagnostics (Ru et al. 2024).
//!
//! `RAGChecker` evaluates a retrieval-augmented generation sample at the
//! granularity of *atomic claims* rather than whole answers, and — crucially —
//! splits the diagnosis into **retriever** and **generator** halves. This is
//! distinct from the RAGAS-style metrics in the `evaluation` module, which score
//! answers and contexts holistically: here every metric is derived by
//! decomposing the response and the ground-truth answer into claims and testing
//! each claim's lexical *entailment* against the retrieved context and the
//! ground truth.
//!
//! A piece of text *entails* a claim when at least
//! [`RagCheckerConfig::entailment_threshold`] of the claim's distinct tokens
//! appear in that text. From those per-claim judgements the checker derives six
//! metrics:
//!
//! - **Retriever** — `claim_recall` (ground-truth claims found in the context)
//!   and `context_precision` (retrieved passages that support a ground-truth
//!   claim).
//! - **Generator** — `faithfulness` (response claims grounded in the context),
//!   `hallucination_rate` (response claims in neither context nor ground truth),
//!   `correctness` (response claims matching the ground truth), and
//!   `noise_sensitivity` (response claims from the context but absent from the
//!   ground truth).
//!
//! The decomposition reuses the same deterministic, model-free strategy as
//! the `claim_decomposition` module: split on sentence terminators, then on
//! clause boundaries. No randomness, network access, or machine-learning model is
//! involved, so results are fully reproducible.
//!
//! # Example
//!
//! ```
//! use oxirag::ragchecker::{RagChecker, RagCheckerConfig};
//! use oxirag::types::Document;
//!
//! let checker = RagChecker::new(RagCheckerConfig::default());
//! let retrieved = vec![Document::new("The Eiffel Tower is located in Paris, France.")];
//! let result = checker
//!     .check(
//!         "The Eiffel Tower is in Paris.",
//!         "The Eiffel Tower stands in Paris.",
//!         &retrieved,
//!     )
//!     .expect("non-empty response and ground truth");
//! assert!(result.metrics.faithfulness > 0.0);
//! ```

pub mod checker;
#[cfg(test)]
mod tests;
pub mod types;

pub use checker::RagChecker;
pub use types::{RagCheckResult, RagCheckerConfig, RagCheckerError, RagCheckerMetrics};
