//! Semantic-entropy uncertainty estimation (Kuhn et al. 2023).
//!
//! Detects hallucination by *sampling* several candidate answers, clustering
//! them by **semantic equivalence** (a bidirectional-entailment stand-in built
//! from token-set Jaccard plus mutual containment), and measuring the
//! [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))
//! of the resulting cluster distribution.
//!
//! Low entropy means the samples collapse into a handful of meanings — the model
//! is confident and consistent.  High entropy means probability mass is spread
//! across many distinct meanings — the model is uncertain and the answer is more
//! likely to be a hallucination.
//!
//! This is **distinct** from:
//! * the `consistency_checker` module, which types *pairwise* conflicts
//!   (numerical / temporal / negation) within a single passage; and
//! * the `hallucination_detector` module, which scores how well each *claim* is
//!   supported by source documents.
//!
//! Here we never compare to a source and never type conflicts — we cluster a set
//! of independently sampled answers and report the entropy over their meanings.
//!
//! # Example
//!
//! ```
//! use oxirag::semantic_entropy::{SemanticEntropyConfig, SemanticEntropyEstimator};
//!
//! let estimator = SemanticEntropyEstimator::new(SemanticEntropyConfig::default());
//! let answers = vec![
//!     "Paris is the capital of France".to_string(),
//!     "The capital of France is Paris".to_string(),
//!     "Berlin is the capital of Germany".to_string(),
//! ];
//! let result = estimator.estimate(&answers).expect("non-empty samples");
//! // The two Paris paraphrases merge, so we get two meaning clusters.
//! assert_eq!(result.num_clusters, 2);
//! assert!(result.entropy > 0.0);
//! ```

pub mod estimator;
#[cfg(test)]
mod tests;
pub mod types;

pub use estimator::SemanticEntropyEstimator;
pub use types::{
    MeaningCluster, SemanticEntropyConfig, SemanticEntropyError, SemanticEntropyResult,
};
