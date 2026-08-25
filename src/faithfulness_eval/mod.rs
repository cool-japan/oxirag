//! Faithfulness evaluation (RAGAS-style): decompose an answer into atomic
//! claims and verify each claim is entailed by the retrieved context passages.
//!
//! # Algorithm
//!
//! 1. **Claim decomposition** — the answer is split at punctuation boundaries
//!    (`.`, `!`, `?`, `;` by default) into short, self-contained claim
//!    fragments.  Fragments with fewer than two vocabulary tokens are discarded.
//!
//! 2. **Entailment scoring (NLI-lite heuristic)** — for every claim, each
//!    context passage is scored by *claim coverage*: the fraction of the
//!    claim's unique lowercase alphanumeric tokens (length ≥ 2) that appear
//!    anywhere in the passage.  The maximum over all passages is the claim's
//!    entailment score; the best-matching passage is the supporting passage.
//!
//! 3. **Aggregation** — the aggregate faithfulness score is
//!    `entailed_claims / total_claims`, where a claim is considered entailed
//!    when its score meets the configured threshold (default `0.5`).
//!
//! The implementation is deterministic and pure Rust (`std` + `thiserror`):
//! no ML model, no embeddings, no randomness, no numeric dependencies.
//!
//! # Example
//!
//! ```
//! use oxirag::faithfulness_eval::{FaithfulnessConfig, FaithfulnessEvaluator};
//!
//! let config = FaithfulnessConfig::default();
//! let evaluator = FaithfulnessEvaluator::new(config);
//!
//! let answer = "The sky is blue. Water is wet.";
//! let context = vec![
//!     "The sky appears blue. Water is a wet liquid.".to_owned(),
//! ];
//!
//! let result = evaluator.evaluate(answer, &context).unwrap();
//! assert_eq!(result.total_claims, 2);
//! assert!(result.faithfulness > 0.0);
//! ```

pub mod evaluator;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use evaluator::FaithfulnessEvaluator;
pub use types::{ClaimEntailment, FaithfulnessConfig, FaithfulnessError, FaithfulnessScore};
