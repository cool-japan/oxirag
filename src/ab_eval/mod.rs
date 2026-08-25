//! A/B evaluation harness for comparing two RAG systems.
//!
//! This module answers a deployment question that single-system metrics cannot:
//! *given two RAG pipelines scored on the **same** queries, is one significantly
//! better than the other?* It works directly on **paired** per-query scores
//! (any metric — answer quality, nDCG, an LLM-judge rating, …) so the comparison
//! controls for query difficulty.
//!
//! [`AbEvaluator::compare`] reports:
//!
//! * the per-system means and the mean paired difference `a[i] - b[i]`;
//! * win / loss / tie counts, where a pair is a *tie* when `|a[i] - b[i]|` is
//!   within [`AbConfig::tie_margin`];
//! * a **paired bootstrap** confidence interval and a two-sided p-value for the
//!   mean difference, plus the resulting [`AbWinner`] (or `None` when the
//!   interval straddles zero).
//!
//! The bootstrap is fully **deterministic** — it uses no `rand`. Each resample
//! index is an FNV-1a hash of the `(sample, position)` pair reduced modulo the
//! sample size, so repeating a comparison on the same inputs yields a
//! byte-identical [`AbResult`]. Everything is pure Rust, `std` + `thiserror`,
//! with no ML or numeric dependencies.
//!
//! # Example
//!
//! ```
//! use oxirag::ab_eval::{AbConfig, AbEvaluator, AbWinner};
//!
//! // System A scores higher than B on every query.
//! let scores_a = [0.9_f32, 0.8, 0.7, 0.95, 0.85];
//! let scores_b = [0.6_f32, 0.5, 0.55, 0.65, 0.5];
//!
//! let evaluator = AbEvaluator::new(AbConfig::default());
//! let result = evaluator.compare(&scores_a, &scores_b).unwrap();
//!
//! assert!(result.mean_diff > 0.0);
//! assert_eq!(result.wins_a, 5);
//! assert_eq!(result.winner, Some(AbWinner::A));
//! assert!(result.ci_low > 0.0);
//! assert!((0.0..=1.0).contains(&result.p_value));
//! ```

pub mod evaluator;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use evaluator::AbEvaluator;
pub use types::{AbConfig, AbError, AbResult, AbWinner};
