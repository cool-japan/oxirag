//! ARES-style automated evaluation with prediction-powered inference (PPI).
//!
//! This module estimates an evaluation *rate* — for example a context-relevance
//! rate, an answer-faithfulness rate, or an answer-relevance rate in `[0, 1]` —
//! when ground-truth labels are scarce. It follows the ARES framework
//! (Saad-Falcon et al. 2023) and the prediction-powered inference (PPI) estimator
//! of Angelopoulos et al. (2023).
//!
//! The setup pairs a *small* human-labeled set with a *large* machine-judged
//! (unlabeled) set:
//!
//! * the labeled set provides `(prediction, true_label)` pairs, where the
//!   prediction is an automated judge's score and the label is the ground truth;
//! * the unlabeled set provides judge predictions only.
//!
//! The judge's mean over the large unlabeled set is precise but biased. PPI
//! debiases it by the judge's **rectifier** `mean(true_label - prediction)`
//! measured on the labeled set:
//!
//! ```text
//! point_estimate = mean(unlabeled prediction) + rectifier
//! variance       = var(unlabeled prediction) / n_unlabeled
//!                + var(true_label - prediction) / n_labeled
//! half_width     = z * sqrt(variance)
//! interval       = point_estimate ± half_width
//! ```
//!
//! Because most of the estimate rests on the large unlabeled set, the PPI
//! interval is typically **tighter** than the classical labeled-only interval
//! `mean(labels) ± z * sqrt(var(labels) / n_labels)`, which is also provided as a
//! baseline. The confidence level is mapped to the normal critical value `z`
//! through a small table (`0.90 → 1.645`, `0.95 → 1.960`, `0.99 → 2.576`) with a
//! `1.960` fallback for any other level.
//!
//! Everything is deterministic, pure Rust (`std` + `thiserror`), with no `rand`,
//! `ndarray`, ML, or other numeric dependency.
//!
//! # Example
//!
//! ```
//! use oxirag::ares_eval::{AresConfig, AresEvaluator};
//!
//! // A judge that is biased low by 0.1 on the labeled set (labels - predictions).
//! let labeled = [(0.4_f32, 0.5_f32), (0.6, 0.7), (0.3, 0.4), (0.8, 0.9)];
//! // Many judge predictions on unlabeled data, averaging 0.5.
//! let unlabeled = [0.4_f32, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5, 0.5];
//!
//! let evaluator = AresEvaluator::new(AresConfig::default());
//! let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();
//!
//! // The rectifier (+0.1) lifts the unlabeled mean (0.5) toward the truth.
//! assert!((ppi.point_estimate - 0.6).abs() < 1e-5);
//! assert!(ppi.ci_low <= ppi.point_estimate);
//! assert!(ppi.point_estimate <= ppi.ci_high);
//! assert!(ppi.half_width >= 0.0);
//! ```

pub mod evaluator;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use evaluator::AresEvaluator;
pub use types::{AresConfig, AresError, PpiInterval};
