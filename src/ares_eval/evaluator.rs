//! The ARES-style prediction-powered evaluator.

use crate::ares_eval::types::{AresConfig, AresError, PpiInterval};

/// Normal critical value `z` for a 90% two-sided interval.
const Z_90: f32 = 1.645;
/// Normal critical value `z` for a 95% two-sided interval.
const Z_95: f32 = 1.960;
/// Normal critical value `z` for a 99% two-sided interval.
const Z_99: f32 = 2.576;

/// Absolute tolerance for matching a configured confidence level against the
/// tabulated levels (`0.90`, `0.95`, `0.99`).
const LEVEL_TOLERANCE: f32 = 1e-4;

// ── AresEvaluator ─────────────────────────────────────────────────────────────

/// Evaluator that estimates a metric rate with **prediction-powered inference**
/// (PPI).
///
/// Following ARES (Saad-Falcon et al. 2023) and the PPI estimator of Angelopoulos
/// et al. (2023), the metric of interest is a *rate* in `[0, 1]` — for instance a
/// context-relevance rate, answer-faithfulness rate, or answer-relevance rate.
/// Two inputs drive the estimate:
///
/// * a **small** human-labeled set of `(prediction, true_label)` pairs, where the
///   prediction is an automated judge's score and the label is the ground truth
///   (`0.0`/`1.0`, or any value in `[0, 1]`); and
/// * a **large** machine-judged set of predictions only (the unlabeled set).
///
/// The judge's mean on the large unlabeled set is precise but biased. PPI
/// *debiases* it by the judge's measured bias on the labeled set — the
/// **rectifier** `mean(true_label - prediction)` — yielding
/// `point_estimate = mean(unlabeled prediction) + rectifier`. Because most of the
/// estimate's mass rests on the large unlabeled set, the resulting confidence
/// interval is typically **tighter** than the classical labeled-only interval
/// whenever the judge is informative.
///
/// The whole computation is deterministic, pure Rust (`std` + `thiserror`), and
/// free of any randomness, ML, or numeric dependency.
#[derive(Debug, Clone, Copy, Default)]
pub struct AresEvaluator {
    /// Configuration controlling the confidence level of reported intervals.
    pub config: AresConfig,
}

impl AresEvaluator {
    /// Create a new evaluator with the given configuration.
    #[must_use]
    pub fn new(config: AresConfig) -> Self {
        Self { config }
    }

    /// The normal-distribution critical value `z` for the configured confidence
    /// level.
    ///
    /// Looks up [`AresConfig::confidence`] in a small table — `0.90 → 1.645`,
    /// `0.95 → 1.960`, `0.99 → 2.576` — matching within a tiny tolerance. Any
    /// other level falls back to the `0.95` critical value (`1.960`).
    #[must_use]
    pub fn z_value(&self) -> f32 {
        let c = self.config.confidence;
        if (c - 0.90).abs() <= LEVEL_TOLERANCE {
            Z_90
        } else if (c - 0.99).abs() <= LEVEL_TOLERANCE {
            Z_99
        } else if (c - 0.95).abs() <= LEVEL_TOLERANCE {
            Z_95
        } else {
            // Documented fallback for unsupported levels: the 95% critical value.
            Z_95
        }
    }

    /// The prediction-powered (PPI) estimate of the metric rate.
    ///
    /// `labeled` holds `(prediction, true_label)` pairs from the small
    /// human-labeled set; `unlabeled_predictions` holds the judge's predictions
    /// on the large unlabeled set. The estimate is computed as:
    ///
    /// * `rectifier = mean(true_label - prediction)` over the labeled set (the
    ///   judge's measured bias);
    /// * `point_estimate = mean(unlabeled prediction) + rectifier` (the debiased
    ///   rate);
    /// * `variance` is the sum of the unlabeled prediction variance divided by
    ///   `n_unlabeled` and the residual variance divided by `n_labeled`, using the
    ///   population variance of each set;
    /// * `half_width = z * sqrt(variance)`, with `z` from [`z_value`](Self::z_value);
    /// * the interval is `point_estimate ± half_width`.
    ///
    /// The point estimate is **not** clamped to `[0, 1]`, so a heavily biased
    /// judge near the boundary can yield an estimate slightly outside the unit
    /// interval; callers may clamp if a strict rate is required.
    ///
    /// # Errors
    ///
    /// Returns [`AresError::EmptyLabeled`] if `labeled` is empty, or
    /// [`AresError::EmptyUnlabeled`] if `unlabeled_predictions` is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn ppi_estimate(
        &self,
        labeled: &[(f32, f32)],
        unlabeled_predictions: &[f32],
    ) -> Result<PpiInterval, AresError> {
        if labeled.is_empty() {
            return Err(AresError::EmptyLabeled);
        }
        if unlabeled_predictions.is_empty() {
            return Err(AresError::EmptyUnlabeled);
        }

        let n_labeled = labeled.len() as f32;
        let n_unlabeled = unlabeled_predictions.len() as f32;

        // Per-pair residuals (label - prediction) over the labeled set drive both
        // the rectifier (their mean) and the labeled-side variance term.
        let residuals: Vec<f32> = labeled.iter().map(|&(pred, label)| label - pred).collect();
        let rectifier = mean(&residuals);
        let residual_var = population_variance(&residuals, rectifier);

        // Mean and variance of the judge's predictions on the large unlabeled set.
        let unlabeled_mean = mean(unlabeled_predictions);
        let unlabeled_var = population_variance(unlabeled_predictions, unlabeled_mean);

        let point_estimate = unlabeled_mean + rectifier;
        let variance = unlabeled_var / n_unlabeled + residual_var / n_labeled;

        Ok(self.interval(point_estimate, variance))
    }

    /// The classical labeled-only estimate of the metric rate.
    ///
    /// This is the baseline that ignores the unlabeled set entirely: the point
    /// estimate is `mean(labels)`, and the interval is
    /// `mean(labels) ± z * sqrt(var(labels) / n_labels)`, where `var` is the
    /// population variance and `z` comes from [`z_value`](Self::z_value).
    ///
    /// Comparing this interval's [`half_width`](PpiInterval::half_width) with the
    /// one from [`ppi_estimate`](Self::ppi_estimate) quantifies how much the large
    /// unlabeled set tightened the estimate.
    ///
    /// # Errors
    ///
    /// Returns [`AresError::EmptyLabeled`] if `labels` is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn classical_estimate(&self, labels: &[f32]) -> Result<PpiInterval, AresError> {
        if labels.is_empty() {
            return Err(AresError::EmptyLabeled);
        }
        let n = labels.len() as f32;
        let point_estimate = mean(labels);
        let label_var = population_variance(labels, point_estimate);
        let variance = label_var / n;
        Ok(self.interval(point_estimate, variance))
    }

    /// Assemble a symmetric [`PpiInterval`] from a point estimate and a variance.
    ///
    /// The half-width is `z * sqrt(variance)`; a non-finite or negative variance
    /// is treated as zero so the half-width is always finite and non-negative.
    fn interval(self, point_estimate: f32, variance: f32) -> PpiInterval {
        let safe_var = if variance.is_finite() && variance > 0.0 {
            variance
        } else {
            0.0
        };
        let half_width = self.z_value() * safe_var.sqrt();
        PpiInterval {
            point_estimate,
            ci_low: point_estimate - half_width,
            ci_high: point_estimate + half_width,
            half_width,
        }
    }
}

/// Arithmetic mean of a non-empty slice.
///
/// Callers guarantee `values` is non-empty.
#[allow(clippy::cast_precision_loss)]
fn mean(values: &[f32]) -> f32 {
    let sum: f32 = values.iter().copied().sum();
    sum / values.len() as f32
}

/// Population variance of a non-empty slice given its precomputed `mean`.
///
/// Uses the population (divide-by-`n`) estimator, which matches the PPI variance
/// formulation. Callers guarantee `values` is non-empty.
#[allow(clippy::cast_precision_loss)]
fn population_variance(values: &[f32], mean: f32) -> f32 {
    let sum_sq: f32 = values.iter().map(|&v| (v - mean).powi(2)).sum();
    sum_sq / values.len() as f32
}
