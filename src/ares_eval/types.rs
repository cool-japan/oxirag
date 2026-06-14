//! Types for the `ares_eval` module.

use thiserror::Error;

// ── AresConfig ────────────────────────────────────────────────────────────────

/// Configuration for [`AresEvaluator`](super::evaluator::AresEvaluator).
///
/// The only knob is [`confidence`](Self::confidence): the central confidence
/// level of every reported interval (e.g. `0.95` for a 95% interval). It is
/// mapped to a normal-distribution critical value `z` by
/// [`AresEvaluator::z_value`](super::evaluator::AresEvaluator::z_value) via a
/// small lookup table (`0.90 → 1.645`, `0.95 → 1.960`, `0.99 → 2.576`) with a
/// documented fallback of `1.960` for any other level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AresConfig {
    /// Central confidence level of the reported interval. Default `0.95`.
    pub confidence: f32,
}

impl Default for AresConfig {
    fn default() -> Self {
        Self { confidence: 0.95 }
    }
}

impl AresConfig {
    /// Create a configuration with default values (95% confidence).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the central confidence level of the reported interval.
    ///
    /// Levels of `0.90`, `0.95`, and `0.99` map to tabulated critical values;
    /// any other level falls back to the `0.95` critical value (`1.960`) when an
    /// interval is computed.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

// ── PpiInterval ───────────────────────────────────────────────────────────────

/// A point estimate of a metric rate together with its confidence interval.
///
/// Returned by both
/// [`AresEvaluator::ppi_estimate`](super::evaluator::AresEvaluator::ppi_estimate)
/// (the prediction-powered, debiased estimate) and
/// [`AresEvaluator::classical_estimate`](super::evaluator::AresEvaluator::classical_estimate)
/// (the labeled-only baseline). The interval is symmetric about the point
/// estimate, so `ci_low = point_estimate - half_width` and
/// `ci_high = point_estimate + half_width`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PpiInterval {
    /// The point estimate of the metric rate.
    pub point_estimate: f32,
    /// Lower edge of the confidence interval (`point_estimate - half_width`).
    pub ci_low: f32,
    /// Upper edge of the confidence interval (`point_estimate + half_width`).
    pub ci_high: f32,
    /// Half-width of the interval (`z * sqrt(variance)`), always non-negative.
    pub half_width: f32,
}

impl PpiInterval {
    /// Full width of the confidence interval (`ci_high - ci_low`, i.e.
    /// `2 * half_width`).
    #[must_use]
    pub fn width(&self) -> f32 {
        self.ci_high - self.ci_low
    }

    /// Whether `value` lies within the closed confidence interval
    /// `[ci_low, ci_high]`.
    #[must_use]
    pub fn contains(&self, value: f32) -> bool {
        value >= self.ci_low && value <= self.ci_high
    }
}

// ── AresError ─────────────────────────────────────────────────────────────────

/// Errors produced by the `ares_eval` module.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum AresError {
    /// The human-labeled set was empty.
    #[error("labeled set is empty")]
    EmptyLabeled,
    /// The machine-judged (unlabeled) set was empty.
    #[error("unlabeled set is empty")]
    EmptyUnlabeled,
}
