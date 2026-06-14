//! Types for the `ab_eval` module.

use thiserror::Error;

// ── AbWinner ──────────────────────────────────────────────────────────────────

/// Which of the two compared systems won the A/B comparison.
///
/// Returned inside [`AbResult::winner`] only when the paired bootstrap confidence
/// interval for the mean difference excludes zero; an interval straddling zero
/// leaves the winner undetermined (`None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbWinner {
    /// System **A** is significantly better (its per-query scores are higher).
    A,
    /// System **B** is significantly better (its per-query scores are higher).
    B,
}

// ── AbConfig ──────────────────────────────────────────────────────────────────

/// Configuration for [`AbEvaluator`](super::evaluator::AbEvaluator).
///
/// Controls the paired bootstrap procedure used to compare two systems:
///
/// * [`bootstrap_samples`](Self::bootstrap_samples) — how many deterministic
///   bootstrap resamples of the per-query differences to draw. More samples give
///   smoother confidence-interval and p-value estimates. Default `1000`.
/// * [`tie_margin`](Self::tie_margin) — the absolute per-query difference within
///   which a pair counts as a *tie* rather than a win for either side. Default
///   `0.0` (only exact ties are ties).
/// * [`confidence`](Self::confidence) — the central confidence level of the
///   reported interval (e.g. `0.95` for a 95% interval). Default `0.95`.
#[derive(Debug, Clone)]
pub struct AbConfig {
    /// Number of deterministic bootstrap resamples to draw. Default `1000`.
    pub bootstrap_samples: usize,
    /// Absolute per-query margin within which a pair counts as a tie. Default `0.0`.
    pub tie_margin: f32,
    /// Central confidence level of the reported interval. Default `0.95`.
    pub confidence: f32,
}

impl Default for AbConfig {
    fn default() -> Self {
        Self {
            bootstrap_samples: 1000,
            tie_margin: 0.0,
            confidence: 0.95,
        }
    }
}

impl AbConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of bootstrap resamples.
    #[must_use]
    pub fn with_bootstrap_samples(mut self, bootstrap_samples: usize) -> Self {
        self.bootstrap_samples = bootstrap_samples;
        self
    }

    /// Set the tie margin.
    #[must_use]
    pub fn with_tie_margin(mut self, tie_margin: f32) -> Self {
        self.tie_margin = tie_margin;
        self
    }

    /// Set the central confidence level of the reported interval.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

// ── AbResult ──────────────────────────────────────────────────────────────────

/// Outcome of a paired A/B comparison of two systems.
///
/// Produced by [`AbEvaluator::compare`](super::evaluator::AbEvaluator::compare).
/// The win/loss/tie counts and the mean statistics are exact; the interval and
/// p-value come from a deterministic paired bootstrap over the per-query
/// differences `a[i] - b[i]`.
#[derive(Debug, Clone)]
pub struct AbResult {
    /// Mean of system **A**'s per-query scores.
    pub mean_a: f32,
    /// Mean of system **B**'s per-query scores.
    pub mean_b: f32,
    /// Mean of the paired differences `a[i] - b[i]` (equals `mean_a - mean_b`).
    pub mean_diff: f32,
    /// Number of queries where **A** beat **B** by more than the tie margin.
    pub wins_a: usize,
    /// Number of queries where **B** beat **A** by more than the tie margin.
    pub wins_b: usize,
    /// Number of queries that fell within the tie margin.
    pub ties: usize,
    /// Lower edge of the bootstrap confidence interval for the mean difference.
    pub ci_low: f32,
    /// Upper edge of the bootstrap confidence interval for the mean difference.
    pub ci_high: f32,
    /// Two-sided bootstrap p-value for the mean difference, in `[0.0, 1.0]`.
    pub p_value: f32,
    /// The winning system, or `None` when the interval straddles zero.
    pub winner: Option<AbWinner>,
}

impl AbResult {
    /// Whether the comparison found a significant winner (the interval excludes
    /// zero, so [`winner`](Self::winner) is `Some`).
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.winner.is_some()
    }

    /// Total number of compared query pairs (`wins_a + wins_b + ties`).
    #[must_use]
    pub fn total(&self) -> usize {
        self.wins_a + self.wins_b + self.ties
    }
}

// ── AbError ───────────────────────────────────────────────────────────────────

/// Errors produced by the `ab_eval` module.
#[derive(Debug, Error)]
pub enum AbError {
    /// The two score lists had different lengths.
    #[error("score lists must be equal length: {a} vs {b}")]
    LengthMismatch {
        /// Length of the first (system **A**) score list.
        a: usize,
        /// Length of the second (system **B**) score list.
        b: usize,
    },
    /// At least one score list was empty.
    #[error("score lists must not be empty")]
    EmptyScores,
}
