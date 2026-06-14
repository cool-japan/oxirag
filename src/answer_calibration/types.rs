//! Types for the `answer_calibration` module.

use thiserror::Error;

// ── ConfidenceSignals ─────────────────────────────────────────────────────────

/// Per-answer signals fed into [`AnswerCalibrator::estimate_confidence`].
///
/// [`AnswerCalibrator::estimate_confidence`]: super::calibrator::AnswerCalibrator::estimate_confidence
///
/// The three signals capture complementary evidence that a generated answer is
/// trustworthy:
///
/// * [`verbalized`](Self::verbalized) — a confidence the model *stated* in its
///   own words (e.g. a parsed "`90%`" or a hedge word). `None` when the answer
///   carried no such signal, in which case its weight is redistributed across
///   the remaining signals.
/// * [`agreement`](Self::agreement) — the fraction of independently sampled
///   answers that agree with the chosen answer (self-consistency).
/// * [`support`](Self::support) — how strongly the retrieved context supports
///   the answer (retrieval grounding).
#[derive(Debug, Clone, Default)]
pub struct ConfidenceSignals {
    /// Confidence the model verbalized, in `[0.0, 1.0]`, or `None` if absent.
    pub verbalized: Option<f32>,
    /// Fraction of sampled answers agreeing with the chosen answer, `[0.0, 1.0]`.
    pub agreement: f32,
    /// Retrieval-support score for the answer, `[0.0, 1.0]`.
    pub support: f32,
}

impl ConfidenceSignals {
    /// Create signals from agreement and support, with no verbalized confidence.
    #[must_use]
    pub fn new(agreement: f32, support: f32) -> Self {
        Self {
            verbalized: None,
            agreement,
            support,
        }
    }

    /// Set the verbalized confidence signal.
    #[must_use]
    pub fn with_verbalized(mut self, verbalized: f32) -> Self {
        self.verbalized = Some(verbalized);
        self
    }

    /// Set the self-agreement signal.
    #[must_use]
    pub fn with_agreement(mut self, agreement: f32) -> Self {
        self.agreement = agreement;
        self
    }

    /// Set the retrieval-support signal.
    #[must_use]
    pub fn with_support(mut self, support: f32) -> Self {
        self.support = support;
        self
    }
}

// ── AnswerCalibratorConfig ────────────────────────────────────────────────────

/// Configuration for [`AnswerCalibrator`](super::calibrator::AnswerCalibrator).
///
/// The three `*_weight` fields control how much each [`ConfidenceSignals`]
/// component contributes to the blended confidence. They need not sum to one —
/// [`estimate_confidence`] normalises them (and redistributes the verbalized
/// weight when that signal is missing).
///
/// [`estimate_confidence`]: super::calibrator::AnswerCalibrator::estimate_confidence
#[derive(Debug, Clone)]
pub struct AnswerCalibratorConfig {
    /// Relative weight of the verbalized-confidence signal. Default `0.34`.
    pub verbalized_weight: f32,
    /// Relative weight of the self-agreement signal. Default `0.33`.
    pub agreement_weight: f32,
    /// Relative weight of the retrieval-support signal. Default `0.33`.
    pub support_weight: f32,
    /// Number of bins used by the reliability diagram / ECE. Default `10`.
    pub num_bins: usize,
}

impl Default for AnswerCalibratorConfig {
    fn default() -> Self {
        Self {
            verbalized_weight: 0.34,
            agreement_weight: 0.33,
            support_weight: 0.33,
            num_bins: 10,
        }
    }
}

impl AnswerCalibratorConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the verbalized-confidence weight.
    #[must_use]
    pub fn with_verbalized_weight(mut self, verbalized_weight: f32) -> Self {
        self.verbalized_weight = verbalized_weight;
        self
    }

    /// Set the self-agreement weight.
    #[must_use]
    pub fn with_agreement_weight(mut self, agreement_weight: f32) -> Self {
        self.agreement_weight = agreement_weight;
        self
    }

    /// Set the retrieval-support weight.
    #[must_use]
    pub fn with_support_weight(mut self, support_weight: f32) -> Self {
        self.support_weight = support_weight;
        self
    }

    /// Set the number of reliability bins (clamped to at least one at use time).
    #[must_use]
    pub fn with_num_bins(mut self, num_bins: usize) -> Self {
        self.num_bins = num_bins;
        self
    }
}

// ── ReliabilityBin ────────────────────────────────────────────────────────────

/// One bucket of a reliability diagram.
///
/// Bins partition the confidence axis `[0.0, 1.0]` into equal-width intervals.
/// A prediction with confidence `c` lands in the bin whose half-open interval
/// `[lower, upper)` contains it (the topmost bin is closed on the right so that
/// confidence `1.0` is counted). A perfectly calibrated bin has
/// [`avg_confidence`](Self::avg_confidence) equal to [`accuracy`](Self::accuracy).
#[derive(Debug, Clone)]
pub struct ReliabilityBin {
    /// Inclusive lower edge of the bin's confidence interval.
    pub lower: f32,
    /// Exclusive upper edge of the bin's confidence interval (inclusive on the
    /// final bin).
    pub upper: f32,
    /// Number of predictions that fell into this bin.
    pub count: usize,
    /// Mean predicted confidence of the predictions in this bin (`0.0` if empty).
    pub avg_confidence: f32,
    /// Fraction of predictions in this bin that were correct (`0.0` if empty).
    pub accuracy: f32,
}

impl ReliabilityBin {
    /// Absolute calibration gap `|avg_confidence - accuracy|` for this bin.
    ///
    /// Returns `0.0` for an empty bin (it contributes nothing to ECE/MCE).
    #[must_use]
    pub fn gap(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.avg_confidence - self.accuracy).abs()
        }
    }
}

// ── CalibrationMetrics ────────────────────────────────────────────────────────

/// Aggregate calibration metrics over a `(confidence, correct)` dataset.
///
/// Produced by [`compute_metrics`](super::metrics::compute_metrics).
///
/// * [`ece`](Self::ece) — Expected Calibration Error: the count-weighted mean of
///   per-bin gaps. Zero for a perfectly calibrated dataset.
/// * [`mce`](Self::mce) — Maximum Calibration Error: the largest per-bin gap.
///   Always `>=` [`ece`](Self::ece).
/// * [`brier`](Self::brier) — Brier score: mean squared error between confidence
///   and the `0/1` outcome, in `[0.0, 1.0]`.
/// * [`bins`](Self::bins) — the reliability diagram the metrics were derived from.
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error, in `[0.0, 1.0]`.
    pub ece: f32,
    /// Maximum Calibration Error, in `[0.0, 1.0]`.
    pub mce: f32,
    /// Brier score, in `[0.0, 1.0]`.
    pub brier: f32,
    /// The reliability bins spanning `[0.0, 1.0]`.
    pub bins: Vec<ReliabilityBin>,
}

impl CalibrationMetrics {
    /// Whether the dataset is considered well calibrated (`ECE < 0.1`).
    #[must_use]
    pub fn is_well_calibrated(&self) -> bool {
        self.ece < 0.1
    }

    /// Total number of predictions summed across all bins.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.bins.iter().map(|b| b.count).sum()
    }
}

// ── AnswerCalibrationError ────────────────────────────────────────────────────

/// Errors produced by the `answer_calibration` module.
#[derive(Debug, Error)]
pub enum AnswerCalibrationError {
    /// The supplied `(confidence, correct)` dataset was empty.
    #[error("empty dataset")]
    EmptyData,
}
