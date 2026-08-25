//! Answer-level confidence calibration and calibration metrics.
//!
//! This module turns the raw signals of a generated answer into a single,
//! trustworthy confidence and lets you *measure* how well any confidence scorer
//! is calibrated over a labelled dataset.
//!
//! It has three layers:
//!
//! 1. **Confidence estimation.** [`AnswerCalibrator::estimate_confidence`] blends
//!    three [`ConfidenceSignals`] — a *verbalized* confidence (parsed from the
//!    answer text, e.g. "`90%`" or a hedge like "maybe"), *self-agreement* (the
//!    fraction of independently sampled answers that match), and *retrieval
//!    support* — into one number, normalising the weights and redistributing the
//!    verbalized weight when that signal is absent.
//! 2. **Calibration metrics.** [`compute_metrics`] reports the Expected
//!    Calibration Error ([`expected_calibration_error`]), Maximum Calibration
//!    Error, the [`brier_score`], and a reliability diagram of
//!    [`ReliabilityBin`]s over a slice of `(confidence, correct)` pairs.
//! 3. **Platt scaling.** [`AnswerCalibrator::fit_platt`] fits a logistic
//!    `sigmoid(a·raw + b)` map from raw to calibrated confidence with a few
//!    deterministic gradient steps; [`AnswerCalibrator::calibrate`] applies it.
//!
//! It is **distinct** from the `layer2_speculator::calibration` module, which
//! calibrates token-level *verification* scores (Platt / isotonic / temperature /
//! histogram) inside the speculator. Here the unit being calibrated is a whole
//! *answer*, and the emphasis is on combining answer-level signals and reporting
//! ECE / MCE / Brier reliability metrics.
//!
//! Everything is pure Rust, deterministic, and free of randomness or external ML.
//!
//! # Example
//!
//! ```
//! use oxirag::answer_calibration::{
//!     AnswerCalibrator, AnswerCalibratorConfig, ConfidenceSignals, compute_metrics,
//! };
//!
//! let calibrator = AnswerCalibrator::new(AnswerCalibratorConfig::default());
//!
//! // Parse a verbalized confidence and blend it with the other signals.
//! let verbalized = calibrator
//!     .extract_verbalized("I am 80% sure the answer is Paris")
//!     .unwrap();
//! let signals = ConfidenceSignals::new(0.75, 0.6).with_verbalized(verbalized);
//! let confidence = calibrator.estimate_confidence(&signals);
//! assert!((0.0..=1.0).contains(&confidence));
//!
//! // Measure calibration of a (confidence, correct) dataset.
//! let data = [(0.9_f32, true), (0.8, true), (0.2, false), (0.1, false)];
//! let metrics = compute_metrics(&data, 10).unwrap();
//! assert!(metrics.mce >= metrics.ece);
//! ```

pub mod calibrator;
pub mod metrics;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use calibrator::AnswerCalibrator;
pub use metrics::{
    brier_score, compute_metrics, expected_calibration_error, maximum_calibration_error,
    reliability_bins,
};
pub use types::{
    AnswerCalibrationError, AnswerCalibratorConfig, CalibrationMetrics, ConfidenceSignals,
    ReliabilityBin,
};
