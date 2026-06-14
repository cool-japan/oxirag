//! Calibration metrics over `(confidence, correct)` datasets.
//!
//! All functions operate on a slice of `(f32, bool)` pairs where the `f32` is a
//! predicted confidence (clamped to `[0.0, 1.0]`) and the `bool` records whether
//! that prediction was correct. They are deterministic and allocation-light.

use crate::answer_calibration::types::{
    AnswerCalibrationError, CalibrationMetrics, ReliabilityBin,
};

/// Index of the bin that confidence `conf` falls into for `num_bins` equal-width
/// bins spanning `[0.0, 1.0]`. The final bin is closed on the right so that a
/// confidence of exactly `1.0` is counted.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn bin_index(conf: f32, num_bins: usize) -> usize {
    let c = conf.clamp(0.0, 1.0);
    let idx = (c * num_bins as f32).floor() as usize;
    idx.min(num_bins - 1)
}

/// Build the reliability bins (the reliability diagram) for a dataset.
///
/// The confidence axis `[0.0, 1.0]` is split into `num_bins` equal-width
/// intervals. Each prediction is assigned to the bin holding its confidence; the
/// bin records its population count, mean confidence, and accuracy.
///
/// `num_bins` is clamped to at least one. The returned bins always partition
/// `[0.0, 1.0]` and their counts sum to the dataset length.
///
/// # Errors
///
/// Returns [`AnswerCalibrationError::EmptyData`] if `conf_correct` is empty.
#[allow(clippy::cast_precision_loss)]
pub fn reliability_bins(
    conf_correct: &[(f32, bool)],
    num_bins: usize,
) -> Result<Vec<ReliabilityBin>, AnswerCalibrationError> {
    if conf_correct.is_empty() {
        return Err(AnswerCalibrationError::EmptyData);
    }
    let num_bins = num_bins.max(1);
    let bin_width = 1.0 / num_bins as f32;

    // Accumulators per bin: (sum_confidence, num_correct, count).
    let mut acc: Vec<(f32, usize, usize)> = vec![(0.0, 0, 0); num_bins];
    for &(conf, correct) in conf_correct {
        let c = conf.clamp(0.0, 1.0);
        let idx = bin_index(c, num_bins);
        acc[idx].0 += c;
        if correct {
            acc[idx].1 += 1;
        }
        acc[idx].2 += 1;
    }

    let bins = acc
        .into_iter()
        .enumerate()
        .map(|(i, (conf_sum, correct_count, count))| {
            let lower = i as f32 * bin_width;
            // Pin the final upper edge to exactly 1.0 to avoid float drift.
            let upper = if i + 1 == num_bins {
                1.0
            } else {
                (i + 1) as f32 * bin_width
            };
            let (avg_confidence, accuracy) = if count == 0 {
                (0.0, 0.0)
            } else {
                (conf_sum / count as f32, correct_count as f32 / count as f32)
            };
            ReliabilityBin {
                lower,
                upper,
                count,
                avg_confidence,
                accuracy,
            }
        })
        .collect();
    Ok(bins)
}

/// Expected Calibration Error: the count-weighted mean of the per-bin calibration
/// gaps `|avg_confidence - accuracy|`.
///
/// Zero for a perfectly calibrated dataset (every populated bin has confidence
/// equal to accuracy); approaches `1.0` for maximally overconfident data.
///
/// # Errors
///
/// Returns [`AnswerCalibrationError::EmptyData`] if `conf_correct` is empty.
#[allow(clippy::cast_precision_loss)]
pub fn expected_calibration_error(
    conf_correct: &[(f32, bool)],
    num_bins: usize,
) -> Result<f32, AnswerCalibrationError> {
    let bins = reliability_bins(conf_correct, num_bins)?;
    let total = conf_correct.len() as f32;
    let ece = bins.iter().map(|b| b.gap() * b.count as f32).sum::<f32>() / total;
    Ok(ece)
}

/// Maximum Calibration Error: the largest per-bin calibration gap across all
/// populated bins.
///
/// Always `>=` the [`expected_calibration_error`] on the same data.
///
/// # Errors
///
/// Returns [`AnswerCalibrationError::EmptyData`] if `conf_correct` is empty.
pub fn maximum_calibration_error(
    conf_correct: &[(f32, bool)],
    num_bins: usize,
) -> Result<f32, AnswerCalibrationError> {
    let bins = reliability_bins(conf_correct, num_bins)?;
    let mce = bins
        .iter()
        .filter(|b| b.count > 0)
        .map(ReliabilityBin::gap)
        .fold(0.0_f32, f32::max);
    Ok(mce)
}

/// Brier score: the mean squared error between predicted confidence and the
/// `0/1` correctness outcome.
///
/// Lies in `[0.0, 1.0]`. A score of `0.0` means every prediction was perfectly
/// confident and right (confidence `1.0` on correct answers, `0.0` on incorrect);
/// `1.0` means maximally confident and always wrong.
///
/// # Errors
///
/// Returns [`AnswerCalibrationError::EmptyData`] if `conf_correct` is empty.
#[allow(clippy::cast_precision_loss)]
pub fn brier_score(conf_correct: &[(f32, bool)]) -> Result<f32, AnswerCalibrationError> {
    if conf_correct.is_empty() {
        return Err(AnswerCalibrationError::EmptyData);
    }
    let sum: f32 = conf_correct
        .iter()
        .map(|&(conf, correct)| {
            let c = conf.clamp(0.0, 1.0);
            let target = if correct { 1.0 } else { 0.0 };
            (c - target).powi(2)
        })
        .sum();
    Ok(sum / conf_correct.len() as f32)
}

/// Compute the full set of calibration metrics (ECE, MCE, Brier, and the
/// reliability diagram) in a single pass over the dataset.
///
/// `num_bins` is clamped to at least one.
///
/// # Errors
///
/// Returns [`AnswerCalibrationError::EmptyData`] if `conf_correct` is empty.
#[allow(clippy::cast_precision_loss)]
pub fn compute_metrics(
    conf_correct: &[(f32, bool)],
    num_bins: usize,
) -> Result<CalibrationMetrics, AnswerCalibrationError> {
    let bins = reliability_bins(conf_correct, num_bins)?;
    let total = conf_correct.len() as f32;

    let ece = bins.iter().map(|b| b.gap() * b.count as f32).sum::<f32>() / total;
    let mce = bins
        .iter()
        .filter(|b| b.count > 0)
        .map(ReliabilityBin::gap)
        .fold(0.0_f32, f32::max);
    let brier = brier_score(conf_correct)?;

    Ok(CalibrationMetrics {
        ece,
        mce,
        brier,
        bins,
    })
}
