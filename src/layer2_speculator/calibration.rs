//! Confidence calibration for the Speculator layer.
//!
//! This module provides methods for calibrating confidence scores to be
//! more accurate and well-calibrated, meaning that a predicted confidence
//! of 80% should correspond to approximately 80% accuracy.

use crate::error::OxiRagError;

/// Method for calibrating confidence scores.
#[derive(Debug, Clone)]
pub enum CalibrationMethod {
    /// Platt scaling using logistic regression.
    Platt,
    /// Isotonic regression (monotonic calibration).
    Isotonic,
    /// Temperature scaling with a fixed temperature parameter.
    TemperatureScaling(f32),
    /// Histogram binning with a specified number of bins.
    HistogramBinning {
        /// Number of bins for the histogram.
        bins: usize,
    },
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        Self::TemperatureScaling(1.0)
    }
}

/// Statistics about calibration quality.
#[derive(Debug, Clone, Default)]
pub struct CalibrationStats {
    /// Expected Calibration Error - average absolute difference between
    /// predicted confidence and actual accuracy.
    pub expected_calibration_error: f32,
    /// Maximum Calibration Error - worst-case calibration error.
    pub maximum_calibration_error: f32,
    /// Brier score - mean squared difference between predictions and outcomes.
    pub brier_score: f32,
    /// Number of samples used for calibration.
    pub num_samples: usize,
}

impl CalibrationStats {
    /// Create new calibration statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if calibration is considered good (ECE < 0.1).
    #[must_use]
    pub fn is_well_calibrated(&self) -> bool {
        self.expected_calibration_error < 0.1
    }
}

/// Confidence calibrator that adjusts raw confidence scores.
pub struct ConfidenceCalibrator {
    method: CalibrationMethod,
    /// Calibration data as tuples of (predicted confidence, was correct).
    calibration_data: Vec<(f32, bool)>,
    is_fitted: bool,
    // Fitted parameters
    platt_a: f32,
    platt_b: f32,
    isotonic_points: Vec<(f32, f32)>,
    histogram_bins: Vec<(f32, f32)>, // (bin_center, calibrated_value)
}

impl ConfidenceCalibrator {
    /// Create a new confidence calibrator with the specified method.
    #[must_use]
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            method,
            calibration_data: Vec::new(),
            is_fitted: false,
            platt_a: 1.0,
            platt_b: 0.0,
            isotonic_points: Vec::new(),
            histogram_bins: Vec::new(),
        }
    }

    /// Create a calibrator with Platt scaling.
    #[must_use]
    pub fn platt() -> Self {
        Self::new(CalibrationMethod::Platt)
    }

    /// Create a calibrator with isotonic regression.
    #[must_use]
    pub fn isotonic() -> Self {
        Self::new(CalibrationMethod::Isotonic)
    }

    /// Create a calibrator with temperature scaling.
    #[must_use]
    pub fn temperature_scaling(temperature: f32) -> Self {
        Self::new(CalibrationMethod::TemperatureScaling(temperature))
    }

    /// Create a calibrator with histogram binning.
    #[must_use]
    pub fn histogram_binning(bins: usize) -> Self {
        Self::new(CalibrationMethod::HistogramBinning { bins })
    }

    /// Add a calibration sample.
    ///
    /// # Arguments
    /// * `predicted` - The predicted confidence score (0.0 to 1.0)
    /// * `actual_correct` - Whether the prediction was actually correct
    pub fn add_sample(&mut self, predicted: f32, actual_correct: bool) {
        self.calibration_data
            .push((predicted.clamp(0.0, 1.0), actual_correct));
        self.is_fitted = false;
    }

    /// Add multiple calibration samples at once.
    pub fn add_samples(&mut self, samples: &[(f32, bool)]) {
        for &(predicted, actual_correct) in samples {
            self.add_sample(predicted, actual_correct);
        }
    }

    /// Clear all calibration data.
    pub fn clear(&mut self) {
        self.calibration_data.clear();
        self.is_fitted = false;
    }

    /// Get the number of calibration samples.
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.calibration_data.len()
    }

    /// Check if the calibrator has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Fit the calibration model to the collected data.
    ///
    /// # Errors
    ///
    /// Returns an error if there is insufficient data for calibration.
    pub fn fit(&mut self) -> Result<(), OxiRagError> {
        if self.calibration_data.len() < 2 {
            return Err(OxiRagError::Config(
                "Insufficient calibration data (need at least 2 samples)".to_string(),
            ));
        }

        match &self.method {
            CalibrationMethod::Platt => {
                self.fit_platt();
                Ok(())
            }
            CalibrationMethod::Isotonic => {
                self.fit_isotonic();
                Ok(())
            }
            CalibrationMethod::TemperatureScaling(_) => {
                // Temperature scaling doesn't need fitting in this simple version
                self.is_fitted = true;
                Ok(())
            }
            CalibrationMethod::HistogramBinning { bins } => {
                self.fit_histogram(*bins);
                Ok(())
            }
        }
    }

    /// Fit Platt scaling parameters using simplified logistic regression.
    fn fit_platt(&mut self) {
        // Simple gradient descent for logistic regression
        // We fit: calibrated = sigmoid(a * raw + b)

        let mut a = 1.0_f32;
        let mut b = 0.0_f32;
        let learning_rate = 0.1;
        let iterations = 100;

        for _ in 0..iterations {
            let mut grad_a = 0.0_f32;
            let mut grad_b = 0.0_f32;

            for &(predicted, actual) in &self.calibration_data {
                let logit = a * predicted + b;
                let sigmoid_val = sigmoid(logit);
                let target = if actual { 1.0 } else { 0.0 };
                let error = sigmoid_val - target;

                grad_a += error * predicted;
                grad_b += error;
            }

            #[allow(clippy::cast_precision_loss)]
            let n = self.calibration_data.len() as f32;
            a -= learning_rate * grad_a / n;
            b -= learning_rate * grad_b / n;
        }

        self.platt_a = a;
        self.platt_b = b;
        self.is_fitted = true;
    }

    /// Fit isotonic regression using pool adjacent violators algorithm.
    fn fit_isotonic(&mut self) {
        // Sort by predicted confidence
        let mut sorted_data = self.calibration_data.clone();
        sorted_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Pool adjacent violators algorithm (PAVA)
        let mut y: Vec<f32> = sorted_data
            .iter()
            .map(|&(_, correct)| if correct { 1.0 } else { 0.0 })
            .collect();
        let mut weights: Vec<f32> = vec![1.0; sorted_data.len()];

        let mut i = 0;
        while i + 1 < y.len() {
            if y[i] > y[i + 1] {
                // Pool the two blocks
                let total_weight = weights[i] + weights[i + 1];
                let new_y = (y[i] * weights[i] + y[i + 1] * weights[i + 1]) / total_weight;
                y[i] = new_y;
                weights[i] = total_weight;

                // Remove the second element
                y.remove(i + 1);
                weights.remove(i + 1);
                sorted_data.remove(i + 1);

                // Go back to check previous blocks
                i = i.saturating_sub(1);
            } else {
                i += 1;
            }
        }

        // Store the isotonic points
        self.isotonic_points = sorted_data
            .iter()
            .zip(y.iter())
            .map(|(&(x, _), &y_val)| (x, y_val))
            .collect();

        self.is_fitted = true;
    }

    /// Fit histogram binning.
    #[allow(clippy::cast_precision_loss)]
    fn fit_histogram(&mut self, bins: usize) {
        let bins = bins.max(1);
        let bin_width = 1.0 / bins as f32;

        self.histogram_bins = (0..bins)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let bin_start = i as f32 * bin_width;
                let bin_end = bin_start + bin_width;
                let bin_center = bin_start + bin_width / 2.0;

                // Find samples in this bin
                let samples_in_bin: Vec<bool> = self
                    .calibration_data
                    .iter()
                    .filter(|&&(pred, _)| pred >= bin_start && pred < bin_end)
                    .map(|&(_, correct)| correct)
                    .collect();

                // Calculate accuracy in this bin
                let calibrated_value = if samples_in_bin.is_empty() {
                    bin_center // Use bin center if no samples
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let accuracy = samples_in_bin.iter().filter(|&&c| c).count() as f32
                        / samples_in_bin.len() as f32;
                    accuracy
                };

                (bin_center, calibrated_value)
            })
            .collect();

        self.is_fitted = true;
    }

    /// Calibrate a raw confidence score.
    ///
    /// If the calibrator has not been fitted, returns the raw score.
    #[must_use]
    pub fn calibrate(&self, raw_confidence: f32) -> f32 {
        let raw = raw_confidence.clamp(0.0, 1.0);

        if !self.is_fitted {
            return raw;
        }

        match &self.method {
            CalibrationMethod::Platt => sigmoid(self.platt_a * raw + self.platt_b),
            CalibrationMethod::Isotonic => self.interpolate_isotonic(raw),
            CalibrationMethod::TemperatureScaling(temp) => {
                let temp = temp.max(0.01); // Prevent division by zero
                // Apply temperature scaling in logit space
                let logit = logit(raw);
                sigmoid(logit / temp)
            }
            CalibrationMethod::HistogramBinning { bins: _ } => self.interpolate_histogram(raw),
        }
    }

    /// Interpolate isotonic calibration.
    fn interpolate_isotonic(&self, raw: f32) -> f32 {
        if self.isotonic_points.is_empty() {
            return raw;
        }

        // Find surrounding points and interpolate
        let mut lower = (0.0_f32, 0.0_f32);
        let mut upper = (1.0_f32, 1.0_f32);

        for &(x, y) in &self.isotonic_points {
            if x <= raw {
                lower = (x, y);
            }
            if x >= raw {
                upper = (x, y);
                break;
            }
        }

        // Linear interpolation
        if (upper.0 - lower.0).abs() < 1e-6 {
            lower.1
        } else {
            let t = (raw - lower.0) / (upper.0 - lower.0);
            lower.1 + t * (upper.1 - lower.1)
        }
    }

    /// Interpolate histogram calibration.
    fn interpolate_histogram(&self, raw: f32) -> f32 {
        if self.histogram_bins.is_empty() {
            return raw;
        }

        // Find the closest bin center
        let mut best_bin = (0.5, 0.5);
        let mut best_dist = f32::MAX;

        for &(center, value) in &self.histogram_bins {
            let dist = (raw - center).abs();
            if dist < best_dist {
                best_dist = dist;
                best_bin = (center, value);
            }
        }

        best_bin.1
    }

    /// Get calibration statistics.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn get_statistics(&self) -> CalibrationStats {
        if self.calibration_data.is_empty() {
            return CalibrationStats::default();
        }

        let num_bins = 10;
        let mut bin_sums: Vec<(f32, f32, usize)> = vec![(0.0, 0.0, 0); num_bins];

        for &(pred, actual) in &self.calibration_data {
            let calibrated = self.calibrate(pred);
            let bin_idx = ((calibrated * (num_bins as f32)).floor() as usize).min(num_bins - 1);
            bin_sums[bin_idx].0 += calibrated;
            bin_sums[bin_idx].1 += if actual { 1.0 } else { 0.0 };
            bin_sums[bin_idx].2 += 1;
        }

        let mut ece = 0.0_f32;
        let mut mce = 0.0_f32;
        let mut total_count = 0_usize;

        for (conf_sum, acc_sum, count) in &bin_sums {
            if *count > 0 {
                let avg_conf = conf_sum / *count as f32;
                let avg_acc = acc_sum / *count as f32;
                let gap = (avg_conf - avg_acc).abs();

                let weighted_gap = gap * *count as f32;
                ece += weighted_gap;
                mce = mce.max(gap);
                total_count += count;
            }
        }

        let ece = if total_count > 0 {
            ece / total_count as f32
        } else {
            0.0
        };

        // Calculate Brier score
        let brier: f32 = self
            .calibration_data
            .iter()
            .map(|&(pred, actual)| {
                let calibrated = self.calibrate(pred);
                let target = if actual { 1.0 } else { 0.0 };
                (calibrated - target).powi(2)
            })
            .sum::<f32>()
            / self.calibration_data.len().max(1) as f32;

        CalibrationStats {
            expected_calibration_error: ece,
            maximum_calibration_error: mce,
            brier_score: brier,
            num_samples: self.calibration_data.len(),
        }
    }

    /// Get the calibration method.
    #[must_use]
    pub fn method(&self) -> &CalibrationMethod {
        &self.method
    }
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self::new(CalibrationMethod::default())
    }
}

/// Sigmoid function for logistic transformation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Logit function (inverse of sigmoid) with clamping for numerical stability.
#[inline]
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::manual_range_contains,
    clippy::similar_names
)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_method_default() {
        let method = CalibrationMethod::default();
        assert!(matches!(method, CalibrationMethod::TemperatureScaling(_)));
    }

    #[test]
    fn test_calibrator_creation() {
        let calibrator = ConfidenceCalibrator::platt();
        assert!(matches!(calibrator.method(), CalibrationMethod::Platt));
        assert!(!calibrator.is_fitted());
        assert_eq!(calibrator.num_samples(), 0);
    }

    #[test]
    fn test_add_sample() {
        let mut calibrator = ConfidenceCalibrator::new(CalibrationMethod::Platt);
        calibrator.add_sample(0.8, true);
        calibrator.add_sample(0.3, false);

        assert_eq!(calibrator.num_samples(), 2);
        assert!(!calibrator.is_fitted());
    }

    #[test]
    fn test_add_samples() {
        let mut calibrator = ConfidenceCalibrator::new(CalibrationMethod::Platt);
        calibrator.add_samples(&[(0.8, true), (0.3, false), (0.6, true)]);

        assert_eq!(calibrator.num_samples(), 3);
    }

    #[test]
    fn test_clear() {
        let mut calibrator = ConfidenceCalibrator::new(CalibrationMethod::Platt);
        calibrator.add_sample(0.8, true);
        calibrator.clear();

        assert_eq!(calibrator.num_samples(), 0);
    }

    #[test]
    fn test_fit_insufficient_data() {
        let mut calibrator = ConfidenceCalibrator::new(CalibrationMethod::Platt);
        calibrator.add_sample(0.8, true);

        let result = calibrator.fit();
        assert!(result.is_err());
    }

    #[test]
    fn test_fit_platt() {
        let mut calibrator = ConfidenceCalibrator::platt();

        // Add samples that represent a well-calibrated model
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let conf = i as f32 / 100.0;
            let correct = conf > 0.5;
            calibrator.add_sample(conf, correct);
        }

        let result = calibrator.fit();
        assert!(result.is_ok());
        assert!(calibrator.is_fitted());
    }

    #[test]
    fn test_fit_isotonic() {
        let mut calibrator = ConfidenceCalibrator::isotonic();

        calibrator.add_samples(&[
            (0.1, false),
            (0.2, false),
            (0.3, true),
            (0.4, false),
            (0.5, true),
            (0.6, true),
            (0.7, true),
            (0.8, true),
            (0.9, true),
        ]);

        let result = calibrator.fit();
        assert!(result.is_ok());
        assert!(calibrator.is_fitted());
    }

    #[test]
    fn test_fit_histogram() {
        let mut calibrator = ConfidenceCalibrator::histogram_binning(10);

        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let conf = i as f32 / 100.0;
            let correct = conf > 0.5;
            calibrator.add_sample(conf, correct);
        }

        let result = calibrator.fit();
        assert!(result.is_ok());
        assert!(calibrator.is_fitted());
    }

    #[test]
    fn test_fit_temperature_scaling() {
        let mut calibrator = ConfidenceCalibrator::temperature_scaling(1.5);
        calibrator.add_sample(0.8, true);
        calibrator.add_sample(0.3, false);

        let result = calibrator.fit();
        assert!(result.is_ok());
        assert!(calibrator.is_fitted());
    }

    #[test]
    fn test_calibrate_unfitted() {
        let calibrator = ConfidenceCalibrator::platt();

        // Should return raw value when not fitted
        assert_eq!(calibrator.calibrate(0.5), 0.5);
        assert_eq!(calibrator.calibrate(0.8), 0.8);
    }

    #[test]
    fn test_calibrate_platt() {
        let mut calibrator = ConfidenceCalibrator::platt();

        // Add overconfident samples
        calibrator.add_samples(&[
            (0.9, false),
            (0.9, false),
            (0.9, true),
            (0.8, false),
            (0.8, true),
            (0.7, true),
            (0.7, true),
            (0.6, true),
        ]);

        calibrator.fit().unwrap();

        let calibrated = calibrator.calibrate(0.9);
        // Should be adjusted down from overconfident predictions
        assert!(calibrated >= 0.0 && calibrated <= 1.0);
    }

    #[test]
    fn test_calibrate_temperature() {
        let mut calibrator = ConfidenceCalibrator::temperature_scaling(2.0);
        calibrator.add_sample(0.5, true);
        calibrator.add_sample(0.5, true);
        calibrator.fit().unwrap();

        // Higher temperature should reduce confidence extremity
        let calibrated_high = calibrator.calibrate(0.9);
        let calibrated_low = calibrator.calibrate(0.1);

        // With temperature > 1, extremes should move toward 0.5
        assert!(calibrated_high < 0.9);
        assert!(calibrated_low > 0.1);
    }

    #[test]
    fn test_calibrate_clamping() {
        let calibrator = ConfidenceCalibrator::platt();

        // Values outside [0, 1] should be clamped
        assert_eq!(calibrator.calibrate(-0.5), 0.0);
        assert_eq!(calibrator.calibrate(1.5), 1.0);
    }

    #[test]
    fn test_get_statistics_empty() {
        let calibrator = ConfidenceCalibrator::platt();
        let stats = calibrator.get_statistics();

        assert_eq!(stats.num_samples, 0);
        assert_eq!(stats.expected_calibration_error, 0.0);
    }

    #[test]
    fn test_get_statistics() {
        let mut calibrator = ConfidenceCalibrator::platt();

        // Perfect calibration
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let conf = i as f32 / 100.0;
            // Perfectly calibrated: high confidence = correct
            let correct = conf > 0.5;
            calibrator.add_sample(conf, correct);
        }

        calibrator.fit().unwrap();
        let stats = calibrator.get_statistics();

        assert_eq!(stats.num_samples, 100);
        assert!(stats.brier_score >= 0.0);
        assert!(stats.expected_calibration_error >= 0.0);
    }

    #[test]
    fn test_calibration_stats_well_calibrated() {
        let stats = CalibrationStats {
            expected_calibration_error: 0.05,
            maximum_calibration_error: 0.1,
            brier_score: 0.15,
            num_samples: 100,
        };

        assert!(stats.is_well_calibrated());

        let bad_stats = CalibrationStats {
            expected_calibration_error: 0.2,
            ..stats
        };

        assert!(!bad_stats.is_well_calibrated());
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_logit() {
        assert!((logit(0.5) - 0.0).abs() < 1e-5);
        assert!(logit(0.99) > 0.0);
        assert!(logit(0.01) < 0.0);
    }

    #[test]
    fn test_default_calibrator() {
        let calibrator = ConfidenceCalibrator::default();
        assert!(matches!(
            calibrator.method(),
            CalibrationMethod::TemperatureScaling(_)
        ));
    }

    #[test]
    fn test_isotonic_interpolation() {
        let mut calibrator = ConfidenceCalibrator::isotonic();

        calibrator.add_samples(&[
            (0.0, false),
            (0.25, false),
            (0.5, true),
            (0.75, true),
            (1.0, true),
        ]);

        calibrator.fit().unwrap();

        // Test interpolation at various points
        let cal_0 = calibrator.calibrate(0.0);
        let cal_50 = calibrator.calibrate(0.5);
        let cal_100 = calibrator.calibrate(1.0);

        assert!(cal_0 <= cal_50);
        assert!(cal_50 <= cal_100);
    }

    #[test]
    fn test_histogram_bins_minimum() {
        let mut calibrator = ConfidenceCalibrator::histogram_binning(0);
        calibrator.add_sample(0.5, true);
        calibrator.add_sample(0.5, false);

        // Should not panic with 0 bins (clamped to 1)
        let result = calibrator.fit();
        assert!(result.is_ok());
    }
}
