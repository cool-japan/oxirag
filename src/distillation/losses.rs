//! Distillation loss functions for knowledge distillation.
//!
//! This module provides various loss functions used in knowledge distillation,
//! enabling the transfer of knowledge from teacher models to student models.

use serde::{Deserialize, Serialize};

/// A trait for computing distillation losses.
///
/// Implementations of this trait provide different loss functions for
/// knowledge distillation, comparing teacher and student model outputs.
pub trait DistillationLoss: Send + Sync {
    /// Compute the loss between teacher and student logits.
    ///
    /// # Arguments
    ///
    /// * `teacher_logits` - Output logits from the teacher model
    /// * `student_logits` - Output logits from the student model
    /// * `labels` - Optional ground truth labels for hard target loss
    ///
    /// # Returns
    ///
    /// The computed loss value (lower is better).
    fn compute(
        &self,
        teacher_logits: &[f32],
        student_logits: &[f32],
        labels: Option<&[f32]>,
    ) -> f32;

    /// Get the name of this loss function.
    fn name(&self) -> &'static str;
}

/// Temperature scaling utilities for softening probability distributions.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureScaling;

impl TemperatureScaling {
    /// Apply temperature scaling to logits to soften the distribution.
    ///
    /// Higher temperatures produce softer distributions, revealing more
    /// information about the relationships between classes.
    ///
    /// # Arguments
    ///
    /// * `logits` - The input logits
    /// * `temperature` - Temperature parameter (T > 1 softens, T < 1 sharpens)
    ///
    /// # Returns
    ///
    /// Softened probability distribution.
    #[must_use]
    pub fn soften(logits: &[f32], temperature: f32) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let temp = temperature.max(f32::EPSILON);
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
        Self::softmax(&scaled)
    }

    /// Compute the softmax of a slice of logits.
    ///
    /// # Arguments
    ///
    /// * `logits` - The input logits
    ///
    /// # Returns
    ///
    /// Probability distribution summing to 1.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        // Numerical stability: subtract max before exp
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        if sum <= f32::EPSILON {
            // Avoid division by zero
            let uniform = 1.0 / logits.len() as f32;
            vec![uniform; logits.len()]
        } else {
            exp_vals.iter().map(|&x| x / sum).collect()
        }
    }

    /// Compute log softmax for numerical stability in cross-entropy.
    ///
    /// # Arguments
    ///
    /// * `logits` - The input logits
    ///
    /// # Returns
    ///
    /// Log probabilities.
    #[must_use]
    pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let shifted: Vec<f32> = logits.iter().map(|&x| x - max_val).collect();
        let log_sum_exp: f32 = shifted.iter().map(|&x| x.exp()).sum::<f32>().ln();

        shifted.iter().map(|&x| x - log_sum_exp).collect()
    }
}

/// Configuration for loss functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// The type of loss function to use.
    pub loss_type: LossType,
    /// Temperature for softening distributions (typically 1.0 - 20.0).
    pub temperature: f32,
    /// Weight for soft target loss (teacher knowledge).
    pub alpha: f32,
    /// Weight for hard target loss (ground truth labels).
    pub beta: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            loss_type: LossType::Combined,
            temperature: 4.0,
            alpha: 0.7,
            beta: 0.3,
        }
    }
}

impl LossConfig {
    /// Create a new loss configuration with the specified loss type.
    #[must_use]
    pub fn with_loss_type(loss_type: LossType) -> Self {
        Self {
            loss_type,
            ..Default::default()
        }
    }

    /// Set the temperature parameter.
    #[must_use]
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(0.1);
        self
    }

    /// Set the alpha weight for soft targets.
    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set the beta weight for hard targets.
    #[must_use]
    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta.clamp(0.0, 1.0);
        self
    }

    /// Validate the configuration.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.temperature > 0.0
            && self.alpha >= 0.0
            && self.alpha <= 1.0
            && self.beta >= 0.0
            && self.beta <= 1.0
    }

    /// Build a loss function from this configuration.
    #[must_use]
    pub fn build(&self) -> Box<dyn DistillationLoss> {
        match self.loss_type {
            LossType::KLDivergence => Box::new(KLDivergenceLoss::new(self.temperature)),
            LossType::MSE => Box::new(MSELoss),
            LossType::Cosine => Box::new(CosineLoss),
            LossType::SoftTarget => Box::new(SoftTargetLoss::new(self.temperature)),
            LossType::HardTarget => Box::new(HardTargetLoss),
            LossType::Combined => {
                let mut combined = CombinedLoss::new();
                combined.add_loss(Box::new(SoftTargetLoss::new(self.temperature)), self.alpha);
                combined.add_loss(Box::new(HardTargetLoss), self.beta);
                Box::new(combined)
            }
        }
    }
}

/// Types of loss functions available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossType {
    /// Kullback-Leibler divergence loss.
    KLDivergence,
    /// Mean squared error loss.
    MSE,
    /// Cosine similarity loss.
    Cosine,
    /// Soft target cross-entropy with temperature.
    SoftTarget,
    /// Hard target cross-entropy with ground truth labels.
    HardTarget,
    /// Weighted combination of multiple losses.
    Combined,
}

/// KL Divergence loss between teacher and student distributions.
///
/// Measures the information lost when using the student distribution
/// to approximate the teacher distribution.
#[derive(Debug, Clone)]
pub struct KLDivergenceLoss {
    /// Temperature for softening distributions.
    temperature: f32,
}

impl KLDivergenceLoss {
    /// Create a new KL divergence loss with the specified temperature.
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature: temperature.max(0.1),
        }
    }

    /// Create with default temperature of 4.0.
    #[must_use]
    pub fn with_default_temperature() -> Self {
        Self::new(4.0)
    }

    /// Get the temperature value.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self::with_default_temperature()
    }
}

impl DistillationLoss for KLDivergenceLoss {
    fn compute(
        &self,
        teacher_logits: &[f32],
        student_logits: &[f32],
        _labels: Option<&[f32]>,
    ) -> f32 {
        if teacher_logits.is_empty() || student_logits.is_empty() {
            return 0.0;
        }

        if teacher_logits.len() != student_logits.len() {
            return f32::MAX;
        }

        let teacher_probs = TemperatureScaling::soften(teacher_logits, self.temperature);
        let student_log_probs = {
            let scaled: Vec<f32> = student_logits
                .iter()
                .map(|&x| x / self.temperature)
                .collect();
            TemperatureScaling::log_softmax(&scaled)
        };

        // KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        // We compute: sum(P * log(P)) - sum(P * log(Q))
        // Since P is teacher_probs and log(Q) is student_log_probs
        let mut kl_div = 0.0;
        for (p, log_q) in teacher_probs.iter().zip(student_log_probs.iter()) {
            if *p > f32::EPSILON {
                let log_p = p.ln();
                kl_div += p * (log_p - log_q);
            }
        }

        // Scale by T^2 as is standard in knowledge distillation
        kl_div * self.temperature * self.temperature
    }

    fn name(&self) -> &'static str {
        "KLDivergence"
    }
}

/// Mean squared error loss between teacher and student outputs.
///
/// Simple L2 loss that directly compares output values.
#[derive(Debug, Clone, Copy, Default)]
pub struct MSELoss;

impl MSELoss {
    /// Create a new MSE loss.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DistillationLoss for MSELoss {
    #[allow(clippy::cast_precision_loss)]
    fn compute(
        &self,
        teacher_logits: &[f32],
        student_logits: &[f32],
        _labels: Option<&[f32]>,
    ) -> f32 {
        if teacher_logits.is_empty() || student_logits.is_empty() {
            return 0.0;
        }

        if teacher_logits.len() != student_logits.len() {
            return f32::MAX;
        }

        let sum_sq: f32 = teacher_logits
            .iter()
            .zip(student_logits.iter())
            .map(|(t, s)| (t - s).powi(2))
            .sum();

        sum_sq / teacher_logits.len() as f32
    }

    fn name(&self) -> &'static str {
        "MSE"
    }
}

/// Cosine similarity loss between teacher and student outputs.
///
/// Measures the angular similarity between output vectors.
/// Loss = 1 - `cosine_similarity` (so 0 when perfectly aligned).
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineLoss;

impl CosineLoss {
    /// Create a new cosine loss.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute cosine similarity between two vectors.
    #[must_use]
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

impl DistillationLoss for CosineLoss {
    fn compute(
        &self,
        teacher_logits: &[f32],
        student_logits: &[f32],
        _labels: Option<&[f32]>,
    ) -> f32 {
        if teacher_logits.is_empty() || student_logits.is_empty() {
            return 0.0;
        }

        if teacher_logits.len() != student_logits.len() {
            return f32::MAX;
        }

        // Loss = 1 - similarity (0 when identical, 2 when opposite)
        1.0 - Self::cosine_similarity(teacher_logits, student_logits)
    }

    fn name(&self) -> &'static str {
        "Cosine"
    }
}

/// Soft target cross-entropy loss with temperature scaling.
///
/// Uses the teacher's softened outputs as targets for the student.
#[derive(Debug, Clone)]
pub struct SoftTargetLoss {
    /// Temperature for softening distributions.
    temperature: f32,
}

impl SoftTargetLoss {
    /// Create a new soft target loss with the specified temperature.
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature: temperature.max(0.1),
        }
    }

    /// Create with default temperature of 4.0.
    #[must_use]
    pub fn with_default_temperature() -> Self {
        Self::new(4.0)
    }

    /// Get the temperature value.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl Default for SoftTargetLoss {
    fn default() -> Self {
        Self::with_default_temperature()
    }
}

impl DistillationLoss for SoftTargetLoss {
    fn compute(
        &self,
        teacher_logits: &[f32],
        student_logits: &[f32],
        _labels: Option<&[f32]>,
    ) -> f32 {
        if teacher_logits.is_empty() || student_logits.is_empty() {
            return 0.0;
        }

        if teacher_logits.len() != student_logits.len() {
            return f32::MAX;
        }

        let teacher_probs = TemperatureScaling::soften(teacher_logits, self.temperature);
        let student_log_probs = {
            let scaled: Vec<f32> = student_logits
                .iter()
                .map(|&x| x / self.temperature)
                .collect();
            TemperatureScaling::log_softmax(&scaled)
        };

        // Cross-entropy: -sum(P * log(Q))
        let mut cross_entropy = 0.0;
        for (p, log_q) in teacher_probs.iter().zip(student_log_probs.iter()) {
            cross_entropy -= p * log_q;
        }

        // Scale by T^2
        cross_entropy * self.temperature * self.temperature
    }

    fn name(&self) -> &'static str {
        "SoftTarget"
    }
}

/// Hard target cross-entropy loss with ground truth labels.
///
/// Standard cross-entropy loss using the true labels.
#[derive(Debug, Clone, Copy, Default)]
pub struct HardTargetLoss;

impl HardTargetLoss {
    /// Create a new hard target loss.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DistillationLoss for HardTargetLoss {
    fn compute(
        &self,
        _teacher_logits: &[f32],
        student_logits: &[f32],
        labels: Option<&[f32]>,
    ) -> f32 {
        let Some(labels) = labels else {
            return 0.0;
        };

        if student_logits.is_empty() || labels.is_empty() {
            return 0.0;
        }

        if student_logits.len() != labels.len() {
            return f32::MAX;
        }

        let student_log_probs = TemperatureScaling::log_softmax(student_logits);

        // Cross-entropy: -sum(Y * log(P))
        let mut cross_entropy = 0.0;
        for (label, log_p) in labels.iter().zip(student_log_probs.iter()) {
            if *label > f32::EPSILON {
                cross_entropy -= label * log_p;
            }
        }

        cross_entropy
    }

    fn name(&self) -> &'static str {
        "HardTarget"
    }
}

/// A weighted combination of multiple loss functions.
///
/// Allows combining different loss types with custom weights.
pub struct CombinedLoss {
    /// The loss functions and their weights.
    losses: Vec<(Box<dyn DistillationLoss>, f32)>,
}

impl CombinedLoss {
    /// Create a new empty combined loss.
    #[must_use]
    pub fn new() -> Self {
        Self { losses: Vec::new() }
    }

    /// Add a loss function with a weight.
    pub fn add_loss(&mut self, loss: Box<dyn DistillationLoss>, weight: f32) {
        if weight > 0.0 {
            self.losses.push((loss, weight));
        }
    }

    /// Create a builder-style combined loss.
    #[must_use]
    pub fn with_loss(mut self, loss: Box<dyn DistillationLoss>, weight: f32) -> Self {
        self.add_loss(loss, weight);
        self
    }

    /// Get the number of loss components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.losses.len()
    }

    /// Check if there are no loss components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.losses.is_empty()
    }

    /// Get the total weight.
    #[must_use]
    pub fn total_weight(&self) -> f32 {
        self.losses.iter().map(|(_, w)| w).sum()
    }

    /// Normalize the weights to sum to 1.0.
    pub fn normalize_weights(&mut self) {
        let total = self.total_weight();
        if total > f32::EPSILON {
            for (_, weight) in &mut self.losses {
                *weight /= total;
            }
        }
    }
}

impl Default for CombinedLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillationLoss for CombinedLoss {
    fn compute(
        &self,
        teacher_logits: &[f32],
        student_logits: &[f32],
        labels: Option<&[f32]>,
    ) -> f32 {
        if self.losses.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for (loss, weight) in &self.losses {
            total_loss += weight * loss.compute(teacher_logits, student_logits, labels);
        }

        total_loss
    }

    fn name(&self) -> &'static str {
        "Combined"
    }
}

impl std::fmt::Debug for CombinedLoss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedLoss")
            .field("num_losses", &self.losses.len())
            .field("total_weight", &self.total_weight())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ========================
    // TemperatureScaling Tests
    // ========================

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = TemperatureScaling::softmax(&logits);

        // Check sum is 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check ordering is preserved
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = TemperatureScaling::softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let probs = TemperatureScaling::softmax(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without the max subtraction trick
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = TemperatureScaling::softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_soften_high_temperature() {
        let logits = vec![0.0, 10.0];

        // Low temperature: sharp distribution
        let sharp = TemperatureScaling::soften(&logits, 1.0);

        // High temperature: soft distribution
        let soft = TemperatureScaling::soften(&logits, 10.0);

        // Soft should be more uniform (higher entropy)
        assert!(soft[0] > sharp[0]);
    }

    #[test]
    fn test_soften_empty() {
        let result = TemperatureScaling::soften(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_log_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = TemperatureScaling::log_softmax(&logits);

        // exp(log_probs) should sum to 1
        let sum: f32 = log_probs.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_empty() {
        let result = TemperatureScaling::log_softmax(&[]);
        assert!(result.is_empty());
    }

    // ========================
    // LossConfig Tests
    // ========================

    #[test]
    fn test_loss_config_default() {
        let config = LossConfig::default();
        assert_eq!(config.loss_type, LossType::Combined);
        assert!((config.temperature - 4.0).abs() < f32::EPSILON);
        assert!((config.alpha - 0.7).abs() < f32::EPSILON);
        assert!((config.beta - 0.3).abs() < f32::EPSILON);
        assert!(config.is_valid());
    }

    #[test]
    fn test_loss_config_builder() {
        let config = LossConfig::with_loss_type(LossType::KLDivergence)
            .temperature(2.0)
            .alpha(0.5)
            .beta(0.5);

        assert_eq!(config.loss_type, LossType::KLDivergence);
        assert!((config.temperature - 2.0).abs() < f32::EPSILON);
        assert!(config.is_valid());
    }

    #[test]
    fn test_loss_config_build() {
        let config = LossConfig::with_loss_type(LossType::MSE);
        let loss = config.build();
        assert_eq!(loss.name(), "MSE");
    }

    #[test]
    fn test_loss_config_clamping() {
        let config = LossConfig::default().alpha(2.0).beta(-0.5);
        assert!((config.alpha - 1.0).abs() < f32::EPSILON);
        assert!((config.beta - 0.0).abs() < f32::EPSILON);
    }

    // ========================
    // KLDivergenceLoss Tests
    // ========================

    #[test]
    fn test_kl_divergence_identical() {
        let loss = KLDivergenceLoss::new(1.0);
        let logits = vec![1.0, 2.0, 3.0];

        let result = loss.compute(&logits, &logits, None);
        // KL divergence of identical distributions should be 0
        assert!(result < 1e-5);
    }

    #[test]
    fn test_kl_divergence_different() {
        let loss = KLDivergenceLoss::new(1.0);
        let teacher = vec![0.0, 0.0, 10.0]; // Peaked at index 2
        let student = vec![10.0, 0.0, 0.0]; // Peaked at index 0

        let result = loss.compute(&teacher, &student, None);
        // Should have significant divergence
        assert!(result > 0.0);
    }

    #[test]
    fn test_kl_divergence_empty() {
        let loss = KLDivergenceLoss::default();
        assert!((loss.compute(&[], &[], None) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_kl_divergence_size_mismatch() {
        let loss = KLDivergenceLoss::default();
        let result = loss.compute(&[1.0, 2.0], &[1.0], None);
        assert_eq!(result, f32::MAX);
    }

    #[test]
    fn test_kl_divergence_name() {
        let loss = KLDivergenceLoss::default();
        assert_eq!(loss.name(), "KLDivergence");
    }

    #[test]
    fn test_kl_divergence_temperature() {
        let loss = KLDivergenceLoss::new(4.0);
        assert!((loss.temperature() - 4.0).abs() < f32::EPSILON);
    }

    // ========================
    // MSELoss Tests
    // ========================

    #[test]
    fn test_mse_identical() {
        let loss = MSELoss::new();
        let logits = vec![1.0, 2.0, 3.0];

        let result = loss.compute(&logits, &logits, None);
        assert!((result - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mse_different() {
        let loss = MSELoss::new();
        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![2.0, 3.0, 4.0]; // Each differs by 1

        let result = loss.compute(&teacher, &student, None);
        // MSE = (1 + 1 + 1) / 3 = 1.0
        assert!((result - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mse_empty() {
        let loss = MSELoss::new();
        assert!((loss.compute(&[], &[], None) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mse_size_mismatch() {
        let loss = MSELoss;
        let result = loss.compute(&[1.0, 2.0], &[1.0], None);
        assert_eq!(result, f32::MAX);
    }

    #[test]
    fn test_mse_name() {
        let loss = MSELoss::new();
        assert_eq!(loss.name(), "MSE");
    }

    // ========================
    // CosineLoss Tests
    // ========================

    #[test]
    fn test_cosine_identical() {
        let loss = CosineLoss::new();
        let logits = vec![1.0, 2.0, 3.0];

        let result = loss.compute(&logits, &logits, None);
        // Identical vectors have cosine similarity of 1, so loss is 0
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_opposite() {
        let loss = CosineLoss::new();
        let teacher = vec![1.0, 0.0, 0.0];
        let student = vec![-1.0, 0.0, 0.0];

        let result = loss.compute(&teacher, &student, None);
        // Opposite vectors have cosine similarity of -1, so loss is 2
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let loss = CosineLoss::new();
        let teacher = vec![1.0, 0.0];
        let student = vec![0.0, 1.0];

        let result = loss.compute(&teacher, &student, None);
        // Orthogonal vectors have cosine similarity of 0, so loss is 1
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_empty() {
        let loss = CosineLoss::new();
        assert!((loss.compute(&[], &[], None) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_size_mismatch() {
        let loss = CosineLoss;
        let result = loss.compute(&[1.0, 2.0], &[1.0], None);
        assert_eq!(result, f32::MAX);
    }

    #[test]
    fn test_cosine_name() {
        let loss = CosineLoss::new();
        assert_eq!(loss.name(), "Cosine");
    }

    #[test]
    fn test_cosine_similarity_zero_norm() {
        let result = CosineLoss::cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]);
        assert!((result - 0.0).abs() < f32::EPSILON);
    }

    // ========================
    // SoftTargetLoss Tests
    // ========================

    #[test]
    fn test_soft_target_identical() {
        let loss = SoftTargetLoss::new(1.0);
        let logits = vec![1.0, 2.0, 3.0];

        let result = loss.compute(&logits, &logits, None);
        // For identical distributions, cross-entropy equals entropy
        // This should be non-zero but consistent
        assert!(result > 0.0);
    }

    #[test]
    fn test_soft_target_different() {
        let loss = SoftTargetLoss::new(1.0);
        let teacher = vec![0.0, 0.0, 10.0];
        let student = vec![10.0, 0.0, 0.0];

        let result = loss.compute(&teacher, &student, None);
        // Cross-entropy should be high
        assert!(result > 0.0);
    }

    #[test]
    fn test_soft_target_empty() {
        let loss = SoftTargetLoss::default();
        assert!((loss.compute(&[], &[], None) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_soft_target_size_mismatch() {
        let loss = SoftTargetLoss::default();
        let result = loss.compute(&[1.0, 2.0], &[1.0], None);
        assert_eq!(result, f32::MAX);
    }

    #[test]
    fn test_soft_target_name() {
        let loss = SoftTargetLoss::default();
        assert_eq!(loss.name(), "SoftTarget");
    }

    #[test]
    fn test_soft_target_temperature() {
        let loss = SoftTargetLoss::new(8.0);
        assert!((loss.temperature() - 8.0).abs() < f32::EPSILON);
    }

    // ========================
    // HardTargetLoss Tests
    // ========================

    #[test]
    fn test_hard_target_with_labels() {
        let loss = HardTargetLoss::new();
        let student = vec![0.0, 0.0, 10.0]; // Peaked at index 2
        let labels = vec![0.0, 0.0, 1.0]; // Ground truth is index 2

        let result = loss.compute(&[], &student, Some(&labels));
        // Student correctly predicts the label, loss should be low
        assert!(result < 1.0);
    }

    #[test]
    fn test_hard_target_wrong_prediction() {
        let loss = HardTargetLoss::new();
        let student = vec![10.0, 0.0, 0.0]; // Peaked at index 0
        let labels = vec![0.0, 0.0, 1.0]; // Ground truth is index 2

        let result = loss.compute(&[], &student, Some(&labels));
        // Student incorrectly predicts, loss should be high
        assert!(result > 1.0);
    }

    #[test]
    fn test_hard_target_no_labels() {
        let loss = HardTargetLoss::new();
        let result = loss.compute(&[], &[1.0, 2.0], None);
        assert!((result - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hard_target_empty() {
        let loss = HardTargetLoss::new();
        assert!((loss.compute(&[], &[], Some(&[])) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hard_target_size_mismatch() {
        let loss = HardTargetLoss;
        let result = loss.compute(&[], &[1.0, 2.0], Some(&[1.0]));
        assert_eq!(result, f32::MAX);
    }

    #[test]
    fn test_hard_target_name() {
        let loss = HardTargetLoss::new();
        assert_eq!(loss.name(), "HardTarget");
    }

    // ========================
    // CombinedLoss Tests
    // ========================

    #[test]
    fn test_combined_empty() {
        let loss = CombinedLoss::new();
        let result = loss.compute(&[1.0], &[1.0], None);
        assert!((result - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_combined_single() {
        let mut loss = CombinedLoss::new();
        loss.add_loss(Box::new(MSELoss), 1.0);

        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![2.0, 3.0, 4.0];

        let result = loss.compute(&teacher, &student, None);
        assert!((result - 1.0).abs() < f32::EPSILON); // Same as MSE alone
    }

    #[test]
    fn test_combined_weighted() {
        let mut loss = CombinedLoss::new();
        loss.add_loss(Box::new(MSELoss), 0.5);

        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![2.0, 3.0, 4.0];

        let result = loss.compute(&teacher, &student, None);
        assert!((result - 0.5).abs() < f32::EPSILON); // 0.5 * 1.0
    }

    #[test]
    fn test_combined_builder() {
        let loss = CombinedLoss::new()
            .with_loss(Box::new(MSELoss), 0.5)
            .with_loss(Box::new(CosineLoss), 0.5);

        assert_eq!(loss.len(), 2);
        assert!(!loss.is_empty());
    }

    #[test]
    fn test_combined_total_weight() {
        let mut loss = CombinedLoss::new();
        loss.add_loss(Box::new(MSELoss), 0.3);
        loss.add_loss(Box::new(CosineLoss), 0.7);

        assert!((loss.total_weight() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_combined_normalize_weights() {
        let mut loss = CombinedLoss::new();
        loss.add_loss(Box::new(MSELoss), 1.0);
        loss.add_loss(Box::new(CosineLoss), 3.0);

        loss.normalize_weights();

        assert!((loss.total_weight() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_combined_zero_weight_ignored() {
        let mut loss = CombinedLoss::new();
        loss.add_loss(Box::new(MSELoss), 0.0);

        assert_eq!(loss.len(), 0);
    }

    #[test]
    fn test_combined_name() {
        let loss = CombinedLoss::new();
        assert_eq!(loss.name(), "Combined");
    }

    #[test]
    fn test_combined_debug() {
        let loss = CombinedLoss::new()
            .with_loss(Box::new(MSELoss), 0.5)
            .with_loss(Box::new(CosineLoss), 0.5);

        let debug_str = format!("{loss:?}");
        assert!(debug_str.contains("CombinedLoss"));
        assert!(debug_str.contains("num_losses"));
    }

    // ========================
    // Integration Tests
    // ========================

    #[test]
    fn test_distillation_loss_trait_object() {
        let losses: Vec<Box<dyn DistillationLoss>> = vec![
            Box::new(KLDivergenceLoss::default()),
            Box::new(MSELoss),
            Box::new(CosineLoss),
            Box::new(SoftTargetLoss::default()),
            Box::new(HardTargetLoss),
        ];

        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![1.0, 2.0, 3.0];
        let labels = vec![0.0, 0.0, 1.0];

        for loss in &losses {
            let _ = loss.compute(&teacher, &student, Some(&labels));
            let _ = loss.name();
        }
    }

    #[test]
    fn test_loss_config_all_types() {
        let types = [
            LossType::KLDivergence,
            LossType::MSE,
            LossType::Cosine,
            LossType::SoftTarget,
            LossType::HardTarget,
            LossType::Combined,
        ];

        for loss_type in types {
            let config = LossConfig::with_loss_type(loss_type);
            let loss = config.build();
            let _ = loss.compute(&[1.0, 2.0], &[1.0, 2.0], Some(&[1.0, 0.0]));
        }
    }

    #[test]
    fn test_typical_distillation_scenario() {
        // Simulate a typical distillation scenario
        let config = LossConfig::default();
        let loss = config.build();

        // Teacher gives high confidence on class 2
        let teacher_logits = vec![0.1, 0.2, 5.0];
        // Student is learning, but not quite there
        let student_logits = vec![0.5, 0.5, 3.0];
        // Ground truth is class 2
        let labels = vec![0.0, 0.0, 1.0];

        let loss_val = loss.compute(&teacher_logits, &student_logits, Some(&labels));
        assert!(loss_val > 0.0);
        assert!(loss_val.is_finite());
    }
}
