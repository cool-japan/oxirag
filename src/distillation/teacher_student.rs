//! Teacher-Student architecture types for knowledge distillation.
//!
//! This module provides traits and types for implementing knowledge distillation
//! between a large teacher model and a smaller student model.
//!
//! # Overview
//!
//! Knowledge distillation transfers knowledge from a large, accurate teacher model
//! to a smaller, faster student model. This module provides:
//!
//! - `TeacherModel` trait for teacher model interface
//! - `StudentModel` trait for student model interface (extends `TeacherModel`)
//! - `DistillationPair` for pairing teacher and student
//! - Configuration and result types for distillation steps
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::distillation::teacher_student::{
//!     TeacherModel, StudentModel, DistillationPair,
//!     DistillationStepConfig, MockTeacherModel, MockStudentModel,
//! };
//!
//! let teacher = MockTeacherModel::new(768);
//! let student = MockStudentModel::new(384);
//! let pair = DistillationPair::new(teacher, student);
//!
//! let config = DistillationStepConfig::default();
//! let input = vec![0.1, 0.2, 0.3];
//! let result = pair.distill_step(&input, &config);
//! ```

use serde::{Deserialize, Serialize};

/// Trait for teacher models in knowledge distillation.
///
/// Teacher models are typically large, accurate models that provide
/// soft targets (probability distributions) for training student models.
pub trait TeacherModel: Send + Sync {
    /// Get the model's output predictions for the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - Input feature vector
    ///
    /// # Returns
    ///
    /// Output predictions (logits or probabilities)
    fn forward(&self, input: &[f32]) -> Vec<f32>;

    /// Get intermediate hidden state representations.
    ///
    /// # Arguments
    ///
    /// * `input` - Input feature vector
    ///
    /// # Returns
    ///
    /// A vector of hidden states from each layer
    fn get_hidden_states(&self, input: &[f32]) -> Vec<Vec<f32>>;

    /// Get attention weight matrices.
    ///
    /// # Arguments
    ///
    /// * `input` - Input feature vector
    ///
    /// # Returns
    ///
    /// A vector of attention weight matrices (flattened)
    fn get_attention_weights(&self, input: &[f32]) -> Vec<Vec<f32>>;

    /// Get the total number of model parameters.
    ///
    /// # Returns
    ///
    /// The parameter count
    fn model_size(&self) -> usize;
}

/// Trait for student models in knowledge distillation.
///
/// Student models are smaller models that learn from teacher models.
/// They have all the capabilities of teacher models plus the ability
/// to update their weights during training.
pub trait StudentModel: TeacherModel {
    /// Update model weights using the provided gradients.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradient values for weight updates
    fn update_weights(&mut self, gradients: &[f32]);
}

/// Configuration for a single distillation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationStepConfig {
    /// Learning rate for weight updates.
    pub learning_rate: f32,
    /// Temperature for softening probability distributions.
    /// Higher temperatures produce softer distributions.
    pub temperature: f32,
    /// Weight for the soft target loss (distillation loss).
    pub soft_target_weight: f32,
    /// Weight for the hard target loss (if labels are available).
    pub hard_target_weight: f32,
    /// Weight for hidden state matching loss.
    pub hidden_state_weight: f32,
    /// Weight for attention matching loss.
    pub attention_weight: f32,
}

impl Default for DistillationStepConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            temperature: 3.0,
            soft_target_weight: 0.7,
            hard_target_weight: 0.3,
            hidden_state_weight: 0.1,
            attention_weight: 0.05,
        }
    }
}

impl DistillationStepConfig {
    /// Create a new configuration with the specified learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr.max(0.0);
        self
    }

    /// Set the temperature for soft targets.
    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.max(0.1);
        self
    }

    /// Set the soft target weight.
    #[must_use]
    pub fn with_soft_target_weight(mut self, weight: f32) -> Self {
        self.soft_target_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the hard target weight.
    #[must_use]
    pub fn with_hard_target_weight(mut self, weight: f32) -> Self {
        self.hard_target_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the hidden state matching weight.
    #[must_use]
    pub fn with_hidden_state_weight(mut self, weight: f32) -> Self {
        self.hidden_state_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the attention matching weight.
    #[must_use]
    pub fn with_attention_weight(mut self, weight: f32) -> Self {
        self.attention_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Check if the configuration is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.learning_rate > 0.0
            && self.temperature > 0.0
            && self.soft_target_weight >= 0.0
            && self.hard_target_weight >= 0.0
    }

    /// Calculate the total weight sum for normalization.
    #[must_use]
    pub fn total_weight(&self) -> f32 {
        self.soft_target_weight
            + self.hard_target_weight
            + self.hidden_state_weight
            + self.attention_weight
    }
}

/// Result of a single distillation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationStepResult {
    /// Total combined loss.
    pub total_loss: f32,
    /// Loss from soft target matching (KL divergence).
    pub soft_target_loss: f32,
    /// Loss from hard target matching (cross-entropy).
    pub hard_target_loss: f32,
    /// Loss from hidden state matching (MSE).
    pub hidden_state_loss: f32,
    /// Loss from attention matching (MSE).
    pub attention_loss: f32,
    /// Computed gradients for weight updates.
    pub gradients: Vec<f32>,
}

impl Default for DistillationStepResult {
    fn default() -> Self {
        Self {
            total_loss: 0.0,
            soft_target_loss: 0.0,
            hard_target_loss: 0.0,
            hidden_state_loss: 0.0,
            attention_loss: 0.0,
            gradients: Vec::new(),
        }
    }
}

impl DistillationStepResult {
    /// Create a new result with all losses set.
    #[must_use]
    pub fn new(
        soft_target_loss: f32,
        hard_target_loss: f32,
        hidden_state_loss: f32,
        attention_loss: f32,
        gradients: Vec<f32>,
        config: &DistillationStepConfig,
    ) -> Self {
        let total_loss = config.soft_target_weight * soft_target_loss
            + config.hard_target_weight * hard_target_loss
            + config.hidden_state_weight * hidden_state_loss
            + config.attention_weight * attention_loss;

        Self {
            total_loss,
            soft_target_loss,
            hard_target_loss,
            hidden_state_loss,
            attention_loss,
            gradients,
        }
    }

    /// Check if the loss values are valid (not NaN or infinite).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.total_loss.is_finite()
            && self.soft_target_loss.is_finite()
            && self.hard_target_loss.is_finite()
            && self.hidden_state_loss.is_finite()
            && self.attention_loss.is_finite()
    }

    /// Get a summary string of the losses.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "total={:.4}, soft={:.4}, hard={:.4}, hidden={:.4}, attn={:.4}",
            self.total_loss,
            self.soft_target_loss,
            self.hard_target_loss,
            self.hidden_state_loss,
            self.attention_loss
        )
    }
}

/// Metrics collected during distillation evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistillationMetrics {
    /// Average accuracy of the student model.
    pub accuracy: f32,
    /// Average loss across evaluation samples.
    pub average_loss: f32,
    /// Compression ratio (teacher size / student size).
    pub compression_ratio: f32,
    /// Number of samples evaluated.
    pub samples_evaluated: usize,
    /// Teacher model size in parameters.
    pub teacher_size: usize,
    /// Student model size in parameters.
    pub student_size: usize,
}

impl DistillationMetrics {
    /// Create new metrics.
    #[must_use]
    pub fn new(
        accuracy: f32,
        average_loss: f32,
        teacher_size: usize,
        student_size: usize,
        samples_evaluated: usize,
    ) -> Self {
        let compression_ratio = if student_size > 0 {
            #[allow(clippy::cast_precision_loss)]
            {
                teacher_size as f32 / student_size as f32
            }
        } else {
            0.0
        };

        Self {
            accuracy,
            average_loss,
            compression_ratio,
            samples_evaluated,
            teacher_size,
            student_size,
        }
    }

    /// Get the memory savings percentage.
    #[must_use]
    pub fn memory_savings_percent(&self) -> f32 {
        if self.compression_ratio > 0.0 {
            (1.0 - 1.0 / self.compression_ratio) * 100.0
        } else {
            0.0
        }
    }

    /// Check if the metrics indicate successful distillation.
    #[must_use]
    pub fn is_successful(&self, min_accuracy: f32, max_loss: f32) -> bool {
        self.accuracy >= min_accuracy && self.average_loss <= max_loss
    }

    /// Get a summary string of the metrics.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "accuracy={:.2}%, loss={:.4}, compression={:.2}x, samples={}",
            self.accuracy * 100.0,
            self.average_loss,
            self.compression_ratio,
            self.samples_evaluated
        )
    }
}

/// Pairs a teacher and student model for distillation.
///
/// This struct manages the distillation process between a teacher
/// and student model, providing methods for training steps and evaluation.
pub struct DistillationPair<T, S>
where
    T: TeacherModel,
    S: StudentModel,
{
    teacher: T,
    student: S,
}

impl<T, S> DistillationPair<T, S>
where
    T: TeacherModel,
    S: StudentModel,
{
    /// Create a new distillation pair.
    #[must_use]
    pub fn new(teacher: T, student: S) -> Self {
        Self { teacher, student }
    }

    /// Get a reference to the teacher model.
    #[must_use]
    pub fn teacher(&self) -> &T {
        &self.teacher
    }

    /// Get a reference to the student model.
    #[must_use]
    pub fn student(&self) -> &S {
        &self.student
    }

    /// Get a mutable reference to the student model.
    pub fn student_mut(&mut self) -> &mut S {
        &mut self.student
    }

    /// Get the compression ratio (teacher size / student size).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        let student_size = self.student.model_size();
        if student_size > 0 {
            self.teacher.model_size() as f32 / student_size as f32
        } else {
            0.0
        }
    }

    /// Perform a single distillation step.
    ///
    /// This computes losses between teacher and student outputs,
    /// generates gradients, and updates the student weights.
    ///
    /// # Arguments
    ///
    /// * `input` - Input feature vector
    /// * `config` - Distillation configuration
    ///
    /// # Returns
    ///
    /// The result of the distillation step including losses and gradients
    pub fn distill_step(
        &mut self,
        input: &[f32],
        config: &DistillationStepConfig,
    ) -> DistillationStepResult {
        // Get teacher outputs
        let teacher_output = self.teacher.forward(input);
        let teacher_hidden = self.teacher.get_hidden_states(input);
        let teacher_attention = self.teacher.get_attention_weights(input);

        // Get student outputs
        let student_output = self.student.forward(input);
        let student_hidden = self.student.get_hidden_states(input);
        let student_attention = self.student.get_attention_weights(input);

        // Compute soft target loss (using temperature-scaled softmax)
        let soft_target_loss =
            Self::compute_kl_divergence(&teacher_output, &student_output, config.temperature);

        // Compute hard target loss (assuming teacher output is the target)
        let hard_target_loss = Self::compute_cross_entropy(&teacher_output, &student_output);

        // Compute hidden state matching loss
        let hidden_state_loss = Self::compute_hidden_state_loss(&teacher_hidden, &student_hidden);

        // Compute attention matching loss
        let attention_loss = Self::compute_attention_loss(&teacher_attention, &student_attention);

        // Compute gradients (simplified - in practice would use backprop)
        let gradients = Self::compute_gradients(
            &student_output,
            &teacher_output,
            config.learning_rate,
            config.temperature,
        );

        // Update student weights
        self.student.update_weights(&gradients);

        DistillationStepResult::new(
            soft_target_loss,
            hard_target_loss,
            hidden_state_loss,
            attention_loss,
            gradients,
            config,
        )
    }

    /// Evaluate the distillation quality on a set of inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of input feature vectors
    ///
    /// # Returns
    ///
    /// Metrics summarizing the distillation quality
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&self, inputs: &[Vec<f32>]) -> DistillationMetrics {
        if inputs.is_empty() {
            return DistillationMetrics::new(
                0.0,
                0.0,
                self.teacher.model_size(),
                self.student.model_size(),
                0,
            );
        }

        let mut total_loss = 0.0;
        let mut correct = 0;

        for input in inputs {
            let teacher_output = self.teacher.forward(input);
            let student_output = self.student.forward(input);

            // Compute loss
            let loss = Self::compute_mse(&teacher_output, &student_output);
            total_loss += loss;

            // Check if student's argmax matches teacher's
            if Self::argmax(&student_output) == Self::argmax(&teacher_output) {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / inputs.len() as f32;
        let average_loss = total_loss / inputs.len() as f32;

        DistillationMetrics::new(
            accuracy,
            average_loss,
            self.teacher.model_size(),
            self.student.model_size(),
            inputs.len(),
        )
    }

    /// Compute KL divergence between teacher and student outputs.
    #[allow(clippy::cast_precision_loss)]
    fn compute_kl_divergence(teacher: &[f32], student: &[f32], temperature: f32) -> f32 {
        if teacher.is_empty() || student.is_empty() {
            return 0.0;
        }

        // Apply temperature-scaled softmax
        let teacher_soft = Self::softmax_with_temperature(teacher, temperature);
        let student_soft = Self::softmax_with_temperature(student, temperature);

        // KL divergence: sum(P * log(P/Q))
        let mut kl = 0.0;
        for (t, s) in teacher_soft.iter().zip(student_soft.iter()) {
            if *t > 1e-10 && *s > 1e-10 {
                kl += t * (t.ln() - s.ln());
            }
        }

        // Scale by T^2 as per Hinton et al.
        kl * temperature * temperature
    }

    /// Compute cross-entropy loss.
    fn compute_cross_entropy(target: &[f32], prediction: &[f32]) -> f32 {
        if target.is_empty() || prediction.is_empty() {
            return 0.0;
        }

        let pred_soft = Self::softmax_with_temperature(prediction, 1.0);
        let target_soft = Self::softmax_with_temperature(target, 1.0);

        let mut loss = 0.0;
        for (t, p) in target_soft.iter().zip(pred_soft.iter()) {
            if *p > 1e-10 {
                loss -= t * p.ln();
            }
        }

        loss
    }

    /// Compute mean squared error between hidden states.
    #[allow(clippy::cast_precision_loss)]
    fn compute_hidden_state_loss(teacher: &[Vec<f32>], student: &[Vec<f32>]) -> f32 {
        if teacher.is_empty() || student.is_empty() {
            return 0.0;
        }

        let num_layers = teacher.len().min(student.len());
        let mut total_loss = 0.0;

        for i in 0..num_layers {
            total_loss += Self::compute_mse(&teacher[i], &student[i]);
        }

        total_loss / num_layers as f32
    }

    /// Compute mean squared error between attention weights.
    #[allow(clippy::cast_precision_loss)]
    fn compute_attention_loss(teacher: &[Vec<f32>], student: &[Vec<f32>]) -> f32 {
        if teacher.is_empty() || student.is_empty() {
            return 0.0;
        }

        let num_heads = teacher.len().min(student.len());
        let mut total_loss = 0.0;

        for i in 0..num_heads {
            total_loss += Self::compute_mse(&teacher[i], &student[i]);
        }

        total_loss / num_heads as f32
    }

    /// Compute mean squared error between two vectors.
    #[allow(clippy::cast_precision_loss)]
    fn compute_mse(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let len = a.len().min(b.len());
        let mut sum = 0.0;

        for i in 0..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }

        sum / len as f32
    }

    /// Apply softmax with temperature scaling.
    fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let temp = temperature.max(0.01);
        let scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();

        // Find max for numerical stability
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = scaled.iter().map(|x| (x - max).exp()).sum();

        scaled.iter().map(|x| (x - max).exp() / exp_sum).collect()
    }

    /// Compute simplified gradients for weight updates.
    fn compute_gradients(
        student_output: &[f32],
        teacher_output: &[f32],
        learning_rate: f32,
        temperature: f32,
    ) -> Vec<f32> {
        if student_output.is_empty() || teacher_output.is_empty() {
            return Vec::new();
        }

        let student_soft = Self::softmax_with_temperature(student_output, temperature);
        let teacher_soft = Self::softmax_with_temperature(teacher_output, temperature);

        // Gradient of KL divergence w.r.t. student logits (simplified)
        student_soft
            .iter()
            .zip(teacher_soft.iter())
            .map(|(s, t)| learning_rate * (s - t))
            .collect()
    }

    /// Find the index of the maximum value.
    fn argmax(values: &[f32]) -> usize {
        if values.is_empty() {
            return 0;
        }

        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }
}

/// A mock teacher model for testing purposes.
#[derive(Debug, Clone)]
pub struct MockTeacherModel {
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    parameter_count: usize,
}

impl MockTeacherModel {
    /// Create a new mock teacher model.
    #[must_use]
    pub fn new(hidden_size: usize) -> Self {
        let num_layers = 12;
        let num_heads = 12;
        // Approximate parameter count for a transformer
        let parameter_count = hidden_size * hidden_size * 4 * num_layers;

        Self {
            hidden_size,
            num_layers,
            num_heads,
            parameter_count,
        }
    }

    /// Create a mock teacher with custom layer count.
    #[must_use]
    pub fn with_layers(hidden_size: usize, num_layers: usize, num_heads: usize) -> Self {
        let parameter_count = hidden_size * hidden_size * 4 * num_layers;

        Self {
            hidden_size,
            num_layers,
            num_heads,
            parameter_count,
        }
    }

    /// Get the hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

impl TeacherModel for MockTeacherModel {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Simple mock: return scaled input with some transformation
        let mut output = vec![0.0; self.hidden_size.min(input.len().max(1))];
        for (i, val) in output.iter_mut().enumerate() {
            if i < input.len() {
                *val = input[i] * 1.5 + 0.1;
            }
        }
        output
    }

    fn get_hidden_states(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut states = Vec::with_capacity(self.num_layers);
        for layer in 0..self.num_layers {
            let mut state = vec![0.0; self.hidden_size.min(input.len().max(1))];
            for (i, val) in state.iter_mut().enumerate() {
                if i < input.len() {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        *val = input[i] * (1.0 + 0.1 * layer as f32);
                    }
                }
            }
            states.push(state);
        }
        states
    }

    fn get_attention_weights(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let seq_len = input.len().max(1);
        let mut weights = Vec::with_capacity(self.num_heads);
        for _head in 0..self.num_heads {
            // Mock attention weights (uniform distribution)
            #[allow(clippy::cast_precision_loss)]
            let weight = vec![1.0 / seq_len as f32; seq_len * seq_len];
            weights.push(weight);
        }
        weights
    }

    fn model_size(&self) -> usize {
        self.parameter_count
    }
}

/// A mock student model for testing purposes.
#[derive(Debug, Clone)]
pub struct MockStudentModel {
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    parameter_count: usize,
    weights: Vec<f32>,
}

impl MockStudentModel {
    /// Create a new mock student model.
    #[must_use]
    pub fn new(hidden_size: usize) -> Self {
        let num_layers = 6;
        let num_heads = 6;
        let parameter_count = hidden_size * hidden_size * 4 * num_layers;
        let weights = vec![1.0; parameter_count];

        Self {
            hidden_size,
            num_layers,
            num_heads,
            parameter_count,
            weights,
        }
    }

    /// Create a mock student with custom layer count.
    #[must_use]
    pub fn with_layers(hidden_size: usize, num_layers: usize, num_heads: usize) -> Self {
        let parameter_count = hidden_size * hidden_size * 4 * num_layers;
        let weights = vec![1.0; parameter_count];

        Self {
            hidden_size,
            num_layers,
            num_heads,
            parameter_count,
            weights,
        }
    }

    /// Get the hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get the current weights.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

impl TeacherModel for MockStudentModel {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Simple mock: return scaled input (slightly different from teacher)
        let mut output = vec![0.0; self.hidden_size.min(input.len().max(1))];
        for (i, val) in output.iter_mut().enumerate() {
            if i < input.len() {
                // Use first weight as a scaling factor
                let scale = if self.weights.is_empty() {
                    1.0
                } else {
                    self.weights[0]
                };
                *val = input[i] * 1.2 * scale + 0.05;
            }
        }
        output
    }

    fn get_hidden_states(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut states = Vec::with_capacity(self.num_layers);
        for layer in 0..self.num_layers {
            let mut state = vec![0.0; self.hidden_size.min(input.len().max(1))];
            for (i, val) in state.iter_mut().enumerate() {
                if i < input.len() {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        *val = input[i] * (1.0 + 0.08 * layer as f32);
                    }
                }
            }
            states.push(state);
        }
        states
    }

    fn get_attention_weights(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let seq_len = input.len().max(1);
        let mut weights = Vec::with_capacity(self.num_heads);
        for _head in 0..self.num_heads {
            #[allow(clippy::cast_precision_loss)]
            let weight = vec![1.0 / seq_len as f32; seq_len * seq_len];
            weights.push(weight);
        }
        weights
    }

    fn model_size(&self) -> usize {
        self.parameter_count
    }
}

impl StudentModel for MockStudentModel {
    fn update_weights(&mut self, gradients: &[f32]) {
        // Simple mock weight update
        for (i, grad) in gradients.iter().enumerate() {
            if i < self.weights.len() {
                self.weights[i] -= grad;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distillation_step_config_default() {
        let config = DistillationStepConfig::default();
        assert!((config.learning_rate - 0.001).abs() < f32::EPSILON);
        assert!((config.temperature - 3.0).abs() < f32::EPSILON);
        assert!(config.is_valid());
    }

    #[test]
    fn test_distillation_step_config_builder() {
        let config = DistillationStepConfig::default()
            .with_learning_rate(0.01)
            .with_temperature(5.0)
            .with_soft_target_weight(0.8)
            .with_hard_target_weight(0.2);

        assert!((config.learning_rate - 0.01).abs() < f32::EPSILON);
        assert!((config.temperature - 5.0).abs() < f32::EPSILON);
        assert!((config.soft_target_weight - 0.8).abs() < f32::EPSILON);
        assert!((config.hard_target_weight - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distillation_step_config_clamping() {
        let config = DistillationStepConfig::default()
            .with_soft_target_weight(1.5)
            .with_temperature(0.0);

        assert!((config.soft_target_weight - 1.0).abs() < f32::EPSILON);
        assert!((config.temperature - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distillation_step_result_creation() {
        let config = DistillationStepConfig::default();
        let result = DistillationStepResult::new(0.5, 0.3, 0.1, 0.05, vec![0.1, 0.2], &config);

        assert!(result.is_valid());
        assert!(result.total_loss > 0.0);
    }

    #[test]
    fn test_distillation_step_result_summary() {
        let result = DistillationStepResult {
            total_loss: 1.0,
            soft_target_loss: 0.5,
            hard_target_loss: 0.3,
            hidden_state_loss: 0.1,
            attention_loss: 0.05,
            gradients: vec![],
        };

        let summary = result.summary();
        assert!(summary.contains("total="));
        assert!(summary.contains("soft="));
    }

    #[test]
    fn test_distillation_metrics_creation() {
        let metrics = DistillationMetrics::new(0.95, 0.1, 1_000_000, 100_000, 100);

        assert!((metrics.accuracy - 0.95).abs() < f32::EPSILON);
        assert!((metrics.average_loss - 0.1).abs() < f32::EPSILON);
        assert!((metrics.compression_ratio - 10.0).abs() < f32::EPSILON);
        assert_eq!(metrics.samples_evaluated, 100);
    }

    #[test]
    fn test_distillation_metrics_memory_savings() {
        let metrics = DistillationMetrics::new(0.9, 0.1, 1000, 100, 10);

        let savings = metrics.memory_savings_percent();
        assert!((savings - 90.0).abs() < 1.0);
    }

    #[test]
    fn test_distillation_metrics_is_successful() {
        let metrics = DistillationMetrics::new(0.9, 0.1, 1000, 100, 10);

        assert!(metrics.is_successful(0.8, 0.5));
        assert!(!metrics.is_successful(0.95, 0.5));
        assert!(!metrics.is_successful(0.8, 0.05));
    }

    #[test]
    fn test_mock_teacher_model_creation() {
        let teacher = MockTeacherModel::new(768);

        assert_eq!(teacher.hidden_size(), 768);
        assert_eq!(teacher.num_layers(), 12);
        assert!(teacher.model_size() > 0);
    }

    #[test]
    fn test_mock_teacher_model_forward() {
        let teacher = MockTeacherModel::new(768);
        let input = vec![0.1, 0.2, 0.3];

        let output = teacher.forward(&input);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_mock_teacher_model_hidden_states() {
        let teacher = MockTeacherModel::new(768);
        let input = vec![0.1, 0.2, 0.3];

        let states = teacher.get_hidden_states(&input);
        assert_eq!(states.len(), teacher.num_layers());
    }

    #[test]
    fn test_mock_teacher_model_attention_weights() {
        let teacher = MockTeacherModel::new(768);
        let input = vec![0.1, 0.2, 0.3];

        let weights = teacher.get_attention_weights(&input);
        assert_eq!(weights.len(), teacher.num_heads());
    }

    #[test]
    fn test_mock_student_model_creation() {
        let student = MockStudentModel::new(384);

        assert_eq!(student.hidden_size(), 384);
        assert_eq!(student.num_layers(), 6);
        assert!(student.model_size() > 0);
    }

    #[test]
    fn test_mock_student_model_forward() {
        let student = MockStudentModel::new(384);
        let input = vec![0.1, 0.2, 0.3];

        let output = student.forward(&input);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_mock_student_model_update_weights() {
        let mut student = MockStudentModel::new(384);
        let original_weight = student.weights()[0];

        student.update_weights(&[0.1, 0.2, 0.3]);

        assert!((student.weights()[0] - (original_weight - 0.1)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distillation_pair_creation() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);

        let pair = DistillationPair::new(teacher, student);

        assert_eq!(pair.teacher().hidden_size(), 768);
        assert_eq!(pair.student().hidden_size(), 384);
    }

    #[test]
    fn test_distillation_pair_compression_ratio() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);
        let pair = DistillationPair::new(teacher, student);

        let ratio = pair.compression_ratio();
        assert!(ratio > 1.0);
    }

    #[test]
    fn test_distillation_pair_distill_step() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);
        let mut pair = DistillationPair::new(teacher, student);

        let config = DistillationStepConfig::default();
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let result = pair.distill_step(&input, &config);

        assert!(result.is_valid());
        assert!(result.total_loss >= 0.0);
    }

    #[test]
    fn test_distillation_pair_evaluate() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);
        let pair = DistillationPair::new(teacher, student);

        let inputs = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let metrics = pair.evaluate(&inputs);

        assert_eq!(metrics.samples_evaluated, 3);
        assert!(metrics.compression_ratio > 0.0);
    }

    #[test]
    fn test_distillation_pair_evaluate_empty() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);
        let pair = DistillationPair::new(teacher, student);

        let inputs: Vec<Vec<f32>> = vec![];
        let metrics = pair.evaluate(&inputs);

        assert_eq!(metrics.samples_evaluated, 0);
    }

    #[test]
    fn test_softmax_with_temperature() {
        let logits = vec![1.0, 2.0, 3.0];

        let soft_low_temp =
            DistillationPair::<MockTeacherModel, MockStudentModel>::softmax_with_temperature(
                &logits, 0.5,
            );
        let soft_high_temp =
            DistillationPair::<MockTeacherModel, MockStudentModel>::softmax_with_temperature(
                &logits, 5.0,
            );

        // Lower temperature should produce more peaked distribution
        let max_low = soft_low_temp
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let max_high = soft_high_temp
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max_low > max_high);

        // Sum should be approximately 1.0
        let sum_low: f32 = soft_low_temp.iter().sum();
        let sum_high: f32 = soft_high_temp.iter().sum();
        assert!((sum_low - 1.0).abs() < 0.01);
        assert!((sum_high - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_mse() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let mse = DistillationPair::<MockTeacherModel, MockStudentModel>::compute_mse(&a, &b);
        assert!(mse.abs() < f32::EPSILON);

        let c = vec![2.0, 3.0, 4.0];
        let mse2 = DistillationPair::<MockTeacherModel, MockStudentModel>::compute_mse(&a, &c);
        assert!((mse2 - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_argmax() {
        let values = vec![0.1, 0.5, 0.3, 0.2];
        let idx = DistillationPair::<MockTeacherModel, MockStudentModel>::argmax(&values);
        assert_eq!(idx, 1);

        let empty: Vec<f32> = vec![];
        let idx_empty = DistillationPair::<MockTeacherModel, MockStudentModel>::argmax(&empty);
        assert_eq!(idx_empty, 0);
    }

    #[test]
    fn test_kl_divergence() {
        let teacher = vec![0.1, 0.2, 0.7];
        let student = vec![0.1, 0.2, 0.7];

        let kl = DistillationPair::<MockTeacherModel, MockStudentModel>::compute_kl_divergence(
            &teacher, &student, 1.0,
        );
        assert!(kl.abs() < 0.1); // Should be close to 0 for identical distributions

        let different = vec![0.7, 0.2, 0.1];
        let kl2 = DistillationPair::<MockTeacherModel, MockStudentModel>::compute_kl_divergence(
            &teacher, &different, 1.0,
        );
        assert!(kl2 > kl); // KL should be higher for different distributions
    }

    #[test]
    fn test_teacher_model_with_layers() {
        let teacher = MockTeacherModel::with_layers(512, 8, 8);
        assert_eq!(teacher.hidden_size(), 512);
        assert_eq!(teacher.num_layers(), 8);
        assert_eq!(teacher.num_heads(), 8);
    }

    #[test]
    fn test_student_model_with_layers() {
        let student = MockStudentModel::with_layers(256, 4, 4);
        assert_eq!(student.hidden_size(), 256);
        assert_eq!(student.num_layers(), 4);
        assert_eq!(student.num_heads(), 4);
    }

    #[test]
    fn test_distillation_step_config_total_weight() {
        let config = DistillationStepConfig::default();
        let total = config.total_weight();
        assert!(total > 0.0);
    }

    #[test]
    fn test_distillation_metrics_summary() {
        let metrics = DistillationMetrics::new(0.9, 0.15, 1000, 100, 50);
        let summary = metrics.summary();
        assert!(summary.contains("accuracy="));
        assert!(summary.contains("compression="));
    }

    #[test]
    fn test_distillation_pair_student_mut() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);
        let mut pair = DistillationPair::new(teacher, student);

        let student_mut = pair.student_mut();
        student_mut.update_weights(&[0.1]);

        assert!((pair.student().weights()[0] - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_empty_input_handling() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);

        let empty: Vec<f32> = vec![];
        let output = teacher.forward(&empty);
        assert!(output.is_empty() || output.len() <= 1);

        let states = teacher.get_hidden_states(&empty);
        assert_eq!(states.len(), teacher.num_layers());

        let student_output = student.forward(&empty);
        assert!(student_output.is_empty() || student_output.len() <= 1);
    }

    #[test]
    fn test_distillation_step_with_empty_input() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);
        let mut pair = DistillationPair::new(teacher, student);

        let config = DistillationStepConfig::default();
        let empty: Vec<f32> = vec![];

        let result = pair.distill_step(&empty, &config);
        assert!(result.total_loss.is_finite());
    }

    #[test]
    fn test_multiple_distillation_steps() {
        let teacher = MockTeacherModel::new(768);
        let student = MockStudentModel::new(384);
        let mut pair = DistillationPair::new(teacher, student);

        let config = DistillationStepConfig::default();
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let mut losses = Vec::new();
        for _ in 0..5 {
            let result = pair.distill_step(&input, &config);
            losses.push(result.total_loss);
        }

        // All losses should be finite
        assert!(losses.iter().all(|l| l.is_finite()));
    }

    #[test]
    fn test_cross_entropy_computation() {
        let target = vec![0.0, 1.0, 0.0];
        let prediction = vec![0.1, 0.8, 0.1];

        let loss = DistillationPair::<MockTeacherModel, MockStudentModel>::compute_cross_entropy(
            &target,
            &prediction,
        );
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_hidden_state_loss_computation() {
        let teacher_states = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let student_states = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

        let loss =
            DistillationPair::<MockTeacherModel, MockStudentModel>::compute_hidden_state_loss(
                &teacher_states,
                &student_states,
            );
        assert!(loss.abs() < f32::EPSILON);

        let different_states = vec![vec![0.5, 0.6], vec![0.7, 0.8]];
        let loss2 =
            DistillationPair::<MockTeacherModel, MockStudentModel>::compute_hidden_state_loss(
                &teacher_states,
                &different_states,
            );
        assert!(loss2 > 0.0);
    }

    #[test]
    fn test_attention_loss_computation() {
        let teacher_attn = vec![vec![0.25, 0.25, 0.25, 0.25]];
        let student_attn = vec![vec![0.25, 0.25, 0.25, 0.25]];

        let loss = DistillationPair::<MockTeacherModel, MockStudentModel>::compute_attention_loss(
            &teacher_attn,
            &student_attn,
        );
        assert!(loss.abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_computation() {
        let student_output = vec![0.1, 0.2, 0.7];
        let teacher_output = vec![0.2, 0.3, 0.5];

        let gradients = DistillationPair::<MockTeacherModel, MockStudentModel>::compute_gradients(
            &student_output,
            &teacher_output,
            0.1,
            1.0,
        );

        assert_eq!(gradients.len(), 3);
        assert!(gradients.iter().all(|g| g.is_finite()));
    }
}
