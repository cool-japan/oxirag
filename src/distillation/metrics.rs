//! Distillation metrics and evaluation module.
//!
//! This module provides types and functionality for evaluating distilled models,
//! comparing teacher and student model performance, and tracking training metrics.

use serde::{Deserialize, Serialize};

/// A test example for model evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExample {
    /// Input query or prompt.
    pub input: String,
    /// Expected output or reference answer.
    pub expected_output: String,
    /// Optional metadata for the example.
    pub metadata: Option<TestExampleMetadata>,
}

/// Metadata for a test example.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestExampleMetadata {
    /// Category or domain of the example.
    pub category: Option<String>,
    /// Difficulty level (0.0 - 1.0).
    pub difficulty: Option<f32>,
    /// Source of the example.
    pub source: Option<String>,
}

impl TestExample {
    /// Create a new test example.
    #[must_use]
    pub fn new(input: impl Into<String>, expected_output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            expected_output: expected_output.into(),
            metadata: None,
        }
    }

    /// Create a test example with metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: TestExampleMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the category for this example.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        let meta = self
            .metadata
            .get_or_insert_with(TestExampleMetadata::default);
        meta.category = Some(category.into());
        self
    }

    /// Set the difficulty for this example.
    #[must_use]
    pub fn with_difficulty(mut self, difficulty: f32) -> Self {
        let meta = self
            .metadata
            .get_or_insert_with(TestExampleMetadata::default);
        meta.difficulty = Some(difficulty.clamp(0.0, 1.0));
        self
    }
}

/// Result of evaluating a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Accuracy score (0.0 - 1.0).
    pub accuracy: f32,
    /// Precision score (0.0 - 1.0).
    pub precision: f32,
    /// Recall score (0.0 - 1.0).
    pub recall: f32,
    /// F1 score (harmonic mean of precision and recall).
    pub f1_score: f32,
    /// Average latency in milliseconds.
    pub latency_ms: f64,
    /// Throughput in examples per second.
    pub throughput: f64,
    /// Model size in bytes.
    pub model_size_bytes: u64,
    /// Peak memory usage in bytes.
    pub memory_usage: u64,
}

impl EvaluationResult {
    /// Create a new evaluation result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an evaluation result with accuracy metrics.
    #[must_use]
    pub fn with_accuracy_metrics(accuracy: f32, precision: f32, recall: f32) -> Self {
        let f1_score = Self::calculate_f1(precision, recall);
        Self {
            accuracy,
            precision,
            recall,
            f1_score,
            ..Default::default()
        }
    }

    /// Calculate F1 score from precision and recall.
    #[must_use]
    pub fn calculate_f1(precision: f32, recall: f32) -> f32 {
        if precision + recall <= 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// Set latency metrics.
    #[must_use]
    pub fn with_latency(mut self, latency_ms: f64, throughput: f64) -> Self {
        self.latency_ms = latency_ms;
        self.throughput = throughput;
        self
    }

    /// Set memory metrics.
    #[must_use]
    pub fn with_memory(mut self, model_size_bytes: u64, memory_usage: u64) -> Self {
        self.model_size_bytes = model_size_bytes;
        self.memory_usage = memory_usage;
        self
    }

    /// Check if the model meets quality thresholds.
    #[must_use]
    pub fn meets_threshold(&self, min_accuracy: f32, max_latency_ms: f64) -> bool {
        self.accuracy >= min_accuracy && self.latency_ms <= max_latency_ms
    }

    /// Calculate overall quality score (weighted combination of metrics).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn overall_quality_score(&self) -> f64 {
        // Weighted combination: accuracy (40%), F1 (30%), normalized latency (30%)
        let accuracy_weight = 0.4;
        let f1_weight = 0.3;
        let latency_weight = 0.3;

        let normalized_latency = if self.latency_ms > 0.0 {
            (1000.0 / self.latency_ms).min(1.0)
        } else {
            1.0
        };

        f64::from(self.accuracy) * accuracy_weight
            + f64::from(self.f1_score) * f1_weight
            + normalized_latency * latency_weight
    }
}

/// Result of comparing teacher and student models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Accuracy retention: student accuracy / teacher accuracy.
    pub accuracy_retention: f64,
    /// Speedup ratio: teacher latency / student latency.
    pub speedup_ratio: f64,
    /// Compression ratio: teacher size / student size.
    pub compression_ratio: f64,
    /// Overall quality score.
    pub quality_score: f64,
    /// Teacher evaluation result.
    pub teacher_result: EvaluationResult,
    /// Student evaluation result.
    pub student_result: EvaluationResult,
}

impl ComparisonResult {
    /// Create a new comparison result from teacher and student evaluations.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_evaluations(teacher: EvaluationResult, student: EvaluationResult) -> Self {
        let accuracy_retention = if teacher.accuracy > 0.0 {
            f64::from(student.accuracy) / f64::from(teacher.accuracy)
        } else {
            0.0
        };

        let speedup_ratio = if student.latency_ms > 0.0 {
            teacher.latency_ms / student.latency_ms
        } else {
            0.0
        };

        let compression_ratio = if student.model_size_bytes > 0 {
            teacher.model_size_bytes as f64 / student.model_size_bytes as f64
        } else {
            0.0
        };

        let quality_score =
            Self::calculate_quality_score(accuracy_retention, speedup_ratio, compression_ratio);

        Self {
            accuracy_retention,
            speedup_ratio,
            compression_ratio,
            quality_score,
            teacher_result: teacher,
            student_result: student,
        }
    }

    /// Calculate overall quality score from component metrics.
    #[must_use]
    fn calculate_quality_score(
        accuracy_retention: f64,
        speedup_ratio: f64,
        compression_ratio: f64,
    ) -> f64 {
        // Quality score combines retention, speedup, and compression
        // Higher is better for all three
        let retention_weight = 0.5;
        let speedup_weight = 0.3;
        let compression_weight = 0.2;

        // Normalize speedup and compression (cap at reasonable values)
        let normalized_speedup = speedup_ratio.min(10.0) / 10.0;
        let normalized_compression = compression_ratio.min(20.0) / 20.0;

        accuracy_retention * retention_weight
            + normalized_speedup * speedup_weight
            + normalized_compression * compression_weight
    }

    /// Check if distillation was successful based on thresholds.
    #[must_use]
    pub fn is_successful(&self, min_accuracy_retention: f64, min_speedup: f64) -> bool {
        self.accuracy_retention >= min_accuracy_retention && self.speedup_ratio >= min_speedup
    }

    /// Get a summary of the comparison.
    #[must_use]
    pub fn summary(&self) -> ComparisonSummary {
        ComparisonSummary {
            accuracy_retention_percent: self.accuracy_retention * 100.0,
            speedup_factor: self.speedup_ratio,
            compression_factor: self.compression_ratio,
            quality_score: self.quality_score,
            student_faster: self.speedup_ratio > 1.0,
            student_smaller: self.compression_ratio > 1.0,
        }
    }
}

/// A summary of the comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Accuracy retention as a percentage.
    pub accuracy_retention_percent: f64,
    /// How many times faster the student is.
    pub speedup_factor: f64,
    /// How many times smaller the student is.
    pub compression_factor: f64,
    /// Overall quality score.
    pub quality_score: f64,
    /// Whether the student is faster than the teacher.
    pub student_faster: bool,
    /// Whether the student is smaller than the teacher.
    pub student_smaller: bool,
}

/// Layer-wise similarity measurement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerSimilarity {
    /// Layer index (0-indexed).
    pub layer_index: usize,
    /// Layer name or identifier.
    pub layer_name: String,
    /// Cosine similarity between teacher and student layer outputs.
    pub cosine_similarity: f32,
    /// Mean squared error between layer outputs.
    pub mse: f32,
    /// Pearson correlation coefficient.
    pub correlation: f32,
}

impl LayerSimilarity {
    /// Create a new layer similarity measurement.
    #[must_use]
    pub fn new(layer_index: usize, layer_name: impl Into<String>) -> Self {
        Self {
            layer_index,
            layer_name: layer_name.into(),
            ..Default::default()
        }
    }

    /// Set similarity metrics.
    #[must_use]
    pub fn with_metrics(mut self, cosine_similarity: f32, mse: f32, correlation: f32) -> Self {
        self.cosine_similarity = cosine_similarity;
        self.mse = mse;
        self.correlation = correlation;
        self
    }

    /// Calculate an aggregate similarity score.
    #[must_use]
    pub fn aggregate_score(&self) -> f32 {
        // Combine cosine similarity and correlation (both higher is better)
        // MSE is inverted since lower is better
        let mse_score = 1.0 / (1.0 + self.mse);
        (self.cosine_similarity + self.correlation + mse_score) / 3.0
    }
}

/// Metrics for measuring knowledge transfer efficiency.
///
/// This type provides detailed metrics about the knowledge transfer process
/// during distillation, including layer-wise similarity measurements.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeTransferMetrics {
    /// Knowledge transfer efficiency (0.0 - 1.0).
    /// Measures how well the student learned from the teacher.
    pub knowledge_transfer_efficiency: f32,
    /// Capacity utilization (0.0 - 1.0).
    /// Measures how much of the student's capacity is being used.
    pub capacity_utilization: f32,
    /// Layer-wise similarity measurements.
    pub layer_wise_similarity: Vec<LayerSimilarity>,
    /// Average layer similarity.
    pub average_layer_similarity: f32,
    /// Training epochs completed.
    pub epochs_completed: usize,
    /// Final training loss.
    pub final_loss: f32,
}

impl KnowledgeTransferMetrics {
    /// Create new knowledge transfer metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set transfer efficiency.
    #[must_use]
    pub fn with_transfer_efficiency(mut self, efficiency: f32) -> Self {
        self.knowledge_transfer_efficiency = efficiency.clamp(0.0, 1.0);
        self
    }

    /// Set capacity utilization.
    #[must_use]
    pub fn with_capacity_utilization(mut self, utilization: f32) -> Self {
        self.capacity_utilization = utilization.clamp(0.0, 1.0);
        self
    }

    /// Set layer-wise similarity measurements.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn with_layer_similarity(mut self, similarities: Vec<LayerSimilarity>) -> Self {
        if similarities.is_empty() {
            self.average_layer_similarity = 0.0;
        } else {
            let sum: f32 = similarities
                .iter()
                .map(LayerSimilarity::aggregate_score)
                .sum();
            self.average_layer_similarity = sum / similarities.len() as f32;
        }
        self.layer_wise_similarity = similarities;
        self
    }

    /// Add a layer similarity measurement.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_layer_similarity(&mut self, similarity: LayerSimilarity) {
        self.layer_wise_similarity.push(similarity);
        // Recalculate average
        let sum: f32 = self
            .layer_wise_similarity
            .iter()
            .map(LayerSimilarity::aggregate_score)
            .sum();
        self.average_layer_similarity = sum / self.layer_wise_similarity.len() as f32;
    }

    /// Get the overall distillation quality score.
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        // Weighted combination of metrics
        let transfer_weight = 0.4;
        let capacity_weight = 0.3;
        let similarity_weight = 0.3;

        self.knowledge_transfer_efficiency * transfer_weight
            + self.capacity_utilization * capacity_weight
            + self.average_layer_similarity * similarity_weight
    }

    /// Check if distillation quality meets thresholds.
    #[must_use]
    pub fn meets_quality_threshold(&self, min_score: f32) -> bool {
        self.overall_score() >= min_score
    }
}

/// Metrics for a single training epoch with comprehensive tracking.
///
/// This type provides detailed per-epoch metrics for tracking training progress
/// during distillation, including loss, accuracy, and timing information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingEpochMetrics {
    /// Epoch number (1-indexed).
    pub epoch: usize,
    /// Training loss.
    pub train_loss: f32,
    /// Validation loss.
    pub val_loss: f32,
    /// Training accuracy.
    pub train_accuracy: f32,
    /// Validation accuracy.
    pub val_accuracy: f32,
    /// Learning rate used in this epoch.
    pub learning_rate: f64,
    /// Temperature used for distillation.
    pub temperature: f32,
    /// Duration of the epoch in seconds.
    pub duration_secs: f64,
    /// Optional additional metrics.
    pub extra_metrics: Option<ExtraEpochMetrics>,
}

/// Additional epoch metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtraEpochMetrics {
    /// Gradient norm.
    pub gradient_norm: Option<f32>,
    /// Number of training samples processed.
    pub samples_processed: Option<usize>,
    /// Memory usage in bytes.
    pub memory_usage: Option<u64>,
}

impl TrainingEpochMetrics {
    /// Create new epoch metrics.
    #[must_use]
    pub fn new(epoch: usize) -> Self {
        Self {
            epoch,
            ..Default::default()
        }
    }

    /// Set loss values.
    #[must_use]
    pub fn with_loss(mut self, train_loss: f32, val_loss: f32) -> Self {
        self.train_loss = train_loss;
        self.val_loss = val_loss;
        self
    }

    /// Set accuracy values.
    #[must_use]
    pub fn with_accuracy(mut self, train_accuracy: f32, val_accuracy: f32) -> Self {
        self.train_accuracy = train_accuracy;
        self.val_accuracy = val_accuracy;
        self
    }

    /// Set training parameters.
    #[must_use]
    pub fn with_params(mut self, learning_rate: f64, temperature: f32) -> Self {
        self.learning_rate = learning_rate;
        self.temperature = temperature;
        self
    }

    /// Set duration.
    #[must_use]
    pub fn with_duration(mut self, duration_secs: f64) -> Self {
        self.duration_secs = duration_secs;
        self
    }

    /// Check if this epoch shows improvement over a previous epoch.
    #[must_use]
    pub fn improved_over(&self, other: &Self) -> bool {
        self.val_loss < other.val_loss
    }

    /// Check if the model is overfitting (train loss much lower than val loss).
    #[must_use]
    pub fn is_overfitting(&self, threshold: f32) -> bool {
        self.train_loss > 0.0 && (self.val_loss - self.train_loss) / self.train_loss > threshold
    }
}

/// Data for plotting training progress.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlotData {
    /// Epoch numbers.
    pub epochs: Vec<usize>,
    /// Training loss values.
    pub train_loss: Vec<f32>,
    /// Validation loss values.
    pub val_loss: Vec<f32>,
    /// Training accuracy values.
    pub train_accuracy: Vec<f32>,
    /// Validation accuracy values.
    pub val_accuracy: Vec<f32>,
    /// Learning rate values.
    pub learning_rate: Vec<f64>,
}

impl PlotData {
    /// Create new plot data.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create plot data from epoch metrics.
    #[must_use]
    pub fn from_metrics(metrics: &[TrainingEpochMetrics]) -> Self {
        let mut plot_data = Self::new();
        for m in metrics {
            plot_data.add_epoch(m);
        }
        plot_data
    }

    /// Add an epoch to the plot data.
    pub fn add_epoch(&mut self, metrics: &TrainingEpochMetrics) {
        self.epochs.push(metrics.epoch);
        self.train_loss.push(metrics.train_loss);
        self.val_loss.push(metrics.val_loss);
        self.train_accuracy.push(metrics.train_accuracy);
        self.val_accuracy.push(metrics.val_accuracy);
        self.learning_rate.push(metrics.learning_rate);
    }

    /// Get the number of epochs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.epochs.len()
    }

    /// Check if the plot data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }

    /// Get min and max values for loss.
    #[must_use]
    pub fn loss_range(&self) -> Option<(f32, f32)> {
        let all_loss: Vec<f32> = self
            .train_loss
            .iter()
            .chain(self.val_loss.iter())
            .copied()
            .collect();

        if all_loss.is_empty() {
            return None;
        }

        let min = all_loss.iter().copied().fold(f32::INFINITY, f32::min);
        let max = all_loss.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Some((min, max))
    }
}

/// Tracks metrics across training epochs.
#[derive(Debug, Clone, Default)]
pub struct MetricsTracker {
    /// History of epoch metrics.
    history: Vec<TrainingEpochMetrics>,
    /// Best epoch index based on validation loss.
    best_epoch_idx: Option<usize>,
    /// Best validation loss seen.
    best_val_loss: f32,
}

impl MetricsTracker {
    /// Create a new metrics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            best_epoch_idx: None,
            best_val_loss: f32::INFINITY,
        }
    }

    /// Record metrics for an epoch.
    pub fn record_epoch(&mut self, metrics: TrainingEpochMetrics) {
        // Check if this is the best epoch
        if metrics.val_loss < self.best_val_loss {
            self.best_val_loss = metrics.val_loss;
            self.best_epoch_idx = Some(self.history.len());
        }

        self.history.push(metrics);
    }

    /// Get the history of all epoch metrics.
    #[must_use]
    pub fn get_history(&self) -> Vec<TrainingEpochMetrics> {
        self.history.clone()
    }

    /// Get a reference to the history.
    #[must_use]
    pub fn history(&self) -> &[TrainingEpochMetrics] {
        &self.history
    }

    /// Get the best epoch (epoch number and metrics).
    #[must_use]
    pub fn best_epoch(&self) -> Option<(usize, TrainingEpochMetrics)> {
        self.best_epoch_idx
            .and_then(|idx| self.history.get(idx).map(|m| (m.epoch, m.clone())))
    }

    /// Get the best epoch index.
    #[must_use]
    pub fn best_epoch_index(&self) -> Option<usize> {
        self.best_epoch_idx
    }

    /// Get data formatted for plotting.
    #[must_use]
    pub fn plot_data(&self) -> PlotData {
        PlotData::from_metrics(&self.history)
    }

    /// Get the number of epochs recorded.
    #[must_use]
    pub fn epoch_count(&self) -> usize {
        self.history.len()
    }

    /// Get the latest epoch metrics.
    #[must_use]
    pub fn latest(&self) -> Option<&TrainingEpochMetrics> {
        self.history.last()
    }

    /// Check if training has converged (validation loss not improving).
    #[must_use]
    pub fn has_converged(&self, patience: usize) -> bool {
        if self.history.len() < patience {
            return false;
        }

        let recent = &self.history[self.history.len() - patience..];
        let first_val_loss = recent.first().map_or(f32::INFINITY, |m| m.val_loss);

        // Converged if no improvement in the last `patience` epochs
        recent.iter().all(|m| m.val_loss >= first_val_loss)
    }

    /// Calculate average training time per epoch.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_epoch_duration(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        let total: f64 = self.history.iter().map(|m| m.duration_secs).sum();
        total / self.history.len() as f64
    }

    /// Get the improvement rate (average decrease in validation loss per epoch).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn improvement_rate(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let first_loss = self.history.first().map_or(0.0, |m| m.val_loss);
        let last_loss = self.history.last().map_or(0.0, |m| m.val_loss);
        let epochs = self.history.len() as f32 - 1.0;

        (first_loss - last_loss) / epochs
    }

    /// Clear all recorded metrics.
    pub fn clear(&mut self) {
        self.history.clear();
        self.best_epoch_idx = None;
        self.best_val_loss = f32::INFINITY;
    }

    /// Get summary statistics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn summary(&self) -> TrackerSummary {
        let avg_train_loss = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().map(|m| m.train_loss).sum::<f32>() / self.history.len() as f32
        };

        let avg_val_loss = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().map(|m| m.val_loss).sum::<f32>() / self.history.len() as f32
        };

        TrackerSummary {
            total_epochs: self.history.len(),
            best_epoch: self
                .best_epoch_idx
                .and_then(|idx| self.history.get(idx).map(|m| m.epoch)),
            best_val_loss: self.best_val_loss,
            final_val_loss: self.history.last().map(|m| m.val_loss),
            avg_train_loss,
            avg_val_loss,
            total_duration_secs: self.history.iter().map(|m| m.duration_secs).sum(),
        }
    }
}

/// Summary of the metrics tracker state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrackerSummary {
    /// Total number of epochs recorded.
    pub total_epochs: usize,
    /// Best epoch number.
    pub best_epoch: Option<usize>,
    /// Best validation loss achieved.
    pub best_val_loss: f32,
    /// Final validation loss.
    pub final_val_loss: Option<f32>,
    /// Average training loss across all epochs.
    pub avg_train_loss: f32,
    /// Average validation loss across all epochs.
    pub avg_val_loss: f32,
    /// Total training duration in seconds.
    pub total_duration_secs: f64,
}

/// Trait for student models that can be evaluated for text generation.
///
/// This trait provides a text-generation interface for student models,
/// distinct from the vector-based interface in `teacher_student` module.
pub trait EvalStudentModel: Send + Sync {
    /// Generate output for a given input.
    fn generate(&self, input: &str) -> String;

    /// Get the model size in bytes.
    fn model_size_bytes(&self) -> u64;

    /// Get model name or identifier.
    fn name(&self) -> &str;
}

/// Trait for teacher models that can be evaluated for text generation.
///
/// This trait provides a text-generation interface for teacher models,
/// distinct from the vector-based interface in `teacher_student` module.
pub trait EvalTeacherModel: Send + Sync {
    /// Generate output for a given input.
    fn generate(&self, input: &str) -> String;

    /// Get the model size in bytes.
    fn model_size_bytes(&self) -> u64;

    /// Get model name or identifier.
    fn name(&self) -> &str;
}

/// Evaluator for distilled models.
#[derive(Debug, Clone, Default)]
pub struct DistillationEvaluator {
    /// Configuration for evaluation.
    config: EvaluatorConfig,
}

/// Configuration for the evaluator.
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Minimum samples required for evaluation.
    pub min_samples: usize,
    /// Timeout per sample in milliseconds.
    pub timeout_ms: u64,
    /// Whether to measure memory usage.
    pub measure_memory: bool,
    /// Similarity threshold for considering outputs as matching.
    pub similarity_threshold: f32,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            min_samples: 10,
            timeout_ms: 5000,
            measure_memory: true,
            similarity_threshold: 0.8,
        }
    }
}

impl DistillationEvaluator {
    /// Create a new evaluator with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an evaluator with custom configuration.
    #[must_use]
    pub fn with_config(config: EvaluatorConfig) -> Self {
        Self { config }
    }

    /// Get the evaluator configuration.
    #[must_use]
    pub fn config(&self) -> &EvaluatorConfig {
        &self.config
    }

    /// Evaluate a student model on test data.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate_model(
        &self,
        model: &dyn EvalStudentModel,
        test_data: &[TestExample],
    ) -> EvaluationResult {
        if test_data.is_empty() {
            return EvaluationResult::default();
        }

        let mut correct = 0;
        let mut total_latency_ms = 0.0;
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        for example in test_data {
            let start = std::time::Instant::now();
            let output = model.generate(&example.input);
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            total_latency_ms += latency;

            let is_match = self.outputs_match(&output, &example.expected_output);

            if is_match {
                correct += 1;
                true_positives += 1;
            } else if !output.is_empty() {
                false_positives += 1;
            }

            if !is_match && !example.expected_output.is_empty() {
                false_negatives += 1;
            }
        }

        let num_examples = test_data.len() as f32;
        let accuracy = correct as f32 / num_examples;
        let avg_latency = total_latency_ms / test_data.len() as f64;
        let throughput = if avg_latency > 0.0 {
            1000.0 / avg_latency
        } else {
            0.0
        };

        let precision = if true_positives + false_positives > 0 {
            true_positives as f32 / (true_positives + false_positives) as f32
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0 {
            true_positives as f32 / (true_positives + false_negatives) as f32
        } else {
            0.0
        };

        let f1_score = EvaluationResult::calculate_f1(precision, recall);

        EvaluationResult {
            accuracy,
            precision,
            recall,
            f1_score,
            latency_ms: avg_latency,
            throughput,
            model_size_bytes: model.model_size_bytes(),
            memory_usage: 0, // Would need actual memory measurement
        }
    }

    /// Compare a teacher and student model on the same test data.
    #[must_use]
    pub fn compare_models(
        &self,
        teacher: &dyn EvalTeacherModel,
        student: &dyn EvalStudentModel,
        data: &[TestExample],
    ) -> ComparisonResult {
        // Create a wrapper to evaluate teacher model
        let teacher_wrapper = TeacherModelWrapper { model: teacher };
        let teacher_result = self.evaluate_model(&teacher_wrapper, data);
        let student_result = self.evaluate_model(student, data);

        ComparisonResult::from_evaluations(teacher_result, student_result)
    }

    /// Check if two outputs match based on similarity.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    fn outputs_match(&self, output: &str, expected: &str) -> bool {
        if output == expected {
            return true;
        }

        // Normalize and compare
        let normalized_output = output.trim().to_lowercase();
        let normalized_expected = expected.trim().to_lowercase();

        if normalized_output == normalized_expected {
            return true;
        }

        // Calculate simple word overlap similarity
        let output_words: std::collections::HashSet<_> =
            normalized_output.split_whitespace().collect();
        let expected_words: std::collections::HashSet<_> =
            normalized_expected.split_whitespace().collect();

        if output_words.is_empty() || expected_words.is_empty() {
            return false;
        }

        let intersection = output_words.intersection(&expected_words).count();
        let union = output_words.union(&expected_words).count();

        if union == 0 {
            return false;
        }

        let similarity = intersection as f32 / union as f32;
        similarity >= self.config.similarity_threshold
    }
}

/// Internal wrapper to evaluate teacher models using the `StudentModel` trait.
struct TeacherModelWrapper<'a> {
    model: &'a dyn EvalTeacherModel,
}

impl EvalStudentModel for TeacherModelWrapper<'_> {
    fn generate(&self, input: &str) -> String {
        self.model.generate(input)
    }

    fn model_size_bytes(&self) -> u64 {
        self.model.model_size_bytes()
    }

    fn name(&self) -> &str {
        self.model.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock student model for testing
    struct MockStudentModel {
        name: String,
        size_bytes: u64,
        responses: std::collections::HashMap<String, String>,
    }

    impl MockStudentModel {
        fn new(name: &str, size_bytes: u64) -> Self {
            Self {
                name: name.to_string(),
                size_bytes,
                responses: std::collections::HashMap::new(),
            }
        }

        fn with_response(mut self, input: &str, output: &str) -> Self {
            self.responses.insert(input.to_string(), output.to_string());
            self
        }
    }

    impl EvalStudentModel for MockStudentModel {
        fn generate(&self, input: &str) -> String {
            self.responses
                .get(input)
                .cloned()
                .unwrap_or_else(|| "default response".to_string())
        }

        fn model_size_bytes(&self) -> u64 {
            self.size_bytes
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    // Mock teacher model for testing
    struct MockTeacherModel {
        name: String,
        size_bytes: u64,
        responses: std::collections::HashMap<String, String>,
    }

    impl MockTeacherModel {
        fn new(name: &str, size_bytes: u64) -> Self {
            Self {
                name: name.to_string(),
                size_bytes,
                responses: std::collections::HashMap::new(),
            }
        }

        fn with_response(mut self, input: &str, output: &str) -> Self {
            self.responses.insert(input.to_string(), output.to_string());
            self
        }
    }

    impl EvalTeacherModel for MockTeacherModel {
        fn generate(&self, input: &str) -> String {
            self.responses
                .get(input)
                .cloned()
                .unwrap_or_else(|| "teacher response".to_string())
        }

        fn model_size_bytes(&self) -> u64 {
            self.size_bytes
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_test_example_creation() {
        let example = TestExample::new("What is Rust?", "A programming language.");
        assert_eq!(example.input, "What is Rust?");
        assert_eq!(example.expected_output, "A programming language.");
        assert!(example.metadata.is_none());
    }

    #[test]
    fn test_test_example_with_metadata() {
        let example = TestExample::new("input", "output")
            .with_category("test")
            .with_difficulty(0.5);

        assert!(example.metadata.is_some());
        let meta = example.metadata.unwrap();
        assert_eq!(meta.category, Some("test".to_string()));
        assert!((meta.difficulty.unwrap() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_evaluation_result_f1_calculation() {
        let f1 = EvaluationResult::calculate_f1(0.8, 0.6);
        let expected = 2.0 * 0.8 * 0.6 / (0.8 + 0.6);
        assert!((f1 - expected).abs() < f32::EPSILON);

        // Test zero case
        let f1_zero = EvaluationResult::calculate_f1(0.0, 0.0);
        assert!((f1_zero - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_evaluation_result_quality_score() {
        let result = EvaluationResult {
            accuracy: 0.9,
            f1_score: 0.85,
            latency_ms: 100.0,
            ..Default::default()
        };

        let score = result.overall_quality_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_evaluation_result_meets_threshold() {
        let result = EvaluationResult {
            accuracy: 0.9,
            latency_ms: 50.0,
            ..Default::default()
        };

        assert!(result.meets_threshold(0.8, 100.0));
        assert!(!result.meets_threshold(0.95, 100.0));
        assert!(!result.meets_threshold(0.8, 30.0));
    }

    #[test]
    fn test_comparison_result_from_evaluations() {
        let teacher = EvaluationResult {
            accuracy: 0.95,
            latency_ms: 200.0,
            model_size_bytes: 1_000_000,
            ..Default::default()
        };

        let student = EvaluationResult {
            accuracy: 0.90,
            latency_ms: 50.0,
            model_size_bytes: 100_000,
            ..Default::default()
        };

        let comparison = ComparisonResult::from_evaluations(teacher, student);

        // Accuracy retention: 0.90 / 0.95 ~ 0.947
        assert!((comparison.accuracy_retention - 0.947).abs() < 0.01);

        // Speedup: 200 / 50 = 4.0
        assert!((comparison.speedup_ratio - 4.0).abs() < 0.01);

        // Compression: 1_000_000 / 100_000 = 10.0
        assert!((comparison.compression_ratio - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_comparison_result_is_successful() {
        let comparison = ComparisonResult {
            accuracy_retention: 0.95,
            speedup_ratio: 3.0,
            ..Default::default()
        };

        assert!(comparison.is_successful(0.9, 2.0));
        assert!(!comparison.is_successful(0.98, 2.0));
        assert!(!comparison.is_successful(0.9, 5.0));
    }

    #[test]
    fn test_layer_similarity() {
        let layer = LayerSimilarity::new(0, "layer_0").with_metrics(0.9, 0.1, 0.85);

        assert_eq!(layer.layer_index, 0);
        assert!((layer.cosine_similarity - 0.9).abs() < f32::EPSILON);

        let score = layer.aggregate_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_knowledge_transfer_metrics() {
        let metrics = KnowledgeTransferMetrics::new()
            .with_transfer_efficiency(0.9)
            .with_capacity_utilization(0.8);

        assert!((metrics.knowledge_transfer_efficiency - 0.9).abs() < f32::EPSILON);
        assert!((metrics.capacity_utilization - 0.8).abs() < f32::EPSILON);

        let score = metrics.overall_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_knowledge_transfer_metrics_with_layers() {
        let layers = vec![
            LayerSimilarity::new(0, "layer_0").with_metrics(0.9, 0.1, 0.85),
            LayerSimilarity::new(1, "layer_1").with_metrics(0.85, 0.15, 0.8),
        ];

        let metrics = KnowledgeTransferMetrics::new().with_layer_similarity(layers);

        assert_eq!(metrics.layer_wise_similarity.len(), 2);
        assert!(metrics.average_layer_similarity > 0.0);
    }

    #[test]
    fn test_training_epoch_metrics() {
        let metrics = TrainingEpochMetrics::new(1)
            .with_loss(0.5, 0.6)
            .with_accuracy(0.8, 0.75)
            .with_params(0.001, 4.0)
            .with_duration(120.0);

        assert_eq!(metrics.epoch, 1);
        assert!((metrics.train_loss - 0.5).abs() < f32::EPSILON);
        assert!((metrics.val_loss - 0.6).abs() < f32::EPSILON);
        assert!((metrics.train_accuracy - 0.8).abs() < f32::EPSILON);
        assert!((metrics.learning_rate - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_training_epoch_metrics_improved_over() {
        let epoch1 = TrainingEpochMetrics::new(1).with_loss(0.5, 0.6);
        let epoch2 = TrainingEpochMetrics::new(2).with_loss(0.4, 0.55);

        assert!(epoch2.improved_over(&epoch1));
        assert!(!epoch1.improved_over(&epoch2));
    }

    #[test]
    fn test_training_epoch_metrics_overfitting() {
        let overfitting = TrainingEpochMetrics::new(1).with_loss(0.1, 0.5);
        let normal = TrainingEpochMetrics::new(1).with_loss(0.3, 0.35);

        assert!(overfitting.is_overfitting(1.0));
        assert!(!normal.is_overfitting(1.0));
    }

    #[test]
    fn test_metrics_tracker_record_epoch() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(TrainingEpochMetrics::new(1).with_loss(0.5, 0.6));
        tracker.record_epoch(TrainingEpochMetrics::new(2).with_loss(0.4, 0.5));
        tracker.record_epoch(TrainingEpochMetrics::new(3).with_loss(0.45, 0.55));

        assert_eq!(tracker.epoch_count(), 3);

        let best = tracker.best_epoch();
        assert!(best.is_some());
        let (epoch, _) = best.unwrap();
        assert_eq!(epoch, 2); // Epoch 2 has lowest val_loss
    }

    #[test]
    fn test_metrics_tracker_history() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(TrainingEpochMetrics::new(1).with_loss(0.5, 0.6));
        tracker.record_epoch(TrainingEpochMetrics::new(2).with_loss(0.4, 0.5));

        let history = tracker.get_history();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_metrics_tracker_plot_data() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(
            TrainingEpochMetrics::new(1)
                .with_loss(0.5, 0.6)
                .with_accuracy(0.7, 0.65),
        );
        tracker.record_epoch(
            TrainingEpochMetrics::new(2)
                .with_loss(0.4, 0.5)
                .with_accuracy(0.8, 0.75),
        );

        let plot_data = tracker.plot_data();

        assert_eq!(plot_data.len(), 2);
        assert_eq!(plot_data.epochs, vec![1, 2]);
        assert_eq!(plot_data.train_loss.len(), 2);
        assert_eq!(plot_data.val_accuracy.len(), 2);
    }

    #[test]
    fn test_metrics_tracker_convergence() {
        let mut tracker = MetricsTracker::new();

        // Simulate converged training (no improvement)
        tracker.record_epoch(TrainingEpochMetrics::new(1).with_loss(0.5, 0.5));
        tracker.record_epoch(TrainingEpochMetrics::new(2).with_loss(0.45, 0.5));
        tracker.record_epoch(TrainingEpochMetrics::new(3).with_loss(0.4, 0.5));

        assert!(tracker.has_converged(3));
        assert!(!tracker.has_converged(4));
    }

    #[test]
    fn test_metrics_tracker_summary() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(
            TrainingEpochMetrics::new(1)
                .with_loss(0.5, 0.6)
                .with_duration(60.0),
        );
        tracker.record_epoch(
            TrainingEpochMetrics::new(2)
                .with_loss(0.4, 0.5)
                .with_duration(55.0),
        );

        let summary = tracker.summary();

        assert_eq!(summary.total_epochs, 2);
        assert_eq!(summary.best_epoch, Some(2));
        assert!((summary.best_val_loss - 0.5).abs() < f32::EPSILON);
        assert!((summary.total_duration_secs - 115.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_tracker_clear() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(TrainingEpochMetrics::new(1).with_loss(0.5, 0.6));
        tracker.clear();

        assert_eq!(tracker.epoch_count(), 0);
        assert!(tracker.best_epoch().is_none());
    }

    #[test]
    fn test_metrics_tracker_latest() {
        let mut tracker = MetricsTracker::new();

        assert!(tracker.latest().is_none());

        tracker.record_epoch(TrainingEpochMetrics::new(1).with_loss(0.5, 0.6));
        tracker.record_epoch(TrainingEpochMetrics::new(2).with_loss(0.4, 0.5));

        let latest = tracker.latest();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().epoch, 2);
    }

    #[test]
    fn test_metrics_tracker_improvement_rate() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(TrainingEpochMetrics::new(1).with_loss(0.0, 1.0));
        tracker.record_epoch(TrainingEpochMetrics::new(2).with_loss(0.0, 0.8));
        tracker.record_epoch(TrainingEpochMetrics::new(3).with_loss(0.0, 0.6));

        let rate = tracker.improvement_rate();
        // (1.0 - 0.6) / 2 = 0.2
        assert!((rate - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_plot_data_from_metrics() {
        let metrics = vec![
            TrainingEpochMetrics::new(1).with_loss(0.5, 0.6),
            TrainingEpochMetrics::new(2).with_loss(0.4, 0.5),
        ];

        let plot_data = PlotData::from_metrics(&metrics);

        assert_eq!(plot_data.len(), 2);
        assert_eq!(plot_data.epochs, vec![1, 2]);
    }

    #[test]
    fn test_plot_data_loss_range() {
        let mut plot_data = PlotData::new();
        plot_data.add_epoch(&TrainingEpochMetrics::new(1).with_loss(0.5, 0.6));
        plot_data.add_epoch(&TrainingEpochMetrics::new(2).with_loss(0.3, 0.7));

        let range = plot_data.loss_range();
        assert!(range.is_some());
        let (min, max) = range.unwrap();
        assert!((min - 0.3).abs() < f32::EPSILON);
        assert!((max - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distillation_evaluator_creation() {
        let evaluator = DistillationEvaluator::new();
        assert_eq!(evaluator.config().min_samples, 10);
    }

    #[test]
    fn test_distillation_evaluator_custom_config() {
        let config = EvaluatorConfig {
            min_samples: 5,
            timeout_ms: 1000,
            measure_memory: false,
            similarity_threshold: 0.9,
        };

        let evaluator = DistillationEvaluator::with_config(config);
        assert_eq!(evaluator.config().min_samples, 5);
        assert_eq!(evaluator.config().timeout_ms, 1000);
    }

    #[test]
    fn test_distillation_evaluator_evaluate_model() {
        let model = MockStudentModel::new("test-student", 50_000)
            .with_response("What is Rust?", "A programming language.");

        let test_data = vec![TestExample::new("What is Rust?", "A programming language.")];

        let evaluator = DistillationEvaluator::new();
        let result = evaluator.evaluate_model(&model, &test_data);

        assert!((result.accuracy - 1.0).abs() < f32::EPSILON);
        assert_eq!(result.model_size_bytes, 50_000);
    }

    #[test]
    fn test_distillation_evaluator_compare_models() {
        let teacher =
            MockTeacherModel::new("teacher", 1_000_000).with_response("test", "teacher answer");

        let student =
            MockStudentModel::new("student", 100_000).with_response("test", "student answer");

        let test_data = vec![TestExample::new("test", "teacher answer")];

        let evaluator = DistillationEvaluator::new();
        let comparison = evaluator.compare_models(&teacher, &student, &test_data);

        // Teacher should have perfect accuracy, student should be close
        assert!(comparison.teacher_result.accuracy > 0.0);
        assert!(comparison.compression_ratio > 1.0); // Student is smaller
    }

    #[test]
    fn test_distillation_evaluator_empty_data() {
        let model = MockStudentModel::new("test", 1000);
        let test_data: Vec<TestExample> = vec![];

        let evaluator = DistillationEvaluator::new();
        let result = evaluator.evaluate_model(&model, &test_data);

        assert!((result.accuracy - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_comparison_summary() {
        let comparison = ComparisonResult {
            accuracy_retention: 0.95,
            speedup_ratio: 4.0,
            compression_ratio: 10.0,
            quality_score: 0.8,
            ..Default::default()
        };

        let summary = comparison.summary();

        assert!((summary.accuracy_retention_percent - 95.0).abs() < 0.01);
        assert!(summary.student_faster);
        assert!(summary.student_smaller);
    }

    #[test]
    fn test_evaluator_config_default() {
        let config = EvaluatorConfig::default();

        assert_eq!(config.min_samples, 10);
        assert_eq!(config.timeout_ms, 5000);
        assert!(config.measure_memory);
        assert!((config.similarity_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tracker_avg_epoch_duration() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(TrainingEpochMetrics::new(1).with_duration(60.0));
        tracker.record_epoch(TrainingEpochMetrics::new(2).with_duration(50.0));
        tracker.record_epoch(TrainingEpochMetrics::new(3).with_duration(70.0));

        let avg = tracker.avg_epoch_duration();
        assert!((avg - 60.0).abs() < 0.01);
    }
}
