//! Evaluation result types for distillation metrics.

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
