//! Distillation evaluator and model evaluation traits.

use super::eval::{ComparisonResult, EvaluationResult, TestExample};

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
