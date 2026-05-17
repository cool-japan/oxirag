//! Tests for the distillation metrics module.

use super::eval::{ComparisonResult, EvaluationResult, TestExample};
use super::evaluator::{
    DistillationEvaluator, EvalStudentModel, EvalTeacherModel, EvaluatorConfig,
};
use super::plot::{PlotData, TrainingEpochMetrics};
use super::tracker::MetricsTracker;
use super::transfer::{KnowledgeTransferMetrics, LayerSimilarity};

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
    let meta = example.metadata.expect("test operation should succeed");
    assert_eq!(meta.category, Some("test".to_string()));
    assert!((meta.difficulty.expect("test operation should succeed") - 0.5).abs() < f32::EPSILON);
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
    let (epoch, _) = best.expect("test operation should succeed");
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
    assert_eq!(latest.expect("test operation should succeed").epoch, 2);
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
    let (min, max) = range.expect("test operation should succeed");
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

    let student = MockStudentModel::new("student", 100_000).with_response("test", "student answer");

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
