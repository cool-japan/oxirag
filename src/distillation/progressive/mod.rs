//! Progressive distillation module for multi-stage model compression.
//!
//! This module implements progressive distillation, which gradually distills
//! knowledge from large teacher models to smaller student models through
//! multiple intermediate stages.
//!
//! # Overview
//!
//! Progressive distillation works by:
//! 1. Starting with a large teacher model.
//! 2. Creating intermediate-sized student models.
//! 3. Distilling knowledge stage by stage until reaching the target size.
//!
//! This approach often yields better results than single-step distillation,
//! especially when there is a large gap between teacher and student model sizes.
//!
//! # Submodules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`types`] | Core data structures (`ModelSize`, `LossWeights`, configs, results) |
//! | [`scheduler`] | `ProgressiveScheduler` and scheduling strategies |
//! | [`distillation`] | `ProgressiveDistillation` orchestrator and `MockProgressiveDistillation` |

pub mod distillation;
pub mod scheduler;
pub mod types;

// ── Re-exports: types ────────────────────────────────────────────────────────
pub use types::{
    EpochMetrics, LossWeights, ModelSize, ProgressiveConfig, ProgressiveResult, StageConfig,
    StageResult,
};

// ── Re-exports: scheduler ────────────────────────────────────────────────────
pub use scheduler::ProgressiveScheduler;

// ── Re-exports: distillation ─────────────────────────────────────────────────
pub use distillation::{MockProgressiveDistillation, ProgressiveDistillation};

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_model_size_creation() {
        let size = ModelSize::new(7000.0, 32, 4096);
        assert!((size.params_millions - 7000.0).abs() < f64::EPSILON);
        assert_eq!(size.num_layers, 32);
        assert_eq!(size.hidden_dim, 4096);
    }

    #[test]
    fn test_model_size_compression_ratio() {
        let large = ModelSize::from_params(7000.0);
        let small = ModelSize::from_params(1000.0);
        let ratio = large.compression_ratio(&small);
        assert!((ratio - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_size_is_smaller_than() {
        let large = ModelSize::from_params(7000.0);
        let small = ModelSize::from_params(1000.0);
        assert!(small.is_smaller_than(&large));
        assert!(!large.is_smaller_than(&small));
    }

    #[test]
    fn test_loss_weights_default() {
        let weights = LossWeights::default();
        assert!((weights.hard_label - 0.5).abs() < f32::EPSILON);
        assert!((weights.soft_label - 0.5).abs() < f32::EPSILON);
        assert!(weights.is_valid());
    }

    #[test]
    fn test_loss_weights_normalized() {
        let weights = LossWeights {
            hard_label: 2.0,
            soft_label: 2.0,
            hidden_state: 0.0,
            attention: 0.0,
        };
        let normalized = weights.normalized();
        assert!((normalized.hard_label - 0.5).abs() < f32::EPSILON);
        assert!((normalized.soft_label - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stage_config_creation() {
        let teacher = ModelSize::from_params(7000.0);
        let student = ModelSize::from_params(3000.0);
        let config = StageConfig::new(teacher, student)
            .with_epochs(5)
            .with_learning_rate(1e-5)
            .with_temperature(3.0);

        assert_eq!(config.num_epochs, 5);
        assert!((config.learning_rate - 1e-5).abs() < f64::EPSILON);
        assert!((config.temperature - 3.0).abs() < f32::EPSILON);
        assert!(config.is_valid());
    }

    #[test]
    fn test_stage_config_validation() {
        let teacher = ModelSize::from_params(1000.0);
        let student = ModelSize::from_params(7000.0); // larger than teacher → invalid
        let config = StageConfig::new(teacher, student);
        assert!(!config.is_valid());
    }

    #[test]
    fn test_epoch_metrics_creation() {
        let metrics = EpochMetrics::new(1, 0.5, 1e-4)
            .with_val_loss(0.6)
            .with_train_accuracy(0.8)
            .with_val_accuracy(0.75)
            .with_duration(10.5);

        assert_eq!(metrics.epoch, 1);
        assert!((metrics.train_loss - 0.5).abs() < f32::EPSILON);
        assert!(
            (metrics.val_loss.expect("test operation should succeed") - 0.6).abs() < f32::EPSILON
        );
        assert!(
            (metrics
                .train_accuracy
                .expect("test operation should succeed")
                - 0.8)
                .abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn test_stage_result_success() {
        let history = vec![
            EpochMetrics::new(1, 0.5, 1e-4)
                .with_val_loss(0.6)
                .with_duration(10.0),
            EpochMetrics::new(2, 0.3, 1e-4)
                .with_val_loss(0.4)
                .with_duration(10.0),
        ];
        let result = StageResult::success(0, 0.3, 7.0, history);

        assert!(result.success);
        assert_eq!(result.stage_idx, 0);
        assert!((result.final_loss - 0.3).abs() < f32::EPSILON);
        assert!((result.compression_achieved - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stage_result_failure() {
        let result = StageResult::failure(0, "Test error");
        assert!(!result.success);
        assert_eq!(result.error_message.as_deref(), Some("Test error"));
    }

    #[test]
    fn test_stage_result_best_val_loss() {
        let history = vec![
            EpochMetrics::new(1, 0.5, 1e-4).with_val_loss(0.6),
            EpochMetrics::new(2, 0.3, 1e-4).with_val_loss(0.35),
            EpochMetrics::new(3, 0.25, 1e-4).with_val_loss(0.4),
        ];
        let result = StageResult::success(0, 0.25, 7.0, history);

        assert!(
            (result
                .best_val_loss()
                .expect("test operation should succeed")
                - 0.35)
                .abs()
                < f32::EPSILON
        );
        assert_eq!(result.best_epoch(), Some(2));
    }

    #[test]
    fn test_progressive_result_from_stages() {
        let stage1 = StageResult::success(0, 0.3, 2.0, vec![]);
        let stage2 = StageResult::success(1, 0.2, 2.0, vec![]);
        let result = ProgressiveResult::from_stages(vec![stage1, stage2]);

        assert!(result.all_stages_success);
        assert_eq!(result.stages_completed, 2);
        assert!((result.total_compression - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progressive_result_with_failure() {
        let stage1 = StageResult::success(0, 0.3, 2.0, vec![]);
        let stage2 = StageResult::failure(1, "Failed");
        let result = ProgressiveResult::from_stages(vec![stage1, stage2]);

        assert!(!result.all_stages_success);
        assert_eq!(result.stages_completed, 1);
        assert!(result.first_failed_stage().is_some());
        assert!(result.last_successful_stage().is_some());
    }

    #[test]
    fn test_linear_scheduler() {
        let scheduler = ProgressiveScheduler::linear(
            ModelSize::from_params(7000.0),
            ModelSize::from_params(1000.0),
            3,
        );

        let sizes = scheduler.generate_sizes();
        assert_eq!(sizes.len(), 4);
        assert!((sizes[0].params_millions - 7000.0).abs() < f64::EPSILON);
        assert!((sizes[3].params_millions - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exponential_scheduler() {
        let scheduler = ProgressiveScheduler::exponential(
            ModelSize::from_params(7000.0),
            ModelSize::from_params(1000.0),
            3,
            0.5,
        );

        let sizes = scheduler.generate_sizes();
        assert_eq!(sizes.len(), 4);
        // Exponential decay means faster reduction early.
        assert!(sizes[1].params_millions > 4000.0);
    }

    #[test]
    fn test_custom_scheduler() {
        let custom_sizes = vec![
            ModelSize::from_params(7000.0),
            ModelSize::from_params(5000.0),
            ModelSize::from_params(2000.0),
            ModelSize::from_params(1000.0),
        ];
        let scheduler = ProgressiveScheduler::custom(custom_sizes);

        let sizes = scheduler.generate_sizes();
        assert_eq!(sizes.len(), 4);
        assert_eq!(scheduler.num_stages(), 3);
    }

    #[test]
    fn test_scheduler_generate_stage_configs() {
        let scheduler = ProgressiveScheduler::linear(
            ModelSize::from_params(7000.0),
            ModelSize::from_params(1000.0),
            2,
        );

        let base_config = StageConfig::default();
        let stages = scheduler.generate_stage_configs(&base_config);

        assert_eq!(stages.len(), 2);
        assert!((stages[0].teacher_size.params_millions - 7000.0).abs() < f64::EPSILON);
        assert!(stages[0].student_size.params_millions < 7000.0);
    }

    #[test]
    fn test_progressive_config_with_stages() {
        let stages = vec![
            StageConfig::new(
                ModelSize::from_params(7000.0),
                ModelSize::from_params(3000.0),
            ),
            StageConfig::new(
                ModelSize::from_params(3000.0),
                ModelSize::from_params(1000.0),
            ),
        ];
        let config = ProgressiveConfig::with_stages(stages);

        assert_eq!(config.effective_stages().len(), 2);
        assert!(config.is_valid());
    }

    #[test]
    fn test_progressive_config_with_scheduler() {
        let scheduler = ProgressiveScheduler::linear(
            ModelSize::from_params(7000.0),
            ModelSize::from_params(1000.0),
            3,
        );
        let config = ProgressiveConfig::with_scheduler(scheduler);

        assert_eq!(config.effective_stages().len(), 3);
        assert!(config.is_valid());
    }

    #[test]
    fn test_progressive_config_max_stages() {
        let scheduler = ProgressiveScheduler::linear(
            ModelSize::from_params(7000.0),
            ModelSize::from_params(1000.0),
            5,
        );
        let config = ProgressiveConfig::with_scheduler(scheduler).with_max_stages(2);

        assert_eq!(config.effective_stages().len(), 2);
    }

    #[test]
    fn test_progressive_distillation_creation() {
        let pd = ProgressiveDistillation::with_defaults();
        assert_eq!(pd.current_stage(), 0);
        assert!(pd.results().is_empty());
    }

    #[test]
    fn test_progressive_distillation_add_stage() {
        let mut pd = ProgressiveDistillation::new(ProgressiveConfig {
            scheduler: None,
            stages: Vec::new(),
            ..Default::default()
        });

        pd.add_stage(StageConfig::new(
            ModelSize::from_params(7000.0),
            ModelSize::from_params(3000.0),
        ));

        pd.add_stage(StageConfig::new(
            ModelSize::from_params(3000.0),
            ModelSize::from_params(1000.0),
        ));

        assert_eq!(pd.num_stages(), 2);
    }

    #[test]
    fn test_progressive_distillation_reset() {
        let mut pd = ProgressiveDistillation::with_defaults();
        pd.current_stage = 2;
        pd.results.push(StageResult::success(0, 0.3, 2.0, vec![]));

        pd.reset();

        assert_eq!(pd.current_stage(), 0);
        assert!(pd.results().is_empty());
    }

    #[test]
    fn test_mock_progressive_distillation_creation() {
        let mock = MockProgressiveDistillation::with_defaults();
        assert_eq!(mock.inner().current_stage(), 0);
    }

    #[test]
    fn test_mock_progressive_distillation_add_stage() {
        let mut mock = MockProgressiveDistillation::new(ProgressiveConfig {
            scheduler: None,
            stages: Vec::new(),
            ..Default::default()
        });

        mock.add_stage(StageConfig::new(
            ModelSize::from_params(7000.0),
            ModelSize::from_params(3000.0),
        ));

        assert_eq!(mock.inner().num_stages(), 1);
    }

    #[cfg(feature = "distillation")]
    mod distillation_tests {
        use super::super::super::collector::TrainingExample;
        use super::*;

        #[test]
        fn test_progressive_distillation_run_stage() {
            let mut pd = ProgressiveDistillation::with_defaults();
            let data = vec![TrainingExample {
                input: "test input".to_string(),
                output: "test output".to_string(),
                confidence: 0.9,
            }];

            let result = pd
                .run_stage(0, &data)
                .expect("test operation should succeed");

            assert!(result.success);
            assert_eq!(result.stage_idx, 0);
            assert!(result.final_loss < 1.0);
            assert_eq!(pd.current_stage(), 1);
        }

        #[test]
        fn test_progressive_distillation_run_stage_invalid_index() {
            let mut pd = ProgressiveDistillation::new(ProgressiveConfig {
                scheduler: Some(ProgressiveScheduler::linear(
                    ModelSize::from_params(7000.0),
                    ModelSize::from_params(1000.0),
                    2,
                )),
                ..Default::default()
            });
            let data = vec![TrainingExample {
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
            }];

            let result = pd.run_stage(10, &data);
            assert!(result.is_err());
        }

        #[test]
        fn test_progressive_distillation_run_stage_empty_data() {
            let mut pd = ProgressiveDistillation::with_defaults();
            let data: Vec<TrainingExample> = vec![];

            let result = pd.run_stage(0, &data);
            assert!(result.is_err());
        }

        #[test]
        fn test_progressive_distillation_run_all() {
            let scheduler = ProgressiveScheduler::linear(
                ModelSize::from_params(7000.0),
                ModelSize::from_params(1000.0),
                2,
            );
            let config = ProgressiveConfig::with_scheduler(scheduler);
            let mut pd = ProgressiveDistillation::new(config);

            let data = vec![TrainingExample {
                input: "test input".to_string(),
                output: "test output".to_string(),
                confidence: 0.9,
            }];

            let result = pd.run_all(&data).expect("test operation should succeed");

            assert!(result.all_stages_success);
            assert_eq!(result.stages_completed, 2);
            assert!(result.total_compression > 1.0);
        }

        #[test]
        fn test_mock_progressive_distillation_run_stage() {
            let mut mock = MockProgressiveDistillation::with_defaults();
            let data = vec![TrainingExample {
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
            }];

            let result = mock
                .run_stage(0, &data)
                .expect("test operation should succeed");

            assert!(result.success);
            assert_eq!(mock.inner().current_stage(), 1);
        }

        #[test]
        fn test_mock_progressive_distillation_simulated_failure() {
            let mut mock =
                MockProgressiveDistillation::with_defaults().with_simulated_failure_at(1);

            let data = vec![TrainingExample {
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
            }];

            let result0 = mock.run_stage(0, &data);
            assert!(result0.is_ok());

            let result1 = mock.run_stage(1, &data);
            assert!(result1.is_err());
        }

        #[test]
        fn test_mock_progressive_distillation_custom_loss() {
            let mut mock = MockProgressiveDistillation::with_defaults().with_custom_loss(0, 0.5);

            let data = vec![TrainingExample {
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
            }];

            let result = mock
                .run_stage(0, &data)
                .expect("test operation should succeed");

            // Final loss should be less than custom starting loss after training.
            assert!(result.final_loss < 0.5);
        }

        #[test]
        fn test_mock_progressive_distillation_run_all() {
            let scheduler = ProgressiveScheduler::linear(
                ModelSize::from_params(7000.0),
                ModelSize::from_params(1000.0),
                2,
            );
            let config = ProgressiveConfig::with_scheduler(scheduler);
            let mut mock = MockProgressiveDistillation::new(config);

            let data = vec![TrainingExample {
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
            }];

            let result = mock.run_all(&data).expect("test operation should succeed");

            assert!(result.all_stages_success);
            assert_eq!(result.stage_results.len(), 2);
        }

        #[test]
        fn test_mock_progressive_distillation_run_all_with_failure() {
            let scheduler = ProgressiveScheduler::linear(
                ModelSize::from_params(7000.0),
                ModelSize::from_params(1000.0),
                3,
            );
            let config = ProgressiveConfig::with_scheduler(scheduler);
            let mut mock = MockProgressiveDistillation::new(config).with_simulated_failure_at(1);

            let data = vec![TrainingExample {
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
            }];

            let result = mock.run_all(&data).expect("test operation should succeed");

            assert!(!result.all_stages_success);
            assert_eq!(result.stages_completed, 1);
            // Execution stops after the first failure (continue_on_failure = false).
            assert_eq!(result.stage_results.len(), 2);
        }

        #[test]
        fn test_mock_progressive_distillation_continue_on_failure() {
            let scheduler = ProgressiveScheduler::linear(
                ModelSize::from_params(7000.0),
                ModelSize::from_params(1000.0),
                3,
            );
            let config = ProgressiveConfig::with_scheduler(scheduler).continue_on_failure(true);
            let mut mock = MockProgressiveDistillation::new(config).with_simulated_failure_at(1);

            let data = vec![TrainingExample {
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
            }];

            let result = mock.run_all(&data).expect("test operation should succeed");

            assert!(!result.all_stages_success);
            // All 3 stages attempted due to continue_on_failure.
            assert_eq!(result.stage_results.len(), 3);
            // Stages 0 and 2 succeeded, stage 1 failed.
            assert_eq!(result.stages_completed, 2);
        }

        #[test]
        fn test_early_stopping() {
            let stages = vec![
                StageConfig::new(
                    ModelSize::from_params(7000.0),
                    ModelSize::from_params(1000.0),
                )
                .with_epochs(10),
            ];

            let config = ProgressiveConfig::with_stages(stages).with_early_stopping(2);

            let mut pd = ProgressiveDistillation::new(config);

            let data = vec![TrainingExample {
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
            }];

            let result = pd
                .run_stage(0, &data)
                .expect("test operation should succeed");

            // Early stopping may have kicked in before all 10 epochs.
            assert!(result.training_history.len() <= 10);
        }
    }
}
