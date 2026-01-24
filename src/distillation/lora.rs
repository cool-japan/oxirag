//! `LoRA` (Low-Rank Adaptation) training types for distillation.
//!
//! This module provides types and traits for `LoRA` fine-tuning pipelines,
//! enabling the creation of specialized lightweight models for frequent queries.

use super::types::QueryPattern;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "distillation")]
use crate::error::{DistillationError, OxiRagError};

/// `LoRA` adapter configuration for fine-tuning.
///
/// Configures the Low-Rank Adaptation parameters used when training
/// a specialized model from collected Q&A pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// `LoRA` rank (typically 4-64). Lower ranks are more efficient but less expressive.
    pub rank: usize,
    /// Scaling factor (`alpha`). Controls the magnitude of `LoRA` updates.
    pub alpha: f32,
    /// Dropout rate for regularization (0.0 - 1.0).
    pub dropout: f32,
    /// Module names to apply `LoRA` adaptation to.
    pub target_modules: Vec<String>,
    /// Learning rate for training.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Batch size for training.
    pub batch_size: usize,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            learning_rate: 1e-4,
            num_epochs: 3,
            batch_size: 4,
        }
    }
}

impl LoraConfig {
    /// Create a new `LoRA` configuration with the specified rank.
    #[must_use]
    pub fn with_rank(rank: usize) -> Self {
        Self {
            rank,
            ..Default::default()
        }
    }

    /// Set the scaling factor (alpha).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout.clamp(0.0, 1.0);
        self
    }

    /// Set the target modules for `LoRA` adaptation.
    #[must_use]
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of epochs.
    #[must_use]
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.num_epochs = epochs;
        self
    }

    /// Set the batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Validate the configuration.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.rank > 0
            && self.alpha > 0.0
            && self.dropout >= 0.0
            && self.dropout <= 1.0
            && self.learning_rate > 0.0
            && self.num_epochs > 0
            && self.batch_size > 0
            && !self.target_modules.is_empty()
    }

    /// Calculate the effective scaling factor.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn scaling_factor(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// Training example for `LoRA` fine-tuning.
///
/// Represents a single input-output pair with an associated weight
/// for importance sampling during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTrainingExample {
    /// The input query/prompt.
    pub input: String,
    /// The expected output/response.
    pub output: String,
    /// Weight for importance sampling (higher = more important).
    pub weight: f32,
}

impl LoraTrainingExample {
    /// Create a new training example with default weight of 1.0.
    #[must_use]
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            weight: 1.0,
        }
    }

    /// Create a new training example with a specified weight.
    #[must_use]
    pub fn with_weight(input: impl Into<String>, output: impl Into<String>, weight: f32) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            weight: weight.max(0.0),
        }
    }

    /// Set the weight for this example.
    #[must_use]
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.0);
        self
    }

    /// Check if this example is valid (non-empty input and output).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.input.trim().is_empty() && !self.output.trim().is_empty() && self.weight > 0.0
    }

    /// Calculate the effective contribution of this example.
    #[must_use]
    pub fn effective_contribution(&self, total_weight: f32) -> f32 {
        if total_weight <= 0.0 {
            0.0
        } else {
            self.weight / total_weight
        }
    }
}

/// Status of a `LoRA` training job.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum TrainingStatus {
    /// Job is queued and waiting to start.
    #[default]
    Pending,
    /// Job is preparing data and initializing.
    Preparing,
    /// Job is actively training.
    Training {
        /// Current epoch number (1-indexed).
        epoch: usize,
        /// Current training loss.
        loss: f32,
    },
    /// Training completed successfully.
    Completed {
        /// Final training loss achieved.
        final_loss: f32,
    },
    /// Training failed with an error.
    Failed {
        /// Description of the error.
        error: String,
    },
}

impl TrainingStatus {
    /// Check if the job is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed { .. } | Self::Failed { .. })
    }

    /// Check if the job is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Preparing | Self::Training { .. })
    }

    /// Check if the job completed successfully.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// Get the current loss if available.
    #[must_use]
    pub fn current_loss(&self) -> Option<f32> {
        match self {
            Self::Training { loss, .. } => Some(*loss),
            Self::Completed { final_loss } => Some(*final_loss),
            _ => None,
        }
    }

    /// Get the current epoch if training.
    #[must_use]
    pub fn current_epoch(&self) -> Option<usize> {
        match self {
            Self::Training { epoch, .. } => Some(*epoch),
            _ => None,
        }
    }
}

/// A `LoRA` training job.
///
/// Represents a complete training job with its configuration, examples,
/// and current status.
#[derive(Debug, Clone)]
pub struct TrainingJob {
    /// Unique identifier for this job.
    pub job_id: String,
    /// The query pattern this job is training for.
    pub pattern: QueryPattern,
    /// `LoRA` configuration for this job.
    pub config: LoraConfig,
    /// Training examples for this job.
    pub examples: Vec<LoraTrainingExample>,
    /// Current status of the job.
    pub status: TrainingStatus,
    /// Unix timestamp when the job was created.
    pub created_at: u64,
    /// Unix timestamp when the job completed (if applicable).
    pub completed_at: Option<u64>,
}

impl TrainingJob {
    /// Create a new training job.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        pattern: QueryPattern,
        config: LoraConfig,
        examples: Vec<LoraTrainingExample>,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            pattern,
            config,
            examples,
            status: TrainingStatus::Pending,
            created_at: super::types::current_timestamp(),
            completed_at: None,
        }
    }

    /// Get the total weight of all examples.
    #[must_use]
    pub fn total_weight(&self) -> f32 {
        self.examples.iter().map(|e| e.weight).sum()
    }

    /// Get the number of valid examples.
    #[must_use]
    pub fn valid_example_count(&self) -> usize {
        self.examples.iter().filter(|e| e.is_valid()).count()
    }

    /// Check if this job is ready to start.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self.status, TrainingStatus::Pending)
            && self.config.is_valid()
            && self.valid_example_count() >= self.config.batch_size
    }

    /// Calculate the estimated number of training steps.
    #[must_use]
    pub fn estimated_steps(&self) -> usize {
        let num_examples = self.valid_example_count();
        if self.config.batch_size == 0 {
            return 0;
        }
        let steps_per_epoch = num_examples.div_ceil(self.config.batch_size);
        steps_per_epoch * self.config.num_epochs
    }

    /// Update the job status.
    pub fn update_status(&mut self, status: TrainingStatus) {
        if status.is_terminal() && self.completed_at.is_none() {
            self.completed_at = Some(super::types::current_timestamp());
        }
        self.status = status;
    }

    /// Mark the job as failed.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.update_status(TrainingStatus::Failed {
            error: error.into(),
        });
    }

    /// Mark the job as completed.
    pub fn complete(&mut self, final_loss: f32) {
        self.update_status(TrainingStatus::Completed { final_loss });
    }

    /// Get the duration of the job in seconds (if completed).
    #[must_use]
    pub fn duration_secs(&self) -> Option<u64> {
        self.completed_at.map(|c| c.saturating_sub(self.created_at))
    }
}

/// Trait for `LoRA` training backends.
///
/// Implementations of this trait handle the actual training process,
/// whether local or remote.
#[async_trait]
pub trait LoraTrainer: Send + Sync {
    /// Create a new training job.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The query pattern to train for
    /// * `examples` - Training examples
    /// * `config` - `LoRA` configuration
    ///
    /// # Returns
    ///
    /// The job ID on success, or an error if job creation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the job cannot be created.
    async fn create_job(
        &mut self,
        pattern: &QueryPattern,
        examples: Vec<LoraTrainingExample>,
        config: LoraConfig,
    ) -> Result<String, OxiRagError>;

    /// Get the status of a training job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job identifier
    ///
    /// # Returns
    ///
    /// The current status, or None if the job doesn't exist.
    async fn get_status(&self, job_id: &str) -> Option<TrainingStatus>;

    /// Cancel a training job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job identifier
    ///
    /// # Errors
    ///
    /// Returns an error if the job cannot be cancelled.
    async fn cancel_job(&mut self, job_id: &str) -> Result<(), OxiRagError>;

    /// List all training jobs.
    fn list_jobs(&self) -> Vec<&TrainingJob>;
}

/// A mock `LoRA` trainer for testing purposes.
///
/// This implementation simulates training without actually running
/// any ML operations, useful for testing the distillation pipeline.
#[derive(Debug, Default)]
pub struct MockLoraTrainer {
    jobs: HashMap<String, TrainingJob>,
    next_job_id: u64,
    simulate_failure: bool,
}

impl MockLoraTrainer {
    /// Create a new mock trainer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the trainer to simulate failures.
    #[must_use]
    pub fn with_simulated_failure(mut self, simulate: bool) -> Self {
        self.simulate_failure = simulate;
        self
    }

    /// Get a job by ID.
    #[must_use]
    pub fn get_job(&self, job_id: &str) -> Option<&TrainingJob> {
        self.jobs.get(job_id)
    }

    /// Get a mutable reference to a job by ID.
    pub fn get_job_mut(&mut self, job_id: &str) -> Option<&mut TrainingJob> {
        self.jobs.get_mut(job_id)
    }

    /// Simulate training progress for a job.
    pub fn simulate_progress(&mut self, job_id: &str, epoch: usize, loss: f32) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.update_status(TrainingStatus::Training { epoch, loss });
        }
    }

    /// Simulate training completion for a job.
    pub fn simulate_completion(&mut self, job_id: &str, final_loss: f32) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.complete(final_loss);
        }
    }

    /// Get the number of active jobs.
    #[must_use]
    pub fn active_job_count(&self) -> usize {
        self.jobs.values().filter(|j| j.status.is_active()).count()
    }

    /// Get the number of completed jobs.
    #[must_use]
    pub fn completed_job_count(&self) -> usize {
        self.jobs.values().filter(|j| j.status.is_success()).count()
    }

    /// Clear all completed jobs.
    pub fn clear_completed(&mut self) {
        self.jobs.retain(|_, j| !j.status.is_terminal());
    }

    /// Generate the next job ID.
    fn generate_job_id(&mut self) -> String {
        self.next_job_id += 1;
        format!("mock-job-{}", self.next_job_id)
    }
}

#[async_trait]
impl LoraTrainer for MockLoraTrainer {
    async fn create_job(
        &mut self,
        pattern: &QueryPattern,
        examples: Vec<LoraTrainingExample>,
        config: LoraConfig,
    ) -> Result<String, OxiRagError> {
        if self.simulate_failure {
            return Err(DistillationError::TrackingFailed("Simulated failure".to_string()).into());
        }

        if !config.is_valid() {
            return Err(
                DistillationError::InvalidConfig("Invalid LoRA configuration".to_string()).into(),
            );
        }

        if examples.is_empty() {
            return Err(DistillationError::CollectionFailed(
                "No training examples provided".to_string(),
            )
            .into());
        }

        let job_id = self.generate_job_id();
        let job = TrainingJob::new(job_id.clone(), pattern.clone(), config, examples);
        self.jobs.insert(job_id.clone(), job);

        Ok(job_id)
    }

    async fn get_status(&self, job_id: &str) -> Option<TrainingStatus> {
        self.jobs.get(job_id).map(|j| j.status.clone())
    }

    async fn cancel_job(&mut self, job_id: &str) -> Result<(), OxiRagError> {
        let job = self.jobs.get_mut(job_id).ok_or_else(|| {
            DistillationError::PatternNotFound(format!("Job not found: {job_id}"))
        })?;

        if job.status.is_terminal() {
            return Err(DistillationError::TrackingFailed(
                "Cannot cancel a completed job".to_string(),
            )
            .into());
        }

        job.fail("Cancelled by user");
        Ok(())
    }

    fn list_jobs(&self) -> Vec<&TrainingJob> {
        self.jobs.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert!((config.alpha - 16.0).abs() < f32::EPSILON);
        assert!(config.is_valid());
    }

    #[test]
    fn test_lora_config_builder() {
        let config = LoraConfig::with_rank(16)
            .with_alpha(32.0)
            .with_dropout(0.1)
            .with_epochs(5)
            .with_batch_size(8);

        assert_eq!(config.rank, 16);
        assert!((config.alpha - 32.0).abs() < f32::EPSILON);
        assert!((config.dropout - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.num_epochs, 5);
        assert_eq!(config.batch_size, 8);
    }

    #[test]
    fn test_lora_config_scaling_factor() {
        let config = LoraConfig {
            rank: 8,
            alpha: 16.0,
            ..Default::default()
        };
        assert!((config.scaling_factor() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lora_config_validation() {
        let invalid_config = LoraConfig {
            rank: 0,
            ..Default::default()
        };
        assert!(!invalid_config.is_valid());

        let invalid_dropout = LoraConfig::default().with_dropout(1.5);
        // Dropout should be clamped to 1.0
        assert!((invalid_dropout.dropout - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_training_example_creation() {
        let example = LoraTrainingExample::new("input", "output");
        assert_eq!(example.input, "input");
        assert_eq!(example.output, "output");
        assert!((example.weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_training_example_with_weight() {
        let example = LoraTrainingExample::with_weight("input", "output", 2.5);
        assert!((example.weight - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_training_example_validation() {
        let valid = LoraTrainingExample::new("input", "output");
        assert!(valid.is_valid());

        let empty_input = LoraTrainingExample::new("", "output");
        assert!(!empty_input.is_valid());

        let zero_weight = LoraTrainingExample::with_weight("input", "output", 0.0);
        assert!(!zero_weight.is_valid());
    }

    #[test]
    fn test_training_status_states() {
        assert!(!TrainingStatus::Pending.is_terminal());
        assert!(!TrainingStatus::Preparing.is_active().not());
        assert!(TrainingStatus::Completed { final_loss: 0.1 }.is_terminal());
        assert!(
            TrainingStatus::Failed {
                error: "test".to_string()
            }
            .is_terminal()
        );
    }

    #[test]
    fn test_training_status_loss() {
        let training = TrainingStatus::Training {
            epoch: 1,
            loss: 0.5,
        };
        assert!((training.current_loss().unwrap() - 0.5).abs() < f32::EPSILON);

        let completed = TrainingStatus::Completed { final_loss: 0.1 };
        assert!((completed.current_loss().unwrap() - 0.1).abs() < f32::EPSILON);

        assert!(TrainingStatus::Pending.current_loss().is_none());
    }

    #[test]
    fn test_training_job_creation() {
        let pattern = QueryPattern::new("test query");
        let config = LoraConfig::default();
        let examples = vec![LoraTrainingExample::new("input", "output")];

        let job = TrainingJob::new("job-1", pattern, config, examples);

        assert_eq!(job.job_id, "job-1");
        assert_eq!(job.status, TrainingStatus::Pending);
        assert!(job.completed_at.is_none());
    }

    #[test]
    fn test_training_job_total_weight() {
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();
        let examples = vec![
            LoraTrainingExample::with_weight("a", "b", 1.0),
            LoraTrainingExample::with_weight("c", "d", 2.0),
            LoraTrainingExample::with_weight("e", "f", 0.5),
        ];

        let job = TrainingJob::new("job", pattern, config, examples);
        assert!((job.total_weight() - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_training_job_status_update() {
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();
        let examples = vec![LoraTrainingExample::new("input", "output")];
        let mut job = TrainingJob::new("job", pattern, config, examples);

        job.update_status(TrainingStatus::Training {
            epoch: 1,
            loss: 0.5,
        });
        assert!(matches!(job.status, TrainingStatus::Training { .. }));

        job.complete(0.1);
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn test_training_job_estimated_steps() {
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default().with_epochs(2).with_batch_size(2);
        let examples = vec![
            LoraTrainingExample::new("a", "b"),
            LoraTrainingExample::new("c", "d"),
            LoraTrainingExample::new("e", "f"),
            LoraTrainingExample::new("g", "h"),
        ];

        let job = TrainingJob::new("job", pattern, config, examples);
        // 4 examples / batch_size 2 = 2 steps per epoch * 2 epochs = 4
        assert_eq!(job.estimated_steps(), 4);
    }

    #[tokio::test]
    async fn test_mock_trainer_create_job() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();
        let examples = vec![LoraTrainingExample::new("input", "output")];

        let result = trainer.create_job(&pattern, examples, config).await;
        assert!(result.is_ok());

        let job_id = result.unwrap();
        assert!(trainer.get_job(&job_id).is_some());
    }

    #[tokio::test]
    async fn test_mock_trainer_get_status() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();
        let examples = vec![LoraTrainingExample::new("input", "output")];

        let job_id = trainer
            .create_job(&pattern, examples, config)
            .await
            .unwrap();
        let status = trainer.get_status(&job_id).await;

        assert!(matches!(status, Some(TrainingStatus::Pending)));
    }

    #[tokio::test]
    async fn test_mock_trainer_cancel_job() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();
        let examples = vec![LoraTrainingExample::new("input", "output")];

        let job_id = trainer
            .create_job(&pattern, examples, config)
            .await
            .unwrap();
        let result = trainer.cancel_job(&job_id).await;

        assert!(result.is_ok());
        assert!(matches!(
            trainer.get_status(&job_id).await,
            Some(TrainingStatus::Failed { .. })
        ));
    }

    #[tokio::test]
    async fn test_mock_trainer_simulate_failure() {
        let mut trainer = MockLoraTrainer::new().with_simulated_failure(true);
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();
        let examples = vec![LoraTrainingExample::new("input", "output")];

        let result = trainer.create_job(&pattern, examples, config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_trainer_list_jobs() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();

        for i in 0..3 {
            let examples = vec![LoraTrainingExample::new(format!("input{i}"), "output")];
            let _ = trainer.create_job(&pattern, examples, config.clone()).await;
        }

        assert_eq!(trainer.list_jobs().len(), 3);
    }

    #[tokio::test]
    async fn test_mock_trainer_simulate_progress() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();
        let examples = vec![LoraTrainingExample::new("input", "output")];

        let job_id = trainer
            .create_job(&pattern, examples, config)
            .await
            .unwrap();

        trainer.simulate_progress(&job_id, 1, 0.5);
        let status = trainer.get_status(&job_id).await.unwrap();
        assert!(matches!(status, TrainingStatus::Training { epoch: 1, .. }));

        trainer.simulate_completion(&job_id, 0.1);
        let status = trainer.get_status(&job_id).await.unwrap();
        assert!(matches!(status, TrainingStatus::Completed { .. }));
    }

    #[tokio::test]
    async fn test_mock_trainer_active_count() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();

        let examples = vec![LoraTrainingExample::new("input", "output")];
        let job_id = trainer
            .create_job(&pattern, examples, config)
            .await
            .unwrap();

        assert_eq!(trainer.active_job_count(), 0); // Pending is not active

        trainer.simulate_progress(&job_id, 1, 0.5);
        assert_eq!(trainer.active_job_count(), 1);

        trainer.simulate_completion(&job_id, 0.1);
        assert_eq!(trainer.active_job_count(), 0);
        assert_eq!(trainer.completed_job_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_trainer_clear_completed() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();

        let examples = vec![LoraTrainingExample::new("input", "output")];
        let job_id = trainer
            .create_job(&pattern, examples, config)
            .await
            .unwrap();

        trainer.simulate_completion(&job_id, 0.1);
        assert_eq!(trainer.list_jobs().len(), 1);

        trainer.clear_completed();
        assert_eq!(trainer.list_jobs().len(), 0);
    }

    #[tokio::test]
    async fn test_mock_trainer_invalid_config() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig {
            rank: 0, // Invalid
            ..Default::default()
        };
        let examples = vec![LoraTrainingExample::new("input", "output")];

        let result = trainer.create_job(&pattern, examples, config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_trainer_empty_examples() {
        let mut trainer = MockLoraTrainer::new();
        let pattern = QueryPattern::new("test");
        let config = LoraConfig::default();
        let examples: Vec<LoraTrainingExample> = vec![];

        let result = trainer.create_job(&pattern, examples, config).await;
        assert!(result.is_err());
    }

    // Trait helper for tests
    trait BoolExt {
        fn not(self) -> bool;
    }

    impl BoolExt for bool {
        fn not(self) -> bool {
            !self
        }
    }
}
