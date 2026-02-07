//! Progressive distillation module for multi-stage model compression.
//!
//! This module implements progressive distillation, which gradually distills
//! knowledge from large teacher models to smaller student models through
//! multiple intermediate stages.
//!
//! # Overview
//!
//! Progressive distillation works by:
//! 1. Starting with a large teacher model
//! 2. Creating intermediate-sized student models
//! 3. Distilling knowledge stage by stage until reaching the target size
//!
//! This approach often yields better results than single-step distillation,
//! especially when there is a large gap between teacher and student model sizes.

use super::collector::TrainingExample;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "distillation")]
use crate::error::{DistillationError, OxiRagError};

/// Model size specification for distillation stages.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelSize {
    /// Number of parameters in millions.
    pub params_millions: f64,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension size.
    pub hidden_dim: usize,
}

impl ModelSize {
    /// Create a new model size specification.
    #[must_use]
    pub const fn new(params_millions: f64, num_layers: usize, hidden_dim: usize) -> Self {
        Self {
            params_millions,
            num_layers,
            hidden_dim,
        }
    }

    /// Create a model size from parameters count only.
    #[must_use]
    pub const fn from_params(params_millions: f64) -> Self {
        Self {
            params_millions,
            num_layers: 0,
            hidden_dim: 0,
        }
    }

    /// Calculate compression ratio between this size and another.
    #[must_use]
    pub fn compression_ratio(&self, other: &Self) -> f64 {
        if other.params_millions <= 0.0 {
            return 0.0;
        }
        self.params_millions / other.params_millions
    }

    /// Check if this model is smaller than another.
    #[must_use]
    pub fn is_smaller_than(&self, other: &Self) -> bool {
        self.params_millions < other.params_millions
    }
}

impl Default for ModelSize {
    fn default() -> Self {
        Self::from_params(7000.0) // 7B params default
    }
}

/// Loss weight configuration for distillation training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossWeights {
    /// Weight for hard label loss (cross-entropy with ground truth).
    pub hard_label: f32,
    /// Weight for soft label loss (KL divergence with teacher).
    pub soft_label: f32,
    /// Weight for hidden state alignment loss.
    pub hidden_state: f32,
    /// Weight for attention alignment loss.
    pub attention: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            hard_label: 0.5,
            soft_label: 0.5,
            hidden_state: 0.0,
            attention: 0.0,
        }
    }
}

impl LossWeights {
    /// Create loss weights with only hard labels.
    #[must_use]
    pub const fn hard_only() -> Self {
        Self {
            hard_label: 1.0,
            soft_label: 0.0,
            hidden_state: 0.0,
            attention: 0.0,
        }
    }

    /// Create loss weights with only soft labels.
    #[must_use]
    pub const fn soft_only() -> Self {
        Self {
            hard_label: 0.0,
            soft_label: 1.0,
            hidden_state: 0.0,
            attention: 0.0,
        }
    }

    /// Create balanced loss weights.
    #[must_use]
    pub const fn balanced() -> Self {
        Self {
            hard_label: 0.5,
            soft_label: 0.5,
            hidden_state: 0.0,
            attention: 0.0,
        }
    }

    /// Create loss weights with hidden state alignment.
    #[must_use]
    pub const fn with_hidden_states(hard: f32, soft: f32, hidden: f32) -> Self {
        Self {
            hard_label: hard,
            soft_label: soft,
            hidden_state: hidden,
            attention: 0.0,
        }
    }

    /// Normalize weights to sum to 1.0.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let sum = self.hard_label + self.soft_label + self.hidden_state + self.attention;
        if sum <= 0.0 {
            return Self::default();
        }
        Self {
            hard_label: self.hard_label / sum,
            soft_label: self.soft_label / sum,
            hidden_state: self.hidden_state / sum,
            attention: self.attention / sum,
        }
    }

    /// Check if weights are valid (non-negative and at least one non-zero).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.hard_label >= 0.0
            && self.soft_label >= 0.0
            && self.hidden_state >= 0.0
            && self.attention >= 0.0
            && (self.hard_label > 0.0
                || self.soft_label > 0.0
                || self.hidden_state > 0.0
                || self.attention > 0.0)
    }
}

/// Configuration for a single distillation stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Teacher model size for this stage.
    pub teacher_size: ModelSize,
    /// Student model size for this stage.
    pub student_size: ModelSize,
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Learning rate for this stage.
    pub learning_rate: f64,
    /// Distillation temperature.
    pub temperature: f32,
    /// Loss weights for this stage.
    pub loss_weights: LossWeights,
    /// Batch size for training.
    pub batch_size: usize,
    /// Optional name for the stage.
    pub stage_name: Option<String>,
    /// Warmup steps for learning rate.
    pub warmup_steps: usize,
    /// Weight decay for regularization.
    pub weight_decay: f64,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            teacher_size: ModelSize::from_params(7000.0),
            student_size: ModelSize::from_params(1000.0),
            num_epochs: 3,
            learning_rate: 1e-4,
            temperature: 2.0,
            loss_weights: LossWeights::default(),
            batch_size: 8,
            stage_name: None,
            warmup_steps: 100,
            weight_decay: 0.01,
        }
    }
}

impl StageConfig {
    /// Create a new stage configuration.
    #[must_use]
    pub fn new(teacher_size: ModelSize, student_size: ModelSize) -> Self {
        Self {
            teacher_size,
            student_size,
            ..Default::default()
        }
    }

    /// Set the number of epochs.
    #[must_use]
    pub const fn with_epochs(mut self, epochs: usize) -> Self {
        self.num_epochs = epochs;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub const fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set the loss weights.
    #[must_use]
    pub fn with_loss_weights(mut self, weights: LossWeights) -> Self {
        self.loss_weights = weights;
        self
    }

    /// Set the batch size.
    #[must_use]
    pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the stage name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.stage_name = Some(name.into());
        self
    }

    /// Get the compression ratio for this stage.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        self.teacher_size.compression_ratio(&self.student_size)
    }

    /// Validate the stage configuration.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.student_size.is_smaller_than(&self.teacher_size)
            && self.num_epochs > 0
            && self.learning_rate > 0.0
            && self.temperature > 0.0
            && self.batch_size > 0
            && self.loss_weights.is_valid()
    }
}

/// Metrics for a single training epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// Epoch number (1-indexed).
    pub epoch: usize,
    /// Training loss.
    pub train_loss: f32,
    /// Validation loss (if available).
    pub val_loss: Option<f32>,
    /// Training accuracy.
    pub train_accuracy: Option<f32>,
    /// Validation accuracy (if available).
    pub val_accuracy: Option<f32>,
    /// Learning rate at this epoch.
    pub learning_rate: f64,
    /// Duration of this epoch in seconds.
    pub duration_secs: f64,
}

impl EpochMetrics {
    /// Create new epoch metrics.
    #[must_use]
    pub fn new(epoch: usize, train_loss: f32, learning_rate: f64) -> Self {
        Self {
            epoch,
            train_loss,
            val_loss: None,
            train_accuracy: None,
            val_accuracy: None,
            learning_rate,
            duration_secs: 0.0,
        }
    }

    /// Set validation loss.
    #[must_use]
    pub const fn with_val_loss(mut self, val_loss: f32) -> Self {
        self.val_loss = Some(val_loss);
        self
    }

    /// Set training accuracy.
    #[must_use]
    pub const fn with_train_accuracy(mut self, accuracy: f32) -> Self {
        self.train_accuracy = Some(accuracy);
        self
    }

    /// Set validation accuracy.
    #[must_use]
    pub const fn with_val_accuracy(mut self, accuracy: f32) -> Self {
        self.val_accuracy = Some(accuracy);
        self
    }

    /// Set duration.
    #[must_use]
    pub const fn with_duration(mut self, duration_secs: f64) -> Self {
        self.duration_secs = duration_secs;
        self
    }
}

/// Result of a single distillation stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    /// Index of the stage (0-indexed).
    pub stage_idx: usize,
    /// Final training loss.
    pub final_loss: f32,
    /// Final accuracy (if available).
    pub accuracy: Option<f32>,
    /// Compression ratio achieved.
    pub compression_achieved: f64,
    /// Training history for all epochs.
    pub training_history: Vec<EpochMetrics>,
    /// Total duration of the stage in seconds.
    pub total_duration_secs: f64,
    /// Whether the stage completed successfully.
    pub success: bool,
    /// Error message if failed.
    pub error_message: Option<String>,
}

impl StageResult {
    /// Create a new successful stage result.
    #[must_use]
    pub fn success(
        stage_idx: usize,
        final_loss: f32,
        compression_achieved: f64,
        training_history: Vec<EpochMetrics>,
    ) -> Self {
        let total_duration_secs: f64 = training_history.iter().map(|e| e.duration_secs).sum();
        let accuracy = training_history.last().and_then(|e| e.val_accuracy);
        Self {
            stage_idx,
            final_loss,
            accuracy,
            compression_achieved,
            training_history,
            total_duration_secs,
            success: true,
            error_message: None,
        }
    }

    /// Create a new failed stage result.
    #[must_use]
    pub fn failure(stage_idx: usize, error: impl Into<String>) -> Self {
        Self {
            stage_idx,
            final_loss: f32::MAX,
            accuracy: None,
            compression_achieved: 0.0,
            training_history: Vec::new(),
            total_duration_secs: 0.0,
            success: false,
            error_message: Some(error.into()),
        }
    }

    /// Get the best validation loss across all epochs.
    #[must_use]
    pub fn best_val_loss(&self) -> Option<f32> {
        self.training_history
            .iter()
            .filter_map(|e| e.val_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the best epoch (lowest validation loss).
    ///
    /// # Panics
    ///
    /// Does not panic - only operates on entries with `Some(val_loss)`.
    #[must_use]
    pub fn best_epoch(&self) -> Option<usize> {
        self.training_history
            .iter()
            .filter_map(|e| e.val_loss.map(|loss| (e.epoch, loss)))
            .min_by(|(_, a_loss), (_, b_loss)| {
                a_loss
                    .partial_cmp(b_loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(epoch, _)| epoch)
    }
}

/// Result of the entire progressive distillation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveResult {
    /// Results for each stage.
    pub stage_results: Vec<StageResult>,
    /// Overall final loss.
    pub final_loss: f32,
    /// Overall compression ratio achieved.
    pub total_compression: f64,
    /// Total duration in seconds.
    pub total_duration_secs: f64,
    /// Number of stages completed successfully.
    pub stages_completed: usize,
    /// Whether all stages completed successfully.
    pub all_stages_success: bool,
}

impl ProgressiveResult {
    /// Create a new progressive result from stage results.
    #[must_use]
    pub fn from_stages(stage_results: Vec<StageResult>) -> Self {
        let stages_completed = stage_results.iter().filter(|s| s.success).count();
        let all_stages_success = stages_completed == stage_results.len();
        let total_duration_secs: f64 = stage_results.iter().map(|s| s.total_duration_secs).sum();

        let final_loss = stage_results.last().map_or(f32::MAX, |s| s.final_loss);

        let total_compression = stage_results
            .iter()
            .filter(|s| s.success)
            .map(|s| s.compression_achieved)
            .product();

        Self {
            stage_results,
            final_loss,
            total_compression,
            total_duration_secs,
            stages_completed,
            all_stages_success,
        }
    }

    /// Get the last successful stage result.
    #[must_use]
    pub fn last_successful_stage(&self) -> Option<&StageResult> {
        self.stage_results.iter().rev().find(|s| s.success)
    }

    /// Get the first failed stage result.
    #[must_use]
    pub fn first_failed_stage(&self) -> Option<&StageResult> {
        self.stage_results.iter().find(|s| !s.success)
    }
}

/// Scheduler for determining model sizes across progressive distillation stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressiveScheduler {
    /// Linearly decrease model size.
    Linear {
        /// Starting model size.
        start_size: ModelSize,
        /// Target model size.
        end_size: ModelSize,
        /// Number of stages.
        num_stages: usize,
    },
    /// Exponentially decrease model size.
    Exponential {
        /// Starting model size.
        start_size: ModelSize,
        /// Target model size.
        end_size: ModelSize,
        /// Number of stages.
        num_stages: usize,
        /// Decay factor (0.0-1.0, lower = faster decay).
        decay_factor: f64,
    },
    /// User-defined custom schedule.
    Custom {
        /// List of model sizes for each stage boundary.
        sizes: Vec<ModelSize>,
    },
}

impl ProgressiveScheduler {
    /// Create a linear scheduler.
    #[must_use]
    pub const fn linear(start_size: ModelSize, end_size: ModelSize, num_stages: usize) -> Self {
        Self::Linear {
            start_size,
            end_size,
            num_stages,
        }
    }

    /// Create an exponential scheduler.
    #[must_use]
    pub const fn exponential(
        start_size: ModelSize,
        end_size: ModelSize,
        num_stages: usize,
        decay_factor: f64,
    ) -> Self {
        Self::Exponential {
            start_size,
            end_size,
            num_stages,
            decay_factor,
        }
    }

    /// Create a custom scheduler with predefined sizes.
    #[must_use]
    pub const fn custom(sizes: Vec<ModelSize>) -> Self {
        Self::Custom { sizes }
    }

    /// Get the number of stages.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        match self {
            Self::Linear { num_stages, .. } | Self::Exponential { num_stages, .. } => *num_stages,
            Self::Custom { sizes } => sizes.len().saturating_sub(1),
        }
    }

    /// Generate the model sizes for all stage boundaries.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn generate_sizes(&self) -> Vec<ModelSize> {
        match self {
            Self::Linear {
                start_size,
                end_size,
                num_stages,
            } => {
                let mut sizes = Vec::with_capacity(*num_stages + 1);
                for i in 0..=*num_stages {
                    let t = if *num_stages == 0 {
                        1.0
                    } else {
                        i as f64 / *num_stages as f64
                    };
                    let params =
                        start_size.params_millions * (1.0 - t) + end_size.params_millions * t;
                    let layers = ((start_size.num_layers as f64 * (1.0 - t)
                        + end_size.num_layers as f64 * t)
                        .round()) as usize;
                    let hidden = ((start_size.hidden_dim as f64 * (1.0 - t)
                        + end_size.hidden_dim as f64 * t)
                        .round()) as usize;
                    sizes.push(ModelSize::new(params, layers, hidden));
                }
                sizes
            }
            Self::Exponential {
                start_size,
                end_size,
                num_stages,
                decay_factor,
            } => {
                let mut sizes = Vec::with_capacity(*num_stages + 1);
                for i in 0..=*num_stages {
                    let t = if *num_stages == 0 {
                        1.0
                    } else {
                        i as f64 / *num_stages as f64
                    };
                    // Exponential interpolation
                    let factor = (1.0 - decay_factor).powf(t);
                    let params = end_size.params_millions
                        + (start_size.params_millions - end_size.params_millions) * factor;
                    let layers = ((end_size.num_layers as f64
                        + (start_size.num_layers as f64 - end_size.num_layers as f64) * factor)
                        .round()) as usize;
                    let hidden = ((end_size.hidden_dim as f64
                        + (start_size.hidden_dim as f64 - end_size.hidden_dim as f64) * factor)
                        .round()) as usize;
                    sizes.push(ModelSize::new(params, layers, hidden));
                }
                sizes
            }
            Self::Custom { sizes } => sizes.clone(),
        }
    }

    /// Generate stage configurations from this scheduler.
    #[must_use]
    pub fn generate_stage_configs(&self, base_config: &StageConfig) -> Vec<StageConfig> {
        let sizes = self.generate_sizes();
        if sizes.len() < 2 {
            return Vec::new();
        }

        sizes
            .windows(2)
            .enumerate()
            .map(|(idx, window)| {
                let teacher_size = window[0];
                let student_size = window[1];
                StageConfig {
                    teacher_size,
                    student_size,
                    stage_name: Some(format!("Stage {}", idx + 1)),
                    ..base_config.clone()
                }
            })
            .collect()
    }
}

impl Default for ProgressiveScheduler {
    fn default() -> Self {
        Self::Linear {
            start_size: ModelSize::from_params(7000.0),
            end_size: ModelSize::from_params(1000.0),
            num_stages: 3,
        }
    }
}

/// Configuration for progressive distillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Explicitly defined stages (if not using scheduler).
    pub stages: Vec<StageConfig>,
    /// Scheduler for generating stages.
    pub scheduler: Option<ProgressiveScheduler>,
    /// Early stopping patience (number of epochs without improvement).
    pub early_stopping_patience: usize,
    /// Minimum improvement threshold for early stopping.
    pub min_improvement: f32,
    /// Whether to save intermediate checkpoints.
    pub save_checkpoints: bool,
    /// Checkpoint directory.
    pub checkpoint_dir: Option<String>,
    /// Maximum number of stages to run.
    pub max_stages: Option<usize>,
    /// Whether to continue on stage failure.
    pub continue_on_failure: bool,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            stages: Vec::new(),
            scheduler: Some(ProgressiveScheduler::default()),
            early_stopping_patience: 3,
            min_improvement: 0.001,
            save_checkpoints: false,
            checkpoint_dir: None,
            max_stages: None,
            continue_on_failure: false,
        }
    }
}

impl ProgressiveConfig {
    /// Create a new configuration with explicit stages.
    #[must_use]
    pub fn with_stages(stages: Vec<StageConfig>) -> Self {
        Self {
            stages,
            scheduler: None,
            ..Default::default()
        }
    }

    /// Create a new configuration with a scheduler.
    #[must_use]
    pub fn with_scheduler(scheduler: ProgressiveScheduler) -> Self {
        Self {
            scheduler: Some(scheduler),
            ..Default::default()
        }
    }

    /// Set early stopping patience.
    #[must_use]
    pub const fn with_early_stopping(mut self, patience: usize) -> Self {
        self.early_stopping_patience = patience;
        self
    }

    /// Enable checkpoint saving.
    #[must_use]
    pub fn with_checkpoints(mut self, dir: impl Into<String>) -> Self {
        self.save_checkpoints = true;
        self.checkpoint_dir = Some(dir.into());
        self
    }

    /// Set maximum stages.
    #[must_use]
    pub const fn with_max_stages(mut self, max: usize) -> Self {
        self.max_stages = Some(max);
        self
    }

    /// Enable continuing on failure.
    #[must_use]
    pub const fn continue_on_failure(mut self, continue_on: bool) -> Self {
        self.continue_on_failure = continue_on;
        self
    }

    /// Get the effective stages (from explicit stages or scheduler).
    #[must_use]
    pub fn effective_stages(&self) -> Vec<StageConfig> {
        if !self.stages.is_empty() {
            let mut stages = self.stages.clone();
            if let Some(max) = self.max_stages {
                stages.truncate(max);
            }
            return stages;
        }

        if let Some(ref scheduler) = self.scheduler {
            let base_config = StageConfig::default();
            let mut stages = scheduler.generate_stage_configs(&base_config);
            if let Some(max) = self.max_stages {
                stages.truncate(max);
            }
            return stages;
        }

        Vec::new()
    }

    /// Validate the configuration.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let stages = self.effective_stages();
        !stages.is_empty() && stages.iter().all(StageConfig::is_valid)
    }
}

/// Progressive distillation orchestrator.
///
/// This struct manages the multi-stage distillation process,
/// running each stage sequentially and collecting results.
#[derive(Debug, Clone)]
pub struct ProgressiveDistillation {
    /// Configuration for progressive distillation.
    config: ProgressiveConfig,
    /// Explicit stages added via `add_stage`.
    stages: Vec<StageConfig>,
    /// Results from completed stages.
    results: Vec<StageResult>,
    /// Current stage index.
    current_stage: usize,
    /// Early stopping state per stage.
    early_stopping_counters: HashMap<usize, usize>,
    /// Best loss per stage for early stopping.
    best_losses: HashMap<usize, f32>,
}

impl ProgressiveDistillation {
    /// Create a new progressive distillation instance.
    #[must_use]
    pub fn new(config: ProgressiveConfig) -> Self {
        Self {
            config,
            stages: Vec::new(),
            results: Vec::new(),
            current_stage: 0,
            early_stopping_counters: HashMap::new(),
            best_losses: HashMap::new(),
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ProgressiveConfig::default())
    }

    /// Add a distillation stage.
    pub fn add_stage(&mut self, config: StageConfig) {
        self.stages.push(config);
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ProgressiveConfig {
        &self.config
    }

    /// Get all configured stages.
    #[must_use]
    pub fn all_stages(&self) -> Vec<StageConfig> {
        if self.stages.is_empty() {
            self.config.effective_stages()
        } else {
            self.stages.clone()
        }
    }

    /// Get the number of stages.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.all_stages().len()
    }

    /// Get the current stage index.
    #[must_use]
    pub const fn current_stage(&self) -> usize {
        self.current_stage
    }

    /// Get results from completed stages.
    #[must_use]
    pub fn results(&self) -> &[StageResult] {
        &self.results
    }

    /// Check if early stopping should trigger for a stage.
    fn should_early_stop(&mut self, stage_idx: usize, current_loss: f32) -> bool {
        let patience = self.config.early_stopping_patience;
        if patience == 0 {
            return false;
        }

        let best_loss = self.best_losses.entry(stage_idx).or_insert(f32::MAX);
        let counter = self.early_stopping_counters.entry(stage_idx).or_insert(0);

        if current_loss < *best_loss - self.config.min_improvement {
            *best_loss = current_loss;
            *counter = 0;
            false
        } else {
            *counter += 1;
            *counter >= patience
        }
    }

    /// Reset early stopping state for a stage.
    fn reset_early_stopping(&mut self, stage_idx: usize) {
        self.early_stopping_counters.remove(&stage_idx);
        self.best_losses.remove(&stage_idx);
    }

    /// Run a single stage of distillation.
    ///
    /// # Errors
    ///
    /// Returns an error if the stage configuration is invalid or training fails.
    #[cfg(feature = "distillation")]
    pub fn run_stage(
        &mut self,
        stage_idx: usize,
        data: &[TrainingExample],
    ) -> Result<StageResult, OxiRagError> {
        // data is used for validation; in real impl it would be used for training
        let _ = data;
        let stages = self.all_stages();
        if stage_idx >= stages.len() {
            return Err(DistillationError::InvalidConfig(format!(
                "Stage index {} out of range (max: {})",
                stage_idx,
                stages.len().saturating_sub(1)
            ))
            .into());
        }

        let stage_config = &stages[stage_idx];
        if !stage_config.is_valid() {
            return Err(DistillationError::InvalidConfig(
                "Invalid stage configuration".to_string(),
            )
            .into());
        }

        if data.is_empty() {
            return Err(DistillationError::CollectionFailed(
                "No training data provided".to_string(),
            )
            .into());
        }

        self.reset_early_stopping(stage_idx);

        // Simulate training (in real implementation, this would call actual training)
        let mut training_history = Vec::new();
        let mut current_loss = 1.0_f32;
        let start_time = std::time::Instant::now();

        for epoch in 1..=stage_config.num_epochs {
            let epoch_start = std::time::Instant::now();

            // Simulate loss decrease
            current_loss *= 0.9;

            let epoch_metrics = EpochMetrics::new(epoch, current_loss, stage_config.learning_rate)
                .with_val_loss(current_loss * 1.1)
                .with_train_accuracy(1.0 - current_loss)
                .with_val_accuracy(1.0 - current_loss * 1.1)
                .with_duration(epoch_start.elapsed().as_secs_f64());

            training_history.push(epoch_metrics);

            if self.should_early_stop(stage_idx, current_loss) {
                break;
            }
        }

        let total_duration = start_time.elapsed().as_secs_f64();
        let compression = stage_config.compression_ratio();

        let result = StageResult {
            stage_idx,
            final_loss: current_loss,
            accuracy: Some(1.0 - current_loss),
            compression_achieved: compression,
            training_history,
            total_duration_secs: total_duration,
            success: true,
            error_message: None,
        };

        // Store result
        if stage_idx >= self.results.len() {
            self.results.push(result.clone());
        } else {
            self.results[stage_idx] = result.clone();
        }

        self.current_stage = stage_idx + 1;

        Ok(result)
    }

    /// Run all stages of distillation.
    ///
    /// # Errors
    ///
    /// Returns an error if any stage fails (unless `continue_on_failure` is set).
    #[cfg(feature = "distillation")]
    pub fn run_all(&mut self, data: &[TrainingExample]) -> Result<ProgressiveResult, OxiRagError> {
        let num_stages = self.num_stages();
        let mut stage_results = Vec::new();

        for stage_idx in 0..num_stages {
            match self.run_stage(stage_idx, data) {
                Ok(result) => {
                    stage_results.push(result);
                }
                Err(e) => {
                    let failed_result = StageResult::failure(stage_idx, e.to_string());
                    stage_results.push(failed_result);

                    if !self.config.continue_on_failure {
                        return Ok(ProgressiveResult::from_stages(stage_results));
                    }
                }
            }
        }

        Ok(ProgressiveResult::from_stages(stage_results))
    }

    /// Reset the distillation state.
    pub fn reset(&mut self) {
        self.results.clear();
        self.current_stage = 0;
        self.early_stopping_counters.clear();
        self.best_losses.clear();
    }
}

impl Default for ProgressiveDistillation {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Mock implementation for testing progressive distillation.
#[derive(Debug, Clone)]
pub struct MockProgressiveDistillation {
    /// Internal progressive distillation instance.
    inner: ProgressiveDistillation,
    /// Whether to simulate failures.
    simulate_failure: bool,
    /// Stage at which to simulate failure.
    failure_stage: Option<usize>,
    /// Custom loss values per stage.
    custom_losses: HashMap<usize, f32>,
}

impl MockProgressiveDistillation {
    /// Create a new mock instance.
    #[must_use]
    pub fn new(config: ProgressiveConfig) -> Self {
        Self {
            inner: ProgressiveDistillation::new(config),
            simulate_failure: false,
            failure_stage: None,
            custom_losses: HashMap::new(),
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ProgressiveConfig::default())
    }

    /// Configure to simulate failure at a specific stage.
    #[must_use]
    pub const fn with_simulated_failure_at(mut self, stage: usize) -> Self {
        self.simulate_failure = true;
        self.failure_stage = Some(stage);
        self
    }

    /// Set custom loss for a stage.
    #[must_use]
    pub fn with_custom_loss(mut self, stage: usize, loss: f32) -> Self {
        self.custom_losses.insert(stage, loss);
        self
    }

    /// Add a distillation stage.
    pub fn add_stage(&mut self, config: StageConfig) {
        self.inner.add_stage(config);
    }

    /// Get the inner distillation instance.
    #[must_use]
    pub fn inner(&self) -> &ProgressiveDistillation {
        &self.inner
    }

    /// Get a mutable reference to the inner instance.
    pub fn inner_mut(&mut self) -> &mut ProgressiveDistillation {
        &mut self.inner
    }

    /// Run a single stage with mock behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if configured to simulate failure at this stage.
    #[cfg(feature = "distillation")]
    pub fn run_stage(
        &mut self,
        stage_idx: usize,
        _data: &[TrainingExample],
    ) -> Result<StageResult, OxiRagError> {
        if self.simulate_failure && self.failure_stage == Some(stage_idx) {
            return Err(DistillationError::TrackingFailed(format!(
                "Simulated failure at stage {stage_idx}"
            ))
            .into());
        }

        let stages = self.inner.all_stages();
        if stage_idx >= stages.len() {
            return Err(DistillationError::InvalidConfig(format!(
                "Stage index {stage_idx} out of range"
            ))
            .into());
        }

        let stage_config = &stages[stage_idx];

        // Generate mock training history
        let custom_loss = self.custom_losses.get(&stage_idx).copied();
        let mut training_history = Vec::new();
        let mut current_loss = custom_loss.unwrap_or(1.0_f32);

        for epoch in 1..=stage_config.num_epochs {
            current_loss *= 0.85;
            training_history.push(
                EpochMetrics::new(epoch, current_loss, stage_config.learning_rate)
                    .with_val_loss(current_loss * 1.05)
                    .with_train_accuracy(1.0 - current_loss)
                    .with_val_accuracy(1.0 - current_loss * 1.05)
                    .with_duration(0.1),
            );
        }

        let result = StageResult::success(
            stage_idx,
            current_loss,
            stage_config.compression_ratio(),
            training_history,
        );

        if stage_idx >= self.inner.results.len() {
            self.inner.results.push(result.clone());
        } else {
            self.inner.results[stage_idx] = result.clone();
        }

        self.inner.current_stage = stage_idx + 1;

        Ok(result)
    }

    /// Run all stages with mock behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if configured to simulate failure and `continue_on_failure` is false.
    #[cfg(feature = "distillation")]
    pub fn run_all(&mut self, data: &[TrainingExample]) -> Result<ProgressiveResult, OxiRagError> {
        let num_stages = self.inner.num_stages();
        let mut stage_results = Vec::new();

        for stage_idx in 0..num_stages {
            match self.run_stage(stage_idx, data) {
                Ok(result) => {
                    stage_results.push(result);
                }
                Err(e) => {
                    let failed_result = StageResult::failure(stage_idx, e.to_string());
                    stage_results.push(failed_result);

                    if !self.inner.config.continue_on_failure {
                        return Ok(ProgressiveResult::from_stages(stage_results));
                    }
                }
            }
        }

        Ok(ProgressiveResult::from_stages(stage_results))
    }

    /// Reset the mock state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl Default for MockProgressiveDistillation {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
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
        let student = ModelSize::from_params(7000.0); // Student larger than teacher
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
        assert!((metrics.val_loss.unwrap() - 0.6).abs() < f32::EPSILON);
        assert!((metrics.train_accuracy.unwrap() - 0.8).abs() < f32::EPSILON);
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

        assert!((result.best_val_loss().unwrap() - 0.35).abs() < f32::EPSILON);
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
        // Exponential decay means faster reduction early
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
        let scheduler = ProgressiveScheduler::custom(custom_sizes.clone());

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
        use super::*;

        #[test]
        fn test_progressive_distillation_run_stage() {
            let mut pd = ProgressiveDistillation::with_defaults();
            let data = vec![TrainingExample {
                input: "test input".to_string(),
                output: "test output".to_string(),
                confidence: 0.9,
            }];

            let result = pd.run_stage(0, &data).unwrap();

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

            let result = pd.run_all(&data).unwrap();

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

            let result = mock.run_stage(0, &data).unwrap();

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

            let result = mock.run_stage(0, &data).unwrap();

            // Final loss should be less than custom starting loss after training
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

            let result = mock.run_all(&data).unwrap();

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

            let result = mock.run_all(&data).unwrap();

            assert!(!result.all_stages_success);
            assert_eq!(result.stages_completed, 1);
            // Only 2 stages because failure stops execution
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

            let result = mock.run_all(&data).unwrap();

            assert!(!result.all_stages_success);
            // All 3 stages attempted due to continue_on_failure
            assert_eq!(result.stage_results.len(), 3);
            // Stages 0 and 2 succeeded, stage 1 failed
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

            let result = pd.run_stage(0, &data).unwrap();

            // Early stopping may have kicked in before all 10 epochs
            assert!(result.training_history.len() <= 10);
        }
    }
}
