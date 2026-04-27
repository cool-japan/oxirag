//! Core types for progressive distillation.
//!
//! This module contains the data structures shared across the progressive
//! distillation subsystem: model-size descriptors, loss-weight configurations,
//! per-stage and per-epoch metrics, and the result types that accumulate
//! training outcomes.

use serde::{Deserialize, Serialize};

// ────────────────────────────────────────────────────────────────────────────
// ModelSize
// ────────────────────────────────────────────────────────────────────────────

/// Model size specification used to describe teacher and student models at each
/// distillation stage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelSize {
    /// Number of parameters in millions.
    pub params_millions: f64,
    /// Number of transformer layers.
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

    /// Create a model size from parameter count only (layers and hidden-dim
    /// are set to zero and must be filled in separately if needed).
    #[must_use]
    pub const fn from_params(params_millions: f64) -> Self {
        Self {
            params_millions,
            num_layers: 0,
            hidden_dim: 0,
        }
    }

    /// Compute the compression ratio of `self` relative to `other`.
    ///
    /// Returns `0.0` if `other.params_millions` is non-positive.
    #[must_use]
    pub fn compression_ratio(&self, other: &Self) -> f64 {
        if other.params_millions <= 0.0 {
            return 0.0;
        }
        self.params_millions / other.params_millions
    }

    /// Return `true` if this model is smaller (fewer parameters) than `other`.
    #[must_use]
    pub fn is_smaller_than(&self, other: &Self) -> bool {
        self.params_millions < other.params_millions
    }
}

impl Default for ModelSize {
    fn default() -> Self {
        Self::from_params(7000.0) // 7 B params
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LossWeights
// ────────────────────────────────────────────────────────────────────────────

/// Per-objective loss weights for a distillation training stage.
///
/// All weights must be non-negative, and at least one must be positive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossWeights {
    /// Weight for hard-label loss (cross-entropy with ground truth).
    pub hard_label: f32,
    /// Weight for soft-label loss (KL divergence with teacher logits).
    pub soft_label: f32,
    /// Weight for hidden-state alignment loss.
    pub hidden_state: f32,
    /// Weight for attention-map alignment loss.
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
    /// Use only hard (ground-truth) labels.
    #[must_use]
    pub const fn hard_only() -> Self {
        Self {
            hard_label: 1.0,
            soft_label: 0.0,
            hidden_state: 0.0,
            attention: 0.0,
        }
    }

    /// Use only soft (teacher) labels.
    #[must_use]
    pub const fn soft_only() -> Self {
        Self {
            hard_label: 0.0,
            soft_label: 1.0,
            hidden_state: 0.0,
            attention: 0.0,
        }
    }

    /// Equal weight between hard and soft labels.
    #[must_use]
    pub const fn balanced() -> Self {
        Self {
            hard_label: 0.5,
            soft_label: 0.5,
            hidden_state: 0.0,
            attention: 0.0,
        }
    }

    /// Hard, soft, and hidden-state alignment with the given weights.
    #[must_use]
    pub const fn with_hidden_states(hard: f32, soft: f32, hidden: f32) -> Self {
        Self {
            hard_label: hard,
            soft_label: soft,
            hidden_state: hidden,
            attention: 0.0,
        }
    }

    /// Return a copy with weights re-scaled so they sum to `1.0`.
    ///
    /// If the total is zero or negative the default weights are returned.
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

    /// Validate that all weights are non-negative and at least one is positive.
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

// ────────────────────────────────────────────────────────────────────────────
// StageConfig
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for a single progressive distillation stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Teacher model size for this stage.
    pub teacher_size: ModelSize,
    /// Student model size for this stage.
    pub student_size: ModelSize,
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Base learning rate.
    pub learning_rate: f64,
    /// Distillation temperature (higher = softer teacher distribution).
    pub temperature: f32,
    /// Objective weights.
    pub loss_weights: LossWeights,
    /// Training batch size.
    pub batch_size: usize,
    /// Optional human-readable stage name.
    pub stage_name: Option<String>,
    /// Linear warm-up steps for the learning-rate scheduler.
    pub warmup_steps: usize,
    /// L2 weight decay for regularisation.
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
    /// Create a new stage configuration with the given teacher/student sizes.
    #[must_use]
    pub fn new(teacher_size: ModelSize, student_size: ModelSize) -> Self {
        Self {
            teacher_size,
            student_size,
            ..Default::default()
        }
    }

    /// Set the number of training epochs.
    #[must_use]
    pub const fn with_epochs(mut self, epochs: usize) -> Self {
        self.num_epochs = epochs;
        self
    }

    /// Set the base learning rate.
    #[must_use]
    pub const fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the distillation temperature.
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

    /// Set the training batch size.
    #[must_use]
    pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Attach a human-readable stage name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.stage_name = Some(name.into());
        self
    }

    /// Compression ratio of teacher parameters relative to student parameters.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        self.teacher_size.compression_ratio(&self.student_size)
    }

    /// Return `true` if the stage configuration is coherent.
    ///
    /// Requires: student smaller than teacher, positive epochs / LR /
    /// temperature / batch size, and valid loss weights.
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

// ────────────────────────────────────────────────────────────────────────────
// EpochMetrics
// ────────────────────────────────────────────────────────────────────────────

/// Training metrics recorded at the end of a single epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// Epoch number (1-indexed).
    pub epoch: usize,
    /// Training loss.
    pub train_loss: f32,
    /// Validation loss (absent when no validation set is used).
    pub val_loss: Option<f32>,
    /// Training accuracy.
    pub train_accuracy: Option<f32>,
    /// Validation accuracy.
    pub val_accuracy: Option<f32>,
    /// Learning rate at this epoch.
    pub learning_rate: f64,
    /// Wall-clock duration of this epoch in seconds.
    pub duration_secs: f64,
}

impl EpochMetrics {
    /// Create new epoch metrics with mandatory fields.
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

    /// Attach a validation loss.
    #[must_use]
    pub const fn with_val_loss(mut self, val_loss: f32) -> Self {
        self.val_loss = Some(val_loss);
        self
    }

    /// Attach a training accuracy.
    #[must_use]
    pub const fn with_train_accuracy(mut self, accuracy: f32) -> Self {
        self.train_accuracy = Some(accuracy);
        self
    }

    /// Attach a validation accuracy.
    #[must_use]
    pub const fn with_val_accuracy(mut self, accuracy: f32) -> Self {
        self.val_accuracy = Some(accuracy);
        self
    }

    /// Set the epoch wall-clock duration.
    #[must_use]
    pub const fn with_duration(mut self, duration_secs: f64) -> Self {
        self.duration_secs = duration_secs;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StageResult
// ────────────────────────────────────────────────────────────────────────────

/// Outcome of a single progressive distillation stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    /// Zero-based index of the stage.
    pub stage_idx: usize,
    /// Final training loss (after the last epoch or early stopping).
    pub final_loss: f32,
    /// Final accuracy (from the last epoch with val accuracy, if any).
    pub accuracy: Option<f32>,
    /// Compression ratio achieved during this stage.
    pub compression_achieved: f64,
    /// Per-epoch training history.
    pub training_history: Vec<EpochMetrics>,
    /// Total wall-clock duration of the stage in seconds.
    pub total_duration_secs: f64,
    /// Whether the stage completed without error.
    pub success: bool,
    /// Human-readable error message if `success == false`.
    pub error_message: Option<String>,
}

impl StageResult {
    /// Construct a successful stage result.
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

    /// Construct a failed stage result.
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

    /// Best (lowest) validation loss across all epochs.
    #[must_use]
    pub fn best_val_loss(&self) -> Option<f32> {
        self.training_history
            .iter()
            .filter_map(|e| e.val_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Epoch number (1-indexed) with the lowest validation loss.
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

// ────────────────────────────────────────────────────────────────────────────
// ProgressiveResult
// ────────────────────────────────────────────────────────────────────────────

/// Aggregate outcome of the entire multi-stage progressive distillation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveResult {
    /// Individual stage results.
    pub stage_results: Vec<StageResult>,
    /// Final loss from the last executed stage.
    pub final_loss: f32,
    /// Product of per-stage compression ratios (only successful stages).
    pub total_compression: f64,
    /// Total wall-clock duration across all stages.
    pub total_duration_secs: f64,
    /// Number of stages that completed successfully.
    pub stages_completed: usize,
    /// Whether every configured stage succeeded.
    pub all_stages_success: bool,
}

impl ProgressiveResult {
    /// Aggregate a vector of stage results into a `ProgressiveResult`.
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

    /// Most recently completed successful stage result, if any.
    #[must_use]
    pub fn last_successful_stage(&self) -> Option<&StageResult> {
        self.stage_results.iter().rev().find(|s| s.success)
    }

    /// First failed stage result, if any.
    #[must_use]
    pub fn first_failed_stage(&self) -> Option<&StageResult> {
        self.stage_results.iter().find(|s| !s.success)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ProgressiveConfig
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the complete progressive distillation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Explicitly defined stage configurations (used when `scheduler` is `None`).
    pub stages: Vec<StageConfig>,
    /// Scheduler used to auto-generate stage configurations.
    pub scheduler: Option<super::scheduler::ProgressiveScheduler>,
    /// Number of epochs without improvement before triggering early stopping.
    pub early_stopping_patience: usize,
    /// Minimum loss improvement required to reset the early-stopping counter.
    pub min_improvement: f32,
    /// Whether to persist intermediate model checkpoints.
    pub save_checkpoints: bool,
    /// Filesystem path for checkpoint files.
    pub checkpoint_dir: Option<String>,
    /// Maximum number of stages to execute (truncates auto-generated lists).
    pub max_stages: Option<usize>,
    /// Whether to continue executing subsequent stages after a stage failure.
    pub continue_on_failure: bool,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            stages: Vec::new(),
            scheduler: Some(super::scheduler::ProgressiveScheduler::default()),
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
    /// Create a configuration with explicitly enumerated stages.
    #[must_use]
    pub fn with_stages(stages: Vec<StageConfig>) -> Self {
        Self {
            stages,
            scheduler: None,
            ..Default::default()
        }
    }

    /// Create a configuration driven by a `ProgressiveScheduler`.
    #[must_use]
    pub fn with_scheduler(scheduler: super::scheduler::ProgressiveScheduler) -> Self {
        Self {
            scheduler: Some(scheduler),
            ..Default::default()
        }
    }

    /// Set the early-stopping patience (number of non-improving epochs).
    #[must_use]
    pub const fn with_early_stopping(mut self, patience: usize) -> Self {
        self.early_stopping_patience = patience;
        self
    }

    /// Enable checkpoint saving to `dir`.
    #[must_use]
    pub fn with_checkpoints(mut self, dir: impl Into<String>) -> Self {
        self.save_checkpoints = true;
        self.checkpoint_dir = Some(dir.into());
        self
    }

    /// Cap the number of stages to `max`.
    #[must_use]
    pub const fn with_max_stages(mut self, max: usize) -> Self {
        self.max_stages = Some(max);
        self
    }

    /// Set whether to continue after a stage failure.
    #[must_use]
    pub const fn continue_on_failure(mut self, continue_on: bool) -> Self {
        self.continue_on_failure = continue_on;
        self
    }

    /// Compute the effective list of stages, respecting explicit stages,
    /// the scheduler, and `max_stages`.
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

    /// Return `true` if the configuration is coherent and will produce at
    /// least one valid stage.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let stages = self.effective_stages();
        !stages.is_empty() && stages.iter().all(StageConfig::is_valid)
    }
}
