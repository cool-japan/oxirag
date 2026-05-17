//! Progressive distillation orchestrator and mock implementation.
//!
//! This module provides the [`ProgressiveDistillation`] struct which manages
//! the multi-stage distillation pipeline, and [`MockProgressiveDistillation`]
//! which simulates the same pipeline for unit tests without requiring real
//! model training.

use std::collections::HashMap;

use super::types::{EpochMetrics, ProgressiveConfig, ProgressiveResult, StageConfig, StageResult};

#[cfg(feature = "distillation")]
use super::super::collector::TrainingExample;

#[cfg(feature = "distillation")]
use crate::error::{DistillationError, OxiRagError};

// ────────────────────────────────────────────────────────────────────────────
// ProgressiveDistillation
// ────────────────────────────────────────────────────────────────────────────

/// Orchestrates the multi-stage progressive distillation process.
///
/// Each stage reduces model size from a teacher to a smaller student.  The
/// struct accumulates per-stage results and provides early-stopping logic that
/// halts training within a stage when validation loss stops improving.
#[derive(Debug, Clone)]
pub struct ProgressiveDistillation {
    /// Global configuration (scheduler, early-stopping, checkpoints, …).
    pub(crate) config: ProgressiveConfig,
    /// Explicit stages added via [`add_stage`].  When non-empty these take
    /// precedence over any scheduler-generated stages.
    ///
    /// [`add_stage`]: ProgressiveDistillation::add_stage
    pub(crate) stages: Vec<StageConfig>,
    /// Results from completed stages.
    pub(crate) results: Vec<StageResult>,
    /// Index of the next stage to run.
    pub(crate) current_stage: usize,
    /// Per-stage early-stopping counters (resets when loss improves).
    early_stopping_counters: HashMap<usize, usize>,
    /// Best loss seen so far per stage (for early-stopping comparison).
    best_losses: HashMap<usize, f32>,
}

impl ProgressiveDistillation {
    /// Create a new orchestrator with the given configuration.
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

    /// Create with default configuration (linear 7B→1B, 3 stages).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ProgressiveConfig::default())
    }

    /// Append an explicit stage configuration.
    pub fn add_stage(&mut self, config: StageConfig) {
        self.stages.push(config);
    }

    /// Return a reference to the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &ProgressiveConfig {
        &self.config
    }

    /// Return the effective stage list, either from explicit stages or the
    /// scheduler.
    #[must_use]
    pub fn all_stages(&self) -> Vec<StageConfig> {
        if self.stages.is_empty() {
            self.config.effective_stages()
        } else {
            self.stages.clone()
        }
    }

    /// Total number of configured stages.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.all_stages().len()
    }

    /// Index of the next stage to execute.
    #[must_use]
    pub const fn current_stage(&self) -> usize {
        self.current_stage
    }

    /// Slice of results for all completed stages.
    #[must_use]
    pub fn results(&self) -> &[StageResult] {
        &self.results
    }

    /// Return `true` if early stopping should be triggered for `stage_idx`.
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

    /// Reset early-stopping state for `stage_idx`.
    fn reset_early_stopping(&mut self, stage_idx: usize) {
        self.early_stopping_counters.remove(&stage_idx);
        self.best_losses.remove(&stage_idx);
    }

    /// Run a single distillation stage.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - `stage_idx` is out of range,
    /// - the stage configuration is invalid,
    /// - or no training data was provided.
    #[cfg(feature = "distillation")]
    pub fn run_stage(
        &mut self,
        stage_idx: usize,
        data: &[TrainingExample],
    ) -> Result<StageResult, OxiRagError> {
        // `data` would feed the actual training loop; validated but unused in
        // this simulation.
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

        let mut training_history = Vec::new();
        let mut current_loss = 1.0_f32;
        let start_time = std::time::Instant::now();

        for epoch in 1..=stage_config.num_epochs {
            let epoch_start = std::time::Instant::now();
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

        if stage_idx >= self.results.len() {
            self.results.push(result.clone());
        } else {
            self.results[stage_idx] = result.clone();
        }

        self.current_stage = stage_idx + 1;

        Ok(result)
    }

    /// Run all configured stages sequentially.
    ///
    /// If a stage fails and `continue_on_failure` is `false`, execution stops
    /// and a partial `ProgressiveResult` is returned.
    ///
    /// # Errors
    ///
    /// Returns `Ok` in all cases: stage failures are encoded inside the
    /// returned `ProgressiveResult`.  An `Err` is only returned by the inner
    /// `run_stage` and is then converted to a `StageResult::failure`.
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

    /// Reset all accumulated state (results, counters, current stage index).
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

// ────────────────────────────────────────────────────────────────────────────
// MockProgressiveDistillation
// ────────────────────────────────────────────────────────────────────────────

/// Test-friendly wrapper around [`ProgressiveDistillation`] that can simulate
/// stage failures and inject custom starting loss values.
#[derive(Debug, Clone)]
pub struct MockProgressiveDistillation {
    /// The underlying orchestrator.
    pub(crate) inner: ProgressiveDistillation,
    /// Whether to simulate a failure at `failure_stage`.
    simulate_failure: bool,
    /// Stage index at which to inject a simulated failure.
    failure_stage: Option<usize>,
    /// Custom starting loss values keyed by stage index.
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

    /// Configure the mock to return an error at `stage`.
    #[must_use]
    pub const fn with_simulated_failure_at(mut self, stage: usize) -> Self {
        self.simulate_failure = true;
        self.failure_stage = Some(stage);
        self
    }

    /// Override the starting loss for `stage`.
    #[must_use]
    pub fn with_custom_loss(mut self, stage: usize, loss: f32) -> Self {
        self.custom_losses.insert(stage, loss);
        self
    }

    /// Append an explicit stage configuration to the inner orchestrator.
    pub fn add_stage(&mut self, config: StageConfig) {
        self.inner.add_stage(config);
    }

    /// Immutable reference to the underlying [`ProgressiveDistillation`].
    #[must_use]
    pub fn inner(&self) -> &ProgressiveDistillation {
        &self.inner
    }

    /// Mutable reference to the underlying [`ProgressiveDistillation`].
    pub fn inner_mut(&mut self) -> &mut ProgressiveDistillation {
        &mut self.inner
    }

    /// Run a single stage with mock behaviour.
    ///
    /// # Errors
    ///
    /// Returns an error if the mock is configured to fail at `stage_idx` or
    /// `stage_idx` is out of range.
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

    /// Run all stages with mock behaviour.
    ///
    /// # Errors
    ///
    /// Returns `Ok` in all cases — stage failures are captured inside the
    /// returned `ProgressiveResult`.
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

    /// Reset the mock to its initial state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl Default for MockProgressiveDistillation {
    fn default() -> Self {
        Self::with_defaults()
    }
}
