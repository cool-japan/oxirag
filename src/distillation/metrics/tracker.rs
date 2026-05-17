//! Metrics tracker for training epoch history.

use serde::{Deserialize, Serialize};

use super::plot::{PlotData, TrainingEpochMetrics};

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
