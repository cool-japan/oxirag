//! Plot data and epoch metrics for distillation training visualization.

use serde::{Deserialize, Serialize};

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
