//! Knowledge transfer metrics for distillation.

use serde::{Deserialize, Serialize};

/// Layer-wise similarity measurement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerSimilarity {
    /// Layer index (0-indexed).
    pub layer_index: usize,
    /// Layer name or identifier.
    pub layer_name: String,
    /// Cosine similarity between teacher and student layer outputs.
    pub cosine_similarity: f32,
    /// Mean squared error between layer outputs.
    pub mse: f32,
    /// Pearson correlation coefficient.
    pub correlation: f32,
}

impl LayerSimilarity {
    /// Create a new layer similarity measurement.
    #[must_use]
    pub fn new(layer_index: usize, layer_name: impl Into<String>) -> Self {
        Self {
            layer_index,
            layer_name: layer_name.into(),
            ..Default::default()
        }
    }

    /// Set similarity metrics.
    #[must_use]
    pub fn with_metrics(mut self, cosine_similarity: f32, mse: f32, correlation: f32) -> Self {
        self.cosine_similarity = cosine_similarity;
        self.mse = mse;
        self.correlation = correlation;
        self
    }

    /// Calculate an aggregate similarity score.
    #[must_use]
    pub fn aggregate_score(&self) -> f32 {
        // Combine cosine similarity and correlation (both higher is better)
        // MSE is inverted since lower is better
        let mse_score = 1.0 / (1.0 + self.mse);
        (self.cosine_similarity + self.correlation + mse_score) / 3.0
    }
}

/// Metrics for measuring knowledge transfer efficiency.
///
/// This type provides detailed metrics about the knowledge transfer process
/// during distillation, including layer-wise similarity measurements.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeTransferMetrics {
    /// Knowledge transfer efficiency (0.0 - 1.0).
    /// Measures how well the student learned from the teacher.
    pub knowledge_transfer_efficiency: f32,
    /// Capacity utilization (0.0 - 1.0).
    /// Measures how much of the student's capacity is being used.
    pub capacity_utilization: f32,
    /// Layer-wise similarity measurements.
    pub layer_wise_similarity: Vec<LayerSimilarity>,
    /// Average layer similarity.
    pub average_layer_similarity: f32,
    /// Training epochs completed.
    pub epochs_completed: usize,
    /// Final training loss.
    pub final_loss: f32,
}

impl KnowledgeTransferMetrics {
    /// Create new knowledge transfer metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set transfer efficiency.
    #[must_use]
    pub fn with_transfer_efficiency(mut self, efficiency: f32) -> Self {
        self.knowledge_transfer_efficiency = efficiency.clamp(0.0, 1.0);
        self
    }

    /// Set capacity utilization.
    #[must_use]
    pub fn with_capacity_utilization(mut self, utilization: f32) -> Self {
        self.capacity_utilization = utilization.clamp(0.0, 1.0);
        self
    }

    /// Set layer-wise similarity measurements.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn with_layer_similarity(mut self, similarities: Vec<LayerSimilarity>) -> Self {
        if similarities.is_empty() {
            self.average_layer_similarity = 0.0;
        } else {
            let sum: f32 = similarities
                .iter()
                .map(LayerSimilarity::aggregate_score)
                .sum();
            self.average_layer_similarity = sum / similarities.len() as f32;
        }
        self.layer_wise_similarity = similarities;
        self
    }

    /// Add a layer similarity measurement.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_layer_similarity(&mut self, similarity: LayerSimilarity) {
        self.layer_wise_similarity.push(similarity);
        // Recalculate average
        let sum: f32 = self
            .layer_wise_similarity
            .iter()
            .map(LayerSimilarity::aggregate_score)
            .sum();
        self.average_layer_similarity = sum / self.layer_wise_similarity.len() as f32;
    }

    /// Get the overall distillation quality score.
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        // Weighted combination of metrics
        let transfer_weight = 0.4;
        let capacity_weight = 0.3;
        let similarity_weight = 0.3;

        self.knowledge_transfer_efficiency * transfer_weight
            + self.capacity_utilization * capacity_weight
            + self.average_layer_similarity * similarity_weight
    }

    /// Check if distillation quality meets thresholds.
    #[must_use]
    pub fn meets_quality_threshold(&self, min_score: f32) -> bool {
        self.overall_score() >= min_score
    }
}
