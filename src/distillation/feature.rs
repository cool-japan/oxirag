//! Feature distillation (intermediate layer matching) module.
//!
//! This module provides feature distillation capabilities that match intermediate
//! layer representations between teacher and student models.
//!
//! # Overview
//!
//! Feature distillation transfers knowledge from a teacher model to a student
//! model by matching intermediate layer representations, not just the final
//! output. This can lead to better student model performance by preserving
//! the internal knowledge structure of the teacher.
//!
//! # Key Components
//!
//! - [`FeatureDistillation`]: Main struct for managing layer mappings and computing losses
//! - [`LayerMapping`]: Maps teacher layers to student layers with optional projections
//! - [`ProjectionType`]: Handles dimension mismatches between layers
//! - [`FeatureLoss`]: Different loss functions for feature matching
//! - [`AttentionTransfer`]: Attention-based knowledge transfer

use serde::{Deserialize, Serialize};

/// Configuration for feature distillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDistillationConfig {
    /// Layer mappings from teacher to student.
    pub layer_mappings: Vec<LayerMapping>,
    /// Feature loss function to use.
    pub feature_loss: FeatureLoss,
    /// Weight for attention transfer loss (0.0 to disable).
    pub attention_weight: f32,
    /// Temperature for softmax in attention transfer.
    pub temperature: f32,
    /// Whether to normalize features before computing loss.
    pub normalize_features: bool,
}

impl Default for FeatureDistillationConfig {
    fn default() -> Self {
        Self {
            layer_mappings: Vec::new(),
            feature_loss: FeatureLoss::MSE,
            attention_weight: 0.0,
            temperature: 1.0,
            normalize_features: true,
        }
    }
}

impl FeatureDistillationConfig {
    /// Create a new configuration with the specified feature loss.
    #[must_use]
    pub fn with_loss(feature_loss: FeatureLoss) -> Self {
        Self {
            feature_loss,
            ..Default::default()
        }
    }

    /// Set the attention weight.
    #[must_use]
    pub fn with_attention_weight(mut self, weight: f32) -> Self {
        self.attention_weight = weight.max(0.0);
        self
    }

    /// Set the temperature for softmax operations.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(0.001);
        self
    }

    /// Set whether to normalize features.
    #[must_use]
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize_features = normalize;
        self
    }

    /// Add a layer mapping.
    #[must_use]
    pub fn with_mapping(mut self, mapping: LayerMapping) -> Self {
        self.layer_mappings.push(mapping);
        self
    }

    /// Add multiple layer mappings.
    #[must_use]
    pub fn with_mappings(mut self, mappings: Vec<LayerMapping>) -> Self {
        self.layer_mappings.extend(mappings);
        self
    }

    /// Check if the configuration is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.temperature > 0.0 && self.attention_weight >= 0.0
    }
}

/// Mapping between teacher and student layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMapping {
    /// Index of the teacher layer to match.
    pub teacher_layer_idx: usize,
    /// Index of the student layer to train.
    pub student_layer_idx: usize,
    /// Projection to handle dimension mismatch.
    pub projection: Option<ProjectionType>,
    /// Importance weight for this layer mapping.
    pub weight: f32,
}

impl LayerMapping {
    /// Create a new layer mapping with default weight of 1.0.
    #[must_use]
    pub fn new(teacher_layer_idx: usize, student_layer_idx: usize) -> Self {
        Self {
            teacher_layer_idx,
            student_layer_idx,
            projection: None,
            weight: 1.0,
        }
    }

    /// Create a layer mapping with a projection.
    #[must_use]
    pub fn with_projection(
        teacher_layer_idx: usize,
        student_layer_idx: usize,
        projection: ProjectionType,
    ) -> Self {
        Self {
            teacher_layer_idx,
            student_layer_idx,
            projection: Some(projection),
            weight: 1.0,
        }
    }

    /// Set the importance weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.0);
        self
    }

    /// Set the projection type.
    #[must_use]
    pub fn with_projection_type(mut self, projection: ProjectionType) -> Self {
        self.projection = Some(projection);
        self
    }

    /// Remove the projection (use identity).
    #[must_use]
    pub fn without_projection(mut self) -> Self {
        self.projection = None;
        self
    }

    /// Check if this mapping requires a projection.
    #[must_use]
    pub fn requires_projection(&self) -> bool {
        self.projection.is_some()
    }
}

/// Projection types to handle dimension mismatches between teacher and student layers.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum ProjectionType {
    /// Identity projection (no transformation).
    #[default]
    Identity,
    /// Linear projection from input dimension to output dimension.
    Linear {
        /// Input dimension.
        in_dim: usize,
        /// Output dimension.
        out_dim: usize,
    },
    /// MLP projection with a hidden layer.
    MLP {
        /// Input dimension.
        in_dim: usize,
        /// Hidden layer dimension.
        hidden_dim: usize,
        /// Output dimension.
        out_dim: usize,
    },
}

impl ProjectionType {
    /// Create a linear projection.
    #[must_use]
    pub fn linear(in_dim: usize, out_dim: usize) -> Self {
        Self::Linear { in_dim, out_dim }
    }

    /// Create an MLP projection.
    #[must_use]
    pub fn mlp(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self::MLP {
            in_dim,
            hidden_dim,
            out_dim,
        }
    }

    /// Check if this is an identity projection.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }

    /// Get the input dimension (None for Identity).
    #[must_use]
    pub fn input_dim(&self) -> Option<usize> {
        match self {
            Self::Identity => None,
            Self::Linear { in_dim, .. } | Self::MLP { in_dim, .. } => Some(*in_dim),
        }
    }

    /// Get the output dimension (None for Identity).
    #[must_use]
    pub fn output_dim(&self) -> Option<usize> {
        match self {
            Self::Identity => None,
            Self::Linear { out_dim, .. } | Self::MLP { out_dim, .. } => Some(*out_dim),
        }
    }

    /// Get the number of trainable parameters.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Linear { in_dim, out_dim } => in_dim * out_dim + out_dim, // weights + bias
            Self::MLP {
                in_dim,
                hidden_dim,
                out_dim,
            } => {
                // First layer: in_dim * hidden_dim + hidden_dim
                // Second layer: hidden_dim * out_dim + out_dim
                (in_dim * hidden_dim + hidden_dim) + (hidden_dim * out_dim + out_dim)
            }
        }
    }
}

/// Feature loss functions for matching layer representations.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeatureLoss {
    /// Mean Squared Error loss.
    #[default]
    MSE,
    /// Cosine similarity loss (1 - `cosine_similarity`).
    Cosine,
    /// L1 (Manhattan) distance loss.
    L1,
    /// Attention transfer loss using Gram matrices.
    Attention,
}

impl FeatureLoss {
    /// Compute the loss between two feature vectors.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, teacher: &[f32], student: &[f32]) -> f32 {
        if teacher.is_empty() || student.is_empty() {
            return 0.0;
        }

        // Handle dimension mismatch by truncating to smaller size
        let len = teacher.len().min(student.len());
        let teacher = &teacher[..len];
        let student = &student[..len];

        match self {
            Self::MSE => {
                let sum: f32 = teacher
                    .iter()
                    .zip(student.iter())
                    .map(|(t, s)| (t - s).powi(2))
                    .sum();
                sum / len as f32
            }
            Self::Cosine => {
                let dot: f32 = teacher.iter().zip(student.iter()).map(|(t, s)| t * s).sum();
                let norm_t: f32 = teacher.iter().map(|t| t.powi(2)).sum::<f32>().sqrt();
                let norm_s: f32 = student.iter().map(|s| s.powi(2)).sum::<f32>().sqrt();

                if norm_t < f32::EPSILON || norm_s < f32::EPSILON {
                    return 1.0; // Maximum dissimilarity
                }

                let cosine_sim = dot / (norm_t * norm_s);
                1.0 - cosine_sim.clamp(-1.0, 1.0)
            }
            Self::L1 => {
                let sum: f32 = teacher
                    .iter()
                    .zip(student.iter())
                    .map(|(t, s)| (t - s).abs())
                    .sum();
                sum / len as f32
            }
            Self::Attention => {
                // For attention loss, we need to compute attention maps
                // This is a simplified version - actual implementation would use
                // spatial attention maps
                let teacher_attn = compute_spatial_attention(teacher);
                let student_attn = compute_spatial_attention(student);

                let sum: f32 = teacher_attn
                    .iter()
                    .zip(student_attn.iter())
                    .map(|(t, s)| (t - s).powi(2))
                    .sum();

                if teacher_attn.is_empty() {
                    0.0
                } else {
                    sum / teacher_attn.len() as f32
                }
            }
        }
    }

    /// Get a human-readable name for this loss type.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::MSE => "Mean Squared Error",
            Self::Cosine => "Cosine Similarity",
            Self::L1 => "L1 Distance",
            Self::Attention => "Attention Transfer",
        }
    }
}

/// Compute spatial attention by normalizing values.
fn compute_spatial_attention(features: &[f32]) -> Vec<f32> {
    if features.is_empty() {
        return Vec::new();
    }

    let sum: f32 = features.iter().map(|f| f.abs()).sum();
    if sum < f32::EPSILON {
        return vec![0.0; features.len()];
    }

    features.iter().map(|f| f.abs() / sum).collect()
}

/// Attention transfer helper for computing attention-based knowledge transfer.
#[derive(Debug, Clone)]
pub struct AttentionTransfer {
    /// Temperature for softmax.
    temperature: f32,
    /// Whether to use spatial attention.
    use_spatial: bool,
}

impl Default for AttentionTransfer {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionTransfer {
    /// Create a new attention transfer helper with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            use_spatial: true,
        }
    }

    /// Create with a specific temperature.
    #[must_use]
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature: temperature.max(0.001),
            use_spatial: true,
        }
    }

    /// Set whether to use spatial attention.
    #[must_use]
    pub fn with_spatial(mut self, use_spatial: bool) -> Self {
        self.use_spatial = use_spatial;
        self
    }

    /// Compute the attention transfer loss between teacher and student attention maps.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_attention_loss(
        &self,
        teacher_attn: &[Vec<f32>],
        student_attn: &[Vec<f32>],
    ) -> f32 {
        if teacher_attn.is_empty() || student_attn.is_empty() {
            return 0.0;
        }

        let num_pairs = teacher_attn.len().min(student_attn.len());
        if num_pairs == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for (t_attn, s_attn) in teacher_attn.iter().zip(student_attn.iter()) {
            total_loss += self.compute_single_attention_loss(t_attn, s_attn);
        }

        total_loss / num_pairs as f32
    }

    /// Compute attention loss for a single pair.
    fn compute_single_attention_loss(&self, teacher: &[f32], student: &[f32]) -> f32 {
        if teacher.is_empty() || student.is_empty() {
            return 0.0;
        }

        // Apply softmax with temperature to get attention distributions
        let teacher_prob = self.softmax_with_temperature(teacher);
        let student_prob = self.softmax_with_temperature(student);

        // Compute KL divergence or L2 distance
        let len = teacher_prob.len().min(student_prob.len());
        if len == 0 {
            return 0.0;
        }

        #[allow(clippy::cast_precision_loss)]
        let loss: f32 = teacher_prob[..len]
            .iter()
            .zip(student_prob[..len].iter())
            .map(|(t, s)| (t - s).powi(2))
            .sum::<f32>()
            / len as f32;

        loss
    }

    /// Apply softmax with temperature scaling.
    #[allow(clippy::cast_precision_loss)]
    fn softmax_with_temperature(&self, values: &[f32]) -> Vec<f32> {
        if values.is_empty() {
            return Vec::new();
        }

        // Scale by temperature
        let scaled: Vec<f32> = values.iter().map(|v| v / self.temperature).collect();

        // Find max for numerical stability
        let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let exp_vals: Vec<f32> = scaled.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        if sum < f32::EPSILON {
            return vec![1.0 / values.len() as f32; values.len()];
        }

        exp_vals.iter().map(|v| v / sum).collect()
    }

    /// Compute Gram matrix for a set of features.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_gram_matrix(&self, features: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if features.is_empty() {
            return Vec::new();
        }

        let n = features.len();
        let mut gram = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                gram[i][j] = self.dot_product(&features[i], &features[j]);
            }
        }

        // Normalize by number of elements
        let feature_dim = features.first().map_or(1, |f| f.len().max(1));
        let norm_factor = 1.0 / (n * feature_dim) as f32;

        for row in &mut gram {
            for val in row {
                *val *= norm_factor;
            }
        }

        gram
    }

    /// Compute dot product of two vectors.
    #[allow(clippy::unused_self)]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        a[..len]
            .iter()
            .zip(b[..len].iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    /// Compute Gram matrix loss between teacher and student.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_gram_loss(
        &self,
        teacher_features: &[Vec<f32>],
        student_features: &[Vec<f32>],
    ) -> f32 {
        let teacher_gram = self.compute_gram_matrix(teacher_features);
        let student_gram = self.compute_gram_matrix(student_features);

        if teacher_gram.is_empty() || student_gram.is_empty() {
            return 0.0;
        }

        let n = teacher_gram.len().min(student_gram.len());
        if n == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        for i in 0..n {
            let row_len = teacher_gram[i].len().min(student_gram[i].len());
            for j in 0..row_len {
                total_loss += (teacher_gram[i][j] - student_gram[i][j]).powi(2);
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total_loss / count as f32
        }
    }
}

/// Main feature distillation struct for managing layer mappings and computing losses.
#[derive(Debug, Clone)]
pub struct FeatureDistillation {
    /// Layer mappings from teacher to student.
    layer_mappings: Vec<LayerMapping>,
    /// Feature loss function.
    feature_loss: FeatureLoss,
    /// Attention transfer helper.
    attention_transfer: AttentionTransfer,
    /// Weight for attention loss.
    attention_weight: f32,
    /// Whether to normalize features.
    normalize_features: bool,
}

impl Default for FeatureDistillation {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureDistillation {
    /// Create a new feature distillation instance with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layer_mappings: Vec::new(),
            feature_loss: FeatureLoss::MSE,
            attention_transfer: AttentionTransfer::new(),
            attention_weight: 0.0,
            normalize_features: true,
        }
    }

    /// Create from a configuration.
    #[must_use]
    pub fn from_config(config: &FeatureDistillationConfig) -> Self {
        Self {
            layer_mappings: config.layer_mappings.clone(),
            feature_loss: config.feature_loss,
            attention_transfer: AttentionTransfer::with_temperature(config.temperature),
            attention_weight: config.attention_weight,
            normalize_features: config.normalize_features,
        }
    }

    /// Set the feature loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: FeatureLoss) -> Self {
        self.feature_loss = loss;
        self
    }

    /// Set the attention weight.
    #[must_use]
    pub fn with_attention_weight(mut self, weight: f32) -> Self {
        self.attention_weight = weight.max(0.0);
        self
    }

    /// Set whether to normalize features.
    #[must_use]
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize_features = normalize;
        self
    }

    /// Add a layer mapping from teacher to student.
    pub fn add_layer_mapping(&mut self, teacher_layer: usize, student_layer: usize) {
        self.layer_mappings
            .push(LayerMapping::new(teacher_layer, student_layer));
    }

    /// Add a layer mapping with a projection.
    pub fn add_layer_mapping_with_projection(
        &mut self,
        teacher_layer: usize,
        student_layer: usize,
        projection: ProjectionType,
    ) {
        self.layer_mappings.push(LayerMapping::with_projection(
            teacher_layer,
            student_layer,
            projection,
        ));
    }

    /// Add a layer mapping with weight.
    pub fn add_weighted_layer_mapping(
        &mut self,
        teacher_layer: usize,
        student_layer: usize,
        weight: f32,
    ) {
        self.layer_mappings
            .push(LayerMapping::new(teacher_layer, student_layer).with_weight(weight));
    }

    /// Get the layer mappings.
    #[must_use]
    pub fn layer_mappings(&self) -> &[LayerMapping] {
        &self.layer_mappings
    }

    /// Get the number of layer mappings.
    #[must_use]
    pub fn mapping_count(&self) -> usize {
        self.layer_mappings.len()
    }

    /// Clear all layer mappings.
    pub fn clear_mappings(&mut self) {
        self.layer_mappings.clear();
    }

    /// Compute the feature loss between teacher and student features.
    ///
    /// Each inner `Vec<f32>` represents features from one layer.
    /// The index in the outer slice corresponds to the layer mapping index.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_feature_loss(
        &self,
        teacher_features: &[Vec<f32>],
        student_features: &[Vec<f32>],
    ) -> f32 {
        if self.layer_mappings.is_empty()
            || teacher_features.is_empty()
            || student_features.is_empty()
        {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut total_weight = 0.0;

        for (i, mapping) in self.layer_mappings.iter().enumerate() {
            // Get features for this mapping
            let teacher_feat = teacher_features.get(i);
            let student_feat = student_features.get(i);

            if let (Some(t_feat), Some(s_feat)) = (teacher_feat, student_feat) {
                // Optionally normalize features
                let (t_normalized, s_normalized) = if self.normalize_features {
                    (normalize_vector(t_feat), normalize_vector(s_feat))
                } else {
                    (t_feat.clone(), s_feat.clone())
                };

                // Compute loss
                let loss = self.feature_loss.compute(&t_normalized, &s_normalized);
                total_loss += loss * mapping.weight;
                total_weight += mapping.weight;
            }
        }

        if total_weight < f32::EPSILON {
            0.0
        } else {
            total_loss / total_weight
        }
    }

    /// Compute combined loss including attention transfer.
    #[must_use]
    pub fn compute_total_loss(
        &self,
        teacher_features: &[Vec<f32>],
        student_features: &[Vec<f32>],
        teacher_attention: Option<&[Vec<f32>]>,
        student_attention: Option<&[Vec<f32>]>,
    ) -> f32 {
        let feature_loss = self.compute_feature_loss(teacher_features, student_features);

        let attention_loss = if self.attention_weight > 0.0 {
            match (teacher_attention, student_attention) {
                (Some(t_attn), Some(s_attn)) => self
                    .attention_transfer
                    .compute_attention_loss(t_attn, s_attn),
                _ => 0.0,
            }
        } else {
            0.0
        };

        feature_loss + self.attention_weight * attention_loss
    }

    /// Get the feature loss type.
    #[must_use]
    pub fn loss_type(&self) -> FeatureLoss {
        self.feature_loss
    }

    /// Get the attention weight.
    #[must_use]
    pub fn attention_weight(&self) -> f32 {
        self.attention_weight
    }

    /// Export configuration.
    #[must_use]
    pub fn to_config(&self) -> FeatureDistillationConfig {
        FeatureDistillationConfig {
            layer_mappings: self.layer_mappings.clone(),
            feature_loss: self.feature_loss,
            attention_weight: self.attention_weight,
            temperature: self.attention_transfer.temperature,
            normalize_features: self.normalize_features,
        }
    }
}

/// Normalize a vector to unit length.
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }

    let norm: f32 = v.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if norm < f32::EPSILON {
        return vec![0.0; v.len()];
    }

    v.iter().map(|x| x / norm).collect()
}

/// A mock feature distillation implementation for testing.
#[derive(Debug, Clone, Default)]
pub struct MockFeatureDistillation {
    /// Simulated loss value to return.
    simulated_loss: f32,
    /// Number of times compute was called.
    compute_count: usize,
    /// Layer mappings for testing.
    layer_mappings: Vec<LayerMapping>,
}

impl MockFeatureDistillation {
    /// Create a new mock instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the simulated loss value.
    #[must_use]
    pub fn with_simulated_loss(mut self, loss: f32) -> Self {
        self.simulated_loss = loss;
        self
    }

    /// Add a layer mapping.
    pub fn add_layer_mapping(&mut self, teacher_layer: usize, student_layer: usize) {
        self.layer_mappings
            .push(LayerMapping::new(teacher_layer, student_layer));
    }

    /// Get the number of times compute was called.
    #[must_use]
    pub fn compute_count(&self) -> usize {
        self.compute_count
    }

    /// Compute feature loss (returns simulated value).
    #[allow(clippy::unused_self)]
    pub fn compute_feature_loss(
        &mut self,
        _teacher_features: &[Vec<f32>],
        _student_features: &[Vec<f32>],
    ) -> f32 {
        self.compute_count += 1;
        self.simulated_loss
    }

    /// Get the layer mappings.
    #[must_use]
    pub fn layer_mappings(&self) -> &[LayerMapping] {
        &self.layer_mappings
    }

    /// Reset the compute count.
    pub fn reset(&mut self) {
        self.compute_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // FeatureDistillationConfig tests
    #[test]
    fn test_config_default() {
        let config = FeatureDistillationConfig::default();
        assert!(config.layer_mappings.is_empty());
        assert_eq!(config.feature_loss, FeatureLoss::MSE);
        assert!((config.attention_weight - 0.0).abs() < f32::EPSILON);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_builder() {
        let config = FeatureDistillationConfig::with_loss(FeatureLoss::Cosine)
            .with_attention_weight(0.5)
            .with_temperature(2.0)
            .with_normalization(false);

        assert_eq!(config.feature_loss, FeatureLoss::Cosine);
        assert!((config.attention_weight - 0.5).abs() < f32::EPSILON);
        assert!((config.temperature - 2.0).abs() < f32::EPSILON);
        assert!(!config.normalize_features);
    }

    #[test]
    fn test_config_with_mappings() {
        let mapping = LayerMapping::new(0, 0);
        let config = FeatureDistillationConfig::default()
            .with_mapping(mapping.clone())
            .with_mappings(vec![LayerMapping::new(1, 1), LayerMapping::new(2, 2)]);

        assert_eq!(config.layer_mappings.len(), 3);
    }

    // LayerMapping tests
    #[test]
    fn test_layer_mapping_creation() {
        let mapping = LayerMapping::new(5, 3);
        assert_eq!(mapping.teacher_layer_idx, 5);
        assert_eq!(mapping.student_layer_idx, 3);
        assert!(mapping.projection.is_none());
        assert!((mapping.weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_layer_mapping_with_projection() {
        let mapping =
            LayerMapping::with_projection(2, 1, ProjectionType::linear(768, 256)).with_weight(0.5);

        assert!(mapping.requires_projection());
        assert!((mapping.weight - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_layer_mapping_without_projection() {
        let mapping =
            LayerMapping::with_projection(0, 0, ProjectionType::Identity).without_projection();

        assert!(!mapping.requires_projection());
    }

    // ProjectionType tests
    #[test]
    fn test_projection_identity() {
        let proj = ProjectionType::Identity;
        assert!(proj.is_identity());
        assert!(proj.input_dim().is_none());
        assert!(proj.output_dim().is_none());
        assert_eq!(proj.parameter_count(), 0);
    }

    #[test]
    fn test_projection_linear() {
        let proj = ProjectionType::linear(768, 256);
        assert!(!proj.is_identity());
        assert_eq!(proj.input_dim(), Some(768));
        assert_eq!(proj.output_dim(), Some(256));
        // Parameters: 768 * 256 (weights) + 256 (bias) = 196864
        assert_eq!(proj.parameter_count(), 768 * 256 + 256);
    }

    #[test]
    fn test_projection_mlp() {
        let proj = ProjectionType::mlp(768, 512, 256);
        assert!(!proj.is_identity());
        assert_eq!(proj.input_dim(), Some(768));
        assert_eq!(proj.output_dim(), Some(256));
        // First layer: 768 * 512 + 512
        // Second layer: 512 * 256 + 256
        let expected = (768 * 512 + 512) + (512 * 256 + 256);
        assert_eq!(proj.parameter_count(), expected);
    }

    // FeatureLoss tests
    #[test]
    fn test_feature_loss_mse() {
        let loss = FeatureLoss::MSE;
        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![1.5, 2.5, 3.5];

        let result = loss.compute(&teacher, &student);
        // MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.75 / 3 = 0.25
        assert!((result - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_feature_loss_cosine() {
        let loss = FeatureLoss::Cosine;

        // Identical vectors should have loss close to 0
        let teacher = vec![1.0, 0.0, 0.0];
        let student = vec![1.0, 0.0, 0.0];
        let result = loss.compute(&teacher, &student);
        assert!(result < 0.001);

        // Orthogonal vectors should have loss close to 1
        let teacher = vec![1.0, 0.0, 0.0];
        let student = vec![0.0, 1.0, 0.0];
        let result = loss.compute(&teacher, &student);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_feature_loss_l1() {
        let loss = FeatureLoss::L1;
        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![2.0, 3.0, 4.0];

        let result = loss.compute(&teacher, &student);
        // L1 = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_feature_loss_empty() {
        let loss = FeatureLoss::MSE;
        assert!((loss.compute(&[], &[]) - 0.0).abs() < f32::EPSILON);
        assert!((loss.compute(&[1.0], &[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feature_loss_name() {
        assert_eq!(FeatureLoss::MSE.name(), "Mean Squared Error");
        assert_eq!(FeatureLoss::Cosine.name(), "Cosine Similarity");
        assert_eq!(FeatureLoss::L1.name(), "L1 Distance");
        assert_eq!(FeatureLoss::Attention.name(), "Attention Transfer");
    }

    // AttentionTransfer tests
    #[test]
    fn test_attention_transfer_creation() {
        let at = AttentionTransfer::new();
        assert!((at.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attention_transfer_with_temperature() {
        let at = AttentionTransfer::with_temperature(2.0);
        assert!((at.temperature - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attention_transfer_loss() {
        let at = AttentionTransfer::new();
        let teacher_attn = vec![vec![0.1, 0.2, 0.7], vec![0.3, 0.3, 0.4]];
        let student_attn = vec![vec![0.2, 0.3, 0.5], vec![0.25, 0.35, 0.4]];

        let loss = at.compute_attention_loss(&teacher_attn, &student_attn);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_attention_transfer_empty() {
        let at = AttentionTransfer::new();
        assert!((at.compute_attention_loss(&[], &[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gram_matrix() {
        let at = AttentionTransfer::new();
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let gram = at.compute_gram_matrix(&features);
        assert_eq!(gram.len(), 2);
        assert_eq!(gram[0].len(), 2);
    }

    #[test]
    fn test_gram_loss() {
        let at = AttentionTransfer::new();
        let teacher = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let student = vec![vec![1.1, 2.1], vec![3.1, 4.1]];

        let loss = at.compute_gram_loss(&teacher, &student);
        assert!(loss >= 0.0);
    }

    // FeatureDistillation tests
    #[test]
    fn test_feature_distillation_creation() {
        let fd = FeatureDistillation::new();
        assert!(fd.layer_mappings().is_empty());
        assert_eq!(fd.loss_type(), FeatureLoss::MSE);
    }

    #[test]
    fn test_feature_distillation_from_config() {
        let config = FeatureDistillationConfig::with_loss(FeatureLoss::Cosine)
            .with_attention_weight(0.3)
            .with_mapping(LayerMapping::new(0, 0));

        let fd = FeatureDistillation::from_config(&config);
        assert_eq!(fd.loss_type(), FeatureLoss::Cosine);
        assert!((fd.attention_weight() - 0.3).abs() < f32::EPSILON);
        assert_eq!(fd.mapping_count(), 1);
    }

    #[test]
    fn test_feature_distillation_add_mapping() {
        let mut fd = FeatureDistillation::new();
        fd.add_layer_mapping(0, 0);
        fd.add_layer_mapping(2, 1);
        fd.add_weighted_layer_mapping(4, 2, 0.5);

        assert_eq!(fd.mapping_count(), 3);
        assert_eq!(fd.layer_mappings()[0].teacher_layer_idx, 0);
        assert_eq!(fd.layer_mappings()[1].teacher_layer_idx, 2);
        assert!((fd.layer_mappings()[2].weight - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feature_distillation_add_mapping_with_projection() {
        let mut fd = FeatureDistillation::new();
        fd.add_layer_mapping_with_projection(0, 0, ProjectionType::linear(768, 256));

        assert_eq!(fd.mapping_count(), 1);
        assert!(fd.layer_mappings()[0].requires_projection());
    }

    #[test]
    fn test_feature_distillation_clear_mappings() {
        let mut fd = FeatureDistillation::new();
        fd.add_layer_mapping(0, 0);
        fd.add_layer_mapping(1, 1);

        fd.clear_mappings();
        assert!(fd.layer_mappings().is_empty());
    }

    #[test]
    fn test_feature_distillation_compute_loss() {
        let mut fd = FeatureDistillation::new();
        fd.add_layer_mapping(0, 0);

        let teacher = vec![vec![1.0, 0.0, 0.0]];
        let student = vec![vec![0.9, 0.1, 0.0]];

        let loss = fd.compute_feature_loss(&teacher, &student);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_feature_distillation_compute_loss_empty() {
        let fd = FeatureDistillation::new();
        let loss = fd.compute_feature_loss(&[], &[]);
        assert!((loss - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feature_distillation_compute_loss_no_mappings() {
        let fd = FeatureDistillation::new();
        let teacher = vec![vec![1.0, 2.0]];
        let student = vec![vec![1.5, 2.5]];

        let loss = fd.compute_feature_loss(&teacher, &student);
        assert!((loss - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feature_distillation_compute_total_loss() {
        let mut fd = FeatureDistillation::new().with_attention_weight(0.5);
        fd.add_layer_mapping(0, 0);

        let teacher = vec![vec![1.0, 0.0]];
        let student = vec![vec![0.9, 0.1]];
        let t_attn = vec![vec![0.6, 0.4]];
        let s_attn = vec![vec![0.5, 0.5]];

        let loss = fd.compute_total_loss(&teacher, &student, Some(&t_attn), Some(&s_attn));
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_feature_distillation_to_config() {
        let mut fd = FeatureDistillation::new()
            .with_loss(FeatureLoss::L1)
            .with_attention_weight(0.2)
            .with_normalization(false);
        fd.add_layer_mapping(0, 0);

        let config = fd.to_config();
        assert_eq!(config.feature_loss, FeatureLoss::L1);
        assert!((config.attention_weight - 0.2).abs() < f32::EPSILON);
        assert!(!config.normalize_features);
        assert_eq!(config.layer_mappings.len(), 1);
    }

    // MockFeatureDistillation tests
    #[test]
    fn test_mock_feature_distillation() {
        let mut mock = MockFeatureDistillation::new().with_simulated_loss(0.5);

        let teacher = vec![vec![1.0]];
        let student = vec![vec![2.0]];

        let loss = mock.compute_feature_loss(&teacher, &student);
        assert!((loss - 0.5).abs() < f32::EPSILON);
        assert_eq!(mock.compute_count(), 1);

        let _ = mock.compute_feature_loss(&teacher, &student);
        assert_eq!(mock.compute_count(), 2);

        mock.reset();
        assert_eq!(mock.compute_count(), 0);
    }

    #[test]
    fn test_mock_add_layer_mapping() {
        let mut mock = MockFeatureDistillation::new();
        mock.add_layer_mapping(0, 0);
        mock.add_layer_mapping(2, 1);

        assert_eq!(mock.layer_mappings().len(), 2);
    }

    // Normalization tests
    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_vector(&v);

        // Expected: [3/5, 4/5] = [0.6, 0.8]
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);

        // Check it's unit length
        let norm: f32 = normalized.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = normalize_vector(&v);

        assert_eq!(normalized.len(), 3);
        for val in &normalized {
            assert!((val - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_normalize_empty_vector() {
        let v: Vec<f32> = vec![];
        let normalized = normalize_vector(&v);
        assert!(normalized.is_empty());
    }

    // Attention loss tests
    #[test]
    fn test_feature_loss_attention() {
        let loss = FeatureLoss::Attention;
        let teacher = vec![0.5, 0.3, 0.2];
        let student = vec![0.4, 0.4, 0.2];

        let result = loss.compute(&teacher, &student);
        assert!(result >= 0.0);
    }

    // Edge case tests
    #[test]
    fn test_dimension_mismatch_handling() {
        let loss = FeatureLoss::MSE;
        let teacher = vec![1.0, 2.0, 3.0, 4.0];
        let student = vec![1.0, 2.0];

        // Should handle by truncating to smaller size
        let result = loss.compute(&teacher, &student);
        assert!((result - 0.0).abs() < f32::EPSILON); // Identical first 2 elements
    }

    #[test]
    fn test_weighted_loss_computation() {
        let mut fd = FeatureDistillation::new();

        // Add two mappings with different weights
        fd.add_weighted_layer_mapping(0, 0, 1.0);
        fd.add_weighted_layer_mapping(1, 1, 2.0);

        let teacher = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let student = vec![vec![0.9, 0.1], vec![0.1, 0.9]];

        let loss = fd.compute_feature_loss(&teacher, &student);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_temperature_clamp() {
        let at = AttentionTransfer::with_temperature(-1.0);
        assert!(at.temperature >= 0.001);

        let at2 = AttentionTransfer::with_temperature(0.0);
        assert!(at2.temperature >= 0.001);
    }

    #[test]
    fn test_projection_default() {
        let proj = ProjectionType::default();
        assert!(proj.is_identity());
    }
}
