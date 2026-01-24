//! State extraction utilities and mock providers.
//!
//! This module provides utilities for extracting hidden states from transformer
//! models, including a mock implementation for testing.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::traits::HiddenStateProvider;
use super::types::{
    HiddenStateConfig, HiddenStateTensor, LayerHiddenState, ModelHiddenStates, ModelKVCache,
    TensorShape,
};
use crate::error::HiddenStateError;

/// Mock hidden state provider for testing.
///
/// Generates deterministic hidden states based on input text hash,
/// useful for testing without requiring actual model inference.
pub struct MockHiddenStateProvider {
    config: HiddenStateConfig,
    model_id: String,
    num_layers: usize,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl MockHiddenStateProvider {
    /// Create a new mock hidden state provider.
    #[must_use]
    pub fn new(model_id: &str, num_layers: usize, hidden_dim: usize) -> Self {
        let num_heads = 12;
        let head_dim = hidden_dim / num_heads;
        Self {
            config: HiddenStateConfig::default(),
            model_id: model_id.to_string(),
            num_layers,
            hidden_dim,
            num_heads,
            head_dim,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(mut self, config: HiddenStateConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the number of attention heads.
    #[must_use]
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self.head_dim = self.hidden_dim / num_heads;
        self
    }

    /// Generate deterministic data based on a seed.
    fn generate_data(seed: u64, size: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(size);
        let mut current = seed;

        for _ in 0..size {
            // Simple LCG for deterministic pseudo-random numbers
            current = current.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let value = ((current >> 16) & 0x7FFF) as f32 / 32767.0;
            // Normalize to [-1, 1]
            data.push(value * 2.0 - 1.0);
        }

        data
    }

    /// Hash text to get a seed.
    fn text_to_seed(text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Estimate sequence length from text (simple word count approximation).
    fn estimate_seq_len(text: &str) -> usize {
        // Rough approximation: ~1.3 tokens per word
        let word_count = text.split_whitespace().count();
        (word_count as f32 * 1.3).ceil() as usize
    }
}

#[async_trait]
impl HiddenStateProvider for MockHiddenStateProvider {
    async fn extract_hidden_states(
        &self,
        text: &str,
    ) -> Result<ModelHiddenStates, HiddenStateError> {
        let seed = Self::text_to_seed(text);
        let seq_len = Self::estimate_seq_len(text).max(1);

        let mut states = ModelHiddenStates::new(&self.model_id, self.num_layers, self.hidden_dim);
        states.sequence_length = seq_len;

        for layer_idx in 0..self.num_layers {
            let layer_seed = seed.wrapping_add(layer_idx as u64);

            // Generate hidden state [1, seq_len, hidden_dim]
            let hidden_size = seq_len * self.hidden_dim;
            let hidden_data = Self::generate_data(layer_seed, hidden_size);
            let hidden_shape = TensorShape::new(vec![1, seq_len, self.hidden_dim]);
            let hidden_tensor =
                HiddenStateTensor::from_vec(hidden_data, hidden_shape).map_err(|e| {
                    HiddenStateError::ProviderError(format!("Failed to create hidden tensor: {e}"))
                })?;

            let mut layer = LayerHiddenState::new(layer_idx, hidden_tensor);

            // Generate attention weights if configured
            if self.config.capture_attention_weights {
                let attn_size = self.num_heads * seq_len * seq_len;
                let attn_data = Self::generate_data(layer_seed.wrapping_add(1000), attn_size);
                let attn_shape = TensorShape::new(vec![1, self.num_heads, seq_len, seq_len]);
                let attn_tensor =
                    HiddenStateTensor::from_vec(attn_data, attn_shape).map_err(|e| {
                        HiddenStateError::ProviderError(format!(
                            "Failed to create attention tensor: {e}"
                        ))
                    })?;
                layer = layer.with_attention_weights(attn_tensor);
            }

            states.add_layer(layer);
        }

        // Generate pooled output [1, hidden_dim]
        let pooled_data = Self::generate_data(seed.wrapping_add(10000), self.hidden_dim);
        let pooled_tensor = HiddenStateTensor::from_vec_1d(pooled_data);
        states.set_pooled_output(pooled_tensor);

        Ok(states)
    }

    async fn extract_with_kv_cache(
        &self,
        text: &str,
        past_kv: Option<&ModelKVCache>,
    ) -> Result<(ModelHiddenStates, ModelKVCache), HiddenStateError> {
        let states = self.extract_hidden_states(text).await?;
        let seq_len = states.sequence_length;

        let mut kv_cache = if let Some(past) = past_kv {
            past.clone()
        } else {
            ModelKVCache::new(
                &self.model_id,
                self.num_layers,
                self.num_heads,
                self.head_dim,
                2048,
            )
        };

        // Append new KV pairs for this text
        let seed = Self::text_to_seed(text);
        for layer_idx in 0..self.num_layers {
            let layer_seed = seed.wrapping_add(layer_idx as u64 + 5000);

            // Generate keys [1, num_heads, seq_len, head_dim]
            let kv_size = self.num_heads * seq_len * self.head_dim;
            let key_data = Self::generate_data(layer_seed, kv_size);
            let value_data = Self::generate_data(layer_seed.wrapping_add(100), kv_size);

            let shape = TensorShape::new(vec![1, self.num_heads, seq_len, self.head_dim]);
            let keys = HiddenStateTensor::from_vec(key_data, shape.clone()).map_err(|e| {
                HiddenStateError::ProviderError(format!("Failed to create keys: {e}"))
            })?;
            let values = HiddenStateTensor::from_vec(value_data, shape).map_err(|e| {
                HiddenStateError::ProviderError(format!("Failed to create values: {e}"))
            })?;

            kv_cache.append_layer(layer_idx, &keys, &values)?;
        }

        Ok((states, kv_cache))
    }

    fn model_config(&self) -> &HiddenStateConfig {
        &self.config
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

/// Utility for computing state similarity.
///
/// Provides methods for comparing hidden state tensors using various
/// distance and similarity metrics.
pub struct StateSimilarity;

impl StateSimilarity {
    /// Compute cosine similarity between two hidden state tensors.
    ///
    /// Returns a value between -1.0 and 1.0, where 1.0 indicates identical
    /// direction and -1.0 indicates opposite direction.
    #[must_use]
    pub fn cosine(a: &HiddenStateTensor, b: &HiddenStateTensor) -> f32 {
        if a.data.is_empty() || b.data.is_empty() {
            return 0.0;
        }

        let len = a.data.len().min(b.data.len());

        let dot_product: f32 = a.data[..len]
            .iter()
            .zip(b.data[..len].iter())
            .map(|(x, y)| x * y)
            .sum();

        let norm_a: f32 = a.data[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.data[..len].iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Compute L2 (Euclidean) distance between tensors.
    ///
    /// Returns the Euclidean distance between the two tensors.
    #[must_use]
    pub fn l2_distance(a: &HiddenStateTensor, b: &HiddenStateTensor) -> f32 {
        if a.data.is_empty() || b.data.is_empty() {
            return 0.0;
        }

        let len = a.data.len().min(b.data.len());

        let sum_sq: f32 = a.data[..len]
            .iter()
            .zip(b.data[..len].iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        sum_sq.sqrt()
    }

    /// Compare layer-wise similarity between two model hidden states.
    ///
    /// Returns a vector of cosine similarities, one for each layer.
    #[must_use]
    pub fn layer_similarity(a: &ModelHiddenStates, b: &ModelHiddenStates) -> Vec<f32> {
        let max_layers = a.layers.len().min(b.layers.len());
        let mut similarities = Vec::with_capacity(max_layers);

        for i in 0..max_layers {
            let sim = Self::cosine(&a.layers[i].hidden_state, &b.layers[i].hidden_state);
            similarities.push(sim);
        }

        similarities
    }

    /// Compute the average layer similarity.
    #[must_use]
    pub fn average_similarity(a: &ModelHiddenStates, b: &ModelHiddenStates) -> f32 {
        let similarities = Self::layer_similarity(a, b);
        if similarities.is_empty() {
            return 0.0;
        }
        similarities.iter().sum::<f32>() / similarities.len() as f32
    }

    /// Compute dot product between two tensors.
    #[must_use]
    pub fn dot_product(a: &HiddenStateTensor, b: &HiddenStateTensor) -> f32 {
        let len = a.data.len().min(b.data.len());
        a.data[..len]
            .iter()
            .zip(b.data[..len].iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    /// Compute the norm of a tensor.
    #[must_use]
    pub fn norm(tensor: &HiddenStateTensor) -> f32 {
        tensor.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize a tensor to unit length.
    #[must_use]
    pub fn normalize(tensor: &HiddenStateTensor) -> HiddenStateTensor {
        let norm = Self::norm(tensor);
        if norm == 0.0 {
            return tensor.clone();
        }

        let normalized_data: Vec<f32> = tensor.data.iter().map(|x| x / norm).collect();
        HiddenStateTensor {
            data: normalized_data,
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
            device: tensor.device,
        }
    }
}

/// Extract specific layers from model hidden states.
pub struct LayerExtractor;

impl LayerExtractor {
    /// Extract hidden states from specific layers.
    #[must_use]
    pub fn extract_layers<'a>(
        states: &'a ModelHiddenStates,
        layer_indices: &[usize],
    ) -> Vec<&'a LayerHiddenState> {
        layer_indices
            .iter()
            .filter_map(|&idx| states.get_layer(idx))
            .collect()
    }

    /// Extract the last N layers.
    #[must_use]
    pub fn extract_last_n(states: &ModelHiddenStates, n: usize) -> Vec<&LayerHiddenState> {
        let start = states.layers.len().saturating_sub(n);
        states.layers[start..].iter().collect()
    }

    /// Extract every Nth layer (for efficiency).
    #[must_use]
    pub fn extract_every_n(states: &ModelHiddenStates, n: usize) -> Vec<&LayerHiddenState> {
        if n == 0 {
            return vec![];
        }
        states.layers.iter().step_by(n).collect()
    }

    /// Extract middle layers (useful for some tasks).
    #[must_use]
    pub fn extract_middle(states: &ModelHiddenStates, count: usize) -> Vec<&LayerHiddenState> {
        let total = states.layers.len();
        if count >= total {
            return states.layers.iter().collect();
        }

        let start = (total - count) / 2;
        let end = start + count;
        states.layers[start..end].iter().collect()
    }
}

/// Pool hidden states using various strategies.
pub struct StatePooling;

impl StatePooling {
    /// Mean pooling across sequence dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor has fewer than 2 dimensions.
    pub fn mean_pool(tensor: &HiddenStateTensor) -> Result<HiddenStateTensor, HiddenStateError> {
        if tensor.shape.ndim() < 2 {
            return Err(HiddenStateError::InvalidDimension(
                "Tensor must have at least 2 dimensions for mean pooling".to_string(),
            ));
        }

        // Assuming shape is [batch, seq_len, hidden_dim]
        let hidden_dim = *tensor.shape.dims.last().unwrap_or(&0);
        let seq_len = if tensor.shape.ndim() >= 2 {
            tensor.shape.dims[tensor.shape.ndim() - 2]
        } else {
            1
        };

        if seq_len == 0 || hidden_dim == 0 {
            return Err(HiddenStateError::InvalidDimension(
                "Invalid tensor dimensions for pooling".to_string(),
            ));
        }

        let batch_size = if tensor.shape.ndim() >= 3 {
            tensor.shape.dims[0]
        } else {
            1
        };

        let mut pooled_data = vec![0.0f32; batch_size * hidden_dim];

        for b in 0..batch_size {
            for h in 0..hidden_dim {
                let mut sum = 0.0f32;
                for s in 0..seq_len {
                    let idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                    if idx < tensor.data.len() {
                        sum += tensor.data[idx];
                    }
                }
                let out_idx = b * hidden_dim + h;
                pooled_data[out_idx] = sum / seq_len as f32;
            }
        }

        let new_shape = if batch_size > 1 {
            TensorShape::new(vec![batch_size, hidden_dim])
        } else {
            TensorShape::new(vec![hidden_dim])
        };

        Ok(HiddenStateTensor {
            data: pooled_data,
            shape: new_shape,
            dtype: tensor.dtype,
            device: tensor.device,
        })
    }

    /// Max pooling across sequence dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor has fewer than 2 dimensions.
    pub fn max_pool(tensor: &HiddenStateTensor) -> Result<HiddenStateTensor, HiddenStateError> {
        if tensor.shape.ndim() < 2 {
            return Err(HiddenStateError::InvalidDimension(
                "Tensor must have at least 2 dimensions for max pooling".to_string(),
            ));
        }

        let hidden_dim = *tensor.shape.dims.last().unwrap_or(&0);
        let seq_len = if tensor.shape.ndim() >= 2 {
            tensor.shape.dims[tensor.shape.ndim() - 2]
        } else {
            1
        };

        if seq_len == 0 || hidden_dim == 0 {
            return Err(HiddenStateError::InvalidDimension(
                "Invalid tensor dimensions for pooling".to_string(),
            ));
        }

        let batch_size = if tensor.shape.ndim() >= 3 {
            tensor.shape.dims[0]
        } else {
            1
        };

        let mut pooled_data = vec![f32::NEG_INFINITY; batch_size * hidden_dim];

        for b in 0..batch_size {
            for h in 0..hidden_dim {
                for s in 0..seq_len {
                    let idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                    if idx < tensor.data.len() {
                        let out_idx = b * hidden_dim + h;
                        pooled_data[out_idx] = pooled_data[out_idx].max(tensor.data[idx]);
                    }
                }
            }
        }

        let new_shape = if batch_size > 1 {
            TensorShape::new(vec![batch_size, hidden_dim])
        } else {
            TensorShape::new(vec![hidden_dim])
        };

        Ok(HiddenStateTensor {
            data: pooled_data,
            shape: new_shape,
            dtype: tensor.dtype,
            device: tensor.device,
        })
    }

    /// CLS token pooling (take first token).
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor has fewer than 2 dimensions.
    pub fn cls_pool(tensor: &HiddenStateTensor) -> Result<HiddenStateTensor, HiddenStateError> {
        if tensor.shape.ndim() < 2 {
            return Err(HiddenStateError::InvalidDimension(
                "Tensor must have at least 2 dimensions for CLS pooling".to_string(),
            ));
        }

        let hidden_dim = *tensor.shape.dims.last().unwrap_or(&0);
        let batch_size = if tensor.shape.ndim() >= 3 {
            tensor.shape.dims[0]
        } else {
            1
        };
        let seq_len = if tensor.shape.ndim() >= 2 {
            tensor.shape.dims[tensor.shape.ndim() - 2]
        } else {
            1
        };

        if seq_len == 0 || hidden_dim == 0 {
            return Err(HiddenStateError::InvalidDimension(
                "Invalid tensor dimensions for CLS pooling".to_string(),
            ));
        }

        let mut pooled_data = Vec::with_capacity(batch_size * hidden_dim);

        for b in 0..batch_size {
            let start_idx = b * seq_len * hidden_dim;
            for h in 0..hidden_dim {
                if start_idx + h < tensor.data.len() {
                    pooled_data.push(tensor.data[start_idx + h]);
                } else {
                    pooled_data.push(0.0);
                }
            }
        }

        let new_shape = if batch_size > 1 {
            TensorShape::new(vec![batch_size, hidden_dim])
        } else {
            TensorShape::new(vec![hidden_dim])
        };

        Ok(HiddenStateTensor {
            data: pooled_data,
            shape: new_shape,
            dtype: tensor.dtype,
            device: tensor.device,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_extract_hidden_states() {
        let provider = MockHiddenStateProvider::new("test-model", 6, 256);
        let states = provider
            .extract_hidden_states("Hello, world!")
            .await
            .unwrap();

        assert_eq!(states.model_id, "test-model");
        assert_eq!(states.num_layers, 6);
        assert_eq!(states.hidden_dim, 256);
        assert_eq!(states.layers.len(), 6);
        assert!(states.pooled_output.is_some());
    }

    #[tokio::test]
    async fn test_mock_provider_deterministic() {
        let provider = MockHiddenStateProvider::new("test-model", 4, 128);

        let states1 = provider.extract_hidden_states("test input").await.unwrap();
        let states2 = provider.extract_hidden_states("test input").await.unwrap();

        // Same input should produce same output
        assert_eq!(
            states1.layers[0].hidden_state.data,
            states2.layers[0].hidden_state.data
        );
    }

    #[tokio::test]
    async fn test_mock_provider_different_inputs() {
        let provider = MockHiddenStateProvider::new("test-model", 4, 128);

        let states1 = provider.extract_hidden_states("input one").await.unwrap();
        let states2 = provider.extract_hidden_states("input two").await.unwrap();

        // Different inputs should produce different outputs
        assert_ne!(
            states1.layers[0].hidden_state.data,
            states2.layers[0].hidden_state.data
        );
    }

    #[tokio::test]
    async fn test_mock_provider_with_attention_weights() {
        let config = HiddenStateConfig::default().with_capture_attention_weights(true);
        let provider = MockHiddenStateProvider::new("test-model", 4, 128).with_config(config);

        let states = provider.extract_hidden_states("test").await.unwrap();

        for layer in &states.layers {
            assert!(layer.attention_weights.is_some());
        }
    }

    #[tokio::test]
    async fn test_mock_provider_with_kv_cache() {
        let provider = MockHiddenStateProvider::new("test-model", 4, 96);
        let (states, kv_cache) = provider
            .extract_with_kv_cache("test input", None)
            .await
            .unwrap();

        assert_eq!(states.model_id, "test-model");
        assert_eq!(kv_cache.model_id, "test-model");
        assert_eq!(kv_cache.layers.len(), 4);
    }

    #[test]
    fn test_state_similarity_cosine() {
        let a = HiddenStateTensor::from_vec_1d(vec![1.0, 0.0, 0.0]);
        let b = HiddenStateTensor::from_vec_1d(vec![1.0, 0.0, 0.0]);

        let sim = StateSimilarity::cosine(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_state_similarity_cosine_orthogonal() {
        let a = HiddenStateTensor::from_vec_1d(vec![1.0, 0.0, 0.0]);
        let b = HiddenStateTensor::from_vec_1d(vec![0.0, 1.0, 0.0]);

        let sim = StateSimilarity::cosine(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_state_similarity_l2_distance() {
        let a = HiddenStateTensor::from_vec_1d(vec![0.0, 0.0, 0.0]);
        let b = HiddenStateTensor::from_vec_1d(vec![3.0, 4.0, 0.0]);

        let dist = StateSimilarity::l2_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_state_similarity_normalize() {
        let tensor = HiddenStateTensor::from_vec_1d(vec![3.0, 4.0]);
        let normalized = StateSimilarity::normalize(&tensor);

        let norm = StateSimilarity::norm(&normalized);
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_layer_extractor_last_n() {
        let mut states = ModelHiddenStates::new("test", 6, 64);
        for i in 0..6 {
            let hidden = HiddenStateTensor::from_vec_1d(vec![i as f32; 64]);
            states.add_layer(LayerHiddenState::new(i, hidden));
        }

        let last_2 = LayerExtractor::extract_last_n(&states, 2);
        assert_eq!(last_2.len(), 2);
        assert_eq!(last_2[0].layer_idx, 4);
        assert_eq!(last_2[1].layer_idx, 5);
    }

    #[test]
    fn test_layer_extractor_every_n() {
        let mut states = ModelHiddenStates::new("test", 6, 64);
        for i in 0..6 {
            let hidden = HiddenStateTensor::from_vec_1d(vec![i as f32; 64]);
            states.add_layer(LayerHiddenState::new(i, hidden));
        }

        let every_2 = LayerExtractor::extract_every_n(&states, 2);
        assert_eq!(every_2.len(), 3);
        assert_eq!(every_2[0].layer_idx, 0);
        assert_eq!(every_2[1].layer_idx, 2);
        assert_eq!(every_2[2].layer_idx, 4);
    }

    #[test]
    fn test_state_pooling_mean() {
        // Shape: [1, 2, 4] - batch=1, seq_len=2, hidden_dim=4
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = HiddenStateTensor::from_vec(data, TensorShape::new(vec![1, 2, 4])).unwrap();

        let pooled = StatePooling::mean_pool(&tensor).unwrap();
        // Mean of each column: [(1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2] = [3, 4, 5, 6]
        assert_eq!(pooled.shape.dims, vec![4]);
        assert!((pooled.data[0] - 3.0).abs() < 0.001);
        assert!((pooled.data[1] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_state_pooling_max() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = HiddenStateTensor::from_vec(data, TensorShape::new(vec![1, 2, 4])).unwrap();

        let pooled = StatePooling::max_pool(&tensor).unwrap();
        // Max of each column: [5, 6, 7, 8]
        assert_eq!(pooled.shape.dims, vec![4]);
        assert!((pooled.data[0] - 5.0).abs() < 0.001);
        assert!((pooled.data[3] - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_state_pooling_cls() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = HiddenStateTensor::from_vec(data, TensorShape::new(vec![1, 2, 4])).unwrap();

        let pooled = StatePooling::cls_pool(&tensor).unwrap();
        // First token: [1, 2, 3, 4]
        assert_eq!(pooled.shape.dims, vec![4]);
        assert!((pooled.data[0] - 1.0).abs() < 0.001);
        assert!((pooled.data[3] - 4.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_layer_similarity() {
        let provider = MockHiddenStateProvider::new("test-model", 4, 64);
        let states1 = provider.extract_hidden_states("hello").await.unwrap();
        let states2 = provider.extract_hidden_states("hello").await.unwrap();

        let similarities = StateSimilarity::layer_similarity(&states1, &states2);
        assert_eq!(similarities.len(), 4);

        // Same input should have similarity of 1.0
        for sim in similarities {
            assert!((sim - 1.0).abs() < 0.001);
        }
    }
}
