//! Core types for hidden state manipulation.
//!
//! This module provides the fundamental data structures for capturing and reusing
//! hidden states from transformer models in Speculative RAG.

#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::HiddenStateError;

/// Tensor shape representation for hidden state tensors.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TensorShape {
    /// Dimensions of the tensor.
    pub dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new tensor shape from dimensions.
    #[must_use]
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Get the total number of elements in the tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }

    /// Get the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Check if the shape is valid (no zero dimensions unless empty).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.dims.is_empty() || self.dims.iter().all(|&d| d > 0)
    }
}

impl From<Vec<usize>> for TensorShape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for TensorShape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DType {
    /// 32-bit floating point.
    #[default]
    F32,
    /// 16-bit floating point.
    F16,
    /// Brain floating point (16-bit).
    BF16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
}

impl DType {
    /// Get the size in bytes for this data type.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
        }
    }
}

/// Device type for tensor storage and computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Device {
    /// CPU device.
    #[default]
    Cpu,
    /// CUDA GPU device with index.
    Cuda(usize),
    /// Metal GPU (Apple Silicon).
    Metal,
}

impl Device {
    /// Check if this is a GPU device.
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda(_) | Self::Metal)
    }

    /// Check if this is a CPU device.
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }
}

/// A tensor of hidden states.
///
/// This is a simplified tensor representation that stores data as f32
/// and can be converted to other types on demand.
#[derive(Debug, Clone)]
pub struct HiddenStateTensor {
    /// The tensor data stored as f32 values.
    pub data: Vec<f32>,
    /// The shape of the tensor.
    pub shape: TensorShape,
    /// The data type for the tensor.
    pub dtype: DType,
    /// The device where the tensor is stored.
    pub device: Device,
}

impl HiddenStateTensor {
    /// Create a tensor filled with zeros.
    #[must_use]
    pub fn zeros(shape: TensorShape, dtype: DType, device: Device) -> Self {
        let numel = shape.numel();
        Self {
            data: vec![0.0; numel],
            shape,
            dtype,
            device,
        }
    }

    /// Create a tensor from a vector of f32 values.
    ///
    /// # Errors
    ///
    /// Returns an error if the data length doesn't match the shape.
    pub fn from_vec(data: Vec<f32>, shape: TensorShape) -> Result<Self, HiddenStateError> {
        let expected = shape.numel();
        if data.len() != expected {
            return Err(HiddenStateError::ShapeMismatch {
                expected: shape.dims.clone(),
                actual: vec![data.len()],
            });
        }
        Ok(Self {
            data,
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
        })
    }

    /// Create a tensor from a vector with inferred 1D shape.
    #[must_use]
    pub fn from_vec_1d(data: Vec<f32>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: TensorShape::new(vec![len]),
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Slice the tensor along a dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension or range is invalid.
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Self, HiddenStateError> {
        if dim >= self.shape.ndim() {
            return Err(HiddenStateError::InvalidDimension(format!(
                "dimension {dim} out of bounds for tensor with {} dimensions",
                self.shape.ndim()
            )));
        }

        let dim_size = self.shape.dims[dim];
        if start >= end || end > dim_size {
            return Err(HiddenStateError::InvalidDimension(format!(
                "invalid slice range [{start}, {end}) for dimension of size {dim_size}"
            )));
        }

        // Calculate strides
        let mut strides = vec![1usize; self.shape.ndim()];
        for i in (0..self.shape.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * self.shape.dims[i + 1];
        }

        // Calculate new shape
        let mut new_dims = self.shape.dims.clone();
        new_dims[dim] = end - start;
        let new_shape = TensorShape::new(new_dims);

        // Copy data
        let slice_stride = strides[dim];
        let outer_size: usize = self.shape.dims[..dim].iter().product();
        let inner_size: usize = if dim + 1 < self.shape.ndim() {
            self.shape.dims[dim + 1..].iter().product()
        } else {
            1
        };

        let mut new_data = Vec::with_capacity(new_shape.numel());

        for outer in 0..outer_size.max(1) {
            for slice_idx in start..end {
                let base_idx = outer * self.shape.dims[dim..].iter().product::<usize>()
                    + slice_idx * slice_stride;
                for inner in 0..inner_size {
                    new_data.push(self.data[base_idx + inner]);
                }
            }
        }

        Ok(Self {
            data: new_data,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device,
        })
    }

    /// Concatenate multiple tensors along a dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if tensors have incompatible shapes.
    pub fn concat(tensors: &[&Self], dim: usize) -> Result<Self, HiddenStateError> {
        if tensors.is_empty() {
            return Ok(Self {
                data: vec![],
                shape: TensorShape::default(),
                dtype: DType::F32,
                device: Device::Cpu,
            });
        }

        let first = tensors[0];
        if dim >= first.shape.ndim() {
            return Err(HiddenStateError::InvalidDimension(format!(
                "dimension {dim} out of bounds for tensor with {} dimensions",
                first.shape.ndim()
            )));
        }

        // Verify all tensors have compatible shapes
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape.ndim() != first.shape.ndim() {
                return Err(HiddenStateError::ShapeMismatch {
                    expected: first.shape.dims.clone(),
                    actual: tensor.shape.dims.clone(),
                });
            }
            for (d, (&a, &b)) in first
                .shape
                .dims
                .iter()
                .zip(tensor.shape.dims.iter())
                .enumerate()
            {
                if d != dim && a != b {
                    return Err(HiddenStateError::ShapeMismatch {
                        expected: first.shape.dims.clone(),
                        actual: tensors[i].shape.dims.clone(),
                    });
                }
            }
        }

        // Calculate new shape
        let mut new_dims = first.shape.dims.clone();
        new_dims[dim] = tensors.iter().map(|t| t.shape.dims[dim]).sum();
        let new_shape = TensorShape::new(new_dims);

        // For simple 1D case or last dimension, just concatenate data
        if first.shape.ndim() == 1 || dim == first.shape.ndim() - 1 {
            let new_data: Vec<f32> = tensors
                .iter()
                .flat_map(|t| t.data.iter().copied())
                .collect();
            return Ok(Self {
                data: new_data,
                shape: new_shape,
                dtype: first.dtype,
                device: first.device,
            });
        }

        // General case: interleave data properly
        let outer_size: usize = first.shape.dims[..dim].iter().product();
        let inner_size: usize = first.shape.dims[dim + 1..].iter().product();

        let mut new_data = Vec::with_capacity(new_shape.numel());

        for outer in 0..outer_size.max(1) {
            for tensor in tensors {
                let tensor_dim_size = tensor.shape.dims[dim];
                for slice_idx in 0..tensor_dim_size {
                    let base_idx = outer * tensor.shape.dims[dim..].iter().product::<usize>()
                        + slice_idx * inner_size;
                    for inner in 0..inner_size {
                        new_data.push(tensor.data[base_idx + inner]);
                    }
                }
            }
        }

        Ok(Self {
            data: new_data,
            shape: new_shape,
            dtype: first.dtype,
            device: first.device,
        })
    }

    /// Move the tensor to a different device.
    ///
    /// Note: This is a no-op in this simplified implementation.
    #[must_use]
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device,
        }
    }

    /// Get the size of the tensor in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len() * self.dtype.size_bytes()
    }

    /// Get the number of elements in the tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a value at a specific index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }

    /// Set a value at a specific index.
    pub fn set(&mut self, index: usize, value: f32) -> Option<()> {
        if index < self.data.len() {
            self.data[index] = value;
            Some(())
        } else {
            None
        }
    }
}

impl Default for HiddenStateTensor {
    fn default() -> Self {
        Self::zeros(TensorShape::default(), DType::F32, Device::Cpu)
    }
}

/// Hidden states from a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerHiddenState {
    /// The layer index (0-indexed).
    pub layer_idx: usize,
    /// The hidden state tensor from this layer.
    pub hidden_state: HiddenStateTensor,
    /// Optional attention weights from this layer.
    pub attention_weights: Option<HiddenStateTensor>,
}

impl LayerHiddenState {
    /// Create a new layer hidden state.
    #[must_use]
    pub fn new(layer_idx: usize, hidden_state: HiddenStateTensor) -> Self {
        Self {
            layer_idx,
            hidden_state,
            attention_weights: None,
        }
    }

    /// Add attention weights to this layer state.
    #[must_use]
    pub fn with_attention_weights(mut self, weights: HiddenStateTensor) -> Self {
        self.attention_weights = Some(weights);
        self
    }

    /// Get the size in bytes of this layer's hidden states.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let hidden_size = self.hidden_state.size_bytes();
        let attn_size = self
            .attention_weights
            .as_ref()
            .map_or(0, HiddenStateTensor::size_bytes);
        hidden_size + attn_size
    }
}

/// Complete hidden states from a model forward pass.
#[derive(Debug, Clone)]
pub struct ModelHiddenStates {
    /// Identifier for the model that produced these states.
    pub model_id: String,
    /// Length of the input sequence in tokens.
    pub sequence_length: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension size.
    pub hidden_dim: usize,
    /// Hidden states from each layer.
    pub layers: Vec<LayerHiddenState>,
    /// Optional pooled output (e.g., \[CLS\] token representation).
    pub pooled_output: Option<HiddenStateTensor>,
    /// Unix timestamp when these states were created.
    pub created_at: u64,
}

impl ModelHiddenStates {
    /// Create a new empty model hidden states container.
    #[must_use]
    pub fn new(model_id: impl Into<String>, num_layers: usize, hidden_dim: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self {
            model_id: model_id.into(),
            sequence_length: 0,
            num_layers,
            hidden_dim,
            layers: Vec::with_capacity(num_layers),
            pooled_output: None,
            created_at: timestamp,
        }
    }

    /// Get a specific layer's hidden state.
    #[must_use]
    pub fn get_layer(&self, idx: usize) -> Option<&LayerHiddenState> {
        self.layers.iter().find(|l| l.layer_idx == idx)
    }

    /// Get the last hidden state (from the final layer).
    #[must_use]
    pub fn last_hidden_state(&self) -> Option<&HiddenStateTensor> {
        self.layers.last().map(|l| &l.hidden_state)
    }

    /// Calculate the total size in bytes of all hidden states.
    #[must_use]
    pub fn total_size_bytes(&self) -> usize {
        let layers_size: usize = self.layers.iter().map(LayerHiddenState::size_bytes).sum();
        let pooled_size = self
            .pooled_output
            .as_ref()
            .map_or(0, HiddenStateTensor::size_bytes);
        layers_size + pooled_size
    }

    /// Extract states for a prefix (first n tokens).
    ///
    /// # Errors
    ///
    /// Returns an error if n is greater than the sequence length.
    pub fn prefix_states(&self, n: usize) -> Result<Self, HiddenStateError> {
        if n > self.sequence_length {
            return Err(HiddenStateError::InvalidDimension(format!(
                "prefix length {} exceeds sequence length {}",
                n, self.sequence_length
            )));
        }

        let mut new_states = Self::new(&self.model_id, self.num_layers, self.hidden_dim);
        new_states.sequence_length = n;
        new_states.created_at = self.created_at;

        for layer in &self.layers {
            // Assuming hidden state shape is [batch, seq_len, hidden_dim]
            // or [seq_len, hidden_dim]
            let new_hidden = if layer.hidden_state.shape.ndim() >= 2 {
                let seq_dim = layer.hidden_state.shape.ndim() - 2;
                layer.hidden_state.slice(seq_dim, 0, n)?
            } else {
                layer.hidden_state.clone()
            };

            let new_attn = if let Some(ref attn) = layer.attention_weights {
                if attn.shape.ndim() >= 2 {
                    let seq_dim = attn.shape.ndim() - 2;
                    Some(attn.slice(seq_dim, 0, n)?)
                } else {
                    Some(attn.clone())
                }
            } else {
                None
            };

            let mut new_layer = LayerHiddenState::new(layer.layer_idx, new_hidden);
            if let Some(attn) = new_attn {
                new_layer = new_layer.with_attention_weights(attn);
            }
            new_states.layers.push(new_layer);
        }

        Ok(new_states)
    }

    /// Concatenate with another set of states.
    ///
    /// # Errors
    ///
    /// Returns an error if the states are incompatible (different model, dimensions, etc.).
    pub fn concat(&self, other: &Self) -> Result<Self, HiddenStateError> {
        if self.model_id != other.model_id {
            return Err(HiddenStateError::ProviderError(format!(
                "cannot concatenate states from different models: {} vs {}",
                self.model_id, other.model_id
            )));
        }

        if self.hidden_dim != other.hidden_dim {
            return Err(HiddenStateError::ShapeMismatch {
                expected: vec![self.hidden_dim],
                actual: vec![other.hidden_dim],
            });
        }

        if self.layers.len() != other.layers.len() {
            return Err(HiddenStateError::ShapeMismatch {
                expected: vec![self.layers.len()],
                actual: vec![other.layers.len()],
            });
        }

        let mut new_states = Self::new(&self.model_id, self.num_layers, self.hidden_dim);
        new_states.sequence_length = self.sequence_length + other.sequence_length;

        for (self_layer, other_layer) in self.layers.iter().zip(other.layers.iter()) {
            // Concatenate hidden states along sequence dimension
            let seq_dim = if self_layer.hidden_state.shape.ndim() >= 2 {
                self_layer.hidden_state.shape.ndim() - 2
            } else {
                0
            };

            let new_hidden = HiddenStateTensor::concat(
                &[&self_layer.hidden_state, &other_layer.hidden_state],
                seq_dim,
            )?;

            let new_attn = match (
                &self_layer.attention_weights,
                &other_layer.attention_weights,
            ) {
                (Some(self_attn), Some(other_attn)) => {
                    let attn_seq_dim = if self_attn.shape.ndim() >= 2 {
                        self_attn.shape.ndim() - 2
                    } else {
                        0
                    };
                    Some(HiddenStateTensor::concat(
                        &[self_attn, other_attn],
                        attn_seq_dim,
                    )?)
                }
                _ => None,
            };

            let mut new_layer = LayerHiddenState::new(self_layer.layer_idx, new_hidden);
            if let Some(attn) = new_attn {
                new_layer = new_layer.with_attention_weights(attn);
            }
            new_states.layers.push(new_layer);
        }

        Ok(new_states)
    }

    /// Add a layer's hidden state.
    pub fn add_layer(&mut self, layer: LayerHiddenState) {
        self.layers.push(layer);
    }

    /// Set the pooled output.
    pub fn set_pooled_output(&mut self, pooled: HiddenStateTensor) {
        self.pooled_output = Some(pooled);
    }
}

/// Key-Value cache for transformer attention.
///
/// Stores pre-computed keys and values from attention layers to avoid
/// recomputation during autoregressive generation.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// The layer index this cache belongs to.
    pub layer_idx: usize,
    /// Cached keys tensor: `[batch, heads, seq_len, head_dim]`.
    pub keys: HiddenStateTensor,
    /// Cached values tensor: `[batch, heads, seq_len, head_dim]`.
    pub values: HiddenStateTensor,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
}

impl KVCache {
    /// Create a new KV cache for a layer.
    #[must_use]
    pub fn new(layer_idx: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let shape = TensorShape::new(vec![1, num_heads, max_seq_len, head_dim]);
        Self {
            layer_idx,
            keys: HiddenStateTensor::zeros(shape.clone(), DType::F32, Device::Cpu),
            values: HiddenStateTensor::zeros(shape, DType::F32, Device::Cpu),
            num_heads,
            head_dim,
        }
    }

    /// Append new keys and values to the cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes are incompatible.
    pub fn append(
        &mut self,
        new_keys: &HiddenStateTensor,
        new_values: &HiddenStateTensor,
    ) -> Result<(), HiddenStateError> {
        // Concatenate along sequence dimension (dim 2)
        self.keys = HiddenStateTensor::concat(&[&self.keys, new_keys], 2)?;
        self.values = HiddenStateTensor::concat(&[&self.values, new_values], 2)?;
        Ok(())
    }

    /// Get the current sequence length in the cache.
    #[must_use]
    pub fn current_length(&self) -> usize {
        if self.keys.shape.ndim() >= 3 {
            self.keys.shape.dims[2]
        } else {
            0
        }
    }

    /// Truncate the cache to a specific length.
    pub fn truncate(&mut self, length: usize) {
        if length < self.current_length() {
            if let Ok(sliced_keys) = self.keys.slice(2, 0, length) {
                self.keys = sliced_keys;
            }
            if let Ok(sliced_values) = self.values.slice(2, 0, length) {
                self.values = sliced_values;
            }
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        let shape = TensorShape::new(vec![1, self.num_heads, 0, self.head_dim]);
        self.keys = HiddenStateTensor::zeros(shape.clone(), DType::F32, self.keys.device);
        self.values = HiddenStateTensor::zeros(shape, DType::F32, self.values.device);
    }

    /// Get the size in bytes of this cache.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.keys.size_bytes() + self.values.size_bytes()
    }
}

/// Complete KV cache for all layers of a model.
#[derive(Debug, Clone)]
pub struct ModelKVCache {
    /// Model identifier.
    pub model_id: String,
    /// KV caches for each layer.
    pub layers: Vec<KVCache>,
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
}

impl ModelKVCache {
    /// Create a new model KV cache.
    #[must_use]
    pub fn new(
        model_id: &str,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|i| KVCache::new(i, num_heads, head_dim, max_seq_len))
            .collect();
        Self {
            model_id: model_id.to_string(),
            layers,
            max_seq_len,
        }
    }

    /// Append new KV pairs to a specific layer.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer index is invalid or shapes are incompatible.
    pub fn append_layer(
        &mut self,
        layer_idx: usize,
        keys: &HiddenStateTensor,
        values: &HiddenStateTensor,
    ) -> Result<(), HiddenStateError> {
        if layer_idx >= self.layers.len() {
            return Err(HiddenStateError::InvalidDimension(format!(
                "layer index {} out of bounds for model with {} layers",
                layer_idx,
                self.layers.len()
            )));
        }
        self.layers[layer_idx].append(keys, values)
    }

    /// Get a specific layer's KV cache.
    #[must_use]
    pub fn get_layer(&self, idx: usize) -> Option<&KVCache> {
        self.layers.get(idx)
    }

    /// Get the current sequence length (from the first layer).
    #[must_use]
    pub fn current_length(&self) -> usize {
        self.layers.first().map_or(0, KVCache::current_length)
    }

    /// Clear all layer caches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Get the total size in bytes of all caches.
    #[must_use]
    pub fn total_size_bytes(&self) -> usize {
        self.layers.iter().map(KVCache::size_bytes).sum()
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Configuration for hidden state operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStateConfig {
    /// Whether to capture hidden states from all layers.
    pub capture_all_layers: bool,
    /// Whether to capture attention weights.
    pub capture_attention_weights: bool,
    /// Maximum number of cache entries.
    pub max_cache_entries: usize,
    /// Data type for tensors.
    pub dtype: DType,
    /// Device for tensor storage.
    pub device: Device,
}

impl Default for HiddenStateConfig {
    fn default() -> Self {
        Self {
            capture_all_layers: true,
            capture_attention_weights: false,
            max_cache_entries: 100,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }
}

impl HiddenStateConfig {
    /// Create a new hidden state configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to capture all layers.
    #[must_use]
    pub fn with_capture_all_layers(mut self, capture: bool) -> Self {
        self.capture_all_layers = capture;
        self
    }

    /// Set whether to capture attention weights.
    #[must_use]
    pub fn with_capture_attention_weights(mut self, capture: bool) -> Self {
        self.capture_attention_weights = capture;
        self
    }

    /// Set the maximum number of cache entries.
    #[must_use]
    pub fn with_max_cache_entries(mut self, max: usize) -> Self {
        self.max_cache_entries = max;
        self
    }

    /// Set the data type.
    #[must_use]
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the device.
    #[must_use]
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_new() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape.dims, vec![2, 3, 4]);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.ndim(), 3);
    }

    #[test]
    fn test_tensor_shape_empty() {
        let shape = TensorShape::new(vec![]);
        assert_eq!(shape.numel(), 0);
        assert_eq!(shape.ndim(), 0);
        assert!(shape.is_valid());
    }

    #[test]
    fn test_dtype_size_bytes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::I64.size_bytes(), 8);
    }

    #[test]
    fn test_device_is_gpu() {
        assert!(!Device::Cpu.is_gpu());
        assert!(Device::Cuda(0).is_gpu());
        assert!(Device::Metal.is_gpu());
    }

    #[test]
    fn test_hidden_state_tensor_zeros() {
        let shape = TensorShape::new(vec![2, 3]);
        let tensor = HiddenStateTensor::zeros(shape.clone(), DType::F32, Device::Cpu);
        assert_eq!(tensor.data.len(), 6);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
        assert_eq!(tensor.shape, shape);
    }

    #[test]
    fn test_hidden_state_tensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = TensorShape::new(vec![2, 3]);
        let tensor = HiddenStateTensor::from_vec(data.clone(), shape).unwrap();
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_hidden_state_tensor_from_vec_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = TensorShape::new(vec![2, 3]);
        let result = HiddenStateTensor::from_vec(data, shape);
        assert!(result.is_err());
    }

    #[test]
    fn test_hidden_state_tensor_slice() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![3, 4]);
        let tensor = HiddenStateTensor::from_vec(data, shape).unwrap();

        // Slice first 2 rows
        let sliced = tensor.slice(0, 0, 2).unwrap();
        assert_eq!(sliced.shape.dims, vec![2, 4]);
        assert_eq!(sliced.data.len(), 8);
    }

    #[test]
    fn test_hidden_state_tensor_concat() {
        let t1 =
            HiddenStateTensor::from_vec(vec![1.0, 2.0, 3.0], TensorShape::new(vec![3])).unwrap();
        let t2 = HiddenStateTensor::from_vec(vec![4.0, 5.0], TensorShape::new(vec![2])).unwrap();

        let result = HiddenStateTensor::concat(&[&t1, &t2], 0).unwrap();
        assert_eq!(result.shape.dims, vec![5]);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_hidden_state_tensor_size_bytes() {
        let tensor = HiddenStateTensor::zeros(TensorShape::new(vec![10]), DType::F32, Device::Cpu);
        assert_eq!(tensor.size_bytes(), 40); // 10 * 4 bytes
    }

    #[test]
    fn test_layer_hidden_state() {
        let hidden =
            HiddenStateTensor::zeros(TensorShape::new(vec![10, 768]), DType::F32, Device::Cpu);
        let layer = LayerHiddenState::new(0, hidden);
        assert_eq!(layer.layer_idx, 0);
        assert!(layer.attention_weights.is_none());
    }

    #[test]
    fn test_model_hidden_states_new() {
        let states = ModelHiddenStates::new("test-model", 12, 768);
        assert_eq!(states.model_id, "test-model");
        assert_eq!(states.num_layers, 12);
        assert_eq!(states.hidden_dim, 768);
        assert!(states.layers.is_empty());
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(0, 12, 64, 512);
        assert_eq!(cache.layer_idx, 0);
        assert_eq!(cache.num_heads, 12);
        assert_eq!(cache.head_dim, 64);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::new(0, 12, 64, 512);
        cache.clear();
        assert_eq!(cache.current_length(), 0);
    }

    #[test]
    fn test_model_kv_cache_new() {
        let cache = ModelKVCache::new("test-model", 12, 12, 64, 512);
        assert_eq!(cache.model_id, "test-model");
        assert_eq!(cache.layers.len(), 12);
        assert_eq!(cache.max_seq_len, 512);
    }

    #[test]
    fn test_hidden_state_config_default() {
        let config = HiddenStateConfig::default();
        assert!(config.capture_all_layers);
        assert!(!config.capture_attention_weights);
        assert_eq!(config.max_cache_entries, 100);
    }

    #[test]
    fn test_hidden_state_config_builder() {
        let config = HiddenStateConfig::new()
            .with_capture_all_layers(false)
            .with_capture_attention_weights(true)
            .with_max_cache_entries(50)
            .with_dtype(DType::F16)
            .with_device(Device::Cuda(0));

        assert!(!config.capture_all_layers);
        assert!(config.capture_attention_weights);
        assert_eq!(config.max_cache_entries, 50);
        assert_eq!(config.dtype, DType::F16);
        assert_eq!(config.device, Device::Cuda(0));
    }
}
