//! Hidden states types for speculative decoding.
//!
//! This module provides types for representing and caching hidden states
//! from transformer models, enabling efficient speculative decoding.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "native")]
use std::sync::RwLock;

use crate::error::SpeculatorError;

/// Hidden states from a single layer of a transformer model.
#[derive(Debug, Clone)]
pub struct LayerHiddenState {
    /// The layer index (0-indexed).
    pub layer_idx: usize,
    /// The hidden state tensor values (flattened).
    pub values: Vec<f32>,
    /// Shape of the tensor: `[batch_size, sequence_length, hidden_dim]`.
    pub shape: [usize; 3],
    /// Optional attention weights for this layer.
    pub attention_weights: Option<Vec<f32>>,
}

impl LayerHiddenState {
    /// Create a new layer hidden state.
    #[must_use]
    pub fn new(layer_idx: usize, values: Vec<f32>, shape: [usize; 3]) -> Self {
        Self {
            layer_idx,
            values,
            shape,
            attention_weights: None,
        }
    }

    /// Set attention weights.
    #[must_use]
    pub fn with_attention_weights(mut self, weights: Vec<f32>) -> Self {
        self.attention_weights = Some(weights);
        self
    }

    /// Get the hidden dimension.
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.shape[2]
    }

    /// Get the sequence length.
    #[must_use]
    pub fn sequence_length(&self) -> usize {
        self.shape[1]
    }

    /// Get the batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.shape[0]
    }

    /// Get the hidden state at a specific position.
    #[must_use]
    pub fn at_position(&self, batch: usize, pos: usize) -> Option<&[f32]> {
        if batch >= self.shape[0] || pos >= self.shape[1] {
            return None;
        }
        let hidden_dim = self.shape[2];
        let start = (batch * self.shape[1] + pos) * hidden_dim;
        let end = start + hidden_dim;
        if end <= self.values.len() {
            Some(&self.values[start..end])
        } else {
            None
        }
    }
}

/// Hidden states from all layers of a model for a given input.
#[derive(Debug, Clone)]
pub struct ModelHiddenStates {
    /// Model identifier.
    pub model_id: String,
    /// Hidden states from each layer.
    pub layers: Vec<LayerHiddenState>,
    /// The input token IDs that produced these hidden states.
    pub input_tokens: Vec<u32>,
    /// Optional pooled representation (e.g., \[CLS\] token or mean pooling).
    pub pooled: Option<Vec<f32>>,
}

impl ModelHiddenStates {
    /// Create new model hidden states.
    #[must_use]
    pub fn new(
        model_id: impl Into<String>,
        layers: Vec<LayerHiddenState>,
        input_tokens: Vec<u32>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            layers,
            input_tokens,
            pooled: None,
        }
    }

    /// Set the pooled representation.
    #[must_use]
    pub fn with_pooled(mut self, pooled: Vec<f32>) -> Self {
        self.pooled = Some(pooled);
        self
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get hidden states for a specific layer.
    #[must_use]
    pub fn layer(&self, idx: usize) -> Option<&LayerHiddenState> {
        self.layers.get(idx)
    }

    /// Get the last layer hidden states.
    #[must_use]
    pub fn last_layer(&self) -> Option<&LayerHiddenState> {
        self.layers.last()
    }

    /// Compute cosine similarity between two hidden state representations.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if let (Some(a), Some(b)) = (&self.pooled, &other.pooled) {
            cosine_similarity_vec(a, b)
        } else if let (Some(a), Some(b)) = (self.last_layer(), other.last_layer()) {
            // Compare the [CLS] token (position 0) hidden states
            if let (Some(vec_a), Some(vec_b)) = (a.at_position(0, 0), b.at_position(0, 0)) {
                cosine_similarity_vec(vec_a, vec_b)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity_vec(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-9 { dot / denom } else { 0.0 }
}

/// Key-Value cache for transformer models.
#[derive(Debug, Clone)]
pub struct ModelKVCache {
    /// Model identifier.
    pub model_id: String,
    /// Cached key tensors for each layer: `layer_idx` -> `(num_heads, seq_len, head_dim)`.
    pub keys: HashMap<usize, Vec<f32>>,
    /// Cached value tensors for each layer.
    pub values: HashMap<usize, Vec<f32>>,
    /// Current sequence length in the cache.
    pub seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
}

impl ModelKVCache {
    /// Create a new empty KV cache.
    #[must_use]
    pub fn new(model_id: impl Into<String>, num_heads: usize, head_dim: usize) -> Self {
        Self {
            model_id: model_id.into(),
            keys: HashMap::new(),
            values: HashMap::new(),
            seq_len: 0,
            num_heads,
            head_dim,
        }
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Get the current sequence length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Add keys and values for a layer.
    pub fn add_layer(&mut self, layer_idx: usize, keys: Vec<f32>, values: Vec<f32>) {
        self.keys.insert(layer_idx, keys);
        self.values.insert(layer_idx, values);
    }

    /// Update the sequence length.
    pub fn set_seq_len(&mut self, seq_len: usize) {
        self.seq_len = seq_len;
    }
}

/// Configuration for hidden state caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStateCacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Whether to cache attention patterns.
    pub cache_attention: bool,
    /// TTL for cache entries in seconds.
    pub ttl_seconds: u64,
    /// Whether to use LRU eviction.
    pub use_lru: bool,
}

impl Default for HiddenStateCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            cache_attention: false,
            ttl_seconds: 3600,
            use_lru: true,
        }
    }
}

/// A cache for storing hidden states.
#[derive(Clone)]
pub struct HiddenStateCache {
    config: HiddenStateCacheConfig,
    #[cfg(feature = "native")]
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    #[cfg(feature = "native")]
    access_order: Arc<RwLock<Vec<String>>>,
    #[cfg(not(feature = "native"))]
    entries: Arc<std::cell::RefCell<HashMap<String, CacheEntry>>>,
    #[cfg(not(feature = "native"))]
    access_order: Arc<std::cell::RefCell<Vec<String>>>,
}

#[derive(Clone)]
struct CacheEntry {
    states: ModelHiddenStates,
    #[allow(dead_code)]
    created_at: std::time::SystemTime,
}

impl HiddenStateCache {
    /// Create a new hidden state cache.
    #[must_use]
    pub fn new(config: HiddenStateCacheConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "native")]
            entries: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "native")]
            access_order: Arc::new(RwLock::new(Vec::new())),
            #[cfg(not(feature = "native"))]
            entries: Arc::new(std::cell::RefCell::new(HashMap::new())),
            #[cfg(not(feature = "native"))]
            access_order: Arc::new(std::cell::RefCell::new(Vec::new())),
        }
    }

    /// Generate a cache key from input bytes (e.g., text content).
    #[must_use]
    pub fn make_key(model_id: &str, content: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        model_id.hash(&mut hasher);
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Generate a cache key from token IDs.
    #[must_use]
    pub fn make_key_from_tokens(model_id: &str, tokens: &[u32]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        model_id.hash(&mut hasher);
        tokens.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get hidden states from the cache.
    #[cfg(feature = "native")]
    #[must_use]
    pub fn get(&self, key: &str) -> Option<ModelHiddenStates> {
        let entries = self.entries.read().ok()?;
        let entry = entries.get(key)?;

        // Update access order for LRU
        if self.config.use_lru
            && let Ok(mut order) = self.access_order.write()
        {
            if let Some(pos) = order.iter().position(|k| k == key) {
                order.remove(pos);
            }
            order.push(key.to_string());
        }

        Some(entry.states.clone())
    }

    /// Get hidden states from the cache (non-native version).
    #[cfg(not(feature = "native"))]
    #[must_use]
    pub fn get(&self, key: &str) -> Option<ModelHiddenStates> {
        let entries = self.entries.borrow();
        let entry = entries.get(key)?;

        // Update access order for LRU
        if self.config.use_lru {
            let mut order = self.access_order.borrow_mut();
            if let Some(pos) = order.iter().position(|k| k == key) {
                order.remove(pos);
            }
            order.push(key.to_string());
        }

        Some(entry.states.clone())
    }

    /// Insert hidden states into the cache.
    #[cfg(feature = "native")]
    pub fn insert(&self, key: String, states: ModelHiddenStates) {
        // Evict if at capacity
        self.evict_if_needed();

        if let Ok(mut entries) = self.entries.write() {
            entries.insert(
                key.clone(),
                CacheEntry {
                    states,
                    created_at: std::time::SystemTime::now(),
                },
            );
        }

        if self.config.use_lru
            && let Ok(mut order) = self.access_order.write()
        {
            order.push(key);
        }
    }

    /// Insert hidden states into the cache (non-native version).
    #[cfg(not(feature = "native"))]
    pub fn insert(&self, key: String, states: ModelHiddenStates) {
        // Evict if at capacity
        self.evict_if_needed();

        let mut entries = self.entries.borrow_mut();
        entries.insert(
            key.clone(),
            CacheEntry {
                states,
                created_at: std::time::SystemTime::now(),
            },
        );

        if self.config.use_lru {
            let mut order = self.access_order.borrow_mut();
            order.push(key);
        }
    }

    /// Evict entries if cache is at capacity.
    #[cfg(feature = "native")]
    fn evict_if_needed(&self) {
        let should_evict = self
            .entries
            .read()
            .map(|e| e.len() >= self.config.max_entries)
            .unwrap_or(false);

        if should_evict
            && self.config.use_lru
            && let (Ok(mut entries), Ok(mut order)) =
                (self.entries.write(), self.access_order.write())
        {
            while entries.len() >= self.config.max_entries && !order.is_empty() {
                let oldest = order.remove(0);
                entries.remove(&oldest);
            }
        }
    }

    /// Evict entries if cache is at capacity (non-native version).
    #[cfg(not(feature = "native"))]
    fn evict_if_needed(&self) {
        let entries_len = self.entries.borrow().len();
        if entries_len >= self.config.max_entries && self.config.use_lru {
            let mut entries = self.entries.borrow_mut();
            let mut order = self.access_order.borrow_mut();
            while entries.len() >= self.config.max_entries && !order.is_empty() {
                let oldest = order.remove(0);
                entries.remove(&oldest);
            }
        }
    }

    /// Clear the cache.
    #[cfg(feature = "native")]
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
        }
        if let Ok(mut order) = self.access_order.write() {
            order.clear();
        }
    }

    /// Clear the cache (non-native version).
    #[cfg(not(feature = "native"))]
    pub fn clear(&self) {
        self.entries.borrow_mut().clear();
        self.access_order.borrow_mut().clear();
    }

    /// Get the number of entries in the cache.
    #[cfg(feature = "native")]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    /// Get the number of entries in the cache (non-native version).
    #[cfg(not(feature = "native"))]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for HiddenStateCache {
    fn default() -> Self {
        Self::new(HiddenStateCacheConfig::default())
    }
}

/// Trait for models that can provide hidden states.
#[async_trait]
pub trait HiddenStateProvider: Send + Sync {
    /// Get hidden states for the given input text.
    async fn get_hidden_states(&self, text: &str) -> Result<ModelHiddenStates, SpeculatorError>;

    /// Get hidden states for the given token IDs.
    async fn get_hidden_states_for_tokens(
        &self,
        tokens: &[u32],
    ) -> Result<ModelHiddenStates, SpeculatorError>;

    /// Get hidden states with KV cache for efficient incremental decoding.
    async fn get_hidden_states_with_cache(
        &self,
        tokens: &[u32],
        past_kv: Option<&ModelKVCache>,
    ) -> Result<(ModelHiddenStates, ModelKVCache), SpeculatorError>;

    /// Get the model ID.
    fn model_id(&self) -> &str;

    /// Get the hidden dimension.
    fn hidden_dim(&self) -> usize;

    /// Get the number of layers.
    fn num_layers(&self) -> usize;
}

/// Strategy for reusing hidden states.
pub trait StateReuseStrategy: Send + Sync {
    /// Determine if cached states can be reused for the given input.
    fn can_reuse(&self, cached: &ModelHiddenStates, new_tokens: &[u32]) -> bool;

    /// Determine which layers' states can be reused.
    fn reusable_layers(&self, cached: &ModelHiddenStates, new_tokens: &[u32]) -> Vec<usize>;
}

/// Simple prefix-based state reuse strategy.
pub struct PrefixReuseStrategy {
    /// Minimum prefix length required for reuse.
    pub min_prefix_length: usize,
}

impl Default for PrefixReuseStrategy {
    fn default() -> Self {
        Self {
            min_prefix_length: 4,
        }
    }
}

impl StateReuseStrategy for PrefixReuseStrategy {
    fn can_reuse(&self, cached: &ModelHiddenStates, new_tokens: &[u32]) -> bool {
        if cached.input_tokens.len() < self.min_prefix_length {
            return false;
        }

        let prefix_len = cached.input_tokens.len().min(new_tokens.len());
        cached.input_tokens[..prefix_len] == new_tokens[..prefix_len]
    }

    fn reusable_layers(&self, cached: &ModelHiddenStates, new_tokens: &[u32]) -> Vec<usize> {
        if self.can_reuse(cached, new_tokens) {
            (0..cached.num_layers()).collect()
        } else {
            Vec::new()
        }
    }
}

/// Mock hidden state provider for testing.
pub struct MockHiddenStateProvider {
    model_id: String,
    hidden_dim: usize,
    num_layers: usize,
}

impl MockHiddenStateProvider {
    /// Create a new mock hidden state provider.
    #[must_use]
    pub fn new(hidden_dim: usize, num_layers: usize) -> Self {
        Self {
            model_id: "mock-hidden-state-provider".to_string(),
            hidden_dim,
            num_layers,
        }
    }

    /// Set the model ID.
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    fn generate_mock_hidden_states(&self, tokens: &[u32]) -> ModelHiddenStates {
        let seq_len = tokens.len();
        let batch_size = 1;

        let layers: Vec<LayerHiddenState> = (0..self.num_layers)
            .map(|layer_idx| {
                let num_values = batch_size * seq_len * self.hidden_dim;
                let values: Vec<f32> = (0..num_values)
                    .map(|i| {
                        // Generate deterministic pseudo-random values
                        let seed = (layer_idx as u64 * 1000 + i as u64)
                            ^ u64::from(tokens.first().copied().unwrap_or(0));
                        #[allow(clippy::cast_precision_loss)]
                        let value = ((seed % 10000) as f32 / 5000.0) - 1.0;
                        value
                    })
                    .collect();

                LayerHiddenState::new(layer_idx, values, [batch_size, seq_len, self.hidden_dim])
            })
            .collect();

        // Generate pooled representation
        let pooled: Vec<f32> = (0..self.hidden_dim)
            .map(|i| {
                let seed = i as u64 ^ u64::from(tokens.first().copied().unwrap_or(0));
                #[allow(clippy::cast_precision_loss)]
                let value = ((seed % 10000) as f32 / 5000.0) - 1.0;
                value
            })
            .collect();

        ModelHiddenStates::new(&self.model_id, layers, tokens.to_vec()).with_pooled(pooled)
    }
}

#[async_trait]
impl HiddenStateProvider for MockHiddenStateProvider {
    async fn get_hidden_states(&self, text: &str) -> Result<ModelHiddenStates, SpeculatorError> {
        // Mock tokenization: just use character bytes as tokens
        let tokens: Vec<u32> = text.bytes().map(u32::from).collect();
        Ok(self.generate_mock_hidden_states(&tokens))
    }

    async fn get_hidden_states_for_tokens(
        &self,
        tokens: &[u32],
    ) -> Result<ModelHiddenStates, SpeculatorError> {
        Ok(self.generate_mock_hidden_states(tokens))
    }

    async fn get_hidden_states_with_cache(
        &self,
        tokens: &[u32],
        _past_kv: Option<&ModelKVCache>,
    ) -> Result<(ModelHiddenStates, ModelKVCache), SpeculatorError> {
        let states = self.generate_mock_hidden_states(tokens);
        let kv_cache = ModelKVCache::new(&self.model_id, 12, 64);
        Ok((states, kv_cache))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_hidden_state_creation() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let state = LayerHiddenState::new(0, values, [1, 2, 3]);

        assert_eq!(state.layer_idx, 0);
        assert_eq!(state.batch_size(), 1);
        assert_eq!(state.sequence_length(), 2);
        assert_eq!(state.hidden_dim(), 3);
    }

    #[test]
    fn test_layer_hidden_state_at_position() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let state = LayerHiddenState::new(0, values, [1, 2, 3]);

        let pos0 = state.at_position(0, 0).unwrap();
        assert_eq!(pos0, &[0.1, 0.2, 0.3]);

        let pos1 = state.at_position(0, 1).unwrap();
        assert_eq!(pos1, &[0.4, 0.5, 0.6]);

        assert!(state.at_position(0, 2).is_none());
        assert!(state.at_position(1, 0).is_none());
    }

    #[test]
    fn test_model_hidden_states() {
        let layer0 = LayerHiddenState::new(0, vec![0.1; 6], [1, 2, 3]);
        let layer1 = LayerHiddenState::new(1, vec![0.2; 6], [1, 2, 3]);

        let states = ModelHiddenStates::new("test-model", vec![layer0, layer1], vec![1, 2]);

        assert_eq!(states.model_id, "test-model");
        assert_eq!(states.num_layers(), 2);
        assert_eq!(states.input_tokens, vec![1, 2]);
        assert!(states.layer(0).is_some());
        assert!(states.layer(1).is_some());
        assert!(states.layer(2).is_none());
    }

    #[test]
    fn test_cosine_similarity() {
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![0.0, 1.0, 0.0];
        let vec_c = vec![1.0, 0.0, 0.0];

        // Orthogonal vectors
        assert!((cosine_similarity_vec(&vec_a, &vec_b) - 0.0).abs() < 1e-5);

        // Same vectors
        assert!((cosine_similarity_vec(&vec_a, &vec_c) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_kv_cache() {
        let mut cache = ModelKVCache::new("test-model", 12, 64);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.add_layer(0, vec![0.1; 768], vec![0.2; 768]);
        cache.set_seq_len(10);

        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_hidden_state_cache() {
        let config = HiddenStateCacheConfig {
            max_entries: 3,
            ..Default::default()
        };
        let cache = HiddenStateCache::new(config);

        let states1 = ModelHiddenStates::new("model", vec![], vec![1, 2]);
        let states2 = ModelHiddenStates::new("model", vec![], vec![3, 4]);
        let states3 = ModelHiddenStates::new("model", vec![], vec![5, 6]);
        let states4 = ModelHiddenStates::new("model", vec![], vec![7, 8]);

        cache.insert("key1".to_string(), states1);
        cache.insert("key2".to_string(), states2);
        cache.insert("key3".to_string(), states3);

        assert_eq!(cache.len(), 3);

        // Should evict key1 when adding key4
        cache.insert("key4".to_string(), states4);
        assert_eq!(cache.len(), 3);
        assert!(cache.get("key1").is_none());
        assert!(cache.get("key4").is_some());
    }

    #[test]
    fn test_hidden_state_cache_make_key() {
        let key1 = HiddenStateCache::make_key("model1", b"test content");
        let key2 = HiddenStateCache::make_key("model1", b"test content");
        let key3 = HiddenStateCache::make_key("model1", b"different content");
        let key4 = HiddenStateCache::make_key("model2", b"test content");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_hidden_state_cache_make_key_from_tokens() {
        let key1 = HiddenStateCache::make_key_from_tokens("model1", &[1, 2, 3]);
        let key2 = HiddenStateCache::make_key_from_tokens("model1", &[1, 2, 3]);
        let key3 = HiddenStateCache::make_key_from_tokens("model1", &[1, 2, 4]);
        let key4 = HiddenStateCache::make_key_from_tokens("model2", &[1, 2, 3]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_prefix_reuse_strategy() {
        let strategy = PrefixReuseStrategy {
            min_prefix_length: 2,
        };

        let cached = ModelHiddenStates::new("model", vec![], vec![1, 2, 3, 4]);

        // Can reuse with matching prefix
        assert!(strategy.can_reuse(&cached, &[1, 2, 3, 4, 5]));

        // Cannot reuse with different prefix
        assert!(!strategy.can_reuse(&cached, &[1, 3, 3, 4]));

        // Short cached states
        let short_cached = ModelHiddenStates::new("model", vec![], vec![1]);
        assert!(!strategy.can_reuse(&short_cached, &[1, 2]));
    }

    #[tokio::test]
    async fn test_mock_hidden_state_provider() {
        let provider = MockHiddenStateProvider::new(768, 12);

        assert_eq!(provider.model_id(), "mock-hidden-state-provider");
        assert_eq!(provider.hidden_dim(), 768);
        assert_eq!(provider.num_layers(), 12);

        let states = provider.get_hidden_states("test").await.unwrap();
        assert_eq!(states.num_layers(), 12);
        assert!(states.pooled.is_some());
    }

    #[tokio::test]
    async fn test_mock_hidden_state_provider_for_tokens() {
        let provider = MockHiddenStateProvider::new(384, 6);
        let tokens = vec![1, 2, 3, 4, 5];

        let states = provider
            .get_hidden_states_for_tokens(&tokens)
            .await
            .unwrap();

        assert_eq!(states.input_tokens, tokens);
        assert_eq!(states.num_layers(), 6);

        // Check that all layers have correct shape
        for layer in &states.layers {
            assert_eq!(layer.batch_size(), 1);
            assert_eq!(layer.sequence_length(), 5);
            assert_eq!(layer.hidden_dim(), 384);
        }
    }

    #[tokio::test]
    async fn test_mock_hidden_state_provider_with_cache() {
        let provider = MockHiddenStateProvider::new(256, 4);
        let tokens = vec![10, 20, 30];

        let (states, kv_cache) = provider
            .get_hidden_states_with_cache(&tokens, None)
            .await
            .unwrap();

        assert_eq!(states.input_tokens, tokens);
        assert_eq!(kv_cache.model_id, provider.model_id());
    }

    #[test]
    fn test_hidden_state_cache_clear() {
        let cache = HiddenStateCache::default();
        let states = ModelHiddenStates::new("model", vec![], vec![1]);

        cache.insert("key1".to_string(), states.clone());
        cache.insert("key2".to_string(), states);

        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_layer_hidden_state_with_attention() {
        let values = vec![0.1; 12];
        let attention = vec![0.25; 4];
        let state =
            LayerHiddenState::new(0, values, [1, 2, 6]).with_attention_weights(attention.clone());

        assert!(state.attention_weights.is_some());
        assert_eq!(state.attention_weights.unwrap(), attention);
    }

    #[test]
    fn test_model_hidden_states_cosine_similarity() {
        let pooled1 = vec![1.0, 0.0, 0.0, 0.0];
        let pooled2 = vec![0.707, 0.707, 0.0, 0.0];

        let states1 = ModelHiddenStates::new("model", vec![], vec![1]).with_pooled(pooled1);
        let states2 = ModelHiddenStates::new("model", vec![], vec![2]).with_pooled(pooled2);

        let sim = states1.cosine_similarity(&states2);
        assert!((sim - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_hidden_state_cache_config_default() {
        let config = HiddenStateCacheConfig::default();
        assert_eq!(config.max_entries, 1000);
        assert!(!config.cache_attention);
        assert_eq!(config.ttl_seconds, 3600);
        assert!(config.use_lru);
    }
}
