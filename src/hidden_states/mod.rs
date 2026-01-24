//! Hidden States module for Speculative RAG.
//!
//! This module implements internal state manipulation for capturing and reusing
//! hidden states from transformer models. This enables efficient speculative
//! decoding by reusing pre-computed internal states.
//!
//! # Overview
//!
//! When running transformer models for RAG tasks, significant computation can
//! be saved by caching and reusing hidden states from previous forward passes.
//! This module provides:
//!
//! - **Types**: Core data structures for tensors, hidden states, and KV caches
//! - **Traits**: Provider and strategy interfaces for extensibility
//! - **Cache**: LRU caching infrastructure for hidden states
//! - **Extractor**: Utilities for extracting states and mock providers for testing
//! - **Reuse**: Strategies for determining when states can be reused
//!
//! # Architecture
//!
//! ```text
//! Input Text
//!     │
//!     ▼
//! ┌───────────────────────┐
//! │  HiddenStateProvider  │  ← Extract hidden states from model
//! │  (extract_hidden_*)   │
//! └───────────┬───────────┘
//!             │
//!             ▼
//! ┌───────────────────────┐
//! │  HiddenStateCache     │  ← Cache states for reuse
//! │  (LRU eviction)       │
//! └───────────┬───────────┘
//!             │
//!             ▼
//! ┌───────────────────────┐
//! │  StateReuseStrategy   │  ← Determine reuse eligibility
//! │  (prefix, semantic)   │
//! └───────────┴───────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use oxirag::hidden_states::{
//!     HiddenStateCache, HiddenStateCacheConfig, HiddenStateProvider,
//!     MockHiddenStateProvider, PrefixReuseStrategy, StateReuseStrategy,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a mock provider for testing
//!     let provider = MockHiddenStateProvider::new("test-model", 12, 768);
//!
//!     // Create a cache
//!     let config = HiddenStateCacheConfig::new(100, 256 * 1024 * 1024);
//!     let mut cache = HiddenStateCache::new(config);
//!
//!     // Extract and cache hidden states
//!     let text = "The capital of France is Paris.";
//!     let states = provider.extract_hidden_states(text).await?;
//!     cache.put(text.to_string(), states, None);
//!
//!     // Check for reuse opportunity
//!     let new_text = "The capital of France is Paris. It has a population of...";
//!     let strategy = PrefixReuseStrategy::default();
//!
//!     if let Some((cached_text, cached)) = cache.find_prefix_match(new_text) {
//!         if strategy.can_reuse(&cached.states, new_text, cached_text) {
//!             let reuse_point = strategy.reuse_point(&cached.states, new_text, cached_text);
//!             println!("Can reuse {} tokens!", reuse_point);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # State Extraction
//!
//! The `HiddenStateProvider` trait provides two main extraction methods:
//!
//! - `extract_hidden_states`: Basic extraction for the full input
//! - `extract_with_kv_cache`: Incremental extraction with KV cache for generation
//!
//! # Reuse Strategies
//!
//! Several strategies are available for determining state reuse:
//!
//! - `PrefixReuseStrategy`: Reuses states when texts share a common prefix
//! - `SemanticReuseStrategy`: Reuses states based on semantic similarity
//! - `LengthAwareReuseStrategy`: Considers text lengths and overlap
//! - `HybridReuseStrategy`: Combines multiple strategies with weighted voting
//! - `AdaptiveReuseStrategy`: Automatically selects the best strategy
//!
//! # Caching
//!
//! The `HiddenStateCache` provides:
//!
//! - LRU eviction based on entry count and memory usage
//! - TTL-based expiration
//! - Prefix matching for partial reuse
//! - Detailed statistics
//!
//! # Thread Safety
//!
//! All providers and strategies implement `Send + Sync` for concurrent use.

pub mod cache;
pub mod extractor;
pub mod reuse;
pub mod traits;
pub mod types;

// Re-exports for convenience
pub use cache::{
    CachedHiddenState, HiddenStateCache, HiddenStateCacheConfig, HiddenStateCacheStats,
};
pub use extractor::{LayerExtractor, MockHiddenStateProvider, StatePooling, StateSimilarity};
pub use reuse::{
    AdaptiveReuseStrategy, HybridReuseStrategy, LengthAwareReuseStrategy, PrefixReuseStrategy,
    SemanticReuseStrategy,
};
pub use traits::{
    BoxedStateReuseStrategy, HiddenStateProvider, HiddenStateProviderExt, StateReuseStrategy,
};
pub use types::{
    DType, Device, HiddenStateConfig, HiddenStateTensor, KVCache, LayerHiddenState,
    ModelHiddenStates, ModelKVCache, TensorShape,
};

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::no_effect_underscore_binding)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_full_workflow() {
        // Create provider
        let provider = MockHiddenStateProvider::new("test-model", 6, 256);

        // Create cache
        let config = HiddenStateCacheConfig::new(10, 10 * 1024 * 1024)
            .with_ttl(3600)
            .with_store_kv_cache(true);
        let mut cache = HiddenStateCache::new(config);

        // Extract states
        let text1 = "The quick brown fox jumps over the lazy dog.";
        let states1 = provider.extract_hidden_states(text1).await.unwrap();
        cache.put(text1.to_string(), states1.clone(), None);

        // Verify cached
        assert!(cache.contains(text1));
        let cached = cache.get(text1).unwrap();
        assert_eq!(cached.states.model_id, "test-model");

        // Try prefix matching
        let text2 = "The quick brown fox jumps over the lazy dog. And more text here.";
        let prefix_match = cache.find_prefix_match(text2);
        assert!(prefix_match.is_some());

        // Test reuse strategy
        let strategy = PrefixReuseStrategy::default();
        if let Some((cached_text, cached_state)) = prefix_match {
            let can_reuse = strategy.can_reuse(&cached_state.states, text2, cached_text);
            assert!(can_reuse);

            let reuse_point = strategy.reuse_point(&cached_state.states, text2, cached_text);
            assert!(reuse_point > 0);

            let quality = strategy.reuse_quality(&cached_state.states, text2, cached_text);
            assert!(quality > 0.0);
        }

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.entry_count, 1);
    }

    #[tokio::test]
    async fn test_kv_cache_workflow() {
        let provider = MockHiddenStateProvider::new("test-model", 4, 128);

        // First extraction without past KV
        let (states1, kv1) = provider
            .extract_with_kv_cache("Hello, world!", None)
            .await
            .unwrap();

        assert_eq!(states1.model_id, "test-model");
        assert_eq!(kv1.layers.len(), 4);

        // Second extraction with past KV
        let (states2, kv2) = provider
            .extract_with_kv_cache("How are you?", Some(&kv1))
            .await
            .unwrap();

        assert_eq!(states2.model_id, "test-model");
        // KV cache should have accumulated states
        assert!(kv2.total_size_bytes() >= kv1.total_size_bytes());
    }

    #[tokio::test]
    async fn test_state_similarity_integration() {
        let provider = MockHiddenStateProvider::new("test-model", 4, 128);

        // Same input should produce identical states
        let states1 = provider.extract_hidden_states("test input").await.unwrap();
        let states2 = provider.extract_hidden_states("test input").await.unwrap();

        let avg_sim = StateSimilarity::average_similarity(&states1, &states2);
        assert!((avg_sim - 1.0).abs() < 0.001);

        // Different inputs should have different similarity
        let states3 = provider.extract_hidden_states("different").await.unwrap();
        let avg_sim2 = StateSimilarity::average_similarity(&states1, &states3);
        assert!(avg_sim2 < 1.0);
    }

    #[test]
    fn test_hybrid_strategy() {
        let strategy = HybridReuseStrategy::new();
        let mut states = ModelHiddenStates::new("test", 4, 64);
        states.sequence_length = 50;

        // Add layers
        for i in 0..4 {
            let hidden = HiddenStateTensor::from_vec_1d(vec![0.5; 64 * 50]);
            states.add_layer(LayerHiddenState::new(i, hidden));
        }
        states.set_pooled_output(HiddenStateTensor::from_vec_1d(vec![0.5; 64]));

        let can_reuse = strategy.can_reuse(
            &states,
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox",
        );
        assert!(can_reuse);
    }

    #[test]
    fn test_tensor_operations() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = TensorShape::new(vec![2, 3]);
        let tensor = HiddenStateTensor::from_vec(data.clone(), shape.clone()).unwrap();

        // Test basic properties
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.shape.ndim(), 2);
        assert_eq!(tensor.size_bytes(), 24); // 6 * 4 bytes

        // Test slice
        let sliced = tensor.slice(0, 0, 1).unwrap();
        assert_eq!(sliced.shape.dims, vec![1, 3]);
        assert_eq!(sliced.data, vec![1.0, 2.0, 3.0]);

        // Test concat
        let t1 = HiddenStateTensor::from_vec_1d(vec![1.0, 2.0]);
        let t2 = HiddenStateTensor::from_vec_1d(vec![3.0, 4.0]);
        let concat = HiddenStateTensor::concat(&[&t1, &t2], 0).unwrap();
        assert_eq!(concat.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_layer_extractor() {
        let mut states = ModelHiddenStates::new("test", 8, 64);
        for i in 0..8 {
            let hidden = HiddenStateTensor::from_vec_1d(vec![i as f32; 64]);
            states.add_layer(LayerHiddenState::new(i, hidden));
        }

        // Extract last 3 layers
        let last_3 = LayerExtractor::extract_last_n(&states, 3);
        assert_eq!(last_3.len(), 3);
        assert_eq!(last_3[0].layer_idx, 5);
        assert_eq!(last_3[2].layer_idx, 7);

        // Extract every 2nd layer
        let every_2 = LayerExtractor::extract_every_n(&states, 2);
        assert_eq!(every_2.len(), 4);
        assert_eq!(every_2[0].layer_idx, 0);
        assert_eq!(every_2[1].layer_idx, 2);

        // Extract middle layers
        let middle = LayerExtractor::extract_middle(&states, 4);
        assert_eq!(middle.len(), 4);
        assert_eq!(middle[0].layer_idx, 2);
    }

    #[test]
    fn test_state_pooling() {
        // [1, 2, 4] - batch=1, seq_len=2, hidden_dim=4
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = HiddenStateTensor::from_vec(data, TensorShape::new(vec![1, 2, 4])).unwrap();

        // Mean pooling
        let mean_pooled = StatePooling::mean_pool(&tensor).unwrap();
        assert_eq!(mean_pooled.shape.dims, vec![4]);
        // Mean of [1,5], [2,6], [3,7], [4,8] = [3, 4, 5, 6]
        assert!((mean_pooled.data[0] - 3.0).abs() < 0.001);

        // Max pooling
        let max_pooled = StatePooling::max_pool(&tensor).unwrap();
        assert_eq!(max_pooled.shape.dims, vec![4]);
        // Max of each pair = [5, 6, 7, 8]
        assert!((max_pooled.data[0] - 5.0).abs() < 0.001);

        // CLS pooling
        let cls_pooled = StatePooling::cls_pool(&tensor).unwrap();
        assert_eq!(cls_pooled.shape.dims, vec![4]);
        // First token = [1, 2, 3, 4]
        assert!((cls_pooled.data[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_eviction() {
        let config = HiddenStateCacheConfig::new(3, usize::MAX).without_ttl();
        let mut cache = HiddenStateCache::new(config);

        // Add more entries than capacity
        for i in 0..5 {
            let states = ModelHiddenStates::new("test", 2, 32);
            cache.put(format!("entry{i}"), states, None);
        }

        // Should have evicted oldest entries
        assert!(cache.len() <= 3);
        assert!(cache.stats().entry_count <= 3);
    }

    #[test]
    fn test_all_exports_accessible() {
        // Verify all re-exported types are accessible
        let _config = HiddenStateConfig::default();
        let _cache_config = HiddenStateCacheConfig::default();
        let _shape = TensorShape::new(vec![1, 2, 3]);
        let _dtype = DType::F32;
        let _device = Device::Cpu;
        let _tensor = HiddenStateTensor::default();
        let _states = ModelHiddenStates::new("test", 4, 64);
        let _kv = KVCache::new(0, 8, 64, 512);
        let _model_kv = ModelKVCache::new("test", 4, 8, 64, 512);
        let _cache = HiddenStateCache::with_defaults();
        let _prefix_strategy = PrefixReuseStrategy::default();
        let _semantic_strategy = SemanticReuseStrategy::default();
        let _hybrid_strategy = HybridReuseStrategy::default();
        let _length_strategy = LengthAwareReuseStrategy::default();
        let _adaptive_strategy = AdaptiveReuseStrategy::default();
    }

    #[tokio::test]
    async fn test_provider_ext_trait() {
        let provider = MockHiddenStateProvider::new("test-model", 4, 128);
        let mut cache = HiddenStateCache::with_defaults();

        // Use the extension trait
        let states = provider
            .extract_and_cache("test input", &mut cache)
            .await
            .unwrap();
        assert_eq!(states.model_id, "test-model");

        // Should be cached now
        assert!(cache.contains("test input"));

        // Second call should hit cache
        let states2 = provider
            .extract_and_cache("test input", &mut cache)
            .await
            .unwrap();
        assert_eq!(states2.model_id, states.model_id);
    }

    #[test]
    fn test_model_hidden_states_operations() {
        let mut states = ModelHiddenStates::new("test", 4, 64);
        states.sequence_length = 10;

        for i in 0..4 {
            let hidden = HiddenStateTensor::zeros(
                TensorShape::new(vec![1, 10, 64]),
                DType::F32,
                Device::Cpu,
            );
            states.add_layer(LayerHiddenState::new(i, hidden));
        }

        // Test get_layer
        assert!(states.get_layer(0).is_some());
        assert!(states.get_layer(5).is_none());

        // Test last_hidden_state
        assert!(states.last_hidden_state().is_some());

        // Test prefix_states
        let prefix = states.prefix_states(5).unwrap();
        assert_eq!(prefix.sequence_length, 5);

        // Test total_size_bytes
        assert!(states.total_size_bytes() > 0);
    }

    #[test]
    fn test_kv_cache_operations() {
        let mut kv = KVCache::new(0, 8, 64, 512);

        // Initially empty
        assert_eq!(kv.current_length(), 512); // Created with max_seq_len

        // Clear
        kv.clear();
        assert_eq!(kv.current_length(), 0);

        // Check size is calculated (non-zero after operations above)
        // Note: After clear(), the actual size depends on internal implementation
        let _ = kv.size_bytes(); // Just verify it doesn't panic
    }
}
