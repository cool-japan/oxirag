//! Embedding cache with LRU eviction.
//!
//! This module provides a caching wrapper for embedding providers that stores
//! computed embeddings in an LRU cache to avoid redundant computations.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::EmbeddingProvider;

/// Configuration for the embedding cache.
#[derive(Debug, Clone)]
pub struct EmbeddingCacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Whether to track cache statistics.
    pub track_stats: bool,
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            track_stats: true,
        }
    }
}

impl EmbeddingCacheConfig {
    /// Create a new cache configuration.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            track_stats: true,
        }
    }

    /// Set whether to track cache statistics.
    #[must_use]
    pub fn with_stats_tracking(mut self, track: bool) -> Self {
        self.track_stats = track;
        self
    }
}

/// Statistics for the embedding cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of evictions.
    pub evictions: u64,
    /// Current number of entries.
    pub entries: usize,
}

impl CacheStats {
    /// Calculate the hit rate as a percentage.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                (self.hits as f64 / total as f64) * 100.0
            }
        }
    }
}

/// An entry in the LRU cache.
struct CacheEntry {
    /// The cached embedding.
    embedding: Vec<f32>,
    /// The last access time (used for LRU ordering).
    last_access: u64,
}

/// A caching wrapper for embedding providers with LRU eviction.
///
/// This wrapper caches embeddings based on the input text hash to avoid
/// redundant embedding computations. When the cache is full, the least
/// recently used entries are evicted.
pub struct CachedEmbeddingProvider<P: EmbeddingProvider> {
    /// The underlying embedding provider.
    provider: P,
    /// The cache configuration.
    config: EmbeddingCacheConfig,
    /// The cache storage (text hash -> embedding).
    cache: RwLock<HashMap<u64, CacheEntry>>,
    /// Monotonically increasing counter for LRU ordering.
    access_counter: AtomicU64,
    /// Cache statistics.
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl<P: EmbeddingProvider> CachedEmbeddingProvider<P> {
    /// Create a new cached embedding provider.
    #[must_use]
    pub fn new(provider: P, config: EmbeddingCacheConfig) -> Self {
        Self {
            provider,
            config,
            cache: RwLock::new(HashMap::new()),
            access_counter: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Create a cached provider with default configuration.
    #[must_use]
    pub fn with_defaults(provider: P) -> Self {
        Self::new(provider, EmbeddingCacheConfig::default())
    }

    /// Get the underlying provider.
    #[must_use]
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Get cache statistics.
    ///
    /// # Panics
    ///
    /// Panics if the cache lock is poisoned.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().expect("cache lock poisoned");
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            entries: cache.len(),
        }
    }

    /// Clear the cache.
    ///
    /// # Panics
    ///
    /// Panics if the cache lock is poisoned.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().expect("cache lock poisoned");
        cache.clear();
    }

    /// Compute a hash for the given text.
    fn hash_text(text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Try to get an embedding from the cache.
    fn get_cached(&self, text_hash: u64) -> Option<Vec<f32>> {
        let mut cache = self.cache.write().expect("cache lock poisoned");
        if let Some(entry) = cache.get_mut(&text_hash) {
            // Update last access time
            entry.last_access = self.access_counter.fetch_add(1, Ordering::Relaxed);
            if self.config.track_stats {
                self.hits.fetch_add(1, Ordering::Relaxed);
            }
            Some(entry.embedding.clone())
        } else {
            if self.config.track_stats {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
            None
        }
    }

    /// Insert an embedding into the cache, evicting LRU entries if necessary.
    fn insert_cached(&self, text_hash: u64, embedding: Vec<f32>) {
        let mut cache = self.cache.write().expect("cache lock poisoned");

        // Evict if necessary
        while cache.len() >= self.config.max_entries {
            // Find the least recently used entry
            let lru_key = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(k, _)| *k);

            if let Some(key) = lru_key {
                cache.remove(&key);
                if self.config.track_stats {
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }

        let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);
        cache.insert(
            text_hash,
            CacheEntry {
                embedding,
                last_access: access_time,
            },
        );
    }
}

#[async_trait]
impl<P: EmbeddingProvider> EmbeddingProvider for CachedEmbeddingProvider<P> {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let text_hash = Self::hash_text(text);

        // Check cache first
        if let Some(embedding) = self.get_cached(text_hash) {
            return Ok(embedding);
        }

        // Cache miss - compute embedding
        let embedding = self.provider.embed(text).await?;

        // Store in cache
        self.insert_cached(text_hash, embedding.clone());

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Separate cached and uncached texts
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut uncached_indices: Vec<usize> = Vec::new();
        let mut uncached_texts: Vec<&str> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let text_hash = Self::hash_text(text);
            if let Some(embedding) = self.get_cached(text_hash) {
                results[i] = Some(embedding);
            } else {
                uncached_indices.push(i);
                uncached_texts.push(text);
            }
        }

        // If all cached, return immediately
        if uncached_texts.is_empty() {
            return Ok(results.into_iter().flatten().collect());
        }

        // Compute embeddings for uncached texts
        let new_embeddings = self.provider.embed_batch(&uncached_texts).await?;

        // Store new embeddings in cache and results
        for (i, embedding) in uncached_indices.into_iter().zip(new_embeddings) {
            let text_hash = Self::hash_text(texts[i]);
            self.insert_cached(text_hash, embedding.clone());
            results[i] = Some(embedding);
        }

        Ok(results.into_iter().flatten().collect())
    }

    fn dimension(&self) -> usize {
        self.provider.dimension()
    }

    fn model_id(&self) -> &str {
        self.provider.model_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer1_echo::embedding::MockEmbeddingProvider;

    #[tokio::test]
    async fn test_cache_hit() {
        let provider = MockEmbeddingProvider::new(64);
        let cached = CachedEmbeddingProvider::with_defaults(provider);

        // First call - cache miss
        let emb1 = cached.embed("test text").await.unwrap();
        assert_eq!(cached.stats().hits, 0);
        assert_eq!(cached.stats().misses, 1);

        // Second call - cache hit
        let emb2 = cached.embed("test text").await.unwrap();
        assert_eq!(cached.stats().hits, 1);
        assert_eq!(cached.stats().misses, 1);

        // Embeddings should be identical
        assert_eq!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_cache_different_texts() {
        let provider = MockEmbeddingProvider::new(64);
        let cached = CachedEmbeddingProvider::with_defaults(provider);

        cached.embed("text 1").await.unwrap();
        cached.embed("text 2").await.unwrap();

        assert_eq!(cached.stats().misses, 2);
        assert_eq!(cached.stats().entries, 2);
    }

    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let provider = MockEmbeddingProvider::new(32);
        let config = EmbeddingCacheConfig::new(3);
        let cached = CachedEmbeddingProvider::new(provider, config);

        // Fill the cache
        cached.embed("text 1").await.unwrap();
        cached.embed("text 2").await.unwrap();
        cached.embed("text 3").await.unwrap();
        assert_eq!(cached.stats().entries, 3);
        assert_eq!(cached.stats().evictions, 0);

        // Access text 1 to make it recently used
        cached.embed("text 1").await.unwrap();

        // Add a new entry - should evict text 2 (LRU)
        cached.embed("text 4").await.unwrap();
        assert_eq!(cached.stats().entries, 3);
        assert_eq!(cached.stats().evictions, 1);

        // text 1 should still be cached (was recently accessed)
        let stats_before = cached.stats();
        cached.embed("text 1").await.unwrap();
        assert_eq!(cached.stats().hits, stats_before.hits + 1);
    }

    #[tokio::test]
    async fn test_cache_batch_partial_hit() {
        let provider = MockEmbeddingProvider::new(32);
        let cached = CachedEmbeddingProvider::with_defaults(provider);

        // Pre-populate cache with some texts
        cached.embed("cached text").await.unwrap();

        // Batch with mixed cached and uncached
        let texts = vec!["cached text", "new text 1", "new text 2"];
        let embeddings = cached.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        // 1 hit for "cached text", 2 misses for new texts + 1 initial miss
        assert_eq!(cached.stats().hits, 1);
        assert_eq!(cached.stats().misses, 3); // 1 initial + 2 from batch
    }

    #[tokio::test]
    async fn test_cache_batch_all_cached() {
        let provider = MockEmbeddingProvider::new(32);
        let cached = CachedEmbeddingProvider::with_defaults(provider);

        // Pre-populate cache
        cached.embed("text 1").await.unwrap();
        cached.embed("text 2").await.unwrap();

        let stats_before = cached.stats();

        // All texts are cached
        let embeddings = cached.embed_batch(&["text 1", "text 2"]).await.unwrap();
        assert_eq!(embeddings.len(), 2);

        // Should have 2 more hits, no additional misses
        assert_eq!(cached.stats().hits, stats_before.hits + 2);
        assert_eq!(cached.stats().misses, stats_before.misses);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let provider = MockEmbeddingProvider::new(32);
        let cached = CachedEmbeddingProvider::with_defaults(provider);

        cached.embed("text 1").await.unwrap();
        cached.embed("text 2").await.unwrap();
        assert_eq!(cached.stats().entries, 2);

        cached.clear_cache();
        assert_eq!(cached.stats().entries, 0);

        // After clear, should be cache miss
        cached.embed("text 1").await.unwrap();
        assert_eq!(cached.stats().misses, 3); // 2 initial + 1 after clear
    }

    #[tokio::test]
    async fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 75,
            misses: 25,
            evictions: 0,
            entries: 100,
        };

        assert!((stats.hit_rate() - 75.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_cache_stats_hit_rate_zero() {
        let stats = CacheStats::default();
        assert!((stats.hit_rate() - 0.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_cache_config_builder() {
        let config = EmbeddingCacheConfig::new(5000).with_stats_tracking(false);

        assert_eq!(config.max_entries, 5000);
        assert!(!config.track_stats);
    }

    #[tokio::test]
    async fn test_cache_dimension_passthrough() {
        let provider = MockEmbeddingProvider::new(256);
        let cached = CachedEmbeddingProvider::with_defaults(provider);

        assert_eq!(cached.dimension(), 256);
    }

    #[tokio::test]
    async fn test_cache_model_id_passthrough() {
        let provider = MockEmbeddingProvider::new(64).with_model_id("test-model");
        let cached = CachedEmbeddingProvider::with_defaults(provider);

        assert_eq!(cached.model_id(), "test-model");
    }

    #[tokio::test]
    async fn test_cache_empty_batch() {
        let provider = MockEmbeddingProvider::new(32);
        let cached = CachedEmbeddingProvider::with_defaults(provider);

        let result = cached.embed_batch(&[]).await;
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }
}
