//! Hidden state caching for efficient reuse.
//!
//! This module provides caching infrastructure for storing and retrieving
//! hidden states from transformer models.

#![allow(clippy::collapsible_if)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::types::{ModelHiddenStates, ModelKVCache};

/// A cached hidden state entry with metadata.
#[derive(Debug, Clone)]
pub struct CachedHiddenState {
    /// The input text that produced these states.
    pub text: String,
    /// The cached hidden states.
    pub states: ModelHiddenStates,
    /// Optional KV cache for incremental generation.
    pub kv_cache: Option<ModelKVCache>,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
}

impl CachedHiddenState {
    /// Create a new cached hidden state entry.
    #[must_use]
    pub fn new(text: String, states: ModelHiddenStates, kv_cache: Option<ModelKVCache>) -> Self {
        let now = Instant::now();
        Self {
            text,
            states,
            kv_cache,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    /// Record an access to this entry.
    pub fn record_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Get the age of this entry in seconds.
    #[must_use]
    pub fn age_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }

    /// Get the time since last access in seconds.
    #[must_use]
    pub fn idle_secs(&self) -> u64 {
        self.last_accessed.elapsed().as_secs()
    }

    /// Estimate the size of this entry in bytes.
    #[must_use]
    pub fn estimated_size_bytes(&self) -> usize {
        let text_size = self.text.len();
        let states_size = self.states.total_size_bytes();
        let kv_size = self
            .kv_cache
            .as_ref()
            .map_or(0, ModelKVCache::total_size_bytes);
        text_size + states_size + kv_size + std::mem::size_of::<Self>()
    }
}

/// Configuration for the hidden state cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStateCacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Maximum memory usage in bytes.
    pub max_bytes: usize,
    /// Optional time-to-live in seconds for cache entries.
    pub ttl_secs: Option<u64>,
    /// Whether to store KV caches with hidden states.
    pub store_kv_cache: bool,
}

impl Default for HiddenStateCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            max_bytes: 256 * 1024 * 1024, // 256 MB
            ttl_secs: Some(3600),         // 1 hour
            store_kv_cache: true,
        }
    }
}

impl HiddenStateCacheConfig {
    /// Create a new configuration with specified limits.
    #[must_use]
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries,
            max_bytes,
            ..Default::default()
        }
    }

    /// Set the TTL in seconds.
    #[must_use]
    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = Some(ttl_secs);
        self
    }

    /// Disable TTL (entries never expire).
    #[must_use]
    pub fn without_ttl(mut self) -> Self {
        self.ttl_secs = None;
        self
    }

    /// Set whether to store KV caches.
    #[must_use]
    pub fn with_store_kv_cache(mut self, store: bool) -> Self {
        self.store_kv_cache = store;
        self
    }
}

/// Cache for storing hidden states with LRU eviction.
pub struct HiddenStateCache {
    config: HiddenStateCacheConfig,
    entries: HashMap<String, CachedHiddenState>,
    access_order: VecDeque<String>,
    total_bytes: usize,
}

impl HiddenStateCache {
    /// Create a new hidden state cache with the given configuration.
    #[must_use]
    pub fn new(config: HiddenStateCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            total_bytes: 0,
        }
    }

    /// Create a cache with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(HiddenStateCacheConfig::default())
    }

    /// Get a cached entry by text key.
    ///
    /// Updates access time and moves to front of LRU queue if found.
    pub fn get(&mut self, text: &str) -> Option<&CachedHiddenState> {
        // Check TTL first
        if let Some(ttl) = self.config.ttl_secs {
            if let Some(entry) = self.entries.get(text) {
                if entry.age_secs() > ttl {
                    // Entry expired, remove it
                    self.remove(text);
                    return None;
                }
            }
        }

        if self.entries.contains_key(text) {
            // Update access time
            if let Some(entry) = self.entries.get_mut(text) {
                entry.record_access();
            }

            // Move to front of LRU queue
            self.access_order.retain(|k| k != text);
            self.access_order.push_front(text.to_string());

            self.entries.get(text)
        } else {
            None
        }
    }

    /// Put a new entry in the cache.
    ///
    /// Evicts old entries if necessary to make room.
    pub fn put(&mut self, text: String, states: ModelHiddenStates, kv_cache: Option<ModelKVCache>) {
        let kv_to_store = if self.config.store_kv_cache {
            kv_cache
        } else {
            None
        };

        let entry = CachedHiddenState::new(text.clone(), states, kv_to_store);
        let entry_size = entry.estimated_size_bytes();

        // Evict if necessary
        self.evict_if_needed(entry_size);

        // Remove old entry if exists
        if let Some(old) = self.entries.remove(&text) {
            self.total_bytes = self.total_bytes.saturating_sub(old.estimated_size_bytes());
            self.access_order.retain(|k| k != &text);
        }

        // Add new entry
        self.total_bytes += entry_size;
        self.entries.insert(text.clone(), entry);
        self.access_order.push_front(text);
    }

    /// Remove an entry from the cache.
    pub fn remove(&mut self, text: &str) -> Option<CachedHiddenState> {
        if let Some(entry) = self.entries.remove(text) {
            self.total_bytes = self
                .total_bytes
                .saturating_sub(entry.estimated_size_bytes());
            self.access_order.retain(|k| k != text);
            Some(entry)
        } else {
            None
        }
    }

    /// Clear all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.total_bytes = 0;
    }

    /// Get the number of entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the total bytes used by the cache.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Find the best prefix match for the given text.
    ///
    /// Returns the key and entry for the longest cached text that is a prefix
    /// of the input text.
    #[must_use]
    pub fn find_prefix_match(&self, text: &str) -> Option<(&str, &CachedHiddenState)> {
        let mut best_match: Option<(&str, &CachedHiddenState)> = None;
        let mut best_len = 0;

        for (key, entry) in &self.entries {
            if text.starts_with(key) && key.len() > best_len {
                // Check TTL
                if let Some(ttl) = self.config.ttl_secs {
                    if entry.age_secs() > ttl {
                        continue;
                    }
                }
                best_len = key.len();
                best_match = Some((key.as_str(), entry));
            }
        }

        best_match
    }

    /// Find entries where the input text is a prefix.
    ///
    /// These entries might have reusable hidden states for the current input.
    #[must_use]
    pub fn find_extensions(&self, text: &str) -> Vec<(&str, &CachedHiddenState)> {
        self.entries
            .iter()
            .filter(|(key, entry)| {
                key.starts_with(text) && {
                    if let Some(ttl) = self.config.ttl_secs {
                        entry.age_secs() <= ttl
                    } else {
                        true
                    }
                }
            })
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    /// Evict entries to make room for a new entry.
    fn evict_if_needed(&mut self, required_bytes: usize) {
        // Evict expired entries first
        self.evict_expired();

        // Evict LRU entries until we have enough space
        while (self.entries.len() >= self.config.max_entries
            || self.total_bytes + required_bytes > self.config.max_bytes)
            && !self.access_order.is_empty()
        {
            if let Some(oldest_key) = self.access_order.pop_back() {
                if let Some(entry) = self.entries.remove(&oldest_key) {
                    self.total_bytes = self
                        .total_bytes
                        .saturating_sub(entry.estimated_size_bytes());
                }
            }
        }
    }

    /// Evict all expired entries.
    pub fn evict_expired(&mut self) -> usize {
        let Some(ttl) = self.config.ttl_secs else {
            return 0;
        };

        let expired_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.age_secs() > ttl)
            .map(|(key, _)| key.clone())
            .collect();

        let count = expired_keys.len();

        for key in expired_keys {
            self.remove(&key);
        }

        count
    }

    /// Get cache statistics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> HiddenStateCacheStats {
        let total_access_count: u64 = self.entries.values().map(|e| e.access_count).sum();
        let avg_age = if self.entries.is_empty() {
            0.0
        } else {
            let total_age: u64 = self.entries.values().map(CachedHiddenState::age_secs).sum();
            total_age as f64 / self.entries.len() as f64
        };

        HiddenStateCacheStats {
            entry_count: self.entries.len(),
            total_bytes: self.total_bytes,
            max_entries: self.config.max_entries,
            max_bytes: self.config.max_bytes,
            total_access_count,
            average_age_secs: avg_age,
        }
    }

    /// Get the cache configuration.
    #[must_use]
    pub fn config(&self) -> &HiddenStateCacheConfig {
        &self.config
    }

    /// Check if the cache contains an entry for the given text.
    #[must_use]
    pub fn contains(&self, text: &str) -> bool {
        if let Some(entry) = self.entries.get(text) {
            if let Some(ttl) = self.config.ttl_secs {
                return entry.age_secs() <= ttl;
            }
            return true;
        }
        false
    }

    /// Get all keys in the cache (for iteration).
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }
}

impl Default for HiddenStateCache {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Statistics about the hidden state cache.
#[derive(Debug, Clone, Default)]
pub struct HiddenStateCacheStats {
    /// Number of entries in the cache.
    pub entry_count: usize,
    /// Total memory usage in bytes.
    pub total_bytes: usize,
    /// Maximum number of entries allowed.
    pub max_entries: usize,
    /// Maximum memory allowed in bytes.
    pub max_bytes: usize,
    /// Total number of accesses across all entries.
    pub total_access_count: u64,
    /// Average age of entries in seconds.
    pub average_age_secs: f64,
}

impl HiddenStateCacheStats {
    /// Get the utilization as a percentage of max entries.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn entry_utilization(&self) -> f64 {
        if self.max_entries == 0 {
            0.0
        } else {
            (self.entry_count as f64 / self.max_entries as f64) * 100.0
        }
    }

    /// Get the memory utilization as a percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn memory_utilization(&self) -> f64 {
        if self.max_bytes == 0 {
            0.0
        } else {
            (self.total_bytes as f64 / self.max_bytes as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hidden_states::types::{
        DType, Device, HiddenStateTensor, LayerHiddenState, TensorShape,
    };

    fn create_test_states(model_id: &str) -> ModelHiddenStates {
        let mut states = ModelHiddenStates::new(model_id, 2, 64);
        states.sequence_length = 10;

        let hidden =
            HiddenStateTensor::zeros(TensorShape::new(vec![1, 10, 64]), DType::F32, Device::Cpu);
        states.add_layer(LayerHiddenState::new(0, hidden.clone()));
        states.add_layer(LayerHiddenState::new(1, hidden));

        states
    }

    #[test]
    fn test_cache_config_default() {
        let config = HiddenStateCacheConfig::default();
        assert_eq!(config.max_entries, 100);
        assert_eq!(config.max_bytes, 256 * 1024 * 1024);
        assert_eq!(config.ttl_secs, Some(3600));
        assert!(config.store_kv_cache);
    }

    #[test]
    fn test_cache_config_builder() {
        let config = HiddenStateCacheConfig::new(50, 128 * 1024 * 1024)
            .with_ttl(1800)
            .with_store_kv_cache(false);

        assert_eq!(config.max_entries, 50);
        assert_eq!(config.max_bytes, 128 * 1024 * 1024);
        assert_eq!(config.ttl_secs, Some(1800));
        assert!(!config.store_kv_cache);
    }

    #[test]
    fn test_cache_put_and_get() {
        let mut cache = HiddenStateCache::with_defaults();
        let states = create_test_states("test-model");

        cache.put("hello world".to_string(), states.clone(), None);

        let retrieved = cache.get("hello world");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().states.model_id, "test-model");
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = HiddenStateCache::with_defaults();
        let result = cache.get("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = HiddenStateCache::with_defaults();
        let states = create_test_states("test-model");

        cache.put("test".to_string(), states, None);
        assert_eq!(cache.len(), 1);

        let removed = cache.remove("test");
        assert!(removed.is_some());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = HiddenStateCache::with_defaults();

        for i in 0..5 {
            let states = create_test_states("test-model");
            cache.put(format!("entry{i}"), states, None);
        }

        assert_eq!(cache.len(), 5);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let config = HiddenStateCacheConfig::new(3, usize::MAX);
        let mut cache = HiddenStateCache::new(config);

        for i in 0..5 {
            let states = create_test_states("test-model");
            cache.put(format!("entry{i}"), states, None);
        }

        // Should have evicted oldest entries
        assert!(cache.len() <= 3);
    }

    #[test]
    fn test_cache_access_updates_lru() {
        let config = HiddenStateCacheConfig::new(3, usize::MAX).without_ttl();
        let mut cache = HiddenStateCache::new(config);

        // Add 3 entries
        for i in 0..3 {
            let states = create_test_states("test-model");
            cache.put(format!("entry{i}"), states, None);
        }

        // Access entry0 to make it recently used
        let _ = cache.get("entry0");

        // Add a new entry, should evict entry1 (oldest not accessed)
        let states = create_test_states("test-model");
        cache.put("entry3".to_string(), states, None);

        // entry0 should still exist
        assert!(cache.contains("entry0"));
    }

    #[test]
    fn test_cache_find_prefix_match() {
        let mut cache = HiddenStateCache::with_defaults();

        let states1 = create_test_states("test-model");
        cache.put("Hello".to_string(), states1, None);

        let states2 = create_test_states("test-model");
        cache.put("Hello, world".to_string(), states2, None);

        // Find prefix for "Hello, world! How are you?"
        let result = cache.find_prefix_match("Hello, world! How are you?");
        assert!(result.is_some());
        let (key, _) = result.unwrap();
        assert_eq!(key, "Hello, world");
    }

    #[test]
    fn test_cache_find_extensions() {
        let mut cache = HiddenStateCache::with_defaults();

        let states1 = create_test_states("test-model");
        cache.put("Hello, world! How are you?".to_string(), states1, None);

        let states2 = create_test_states("test-model");
        cache.put("Hello, world! What's up?".to_string(), states2, None);

        let states3 = create_test_states("test-model");
        cache.put("Goodbye".to_string(), states3, None);

        let extensions = cache.find_extensions("Hello");
        assert_eq!(extensions.len(), 2);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = HiddenStateCache::with_defaults();

        for i in 0..3 {
            let states = create_test_states("test-model");
            cache.put(format!("entry{i}"), states, None);
        }

        // Access one entry
        let _ = cache.get("entry0");

        let stats = cache.stats();
        assert_eq!(stats.entry_count, 3);
        assert!(stats.total_bytes > 0);
        assert_eq!(stats.total_access_count, 1);
    }

    #[test]
    fn test_cache_contains() {
        let mut cache = HiddenStateCache::with_defaults();
        let states = create_test_states("test-model");

        cache.put("test".to_string(), states, None);

        assert!(cache.contains("test"));
        assert!(!cache.contains("other"));
    }

    #[test]
    fn test_cache_keys() {
        let mut cache = HiddenStateCache::with_defaults();

        for i in 0..3 {
            let states = create_test_states("test-model");
            cache.put(format!("entry{i}"), states, None);
        }

        let keys = cache.keys();
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn test_cached_entry_metadata() {
        let states = create_test_states("test-model");
        let mut entry = CachedHiddenState::new("test".to_string(), states, None);

        assert_eq!(entry.access_count, 0);
        entry.record_access();
        assert_eq!(entry.access_count, 1);

        assert!(entry.age_secs() < 1);
        assert!(entry.estimated_size_bytes() > 0);
    }

    #[test]
    fn test_cache_stats_utilization() {
        let stats = HiddenStateCacheStats {
            entry_count: 50,
            total_bytes: 128 * 1024 * 1024,
            max_entries: 100,
            max_bytes: 256 * 1024 * 1024,
            total_access_count: 0,
            average_age_secs: 0.0,
        };

        assert!((stats.entry_utilization() - 50.0).abs() < 0.001);
        assert!((stats.memory_utilization() - 50.0).abs() < 0.001);
    }
}
