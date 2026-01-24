//! In-memory prefix cache implementation.
//!
//! This module provides a thread-safe, LRU-based in-memory cache
//! for storing KV cache entries.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;

use super::traits::PrefixCacheStore;
use super::types::{
    CacheKey, CacheLookupResult, CacheStats, ContextFingerprint, KVCacheEntry, PrefixCacheConfig,
};
use crate::error::OxiRagError;

/// An in-memory implementation of `PrefixCacheStore` with LRU eviction.
///
/// This cache provides:
/// - LRU (Least Recently Used) eviction when capacity is reached
/// - TTL-based expiration
/// - Thread-safe access via `RwLock`
/// - Prefix matching for partial cache hits
/// - Memory tracking
///
/// # Example
///
/// ```rust,ignore
/// use oxirag::prefix_cache::{InMemoryPrefixCache, PrefixCacheConfig, PrefixCacheStore};
///
/// let config = PrefixCacheConfig::new(1000, 512 * 1024 * 1024);
/// let mut cache = InMemoryPrefixCache::new(config);
///
/// // Store and retrieve entries
/// cache.put(entry).await?;
/// let retrieved = cache.get(&fingerprint).await;
/// ```
#[derive(Debug)]
pub struct InMemoryPrefixCache {
    /// Configuration for this cache.
    config: PrefixCacheConfig,
    /// Internal storage protected by `RwLock`.
    pub(crate) inner: Arc<RwLock<CacheInner>>,
}

/// Internal cache state.
#[derive(Debug)]
pub(crate) struct CacheInner {
    /// Entries keyed by cache key.
    pub(crate) entries: HashMap<CacheKey, KVCacheEntry>,
    /// Mapping from fingerprint hash to cache key for fast lookup.
    pub(crate) fingerprint_index: HashMap<u64, CacheKey>,
    /// LRU ordering: Vec of cache keys.
    /// Kept sorted by access time (oldest first).
    lru_order: Vec<CacheKey>,
    /// Statistics.
    stats: CacheStats,
    /// Default TTL for new entries.
    default_ttl: Option<Duration>,
    /// Next key ID for generating unique keys.
    next_key_id: u64,
}

impl CacheInner {
    fn new(default_ttl: Option<Duration>) -> Self {
        Self {
            entries: HashMap::new(),
            fingerprint_index: HashMap::new(),
            lru_order: Vec::new(),
            stats: CacheStats::default(),
            default_ttl,
            next_key_id: 0,
        }
    }

    fn generate_key(&mut self) -> CacheKey {
        let key = format!("pc_{}", self.next_key_id);
        self.next_key_id += 1;
        key
    }

    fn update_lru(&mut self, key: &CacheKey) {
        // Remove from current position if exists
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
        // Add to end (most recently used)
        self.lru_order.push(key.clone());
    }

    fn remove_from_lru(&mut self, key: &CacheKey) {
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
    }

    fn get_lru_key(&self) -> Option<CacheKey> {
        self.lru_order.first().cloned()
    }

    fn calculate_memory(&self) -> usize {
        self.entries
            .values()
            .map(KVCacheEntry::estimated_size)
            .sum()
    }

    fn update_stats(&mut self) {
        self.stats.total_bytes = self.calculate_memory();
        self.stats.entry_count = self.entries.len();
    }
}

impl InMemoryPrefixCache {
    /// Create a new in-memory prefix cache with the given configuration.
    #[must_use]
    pub fn new(config: PrefixCacheConfig) -> Self {
        let default_ttl = if config.default_ttl_secs > 0 {
            Some(Duration::from_secs(config.default_ttl_secs))
        } else {
            None
        };

        Self {
            config,
            inner: Arc::new(RwLock::new(CacheInner::new(default_ttl))),
        }
    }

    /// Create a cache with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(PrefixCacheConfig::default())
    }

    /// Get the cache configuration.
    #[must_use]
    pub fn config(&self) -> &PrefixCacheConfig {
        &self.config
    }

    /// Perform a lookup that can return partial matches.
    ///
    /// This is useful for finding the longest cached prefix when
    /// exact matches are not available.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn lookup(&self, fingerprint: &ContextFingerprint) -> CacheLookupResult {
        let inner = self.inner.read().expect("lock poisoned");

        // First try exact match
        if let Some(key) = inner.fingerprint_index.get(&fingerprint.hash)
            && let Some(entry) = inner.entries.get(key)
            && !entry.is_expired()
            && entry.fingerprint.prefix_length == fingerprint.prefix_length
        {
            return CacheLookupResult::Hit(entry.clone());
        }

        // Try to find a prefix match
        let mut best_match: Option<&KVCacheEntry> = None;
        let mut best_length = 0;

        for entry in inner.entries.values() {
            if entry.is_expired() {
                continue;
            }
            if entry.fingerprint.is_prefix_of(fingerprint)
                && entry.fingerprint.prefix_length > best_length
            {
                best_match = Some(entry);
                best_length = entry.fingerprint.prefix_length;
            }
        }

        match best_match {
            Some(entry) if best_length == fingerprint.prefix_length => {
                CacheLookupResult::Hit(entry.clone())
            }
            Some(entry) => CacheLookupResult::PartialHit {
                entry: entry.clone(),
                remaining_length: fingerprint.prefix_length - best_length,
            },
            None => CacheLookupResult::Miss,
        }
    }

    /// Evict entries to make room for a new entry of the given size.
    fn evict_for_space(&self, needed_bytes: usize) -> Result<(), OxiRagError> {
        let mut inner = self.inner.write().expect("lock poisoned");

        while inner.calculate_memory() + needed_bytes > self.config.max_memory_bytes
            || inner.entries.len() >= self.config.max_entries
        {
            if let Some(lru_key) = inner.get_lru_key() {
                if let Some(entry) = inner.entries.remove(&lru_key) {
                    inner.fingerprint_index.remove(&entry.fingerprint.hash);
                    inner.remove_from_lru(&lru_key);
                    inner.stats.record_eviction();
                } else {
                    // Key in LRU but not in entries - clean up
                    inner.remove_from_lru(&lru_key);
                }
            } else {
                // No more entries to evict
                break;
            }
        }

        // Check if we have enough space now
        if inner.entries.len() >= self.config.max_entries {
            return Err(OxiRagError::Config(
                "Cache at maximum entry capacity".to_string(),
            ));
        }
        if inner.calculate_memory() + needed_bytes > self.config.max_memory_bytes {
            return Err(OxiRagError::Config(
                "Cache at maximum memory capacity".to_string(),
            ));
        }

        Ok(())
    }
}

impl Clone for InMemoryPrefixCache {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            inner: Arc::clone(&self.inner),
        }
    }
}

#[async_trait]
impl PrefixCacheStore for InMemoryPrefixCache {
    async fn get(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        let mut inner = self.inner.write().expect("lock poisoned");

        let Some(key) = inner.fingerprint_index.get(&fingerprint.hash).cloned() else {
            inner.stats.record_miss();
            return None;
        };

        // Check if entry exists and if it's expired
        let is_expired = inner
            .entries
            .get(&key)
            .is_some_and(KVCacheEntry::is_expired);

        if is_expired {
            // Remove expired entry
            if let Some(removed) = inner.entries.remove(&key) {
                inner.fingerprint_index.remove(&removed.fingerprint.hash);
            }
            inner.remove_from_lru(&key);
            inner.stats.record_expiration();
            inner.stats.record_miss();
            inner.update_stats();
            return None;
        }

        // Get, update access, and return clone
        if let Some(entry) = inner.entries.get_mut(&key) {
            entry.record_access();
            let result = entry.clone();
            inner.update_lru(&key);
            inner.stats.record_hit();
            return Some(result);
        }

        inner.stats.record_miss();
        None
    }

    async fn put(&mut self, entry: KVCacheEntry) -> Result<CacheKey, OxiRagError> {
        let entry_size = entry.estimated_size();
        self.evict_for_space(entry_size)?;

        let mut inner = self.inner.write().expect("lock poisoned");

        // Generate a key if the entry doesn't have one
        let key = if entry.key.is_empty() {
            inner.generate_key()
        } else {
            entry.key.clone()
        };

        // Apply default TTL if entry doesn't have one
        let mut entry = entry;
        if entry.ttl.is_none() {
            entry.ttl = inner.default_ttl;
        }
        entry.key.clone_from(&key);

        // Remove any existing entry with the same fingerprint
        if let Some(old_key) = inner
            .fingerprint_index
            .get(&entry.fingerprint.hash)
            .cloned()
        {
            inner.entries.remove(&old_key);
            inner.remove_from_lru(&old_key);
        }

        inner
            .fingerprint_index
            .insert(entry.fingerprint.hash, key.clone());
        inner.entries.insert(key.clone(), entry);
        inner.update_lru(&key);
        inner.update_stats();

        Ok(key)
    }

    async fn remove(&mut self, key: &CacheKey) -> Option<KVCacheEntry> {
        let mut inner = self.inner.write().expect("lock poisoned");

        if let Some(entry) = inner.entries.remove(key) {
            inner.fingerprint_index.remove(&entry.fingerprint.hash);
            inner.remove_from_lru(key);
            inner.update_stats();
            Some(entry)
        } else {
            None
        }
    }

    async fn contains(&self, fingerprint: &ContextFingerprint) -> bool {
        let inner = self.inner.read().expect("lock poisoned");

        if let Some(key) = inner.fingerprint_index.get(&fingerprint.hash)
            && let Some(entry) = inner.entries.get(key)
        {
            return !entry.is_expired();
        }
        false
    }

    async fn clear(&mut self) {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.entries.clear();
        inner.fingerprint_index.clear();
        inner.lru_order.clear();
        inner.update_stats();
    }

    fn stats(&self) -> CacheStats {
        let inner = self.inner.read().expect("lock poisoned");
        inner.stats.clone()
    }

    fn len(&self) -> usize {
        let inner = self.inner.read().expect("lock poisoned");
        inner.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    async fn find_prefix_match(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        let inner = self.inner.read().expect("lock poisoned");

        let mut best_match: Option<&KVCacheEntry> = None;
        let mut best_length = 0;

        for entry in inner.entries.values() {
            if entry.is_expired() {
                continue;
            }
            if entry.fingerprint.is_prefix_of(fingerprint)
                && entry.fingerprint.prefix_length > best_length
                && entry.fingerprint.prefix_length < fingerprint.prefix_length
            {
                best_match = Some(entry);
                best_length = entry.fingerprint.prefix_length;
            }
        }

        best_match.cloned()
    }

    async fn evict_expired(&mut self) -> usize {
        let mut inner = self.inner.write().expect("lock poisoned");
        let mut expired_keys = Vec::new();

        for (key, entry) in &inner.entries {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        let count = expired_keys.len();
        for key in expired_keys {
            if let Some(entry) = inner.entries.remove(&key) {
                inner.fingerprint_index.remove(&entry.fingerprint.hash);
                inner.remove_from_lru(&key);
                inner.stats.record_expiration();
            }
        }

        inner.update_stats();
        count
    }

    fn memory_usage(&self) -> usize {
        let inner = self.inner.read().expect("lock poisoned");
        inner.calculate_memory()
    }
}

#[cfg(test)]
#[allow(clippy::cast_sign_loss, clippy::float_cmp)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_entry(id: &str, hash: u64, kv_size: usize) -> KVCacheEntry {
        let fp = ContextFingerprint::new(hash, 100, format!("test {id}"));
        KVCacheEntry::new(id, fp, vec![0.0; kv_size], 100)
    }

    #[tokio::test]
    async fn test_cache_put_and_get() {
        let mut cache = InMemoryPrefixCache::with_defaults();
        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        let key = cache.put(entry).await.unwrap();
        assert!(!key.is_empty());

        let retrieved = cache.get(&fingerprint).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().fingerprint.hash, 12345);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = InMemoryPrefixCache::with_defaults();
        let fingerprint = ContextFingerprint::new(99999, 100, "nonexistent");

        let result = cache.get(&fingerprint).await;
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_cache_contains() {
        let mut cache = InMemoryPrefixCache::with_defaults();
        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        assert!(!cache.contains(&fingerprint).await);

        cache.put(entry).await.unwrap();

        assert!(cache.contains(&fingerprint).await);
    }

    #[tokio::test]
    async fn test_cache_remove() {
        let mut cache = InMemoryPrefixCache::with_defaults();
        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        let key = cache.put(entry).await.unwrap();
        assert_eq!(cache.len(), 1);

        let removed = cache.remove(&key).await;
        assert!(removed.is_some());
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains(&fingerprint).await);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let mut cache = InMemoryPrefixCache::with_defaults();

        for i in 0..5 {
            let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
            cache.put(entry).await.unwrap();
        }

        assert_eq!(cache.len(), 5);

        cache.clear().await;

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn test_cache_stats_tracking() {
        let mut cache = InMemoryPrefixCache::with_defaults();
        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        cache.put(entry).await.unwrap();

        // Hit
        cache.get(&fingerprint).await;
        cache.get(&fingerprint).await;

        // Miss
        let missing_fp = ContextFingerprint::new(99999, 100, "missing");
        cache.get(&missing_fp).await;

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 66.666_666_666_666_66).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let config = PrefixCacheConfig::new(3, 10 * 1024 * 1024); // Max 3 entries
        let mut cache = InMemoryPrefixCache::new(config);

        // Add 3 entries
        let entry1 = create_test_entry("test1", 1, 10);
        let entry2 = create_test_entry("test2", 2, 10);
        let entry3 = create_test_entry("test3", 3, 10);

        let fp1 = entry1.fingerprint.clone();
        let fp2 = entry2.fingerprint.clone();
        let fp3 = entry3.fingerprint.clone();

        cache.put(entry1).await.unwrap();
        cache.put(entry2).await.unwrap();
        cache.put(entry3).await.unwrap();

        assert_eq!(cache.len(), 3);

        // Access entry1 and entry3 to make entry2 the LRU
        cache.get(&fp1).await;
        cache.get(&fp3).await;

        // Add a 4th entry - should evict entry2 (LRU)
        let entry4 = create_test_entry("test4", 4, 10);
        cache.put(entry4).await.unwrap();

        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&fp2).await); // entry2 should be evicted
        assert!(cache.contains(&fp1).await);
        assert!(cache.contains(&fp3).await);

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn test_cache_ttl_expiration() {
        let mut cache = InMemoryPrefixCache::with_defaults();

        // Create entry with immediate TTL (0 seconds = expires immediately)
        let fp = ContextFingerprint::new(12345, 100, "test");
        let entry = KVCacheEntry::new("test1", fp.clone(), vec![0.0; 10], 100)
            .with_ttl(Duration::from_secs(0)); // Immediate expiration

        cache.put(entry).await.unwrap();

        // Entry should be expired immediately
        std::thread::sleep(Duration::from_millis(1));
        let result = cache.get(&fp).await;
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.expirations, 1);
    }

    #[tokio::test]
    async fn test_cache_evict_expired() {
        let mut cache = InMemoryPrefixCache::with_defaults();

        // Add entries with immediate expiration
        for i in 0..5 {
            let fp = ContextFingerprint::new(i as u64, 100, format!("test {i}"));
            let entry = KVCacheEntry::new(format!("test{i}"), fp, vec![0.0; 10], 100)
                .with_ttl(Duration::from_secs(0)); // Immediate expiration
            cache.put(entry).await.unwrap();
        }

        assert_eq!(cache.len(), 5);

        std::thread::sleep(Duration::from_millis(1));
        let evicted = cache.evict_expired().await;

        assert_eq!(evicted, 5);
        assert_eq!(cache.len(), 0);
    }

    #[tokio::test]
    async fn test_cache_memory_tracking() {
        let mut cache = InMemoryPrefixCache::with_defaults();

        let entry = create_test_entry("test1", 12345, 100);
        cache.put(entry).await.unwrap();

        let memory = cache.memory_usage();
        assert!(memory > 0);
        assert!(memory >= 100 * std::mem::size_of::<f32>());
    }

    #[tokio::test]
    async fn test_cache_memory_limit() {
        let config = PrefixCacheConfig::new(1000, 1000); // Very small memory limit
        let mut cache = InMemoryPrefixCache::new(config);

        // Try to add entries that exceed memory limit
        for i in 0..10 {
            let entry = create_test_entry(&format!("test{i}"), i as u64, 100);
            let _ = cache.put(entry).await;
        }

        // Should have evicted some entries
        assert!(cache.memory_usage() <= 1000 || cache.len() <= 2);
    }

    #[tokio::test]
    async fn test_cache_prefix_match() {
        let mut cache = InMemoryPrefixCache::with_defaults();

        // Add entry with prefix length 50
        let short_fp = ContextFingerprint::new(100, 50, "short");
        let short_entry = KVCacheEntry::new("short", short_fp.clone(), vec![1.0; 10], 50);
        cache.put(short_entry).await.unwrap();

        // Look for prefix of longer content
        let long_fp = ContextFingerprint::new(200, 100, "long");
        let prefix_match = cache.find_prefix_match(&long_fp).await;

        assert!(prefix_match.is_some());
        assert_eq!(prefix_match.unwrap().fingerprint.prefix_length, 50);
    }

    #[tokio::test]
    async fn test_cache_lookup_result_types() {
        let mut cache = InMemoryPrefixCache::with_defaults();

        // Add entry with prefix length 50
        let fp = ContextFingerprint::new(100, 50, "test");
        let entry = KVCacheEntry::new("test", fp.clone(), vec![1.0; 10], 50);
        cache.put(entry).await.unwrap();

        // Exact match
        let result = cache.lookup(&fp);
        assert!(matches!(result, CacheLookupResult::Hit(_)));

        // Miss - use a shorter prefix length so is_prefix_of won't match
        // (since 50 is not <= 30, no prefix match is found)
        let missing_fp = ContextFingerprint::new(999, 30, "missing");
        let result = cache.lookup(&missing_fp);
        assert!(matches!(result, CacheLookupResult::Miss));
    }

    #[tokio::test]
    async fn test_cache_update_existing() {
        let mut cache = InMemoryPrefixCache::with_defaults();

        let fp = ContextFingerprint::new(12345, 100, "test");
        let entry1 = KVCacheEntry::new("test1", fp.clone(), vec![1.0; 10], 100);
        let entry2 = KVCacheEntry::new("test2", fp.clone(), vec![2.0; 10], 100);

        cache.put(entry1).await.unwrap();
        cache.put(entry2).await.unwrap();

        // Should only have one entry (updated)
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get(&fp).await.unwrap();
        assert_eq!(retrieved.kv_data[0], 2.0);
    }

    #[tokio::test]
    async fn test_cache_clone_shares_state() {
        let mut cache1 = InMemoryPrefixCache::with_defaults();
        let cache2 = cache1.clone();

        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        cache1.put(entry).await.unwrap();

        // Both caches should see the entry
        assert!(cache2.contains(&fingerprint).await);
        assert_eq!(cache1.len(), cache2.len());
    }

    #[tokio::test]
    async fn test_cache_access_updates_lru() {
        let config = PrefixCacheConfig::new(2, 10 * 1024 * 1024);
        let mut cache = InMemoryPrefixCache::new(config);

        let entry1 = create_test_entry("test1", 1, 10);
        let entry2 = create_test_entry("test2", 2, 10);

        let fp1 = entry1.fingerprint.clone();
        let fp2 = entry2.fingerprint.clone();

        cache.put(entry1).await.unwrap();
        cache.put(entry2).await.unwrap();

        // Access entry1 to make it more recent than entry2
        cache.get(&fp1).await;

        // Add entry3 - should evict entry2 (LRU)
        let entry3 = create_test_entry("test3", 3, 10);
        cache.put(entry3).await.unwrap();

        assert!(cache.contains(&fp1).await);
        assert!(!cache.contains(&fp2).await);
    }
}
