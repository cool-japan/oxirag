//! Core types for prefix caching.
//!
//! This module provides the fundamental data structures for managing
//! KV cache entries for context-aware prefix caching.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Unique identifier for cached KV entries.
pub type CacheKey = String;

/// Configuration for the prefix cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixCacheConfig {
    /// Maximum number of entries allowed in the cache.
    pub max_entries: usize,
    /// Maximum memory usage in bytes.
    pub max_memory_bytes: usize,
    /// Default time-to-live in seconds for cache entries.
    pub default_ttl_secs: u64,
    /// Whether to enable compression for cached data.
    pub enable_compression: bool,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            default_ttl_secs: 3600,              // 1 hour
            enable_compression: false,
        }
    }
}

impl PrefixCacheConfig {
    /// Create a new prefix cache configuration.
    #[must_use]
    pub fn new(max_entries: usize, max_memory_bytes: usize) -> Self {
        Self {
            max_entries,
            max_memory_bytes,
            ..Default::default()
        }
    }

    /// Set the default TTL in seconds.
    #[must_use]
    pub fn with_default_ttl(mut self, ttl_secs: u64) -> Self {
        self.default_ttl_secs = ttl_secs;
        self
    }

    /// Enable or disable compression.
    #[must_use]
    pub fn with_compression(mut self, enable: bool) -> Self {
        self.enable_compression = enable;
        self
    }
}

/// Statistics for cache performance monitoring.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted.
    pub evictions: u64,
    /// Total memory usage in bytes.
    pub total_bytes: usize,
    /// Number of entries currently in cache.
    pub entry_count: usize,
    /// Number of expired entries removed.
    pub expirations: u64,
}

impl CacheStats {
    /// Calculate the hit rate as a percentage.
    ///
    /// Note: For very large counters (> 2^52), precision may be lost
    /// when converting to f64.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Reset all statistics to zero.
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.expirations = 0;
    }

    /// Record a cache hit.
    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    /// Record a cache miss.
    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    /// Record an eviction.
    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }

    /// Record an expiration.
    pub fn record_expiration(&mut self) {
        self.expirations += 1;
    }

    /// Update memory usage.
    pub fn update_memory(&mut self, bytes: usize, count: usize) {
        self.total_bytes = bytes;
        self.entry_count = count;
    }
}

/// Fingerprint for identifying context content.
///
/// The fingerprint provides a way to uniquely identify cached context
/// based on its content hash and metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextFingerprint {
    /// Hash of the content for quick comparison.
    pub hash: u64,
    /// Length of the prefix in tokens or characters.
    pub prefix_length: usize,
    /// A short summary of the content for debugging.
    pub content_summary: String,
}

impl ContextFingerprint {
    /// Create a new context fingerprint.
    #[must_use]
    pub fn new(hash: u64, prefix_length: usize, content_summary: impl Into<String>) -> Self {
        Self {
            hash,
            prefix_length,
            content_summary: content_summary.into(),
        }
    }

    /// Check if this fingerprint represents a prefix of another.
    ///
    /// This is useful for partial cache hits where we can reuse
    /// the cached KV state for a common prefix.
    #[must_use]
    pub fn is_prefix_of(&self, other: &Self) -> bool {
        // A fingerprint is a prefix if:
        // 1. Its length is shorter or equal
        // 2. The hashes would match for the prefix portion
        // For simplicity, we compare length only here.
        // Real implementation would need rolling hash or prefix-aware hashing.
        self.prefix_length <= other.prefix_length
    }
}

/// A cached KV entry representing processed context.
///
/// This stores the pre-computed key-value pairs from transformer
/// attention layers, allowing fast reuse of "premise knowledge".
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Unique key for this cache entry.
    pub key: CacheKey,
    /// Fingerprint identifying the cached context.
    pub fingerprint: ContextFingerprint,
    /// The cached KV data (simplified as f32 vector).
    pub kv_data: Vec<f32>,
    /// Length of the cached sequence in tokens.
    pub sequence_length: usize,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Optional time-to-live for this entry.
    pub ttl: Option<Duration>,
}

impl KVCacheEntry {
    /// Create a new KV cache entry.
    #[must_use]
    pub fn new(
        key: impl Into<CacheKey>,
        fingerprint: ContextFingerprint,
        kv_data: Vec<f32>,
        sequence_length: usize,
    ) -> Self {
        let now = Instant::now();
        Self {
            key: key.into(),
            fingerprint,
            kv_data,
            sequence_length,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            ttl: None,
        }
    }

    /// Set the TTL for this entry.
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set the TTL from seconds.
    #[must_use]
    pub fn with_ttl_secs(mut self, secs: u64) -> Self {
        self.ttl = Some(Duration::from_secs(secs));
        self
    }

    /// Check if this entry has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() >= ttl
        } else {
            false
        }
    }

    /// Get the age of this entry.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last access.
    #[must_use]
    pub fn time_since_access(&self) -> Duration {
        self.last_accessed.elapsed()
    }

    /// Record an access to this entry.
    pub fn record_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Estimate the memory size of this entry in bytes.
    #[must_use]
    pub fn estimated_size(&self) -> usize {
        // Size of kv_data in bytes
        let kv_size = self.kv_data.len() * std::mem::size_of::<f32>();
        // Size of key string
        let key_size = self.key.len();
        // Size of content_summary
        let summary_size = self.fingerprint.content_summary.len();
        // Fixed overhead for struct fields
        let overhead = std::mem::size_of::<Self>();

        kv_size + key_size + summary_size + overhead
    }
}

/// Result of a cache lookup operation.
#[derive(Debug, Clone)]
pub enum CacheLookupResult {
    /// Exact match found in cache.
    Hit(KVCacheEntry),
    /// No match found.
    Miss,
    /// Partial match found - the cached entry is a prefix of the requested context.
    PartialHit {
        /// The cached entry that partially matches.
        entry: KVCacheEntry,
        /// Number of tokens/positions that need to be computed.
        remaining_length: usize,
    },
}

impl CacheLookupResult {
    /// Check if this is a hit (exact or partial).
    #[must_use]
    pub fn is_hit(&self) -> bool {
        matches!(self, Self::Hit(_) | Self::PartialHit { .. })
    }

    /// Check if this is a miss.
    #[must_use]
    pub fn is_miss(&self) -> bool {
        matches!(self, Self::Miss)
    }

    /// Get the cached entry if available.
    #[must_use]
    pub fn entry(&self) -> Option<&KVCacheEntry> {
        match self {
            Self::Hit(entry) | Self::PartialHit { entry, .. } => Some(entry),
            Self::Miss => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_cache_config_default() {
        let config = PrefixCacheConfig::default();
        assert_eq!(config.max_entries, 1000);
        assert_eq!(config.max_memory_bytes, 512 * 1024 * 1024);
        assert_eq!(config.default_ttl_secs, 3600);
        assert!(!config.enable_compression);
    }

    #[test]
    fn test_prefix_cache_config_builder() {
        let config = PrefixCacheConfig::new(500, 256 * 1024 * 1024)
            .with_default_ttl(1800)
            .with_compression(true);

        assert_eq!(config.max_entries, 500);
        assert_eq!(config.max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(config.default_ttl_secs, 1800);
        assert!(config.enable_compression);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let mut stats = CacheStats::default();
        assert!(stats.hit_rate().abs() < f64::EPSILON);

        stats.hits = 75;
        stats.misses = 25;
        assert!((stats.hit_rate() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_recording() {
        let mut stats = CacheStats::default();

        stats.record_hit();
        stats.record_hit();
        stats.record_miss();
        stats.record_eviction();
        stats.record_expiration();

        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.expirations, 1);
    }

    #[test]
    fn test_cache_stats_reset() {
        let mut stats = CacheStats {
            hits: 100,
            misses: 50,
            evictions: 10,
            total_bytes: 1000,
            entry_count: 5,
            expirations: 3,
        };

        stats.reset();

        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.expirations, 0);
        // total_bytes and entry_count are not reset
        assert_eq!(stats.total_bytes, 1000);
    }

    #[test]
    fn test_context_fingerprint_creation() {
        let fp = ContextFingerprint::new(12345, 100, "Test context...");

        assert_eq!(fp.hash, 12345);
        assert_eq!(fp.prefix_length, 100);
        assert_eq!(fp.content_summary, "Test context...");
    }

    #[test]
    fn test_context_fingerprint_is_prefix_of() {
        let short_fp = ContextFingerprint::new(100, 50, "Short");
        let long_fp = ContextFingerprint::new(200, 100, "Long");

        assert!(short_fp.is_prefix_of(&long_fp));
        assert!(!long_fp.is_prefix_of(&short_fp));
        assert!(short_fp.is_prefix_of(&short_fp)); // Same length
    }

    #[test]
    fn test_kv_cache_entry_creation() {
        let fp = ContextFingerprint::new(1, 10, "test");
        let entry = KVCacheEntry::new("key1", fp.clone(), vec![1.0, 2.0, 3.0], 10);

        assert_eq!(entry.key, "key1");
        assert_eq!(entry.fingerprint, fp);
        assert_eq!(entry.kv_data.len(), 3);
        assert_eq!(entry.sequence_length, 10);
        assert_eq!(entry.access_count, 0);
        assert!(entry.ttl.is_none());
    }

    #[test]
    fn test_kv_cache_entry_with_ttl() {
        let fp = ContextFingerprint::new(1, 10, "test");
        let entry = KVCacheEntry::new("key1", fp, vec![], 10).with_ttl_secs(60);

        assert!(entry.ttl.is_some());
        assert_eq!(entry.ttl.unwrap(), Duration::from_secs(60));
    }

    #[test]
    fn test_kv_cache_entry_expiration() {
        let fp = ContextFingerprint::new(1, 10, "test");
        let entry = KVCacheEntry::new("key1", fp.clone(), vec![], 10);

        // Entry without TTL never expires
        assert!(!entry.is_expired());

        // Entry with very long TTL is not expired
        let entry_long =
            KVCacheEntry::new("key2", fp.clone(), vec![], 10).with_ttl(Duration::from_secs(3600));
        assert!(!entry_long.is_expired());

        // Entry with zero TTL is immediately expired
        let entry_zero = KVCacheEntry::new("key3", fp, vec![], 10).with_ttl(Duration::from_secs(0));
        assert!(entry_zero.is_expired());
    }

    #[test]
    fn test_kv_cache_entry_record_access() {
        let fp = ContextFingerprint::new(1, 10, "test");
        let mut entry = KVCacheEntry::new("key1", fp, vec![], 10);

        assert_eq!(entry.access_count, 0);
        let initial_access = entry.last_accessed;

        // Small sleep to ensure time difference
        std::thread::sleep(std::time::Duration::from_millis(1));
        entry.record_access();

        assert_eq!(entry.access_count, 1);
        assert!(entry.last_accessed > initial_access);
    }

    #[test]
    fn test_kv_cache_entry_estimated_size() {
        let fp = ContextFingerprint::new(1, 10, "summary");
        let entry = KVCacheEntry::new("testkey", fp, vec![1.0; 100], 10);

        let size = entry.estimated_size();
        // At minimum: 100 f32 values = 400 bytes
        assert!(size >= 400);
    }

    #[test]
    fn test_cache_lookup_result_hit() {
        let fp = ContextFingerprint::new(1, 10, "test");
        let entry = KVCacheEntry::new("key1", fp, vec![], 10);
        let result = CacheLookupResult::Hit(entry.clone());

        assert!(result.is_hit());
        assert!(!result.is_miss());
        assert!(result.entry().is_some());
        assert_eq!(result.entry().unwrap().key, "key1");
    }

    #[test]
    fn test_cache_lookup_result_miss() {
        let result = CacheLookupResult::Miss;

        assert!(!result.is_hit());
        assert!(result.is_miss());
        assert!(result.entry().is_none());
    }

    #[test]
    fn test_cache_lookup_result_partial_hit() {
        let fp = ContextFingerprint::new(1, 10, "test");
        let entry = KVCacheEntry::new("key1", fp, vec![], 10);
        let result = CacheLookupResult::PartialHit {
            entry: entry.clone(),
            remaining_length: 5,
        };

        assert!(result.is_hit());
        assert!(!result.is_miss());
        assert!(result.entry().is_some());
    }
}
