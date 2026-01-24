//! Hierarchical cache with L1/L2/L3 tiers.
//!
//! This module implements a multi-tiered cache hierarchy where hot data
//! stays in fast L1 cache, warm data moves to L2, and cold data can be
//! stored in L3 (simulating disk-based storage).

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use super::store::InMemoryPrefixCache;
use super::traits::PrefixCacheStore;
use super::types::{CacheKey, CacheStats, ContextFingerprint, KVCacheEntry, PrefixCacheConfig};
use crate::error::OxiRagError;

/// Configuration for a single cache tier.
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Maximum number of entries in this tier.
    pub max_entries: usize,
    /// Maximum memory usage in bytes for this tier.
    pub max_bytes: usize,
    /// Optional time-to-live for entries in this tier.
    pub ttl_secs: Option<u64>,
}

impl TierConfig {
    /// Create a new tier configuration.
    #[must_use]
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries,
            max_bytes,
            ttl_secs: None,
        }
    }

    /// Set the TTL for this tier.
    #[must_use]
    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = Some(ttl_secs);
        self
    }

    /// Convert to `PrefixCacheConfig`.
    #[must_use]
    pub fn to_prefix_cache_config(&self) -> PrefixCacheConfig {
        let mut config = PrefixCacheConfig::new(self.max_entries, self.max_bytes);
        if let Some(ttl) = self.ttl_secs {
            config = config.with_default_ttl(ttl);
        }
        config
    }
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            max_bytes: 64 * 1024 * 1024, // 64 MB
            ttl_secs: None,
        }
    }
}

/// Configuration for the hierarchical cache.
#[derive(Debug, Clone)]
pub struct HierarchicalCacheConfig {
    /// L1 (hot) cache configuration.
    pub l1_config: TierConfig,
    /// L2 (warm) cache configuration.
    pub l2_config: TierConfig,
    /// L3 (cold) cache configuration. None disables L3.
    pub l3_config: Option<TierConfig>,
    /// Number of accesses needed to promote an entry to a higher tier.
    pub promotion_threshold: u32,
    /// Seconds since last access before an entry is demoted.
    pub demotion_threshold_secs: u64,
    /// Enable automatic background tier management.
    pub enable_auto_management: bool,
}

impl Default for HierarchicalCacheConfig {
    fn default() -> Self {
        Self {
            l1_config: TierConfig::new(100, 64 * 1024 * 1024).with_ttl(1800), // 64MB, 30min
            l2_config: TierConfig::new(500, 256 * 1024 * 1024).with_ttl(3600), // 256MB, 1hr
            l3_config: Some(TierConfig::new(1000, 512 * 1024 * 1024).with_ttl(7200)), // 512MB, 2hr
            promotion_threshold: 3,
            demotion_threshold_secs: 300, // 5 minutes
            enable_auto_management: true,
        }
    }
}

impl HierarchicalCacheConfig {
    /// Create a configuration with only L1 and L2 (no L3).
    #[must_use]
    pub fn two_tier(l1: TierConfig, l2: TierConfig) -> Self {
        Self {
            l1_config: l1,
            l2_config: l2,
            l3_config: None,
            ..Default::default()
        }
    }

    /// Create a full three-tier configuration.
    #[must_use]
    pub fn three_tier(l1: TierConfig, l2: TierConfig, l3: TierConfig) -> Self {
        Self {
            l1_config: l1,
            l2_config: l2,
            l3_config: Some(l3),
            ..Default::default()
        }
    }

    /// Set the promotion threshold.
    #[must_use]
    pub fn with_promotion_threshold(mut self, threshold: u32) -> Self {
        self.promotion_threshold = threshold;
        self
    }

    /// Set the demotion threshold in seconds.
    #[must_use]
    pub fn with_demotion_threshold_secs(mut self, secs: u64) -> Self {
        self.demotion_threshold_secs = secs;
        self
    }
}

/// Tier identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    /// L1 - Hot cache (fastest, smallest).
    L1,
    /// L2 - Warm cache (medium speed and size).
    L2,
    /// L3 - Cold cache (slowest, largest).
    L3,
}

impl CacheTier {
    /// Get the next lower tier.
    #[must_use]
    pub fn demote(&self) -> Option<Self> {
        match self {
            Self::L1 => Some(Self::L2),
            Self::L2 => Some(Self::L3),
            Self::L3 => None,
        }
    }

    /// Get the next higher tier.
    #[must_use]
    pub fn promote(&self) -> Option<Self> {
        match self {
            Self::L1 => None,
            Self::L2 => Some(Self::L1),
            Self::L3 => Some(Self::L2),
        }
    }
}

/// Statistics for the hierarchical cache.
#[derive(Debug, Clone, Default)]
pub struct HierarchicalCacheStats {
    /// Total hits across all tiers.
    pub total_hits: u64,
    /// Total misses.
    pub total_misses: u64,
    /// L1 hits.
    pub l1_hits: u64,
    /// L2 hits.
    pub l2_hits: u64,
    /// L3 hits.
    pub l3_hits: u64,
    /// Number of promotions.
    pub promotions: u64,
    /// Number of demotions.
    pub demotions: u64,
    /// L1 entry count.
    pub l1_entries: usize,
    /// L2 entry count.
    pub l2_entries: usize,
    /// L3 entry count.
    pub l3_entries: usize,
    /// Total memory usage in bytes.
    pub total_memory: usize,
}

impl HierarchicalCacheStats {
    /// Calculate the overall hit rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            (self.total_hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate the L1 hit rate (hits that came from L1).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            (self.l1_hits as f64 / total as f64) * 100.0
        }
    }

    /// Get total entry count across all tiers.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.l1_entries + self.l2_entries + self.l3_entries
    }
}

/// Internal state for the hierarchical cache.
#[derive(Debug)]
struct HierarchicalCacheInner {
    stats: HierarchicalCacheStats,
    last_rebalance: Instant,
}

/// Hierarchical cache with L1/L2/L3 tiers.
///
/// This cache implementation provides automatic tier management:
/// - Hot (frequently accessed) data stays in L1
/// - Warm data moves to L2
/// - Cold data can be stored in L3
///
/// Entries are automatically promoted on access and demoted based on
/// time since last access.
#[derive(Debug)]
pub struct HierarchicalCache {
    /// L1 (hot) cache - fast, small.
    l1_memory: InMemoryPrefixCache,
    /// L2 (warm) cache - medium speed and size.
    l2_spillover: InMemoryPrefixCache,
    /// L3 (cold) cache - slow, large (optional).
    l3_cold: Option<InMemoryPrefixCache>,
    /// Access count to promote an entry.
    promotion_threshold: u32,
    /// Time since last access to demote (in seconds).
    demotion_threshold_secs: u64,
    /// Internal state.
    inner: Arc<RwLock<HierarchicalCacheInner>>,
    /// Configuration.
    config: HierarchicalCacheConfig,
}

impl HierarchicalCache {
    /// Create a new hierarchical cache with the given configuration.
    #[must_use]
    pub fn new(config: HierarchicalCacheConfig) -> Self {
        let l1 = InMemoryPrefixCache::new(config.l1_config.to_prefix_cache_config());
        let l2 = InMemoryPrefixCache::new(config.l2_config.to_prefix_cache_config());
        let l3 = config
            .l3_config
            .as_ref()
            .map(|cfg| InMemoryPrefixCache::new(cfg.to_prefix_cache_config()));

        Self {
            l1_memory: l1,
            l2_spillover: l2,
            l3_cold: l3,
            promotion_threshold: config.promotion_threshold,
            demotion_threshold_secs: config.demotion_threshold_secs,
            inner: Arc::new(RwLock::new(HierarchicalCacheInner {
                stats: HierarchicalCacheStats::default(),
                last_rebalance: Instant::now(),
            })),
            config,
        }
    }

    /// Create a hierarchical cache with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(HierarchicalCacheConfig::default())
    }

    /// Get the cache configuration.
    #[must_use]
    pub fn config(&self) -> &HierarchicalCacheConfig {
        &self.config
    }

    /// Get hierarchical cache statistics.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn hierarchical_stats(&self) -> HierarchicalCacheStats {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.stats.l1_entries = self.l1_memory.len();
        inner.stats.l2_entries = self.l2_spillover.len();
        inner.stats.l3_entries = self.l3_cold.as_ref().map_or(0, InMemoryPrefixCache::len);
        inner.stats.total_memory = self.l1_memory.memory_usage()
            + self.l2_spillover.memory_usage()
            + self
                .l3_cold
                .as_ref()
                .map_or(0, InMemoryPrefixCache::memory_usage);
        inner.stats.clone()
    }

    /// Promote an entry to a higher tier.
    async fn promote_entry(
        &mut self,
        entry: KVCacheEntry,
        from_tier: CacheTier,
    ) -> Result<(), OxiRagError> {
        let target_tier = from_tier.promote();

        match target_tier {
            Some(CacheTier::L1) => {
                // Promote from L2 to L1
                let _ = self.l2_spillover.remove(&entry.key).await;
                self.l1_memory.put(entry).await?;
            }
            Some(CacheTier::L2) => {
                // Promote from L3 to L2
                if let Some(ref mut l3) = self.l3_cold {
                    let _ = l3.remove(&entry.key).await;
                }
                self.l2_spillover.put(entry).await?;
            }
            _ => {
                // Already at L1 or invalid tier
                return Ok(());
            }
        }

        let mut inner = self.inner.write().expect("lock poisoned");
        inner.stats.promotions += 1;

        Ok(())
    }

    /// Demote an entry to a lower tier.
    async fn demote_entry(
        &mut self,
        entry: KVCacheEntry,
        from_tier: CacheTier,
    ) -> Result<(), OxiRagError> {
        let target_tier = from_tier.demote();

        match target_tier {
            Some(CacheTier::L2) => {
                // Demote from L1 to L2
                let _ = self.l1_memory.remove(&entry.key).await;
                self.l2_spillover.put(entry).await?;
            }
            Some(CacheTier::L3) => {
                // Demote from L2 to L3
                let _ = self.l2_spillover.remove(&entry.key).await;
                if let Some(ref mut l3) = self.l3_cold {
                    l3.put(entry).await?;
                }
                // If no L3, entry is evicted
            }
            Some(CacheTier::L1) => {
                // This case shouldn't happen (can't demote to L1)
                // But we handle it for exhaustive match
            }
            None => {
                // L3 -> evicted
                if let Some(ref mut l3) = self.l3_cold {
                    let _ = l3.remove(&entry.key).await;
                }
            }
        }

        let mut inner = self.inner.write().expect("lock poisoned");
        inner.stats.demotions += 1;

        Ok(())
    }

    /// Find which tier an entry is in.
    ///
    /// # Panics
    ///
    /// Panics if any internal lock is poisoned.
    #[must_use]
    pub fn find_tier(&self, fingerprint: &ContextFingerprint) -> Option<CacheTier> {
        let inner_l1 = self.l1_memory.inner.read().expect("lock poisoned");
        if inner_l1.fingerprint_index.contains_key(&fingerprint.hash) {
            return Some(CacheTier::L1);
        }
        drop(inner_l1);

        let inner_l2 = self.l2_spillover.inner.read().expect("lock poisoned");
        if inner_l2.fingerprint_index.contains_key(&fingerprint.hash) {
            return Some(CacheTier::L2);
        }
        drop(inner_l2);

        if let Some(ref l3) = self.l3_cold {
            let inner_l3 = l3.inner.read().expect("lock poisoned");
            if inner_l3.fingerprint_index.contains_key(&fingerprint.hash) {
                return Some(CacheTier::L3);
            }
        }

        None
    }

    /// Rebalance tiers by promoting hot entries and demoting cold ones.
    ///
    /// Returns the number of entries moved.
    ///
    /// # Panics
    ///
    /// Panics if any internal lock is poisoned.
    pub async fn rebalance(&mut self) -> usize {
        let mut moved = 0;
        let demotion_threshold = Duration::from_secs(self.demotion_threshold_secs);

        // Collect entries to demote from L1
        let l1_entries_to_demote: Vec<KVCacheEntry> = {
            let inner = self.l1_memory.inner.read().expect("lock poisoned");
            inner
                .entries
                .values()
                .filter(|e| e.time_since_access() > demotion_threshold)
                .cloned()
                .collect()
        };

        for entry in l1_entries_to_demote {
            if self.demote_entry(entry, CacheTier::L1).await.is_ok() {
                moved += 1;
            }
        }

        // Collect entries to demote from L2
        let l2_entries_to_demote: Vec<KVCacheEntry> = {
            let inner = self.l2_spillover.inner.read().expect("lock poisoned");
            inner
                .entries
                .values()
                .filter(|e| e.time_since_access() > demotion_threshold * 2)
                .cloned()
                .collect()
        };

        for entry in l2_entries_to_demote {
            if self.demote_entry(entry, CacheTier::L2).await.is_ok() {
                moved += 1;
            }
        }

        // Collect entries to promote from L2 to L1
        let l2_entries_to_promote: Vec<KVCacheEntry> = {
            let inner = self.l2_spillover.inner.read().expect("lock poisoned");
            inner
                .entries
                .values()
                .filter(|e| e.access_count >= u64::from(self.promotion_threshold))
                .cloned()
                .collect()
        };

        for entry in l2_entries_to_promote {
            if self.promote_entry(entry, CacheTier::L2).await.is_ok() {
                moved += 1;
            }
        }

        // Update last rebalance time
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.last_rebalance = Instant::now();

        moved
    }

    /// Get the time since last rebalance.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn time_since_rebalance(&self) -> Duration {
        let inner = self.inner.read().expect("lock poisoned");
        inner.last_rebalance.elapsed()
    }

    /// Check if a rebalance is needed.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn needs_rebalance(&self) -> bool {
        self.time_since_rebalance() > Duration::from_secs(60)
    }
}

impl Clone for HierarchicalCache {
    fn clone(&self) -> Self {
        Self {
            l1_memory: self.l1_memory.clone(),
            l2_spillover: self.l2_spillover.clone(),
            l3_cold: self.l3_cold.clone(),
            promotion_threshold: self.promotion_threshold,
            demotion_threshold_secs: self.demotion_threshold_secs,
            inner: Arc::clone(&self.inner),
            config: self.config.clone(),
        }
    }
}

#[async_trait]
impl PrefixCacheStore for HierarchicalCache {
    async fn get(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        // Check L1 first
        if let Some(entry) = self.l1_memory.get(fingerprint).await {
            let mut inner = self.inner.write().expect("lock poisoned");
            inner.stats.total_hits += 1;
            inner.stats.l1_hits += 1;
            return Some(entry);
        }

        // Check L2
        if let Some(entry) = self.l2_spillover.get(fingerprint).await {
            let mut inner = self.inner.write().expect("lock poisoned");
            inner.stats.total_hits += 1;
            inner.stats.l2_hits += 1;
            // Note: Promotion handled separately to avoid borrow issues
            return Some(entry);
        }

        // Check L3
        if let Some(ref l3) = self.l3_cold
            && let Some(entry) = l3.get(fingerprint).await
        {
            let mut inner = self.inner.write().expect("lock poisoned");
            inner.stats.total_hits += 1;
            inner.stats.l3_hits += 1;
            return Some(entry);
        }

        // Miss
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.stats.total_misses += 1;
        None
    }

    async fn put(&mut self, entry: KVCacheEntry) -> Result<CacheKey, OxiRagError> {
        // New entries always go to L1
        self.l1_memory.put(entry).await
    }

    async fn remove(&mut self, key: &CacheKey) -> Option<KVCacheEntry> {
        // Try to remove from all tiers
        if let Some(entry) = self.l1_memory.remove(key).await {
            return Some(entry);
        }
        if let Some(entry) = self.l2_spillover.remove(key).await {
            return Some(entry);
        }
        if let Some(ref mut l3) = self.l3_cold
            && let Some(entry) = l3.remove(key).await
        {
            return Some(entry);
        }
        None
    }

    async fn contains(&self, fingerprint: &ContextFingerprint) -> bool {
        self.l1_memory.contains(fingerprint).await
            || self.l2_spillover.contains(fingerprint).await
            || self.l3_cold.as_ref().is_some_and(|l3| {
                let inner = l3.inner.read().expect("lock poisoned");
                inner.fingerprint_index.contains_key(&fingerprint.hash)
            })
    }

    async fn clear(&mut self) {
        self.l1_memory.clear().await;
        self.l2_spillover.clear().await;
        if let Some(ref mut l3) = self.l3_cold {
            l3.clear().await;
        }

        let mut inner = self.inner.write().expect("lock poisoned");
        inner.stats = HierarchicalCacheStats::default();
    }

    fn stats(&self) -> CacheStats {
        // Combine stats from all tiers
        let l1_stats = self.l1_memory.stats();
        let l2_stats = self.l2_spillover.stats();
        let l3_stats = self
            .l3_cold
            .as_ref()
            .map_or_else(CacheStats::default, InMemoryPrefixCache::stats);

        CacheStats {
            hits: l1_stats.hits + l2_stats.hits + l3_stats.hits,
            misses: l1_stats.misses, // Only count misses from top level lookup
            evictions: l1_stats.evictions + l2_stats.evictions + l3_stats.evictions,
            total_bytes: l1_stats.total_bytes + l2_stats.total_bytes + l3_stats.total_bytes,
            entry_count: l1_stats.entry_count + l2_stats.entry_count + l3_stats.entry_count,
            expirations: l1_stats.expirations + l2_stats.expirations + l3_stats.expirations,
        }
    }

    fn len(&self) -> usize {
        self.l1_memory.len()
            + self.l2_spillover.len()
            + self.l3_cold.as_ref().map_or(0, InMemoryPrefixCache::len)
    }

    fn is_empty(&self) -> bool {
        self.l1_memory.is_empty()
            && self.l2_spillover.is_empty()
            && self
                .l3_cold
                .as_ref()
                .is_none_or(InMemoryPrefixCache::is_empty)
    }

    async fn find_prefix_match(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        // Check all tiers for prefix match
        if let Some(entry) = self.l1_memory.find_prefix_match(fingerprint).await {
            return Some(entry);
        }
        if let Some(entry) = self.l2_spillover.find_prefix_match(fingerprint).await {
            return Some(entry);
        }
        if let Some(ref l3) = self.l3_cold
            && let Some(entry) = l3.find_prefix_match(fingerprint).await
        {
            return Some(entry);
        }
        None
    }

    async fn evict_expired(&mut self) -> usize {
        let mut count = 0;
        count += self.l1_memory.evict_expired().await;
        count += self.l2_spillover.evict_expired().await;
        if let Some(ref mut l3) = self.l3_cold {
            count += l3.evict_expired().await;
        }
        count
    }

    fn memory_usage(&self) -> usize {
        self.l1_memory.memory_usage()
            + self.l2_spillover.memory_usage()
            + self
                .l3_cold
                .as_ref()
                .map_or(0, InMemoryPrefixCache::memory_usage)
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::field_reassign_with_default
)]
mod tests {
    use super::*;

    fn create_test_entry(id: &str, hash: u64) -> KVCacheEntry {
        let fp = ContextFingerprint::new(hash, 100, format!("test {id}"));
        KVCacheEntry::new(id, fp, vec![0.0; 10], 100)
    }

    #[test]
    fn test_tier_config_default() {
        let config = TierConfig::default();
        assert_eq!(config.max_entries, 100);
        assert_eq!(config.max_bytes, 64 * 1024 * 1024);
        assert!(config.ttl_secs.is_none());
    }

    #[test]
    fn test_tier_config_with_ttl() {
        let config = TierConfig::new(50, 32 * 1024 * 1024).with_ttl(300);
        assert_eq!(config.max_entries, 50);
        assert_eq!(config.ttl_secs, Some(300));
    }

    #[test]
    fn test_tier_config_to_prefix_cache_config() {
        let tier = TierConfig::new(100, 64 * 1024 * 1024).with_ttl(600);
        let config = tier.to_prefix_cache_config();

        assert_eq!(config.max_entries, 100);
        assert_eq!(config.max_memory_bytes, 64 * 1024 * 1024);
        assert_eq!(config.default_ttl_secs, 600);
    }

    #[test]
    fn test_hierarchical_cache_config_default() {
        let config = HierarchicalCacheConfig::default();
        assert_eq!(config.promotion_threshold, 3);
        assert_eq!(config.demotion_threshold_secs, 300);
        assert!(config.l3_config.is_some());
    }

    #[test]
    fn test_hierarchical_cache_config_two_tier() {
        let config = HierarchicalCacheConfig::two_tier(
            TierConfig::new(50, 32 * 1024 * 1024),
            TierConfig::new(200, 128 * 1024 * 1024),
        );
        assert!(config.l3_config.is_none());
    }

    #[test]
    fn test_cache_tier_promote_demote() {
        assert_eq!(CacheTier::L3.promote(), Some(CacheTier::L2));
        assert_eq!(CacheTier::L2.promote(), Some(CacheTier::L1));
        assert_eq!(CacheTier::L1.promote(), None);

        assert_eq!(CacheTier::L1.demote(), Some(CacheTier::L2));
        assert_eq!(CacheTier::L2.demote(), Some(CacheTier::L3));
        assert_eq!(CacheTier::L3.demote(), None);
    }

    #[test]
    fn test_hierarchical_cache_stats_hit_rate() {
        let mut stats = HierarchicalCacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.total_hits = 80;
        stats.total_misses = 20;
        assert!((stats.hit_rate() - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_hierarchical_cache_stats_l1_hit_rate() {
        let mut stats = HierarchicalCacheStats::default();
        stats.total_hits = 100;
        stats.total_misses = 0;
        stats.l1_hits = 60;
        stats.l2_hits = 30;
        stats.l3_hits = 10;

        assert!((stats.l1_hit_rate() - 60.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_hierarchical_cache_new() {
        let cache = HierarchicalCache::with_defaults();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn test_hierarchical_cache_put_get() {
        let mut cache = HierarchicalCache::with_defaults();
        let entry = create_test_entry("test1", 12345);
        let fingerprint = entry.fingerprint.clone();

        let key = cache.put(entry).await.unwrap();
        assert!(!key.is_empty());

        let retrieved = cache.get(&fingerprint).await;
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_hierarchical_cache_contains() {
        let mut cache = HierarchicalCache::with_defaults();
        let entry = create_test_entry("test1", 12345);
        let fingerprint = entry.fingerprint.clone();

        assert!(!cache.contains(&fingerprint).await);
        cache.put(entry).await.unwrap();
        assert!(cache.contains(&fingerprint).await);
    }

    #[tokio::test]
    async fn test_hierarchical_cache_remove() {
        let mut cache = HierarchicalCache::with_defaults();
        let entry = create_test_entry("test1", 12345);
        let fingerprint = entry.fingerprint.clone();

        let key = cache.put(entry).await.unwrap();
        assert!(cache.contains(&fingerprint).await);

        let removed = cache.remove(&key).await;
        assert!(removed.is_some());
        assert!(!cache.contains(&fingerprint).await);
    }

    #[tokio::test]
    async fn test_hierarchical_cache_clear() {
        let mut cache = HierarchicalCache::with_defaults();

        for i in 0..5 {
            let entry = create_test_entry(&format!("test{i}"), i as u64);
            cache.put(entry).await.unwrap();
        }

        assert_eq!(cache.len(), 5);

        cache.clear().await;
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn test_hierarchical_cache_stats() {
        let mut cache = HierarchicalCache::with_defaults();
        let entry = create_test_entry("test1", 12345);
        let fingerprint = entry.fingerprint.clone();

        cache.put(entry).await.unwrap();

        // Hit
        cache.get(&fingerprint).await;

        // Miss
        let missing_fp = ContextFingerprint::new(99999, 100, "missing");
        cache.get(&missing_fp).await;

        let h_stats = cache.hierarchical_stats();
        assert_eq!(h_stats.total_hits, 1);
        assert_eq!(h_stats.total_misses, 1);
        assert_eq!(h_stats.l1_hits, 1);
    }

    #[tokio::test]
    async fn test_hierarchical_cache_find_tier() {
        let mut cache = HierarchicalCache::with_defaults();
        let entry = create_test_entry("test1", 12345);
        let fingerprint = entry.fingerprint.clone();

        cache.put(entry).await.unwrap();

        let tier = cache.find_tier(&fingerprint);
        assert_eq!(tier, Some(CacheTier::L1));

        let missing_fp = ContextFingerprint::new(99999, 100, "missing");
        let tier = cache.find_tier(&missing_fp);
        assert_eq!(tier, None);
    }

    #[tokio::test]
    async fn test_hierarchical_cache_memory_usage() {
        let mut cache = HierarchicalCache::with_defaults();

        let entry = create_test_entry("test1", 12345);
        cache.put(entry).await.unwrap();

        let memory = cache.memory_usage();
        assert!(memory > 0);
    }

    #[tokio::test]
    async fn test_hierarchical_cache_evict_expired() {
        let mut cache = HierarchicalCache::with_defaults();

        // Add entries with immediate expiration (TTL = 0)
        for i in 0..3 {
            let fp = ContextFingerprint::new(i as u64, 100, format!("test{i}"));
            let entry = KVCacheEntry::new(format!("test{i}"), fp, vec![0.0; 10], 100)
                .with_ttl(std::time::Duration::from_secs(0)); // Immediate expiry
            cache.put(entry).await.unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(1));
        let evicted = cache.evict_expired().await;
        assert_eq!(evicted, 3);
    }

    #[tokio::test]
    async fn test_hierarchical_cache_clone_shares_state() {
        let mut cache1 = HierarchicalCache::with_defaults();
        let cache2 = cache1.clone();

        let entry = create_test_entry("test1", 12345);
        let fingerprint = entry.fingerprint.clone();

        cache1.put(entry).await.unwrap();

        assert!(cache2.contains(&fingerprint).await);
        assert_eq!(cache1.len(), cache2.len());
    }

    #[tokio::test]
    async fn test_hierarchical_cache_needs_rebalance() {
        let cache = HierarchicalCache::with_defaults();
        // Just created, shouldn't need rebalance
        assert!(!cache.needs_rebalance());
    }

    #[tokio::test]
    async fn test_hierarchical_cache_two_tier() {
        let config = HierarchicalCacheConfig::two_tier(
            TierConfig::new(10, 1024 * 1024),
            TierConfig::new(50, 5 * 1024 * 1024),
        );
        let mut cache = HierarchicalCache::new(config);

        let entry = create_test_entry("test1", 12345);
        let fingerprint = entry.fingerprint.clone();

        cache.put(entry).await.unwrap();
        assert!(cache.get(&fingerprint).await.is_some());
    }

    #[tokio::test]
    async fn test_hierarchical_cache_prefix_match() {
        let mut cache = HierarchicalCache::with_defaults();

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
}
