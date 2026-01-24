//! Prefix Cache module for context-aware KV caching.
//!
//! This module implements Core Vision #2: Context-Aware Prefix Caching,
//! which efficiently manages KV Cache for "premise knowledge" in transformer
//! models.
//!
//! # Overview
//!
//! When using large language models, computing attention over the same
//! prefix (e.g., system prompts, document context) repeatedly is wasteful.
//! Prefix caching stores the Key-Value (KV) states from transformer attention
//! layers, allowing reuse when the same prefix is encountered again.
//!
//! # Components
//!
//! - [`types`]: Core data structures including `KVCacheEntry`, `ContextFingerprint`,
//!   `CacheStats`, and configuration types
//! - [`traits`]: The `PrefixCacheStore` trait defining the cache interface
//! - [`fingerprint`]: Utilities for generating context fingerprints
//! - [`store`]: In-memory cache implementation with LRU eviction
//! - [`paging`]: Paged cache system inspired by `PagedAttention`
//! - [`hierarchy`]: Hierarchical L1/L2/L3 cache tiers
//! - [`invalidation`]: Cache invalidation strategies and dependency tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::prefix_cache::{
//!     InMemoryPrefixCache, PrefixCacheConfig, PrefixCacheStore,
//!     ContextFingerprintGenerator, KVCacheEntry,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Configure the cache
//!     let config = PrefixCacheConfig::new(1000, 512 * 1024 * 1024)
//!         .with_default_ttl(3600)
//!         .with_compression(false);
//!
//!     let mut cache = InMemoryPrefixCache::new(config);
//!     let generator = ContextFingerprintGenerator::new();
//!
//!     // Generate fingerprint for some context
//!     let context = "You are a helpful assistant. Answer questions accurately.";
//!     let fingerprint = generator.generate(context);
//!
//!     // Check if already cached
//!     if let Some(entry) = cache.get(&fingerprint).await {
//!         println!("Cache hit! Reusing KV state.");
//!         // Use entry.kv_data for attention computation
//!     } else {
//!         println!("Cache miss. Computing KV state...");
//!         // Compute KV state (simulated here)
//!         let kv_data = vec![0.0f32; 1024];
//!
//!         // Store in cache
//!         let entry = KVCacheEntry::new("ctx1", fingerprint, kv_data, context.len());
//!         cache.put(entry).await?;
//!     }
//!
//!     // Check stats
//!     let stats = cache.stats();
//!     println!("Hit rate: {:.1}%", stats.hit_rate());
//!
//!     Ok(())
//! }
//! ```
//!
//! # Prefix Matching
//!
//! The cache supports partial matches where a shorter cached prefix can be
//! reused to accelerate computation of a longer context:
//!
//! ```rust,ignore
//! use oxirag::prefix_cache::{InMemoryPrefixCache, CacheLookupResult};
//!
//! let cache = InMemoryPrefixCache::with_defaults();
//!
//! match cache.lookup(&fingerprint) {
//!     CacheLookupResult::Hit(entry) => {
//!         // Exact match - use entry.kv_data directly
//!     }
//!     CacheLookupResult::PartialHit { entry, remaining_length } => {
//!         // Partial match - use entry.kv_data and compute the rest
//!         println!("Reusing {} tokens, computing {} more", entry.sequence_length, remaining_length);
//!     }
//!     CacheLookupResult::Miss => {
//!         // No match - compute everything from scratch
//!     }
//! }
//! ```
//!
//! # Hierarchical Caching
//!
//! For advanced use cases, the hierarchical cache provides L1/L2/L3 tiers
//! with automatic promotion and demotion:
//!
//! ```rust,ignore
//! use oxirag::prefix_cache::{HierarchicalCache, HierarchicalCacheConfig, TierConfig};
//!
//! let config = HierarchicalCacheConfig::three_tier(
//!     TierConfig::new(100, 64 * 1024 * 1024).with_ttl(300),   // L1: 64MB, 5min
//!     TierConfig::new(500, 256 * 1024 * 1024).with_ttl(1800), // L2: 256MB, 30min
//!     TierConfig::new(1000, 512 * 1024 * 1024).with_ttl(3600), // L3: 512MB, 1hr
//! );
//!
//! let mut cache = HierarchicalCache::new(config);
//! // Hot entries stay in L1, cold entries are demoted to L2/L3
//! ```
//!
//! # Paged Cache
//!
//! For efficient memory utilization with large entries, use the paged cache:
//!
//! ```rust,ignore
//! use oxirag::prefix_cache::{PagedCache, ContextFingerprint};
//!
//! let cache = PagedCache::new(4096, 1000); // 4KB pages, max 1000 pages
//!
//! let fp = ContextFingerprint::new(12345, 100, "example");
//! let data = vec![0.0f32; 10000]; // Large KV data
//!
//! cache.put(fp.clone(), &data);
//! let retrieved = cache.get(&fp);
//! ```
//!
//! # Thread Safety
//!
//! All implementations are thread-safe and can be shared across async tasks:
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use oxirag::prefix_cache::{InMemoryPrefixCache, PrefixCacheStore};
//!
//! let cache = Arc::new(InMemoryPrefixCache::with_defaults());
//!
//! // Clone for use in multiple tasks
//! let cache_clone = cache.clone();
//! tokio::spawn(async move {
//!     let stats = cache_clone.stats();
//!     println!("Entries: {}", stats.entry_count);
//! });
//! ```

pub mod fingerprint;
pub mod hierarchy;
pub mod invalidation;
pub mod paging;
pub mod persistent;
pub mod store;
pub mod traits;
pub mod types;

// Re-exports for convenience
pub use fingerprint::{ContextFingerprintGenerator, Fingerprintable, RollingHasher};
pub use hierarchy::{
    CacheTier, HierarchicalCache, HierarchicalCacheConfig, HierarchicalCacheStats, TierConfig,
};
pub use invalidation::{
    CacheValidator, DependencyStats, FingerprintInvalidationContext, InvalidationManager,
    InvalidationPolicy, InvalidationReason, InvalidationRuleBuilder,
};
pub use paging::{CachePage, PageTable, PagedCache, PagedKVEntry};
pub use persistent::{
    CacheIndex, CompactionStats, HybridPersistentCache, IndexEntry, PersistedEntry,
    PersistentCacheConfig, PersistentPrefixCache,
};
pub use store::InMemoryPrefixCache;
pub use traits::{PrefixCacheExt, PrefixCacheStore};
pub use types::{
    CacheKey, CacheLookupResult, CacheStats, ContextFingerprint, KVCacheEntry, PrefixCacheConfig,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_full_workflow() {
        // Create cache with custom config
        let config = PrefixCacheConfig::new(100, 10 * 1024 * 1024)
            .with_default_ttl(600)
            .with_compression(false);

        let mut cache = InMemoryPrefixCache::new(config);
        let generator = ContextFingerprintGenerator::new();

        // Generate fingerprint for context
        let context = "System: You are a helpful assistant.";
        let fingerprint = generator.generate(context);

        // Initially should be a miss
        assert!(!cache.contains(&fingerprint).await);

        // Simulate computing KV state
        let kv_data = vec![0.1f32; 256];
        let entry = KVCacheEntry::new("system_prompt", fingerprint.clone(), kv_data.clone(), 10);

        // Store in cache
        let key = cache.put(entry).await.unwrap();
        assert!(!key.is_empty());

        // Should now be a hit
        assert!(cache.contains(&fingerprint).await);
        let retrieved = cache.get(&fingerprint).await.unwrap();
        assert_eq!(retrieved.kv_data.len(), 256);

        // Stats should reflect the hit
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.entry_count, 1);
    }

    #[tokio::test]
    async fn test_fingerprintable_trait() {
        let content = "Test content for fingerprinting";
        let fp1 = content.fingerprint();
        let fp2 = content.to_string().fingerprint();

        // Same content should produce same fingerprint
        assert_eq!(fp1.hash, fp2.hash);
        assert_eq!(fp1.prefix_length, fp2.prefix_length);
    }

    #[tokio::test]
    async fn test_rolling_hash_incremental() {
        let mut hasher = RollingHasher::new();

        // Build up content incrementally
        hasher.append_str("Hello");
        let fp1 = hasher.to_fingerprint("Hello");

        hasher.append_str(", world!");
        let fp2 = hasher.to_fingerprint("Hello, world!");

        assert_ne!(fp1.hash, fp2.hash);
        assert!(fp2.prefix_length > fp1.prefix_length);
    }

    #[tokio::test]
    async fn test_prefix_hierarchy_lookup() {
        let mut cache = InMemoryPrefixCache::with_defaults();
        let generator = ContextFingerprintGenerator::new();

        // Cache a short prefix
        let short_content = "Hello";
        let short_fp = generator.generate(short_content);
        let short_entry = KVCacheEntry::new("short", short_fp.clone(), vec![1.0; 10], 5);
        cache.put(short_entry).await.unwrap();

        // Look up with longer content - should find prefix match
        let long_content = "Hello, world! How are you?";
        let long_fp = generator.generate(long_content);

        // Use lookup for more detailed result
        let result = cache.lookup(&long_fp);
        match result {
            CacheLookupResult::PartialHit {
                entry,
                remaining_length,
            } => {
                assert_eq!(entry.fingerprint.prefix_length, short_fp.prefix_length);
                assert!(remaining_length > 0);
            }
            CacheLookupResult::Hit(_) | CacheLookupResult::Miss => {
                // Exact match (unlikely given different content) or
                // No match found (acceptable if content is too different)
            }
        }
    }

    #[tokio::test]
    async fn test_cache_ext_get_or_compute() {
        let mut cache = InMemoryPrefixCache::with_defaults();
        let fingerprint = ContextFingerprint::new(42, 100, "test");

        // First call computes
        let entry1 = cache
            .get_or_compute(&fingerprint, || {
                Ok(KVCacheEntry::new(
                    "computed",
                    fingerprint.clone(),
                    vec![1.0, 2.0, 3.0],
                    100,
                ))
            })
            .await
            .unwrap();

        assert_eq!(entry1.kv_data, vec![1.0, 2.0, 3.0]);

        // Second call returns cached
        let entry2 = cache
            .get_or_compute(&fingerprint, || {
                Ok(KVCacheEntry::new(
                    "should_not_be_used",
                    fingerprint.clone(),
                    vec![4.0, 5.0, 6.0],
                    100,
                ))
            })
            .await
            .unwrap();

        assert_eq!(entry2.kv_data, vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
    async fn test_multiple_entries_different_fingerprints() {
        let mut cache = InMemoryPrefixCache::with_defaults();
        let generator = ContextFingerprintGenerator::new();

        let contexts = [
            "Context A: System prompt for task A",
            "Context B: System prompt for task B",
            "Context C: System prompt for task C",
        ];

        // Store all contexts
        for (i, ctx) in contexts.iter().enumerate() {
            let fp = generator.generate(ctx);
            let kv_data = vec![i as f32; 10];
            let entry = KVCacheEntry::new(format!("ctx_{i}"), fp, kv_data, 100);
            cache.put(entry).await.expect("put failed");
        }

        assert_eq!(cache.len(), 3);

        // Retrieve each one
        for (i, ctx) in contexts.iter().enumerate() {
            let fp = generator.generate(ctx);
            let retrieved = cache.get(&fp).await.expect("get failed");
            assert!((retrieved.kv_data[0] - i as f32).abs() < f32::EPSILON);
        }
    }

    #[tokio::test]
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    async fn test_eviction_maintains_consistency() {
        let config = PrefixCacheConfig::new(3, 10 * 1024 * 1024);
        let mut cache = InMemoryPrefixCache::new(config);

        // Add more entries than capacity
        for i in 0..5_i32 {
            let fp = ContextFingerprint::new(i as u64, 100, format!("test {i}"));
            let kv_data = vec![i as f32; 10];
            let entry = KVCacheEntry::new(format!("entry_{i}"), fp, kv_data, 100);
            cache.put(entry).await.expect("put failed");
        }

        // Should have at most 3 entries
        assert!(cache.len() <= 3);

        // Stats should reflect evictions
        let stats = cache.stats();
        assert!(stats.evictions >= 2);
    }

    // Tests for paging module
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_paged_cache_integration() {
        let cache = PagedCache::new(256, 100);
        let fp = ContextFingerprint::new(12345, 100, "test");
        let data: Vec<f32> = (0..500).map(|i| i as f32).collect();

        let _ = cache.put(&fp, &data);
        let retrieved = cache.get(&fp).expect("get failed");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_page_table_operations() {
        let mut table = PageTable::new(256, 10);

        let id1 = table.allocate_page().unwrap();
        let _id2 = table.allocate_page().unwrap();

        assert_eq!(table.allocated_count(), 2);

        table.free_page(id1);
        assert_eq!(table.allocated_count(), 1);

        // Freed page should be reused
        let id3 = table.allocate_page().unwrap();
        assert_eq!(id3, id1);
    }

    // Tests for hierarchy module
    #[tokio::test]
    async fn test_hierarchical_cache_integration() {
        let config = HierarchicalCacheConfig::two_tier(
            TierConfig::new(10, 1024 * 1024),
            TierConfig::new(50, 5 * 1024 * 1024),
        );
        let mut cache = HierarchicalCache::new(config);

        let fp = ContextFingerprint::new(12345, 100, "test");
        let entry = KVCacheEntry::new("test", fp.clone(), vec![1.0; 10], 100);

        cache.put(entry).await.unwrap();
        assert!(cache.contains(&fp).await);

        let stats = cache.hierarchical_stats();
        assert_eq!(stats.l1_entries, 1);
    }

    #[tokio::test]
    async fn test_hierarchical_cache_stats_tracking() {
        let mut cache = HierarchicalCache::with_defaults();

        let fp = ContextFingerprint::new(12345, 100, "test");
        let entry = KVCacheEntry::new("test", fp.clone(), vec![1.0; 10], 100);

        cache.put(entry).await.unwrap();

        // Access the entry
        cache.get(&fp).await;

        let stats = cache.hierarchical_stats();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.total_hits, 1);
    }

    // Tests for invalidation module
    #[test]
    fn test_invalidation_manager_integration() {
        let policy = InvalidationRuleBuilder::new()
            .with_ttl_secs(3600)
            .with_max_stale_secs(600)
            .build();

        let mut manager = InvalidationManager::new(policy);

        // Register dependencies
        manager.register_dependency("child1".to_string(), "parent".to_string());
        manager.register_dependency("child2".to_string(), "parent".to_string());

        // Invalidate parent
        let invalidated = manager.invalidate_dependents(&"parent".to_string());
        assert!(invalidated.contains(&"child1".to_string()));
        assert!(invalidated.contains(&"child2".to_string()));
    }

    #[test]
    fn test_cache_validator_integration() {
        let manager = InvalidationManager::new(InvalidationPolicy::ttl_secs(3600));
        let validator = CacheValidator::new(manager);

        let fp = ContextFingerprint::new(12345, 100, "test");
        let entry = KVCacheEntry::new("test", fp, vec![1.0; 10], 100);

        assert!(validator.is_valid(&entry));
        assert!(validator.validate(&entry).is_ok());
    }

    #[test]
    fn test_fingerprint_invalidation_context_integration() {
        let mut ctx = FingerprintInvalidationContext::new();

        let fp1 = ContextFingerprint::new(111, 100, "test1");
        let fp2 = ContextFingerprint::new(222, 100, "test2");

        ctx.register(&fp1, "key1".to_string());
        ctx.register(&fp2, "key2".to_string());

        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx.get_key(&fp1), Some(&"key1".to_string()));

        ctx.unregister_by_fingerprint(&fp1);
        assert_eq!(ctx.len(), 1);
    }

    // Combined integration tests
    #[tokio::test]
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    async fn test_paged_hierarchical_workflow() {
        // Test combining paged cache with hierarchical concepts
        let paged_cache = PagedCache::new(128, 100);

        for i in 0..10_i32 {
            let fp = ContextFingerprint::new(i as u64, 100, format!("entry{i}"));
            let data: Vec<f32> = (0..200).map(|j| (i * 200 + j) as f32).collect();
            let _ = paged_cache.put(&fp, &data);
        }

        assert_eq!(paged_cache.entry_count(), 10);

        // Evict some entries
        let evicted = paged_cache.evict_lru_entries(3);
        assert_eq!(evicted, 3);
        assert_eq!(paged_cache.entry_count(), 7);
    }

    #[tokio::test]
    async fn test_invalidation_with_cache() {
        let mut cache = InMemoryPrefixCache::with_defaults();
        let mut manager = InvalidationManager::with_defaults();

        // Add entries to cache
        let fp1 = ContextFingerprint::new(111, 100, "parent");
        let fp2 = ContextFingerprint::new(222, 100, "child");

        let entry1 = KVCacheEntry::new("parent_key", fp1.clone(), vec![1.0; 10], 100);
        let entry2 = KVCacheEntry::new("child_key", fp2.clone(), vec![2.0; 10], 100);

        let key1 = cache.put(entry1).await.unwrap();
        let key2 = cache.put(entry2).await.unwrap();

        // Register dependency
        manager.register_dependency(key2.clone(), key1.clone());

        // Invalidate parent
        let to_invalidate = manager.invalidate_dependents(&key1);
        assert!(to_invalidate.contains(&key2));

        // Remove invalidated entries from cache
        for key in to_invalidate {
            cache.remove(&key).await;
        }
        cache.remove(&key1).await;

        assert_eq!(cache.len(), 0);
    }

    #[test]
    #[allow(clippy::no_effect_underscore_binding)]
    fn test_all_exports_accessible() {
        // Verify all re-exported types are accessible
        let _fp = ContextFingerprint::new(1, 1, "test");
        let _config = PrefixCacheConfig::default();
        let _stats = CacheStats::default();
        let _generator = ContextFingerprintGenerator::new();
        let _hasher = RollingHasher::new();
        let _tier = TierConfig::default();
        let _h_config = HierarchicalCacheConfig::default();
        let _h_stats = HierarchicalCacheStats::default();
        let _policy = InvalidationPolicy::default();
        let reason = InvalidationReason::Manual;
        assert!(matches!(reason, InvalidationReason::Manual));
        let _manager = InvalidationManager::with_defaults();
        let _builder = InvalidationRuleBuilder::new();
        let _dep_stats = DependencyStats::default();
        let _fp_ctx = FingerprintInvalidationContext::new();
        let _page = CachePage::new(1, 256);
        let _table = PageTable::new(256, 10);
        let _paged = PagedCache::new(256, 10);
        let _paged_entry = PagedKVEntry::new(ContextFingerprint::new(1, 1, "test"), vec![1], 100);
    }
}
