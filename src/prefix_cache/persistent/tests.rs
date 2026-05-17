//! Tests for the persistent prefix-cache backend.

#![cfg(test)]
#![allow(clippy::cast_sign_loss)]

use super::store::{HybridPersistentCache, PersistentPrefixCache};
use super::types::{CacheIndex, IndexEntry, PersistentCacheConfig};
use crate::prefix_cache::traits::PrefixCacheStore;
use crate::prefix_cache::types::{ContextFingerprint, KVCacheEntry, PrefixCacheConfig};
use std::time::Duration;
use tempfile::TempDir;

fn create_test_entry(id: &str, hash: u64, kv_size: usize) -> KVCacheEntry {
    let fp = ContextFingerprint::new(hash, 100, format!("test {id}"));
    KVCacheEntry::new(id, fp, vec![0.0; kv_size], 100)
}

fn create_temp_config() -> (TempDir, PersistentCacheConfig) {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let config = PersistentCacheConfig::new(temp_dir.path());
    (temp_dir, config)
}

// Test 1: File creation and loading
#[tokio::test]
async fn test_persistent_cache_creation() {
    let (_temp_dir, config) = create_temp_config();
    let cache = PersistentPrefixCache::open(config.clone());
    assert!(cache.is_ok());

    let cache = cache.expect("test operation should succeed");
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

// Test 2: Entry persistence and retrieval
#[tokio::test]
async fn test_persistent_cache_put_and_get() {
    let (_temp_dir, config) = create_temp_config();
    let mut cache = PersistentPrefixCache::open(config).expect("test operation should succeed");

    let entry = create_test_entry("test1", 12345, 10);
    let fingerprint = entry.fingerprint.clone();

    let key = cache
        .put(entry)
        .await
        .expect("test operation should succeed");
    assert!(!key.is_empty());

    let retrieved = cache.get(&fingerprint).await;
    assert!(retrieved.is_some());
    assert_eq!(
        retrieved
            .expect("test operation should succeed")
            .fingerprint
            .hash,
        12345
    );
}

// Test 3: Index operations - save and load
#[tokio::test]
async fn test_persistent_cache_index_persistence() {
    let (temp_dir, config) = create_temp_config();

    // Create cache and add entry
    {
        let mut cache =
            PersistentPrefixCache::open(config.clone()).expect("test operation should succeed");
        let entry = create_test_entry("test1", 12345, 10);
        cache
            .put(entry)
            .await
            .expect("test operation should succeed");
        cache.save_index().expect("test operation should succeed");
    }

    // Reopen and verify
    {
        let cache = PersistentPrefixCache::open(PersistentCacheConfig::new(temp_dir.path()))
            .expect("test operation should succeed");
        let fp = ContextFingerprint::new(12345, 100, "test test1");
        assert!(cache.contains(&fp).await);
    }
}

// Test 4: Multiple entries
#[tokio::test]
async fn test_persistent_cache_multiple_entries() {
    let (_temp_dir, config) = create_temp_config();
    let mut cache = PersistentPrefixCache::open(config).expect("test operation should succeed");

    for i in 0..5 {
        let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
        cache
            .put(entry)
            .await
            .expect("test operation should succeed");
    }

    assert_eq!(cache.len(), 5);

    for i in 0..5 {
        let fp = ContextFingerprint::new(i as u64, 100, format!("test test{i}"));
        assert!(cache.contains(&fp).await);
    }
}

// Test 5: Remove entry
#[tokio::test]
async fn test_persistent_cache_remove() {
    let (_temp_dir, config) = create_temp_config();
    let mut cache = PersistentPrefixCache::open(config).expect("test operation should succeed");

    let entry = create_test_entry("test1", 12345, 10);
    let fingerprint = entry.fingerprint.clone();
    let key = cache
        .put(entry)
        .await
        .expect("test operation should succeed");

    assert!(cache.contains(&fingerprint).await);

    let removed = cache.remove(&key).await;
    assert!(removed.is_some());
    assert!(!cache.contains(&fingerprint).await);
}

// Test 6: Clear cache
#[tokio::test]
async fn test_persistent_cache_clear() {
    let (_temp_dir, config) = create_temp_config();
    let mut cache = PersistentPrefixCache::open(config).expect("test operation should succeed");

    for i in 0..5 {
        let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
        cache
            .put(entry)
            .await
            .expect("test operation should succeed");
    }

    assert_eq!(cache.len(), 5);

    cache.clear().await;
    assert!(cache.is_empty());
}

// Test 7: Compaction
#[tokio::test]
async fn test_persistent_cache_compaction() {
    let (_temp_dir, config) = create_temp_config();
    let mut cache = PersistentPrefixCache::open(config).expect("test operation should succeed");

    // Add entries with immediate expiration
    for i in 0..5 {
        let fp = ContextFingerprint::new(i as u64, 100, format!("test {i}"));
        let entry = KVCacheEntry::new(format!("test{i}"), fp, vec![0.0; 10], 100)
            .with_ttl(Duration::from_secs(0));
        cache
            .put(entry)
            .await
            .expect("test operation should succeed");
    }

    std::thread::sleep(Duration::from_millis(10));

    let stats = cache.compact().expect("test operation should succeed");
    assert_eq!(stats.entries_removed, 5);
}

// Test 8: TTL expiration
#[tokio::test]
async fn test_persistent_cache_ttl_expiration() {
    let (_temp_dir, config) = create_temp_config();
    let mut cache = PersistentPrefixCache::open(config).expect("test operation should succeed");

    let fp = ContextFingerprint::new(12345, 100, "test");
    let entry = KVCacheEntry::new("test1", fp.clone(), vec![0.0; 10], 100)
        .with_ttl(Duration::from_secs(0));

    cache
        .put(entry)
        .await
        .expect("test operation should succeed");

    std::thread::sleep(Duration::from_millis(10));
    let result = cache.get(&fp).await;
    assert!(result.is_none());
}

// Test 9: Evict expired entries
#[tokio::test]
async fn test_persistent_cache_evict_expired() {
    let (_temp_dir, config) = create_temp_config();
    let mut cache = PersistentPrefixCache::open(config).expect("test operation should succeed");

    for i in 0..5 {
        let fp = ContextFingerprint::new(i as u64, 100, format!("test {i}"));
        let entry = KVCacheEntry::new(format!("test{i}"), fp, vec![0.0; 10], 100)
            .with_ttl(Duration::from_secs(0));
        cache
            .put(entry)
            .await
            .expect("test operation should succeed");
    }

    assert_eq!(cache.len(), 5);

    std::thread::sleep(Duration::from_millis(10));
    let evicted = cache.evict_expired().await;

    assert_eq!(evicted, 5);
    assert!(cache.is_empty());
}

// Test 10: Hybrid cache creation
#[tokio::test]
async fn test_hybrid_cache_creation() {
    let (_temp_dir, persistent_config) = create_temp_config();
    let memory_config = PrefixCacheConfig::default();

    let cache = HybridPersistentCache::new(memory_config, persistent_config);
    assert!(cache.is_ok());

    let cache = cache.expect("test operation should succeed");
    assert!(cache.is_empty());
}

// Test 11: Hybrid cache put and get
#[tokio::test]
async fn test_hybrid_cache_put_and_get() {
    let (_temp_dir, persistent_config) = create_temp_config();
    let memory_config = PrefixCacheConfig::default();

    let mut cache = HybridPersistentCache::new(memory_config, persistent_config)
        .expect("test operation should succeed");

    let entry = create_test_entry("test1", 12345, 10);
    let fingerprint = entry.fingerprint.clone();

    cache
        .put(entry)
        .await
        .expect("test operation should succeed");

    let retrieved = cache.get(&fingerprint).await;
    assert!(retrieved.is_some());
}

// Test 12: Hybrid cache write-through
#[tokio::test]
async fn test_hybrid_cache_write_through() {
    let (_temp_dir, persistent_config) = create_temp_config();
    let memory_config = PrefixCacheConfig::default();

    let mut cache = HybridPersistentCache::new(memory_config, persistent_config)
        .expect("test operation should succeed")
        .with_write_through(true);

    let entry = create_test_entry("test1", 12345, 10);
    let fingerprint = entry.fingerprint.clone();

    cache
        .put(entry)
        .await
        .expect("test operation should succeed");

    // Should be in both caches
    assert!(cache.memory_cache.contains(&fingerprint).await);
    assert!(cache.persistent_cache.contains(&fingerprint).await);
}

// Test 13: Hybrid cache flush to disk
#[tokio::test]
async fn test_hybrid_cache_flush_to_disk() {
    let (_temp_dir, persistent_config) = create_temp_config();
    let memory_config = PrefixCacheConfig::default();

    let mut cache = HybridPersistentCache::new(memory_config, persistent_config)
        .expect("test operation should succeed")
        .with_write_through(false); // Disable write-through

    // Add entries only to memory
    for i in 0..5 {
        let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
        cache
            .memory_cache
            .put(entry)
            .await
            .expect("test operation should succeed");
    }

    assert_eq!(cache.memory_cache.len(), 5);
    assert_eq!(cache.persistent_cache.len(), 0);

    // Flush to disk
    let flushed = cache
        .flush_to_disk()
        .await
        .expect("test operation should succeed");
    assert_eq!(flushed, 5);
    assert_eq!(cache.persistent_cache.len(), 5);
}

// Test 14: Hybrid cache warm cache
#[tokio::test]
async fn test_hybrid_cache_warm_cache() {
    let (temp_dir, persistent_config) = create_temp_config();

    // First, populate persistent cache
    {
        let memory_config = PrefixCacheConfig::default();
        let mut cache = HybridPersistentCache::new(memory_config, persistent_config)
            .expect("test operation should succeed");

        for i in 0..5 {
            let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
            cache
                .persistent_cache
                .put(entry)
                .await
                .expect("test operation should succeed");
        }
        cache.sync().expect("test operation should succeed");
    }

    // Reopen and warm cache
    {
        let memory_config = PrefixCacheConfig::default();
        let mut cache = HybridPersistentCache::new(
            memory_config,
            PersistentCacheConfig::new(temp_dir.path()),
        )
        .expect("test operation should succeed");

        assert_eq!(cache.memory_cache.len(), 0);
        assert_eq!(cache.persistent_cache.len(), 5);

        let loaded = cache
            .warm_cache(3)
            .await
            .expect("test operation should succeed");
        assert_eq!(loaded, 3);
        assert_eq!(cache.memory_cache.len(), 3);
    }
}

// Test 15: Hybrid cache combined stats
#[tokio::test]
async fn test_hybrid_cache_combined_stats() {
    let (_temp_dir, persistent_config) = create_temp_config();
    let memory_config = PrefixCacheConfig::default();

    let mut cache = HybridPersistentCache::new(memory_config, persistent_config)
        .expect("test operation should succeed");

    let entry = create_test_entry("test1", 12345, 10);
    let fingerprint = entry.fingerprint.clone();

    cache
        .put(entry)
        .await
        .expect("test operation should succeed");
    cache.get(&fingerprint).await;

    let (memory_stats, _persistent_stats) = cache.combined_stats();
    assert_eq!(memory_stats.hits, 1);
}

// Test 16: PersistedEntry conversion
#[test]
fn test_persisted_entry_conversion() {
    use super::types::PersistedEntry;
    let fp = ContextFingerprint::new(12345, 100, "test content");
    let entry =
        KVCacheEntry::new("key1", fp.clone(), vec![1.0, 2.0, 3.0], 50).with_ttl_secs(3600);

    let persisted = PersistedEntry::from_kv_entry(&entry);
    assert_eq!(persisted.key, "key1");
    assert_eq!(persisted.fingerprint_hash, 12345);
    assert_eq!(persisted.kv_data, vec![1.0, 2.0, 3.0]);
    assert_eq!(persisted.ttl_secs, Some(3600));

    let restored = persisted.to_kv_entry();
    assert_eq!(restored.key, "key1");
    assert_eq!(restored.fingerprint.hash, 12345);
    assert_eq!(restored.kv_data, vec![1.0, 2.0, 3.0]);
}

// Test 17: CacheIndex operations
#[test]
fn test_cache_index_operations() {
    let mut index = CacheIndex::new();

    let entry = IndexEntry {
        fingerprint_hash: 12345,
        file_offset: 0,
        entry_size: 100,
        created_at_unix: 0,
        ttl_secs: None,
    };

    index.add("key1".to_string(), entry);
    assert_eq!(index.entry_count, 1);
    assert!(index.contains_hash(12345));
    assert_eq!(index.get_key_by_hash(12345), Some(&"key1".to_string()));

    let removed = index.remove("key1");
    assert!(removed.is_some());
    assert_eq!(index.entry_count, 0);
    assert!(!index.contains_hash(12345));
}

// Test 18: PersistentCacheConfig builder
#[test]
fn test_persistent_cache_config_builder() {
    let tmp = tempfile::TempDir::new().expect("test operation should succeed");
    let config = PersistentCacheConfig::new(tmp.path())
        .with_max_file_size(100 * 1024 * 1024)
        .with_sync_interval(30)
        .with_compression(true)
        .with_memory_index(false);

    assert_eq!(config.max_file_size_bytes, 100 * 1024 * 1024);
    assert_eq!(config.sync_interval_secs, 30);
    assert!(config.compression_enabled);
    assert!(!config.index_in_memory);
}

// Test 19: Persistent cache sync
#[tokio::test]
async fn test_persistent_cache_sync() {
    let (_temp_dir, config) = create_temp_config();
    let mut cache = PersistentPrefixCache::open(config).expect("test operation should succeed");

    let entry = create_test_entry("test1", 12345, 10);
    cache
        .put(entry)
        .await
        .expect("test operation should succeed");

    let result = cache.sync();
    assert!(result.is_ok());
}

// Test 20: Hybrid cache remove from both
#[tokio::test]
async fn test_hybrid_cache_remove() {
    let (_temp_dir, persistent_config) = create_temp_config();
    let memory_config = PrefixCacheConfig::default();

    let mut cache = HybridPersistentCache::new(memory_config, persistent_config)
        .expect("test operation should succeed");

    let entry = create_test_entry("test1", 12345, 10);
    let fingerprint = entry.fingerprint.clone();
    let key = cache
        .put(entry)
        .await
        .expect("test operation should succeed");

    assert!(cache.contains(&fingerprint).await);

    let removed = cache.remove(&key).await;
    assert!(removed.is_some());
    assert!(!cache.memory_cache.contains(&fingerprint).await);
}
