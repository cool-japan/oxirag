//! Tests for the redb prefix-cache backend.

#![cfg(test)]
#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::float_cmp)]

use std::time::Duration;

use super::store::RedbPrefixCache;
use crate::prefix_cache::traits::PrefixCacheStore;
use crate::prefix_cache::types::{ContextFingerprint, KVCacheEntry, PrefixCacheConfig};

/// Construct a test entry with a given string id, hash, and kv data size.
fn make_entry(id: &str, hash: u64, prefix_len: usize, kv_size: usize) -> KVCacheEntry {
    let fp = ContextFingerprint::new(hash, prefix_len, format!("summary for {id}"));
    KVCacheEntry::new(id, fp, vec![1.0_f32; kv_size], prefix_len)
}

/// Open a fresh [`RedbPrefixCache`] backed by a file inside `dir`.
fn open_cache(dir: &tempfile::TempDir) -> RedbPrefixCache {
    let path = dir.path().join("test.redb");
    RedbPrefixCache::new(path, PrefixCacheConfig::default())
        .expect("RedbPrefixCache::new should succeed")
}

// -----------------------------------------------------------------------
// Test 1 – basic put / get round-trip
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_put_and_get() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let mut cache = open_cache(&dir);

    let entry = make_entry("e1", 111, 100, 16);
    let fp = entry.fingerprint.clone();

    let key = cache.put(entry).await.expect("put should succeed");
    assert!(!key.is_empty(), "returned key must not be empty");

    let retrieved = cache.get(&fp).await;
    assert!(retrieved.is_some(), "entry should be retrievable after put");
    let hit = retrieved.expect("get returned None unexpectedly");
    assert_eq!(hit.fingerprint.hash, 111, "fingerprint hash mismatch");
    assert_eq!(hit.kv_data.len(), 16, "kv_data length mismatch");
}

// -----------------------------------------------------------------------
// Test 2 – contains() returns false for a miss
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_contains_false_after_miss() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let cache = open_cache(&dir);

    let absent_fp = ContextFingerprint::new(999_999, 50, "no such entry");
    assert!(
        !cache.contains(&absent_fp).await,
        "contains() must return false for an absent fingerprint"
    );
}

// -----------------------------------------------------------------------
// Test 3 – remove deletes the entry
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_remove() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let mut cache = open_cache(&dir);

    let entry = make_entry("e2", 222, 100, 8);
    let fp = entry.fingerprint.clone();

    // `put()` returns the `entry.key` field ("e2"), but the table-level key
    // is the composite fingerprint key ("222:100").  Both forms work with
    // `remove()`: passing the composite key hits the direct path, while the
    // returned CacheKey triggers the scan-fallback path.  We test both.

    // First pass: remove via the CacheKey returned from put().
    cache.put(entry.clone()).await.expect("put should succeed");
    assert!(cache.contains(&fp).await, "entry must exist before remove");

    let returned_key = entry.key.clone(); // "e2"
    let removed = cache.remove(&returned_key).await;
    assert!(
        removed.is_some(),
        "remove() via CacheKey should return the deleted entry"
    );
    assert!(
        !cache.contains(&fp).await,
        "entry must not exist after remove via CacheKey"
    );

    // Second pass: remove via the raw composite table key.
    let entry2 = make_entry("e2b", 222, 100, 8);
    cache.put(entry2).await.expect("second put should succeed");
    assert!(
        cache.contains(&fp).await,
        "entry must exist for second remove"
    );

    let raw_key = format!("{}:{}", 222_u64, 100_usize);
    let removed2 = cache.remove(&raw_key).await;
    assert!(
        removed2.is_some(),
        "remove() via raw composite key should return the deleted entry"
    );
    assert!(
        !cache.contains(&fp).await,
        "entry must not exist after remove via raw key"
    );
    assert_eq!(cache.len(), 0, "cache must be empty after all removes");
}

// -----------------------------------------------------------------------
// Test 4 – clear empties the cache
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_clear() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let mut cache = open_cache(&dir);

    for i in 0_u64..5 {
        let entry = make_entry(&format!("e{i}"), i, 100, 4);
        cache.put(entry).await.expect("put should succeed");
    }

    assert_eq!(cache.len(), 5, "cache should hold 5 entries before clear");

    cache.clear().await;

    assert!(cache.is_empty(), "cache must be empty after clear");
    assert_eq!(cache.len(), 0, "len() must return 0 after clear");
    assert_eq!(
        cache.memory_usage(),
        0,
        "memory_usage() must return 0 after clear"
    );
}

// -----------------------------------------------------------------------
// Test 5 – len() and is_empty() track insertions correctly
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_len_and_is_empty() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let mut cache = open_cache(&dir);

    assert!(cache.is_empty(), "fresh cache must be empty");
    assert_eq!(cache.len(), 0);

    let entry = make_entry("e1", 1, 10, 4);
    cache.put(entry).await.expect("put should succeed");
    assert!(!cache.is_empty(), "cache must not be empty after put");
    assert_eq!(cache.len(), 1);

    let entry2 = make_entry("e2", 2, 20, 4);
    cache.put(entry2).await.expect("put should succeed");
    assert_eq!(cache.len(), 2);
}

// -----------------------------------------------------------------------
// Test 6 – stats eviction counter increments on capacity overflow
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_stats_tracking() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let path = dir.path().join("stats.redb");

    // Use a tiny capacity to force an eviction.
    let config = PrefixCacheConfig {
        max_entries: 2,
        max_memory_bytes: 512 * 1024 * 1024,
        default_ttl_secs: 3600,
        enable_compression: false,
    };
    let mut cache =
        RedbPrefixCache::new(path, config).expect("RedbPrefixCache::new should succeed");

    cache
        .put(make_entry("e1", 1, 10, 4))
        .await
        .expect("put should succeed");
    cache
        .put(make_entry("e2", 2, 20, 4))
        .await
        .expect("put should succeed");

    assert_eq!(cache.len(), 2, "should have 2 entries after 2 puts");
    assert_eq!(cache.stats().evictions, 0, "no evictions yet");

    // Third put triggers eviction because max_entries == 2.
    cache
        .put(make_entry("e3", 3, 30, 4))
        .await
        .expect("put should succeed");

    assert_eq!(cache.len(), 2, "should still have 2 entries after eviction");
    assert_eq!(
        cache.stats().evictions,
        1,
        "one eviction should have been recorded"
    );
}

// -----------------------------------------------------------------------
// Test 7 – TTL=0 causes immediate expiry on get()
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_ttl_expiry() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let mut cache = open_cache(&dir);

    let fp = ContextFingerprint::new(777, 100, "ttl test");
    let entry = KVCacheEntry::new("ttl_key", fp.clone(), vec![0.0; 8], 100)
        .with_ttl(Duration::from_secs(0)); // Expires immediately.

    cache.put(entry).await.expect("put should succeed");

    // The entry is in the DB but its TTL is 0 → get() should see it as expired.
    // Give any sub-millisecond clock a chance to advance.
    std::thread::sleep(Duration::from_millis(5));

    let result = cache.get(&fp).await;
    assert!(
        result.is_none(),
        "entry with TTL=0 must not be returned by get()"
    );
}

// -----------------------------------------------------------------------
// Test 8 – evict_expired() removes expired rows and returns count
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_evict_expired() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let mut cache = open_cache(&dir);

    // Insert 3 entries with immediate TTL and 2 without.
    for i in 0_u64..3 {
        let fp = ContextFingerprint::new(
            i,
            100 + usize::try_from(i).expect("i fits usize"),
            format!("exp {i}"),
        );
        let entry = KVCacheEntry::new(format!("exp_{i}"), fp, vec![0.0; 4], 100)
            .with_ttl(Duration::from_secs(0));
        cache.put(entry).await.expect("put should succeed");
    }
    for i in 10_u64..12 {
        let fp = ContextFingerprint::new(
            i,
            200 + usize::try_from(i).expect("i fits usize"),
            format!("live {i}"),
        );
        let entry = KVCacheEntry::new(format!("live_{i}"), fp, vec![1.0; 4], 100);
        cache.put(entry).await.expect("put should succeed");
    }

    assert_eq!(cache.len(), 5, "should have 5 entries before eviction");

    std::thread::sleep(Duration::from_millis(10));

    let evicted = cache.evict_expired().await;
    assert_eq!(evicted, 3, "three expired entries should have been evicted");
    assert_eq!(cache.len(), 2, "two live entries should remain");
}

// -----------------------------------------------------------------------
// Test 9 – find_prefix_match returns longest shorter prefix
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_find_prefix_match() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let mut cache = open_cache(&dir);

    // Insert a short prefix (length 30) and a medium one (length 60).
    let short_fp = ContextFingerprint::new(100, 30, "short prefix");
    let medium_fp = ContextFingerprint::new(200, 60, "medium prefix");

    cache
        .put(make_entry("short", 100, 30, 4))
        .await
        .expect("put should succeed");
    cache
        .put(make_entry("medium", 200, 60, 4))
        .await
        .expect("put should succeed");

    // Query with prefix_length=100 – both entries qualify; medium wins.
    let query_fp = ContextFingerprint::new(999, 100, "long query");
    let result = cache.find_prefix_match(&query_fp).await;

    assert!(result.is_some(), "a prefix match should be found");
    let matched = result.expect("find_prefix_match returned None unexpectedly");
    assert_eq!(
        matched.fingerprint.prefix_length, 60,
        "the longest qualifying prefix (60) should be returned"
    );

    // Query with prefix_length=50 – only the short one qualifies.
    let mid_query = ContextFingerprint::new(998, 50, "mid query");
    let result2 = cache.find_prefix_match(&mid_query).await;
    assert!(result2.is_some(), "short prefix should match mid query");
    assert_eq!(
        result2
            .expect("find_prefix_match returned None unexpectedly")
            .fingerprint
            .prefix_length,
        30
    );

    // Query with prefix_length=10 – nothing qualifies.
    let tiny_query = ContextFingerprint::new(997, 10, "tiny query");
    let result3 = cache.find_prefix_match(&tiny_query).await;
    assert!(
        result3.is_none(),
        "no prefix match should exist for a shorter-than-all query"
    );

    // Sanity: exact match doesn't count as prefix.
    let exact_query = ContextFingerprint::new(997, 30, "exact query");
    let result4 = cache.find_prefix_match(&exact_query).await;
    assert!(
        result4.is_none(),
        "an entry with the same prefix_length should not be a prefix match"
    );

    let _ = short_fp;
    let _ = medium_fp;
}

// -----------------------------------------------------------------------
// Test 10 – memory_usage() reflects inserted data
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_redb_memory_usage() {
    let dir = tempfile::TempDir::new().expect("temp dir creation should succeed");
    let mut cache = open_cache(&dir);

    assert_eq!(
        cache.memory_usage(),
        0,
        "memory usage must be 0 for an empty cache"
    );

    let entry = make_entry("mem_test", 42, 100, 256);
    let expected_min = 256 * std::mem::size_of::<f32>(); // 1024 bytes minimum for kv_data.

    cache.put(entry).await.expect("put should succeed");

    assert!(
        cache.memory_usage() >= expected_min,
        "memory_usage() ({}) should be at least {} bytes for 256 f32 values",
        cache.memory_usage(),
        expected_min
    );

    cache.clear().await;
    assert_eq!(
        cache.memory_usage(),
        0,
        "memory usage must return to 0 after clear"
    );
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::pedantic
)]
mod prop_tests {
    use std::time::Duration;

    use proptest::prelude::*;

    use super::super::store::RedbPrefixCache;
    use crate::prefix_cache::traits::PrefixCacheStore;
    use crate::prefix_cache::types::{ContextFingerprint, KVCacheEntry, PrefixCacheConfig};

    // -----------------------------------------------------------------------
    // Strategies
    // -----------------------------------------------------------------------

    /// Strategy: generate an arbitrary [`ContextFingerprint`].
    fn arb_fingerprint() -> impl Strategy<Value = ContextFingerprint> {
        (any::<u64>(), 1usize..1000usize, "[a-z]{1,20}")
            .prop_map(|(hash, len, summary)| ContextFingerprint::new(hash, len, summary))
    }

    /// Strategy: generate arbitrary KV data (1..=128 f32 values in [-1, 1)).
    fn arb_kv_data() -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-1.0f32..1.0f32, 1..=128)
    }

    /// Open a fresh [`RedbPrefixCache`] backed by a unique file in `dir`.
    fn open_prop_cache(dir: &tempfile::TempDir, suffix: &str) -> RedbPrefixCache {
        let path = dir.path().join(format!("prop_{suffix}.redb"));
        RedbPrefixCache::new(path, PrefixCacheConfig::default())
            .expect("RedbPrefixCache::new should succeed in proptest")
    }

    // -----------------------------------------------------------------------
    // Test 1 – put / get round-trip preserves kv_data
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_put_get_roundtrip(
            fp in arb_fingerprint(),
            kv_data in arb_kv_data(),
        ) {
            let dir = tempfile::TempDir::new().expect("tempdir creation must succeed");
            let mut cache = open_prop_cache(&dir, "rtrip");

            let entry = KVCacheEntry::new(
                "prop_key",
                fp.clone(),
                kv_data.clone(),
                fp.prefix_length,
            );

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                cache.put(entry).await.expect("put must succeed");
                let retrieved = cache.get(&fp).await;
                prop_assert!(
                    retrieved.is_some(),
                    "get() must return Some after a successful put"
                );
                let hit = retrieved.expect("checked above");
                prop_assert_eq!(
                    hit.kv_data.len(),
                    kv_data.len(),
                    "kv_data length must be preserved through the round-trip"
                );
                for (i, (got, expected)) in hit.kv_data.iter().zip(kv_data.iter()).enumerate() {
                    prop_assert!(
                        (got - expected).abs() < f32::EPSILON,
                        "kv_data[{}] mismatch: got {got}, expected {expected}",
                        i
                    );
                }
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 – contains() returns true after a put
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_contains_after_put(
            fp in arb_fingerprint(),
            kv_data in arb_kv_data(),
        ) {
            let dir = tempfile::TempDir::new().expect("tempdir creation must succeed");
            let mut cache = open_prop_cache(&dir, "contains");

            let entry = KVCacheEntry::new("ck", fp.clone(), kv_data, fp.prefix_length);

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                cache.put(entry).await.expect("put must succeed");
                prop_assert!(
                    cache.contains(&fp).await,
                    "contains() must return true for a fingerprint that was just put"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 – len() increments correctly for N distinct fingerprints
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_len_increments(
            // Produce up to 8 distinct hashes; use u8 to keep the pool small
            hashes in prop::collection::hash_set(any::<u8>().prop_map(|b| b as u64), 1..=8usize),
        ) {
            let dir = tempfile::TempDir::new().expect("tempdir creation must succeed");
            let mut cache = open_prop_cache(&dir, "len");
            let n = hashes.len();

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                for (idx, hash) in hashes.into_iter().enumerate() {
                    let fp = ContextFingerprint::new(hash, idx + 1, "s");
                    let entry = KVCacheEntry::new(format!("k{idx}"), fp, vec![0.0_f32; 4], idx + 1);
                    cache.put(entry).await.expect("put must succeed");
                }
                prop_assert_eq!(
                    cache.len(),
                    n,
                    "len() must equal the number of distinct puts, got {} expected {}",
                    cache.len(),
                    n
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 – remove() decrements len and returns the removed entry
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_remove_decrements_len(
            fp in arb_fingerprint(),
            kv_data in arb_kv_data(),
        ) {
            let dir = tempfile::TempDir::new().expect("tempdir creation must succeed");
            let mut cache = open_prop_cache(&dir, "remove");

            let raw_key = format!("{}:{}", fp.hash, fp.prefix_length);
            let entry = KVCacheEntry::new("rk", fp.clone(), kv_data, fp.prefix_length);

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                cache.put(entry).await.expect("put must succeed");
                prop_assert_eq!(cache.len(), 1, "len must be 1 after one put");

                let removed = cache.remove(&raw_key).await;
                prop_assert!(
                    removed.is_some(),
                    "remove() must return Some for a key that exists"
                );
                prop_assert_eq!(
                    cache.len(),
                    0,
                    "len must drop to 0 after the only entry is removed"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 – clear() empties the cache regardless of how many entries exist
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_clear_empties_cache(
            hashes in prop::collection::hash_set(any::<u8>().prop_map(|b| b as u64), 1..=8usize),
        ) {
            let dir = tempfile::TempDir::new().expect("tempdir creation must succeed");
            let mut cache = open_prop_cache(&dir, "clear");

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                for (idx, hash) in hashes.into_iter().enumerate() {
                    let fp = ContextFingerprint::new(hash, idx + 1, "c");
                    let entry = KVCacheEntry::new(format!("c{idx}"), fp, vec![1.0_f32; 2], idx + 1);
                    cache.put(entry).await.expect("put must succeed");
                }
                prop_assume!(cache.len() > 0);

                cache.clear().await;

                prop_assert_eq!(
                    cache.len(),
                    0,
                    "len() must be 0 after clear()"
                );
                prop_assert!(cache.is_empty(), "is_empty() must return true after clear()");
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 – two distinct fingerprints do not interfere with each other
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_distinct_fingerprints_independent(
            hash1 in 0u64..u64::MAX / 2,
            hash2 in (u64::MAX / 2)..u64::MAX,
            kv1 in arb_kv_data(),
            kv2 in arb_kv_data(),
        ) {
            let dir = tempfile::TempDir::new().expect("tempdir creation must succeed");
            let mut cache = open_prop_cache(&dir, "indep");

            let fp1 = ContextFingerprint::new(hash1, 10, "fp1");
            let fp2 = ContextFingerprint::new(hash2, 20, "fp2");
            let len1 = kv1.len();
            let len2 = kv2.len();

            let entry1 = KVCacheEntry::new("k1", fp1.clone(), kv1, 10);
            let entry2 = KVCacheEntry::new("k2", fp2.clone(), kv2, 20);

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                cache.put(entry1).await.expect("put fp1 must succeed");
                cache.put(entry2).await.expect("put fp2 must succeed");

                let r1 = cache.get(&fp1).await;
                let r2 = cache.get(&fp2).await;

                prop_assert!(r1.is_some(), "fp1 must be retrievable");
                prop_assert!(r2.is_some(), "fp2 must be retrievable");

                prop_assert_eq!(
                    r1.expect("checked").kv_data.len(),
                    len1,
                    "fp1 kv_data length must be unchanged"
                );
                prop_assert_eq!(
                    r2.expect("checked").kv_data.len(),
                    len2,
                    "fp2 kv_data length must be unchanged"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 – find_prefix_match returns a cached entry shorter than the query
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_find_prefix_match_shorter_found(
            short_len in 1usize..50usize,
            long_len  in 51usize..200usize,
            hash_cached in any::<u64>(),
            hash_query  in any::<u64>(),
        ) {
            let dir = tempfile::TempDir::new().expect("tempdir creation must succeed");
            let mut cache = open_prop_cache(&dir, "pfx");

            // Cache an entry with `short_len`; query with `long_len` (> short_len).
            let fp_short = ContextFingerprint::new(hash_cached, short_len, "short");
            let fp_query  = ContextFingerprint::new(hash_query,  long_len,  "query");

            let entry = KVCacheEntry::new("ps", fp_short, vec![0.5_f32; 4], short_len);

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                cache.put(entry).await.expect("put must succeed");

                let matched = cache.find_prefix_match(&fp_query).await;
                prop_assert!(
                    matched.is_some(),
                    "find_prefix_match must find the shorter cached entry \
                     (cached_len={short_len}, query_len={long_len})"
                );
                prop_assert_eq!(
                    matched.expect("checked").fingerprint.prefix_length,
                    short_len,
                    "matched entry must have the cached prefix_length"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 – entry with TTL=0 is not returned by get() (expires immediately)
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_ttl_zero_expires_immediately(
            fp in arb_fingerprint(),
            kv_data in arb_kv_data(),
        ) {
            let dir = tempfile::TempDir::new().expect("tempdir creation must succeed");
            let mut cache = open_prop_cache(&dir, "ttl0");

            let entry = KVCacheEntry::new("ttl_prop", fp.clone(), kv_data, fp.prefix_length)
                .with_ttl(Duration::from_secs(0)); // expires immediately

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                cache.put(entry).await.expect("put must succeed");
                // Give the clock at least 1 ms to advance past the 0-second TTL.
                std::thread::sleep(Duration::from_millis(5));

                let result = cache.get(&fp).await;
                prop_assert!(
                    result.is_none(),
                    "get() must return None for an entry with TTL=0 after any non-zero wall-clock duration"
                );
                Ok(())
            })?;
        }
    }
}
