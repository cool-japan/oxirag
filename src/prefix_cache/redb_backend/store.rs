//! Main [`RedbPrefixCache`] struct and [`PrefixCacheStore`] implementation.

use std::path::Path;

use async_trait::async_trait;
use redb::{Database, ReadableDatabase, ReadableTable};

use super::super::traits::PrefixCacheStore;
use super::super::types::{
    CacheKey, CacheStats, ContextFingerprint, KVCacheEntry, PrefixCacheConfig,
};
use super::ops::{
    decode_entry_static, delete_by_raw_key, encode_entry, find_oldest_key, rebuild_stats,
    scan_counts,
};
use super::types::{CACHE_TABLE, PersistedKVEntry};
use crate::error::OxiRagError;

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// An ACID-compliant, redb-backed implementation of [`PrefixCacheStore`].
///
/// Entries are stored in a single redb table keyed by a composite
/// `"<hash>:<prefix_length>"` string. All reads and writes use explicit
/// transactions that are committed before returning, ensuring durability
/// even in the face of unexpected process termination.
///
/// ## TTL handling
///
/// TTL expiry is checked lazily on `get()` / `contains()` and eagerly on
/// `evict_expired()`. Expired entries are physically deleted when encountered.
///
/// ## Capacity enforcement
///
/// When `config.max_entries` would be exceeded by a `put()`, the backend
/// performs a full table scan to identify and delete the oldest entry by
/// `created_at_secs` before inserting the new one. This is O(n) but avoids
/// the need for a separate sorted-order structure in redb.
///
/// ## Memory tracking
///
/// `stats.total_bytes` is maintained incrementally on each mutation.
/// Because this is an embedded database, "memory usage" refers to the
/// estimated serialised size of all live entries, not actual process RSS.
#[cfg(feature = "prefix-cache-redb")]
pub struct RedbPrefixCache {
    /// The underlying redb database handle.
    pub(super) db: Database,
    /// Cache behaviour configuration.
    pub(super) config: PrefixCacheConfig,
    /// Running statistics.
    pub(super) stats: CacheStats,
    /// Default TTL (seconds) applied when an entry has no per-entry TTL.
    /// Zero means entries never expire by default.
    pub(super) ttl_secs: u64,
}

impl std::fmt::Debug for RedbPrefixCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedbPrefixCache")
            .field("config", &self.config)
            .field("ttl_secs", &self.ttl_secs)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Inherent impl
// ---------------------------------------------------------------------------

impl RedbPrefixCache {
    /// Open or create a redb database at `path` and initialise the prefix-cache
    /// table.
    ///
    /// If the file already exists, its existing data is preserved and the
    /// statistics counters (`hits`, `misses`, etc.) are reconstructed from the
    /// current table contents.
    ///
    /// # Errors
    ///
    /// Returns [`OxiRagError::Config`] if the database cannot be created or the
    /// initial table setup transaction fails.
    pub fn new(path: impl AsRef<Path>, config: PrefixCacheConfig) -> Result<Self, OxiRagError> {
        let db = Database::create(path.as_ref())
            .map_err(|e| OxiRagError::Config(format!("redb open failed: {e}")))?;

        // Ensure the table exists by opening a write transaction.
        {
            let write_txn = db
                .begin_write()
                .map_err(|e| OxiRagError::Config(format!("redb begin_write failed: {e}")))?;
            // Opening the table in write mode creates it if absent.
            write_txn
                .open_table(CACHE_TABLE)
                .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;
            write_txn
                .commit()
                .map_err(|e| OxiRagError::Config(format!("redb commit failed: {e}")))?;
        }

        // Bootstrap stats from existing data so `len()` / `memory_usage()` are
        // accurate immediately after opening a pre-existing database.
        let (entry_count, total_bytes) = scan_counts(&db)?;
        let ttl_secs = config.default_ttl_secs;

        let mut stats = CacheStats::default();
        stats.update_memory(total_bytes, entry_count);

        Ok(Self {
            db,
            config,
            stats,
            ttl_secs,
        })
    }

    /// Set the default TTL that is applied when no per-entry TTL is present.
    ///
    /// A value of `0` means "never expire by default".
    #[must_use]
    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = ttl_secs;
        self
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Build the composite table key for a fingerprint.
    pub(super) fn fp_key(fp: &ContextFingerprint) -> String {
        format!("{}:{}", fp.hash, fp.prefix_length)
    }
}

// ---------------------------------------------------------------------------
// PrefixCacheStore impl
// ---------------------------------------------------------------------------

#[async_trait]
#[cfg(feature = "prefix-cache-redb")]
impl PrefixCacheStore for RedbPrefixCache {
    /// Retrieve a cache entry by its fingerprint.
    ///
    /// Performs an exact-key lookup, checks wall-clock TTL expiry, then
    /// returns the entry if it is still valid.  On an expiry-triggered miss,
    /// the expired row is physically removed.
    async fn get(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        let raw_key = Self::fp_key(fingerprint);

        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(CACHE_TABLE).ok()?;
        let guard = table.get(raw_key.as_str()).ok()??;
        let persisted = decode_entry_static(guard.value()).ok()?;
        drop(guard);
        drop(table);
        drop(read_txn);

        if persisted.is_expired() {
            // Best-effort physical deletion of expired entry.
            // We need &mut self to delete, but get() only has &self.
            // We use an interior-mutability pattern via a write transaction
            // directly on the db handle (which is not wrapped in a lock here).
            // Safety: redb handles concurrent write correctly; at worst we
            // race with another write and the row will be cleaned up by the
            // next evict_expired().
            //
            // We cannot mutate self.stats here without &mut self, so we skip
            // the stat update and leave it for evict_expired().
            if let Ok(write_txn) = self.db.begin_write() {
                if let Ok(mut table) = write_txn.open_table(CACHE_TABLE) {
                    let _ = table.remove(raw_key.as_str());
                }
                let _ = write_txn.commit();
            }
            return None;
        }

        Some(persisted.to_kv_entry())
    }

    /// Store a cache entry, enforcing `max_entries` capacity.
    ///
    /// If `max_entries` would be exceeded, the oldest entry by creation
    /// timestamp is evicted before the new entry is inserted (LRU-by-age
    /// approximation). The entry's key is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying redb transaction fails.
    async fn put(&mut self, entry: KVCacheEntry) -> Result<CacheKey, OxiRagError> {
        let mut persisted = PersistedKVEntry::from_kv_entry(&entry);

        // Apply default TTL if the entry has none and the default is non-zero.
        if persisted.ttl_secs.is_none() && self.ttl_secs > 0 {
            persisted.ttl_secs = Some(self.ttl_secs);
        }

        let raw_key = format!(
            "{}:{}",
            persisted.fingerprint_hash, persisted.fingerprint_prefix_length
        );
        let entry_size = persisted.estimated_size();
        let return_key = persisted.key.clone();

        // Capacity enforcement: evict the oldest entry when we are at the limit.
        // We check by count (not memory) to match the persistent.rs pattern.
        if self.stats.entry_count >= self.config.max_entries
            && let Some(oldest_key) = find_oldest_key(&self.db)?
            && let Some(old_entry) = delete_by_raw_key(&self.db, &oldest_key)?
        {
            let old_size = old_entry.estimated_size();
            self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(old_size);
            self.stats.record_eviction();
        }

        let bytes = encode_entry(&persisted)?;

        // Check whether a row with this key already existed (fingerprint update).
        let old_size: usize = {
            let read_txn = self
                .db
                .begin_read()
                .map_err(|e| OxiRagError::Config(format!("redb begin_read failed: {e}")))?;
            let table = read_txn
                .open_table(CACHE_TABLE)
                .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;
            table
                .get(raw_key.as_str())
                .ok()
                .flatten()
                .and_then(|g| decode_entry_static(g.value()).ok())
                .map_or(0, |e| e.estimated_size())
        };

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| OxiRagError::Config(format!("redb begin_write failed: {e}")))?;
        {
            let mut table = write_txn
                .open_table(CACHE_TABLE)
                .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;
            table
                .insert(raw_key.as_str(), bytes.as_slice())
                .map_err(|e| OxiRagError::Config(format!("redb insert failed: {e}")))?;
        }
        write_txn
            .commit()
            .map_err(|e| OxiRagError::Config(format!("redb commit failed: {e}")))?;

        // Update stats.
        if old_size == 0 {
            // Brand-new entry.
            self.stats.entry_count += 1;
            self.stats.total_bytes += entry_size;
        } else {
            // Replaced existing entry; adjust size delta.
            self.stats.total_bytes = self
                .stats
                .total_bytes
                .saturating_sub(old_size)
                .saturating_add(entry_size);
        }

        Ok(return_key)
    }

    /// Remove an entry identified by its cache key.
    ///
    /// The cache key stored in the database is the raw composite key
    /// `"<hash>:<prefix_length>"`.  If the caller supplies the original
    /// [`CacheKey`] (which may differ), we perform a full scan to find a
    /// matching entry.
    ///
    /// Returns the removed entry if found, or `None` if the key is unknown.
    async fn remove(&mut self, key: &CacheKey) -> Option<KVCacheEntry> {
        // Strategy: First try interpreting `key` directly as a raw table key.
        // If that misses, fall back to a full scan matching `entry.key == key`.

        // Direct attempt.
        if let Ok(Some(persisted)) = delete_by_raw_key(&self.db, key) {
            let size = persisted.estimated_size();
            self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(size);
            return Some(persisted.to_kv_entry());
        }

        // Full-scan fallback: find the raw table key whose stored `key` field
        // matches the requested CacheKey.
        let matching_raw_key: Option<String> = {
            let read_txn = self.db.begin_read().ok()?;
            let table = read_txn.open_table(CACHE_TABLE).ok()?;
            let mut found = None;
            let iter = table.iter().ok()?;
            for (raw_key, value) in iter.flatten() {
                if let Ok(entry) = decode_entry_static(value.value())
                    && &entry.key == key
                {
                    found = Some(raw_key.value().to_owned());
                    break;
                }
            }
            found
        };

        if let Some(raw_key) = matching_raw_key
            && let Ok(Some(persisted)) = delete_by_raw_key(&self.db, &raw_key)
        {
            let size = persisted.estimated_size();
            self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(size);
            return Some(persisted.to_kv_entry());
        }

        None
    }

    /// Return `true` if the fingerprint has a live (non-expired) entry in the
    /// cache.  Does not update access statistics.
    async fn contains(&self, fingerprint: &ContextFingerprint) -> bool {
        let raw_key = Self::fp_key(fingerprint);

        let Ok(read_txn) = self.db.begin_read() else {
            return false;
        };
        let Ok(table) = read_txn.open_table(CACHE_TABLE) else {
            return false;
        };
        let Ok(Some(guard)) = table.get(raw_key.as_str()) else {
            return false;
        };
        decode_entry_static(guard.value()).is_ok_and(|e| !e.is_expired())
    }

    /// Delete all entries from the cache and reset statistics.
    async fn clear(&mut self) {
        // Phase 1: collect all existing keys via a read transaction.
        let keys: Vec<String> = {
            let Ok(read_txn) = self.db.begin_read() else {
                return;
            };
            let Ok(table) = read_txn.open_table(CACHE_TABLE) else {
                return;
            };
            let Ok(iter) = table.iter() else { return };
            iter.filter_map(Result::ok)
                .map(|(k, _)| k.value().to_owned())
                .collect()
        };

        if keys.is_empty() {
            self.stats.entry_count = 0;
            self.stats.total_bytes = 0;
            return;
        }

        // Phase 2: delete all collected keys in a single write transaction.
        let Ok(write_txn) = self.db.begin_write() else {
            return;
        };
        {
            let Ok(mut table) = write_txn.open_table(CACHE_TABLE) else {
                return;
            };
            for k in &keys {
                let _ = table.remove(k.as_str());
            }
        }
        let _ = write_txn.commit();

        self.stats.entry_count = 0;
        self.stats.total_bytes = 0;
    }

    /// Return a snapshot of the current cache statistics.
    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Return the number of live entries currently in the cache.
    fn len(&self) -> usize {
        self.stats.entry_count
    }

    /// Return `true` if the cache contains no live entries.
    fn is_empty(&self) -> bool {
        self.stats.entry_count == 0
    }

    /// Find the longest cached entry whose `prefix_length` is strictly less
    /// than `fingerprint.prefix_length` (a proper prefix of the query).
    ///
    /// This is an O(n) scan over all live entries. Entries that are expired are
    /// skipped.  Returns the entry with the greatest `prefix_length` that still
    /// satisfies the prefix condition, or `None` if no such entry exists.
    async fn find_prefix_match(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(CACHE_TABLE).ok()?;
        let iter = table.iter().ok()?;

        let mut best: Option<PersistedKVEntry> = None;
        let mut best_len = 0usize;

        for result in iter {
            // Skip items that produce I/O errors rather than aborting the scan.
            let Ok((_raw_key, value)) = result else {
                continue;
            };
            let Ok(entry) = decode_entry_static(value.value()) else {
                continue;
            };

            if entry.is_expired() {
                continue;
            }

            // For find_prefix_match we want entries strictly shorter than the
            // query (partial match); equal-length counts as exact, not prefix.
            if entry.fingerprint_prefix_length < fingerprint.prefix_length
                && entry.fingerprint_prefix_length > best_len
            {
                best_len = entry.fingerprint_prefix_length;
                best = Some(entry);
            }
        }

        best.map(|e| e.to_kv_entry())
    }

    /// Scan the table for expired entries and physically delete them.
    ///
    /// Returns the count of entries removed.
    async fn evict_expired(&mut self) -> usize {
        // Collect expired keys in a read pass.
        let expired_keys: Vec<String> = {
            let Ok(read_txn) = self.db.begin_read() else {
                return 0;
            };
            let Ok(table) = read_txn.open_table(CACHE_TABLE) else {
                return 0;
            };
            let Ok(iter) = table.iter() else { return 0 };
            iter.filter_map(Result::ok)
                .filter_map(|(k, v)| {
                    decode_entry_static(v.value())
                        .ok()
                        .filter(PersistedKVEntry::is_expired)
                        .map(|_| k.value().to_owned())
                })
                .collect()
        };

        let count = expired_keys.len();
        if count == 0 {
            return 0;
        }

        // Delete in a single write transaction.
        if let Ok(write_txn) = self.db.begin_write() {
            if let Ok(mut table) = write_txn.open_table(CACHE_TABLE) {
                for k in &expired_keys {
                    let _ = table.remove(k.as_str());
                }
            }
            let _ = write_txn.commit();
        }

        self.stats.expirations = self.stats.expirations.saturating_add(count as u64);
        // Rebuild accurate counts since we deleted an unknown total byte size.
        let _ = rebuild_stats(&self.db, &mut self.stats);

        count
    }

    /// Return the estimated total byte size of all live entries.
    ///
    /// This is a running accumulation of `PersistedKVEntry::estimated_size()`
    /// values and may differ slightly from the actual on-disk size due to JSON
    /// encoding overhead.
    fn memory_usage(&self) -> usize {
        self.stats.total_bytes
    }
}

// ---------------------------------------------------------------------------
// Stat helpers that require &mut self but are called from &self get()
// ---------------------------------------------------------------------------

// Note: get() uses &self per the trait contract, so we perform stat updates
// lazily (hits/misses are NOT updated in get() because the trait provides no
// &mut self there). Instead, callers should track hits/misses externally or
// the cache can be wrapped in a mutex. The stats fields `hits` and `misses`
// remain accurate only when put/remove/evict_expired are used. This matches
// the persistent.rs behaviour where stats.record_hit() is called under an
// Arc<RwLock<CacheStats>>.
//
// If the project requires accurate get() stats, wrapping RedbPrefixCache in
// a Mutex<RedbPrefixCache> adapter at a higher layer is the idiomatic solution.
