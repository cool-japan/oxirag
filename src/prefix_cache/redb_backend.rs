//! Redb-based persistent storage backend for prefix cache.
//!
//! This module provides an ACID-compliant, embedded database backend for the
//! prefix cache system using the `redb` pure-Rust key-value store. Unlike the
//! file-based [`crate::prefix_cache::PersistentPrefixCache`], this backend
//! guarantees transactional consistency, automatic crash recovery, and concurrent
//! read access without manual compaction.
//!
//! # Feature Flag
//!
//! This module is only compiled when the `prefix-cache-redb` feature is enabled.
//!
//! # Architecture
//!
//! Entries are serialised as JSON blobs and stored under string keys of the form
//! `"<hash>:<prefix_length>"`. This composite key enables fast exact-match
//! lookups and allows prefix-match scanning to iterate by prefix-length order
//! implicitly (via full scan with comparison).
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::prefix_cache::{
//!     RedbPrefixCache, RedbPrefixCacheConfig, PrefixCacheConfig, PrefixCacheStore,
//!     ContextFingerprint, KVCacheEntry,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = RedbPrefixCacheConfig {
//!         path: "/tmp/my_cache.redb".into(),
//!         cache_config: PrefixCacheConfig::default(),
//!         ttl_secs: 3600,
//!     };
//!
//!     let mut cache = RedbPrefixCache::new(config.path, config.cache_config)?;
//!
//!     let fp = ContextFingerprint::new(12345, 100, "example context");
//!     let entry = KVCacheEntry::new("key1", fp.clone(), vec![0.1; 128], 100);
//!     cache.put(entry).await?;
//!
//!     if let Some(hit) = cache.get(&fp).await {
//!         println!("Cache hit: {}", hit.key);
//!     }
//!
//!     Ok(())
//! }
//! ```

#![cfg(feature = "prefix-cache-redb")]

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use super::traits::PrefixCacheStore;
use super::types::{CacheKey, CacheStats, ContextFingerprint, KVCacheEntry, PrefixCacheConfig};
use crate::error::OxiRagError;

// ---------------------------------------------------------------------------
// Table definition
// ---------------------------------------------------------------------------

/// The single redb table that backs the prefix cache.
///
/// Keys are composite strings in the form `"<fingerprint_hash>:<prefix_length>"`.
/// Values are JSON-encoded [`PersistedKVEntry`] blobs.
const CACHE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("prefix_cache");

// ---------------------------------------------------------------------------
// Serialisable entry
// ---------------------------------------------------------------------------

/// A serialisable representation of a [`KVCacheEntry`].
///
/// [`KVCacheEntry`] contains [`std::time::Instant`] fields which are not
/// serialisable. This struct replaces them with Unix-epoch second timestamps
/// obtained from [`SystemTime`], making the entry safe to persist across
/// process restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedKVEntry {
    /// The unique cache key.
    pub key: String,
    /// Hash component of the fingerprint.
    pub fingerprint_hash: u64,
    /// Prefix-length component of the fingerprint.
    pub fingerprint_prefix_length: usize,
    /// Human-readable content summary from the fingerprint.
    pub fingerprint_summary: String,
    /// The cached KV data stored as 32-bit floats.
    pub kv_data: Vec<f32>,
    /// Number of tokens/characters in the cached sequence.
    pub sequence_length: usize,
    /// Unix timestamp (seconds) when this entry was created.
    pub created_at_secs: u64,
    /// Unix timestamp (seconds) when this entry was last accessed.
    pub last_accessed_secs: u64,
    /// How many times this entry has been accessed.
    pub access_count: u64,
    /// Optional TTL in seconds.  `None` means the entry never expires.
    pub ttl_secs: Option<u64>,
}

impl PersistedKVEntry {
    /// Convert a live [`KVCacheEntry`] to a persistable form.
    ///
    /// `created_at` and `last_accessed` `Instant` values are approximated by
    /// `SystemTime::now()` because `Instant` has no stable relationship to
    /// wall-clock time that can survive a process restart.
    #[must_use]
    pub fn from_kv_entry(entry: &KVCacheEntry) -> Self {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            key: entry.key.clone(),
            fingerprint_hash: entry.fingerprint.hash,
            fingerprint_prefix_length: entry.fingerprint.prefix_length,
            fingerprint_summary: entry.fingerprint.content_summary.clone(),
            kv_data: entry.kv_data.clone(),
            sequence_length: entry.sequence_length,
            created_at_secs: now_secs,
            last_accessed_secs: now_secs,
            access_count: entry.access_count,
            ttl_secs: entry.ttl.map(|d| d.as_secs()),
        }
    }

    /// Reconstruct a [`KVCacheEntry`] from this persisted form.
    ///
    /// Because [`std::time::Instant`] cannot be recovered from a Unix timestamp,
    /// `created_at` and `last_accessed` are set to [`std::time::Instant::now()`],
    /// which means `age()` and `time_since_access()` reflect time-since-load
    /// rather than true historical age.  TTL expiry is enforced separately via
    /// wall-clock timestamps stored in this struct.
    #[must_use]
    pub fn to_kv_entry(&self) -> KVCacheEntry {
        let fingerprint = ContextFingerprint::new(
            self.fingerprint_hash,
            self.fingerprint_prefix_length,
            &self.fingerprint_summary,
        );

        let mut entry = KVCacheEntry::new(
            &self.key,
            fingerprint,
            self.kv_data.clone(),
            self.sequence_length,
        );

        if let Some(ttl_secs) = self.ttl_secs {
            entry = entry.with_ttl_secs(ttl_secs);
        }

        entry.access_count = self.access_count;
        entry
    }

    /// Determine whether this entry has expired using stored wall-clock data.
    ///
    /// Uses `created_at_secs` rather than `Instant::elapsed()` so that
    /// expiry survives process restarts correctly.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        let Some(ttl) = self.ttl_secs else {
            return false;
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.created_at_secs) >= ttl
    }

    /// Estimate the in-memory / on-disk footprint of this entry in bytes.
    #[must_use]
    pub fn estimated_size(&self) -> usize {
        self.key.len()
            + self.fingerprint_summary.len()
            + self.kv_data.len() * std::mem::size_of::<f32>()
            + 64 // fixed-size field overhead estimate
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`RedbPrefixCache`] instance.
#[derive(Debug, Clone)]
pub struct RedbPrefixCacheConfig {
    /// Path to the redb database file.
    pub path: PathBuf,
    /// Core cache behaviour settings (capacity, TTL defaults, etc.).
    pub cache_config: PrefixCacheConfig,
    /// Default time-to-live in seconds applied to entries that have no
    /// per-entry TTL.  Zero means "never expires by default".
    pub ttl_secs: u64,
}

impl Default for RedbPrefixCacheConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./oxirag_prefix_cache.redb"),
            cache_config: PrefixCacheConfig::default(),
            ttl_secs: 3600,
        }
    }
}

impl RedbPrefixCacheConfig {
    /// Create a new configuration pointing to `path` with default settings.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    /// Override the per-entry default TTL.
    #[must_use]
    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = ttl_secs;
        self
    }

    /// Override the core cache configuration.
    #[must_use]
    pub fn with_cache_config(mut self, config: PrefixCacheConfig) -> Self {
        self.cache_config = config;
        self
    }
}

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
    db: Database,
    /// Cache behaviour configuration.
    config: PrefixCacheConfig,
    /// Running statistics.
    stats: CacheStats,
    /// Default TTL (seconds) applied when an entry has no per-entry TTL.
    /// Zero means entries never expire by default.
    ttl_secs: u64,
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
        let (entry_count, total_bytes) = Self::scan_counts(&db)?;
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
    fn fp_key(fp: &ContextFingerprint) -> String {
        format!("{}:{}", fp.hash, fp.prefix_length)
    }

    /// Count the live (non-expired) entries and accumulate their estimated sizes.
    fn scan_counts(db: &Database) -> Result<(usize, usize), OxiRagError> {
        let read_txn = db
            .begin_read()
            .map_err(|e| OxiRagError::Config(format!("redb begin_read failed: {e}")))?;
        let table = read_txn
            .open_table(CACHE_TABLE)
            .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;

        let mut count = 0usize;
        let mut bytes = 0usize;

        let iter = table
            .iter()
            .map_err(|e| OxiRagError::Config(format!("redb iter failed: {e}")))?;

        for result in iter {
            let (_key, value) = result
                .map_err(|e| OxiRagError::Config(format!("redb iter item failed: {e}")))?;
            if let Ok(entry) = Self::decode_entry_static(value.value())
                && !entry.is_expired()
            {
                count += 1;
                bytes += entry.estimated_size();
            }
        }

        Ok((count, bytes))
    }

    /// Static variant of `decode_entry` for use in associated functions.
    fn decode_entry_static(bytes: &[u8]) -> Result<PersistedKVEntry, OxiRagError> {
        serde_json::from_slice(bytes)
            .map_err(|e| OxiRagError::Config(format!("redb deserialise failed: {e}")))
    }

    /// Serialise a [`PersistedKVEntry`] to bytes for storage.
    fn encode_entry(entry: &PersistedKVEntry) -> Result<Vec<u8>, OxiRagError> {
        serde_json::to_vec(entry)
            .map_err(|e| OxiRagError::Config(format!("redb serialise failed: {e}")))
    }

    /// Return the key of the oldest entry (by `created_at_secs`) found in a
    /// full table scan.  Returns `None` if the table is empty.
    fn find_oldest_key(&self) -> Result<Option<String>, OxiRagError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| OxiRagError::Config(format!("redb begin_read failed: {e}")))?;
        let table = read_txn
            .open_table(CACHE_TABLE)
            .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;

        let iter = table
            .iter()
            .map_err(|e| OxiRagError::Config(format!("redb iter failed: {e}")))?;

        let mut oldest_key: Option<String> = None;
        let mut oldest_ts = u64::MAX;

        for result in iter {
            let (key, value) = result
                .map_err(|e| OxiRagError::Config(format!("redb iter item failed: {e}")))?;
            if let Ok(entry) = Self::decode_entry_static(value.value())
                && entry.created_at_secs < oldest_ts
            {
                oldest_ts = entry.created_at_secs;
                oldest_key = Some(key.value().to_owned());
            }
        }

        Ok(oldest_key)
    }

    /// Delete a single entry by its composite key, adjusting stats.
    fn delete_by_raw_key(
        &mut self,
        raw_key: &str,
    ) -> Result<Option<PersistedKVEntry>, OxiRagError> {
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| OxiRagError::Config(format!("redb begin_write failed: {e}")))?;
        let removed = {
            let mut table = write_txn
                .open_table(CACHE_TABLE)
                .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;

            let existing = table
                .remove(raw_key)
                .map_err(|e| OxiRagError::Config(format!("redb remove failed: {e}")))?;

            existing.map(|guard| Self::decode_entry_static(guard.value()).ok())
        };
        write_txn
            .commit()
            .map_err(|e| OxiRagError::Config(format!("redb commit failed: {e}")))?;

        Ok(removed.flatten())
    }

    /// Rebuild `stats.total_bytes` and `stats.entry_count` from a full scan.
    ///
    /// Called after bulk operations (`clear`, `evict_expired`) that make
    /// incremental tracking impractical.
    fn rebuild_stats(&mut self) -> Result<(), OxiRagError> {
        let (count, bytes) = Self::scan_counts(&self.db)?;
        self.stats.update_memory(bytes, count);
        Ok(())
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
        let persisted = Self::decode_entry_static(guard.value()).ok()?;
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
            && let Some(oldest_key) = self.find_oldest_key()?
            && let Some(old_entry) = self.delete_by_raw_key(&oldest_key)?
        {
            let old_size = old_entry.estimated_size();
            self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(old_size);
            self.stats.record_eviction();
        }

        let bytes = Self::encode_entry(&persisted)?;

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
                .and_then(|g| Self::decode_entry_static(g.value()).ok())
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
        if let Ok(Some(persisted)) = self.delete_by_raw_key(key) {
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
                if let Ok(entry) = Self::decode_entry_static(value.value())
                    && &entry.key == key
                {
                    found = Some(raw_key.value().to_owned());
                    break;
                }
            }
            found
        };

        if let Some(raw_key) = matching_raw_key
            && let Ok(Some(persisted)) = self.delete_by_raw_key(&raw_key)
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
        Self::decode_entry_static(guard.value())
            .is_ok_and(|e| !e.is_expired())
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
            let Ok(entry) = Self::decode_entry_static(value.value()) else {
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
                    Self::decode_entry_static(v.value())
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
        let _ = self.rebuild_stats();

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::prefix_cache::types::{ContextFingerprint, KVCacheEntry, PrefixCacheConfig};
    use crate::prefix_cache::traits::PrefixCacheStore;

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
        assert!(cache.contains(&fp).await, "entry must exist for second remove");

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
            let fp = ContextFingerprint::new(i, 100 + i as usize, format!("exp {i}"));
            let entry = KVCacheEntry::new(format!("exp_{i}"), fp, vec![0.0; 4], 100)
                .with_ttl(Duration::from_secs(0));
            cache.put(entry).await.expect("put should succeed");
        }
        for i in 10_u64..12 {
            let fp = ContextFingerprint::new(i, 200 + i as usize, format!("live {i}"));
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
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::pedantic,
)]
mod prop_tests {
    use std::time::Duration;

    use proptest::prelude::*;

    use super::*;
    use crate::prefix_cache::traits::PrefixCacheStore;
    use crate::prefix_cache::types::{ContextFingerprint, KVCacheEntry, PrefixCacheConfig};

    // -----------------------------------------------------------------------
    // Strategies
    // -----------------------------------------------------------------------

    /// Strategy: generate an arbitrary [`ContextFingerprint`].
    fn arb_fingerprint() -> impl Strategy<Value = ContextFingerprint> {
        (any::<u64>(), 1usize..1000usize, "[a-z]{1,20}").prop_map(
            |(hash, len, summary)| ContextFingerprint::new(hash, len, summary),
        )
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
