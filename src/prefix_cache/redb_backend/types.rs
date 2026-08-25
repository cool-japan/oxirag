//! Types and serialisable structs for the redb prefix-cache backend.

use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use super::super::types::{ContextFingerprint, KVCacheEntry, PrefixCacheConfig};

// ---------------------------------------------------------------------------
// Table definition
// ---------------------------------------------------------------------------

/// The single redb table that backs the prefix cache.
///
/// Keys are composite strings `"<fingerprint_hash>:<prefix_length>"`.
/// Values are JSON-encoded [`PersistedKVEntry`] blobs.
pub const CACHE_TABLE: redb::TableDefinition<&str, &[u8]> =
    redb::TableDefinition::new("prefix_cache");

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
    /// `crate::time::system_now()` because `Instant` has no stable relationship to
    /// wall-clock time that can survive a process restart.
    #[must_use]
    pub fn from_kv_entry(entry: &KVCacheEntry) -> Self {
        let now_secs = crate::time::system_now()
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
    /// `created_at` and `last_accessed` are set to [`crate::time::Instant::now()`],
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
        let now = crate::time::system_now()
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

/// Configuration for a [`super::store::RedbPrefixCache`] instance.
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
