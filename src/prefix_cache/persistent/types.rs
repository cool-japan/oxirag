//! Types and configuration for the persistent prefix-cache backend.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use super::super::types::{ContextFingerprint, KVCacheEntry};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for persistent cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentCacheConfig {
    /// Path to the storage directory.
    pub storage_path: PathBuf,
    /// Maximum file size in bytes before rotation.
    pub max_file_size_bytes: usize,
    /// Interval in seconds between automatic syncs.
    pub sync_interval_secs: u64,
    /// Whether to enable compression for stored data.
    pub compression_enabled: bool,
    /// Whether to keep the index in memory for faster lookups.
    pub index_in_memory: bool,
}

impl Default for PersistentCacheConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./prefix_cache_data"),
            max_file_size_bytes: 256 * 1024 * 1024, // 256 MB
            sync_interval_secs: 60,
            compression_enabled: false,
            index_in_memory: true,
        }
    }
}

impl PersistentCacheConfig {
    /// Create a new persistent cache configuration.
    #[must_use]
    pub fn new(storage_path: impl Into<PathBuf>) -> Self {
        Self {
            storage_path: storage_path.into(),
            ..Default::default()
        }
    }

    /// Set the maximum file size.
    #[must_use]
    pub fn with_max_file_size(mut self, bytes: usize) -> Self {
        self.max_file_size_bytes = bytes;
        self
    }

    /// Set the sync interval.
    #[must_use]
    pub fn with_sync_interval(mut self, secs: u64) -> Self {
        self.sync_interval_secs = secs;
        self
    }

    /// Enable or disable compression.
    #[must_use]
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    /// Enable or disable in-memory index.
    #[must_use]
    pub fn with_memory_index(mut self, enabled: bool) -> Self {
        self.index_in_memory = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// Serialisable entry
// ---------------------------------------------------------------------------

/// Serializable cache entry for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEntry {
    /// The cache key.
    pub key: String,
    /// Hash of the fingerprint.
    pub fingerprint_hash: u64,
    /// Prefix length from the fingerprint.
    pub fingerprint_prefix_length: usize,
    /// Summary from the fingerprint.
    pub fingerprint_summary: String,
    /// The KV data (stored as `Vec<f32>`).
    pub kv_data: Vec<f32>,
    /// Sequence length.
    pub sequence_length: usize,
    /// Unix timestamp when created.
    pub created_at_unix: u64,
    /// Optional TTL in seconds.
    pub ttl_secs: Option<u64>,
    /// Access count.
    pub access_count: u64,
}

impl PersistedEntry {
    /// Create a new persisted entry from a KV cache entry.
    #[must_use]
    pub fn from_kv_entry(entry: &KVCacheEntry) -> Self {
        let created_at_unix = crate::time::system_now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        Self {
            key: entry.key.clone(),
            fingerprint_hash: entry.fingerprint.hash,
            fingerprint_prefix_length: entry.fingerprint.prefix_length,
            fingerprint_summary: entry.fingerprint.content_summary.clone(),
            kv_data: entry.kv_data.clone(),
            sequence_length: entry.sequence_length,
            created_at_unix,
            ttl_secs: entry.ttl.map(|d| d.as_secs()),
            access_count: entry.access_count,
        }
    }

    /// Convert back to a KV cache entry.
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

        entry
    }

    /// Check if this entry has expired based on TTL.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if let Some(ttl_secs) = self.ttl_secs {
            let now = crate::time::system_now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            now.saturating_sub(self.created_at_unix) >= ttl_secs
        } else {
            false
        }
    }

    /// Estimate the serialized size of this entry.
    #[must_use]
    pub fn estimated_size(&self) -> usize {
        // Rough estimate: key + summary + kv_data + fixed overhead
        self.key.len()
            + self.fingerprint_summary.len()
            + self.kv_data.len() * std::mem::size_of::<f32>()
            + 100 // Fixed overhead for other fields
    }
}

// ---------------------------------------------------------------------------
// Index types
// ---------------------------------------------------------------------------

/// Index entry for quick lookup without reading full data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Hash of the fingerprint for matching.
    pub fingerprint_hash: u64,
    /// Offset in the data file.
    pub file_offset: u64,
    /// Size of the entry in bytes.
    pub entry_size: usize,
    /// Unix timestamp when created.
    pub created_at_unix: u64,
    /// Optional TTL in seconds.
    pub ttl_secs: Option<u64>,
}

impl IndexEntry {
    /// Check if this entry has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if let Some(ttl_secs) = self.ttl_secs {
            let now = crate::time::system_now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            now.saturating_sub(self.created_at_unix) >= ttl_secs
        } else {
            false
        }
    }
}

/// Persistent cache index for tracking all entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheIndex {
    /// Map from cache key to index entry.
    pub entries: HashMap<String, IndexEntry>,
    /// Map from fingerprint hash to cache key.
    pub fingerprint_to_key: HashMap<u64, String>,
    /// Total size of all entries in bytes.
    pub total_size_bytes: usize,
    /// Number of entries.
    pub entry_count: usize,
    /// Unix timestamp of last compaction.
    pub last_compaction: Option<u64>,
    /// Version number for compatibility.
    pub version: u32,
}

impl CacheIndex {
    /// Create a new empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            fingerprint_to_key: HashMap::new(),
            total_size_bytes: 0,
            entry_count: 0,
            last_compaction: None,
            version: 1,
        }
    }

    /// Add an entry to the index.
    pub fn add(&mut self, key: String, entry: IndexEntry) {
        self.total_size_bytes += entry.entry_size;
        self.fingerprint_to_key
            .insert(entry.fingerprint_hash, key.clone());
        self.entries.insert(key, entry);
        self.entry_count = self.entries.len();
    }

    /// Remove an entry from the index.
    pub fn remove(&mut self, key: &str) -> Option<IndexEntry> {
        if let Some(entry) = self.entries.remove(key) {
            self.fingerprint_to_key.remove(&entry.fingerprint_hash);
            self.total_size_bytes = self.total_size_bytes.saturating_sub(entry.entry_size);
            self.entry_count = self.entries.len();
            Some(entry)
        } else {
            None
        }
    }

    /// Get a key by fingerprint hash.
    #[must_use]
    pub fn get_key_by_hash(&self, hash: u64) -> Option<&String> {
        self.fingerprint_to_key.get(&hash)
    }

    /// Get an index entry by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&IndexEntry> {
        self.entries.get(key)
    }

    /// Check if a fingerprint hash exists.
    #[must_use]
    pub fn contains_hash(&self, hash: u64) -> bool {
        self.fingerprint_to_key.contains_key(&hash)
    }
}

/// Statistics from a compaction operation.
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    /// Number of entries removed during compaction.
    pub entries_removed: usize,
    /// Bytes reclaimed during compaction.
    pub bytes_reclaimed: usize,
    /// Duration of compaction in milliseconds.
    pub duration_ms: u64,
}
