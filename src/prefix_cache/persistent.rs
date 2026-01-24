//! Persistent storage backend for prefix cache.
//!
//! This module provides a file-based persistent cache that can serve as L2/L3 storage,
//! implementing disk-based persistence for KV cache entries with indexing, compaction,
//! and hybrid memory+disk caching strategies.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::store::InMemoryPrefixCache;
use super::traits::PrefixCacheStore;
use super::types::{CacheKey, CacheStats, ContextFingerprint, KVCacheEntry, PrefixCacheConfig};
use crate::error::OxiRagError;

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
        let created_at_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

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
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
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
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
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

/// File-based persistent prefix cache.
///
/// This implementation stores cache entries in a data file with a separate
/// index file for fast lookups. It supports:
/// - Append-only writes for durability
/// - In-memory index for fast lookups
/// - Compaction to reclaim space from deleted entries
/// - Automatic expiration of TTL-based entries
pub struct PersistentPrefixCache {
    /// Configuration for this cache.
    config: PersistentCacheConfig,
    /// The cache index.
    index: Arc<RwLock<CacheIndex>>,
    /// Path to the data file.
    data_file: PathBuf,
    /// Path to the index file.
    index_file: PathBuf,
    /// Whether the index has uncommitted changes.
    dirty: Arc<RwLock<bool>>,
    /// Cache statistics.
    stats: Arc<RwLock<CacheStats>>,
    /// Next key ID for generation.
    next_key_id: Arc<RwLock<u64>>,
}

impl std::fmt::Debug for PersistentPrefixCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentPrefixCache")
            .field("config", &self.config)
            .field("data_file", &self.data_file)
            .field("index_file", &self.index_file)
            .finish_non_exhaustive()
    }
}

impl PersistentPrefixCache {
    /// Create or open a persistent cache at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage directory cannot be created or
    /// if existing index/data files cannot be loaded.
    pub fn open(config: PersistentCacheConfig) -> Result<Self, OxiRagError> {
        // Create storage directory if it doesn't exist
        fs::create_dir_all(&config.storage_path)?;

        let data_file = config.storage_path.join("cache_data.bin");
        let index_file = config.storage_path.join("cache_index.json");

        let mut cache = Self {
            config,
            index: Arc::new(RwLock::new(CacheIndex::new())),
            data_file,
            index_file,
            dirty: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            next_key_id: Arc::new(RwLock::new(0)),
        };

        // Load existing index if present
        cache.load_index()?;

        Ok(cache)
    }

    /// Load index from disk.
    fn load_index(&mut self) -> Result<(), OxiRagError> {
        if !self.index_file.exists() {
            return Ok(());
        }

        let file = File::open(&self.index_file)?;
        let reader = BufReader::new(file);
        let index: CacheIndex = serde_json::from_reader(reader)?;

        // Find the highest key ID from existing entries
        let max_key_id = index
            .entries
            .keys()
            .filter_map(|k| k.strip_prefix("pc_"))
            .filter_map(|s| s.parse::<u64>().ok())
            .max()
            .unwrap_or(0);

        *self.next_key_id.write().expect("lock poisoned") = max_key_id + 1;
        *self.index.write().expect("lock poisoned") = index;

        Ok(())
    }

    /// Save index to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the index file cannot be written.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    pub fn save_index(&self) -> Result<(), OxiRagError> {
        let index = self.index.read().expect("lock poisoned");
        let file = File::create(&self.index_file)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &*index)?;
        *self.dirty.write().expect("lock poisoned") = false;
        Ok(())
    }

    /// Append entry to data file and return the offset.
    fn append_entry(&mut self, entry: &PersistedEntry) -> Result<(u64, usize), OxiRagError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.data_file)?;

        let offset = file.seek(SeekFrom::End(0))?;

        // Serialize entry to JSON with newline delimiter
        let data = serde_json::to_vec(entry)?;
        let size = data.len();

        // Write length prefix for easier reading
        let len_bytes = (size as u64).to_le_bytes();
        file.write_all(&len_bytes)?;
        file.write_all(&data)?;

        *self.dirty.write().expect("lock poisoned") = true;

        Ok((offset, size + 8)) // +8 for length prefix
    }

    /// Read entry from data file at offset.
    #[allow(clippy::cast_possible_truncation)]
    fn read_entry(&self, offset: u64) -> Result<PersistedEntry, OxiRagError> {
        let mut file = File::open(&self.data_file)?;
        file.seek(SeekFrom::Start(offset))?;

        // Read length prefix
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        // Note: On 32-bit systems this could truncate, but we're targeting 64-bit
        let size = u64::from_le_bytes(len_bytes) as usize;

        // Read entry data
        let mut data = vec![0u8; size];
        file.read_exact(&mut data)?;

        let entry: PersistedEntry = serde_json::from_slice(&data)?;
        Ok(entry)
    }

    /// Compact the data file by removing deleted/expired entries.
    ///
    /// # Errors
    ///
    /// Returns an error if compaction fails due to I/O issues.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[allow(clippy::cast_possible_truncation)]
    pub fn compact(&mut self) -> Result<CompactionStats, OxiRagError> {
        let start = std::time::Instant::now();

        let mut stats = CompactionStats::default();
        let mut index = self.index.write().expect("lock poisoned");

        // Find expired entries
        let expired_keys: Vec<String> = index
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in &expired_keys {
            if let Some(entry) = index.remove(key) {
                stats.entries_removed += 1;
                stats.bytes_reclaimed += entry.entry_size;
            }
        }

        // If we have entries, rewrite the data file
        if stats.entries_removed > 0 && !index.entries.is_empty() {
            drop(index); // Release lock before I/O

            // Read all valid entries
            let mut valid_entries = Vec::new();
            let index = self.index.read().expect("lock poisoned");
            for (key, idx_entry) in &index.entries {
                if let Ok(entry) = self.read_entry(idx_entry.file_offset) {
                    valid_entries.push((key.clone(), entry));
                }
            }
            drop(index);

            // Create new data file
            let temp_data = self.data_file.with_extension("tmp");
            {
                let mut file = File::create(&temp_data)?;

                let mut new_index = CacheIndex::new();
                for (key, entry) in valid_entries {
                    let offset = file.stream_position()?;
                    let data = serde_json::to_vec(&entry)?;
                    let size = data.len();

                    let len_bytes = (size as u64).to_le_bytes();
                    file.write_all(&len_bytes)?;
                    file.write_all(&data)?;

                    let idx_entry = IndexEntry {
                        fingerprint_hash: entry.fingerprint_hash,
                        file_offset: offset,
                        entry_size: size + 8,
                        created_at_unix: entry.created_at_unix,
                        ttl_secs: entry.ttl_secs,
                    };
                    new_index.add(key, idx_entry);
                }

                new_index.last_compaction = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                );

                *self.index.write().expect("lock poisoned") = new_index;
            }

            // Atomic rename
            fs::rename(&temp_data, &self.data_file)?;
            self.save_index()?;
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;
        Ok(stats)
    }

    /// Sync all pending changes to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    pub fn sync(&self) -> Result<(), OxiRagError> {
        let dirty = *self.dirty.read().expect("lock poisoned");
        if dirty {
            self.save_index()?;
        }
        Ok(())
    }

    /// Generate a unique cache key.
    fn generate_key(&self) -> CacheKey {
        let mut next_id = self.next_key_id.write().expect("lock poisoned");
        let key = format!("pc_{}", *next_id);
        *next_id += 1;
        key
    }

    /// Get the number of entries in the cache.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.index.read().expect("lock poisoned").entry_count
    }
}

impl Clone for PersistentPrefixCache {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            index: Arc::clone(&self.index),
            data_file: self.data_file.clone(),
            index_file: self.index_file.clone(),
            dirty: Arc::clone(&self.dirty),
            stats: Arc::clone(&self.stats),
            next_key_id: Arc::clone(&self.next_key_id),
        }
    }
}

#[async_trait]
impl PrefixCacheStore for PersistentPrefixCache {
    async fn get(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        let index = self.index.read().expect("lock poisoned");

        // Look up by fingerprint hash
        let key = index.get_key_by_hash(fingerprint.hash)?;
        let idx_entry = index.get(key)?;

        // Check expiration
        if idx_entry.is_expired() {
            let mut stats = self.stats.write().expect("lock poisoned");
            stats.record_miss();
            stats.record_expiration();
            return None;
        }

        let offset = idx_entry.file_offset;
        drop(index); // Release lock before I/O

        // Read from disk
        if let Ok(persisted) = self.read_entry(offset) {
            let mut stats = self.stats.write().expect("lock poisoned");
            stats.record_hit();
            Some(persisted.to_kv_entry())
        } else {
            let mut stats = self.stats.write().expect("lock poisoned");
            stats.record_miss();
            None
        }
    }

    async fn put(&mut self, entry: KVCacheEntry) -> Result<CacheKey, OxiRagError> {
        let persisted = PersistedEntry::from_kv_entry(&entry);

        // Generate key if needed
        let key = if entry.key.is_empty() {
            self.generate_key()
        } else {
            entry.key.clone()
        };

        // Remove existing entry with same fingerprint
        {
            let mut index = self.index.write().expect("lock poisoned");
            if let Some(old_key) = index.get_key_by_hash(persisted.fingerprint_hash).cloned() {
                index.remove(&old_key);
            }
        }

        // Append to data file
        let (offset, size) = self.append_entry(&persisted)?;

        // Update index
        let idx_entry = IndexEntry {
            fingerprint_hash: persisted.fingerprint_hash,
            file_offset: offset,
            entry_size: size,
            created_at_unix: persisted.created_at_unix,
            ttl_secs: persisted.ttl_secs,
        };

        {
            let mut index = self.index.write().expect("lock poisoned");
            index.add(key.clone(), idx_entry);
        }

        // Update stats
        {
            let index = self.index.read().expect("lock poisoned");
            let mut stats = self.stats.write().expect("lock poisoned");
            stats.update_memory(index.total_size_bytes, index.entry_count);
        }

        Ok(key)
    }

    async fn remove(&mut self, key: &CacheKey) -> Option<KVCacheEntry> {
        let mut index = self.index.write().expect("lock poisoned");

        if let Some(idx_entry) = index.remove(key) {
            let offset = idx_entry.file_offset;
            drop(index);

            // Try to read the entry before it's "removed" (marked as deleted)
            if let Ok(persisted) = self.read_entry(offset) {
                *self.dirty.write().expect("lock poisoned") = true;
                return Some(persisted.to_kv_entry());
            }
        }
        None
    }

    async fn contains(&self, fingerprint: &ContextFingerprint) -> bool {
        let index = self.index.read().expect("lock poisoned");
        if let Some(key) = index.get_key_by_hash(fingerprint.hash)
            && let Some(entry) = index.get(key)
        {
            return !entry.is_expired();
        }
        false
    }

    async fn clear(&mut self) {
        *self.index.write().expect("lock poisoned") = CacheIndex::new();
        *self.dirty.write().expect("lock poisoned") = true;

        // Remove data file
        let _ = fs::remove_file(&self.data_file);

        self.save_index().ok();
    }

    fn stats(&self) -> CacheStats {
        self.stats.read().expect("lock poisoned").clone()
    }

    fn len(&self) -> usize {
        self.index.read().expect("lock poisoned").entry_count
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    async fn evict_expired(&mut self) -> usize {
        let mut index = self.index.write().expect("lock poisoned");

        let expired_keys: Vec<String> = index
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            index.remove(&key);
            let mut stats = self.stats.write().expect("lock poisoned");
            stats.record_expiration();
        }

        if count > 0 {
            *self.dirty.write().expect("lock poisoned") = true;
        }

        count
    }

    fn memory_usage(&self) -> usize {
        self.index.read().expect("lock poisoned").total_size_bytes
    }
}

/// Hybrid cache combining memory and persistent storage.
///
/// This cache uses an in-memory cache as L1 and a persistent cache as L2,
/// providing fast access for hot data while persisting all entries to disk.
pub struct HybridPersistentCache {
    /// In-memory L1 cache.
    memory_cache: InMemoryPrefixCache,
    /// Persistent L2 cache.
    persistent_cache: PersistentPrefixCache,
    /// Whether to write to both caches immediately.
    write_through: bool,
    /// Whether to check persistent cache on memory miss.
    read_through: bool,
}

impl std::fmt::Debug for HybridPersistentCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridPersistentCache")
            .field("memory_cache", &self.memory_cache)
            .field("persistent_cache", &self.persistent_cache)
            .field("write_through", &self.write_through)
            .field("read_through", &self.read_through)
            .finish()
    }
}

impl HybridPersistentCache {
    /// Create a new hybrid cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the persistent cache cannot be opened.
    pub fn new(
        memory_config: PrefixCacheConfig,
        persistent_config: PersistentCacheConfig,
    ) -> Result<Self, OxiRagError> {
        Ok(Self {
            memory_cache: InMemoryPrefixCache::new(memory_config),
            persistent_cache: PersistentPrefixCache::open(persistent_config)?,
            write_through: true,
            read_through: true,
        })
    }

    /// Create with default configurations.
    ///
    /// # Errors
    ///
    /// Returns an error if the persistent cache cannot be opened.
    pub fn with_defaults() -> Result<Self, OxiRagError> {
        Self::new(
            PrefixCacheConfig::default(),
            PersistentCacheConfig::default(),
        )
    }

    /// Set write-through mode.
    #[must_use]
    pub fn with_write_through(mut self, enabled: bool) -> Self {
        self.write_through = enabled;
        self
    }

    /// Set read-through mode.
    #[must_use]
    pub fn with_read_through(mut self, enabled: bool) -> Self {
        self.read_through = enabled;
        self
    }

    /// Flush memory cache to persistent storage.
    ///
    /// # Errors
    ///
    /// Returns an error if entries cannot be persisted.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    pub async fn flush_to_disk(&mut self) -> Result<usize, OxiRagError> {
        let mut count = 0;

        // Get all entries from memory cache - collect before async operations
        let entries: Vec<KVCacheEntry> = {
            let inner = self.memory_cache.inner.read().expect("lock poisoned");
            inner.entries.values().cloned().collect()
        };

        for entry in entries {
            // Check if already in persistent cache
            if !self.persistent_cache.contains(&entry.fingerprint).await {
                self.persistent_cache.put(entry).await?;
                count += 1;
            }
        }

        self.persistent_cache.sync()?;
        Ok(count)
    }

    /// Load entries from persistent storage to memory cache.
    ///
    /// # Errors
    ///
    /// Returns an error if entries cannot be loaded.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    pub async fn warm_cache(&mut self, count: usize) -> Result<usize, OxiRagError> {
        let mut loaded = 0;

        // Get keys from persistent index - collect before async operations
        let keys: Vec<(String, u64)> = {
            let index = self.persistent_cache.index.read().expect("lock poisoned");
            index
                .entries
                .iter()
                .filter(|(_, e)| !e.is_expired())
                .take(count)
                .map(|(k, e)| (k.clone(), e.file_offset))
                .collect()
        };

        for (_, offset) in keys {
            if loaded >= count {
                break;
            }

            if let Ok(persisted) = self.persistent_cache.read_entry(offset) {
                let entry = persisted.to_kv_entry();
                if !self.memory_cache.contains(&entry.fingerprint).await {
                    self.memory_cache.put(entry).await?;
                    loaded += 1;
                }
            }
        }

        Ok(loaded)
    }

    /// Get statistics for both caches.
    #[must_use]
    pub fn combined_stats(&self) -> (CacheStats, CacheStats) {
        (self.memory_cache.stats(), self.persistent_cache.stats())
    }

    /// Sync persistent cache to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if sync fails.
    pub fn sync(&self) -> Result<(), OxiRagError> {
        self.persistent_cache.sync()
    }

    /// Compact the persistent cache.
    ///
    /// # Errors
    ///
    /// Returns an error if compaction fails.
    pub fn compact(&mut self) -> Result<CompactionStats, OxiRagError> {
        self.persistent_cache.compact()
    }
}

impl Clone for HybridPersistentCache {
    fn clone(&self) -> Self {
        Self {
            memory_cache: self.memory_cache.clone(),
            persistent_cache: self.persistent_cache.clone(),
            write_through: self.write_through,
            read_through: self.read_through,
        }
    }
}

#[async_trait]
impl PrefixCacheStore for HybridPersistentCache {
    async fn get(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        // Try memory cache first
        if let Some(entry) = self.memory_cache.get(fingerprint).await {
            return Some(entry);
        }

        // Try persistent cache if read-through is enabled
        if self.read_through
            && let Some(entry) = self.persistent_cache.get(fingerprint).await
        {
            // Promote to memory cache (best effort)
            // Note: We can't modify self here, so promotion would need
            // to be handled separately or use interior mutability
            return Some(entry);
        }

        None
    }

    async fn put(&mut self, entry: KVCacheEntry) -> Result<CacheKey, OxiRagError> {
        // Always put in memory cache
        let key = self.memory_cache.put(entry.clone()).await?;

        // Write through to persistent cache if enabled
        if self.write_through {
            self.persistent_cache.put(entry).await?;
        }

        Ok(key)
    }

    async fn remove(&mut self, key: &CacheKey) -> Option<KVCacheEntry> {
        // Remove from both caches
        let memory_entry = self.memory_cache.remove(key).await;
        let persistent_entry = self.persistent_cache.remove(key).await;

        // Return memory entry if available, otherwise persistent
        memory_entry.or(persistent_entry)
    }

    async fn contains(&self, fingerprint: &ContextFingerprint) -> bool {
        self.memory_cache.contains(fingerprint).await
            || (self.read_through && self.persistent_cache.contains(fingerprint).await)
    }

    async fn clear(&mut self) {
        self.memory_cache.clear().await;
        self.persistent_cache.clear().await;
    }

    fn stats(&self) -> CacheStats {
        // Return memory cache stats (primary)
        self.memory_cache.stats()
    }

    fn len(&self) -> usize {
        // Return total unique entries (approximate - memory + disk-only)
        self.memory_cache.len() + self.persistent_cache.len()
    }

    fn is_empty(&self) -> bool {
        self.memory_cache.is_empty() && self.persistent_cache.is_empty()
    }

    async fn evict_expired(&mut self) -> usize {
        let memory_evicted = self.memory_cache.evict_expired().await;
        let persistent_evicted = self.persistent_cache.evict_expired().await;
        memory_evicted + persistent_evicted
    }

    fn memory_usage(&self) -> usize {
        self.memory_cache.memory_usage() + self.persistent_cache.memory_usage()
    }
}

#[cfg(test)]
#[allow(clippy::cast_sign_loss)]
mod tests {
    use super::*;
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

        let cache = cache.unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // Test 2: Entry persistence and retrieval
    #[tokio::test]
    async fn test_persistent_cache_put_and_get() {
        let (_temp_dir, config) = create_temp_config();
        let mut cache = PersistentPrefixCache::open(config).unwrap();

        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        let key = cache.put(entry).await.unwrap();
        assert!(!key.is_empty());

        let retrieved = cache.get(&fingerprint).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().fingerprint.hash, 12345);
    }

    // Test 3: Index operations - save and load
    #[tokio::test]
    async fn test_persistent_cache_index_persistence() {
        let (temp_dir, config) = create_temp_config();

        // Create cache and add entry
        {
            let mut cache = PersistentPrefixCache::open(config.clone()).unwrap();
            let entry = create_test_entry("test1", 12345, 10);
            cache.put(entry).await.unwrap();
            cache.save_index().unwrap();
        }

        // Reopen and verify
        {
            let cache =
                PersistentPrefixCache::open(PersistentCacheConfig::new(temp_dir.path())).unwrap();
            let fp = ContextFingerprint::new(12345, 100, "test test1");
            assert!(cache.contains(&fp).await);
        }
    }

    // Test 4: Multiple entries
    #[tokio::test]
    async fn test_persistent_cache_multiple_entries() {
        let (_temp_dir, config) = create_temp_config();
        let mut cache = PersistentPrefixCache::open(config).unwrap();

        for i in 0..5 {
            let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
            cache.put(entry).await.unwrap();
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
        let mut cache = PersistentPrefixCache::open(config).unwrap();

        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();
        let key = cache.put(entry).await.unwrap();

        assert!(cache.contains(&fingerprint).await);

        let removed = cache.remove(&key).await;
        assert!(removed.is_some());
        assert!(!cache.contains(&fingerprint).await);
    }

    // Test 6: Clear cache
    #[tokio::test]
    async fn test_persistent_cache_clear() {
        let (_temp_dir, config) = create_temp_config();
        let mut cache = PersistentPrefixCache::open(config).unwrap();

        for i in 0..5 {
            let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
            cache.put(entry).await.unwrap();
        }

        assert_eq!(cache.len(), 5);

        cache.clear().await;
        assert!(cache.is_empty());
    }

    // Test 7: Compaction
    #[tokio::test]
    async fn test_persistent_cache_compaction() {
        let (_temp_dir, config) = create_temp_config();
        let mut cache = PersistentPrefixCache::open(config).unwrap();

        // Add entries with immediate expiration
        for i in 0..5 {
            let fp = ContextFingerprint::new(i as u64, 100, format!("test {i}"));
            let entry = KVCacheEntry::new(format!("test{i}"), fp, vec![0.0; 10], 100)
                .with_ttl(Duration::from_secs(0));
            cache.put(entry).await.unwrap();
        }

        std::thread::sleep(Duration::from_millis(10));

        let stats = cache.compact().unwrap();
        assert_eq!(stats.entries_removed, 5);
    }

    // Test 8: TTL expiration
    #[tokio::test]
    async fn test_persistent_cache_ttl_expiration() {
        let (_temp_dir, config) = create_temp_config();
        let mut cache = PersistentPrefixCache::open(config).unwrap();

        let fp = ContextFingerprint::new(12345, 100, "test");
        let entry = KVCacheEntry::new("test1", fp.clone(), vec![0.0; 10], 100)
            .with_ttl(Duration::from_secs(0));

        cache.put(entry).await.unwrap();

        std::thread::sleep(Duration::from_millis(10));
        let result = cache.get(&fp).await;
        assert!(result.is_none());
    }

    // Test 9: Evict expired entries
    #[tokio::test]
    async fn test_persistent_cache_evict_expired() {
        let (_temp_dir, config) = create_temp_config();
        let mut cache = PersistentPrefixCache::open(config).unwrap();

        for i in 0..5 {
            let fp = ContextFingerprint::new(i as u64, 100, format!("test {i}"));
            let entry = KVCacheEntry::new(format!("test{i}"), fp, vec![0.0; 10], 100)
                .with_ttl(Duration::from_secs(0));
            cache.put(entry).await.unwrap();
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

        let cache = cache.unwrap();
        assert!(cache.is_empty());
    }

    // Test 11: Hybrid cache put and get
    #[tokio::test]
    async fn test_hybrid_cache_put_and_get() {
        let (_temp_dir, persistent_config) = create_temp_config();
        let memory_config = PrefixCacheConfig::default();

        let mut cache = HybridPersistentCache::new(memory_config, persistent_config).unwrap();

        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        cache.put(entry).await.unwrap();

        let retrieved = cache.get(&fingerprint).await;
        assert!(retrieved.is_some());
    }

    // Test 12: Hybrid cache write-through
    #[tokio::test]
    async fn test_hybrid_cache_write_through() {
        let (_temp_dir, persistent_config) = create_temp_config();
        let memory_config = PrefixCacheConfig::default();

        let mut cache = HybridPersistentCache::new(memory_config, persistent_config)
            .unwrap()
            .with_write_through(true);

        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        cache.put(entry).await.unwrap();

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
            .unwrap()
            .with_write_through(false); // Disable write-through

        // Add entries only to memory
        for i in 0..5 {
            let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
            cache.memory_cache.put(entry).await.unwrap();
        }

        assert_eq!(cache.memory_cache.len(), 5);
        assert_eq!(cache.persistent_cache.len(), 0);

        // Flush to disk
        let flushed = cache.flush_to_disk().await.unwrap();
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
            let mut cache = HybridPersistentCache::new(memory_config, persistent_config).unwrap();

            for i in 0..5 {
                let entry = create_test_entry(&format!("test{i}"), i as u64, 10);
                cache.persistent_cache.put(entry).await.unwrap();
            }
            cache.sync().unwrap();
        }

        // Reopen and warm cache
        {
            let memory_config = PrefixCacheConfig::default();
            let mut cache = HybridPersistentCache::new(
                memory_config,
                PersistentCacheConfig::new(temp_dir.path()),
            )
            .unwrap();

            assert_eq!(cache.memory_cache.len(), 0);
            assert_eq!(cache.persistent_cache.len(), 5);

            let loaded = cache.warm_cache(3).await.unwrap();
            assert_eq!(loaded, 3);
            assert_eq!(cache.memory_cache.len(), 3);
        }
    }

    // Test 15: Hybrid cache combined stats
    #[tokio::test]
    async fn test_hybrid_cache_combined_stats() {
        let (_temp_dir, persistent_config) = create_temp_config();
        let memory_config = PrefixCacheConfig::default();

        let mut cache = HybridPersistentCache::new(memory_config, persistent_config).unwrap();

        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();

        cache.put(entry).await.unwrap();
        cache.get(&fingerprint).await;

        let (memory_stats, _persistent_stats) = cache.combined_stats();
        assert_eq!(memory_stats.hits, 1);
    }

    // Test 16: PersistedEntry conversion
    #[test]
    fn test_persisted_entry_conversion() {
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
        let config = PersistentCacheConfig::new("/tmp/test")
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
        let mut cache = PersistentPrefixCache::open(config).unwrap();

        let entry = create_test_entry("test1", 12345, 10);
        cache.put(entry).await.unwrap();

        let result = cache.sync();
        assert!(result.is_ok());
    }

    // Test 20: Hybrid cache remove from both
    #[tokio::test]
    async fn test_hybrid_cache_remove() {
        let (_temp_dir, persistent_config) = create_temp_config();
        let memory_config = PrefixCacheConfig::default();

        let mut cache = HybridPersistentCache::new(memory_config, persistent_config).unwrap();

        let entry = create_test_entry("test1", 12345, 10);
        let fingerprint = entry.fingerprint.clone();
        let key = cache.put(entry).await.unwrap();

        assert!(cache.contains(&fingerprint).await);

        let removed = cache.remove(&key).await;
        assert!(removed.is_some());
        assert!(!cache.memory_cache.contains(&fingerprint).await);
    }
}
