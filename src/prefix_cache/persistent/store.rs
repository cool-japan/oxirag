//! `PersistentPrefixCache` and `HybridPersistentCache` structs with
//! `PrefixCacheStore` implementations.

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use super::super::store::InMemoryPrefixCache;
use super::super::traits::PrefixCacheStore;
use super::super::types::{
    CacheKey, CacheStats, ContextFingerprint, KVCacheEntry, PrefixCacheConfig,
};
use super::io::{append_entry, load_index, read_entry, rewrite_data_file, save_index};
use super::types::{
    CacheIndex, CompactionStats, IndexEntry, PersistedEntry, PersistentCacheConfig,
};
use crate::error::OxiRagError;

// ---------------------------------------------------------------------------
// PersistentPrefixCache
// ---------------------------------------------------------------------------

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
    pub(super) config: PersistentCacheConfig,
    /// The cache index.
    pub(super) index: Arc<RwLock<CacheIndex>>,
    /// Path to the data file.
    pub(super) data_file: PathBuf,
    /// Path to the index file.
    pub(super) index_file: PathBuf,
    /// Whether the index has uncommitted changes.
    pub(super) dirty: Arc<RwLock<bool>>,
    /// Cache statistics.
    pub(super) stats: Arc<RwLock<CacheStats>>,
    /// Next key ID for generation.
    pub(super) next_key_id: Arc<RwLock<u64>>,
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
        cache.load_index_from_disk()?;

        Ok(cache)
    }

    /// Load index from disk.
    fn load_index_from_disk(&mut self) -> Result<(), OxiRagError> {
        let Some(index) = load_index(&self.index_file)? else {
            return Ok(());
        };

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
        save_index(&self.index_file, &index, &self.dirty)
    }

    /// Read an entry from the data file at the given byte offset.
    pub(super) fn read_entry(&self, offset: u64) -> Result<PersistedEntry, OxiRagError> {
        read_entry(&self.data_file, offset)
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
                if let Ok(entry) = read_entry(&self.data_file, idx_entry.file_offset) {
                    valid_entries.push((key.clone(), entry));
                }
            }
            drop(index);

            let new_index = rewrite_data_file(&self.data_file, valid_entries)?;
            *self.index.write().expect("lock poisoned") = new_index;
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
        if let Ok(persisted) = read_entry(&self.data_file, offset) {
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
        let (offset, size) = append_entry(&self.data_file, &persisted, &self.dirty)?;

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
            if let Ok(persisted) = read_entry(&self.data_file, offset) {
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

// ---------------------------------------------------------------------------
// HybridPersistentCache
// ---------------------------------------------------------------------------

/// Hybrid cache combining memory and persistent storage.
///
/// This cache uses an in-memory cache as L1 and a persistent cache as L2,
/// providing fast access for hot data while persisting all entries to disk.
pub struct HybridPersistentCache {
    /// In-memory L1 cache.
    pub(super) memory_cache: InMemoryPrefixCache,
    /// Persistent L2 cache.
    pub(super) persistent_cache: PersistentPrefixCache,
    /// Whether to write to both caches immediately.
    pub(super) write_through: bool,
    /// Whether to check persistent cache on memory miss.
    pub(super) read_through: bool,
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
