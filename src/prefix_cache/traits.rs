//! Traits for prefix cache storage backends.
//!
//! This module defines the core trait that prefix cache implementations must fulfill.

use async_trait::async_trait;

use super::types::{CacheKey, CacheStats, ContextFingerprint, KVCacheEntry};
use crate::error::OxiRagError;

/// A trait for prefix cache storage backends.
///
/// Implementations of this trait provide storage and retrieval of KV cache
/// entries for context-aware prefix caching. The cache helps avoid redundant
/// computation by storing pre-computed transformer attention states.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to support concurrent access
/// from multiple tasks.
///
/// # Example
///
/// ```rust,ignore
/// use oxirag::prefix_cache::{InMemoryPrefixCache, PrefixCacheStore, ContextFingerprint};
///
/// #[tokio::main]
/// async fn main() {
///     let mut cache = InMemoryPrefixCache::new(Default::default());
///
///     let fingerprint = ContextFingerprint::new(12345, 100, "sample context");
///
///     // Check if context is cached
///     if cache.contains(&fingerprint).await {
///         if let Some(entry) = cache.get(&fingerprint).await {
///             // Use cached KV state
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait PrefixCacheStore: Send + Sync {
    /// Retrieve a cached entry by its fingerprint.
    ///
    /// Returns `Some(entry)` if found and not expired, `None` otherwise.
    /// Accessing an entry updates its last-accessed time.
    async fn get(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry>;

    /// Store a new cache entry.
    ///
    /// Returns the cache key assigned to the entry.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The cache is at capacity and eviction fails
    /// - The entry exceeds maximum size limits
    async fn put(&mut self, entry: KVCacheEntry) -> Result<CacheKey, OxiRagError>;

    /// Remove an entry by its cache key.
    ///
    /// Returns the removed entry if it existed.
    async fn remove(&mut self, key: &CacheKey) -> Option<KVCacheEntry>;

    /// Check if an entry exists for the given fingerprint.
    ///
    /// This method should be efficient and not update access statistics.
    async fn contains(&self, fingerprint: &ContextFingerprint) -> bool;

    /// Clear all entries from the cache.
    async fn clear(&mut self);

    /// Get current cache statistics.
    fn stats(&self) -> CacheStats;

    /// Get the number of entries in the cache.
    fn len(&self) -> usize;

    /// Check if the cache is empty.
    fn is_empty(&self) -> bool;

    /// Find entries that could serve as a prefix for the given fingerprint.
    ///
    /// This enables partial cache hits where a shorter cached context
    /// can be extended rather than recomputed from scratch.
    async fn find_prefix_match(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        // Default implementation returns None (no prefix matching support)
        let _ = fingerprint;
        None
    }

    /// Remove expired entries from the cache.
    ///
    /// Returns the number of entries removed.
    async fn evict_expired(&mut self) -> usize;

    /// Get memory usage in bytes.
    fn memory_usage(&self) -> usize;
}

/// Extension trait for prefix cache operations with better ergonomics.
#[async_trait]
pub trait PrefixCacheExt: PrefixCacheStore {
    /// Get or compute a cache entry.
    ///
    /// If the fingerprint exists in the cache, returns it.
    /// Otherwise, calls the compute function and stores the result.
    async fn get_or_compute<F>(
        &mut self,
        fingerprint: &ContextFingerprint,
        compute: F,
    ) -> Result<KVCacheEntry, OxiRagError>
    where
        F: FnOnce() -> Result<KVCacheEntry, OxiRagError> + Send;
}

#[async_trait]
impl<T: PrefixCacheStore + Send> PrefixCacheExt for T {
    async fn get_or_compute<F>(
        &mut self,
        fingerprint: &ContextFingerprint,
        compute: F,
    ) -> Result<KVCacheEntry, OxiRagError>
    where
        F: FnOnce() -> Result<KVCacheEntry, OxiRagError> + Send,
    {
        if let Some(entry) = self.get(fingerprint).await {
            return Ok(entry);
        }

        let entry = compute()?;
        self.put(entry.clone()).await?;
        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix_cache::InMemoryPrefixCache;
    use crate::prefix_cache::types::PrefixCacheConfig;

    #[tokio::test]
    async fn test_prefix_cache_ext_get_or_compute() {
        let mut cache = InMemoryPrefixCache::new(PrefixCacheConfig::default());
        let fingerprint = ContextFingerprint::new(12345, 100, "test");

        // First call should compute
        let mut computed = false;
        let entry = cache
            .get_or_compute(&fingerprint, || {
                computed = true;
                Ok(KVCacheEntry::new(
                    "key1",
                    fingerprint.clone(),
                    vec![1.0, 2.0],
                    100,
                ))
            })
            .await
            .unwrap();

        assert!(computed);
        assert_eq!(entry.fingerprint, fingerprint);

        // Second call should hit cache
        let mut computed2 = false;
        let entry2 = cache
            .get_or_compute(&fingerprint, || {
                computed2 = true;
                Ok(KVCacheEntry::new(
                    "key2",
                    fingerprint.clone(),
                    vec![3.0, 4.0],
                    100,
                ))
            })
            .await
            .unwrap();

        assert!(!computed2);
        assert_eq!(entry2.kv_data, vec![1.0, 2.0]); // Original data
    }
}
