//! `CollectionStore` trait and `InMemoryCollectionStore` implementation.
//!
//! The `CollectionStore` abstracts over any backend that can persist collection
//! registrations (metadata, config, stats). The `InMemoryCollectionStore` is a
//! pure-Rust, no-persistence implementation suitable for tests and single-node
//! deployments.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use super::types::{
    Collection, CollectionConfig, CollectionError, CollectionId, CollectionMetadata,
    CollectionStats,
};

// ── CollectionStore trait ─────────────────────────────────────────────────────

/// Persistent registry of collection definitions and statistics.
#[async_trait]
pub trait CollectionStore: Send + Sync {
    /// Register a new collection.
    ///
    /// # Errors
    ///
    /// - [`CollectionError::AlreadyExists`] if the ID is already registered.
    /// - [`CollectionError::InvalidName`] if `id` did not pass normalisation.
    /// - [`CollectionError::CapacityExceeded`] if the store is at its maximum
    ///   number of collections.
    async fn create(
        &self,
        id: CollectionId,
        config: CollectionConfig,
        metadata: CollectionMetadata,
    ) -> Result<Collection, CollectionError>;

    /// Return the named collection.
    ///
    /// # Errors
    ///
    /// [`CollectionError::NotFound`] if `id` is not registered.
    async fn get(&self, id: &CollectionId) -> Result<Collection, CollectionError>;

    /// Return all registered collections (unsorted).
    async fn list(&self) -> Vec<Collection>;

    /// Remove a collection from the registry.
    ///
    /// # Errors
    ///
    /// [`CollectionError::NotFound`] if `id` is not registered.
    async fn delete(&self, id: &CollectionId) -> Result<(), CollectionError>;

    /// Return runtime statistics for a collection.
    ///
    /// # Errors
    ///
    /// [`CollectionError::NotFound`] if `id` is not registered.
    async fn stats(&self, id: &CollectionId) -> Result<CollectionStats, CollectionError>;

    /// Replace the metadata for an existing collection.
    ///
    /// # Errors
    ///
    /// [`CollectionError::NotFound`] if `id` is not registered.
    async fn update_metadata(
        &self,
        id: &CollectionId,
        metadata: CollectionMetadata,
    ) -> Result<(), CollectionError>;

    /// Return the total number of registered collections.
    async fn collection_count(&self) -> usize;
}

// ── Internal entry ────────────────────────────────────────────────────────────

/// Combined storage unit kept in the `HashMap`.
struct StoreEntry {
    collection: Collection,
    stats: CollectionStats,
}

// ── InMemoryCollectionStore ───────────────────────────────────────────────────

/// Thread-safe, in-memory implementation of [`CollectionStore`].
///
/// Collections are stored in a `HashMap` behind a `RwLock`.  Reads are cheap
/// (shared lock); writes acquire an exclusive lock for the duration of the
/// mutation.
pub struct InMemoryCollectionStore {
    entries: Arc<RwLock<HashMap<CollectionId, StoreEntry>>>,
    /// Maximum number of collections that may be registered at once.
    max_collections: usize,
}

impl InMemoryCollectionStore {
    /// Create a new store with the default capacity of 100 collections.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            max_collections: 100,
        }
    }

    /// Create a new store with a custom maximum number of collections.
    #[must_use]
    pub fn with_max_collections(max_collections: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            max_collections,
        }
    }
}

impl Default for InMemoryCollectionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CollectionStore for InMemoryCollectionStore {
    async fn create(
        &self,
        id: CollectionId,
        config: CollectionConfig,
        metadata: CollectionMetadata,
    ) -> Result<Collection, CollectionError> {
        let mut entries = self.entries.write().await;

        if entries.contains_key(&id) {
            return Err(CollectionError::AlreadyExists(id.as_str().to_string()));
        }

        if entries.len() >= self.max_collections {
            return Err(CollectionError::CapacityExceeded(format!(
                "store has reached its maximum of {} collections",
                self.max_collections
            )));
        }

        let collection = Collection::new(id.clone(), config, metadata);
        let stats = CollectionStats::new(id.clone());

        entries.insert(
            id,
            StoreEntry {
                collection: collection.clone(),
                stats,
            },
        );

        Ok(collection)
    }

    async fn get(&self, id: &CollectionId) -> Result<Collection, CollectionError> {
        let entries = self.entries.read().await;
        entries
            .get(id)
            .map(|e| e.collection.clone())
            .ok_or_else(|| CollectionError::NotFound(id.as_str().to_string()))
    }

    async fn list(&self) -> Vec<Collection> {
        let entries = self.entries.read().await;
        entries.values().map(|e| e.collection.clone()).collect()
    }

    async fn delete(&self, id: &CollectionId) -> Result<(), CollectionError> {
        let mut entries = self.entries.write().await;
        entries
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| CollectionError::NotFound(id.as_str().to_string()))
    }

    async fn stats(&self, id: &CollectionId) -> Result<CollectionStats, CollectionError> {
        let entries = self.entries.read().await;
        entries
            .get(id)
            .map(|e| e.stats.clone())
            .ok_or_else(|| CollectionError::NotFound(id.as_str().to_string()))
    }

    async fn update_metadata(
        &self,
        id: &CollectionId,
        metadata: CollectionMetadata,
    ) -> Result<(), CollectionError> {
        let mut entries = self.entries.write().await;
        match entries.get_mut(id) {
            Some(entry) => {
                entry.collection.metadata = metadata;
                Ok(())
            }
            None => Err(CollectionError::NotFound(id.as_str().to_string())),
        }
    }

    async fn collection_count(&self) -> usize {
        self.entries.read().await.len()
    }
}
