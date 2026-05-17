//! `CollectionIndex` — per-collection vector stores with federated search.
//!
//! Each collection has its own `InMemoryVectorStore` (bounded by
//! `CollectionConfig.max_documents`).  The `CollectionIndex` manages
//! creation/deletion of per-collection stores in lock-step with the
//! `CollectionStore` registry, and provides:
//!
//! - Single-collection indexing and search.
//! - Cross-collection search with Reciprocal Rank Fusion (RRF).
//! - `search_all` that spans every registered collection.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::layer1_echo::{InMemoryVectorStore, IndexedDocument, VectorStore};
use crate::types::{Document, DocumentId, SearchResult};

use super::store::CollectionStore;
use super::types::{
    Collection, CollectionConfig, CollectionError, CollectionId, CollectionMetadata,
    CollectionStats, FederatedResult,
};

// ── RRF constant ──────────────────────────────────────────────────────────────

/// Standard constant used in Reciprocal Rank Fusion.
const RRF_K: f32 = 60.0;

// ── Per-collection storage ────────────────────────────────────────────────────

/// Stores a single collection's vector store behind a Mutex so that concurrent
/// writes (which require `&mut VectorStore`) can be serialised.
struct CollectionSlot {
    store: Mutex<InMemoryVectorStore>,
}

impl CollectionSlot {
    fn new(config: &CollectionConfig) -> Self {
        let store = match config.max_documents {
            Some(cap) => {
                InMemoryVectorStore::new(config.embedding_dimension).with_max_capacity(cap)
            }
            None => InMemoryVectorStore::new(config.embedding_dimension),
        };
        Self {
            store: Mutex::new(store),
        }
    }
}

// ── CollectionIndex ───────────────────────────────────────────────────────────

/// High-level index that maps collection IDs to individual vector stores.
///
/// `S` is the backing `CollectionStore` (registry of metadata/config/stats).
pub struct CollectionIndex<S: CollectionStore> {
    store: Arc<S>,
    slots: Arc<RwLock<HashMap<CollectionId, CollectionSlot>>>,
}

impl<S: CollectionStore> CollectionIndex<S> {
    /// Create a new index backed by `store`.
    pub fn new(store: S) -> Self {
        Self {
            store: Arc::new(store),
            slots: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // ── Collection lifecycle ──────────────────────────────────────────────────

    /// Create a new collection in both the store and the slot map.
    ///
    /// # Errors
    ///
    /// Propagates errors from `CollectionStore::create`.
    pub async fn create_collection(
        &self,
        id: CollectionId,
        config: CollectionConfig,
        metadata: CollectionMetadata,
    ) -> Result<Collection, CollectionError> {
        let collection = self
            .store
            .create(id.clone(), config.clone(), metadata)
            .await?;

        let mut slots = self.slots.write().await;
        slots.insert(id, CollectionSlot::new(&config));

        Ok(collection)
    }

    /// Retrieve collection metadata from the store.
    ///
    /// # Errors
    ///
    /// [`CollectionError::NotFound`] if `id` is not registered.
    pub async fn get_collection(&self, id: &CollectionId) -> Result<Collection, CollectionError> {
        self.store.get(id).await
    }

    /// List all collections.
    pub async fn list_collections(&self) -> Vec<Collection> {
        self.store.list().await
    }

    /// Delete a collection and its associated vector store.
    ///
    /// # Errors
    ///
    /// [`CollectionError::NotFound`] if `id` is not registered.
    pub async fn delete_collection(&self, id: &CollectionId) -> Result<(), CollectionError> {
        self.store.delete(id).await?;
        let mut slots = self.slots.write().await;
        slots.remove(id);
        Ok(())
    }

    /// Retrieve stats for a collection.
    ///
    /// # Errors
    ///
    /// [`CollectionError::NotFound`] if `id` is not registered.
    pub async fn stats(&self, id: &CollectionId) -> Result<CollectionStats, CollectionError> {
        self.store.stats(id).await
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    /// Index a document with a pre-computed embedding into `collection_id`.
    ///
    /// # Errors
    ///
    /// - [`CollectionError::NotFound`] — collection not registered.
    /// - [`CollectionError::DimensionMismatch`] — embedding length wrong.
    /// - [`CollectionError::CapacityExceeded`] — collection is full.
    pub async fn index_document(
        &self,
        collection_id: &CollectionId,
        doc: Document,
        embedding: Vec<f32>,
    ) -> Result<Uuid, CollectionError> {
        // Validate dimension against config.
        let collection = self.store.get(collection_id).await?;
        let expected_dim = collection.config().embedding_dimension;
        if embedding.len() != expected_dim {
            return Err(CollectionError::DimensionMismatch {
                expected: expected_dim,
                got: embedding.len(),
            });
        }

        let slots = self.slots.read().await;
        let slot = slots
            .get(collection_id)
            .ok_or_else(|| CollectionError::NotFound(collection_id.as_str().to_string()))?;

        let doc_uuid: Uuid = {
            // Parse the DocumentId (which is a UUID string) or generate a fresh one.
            doc.id
                .as_str()
                .parse::<Uuid>()
                .unwrap_or_else(|_| Uuid::new_v4())
        };

        let indexed = IndexedDocument::new(doc, embedding);

        let mut store_guard = slot.store.lock().await;
        store_guard
            .insert(indexed)
            .await
            .map_err(|e| CollectionError::StorageError(e.to_string()))?;

        Ok(doc_uuid)
    }

    // ── Single-collection search ──────────────────────────────────────────────

    /// Search within a single collection.
    ///
    /// # Errors
    ///
    /// - [`CollectionError::NotFound`] — collection not registered.
    /// - [`CollectionError::DimensionMismatch`] — query embedding length wrong.
    pub async fn search(
        &self,
        collection_id: &CollectionId,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, CollectionError> {
        // Validate collection existence and dimension.
        let collection = self.store.get(collection_id).await?;
        let expected_dim = collection.config().embedding_dimension;
        if query_embedding.len() != expected_dim {
            return Err(CollectionError::DimensionMismatch {
                expected: expected_dim,
                got: query_embedding.len(),
            });
        }

        let start = Instant::now();

        let slots = self.slots.read().await;
        let slot = slots
            .get(collection_id)
            .ok_or_else(|| CollectionError::NotFound(collection_id.as_str().to_string()))?;

        let store_guard = slot.store.lock().await;
        let results = store_guard
            .search(query_embedding, top_k, None)
            .await
            .map_err(|e| CollectionError::StorageError(e.to_string()))?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        drop(store_guard);
        drop(slots);

        // Update stats (best-effort; noop in the current trait model).
        Self::record_search_latency_noop(collection_id, latency_ms);

        Ok(results)
    }

    /// Record a search latency measurement (best-effort noop).
    fn record_search_latency_noop(id: &CollectionId, latency_ms: f64) {
        // The CollectionStore trait has no mutable stats handle; for a real
        // persistent store this would issue a write.  Here we acknowledge the
        // parameters to silence dead-code warnings.
        let _ = (id, latency_ms);
    }

    // ── Cross-collection search with RRF ──────────────────────────────────────

    /// Search across multiple named collections and merge results with RRF.
    ///
    /// RRF score = `Σ_collection`  1 / (K + `rank_in_collection`)
    /// where K = 60 (standard constant).
    ///
    /// Results are deduplicated by `document_id`; the highest fused score wins.
    /// The returned list is sorted by descending fused score and contains at
    /// most `top_k_per_collection * collection_ids.len()` entries.
    ///
    /// # Errors
    ///
    /// Propagates per-collection errors but continues searching other
    /// collections; if *all* collections fail the last error is returned.
    pub async fn cross_collection_search(
        &self,
        collection_ids: &[CollectionId],
        query_embedding: &[f32],
        top_k_per_collection: usize,
    ) -> Result<Vec<FederatedResult>, CollectionError> {
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Map: document_id → (best content, best collection, accumulated RRF score)
        let mut rrf_scores: HashMap<String, (String, String, f32)> = HashMap::new();

        let mut last_error: Option<CollectionError> = None;
        let mut any_ok = false;

        for cid in collection_ids {
            let results = match self
                .search(cid, query_embedding, top_k_per_collection)
                .await
            {
                Ok(r) => {
                    any_ok = true;
                    r
                }
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };

            for (rank, sr) in results.iter().enumerate() {
                let doc_id = sr.document.id.to_string();
                #[allow(clippy::cast_precision_loss)]
                let rrf_contribution = 1.0 / (RRF_K + rank as f32);

                let entry = rrf_scores.entry(doc_id).or_insert_with(|| {
                    (sr.document.content.clone(), cid.as_str().to_string(), 0.0)
                });
                entry.2 += rrf_contribution;
            }
        }

        if !any_ok && let Some(e) = last_error {
            return Err(e);
        }

        // Build FederatedResult list and sort by descending score.
        let mut merged: Vec<FederatedResult> = rrf_scores
            .into_iter()
            .map(
                |(doc_id, (content, collection_id, score))| FederatedResult {
                    content,
                    score,
                    collection_id,
                    document_id: doc_id,
                    rank: 0, // filled in after sorting
                },
            )
            .collect();

        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let max_results = top_k_per_collection.saturating_mul(collection_ids.len());
        merged.truncate(max_results);

        for (rank, result) in merged.iter_mut().enumerate() {
            result.rank = rank;
        }

        Ok(merged)
    }

    /// Search every registered collection using RRF fusion.
    ///
    /// Equivalent to `cross_collection_search` over all collection IDs returned
    /// by `list_collections`.
    ///
    /// # Errors
    ///
    /// Returns `Ok(Vec::new())` when there are no collections.
    pub async fn search_all(
        &self,
        query_embedding: &[f32],
        top_k_per_collection: usize,
    ) -> Result<Vec<FederatedResult>, CollectionError> {
        let collections = self.store.list().await;
        if collections.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<CollectionId> = collections.into_iter().map(|c| c.id().clone()).collect();
        self.cross_collection_search(&ids, query_embedding, top_k_per_collection)
            .await
    }

    /// Return the number of documents currently indexed in `collection_id`.
    ///
    /// # Errors
    ///
    /// [`CollectionError::NotFound`] if `collection_id` is not registered.
    pub async fn document_count(
        &self,
        collection_id: &CollectionId,
    ) -> Result<usize, CollectionError> {
        // Confirm the collection exists in the registry.
        let _ = self.store.get(collection_id).await?;

        let slots = self.slots.read().await;
        match slots.get(collection_id) {
            Some(slot) => {
                let guard = slot.store.lock().await;
                Ok(guard.count().await)
            }
            None => Err(CollectionError::NotFound(
                collection_id.as_str().to_string(),
            )),
        }
    }

    /// Return the ID of the underlying document before it is assigned by the
    /// store.  This helper is used in tests only.
    #[must_use]
    pub fn make_document_id() -> DocumentId {
        DocumentId::from_string(Uuid::new_v4().to_string())
    }
}
