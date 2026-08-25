//! IndexedDB-backed vector store for WASM targets.
//!
//! This module provides a persistent [`IndexedDbVectorStore`] that stores document
//! embeddings in the browser's native IndexedDB, enabling cross-session persistence
//! without a server-side backend.
//!
//! # Architecture
//!
//! Each document is stored as a JSON object in the `"documents"` object store with
//! a keyPath of `"id"`. Full-table scan similarity search is used deliberately —
//! WASM edge deployments operate over small-to-medium corpora where ANN index overhead
//! is unnecessary.
//!
//! # Feature gate
//!
//! This file is compiled only when **both** `target_arch = "wasm32"` and the
//! `wasm-indexeddb` Cargo feature are active.

#![cfg(all(target_arch = "wasm32", feature = "wasm-indexeddb"))]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::error::VectorStoreError;
use crate::layer1_echo::filter::MetadataFilter;
use crate::layer1_echo::similarity::compute_similarity;
use crate::layer1_echo::traits::{IndexedDocument, SimilarityMetric, VectorStore};
use crate::types::{Document, DocumentId, SearchResult};

/// Name of the IndexedDB database used by this store.
const DB_NAME: &str = "oxirag_vectors";
/// Name of the object store within the database.
const STORE_NAME: &str = "documents";
/// Schema version. Increment when the object store layout changes.
const DB_VERSION: u32 = 1;

// ────────────────────────────────────────────────────────────────────────────
// On-disk representation
// ────────────────────────────────────────────────────────────────────────────

/// Serialisable entry persisted in the IndexedDB object store.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexedDbEntry {
    /// Document id — used as the IDB keyPath.
    id: String,
    /// Plain-text content of the document.
    content: String,
    /// Optional human-readable title.
    title: Option<String>,
    /// Dense embedding vector.
    embedding: Vec<f32>,
    /// Arbitrary metadata key-value pairs.
    metadata: HashMap<String, String>,
}

impl IndexedDbEntry {
    fn from_indexed_document(indexed: &IndexedDocument) -> Self {
        Self {
            id: indexed.document.id.to_string(),
            content: indexed.document.content.clone(),
            title: indexed.document.title.clone(),
            embedding: indexed.embedding.clone(),
            metadata: indexed.document.metadata.clone(),
        }
    }

    fn into_indexed_document(self) -> IndexedDocument {
        let mut doc = Document::new(self.content).with_id(DocumentId::from_string(self.id));
        if let Some(t) = self.title {
            doc = doc.with_title(t);
        }
        for (k, v) in self.metadata {
            doc = doc.with_metadata(k, v);
        }
        IndexedDocument::new(doc, self.embedding)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// IDB helpers
// ────────────────────────────────────────────────────────────────────────────

/// Convert a [`JsValue`] error into [`VectorStoreError::StorageError`].
///
/// By value rather than by reference so it can be named directly in
/// `.map_err(js_err)`, which is every one of its ~20 call sites.
#[allow(clippy::needless_pass_by_value)]
fn js_err(js: JsValue) -> VectorStoreError {
    VectorStoreError::StorageError(js.as_string().unwrap_or_else(|| format!("{js:?}")))
}

/// Open (or upgrade) the vector store database and return the [`web_sys::IdbDatabase`].
///
/// If the database does not exist yet the `onupgradeneeded` callback creates the
/// `"documents"` object store with `keyPath = "id"`.
async fn open_db() -> Result<web_sys::IdbDatabase, VectorStoreError> {
    // A `Window` OR a `WorkerGlobalScope`: IndexedDB is available to workers,
    // and a RAG pipeline belongs in one. Reaching only for `window()` reported
    // "not available in this browser" on the host where this engine should run.
    let scope = crate::global_scope::GlobalScope::current().ok_or_else(|| {
        VectorStoreError::StorageError(
            "no Window or WorkerGlobalScope (not a browser context)".into(),
        )
    })?;

    let idb_factory = scope.indexed_db().map_err(js_err)?.ok_or_else(|| {
        VectorStoreError::StorageError("IndexedDB not available in this browser".into())
    })?;

    let open_request: web_sys::IdbOpenDbRequest = idb_factory
        .open_with_u32(DB_NAME, DB_VERSION)
        .map_err(js_err)?;

    // Set up the schema on first creation / version upgrade.
    // We use an out-of-line key (no keyPath) so we can pass the document id
    // explicitly in every add/put/get/delete call.  This avoids the need for
    // the `IdbObjectStoreParameters` web-sys feature flag.
    let on_upgrade: Closure<dyn FnMut(web_sys::IdbVersionChangeEvent)> =
        Closure::once(|event: web_sys::IdbVersionChangeEvent| {
            if let Some(target) = event.target() {
                let req: web_sys::IdbOpenDbRequest = target.unchecked_into();
                // web-sys 0.3.104 returns the raw `JsValue`; a request that has
                // not settled yields `undefined` rather than `None`.
                if let Ok(result) = req.result()
                    && !result.is_undefined()
                    && !result.is_null()
                {
                    let db: web_sys::IdbDatabase = result.unchecked_into();
                    if !db.object_store_names().contains(STORE_NAME) {
                        // Plain createObjectStore — out-of-line key, no autoIncrement.
                        let _ = db.create_object_store(STORE_NAME);
                    }
                }
            }
        });
    open_request.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
    on_upgrade.forget();

    // Clone the request so we can retrieve the result after the Promise resolves
    // while also moving the original into the Promise callbacks.
    let open_request_clone = open_request.clone();

    // Build a Promise that resolves/rejects from the IDB request callbacks,
    // because IdbOpenDbRequest is not itself a Promise.
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        // IdbOpenDbRequest extends IdbRequest in the DOM — wasm-bindgen generates
        // AsRef<IdbRequest> so we can call the inherited callback setters.
        let req_as_idb: &web_sys::IdbRequest =
            AsRef::<web_sys::IdbRequest>::as_ref(&open_request_clone);

        let onsuccess: Closure<dyn FnMut(web_sys::Event)> =
            Closure::once(move |_: web_sys::Event| {
                let _ = resolve.call0(&JsValue::undefined());
            });
        req_as_idb.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
        onsuccess.forget();

        let req_as_idb2: &web_sys::IdbRequest =
            AsRef::<web_sys::IdbRequest>::as_ref(&open_request_clone);
        let onerror: Closure<dyn FnMut(web_sys::Event)> =
            Closure::once(move |_: web_sys::Event| {
                let _ = reject.call1(&JsValue::undefined(), &JsValue::from_str("IDB open failed"));
            });
        req_as_idb2.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    });

    JsFuture::from(promise).await.map_err(js_err)?;

    // Retrieve the IdbDatabase from the original request handle.
    let result = open_request.result().map_err(js_err)?;
    if result.is_undefined() || result.is_null() {
        return Err(VectorStoreError::StorageError(
            "IDB open succeeded but result is null".into(),
        ));
    }
    let db: web_sys::IdbDatabase = result.unchecked_into();

    Ok(db)
}

/// Execute a read-write transaction on the `"documents"` store and return
/// the typed result of a single IDB request built by `f`.
async fn with_rw_store<F>(f: F) -> Result<JsValue, VectorStoreError>
where
    F: FnOnce(&web_sys::IdbObjectStore) -> Result<web_sys::IdbRequest, VectorStoreError>,
{
    let db = open_db().await?;

    let tx = db
        .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
        .map_err(js_err)?;

    let store = tx.object_store(STORE_NAME).map_err(js_err)?;

    let request = f(&store)?;

    await_idb_request(request).await
}

/// Execute a read-only transaction on the `"documents"` store and return
/// the typed result of a single IDB request built by `f`.
async fn with_ro_store<F>(f: F) -> Result<JsValue, VectorStoreError>
where
    F: FnOnce(&web_sys::IdbObjectStore) -> Result<web_sys::IdbRequest, VectorStoreError>,
{
    let db = open_db().await?;

    let tx = db.transaction_with_str(STORE_NAME).map_err(js_err)?;

    let store = tx.object_store(STORE_NAME).map_err(js_err)?;

    let request = f(&store)?;

    await_idb_request(request).await
}

/// Wrap an [`web_sys::IdbRequest`] into a [`JsFuture`] by wiring up its
/// `onsuccess` and `onerror` callbacks through a `js_sys::Promise`.
async fn await_idb_request(request: web_sys::IdbRequest) -> Result<JsValue, VectorStoreError> {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let resolve_clone = resolve.clone();
        let req_clone = request.clone();

        let onsuccess: Closure<dyn FnMut(web_sys::Event)> =
            Closure::once(move |_e: web_sys::Event| {
                let result = req_clone.result().unwrap_or(JsValue::undefined());
                let _ = resolve_clone.call1(&JsValue::undefined(), &result);
            });
        request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
        onsuccess.forget();

        let onerror: Closure<dyn FnMut(web_sys::Event)> =
            Closure::once(move |e: web_sys::Event| {
                let _ = reject.call1(&JsValue::undefined(), &JsValue::from(e));
            });
        request.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    });

    JsFuture::from(promise).await.map_err(js_err)
}

/// Retrieve all [`IndexedDbEntry`] values from the object store.
async fn get_all_entries() -> Result<Vec<IndexedDbEntry>, VectorStoreError> {
    let result = with_ro_store(|store| store.get_all().map_err(js_err)).await?;

    if result.is_undefined() || result.is_null() {
        return Ok(Vec::new());
    }

    let array: js_sys::Array = result.unchecked_into();
    let mut entries = Vec::with_capacity(array.length() as usize);

    for i in 0..array.length() {
        let item = array.get(i);
        let entry: IndexedDbEntry = serde_wasm_bindgen::from_value(item)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        entries.push(entry);
    }

    Ok(entries)
}

// ────────────────────────────────────────────────────────────────────────────
// Public struct
// ────────────────────────────────────────────────────────────────────────────

/// An IndexedDB-backed [`VectorStore`] for use in WASM (browser) environments.
///
/// Documents and their embedding vectors are persisted across page reloads via the
/// browser's IndexedDB API. Similarity search performs a full table scan, which is
/// intentional for the small-to-medium corpora typical in edge RAG deployments.
///
/// # Example (WASM context)
///
/// ```rust,ignore
/// use oxirag::layer1_echo::{IndexedDbVectorStore, VectorStore};
/// use oxirag::layer1_echo::traits::SimilarityMetric;
///
/// let store = IndexedDbVectorStore::new(384)
///     .with_metric(SimilarityMetric::Cosine);
///
/// // Use as a VectorStore via trait methods.
/// ```
pub struct IndexedDbVectorStore {
    /// Expected embedding dimension.
    dimension: usize,
    /// Similarity metric for nearest-neighbour search.
    metric: SimilarityMetric,
}

impl IndexedDbVectorStore {
    /// Create a new store backed by the default database (`"oxirag_vectors"`).
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: SimilarityMetric::Cosine,
        }
    }

    /// Override the similarity metric.
    #[must_use]
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Validate that an embedding's dimension matches the store dimension.
    fn check_dimension(&self, embedding: &[f32]) -> Result<(), VectorStoreError> {
        if embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// VectorStore impl
// ────────────────────────────────────────────────────────────────────────────

#[async_trait(?Send)]
impl VectorStore for IndexedDbVectorStore {
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError> {
        self.check_dimension(&doc.embedding)?;

        let entry = IndexedDbEntry::from_indexed_document(&doc);
        let doc_id_str = entry.id.clone();
        let js_entry = serde_wasm_bindgen::to_value(&entry)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let js_key = JsValue::from_str(&doc_id_str);

        // Use `add_with_key` so a duplicate key raises a ConstraintError.
        with_rw_store(|store| store.add_with_key(&js_entry, &js_key).map_err(js_err))
            .await
            .map_err(|e| {
                // IDB raises a ConstraintError for duplicate keys.
                let msg = e.to_string();
                if msg.contains("ConstraintError") || msg.contains("constraint") {
                    VectorStoreError::DuplicateId(doc_id_str.clone())
                } else {
                    e
                }
            })?;

        Ok(())
    }

    async fn insert_batch(&mut self, docs: Vec<IndexedDocument>) -> Result<(), VectorStoreError> {
        for doc in docs {
            self.insert(doc).await?;
        }
        Ok(())
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<IndexedDocument>, VectorStoreError> {
        let key = JsValue::from_str(id.as_str());

        let result = with_ro_store(|store| store.get(&key).map_err(js_err)).await?;

        if result.is_undefined() || result.is_null() {
            return Ok(None);
        }

        let entry: IndexedDbEntry = serde_wasm_bindgen::from_value(result)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(Some(entry.into_indexed_document()))
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        // Check existence first (IDB delete is idempotent).
        let exists = self.get(id).await?.is_some();
        if !exists {
            return Ok(false);
        }

        let key = JsValue::from_str(id.as_str());
        with_rw_store(|store| store.delete(&key).map_err(js_err)).await?;

        Ok(true)
    }

    async fn update(
        &mut self,
        id: &DocumentId,
        embedding: Vec<f32>,
    ) -> Result<bool, VectorStoreError> {
        self.check_dimension(&embedding)?;

        let existing = self
            .get(id)
            .await?
            .ok_or_else(|| VectorStoreError::NotFound(id.to_string()))?;

        let mut updated = existing;
        updated.embedding = embedding;

        let entry = IndexedDbEntry::from_indexed_document(&updated);
        let js_key = JsValue::from_str(&entry.id);
        let js_entry = serde_wasm_bindgen::to_value(&entry)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        // `put_with_key` overwrites an existing record.
        with_rw_store(|store| store.put_with_key(&js_entry, &js_key).map_err(js_err)).await?;

        Ok(true)
    }

    async fn upsert(&mut self, doc: IndexedDocument) -> Result<bool, VectorStoreError> {
        self.check_dimension(&doc.embedding)?;

        let is_insert = self.get(&doc.document.id).await?.is_none();

        let entry = IndexedDbEntry::from_indexed_document(&doc);
        let js_key = JsValue::from_str(&entry.id);
        let js_entry = serde_wasm_bindgen::to_value(&entry)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        with_rw_store(|store| store.put_with_key(&js_entry, &js_key).map_err(js_err)).await?;

        Ok(is_insert)
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.check_dimension(query_embedding)?;

        let entries = get_all_entries().await?;
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let metric = self.metric;
        let mut scored: Vec<(f32, IndexedDbEntry)> = entries
            .into_iter()
            .filter_map(|e| {
                if e.embedding.len() != self.dimension {
                    return None;
                }
                let score = compute_similarity(query_embedding, &e.embedding, metric);
                if let Some(min) = min_score
                    && score < min
                {
                    return None;
                }
                Some((score, e))
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let results = scored
            .into_iter()
            .enumerate()
            .map(|(rank, (score, entry))| {
                SearchResult::new(entry.into_indexed_document().document, score, rank)
            })
            .collect();

        Ok(results)
    }

    async fn search_with_filter(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.check_dimension(query_embedding)?;

        let entries = get_all_entries().await?;
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let metric = self.metric;
        let mut scored: Vec<(f32, IndexedDbEntry)> = entries
            .into_iter()
            .filter_map(|e| {
                if e.embedding.len() != self.dimension {
                    return None;
                }
                // Apply metadata filter if provided.
                if let Some(f) = filter
                    && !f.matches(&e.metadata)
                {
                    return None;
                }
                let score = compute_similarity(query_embedding, &e.embedding, metric);
                if let Some(min) = min_score
                    && score < min
                {
                    return None;
                }
                Some((score, e))
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let results = scored
            .into_iter()
            .enumerate()
            .map(|(rank, (score, entry))| {
                SearchResult::new(entry.into_indexed_document().document, score, rank)
            })
            .collect();

        Ok(results)
    }

    async fn count(&self) -> usize {
        let result = with_ro_store(|store| store.count().map_err(js_err)).await;

        match result {
            // `IDBObjectStore.count()` returns a non-negative integer well inside
            // `usize` on wasm32; a browser reporting anything else counts as zero
            // rather than as a wrapped-around number.
            Ok(v) => v.as_f64().map_or(0, |n| {
                if n.is_finite() && n >= 0.0 {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        n as usize
                    }
                } else {
                    0
                }
            }),
            Err(_) => 0,
        }
    }

    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        with_rw_store(|store| store.clear().map_err(js_err)).await?;
        Ok(())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn similarity_metric(&self) -> SimilarityMetric {
        self.metric
    }
}

// ────────────────────────────────────────────────────────────────────────────
// WASM-bindgen tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn make_doc(content: &str, dim: usize) -> IndexedDocument {
        let doc = Document::new(content);
        let embedding: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32)).collect();
        IndexedDocument::new(doc, embedding)
    }

    /// Insert a document and verify it can be searched.
    #[wasm_bindgen_test]
    async fn test_indexeddb_insert_and_search() {
        let mut store = IndexedDbVectorStore::new(4);

        // Start clean.
        store.clear().await.expect("clear should succeed");

        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let doc = Document::new("hello world");
        let indexed = IndexedDocument::new(doc, query.clone());

        store.insert(indexed).await.expect("insert should succeed");

        let results = store
            .search(&query, 5, None)
            .await
            .expect("search should succeed");

        assert_eq!(results.len(), 1, "expected exactly one result");
        assert_eq!(results[0].document.content, "hello world");
    }

    /// Verify `count` and `clear` behave correctly.
    #[wasm_bindgen_test]
    async fn test_indexeddb_count_and_clear() {
        let mut store = IndexedDbVectorStore::new(4);
        store.clear().await.expect("clear should succeed");

        assert_eq!(store.count().await, 0);

        let d1 = make_doc("doc one", 4);
        let d2 = make_doc("doc two", 4);
        store.insert(d1).await.expect("insert d1");
        store.insert(d2).await.expect("insert d2");
        assert_eq!(store.count().await, 2);

        store.clear().await.expect("clear should succeed");
        assert_eq!(store.count().await, 0);
    }

    /// Ensure `dimension()` returns the value given at construction.
    #[wasm_bindgen_test]
    async fn test_indexeddb_dimension() {
        let store = IndexedDbVectorStore::new(128);
        assert_eq!(store.dimension(), 128);

        let store2 = IndexedDbVectorStore::new(384);
        assert_eq!(store2.dimension(), 384);
    }
}
