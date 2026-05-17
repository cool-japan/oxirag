//! Persistent vector store backed by `redb` — an embedded, ACID-compliant
//! key-value store written in pure Rust.
//!
//! Every document (with its embedding) is serialized as JSON and stored under
//! its [`DocumentId`] string key inside a single `redb` table named
//! `"documents"`.  Similarity search loads all entries into memory and
//! delegates to [`top_k_similar`](crate::layer1_echo::similarity::top_k_similar),
//! matching the behaviour of [`InMemoryVectorStore`](super::InMemoryVectorStore).

#![cfg(feature = "echo-redb")]

use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::error::VectorStoreError;
use crate::layer1_echo::filter::MetadataFilter;
use crate::layer1_echo::similarity::top_k_similar;
use crate::layer1_echo::traits::{IndexedDocument, SimilarityMetric, VectorStore};
use crate::types::{Document, DocumentId, SearchResult};

// ---------------------------------------------------------------------------
// Table definition
// ---------------------------------------------------------------------------

/// The single redb table that stores all persisted documents.
const DOCS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("documents");

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

/// Flattened, fully-serializable mirror of [`IndexedDocument`].
///
/// We avoid storing `DateTime<Utc>` directly as a byte-level representation
/// and instead convert to RFC-3339 strings so the serialised format is both
/// human-readable and version-stable.
#[derive(Debug, Serialize, Deserialize)]
struct PersistedDocument {
    document_id: String,
    document_content: String,
    document_title: Option<String>,
    document_source: Option<String>,
    document_metadata: HashMap<String, String>,
    /// [`DateTime<Utc>`] serialised to RFC-3339 (e.g. `"2024-01-01T00:00:00Z"`).
    created_at_rfc3339: String,
    /// [`DateTime<Utc>`] serialised to RFC-3339.
    updated_at_rfc3339: String,
    embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// A vector store that persists every document and embedding to disk using
/// [`redb`](https://docs.rs/redb) — a pure-Rust, ACID-compliant embedded
/// database.
///
/// # Persistence guarantee
///
/// Every successful [`VectorStore::insert`] / [`VectorStore::upsert`] /
/// [`VectorStore::delete`] / [`VectorStore::update`] / [`VectorStore::clear`]
/// call commits a `redb` write transaction before returning, so the store is
/// durable even if the process terminates immediately afterwards.
///
/// # Thread safety
///
/// `redb::Database` is `Send + Sync`.  The store itself is `Send + Sync` and
/// can therefore be used behind an `Arc<Mutex<…>>` when shared across async
/// tasks.
pub struct RedbVectorStore {
    /// The underlying redb database handle.
    db: Database,
    /// Expected embedding dimension for all stored vectors.
    dimension: usize,
    /// Similarity metric used during search.
    metric: SimilarityMetric,
    /// Cached document count; kept in sync with the database.
    count: usize,
}

impl RedbVectorStore {
    /// Open (or create) a `RedbVectorStore` at the given filesystem path.
    ///
    /// If the file does not exist it is created.  If it already exists the
    /// existing data is preserved and `count` is initialised by counting the
    /// entries currently in the table.
    ///
    /// # Errors
    ///
    /// Returns [`VectorStoreError::StorageError`] when the database cannot be
    /// opened or the initial table inspection fails.
    pub fn new(path: impl AsRef<Path>, dimension: usize) -> Result<Self, VectorStoreError> {
        let db =
            Database::create(path).map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        // Ensure the table exists (creates it on first open).
        {
            let write_txn = db
                .begin_write()
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            // Opening (and immediately dropping) the table is enough to
            // create the table definition in the database.
            {
                let _table = write_txn
                    .open_table(DOCS_TABLE)
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
                // `_table` is dropped here, releasing the borrow on `write_txn`.
            }
            write_txn
                .commit()
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        }

        // Count pre-existing entries so that `self.count` is correct when
        // reopening an existing store.
        let count = {
            let read_txn = db
                .begin_read()
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            let table = read_txn
                .open_table(DOCS_TABLE)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            usize::try_from(
                table
                    .len()
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?,
            )
            .unwrap_or(0)
        };

        Ok(Self {
            db,
            dimension,
            metric: SimilarityMetric::Cosine,
            count,
        })
    }

    /// Override the similarity metric used for vector search.
    ///
    /// The default metric is [`SimilarityMetric::Cosine`].
    #[must_use]
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Serialize an [`IndexedDocument`] into a [`PersistedDocument`].
    fn to_persisted(doc: &IndexedDocument) -> PersistedDocument {
        PersistedDocument {
            document_id: doc.document.id.as_str().to_string(),
            document_content: doc.document.content.clone(),
            document_title: doc.document.title.clone(),
            document_source: doc.document.source.clone(),
            document_metadata: doc.document.metadata.clone(),
            created_at_rfc3339: doc.document.created_at.to_rfc3339(),
            updated_at_rfc3339: doc.document.updated_at.to_rfc3339(),
            embedding: doc.embedding.clone(),
        }
    }

    /// Deserialize a [`PersistedDocument`] back into an [`IndexedDocument`].
    ///
    /// # Errors
    ///
    /// Returns a [`VectorStoreError::StorageError`] when the timestamps cannot
    /// be parsed (should never happen for data we wrote ourselves).
    fn from_persisted(p: PersistedDocument) -> Result<IndexedDocument, VectorStoreError> {
        let created_at = p.created_at_rfc3339.parse::<DateTime<Utc>>().map_err(|e| {
            VectorStoreError::StorageError(format!("bad created_at timestamp: {e}"))
        })?;
        let updated_at = p.updated_at_rfc3339.parse::<DateTime<Utc>>().map_err(|e| {
            VectorStoreError::StorageError(format!("bad updated_at timestamp: {e}"))
        })?;

        let doc = Document {
            id: DocumentId::from_string(p.document_id),
            content: p.document_content,
            title: p.document_title,
            source: p.document_source,
            metadata: p.document_metadata,
            created_at,
            updated_at,
        };
        Ok(IndexedDocument::new(doc, p.embedding))
    }

    /// Load every document currently in the database.
    ///
    /// This is used by the search methods.  For workloads with millions of
    /// vectors an index structure would be needed, but that is out of scope
    /// for the current implementation.
    ///
    /// # Errors
    ///
    /// Returns a [`VectorStoreError::StorageError`] on any I/O or
    /// deserialisation failure.
    fn load_all_documents(&self) -> Result<Vec<IndexedDocument>, VectorStoreError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let table = read_txn
            .open_table(DOCS_TABLE)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let mut docs = Vec::new();
        for entry in table
            .iter()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
        {
            let (_key, value) = entry.map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            let persisted: PersistedDocument = serde_json::from_slice(value.value())
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            docs.push(Self::from_persisted(persisted)?);
        }
        Ok(docs)
    }

    /// Fetch a single raw entry by key without touching the count cache.
    fn fetch_raw(&self, id: &DocumentId) -> Result<Option<IndexedDocument>, VectorStoreError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let table = read_txn
            .open_table(DOCS_TABLE)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        match table
            .get(id.as_str())
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
        {
            Some(guard) => {
                let persisted: PersistedDocument = serde_json::from_slice(guard.value())
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
                Ok(Some(Self::from_persisted(persisted)?))
            }
            None => Ok(None),
        }
    }
}

// ---------------------------------------------------------------------------
// VectorStore implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl VectorStore for RedbVectorStore {
    /// Insert a new document into the store.
    ///
    /// # Errors
    ///
    /// - [`VectorStoreError::DimensionMismatch`] — embedding length differs
    ///   from the store's configured dimension.
    /// - [`VectorStoreError::DuplicateId`] — a document with the same ID
    ///   already exists.
    /// - [`VectorStoreError::StorageError`] — underlying I/O failure.
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError> {
        if doc.embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: doc.embedding.len(),
            });
        }

        // Duplicate-ID check.
        if self.fetch_raw(&doc.document.id)?.is_some() {
            return Err(VectorStoreError::DuplicateId(
                doc.document.id.as_str().to_string(),
            ));
        }

        let persisted = Self::to_persisted(&doc);
        let bytes = serde_json::to_vec(&persisted)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(DOCS_TABLE)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            table
                .insert(doc.document.id.as_str(), bytes.as_slice())
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        self.count += 1;
        Ok(())
    }

    /// Insert a batch of documents, stopping on the first error.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by [`VectorStore::insert`].
    async fn insert_batch(&mut self, docs: Vec<IndexedDocument>) -> Result<(), VectorStoreError> {
        for doc in docs {
            self.insert(doc).await?;
        }
        Ok(())
    }

    /// Retrieve a document by its ID.
    ///
    /// Returns `Ok(None)` when no document with the given ID exists.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::StorageError`] on I/O or deserialisation failure.
    async fn get(&self, id: &DocumentId) -> Result<Option<IndexedDocument>, VectorStoreError> {
        self.fetch_raw(id)
    }

    /// Delete a document by its ID.
    ///
    /// Returns `Ok(true)` when the document existed and was removed, or
    /// `Ok(false)` when no document with that ID was found.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::StorageError`] on I/O failure.
    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let removed = {
            let mut table = write_txn
                .open_table(DOCS_TABLE)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            table
                .remove(id.as_str())
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
                .is_some()
        };
        write_txn
            .commit()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        if removed {
            self.count = self.count.saturating_sub(1);
        }
        Ok(removed)
    }

    /// Replace the embedding of an existing document.
    ///
    /// Returns `Ok(true)` on success.
    ///
    /// # Errors
    ///
    /// - [`VectorStoreError::DimensionMismatch`] — new embedding has wrong
    ///   length.
    /// - [`VectorStoreError::NotFound`] — no document with the given ID.
    /// - [`VectorStoreError::StorageError`] — underlying I/O failure.
    async fn update(
        &mut self,
        id: &DocumentId,
        embedding: Vec<f32>,
    ) -> Result<bool, VectorStoreError> {
        if embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let mut existing = self
            .fetch_raw(id)?
            .ok_or_else(|| VectorStoreError::NotFound(id.as_str().to_string()))?;

        existing.embedding = embedding;
        existing.document.updated_at = Utc::now();

        let persisted = Self::to_persisted(&existing);
        let bytes = serde_json::to_vec(&persisted)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(DOCS_TABLE)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            table
                .insert(id.as_str(), bytes.as_slice())
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(true)
    }

    /// Insert or update a document (upsert).
    ///
    /// Returns `Ok(true)` when the document was newly inserted, or `Ok(false)`
    /// when an existing document was updated.
    ///
    /// # Errors
    ///
    /// - [`VectorStoreError::DimensionMismatch`] — embedding has wrong length.
    /// - [`VectorStoreError::StorageError`] — underlying I/O failure.
    async fn upsert(&mut self, doc: IndexedDocument) -> Result<bool, VectorStoreError> {
        if doc.embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: doc.embedding.len(),
            });
        }

        let is_insert = self.fetch_raw(&doc.document.id)?.is_none();

        let persisted = Self::to_persisted(&doc);
        let bytes = serde_json::to_vec(&persisted)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(DOCS_TABLE)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            table
                .insert(doc.document.id.as_str(), bytes.as_slice())
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        if is_insert {
            self.count += 1;
        }
        Ok(is_insert)
    }

    /// Search for the `top_k` most similar documents to `query_embedding`.
    ///
    /// All documents are loaded from the database and scored in memory using
    /// [`top_k_similar`].  A `min_score` threshold filters out low-confidence
    /// matches.
    ///
    /// # Errors
    ///
    /// - [`VectorStoreError::DimensionMismatch`] — query has wrong length.
    /// - [`VectorStoreError::StorageError`] — I/O or deserialisation failure.
    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query_embedding.len(),
            });
        }

        let all_docs = self.load_all_documents()?;
        if all_docs.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings: Vec<Vec<f32>> = all_docs.iter().map(|d| d.embedding.clone()).collect();
        let top_indices =
            top_k_similar(query_embedding, &embeddings, top_k, self.metric, min_score);

        let results = top_indices
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, score))| {
                SearchResult::new(all_docs[idx].document.clone(), score, rank)
            })
            .collect();

        Ok(results)
    }

    /// Search for the `top_k` most similar documents, pre-filtering by
    /// `MetadataFilter` before scoring.
    ///
    /// Only documents whose metadata matches the filter are considered during
    /// similarity scoring.  When `filter` is `None` the behaviour is identical
    /// to [`VectorStore::search`].
    ///
    /// # Errors
    ///
    /// - [`VectorStoreError::DimensionMismatch`] — query has wrong length.
    /// - [`VectorStoreError::StorageError`] — I/O or deserialisation failure.
    async fn search_with_filter(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query_embedding.len(),
            });
        }

        let all_docs = self.load_all_documents()?;
        if all_docs.is_empty() {
            return Ok(Vec::new());
        }

        let filtered: Vec<&IndexedDocument> = match filter {
            Some(f) => all_docs
                .iter()
                .filter(|d| f.matches(&d.document.metadata))
                .collect(),
            None => all_docs.iter().collect(),
        };

        if filtered.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings: Vec<Vec<f32>> = filtered.iter().map(|d| d.embedding.clone()).collect();
        let top_indices =
            top_k_similar(query_embedding, &embeddings, top_k, self.metric, min_score);

        let results = top_indices
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, score))| {
                SearchResult::new(filtered[idx].document.clone(), score, rank)
            })
            .collect();

        Ok(results)
    }

    /// Return the number of documents currently stored.
    async fn count(&self) -> usize {
        self.count
    }

    /// Delete all documents from the store and reset the count to zero.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::StorageError`] on I/O failure.
    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(DOCS_TABLE)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

            // Collect all keys first, then delete them — avoids mutating
            // the table while iterating the same cursor.
            let keys: Vec<String> = table
                .iter()
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
                .map(|entry| {
                    entry
                        .map(|(k, _v)| k.value().to_string())
                        .map_err(|e| VectorStoreError::StorageError(e.to_string()))
                })
                .collect::<Result<_, VectorStoreError>>()?;

            for key in &keys {
                table
                    .remove(key.as_str())
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            }
        }
        write_txn
            .commit()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        self.count = 0;
        Ok(())
    }

    /// Return the embedding dimension that this store was configured with.
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// Return the similarity metric used for vector search.
    fn similarity_metric(&self) -> SimilarityMetric {
        self.metric
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::pedantic)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::types::Document;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_store(dir: &TempDir, dim: usize) -> RedbVectorStore {
        RedbVectorStore::new(dir.path().join("store.redb"), dim)
            .expect("store creation should succeed")
    }

    fn make_doc(content: &str, embedding: Vec<f32>) -> IndexedDocument {
        IndexedDocument::new(Document::new(content), embedding)
    }

    fn make_doc_with_meta(
        content: &str,
        embedding: Vec<f32>,
        meta: &[(&str, &str)],
    ) -> IndexedDocument {
        let mut doc = Document::new(content);
        for (k, v) in meta {
            doc = doc.with_metadata(*k, *v);
        }
        IndexedDocument::new(doc, embedding)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_redb_vector_store_insert_and_get() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 3);

        let doc = make_doc("hello world", vec![1.0, 0.0, 0.0]);
        let id = doc.document.id.clone();

        store.insert(doc).await.expect("insert should succeed");

        let retrieved = store.get(&id).await.expect("get should succeed");
        assert!(retrieved.is_some(), "document should be found");

        let retrieved = retrieved.expect("checked above");
        assert_eq!(retrieved.document.content, "hello world");
        assert_eq!(retrieved.embedding, vec![1.0, 0.0, 0.0]);
        assert_eq!(store.count().await, 1);
    }

    #[tokio::test]
    async fn test_redb_vector_store_dimension_mismatch() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 3);

        // Embedding has 2 components but store expects 3.
        let doc = make_doc("bad dim", vec![1.0, 0.0]);
        let result = store.insert(doc).await;

        assert!(
            matches!(
                result,
                Err(VectorStoreError::DimensionMismatch {
                    expected: 3,
                    actual: 2
                })
            ),
            "expected DimensionMismatch"
        );
    }

    #[tokio::test]
    async fn test_redb_vector_store_duplicate_id_error() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 2);

        let doc1 = make_doc("first", vec![1.0, 0.0]);
        let id = doc1.document.id.clone();

        store
            .insert(doc1)
            .await
            .expect("first insert should succeed");

        // Attempt to insert a different document under the same ID.
        let mut doc2 = make_doc("second", vec![0.0, 1.0]);
        doc2.document.id = id;

        let result = store.insert(doc2).await;
        assert!(
            matches!(result, Err(VectorStoreError::DuplicateId(_))),
            "expected DuplicateId error"
        );
    }

    #[tokio::test]
    async fn test_redb_vector_store_delete() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 2);

        let doc = make_doc("to be deleted", vec![1.0, 0.0]);
        let id = doc.document.id.clone();

        store.insert(doc).await.expect("insert should succeed");
        assert_eq!(store.count().await, 1);

        let deleted = store.delete(&id).await.expect("delete should succeed");
        assert!(deleted, "delete should return true for existing document");
        assert_eq!(store.count().await, 0);

        // Deleting a non-existent document should return false.
        let deleted_again = store
            .delete(&id)
            .await
            .expect("second delete should not error");
        assert!(!deleted_again, "second delete should return false");
    }

    #[tokio::test]
    async fn test_redb_vector_store_update() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 2);

        let doc = make_doc("update me", vec![1.0, 0.0]);
        let id = doc.document.id.clone();

        store.insert(doc).await.expect("insert should succeed");

        // Update the embedding.
        let updated = store
            .update(&id, vec![0.0, 1.0])
            .await
            .expect("update should succeed");
        assert!(updated, "update should return true");

        let retrieved = store
            .get(&id)
            .await
            .expect("get should succeed")
            .expect("document should exist");
        assert_eq!(retrieved.embedding, vec![0.0, 1.0]);

        // Update on a non-existent ID should return NotFound.
        let missing_id = DocumentId::new();
        let result = store.update(&missing_id, vec![0.5, 0.5]).await;
        assert!(
            matches!(result, Err(VectorStoreError::NotFound(_))),
            "expected NotFound"
        );
    }

    #[tokio::test]
    async fn test_redb_vector_store_upsert_insert() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 2);

        let doc = make_doc("new doc", vec![1.0, 0.0]);
        let id = doc.document.id.clone();

        let was_insert = store.upsert(doc).await.expect("upsert should succeed");
        assert!(was_insert, "upsert of new doc should return true");
        assert_eq!(store.count().await, 1);

        let retrieved = store.get(&id).await.expect("get should succeed");
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_redb_vector_store_upsert_update() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 2);

        let doc = make_doc("original", vec![1.0, 0.0]);
        let id = doc.document.id.clone();
        store.insert(doc).await.expect("insert should succeed");

        // Upsert with same ID — should be an update.
        let mut updated_doc = make_doc("updated content", vec![0.0, 1.0]);
        updated_doc.document.id = id.clone();

        let was_insert = store
            .upsert(updated_doc)
            .await
            .expect("upsert should succeed");
        assert!(!was_insert, "upsert of existing doc should return false");
        assert_eq!(
            store.count().await,
            1,
            "count should not increase on update"
        );

        let retrieved = store
            .get(&id)
            .await
            .expect("get should succeed")
            .expect("document should exist");
        // The new embedding was persisted.
        assert_eq!(retrieved.embedding, vec![0.0, 1.0]);
    }

    #[tokio::test]
    async fn test_redb_vector_store_search() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 2);

        store
            .insert(make_doc("alpha", vec![1.0, 0.0]))
            .await
            .expect("insert alpha");
        store
            .insert(make_doc("beta", vec![0.0, 1.0]))
            .await
            .expect("insert beta");
        store
            .insert(make_doc("gamma", vec![0.8_f32, 0.6_f32]))
            .await
            .expect("insert gamma");

        // Query closest to alpha.
        let results = store
            .search(&[1.0, 0.0], 2, None)
            .await
            .expect("search should succeed");

        assert_eq!(results.len(), 2, "should return 2 results");
        assert_eq!(
            results[0].document.content, "alpha",
            "alpha should rank first"
        );
        assert_eq!(results[0].rank, 0, "first result should have rank 0");
        assert!(
            (results[0].score - 1.0_f32).abs() < 1e-5,
            "score for identical direction should be ~1.0"
        );
    }

    #[tokio::test]
    async fn test_redb_vector_store_search_with_filter() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 2);

        store
            .insert(make_doc_with_meta(
                "science doc",
                vec![1.0, 0.0],
                &[("category", "science")],
            ))
            .await
            .expect("insert science");

        store
            .insert(make_doc_with_meta(
                "tech doc",
                vec![0.9_f32, 0.1_f32],
                &[("category", "technology")],
            ))
            .await
            .expect("insert tech");

        store
            .insert(make_doc_with_meta(
                "art doc",
                vec![0.8_f32, 0.2_f32],
                &[("category", "art")],
            ))
            .await
            .expect("insert art");

        let filter = MetadataFilter::eq("category", "science");
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .expect("search_with_filter should succeed");

        assert_eq!(results.len(), 1, "only science doc should match filter");
        assert_eq!(results[0].document.content, "science doc");
    }

    #[tokio::test]
    async fn test_redb_vector_store_clear() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp, 2);

        store
            .insert(make_doc("doc1", vec![1.0, 0.0]))
            .await
            .expect("insert doc1");
        store
            .insert(make_doc("doc2", vec![0.0, 1.0]))
            .await
            .expect("insert doc2");

        assert_eq!(store.count().await, 2);

        store.clear().await.expect("clear should succeed");
        assert_eq!(store.count().await, 0, "count should be 0 after clear");

        // The store should be usable after clearing.
        store
            .insert(make_doc("fresh", vec![1.0, 0.0]))
            .await
            .expect("insert after clear should succeed");
        assert_eq!(store.count().await, 1);
    }

    #[tokio::test]
    async fn test_redb_vector_store_persistence() {
        let tmp = TempDir::new().expect("tempdir");
        let db_path = tmp.path().join("persist.redb");

        let doc_id = {
            // Scope: create store, insert, and drop it.
            let mut store = RedbVectorStore::new(&db_path, 3).expect("first open should succeed");
            let doc = make_doc("persisted content", vec![0.5, 0.5, 0.0]);
            let id = doc.document.id.clone();
            store.insert(doc).await.expect("insert should succeed");
            assert_eq!(store.count().await, 1);
            id
            // `store` is dropped here — the database file remains on disk.
        };

        // Reopen from the same path and verify the document is still there.
        let store = RedbVectorStore::new(&db_path, 3).expect("second open should succeed");
        assert_eq!(store.count().await, 1, "count should survive reopen");

        let retrieved = store
            .get(&doc_id)
            .await
            .expect("get should succeed")
            .expect("document should still exist after reopen");

        assert_eq!(retrieved.document.content, "persisted content");
        assert_eq!(retrieved.embedding, vec![0.5, 0.5, 0.0]);
    }
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::pedantic)]
mod prop_tests {
    use proptest::prelude::*;
    use tempfile::TempDir;

    use super::*;
    use crate::types::Document;

    // -----------------------------------------------------------------------
    // Strategies
    // -----------------------------------------------------------------------

    /// Strategy: generate an embedding of exactly `dim` f32 values in [-1, 1).
    fn arb_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-1.0f32..1.0f32, dim..=dim)
    }

    /// Strategy: generate an 8–16 character alphanumeric document ID string.
    fn arb_doc_id_str() -> impl Strategy<Value = String> {
        "[a-z0-9]{8,16}".prop_map(|s| s)
    }

    /// Create a fresh store backed by a unique file per test invocation.
    fn open_prop_store(dir: &TempDir, dim: usize, suffix: &str) -> RedbVectorStore {
        RedbVectorStore::new(dir.path().join(format!("prop_{suffix}.redb")), dim)
            .expect("RedbVectorStore::new must succeed in proptest")
    }

    fn make_indexed_doc(id_str: &str, content: &str, embedding: Vec<f32>) -> IndexedDocument {
        let doc = Document::new(content).with_id(id_str);
        IndexedDocument::new(doc, embedding)
    }

    // -----------------------------------------------------------------------
    // Test 1 – insert then get returns same content and embedding
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_insert_and_get_roundtrip(
            id_str   in arb_doc_id_str(),
            content  in "[a-z ]{5,40}",
            embedding in arb_embedding(4),
        ) {
            let dir = TempDir::new().expect("tempdir must be created");
            let mut store = open_prop_store(&dir, 4, "rtrip");

            let doc = make_indexed_doc(&id_str, &content, embedding.clone());
            let doc_id = doc.document.id.clone();

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                store.insert(doc).await.expect("insert must succeed");

                let result = store.get(&doc_id).await.expect("get must not error");
                prop_assert!(result.is_some(), "get must return Some after insert");

                let retrieved = result.expect("checked");
                prop_assert_eq!(
                    &retrieved.document.content,
                    &content,
                    "document content must survive the insert/get round-trip"
                );
                prop_assert_eq!(
                    retrieved.embedding.len(),
                    embedding.len(),
                    "embedding length must be preserved"
                );
                for (i, (got, exp)) in retrieved.embedding.iter().zip(embedding.iter()).enumerate() {
                    prop_assert!(
                        (got - exp).abs() < f32::EPSILON,
                        "embedding[{i}] mismatch: {got} vs {exp}"
                    );
                }
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 – inserting the same doc_id twice returns DuplicateId
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_duplicate_id_fails(
            id_str    in arb_doc_id_str(),
            embedding in arb_embedding(4),
        ) {
            let dir = TempDir::new().expect("tempdir must be created");
            let mut store = open_prop_store(&dir, 4, "dup");

            let doc1 = make_indexed_doc(&id_str, "first", embedding.clone());
            let mut doc2 = make_indexed_doc(&id_str, "second", embedding);
            // Force doc2 to use the exact same DocumentId as doc1.
            doc2.document.id = doc1.document.id.clone();

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                store.insert(doc1).await.expect("first insert must succeed");

                let result = store.insert(doc2).await;
                prop_assert!(
                    matches!(result, Err(VectorStoreError::DuplicateId(_))),
                    "second insert with the same id must return DuplicateId, got: {result:?}"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 – inserting a doc with the wrong embedding length returns DimensionMismatch
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_dimension_mismatch_fails(
            // Embedding shorter or longer than the configured dim (4).
            bad_len in prop::sample::select(vec![1usize, 2, 3, 5, 8, 16]),
        ) {
            let dir = TempDir::new().expect("tempdir must be created");
            let mut store = open_prop_store(&dir, 4, "dim");

            // Build a bad embedding of the wrong length.
            let bad_embedding = vec![0.5_f32; bad_len];
            let doc = make_indexed_doc("dim_mismatch_id", "content", bad_embedding);

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                let result = store.insert(doc).await;
                prop_assert!(
                    matches!(
                        result,
                        Err(VectorStoreError::DimensionMismatch { expected: 4, actual: _ })
                    ),
                    "inserting embedding of wrong length must return DimensionMismatch, got: {result:?}"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 – delete returns Ok(true) for an existing document
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_delete_returns_true_when_exists(
            id_str    in arb_doc_id_str(),
            embedding in arb_embedding(4),
        ) {
            let dir = TempDir::new().expect("tempdir must be created");
            let mut store = open_prop_store(&dir, 4, "del_true");

            let doc = make_indexed_doc(&id_str, "to delete", embedding);
            let doc_id = doc.document.id.clone();

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                store.insert(doc).await.expect("insert must succeed");

                let deleted = store.delete(&doc_id).await.expect("delete must not error");
                prop_assert!(deleted, "delete must return true for an existing document");
                prop_assert_eq!(
                    store.count().await,
                    0,
                    "count must be 0 after deleting the only document"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 – delete returns Ok(false) for a missing document
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_delete_returns_false_when_missing(
            id_str in arb_doc_id_str(),
        ) {
            let dir = TempDir::new().expect("tempdir must be created");
            let mut store = open_prop_store(&dir, 4, "del_false");

            let absent_id = DocumentId::from_string(&id_str);

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                let deleted = store.delete(&absent_id).await.expect("delete must not error");
                prop_assert!(!deleted, "delete must return false for a missing document");
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 – count() matches the number of successful inserts
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_count_matches_insertions(
            // Generate 1..=8 distinct id strings (HashSet guarantees uniqueness).
            id_strs in prop::collection::hash_set(arb_doc_id_str(), 1..=8usize),
        ) {
            let dir = TempDir::new().expect("tempdir must be created");
            let mut store = open_prop_store(&dir, 4, "count");
            let n = id_strs.len();

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                for (idx, id_str) in id_strs.into_iter().enumerate() {
                    let embedding = vec![idx as f32 * 0.1; 4];
                    let doc = make_indexed_doc(&id_str, "doc content", embedding);
                    store.insert(doc).await.expect("each insert must succeed");
                }
                prop_assert_eq!(
                    store.count().await,
                    n,
                    "count() must equal the number of successful inserts"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 – search returns at most top_k results
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_search_returns_at_most_top_k(
            // 2..=8 distinct id strings for documents
            id_strs in prop::collection::hash_set(arb_doc_id_str(), 2..=8usize),
            top_k   in 1usize..=5usize,
        ) {
            let dir = TempDir::new().expect("tempdir must be created");
            let mut store = open_prop_store(&dir, 4, "srch");

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                let mut count = 0usize;
                for id_str in &id_strs {
                    let embedding = vec![0.25_f32; 4];
                    let doc = make_indexed_doc(id_str, "doc", embedding);
                    if store.insert(doc).await.is_ok() {
                        count += 1;
                    }
                }

                let query = vec![1.0_f32; 4];
                let results = store
                    .search(&query, top_k, None)
                    .await
                    .expect("search must not error");

                prop_assert!(
                    results.len() <= top_k,
                    "search must return at most top_k={top_k} results, got {}",
                    results.len()
                );
                prop_assert!(
                    results.len() <= count,
                    "search must not return more results than documents inserted"
                );
                Ok(())
            })?;
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 – upsert returns true on new id, false on second call
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_upsert_insert_returns_true(
            id_str    in arb_doc_id_str(),
            embedding in arb_embedding(4),
        ) {
            let dir = TempDir::new().expect("tempdir must be created");
            let mut store = open_prop_store(&dir, 4, "upsert");

            let doc1 = make_indexed_doc(&id_str, "first content", embedding.clone());
            let doc1_id = doc1.document.id.clone();
            let mut doc2 = make_indexed_doc(&id_str, "second content", embedding);
            doc2.document.id = doc1_id.clone();

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime must start");
            rt.block_on(async {
                // First upsert on a new id must return true (was an insert).
                let first = store.upsert(doc1).await.expect("first upsert must succeed");
                prop_assert!(first, "first upsert must return true (new document)");
                prop_assert_eq!(store.count().await, 1, "count must be 1 after first upsert");

                // Second upsert on the same id must return false (was an update).
                let second = store.upsert(doc2).await.expect("second upsert must succeed");
                prop_assert!(!second, "second upsert must return false (existing document)");
                prop_assert_eq!(
                    store.count().await,
                    1,
                    "count must remain 1 after an update upsert"
                );
                Ok(())
            })?;
        }
    }
}
