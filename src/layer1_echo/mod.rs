//! Layer 1: Echo - Semantic search with vector embeddings.
//!
//! The Echo layer provides semantic search capabilities using:
//! - Embedding providers to convert text to vectors
//! - Vector stores to index and search documents
//! - Similarity metrics for comparing vectors
//! - Approximate nearest neighbor (ANN) search using HNSW

pub mod ann;
pub mod embedding;
pub mod filter;
pub mod multi_vector;
pub mod similarity;
pub mod storage;
pub mod traits;

pub use ann::{AnnConfig, AnnStats, AnnVectorStore, HnswIndex, HnswNode};
pub use embedding::{
    CacheStats, CachedEmbeddingProvider, CandleDevice, CandleEmbeddingConfig, EmbeddingCacheConfig,
    MockEmbeddingProvider,
};
pub use filter::MetadataFilter;
pub use multi_vector::{
    InMemoryMultiVectorStore, MaxSimScore, MockTokenEmbeddingProvider, MultiVectorDocument,
    MultiVectorSearchResult, MultiVectorStore, TokenEmbedding, TokenEmbeddingProvider, TokenMatch,
};
pub use similarity::{
    batch_similarities, compute_similarity, cosine_similarity, dot_product,
    euclidean_to_similarity, normalize, top_k_similar,
};
pub use storage::InMemoryVectorStore;
pub use traits::{Echo, EmbeddingProvider, IndexedDocument, SimilarityMetric, VectorStore};

#[cfg(feature = "speculator")]
pub use embedding::CandleEmbeddingProvider;

use async_trait::async_trait;

use crate::error::EmbeddingError;
use crate::types::{Document, DocumentId, SearchResult};

/// The default Echo implementation combining an embedding provider and vector store.
pub struct EchoLayer<E: EmbeddingProvider, V: VectorStore> {
    embedding_provider: E,
    vector_store: V,
}

impl<E: EmbeddingProvider, V: VectorStore> EchoLayer<E, V> {
    /// Create a new Echo layer with the given embedding provider and vector store.
    #[must_use]
    pub fn new(embedding_provider: E, vector_store: V) -> Self {
        Self {
            embedding_provider,
            vector_store,
        }
    }

    /// Get a reference to the embedding provider.
    #[must_use]
    pub fn embedding_provider(&self) -> &E {
        &self.embedding_provider
    }

    /// Get a reference to the vector store.
    #[must_use]
    pub fn vector_store(&self) -> &V {
        &self.vector_store
    }

    /// Update an existing document (re-embeds the content).
    /// Returns `Ok(true)` if the document was updated, or an error if not found.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Embedding generation fails
    /// - The document does not exist in the store
    pub async fn update_document(&mut self, doc: &Document) -> Result<bool, EmbeddingError> {
        let embedding = self.embedding_provider.embed(&doc.content).await?;

        self.vector_store
            .update(&doc.id, embedding)
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))
    }

    /// Insert or update a document (upsert).
    /// Returns `Ok(true)` if inserted, `Ok(false)` if updated.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Embedding generation fails
    /// - The vector store operation fails (e.g., capacity exceeded for inserts)
    pub async fn upsert_document(&mut self, doc: Document) -> Result<bool, EmbeddingError> {
        let embedding = self.embedding_provider.embed(&doc.content).await?;
        let indexed = IndexedDocument::new(doc, embedding);

        self.vector_store
            .upsert(indexed)
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))
    }

    /// Search for documents similar to the query with metadata filtering.
    ///
    /// This method combines semantic search with metadata filtering, returning
    /// only documents that match both the query similarity and the provided filter.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to search for.
    /// * `top_k` - Maximum number of results to return.
    /// * `min_score` - Optional minimum similarity score threshold.
    /// * `filter` - Optional metadata filter to apply.
    ///
    /// # Returns
    ///
    /// A vector of search results matching both similarity and filter criteria.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation or vector search fails.
    pub async fn search_with_filter(
        &self,
        query: &str,
        top_k: usize,
        min_score: Option<f32>,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, EmbeddingError> {
        let query_embedding = self.embedding_provider.embed(query).await?;

        self.vector_store
            .search_with_filter(&query_embedding, top_k, min_score, filter)
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))
    }
}

#[async_trait]
impl<E: EmbeddingProvider, V: VectorStore> Echo for EchoLayer<E, V> {
    async fn index(&mut self, document: Document) -> Result<DocumentId, EmbeddingError> {
        let embedding = self.embedding_provider.embed(&document.content).await?;
        let id = document.id.clone();
        let indexed = IndexedDocument::new(document, embedding);

        self.vector_store
            .insert(indexed)
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))?;

        Ok(id)
    }

    async fn index_batch(
        &mut self,
        documents: Vec<Document>,
    ) -> Result<Vec<DocumentId>, EmbeddingError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let contents: Vec<&str> = documents.iter().map(|d| d.content.as_str()).collect();
        let embeddings = self.embedding_provider.embed_batch(&contents).await?;

        let ids: Vec<DocumentId> = documents.iter().map(|d| d.id.clone()).collect();

        let indexed: Vec<IndexedDocument> = documents
            .into_iter()
            .zip(embeddings)
            .map(|(doc, emb)| IndexedDocument::new(doc, emb))
            .collect();

        self.vector_store
            .insert_batch(indexed)
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))?;

        Ok(ids)
    }

    async fn search(
        &self,
        query: &str,
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, EmbeddingError> {
        let query_embedding = self.embedding_provider.embed(query).await?;

        self.vector_store
            .search(&query_embedding, top_k, min_score)
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<Document>, EmbeddingError> {
        let indexed = self
            .vector_store
            .get(id)
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))?;

        Ok(indexed.map(|i| i.document))
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, EmbeddingError> {
        self.vector_store
            .delete(id)
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))
    }

    async fn count(&self) -> usize {
        self.vector_store.count().await
    }

    async fn clear(&mut self) -> Result<(), EmbeddingError> {
        self.vector_store
            .clear()
            .await
            .map_err(|e| EmbeddingError::Backend(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_layer_index_and_search() {
        let provider = MockEmbeddingProvider::new(64);
        let store = InMemoryVectorStore::new(64);
        let mut echo = EchoLayer::new(provider, store);

        // Index some documents
        let doc1 = Document::new("The quick brown fox");
        let doc2 = Document::new("A lazy dog sleeps");
        let doc3 = Document::new("The quick brown dog");

        echo.index(doc1).await.unwrap();
        echo.index(doc2).await.unwrap();
        echo.index(doc3).await.unwrap();

        assert_eq!(echo.count().await, 3);

        // Search
        let results = echo.search("quick fox", 2, None).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_echo_layer_batch_index() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        let docs = vec![
            Document::new("Document one"),
            Document::new("Document two"),
            Document::new("Document three"),
        ];

        let ids = echo.index_batch(docs).await.unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(echo.count().await, 3);
    }

    #[tokio::test]
    async fn test_echo_layer_get_and_delete() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        let doc = Document::new("Test document");
        let id = doc.id.clone();

        echo.index(doc).await.unwrap();

        // Get
        let retrieved = echo.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test document");

        // Delete
        let deleted = echo.delete(&id).await.unwrap();
        assert!(deleted);

        // Verify deleted
        let retrieved = echo.get(&id).await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_echo_layer_clear() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        echo.index(Document::new("doc1")).await.unwrap();
        echo.index(Document::new("doc2")).await.unwrap();

        assert_eq!(echo.count().await, 2);

        echo.clear().await.unwrap();
        assert_eq!(echo.count().await, 0);
    }

    #[tokio::test]
    async fn test_echo_layer_update_document() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        let doc = Document::new("original content");
        let id = doc.id.clone();

        echo.index(doc).await.unwrap();

        // Create updated document with same ID
        let mut updated_doc = Document::new("updated content");
        updated_doc.id = id.clone();

        let updated = echo.update_document(&updated_doc).await.unwrap();
        assert!(updated);

        // The document content in the store is not changed by update
        // (only the embedding is updated), so we verify the count is still 1
        assert_eq!(echo.count().await, 1);
    }

    #[tokio::test]
    async fn test_echo_layer_update_nonexistent_document() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        let doc = Document::new("content");
        // Document not indexed, so update should fail
        let result = echo.update_document(&doc).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_echo_layer_upsert_new_document() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        let doc = Document::new("new document");
        let id = doc.id.clone();

        let inserted = echo.upsert_document(doc).await.unwrap();
        assert!(inserted); // Should return true for insert

        assert_eq!(echo.count().await, 1);

        let retrieved = echo.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "new document");
    }

    #[tokio::test]
    async fn test_echo_layer_upsert_existing_document() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        let doc = Document::new("original document");
        let id = doc.id.clone();

        echo.index(doc).await.unwrap();

        // Create updated document with same ID
        let mut updated_doc = Document::new("updated document");
        updated_doc.id = id.clone();

        let inserted = echo.upsert_document(updated_doc).await.unwrap();
        assert!(!inserted); // Should return false for update

        assert_eq!(echo.count().await, 1);

        let retrieved = echo.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "updated document");
    }

    #[tokio::test]
    async fn test_echo_layer_search_with_filter() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        // Index documents with different categories
        echo.index(
            Document::new("Science article about physics").with_metadata("category", "science"),
        )
        .await
        .unwrap();
        echo.index(
            Document::new("Technology news about AI").with_metadata("category", "technology"),
        )
        .await
        .unwrap();
        echo.index(Document::new("Art exhibition review").with_metadata("category", "art"))
            .await
            .unwrap();

        // Search with filter for science category
        let filter = MetadataFilter::eq("category", "science");
        let results = echo
            .search_with_filter("physics", 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].document.metadata.get("category"),
            Some(&"science".to_string())
        );
    }

    #[tokio::test]
    async fn test_echo_layer_search_with_filter_complex() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        // Index documents with multiple metadata fields
        echo.index(
            Document::new("Published science paper")
                .with_metadata("category", "science")
                .with_metadata("status", "published"),
        )
        .await
        .unwrap();
        echo.index(
            Document::new("Draft science paper")
                .with_metadata("category", "science")
                .with_metadata("status", "draft"),
        )
        .await
        .unwrap();
        echo.index(
            Document::new("Published tech blog")
                .with_metadata("category", "technology")
                .with_metadata("status", "published"),
        )
        .await
        .unwrap();

        // Search for published science documents
        let filter = MetadataFilter::and(vec![
            MetadataFilter::eq("category", "science"),
            MetadataFilter::eq("status", "published"),
        ]);
        let results = echo
            .search_with_filter("paper", 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].document.content.contains("Published science"));
    }

    #[tokio::test]
    async fn test_echo_layer_search_with_filter_none() {
        let provider = MockEmbeddingProvider::new(32);
        let store = InMemoryVectorStore::new(32);
        let mut echo = EchoLayer::new(provider, store);

        echo.index(Document::new("doc1").with_metadata("cat", "a"))
            .await
            .unwrap();
        echo.index(Document::new("doc2").with_metadata("cat", "b"))
            .await
            .unwrap();

        // Search without filter should return all
        let results = echo
            .search_with_filter("doc", 10, None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }
}
