//! Traits for the Echo (semantic search) layer.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{EmbeddingError, VectorStoreError};
use crate::layer1_echo::filter::MetadataFilter;
use crate::types::{Document, DocumentId, SearchResult};

/// The similarity metric to use for vector comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SimilarityMetric {
    /// Cosine similarity: measures the angle between vectors.
    /// Range: [-1, 1], higher is more similar.
    #[default]
    Cosine,
    /// Euclidean distance: measures straight-line distance.
    /// Converted to similarity: 1 / (1 + distance).
    Euclidean,
    /// Dot product: sum of element-wise products.
    /// Requires normalized vectors for meaningful comparison.
    DotProduct,
}

/// Provider for generating embeddings from text.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding for a single text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embeddings for multiple texts in a batch.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Get the dimension of the embeddings produced by this provider.
    fn dimension(&self) -> usize;

    /// Get the model identifier.
    fn model_id(&self) -> &str;
}

/// An indexed document with its embedding vector.
#[derive(Debug, Clone)]
pub struct IndexedDocument {
    /// The document.
    pub document: Document,
    /// The embedding vector.
    pub embedding: Vec<f32>,
}

impl IndexedDocument {
    /// Create a new indexed document.
    #[must_use]
    pub fn new(document: Document, embedding: Vec<f32>) -> Self {
        Self {
            document,
            embedding,
        }
    }
}

/// Storage for document embeddings with similarity search.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert a document with its embedding.
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError>;

    /// Insert multiple documents with their embeddings.
    async fn insert_batch(&mut self, docs: Vec<IndexedDocument>) -> Result<(), VectorStoreError>;

    /// Get a document by its ID.
    async fn get(&self, id: &DocumentId) -> Result<Option<IndexedDocument>, VectorStoreError>;

    /// Delete a document by its ID.
    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError>;

    /// Update an existing document's embedding.
    /// Returns `Ok(true)` if the document was updated, or an error if not found.
    async fn update(
        &mut self,
        id: &DocumentId,
        embedding: Vec<f32>,
    ) -> Result<bool, VectorStoreError>;

    /// Insert or update a document (upsert).
    /// Returns `Ok(true)` if inserted, `Ok(false)` if updated.
    async fn upsert(&mut self, doc: IndexedDocument) -> Result<bool, VectorStoreError>;

    /// Search for similar documents.
    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;

    /// Search for similar documents with metadata filtering.
    ///
    /// This method combines vector similarity search with metadata filtering,
    /// returning only documents that match the provided filter.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query vector to search for.
    /// * `top_k` - Maximum number of results to return.
    /// * `min_score` - Optional minimum similarity score threshold.
    /// * `filter` - Optional metadata filter to apply.
    ///
    /// # Returns
    ///
    /// A vector of search results matching both the similarity and filter criteria.
    async fn search_with_filter(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;

    /// Get the number of documents in the store.
    async fn count(&self) -> usize;

    /// Clear all documents from the store.
    async fn clear(&mut self) -> Result<(), VectorStoreError>;

    /// Get the expected embedding dimension.
    fn dimension(&self) -> usize;

    /// Get the similarity metric used by this store.
    fn similarity_metric(&self) -> SimilarityMetric;
}

/// The Echo layer: combines embedding and vector search.
#[async_trait]
pub trait Echo: Send + Sync {
    /// Index a document.
    async fn index(&mut self, document: Document) -> Result<DocumentId, EmbeddingError>;

    /// Index multiple documents.
    async fn index_batch(
        &mut self,
        documents: Vec<Document>,
    ) -> Result<Vec<DocumentId>, EmbeddingError>;

    /// Search for documents similar to the query.
    async fn search(
        &self,
        query: &str,
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, EmbeddingError>;

    /// Get a document by ID.
    async fn get(&self, id: &DocumentId) -> Result<Option<Document>, EmbeddingError>;

    /// Delete a document by ID.
    async fn delete(&mut self, id: &DocumentId) -> Result<bool, EmbeddingError>;

    /// Get the number of indexed documents.
    async fn count(&self) -> usize;

    /// Clear all indexed documents.
    async fn clear(&mut self) -> Result<(), EmbeddingError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_metric_default() {
        let metric = SimilarityMetric::default();
        assert_eq!(metric, SimilarityMetric::Cosine);
    }

    #[test]
    fn test_indexed_document_creation() {
        let doc = Document::new("test content");
        let embedding = vec![0.1, 0.2, 0.3];
        let indexed = IndexedDocument::new(doc.clone(), embedding.clone());

        assert_eq!(indexed.document.content, "test content");
        assert_eq!(indexed.embedding, embedding);
    }
}
