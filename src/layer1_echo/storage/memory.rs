//! In-memory vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::error::VectorStoreError;
use crate::layer1_echo::filter::MetadataFilter;
use crate::layer1_echo::similarity::top_k_similar;
use crate::layer1_echo::traits::{IndexedDocument, SimilarityMetric, VectorStore};
use crate::types::{DocumentId, SearchResult};

/// An in-memory vector store for development and testing.
pub struct InMemoryVectorStore {
    /// The stored documents and embeddings.
    documents: RwLock<HashMap<DocumentId, IndexedDocument>>,
    /// The embedding dimension.
    dimension: usize,
    /// The similarity metric to use.
    metric: SimilarityMetric,
    /// Maximum capacity (0 = unlimited).
    max_capacity: usize,
}

impl InMemoryVectorStore {
    /// Create a new in-memory vector store.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
            dimension,
            metric: SimilarityMetric::Cosine,
            max_capacity: 0,
        }
    }

    /// Set the similarity metric.
    #[must_use]
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the maximum capacity.
    #[must_use]
    pub fn with_max_capacity(mut self, max_capacity: usize) -> Self {
        self.max_capacity = max_capacity;
        self
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError> {
        if doc.embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: doc.embedding.len(),
            });
        }

        let mut docs = self.documents.write().await;

        if self.max_capacity > 0 && docs.len() >= self.max_capacity {
            return Err(VectorStoreError::CapacityExceeded {
                max: self.max_capacity,
                current: docs.len(),
            });
        }

        if docs.contains_key(&doc.document.id) {
            return Err(VectorStoreError::DuplicateId(doc.document.id.to_string()));
        }

        docs.insert(doc.document.id.clone(), doc);
        Ok(())
    }

    async fn insert_batch(&mut self, batch: Vec<IndexedDocument>) -> Result<(), VectorStoreError> {
        for doc in batch {
            self.insert(doc).await?;
        }
        Ok(())
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<IndexedDocument>, VectorStoreError> {
        let docs = self.documents.read().await;
        Ok(docs.get(id).cloned())
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        let mut docs = self.documents.write().await;
        Ok(docs.remove(id).is_some())
    }

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

        let mut docs = self.documents.write().await;

        match docs.get_mut(id) {
            Some(indexed_doc) => {
                indexed_doc.embedding = embedding;
                Ok(true)
            }
            None => Err(VectorStoreError::NotFound(id.to_string())),
        }
    }

    async fn upsert(&mut self, doc: IndexedDocument) -> Result<bool, VectorStoreError> {
        if doc.embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: doc.embedding.len(),
            });
        }

        let mut docs = self.documents.write().await;

        // Check capacity only for new insertions
        let is_insert = !docs.contains_key(&doc.document.id);
        if is_insert && self.max_capacity > 0 && docs.len() >= self.max_capacity {
            return Err(VectorStoreError::CapacityExceeded {
                max: self.max_capacity,
                current: docs.len(),
            });
        }

        docs.insert(doc.document.id.clone(), doc);
        Ok(is_insert)
    }

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

        let docs = self.documents.read().await;

        if docs.is_empty() {
            return Ok(Vec::new());
        }

        // Collect all embeddings and their IDs
        let entries: Vec<(&DocumentId, &IndexedDocument)> = docs.iter().collect();
        let embeddings: Vec<Vec<f32>> = entries.iter().map(|(_, d)| d.embedding.clone()).collect();

        // Find top-k similar
        let top_indices =
            top_k_similar(query_embedding, &embeddings, top_k, self.metric, min_score);

        // Build results
        let results = top_indices
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, score))| {
                let doc = entries[idx].1.document.clone();
                SearchResult::new(doc, score, rank)
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
        if query_embedding.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query_embedding.len(),
            });
        }

        let docs = self.documents.read().await;

        if docs.is_empty() {
            return Ok(Vec::new());
        }

        // Filter documents first if a filter is provided
        let filtered_entries: Vec<(&DocumentId, &IndexedDocument)> = match filter {
            Some(f) => docs
                .iter()
                .filter(|(_, indexed_doc)| f.matches(&indexed_doc.document.metadata))
                .collect(),
            None => docs.iter().collect(),
        };

        if filtered_entries.is_empty() {
            return Ok(Vec::new());
        }

        // Collect embeddings from filtered documents
        let embeddings: Vec<Vec<f32>> = filtered_entries
            .iter()
            .map(|(_, d)| d.embedding.clone())
            .collect();

        // Find top-k similar among filtered documents
        let top_indices =
            top_k_similar(query_embedding, &embeddings, top_k, self.metric, min_score);

        // Build results
        let results = top_indices
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, score))| {
                let doc = filtered_entries[idx].1.document.clone();
                SearchResult::new(doc, score, rank)
            })
            .collect();

        Ok(results)
    }

    async fn count(&self) -> usize {
        self.documents.read().await.len()
    }

    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        self.documents.write().await.clear();
        Ok(())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn similarity_metric(&self) -> SimilarityMetric {
        self.metric
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::types::Document;

    fn create_test_doc(content: &str, embedding: Vec<f32>) -> IndexedDocument {
        IndexedDocument::new(Document::new(content), embedding)
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let mut store = InMemoryVectorStore::new(3);
        let doc = create_test_doc("test", vec![1.0, 0.0, 0.0]);
        let id = doc.document.id.clone();

        store.insert(doc).await.unwrap();

        let retrieved = store.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().document.content, "test");
    }

    #[tokio::test]
    async fn test_insert_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new(3);
        let doc = create_test_doc("test", vec![1.0, 0.0]); // wrong dimension

        let result = store.insert(doc).await;
        assert!(matches!(
            result,
            Err(VectorStoreError::DimensionMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn test_insert_duplicate() {
        let mut store = InMemoryVectorStore::new(3);
        let doc1 = create_test_doc("test1", vec![1.0, 0.0, 0.0]);
        let id = doc1.document.id.clone();
        let mut doc2 = create_test_doc("test2", vec![0.0, 1.0, 0.0]);
        doc2.document.id = id;

        store.insert(doc1).await.unwrap();
        let result = store.insert(doc2).await;
        assert!(matches!(result, Err(VectorStoreError::DuplicateId(_))));
    }

    #[tokio::test]
    async fn test_delete() {
        let mut store = InMemoryVectorStore::new(3);
        let doc = create_test_doc("test", vec![1.0, 0.0, 0.0]);
        let id = doc.document.id.clone();

        store.insert(doc).await.unwrap();
        assert_eq!(store.count().await, 1);

        let deleted = store.delete(&id).await.unwrap();
        assert!(deleted);
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let mut store = InMemoryVectorStore::new(3);
        let deleted = store.delete(&DocumentId::new()).await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_search() {
        let mut store = InMemoryVectorStore::new(2);

        // Add some documents with different embeddings
        let doc1 = create_test_doc("similar", vec![1.0, 0.0]);
        let doc2 = create_test_doc("orthogonal", vec![0.0, 1.0]);
        let doc3 = create_test_doc("somewhat similar", vec![0.8, 0.6]);

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();
        store.insert(doc3).await.unwrap();

        // Search with a query similar to doc1
        let results = store.search(&[1.0, 0.0], 2, None).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.content, "similar");
        assert_eq!(results[0].rank, 0);
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_search_with_min_score() {
        let mut store = InMemoryVectorStore::new(2);

        let doc1 = create_test_doc("similar", vec![1.0, 0.0]);
        let doc2 = create_test_doc("orthogonal", vec![0.0, 1.0]);

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();

        // Search with high min_score
        let results = store.search(&[1.0, 0.0], 10, Some(0.9)).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "similar");
    }

    #[tokio::test]
    async fn test_search_empty_store() {
        let store = InMemoryVectorStore::new(3);
        let results = store.search(&[1.0, 0.0, 0.0], 10, None).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_capacity_limit() {
        let mut store = InMemoryVectorStore::new(2).with_max_capacity(2);

        store
            .insert(create_test_doc("doc1", vec![1.0, 0.0]))
            .await
            .unwrap();
        store
            .insert(create_test_doc("doc2", vec![0.0, 1.0]))
            .await
            .unwrap();

        let result = store.insert(create_test_doc("doc3", vec![0.5, 0.5])).await;

        assert!(matches!(
            result,
            Err(VectorStoreError::CapacityExceeded { .. })
        ));
    }

    #[tokio::test]
    async fn test_clear() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc("doc1", vec![1.0, 0.0]))
            .await
            .unwrap();
        store
            .insert(create_test_doc("doc2", vec![0.0, 1.0]))
            .await
            .unwrap();

        assert_eq!(store.count().await, 2);

        store.clear().await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_different_metrics() {
        let mut cosine_store = InMemoryVectorStore::new(2).with_metric(SimilarityMetric::Cosine);
        let mut dot_store = InMemoryVectorStore::new(2).with_metric(SimilarityMetric::DotProduct);

        let doc = create_test_doc("test", vec![2.0, 0.0]);

        cosine_store.insert(doc.clone()).await.unwrap();
        dot_store.insert(doc).await.unwrap();

        let cosine_results = cosine_store.search(&[1.0, 0.0], 1, None).await.unwrap();
        let dot_results = dot_store.search(&[1.0, 0.0], 1, None).await.unwrap();

        // Cosine should be 1.0 (same direction)
        assert!((cosine_results[0].score - 1.0).abs() < 1e-6);
        // Dot product should be 2.0
        assert!((dot_results[0].score - 2.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_update_existing_document() {
        let mut store = InMemoryVectorStore::new(2);
        let doc = create_test_doc("test", vec![1.0, 0.0]);
        let id = doc.document.id.clone();

        store.insert(doc).await.unwrap();

        // Update the embedding
        let updated = store.update(&id, vec![0.0, 1.0]).await.unwrap();
        assert!(updated);

        // Verify the embedding was updated
        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.embedding, vec![0.0, 1.0]);
    }

    #[tokio::test]
    async fn test_update_nonexistent_document() {
        let mut store = InMemoryVectorStore::new(2);
        let id = DocumentId::new();

        let result = store.update(&id, vec![1.0, 0.0]).await;
        assert!(matches!(result, Err(VectorStoreError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_update_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new(2);
        let doc = create_test_doc("test", vec![1.0, 0.0]);
        let id = doc.document.id.clone();

        store.insert(doc).await.unwrap();

        let result = store.update(&id, vec![1.0, 0.0, 0.0]).await; // wrong dimension
        assert!(matches!(
            result,
            Err(VectorStoreError::DimensionMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn test_upsert_new_document() {
        let mut store = InMemoryVectorStore::new(2);
        let doc = create_test_doc("test", vec![1.0, 0.0]);
        let id = doc.document.id.clone();

        // Upsert a new document
        let inserted = store.upsert(doc).await.unwrap();
        assert!(inserted); // Should return true for insert

        assert_eq!(store.count().await, 1);

        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.document.content, "test");
    }

    #[tokio::test]
    async fn test_upsert_existing_document() {
        let mut store = InMemoryVectorStore::new(2);
        let doc = create_test_doc("test", vec![1.0, 0.0]);
        let id = doc.document.id.clone();

        store.insert(doc).await.unwrap();

        // Create a new document with the same ID but different content
        let mut updated_doc = create_test_doc("updated content", vec![0.0, 1.0]);
        updated_doc.document.id = id.clone();

        // Upsert should update
        let inserted = store.upsert(updated_doc).await.unwrap();
        assert!(!inserted); // Should return false for update

        assert_eq!(store.count().await, 1);

        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.document.content, "updated content");
        assert_eq!(retrieved.embedding, vec![0.0, 1.0]);
    }

    #[tokio::test]
    async fn test_upsert_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new(2);
        let doc = create_test_doc("test", vec![1.0, 0.0, 0.0]); // wrong dimension

        let result = store.upsert(doc).await;
        assert!(matches!(
            result,
            Err(VectorStoreError::DimensionMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn test_upsert_respects_capacity_for_inserts() {
        let mut store = InMemoryVectorStore::new(2).with_max_capacity(1);
        let doc1 = create_test_doc("doc1", vec![1.0, 0.0]);
        let id1 = doc1.document.id.clone();

        store.insert(doc1).await.unwrap();

        // Upsert existing document should work (update)
        let mut updated_doc = create_test_doc("doc1 updated", vec![0.5, 0.5]);
        updated_doc.document.id = id1;
        let inserted = store.upsert(updated_doc).await.unwrap();
        assert!(!inserted);

        // Upsert new document should fail due to capacity
        let doc2 = create_test_doc("doc2", vec![0.0, 1.0]);
        let result = store.upsert(doc2).await;
        assert!(matches!(
            result,
            Err(VectorStoreError::CapacityExceeded { .. })
        ));
    }

    #[tokio::test]
    async fn test_update_embedding_affects_search() {
        let mut store = InMemoryVectorStore::new(2);

        let doc1 = create_test_doc("doc1", vec![1.0, 0.0]);
        let id1 = doc1.document.id.clone();
        let doc2 = create_test_doc("doc2", vec![0.0, 1.0]);

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();

        // Initially, doc1 should be most similar to [1.0, 0.0]
        let results = store.search(&[1.0, 0.0], 1, None).await.unwrap();
        assert_eq!(results[0].document.content, "doc1");

        // Update doc1's embedding to [0.0, 1.0]
        store.update(&id1, vec![0.0, 1.0]).await.unwrap();

        // Now doc2 should be most similar to [1.0, 0.0] (doc1 is orthogonal)
        // Actually both have the same embedding now, but doc2 was the original
        // Let's search for [1.0, 0.0] - both are now orthogonal
        let results = store.search(&[0.0, 1.0], 2, None).await.unwrap();
        // Both should have score 1.0 now
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert!((results[1].score - 1.0).abs() < 1e-6);
    }

    fn create_test_doc_with_metadata(
        content: &str,
        embedding: Vec<f32>,
        metadata: &[(&str, &str)],
    ) -> IndexedDocument {
        let mut doc = Document::new(content);
        for (key, value) in metadata {
            doc = doc.with_metadata(*key, *value);
        }
        IndexedDocument::new(doc, embedding)
    }

    #[tokio::test]
    async fn test_search_with_filter_eq() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "science doc",
                vec![1.0, 0.0],
                &[("category", "science")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "tech doc",
                vec![0.9, 0.1],
                &[("category", "technology")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "art doc",
                vec![0.8, 0.2],
                &[("category", "art")],
            ))
            .await
            .unwrap();

        // Search with category filter
        let filter = MetadataFilter::eq("category", "science");
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "science doc");
    }

    #[tokio::test]
    async fn test_search_with_filter_ne() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "published doc",
                vec![1.0, 0.0],
                &[("status", "published")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "draft doc",
                vec![0.9, 0.1],
                &[("status", "draft")],
            ))
            .await
            .unwrap();

        // Search for non-draft documents
        let filter = MetadataFilter::ne("status", "draft");
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "published doc");
    }

    #[tokio::test]
    async fn test_search_with_filter_contains() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "rust programming",
                vec![1.0, 0.0],
                &[("title", "Learning Rust Programming")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "python programming",
                vec![0.9, 0.1],
                &[("title", "Python Guide")],
            ))
            .await
            .unwrap();

        // Search for titles containing "Rust"
        let filter = MetadataFilter::contains("title", "Rust");
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "rust programming");
    }

    #[tokio::test]
    async fn test_search_with_filter_exists() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "with author",
                vec![1.0, 0.0],
                &[("author", "John Doe")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "without author",
                vec![0.9, 0.1],
                &[("category", "general")],
            ))
            .await
            .unwrap();

        // Search for documents with author field
        let filter = MetadataFilter::exists("author");
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "with author");
    }

    #[tokio::test]
    async fn test_search_with_filter_not_exists() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "deprecated doc",
                vec![1.0, 0.0],
                &[("deprecated", "true")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "active doc",
                vec![0.9, 0.1],
                &[("status", "active")],
            ))
            .await
            .unwrap();

        // Search for documents without deprecated field
        let filter = MetadataFilter::not_exists("deprecated");
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "active doc");
    }

    #[tokio::test]
    async fn test_search_with_filter_and() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "published science",
                vec![1.0, 0.0],
                &[("status", "published"), ("category", "science")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "draft science",
                vec![0.9, 0.1],
                &[("status", "draft"), ("category", "science")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "published tech",
                vec![0.8, 0.2],
                &[("status", "published"), ("category", "technology")],
            ))
            .await
            .unwrap();

        // Search for published science documents
        let filter = MetadataFilter::and(vec![
            MetadataFilter::eq("status", "published"),
            MetadataFilter::eq("category", "science"),
        ]);
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "published science");
    }

    #[tokio::test]
    async fn test_search_with_filter_or() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "science doc",
                vec![1.0, 0.0],
                &[("category", "science")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "tech doc",
                vec![0.9, 0.1],
                &[("category", "technology")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "art doc",
                vec![0.8, 0.2],
                &[("category", "art")],
            ))
            .await
            .unwrap();

        // Search for science OR technology documents
        let filter = MetadataFilter::or(vec![
            MetadataFilter::eq("category", "science"),
            MetadataFilter::eq("category", "technology"),
        ]);
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        // Verify both science and tech are present (order by similarity)
        let contents: Vec<&str> = results
            .iter()
            .map(|r| r.document.content.as_str())
            .collect();
        assert!(contents.contains(&"science doc"));
        assert!(contents.contains(&"tech doc"));
    }

    #[tokio::test]
    async fn test_search_with_filter_complex_nested() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "published science",
                vec![1.0, 0.0],
                &[("status", "published"), ("category", "science")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "published tech",
                vec![0.9, 0.1],
                &[("status", "published"), ("category", "technology")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "draft science",
                vec![0.8, 0.2],
                &[("status", "draft"), ("category", "science")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "published art",
                vec![0.7, 0.3],
                &[("status", "published"), ("category", "art")],
            ))
            .await
            .unwrap();

        // Search for: published AND (science OR technology)
        let filter = MetadataFilter::and(vec![
            MetadataFilter::eq("status", "published"),
            MetadataFilter::or(vec![
                MetadataFilter::eq("category", "science"),
                MetadataFilter::eq("category", "technology"),
            ]),
        ]);
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        let contents: Vec<&str> = results
            .iter()
            .map(|r| r.document.content.as_str())
            .collect();
        assert!(contents.contains(&"published science"));
        assert!(contents.contains(&"published tech"));
        assert!(!contents.contains(&"draft science"));
        assert!(!contents.contains(&"published art"));
    }

    #[tokio::test]
    async fn test_search_with_filter_none() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "doc1",
                vec![1.0, 0.0],
                &[("category", "a")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "doc2",
                vec![0.9, 0.1],
                &[("category", "b")],
            ))
            .await
            .unwrap();

        // Search without filter should return all
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, None)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_with_filter_empty_result() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "doc1",
                vec![1.0, 0.0],
                &[("category", "a")],
            ))
            .await
            .unwrap();

        // Search with filter that matches nothing
        let filter = MetadataFilter::eq("category", "nonexistent");
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_with_filter_dimension_mismatch() {
        let store = InMemoryVectorStore::new(2);
        let filter = MetadataFilter::eq("a", "b");

        let result = store
            .search_with_filter(&[1.0, 0.0, 0.0], 10, None, Some(&filter))
            .await;

        assert!(matches!(
            result,
            Err(VectorStoreError::DimensionMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn test_search_with_filter_empty_store() {
        let store = InMemoryVectorStore::new(2);
        let filter = MetadataFilter::eq("a", "b");

        let results = store
            .search_with_filter(&[1.0, 0.0], 10, None, Some(&filter))
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_with_filter_and_min_score() {
        let mut store = InMemoryVectorStore::new(2);

        store
            .insert(create_test_doc_with_metadata(
                "high similarity",
                vec![1.0, 0.0],
                &[("category", "science")],
            ))
            .await
            .unwrap();
        store
            .insert(create_test_doc_with_metadata(
                "low similarity",
                vec![0.0, 1.0],
                &[("category", "science")],
            ))
            .await
            .unwrap();

        // Search with filter and high min_score
        let filter = MetadataFilter::eq("category", "science");
        let results = store
            .search_with_filter(&[1.0, 0.0], 10, Some(0.9), Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "high similarity");
    }

    #[tokio::test]
    async fn test_search_with_filter_respects_top_k() {
        let mut store = InMemoryVectorStore::new(2);

        for i in 0..5 {
            store
                .insert(create_test_doc_with_metadata(
                    &format!("doc{i}"),
                    vec![1.0 - (i as f32 * 0.1), i as f32 * 0.1],
                    &[("category", "science")],
                ))
                .await
                .unwrap();
        }

        let filter = MetadataFilter::eq("category", "science");
        let results = store
            .search_with_filter(&[1.0, 0.0], 2, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
    }
}
