//! ANN-enhanced vector store implementation.

use std::collections::HashMap;

use crate::error::VectorStoreError;
use crate::layer1_echo::filter::MetadataFilter;
use crate::layer1_echo::traits::{IndexedDocument, SimilarityMetric, VectorStore};
use crate::types::{Document, DocumentId, SearchResult};

use super::config::AnnConfig;
use super::hnsw::HnswIndex;
use super::stats::AnnStats;

/// ANN-enhanced vector store.
pub struct AnnVectorStore {
    index: HnswIndex,
    documents: HashMap<DocumentId, Document>,
    config: AnnConfig,
}

impl AnnVectorStore {
    /// Create a new ANN vector store.
    #[must_use]
    pub fn new(dimension: usize, config: AnnConfig) -> Self {
        Self {
            index: HnswIndex::new(dimension, config.clone()),
            documents: HashMap::new(),
            config,
        }
    }

    /// Create a new ANN vector store with default configuration.
    #[must_use]
    pub fn with_default_config(dimension: usize) -> Self {
        Self::new(dimension, AnnConfig::default())
    }

    /// Get the HNSW index statistics.
    #[must_use]
    pub fn stats(&self) -> AnnStats {
        self.index.stats()
    }

    /// Get a reference to the underlying HNSW index.
    #[must_use]
    pub fn index(&self) -> &HnswIndex {
        &self.index
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl VectorStore for AnnVectorStore {
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError> {
        if doc.embedding.len() != self.index.dimension() {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension(),
                actual: doc.embedding.len(),
            });
        }

        if self.documents.contains_key(&doc.document.id) {
            return Err(VectorStoreError::DuplicateId(doc.document.id.to_string()));
        }

        self.index.insert(doc.document.id.clone(), doc.embedding)?;
        self.documents.insert(doc.document.id.clone(), doc.document);
        Ok(())
    }

    async fn insert_batch(&mut self, docs: Vec<IndexedDocument>) -> Result<(), VectorStoreError> {
        for doc in docs {
            self.insert(doc).await?;
        }
        Ok(())
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<IndexedDocument>, VectorStoreError> {
        match (self.documents.get(id), self.index.get_node(id)) {
            (Some(doc), Some(node)) => {
                Ok(Some(IndexedDocument::new(doc.clone(), node.vector.clone())))
            }
            _ => Ok(None),
        }
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        let removed = self.index.remove(id).is_some();
        self.documents.remove(id);
        Ok(removed)
    }

    async fn update(
        &mut self,
        id: &DocumentId,
        embedding: Vec<f32>,
    ) -> Result<bool, VectorStoreError> {
        if embedding.len() != self.index.dimension() {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension(),
                actual: embedding.len(),
            });
        }

        if !self.documents.contains_key(id) {
            return Err(VectorStoreError::NotFound(id.to_string()));
        }

        self.index.insert(id.clone(), embedding)?;
        Ok(true)
    }

    async fn upsert(&mut self, doc: IndexedDocument) -> Result<bool, VectorStoreError> {
        if doc.embedding.len() != self.index.dimension() {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension(),
                actual: doc.embedding.len(),
            });
        }

        let is_new = !self.documents.contains_key(&doc.document.id);
        self.index.insert(doc.document.id.clone(), doc.embedding)?;
        self.documents.insert(doc.document.id.clone(), doc.document);
        Ok(is_new)
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.index.dimension() {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension(),
                actual: query_embedding.len(),
            });
        }

        let min_score = min_score.unwrap_or(f32::NEG_INFINITY);
        let results = self
            .index
            .search_with_threshold(query_embedding, top_k, min_score);

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (id, score))| {
                self.documents
                    .get(&id)
                    .map(|doc| SearchResult::new(doc.clone(), score, rank))
            })
            .collect();

        Ok(search_results)
    }

    async fn search_with_filter(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.index.dimension() {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension(),
                actual: query_embedding.len(),
            });
        }

        // For filtered search, we need to search more candidates and then filter
        // This is a simple approach; more sophisticated methods exist
        let search_multiplier = 10; // Search more to account for filtered out results
        let extended_k = top_k * search_multiplier;

        let min_score = min_score.unwrap_or(f32::NEG_INFINITY);
        let results = self
            .index
            .search_with_threshold(query_embedding, extended_k, min_score);

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .filter_map(|(id, score)| {
                self.documents.get(&id).and_then(|doc| {
                    // Apply filter if provided
                    match filter {
                        Some(f) if !f.matches(&doc.metadata) => None,
                        _ => Some((doc.clone(), score)),
                    }
                })
            })
            .take(top_k)
            .enumerate()
            .map(|(rank, (doc, score))| SearchResult::new(doc, score, rank))
            .collect();

        Ok(search_results)
    }

    async fn count(&self) -> usize {
        self.documents.len()
    }

    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        self.index.clear();
        self.documents.clear();
        Ok(())
    }

    fn dimension(&self) -> usize {
        self.index.dimension()
    }

    fn similarity_metric(&self) -> SimilarityMetric {
        self.config.distance_metric
    }
}
