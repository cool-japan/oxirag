//! Multi-vector document support for ColBERT-style late interaction.
//!
//! This module provides support for multi-vector representations of documents,
//! where each token has its own embedding. This enables fine-grained matching
//! between query and document tokens using the `MaxSim` scoring function.
//!
//! # ColBERT-style Late Interaction
//!
//! The late interaction mechanism works as follows:
//! 1. Each query token is compared against all document tokens
//! 2. For each query token, the maximum similarity across document tokens is computed
//! 3. These maximum similarities are summed to produce the final score
//!
//! This allows for more precise matching than single-vector representations
//! while being more efficient than full cross-attention.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{EmbeddingError, VectorStoreError};
use crate::layer1_echo::similarity::cosine_similarity;
use crate::layer1_echo::traits::SimilarityMetric;
use crate::types::{Document, DocumentId};

/// Embedding for a single token with position information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEmbedding {
    /// The token text (optional, for debugging/analysis).
    pub token: Option<String>,
    /// Position of the token in the original text (0-indexed).
    pub position: usize,
    /// The embedding vector for this token.
    pub embedding: Vec<f32>,
}

impl TokenEmbedding {
    /// Create a new token embedding.
    #[must_use]
    pub fn new(position: usize, embedding: Vec<f32>) -> Self {
        Self {
            token: None,
            position,
            embedding,
        }
    }

    /// Create a new token embedding with token text.
    #[must_use]
    pub fn with_token(position: usize, token: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            token: Some(token.into()),
            position,
            embedding,
        }
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }
}

/// A document with multiple token-level embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorDocument {
    /// The underlying document.
    pub document: Document,
    /// Token-level embeddings for the document.
    pub token_embeddings: Vec<TokenEmbedding>,
}

impl MultiVectorDocument {
    /// Create a new multi-vector document.
    #[must_use]
    pub fn new(document: Document, token_embeddings: Vec<TokenEmbedding>) -> Self {
        Self {
            document,
            token_embeddings,
        }
    }

    /// Get the number of token embeddings.
    #[must_use]
    pub fn num_tokens(&self) -> usize {
        self.token_embeddings.len()
    }

    /// Get the embedding dimension (from first token, or 0 if empty).
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.token_embeddings
            .first()
            .map_or(0, |t| t.embedding.len())
    }
}

/// Information about token-level matches in a search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMatch {
    /// Index of the query token.
    pub query_token_idx: usize,
    /// Index of the matched document token.
    pub doc_token_idx: usize,
    /// Similarity score between the tokens.
    pub similarity: f32,
}

impl TokenMatch {
    /// Create a new token match.
    #[must_use]
    pub fn new(query_token_idx: usize, doc_token_idx: usize, similarity: f32) -> Self {
        Self {
            query_token_idx,
            doc_token_idx,
            similarity,
        }
    }
}

/// Search result with token-level matching information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorSearchResult {
    /// The matched document.
    pub document: Document,
    /// Overall `MaxSim` score.
    pub score: f32,
    /// Rank in the result set (0-indexed).
    pub rank: usize,
    /// Token-level match information (best match for each query token).
    pub token_matches: Vec<TokenMatch>,
}

impl MultiVectorSearchResult {
    /// Create a new multi-vector search result.
    #[must_use]
    pub fn new(
        document: Document,
        score: f32,
        rank: usize,
        token_matches: Vec<TokenMatch>,
    ) -> Self {
        Self {
            document,
            score,
            rank,
            token_matches,
        }
    }
}

/// Maximum similarity score computation (ColBERT-style).
///
/// For each query token, finds the maximum similarity across all document tokens,
/// then sums these maximum similarities to produce the final score.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxSimScore {
    /// The similarity metric to use for token comparisons.
    pub metric: SimilarityMetric,
}

impl MaxSimScore {
    /// Create a new `MaxSim` scorer with the given similarity metric.
    #[must_use]
    pub fn new(metric: SimilarityMetric) -> Self {
        Self { metric }
    }

    /// Compute the `MaxSim` score between query tokens and document tokens.
    ///
    /// # Arguments
    ///
    /// * `query_tokens` - Token embeddings for the query.
    /// * `doc_tokens` - Token embeddings for the document.
    ///
    /// # Returns
    ///
    /// A tuple of (total score, vector of best matches for each query token).
    #[must_use]
    pub fn compute(
        &self,
        query_tokens: &[TokenEmbedding],
        doc_tokens: &[TokenEmbedding],
    ) -> (f32, Vec<TokenMatch>) {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return (0.0, Vec::new());
        }

        let mut total_score = 0.0;
        let mut token_matches = Vec::with_capacity(query_tokens.len());

        for (q_idx, q_token) in query_tokens.iter().enumerate() {
            let mut max_sim = f32::NEG_INFINITY;
            let mut best_doc_idx = 0;

            for (d_idx, d_token) in doc_tokens.iter().enumerate() {
                let sim = self.compute_token_similarity(&q_token.embedding, &d_token.embedding);
                if sim > max_sim {
                    max_sim = sim;
                    best_doc_idx = d_idx;
                }
            }

            total_score += max_sim;
            token_matches.push(TokenMatch::new(q_idx, best_doc_idx, max_sim));
        }

        (total_score, token_matches)
    }

    /// Compute similarity between two token embeddings.
    fn compute_token_similarity(self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            SimilarityMetric::Cosine => cosine_similarity(a, b),
            SimilarityMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
            SimilarityMetric::Euclidean => {
                let distance: f32 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                1.0 / (1.0 + distance)
            }
        }
    }

    /// Compute the `MaxSim` score only (without match details).
    #[must_use]
    pub fn score(&self, query_tokens: &[TokenEmbedding], doc_tokens: &[TokenEmbedding]) -> f32 {
        self.compute(query_tokens, doc_tokens).0
    }
}

/// Provider for generating token-level embeddings from text.
#[async_trait]
pub trait TokenEmbeddingProvider: Send + Sync {
    /// Generate token-level embeddings for a text.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails.
    async fn embed_tokens(&self, text: &str) -> Result<Vec<TokenEmbedding>, EmbeddingError>;

    /// Generate token-level embeddings for multiple texts.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails.
    async fn embed_tokens_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<TokenEmbedding>>, EmbeddingError>;

    /// Get the dimension of the embeddings produced by this provider.
    fn dimension(&self) -> usize;

    /// Get the model identifier.
    fn model_id(&self) -> &str;
}

/// Storage for multi-vector documents with `MaxSim` search.
#[async_trait]
pub trait MultiVectorStore: Send + Sync {
    /// Insert a multi-vector document.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    async fn insert(&mut self, doc: MultiVectorDocument) -> Result<(), VectorStoreError>;

    /// Insert multiple multi-vector documents.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails.
    async fn insert_batch(
        &mut self,
        docs: Vec<MultiVectorDocument>,
    ) -> Result<(), VectorStoreError>;

    /// Get a document by its ID.
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails.
    async fn get(&self, id: &DocumentId) -> Result<Option<MultiVectorDocument>, VectorStoreError>;

    /// Delete a document by its ID.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError>;

    /// Search for similar documents using `MaxSim` scoring.
    ///
    /// # Errors
    ///
    /// Returns an error if search fails.
    async fn search(
        &self,
        query_tokens: &[TokenEmbedding],
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<MultiVectorSearchResult>, VectorStoreError>;

    /// Get the number of documents in the store.
    async fn count(&self) -> usize;

    /// Clear all documents from the store.
    ///
    /// # Errors
    ///
    /// Returns an error if clearing fails.
    async fn clear(&mut self) -> Result<(), VectorStoreError>;

    /// Get the expected embedding dimension.
    fn dimension(&self) -> usize;

    /// Get the similarity metric used by this store.
    fn similarity_metric(&self) -> SimilarityMetric;
}

/// Mock token embedding provider for testing.
pub struct MockTokenEmbeddingProvider {
    dimension: usize,
    model_id: String,
}

impl MockTokenEmbeddingProvider {
    /// Create a new mock token embedding provider.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model_id: "mock-token-embedder".to_string(),
        }
    }

    /// Create a mock token embedding provider with a custom model ID.
    #[must_use]
    pub fn with_model_id(dimension: usize, model_id: impl Into<String>) -> Self {
        Self {
            dimension,
            model_id: model_id.into(),
        }
    }

    /// Generate a deterministic embedding from text for testing.
    fn generate_embedding(&self, text: &str, position: usize) -> Vec<f32> {
        let mut embedding = vec![0.0; self.dimension];
        let bytes = text.as_bytes();

        for (i, &byte) in bytes.iter().take(self.dimension).enumerate() {
            embedding[i] = (f32::from(byte) / 255.0) * 2.0 - 1.0;
        }

        // Add position influence
        if self.dimension > 0 {
            #[allow(clippy::cast_precision_loss)]
            let pos_factor = (position as f32).sin() * 0.1;
            for val in &mut embedding {
                *val += pos_factor;
            }
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl TokenEmbeddingProvider for MockTokenEmbeddingProvider {
    async fn embed_tokens(&self, text: &str) -> Result<Vec<TokenEmbedding>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Split text into words as "tokens"
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let token_embeddings: Vec<TokenEmbedding> = tokens
            .iter()
            .enumerate()
            .map(|(pos, token)| {
                let embedding = self.generate_embedding(token, pos);
                TokenEmbedding::with_token(pos, *token, embedding)
            })
            .collect();

        Ok(token_embeddings)
    }

    async fn embed_tokens_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<TokenEmbedding>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_tokens(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// In-memory storage for multi-vector documents with `MaxSim` search.
pub struct InMemoryMultiVectorStore {
    documents: HashMap<DocumentId, MultiVectorDocument>,
    dimension: usize,
    metric: SimilarityMetric,
    max_sim: MaxSimScore,
}

impl InMemoryMultiVectorStore {
    /// Create a new in-memory multi-vector store.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            documents: HashMap::new(),
            dimension,
            metric: SimilarityMetric::Cosine,
            max_sim: MaxSimScore::new(SimilarityMetric::Cosine),
        }
    }

    /// Create a new store with a specific similarity metric.
    #[must_use]
    pub fn with_metric(dimension: usize, metric: SimilarityMetric) -> Self {
        Self {
            documents: HashMap::new(),
            dimension,
            metric,
            max_sim: MaxSimScore::new(metric),
        }
    }
}

#[async_trait]
impl MultiVectorStore for InMemoryMultiVectorStore {
    async fn insert(&mut self, doc: MultiVectorDocument) -> Result<(), VectorStoreError> {
        // Validate dimension
        if let Some(token) = doc.token_embeddings.first()
            && token.embedding.len() != self.dimension
        {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: token.embedding.len(),
            });
        }

        self.documents.insert(doc.document.id.clone(), doc);
        Ok(())
    }

    async fn insert_batch(
        &mut self,
        docs: Vec<MultiVectorDocument>,
    ) -> Result<(), VectorStoreError> {
        for doc in docs {
            self.insert(doc).await?;
        }
        Ok(())
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<MultiVectorDocument>, VectorStoreError> {
        Ok(self.documents.get(id).cloned())
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        Ok(self.documents.remove(id).is_some())
    }

    async fn search(
        &self,
        query_tokens: &[TokenEmbedding],
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<MultiVectorSearchResult>, VectorStoreError> {
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Compute scores for all documents
        let mut scored: Vec<(DocumentId, f32, Vec<TokenMatch>)> = self
            .documents
            .iter()
            .map(|(id, doc)| {
                let (score, matches) = self.max_sim.compute(query_tokens, &doc.token_embeddings);
                (id.clone(), score, matches)
            })
            .filter(|(_, score, _)| min_score.is_none_or(|min| *score >= min))
            .collect();

        // Sort by descending score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        scored.truncate(top_k);

        // Convert to results
        let results = scored
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (id, score, token_matches))| {
                self.documents.get(&id).map(|doc| {
                    MultiVectorSearchResult::new(doc.document.clone(), score, rank, token_matches)
                })
            })
            .collect();

        Ok(results)
    }

    async fn count(&self) -> usize {
        self.documents.len()
    }

    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        self.documents.clear();
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
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embedding_creation() {
        let embedding = vec![0.1, 0.2, 0.3];
        let token = TokenEmbedding::new(0, embedding.clone());

        assert_eq!(token.position, 0);
        assert_eq!(token.embedding, embedding);
        assert!(token.token.is_none());
        assert_eq!(token.dimension(), 3);
    }

    #[test]
    fn test_token_embedding_with_token() {
        let embedding = vec![0.1, 0.2, 0.3];
        let token = TokenEmbedding::with_token(1, "hello", embedding.clone());

        assert_eq!(token.position, 1);
        assert_eq!(token.token, Some("hello".to_string()));
        assert_eq!(token.embedding, embedding);
    }

    #[test]
    fn test_multi_vector_document() {
        let doc = Document::new("test content");
        let tokens = vec![
            TokenEmbedding::new(0, vec![0.1, 0.2]),
            TokenEmbedding::new(1, vec![0.3, 0.4]),
        ];

        let mv_doc = MultiVectorDocument::new(doc.clone(), tokens);

        assert_eq!(mv_doc.document.content, "test content");
        assert_eq!(mv_doc.num_tokens(), 2);
        assert_eq!(mv_doc.dimension(), 2);
    }

    #[test]
    fn test_multi_vector_document_empty() {
        let doc = Document::new("test");
        let mv_doc = MultiVectorDocument::new(doc, Vec::new());

        assert_eq!(mv_doc.num_tokens(), 0);
        assert_eq!(mv_doc.dimension(), 0);
    }

    #[test]
    fn test_token_match() {
        let m = TokenMatch::new(0, 2, 0.95);

        assert_eq!(m.query_token_idx, 0);
        assert_eq!(m.doc_token_idx, 2);
        assert!((m.similarity - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_max_sim_score_empty() {
        let scorer = MaxSimScore::new(SimilarityMetric::Cosine);

        let query: Vec<TokenEmbedding> = Vec::new();
        let doc: Vec<TokenEmbedding> = Vec::new();

        let (score, matches) = scorer.compute(&query, &doc);

        assert_eq!(score, 0.0);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_max_sim_score_identical() {
        let scorer = MaxSimScore::new(SimilarityMetric::Cosine);

        let embedding = vec![1.0, 0.0, 0.0];
        let query = vec![TokenEmbedding::new(0, embedding.clone())];
        let doc = vec![TokenEmbedding::new(0, embedding)];

        let (score, matches) = scorer.compute(&query, &doc);

        assert!((score - 1.0).abs() < 1e-6);
        assert_eq!(matches.len(), 1);
        assert!((matches[0].similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_sim_score_multiple_tokens() {
        let scorer = MaxSimScore::new(SimilarityMetric::Cosine);

        // Query with 2 tokens
        let query = vec![
            TokenEmbedding::new(0, vec![1.0, 0.0]),
            TokenEmbedding::new(1, vec![0.0, 1.0]),
        ];

        // Doc with 3 tokens, including matches for both query tokens
        let doc = vec![
            TokenEmbedding::new(0, vec![0.5, 0.5]), // partial match
            TokenEmbedding::new(1, vec![1.0, 0.0]), // best match for query[0]
            TokenEmbedding::new(2, vec![0.0, 1.0]), // best match for query[1]
        ];

        let (score, matches) = scorer.compute(&query, &doc);

        // Each query token should find its best match (score ~1.0 each)
        assert!(score > 1.9);
        assert_eq!(matches.len(), 2);

        // Query token 0 should match doc token 1
        assert_eq!(matches[0].doc_token_idx, 1);
        assert!((matches[0].similarity - 1.0).abs() < 1e-6);

        // Query token 1 should match doc token 2
        assert_eq!(matches[1].doc_token_idx, 2);
        assert!((matches[1].similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_sim_score_with_dot_product() {
        let scorer = MaxSimScore::new(SimilarityMetric::DotProduct);

        let query = vec![TokenEmbedding::new(0, vec![2.0, 3.0])];
        let doc = vec![TokenEmbedding::new(0, vec![4.0, 5.0])];

        let score = scorer.score(&query, &doc);

        // Dot product: 2*4 + 3*5 = 8 + 15 = 23
        assert!((score - 23.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_mock_token_embedding_provider() {
        let provider = MockTokenEmbeddingProvider::new(32);

        assert_eq!(provider.dimension(), 32);
        assert_eq!(provider.model_id(), "mock-token-embedder");

        let tokens = provider.embed_tokens("hello world test").await.unwrap();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[0].token, Some("hello".to_string()));
        assert_eq!(tokens[0].embedding.len(), 32);
    }

    #[tokio::test]
    async fn test_mock_token_embedding_provider_empty() {
        let provider = MockTokenEmbeddingProvider::new(32);

        let result = provider.embed_tokens("").await;
        assert!(result.is_err());

        let result = provider.embed_tokens("   ").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_token_embedding_provider_batch() {
        let provider = MockTokenEmbeddingProvider::new(16);

        let texts = ["hello world", "foo bar baz"];
        let results = provider.embed_tokens_batch(&texts).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 3);
    }

    #[tokio::test]
    async fn test_in_memory_multi_vector_store_insert() {
        let mut store = InMemoryMultiVectorStore::new(32);
        let provider = MockTokenEmbeddingProvider::new(32);

        let doc = Document::new("test document");
        let tokens = provider.embed_tokens("test document").await.unwrap();
        let mv_doc = MultiVectorDocument::new(doc.clone(), tokens);

        store.insert(mv_doc).await.unwrap();

        assert_eq!(store.count().await, 1);
        let retrieved = store.get(&doc.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().document.content, "test document");
    }

    #[tokio::test]
    async fn test_in_memory_multi_vector_store_dimension_mismatch() {
        let mut store = InMemoryMultiVectorStore::new(32);

        let doc = Document::new("test");
        let tokens = vec![TokenEmbedding::new(0, vec![0.1, 0.2])]; // dimension 2, expected 32
        let mv_doc = MultiVectorDocument::new(doc, tokens);

        let result = store.insert(mv_doc).await;
        assert!(matches!(
            result,
            Err(VectorStoreError::DimensionMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn test_in_memory_multi_vector_store_search() {
        let mut store = InMemoryMultiVectorStore::new(32);
        let provider = MockTokenEmbeddingProvider::new(32);

        // Index some documents
        let doc1 = Document::new("quick brown fox");
        let tokens1 = provider.embed_tokens("quick brown fox").await.unwrap();
        store
            .insert(MultiVectorDocument::new(doc1, tokens1))
            .await
            .unwrap();

        let doc2 = Document::new("lazy dog sleeps");
        let tokens2 = provider.embed_tokens("lazy dog sleeps").await.unwrap();
        store
            .insert(MultiVectorDocument::new(doc2, tokens2))
            .await
            .unwrap();

        let doc3 = Document::new("quick fox jumps");
        let tokens3 = provider.embed_tokens("quick fox jumps").await.unwrap();
        store
            .insert(MultiVectorDocument::new(doc3, tokens3))
            .await
            .unwrap();

        // Search
        let query_tokens = provider.embed_tokens("quick fox").await.unwrap();
        let results = store.search(&query_tokens, 2, None).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].rank, 0);
        assert_eq!(results[1].rank, 1);

        // The first result should be either "quick brown fox" or "quick fox jumps"
        // since they both contain "quick" and "fox"
        assert!(
            results[0].document.content.contains("quick")
                || results[0].document.content.contains("fox")
        );
    }

    #[tokio::test]
    async fn test_in_memory_multi_vector_store_search_with_min_score() {
        let mut store = InMemoryMultiVectorStore::new(32);
        let provider = MockTokenEmbeddingProvider::new(32);

        let doc = Document::new("test doc");
        let tokens = provider.embed_tokens("test doc").await.unwrap();
        store
            .insert(MultiVectorDocument::new(doc, tokens))
            .await
            .unwrap();

        // Search with very high min_score should return empty
        let query_tokens = provider.embed_tokens("completely different").await.unwrap();
        let results = store.search(&query_tokens, 10, Some(100.0)).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_in_memory_multi_vector_store_delete() {
        let mut store = InMemoryMultiVectorStore::new(32);
        let provider = MockTokenEmbeddingProvider::new(32);

        let doc = Document::new("test");
        let id = doc.id.clone();
        let tokens = provider.embed_tokens("test").await.unwrap();
        store
            .insert(MultiVectorDocument::new(doc, tokens))
            .await
            .unwrap();

        assert!(store.delete(&id).await.unwrap());
        assert_eq!(store.count().await, 0);

        // Delete non-existent
        assert!(!store.delete(&id).await.unwrap());
    }

    #[tokio::test]
    async fn test_in_memory_multi_vector_store_clear() {
        let mut store = InMemoryMultiVectorStore::new(32);
        let provider = MockTokenEmbeddingProvider::new(32);

        for i in 0..5 {
            let doc = Document::new(format!("doc {i}"));
            let tokens = provider.embed_tokens(&format!("doc {i}")).await.unwrap();
            store
                .insert(MultiVectorDocument::new(doc, tokens))
                .await
                .unwrap();
        }

        assert_eq!(store.count().await, 5);

        store.clear().await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_in_memory_multi_vector_store_batch_insert() {
        let mut store = InMemoryMultiVectorStore::new(32);
        let provider = MockTokenEmbeddingProvider::new(32);

        let mut docs = Vec::new();
        for i in 0..3 {
            let doc = Document::new(format!("document {i}"));
            let tokens = provider
                .embed_tokens(&format!("document {i}"))
                .await
                .unwrap();
            docs.push(MultiVectorDocument::new(doc, tokens));
        }

        store.insert_batch(docs).await.unwrap();
        assert_eq!(store.count().await, 3);
    }

    #[test]
    fn test_multi_vector_search_result() {
        let doc = Document::new("test");
        let matches = vec![TokenMatch::new(0, 1, 0.9), TokenMatch::new(1, 2, 0.8)];

        let result = MultiVectorSearchResult::new(doc.clone(), 1.7, 0, matches);

        assert_eq!(result.document.content, "test");
        assert!((result.score - 1.7).abs() < 1e-6);
        assert_eq!(result.rank, 0);
        assert_eq!(result.token_matches.len(), 2);
    }

    #[test]
    fn test_store_metric_config() {
        let store = InMemoryMultiVectorStore::with_metric(64, SimilarityMetric::DotProduct);

        assert_eq!(store.dimension(), 64);
        assert_eq!(store.similarity_metric(), SimilarityMetric::DotProduct);
    }

    #[tokio::test]
    async fn test_mock_provider_custom_model_id() {
        let provider = MockTokenEmbeddingProvider::with_model_id(32, "custom-model");

        assert_eq!(provider.model_id(), "custom-model");
    }

    #[tokio::test]
    async fn test_search_empty_query() {
        let store = InMemoryMultiVectorStore::new(32);

        let results = store.search(&[], 10, None).await.unwrap();
        assert!(results.is_empty());
    }
}
