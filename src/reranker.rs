//! Reranking support for search results.
//!
//! Provides a flexible reranking system with multiple strategies:
//! - `MockReranker` - For testing purposes
//! - `KeywordReranker` - BM25-style keyword matching
//! - `SemanticReranker` - Uses embeddings for re-scoring
//! - `HybridReranker` - Combines keyword and semantic approaches
//! - `CrossEncoderReranker` - Interface for future ML-based rerankers
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::reranker::{Reranker, RerankerConfig, KeywordReranker};
//!
//! let config = RerankerConfig::default();
//! let reranker = KeywordReranker::new(config);
//!
//! let results = reranker.rerank(&query, search_results).await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::OxiRagError;
use crate::layer1_echo::EmbeddingProvider;
use crate::layer1_echo::similarity::cosine_similarity;
use crate::types::{Query, SearchResult};

/// Configuration for reranker behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Maximum number of results to rerank.
    pub top_k: usize,
    /// Minimum score threshold for results to be included.
    pub min_score_threshold: f32,
    /// Batch size for processing.
    pub batch_size: usize,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score_threshold: 0.0,
            batch_size: 32,
        }
    }
}

impl RerankerConfig {
    /// Create a new config with the specified `top_k`.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Create a new config with the specified `min_score_threshold`.
    #[must_use]
    pub fn with_min_score_threshold(mut self, threshold: f32) -> Self {
        self.min_score_threshold = threshold;
        self
    }

    /// Create a new config with the specified `batch_size`.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// Trait for reranking search results.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank a list of search results based on the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query used for reranking.
    /// * `results` - The search results to rerank.
    ///
    /// # Returns
    ///
    /// A reordered vector of search results with updated scores.
    async fn rerank(
        &self,
        query: &Query,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError>;

    /// Score a query-document pair.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text.
    /// * `document` - The document text.
    ///
    /// # Returns
    ///
    /// A relevance score for the pair (higher is better).
    async fn score_pair(&self, query: &str, document: &str) -> Result<f32, OxiRagError>;

    /// Get the reranker configuration.
    fn config(&self) -> &RerankerConfig;
}

/// Trait for cross-encoder based rerankers (interface for future ML-based implementations).
#[async_trait]
pub trait CrossEncoderReranker: Send + Sync {
    /// Encode a query-document pair and return a relevance score.
    ///
    /// Cross-encoders process the query and document together,
    /// allowing for more accurate relevance scoring compared to
    /// bi-encoder approaches.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text.
    /// * `document` - The document text.
    ///
    /// # Returns
    ///
    /// A relevance score for the pair.
    async fn encode_pair(&self, query: &str, document: &str) -> Result<f32, OxiRagError>;

    /// Batch encode multiple query-document pairs.
    ///
    /// # Arguments
    ///
    /// * `pairs` - Vector of (query, document) tuples.
    ///
    /// # Returns
    ///
    /// A vector of relevance scores.
    async fn encode_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, OxiRagError>;
}

/// A mock reranker for testing purposes.
#[derive(Debug, Clone)]
pub struct MockReranker {
    config: RerankerConfig,
    /// Score multiplier for testing.
    score_multiplier: f32,
}

impl MockReranker {
    /// Create a new mock reranker with default config.
    #[must_use]
    pub fn new(config: RerankerConfig) -> Self {
        Self {
            config,
            score_multiplier: 1.0,
        }
    }

    /// Set the score multiplier.
    #[must_use]
    pub fn with_score_multiplier(mut self, multiplier: f32) -> Self {
        self.score_multiplier = multiplier;
        self
    }
}

impl Default for MockReranker {
    fn default() -> Self {
        Self::new(RerankerConfig::default())
    }
}

#[async_trait]
impl Reranker for MockReranker {
    async fn rerank(
        &self,
        _query: &Query,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError> {
        // Apply score multiplier and filter
        for result in &mut results {
            result.score *= self.score_multiplier;
        }

        // Filter by threshold
        results.retain(|r| r.score >= self.config.min_score_threshold);

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        results.truncate(self.config.top_k);

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(results)
    }

    async fn score_pair(&self, _query: &str, _document: &str) -> Result<f32, OxiRagError> {
        // Return a fixed score for testing
        Ok(0.5 * self.score_multiplier)
    }

    fn config(&self) -> &RerankerConfig {
        &self.config
    }
}

/// A mock cross-encoder reranker for testing.
#[derive(Debug, Clone)]
pub struct MockCrossEncoderReranker {
    /// Base score to return for all pairs.
    base_score: f32,
}

impl MockCrossEncoderReranker {
    /// Create a new mock cross-encoder reranker.
    #[must_use]
    pub fn new(base_score: f32) -> Self {
        Self { base_score }
    }
}

impl Default for MockCrossEncoderReranker {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[async_trait]
impl CrossEncoderReranker for MockCrossEncoderReranker {
    async fn encode_pair(&self, query: &str, document: &str) -> Result<f32, OxiRagError> {
        // Simple mock: score based on word overlap
        let query_lower = query.to_lowercase();
        let doc_lower = document.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();
        let doc_words: std::collections::HashSet<&str> = doc_lower.split_whitespace().collect();

        let overlap = query_words.intersection(&doc_words).count();
        let union = query_words.union(&doc_words).count();

        if union == 0 {
            return Ok(self.base_score);
        }

        #[allow(clippy::cast_precision_loss)]
        let jaccard = overlap as f32 / union as f32;
        Ok(self.base_score + jaccard * (1.0 - self.base_score))
    }

    async fn encode_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, OxiRagError> {
        let mut scores = Vec::with_capacity(pairs.len());
        for (query, document) in pairs {
            scores.push(self.encode_pair(query, document).await?);
        }
        Ok(scores)
    }
}

/// BM25-style keyword reranker.
///
/// Uses term frequency and inverse document frequency to score
/// query-document relevance.
#[derive(Debug, Clone)]
pub struct KeywordReranker {
    config: RerankerConfig,
    /// BM25 k1 parameter (term frequency saturation).
    k1: f32,
    /// BM25 b parameter (length normalization).
    b: f32,
}

impl KeywordReranker {
    /// Create a new keyword reranker.
    #[must_use]
    pub fn new(config: RerankerConfig) -> Self {
        Self {
            config,
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Set BM25 k1 parameter.
    #[must_use]
    pub fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    /// Set BM25 b parameter.
    #[must_use]
    pub fn with_b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    /// Compute BM25 score for a query-document pair.
    #[allow(clippy::cast_precision_loss)]
    fn compute_bm25(&self, query: &str, document: &str, avg_doc_len: f32) -> f32 {
        let query_lower = query.to_lowercase();
        let doc_lower = document.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
        let doc_terms: Vec<&str> = doc_lower.split_whitespace().collect();
        let doc_len = doc_terms.len() as f32;

        // Build term frequency map for document
        let mut tf_map: HashMap<&str, usize> = HashMap::new();
        for term in &doc_terms {
            *tf_map.entry(*term).or_insert(0) += 1;
        }

        let mut score = 0.0;
        for term in &query_terms {
            if let Some(&tf) = tf_map.get(term) {
                let tf_score = tf as f32;
                // BM25 term frequency component
                let numerator = tf_score * (self.k1 + 1.0);
                let denominator =
                    tf_score + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len));
                score += numerator / denominator;
            }
        }

        score
    }
}

impl Default for KeywordReranker {
    fn default() -> Self {
        Self::new(RerankerConfig::default())
    }
}

#[async_trait]
impl Reranker for KeywordReranker {
    async fn rerank(
        &self,
        query: &Query,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError> {
        if results.is_empty() {
            return Ok(results);
        }

        // Calculate average document length
        let total_len: usize = results.iter().map(|r| r.document.content.len()).sum();
        #[allow(clippy::cast_precision_loss)]
        let avg_doc_len = total_len as f32 / results.len() as f32;

        // Score each document
        for result in &mut results {
            let bm25_score = self.compute_bm25(&query.text, &result.document.content, avg_doc_len);
            // Combine BM25 with original score (weighted average)
            result.score = 0.5 * result.score + 0.5 * bm25_score.min(1.0);
        }

        // Filter by threshold
        results.retain(|r| r.score >= self.config.min_score_threshold);

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        results.truncate(self.config.top_k);

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(results)
    }

    async fn score_pair(&self, query: &str, document: &str) -> Result<f32, OxiRagError> {
        // Assume average document length of 100 words for single pair scoring
        let avg_doc_len = 100.0;
        Ok(self.compute_bm25(query, document, avg_doc_len))
    }

    fn config(&self) -> &RerankerConfig {
        &self.config
    }
}

/// Semantic reranker using embeddings.
///
/// Re-scores documents using cosine similarity between
/// query and document embeddings.
pub struct SemanticReranker<E: EmbeddingProvider> {
    config: RerankerConfig,
    embedding_provider: E,
}

impl<E: EmbeddingProvider> SemanticReranker<E> {
    /// Create a new semantic reranker.
    #[must_use]
    pub fn new(config: RerankerConfig, embedding_provider: E) -> Self {
        Self {
            config,
            embedding_provider,
        }
    }

    /// Get a reference to the embedding provider.
    #[must_use]
    pub fn embedding_provider(&self) -> &E {
        &self.embedding_provider
    }
}

#[async_trait]
impl<E: EmbeddingProvider> Reranker for SemanticReranker<E> {
    async fn rerank(
        &self,
        query: &Query,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError> {
        if results.is_empty() {
            return Ok(results);
        }

        // Get query embedding
        let query_embedding = self
            .embedding_provider
            .embed(&query.text)
            .await
            .map_err(OxiRagError::Embedding)?;

        // Get document embeddings in batches
        let mut doc_embeddings = Vec::with_capacity(results.len());
        for chunk in results.chunks(self.config.batch_size) {
            let texts: Vec<&str> = chunk.iter().map(|r| r.document.content.as_str()).collect();
            let embeddings = self
                .embedding_provider
                .embed_batch(&texts)
                .await
                .map_err(OxiRagError::Embedding)?;
            doc_embeddings.extend(embeddings);
        }

        // Score each document using cosine similarity
        for (result, doc_embedding) in results.iter_mut().zip(doc_embeddings.iter()) {
            let semantic_score = cosine_similarity(&query_embedding, doc_embedding);
            // Combine semantic score with original score
            result.score = 0.5 * result.score + 0.5 * semantic_score;
        }

        // Filter by threshold
        results.retain(|r| r.score >= self.config.min_score_threshold);

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        results.truncate(self.config.top_k);

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(results)
    }

    async fn score_pair(&self, query: &str, document: &str) -> Result<f32, OxiRagError> {
        let query_embedding = self
            .embedding_provider
            .embed(query)
            .await
            .map_err(OxiRagError::Embedding)?;
        let doc_embedding = self
            .embedding_provider
            .embed(document)
            .await
            .map_err(OxiRagError::Embedding)?;

        Ok(cosine_similarity(&query_embedding, &doc_embedding))
    }

    fn config(&self) -> &RerankerConfig {
        &self.config
    }
}

/// Hybrid reranker combining keyword and semantic approaches.
///
/// Uses a weighted combination of BM25 keyword matching and
/// semantic similarity for more robust reranking.
pub struct HybridReranker<E: EmbeddingProvider> {
    config: RerankerConfig,
    keyword_reranker: KeywordReranker,
    semantic_reranker: SemanticReranker<E>,
    /// Weight for keyword score (0.0 to 1.0).
    keyword_weight: f32,
}

impl<E: EmbeddingProvider> HybridReranker<E> {
    /// Create a new hybrid reranker.
    #[must_use]
    pub fn new(config: RerankerConfig, embedding_provider: E) -> Self {
        let keyword_reranker = KeywordReranker::new(config.clone());
        let semantic_reranker = SemanticReranker::new(config.clone(), embedding_provider);
        Self {
            config,
            keyword_reranker,
            semantic_reranker,
            keyword_weight: 0.3,
        }
    }

    /// Set the keyword weight (0.0 to 1.0).
    ///
    /// The semantic weight will be (1.0 - `keyword_weight`).
    #[must_use]
    pub fn with_keyword_weight(mut self, weight: f32) -> Self {
        self.keyword_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Get the keyword weight.
    #[must_use]
    pub fn keyword_weight(&self) -> f32 {
        self.keyword_weight
    }

    /// Get the semantic weight.
    #[must_use]
    pub fn semantic_weight(&self) -> f32 {
        1.0 - self.keyword_weight
    }
}

#[async_trait]
impl<E: EmbeddingProvider> Reranker for HybridReranker<E> {
    async fn rerank(
        &self,
        query: &Query,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError> {
        if results.is_empty() {
            return Ok(results);
        }

        // Get keyword scores
        let keyword_results = self.keyword_reranker.rerank(query, results.clone()).await?;
        let keyword_scores: HashMap<String, f32> = keyword_results
            .iter()
            .map(|r| (r.document.id.as_str().to_string(), r.score))
            .collect();

        // Get semantic scores
        let semantic_results = self.semantic_reranker.rerank(query, results).await?;
        let mut final_results: Vec<SearchResult> = semantic_results
            .into_iter()
            .map(|mut r| {
                let keyword_score = keyword_scores
                    .get(r.document.id.as_str())
                    .copied()
                    .unwrap_or(0.0);
                let semantic_score = r.score;
                r.score =
                    self.keyword_weight * keyword_score + self.semantic_weight() * semantic_score;
                r
            })
            .collect();

        // Filter by threshold
        final_results.retain(|r| r.score >= self.config.min_score_threshold);

        // Sort by score descending
        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        final_results.truncate(self.config.top_k);

        // Update ranks
        for (i, result) in final_results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(final_results)
    }

    async fn score_pair(&self, query: &str, document: &str) -> Result<f32, OxiRagError> {
        let keyword_score = self.keyword_reranker.score_pair(query, document).await?;
        let semantic_score = self.semantic_reranker.score_pair(query, document).await?;

        Ok(self.keyword_weight * keyword_score + self.semantic_weight() * semantic_score)
    }

    fn config(&self) -> &RerankerConfig {
        &self.config
    }
}

/// Fusion strategy for combining scores from multiple rerankers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FusionStrategy {
    /// Weighted average of scores.
    #[default]
    Weighted,
    /// Cascade: use next reranker only for top results.
    Cascade,
    /// Reciprocal Rank Fusion (RRF).
    ReciprocalRankFusion,
}

/// Pipeline for chaining multiple rerankers.
pub struct RerankerPipeline {
    config: RerankerConfig,
    rerankers: Vec<Box<dyn Reranker>>,
    weights: Vec<f32>,
    fusion_strategy: FusionStrategy,
    /// Number of results to pass to next stage (for cascade).
    cascade_top_k: usize,
}

impl RerankerPipeline {
    /// Create a new reranker pipeline.
    #[must_use]
    pub fn new(config: RerankerConfig) -> Self {
        Self {
            config,
            rerankers: Vec::new(),
            weights: Vec::new(),
            fusion_strategy: FusionStrategy::Weighted,
            cascade_top_k: 20,
        }
    }

    /// Add a reranker to the pipeline with a weight.
    pub fn add_reranker(&mut self, reranker: Box<dyn Reranker>, weight: f32) {
        self.rerankers.push(reranker);
        self.weights.push(weight);
    }

    /// Set the fusion strategy.
    #[must_use]
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Set the cascade `top_k` (for cascade strategy).
    #[must_use]
    pub fn with_cascade_top_k(mut self, top_k: usize) -> Self {
        self.cascade_top_k = top_k;
        self
    }

    /// Get the number of rerankers in the pipeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rerankers.len()
    }

    /// Check if the pipeline is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rerankers.is_empty()
    }

    /// Execute the pipeline with weighted fusion.
    async fn execute_weighted(
        &self,
        query: &Query,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError> {
        if self.rerankers.is_empty() {
            return Ok(results);
        }

        // Normalize weights
        let total_weight: f32 = self.weights.iter().sum();
        #[allow(clippy::cast_precision_loss)]
        let normalized_weights: Vec<f32> = if total_weight > 0.0 {
            self.weights.iter().map(|w| w / total_weight).collect()
        } else {
            vec![1.0 / self.rerankers.len() as f32; self.rerankers.len()]
        };

        // Collect scores from all rerankers
        let mut all_scores: HashMap<String, Vec<f32>> = HashMap::new();
        for result in &results {
            all_scores.insert(result.document.id.as_str().to_string(), Vec::new());
        }

        for reranker in &self.rerankers {
            let reranked = reranker.rerank(query, results.clone()).await?;
            for result in &reranked {
                if let Some(scores) = all_scores.get_mut(result.document.id.as_str()) {
                    scores.push(result.score);
                }
            }
        }

        // Combine scores with weighted average
        let mut final_results: Vec<SearchResult> = results
            .into_iter()
            .map(|mut r| {
                if let Some(scores) = all_scores.get(r.document.id.as_str()) {
                    let combined: f32 = scores
                        .iter()
                        .zip(normalized_weights.iter())
                        .map(|(s, w)| s * w)
                        .sum();
                    r.score = combined;
                }
                r
            })
            .collect();

        self.finalize_results(&mut final_results);
        Ok(final_results)
    }

    /// Execute the pipeline with cascade strategy.
    async fn execute_cascade(
        &self,
        query: &Query,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError> {
        for reranker in &self.rerankers {
            results.truncate(self.cascade_top_k);
            results = reranker.rerank(query, results).await?;
        }

        self.finalize_results(&mut results);
        Ok(results)
    }

    /// Execute the pipeline with Reciprocal Rank Fusion.
    async fn execute_rrf(
        &self,
        query: &Query,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError> {
        const K: f32 = 60.0; // RRF constant

        if self.rerankers.is_empty() {
            return Ok(results);
        }

        // Collect ranks from all rerankers
        let mut rrf_scores: HashMap<String, f32> = HashMap::new();

        for reranker in &self.rerankers {
            let reranked = reranker.rerank(query, results.clone()).await?;
            for result in &reranked {
                let doc_id = result.document.id.as_str().to_string();
                #[allow(clippy::cast_precision_loss)]
                let rank = result.rank as f32;
                let rrf_contribution = 1.0 / (K + rank + 1.0);
                *rrf_scores.entry(doc_id).or_insert(0.0) += rrf_contribution;
            }
        }

        // Apply RRF scores
        let mut final_results: Vec<SearchResult> = results
            .into_iter()
            .map(|mut r| {
                if let Some(&rrf_score) = rrf_scores.get(r.document.id.as_str()) {
                    r.score = rrf_score;
                }
                r
            })
            .collect();

        self.finalize_results(&mut final_results);
        Ok(final_results)
    }

    /// Finalize results: filter, sort, truncate, and update ranks.
    fn finalize_results(&self, results: &mut Vec<SearchResult>) {
        // Filter by threshold
        results.retain(|r| r.score >= self.config.min_score_threshold);

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        results.truncate(self.config.top_k);

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
    }
}

#[async_trait]
impl Reranker for RerankerPipeline {
    async fn rerank(
        &self,
        query: &Query,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, OxiRagError> {
        match self.fusion_strategy {
            FusionStrategy::Weighted => self.execute_weighted(query, results).await,
            FusionStrategy::Cascade => self.execute_cascade(query, results).await,
            FusionStrategy::ReciprocalRankFusion => self.execute_rrf(query, results).await,
        }
    }

    async fn score_pair(&self, query: &str, document: &str) -> Result<f32, OxiRagError> {
        if self.rerankers.is_empty() {
            return Ok(0.0);
        }

        // Use first reranker for pair scoring
        self.rerankers[0].score_pair(query, document).await
    }

    fn config(&self) -> &RerankerConfig {
        &self.config
    }
}

/// Builder for creating a reranker pipeline.
pub struct RerankerPipelineBuilder {
    config: RerankerConfig,
    rerankers: Vec<(Box<dyn Reranker>, f32)>,
    fusion_strategy: FusionStrategy,
    cascade_top_k: usize,
}

impl RerankerPipelineBuilder {
    /// Create a new pipeline builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RerankerConfig::default(),
            rerankers: Vec::new(),
            fusion_strategy: FusionStrategy::Weighted,
            cascade_top_k: 20,
        }
    }

    /// Set the config.
    #[must_use]
    pub fn with_config(mut self, config: RerankerConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a reranker with weight.
    #[must_use]
    pub fn add_reranker(mut self, reranker: Box<dyn Reranker>, weight: f32) -> Self {
        self.rerankers.push((reranker, weight));
        self
    }

    /// Set the fusion strategy.
    #[must_use]
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Set the cascade `top_k`.
    #[must_use]
    pub fn with_cascade_top_k(mut self, top_k: usize) -> Self {
        self.cascade_top_k = top_k;
        self
    }

    /// Build the pipeline.
    #[must_use]
    pub fn build(self) -> RerankerPipeline {
        let mut pipeline = RerankerPipeline::new(self.config)
            .with_fusion_strategy(self.fusion_strategy)
            .with_cascade_top_k(self.cascade_top_k);

        for (reranker, weight) in self.rerankers {
            pipeline.add_reranker(reranker, weight);
        }

        pipeline
    }
}

impl Default for RerankerPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::similar_names)]
mod tests {
    use super::*;
    use crate::layer1_echo::MockEmbeddingProvider;
    use crate::types::Document;

    fn create_test_results() -> Vec<SearchResult> {
        vec![
            SearchResult::new(
                Document::new("Rust is a systems programming language"),
                0.9,
                0,
            ),
            SearchResult::new(Document::new("Python is great for data science"), 0.7, 1),
            SearchResult::new(Document::new("JavaScript runs in browsers"), 0.5, 2),
            SearchResult::new(Document::new("Rust prevents memory safety issues"), 0.8, 3),
        ]
    }

    #[tokio::test]
    async fn test_mock_reranker() {
        let config = RerankerConfig::default().with_top_k(3);
        let reranker = MockReranker::new(config);

        let query = Query::new("programming languages");
        let results = create_test_results();

        let reranked = reranker.rerank(&query, results).await.unwrap();

        assert_eq!(reranked.len(), 3);
        assert!(reranked[0].score >= reranked[1].score);
        assert!(reranked[1].score >= reranked[2].score);

        // Check ranks are updated
        for (i, r) in reranked.iter().enumerate() {
            assert_eq!(r.rank, i);
        }
    }

    #[tokio::test]
    async fn test_mock_reranker_score_multiplier() {
        let config = RerankerConfig::default();
        let reranker = MockReranker::new(config).with_score_multiplier(0.5);

        let query = Query::new("test");
        let results = vec![SearchResult::new(Document::new("test document"), 1.0, 0)];

        let reranked = reranker.rerank(&query, results).await.unwrap();

        assert!((reranked[0].score - 0.5).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mock_reranker_min_score_threshold() {
        let config = RerankerConfig::default().with_min_score_threshold(0.6);
        let reranker = MockReranker::new(config);

        let query = Query::new("test");
        let results = create_test_results();

        let reranked = reranker.rerank(&query, results).await.unwrap();

        for r in &reranked {
            assert!(r.score >= 0.6);
        }
    }

    #[tokio::test]
    async fn test_mock_cross_encoder() {
        let encoder = MockCrossEncoderReranker::default();

        let score = encoder
            .encode_pair("Rust programming", "Rust is great")
            .await
            .unwrap();
        assert!(score > 0.5); // Should have some overlap

        let score_no_overlap = encoder
            .encode_pair("Rust programming", "cats and dogs")
            .await
            .unwrap();
        assert!(score_no_overlap < score); // Less overlap
    }

    #[tokio::test]
    async fn test_mock_cross_encoder_batch() {
        let encoder = MockCrossEncoderReranker::default();

        let pairs = vec![("query1", "document1 query1"), ("query2", "unrelated text")];

        let scores = encoder.encode_pairs(&pairs).await.unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]); // First pair has overlap
    }

    #[tokio::test]
    async fn test_keyword_reranker() {
        let config = RerankerConfig::default().with_top_k(3);
        let reranker = KeywordReranker::new(config);

        let query = Query::new("Rust programming language");
        let results = create_test_results();

        let reranked = reranker.rerank(&query, results).await.unwrap();

        assert_eq!(reranked.len(), 3);
        // Rust-related documents should rank higher
        assert!(reranked[0].document.content.to_lowercase().contains("rust"));
    }

    #[tokio::test]
    async fn test_keyword_reranker_score_pair() {
        let reranker = KeywordReranker::default();

        let score = reranker
            .score_pair("Rust programming", "Rust is a programming language")
            .await
            .unwrap();
        assert!(score > 0.0);

        let score_no_match = reranker
            .score_pair("Rust programming", "cats and dogs")
            .await
            .unwrap();
        assert!(score_no_match < score);
    }

    #[tokio::test]
    async fn test_semantic_reranker() {
        let config = RerankerConfig::default().with_top_k(3);
        let embedding_provider = MockEmbeddingProvider::new(64);
        let reranker = SemanticReranker::new(config, embedding_provider);

        let query = Query::new("programming");
        let results = create_test_results();

        let reranked = reranker.rerank(&query, results).await.unwrap();

        assert_eq!(reranked.len(), 3);
        // Results should be sorted by score
        assert!(reranked[0].score >= reranked[1].score);
        assert!(reranked[1].score >= reranked[2].score);
    }

    #[tokio::test]
    async fn test_hybrid_reranker() {
        let config = RerankerConfig::default().with_top_k(3);
        let embedding_provider = MockEmbeddingProvider::new(64);
        let reranker = HybridReranker::new(config, embedding_provider).with_keyword_weight(0.4);

        assert!((reranker.keyword_weight() - 0.4).abs() < 0.001);
        assert!((reranker.semantic_weight() - 0.6).abs() < 0.001);

        let query = Query::new("Rust programming");
        let results = create_test_results();

        let reranked = reranker.rerank(&query, results).await.unwrap();

        assert_eq!(reranked.len(), 3);
    }

    #[tokio::test]
    async fn test_reranker_pipeline_weighted() {
        let config = RerankerConfig::default().with_top_k(3);

        let mut pipeline =
            RerankerPipeline::new(config).with_fusion_strategy(FusionStrategy::Weighted);

        pipeline.add_reranker(Box::new(MockReranker::default()), 1.0);
        pipeline.add_reranker(Box::new(KeywordReranker::default()), 1.0);

        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());

        let query = Query::new("Rust");
        let results = create_test_results();

        let reranked = pipeline.rerank(&query, results).await.unwrap();
        assert_eq!(reranked.len(), 3);
    }

    #[tokio::test]
    async fn test_reranker_pipeline_cascade() {
        let config = RerankerConfig::default().with_top_k(2);

        let mut pipeline = RerankerPipeline::new(config)
            .with_fusion_strategy(FusionStrategy::Cascade)
            .with_cascade_top_k(3);

        pipeline.add_reranker(Box::new(MockReranker::default()), 1.0);
        pipeline.add_reranker(Box::new(KeywordReranker::default()), 1.0);

        let query = Query::new("Rust");
        let results = create_test_results();

        let reranked = pipeline.rerank(&query, results).await.unwrap();
        assert_eq!(reranked.len(), 2);
    }

    #[tokio::test]
    async fn test_reranker_pipeline_rrf() {
        let config = RerankerConfig::default().with_top_k(3);

        let mut pipeline = RerankerPipeline::new(config)
            .with_fusion_strategy(FusionStrategy::ReciprocalRankFusion);

        pipeline.add_reranker(Box::new(MockReranker::default()), 1.0);
        pipeline.add_reranker(Box::new(KeywordReranker::default()), 1.0);

        let query = Query::new("Rust");
        let results = create_test_results();

        let reranked = pipeline.rerank(&query, results).await.unwrap();
        assert_eq!(reranked.len(), 3);
    }

    #[tokio::test]
    async fn test_reranker_pipeline_builder() {
        let config = RerankerConfig::default().with_top_k(2);

        let pipeline = RerankerPipelineBuilder::new()
            .with_config(config)
            .with_fusion_strategy(FusionStrategy::Cascade)
            .with_cascade_top_k(3)
            .add_reranker(Box::new(MockReranker::default()), 1.0)
            .add_reranker(Box::new(KeywordReranker::default()), 1.0)
            .build();

        assert_eq!(pipeline.len(), 2);

        let query = Query::new("test");
        let results = create_test_results();

        let reranked = pipeline.rerank(&query, results).await.unwrap();
        assert!(reranked.len() <= 2);
    }

    #[tokio::test]
    async fn test_reranker_pipeline_empty() {
        let config = RerankerConfig::default();
        let pipeline = RerankerPipeline::new(config);

        assert!(pipeline.is_empty());

        let query = Query::new("test");
        let results = create_test_results();

        let reranked = pipeline.rerank(&query, results.clone()).await.unwrap();
        assert_eq!(reranked.len(), results.len());
    }

    #[tokio::test]
    async fn test_reranker_empty_results() {
        let reranker = MockReranker::default();

        let query = Query::new("test");
        let results: Vec<SearchResult> = vec![];

        let reranked = reranker.rerank(&query, results).await.unwrap();
        assert!(reranked.is_empty());
    }

    #[tokio::test]
    async fn test_reranker_config_builder() {
        let config = RerankerConfig::default()
            .with_top_k(5)
            .with_min_score_threshold(0.3)
            .with_batch_size(16);

        assert_eq!(config.top_k, 5);
        assert!((config.min_score_threshold - 0.3).abs() < 0.001);
        assert_eq!(config.batch_size, 16);
    }

    #[tokio::test]
    async fn test_result_reordering() {
        let config = RerankerConfig::default().with_top_k(10);
        let reranker = MockReranker::new(config);

        let query = Query::new("test");
        let results = vec![
            SearchResult::new(Document::new("doc1"), 0.3, 0),
            SearchResult::new(Document::new("doc2"), 0.9, 1),
            SearchResult::new(Document::new("doc3"), 0.6, 2),
        ];

        let reranked = reranker.rerank(&query, results).await.unwrap();

        // Should be reordered by score
        assert!((reranked[0].score - 0.9).abs() < 0.001);
        assert!((reranked[1].score - 0.6).abs() < 0.001);
        assert!((reranked[2].score - 0.3).abs() < 0.001);

        // Ranks should be updated
        assert_eq!(reranked[0].rank, 0);
        assert_eq!(reranked[1].rank, 1);
        assert_eq!(reranked[2].rank, 2);
    }

    #[tokio::test]
    async fn test_score_adjustments() {
        let config = RerankerConfig::default();
        let reranker = MockReranker::new(config).with_score_multiplier(2.0);

        let query = Query::new("test");
        let results = vec![SearchResult::new(Document::new("doc"), 0.4, 0)];

        let reranked = reranker.rerank(&query, results).await.unwrap();

        assert!((reranked[0].score - 0.8).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_top_k_filtering() {
        let config = RerankerConfig::default().with_top_k(2);
        let reranker = MockReranker::new(config);

        let query = Query::new("test");
        let results = vec![
            SearchResult::new(Document::new("doc1"), 0.9, 0),
            SearchResult::new(Document::new("doc2"), 0.8, 1),
            SearchResult::new(Document::new("doc3"), 0.7, 2),
            SearchResult::new(Document::new("doc4"), 0.6, 3),
        ];

        let reranked = reranker.rerank(&query, results).await.unwrap();

        assert_eq!(reranked.len(), 2);
        assert!((reranked[0].score - 0.9).abs() < 0.001);
        assert!((reranked[1].score - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_fusion_strategy_default() {
        let strategy = FusionStrategy::default();
        assert_eq!(strategy, FusionStrategy::Weighted);
    }

    #[tokio::test]
    async fn test_score_pair_mock() {
        let reranker = MockReranker::default().with_score_multiplier(1.5);

        let score = reranker.score_pair("query", "document").await.unwrap();
        assert!((score - 0.75).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_pipeline_score_pair() {
        let config = RerankerConfig::default();
        let mut pipeline = RerankerPipeline::new(config);

        // Empty pipeline
        let score = pipeline.score_pair("query", "document").await.unwrap();
        assert!((score - 0.0).abs() < 0.001);

        // With reranker
        pipeline.add_reranker(Box::new(KeywordReranker::default()), 1.0);
        let score = pipeline
            .score_pair("Rust programming", "Rust is great")
            .await
            .unwrap();
        assert!(score > 0.0);
    }
}
