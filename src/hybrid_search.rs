//! Hybrid search combining dense (vector) and sparse (BM25) retrieval.
//!
//! This module provides a comprehensive hybrid search implementation that combines:
//! - Dense retrieval using embedding vectors (semantic similarity)
//! - Sparse retrieval using BM25 weighting (lexical matching)
//!
//! The combination of both approaches typically yields better retrieval performance
//! than either method alone, as they capture complementary aspects of relevance.

#![allow(clippy::cast_precision_loss)] // Intentional: usize to f32 for scoring

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;

use crate::error::VectorStoreError;
use crate::types::DocumentId;

/// A sparse vector representation for BM25-based retrieval.
///
/// Sparse vectors are efficient for representing term-based document features
/// where most dimensions are zero. Only non-zero values are stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Indices of non-zero elements.
    pub indices: Vec<usize>,
    /// Values at the corresponding indices.
    pub values: Vec<f32>,
    /// The total dimension of the vector space.
    pub dimension: usize,
}

impl SparseVector {
    /// Create a new sparse vector.
    ///
    /// # Arguments
    /// * `indices` - Indices of non-zero elements (must be sorted and unique)
    /// * `values` - Values at the corresponding indices
    /// * `dimension` - Total dimension of the vector space
    ///
    /// # Panics
    /// Panics if indices and values have different lengths.
    #[must_use]
    pub fn new(indices: Vec<usize>, values: Vec<f32>, dimension: usize) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Indices and values must have the same length"
        );
        Self {
            indices,
            values,
            dimension,
        }
    }

    /// Create an empty sparse vector with the given dimension.
    #[must_use]
    pub fn empty(dimension: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            dimension,
        }
    }

    /// Get the number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Check if the vector is empty (all zeros).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Compute the dot product with another sparse vector.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }

        result
    }

    /// Compute the L2 norm of the vector.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Compute cosine similarity with another sparse vector.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = self.dot(other);
        let norm_a = self.norm();
        let norm_b = other.norm();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Convert to a dense vector representation.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.dimension];
        for (idx, val) in self.indices.iter().zip(self.values.iter()) {
            if *idx < self.dimension {
                dense[*idx] = *val;
            }
        }
        dense
    }

    /// Create from a dense vector, keeping only non-zero values.
    #[must_use]
    pub fn from_dense(dense: &[f32]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &v) in dense.iter().enumerate() {
            if v.abs() > f32::EPSILON {
                indices.push(i);
                values.push(v);
            }
        }

        Self {
            indices,
            values,
            dimension: dense.len(),
        }
    }
}

impl Default for SparseVector {
    fn default() -> Self {
        Self::empty(0)
    }
}

/// Trait for sparse vector storage with similarity search.
#[async_trait]
pub trait SparseVectorStore: Send + Sync {
    /// Insert a document with its sparse vector representation.
    ///
    /// # Arguments
    /// * `id` - The document identifier
    /// * `vector` - The sparse vector representation
    async fn insert(
        &mut self,
        id: DocumentId,
        vector: SparseVector,
    ) -> Result<(), VectorStoreError>;

    /// Search for documents similar to the query vector.
    ///
    /// # Arguments
    /// * `query` - The query sparse vector
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    /// A vector of `(DocumentId, f32)` pairs sorted by descending score.
    ///
    /// # Errors
    /// Returns an error if the search operation fails.
    async fn search(
        &self,
        query: &SparseVector,
        top_k: usize,
    ) -> Result<Vec<(DocumentId, f32)>, VectorStoreError>;

    /// Get the sparse vector for a document.
    async fn get(&self, id: &DocumentId) -> Result<Option<SparseVector>, VectorStoreError>;

    /// Delete a document from the store.
    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError>;

    /// Get the number of documents in the store.
    async fn count(&self) -> usize;

    /// Clear all documents from the store.
    async fn clear(&mut self) -> Result<(), VectorStoreError>;
}

/// BM25 parameters for scoring.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BM25Params {
    /// Controls the impact of term frequency saturation (typically 1.2-2.0).
    pub k1: f32,
    /// Controls the impact of document length normalization (typically 0.75).
    pub b: f32,
    /// Smoothing factor for IDF calculation (typically 0.5).
    pub delta: f32,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            delta: 0.5,
        }
    }
}

/// BM25 encoder for converting text to sparse vectors.
///
/// Implements the BM25 weighting scheme with IDF (Inverse Document Frequency)
/// for effective lexical matching.
pub struct BM25Encoder {
    /// Vocabulary mapping from terms to indices.
    vocabulary: HashMap<String, usize>,
    /// Inverse document frequency for each term.
    idf: HashMap<String, f32>,
    /// Document frequency for each term.
    document_frequencies: HashMap<String, usize>,
    /// Total number of documents indexed.
    total_documents: usize,
    /// Average document length (in terms).
    average_doc_length: f32,
    /// Sum of all document lengths for computing average.
    total_terms: usize,
    /// BM25 parameters.
    params: BM25Params,
    /// Vocabulary size (dimension of sparse vectors).
    vocab_size: usize,
}

impl BM25Encoder {
    /// Create a new BM25 encoder with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
            average_doc_length: 0.0,
            total_terms: 0,
            params: BM25Params::default(),
            vocab_size: 0,
        }
    }

    /// Create a BM25 encoder with custom parameters.
    #[must_use]
    pub fn with_params(params: BM25Params) -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
            average_doc_length: 0.0,
            total_terms: 0,
            params,
            vocab_size: 0,
        }
    }

    /// Get the vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the total number of documents indexed.
    #[must_use]
    pub fn total_documents(&self) -> usize {
        self.total_documents
    }

    /// Get the BM25 parameters.
    #[must_use]
    pub fn params(&self) -> &BM25Params {
        &self.params
    }

    /// Tokenize text into terms.
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(String::from)
            .collect()
    }

    /// Fit the encoder on a corpus of documents.
    ///
    /// This builds the vocabulary and computes IDF values.
    pub fn fit(&mut self, documents: &[&str]) {
        self.vocabulary.clear();
        self.document_frequencies.clear();
        self.idf.clear();
        self.total_documents = documents.len();
        self.total_terms = 0;

        // First pass: build vocabulary and count document frequencies
        for doc in documents {
            let terms = Self::tokenize(doc);
            self.total_terms += terms.len();

            let unique_terms: HashSet<_> = terms.into_iter().collect();
            for term in unique_terms {
                // Add to vocabulary if new
                if !self.vocabulary.contains_key(&term) {
                    let idx = self.vocabulary.len();
                    self.vocabulary.insert(term.clone(), idx);
                }

                // Increment document frequency
                *self.document_frequencies.entry(term).or_insert(0) += 1;
            }
        }

        self.vocab_size = self.vocabulary.len();
        self.average_doc_length = if self.total_documents > 0 {
            self.total_terms as f32 / self.total_documents as f32
        } else {
            0.0
        };

        // Compute IDF for all terms using BM25 IDF variant with +1 smoothing
        let n = self.total_documents as f32;
        for (term, df) in &self.document_frequencies {
            let df_f = *df as f32;
            // BM25 IDF formula: log((N + 1) / (df + delta))
            // This variant ensures IDF is always positive
            let idf = ((n + 1.0) / (df_f + self.params.delta)).ln();
            self.idf.insert(term.clone(), idf.max(f32::EPSILON));
        }
    }

    /// Fit the encoder on owned strings.
    pub fn fit_owned(&mut self, documents: &[String]) {
        let refs: Vec<&str> = documents.iter().map(String::as_str).collect();
        self.fit(&refs);
    }

    /// Encode a single text into a sparse vector.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    ///
    /// # Returns
    /// A sparse vector with BM25-weighted term frequencies.
    #[must_use]
    pub fn encode(&self, text: &str) -> SparseVector {
        if self.vocab_size == 0 {
            return SparseVector::empty(0);
        }

        let terms = Self::tokenize(text);
        let doc_length = terms.len() as f32;

        // Count term frequencies
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for term in terms {
            *term_freqs.entry(term).or_insert(0) += 1;
        }

        // Compute BM25 scores
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (term, tf) in term_freqs {
            if let Some(&idx) = self.vocabulary.get(&term) {
                let idf = self.idf.get(&term).copied().unwrap_or(0.0);
                let tf_f = tf as f32;

                // BM25 term frequency component
                let length_norm = 1.0 - self.params.b
                    + self.params.b * doc_length / self.average_doc_length.max(1.0);
                let tf_score =
                    (tf_f * (self.params.k1 + 1.0)) / (tf_f + self.params.k1 * length_norm);

                let score = idf * tf_score;
                if score > 0.0 {
                    indices.push(idx);
                    values.push(score);
                }
            }
        }

        // Sort by indices for efficient sparse operations
        let mut pairs: Vec<_> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|(idx, _)| *idx);

        let (indices, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();

        SparseVector::new(indices, values, self.vocab_size)
    }

    /// Encode multiple texts into sparse vectors.
    ///
    /// # Arguments
    /// * `texts` - The texts to encode
    ///
    /// # Returns
    /// A vector of sparse vectors.
    #[must_use]
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<SparseVector> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Encode multiple owned strings into sparse vectors.
    #[must_use]
    pub fn encode_batch_owned(&self, texts: &[String]) -> Vec<SparseVector> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
}

impl Default for BM25Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from hybrid search combining dense and sparse scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    /// The document identifier.
    pub document_id: DocumentId,
    /// Score from dense (vector) retrieval.
    pub dense_score: f32,
    /// Score from sparse (BM25) retrieval.
    pub sparse_score: f32,
    /// Combined score after fusion.
    pub combined_score: f32,
    /// Rank in the final result set.
    pub rank: usize,
}

impl HybridResult {
    /// Create a new hybrid result.
    #[must_use]
    pub fn new(
        document_id: DocumentId,
        dense_score: f32,
        sparse_score: f32,
        combined_score: f32,
        rank: usize,
    ) -> Self {
        Self {
            document_id,
            dense_score,
            sparse_score,
            combined_score,
            rank,
        }
    }
}

/// Strategy for fusing dense and sparse retrieval scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Weighted sum of normalized scores.
    ///
    /// `Combined = dense_weight * dense_score + sparse_weight * sparse_score`
    WeightedSum {
        /// Weight for dense scores (0.0 to 1.0).
        dense_weight: f32,
        /// Weight for sparse scores (0.0 to 1.0).
        sparse_weight: f32,
    },

    /// Reciprocal Rank Fusion (RRF).
    ///
    /// `RRF = sum(1 / (k + rank_i))` for each ranking.
    /// This method is robust and doesn't require score normalization.
    ReciprocalRankFusion {
        /// The `k` parameter (typically 60).
        k: usize,
    },

    /// Distribution-based normalization fusion.
    ///
    /// Normalizes scores based on their distribution (mean and std dev)
    /// before combining with weights.
    DistributionBased {
        /// Weight for dense scores after normalization.
        dense_weight: f32,
        /// Weight for sparse scores after normalization.
        sparse_weight: f32,
    },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::WeightedSum {
            dense_weight: 0.5,
            sparse_weight: 0.5,
        }
    }
}

impl FusionStrategy {
    /// Create a weighted sum fusion strategy.
    #[must_use]
    pub fn weighted_sum(dense_weight: f32, sparse_weight: f32) -> Self {
        Self::WeightedSum {
            dense_weight,
            sparse_weight,
        }
    }

    /// Create a reciprocal rank fusion strategy.
    #[must_use]
    pub fn rrf(k: usize) -> Self {
        Self::ReciprocalRankFusion { k }
    }

    /// Create a distribution-based fusion strategy.
    #[must_use]
    pub fn distribution_based(dense_weight: f32, sparse_weight: f32) -> Self {
        Self::DistributionBased {
            dense_weight,
            sparse_weight,
        }
    }
}

/// Configuration for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// The fusion strategy to use.
    pub fusion_strategy: FusionStrategy,
    /// Weight for dense scores (used by some strategies).
    pub dense_weight: f32,
    /// Weight for sparse scores (used by some strategies).
    pub sparse_weight: f32,
    /// Whether to normalize scores before fusion.
    pub normalize_scores: bool,
    /// Minimum score threshold for results.
    pub min_score: Option<f32>,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: FusionStrategy::default(),
            dense_weight: 0.5,
            sparse_weight: 0.5,
            normalize_scores: true,
            min_score: None,
        }
    }
}

impl HybridConfig {
    /// Create a new hybrid config with weighted sum fusion.
    #[must_use]
    pub fn weighted_sum(dense_weight: f32, sparse_weight: f32) -> Self {
        Self {
            fusion_strategy: FusionStrategy::weighted_sum(dense_weight, sparse_weight),
            dense_weight,
            sparse_weight,
            normalize_scores: true,
            min_score: None,
        }
    }

    /// Create a new hybrid config with RRF fusion.
    #[must_use]
    pub fn rrf(k: usize) -> Self {
        Self {
            fusion_strategy: FusionStrategy::rrf(k),
            dense_weight: 0.5,
            sparse_weight: 0.5,
            normalize_scores: false,
            min_score: None,
        }
    }

    /// Set the minimum score threshold.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Set whether to normalize scores.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize_scores = normalize;
        self
    }
}

/// Hybrid searcher combining dense and sparse retrieval.
pub struct HybridSearcher<S: SparseVectorStore> {
    /// The sparse vector store for BM25-based retrieval.
    sparse_store: S,
    /// The BM25 encoder for query encoding.
    encoder: BM25Encoder,
    /// Configuration for hybrid search.
    config: HybridConfig,
}

impl<S: SparseVectorStore> HybridSearcher<S> {
    /// Create a new hybrid searcher.
    #[must_use]
    pub fn new(sparse_store: S, encoder: BM25Encoder, config: HybridConfig) -> Self {
        Self {
            sparse_store,
            encoder,
            config,
        }
    }

    /// Get a reference to the sparse store.
    #[must_use]
    pub fn sparse_store(&self) -> &S {
        &self.sparse_store
    }

    /// Get a mutable reference to the sparse store.
    pub fn sparse_store_mut(&mut self) -> &mut S {
        &mut self.sparse_store
    }

    /// Get a reference to the encoder.
    #[must_use]
    pub fn encoder(&self) -> &BM25Encoder {
        &self.encoder
    }

    /// Get a reference to the config.
    #[must_use]
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Perform hybrid search combining dense and sparse results.
    ///
    /// # Arguments
    /// * `query` - The text query for sparse search
    /// * `dense_results` - Pre-computed dense search results as `(id, score)` pairs
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    /// A vector of hybrid results sorted by combined score.
    ///
    /// # Errors
    /// Returns an error if the sparse search operation fails.
    pub async fn search(
        &self,
        query: &str,
        dense_results: &[(DocumentId, f32)],
        top_k: usize,
    ) -> Result<Vec<HybridResult>, VectorStoreError> {
        // Encode query for sparse search
        let sparse_query = self.encoder.encode(query);

        // Perform sparse search
        let sparse_results = self.sparse_store.search(&sparse_query, top_k * 2).await?;

        // Fuse results
        let results = self.fuse_results(dense_results, &sparse_results, top_k);

        Ok(results)
    }

    /// Perform hybrid search with a pre-encoded sparse query.
    ///
    /// # Arguments
    /// * `sparse_query` - Pre-encoded sparse query vector
    /// * `dense_results` - Pre-computed dense search results
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Errors
    /// Returns an error if the sparse search operation fails.
    pub async fn search_with_sparse(
        &self,
        sparse_query: &SparseVector,
        dense_results: &[(DocumentId, f32)],
        top_k: usize,
    ) -> Result<Vec<HybridResult>, VectorStoreError> {
        let sparse_results = self.sparse_store.search(sparse_query, top_k * 2).await?;
        let results = self.fuse_results(dense_results, &sparse_results, top_k);
        Ok(results)
    }

    /// Fuse dense and sparse results according to the configured strategy.
    fn fuse_results(
        &self,
        dense_results: &[(DocumentId, f32)],
        sparse_results: &[(DocumentId, f32)],
        top_k: usize,
    ) -> Vec<HybridResult> {
        match &self.config.fusion_strategy {
            FusionStrategy::WeightedSum {
                dense_weight,
                sparse_weight,
            } => self.fuse_weighted_sum(
                dense_results,
                sparse_results,
                *dense_weight,
                *sparse_weight,
                top_k,
            ),
            FusionStrategy::ReciprocalRankFusion { k } => {
                self.fuse_rrf(dense_results, sparse_results, *k, top_k)
            }
            FusionStrategy::DistributionBased {
                dense_weight,
                sparse_weight,
            } => self.fuse_distribution_based(
                dense_results,
                sparse_results,
                *dense_weight,
                *sparse_weight,
                top_k,
            ),
        }
    }

    /// Fuse using weighted sum.
    fn fuse_weighted_sum(
        &self,
        dense_results: &[(DocumentId, f32)],
        sparse_results: &[(DocumentId, f32)],
        dense_weight: f32,
        sparse_weight: f32,
        top_k: usize,
    ) -> Vec<HybridResult> {
        let mut scores: HashMap<DocumentId, (f32, f32)> = HashMap::new();

        // Normalize dense scores if configured
        let (dense_min, dense_max) = if self.config.normalize_scores {
            Self::score_range(dense_results)
        } else {
            (0.0, 1.0)
        };

        let (sparse_min, sparse_max) = if self.config.normalize_scores {
            Self::score_range(sparse_results)
        } else {
            (0.0, 1.0)
        };

        // Collect dense scores
        for (id, score) in dense_results {
            let normalized = Self::normalize_score(*score, dense_min, dense_max);
            scores.entry(id.clone()).or_insert((0.0, 0.0)).0 = normalized;
        }

        // Collect sparse scores
        for (id, score) in sparse_results {
            let normalized = Self::normalize_score(*score, sparse_min, sparse_max);
            scores.entry(id.clone()).or_insert((0.0, 0.0)).1 = normalized;
        }

        // Compute combined scores
        let mut results: Vec<HybridResult> = scores
            .into_iter()
            .map(|(id, (dense, sparse))| {
                let combined = dense_weight * dense + sparse_weight * sparse;
                HybridResult::new(id, dense, sparse, combined, 0)
            })
            .filter(|r| {
                self.config
                    .min_score
                    .is_none_or(|min| r.combined_score >= min)
            })
            .collect();

        // Sort by combined score (descending)
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks and truncate
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
        results.truncate(top_k);

        results
    }

    /// Fuse using Reciprocal Rank Fusion.
    fn fuse_rrf(
        &self,
        dense_results: &[(DocumentId, f32)],
        sparse_results: &[(DocumentId, f32)],
        k: usize,
        top_k: usize,
    ) -> Vec<HybridResult> {
        let mut rrf_scores: HashMap<DocumentId, (f32, f32, f32)> = HashMap::new();

        // Create sorted rankings
        let mut dense_sorted: Vec<_> = dense_results.to_vec();
        dense_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut sparse_sorted: Vec<_> = sparse_results.to_vec();
        sparse_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute RRF scores from dense ranking
        for (rank, (id, score)) in dense_sorted.iter().enumerate() {
            let rrf = 1.0 / (k as f32 + rank as f32 + 1.0);
            let entry = rrf_scores.entry(id.clone()).or_insert((0.0, 0.0, 0.0));
            entry.0 = *score; // dense score
            entry.2 += rrf; // combined RRF
        }

        // Compute RRF scores from sparse ranking
        for (rank, (id, score)) in sparse_sorted.iter().enumerate() {
            let rrf = 1.0 / (k as f32 + rank as f32 + 1.0);
            let entry = rrf_scores.entry(id.clone()).or_insert((0.0, 0.0, 0.0));
            entry.1 = *score; // sparse score
            entry.2 += rrf; // combined RRF
        }

        // Build results
        let mut results: Vec<HybridResult> = rrf_scores
            .into_iter()
            .map(|(id, (dense, sparse, combined))| {
                HybridResult::new(id, dense, sparse, combined, 0)
            })
            .filter(|r| {
                self.config
                    .min_score
                    .is_none_or(|min| r.combined_score >= min)
            })
            .collect();

        // Sort by RRF score (descending)
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks and truncate
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
        results.truncate(top_k);

        results
    }

    /// Fuse using distribution-based normalization.
    fn fuse_distribution_based(
        &self,
        dense_results: &[(DocumentId, f32)],
        sparse_results: &[(DocumentId, f32)],
        dense_weight: f32,
        sparse_weight: f32,
        top_k: usize,
    ) -> Vec<HybridResult> {
        let mut scores: HashMap<DocumentId, (f32, f32)> = HashMap::new();

        // Compute statistics for dense scores
        let (dense_mean, dense_std) = Self::score_stats(dense_results);
        let (sparse_mean, sparse_std) = Self::score_stats(sparse_results);

        // Collect and normalize dense scores
        for (id, score) in dense_results {
            let normalized = Self::z_normalize(*score, dense_mean, dense_std);
            scores.entry(id.clone()).or_insert((0.0, 0.0)).0 = normalized;
        }

        // Collect and normalize sparse scores
        for (id, score) in sparse_results {
            let normalized = Self::z_normalize(*score, sparse_mean, sparse_std);
            scores.entry(id.clone()).or_insert((0.0, 0.0)).1 = normalized;
        }

        // Compute combined scores
        let mut results: Vec<HybridResult> = scores
            .into_iter()
            .map(|(id, (dense, sparse))| {
                let combined = dense_weight * dense + sparse_weight * sparse;
                HybridResult::new(id, dense, sparse, combined, 0)
            })
            .filter(|r| {
                self.config
                    .min_score
                    .is_none_or(|min| r.combined_score >= min)
            })
            .collect();

        // Sort by combined score (descending)
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks and truncate
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
        results.truncate(top_k);

        results
    }

    /// Compute the min and max scores from results.
    fn score_range(results: &[(DocumentId, f32)]) -> (f32, f32) {
        if results.is_empty() {
            return (0.0, 1.0);
        }

        let min = results
            .iter()
            .map(|(_, s)| *s)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let max = results
            .iter()
            .map(|(_, s)| *s)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        (min, max)
    }

    /// Normalize a score to [0, 1] range.
    fn normalize_score(score: f32, min: f32, max: f32) -> f32 {
        if (max - min).abs() < f32::EPSILON {
            return 0.5;
        }
        (score - min) / (max - min)
    }

    /// Compute mean and standard deviation of scores.
    fn score_stats(results: &[(DocumentId, f32)]) -> (f32, f32) {
        if results.is_empty() {
            return (0.0, 1.0);
        }

        let scores: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
        let n = scores.len() as f32;
        let mean = scores.iter().sum::<f32>() / n;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt().max(f32::EPSILON);

        (mean, std)
    }

    /// Z-score normalization.
    fn z_normalize(score: f32, mean: f32, std: f32) -> f32 {
        (score - mean) / std
    }
}

/// In-memory sparse vector store for development and testing.
pub struct InMemorySparseStore {
    /// Stored sparse vectors by document ID.
    vectors: RwLock<HashMap<DocumentId, SparseVector>>,
    /// Inverted index: term index -> set of document IDs.
    inverted_index: RwLock<HashMap<usize, HashSet<DocumentId>>>,
}

impl InMemorySparseStore {
    /// Create a new in-memory sparse store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            inverted_index: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemorySparseStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SparseVectorStore for InMemorySparseStore {
    async fn insert(
        &mut self,
        id: DocumentId,
        vector: SparseVector,
    ) -> Result<(), VectorStoreError> {
        // Update inverted index
        {
            let mut index = self.inverted_index.write().await;
            for &idx in &vector.indices {
                index.entry(idx).or_default().insert(id.clone());
            }
        }

        // Store the vector
        {
            let mut vectors = self.vectors.write().await;
            vectors.insert(id, vector);
        }

        Ok(())
    }

    async fn search(
        &self,
        query: &SparseVector,
        top_k: usize,
    ) -> Result<Vec<(DocumentId, f32)>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        let inverted_index = self.inverted_index.read().await;

        // Find candidate documents using inverted index
        let mut candidates: HashSet<DocumentId> = HashSet::new();
        for &idx in &query.indices {
            if let Some(doc_ids) = inverted_index.get(&idx) {
                candidates.extend(doc_ids.iter().cloned());
            }
        }

        // Score candidates
        let mut scored: Vec<(DocumentId, f32)> = candidates
            .into_iter()
            .filter_map(|id| {
                vectors.get(&id).map(|vec| {
                    let score = query.dot(vec);
                    (id, score)
                })
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.truncate(top_k);
        Ok(scored)
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<SparseVector>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        Ok(vectors.get(id).cloned())
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        // Remove from vectors and get the vector for index cleanup
        let vector = {
            let mut vectors = self.vectors.write().await;
            vectors.remove(id)
        };

        if let Some(vec) = vector {
            // Clean up inverted index
            let mut index = self.inverted_index.write().await;
            for &idx in &vec.indices {
                if let Some(doc_ids) = index.get_mut(&idx) {
                    doc_ids.remove(id);
                    if doc_ids.is_empty() {
                        index.remove(&idx);
                    }
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn count(&self) -> usize {
        self.vectors.read().await.len()
    }

    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        self.vectors.write().await.clear();
        self.inverted_index.write().await.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_new() {
        let v = SparseVector::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0], 10);
        assert_eq!(v.indices, vec![0, 2, 5]);
        assert_eq!(v.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dimension, 10);
        assert_eq!(v.nnz(), 3);
    }

    #[test]
    fn test_sparse_vector_empty() {
        let v = SparseVector::empty(100);
        assert!(v.is_empty());
        assert_eq!(v.nnz(), 0);
        assert_eq!(v.dimension, 100);
    }

    #[test]
    fn test_sparse_vector_dot() {
        let a = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 10);
        let b = SparseVector::new(vec![1, 2, 4], vec![1.0, 2.0, 1.0], 10);
        // Matching indices: 2 (2.0*2.0=4.0) and 4 (3.0*1.0=3.0)
        assert!((a.dot(&b) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_norm() {
        let v = SparseVector::new(vec![0, 1], vec![3.0, 4.0], 10);
        assert!((v.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_cosine_similarity() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 0.0], 10);
        let b = SparseVector::new(vec![0, 1], vec![1.0, 0.0], 10);
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-6);

        let c = SparseVector::new(vec![0, 1], vec![0.0, 1.0], 10);
        assert!(a.cosine_similarity(&c).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_to_dense() {
        let v = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 5);
        let dense = v.to_dense();
        assert_eq!(dense, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_sparse_vector_from_dense() {
        let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let sparse = SparseVector::from_dense(&dense);
        assert_eq!(sparse.indices, vec![0, 2, 4]);
        assert_eq!(sparse.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(sparse.dimension, 5);
    }

    #[test]
    fn test_bm25_encoder_tokenize() {
        let tokens = BM25Encoder::tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(tokens.contains(&"is".to_string())); // "is" has length 2, so it's included
        // Single character words should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_bm25_encoder_fit() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["the quick brown fox", "the lazy dog", "quick brown dog"];
        encoder.fit(&docs);

        assert_eq!(encoder.total_documents(), 3);
        assert!(encoder.vocab_size() > 0);
        // "the" appears in 2 docs, should have lower IDF
        // "fox" appears in 1 doc, should have higher IDF
    }

    #[test]
    fn test_bm25_encoder_encode() {
        let mut encoder = BM25Encoder::new();
        let docs = vec![
            "the quick brown fox jumps",
            "the lazy dog sleeps",
            "quick brown dog runs",
        ];
        encoder.fit(&docs);

        let query = "quick brown";
        let sparse = encoder.encode(query);

        assert!(!sparse.is_empty());
        assert!(sparse.nnz() > 0);
    }

    #[test]
    fn test_bm25_encoder_encode_batch() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["document one", "document two"];
        encoder.fit(&docs);

        let queries = vec!["query one", "query two"];
        let vectors = encoder.encode_batch(&queries);

        assert_eq!(vectors.len(), 2);
    }

    #[test]
    fn test_bm25_params_default() {
        let params = BM25Params::default();
        assert!((params.k1 - 1.5).abs() < 1e-6);
        assert!((params.b - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_hybrid_result_new() {
        let result = HybridResult::new(DocumentId::from("doc1"), 0.8, 0.6, 0.7, 0);
        assert_eq!(result.document_id.as_str(), "doc1");
        assert!((result.dense_score - 0.8).abs() < 1e-6);
        assert!((result.sparse_score - 0.6).abs() < 1e-6);
        assert!((result.combined_score - 0.7).abs() < 1e-6);
        assert_eq!(result.rank, 0);
    }

    #[test]
    fn test_fusion_strategy_default() {
        let strategy = FusionStrategy::default();
        match strategy {
            FusionStrategy::WeightedSum {
                dense_weight,
                sparse_weight,
            } => {
                assert!((dense_weight - 0.5).abs() < 1e-6);
                assert!((sparse_weight - 0.5).abs() < 1e-6);
            }
            _ => panic!("Expected WeightedSum as default"),
        }
    }

    #[test]
    fn test_fusion_strategy_rrf() {
        let strategy = FusionStrategy::rrf(60);
        match strategy {
            FusionStrategy::ReciprocalRankFusion { k } => {
                assert_eq!(k, 60);
            }
            _ => panic!("Expected ReciprocalRankFusion"),
        }
    }

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert!((config.dense_weight - 0.5).abs() < 1e-6);
        assert!((config.sparse_weight - 0.5).abs() < 1e-6);
        assert!(config.normalize_scores);
        assert!(config.min_score.is_none());
    }

    #[test]
    fn test_hybrid_config_weighted_sum() {
        let config = HybridConfig::weighted_sum(0.7, 0.3);
        assert!((config.dense_weight - 0.7).abs() < 1e-6);
        assert!((config.sparse_weight - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_hybrid_config_rrf() {
        let config = HybridConfig::rrf(60);
        match config.fusion_strategy {
            FusionStrategy::ReciprocalRankFusion { k } => {
                assert_eq!(k, 60);
            }
            _ => panic!("Expected RRF fusion strategy"),
        }
    }

    #[tokio::test]
    async fn test_in_memory_sparse_store_insert_and_get() {
        let mut store = InMemorySparseStore::new();
        let id = DocumentId::from("doc1");
        let vector = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 10);

        store.insert(id.clone(), vector.clone()).await.unwrap();

        let retrieved = store.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.indices, vector.indices);
        assert_eq!(retrieved.values, vector.values);
    }

    #[tokio::test]
    async fn test_in_memory_sparse_store_search() {
        let mut store = InMemorySparseStore::new();

        // Insert documents
        store
            .insert(
                DocumentId::from("doc1"),
                SparseVector::new(vec![0, 1], vec![1.0, 0.5], 10),
            )
            .await
            .unwrap();
        store
            .insert(
                DocumentId::from("doc2"),
                SparseVector::new(vec![0, 2], vec![0.5, 1.0], 10),
            )
            .await
            .unwrap();
        store
            .insert(
                DocumentId::from("doc3"),
                SparseVector::new(vec![3, 4], vec![1.0, 1.0], 10),
            )
            .await
            .unwrap();

        // Search with query that matches doc1 and doc2
        let query = SparseVector::new(vec![0, 1], vec![1.0, 1.0], 10);
        let results = store.search(&query, 10).await.unwrap();

        // doc1 should have highest score: 1.0*1.0 + 0.5*1.0 = 1.5
        // doc2 should have lower score: 0.5*1.0 = 0.5
        // doc3 should not match (no overlapping indices)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.as_str(), "doc1");
        assert!((results[0].1 - 1.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_in_memory_sparse_store_delete() {
        let mut store = InMemorySparseStore::new();
        let id = DocumentId::from("doc1");
        let vector = SparseVector::new(vec![0, 2], vec![1.0, 2.0], 10);

        store.insert(id.clone(), vector).await.unwrap();
        assert_eq!(store.count().await, 1);

        let deleted = store.delete(&id).await.unwrap();
        assert!(deleted);
        assert_eq!(store.count().await, 0);

        let deleted = store.delete(&id).await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_in_memory_sparse_store_clear() {
        let mut store = InMemorySparseStore::new();

        store
            .insert(
                DocumentId::from("doc1"),
                SparseVector::new(vec![0], vec![1.0], 10),
            )
            .await
            .unwrap();
        store
            .insert(
                DocumentId::from("doc2"),
                SparseVector::new(vec![1], vec![1.0], 10),
            )
            .await
            .unwrap();

        assert_eq!(store.count().await, 2);

        store.clear().await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_hybrid_searcher_weighted_sum() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["quick brown fox", "lazy dog", "brown dog"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store.insert(id, vector).await.unwrap();
        }

        let config = HybridConfig::weighted_sum(0.5, 0.5);
        let searcher = HybridSearcher::new(store, encoder, config);

        // Dense results (simulated)
        let dense_results = vec![
            (DocumentId::from("doc1"), 0.9),
            (DocumentId::from("doc2"), 0.5),
            (DocumentId::from("doc3"), 0.7),
        ];

        let results = searcher
            .search("brown fox", &dense_results, 3)
            .await
            .unwrap();

        assert!(!results.is_empty());
        // Results should be sorted by combined score
        for i in 1..results.len() {
            assert!(results[i - 1].combined_score >= results[i].combined_score);
        }
    }

    #[tokio::test]
    async fn test_hybrid_searcher_rrf() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["quick brown fox", "lazy dog", "brown dog"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store.insert(id, vector).await.unwrap();
        }

        let config = HybridConfig::rrf(60);
        let searcher = HybridSearcher::new(store, encoder, config);

        let dense_results = vec![
            (DocumentId::from("doc1"), 0.9),
            (DocumentId::from("doc3"), 0.7),
            (DocumentId::from("doc2"), 0.5),
        ];

        let results = searcher.search("brown", &dense_results, 3).await.unwrap();

        assert!(!results.is_empty());
        // Results should be sorted by RRF score
        for i in 1..results.len() {
            assert!(results[i - 1].combined_score >= results[i].combined_score);
        }
    }

    #[tokio::test]
    async fn test_hybrid_searcher_distribution_based() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["quick brown fox", "lazy dog", "brown dog"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store.insert(id, vector).await.unwrap();
        }

        let config = HybridConfig {
            fusion_strategy: FusionStrategy::distribution_based(0.5, 0.5),
            dense_weight: 0.5,
            sparse_weight: 0.5,
            normalize_scores: true,
            min_score: None,
        };
        let searcher = HybridSearcher::new(store, encoder, config);

        let dense_results = vec![
            (DocumentId::from("doc1"), 0.9),
            (DocumentId::from("doc2"), 0.5),
            (DocumentId::from("doc3"), 0.7),
        ];

        let results = searcher
            .search("brown fox", &dense_results, 3)
            .await
            .unwrap();

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_searcher_with_min_score() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["quick brown fox", "lazy dog"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store.insert(id, vector).await.unwrap();
        }

        let config = HybridConfig::weighted_sum(0.5, 0.5).with_min_score(0.8);
        let searcher = HybridSearcher::new(store, encoder, config);

        let dense_results = vec![
            (DocumentId::from("doc1"), 0.9),
            (DocumentId::from("doc2"), 0.3),
        ];

        let results = searcher
            .search("quick fox", &dense_results, 10)
            .await
            .unwrap();

        // All results should have combined_score >= 0.8
        for result in &results {
            assert!(result.combined_score >= 0.8);
        }
    }

    #[test]
    fn test_normalize_score() {
        // Test normalization
        let score = HybridSearcher::<InMemorySparseStore>::normalize_score(0.5, 0.0, 1.0);
        assert!((score - 0.5).abs() < 1e-6);

        let score = HybridSearcher::<InMemorySparseStore>::normalize_score(5.0, 0.0, 10.0);
        assert!((score - 0.5).abs() < 1e-6);

        // Test edge case: min == max
        let score = HybridSearcher::<InMemorySparseStore>::normalize_score(5.0, 5.0, 5.0);
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_score_stats() {
        let results = vec![
            (DocumentId::from("a"), 1.0),
            (DocumentId::from("b"), 2.0),
            (DocumentId::from("c"), 3.0),
        ];

        let (mean, std) = HybridSearcher::<InMemorySparseStore>::score_stats(&results);
        assert!((mean - 2.0).abs() < 1e-6);
        // std = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2) / 3) = sqrt(2/3)
        let expected_std = (2.0_f32 / 3.0).sqrt();
        assert!((std - expected_std).abs() < 1e-6);
    }

    #[test]
    fn test_z_normalize() {
        let normalized = HybridSearcher::<InMemorySparseStore>::z_normalize(5.0, 3.0, 2.0);
        // (5 - 3) / 2 = 1.0
        assert!((normalized - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bm25_encoder_with_custom_params() {
        let params = BM25Params {
            k1: 2.0,
            b: 0.5,
            delta: 1.0,
        };
        let encoder = BM25Encoder::with_params(params);
        assert!((encoder.params().k1 - 2.0).abs() < 1e-6);
        assert!((encoder.params().b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_default() {
        let v = SparseVector::default();
        assert!(v.is_empty());
        assert_eq!(v.dimension, 0);
    }

    #[tokio::test]
    async fn test_hybrid_searcher_empty_results() {
        let encoder = BM25Encoder::new();
        let store = InMemorySparseStore::new();
        let config = HybridConfig::default();
        let searcher = HybridSearcher::new(store, encoder, config);

        let results = searcher.search("query", &[], 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_searcher_search_with_sparse() {
        let mut encoder = BM25Encoder::new();
        let docs = vec!["hello world", "world peace"];
        encoder.fit(&docs);

        let mut store = InMemorySparseStore::new();
        for (i, doc) in docs.iter().enumerate() {
            let id = DocumentId::from(format!("doc{}", i + 1));
            let vector = encoder.encode(doc);
            store.insert(id, vector).await.unwrap();
        }

        let config = HybridConfig::default();
        let searcher = HybridSearcher::new(store, encoder, config);

        let sparse_query = searcher.encoder().encode("world");
        let dense_results = vec![
            (DocumentId::from("doc1"), 0.8),
            (DocumentId::from("doc2"), 0.6),
        ];

        let results = searcher
            .search_with_sparse(&sparse_query, &dense_results, 10)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }
}
