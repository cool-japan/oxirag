//! Core data types for hybrid search.
//!
//! This module defines the fundamental types used throughout the hybrid search
//! system: sparse vectors, BM25 parameters, search results, fusion strategies,
//! and configuration.

#![allow(clippy::cast_precision_loss)] // Intentional: usize to f32 for scoring

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

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
