//! Score fusion algorithms for hybrid dense+sparse retrieval.
//!
//! This module contains `HybridSearcher`, which accepts pre-computed dense search
//! results and combines them with BM25 sparse results using one of three strategies:
//!
//! - **Weighted Sum** – min-max normalise then linearly combine
//! - **Reciprocal Rank Fusion (RRF)** – rank-based, no normalisation required
//! - **Distribution-Based** – z-score normalise then linearly combine

#![allow(clippy::cast_precision_loss)] // Intentional: usize to f32 for scoring

use std::collections::HashMap;

use crate::error::VectorStoreError;
use crate::types::DocumentId;

use super::bm25::BM25Encoder;
use super::types::{FusionStrategy, HybridConfig, HybridResult, SparseVector, SparseVectorStore};

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
    pub(crate) fn score_range(results: &[(DocumentId, f32)]) -> (f32, f32) {
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
    pub(crate) fn normalize_score(score: f32, min: f32, max: f32) -> f32 {
        if (max - min).abs() < f32::EPSILON {
            return 0.5;
        }
        (score - min) / (max - min)
    }

    /// Compute mean and standard deviation of scores.
    pub(crate) fn score_stats(results: &[(DocumentId, f32)]) -> (f32, f32) {
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
    pub(crate) fn z_normalize(score: f32, mean: f32, std: f32) -> f32 {
        (score - mean) / std
    }
}
