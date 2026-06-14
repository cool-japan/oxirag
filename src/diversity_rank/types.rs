//! Shared types for the determinantal-point-process-inspired diversity ranker.

use serde::{Deserialize, Serialize};

/// Similarity function used to measure redundancy between two embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SimilarityKind {
    /// Cosine similarity over L2-normalised vectors.
    #[default]
    Cosine,
}

/// Configuration for [`DiversityRanker`](crate::diversity_rank::DiversityRanker).
///
/// The ranker performs greedy MAP inference for a determinantal point process
/// (DPP). It picks items one at a time, each time maximising a *volume-style*
/// marginal gain that rewards both high quality and dissimilarity to the
/// already-selected set (see the module-level documentation for the exact
/// formula).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Maximum number of items to select.
    ///
    /// Defaults to `5`.
    pub k: usize,

    /// Exponent-free multiplicative weight applied to each quality value before
    /// it is squared into the gain.
    ///
    /// Larger values bias the selection toward high-quality items at the
    /// expense of diversity. Defaults to `1.0`.
    pub quality_weight: f32,

    /// Dimensionality of the deterministic FNV-1a pseudo-embeddings.
    ///
    /// Defaults to `128`.
    pub dim: usize,

    /// Similarity function used to compute pairwise redundancy.
    ///
    /// Defaults to [`SimilarityKind::Cosine`].
    pub similarity: SimilarityKind,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            k: 5,
            quality_weight: 1.0,
            dim: 128,
            similarity: SimilarityKind::Cosine,
        }
    }
}

impl DiversityConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of items to select.
    #[must_use]
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the quality weight applied before squaring into the gain.
    #[must_use]
    pub fn with_quality_weight(mut self, quality_weight: f32) -> Self {
        self.quality_weight = quality_weight;
        self
    }

    /// Set the pseudo-embedding dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the similarity function.
    #[must_use]
    pub fn with_similarity(mut self, similarity: SimilarityKind) -> Self {
        self.similarity = similarity;
        self
    }
}

/// Outcome of a greedy DPP selection over a candidate set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiversitySelection {
    /// Indices of the selected candidates, in selection order.
    pub selected: Vec<usize>,

    /// Mean pairwise dissimilarity (`1 - similarity`) of the selected set.
    ///
    /// Ranges over `[0.0, 1.0]` for cosine similarity of non-negative
    /// embeddings; higher means a more diverse selection. A selection of fewer
    /// than two items has a diversity score of `0.0` (no pairs).
    pub diversity_score: f32,
}

impl DiversitySelection {
    /// Create a new selection result.
    #[must_use]
    pub fn new(selected: Vec<usize>, diversity_score: f32) -> Self {
        Self {
            selected,
            diversity_score,
        }
    }
}

/// Errors produced by the diversity ranker.
#[derive(Debug, thiserror::Error)]
pub enum DiversityRankError {
    /// The candidate set was empty.
    #[error("candidates must not be empty")]
    EmptyCandidates,

    /// The `qualities` and `embeddings` slices had different lengths.
    #[error("qualities/embeddings length mismatch")]
    LengthMismatch,
}
