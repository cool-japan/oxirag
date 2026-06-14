//! Shared types for the `retrieval_diversity` module.

use thiserror::Error;

// ── RetrievalDiversityConfig ──────────────────────────────────────────────────

/// Configuration for [`DiversityScorer`](super::DiversityScorer).
///
/// Diversity and coverage metrics are computed over a ranked result list. The
/// `alpha` parameter controls the novelty discount of α-nDCG, and `dim` sets the
/// dimensionality of the deterministic FNV-1a pseudo-embeddings used by
/// Intra-List Diversity.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalDiversityConfig {
    /// Novelty-discount parameter for α-nDCG, in `[0.0, 1.0]`.
    ///
    /// Larger values reward redundancy more (less novelty discounting); `alpha`
    /// equal to `1.0` removes the novelty discount entirely, so each subtopic
    /// contributes full gain regardless of how often it was seen above. Defaults
    /// to `0.5`.
    pub alpha: f32,

    /// Dimensionality of the deterministic FNV-1a pseudo-embeddings.
    ///
    /// Used only by Intra-List Diversity. Defaults to `128`.
    pub dim: usize,
}

impl Default for RetrievalDiversityConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            dim: 128,
        }
    }
}

impl RetrievalDiversityConfig {
    /// Create a new configuration with default values (`alpha = 0.5`, `dim = 128`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the α-nDCG novelty-discount parameter.
    ///
    /// The value is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set the pseudo-embedding dimensionality used by Intra-List Diversity.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }
}

// ── DiversityMetrics ──────────────────────────────────────────────────────────

/// The three diversity and coverage metrics computed over a ranked result list.
///
/// All three fields lie in `[0.0, 1.0]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiversityMetrics {
    /// Intra-List Diversity: mean pairwise dissimilarity (`1 - cosine`) of the
    /// top-`k` results' embeddings.
    pub ild: f32,

    /// Subtopic recall: fraction of all subtopics covered by the top-`k`
    /// results.
    pub s_recall: f32,

    /// α-nDCG: novelty-discounted, normalised discounted cumulative gain.
    pub alpha_ndcg: f32,
}

impl DiversityMetrics {
    /// Create a new metrics bundle from its three components.
    #[must_use]
    pub fn new(ild: f32, s_recall: f32, alpha_ndcg: f32) -> Self {
        Self {
            ild,
            s_recall,
            alpha_ndcg,
        }
    }
}

// ── RetrievalDiversityError ───────────────────────────────────────────────────

/// Errors produced by the `retrieval_diversity` module.
#[derive(Debug, Error)]
pub enum RetrievalDiversityError {
    /// The result list was empty.
    #[error("results must not be empty")]
    EmptyResults,

    /// The `texts` and `result_subtopics` slices had different lengths.
    #[error("texts/subtopics length mismatch")]
    LengthMismatch,
}
