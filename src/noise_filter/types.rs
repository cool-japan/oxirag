//! Types for the `noise_filter` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── NoiseConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the noise / distractor filter.
///
/// Each retrieved passage is scored along two axes — its **relevance** to the
/// query and its **consensus** alignment with the centroid of the retrieved set
/// — and the two are blended into a single `combined` score. Passages whose
/// combined score falls below [`NoiseConfig::relevance_threshold`] are flagged
/// as noise (distractors or topic-drift outliers).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Minimum combined score for a passage to be kept.
    ///
    /// Passages scoring strictly below this value are flagged as noise.
    /// Defaults to `0.15`.
    pub relevance_threshold: f32,
    /// Weight of the consensus term in the blended `combined` score.
    ///
    /// The combined score is
    /// `(1 - consensus_weight) * relevance + consensus_weight * consensus`.
    /// Defaults to `0.3`.
    pub consensus_weight: f32,
    /// Dimensionality of the deterministic FNV-1a pseudo-embeddings.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Minimum number of passages to keep even when every passage is flagged.
    ///
    /// The highest-`combined` passages are retained to satisfy this floor.
    /// Defaults to `1`.
    pub keep_min: usize,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            relevance_threshold: 0.15,
            consensus_weight: 0.3,
            dim: 128,
            keep_min: 1,
        }
    }
}

impl NoiseConfig {
    /// Create a new configuration with the default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the relevance (combined-score) threshold below which a passage is noise.
    #[must_use]
    pub fn with_relevance_threshold(mut self, v: f32) -> Self {
        self.relevance_threshold = v;
        self
    }

    /// Set the weight of the consensus term in the blended score.
    #[must_use]
    pub fn with_consensus_weight(mut self, v: f32) -> Self {
        self.consensus_weight = v;
        self
    }

    /// Set the embedding dimensionality.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }

    /// Set the minimum number of passages to keep.
    #[must_use]
    pub fn with_keep_min(mut self, v: usize) -> Self {
        self.keep_min = v;
        self
    }
}

// ── PassageAssessment ───────────────────────────────────────────────────────────

/// Per-passage noise assessment produced by [`crate::noise_filter::NoiseFilter::assess`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PassageAssessment {
    /// Index of the passage within the input slice.
    pub index: usize,
    /// Relevance to the query in `[0, 1]` (query similarity blended with lexical overlap).
    pub relevance: f32,
    /// Consensus alignment in `[0, 1]` (similarity to the retrieved-set centroid).
    pub consensus: f32,
    /// Blended score: `(1 - consensus_weight) * relevance + consensus_weight * consensus`.
    pub combined: f32,
    /// `true` when `combined` falls below the configured relevance threshold.
    pub is_noise: bool,
}

// ── NoiseReport ─────────────────────────────────────────────────────────────────

/// Summary of a filtering pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseReport {
    /// Number of passages retained.
    pub kept: usize,
    /// Number of passages removed as noise.
    pub removed: usize,
    /// The per-passage assessments (in input order).
    pub assessments: Vec<PassageAssessment>,
}

// ── NoiseFilterError ────────────────────────────────────────────────────────────

/// Errors from the `noise_filter` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NoiseFilterError {
    /// The query string was empty (or whitespace only).
    #[error("query must not be empty")]
    EmptyQuery,
}
