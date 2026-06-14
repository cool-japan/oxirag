//! Configuration and error types for the ensemble retriever.
//!
//! This module defines the public configuration surface for the ensemble
//! retriever: the [`EnsembleFusion`] strategy selector, the [`EnsembleConfig`]
//! builder, and the [`EnsembleError`] error type. The orchestration logic and
//! the [`SubRetriever`] trait live in the sibling `retriever` module and are
//! re-exported from the crate's `ensemble_retriever` module.
//!
//! [`SubRetriever`]: crate::ensemble_retriever::SubRetriever

/// The strategy used to fuse the per-retriever candidate lists into one ranking.
///
/// Both strategies are *weighted*: every registered sub-retriever carries a
/// non-negative weight, and its contribution to a document's fused score is
/// scaled by that weight. Documents surfaced by several retrievers accumulate
/// contributions from each, so cross-retriever agreement is rewarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EnsembleFusion {
    /// Weighted score fusion: each retriever's raw scores are min-max normalized
    /// to the closed interval `[0, 1]` independently, then a document's fused
    /// score is the sum over retrievers of `weight * normalized_score`.
    ///
    /// Normalizing per retriever lets strategies with different score scales
    /// (for example a token-overlap lexical scorer and a cosine vector scorer)
    /// combine on equal footing.
    WeightedScore,
    /// Weighted Reciprocal Rank Fusion: a document at 0-based rank `r` within a
    /// retriever's list contributes `weight * 1 / (rrf_k + r + 1)`; contributions
    /// are summed across retrievers.
    ///
    /// Because the contribution depends only on rank, the raw score scale is
    /// irrelevant, which makes this the robust default.
    #[default]
    WeightedRrf,
}

/// Configuration for the ensemble retriever.
///
/// Construct with [`EnsembleConfig::default`] or [`EnsembleConfig::new`] and
/// refine using the `#[must_use]` builder methods.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "ensemble-retriever")]
/// # {
/// use oxirag::ensemble_retriever::{EnsembleConfig, EnsembleFusion};
///
/// let config = EnsembleConfig::new()
///     .with_fusion(EnsembleFusion::WeightedScore)
///     .with_rrf_k(40.0)
///     .with_top_n(5);
/// assert_eq!(config.fusion, EnsembleFusion::WeightedScore);
/// assert_eq!(config.top_n, 5);
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleConfig {
    /// The fusion strategy used to combine per-retriever lists. Defaults to
    /// [`EnsembleFusion::WeightedRrf`].
    pub fusion: EnsembleFusion,
    /// Smoothing constant `k` for [`EnsembleFusion::WeightedRrf`]. Larger values
    /// flatten the rank weighting. Defaults to `60.0`.
    pub rrf_k: f32,
    /// Maximum number of fused results to return. `0` means return all results.
    /// Defaults to `0`.
    pub top_n: usize,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            fusion: EnsembleFusion::WeightedRrf,
            rrf_k: 60.0,
            top_n: 0,
        }
    }
}

impl EnsembleConfig {
    /// Create a new configuration with default values.
    ///
    /// Equivalent to [`EnsembleConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the fusion strategy.
    #[must_use]
    pub fn with_fusion(mut self, fusion: EnsembleFusion) -> Self {
        self.fusion = fusion;
        self
    }

    /// Set the RRF smoothing constant `k` used by [`EnsembleFusion::WeightedRrf`].
    #[must_use]
    pub fn with_rrf_k(mut self, rrf_k: f32) -> Self {
        self.rrf_k = rrf_k;
        self
    }

    /// Set the maximum number of fused results to return (`0` = all).
    #[must_use]
    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }
}

/// Errors that can occur while running the ensemble retriever.
#[derive(Debug, thiserror::Error)]
pub enum EnsembleError {
    /// No sub-retrievers have been registered, so there is nothing to run.
    #[error("no sub-retrievers registered")]
    NoRetrievers,
    /// The query string was empty or whitespace-only.
    #[error("query must not be empty")]
    EmptyQuery,
}
