//! Configuration and error types for multi-list rank/score fusion.
//!
//! This module defines the public configuration surface for [`RankFusion`]:
//! the [`FusionMethod`] selector, the per-list [`ScoreNormalization`] options,
//! the [`RankFusionConfig`] builder, and the [`RankFusionError`] error type.
//!
//! [`RankFusion`]: crate::rank_fusion::RankFusion

/// The fusion method used to combine multiple ranked result lists into one.
///
/// All methods deduplicate documents by their [`DocumentId`] string and produce
/// a single fused score per unique document. Score-based methods
/// ([`Self::CombSum`], [`Self::CombMnz`], [`Self::CombAnz`], [`Self::WeightedSum`])
/// consume the (optionally normalized) per-list scores, while rank-based methods
/// ([`Self::Borda`], [`Self::Isr`], [`Self::Rrf`]) ignore raw scores and derive
/// their contribution purely from the position of a document within each list.
///
/// [`DocumentId`]: crate::types::DocumentId
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FusionMethod {
    /// `CombSUM`: the fused score is the sum of the (normalized) scores of a
    /// document across every list in which it appears.
    #[default]
    CombSum,
    /// `CombMNZ`: [`Self::CombSum`] multiplied by the number of lists in which
    /// the document appears. Rewards documents that multiple retrievers agree on.
    CombMnz,
    /// `CombANZ`: [`Self::CombSum`] divided by the number of lists in which the
    /// document appears (the mean of the per-list scores).
    CombAnz,
    /// Borda count: a document at 0-based rank `r` in a list of length `L`
    /// contributes `L - r` points; contributions are summed across lists.
    Borda,
    /// Inverse Square Rank: a document at 0-based rank `r` contributes
    /// `1 / (r + 1)^2`; contributions are summed across lists.
    Isr,
    /// Weighted sum: a per-list weighted sum of the (normalized) scores. Requires
    /// [`RankFusionConfig::weights`] with one weight per input list.
    WeightedSum,
    /// Reciprocal Rank Fusion: a document at 0-based rank `r` contributes
    /// `1 / (rrf_k + r + 1)`; contributions are summed across lists.
    Rrf,
}

/// Per-list score normalization applied before score-based fusion methods.
///
/// Normalization is performed independently on each input list so that lists
/// produced by retrievers with different score scales can be combined fairly.
/// It only affects score-based methods; rank-based methods
/// ([`FusionMethod::Borda`], [`FusionMethod::Isr`], [`FusionMethod::Rrf`])
/// derive their contribution from rank and are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScoreNormalization {
    /// Use the raw scores unchanged.
    None,
    /// Min-max scaling to the closed interval `[0, 1]`. If a list has a single
    /// distinct score (zero range), every score maps to `1.0`.
    #[default]
    MinMax,
    /// Z-score standardization: subtract the mean and divide by the population
    /// standard deviation, yielding a mean of approximately `0`. If the standard
    /// deviation is zero, every score maps to `0.0`.
    ZScore,
    /// Scale each list so its scores sum to `1.0`. If the sum of scores is zero,
    /// the list is left unchanged.
    SumTo1,
}

/// Configuration for [`RankFusion`].
///
/// Construct with [`RankFusionConfig::default`] or [`RankFusionConfig::new`] and
/// refine using the `#[must_use]` builder methods.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rank-fusion")]
/// # {
/// use oxirag::rank_fusion::{RankFusionConfig, FusionMethod, ScoreNormalization};
///
/// let config = RankFusionConfig::new()
///     .with_method(FusionMethod::WeightedSum)
///     .with_normalization(ScoreNormalization::MinMax)
///     .with_weights(vec![0.7, 0.3])
///     .with_top_n(5);
/// assert_eq!(config.method, FusionMethod::WeightedSum);
/// assert_eq!(config.top_n, 5);
/// # }
/// ```
///
/// [`RankFusion`]: crate::rank_fusion::RankFusion
#[derive(Debug, Clone)]
pub struct RankFusionConfig {
    /// The fusion method used to combine lists. Defaults to
    /// [`FusionMethod::CombSum`].
    pub method: FusionMethod,
    /// Per-list score normalization applied before score-based methods.
    /// Defaults to [`ScoreNormalization::MinMax`].
    pub normalization: ScoreNormalization,
    /// Per-list weights for [`FusionMethod::WeightedSum`]. Must contain exactly
    /// one weight per input list when that method is selected. Defaults to empty.
    pub weights: Vec<f32>,
    /// Smoothing constant `k` for [`FusionMethod::Rrf`]. Defaults to `60.0`.
    pub rrf_k: f32,
    /// Maximum number of fused results to return. `0` means return all results.
    /// Defaults to `0`.
    pub top_n: usize,
}

impl Default for RankFusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::CombSum,
            normalization: ScoreNormalization::MinMax,
            weights: Vec::new(),
            rrf_k: 60.0,
            top_n: 0,
        }
    }
}

impl RankFusionConfig {
    /// Create a new configuration with default values.
    ///
    /// Equivalent to [`RankFusionConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the fusion method.
    #[must_use]
    pub fn with_method(mut self, method: FusionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the per-list score normalization.
    #[must_use]
    pub fn with_normalization(mut self, normalization: ScoreNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Set the per-list weights used by [`FusionMethod::WeightedSum`].
    #[must_use]
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        self.weights = weights;
        self
    }

    /// Set the RRF smoothing constant `k`.
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

/// Errors that can occur during rank/score fusion.
#[derive(Debug, thiserror::Error)]
pub enum RankFusionError {
    /// No input lists were provided to fuse.
    #[error("no input lists")]
    NoLists,
    /// The number of weights does not match the number of input lists.
    ///
    /// Only [`FusionMethod::WeightedSum`] requires weights, and it requires
    /// exactly one weight per list.
    #[error("weights length {weights} != list count {lists}")]
    WeightMismatch {
        /// The number of weights supplied in the configuration.
        weights: usize,
        /// The number of input lists provided to fuse.
        lists: usize,
    },
}
