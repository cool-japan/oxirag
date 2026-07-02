//! Core types for the `llmlingua` module: configuration, compression target,
//! per-segment statistics, the compression result, and the error enum.

use thiserror::Error;

// ── CompressionTarget ──────────────────────────────────────────────────────────

/// How aggressively the compressor should shrink the prompt.
///
/// A [`PerplexityCompressor`](super::compressor::PerplexityCompressor) accepts
/// either a *relative* retention ratio or an *absolute* token budget; both are
/// ultimately converted to a target token count against the tokenised input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionTarget {
    /// Fraction of the original token count to **retain**, in `(0.0, 1.0]`.
    ///
    /// `0.6` keeps roughly 60% of the tokens; `1.0` keeps everything.
    Ratio(f32),
    /// Absolute number of tokens to retain. Values at or above the input's
    /// token count are treated as "keep everything".
    TokenBudget(usize),
}

impl Default for CompressionTarget {
    /// Defaults to [`CompressionTarget::Ratio`]`(0.5)` — retain half the tokens.
    fn default() -> Self {
        Self::Ratio(0.5)
    }
}

// ── LlmLinguaConfig ────────────────────────────────────────────────────────────

/// Configuration for [`PerplexityCompressor`](super::compressor::PerplexityCompressor).
///
/// Every knob controls one part of the coarse-to-fine, perplexity-driven
/// pipeline described at the [module level](super). Construct with
/// [`LlmLinguaConfig::new`] and adjust via the chainable `with_*` setters.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmLinguaConfig {
    /// Order of the surrogate n-gram language model (`1` = unigram, `2` =
    /// bigram, `3` = trigram). Higher orders back off recursively to lower
    /// ones. Must be in `1..=8`. Default: `3`.
    pub ngram_order: usize,
    /// Add-k (Laplace-style) smoothing constant applied at the **unigram**
    /// base of the interpolation, guaranteeing every vocabulary token — and
    /// every out-of-vocabulary token — receives non-zero probability. Must be
    /// finite and `> 0`. Default: `1.0` (classic Laplace).
    pub add_k: f64,
    /// Jelinek-Mercer interpolation weight in `(0.0, 1.0]` given to the
    /// **higher-order** maximum-likelihood estimate at each level; the
    /// remaining `1 - lambda` mass is delegated to the recursively-smoothed
    /// lower order. Larger values trust longer contexts more. Default: `0.7`.
    pub interpolation_lambda: f64,
    /// The desired output size (relative ratio or absolute budget).
    /// Default: [`CompressionTarget::Ratio`]`(0.5)`.
    pub target: CompressionTarget,
    /// Minimum number of segments (sentences) that coarse Stage A is never
    /// allowed to drop below, however tight the budget. Must be `>= 1`.
    /// Default: `1`.
    pub min_segments_retained: usize,
    /// Upper bound, in `[0.0, 1.0]`, on the fraction of a *single* surviving
    /// segment's tokens that fine Stage B may delete, protecting readability.
    /// Default: `0.8`.
    pub max_local_drop_ratio: f32,
    /// Exponent `gamma` applied to per-segment information density when the
    /// [`BudgetController`](super::budget::BudgetController) allocates the
    /// budget: a segment's keep weight is `len * density.powf(gamma)`, so
    /// `gamma = 0` yields a uniform per-segment rate and larger values let
    /// denser segments keep proportionally more of their own tokens. Must be
    /// finite and `>= 0`. Default: `1.0`.
    pub density_emphasis: f32,
    /// When `true`, tokens containing a digit (dates, quantities, identifiers)
    /// are protected from fine pruning. Default: `true`.
    pub protect_numbers: bool,
    /// When `true`, tokens whose surface form starts with an uppercase letter
    /// (proper-noun-looking) are protected from fine pruning. Default: `true`.
    pub protect_capitalized: bool,
    /// When `true`, the first and last token of every surviving segment are
    /// protected from fine pruning, preserving sentence structure.
    /// Default: `true`.
    pub protect_boundaries: bool,
}

impl Default for LlmLinguaConfig {
    fn default() -> Self {
        Self {
            ngram_order: 3,
            add_k: 1.0,
            interpolation_lambda: 0.7,
            target: CompressionTarget::default(),
            min_segments_retained: 1,
            max_local_drop_ratio: 0.8,
            density_emphasis: 1.0,
            protect_numbers: true,
            protect_capitalized: true,
            protect_boundaries: true,
        }
    }
}

impl LlmLinguaConfig {
    /// Creates a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the surrogate n-gram model order.
    #[must_use]
    pub fn with_ngram_order(mut self, ngram_order: usize) -> Self {
        self.ngram_order = ngram_order;
        self
    }

    /// Sets the add-k smoothing constant.
    #[must_use]
    pub fn with_add_k(mut self, add_k: f64) -> Self {
        self.add_k = add_k;
        self
    }

    /// Sets the Jelinek-Mercer interpolation weight.
    #[must_use]
    pub fn with_interpolation_lambda(mut self, interpolation_lambda: f64) -> Self {
        self.interpolation_lambda = interpolation_lambda;
        self
    }

    /// Sets the compression target directly.
    #[must_use]
    pub fn with_target(mut self, target: CompressionTarget) -> Self {
        self.target = target;
        self
    }

    /// Convenience setter for a [`CompressionTarget::Ratio`] target.
    #[must_use]
    pub fn with_ratio(mut self, ratio: f32) -> Self {
        self.target = CompressionTarget::Ratio(ratio);
        self
    }

    /// Convenience setter for a [`CompressionTarget::TokenBudget`] target.
    #[must_use]
    pub fn with_token_budget(mut self, budget: usize) -> Self {
        self.target = CompressionTarget::TokenBudget(budget);
        self
    }

    /// Sets the minimum number of segments Stage A must retain.
    #[must_use]
    pub fn with_min_segments_retained(mut self, min_segments_retained: usize) -> Self {
        self.min_segments_retained = min_segments_retained;
        self
    }

    /// Sets the per-segment maximum local drop ratio.
    #[must_use]
    pub fn with_max_local_drop_ratio(mut self, max_local_drop_ratio: f32) -> Self {
        self.max_local_drop_ratio = max_local_drop_ratio;
        self
    }

    /// Sets the density-emphasis exponent used by the budget controller.
    #[must_use]
    pub fn with_density_emphasis(mut self, density_emphasis: f32) -> Self {
        self.density_emphasis = density_emphasis;
        self
    }

    /// Sets whether numeric tokens are protected from fine pruning.
    #[must_use]
    pub fn with_protect_numbers(mut self, protect_numbers: bool) -> Self {
        self.protect_numbers = protect_numbers;
        self
    }

    /// Sets whether capitalized (proper-noun-looking) tokens are protected.
    #[must_use]
    pub fn with_protect_capitalized(mut self, protect_capitalized: bool) -> Self {
        self.protect_capitalized = protect_capitalized;
        self
    }

    /// Sets whether segment-boundary tokens are protected.
    #[must_use]
    pub fn with_protect_boundaries(mut self, protect_boundaries: bool) -> Self {
        self.protect_boundaries = protect_boundaries;
        self
    }

    /// Validates every numeric invariant of the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LlmLinguaError::InvalidConfig`] when:
    /// - `ngram_order` is `0` or greater than `8`;
    /// - `add_k` is non-finite or `<= 0`;
    /// - `interpolation_lambda` is non-finite or outside `(0.0, 1.0]`;
    /// - the [`CompressionTarget::Ratio`] value is non-finite or outside
    ///   `(0.0, 1.0]`, or the [`CompressionTarget::TokenBudget`] value is `0`;
    /// - `min_segments_retained` is `0`;
    /// - `max_local_drop_ratio` is non-finite or outside `[0.0, 1.0]`;
    /// - `density_emphasis` is non-finite or negative.
    pub fn validate(&self) -> Result<(), LlmLinguaError> {
        if self.ngram_order == 0 || self.ngram_order > 8 {
            return Err(LlmLinguaError::InvalidConfig(format!(
                "ngram_order must be in 1..=8, got {}",
                self.ngram_order
            )));
        }
        if !self.add_k.is_finite() || self.add_k <= 0.0 {
            return Err(LlmLinguaError::InvalidConfig(format!(
                "add_k must be finite and > 0, got {}",
                self.add_k
            )));
        }
        if !self.interpolation_lambda.is_finite()
            || self.interpolation_lambda <= 0.0
            || self.interpolation_lambda > 1.0
        {
            return Err(LlmLinguaError::InvalidConfig(format!(
                "interpolation_lambda must be in (0.0, 1.0], got {}",
                self.interpolation_lambda
            )));
        }
        match self.target {
            CompressionTarget::Ratio(ratio) => {
                if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 {
                    return Err(LlmLinguaError::InvalidConfig(format!(
                        "target ratio must be in (0.0, 1.0], got {ratio}"
                    )));
                }
            }
            CompressionTarget::TokenBudget(budget) => {
                if budget == 0 {
                    return Err(LlmLinguaError::InvalidConfig(
                        "target token budget must be at least 1".to_string(),
                    ));
                }
            }
        }
        if self.min_segments_retained == 0 {
            return Err(LlmLinguaError::InvalidConfig(
                "min_segments_retained must be at least 1".to_string(),
            ));
        }
        if !self.max_local_drop_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.max_local_drop_ratio)
        {
            return Err(LlmLinguaError::InvalidConfig(format!(
                "max_local_drop_ratio must be in [0.0, 1.0], got {}",
                self.max_local_drop_ratio
            )));
        }
        if !self.density_emphasis.is_finite() || self.density_emphasis < 0.0 {
            return Err(LlmLinguaError::InvalidConfig(format!(
                "density_emphasis must be finite and >= 0, got {}",
                self.density_emphasis
            )));
        }
        Ok(())
    }
}

// ── SegmentStats ───────────────────────────────────────────────────────────────

/// Per-segment introspection record produced alongside a compression.
///
/// One [`SegmentStats`] is emitted for **every** input segment, in input
/// order, whether or not it survived coarse Stage A, so callers can audit
/// exactly what the compressor did.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentStats {
    /// Zero-based index of the segment in the original input order.
    pub index: usize,
    /// Number of tokens the segment contained before compression.
    pub original_tokens: usize,
    /// Number of tokens retained after both stages (`0` for a segment dropped
    /// entirely by Stage A).
    pub retained_tokens: usize,
    /// Average token surprisal (bits) — the segment's information **density**;
    /// the signal Stage A ranks by and the budget controller allocates against.
    pub density: f32,
    /// Whether the segment survived coarse Stage A (and thus contributes text).
    pub retained: bool,
    /// The compressed segment text (empty when `retained` is `false`).
    pub text: String,
}

impl SegmentStats {
    /// Number of tokens deleted from this segment (`original - retained`).
    #[must_use]
    pub fn dropped_tokens(&self) -> usize {
        self.original_tokens.saturating_sub(self.retained_tokens)
    }
}

// ── CompressionResult ──────────────────────────────────────────────────────────

/// The full outcome of a [`PerplexityCompressor`](super::compressor::PerplexityCompressor)
/// run.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionResult {
    /// The compressed prompt text.
    pub compressed_text: String,
    /// Token count of the original input.
    pub original_token_count: usize,
    /// Token count of [`compressed_text`](Self::compressed_text).
    pub compressed_token_count: usize,
    /// The retention ratio that was *requested* (for a
    /// [`CompressionTarget::TokenBudget`] this is `budget / original`, clamped
    /// to `1.0`).
    pub requested_ratio: f32,
    /// The retention ratio that was actually *achieved*
    /// (`compressed / original`); may differ from
    /// [`requested_ratio`](Self::requested_ratio) because of the
    /// minimum-segments floor, per-segment drop cap, and protected tokens.
    pub achieved_ratio: f32,
    /// Per-segment statistics, in original input order.
    pub segment_stats: Vec<SegmentStats>,
}

impl CompressionResult {
    /// Fraction of tokens removed (`1 - achieved_ratio`), clamped to `[0, 1]`.
    #[must_use]
    pub fn compression_rate(&self) -> f32 {
        (1.0 - self.achieved_ratio).clamp(0.0, 1.0)
    }

    /// Absolute number of tokens removed by compression.
    #[must_use]
    pub fn tokens_saved(&self) -> usize {
        self.original_token_count
            .saturating_sub(self.compressed_token_count)
    }

    /// Number of segments dropped entirely by coarse Stage A.
    #[must_use]
    pub fn dropped_segment_count(&self) -> usize {
        self.segment_stats.iter().filter(|s| !s.retained).count()
    }
}

// ── LlmLinguaError ─────────────────────────────────────────────────────────────

/// Errors produced by the `llmlingua` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LlmLinguaError {
    /// The input contained no tokenizable content (empty or whitespace only).
    #[error("input is empty or contains no tokenizable content")]
    EmptyInput,
    /// The supplied [`LlmLinguaConfig`] failed [`LlmLinguaConfig::validate`].
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}
