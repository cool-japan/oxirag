//! Core types for the `recomp` module.
//!
//! These types describe the configuration, decisions, and errors produced by
//! the RECOMP pipeline: a [`CompressorStrategy`] choice between an
//! [`ExtractiveSummaryCompressor`](super::extractive::ExtractiveSummaryCompressor)
//! and an
//! [`AbstractiveSummaryCompressor`](super::abstractive::AbstractiveSummaryCompressor),
//! gated by a selective-augmentation decision ([`RecompDecision`]).

use thiserror::Error;

// ── CompressorStrategy ────────────────────────────────────────────────────────

/// The compressor implementation used by [`RecompPipeline`](super::pipeline::RecompPipeline).
///
/// RECOMP's central contribution is offering *two* compressor families rather
/// than a single fixed one:
///
/// - [`CompressorStrategy::Extractive`] selects and repacks original sentences
///   verbatim, preserving their relative order.
/// - [`CompressorStrategy::AbstractiveLite`] selects the same top-scoring
///   sentences but stitches them into a single template-fused paragraph. It is
///   a deterministic, pure-Rust heuristic — **not** neural abstractive
///   summarization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressorStrategy {
    /// Sentence-extractive compression: verbatim sentences, original order
    /// preserved, packed within the token budget. This is the default.
    #[default]
    Extractive,
    /// Template-based extractive-fusion "abstractive-lite" compression: the
    /// same relevance-scored sentences, lightly stitched together with
    /// connective phrases into a single synthetic paragraph.
    AbstractiveLite,
}

impl CompressorStrategy {
    /// Returns a stable lower-case identifier for the strategy.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Extractive => "extractive",
            Self::AbstractiveLite => "abstractive_lite",
        }
    }
}

impl std::fmt::Display for CompressorStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── RecompConfig ───────────────────────────────────────────────────────────────

/// Configuration for the [`RecompPipeline`](super::pipeline::RecompPipeline).
///
/// | Field | Default | Purpose |
/// |-------|---------|---------|
/// | `token_budget` | `150` | Approximate word budget for the compressed output |
/// | `augmentation_threshold` | `0.15` | Minimum mean lexical relevance for the selective-augmentation gate to proceed |
/// | `strategy` | [`CompressorStrategy::Extractive`] | Which compressor implements the compression step |
/// | `dedup_similarity_threshold` | `0.85` | Jaccard threshold above which two sentences are treated as near-duplicates |
#[derive(Debug, Clone, PartialEq)]
pub struct RecompConfig {
    /// Approximate word budget for the compressed output.
    ///
    /// Must be greater than `0`. Defaults to `150`.
    pub token_budget: usize,
    /// Minimum mean per-passage lexical relevance (in `[0.0, 1.0]`) required
    /// before the pipeline will bother compressing at all.
    ///
    /// When the computed relevance score falls below this threshold, RECOMP's
    /// selective-augmentation gate fires and
    /// [`RecompPipeline::compress`](super::pipeline::RecompPipeline::compress)
    /// returns [`RecompDecision::Skip`] instead of a compressed summary.
    /// Defaults to `0.15`.
    pub augmentation_threshold: f32,
    /// Which compressor family performs the compression step once the gate
    /// has passed. Defaults to [`CompressorStrategy::Extractive`].
    pub strategy: CompressorStrategy,
    /// Jaccard similarity threshold (in `[0.0, 1.0]`) above which two
    /// sentences are considered near-identical and collapsed to one.
    /// Defaults to `0.85`.
    pub dedup_similarity_threshold: f32,
}

impl Default for RecompConfig {
    fn default() -> Self {
        Self {
            token_budget: 150,
            augmentation_threshold: 0.15,
            strategy: CompressorStrategy::Extractive,
            dedup_similarity_threshold: 0.85,
        }
    }
}

impl RecompConfig {
    /// Creates a new [`RecompConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the token budget.
    #[must_use]
    pub fn with_token_budget(mut self, token_budget: usize) -> Self {
        self.token_budget = token_budget;
        self
    }

    /// Sets the selective-augmentation threshold.
    #[must_use]
    pub fn with_augmentation_threshold(mut self, augmentation_threshold: f32) -> Self {
        self.augmentation_threshold = augmentation_threshold;
        self
    }

    /// Sets the compressor strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: CompressorStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the near-duplicate-sentence Jaccard threshold.
    #[must_use]
    pub fn with_dedup_similarity_threshold(mut self, dedup_similarity_threshold: f32) -> Self {
        self.dedup_similarity_threshold = dedup_similarity_threshold;
        self
    }

    /// Validates the configuration's numeric bounds.
    ///
    /// # Errors
    ///
    /// Returns [`RecompError::InvalidConfig`] when `token_budget` is `0`, or
    /// when `augmentation_threshold` / `dedup_similarity_threshold` are not
    /// finite values within `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), RecompError> {
        if self.token_budget == 0 {
            return Err(RecompError::InvalidConfig(
                "token_budget must be greater than 0".to_string(),
            ));
        }
        if !self.augmentation_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.augmentation_threshold)
        {
            return Err(RecompError::InvalidConfig(format!(
                "augmentation_threshold must be a finite value in [0.0, 1.0], got {}",
                self.augmentation_threshold
            )));
        }
        if !self.dedup_similarity_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.dedup_similarity_threshold)
        {
            return Err(RecompError::InvalidConfig(format!(
                "dedup_similarity_threshold must be a finite value in [0.0, 1.0], got {}",
                self.dedup_similarity_threshold
            )));
        }
        Ok(())
    }
}

// ── RecompDecision ─────────────────────────────────────────────────────────────

/// The outcome of RECOMP's selective-augmentation gate.
///
/// This is RECOMP's distinguishing contribution over plain context
/// compression: rather than *always* compressing and feeding the result to
/// the generator, the pipeline first estimates whether the retrieved
/// passages are relevant enough to be worth augmenting with at all.
#[derive(Debug, Clone, PartialEq)]
pub enum RecompDecision {
    /// The gate passed: `String` is the compressed text to feed the
    /// generator alongside (or instead of) the raw passages.
    Augment(String),
    /// The gate did not pass: compression was judged not worth performing
    /// because the retrieved passages were not relevant enough to the
    /// query. `reason` is a human-readable explanation.
    Skip {
        /// Human-readable explanation of why augmentation was skipped.
        reason: String,
    },
}

impl RecompDecision {
    /// Returns `true` if this decision is [`RecompDecision::Augment`].
    #[must_use]
    pub fn is_augment(&self) -> bool {
        matches!(self, Self::Augment(_))
    }

    /// Returns `true` if this decision is [`RecompDecision::Skip`].
    #[must_use]
    pub fn is_skip(&self) -> bool {
        matches!(self, Self::Skip { .. })
    }

    /// Returns the compressed text when this decision is
    /// [`RecompDecision::Augment`], or `None` otherwise.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Augment(text) => Some(text.as_str()),
            Self::Skip { .. } => None,
        }
    }
}

// ── RecompOutcome ──────────────────────────────────────────────────────────────

/// The full result of a single [`RecompPipeline::compress`](super::pipeline::RecompPipeline::compress) call.
#[derive(Debug, Clone, PartialEq)]
pub struct RecompOutcome {
    /// Whether the pipeline augmented (compressed) or skipped the passages.
    pub decision: RecompDecision,
    /// The mean per-passage lexical relevance score (in `[0.0, 1.0]`) computed
    /// by the selective-augmentation gate.
    pub relevance_score: f32,
    /// Number of passages that were passed in for this call.
    pub original_passage_count: usize,
    /// Approximate word count of the compressed text; `0` when the decision
    /// is [`RecompDecision::Skip`].
    pub compressed_token_count: usize,
}

// ── RecompError ────────────────────────────────────────────────────────────────

/// Errors produced by the `recomp` module.
#[derive(Debug, Error, PartialEq)]
pub enum RecompError {
    /// The supplied query was empty or contained only whitespace.
    #[error("query is empty")]
    EmptyQuery,
    /// The supplied passage slice was empty or contained only blank strings.
    #[error("passages must not be empty")]
    EmptyPassages,
    /// The pipeline configuration failed validation.
    #[error("invalid recomp config: {0}")]
    InvalidConfig(String),
}
