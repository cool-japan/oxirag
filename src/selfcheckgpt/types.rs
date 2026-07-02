//! Core types for the `selfcheckgpt` module.

use thiserror::Error;

// ── SelfCheckVariant ──────────────────────────────────────────────────────────

/// Which zero-resource inconsistency-scoring heuristic to use.
///
/// Every variant estimates the same quantity — how well a main-response
/// *sentence* is corroborated by `K` independently sampled responses to the
/// same prompt — but each uses a different signal.  None of these variants
/// call out to an actual language model, entailment classifier, or QA system;
/// they are honest, purely lexical/statistical proxies for the corresponding
/// techniques described in Manakul et al. (EMNLP 2023).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelfCheckVariant {
    /// Pooled n-gram frequency model over the `K` samples plus the main
    /// response itself. A sentence whose n-grams rarely (or never) recur in
    /// the pool gets a high inconsistency score. Purely count-based — no
    /// external language model is used.
    NGram,
    /// Lexical-overlap (Jaccard) proxy for the paper's NLI-based variant.
    /// For each sample, the most similar sentence is found; overlap at or
    /// above a fixed threshold is treated as "entailment", below as
    /// "contradiction/neutral". This is **not** a real NLI classifier.
    NliLite,
    /// Cloze/keyword-masking proxy for the paper's QA-based variant. The
    /// sentence's most salient content token is masked and each sample is
    /// checked for that same token via keyword matching. This is **not** a
    /// real question-generation/question-answering model.
    QaLite,
}

impl Default for SelfCheckVariant {
    /// Defaults to [`SelfCheckVariant::NGram`], the purely count-based
    /// variant that requires no notion of "similarity threshold".
    fn default() -> Self {
        Self::NGram
    }
}

impl SelfCheckVariant {
    /// Returns a stable lower-case/snake_case identifier for the variant.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NGram => "ngram",
            Self::NliLite => "nli_lite",
            Self::QaLite => "qa_lite",
        }
    }
}

impl std::fmt::Display for SelfCheckVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── SelfCheckConfig ────────────────────────────────────────────────────────────

/// Configuration for [`SelfCheckScorer`](super::scorer::SelfCheckScorer).
#[derive(Debug, Clone, PartialEq)]
pub struct SelfCheckConfig {
    /// Which scoring heuristic to apply. Default: [`SelfCheckVariant::NGram`].
    pub variant: SelfCheckVariant,
    /// Size of the n-gram window used by [`SelfCheckVariant::NGram`]. Ignored
    /// by the other two variants. Must be `>= 1`. Default: `3` (trigrams).
    pub ngram_size: usize,
    /// Minimum per-sentence inconsistency score (in `[0.0, 1.0]`) at or above
    /// which a sentence is flagged as
    /// [`is_hallucination`](super::types::SentenceCheck::is_hallucination).
    /// Default: `0.5`.
    pub hallucination_threshold: f32,
    /// Minimum number of sampled responses required to run a check. The
    /// `SelfCheckGPT` paper samples `K` stochastic responses; too few samples
    /// make the consistency signal unreliable. Must be `>= 1`.
    /// Default: `3`.
    pub min_samples: usize,
}

impl Default for SelfCheckConfig {
    fn default() -> Self {
        Self {
            variant: SelfCheckVariant::NGram,
            ngram_size: 3,
            hallucination_threshold: 0.5,
            min_samples: 3,
        }
    }
}

impl SelfCheckConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the scoring variant.
    #[must_use]
    pub fn with_variant(mut self, variant: SelfCheckVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Sets the n-gram window size (used only by [`SelfCheckVariant::NGram`]).
    #[must_use]
    pub fn with_ngram_size(mut self, ngram_size: usize) -> Self {
        self.ngram_size = ngram_size;
        self
    }

    /// Sets the hallucination-flagging threshold.
    #[must_use]
    pub fn with_hallucination_threshold(mut self, hallucination_threshold: f32) -> Self {
        self.hallucination_threshold = hallucination_threshold;
        self
    }

    /// Sets the minimum required sample count.
    #[must_use]
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Validates the configuration's numeric invariants.
    ///
    /// # Errors
    ///
    /// Returns [`SelfCheckError::InvalidConfig`] when:
    /// - `ngram_size == 0`;
    /// - `hallucination_threshold` is outside `[0.0, 1.0]` or non-finite;
    /// - `min_samples == 0`.
    pub fn validate(&self) -> Result<(), SelfCheckError> {
        if self.ngram_size == 0 {
            return Err(SelfCheckError::InvalidConfig(
                "ngram_size must be at least 1".to_string(),
            ));
        }
        if !self.hallucination_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.hallucination_threshold)
        {
            return Err(SelfCheckError::InvalidConfig(format!(
                "hallucination_threshold must be a finite value in [0.0, 1.0], got {}",
                self.hallucination_threshold
            )));
        }
        if self.min_samples == 0 {
            return Err(SelfCheckError::InvalidConfig(
                "min_samples must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ── SentenceCheck ──────────────────────────────────────────────────────────────

/// Per-sentence inconsistency result for a single main-response sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceCheck {
    /// The original sentence text, as split from the main response.
    pub sentence: String,
    /// Inconsistency score in `[0.0, 1.0]`. Higher means less corroborated by
    /// the sampled responses (and therefore more likely to be hallucinated).
    pub inconsistency_score: f32,
    /// `true` when `inconsistency_score >= `
    /// [`SelfCheckConfig::hallucination_threshold`].
    pub is_hallucination: bool,
}

// ── SelfCheckScore ─────────────────────────────────────────────────────────────

/// Full result of scoring a main response against its samples.
///
/// Produced by
/// [`SelfCheckScorer::score`](super::scorer::SelfCheckScorer::score).
#[derive(Debug, Clone, PartialEq)]
pub struct SelfCheckScore {
    /// Per-sentence inconsistency results, in the order the sentences appear
    /// in the main response.
    pub sentence_checks: Vec<SentenceCheck>,
    /// Document-level score: the mean of all
    /// [`SentenceCheck::inconsistency_score`] values (`0.0` when there are no
    /// sentences).
    pub overall_score: f32,
    /// The variant that produced this score.
    pub variant_used: SelfCheckVariant,
}

impl SelfCheckScore {
    /// Number of sentences flagged as hallucinations.
    #[must_use]
    pub fn hallucination_count(&self) -> usize {
        self.sentence_checks
            .iter()
            .filter(|c| c.is_hallucination)
            .count()
    }

    /// `true` when no sentence was flagged as a hallucination.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.hallucination_count() == 0
    }
}

// ── SelfCheckError ─────────────────────────────────────────────────────────────

/// Errors produced by the `selfcheckgpt` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SelfCheckError {
    /// The main response was empty or contained only whitespace.
    #[error("main response is empty")]
    EmptyResponse,
    /// Fewer samples were supplied than
    /// [`SelfCheckConfig::min_samples`] requires.
    #[error("insufficient samples: got {got}, need at least {need}")]
    InsufficientSamples {
        /// Number of samples actually supplied.
        got: usize,
        /// Minimum number of samples required.
        need: usize,
    },
    /// The supplied [`SelfCheckConfig`] failed validation.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}
