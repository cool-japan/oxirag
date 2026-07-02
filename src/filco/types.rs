//! Core types for the `filco` module: the [`FilterMeasure`] choice, the
//! [`FilcoConfig`] that parameterizes it, the per-sentence [`ScoredSentence`]
//! record, the aggregate [`FilcoReport`], and [`FilcoError`].

use thiserror::Error;

// ── FilterMeasure ─────────────────────────────────────────────────────────────

/// Which of FILCO's three sentence-level filtering measures
/// [`FilcoFilter`](super::filter::FilcoFilter) applies when scoring a
/// sentence.
///
/// Wang et al. (2023), "Learning to Filter Context for Retrieval-Augmented
/// Generation", score every retrieved sentence with one of three measures and
/// keep only the sentences that clear the corresponding threshold:
///
/// | Measure | Signal | Threshold field |
/// |---------|--------|------------------|
/// | [`FilterMeasure::StrInc`] (default) | String Inclusion — does the sentence contain a plausible answer span? | implicit (binary match / no match) |
/// | [`FilterMeasure::LexicalOverlap`] | Jaccard token overlap between the sentence and the query | [`FilcoConfig::lexical_threshold`] |
/// | [`FilterMeasure::CxmiLite`] | Heuristic proxy for Conditional Cross-Mutual Information | [`FilcoConfig::cxmi_threshold`] |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMeasure {
    /// String Inclusion (STRINC): keep a sentence when it contains a
    /// plausible answer span. See
    /// [`FilcoFilter::filter`](super::filter::FilcoFilter::filter) and
    /// [`FilcoFilter::filter_with_answer`](super::filter::FilcoFilter::filter_with_answer)
    /// for the heuristic (no known answer) and strict (known answer)
    /// variants respectively.
    #[default]
    StrInc,
    /// Lexical overlap: Jaccard similarity between the sentence's token set
    /// and the query's token set.
    LexicalOverlap,
    /// CXMI-lite: a deterministic, model-free proxy for the paper's
    /// Conditional Cross-Mutual Information measure.
    CxmiLite,
}

impl FilterMeasure {
    /// Returns a stable lower-case identifier for the measure.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::StrInc => "strinc",
            Self::LexicalOverlap => "lexical_overlap",
            Self::CxmiLite => "cxmi_lite",
        }
    }
}

impl std::fmt::Display for FilterMeasure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── FilcoConfig ────────────────────────────────────────────────────────────────

/// Configuration for [`FilcoFilter`](super::filter::FilcoFilter).
///
/// | Field | Default | Purpose |
/// |-------|---------|---------|
/// | `lexical_threshold` | `0.15` | Minimum Jaccard overlap for [`FilterMeasure::LexicalOverlap`] to keep a sentence |
/// | `cxmi_threshold` | `0.05` | Minimum CXMI-lite gain for [`FilterMeasure::CxmiLite`] to keep a sentence |
/// | `strinc_min_ngram` | `2` | Minimum contiguous shared token n-gram length for the heuristic STRINC match |
/// | `measure` | [`FilterMeasure::StrInc`] | Which measure [`FilcoFilter::filter`](super::filter::FilcoFilter::filter) uses |
#[derive(Debug, Clone, PartialEq)]
pub struct FilcoConfig {
    /// Minimum Jaccard token-overlap score (in `[0.0, 1.0]`) a sentence must
    /// exceed to be kept under [`FilterMeasure::LexicalOverlap`].
    ///
    /// Defaults to `0.15`.
    pub lexical_threshold: f32,
    /// Minimum CXMI-lite score a sentence must exceed to be kept under
    /// [`FilterMeasure::CxmiLite`].
    ///
    /// Defaults to `0.05`.
    pub cxmi_threshold: f32,
    /// Minimum length (in tokens) of a contiguous n-gram shared between a
    /// sentence and the query for the heuristic (no known answer)
    /// [`FilterMeasure::StrInc`] variant to consider the sentence a match on
    /// n-gram grounds alone.
    ///
    /// Must be `>= 1`. Defaults to `2`.
    pub strinc_min_ngram: usize,
    /// Which filtering measure [`FilcoFilter::filter`](super::filter::FilcoFilter::filter)
    /// applies.
    ///
    /// Defaults to [`FilterMeasure::StrInc`].
    pub measure: FilterMeasure,
}

impl Default for FilcoConfig {
    fn default() -> Self {
        Self {
            lexical_threshold: 0.15,
            cxmi_threshold: 0.05,
            strinc_min_ngram: 2,
            measure: FilterMeasure::StrInc,
        }
    }
}

impl FilcoConfig {
    /// Creates a new [`FilcoConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the lexical-overlap threshold.
    #[must_use]
    pub fn with_lexical_threshold(mut self, lexical_threshold: f32) -> Self {
        self.lexical_threshold = lexical_threshold;
        self
    }

    /// Sets the CXMI-lite threshold.
    #[must_use]
    pub fn with_cxmi_threshold(mut self, cxmi_threshold: f32) -> Self {
        self.cxmi_threshold = cxmi_threshold;
        self
    }

    /// Sets the minimum shared n-gram length used by the heuristic STRINC
    /// variant.
    #[must_use]
    pub fn with_strinc_min_ngram(mut self, strinc_min_ngram: usize) -> Self {
        self.strinc_min_ngram = strinc_min_ngram;
        self
    }

    /// Sets the filtering measure.
    #[must_use]
    pub fn with_measure(mut self, measure: FilterMeasure) -> Self {
        self.measure = measure;
        self
    }

    /// Validates the configuration's numeric bounds.
    ///
    /// # Errors
    ///
    /// Returns [`FilcoError::InvalidConfig`] when `lexical_threshold` or
    /// `cxmi_threshold` are not finite values within `[0.0, 1.0]`, or when
    /// `strinc_min_ngram` is `0`.
    pub fn validate(&self) -> Result<(), FilcoError> {
        if !self.lexical_threshold.is_finite() || !(0.0..=1.0).contains(&self.lexical_threshold) {
            return Err(FilcoError::InvalidConfig(format!(
                "lexical_threshold must be a finite value in [0.0, 1.0], got {}",
                self.lexical_threshold
            )));
        }
        if !self.cxmi_threshold.is_finite() || !(0.0..=1.0).contains(&self.cxmi_threshold) {
            return Err(FilcoError::InvalidConfig(format!(
                "cxmi_threshold must be a finite value in [0.0, 1.0], got {}",
                self.cxmi_threshold
            )));
        }
        if self.strinc_min_ngram == 0 {
            return Err(FilcoError::InvalidConfig(
                "strinc_min_ngram must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ── ScoredSentence ─────────────────────────────────────────────────────────────

/// A single sentence extracted from the input passages, together with its
/// score under the configured [`FilterMeasure`] and the keep/drop decision.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredSentence {
    /// The sentence's original surface text.
    pub text: String,
    /// The sentence's score under the configured [`FilterMeasure`].
    ///
    /// For [`FilterMeasure::StrInc`] the score is `1.0` for a match and a
    /// value in `[0.0, 1.0)` otherwise. For [`FilterMeasure::LexicalOverlap`]
    /// the score is a Jaccard similarity in `[0.0, 1.0]`. For
    /// [`FilterMeasure::CxmiLite`] the score is a non-negative relevance
    /// gain, typically in `[0.0, 1.0]`.
    pub score: f32,
    /// `true` if the sentence cleared the measure's threshold and was kept.
    pub kept: bool,
}

// ── FilcoReport ────────────────────────────────────────────────────────────────

/// The result of a single [`FilcoFilter::filter`](super::filter::FilcoFilter::filter)
/// (or
/// [`FilcoFilter::filter_with_answer`](super::filter::FilcoFilter::filter_with_answer))
/// call.
#[derive(Debug, Clone, PartialEq)]
pub struct FilcoReport {
    /// The filtered passages, one per input passage and in the same order,
    /// each reassembled from its kept sentences (in their original relative
    /// order) joined by single spaces. A passage with no surviving sentences
    /// becomes an empty string at its original index.
    pub filtered_passages: Vec<String>,
    /// Every sentence extracted from `passages`, flattened across all
    /// passages in original document order, together with its score and
    /// keep/drop decision.
    pub scored_sentences: Vec<ScoredSentence>,
    /// Total number of sentences extracted from the input passages.
    pub original_sentence_count: usize,
    /// Number of sentences that cleared the threshold and were kept.
    pub kept_sentence_count: usize,
    /// `kept_sentence_count / original_sentence_count`, in `[0.0, 1.0]`.
    /// `0.0` when `original_sentence_count` is `0`.
    pub compression_ratio: f32,
}

impl FilcoReport {
    /// Returns `true` when no sentence was kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kept_sentence_count == 0
    }
}

// ── FilcoError ─────────────────────────────────────────────────────────────────

/// Errors produced by the `filco` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FilcoError {
    /// The supplied query was empty or contained only whitespace.
    #[error("query is empty")]
    EmptyQuery,
    /// The supplied passage slice was empty, or every passage was empty or
    /// whitespace-only.
    #[error("passages must not be empty")]
    EmptyPassages,
    /// The filter configuration failed validation.
    #[error("invalid filco config: {0}")]
    InvalidConfig(String),
}
