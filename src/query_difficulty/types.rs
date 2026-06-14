//! Core types for query-difficulty prediction and answerability estimation.
//!
//! This module provides the building blocks for estimating how hard a natural-
//! language query is to answer.  Difficulty is expressed as a continuous score
//! in `[0.0, 1.0]` which is then bucketed into a [`DifficultyBand`].  The
//! decomposed [`DifficultySignals`] struct exposes every contributing dimension
//! for downstream explainability.
//!
//! ## Score formula
//!
//! ```text
//! score =   negation_weight    * has_negation
//!         + multi_hop_weight   * multi_hop_indicator
//!         + ambiguity_weight   * ambiguity_marker
//!         + length_weight      * length_score       (token_count / 15, clamped to 1)
//!         + rare_word_weight   * rare_word_ratio    (fraction of tokens outside STOP_WORDS)
//! ```
//!
//! Default weights sum to 1.0, so the score is already in `[0.0, 1.0]` when
//! all signals fire at maximum strength.

use thiserror::Error;

// ── STOP_WORDS ────────────────────────────────────────────────────────────────

/// Common English words that are excluded from `rare_word_ratio` counting.
///
/// A token that appears in this list is treated as *not rare* (a function
/// word).  Words outside this list increment the rare-word counter and
/// therefore raise the difficulty score.
pub static STOP_WORDS: &[&str] = &[
    // Articles & determiners
    "a", "an", "the", // Prepositions
    "in", "on", "at", "by", "for", "of", "to", "with", "as", "from", "into", "through", "upon",
    "onto", "about", "against", "within", // Copula / auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    // Modal verbs
    "will", "would", "could", "should", "may", "might", "can", "shall",
    // Demonstratives & locatives
    "this", "that", "these", "those", "here", "there", // Question words
    "who", "which", "what", "where", "when", "why", "how", // Conjunctions
    "and", "or", "but", "if", "so", "yet", "nor", "while", "both", "either", "also", "neither",
    // Negation helpers (the negative words themselves trigger a separate
    // signal; keeping them in STOP_WORDS prevents them from also inflating
    // rare_word_ratio)
    "not", "no", // Pronouns
    "it", "its", "we", "you", "he", "she", "they", "them", "his", "her", "our", "your", "my", "me",
    "us", "him", "their", // Quantifiers / determiners
    "each", "any", "all", "most", "some", "such", "more", "other", "one", "two", "very", "just",
    "up", "out",
];

// ── NEGATION_WORDS ────────────────────────────────────────────────────────────

/// Words and short phrases that signal negation.
///
/// Each entry is matched against the lowercased query with whole-word (or
/// phrase) boundary checking, so `"no"` will not accidentally match inside
/// `"notable"`.
pub static NEGATION_WORDS: &[&str] = &[
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "without",
    "cannot",
    "isn't",
    "aren't",
    "don't",
    "didn't",
    "doesn't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "can't",
    "needn't",
    "hardly",
    "scarcely",
    "barely",
    "nothing",
    "nobody",
    "nowhere",
    "none",
];

// ── MULTI_HOP_INDICATORS ──────────────────────────────────────────────────────

/// Words and phrases that indicate a query spans multiple reasoning steps.
///
/// These signal that answering the query requires chaining facts across several
/// retrieval passes or knowledge-base nodes, which generally increases
/// difficulty.
pub static MULTI_HOP_INDICATORS: &[&str] = &[
    "and",
    "both",
    "also",
    "additionally",
    "furthermore",
    "while",
    "whereas",
    "compared to",
    "between",
    "relationship between",
    "connection between",
    "difference between",
    "similarities between",
    "contrast",
    "comparison",
    "alongside",
    "together with",
    "in addition",
    "as well as",
    "not only",
    "moreover",
    "simultaneously",
    "at the same time",
];

// ── AMBIGUITY_MARKERS ─────────────────────────────────────────────────────────

/// Words and phrases that signal semantic ambiguity or underspecification.
///
/// Their presence suggests that the question itself may be unclear or that
/// multiple valid answers exist, both of which increase answerability
/// difficulty.
pub static AMBIGUITY_MARKERS: &[&str] = &[
    "who",
    "which one",
    "or",
    "either",
    "it depends",
    "unclear",
    "vague",
    "maybe",
    "perhaps",
    "possibly",
    "ambiguous",
    "uncertain",
    "not sure",
    "might be",
    "could be",
    "any of",
    "one of",
    "some of",
    "in some cases",
];

// ── DifficultyBand ────────────────────────────────────────────────────────────

/// The difficulty band to which a query is assigned after scoring.
///
/// | Band | Score range | Interpretation |
/// |------|-------------|----------------|
/// | [`Easy`](DifficultyBand::Easy) | `[0.0, 0.3)` | Direct factoid; simple retrieval likely sufficient |
/// | [`Medium`](DifficultyBand::Medium) | `[0.3, 0.55)` | Moderate complexity; standard RAG pipeline |
/// | [`Hard`](DifficultyBand::Hard) | `[0.55, 0.75)` | Multi-hop or negation-heavy; advanced retrieval advised |
/// | [`Ambiguous`](DifficultyBand::Ambiguous) | `[0.75, 1.0]` | Underspecified or contradictory; may need clarification |
#[derive(Debug, Clone, PartialEq)]
pub enum DifficultyBand {
    /// Score `< 0.3`: straightforward factoid question.
    Easy,
    /// Score in `[0.3, 0.55)`: moderate complexity.
    Medium,
    /// Score in `[0.55, 0.75)`: multi-hop reasoning or heavy negation.
    Hard,
    /// Score `>= 0.75`: semantically ambiguous or underspecified.
    Ambiguous,
}

impl DifficultyBand {
    /// Returns a stable lower-case identifier for the band.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Easy => "easy",
            Self::Medium => "medium",
            Self::Hard => "hard",
            Self::Ambiguous => "ambiguous",
        }
    }
}

impl std::fmt::Display for DifficultyBand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── DifficultySignals ─────────────────────────────────────────────────────────

/// Decomposed per-dimension signals produced by [`DifficultyPredictor::predict`].
///
/// Each field is an independently interpretable feature.  Downstream callers
/// can inspect them for routing decisions or explainability logging without
/// having to re-run the predictor.
///
/// [`DifficultyPredictor::predict`]: super::predictor::DifficultyPredictor::predict
#[derive(Debug, Clone, PartialEq)]
pub struct DifficultySignals {
    /// `true` when any [`NEGATION_WORDS`] entry is found in the query.
    pub has_negation: bool,
    /// `true` when any [`MULTI_HOP_INDICATORS`] entry is found, suggesting
    /// that the query requires chaining multiple retrieval steps.
    pub multi_hop_indicator: bool,
    /// `true` when any [`AMBIGUITY_MARKERS`] entry is found, suggesting the
    /// query is underspecified or admits multiple valid readings.
    pub ambiguity_marker: bool,
    /// Number of tokens after splitting on non-alphanumeric characters and
    /// discarding single-character tokens.
    pub token_count: usize,
    /// Fraction of tokens that are *not* in [`STOP_WORDS`], in `[0.0, 1.0]`.
    /// Higher values indicate more domain-specific or unusual vocabulary.
    pub rare_word_ratio: f32,
}

// ── DifficultyScore ───────────────────────────────────────────────────────────

/// The complete result of a single query-difficulty prediction.
///
/// Produced by [`DifficultyPredictor::predict`].  It holds the original query
/// string, the aggregate difficulty `score` in `[0.0, 1.0]`, the assigned
/// [`DifficultyBand`], and the decomposed [`DifficultySignals`] that explain
/// how the score was reached.
///
/// [`DifficultyPredictor::predict`]: super::predictor::DifficultyPredictor::predict
#[derive(Debug, Clone, PartialEq)]
pub struct DifficultyScore {
    /// The original query text, unchanged.
    pub query: String,
    /// Aggregate difficulty score in `[0.0, 1.0]`.  Higher values indicate
    /// greater predicted difficulty.
    pub score: f32,
    /// The difficulty band derived from `score`.
    pub band: DifficultyBand,
    /// Per-dimension signals that contributed to `score`.
    pub signals: DifficultySignals,
}

// ── DifficultyConfig ──────────────────────────────────────────────────────────

/// Configuration for the [`DifficultyPredictor`].
///
/// Each weight controls how strongly the corresponding dimension contributes
/// to the aggregate difficulty score.  With the default values the five weights
/// sum to `1.0`, so the score is already in `[0.0, 1.0]` when all signals are
/// maximally active.
///
/// | Field | Default | Dimension |
/// |-------|---------|-----------|
/// | `negation_weight` | `0.20` | Presence of negation words |
/// | `multi_hop_weight` | `0.30` | Multi-hop / relational indicators |
/// | `ambiguity_weight` | `0.20` | Ambiguity markers |
/// | `length_weight` | `0.15` | Normalised query length |
/// | `rare_word_weight` | `0.15` | Fraction of rare (non-stop) words |
///
/// [`DifficultyPredictor`]: super::predictor::DifficultyPredictor
#[derive(Debug, Clone, PartialEq)]
pub struct DifficultyConfig {
    /// Weight of the negation-detection signal in the aggregate score.
    ///
    /// Default: `0.2`.
    pub negation_weight: f32,
    /// Weight of the multi-hop indicator signal in the aggregate score.
    ///
    /// Default: `0.3`.
    pub multi_hop_weight: f32,
    /// Weight of the ambiguity-marker signal in the aggregate score.
    ///
    /// Default: `0.2`.
    pub ambiguity_weight: f32,
    /// Weight of the normalised-length signal in the aggregate score.
    ///
    /// Default: `0.15`.
    pub length_weight: f32,
    /// Weight of the rare-word-ratio signal in the aggregate score.
    ///
    /// Default: `0.15`.
    pub rare_word_weight: f32,
}

impl Default for DifficultyConfig {
    fn default() -> Self {
        Self {
            negation_weight: 0.2,
            multi_hop_weight: 0.3,
            ambiguity_weight: 0.2,
            length_weight: 0.15,
            rare_word_weight: 0.15,
        }
    }
}

impl DifficultyConfig {
    /// Creates a new [`DifficultyConfig`] with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the negation-signal weight.
    #[must_use]
    pub fn with_negation_weight(mut self, w: f32) -> Self {
        self.negation_weight = w;
        self
    }

    /// Sets the multi-hop-signal weight.
    #[must_use]
    pub fn with_multi_hop_weight(mut self, w: f32) -> Self {
        self.multi_hop_weight = w;
        self
    }

    /// Sets the ambiguity-signal weight.
    #[must_use]
    pub fn with_ambiguity_weight(mut self, w: f32) -> Self {
        self.ambiguity_weight = w;
        self
    }

    /// Sets the length-signal weight.
    #[must_use]
    pub fn with_length_weight(mut self, w: f32) -> Self {
        self.length_weight = w;
        self
    }

    /// Sets the rare-word-ratio signal weight.
    #[must_use]
    pub fn with_rare_word_weight(mut self, w: f32) -> Self {
        self.rare_word_weight = w;
        self
    }
}

// ── DifficultyError ───────────────────────────────────────────────────────────

/// Errors produced by the `query_difficulty` module.
#[derive(Debug, Error, PartialEq)]
pub enum DifficultyError {
    /// The supplied query was empty or contained only whitespace.
    #[error("query is empty")]
    EmptyQuery,
    /// Score computation failed (e.g. non-finite weights produced NaN/Inf).
    #[error("prediction failed: {0}")]
    PredictionFailed(String),
}
