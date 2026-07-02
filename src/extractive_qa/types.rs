//! Types for the `extractive_qa` module: question-type classification,
//! candidate/answer span records, the scoring breakdown, configuration, and
//! the crate-local error/result aliases.
//!
//! The actual classification, candidate generation, and scoring logic lives
//! in the [`engine`](super::engine) submodule; this file only defines the
//! data that flows through it.

use thiserror::Error;

// ── QuestionType ──────────────────────────────────────────────────────────────

/// The interrogative category of a question, used to select which
/// content-match heuristic [`crate::extractive_qa::ExtractiveQaEngine`]
/// applies when scoring candidate answer spans.
///
/// Classification is a simple keyword/pattern match on the question text
/// (see [`ExtractiveQaConfig::type_keywords`]); it is not a syntactic parse,
/// so it can be fooled by unusual phrasing. `Other` is the honest fallback
/// for every question that does not contain a recognised interrogative
/// keyword (including bare "how" questions such as "How did the war end?",
/// which are deliberately **not** folded into [`QuestionType::HowMany`] /
/// [`QuestionType::HowMuch`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionType {
    /// Asks for a person or organisation ("who…").
    Who,
    /// Asks for a general fact, definition, or entity ("what…").
    What,
    /// Asks for a date, year, or time ("when…").
    When,
    /// Asks for a place ("where…").
    Where,
    /// Asks for a count ("how many…").
    HowMany,
    /// Asks for a quantity, price, or extent ("how much…").
    HowMuch,
    /// Asks for a reason ("why…").
    Why,
    /// Asks for a selection among alternatives ("which…").
    Which,
    /// No recognised interrogative keyword was found.
    Other,
}

impl QuestionType {
    /// Return a stable, lowercase, `snake_case` name for this question type.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Who => "who",
            Self::What => "what",
            Self::When => "when",
            Self::Where => "where",
            Self::HowMany => "how_many",
            Self::HowMuch => "how_much",
            Self::Why => "why",
            Self::Which => "which",
            Self::Other => "other",
        }
    }
}

// ── SpanCandidate ────────────────────────────────────────────────────────────

/// A candidate answer span: a contiguous run of whitespace-delimited tokens
/// drawn from a passage, before scoring.
///
/// `start_token`/`end_token` are token indices into the passage's
/// `str::split_whitespace()` sequence, with `end_token` **exclusive**
/// (following Rust slice convention: the span covers tokens
/// `start_token..end_token`). `text` is the verbatim run of source tokens
/// re-joined with single spaces — a contiguous slice of the source token
/// sequence, though inter-token whitespace from the original passage (extra
/// spaces, newlines) is normalised to a single space in the process.
///
/// Produced by
/// [`ExtractiveQaEngine::generate_candidates`](crate::extractive_qa::ExtractiveQaEngine::generate_candidates);
/// consumed by
/// [`ExtractiveQaEngine::score_span`](crate::extractive_qa::ExtractiveQaEngine::score_span).
/// Scoring re-derives a candidate's tokens from a *given* passage using
/// `start_token`/`end_token` rather than trusting `text`, so a candidate
/// should only be scored against the passage it was generated from (or
/// another passage that is index-consistent with it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanCandidate {
    /// Index of the first token of the span (inclusive).
    pub start_token: usize,
    /// Index one past the last token of the span (exclusive).
    pub end_token: usize,
    /// The verbatim text of the span: source tokens `start_token..end_token`
    /// re-joined with single spaces.
    pub text: String,
}

impl SpanCandidate {
    /// Construct a new candidate span.
    #[must_use]
    pub fn new(start_token: usize, end_token: usize, text: impl Into<String>) -> Self {
        Self {
            start_token,
            end_token,
            text: text.into(),
        }
    }

    /// Number of tokens covered by this span (`end_token - start_token`,
    /// saturating at zero for a malformed candidate where
    /// `end_token < start_token`).
    #[must_use]
    pub fn len_tokens(&self) -> usize {
        self.end_token.saturating_sub(self.start_token)
    }
}

// ── SpanScore ─────────────────────────────────────────────────────────────────

/// The scoring breakdown for one [`SpanCandidate`], exposed for
/// transparency rather than collapsed into a single opaque number.
///
/// `total` is the weighted blend of the other three fields (see
/// [`ExtractiveQaConfig`] for the weights); it is what
/// [`ExtractiveQaEngine::answer`](crate::extractive_qa::ExtractiveQaEngine::answer)
/// and
/// [`ExtractiveQaEngine::top_k_spans`](crate::extractive_qa::ExtractiveQaEngine::top_k_spans)
/// rank candidates by. Each of `type_match`, `context_overlap`, and
/// `length_prior` independently lies in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpanScore {
    /// Question-type-aware content match: how well the span *itself* fits
    /// the pattern expected for the classified [`QuestionType`] (a
    /// capitalized run for `Who`, a date/year token for `When`, a numeric
    /// token for `HowMany`/`HowMuch`, a capitalized token or
    /// prepositional-phrase context for `Where`; lexical overlap between the
    /// span and the question's content words for `What`/`Why`/`Which`/`Other`).
    pub type_match: f32,
    /// Lexical overlap between the question's content words and the words
    /// immediately surrounding the span (not the span itself).
    pub context_overlap: f32,
    /// A length prior in `[0.0, 1.0]` that mildly penalises long spans
    /// (`1.0` at a single token, decaying towards `0.0` as the span grows).
    pub length_prior: f32,
    /// The weighted combination of the three signals above; see
    /// [`ExtractiveQaConfig::weight_type_match`],
    /// [`ExtractiveQaConfig::weight_context_overlap`], and
    /// [`ExtractiveQaConfig::weight_length_prior`].
    pub total: f32,
}

// ── AnswerSpan ────────────────────────────────────────────────────────────────

/// A scored answer span returned by
/// [`ExtractiveQaEngine::answer`](crate::extractive_qa::ExtractiveQaEngine::answer)
/// or
/// [`ExtractiveQaEngine::top_k_spans`](crate::extractive_qa::ExtractiveQaEngine::top_k_spans).
///
/// Carries the verbatim `text`, its token offsets into the source passage,
/// the [`QuestionType`] the question was classified as, and the full
/// [`SpanScore`] breakdown that produced it — so callers can inspect *why*
/// a span was selected, not just *that* it was.
#[derive(Debug, Clone, PartialEq)]
pub struct AnswerSpan {
    /// The verbatim answer text (source tokens re-joined with single spaces;
    /// see [`SpanCandidate::text`]).
    pub text: String,
    /// Index of the first token of the answer (inclusive).
    pub start_token: usize,
    /// Index one past the last token of the answer (exclusive).
    pub end_token: usize,
    /// The classified type of the question this span answers.
    pub question_type: QuestionType,
    /// The scoring breakdown that produced this span.
    pub score: SpanScore,
}

impl AnswerSpan {
    /// Number of tokens covered by this answer span.
    #[must_use]
    pub fn len_tokens(&self) -> usize {
        self.end_token.saturating_sub(self.start_token)
    }
}

// ── ExtractiveQaConfig ───────────────────────────────────────────────────────

/// Configuration for [`crate::extractive_qa::ExtractiveQaEngine`].
///
/// # Fields & defaults
///
/// | Field | Meaning | Default |
/// |-------|---------|---------|
/// | [`max_span_tokens`](Self::max_span_tokens) | Longest candidate span considered | `10` |
/// | [`context_window_tokens`](Self::context_window_tokens) | Tokens of surrounding context inspected on each side | `6` |
/// | [`weight_type_match`](Self::weight_type_match) | Blend weight for the type-match signal | `0.5` |
/// | [`weight_context_overlap`](Self::weight_context_overlap) | Blend weight for the context-overlap signal | `0.3` |
/// | [`weight_length_prior`](Self::weight_length_prior) | Blend weight for the length prior | `0.2` |
/// | [`length_prior_decay`](Self::length_prior_decay) | Decay rate of the length prior | `0.15` |
/// | [`type_keywords`](Self::type_keywords) | Keyword phrases used to classify questions | see [`QuestionType`] |
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractiveQaConfig {
    /// Longest candidate span considered, in tokens. Candidate generation is
    /// bounded to `O(passage_len * max_span_tokens)` rather than the
    /// `O(passage_len^2)` of considering every possible span; the trade-off
    /// is that no answer longer than `max_span_tokens` can ever be returned.
    /// `SQuAD`-style extractive answers are almost always short (a handful of
    /// tokens), so this is a favourable trade-off in the common case; raise
    /// it for passages that plausibly need longer answers. Defaults to `10`.
    pub max_span_tokens: usize,
    /// Number of tokens of surrounding context inspected on *each* side of a
    /// candidate span when computing lexical context overlap. Defaults to
    /// `6`.
    pub context_window_tokens: usize,
    /// Blend weight for the question-type-aware content-match signal.
    /// Defaults to `0.5`.
    pub weight_type_match: f32,
    /// Blend weight for the lexical context-overlap signal. Defaults to
    /// `0.3`.
    pub weight_context_overlap: f32,
    /// Blend weight for the length prior. Defaults to `0.2`.
    pub weight_length_prior: f32,
    /// Decay rate of the length prior: `length_prior = 1 / (1 + decay *
    /// (len - 1))`, so it equals `1.0` at a single token and decreases
    /// smoothly (never reaching zero) as the span grows. Defaults to
    /// `0.15`.
    pub length_prior_decay: f32,
    /// Ordered `(question type, keyword phrases)` mappings used to classify
    /// a question's [`QuestionType`]. The question is scanned left to
    /// right; the **leftmost** token position at which any keyword phrase
    /// matches determines the type, with ties at the same position broken
    /// by the order of this list. A question that matches no phrase is
    /// classified [`QuestionType::Other`]. Overridable via
    /// [`ExtractiveQaConfig::with_type_keywords`] or, to patch a single
    /// type's phrases, [`ExtractiveQaConfig::with_keywords_for`].
    pub type_keywords: Vec<(QuestionType, Vec<String>)>,
}

impl Default for ExtractiveQaConfig {
    fn default() -> Self {
        Self {
            max_span_tokens: 10,
            context_window_tokens: 6,
            weight_type_match: 0.5,
            weight_context_overlap: 0.3,
            weight_length_prior: 0.2,
            length_prior_decay: 0.15,
            type_keywords: default_type_keywords(),
        }
    }
}

impl ExtractiveQaConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the longest candidate span considered, in tokens.
    #[must_use]
    pub fn with_max_span_tokens(mut self, max_span_tokens: usize) -> Self {
        self.max_span_tokens = max_span_tokens;
        self
    }

    /// Override the number of tokens of surrounding context inspected on
    /// each side of a candidate span.
    #[must_use]
    pub fn with_context_window_tokens(mut self, context_window_tokens: usize) -> Self {
        self.context_window_tokens = context_window_tokens;
        self
    }

    /// Override the blend weight for the question-type-aware content-match
    /// signal.
    #[must_use]
    pub fn with_weight_type_match(mut self, weight_type_match: f32) -> Self {
        self.weight_type_match = weight_type_match;
        self
    }

    /// Override the blend weight for the lexical context-overlap signal.
    #[must_use]
    pub fn with_weight_context_overlap(mut self, weight_context_overlap: f32) -> Self {
        self.weight_context_overlap = weight_context_overlap;
        self
    }

    /// Override the blend weight for the length prior.
    #[must_use]
    pub fn with_weight_length_prior(mut self, weight_length_prior: f32) -> Self {
        self.weight_length_prior = weight_length_prior;
        self
    }

    /// Override the decay rate of the length prior.
    #[must_use]
    pub fn with_length_prior_decay(mut self, length_prior_decay: f32) -> Self {
        self.length_prior_decay = length_prior_decay;
        self
    }

    /// Replace the entire question-type keyword table.
    #[must_use]
    pub fn with_type_keywords(mut self, type_keywords: Vec<(QuestionType, Vec<String>)>) -> Self {
        self.type_keywords = type_keywords;
        self
    }

    /// Override the keyword phrases for a single `question_type`, leaving
    /// every other type's phrases untouched. Replaces the existing entry
    /// for `question_type` if one is present, or appends a new one
    /// (participating in leftmost-match classification at its new position
    /// in the table) otherwise.
    #[must_use]
    pub fn with_keywords_for(mut self, question_type: QuestionType, keywords: Vec<String>) -> Self {
        if let Some(entry) = self
            .type_keywords
            .iter_mut()
            .find(|(existing_type, _)| *existing_type == question_type)
        {
            entry.1 = keywords;
        } else {
            self.type_keywords.push((question_type, keywords));
        }
        self
    }
}

/// The default question-type keyword table.
///
/// `HowMany`/`HowMuch` are listed first so that, at the same leftmost token
/// position, a two-word phrase match ("how many"/"how much") takes priority
/// over any other mapping — in practice this never contends with anything
/// else, since bare "how" is not a keyword for any type, but keeping the
/// multi-word phrases first documents the intended precedence explicitly.
fn default_type_keywords() -> Vec<(QuestionType, Vec<String>)> {
    vec![
        (QuestionType::HowMany, vec!["how many".to_string()]),
        (QuestionType::HowMuch, vec!["how much".to_string()]),
        (
            QuestionType::Who,
            vec!["who".to_string(), "whom".to_string()],
        ),
        (QuestionType::What, vec!["what".to_string()]),
        (QuestionType::When, vec!["when".to_string()]),
        (QuestionType::Where, vec!["where".to_string()]),
        (QuestionType::Why, vec!["why".to_string()]),
        (QuestionType::Which, vec!["which".to_string()]),
    ]
}

// ── ExtractiveQaResult / ExtractiveQaError ──────────────────────────────────

/// Convenience alias for a [`Result`] whose error is [`ExtractiveQaError`].
pub type ExtractiveQaResult<T> = Result<T, ExtractiveQaError>;

/// Errors produced by the `extractive_qa` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ExtractiveQaError {
    /// The supplied question was empty (or whitespace-only) after trimming.
    #[error("question must not be empty")]
    EmptyQuestion,
    /// The supplied passage was empty (or whitespace-only) after trimming.
    #[error("passage must not be empty")]
    EmptyPassage,
    /// No candidate answer spans could be generated for the passage under
    /// the current configuration — either the passage tokenized to zero
    /// tokens (already rejected earlier as [`ExtractiveQaError::EmptyPassage`]
    /// in practice) or [`ExtractiveQaConfig::max_span_tokens`] is `0`.
    #[error(
        "no viable answer span candidates: passage has {passage_tokens} token(s) but \
         max_span_tokens is {max_span_tokens}"
    )]
    NoViableCandidates {
        /// The configured [`ExtractiveQaConfig::max_span_tokens`] at the
        /// time of the call.
        max_span_tokens: usize,
        /// The number of whitespace tokens the passage contained.
        passage_tokens: usize,
    },
    /// [`ExtractiveQaEngine::score_span`](crate::extractive_qa::ExtractiveQaEngine::score_span)
    /// was called with a [`SpanCandidate`] whose bounds are not a valid,
    /// non-empty range within the given passage.
    #[error(
        "invalid span bounds: start_token={start_token}, end_token={end_token}, passage has \
         {passage_tokens} token(s)"
    )]
    InvalidSpanBounds {
        /// The candidate's `start_token`.
        start_token: usize,
        /// The candidate's `end_token`.
        end_token: usize,
        /// The number of whitespace tokens the passage contained.
        passage_tokens: usize,
    },
}
