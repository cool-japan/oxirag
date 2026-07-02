//! [`ExtractiveQaEngine`] — question-type classification, bounded candidate
//! span generation, and question-type-aware boundary scoring.
//!
//! # Lexical helpers
//!
//! Deliberately self-contained (not imported from `searchain`,
//! `quote_grounding`, or any other module): each module in this crate owns
//! its own small lexical toolkit rather than sharing private helpers across
//! module boundaries.

use std::cmp::Ordering;
use std::collections::HashSet;

use super::types::{
    AnswerSpan, ExtractiveQaConfig, ExtractiveQaError, ExtractiveQaResult, QuestionType,
    SpanCandidate, SpanScore,
};

// ── lexical constants ────────────────────────────────────────────────────────

/// Minimum character length for a token to count as a *content* word (used
/// for lexical-overlap scoring, not for the question-type keyword match).
const MIN_CONTENT_WORD_LEN: usize = 3;

/// Stopwords excluded from the content vocabulary used for overlap scoring.
/// Includes ordinary English stopwords plus the interrogative words
/// themselves (so a span containing the literal word "who" does not get
/// overlap credit for echoing a "Who" question's own interrogative).
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
    "that", "this", "with", "from", "they", "have", "were", "what", "your", "when", "them", "then",
    "than", "into", "some", "such", "only", "also", "been", "more", "very", "will", "would",
    "there", "their", "which", "about", "could", "these", "those", "does", "where", "why", "whom",
    "much", "many", "after", "before", "during", "between", "again", "here", "both", "each",
    "other", "most", "same", "own", "just", "should", "being", "doing", "having", "because",
    "while", "against", "whose", "whether", "among", "under", "over",
];

/// Month names and common abbreviations recognised by the `When`
/// date-likeness heuristic.
const MONTH_NAMES: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
];

/// Prepositions that, immediately preceding a span, boost the `Where`
/// content-match heuristic (a prepositional-phrase context signal).
const LOCATION_PREPOSITIONS: &[&str] = &["in", "at", "near"];

/// Inclusive range of plausible calendar years for the `When` heuristic's
/// year-likeness check.
const YEAR_RANGE: std::ops::RangeInclusive<u32> = 1000..=2100;

// ── generic word helpers ─────────────────────────────────────────────────────

/// The alphanumeric core of a whitespace-delimited `word`: leading and
/// trailing non-alphanumeric characters (punctuation) are trimmed, but
/// interior characters (e.g. an apostrophe in "O'Brien") are preserved.
fn core_word(word: &str) -> &str {
    word.trim_matches(|c: char| !c.is_alphanumeric())
}

/// [`core_word`], lowercased.
fn core_lower(word: &str) -> String {
    core_word(word).to_lowercase()
}

/// Return `true` when `word`'s alphanumeric core is at least
/// [`MIN_CONTENT_WORD_LEN`] characters, is not a stopword, and is not a
/// purely numeric token (numeric tokens are handled by the dedicated
/// `HowMany`/`HowMuch`/`When` heuristics instead of generic lexical
/// overlap).
fn is_content_word(word: &str) -> bool {
    let core = core_word(word);
    if core.chars().count() < MIN_CONTENT_WORD_LEN {
        return false;
    }
    if core.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    !STOPWORDS.contains(&core.to_lowercase().as_str())
}

/// The distinct set of content words in `text` (whitespace-tokenized,
/// lowercased cores; see [`is_content_word`]).
fn content_word_set(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .filter(|w| is_content_word(w))
        .map(core_lower)
        .collect()
}

/// Parse `word`'s alphanumeric core as a number, tolerating thousands
/// separators and surrounding currency/percent symbols (e.g. `"$3,500"` and
/// `"50%"` both parse). Returns `None` when the core has no digits at all,
/// or contains more than one decimal point.
///
/// Deliberately covers digit-form numerals only, not spelled-out number
/// words ("two", "three") — see the `HowMany`/`HowMuch` discussion in the
/// [module documentation](crate::extractive_qa).
fn numeric_value(word: &str) -> Option<f64> {
    let core = core_word(word);
    let cleaned: String = core
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        return None;
    }
    if cleaned.matches('.').count() > 1 {
        return None;
    }
    cleaned.parse::<f64>().ok()
}

/// Return `true` when `word`'s alphanumeric core is a plausible 4-digit
/// calendar year (see [`YEAR_RANGE`]).
fn is_year_like(word: &str) -> bool {
    let core = core_word(word);
    if core.chars().count() != 4 || !core.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    core.parse::<u32>()
        .is_ok_and(|year| YEAR_RANGE.contains(&year))
}

/// Return `true` when `word`'s alphanumeric core is a recognised month name
/// or abbreviation (see [`MONTH_NAMES`]).
fn is_month_name(word: &str) -> bool {
    MONTH_NAMES.contains(&core_word(word).to_lowercase().as_str())
}

/// Return `true` when `word` looks date-like: a plausible year or a month
/// name.
fn is_date_like(word: &str) -> bool {
    is_year_like(word) || is_month_name(word)
}

/// Return `true` when `word`'s alphanumeric core starts with an alphabetic
/// character.
fn starts_alphabetic(word: &str) -> bool {
    core_word(word)
        .chars()
        .next()
        .is_some_and(char::is_alphabetic)
}

/// Return `true` when `word`'s alphanumeric core starts with an uppercase
/// character.
fn starts_uppercase(word: &str) -> bool {
    core_word(word)
        .chars()
        .next()
        .is_some_and(char::is_uppercase)
}

// ── question classification ──────────────────────────────────────────────────

/// Classify `question` by leftmost keyword-phrase match against
/// `type_keywords` (see [`ExtractiveQaConfig::type_keywords`]).
///
/// The question is tokenized on whitespace, and every token's alphanumeric
/// core is lowercased. Every `(question_type, phrases)` entry is checked at
/// every token position, left to right; the **leftmost** position at which
/// any phrase matches wins, with ties at the same position broken by the
/// order of `type_keywords`. Returns [`QuestionType::Other`] when nothing
/// matches (including for a blank question).
fn classify_question_impl(
    question: &str,
    type_keywords: &[(QuestionType, Vec<String>)],
) -> QuestionType {
    let tokens: Vec<String> = question.split_whitespace().map(core_lower).collect();
    let mut best: Option<(usize, QuestionType)> = None;

    for position in 0..tokens.len() {
        for (question_type, keywords) in type_keywords {
            for keyword in keywords {
                let keyword_tokens: Vec<String> =
                    keyword.split_whitespace().map(str::to_lowercase).collect();
                if keyword_tokens.is_empty() || position + keyword_tokens.len() > tokens.len() {
                    continue;
                }
                let window = &tokens[position..position + keyword_tokens.len()];
                if window == keyword_tokens.as_slice() {
                    let should_update = match best {
                        Some((best_position, _)) => position < best_position,
                        None => true,
                    };
                    if should_update {
                        best = Some((position, *question_type));
                    }
                }
            }
        }
    }

    best.map_or(QuestionType::Other, |(_, question_type)| question_type)
}

/// The distinct set of content words in `question` (see
/// [`is_content_word`]). Since interrogative words are themselves
/// stopwords, this naturally excludes them.
fn question_content_words(question: &str) -> HashSet<String> {
    content_word_set(question)
}

// ── candidate generation ─────────────────────────────────────────────────────

/// Generate every contiguous token window of `passage_tokens` up to
/// `max_span_tokens` tokens long.
///
/// Bounded to `O(n * max_span_tokens)` candidates rather than the
/// `O(n^2)` of considering every possible span (see
/// [`ExtractiveQaConfig::max_span_tokens`] for the trade-off). Returns an
/// empty vector when `passage_tokens` is empty or `max_span_tokens` is `0`
/// — zero-token spans are never generated.
fn generate_candidates_impl(passage_tokens: &[&str], max_span_tokens: usize) -> Vec<SpanCandidate> {
    let token_count = passage_tokens.len();
    let mut candidates = Vec::new();
    if max_span_tokens == 0 || token_count == 0 {
        return candidates;
    }

    for start in 0..token_count {
        let longest_here = max_span_tokens.min(token_count - start);
        for len in 1..=longest_here {
            let end = start + len;
            let text = passage_tokens[start..end].join(" ");
            candidates.push(SpanCandidate::new(start, end, text));
        }
    }
    candidates
}

// ── boundary scoring ──────────────────────────────────────────────────────────

/// `Who` content match: the fraction of the span's alphabetic tokens that
/// are capitalized. Spans with no alphabetic tokens (e.g. purely numeric)
/// score `0.0`.
#[allow(clippy::cast_precision_loss)]
fn capitalized_fraction(span_tokens: &[&str]) -> f32 {
    let alphabetic_count = span_tokens.iter().filter(|t| starts_alphabetic(t)).count();
    if alphabetic_count == 0 {
        return 0.0;
    }
    let capitalized_count = span_tokens.iter().filter(|t| starts_uppercase(t)).count();
    capitalized_count as f32 / alphabetic_count as f32
}

/// `When` content match: the fraction of the span's tokens that are
/// date-like (a plausible year or a month name).
#[allow(clippy::cast_precision_loss)]
fn date_like_fraction(span_tokens: &[&str]) -> f32 {
    if span_tokens.is_empty() {
        return 0.0;
    }
    let hits = span_tokens.iter().filter(|t| is_date_like(t)).count();
    hits as f32 / span_tokens.len() as f32
}

/// `HowMany`/`HowMuch` content match: the fraction of the span's tokens
/// that parse as a digit-form number.
#[allow(clippy::cast_precision_loss)]
fn numeric_fraction(span_tokens: &[&str]) -> f32 {
    if span_tokens.is_empty() {
        return 0.0;
    }
    let hits = span_tokens
        .iter()
        .filter(|t| numeric_value(t).is_some())
        .count();
    hits as f32 / span_tokens.len() as f32
}

/// `Where` content match: half capitalization signal (as [`Who`], reused
/// since place names are also typically capitalized), half
/// prepositional-phrase context signal (`1.0` when the token immediately
/// preceding the span is "in"/"at"/"near", else `0.0`). This lets a
/// lowercase span in a strong prepositional context ("near the river")
/// still score respectably, which pure capitalization (the `Who` heuristic)
/// would miss entirely.
///
/// [`Who`]: QuestionType::Who
fn where_score(span_tokens: &[&str], passage_tokens: &[&str], start: usize) -> f32 {
    let capitalization = capitalized_fraction(span_tokens);
    let preceded_by_preposition = start
        .checked_sub(1)
        .and_then(|prev_index| passage_tokens.get(prev_index))
        .map(|prev_token| core_lower(prev_token))
        .is_some_and(|prev_lower| LOCATION_PREPOSITIONS.contains(&prev_lower.as_str()));
    let preposition_bonus = if preceded_by_preposition { 1.0 } else { 0.0 };
    0.5 * capitalization + 0.5 * preposition_bonus
}

/// `What`/`Why`/`Which`/`Other` content match: lexical overlap between the
/// span's own content words and the question's content words, as the
/// fraction of the question's content words that the span echoes. `0.0`
/// when either set is empty.
#[allow(clippy::cast_precision_loss)]
fn lexical_overlap(span_tokens: &[&str], question_content: &HashSet<String>) -> f32 {
    if question_content.is_empty() {
        return 0.0;
    }
    let span_content: HashSet<String> = span_tokens
        .iter()
        .filter(|t| is_content_word(t))
        .map(|t| core_lower(t))
        .collect();
    if span_content.is_empty() {
        return 0.0;
    }
    let hits = question_content.intersection(&span_content).count();
    hits as f32 / question_content.len() as f32
}

/// Dispatch to the question-type-appropriate content-match heuristic.
fn type_match_score(
    question_type: QuestionType,
    span_tokens: &[&str],
    passage_tokens: &[&str],
    start: usize,
    question_content: &HashSet<String>,
) -> f32 {
    match question_type {
        QuestionType::Who => capitalized_fraction(span_tokens),
        QuestionType::When => date_like_fraction(span_tokens),
        QuestionType::HowMany | QuestionType::HowMuch => numeric_fraction(span_tokens),
        QuestionType::Where => where_score(span_tokens, passage_tokens, start),
        QuestionType::What | QuestionType::Why | QuestionType::Which | QuestionType::Other => {
            lexical_overlap(span_tokens, question_content)
        }
    }
}

/// Lexical context overlap: the fraction of the question's content words
/// that appear among the content words of the `window` tokens immediately
/// before `start` and immediately after `end` (both clamped to the passage
/// bounds). `0.0` when the question has no content words, or when the
/// surrounding context contributes no content words at all.
#[allow(clippy::cast_precision_loss)]
fn context_overlap_score(
    passage_tokens: &[&str],
    start: usize,
    end: usize,
    question_content: &HashSet<String>,
    window: usize,
) -> f32 {
    if question_content.is_empty() {
        return 0.0;
    }
    let before_start = start.saturating_sub(window);
    let after_end = end.saturating_add(window).min(passage_tokens.len());

    let mut context_content: HashSet<String> = HashSet::new();
    for token in &passage_tokens[before_start..start] {
        if is_content_word(token) {
            context_content.insert(core_lower(token));
        }
    }
    for token in &passage_tokens[end..after_end] {
        if is_content_word(token) {
            context_content.insert(core_lower(token));
        }
    }
    if context_content.is_empty() {
        return 0.0;
    }

    let hits = question_content.intersection(&context_content).count();
    hits as f32 / question_content.len() as f32
}

/// Length prior: `1 / (1 + decay * (len - 1))`. Equals `1.0` at a single
/// token and decreases smoothly (asymptotically, never reaching zero) as
/// `len` grows, mildly penalising long spans. Defensively returns `0.0` for
/// a degenerate zero-length span (never produced by
/// [`generate_candidates_impl`], but guarded here since [`compute_span_score`]
/// is also reachable from the caller-facing
/// [`ExtractiveQaEngine::score_span`] with a hand-built [`SpanCandidate`]).
#[allow(clippy::cast_precision_loss)]
fn length_prior_score(len: usize, decay: f32) -> f32 {
    if len == 0 {
        return 0.0;
    }
    1.0 / (1.0 + decay * (len - 1) as f32)
}

/// Compute the full [`SpanScore`] for one span, given its tokens (already
/// sliced from `passage_tokens[start..start + span_tokens.len()]`).
fn compute_span_score(
    span_tokens: &[&str],
    passage_tokens: &[&str],
    start: usize,
    question_type: QuestionType,
    question_content: &HashSet<String>,
    config: &ExtractiveQaConfig,
) -> SpanScore {
    let type_match = type_match_score(
        question_type,
        span_tokens,
        passage_tokens,
        start,
        question_content,
    );
    let end = start + span_tokens.len();
    let context_overlap = context_overlap_score(
        passage_tokens,
        start,
        end,
        question_content,
        config.context_window_tokens,
    );
    let length_prior = length_prior_score(span_tokens.len(), config.length_prior_decay);
    let total = config.weight_type_match * type_match
        + config.weight_context_overlap * context_overlap
        + config.weight_length_prior * length_prior;
    SpanScore {
        type_match,
        context_overlap,
        length_prior,
        total,
    }
}

// ── ExtractiveQaEngine ────────────────────────────────────────────────────────

/// `SQuAD`-style extractive question answering: classifies a question's
/// [`QuestionType`], generates bounded candidate answer spans from a
/// passage, scores each by a question-type-aware blend of signals, and
/// returns the best (or the top `k`).
///
/// See the [module documentation](crate::extractive_qa) for the algorithm
/// overview and how this differs from `quote_grounding` and `self_ask`.
#[derive(Debug, Clone, Default)]
pub struct ExtractiveQaEngine {
    /// Configuration for this engine.
    pub config: ExtractiveQaConfig,
}

impl ExtractiveQaEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: ExtractiveQaConfig) -> Self {
        Self { config }
    }

    /// Classify `question`'s [`QuestionType`] using
    /// [`ExtractiveQaConfig::type_keywords`]. Infallible: an empty or
    /// unrecognised question simply classifies as [`QuestionType::Other`].
    #[must_use]
    pub fn classify_question(&self, question: &str) -> QuestionType {
        classify_question_impl(question, &self.config.type_keywords)
    }

    /// Generate every candidate answer span in `passage`, up to
    /// [`ExtractiveQaConfig::max_span_tokens`] tokens long.
    ///
    /// This is the raw, unscored candidate set (see
    /// [`ExtractiveQaEngine::score_span`] to score one). An empty result is
    /// not an error at this layer — it honestly reflects, for example, a
    /// configured `max_span_tokens` of `0` — but [`ExtractiveQaEngine::answer`]
    /// and [`ExtractiveQaEngine::top_k_spans`] do treat an empty candidate
    /// set as [`ExtractiveQaError::NoViableCandidates`], since answering
    /// requires at least one candidate.
    ///
    /// # Errors
    ///
    /// Returns [`ExtractiveQaError::EmptyPassage`] when `passage` is blank
    /// after trimming.
    pub fn generate_candidates(&self, passage: &str) -> ExtractiveQaResult<Vec<SpanCandidate>> {
        if passage.trim().is_empty() {
            return Err(ExtractiveQaError::EmptyPassage);
        }
        let passage_tokens: Vec<&str> = passage.split_whitespace().collect();
        Ok(generate_candidates_impl(
            &passage_tokens,
            self.config.max_span_tokens,
        ))
    }

    /// Score a single `candidate` against `question` and `passage`.
    ///
    /// `candidate`'s tokens are re-derived from `passage` using its
    /// `start_token`/`end_token` (not from `candidate.text`), so `candidate`
    /// must be index-consistent with `passage` — typically because it came
    /// from `self.generate_candidates(passage)` on this very passage.
    ///
    /// # Errors
    ///
    /// - [`ExtractiveQaError::EmptyQuestion`] if `question` is blank.
    /// - [`ExtractiveQaError::EmptyPassage`] if `passage` is blank.
    /// - [`ExtractiveQaError::InvalidSpanBounds`] if `candidate`'s bounds
    ///   are not a valid, non-empty range within `passage`'s tokens.
    pub fn score_span(
        &self,
        question: &str,
        passage: &str,
        candidate: &SpanCandidate,
    ) -> ExtractiveQaResult<SpanScore> {
        if question.trim().is_empty() {
            return Err(ExtractiveQaError::EmptyQuestion);
        }
        if passage.trim().is_empty() {
            return Err(ExtractiveQaError::EmptyPassage);
        }
        let passage_tokens: Vec<&str> = passage.split_whitespace().collect();
        // `<[T]>::get` on a `Range` returns `None` when `end > len` or when
        // `start > end`; the extra `start < end` guard additionally rejects
        // a well-formed but empty (zero-length) range. Together these avoid
        // ever risking the panic that direct `passage_tokens[start..end]`
        // indexing could raise on a caller-supplied (not internally
        // generated) candidate.
        let span_tokens = match passage_tokens.get(candidate.start_token..candidate.end_token) {
            Some(tokens) if candidate.start_token < candidate.end_token => tokens,
            _ => {
                return Err(ExtractiveQaError::InvalidSpanBounds {
                    start_token: candidate.start_token,
                    end_token: candidate.end_token,
                    passage_tokens: passage_tokens.len(),
                });
            }
        };

        let question_type = self.classify_question(question);
        let question_content = question_content_words(question);
        Ok(compute_span_score(
            span_tokens,
            &passage_tokens,
            candidate.start_token,
            question_type,
            &question_content,
            &self.config,
        ))
    }

    /// Validate `question`/`passage`, classify the question, generate every
    /// candidate, and score them all — the shared core of
    /// [`ExtractiveQaEngine::answer`] and [`ExtractiveQaEngine::top_k_spans`].
    fn score_all(
        &self,
        question: &str,
        passage: &str,
    ) -> ExtractiveQaResult<(QuestionType, Vec<(SpanCandidate, SpanScore)>)> {
        if question.trim().is_empty() {
            return Err(ExtractiveQaError::EmptyQuestion);
        }
        if passage.trim().is_empty() {
            return Err(ExtractiveQaError::EmptyPassage);
        }

        let question_type = self.classify_question(question);
        let question_content = question_content_words(question);
        let passage_tokens: Vec<&str> = passage.split_whitespace().collect();
        let candidates = generate_candidates_impl(&passage_tokens, self.config.max_span_tokens);
        if candidates.is_empty() {
            return Err(ExtractiveQaError::NoViableCandidates {
                max_span_tokens: self.config.max_span_tokens,
                passage_tokens: passage_tokens.len(),
            });
        }

        let scored = candidates
            .into_iter()
            .map(|candidate| {
                let span_tokens = &passage_tokens[candidate.start_token..candidate.end_token];
                let score = compute_span_score(
                    span_tokens,
                    &passage_tokens,
                    candidate.start_token,
                    question_type,
                    &question_content,
                    &self.config,
                );
                (candidate, score)
            })
            .collect();

        Ok((question_type, scored))
    }

    /// Classify `question`, generate and score every candidate span in
    /// `passage`, and return the single highest-scoring [`AnswerSpan`].
    ///
    /// Ties in `SpanScore::total` are broken deterministically in favour of
    /// the candidate generated first (ascending start token, then ascending
    /// length) — the same span every time for the same inputs.
    ///
    /// # Errors
    ///
    /// - [`ExtractiveQaError::EmptyQuestion`] if `question` is blank.
    /// - [`ExtractiveQaError::EmptyPassage`] if `passage` is blank.
    /// - [`ExtractiveQaError::NoViableCandidates`] if no candidate spans
    ///   could be generated (for example
    ///   [`ExtractiveQaConfig::max_span_tokens`] is `0`).
    pub fn answer(&self, question: &str, passage: &str) -> ExtractiveQaResult<AnswerSpan> {
        let (question_type, scored) = self.score_all(question, passage)?;

        let mut best: Option<(SpanCandidate, SpanScore)> = None;
        for (candidate, score) in scored {
            let take_it = match &best {
                None => true,
                Some((_, best_score)) => score.total > best_score.total,
            };
            if take_it {
                best = Some((candidate, score));
            }
        }

        match best {
            Some((candidate, score)) => Ok(AnswerSpan {
                text: candidate.text,
                start_token: candidate.start_token,
                end_token: candidate.end_token,
                question_type,
                score,
            }),
            // Unreachable in practice: `score_all` already returns
            // `NoViableCandidates` for an empty candidate set, so `scored`
            // is always non-empty here and the loop above always sets
            // `best`. Handled honestly (not via `unreachable!()`) rather
            // than fabricating a placeholder answer.
            None => Err(ExtractiveQaError::NoViableCandidates {
                max_span_tokens: self.config.max_span_tokens,
                passage_tokens: passage.split_whitespace().count(),
            }),
        }
    }

    /// Classify `question`, generate and score every candidate span in
    /// `passage`, and return up to `k` [`AnswerSpan`]s in descending score
    /// order.
    ///
    /// Every returned span has a distinct `(start_token, end_token)` pair
    /// (candidate generation never produces the same span twice). Ties in
    /// `SpanScore::total` are broken by ascending start token, then
    /// ascending end token. `k == 0` returns an empty vector; `k` larger
    /// than the number of candidates returns all of them.
    ///
    /// # Errors
    ///
    /// Same as [`ExtractiveQaEngine::answer`].
    pub fn top_k_spans(
        &self,
        question: &str,
        passage: &str,
        k: usize,
    ) -> ExtractiveQaResult<Vec<AnswerSpan>> {
        let (question_type, mut scored) = self.score_all(question, passage)?;

        scored.sort_by(|(candidate_a, score_a), (candidate_b, score_b)| {
            score_b
                .total
                .partial_cmp(&score_a.total)
                .unwrap_or(Ordering::Equal)
                .then_with(|| candidate_a.start_token.cmp(&candidate_b.start_token))
                .then_with(|| candidate_a.end_token.cmp(&candidate_b.end_token))
        });

        Ok(scored
            .into_iter()
            .take(k)
            .map(|(candidate, score)| AnswerSpan {
                text: candidate.text,
                start_token: candidate.start_token,
                end_token: candidate.end_token,
                question_type,
                score,
            })
            .collect())
    }
}
