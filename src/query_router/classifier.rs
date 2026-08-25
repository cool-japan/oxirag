//! Intent classifier trait and implementations.
//!
//! The primary production classifier is [`HeuristicIntentClassifier`], which
//! uses keyword/pattern matching to score each [`QueryIntent`]. A
//! [`MockIntentClassifier`] is provided for deterministic unit testing.

use async_trait::async_trait;

use super::types::{IntentScores, QueryIntent, QueryRouterError};

// ── Helper ────────────────────────────────────────────────────────────────────

/// Returns `true` if `needle` appears at a word boundary within `haystack`.
///
/// A word boundary is defined as either the start/end of the string or a
/// non-alphanumeric character adjacent to the needle.
fn at_word_boundary(haystack: &str, needle: &str) -> bool {
    let hay_bytes = haystack.as_bytes();
    let needle_len = needle.len();
    let hay_len = haystack.len();

    if needle_len > hay_len {
        return false;
    }

    let mut start = 0_usize;
    while start + needle_len <= hay_len {
        if let Some(pos) = haystack[start..].find(needle) {
            let abs = start + pos;
            let left_ok = abs == 0
                || !hay_bytes
                    .get(abs - 1)
                    .is_some_and(u8::is_ascii_alphanumeric);
            let right_ok = abs + needle_len >= hay_len
                || !hay_bytes
                    .get(abs + needle_len)
                    .is_some_and(u8::is_ascii_alphanumeric);
            if left_ok && right_ok {
                return true;
            }
            start = abs + 1;
        } else {
            break;
        }
    }
    false
}

/// Returns `true` if `haystack` contains a run of exactly four ASCII digits
/// (a year-like token).
fn contains_year_like(haystack: &str) -> bool {
    let bytes = haystack.as_bytes();
    if bytes.len() < 4 {
        return false;
    }
    for window in bytes.windows(4) {
        if window.iter().all(u8::is_ascii_digit) {
            // Ensure it is not part of a longer digit run (5+ digits)
            let start_ok = true; // windows already checks exactly 4
            let _ = start_ok;
            return true;
        }
    }
    false
}

// ── Signal tables ─────────────────────────────────────────────────────────────

/// Keyword signals for [`QueryIntent::Definitional`].
const DEFINITIONAL_SIGNALS: &[&str] = &[
    "what is ",
    "what are ",
    "define ",
    "definition of ",
    "meaning of ",
    "explain ",
];

/// Keyword signals for [`QueryIntent::Comparative`].
const COMPARATIVE_SIGNALS: &[&str] = &[
    " vs ",
    " versus ",
    "compare ",
    "difference between ",
    "better than ",
    "vs.",
    " compared to ",
];

/// Keyword signals for [`QueryIntent::Temporal`].
const TEMPORAL_SIGNALS: &[&str] = &[
    "when ",
    "before ",
    "after ",
    "history of ",
    "timeline ",
    "since ",
    "until ",
    "date of ",
];

/// Keyword signals for [`QueryIntent::Aggregation`].
const AGGREGATION_SIGNALS: &[&str] = &[
    "how many ",
    "count of ",
    "total ",
    "sum of ",
    "average ",
    "list all ",
    "number of ",
];

/// Keyword signals for [`QueryIntent::MultiHop`].
const MULTI_HOP_SIGNALS: &[&str] = &[" and then ", "both ", " that ", " which "];

/// Keyword signals for [`QueryIntent::Navigational`].
const NAVIGATIONAL_SIGNALS: &[&str] = &[
    "go to ",
    "open ",
    "show me the ",
    "navigate to ",
    "find the page",
];

/// Keyword signals for [`QueryIntent::Exploratory`].
const EXPLORATORY_SIGNALS: &[&str] = &[
    "tell me about ",
    "overview of ",
    "explain ",
    "learn about ",
    "introduction to ",
];

/// Wh-prefixes that indicate a [`QueryIntent::Factual`] question.
const FACTUAL_WH_PREFIXES: &[&str] = &["who ", "where ", "which ", "what ", "why ", "how "];

/// Pronouns that suggest a [`QueryIntent::Conversational`] (follow-up) query.
const CONVERSATIONAL_PRONOUNS: &[&str] = &["it ", "this ", "that ", "they ", "he ", "she ", "we "];

/// Non-whitespace character threshold below which a query is considered
/// conversational / short.
const CONVERSATIONAL_SHORT_THRESHOLD: usize = 30;

/// Floor score always injected for [`QueryIntent::Unknown`].
const UNKNOWN_FLOOR: f32 = 0.05;

// ── IntentClassifier trait ────────────────────────────────────────────────────

/// Asynchronous intent classifier interface.
///
/// Implementors analyse a raw query string and return a scored
/// [`IntentScores`] distribution over all [`QueryIntent`] variants.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait IntentClassifier: Send + Sync {
    /// Classifies `query` and returns a scored distribution.
    async fn classify(&self, query: &str) -> Result<IntentScores, QueryRouterError>;

    /// Convenience wrapper that returns only the top-scoring `(intent, score)`.
    async fn classify_top(&self, query: &str) -> Result<(QueryIntent, f32), QueryRouterError> {
        Ok(self.classify(query).await?.top())
    }
}

// ── HeuristicIntentClassifier ─────────────────────────────────────────────────

/// Keyword and pattern-based heuristic intent classifier.
///
/// Uses pre-defined signal tables (substring matching in lowercase) to produce
/// a confidence score for each [`QueryIntent`]. Custom keywords can be injected
/// via the builder method [`HeuristicIntentClassifier::with_keyword`].
///
/// The classifier always emits an [`IntentScores`] that is non-empty: a floor
/// entry of `(Unknown, 0.05)` is always present.
///
/// # Scoring algorithm
///
/// For each intent, the number of matched signals is counted and the raw count
/// is normalised to `[0.0, 1.0]` (capped at `1.0`). The normalisation
/// denominator is the number of signals in that intent's table, so matching
/// all signals produces a score of exactly `1.0`.
pub struct HeuristicIntentClassifier {
    /// Additional user-supplied keyword → intent mappings.
    extra_keywords: Vec<(QueryIntent, String)>,
}

impl HeuristicIntentClassifier {
    /// Creates a new classifier with the default signal tables.
    #[must_use]
    pub fn new() -> Self {
        Self {
            extra_keywords: Vec::new(),
        }
    }

    /// Adds a custom keyword signal for the given intent.
    ///
    /// The keyword is matched as a plain substring of the lowercased query.
    #[must_use]
    pub fn with_keyword(mut self, intent: QueryIntent, keyword: impl Into<String>) -> Self {
        self.extra_keywords.push((intent, keyword.into()));
        self
    }

    /// Scores the match quality of `signals` against `lower`.
    ///
    /// Scoring uses a two-part formula:
    /// - `base_score` = `0.5` if any signal matches, scaled linearly up to `1.0` as
    ///   the fraction of matching signals approaches `1.0`.
    ///
    /// This ensures that even a single matched signal yields a score ≥ `0.5`,
    /// which is high enough to outcompete the Conversational short-query heuristic
    /// (capped at `0.45`) when a specific intent is detected.
    #[allow(clippy::cast_precision_loss)]
    // Safety: signal arrays are tiny (<10 entries); usize fits in f32 without
    // precision loss for these small counts.
    fn count_signals(lower: &str, signals: &[&str]) -> f32 {
        if signals.is_empty() {
            return 0.0;
        }
        let hits = signals.iter().filter(|s| lower.contains(*s)).count();
        if hits == 0 {
            return 0.0;
        }
        // Base 0.5 for any match; scale remaining 0.5 by fraction of matches.
        let fraction = hits as f32 / signals.len() as f32;
        (0.5_f32 + fraction * 0.5_f32).min(1.0)
    }

    /// Counts extra keyword hits for `intent` in the lowercased `lower` query.
    ///
    /// Returns `0.0` if no extra keywords are registered for this intent.
    #[allow(clippy::cast_precision_loss)]
    // Safety: extra_keywords is user-supplied and typically small; the count is
    // bounded and will not overflow f32 meaningfully.
    fn count_extra(&self, lower: &str, intent: QueryIntent) -> f32 {
        let relevant: Vec<&str> = self
            .extra_keywords
            .iter()
            .filter(|(i, _)| *i == intent)
            .map(|(_, kw)| kw.as_str())
            .collect();
        if relevant.is_empty() {
            return 0.0;
        }
        let hits = relevant.iter().filter(|kw| lower.contains(*kw)).count();
        if hits == 0 {
            return 0.0;
        }
        let fraction = hits as f32 / relevant.len() as f32;
        (0.5_f32 + fraction * 0.5_f32).min(1.0)
    }

    /// Combines built-in signal score and extra keyword score, capping at `1.0`.
    fn score_intent(&self, lower: &str, intent: QueryIntent, signals: &[&str]) -> f32 {
        let base = Self::count_signals(lower, signals);
        let extra = self.count_extra(lower, intent);
        (base + extra).min(1.0)
    }
}

impl Default for HeuristicIntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl IntentClassifier for HeuristicIntentClassifier {
    /// Classifies the query using keyword/pattern heuristics.
    ///
    /// Returns [`QueryRouterError::EmptyQuery`] if `query` is blank.
    async fn classify(&self, query: &str) -> Result<IntentScores, QueryRouterError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(QueryRouterError::EmptyQuery);
        }

        let lower = trimmed.to_lowercase();
        let mut raw: Vec<(QueryIntent, f32)> = Vec::new();

        // ── Definitional ──────────────────────────────────────────────────────
        let definitional =
            self.score_intent(&lower, QueryIntent::Definitional, DEFINITIONAL_SIGNALS);
        if definitional > 0.0 {
            raw.push((QueryIntent::Definitional, definitional));
        }

        // ── Comparative ───────────────────────────────────────────────────────
        let comparative = self.score_intent(&lower, QueryIntent::Comparative, COMPARATIVE_SIGNALS);
        if comparative > 0.0 {
            raw.push((QueryIntent::Comparative, comparative));
        }

        // ── Temporal ──────────────────────────────────────────────────────────
        let temporal_base = self.score_intent(&lower, QueryIntent::Temporal, TEMPORAL_SIGNALS);
        let year_bonus = if contains_year_like(&lower) {
            0.3_f32
        } else {
            0.0_f32
        };
        let temporal = (temporal_base + year_bonus).min(1.0);
        if temporal > 0.0 {
            raw.push((QueryIntent::Temporal, temporal));
        }

        // ── Aggregation ───────────────────────────────────────────────────────
        let aggregation = self.score_intent(&lower, QueryIntent::Aggregation, AGGREGATION_SIGNALS);
        if aggregation > 0.0 {
            raw.push((QueryIntent::Aggregation, aggregation));
        }

        // ── MultiHop ──────────────────────────────────────────────────────────
        let multi_hop_kw = self.score_intent(&lower, QueryIntent::MultiHop, MULTI_HOP_SIGNALS);
        let multi_question_bonus = if lower.chars().filter(|c| *c == '?').count() > 1 {
            0.4_f32
        } else {
            0.0_f32
        };
        let multi_hop = (multi_hop_kw + multi_question_bonus).min(1.0);
        if multi_hop > 0.0 {
            raw.push((QueryIntent::MultiHop, multi_hop));
        }

        // ── Navigational ──────────────────────────────────────────────────────
        let navigational =
            self.score_intent(&lower, QueryIntent::Navigational, NAVIGATIONAL_SIGNALS);
        if navigational > 0.0 {
            raw.push((QueryIntent::Navigational, navigational));
        }

        // ── Conversational ────────────────────────────────────────────────────
        // Conversational scores are deliberately kept below 0.5 so that any
        // specific intent that fires (score ≥ 0.5 by the count_signals formula)
        // will always outrank the conversational heuristic.
        let non_ws_len = lower.chars().filter(|c| !c.is_whitespace()).count();
        let is_short = non_ws_len < CONVERSATIONAL_SHORT_THRESHOLD;
        let starts_with_pronoun = CONVERSATIONAL_PRONOUNS
            .iter()
            .any(|p| at_word_boundary(&lower, p.trim()) && lower.starts_with(p));
        let conversational_score = if starts_with_pronoun {
            // Pronoun-start is a stronger conversational signal.
            0.45_f32
        } else if is_short {
            // Short query is a weaker signal — loses to any specific keyword match.
            0.35_f32
        } else {
            self.count_extra(&lower, QueryIntent::Conversational)
        };
        if conversational_score > 0.0 {
            raw.push((QueryIntent::Conversational, conversational_score));
        }

        // ── Exploratory ───────────────────────────────────────────────────────
        let exploratory = self.score_intent(&lower, QueryIntent::Exploratory, EXPLORATORY_SIGNALS);
        if exploratory > 0.0 {
            raw.push((QueryIntent::Exploratory, exploratory));
        }

        // ── Factual (wh-questions not matched above) ──────────────────────────
        let already_classified = !raw.is_empty();
        let is_wh = FACTUAL_WH_PREFIXES
            .iter()
            .any(|prefix| lower.starts_with(prefix));
        if is_wh && !already_classified {
            let factual_extra = self.count_extra(&lower, QueryIntent::Factual);
            raw.push((QueryIntent::Factual, (0.5_f32 + factual_extra).min(1.0)));
        } else if is_wh {
            // Wh-question that was also matched above — add a secondary factual signal
            let factual_extra = self.count_extra(&lower, QueryIntent::Factual);
            let score = (0.3_f32 + factual_extra).min(1.0);
            raw.push((QueryIntent::Factual, score));
        } else {
            let factual_extra = self.count_extra(&lower, QueryIntent::Factual);
            if factual_extra > 0.0 {
                raw.push((QueryIntent::Factual, factual_extra));
            }
        }

        // ── Unknown floor ─────────────────────────────────────────────────────
        raw.push((QueryIntent::Unknown, UNKNOWN_FLOOR));

        Ok(IntentScores::new(raw))
    }
}

// ── MockIntentClassifier ──────────────────────────────────────────────────────

/// A scripted intent classifier for deterministic unit testing.
///
/// Returns a fixed [`IntentScores`] regardless of the query text. Use
/// [`MockIntentClassifier::new`] for a single-intent response or
/// [`MockIntentClassifier::with_scores`] for a fully scripted distribution.
pub struct MockIntentClassifier {
    /// The scripted scores returned for every query.
    scripted: IntentScores,
}

impl MockIntentClassifier {
    /// Creates a classifier that always returns `intent` with `confidence`.
    ///
    /// An `Unknown` floor entry is automatically appended so the result is
    /// always non-empty.
    #[must_use]
    pub fn new(intent: QueryIntent, confidence: f32) -> Self {
        let scores = vec![(intent, confidence), (QueryIntent::Unknown, UNKNOWN_FLOOR)];
        Self {
            scripted: IntentScores::new(scores),
        }
    }

    /// Creates a classifier that returns exactly the provided [`IntentScores`].
    #[must_use]
    pub fn with_scores(scores: IntentScores) -> Self {
        Self { scripted: scores }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl IntentClassifier for MockIntentClassifier {
    /// Returns the scripted [`IntentScores`] without inspecting `query`.
    async fn classify(&self, _query: &str) -> Result<IntentScores, QueryRouterError> {
        Ok(self.scripted.clone())
    }
}
