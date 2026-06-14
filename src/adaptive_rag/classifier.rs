//! Heuristic query-complexity classifier.
//!
//! The classifier accumulates a *signed* weighted signal score from a query and
//! normalises it into `[0.0, 1.0]`. Positive signals (multi-hop cue words,
//! multiple entities, conjunctions, multiple question marks, comparatives) raise
//! the score toward [`QueryComplexity::MultiStep`]; negative signals (trivial
//! closed openers, very short queries) lower it toward
//! [`QueryComplexity::Straightforward`]. The result is mapped to a tier via the
//! thresholds in [`AdaptiveRagConfig`].
//!
//! Classification is fully deterministic: the same query always yields the same
//! score, signals, and tier.

use super::types::{
    AdaptiveRagConfig, AdaptiveRagError, ComplexityClassification, ComplexitySignal,
    QueryComplexity,
};

// ── Signal weights ────────────────────────────────────────────────────────────

/// Weight subtracted when the query opens with a trivial closed-question word.
const W_TRIVIAL_OPENER: f32 = -0.30;
/// Weight subtracted when the query is very short and not a wh-question.
const W_VERY_SHORT: f32 = -0.15;
/// Weight added for a single wh-word paired with exactly one entity token.
const W_WH_SINGLE_ENTITY: f32 = 0.40;
/// Weight added per distinct multi-hop cue word present.
const W_MULTI_HOP_CUE: f32 = 0.30;
/// Weight added when a clause-joining conjunction (`and` / `or`) is present.
const W_CONJUNCTION: f32 = 0.15;
/// Weight added when two or more capitalised entity tokens are present.
const W_MULTI_ENTITY: f32 = 0.25;
/// Weight added when more than one question mark is present.
const W_MULTI_QUESTION: f32 = 0.25;
/// Weight added when a superlative / comparative marker is present.
const W_COMPARATIVE: f32 = 0.15;
/// Weight added for a long query (raises complexity mildly).
const W_LONG_QUERY: f32 = 0.10;

/// Token count at or below which a (non-wh) query is "very short".
const VERY_SHORT_TOKENS: usize = 4;
/// Token count at or above which a query is "long".
const LONG_TOKENS: usize = 14;

/// Multi-hop cue words that signal multi-step reasoning.
const MULTI_HOP_CUES: &[&str] = &[
    "compare",
    "comparison",
    "difference",
    "versus",
    "vs",
    "both",
    "after",
    "before",
    "then",
    "cause",
    "because",
    "while",
    "whereas",
];

/// Clause-joining conjunctions.
const CONJUNCTIONS: &[&str] = &["and", "or"];

/// Whole-word superlative / comparative markers (suffix forms handled
/// separately).
const COMPARATIVE_WORDS: &[&str] = &["most", "least", "more", "fewer", "than"];

/// Wh-words that introduce a factoid question.
const WH_WORDS: &[&str] = &["who", "what", "when", "where", "which", "whom", "whose"];

/// Trivial closed-question openers that suggest a non-retrieval answer.
const TRIVIAL_OPENERS: &[&str] = &["is", "are", "was", "were", "does", "do", "did"];

// ── ComplexityClassifier ──────────────────────────────────────────────────────

/// Heuristic, deterministic query-complexity classifier.
///
/// Produces a [`ComplexityClassification`] carrying the resolved
/// [`QueryComplexity`] tier, a normalised score in `[0.0, 1.0]`, and the list of
/// [`ComplexitySignal`]s that fired.
#[derive(Debug, Clone)]
pub struct ComplexityClassifier {
    /// The configuration governing tier thresholds.
    pub config: AdaptiveRagConfig,
}

impl ComplexityClassifier {
    /// Creates a new classifier with the given configuration.
    #[must_use]
    pub fn new(config: AdaptiveRagConfig) -> Self {
        Self { config }
    }

    /// Classifies the complexity of `query`.
    ///
    /// # Errors
    ///
    /// Returns [`AdaptiveRagError::EmptyQuery`] if `query` is empty or contains
    /// only whitespace.
    pub fn classify(&self, query: &str) -> Result<ComplexityClassification, AdaptiveRagError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(AdaptiveRagError::EmptyQuery);
        }

        let lower = trimmed.to_lowercase();
        let tokens = tokenize(&lower);
        let token_count = tokens.len();
        let first_token = tokens.first().map(String::as_str);
        let is_wh = first_token.is_some_and(|t| WH_WORDS.contains(&t));
        let entity_count = count_entities(trimmed);
        let question_marks = trimmed.chars().filter(|c| *c == '?').count();

        let mut signals: Vec<ComplexitySignal> = Vec::new();
        let mut score = 0.0_f32;

        // ── Trivial closed opener (lowers) ────────────────────────────────────
        if first_token.is_some_and(|t| TRIVIAL_OPENERS.contains(&t))
            && token_count <= VERY_SHORT_TOKENS
        {
            score += W_TRIVIAL_OPENER;
            signals.push(ComplexitySignal::new("trivial_opener", W_TRIVIAL_OPENER));
        }

        // ── Very short, non-wh query (lowers) ─────────────────────────────────
        if !is_wh && token_count <= VERY_SHORT_TOKENS {
            score += W_VERY_SHORT;
            signals.push(ComplexitySignal::new("very_short", W_VERY_SHORT));
        }

        // ── Multi-hop cue words (raise, per distinct cue) ─────────────────────
        for cue in MULTI_HOP_CUES {
            if tokens.iter().any(|t| t == cue) {
                score += W_MULTI_HOP_CUE;
                signals.push(ComplexitySignal::new(
                    format!("multi_hop_cue:{cue}"),
                    W_MULTI_HOP_CUE,
                ));
            }
        }

        // ── Clause-joining conjunction (raises) ───────────────────────────────
        if tokens.iter().any(|t| CONJUNCTIONS.contains(&t.as_str())) {
            score += W_CONJUNCTION;
            signals.push(ComplexitySignal::new("conjunction", W_CONJUNCTION));
        }

        // ── Two or more entity tokens (raises toward MultiStep) ───────────────
        if entity_count >= 2 {
            score += W_MULTI_ENTITY;
            signals.push(ComplexitySignal::new("multi_entity", W_MULTI_ENTITY));
        }

        // ── Multiple question marks (raises) ──────────────────────────────────
        if question_marks > 1 {
            score += W_MULTI_QUESTION;
            signals.push(ComplexitySignal::new("multi_question", W_MULTI_QUESTION));
        }

        // ── Superlatives / comparatives (raise) ───────────────────────────────
        if has_comparative(&tokens) {
            score += W_COMPARATIVE;
            signals.push(ComplexitySignal::new("comparative", W_COMPARATIVE));
        }

        // ── Long query (raises mildly) ────────────────────────────────────────
        if token_count >= LONG_TOKENS {
            score += W_LONG_QUERY;
            signals.push(ComplexitySignal::new("long_query", W_LONG_QUERY));
        }

        // ── Single wh-word with exactly one entity ⇒ single-step factoid ──────
        if is_wh && entity_count == 1 {
            score += W_WH_SINGLE_ENTITY;
            signals.push(ComplexitySignal::new(
                "wh_single_entity",
                W_WH_SINGLE_ENTITY,
            ));
        }

        let normalized = score.clamp(0.0, 1.0);
        let complexity = self.config.tier_for_score(normalized);

        Ok(ComplexityClassification {
            complexity,
            score: normalized,
            signals,
        })
    }

    /// Convenience wrapper returning only the resolved [`QueryComplexity`].
    ///
    /// # Errors
    ///
    /// Returns [`AdaptiveRagError::EmptyQuery`] if `query` is empty.
    pub fn classify_tier(&self, query: &str) -> Result<QueryComplexity, AdaptiveRagError> {
        Ok(self.classify(query)?.complexity)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Splits `text` into lowercase alphanumeric tokens.
///
/// Single-character tokens are retained so that token *counts* remain faithful
/// to the original query length.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Counts capitalised entity tokens in the *original-case* `text`.
///
/// The first token of the query is excluded (sentence-initial capitalisation is
/// not evidence of a proper noun), as are purely numeric tokens. A token counts
/// as an entity if its first character is an uppercase letter.
fn count_entities(text: &str) -> usize {
    let raw_tokens: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    raw_tokens
        .iter()
        .enumerate()
        .filter(|(index, token)| {
            if *index == 0 {
                return false;
            }
            token
                .chars()
                .next()
                .is_some_and(|c| c.is_uppercase() && c.is_alphabetic())
        })
        .count()
}

/// Returns `true` if any token is a comparative/superlative marker, either a
/// whole word in [`COMPARATIVE_WORDS`] or an `-est` superlative suffix.
fn has_comparative(tokens: &[String]) -> bool {
    tokens.iter().any(|token| {
        COMPARATIVE_WORDS.contains(&token.as_str())
            || (token.len() >= 4
                && token.ends_with("est")
                && token.chars().all(char::is_alphabetic))
    })
}
