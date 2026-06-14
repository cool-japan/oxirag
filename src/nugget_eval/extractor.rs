//! Nugget extraction from reference answers.
use crate::nugget_eval::types::{Nugget, NuggetImportance};

// ── tokenization helpers ──────────────────────────────────────────────────────

/// Split `text` into lowercase tokens of length `>= 2`, breaking on any
/// non-alphanumeric character.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Split a reference answer into raw clause strings.
///
/// Clauses are delimited by sentence and intra-sentence punctuation
/// (`.`, `!`, `?`, `;`, `,`) as well as the coordinating conjunction `" and "`.
/// Empty / whitespace-only fragments are dropped.
pub(crate) fn split_clauses(reference: &str) -> Vec<String> {
    // First split on the punctuation set, then further split each fragment on
    // the literal coordinating conjunction " and " (case-insensitive).
    let mut clauses = Vec::new();
    for raw in reference.split(['.', '!', '?', ';', ',']) {
        for piece in split_on_and(raw) {
            let trimmed = piece.trim();
            if !trimmed.is_empty() {
                clauses.push(trimmed.to_string());
            }
        }
    }
    clauses
}

/// Split a fragment on the coordinating conjunction `" and "` (case-insensitive),
/// preserving the original casing of the surrounding text.
fn split_on_and(fragment: &str) -> Vec<String> {
    let lower = fragment.to_lowercase();
    let needle = " and ";
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut search_from = 0usize;
    while let Some(rel) = lower[search_from..].find(needle) {
        let idx = search_from + rel;
        parts.push(fragment[start..idx].to_string());
        start = idx + needle.len();
        search_from = start;
    }
    parts.push(fragment[start..].to_string());
    parts
}

// ── NuggetExtractor ───────────────────────────────────────────────────────────

/// Decomposes a reference answer into weighted [`Nugget`]s.
pub trait NuggetExtractor {
    /// Decompose a reference answer into weighted nuggets.
    fn extract(&self, reference: &str) -> Vec<Nugget>;
}

// ── HeuristicNuggetExtractor ──────────────────────────────────────────────────

/// Heuristic, dependency-free implementation of [`NuggetExtractor`].
///
/// The reference answer is split into clauses on sentence and intra-sentence
/// punctuation plus the coordinating conjunction `" and "`. A clause becomes a
/// nugget only if it contains at least `min_nugget_tokens` tokens; the
/// `min_nugget_tokens` value is supplied through
/// [`HeuristicNuggetExtractor::with_min_tokens`]. A nugget is graded
/// [`NuggetImportance::Vital`] when its source clause contains a named entity
/// (a capitalized token) or a numeric token, otherwise [`NuggetImportance::Okay`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeuristicNuggetExtractor {
    /// Minimum token count a clause must have to become a nugget.
    min_nugget_tokens: usize,
}

impl Default for HeuristicNuggetExtractor {
    fn default() -> Self {
        Self {
            min_nugget_tokens: 2,
        }
    }
}

impl HeuristicNuggetExtractor {
    /// Create a new extractor with the default minimum token count (`2`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an extractor that discards clauses with fewer than `min_tokens`
    /// tokens.
    #[must_use]
    pub fn with_min_tokens(min_tokens: usize) -> Self {
        Self {
            min_nugget_tokens: min_tokens,
        }
    }

    /// The minimum token count a clause must have to become a nugget.
    #[must_use]
    pub fn min_nugget_tokens(&self) -> usize {
        self.min_nugget_tokens
    }

    /// Classify a clause as [`NuggetImportance::Vital`] when it contains a
    /// capitalized token (entity) or a numeric token, else
    /// [`NuggetImportance::Okay`].
    fn classify(clause: &str) -> NuggetImportance {
        let has_entity_or_number = clause
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .any(|tok| {
                let has_digit = tok.chars().any(|c| c.is_ascii_digit());
                let is_capitalized = tok.chars().next().is_some_and(char::is_uppercase);
                has_digit || is_capitalized
            });
        if has_entity_or_number {
            NuggetImportance::Vital
        } else {
            NuggetImportance::Okay
        }
    }
}

impl NuggetExtractor for HeuristicNuggetExtractor {
    fn extract(&self, reference: &str) -> Vec<Nugget> {
        split_clauses(reference)
            .into_iter()
            .filter(|clause| tokenize(clause).len() >= self.min_nugget_tokens)
            .map(|clause| {
                let importance = Self::classify(&clause);
                Nugget::new(clause, importance)
            })
            .collect()
    }
}
