//! Conflict detection between retrieved passages.
//!
//! The detector walks every pair of passages, splits each into sentences, and
//! flags any sentence pair that describes the same subject (shares enough content
//! tokens) yet disagrees via a negation, numeric, or temporal mismatch.

use std::collections::HashSet;

use super::types::{
    ConflictKind, KnowledgeConflictConfig, KnowledgeConflictError, PassageConflict,
};
use crate::types::Document;

// ── lexical helpers ─────────────────────────────────────────────────────────────

/// Tokenize `text`: split on non-alphanumeric boundaries, lowercase, keep tokens
/// of length `>= 2`.
pub(super) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .filter(|s| s.chars().count() >= 2)
        .collect()
}

/// Split `text` into sentences on `'.'`, `'!'`, and `'?'`.
///
/// Trimmed, non-empty fragments are returned. A trailing fragment without a
/// terminator is also included.
pub(super) fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

/// Negation markers. Whole-token markers and a substring marker (`n't`).
const NEGATION_TOKENS: &[&str] = &[
    "not", "no", "never", "without", "fails", "fail", "cannot", "false", "neither", "nor",
];

/// Return `true` when `text` exhibits a negation signal.
///
/// Detected via the whole-token markers in `NEGATION_TOKENS` or the contracted
/// `n't` suffix (e.g. `isn't`, `wasn't`, `doesn't`).
pub(super) fn has_negation(text: &str) -> bool {
    let lower = text.to_lowercase();
    if lower.contains("n't") {
        return true;
    }
    let tokens = tokenize(&lower);
    tokens.iter().any(|t| NEGATION_TOKENS.contains(&t.as_str()))
}

/// Extract numeric tokens from `text`.
///
/// Pure digit runs are parsed; surrounding punctuation (commas, currency,
/// percent signs) is stripped. Four-digit year-like values are *excluded* so
/// that temporal mismatches are not double-counted as numeric mismatches.
pub(super) fn extract_numbers(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
                return None;
            }
            // Exclude 4-digit year-like integers; those are handled as temporal.
            let digit_count = cleaned.chars().filter(char::is_ascii_digit).count();
            if digit_count == 4
                && !cleaned.contains('.')
                && cleaned
                    .parse::<u32>()
                    .is_ok_and(|value| (1000..=2100).contains(&value))
            {
                return None;
            }
            cleaned.parse::<f64>().ok()
        })
        .collect()
}

/// Extract 4-digit year tokens in the range `1000..=2100`.
pub(super) fn extract_years(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned: String = token.chars().filter(char::is_ascii_digit).collect();
            if cleaned.len() == 4 {
                cleaned
                    .parse::<u32>()
                    .ok()
                    .filter(|&y| (1000..=2100).contains(&y))
            } else {
                None
            }
        })
        .collect()
}

/// Distinct content tokens shared between two sentences.
///
/// Numeric and year-like tokens are excluded from the subject vocabulary so that
/// shared *subjects* (nouns) are required rather than shared *values*.
pub(super) fn shared_terms(a: &str, b: &str) -> Vec<String> {
    let to_subject_set = |text: &str| -> HashSet<String> {
        tokenize(text)
            .into_iter()
            .filter(|t| !t.chars().all(|c| c.is_ascii_digit()))
            .collect()
    };
    let a_set = to_subject_set(a);
    let b_set = to_subject_set(b);
    a_set.intersection(&b_set).cloned().collect()
}

/// Whether two number sets disagree (both non-empty and sharing no common value).
fn numbers_conflict(a: &[f64], b: &[f64]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    !a.iter()
        .any(|x| b.iter().any(|y| (x - y).abs() < f64::EPSILON))
}

/// Whether two year sets disagree (both non-empty and sharing no common value).
fn years_conflict(a: &[u32], b: &[u32]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    !a.iter().any(|x| b.contains(x))
}

/// Classify the contradiction (if any) between two sentences already known to
/// share enough subject vocabulary.
///
/// Precedence is `Negation` → `Temporal` → `Numeric`, mirroring the most
/// semantically decisive signal first.
fn classify(claim_a: &str, claim_b: &str) -> Option<ConflictKind> {
    // ── Negation mismatch ──────────────────────────────────────────────────
    if has_negation(claim_a) ^ has_negation(claim_b) {
        return Some(ConflictKind::Negation);
    }

    // ── Temporal mismatch (different years) ────────────────────────────────
    let years_a = extract_years(claim_a);
    let years_b = extract_years(claim_b);
    if years_conflict(&years_a, &years_b) {
        return Some(ConflictKind::Temporal);
    }

    // ── Numeric mismatch (different non-year numbers) ──────────────────────
    let nums_a = extract_numbers(claim_a);
    let nums_b = extract_numbers(claim_b);
    if numbers_conflict(&nums_a, &nums_b) {
        return Some(ConflictKind::Numeric);
    }

    None
}

// ── ConflictDetector ────────────────────────────────────────────────────────────

/// Detects contradictions *between* retrieved passages.
///
/// Unlike `consistency_checker` (which scans a single generated answer for
/// internal contradictions) and `fact_check` (which scores a claim against a
/// corpus and emits a verdict), the detector compares passages against each other
/// and reports every sentence-level disagreement as a [`PassageConflict`].
#[derive(Debug, Clone)]
pub struct ConflictDetector {
    /// Configuration controlling the shared-term gate.
    pub config: KnowledgeConflictConfig,
}

impl ConflictDetector {
    /// Construct a detector with the given configuration.
    #[must_use]
    pub fn new(config: KnowledgeConflictConfig) -> Self {
        Self { config }
    }

    /// Detect contradictions across all pairs of `docs`.
    ///
    /// For every pair of distinct passages `(a, b)` and every pair of sentences
    /// drawn from them, a [`PassageConflict`] is emitted when the sentences share
    /// at least `min_shared_terms` content tokens (the *same subject*) and exhibit
    /// a negation, temporal, or numeric mismatch.
    ///
    /// Returns an empty vector when fewer than two documents are supplied. The
    /// output ordering is deterministic: by `(passage_a, passage_b)` and then by
    /// the order in which sentence pairs are visited.
    #[must_use]
    pub fn detect(&self, docs: &[Document]) -> Vec<PassageConflict> {
        let mut conflicts: Vec<PassageConflict> = Vec::new();
        if docs.len() < 2 {
            return conflicts;
        }

        let min_shared = self.config.min_shared_terms;
        for (a, doc_a) in docs.iter().enumerate() {
            let sentences_a = split_sentences(&doc_a.content);
            for (offset, doc_b) in docs[(a + 1)..].iter().enumerate() {
                let b = a + 1 + offset;
                let sentences_b = split_sentences(&doc_b.content);
                for sa in &sentences_a {
                    for sb in &sentences_b {
                        if shared_terms(sa, sb).len() < min_shared {
                            continue;
                        }
                        if let Some(kind) = classify(sa, sb) {
                            conflicts.push(PassageConflict::new(
                                a,
                                b,
                                sa.clone(),
                                sb.clone(),
                                kind,
                            ));
                        }
                    }
                }
            }
        }

        conflicts
    }

    /// Detect contradictions across all pairs of `docs`, erroring on an empty
    /// corpus.
    ///
    /// Behaves like [`ConflictDetector::detect`] but signals an explicit error
    /// when no documents are provided, which is useful when an empty corpus is a
    /// programmer error rather than a benign "nothing to compare" case.
    ///
    /// # Errors
    ///
    /// Returns [`KnowledgeConflictError::EmptyCorpus`] when `docs` is empty.
    pub fn detect_checked(
        &self,
        docs: &[Document],
    ) -> Result<Vec<PassageConflict>, KnowledgeConflictError> {
        if docs.is_empty() {
            return Err(KnowledgeConflictError::EmptyCorpus);
        }
        Ok(self.detect(docs))
    }
}

impl Default for ConflictDetector {
    fn default() -> Self {
        Self::new(KnowledgeConflictConfig::default())
    }
}
