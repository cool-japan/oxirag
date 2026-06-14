//! Atomic claim extraction: sentence/clause splitting and decontextualization.
//!
//! The decomposition is fully deterministic and model-free. An answer is split
//! into sentences, each sentence into clauses, tiny clauses are dropped, and —
//! when enabled — a leading pronoun in a clause is rewritten to the answer's
//! main subject so the resulting claim stands on its own.

use super::types::{AtomicClaim, ClaimDecompConfig, ClaimDecompError};

// ── Tokeniser ─────────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens of length >= 2.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── Sentence splitter ─────────────────────────────────────────────────────────

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, and `?`.
#[must_use]
pub fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

// ── Clause splitter ───────────────────────────────────────────────────────────

/// Clause-boundary markers (longest first so that multi-word boundaries win).
const CLAUSE_BOUNDARIES: &[&str] = &[" which ", " but ", " and ", " who ", " that ", "; ", ", "];

/// Split a single sentence into clause-level fragments at clause boundaries.
///
/// Boundaries are " and ", " but ", "; ", ", ", " which ", " who ", and
/// " that ". Empty fragments are dropped and each fragment is trimmed.
#[must_use]
pub fn split_clauses(sentence: &str) -> Vec<String> {
    let mut fragments: Vec<String> = vec![sentence.to_string()];
    for boundary in CLAUSE_BOUNDARIES {
        let mut next: Vec<String> = Vec::new();
        for fragment in &fragments {
            for piece in fragment.split(boundary) {
                next.push(piece.to_string());
            }
        }
        fragments = next;
    }
    fragments
        .into_iter()
        .map(|f| f.trim().to_string())
        .filter(|f| !f.is_empty())
        .collect()
}

// ── Subject detection & pronoun resolution ────────────────────────────────────

/// Leading pronouns that trigger subject substitution during decontextualization.
const PRONOUNS: &[&str] = &[
    "it", "he", "she", "they", "this", "these", "those", "their", "its",
];

/// Determine the heuristic main subject of an answer.
///
/// The subject is the first capitalised multi-character token in the answer;
/// when none exists, `None` is returned and decontextualization is skipped.
fn answer_subject(answer: &str) -> Option<String> {
    for token in answer.split(|c: char| !c.is_alphanumeric()) {
        if token.chars().count() < 2 {
            continue;
        }
        if let Some(first) = token.chars().next()
            && first.is_uppercase()
        {
            return Some(token.to_string());
        }
    }
    None
}

/// Return the first token of `clause` (alphanumeric run), lowercased.
fn first_token_lower(clause: &str) -> Option<String> {
    clause
        .split(|c: char| !c.is_alphanumeric())
        .find(|t| !t.is_empty())
        .map(str::to_lowercase)
}

// ── ClaimExtractor trait ──────────────────────────────────────────────────────

/// Decomposes a generated answer into atomic, self-contained claims.
pub trait ClaimExtractor {
    /// Extract atomic claims from an answer.
    fn extract(&self, answer: &str) -> Vec<AtomicClaim>;
}

// ── HeuristicAtomicExtractor ──────────────────────────────────────────────────

/// Rule-based atomic claim extractor (no model, fully deterministic).
///
/// Sentences are split into clauses, clauses shorter than
/// [`ClaimDecompConfig::min_tokens`] are dropped, leading pronouns are optionally
/// rewritten to the answer subject, and the result is capped at
/// [`ClaimDecompConfig::max_claims`].
#[derive(Debug, Clone, Default)]
pub struct HeuristicAtomicExtractor {
    /// Decomposition configuration.
    pub config: ClaimDecompConfig,
}

impl HeuristicAtomicExtractor {
    /// Create a new extractor with the given configuration.
    #[must_use]
    pub fn new(config: ClaimDecompConfig) -> Self {
        Self { config }
    }

    /// Decontextualize a clause by rewriting a leading pronoun to `subject`.
    ///
    /// Returns the (possibly rewritten) clause text and a flag indicating whether
    /// a substitution actually occurred.
    fn decontextualize(&self, clause: &str, subject: Option<&str>) -> (String, bool) {
        if !self.config.decontextualize {
            return (clause.to_string(), false);
        }
        let Some(subject) = subject else {
            return (clause.to_string(), false);
        };
        let Some(first) = first_token_lower(clause) else {
            return (clause.to_string(), false);
        };
        if PRONOUNS.contains(&first.as_str()) {
            // Replace the leading pronoun token with the subject, preserving the
            // remainder of the clause.
            let rest = clause
                .split_whitespace()
                .skip(1)
                .collect::<Vec<_>>()
                .join(" ");
            let rewritten = if rest.is_empty() {
                subject.to_string()
            } else {
                format!("{subject} {rest}")
            };
            (rewritten, true)
        } else {
            (clause.to_string(), false)
        }
    }
}

impl ClaimExtractor for HeuristicAtomicExtractor {
    fn extract(&self, answer: &str) -> Vec<AtomicClaim> {
        if self.config.max_claims == 0 {
            return Vec::new();
        }
        let subject = answer_subject(answer);
        let subject_ref = subject.as_deref();
        let mut out: Vec<AtomicClaim> = Vec::new();
        for (sentence_idx, sentence) in split_sentences(answer).into_iter().enumerate() {
            for clause in split_clauses(&sentence) {
                if tokenize(&clause).len() < self.config.min_tokens {
                    continue;
                }
                let (text, decontextualized) = self.decontextualize(&clause, subject_ref);
                out.push(AtomicClaim::new(text, sentence_idx, decontextualized));
                if out.len() >= self.config.max_claims {
                    return out;
                }
            }
        }
        out
    }
}

// ── AtomicClaimExtractor ──────────────────────────────────────────────────────

/// Validating front-end over [`HeuristicAtomicExtractor`].
///
/// Wraps the heuristic extractor with input validation, returning a
/// [`ClaimDecompError`] for empty answers rather than an empty claim list.
#[derive(Debug, Clone, Default)]
pub struct AtomicClaimExtractor {
    /// Decomposition configuration.
    pub config: ClaimDecompConfig,
    /// The underlying heuristic extractor.
    pub extractor: HeuristicAtomicExtractor,
}

impl AtomicClaimExtractor {
    /// Create a new extractor with the given configuration.
    #[must_use]
    pub fn new(config: ClaimDecompConfig) -> Self {
        Self {
            config: config.clone(),
            extractor: HeuristicAtomicExtractor::new(config),
        }
    }

    /// Decompose `answer` into atomic claims.
    ///
    /// # Errors
    ///
    /// Returns [`ClaimDecompError::EmptyAnswer`] when `answer` is blank.
    pub fn decompose(&self, answer: &str) -> Result<Vec<AtomicClaim>, ClaimDecompError> {
        if answer.trim().is_empty() {
            return Err(ClaimDecompError::EmptyAnswer);
        }
        Ok(self.extractor.extract(answer))
    }

    /// Count the atomic claims `answer` decomposes into.
    ///
    /// # Errors
    ///
    /// Returns [`ClaimDecompError::EmptyAnswer`] when `answer` is blank.
    pub fn count(&self, answer: &str) -> Result<usize, ClaimDecompError> {
        Ok(self.decompose(answer)?.len())
    }
}
