//! Proposition extraction from documents.

use crate::proposition::types::PropositionConfig;
use crate::types::Document;

// ── Lexical pseudo-embedding ──────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise → hash each token to a bucket with FNV-1a → accumulate
/// per-bucket counts → L2-normalise to the requested `dim`.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        // Deterministic hash: FNV-1a over the lowercased token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.to_lowercase().as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

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
/// Boundaries are " and ", " but ", "; ", " which ", " who ", " that ", and
/// ", " (commas separating independent clauses).
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

// ── Pronoun resolution ────────────────────────────────────────────────────────

/// Leading pronouns that trigger subject substitution.
const PRONOUNS: &[&str] = &[
    "it", "he", "she", "they", "this", "these", "those", "their", "its",
];

/// Determine the heuristic subject of a document.
///
/// The subject is the first capitalised multi-character token in the content;
/// when none exists, the document title is used as a fallback.
fn document_subject(doc: &Document) -> Option<String> {
    for token in doc.content.split(|c: char| !c.is_alphanumeric()) {
        if token.chars().count() < 2 {
            continue;
        }
        if let Some(first) = token.chars().next()
            && first.is_uppercase()
        {
            return Some(token.to_string());
        }
    }
    doc.title.clone()
}

/// Return the first token of `clause` (alphanumeric run), lowercased.
fn first_token_lower(clause: &str) -> Option<String> {
    clause
        .split(|c: char| !c.is_alphanumeric())
        .find(|t| !t.is_empty())
        .map(str::to_lowercase)
}

// ── PropositionExtractor trait ────────────────────────────────────────────────

/// Decomposes a document into self-contained proposition strings.
pub trait PropositionExtractor {
    /// Extract self-contained proposition strings from a document.
    fn extract(&self, doc: &Document) -> Vec<String>;
}

// ── HeuristicPropositionExtractor ─────────────────────────────────────────────

/// Rule-based proposition extractor (no model, fully deterministic).
///
/// Sentences are split into clauses, tiny clauses are dropped, leading pronouns
/// are optionally resolved to the document subject, and the result is capped at
/// [`PropositionConfig::max_propositions_per_doc`].
#[derive(Debug, Clone, Default)]
pub struct HeuristicPropositionExtractor {
    /// Extraction configuration.
    pub config: PropositionConfig,
}

impl HeuristicPropositionExtractor {
    /// Create a new extractor with the given configuration.
    #[must_use]
    pub fn new(config: PropositionConfig) -> Self {
        Self { config }
    }

    /// Extract propositions paired with their source-sentence index.
    ///
    /// This is the single source of truth for decomposition; both the
    /// [`PropositionExtractor::extract`] trait method and the index build on it.
    #[must_use]
    pub fn extract_with_positions(&self, doc: &Document) -> Vec<(usize, String)> {
        let subject = document_subject(doc);
        let subject_ref = subject.as_deref();
        let mut out: Vec<(usize, String)> = Vec::new();
        for (sentence_idx, sentence) in split_sentences(&doc.content).into_iter().enumerate() {
            for clause in split_clauses(&sentence) {
                if tokenize(&clause).len() < self.config.min_tokens {
                    continue;
                }
                out.push((sentence_idx, self.resolve(&clause, subject_ref)));
                if out.len() >= self.config.max_propositions_per_doc {
                    return out;
                }
            }
        }
        out
    }

    /// Resolve a leading pronoun in `clause` to `subject` when applicable.
    fn resolve(&self, clause: &str, subject: Option<&str>) -> String {
        if !self.config.resolve_pronouns {
            return clause.to_string();
        }
        let Some(subject) = subject else {
            return clause.to_string();
        };
        let Some(first) = first_token_lower(clause) else {
            return clause.to_string();
        };
        if PRONOUNS.contains(&first.as_str()) {
            // Replace the leading pronoun token with the subject, preserving the
            // remainder of the clause.
            let rest = clause
                .split_whitespace()
                .skip(1)
                .collect::<Vec<_>>()
                .join(" ");
            if rest.is_empty() {
                subject.to_string()
            } else {
                format!("{subject} {rest}")
            }
        } else {
            clause.to_string()
        }
    }
}

impl PropositionExtractor for HeuristicPropositionExtractor {
    fn extract(&self, doc: &Document) -> Vec<String> {
        self.extract_with_positions(doc)
            .into_iter()
            .map(|(_, text)| text)
            .collect()
    }
}
