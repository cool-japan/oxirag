//! The Astute RAG consolidator: reconcile internal and external knowledge.
//!
//! The consolidator extracts salient external statements from retrieved
//! documents (scoring their reliability by cross-document corroboration),
//! pairs each internal (parametric) claim against the most topically related
//! external claim, detects conflicts via shared-subject + negation/number
//! mismatch heuristics, resolves each conflict by reliability and the
//! configured `prefer_external` policy, and synthesizes a consolidated answer
//! with source attribution.

use std::collections::HashSet;

use crate::types::{Document, DocumentId};

use super::types::{
    AstuteConfig, AstuteError, ConsolidatedKnowledge, InternalKnowledge, KnowledgeConflict,
    KnowledgeSource, KnowledgeStatement,
};

// ── lexical helpers ───────────────────────────────────────────────────────────

/// Tokenize `text`: split on non-alphanumeric chars, lowercase, keep tokens of
/// length `>= 2`.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() >= 2)
        .collect()
}

/// Tokenize `text` into a deduplicated set (same rules as [`tokenize`]).
fn token_set(text: &str) -> HashSet<String> {
    tokenize(text).into_iter().collect()
}

/// Stop words excluded when computing a claim's *subject* (content) terms.
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had", "and", "or", "but", "it",
    "in", "of", "to", "for", "at", "by", "be", "do", "so", "if", "as", "not", "no", "never", "on",
    "with", "that", "this", "than", "from", "its", "their", "his", "her",
];

/// Content-bearing terms of a claim: tokens of length `>= 3`, excluding stop
/// words and pure-numeric tokens.
fn subject_terms(text: &str) -> HashSet<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| {
            t.len() >= 3
                && !STOP_WORDS.contains(&t.as_str())
                && !t.chars().all(|c| c.is_ascii_digit())
        })
        .collect()
}

/// Split `text` into sentences on `'.'`, `'!'`, and `'?'`, trimming whitespace
/// and dropping empties.
fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

/// Negation markers used by the conflict heuristic.
const NEGATION_PATTERNS: &[&str] = &[
    "not ",
    "never ",
    " no ",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "doesn't",
    "don't",
    "won't",
    "can't",
    "cannot",
    "couldn't",
    "didn't",
    "without ",
    "false",
    "incorrect",
];

/// Whether `text` contains an explicit negation marker.
fn contains_negation(text: &str) -> bool {
    // Pad with spaces so leading/trailing single-word markers match.
    let padded = format!(" {} ", text.to_lowercase());
    NEGATION_PATTERNS.iter().any(|p| padded.contains(p))
}

/// Extract numeric values from `text` (stripping non-digit boundary chars).
fn extract_numbers(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            // Reject bare "-" / "." / "" that would not parse.
            cleaned.parse::<f64>().ok()
        })
        .collect()
}

/// Jaccard overlap between the token sets of `a` and `b`.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let score = inter as f32 / union as f32;
    score
}

// ── AstuteConsolidator ────────────────────────────────────────────────────────

/// Reconciles a model's parametric knowledge with retrieved external knowledge.
///
/// See the module-level documentation for the full algorithm. The consolidator
/// is deterministic and dependency-free: identical inputs always produce
/// identical outputs.
#[derive(Debug, Clone)]
pub struct AstuteConsolidator {
    /// Configuration controlling reliability thresholds and resolution policy.
    pub config: AstuteConfig,
}

impl AstuteConsolidator {
    /// Construct a consolidator with the given configuration.
    #[must_use]
    pub fn new(config: AstuteConfig) -> Self {
        Self { config }
    }

    /// Extract salient external statements from the retrieved `docs`.
    ///
    /// Each document is split into sentences; every sentence becomes a
    /// candidate [`KnowledgeStatement`] tagged with its originating
    /// [`DocumentId`]. The `reliability` of each statement is derived from
    /// **corroboration**: the fraction of *other* documents whose content
    /// shares the statement's subject terms. A claim echoed across many
    /// documents is more reliable than one appearing in a single source.
    #[must_use]
    pub fn external_statements(&self, docs: &[Document]) -> Vec<KnowledgeStatement> {
        // Pre-compute each document's full token set for corroboration checks.
        let doc_tokens: Vec<(DocumentId, HashSet<String>)> = docs
            .iter()
            .map(|d| (d.id.clone(), token_set(&d.content)))
            .collect();

        let mut statements = Vec::new();
        for (idx, doc) in docs.iter().enumerate() {
            for sentence in split_sentences(&doc.content) {
                let subj = subject_terms(&sentence);
                if subj.is_empty() {
                    continue;
                }
                let reliability = corroboration(idx, &subj, &doc_tokens);
                statements.push(KnowledgeStatement::external(
                    sentence,
                    doc.id.clone(),
                    reliability,
                ));
            }
        }
        statements
    }

    /// Detect a conflict between an `internal` claim and an `external` claim.
    ///
    /// Returns `true` when the two claims share a subject (at least one
    /// content-bearing term in common) **and** disagree, where disagreement is
    /// either a negation mismatch (exactly one claim is negated) or a numerical
    /// mismatch (both claims carry numbers but no number is shared).
    #[must_use]
    pub fn detect_conflict(&self, internal: &str, external: &str) -> bool {
        let shared = subject_terms(internal);
        let other = subject_terms(external);
        let common = shared.intersection(&other).count();
        if common == 0 {
            return false;
        }

        // Negation mismatch: exactly one side is negated.
        if contains_negation(internal) ^ contains_negation(external) {
            return true;
        }

        // Numerical mismatch: both carry numbers, but they share none.
        let nums_i = extract_numbers(internal);
        let nums_e = extract_numbers(external);
        if !nums_i.is_empty() && !nums_e.is_empty() {
            let shares_number = nums_i
                .iter()
                .any(|a| nums_e.iter().any(|b| (a - b).abs() < f64::EPSILON));
            if !shares_number {
                return true;
            }
        }

        false
    }

    /// Reconcile internal and external knowledge for `query`.
    ///
    /// Internal claims are paired against the most topically related external
    /// claim (by token Jaccard). Conflicting pairs are resolved by reliability
    /// and the `prefer_external` policy; non-conflicting internal and external
    /// statements are both retained. The winning statements are then used to
    /// synthesize an attributed answer.
    ///
    /// # Errors
    ///
    /// Returns [`AstuteError::EmptyQuery`] when `query` is blank, and
    /// [`AstuteError::NoKnowledge`] when neither internal claims nor external
    /// documents yield any usable statement.
    pub fn consolidate(
        &self,
        query: &str,
        internal: &[String],
        docs: &[Document],
    ) -> Result<ConsolidatedKnowledge, AstuteError> {
        if query.trim().is_empty() {
            return Err(AstuteError::EmptyQuery);
        }

        let external = self.external_statements(docs);
        let internal_claims: Vec<&String> =
            internal.iter().filter(|s| !s.trim().is_empty()).collect();

        if internal_claims.is_empty() && external.is_empty() {
            return Err(AstuteError::NoKnowledge);
        }

        // Token sets for external statements (reused for topic pairing).
        let external_tokens: Vec<HashSet<String>> =
            external.iter().map(|s| token_set(&s.content)).collect();

        let mut conflicts: Vec<KnowledgeConflict> = Vec::new();
        let mut winners: Vec<KnowledgeStatement> = Vec::new();
        // Track which external statements were consumed by a conflict so they
        // are not also emitted as standalone survivors.
        let mut external_used = vec![false; external.len()];

        for claim in &internal_claims {
            let claim_tokens = token_set(claim);
            // Find the most topically related external statement.
            let best = external
                .iter()
                .enumerate()
                .map(|(i, stmt)| (i, jaccard(&claim_tokens, &external_tokens[i]), stmt))
                .filter(|(_, overlap, _)| *overlap > 0.0)
                .max_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Tie-break on index for determinism (earlier wins).
                        .then(b.0.cmp(&a.0))
                });

            match best {
                Some((idx, _, ext_stmt)) if self.detect_conflict(claim, &ext_stmt.content) => {
                    let resolution = self.resolve(ext_stmt);
                    let winner = match &resolution {
                        KnowledgeSource::External(_) => ext_stmt.clone(),
                        KnowledgeSource::Internal => {
                            KnowledgeStatement::internal((*claim).clone(), 1.0)
                        }
                    };
                    conflicts.push(KnowledgeConflict::new(
                        (*claim).clone(),
                        ext_stmt.content.clone(),
                        resolution,
                    ));
                    winners.push(winner);
                    external_used[idx] = true;
                }
                // No conflict (or no related external claim): keep the internal
                // statement as parametric knowledge.
                _ => {
                    winners.push(KnowledgeStatement::internal((*claim).clone(), 1.0));
                }
            }
        }

        // Append every external statement not already consumed by a conflict.
        for (i, stmt) in external.iter().enumerate() {
            if !external_used[i] {
                winners.push(stmt.clone());
            }
        }

        let answer = synthesize_answer(query, &winners);
        Ok(ConsolidatedKnowledge::new(winners, conflicts, answer))
    }

    /// Reconcile knowledge using an [`InternalKnowledge`] model.
    ///
    /// Recalls the model's parametric claims for `query` and delegates to
    /// [`AstuteConsolidator::consolidate`].
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`AstuteConsolidator::consolidate`].
    pub fn run<K: InternalKnowledge>(
        &self,
        query: &str,
        model: &K,
        docs: &[Document],
    ) -> Result<ConsolidatedKnowledge, AstuteError> {
        let internal = model.recall(query);
        self.consolidate(query, &internal, docs)
    }

    /// Decide which source wins a conflict against `external`.
    ///
    /// External knowledge wins when `prefer_external` is set and the external
    /// claim is corroborated (its reliability meets `reliability_threshold`).
    /// Otherwise internal (parametric) knowledge is trusted.
    fn resolve(&self, external: &KnowledgeStatement) -> KnowledgeSource {
        let corroborated = external.reliability >= self.config.reliability_threshold;
        if self.config.prefer_external && corroborated {
            external.source.clone()
        } else {
            KnowledgeSource::Internal
        }
    }
}

impl Default for AstuteConsolidator {
    fn default() -> Self {
        Self::new(AstuteConfig::default())
    }
}

// ── free helpers ──────────────────────────────────────────────────────────────

/// Corroboration score for a claim whose subject is `subj`, appearing in the
/// document at `self_idx`.
///
/// Returns the fraction of *all* documents (including the source) whose token
/// set contains every subject term, clamped to `[0.0, 1.0]`. A lone claim
/// therefore scores `1.0 / n`, while a claim echoed everywhere scores `1.0`.
fn corroboration(
    self_idx: usize,
    subj: &HashSet<String>,
    doc_tokens: &[(DocumentId, HashSet<String>)],
) -> f32 {
    let n = doc_tokens.len();
    if n == 0 {
        return 0.0;
    }
    let mut supporting = 0_usize;
    for (idx, (_, tokens)) in doc_tokens.iter().enumerate() {
        // The source document trivially supports its own claim.
        let supports = idx == self_idx || subj.iter().all(|t| tokens.contains(t));
        if supports {
            supporting += 1;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let score = supporting as f32 / n as f32;
    score.clamp(0.0, 1.0)
}

/// Synthesize a consolidated, attributed answer from the `winners`.
///
/// The answer opens with the query, then lists each winning statement prefixed
/// with its source label (`[internal]` / `[external]`). It is always non-empty
/// because the consolidator guarantees at least one statement.
fn synthesize_answer(query: &str, winners: &[KnowledgeStatement]) -> String {
    let mut out = format!("Consolidated answer for: {}", query.trim());
    if winners.is_empty() {
        // Defensive: the caller guarantees at least one winner, but keep the
        // answer non-empty regardless.
        out.push_str(
            "\nNo reconciled knowledge was available; answer is based on the query alone.",
        );
        return out;
    }
    for stmt in winners {
        out.push('\n');
        out.push('[');
        out.push_str(stmt.source.as_str());
        out.push_str("] ");
        out.push_str(stmt.content.trim());
        if !stmt.content.trim().ends_with('.') {
            out.push('.');
        }
    }
    out
}
