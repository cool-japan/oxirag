//! Types for the `speculative_drafting` module.

use thiserror::Error;

use crate::types::Document;

// ── SpecDraftConfig ──────────────────────────────────────────────────────────────

/// Configuration for [`SpeculativeDrafter`](crate::speculative_drafting::SpeculativeDrafter).
///
/// The retrieved corpus is partitioned into at most `num_clusters` diverse
/// subsets; one draft answer is generated per non-empty cluster. Each draft is
/// then scored by blending a verifier *support* score (weight
/// [`Self::verify_weight`]) with a *self-consistency* agreement score (weight
/// [`Self::consistency_weight`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SpecDraftConfig {
    /// Maximum number of document clusters (and hence drafts). Defaults to `3`.
    pub num_clusters: usize,
    /// Dimensionality of the lexical pseudo-embeddings used for clustering.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Weight applied to a draft's verifier support score. Defaults to `0.6`.
    pub verify_weight: f32,
    /// Weight applied to a draft's self-consistency agreement score.
    ///
    /// Defaults to `0.4`.
    pub consistency_weight: f32,
}

impl Default for SpecDraftConfig {
    fn default() -> Self {
        Self {
            num_clusters: 3,
            dim: 128,
            verify_weight: 0.6,
            consistency_weight: 0.4,
        }
    }
}

impl SpecDraftConfig {
    /// Create a config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of document clusters.
    #[must_use]
    pub fn with_num_clusters(mut self, num_clusters: usize) -> Self {
        self.num_clusters = num_clusters;
        self
    }

    /// Set the pseudo-embedding dimensionality used for clustering.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the weight applied to the verifier support score.
    #[must_use]
    pub fn with_verify_weight(mut self, verify_weight: f32) -> Self {
        self.verify_weight = verify_weight;
        self
    }

    /// Set the weight applied to the self-consistency agreement score.
    #[must_use]
    pub fn with_consistency_weight(mut self, consistency_weight: f32) -> Self {
        self.consistency_weight = consistency_weight;
        self
    }
}

// ── Drafter ──────────────────────────────────────────────────────────────────────

/// A generator that drafts an answer from one *subset* of retrieved documents.
///
/// In Speculative RAG (Wang et al., 2024) each draft sees only the documents of
/// a single cluster, so distinct clusters yield distinct perspectives. The
/// trait is [`Sync`] so drafts can be produced in parallel across clusters.
pub trait Drafter: Sync {
    /// Draft an answer from a document subset.
    fn draft(&self, query: &str, docs: &[&Document]) -> String;
}

/// A verifier that scores how well a draft is grounded in its supporting docs.
///
/// The trait is [`Sync`] so drafts can be verified in parallel across clusters.
pub trait DraftVerifier: Sync {
    /// Score how well a draft is supported by its docs, in `[0, 1]`.
    fn verify(&self, query: &str, draft: &str, docs: &[&Document]) -> f32;
}

// ── MockDrafter ──────────────────────────────────────────────────────────────────

/// Deterministic [`Drafter`] that echoes the most salient document content.
///
/// The draft concatenates, for each document in the cluster, the document's
/// title (when present) and the leading sentence of its content, prefixed with
/// the query. The output is fully determined by the inputs, so identical
/// clusters yield identical drafts.
#[derive(Debug, Clone, Default)]
pub struct MockDrafter;

impl MockDrafter {
    /// Create a new mock drafter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Drafter for MockDrafter {
    fn draft(&self, query: &str, docs: &[&Document]) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(docs.len() + 1);
        parts.push(query.trim().to_string());
        for doc in docs {
            if let Some(title) = &doc.title {
                let title = title.trim();
                if !title.is_empty() {
                    parts.push(title.to_string());
                }
            }
            let snippet = salient_snippet(&doc.content);
            if !snippet.is_empty() {
                parts.push(snippet);
            }
        }
        parts.join(" ")
    }
}

/// Extract the leading sentence (up to the first `.`, `!`, or `?`) of `text`,
/// trimmed. Falls back to the whole trimmed string when no terminator is found.
fn salient_snippet(text: &str) -> String {
    let trimmed = text.trim();
    let end = trimmed
        .find(['.', '!', '?'])
        .map_or(trimmed.len(), |i| i + 1);
    trimmed[..end].trim().to_string()
}

// ── MockDraftVerifier ────────────────────────────────────────────────────────────

/// Deterministic [`DraftVerifier`] scoring draft–document token overlap.
///
/// The support score is the fraction of the draft's content tokens that also
/// appear anywhere in the concatenated cluster documents (a recall-style
/// grounding measure), in `[0, 1]`. Tokens contributed solely by the query are
/// ignored so the score reflects evidence drawn from the documents.
#[derive(Debug, Clone, Default)]
pub struct MockDraftVerifier;

impl MockDraftVerifier {
    /// Create a new mock verifier.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DraftVerifier for MockDraftVerifier {
    #[allow(clippy::cast_precision_loss)]
    fn verify(&self, query: &str, draft: &str, docs: &[&Document]) -> f32 {
        let query_tokens: std::collections::HashSet<String> = tokenize(query).collect();
        let draft_tokens: std::collections::HashSet<String> = tokenize(draft)
            .filter(|t| !query_tokens.contains(t))
            .collect();
        if draft_tokens.is_empty() {
            return 0.0;
        }
        let doc_tokens: std::collections::HashSet<String> = docs
            .iter()
            .flat_map(|d| tokenize(&d.content))
            .chain(
                docs.iter()
                    .filter_map(|d| d.title.as_deref())
                    .flat_map(tokenize),
            )
            .collect();
        let supported = draft_tokens
            .iter()
            .filter(|t| doc_tokens.contains(*t))
            .count();
        supported as f32 / draft_tokens.len() as f32
    }
}

// ── DraftCandidate ───────────────────────────────────────────────────────────────

/// One drafted answer together with its verification and consistency scores.
#[derive(Debug, Clone)]
pub struct DraftCandidate {
    /// The drafted answer text.
    pub content: String,
    /// Index of the cluster (into the cluster list) this draft was generated from.
    pub cluster_id: usize,
    /// Verifier support score for this draft, in `[0, 1]`.
    pub support_score: f32,
    /// Mean token-Jaccard agreement with the *other* drafts, in `[0, 1]`.
    pub self_consistency: f32,
    /// Weighted blend of support and self-consistency, in `[0, 1]`.
    pub total_score: f32,
}

// ── SpeculativeOutput ────────────────────────────────────────────────────────────

/// Result of speculative drafting over a clustered corpus.
#[derive(Debug, Clone)]
pub struct SpeculativeOutput {
    /// The content of the highest-scoring draft.
    pub best: String,
    /// All draft candidates, sorted by descending total score (then tie-break).
    pub drafts: Vec<DraftCandidate>,
    /// Confidence — the winning draft's total score, in `[0, 1]`.
    pub confidence: f32,
}

// ── SpecDraftError ───────────────────────────────────────────────────────────────

/// Errors from the `speculative_drafting` module.
#[derive(Debug, Error)]
pub enum SpecDraftError {
    /// The query was empty after trimming.
    #[error("query must not be empty")]
    EmptyQuery,
    /// The retrieved corpus contained no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
}

// ── Tokenization helpers ─────────────────────────────────────────────────────────

/// Tokenize `text` into lowercase alphanumeric tokens of length `>= 2`.
pub(crate) fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
}

/// Token-set Jaccard similarity between two answer strings, in `[0, 1]`.
///
/// Returns `0.0` when both token sets are empty (no shared evidence).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn token_jaccard(a: &str, b: &str) -> f32 {
    let set_a: std::collections::HashSet<String> = tokenize(a).collect();
    let set_b: std::collections::HashSet<String> = tokenize(b).collect();
    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = set_a.intersection(&set_b).count();
    intersection as f32 / union as f32
}
