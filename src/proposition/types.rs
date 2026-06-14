//! Types for the `proposition` module.

use thiserror::Error;

use crate::types::DocumentId;

// ── PropositionConfig ─────────────────────────────────────────────────────────

/// Configuration for proposition extraction and indexing.
#[derive(Debug, Clone)]
pub struct PropositionConfig {
    /// Minimum number of tokens a clause must contain to become a proposition.
    ///
    /// Defaults to `3`.
    pub min_tokens: usize,
    /// Maximum number of propositions retained per document.
    ///
    /// Defaults to `64`.
    pub max_propositions_per_doc: usize,
    /// When `true`, leading pronouns are rewritten to the document subject.
    ///
    /// Defaults to `true`.
    pub resolve_pronouns: bool,
    /// Embedding dimension of the lexical pseudo-embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
}

impl Default for PropositionConfig {
    fn default() -> Self {
        Self {
            min_tokens: 3,
            max_propositions_per_doc: 64,
            resolve_pronouns: true,
            dim: 128,
        }
    }
}

impl PropositionConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum number of tokens a clause must contain.
    #[must_use]
    pub fn with_min_tokens(mut self, v: usize) -> Self {
        self.min_tokens = v;
        self
    }

    /// Set the maximum number of propositions retained per document.
    #[must_use]
    pub fn with_max_propositions_per_doc(mut self, v: usize) -> Self {
        self.max_propositions_per_doc = v;
        self
    }

    /// Set whether leading pronouns are rewritten to the document subject.
    #[must_use]
    pub fn with_resolve_pronouns(mut self, v: bool) -> Self {
        self.resolve_pronouns = v;
        self
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }
}

// ── Proposition ───────────────────────────────────────────────────────────────

/// An atomic, self-contained factual statement extracted from a document.
#[derive(Debug, Clone)]
pub struct Proposition {
    /// Sequential identifier, unique within an index.
    pub id: usize,
    /// The natural-language proposition text.
    pub text: String,
    /// Identifier of the parent document this proposition was extracted from.
    pub parent_id: DocumentId,
    /// Index of the source sentence within the parent document (0-based).
    pub source_sentence: usize,
    /// Lexical pseudo-embedding: hash-bucketed L2-normalised vector.
    pub embedding: Vec<f32>,
}

// ── PropositionHit ────────────────────────────────────────────────────────────

/// A scored proposition returned by [`PropositionIndex::search`].
///
/// [`PropositionIndex::search`]: crate::proposition::PropositionIndex::search
#[derive(Debug, Clone)]
pub struct PropositionHit {
    /// The matched proposition.
    pub proposition: Proposition,
    /// Cosine similarity to the query (higher is more relevant).
    pub score: f32,
}

// ── PropositionError ──────────────────────────────────────────────────────────

/// Errors from the `proposition` module.
#[derive(Debug, Error)]
pub enum PropositionError {
    /// The index holds no propositions.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The supplied query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
}
