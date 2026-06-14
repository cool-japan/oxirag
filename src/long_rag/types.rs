//! Types for the `long_rag` module.

use thiserror::Error;

use crate::types::DocumentId;

// ── GroupingStrategy ──────────────────────────────────────────────────────────

/// Strategy for assembling source documents into long retrieval units.
///
/// `LongRAG` deliberately favours *fewer, longer* units over many short chunks,
/// so each strategy describes a different way of bundling related material
/// together up to a token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupingStrategy {
    /// Emit exactly one unit per source document, capped at the token budget.
    ///
    /// This is the canonical `LongRAG` grouping: a whole document (a Wikipedia
    /// article, say) is treated as a single retrieval unit.
    #[default]
    ByDocument,
    /// Merge consecutive documents whose lexical embeddings are similar.
    ///
    /// Adjacent documents are concatenated while their cosine similarity meets
    /// the configured threshold and the running unit stays within the budget,
    /// keeping topically coherent material together.
    BySemanticAdjacency,
    /// Concatenate every document in order, then cut into fixed token windows.
    ///
    /// Document boundaries are ignored; the corpus is treated as one stream that
    /// is sliced into units of at most the token budget.
    FixedTokenWindow,
}

// ── LongRagConfig ─────────────────────────────────────────────────────────────

/// Configuration for `LongRAG` grouping and retrieval.
#[derive(Debug, Clone)]
pub struct LongRagConfig {
    /// Maximum number of whitespace-delimited tokens allowed in a long unit.
    ///
    /// Defaults to `2000`.
    pub max_unit_tokens: usize,
    /// How source documents are bundled into long units.
    ///
    /// Defaults to [`GroupingStrategy::ByDocument`].
    pub group_by: GroupingStrategy,
    /// Embedding dimension of the lexical pseudo-embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Minimum cosine similarity for merging adjacent documents.
    ///
    /// Only consulted by [`GroupingStrategy::BySemanticAdjacency`]. Defaults to
    /// `0.3`.
    pub similarity_threshold: f32,
}

impl Default for LongRagConfig {
    fn default() -> Self {
        Self {
            max_unit_tokens: 2000,
            group_by: GroupingStrategy::ByDocument,
            dim: 128,
            similarity_threshold: 0.3,
        }
    }
}

impl LongRagConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of tokens allowed in a long unit.
    #[must_use]
    pub fn with_max_unit_tokens(mut self, v: usize) -> Self {
        self.max_unit_tokens = v;
        self
    }

    /// Set the grouping strategy.
    #[must_use]
    pub fn with_group_by(mut self, v: GroupingStrategy) -> Self {
        self.group_by = v;
        self
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }

    /// Set the cosine-similarity threshold for semantic-adjacency merging.
    #[must_use]
    pub fn with_similarity_threshold(mut self, v: f32) -> Self {
        self.similarity_threshold = v;
        self
    }
}

// ── LongUnit ──────────────────────────────────────────────────────────────────

/// A single long retrieval unit assembled from one or more source documents.
///
/// Unlike a parent-document expansion (which starts from a small matched chunk
/// and grows outward), a `LongUnit` is built up-front by bundling whole
/// documents together, so retrieval ranks a small number of long units directly.
#[derive(Debug, Clone)]
pub struct LongUnit {
    /// Sequential identifier, unique within a grouping pass.
    pub id: usize,
    /// The concatenated (and possibly truncated) text of the unit.
    pub text: String,
    /// Identifiers of every source document contributing to this unit.
    pub source_ids: Vec<DocumentId>,
    /// Whitespace-delimited word count of [`LongUnit::text`].
    pub token_count: usize,
    /// Lexical pseudo-embedding: hash-bucketed, L2-normalised vector.
    pub embedding: Vec<f32>,
}

// ── LongHit ───────────────────────────────────────────────────────────────────

/// A scored long unit returned by [`LongRagRetriever::search`].
///
/// [`LongRagRetriever::search`]: crate::long_rag::LongRagRetriever::search
#[derive(Debug, Clone)]
pub struct LongHit {
    /// The matched long unit.
    pub unit: LongUnit,
    /// Cosine similarity between the query and the unit (higher is better).
    pub score: f32,
}

// ── LongRagError ──────────────────────────────────────────────────────────────

/// Errors from the `long_rag` module.
#[derive(Debug, Error)]
pub enum LongRagError {
    /// The retriever holds no long units.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The supplied query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
}
