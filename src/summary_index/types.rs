//! Types for the `summary_index` module.

use thiserror::Error;

use crate::types::{Document, DocumentId};

// ── SummaryConfig ─────────────────────────────────────────────────────────────

/// Configuration for a [`SummaryIndex`].
///
/// [`SummaryIndex`]: crate::summary_index::SummaryIndex
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    /// Number of salient sentences retained in each document summary.
    ///
    /// Defaults to `3`.
    pub summary_sentences: usize,
    /// Embedding dimension of the lexical pseudo-embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            summary_sentences: 3,
            dim: 128,
        }
    }
}

impl SummaryConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of salient sentences retained in each summary.
    #[must_use]
    pub fn with_summary_sentences(mut self, v: usize) -> Self {
        self.summary_sentences = v;
        self
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }
}

// ── DocumentSummary ───────────────────────────────────────────────────────────

/// An extractive summary of a single document together with its embedding.
///
/// The embedding is computed from the [`summary`](DocumentSummary::summary)
/// text, not the full parent document; this is what makes the index
/// summary-centric.
#[derive(Debug, Clone)]
pub struct DocumentSummary {
    /// Identifier of the parent document this summary was derived from.
    pub doc_id: DocumentId,
    /// The extractive summary text (salient sentences in original order).
    pub summary: String,
    /// Lexical pseudo-embedding of the summary: hash-bucketed, L2-normalised.
    pub embedding: Vec<f32>,
}

// ── SummaryHit ────────────────────────────────────────────────────────────────

/// A scored search result returned by [`SummaryIndex::search`].
///
/// The query is matched against document summaries, but the **full parent
/// document** is returned so that downstream consumers retain all context.
///
/// [`SummaryIndex::search`]: crate::summary_index::SummaryIndex::search
#[derive(Debug, Clone)]
pub struct SummaryHit {
    /// The full parent document (not the summary).
    pub document: Document,
    /// The extractive summary that produced the match.
    pub summary: String,
    /// Cosine similarity between the query and the summary (higher is better).
    pub score: f32,
}

// ── SummaryIndexError ─────────────────────────────────────────────────────────

/// Errors from the `summary_index` module.
#[derive(Debug, Error)]
pub enum SummaryIndexError {
    /// The index holds no document summaries.
    #[error("index is empty")]
    EmptyIndex,
    /// The supplied query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
}
