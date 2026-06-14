//! Types for the `cross_lingual` module.

use thiserror::Error;

use crate::types::Document;

// ── CrossLingualConfig ──────────────────────────────────────────────────────

/// Configuration for cross-lingual retrieval.
#[derive(Debug, Clone)]
pub struct CrossLingualConfig {
    /// Embedding dimension of the lexical pseudo-embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// When `true`, query tokens are expanded with their lexicon translations.
    ///
    /// Defaults to `true`.
    pub expand_with_lexicon: bool,
}

impl Default for CrossLingualConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            expand_with_lexicon: true,
        }
    }
}

impl CrossLingualConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set whether query tokens are expanded with their lexicon translations.
    #[must_use]
    pub fn with_expand_with_lexicon(mut self, expand: bool) -> Self {
        self.expand_with_lexicon = expand;
        self
    }
}

// ── CrossLingualHit ─────────────────────────────────────────────────────────

/// A scored document returned by cross-lingual search.
#[derive(Debug, Clone)]
pub struct CrossLingualHit {
    /// The matched document.
    pub document: Document,
    /// Cosine similarity to the expanded query (higher is more relevant).
    pub score: f32,
}

// ── CrossLingualError ───────────────────────────────────────────────────────

/// Errors from the `cross_lingual` module.
#[derive(Debug, Error)]
pub enum CrossLingualError {
    /// The retriever holds no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The supplied query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
}
