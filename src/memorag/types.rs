//! Configuration, result, and error types for the `memorag` module.

use thiserror::Error;

use crate::types::Document;

// ── MemoRagConfig ─────────────────────────────────────────────────────────────

/// Configuration for a [`MemoRagEngine`].
///
/// The defaults follow the `MemoRAG` recipe of Qian et al. (2024): a small
/// corpus-wide gist of the most salient sentences, a handful of retrieval clues,
/// and a compact set of memory key terms.
///
/// [`MemoRagEngine`]: crate::memorag::MemoRagEngine
#[derive(Debug, Clone)]
pub struct MemoRagConfig {
    /// Number of salient sentences retained in the corpus-wide memory gist.
    ///
    /// Defaults to `5`.
    pub gist_sentences: usize,
    /// Maximum number of retrieval clues generated per query.
    ///
    /// Defaults to `3`.
    pub num_clues: usize,
    /// Number of memory key terms distilled from the corpus.
    ///
    /// Defaults to `10`.
    pub key_terms: usize,
    /// Embedding dimension of the lexical pseudo-embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
}

impl Default for MemoRagConfig {
    fn default() -> Self {
        Self {
            gist_sentences: 5,
            num_clues: 3,
            key_terms: 10,
            dim: 128,
        }
    }
}

impl MemoRagConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of salient sentences retained in the memory gist.
    #[must_use]
    pub fn with_gist_sentences(mut self, v: usize) -> Self {
        self.gist_sentences = v;
        self
    }

    /// Set the maximum number of retrieval clues generated per query.
    #[must_use]
    pub fn with_num_clues(mut self, v: usize) -> Self {
        self.num_clues = v;
        self
    }

    /// Set the number of memory key terms distilled from the corpus.
    #[must_use]
    pub fn with_key_terms(mut self, v: usize) -> Self {
        self.key_terms = v;
        self
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }
}

// ── MemoryGist ────────────────────────────────────────────────────────────────

/// A compressed, corpus-wide global memory.
///
/// The gist is the artefact that distinguishes `MemoRAG` from per-document
/// summary indexes: it captures the salient *content of the whole corpus* in a
/// short summary plus a ranked list of key terms. Those key terms are later used
/// to expand queries into retrieval clues.
#[derive(Debug, Clone)]
pub struct MemoryGist {
    /// The corpus-wide extractive summary (salient sentences in original order).
    pub summary: String,
    /// The most salient corpus terms, ranked by descending corpus frequency.
    pub key_terms: Vec<String>,
}

// ── MemoHit ───────────────────────────────────────────────────────────────────

/// A scored evidence document retrieved via a memory-derived clue.
///
/// Each hit records the [`clue`](MemoHit::clue) that won the document — i.e. the
/// surrogate sub-query whose embedding scored the document highest across all
/// generated clues.
#[derive(Debug, Clone)]
pub struct MemoHit {
    /// The retrieved evidence document.
    pub document: Document,
    /// Cosine similarity between the winning clue and the document.
    pub score: f32,
    /// The retrieval clue that produced this (winning) score for the document.
    pub clue: String,
}

// ── MemoRagError ──────────────────────────────────────────────────────────────

/// Errors from the `memorag` module.
#[derive(Debug, Error)]
pub enum MemoRagError {
    /// The supplied corpus held no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The supplied query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
    /// A clue or retrieval operation was attempted before the memory was built.
    #[error("memory not built")]
    MemoryNotBuilt,
}
