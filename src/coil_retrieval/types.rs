//! Types, configuration, and errors for the `coil_retrieval` module.
//!
//! COIL (Gao, Dai, Callan 2021) represents a document as a set of
//! *contextualised* per-token vectors, keyed in an inverted list by their
//! **surface token string**. This file defines the building blocks:
//!
//! - [`CoilConfig`] — encoding and scoring parameters.
//! - [`CoilScoreMode`] — COIL-tok versus COIL-full scoring.
//! - [`CoilTokenVector`] — one contextualised token embedding.
//! - [`CoilPosting`] — one token occurrence inside a specific document.
//! - [`CoilDocument`] — a fully encoded document (token vectors plus a CLS
//!   vector).
//! - [`CoilError`] / [`CoilResult`] — the module's error type and result
//!   alias.

use thiserror::Error;

use crate::types::DocumentId;

// ── CoilError / CoilResult ────────────────────────────────────────────────────

/// Errors produced by the `coil_retrieval` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CoilError {
    /// The corpus passed to
    /// [`CoilRetriever::index`](crate::coil_retrieval::CoilRetriever::index)
    /// contained no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The query contained no usable (alphanumeric) tokens after
    /// tokenisation.
    #[error("query must not be empty")]
    EmptyQuery,
    /// A search was attempted before the retriever had been indexed with
    /// [`CoilRetriever::index`](crate::coil_retrieval::CoilRetriever::index).
    #[error("retriever not indexed")]
    NotIndexed,
}

/// Convenient result alias for the `coil_retrieval` module.
pub type CoilResult<T> = Result<T, CoilError>;

// ── CoilScoreMode ─────────────────────────────────────────────────────────────

/// Which COIL scoring variant to apply.
///
/// COIL exposes two closely related scorers (Gao, Dai, Callan 2021). Both sum
/// an *exact-surface-token* contextualised max-similarity over the query
/// tokens; COIL-full additionally blends in a document-level (CLS) semantic
/// term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoilScoreMode {
    /// COIL-tok: the score is *only* the sum, over query tokens, of the exact
    /// surface-token contextualised max-similarity. A document that shares no
    /// surface token with the query is never scored (pure lexical gating).
    Tok,
    /// COIL-full: COIL-tok plus a `lambda`-weighted document-level CLS
    /// similarity term. The CLS term lets a semantically related document be
    /// retrieved even without any exact lexical overlap, so every indexed
    /// document is a scoring candidate.
    #[default]
    Full,
}

impl CoilScoreMode {
    /// Return a stable lowercase string representation of the mode.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tok => "tok",
            Self::Full => "full",
        }
    }

    /// Return `true` when this mode adds the document-level CLS term
    /// (COIL-full).
    #[must_use]
    pub fn includes_cls(&self) -> bool {
        matches!(self, Self::Full)
    }
}

// ── CoilTokenVector ───────────────────────────────────────────────────────────

/// A single contextualised token vector.
///
/// Unlike a plain bag-of-tokens embedding, the same surface token receives a
/// *different* vector in different contexts, because the token's own
/// hash-embedding is blended with a small window of its neighbours. The
/// embedding is L2-normalised, so [`CoilTokenVector::dot`] is a cosine
/// similarity in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoilTokenVector {
    /// The lowercased surface token string this vector represents.
    pub surface: String,
    /// Zero-based position of the token in its source token sequence.
    pub position: usize,
    /// The L2-normalised contextualised embedding.
    pub vector: Vec<f32>,
}

impl CoilTokenVector {
    /// Construct a new contextualised token vector.
    #[must_use]
    pub fn new(surface: impl Into<String>, position: usize, vector: Vec<f32>) -> Self {
        Self {
            surface: surface.into(),
            position,
            vector,
        }
    }

    /// Dimensionality of the embedding.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Dot product (cosine similarity, since both vectors are L2-normalised)
    /// against another token vector.
    ///
    /// When the two embeddings differ in length the shorter length governs,
    /// so the operation never panics.
    #[must_use]
    pub fn dot(&self, other: &CoilTokenVector) -> f32 {
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

// ── CoilPosting ───────────────────────────────────────────────────────────────

/// A single posting in a
/// [`CoilInvertedIndex`](crate::coil_retrieval::CoilInvertedIndex): one
/// occurrence of a surface token inside a specific document, paired with that
/// occurrence's contextualised token vector.
///
/// COIL stores one posting *per occurrence*, so a surface token appearing
/// three times in a document contributes three postings (each with a
/// potentially different contextualised vector).
#[derive(Debug, Clone, PartialEq)]
pub struct CoilPosting {
    /// Identifier of the document this occurrence belongs to.
    pub doc_id: DocumentId,
    /// The contextualised token vector for this occurrence.
    pub token_vector: CoilTokenVector,
}

impl CoilPosting {
    /// Construct a new posting.
    #[must_use]
    pub fn new(doc_id: DocumentId, token_vector: CoilTokenVector) -> Self {
        Self {
            doc_id,
            token_vector,
        }
    }

    /// Borrow the surface token string of this posting.
    #[must_use]
    pub fn surface(&self) -> &str {
        &self.token_vector.surface
    }
}

// ── CoilDocument ──────────────────────────────────────────────────────────────

/// A fully encoded document: its identifier, one contextualised token vector
/// per token position, and a single document-level CLS vector.
///
/// This is the output of the encoder and the input consumed by
/// [`CoilInvertedIndex::insert_document`](crate::coil_retrieval::CoilInvertedIndex::insert_document).
#[derive(Debug, Clone, PartialEq)]
pub struct CoilDocument {
    /// Identifier of the encoded document.
    pub doc_id: DocumentId,
    /// One contextualised token vector per token position, in source order.
    pub token_vectors: Vec<CoilTokenVector>,
    /// The L2-normalised document-level CLS vector (mean of the document's
    /// per-token base embeddings).
    pub cls_vector: Vec<f32>,
}

impl CoilDocument {
    /// Construct a new encoded document.
    #[must_use]
    pub fn new(
        doc_id: DocumentId,
        token_vectors: Vec<CoilTokenVector>,
        cls_vector: Vec<f32>,
    ) -> Self {
        Self {
            doc_id,
            token_vectors,
            cls_vector,
        }
    }

    /// Number of token vectors (i.e. tokens) in the document.
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_vectors.len()
    }

    /// Return `true` when the document produced no token vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_vectors.is_empty()
    }
}

// ── CoilConfig ────────────────────────────────────────────────────────────────

/// Configuration for the COIL encoder and retriever.
///
/// The defaults mirror the other retrieval modules in this crate: a 128-dim
/// embedding, a context window of two tokens on each side, a moderate
/// neighbour blend, and COIL-full scoring with an equal-weight CLS term.
#[derive(Debug, Clone, PartialEq)]
pub struct CoilConfig {
    /// Dimensionality of every per-token and CLS embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Radius of the local context window blended into each token's
    /// contextualised vector: positions `p - context_window ..= p +
    /// context_window` (excluding `p`) contribute.
    ///
    /// Defaults to `2`.
    pub context_window: usize,
    /// Weight applied to the averaged neighbour embedding when forming a
    /// contextualised token vector. A larger value makes context matter more
    /// (stronger polysemy); `0.0` disables contextualisation entirely.
    ///
    /// Defaults to `0.5`.
    pub context_weight: f32,
    /// Weight of the document-level CLS similarity term in COIL-full scoring.
    /// Ignored in COIL-tok mode.
    ///
    /// Defaults to `1.0`.
    pub lambda: f32,
    /// Which scoring variant [`CoilRetriever::search`](crate::coil_retrieval::CoilRetriever::search)
    /// applies.
    ///
    /// Defaults to [`CoilScoreMode::Full`].
    pub score_mode: CoilScoreMode,
    /// Default number of results returned by
    /// [`CoilRetriever::search_default`](crate::coil_retrieval::CoilRetriever::search_default).
    ///
    /// Defaults to `10`.
    pub top_k: usize,
}

impl Default for CoilConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            context_window: 2,
            context_weight: 0.5,
            lambda: 1.0,
            score_mode: CoilScoreMode::Full,
            top_k: 10,
        }
    }
}

impl CoilConfig {
    /// Create a configuration with the default COIL parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-token and CLS embedding dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the context-window radius blended into each token vector.
    #[must_use]
    pub fn with_context_window(mut self, context_window: usize) -> Self {
        self.context_window = context_window;
        self
    }

    /// Set the neighbour-blend weight used when contextualising tokens.
    #[must_use]
    pub fn with_context_weight(mut self, context_weight: f32) -> Self {
        self.context_weight = context_weight;
        self
    }

    /// Set the CLS term weight used by COIL-full scoring.
    #[must_use]
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the scoring variant (COIL-tok or COIL-full).
    #[must_use]
    pub fn with_score_mode(mut self, score_mode: CoilScoreMode) -> Self {
        self.score_mode = score_mode;
        self
    }

    /// Set the default result count.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}
