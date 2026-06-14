//! Types for the `late_chunking` module.
use thiserror::Error;

// ── LatePooling ───────────────────────────────────────────────────────────────

/// Pooling strategy used to combine a chunk's contextual token vectors into a
/// single chunk embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LatePooling {
    /// Element-wise mean over the chunk's token vectors.
    #[default]
    Mean,
    /// Element-wise maximum over the chunk's token vectors.
    Max,
}

// ── LateChunkConfig ───────────────────────────────────────────────────────────

/// Configuration for [`LateChunker`](crate::late_chunking::LateChunker).
#[derive(Debug, Clone, PartialEq)]
pub struct LateChunkConfig {
    /// Dimensionality of the token and chunk embedding vectors.
    pub dim: usize,
    /// Number of tokens per chunk window.
    pub chunk_size_tokens: usize,
    /// Number of overlapping tokens between consecutive chunk windows.
    pub overlap_tokens: usize,
    /// Blend weight in `[0.0, 1.0]` mixing the document-mean vector into each
    /// base token vector to simulate document-aware contextualisation.
    ///
    /// At `0.0` the contextual vectors equal the base vectors, so late chunks
    /// reduce to naive chunks.
    pub context_weight: f32,
    /// Pooling strategy applied over each chunk's contextual token vectors.
    pub pooling: LatePooling,
}

impl Default for LateChunkConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            chunk_size_tokens: 64,
            overlap_tokens: 8,
            context_weight: 0.3,
            pooling: LatePooling::Mean,
        }
    }
}

impl LateChunkConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the number of tokens per chunk window.
    #[must_use]
    pub fn with_chunk_size_tokens(mut self, chunk_size_tokens: usize) -> Self {
        self.chunk_size_tokens = chunk_size_tokens;
        self
    }

    /// Set the number of overlapping tokens between consecutive windows.
    #[must_use]
    pub fn with_overlap_tokens(mut self, overlap_tokens: usize) -> Self {
        self.overlap_tokens = overlap_tokens;
        self
    }

    /// Set the document-context blend weight.
    #[must_use]
    pub fn with_context_weight(mut self, context_weight: f32) -> Self {
        self.context_weight = context_weight;
        self
    }

    /// Set the pooling strategy.
    #[must_use]
    pub fn with_pooling(mut self, pooling: LatePooling) -> Self {
        self.pooling = pooling;
        self
    }
}

// ── LateChunk ─────────────────────────────────────────────────────────────────

/// A single chunk produced by late chunking, carrying its token span and the
/// late-pooled contextual embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct LateChunk {
    /// The reconstructed text of the chunk (tokens joined by a single space).
    pub text: String,
    /// Index of the first token of this chunk within the document (inclusive).
    pub token_start: usize,
    /// Index one past the last token of this chunk within the document (exclusive).
    pub token_end: usize,
    /// The L2-normalised embedding for this chunk.
    pub embedding: Vec<f32>,
}

// ── LateChunkError ────────────────────────────────────────────────────────────

/// Errors from the `late_chunking` module.
#[derive(Debug, Error)]
pub enum LateChunkError {
    /// The supplied document contained no usable tokens.
    #[error("document must not be empty")]
    EmptyDocument,
}
