//! Configuration for document chunking strategies.

use serde::{Deserialize, Serialize};

/// Configuration controlling how documents are split into chunks.
///
/// All size measurements are in Unicode scalar values (chars), not bytes.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "chunking")]
/// # {
/// use oxirag::chunking::ChunkConfig;
///
/// let config = ChunkConfig::default()
///     .with_chunk_size(512)
///     .with_chunk_overlap(64)
///     .with_min_chunk_size(32);
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Maximum number of characters per chunk (default: 1000).
    pub chunk_size: usize,
    /// Number of characters that overlap between consecutive chunks (default: 200).
    pub chunk_overlap: usize,
    /// Minimum number of characters a chunk must contain to be kept (default: 50).
    pub min_chunk_size: usize,
    /// Whether to strip leading/trailing whitespace from each chunk (default: true).
    pub strip_whitespace: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            min_chunk_size: 50,
            strip_whitespace: true,
        }
    }
}

impl ChunkConfig {
    /// Create a new `ChunkConfig` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum chunk size in characters.
    ///
    /// # Panics
    ///
    /// Does not panic, but if `size` is less than or equal to `chunk_overlap`,
    /// chunking strategies may produce degenerate results.  Callers should
    /// ensure `chunk_size > chunk_overlap`.
    #[must_use]
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the overlap between consecutive chunks in characters.
    #[must_use]
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Set the minimum chunk size in characters.
    ///
    /// Chunks shorter than this value are discarded by all strategies.
    #[must_use]
    pub fn with_min_chunk_size(mut self, min: usize) -> Self {
        self.min_chunk_size = min;
        self
    }

    /// Set whether to strip leading/trailing whitespace from every chunk.
    #[must_use]
    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    /// Return the effective step size between chunk start positions.
    ///
    /// Guarantees a minimum step of 1 so that iteration always advances.
    #[must_use]
    pub(crate) fn step(&self) -> usize {
        if self.chunk_size > self.chunk_overlap {
            self.chunk_size - self.chunk_overlap
        } else {
            // Degenerate config: fall back to stepping one character at a time.
            1
        }
    }
}
