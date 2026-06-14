//! Types for the `product_quantization` module.

use thiserror::Error;

use crate::types::DocumentId;

// ── PqConfig ──────────────────────────────────────────────────────────────────

/// Configuration for a [`ProductQuantizer`](super::ProductQuantizer).
///
/// Product quantization splits each `dim`-dimensional vector into
/// `num_subspaces` contiguous subvectors of equal length and learns a separate
/// codebook of `K = 2^codebook_bits` centroids for each subspace. The
/// `codebook_bits` value is capped at `8` so that every codeword fits in a
/// single [`u8`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PqConfig {
    /// Number of contiguous subspaces the input vector is split into.
    ///
    /// Defaults to `4`. Must evenly divide [`dim`](Self::dim).
    pub num_subspaces: usize,
    /// Number of bits per codeword, so each codebook has `2^codebook_bits`
    /// centroids.
    ///
    /// Defaults to `8` (the maximum). Must be `<= 8`.
    pub codebook_bits: usize,
    /// Dimensionality of the input vectors.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Maximum number of Lloyd (k-means) iterations per subspace.
    ///
    /// Defaults to `10`.
    pub kmeans_iters: usize,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 4,
            codebook_bits: 8,
            dim: 128,
            kmeans_iters: 10,
        }
    }
}

impl PqConfig {
    /// Create a new configuration with the [default](Self::default) values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of subspaces.
    #[must_use]
    pub fn with_num_subspaces(mut self, num_subspaces: usize) -> Self {
        self.num_subspaces = num_subspaces;
        self
    }

    /// Set the number of bits per codeword.
    #[must_use]
    pub fn with_codebook_bits(mut self, codebook_bits: usize) -> Self {
        self.codebook_bits = codebook_bits;
        self
    }

    /// Set the input vector dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the maximum number of k-means iterations.
    #[must_use]
    pub fn with_kmeans_iters(mut self, kmeans_iters: usize) -> Self {
        self.kmeans_iters = kmeans_iters;
        self
    }

    /// Number of centroids per codebook (`2^codebook_bits`).
    #[must_use]
    pub fn codebook_size(&self) -> usize {
        1usize << self.codebook_bits
    }

    /// Length of each subvector (`dim / num_subspaces`).
    ///
    /// Returns `0` when `num_subspaces` is `0` to avoid division by zero;
    /// such a configuration is rejected by [`validate`](Self::validate).
    #[must_use]
    pub fn subspace_dim(&self) -> usize {
        self.dim.checked_div(self.num_subspaces).unwrap_or(0)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PqError::InvalidConfig`] when `dim` is not divisible by
    /// `num_subspaces` (including the `num_subspaces == 0` case), and
    /// [`PqError::DimMismatch`] when `codebook_bits` exceeds `8`.
    pub fn validate(&self) -> Result<(), PqError> {
        if self.num_subspaces == 0 || !self.dim.is_multiple_of(self.num_subspaces) {
            return Err(PqError::InvalidConfig {
                dim: self.dim,
                subspaces: self.num_subspaces,
            });
        }
        if self.codebook_bits > 8 {
            return Err(PqError::DimMismatch);
        }
        Ok(())
    }
}

// ── PqCode ────────────────────────────────────────────────────────────────────

/// A product-quantized vector: one codeword index per subspace.
///
/// The [`codes`](Self::codes) length equals
/// [`num_subspaces`](PqConfig::num_subspaces).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PqCode {
    /// Per-subspace centroid indices.
    pub codes: Vec<u8>,
}

impl PqCode {
    /// Create a new code from raw centroid indices.
    #[must_use]
    pub fn new(codes: Vec<u8>) -> Self {
        Self { codes }
    }

    /// Number of subspaces (codewords) this code spans.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Return `true` when the code holds no subspaces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

// ── PqHit ─────────────────────────────────────────────────────────────────────

/// A single search result from a [`PqIndex`](super::PqIndex).
#[derive(Debug, Clone, PartialEq)]
pub struct PqHit {
    /// Identifier of the matched document.
    pub id: DocumentId,
    /// Asymmetric (squared-L2) distance to the query; smaller is closer.
    pub distance: f32,
}

impl PqHit {
    /// Create a new hit.
    #[must_use]
    pub fn new(id: DocumentId, distance: f32) -> Self {
        Self { id, distance }
    }
}

// ── PqError ───────────────────────────────────────────────────────────────────

/// Errors from the `product_quantization` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PqError {
    /// `dim` is not an integer multiple of `num_subspaces`.
    #[error("dim {dim} not divisible by num_subspaces {subspaces}")]
    InvalidConfig {
        /// The configured input dimensionality.
        dim: usize,
        /// The configured number of subspaces.
        subspaces: usize,
    },
    /// A vector did not match the configured dimensionality (or `codebook_bits`
    /// exceeded the supported maximum).
    #[error("vector dim mismatch")]
    DimMismatch,
    /// An operation requiring a trained quantizer was attempted before training.
    #[error("quantizer not trained")]
    NotTrained,
    /// Training was requested with an empty vector set.
    #[error("training set is empty")]
    EmptyTrainingSet,
}
