//! Types for the `residual_vq` module: configuration, codes, hits and errors.

use thiserror::Error;

use crate::types::DocumentId;

// ── ResidualVqConfig ────────────────────────────────────────────────────────────

/// Configuration for a [`ResidualQuantizer`](super::ResidualQuantizer).
///
/// Residual (multi-stage) vector quantization encodes a full `dim`-dimensional
/// vector through a cascade of `num_stages` codebooks applied **sequentially**.
/// Each stage holds `codebook_size` codewords and quantizes the *residual* left
/// over by every preceding stage, so the reconstruction is the additive sum of
/// one codeword per stage. This contrasts with product quantization, which
/// splits a vector into independent subspaces quantized in parallel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualVqConfig {
    /// Number of sequential quantization stages (`M`), one codebook per stage.
    ///
    /// Defaults to `4`. With `num_stages == 1` the quantizer degenerates to a
    /// single plain k-means codebook.
    pub num_stages: usize,
    /// Number of codewords per stage codebook (`K`).
    ///
    /// Defaults to `256`.
    pub codebook_size: usize,
    /// Dimensionality of the input vectors.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Maximum number of Lloyd (k-means) iterations run per stage.
    ///
    /// Defaults to `10`. Internally treated as at least `1` so that stage
    /// centroids are always cluster means.
    pub max_kmeans_iterations: usize,
    /// Beam width used by [`encode`](super::ResidualQuantizer::encode).
    ///
    /// Defaults to `1`, which is plain greedy per-stage assignment. A value
    /// greater than `1` keeps that many partial-residual candidates at each
    /// stage, recovering a lower joint reconstruction error than greedy.
    pub beam_width: usize,
    /// Seed for the deterministic FNV-1a-based k-means centroid initialization.
    ///
    /// Defaults to a fixed constant so that training is fully reproducible; no
    /// `rand` crate is used anywhere in this module.
    pub seed: u64,
}

impl Default for ResidualVqConfig {
    fn default() -> Self {
        Self {
            num_stages: 4,
            codebook_size: 256,
            dim: 128,
            max_kmeans_iterations: 10,
            beam_width: 1,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

impl ResidualVqConfig {
    /// Create a new configuration with the [default](Self::default) values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of sequential stages (`M`).
    #[must_use]
    pub fn with_num_stages(mut self, num_stages: usize) -> Self {
        self.num_stages = num_stages;
        self
    }

    /// Set the number of codewords per stage (`K`).
    #[must_use]
    pub fn with_codebook_size(mut self, codebook_size: usize) -> Self {
        self.codebook_size = codebook_size;
        self
    }

    /// Set the input vector dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the maximum number of k-means iterations per stage.
    #[must_use]
    pub fn with_max_kmeans_iterations(mut self, max_kmeans_iterations: usize) -> Self {
        self.max_kmeans_iterations = max_kmeans_iterations;
        self
    }

    /// Set the encode-time beam width (`1` = greedy).
    #[must_use]
    pub fn with_beam_width(mut self, beam_width: usize) -> Self {
        self.beam_width = beam_width;
        self
    }

    /// Set the deterministic seed for centroid initialization.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ResidualVqError::InvalidConfig`] when `num_stages`,
    /// `codebook_size`, `dim` or `beam_width` is `0`.
    pub fn validate(&self) -> Result<(), ResidualVqError> {
        if self.num_stages == 0 {
            return Err(ResidualVqError::InvalidConfig {
                reason: "num_stages must be greater than zero",
            });
        }
        if self.codebook_size == 0 {
            return Err(ResidualVqError::InvalidConfig {
                reason: "codebook_size must be greater than zero",
            });
        }
        if self.dim == 0 {
            return Err(ResidualVqError::InvalidConfig {
                reason: "dim must be greater than zero",
            });
        }
        if self.beam_width == 0 {
            return Err(ResidualVqError::InvalidConfig {
                reason: "beam_width must be greater than zero",
            });
        }
        Ok(())
    }
}

// ── ResidualCode ────────────────────────────────────────────────────────────────

/// A residual-quantized vector: one codeword index per stage.
///
/// The [`indices`](Self::indices) length equals
/// [`num_stages`](ResidualVqConfig::num_stages), and entry `m` selects a
/// codeword from stage `m`'s codebook. Reconstruction sums the selected
/// codewords across all stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualCode {
    /// Per-stage codeword indices (`indices[m]` selects from stage `m`).
    pub indices: Vec<usize>,
}

impl ResidualCode {
    /// Create a new code from raw per-stage codeword indices.
    #[must_use]
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }

    /// Number of stages (codewords) this code spans.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Return `true` when the code holds no stages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

// ── ResidualVqHit ───────────────────────────────────────────────────────────────

/// A single search result from a [`ResidualVqIndex`](super::ResidualVqIndex).
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualVqHit {
    /// Identifier of the matched document.
    pub id: DocumentId,
    /// Estimated (squared-L2) distance to the query; smaller is closer.
    pub distance: f32,
}

impl ResidualVqHit {
    /// Create a new hit.
    #[must_use]
    pub fn new(id: DocumentId, distance: f32) -> Self {
        Self { id, distance }
    }
}

// ── ResidualVqError ─────────────────────────────────────────────────────────────

/// Errors from the `residual_vq` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ResidualVqError {
    /// A configuration field held an invalid value.
    #[error("invalid residual-vq configuration: {reason}")]
    InvalidConfig {
        /// Human-readable reason the configuration was rejected.
        reason: &'static str,
    },
    /// A vector length did not match the configured dimensionality.
    #[error("vector dim mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// The configured (expected) dimensionality.
        expected: usize,
        /// The dimensionality actually supplied.
        got: usize,
    },
    /// An operation requiring a trained quantizer ran before training.
    #[error("quantizer not trained")]
    NotTrained,
    /// Training (or building an index) was requested with an empty vector set.
    #[error("training set is empty")]
    EmptyTrainingSet,
}
