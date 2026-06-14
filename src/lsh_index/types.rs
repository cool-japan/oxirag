//! Types for the `lsh_index` module.
//!
//! Defines the shared configuration ([`LshConfig`]), the search-result record
//! ([`LshHit`]) and the error enumeration ([`LshError`]) used throughout both
//! the random-hyperplane (cosine) and `MinHash` (Jaccard) LSH indexes.

use thiserror::Error;

// ── LshConfig ─────────────────────────────────────────────────────────────────

/// Configuration shared by all LSH index variants.
///
/// # Random-hyperplane (dense / cosine) index
///
/// Each indexed vector is projected onto `num_planes` random unit hyperplanes,
/// and the sign of each projection becomes one bit of the hash signature.
/// Vectors that share many sign-bits are candidates for cosine-similarity
/// re-ranking.
///
/// # `MinHash` (sparse / Jaccard) index
///
/// `num_bands × rows_per_band` independent min-hash functions are used.
/// The `rows_per_band` hashes within each band are concatenated and mapped to
/// a bucket; two sets that collide in at least one band are candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LshConfig {
    /// Dimensionality of the indexed dense vectors.
    ///
    /// Ignored by the `MinHash` index.  Defaults to `128`.
    pub dim: usize,

    /// Number of random hyperplanes for the dense cosine index.
    ///
    /// More planes → finer discrimination, but larger signatures.
    /// Defaults to `16`.
    pub num_planes: usize,

    /// Number of banding groups used by the `MinHash` index.
    ///
    /// Defaults to `4`.
    pub num_bands: usize,

    /// Number of hash rows per band in the `MinHash` index.
    ///
    /// Total min-hash functions = `num_bands × rows_per_band`.
    /// Defaults to `4`.
    pub rows_per_band: usize,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            num_planes: 16,
            num_bands: 4,
            rows_per_band: 4,
        }
    }
}

impl LshConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vector dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the number of random hyperplanes.
    #[must_use]
    pub fn with_num_planes(mut self, num_planes: usize) -> Self {
        self.num_planes = num_planes;
        self
    }

    /// Set the number of `MinHash` bands.
    #[must_use]
    pub fn with_num_bands(mut self, num_bands: usize) -> Self {
        self.num_bands = num_bands;
        self
    }

    /// Set the number of hash rows per `MinHash` band.
    #[must_use]
    pub fn with_rows_per_band(mut self, rows_per_band: usize) -> Self {
        self.rows_per_band = rows_per_band;
        self
    }

    /// Total number of min-hash functions (`num_bands × rows_per_band`).
    #[must_use]
    pub fn total_minhash_funcs(&self) -> usize {
        self.num_bands.saturating_mul(self.rows_per_band)
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LshError::InvalidConfig`] when `dim` or `num_planes` is zero,
    /// or when `num_bands` or `rows_per_band` is zero.
    pub fn validate(&self) -> Result<(), LshError> {
        if self.dim == 0 {
            return Err(LshError::InvalidConfig("dim must be > 0".into()));
        }
        if self.num_planes == 0 {
            return Err(LshError::InvalidConfig("num_planes must be > 0".into()));
        }
        if self.num_bands == 0 {
            return Err(LshError::InvalidConfig("num_bands must be > 0".into()));
        }
        if self.rows_per_band == 0 {
            return Err(LshError::InvalidConfig("rows_per_band must be > 0".into()));
        }
        Ok(())
    }
}

// ── LshHit ────────────────────────────────────────────────────────────────────

/// A single search result returned by an LSH index search.
///
/// For the dense random-hyperplane index the [`score`](Self::score) is the
/// exact cosine similarity `∈ [-1, 1]` recomputed over candidates; it is
/// normalised to `[0, 1]` by `(cosine + 1) / 2` before being stored.
///
/// For the `MinHash` index the score is the estimated Jaccard similarity
/// `∈ [0, 1]`.
///
/// Hits are returned in descending score order (highest similarity first).
#[derive(Debug, Clone, PartialEq)]
pub struct LshHit {
    /// Identifier of the matched document.
    pub id: String,
    /// Similarity score in `[0, 1]`; higher is more similar.
    pub score: f32,
}

impl LshHit {
    /// Create a new hit.
    #[must_use]
    pub fn new(id: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
        }
    }
}

// ── LshError ──────────────────────────────────────────────────────────────────

/// Errors produced by the `lsh_index` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LshError {
    /// A supplied vector did not match the configured dimensionality.
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// Expected dimensionality from [`LshConfig::dim`].
        expected: usize,
        /// Actual length of the supplied vector.
        got: usize,
    },

    /// The index was queried before any vectors were inserted.
    #[error("index is empty")]
    EmptyIndex,

    /// The configuration contains an invalid value.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// A requested `k` of zero was passed to `search`.
    #[error("k must be >= 1")]
    InvalidK,
}
