//! Types for the `hnsw_index` module.
//!
//! Defines [`HnswConfig`] (construction / search parameters), [`HnswHit`]
//! (a single ranked result), and [`HnswError`] (the error enumeration).

use thiserror::Error;

// ── HnswConfig ────────────────────────────────────────────────────────────────

/// Configuration for an [`HnswIndex`](crate::hnsw_index::HnswIndex).
///
/// HNSW builds a hierarchical proximity graph whose layers progressively
/// thin out. The key knobs are:
///
/// * `m` — max bidirectional neighbors per layer (accuracy ↔ memory trade-off).
/// * `ef_construction` — beam width while inserting (higher = better recall,
///   slower builds).
/// * `ef_search` — beam width while querying (higher = better recall, slower
///   queries).
/// * `max_layers` — maximum graph depth.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "hnsw")] {
/// use oxirag::hnsw_index::HnswConfig;
///
/// let cfg = HnswConfig::new()
///     .with_dim(64)
///     .with_m(16)
///     .with_ef_construction(200)
///     .with_ef_search(50)
///     .with_max_layers(6);
///
/// cfg.validate().unwrap();
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswConfig {
    /// Dimensionality of the indexed vectors.
    pub dim: usize,
    /// Maximum number of bidirectional neighbors per node per layer.
    ///
    /// Defaults to `16`.
    pub m: usize,
    /// Beam width used during insertion into the graph.
    ///
    /// Defaults to `200`.
    pub ef_construction: usize,
    /// Beam width used during similarity search.
    ///
    /// Defaults to `50`.
    pub ef_search: usize,
    /// Maximum number of graph layers.
    ///
    /// Defaults to `6`.
    pub max_layers: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_layers: 6,
        }
    }
}

impl HnswConfig {
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

    /// Set the maximum number of bidirectional neighbors (`M`).
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the beam width for index construction (`ef_construction`).
    #[must_use]
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the beam width for search (`ef_search`).
    #[must_use]
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Set the maximum number of graph layers.
    #[must_use]
    pub fn with_max_layers(mut self, max_layers: usize) -> Self {
        self.max_layers = max_layers;
        self
    }

    /// Validate the configuration, returning an error if any field is invalid.
    ///
    /// # Errors
    ///
    /// Returns [`HnswError::InvalidConfig`] if `dim`, `m`, `ef_construction`,
    /// `ef_search`, or `max_layers` is zero.
    pub fn validate(&self) -> Result<(), HnswError> {
        if self.dim == 0 {
            return Err(HnswError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }
        if self.m == 0 {
            return Err(HnswError::InvalidConfig(
                "m must be greater than zero".into(),
            ));
        }
        if self.ef_construction == 0 {
            return Err(HnswError::InvalidConfig(
                "ef_construction must be greater than zero".into(),
            ));
        }
        if self.ef_search == 0 {
            return Err(HnswError::InvalidConfig(
                "ef_search must be greater than zero".into(),
            ));
        }
        if self.max_layers == 0 {
            return Err(HnswError::InvalidConfig(
                "max_layers must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

// ── HnswHit ───────────────────────────────────────────────────────────────────

/// A single search result returned by
/// [`HnswIndex::search`](crate::hnsw_index::HnswIndex::search).
///
/// Hits are ranked by descending [`score`](HnswHit::score); the most similar
/// vector appears first. Scores are in `[0.0, 1.0]` where `1.0` is a perfect
/// cosine match.
#[derive(Debug, Clone, PartialEq)]
pub struct HnswHit {
    /// String identifier of the matched entry.
    pub id: String,
    /// Similarity score in `[0.0, 1.0]`: `(cosine + 1.0) / 2.0`.
    pub score: f32,
}

impl HnswHit {
    /// Create a new hit from a string identifier and a score.
    #[must_use]
    pub fn new(id: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
        }
    }
}

// ── HnswError ─────────────────────────────────────────────────────────────────

/// Errors produced by the `hnsw_index` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HnswError {
    /// A supplied vector did not match the configured dimensionality.
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// The dimension the index was configured with.
        expected: usize,
        /// The dimension of the supplied vector.
        got: usize,
    },
    /// Search was attempted on an empty index.
    #[error("index is empty")]
    EmptyIndex,
    /// A configuration field is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// The requested `k` is zero.
    #[error("k must be greater than zero")]
    InvalidK,
}
