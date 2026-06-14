//! Types for the `ivf_index` module.
//!
//! Defines the public configuration ([`IvfConfig`]), the search-result record
//! ([`IvfHit`]) and the error enumeration ([`IvfError`]) used throughout the
//! Inverted-File ANN index.

use thiserror::Error;

// ── IvfConfig ─────────────────────────────────────────────────────────────────

/// Configuration for an [`IvfIndex`](crate::ivf_index::IvfIndex).
///
/// An IVF (Inverted File) index partitions the vector space into `num_cells`
/// Voronoi regions whose centroids form a *coarse quantizer*. At search time the
/// `nprobe` nearest centroids are located and only their inverted lists are
/// scanned, trading a small amount of recall for a large speed-up over an
/// exhaustive scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfConfig {
    /// Number of coarse-quantizer cells (k-means centroids).
    ///
    /// Defaults to `16`.
    pub num_cells: usize,
    /// Number of nearest cells to scan at query time.
    ///
    /// Defaults to `4`. Setting `nprobe == num_cells` makes the search exact.
    pub nprobe: usize,
    /// Dimensionality of the indexed vectors.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Maximum number of Lloyd iterations run while training the quantizer.
    ///
    /// Defaults to `10`.
    pub kmeans_iters: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            num_cells: 16,
            nprobe: 4,
            dim: 128,
            kmeans_iters: 10,
        }
    }
}

impl IvfConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of coarse-quantizer cells.
    #[must_use]
    pub fn with_num_cells(mut self, num_cells: usize) -> Self {
        self.num_cells = num_cells;
        self
    }

    /// Set the number of nearest cells to scan at query time.
    #[must_use]
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the dimensionality of the indexed vectors.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the maximum number of k-means (Lloyd) iterations used in training.
    #[must_use]
    pub fn with_kmeans_iters(mut self, kmeans_iters: usize) -> Self {
        self.kmeans_iters = kmeans_iters;
        self
    }
}

// ── IvfHit ────────────────────────────────────────────────────────────────────

/// A single search result returned by
/// [`IvfIndex::search`](crate::ivf_index::IvfIndex::search).
///
/// Hits are ranked by ascending [`distance`](IvfHit::distance), so the most
/// similar vector appears first.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfHit {
    /// Identifier of the matched document.
    pub id: crate::types::DocumentId,
    /// Squared L2 distance between the query and the matched vector.
    pub distance: f32,
}

impl IvfHit {
    /// Create a new hit from an identifier and a distance.
    #[must_use]
    pub fn new(id: crate::types::DocumentId, distance: f32) -> Self {
        Self { id, distance }
    }
}

// ── IvfError ──────────────────────────────────────────────────────────────────

/// Errors produced by the `ivf_index` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum IvfError {
    /// A supplied vector did not match the configured dimensionality.
    #[error("vector dim mismatch")]
    DimMismatch,
    /// An operation requiring a trained quantizer was attempted before training.
    #[error("index not trained")]
    NotTrained,
    /// Training was requested with an empty vector set.
    #[error("training set is empty")]
    EmptyTrainingSet,
}
