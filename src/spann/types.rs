//! Core types for the `spann` module: configuration, metric, posting lists,
//! search hits, index statistics and the error enumeration.
//!
//! SPANN (Chen et al., `NeurIPS` 2021, "SPANN: Highly-efficient Billion-scale
//! Approximate Nearest Neighbour Search") stores a small set of *centroids*
//! (the "head", kept in memory) together with per-centroid *posting lists*
//! (the "tail") that hold the member identifiers routed to each centroid.
//! This module defines the data that flows between the clustering stage
//! ([`crate::spann::cluster`]) and the index ([`crate::spann::index`]).

use thiserror::Error;

// ── SpannMetric ───────────────────────────────────────────────────────────────

/// Distance/similarity metric used throughout a
/// [`SpannIndex`](crate::spann::SpannIndex).
///
/// Internally every metric is converted to a *distance* where a smaller
/// value means "closer" — this lets clustering, boundary-closure
/// replication and the RNG pruning rule share one implementation across all
/// three metrics. Search results, however, report a *score* where a larger
/// value means "more similar", following the convention already used by
/// [`crate::layer1_echo::traits::SimilarityMetric`].
///
/// | Metric | Internal distance (smaller = closer) | Reported [`SpannHit::score`] (larger = better) |
/// |--------|----------------------------------------|--------------------------------------------------|
/// | [`Cosine`](SpannMetric::Cosine) (default) | `1.0 - cosine_similarity(a, b)` | raw cosine similarity in `[-1.0, 1.0]` |
/// | [`L2`](SpannMetric::L2) | Euclidean distance | `1.0 / (1.0 + distance)` |
/// | [`Dot`](SpannMetric::Dot) | negated dot product | raw dot product |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpannMetric {
    /// Cosine similarity: measures the angle between vectors, ignoring
    /// magnitude.
    #[default]
    Cosine,
    /// Euclidean (L2) distance.
    L2,
    /// Raw dot product. Requires normalised vectors for a meaningful
    /// magnitude-independent comparison.
    Dot,
}

// ── SpannConfig ───────────────────────────────────────────────────────────────

/// Configuration for a [`SpannIndex`](crate::spann::SpannIndex).
///
/// SPANN partitions the corpus into `num_postings` balanced clusters (the
/// centroid "head"), replicates boundary points into every centroid within
/// `boundary_epsilon` of their nearest centroid (capped at
/// `replica_count`), then prunes redundant replicas with the
/// relative-neighbourhood-graph (RNG) rule. At query time the `nprobe`
/// nearest centroids are scanned.
#[derive(Debug, Clone, PartialEq)]
pub struct SpannConfig {
    /// Number of centroids ("head" size) to build during clustering.
    ///
    /// Defaults to `16`. Clamped to the number of indexed points if larger.
    pub num_postings: usize,
    /// Maximum number of posting lists a single point may be replicated
    /// into during boundary-closure replication.
    ///
    /// Defaults to `8`.
    pub replica_count: usize,
    /// Relative slack used by boundary-closure replication: a point `x` is
    /// also assigned to centroid `c` (beyond its nearest centroid) whenever
    /// `dist(x, c) <= dist(x, nearest) + boundary_epsilon * |dist(x, nearest)|`
    /// — equivalently `(1.0 + boundary_epsilon) * dist(x, nearest)` for the
    /// non-negative distances used by [`SpannMetric::L2`] and
    /// [`SpannMetric::Cosine`]. The magnitude-scaled form keeps the slack a
    /// genuine *loosening* even for [`SpannMetric::Dot`], whose internal
    /// distance (a negated dot product) can be negative.
    ///
    /// Defaults to `0.1`. Must be finite and non-negative.
    pub boundary_epsilon: f32,
    /// Soft cap on the number of *primary* members a posting list may hold
    /// after balanced clustering. Clusters that exceed this are recursively
    /// re-clustered into two, so no primary assignment is oversized. Note
    /// that boundary-closure replication may still add replicas beyond this
    /// limit afterwards — that inflation is inherent to SPANN and is not
    /// itself re-balanced.
    ///
    /// Defaults to `100`.
    pub posting_limit: usize,
    /// Number of nearest centroids probed at search time.
    ///
    /// Defaults to `8`. Clamped to `num_postings` if larger.
    pub nprobe: usize,
    /// Maximum number of Lloyd k-means iterations run per clustering call
    /// (both the initial balanced clustering and every re-cluster-into-two
    /// split).
    ///
    /// Defaults to `25`.
    pub max_iters: usize,
    /// Distance/similarity metric used for clustering, replication and
    /// search.
    ///
    /// Defaults to [`SpannMetric::Cosine`].
    pub metric: SpannMetric,
}

impl Default for SpannConfig {
    fn default() -> Self {
        Self {
            num_postings: 16,
            replica_count: 8,
            boundary_epsilon: 0.1,
            posting_limit: 100,
            nprobe: 8,
            max_iters: 25,
            metric: SpannMetric::Cosine,
        }
    }
}

impl SpannConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of centroids to build during clustering.
    #[must_use]
    pub fn with_num_postings(mut self, num_postings: usize) -> Self {
        self.num_postings = num_postings;
        self
    }

    /// Set the maximum boundary-closure replica count per point.
    #[must_use]
    pub fn with_replica_count(mut self, replica_count: usize) -> Self {
        self.replica_count = replica_count;
        self
    }

    /// Set the boundary-closure relative slack `epsilon`.
    #[must_use]
    pub fn with_boundary_epsilon(mut self, boundary_epsilon: f32) -> Self {
        self.boundary_epsilon = boundary_epsilon;
        self
    }

    /// Set the soft posting-list size limit enforced during balanced
    /// clustering.
    #[must_use]
    pub fn with_posting_limit(mut self, posting_limit: usize) -> Self {
        self.posting_limit = posting_limit;
        self
    }

    /// Set the number of nearest centroids probed at search time.
    #[must_use]
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the maximum number of k-means iterations per clustering call.
    #[must_use]
    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Set the distance/similarity metric.
    #[must_use]
    pub fn with_metric(mut self, metric: SpannMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Validate the configuration, returning an error if any field is out
    /// of range.
    ///
    /// # Errors
    ///
    /// Returns [`SpannError::InvalidConfig`] when `num_postings`,
    /// `replica_count`, `posting_limit`, `nprobe` or `max_iters` is zero, or
    /// when `boundary_epsilon` is negative or non-finite.
    pub fn validate(&self) -> Result<(), SpannError> {
        if self.num_postings == 0 {
            return Err(SpannError::InvalidConfig(
                "num_postings must be greater than zero".to_string(),
            ));
        }
        if self.replica_count == 0 {
            return Err(SpannError::InvalidConfig(
                "replica_count must be greater than zero".to_string(),
            ));
        }
        if self.posting_limit == 0 {
            return Err(SpannError::InvalidConfig(
                "posting_limit must be greater than zero".to_string(),
            ));
        }
        if self.nprobe == 0 {
            return Err(SpannError::InvalidConfig(
                "nprobe must be greater than zero".to_string(),
            ));
        }
        if self.max_iters == 0 {
            return Err(SpannError::InvalidConfig(
                "max_iters must be greater than zero".to_string(),
            ));
        }
        if !self.boundary_epsilon.is_finite() || self.boundary_epsilon < 0.0 {
            return Err(SpannError::InvalidConfig(
                "boundary_epsilon must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

// ── Posting ───────────────────────────────────────────────────────────────────

/// A single posting list: one centroid of the SPANN "head" together with
/// the identifiers of every point (the "tail") routed to it, either as a
/// primary balanced-clustering assignment or as a boundary-closure replica.
///
/// A point's identifier may appear in more than one [`Posting`] when
/// boundary-closure replication assigned it to several nearby centroids.
#[derive(Debug, Clone, PartialEq)]
pub struct Posting {
    /// The centroid vector for this posting list.
    pub centroid: Vec<f32>,
    /// Identifiers of every member (primary or replicated) routed to this
    /// centroid.
    pub member_ids: Vec<String>,
}

// ── SpannHit ──────────────────────────────────────────────────────────────────

/// A single search result returned by
/// [`SpannIndex::search`](crate::spann::SpannIndex::search).
///
/// Hits are ranked by descending [`score`](SpannHit::score); the most
/// similar point appears first. The scale of `score` depends on the
/// index's configured [`SpannMetric`] — see the metric's documentation.
#[derive(Debug, Clone, PartialEq)]
pub struct SpannHit {
    /// Identifier of the matched point.
    pub id: String,
    /// Similarity score; higher is more similar. Scale depends on
    /// [`SpannMetric`].
    pub score: f32,
}

// ── SpannStats ────────────────────────────────────────────────────────────────

/// Summary statistics describing the posting lists of a
/// [`SpannIndex`](crate::spann::SpannIndex), useful for diagnosing
/// clustering balance and replication overhead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpannStats {
    /// Number of posting lists (centroids) in the index.
    pub num_postings: usize,
    /// Total number of `(point, posting)` memberships across every posting
    /// list, i.e. the sum of all posting-list lengths. Always
    /// `>= number of indexed points` because of boundary-closure
    /// replication.
    pub total_replicas: usize,
    /// Average posting-list length (`total_replicas / num_postings`).
    pub avg_posting_len: f32,
    /// Length of the largest posting list.
    pub max_posting_len: usize,
    /// Length of the smallest posting list.
    pub min_posting_len: usize,
}

// ── SpannError ────────────────────────────────────────────────────────────────

/// Errors produced by the `spann` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SpannError {
    /// An operation requiring indexed points was attempted on an empty
    /// index (or [`SpannIndex::build`](crate::spann::SpannIndex::build) was
    /// called with no items).
    #[error("index is empty")]
    EmptyIndex,
    /// A supplied vector did not match the index's dimensionality.
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimensionality the index was built with.
        expected: usize,
        /// The dimensionality of the offending vector.
        got: usize,
    },
    /// A configuration field is out of range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// The supplied query vector was empty while the index expects a
    /// non-zero dimensionality.
    #[error("query is empty")]
    EmptyQuery,
}
