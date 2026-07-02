//! Types for the `rp_tree_index` module.
//!
//! Defines the shared configuration ([`RpTreeConfig`]), the distance metric
//! ([`RpTreeMetric`]), the space-partitioning primitives
//! ([`RpTreeHyperplane`] and [`RpTreeNode`]), the ranked search result
//! ([`RpTreeHit`]), the error enumeration ([`RpTreeError`]) and the crate-wide
//! result alias ([`RpTreeResult`]).
//!
//! All numeric work is carried out in `f32`; there are no external RNG,
//! `ndarray`, or BLAS dependencies. Random splitting directions are derived
//! deterministically inside [`tree`](crate::rp_tree_index) from a seeded
//! `splitmix64` generator, so a fixed configuration always reproduces the same
//! forest.

use thiserror::Error;

// ── RpTreeMetric ────────────────────────────────────────────────────────────

/// Distance metric used both for the recursive space partition and for the
/// final exact re-ranking of candidate points.
///
/// * [`RpTreeMetric::Cosine`] — vectors are L2-normalised at build and query
///   time, so the cosine similarity reduces to a plain dot product and the
///   equidistant split hyperplane becomes the angular bisector of the two
///   sampled points.
/// * [`RpTreeMetric::L2`] — vectors are kept as-is; the split hyperplane is the
///   perpendicular bisector of the two sampled points and ranking uses squared
///   Euclidean distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RpTreeMetric {
    /// Cosine similarity over L2-normalised vectors (recommended default).
    #[default]
    Cosine,
    /// Euclidean (L2) distance over raw vectors.
    L2,
}

impl RpTreeMetric {
    /// Return `true` when this is the cosine metric.
    #[must_use]
    pub fn is_cosine(self) -> bool {
        matches!(self, Self::Cosine)
    }

    /// Return `true` when this is the L2 metric.
    #[must_use]
    pub fn is_l2(self) -> bool {
        matches!(self, Self::L2)
    }

    /// Pre-process a vector for storage / querying under this metric.
    ///
    /// For [`RpTreeMetric::Cosine`] the vector is L2-normalised to unit length;
    /// for [`RpTreeMetric::L2`] it is returned unchanged.
    #[must_use]
    pub(crate) fn preprocess(self, vector: Vec<f32>) -> Vec<f32> {
        match self {
            Self::Cosine => normalized(&vector),
            Self::L2 => vector,
        }
    }

    /// Compute the display similarity score in `[0, 1]` between two
    /// already-pre-processed vectors, where a larger value means *more*
    /// similar.
    ///
    /// The score is a strictly monotonic function of the underlying metric, so
    /// ranking candidates by descending score reproduces the exact metric
    /// ordering.
    ///
    /// * Cosine: `(dot(a, b) + 1) / 2`, clamped to `[0, 1]`.
    /// * L2: `1 / (1 + euclidean_distance(a, b))`, in `(0, 1]`.
    #[must_use]
    pub(crate) fn score(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => {
                let cos = dot(a, b).clamp(-1.0, 1.0);
                f32::midpoint(cos, 1.0).clamp(0.0, 1.0)
            }
            Self::L2 => {
                let dist = l2_squared(a, b).max(0.0).sqrt();
                1.0 / (1.0 + dist)
            }
        }
    }
}

// ── RpTreeConfig ────────────────────────────────────────────────────────────

/// Configuration for a random-projection tree forest.
///
/// The knobs trade recall against build/query cost:
///
/// * `n_trees` — number of independent trees in the forest. Each tree uses a
///   different derived seed and therefore a different sequence of split
///   directions, so their approximation errors are complementary; searching
///   more trees raises recall at a linear cost.
/// * `leaf_size` — recursion stops (a leaf is emitted) once a node holds at
///   most this many points. Smaller leaves partition space more finely.
/// * `search_k` — candidate budget: the priority-queue descent keeps expanding
///   nodes until this many distinct candidate points have been gathered (or the
///   queue empties). Larger budgets raise recall and cost. When `search_k` is
///   at least the number of indexed points the search degenerates to an exact
///   brute-force scan.
/// * `max_depth` — hard cap on recursion depth; a safety net that forces a leaf
///   even when a node still holds more than `leaf_size` points. Guarantees
///   termination and bounds stack usage.
/// * `seed` — root seed for the deterministic `splitmix64` generator.
/// * `metric` — [`RpTreeMetric`] used for splitting and exact re-ranking.
/// * `dim` — dimensionality every indexed / queried vector must have.
///
/// # Examples
///
/// ```
/// use oxirag::rp_tree_index::{RpTreeConfig, RpTreeMetric};
///
/// let cfg = RpTreeConfig::new()
///     .with_dim(16)
///     .with_n_trees(8)
///     .with_leaf_size(4)
///     .with_search_k(64)
///     .with_metric(RpTreeMetric::Cosine);
///
/// cfg.validate().expect("configuration is valid");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpTreeConfig {
    /// Number of independent trees in the forest. Defaults to `10`.
    pub n_trees: usize,
    /// Maximum number of points held by a leaf node. Defaults to `16`.
    pub leaf_size: usize,
    /// Candidate budget for a single search. Defaults to `100`.
    pub search_k: usize,
    /// Hard cap on recursion depth (termination safety net). Defaults to `32`.
    pub max_depth: usize,
    /// Root seed for the deterministic split-direction generator. Defaults to
    /// `0x_5EED_C0DE`.
    pub seed: u64,
    /// Distance metric for splitting and re-ranking. Defaults to
    /// [`RpTreeMetric::Cosine`].
    pub metric: RpTreeMetric,
    /// Dimensionality of the indexed vectors. Defaults to `128`.
    pub dim: usize,
}

impl Default for RpTreeConfig {
    fn default() -> Self {
        Self {
            n_trees: 10,
            leaf_size: 16,
            search_k: 100,
            max_depth: 32,
            seed: 0x_5EED_C0DE,
            metric: RpTreeMetric::Cosine,
            dim: 128,
        }
    }
}

impl RpTreeConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of trees in the forest.
    #[must_use]
    pub fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }

    /// Set the maximum number of points held by a leaf node.
    #[must_use]
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size;
        self
    }

    /// Set the per-query candidate budget.
    #[must_use]
    pub fn with_search_k(mut self, search_k: usize) -> Self {
        self.search_k = search_k;
        self
    }

    /// Set the hard recursion-depth cap.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the root seed for the deterministic split generator.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the distance metric.
    #[must_use]
    pub fn with_metric(mut self, metric: RpTreeMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the vector dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RpTreeError::InvalidConfig`] when `dim`, `n_trees`,
    /// `leaf_size`, `search_k`, or `max_depth` is zero.
    pub fn validate(&self) -> Result<(), RpTreeError> {
        if self.dim == 0 {
            return Err(RpTreeError::InvalidConfig("dim must be > 0".into()));
        }
        if self.n_trees == 0 {
            return Err(RpTreeError::InvalidConfig("n_trees must be > 0".into()));
        }
        if self.leaf_size == 0 {
            return Err(RpTreeError::InvalidConfig("leaf_size must be > 0".into()));
        }
        if self.search_k == 0 {
            return Err(RpTreeError::InvalidConfig("search_k must be > 0".into()));
        }
        if self.max_depth == 0 {
            return Err(RpTreeError::InvalidConfig("max_depth must be > 0".into()));
        }
        Ok(())
    }
}

// ── RpTreeHyperplane ────────────────────────────────────────────────────────

/// A splitting hyperplane `{ x : dot(normal, x) = bias }`.
///
/// Built as the *equidistant* hyperplane of two sampled points `p` and `q`: the
/// `normal` is `p - q` and the `bias` is `dot(normal, (p + q) / 2)`, i.e. the
/// locus of points equidistant (in the configured metric's underlying inner
/// product) from `p` and `q`.
///
/// The *degenerate* hyperplane (all-zero `normal`, zero `bias`) is produced only
/// as a last-resort fallback when a node's points cannot be separated
/// geometrically (for example, all points are identical); every query then has
/// [`margin`](RpTreeHyperplane::margin) exactly `0`, so both children are
/// explored with equal priority.
#[derive(Debug, Clone, PartialEq)]
pub struct RpTreeHyperplane {
    /// Hyperplane normal direction (`p - q` for sampled points `p`, `q`).
    pub normal: Vec<f32>,
    /// Offset such that the plane is `{ x : dot(normal, x) = bias }`.
    pub bias: f32,
}

impl RpTreeHyperplane {
    /// Construct a hyperplane from an explicit `normal` and `bias`.
    #[must_use]
    pub fn new(normal: Vec<f32>, bias: f32) -> Self {
        Self { normal, bias }
    }

    /// Construct the degenerate zero hyperplane of dimensionality `dim`.
    ///
    /// Every point has a margin of exactly `0` against this hyperplane.
    #[must_use]
    pub fn degenerate(dim: usize) -> Self {
        Self {
            normal: vec![0.0; dim],
            bias: 0.0,
        }
    }

    /// Signed distance functional `dot(normal, vector) - bias`.
    ///
    /// A non-negative margin routes a point to the *right* child; a negative
    /// margin routes it to the *left* child. The magnitude measures how far the
    /// point sits from the boundary and drives the priority-queue back-tracking
    /// during search.
    #[must_use]
    pub fn margin(&self, vector: &[f32]) -> f32 {
        dot(&self.normal, vector) - self.bias
    }
}

// ── RpTreeNode ──────────────────────────────────────────────────────────────

/// A single node of a random-projection tree, stored inside a flat arena.
///
/// A node is either a [`RpTreeNode::Leaf`] holding the internal indices of the
/// points that fell into it, or a [`RpTreeNode::Internal`] holding a splitting
/// [`RpTreeHyperplane`] and the arena indices of its two children.
///
/// The `usize` identifiers are *internal* point indices (positions within the
/// forest's point store), not the caller-facing string ids.
#[derive(Debug, Clone, PartialEq)]
pub enum RpTreeNode {
    /// A leaf node holding the internal indices of its member points.
    Leaf {
        /// Internal point indices contained in this leaf.
        ids: Vec<usize>,
    },
    /// An internal node that partitions its points with a hyperplane.
    Internal {
        /// The splitting hyperplane; points with a negative margin descend
        /// left, points with a non-negative margin descend right.
        hyperplane: RpTreeHyperplane,
        /// Arena index of the left child (negative-margin side).
        left: usize,
        /// Arena index of the right child (non-negative-margin side).
        right: usize,
    },
}

impl RpTreeNode {
    /// Return `true` when this node is a leaf.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    /// Number of points held directly by this node when it is a leaf, or `0`
    /// for an internal node.
    #[must_use]
    pub fn leaf_len(&self) -> usize {
        match self {
            Self::Leaf { ids } => ids.len(),
            Self::Internal { .. } => 0,
        }
    }
}

// ── RpTreeHit ───────────────────────────────────────────────────────────────

/// A single ranked search result.
///
/// Hits are returned in descending [`score`](RpTreeHit::score) order (most
/// similar first). Scores lie in `[0, 1]`, where `1.0` is a perfect match.
#[derive(Debug, Clone, PartialEq)]
pub struct RpTreeHit {
    /// Caller-supplied identifier of the matched point.
    pub id: String,
    /// Similarity score in `[0, 1]`; higher is more similar.
    pub score: f32,
}

impl RpTreeHit {
    /// Create a new hit.
    #[must_use]
    pub fn new(id: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
        }
    }
}

// ── RpTreeError / RpTreeResult ──────────────────────────────────────────────

/// Errors produced by the `rp_tree_index` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RpTreeError {
    /// A supplied vector did not match the configured dimensionality.
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// Expected dimensionality from [`RpTreeConfig::dim`].
        expected: usize,
        /// Actual length of the offending vector.
        got: usize,
    },

    /// A forest build was attempted with no points.
    #[error("cannot build an index from an empty point set")]
    EmptyPoints,

    /// The index was queried before it held any points.
    #[error("index is empty")]
    EmptyIndex,

    /// A `k` of zero was passed to `search`.
    #[error("k must be >= 1")]
    InvalidK,

    /// The configuration contained an invalid value.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Convenience result alias for the `rp_tree_index` module.
pub type RpTreeResult<T> = Result<T, RpTreeError>;

// ── vector primitives ───────────────────────────────────────────────────────

/// Dot product of two equal-length vectors.
///
/// Pairs are consumed by [`Iterator::zip`], so any trailing elements of the
/// longer slice are ignored; callers always pass equal-length vectors.
#[must_use]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared Euclidean distance between two equal-length vectors.
#[must_use]
pub(crate) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Return an L2-normalised copy of `vector` (unit length, guarded against a
/// zero-norm input).
#[must_use]
pub(crate) fn normalized(vector: &[f32]) -> Vec<f32> {
    let norm = dot(vector, vector).sqrt().max(1e-9);
    vector.iter().map(|x| x / norm).collect()
}
