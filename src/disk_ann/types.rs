//! Types for the `disk_ann` module.
//!
//! Defines [`DiskAnnConfig`] (Vamana build/search parameters), [`DiskAnnMetric`]
//! (the distance/similarity function used throughout the graph), [`DiskAnnHit`]
//! (a single ranked result), and [`DiskAnnError`] (the error enumeration).

use thiserror::Error;

// ── DiskAnnMetric ─────────────────────────────────────────────────────────────

/// The distance/similarity function used to build and query a
/// [`DiskAnnIndex`](crate::disk_ann::DiskAnnIndex).
///
/// # Distance vs. score convention
///
/// Internally, the Vamana graph algorithms (`GreedySearch`, `RobustPrune`)
/// always operate on a **distance** where *smaller is closer*:
///
/// | Metric | Internal distance |
/// |--------|--------------------|
/// | [`Cosine`](DiskAnnMetric::Cosine) | `1.0 - cosine_similarity(a, b)`, in `[0.0, 2.0]` |
/// | [`L2`](DiskAnnMetric::L2) | Euclidean distance `\|\|a - b\|\|` |
/// | [`Dot`](DiskAnnMetric::Dot) | negated dot product `-(a · b)` |
///
/// Externally, [`DiskAnnHit::score`] always follows the opposite convention —
/// **larger is better** — so results are always sorted best-first regardless
/// of metric:
///
/// | Metric | Reported score |
/// |--------|-----------------|
/// | [`Cosine`](DiskAnnMetric::Cosine) | raw cosine similarity, in `[-1.0, 1.0]` |
/// | [`L2`](DiskAnnMetric::L2) | negated Euclidean distance `-\|\|a - b\|\|` (closer ⇒ larger, less-negative score) |
/// | [`Dot`](DiskAnnMetric::Dot) | raw dot product |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiskAnnMetric {
    /// Cosine similarity, normalising both vectors before comparison.
    ///
    /// This is the default metric.
    #[default]
    Cosine,
    /// Euclidean (L2) distance.
    L2,
    /// Raw (unnormalised) dot product, useful for pre-normalised embeddings
    /// or maximum-inner-product search (MIPS).
    Dot,
}

/// Dot product of two equal-length vectors.
///
/// Callers are expected to supply equal-length slices; a length mismatch
/// silently truncates to the shorter of the two (this never occurs on the
/// validated paths inside this module, since every vector is checked against
/// [`DiskAnnConfig::dim`]-equivalent lengths before reaching this function).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm (`\|\|a\|\|`) of a vector.
fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Cosine similarity between `a` and `b`, clamped to `[-1.0, 1.0]`.
///
/// Returns `0.0` if either vector has (numerically) zero norm, avoiding a
/// division by zero.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let denom = norm(a) * norm(b);
    if denom <= f32::EPSILON {
        0.0
    } else {
        (dot(a, b) / denom).clamp(-1.0, 1.0)
    }
}

/// Euclidean (L2) distance between `a` and `b`.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

impl DiskAnnMetric {
    /// The internal Vamana **distance** between `a` and `b`: smaller means closer.
    ///
    /// See the [`DiskAnnMetric`] documentation for the exact formula used by
    /// each variant.
    #[must_use]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => 1.0 - cosine_similarity(a, b),
            Self::L2 => l2_distance(a, b),
            Self::Dot => -dot(a, b),
        }
    }

    /// The externally reported **score** between `a` and `b`: larger means better.
    ///
    /// See the [`DiskAnnMetric`] documentation for the exact formula used by
    /// each variant.
    #[must_use]
    pub fn score(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => cosine_similarity(a, b),
            Self::L2 => -l2_distance(a, b),
            Self::Dot => dot(a, b),
        }
    }
}

// ── DiskAnnConfig ─────────────────────────────────────────────────────────────

/// Configuration for a [`DiskAnnIndex`](crate::disk_ann::DiskAnnIndex).
///
/// `DiskANN` builds a single-layer Vamana proximity graph. The key knobs are:
///
/// * `max_degree` (`R`) — maximum out-degree per node (accuracy ↔ memory trade-off).
/// * `search_list_size` (`L`) — beam width used both while building (`GreedySearch`
///   during construction) and while querying.
/// * `alpha` — the `RobustPrune` occlusion factor. Values `> 1.0` allow a few
///   longer-range edges to survive pruning, shrinking the graph's diameter and
///   improving recall at the cost of a slightly denser graph.
/// * `metric` — the distance/similarity function; see [`DiskAnnMetric`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "disk-ann")] {
/// use oxirag::disk_ann::{DiskAnnConfig, DiskAnnMetric};
///
/// let cfg = DiskAnnConfig::new()
///     .with_max_degree(16)
///     .with_search_list_size(32)
///     .with_alpha(1.2)
///     .with_metric(DiskAnnMetric::L2);
///
/// cfg.validate().unwrap();
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskAnnConfig {
    /// Maximum out-degree (`R`) of every node in the Vamana graph.
    ///
    /// Defaults to `32`.
    pub max_degree: usize,
    /// Beam width (`L`) used by `GreedySearch`, both during construction and
    /// during querying.
    ///
    /// Defaults to `64`.
    pub search_list_size: usize,
    /// The `RobustPrune` occlusion factor (`alpha`). Must be `>= 1.0`.
    ///
    /// Defaults to `1.2`.
    pub alpha: f32,
    /// The distance/similarity function used throughout the index.
    ///
    /// Defaults to [`DiskAnnMetric::Cosine`].
    pub metric: DiskAnnMetric,
}

impl Default for DiskAnnConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,
            search_list_size: 64,
            alpha: 1.2,
            metric: DiskAnnMetric::Cosine,
        }
    }
}

impl DiskAnnConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum out-degree (`R`).
    #[must_use]
    pub fn with_max_degree(mut self, max_degree: usize) -> Self {
        self.max_degree = max_degree;
        self
    }

    /// Set the `GreedySearch` beam width (`L`).
    #[must_use]
    pub fn with_search_list_size(mut self, search_list_size: usize) -> Self {
        self.search_list_size = search_list_size;
        self
    }

    /// Set the `RobustPrune` occlusion factor (`alpha`).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the distance/similarity metric.
    #[must_use]
    pub fn with_metric(mut self, metric: DiskAnnMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Validate the configuration, returning an error if any field is invalid.
    ///
    /// # Errors
    ///
    /// Returns [`DiskAnnError::InvalidConfig`] if `max_degree` or
    /// `search_list_size` is zero, or if `alpha` is not finite or `< 1.0`.
    pub fn validate(&self) -> Result<(), DiskAnnError> {
        if self.max_degree == 0 {
            return Err(DiskAnnError::InvalidConfig(
                "max_degree must be greater than zero".into(),
            ));
        }
        if self.search_list_size == 0 {
            return Err(DiskAnnError::InvalidConfig(
                "search_list_size must be greater than zero".into(),
            ));
        }
        if !self.alpha.is_finite() || self.alpha < 1.0 {
            return Err(DiskAnnError::InvalidConfig(
                "alpha must be finite and >= 1.0".into(),
            ));
        }
        Ok(())
    }
}

// ── DiskAnnHit ────────────────────────────────────────────────────────────────

/// A single search result returned by
/// [`DiskAnnIndex::search`](crate::disk_ann::DiskAnnIndex::search).
///
/// Hits are always ranked by descending [`score`](DiskAnnHit::score) — the most
/// similar vector appears first — regardless of which [`DiskAnnMetric`] is
/// configured. See the [`DiskAnnMetric`] documentation for the exact scoring
/// convention used by each metric.
#[derive(Debug, Clone, PartialEq)]
pub struct DiskAnnHit {
    /// String identifier of the matched entry.
    pub id: String,
    /// Similarity score; larger is better. See [`DiskAnnMetric`] for the
    /// per-metric convention.
    pub score: f32,
}

impl DiskAnnHit {
    /// Create a new hit from a string identifier and a score.
    #[must_use]
    pub fn new(id: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
        }
    }
}

// ── DiskAnnError ──────────────────────────────────────────────────────────────

/// Errors produced by the `disk_ann` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DiskAnnError {
    /// An operation requiring at least one indexed vector was attempted on an
    /// empty index (or `build` was called with an empty item list).
    #[error("index is empty")]
    EmptyIndex,
    /// A supplied vector did not match the index's dimensionality.
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension the index was built with.
        expected: usize,
        /// The dimension of the offending vector.
        got: usize,
    },
    /// A configuration field is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// The supplied query vector had zero length.
    #[error("query vector is empty")]
    EmptyQuery,
}
