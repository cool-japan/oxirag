//! Types for the `index_maintenance` module.
//!
//! Defines [`MaintenanceMetric`] (the distance/score convention shared by every
//! graph algorithm in this module), [`MaintenanceConfig`] (the `R`/`L`/`alpha`
//! knobs plus the auto-consolidation threshold), [`MaintenanceHit`] (a single
//! ranked result), [`MaintenanceStats`] (a health snapshot of the live graph),
//! [`ConsolidationReport`] (deterministic counters returned by
//! [`MaintainableIndex::consolidate`](crate::index_maintenance::MaintainableIndex::consolidate)),
//! and [`MaintenanceError`].

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── MaintenanceMetric ─────────────────────────────────────────────────────────

/// The distance/similarity function used to build, mutate and query a
/// [`MaintainableIndex`](crate::index_maintenance::MaintainableIndex).
///
/// # Distance vs. score convention
///
/// Every graph algorithm in this module (`GreedySearch`, `RobustPrune`, the
/// consolidation bridge, the connectivity repair pass) operates on a
/// **distance** where *smaller is closer*:
///
/// | Metric | Internal distance |
/// |--------|--------------------|
/// | [`Cosine`](MaintenanceMetric::Cosine) | `1.0 - cosine_similarity(a, b)`, in `[0.0, 2.0]` |
/// | [`L2`](MaintenanceMetric::L2) | Euclidean distance `\|\|a - b\|\|` |
/// | [`Dot`](MaintenanceMetric::Dot) | negated dot product `-(a · b)` |
///
/// [`MaintenanceHit::score`] uses the opposite convention — **larger is
/// better** — so hits are always ranked best-first regardless of metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MaintenanceMetric {
    /// Cosine similarity, normalising both vectors before comparison.
    ///
    /// This is the default metric.
    #[default]
    Cosine,
    /// Euclidean (L2) distance.
    L2,
    /// Raw (unnormalised) dot product — maximum-inner-product search.
    Dot,
}

/// Dot product of two vectors, truncating to the shorter of the two.
///
/// Every call site validates vector length against the index dimension first,
/// so truncation never occurs on the live paths.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm of a vector.
fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Cosine similarity clamped to `[-1.0, 1.0]`; `0.0` when either vector has a
/// numerically zero norm (avoids a division by zero).
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

impl MaintenanceMetric {
    /// The internal graph **distance** between `a` and `b`: smaller is closer.
    ///
    /// See the [`MaintenanceMetric`] documentation for the per-variant formula.
    #[must_use]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => 1.0 - cosine_similarity(a, b),
            Self::L2 => l2_distance(a, b),
            Self::Dot => -dot(a, b),
        }
    }

    /// The externally reported **score** between `a` and `b`: larger is better.
    ///
    /// See the [`MaintenanceMetric`] documentation for the per-variant formula.
    #[must_use]
    pub fn score(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => cosine_similarity(a, b),
            Self::L2 => -l2_distance(a, b),
            Self::Dot => dot(a, b),
        }
    }
}

// ── MaintenanceConfig ─────────────────────────────────────────────────────────

/// Configuration for a
/// [`MaintainableIndex`](crate::index_maintenance::MaintainableIndex).
///
/// * `max_degree` (`R`) — the hard cap on every live node's out-degree. This is
///   a **maintained invariant**, not a hint: insert, consolidation and the
///   connectivity repair pass all re-prune back down to `R`.
/// * `search_list_size` (`L`) — the `GreedySearch` beam width, used both while
///   building and while querying.
/// * `alpha` — the `RobustPrune` α-occlusion factor. Values `> 1.0` let a few
///   long-range edges survive pruning, shrinking the graph diameter.
/// * `tombstone_threshold` — when the tombstoned fraction of the slot table
///   reaches this value, [`MaintainableIndex::delete`](crate::index_maintenance::MaintainableIndex::delete)
///   auto-consolidates. `None` (the default) means consolidation is entirely manual.
/// * `repair_on_insert` — run the connectivity repair pass after every insert
///   so the "no orphaned live node" invariant holds continuously rather than
///   only at consolidation boundaries. Costs `O(N + E)` per insert; turn it off
///   for bulk loads (`build` does this internally and repairs once at the end).
///
/// # Why `max_degree >= 2`
///
/// The connectivity repair pass in
/// [`consolidate`](crate::index_maintenance::MaintainableIndex::consolidate)
/// relies on a pigeonhole argument that needs an average in-degree of at least
/// two inside a degree-saturated component (see the module documentation), so
/// `R == 1` is rejected by [`MaintenanceConfig::validate`]. An out-degree-one
/// proximity graph is a linked list and is useless for search anyway.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "index-maintenance")] {
/// use oxirag::index_maintenance::{MaintenanceConfig, MaintenanceMetric};
///
/// let cfg = MaintenanceConfig::new()
///     .with_max_degree(16)
///     .with_search_list_size(48)
///     .with_alpha(1.2)
///     .with_metric(MaintenanceMetric::L2)
///     .with_tombstone_threshold(Some(0.2));
///
/// cfg.validate().unwrap();
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceConfig {
    /// Maximum out-degree (`R`) of every node. Defaults to `32`.
    pub max_degree: usize,
    /// `GreedySearch` beam width (`L`). Defaults to `64`.
    pub search_list_size: usize,
    /// `RobustPrune` occlusion factor (`alpha`), must be `>= 1.0`. Defaults to `1.2`.
    pub alpha: f32,
    /// Distance/similarity function. Defaults to [`MaintenanceMetric::Cosine`].
    pub metric: MaintenanceMetric,
    /// Tombstoned-fraction threshold (in `(0.0, 1.0]`) at which a `delete`
    /// triggers an automatic consolidation. Defaults to `None` (manual).
    pub tombstone_threshold: Option<f32>,
    /// Run the connectivity repair pass after every insert. Defaults to `true`.
    pub repair_on_insert: bool,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,
            search_list_size: 64,
            alpha: 1.2,
            metric: MaintenanceMetric::Cosine,
            tombstone_threshold: None,
            repair_on_insert: true,
        }
    }
}

impl MaintenanceConfig {
    /// Create a configuration with default values.
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
    pub fn with_metric(mut self, metric: MaintenanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the auto-consolidation tombstone threshold.
    #[must_use]
    pub fn with_tombstone_threshold(mut self, tombstone_threshold: Option<f32>) -> Self {
        self.tombstone_threshold = tombstone_threshold;
        self
    }

    /// Enable or disable the post-insert connectivity repair pass.
    #[must_use]
    pub fn with_repair_on_insert(mut self, repair_on_insert: bool) -> Self {
        self.repair_on_insert = repair_on_insert;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`MaintenanceError::InvalidConfig`] if `max_degree < 2`, if
    /// `search_list_size` is zero, if `alpha` is not finite or `< 1.0`, or if
    /// `tombstone_threshold` is `Some(t)` with `t` outside `(0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), MaintenanceError> {
        if self.max_degree < 2 {
            return Err(MaintenanceError::InvalidConfig(
                "max_degree must be at least 2 (see MaintenanceConfig docs)".into(),
            ));
        }
        if self.search_list_size == 0 {
            return Err(MaintenanceError::InvalidConfig(
                "search_list_size must be greater than zero".into(),
            ));
        }
        if !self.alpha.is_finite() || self.alpha < 1.0 {
            return Err(MaintenanceError::InvalidConfig(
                "alpha must be finite and >= 1.0".into(),
            ));
        }
        if let Some(threshold) = self.tombstone_threshold
            && (!threshold.is_finite() || threshold <= 0.0 || threshold > 1.0)
        {
            return Err(MaintenanceError::InvalidConfig(
                "tombstone_threshold must lie in (0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }
}

// ── MaintenanceHit ────────────────────────────────────────────────────────────

/// A single search result from
/// [`MaintainableIndex::search`](crate::index_maintenance::MaintainableIndex::search).
///
/// Hits are ranked by descending [`score`](MaintenanceHit::score) and never
/// include tombstoned entries — a tombstoned node is traversed *through* during
/// search but is never *reported*.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceHit {
    /// External identifier of the matched entry.
    pub id: String,
    /// Similarity score; larger is better. See [`MaintenanceMetric`].
    pub score: f32,
}

impl MaintenanceHit {
    /// Create a hit from an external identifier and a score.
    #[must_use]
    pub fn new(id: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
        }
    }
}

// ── MaintenanceStats ──────────────────────────────────────────────────────────

/// A health snapshot of the graph, produced by
/// [`MaintainableIndex::stats`](crate::index_maintenance::MaintainableIndex::stats).
///
/// [`orphan_count`](MaintenanceStats::orphan_count) is computed under exactly
/// the traversal rules `search` uses: a BFS from the entry point that walks
/// *through* tombstoned nodes. A live node that BFS cannot reach can never be
/// returned by any query, so `orphan_count > 0` is the direct, measurable
/// symptom of the connectivity damage that a naive hard delete inflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceStats {
    /// Number of live (non-tombstoned) nodes.
    pub live_nodes: usize,
    /// Number of tombstoned nodes still occupying a slot.
    pub tombstoned_nodes: usize,
    /// Total slots in the node table (`live_nodes + tombstoned_nodes`).
    pub total_slots: usize,
    /// Total directed out-edges over all slots.
    pub edge_count: usize,
    /// Mean out-degree over **live** nodes, scaled by 1000 and rounded, so the
    /// snapshot stays `Eq`/`Hash`-friendly. Use
    /// [`mean_out_degree`](MaintenanceStats::mean_out_degree) for the `f32`.
    pub mean_out_degree_milli: u64,
    /// Largest out-degree over all slots. Must never exceed `R`.
    pub max_out_degree: usize,
    /// Live nodes unreachable from the entry point (see the type-level docs).
    pub orphan_count: usize,
    /// Whether the current entry point is itself tombstoned.
    pub entry_point_tombstoned: bool,
}

impl MaintenanceStats {
    /// Mean out-degree over live nodes as an `f32`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_out_degree(&self) -> f32 {
        self.mean_out_degree_milli as f32 / 1000.0
    }
}

// ── ConsolidationReport ───────────────────────────────────────────────────────

/// Deterministic counters returned by
/// [`MaintainableIndex::consolidate`](crate::index_maintenance::MaintainableIndex::consolidate).
///
/// Every field is a count, never a timing: the same index consolidated twice
/// from the same state produces byte-identical reports, which makes the module
/// testable without wall-clock flakiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Tombstoned nodes physically removed from the slot table.
    pub nodes_removed: usize,
    /// Edges `p -> t` (live `p`, tombstoned `t`) that were bridged over.
    pub edges_rewired: usize,
    /// `RobustPrune` invocations performed during the bridge and repair passes.
    pub prunes_performed: usize,
    /// Live nodes that the repair pass had to re-attach to the reachable set.
    pub orphans_repaired: usize,
    /// Whether the entry point had to be re-selected (its node was tombstoned).
    pub entry_point_reselected: bool,
    /// Live nodes remaining after consolidation.
    pub live_nodes_after: usize,
}

impl ConsolidationReport {
    /// `true` if the consolidation was a no-op (no tombstones to collect).
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.nodes_removed == 0 && self.edges_rewired == 0 && self.prunes_performed == 0
    }
}

// ── MaintenanceError ──────────────────────────────────────────────────────────

/// Errors produced by the `index_maintenance` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MaintenanceError {
    /// A configuration field is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// `build` was called with an empty item list, so the dimension cannot be
    /// inferred. Use [`MaintainableIndex::new`](crate::index_maintenance::MaintainableIndex::new)
    /// to start from an explicitly-dimensioned empty index.
    #[error("cannot build an index from an empty item list")]
    EmptyIndex,
    /// A supplied vector did not match the index dimension.
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension the index was created with.
        expected: usize,
        /// The dimension of the offending vector.
        got: usize,
    },
    /// The index dimension must be greater than zero.
    #[error("index dimension must be greater than zero")]
    ZeroDimension,
    /// An insert targeted an external id that is already **live**.
    #[error("duplicate id: {0} is already live in the index")]
    DuplicateId(String),
    /// A delete targeted an external id that is not live in the index.
    #[error("unknown id: {0}")]
    UnknownId(String),
}
