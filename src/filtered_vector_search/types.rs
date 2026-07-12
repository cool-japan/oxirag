//! Core data types for predicate-constrained approximate nearest-neighbor
//! search: attribute values, metadata records, distance metrics, strategy
//! selection, search configuration, results, statistics, and errors.
//!
//! Nothing in this file performs search. It defines the vocabulary the rest
//! of the module speaks:
//!
//! - [`AttrValue`] / [`FilteredMetadata`] — the attribute model a predicate
//!   is evaluated against.
//! - [`FilterBound`] — one end of a numeric range constraint.
//! - [`FilterStrategy`] — which of the three filtering execution plans to run.
//! - [`FilteredDistanceMetric`] — how vector proximity is measured.
//! - [`FilteredSearchConfig`] — graph-build, traversal, and strategy-selection
//!   knobs.
//! - [`FilteredHit`] / [`FilteredSearchStats`] / [`SelectivityEstimate`] — what
//!   a search returns.
//! - [`FilteredSearchError`] — the module's error enumeration.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── AttrValue ────────────────────────────────────────────────────────────────

/// The canonical bit pattern used for every NaN, so that NaN keys are
/// self-consistent under [`Eq`] and [`Hash`].
const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;

/// A single scalar attribute value attached to an indexed vector.
///
/// # Equality and hashing semantics
///
/// [`AttrValue`] is used as a hash-map key by the module's inverted attribute
/// index, so its [`PartialEq`], [`Eq`] and [`Hash`] impls must agree. They are
/// **strictly type-tagged**:
///
/// - `Int(3) != Float(3.0)`. Equality is *identity*, not numeric coercion. A
///   predicate `Eq` therefore never matches across variants.
/// - Float values are compared and hashed via a *canonicalized* bit pattern:
///   every NaN collapses to one canonical NaN (so `NaN == NaN` here, unlike
///   raw IEEE-754, which would break the reflexivity that [`Eq`] requires),
///   and `-0.0` is folded to `+0.0` (so `-0.0 == 0.0`, as arithmetic
///   intuition demands). Every other float compares by its exact bits.
///
/// Ordering, by contrast, **does** coerce numerically: a
/// [`FilterPredicate::Range`](crate::filtered_vector_search::FilterPredicate::Range)
/// with `f64` bounds matches `Int` values as well as `Float` values, because a
/// range constraint is a statement about the number line rather than about
/// type identity. See [`AttrValue::as_f64`].
///
/// This asymmetry (equality is exact, ordering is numeric) is deliberate and
/// mirrors how relational databases treat typed columns versus range scans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttrValue {
    /// A UTF-8 string, e.g. a category, tenant id, or language tag.
    Str(String),
    /// A signed 64-bit integer, e.g. a year, a version, or a count.
    Int(i64),
    /// A double-precision float, e.g. a price or a quality score.
    Float(f64),
    /// A boolean flag, e.g. `public` or `deprecated`.
    Bool(bool),
}

impl AttrValue {
    /// Canonicalize a float to a bit pattern usable as a hash key.
    ///
    /// All NaNs collapse to one representative and `-0.0` folds to `+0.0`, so
    /// that the resulting key is consistent with the [`PartialEq`] impl.
    fn float_key(value: f64) -> u64 {
        if value.is_nan() {
            CANONICAL_NAN_BITS
        } else if value == 0.0 {
            // Covers both `+0.0` and `-0.0`.
            0.0_f64.to_bits()
        } else {
            value.to_bits()
        }
    }

    /// The numeric view of this value, used for range (ordering) comparisons.
    ///
    /// `Int` and `Float` both project onto the real line; `Str` and `Bool` do
    /// not participate in range constraints and yield `None`.
    ///
    /// # Precision
    ///
    /// An `Int` whose magnitude exceeds `2^53` cannot be represented exactly
    /// as an `f64`, so its projection is rounded to the nearest representable
    /// double. Range comparisons against such values are therefore accurate to
    /// within one unit in the last place of the `f64` grid, not exact. This is
    /// the same trade-off SQL engines make when comparing `BIGINT` against
    /// `DOUBLE PRECISION`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            #[allow(clippy::cast_precision_loss)]
            Self::Int(value) => Some(*value as f64),
            Self::Float(value) => Some(*value),
            Self::Str(_) | Self::Bool(_) => None,
        }
    }

    /// The string view of this value, or `None` for non-string variants.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// The boolean view of this value, or `None` for non-boolean variants.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Whether this value participates in numeric range constraints.
    #[must_use]
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Int(_) | Self::Float(_))
    }

    /// A short, stable name for this value's variant, used in error messages.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Str(_) => "str",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Bool(_) => "bool",
        }
    }
}

impl PartialEq for AttrValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Str(left), Self::Str(right)) => left == right,
            (Self::Int(left), Self::Int(right)) => left == right,
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Float(left), Self::Float(right)) => {
                Self::float_key(*left) == Self::float_key(*right)
            }
            _ => false,
        }
    }
}

// Reflexive by construction: the `Float` arm compares canonicalized bit
// patterns, so `NaN == NaN` holds here even though it does not in IEEE-754.
impl Eq for AttrValue {}

impl std::hash::Hash for AttrValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Discriminant first, so that `Int(3)` and `Float(3.0)` land in
        // different hash buckets, matching the type-tagged `PartialEq`.
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Str(value) => value.hash(state),
            Self::Int(value) => value.hash(state),
            Self::Bool(value) => value.hash(state),
            Self::Float(value) => Self::float_key(*value).hash(state),
        }
    }
}

impl From<&str> for AttrValue {
    fn from(value: &str) -> Self {
        Self::Str(value.to_string())
    }
}

impl From<String> for AttrValue {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<i64> for AttrValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for AttrValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for AttrValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

// ── FilteredMetadata ─────────────────────────────────────────────────────────

/// The attribute map carried by every vector in a
/// [`FilteredVectorIndex`](crate::filtered_vector_search::FilteredVectorIndex).
///
/// A missing key is semantically distinct from a present key holding a
/// "falsy" value: predicates such as
/// [`Exists`](crate::filtered_vector_search::FilterPredicate::Exists) and
/// [`Ne`](crate::filtered_vector_search::FilterPredicate::Ne) test for
/// presence, and every leaf predicate evaluates to `false` on an absent
/// attribute.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilteredMetadata {
    attributes: HashMap<String, AttrValue>,
}

impl FilteredMetadata {
    /// Create an empty metadata record.
    #[must_use]
    pub fn new() -> Self {
        Self {
            attributes: HashMap::new(),
        }
    }

    /// Insert (or overwrite) `attr` and return `self`, for chained building.
    #[must_use]
    pub fn with(mut self, attr: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        self.attributes.insert(attr.into(), value.into());
        self
    }

    /// Insert (or overwrite) `attr`, returning the previous value if any.
    pub fn insert(
        &mut self,
        attr: impl Into<String>,
        value: impl Into<AttrValue>,
    ) -> Option<AttrValue> {
        self.attributes.insert(attr.into(), value.into())
    }

    /// Look up an attribute by name.
    #[must_use]
    pub fn get(&self, attr: &str) -> Option<&AttrValue> {
        self.attributes.get(attr)
    }

    /// Whether `attr` is present (regardless of its value).
    #[must_use]
    pub fn contains(&self, attr: &str) -> bool {
        self.attributes.contains_key(attr)
    }

    /// The number of attributes present.
    #[must_use]
    pub fn len(&self) -> usize {
        self.attributes.len()
    }

    /// Whether this record carries no attributes at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    /// Iterate over `(name, value)` pairs in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &AttrValue)> {
        self.attributes.iter()
    }
}

impl<'a> IntoIterator for &'a FilteredMetadata {
    type Item = (&'a String, &'a AttrValue);
    type IntoIter = std::collections::hash_map::Iter<'a, String, AttrValue>;

    fn into_iter(self) -> Self::IntoIter {
        self.attributes.iter()
    }
}

impl FromIterator<(String, AttrValue)> for FilteredMetadata {
    fn from_iter<I: IntoIterator<Item = (String, AttrValue)>>(iter: I) -> Self {
        Self {
            attributes: iter.into_iter().collect(),
        }
    }
}

// ── FilterBound ──────────────────────────────────────────────────────────────

/// One end of a numeric range constraint.
///
/// Bounds are always expressed as `f64` and are compared against
/// [`AttrValue::as_f64`], so a single range predicate constrains `Int` and
/// `Float` attributes uniformly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FilterBound {
    /// The bound is part of the accepted interval (`>=` for a lower bound,
    /// `<=` for an upper bound).
    Inclusive(f64),
    /// The bound is excluded from the accepted interval (`>` for a lower
    /// bound, `<` for an upper bound).
    Exclusive(f64),
    /// No constraint on this end (`-inf` for a lower bound, `+inf` for an
    /// upper bound).
    Unbounded,
}

impl FilterBound {
    /// Whether `value` satisfies this bound when it is used as a *lower* bound.
    #[must_use]
    pub fn accepts_as_lower(&self, value: f64) -> bool {
        match self {
            Self::Inclusive(bound) => value >= *bound,
            Self::Exclusive(bound) => value > *bound,
            Self::Unbounded => true,
        }
    }

    /// Whether `value` satisfies this bound when it is used as an *upper* bound.
    #[must_use]
    pub fn accepts_as_upper(&self, value: f64) -> bool {
        match self {
            Self::Inclusive(bound) => value <= *bound,
            Self::Exclusive(bound) => value < *bound,
            Self::Unbounded => true,
        }
    }

    /// The numeric position of this bound, or `None` when unbounded.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        match self {
            Self::Inclusive(bound) | Self::Exclusive(bound) => Some(*bound),
            Self::Unbounded => None,
        }
    }

    /// The half-line this bound cuts out, projected onto the real line, for
    /// use by the selectivity estimator: `(-inf, x]` style bounds become a
    /// concrete `f64`, unbounded ends become `±inf`.
    #[must_use]
    pub fn as_lower_f64(&self) -> f64 {
        self.value().unwrap_or(f64::NEG_INFINITY)
    }

    /// Counterpart of [`FilterBound::as_lower_f64`] for the upper end.
    #[must_use]
    pub fn as_upper_f64(&self) -> f64 {
        self.value().unwrap_or(f64::INFINITY)
    }
}

// ── FilterStrategy ───────────────────────────────────────────────────────────

/// The execution plan used to combine vector proximity with a metadata
/// predicate.
///
/// The three concrete plans have sharply different cost/recall profiles; see
/// the module documentation for the full cost model and
/// [`StrategySelector`](crate::filtered_vector_search::StrategySelector) for
/// the rule that picks between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FilterStrategy {
    /// Estimate the predicate's selectivity, then delegate to whichever of the
    /// three concrete strategies the cost model prefers. This is the default.
    #[default]
    Auto,
    /// Run an *unconstrained* graph search for `top_k * post_filter_multiplier`
    /// candidates, then discard the ones that fail the predicate and truncate
    /// to `top_k`.
    ///
    /// Cheap, trivially implemented — and **broken below a selectivity of
    /// `1 / post_filter_multiplier`**, because the over-fetch simply does not
    /// reach far enough down the unconstrained ranking to contain `top_k`
    /// matching vectors. This is the naive baseline the rest of the module
    /// exists to beat.
    PostFilter,
    /// Materialize the matching subset from the inverted attribute index, then
    /// scan it exhaustively.
    ///
    /// **Exact by construction** — it always returns the true top-`k` of the
    /// matching set — but its cost is linear in the size of that set, so it is
    /// only attractive when the predicate is highly selective.
    PreFilter,
    /// Predicate-aware traversal of the proximity graph (ACORN): the result
    /// heap only ever admits vectors that satisfy the predicate, but the
    /// traversal is allowed to *route through* vectors that do not, by
    /// expanding their neighbor lists (a two-hop step).
    ///
    /// This keeps the predicate-induced subgraph navigable even when the
    /// predicate is selective enough that a naive filtered traversal would
    /// fragment into disconnected islands and dead-end immediately.
    InFilter,
}

impl FilterStrategy {
    /// A short, stable name for this strategy, for logs and stats.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::PostFilter => "post_filter",
            Self::PreFilter => "pre_filter",
            Self::InFilter => "in_filter",
        }
    }

    /// Whether this strategy returns the exact top-`k` of the matching set
    /// (as opposed to an approximation of it).
    #[must_use]
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::PreFilter)
    }
}

// ── FilteredDistanceMetric ───────────────────────────────────────────────────

/// The proximity function used to build the graph and rank results.
///
/// Both the graph algorithms and the result heaps rank on a **distance**
/// (smaller is closer). [`FilteredHit::score`] flips the convention so that
/// larger is always better, whichever metric is in play.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FilteredDistanceMetric {
    /// Cosine distance, `1 - cosine_similarity(a, b)`, in `[0, 2]`.
    ///
    /// A zero-norm vector is defined to be at distance `1.0` (i.e. orthogonal)
    /// from everything, including itself, since its direction is undefined.
    /// This is the default.
    #[default]
    Cosine,
    /// Euclidean (L2) distance, `||a - b||`.
    ///
    /// Ranking is performed on the *squared* distance (which is monotone in
    /// the true distance, so the ordering is identical) and the square root is
    /// taken only when a hit is materialized.
    Euclidean,
}

impl FilteredDistanceMetric {
    /// A short, stable name for this metric.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Euclidean => "euclidean",
        }
    }
}

// ── FilteredHit ──────────────────────────────────────────────────────────────

/// One ranked result of a filtered search.
///
/// Every hit is guaranteed to satisfy the predicate that produced it — that
/// invariant holds for all three strategies, including `PostFilter` (which
/// simply returns fewer hits than requested when its over-fetch falls short).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilteredHit {
    /// The caller-supplied identifier of the matching vector.
    pub id: String,
    /// Distance from the query under the index's metric; smaller is closer.
    pub distance: f32,
    /// Monotone-decreasing transform of [`FilteredHit::distance`]: larger is
    /// better, for callers that expect a similarity-style score.
    ///
    /// For [`FilteredDistanceMetric::Cosine`] this is the raw cosine
    /// similarity in `[-1, 1]`; for [`FilteredDistanceMetric::Euclidean`] it is
    /// the negated Euclidean distance.
    pub score: f32,
    /// Zero-based position of this hit in the returned ranking.
    pub rank: usize,
}

impl FilteredHit {
    /// Construct a hit.
    #[must_use]
    pub fn new(id: impl Into<String>, distance: f32, score: f32, rank: usize) -> Self {
        Self {
            id: id.into(),
            distance,
            score,
            rank,
        }
    }
}

// ── SelectivityEstimate ──────────────────────────────────────────────────────

/// The estimator's verdict on how much of the corpus a predicate admits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectivityEstimate {
    /// Estimated fraction of the corpus satisfying the predicate, clamped to
    /// `[0, 1]`.
    pub selectivity: f64,
    /// `selectivity * total_vectors`, i.e. the estimated number of matches.
    pub estimated_matches: f64,
    /// The corpus size the estimate was computed against.
    pub total_vectors: usize,
    /// Whether every leaf of the predicate could be estimated from exact
    /// statistics.
    ///
    /// This is `false` when any leaf had to fall back to a heuristic — an
    /// attribute whose distinct-value counts overflowed the tracking cap, an
    /// attribute the estimator has never seen, or a numeric range whose
    /// endpoints landed inside a histogram bucket (forcing the
    /// within-bucket-uniformity assumption). It does **not** account for the
    /// independence assumption used by `And`/`Or`, which is always in play;
    /// see [`SelectivityEstimate::independence_assumed`].
    pub exact_leaves: bool,
    /// Whether the estimate combined two or more sub-predicates under the
    /// independence assumption (which `And` and `Or` always make).
    ///
    /// When this is `true` the estimate can be arbitrarily wrong for
    /// correlated attributes — the classic failure mode of every textbook
    /// cardinality estimator — and the number should be read as a planning
    /// hint, never as a count.
    pub independence_assumed: bool,
}

impl SelectivityEstimate {
    /// Construct an estimate, clamping `selectivity` into `[0, 1]`.
    #[must_use]
    pub fn new(
        selectivity: f64,
        total_vectors: usize,
        exact_leaves: bool,
        independence_assumed: bool,
    ) -> Self {
        let selectivity = if selectivity.is_finite() {
            selectivity.clamp(0.0, 1.0)
        } else {
            // A non-finite intermediate can only arise from a corrupted
            // statistic; degrade to "assume everything matches", which is the
            // safe direction (it never causes a strategy that loses recall).
            1.0
        };
        #[allow(clippy::cast_precision_loss)]
        let estimated_matches = selectivity * total_vectors as f64;
        Self {
            selectivity,
            estimated_matches,
            total_vectors,
            exact_leaves,
            independence_assumed,
        }
    }

    /// Whether this estimate rests on no approximation at all: exact leaves
    /// and no independence assumption.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.exact_leaves && !self.independence_assumed
    }
}

// ── FilteredSearchStats ──────────────────────────────────────────────────────

/// Instrumentation from a single filtered search, returned alongside the hits
/// by
/// [`FilteredVectorIndex::search_with_stats`](crate::filtered_vector_search::FilteredVectorIndex::search_with_stats).
///
/// These counters are what make the module's central claim falsifiable: they
/// let a caller (and the test suite) see *why* one strategy beats another,
/// rather than taking it on faith.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FilteredSearchStats {
    /// The strategy actually executed. When the caller asked for
    /// [`FilterStrategy::Auto`] this is the concrete plan the selector chose,
    /// never `Auto` itself.
    pub strategy: FilterStrategy,
    /// Graph nodes whose neighbor lists were expanded (`PreFilter` reports
    /// `0`, since it does not touch the graph).
    pub nodes_visited: usize,
    /// Vector-to-vector distance computations performed.
    pub distance_computations: usize,
    /// Full-predicate evaluations against a node's metadata.
    ///
    /// This counts *nodes tested*, not AST nodes walked, and it excludes hits
    /// on the per-search memoization cache — so it measures real predicate
    /// work, which is exactly the quantity the two-hop expansion trades
    /// against graph hops.
    pub predicate_evaluations: usize,
    /// Distinct nodes that entered the candidate pool at any point.
    pub candidates_considered: usize,
    /// Nodes discovered by the ACORN two-hop step, i.e. reached by routing
    /// *through* a neighbor that failed the predicate. Non-zero only for
    /// [`FilterStrategy::InFilter`].
    pub two_hop_expansions: usize,
    /// Candidates that satisfied the predicate and were therefore eligible for
    /// the result heap.
    pub matched_candidates: usize,
    /// The selectivity estimate that drove strategy selection, when the caller
    /// asked for [`FilterStrategy::Auto`].
    pub selectivity: Option<SelectivityEstimate>,
}

// ── FilteredSearchConfig ─────────────────────────────────────────────────────

/// Build-time, search-time, and strategy-selection parameters.
///
/// Every default is documented with the reasoning behind it; see the module
/// documentation for how `post_filter_multiplier`, `prefilter_threshold` and
/// `postfilter_threshold` interlock to keep each strategy out of the regime
/// where it fails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilteredSearchConfig {
    /// Proximity function used throughout.
    pub metric: FilteredDistanceMetric,

    // ── Graph construction ───────────────────────────────────────────────
    /// Maximum out-degree `R` of the proximity graph.
    ///
    /// The two-hop expansion reaches up to `R^2` nodes, so `R` directly
    /// controls how selective a predicate the graph can stay navigable under.
    pub graph_degree: usize,
    /// Beam width `L_build` used by the greedy search that seeds each node's
    /// candidate neighbor set during construction. Must be `>= graph_degree`
    /// for the pruner to have anything to prune.
    pub build_beam_width: usize,
    /// The `alpha` of the Vamana robust-prune occlusion rule.
    ///
    /// `alpha == 1.0` keeps only the relative-neighborhood edges; `alpha > 1.0`
    /// deliberately retains some "long-range" edges that a strict relative
    /// neighborhood graph would occlude, which is what gives the graph its
    /// small-world diameter. Must be `>= 1.0`.
    pub prune_alpha: f64,

    // ── Search ───────────────────────────────────────────────────────────
    /// Beam width `L` of the search-time candidate pool. Larger means more
    /// nodes expanded, higher recall, more work.
    pub search_beam_width: usize,
    /// Number of graph entry points a search starts from.
    ///
    /// The first is the medoid; the rest are deterministically dispersed over
    /// the id space. Multiple seeds make a filtered traversal materially more
    /// robust, because a single seed can sit in a region of the graph from
    /// which the matching set happens to be hard to reach.
    pub seed_count: usize,

    // ── ACORN (InFilter) ─────────────────────────────────────────────────
    /// The ACORN neighbor-expansion factor `gamma`: how many predicate-*failing*
    /// neighbors a single expansion may route **through**, harvesting the
    /// predicate-satisfying members of each one's neighbor list.
    ///
    /// This is the knob that decides how selective a predicate the traversal can
    /// survive. The ACORN paper's guidance is `gamma ≈ 1 / s_min`, the inverse
    /// of the lowest selectivity you intend to support (the paper spends it on a
    /// `graph_degree * gamma`-degree index; this module spends it at search
    /// time instead, on a fixed-degree graph, which costs nothing to build and
    /// lets a single index serve predicates of any selectivity). It is naturally
    /// capped by `graph_degree`, since a node cannot have more failing neighbors
    /// than it has neighbors.
    ///
    /// **Raising it is far cheaper than it looks.** The two-hop step computes a
    /// distance only for the second-hop nodes that actually *match*, and its
    /// predicate evaluations are memoized per search, so most of the extra work
    /// is walking neighbor lists that are already in cache. Measured on the
    /// module's own 2,000-vector fixture at 1% selectivity, going from
    /// `gamma = 3` to `gamma = 16` moved recall@10 from **0.77 to 0.96** while
    /// distance computations rose only from 630 to 696 — a 25-point recall gain
    /// for a 10% cost increase. Hence the default of 16 rather than something
    /// timid.
    ///
    /// `gamma == 1` still performs the two-hop step — routing through failing
    /// neighbors is what *defines* ACORN — but through only the single closest
    /// failing neighbor.
    pub neighbor_expansion_gamma: usize,

    /// The two-hop gate: the number of predicate-*matching* neighbors below
    /// which a node's expansion triggers the two-hop step.
    ///
    /// This is deliberately **not** derived from `neighbor_expansion_gamma`,
    /// though the temptation is strong (and the first draft of this module
    /// succumbed to it, using `graph_degree / gamma`). The two are answering
    /// different questions — *"is the predicate subgraph around this node too
    /// sparse to navigate?"* versus *"how hard should I work when it is?"* — and
    /// tying them together means that turning the expansion up
    /// (`gamma -> graph_degree`) silently turns the gate *off*
    /// (`graph_degree / gamma -> 1`), so the two-hop step stops firing exactly
    /// as you ask for more of it. That misconfiguration is directly measurable:
    /// it cost 2 points of recall at 10% selectivity on the module's fixture
    /// before the knobs were separated.
    ///
    /// The default of 8 (a quarter of the default `graph_degree` of 32) fires
    /// the two-hop step below roughly 25% selectivity — which is about where a
    /// degree-32 graph's predicate subgraph starts to thin out — and skips it
    /// above, keeping `InFilter` at roughly unfiltered cost for permissive
    /// predicates.
    pub min_predicate_neighbors: usize,
    /// Fraction of the candidate pool reserved for predicate-*matching* nodes.
    ///
    /// Without a reserve, a selective predicate lets non-matching routing
    /// nodes — which cluster tightly around the query — monopolize every pool
    /// slot, so matching nodes are discovered but never *expanded*, and the
    /// traversal never explores the matching region. Reserving a slice of the
    /// pool for matching nodes guarantees the traversal follows the matches it
    /// finds. Clamped to `[0, 1)`.
    pub matching_reserve_ratio: f64,
    /// Hard ceiling on node expansions per search, as a multiple of
    /// `search_beam_width`.
    ///
    /// A selective predicate can force the traversal to expand far more nodes
    /// than an unfiltered one (it cannot terminate on a full result heap
    /// because the heap may never fill), so this bounds the tail. Reaching the
    /// ceiling is a signal that `PreFilter` would have been the better plan.
    pub max_visit_multiplier: usize,

    // ── PostFilter ───────────────────────────────────────────────────────
    /// Over-fetch multiplier `m` for [`FilterStrategy::PostFilter`]: the
    /// unconstrained search retrieves `top_k * m` candidates before filtering.
    ///
    /// Defaults to `10`, matching the legacy fixed multiplier this module
    /// supersedes. See the module docs for why a *fixed* `m` cannot work below
    /// a selectivity of `1 / m`.
    pub post_filter_multiplier: usize,

    // ── Strategy selection ───────────────────────────────────────────────
    /// Estimated selectivity at or below which [`FilterStrategy::PreFilter`] is
    /// chosen.
    pub prefilter_threshold: f64,
    /// Estimated selectivity at or above which [`FilterStrategy::PostFilter`] is
    /// chosen.
    ///
    /// Must satisfy `postfilter_threshold >= 1 / post_filter_multiplier` for
    /// the selector to be sound; [`FilteredSearchConfig::validate`] enforces
    /// this.
    pub postfilter_threshold: f64,
    /// Absolute ceiling on the estimated match count below which `PreFilter` is
    /// chosen regardless of the selectivity ratio.
    ///
    /// On a small corpus a "1%" predicate matches a handful of vectors and an
    /// exact scan is free; on a huge corpus 1% may still be millions of
    /// vectors. The selector therefore gates on *both* the ratio and the
    /// absolute count.
    pub prefilter_max_matches: usize,

    // ── Statistics ───────────────────────────────────────────────────────
    /// Number of equi-width buckets per numeric attribute histogram.
    pub histogram_buckets: usize,
    /// Maximum distinct values tracked exactly per attribute before the
    /// estimator falls back to a `1 / distinct` heuristic (and reports
    /// [`SelectivityEstimate::exact_leaves`] as `false`).
    pub max_tracked_values: usize,
    /// Maximum distinct values for which the inverted index keeps posting
    /// lists. Past this, equality lookups on that attribute degrade to a full
    /// scan — which is still *exact*, merely slower.
    pub max_postings_per_attr: usize,

    /// Seed for the module's internal deterministic PRNG (used only to
    /// disperse graph entry seeds and to order construction). Two indexes
    /// built from the same data with the same seed are byte-for-byte
    /// identical.
    pub seed: u64,
}

impl Default for FilteredSearchConfig {
    fn default() -> Self {
        Self {
            metric: FilteredDistanceMetric::default(),
            graph_degree: 32,
            build_beam_width: 64,
            prune_alpha: 1.2,
            search_beam_width: 64,
            seed_count: 4,
            neighbor_expansion_gamma: 16,
            min_predicate_neighbors: 8,
            matching_reserve_ratio: 0.25,
            max_visit_multiplier: 16,
            post_filter_multiplier: 10,
            prefilter_threshold: 0.01,
            postfilter_threshold: 0.80,
            prefilter_max_matches: 4_096,
            histogram_buckets: 32,
            max_tracked_values: 4_096,
            max_postings_per_attr: 65_536,
            seed: 0x5eed_f11e_5ea1_c400,
        }
    }
}

impl FilteredSearchConfig {
    /// Validate every invariant the algorithms rely on.
    ///
    /// # Errors
    ///
    /// Returns [`FilteredSearchError::InvalidConfig`] naming the offending
    /// field when a parameter is out of range or when two parameters are
    /// mutually inconsistent (notably `postfilter_threshold` being low enough
    /// that the selector could route a query into `PostFilter`'s failure
    /// regime).
    pub fn validate(&self) -> Result<(), FilteredSearchError> {
        self.validate_traversal()?;
        self.validate_planning()
    }

    /// Validate the graph-construction and traversal parameters.
    fn validate_traversal(&self) -> Result<(), FilteredSearchError> {
        let invalid = |field: &str, reason: String| FilteredSearchError::InvalidConfig {
            field: field.to_string(),
            reason,
        };

        if self.graph_degree == 0 {
            return Err(invalid("graph_degree", "must be at least 1".to_string()));
        }
        if self.build_beam_width < self.graph_degree {
            return Err(invalid(
                "build_beam_width",
                format!(
                    "must be >= graph_degree ({}), got {}",
                    self.graph_degree, self.build_beam_width
                ),
            ));
        }
        if !(self.prune_alpha.is_finite() && self.prune_alpha >= 1.0) {
            return Err(invalid(
                "prune_alpha",
                format!("must be finite and >= 1.0, got {}", self.prune_alpha),
            ));
        }
        if self.search_beam_width == 0 {
            return Err(invalid(
                "search_beam_width",
                "must be at least 1".to_string(),
            ));
        }
        if self.seed_count == 0 {
            return Err(invalid("seed_count", "must be at least 1".to_string()));
        }
        if self.neighbor_expansion_gamma == 0 {
            return Err(invalid(
                "neighbor_expansion_gamma",
                "must be at least 1; gamma = 1 still performs the two-hop step".to_string(),
            ));
        }
        if self.min_predicate_neighbors == 0 {
            return Err(invalid(
                "min_predicate_neighbors",
                "must be at least 1; a gate of 0 would disable the two-hop step entirely, \
                 which is the naive filtered traversal this module exists to replace"
                    .to_string(),
            ));
        }
        if !(self.matching_reserve_ratio.is_finite()
            && (0.0..1.0).contains(&self.matching_reserve_ratio))
        {
            return Err(invalid(
                "matching_reserve_ratio",
                format!(
                    "must lie in [0.0, 1.0), got {}",
                    self.matching_reserve_ratio
                ),
            ));
        }
        if self.max_visit_multiplier == 0 {
            return Err(invalid(
                "max_visit_multiplier",
                "must be at least 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate the strategy-selection and statistics parameters — including the
    /// cross-field invariant that keeps the planner out of `PostFilter`'s
    /// failure regime.
    fn validate_planning(&self) -> Result<(), FilteredSearchError> {
        let invalid = |field: &str, reason: String| FilteredSearchError::InvalidConfig {
            field: field.to_string(),
            reason,
        };

        if self.post_filter_multiplier == 0 {
            return Err(invalid(
                "post_filter_multiplier",
                "must be at least 1".to_string(),
            ));
        }
        if !(self.prefilter_threshold.is_finite()
            && (0.0..=1.0).contains(&self.prefilter_threshold))
        {
            return Err(invalid(
                "prefilter_threshold",
                format!("must lie in [0.0, 1.0], got {}", self.prefilter_threshold),
            ));
        }
        if !(self.postfilter_threshold.is_finite()
            && (0.0..=1.0).contains(&self.postfilter_threshold))
        {
            return Err(invalid(
                "postfilter_threshold",
                format!("must lie in [0.0, 1.0], got {}", self.postfilter_threshold),
            ));
        }
        if self.prefilter_threshold > self.postfilter_threshold {
            return Err(invalid(
                "prefilter_threshold",
                format!(
                    "must be <= postfilter_threshold ({}), got {}",
                    self.postfilter_threshold, self.prefilter_threshold
                ),
            ));
        }

        // Soundness of the selector: `PostFilter` retrieves `k * m` candidates
        // and expects at least `k` of them to match. Under the (standard)
        // assumption that the predicate is independent of the query's
        // proximity ranking, the expected number of matches among them is
        // `s * k * m`, so `s >= 1 / m` is the bare minimum for the over-fetch
        // to be able to fill the result set at all. Routing a query with
        // `s < 1 / m` into `PostFilter` guarantees truncated results, so the
        // selector must never be configured to do it.
        #[allow(clippy::cast_precision_loss)]
        let post_filter_break_even = 1.0 / self.post_filter_multiplier as f64;
        if self.postfilter_threshold < post_filter_break_even {
            return Err(invalid(
                "postfilter_threshold",
                format!(
                    "must be >= 1 / post_filter_multiplier ({post_filter_break_even}) or the \
                     selector would route queries into PostFilter's failure regime; got {}",
                    self.postfilter_threshold
                ),
            ));
        }

        if self.histogram_buckets == 0 {
            return Err(invalid(
                "histogram_buckets",
                "must be at least 1".to_string(),
            ));
        }
        if self.max_tracked_values == 0 {
            return Err(invalid(
                "max_tracked_values",
                "must be at least 1".to_string(),
            ));
        }
        Ok(())
    }

    /// The absolute cap on node expansions implied by
    /// `search_beam_width * max_visit_multiplier`.
    #[must_use]
    pub fn max_visits(&self) -> usize {
        self.search_beam_width
            .saturating_mul(self.max_visit_multiplier)
    }

    /// The number of candidate-pool slots reserved for predicate-matching
    /// nodes, derived from [`FilteredSearchConfig::matching_reserve_ratio`].
    #[must_use]
    pub fn matching_reserve(&self) -> usize {
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let reserve = (self.search_beam_width as f64 * self.matching_reserve_ratio) as usize;
        reserve.min(self.search_beam_width)
    }
}

// ── FilteredSearchError ──────────────────────────────────────────────────────

/// Everything that can go wrong while building or querying a
/// [`FilteredVectorIndex`](crate::filtered_vector_search::FilteredVectorIndex).
///
/// Note what is *not* an error: an empty index, a predicate that matches
/// nothing, and a `k` larger than the number of matching vectors all return an
/// `Ok` result (empty or short, respectively). Those are legitimate query
/// outcomes, not failures.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FilteredSearchError {
    /// A vector's length disagrees with the index's dimension.
    #[error("dimension mismatch: index holds {expected}-dimensional vectors, got {actual}")]
    DimensionMismatch {
        /// The index's dimension.
        expected: usize,
        /// The dimension of the offending vector.
        actual: usize,
    },

    /// A zero-dimensional vector was supplied.
    #[error("vectors must have at least one dimension")]
    EmptyVector,

    /// A vector contained a NaN or an infinity, which would poison every
    /// distance computation it participates in.
    #[error("vector {id} contains a non-finite component at index {component}")]
    NonFiniteVector {
        /// The identifier of the offending vector.
        id: String,
        /// The index of the first non-finite component.
        component: usize,
    },

    /// An identifier was inserted twice.
    #[error("duplicate vector id: {0}")]
    DuplicateId(String),

    /// A configuration parameter is out of range or inconsistent with another.
    #[error("invalid configuration for `{field}`: {reason}")]
    InvalidConfig {
        /// The offending field's name.
        field: String,
        /// Why it is invalid.
        reason: String,
    },

    /// The predicate AST is malformed (an inverted range, a NaN bound, an
    /// empty attribute name, ...).
    #[error("invalid predicate: {reason}")]
    InvalidPredicate {
        /// Why the predicate is malformed.
        reason: String,
    },
}

// ── Small shared helper ──────────────────────────────────────────────────────

/// Increment `key`'s counter in `counts`, returning `true` when the key was
/// newly created (used by the estimator and the inverted index to track
/// distinct-value cardinality without a second pass).
pub(crate) fn bump_count<K: std::hash::Hash + Eq>(counts: &mut HashMap<K, u64>, key: K) -> bool {
    match counts.entry(key) {
        Entry::Occupied(mut occupied) => {
            *occupied.get_mut() += 1;
            false
        }
        Entry::Vacant(vacant) => {
            vacant.insert(1);
            true
        }
    }
}
