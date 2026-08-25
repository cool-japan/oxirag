//! [`FilteredVectorIndex`]: the public surface — build, insert, plan, search.
//!
//! The index owns four cooperating structures:
//!
//! | Structure | Answers |
//! |---|---|
//! | the Vamana proximity graph | "which vectors are near this query?" |
//! | the inverted attribute index | "which vectors satisfy this predicate?" (cheaply, without scanning) |
//! | [`SelectivityEstimator`] | "roughly how many satisfy it?" (before touching either of the above) |
//! | [`StrategySelector`] | "given that, which of the three plans should run?" |
//!
//! Every strategy verifies the real predicate against the real metadata before
//! admitting a hit, so **no hit that fails the predicate is ever returned**,
//! whatever the estimator believed and whichever plan ran.

use std::collections::HashMap;

use super::attr_index::FilteredAttributeIndex;
use super::graph::{AcornSearchParams, FilteredGraph, FilteredPredicateOracle};
use super::predicate::FilterPredicate;
use super::selectivity::{SelectivityEstimator, StrategySelector};
use super::types::{
    FilterStrategy, FilteredHit, FilteredMetadata, FilteredSearchConfig, FilteredSearchError,
    FilteredSearchStats, SelectivityEstimate,
};

// ── FilteredVectorRecord ─────────────────────────────────────────────────────

/// One vector, its identifier, and its attributes — the unit
/// [`FilteredVectorIndex::build`] consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredVectorRecord {
    /// Caller-supplied identifier, echoed back on every [`FilteredHit`].
    pub id: String,
    /// The embedding.
    pub vector: Vec<f32>,
    /// The attributes predicates are evaluated against.
    pub metadata: FilteredMetadata,
}

impl FilteredVectorRecord {
    /// Construct a record.
    #[must_use]
    pub fn new(id: impl Into<String>, vector: Vec<f32>, metadata: FilteredMetadata) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata,
        }
    }
}

// ── Predicate oracle over the index's metadata ───────────────────────────────

/// Answers the graph's "does node `id` match?" questions from the index's
/// metadata, memoizing as it goes.
///
/// Memoization is not an optimization here so much as a correctness-of-
/// -measurement concern: the ACORN two-hop step revisits the same node many
/// times per search, and without a cache the reported
/// [`predicate_evaluations`](FilteredSearchStats::predicate_evaluations) would
/// measure the traversal's redundancy rather than its real predicate cost.
struct MetadataOracle<'a> {
    predicate: &'a FilterPredicate,
    metadata: &'a [FilteredMetadata],
    cache: HashMap<u32, bool>,
    evaluations: usize,
}

impl<'a> MetadataOracle<'a> {
    fn new(predicate: &'a FilterPredicate, metadata: &'a [FilteredMetadata]) -> Self {
        Self {
            predicate,
            metadata,
            cache: HashMap::new(),
            evaluations: 0,
        }
    }
}

impl FilteredPredicateOracle for MetadataOracle<'_> {
    fn matches(&mut self, id: u32) -> bool {
        if let Some(&cached) = self.cache.get(&id) {
            return cached;
        }
        let verdict = self
            .metadata
            .get(id as usize)
            .is_some_and(|metadata| self.predicate.matches(metadata));
        self.cache.insert(id, verdict);
        self.evaluations += 1;
        verdict
    }

    fn evaluations(&self) -> usize {
        self.evaluations
    }
}

// ── Bounded top-k accumulator (used by the exhaustive strategies) ────────────

/// Keeps the `capacity` closest `(node, distance)` pairs, ties broken by
/// ascending node id so that results are deterministic.
struct TopK {
    capacity: usize,
    entries: Vec<(u32, f32)>,
}

impl TopK {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity.saturating_add(1)),
        }
    }

    fn offer(&mut self, id: u32, distance: f32) {
        if self.capacity == 0 {
            return;
        }
        let position = self
            .entries
            .binary_search_by(|(other_id, other_distance)| {
                other_distance
                    .partial_cmp(&distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(other_id.cmp(&id))
            })
            .unwrap_or_else(|insertion_point| insertion_point);
        if position >= self.capacity {
            return;
        }
        self.entries.insert(position, (id, distance));
        self.entries.truncate(self.capacity);
    }

    fn into_sorted(self) -> Vec<(u32, f32)> {
        self.entries
    }
}

// ── FilteredVectorIndex ──────────────────────────────────────────────────────

/// A predicate-constrained approximate nearest-neighbor index.
///
/// See the [module documentation](super) for the algorithm, the cost model, and
/// the recall argument.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "filtered-vector-search")]
/// # {
/// use oxirag::filtered_vector_search::{
///     FilterPredicate, FilteredMetadata, FilteredSearchConfig, FilteredVectorIndex,
///     FilteredVectorRecord,
/// };
///
/// let records = vec![
///     FilteredVectorRecord::new(
///         "a",
///         vec![1.0, 0.0],
///         FilteredMetadata::new().with("lang", "en").with("year", 2024_i64),
///     ),
///     FilteredVectorRecord::new(
///         "b",
///         vec![0.9, 0.1],
///         FilteredMetadata::new().with("lang", "ja").with("year", 2021_i64),
///     ),
/// ];
///
/// let index = FilteredVectorIndex::build(2, FilteredSearchConfig::default(), records)
///     .expect("records are well-formed");
///
/// // Nearest to [1, 0] is "a" — but only "b" is Japanese.
/// let predicate = FilterPredicate::eq("lang", "ja");
/// let hits = index
///     .search(&[1.0, 0.0], 1, &predicate)
///     .expect("query is well-formed");
///
/// assert_eq!(hits.len(), 1);
/// assert_eq!(hits[0].id, "b");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FilteredVectorIndex {
    config: FilteredSearchConfig,
    graph: FilteredGraph,
    ids: Vec<String>,
    id_lookup: HashMap<String, u32>,
    metadata: Vec<FilteredMetadata>,
    attribute_index: FilteredAttributeIndex,
    estimator: SelectivityEstimator,
    selector: StrategySelector,
}

impl FilteredVectorIndex {
    // ── Construction ─────────────────────────────────────────────────────

    /// Create an empty index over `dimension`-dimensional vectors.
    ///
    /// # Errors
    ///
    /// [`FilteredSearchError::EmptyVector`] when `dimension` is zero, or
    /// [`FilteredSearchError::InvalidConfig`] when `config` is inconsistent.
    pub fn new(
        dimension: usize,
        config: FilteredSearchConfig,
    ) -> Result<Self, FilteredSearchError> {
        if dimension == 0 {
            return Err(FilteredSearchError::EmptyVector);
        }
        config.validate()?;

        let graph = FilteredGraph::new(
            dimension,
            config.metric,
            config.graph_degree,
            config.build_beam_width,
            config.prune_alpha,
            config.seed,
        );
        let estimator = SelectivityEstimator::from_config(&config);
        let selector = StrategySelector::from_config(&config);
        let attribute_index = FilteredAttributeIndex::new(config.max_postings_per_attr);

        Ok(Self {
            config,
            graph,
            ids: Vec::new(),
            id_lookup: HashMap::new(),
            metadata: Vec::new(),
            attribute_index,
            estimator,
            selector,
        })
    }

    /// Create an empty index with the default configuration.
    ///
    /// # Errors
    ///
    /// [`FilteredSearchError::EmptyVector`] when `dimension` is zero.
    pub fn with_default_config(dimension: usize) -> Result<Self, FilteredSearchError> {
        Self::new(dimension, FilteredSearchConfig::default())
    }

    /// Build an index from a batch of records, then [`optimize`](Self::optimize)
    /// it.
    ///
    /// This is the preferred entry point: the two-pass Vamana rebuild that
    /// `optimize` performs produces a materially better-connected graph than
    /// the sequence of incremental insertions alone, and better connectivity is
    /// exactly what a *filtered* traversal depends on.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`FilteredVectorIndex::insert_record`].
    pub fn build(
        dimension: usize,
        config: FilteredSearchConfig,
        records: impl IntoIterator<Item = FilteredVectorRecord>,
    ) -> Result<Self, FilteredSearchError> {
        let mut index = Self::new(dimension, config)?;
        for record in records {
            index.insert_record(record)?;
        }
        index.optimize();
        Ok(index)
    }

    /// Insert one vector.
    ///
    /// The vector is linked into the graph immediately (a single Vamana
    /// insertion), so the index is queryable after every insert. Call
    /// [`optimize`](Self::optimize) after a batch of inserts to rebuild the
    /// graph properly.
    ///
    /// # Errors
    ///
    /// - [`FilteredSearchError::DimensionMismatch`] when the vector's length
    ///   disagrees with the index's dimension.
    /// - [`FilteredSearchError::NonFiniteVector`] when it contains a NaN or an
    ///   infinity, which would poison every distance it participates in.
    /// - [`FilteredSearchError::DuplicateId`] when `id` is already present.
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: FilteredMetadata,
    ) -> Result<(), FilteredSearchError> {
        let id = id.into();
        if vector.len() != self.graph.dimension() {
            return Err(FilteredSearchError::DimensionMismatch {
                expected: self.graph.dimension(),
                actual: vector.len(),
            });
        }
        if let Some(component) = vector.iter().position(|value| !value.is_finite()) {
            return Err(FilteredSearchError::NonFiniteVector { id, component });
        }
        if self.id_lookup.contains_key(&id) {
            return Err(FilteredSearchError::DuplicateId(id));
        }

        let node = self.graph.insert(vector);
        self.id_lookup.insert(id.clone(), node);
        self.ids.push(id);
        self.attribute_index.observe(node, &metadata);
        self.estimator.observe(&metadata);
        self.metadata.push(metadata);
        Ok(())
    }

    /// Insert one [`FilteredVectorRecord`].
    ///
    /// # Errors
    ///
    /// See [`FilteredVectorIndex::insert`].
    pub fn insert_record(
        &mut self,
        record: FilteredVectorRecord,
    ) -> Result<(), FilteredSearchError> {
        self.insert(record.id, record.vector, record.metadata)
    }

    /// Rebuild the proximity graph with the full two-pass Vamana algorithm and
    /// merge the inverted index's pending numeric values into its sorted
    /// arrays.
    ///
    /// Idempotent, and safe to call at any time. Cost is `O(N * L * R * d)`.
    pub fn optimize(&mut self) {
        self.attribute_index.finalize();
        self.graph.rebuild();
    }

    // ── Introspection ────────────────────────────────────────────────────

    /// The number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the index holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The vector dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.graph.dimension()
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &FilteredSearchConfig {
        &self.config
    }

    /// The metadata attached to `id`, if it is indexed.
    #[must_use]
    pub fn metadata_of(&self, id: &str) -> Option<&FilteredMetadata> {
        self.id_lookup
            .get(id)
            .and_then(|node| self.metadata.get(*node as usize))
    }

    /// The selectivity estimator, for callers who want to inspect the
    /// statistics directly.
    #[must_use]
    pub fn estimator(&self) -> &SelectivityEstimator {
        &self.estimator
    }

    // ── Planning ─────────────────────────────────────────────────────────

    /// Estimate what fraction of the corpus `predicate` admits, without
    /// touching a vector.
    ///
    /// # Errors
    ///
    /// [`FilteredSearchError::InvalidPredicate`] when the AST is malformed.
    pub fn estimate_selectivity(
        &self,
        predicate: &FilterPredicate,
    ) -> Result<SelectivityEstimate, FilteredSearchError> {
        predicate.validate()?;
        Ok(self.estimator.estimate(predicate))
    }

    /// The strategy [`FilterStrategy::Auto`] would choose for this predicate
    /// and `top_k`, together with the estimate that drove the choice.
    ///
    /// Exposed so that callers (and the test suite) can check the planner's
    /// reasoning without running a search.
    ///
    /// # Errors
    ///
    /// [`FilteredSearchError::InvalidPredicate`] when the AST is malformed.
    pub fn plan(
        &self,
        predicate: &FilterPredicate,
        top_k: usize,
    ) -> Result<(FilterStrategy, SelectivityEstimate), FilteredSearchError> {
        let estimate = self.estimate_selectivity(predicate)?;
        let strategy = self.selector.select(&estimate, top_k);
        Ok((strategy, estimate))
    }

    // ── Search ───────────────────────────────────────────────────────────

    /// Search for the `top_k` nearest vectors satisfying `predicate`, letting
    /// the planner choose the strategy.
    ///
    /// # Errors
    ///
    /// See [`FilteredVectorIndex::search_with_strategy`].
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        predicate: &FilterPredicate,
    ) -> Result<Vec<FilteredHit>, FilteredSearchError> {
        self.search_with_stats(query, top_k, predicate)
            .map(|(hits, _)| hits)
    }

    /// [`search`](Self::search), also returning the instrumentation.
    ///
    /// # Errors
    ///
    /// See [`FilteredVectorIndex::search_with_strategy`].
    pub fn search_with_stats(
        &self,
        query: &[f32],
        top_k: usize,
        predicate: &FilterPredicate,
    ) -> Result<(Vec<FilteredHit>, FilteredSearchStats), FilteredSearchError> {
        self.search_with_strategy(query, top_k, predicate, FilterStrategy::Auto)
    }

    /// Search with an explicitly chosen strategy.
    ///
    /// Passing [`FilterStrategy::Auto`] is equivalent to
    /// [`search_with_stats`](Self::search_with_stats). The other three run the
    /// named plan unconditionally — including
    /// [`FilterStrategy::PostFilter`] in the
    /// regime where it is known to fail, which is deliberate: it is the
    /// baseline the module's headline test measures itself against, and it is
    /// only honest to let it fail visibly.
    ///
    /// # Errors
    ///
    /// - [`FilteredSearchError::DimensionMismatch`] when `query` has the wrong
    ///   length.
    /// - [`FilteredSearchError::NonFiniteVector`] when `query` contains a NaN
    ///   or an infinity.
    /// - [`FilteredSearchError::InvalidPredicate`] when the AST is malformed.
    pub fn search_with_strategy(
        &self,
        query: &[f32],
        top_k: usize,
        predicate: &FilterPredicate,
        strategy: FilterStrategy,
    ) -> Result<(Vec<FilteredHit>, FilteredSearchStats), FilteredSearchError> {
        if query.len() != self.graph.dimension() {
            return Err(FilteredSearchError::DimensionMismatch {
                expected: self.graph.dimension(),
                actual: query.len(),
            });
        }
        if let Some(component) = query.iter().position(|value| !value.is_finite()) {
            return Err(FilteredSearchError::NonFiniteVector {
                id: "<query>".to_string(),
                component,
            });
        }
        predicate.validate()?;

        let mut stats = FilteredSearchStats::default();

        let resolved = match strategy {
            FilterStrategy::Auto => {
                let estimate = self.estimator.estimate(predicate);
                let chosen = self.selector.select(&estimate, top_k);
                stats.selectivity = Some(estimate);
                chosen
            }
            concrete => concrete,
        };
        stats.strategy = resolved;

        if self.is_empty() || top_k == 0 {
            return Ok((Vec::new(), stats));
        }

        let query_norm = FilteredGraph::norm_of(query);
        let ranked = match resolved {
            FilterStrategy::PreFilter => {
                self.run_pre_filter(query, query_norm, top_k, predicate, &mut stats)
            }
            FilterStrategy::PostFilter => {
                self.run_post_filter(query, query_norm, top_k, predicate, &mut stats)
            }
            // `Auto` was resolved to a concrete plan above and cannot reach
            // here; treating it as `InFilter` keeps the match exhaustive
            // without an unreachable panic.
            FilterStrategy::InFilter | FilterStrategy::Auto => {
                self.run_in_filter(query, query_norm, top_k, predicate, &mut stats)
            }
        };

        Ok((self.materialize(ranked), stats))
    }

    /// An exhaustive, exact search over every indexed vector.
    ///
    /// This deliberately ignores the inverted index and the graph alike: it
    /// evaluates the predicate against all `N` records and computes all `N`
    /// distances. It exists as *ground truth* — for evaluating the approximate
    /// strategies, and for callers who need a guaranteed-correct answer at any
    /// price.
    ///
    /// # Errors
    ///
    /// See [`FilteredVectorIndex::search_with_strategy`].
    pub fn exact_search(
        &self,
        query: &[f32],
        top_k: usize,
        predicate: &FilterPredicate,
    ) -> Result<Vec<FilteredHit>, FilteredSearchError> {
        if query.len() != self.graph.dimension() {
            return Err(FilteredSearchError::DimensionMismatch {
                expected: self.graph.dimension(),
                actual: query.len(),
            });
        }
        if let Some(component) = query.iter().position(|value| !value.is_finite()) {
            return Err(FilteredSearchError::NonFiniteVector {
                id: "<query>".to_string(),
                component,
            });
        }
        predicate.validate()?;
        if self.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let query_norm = FilteredGraph::norm_of(query);
        let mut top = TopK::new(top_k);
        for (node, metadata) in self.metadata.iter().enumerate() {
            if !predicate.matches(metadata) {
                continue;
            }
            #[allow(clippy::cast_possible_truncation)]
            let node = node as u32;
            top.offer(node, self.graph.distance_to(query, query_norm, node));
        }
        Ok(self.materialize(top.into_sorted()))
    }

    // ── Strategy implementations ─────────────────────────────────────────

    /// Materialize the matching subset from the inverted index, then scan it
    /// exhaustively.
    ///
    /// Exact by construction: every record that satisfies the predicate has its
    /// distance computed, so the top-`k` of the matching set is returned
    /// verbatim. The inverted index may hand back a *superset* of the matching
    /// set, or refuse to answer at all — in which case the fallback is the full
    /// id range. Either way the subsequent [`FilterPredicate::matches`] pass
    /// reduces it to exactly the matching set, so correctness never depends on
    /// how clever the resolver was; only the cost does.
    fn run_pre_filter(
        &self,
        query: &[f32],
        query_norm: f32,
        top_k: usize,
        predicate: &FilterPredicate,
        stats: &mut FilteredSearchStats,
    ) -> Vec<(u32, f32)> {
        let candidates: Vec<u32> = self.attribute_index.resolve(predicate).map_or_else(
            || {
                // The inverted index declined to answer (a negated superset, an
                // attribute past its cardinality guard, ...). Fall back to the
                // full id range: slower, and every bit as exact once the
                // verification pass below has run.
                #[allow(clippy::cast_possible_truncation)]
                let all: Vec<u32> = (0..self.attribute_index.total() as u32).collect();
                all
            },
            |resolved| resolved.ids,
        );
        stats.candidates_considered = candidates.len();

        let mut top = TopK::new(top_k);
        for node in candidates {
            let Some(metadata) = self.metadata.get(node as usize) else {
                continue;
            };
            stats.predicate_evaluations += 1;
            if !predicate.matches(metadata) {
                continue;
            }
            stats.matched_candidates += 1;
            stats.distance_computations += 1;
            top.offer(node, self.graph.distance_to(query, query_norm, node));
        }
        top.into_sorted()
    }

    /// Search the graph *unconstrained* for `top_k * post_filter_multiplier`
    /// candidates, then drop the ones that fail the predicate.
    ///
    /// # The failure mode, stated plainly
    ///
    /// The over-fetch is a fixed multiple `m` of `k`. If the predicate is
    /// independent of the query's proximity ranking — the ordinary case — then
    /// among the unconstrained top `k * m`, roughly `s * k * m` satisfy it.
    /// Filling the result set therefore needs
    ///
    /// ```text
    /// s * k * m >= k    <=>    s >= 1 / m
    /// ```
    ///
    /// With the default `m = 10`, any predicate matching less than 10% of the
    /// corpus starts returning short, and by `s = 1%` the method returns
    /// *roughly one* correct hit out of ten — a recall of about `0.1`, and
    /// often `0.0`. No beam width fixes this: the matching vectors are simply
    /// not in the list that was fetched, because the list was ranked without
    /// reference to the predicate at all.
    ///
    /// Enlarging `m` to `1 / s` restores recall — at the cost of an
    /// unconstrained search for `k / s` candidates, which at `s = 0.1%` means
    /// fetching a thousand times `k`. That is not an over-fetch, that is a
    /// scan with extra steps, and it is why the other two strategies exist.
    fn run_post_filter(
        &self,
        query: &[f32],
        query_norm: f32,
        top_k: usize,
        predicate: &FilterPredicate,
        stats: &mut FilteredSearchStats,
    ) -> Vec<(u32, f32)> {
        let over_fetch = top_k
            .saturating_mul(self.config.post_filter_multiplier)
            .max(top_k);
        let beam_width = self.config.search_beam_width.max(over_fetch);

        let raw = self.graph.greedy_search(
            query,
            query_norm,
            over_fetch,
            beam_width,
            self.config.seed_count,
            stats,
        );

        let mut kept: Vec<(u32, f32)> = Vec::with_capacity(top_k);
        for (node, distance) in raw {
            let Some(metadata) = self.metadata.get(node as usize) else {
                continue;
            };
            stats.predicate_evaluations += 1;
            if !predicate.matches(metadata) {
                continue;
            }
            stats.matched_candidates += 1;
            if kept.len() < top_k {
                kept.push((node, distance));
            }
        }
        kept
    }

    /// The ACORN traversal — see the [module documentation](super) for the
    /// algorithm and the recall argument.
    fn run_in_filter(
        &self,
        query: &[f32],
        query_norm: f32,
        top_k: usize,
        predicate: &FilterPredicate,
        stats: &mut FilteredSearchStats,
    ) -> Vec<(u32, f32)> {
        let params = AcornSearchParams {
            top_k,
            beam_width: self.config.search_beam_width,
            matching_reserve: self.config.matching_reserve(),
            gamma: self.config.neighbor_expansion_gamma,
            min_predicate_neighbors: self.config.min_predicate_neighbors,
            max_visits: self.config.max_visits(),
            seed_count: self.config.seed_count,
        };
        let mut oracle = MetadataOracle::new(predicate, &self.metadata);
        self.graph
            .acorn_search(query, query_norm, params, &mut oracle, stats)
    }

    /// Turn internal `(node, distance)` pairs into public hits.
    fn materialize(&self, ranked: Vec<(u32, f32)>) -> Vec<FilteredHit> {
        ranked
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (node, distance))| {
                self.ids.get(node as usize).map(|id| {
                    FilteredHit::new(id.clone(), distance, self.graph.score_of(distance), rank)
                })
            })
            .collect()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
impl FilteredVectorIndex {
    /// The internal graph, for white-box tests of connectivity and degree.
    pub(crate) fn graph(&self) -> &FilteredGraph {
        &self.graph
    }

    /// The internal node id for a caller-supplied id.
    pub(crate) fn node_of(&self, id: &str) -> Option<u32> {
        self.id_lookup.get(id).copied()
    }
}
