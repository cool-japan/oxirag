//! The proximity graph: a self-contained Vamana-style index, plus the two
//! traversals that run over it — an ordinary unconstrained greedy search, and
//! the predicate-aware ACORN traversal that is the point of this module.
//!
//! # The graph
//!
//! Construction follows Vamana (the graph underlying `DiskANN`): start from a
//! random `R`-regular digraph, then make two passes over a random permutation
//! of the nodes; for each node `p`, run a greedy search from the medoid,
//! collect **every** node whose distance was computed along the way, and hand
//! that visited set to `robust_prune`, which selects `p`'s out-neighbors.
//!
//! `robust_prune` is what makes the graph navigable rather than merely
//! locally-dense. Repeatedly take the closest surviving candidate `p*`, keep
//! it, and then *occlude* every remaining candidate `p'` that satisfies
//!
//! ```text
//! alpha * d(p*, p') <= d(p, p')
//! ```
//!
//! i.e. drop `p'` whenever routing to it *via* `p*` is (up to a slack factor
//! `alpha`) no worse than the direct edge. With `alpha == 1` this yields the
//! relative neighborhood graph, which is beautifully sparse and has a terrible
//! diameter. With `alpha > 1` some of those occluded long-range edges survive,
//! and the graph acquires the small-world property that makes greedy descent
//! converge in a logarithmic number of hops. The first pass uses `alpha = 1`
//! (building a clean skeleton), the second uses the configured `alpha`
//! (adding the long-range shortcuts) — exactly as Vamana prescribes.
//!
//! Everything here is deterministic: the only randomness is a hand-rolled
//! `SplitMix64` seeded from the configuration, so two indexes built from the
//! same records with the same seed are identical.
//!
//! # ACORN: why a filtered traversal needs to route through non-matching nodes
//!
//! Suppose you want the nearest neighbors of `q` *among the vectors satisfying
//! a predicate `P`*. The obvious thing to do is run greedy search but refuse to
//! traverse into any node failing `P`. This does not work, and the reason is
//! structural rather than incidental.
//!
//! Greedy graph search only works because the graph is navigable: from
//! anywhere, a short path of ever-closer nodes leads to the query's
//! neighborhood. But navigability is a property of the *whole* graph. Delete
//! the nodes failing `P` and you are left with the **induced subgraph** on the
//! matching set — and an induced subgraph of a sparse navigable graph is, in
//! general, neither navigable nor even *connected*. At a selectivity of `s`,
//! a node of degree `R` retains on average only `s * R` of its edges. Once
//! `s * R < 1` — which for `R = 32` means any predicate more selective than
//! about 3% — the induced subgraph is below the percolation threshold and
//! shatters into isolated fragments. A traversal that refuses to leave the
//! matching set dead-ends after a hop or two, and recall collapses. This is not
//! a tuning problem; no beam width fixes a disconnected graph.
//!
//! ACORN's answer, implemented in [`FilteredGraph::acorn_search`], is to
//! separate two things the naive traversal conflates:
//!
//! - **What may be returned.** Only nodes satisfying `P` are ever admitted to
//!   the result heap. This is non-negotiable and holds unconditionally.
//! - **What may be traversed.** *Every* node may be traversed — including
//!   nodes that fail `P`. They are enqueued as **routing nodes**: expanded
//!   like any other node, never returned.
//!
//! Because the traversal is free to move through the whole graph, it inherits
//! the whole graph's navigability. Restricting the *results* costs nothing
//! structurally.
//!
//! On top of that, ACORN adds the **two-hop expansion** that gives the module
//! its teeth. When expanding a node `u`, a neighbor `v` that fails `P` would
//! ordinarily contribute nothing but a future pool slot. Instead, the
//! traversal immediately reaches *through* `v` and harvests the members of
//! `v`'s own neighbor list that *do* satisfy `P`. A single expansion therefore
//! discovers matches up to two hops away — reaching `O(R^2)` candidates
//! instead of `O(R)` — which is precisely what restores a usable supply of
//! matching candidates when `s * R < 1`. It also finds matches that would
//! otherwise be missed entirely, since `v` may well be evicted from the
//! bounded candidate pool before it is ever popped.
//!
//! The expansion is *adaptive*: it is skipped whenever `u` already has at least
//! `min_predicate_neighbors` matching neighbors, because at that density the
//! predicate subgraph is navigable on its own and the two-hop tax buys nothing.
//! So `InFilter` degrades to roughly the cost of an unfiltered search as `s`
//! approaches 1, and pays its full `gamma`-fold cost only where that cost is
//! the price of not returning garbage.
//!
//! The gate is a **separate knob** from `gamma`, and that separation is not
//! fussiness. Deriving it from `gamma` — the obvious `degree / gamma` — couples
//! "is the subgraph too sparse to navigate?" to "how hard should I work when it
//! is?", so that turning the expansion up (`gamma -> degree`) drives the gate
//! down to 1 and the two-hop step stops firing precisely as you ask for more of
//! it. That is measurable, not theoretical: on this module's own fixture the
//! coupled version lost two points of recall at 10% selectivity, for strictly
//! more configured effort.
//!
//! # What is and is not guaranteed
//!
//! - **Every returned hit satisfies the predicate.** Unconditional: the result
//!   heap admits nothing else.
//! - **With an unbounded budget the traversal is exhaustive.** Routing nodes
//!   let the frontier reach every node in the entry points' connected
//!   component, so with `search_beam_width` and `max_visits` large enough,
//!   `acorn_search` returns the exact filtered top-`k`. Recall is therefore a
//!   pure function of budget, not a structural limit — which is exactly what
//!   the naive filtered traversal cannot say.
//! - **With a bounded budget it is approximate**, and its recall degrades as
//!   `s` falls, because a bounded, distance-ordered pool clusters around `q`
//!   and a sufficiently rare matching set may lie outside what that pool
//!   reaches. This is a real limit, and it is the reason the strategy selector
//!   hands sufficiently selective predicates to `PreFilter` instead — which is
//!   *exact*, and at that selectivity also *cheaper*.

use std::collections::HashSet;

use super::types::{FilteredDistanceMetric, FilteredSearchStats};

// ── SplitMix64 ───────────────────────────────────────────────────────────────

/// A hand-rolled `SplitMix64` PRNG.
///
/// The crate takes no dependency on `rand`, and graph construction needs a
/// deterministic source of randomness (for the initial random regular graph and
/// for the insertion permutation). `SplitMix64` is the standard choice: a
/// single multiply-xorshift finalizer over a counter, statistically sound for
/// this purpose and reproducible bit-for-bit across platforms.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 raw bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `[0, bound)`. Returns `0` when `bound` is `0`.
    pub(crate) fn next_bounded(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        #[allow(clippy::cast_possible_truncation)]
        let value = (self.next_u64() % bound as u64) as usize;
        value
    }

    /// A float in `[0, 1)`. Used by the test suite to synthesize corpora.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn next_f32(&mut self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit
    }

    /// Fisher-Yates shuffle of `items`.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let swap_with = self.next_bounded(index + 1);
            items.swap(index, swap_with);
        }
    }
}

// ── Predicate oracle ─────────────────────────────────────────────────────────

/// The graph's view of a metadata predicate.
///
/// The graph does not know what metadata *is* — it only needs to ask, of a node
/// id, "does this one match?". Implementations are expected to memoize, because
/// the two-hop expansion asks about the same node many times over the course of
/// a single search; [`FilteredPredicateOracle::evaluations`] reports the number
/// of *uncached* evaluations, so the statistics measure real predicate work.
pub(crate) trait FilteredPredicateOracle {
    /// Whether node `id` satisfies the predicate.
    fn matches(&mut self, id: u32) -> bool;

    /// The number of genuine (cache-missing) predicate evaluations performed.
    fn evaluations(&self) -> usize;
}

// ── Bounded result collector ─────────────────────────────────────────────────

/// Keeps the `capacity` closest `(id, distance)` pairs offered to it.
///
/// Deliberately *not* the candidate pool: a matching node discovered far from
/// the query is worth remembering as a result even when it is too far to earn a
/// pool slot, and conversely a routing node earns a pool slot without ever
/// being a result.
#[derive(Debug)]
struct BoundedResults {
    capacity: usize,
    entries: Vec<(u32, f32)>,
}

impl BoundedResults {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity.saturating_add(1)),
        }
    }

    /// Offer a candidate; keeps it if it belongs among the best `capacity`.
    ///
    /// Ties (equal distances) are broken by ascending node id, so the output is
    /// deterministic regardless of discovery order.
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

// ── Candidate pool ───────────────────────────────────────────────────────────

/// One entry of the search-time candidate pool.
#[derive(Debug, Clone, Copy)]
struct PoolEntry {
    id: u32,
    distance: f32,
    /// Whether this node satisfies the predicate. `false` marks a **routing
    /// node**: traversable, never returnable.
    matched: bool,
    /// Whether this node's neighbor list has already been expanded.
    expanded: bool,
}

/// Insert `entry` into a distance-sorted pool, keeping the ordering.
fn pool_insert(pool: &mut Vec<PoolEntry>, entry: PoolEntry) {
    let position = pool
        .binary_search_by(|other| {
            other
                .distance
                .partial_cmp(&entry.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(other.id.cmp(&entry.id))
        })
        .unwrap_or_else(|insertion_point| insertion_point);
    pool.insert(position, entry);
}

/// Shrink a distance-sorted pool to `capacity`, while guaranteeing that at
/// least `reserve` predicate-*matching* entries survive whenever that many are
/// available.
///
/// # Why the reserve exists
///
/// Under a selective predicate the pool fills with routing nodes, which cluster
/// tightly around the query — they are, after all, simply the query's nearest
/// neighbors, most of which fail the predicate. A plain distance-ordered
/// truncation therefore evicts every matching node that is not itself among the
/// query's `L` nearest vectors overall. Those matches still reach the *result*
/// heap, so they are not lost outright — but they are never **expanded**, which
/// means the traversal never follows them into the region of the graph where
/// their neighbors (also likely to match, since attributes correlate with
/// position far more often than not) are waiting. The search discovers the
/// matching region and then declines to explore it.
///
/// Reserving a slice of the pool for matching nodes fixes exactly that: the
/// closest matches always get expanded, however crowded the query's immediate
/// neighborhood is with non-matching vectors.
fn pool_truncate(pool: &mut Vec<PoolEntry>, capacity: usize, reserve: usize) {
    if pool.len() <= capacity {
        return;
    }
    let reserve = reserve.min(capacity);

    let mut keep = vec![false; pool.len()];
    for slot in keep.iter_mut().take(capacity) {
        *slot = true;
    }

    if reserve > 0 {
        let matching_kept = pool
            .iter()
            .take(capacity)
            .filter(|entry| entry.matched)
            .count();
        let shortfall = reserve.saturating_sub(matching_kept);
        if shortfall > 0 {
            // The closest matching entries among those about to be dropped.
            let promote: Vec<usize> = pool
                .iter()
                .enumerate()
                .skip(capacity)
                .filter(|(_, entry)| entry.matched)
                .map(|(index, _)| index)
                .take(shortfall)
                .collect();

            // The farthest routing entries among those we were about to keep.
            let evict: Vec<usize> = (0..capacity)
                .rev()
                .filter(|&index| !pool[index].matched)
                .take(promote.len())
                .collect();

            for (&promoted, &evicted) in promote.iter().zip(evict.iter()) {
                keep[promoted] = true;
                keep[evicted] = false;
            }
        }
    }

    let mut index = 0;
    pool.retain(|_| {
        let survives = keep[index];
        index += 1;
        survives
    });
}

// ── Search parameters ────────────────────────────────────────────────────────

/// The knobs [`FilteredGraph::acorn_search`] reads, bundled so the signature
/// stays legible.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AcornSearchParams {
    /// How many hits to return.
    pub(crate) top_k: usize,
    /// Size of the candidate pool.
    pub(crate) beam_width: usize,
    /// Pool slots guaranteed to predicate-matching nodes.
    pub(crate) matching_reserve: usize,
    /// ACORN neighbor-expansion factor: how many predicate-failing neighbors a
    /// single expansion may route *through*.
    pub(crate) gamma: usize,
    /// The two-hop gate: below this many predicate-matching neighbors, an
    /// expansion routes through its failing ones.
    pub(crate) min_predicate_neighbors: usize,
    /// Hard ceiling on node expansions.
    pub(crate) max_visits: usize,
    /// How many entry points to seed the traversal from.
    pub(crate) seed_count: usize,
}

// ── FilteredGraph ────────────────────────────────────────────────────────────

/// A Vamana proximity graph over the indexed vectors.
///
/// Owns the vectors (and their cached norms, for the cosine metric); metadata,
/// identifiers and statistics live in
/// [`FilteredVectorIndex`](super::FilteredVectorIndex).
#[derive(Debug, Clone)]
pub(crate) struct FilteredGraph {
    dimension: usize,
    metric: FilteredDistanceMetric,
    degree: usize,
    build_beam_width: usize,
    prune_alpha: f64,
    seed: u64,
    vectors: Vec<Vec<f32>>,
    norms: Vec<f32>,
    neighbors: Vec<Vec<u32>>,
    medoid: u32,
}

impl FilteredGraph {
    /// Create an empty graph.
    pub(crate) fn new(
        dimension: usize,
        metric: FilteredDistanceMetric,
        degree: usize,
        build_beam_width: usize,
        prune_alpha: f64,
        seed: u64,
    ) -> Self {
        Self {
            dimension,
            metric,
            degree: degree.max(1),
            build_beam_width: build_beam_width.max(degree.max(1)),
            prune_alpha,
            seed,
            vectors: Vec::new(),
            norms: Vec::new(),
            neighbors: Vec::new(),
            medoid: 0,
        }
    }

    /// The number of vectors in the graph.
    pub(crate) fn len(&self) -> usize {
        self.vectors.len()
    }

    /// The vector dimension.
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    /// The out-neighbors of `id`.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn neighbors_of(&self, id: u32) -> &[u32] {
        &self.neighbors[id as usize]
    }

    /// The graph's entry point (the medoid, after a rebuild).
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn medoid(&self) -> u32 {
        self.medoid
    }

    // ── Distance ─────────────────────────────────────────────────────────

    /// The L2 norm of a vector, used to normalize cosine distances.
    pub(crate) fn norm_of(vector: &[f32]) -> f32 {
        vector
            .iter()
            .map(|component| component * component)
            .sum::<f32>()
            .sqrt()
    }

    /// Distance from an arbitrary query to an indexed node.
    ///
    /// The metric's *true* value is returned (not a monotone surrogate such as
    /// squared L2), because `robust_prune`'s `alpha` slack factor is defined on
    /// real distances and squaring them would silently reinterpret `alpha` as
    /// `sqrt(alpha)`. The extra square root is negligible next to the
    /// `d`-dimensional dot product it accompanies.
    pub(crate) fn distance_to(&self, query: &[f32], query_norm: f32, id: u32) -> f32 {
        let target = &self.vectors[id as usize];
        match self.metric {
            FilteredDistanceMetric::Cosine => {
                let target_norm = self.norms[id as usize];
                if query_norm <= 0.0 || target_norm <= 0.0 {
                    // A zero vector has no direction; define it to be
                    // orthogonal to everything.
                    return 1.0;
                }
                let dot: f32 = query
                    .iter()
                    .zip(target.iter())
                    .map(|(left, right)| left * right)
                    .sum();
                (1.0 - dot / (query_norm * target_norm)).clamp(0.0, 2.0)
            }
            FilteredDistanceMetric::Euclidean => query
                .iter()
                .zip(target.iter())
                .map(|(left, right)| (left - right) * (left - right))
                .sum::<f32>()
                .sqrt(),
        }
    }

    /// Distance between two indexed nodes.
    fn distance_between(&self, left: u32, right: u32) -> f32 {
        let query = &self.vectors[left as usize];
        let query_norm = self.norms[left as usize];
        self.distance_to(query, query_norm, right)
    }

    /// The similarity-style score reported to callers: larger is better.
    pub(crate) fn score_of(&self, distance: f32) -> f32 {
        match self.metric {
            FilteredDistanceMetric::Cosine => 1.0 - distance,
            FilteredDistanceMetric::Euclidean => -distance,
        }
    }

    // ── Insertion ────────────────────────────────────────────────────────

    /// Append a vector and wire it into the graph incrementally (a single
    /// Vamana insertion: greedy-search, robust-prune, add back-edges).
    ///
    /// The caller is responsible for having validated the vector's dimension
    /// and finiteness.
    pub(crate) fn insert(&mut self, vector: Vec<f32>) -> u32 {
        let norm = Self::norm_of(&vector);
        #[allow(clippy::cast_possible_truncation)]
        let id = self.vectors.len() as u32;
        self.vectors.push(vector);
        self.norms.push(norm);
        self.neighbors.push(Vec::new());

        if id == 0 {
            self.medoid = 0;
            return id;
        }

        let query = self.vectors[id as usize].clone();
        let visited = self.greedy_collect(&query, norm, self.build_beam_width, &[self.medoid], id);
        let pruned = self.robust_prune(id, visited, self.prune_alpha);
        self.neighbors[id as usize].clone_from(&pruned);

        for neighbor in pruned {
            self.add_back_edge(neighbor, id);
        }
        id
    }

    /// Add `from -> to`, re-pruning `from`'s neighbor list if it overflows.
    fn add_back_edge(&mut self, from: u32, to: u32) {
        if from == to {
            return;
        }
        if self.neighbors[from as usize].contains(&to) {
            return;
        }
        self.neighbors[from as usize].push(to);
        if self.neighbors[from as usize].len() > self.degree {
            let candidates: Vec<(u32, f32)> = self.neighbors[from as usize]
                .clone()
                .into_iter()
                .map(|candidate| (candidate, self.distance_between(from, candidate)))
                .collect();
            self.neighbors[from as usize] = self.robust_prune(from, candidates, self.prune_alpha);
        }
    }

    // ── Construction ─────────────────────────────────────────────────────

    /// Rebuild the whole graph with the two-pass Vamana algorithm.
    ///
    /// Pass one prunes with `alpha = 1` (the relative neighborhood graph — a
    /// clean, sparse skeleton); pass two prunes with the configured `alpha`
    /// (retaining long-range edges the skeleton occluded, which is what gives
    /// greedy descent its logarithmic hop count).
    pub(crate) fn rebuild(&mut self) {
        let count = self.vectors.len();
        if count == 0 {
            self.medoid = 0;
            return;
        }
        self.medoid = self.compute_medoid();

        // Start from a random R-regular digraph so that the very first greedy
        // searches have something to descend through.
        let mut rng = SplitMix64::new(self.seed);
        for id in 0..count {
            let mut chosen: Vec<u32> = Vec::with_capacity(self.degree.min(count));
            let wanted = self.degree.min(count.saturating_sub(1));
            let mut attempts = 0;
            while chosen.len() < wanted && attempts < wanted * 8 + 16 {
                attempts += 1;
                #[allow(clippy::cast_possible_truncation)]
                let candidate = rng.next_bounded(count) as u32;
                #[allow(clippy::cast_possible_truncation)]
                let self_id = id as u32;
                if candidate != self_id && !chosen.contains(&candidate) {
                    chosen.push(candidate);
                }
            }
            self.neighbors[id] = chosen;
        }

        #[allow(clippy::cast_possible_truncation)]
        let mut order: Vec<u32> = (0..count as u32).collect();
        rng.shuffle(&mut order);

        for alpha in [1.0_f64, self.prune_alpha] {
            for &id in &order {
                let query = self.vectors[id as usize].clone();
                let norm = self.norms[id as usize];
                let mut visited =
                    self.greedy_collect(&query, norm, self.build_beam_width, &[self.medoid], id);

                // Fold in the edges the node already has, so a pass can only
                // improve a neighbor list, never lose a good edge it cannot
                // rediscover.
                for &existing in &self.neighbors[id as usize].clone() {
                    if existing != id && !visited.iter().any(|(other, _)| *other == existing) {
                        visited.push((existing, self.distance_between(id, existing)));
                    }
                }

                let pruned = self.robust_prune(id, visited, alpha);
                self.neighbors[id as usize].clone_from(&pruned);

                for neighbor in pruned {
                    self.add_back_edge_with_alpha(neighbor, id, alpha);
                }
            }
        }
    }

    /// `add_back_edge`, but pruning with an explicit `alpha` (the two
    /// construction passes use different ones).
    fn add_back_edge_with_alpha(&mut self, from: u32, to: u32, alpha: f64) {
        if from == to || self.neighbors[from as usize].contains(&to) {
            return;
        }
        self.neighbors[from as usize].push(to);
        if self.neighbors[from as usize].len() > self.degree {
            let candidates: Vec<(u32, f32)> = self.neighbors[from as usize]
                .clone()
                .into_iter()
                .map(|candidate| (candidate, self.distance_between(from, candidate)))
                .collect();
            self.neighbors[from as usize] = self.robust_prune(from, candidates, alpha);
        }
    }

    /// The node closest to the corpus centroid — a far better entry point than
    /// an arbitrary node, because greedy descent from the "middle" of the data
    /// reaches any region in few hops.
    fn compute_medoid(&self) -> u32 {
        let count = self.vectors.len();
        if count == 0 {
            return 0;
        }
        let mut centroid = vec![0.0_f32; self.dimension];
        for vector in &self.vectors {
            for (accumulator, component) in centroid.iter_mut().zip(vector.iter()) {
                *accumulator += *component;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / count as f32;
        for component in &mut centroid {
            *component *= scale;
        }
        let centroid_norm = Self::norm_of(&centroid);

        let mut best_id = 0_u32;
        let mut best_distance = f32::INFINITY;
        for id in 0..count {
            #[allow(clippy::cast_possible_truncation)]
            let id = id as u32;
            let distance = self.distance_to(&centroid, centroid_norm, id);
            if distance < best_distance {
                best_distance = distance;
                best_id = id;
            }
        }
        best_id
    }

    /// The Vamana `RobustPrune`: select at most `degree` out-neighbors for
    /// `node` from `candidates`, dropping any candidate whose direct edge is
    /// (up to the slack `alpha`) redundant with a route through an
    /// already-selected, closer neighbor.
    ///
    /// See the [module documentation](self) for why `alpha > 1` is what makes
    /// the graph navigable.
    fn robust_prune(&self, node: u32, mut candidates: Vec<(u32, f32)>, alpha: f64) -> Vec<u32> {
        candidates.retain(|(id, _)| *id != node);
        candidates.sort_by(|(left_id, left_distance), (right_id, right_distance)| {
            left_distance
                .partial_cmp(right_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left_id.cmp(right_id))
        });
        candidates.dedup_by_key(|(id, _)| *id);

        #[allow(clippy::cast_possible_truncation)]
        let alpha = alpha as f32;
        let mut selected: Vec<u32> = Vec::with_capacity(self.degree);
        while let Some((chosen, _)) = candidates.first().copied() {
            selected.push(chosen);
            if selected.len() >= self.degree {
                break;
            }
            candidates.retain(|(candidate, distance_to_node)| {
                if *candidate == chosen {
                    return false;
                }
                // Keep `candidate` only if the detour through `chosen` is
                // genuinely worse than the direct edge; otherwise `chosen`
                // occludes it.
                let via_chosen = self.distance_between(chosen, *candidate);
                alpha * via_chosen > *distance_to_node
            });
        }
        selected
    }

    // ── Traversal ────────────────────────────────────────────────────────

    /// Deterministic entry points: the medoid first, then nodes dispersed over
    /// the id space.
    ///
    /// Multiple seeds matter far more for a *filtered* traversal than for an
    /// unfiltered one: a single seed can easily sit in a region from which the
    /// matching set is awkward to reach, and unlike the unfiltered case there
    /// is no guarantee that greedy descent toward the query passes anywhere
    /// near a match.
    fn entry_points(&self, seed_count: usize) -> Vec<u32> {
        let count = self.len();
        if count == 0 {
            return Vec::new();
        }
        let mut entries = vec![self.medoid];
        let wanted = seed_count.max(1).min(count);
        let mut rng = SplitMix64::new(self.seed ^ (count as u64).wrapping_mul(0x9e37_79b9));
        let mut attempts = 0;
        while entries.len() < wanted && attempts < wanted * 8 + 16 {
            attempts += 1;
            #[allow(clippy::cast_possible_truncation)]
            let candidate = rng.next_bounded(count) as u32;
            if !entries.contains(&candidate) {
                entries.push(candidate);
            }
        }
        entries
    }

    /// Unconstrained greedy search that returns **every** node whose distance
    /// was computed — the visited set Vamana's `RobustPrune` consumes.
    ///
    /// `exclude` is the node currently being (re)linked, which must never
    /// become its own neighbor.
    fn greedy_collect(
        &self,
        query: &[f32],
        query_norm: f32,
        beam_width: usize,
        entry_points: &[u32],
        exclude: u32,
    ) -> Vec<(u32, f32)> {
        let mut pool: Vec<PoolEntry> = Vec::with_capacity(beam_width + 1);
        let mut seen: HashSet<u32> = HashSet::new();
        let mut visited: Vec<(u32, f32)> = Vec::new();

        for &entry in entry_points {
            if entry as usize >= self.len() || !seen.insert(entry) {
                continue;
            }
            let distance = self.distance_to(query, query_norm, entry);
            if entry != exclude {
                visited.push((entry, distance));
            }
            pool_insert(
                &mut pool,
                PoolEntry {
                    id: entry,
                    distance,
                    matched: true,
                    expanded: false,
                },
            );
        }

        while let Some(position) = pool.iter().position(|entry| !entry.expanded) {
            pool[position].expanded = true;
            let current = pool[position].id;

            for &neighbor in &self.neighbors[current as usize] {
                if !seen.insert(neighbor) {
                    continue;
                }
                let distance = self.distance_to(query, query_norm, neighbor);
                if neighbor != exclude {
                    visited.push((neighbor, distance));
                }
                pool_insert(
                    &mut pool,
                    PoolEntry {
                        id: neighbor,
                        distance,
                        matched: true,
                        expanded: false,
                    },
                );
                pool_truncate(&mut pool, beam_width, 0);
            }
        }

        visited
    }

    /// An ordinary unconstrained top-`k` search — the first stage of
    /// `PostFilter`.
    ///
    /// Note that the *only* difference between this and
    /// [`FilteredGraph::acorn_search`] under an always-true predicate is that
    /// this one has no two-hop step to skip and no reserve to honor. The
    /// contrast the module's headline test measures is therefore genuinely
    /// about the filtering, not about two unrelated search implementations.
    pub(crate) fn greedy_search(
        &self,
        query: &[f32],
        query_norm: f32,
        top_k: usize,
        beam_width: usize,
        seed_count: usize,
        stats: &mut FilteredSearchStats,
    ) -> Vec<(u32, f32)> {
        if self.len() == 0 || top_k == 0 {
            return Vec::new();
        }
        let beam_width = beam_width.max(top_k).max(1);
        let mut pool: Vec<PoolEntry> = Vec::with_capacity(beam_width + 1);
        let mut seen: HashSet<u32> = HashSet::new();
        let mut results = BoundedResults::new(top_k);

        for entry in self.entry_points(seed_count) {
            if !seen.insert(entry) {
                continue;
            }
            let distance = self.distance_to(query, query_norm, entry);
            stats.distance_computations += 1;
            stats.candidates_considered += 1;
            results.offer(entry, distance);
            pool_insert(
                &mut pool,
                PoolEntry {
                    id: entry,
                    distance,
                    matched: true,
                    expanded: false,
                },
            );
        }

        while let Some(position) = pool.iter().position(|entry| !entry.expanded) {
            pool[position].expanded = true;
            let current = pool[position].id;
            stats.nodes_visited += 1;

            for &neighbor in &self.neighbors[current as usize] {
                if !seen.insert(neighbor) {
                    continue;
                }
                let distance = self.distance_to(query, query_norm, neighbor);
                stats.distance_computations += 1;
                stats.candidates_considered += 1;
                results.offer(neighbor, distance);
                pool_insert(
                    &mut pool,
                    PoolEntry {
                        id: neighbor,
                        distance,
                        matched: true,
                        expanded: false,
                    },
                );
                pool_truncate(&mut pool, beam_width, 0);
            }
        }

        results.into_sorted()
    }

    /// The ACORN traversal: predicate-aware greedy search over the proximity
    /// graph.
    ///
    /// Returns the `top_k` closest nodes that satisfy the oracle's predicate,
    /// best first. See the [module documentation](self) for the algorithm, the
    /// reason routing nodes are indispensable, and exactly what is and is not
    /// guaranteed about recall.
    ///
    /// # Complexity
    ///
    /// With degree `R`, beam `L`, expansion factor `gamma` and visit ceiling
    /// `V`, one search performs at most `V` expansions, each costing `O(R)`
    /// distance computations and `O(R * (1 + gamma))` predicate evaluations
    /// (memoized, so repeats are free) — hence `O(V * R * d)` arithmetic and
    /// `O(V * R * gamma)` predicate work in the worst case, and considerably
    /// less in practice because the adaptive gate skips the two-hop step
    /// wherever the predicate subgraph is dense enough to navigate unaided.
    pub(crate) fn acorn_search(
        &self,
        query: &[f32],
        query_norm: f32,
        params: AcornSearchParams,
        oracle: &mut dyn FilteredPredicateOracle,
        stats: &mut FilteredSearchStats,
    ) -> Vec<(u32, f32)> {
        if self.len() == 0 || params.top_k == 0 {
            stats.predicate_evaluations += oracle.evaluations();
            return Vec::new();
        }

        let beam_width = params.beam_width.max(params.top_k).max(1);
        let reserve = params.matching_reserve.min(beam_width);
        // Below this many matching neighbors, the predicate subgraph around a
        // node is too sparse to navigate and the two-hop expansion earns its
        // cost. Above it, the two-hop step is skipped entirely. Note this gate
        // is independent of `gamma` — see `min_predicate_neighbors` for why
        // deriving it from `gamma` (as the obvious `degree / gamma`) silently
        // disables the two-hop step exactly when you turn `gamma` up.
        let two_hop_threshold = params.min_predicate_neighbors.max(1);

        let mut pool: Vec<PoolEntry> = Vec::with_capacity(beam_width + 1);
        let mut seen: HashSet<u32> = HashSet::new();
        let mut results = BoundedResults::new(params.top_k);
        let mut visits = 0_usize;

        for entry in self.entry_points(params.seed_count) {
            if !seen.insert(entry) {
                continue;
            }
            let distance = self.distance_to(query, query_norm, entry);
            stats.distance_computations += 1;
            stats.candidates_considered += 1;
            let matched = oracle.matches(entry);
            if matched {
                stats.matched_candidates += 1;
                results.offer(entry, distance);
            }
            pool_insert(
                &mut pool,
                PoolEntry {
                    id: entry,
                    distance,
                    matched,
                    expanded: false,
                },
            );
        }

        while visits < params.max_visits {
            let Some(position) = pool.iter().position(|entry| !entry.expanded) else {
                break;
            };
            pool[position].expanded = true;
            let current = pool[position].id;
            visits += 1;
            stats.nodes_visited += 1;

            // ── One-hop: classify every neighbor. ────────────────────────
            //
            // Matching neighbors become results *and* pool members. Failing
            // neighbors become **routing nodes**: pool members that can be
            // expanded but never returned. Admitting them is what preserves the
            // full graph's navigability under an arbitrary predicate.
            let mut matching_neighbors = 0_usize;
            let mut failing: Vec<(u32, f32)> = Vec::new();

            for &neighbor in &self.neighbors[current as usize] {
                if !seen.insert(neighbor) {
                    // Already routed through; still counts toward the local
                    // density estimate that gates the two-hop step, so that a
                    // re-visited dense region is not needlessly re-expanded.
                    if oracle.matches(neighbor) {
                        matching_neighbors += 1;
                    }
                    continue;
                }
                let distance = self.distance_to(query, query_norm, neighbor);
                stats.distance_computations += 1;
                stats.candidates_considered += 1;

                let matched = oracle.matches(neighbor);
                if matched {
                    matching_neighbors += 1;
                    stats.matched_candidates += 1;
                    results.offer(neighbor, distance);
                } else {
                    failing.push((neighbor, distance));
                }

                pool_insert(
                    &mut pool,
                    PoolEntry {
                        id: neighbor,
                        distance,
                        matched,
                        expanded: false,
                    },
                );
            }

            // ── Two-hop (ACORN-gamma): route *through* failing neighbors. ─
            //
            // Skipped when the predicate subgraph around `current` is already
            // dense enough to navigate on its own — which is what keeps
            // `InFilter`'s cost close to an unfiltered search's at high
            // selectivity.
            if matching_neighbors < two_hop_threshold && !failing.is_empty() {
                self.expand_two_hop(
                    query,
                    query_norm,
                    &mut failing,
                    params.gamma.max(1),
                    &mut seen,
                    &mut pool,
                    &mut results,
                    oracle,
                    stats,
                );
            }

            pool_truncate(&mut pool, beam_width, reserve);
        }

        stats.predicate_evaluations += oracle.evaluations();
        results.into_sorted()
    }

    /// The ACORN two-hop step: route through the `gamma` predicate-failing
    /// neighbors *closest to the query* and harvest the predicate-satisfying
    /// members of each one's neighbor list.
    ///
    /// Routing through the closest failures rather than an arbitrary `gamma` of
    /// them keeps the expansion *directed* — it follows the query's gradient —
    /// instead of splashing outward breadth-first.
    ///
    /// Second-hop nodes that *fail* the predicate are deliberately **not**
    /// enqueued. Admitting them would make routing nodes themselves generate
    /// routing nodes, and the traversal would degenerate into an unbounded
    /// breadth-first search of the whole graph. One hop of slack is what ACORN
    /// buys; two would simply be a scan.
    #[allow(clippy::too_many_arguments)]
    fn expand_two_hop(
        &self,
        query: &[f32],
        query_norm: f32,
        failing: &mut [(u32, f32)],
        gamma: usize,
        seen: &mut HashSet<u32>,
        pool: &mut Vec<PoolEntry>,
        results: &mut BoundedResults,
        oracle: &mut dyn FilteredPredicateOracle,
        stats: &mut FilteredSearchStats,
    ) {
        failing.sort_by(|(left_id, left_distance), (right_id, right_distance)| {
            left_distance
                .partial_cmp(right_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left_id.cmp(right_id))
        });

        for &(router, _) in failing.iter().take(gamma) {
            stats.two_hop_expansions += 1;
            for &distant in &self.neighbors[router as usize] {
                // The predicate is checked *before* the distance, so a
                // second-hop node that fails costs one memoized predicate
                // lookup and nothing else. This is why a large `gamma` is far
                // cheaper than it looks: the expensive part (the
                // `d`-dimensional distance) is paid only for nodes that can
                // actually become results.
                if !oracle.matches(distant) || !seen.insert(distant) {
                    continue;
                }
                let distance = self.distance_to(query, query_norm, distant);
                stats.distance_computations += 1;
                stats.candidates_considered += 1;
                stats.matched_candidates += 1;
                results.offer(distant, distance);
                pool_insert(
                    pool,
                    PoolEntry {
                        id: distant,
                        distance,
                        matched: true,
                        expanded: false,
                    },
                );
            }
        }
    }
}
