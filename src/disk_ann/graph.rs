//! The Vamana proximity graph and its two core algorithms: `GreedySearch`
//! (beam search toward a query) and `RobustPrune` (α-occlusion neighbor
//! selection).
//!
//! [`VamanaGraph`] is a **single-layer** directed graph with a fixed **medoid**
//! entry point — distinct from the multi-layer, randomly-leveled structure
//! used by [`crate::hnsw_index`]. Every node keeps at most `R` out-edges,
//! chosen and refined by the algorithms in this module.

use std::cmp::Ordering;

use super::types::DiskAnnMetric;

// ── Deterministic hashing ─────────────────────────────────────────────────────

/// FNV-1a hash of a `u64` value, mirroring the idiom used in
/// [`crate::hnsw_index`] for deterministic pseudo-random choices.
#[inline]
fn fnv_u64(x: u64) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in x.to_le_bytes() {
        h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
    }
    h
}

/// Deterministic pseudo-random rank key for the ordered node-index pair
/// `(i, j)`.
///
/// Used to build the initial random `R`-regular graph: for node `i`, every
/// candidate neighbor `j` is ranked by this key and the lowest-keyed `R`
/// candidates are chosen as the initial out-edges. Hashing the pair (rather
/// than just `j`) gives every node an independent-looking permutation without
/// any `rand`/`rand_distr` dependency.
#[inline]
fn pair_rank_key(i: usize, j: usize) -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    let j_mixed = (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    #[allow(clippy::cast_possible_truncation)]
    let i_hash = fnv_u64(i as u64);
    fnv_u64(i_hash ^ j_mixed)
}

/// Build an initial random `R`-regular directed adjacency list over `n` nodes.
///
/// Each node `i` receives `min(max_degree, n - 1)` out-edges chosen by
/// [`pair_rank_key`]. This is the "random init" step of Vamana construction,
/// forming the starting graph over which the two `RobustPrune` build passes
/// operate.
fn init_random_edges(n: usize, max_degree: usize) -> Vec<Vec<usize>> {
    let degree = max_degree.min(n.saturating_sub(1));
    (0..n)
        .map(|i| {
            let mut ranked: Vec<(u64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (pair_rank_key(i, j), j))
                .collect();
            ranked.sort_unstable_by_key(|&(key, _)| key);
            ranked.into_iter().take(degree).map(|(_, j)| j).collect()
        })
        .collect()
}

// ── VamanaGraph ───────────────────────────────────────────────────────────────

/// A single-layer Vamana proximity graph with a fixed medoid entry point.
///
/// Every node holds at most `R` out-edges, where `R` is the configured
/// `max_degree`. Unlike a multi-layer HNSW graph, there is exactly one
/// layer, and the fixed entry point is the data set's **medoid** rather
/// than a randomly promoted node.
#[derive(Debug, Clone)]
pub struct VamanaGraph {
    /// `out_edges[i]` lists the out-neighbor node indices of node `i`.
    out_edges: Vec<Vec<usize>>,
    /// Index of the medoid node — the fixed entry point for every search.
    medoid: usize,
}

impl VamanaGraph {
    /// Construct a graph directly from a pre-built adjacency list and medoid
    /// index.
    #[must_use]
    pub fn new(out_edges: Vec<Vec<usize>>, medoid: usize) -> Self {
        Self { out_edges, medoid }
    }

    /// Build the initial random `R`-regular graph used as the starting point
    /// for Vamana construction.
    ///
    /// See [`init_random_edges`] for the deterministic FNV-1a-seeded
    /// selection rule.
    pub(crate) fn random_init(n: usize, max_degree: usize, medoid: usize) -> Self {
        Self {
            out_edges: init_random_edges(n, max_degree),
            medoid,
        }
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.out_edges.len()
    }

    /// `true` if the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.out_edges.is_empty()
    }

    /// The medoid node index — the fixed entry point for every search.
    #[must_use]
    pub fn medoid(&self) -> usize {
        self.medoid
    }

    /// The out-neighbors of node `idx`.
    #[must_use]
    pub fn neighbors_of(&self, idx: usize) -> &[usize] {
        &self.out_edges[idx]
    }

    /// Replace the out-neighbor list of node `idx`.
    pub(crate) fn set_neighbors(&mut self, idx: usize, neighbors: Vec<usize>) {
        self.out_edges[idx] = neighbors;
    }

    /// Append a back-edge `idx -> neighbor` if it is not already present
    /// (and is not a self-loop).
    pub(crate) fn add_back_edge(&mut self, idx: usize, neighbor: usize) {
        if idx != neighbor && !self.out_edges[idx].contains(&neighbor) {
            self.out_edges[idx].push(neighbor);
        }
    }

    /// `GreedySearch(start, query, l)`: beam search from `start` toward
    /// `query` with a working set bounded at size `l`.
    ///
    /// Repeatedly expands the closest unvisited candidate in the working set,
    /// folding its out-neighbors in, until every candidate in the working set
    /// has been visited (or the graph is exhausted). Returns:
    ///
    /// * the final working set as `(distance, node index)` pairs sorted by
    ///   ascending distance (closest first) — used as the "top results" for
    ///   queries;
    /// * the full visited set, in visitation order — used as the candidate
    ///   set `V` for `RobustPrune` during construction.
    pub(crate) fn greedy_search(
        &self,
        vectors: &[Vec<f32>],
        metric: DiskAnnMetric,
        start: usize,
        query: &[f32],
        l: usize,
    ) -> (Vec<(f32, usize)>, Vec<usize>) {
        let mut working_set: Vec<(f32, usize)> =
            vec![(metric.distance(query, &vectors[start]), start)];
        let mut is_visited = vec![false; vectors.len()];
        let mut visit_order: Vec<usize> = Vec::new();

        loop {
            // Locate the closest unvisited candidate currently in the working set.
            let mut closest_pos: Option<usize> = None;
            let mut closest_dist = f32::INFINITY;
            for (pos, &(dist, idx)) in working_set.iter().enumerate() {
                if !is_visited[idx] && dist < closest_dist {
                    closest_dist = dist;
                    closest_pos = Some(pos);
                }
            }

            let Some(pos) = closest_pos else {
                break;
            };
            let (_, current) = working_set[pos];
            is_visited[current] = true;
            visit_order.push(current);

            for &neighbor in self.neighbors_of(current) {
                if is_visited[neighbor] || working_set.iter().any(|&(_, idx)| idx == neighbor) {
                    continue;
                }
                let dist = metric.distance(query, &vectors[neighbor]);
                working_set.push((dist, neighbor));
            }

            if working_set.len() > l {
                working_set.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                working_set.truncate(l);
            }
        }

        working_set.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        (working_set, visit_order)
    }
}

// ── RobustPrune ───────────────────────────────────────────────────────────────

/// `RobustPrune(p, candidates, alpha, r)`: greedy α-occlusion neighbor
/// selection.
///
/// Sorts `candidates` by distance to `p` (implicitly, via repeated
/// closest-first selection); greedily keeps the closest remaining point
/// `p*`, then drops every other candidate `v'` for which
/// `alpha * distance(p*, v') <= distance(p, v')` — meaning `p*` already
/// "occludes" `v'` well enough that a direct edge to `v'` would be
/// redundant. Repeats until `r` neighbors have been chosen or `candidates`
/// is exhausted.
///
/// `p` itself is filtered out of `candidates` defensively, and duplicate
/// entries are removed before pruning begins.
pub(crate) fn robust_prune(
    vectors: &[Vec<f32>],
    metric: DiskAnnMetric,
    p: usize,
    candidates: Vec<usize>,
    alpha: f32,
    r: usize,
) -> Vec<usize> {
    let mut pool: Vec<usize> = candidates.into_iter().filter(|&v| v != p).collect();
    pool.sort_unstable();
    pool.dedup();

    let mut selected: Vec<usize> = Vec::with_capacity(r.min(pool.len()));

    while !pool.is_empty() && selected.len() < r {
        // Find the pool entry closest to `p`.
        let mut best_pos = 0usize;
        let mut best_dist = f32::INFINITY;
        for (pos, &candidate) in pool.iter().enumerate() {
            let dist = metric.distance(&vectors[p], &vectors[candidate]);
            if dist < best_dist {
                best_dist = dist;
                best_pos = pos;
            }
        }
        let chosen = pool.remove(best_pos);
        selected.push(chosen);

        if selected.len() >= r {
            break;
        }

        // Alpha-occlusion rule: drop remaining candidates already covered by
        // `chosen` within a factor of `alpha`.
        pool.retain(|&candidate| {
            let d_new = metric.distance(&vectors[chosen], &vectors[candidate]);
            let d_old = metric.distance(&vectors[p], &vectors[candidate]);
            alpha * d_new > d_old
        });
    }

    selected
}
