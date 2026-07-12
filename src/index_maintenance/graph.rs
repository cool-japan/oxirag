//! The mutable proximity graph and its primitives: `GreedySearch` (beam search
//! that traverses *through* tombstones), `RobustPrune` (α-occlusion neighbour
//! selection, with a pinned-edge variant used by the connectivity repair pass),
//! reachability analysis, and the deterministic random `R`-regular
//! initialisation used by the two-pass Vamana build.
//!
//! Everything here operates on a flat `&[MaintenanceNode]` slot table. Slots are
//! **dense**: `insert` appends, `delete` only flips a tombstone flag, and
//! [`consolidate`](crate::index_maintenance::MaintainableIndex::consolidate) is
//! the only operation that ever removes a slot — which it does by compacting the
//! whole table and remapping every edge in one pass. There are therefore never
//! any holes, and no edge can ever dangle.

use super::types::MaintenanceMetric;

// ── MaintenanceNode ───────────────────────────────────────────────────────────

/// One slot in the graph's node table.
///
/// A tombstoned node keeps its vector **and its out-edges**. That is the whole
/// point of the tombstone: `GreedySearch` still expands it, so every path that
/// used to run through it still exists, and the graph loses no navigability
/// between the `delete` and the next `consolidate`. Only the *result filter*
/// knows about the tombstone.
#[derive(Debug, Clone)]
pub(crate) struct MaintenanceNode {
    /// Stable external identifier supplied by the caller.
    pub(crate) external_id: String,
    /// The embedding stored in this slot.
    pub(crate) vector: Vec<f32>,
    /// Out-neighbour slot indices; length is invariantly `<= R`.
    pub(crate) out_edges: Vec<usize>,
    /// `true` once the node is deleted but before it is physically collected.
    pub(crate) tombstoned: bool,
}

impl MaintenanceNode {
    /// Create a fresh live node with no out-edges.
    pub(crate) fn new(external_id: String, vector: Vec<f32>) -> Self {
        Self {
            external_id,
            vector,
            out_edges: Vec::new(),
            tombstoned: false,
        }
    }
}

// ── Deterministic pseudo-randomness ───────────────────────────────────────────

/// `SplitMix64` finalising mix — a deterministic, well-distributed 64-bit hash.
///
/// Used instead of `rand`/`rand_distr` (a hard project constraint) to seed the
/// initial `R`-regular graph. Every "random" choice this module makes routes
/// through here, so a build is bit-for-bit reproducible.
#[inline]
pub(crate) fn split_mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic rank key for the ordered slot pair `(i, j)`.
///
/// Hashing the *pair* (rather than just `j`) gives every node an independent
/// looking permutation of its candidate neighbours, which is what the random
/// `R`-regular initialisation needs.
#[inline]
fn pair_rank_key(i: usize, j: usize) -> u64 {
    let i_mixed = split_mix64(i as u64);
    split_mix64(i_mixed ^ (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Build an initial random `R`-regular directed adjacency list over `n` slots.
///
/// Each slot receives `min(max_degree, n - 1)` out-edges chosen by
/// [`pair_rank_key`]. This is the "random init" step of a Vamana build: the two
/// `RobustPrune` passes then refine it into a navigable proximity graph.
pub(crate) fn random_regular_edges(n: usize, max_degree: usize) -> Vec<Vec<usize>> {
    let degree = max_degree.min(n.saturating_sub(1));
    (0..n)
        .map(|i| {
            let mut ranked: Vec<(u64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (pair_rank_key(i, j), j))
                .collect();
            ranked.sort_unstable();
            ranked.into_iter().take(degree).map(|(_, j)| j).collect()
        })
        .collect()
}

// ── Medoid ────────────────────────────────────────────────────────────────────

/// Corpora at or below this size get the exact `O(n^2)` medoid; larger ones use
/// the (deterministic) centroid-nearest approximation so entry-point selection
/// stays sub-quadratic.
const MEDOID_EXACT_THRESHOLD: usize = 512;

/// The medoid of the **live** nodes: the live slot minimising the summed
/// distance to every other live slot.
///
/// Returns `None` when no live node exists. Tombstoned nodes are excluded —
/// entering the graph at a dying node is legal (searches route straight through
/// it) but it makes for a poor, soon-to-be-collected entry point.
pub(crate) fn live_medoid(nodes: &[MaintenanceNode], metric: MaintenanceMetric) -> Option<usize> {
    let live: Vec<usize> = (0..nodes.len()).filter(|&i| !nodes[i].tombstoned).collect();
    if live.is_empty() {
        return None;
    }

    if live.len() <= MEDOID_EXACT_THRESHOLD {
        let mut best = live[0];
        let mut best_sum = f32::INFINITY;
        for &i in &live {
            let mut sum = 0.0f32;
            for &j in &live {
                if i != j {
                    sum += metric.distance(&nodes[i].vector, &nodes[j].vector);
                }
            }
            if sum < best_sum {
                best_sum = sum;
                best = i;
            }
        }
        return Some(best);
    }

    // Centroid-nearest approximation: an explicit, honest approximation of the
    // medoid rather than a silently-degraded exact computation.
    let dim = nodes[live[0]].vector.len();
    let mut centroid = vec![0.0f32; dim];
    for &i in &live {
        for (c, x) in centroid.iter_mut().zip(nodes[i].vector.iter()) {
            *c += x;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = live.len() as f32;
    for c in &mut centroid {
        *c /= count;
    }

    let mut best = live[0];
    let mut best_dist = f32::INFINITY;
    for &i in &live {
        let dist = metric.distance(&centroid, &nodes[i].vector);
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }
    Some(best)
}

// ── GreedySearch ──────────────────────────────────────────────────────────────

/// Truncate the beam to the dual budget: the `l` closest **live** candidates and
/// the `l` closest **tombstoned** candidates.
///
/// A plain size-`l` beam would let tombstones crowd live candidates out of the
/// working set, so a heavily-tombstoned index would return fewer than `k` hits
/// even though the graph itself is undamaged — the classic reason implementations
/// are told to "raise `L` in proportion to the deleted fraction". Giving
/// tombstones their own budget does that automatically and keeps the *live*
/// exploration budget of a tombstoned index **exactly equal** to that of a
/// freshly-consolidated one, which is what makes the recall comparison between
/// the two regimes an apples-to-apples measurement.
///
/// The cost is bounded: the beam never exceeds `2 * l` entries, and once
/// `consolidate` has run there are no tombstones left, so the beam collapses back
/// to `l`.
fn truncate_beam(beam: &mut Vec<(f32, usize)>, nodes: &[MaintenanceNode], l: usize) {
    if beam.len() <= l {
        return;
    }
    beam.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut live_kept = 0usize;
    let mut tomb_kept = 0usize;
    beam.retain(|&(_, idx)| {
        if nodes[idx].tombstoned {
            tomb_kept += 1;
            tomb_kept <= l
        } else {
            live_kept += 1;
            live_kept <= l
        }
    });
}

/// `GreedySearch(query, entry, l)`: beam search from `entry` toward `query`.
///
/// Repeatedly expands the closest not-yet-expanded candidate in the working set,
/// folding its out-neighbours in, until every candidate in the working set has
/// been expanded. **Tombstoned nodes are expanded like any other node** — that is
/// the connectivity guarantee that makes a soft delete safe.
///
/// Returns:
///
/// * the final working set as `(distance, slot)` pairs sorted closest-first
///   (tombstones included — callers filter them out of *results*, and
///   [`super::index::MaintainableIndex::search`] does exactly that);
/// * the expansion order, used as the `RobustPrune` candidate set `V` during
///   insert and build.
///
/// A node that is evicted from the beam by [`truncate_beam`] is never re-admitted
/// (it is already marked as seen), which is what bounds the search and makes it
/// terminate: every iteration marks one more slot as expanded, so at most `n`
/// iterations run.
pub(crate) fn greedy_search(
    nodes: &[MaintenanceNode],
    metric: MaintenanceMetric,
    entry: usize,
    query: &[f32],
    l: usize,
) -> (Vec<(f32, usize)>, Vec<usize>) {
    let n = nodes.len();
    if n == 0 || entry >= n {
        return (Vec::new(), Vec::new());
    }
    let l = l.max(1);

    let mut beam: Vec<(f32, usize)> = vec![(metric.distance(query, &nodes[entry].vector), entry)];
    let mut seen = vec![false; n];
    let mut expanded = vec![false; n];
    seen[entry] = true;
    let mut expansion_order: Vec<usize> = Vec::new();

    loop {
        let mut best_pos: Option<usize> = None;
        let mut best_dist = f32::INFINITY;
        for (pos, &(dist, idx)) in beam.iter().enumerate() {
            if !expanded[idx] && dist < best_dist {
                best_dist = dist;
                best_pos = Some(pos);
            }
        }
        let Some(pos) = best_pos else {
            break;
        };

        let current = beam[pos].1;
        expanded[current] = true;
        expansion_order.push(current);

        for &neighbor in &nodes[current].out_edges {
            if seen[neighbor] {
                continue;
            }
            seen[neighbor] = true;
            beam.push((metric.distance(query, &nodes[neighbor].vector), neighbor));
        }

        truncate_beam(&mut beam, nodes, l);
    }

    beam.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    (beam, expansion_order)
}

// ── RobustPrune ───────────────────────────────────────────────────────────────

/// `RobustPrune(p, candidates, alpha, r)`: greedy α-occlusion neighbour selection.
///
/// Repeatedly takes the candidate `p*` closest to `p`, then discards every
/// remaining candidate `v'` that `p*` already "occludes":
///
/// ```text
/// alpha * distance(p*, v') <= distance(p, v')   =>   drop v'
/// ```
///
/// The intuition is that a greedy search standing on `p` and heading for `v'`
/// would hop to `p*` anyway, so a direct `p -> v'` edge buys nothing; `alpha > 1`
/// loosens the rule and lets a few long-range edges survive, which shortens the
/// graph's diameter.
///
/// `p` itself is filtered out defensively and duplicates are removed, so the
/// result is always a self-loop-free set of at most `r` distinct slots.
pub(crate) fn robust_prune(
    nodes: &[MaintenanceNode],
    metric: MaintenanceMetric,
    p: usize,
    candidates: Vec<usize>,
    alpha: f32,
    r: usize,
) -> Vec<usize> {
    robust_prune_with_pinned(nodes, metric, p, candidates, &[], alpha, r)
}

/// `RobustPrune` with a set of **pinned** neighbours that the prune may never
/// drop.
///
/// The pinned slots are placed in the result unconditionally and the α-occlusion
/// selection then fills the remaining `r - |pinned|` slots from `candidates`.
/// This is what lets the connectivity repair pass add an edge `p -> u` and
/// simultaneously bring `p` back under the degree cap *without any risk of the
/// prune immediately discarding the very edge that was just added* — which is
/// exactly the failure mode a naive "add then prune" would hit whenever `u` is
/// occluded by a closer neighbour of `p`.
///
/// If `pinned.len() >= r` the result is the first `r` pinned slots: the caller's
/// hard requirement wins over the degree budget's soft preference, and the
/// degree cap still holds.
pub(crate) fn robust_prune_with_pinned(
    nodes: &[MaintenanceNode],
    metric: MaintenanceMetric,
    p: usize,
    candidates: Vec<usize>,
    pinned: &[usize],
    alpha: f32,
    r: usize,
) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::with_capacity(r);
    for &pin in pinned {
        if pin != p && !selected.contains(&pin) && selected.len() < r {
            selected.push(pin);
        }
    }
    if selected.len() >= r {
        return selected;
    }

    let mut pool: Vec<usize> = candidates
        .into_iter()
        .filter(|&v| v != p && !selected.contains(&v))
        .collect();
    pool.sort_unstable();
    pool.dedup();

    while !pool.is_empty() && selected.len() < r {
        // Closest remaining candidate to `p` (deterministic tie-break by slot).
        let mut best_pos = 0usize;
        let mut best_dist = f32::INFINITY;
        for (pos, &candidate) in pool.iter().enumerate() {
            let dist = metric.distance(&nodes[p].vector, &nodes[candidate].vector);
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

        pool.retain(|&candidate| {
            let d_new = metric.distance(&nodes[chosen].vector, &nodes[candidate].vector);
            let d_old = metric.distance(&nodes[p].vector, &nodes[candidate].vector);
            alpha * d_new > d_old
        });
    }

    selected
}

// ── Reachability ──────────────────────────────────────────────────────────────

/// The set of slots reachable from `entry`, following out-edges.
///
/// `through_tombstones` selects the traversal model:
///
/// * `true` — the model `search` actually uses: tombstoned nodes are walked
///   *through*, so the reachable set is the set of slots any query could ever
///   touch.
/// * `false` — live-only traversal: the model that will hold *after* the next
///   consolidation physically collects the tombstones. A live node reachable only
///   via a tombstoned node is fine today and would be orphaned tomorrow — which is
///   precisely what the consolidation bridge pass exists to prevent.
pub(crate) fn reachable_from(
    nodes: &[MaintenanceNode],
    entry: Option<usize>,
    through_tombstones: bool,
) -> Vec<bool> {
    let mut reached = vec![false; nodes.len()];
    let Some(entry) = entry else {
        return reached;
    };
    if entry >= nodes.len() || (!through_tombstones && nodes[entry].tombstoned) {
        return reached;
    }

    reached[entry] = true;
    let mut stack = vec![entry];
    while let Some(current) = stack.pop() {
        for &neighbor in &nodes[current].out_edges {
            if reached[neighbor] {
                continue;
            }
            if !through_tombstones && nodes[neighbor].tombstoned {
                continue;
            }
            reached[neighbor] = true;
            stack.push(neighbor);
        }
    }
    reached
}

/// Live slots that `reached` does not cover, in ascending slot order.
pub(crate) fn unreached_live(nodes: &[MaintenanceNode], reached: &[bool]) -> Vec<usize> {
    (0..nodes.len())
        .filter(|&i| !nodes[i].tombstoned && !reached[i])
        .collect()
}
