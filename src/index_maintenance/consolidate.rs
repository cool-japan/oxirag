//! Consolidation (tombstone collection + slot compaction) and the connectivity
//! repair pass.
//!
//! # The bridge rule
//!
//! A tombstoned node `t` is a node that queries must not *report* but that the
//! graph still *routes through*. Collecting it therefore cannot simply drop the
//! edges that point at it: every in-neighbour `p` of `t` used `t` as its gateway
//! to `t`'s neighbourhood, and deleting `p -> t` without a replacement is exactly
//! the naive hard delete this module exists to fix.
//!
//! Instead, for every live `p` that points at some tombstoned `t`, the edge
//! `p -> t` is replaced by edges from `p` to `t`'s **live frontier** — the live
//! nodes reachable from `t` through a path consisting only of tombstoned nodes —
//! and the enlarged candidate set is then `RobustPrune`d back down to `R`. The
//! transitive closure over tombstones matters: with `p -> t1 -> t2 -> x` and both
//! `t1` and `t2` tombstoned, `t1`'s *immediate* live out-neighbours are empty, and
//! a non-transitive bridge would silently sever `p` from `x`.
//!
//! # Why the compaction cannot dangle
//!
//! After the bridge pass, **no live node holds an edge to a tombstoned node**: a
//! live node either had no tombstoned out-neighbour (untouched) or was re-pruned
//! from a candidate set that contains only live nodes. Tombstoned slots are then
//! dropped wholesale and the surviving slots are renumbered in ascending order, so
//! every remaining edge is remapped through a total old-slot -> new-slot function.
//! No edge can point at a removed slot, by construction.
//!
//! # The orphan-freedom argument
//!
//! `RobustPrune` is a *heuristic*: it preserves navigability in practice but
//! guarantees nothing. Bridging `p` can therefore still discard `p`'s last edge to
//! some third node `w`, and after enough churn a live node can end up with no
//! in-edge from the reachable component at all — an **orphan**, invisible to every
//! query no matter how large the beam. The repair pass in
//! `MaintainableIndex::repair_connectivity` closes that hole *provably* rather
//! than hopefully. Let `Reached` be the BFS closure of the entry point under the
//! traversal rules `search` uses (i.e. through tombstones), and let `u` be a live
//! node outside it. `R >= 2` is enforced by [`MaintenanceConfig::validate`](super::types::MaintenanceConfig::validate).
//!
//! * **Case A — some `p` in `Reached` has out-degree `< R`.** Append `p -> u`. `u`
//!   joins `Reached`, no edge is lost, and the degree cap still holds.
//! * **Case B — every `p` in `Reached` is saturated at exactly `R`.** Every
//!   out-edge of a reached node lands inside `Reached` (that is what "reached"
//!   means), so `Reached` contains exactly `R * |Reached|` internal edges and the
//!   mean in-degree inside `Reached` is `R >= 2`. By pigeonhole some `w` in
//!   `Reached` has in-degree `>= 2`. Pick such a `w` together with one of its
//!   in-neighbours `p`, drop `p -> w`, and append `p -> u`. `w` keeps at least one
//!   in-edge from a still-reached node, so `w` — and everything downstream of it —
//!   stays reached; `u` becomes reached; and `p`'s degree is unchanged.
//!
//! Both cases strictly shrink the orphan set and neither can create a new orphan,
//! so the pass terminates in at most `|live|` iterations and ends with **every live
//! node reachable from the entry point at out-degree `<= R`**. Case B is why
//! `max_degree == 1` is rejected: an out-degree-one digraph can have mean in-degree
//! `1`, the pigeonhole fails, and no safe victim edge need exist.

use std::collections::HashMap;

use super::graph::{
    MaintenanceNode, greedy_search, live_medoid, reachable_from, robust_prune, unreached_live,
};
use super::index::MaintainableIndex;
use super::types::ConsolidationReport;

/// The live nodes reachable from `start` through a path of tombstoned nodes only.
///
/// `start` is itself tombstoned. The BFS never re-enters a node, so cycles among
/// tombstones (which are perfectly possible — nothing stops two dying nodes from
/// pointing at each other) terminate cleanly.
fn live_frontier(nodes: &[MaintenanceNode], start: usize) -> Vec<usize> {
    let mut visited = vec![false; nodes.len()];
    visited[start] = true;
    let mut stack = vec![start];
    let mut frontier: Vec<usize> = Vec::new();

    while let Some(current) = stack.pop() {
        for &neighbor in &nodes[current].out_edges {
            if visited[neighbor] {
                continue;
            }
            visited[neighbor] = true;
            if nodes[neighbor].tombstoned {
                stack.push(neighbor);
            } else {
                frontier.push(neighbor);
            }
        }
    }

    frontier.sort_unstable();
    frontier
}

impl MaintainableIndex {
    /// Collect every tombstone: bridge its in-edges, compact the slot table, and
    /// restore the graph invariants.
    ///
    /// The pass is a no-op — literally, not merely cheap — when there is no
    /// tombstone: no edge is touched, no slot moves, and the returned report
    /// satisfies [`ConsolidationReport::is_noop`].
    ///
    /// Otherwise it runs, in order:
    ///
    /// 1. **Bridge.** Every live `p` with a tombstoned out-neighbour `t` has
    ///    `p -> t` replaced by edges to `t`'s live frontier, then `RobustPrune`d
    ///    back to `R`. See the module documentation for why the frontier is a
    ///    transitive closure and not just `t`'s immediate live neighbours.
    /// 2. **Compact.** Tombstoned slots are dropped and the survivors are
    ///    renumbered densely; every edge and the entry point are remapped, and
    ///    `id_map` is rebuilt — so **external ids stay valid even though every
    ///    internal slot index has moved**. The entry point is re-selected (to the
    ///    medoid of the live set) if the old one was tombstoned.
    /// 3. **Re-seed.** A live node whose out-edges were entirely tombstoned can
    ///    come out of step 1 with an empty adjacency list; it is given fresh
    ///    out-edges by a `GreedySearch` + `RobustPrune` from the entry point.
    /// 4. **Refine.** The bridged nodes get a proper two-pass Vamana refinement —
    ///    the bridge is only a *local* repair, and a graph rebuilt from purely local
    ///    candidate pools is connected, degree-capped, and measurably less navigable.
    ///    See `refine_bridged`; this step is what makes
    ///    consolidation recall-neutral rather than merely recall-survivable.
    /// 5. **Repair connectivity**, so that no live node is left unreachable.
    ///
    /// Every counter in the returned [`ConsolidationReport`] is deterministic:
    /// consolidating the same state twice yields the same report.
    pub fn consolidate(&mut self) -> ConsolidationReport {
        if self.tombstone_count == 0 {
            return ConsolidationReport {
                live_nodes_after: self.live_len(),
                ..ConsolidationReport::default()
            };
        }

        let mut report = ConsolidationReport {
            nodes_removed: self.tombstone_count,
            ..ConsolidationReport::default()
        };

        let bridged = self.bridge_tombstones(&mut report);
        let bridged = self.compact_slots(&mut report, &bridged);
        self.reseed_isolated(&mut report);
        self.refine_bridged(&bridged, &mut report);
        report.orphans_repaired = self.repair_connectivity();
        report.live_nodes_after = self.live_len();
        report
    }

    /// Step 1: replace every `live -> tombstoned` edge with edges to the
    /// tombstone's live frontier, re-pruning back to `R`.
    ///
    /// Returns the slots whose adjacency list was rebuilt, which `refine_bridged`
    /// then re-optimises.
    fn bridge_tombstones(&mut self, report: &mut ConsolidationReport) -> Vec<usize> {
        let metric = self.config.metric;
        let alpha = self.config.alpha;
        let max_degree = self.config.max_degree;
        let mut frontier_cache: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut bridged: Vec<usize> = Vec::new();

        for p in 0..self.nodes.len() {
            if self.nodes[p].tombstoned {
                continue;
            }
            let dead_out: Vec<usize> = self.nodes[p]
                .out_edges
                .iter()
                .copied()
                .filter(|&n| self.nodes[n].tombstoned)
                .collect();
            if dead_out.is_empty() {
                continue;
            }
            report.edges_rewired += dead_out.len();

            let mut candidates: Vec<usize> = self.nodes[p]
                .out_edges
                .iter()
                .copied()
                .filter(|&n| !self.nodes[n].tombstoned)
                .collect();
            for dead in dead_out {
                let frontier = frontier_cache
                    .entry(dead)
                    .or_insert_with(|| live_frontier(&self.nodes, dead));
                candidates.extend(frontier.iter().copied().filter(|&w| w != p));
            }
            candidates.sort_unstable();
            candidates.dedup();

            report.prunes_performed += 1;
            self.nodes[p].out_edges =
                robust_prune(&self.nodes, metric, p, candidates, alpha, max_degree);
            bridged.push(p);
        }

        bridged
    }

    /// Step 4: re-optimise the adjacency list of every node the bridge rebuilt.
    ///
    /// # Why the bridge alone is not enough
    ///
    /// The bridge is a *local* repair: it can only choose `p`'s new out-edges from
    /// `p`'s own surviving neighbours plus the live frontiers of `p`'s dead ones.
    /// That candidate pool is far poorer than the one the node was originally built
    /// from — a full `GreedySearch` visited set, which sweeps the whole graph from
    /// the entry point and typically holds hundreds of well-spread candidates. After
    /// a large delete batch, a *third* of the graph can have its adjacency list
    /// re-derived from that impoverished pool, and the result is a graph that is
    /// perfectly connected, perfectly degree-capped, and measurably **less
    /// navigable**: greedy search stalls in neighbourhoods whose edges no longer
    /// point anywhere useful. Measured on the churn benchmark at `R = 6`, bridging
    /// alone costs about six points of recall@10 against an equivalent freshly-built
    /// graph — with zero orphans and a full out-degree of `R` the whole time, which
    /// is exactly why the connectivity invariants alone cannot catch it.
    ///
    /// So consolidation finishes with a Vamana refinement restricted to the bridged
    /// nodes: `GreedySearch` from the entry point for a proper candidate pool,
    /// `RobustPrune` to re-select α-diverse out-edges, and the backward edges
    /// installed as usual. The bridge still has to run first — it is what keeps the
    /// graph navigable *for these very searches* — but it is a scaffold, not the
    /// final structure.
    ///
    /// The pass runs **twice**, at `alpha = 1.0` and then at the configured `alpha`,
    /// exactly as the two-pass build does: the first pass settles the backward-edge
    /// churn that re-pruning the refined nodes' neighbours inevitably causes, and the
    /// second re-admits the long-range edges that keep the graph's diameter small. A
    /// single pass measurably under-delivers (recall@10 `0.930` versus `0.963` on the
    /// churn benchmark; the freshly-built reference graph scores `0.954`, so the
    /// two-pass refinement makes consolidation recall-**neutral** rather than merely
    /// recall-survivable).
    ///
    /// The cost is bounded by the size of the delete batch, not by the index: only
    /// nodes that actually pointed at a tombstone are refined. A compaction is a
    /// batch operation and is allowed to pay this; it is still far cheaper than the
    /// full rebuild that the naive alternative eventually forces.
    fn refine_bridged(&mut self, bridged: &[usize], report: &mut ConsolidationReport) {
        let Some(entry) = self.entry else {
            return;
        };
        if self.nodes.len() < 2 {
            return;
        }
        let metric = self.config.metric;
        let alpha = self.config.alpha;
        let max_degree = self.config.max_degree;
        let l = self.config.search_list_size;

        for pass_alpha in [1.0f32, alpha] {
            for &p in bridged {
                let query = self.nodes[p].vector.clone();
                let (_, expanded) = greedy_search(&self.nodes, metric, entry, &query, l);
                let candidates: Vec<usize> = expanded.into_iter().filter(|&v| v != p).collect();
                if candidates.is_empty() {
                    continue;
                }
                let selected =
                    robust_prune(&self.nodes, metric, p, candidates, pass_alpha, max_degree);
                report.prunes_performed += 1;
                self.nodes[p].out_edges.clone_from(&selected);
                self.install_backward_edges(p, &selected, pass_alpha);
            }
        }
    }

    /// Step 2: drop tombstoned slots, renumber the survivors densely, and remap
    /// every edge, the entry point, `id_map` and the caller's `bridged` slot list.
    fn compact_slots(&mut self, report: &mut ConsolidationReport, bridged: &[usize]) -> Vec<usize> {
        let old_len = self.nodes.len();
        let mut remap: Vec<Option<usize>> = vec![None; old_len];
        let mut next_slot = 0usize;
        for (old, node) in self.nodes.iter().enumerate() {
            if !node.tombstoned {
                remap[old] = Some(next_slot);
                next_slot += 1;
            }
        }

        let entry_died = self.entry.is_some_and(|slot| self.nodes[slot].tombstoned);

        let mut compacted: Vec<MaintenanceNode> = Vec::with_capacity(next_slot);
        for old in 0..old_len {
            if remap[old].is_none() {
                continue;
            }
            let mut node = self.nodes[old].clone();
            // Every surviving edge points at a live slot: `bridge_tombstones`
            // removed every `live -> tombstoned` edge, so `remap` is total here.
            node.out_edges = node
                .out_edges
                .iter()
                .filter_map(|&target| remap[target])
                .collect();
            compacted.push(node);
        }

        self.nodes = compacted;
        self.tombstone_count = 0;
        self.id_map = self
            .nodes
            .iter()
            .enumerate()
            .map(|(slot, node)| (node.external_id.clone(), slot))
            .collect();

        self.entry = if self.nodes.is_empty() {
            None
        } else if entry_died {
            report.entry_point_reselected = true;
            live_medoid(&self.nodes, self.config.metric)
        } else {
            self.entry.and_then(|slot| remap[slot])
        };

        // Every bridged node is live by construction, so `remap` is total on it.
        bridged.iter().filter_map(|&slot| remap[slot]).collect()
    }

    /// Step 3: give fresh out-edges to any live node the bridge pass left with an
    /// empty adjacency list (every one of its neighbours was tombstoned and had an
    /// empty live frontier).
    fn reseed_isolated(&mut self, report: &mut ConsolidationReport) {
        let Some(entry) = self.entry else {
            return;
        };
        if self.nodes.len() < 2 {
            return;
        }
        let metric = self.config.metric;
        let alpha = self.config.alpha;
        let max_degree = self.config.max_degree;

        for p in 0..self.nodes.len() {
            if !self.nodes[p].out_edges.is_empty() {
                continue;
            }
            let query = self.nodes[p].vector.clone();
            let (_, expanded) = greedy_search(
                &self.nodes,
                metric,
                entry,
                &query,
                self.config.search_list_size,
            );
            let candidates: Vec<usize> = expanded.into_iter().filter(|&v| v != p).collect();
            if candidates.is_empty() {
                continue;
            }
            let selected = robust_prune(&self.nodes, metric, p, candidates, alpha, max_degree);
            report.prunes_performed += 1;
            self.nodes[p].out_edges = selected;
        }
    }

    /// Re-attach every live node that the entry point's BFS closure cannot reach,
    /// without ever exceeding the out-degree cap `R`.
    ///
    /// Returns the number of orphans repaired. See the module documentation for
    /// the termination and correctness argument (Case A / Case B); the two cases
    /// below implement it directly.
    ///
    /// The reachability model is the one `search` actually uses — BFS *through*
    /// tombstones — so a node this pass leaves reachable is a node a query can
    /// really return.
    pub(crate) fn repair_connectivity(&mut self) -> usize {
        let Some(entry) = self.entry else {
            return 0;
        };
        if self.nodes.is_empty() {
            return 0;
        }
        let metric = self.config.metric;
        let max_degree = self.config.max_degree;
        let mut repaired = 0usize;

        loop {
            let reached = reachable_from(&self.nodes, Some(entry), true);
            let unreached = unreached_live(&self.nodes, &reached);
            let Some(&orphan) = unreached.first() else {
                break;
            };
            let reached_slots: Vec<usize> = (0..self.nodes.len()).filter(|&i| reached[i]).collect();

            // ── Case A: a reached host with spare degree. Prefer a live host
            // (it survives the next consolidation), then the nearest one.
            let mut host: Option<(bool, f32, usize)> = None;
            for &p in &reached_slots {
                if self.nodes[p].out_edges.len() >= max_degree {
                    continue;
                }
                let dist = metric.distance(&self.nodes[p].vector, &self.nodes[orphan].vector);
                let key = (self.nodes[p].tombstoned, dist, p);
                if host.is_none_or(|best| Self::host_key_lt(key, best)) {
                    host = Some(key);
                }
            }
            if let Some((_, _, p)) = host {
                self.nodes[p].out_edges.push(orphan);
                repaired += 1;
                continue;
            }

            // ── Case B: every reached slot is saturated at exactly `R`. The mean
            // in-degree inside the reached component is therefore `R >= 2`, so some
            // reached `w` has in-degree `>= 2` and one of its in-edges can be
            // re-pointed at the orphan without ever un-reaching `w`.
            let mut in_degree = vec![0usize; self.nodes.len()];
            for &p in &reached_slots {
                for &target in &self.nodes[p].out_edges {
                    in_degree[target] += 1;
                }
            }

            let mut swap: Option<(bool, f32, usize, usize)> = None;
            for &p in &reached_slots {
                // Sacrifice `p`'s *least useful* out-edge: the farthest one that
                // still leaves its target with another in-edge from the component.
                let mut victim: Option<(f32, usize)> = None;
                for &target in &self.nodes[p].out_edges {
                    if in_degree[target] < 2 {
                        continue;
                    }
                    let dist = metric.distance(&self.nodes[p].vector, &self.nodes[target].vector);
                    let is_better = victim.is_none_or(|(best_dist, best_target): (f32, usize)| {
                        dist.total_cmp(&best_dist)
                            .then_with(|| best_target.cmp(&target))
                            .is_gt()
                    });
                    if is_better {
                        victim = Some((dist, target));
                    }
                }
                let Some((_, target)) = victim else {
                    continue;
                };
                let dist = metric.distance(&self.nodes[p].vector, &self.nodes[orphan].vector);
                let key = (self.nodes[p].tombstoned, dist, p, target);
                let is_better = swap.is_none_or(|best| {
                    Self::host_key_lt((key.0, key.1, key.2), (best.0, best.1, best.2))
                });
                if is_better {
                    swap = Some(key);
                }
            }

            let Some((_, _, p, victim)) = swap else {
                // Unreachable given `max_degree >= 2` (see the module docs): a
                // saturated component always exposes a safe victim edge. Bail out
                // rather than spin, so a future invariant change can never hang.
                break;
            };
            self.nodes[p].out_edges.retain(|&target| target != victim);
            self.nodes[p].out_edges.push(orphan);
            repaired += 1;
        }

        repaired
    }

    /// Lexicographic ordering on a repair-host key `(is_tombstoned, distance,
    /// slot)`: live hosts beat tombstoned ones, then nearer beats farther, then the
    /// lower slot index wins (which keeps the pass deterministic).
    fn host_key_lt(a: (bool, f32, usize), b: (bool, f32, usize)) -> bool {
        a.0.cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
            .is_lt()
    }
}
