//! [`MaintainableIndex`]: a mutable proximity-graph ANN index with in-place
//! insert (with backward-edge repair), tombstone delete, and compacting
//! consolidation.
//!
//! This file holds the slot table, the two-pass Vamana build, `insert`, `delete`,
//! `search` and `stats`. The consolidation and connectivity-repair passes live in
//! the `consolidate` sibling module.

use std::collections::HashMap;

use super::graph::{
    MaintenanceNode, greedy_search, live_medoid, random_regular_edges, reachable_from,
    robust_prune, robust_prune_with_pinned, unreached_live,
};
use super::types::{
    ConsolidationReport, MaintenanceConfig, MaintenanceError, MaintenanceHit, MaintenanceStats,
};

/// A mutable proximity-graph ANN index that survives churn.
///
/// # Slot model
///
/// Vectors live in a dense `Vec` of slots. External `String` ids are mapped to
/// slots through an internal `id_map`, which contains **live slots only**:
///
/// * `insert` appends a slot and adds the mapping;
/// * `delete` drops the mapping and flips the slot's tombstone — no edge is ever
///   touched, so the graph's navigability is *unchanged* by a delete;
/// * `consolidate` is the only operation that removes slots. It bridges every
///   in-edge of every tombstone, compacts the table, remaps every edge, repairs
///   any orphan, and rebuilds `id_map` — so external ids stay valid across the
///   compaction even though the internal slot indices all move.
///
/// See the module documentation for the connectivity argument that makes each of
/// those steps safe.
#[derive(Debug, Clone)]
pub struct MaintainableIndex {
    /// Build/search/maintenance knobs.
    pub(crate) config: MaintenanceConfig,
    /// Dimensionality every stored vector must match.
    pub(crate) dim: usize,
    /// The dense slot table. Never has holes.
    pub(crate) nodes: Vec<MaintenanceNode>,
    /// External id -> slot, for **live** slots only.
    pub(crate) id_map: HashMap<String, usize>,
    /// Entry point for every `GreedySearch`. `None` only for an empty table.
    pub(crate) entry: Option<usize>,
    /// Number of tombstoned slots (kept incrementally; equals the count of
    /// `tombstoned` flags in `nodes`).
    pub(crate) tombstone_count: usize,
}

impl MaintainableIndex {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create an empty index of dimension `dim`.
    ///
    /// # Errors
    ///
    /// Returns [`MaintenanceError::ZeroDimension`] if `dim == 0`, or
    /// [`MaintenanceError::InvalidConfig`] if `config` fails
    /// [`MaintenanceConfig::validate`].
    pub fn new(dim: usize, config: MaintenanceConfig) -> Result<Self, MaintenanceError> {
        config.validate()?;
        if dim == 0 {
            return Err(MaintenanceError::ZeroDimension);
        }
        Ok(Self {
            config,
            dim,
            nodes: Vec::new(),
            id_map: HashMap::new(),
            entry: None,
            tombstone_count: 0,
        })
    }

    /// Build an index from a corpus with a two-pass Vamana construction.
    ///
    /// The graph starts as a deterministic random `R`-regular digraph and is then
    /// refined twice: once with `alpha = 1.0` (which yields a sparse, uniformly
    /// connected graph) and once with the configured `alpha` (which re-admits a
    /// few long-range edges and shrinks the graph diameter). Each pass visits
    /// every node `p` in slot order, runs `GreedySearch(entry, p, L)` to collect a
    /// candidate set, `RobustPrune`s it into `p`'s out-edges, and then installs
    /// the **backward** edges `n -> p` — re-pruning any `n` that overflows `R`.
    ///
    /// Duplicate ids inside `items` are rejected; the first occurrence wins the
    /// error report.
    ///
    /// # Errors
    ///
    /// Returns [`MaintenanceError::EmptyIndex`] for an empty `items` list (the
    /// dimension cannot be inferred — use [`MaintainableIndex::new`]),
    /// [`MaintenanceError::DimensionMismatch`] if the items disagree on
    /// dimension, [`MaintenanceError::ZeroDimension`] for zero-length vectors,
    /// [`MaintenanceError::DuplicateId`] for a repeated id, or
    /// [`MaintenanceError::InvalidConfig`] for a bad config.
    pub fn build(
        items: Vec<(String, Vec<f32>)>,
        config: MaintenanceConfig,
    ) -> Result<Self, MaintenanceError> {
        config.validate()?;
        let Some((_, first)) = items.first() else {
            return Err(MaintenanceError::EmptyIndex);
        };
        let dim = first.len();
        if dim == 0 {
            return Err(MaintenanceError::ZeroDimension);
        }

        let mut index = Self::new(dim, config)?;
        for (id, vector) in items {
            if vector.len() != dim {
                return Err(MaintenanceError::DimensionMismatch {
                    expected: dim,
                    got: vector.len(),
                });
            }
            if index.id_map.contains_key(&id) {
                return Err(MaintenanceError::DuplicateId(id));
            }
            let slot = index.nodes.len();
            index.id_map.insert(id.clone(), slot);
            index.nodes.push(MaintenanceNode::new(id, vector));
        }

        index.entry = live_medoid(&index.nodes, index.config.metric);
        let n = index.nodes.len();
        let edges = random_regular_edges(n, index.config.max_degree);
        for (node, out_edges) in index.nodes.iter_mut().zip(edges) {
            node.out_edges = out_edges;
        }

        index.vamana_pass(1.0);
        index.vamana_pass(index.config.alpha);
        index.repair_connectivity();

        Ok(index)
    }

    /// One Vamana refinement pass over every slot, in deterministic slot order.
    fn vamana_pass(&mut self, alpha: f32) {
        let Some(entry) = self.entry else {
            return;
        };
        let metric = self.config.metric;
        let max_degree = self.config.max_degree;
        let l = self.config.search_list_size;

        for p in 0..self.nodes.len() {
            let query = self.nodes[p].vector.clone();
            let (_, expanded) = greedy_search(&self.nodes, metric, entry, &query, l);
            let selected = robust_prune(&self.nodes, metric, p, expanded, alpha, max_degree);
            self.nodes[p].out_edges.clone_from(&selected);
            self.install_backward_edges(p, &selected, alpha);
        }
    }

    /// Install the backward edges `n -> p` for every newly chosen out-neighbour
    /// `n` of `p`, re-pruning any `n` whose out-degree overflows `R`.
    ///
    /// `p` is **pinned** during that re-prune. Without the pin, `RobustPrune` is
    /// free to discard the very edge that was just added — `p` is a brand-new
    /// node, so it is frequently occluded by a closer, older neighbour of `n` —
    /// and a node whose backward edges are all immediately pruned away has
    /// in-degree zero: it is reachable by no query at all, even though the graph
    /// looks perfectly healthy from the outside. Pinning makes the backward-edge
    /// repair actually load-bearing: after `install_backward_edges`, every
    /// out-neighbour of `p` holds an edge back to `p`, and every one of those
    /// out-neighbours was reached by the `GreedySearch` from the entry point, so
    /// `p` is reachable from the entry point too.
    pub(super) fn install_backward_edges(&mut self, p: usize, selected: &[usize], alpha: f32) {
        let metric = self.config.metric;
        let max_degree = self.config.max_degree;

        for &neighbor in selected {
            if neighbor == p {
                continue;
            }
            if !self.nodes[neighbor].out_edges.contains(&p) {
                self.nodes[neighbor].out_edges.push(p);
            }
            if self.nodes[neighbor].out_edges.len() > max_degree {
                let pool = self.nodes[neighbor].out_edges.clone();
                let pruned = robust_prune_with_pinned(
                    &self.nodes,
                    metric,
                    neighbor,
                    pool,
                    &[p],
                    alpha,
                    max_degree,
                );
                self.nodes[neighbor].out_edges = pruned;
            }
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Insert `vector` under the external id `id`.
    ///
    /// The insert is fully in place — no rebuild, no side index:
    ///
    /// 1. `GreedySearch(entry, vector, L)` collects the candidate set (walking
    ///    *through* any tombstones on the way);
    /// 2. `RobustPrune` over the **live** candidates chooses the new node's
    ///    out-edges (a brand-new node never spends its degree budget on nodes
    ///    that the next consolidation is going to collect);
    /// 3. the **backward** edges `n -> new` are installed with the new node
    ///    pinned, so the new node provably ends up reachable from the entry point
    ///    — see `install_backward_edges`.
    ///
    /// If [`MaintenanceConfig::repair_on_insert`] is set, the connectivity repair
    /// pass then runs, restoring the "no orphaned live node" invariant that step 3
    /// can disturb *for other nodes* (re-pruning a neighbour `n` may drop `n`'s
    /// last edge to some third node).
    ///
    /// An id that was previously deleted may be re-inserted: the tombstoned slot
    /// keeps its (now unmapped) copy of the id until the next consolidation
    /// collects it, and the external id resolves to the new slot from here on.
    ///
    /// # Errors
    ///
    /// Returns [`MaintenanceError::DuplicateId`] if `id` is already **live**, or
    /// [`MaintenanceError::DimensionMismatch`] if `vector` has the wrong length.
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
    ) -> Result<(), MaintenanceError> {
        let id = id.into();
        if vector.len() != self.dim {
            return Err(MaintenanceError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.id_map.contains_key(&id) {
            return Err(MaintenanceError::DuplicateId(id));
        }

        let slot = self.nodes.len();
        self.id_map.insert(id.clone(), slot);
        self.nodes.push(MaintenanceNode::new(id, vector));

        let Some(entry) = self.entry else {
            // First node in the table: it is the entry point by definition.
            self.entry = Some(slot);
            return Ok(());
        };

        let metric = self.config.metric;
        let alpha = self.config.alpha;
        let max_degree = self.config.max_degree;
        let l = self.config.search_list_size;

        let query = self.nodes[slot].vector.clone();
        let (_, expanded) = greedy_search(&self.nodes, metric, entry, &query, l);
        let candidates: Vec<usize> = expanded
            .into_iter()
            .filter(|&v| v != slot && !self.nodes[v].tombstoned)
            .collect();
        let selected = robust_prune(&self.nodes, metric, slot, candidates, alpha, max_degree);
        self.nodes[slot].out_edges.clone_from(&selected);
        self.install_backward_edges(slot, &selected, alpha);

        if self.config.repair_on_insert {
            self.repair_connectivity();
        }
        Ok(())
    }

    /// Delete the entry with external id `id` by **tombstoning** it.
    ///
    /// No edge is removed and no vector is dropped: the node keeps its out-edges
    /// and is still expanded by `GreedySearch`, so every path that ran through it
    /// still runs through it. Only the result filter changes — a tombstoned node
    /// can never be returned by [`MaintainableIndex::search`].
    ///
    /// This is the crux of the module. The obvious alternative — physically
    /// unlinking the node and dropping the in-edges that point at it, as
    /// `layer1_echo`'s HNSW `remove` does — leaves the former in-neighbours with a
    /// hole where an edge used to be and *no replacement*, so every delete
    /// permanently erodes the graph's navigability and recall decays with churn.
    /// Here, connectivity is preserved until [`MaintainableIndex::consolidate`]
    /// can rewire the in-neighbours onto the tombstone's own out-neighbours.
    ///
    /// If [`MaintenanceConfig::tombstone_threshold`] is set and the tombstoned
    /// fraction of the slot table reaches it, a consolidation runs immediately and
    /// its report is returned.
    ///
    /// # Errors
    ///
    /// Returns [`MaintenanceError::UnknownId`] if `id` is not currently live
    /// (never inserted, or already deleted).
    pub fn delete(&mut self, id: &str) -> Result<Option<ConsolidationReport>, MaintenanceError> {
        let Some(slot) = self.id_map.remove(id) else {
            return Err(MaintenanceError::UnknownId(id.to_string()));
        };
        self.nodes[slot].tombstoned = true;
        self.tombstone_count += 1;

        if let Some(threshold) = self.config.tombstone_threshold {
            #[allow(clippy::cast_precision_loss)]
            let ratio = self.tombstone_count as f32 / self.nodes.len() as f32;
            if ratio >= threshold {
                return Ok(Some(self.consolidate()));
            }
        }
        Ok(None)
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Search for the `k` nearest **live** entries to `query`.
    ///
    /// The beam walks through tombstoned nodes but never reports them. Hits are
    /// ranked by descending [`MaintenanceHit::score`].
    ///
    /// Returns an empty vector when the index holds no live node.
    ///
    /// # Errors
    ///
    /// Returns [`MaintenanceError::DimensionMismatch`] if `query` has the wrong
    /// length.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<MaintenanceHit>, MaintenanceError> {
        if query.len() != self.dim {
            return Err(MaintenanceError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 || self.live_len() == 0 {
            return Ok(Vec::new());
        }
        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };

        let metric = self.config.metric;
        let l = self.config.search_list_size.max(k);
        let (beam, _) = greedy_search(&self.nodes, metric, entry, query, l);

        Ok(beam
            .into_iter()
            .filter(|&(_, slot)| !self.nodes[slot].tombstoned)
            .take(k)
            .map(|(_, slot)| {
                MaintenanceHit::new(
                    self.nodes[slot].external_id.clone(),
                    metric.score(query, &self.nodes[slot].vector),
                )
            })
            .collect())
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    /// Number of live entries.
    #[must_use]
    pub fn live_len(&self) -> usize {
        self.nodes.len() - self.tombstone_count
    }

    /// `true` if the index holds no live entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_len() == 0
    }

    /// Number of tombstoned slots awaiting consolidation.
    #[must_use]
    pub fn tombstone_len(&self) -> usize {
        self.tombstone_count
    }

    /// Total slots currently allocated (live plus tombstoned).
    #[must_use]
    pub fn slot_len(&self) -> usize {
        self.nodes.len()
    }

    /// Dimensionality of the indexed vectors.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// The configuration this index was created with.
    #[must_use]
    pub fn config(&self) -> &MaintenanceConfig {
        &self.config
    }

    /// `true` if `id` is currently live.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.id_map.contains_key(id)
    }

    /// The stored vector for a live external id.
    #[must_use]
    pub fn vector_of(&self, id: &str) -> Option<&[f32]> {
        self.id_map
            .get(id)
            .map(|&slot| self.nodes[slot].vector.as_slice())
    }

    /// The external id of the current entry point, if any.
    #[must_use]
    pub fn entry_id(&self) -> Option<&str> {
        self.entry.map(|slot| self.nodes[slot].external_id.as_str())
    }

    /// A health snapshot of the graph.
    ///
    /// `orphan_count` is measured under the traversal rules `search` actually
    /// uses (BFS from the entry point, *through* tombstones), so it counts live
    /// nodes that no query could ever return.
    #[must_use]
    pub fn stats(&self) -> MaintenanceStats {
        let reached = reachable_from(&self.nodes, self.entry, true);
        let orphan_count = unreached_live(&self.nodes, &reached).len();

        let edge_count: usize = self.nodes.iter().map(|n| n.out_edges.len()).sum();
        let max_out_degree = self
            .nodes
            .iter()
            .map(|n| n.out_edges.len())
            .max()
            .unwrap_or(0);
        let live_nodes = self.live_len();
        let live_edges: usize = self
            .nodes
            .iter()
            .filter(|n| !n.tombstoned)
            .map(|n| n.out_edges.len())
            .sum();
        let mean_out_degree_milli = if live_nodes == 0 {
            0
        } else {
            (live_edges as u64 * 1000) / live_nodes as u64
        };

        MaintenanceStats {
            live_nodes,
            tombstoned_nodes: self.tombstone_count,
            total_slots: self.nodes.len(),
            edge_count,
            mean_out_degree_milli,
            max_out_degree,
            orphan_count,
            entry_point_tombstoned: self.entry.is_some_and(|slot| self.nodes[slot].tombstoned),
        }
    }
}
