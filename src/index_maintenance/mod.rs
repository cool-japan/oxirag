//! Online index maintenance: a **mutable** proximity-graph ANN index with
//! in-place inserts, tombstone deletes and compacting consolidation — a live
//! index that survives churn without a full rebuild (`FreshDiskANN`, Singh et
//! al., 2021).
//!
//! # Why this module exists
//!
//! Every other graph/quantisation index in this crate is *build-once* or
//! *append-only*: `disk_ann` and `spann` expose a `build` and
//! nothing else; `hnsw_index`, `lsh_index` and
//! `ivf_index` can grow but can never shrink. The one deletion path that
//! does exist elsewhere in the crate — `layer1_echo`'s HNSW `remove` — is
//! *structurally naive*: it unlinks the node and strips the in-edges that pointed
//! at it, leaving each former in-neighbour with a hole where an edge used to be
//! and **no replacement edge**, and it runs no compaction pass afterwards. Nothing
//! ever re-attaches the neighbourhood the dead node used to bridge, so every
//! delete permanently erodes the graph's navigability: recall decays with churn,
//! silently, and a rebuild is the only cure.
//!
//! This module is the principled fix, and it *measures* the fix rather than
//! asserting it — see the [headline experiment](#the-headline-experiment) below.
//!
//! # The three operations
//!
//! **Insert** ([`MaintainableIndex::insert`]) is fully in place. `GreedySearch`
//! collects a candidate set, `RobustPrune` turns it into the new node's out-edges,
//! and then — the part that actually matters — the **backward** edges `n -> new`
//! are installed on each chosen neighbour `n`, re-pruning any `n` that overflows
//! the degree cap `R`. A forward-only insert produces a node with in-degree zero:
//! it is present, it is correct, and no query can ever reach it. The new node is
//! *pinned* during that re-prune, so `RobustPrune` cannot immediately discard the
//! very edges that make it reachable.
//!
//! **Delete** ([`MaintainableIndex::delete`]) writes a **tombstone**. The node
//! keeps its vector and — crucially — keeps its out-edges, and `GreedySearch`
//! keeps expanding it. A tombstoned node is excluded from search *results* but is
//! still traversed *through*, so every path that ran through it still runs through
//! it and the graph's navigability is exactly what it was before the delete. The
//! index pays for this in beam pressure, not in recall.
//!
//! **Consolidate** ([`MaintainableIndex::consolidate`]) collects the tombstones.
//! For every live `p` pointing at a tombstoned `t`, the edge `p -> t` is replaced
//! by edges to `t`'s **live frontier** (the live nodes reachable from `t` through
//! tombstoned nodes only — a transitive closure, because chains of dead nodes are
//! ordinary) and `p` is `RobustPrune`d back to `R`. Then the tombstones are
//! physically dropped, the slot table is compacted and every edge is remapped, the
//! entry point is re-selected if it died, the bridged nodes are **re-optimised** by
//! a two-pass Vamana refinement, and a **connectivity repair pass** runs. External
//! ids survive the compaction: [`MaintainableIndex`] keeps a stable external-id ->
//! slot map and rebuilds it as part of the compaction, so callers never see an
//! internal slot index move.
//!
//! The refinement step is not decoration. The bridge is a purely *local* repair —
//! it can only rewire `p` using `p`'s own surviving neighbours and the frontiers of
//! its dead ones — and a graph whose adjacency lists have been re-derived from such
//! an impoverished candidate pool comes out fully connected, exactly degree-capped,
//! and measurably **less navigable**. Bridging alone costs ~6 points of recall@10 on
//! the churn benchmark below, with zero orphans and a full out-degree the entire
//! time, which is precisely why the structural invariants alone cannot catch it. The
//! two-pass refinement over the bridged nodes buys it all back.
//!
//! # Orphan freedom is proved, not hoped for
//!
//! `RobustPrune` is a heuristic. Bridging a node can still discard its last edge
//! to some third node, so a rigorous mutable index cannot simply assume the graph
//! stays connected. The connectivity repair pass closes that hole with
//! a terminating argument (given `R >= 2`, which [`MaintenanceConfig::validate`]
//! enforces): let `Reached` be the BFS closure of the entry point, and `u` a live
//! node outside it.
//!
//! * If some `p` in `Reached` has out-degree `< R`, append `p -> u`: `u` is reached,
//!   nothing is lost.
//! * Otherwise every `p` in `Reached` is saturated at `R`, and since every out-edge
//!   of a reached node lands inside `Reached`, the component holds `R * |Reached|`
//!   internal edges — mean in-degree `R >= 2`. By pigeonhole some `w` in `Reached`
//!   has in-degree `>= 2`; re-point one of its in-edges `p -> w` at `u`. `w` keeps
//!   another in-edge from the component so it stays reached, `u` becomes reached,
//!   and `p`'s degree is unchanged.
//!
//! Each step strictly shrinks the orphan set and neither step can create a new
//! orphan, so the pass terminates and leaves **every live node reachable from the
//! entry point at out-degree `<= R`**. That is a hard invariant of this module,
//! checked after every mutation in the test suite.
//!
//! # The headline experiment
//!
//! The module's justification is a measurement, not a claim. `tests.rs` builds a
//! clustered corpus, records exact brute-force ground truth, measures `recall@10`,
//! then applies the same churn workload (delete 30% of the corpus, insert 30% new
//! vectors) under three regimes and re-measures against fresh ground truth:
//!
//! | Regime | Delete strategy | recall@10 | orphaned live nodes |
//! |--------|-----------------|-----------|---------------------|
//! | — | *pre-churn baseline* | 0.954 | 0 |
//! | (a) naive hard delete | unlink the node, strip in-edges, no reconnection (what `layer1_echo` does) | **0.875** | **49** |
//! | (b) tombstone, unconsolidated | soft delete, traverse through | 0.958 | 6 |
//! | (c) tombstone + `consolidate()` | soft delete, then bridge + compact + refine + repair | **0.963** | **0** |
//!
//! Regimes (a) and (c) end with *no* tombstones in the graph, so they explore the
//! same number of nodes per query: the comparison is budget-identical and the gap is
//! pure graph quality. (b) keeps its recall but still carries the dead weight; (c)
//! keeps its recall *and* hands back the memory.
//!
//! The orphan column is the mechanism, and it is categorical rather than a matter of
//! degree. Measured on the deletes alone — before any insert can confound it — the
//! soft delete strands **exactly zero** live nodes (it does not touch a single edge)
//! while the hard delete strands **40**. An orphaned node is unreachable from the
//! entry point: no query returns it at any beam width, ever, and nothing in the naive
//! scheme will ever repair it. The recall loss tracks that orphan fraction almost
//! exactly. (b)'s six orphans have a different origin — with `repair_on_insert`
//! disabled, as the benchmark does to isolate the delete strategy, an *insert* can
//! prune away some third node's last in-edge; that is exactly what the repair pass
//! exists to fix, and (c) ends at zero.
//!
//! A denser graph (large `R`) hides the naive damage, because a high in-degree
//! absorbs the stripped edges. The failure is real but only bites at the sparse
//! operating points that large indexes actually run at — `R` stays fixed while `N`
//! grows, and Vamana's in-degree distribution has a long low tail.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "index-maintenance")] {
//! use oxirag::index_maintenance::{MaintainableIndex, MaintenanceConfig};
//!
//! fn embed(seed: u64, dim: usize) -> Vec<f32> {
//!     (0..dim)
//!         .map(|i| {
//!             let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
//!             z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
//!             z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
//!             ((z >> 40) as f32) / 16_777_216.0 - 0.5
//!         })
//!         .collect()
//! }
//!
//! let items: Vec<(String, Vec<f32>)> = (0..64u64)
//!     .map(|i| (format!("doc_{i}"), embed(i, 16)))
//!     .collect();
//!
//! let config = MaintenanceConfig::new()
//!     .with_max_degree(12)
//!     .with_search_list_size(32);
//! let mut index = MaintainableIndex::build(items, config).unwrap();
//!
//! // Churn: the index stays queryable throughout.
//! index.delete("doc_7").unwrap();
//! index.insert("doc_64", embed(64, 16)).unwrap();
//!
//! // A tombstoned node is routed through but never reported.
//! let hits = index.search(&embed(7, 16), 5).unwrap();
//! assert!(hits.iter().all(|h| h.id != "doc_7"));
//!
//! // Consolidation collects the tombstone and leaves the graph healthy.
//! let report = index.consolidate();
//! assert_eq!(report.nodes_removed, 1);
//! let stats = index.stats();
//! assert_eq!(stats.tombstoned_nodes, 0);
//! assert_eq!(stats.orphan_count, 0);
//! # }
//! ```

mod consolidate;
mod graph;
pub mod index;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use index::MaintainableIndex;
pub use types::{
    ConsolidationReport, MaintenanceConfig, MaintenanceError, MaintenanceHit, MaintenanceMetric,
    MaintenanceStats,
};
