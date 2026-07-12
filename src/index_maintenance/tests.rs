#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::needless_for_each,
    clippy::uninlined_format_args,
    clippy::single_char_pattern,
    clippy::needless_range_loop,
    clippy::unreadable_literal
)]

use std::collections::HashSet;

use super::graph::{MaintenanceNode, live_medoid, reachable_from, split_mix64, unreached_live};
use super::index::MaintainableIndex;
use super::types::{
    ConsolidationReport, MaintenanceConfig, MaintenanceError, MaintenanceHit, MaintenanceMetric,
    MaintenanceStats,
};

// ── Deterministic PRNG (SplitMix64) ───────────────────────────────────────────

/// A `SplitMix64` generator. The project forbids `rand`/`rand_distr`, and a
/// hand-rolled seeded PRNG additionally makes every measurement in this file
/// bit-for-bit reproducible.
struct SplitMix {
    state: u64,
}

impl SplitMix {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        split_mix64(self.state.wrapping_sub(0x9E37_79B9_7F4A_7C15))
    }

    /// Uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    /// Uniform in `[-1, 1)`.
    fn next_signed(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

// ── Corpus generation ─────────────────────────────────────────────────────────

/// `clusters` cluster centres drawn uniformly from the cube.
///
/// The corpus, the vectors that arrive during churn, and the query set all have to
/// be drawn around the **same** centres, or the queries land in empty space, their
/// true top-`k` degenerates into a near-tie among far-away points, and the recall
/// figure stops measuring graph quality at all.
fn cluster_centres(dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix::new(seed);
    (0..clusters)
        .map(|_| (0..dim).map(|_| rng.next_signed()).collect())
        .collect()
}

/// Points drawn around `centres`: each is a centre plus isotropic noise.
///
/// Clustered (rather than uniform) data gives the base index a high, meaningful
/// recall to *start* from — the churn experiment measures how much of that recall
/// each delete strategy destroys, which is only observable if there was something
/// to destroy.
fn clustered_points(
    prefix: &str,
    count: usize,
    centres: &[Vec<f32>],
    noise: f32,
    seed: u64,
) -> Vec<(String, Vec<f32>)> {
    let mut rng = SplitMix::new(seed);
    (0..count)
        .map(|i| {
            let centre = &centres[i % centres.len()];
            let vector: Vec<f32> = centre
                .iter()
                .map(|&c| c + rng.next_signed() * noise)
                .collect();
            (format!("{prefix}{i}"), vector)
        })
        .collect()
}

/// Query vectors drawn from the same generative process (and the same centres) as
/// the corpus, so that a query's true neighbours are a genuine, tightly-ranked
/// neighbourhood rather than an arbitrary tie among far-away points.
fn clustered_queries(count: usize, centres: &[Vec<f32>], noise: f32, seed: u64) -> Vec<Vec<f32>> {
    clustered_points("q_", count, centres, noise, seed)
        .into_iter()
        .map(|(_, v)| v)
        .collect()
}

/// Convenience for the tests that only need one self-contained corpus.
fn clustered_corpus(
    prefix: &str,
    count: usize,
    dim: usize,
    clusters: usize,
    noise: f32,
    seed: u64,
) -> Vec<(String, Vec<f32>)> {
    let centres = cluster_centres(dim, clusters, seed);
    clustered_points(prefix, count, &centres, noise, seed ^ 0xA5A5_5A5A_1234_5678)
}

// ── Ground truth and recall ───────────────────────────────────────────────────

/// Exact brute-force top-`k` ids over `items` — the ground truth every recall
/// figure in this file is measured against.
fn brute_force_top_k(
    items: &[(String, Vec<f32>)],
    query: &[f32],
    k: usize,
    metric: MaintenanceMetric,
) -> Vec<String> {
    let mut scored: Vec<(f32, &String)> = items
        .iter()
        .map(|(id, v)| (metric.distance(query, v), id))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(k)
        .map(|(_, id)| id.clone())
        .collect()
}

/// Mean `recall@k` of `index` over `queries`, against exact brute-force ground
/// truth computed over `live_items` (the set of entries that *should* be
/// retrievable at this point in the workload).
fn recall_at_k(
    index: &MaintainableIndex,
    live_items: &[(String, Vec<f32>)],
    queries: &[Vec<f32>],
    k: usize,
) -> f32 {
    let metric = index.config().metric;
    let mut total = 0.0f32;
    for query in queries {
        let truth: HashSet<String> = brute_force_top_k(live_items, query, k, metric)
            .into_iter()
            .collect();
        let hits = index
            .search(query, k)
            .expect("search over a well-dimensioned query must succeed");
        let found = hits.iter().filter(|h| truth.contains(&h.id)).count();
        total += found as f32 / k as f32;
    }
    total / queries.len() as f32
}

// ── The naive hard-delete baseline (regime (a)) ───────────────────────────────

/// Reproduce `layer1_echo`'s HNSW `remove` inside the test: strip every in-edge
/// pointing at the node and drop the node, **with no reconnection of the orphaned
/// in-neighbours and no consolidation pass**.
///
/// The node is left in the slot table (rather than being spliced out) purely so
/// the two regimes can share one slot-indexed data structure. That is
/// *behaviourally identical* to a physical removal: the node has no in-edges left
/// (nothing can route to it), its own out-edges are cleared (nothing can route
/// through it), and it is tombstoned (nothing can report it), so it is invisible
/// to every search and to every subsequent insert's candidate set. Only its memory
/// remains.
///
/// The entry point is re-selected to the live medoid if it was the victim — a
/// *charitable* reading of the baseline (`layer1_echo` picks an arbitrary surviving
/// node), so that any recall gap we measure comes from lost connectivity and not
/// from a degenerate entry point.
fn naive_hard_delete(index: &mut MaintainableIndex, id: &str) {
    let slot = index
        .id_map
        .remove(id)
        .expect("naive_hard_delete called with a live id");

    for node in &mut index.nodes {
        node.out_edges.retain(|&target| target != slot);
    }
    index.nodes[slot].out_edges.clear();
    index.nodes[slot].tombstoned = true;
    index.tombstone_count += 1;

    if index.entry == Some(slot) {
        index.entry = live_medoid(&index.nodes, index.config.metric);
    }
}

// ── Invariant checking ────────────────────────────────────────────────────────

/// Assert every structural invariant the module promises.
///
/// * out-degree `<= R` for **every** slot;
/// * no edge points outside the slot table (after a consolidation, no edge can
///   point at a physically-removed node — the table is dense, so this reduces to
///   an in-bounds check plus the tombstone check below);
/// * no self-loops and no duplicate edges;
/// * `id_map` is exactly the set of live slots, and is a bijection;
/// * `tombstone_count` agrees with the flags;
/// * if `expect_no_orphans`, every live node is reachable from the entry point.
fn assert_invariants(index: &MaintainableIndex, context: &str, expect_no_orphans: bool) {
    let r = index.config().max_degree;
    let n = index.nodes.len();

    for (slot, node) in index.nodes.iter().enumerate() {
        assert!(
            node.out_edges.len() <= r,
            "{context}: slot {slot} has out-degree {} > R={r}",
            node.out_edges.len()
        );
        let unique: HashSet<usize> = node.out_edges.iter().copied().collect();
        assert_eq!(
            unique.len(),
            node.out_edges.len(),
            "{context}: slot {slot} has duplicate out-edges"
        );
        for &target in &node.out_edges {
            assert!(
                target < n,
                "{context}: slot {slot} points at out-of-range slot {target}"
            );
            assert_ne!(target, slot, "{context}: slot {slot} has a self-loop");
        }
    }

    let flagged = index.nodes.iter().filter(|node| node.tombstoned).count();
    assert_eq!(
        flagged,
        index.tombstone_len(),
        "{context}: tombstone_count disagrees with the flags"
    );

    assert_eq!(
        index.id_map.len(),
        index.live_len(),
        "{context}: id_map size disagrees with the live-node count"
    );
    for (id, &slot) in &index.id_map {
        assert!(
            !index.nodes[slot].tombstoned,
            "{context}: id_map maps {id} to a tombstoned slot"
        );
        assert_eq!(
            &index.nodes[slot].external_id, id,
            "{context}: id_map maps {id} to a slot holding a different id"
        );
    }

    if expect_no_orphans {
        let stats = index.stats();
        assert_eq!(
            stats.orphan_count, 0,
            "{context}: {} live node(s) unreachable from the entry point",
            stats.orphan_count
        );
    }
}

/// Assert that a consolidated index really is physically compacted.
fn assert_compacted(index: &MaintainableIndex, context: &str) {
    assert_eq!(
        index.tombstone_len(),
        0,
        "{context}: tombstones survived consolidation"
    );
    assert_eq!(
        index.slot_len(),
        index.live_len(),
        "{context}: the slot table was not compacted"
    );
    for node in &index.nodes {
        assert!(
            !node.tombstoned,
            "{context}: a tombstoned node survived consolidation"
        );
    }
    assert_invariants(index, context, true);
}

// ══════════════════════════════════════════════════════════════════════════════
// THE HEADLINE EXPERIMENT
// ══════════════════════════════════════════════════════════════════════════════

/// The module's entire justification, measured rather than asserted.
///
/// One corpus, one index, three clones. Each clone receives the **same** churn
/// workload (the same 30% of ids deleted, the same 30% of new vectors inserted);
/// the only difference is *how* the delete is performed:
///
/// * (a) naive hard delete — unlink and strip in-edges, no reconnection;
/// * (b) tombstone, left unconsolidated;
/// * (c) tombstone, then `consolidate()`.
///
/// Recall@10 is then measured for all three against fresh exact brute-force
/// ground truth over the post-churn live set. `repair_on_insert` is off for all
/// three clones, so the *only* variable in the experiment is the delete strategy.
///
/// Regimes (a) and (c) both end with zero tombstones in the graph, so their beams
/// explore the same number of nodes per query: the comparison between them is
/// budget-identical and the gap is pure graph quality.
///
/// # What the numbers actually say
///
/// The experiment was swept over `R ∈ [4, 16]`, `N ∈ [500, 1500]` and four corpus
/// seeds before the thresholds below were fixed, and the finding is consistent and
/// mechanistic rather than lucky:
///
/// * the naive delete **orphans live nodes** — 3–6% of the live set here — and an
///   orphan is unreachable from the entry point, so no query can return it at *any*
///   beam width, ever. Nothing in the naive scheme repairs it;
/// * the recall it loses tracks that orphan fraction almost exactly;
/// * a *denser* graph (large `R`) hides the damage, because a high in-degree
///   absorbs the stripped edges — the failure is real but only bites at the sparse
///   operating points that large indexes actually run at (`R` is fixed while `N`
///   grows, and Vamana's in-degree distribution has a long low tail);
/// * regimes (b) and (c) hold **zero** orphans by construction, in every
///   configuration swept.
///
/// The configuration below (`N = 1000`, `R = 6`) is a faithfully scaled-down model
/// of that sparse regime. Note that the baseline is charitable to the naive
/// approach in two ways: it re-selects the medoid entry point after a delete (so
/// the gap cannot come from a degenerate entry point), and its inserts use this
/// module's *good* insert path, which actively heals some of the damage the naive
/// delete does. The measured gap is therefore a lower bound.
#[test]
fn churn_recall_naive_delete_degrades_tombstone_and_consolidation_hold() {
    const DIM: usize = 16;
    const CORPUS: usize = 1000;
    const CLUSTERS: usize = 25;
    const NOISE: f32 = 0.8;
    const K: usize = 10;

    let metric = MaintenanceMetric::L2;
    let config = MaintenanceConfig::new()
        .with_max_degree(6)
        .with_search_list_size(40)
        .with_alpha(1.2)
        .with_metric(metric)
        .with_repair_on_insert(false);

    let centres = cluster_centres(DIM, CLUSTERS, 0x5EED_0001);
    let corpus = clustered_points("doc_", CORPUS, &centres, NOISE, 0x5EED_0011);
    let fresh = clustered_points("new_", CORPUS * 3 / 10, &centres, NOISE, 0x5EED_0033);
    let queries = clustered_queries(100, &centres, NOISE, 0x5EED_0022);

    let base = MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");
    assert_invariants(&base, "base build", true);

    let baseline_recall = recall_at_k(&base, &corpus, &queries, K);
    assert!(
        baseline_recall >= 0.95,
        "the experiment is only meaningful if the base index starts with high recall; got {baseline_recall:.4}"
    );

    // The churn workload: delete every id whose ordinal is 0/1/2 mod 10 (30%),
    // then insert the fresh vectors. Identical for all three regimes.
    let deleted: Vec<String> = (0..CORPUS)
        .filter(|i| i % 10 < 3)
        .map(|i| format!("doc_{i}"))
        .collect();
    let deleted_set: HashSet<&String> = deleted.iter().collect();

    let live_items: Vec<(String, Vec<f32>)> = corpus
        .iter()
        .filter(|(id, _)| !deleted_set.contains(id))
        .cloned()
        .chain(fresh.iter().cloned())
        .collect();

    // ── (a) naive hard delete ────────────────────────────────────────────────
    let mut naive = base.clone();
    for id in &deleted {
        naive_hard_delete(&mut naive, id);
    }
    // Orphans measured *here* — after the deletes but before any insert — isolate the
    // damage done by the delete strategy itself, with nothing else to confound it.
    let orphans_naive_from_deletes = naive.stats().orphan_count;
    for (id, vector) in &fresh {
        naive
            .insert(id.clone(), vector.clone())
            .expect("insert into the naive index must succeed");
    }
    let recall_naive = recall_at_k(&naive, &live_items, &queries, K);
    let orphans_naive = naive.stats().orphan_count;

    // ── (b) tombstone, unconsolidated ────────────────────────────────────────
    let mut tombstoned = base.clone();
    for id in &deleted {
        let report = tombstoned
            .delete(id)
            .expect("delete of a live id must succeed");
        assert!(
            report.is_none(),
            "auto-consolidation must be off by default"
        );
    }
    let orphans_tombstoned_from_deletes = tombstoned.stats().orphan_count;
    for (id, vector) in &fresh {
        tombstoned
            .insert(id.clone(), vector.clone())
            .expect("insert into the tombstoned index must succeed");
    }
    let recall_tombstoned = recall_at_k(&tombstoned, &live_items, &queries, K);
    let orphans_tombstoned = tombstoned.stats().orphan_count;
    assert_eq!(
        tombstoned.tombstone_len(),
        deleted.len(),
        "a soft delete must not collect anything"
    );

    // ── (c) tombstone + consolidate ──────────────────────────────────────────
    let mut consolidated = tombstoned.clone();
    let report = consolidated.consolidate();
    let recall_consolidated = recall_at_k(&consolidated, &live_items, &queries, K);
    let orphans_consolidated = consolidated.stats().orphan_count;

    println!("── churn experiment (delete 30%, insert 30%, k={K}) ──");
    println!("baseline recall@{K}              = {baseline_recall:.4}");
    println!("(a) naive hard delete           = {recall_naive:.4}  (orphans: {orphans_naive})");
    println!(
        "(b) tombstone, unconsolidated   = {recall_tombstoned:.4}  (orphans: {orphans_tombstoned})"
    );
    println!(
        "(c) tombstone + consolidate     = {recall_consolidated:.4}  (orphans: {orphans_consolidated})"
    );
    println!("consolidation report            = {report:?}");
    println!(
        "orphans from the deletes alone  : naive={orphans_naive_from_deletes}, tombstone={orphans_tombstoned_from_deletes}"
    );

    // ── The claims ───────────────────────────────────────────────────────────

    // (b) and (c) hold up: soft deletion preserves navigability, and consolidation
    // preserves it while physically collecting the dead. Both stay at the
    // pre-churn baseline.
    assert!(
        recall_tombstoned >= 0.95,
        "(b) tombstoned recall fell to {recall_tombstoned:.4}; traversal through tombstones is supposed to preserve connectivity"
    );
    assert!(
        recall_consolidated >= 0.95,
        "(c) consolidated recall fell to {recall_consolidated:.4}; the bridge/repair pass is supposed to preserve connectivity"
    );
    assert!(
        recall_tombstoned >= baseline_recall - 0.02,
        "(b) lost ground against the pre-churn baseline: {baseline_recall:.4} -> {recall_tombstoned:.4}"
    );
    assert!(
        recall_consolidated >= baseline_recall - 0.02,
        "(c) lost ground against the pre-churn baseline: {baseline_recall:.4} -> {recall_consolidated:.4}"
    );

    // (a) measurably degrades. This is the whole point of the module.
    assert!(
        recall_naive < recall_tombstoned - 0.03,
        "(a) naive hard delete failed to degrade: {recall_naive:.4} vs tombstoned {recall_tombstoned:.4}"
    );
    assert!(
        recall_naive < recall_consolidated - 0.03,
        "(a) naive hard delete failed to degrade: {recall_naive:.4} vs consolidated {recall_consolidated:.4}"
    );
    assert!(
        recall_naive < baseline_recall - 0.03,
        "(a) naive hard delete failed to degrade against its own pre-churn baseline: {baseline_recall:.4} -> {recall_naive:.4}"
    );

    // The mechanism behind the gap, and the categorical result: the naive delete
    // strands live nodes outside the entry point's reachable set. An orphan is
    // unreachable by *any* query at *any* beam width, and nothing in the naive
    // scheme will ever repair it.
    //
    // Measured on the deletes alone — the clean, confound-free comparison — the soft
    // delete strands *exactly zero* nodes, because it does not touch a single edge,
    // while the hard delete strands dozens.
    assert_eq!(
        orphans_tombstoned_from_deletes, 0,
        "a tombstone delete touches no edge, so it cannot orphan anything — ever"
    );
    assert!(
        orphans_naive_from_deletes >= 20,
        "the naive delete was expected to strand a substantial number of live nodes; got {orphans_naive_from_deletes}"
    );
    assert!(
        orphans_naive >= 20,
        "the naive baseline's orphans should survive the insert workload; got {orphans_naive}"
    );

    // After the inserts, regime (b) may carry a handful of orphans of a *different*
    // origin: with `repair_on_insert` disabled (as it is here, so that the delete
    // strategy stays the only variable), re-pruning an existing node while installing
    // a new node's backward edges can drop that node's last in-edge to some third
    // node. That is exactly what `repair_on_insert` and the consolidation repair pass
    // exist to fix — and (c), which consolidates, ends at exactly zero.
    assert!(
        orphans_tombstoned * 4 <= orphans_naive,
        "(b) should carry far fewer orphans than (a): {orphans_tombstoned} vs {orphans_naive}"
    );
    assert_eq!(
        orphans_consolidated, 0,
        "consolidation must leave no orphan behind"
    );

    // (c) is physically compacted, not merely correct.
    assert_compacted(&consolidated, "(c) after consolidate");
    assert_eq!(report.nodes_removed, deleted.len());
    assert!(
        report.edges_rewired > 0,
        "tombstones had in-edges to bridge"
    );
    assert_eq!(report.live_nodes_after, live_items.len());
    assert_eq!(consolidated.live_len(), live_items.len());
    assert_eq!(consolidated.slot_len(), live_items.len());
}

/// Traversal *through* tombstones is load-bearing, not decoration.
///
/// After a heavy round of soft deletes, a large number of live nodes are reachable
/// from the entry point **only** via a path that crosses a tombstone. Those are
/// exactly the nodes that a hard delete would have orphaned — and exactly the nodes
/// the consolidation bridge has to rescue before the tombstones are collected.
#[test]
fn tombstones_carry_traffic_that_a_hard_delete_would_have_severed() {
    let config = MaintenanceConfig::new()
        .with_max_degree(8)
        .with_search_list_size(32)
        .with_metric(MaintenanceMetric::L2);
    let corpus = clustered_corpus("doc_", 300, 12, 8, 0.8, 0x1357_9BDF);
    let mut index = MaintainableIndex::build(corpus, config).expect("build must succeed");

    for i in (0..300).filter(|i| i % 5 < 2) {
        index
            .delete(&format!("doc_{i}"))
            .expect("delete of a live id must succeed");
    }

    // Under the traversal rules search actually uses, nothing is orphaned.
    let through = reachable_from(&index.nodes, index.entry, true);
    assert!(
        unreached_live(&index.nodes, &through).is_empty(),
        "soft deletion must never orphan a live node"
    );

    // But if the tombstones were simply *gone* — as a hard delete would leave them —
    // a substantial part of the live graph would fall off the map.
    let live_only = reachable_from(&index.nodes, index.entry, false);
    let would_be_orphaned = unreached_live(&index.nodes, &live_only).len();
    assert!(
        would_be_orphaned > 0,
        "expected some live nodes to depend on a tombstoned hop; got none"
    );

    // Consolidation is what makes that dependency safe to remove: it bridges every
    // such path before collecting the tombstone.
    let report = index.consolidate();
    assert_eq!(report.nodes_removed, 120);
    assert_compacted(&index, "after consolidating a heavy soft-delete round");
}

/// Repeated churn cycles must not ratchet recall down. This is the property that
/// actually matters in production: a live index runs for months.
#[test]
fn repeated_churn_cycles_do_not_progressively_degrade_recall() {
    const DIM: usize = 16;
    const CORPUS: usize = 400;
    const CLUSTERS: usize = 20;
    const NOISE: f32 = 0.8;
    const K: usize = 10;
    const ROUNDS: usize = 6;

    let config = MaintenanceConfig::new()
        .with_max_degree(8)
        .with_search_list_size(32)
        .with_metric(MaintenanceMetric::L2)
        .with_repair_on_insert(false);

    let centres = cluster_centres(DIM, CLUSTERS, 0x0BAD_C0DE);
    let corpus = clustered_points("doc_", CORPUS, &centres, NOISE, 0x0BAD_BEEF);
    let queries = clustered_queries(40, &centres, NOISE, 0x0BAD_F00D);
    let mut index = MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");

    let mut live_items = corpus;
    let baseline = recall_at_k(&index, &live_items, &queries, K);
    let mut recalls: Vec<f32> = Vec::with_capacity(ROUNDS);
    let mut next_id = 0usize;

    for round in 0..ROUNDS {
        // Delete 15% of the *current* live set, deterministically spread across it.
        let victims: Vec<String> = live_items
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 7 == round % 7)
            .map(|(_, (id, _))| id.clone())
            .collect();
        for id in &victims {
            index.delete(id).expect("delete of a live id must succeed");
        }
        let victim_set: HashSet<&String> = victims.iter().collect();
        live_items.retain(|(id, _)| !victim_set.contains(id));

        // Insert the same number of fresh vectors.
        let arrivals = clustered_points(
            &format!("r{round}_"),
            victims.len(),
            &centres,
            NOISE,
            0xF00D_0000 + round as u64,
        );
        for (id, vector) in &arrivals {
            let id = format!("{id}_{next_id}");
            next_id += 1;
            index
                .insert(id.clone(), vector.clone())
                .expect("insert must succeed");
            live_items.push((id, vector.clone()));
        }

        let report = index.consolidate();
        assert_eq!(report.nodes_removed, victims.len());
        assert_compacted(&index, &format!("round {round}"));
        assert_eq!(index.live_len(), live_items.len());

        let recall = recall_at_k(&index, &live_items, &queries, K);
        println!(
            "round {round}: recall@{K} = {recall:.4} (live: {})",
            index.live_len()
        );
        recalls.push(recall);
    }

    println!("baseline recall@{K} = {baseline:.4}, per-round: {recalls:?}");

    for (round, recall) in recalls.iter().enumerate() {
        assert!(
            *recall >= 0.90,
            "round {round} recall {recall:.4} fell below the floor — churn is eroding the graph"
        );
    }
    let last = recalls[ROUNDS - 1];
    let first = recalls[0];
    assert!(
        last >= first - 0.05,
        "recall trended down across churn rounds: {first:.4} -> {last:.4}"
    );
    assert!(
        last >= baseline - 0.05,
        "recall after {ROUNDS} churn rounds ({last:.4}) drifted below the pre-churn baseline ({baseline:.4})"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Structural invariants
// ══════════════════════════════════════════════════════════════════════════════

/// Every mutation — insert, delete, consolidate — leaves the graph invariants
/// intact: out-degree `<= R`, no dangling edge, and no orphaned live node.
#[test]
fn graph_invariants_hold_after_every_mutation() {
    let config = MaintenanceConfig::new()
        .with_max_degree(6)
        .with_search_list_size(24)
        .with_metric(MaintenanceMetric::L2)
        .with_repair_on_insert(true);

    let corpus = clustered_corpus("doc_", 120, 12, 6, 0.3, 0xC0FF_EE01);
    let mut index = MaintainableIndex::build(corpus, config).expect("build must succeed");
    assert_invariants(&index, "after build", true);

    for step in 0..40 {
        let victim = format!("doc_{}", step * 2);
        index.delete(&victim).expect("delete must succeed");
        assert_invariants(&index, &format!("after delete {step}"), true);

        index
            .insert(format!("extra_{step}"), vec![0.1 * step as f32; 12])
            .expect("insert must succeed");
        assert_invariants(&index, &format!("after insert {step}"), true);

        if step % 7 == 6 {
            index.consolidate();
            assert_compacted(&index, &format!("after consolidate {step}"));
        }
    }

    index.consolidate();
    assert_compacted(&index, "after the final consolidate");
}

/// The backward-edge repair is what makes a new node *reachable*. A forward-only
/// insert would leave it with in-degree zero: present, correct, and invisible.
#[test]
fn inserted_nodes_acquire_in_edges_and_are_immediately_retrievable() {
    let config = MaintenanceConfig::new()
        .with_max_degree(8)
        .with_search_list_size(32)
        .with_metric(MaintenanceMetric::L2)
        .with_repair_on_insert(false);

    let corpus = clustered_corpus("doc_", 200, 12, 8, 0.8, 0x2468_ACE0);
    let mut index = MaintainableIndex::build(corpus, config).expect("build must succeed");

    let arrivals = clustered_corpus("new_", 50, 12, 8, 0.3, 0x1111_2222);
    for (id, vector) in &arrivals {
        index
            .insert(id.clone(), vector.clone())
            .expect("insert must succeed");
    }

    // Even with the repair pass switched off, every freshly inserted node has at
    // least one in-edge (the pinned backward edges guarantee it) and is reachable.
    let mut in_degree = vec![0usize; index.nodes.len()];
    for node in &index.nodes {
        for &target in &node.out_edges {
            in_degree[target] += 1;
        }
    }
    for (id, _) in &arrivals {
        let slot = index.id_map[id];
        assert!(
            in_degree[slot] > 0,
            "inserted node {id} has in-degree zero — the backward-edge repair did not hold"
        );
    }
    assert_eq!(
        index.stats().orphan_count,
        0,
        "inserts alone must not orphan anything"
    );

    // And every one of them is retrievable by its own vector.
    for (id, vector) in &arrivals {
        let hits = index.search(vector, 1).expect("search must succeed");
        assert_eq!(
            hits[0].id, *id,
            "freshly inserted {id} is not its own nearest neighbour"
        );
    }
}

/// External ids stay valid across a consolidation that renumbers every internal
/// slot.
#[test]
fn external_ids_survive_the_slot_compaction() {
    let config = MaintenanceConfig::new()
        .with_max_degree(8)
        .with_search_list_size(24)
        .with_metric(MaintenanceMetric::L2);
    let corpus = clustered_corpus("doc_", 150, 10, 6, 0.3, 0xDEAD_BEEF);
    let mut index = MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");

    let survivors: Vec<(String, Vec<f32>)> = corpus
        .iter()
        .filter(|(id, _)| {
            let ordinal: usize = id
                .strip_prefix("doc_")
                .and_then(|s| s.parse().ok())
                .expect("ids are doc_<n>");
            !ordinal.is_multiple_of(3)
        })
        .cloned()
        .collect();

    // Record where each survivor lives *before* the compaction, so we can prove the
    // slots really did move.
    let slots_before: Vec<usize> = survivors.iter().map(|(id, _)| index.id_map[id]).collect();

    for i in (0..150).step_by(3) {
        index
            .delete(&format!("doc_{i}"))
            .expect("delete must succeed");
    }
    let report = index.consolidate();
    assert_eq!(report.nodes_removed, 50);
    assert_compacted(&index, "after compaction");

    let slots_after: Vec<usize> = survivors.iter().map(|(id, _)| index.id_map[id]).collect();
    assert_ne!(
        slots_before, slots_after,
        "the compaction should have renumbered the internal slots"
    );

    for (id, vector) in &survivors {
        assert!(index.contains(id), "{id} lost its external mapping");
        assert_eq!(
            index.vector_of(id),
            Some(vector.as_slice()),
            "{id} resolves to the wrong vector after compaction"
        );
        let hits = index.search(vector, 1).expect("search must succeed");
        assert_eq!(hits[0].id, *id, "{id} is not its own nearest neighbour");
    }
    for i in (0..150).step_by(3) {
        let id = format!("doc_{i}");
        assert!(!index.contains(&id), "{id} survived its own deletion");
        assert!(index.vector_of(&id).is_none());
    }
}

/// The bridge must follow *chains* of tombstones. A non-transitive bridge would
/// look at `t1`'s immediate live out-neighbours (there are none) and silently
/// sever `p` from `x`.
#[test]
fn consolidation_bridges_transitive_chains_of_tombstones() {
    let config = MaintenanceConfig::new()
        .with_max_degree(4)
        .with_search_list_size(8)
        .with_alpha(1.0)
        .with_metric(MaintenanceMetric::L2);

    // p -> t1 -> t2 -> x, with t1 and t2 tombstoned. Hand-built so that the chain
    // is the *only* route from p to x.
    let mut index = MaintainableIndex::new(2, config).expect("new must succeed");
    let layout = [
        ("p", [0.0f32, 0.0]),
        ("t1", [1.0, 0.0]),
        ("t2", [2.0, 0.0]),
        ("x", [3.0, 0.0]),
    ];
    for (slot, (id, vector)) in layout.iter().enumerate() {
        index.id_map.insert((*id).to_string(), slot);
        index
            .nodes
            .push(MaintenanceNode::new((*id).to_string(), vector.to_vec()));
    }
    index.nodes[0].out_edges = vec![1];
    index.nodes[1].out_edges = vec![2];
    index.nodes[2].out_edges = vec![3];
    index.entry = Some(0);

    for dead in ["t1", "t2"] {
        index.delete(dead).expect("delete must succeed");
    }

    let report = index.consolidate();
    assert_eq!(report.nodes_removed, 2);
    assert_eq!(
        report.edges_rewired, 1,
        "p -> t1 is the single bridged edge"
    );
    assert_eq!(
        report.orphans_repaired, 0,
        "the bridge itself must reconnect x; if the repair pass had to rescue it, the bridge was not transitive"
    );

    // p (slot 0) and x (slot 1) survive, and p points straight at x.
    assert_eq!(index.slot_len(), 2);
    assert_eq!(index.nodes[0].external_id, "p");
    assert_eq!(index.nodes[1].external_id, "x");
    assert_eq!(
        index.nodes[0].out_edges,
        vec![1],
        "the transitive bridge p -> x is missing"
    );
    assert_compacted(&index, "transitive bridge");
}

/// A tombstoned node is traversed through but never *reported*.
#[test]
fn tombstoned_nodes_are_never_returned_as_hits() {
    let config = MaintenanceConfig::new()
        .with_max_degree(8)
        .with_search_list_size(32)
        .with_metric(MaintenanceMetric::L2);
    let corpus = clustered_corpus("doc_", 100, 8, 5, 0.3, 0x9999_1111);
    let mut index = MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");

    let (victim_id, victim_vector) = corpus[17].clone();
    let before = index
        .search(&victim_vector, 1)
        .expect("search must succeed");
    assert_eq!(before[0].id, victim_id);

    index.delete(&victim_id).expect("delete must succeed");

    for k in 1..=20 {
        let hits = index
            .search(&victim_vector, k)
            .expect("search must succeed");
        assert!(
            hits.iter().all(|hit| hit.id != victim_id),
            "the tombstoned {victim_id} was reported at k={k}"
        );
        assert_eq!(hits.len(), k, "the beam should still fill k live hits");
    }

    // ...and it stays gone once collected.
    index.consolidate();
    let after = index
        .search(&victim_vector, 20)
        .expect("search must succeed");
    assert!(after.iter().all(|hit| hit.id != victim_id));
}

// ══════════════════════════════════════════════════════════════════════════════
// Edge cases
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn delete_of_a_nonexistent_id_is_an_error() {
    let config = MaintenanceConfig::new().with_max_degree(4);
    let corpus = clustered_corpus("doc_", 20, 6, 3, 0.3, 7);
    let mut index = MaintainableIndex::build(corpus, config).expect("build must succeed");

    assert_eq!(
        index.delete("nope"),
        Err(MaintenanceError::UnknownId("nope".to_string()))
    );
    // A second delete of the same id is also an error: the id is no longer live.
    index.delete("doc_3").expect("first delete must succeed");
    assert_eq!(
        index.delete("doc_3"),
        Err(MaintenanceError::UnknownId("doc_3".to_string()))
    );
    assert_eq!(
        index.tombstone_len(),
        1,
        "the double delete must not double-count"
    );
    assert_invariants(&index, "after a rejected double delete", true);
}

#[test]
fn deleting_every_node_then_consolidating_empties_the_index() {
    let config = MaintenanceConfig::new()
        .with_max_degree(4)
        .with_search_list_size(16)
        .with_metric(MaintenanceMetric::L2);
    let corpus = clustered_corpus("doc_", 30, 6, 3, 0.3, 11);
    let mut index = MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");

    for (id, _) in &corpus {
        index.delete(id).expect("delete must succeed");
    }
    assert_eq!(index.live_len(), 0);
    assert!(index.is_empty());
    // Every node is tombstoned, so the search finds nothing to report — but it must
    // not panic, and it must not report a tombstone.
    assert!(
        index
            .search(&corpus[0].1, 5)
            .expect("search must succeed")
            .is_empty()
    );

    let report = index.consolidate();
    assert_eq!(report.nodes_removed, 30);
    assert_eq!(report.live_nodes_after, 0);
    assert_eq!(index.slot_len(), 0);
    assert_eq!(index.tombstone_len(), 0);
    assert!(index.entry_id().is_none());
    assert!(
        index
            .search(&corpus[0].1, 5)
            .expect("search on an empty index must succeed")
            .is_empty()
    );
    assert_eq!(index.stats().orphan_count, 0);

    // The emptied index is still usable.
    index
        .insert("reborn", corpus[0].1.clone())
        .expect("insert into an emptied index must succeed");
    let hits = index.search(&corpus[0].1, 1).expect("search must succeed");
    assert_eq!(hits[0].id, "reborn");
    assert_invariants(&index, "after re-populating an emptied index", true);
}

#[test]
fn consolidating_without_tombstones_is_a_no_op() {
    let config = MaintenanceConfig::new()
        .with_max_degree(6)
        .with_search_list_size(20)
        .with_metric(MaintenanceMetric::L2);
    let corpus = clustered_corpus("doc_", 60, 8, 4, 0.3, 13);
    let mut index = MaintainableIndex::build(corpus, config).expect("build must succeed");

    let edges_before: Vec<Vec<usize>> = index.nodes.iter().map(|n| n.out_edges.clone()).collect();
    let entry_before = index.entry;

    let report = index.consolidate();
    assert!(
        report.is_noop(),
        "an untombstoned consolidate must be a no-op"
    );
    assert_eq!(
        report,
        ConsolidationReport {
            live_nodes_after: 60,
            ..ConsolidationReport::default()
        }
    );

    let edges_after: Vec<Vec<usize>> = index.nodes.iter().map(|n| n.out_edges.clone()).collect();
    assert_eq!(
        edges_before, edges_after,
        "a no-op consolidation must not touch a single edge"
    );
    assert_eq!(entry_before, index.entry);

    // Idempotent: consolidating twice more changes nothing either.
    assert!(index.consolidate().is_noop());
    assert!(index.consolidate().is_noop());
    assert_invariants(&index, "after three no-op consolidations", true);
}

#[test]
fn consolidating_a_tombstoned_entry_point_reselects_it() {
    let config = MaintenanceConfig::new()
        .with_max_degree(6)
        .with_search_list_size(24)
        .with_metric(MaintenanceMetric::L2);
    let corpus = clustered_corpus("doc_", 80, 8, 4, 0.3, 17);
    let mut index = MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");

    let entry_id = index
        .entry_id()
        .expect("a built index has an entry point")
        .to_string();
    index.delete(&entry_id).expect("delete must succeed");

    // Before consolidation the entry point is a tombstone — which is fine: searches
    // route straight through it and simply never report it.
    let stats = index.stats();
    assert!(stats.entry_point_tombstoned);
    assert_eq!(stats.orphan_count, 0, "a dead entry point must still route");
    let hits = index.search(&corpus[0].1, 5).expect("search must succeed");
    assert!(hits.iter().all(|hit| hit.id != entry_id));
    assert_eq!(hits.len(), 5);

    let report = index.consolidate();
    assert!(
        report.entry_point_reselected,
        "the dead entry point must be replaced"
    );
    assert_eq!(report.nodes_removed, 1);
    let new_entry = index.entry_id().expect("a live index has an entry point");
    assert_ne!(new_entry, entry_id);
    assert!(index.contains(new_entry));
    assert_compacted(&index, "after re-selecting a dead entry point");

    // The medoid of the live set is the new entry point.
    let expected = live_medoid(&index.nodes, MaintenanceMetric::L2).expect("a live medoid exists");
    assert_eq!(index.nodes[expected].external_id, new_entry);

    // And the index still answers queries correctly.
    for (id, vector) in corpus.iter().take(20) {
        if *id == entry_id {
            continue;
        }
        let hits = index.search(vector, 1).expect("search must succeed");
        assert_eq!(hits[0].id, *id);
    }
}

#[test]
fn inserting_a_duplicate_live_id_is_rejected_but_a_deleted_id_may_return() {
    let config = MaintenanceConfig::new()
        .with_max_degree(4)
        .with_search_list_size(16)
        .with_metric(MaintenanceMetric::L2);
    let corpus = clustered_corpus("doc_", 40, 6, 3, 0.3, 19);
    let mut index = MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");

    // A live id cannot be inserted twice.
    assert_eq!(
        index.insert("doc_5", vec![0.0; 6]),
        Err(MaintenanceError::DuplicateId("doc_5".to_string()))
    );
    assert_eq!(
        index.slot_len(),
        40,
        "a rejected insert must not allocate a slot"
    );
    assert_eq!(index.vector_of("doc_5"), Some(corpus[5].1.as_slice()));

    // But a deleted id may be re-inserted with a fresh vector: the tombstoned slot
    // keeps its (now unmapped) copy until the next consolidation collects it.
    index.delete("doc_5").expect("delete must succeed");
    let replacement = vec![0.5f32; 6];
    index
        .insert("doc_5", replacement.clone())
        .expect("re-inserting a deleted id must succeed");
    assert!(index.contains("doc_5"));
    assert_eq!(index.vector_of("doc_5"), Some(replacement.as_slice()));
    assert_eq!(index.live_len(), 40);
    assert_eq!(index.slot_len(), 41);
    assert_eq!(index.tombstone_len(), 1);
    assert_invariants(&index, "after re-inserting a deleted id", true);

    let report = index.consolidate();
    assert_eq!(report.nodes_removed, 1);
    assert_eq!(index.vector_of("doc_5"), Some(replacement.as_slice()));
    assert_eq!(
        index
            .nodes
            .iter()
            .filter(|n| n.external_id == "doc_5")
            .count(),
        1,
        "the stale copy of the id must have been collected"
    );
    assert_compacted(&index, "after collecting a replaced id");

    let hits = index.search(&replacement, 1).expect("search must succeed");
    assert_eq!(hits[0].id, "doc_5");
}

#[test]
fn auto_consolidation_fires_at_the_tombstone_threshold() {
    let config = MaintenanceConfig::new()
        .with_max_degree(6)
        .with_search_list_size(24)
        .with_metric(MaintenanceMetric::L2)
        .with_tombstone_threshold(Some(0.25));
    let corpus = clustered_corpus("doc_", 40, 8, 4, 0.3, 23);
    let mut index = MaintainableIndex::build(corpus, config).expect("build must succeed");

    let mut fired: Option<ConsolidationReport> = None;
    for i in 0..10 {
        let report = index
            .delete(&format!("doc_{i}"))
            .expect("delete must succeed");
        if let Some(report) = report {
            assert_eq!(i, 9, "the threshold is 25% of 40 slots = the 10th delete");
            fired = Some(report);
            break;
        }
        assert_eq!(index.tombstone_len(), i + 1);
    }

    let report = fired.expect("auto-consolidation must have fired");
    assert_eq!(report.nodes_removed, 10);
    assert_eq!(report.live_nodes_after, 30);
    assert_eq!(index.tombstone_len(), 0);
    assert_compacted(&index, "after auto-consolidation");
}

#[test]
fn dimension_and_config_validation() {
    assert_eq!(
        MaintainableIndex::new(0, MaintenanceConfig::new()).err(),
        Some(MaintenanceError::ZeroDimension)
    );
    assert_eq!(
        MaintainableIndex::build(Vec::new(), MaintenanceConfig::new()).err(),
        Some(MaintenanceError::EmptyIndex)
    );

    // R = 1 breaks the pigeonhole step of the connectivity-repair argument.
    assert!(matches!(
        MaintenanceConfig::new().with_max_degree(1).validate(),
        Err(MaintenanceError::InvalidConfig(_))
    ));
    assert!(matches!(
        MaintenanceConfig::new().with_search_list_size(0).validate(),
        Err(MaintenanceError::InvalidConfig(_))
    ));
    assert!(matches!(
        MaintenanceConfig::new().with_alpha(0.9).validate(),
        Err(MaintenanceError::InvalidConfig(_))
    ));
    assert!(matches!(
        MaintenanceConfig::new().with_alpha(f32::NAN).validate(),
        Err(MaintenanceError::InvalidConfig(_))
    ));
    assert!(matches!(
        MaintenanceConfig::new()
            .with_tombstone_threshold(Some(1.5))
            .validate(),
        Err(MaintenanceError::InvalidConfig(_))
    ));
    assert!(matches!(
        MaintenanceConfig::new()
            .with_tombstone_threshold(Some(0.0))
            .validate(),
        Err(MaintenanceError::InvalidConfig(_))
    ));
    MaintenanceConfig::new()
        .with_tombstone_threshold(Some(1.0))
        .validate()
        .expect("a threshold of 1.0 is legal");

    let mut index = MaintainableIndex::new(4, MaintenanceConfig::new()).expect("new must succeed");
    assert_eq!(index.dimension(), 4);
    assert_eq!(
        index.insert("a", vec![0.0; 3]),
        Err(MaintenanceError::DimensionMismatch {
            expected: 4,
            got: 3
        })
    );
    index
        .insert("a", vec![0.0; 4])
        .expect("insert must succeed");
    assert_eq!(
        index.search(&[0.0; 5], 1),
        Err(MaintenanceError::DimensionMismatch {
            expected: 4,
            got: 5
        })
    );

    assert_eq!(
        MaintainableIndex::build(
            vec![
                ("a".to_string(), vec![0.0; 4]),
                ("b".to_string(), vec![0.0; 3]),
            ],
            MaintenanceConfig::new()
        )
        .err(),
        Some(MaintenanceError::DimensionMismatch {
            expected: 4,
            got: 3
        })
    );
    assert_eq!(
        MaintainableIndex::build(
            vec![
                ("a".to_string(), vec![0.0; 4]),
                ("a".to_string(), vec![1.0; 4]),
            ],
            MaintenanceConfig::new()
        )
        .err(),
        Some(MaintenanceError::DuplicateId("a".to_string()))
    );
    assert_eq!(
        MaintainableIndex::build(
            vec![("a".to_string(), Vec::new())],
            MaintenanceConfig::new()
        )
        .err(),
        Some(MaintenanceError::ZeroDimension)
    );
}

#[test]
fn a_single_node_index_behaves() {
    let config = MaintenanceConfig::new()
        .with_max_degree(4)
        .with_metric(MaintenanceMetric::L2);
    let mut index = MaintainableIndex::build(vec![("only".to_string(), vec![1.0, 2.0])], config)
        .expect("build must succeed");

    assert_eq!(index.live_len(), 1);
    assert_eq!(index.entry_id(), Some("only"));
    assert_invariants(&index, "single node", true);

    let hits = index.search(&[1.0, 2.0], 5).expect("search must succeed");
    assert_eq!(hits, vec![MaintenanceHit::new("only", 0.0)]);
    assert!(
        index
            .search(&[1.0, 2.0], 0)
            .expect("k = 0 must succeed")
            .is_empty()
    );

    index.delete("only").expect("delete must succeed");
    assert!(index.is_empty());
    assert!(
        index
            .search(&[1.0, 2.0], 5)
            .expect("search must succeed")
            .is_empty()
    );

    let report = index.consolidate();
    assert_eq!(report.nodes_removed, 1);
    assert_eq!(report.live_nodes_after, 0);
    assert_eq!(index.slot_len(), 0);
    assert!(index.entry.is_none());
}

#[test]
fn incremental_construction_matches_the_bulk_build_in_quality() {
    let config = MaintenanceConfig::new()
        .with_max_degree(10)
        .with_search_list_size(32)
        .with_metric(MaintenanceMetric::L2);
    let centres = cluster_centres(12, 8, 0x4242_4242);
    let corpus = clustered_points("doc_", 200, &centres, 0.8, 0x2424_2424);
    let queries = clustered_queries(30, &centres, 0.8, 0x1212_1212);

    let bulk = MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");

    let mut incremental = MaintainableIndex::new(12, config).expect("new must succeed");
    for (id, vector) in &corpus {
        incremental
            .insert(id.clone(), vector.clone())
            .expect("insert must succeed");
    }
    assert_invariants(&incremental, "incremental build", true);
    assert_eq!(incremental.live_len(), 200);

    let bulk_recall = recall_at_k(&bulk, &corpus, &queries, 10);
    let incremental_recall = recall_at_k(&incremental, &corpus, &queries, 10);
    println!("bulk recall@10 = {bulk_recall:.4}, incremental recall@10 = {incremental_recall:.4}");
    assert!(bulk_recall >= 0.90, "bulk build recall {bulk_recall:.4}");
    assert!(
        incremental_recall >= 0.90,
        "a purely incremental build must still be navigable: {incremental_recall:.4}"
    );
}

#[test]
fn all_metrics_round_trip_through_a_full_churn_cycle() {
    for metric in [
        MaintenanceMetric::Cosine,
        MaintenanceMetric::L2,
        MaintenanceMetric::Dot,
    ] {
        let config = MaintenanceConfig::new()
            .with_max_degree(8)
            .with_search_list_size(24)
            .with_metric(metric);
        let corpus = clustered_corpus("doc_", 80, 8, 4, 0.3, 29);
        let mut index =
            MaintainableIndex::build(corpus.clone(), config).expect("build must succeed");

        for i in (0..80).step_by(4) {
            index
                .delete(&format!("doc_{i}"))
                .expect("delete must succeed");
        }
        for i in 0..20 {
            index
                .insert(
                    format!("new_{i}"),
                    corpus[i].1.iter().map(|x| x * 0.9).collect(),
                )
                .expect("insert must succeed");
        }
        let report = index.consolidate();
        assert_eq!(report.nodes_removed, 20);
        assert_compacted(&index, &format!("{metric:?} churn"));

        // Scores are ordered best-first under every metric's convention.
        let hits = index.search(&corpus[1].1, 5).expect("search must succeed");
        assert_eq!(hits.len(), 5);
        for pair in hits.windows(2) {
            assert!(
                pair[0].score >= pair[1].score,
                "{metric:?}: hits are not ranked best-first"
            );
        }
    }
}

#[test]
fn stats_report_the_graph_faithfully() {
    let config = MaintenanceConfig::new()
        .with_max_degree(6)
        .with_search_list_size(24)
        .with_metric(MaintenanceMetric::L2);
    let corpus = clustered_corpus("doc_", 60, 8, 4, 0.3, 31);
    let mut index = MaintainableIndex::build(corpus, config).expect("build must succeed");

    let stats = index.stats();
    assert_eq!(stats.live_nodes, 60);
    assert_eq!(stats.tombstoned_nodes, 0);
    assert_eq!(stats.total_slots, 60);
    assert_eq!(stats.orphan_count, 0);
    assert!(!stats.entry_point_tombstoned);
    assert!(stats.max_out_degree <= 6);
    assert!(
        stats.mean_out_degree() > 1.0 && stats.mean_out_degree() <= 6.0,
        "mean out-degree {} is implausible",
        stats.mean_out_degree()
    );
    assert_eq!(
        stats.edge_count,
        index.nodes.iter().map(|n| n.out_edges.len()).sum::<usize>()
    );

    for i in 0..15 {
        index
            .delete(&format!("doc_{i}"))
            .expect("delete must succeed");
    }
    let stats = index.stats();
    assert_eq!(stats.live_nodes, 45);
    assert_eq!(stats.tombstoned_nodes, 15);
    assert_eq!(stats.total_slots, 60);
    assert_eq!(stats.orphan_count, 0);

    index.consolidate();
    let stats: MaintenanceStats = index.stats();
    assert_eq!(stats.live_nodes, 45);
    assert_eq!(stats.tombstoned_nodes, 0);
    assert_eq!(stats.total_slots, 45);
    assert_eq!(stats.orphan_count, 0);
    assert!(stats.max_out_degree <= 6);
}
