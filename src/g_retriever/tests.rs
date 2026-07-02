//! Tests for the `g_retriever` module.
//!
//! The suite splits cleanly in two. The [`PcstSolver`] tests drive the
//! Prize-Collecting Steiner Tree core with *explicit* prizes and costs, so the
//! Goemans–Williamson growth and strong-pruning behaviour can be asserted
//! exactly (Steiner inclusion, expensive-edge exclusion, forests, rooting,
//! apex-away-from-root selection, determinism). The [`GRetrieverEngine`] tests
//! drive the end-to-end pipeline with the deterministic FNV-1a
//! pseudo-embeddings, exercising relevance prizes, the prize floor, weighted
//! edges, edge prizes, the size cap, error paths, and textualisation.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::uninlined_format_args
)]

use super::engine::GRetrieverEngine;
use super::pcst::PcstSolver;
use super::types::{
    GRetrieverConfig, GRetrieverEntity, GRetrieverError, GRetrieverRelation, GRetrieverRootMode,
    GRetrieverSubgraph, PcstEdge, PcstForest, PcstNode,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn nodes(prizes: &[f64]) -> Vec<PcstNode> {
    prizes
        .iter()
        .enumerate()
        .map(|(index, &prize)| PcstNode::new(index, prize))
        .collect()
}

fn edge(source: usize, target: usize, cost: f64) -> PcstEdge {
    PcstEdge::new(source, target, cost)
}

/// Solve with pruning on, unrooted, full-forest output.
fn solve_pruned(node_list: &[PcstNode], edge_list: &[PcstEdge]) -> PcstForest {
    PcstSolver::new(true, false)
        .solve(node_list, edge_list)
        .expect("valid instance")
}

// ── PcstNode / PcstEdge / PcstForest basics ───────────────────────────────────

#[test]
fn pcst_node_new_stores_fields() {
    let node = PcstNode::new(3, 4.5);
    assert_eq!(node.id, 3);
    assert_eq!(node.prize, 4.5);
}

#[test]
fn pcst_edge_other_returns_opposite_endpoint() {
    let e = edge(2, 7, 1.0);
    assert_eq!(e.other(2), Some(7));
    assert_eq!(e.other(7), Some(2));
    assert_eq!(e.other(9), None);
}

#[test]
fn pcst_forest_new_sorts_and_dedups() {
    let forest = PcstForest::new(5, vec![3, 1, 1, 2], vec![2, 0, 2]);
    assert_eq!(forest.node_indices, vec![1, 2, 3]);
    assert_eq!(forest.edge_indices, vec![0, 2]);
    assert_eq!(forest.node_count, 5);
    assert!(!forest.is_empty());
    assert_eq!(forest.len(), 3);
}

#[test]
fn pcst_forest_totals_and_net_value() {
    let node_list = nodes(&[2.0, 3.0, 5.0]);
    let edge_list = vec![edge(0, 1, 1.0), edge(1, 2, 2.0)];
    let forest = PcstForest::new(3, vec![0, 1, 2], vec![0, 1]);
    assert_eq!(forest.total_prize(&node_list), 10.0);
    assert_eq!(forest.total_cost(&edge_list), 3.0);
    assert_eq!(forest.net_value(&node_list, &edge_list), 7.0);
}

#[test]
fn pcst_forest_components_splits_disconnected() {
    // 0-1 connected; 2-3 connected; both isolated from each other.
    let edge_list = vec![edge(0, 1, 1.0), edge(2, 3, 1.0)];
    let forest = PcstForest::new(4, vec![0, 1, 2, 3], vec![0, 1]);
    let comps = forest.components(&edge_list);
    assert_eq!(comps.len(), 2);
    assert_eq!(comps[0], vec![0, 1]);
    assert_eq!(comps[1], vec![2, 3]);
}

#[test]
fn pcst_forest_components_singletons() {
    let edge_list: Vec<PcstEdge> = Vec::new();
    let forest = PcstForest::new(3, vec![0, 1, 2], vec![]);
    let comps = forest.components(&edge_list);
    assert_eq!(comps.len(), 3);
}

// ── PcstSolver: growth + strong pruning core ──────────────────────────────────

#[test]
fn single_high_prize_node_is_retrieved() {
    let forest = solve_pruned(&nodes(&[7.0]), &[]);
    assert_eq!(forest.selected_nodes(), &[0]);
    assert!(forest.selected_edges().is_empty());
}

#[test]
fn steiner_node_is_included_to_connect_terminals() {
    // Terminals 0 and 2 (prize 10) joined through a zero-prize waypoint 1.
    // In single-component mode a connected result is required, so the Steiner
    // waypoint MUST be pulled in to link the two terminals.
    let node_list = nodes(&[10.0, 0.0, 10.0]);
    let edge_list = vec![edge(0, 1, 1.0), edge(1, 2, 1.0)];
    let forest = PcstSolver::new(true, true)
        .solve(&node_list, &edge_list)
        .expect("valid instance");
    assert_eq!(forest.selected_nodes(), &[0, 1, 2]);
    assert_eq!(forest.selected_edges().len(), 2);
    assert_eq!(forest.components(&edge_list).len(), 1);
    // Net value = 20 collected - 2 cost = 18.
    assert!((forest.net_value(&node_list, &edge_list) - 18.0).abs() < 1e-9);
}

#[test]
fn edge_costlier_than_prize_is_excluded() {
    // Node 1's prize (3) does not pay for the edge cost (5) to reach it.
    let node_list = nodes(&[10.0, 3.0]);
    let edge_list = vec![edge(0, 1, 5.0)];
    let forest = solve_pruned(&node_list, &edge_list);
    assert_eq!(forest.selected_nodes(), &[0]);
    assert!(forest.selected_edges().is_empty());
}

#[test]
fn edge_cheaper_than_prize_is_included() {
    // With a cheap edge (2 < 3), node 1 is worth reaching.
    let node_list = nodes(&[10.0, 3.0]);
    let edge_list = vec![edge(0, 1, 2.0)];
    let forest = solve_pruned(&node_list, &edge_list);
    assert_eq!(forest.selected_nodes(), &[0, 1]);
    assert_eq!(forest.selected_edges(), &[0]);
}

#[test]
fn disconnected_terminals_with_no_affordable_path_return_a_forest() {
    // Two high-prize nodes joined only by a prohibitively expensive edge.
    let node_list = nodes(&[10.0, 10.0]);
    let edge_list = vec![edge(0, 1, 100.0)];
    let forest = solve_pruned(&node_list, &edge_list);
    assert_eq!(forest.selected_nodes(), &[0, 1]);
    assert!(forest.selected_edges().is_empty());
    assert_eq!(forest.components(&edge_list).len(), 2);
}

#[test]
fn zero_prize_isolated_nodes_are_dropped() {
    let node_list = nodes(&[5.0, 0.0, 0.0]);
    let forest = solve_pruned(&node_list, &[]);
    assert_eq!(forest.selected_nodes(), &[0]);
}

#[test]
fn negative_prize_is_clamped_and_dropped() {
    let node_list = nodes(&[5.0, -3.0]);
    let edge_list = vec![edge(0, 1, 0.5)];
    let forest = solve_pruned(&node_list, &edge_list);
    // Node 1 contributes no prize, so paying 0.5 to reach it is not worth it.
    assert_eq!(forest.selected_nodes(), &[0]);
}

#[test]
fn apex_can_lie_away_from_the_component_root() {
    // A single growth tree 0 -(5)- 1 -(1)- 2, prizes 1, 10, 10. Strong pruning
    // roots the tree at node 0, yet the optimal connected subtree is {1, 2}
    // (net 19) whose apex is the interior node 1, not the rooting root: node 0
    // does not pay for the cost-5 edge that would attach it.
    let node_list = nodes(&[1.0, 10.0, 10.0]);
    let edge_list = vec![edge(0, 1, 5.0), edge(1, 2, 1.0)];
    let forest = solve_pruned(&node_list, &edge_list);
    assert_eq!(forest.selected_nodes(), &[1, 2]);
    assert_eq!(forest.selected_edges(), &[1]);
    assert!((forest.net_value(&node_list, &edge_list) - 19.0).abs() < 1e-9);
}

#[test]
fn star_hub_keeps_relevant_spokes_and_drops_irrelevant_ones() {
    // Hub 0 (prize 6) with three spokes: 1 (prize 5, cheap), 2 (prize 0),
    // 3 (prize 4 but expensive edge 10).
    let node_list = nodes(&[6.0, 5.0, 0.0, 4.0]);
    let edge_list = vec![edge(0, 1, 1.0), edge(0, 2, 1.0), edge(0, 3, 10.0)];
    let forest = solve_pruned(&node_list, &edge_list);
    assert_eq!(forest.selected_nodes(), &[0, 1]);
    assert_eq!(forest.selected_edges(), &[0]);
}

#[test]
fn tie_broken_apex_is_smallest_id() {
    // Two isolated equal-prize nodes: each is its own singleton component.
    let node_list = nodes(&[4.0, 4.0]);
    let forest = solve_pruned(&node_list, &[]);
    assert_eq!(forest.selected_nodes(), &[0, 1]);
}

// ── pruning on/off and component selection ────────────────────────────────────

#[test]
fn pruning_disabled_returns_raw_growth_forest() {
    // With pruning off, the growth forest keeps the merge edges even though
    // strong pruning would trim the expensive one.
    let node_list = nodes(&[10.0, 3.0]);
    let edge_list = vec![edge(0, 1, 5.0)];
    let forest = PcstSolver::new(false, false)
        .solve(&node_list, &edge_list)
        .expect("valid");
    assert_eq!(forest.selected_edges(), &[0]);
    assert!(forest.selected_nodes().contains(&0));
    assert!(forest.selected_nodes().contains(&1));
}

#[test]
fn full_forest_keeps_all_positive_components() {
    // Component {0,1} (net 19) and singleton {2} (net 5).
    let node_list = nodes(&[10.0, 10.0, 5.0]);
    let edge_list = vec![edge(0, 1, 1.0)];
    let forest = PcstSolver::new(true, false)
        .solve(&node_list, &edge_list)
        .expect("valid");
    assert_eq!(forest.selected_nodes(), &[0, 1, 2]);
    assert_eq!(forest.components(&edge_list).len(), 2);
}

#[test]
fn single_component_keeps_only_best_component() {
    let node_list = nodes(&[10.0, 10.0, 5.0]);
    let edge_list = vec![edge(0, 1, 1.0)];
    let forest = PcstSolver::new(true, true)
        .solve(&node_list, &edge_list)
        .expect("valid");
    assert_eq!(forest.selected_nodes(), &[0, 1]);
    assert_eq!(forest.components(&edge_list).len(), 1);
}

#[test]
fn single_component_tie_prefers_smallest_min_id() {
    // Two equal-net components {0,1} and {2,3}; the lower-id one wins.
    let node_list = nodes(&[10.0, 10.0, 10.0, 10.0]);
    let edge_list = vec![edge(0, 1, 1.0), edge(2, 3, 1.0)];
    let forest = PcstSolver::new(true, true)
        .solve(&node_list, &edge_list)
        .expect("valid");
    assert_eq!(forest.selected_nodes(), &[0, 1]);
}

// ── rooted PCST ───────────────────────────────────────────────────────────────

#[test]
fn rooted_solve_keeps_root_even_when_prize_is_zero() {
    // Node 2 is an isolated zero-prize node; component {0,1} has net 19.
    let node_list = nodes(&[10.0, 10.0, 0.0]);
    let edge_list = vec![edge(0, 1, 1.0)];
    let forest = PcstSolver::new(true, false)
        .with_root(Some(2))
        .solve(&node_list, &edge_list)
        .expect("valid");
    // Rooted mode returns the single tree containing the root (node 2 only).
    assert_eq!(forest.selected_nodes(), &[2]);
}

#[test]
fn rooted_solve_keeps_relevant_and_prunes_expensive_dead_end() {
    // 0(root, prize 0) -(1)- 1(prize 10) -(100)- 2(prize 0).
    let node_list = nodes(&[0.0, 10.0, 0.0]);
    let edge_list = vec![edge(0, 1, 1.0), edge(1, 2, 100.0)];
    let forest = PcstSolver::new(true, false)
        .with_root(Some(0))
        .solve(&node_list, &edge_list)
        .expect("valid");
    assert_eq!(forest.selected_nodes(), &[0, 1]);
    assert_eq!(forest.selected_edges(), &[0]);
}

#[test]
fn rooted_out_of_range_root_is_an_error() {
    let node_list = nodes(&[1.0, 2.0]);
    let result = PcstSolver::new(true, false)
        .with_root(Some(9))
        .solve(&node_list, &[]);
    assert!(matches!(result, Err(GRetrieverError::InvalidGraph { .. })));
}

// ── determinism and monotonicity ──────────────────────────────────────────────

#[test]
fn solver_is_deterministic() {
    let node_list = nodes(&[10.0, 0.0, 8.0, 3.0, 6.0]);
    let edge_list = vec![
        edge(0, 1, 1.0),
        edge(1, 2, 1.0),
        edge(2, 3, 5.0),
        edge(0, 4, 2.0),
    ];
    let first = solve_pruned(&node_list, &edge_list);
    let second = solve_pruned(&node_list, &edge_list);
    assert_eq!(first, second);
}

#[test]
fn higher_edge_cost_weakly_shrinks_the_subgraph() {
    // A -(cost)- B, prizes 10 and 4, in single-component (one connected
    // subgraph) mode. B is worth attaching exactly while the connecting cost
    // stays below its prize; past that the best connected subgraph is A alone.
    // The retrieved size is monotone non-increasing in the edge cost.
    let node_list = nodes(&[10.0, 4.0]);
    let count_for = |cost: f64| -> usize {
        PcstSolver::new(true, true)
            .solve(&node_list, &[edge(0, 1, cost)])
            .expect("valid")
            .selected_nodes()
            .len()
    };
    assert_eq!(count_for(1.0), 2);
    assert_eq!(count_for(3.0), 2);
    assert_eq!(count_for(5.0), 1);
    assert_eq!(count_for(100.0), 1);
    assert!(count_for(5.0) <= count_for(3.0));
    assert!(count_for(3.0) <= count_for(1.0));
}

// ── solver validation errors ──────────────────────────────────────────────────

#[test]
fn solver_rejects_edge_endpoint_out_of_range() {
    let node_list = nodes(&[1.0, 2.0]);
    let edge_list = vec![edge(0, 5, 1.0)];
    let result = PcstSolver::new(true, false).solve(&node_list, &edge_list);
    assert!(matches!(result, Err(GRetrieverError::InvalidGraph { .. })));
}

#[test]
fn solver_rejects_non_finite_prize() {
    let node_list = vec![PcstNode::new(0, f64::NAN)];
    let result = PcstSolver::new(true, false).solve(&node_list, &[]);
    assert!(matches!(result, Err(GRetrieverError::InvalidGraph { .. })));
}

#[test]
fn solver_rejects_non_finite_cost() {
    let node_list = nodes(&[1.0, 2.0]);
    let edge_list = vec![edge(0, 1, f64::INFINITY)];
    let result = PcstSolver::new(true, false).solve(&node_list, &edge_list);
    assert!(matches!(result, Err(GRetrieverError::InvalidGraph { .. })));
}

#[test]
fn grow_phase_produces_merge_edges() {
    let node_list = nodes(&[10.0, 10.0]);
    let edge_list = vec![edge(0, 1, 1.0)];
    let growth = PcstSolver::new(true, false).grow(&node_list, &edge_list);
    assert_eq!(growth.selected_edges(), &[0]);
}

// ── large-ish sanity ──────────────────────────────────────────────────────────

#[test]
fn large_chain_terminates_with_sane_result() {
    let count = 40;
    let prize_list: Vec<f64> = (0..count)
        .map(|i| if i % 2 == 0 { 2.0 } else { 0.0 })
        .collect();
    let node_list = nodes(&prize_list);
    let edge_list: Vec<PcstEdge> = (0..count - 1).map(|i| edge(i, i + 1, 0.5)).collect();
    let forest = solve_pruned(&node_list, &edge_list);

    assert!(!forest.is_empty());
    assert!(forest.net_value(&node_list, &edge_list) >= -1e-9);
    // Every kept edge connects two kept nodes.
    let kept: std::collections::HashSet<usize> = forest.selected_nodes().iter().copied().collect();
    for &ei in forest.selected_edges() {
        let e = edge_list[ei];
        assert!(kept.contains(&e.source) && kept.contains(&e.target));
    }
    // Determinism on the large instance.
    assert_eq!(forest, solve_pruned(&node_list, &edge_list));
}

#[test]
fn large_grid_is_connected_when_edges_are_cheap() {
    // 5x5 grid, all nodes relevant, very cheap edges: expect one component.
    let side = 5;
    let node_list = nodes(&vec![3.0; side * side]);
    let mut edge_list = Vec::new();
    for row in 0..side {
        for col in 0..side {
            let id = row * side + col;
            if col + 1 < side {
                edge_list.push(edge(id, id + 1, 0.1));
            }
            if row + 1 < side {
                edge_list.push(edge(id, id + side, 0.1));
            }
        }
    }
    let forest = PcstSolver::new(true, true)
        .solve(&node_list, &edge_list)
        .expect("valid");
    assert_eq!(forest.selected_nodes().len(), side * side);
    assert_eq!(forest.components(&edge_list).len(), 1);
}

// ── GRetrieverConfig ──────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = GRetrieverConfig::default();
    assert_eq!(config.edge_cost, 1.0);
    assert_eq!(config.edge_cost_scale, 1.0);
    assert_eq!(config.prize_floor, 0.1);
    assert_eq!(config.prize_scale, 1.0);
    assert_eq!(config.embed_dim, 64);
    assert!(!config.edge_prizes);
    assert!(config.prune);
    assert!(!config.single_component);
    assert_eq!(config.root_mode, GRetrieverRootMode::Unrooted);
    assert_eq!(config.max_subgraph_size, None);
}

#[test]
fn config_builders_chain() {
    let config = GRetrieverConfig::new()
        .with_edge_cost(2.0)
        .with_edge_cost_scale(1.5)
        .with_prize_floor(0.2)
        .with_prize_scale(3.0)
        .with_embed_dim(128)
        .with_edge_prizes(true)
        .with_prune(false)
        .with_single_component(true)
        .with_root_mode(GRetrieverRootMode::Rooted {
            entity_id: "r".to_string(),
        })
        .with_max_subgraph_size(Some(4));
    assert_eq!(config.edge_cost, 2.0);
    assert_eq!(config.edge_cost_scale, 1.5);
    assert_eq!(config.prize_floor, 0.2);
    assert_eq!(config.prize_scale, 3.0);
    assert_eq!(config.embed_dim, 128);
    assert!(config.edge_prizes);
    assert!(!config.prune);
    assert!(config.single_component);
    assert_eq!(config.max_subgraph_size, Some(4));
}

#[test]
fn config_validate_accepts_default() {
    assert!(GRetrieverConfig::default().validate().is_ok());
}

#[test]
fn config_validate_rejects_non_positive_edge_cost() {
    let config = GRetrieverConfig::default().with_edge_cost(0.0);
    assert!(matches!(
        config.validate(),
        Err(GRetrieverError::InvalidConfig { .. })
    ));
}

#[test]
fn config_validate_rejects_out_of_range_floor() {
    let config = GRetrieverConfig::default().with_prize_floor(1.5);
    assert!(matches!(
        config.validate(),
        Err(GRetrieverError::InvalidConfig { .. })
    ));
}

#[test]
fn config_validate_rejects_zero_embed_dim() {
    let config = GRetrieverConfig::default().with_embed_dim(0);
    assert!(matches!(
        config.validate(),
        Err(GRetrieverError::InvalidConfig { .. })
    ));
}

#[test]
fn config_validate_rejects_zero_max_size() {
    let config = GRetrieverConfig::default().with_max_subgraph_size(Some(0));
    assert!(matches!(
        config.validate(),
        Err(GRetrieverError::InvalidConfig { .. })
    ));
}

// ── engine helpers ────────────────────────────────────────────────────────────

fn entity(id: &str, text: &str) -> GRetrieverEntity {
    GRetrieverEntity::new(id, text)
}

fn relation(source: &str, target: &str, label: &str) -> GRetrieverRelation {
    GRetrieverRelation::new(source, target, label)
}

// ── GRetrieverEngine: error paths ─────────────────────────────────────────────

#[test]
fn engine_empty_query_is_an_error() {
    let engine = GRetrieverEngine::default();
    let entities = vec![entity("a", "alpha")];
    let result = engine.retrieve("   ", &entities, &[]);
    assert!(matches!(result, Err(GRetrieverError::EmptyQuery)));
}

#[test]
fn engine_empty_graph_is_an_error() {
    let engine = GRetrieverEngine::default();
    let result = engine.retrieve("alpha", &[], &[]);
    assert!(matches!(result, Err(GRetrieverError::EmptyGraph)));
}

#[test]
fn engine_no_relevant_node_is_an_error() {
    // With the prize floor at its maximum, only an entity whose text is
    // identical to the query survives; these entities differ from it, so every
    // prize is zeroed and there is no relevant node to anchor on.
    let config = GRetrieverConfig::default().with_prize_floor(1.0);
    let engine = GRetrieverEngine::new(config);
    let entities = vec![
        entity("a", "photosynthesis chloroplast"),
        entity("b", "volcanic basalt geology"),
    ];
    let result = engine.retrieve("quarterly financial derivatives", &entities, &[]);
    assert!(matches!(result, Err(GRetrieverError::NoRelevantNode)));
}

#[test]
fn engine_dangling_relation_is_an_error() {
    let engine = GRetrieverEngine::default();
    let entities = vec![entity("a", "alpha topic")];
    let relations = vec![relation("a", "ghost", "links to")];
    let result = engine.retrieve("alpha topic", &entities, &relations);
    match result {
        Err(GRetrieverError::DanglingRelation { index, endpoint }) => {
            assert_eq!(index, 0);
            assert_eq!(endpoint, "ghost");
        }
        other => panic!("expected DanglingRelation, got {other:?}"),
    }
}

#[test]
fn engine_non_positive_weight_is_an_error() {
    let engine = GRetrieverEngine::default();
    let entities = vec![entity("a", "alpha topic"), entity("b", "alpha topic")];
    let relations = vec![relation("a", "b", "links").with_weight(0.0)];
    let result = engine.retrieve("alpha topic", &entities, &relations);
    assert!(matches!(result, Err(GRetrieverError::InvalidGraph { .. })));
}

#[test]
fn engine_root_not_found_is_an_error() {
    let config = GRetrieverConfig::default().with_root_mode(GRetrieverRootMode::Rooted {
        entity_id: "missing".to_string(),
    });
    let engine = GRetrieverEngine::new(config);
    let entities = vec![entity("a", "alpha topic")];
    let result = engine.retrieve("alpha topic", &entities, &[]);
    match result {
        Err(GRetrieverError::RootNotFound { entity_id }) => assert_eq!(entity_id, "missing"),
        other => panic!("expected RootNotFound, got {other:?}"),
    }
}

#[test]
fn engine_invalid_config_propagates() {
    let engine = GRetrieverEngine::new(GRetrieverConfig::default().with_edge_cost(-1.0));
    let entities = vec![entity("a", "alpha topic")];
    let result = engine.retrieve("alpha topic", &entities, &[]);
    assert!(matches!(result, Err(GRetrieverError::InvalidConfig { .. })));
}

// ── GRetrieverEngine: retrieval behaviour ─────────────────────────────────────

#[test]
fn engine_retrieves_relevant_entity() {
    let query = "graph neural network question answering";
    let entities = vec![
        entity("gnn", query),
        entity("task", "graph question answering"),
        entity("fruit", "banana smoothie recipe"),
    ];
    let relations = vec![
        relation("gnn", "task", "used for"),
        relation("task", "fruit", "unrelated to"),
    ];
    let engine = GRetrieverEngine::default();
    let subgraph = engine.retrieve(query, &entities, &relations).expect("ok");
    assert!(subgraph.contains_entity("gnn"));
    assert!(!subgraph.contains_entity("fruit"));
}

#[test]
fn engine_prizes_rank_on_topic_above_off_topic() {
    let query = "quantum error correction codes";
    let entities = vec![
        entity("hit", query),
        entity("miss", "medieval european cathedral architecture"),
    ];
    let engine = GRetrieverEngine::new(GRetrieverConfig::default().with_prize_floor(0.0));
    let subgraph = engine.retrieve(query, &entities, &[]).expect("ok");
    // The on-topic entity is present and carries a strictly positive prize.
    let hit_prize = subgraph
        .entities
        .iter()
        .position(|e| e.id == "hit")
        .map(|i| subgraph.node_prizes[i])
        .expect("hit present");
    assert!(hit_prize > 0.0);
}

#[test]
fn engine_prize_floor_filters_partial_matches() {
    let query = "reinforcement learning policy gradient";
    // `exact` matches the query verbatim (cosine 1.0); `partial` shares only
    // some content, so a floor of 1.0 zeroes its prize.
    let entities = vec![
        entity("exact", query),
        entity(
            "partial",
            "reinforcement learning policy gradient methods for robotics control",
        ),
    ];
    let relations = vec![relation("exact", "partial", "extends")];

    let low_floor = GRetrieverConfig::default()
        .with_prize_floor(0.0)
        .with_prize_scale(10.0)
        .with_edge_cost(0.1);
    let included = GRetrieverEngine::new(low_floor)
        .retrieve(query, &entities, &relations)
        .expect("ok");
    assert!(included.contains_entity("partial"));

    let high_floor = GRetrieverConfig::default().with_prize_floor(1.0);
    let filtered = GRetrieverEngine::new(high_floor)
        .retrieve(query, &entities, &relations)
        .expect("ok");
    assert!(filtered.contains_entity("exact"));
    assert!(!filtered.contains_entity("partial"));
}

#[test]
fn engine_is_deterministic() {
    let query = "distributed consensus protocol";
    let entities = vec![
        entity("a", query),
        entity("b", "distributed consensus protocol variant"),
        entity("c", "irrelevant cooking topic"),
    ];
    let relations = vec![
        relation("a", "b", "related"),
        relation("b", "c", "unrelated"),
    ];
    let engine = GRetrieverEngine::default();
    let first = engine.retrieve(query, &entities, &relations).expect("ok");
    let second = engine.retrieve(query, &entities, &relations).expect("ok");
    assert_eq!(first, second);
}

#[test]
fn engine_textualize_lists_nodes_and_edges() {
    let query = "supernova nucleosynthesis";
    let entities = vec![
        entity("a", query),
        entity("b", "supernova nucleosynthesis yields"),
    ];
    let relations = vec![relation("a", "b", "produces")];
    let config = GRetrieverConfig::default()
        .with_prize_scale(10.0)
        .with_edge_cost(0.1);
    let subgraph = GRetrieverEngine::new(config)
        .retrieve(query, &entities, &relations)
        .expect("ok");
    let text = subgraph.textualize();
    assert!(text.contains("Nodes:"));
    assert!(text.contains("Edges:"));
    assert!(text.contains("a: supernova nucleosynthesis"));
}

#[test]
fn engine_summary_is_internally_consistent() {
    let query = "photonic integrated circuits";
    let entities = vec![
        entity("a", query),
        entity("b", "photonic integrated circuits design"),
    ];
    let relations = vec![relation("a", "b", "relates")];
    let config = GRetrieverConfig::default()
        .with_prize_scale(10.0)
        .with_edge_cost(0.1);
    let subgraph = GRetrieverEngine::new(config)
        .retrieve(query, &entities, &relations)
        .expect("ok");
    // node_prizes align with entities; net value is prize minus cost.
    assert_eq!(subgraph.node_prizes.len(), subgraph.entities.len());
    let node_prize_sum: f64 = subgraph.node_prizes.iter().sum();
    assert!(subgraph.total_prize >= node_prize_sum - 1e-9);
    assert!((subgraph.net_value() - (subgraph.total_prize - subgraph.total_cost)).abs() < 1e-9);
    assert!(subgraph.num_components >= 1);
}

#[test]
fn engine_weighted_edge_can_price_a_connection_out() {
    let query = "spectral graph theory";
    let entities = vec![
        entity("a", query),
        entity("b", "spectral graph theory eigenvalues"),
    ];
    let config = GRetrieverConfig::default()
        .with_prize_scale(1.0)
        .with_edge_cost(1.0);

    // Cheap weight: the connection is affordable and both nodes are retrieved.
    let cheap = vec![relation("a", "b", "relates").with_weight(0.1)];
    let cheap_graph = GRetrieverEngine::new(config.clone())
        .retrieve(query, &entities, &cheap)
        .expect("ok");
    assert!(cheap_graph.contains_entity("a"));
    assert!(cheap_graph.contains_entity("b"));
    assert_eq!(cheap_graph.relations.len(), 1);

    // Prohibitive weight: the edge is dropped (the nodes may still stand alone).
    let dear = vec![relation("a", "b", "relates").with_weight(1000.0)];
    let dear_graph = GRetrieverEngine::new(config)
        .retrieve(query, &entities, &dear)
        .expect("ok");
    assert!(dear_graph.relations.is_empty());
}

#[test]
fn engine_edge_prizes_pull_in_a_relevant_relation() {
    // Endpoints carry no query relevance, but the relation label does; with
    // edge prizes enabled the relation (and its endpoints) can be retrieved.
    let query = "collaborated with";
    let entities = vec![
        entity("a", "person one placeholder"),
        entity("b", "person two placeholder"),
    ];
    let relations = vec![relation("a", "b", query)];
    let config = GRetrieverConfig::default()
        .with_edge_prizes(true)
        .with_prize_scale(20.0)
        .with_edge_cost(0.1);
    let subgraph = GRetrieverEngine::new(config)
        .retrieve(query, &entities, &relations)
        .expect("ok");
    assert_eq!(subgraph.relations.len(), 1);
    assert!(subgraph.contains_entity("a"));
    assert!(subgraph.contains_entity("b"));
    assert!(subgraph.total_prize > 0.0);
}

#[test]
fn engine_size_cap_trims_the_subgraph() {
    let query = "topic";
    // A star of five relevant nodes; cap the result to three entities.
    let entities = vec![
        entity("hub", "topic"),
        entity("s1", "topic one"),
        entity("s2", "topic two"),
        entity("s3", "topic three"),
        entity("s4", "topic four"),
    ];
    let relations = vec![
        relation("hub", "s1", "topic"),
        relation("hub", "s2", "topic"),
        relation("hub", "s3", "topic"),
        relation("hub", "s4", "topic"),
    ];
    let config = GRetrieverConfig::default()
        .with_prize_scale(10.0)
        .with_edge_cost(0.1)
        .with_max_subgraph_size(Some(3));
    let subgraph = GRetrieverEngine::new(config)
        .retrieve(query, &entities, &relations)
        .expect("ok");
    assert!(subgraph.len() <= 3);
}

#[test]
fn engine_rooted_retrieval_contains_the_root() {
    let query = "molecular dynamics simulation";
    let entities = vec![
        entity("root", "unrelated anchoring node"),
        entity("hit", query),
    ];
    let relations = vec![relation("root", "hit", "connects")];
    let config = GRetrieverConfig::default()
        .with_prize_scale(10.0)
        .with_edge_cost(0.1)
        .with_root_mode(GRetrieverRootMode::Rooted {
            entity_id: "root".to_string(),
        });
    let subgraph = GRetrieverEngine::new(config)
        .retrieve(query, &entities, &relations)
        .expect("ok");
    assert!(subgraph.contains_entity("root"));
}

#[test]
fn engine_single_component_returns_one_component() {
    let query = "topic alpha";
    // Two separate relevant components.
    let entities = vec![
        entity("a", "topic alpha"),
        entity("b", "topic alpha beta"),
        entity("c", "topic alpha gamma"),
        entity("d", "topic alpha delta"),
    ];
    let relations = vec![
        relation("a", "b", "topic alpha"),
        relation("c", "d", "topic alpha"),
    ];
    let config = GRetrieverConfig::default()
        .with_prize_scale(10.0)
        .with_edge_cost(0.1)
        .with_single_component(true);
    let subgraph = GRetrieverEngine::new(config)
        .retrieve(query, &entities, &relations)
        .expect("ok");
    assert_eq!(subgraph.num_components, 1);
    assert!(subgraph.is_connected());
}

#[test]
fn engine_subgraph_roundtrips_through_a_temp_file() {
    // Exercises the textualisation as a downstream artifact via a temp file.
    let query = "isotope decay chain";
    let entities = vec![entity("a", query), entity("b", "isotope decay chain step")];
    let relations = vec![relation("a", "b", "leads to")];
    let config = GRetrieverConfig::default()
        .with_prize_scale(10.0)
        .with_edge_cost(0.1);
    let subgraph: GRetrieverSubgraph = GRetrieverEngine::new(config)
        .retrieve(query, &entities, &relations)
        .expect("ok");

    let mut path = std::env::temp_dir();
    path.push(format!("oxirag_g_retriever_{}.txt", std::process::id()));
    std::fs::write(&path, subgraph.textualize()).expect("write");
    let read_back = std::fs::read_to_string(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert!(read_back.contains("Nodes:"));
    assert!(read_back.contains("a: isotope decay chain"));
}
