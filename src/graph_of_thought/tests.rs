//! Tests for the `graph_of_thought` module.

#![allow(clippy::float_cmp, clippy::similar_names)]

use super::engine::GraphOfThoughtEngine;
use super::graph::{Thought, ThoughtGraph};
use super::types::{
    GotConfig, GotError, GotOperation, GotOutput, MockThoughtAggregator, MockThoughtGenerator,
    MockThoughtScorer, ThoughtAggregator, ThoughtGenerator, ThoughtScorer,
};

// ── Helpers ──────────────────────────────────────────────────────────────────────

/// A scorer that returns a fixed score per exact thought content, falling back
/// to `default` when no mapping matches. Enables precise control of best/tie
/// behaviour in tests.
struct MapScorer {
    map: Vec<(String, f32)>,
    default: f32,
}

impl MapScorer {
    fn new(map: &[(&str, f32)], default: f32) -> Self {
        Self {
            map: map.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
            default,
        }
    }
}

impl ThoughtScorer for MapScorer {
    fn score(&self, _query: &str, thought: &str) -> f32 {
        for (content, score) in &self.map {
            if content == thought {
                return *score;
            }
        }
        self.default
    }
}

/// A generator that yields a fixed list of contents, ignoring `k`.
struct FixedGenerator {
    items: Vec<String>,
}

impl FixedGenerator {
    fn new(items: &[&str]) -> Self {
        Self {
            items: items.iter().map(|s| (*s).to_string()).collect(),
        }
    }
}

impl ThoughtGenerator for FixedGenerator {
    fn generate(&self, _query: &str, _context: &[&str], _k: usize) -> Vec<String> {
        self.items.clone()
    }
}

fn default_engine() -> GraphOfThoughtEngine {
    GraphOfThoughtEngine::new(GotConfig::default())
}

// ── GotConfig defaults & builders ──────────────────────────────────────────────────

#[test]
fn test_config_default_max_nodes() {
    assert_eq!(GotConfig::default().max_nodes, 32);
}

#[test]
fn test_config_default_branch_factor() {
    assert_eq!(GotConfig::default().branch_factor, 3);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(GotConfig::new(), GotConfig::default());
}

#[test]
fn test_config_with_max_nodes() {
    assert_eq!(GotConfig::new().with_max_nodes(8).max_nodes, 8);
}

#[test]
fn test_config_with_branch_factor() {
    assert_eq!(GotConfig::new().with_branch_factor(5).branch_factor, 5);
}

// ── GotOperation ───────────────────────────────────────────────────────────────────

#[test]
fn test_operation_as_str_generate() {
    assert_eq!(GotOperation::Generate { k: 3 }.as_str(), "generate");
}

#[test]
fn test_operation_as_str_aggregate() {
    assert_eq!(GotOperation::Aggregate.as_str(), "aggregate");
}

#[test]
fn test_operation_as_str_refine() {
    assert_eq!(GotOperation::Refine.as_str(), "refine");
}

#[test]
fn test_operation_as_str_keep_best() {
    assert_eq!(GotOperation::KeepBest { n: 2 }.as_str(), "keep_best");
}

// ── Thought ───────────────────────────────────────────────────────────────────────

#[test]
fn test_thought_new() {
    let t = Thought::new(2, "hello", 0.5);
    assert_eq!(t.id, 2);
    assert_eq!(t.content, "hello");
    assert_eq!(t.score, 0.5);
}

// ── ThoughtGraph: add_node / node / len / is_empty ─────────────────────────────────

#[test]
fn test_graph_new_is_empty() {
    let graph = ThoughtGraph::new();
    assert!(graph.is_empty());
    assert_eq!(graph.len(), 0);
    assert!(graph.best().is_none());
}

#[test]
fn test_graph_add_node_returns_sequential_ids() {
    let mut graph = ThoughtGraph::new();
    let a = graph.add_node("a", 0.1);
    let b = graph.add_node("b", 0.2);
    let c = graph.add_node("c", 0.3);
    assert_eq!((a, b, c), (0, 1, 2));
    assert_eq!(graph.len(), 3);
    assert!(!graph.is_empty());
}

#[test]
fn test_graph_node_content_and_score() {
    let mut graph = ThoughtGraph::new();
    let id = graph.add_node("thinking", 0.7);
    let node = graph.node(id).expect("node should exist");
    assert_eq!(node.id, id);
    assert_eq!(node.content, "thinking");
    assert_eq!(node.score, 0.7);
}

// ── ThoughtGraph: add_edge / parents_of / children_of ──────────────────────────────

#[test]
fn test_graph_add_edge_and_parents_of() {
    let mut graph = ThoughtGraph::new();
    let a = graph.add_node("a", 0.1);
    let b = graph.add_node("b", 0.2);
    let c = graph.add_node("c", 0.3);
    graph.add_edge(a, c);
    graph.add_edge(b, c);
    let mut parents = graph.parents_of(c);
    parents.sort_unstable();
    assert_eq!(parents, vec![a, b]);
}

#[test]
fn test_graph_children_of() {
    let mut graph = ThoughtGraph::new();
    let a = graph.add_node("a", 0.1);
    let b = graph.add_node("b", 0.2);
    let c = graph.add_node("c", 0.3);
    graph.add_edge(a, b);
    graph.add_edge(a, c);
    assert_eq!(graph.children_of(a), vec![b, c]);
}

#[test]
fn test_graph_parents_of_excludes_self_edge() {
    let mut graph = ThoughtGraph::new();
    let a = graph.add_node("a", 0.1);
    graph.add_edge(a, a);
    assert!(graph.parents_of(a).is_empty());
    assert!(graph.children_of(a).is_empty());
    assert_eq!(graph.edge_count(), 1);
}

#[test]
fn test_graph_add_edge_out_of_range_ignored() {
    let mut graph = ThoughtGraph::new();
    let a = graph.add_node("a", 0.1);
    graph.add_edge(a, 99);
    graph.add_edge(99, a);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_graph_edges_accessor() {
    let mut graph = ThoughtGraph::new();
    let a = graph.add_node("a", 0.1);
    let b = graph.add_node("b", 0.2);
    graph.add_edge(a, b);
    assert_eq!(graph.edges(), &[(a, b)]);
}

// ── ThoughtGraph: best / tie-break ─────────────────────────────────────────────────

#[test]
fn test_graph_best_max_score() {
    let mut graph = ThoughtGraph::new();
    graph.add_node("low", 0.2);
    graph.add_node("high", 0.9);
    graph.add_node("mid", 0.5);
    assert_eq!(graph.best().expect("best").content, "high");
}

#[test]
fn test_graph_best_tie_break_lowest_id() {
    let mut graph = ThoughtGraph::new();
    let first = graph.add_node("first", 0.8);
    graph.add_node("second", 0.8);
    let best = graph.best().expect("best");
    assert_eq!(best.id, first);
    assert_eq!(best.content, "first");
}

// ── Generate operation ─────────────────────────────────────────────────────────────

#[test]
fn test_generate_branches_k_nodes() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 4 }];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert_eq!(out.graph.len(), 4);
}

#[test]
fn test_generate_from_empty_frontier_has_no_parents() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    for node in out.graph.nodes() {
        assert!(out.graph.parents_of(node.id).is_empty());
    }
    assert_eq!(out.graph.edge_count(), 0);
}

#[test]
fn test_generate_then_generate_branches_each_parent() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    // First gen: 2 nodes. Second gen: each of the 2 branches 2 -> 4 children.
    let plan = [
        GotOperation::Generate { k: 2 },
        GotOperation::Generate { k: 2 },
    ];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert_eq!(out.graph.len(), 6);
    // The 4 children each have exactly one parent.
    let children: Vec<_> = out
        .graph
        .nodes()
        .iter()
        .filter(|t| !out.graph.parents_of(t.id).is_empty())
        .collect();
    assert_eq!(children.len(), 4);
    for child in children {
        assert_eq!(out.graph.parents_of(child.id).len(), 1);
    }
}

#[test]
fn test_generate_k_zero_uses_branch_factor() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let engine = GraphOfThoughtEngine::new(GotConfig::new().with_branch_factor(5));
    let plan = [GotOperation::Generate { k: 0 }];
    let out = engine
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert_eq!(out.graph.len(), 5);
}

#[test]
fn test_generate_scores_applied() {
    // MockThoughtScorer: score = 0.1 * trailing_number, so "t 3" => 0.3.
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    let best = out.graph.best().expect("best");
    assert_eq!(best.content, "t 3");
    assert!((best.score - 0.3).abs() < 1e-6);
}

// ── Aggregate operation ────────────────────────────────────────────────────────────

#[test]
fn test_aggregate_merges_frontier_into_one_node() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }, GotOperation::Aggregate];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    // 3 generated + 1 aggregated.
    assert_eq!(out.graph.len(), 4);
    let merged = out.graph.node(3).expect("merged node");
    assert!(merged.content.starts_with("merged:"));
}

#[test]
fn test_aggregate_new_node_parents_are_frontier() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }, GotOperation::Aggregate];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    let merged_id = 3;
    let mut parents = out.graph.parents_of(merged_id);
    parents.sort_unstable();
    assert_eq!(parents, vec![0, 1, 2]);
}

#[test]
fn test_aggregate_content_includes_all_thoughts() {
    let generator = FixedGenerator::new(&["alpha", "beta"]);
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::new(" & ");
    let plan = [GotOperation::Generate { k: 2 }, GotOperation::Aggregate];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    let merged = out.graph.node(2).expect("merged");
    assert_eq!(merged.content, "merged: alpha & beta");
}

#[test]
fn test_aggregate_empty_frontier_adds_nothing() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    // Aggregate with nothing in the frontier: NoThoughts error.
    let plan = [GotOperation::Aggregate];
    let err = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect_err("should error");
    assert!(matches!(err, GotError::NoThoughts));
}

// ── Refine operation ───────────────────────────────────────────────────────────────

#[test]
fn test_refine_replaces_best_when_improved() {
    // Generate one node "a" (score 0.2), then refine to "b" (score 0.9). A
    // two-phase scripted generator yields "a" on the first call and "b" on the
    // second (the refine call).
    struct Scripted {
        calls: std::cell::RefCell<usize>,
        outputs: Vec<Vec<String>>,
    }
    impl ThoughtGenerator for Scripted {
        fn generate(&self, _q: &str, _c: &[&str], _k: usize) -> Vec<String> {
            let mut idx = self.calls.borrow_mut();
            let out = self.outputs.get(*idx).cloned().unwrap_or_default();
            *idx += 1;
            out
        }
    }
    let scripted = Scripted {
        calls: std::cell::RefCell::new(0),
        outputs: vec![vec!["a".to_string()], vec!["b".to_string()]],
    };
    let scorer = MapScorer::new(&[("a", 0.2), ("b", 0.9)], 0.0);

    let plan = [GotOperation::Generate { k: 1 }, GotOperation::Refine];
    let out = default_engine()
        .run(
            "q",
            &plan,
            &scripted,
            &scorer,
            &MockThoughtAggregator::default(),
        )
        .expect("run");
    // Graph: node 0 = "a", node 1 = "b" (refinement of 0), provenance edge 0->1.
    // The refinement is a new node, so the edge connects distinct ids; node 1's
    // parent is the original node 0.
    assert_eq!(out.graph.len(), 2);
    assert_eq!(out.graph.edges(), &[(0, 1)]);
    assert_eq!(out.graph.parents_of(1), vec![0]);
    assert_eq!(out.graph.children_of(0), vec![1]);
    // The improved refinement (score 0.9) wins over the original (0.2).
    assert_eq!(out.best_thought, "b");
}

#[test]
fn test_refine_keeps_original_when_not_improved() {
    struct Scripted {
        calls: std::cell::RefCell<usize>,
        outputs: Vec<Vec<String>>,
    }
    impl ThoughtGenerator for Scripted {
        fn generate(&self, _q: &str, _c: &[&str], _k: usize) -> Vec<String> {
            let mut idx = self.calls.borrow_mut();
            let out = self.outputs.get(*idx).cloned().unwrap_or_default();
            *idx += 1;
            out
        }
    }
    let scripted = Scripted {
        calls: std::cell::RefCell::new(0),
        outputs: vec![vec!["good".to_string()], vec!["worse".to_string()]],
    };
    let scorer = MapScorer::new(&[("good", 0.9), ("worse", 0.1)], 0.0);
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 1 }, GotOperation::Refine];
    let out = default_engine()
        .run("q", &plan, &scripted, &scorer, &aggregator)
        .expect("run");
    // Refinement is recorded (node 1) but the best overall is still "good".
    assert_eq!(out.graph.len(), 2);
    assert_eq!(out.best_thought, "good");
    // Self-edge from original (0) to refinement (1) recorded.
    assert_eq!(out.graph.edge_count(), 1);
}

#[test]
fn test_refine_on_empty_frontier_is_noop() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    // Refine with empty frontier: nothing happens, then NoThoughts.
    let plan = [GotOperation::Refine];
    let err = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect_err("err");
    assert!(matches!(err, GotError::NoThoughts));
}

#[test]
fn test_refine_targets_best_frontier_node() {
    // Frontier after generate: three nodes "t 1","t 2","t 3" scored 0.1,0.2,0.3.
    // Refine should target "t 3". We script the refine output as "t 9".
    struct Scripted {
        calls: std::cell::RefCell<usize>,
    }
    impl ThoughtGenerator for Scripted {
        fn generate(&self, _q: &str, _c: &[&str], k: usize) -> Vec<String> {
            let mut idx = self.calls.borrow_mut();
            let phase = *idx;
            *idx += 1;
            if phase == 0 {
                (1..=k).map(|n| format!("t {n}")).collect()
            } else {
                vec!["t 9".to_string()]
            }
        }
    }
    let scripted = Scripted {
        calls: std::cell::RefCell::new(0),
    };
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }, GotOperation::Refine];
    let out = default_engine()
        .run("q", &plan, &scripted, &scorer, &aggregator)
        .expect("run");
    // The refinement (node 3) should hang off the best original node id 2 ("t 3").
    assert_eq!(out.graph.edges(), &[(2, 3)]);
    assert_eq!(out.best_thought, "t 9");
}

// ── KeepBest operation ─────────────────────────────────────────────────────────────

#[test]
fn test_keep_best_prunes_frontier() {
    // Generate 4 -> frontier scores 0.1..0.4. KeepBest{2} keeps the two best,
    // then Aggregate merges only those two.
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [
        GotOperation::Generate { k: 4 },
        GotOperation::KeepBest { n: 2 },
        GotOperation::Aggregate,
    ];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    // Merged node is id 4; its parents are the top-2 (ids 2 and 3: "t 3","t 4").
    let mut parents = out.graph.parents_of(4);
    parents.sort_unstable();
    assert_eq!(parents, vec![2, 3]);
}

#[test]
fn test_keep_best_keeps_top_by_score_into_aggregate_content() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::new(",");
    let plan = [
        GotOperation::Generate { k: 4 },
        GotOperation::KeepBest { n: 2 },
        GotOperation::Aggregate,
    ];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    let merged = out.graph.node(4).expect("merged");
    // KeepBest returns descending-score order: "t 4" then "t 3".
    assert_eq!(merged.content, "merged: t 4,t 3");
}

#[test]
fn test_keep_best_zero_clears_frontier() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    // After KeepBest{0} the frontier is empty; Aggregate then adds nothing.
    let plan = [
        GotOperation::Generate { k: 3 },
        GotOperation::KeepBest { n: 0 },
        GotOperation::Aggregate,
    ];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    // Only the 3 generated nodes; aggregate of empty frontier is a no-op.
    assert_eq!(out.graph.len(), 3);
}

#[test]
fn test_keep_best_tie_break_lower_id() {
    // Two thoughts with equal score; KeepBest{1} should retain the lower id.
    let generator = FixedGenerator::new(&["x", "y"]);
    let scorer = MapScorer::new(&[("x", 0.5), ("y", 0.5)], 0.0);
    let aggregator = MockThoughtAggregator::default();
    let plan = [
        GotOperation::Generate { k: 2 },
        GotOperation::KeepBest { n: 1 },
        GotOperation::Aggregate,
    ];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    // Aggregated node id 2 should have exactly one parent: id 0 ("x").
    assert_eq!(out.graph.parents_of(2), vec![0]);
}

// ── max_nodes cap ──────────────────────────────────────────────────────────────────

#[test]
fn test_max_nodes_cap_respected_single_generate() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let engine = GraphOfThoughtEngine::new(GotConfig::new().with_max_nodes(3));
    let plan = [GotOperation::Generate { k: 10 }];
    let out = engine
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert_eq!(out.graph.len(), 3);
}

#[test]
fn test_max_nodes_cap_respected_across_ops() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let engine = GraphOfThoughtEngine::new(GotConfig::new().with_max_nodes(5));
    let plan = [
        GotOperation::Generate { k: 4 },
        GotOperation::Generate { k: 4 },
        GotOperation::Aggregate,
    ];
    let out = engine
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert!(out.graph.len() <= 5);
}

#[test]
fn test_max_nodes_blocks_aggregate_when_full() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let engine = GraphOfThoughtEngine::new(GotConfig::new().with_max_nodes(2));
    let plan = [GotOperation::Generate { k: 2 }, GotOperation::Aggregate];
    let out = engine
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    // Full at 2 nodes: aggregate cannot add the merged node.
    assert_eq!(out.graph.len(), 2);
}

// ── Multi-op plan & final answer ───────────────────────────────────────────────────

#[test]
fn test_run_multi_op_plan_non_empty_answer() {
    let generator = MockThoughtGenerator::new("idea");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [
        GotOperation::Generate { k: 3 },
        GotOperation::KeepBest { n: 2 },
        GotOperation::Aggregate,
        GotOperation::Refine,
    ];
    let out = default_engine()
        .run("solve it", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert!(!out.final_answer.is_empty());
    assert!(!out.graph.is_empty());
}

#[test]
fn test_run_final_answer_equals_best_thought() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert_eq!(out.final_answer, out.best_thought);
    assert_eq!(out.final_answer, out.graph.best().expect("best").content);
}

#[test]
fn test_run_output_graph_is_populated() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 2 }, GotOperation::Aggregate];
    // Destructure the output struct to exercise all public fields.
    let GotOutput {
        graph,
        best_thought,
        final_answer,
    } = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert_eq!(graph.len(), 3);
    assert_eq!(best_thought, final_answer);
}

#[test]
fn test_run_aggregated_thought_can_win() {
    // Make the aggregated thought outscore every generated one.
    let generator = FixedGenerator::new(&["a", "b"]);
    let scorer = MapScorer::new(&[("a", 0.2), ("b", 0.3), ("merged: a + b", 0.99)], 0.0);
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 2 }, GotOperation::Aggregate];
    let out = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run");
    assert_eq!(out.final_answer, "merged: a + b");
}

// ── Errors ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_query_errors() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }];
    let err = default_engine()
        .run("", &plan, &generator, &scorer, &aggregator)
        .expect_err("err");
    assert!(matches!(err, GotError::EmptyQuery));
}

#[test]
fn test_whitespace_query_errors() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }];
    let err = default_engine()
        .run("   \t\n ", &plan, &generator, &scorer, &aggregator)
        .expect_err("err");
    assert!(matches!(err, GotError::EmptyQuery));
}

#[test]
fn test_empty_plan_yields_no_thoughts() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan: [GotOperation; 0] = [];
    let err = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect_err("err");
    assert!(matches!(err, GotError::NoThoughts));
}

#[test]
fn test_generator_returning_nothing_yields_no_thoughts() {
    let generator = FixedGenerator::new(&[]);
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [GotOperation::Generate { k: 3 }];
    let err = default_engine()
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect_err("err");
    assert!(matches!(err, GotError::NoThoughts));
}

#[test]
fn test_error_display_messages() {
    assert_eq!(GotError::EmptyQuery.to_string(), "query must not be empty");
    assert_eq!(GotError::NoThoughts.to_string(), "no thoughts generated");
}

// ── Determinism ────────────────────────────────────────────────────────────────────

#[test]
fn test_determinism_repeated_runs() {
    let generator = MockThoughtGenerator::new("t");
    let scorer = MockThoughtScorer::default();
    let aggregator = MockThoughtAggregator::default();
    let plan = [
        GotOperation::Generate { k: 3 },
        GotOperation::KeepBest { n: 2 },
        GotOperation::Aggregate,
    ];
    let engine = default_engine();
    let a = engine
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run a");
    let b = engine
        .run("q", &plan, &generator, &scorer, &aggregator)
        .expect("run b");
    assert_eq!(a.final_answer, b.final_answer);
    assert_eq!(a.graph, b.graph);
}

// ── Mocks ──────────────────────────────────────────────────────────────────────────

#[test]
fn test_mock_generator_output_shape() {
    let generator = MockThoughtGenerator::new("step");
    let out = generator.generate("q", &[], 3);
    assert_eq!(out, vec!["step 1", "step 2", "step 3"]);
}

#[test]
fn test_mock_scorer_trailing_number() {
    let scorer = MockThoughtScorer::new(0.1);
    assert!((scorer.score("q", "x 7") - 0.7).abs() < 1e-6);
}

#[test]
fn test_mock_scorer_no_trailing_number_is_zero() {
    let scorer = MockThoughtScorer::default();
    assert_eq!(scorer.score("q", "no number here"), 0.0);
}

#[test]
fn test_mock_aggregator_joins_with_separator() {
    let aggregator = MockThoughtAggregator::new(" | ");
    assert_eq!(
        aggregator.aggregate("q", &["a", "b", "c"]),
        "merged: a | b | c"
    );
}

#[test]
fn test_mock_aggregator_default_separator() {
    let aggregator = MockThoughtAggregator::default();
    assert_eq!(aggregator.aggregate("q", &["a", "b"]), "merged: a + b");
}
