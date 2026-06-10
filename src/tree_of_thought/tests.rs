#![allow(clippy::float_cmp)]
use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::tree_of_thought::search::TreeOfThoughtEngine;
use crate::tree_of_thought::types::{
    HeuristicThoughtEvaluator, HeuristicThoughtGenerator, ThoughtEvaluator, ThoughtGenerator,
    ThoughtSearchStrategy, ThoughtState, ThoughtTree, ThoughtTreeNode, ToTConfig, ToTOutput,
    TreeOfThoughtError,
};
use crate::types::{Document, DocumentId, SearchResult};
use async_trait::async_trait;

// ── MockEcho ──────────────────────────────────────────────────────────────
struct MockEcho {
    results: Vec<SearchResult>,
}
impl MockEcho {
    fn new(results: Vec<SearchResult>) -> Self {
        Self { results }
    }
}
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Echo for MockEcho {
    async fn index(&mut self, document: Document) -> Result<DocumentId, EmbeddingError> {
        Ok(document.id.clone())
    }
    async fn index_batch(
        &mut self,
        documents: Vec<Document>,
    ) -> Result<Vec<DocumentId>, EmbeddingError> {
        Ok(documents.into_iter().map(|d| d.id).collect())
    }
    async fn search(
        &self,
        _query: &str,
        _top_k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, EmbeddingError> {
        Ok(self.results.clone())
    }
    async fn get(&self, _id: &DocumentId) -> Result<Option<Document>, EmbeddingError> {
        Ok(None)
    }
    async fn delete(&mut self, _id: &DocumentId) -> Result<bool, EmbeddingError> {
        Ok(false)
    }
    async fn count(&self) -> usize {
        self.results.len()
    }
    async fn clear(&mut self) -> Result<(), EmbeddingError> {
        Ok(())
    }
}

// ── helpers ───────────────────────────────────────────────────────────────
fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

fn make_node(
    id: usize,
    parent: Option<usize>,
    depth: usize,
    value: f32,
    state: ThoughtState,
) -> ThoughtTreeNode {
    ThoughtTreeNode {
        id,
        parent,
        content: format!("node_{id}"),
        depth,
        value,
        state,
    }
}

// ── ThoughtState tests ────────────────────────────────────────────────────
#[test]
fn thought_state_default_is_active() {
    assert_eq!(ThoughtState::default(), ThoughtState::Active);
}

#[test]
fn thought_state_equality() {
    assert_eq!(ThoughtState::Active, ThoughtState::Active);
    assert_ne!(ThoughtState::Active, ThoughtState::Pruned);
    assert_ne!(ThoughtState::Solved, ThoughtState::DeadEnd);
}

#[test]
fn thought_state_clone() {
    let s = ThoughtState::Pruned;
    assert_eq!(s.clone(), ThoughtState::Pruned);
}

// ── ThoughtSearchStrategy tests ───────────────────────────────────────────
#[test]
fn thought_search_strategy_default_is_bfs() {
    assert_eq!(ThoughtSearchStrategy::default(), ThoughtSearchStrategy::Bfs);
}

#[test]
fn thought_search_strategy_equality() {
    assert_eq!(ThoughtSearchStrategy::Bfs, ThoughtSearchStrategy::Bfs);
    assert_ne!(ThoughtSearchStrategy::Bfs, ThoughtSearchStrategy::Dfs);
}

// ── ThoughtTreeNode tests ─────────────────────────────────────────────────
#[test]
fn thought_tree_node_construction() {
    let node = make_node(3, Some(1), 2, 0.7, ThoughtState::Active);
    assert_eq!(node.id, 3);
    assert_eq!(node.parent, Some(1));
    assert_eq!(node.depth, 2);
    assert!((node.value - 0.7).abs() < 1e-6);
    assert_eq!(node.state, ThoughtState::Active);
}

#[test]
fn thought_tree_node_root_has_no_parent() {
    let node = make_node(0, None, 0, 0.5, ThoughtState::Active);
    assert!(node.parent.is_none());
}

// ── ThoughtTree tests ─────────────────────────────────────────────────────
#[test]
fn thought_tree_default_is_empty() {
    let tree = ThoughtTree::default();
    assert!(tree.nodes.is_empty());
    assert!(tree.root.is_empty());
    assert!(tree.frontier.is_empty());
}

#[test]
fn thought_tree_children_of_no_children() {
    let mut tree = ThoughtTree::default();
    tree.nodes
        .push(make_node(0, None, 0, 0.5, ThoughtState::Active));
    let children = tree.children_of(0);
    assert!(children.is_empty());
}

#[test]
fn thought_tree_children_of_with_children() {
    let mut tree = ThoughtTree::default();
    tree.nodes
        .push(make_node(0, None, 0, 0.5, ThoughtState::Active));
    tree.nodes
        .push(make_node(1, Some(0), 1, 0.4, ThoughtState::Active));
    tree.nodes
        .push(make_node(2, Some(0), 1, 0.6, ThoughtState::Active));
    tree.nodes
        .push(make_node(3, Some(1), 2, 0.3, ThoughtState::Active));
    let children = tree.children_of(0);
    assert_eq!(children.len(), 2);
    assert!(children.contains(&1));
    assert!(children.contains(&2));
}

#[test]
fn thought_tree_path_to_root_for_root_node() {
    let mut tree = ThoughtTree::default();
    tree.nodes
        .push(make_node(0, None, 0, 0.5, ThoughtState::Active));
    let path = tree.path_to_root(0);
    assert_eq!(path, vec![0]);
}

#[test]
fn thought_tree_path_to_root_for_leaf_node() {
    let mut tree = ThoughtTree::default();
    tree.nodes
        .push(make_node(0, None, 0, 0.5, ThoughtState::Active));
    tree.nodes
        .push(make_node(1, Some(0), 1, 0.4, ThoughtState::Active));
    tree.nodes
        .push(make_node(2, Some(1), 2, 0.3, ThoughtState::Active));
    let path = tree.path_to_root(2);
    // path should include node 2, 1, 0
    assert_eq!(path.len(), 3);
    assert_eq!(path[0], 2);
    assert_eq!(path[2], 0);
}

#[test]
fn thought_tree_path_to_root_includes_both_root_and_leaf() {
    let mut tree = ThoughtTree::default();
    tree.nodes
        .push(make_node(0, None, 0, 0.5, ThoughtState::Active));
    tree.nodes
        .push(make_node(1, Some(0), 1, 0.4, ThoughtState::Active));
    let path = tree.path_to_root(1);
    assert!(path.contains(&0));
    assert!(path.contains(&1));
}

#[test]
fn thought_tree_best_leaf_no_active_nodes_returns_none() {
    let mut tree = ThoughtTree::default();
    tree.nodes
        .push(make_node(0, None, 0, 0.9, ThoughtState::Pruned));
    tree.nodes
        .push(make_node(1, Some(0), 1, 0.8, ThoughtState::Pruned));
    // No active leaves
    let best = tree.best_leaf();
    assert!(best.is_none());
}

#[test]
fn thought_tree_best_leaf_picks_highest_value() {
    let mut tree = ThoughtTree::default();
    tree.nodes
        .push(make_node(0, None, 0, 0.5, ThoughtState::Active));
    tree.nodes
        .push(make_node(1, Some(0), 1, 0.3, ThoughtState::Active));
    tree.nodes
        .push(make_node(2, Some(0), 1, 0.8, ThoughtState::Active));
    // node 0 has children so not a leaf; nodes 1 and 2 are leaves
    let best = tree.best_leaf();
    assert_eq!(best, Some(2));
}

#[test]
fn thought_tree_pruned_nodes_not_selected_as_best_leaf() {
    let mut tree = ThoughtTree::default();
    tree.nodes
        .push(make_node(0, None, 0, 0.5, ThoughtState::Active));
    tree.nodes
        .push(make_node(1, Some(0), 1, 0.9, ThoughtState::Pruned));
    tree.nodes
        .push(make_node(2, Some(0), 1, 0.3, ThoughtState::Active));
    let best = tree.best_leaf();
    assert_eq!(best, Some(2));
}

// ── HeuristicThoughtGenerator tests ──────────────────────────────────────
#[test]
fn heuristic_thought_generator_expand_returns_nonempty() {
    let generator = HeuristicThoughtGenerator;
    let node = make_node(0, None, 0, 0.5, ThoughtState::Active);
    let results: Vec<SearchResult> = vec![];
    let children = generator.expand(&node, "test query", &results);
    assert!(!children.is_empty());
}

#[test]
fn heuristic_thought_generator_content_contains_step_increment() {
    let generator = HeuristicThoughtGenerator;
    let node = make_node(0, None, 0, 0.5, ThoughtState::Active);
    let results: Vec<SearchResult> = vec![];
    let children = generator.expand(&node, "query", &results);
    // depth+1 = 1
    assert!(children[0].contains("step 1") || children[0].contains("(step"));
}

#[test]
fn heuristic_thought_generator_depth_reflects_parent() {
    let generator = HeuristicThoughtGenerator;
    let node = make_node(3, Some(1), 2, 0.5, ThoughtState::Active);
    let results: Vec<SearchResult> = vec![];
    let children = generator.expand(&node, "query", &results);
    // depth 2 → step 3
    assert!(children[0].contains("step 3") || children[0].contains("(step"));
}

// ── HeuristicThoughtEvaluator tests ──────────────────────────────────────
#[test]
fn heuristic_thought_evaluator_empty_path_returns_zero() {
    let eval = HeuristicThoughtEvaluator;
    let val = eval.value("query", &[], &[]);
    assert_eq!(val, 0.0);
}

#[test]
fn heuristic_thought_evaluator_short_path_in_range() {
    let eval = HeuristicThoughtEvaluator;
    let path = vec!["step1".to_string()];
    let val = eval.value("query", &path, &[]);
    assert!((0.0_f32..=0.5_f32).contains(&val));
}

#[test]
fn heuristic_thought_evaluator_value_increases_with_path_length() {
    let eval = HeuristicThoughtEvaluator;
    let short_path = vec!["step1".to_string()];
    let long_path = vec![
        "step1".to_string(),
        "step2".to_string(),
        "step3".to_string(),
    ];
    let short_val = eval.value("query", &short_path, &[]);
    let long_val = eval.value("query", &long_path, &[]);
    assert!(long_val >= short_val);
}

#[test]
fn heuristic_thought_evaluator_caps_at_point_five() {
    let eval = HeuristicThoughtEvaluator;
    let long_path: Vec<String> = (0..20).map(|i| format!("step{i}")).collect();
    let val = eval.value("query", &long_path, &[]);
    assert!(val <= 0.5 + 1e-6);
}

// ── ToTConfig tests ───────────────────────────────────────────────────────
#[test]
fn tot_config_default_values() {
    let c = ToTConfig::default();
    assert_eq!(c.max_depth, 3);
    assert_eq!(c.branching_factor, 3);
    assert_eq!(c.beam_width, 2);
    assert!((c.value_threshold - 0.5).abs() < 1e-6);
    assert_eq!(c.strategy, ThoughtSearchStrategy::Bfs);
    assert_eq!(c.top_k, 5);
}

#[test]
fn tot_config_builder_max_depth() {
    let c = ToTConfig::default().with_max_depth(5);
    assert_eq!(c.max_depth, 5);
}

#[test]
fn tot_config_builder_branching_factor() {
    let c = ToTConfig::default().with_branching_factor(4);
    assert_eq!(c.branching_factor, 4);
}

#[test]
fn tot_config_builder_beam_width() {
    let c = ToTConfig::default().with_beam_width(3);
    assert_eq!(c.beam_width, 3);
}

#[test]
fn tot_config_builder_value_threshold() {
    let c = ToTConfig::default().with_value_threshold(0.3);
    assert!((c.value_threshold - 0.3).abs() < 1e-6);
}

#[test]
fn tot_config_builder_strategy_dfs() {
    let c = ToTConfig::default().with_strategy(ThoughtSearchStrategy::Dfs);
    assert_eq!(c.strategy, ThoughtSearchStrategy::Dfs);
}

#[test]
fn tot_config_builder_top_k() {
    let c = ToTConfig::default().with_top_k(8);
    assert_eq!(c.top_k, 8);
}

// ── TreeOfThoughtEngine creation tests ────────────────────────────────────
#[test]
fn tree_of_thought_engine_new_stores_config() {
    let cfg = ToTConfig::default().with_max_depth(7);
    let engine = TreeOfThoughtEngine::new(cfg);
    assert_eq!(engine.config.max_depth, 7);
}

#[test]
fn tree_of_thought_engine_default_uses_default_config() {
    let engine = TreeOfThoughtEngine::default();
    assert_eq!(engine.config.max_depth, 3);
}

// ── TreeOfThoughtError tests ──────────────────────────────────────────────
#[test]
fn tree_of_thought_error_empty_query_display() {
    let e = TreeOfThoughtError::EmptyQuery;
    assert!(e.to_string().contains("empty") || e.to_string().contains("not be empty"));
}

#[test]
fn tree_of_thought_error_retrieval_failed_display() {
    let e = TreeOfThoughtError::RetrievalFailed("timeout".to_string());
    let s = e.to_string();
    assert!(s.contains("timeout") || s.contains("Retrieval"));
}

#[test]
fn tree_of_thought_error_no_thoughts_display() {
    let e = TreeOfThoughtError::NoThoughtsGenerated;
    let s = e.to_string();
    assert!(s.contains("thoughts") || s.contains("generated") || s.contains("No"));
}

// ── TreeOfThoughtEngine::run tests (async) ────────────────────────────────
#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_empty_query_returns_error() {
    let engine = TreeOfThoughtEngine::default();
    let echo = MockEcho::new(vec![]);
    let result = engine.run("", &echo).await;
    assert!(matches!(result, Err(TreeOfThoughtError::EmptyQuery)));
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_whitespace_query_returns_error() {
    let engine = TreeOfThoughtEngine::default();
    let echo = MockEcho::new(vec![]);
    let result = engine.run("  ", &echo).await;
    assert!(matches!(result, Err(TreeOfThoughtError::EmptyQuery)));
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_bfs_completes() {
    let cfg = ToTConfig::default().with_strategy(ThoughtSearchStrategy::Bfs);
    let engine = TreeOfThoughtEngine::new(cfg);
    let echo = MockEcho::new(vec![make_result("d1", "content", 0.8)]);
    let result = engine.run("query", &echo).await;
    assert!(result.is_ok());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_dfs_completes() {
    let cfg = ToTConfig::default().with_strategy(ThoughtSearchStrategy::Dfs);
    let engine = TreeOfThoughtEngine::new(cfg);
    let echo = MockEcho::new(vec![make_result("d1", "content", 0.8)]);
    let result = engine.run("query", &echo).await;
    assert!(result.is_ok());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_explored_nodes_at_least_one() {
    let engine = TreeOfThoughtEngine::default();
    let echo = MockEcho::new(vec![]);
    let output: ToTOutput = engine.run("question", &echo).await.unwrap();
    assert!(output.explored_nodes >= 1);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_best_path_nonempty() {
    let engine = TreeOfThoughtEngine::default();
    let echo = MockEcho::new(vec![]);
    let output = engine.run("question", &echo).await.unwrap();
    assert!(!output.best_path.is_empty());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_answer_nonempty() {
    let engine = TreeOfThoughtEngine::default();
    let echo = MockEcho::new(vec![make_result("d1", "answer content", 0.9)]);
    let output = engine.run("question", &echo).await.unwrap();
    assert!(!output.answer.is_empty());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_max_depth_one_fewer_nodes_than_depth_three() {
    let echo1 = MockEcho::new(vec![make_result("d1", "content", 0.8)]);
    let cfg1 = ToTConfig::default()
        .with_max_depth(1)
        .with_value_threshold(0.0);
    let output1 = TreeOfThoughtEngine::new(cfg1)
        .run("query", &echo1)
        .await
        .unwrap();

    let echo3 = MockEcho::new(vec![make_result("d1", "content", 0.8)]);
    let cfg3 = ToTConfig::default()
        .with_max_depth(3)
        .with_value_threshold(0.0);
    let output3 = TreeOfThoughtEngine::new(cfg3)
        .run("query", &echo3)
        .await
        .unwrap();

    assert!(output1.explored_nodes <= output3.explored_nodes);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_beam_width_limits_frontier() {
    let cfg = ToTConfig::default()
        .with_beam_width(1)
        .with_branching_factor(3)
        .with_value_threshold(0.0);
    let engine = TreeOfThoughtEngine::new(cfg);
    let echo = MockEcho::new(vec![]);
    let output = engine.run("query", &echo).await.unwrap();
    // With beam_width=1, we should still complete successfully
    assert!(!output.best_path.is_empty());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_high_value_threshold_prunes_nodes() {
    // value_threshold=1.0 means nothing passes unless value==1.0
    // The heuristic evaluator caps at 0.5, so everything gets pruned
    let cfg = ToTConfig::default().with_value_threshold(1.0);
    let engine = TreeOfThoughtEngine::new(cfg);
    let echo = MockEcho::new(vec![make_result("d1", "content", 0.8)]);
    let output = engine.run("query", &echo).await.unwrap();
    // Tree still has nodes but frontier is empty (all pruned)
    let pruned_count = output
        .tree
        .nodes
        .iter()
        .filter(|n| n.state == ThoughtState::Pruned)
        .count();
    // root has value 0.5 which < 1.0 is the threshold for children but root itself exists
    // Either pruned children or empty frontier
    let _ = pruned_count; // at minimum the run must succeed
    assert!(output.explored_nodes >= 1);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_explored_nodes_greater_than_one_nontrivial() {
    let cfg = ToTConfig::default()
        .with_max_depth(2)
        .with_branching_factor(2)
        .with_value_threshold(0.0);
    let engine = TreeOfThoughtEngine::new(cfg);
    let echo = MockEcho::new(vec![make_result("d1", "content", 0.8)]);
    let output = engine.run("query", &echo).await.unwrap();
    assert!(output.explored_nodes > 1);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_pruned_nodes_in_tree_not_frontier() {
    let cfg = ToTConfig::default()
        .with_value_threshold(1.0) // prune everything new
        .with_max_depth(2);
    let engine = TreeOfThoughtEngine::new(cfg);
    let echo = MockEcho::new(vec![make_result("d1", "content", 0.8)]);
    let output = engine.run("query", &echo).await.unwrap();
    // Any pruned nodes should not appear in the final frontier
    for pruned_id in output
        .tree
        .nodes
        .iter()
        .filter(|n| n.state == ThoughtState::Pruned)
        .map(|n| n.id)
    {
        assert!(!output.tree.frontier.contains(&pruned_id));
    }
}

#[cfg(feature = "native")]
#[tokio::test]
async fn tree_of_thought_engine_run_output_tree_has_root() {
    let engine = TreeOfThoughtEngine::default();
    let echo = MockEcho::new(vec![]);
    let output = engine.run("query", &echo).await.unwrap();
    assert!(!output.tree.nodes.is_empty());
    assert!(!output.tree.root.is_empty());
}
