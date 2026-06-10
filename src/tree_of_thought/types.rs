//! Types for the `tree_of_thought` module.
use thiserror::Error;
// ── ThoughtState ──────────────────────────────────────────────────────────────
/// State of a node in the thought tree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ThoughtState {
    /// Node is active and can be expanded.
    #[default]
    Active,
    /// Node was pruned due to low value.
    Pruned,
    /// Node was identified as a solution.
    Solved,
    /// Node is a dead-end with no valid continuations.
    DeadEnd,
}
// ── ThoughtSearchStrategy ─────────────────────────────────────────────────────
/// Search strategy for the thought tree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ThoughtSearchStrategy {
    /// Breadth-first search.
    #[default]
    Bfs,
    /// Depth-first search.
    Dfs,
}
// ── ThoughtTreeNode ───────────────────────────────────────────────────────────
/// A single node in the thought tree.
#[derive(Debug, Clone)]
pub struct ThoughtTreeNode {
    /// Unique ID for this node (0-indexed).
    pub id: usize,
    /// Parent node ID; `None` for the root.
    pub parent: Option<usize>,
    /// The reasoning content of this thought.
    pub content: String,
    /// Depth in the tree (root = 0).
    pub depth: usize,
    /// Heuristic value score in [0.0, 1.0].
    pub value: f32,
    /// Current state of this node.
    pub state: ThoughtState,
}
// ── ThoughtTree ───────────────────────────────────────────────────────────────
/// The full tree of thoughts.
#[derive(Debug, Clone, Default)]
pub struct ThoughtTree {
    /// All nodes in the tree, indexed by ID.
    pub nodes: Vec<ThoughtTreeNode>,
    /// IDs of root nodes.
    pub root: Vec<usize>,
    /// IDs of currently active frontier nodes.
    pub frontier: Vec<usize>,
}
impl ThoughtTree {
    /// Return IDs of direct children of `parent_id`.
    #[must_use]
    pub fn children_of(&self, parent_id: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.parent == Some(parent_id))
            .map(|n| n.id)
            .collect()
    }
    /// Return the path from `node_id` up to the root (inclusive, root last).
    #[must_use]
    pub fn path_to_root(&self, node_id: usize) -> Vec<usize> {
        let mut path = vec![node_id];
        let mut cur = node_id;
        while let Some(node) = self.nodes.get(cur) {
            if let Some(p) = node.parent {
                path.push(p);
                cur = p;
            } else {
                break;
            }
        }
        path
    }
    /// Return the ID of the leaf with the highest value.
    #[must_use]
    pub fn best_leaf(&self) -> Option<usize> {
        self.nodes
            .iter()
            .filter(|n| self.children_of(n.id).is_empty() && n.state == ThoughtState::Active)
            .max_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|n| n.id)
    }
}
// ── ThoughtGenerator ──────────────────────────────────────────────────────────
/// Synchronous trait for generating child thought texts.
pub trait ThoughtGenerator {
    /// Expand `thought` into candidate child content strings.
    fn expand(
        &self,
        thought: &ThoughtTreeNode,
        query: &str,
        context: &[crate::types::SearchResult],
    ) -> Vec<String>;
}
// ── ThoughtEvaluator ──────────────────────────────────────────────────────────
/// Synchronous trait for scoring a thought path.
pub trait ThoughtEvaluator {
    /// Score `path` of thought contents given `query` and retrieval `context`.
    ///
    /// Returns a value in [0.0, 1.0].
    fn value(&self, query: &str, path: &[String], context: &[crate::types::SearchResult]) -> f32;
}
// ── HeuristicThoughtGenerator ─────────────────────────────────────────────────
/// Lexical heuristic thought generator.
#[derive(Debug, Clone, Default)]
pub struct HeuristicThoughtGenerator;
impl ThoughtGenerator for HeuristicThoughtGenerator {
    fn expand(
        &self,
        thought: &ThoughtTreeNode,
        _query: &str,
        _context: &[crate::types::SearchResult],
    ) -> Vec<String> {
        vec![format!(
            "{} → (step {})",
            thought.content,
            thought.depth + 1
        )]
    }
}
// ── HeuristicThoughtEvaluator ─────────────────────────────────────────────────
/// Lexical heuristic thought evaluator.
#[derive(Debug, Clone, Default)]
pub struct HeuristicThoughtEvaluator;
impl ThoughtEvaluator for HeuristicThoughtEvaluator {
    fn value(&self, _query: &str, path: &[String], _context: &[crate::types::SearchResult]) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        {
            0.5_f32.min(path.len() as f32 * 0.1)
        }
    }
}
// ── ToTConfig ─────────────────────────────────────────────────────────────────
/// Configuration for `TreeOfThoughtEngine`.
#[derive(Debug, Clone)]
pub struct ToTConfig {
    /// Maximum tree depth. Defaults to `3`.
    pub max_depth: usize,
    /// Maximum child thoughts per expansion. Defaults to `3`.
    pub branching_factor: usize,
    /// Number of nodes to keep active per level (beam). Defaults to `2`.
    pub beam_width: usize,
    /// Minimum value for a node to stay active. Defaults to `0.5`.
    pub value_threshold: f32,
    /// Search strategy. Defaults to [`ThoughtSearchStrategy::Bfs`].
    pub strategy: ThoughtSearchStrategy,
    /// Documents to retrieve for the initial context. Defaults to `5`.
    pub top_k: usize,
}
impl Default for ToTConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            branching_factor: 3,
            beam_width: 2,
            value_threshold: 0.5,
            strategy: ThoughtSearchStrategy::Bfs,
            top_k: 5,
        }
    }
}
impl ToTConfig {
    /// Set the max depth.
    #[must_use]
    pub fn with_max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }
    /// Set the branching factor.
    #[must_use]
    pub fn with_branching_factor(mut self, v: usize) -> Self {
        self.branching_factor = v;
        self
    }
    /// Set the beam width.
    #[must_use]
    pub fn with_beam_width(mut self, v: usize) -> Self {
        self.beam_width = v;
        self
    }
    /// Set the value threshold.
    #[must_use]
    pub fn with_value_threshold(mut self, v: f32) -> Self {
        self.value_threshold = v;
        self
    }
    /// Set the search strategy.
    #[must_use]
    pub fn with_strategy(mut self, v: ThoughtSearchStrategy) -> Self {
        self.strategy = v;
        self
    }
    /// Set the retrieval top-k.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }
}
// ── ToTOutput ─────────────────────────────────────────────────────────────────
/// Output of a Tree-of-Thoughts search.
#[derive(Debug, Clone)]
pub struct ToTOutput {
    /// Synthesized final answer from the best path.
    pub answer: String,
    /// Contents of the nodes along the best reasoning path.
    pub best_path: Vec<String>,
    /// The full thought tree that was explored.
    pub tree: ThoughtTree,
    /// Total number of nodes explored.
    pub explored_nodes: usize,
}
// ── TreeOfThoughtError ────────────────────────────────────────────────────────
/// Errors from the `tree_of_thought` module.
#[derive(Debug, Error)]
pub enum TreeOfThoughtError {
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
    /// The underlying retrieval step failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
    /// No thoughts were generated during the search.
    #[error("No thoughts were generated during the search")]
    NoThoughtsGenerated,
}
