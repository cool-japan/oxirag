//! Tree-of-Thoughts search engine.
use crate::tree_of_thought::types::{
    HeuristicThoughtEvaluator, HeuristicThoughtGenerator, ThoughtEvaluator, ThoughtGenerator,
    ThoughtSearchStrategy, ThoughtState, ThoughtTree, ThoughtTreeNode, ToTConfig, ToTOutput,
    TreeOfThoughtError,
};
use crate::types::SearchResult;

// ── TreeOfThoughtEngine ───────────────────────────────────────────────────────

/// Tree-of-Thoughts engine — branches reasoning into a tree and searches for the best path.
///
/// The [`Echo`] layer is provided *per call* through the generic `run<E>` method.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
#[derive(Debug, Clone)]
pub struct TreeOfThoughtEngine {
    /// Configuration for this engine.
    pub config: ToTConfig,
}

impl TreeOfThoughtEngine {
    /// Create a new engine with the given config.
    #[must_use]
    pub fn new(config: ToTConfig) -> Self {
        Self { config }
    }

    /// Run Tree-of-Thoughts search for `query`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeOfThoughtError::EmptyQuery`] if `query` is empty.
    /// Returns [`TreeOfThoughtError::RetrievalFailed`] if retrieval fails.
    /// Returns [`TreeOfThoughtError::NoThoughtsGenerated`] if the tree is empty after search.
    pub async fn run<E>(&self, query: &str, echo: &E) -> Result<ToTOutput, TreeOfThoughtError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(TreeOfThoughtError::EmptyQuery);
        }

        let results: Vec<SearchResult> = echo
            .search(query, self.config.top_k, None)
            .await
            .map_err(|e| TreeOfThoughtError::RetrievalFailed(e.to_string()))?;

        let generator = HeuristicThoughtGenerator;
        let evaluator = HeuristicThoughtEvaluator;

        let mut tree = ThoughtTree::default();
        let root = ThoughtTreeNode {
            id: 0,
            parent: None,
            content: format!("Initial thought about: {query}"),
            depth: 0,
            value: 0.5,
            state: ThoughtState::Active,
        };
        tree.nodes.push(root);
        tree.root.push(0);
        tree.frontier.push(0);

        let max_nodes =
            self.config.max_depth * self.config.branching_factor * self.config.beam_width + 1;
        let mut explored = 1usize;

        match self.config.strategy {
            ThoughtSearchStrategy::Bfs => {
                let mut depth = 0;
                while depth < self.config.max_depth
                    && !tree.frontier.is_empty()
                    && explored < max_nodes
                {
                    let frontier = std::mem::take(&mut tree.frontier);
                    let mut scored: Vec<(usize, f32)> = Vec::new();
                    for parent_id in frontier {
                        let parent = tree.nodes[parent_id].clone();
                        let children_content = generator.expand(&parent, query, &results);
                        for content in children_content
                            .into_iter()
                            .take(self.config.branching_factor)
                        {
                            let id = tree.nodes.len();
                            let path: Vec<String> = tree
                                .path_to_root(parent_id)
                                .into_iter()
                                .map(|i| tree.nodes[i].content.clone())
                                .collect();
                            let value = evaluator.value(query, &path, &results).clamp(0.0, 1.0);
                            let state = if value >= self.config.value_threshold {
                                ThoughtState::Active
                            } else {
                                ThoughtState::Pruned
                            };
                            tree.nodes.push(ThoughtTreeNode {
                                id,
                                parent: Some(parent_id),
                                content,
                                depth: depth + 1,
                                value,
                                state: state.clone(),
                            });
                            if state == ThoughtState::Active {
                                scored.push((id, value));
                            }
                            explored += 1;
                        }
                    }
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    tree.frontier = scored
                        .into_iter()
                        .take(self.config.beam_width)
                        .map(|(id, _)| id)
                        .collect();
                    depth += 1;
                }
            }
            ThoughtSearchStrategy::Dfs => {
                let mut stack = tree.frontier.clone();
                tree.frontier.clear();
                while let Some(parent_id) = stack.pop() {
                    if explored >= max_nodes {
                        break;
                    }
                    let parent = tree.nodes[parent_id].clone();
                    if parent.depth >= self.config.max_depth {
                        continue;
                    }
                    let children_content = generator.expand(&parent, query, &results);
                    for content in children_content
                        .into_iter()
                        .take(self.config.branching_factor)
                    {
                        let id = tree.nodes.len();
                        let path: Vec<String> = tree
                            .path_to_root(parent_id)
                            .into_iter()
                            .map(|i| tree.nodes[i].content.clone())
                            .collect();
                        let value = evaluator.value(query, &path, &results).clamp(0.0, 1.0);
                        let state = if value >= self.config.value_threshold {
                            ThoughtState::Active
                        } else {
                            ThoughtState::Pruned
                        };
                        tree.nodes.push(ThoughtTreeNode {
                            id,
                            parent: Some(parent_id),
                            content,
                            depth: parent.depth + 1,
                            value,
                            state: state.clone(),
                        });
                        if state == ThoughtState::Active {
                            stack.push(id);
                        }
                        explored += 1;
                    }
                }
            }
        }

        let best_leaf = tree
            .best_leaf()
            .or(Some(0))
            .ok_or(TreeOfThoughtError::NoThoughtsGenerated)?;
        let best_path_ids = tree.path_to_root(best_leaf);
        let best_path: Vec<String> = best_path_ids
            .iter()
            .rev()
            .map(|&i| tree.nodes[i].content.clone())
            .collect();
        let answer = synthesize(query, &best_path, &results);

        Ok(ToTOutput {
            answer,
            best_path,
            tree,
            explored_nodes: explored,
        })
    }
}

impl Default for TreeOfThoughtEngine {
    fn default() -> Self {
        Self::new(ToTConfig::default())
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn synthesize(query: &str, path: &[String], results: &[SearchResult]) -> String {
    let path_summary = path.last().cloned().unwrap_or_else(|| query.to_string());
    if results.is_empty() {
        return path_summary;
    }
    let ctx = results
        .iter()
        .take(2)
        .map(|r| r.document.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    format!("{path_summary} {ctx}")
}
