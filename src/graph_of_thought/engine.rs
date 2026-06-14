//! [`GraphOfThoughtEngine`] — executes a plan of graph operations over thoughts.

use crate::graph_of_thought::graph::ThoughtGraph;
use crate::graph_of_thought::types::{
    GotConfig, GotError, GotOperation, GotOutput, ThoughtAggregator, ThoughtGenerator,
    ThoughtScorer,
};

// ── GraphOfThoughtEngine ─────────────────────────────────────────────────────────

/// Engine that builds a [`ThoughtGraph`] by executing a sequence of
/// [`GotOperation`]s.
///
/// The engine maintains a *frontier* — the ids of the thoughts currently in
/// play — and applies each operation in the plan to it, in order:
///
/// - [`GotOperation::Generate`] branches `k` new thoughts from every frontier
///   thought (or from scratch when the frontier is empty), scoring each and
///   linking it to its parent; the new thoughts become the frontier. Generation
///   stops adding nodes once [`GotConfig::max_nodes`] is reached.
/// - [`GotOperation::Aggregate`] merges all frontier thoughts into one
///   synthesized thought (with the merged thoughts as parents); it becomes the
///   sole frontier.
/// - [`GotOperation::Refine`] regenerates and rescores the best frontier
///   thought via a self-edge, replacing it in the frontier when the refinement
///   scores at least as high.
/// - [`GotOperation::KeepBest`] prunes the frontier to its top-`n` thoughts by
///   score (ties broken toward lower ids).
///
/// All steps are fully deterministic given deterministic trait implementations.
#[derive(Debug, Clone, Default)]
pub struct GraphOfThoughtEngine {
    /// Configuration for this engine.
    pub config: GotConfig,
}

impl GraphOfThoughtEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: GotConfig) -> Self {
        Self { config }
    }

    /// Execute `plan` for `query`, returning the constructed graph and answer.
    ///
    /// The operations in `plan` are applied in order to an initially empty
    /// frontier. The final answer is the content of the highest-scoring thought
    /// in the whole graph (see [`ThoughtGraph::best`]).
    ///
    /// # Errors
    ///
    /// - [`GotError::EmptyQuery`] when `query` is empty after trimming.
    /// - [`GotError::NoThoughts`] when the plan produces no thoughts at all
    ///   (e.g. an empty plan, or a plan that never generates).
    pub fn run<G, S, A>(
        &self,
        query: &str,
        plan: &[GotOperation],
        generator: &G,
        scorer: &S,
        aggregator: &A,
    ) -> Result<GotOutput, GotError>
    where
        G: ThoughtGenerator + ?Sized,
        S: ThoughtScorer + ?Sized,
        A: ThoughtAggregator + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(GotError::EmptyQuery);
        }

        let mut graph = ThoughtGraph::new();
        let mut frontier: Vec<usize> = Vec::new();

        for operation in plan {
            match *operation {
                GotOperation::Generate { k } => {
                    frontier =
                        self.apply_generate(query, k, &mut graph, &frontier, generator, scorer);
                }
                GotOperation::Aggregate => {
                    frontier =
                        self.apply_aggregate(query, &mut graph, &frontier, aggregator, scorer);
                }
                GotOperation::Refine => {
                    self.apply_refine(query, &mut graph, &mut frontier, generator, scorer);
                }
                GotOperation::KeepBest { n } => {
                    frontier = Self::apply_keep_best(&graph, &frontier, n);
                }
            }
        }

        let best = graph.best().ok_or(GotError::NoThoughts)?;
        let best_thought = best.content.clone();
        let final_answer = best_thought.clone();

        Ok(GotOutput {
            graph,
            best_thought,
            final_answer,
        })
    }

    /// Branch `k` new thoughts from each frontier thought (or from scratch),
    /// score them, link them to their parent, and return the new frontier.
    fn apply_generate<G, S>(
        &self,
        query: &str,
        k: usize,
        graph: &mut ThoughtGraph,
        frontier: &[usize],
        generator: &G,
        scorer: &S,
    ) -> Vec<usize>
    where
        G: ThoughtGenerator + ?Sized,
        S: ThoughtScorer + ?Sized,
    {
        let effective_k = if k == 0 { self.config.branch_factor } else { k };
        let mut next_frontier: Vec<usize> = Vec::new();

        // Branch from each frontier thought; when the frontier is empty, branch
        // once from an empty context (the initial generation).
        let parents: Vec<Option<usize>> = if frontier.is_empty() {
            vec![None]
        } else {
            frontier.iter().map(|id| Some(*id)).collect()
        };

        for parent in parents {
            if graph.len() >= self.config.max_nodes {
                break;
            }
            let context: Vec<&str> = match parent {
                Some(id) => graph
                    .node(id)
                    .map(|t| t.content.as_str())
                    .into_iter()
                    .collect(),
                None => Vec::new(),
            };
            let candidates = generator.generate(query, &context, effective_k);
            for content in candidates {
                if graph.len() >= self.config.max_nodes {
                    break;
                }
                let score = scorer.score(query, &content);
                let child = graph.add_node(content, score);
                if let Some(parent_id) = parent {
                    graph.add_edge(parent_id, child);
                }
                next_frontier.push(child);
            }
        }

        next_frontier
    }

    /// Merge the frontier thoughts into one synthesized thought with the merged
    /// thoughts as parents, returning it as the sole new frontier.
    ///
    /// An empty frontier is left unchanged (there is nothing to merge).
    fn apply_aggregate<A, S>(
        &self,
        query: &str,
        graph: &mut ThoughtGraph,
        frontier: &[usize],
        aggregator: &A,
        scorer: &S,
    ) -> Vec<usize>
    where
        A: ThoughtAggregator + ?Sized,
        S: ThoughtScorer + ?Sized,
    {
        if frontier.is_empty() {
            return Vec::new();
        }
        if graph.len() >= self.config.max_nodes {
            return frontier.to_vec();
        }

        let contents: Vec<String> = frontier
            .iter()
            .filter_map(|id| graph.node(*id).map(|t| t.content.clone()))
            .collect();
        let borrowed: Vec<&str> = contents.iter().map(String::as_str).collect();

        let merged = aggregator.aggregate(query, &borrowed);
        let score = scorer.score(query, &merged);
        let new_id = graph.add_node(merged, score);
        for parent in frontier {
            graph.add_edge(*parent, new_id);
        }

        vec![new_id]
    }

    /// Regenerate and rescore the best frontier thought via a self-edge.
    ///
    /// The single best (highest-scoring, lowest-id) frontier thought is
    /// re-generated; the refinement is added as a new node with a self-edge from
    /// the original. When it scores at least as high as the original, it
    /// replaces the original in the frontier. A non-improving refinement leaves
    /// the frontier unchanged (but is still recorded in the graph).
    fn apply_refine<G, S>(
        &self,
        query: &str,
        graph: &mut ThoughtGraph,
        frontier: &mut [usize],
        generator: &G,
        scorer: &S,
    ) where
        G: ThoughtGenerator + ?Sized,
        S: ThoughtScorer + ?Sized,
    {
        if graph.len() >= self.config.max_nodes {
            return;
        }
        let Some(best_pos) = best_frontier_position(graph, frontier) else {
            return;
        };
        let best_id = frontier[best_pos];
        // Clone the original's content and score, releasing the immutable borrow
        // of `graph` before we add the refinement node.
        let Some((original_content, original_score)) =
            graph.node(best_id).map(|t| (t.content.clone(), t.score))
        else {
            return;
        };
        let context: Vec<&str> = vec![original_content.as_str()];

        let mut candidates = generator.generate(query, &context, 1);
        let Some(refined_content) = candidates.pop() else {
            return;
        };
        let refined_score = scorer.score(query, &refined_content);
        let refined = graph.add_node(refined_content, refined_score);
        graph.add_edge(best_id, refined);

        if refined_score >= original_score {
            frontier[best_pos] = refined;
        }
    }

    /// Prune the frontier to its top-`n` thoughts by score.
    ///
    /// Ties are broken toward lower ids, matching [`ThoughtGraph::best`]. The
    /// retained ids are returned in descending-score order. `n == 0` clears the
    /// frontier.
    fn apply_keep_best(graph: &ThoughtGraph, frontier: &[usize], n: usize) -> Vec<usize> {
        let mut ranked: Vec<usize> = frontier.to_vec();
        ranked.sort_by(|a, b| frontier_order(graph, *a, *b));
        ranked.truncate(n);
        ranked
    }
}

// ── Ordering helpers ─────────────────────────────────────────────────────────────

/// Total order over frontier thought ids: score descending, then id ascending.
fn frontier_order(graph: &ThoughtGraph, a: usize, b: usize) -> std::cmp::Ordering {
    let score_a = graph.node(a).map_or(f32::NEG_INFINITY, |t| t.score);
    let score_b = graph.node(b).map_or(f32::NEG_INFINITY, |t| t.score);
    score_b
        .partial_cmp(&score_a)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.cmp(&b))
}

/// Position (within `frontier`) of the best frontier thought, or `None` when the
/// frontier is empty. Ties break toward the lowest id.
fn best_frontier_position(graph: &ThoughtGraph, frontier: &[usize]) -> Option<usize> {
    frontier
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| frontier_order(graph, **a, **b))
        .map(|(pos, _)| pos)
}
