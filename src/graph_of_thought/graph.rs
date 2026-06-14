//! [`Thought`] and [`ThoughtGraph`] — the directed thought graph (DAG) data model.

// ── Thought ─────────────────────────────────────────────────────────────────────

/// A single thought: a vertex in a [`ThoughtGraph`].
#[derive(Debug, Clone, PartialEq)]
pub struct Thought {
    /// Stable identifier — the thought's index in the graph's node list.
    pub id: usize,
    /// The thought's textual content.
    pub content: String,
    /// The thought's quality score, conventionally in `[0, 1]`.
    pub score: f32,
}

impl Thought {
    /// Create a new thought with the given `id`, `content`, and `score`.
    #[must_use]
    pub fn new(id: usize, content: impl Into<String>, score: f32) -> Self {
        Self {
            id,
            content: content.into(),
            score,
        }
    }
}

// ── ThoughtGraph ────────────────────────────────────────────────────────────────

/// A directed acyclic graph of [`Thought`]s.
///
/// Nodes are stored in insertion order; a node's [`Thought::id`] equals its
/// index in [`ThoughtGraph::nodes`]. Edges are directed `(from, to)` pairs that
/// record provenance — a [`GotOperation::Generate`](crate::graph_of_thought::GotOperation::Generate)
/// edge points from a parent thought to each branched child, an
/// [`GotOperation::Aggregate`](crate::graph_of_thought::GotOperation::Aggregate)
/// edge points from each merged thought to the synthesized result, and a
/// [`GotOperation::Refine`](crate::graph_of_thought::GotOperation::Refine)
/// edge is a self-edge from a thought to its improved version.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThoughtGraph {
    /// All thoughts, indexed by [`Thought::id`].
    nodes: Vec<Thought>,
    /// Directed `(from, to)` edges by node id.
    edges: Vec<(usize, usize)>,
}

impl ThoughtGraph {
    /// Create a new, empty thought graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node with `content` and `score`, returning its new id.
    ///
    /// The returned id equals the node's index in insertion order.
    pub fn add_node(&mut self, content: impl Into<String>, score: f32) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Thought::new(id, content, score));
        id
    }

    /// Add a directed edge `from -> to`.
    ///
    /// Endpoints that are out of range are ignored, so callers never need to
    /// guard against stale ids; this keeps the graph free of dangling edges.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        if from < self.nodes.len() && to < self.nodes.len() {
            self.edges.push((from, to));
        }
    }

    /// Borrow the thought with the given `id`, if it exists.
    #[must_use]
    pub fn node(&self, id: usize) -> Option<&Thought> {
        self.nodes.get(id)
    }

    /// Ids of the direct parents of `id` (sources of edges into `id`).
    ///
    /// Self-edges (`id -> id`, produced by refinement) are excluded so a
    /// refined thought is not reported as its own parent. Parents are returned
    /// in edge-insertion order.
    #[must_use]
    pub fn parents_of(&self, id: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|(from, to)| *to == id && *from != id)
            .map(|(from, _)| *from)
            .collect()
    }

    /// Ids of the direct children of `id` (targets of edges out of `id`).
    ///
    /// Self-edges are excluded. Children are returned in edge-insertion order.
    #[must_use]
    pub fn children_of(&self, id: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|(from, to)| *from == id && *to != id)
            .map(|(_, to)| *to)
            .collect()
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Total number of edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Borrow all directed edges as `(from, to)` pairs, in insertion order.
    #[must_use]
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Borrow all thoughts, in insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[Thought] {
        &self.nodes
    }

    /// The highest-scoring thought, breaking ties toward the lowest id.
    ///
    /// Returns `None` for an empty graph.
    #[must_use]
    pub fn best(&self) -> Option<&Thought> {
        self.nodes.iter().fold(None, |best, candidate| match best {
            None => Some(candidate),
            Some(current) => {
                // Strictly-greater keeps the earliest (lowest-id) thought on ties.
                if candidate.score > current.score {
                    Some(candidate)
                } else {
                    Some(current)
                }
            }
        })
    }
}
