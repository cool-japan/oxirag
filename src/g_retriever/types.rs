//! Types for the `g_retriever` module.
//!
//! These types describe **G-Retriever** (He et al. 2024, "G-Retriever:
//! Retrieval-Augmented Generation for Textual Graph Understanding and Question
//! Answering"): the retrieval of a query-relevant *subgraph* of a textual
//! knowledge graph, framed as a **Prize-Collecting Steiner Tree (PCST)**
//! optimisation.
//!
//! Every graph node ([`GRetrieverEntity`]) is assigned a nonnegative *prize*
//! measuring its relevance to the query; every graph edge
//! ([`GRetrieverRelation`]) is assigned a positive *cost*. The retrieved
//! subgraph is the connected structure that maximises collected prize minus
//! paid cost — so a highly relevant node is worth "buying" edges to reach,
//! while an off-topic node is left out unless it is a cheap *Steiner* waypoint
//! needed to connect two relevant nodes.
//!
//! The algorithmic core lives in [`PcstSolver`](super::pcst::PcstSolver),
//! which operates over the abstract [`PcstNode`] / [`PcstEdge`] view and
//! returns a [`PcstForest`]. The engine
//! ([`GRetrieverEngine`](super::engine::GRetrieverEngine)) translates a
//! knowledge graph plus a query into that view, runs the solver, and maps the
//! result back into a [`GRetrieverSubgraph`].

use thiserror::Error;

// ── GRetrieverError ───────────────────────────────────────────────────────────

/// Errors produced by the `g_retriever` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GRetrieverError {
    /// The query string was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,
    /// The knowledge graph contained no entities at all.
    #[error("knowledge graph must contain at least one entity")]
    EmptyGraph,
    /// No entity scored a prize above the configured floor, so there is no
    /// relevant node to anchor an unrooted retrieval around.
    #[error("no entity is relevant to the query (all prizes fell at or below the floor)")]
    NoRelevantNode,
    /// A relation referenced an entity id that is not present in the entity
    /// set.
    #[error("relation {index} references unknown entity id {endpoint:?}")]
    DanglingRelation {
        /// The position of the offending relation in the relation slice.
        index: usize,
        /// The endpoint id that could not be resolved.
        endpoint: String,
    },
    /// A rooted retrieval named a root entity id that is not present in the
    /// entity set.
    #[error("root entity id {entity_id:?} is not present in the knowledge graph")]
    RootNotFound {
        /// The requested root entity id.
        entity_id: String,
    },
    /// The configuration held an invalid value (e.g. a non-positive edge cost
    /// or a prize floor outside `[0, 1]`).
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// A human-readable explanation of what was invalid.
        reason: String,
    },
    /// The abstract PCST view handed to the solver was structurally invalid
    /// (e.g. an edge endpoint out of range or a non-finite prize/cost).
    #[error("invalid PCST graph: {reason}")]
    InvalidGraph {
        /// A human-readable explanation of what was invalid.
        reason: String,
    },
}

/// Convenience result alias for the `g_retriever` module.
pub type GRetrieverResult<T> = core::result::Result<T, GRetrieverError>;

// ── GRetrieverRootMode ────────────────────────────────────────────────────────

/// Whether the Prize-Collecting Steiner Tree is solved *unrooted* (the general
/// case, which may return a forest of disjoint trees) or *rooted* at a
/// mandated entity (which always returns a single tree containing that
/// entity).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum GRetrieverRootMode {
    /// Unrooted PCST: no vertex is forced into the solution. Clusters
    /// deactivate once their moat has paid off their prize, and the pruned
    /// result may consist of several disconnected components.
    #[default]
    Unrooted,
    /// Rooted PCST: the named entity's cluster never deactivates, so the
    /// solution is a single connected tree that always contains this entity —
    /// even when the entity itself scored no prize.
    Rooted {
        /// The id of the entity that must appear in the retrieved subgraph.
        entity_id: String,
    },
}

// ── GRetrieverConfig ──────────────────────────────────────────────────────────

/// Configuration for [`GRetrieverEngine`](super::engine::GRetrieverEngine).
#[derive(Debug, Clone, PartialEq)]
pub struct GRetrieverConfig {
    /// The base cost charged for including one relation (edge) in the
    /// subgraph. Must be strictly positive. Higher values discourage long
    /// connective paths, shrinking the retrieved subgraph. Defaults to `1.0`.
    pub edge_cost: f64,
    /// A multiplicative scale applied to every edge cost on top of
    /// [`edge_cost`](Self::edge_cost) and any per-relation weight. Must be
    /// strictly positive. Defaults to `1.0`.
    pub edge_cost_scale: f64,
    /// The relevance floor in `[0, 1]`: a node whose raw query relevance is
    /// strictly below this value is assigned a prize of `0` (it is treated as
    /// irrelevant and can only enter the subgraph as a Steiner waypoint).
    /// Defaults to `0.1`.
    pub prize_floor: f64,
    /// A multiplicative scale applied to every (above-floor) node prize. Must
    /// be strictly positive. Raising it makes relevant nodes worth reaching
    /// through more (or more expensive) edges. Defaults to `1.0`.
    pub prize_scale: f64,
    /// The dimensionality of the deterministic FNV-1a pseudo-embeddings used
    /// to score query relevance. Must be strictly positive. Defaults to `64`.
    pub embed_dim: usize,
    /// When `true`, relations are also assigned prizes (the relevance of the
    /// relation label to the query), realised via the classic PCST reduction
    /// that routes each prized edge through a virtual midpoint node — matching
    /// G-Retriever, which prizes both nodes and edges. Defaults to `false`
    /// (node prizes only).
    pub edge_prizes: bool,
    /// When `true`, the growth forest is refined by strong pruning; when
    /// `false`, the raw Goemans–Williamson growth forest is returned as-is.
    /// Defaults to `true`.
    pub prune: bool,
    /// When `true`, an unrooted retrieval returns only the single
    /// highest-net-value connected component; when `false`, every positive-net
    /// component is returned as a forest. Ignored in rooted mode (which is
    /// inherently single-tree). Defaults to `false`.
    pub single_component: bool,
    /// Whether the PCST is solved unrooted or rooted at a mandated entity.
    /// Defaults to [`GRetrieverRootMode::Unrooted`].
    pub root_mode: GRetrieverRootMode,
    /// An optional cap on the number of entities in the returned subgraph.
    /// When the pruned subgraph exceeds this many entities, the
    /// lowest-prize leaves are trimmed until the cap is met (preserving
    /// connectivity). `None` means unbounded. Defaults to `None`.
    pub max_subgraph_size: Option<usize>,
}

impl Default for GRetrieverConfig {
    fn default() -> Self {
        Self {
            edge_cost: 1.0,
            edge_cost_scale: 1.0,
            prize_floor: 0.1,
            prize_scale: 1.0,
            embed_dim: 64,
            edge_prizes: false,
            prune: true,
            single_component: false,
            root_mode: GRetrieverRootMode::Unrooted,
            max_subgraph_size: None,
        }
    }
}

impl GRetrieverConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base per-edge cost.
    #[must_use]
    pub fn with_edge_cost(mut self, edge_cost: f64) -> Self {
        self.edge_cost = edge_cost;
        self
    }

    /// Set the multiplicative edge-cost scale.
    #[must_use]
    pub fn with_edge_cost_scale(mut self, edge_cost_scale: f64) -> Self {
        self.edge_cost_scale = edge_cost_scale;
        self
    }

    /// Set the relevance floor below which a node's prize is zeroed.
    #[must_use]
    pub fn with_prize_floor(mut self, prize_floor: f64) -> Self {
        self.prize_floor = prize_floor;
        self
    }

    /// Set the multiplicative node-prize scale.
    #[must_use]
    pub fn with_prize_scale(mut self, prize_scale: f64) -> Self {
        self.prize_scale = prize_scale;
        self
    }

    /// Set the pseudo-embedding dimensionality.
    #[must_use]
    pub fn with_embed_dim(mut self, embed_dim: usize) -> Self {
        self.embed_dim = embed_dim;
        self
    }

    /// Enable or disable edge prizes.
    #[must_use]
    pub fn with_edge_prizes(mut self, edge_prizes: bool) -> Self {
        self.edge_prizes = edge_prizes;
        self
    }

    /// Enable or disable strong pruning of the growth forest.
    #[must_use]
    pub fn with_prune(mut self, prune: bool) -> Self {
        self.prune = prune;
        self
    }

    /// Enable or disable single-component (best-tree-only) unrooted output.
    #[must_use]
    pub fn with_single_component(mut self, single_component: bool) -> Self {
        self.single_component = single_component;
        self
    }

    /// Set the root mode (unrooted or rooted at a named entity).
    #[must_use]
    pub fn with_root_mode(mut self, root_mode: GRetrieverRootMode) -> Self {
        self.root_mode = root_mode;
        self
    }

    /// Set the maximum number of entities in the returned subgraph.
    #[must_use]
    pub fn with_max_subgraph_size(mut self, max_subgraph_size: Option<usize>) -> Self {
        self.max_subgraph_size = max_subgraph_size;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GRetrieverError::InvalidConfig`] when any field holds an
    /// out-of-range or non-finite value.
    pub fn validate(&self) -> GRetrieverResult<()> {
        if !self.edge_cost.is_finite() || self.edge_cost <= 0.0 {
            return Err(GRetrieverError::InvalidConfig {
                reason: format!(
                    "edge_cost must be finite and positive, got {}",
                    self.edge_cost
                ),
            });
        }
        if !self.edge_cost_scale.is_finite() || self.edge_cost_scale <= 0.0 {
            return Err(GRetrieverError::InvalidConfig {
                reason: format!(
                    "edge_cost_scale must be finite and positive, got {}",
                    self.edge_cost_scale
                ),
            });
        }
        if !self.prize_scale.is_finite() || self.prize_scale <= 0.0 {
            return Err(GRetrieverError::InvalidConfig {
                reason: format!(
                    "prize_scale must be finite and positive, got {}",
                    self.prize_scale
                ),
            });
        }
        if !self.prize_floor.is_finite() || !(0.0..=1.0).contains(&self.prize_floor) {
            return Err(GRetrieverError::InvalidConfig {
                reason: format!(
                    "prize_floor must be finite and in [0, 1], got {}",
                    self.prize_floor
                ),
            });
        }
        if self.embed_dim == 0 {
            return Err(GRetrieverError::InvalidConfig {
                reason: "embed_dim must be strictly positive".to_string(),
            });
        }
        if self.max_subgraph_size == Some(0) {
            return Err(GRetrieverError::InvalidConfig {
                reason: "max_subgraph_size must be None or a positive value".to_string(),
            });
        }
        Ok(())
    }
}

// ── GRetrieverEntity ──────────────────────────────────────────────────────────

/// A node of the input knowledge graph: an entity with a stable id and a
/// free-text description that is scored for query relevance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GRetrieverEntity {
    /// The stable identifier used by relations to reference this entity.
    pub id: String,
    /// The textual description of the entity, embedded to score relevance.
    pub text: String,
}

impl GRetrieverEntity {
    /// Create a new entity from an id and its textual description.
    #[must_use]
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

// ── GRetrieverRelation ────────────────────────────────────────────────────────

/// An edge of the input knowledge graph: a labelled relation between two
/// entities, optionally carrying a per-relation cost weight.
#[derive(Debug, Clone, PartialEq)]
pub struct GRetrieverRelation {
    /// The id of the source entity.
    pub source_id: String,
    /// The id of the target entity.
    pub target_id: String,
    /// A human-readable relation label (also used to score edge relevance
    /// when edge prizes are enabled).
    pub label: String,
    /// An optional multiplicative weight on this relation's cost. `None` is
    /// treated as `1.0`. Values above `1.0` make the relation more expensive
    /// to include; values below `1.0` make it cheaper.
    pub weight: Option<f64>,
}

impl GRetrieverRelation {
    /// Create a new relation with unit weight.
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            target_id: target_id.into(),
            label: label.into(),
            weight: None,
        }
    }

    /// Attach a per-relation cost weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = Some(weight);
        self
    }
}

// ── GRetrieverSubgraph ────────────────────────────────────────────────────────

/// The subgraph retrieved by [`GRetrieverEngine`](super::engine::GRetrieverEngine):
/// the selected entities and relations together with the collected-prize /
/// paid-cost summary.
#[derive(Debug, Clone, PartialEq)]
pub struct GRetrieverSubgraph {
    /// The selected entities, in the order chosen by the engine.
    pub entities: Vec<GRetrieverEntity>,
    /// The prize assigned to each entity in [`entities`](Self::entities),
    /// index-aligned with it.
    pub node_prizes: Vec<f64>,
    /// The selected relations connecting the entities.
    pub relations: Vec<GRetrieverRelation>,
    /// The total prize collected: the sum of the selected node prizes plus any
    /// selected edge prizes.
    pub total_prize: f64,
    /// The total cost paid: the sum of the selected relations' full edge
    /// costs.
    pub total_cost: f64,
    /// The number of connected components in the retrieved subgraph (`1` for a
    /// single connected tree; more when a forest is returned).
    pub num_components: usize,
}

impl GRetrieverSubgraph {
    /// The net value of the subgraph: `total_prize - total_cost`.
    #[must_use]
    pub fn net_value(&self) -> f64 {
        self.total_prize - self.total_cost
    }

    /// The number of entities in the subgraph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Returns `true` when the subgraph contains no entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Returns `true` when the subgraph is a single connected component.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.num_components <= 1
    }

    /// Returns `true` when an entity with `id` is present in the subgraph.
    #[must_use]
    pub fn contains_entity(&self, id: &str) -> bool {
        self.entities.iter().any(|entity| entity.id == id)
    }

    /// Render the subgraph as a plain-text block suitable for prompting a
    /// downstream language model: a `Nodes:` section listing `id: text` and an
    /// `Edges:` section listing `source -[label]-> target`.
    #[must_use]
    pub fn textualize(&self) -> String {
        let mut out = String::from("Nodes:\n");
        for entity in &self.entities {
            out.push_str("- ");
            out.push_str(&entity.id);
            out.push_str(": ");
            out.push_str(&entity.text);
            out.push('\n');
        }
        out.push_str("Edges:\n");
        for relation in &self.relations {
            out.push_str("- ");
            out.push_str(&relation.source_id);
            out.push_str(" -[");
            out.push_str(&relation.label);
            out.push_str("]-> ");
            out.push_str(&relation.target_id);
            out.push('\n');
        }
        out
    }
}

// ── PcstNode ──────────────────────────────────────────────────────────────────

/// A node in the abstract Prize-Collecting Steiner Tree view: an index-like id
/// paired with a nonnegative prize.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcstNode {
    /// The node's zero-based id, equal to its position in the solver's node
    /// slice.
    pub id: usize,
    /// The node's nonnegative prize (query relevance). Negative or non-finite
    /// values are clamped to `0` by the solver.
    pub prize: f64,
}

impl PcstNode {
    /// Create a new node from an id and a prize.
    #[must_use]
    pub fn new(id: usize, prize: f64) -> Self {
        Self { id, prize }
    }
}

// ── PcstEdge ──────────────────────────────────────────────────────────────────

/// An undirected edge in the abstract Prize-Collecting Steiner Tree view: a
/// pair of endpoint node ids and a nonnegative cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcstEdge {
    /// The first endpoint's node id.
    pub source: usize,
    /// The second endpoint's node id.
    pub target: usize,
    /// The nonnegative cost of including this edge. Negative or non-finite
    /// values are clamped to `0` by the solver.
    pub cost: f64,
}

impl PcstEdge {
    /// Create a new edge from two endpoint ids and a cost.
    #[must_use]
    pub fn new(source: usize, target: usize, cost: f64) -> Self {
        Self {
            source,
            target,
            cost,
        }
    }

    /// Return the endpoint of this edge that is not `node`, or `None` when
    /// `node` is not an endpoint.
    #[must_use]
    pub fn other(&self, node: usize) -> Option<usize> {
        if self.source == node {
            Some(self.target)
        } else if self.target == node {
            Some(self.source)
        } else {
            None
        }
    }
}

// ── PcstForest ────────────────────────────────────────────────────────────────

/// A forest over an abstract PCST graph: the set of retained nodes and the set
/// of retained edges (by index into the solver's edge slice).
///
/// The Goemans–Williamson growth phase and the strong-pruning phase both
/// produce a [`PcstForest`]; a growth forest records the connective structure
/// discovered by moat growth, while a pruned forest records the final
/// prize-maximising subgraph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PcstForest {
    /// The number of nodes in the underlying graph (the solver's node count),
    /// used to bound node indices.
    pub node_count: usize,
    /// The retained node ids, sorted ascending and deduplicated.
    pub node_indices: Vec<usize>,
    /// The retained edge indices (into the solver's edge slice), sorted
    /// ascending.
    pub edge_indices: Vec<usize>,
}

impl PcstForest {
    /// Create a forest from a node count and retained node/edge index sets.
    /// The index sets are sorted and deduplicated.
    #[must_use]
    pub fn new(
        node_count: usize,
        mut node_indices: Vec<usize>,
        mut edge_indices: Vec<usize>,
    ) -> Self {
        node_indices.sort_unstable();
        node_indices.dedup();
        edge_indices.sort_unstable();
        edge_indices.dedup();
        Self {
            node_count,
            node_indices,
            edge_indices,
        }
    }

    /// Returns `true` when the forest retains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_indices.is_empty()
    }

    /// The number of retained nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.node_indices.len()
    }

    /// The retained node ids.
    #[must_use]
    pub fn selected_nodes(&self) -> &[usize] {
        &self.node_indices
    }

    /// The retained edge indices.
    #[must_use]
    pub fn selected_edges(&self) -> &[usize] {
        &self.edge_indices
    }

    /// The total cost of the retained edges.
    #[must_use]
    pub fn total_cost(&self, edges: &[PcstEdge]) -> f64 {
        self.edge_indices
            .iter()
            .filter_map(|&ei| edges.get(ei))
            .map(|edge| edge.cost.max(0.0))
            .sum()
    }

    /// The total prize of the retained nodes.
    #[must_use]
    pub fn total_prize(&self, nodes: &[PcstNode]) -> f64 {
        self.node_indices
            .iter()
            .filter_map(|&ni| nodes.get(ni))
            .map(|node| node.prize.max(0.0))
            .sum()
    }

    /// The net value of the forest: total node prize minus total edge cost.
    #[must_use]
    pub fn net_value(&self, nodes: &[PcstNode], edges: &[PcstEdge]) -> f64 {
        self.total_prize(nodes) - self.total_cost(edges)
    }

    /// The connected components of the forest, as sorted lists of node ids.
    ///
    /// Components are induced by the retained edges over the retained nodes; a
    /// retained node with no retained incident edge forms its own singleton
    /// component. The returned components are ordered by their smallest node
    /// id.
    #[must_use]
    pub fn components(&self, edges: &[PcstEdge]) -> Vec<Vec<usize>> {
        if self.node_indices.is_empty() {
            return Vec::new();
        }
        // Map node id -> local position in `node_indices`.
        let mut position: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(self.node_indices.len());
        for (pos, &node) in self.node_indices.iter().enumerate() {
            position.insert(node, pos);
        }

        let mut union = UnionFind::new(self.node_indices.len());
        for &ei in &self.edge_indices {
            let Some(edge) = edges.get(ei) else {
                continue;
            };
            if let (Some(&pa), Some(&pb)) = (position.get(&edge.source), position.get(&edge.target))
            {
                union.union(pa, pb);
            }
        }

        let mut buckets: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (pos, &node) in self.node_indices.iter().enumerate() {
            let root = union.find(pos);
            buckets.entry(root).or_default().push(node);
        }

        let mut components: Vec<Vec<usize>> = buckets
            .into_values()
            .map(|mut comp| {
                comp.sort_unstable();
                comp
            })
            .collect();
        components.sort_by_key(|comp| comp.first().copied().unwrap_or(usize::MAX));
        components
    }
}

// ── UnionFind (shared, private) ───────────────────────────────────────────────

/// A minimal disjoint-set (union-find) structure with path halving and union
/// by rank, used by both the solver and [`PcstForest::components`].
#[derive(Debug, Clone)]
pub(super) struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    /// Create a union-find over `n` singleton elements `0..n`.
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the representative of `x`, applying path halving.
    pub(super) fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Union the sets containing `a` and `b`, returning the representative of
    /// the merged set.
    pub(super) fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => {
                self.parent[ra] = rb;
                rb
            }
            std::cmp::Ordering::Greater => {
                self.parent[rb] = ra;
                ra
            }
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
                ra
            }
        }
    }
}
