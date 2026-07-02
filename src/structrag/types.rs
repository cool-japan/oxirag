//! Types, configuration, and errors for the `structrag` module.
//!
//! Five concrete knowledge structures are modelled: [`StructRagTable`],
//! [`StructRagGraph`], [`StructRagTree`], [`StructRagCatalogue`], and
//! [`StructRagAlgorithm`], unified behind [`StructRagKnowledgeStructure`].
//! [`StructRagStructureKind`] names the five kinds; [`StructRagRoutingDecision`]
//! is the output of [`crate::structrag::StructRagRouter::route`], and
//! [`StructRagResult`] is the output of [`crate::structrag::StructRagEngine::run`].

use std::collections::VecDeque;
use std::fmt;

use thiserror::Error;

// ── StructRagPassage ────────────────────────────────────────────────────────

/// A single retrieved passage supplied to [`crate::structrag::StructRagEngine`]
/// for restructuring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructRagPassage {
    /// Caller-assigned identifier (e.g. a document or chunk id). Propagated
    /// into every knowledge-structure element derived from this passage, so a
    /// downstream consumer can trace any table cell, graph edge, tree node,
    /// catalogue item, or algorithm step back to its source text.
    pub id: String,
    /// The passage's raw text.
    pub text: String,
}

impl StructRagPassage {
    /// Create a new passage.
    #[must_use]
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

// ── StructRagStructureKind ──────────────────────────────────────────────────

/// The kind of knowledge structure that retrieved passages can be
/// restructured into.
///
/// [`crate::structrag::StructRagRouter`] infers which kind best fits a query;
/// [`crate::structrag::StructRagRestructurer`] then builds the corresponding
/// concrete structure ([`StructRagTable`], [`StructRagGraph`],
/// [`StructRagTree`], [`StructRagCatalogue`], or [`StructRagAlgorithm`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructRagStructureKind {
    /// Rows and columns — best for comparison and statistical queries across
    /// several items (e.g. "which product is cheapest?").
    Table,
    /// Entities and labelled edges — best for relationship/connection
    /// queries (e.g. "how is X related to Y?").
    Graph,
    /// A parent/child hierarchy — best for taxonomy, classification, and
    /// decomposition queries (e.g. "what is the hierarchy of X?").
    Tree,
    /// A keyed list of items with (possibly heterogeneous) attributes — best
    /// for enumeration/inventory queries (e.g. "list all the X").
    Catalogue,
    /// An ordered sequence of steps — best for procedural queries (e.g. "how
    /// do I set up X, step by step?").
    Algorithm,
}

impl StructRagStructureKind {
    /// Stable lowercase string identifier.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Graph => "graph",
            Self::Tree => "tree",
            Self::Catalogue => "catalogue",
            Self::Algorithm => "algorithm",
        }
    }

    /// All five structure kinds, in a fixed canonical order. Used as the
    /// default routing tie-break order and for exhaustive iteration.
    #[must_use]
    pub fn all() -> [Self; 5] {
        [
            Self::Table,
            Self::Graph,
            Self::Tree,
            Self::Catalogue,
            Self::Algorithm,
        ]
    }
}

impl fmt::Display for StructRagStructureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── StructRagTable ───────────────────────────────────────────────────────────

/// A single row of a [`StructRagTable`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StructRagTableRow {
    /// Cell values, aligned positionally with [`StructRagTable::columns`].
    /// An empty string means the column's value was not extractable from
    /// this row's source passage.
    pub cells: Vec<String>,
    /// Id of the passage this row was derived from.
    pub source_passage_id: String,
}

/// Rows and columns extracted from retrieved passages — the structure chosen
/// for comparison and statistical queries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StructRagTable {
    /// Column names, in first-appearance order across the source passages.
    pub columns: Vec<String>,
    /// Table rows, one per contributing passage.
    pub rows: Vec<StructRagTableRow>,
}

impl StructRagTable {
    /// Index of `column` (case-insensitive), or `None` if absent.
    #[must_use]
    pub fn column_index(&self, column: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c.eq_ignore_ascii_case(column))
    }

    /// The value of `column` in `row`, or `None` if either is out of range.
    #[must_use]
    pub fn cell(&self, row: usize, column: &str) -> Option<&str> {
        let idx = self.column_index(column)?;
        self.rows.get(row)?.cells.get(idx).map(String::as_str)
    }

    /// Parse `column` as a numeric column: one [`Option<f64>`] per row, in
    /// row order, `None` where the cell is empty or not numeric. Returns an
    /// empty vector if `column` does not exist.
    #[must_use]
    pub fn numeric_column(&self, column: &str) -> Vec<Option<f64>> {
        let Some(idx) = self.column_index(column) else {
            return Vec::new();
        };
        self.rows
            .iter()
            .map(|r| r.cells.get(idx).and_then(|c| parse_numeric_cell(c)))
            .collect()
    }
}

/// Parse a table cell as a floating-point number, tolerating a leading
/// currency symbol and a trailing `%`.
fn parse_numeric_cell(cell: &str) -> Option<f64> {
    let cleaned: String = cell
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    if cleaned.is_empty() || cleaned == "-" {
        None
    } else {
        cleaned.parse::<f64>().ok()
    }
}

// ── StructRagGraph ───────────────────────────────────────────────────────────

/// A single entity node in a [`StructRagGraph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructRagGraphNode {
    /// Stable identifier: the entity label, trimmed and lower-cased.
    pub id: String,
    /// Display label, in the original case of its first occurrence.
    pub label: String,
    /// Number of times this entity was mentioned across all passages.
    pub mentions: usize,
}

/// A single directed, labelled edge in a [`StructRagGraph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructRagGraphEdge {
    /// Id of the source node.
    pub source: String,
    /// Id of the target node.
    pub target: String,
    /// The relation label connecting `source` to `target` (e.g.
    /// `"founded_by"`, or `"related_to"` when no specific connective text
    /// was found between the two entity mentions).
    pub relation: String,
    /// Id of the passage this edge was derived from.
    pub source_passage_id: String,
}

/// Entities and relations extracted from retrieved passages — the structure
/// chosen for relationship and connection queries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StructRagGraph {
    /// Distinct entities mentioned in the source passages.
    pub nodes: Vec<StructRagGraphNode>,
    /// Relations between entities.
    pub edges: Vec<StructRagGraphEdge>,
}

impl StructRagGraph {
    /// Find a node by label (case-insensitive, whitespace-trimmed).
    #[must_use]
    pub fn find_node(&self, label: &str) -> Option<&StructRagGraphNode> {
        let key = label.trim().to_lowercase();
        self.nodes.iter().find(|n| n.id == key)
    }

    /// Edges whose source is `node_id`.
    #[must_use]
    pub fn edges_from(&self, node_id: &str) -> Vec<&StructRagGraphEdge> {
        self.edges.iter().filter(|e| e.source == node_id).collect()
    }

    /// Ids of nodes directly connected to `node_id`, treating edges as
    /// undirected for reachability purposes. May contain duplicates when
    /// more than one edge connects the same pair.
    #[must_use]
    pub fn neighbors(&self, node_id: &str) -> Vec<String> {
        let mut out = Vec::new();
        for edge in &self.edges {
            if edge.source == node_id {
                out.push(edge.target.clone());
            } else if edge.target == node_id {
                out.push(edge.source.clone());
            }
        }
        out
    }
}

// ── StructRagTree ────────────────────────────────────────────────────────────

/// A single node in a [`StructRagTree`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructRagTreeNode {
    /// Zero-based identifier, stable within the owning [`StructRagTree`].
    pub id: usize,
    /// Display label.
    pub label: String,
    /// Id of the parent node, or `None` for a root.
    pub parent: Option<usize>,
    /// Ids of direct children, in the order they were discovered.
    pub children: Vec<usize>,
    /// Distance from the nearest root (`0` for roots).
    pub depth: usize,
    /// Id of the passage this node was derived from.
    pub source_passage_id: String,
}

/// A parent/child hierarchy extracted from retrieved passages — the
/// structure chosen for taxonomy, classification, and decomposition
/// queries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StructRagTree {
    /// All nodes in the tree (in fact a forest: more than one root is
    /// possible), indexed by [`StructRagTreeNode::id`].
    pub nodes: Vec<StructRagTreeNode>,
    /// Ids of nodes with no parent, ascending.
    pub root_ids: Vec<usize>,
}

impl StructRagTree {
    /// Find a node by label (case-insensitive, whitespace-trimmed).
    #[must_use]
    pub fn find(&self, label: &str) -> Option<&StructRagTreeNode> {
        let key = label.trim().to_lowercase();
        self.nodes
            .iter()
            .find(|n| n.label.trim().to_lowercase() == key)
    }

    /// Ids of `id`'s ancestors, ordered from the root down to (but
    /// excluding) `id` itself. Empty when `id` is a root or unknown.
    #[must_use]
    pub fn ancestors(&self, id: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut current = self
            .nodes
            .iter()
            .find(|n| n.id == id)
            .and_then(|n| n.parent);
        // Bounded by node count: a well-formed tree cannot have a longer
        // ancestor chain, and this guards against a malformed cycle.
        for _ in 0..self.nodes.len() {
            let Some(pid) = current else { break };
            out.push(pid);
            current = self
                .nodes
                .iter()
                .find(|n| n.id == pid)
                .and_then(|n| n.parent);
        }
        out.reverse();
        out
    }

    /// Ids of all of `id`'s descendants (children, grandchildren, ...), in
    /// breadth-first order.
    #[must_use]
    pub fn descendants(&self, id: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(id);
        while let Some(current) = queue.pop_front() {
            if let Some(node) = self.nodes.iter().find(|n| n.id == current) {
                for &child in &node.children {
                    out.push(child);
                    queue.push_back(child);
                }
            }
        }
        out
    }
}

// ── StructRagCatalogue ───────────────────────────────────────────────────────

/// A single entry in a [`StructRagCatalogue`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructRagCatalogueItem {
    /// The item's key (e.g. its name).
    pub key: String,
    /// Ordered `(attribute name, value)` pairs. Unlike [`StructRagTable`]
    /// rows, items are not required to share the same attribute set.
    pub attributes: Vec<(String, String)>,
    /// Id of the passage this item was derived from.
    pub source_passage_id: String,
}

/// A keyed list of items with (possibly heterogeneous) attributes —
/// extracted from retrieved passages, the structure chosen for enumeration
/// and inventory queries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StructRagCatalogue {
    /// The catalogue's items, one per contributing passage.
    pub items: Vec<StructRagCatalogueItem>,
}

impl StructRagCatalogue {
    /// Find an item by key (case-insensitive, whitespace-trimmed).
    #[must_use]
    pub fn find(&self, key: &str) -> Option<&StructRagCatalogueItem> {
        let needle = key.trim().to_lowercase();
        self.items
            .iter()
            .find(|i| i.key.trim().to_lowercase() == needle)
    }
}

// ── StructRagAlgorithm ───────────────────────────────────────────────────────

/// A single step in a [`StructRagAlgorithm`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructRagStep {
    /// One-based position of this step in the procedure.
    pub index: usize,
    /// The step's action text.
    pub action: String,
    /// Id of the passage this step was derived from.
    pub source_passage_id: String,
}

/// An ordered sequence of steps extracted from retrieved passages — the
/// structure chosen for procedural, step-by-step queries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StructRagAlgorithm {
    /// The procedure's steps, in execution order.
    pub steps: Vec<StructRagStep>,
}

// ── StructRagKnowledgeStructure ─────────────────────────────────────────────

/// The concrete knowledge structure built by
/// [`crate::structrag::StructRagRestructurer`] for a chosen
/// [`StructRagStructureKind`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructRagKnowledgeStructure {
    /// A [`StructRagTable`].
    Table(StructRagTable),
    /// A [`StructRagGraph`].
    Graph(StructRagGraph),
    /// A [`StructRagTree`].
    Tree(StructRagTree),
    /// A [`StructRagCatalogue`].
    Catalogue(StructRagCatalogue),
    /// A [`StructRagAlgorithm`].
    Algorithm(StructRagAlgorithm),
}

impl StructRagKnowledgeStructure {
    /// The [`StructRagStructureKind`] this structure was built as.
    #[must_use]
    pub fn kind(&self) -> StructRagStructureKind {
        match self {
            Self::Table(_) => StructRagStructureKind::Table,
            Self::Graph(_) => StructRagStructureKind::Graph,
            Self::Tree(_) => StructRagStructureKind::Tree,
            Self::Catalogue(_) => StructRagStructureKind::Catalogue,
            Self::Algorithm(_) => StructRagStructureKind::Algorithm,
        }
    }

    /// Number of primary elements the structure holds: rows for a table,
    /// nodes for a graph or tree, items for a catalogue, steps for an
    /// algorithm.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Table(t) => t.rows.len(),
            Self::Graph(g) => g.nodes.len(),
            Self::Tree(t) => t.nodes.len(),
            Self::Catalogue(c) => c.items.len(),
            Self::Algorithm(a) => a.steps.len(),
        }
    }

    /// `true` when [`StructRagKnowledgeStructure::len`] is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the inner [`StructRagTable`], or `None` if this is a different
    /// kind.
    #[must_use]
    pub fn as_table(&self) -> Option<&StructRagTable> {
        match self {
            Self::Table(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow the inner [`StructRagGraph`], or `None` if this is a different
    /// kind.
    #[must_use]
    pub fn as_graph(&self) -> Option<&StructRagGraph> {
        match self {
            Self::Graph(g) => Some(g),
            _ => None,
        }
    }

    /// Borrow the inner [`StructRagTree`], or `None` if this is a different
    /// kind.
    #[must_use]
    pub fn as_tree(&self) -> Option<&StructRagTree> {
        match self {
            Self::Tree(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow the inner [`StructRagCatalogue`], or `None` if this is a
    /// different kind.
    #[must_use]
    pub fn as_catalogue(&self) -> Option<&StructRagCatalogue> {
        match self {
            Self::Catalogue(c) => Some(c),
            _ => None,
        }
    }

    /// Borrow the inner [`StructRagAlgorithm`], or `None` if this is a
    /// different kind.
    #[must_use]
    pub fn as_algorithm(&self) -> Option<&StructRagAlgorithm> {
        match self {
            Self::Algorithm(a) => Some(a),
            _ => None,
        }
    }
}

// ── StructRagRoutingDecision ────────────────────────────────────────────────

/// The output of [`crate::structrag::StructRagRouter::route`]: the chosen
/// knowledge-structure kind, a confidence score, a human-readable rationale,
/// and the full per-kind score breakdown (for transparency and debugging).
#[derive(Debug, Clone, PartialEq)]
pub struct StructRagRoutingDecision {
    /// The chosen structure kind.
    pub kind: StructRagStructureKind,
    /// Confidence in `kind`, in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Human-readable explanation of the routing decision: which cue phrases
    /// matched, the resulting confidence, and whether a fallback rule (low
    /// confidence or a disabled top choice) was applied.
    pub rationale: String,
    /// Every kind's score, descending, ties broken by
    /// [`StructRagConfig::tie_break_order`]. Always has one entry per
    /// [`StructRagStructureKind::all`] variant.
    pub scores: Vec<(StructRagStructureKind, f32)>,
}

impl StructRagRoutingDecision {
    /// The score computed for `kind`, or `0.0` if absent from
    /// [`StructRagRoutingDecision::scores`] (should not happen in practice).
    #[must_use]
    pub fn score_of(&self, kind: StructRagStructureKind) -> f32 {
        self.scores
            .iter()
            .find(|(k, _)| *k == kind)
            .map_or(0.0, |(_, s)| *s)
    }
}

// ── StructRagResult ──────────────────────────────────────────────────────────

/// The complete output of [`crate::structrag::StructRagEngine::run`]: the
/// original query, the routing decision, the restructured knowledge (kept
/// for transparency/inspection), and the reasoner's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct StructRagResult {
    /// The original query (trimmed).
    pub query: String,
    /// How the query was routed to [`StructRagResult::structure`]'s kind.
    pub routing: StructRagRoutingDecision,
    /// The knowledge structure the retrieved passages were restructured
    /// into.
    pub structure: StructRagKnowledgeStructure,
    /// The reasoner's answer, produced by reading over `structure`.
    pub answer: String,
}

// ── StructRagConfig ──────────────────────────────────────────────────────────

/// Configuration for the `structrag` pipeline: which structure kinds are
/// available to the router, the router's confidence handling, the structure
/// size cap, and the tie-break order.
#[derive(Debug, Clone, PartialEq)]
pub struct StructRagConfig {
    /// Structure kinds the router is allowed to choose. Must be non-empty
    /// for [`crate::structrag::StructRagRouter::route`] to succeed. Defaults
    /// to all five kinds ([`StructRagStructureKind::all`]).
    pub enabled_kinds: Vec<StructRagStructureKind>,
    /// Minimum confidence (in `[0.0, 1.0]`) the top enabled kind must reach;
    /// below this, [`StructRagConfig::default_kind`] is used instead (when
    /// it is itself enabled). Defaults to `0.15`.
    pub min_confidence: f32,
    /// The structure kind used when the top enabled kind's confidence falls
    /// below [`StructRagConfig::min_confidence`]. Defaults to
    /// [`StructRagStructureKind::Catalogue`], a generally-applicable
    /// fallback for queries with no strong structural cue.
    pub default_kind: StructRagStructureKind,
    /// Maximum number of primary elements (table rows, graph nodes,
    /// tree nodes, catalogue items, algorithm steps) a restructured
    /// [`StructRagKnowledgeStructure`] may contain; excess elements are
    /// deterministically truncated. Defaults to `256`.
    pub max_structure_size: usize,
    /// Preference order used to break exact score ties between structure
    /// kinds, most-preferred first. Kinds absent from this list rank after
    /// every listed kind, in [`StructRagStructureKind::all`] order. Defaults
    /// to [`StructRagStructureKind::all`].
    pub tie_break_order: Vec<StructRagStructureKind>,
}

impl Default for StructRagConfig {
    fn default() -> Self {
        Self {
            enabled_kinds: StructRagStructureKind::all().to_vec(),
            min_confidence: 0.15,
            default_kind: StructRagStructureKind::Catalogue,
            max_structure_size: 256,
            tie_break_order: StructRagStructureKind::all().to_vec(),
        }
    }
}

impl StructRagConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the set of enabled structure kinds.
    #[must_use]
    pub fn with_enabled_kinds(mut self, kinds: Vec<StructRagStructureKind>) -> Self {
        self.enabled_kinds = kinds;
        self
    }

    /// Remove a single kind from the enabled set.
    #[must_use]
    pub fn with_disabled_kind(mut self, kind: StructRagStructureKind) -> Self {
        self.enabled_kinds.retain(|k| *k != kind);
        self
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set the fallback kind used below the confidence threshold.
    #[must_use]
    pub fn with_default_kind(mut self, default_kind: StructRagStructureKind) -> Self {
        self.default_kind = default_kind;
        self
    }

    /// Set the maximum structure size.
    #[must_use]
    pub fn with_max_structure_size(mut self, max_structure_size: usize) -> Self {
        self.max_structure_size = max_structure_size;
        self
    }

    /// Replace the tie-break order.
    #[must_use]
    pub fn with_tie_break_order(mut self, tie_break_order: Vec<StructRagStructureKind>) -> Self {
        self.tie_break_order = tie_break_order;
        self
    }

    /// `true` when `kind` is a member of [`StructRagConfig::enabled_kinds`].
    #[must_use]
    pub fn is_enabled(&self, kind: StructRagStructureKind) -> bool {
        self.enabled_kinds.contains(&kind)
    }
}

// ── StructRagError ───────────────────────────────────────────────────────────

/// Errors from the `structrag` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StructRagError {
    /// The query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,
    /// No passages were supplied to restructure.
    #[error("no passages supplied")]
    EmptyPassages,
    /// [`StructRagConfig::enabled_kinds`] was empty, so no structure kind
    /// was available for the router to choose.
    #[error("no structure kinds are enabled in the configuration")]
    NoEnabledStructureKinds,
    /// A specific structure kind was requested (bypassing the router) but is
    /// disabled in the configuration.
    #[error("structure kind '{0}' is disabled in the configuration")]
    DisabledStructureKind(StructRagStructureKind),
}
