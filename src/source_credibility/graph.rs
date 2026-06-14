//! The citation graph and its `PageRank` implementation.
//!
//! A [`SourceGraph`] is a directed graph whose nodes are [`DocumentId`]s and
//! whose edges represent **citations** (`from` cites `to`).  `PageRank` over this
//! graph yields an authority score per document: documents cited by many
//! (themselves authoritative) documents score higher.
//!
//! The `PageRank` implementation is the standard iterative power method with
//! explicit handling of **dangling nodes** (nodes with no outgoing citation),
//! whose rank mass is redistributed uniformly across all nodes so that the
//! score vector continues to sum to `1.0`.

use std::collections::{HashMap, HashSet};

use super::types::SourceCredibilityError;
use crate::types::DocumentId;

/// A directed citation graph over documents.
///
/// Edges point **from** the citing document **to** the cited document.
#[derive(Debug, Clone, Default)]
pub struct SourceGraph {
    /// Outgoing adjacency: citing document → documents it cites.
    adjacency: HashMap<DocumentId, Vec<DocumentId>>,
    /// All known nodes (citing *and* cited documents).
    nodes: HashSet<DocumentId>,
}

impl SourceGraph {
    /// Create an empty citation graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    /// Register a node without any citations.
    ///
    /// Idempotent: adding an existing node is a no-op.
    pub fn add_node(&mut self, id: DocumentId) {
        self.nodes.insert(id);
    }

    /// Record that `from` cites `to`.
    ///
    /// Both endpoints are registered as nodes.  Duplicate citations are kept
    /// (they increase the out-degree, matching real citation multiplicity) but
    /// self-citations are ignored, as they carry no authority signal.
    pub fn add_citation(&mut self, from: DocumentId, to: DocumentId) {
        self.nodes.insert(from.clone());
        self.nodes.insert(to.clone());
        if from == to {
            return;
        }
        self.adjacency.entry(from).or_default().push(to);
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of citation edges in the graph (counting multiplicity).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(Vec::len).sum()
    }

    /// Return `true` when the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Outgoing citations from `id`, if any are recorded.
    #[must_use]
    pub fn citations_from(&self, id: &DocumentId) -> Option<&[DocumentId]> {
        self.adjacency.get(id).map(Vec::as_slice)
    }

    /// Compute `PageRank` over the citation graph.
    ///
    /// Standard iterative power method:
    ///
    /// `r_i = (1 - d) / N + d * (Σ_{j → i} r_j / outdeg(j) + dangling_mass / N)`
    ///
    /// where `dangling_mass` is the total rank held by nodes with no outgoing
    /// edges, redistributed uniformly.  The returned scores sum to `≈ 1.0`.
    ///
    /// An empty graph yields an empty map.  `damping` is clamped to `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pagerank(&self, damping: f32, iterations: usize) -> HashMap<DocumentId, f32> {
        let n = self.nodes.len();
        if n == 0 {
            return HashMap::new();
        }

        let damping = damping.clamp(0.0, 1.0);
        let n_f = n as f32;
        let teleport = (1.0 - damping) / n_f;

        // Stable node order so iteration / dangling handling is deterministic.
        let mut node_list: Vec<&DocumentId> = self.nodes.iter().collect();
        node_list.sort_by(|a, b| a.as_str().cmp(b.as_str()));

        // Out-degree per node (after self-loop filtering already done at insert).
        let out_degree: HashMap<&DocumentId, usize> = node_list
            .iter()
            .map(|&id| {
                let deg = self.adjacency.get(id).map_or(0, Vec::len);
                (id, deg)
            })
            .collect();

        // Reverse adjacency contributions: for node i, the list of (citing node j,
        // outdeg(j)) so we can pull rank mass each iteration.
        let mut incoming: HashMap<&DocumentId, Vec<&DocumentId>> =
            node_list.iter().map(|&id| (id, Vec::new())).collect();
        for (from, targets) in &self.adjacency {
            for to in targets {
                // `to` is always a node, `from` is always a node.
                if let Some(list) = incoming.get_mut(to) {
                    list.push(from);
                }
            }
        }

        // Initialise uniformly.
        let mut rank: HashMap<&DocumentId, f32> =
            node_list.iter().map(|&id| (id, 1.0 / n_f)).collect();

        for _ in 0..iterations {
            // Dangling mass: total rank held by zero-out-degree nodes.
            let dangling_mass: f32 = node_list
                .iter()
                .filter(|&&id| out_degree[id] == 0)
                .map(|&id| rank[id])
                .sum();
            let dangling_share = damping * dangling_mass / n_f;

            let mut next: HashMap<&DocumentId, f32> = HashMap::with_capacity(n);
            for &id in &node_list {
                let mut inbound = 0.0f32;
                for &j in &incoming[id] {
                    let deg = out_degree[j];
                    if deg > 0 {
                        inbound += rank[j] / deg as f32;
                    }
                }
                let score = teleport + dangling_share + damping * inbound;
                next.insert(id, score);
            }
            rank = next;
        }

        rank.into_iter().map(|(id, r)| (id.clone(), r)).collect()
    }

    /// Like [`SourceGraph::pagerank`], but returns
    /// [`SourceCredibilityError::EmptyGraph`] instead of an empty map.
    ///
    /// # Errors
    ///
    /// Returns [`SourceCredibilityError::EmptyGraph`] when the graph has no nodes.
    pub fn pagerank_checked(
        &self,
        damping: f32,
        iterations: usize,
    ) -> Result<HashMap<DocumentId, f32>, SourceCredibilityError> {
        if self.nodes.is_empty() {
            return Err(SourceCredibilityError::EmptyGraph);
        }
        Ok(self.pagerank(damping, iterations))
    }
}
