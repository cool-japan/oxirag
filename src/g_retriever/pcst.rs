//! [`PcstSolver`] — a Goemans–Williamson primal-dual solver for the
//! Prize-Collecting Steiner Tree (PCST) problem.
//!
//! The Prize-Collecting Steiner Tree problem asks, given a graph with
//! nonnegative node *prizes* and nonnegative edge *costs*, for the connected
//! subgraph (a tree, or a forest in the unrooted case) that **maximises
//! collected prize minus paid edge cost** — equivalently, minimises
//! `(unpaid prize) + (paid cost)`. It is NP-hard; the classic
//! Goemans–Williamson (GW) primal-dual scheme delivers a `2`-approximation and
//! is exactly the family of algorithm used by the `pcst_fast` library that the
//! original G-Retriever paper relies on.
//!
//! The solver runs in two phases:
//!
//! 1. **Growth (moat growing).** Every node starts as its own active,
//!    singleton cluster whose *moat budget* equals its prize. Time flows
//!    uniformly; each active cluster's moat grows, chipping away at the slack
//!    of every edge crossing its boundary and at its own remaining prize
//!    budget. The next event is whichever comes first: either an inter-cluster
//!    edge goes *tight* (its slack hits zero), so the two clusters **merge** and
//!    the edge joins the forest; or an active cluster's moat exhausts its prize
//!    budget, so the cluster **deactivates** and grows no further. Growth ends
//!    when no event remains — all clusters inactive, or (in rooted mode) the
//!    root's cluster has no crossing edge left.
//! 2. **Strong pruning.** The growth forest is over-inclusive (moat growing
//!    can merge across an edge whose cost the connected prize does not
//!    justify). Rooting each tree and running the Johnson–Minkoff–Phillips
//!    dynamic program, every subtree whose net contribution (subtree prize
//!    minus the connecting edge's cost) is negative is pruned away, leaving the
//!    exact maximum-weight connected subtree of each growth tree.
//!
//! The result is returned as a [`PcstForest`].

use super::types::{GRetrieverError, GRetrieverResult, PcstEdge, PcstForest, PcstNode, UnionFind};

/// Numerical tolerance for treating a moat/slack/net quantity as zero.
const EPS: f64 = 1e-9;

// ── PcstSolver ────────────────────────────────────────────────────────────────

/// A Goemans–Williamson Prize-Collecting Steiner Tree solver.
///
/// The solver is configured with three switches — whether to strong-prune the
/// growth forest, whether an unrooted solve should collapse to its single best
/// component, and an optional mandated root — and is then applied to an
/// abstract [`PcstNode`] / [`PcstEdge`] view via [`solve`](Self::solve).
#[derive(Debug, Clone, Default)]
pub struct PcstSolver {
    /// When `true`, [`solve`](Self::solve) strong-prunes the growth forest;
    /// when `false`, it returns the raw growth forest.
    prune_enabled: bool,
    /// When `true`, an unrooted solve keeps only the single highest-net-value
    /// component. Ignored when a root is set.
    single_component: bool,
    /// An optional mandated root node id. When set, the solve is *rooted*: the
    /// root's cluster never deactivates and the pruned result is the single
    /// tree containing the root.
    root: Option<usize>,
}

impl PcstSolver {
    /// Create a solver.
    ///
    /// - `prune_enabled` — strong-prune the growth forest.
    /// - `single_component` — for an unrooted solve, keep only the single
    ///   best-net-value component.
    #[must_use]
    pub fn new(prune_enabled: bool, single_component: bool) -> Self {
        Self {
            prune_enabled,
            single_component,
            root: None,
        }
    }

    /// Set (or clear) the mandated root node id, turning the solve rooted (or
    /// back to unrooted).
    #[must_use]
    pub fn with_root(mut self, root: Option<usize>) -> Self {
        self.root = root;
        self
    }

    /// Whether strong pruning is enabled.
    #[must_use]
    pub fn prune_enabled(&self) -> bool {
        self.prune_enabled
    }

    /// The mandated root node id, if any.
    #[must_use]
    pub fn root(&self) -> Option<usize> {
        self.root
    }

    /// Solve the PCST instance over `nodes` and `edges`.
    ///
    /// Runs the Goemans–Williamson growth phase and, when
    /// [`prune_enabled`](Self::prune_enabled) is set, the strong-pruning phase.
    ///
    /// # Errors
    ///
    /// Returns [`GRetrieverError::InvalidGraph`] when a node prize or edge cost
    /// is non-finite, an edge endpoint is out of range, or a mandated root id
    /// is out of range.
    pub fn solve(&self, nodes: &[PcstNode], edges: &[PcstEdge]) -> GRetrieverResult<PcstForest> {
        let n = nodes.len();
        for (index, node) in nodes.iter().enumerate() {
            if !node.prize.is_finite() {
                return Err(GRetrieverError::InvalidGraph {
                    reason: format!("node {index} has a non-finite prize"),
                });
            }
        }
        for (index, edge) in edges.iter().enumerate() {
            if edge.source >= n || edge.target >= n {
                return Err(GRetrieverError::InvalidGraph {
                    reason: format!(
                        "edge {index} has an endpoint out of range for {n} nodes ({}, {})",
                        edge.source, edge.target
                    ),
                });
            }
            if !edge.cost.is_finite() {
                return Err(GRetrieverError::InvalidGraph {
                    reason: format!("edge {index} has a non-finite cost"),
                });
            }
        }
        if let Some(root) = self.root.filter(|&root| root >= n) {
            return Err(GRetrieverError::InvalidGraph {
                reason: format!("root {root} is out of range for {n} nodes"),
            });
        }

        let growth = self.grow(nodes, edges);
        if self.prune_enabled {
            Ok(self.prune(nodes, edges, &growth))
        } else {
            Ok(growth)
        }
    }

    /// Run only the Goemans–Williamson growth (moat-growing) phase, returning
    /// the growth forest.
    ///
    /// The returned forest's edges are those added by cluster merges; its node
    /// set is those edge endpoints together with any isolated node that still
    /// carried a positive prize (a valid singleton tree). Callers should
    /// normally use [`solve`](Self::solve); this is exposed for inspection and
    /// testing.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn grow(&self, nodes: &[PcstNode], edges: &[PcstEdge]) -> PcstForest {
        let n = nodes.len();
        let mut union = UnionFind::new(n);

        // Per-cluster (keyed at its union-find root) moat budget and activity.
        let mut prize_remaining: Vec<f64> = nodes.iter().map(|node| node.prize.max(0.0)).collect();
        let mut active: Vec<bool> = prize_remaining.iter().map(|&p| p > EPS).collect();
        if let Some(root) = self.root.filter(|&root| root < n) {
            active[root] = true;
        }

        // Per-edge remaining slack; dead edges are internal (both endpoints in
        // one cluster) or already merged.
        let mut slack: Vec<f64> = edges.iter().map(|edge| edge.cost.max(0.0)).collect();
        let mut alive: Vec<bool> = vec![true; edges.len()];
        let mut forest_edges: Vec<usize> = Vec::new();

        loop {
            // ── Find the earliest event. ──────────────────────────────────
            let mut best_delta = f64::INFINITY;
            let mut best_edge: Option<usize> = None;
            let mut best_deact: Option<usize> = None;

            for (index, edge) in edges.iter().enumerate() {
                if !alive[index] {
                    continue;
                }
                let ru = union.find(edge.source);
                let rv = union.find(edge.target);
                if ru == rv {
                    alive[index] = false;
                    continue;
                }
                let num_active = u8::from(active[ru]) + u8::from(active[rv]);
                if num_active == 0 {
                    continue;
                }
                let delta = slack[index] / f64::from(num_active);
                if delta < best_delta {
                    best_delta = delta;
                    best_edge = Some(index);
                    best_deact = None;
                }
            }

            for root in 0..n {
                if union.find(root) != root || !active[root] {
                    continue;
                }
                if self.root.is_some_and(|forced| union.find(forced) == root) {
                    continue; // The rooted cluster never deactivates.
                }
                let delta = prize_remaining[root];
                if delta < best_delta {
                    best_delta = delta;
                    best_deact = Some(root);
                    best_edge = None;
                }
            }

            if best_edge.is_none() && best_deact.is_none() {
                break;
            }
            let delta = best_delta.max(0.0);

            // ── Advance all active clusters (and crossing-edge slacks). ────
            if delta > 0.0 {
                for root in 0..n {
                    if union.find(root) == root && active[root] {
                        prize_remaining[root] = (prize_remaining[root] - delta).max(0.0);
                    }
                }
                for (index, edge) in edges.iter().enumerate() {
                    if !alive[index] {
                        continue;
                    }
                    let ru = union.find(edge.source);
                    let rv = union.find(edge.target);
                    if ru == rv {
                        continue;
                    }
                    let num_active = u8::from(active[ru]) + u8::from(active[rv]);
                    if num_active == 0 {
                        continue;
                    }
                    slack[index] = (slack[index] - f64::from(num_active) * delta).max(0.0);
                }
            }

            // ── Apply the event. ──────────────────────────────────────────
            if let Some(index) = best_edge {
                let edge = edges[index];
                let ru = union.find(edge.source);
                let rv = union.find(edge.target);
                let merged_budget = prize_remaining[ru] + prize_remaining[rv];
                let new_root = union.union(ru, rv);
                prize_remaining[new_root] = merged_budget;
                let contains_root = self
                    .root
                    .is_some_and(|forced| union.find(forced) == new_root);
                active[new_root] = contains_root || merged_budget > EPS;
                alive[index] = false;
                forest_edges.push(index);
            } else if let Some(root) = best_deact {
                active[root] = false;
                prize_remaining[root] = 0.0;
            }
        }

        // Node set: forest-edge endpoints plus still-relevant isolated nodes.
        let mut node_set: Vec<usize> = Vec::new();
        for &index in &forest_edges {
            node_set.push(edges[index].source);
            node_set.push(edges[index].target);
        }
        for node in nodes {
            if node.prize.max(0.0) > EPS {
                node_set.push(node.id.min(n.saturating_sub(1)));
            }
        }
        forest_edges.sort_unstable();
        PcstForest::new(n, node_set, forest_edges)
    }

    /// Strong-prune a growth `forest`, returning the maximum-weight connected
    /// subgraph(s).
    ///
    /// In unrooted mode each connected component of the growth forest is
    /// pruned independently to its best connected subtree; components whose
    /// best subtree has non-positive net value are dropped, and — when the
    /// solver was built with `single_component` — only the single
    /// highest-net-value surviving component is kept. In rooted mode the single
    /// tree containing the root is returned (the root is always kept).
    ///
    /// Callers should normally use [`solve`](Self::solve); this is exposed for
    /// inspection and testing.
    #[must_use]
    pub fn prune(&self, nodes: &[PcstNode], edges: &[PcstEdge], forest: &PcstForest) -> PcstForest {
        let n = nodes.len();
        let adjacency = build_adjacency(n, edges, &forest.edge_indices);

        if let Some(root) = self.root.filter(|&root| root < n) {
            let (kept_nodes, kept_edges) = prune_rooted(root, nodes, edges, &adjacency);
            return PcstForest::new(n, kept_nodes, kept_edges);
        }

        let mut component_of = vec![usize::MAX; n];
        let mut kept_nodes: Vec<usize> = Vec::new();
        let mut kept_edges: Vec<usize> = Vec::new();
        let mut best: Option<(f64, usize, Vec<usize>, Vec<usize>)> = None;

        for start in 0..n {
            if component_of[start] != usize::MAX {
                continue;
            }
            let component = collect_component(start, &adjacency, &mut component_of);
            let (comp_nodes, comp_edges, net) =
                prune_component(&component, nodes, edges, &adjacency);
            if comp_nodes.is_empty() {
                continue;
            }
            if self.single_component {
                let min_id = comp_nodes.iter().copied().min().unwrap_or(usize::MAX);
                let is_better = match &best {
                    None => true,
                    Some((best_net, best_min, _, _)) => {
                        net > *best_net + EPS
                            || ((net - *best_net).abs() <= EPS && min_id < *best_min)
                    }
                };
                if is_better {
                    best = Some((net, min_id, comp_nodes, comp_edges));
                }
            } else {
                kept_nodes.extend(comp_nodes);
                kept_edges.extend(comp_edges);
            }
        }

        // `best` is only ever populated in single-component mode.
        if let Some((_, _, comp_nodes, comp_edges)) = best {
            kept_nodes = comp_nodes;
            kept_edges = comp_edges;
        }

        PcstForest::new(n, kept_nodes, kept_edges)
    }
}

// ── pruning helpers ───────────────────────────────────────────────────────────

/// Build an adjacency list `adjacency[u] = [(neighbour, edge_index), ...]` from
/// the forest's retained edges. Out-of-range or missing edges are skipped.
fn build_adjacency(
    n: usize,
    edges: &[PcstEdge],
    edge_indices: &[usize],
) -> Vec<Vec<(usize, usize)>> {
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for &index in edge_indices {
        if let Some(edge) = edges
            .get(index)
            .filter(|edge| edge.source < n && edge.target < n)
        {
            adjacency[edge.source].push((edge.target, index));
            adjacency[edge.target].push((edge.source, index));
        }
    }
    adjacency
}

/// Collect the connected component containing `start` by iterative DFS over
/// `adjacency`, marking every visited node in `component_of`. Returns the
/// component's node ids, sorted ascending.
fn collect_component(
    start: usize,
    adjacency: &[Vec<(usize, usize)>],
    component_of: &mut [usize],
) -> Vec<usize> {
    let label = start;
    let mut stack = vec![start];
    component_of[start] = label;
    let mut component = Vec::new();
    while let Some(node) = stack.pop() {
        component.push(node);
        for &(neighbour, _) in &adjacency[node] {
            if component_of[neighbour] == usize::MAX {
                component_of[neighbour] = label;
                stack.push(neighbour);
            }
        }
    }
    component.sort_unstable();
    component
}

/// Build a rooted spanning tree of `root`'s component: the discovery order,
/// and per-node parent and parent-edge maps (indexed by node id).
fn build_rooted_tree(
    root: usize,
    n: usize,
    adjacency: &[Vec<(usize, usize)>],
) -> (Vec<usize>, Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut parent_edge: Vec<Option<usize>> = vec![None; n];
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);
    visited[root] = true;
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &(neighbour, edge_index) in &adjacency[node] {
            if !visited[neighbour] {
                visited[neighbour] = true;
                parent[neighbour] = Some(node);
                parent_edge[neighbour] = Some(edge_index);
                queue.push_back(neighbour);
            }
        }
    }
    (order, parent, parent_edge)
}

/// Compute the Johnson–Minkoff–Phillips subtree values.
///
/// `dp[v] = prize(v) + Σ_{c child of v} max(0, dp[c] - cost(v, c))` — the net
/// value of the best connected subtree that lies within `v`'s subtree and
/// contains `v`. Computed bottom-up by processing the BFS `order` in reverse.
fn compute_subtree_values(
    order: &[usize],
    parent: &[Option<usize>],
    parent_edge: &[Option<usize>],
    nodes: &[PcstNode],
    edges: &[PcstEdge],
    n: usize,
) -> Vec<f64> {
    let mut dp = vec![0.0_f64; n];
    for &node in order {
        dp[node] = nodes.get(node).map_or(0.0, |value| value.prize.max(0.0));
    }
    for &node in order.iter().rev() {
        if let (Some(par), Some(edge_index)) = (parent[node], parent_edge[node]) {
            let cost = edges.get(edge_index).map_or(0.0, |edge| edge.cost.max(0.0));
            let contribution = (dp[node] - cost).max(0.0);
            dp[par] += contribution;
        }
    }
    dp
}

/// Reconstruct the optimal subtree with apex `apex`: descend through the
/// rooted tree's child edges, keeping a child exactly when its net
/// contribution through the connecting edge is positive.
fn reconstruct_subtree(
    apex: usize,
    parent: &[Option<usize>],
    dp: &[f64],
    edges: &[PcstEdge],
    adjacency: &[Vec<(usize, usize)>],
) -> (Vec<usize>, Vec<usize>) {
    let mut kept_nodes = vec![apex];
    let mut kept_edges = Vec::new();
    let mut stack = vec![apex];
    while let Some(node) = stack.pop() {
        for &(child, edge_index) in &adjacency[node] {
            if parent[child] == Some(node) {
                let cost = edges.get(edge_index).map_or(0.0, |edge| edge.cost.max(0.0));
                if dp[child] - cost > EPS {
                    kept_nodes.push(child);
                    kept_edges.push(edge_index);
                    stack.push(child);
                }
            }
        }
    }
    kept_nodes.sort_unstable();
    kept_edges.sort_unstable();
    (kept_nodes, kept_edges)
}

/// Strong-prune one unrooted `component` to its maximum-weight connected
/// subtree. Returns the kept nodes, kept edges, and the net value. When the
/// best net value is non-positive, an empty result is returned (the component
/// is dropped entirely).
fn prune_component(
    component: &[usize],
    nodes: &[PcstNode],
    edges: &[PcstEdge],
    adjacency: &[Vec<(usize, usize)>],
) -> (Vec<usize>, Vec<usize>, f64) {
    let n = nodes.len();
    if component.is_empty() {
        return (Vec::new(), Vec::new(), 0.0);
    }
    let root = component.iter().copied().min().unwrap_or(0);
    let (order, parent, parent_edge) = build_rooted_tree(root, n, adjacency);
    let dp = compute_subtree_values(&order, &parent, &parent_edge, nodes, edges, n);

    // Apex = argmax dp over the component (smallest id breaks ties).
    let mut apex = root;
    let mut best_net = dp.get(root).copied().unwrap_or(0.0);
    for &node in &order {
        if dp[node] > best_net {
            best_net = dp[node];
            apex = node;
        }
    }

    if best_net <= EPS {
        return (Vec::new(), Vec::new(), 0.0);
    }
    let (kept_nodes, kept_edges) = reconstruct_subtree(apex, &parent, &dp, edges, adjacency);
    (kept_nodes, kept_edges, best_net)
}

/// Strong-prune in rooted mode: return the single tree containing `root`,
/// keeping the root unconditionally and each child subtree only when it pays
/// for itself.
fn prune_rooted(
    root: usize,
    nodes: &[PcstNode],
    edges: &[PcstEdge],
    adjacency: &[Vec<(usize, usize)>],
) -> (Vec<usize>, Vec<usize>) {
    let n = nodes.len();
    let (order, parent, parent_edge) = build_rooted_tree(root, n, adjacency);
    let dp = compute_subtree_values(&order, &parent, &parent_edge, nodes, edges, n);
    reconstruct_subtree(root, &parent, &dp, edges, adjacency)
}
