//! Single random-projection tree: deterministic recursive space partitioning.
//!
//! A tree is a flat arena of [`RpTreeNode`]s. Construction proceeds
//! recursively: a point set of at most `leaf_size` members (or one reached at
//! the `max_depth` cap) becomes a leaf; otherwise an *equidistant* splitting
//! hyperplane is chosen and the points are partitioned into the two half-spaces
//! it induces, each recursed upon.
//!
//! # Choosing a split (Annoy-style)
//!
//! Two distinct points are sampled from the current set with a deterministic
//! `splitmix64` draw. Their difference is the hyperplane normal and their
//! midpoint fixes the offset, so the plane is equidistant from the pair. Points
//! are routed by the sign of [`RpTreeHyperplane::margin`].
//!
//! # Guaranteed termination
//!
//! If a sampled pair fails to separate the set (all points land on one side, or
//! the pair is identical), a few further pairs are tried; failing that, a
//! *median-by-rank* fallback splits the set into two non-empty halves by sorted
//! margin — and if even a usable direction cannot be found (all points
//! identical) the halves are taken by position under a degenerate hyperplane.
//! Because every internal node therefore yields two strictly smaller, non-empty
//! children, and because depth is capped by `max_depth`, recursion always
//! terminates with bounded stack usage.

// Casts between `usize` and `u64` are intentional and lossless on the 64-bit
// targets this crate supports; the modulo reductions cannot truncate a value
// that is already bounded by the (usize) slice length.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use super::types::{RpTreeConfig, RpTreeHyperplane, RpTreeNode, RpTreeResult};

/// Number of random pairs sampled while looking for a separating hyperplane
/// before the deterministic median-by-rank fallback is used.
const MAX_SPLIT_ATTEMPTS: usize = 3;

/// Squared-norm below which a candidate normal is treated as degenerate
/// (the two sampled points coincide).
const DEGENERATE_NORM_SQ: f32 = 1e-12;

// ── SplitMix64 ──────────────────────────────────────────────────────────────

/// A minimal, dependency-free `splitmix64` pseudo-random generator.
///
/// Deterministic given its seed; used only to sample split-point indices, so
/// the same configuration always reconstructs the same forest.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator seeded with `seed`.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the generator and return the next 64-bit output.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Return a pseudo-random index in `[0, bound)`.
    ///
    /// Returns `0` when `bound` is `0` (never happens on the split path, which
    /// always passes a set size of at least two).
    pub(crate) fn index(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}

/// Derive a well-mixed, distinct seed for tree `tree_idx` of a forest seeded
/// with `forest_seed`.
///
/// Distinct tree indices yield distinct seeds (hence complementary splits),
/// while the same `forest_seed` always reproduces the same per-tree seeds.
pub(crate) fn tree_seed(forest_seed: u64, tree_idx: usize) -> u64 {
    let mixed = forest_seed ^ (tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    SplitMix64::new(mixed).next_u64()
}

// ── RpTree ──────────────────────────────────────────────────────────────────

/// A single random-projection tree over a shared point store.
///
/// Leaves hold internal point indices into the forest's parallel `id`/`vector`
/// stores; the tree itself owns no vector data.
#[derive(Debug, Clone)]
pub(crate) struct RpTree {
    /// Flat arena of nodes; children always precede their parent.
    nodes: Vec<RpTreeNode>,
    /// Arena index of the root node.
    root: usize,
}

impl RpTree {
    /// Build a tree over `point_indices` (positions into `vectors`).
    ///
    /// # Errors
    ///
    /// Propagates any error surfaced while partitioning; in practice the
    /// partitioning is infallible once the configuration has been validated, so
    /// this returns [`Ok`] for every non-empty point set.
    pub(crate) fn build(
        point_indices: Vec<usize>,
        vectors: &[Vec<f32>],
        config: &RpTreeConfig,
        seed: u64,
    ) -> RpTreeResult<Self> {
        let mut nodes: Vec<RpTreeNode> = Vec::new();
        let mut rng = SplitMix64::new(seed);
        let root = build_node(&mut nodes, &mut rng, point_indices, vectors, config, 0)?;
        Ok(Self { nodes, root })
    }

    /// Borrow the node arena (used for structural assertions in tests).
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn nodes(&self) -> &[RpTreeNode] {
        &self.nodes
    }

    /// Arena index of the root node.
    pub(crate) fn root(&self) -> usize {
        self.root
    }

    /// Total number of nodes (internal plus leaf).
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Fetch a node by arena index, or `None` when out of range.
    pub(crate) fn node(&self, idx: usize) -> Option<&RpTreeNode> {
        self.nodes.get(idx)
    }

    /// Deepest leaf depth actually realised in the tree (root leaf ⇒ `0`).
    ///
    /// Traverses with an explicit stack, so it never recurses.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn observed_depth(&self) -> usize {
        let mut stack: Vec<(usize, usize)> = vec![(self.root, 0)];
        let mut deepest = 0;
        while let Some((idx, depth)) = stack.pop() {
            match self.nodes.get(idx) {
                Some(RpTreeNode::Leaf { .. }) => {
                    if depth > deepest {
                        deepest = depth;
                    }
                }
                Some(RpTreeNode::Internal { left, right, .. }) => {
                    stack.push((*left, depth + 1));
                    stack.push((*right, depth + 1));
                }
                None => {}
            }
        }
        deepest
    }

    /// Collect the member counts of every leaf, in arena order.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn leaf_sizes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                RpTreeNode::Leaf { ids } => Some(ids.len()),
                RpTreeNode::Internal { .. } => None,
            })
            .collect()
    }

    /// Collect every internal point index reachable from the leaves, sorted.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn collect_all_ids(&self) -> Vec<usize> {
        let mut all = Vec::new();
        for node in &self.nodes {
            if let RpTreeNode::Leaf { ids } = node {
                all.extend_from_slice(ids);
            }
        }
        all.sort_unstable();
        all
    }
}

// ── recursive construction ──────────────────────────────────────────────────

/// Recursively build a subtree over `set`, pushing nodes into `nodes` and
/// returning the arena index of the produced node.
///
/// Children are pushed before their parent, so the returned index is always the
/// last element appended for this call's subtree.
fn build_node(
    nodes: &mut Vec<RpTreeNode>,
    rng: &mut SplitMix64,
    set: Vec<usize>,
    vectors: &[Vec<f32>],
    config: &RpTreeConfig,
    depth: usize,
) -> RpTreeResult<usize> {
    // Termination: small enough, or the depth cap has been reached.
    if set.len() <= config.leaf_size || depth >= config.max_depth {
        let idx = nodes.len();
        nodes.push(RpTreeNode::Leaf { ids: set });
        return Ok(idx);
    }

    let (hyperplane, left_set, right_set) = find_split(&set, vectors, config.dim, rng);

    let left = build_node(nodes, rng, left_set, vectors, config, depth + 1)?;
    let right = build_node(nodes, rng, right_set, vectors, config, depth + 1)?;

    let idx = nodes.len();
    nodes.push(RpTreeNode::Internal {
        hyperplane,
        left,
        right,
    });
    Ok(idx)
}

/// Choose a split for `set` (guaranteed to have at least two members) and
/// return the hyperplane together with the two non-empty child sets.
fn find_split(
    set: &[usize],
    vectors: &[Vec<f32>],
    dim: usize,
    rng: &mut SplitMix64,
) -> (RpTreeHyperplane, Vec<usize>, Vec<usize>) {
    let n = set.len();
    let mut last_direction: Option<RpTreeHyperplane> = None;

    for _ in 0..MAX_SPLIT_ATTEMPTS {
        let (ia, ib) = draw_two_distinct(n, rng);
        let Some(hyperplane) = equidistant_hyperplane(&vectors[set[ia]], &vectors[set[ib]], dim)
        else {
            continue; // sampled points coincide; try another pair
        };
        let (left, right) = partition(set, vectors, &hyperplane);
        if !left.is_empty() && !right.is_empty() {
            return (hyperplane, left, right);
        }
        last_direction = Some(hyperplane);
    }

    // Fallback: split by the median rank of the margins so both halves are
    // non-empty. Reuse the last usable direction, or a degenerate plane when
    // no direction separated the (e.g. all-identical) points.
    let hyperplane = last_direction.unwrap_or_else(|| RpTreeHyperplane::degenerate(dim));
    median_split(set, vectors, hyperplane)
}

/// Draw two distinct indices in `[0, n)`; `n` is always at least two.
fn draw_two_distinct(n: usize, rng: &mut SplitMix64) -> (usize, usize) {
    let ia = rng.index(n);
    let mut ib = rng.index(n);
    let mut tries = 0;
    while ib == ia && tries < 8 {
        ib = rng.index(n);
        tries += 1;
    }
    if ib == ia {
        ib = (ia + 1) % n;
    }
    (ia, ib)
}

/// Build the equidistant hyperplane of two points, or `None` when they
/// coincide (a degenerate, zero-length normal).
fn equidistant_hyperplane(a: &[f32], b: &[f32], dim: usize) -> Option<RpTreeHyperplane> {
    let mut normal = vec![0.0f32; dim];
    let mut norm_sq = 0.0f32;
    for (slot, (&av, &bv)) in normal.iter_mut().zip(a.iter().zip(b.iter())) {
        let component = av - bv;
        *slot = component;
        norm_sq += component * component;
    }
    if norm_sq <= DEGENERATE_NORM_SQ {
        return None;
    }
    // bias = dot(normal, midpoint) = 0.5 * dot(a - b, a + b) = 0.5 (|a|^2 - |b|^2)
    let mut bias = 0.0f32;
    for (&nd, (&av, &bv)) in normal.iter().zip(a.iter().zip(b.iter())) {
        bias += nd * (av + bv);
    }
    bias *= 0.5;
    Some(RpTreeHyperplane::new(normal, bias))
}

/// Partition `set` by the sign of each point's margin: non-negative to the
/// right, negative to the left.
fn partition(
    set: &[usize],
    vectors: &[Vec<f32>],
    hyperplane: &RpTreeHyperplane,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &pi in set {
        if hyperplane.margin(&vectors[pi]) >= 0.0 {
            right.push(pi);
        } else {
            left.push(pi);
        }
    }
    (left, right)
}

/// Deterministic median-by-rank split: sort the set by margin (ties broken by
/// point index) and cut at the midpoint. Both halves are non-empty because the
/// set has at least two members.
fn median_split(
    set: &[usize],
    vectors: &[Vec<f32>],
    hyperplane: RpTreeHyperplane,
) -> (RpTreeHyperplane, Vec<usize>, Vec<usize>) {
    let mut ranked: Vec<usize> = set.to_vec();
    ranked.sort_by(|&x, &y| {
        let mx = hyperplane.margin(&vectors[x]);
        let my = hyperplane.margin(&vectors[y]);
        mx.total_cmp(&my).then_with(|| x.cmp(&y))
    });
    let mid = ranked.len() / 2;
    let right = ranked.split_off(mid);
    (hyperplane, ranked, right)
}
