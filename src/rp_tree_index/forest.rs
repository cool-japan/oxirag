//! Random-projection tree *forest* and the public [`RpTreeIndex`] wrapper.
//!
//! A [`RpTreeForest`] owns the point store (parallel `id` and `vector` vectors)
//! and `n_trees` independent [`RpTree`]s, each grown from a distinct derived
//! seed so their approximation errors are complementary.
//!
//! # Search
//!
//! A query descends *every* tree at once through a single shared max-priority
//! queue. Each queued item is `(priority, tree, node)`; the priority is the
//! running minimum of the signed margins along the path taken to reach that
//! node (seeded at `+∞` for every tree root). Popping the highest-priority item
//! first therefore follows the natural root-to-leaf descent in each tree, while
//! the sibling branch keeps a priority of `min(path, -margin)` — close to zero,
//! hence explored soon, exactly when the query lay near that branch's boundary.
//!
//! Leaves contribute their member points to a deduplicated candidate set;
//! expansion continues until `search_k` distinct candidates have been gathered
//! or the queue empties. The candidates are then re-scored *exactly* against the
//! query and the true top-`k` returned — so the tree structure only proposes
//! candidates while the final ranking is exact.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use super::tree::{RpTree, tree_seed};
use super::types::{RpTreeConfig, RpTreeError, RpTreeHit, RpTreeNode, RpTreeResult};

// ── priority-queue item ─────────────────────────────────────────────────────

/// One entry of the search priority queue: a not-yet-expanded node in some
/// tree, tagged with the running path priority.
#[derive(Clone, Copy)]
struct PqItem {
    /// Running minimum margin along the path to this node (`+∞` at a root).
    priority: f32,
    /// Index of the owning tree.
    tree: usize,
    /// Arena index of the node within that tree.
    node: usize,
}

impl PartialEq for PqItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PqItem {}

impl Ord for PqItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Total order via `f32::total_cmp` (never panics), with deterministic
        // tie-breaking on (tree, node) so the heap order is fully reproducible.
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| self.tree.cmp(&other.tree))
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for PqItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── RpTreeForest ────────────────────────────────────────────────────────────

/// A forest of independent random-projection trees over a shared point store.
///
/// Construct it with [`RpTreeForest::build`] and query it with
/// [`RpTreeForest::search`]. Most callers should use the higher-level
/// [`RpTreeIndex`] wrapper instead.
#[derive(Debug, Clone)]
pub struct RpTreeForest {
    /// Effective configuration (validated at build time).
    config: RpTreeConfig,
    /// Caller-supplied identifiers, indexed by internal point index.
    ids: Vec<String>,
    /// Pre-processed vectors, indexed by internal point index.
    vectors: Vec<Vec<f32>>,
    /// The independent trees.
    trees: Vec<RpTree>,
}

impl RpTreeForest {
    /// Build a forest from `points` (each an `(id, vector)` pair).
    ///
    /// Vectors are pre-processed for the configured metric (L2-normalised for
    /// cosine) before both partitioning and storage.
    ///
    /// # Errors
    ///
    /// * [`RpTreeError::InvalidConfig`] — the configuration is invalid.
    /// * [`RpTreeError::EmptyPoints`] — `points` is empty.
    /// * [`RpTreeError::DimMismatch`] — some vector's length is not
    ///   [`RpTreeConfig::dim`].
    pub fn build(points: Vec<(String, Vec<f32>)>, config: RpTreeConfig) -> RpTreeResult<Self> {
        config.validate()?;
        if points.is_empty() {
            return Err(RpTreeError::EmptyPoints);
        }

        let mut ids = Vec::with_capacity(points.len());
        let mut vectors = Vec::with_capacity(points.len());
        for (id, vector) in points {
            if vector.len() != config.dim {
                return Err(RpTreeError::DimMismatch {
                    expected: config.dim,
                    got: vector.len(),
                });
            }
            ids.push(id);
            vectors.push(config.metric.preprocess(vector));
        }

        let point_indices: Vec<usize> = (0..vectors.len()).collect();
        let mut trees = Vec::with_capacity(config.n_trees);
        for tree_idx in 0..config.n_trees {
            let seed = tree_seed(config.seed, tree_idx);
            let tree = RpTree::build(point_indices.clone(), &vectors, &config, seed)?;
            trees.push(tree);
        }

        Ok(Self {
            config,
            ids,
            vectors,
            trees,
        })
    }

    /// Number of indexed points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Return `true` when the forest holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Number of trees in the forest.
    #[must_use]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Borrow the effective configuration.
    #[must_use]
    pub fn config(&self) -> &RpTreeConfig {
        &self.config
    }

    /// Borrow the trees (crate-internal; used for structural assertions).
    #[cfg(test)]
    pub(crate) fn trees(&self) -> &[RpTree] {
        &self.trees
    }

    /// Search for the `k` points most similar to `query`.
    ///
    /// Candidate points are gathered from leaves visited across all trees via
    /// the shared priority queue until `search_k` distinct candidates are found
    /// (or the queue empties), then re-scored exactly and returned in
    /// descending score order. Fewer than `k` hits are returned only when the
    /// forest holds fewer than `k` points.
    ///
    /// # Errors
    ///
    /// * [`RpTreeError::EmptyIndex`] — the forest holds no points.
    /// * [`RpTreeError::InvalidK`] — `k` is zero.
    /// * [`RpTreeError::DimMismatch`] — `query.len()` is not
    ///   [`RpTreeConfig::dim`].
    pub fn search(&self, query: &[f32], k: usize) -> RpTreeResult<Vec<RpTreeHit>> {
        if self.is_empty() {
            return Err(RpTreeError::EmptyIndex);
        }
        if k == 0 {
            return Err(RpTreeError::InvalidK);
        }
        if query.len() != self.config.dim {
            return Err(RpTreeError::DimMismatch {
                expected: self.config.dim,
                got: query.len(),
            });
        }

        let processed = self.config.metric.preprocess(query.to_vec());
        let candidates = self.gather_candidates(&processed);
        Ok(self.rank_candidates(&processed, &candidates, k))
    }

    /// Descend every tree through the shared priority queue, returning the
    /// deduplicated internal indices of all candidate points, in the order they
    /// were first discovered.
    fn gather_candidates(&self, query: &[f32]) -> Vec<usize> {
        let mut heap: BinaryHeap<PqItem> = BinaryHeap::new();
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            heap.push(PqItem {
                priority: f32::INFINITY,
                tree: tree_idx,
                node: tree.root(),
            });
        }

        let mut seen: HashSet<usize> = HashSet::new();
        let mut candidates: Vec<usize> = Vec::new();
        let budget = self.config.search_k;

        while candidates.len() < budget {
            let Some(item) = heap.pop() else {
                break;
            };
            let Some(tree) = self.trees.get(item.tree) else {
                continue;
            };
            match tree.node(item.node) {
                Some(RpTreeNode::Leaf { ids }) => {
                    for &pid in ids {
                        if seen.insert(pid) {
                            candidates.push(pid);
                        }
                    }
                }
                Some(RpTreeNode::Internal {
                    hyperplane,
                    left,
                    right,
                }) => {
                    let margin = hyperplane.margin(query);
                    // The natural side (matching the margin sign) keeps the
                    // higher priority; the opposite side is capped at
                    // `-|margin|`, so a query near the boundary keeps that
                    // sibling near the top of the queue.
                    heap.push(PqItem {
                        priority: item.priority.min(margin),
                        tree: item.tree,
                        node: *right,
                    });
                    heap.push(PqItem {
                        priority: item.priority.min(-margin),
                        tree: item.tree,
                        node: *left,
                    });
                }
                None => {}
            }
        }

        candidates
    }

    /// Exactly re-score `candidates` against `query` and return the top-`k`
    /// hits in descending score order (ties broken by ascending internal index
    /// for determinism).
    fn rank_candidates(&self, query: &[f32], candidates: &[usize], k: usize) -> Vec<RpTreeHit> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&idx| (self.config.metric.score(query, &self.vectors[idx]), idx))
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(k);

        scored
            .into_iter()
            .map(|(score, idx)| RpTreeHit::new(self.ids[idx].clone(), score))
            .collect()
    }
}

// ── RpTreeIndex ─────────────────────────────────────────────────────────────

/// Public random-projection tree forest index.
///
/// This is the primary entry point: build it from a set of `(id, vector)`
/// points with [`RpTreeIndex::build`], then retrieve approximate nearest
/// neighbours with [`RpTreeIndex::search`]. Internally it wraps a
/// [`RpTreeForest`]; the candidate structure is approximate but the returned
/// ranking is computed by exact distance, so results are accurate for a
/// sufficiently large [`RpTreeConfig::search_k`].
///
/// # Examples
///
/// ```
/// use oxirag::rp_tree_index::{RpTreeConfig, RpTreeIndex, RpTreeMetric};
///
/// let cfg = RpTreeConfig::new()
///     .with_dim(3)
///     .with_n_trees(4)
///     .with_leaf_size(1)
///     .with_search_k(16)
///     .with_metric(RpTreeMetric::Cosine);
///
/// let points = vec![
///     ("a".to_string(), vec![1.0, 0.0, 0.0]),
///     ("b".to_string(), vec![0.0, 1.0, 0.0]),
///     ("c".to_string(), vec![0.9, 0.1, 0.0]),
/// ];
/// let index = RpTreeIndex::build(points, cfg).expect("build");
///
/// let hits = index.search(&[1.0, 0.0, 0.0], 2).expect("search");
/// assert_eq!(hits[0].id, "a");
/// assert!(hits[0].score >= hits[1].score);
/// ```
#[derive(Debug, Clone)]
pub struct RpTreeIndex {
    /// The underlying forest.
    forest: RpTreeForest,
}

impl RpTreeIndex {
    /// Build an index from `points` under `config`.
    ///
    /// # Errors
    ///
    /// See [`RpTreeForest::build`] for the full set of error conditions
    /// (invalid config, empty point set, dimension mismatch).
    pub fn build(points: Vec<(String, Vec<f32>)>, config: RpTreeConfig) -> RpTreeResult<Self> {
        Ok(Self {
            forest: RpTreeForest::build(points, config)?,
        })
    }

    /// Search for the `k` nearest neighbours of `query`.
    ///
    /// # Errors
    ///
    /// See [`RpTreeForest::search`] (empty index, `k == 0`, dimension
    /// mismatch).
    pub fn search(&self, query: &[f32], k: usize) -> RpTreeResult<Vec<RpTreeHit>> {
        self.forest.search(query, k)
    }

    /// Number of indexed points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.forest.len()
    }

    /// Return `true` when the index holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.forest.is_empty()
    }

    /// Number of trees in the underlying forest.
    #[must_use]
    pub fn num_trees(&self) -> usize {
        self.forest.num_trees()
    }

    /// Borrow the effective configuration.
    #[must_use]
    pub fn config(&self) -> &RpTreeConfig {
        self.forest.config()
    }

    /// Borrow the underlying forest.
    #[must_use]
    pub fn forest(&self) -> &RpTreeForest {
        &self.forest
    }
}
