//! Random-projection tree **forest** — an Annoy-style approximate
//! nearest-neighbour index.
//!
//! This module builds a *forest* of independent binary space-partitioning
//! trees. Each [`RpTreeIndex`] owns a shared point store and
//! [`RpTreeConfig::n_trees`] trees; each tree recursively splits its points
//! with *equidistant* hyperplanes (the perpendicular/angular bisector of two
//! deterministically sampled points), so descending a tree narrows a query to a
//! small leaf. Multiple trees, each grown from a different derived seed, make
//! complementary partitioning mistakes; a shared priority queue searches all of
//! them and the merged candidate set is re-ranked by *exact* distance.
//!
//! # How this differs from its neighbours in the crate
//!
//! * Versus [`lsh_index`](crate::lsh_index): LSH performs **flat sign-bit
//!   bucket hashing** — every vector is reduced to a fixed-width signature and
//!   dropped into a hash bucket, with *no tree and no hierarchy*. An
//!   `rp_tree_index` builds a **recursive binary space-partitioning tree** per
//!   tree, and searches several such trees with priority-queue back-tracking
//!   rather than a single bucket lookup.
//! * Versus [`hnsw_index`](crate::hnsw_index): HNSW is a **navigable proximity
//!   graph** whose edges connect nearby points across hierarchical layers.
//!   `rp_tree_index` stores no neighbour edges at all; it stores
//!   **space-partitioning trees** and reaches candidates by descending
//!   hyperplane splits, not by hopping between adjacent points.
//!
//! # Algorithm
//!
//! 1. **Single-tree construction** (`tree`, recursive). A point set of at
//!    most [`RpTreeConfig::leaf_size`] members becomes a leaf. Otherwise two
//!    distinct points are sampled with a seeded `splitmix64` draw, their
//!    equidistant hyperplane is formed, and the points are partitioned by the
//!    sign of their margin; each half is recursed upon. Degenerate splits (all
//!    points on one side, or coincident samples) fall back to further sampled
//!    pairs and finally a deterministic median-by-rank split, and a
//!    [`RpTreeConfig::max_depth`] cap guarantees termination.
//! 2. **Forest** (`forest`). [`RpTreeConfig::n_trees`] trees are grown, each
//!    from a seed derived from the forest seed and the tree index, so no two
//!    trees split identically.
//! 3. **Search** (`forest`). All trees are descended together through one
//!    shared max-priority queue keyed on the running minimum margin; visited
//!    leaves feed a deduplicated candidate set, which is re-scored exactly to
//!    yield the true top-`k`.
//!
//! The generator is fully deterministic: a fixed [`RpTreeConfig`] and point set
//! always reproduce the same forest and the same search results — no `rand`,
//! `ndarray`, or other external dependency is involved.
//!
//! # Example
//!
//! ```
//! use oxirag::rp_tree_index::{RpTreeConfig, RpTreeIndex, RpTreeMetric};
//!
//! let cfg = RpTreeConfig::new()
//!     .with_dim(4)
//!     .with_n_trees(6)
//!     .with_leaf_size(2)
//!     .with_search_k(32)
//!     .with_metric(RpTreeMetric::Cosine);
//!
//! let points = vec![
//!     ("north".to_string(), vec![0.0, 1.0, 0.0, 0.0]),
//!     ("south".to_string(), vec![0.0, -1.0, 0.0, 0.0]),
//!     ("east".to_string(), vec![1.0, 0.0, 0.0, 0.0]),
//!     ("west".to_string(), vec![-1.0, 0.0, 0.0, 0.0]),
//! ];
//! let index = RpTreeIndex::build(points, cfg).expect("build succeeds");
//!
//! let hits = index.search(&[0.0, 0.9, 0.0, 0.0], 1).expect("search succeeds");
//! assert_eq!(hits[0].id, "north");
//! ```

mod forest;
mod tree;
mod types;

#[cfg(test)]
mod tests;

pub use forest::{RpTreeForest, RpTreeIndex};
pub use types::{
    RpTreeConfig, RpTreeError, RpTreeHit, RpTreeHyperplane, RpTreeMetric, RpTreeNode, RpTreeResult,
};
