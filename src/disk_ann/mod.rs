//! `DiskANN` / Vamana: a single-layer graph-based approximate-nearest-neighbour
//! index (Subramanya et al., `NeurIPS` 2019).
//!
//! Unlike the multi-layer, randomly-leveled graph built by
//! [`crate::hnsw_index`], Vamana builds a single proximity graph whose entry
//! point is the data set's **medoid** — the point minimizing the sum of
//! distances to every other point — rather than a randomly promoted
//! top-layer node. Graph quality comes from the **`RobustPrune`**
//! α-occlusion rule, applied over two build passes: an initial pass with
//! `alpha = 1.0` produces a sparse, well-connected graph, and a second pass
//! with the configured `alpha` (default `1.2`) admits a few longer-range
//! edges that shrink the graph's diameter and improve recall.
//!
//! This is a **pure Rust**, in-memory approximation of the Vamana graph
//! algorithm — no ML, network, or external-crate dependencies beyond
//! `thiserror`. Every "random" choice (the initial `R`-regular graph) comes
//! from a seeded FNV-1a hash of the node-index pair, so `build` is fully
//! reproducible: identical input always produces an identical graph.
//!
//! # Algorithm
//!
//! 1. **Medoid.** Compute (or, above [`index::DiskAnnIndex`]'s internal
//!    exact-computation threshold, approximate via the point nearest the
//!    centroid) the point minimizing the sum of distances to all others;
//!    this is the fixed entry point for every search.
//! 2. **Random init.** Each node receives up to `R` out-edges chosen by a
//!    deterministic pseudo-random permutation seeded by an FNV-1a hash of
//!    the `(i, j)` node-index pair.
//! 3. **`GreedySearch(s, x_q, L)`.** Beam search from `s` toward `x_q` with a
//!    working set bounded at size `L`: repeatedly expand the closest
//!    unvisited candidate, folding its out-neighbors into the working set,
//!    until every candidate in the working set has been visited. Returns the
//!    (up to `L`) closest candidates plus the full visited set.
//! 4. **`RobustPrune(p, V, alpha, R)`.** Sort `V` by distance to `p`; greedily
//!    keep the closest remaining point `p*` and drop every other `v'` for
//!    which `alpha * d(p*, v') <= d(p, v')` (the α-occlusion rule); repeat
//!    until `R` neighbors are chosen or `V` is exhausted.
//! 5. **Build.** For every point `p` (processed in a fixed, deterministic
//!    index order), run `GreedySearch(medoid, p, L)` to obtain a visited set
//!    `V`; set `p`'s out-edges to `RobustPrune(p, V, alpha, R)`; then, for
//!    every new neighbor `j`, add a back-edge `j -> p`, re-pruning `j`'s
//!    out-edges whenever they exceed `R`. The whole pass runs **twice**:
//!    once with `alpha = 1.0`, then again with the configured `alpha`.
//! 6. **Search.** `GreedySearch(medoid, query, max(L, k))`, truncated to the
//!    top `k`.
//!
//! # Distinct from `hnsw_index`
//!
//! | | [`crate::hnsw_index`] | `disk_ann` |
//! |-|------------------------|------------|
//! | Graph shape | Multi-layer skip-list | Single layer |
//! | Entry point | Highest-layer node (random level assignment) | Medoid |
//! | Neighbor selection | Keep the `M` closest neighbors | `RobustPrune` α-occlusion, two-pass build |
//! | Determinism source | FNV-1a hash of the node index (level assignment) | FNV-1a hash of the `(i, j)` node-index pair (initial edges) |
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "disk-ann")] {
//! use oxirag::disk_ann::{DiskAnnConfig, DiskAnnIndex, DiskAnnMetric};
//!
//! fn fnv_embed(text: &str, dim: usize) -> Vec<f32> {
//!     let mut v = vec![0.0f32; dim];
//!     let bytes = text.as_bytes();
//!     for (i, slot) in v.iter_mut().enumerate() {
//!         let mut h: u64 = 14_695_981_039_346_656_037;
//!         for &b in bytes { h = h.wrapping_mul(1_099_511_628_211) ^ b as u64; }
//!         h = h.wrapping_mul(1_099_511_628_211) ^ i as u64;
//!         *slot = ((h >> 32) as f32) / u32::MAX as f32 * 2.0 - 1.0;
//!     }
//!     v
//! }
//!
//! let items = vec![
//!     ("rust".to_string(), fnv_embed("rust language", 16)),
//!     ("python".to_string(), fnv_embed("python language", 16)),
//!     ("java".to_string(), fnv_embed("java language", 16)),
//! ];
//!
//! let config = DiskAnnConfig::new()
//!     .with_max_degree(8)
//!     .with_search_list_size(16)
//!     .with_metric(DiskAnnMetric::Cosine);
//!
//! let index = DiskAnnIndex::build(items, config).unwrap();
//! let hits = index.search(&fnv_embed("rust language", 16), 1).unwrap();
//! assert_eq!(hits[0].id, "rust");
//! # }
//! ```

pub mod graph;
pub mod index;
pub mod types;

#[cfg(test)]
mod tests;

pub use graph::VamanaGraph;
pub use index::DiskAnnIndex;
pub use types::{DiskAnnConfig, DiskAnnError, DiskAnnHit, DiskAnnMetric};
