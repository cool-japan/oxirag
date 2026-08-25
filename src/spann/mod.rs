//! SPANN: memory-disk hybrid approximate nearest-neighbour search
//! (Chen et al., `NeurIPS` 2021, "SPANN: Highly-efficient Billion-scale
//! Approximate Nearest Neighbour Search").
//!
//! SPANN stores a small "head" of centroids — cheap enough to scan
//! exhaustively or probe — together with per-centroid posting lists (the
//! "tail") that carry the actual member identifiers. This module implements
//! the algorithm's defining ideas in-memory:
//!
//! 1. **Balanced clustering.** Deterministic Lloyd k-means (FNV-1a seeded,
//!    no `rand` dependency) partitions the corpus into `num_postings`
//!    centroids. Any posting list whose *primary* membership exceeds
//!    [`SpannConfig::posting_limit`] is recursively re-clustered into two,
//!    so no list is arbitrarily oversized before replication.
//! 2. **Boundary-closure replication.** Every point is assigned not only to
//!    its nearest centroid but to every centroid within
//!    `(1 + boundary_epsilon)` of that nearest distance, capped at
//!    `replica_count`. This is what makes points that sit near a Voronoi
//!    boundary still retrievable when a query probes a neighbouring cell.
//! 3. **RNG-rule pruning.** Naively replicating into every nearby centroid
//!    wastes space and search time on redundant copies, so a candidate
//!    centroid `c2` is dropped whenever an already-accepted, closer
//!    centroid `c1` satisfies `dist(c1, c2) < dist(point, c2)` — the
//!    relative-neighbourhood-graph rule from proximity-graph literature.
//! 4. **Search** probes the `nprobe` nearest centroids, scans only their
//!    posting lists, de-duplicates points reachable through more than one
//!    list, and returns the top-`k` scored hits.
//!
//! # Distinct from `ivf_index`
//!
//! [`crate::ivf_index`] is a plain inverted-file index: every vector is
//! filed into exactly one nearest-centroid list, with no balancing and no
//! replication. SPANN's defining contribution over plain IVF is precisely
//! the combination of posting-limit-driven balanced clustering, boundary
//! replication and RNG pruning implemented here.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|-----------------|
//! | [`SpannConfig`] | Cluster count, replica count, epsilon, posting limit, nprobe, iterations, metric |
//! | [`SpannMetric`] | Cosine (default) / L2 / Dot |
//! | `cluster` (private) | Deterministic k-means, balanced-cluster splitting, boundary replication, RNG pruning |
//! | [`SpannIndex`] | Build (cluster + replicate) and search (probe + dedup + rank) |
//! | [`Posting`] | One centroid plus its member identifiers |
//! | [`SpannHit`] | A single ranked search result |
//! | [`SpannStats`] | Posting-list balance/replication diagnostics |
//! | [`SpannError`] | Dimension / configuration / empty-input failures |
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "spann")] {
//! use oxirag::spann::{SpannConfig, SpannIndex};
//!
//! let items = vec![
//!     ("a".to_string(), vec![1.0, 0.0, 0.0]),
//!     ("b".to_string(), vec![0.9, 0.1, 0.0]),
//!     ("c".to_string(), vec![0.0, 1.0, 0.0]),
//!     ("d".to_string(), vec![0.0, 0.0, 1.0]),
//! ];
//! let config = SpannConfig::new().with_num_postings(2).with_nprobe(2);
//! let index = SpannIndex::build(items, config).unwrap();
//!
//! let hits = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
//! assert_eq!(hits[0].id, "a");
//! assert!(hits[0].score >= 0.0);
//! # }
//! ```

mod cluster;
mod index;
mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use index::SpannIndex;
pub use types::{Posting, SpannConfig, SpannError, SpannHit, SpannMetric, SpannStats};
