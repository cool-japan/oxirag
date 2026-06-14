//! Inverted-File (IVF) approximate-nearest-neighbour index.
//!
//! IVF accelerates nearest-neighbour search by partitioning the vector space
//! into `num_cells` Voronoi regions. A *coarse quantizer* — a set of k-means
//! centroids — defines those regions; every indexed vector is filed into the
//! inverted list belonging to its nearest centroid. A query is answered by
//! locating the `nprobe` nearest centroids and scanning only their lists,
//! ranking candidates by squared L2 distance. This is distinct from the HNSW
//! graph index in `layer1_echo::ann`: HNSW is a navigable proximity graph,
//! whereas IVF is clustering plus inverted lists.
//!
//! The quantizer is trained with deterministic Lloyd k-means using a spread
//! initialisation, so building the same index twice yields identical results.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`IvfConfig`] | Cell count, probe count, dimension, training iterations |
//! | [`IvfIndex`] | Coarse quantizer, inverted lists and search |
//! | [`IvfHit`] | A single ranked search result |
//! | [`IvfError`] | Dimension / training / empty-set failures |
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "ivf-index")] {
//! use oxirag::ivf_index::{IvfConfig, IvfIndex};
//! use oxirag::types::DocumentId;
//!
//! let config = IvfConfig::new()
//!     .with_num_cells(4)
//!     .with_nprobe(4)
//!     .with_dim(3);
//! let mut index = IvfIndex::new(config);
//! index
//!     .build(&[
//!         (DocumentId::from_string("a"), vec![1.0, 0.0, 0.0]),
//!         (DocumentId::from_string("b"), vec![0.0, 1.0, 0.0]),
//!         (DocumentId::from_string("c"), vec![0.0, 0.0, 1.0]),
//!     ])
//!     .unwrap();
//!
//! let hits = index.search(&[0.9, 0.1, 0.0], 1).unwrap();
//! assert_eq!(hits[0].id, DocumentId::from_string("a"));
//! # }
//! ```

mod index;
mod types;

#[cfg(test)]
mod tests;

pub use index::IvfIndex;
pub use types::{IvfConfig, IvfError, IvfHit};
