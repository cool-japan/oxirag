//! Hierarchical Navigable Small World (HNSW) approximate-nearest-neighbour index.
//!
//! HNSW builds a multi-layer proximity graph over a set of vectors. Insertion
//! assigns each new node to a random set of layers (level 0 is always present)
//! and connects it to its `M` nearest neighbours via bidirectional edges.
//! Search descends greedily from the topmost occupied layer and performs an
//! `ef_search`-wide beam scan at layer 0.
//!
//! The implementation is **pure Rust** with no ML, network, or external-crate
//! dependencies beyond `thiserror`. All random-like choices are replaced by
//! deterministic FNV-1a hashing of the node index so builds are reproducible.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`HnswConfig`] | Dimension, M, `ef_construction`, `ef_search`, `max_layers` |
//! | [`HnswIndex`]  | Graph storage, insertion, and multi-layer beam search |
//! | [`HnswHit`]    | A single ranked search result |
//! | [`HnswError`]  | Dimension / empty-index / config / k failures |
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "hnsw")] {
//! use oxirag::hnsw_index::{HnswConfig, HnswIndex};
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
//!     let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
//!     v.iter_mut().for_each(|x| *x /= norm);
//!     v
//! }
//!
//! let cfg = HnswConfig::new()
//!     .with_dim(16)
//!     .with_m(4)
//!     .with_ef_construction(20)
//!     .with_ef_search(10);
//!
//! let mut index = HnswIndex::new(cfg);
//! index.insert("rust",   fnv_embed("rust language",   16)).unwrap();
//! index.insert("python", fnv_embed("python language", 16)).unwrap();
//! index.insert("java",   fnv_embed("java language",   16)).unwrap();
//!
//! let hits = index.search(&fnv_embed("rust language", 16), 1).unwrap();
//! assert_eq!(hits[0].id, "rust");
//! assert!(hits[0].score > 0.99);
//! # }
//! ```

pub mod index;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use index::HnswIndex;
pub use types::{HnswConfig, HnswError, HnswHit};
