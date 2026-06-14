//! Locality-Sensitive Hashing (LSH) approximate nearest-neighbour indexes.
//!
//! Two complementary schemes are provided:
//!
//! ## Random-hyperplane LSH — [`LshIndex`]
//!
//! Targets **dense float vectors** with **cosine similarity**.  Each vector is
//! projected onto `num_planes` deterministically-generated unit hyperplanes
//! (FNV-1a hashed from `(plane_idx, dim_idx)` pairs, so no external RNG is
//! required).  The sign of each projection becomes one bit of a compact
//! bit-signature; vectors with matching signatures land in the same bucket and
//! are re-ranked by exact cosine similarity at query time.
//!
//! ## `MinHash` LSH — [`MinHashIndex`]
//!
//! Targets **sparse sets / shingled text** with **Jaccard similarity**.
//! `num_bands × rows_per_band` independent min-hash functions are applied to
//! each set of element hashes; within each band the `rows_per_band` hashes are
//! folded into a single bucket key.  Two sets that collide in at least one band
//! become candidates, which are then re-ranked by exact Jaccard similarity.
//!
//! # Architecture
//!
//! | Type | Role |
//! |------|------|
//! | [`LshConfig`] | Shared configuration (dim, planes, bands, rows) |
//! | [`LshIndex`] | Dense cosine LSH (random hyperplanes) |
//! | [`MinHashIndex`] | Sparse Jaccard LSH (banded `MinHash`) |
//! | [`LshHit`] | Search result with `id: String` and `score: f32 ∈ [0,1]` |
//! | [`LshError`] | Error variants for dim mismatch, empty index, bad config |
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "lsh")] {
//! use oxirag::lsh_index::{LshConfig, LshIndex};
//!
//! let cfg = LshConfig::new().with_dim(4).with_num_planes(8);
//! let mut idx = LshIndex::new(cfg).unwrap();
//!
//! idx.insert("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
//! idx.insert("b", vec![0.0, 1.0, 0.0, 0.0]).unwrap();
//! idx.insert("c", vec![0.9, 0.1, 0.0, 0.0]).unwrap();
//!
//! let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
//! assert!(!hits.is_empty());
//! assert!(hits[0].score >= 0.0 && hits[0].score <= 1.0);
//! # }
//! ```

mod index;
mod types;

#[cfg(test)]
mod tests;

pub use index::{LshIndex, MinHashIndex};
pub use types::{LshConfig, LshError, LshHit};
