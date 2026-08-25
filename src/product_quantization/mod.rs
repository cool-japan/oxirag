//! Product Quantization (PQ) for compact approximate nearest-neighbour search.
//!
//! Implements the product quantizer of Jégou, Douze and Schmid
//! (*Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011).
//! Each `dim`-dimensional vector is split into `num_subspaces` contiguous
//! subvectors; per subspace a codebook of `K = 2^codebook_bits` centroids is
//! learned with a deterministic k-means-lite. A vector is then stored as
//! `num_subspaces` one-byte codeword indices, and queries are scored with
//! **Asymmetric Distance Computation** (ADC): the full-precision query is kept,
//! per-subspace query→centroid squared distances are precomputed into lookup
//! tables, and a code's distance is the sum of its looked-up entries.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`PqConfig`] | Subspace / codebook / dimensionality settings + validation |
//! | [`ProductQuantizer`] | Codebook training, encode / decode, ADC scoring |
//! | [`PqCode`] | A quantized vector (one [`u8`] codeword per subspace) |
//! | [`PqIndex`] | Builds codes for a corpus and runs ADC top-k search |
//! | [`PqHit`] | A ranked search result (`id` + squared-L2 `distance`) |
//! | [`PqError`] | Error type for configuration and runtime failures |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "product-quantization")] {
//! use oxirag::prelude::*;
//!
//! let config = PqConfig::new().with_dim(8).with_num_subspaces(2).with_codebook_bits(2);
//! let mut index = PqIndex::new(config);
//! index.build(&items).unwrap();
//! let hits = index.search(&query, 5).unwrap();
//! # }
//! ```

pub mod index;
pub mod quantizer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use index::PqIndex;
pub use quantizer::ProductQuantizer;
pub use types::{PqCode, PqConfig, PqError, PqHit};
