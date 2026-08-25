//! Residual / multi-stage vector quantization (RVQ) for compact approximate
//! nearest-neighbour search.
//!
//! Unlike product quantization (which splits a vector into independent
//! subspaces quantized in parallel), RVQ quantizes the **full** vector through
//! a cascade of `M = num_stages` codebooks applied **sequentially**. Each stage
//! encodes the residual left over by the preceding stages, so a vector is
//! reconstructed as the additive sum of one codeword per stage. This yields a
//! much finer approximation for a given codebook size, at the cost of
//! sequential (rather than parallel) encoding and decoding.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ResidualVqConfig`] | Stage / codebook / dimensionality / beam settings + validation |
//! | [`ResidualQuantizer`] | Cascaded codebook training, greedy/beam encode, decode, distance estimation |
//! | [`ResidualCode`] | A quantized vector (one codeword index per stage) |
//! | [`ResidualVqIndex`] | Trains a corpus and runs asymmetric-lookup top-k search |
//! | [`ResidualVqHit`] | A ranked search result (`id` + squared-L2 `distance`) |
//! | [`ResidualVqError`] | Error type for configuration and runtime failures |
//!
//! # Distance estimation
//!
//! - **Symmetric** — reconstruct both operands and compare the reconstructions.
//! - **Asymmetric** — keep the query at full precision; precompute per-stage
//!   query→codeword inner-product tables, then estimate the squared-L2 distance
//!   to an encoded database vector as `‖q‖² + ‖x̂‖² − 2 Σₘ ⟨q, codewordₘ⟩` with
//!   `M` table lookups plus a per-vector cached norm. This is the primary
//!   [`ResidualVqIndex::search`] path.
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "residual-vq")] {
//! use oxirag::residual_vq::{ResidualVqConfig, ResidualVqIndex};
//! use oxirag::types::DocumentId;
//!
//! let config = ResidualVqConfig::new().with_dim(8).with_num_stages(3).with_codebook_size(16);
//! let mut index = ResidualVqIndex::new(config);
//! index.build(&items).unwrap();
//! let hits = index.search(&query, 5).unwrap();
//! # }
//! ```

pub mod index;
pub mod quantizer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use index::ResidualVqIndex;
pub use quantizer::ResidualQuantizer;
pub use types::{ResidualCode, ResidualVqConfig, ResidualVqError, ResidualVqHit};
