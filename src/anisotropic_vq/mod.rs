//! Anisotropic Vector Quantization (`ScaNN`-style) for maximum inner-product
//! search.
//!
//! Implements the score-aware quantization of Guo, Sun, Lindgren, Geng, Simcha,
//! Chern & Kumar (*Accelerating Large-Scale Inference with Anisotropic Vector
//! Quantization*, ICML 2020 — `ScaNN`). Ordinary vector quantization minimizes
//! the plain squared reconstruction error `‖x − q‖²`. For inner-product /
//! cosine ranking, though, errors are **not** equally harmful in every
//! direction: an error component *parallel* to a database vector's own direction
//! distorts its estimated score for exactly the queries that rank it highly,
//! whereas an *orthogonal* error mostly averages out. Anisotropic VQ therefore
//! reweights the loss to penalize parallel error more.
//!
//! # The loss (implemented faithfully)
//!
//! For a training vector `x` (unit direction `x̂ = x/‖x‖`) and codeword `q`,
//! decompose the residual `e = x − q`:
//!
//! ```text
//! e_∥ = (e · x̂) x̂     (parallel)      e_⊥ = e − e_∥     (orthogonal)
//! L(x, q) = h_∥ ‖e_∥‖² + h_⊥ ‖e_⊥‖² = (x − q)ᵀ W_x (x − q)
//! ```
//!
//! with `h_⊥ = 1.0`, `h_∥ = parallel_weight_multiplier ≥ 1.0`, and
//! `W_x = h_∥ x̂x̂ᵀ + h_⊥(I − x̂x̂ᵀ)`.
//!
//! # Weighted Lloyd training
//!
//! The codebook is fit by a weighted k-means variant. The **assignment** step
//! sends each `x` to `argmin_q (x − q)ᵀ W_x (x − q)` (direction-aware, not plain
//! L2). The **update** step is the algorithmic centerpiece: the new codeword is
//! the minimizer of `Σ_x (x − q)ᵀ W_x (x − q)`, i.e. the solution of the normal
//! equations `(Σ_x W_x) q* = Σ_x W_x x` — a per-cluster weighted
//! least-squares problem solved by a hand-rolled `d×d` Gaussian elimination
//! (pure Rust, `f64`). Both steps minimize a shared objective, so the recorded
//! [`loss_trajectory`](quantizer::AnisotropicQuantizer::loss_trajectory) is
//! non-increasing.
//!
//! Both a single full-vector codebook (`num_subspaces = 1`) and product-style
//! anisotropic quantization (`num_subspaces > 1`, one codebook per subvector)
//! are supported. Queries are scored by asymmetric lookup against the learned
//! codewords.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`AnisotropicVqConfig`] | Dimensionality / codebook / weight settings + validation |
//! | [`AnisotropicQuantizer`] | Weighted-Lloyd training, encode / decode, asymmetric scoring |
//! | [`AnisotropicCode`] | A quantized vector (one [`u16`] codeword index per subspace) |
//! | [`AnisotropicVqIndex`] | Builds codes for a corpus and runs top-k search |
//! | [`AnisotropicVqHit`] | A ranked search result (`id` + `estimated_distance`) |
//! | [`AnisotropicVqMetric`] | Ranking metric (inner-product / L2) |
//! | [`AnisotropicVqError`] | Error type for configuration and runtime failures |
//! | [`decompose_residual`] / [`anisotropic_loss`] | Public helpers for the parallel/orthogonal split and weighted loss |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "anisotropic-vq")] {
//! use oxirag::prelude::*;
//!
//! let config = AnisotropicVqConfig::new()
//!     .with_dim(8)
//!     .with_num_codewords(16)
//!     .with_parallel_weight_multiplier(4.0);
//! let index = AnisotropicVqIndex::build_from(&items, config).unwrap();
//! let hits = index.search(&query, 5).unwrap();
//! # }
//! ```

pub mod index;
pub mod quantizer;
#[cfg(test)]
mod tests;
pub mod types;

pub use index::AnisotropicVqIndex;
pub use quantizer::{AnisotropicQuantizer, anisotropic_loss, decompose_residual};
pub use types::{
    AnisotropicCode, AnisotropicVqConfig, AnisotropicVqError, AnisotropicVqHit,
    AnisotropicVqMetric, DEFAULT_ANISOTROPIC_VQ_SEED, MAX_CODEBOOK_SIZE,
};
