//! `RaBitQ`: randomized bit quantization with an unbiased distance estimator.
//!
//! Implements the quantizer of Gao & Long, *`RaBitQ`: Quantizing High-Dimensional
//! Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor
//! Search* (SIGMOD 2024). `RaBitQ` compresses each `D`-dimensional vector to a
//! single bit per dimension after a random orthogonal rotation, then provides
//! a provably **unbiased** estimator of the inner product / distance between
//! a full-precision query and a quantized database vector, together with an
//! `O(1/√D)`-scale error bound.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`RaBitQConfig`] | Dimensionality, rotation seed, and ranking metric |
//! | [`RaBitQMetric`] | `L2` (default) / `InnerProduct` / `Cosine` ranking |
//! | [`rotation::Rotation`] | Deterministic pseudo-random `D×D` orthogonal matrix `P` |
//! | [`RaBitQuantizer`] | Centroid + rotation training, 1-bit encoding, unbiased estimators |
//! | [`RaBitQCode`] | A quantized vector: packed bits + `factor` + `norm` |
//! | [`RaBitQIndex`] | Builds codes for a corpus and runs brute-force top-k search |
//! | [`RaBitQHit`] | A ranked search result (`id` + `estimated_distance`) |
//! | [`RaBitQError`] | Error type for configuration and runtime failures |
//!
//! # The algorithm
//!
//! ## 1. Preprocessing
//!
//! For the dataset centroid `c` (the mean of all indexed vectors) and a raw
//! vector `o_raw`, form the centroid residual `o = o_raw - c`, its norm
//! `‖o‖`, and the unit direction `o_unit = o / ‖o‖`.
//!
//! ## 2. Random orthogonal rotation
//!
//! A deterministic pseudo-random `D×D` orthogonal matrix `P` is derived from
//! a `u64` seed with **no external randomness crate**: FNV-1a hashing of
//! `(seed, index)` produces a stream of pseudo-uniform values, the Box–Muller
//! transform turns pairs of them into pseudo-Gaussian samples, and modified
//! Gram–Schmidt orthonormalizes the resulting `D×D` matrix's columns so that
//! `PᵀP = P Pᵀ = I` (see [`rotation`] for the full derivation). The residual is
//! rotated: `o' = P · o_unit`.
//!
//! Because `P` is orthogonal, it is an **isometry**: `⟨P a, P b⟩ = ⟨a, b⟩` for
//! any vectors `a, b`. This is the property that lets a code computed
//! entirely in rotated space serve as an unbiased estimator of inner products
//! in the *original* space.
//!
//! ## 3. One-bit code
//!
//! Each rotated coordinate is thresholded at zero: `b_i = 1` if `o'_i > 0`,
//! else `0`. The corresponding codebook vector is the scaled hypercube vertex
//! `x̄_i = (2·b_i - 1)/√D`, which has unit norm (`‖x̄‖ = 1`) and is the closest
//! hypercube vertex to `o'` in the same orthant.
//!
//! ## 4. Per-vector factor
//!
//! [`RaBitQCode`] stores `factor = ⟨x̄, o'⟩` (the cosine between the code and
//! the true rotated unit residual — always `≥ 0`) and `norm = ‖o‖`. The bits
//! themselves are packed 1-bit-per-dimension into `u64` words; `x̄` is never
//! materialized as a stored vector, only recomputed on demand from the bits.
//!
//! ## 5. Query-time estimation
//!
//! For any vector `v` expressed in original (un-rotated) coordinates, the
//! ratio
//!
//! ```text
//! est(v) = ⟨x̄, P·v⟩ / ⟨x̄, o'⟩
//! ```
//!
//! is an unbiased estimator of `⟨o_unit, v⟩`, with error concentrating at
//! scale `O(1/√(D-1))` (Gao & Long, Theorem 3.3; exposed here as
//! [`RaBitQuantizer::error_bound`] `= 1/√D`). [`RaBitQuantizer::estimate_ip`]
//! and [`RaBitQuantizer::estimate_l2_sq`] each apply `est` to a different
//! choice of `v` (the raw query, or the centered residual query
//! respectively) and combine it with exact centroid arithmetic to reconstruct
//! estimates in the *original* vector space — see [`quantizer`] for the full
//! derivation of both identities.
//!
//! # Distinctness from other quantization modules
//!
//! | Module | Codebook | Estimator |
//! |--------|----------|-----------|
//! | [`crate::scalar_quantization`] | Per-dimension uniform int8 / sign-bit binary | Deterministic (no rotation, no unbiasedness guarantee) |
//! | [`crate::product_quantization`] | Learned per-subspace k-means centroids | Asymmetric Distance Computation (ADC) table lookup |
//! | **`rabitq`** | Single global scaled-hypercube vertex per vector, in a **randomly rotated** space | Provably **unbiased** ratio estimator with an explicit `O(1/√D)` error bound |
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "rabitq")] {
//! use oxirag::rabitq::{RaBitQConfig, RaBitQIndex};
//!
//! // A handful of 4-dimensional vectors in two well-separated clusters.
//! let items = vec![
//!     ("a".to_string(), vec![1.0_f32, 0.0, 0.0, 0.0]),
//!     ("b".to_string(), vec![0.9_f32, 0.1, 0.0, 0.0]),
//!     ("c".to_string(), vec![0.0_f32, 0.0, 1.0, 0.0]),
//!     ("d".to_string(), vec![0.0_f32, 0.0, 0.9, 0.1]),
//! ];
//!
//! let config = RaBitQConfig::new(4).unwrap();
//! let index = RaBitQIndex::build(items, config).unwrap();
//!
//! // Query close to the first cluster.
//! let query = vec![1.0_f32, 0.05, 0.0, 0.0];
//! let hits = index.search(&query, 2).unwrap();
//! assert_eq!(hits.len(), 2);
//! assert!(hits[0].id == "a" || hits[0].id == "b");
//!
//! // The estimator's error is bounded at 1/sqrt(D) scale.
//! let bound = index.quantizer().error_bound();
//! assert!((bound - 0.5).abs() < 1e-6); // 1/sqrt(4) = 0.5
//! # }
//! ```

pub mod index;
pub mod quantizer;
pub mod rotation;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use index::RaBitQIndex;
pub use quantizer::RaBitQuantizer;
pub use types::{RaBitQCode, RaBitQConfig, RaBitQError, RaBitQHit, RaBitQMetric};
