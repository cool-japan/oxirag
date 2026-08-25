//! Iterative Quantization (ITQ): learning compact binary codes for approximate
//! nearest-neighbor search.
//!
//! Implements Gong & Lazebnik, *Iterative Quantization: A Procrustean Approach
//! to Learning Binary Codes* (CVPR 2011). ITQ compresses each `D`-dimensional
//! vector to a `k`-bit binary code such that Hamming distance between codes
//! approximates similarity in the original space. Its distinguishing feature —
//! and the whole point of this module — is that it **learns** the rotation used
//! before 1-bit quantization from the training-data distribution, rather than
//! fixing an unlearned random rotation (as the sibling [`crate::rabitq`] module
//! does).
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ItqConfig`] | Code length (`num_bits`), iteration/convergence budget, seeds |
//! | [`linalg`] | Symmetric Jacobi eigensolver, `k x k` SVD, Procrustes, random orthogonal matrix |
//! | [`ItqHasher`] | PCA + alternating-minimization training, per-vector encoding |
//! | [`ItqCode`] | A packed binary code (+ Hamming distance) |
//! | [`ItqIndex`] | Builds codes for a corpus and runs brute-force Hamming top-k search |
//! | [`ItqHit`] | A ranked search result (`id` + `hamming_distance`) |
//! | [`ItqError`] | Error type for configuration and runtime failures |
//!
//! # The algorithm
//!
//! ## 1. PCA dimensionality reduction
//!
//! The training matrix `X` (`n x D`) is centered, its `D x D` covariance
//! `C = (1/n) X_c^T X_c` is formed, and the top-`k` eigenvectors `V` (the
//! directions of greatest variance) are extracted with a pure-Rust cyclic
//! Jacobi eigensolver. The data is projected: `Z = X_c · V` (`n x k`).
//!
//! ## 2. Learning the rotation (alternating minimization)
//!
//! ITQ seeks a `k x k` orthogonal rotation `R` minimizing the quantization loss
//! `‖B − Z R‖_F^2`, where `B ∈ {-1, +1}^{n×k}` is the binary code matrix. `R`
//! is initialized to a deterministic pseudo-random orthogonal matrix, then
//! refined by alternating two closed-form minimizations until convergence:
//!
//! - **B-step** (fix `R`): `B = sign(Z R)`, thresholding each coordinate at
//!   zero with the `sign(0) := +1` convention.
//! - **R-step** (fix `B`): the *orthogonal Procrustes problem* — minimize
//!   `‖B − Z R‖_F` subject to `R^T R = I`. With `M = Z^T B = U Σ V_m^T`, the
//!   optimum is `R = U V_m^T`.
//!
//! Each step is a global minimizer of its subproblem, so the objective is
//! monotonically non-increasing across iterations (recorded in
//! [`ItqHasher::loss_history`]).
//!
//! ## 3. Encoding and search
//!
//! A new vector `x` is centered, projected (`z = (x − mu) · V`), rotated
//! (`t = z R`), and thresholded: bit `i` is set when `t_i >= 0`. Similarity is
//! ranked by Hamming distance between codes.
//!
//! # No external randomness / no external linear algebra
//!
//! The initial rotation is derived deterministically from a `u64` seed (FNV-1a
//! hashing → Box–Muller → modified Gram–Schmidt), and all of PCA, the SVD, and
//! Procrustes are implemented from scratch in [`linalg`]; there is no `rand`,
//! `ndarray`, or BLAS/LAPACK dependency.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "itq-hashing")] {
//! use oxirag::itq_hashing::{ItqConfig, ItqIndex};
//!
//! // Two well-separated clusters of 4-dimensional vectors.
//! let items = vec![
//!     ("a".to_string(), vec![10.0_f32, 0.0, 0.0, 0.0]),
//!     ("b".to_string(), vec![9.5_f32, 0.2, 0.0, 0.0]),
//!     ("c".to_string(), vec![0.0_f32, 0.0, 10.0, 0.0]),
//!     ("d".to_string(), vec![0.0_f32, 0.1, 9.7, 0.0]),
//! ];
//!
//! let config = ItqConfig::new(4).unwrap();
//! let index = ItqIndex::build(items, config).unwrap();
//!
//! // Query close to the first cluster.
//! let hits = index.search(&[10.0_f32, 0.1, 0.0, 0.0], 2).unwrap();
//! assert_eq!(hits.len(), 2);
//! # }
//! ```

pub mod hasher;
pub mod index;
pub mod linalg;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use hasher::ItqHasher;
pub use index::ItqIndex;
pub use types::{DEFAULT_ITQ_SEED, ItqCode, ItqConfig, ItqError, ItqHit};
