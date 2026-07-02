//! `EigenScore` hallucination detection via the differential entropy of the
//! sampled-response embedding covariance ("INSIDE", Chen et al., ICLR 2024,
//! *"INSIDE: LLMs' Internal States Retain the Power of Hallucination
//! Detection"*).
//!
//! `EigenScore` measures **self-consistency** across `K` sampled responses to
//! the same prompt: sample the model `K` times, embed each response, and look
//! at how "spread out" the `K` embeddings are. A confident, non-hallucinating
//! model tends to produce responses that cluster tightly (low spread); an
//! uncertain, hallucinating model produces responses that scatter across
//! embedding space (high spread). `EigenScore` turns that spread into a single
//! number via the *differential entropy* of the embeddings' covariance.
//!
//! # Algorithm
//!
//! Given `K` response embeddings `Z = [z_1, ..., z_K]` (each of dimension
//! `D`):
//!
//! 1. **Feature clipping (optional).** If
//!    [`EigenScoreConfig::clip_percentile`] `= Some(p)`, each coordinate of
//!    each `z_i` is clipped to the empirical `[p, 1-p]` quantile band computed
//!    across the `K` samples, suppressing outlier dimensions before anything
//!    else happens.
//! 2. **Center**: `Z_c = Z - mean(Z)`, i.e. subtract the mean embedding
//!    (across the `K` samples) from every `z_i`. Equivalently, apply the
//!    centering matrix `J_K = I_K - (1/K) * 1 * 1^T` on the left.
//! 3. **Gram matrix**: form the `K x K` matrix `G = Z_c * Z_c^T`. Because `K`
//!    is small (typically 5-20), this is far cheaper than working with the
//!    `D x D` covariance directly, and shares the same non-zero eigenvalues.
//! 4. **Regularize**: `Sigma = (1/K) * G + alpha * I_K`, with small `alpha`
//!    (default `1e-3`, [`EigenScoreConfig::regularization`]) to keep `Sigma`
//!    strictly positive-definite even when the `K` embeddings are exactly
//!    collinear.
//! 5. **Eigen-decompose** the symmetric positive-semidefinite `Sigma` via the
//!    cyclic-Jacobi algorithm ([`symmetric_eigenvalues`]; see [`jacobi`] for
//!    the rotation math) to obtain eigenvalues `lambda_1, ..., lambda_K`.
//! 6. **`EigenScore`** `= (1/K) * sum_i ln(lambda_i)` — the mean log-eigenvalue,
//!    i.e. `(1/K) * ln(det(Sigma))`, the differential-entropy term of a
//!    `K`-dimensional Gaussian with covariance `Sigma` (up to the constant
//!    `(K/2) * ln(2 * pi * e)` that INSIDE drops since it does not depend on
//!    the samples).
//!
//! Low `EigenScore` means the `K` samples embed into a tight, low-volume
//! ellipsoid (`Sigma` close to singular, small eigenvalues, very negative
//! log) — self-consistent, likely faithful. High `EigenScore` means the
//! samples spread across a large-volume region (larger eigenvalues, less
//! negative or positive log) — inconsistent, likely hallucinated.
//! [`EigenScoreResult::is_hallucination`] thresholds the score at
//! [`EigenScoreConfig::hallucination_threshold`].
//!
//! A text-only convenience path,
//! [`EigenScoreDetector::score_responses`], derives the `K` embeddings
//! directly from `K` response strings using a deterministic FNV-1a
//! character-trigram hash (mirroring the pseudo-embedding approach used by
//! the `semantic_router` module), so the detector can be exercised without a
//! real embedding model.
//!
//! # Distinctness
//!
//! `EigenScore` is a genuinely different signal from the codebase's other
//! hallucination/consistency modules:
//!
//! | Module | Signal |
//! |--------|--------|
//! | `hallucination_detector` | Lexical support of each *claim* against source *documents* |
//! | `self_consistency` | Majority vote over *clustered final answers* from `K` reasoning paths |
//! | `semantic_entropy` | Shannon entropy over *discrete meaning clusters* of `K` sampled answers |
//! | **`eigenscore`** | Differential entropy (log-determinant) of the **continuous embedding covariance** of `K` sampled responses — no clustering, no source comparison, just the geometry of the sample cloud |
//!
//! # Example
//!
//! ```
//! use oxirag::eigenscore::{EigenScoreConfig, EigenScoreDetector};
//!
//! let detector = EigenScoreDetector::new(EigenScoreConfig::default());
//!
//! // Three paraphrases of the same fact: the model is self-consistent.
//! let consistent = [
//!     "Paris is the capital of France.",
//!     "The capital of France is Paris.",
//!     "France's capital city is Paris.",
//! ];
//! let consistent_result = detector.score_responses(&consistent).expect(">= 2 samples");
//!
//! // Three wildly divergent guesses: the model is inconsistent / uncertain.
//! let divergent = [
//!     "Paris is the capital of France.",
//!     "Tokyo is a bustling megacity in Japan.",
//!     "Mitochondria are the powerhouse of the cell.",
//! ];
//! let divergent_result = detector.score_responses(&divergent).expect(">= 2 samples");
//!
//! // Tightly clustered responses embed into a lower-volume ellipsoid, so
//! // their EigenScore (mean log-eigenvalue) is lower.
//! assert!(consistent_result.score < divergent_result.score);
//! assert_eq!(consistent_result.eigenvalues.len(), 3);
//! ```

pub mod detector;
pub mod jacobi;
#[cfg(test)]
mod tests;
pub mod types;

pub use detector::EigenScoreDetector;
pub use jacobi::symmetric_eigenvalues;
pub use types::{EigenScoreConfig, EigenScoreError, EigenScoreResult};
