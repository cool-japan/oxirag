//! Dense symmetric-positive-definite linear algebra for closed-form knowledge edits.
//!
//! Everything here is hand-rolled `std`-only `f64` arithmetic over **row-major**
//! matrices stored flat in a `Vec<f64>` (element `(i, j)` of an `r × c` matrix lives at
//! `i * c + j`). The crate's `SciRS2` policy forbids an `ndarray` dependency, and the
//! dimensions a knowledge edit works at — the key width `d` of one transformer `MLP`
//! projection and the value width `m` of its output — are small enough that the
//! `O(d^2)` / `O(d^3)` kernels below are already the right shape.
//!
//! # Why this file exists rather than reusing `bandit_ranker::linalg`
//!
//! `crate::bandit_ranker` already ships a strong dense-`f64` kernel set with the same
//! row-major convention, and two of its functions are *literally* the two halves of the
//! rank-1 edit formula. It is nevertheless not imported here, for the same reason
//! `knowledge_unlearning` hand-rolls its own `MinHash` rather than importing
//! `semantic_dedup`'s: `bandit-ranker` and `knowledge-editing` are **independent Cargo
//! features**. A `use crate::bandit_ranker::linalg::…` in production code would silently
//! entangle them, so that building with `--features knowledge-editing` alone would fail
//! to compile. Feature independence is a compile-time property that only a hand-rolled
//! copy preserves.
//!
//! The two modules also want *different* kernels. A bandit maintains `A^-1` incrementally
//! and never factors `A`; a knowledge edit is a one-shot solve against a **fixed**
//! second-moment matrix `C = E[k k^T]` that is known up front and reused for every edit.
//! The right tool for that is a Cholesky factorization computed once and applied by
//! triangular substitution forever after — not a Sherman–Morrison chain, and emphatically
//! not an explicit inverse.
//!
//! # The one identity that makes the rank-1 edit safe
//!
//! The rank-1 edit divides by the quadratic form `q = k^T C^-1 k`. Written naively — form
//! `C^-1`, then contract it against `k` twice — `q` is a sum of `d^2` signed products, and
//! for a `k` that is nearly `C`-orthogonal to everything the rounding error of that sum is
//! the same size as the true value, so the computed `q` can come out *negative* and the
//! edit divides by garbage. That is a real failure mode and it is entirely avoidable.
//!
//! Factor `C = L L^T` once. Then
//!
//! ```text
//! q = k^T C^-1 k = k^T L^-T L^-1 k = (L^-1 k)^T (L^-1 k) = || L^-1 k ||^2
//! ```
//!
//! and `L^-1 k` is one forward substitution. Computed this way `q` is a **sum of squares**:
//! it is non-negative *structurally*, not by a clamp, and it is strictly positive for every
//! `k != 0` because `L` is invertible. [`spd_quadratic_form`] is exactly this, and it is
//! why [`super::RankOneEdit`] needs no `max(0.0)` fudge anywhere and can treat a non-positive
//! `q` as *proof* that the edit key is the zero vector rather than as rounding noise to be
//! papered over.
//!
//! The same factor applies `C^-1` to a vector — [`spd_solve`] — with a forward and a
//! backward substitution, at `O(d^2)`, and with the backward-error guarantees of triangular
//! solves rather than those of an explicitly formed inverse.
//!
//! # What is deliberately absent
//!
//! There is no `inverse()` in the production surface. The test module hand-rolls a
//! Gauss–Jordan inverse as an **independent oracle** to check the Cholesky path against;
//! shipping one would invite exactly the "just invert `C`, it's small" regression this file
//! exists to prevent.

use thiserror::Error;

/// The base ridge multiplier used by [`cholesky_with_jitter`], relative to the magnitude of
/// the matrix being factored.
pub const BASE_CHOLESKY_JITTER: f64 = 1e-12;

/// How many escalating ridges [`cholesky_with_jitter`] tries before giving up. Each attempt
/// multiplies the ridge by ten, so the ladder spans fourteen orders of magnitude — from a
/// ridge beneath the rounding noise up to one that dominates the matrix entirely.
pub const MAX_CHOLESKY_JITTER_ATTEMPTS: usize = 14;

/// Errors raised by the dense kernels in this module.
///
/// Deliberately **not** named `LinalgError`: that name is already taken by
/// `bandit_ranker::LinalgError`, which the crate re-exports from its flat prelude. Two types
/// with the same name in one prelude is a collision that only bites at the point of use.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum KnowledgeEditLinalgError {
    /// A matrix or vector argument did not have the length its declared shape requires.
    #[error("dimension mismatch in {what}: expected {expected} elements, got {actual}")]
    DimensionMismatch {
        /// Which argument was malformed.
        what: &'static str,
        /// The number of elements the operation required.
        expected: usize,
        /// The number of elements it actually received.
        actual: usize,
    },
    /// An input, or a computed result, contained a `NaN` or an infinity. Every kernel here
    /// rejects non-finite values rather than letting them contaminate a memory matrix, from
    /// which they could never be recovered.
    #[error("non-finite value encountered in {what}")]
    NonFinite {
        /// Which quantity was non-finite.
        what: &'static str,
    },
    /// Cholesky factorization hit a non-positive pivot: the input is not positive definite.
    /// [`cholesky_with_jitter`] recovers from this by adding a ridge; [`cholesky_lower`]
    /// reports it.
    #[error("matrix is not positive definite: non-positive pivot {pivot} at index {index}")]
    NotPositiveDefinite {
        /// The diagonal index at which the factorization failed.
        index: usize,
        /// The non-positive pivot found there.
        pivot: f64,
    },
    /// [`cholesky_with_jitter`] exhausted its ridge ladder, meaning the input was not merely
    /// *drifted* out of the positive-definite cone but grossly indefinite.
    #[error("cholesky failed after {attempts} jitter attempts (last ridge {last_jitter})")]
    JitterExhausted {
        /// How many ridges were tried.
        attempts: usize,
        /// The largest ridge that was tried.
        last_jitter: f64,
    },
    /// A dimension of `0` was supplied. Every routine here needs at least one row.
    #[error("dimension must be at least 1")]
    ZeroDimension,
}

/// Build the `dim × dim` matrix `scale * I`, row-major.
#[must_use]
pub fn scaled_identity(dim: usize, scale: f64) -> Vec<f64> {
    let mut matrix = vec![0.0; dim * dim];
    for i in 0..dim {
        matrix[i * dim + i] = scale;
    }
    matrix
}

/// Dot product of two equal-length vectors.
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::DimensionMismatch`] if the lengths differ.
pub fn dot(lhs: &[f64], rhs: &[f64]) -> Result<f64, KnowledgeEditLinalgError> {
    if lhs.len() != rhs.len() {
        return Err(KnowledgeEditLinalgError::DimensionMismatch {
            what: "dot product operands",
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }
    Ok(lhs.iter().zip(rhs).map(|(a, b)| a * b).sum())
}

/// Euclidean norm of a vector.
#[must_use]
pub fn l2_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Euclidean distance between two equal-length vectors.
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::DimensionMismatch`] if the lengths differ.
pub fn l2_distance(lhs: &[f64], rhs: &[f64]) -> Result<f64, KnowledgeEditLinalgError> {
    if lhs.len() != rhs.len() {
        return Err(KnowledgeEditLinalgError::DimensionMismatch {
            what: "distance operands",
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt())
}

/// Sum of the squares of every entry of a matrix or vector — the squared Frobenius norm.
#[must_use]
pub fn frobenius_norm_sq(matrix: &[f64]) -> f64 {
    matrix.iter().map(|v| v * v).sum()
}

/// Matrix–vector product `M x` for a row-major `rows × cols` matrix.
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::DimensionMismatch`] if `matrix` is not `rows * cols` long or
/// `vector` is not `cols` long.
pub fn mat_vec(
    matrix: &[f64],
    vector: &[f64],
    rows: usize,
    cols: usize,
) -> Result<Vec<f64>, KnowledgeEditLinalgError> {
    check_matrix(matrix, rows, cols, "matrix")?;
    check_vector(vector, cols, "vector")?;
    let mut out = vec![0.0; rows];
    for (slot, row) in out.iter_mut().zip(matrix.chunks_exact(cols)) {
        let mut acc = 0.0;
        for (m, v) in row.iter().zip(vector) {
            acc += m * v;
        }
        *slot = acc;
    }
    Ok(out)
}

/// Accumulate a scaled outer product into a row-major `rows × cols` matrix:
/// `target += alpha * (left ⊗ right)`, i.e. `target[i][j] += alpha * left[i] * right[j]`.
///
/// This is how a rank-1 (or rank-`E`) edit is written into a memory matrix — the delta is
/// never materialized densely on the production path.
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::DimensionMismatch`] on a malformed argument, or
/// [`KnowledgeEditLinalgError::NonFinite`] if the accumulation would introduce a `NaN` or an
/// infinity into `target`.
pub fn add_scaled_outer_product(
    target: &mut [f64],
    alpha: f64,
    left: &[f64],
    right: &[f64],
    rows: usize,
    cols: usize,
) -> Result<(), KnowledgeEditLinalgError> {
    check_matrix(target, rows, cols, "outer-product target")?;
    check_vector(left, rows, "outer-product left factor")?;
    check_vector(right, cols, "outer-product right factor")?;
    if !alpha.is_finite() {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "outer-product scale",
        });
    }
    for (row, &l) in target.chunks_exact_mut(cols).zip(left) {
        let scaled = alpha * l;
        for (slot, &r) in row.iter_mut().zip(right) {
            *slot += scaled * r;
        }
    }
    if target.iter().any(|v| !v.is_finite()) {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "outer-product result",
        });
    }
    Ok(())
}

/// Replace `matrix` with `(matrix + matrix^T) / 2` in place — the orthogonal projection onto
/// the symmetric matrices.
///
/// An empirical second moment `sum_i k_i k_i^T` is symmetric in exact arithmetic but picks up
/// asymmetric rounding as the sum accumulates. Projecting back is the identity mathematically,
/// costs `O(d^2)`, and means the Cholesky below reads a matrix that does not depend on which
/// half of the drift it happened to look at.
pub fn symmetrize(matrix: &mut [f64], dim: usize) {
    for i in 0..dim {
        for j in (i + 1)..dim {
            let mean = 0.5 * (matrix[i * dim + j] + matrix[j * dim + i]);
            matrix[i * dim + j] = mean;
            matrix[j * dim + i] = mean;
        }
    }
}

/// Lower-triangular Cholesky factor `L` of a symmetric positive-definite `M`, with
/// `L L^T = M`. Entries strictly above the diagonal of the returned row-major matrix are zero.
///
/// The standard Cholesky–Banachiewicz recurrence, row by row:
///
/// ```text
/// L[i][j] = (M[i][j] - sum_{k < j} L[i][k] L[j][k]) / L[j][j]     (j < i)
/// L[i][i] = sqrt(M[i][i] - sum_{k < i} L[i][k]^2)
/// ```
///
/// Only the lower triangle of `M` is read.
///
/// # Errors
///
/// * [`KnowledgeEditLinalgError::ZeroDimension`] / [`KnowledgeEditLinalgError::DimensionMismatch`]
///   on a malformed argument.
/// * [`KnowledgeEditLinalgError::NonFinite`] if `M` contains a `NaN` or an infinity.
/// * [`KnowledgeEditLinalgError::NotPositiveDefinite`] the moment a diagonal pivot comes out
///   non-positive — guarded *before* the `sqrt`, because `sqrt` of a negative is a `NaN` that
///   would then propagate into every subsequent solve. Use [`cholesky_with_jitter`] when a
///   merely-drifted matrix should be repaired instead of rejected.
pub fn cholesky_lower(matrix: &[f64], dim: usize) -> Result<Vec<f64>, KnowledgeEditLinalgError> {
    if dim == 0 {
        return Err(KnowledgeEditLinalgError::ZeroDimension);
    }
    check_matrix(matrix, dim, dim, "cholesky input")?;
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "cholesky input matrix",
        });
    }

    let mut lower = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut acc = matrix[i * dim + j];
            for k in 0..j {
                acc -= lower[i * dim + k] * lower[j * dim + k];
            }
            if i == j {
                if !acc.is_finite() || acc <= 0.0 {
                    return Err(KnowledgeEditLinalgError::NotPositiveDefinite {
                        index: i,
                        pivot: acc,
                    });
                }
                lower[i * dim + i] = acc.sqrt();
            } else {
                let pivot = lower[j * dim + j];
                if pivot == 0.0 {
                    return Err(KnowledgeEditLinalgError::NotPositiveDefinite {
                        index: j,
                        pivot: 0.0,
                    });
                }
                lower[i * dim + j] = acc / pivot;
            }
        }
    }
    Ok(lower)
}

/// Cholesky factor of `M`, adding an escalating diagonal ridge if and only if `M` is not
/// factorizable as given. Returns `(L, jitter)` with `L L^T = M + jitter * I`, and
/// `jitter == 0.0` when no repair was needed.
///
/// # Why the ridge is scale-aware
///
/// A fixed `1e-12` ridge is meaningless without knowing the scale of `M`: it is a colossal
/// perturbation to a matrix whose entries are `1e-15`, and it is beneath the rounding noise of
/// one whose entries are `1e6`. The ridge is therefore measured in units of `M`'s own
/// magnitude — the mean diagonal entry, floored at one so a zero matrix still gets a usable
/// ridge. Attempt `k` tries `BASE_CHOLESKY_JITTER * scale * 10^k`.
///
/// The applied `jitter` is *returned* rather than swallowed, so a caller (or a test) can see
/// whether the repair path engaged at all and how hard it had to push. A second-moment matrix
/// `C = E[k k^T] + ridge * I` built by [`super::EditableMemory::from_preserved_keys`] should
/// always factor with `jitter == 0.0`; a non-zero jitter there is a signal worth surfacing,
/// and it is surfaced — see [`super::EditResult::cholesky_jitter`].
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::JitterExhausted`] if even the largest ridge in the ladder leaves
/// `M` unfactorizable, plus the dimension and finiteness errors of [`cholesky_lower`].
pub fn cholesky_with_jitter(
    matrix: &[f64],
    dim: usize,
) -> Result<(Vec<f64>, f64), KnowledgeEditLinalgError> {
    if dim == 0 {
        return Err(KnowledgeEditLinalgError::ZeroDimension);
    }
    check_matrix(matrix, dim, dim, "cholesky input")?;

    match cholesky_lower(matrix, dim) {
        Ok(lower) => return Ok((lower, 0.0)),
        Err(KnowledgeEditLinalgError::NotPositiveDefinite { .. }) => {}
        Err(other) => return Err(other),
    }

    let trace: f64 = (0..dim).map(|i| matrix[i * dim + i]).sum();
    #[allow(clippy::cast_precision_loss)] // `dim` is a memory width: hundreds at most.
    let mean_diagonal = trace / dim as f64;
    if !mean_diagonal.is_finite() {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "cholesky input matrix trace",
        });
    }
    let scale = mean_diagonal.abs().max(1.0);

    let mut ridged = matrix.to_vec();
    let mut previous_jitter = 0.0;
    let mut jitter = BASE_CHOLESKY_JITTER * scale;
    for _ in 0..MAX_CHOLESKY_JITTER_ATTEMPTS {
        // Adjust the diagonal *incrementally* so the ridge never accumulates: after this loop
        // the diagonal holds `matrix[i][i] + jitter` exactly once, whatever the last attempt
        // left behind.
        let delta = jitter - previous_jitter;
        for i in 0..dim {
            ridged[i * dim + i] += delta;
        }
        match cholesky_lower(&ridged, dim) {
            Ok(lower) => return Ok((lower, jitter)),
            Err(KnowledgeEditLinalgError::NotPositiveDefinite { .. }) => {
                previous_jitter = jitter;
                jitter *= 10.0;
            }
            Err(other) => return Err(other),
        }
    }
    Err(KnowledgeEditLinalgError::JitterExhausted {
        attempts: MAX_CHOLESKY_JITTER_ATTEMPTS,
        last_jitter: previous_jitter,
    })
}

/// Forward substitution: solve `L y = b` for a lower-triangular `L`, returning `y`.
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::DimensionMismatch`] on a malformed argument;
/// [`KnowledgeEditLinalgError::NotPositiveDefinite`] if a diagonal entry of `L` is zero or
/// non-finite (which for a genuine Cholesky factor is impossible — every pivot was checked
/// strictly positive when the factor was built);
/// [`KnowledgeEditLinalgError::NonFinite`] if the solve overflows.
pub fn solve_lower(
    lower: &[f64],
    b: &[f64],
    dim: usize,
) -> Result<Vec<f64>, KnowledgeEditLinalgError> {
    check_matrix(lower, dim, dim, "lower-triangular factor")?;
    check_vector(b, dim, "right-hand side")?;
    let mut y = vec![0.0; dim];
    for i in 0..dim {
        let mut acc = b[i];
        for j in 0..i {
            acc -= lower[i * dim + j] * y[j];
        }
        let pivot = lower[i * dim + i];
        if !pivot.is_finite() || pivot == 0.0 {
            return Err(KnowledgeEditLinalgError::NotPositiveDefinite { index: i, pivot });
        }
        y[i] = acc / pivot;
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "forward substitution result",
        });
    }
    Ok(y)
}

/// Backward substitution against the *transpose* of a lower-triangular factor: solve
/// `L^T x = y` for `x`, exploiting `(L^T)[i][j] == L[j][i]` rather than materializing `L^T`.
///
/// # Errors
///
/// The same conditions as [`solve_lower`].
pub fn solve_upper_transposed(
    lower: &[f64],
    y: &[f64],
    dim: usize,
) -> Result<Vec<f64>, KnowledgeEditLinalgError> {
    check_matrix(lower, dim, dim, "lower-triangular factor")?;
    check_vector(y, dim, "right-hand side")?;
    let mut x = vec![0.0; dim];
    for i in (0..dim).rev() {
        let mut acc = y[i];
        for j in (i + 1)..dim {
            acc -= lower[j * dim + i] * x[j];
        }
        let pivot = lower[i * dim + i];
        if !pivot.is_finite() || pivot == 0.0 {
            return Err(KnowledgeEditLinalgError::NotPositiveDefinite { index: i, pivot });
        }
        x[i] = acc / pivot;
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "backward substitution result",
        });
    }
    Ok(x)
}

/// Apply `C^-1` to a vector, given the Cholesky factor `L` of `C = L L^T`:
/// `C^-1 b = L^-T (L^-1 b)`, one forward and one backward substitution, `O(d^2)`.
///
/// This is the *only* way `C^-1` is ever applied on the production path. `C^-1` is never
/// formed.
///
/// # Errors
///
/// The same conditions as [`solve_lower`].
pub fn spd_solve(
    lower: &[f64],
    b: &[f64],
    dim: usize,
) -> Result<Vec<f64>, KnowledgeEditLinalgError> {
    let y = solve_lower(lower, b, dim)?;
    solve_upper_transposed(lower, &y, dim)
}

/// The quadratic form `b^T C^-1 b`, given the Cholesky factor `L` of `C = L L^T`.
///
/// Computed as `|| L^-1 b ||^2` — a **sum of squares**, hence non-negative *structurally*
/// rather than by a clamp, and strictly positive for every `b != 0` because `L` is invertible.
/// See the [module documentation](self) for why this matters: it is what lets the rank-1 edit
/// treat a non-positive denominator as proof of a zero key rather than as rounding noise.
///
/// # Errors
///
/// The same conditions as [`solve_lower`], plus [`KnowledgeEditLinalgError::NonFinite`] if the
/// result overflows.
pub fn spd_quadratic_form(
    lower: &[f64],
    b: &[f64],
    dim: usize,
) -> Result<f64, KnowledgeEditLinalgError> {
    let y = solve_lower(lower, b, dim)?;
    let value: f64 = y.iter().map(|v| v * v).sum();
    if !value.is_finite() {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "quadratic form b^T C^-1 b",
        });
    }
    Ok(value)
}

/// `L^T x` for a lower-triangular `L`, exploiting `(L^T)[i][j] == L[j][i]`.
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::DimensionMismatch`] on a malformed argument.
pub fn transposed_lower_mat_vec(
    lower: &[f64],
    x: &[f64],
    dim: usize,
) -> Result<Vec<f64>, KnowledgeEditLinalgError> {
    check_matrix(lower, dim, dim, "lower-triangular factor")?;
    check_vector(x, dim, "vector")?;
    let mut out = vec![0.0; dim];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &xj) in x.iter().enumerate().skip(i) {
            acc += lower[j * dim + i] * xj;
        }
        *slot = acc;
    }
    Ok(out)
}

/// The `C`-weighted squared Frobenius norm of a `rows × cols` matrix `D`:
/// `tr(D C D^T)`, computed from the Cholesky factor `L` of `C = L L^T`.
///
/// Since `tr(D C D^T) = sum_i (row_i^T L) (L^T row_i) = sum_i || L^T row_i ||^2`, this too is
/// a **sum of squares** and cannot come out negative.
///
/// This quantity is the whole point of the module: for the delta `D = W' - W` produced by a
/// closed-form edit against the second-moment matrix `C = E[k k^T] + ridge * I` of a preserved
/// key set of size `N`,
///
/// ```text
/// tr(D C D^T) = (1/N) * sum_i || D k_i ||^2  +  ridge * || D ||_F^2
/// ```
///
/// so `sqrt(tr(D C D^T))` is an exact upper bound on the root-mean-square collateral drift the
/// edit inflicts on the preserved keys, and the slack in that bound is exactly
/// `ridge * ||D||_F^2`. That identity is asserted directly in this module's tests.
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::DimensionMismatch`] on a malformed argument;
/// [`KnowledgeEditLinalgError::NonFinite`] if the result overflows.
pub fn weighted_frobenius_norm_sq_from_cholesky(
    delta: &[f64],
    lower: &[f64],
    rows: usize,
    cols: usize,
) -> Result<f64, KnowledgeEditLinalgError> {
    check_matrix(delta, rows, cols, "weighted-norm matrix")?;
    check_matrix(lower, cols, cols, "weighted-norm cholesky factor")?;
    let mut total = 0.0;
    for row in delta.chunks_exact(cols) {
        let projected = transposed_lower_mat_vec(lower, row, cols)?;
        total += projected.iter().map(|v| v * v).sum::<f64>();
    }
    if !total.is_finite() {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "weighted Frobenius norm",
        });
    }
    Ok(total)
}

/// The `C`-weighted squared Frobenius norm `tr(D C D^T)` computed **directly from `C`**,
/// without a factorization.
///
/// Structurally different from [`weighted_frobenius_norm_sq_from_cholesky`] — a sum of signed
/// products rather than a sum of squares — and therefore useful as an independent cross-check
/// of it. The Cholesky route is the one the production path uses.
///
/// # Errors
///
/// [`KnowledgeEditLinalgError::DimensionMismatch`] on a malformed argument;
/// [`KnowledgeEditLinalgError::NonFinite`] if the result overflows.
pub fn weighted_frobenius_norm_sq(
    delta: &[f64],
    covariance: &[f64],
    rows: usize,
    cols: usize,
) -> Result<f64, KnowledgeEditLinalgError> {
    check_matrix(delta, rows, cols, "weighted-norm matrix")?;
    check_matrix(covariance, cols, cols, "weighted-norm covariance")?;
    let mut total = 0.0;
    for row in delta.chunks_exact(cols) {
        let product = mat_vec(covariance, row, cols, cols)?;
        total += dot(row, &product)?;
    }
    if !total.is_finite() {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "weighted Frobenius norm",
        });
    }
    Ok(total)
}

/// `trace(C^-1)` from the Cholesky factor `L` of `C = L L^T`.
///
/// `C^-1 = L^-T L^-1`, so `trace(C^-1) = || L^-1 ||_F^2` — computed by solving `L X = I` one
/// unit column at a time and summing the squares. `O(d^3 / 2)`.
///
/// This is used to state a *rigorous, computable* bound on the collateral drift of a rank-`E`
/// multi-edit. For an `SPD` Gram matrix `G`, every eigenvalue of `G^-1` is at most
/// `trace(G^-1)` (they are all positive and they sum to the trace), so
/// `tr(R G^-1 R^T) <= trace(G^-1) * ||R||_F^2` for any `R`. The left-hand side is the exact
/// squared `C`-weighted norm of the multi-edit; the right-hand side is a bound a caller can
/// evaluate *before* committing the edit.
///
/// # Errors
///
/// The same conditions as [`solve_lower`], plus [`KnowledgeEditLinalgError::NonFinite`] if the
/// result overflows.
pub fn inverse_trace_from_cholesky(
    lower: &[f64],
    dim: usize,
) -> Result<f64, KnowledgeEditLinalgError> {
    check_matrix(lower, dim, dim, "lower-triangular factor")?;
    let mut total = 0.0;
    for column in 0..dim {
        let mut unit = vec![0.0; dim];
        unit[column] = 1.0;
        let solved = solve_lower(lower, &unit, dim)?;
        total += solved.iter().map(|v| v * v).sum::<f64>();
    }
    if !total.is_finite() {
        return Err(KnowledgeEditLinalgError::NonFinite {
            what: "trace of the inverse",
        });
    }
    Ok(total)
}

/// Verify a slice is a `rows × cols` row-major matrix.
fn check_matrix(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    what: &'static str,
) -> Result<(), KnowledgeEditLinalgError> {
    if matrix.len() == rows * cols {
        Ok(())
    } else {
        Err(KnowledgeEditLinalgError::DimensionMismatch {
            what,
            expected: rows * cols,
            actual: matrix.len(),
        })
    }
}

/// Verify a slice is a `dim`-vector.
fn check_vector(
    vector: &[f64],
    dim: usize,
    what: &'static str,
) -> Result<(), KnowledgeEditLinalgError> {
    if vector.len() == dim {
        Ok(())
    } else {
        Err(KnowledgeEditLinalgError::DimensionMismatch {
            what,
            expected: dim,
            actual: vector.len(),
        })
    }
}
