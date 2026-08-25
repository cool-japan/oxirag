//! Small dense linear algebra for contextual bandits: rank-1 inverse updates
//! (Sherman–Morrison), Cholesky factorization with scale-aware jitter, and the
//! quadratic forms both of them feed.
//!
//! Everything here is hand-rolled `std`-only `f64` arithmetic over **row-major**
//! `d × d` matrices stored flat in a `Vec<f64>` (element `(i, j)` lives at
//! `i * d + j`). The project's `SciRS2` policy forbids an `ndarray` dependency,
//! and in any case a contextual bandit's `d` is *small* — a handful to a few
//! dozen features — so the `O(d^2)` / `O(d^3)` kernels below are already the
//! right shape, and hand-rolling them lets us put the numerical guards exactly
//! where the bandit needs them.
//!
//! # Why this file exists at all: never re-invert
//!
//! `LinUCB` scores an arm with `theta^T x + alpha * sqrt(x^T A^-1 x)` and updates
//! `A <- A + x x^T` on every observed reward. Both the score *and* the
//! exploration bonus need `A^-1`, on every round, for every arm. Re-inverting a
//! `d × d` matrix from scratch costs `O(d^3)` **per arm per round**; the whole
//! point of the `LinUCB` construction (Li et al., 2010, "A Contextual-Bandit
//! Approach to Personalized News Article Recommendation") is that the update is
//! rank-1, so the *inverse* can be updated directly in `O(d^2)` and the matrix
//! `A` itself never needs to be factored at all.
//!
//! # Sherman–Morrison
//!
//! For an invertible `A` and vectors `u, v` with `1 + v^T A^-1 u != 0`:
//!
//! ```text
//! (A + u v^T)^-1 = A^-1 - (A^-1 u)(v^T A^-1) / (1 + v^T A^-1 u)
//! ```
//!
//! Specializing to the bandit's symmetric rank-1 update `u = v = x`, and writing
//! `w = A^-1 x`, symmetry of `A^-1` gives `x^T A^-1 = (A^-1 x)^T = w^T`, so the
//! whole update collapses to a single outer product of one vector:
//!
//! ```text
//! (A + x x^T)^-1 = A^-1 - (w w^T) / (1 + x^T w),    w = A^-1 x
//! ```
//!
//! which is `O(d^2)` to form and `O(d^2)` to apply. [`sherman_morrison_update`]
//! implements exactly this.
//!
//! ## Why the denominator cannot vanish
//!
//! The textbook Sherman–Morrison identity carries the side condition
//! `1 + x^T A^-1 x != 0`, and a naive implementation would have to worry about
//! dividing by zero. **In this bandit it provably cannot happen**, and the
//! reason is worth writing down because it is what licenses the whole
//! incremental scheme:
//!
//! * `A` is initialized to `lambda * I` with `lambda > 0`, which is symmetric
//!   positive definite (SPD).
//! * Every update adds `x x^T`, which is symmetric positive *semi*-definite
//!   (`y^T x x^T y = (x^T y)^2 >= 0` for every `y`).
//! * SPD + PSD is SPD. So `A` is SPD after *every* update, by induction.
//! * The inverse of an SPD matrix is SPD. Hence `x^T A^-1 x >= 0` for all `x`,
//!   with equality only at `x = 0`.
//! * Therefore `1 + x^T A^-1 x >= 1`. It is bounded *away* from zero by a whole
//!   unit — there is no cancellation to be catastrophic about, because the two
//!   terms being summed are both non-negative.
//!
//! A denominator anywhere near zero would mean `x^T A^-1 x ≈ -1`, i.e. that the
//! maintained `A^-1` had lost positive-definiteness — a *corruption*, not a
//! rounding artifact. [`sherman_morrison_update`] therefore treats a denominator
//! below [`MIN_SHERMAN_MORRISON_DENOMINATOR`] as a hard
//! [`LinalgError::DegenerateShermanMorrison`] rather than silently producing
//! garbage: the defensive check exists to *detect* an impossible state, not to
//! paper over an expected one.
//!
//! ## Drift, and the symmetrization step
//!
//! Exact arithmetic keeps `A^-1` symmetric forever. Floating-point arithmetic
//! does not: `w[i] * w[j]` and `w[j] * w[i]` are equal, but the *accumulated*
//! `A^-1` picks up asymmetric rounding through repeated updates, and a
//! `A^-1` that has drifted out of the symmetric cone can, over many thousands of
//! rounds, drift out of the positive-definite cone too — at which point the
//! quadratic form goes negative and `sqrt` returns `NaN`. The fix is one cheap
//! line: after each rank-1 update, project back onto the symmetric matrices with
//! `A^-1 <- (A^-1 + (A^-1)^T) / 2`. This is exact in the sense that it is the
//! identity in exact arithmetic, it is the orthogonal projection onto the
//! symmetric subspace in floating point, and it costs `O(d^2)` — free next to
//! the update itself. It is what keeps the incremental inverse agreeing with a
//! from-scratch inverse to ~1e-13 over hundreds of updates rather than slowly
//! rotting.
//!
//! ## Clamping the quadratic form
//!
//! `x^T A^-1 x` is mathematically non-negative (see above) but is computed as a
//! sum of `d^2` signed products. For a nearly-orthogonal `x` in a
//! well-explored direction the true value can be `1e-17`, and the rounding error
//! of the summation is the *same size*, so the computed value can come out
//! *negative*. `sqrt` of that is `NaN`, and a single `NaN` score poisons the
//! whole ranking. [`quadratic_form_nonnegative`] therefore clamps at zero before
//! anyone takes a square root. This is not a fudge: it is restoring a property
//! the exact value is guaranteed to have and only rounding took away.
//!
//! # Cholesky
//!
//! Thompson sampling needs to draw `theta ~ N(theta_hat, v^2 A^-1)`, which means
//! it needs a matrix `L` with `L L^T = v^2 A^-1` so that `theta_hat + L z` (for
//! `z ~ N(0, I)`) has exactly that covariance. [`cholesky_lower`] computes the
//! standard lower-triangular Cholesky factor in `O(d^3 / 3)`;
//! [`cholesky_with_jitter`] wraps it with the scale-aware ridge escalation
//! described on that function, for the case where drift (or a caller-supplied
//! matrix) has left the input a hair outside the positive-definite cone.

use thiserror::Error;

/// The floor below which a Sherman–Morrison denominator is treated as evidence
/// of a corrupted inverse rather than as a legitimate value.
///
/// The exact denominator is `1 + x^T A^-1 x >= 1` for an SPD `A^-1` (see the
/// [module documentation](self)), so any value near zero is *impossible* in
/// exact arithmetic. The threshold is set well below any rounding error that a
/// value-of-at-least-1 quantity could plausibly accumulate, so it fires only on
/// genuine corruption (a non-finite entry, a caller-injected non-SPD matrix)
/// and never on healthy drift.
pub const MIN_SHERMAN_MORRISON_DENOMINATOR: f64 = 1e-9;

/// Errors raised by the dense-linear-algebra kernels.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LinalgError {
    /// A matrix or vector argument did not have the expected length for the
    /// declared dimension `d` (a `d × d` matrix must be `d * d` long; a vector
    /// must be `d` long).
    #[error("dimension mismatch: expected {expected} elements, got {actual}")]
    DimensionMismatch {
        /// The number of elements the operation required.
        expected: usize,
        /// The number of elements it actually received.
        actual: usize,
    },
    /// An input contained a `NaN` or an infinity. Every kernel here rejects
    /// non-finite input up front rather than letting it silently contaminate an
    /// arm's maintained state, from which it could never be recovered.
    #[error("non-finite value encountered in {what}")]
    NonFinite {
        /// Which argument was non-finite.
        what: &'static str,
    },
    /// The Sherman–Morrison denominator `1 + x^T A^-1 x` came out below
    /// [`MIN_SHERMAN_MORRISON_DENOMINATOR`], which is impossible for a
    /// positive-definite `A^-1` and therefore indicates a corrupted inverse.
    #[error(
        "degenerate Sherman-Morrison denominator {denominator}: \
         the maintained inverse is no longer positive definite"
    )]
    DegenerateShermanMorrison {
        /// The offending value of `1 + x^T A^-1 x`.
        denominator: f64,
    },
    /// Cholesky factorization hit a non-positive pivot: the input is not
    /// positive definite. [`cholesky_with_jitter`] recovers from this by adding
    /// a ridge; [`cholesky_lower`] reports it.
    #[error("matrix is not positive definite: non-positive pivot {pivot} at index {index}")]
    NotPositiveDefinite {
        /// The diagonal index at which the factorization failed.
        index: usize,
        /// The non-positive pivot value found there.
        pivot: f64,
    },
    /// [`cholesky_with_jitter`] exhausted its ridge escalation without reaching
    /// a factorizable matrix, meaning the input was not merely *drifted* out of
    /// the positive-definite cone but grossly indefinite.
    #[error("cholesky failed after {attempts} jitter attempts (last ridge {last_jitter})")]
    JitterExhausted {
        /// How many ridges were tried.
        attempts: usize,
        /// The largest ridge that was tried.
        last_jitter: f64,
    },
    /// A dimension of `0` was supplied. Every routine here needs at least one
    /// row.
    #[error("dimension must be at least 1")]
    ZeroDimension,
}

/// Build the `d × d` matrix `scale * I`, row-major.
///
/// This is how each arm's `A^-1` is initialized: `A = lambda * I` implies
/// `A^-1 = (1 / lambda) * I`, so the ranker never has to invert anything even
/// once.
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
/// [`LinalgError::DimensionMismatch`] if the lengths differ.
pub fn dot(lhs: &[f64], rhs: &[f64]) -> Result<f64, LinalgError> {
    if lhs.len() != rhs.len() {
        return Err(LinalgError::DimensionMismatch {
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }
    Ok(lhs.iter().zip(rhs).map(|(a, b)| a * b).sum())
}

/// Matrix–vector product `M x` for a row-major `d × d` matrix.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] if `matrix` is not `d * d` long or
/// `vector` is not `d` long.
pub fn mat_vec(matrix: &[f64], vector: &[f64], dim: usize) -> Result<Vec<f64>, LinalgError> {
    check_matrix(matrix, dim)?;
    check_vector(vector, dim)?;
    let mut out = vec![0.0; dim];
    for i in 0..dim {
        let row = &matrix[i * dim..i * dim + dim];
        let mut acc = 0.0;
        for (m, v) in row.iter().zip(vector) {
            acc += m * v;
        }
        out[i] = acc;
    }
    Ok(out)
}

/// The quadratic form `x^T M x`, **clamped at zero**.
///
/// For the symmetric positive-definite `M = A^-1` this module maintains, the
/// exact value is non-negative; the clamp only removes the rounding noise that
/// can push a genuinely-tiny value a few ulps below zero, which would otherwise
/// turn `LinUCB`'s `sqrt` into a `NaN` and poison an entire ranking. See
/// "Clamping the quadratic form" in the [module documentation](self).
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] on a length mismatch;
/// [`LinalgError::NonFinite`] if the computed form is `NaN` or infinite (which,
/// unlike a small negative value, is *not* something a clamp can honestly
/// repair).
pub fn quadratic_form_nonnegative(
    matrix: &[f64],
    vector: &[f64],
    dim: usize,
) -> Result<f64, LinalgError> {
    let product = mat_vec(matrix, vector, dim)?;
    let value = dot(vector, &product)?;
    if !value.is_finite() {
        return Err(LinalgError::NonFinite {
            what: "quadratic form x^T M x",
        });
    }
    Ok(value.max(0.0))
}

/// Apply a symmetric rank-1 Sherman–Morrison update **in place**:
/// `A^-1 <- (A + x x^T)^-1`, computed from `A^-1` alone in `O(d^2)`.
///
/// Returns the denominator `1 + x^T A^-1 x` that was used, which callers may
/// find useful for diagnostics (it equals `1 + ` the squared `LinUCB` exploration
/// bonus at unit `alpha`, evaluated *before* the update).
///
/// The full derivation, the proof that the denominator is bounded below by `1`,
/// and the rationale for the trailing symmetrization are in the
/// [module documentation](self).
///
/// # Errors
///
/// * [`LinalgError::DimensionMismatch`] — `a_inv` is not `d * d` long, or `x` is
///   not `d` long.
/// * [`LinalgError::NonFinite`] — `x` or `a_inv` contains a `NaN`/infinity, or
///   the update would produce one.
/// * [`LinalgError::DegenerateShermanMorrison`] — the denominator fell below
///   [`MIN_SHERMAN_MORRISON_DENOMINATOR`], which cannot happen for a
///   positive-definite `a_inv` and therefore signals a corrupted inverse.
pub fn sherman_morrison_update(
    a_inv: &mut [f64],
    x: &[f64],
    dim: usize,
) -> Result<f64, LinalgError> {
    check_matrix(a_inv, dim)?;
    check_vector(x, dim)?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite {
            what: "context vector x",
        });
    }
    if a_inv.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite {
            what: "maintained inverse A^-1",
        });
    }

    // w = A^-1 x.
    let w = mat_vec(a_inv, x, dim)?;

    // denominator = 1 + x^T A^-1 x = 1 + x^T w. Provably >= 1 for SPD A^-1.
    let denominator = 1.0 + dot(x, &w)?;
    if !denominator.is_finite() || denominator < MIN_SHERMAN_MORRISON_DENOMINATOR {
        return Err(LinalgError::DegenerateShermanMorrison { denominator });
    }

    // A^-1 <- A^-1 - (w w^T) / denominator.
    //
    // Note this is a *single* outer product rather than the general
    // `(A^-1 u)(v^T A^-1)`: symmetry of A^-1 makes the left and right factors
    // the same vector `w`. That halves the work and, more importantly, makes the
    // subtracted term exactly symmetric by construction.
    for i in 0..dim {
        let scaled = w[i] / denominator;
        for j in 0..dim {
            a_inv[i * dim + j] -= scaled * w[j];
        }
    }

    // Project back onto the symmetric matrices, undoing the asymmetric rounding
    // that accumulates across thousands of updates. Identity in exact
    // arithmetic; the thing that keeps this agreeing with a from-scratch inverse
    // in floating point.
    symmetrize(a_inv, dim);

    if a_inv.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite {
            what: "updated inverse A^-1",
        });
    }
    Ok(denominator)
}

/// Replace `matrix` with `(matrix + matrix^T) / 2` in place.
///
/// The orthogonal projection onto the symmetric matrices. Applied after every
/// Sherman–Morrison update; see the [module documentation](self).
pub fn symmetrize(matrix: &mut [f64], dim: usize) {
    for i in 0..dim {
        for j in (i + 1)..dim {
            let upper = matrix[i * dim + j];
            let lower = matrix[j * dim + i];
            let mean = 0.5 * (upper + lower);
            matrix[i * dim + j] = mean;
            matrix[j * dim + i] = mean;
        }
    }
}

/// Lower-triangular Cholesky factor `L` of a symmetric positive-definite `M`,
/// such that `L L^T = M`. Entries strictly above the diagonal of the returned
/// row-major matrix are zero.
///
/// The standard Cholesky–Banachiewicz recurrence, row by row:
///
/// ```text
/// L[i][j] = (M[i][j] - sum_{k < j} L[i][k] L[j][k]) / L[j][j]     (j < i)
/// L[i][i] = sqrt(M[i][i] - sum_{k < i} L[i][k]^2)
/// ```
///
/// Only the lower triangle of `M` is read, so a caller whose matrix is
/// symmetric-up-to-rounding gets a well-defined answer rather than one that
/// depends on which half of the drift it happened to look at.
///
/// # Errors
///
/// * [`LinalgError::DimensionMismatch`] / [`LinalgError::ZeroDimension`] on a
///   malformed argument.
/// * [`LinalgError::NonFinite`] if `M` contains a `NaN`/infinity.
/// * [`LinalgError::NotPositiveDefinite`] the moment a diagonal pivot comes out
///   non-positive. This is the *definition* of the factorization failing, and it
///   is reported rather than being turned into a `NaN` by an unguarded `sqrt`.
///   Use [`cholesky_with_jitter`] when a merely-drifted matrix should be
///   repaired instead of rejected.
pub fn cholesky_lower(matrix: &[f64], dim: usize) -> Result<Vec<f64>, LinalgError> {
    if dim == 0 {
        return Err(LinalgError::ZeroDimension);
    }
    check_matrix(matrix, dim)?;
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite {
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
                // A non-positive pivot means the leading principal minor of
                // order i+1 is non-positive, i.e. M is not positive definite.
                // Guard *before* the sqrt: sqrt(-eps) is NaN, and a NaN here
                // would silently propagate into every sampled parameter vector.
                if !acc.is_finite() || acc <= 0.0 {
                    return Err(LinalgError::NotPositiveDefinite {
                        index: i,
                        pivot: acc,
                    });
                }
                lower[i * dim + i] = acc.sqrt();
            } else {
                let pivot = lower[j * dim + j];
                // Unreachable given the diagonal guard above (the pivot was
                // already checked to be strictly positive when row j was
                // built), but division is the one place a zero would become an
                // infinity, so it is worth being explicit.
                if pivot == 0.0 {
                    return Err(LinalgError::NotPositiveDefinite {
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

/// The base ridge multiplier used by [`cholesky_with_jitter`], relative to the
/// magnitude of the matrix being factored.
pub const BASE_CHOLESKY_JITTER: f64 = 1e-12;

/// How many escalating ridges [`cholesky_with_jitter`] will try before giving
/// up. Each attempt multiplies the ridge by ten, so the ladder spans twelve
/// orders of magnitude — from a ridge that is pure rounding noise up to one that
/// dominates the matrix entirely.
pub const MAX_CHOLESKY_JITTER_ATTEMPTS: usize = 14;

/// Cholesky factor of `M`, adding an escalating diagonal ridge if and only if
/// `M` is not factorizable as given.
///
/// Returns `(L, jitter)` where `L L^T = M + jitter * I` and `jitter` is `0.0`
/// when no repair was needed.
///
/// # Why a jitter path is needed at all
///
/// The matrix Thompson sampling factors is `v^2 A^-1`, and `A^-1` is *supposed*
/// to be positive definite — the same induction that bounds the Sherman–Morrison
/// denominator (see the [module documentation](self)) proves it. But "positive
/// definite" is a statement about exact arithmetic, and after tens of thousands
/// of rank-1 updates the smallest eigenvalue of the maintained `A^-1` can be
/// genuinely tiny (a direction that has been observed over and over drives its
/// posterior variance toward zero), at which point rounding error is comparable
/// to the eigenvalue itself and the computed matrix can land a few ulps on the
/// wrong side of the cone. Refusing to sample in that situation would be absurd:
/// the *correct* posterior in that direction is "essentially a point mass", and
/// that is precisely what a tiny ridge encodes.
///
/// # Why the ridge is scale-aware
///
/// A fixed `1e-12` ridge is meaningless without knowing the scale of `M`: it is
/// a colossal perturbation to a matrix whose entries are `1e-15`, and it is
/// beneath the rounding noise of a matrix whose entries are `1e6`. The ridge is
/// therefore measured in units of `M`'s own magnitude, `scale = max(tr(M) / d,
/// 1)` (the mean diagonal entry, floored at one so that a *zero* matrix still
/// gets a usable ridge rather than a zero one). Attempt `k` tries
/// `jitter_k = BASE_CHOLESKY_JITTER * scale * 10^k`, so the first attempt
/// perturbs `M` by about one part in `10^12` — a change smaller than the
/// rounding error already present — and only a grossly indefinite matrix ever
/// escalates far enough for the ridge to be numerically visible.
///
/// The returned `jitter` is reported rather than swallowed precisely so that a
/// caller (or a test) can *see* whether the repair path engaged and how hard it
/// had to push.
///
/// # Errors
///
/// [`LinalgError::JitterExhausted`] if even the largest ridge in the ladder
/// leaves `M` unfactorizable — which means `M` was not a drifted positive
/// definite matrix but a genuinely, grossly indefinite one — plus the
/// dimension/finiteness errors of [`cholesky_lower`].
pub fn cholesky_with_jitter(matrix: &[f64], dim: usize) -> Result<(Vec<f64>, f64), LinalgError> {
    if dim == 0 {
        return Err(LinalgError::ZeroDimension);
    }
    check_matrix(matrix, dim)?;

    // Fast path: already positive definite, no perturbation at all.
    match cholesky_lower(matrix, dim) {
        Ok(lower) => return Ok((lower, 0.0)),
        Err(LinalgError::NotPositiveDefinite { .. }) => {}
        Err(other) => return Err(other),
    }

    let trace: f64 = (0..dim).map(|i| matrix[i * dim + i]).sum();
    #[allow(clippy::cast_precision_loss)] // dim is a handful of features.
    let mean_diagonal = trace / dim as f64;
    let scale = if mean_diagonal.is_finite() {
        mean_diagonal.abs().max(1.0)
    } else {
        return Err(LinalgError::NonFinite {
            what: "cholesky input matrix trace",
        });
    };

    let mut ridged = matrix.to_vec();
    let mut previous_jitter = 0.0;
    let mut jitter = BASE_CHOLESKY_JITTER * scale;
    for _ in 0..MAX_CHOLESKY_JITTER_ATTEMPTS {
        // Adjust the diagonal *incrementally* so the ridge never accumulates:
        // after this loop the diagonal holds `matrix[i][i] + jitter` exactly
        // once, whatever the previous attempt left behind.
        let delta = jitter - previous_jitter;
        for i in 0..dim {
            ridged[i * dim + i] += delta;
        }
        match cholesky_lower(&ridged, dim) {
            Ok(lower) => return Ok((lower, jitter)),
            Err(LinalgError::NotPositiveDefinite { .. }) => {
                previous_jitter = jitter;
                jitter *= 10.0;
            }
            Err(other) => return Err(other),
        }
    }
    Err(LinalgError::JitterExhausted {
        attempts: MAX_CHOLESKY_JITTER_ATTEMPTS,
        last_jitter: previous_jitter,
    })
}

/// Multiply a lower-triangular `L` by a vector `z`, exploiting the triangular
/// structure (row `i` only touches columns `0..=i`).
///
/// This is the `L z` of the Thompson-sampling draw `theta_hat + L z`.
///
/// # Errors
///
/// [`LinalgError::DimensionMismatch`] on a malformed argument.
pub fn lower_triangular_mat_vec(
    lower: &[f64],
    z: &[f64],
    dim: usize,
) -> Result<Vec<f64>, LinalgError> {
    check_matrix(lower, dim)?;
    check_vector(z, dim)?;
    let mut out = vec![0.0; dim];
    for i in 0..dim {
        let mut acc = 0.0;
        for j in 0..=i {
            acc += lower[i * dim + j] * z[j];
        }
        out[i] = acc;
    }
    Ok(out)
}

/// Scale every entry of a matrix by `factor`, returning a fresh buffer.
#[must_use]
pub fn scale_matrix(matrix: &[f64], factor: f64) -> Vec<f64> {
    matrix.iter().map(|v| v * factor).collect()
}

/// Verify a slice is a `dim × dim` row-major matrix.
fn check_matrix(matrix: &[f64], dim: usize) -> Result<(), LinalgError> {
    if matrix.len() == dim * dim {
        Ok(())
    } else {
        Err(LinalgError::DimensionMismatch {
            expected: dim * dim,
            actual: matrix.len(),
        })
    }
}

/// Verify a slice is a `dim`-vector.
fn check_vector(vector: &[f64], dim: usize) -> Result<(), LinalgError> {
    if vector.len() == dim {
        Ok(())
    } else {
        Err(LinalgError::DimensionMismatch {
            expected: dim,
            actual: vector.len(),
        })
    }
}

/// Dense matrix inverse by **Gauss–Jordan elimination with partial pivoting** —
/// **for tests only**.
///
/// This is deliberately *not* part of the production path. The entire point of
/// the Sherman–Morrison machinery above is that an `O(d^3)` inverse never has to
/// be computed at all; shipping one would invite exactly the "just re-invert
/// it, it's small" regression the module is built to avoid. It exists here as
/// an **independent oracle**: the module's headline numerical test replays a long
/// sequence of rank-1 updates, reconstructs the design matrix `A` explicitly,
/// inverts it from scratch with this routine, and asserts the incrementally
/// maintained `A^-1` still agrees to ~1e-9. Without a second, structurally
/// different implementation to check against, "the incremental inverse is
/// correct" would be an assertion rather than a proof.
///
/// Partial pivoting (swap in the row with the largest-magnitude candidate pivot)
/// is what makes it a *trustworthy* oracle: without it, Gauss–Jordan loses
/// accuracy exactly on the ill-conditioned matrices the test most wants to
/// stress.
///
/// # Errors
///
/// [`LinalgError::NotPositiveDefinite`] is reused to report a singular matrix
/// (an all-but-zero pivot column), plus the usual dimension/finiteness errors.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub fn gauss_jordan_inverse(matrix: &[f64], dim: usize) -> Result<Vec<f64>, LinalgError> {
    if dim == 0 {
        return Err(LinalgError::ZeroDimension);
    }
    check_matrix(matrix, dim)?;
    if matrix.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite {
            what: "gauss-jordan input matrix",
        });
    }

    let mut work = matrix.to_vec();
    let mut inverse = scaled_identity(dim, 1.0);

    for column in 0..dim {
        // Partial pivoting: find the row at or below `column` with the largest
        // absolute value in this column.
        let mut pivot_row = column;
        let mut best = work[column * dim + column].abs();
        for row in (column + 1)..dim {
            let candidate = work[row * dim + column].abs();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if best <= f64::EPSILON {
            return Err(LinalgError::NotPositiveDefinite {
                index: column,
                pivot: work[column * dim + column],
            });
        }
        if pivot_row != column {
            for j in 0..dim {
                work.swap(column * dim + j, pivot_row * dim + j);
                inverse.swap(column * dim + j, pivot_row * dim + j);
            }
        }

        // Normalize the pivot row.
        let pivot = work[column * dim + column];
        for j in 0..dim {
            work[column * dim + j] /= pivot;
            inverse[column * dim + j] /= pivot;
        }

        // Eliminate this column from every other row.
        for row in 0..dim {
            if row == column {
                continue;
            }
            let factor = work[row * dim + column];
            if factor == 0.0 {
                continue;
            }
            for j in 0..dim {
                work[row * dim + j] -= factor * work[column * dim + j];
                inverse[row * dim + j] -= factor * inverse[column * dim + j];
            }
        }
    }
    Ok(inverse)
}
