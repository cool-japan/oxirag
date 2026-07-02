//! Cyclic Jacobi eigenvalue algorithm for small, dense, symmetric matrices.
//!
//! The Jacobi eigenvalue algorithm iteratively drives a real symmetric matrix
//! `A` towards a diagonal matrix by applying a sequence of plane (Givens)
//! rotations, each of which annihilates one off-diagonal entry. Because a
//! later rotation can partially "undo" an earlier one, the algorithm sweeps
//! repeatedly through all off-diagonal positions until the accumulated
//! off-diagonal energy falls below a tolerance. This implementation uses the
//! **cyclic** variant: each sweep visits every pair `(p, q)` with `p < q` in a
//! fixed row-major order (as opposed to "classical" Jacobi, which always
//! targets the single largest off-diagonal entry — more expensive to locate
//! per step, but not needed for the small `K x K` matrices `EigenScore` works
//! with, where `K` is on the order of 5-20 samples).
//!
//! # Rotation math
//!
//! For a chosen off-diagonal pair `(p, q)` with `a_pq != 0`, we seek a
//! rotation angle `theta` (in the `(p, q)` plane) such that applying the
//! orthogonal similarity transform `A' = J^T A J` zeroes `a'_pq`, where `J` is
//! the identity except for `J_pp = J_qq = c = cos(theta)` and
//! `J_pq = -J_qp = s = sin(theta)`.
//!
//! Requiring `a'_pq = 0` yields the classical identity
//!
//! ```text
//! cot(2*theta) = (a_qq - a_pp) / (2 * a_pq)
//! ```
//!
//! Rather than compute `theta` directly (which is numerically delicate near
//! `a_pq ~ 0`), we solve for `t = tan(theta)` via the equivalent quadratic
//! `t^2 + 2*t*cot(2*theta) - 1 = 0`, take `phi = cot(2*theta)`, and pick the
//! root of smaller magnitude for stability (Golub & Van Loan, *Matrix
//! Computations*, section on the Jacobi method; also Numerical Recipes'
//! `jacobi` routine):
//!
//! ```text
//! t = sign(phi) / (|phi| + sqrt(phi^2 + 1))
//! c = 1 / sqrt(t^2 + 1)
//! s = t * c
//! ```
//!
//! With `c` and `s` in hand, the affected entries update as:
//!
//! ```text
//! a'_pp = a_pp - t * a_pq
//! a'_qq = a_qq + t * a_pq
//! a'_pq = a'_qp = 0
//! a'_ip = c * a_ip - s * a_iq        (for i != p, q; mirrored into a'_pi)
//! a'_iq = s * a_ip + c * a_iq        (for i != p, q; mirrored into a'_qi)
//! ```
//!
//! all other entries are unchanged. Repeating this for every pair each sweep,
//! and repeating sweeps, converges quadratically once the off-diagonal norm
//! is small; the diagonal of the limit matrix holds the eigenvalues.
//!
//! `1x1` and `2x2` matrices are handled analytically (no rotations needed):
//! for `1x1`, the sole diagonal entry *is* the eigenvalue; for `2x2`
//! `[[a, b], [b, d]]`, the eigenvalues are the closed form
//! `(a+d)/2 +/- sqrt(((a-d)/2)^2 + b^2)`.

use std::cmp::Ordering;

use super::types::EigenScoreError;

/// Default maximum number of cyclic-Jacobi sweeps used by
/// [`symmetric_eigenvalues`], matching [`super::types::EigenScoreConfig`]'s
/// default.
pub(crate) const DEFAULT_JACOBI_MAX_SWEEPS: usize = 100;

/// Default off-diagonal convergence tolerance used by
/// [`symmetric_eigenvalues`], matching [`super::types::EigenScoreConfig`]'s
/// default.
pub(crate) const DEFAULT_JACOBI_TOL: f64 = 1e-10;

/// Off-diagonal entries with absolute value at or below this are treated as
/// already zero (skipped) to avoid an unnecessary/degenerate rotation.
const ROTATION_EPS: f64 = 1e-300;

/// Relative tolerance used when checking that a supplied matrix is symmetric.
const SYMMETRY_TOL: f64 = 1e-6;

/// Validate that `matrix` is square with finite entries and (approximately)
/// symmetric, returning its dimension `n`.
fn validate_square_symmetric(matrix: &[Vec<f64>]) -> Result<usize, EigenScoreError> {
    let n = matrix.len();
    for row in matrix {
        if row.len() != n {
            return Err(EigenScoreError::DimensionMismatch {
                expected: n,
                got: row.len(),
            });
        }
        for &value in row {
            if !value.is_finite() {
                return Err(EigenScoreError::NonFinite);
            }
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in (i + 1)..n {
            let a_ij = matrix[i][j];
            let a_ji = matrix[j][i];
            let scale = 1.0 + a_ij.abs().max(a_ji.abs());
            if (a_ij - a_ji).abs() > SYMMETRY_TOL * scale {
                return Err(EigenScoreError::InvalidConfig(format!(
                    "matrix is not symmetric at ({i}, {j}): {a_ij} vs {a_ji}"
                )));
            }
        }
    }
    Ok(n)
}

/// Sort eigenvalues ascending, treating any (impossible after the finiteness
/// check above) incomparable pair as equal.
fn sort_ascending(values: &mut [f64]) {
    values.sort_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
}

/// Compute the eigenvalues of a small dense symmetric matrix via the cyclic
/// Jacobi algorithm, using explicit `max_sweeps` and `tol` parameters.
///
/// This is the configurable core used internally by
/// [`EigenScoreDetector`](super::detector::EigenScoreDetector); the public
/// [`symmetric_eigenvalues`] wraps it with the library defaults.
///
/// # Errors
///
/// Returns [`EigenScoreError::DimensionMismatch`] if `matrix` is not square,
/// [`EigenScoreError::NonFinite`] if any entry is `NaN`/infinite, and
/// [`EigenScoreError::InvalidConfig`] if `matrix` is not (approximately)
/// symmetric.
#[allow(clippy::similar_names, clippy::many_single_char_names)]
pub(crate) fn jacobi_eigenvalues(
    matrix: &[Vec<f64>],
    max_sweeps: usize,
    tol: f64,
) -> Result<Vec<f64>, EigenScoreError> {
    let n = validate_square_symmetric(matrix)?;

    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![matrix[0][0]]);
    }
    if n == 2 {
        let a = matrix[0][0];
        let b = matrix[0][1];
        let d = matrix[1][1];
        let mid = 0.5 * (a + d);
        let half_diff = 0.5 * (a - d);
        let radius = half_diff.mul_add(half_diff, b * b).sqrt();
        let mut eigenvalues = vec![mid - radius, mid + radius];
        sort_ascending(&mut eigenvalues);
        return Ok(eigenvalues);
    }

    // General n >= 3: cyclic Jacobi sweeps over a mutable working copy.
    // `max_sweeps == 0` intentionally performs zero rotations (the caller
    // gets the raw, un-diagonalized entries back); this low-level helper
    // trusts its caller's budget rather than silently overriding it.
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let effective_tol = tol.max(0.0);

    for _sweep in 0..max_sweeps {
        let mut off_norm_sq = 0.0_f64;
        for (i, row) in a.iter().enumerate() {
            for &value in &row[(i + 1)..] {
                off_norm_sq += 2.0 * value * value;
            }
        }
        if off_norm_sq.sqrt() < effective_tol {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = a[p][q];
                if a_pq.abs() <= ROTATION_EPS {
                    continue;
                }
                let a_pp = a[p][p];
                let a_qq = a[q][q];
                let phi = (a_qq - a_pp) / (2.0 * a_pq);
                let t = if phi >= 0.0 {
                    1.0 / (phi + phi.mul_add(phi, 1.0).sqrt())
                } else {
                    1.0 / (phi - phi.mul_add(phi, 1.0).sqrt())
                };
                let c = 1.0 / t.mul_add(t, 1.0).sqrt();
                let s = t * c;

                a[p][p] = a_pp - t * a_pq;
                a[q][q] = a_qq + t * a_pq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                #[allow(clippy::needless_range_loop)]
                for i in 0..n {
                    if i != p && i != q {
                        let a_ip = a[i][p];
                        let a_iq = a[i][q];
                        let new_ip = c.mul_add(a_ip, -(s * a_iq));
                        let new_iq = s.mul_add(a_ip, c * a_iq);
                        a[i][p] = new_ip;
                        a[p][i] = new_ip;
                        a[i][q] = new_iq;
                        a[q][i] = new_iq;
                    }
                }
            }
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    sort_ascending(&mut eigenvalues);
    Ok(eigenvalues)
}

/// Compute the eigenvalues of a small dense symmetric matrix via the cyclic
/// Jacobi algorithm (see the [module docs](self) for the rotation math),
/// using the library's default sweep budget (`100`) and convergence
/// tolerance (`1e-10`) — the same defaults as
/// [`EigenScoreConfig`](super::types::EigenScoreConfig)'s
/// `jacobi_max_sweeps` and `jacobi_tol` fields.
///
/// `matrix` must be square; each row is compared against the others for
/// dimension consistency. The returned eigenvalues are ascending. Computation
/// is carried out entirely in `f64` for numerical stability regardless of the
/// precision of the caller's data.
///
/// # Errors
///
/// Returns [`EigenScoreError::DimensionMismatch`] if any row's length differs
/// from the matrix dimension, [`EigenScoreError::NonFinite`] if any entry is
/// `NaN` or infinite, and [`EigenScoreError::InvalidConfig`] if `matrix` is
/// not (approximately) symmetric.
pub fn symmetric_eigenvalues(matrix: &[Vec<f64>]) -> Result<Vec<f64>, EigenScoreError> {
    jacobi_eigenvalues(matrix, DEFAULT_JACOBI_MAX_SWEEPS, DEFAULT_JACOBI_TOL)
}
