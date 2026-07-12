//! Pure-Rust dense linear-algebra primitives backing the ITQ hasher.
//!
//! Everything here operates on small, dense, row-major `Vec<Vec<f64>>`
//! matrices — ITQ works in the `k`-dimensional code space, where `k` (the
//! number of hash bits) is on the order of 16–128, so the classic `O(k^3)`
//! dense algorithms are entirely adequate and there is no dependency on any
//! external BLAS/LAPACK.
//!
//! The module provides four building blocks:
//!
//! 1. `jacobi_symmetric` — the cyclic-Jacobi eigenvalue algorithm extended to
//!    accumulate **eigenvectors** (the sibling `eigenscore` module's solver
//!    returns only eigenvalues and is not exported, so ITQ carries its own).
//!    Powers PCA (top-`k` eigenvectors of the `D x D` covariance) and the SVD
//!    below.
//! 2. `svd_square` — a `k x k` singular value decomposition built on the
//!    symmetric eigendecomposition of `M^T M`, with graceful handling of
//!    near-zero singular values via orthonormal basis completion.
//! 3. `orthogonal_procrustes` — the closed-form solver `R = U V^T` of the
//!    orthogonal Procrustes problem `min ‖B − Z R‖_F  s.t.  R^T R = I`.
//! 4. `random_orthogonal` — a deterministic, `rand`-free pseudo-random
//!    orthogonal matrix (FNV-1a → Box–Muller → modified Gram–Schmidt), used to
//!    initialize the alternating minimization.
//!
//! All arithmetic is performed in `f64`.
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::cmp::Ordering;

use super::types::ItqError;

/// Off-diagonal entries with absolute value at or below this are treated as
/// already zero (skipped) to avoid an unnecessary/degenerate Jacobi rotation.
const ROTATION_EPS: f64 = 1e-300;

/// Threshold below which a column norm is treated as collapsed during
/// Gram–Schmidt orthonormalization.
const GS_EPS: f64 = 1e-9;

/// The `(U, singular values, V)` triple returned by [`svd_square`], with `U`
/// and `V` in column-oriented row-major storage.
type SvdDecomposition = (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>);

// ── FNV-1a based pseudo-Gaussian stream ──────────────────────────────────────

/// FNV-1a 64-bit hash (offset basis `14695981039346656037`, prime
/// `1099511628211`) of the little-endian byte concatenation of `(seed, index)`.
fn fnv1a_u64(seed: u64, index: u64) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in seed.to_le_bytes().iter().chain(index.to_le_bytes().iter()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Map a 64-bit hash to an open-interval uniform `f64` in `(0, 1)`.
///
/// The open interval avoids exactly `0.0` (which would make `ln(u1)` diverge in
/// the Box–Muller transform below) and exactly `1.0`.
fn hash_to_unit_open(h: u64) -> f64 {
    let v = (h as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    v.clamp(1e-15, 1.0 - 1e-15)
}

/// The `k`-th (0-indexed) standard-normal pseudo-random sample, deterministic
/// in `(seed, k)`, produced via the Box–Muller transform.
fn box_muller_sample(seed: u64, k: u64) -> f64 {
    let pair = k / 2;
    let u1 = hash_to_unit_open(fnv1a_u64(seed, 2 * pair));
    let u2 = hash_to_unit_open(fnv1a_u64(seed, 2 * pair + 1));
    let radius = (-2.0 * u1.ln()).sqrt();
    if k.is_multiple_of(2) {
        radius * (2.0 * std::f64::consts::PI * u2).cos()
    } else {
        radius * (2.0 * std::f64::consts::PI * u2).sin()
    }
}

// ── small dense matrix helpers ───────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[must_use]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Product `a · b^T` of two `k x k` row-major matrices: result `[i][j]` is the
/// dot product of row `i` of `a` with row `j` of `b`.
#[must_use]
pub(crate) fn mat_mul_abt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row_a| b.iter().map(|row_b| dot(row_a, row_b)).collect())
        .collect()
}

/// Validate that `matrix` is square with finite entries, returning its size.
///
/// # Errors
///
/// Returns [`ItqError::Numerical`] if any row's length differs from the number
/// of rows, and [`ItqError::NonFinite`] if any entry is `NaN`/infinite.
fn ensure_square_finite(matrix: &[Vec<f64>]) -> Result<usize, ItqError> {
    let n = matrix.len();
    for row in matrix {
        if row.len() != n {
            return Err(ItqError::Numerical(format!(
                "expected a square matrix (size {n}), found a row of length {}",
                row.len()
            )));
        }
        for &value in row {
            if !value.is_finite() {
                return Err(ItqError::NonFinite);
            }
        }
    }
    Ok(n)
}

// ── symmetric eigensolver (eigenvalues + eigenvectors) ───────────────────────

/// Cyclic-Jacobi eigendecomposition of a dense symmetric matrix, returning both
/// the eigenvalues (descending) and the corresponding orthonormal eigenvectors.
///
/// The returned eigenvector matrix is column-oriented in row-major storage:
/// `eigenvectors[i][j]` is the `i`-th component of the `j`-th eigenvector, and
/// column `j` pairs with `eigenvalues[j]`. For a symmetric input `A` the result
/// satisfies `A · v_j = eigenvalues[j] · v_j` and `V^T V = I`.
///
/// The rotation math mirrors the sibling `eigenscore` module's solver (solving
/// for `t = tan(theta)` via the smaller-magnitude root for stability), extended
/// here to accumulate the product of plane rotations into the eigenvector
/// matrix. `1x1` inputs are returned directly.
///
/// # Errors
///
/// Returns [`ItqError::Numerical`] if `matrix` is not square and
/// [`ItqError::NonFinite`] if any entry is `NaN`/infinite.
pub(crate) fn jacobi_symmetric(
    matrix: &[Vec<f64>],
    max_sweeps: usize,
    tol: f64,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), ItqError> {
    let n = ensure_square_finite(matrix)?;

    let mut vecs = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        vecs[i][i] = 1.0;
    }

    if n == 0 {
        return Ok((Vec::new(), vecs));
    }
    if n == 1 {
        return Ok((vec![matrix[0][0]], vecs));
    }

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

                // Accumulate the plane rotation into the eigenvector matrix by
                // mixing its columns p and q (V <- V · G).
                for i in 0..n {
                    let v_ip = vecs[i][p];
                    let v_iq = vecs[i][q];
                    vecs[i][p] = c.mul_add(v_ip, -(s * v_iq));
                    vecs[i][q] = s.mul_add(v_ip, c * v_iq);
                }
            }
        }
    }

    let raw_values: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    // Sort eigenvalues descending, permuting the eigenvector columns to match.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        raw_values[y]
            .partial_cmp(&raw_values[x])
            .unwrap_or(Ordering::Equal)
    });

    let eigenvalues: Vec<f64> = order.iter().map(|&idx| raw_values[idx]).collect();
    let mut eigenvectors = vec![vec![0.0_f64; n]; n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            eigenvectors[row][new_col] = vecs[row][old_col];
        }
    }

    Ok((eigenvalues, eigenvectors))
}

// ── square SVD ───────────────────────────────────────────────────────────────

/// Singular value decomposition of a `k x k` matrix `M`, returning
/// `(u, sigma, v)` such that `M ≈ U · diag(sigma) · V^T` with `U`, `V`
/// orthonormal (`k x k`, column-oriented row-major storage, matching
/// [`jacobi_symmetric`]) and `sigma` the singular values in descending order.
///
/// Uses the **one-sided (Hestenes) Jacobi** algorithm operating directly on
/// `M`: repeated plane rotations applied to pairs of columns drive them towards
/// mutual orthogonality (right rotations accumulate into `V`). At convergence
/// the columns of the rotated `A = M V` are orthogonal; each singular value is
/// the norm of a column and each left singular vector is that column normalized.
/// Working on `M` directly — rather than eigendecomposing `M^T M`, which squares
/// the condition number and destroys accuracy for the small singular values —
/// is essential here: ITQ's PCA projection produces columns whose variances
/// span many orders of magnitude, and only an accurate SVD keeps the Procrustes
/// rotation a true minimizer (so the alternating objective stays monotone).
///
/// Columns that collapse to (numerically) zero norm have no left singular
/// vector; those `U` columns are filled by **completing** the accepted set to a
/// full orthonormal basis via modified Gram–Schmidt against the standard basis.
/// Because those directions carry no energy, any orthonormal completion yields
/// the same reconstruction and keeps `U` genuinely orthogonal.
///
/// # Errors
///
/// Returns [`ItqError::Numerical`] if `m` is not square, [`ItqError::NonFinite`]
/// if any entry is `NaN`/infinite, and [`ItqError::Numerical`] in the
/// (dimensionally impossible) event that the orthonormal completion cannot be
/// finished.
pub(crate) fn svd_square(
    m: &[Vec<f64>],
    max_sweeps: usize,
    tol: f64,
) -> Result<SvdDecomposition, ItqError> {
    let k = ensure_square_finite(m)?;
    if k == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    // `a` holds the column-rotated copy of M (a = M V); `v` accumulates the
    // right rotations (starting from the identity).
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut v = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        v[i][i] = 1.0;
    }

    // Relative threshold below which a column pair is treated as already
    // orthogonal and its rotation skipped.
    let rel_tol = tol.max(0.0).max(1e-15);

    if k >= 2 {
        for _sweep in 0..max_sweeps {
            let mut rotated = false;
            for p in 0..k {
                for q in (p + 1)..k {
                    let mut alpha = 0.0_f64; // ‖col p‖²
                    let mut beta = 0.0_f64; // ‖col q‖²
                    let mut gamma = 0.0_f64; // <col p, col q>
                    for i in 0..k {
                        let a_ip = a[i][p];
                        let a_iq = a[i][q];
                        alpha += a_ip * a_ip;
                        beta += a_iq * a_iq;
                        gamma += a_ip * a_iq;
                    }
                    let denom = (alpha * beta).sqrt();
                    if gamma.abs() <= ROTATION_EPS || denom <= ROTATION_EPS {
                        continue;
                    }
                    if gamma.abs() <= rel_tol * denom {
                        continue;
                    }

                    // Jacobi angle diagonalizing [[alpha, gamma], [gamma, beta]].
                    let zeta = (beta - alpha) / (2.0 * gamma);
                    let t = if zeta == 0.0 {
                        1.0
                    } else {
                        zeta.signum() / (zeta.abs() + zeta.mul_add(zeta, 1.0).sqrt())
                    };
                    let c = 1.0 / t.mul_add(t, 1.0).sqrt();
                    let s = c * t;

                    for i in 0..k {
                        let a_ip = a[i][p];
                        let a_iq = a[i][q];
                        a[i][p] = c.mul_add(a_ip, -(s * a_iq));
                        a[i][q] = s.mul_add(a_ip, c * a_iq);
                        let v_ip = v[i][p];
                        let v_iq = v[i][q];
                        v[i][p] = c.mul_add(v_ip, -(s * v_iq));
                        v[i][q] = s.mul_add(v_ip, c * v_iq);
                    }
                    rotated = true;
                }
            }
            if !rotated {
                break;
            }
        }
    }

    // Singular values are the norms of the (now orthogonal) columns of `a`.
    let raw_sigma: Vec<f64> = (0..k)
        .map(|j| (0..k).map(|i| a[i][j] * a[i][j]).sum::<f64>().sqrt())
        .collect();

    // Sort columns by descending singular value.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&x, &y| {
        raw_sigma[y]
            .partial_cmp(&raw_sigma[x])
            .unwrap_or(Ordering::Equal)
    });

    let sigma: Vec<f64> = order.iter().map(|&j| raw_sigma[j]).collect();
    let s_max = sigma.first().copied().unwrap_or(0.0);
    // Only genuinely null directions (no left singular vector) are completed;
    // real small singular values are accepted so `U Σ V^T` reconstructs `M`.
    let sv_tol = 1e-12_f64.mul_add(s_max, 1e-300);

    let mut u = vec![vec![0.0_f64; k]; k];
    let mut v_sorted = vec![vec![0.0_f64; k]; k];
    // Accepted, mutually orthonormal left singular vectors (for completion).
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut deferred: Vec<usize> = Vec::new();

    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..k {
            v_sorted[row][new_col] = v[row][old_col];
        }
        if sigma[new_col] > sv_tol {
            let inv = 1.0 / sigma[new_col];
            let col: Vec<f64> = (0..k).map(|row| a[row][old_col] * inv).collect();
            for row in 0..k {
                u[row][new_col] = col[row];
            }
            basis.push(col);
        } else {
            deferred.push(new_col);
        }
    }

    if !deferred.is_empty() {
        complete_left_singular_vectors(&mut u, &mut basis, &deferred, k)?;
    }

    Ok((u, sigma, v_sorted))
}

/// Fill the columns of `u` listed in `deferred` (those with a ~zero singular
/// value, hence no natural left singular vector) with vectors completing the
/// already-orthonormal `basis` to a full orthonormal set.
///
/// Candidates are drawn from the standard basis and orthonormalized against
/// `basis` with modified Gram–Schmidt run **twice** ("twice is enough", Kahan):
/// a single pass loses orthogonality catastrophically when a candidate is
/// nearly in the span of `basis` (its residual is tiny and normalizing it
/// amplifies the rounding error), which is exactly the situation once `basis`
/// is nearly full for a low-rank `M`.
///
/// # Errors
///
/// Returns [`ItqError::Numerical`] in the (dimensionally impossible) event that
/// the standard basis is exhausted before every deferred column is filled.
fn complete_left_singular_vectors(
    u: &mut [Vec<f64>],
    basis: &mut Vec<Vec<f64>>,
    deferred: &[usize],
    k: usize,
) -> Result<(), ItqError> {
    let mut candidate = 0usize;
    for &j in deferred {
        let mut chosen: Option<Vec<f64>> = None;
        while candidate < k {
            let mut e = vec![0.0_f64; k];
            e[candidate] = 1.0;
            candidate += 1;
            for _pass in 0..2 {
                for b in basis.iter() {
                    let d = dot(&e, b);
                    for (value, &bv) in e.iter_mut().zip(b.iter()) {
                        *value -= d * bv;
                    }
                }
            }
            let norm = e.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > GS_EPS {
                for value in &mut e {
                    *value /= norm;
                }
                chosen = Some(e);
                break;
            }
        }
        let column = chosen.ok_or_else(|| {
            ItqError::Numerical(
                "failed to complete an orthonormal basis for a zero singular value".into(),
            )
        })?;
        for row in 0..k {
            u[row][j] = column[row];
        }
        basis.push(column);
    }
    Ok(())
}

// ── orthogonal Procrustes ────────────────────────────────────────────────────

/// Solve the orthogonal Procrustes problem for a `k x k` matrix `M = Z^T B`:
/// return the orthogonal `R` that minimizes `‖B − Z R‖_F` subject to
/// `R^T R = I`.
///
/// With the SVD `M = U Σ V^T`, the maximizer of `trace(R^T M)` over orthogonal
/// `R` is `R = U V^T` (Schönemann 1966). Note that this is exactly optimal even
/// when `M` is rank-deficient: the arbitrary orthonormal completion chosen for
/// zero-singular-value directions of `U` does not change the objective, since
/// those directions are weighted by a zero singular value.
///
/// # Errors
///
/// Propagates any error from [`svd_square`].
pub(crate) fn orthogonal_procrustes(
    m: &[Vec<f64>],
    max_sweeps: usize,
    tol: f64,
) -> Result<Vec<Vec<f64>>, ItqError> {
    let (u, _sigma, v) = svd_square(m, max_sweeps, tol)?;
    Ok(mat_mul_abt(&u, &v))
}

// ── deterministic random orthogonal matrix ───────────────────────────────────

/// Deterministically construct a `k x k` orthogonal matrix from `seed`, with no
/// external random-number crate.
///
/// A `k x k` matrix of pseudo-Gaussian entries (FNV-1a → Box–Muller) is
/// orthonormalized column-by-column with modified Gram–Schmidt, yielding an
/// orthogonal matrix `R` with `R^T R = R R^T = I`, returned in row-major
/// storage (`R[i][j]`).
///
/// # Errors
///
/// - [`ItqError::InvalidConfig`] if `k == 0`.
/// - [`ItqError::Numerical`] in the astronomically unlikely event that a
///   Gram–Schmidt column collapses to (numerically) zero norm for the given
///   `(seed, k)` pair; callers should pick a different seed.
pub(crate) fn random_orthogonal(k: usize, seed: u64) -> Result<Vec<Vec<f64>>, ItqError> {
    if k == 0 {
        return Err(ItqError::InvalidConfig(
            "rotation dimension must be greater than zero".into(),
        ));
    }

    // Column-major pseudo-Gaussian matrix: cols[c][r] is entry (r, c).
    let mut cols: Vec<Vec<f64>> = (0..k)
        .map(|c| {
            (0..k)
                .map(|r| box_muller_sample(seed, (c * k + r) as u64))
                .collect()
        })
        .collect();

    for col_idx in 0..k {
        for prev in 0..col_idx {
            let q_prev = cols[prev].clone();
            let projection = dot(&cols[col_idx], &q_prev);
            for (value, &basis_value) in cols[col_idx].iter_mut().zip(q_prev.iter()) {
                *value -= projection * basis_value;
            }
        }
        let norm = cols[col_idx].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < GS_EPS {
            return Err(ItqError::Numerical(format!(
                "degenerate random basis for seed {seed} and dim {k} \
                 (column {col_idx} collapsed during Gram-Schmidt); pick a different seed"
            )));
        }
        for value in &mut cols[col_idx] {
            *value /= norm;
        }
    }

    // Transpose to row-major R where R[i][j] = cols[j][i].
    let mut rows = vec![vec![0.0_f64; k]; k];
    for (j, col) in cols.iter().enumerate() {
        for (i, &value) in col.iter().enumerate() {
            rows[i][j] = value;
        }
    }
    Ok(rows)
}
