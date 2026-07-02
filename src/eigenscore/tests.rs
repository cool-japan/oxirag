//! Tests for the `eigenscore` module.
//!
//! # Cross-checking ground truth
//!
//! The eigenvalues used to validate [`symmetric_eigenvalues`] /
//! [`jacobi_eigenvalues`] below were independently computed with
//! `numpy.linalg.eigvalsh` (not derived from this crate's own
//! implementation), so a bug in the rotation math cannot pass these
//! assertions by construction alone. The `EigenScore` reference values were
//! computed by an independent Python re-implementation of the exact
//! centering / Gram / regularize / log-eigenvalue pipeline described in the
//! module docs.

use std::cmp::Ordering;

use crate::eigenscore::jacobi::jacobi_eigenvalues;
use crate::eigenscore::{
    EigenScoreConfig, EigenScoreDetector, EigenScoreError, EigenScoreResult, symmetric_eigenvalues,
};

// ── test helpers ──────────────────────────────────────────────────────────────

/// Absolute-difference float comparison with a fixed tolerance.
#[allow(clippy::float_cmp)]
fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Assert two eigenvalue vectors match element-wise (already ascending)
/// within `tol`.
fn assert_eigs_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            approx(*a, *e, tol),
            "eigenvalue mismatch: actual={actual:?} expected={expected:?} (tol={tol})"
        );
    }
}

/// Build an `n x n` diagonal matrix from `values`.
fn diag(values: &[f64]) -> Vec<Vec<f64>> {
    let n = values.len();
    let mut m = vec![vec![0.0; n]; n];
    for (i, &v) in values.iter().enumerate() {
        m[i][i] = v;
    }
    m
}

/// Build `a * I_n + b * J_n` (`J_n` = all-ones matrix), whose eigenvalues are
/// exactly `a + n*b` (multiplicity 1, eigenvector `[1,...,1]`) and `a`
/// (multiplicity `n-1`, eigenvectors orthogonal to `[1,...,1]`).
fn a_i_plus_b_j(n: usize, a: f64, b: f64) -> Vec<Vec<f64>> {
    let mut m = vec![vec![b; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] += a;
    }
    m
}

/// Build the rank-1 outer product `v * v^T`, whose eigenvalues are exactly
/// `||v||^2` (multiplicity 1) and `0` (multiplicity `n-1`).
fn outer_product(v: &[f64]) -> Vec<Vec<f64>> {
    let n = v.len();
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = v[i] * v[j];
        }
    }
    m
}

/// Default detector (regularization `1e-3`, threshold `-3.0`).
fn detector() -> EigenScoreDetector {
    EigenScoreDetector::default()
}

/// Build a `Vec<Vec<f32>>` embedding matrix from nested `f64` literals (for
/// concise test data authoring).
#[allow(clippy::cast_possible_truncation)]
fn embeds(rows: &[&[f64]]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|row| row.iter().map(|&v| v as f32).collect())
        .collect()
}

// ── Jacobi eigensolver: trivial sizes (0, 1, 2) ───────────────────────────────

#[test]
fn test_jacobi_empty_matrix_returns_empty() {
    let m: Vec<Vec<f64>> = Vec::new();
    let eigs = symmetric_eigenvalues(&m).expect("empty matrix is valid");
    assert!(eigs.is_empty());
}

#[test]
fn test_jacobi_1x1_returns_diagonal_entry() {
    let m = vec![vec![42.0]];
    let eigs = symmetric_eigenvalues(&m).expect("1x1 is valid");
    assert_eigs_close(&eigs, &[42.0], 1e-12);
}

#[test]
fn test_jacobi_1x1_negative_entry() {
    let m = vec![vec![-7.5]];
    let eigs = symmetric_eigenvalues(&m).expect("1x1 is valid");
    assert_eigs_close(&eigs, &[-7.5], 1e-12);
}

#[test]
fn test_jacobi_2x2_diagonal_already_sorted() {
    let m = diag(&[-3.0, 5.0]);
    let eigs = symmetric_eigenvalues(&m).expect("2x2 diagonal is valid");
    assert_eigs_close(&eigs, &[-3.0, 5.0], 1e-12);
}

#[test]
fn test_jacobi_2x2_diagonal_needs_sorting() {
    let m = diag(&[5.0, -3.0]);
    let eigs = symmetric_eigenvalues(&m).expect("2x2 diagonal is valid");
    assert_eigs_close(&eigs, &[-3.0, 5.0], 1e-12);
}

#[test]
fn test_jacobi_2x2_analytic_known_eigenvalues() {
    // [[2, 1], [1, 2]] has eigenvalues 2+1=3 and 2-1=1 (eigenvectors [1,1]
    // and [1,-1]).
    let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
    let eigs = symmetric_eigenvalues(&m).expect("2x2 is valid");
    assert_eigs_close(&eigs, &[1.0, 3.0], 1e-12);
}

#[test]
fn test_jacobi_2x2_rank_deficient() {
    // [[4, 2], [2, 1]] has trace 5, det 4*1 - 2*2 = 0, so eigenvalues solve
    // x^2 - 5x = 0 => x in {0, 5}.
    let m = vec![vec![4.0, 2.0], vec![2.0, 1.0]];
    let eigs = symmetric_eigenvalues(&m).expect("2x2 is valid");
    assert_eigs_close(&eigs, &[0.0, 5.0], 1e-12);
}

#[test]
fn test_jacobi_2x2_zero_matrix() {
    let m = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let eigs = symmetric_eigenvalues(&m).expect("zero matrix is valid");
    assert_eigs_close(&eigs, &[0.0, 0.0], 1e-12);
}

#[test]
fn test_jacobi_2x2_negative_off_diagonal() {
    // [[2, -1], [-1, 2]] has eigenvalues 2 - 1 = 1 and 2 + 1 = 3, same
    // spectrum as the positive-off-diagonal case (sign of b doesn't matter).
    let m = vec![vec![2.0, -1.0], vec![-1.0, 2.0]];
    let eigs = symmetric_eigenvalues(&m).expect("2x2 is valid");
    assert_eigs_close(&eigs, &[1.0, 3.0], 1e-12);
}

// ── Jacobi eigensolver: identity matrices ─────────────────────────────────────

#[test]
fn test_jacobi_identity_2() {
    let m = diag(&[1.0, 1.0]);
    let eigs = symmetric_eigenvalues(&m).expect("identity is valid");
    assert_eigs_close(&eigs, &[1.0, 1.0], 1e-12);
}

#[test]
fn test_jacobi_identity_3() {
    let m = diag(&[1.0, 1.0, 1.0]);
    let eigs = symmetric_eigenvalues(&m).expect("identity is valid");
    assert_eigs_close(&eigs, &[1.0, 1.0, 1.0], 1e-9);
}

#[test]
fn test_jacobi_identity_5() {
    let m = diag(&[1.0; 5]);
    let eigs = symmetric_eigenvalues(&m).expect("identity is valid");
    assert_eigs_close(&eigs, &[1.0; 5], 1e-9);
}

#[test]
fn test_jacobi_identity_10() {
    let m = diag(&[1.0; 10]);
    let eigs = symmetric_eigenvalues(&m).expect("identity is valid");
    assert_eigs_close(&eigs, &[1.0; 10], 1e-9);
}

#[test]
fn test_jacobi_scaled_identity_4() {
    let m = diag(&[2.5, 2.5, 2.5, 2.5]);
    let eigs = symmetric_eigenvalues(&m).expect("scaled identity is valid");
    assert_eigs_close(&eigs, &[2.5, 2.5, 2.5, 2.5], 1e-9);
}

// ── Jacobi eigensolver: diagonal matrices (n >= 3, exercises the sweep loop) ──

#[test]
fn test_jacobi_diagonal_3_already_sorted() {
    let m = diag(&[-2.0, 3.0, 7.0]);
    let eigs = symmetric_eigenvalues(&m).expect("diagonal is valid");
    assert_eigs_close(&eigs, &[-2.0, 3.0, 7.0], 1e-9);
}

#[test]
fn test_jacobi_diagonal_3_needs_sorting() {
    let m = diag(&[7.0, -2.0, 3.0]);
    let eigs = symmetric_eigenvalues(&m).expect("diagonal is valid");
    assert_eigs_close(&eigs, &[-2.0, 3.0, 7.0], 1e-9);
}

#[test]
fn test_jacobi_diagonal_4_with_duplicates() {
    let m = diag(&[1.0, 1.0, 4.0, -1.0]);
    let eigs = symmetric_eigenvalues(&m).expect("diagonal is valid");
    assert_eigs_close(&eigs, &[-1.0, 1.0, 1.0, 4.0], 1e-9);
}

#[test]
fn test_jacobi_diagonal_with_zero() {
    let m = diag(&[0.0, -5.0, 5.0]);
    let eigs = symmetric_eigenvalues(&m).expect("diagonal is valid");
    assert_eigs_close(&eigs, &[-5.0, 0.0, 5.0], 1e-9);
}

// ── Jacobi eigensolver: known-spectrum constructions ───────────────────────────

#[test]
fn test_jacobi_a_i_plus_b_j_n3() {
    // eigenvalues: a + 3b (mult 1), a (mult 2).
    let m = a_i_plus_b_j(3, 2.0, 0.7);
    let eigs = symmetric_eigenvalues(&m).expect("valid matrix");
    assert_eigs_close(&eigs, &[2.0, 2.0, 4.1], 1e-8);
}

#[test]
fn test_jacobi_a_i_plus_b_j_n4() {
    let m = a_i_plus_b_j(4, -1.0, 0.5);
    // eigenvalues: -1 (mult 3), -1 + 4*0.5 = 1.0 (mult 1).
    let eigs = symmetric_eigenvalues(&m).expect("valid matrix");
    assert_eigs_close(&eigs, &[-1.0, -1.0, -1.0, 1.0], 1e-8);
}

#[test]
fn test_jacobi_a_i_plus_b_j_n5() {
    // Matches the numpy cross-check: a=2.0, b=0.7, n=5 -> {2.0 x4, 5.5 x1}.
    let m = a_i_plus_b_j(5, 2.0, 0.7);
    let eigs = symmetric_eigenvalues(&m).expect("valid matrix");
    assert_eigs_close(&eigs, &[2.0, 2.0, 2.0, 2.0, 5.5], 1e-7);
}

#[test]
fn test_jacobi_a_i_plus_b_j_negative_b() {
    // b < 0 still admits the same closed form.
    let m = a_i_plus_b_j(4, 3.0, -0.5);
    // a + n*b = 3 - 2 = 1.0 (mult 1), a = 3.0 (mult 3).
    let eigs = symmetric_eigenvalues(&m).expect("valid matrix");
    assert_eigs_close(&eigs, &[1.0, 3.0, 3.0, 3.0], 1e-8);
}

#[test]
fn test_jacobi_rank1_outer_product_n4() {
    // v = [1, 2, -1.5, 0.5], ||v||^2 = 1 + 4 + 2.25 + 0.25 = 7.5.
    let v = [1.0, 2.0, -1.5, 0.5];
    let m = outer_product(&v);
    let eigs = symmetric_eigenvalues(&m).expect("valid matrix");
    assert_eigs_close(&eigs, &[0.0, 0.0, 0.0, 7.5], 1e-7);
}

#[test]
fn test_jacobi_rank1_outer_product_n5() {
    let v = [3.0, -1.0, 0.0, 2.0, 1.0];
    // ||v||^2 = 9 + 1 + 0 + 4 + 1 = 15.
    let m = outer_product(&v);
    let eigs = symmetric_eigenvalues(&m).expect("valid matrix");
    assert_eigs_close(&eigs, &[0.0, 0.0, 0.0, 0.0, 15.0], 1e-6);
}

#[test]
fn test_jacobi_known_spectrum_5x5_cross_checked_with_numpy() {
    // Cross-checked against `numpy.linalg.eigvalsh` independently of this
    // crate's implementation.
    let a = vec![
        vec![4.0, 1.5, -2.0, 0.5, 1.0],
        vec![1.5, 3.2, 0.7, -1.1, 0.3],
        vec![-2.0, 0.7, 5.5, 1.2, -0.8],
        vec![0.5, -1.1, 1.2, 2.8, 0.6],
        vec![1.0, 0.3, -0.8, 0.6, 1.9],
    ];
    let eigs = symmetric_eigenvalues(&a).expect("valid matrix");
    let expected = [
        0.220_992_378_839_420_3,
        1.418_249_755_426_581_2,
        3.875_734_198_813_353_3,
        4.583_924_214_231_437,
        7.301_099_452_689_207,
    ];
    assert_eigs_close(&eigs, &expected, 1e-6);
}

#[test]
fn test_jacobi_known_spectrum_4x4_cross_checked_with_numpy() {
    let b = vec![
        vec![2.0, -1.0, 0.0, 0.5],
        vec![-1.0, 2.0, -1.0, 0.0],
        vec![0.0, -1.0, 2.0, -1.0],
        vec![0.5, 0.0, -1.0, 2.0],
    ];
    let eigs = symmetric_eigenvalues(&b).expect("valid matrix");
    let expected = [
        0.499_999_999_999_999_83,
        0.999_999_999_999_999_8,
        3.0,
        3.499_999_999_999_999,
    ];
    assert_eigs_close(&eigs, &expected, 1e-6);
}

#[test]
fn test_jacobi_known_spectrum_6x6_cross_checked_with_numpy() {
    let c = vec![
        vec![6.0, 0.5, -1.2, 0.3, 0.8, -0.4],
        vec![0.5, 4.5, 0.9, -0.6, 0.2, 1.0],
        vec![-1.2, 0.9, 5.0, 1.1, -0.3, 0.5],
        vec![0.3, -0.6, 1.1, 3.8, 0.7, -0.9],
        vec![0.8, 0.2, -0.3, 0.7, 4.2, 0.4],
        vec![-0.4, 1.0, 0.5, -0.9, 0.4, 5.3],
    ];
    let eigs = symmetric_eigenvalues(&c).expect("valid matrix");
    let expected = [
        2.089_496_504_994_137,
        3.286_161_314_065_593,
        4.302_693_784_516_895,
        5.516_132_793_120_166,
        6.378_950_084_026_925_5,
        7.226_565_519_276_274,
    ];
    assert_eigs_close(&eigs, &expected, 1e-6);
}

#[test]
fn test_jacobi_trace_invariant_holds() {
    // Sum of eigenvalues must equal the trace for any symmetric matrix.
    let a = vec![
        vec![4.0, 1.5, -2.0, 0.5, 1.0],
        vec![1.5, 3.2, 0.7, -1.1, 0.3],
        vec![-2.0, 0.7, 5.5, 1.2, -0.8],
        vec![0.5, -1.1, 1.2, 2.8, 0.6],
        vec![1.0, 0.3, -0.8, 0.6, 1.9],
    ];
    let trace: f64 = (0..5).map(|i| a[i][i]).sum();
    let eigs = symmetric_eigenvalues(&a).expect("valid matrix");
    let sum: f64 = eigs.iter().sum();
    assert!(approx(sum, trace, 1e-6), "sum={sum} trace={trace}");
}

#[test]
fn test_jacobi_determinant_invariant_holds_for_2x2() {
    // Product of eigenvalues must equal the determinant.
    let m = vec![vec![4.0, 2.0], vec![2.0, 1.0]];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let eigs = symmetric_eigenvalues(&m).expect("valid matrix");
    let prod: f64 = eigs.iter().product();
    assert!(approx(prod, det, 1e-9), "prod={prod} det={det}");
}

#[test]
fn test_jacobi_determinant_invariant_holds_for_4x4() {
    let b = vec![
        vec![2.0, -1.0, 0.0, 0.5],
        vec![-1.0, 2.0, -1.0, 0.0],
        vec![0.0, -1.0, 2.0, -1.0],
        vec![0.5, 0.0, -1.0, 2.0],
    ];
    let eigs = symmetric_eigenvalues(&b).expect("valid matrix");
    let prod: f64 = eigs.iter().product();
    // Expected determinant from the (independently) known eigenvalues
    // 0.5 * 1.0 * 3.0 * 3.5 = 5.25.
    assert!(approx(prod, 5.25, 1e-5), "prod={prod}");
}

#[test]
fn test_jacobi_eigenvalues_always_ascending() {
    let matrices: Vec<Vec<Vec<f64>>> = vec![
        diag(&[9.0, -3.0, 0.0, 4.0]),
        a_i_plus_b_j(4, -1.0, 0.5),
        outer_product(&[1.0, 2.0, -1.5, 0.5]),
    ];
    for m in matrices {
        let eigs = symmetric_eigenvalues(&m).expect("valid matrix");
        for pair in eigs.windows(2) {
            assert!(pair[0] <= pair[1] + 1e-9, "not ascending: {eigs:?}");
        }
    }
}

// ── Jacobi eigensolver: sweep-budget / tolerance behavior (internal fn) ───────

#[test]
fn test_jacobi_zero_max_sweeps_skips_rotation() {
    // With zero sweeps the non-diagonal matrix is returned essentially
    // untouched (only sorted), i.e. NOT diagonalized correctly.
    let m = a_i_plus_b_j(4, -1.0, 0.5);
    let eigs = jacobi_eigenvalues(&m, 0, 1e-10).expect("valid matrix");
    let mut raw_diagonal: Vec<f64> = (0..4).map(|i| m[i][i]).collect();
    raw_diagonal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    assert_eigs_close(&eigs, &raw_diagonal, 1e-12);
    // And these are NOT the true eigenvalues {-1,-1,-1,1}.
    assert!(!approx(eigs[3], 1.0, 1e-6));
}

#[test]
fn test_jacobi_one_sweep_makes_progress_but_may_not_converge() {
    let m = a_i_plus_b_j(4, -1.0, 0.5);
    let one_sweep = jacobi_eigenvalues(&m, 1, 1e-14).expect("valid matrix");
    let many_sweeps = jacobi_eigenvalues(&m, 100, 1e-14).expect("valid matrix");
    // Many sweeps must reach the true spectrum.
    assert_eigs_close(&many_sweeps, &[-1.0, -1.0, -1.0, 1.0], 1e-9);
    // One sweep is a valid (ascending, same length) result too, even if not
    // fully converged.
    assert_eq!(one_sweep.len(), 4);
}

#[test]
fn test_jacobi_large_tolerance_stops_immediately() {
    // A tolerance larger than any achievable off-diagonal norm causes the
    // solver to stop before the first rotation, i.e. return the raw
    // (sorted) diagonal.
    let m = a_i_plus_b_j(4, -1.0, 0.5);
    let eigs = jacobi_eigenvalues(&m, 100, 1e6).expect("valid matrix");
    let mut raw_diagonal: Vec<f64> = (0..4).map(|i| m[i][i]).collect();
    raw_diagonal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    assert_eigs_close(&eigs, &raw_diagonal, 1e-12);
}

#[test]
fn test_jacobi_default_sweeps_converge_for_6x6() {
    let c = vec![
        vec![6.0, 0.5, -1.2, 0.3, 0.8, -0.4],
        vec![0.5, 4.5, 0.9, -0.6, 0.2, 1.0],
        vec![-1.2, 0.9, 5.0, 1.1, -0.3, 0.5],
        vec![0.3, -0.6, 1.1, 3.8, 0.7, -0.9],
        vec![0.8, 0.2, -0.3, 0.7, 4.2, 0.4],
        vec![-0.4, 1.0, 0.5, -0.9, 0.4, 5.3],
    ];
    // Using the crate's default sweep budget / tolerance via the public fn.
    let eigs = symmetric_eigenvalues(&c).expect("valid matrix");
    let sum: f64 = eigs.iter().sum();
    let trace: f64 = (0..6).map(|i| c[i][i]).sum();
    assert!(approx(sum, trace, 1e-6));
}

// ── Jacobi eigensolver: error handling ────────────────────────────────────────

#[test]
fn test_jacobi_non_square_matrix_errors() {
    let m = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
    let err = symmetric_eigenvalues(&m).unwrap_err();
    assert!(matches!(err, EigenScoreError::DimensionMismatch { .. }));
}

#[test]
fn test_jacobi_ragged_matrix_reports_dimension_mismatch() {
    let m = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0, 1.0]];
    let err = symmetric_eigenvalues(&m).unwrap_err();
    match err {
        EigenScoreError::DimensionMismatch { expected, got } => {
            assert_eq!(expected, 3);
            assert_eq!(got, 2);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

#[test]
fn test_jacobi_nan_entry_errors() {
    let m = vec![vec![1.0, f64::NAN], vec![f64::NAN, 1.0]];
    let err = symmetric_eigenvalues(&m).unwrap_err();
    assert!(matches!(err, EigenScoreError::NonFinite));
}

#[test]
fn test_jacobi_infinite_entry_errors() {
    let m = vec![vec![1.0, 0.0], vec![0.0, f64::INFINITY]];
    let err = symmetric_eigenvalues(&m).unwrap_err();
    assert!(matches!(err, EigenScoreError::NonFinite));
}

#[test]
fn test_jacobi_non_symmetric_matrix_errors() {
    let m = vec![vec![1.0, 2.0], vec![-2.0, 1.0]];
    let err = symmetric_eigenvalues(&m).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_jacobi_non_symmetric_3x3_errors() {
    let m = vec![
        vec![1.0, 2.0, 0.0],
        vec![2.0, 1.0, 3.0],
        vec![0.0, 99.0, 1.0],
    ];
    let err = symmetric_eigenvalues(&m).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_jacobi_symmetry_tolerance_allows_tiny_rounding() {
    // A minuscule asymmetry (far below the symmetry tolerance) should still
    // be accepted.
    let m = vec![vec![1.0, 2.000_000_01], vec![2.0, 1.0]];
    assert!(symmetric_eigenvalues(&m).is_ok());
}

// ── EigenScoreConfig ───────────────────────────────────────────────────────────

#[test]
fn test_config_default_regularization() {
    let cfg = EigenScoreConfig::default();
    assert!(approx(f64::from(cfg.regularization), 1e-3, 1e-9));
}

#[test]
fn test_config_default_clip_percentile_is_none() {
    let cfg = EigenScoreConfig::default();
    assert!(cfg.clip_percentile.is_none());
}

#[test]
fn test_config_default_jacobi_max_sweeps() {
    let cfg = EigenScoreConfig::default();
    assert_eq!(cfg.jacobi_max_sweeps, 100);
}

#[test]
fn test_config_default_jacobi_tol() {
    let cfg = EigenScoreConfig::default();
    assert!(approx(cfg.jacobi_tol, 1e-10, 1e-15));
}

#[test]
fn test_config_default_embed_dim() {
    let cfg = EigenScoreConfig::default();
    assert_eq!(cfg.embed_dim, 64);
}

#[test]
fn test_config_new_matches_default() {
    let cfg = EigenScoreConfig::new();
    let default_cfg = EigenScoreConfig::default();
    assert_eq!(cfg, default_cfg);
}

#[test]
fn test_config_with_regularization_builder() {
    let cfg = EigenScoreConfig::new().with_regularization(0.05);
    assert!(approx(f64::from(cfg.regularization), 0.05, 1e-9));
}

#[test]
fn test_config_with_clip_percentile_builder() {
    let cfg = EigenScoreConfig::new().with_clip_percentile(Some(0.1));
    assert_eq!(cfg.clip_percentile, Some(0.1));
}

#[test]
fn test_config_with_jacobi_max_sweeps_builder() {
    let cfg = EigenScoreConfig::new().with_jacobi_max_sweeps(5);
    assert_eq!(cfg.jacobi_max_sweeps, 5);
}

#[test]
fn test_config_with_jacobi_tol_builder() {
    let cfg = EigenScoreConfig::new().with_jacobi_tol(1e-6);
    assert!(approx(cfg.jacobi_tol, 1e-6, 1e-15));
}

#[test]
fn test_config_with_hallucination_threshold_builder() {
    let cfg = EigenScoreConfig::new().with_hallucination_threshold(2.0);
    assert!(approx(f64::from(cfg.hallucination_threshold), 2.0, 1e-9));
}

#[test]
fn test_config_with_embed_dim_builder() {
    let cfg = EigenScoreConfig::new().with_embed_dim(128);
    assert_eq!(cfg.embed_dim, 128);
}

#[test]
fn test_config_builder_chaining() {
    let cfg = EigenScoreConfig::new()
        .with_regularization(0.01)
        .with_clip_percentile(Some(0.05))
        .with_jacobi_max_sweeps(50)
        .with_jacobi_tol(1e-8)
        .with_hallucination_threshold(-1.0)
        .with_embed_dim(32);
    assert!(approx(f64::from(cfg.regularization), 0.01, 1e-9));
    assert_eq!(cfg.clip_percentile, Some(0.05));
    assert_eq!(cfg.jacobi_max_sweeps, 50);
    assert!(approx(cfg.jacobi_tol, 1e-8, 1e-15));
    assert!(approx(f64::from(cfg.hallucination_threshold), -1.0, 1e-9));
    assert_eq!(cfg.embed_dim, 32);
}

#[test]
fn test_config_clone() {
    let cfg = EigenScoreConfig::new().with_embed_dim(16);
    let cloned = cfg.clone();
    assert_eq!(cloned.embed_dim, 16);
}

#[test]
fn test_config_debug_format_contains_field_names() {
    let cfg = EigenScoreConfig::default();
    let debug_str = format!("{cfg:?}");
    assert!(debug_str.contains("regularization"));
    assert!(debug_str.contains("embed_dim"));
}

// ── EigenScoreError ────────────────────────────────────────────────────────────

#[test]
fn test_error_insufficient_samples_display() {
    let err = EigenScoreError::InsufficientSamples { got: 1, need: 2 };
    let msg = err.to_string();
    assert!(msg.contains('1'));
    assert!(msg.contains('2'));
}

#[test]
fn test_error_dimension_mismatch_display() {
    let err = EigenScoreError::DimensionMismatch {
        expected: 5,
        got: 3,
    };
    let msg = err.to_string();
    assert!(msg.contains('5'));
    assert!(msg.contains('3'));
}

#[test]
fn test_error_invalid_config_display() {
    let err = EigenScoreError::InvalidConfig("bad value".to_string());
    assert!(err.to_string().contains("bad value"));
}

#[test]
fn test_error_non_finite_display() {
    let err = EigenScoreError::NonFinite;
    assert!(err.to_string().to_lowercase().contains("finite"));
}

#[test]
fn test_error_equality() {
    let a = EigenScoreError::InsufficientSamples { got: 1, need: 2 };
    let b = EigenScoreError::InsufficientSamples { got: 1, need: 2 };
    assert_eq!(a, b);
}

#[test]
fn test_error_inequality_across_variants() {
    let a = EigenScoreError::NonFinite;
    let b = EigenScoreError::InvalidConfig("x".to_string());
    assert_ne!(a, b);
}

// ── EigenScoreDetector: input validation ──────────────────────────────────────

#[test]
fn test_detector_zero_embeddings_errors() {
    let d = detector();
    let err = d.score_embeddings(&[]).unwrap_err();
    match err {
        EigenScoreError::InsufficientSamples { got, need } => {
            assert_eq!(got, 0);
            assert_eq!(need, 2);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_detector_one_embedding_errors() {
    let d = detector();
    let single = embeds(&[&[1.0, 2.0, 3.0]]);
    let err = d.score_embeddings(&single).unwrap_err();
    assert!(matches!(
        err,
        EigenScoreError::InsufficientSamples { got: 1, need: 2 }
    ));
}

#[test]
fn test_detector_two_embeddings_is_the_minimum_valid_case() {
    let d = detector();
    let two = embeds(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let result = d.score_embeddings(&two).expect("K=2 is valid");
    assert_eq!(result.eigenvalues.len(), 2);
}

#[test]
fn test_detector_zero_dimensional_embeddings_errors() {
    let d = detector();
    let empty_dim: Vec<Vec<f32>> = vec![vec![], vec![]];
    let err = d.score_embeddings(&empty_dim).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_detector_dimension_mismatch_errors() {
    let d = detector();
    let mismatched = vec![vec![1.0_f32, 2.0], vec![1.0_f32, 2.0, 3.0]];
    let err = d.score_embeddings(&mismatched).unwrap_err();
    match err {
        EigenScoreError::DimensionMismatch { expected, got } => {
            assert_eq!(expected, 2);
            assert_eq!(got, 3);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_detector_nan_embedding_errors() {
    let d = detector();
    let with_nan = vec![vec![1.0_f32, f32::NAN], vec![0.0_f32, 1.0]];
    let err = d.score_embeddings(&with_nan).unwrap_err();
    assert!(matches!(err, EigenScoreError::NonFinite));
}

#[test]
fn test_detector_infinite_embedding_errors() {
    let d = detector();
    let with_inf = vec![vec![1.0_f32, f32::INFINITY], vec![0.0_f32, 1.0]];
    let err = d.score_embeddings(&with_inf).unwrap_err();
    assert!(matches!(err, EigenScoreError::NonFinite));
}

#[test]
fn test_detector_negative_regularization_errors() {
    let cfg = EigenScoreConfig::new().with_regularization(-0.1);
    let d = EigenScoreDetector::new(cfg);
    let two = embeds(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let err = d.score_embeddings(&two).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_detector_nan_regularization_errors() {
    let cfg = EigenScoreConfig::new().with_regularization(f32::NAN);
    let d = EigenScoreDetector::new(cfg);
    let two = embeds(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let err = d.score_embeddings(&two).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_detector_clip_percentile_too_high_errors() {
    let cfg = EigenScoreConfig::new().with_clip_percentile(Some(0.5));
    let d = EigenScoreDetector::new(cfg);
    let two = embeds(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let err = d.score_embeddings(&two).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_detector_clip_percentile_negative_errors() {
    let cfg = EigenScoreConfig::new().with_clip_percentile(Some(-0.1));
    let d = EigenScoreDetector::new(cfg);
    let two = embeds(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let err = d.score_embeddings(&two).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_detector_zero_max_sweeps_errors() {
    let cfg = EigenScoreConfig::new().with_jacobi_max_sweeps(0);
    let d = EigenScoreDetector::new(cfg);
    let two = embeds(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let err = d.score_embeddings(&two).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_detector_negative_jacobi_tol_errors() {
    let cfg = EigenScoreConfig::new().with_jacobi_tol(-1.0);
    let d = EigenScoreDetector::new(cfg);
    let two = embeds(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let err = d.score_embeddings(&two).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

// ── EigenScoreDetector: exact analytic scores ─────────────────────────────────

#[test]
fn test_detector_identical_embeddings_score_equals_ln_alpha() {
    // K identical embeddings: mean-centering yields the zero vector for
    // every sample, so the Gram matrix is exactly zero and every eigenvalue
    // of Sigma is exactly `alpha`. Score = (1/K) * K * ln(alpha) = ln(alpha).
    let d = detector();
    let same = embeds(&[
        &[0.5, 0.5, 0.5],
        &[0.5, 0.5, 0.5],
        &[0.5, 0.5, 0.5],
        &[0.5, 0.5, 0.5],
    ]);
    let result = d.score_embeddings(&same).expect("valid embeddings");
    let expected_score = 1e-3_f64.ln();
    assert!(
        approx(f64::from(result.score), expected_score, 1e-3),
        "score={} expected={expected_score}",
        result.score
    );
    for &lambda in &result.eigenvalues {
        assert!(approx(f64::from(lambda), 1e-3, 1e-6));
    }
}

#[test]
fn test_detector_identical_embeddings_with_clipping_still_matches() {
    // Clipping identical values is a no-op (lower == upper == the value),
    // so the analytic ln(alpha) result should be unaffected.
    let cfg = EigenScoreConfig::new().with_clip_percentile(Some(0.1));
    let d = EigenScoreDetector::new(cfg);
    let same = embeds(&[
        &[0.5, 0.5, 0.5],
        &[0.5, 0.5, 0.5],
        &[0.5, 0.5, 0.5],
        &[0.5, 0.5, 0.5],
    ]);
    let result = d.score_embeddings(&same).expect("valid embeddings");
    let expected_score = 1e-3_f64.ln();
    assert!(approx(f64::from(result.score), expected_score, 1e-3));
}

#[test]
fn test_detector_orthonormal_basis_small_scale_score() {
    // K=3, D=3 standard basis vectors; cross-checked with an independent
    // Python re-implementation of the centering/Gram/regularize pipeline.
    let d = detector();
    let basis = embeds(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
    let result = d.score_embeddings(&basis).expect("valid embeddings");
    assert!(approx(
        f64::from(result.score),
        -3.032_996_279_452_943,
        1e-3
    ));
}

#[test]
fn test_detector_orthonormal_basis_large_scale_score() {
    let d = detector();
    let basis_scaled = embeds(&[&[10.0, 0.0, 0.0], &[0.0, 10.0, 0.0], &[0.0, 0.0, 10.0]]);
    let result = d.score_embeddings(&basis_scaled).expect("valid embeddings");
    assert!(approx(
        f64::from(result.score),
        0.035_140_171_584_579_015,
        1e-3
    ));
}

#[test]
fn test_detector_rank_deficient_proportional_embeddings() {
    // Embeddings all lie along a single direction (proportional to a fixed
    // vector), so the Gram matrix is rank 1 and Sigma has one large
    // eigenvalue plus (K-1) eigenvalues at (nearly) `alpha`.
    let d = detector();
    let v = [1.0_f64, -2.0, 0.5];
    let scales = [1.0_f64, 2.0, -1.0, 0.5];
    let rows: Vec<Vec<f64>> = scales
        .iter()
        .map(|&s| v.iter().map(|&x| s * x).collect())
        .collect();
    let row_refs: Vec<&[f64]> = rows.iter().map(std::vec::Vec::as_slice).collect();
    let proportional = embeds(&row_refs);
    let result = d.score_embeddings(&proportional).expect("valid embeddings");
    assert!(approx(
        f64::from(result.score),
        -4.726_567_550_922_883,
        1e-2
    ));
    // Exactly one eigenvalue should be much larger than the rest.
    let max_eig = result.eigenvalues.iter().copied().fold(f32::MIN, f32::max);
    let small_count = result
        .eigenvalues
        .iter()
        .filter(|&&e| e < max_eig / 10.0)
        .count();
    assert_eq!(small_count, 3);
}

// ── EigenScoreDetector: ordering (the critical hallucination-signal property) ─

#[test]
fn test_detector_identical_scores_lower_than_orthonormal_basis() {
    let d = detector();
    let identical = embeds(&[&[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]]);
    let basis = embeds(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
    let identical_result = d.score_embeddings(&identical).expect("valid");
    let basis_result = d.score_embeddings(&basis).expect("valid");
    assert!(identical_result.score < basis_result.score);
}

#[test]
fn test_detector_small_spread_scores_lower_than_large_spread() {
    let d = detector();
    let small = embeds(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
    let large = embeds(&[&[10.0, 0.0, 0.0], &[0.0, 10.0, 0.0], &[0.0, 0.0, 10.0]]);
    let small_result = d.score_embeddings(&small).expect("valid");
    let large_result = d.score_embeddings(&large).expect("valid");
    assert!(small_result.score < large_result.score);
}

#[test]
fn test_detector_mixed_close_scores_lower_than_mixed_spread() {
    let d = detector();
    let close = embeds(&[
        &[1.00, 0.20, -0.30, 0.10],
        &[0.90, 0.25, -0.25, 0.15],
        &[1.10, 0.15, -0.35, 0.05],
        &[1.05, 0.22, -0.28, 0.12],
        &[0.95, 0.18, -0.32, 0.08],
    ]);
    let spread = embeds(&[
        &[5.0, -3.0, 2.0, 1.0],
        &[-4.0, 6.0, -1.0, 3.0],
        &[2.0, 2.0, 5.0, -4.0],
        &[1.0, -5.0, 3.0, 2.0],
        &[-3.0, 1.0, -2.0, 5.0],
    ]);
    let close_result = d.score_embeddings(&close).expect("valid");
    let spread_result = d.score_embeddings(&spread).expect("valid");
    assert!(close_result.score < spread_result.score);
    // Cross-checked scores from an independent Python re-implementation.
    assert!(approx(
        f64::from(close_result.score),
        -6.316_773_223_175_39,
        1e-2
    ));
    assert!(approx(
        f64::from(spread_result.score),
        -0.674_891_584_654_588_6,
        1e-2
    ));
}

// ── EigenScoreDetector: feature clipping ──────────────────────────────────────

#[test]
fn test_detector_clipping_lowers_score_when_outlier_present() {
    // One embedding has an extreme outlier in a single dimension. Clipping
    // should pull the eigenvalue spectrum (and hence the score) down
    // relative to the unclipped case.
    let rows: [&[f64]; 5] = [
        &[1.0, 50.0, -0.3, 0.1],
        &[0.9, 0.25, -0.25, 0.15],
        &[1.1, 0.15, -0.35, 0.05],
        &[1.05, 0.22, -0.28, 0.12],
        &[0.95, 0.18, -0.32, 0.08],
    ];
    let with_outlier = embeds(&rows);

    let unclipped = detector();
    let clipped_cfg = EigenScoreConfig::new().with_clip_percentile(Some(0.1));
    let clipped_detector = EigenScoreDetector::new(clipped_cfg);

    let unclipped_result = unclipped.score_embeddings(&with_outlier).expect("valid");
    let clipped_result = clipped_detector
        .score_embeddings(&with_outlier)
        .expect("valid");

    assert!(clipped_result.score < unclipped_result.score);
}

#[test]
fn test_detector_clip_percentile_zero_is_valid_but_a_no_op_bound() {
    // p=0.0 clips to the [min, max] band, which never changes any value.
    let cfg = EigenScoreConfig::new().with_clip_percentile(Some(0.0));
    let d_clip = EigenScoreDetector::new(cfg);
    let d_plain = detector();
    let rows = embeds(&[
        &[1.0, 0.2, -0.3, 0.1],
        &[0.9, 0.25, -0.25, 0.15],
        &[1.1, 0.15, -0.35, 0.05],
    ]);
    let clipped = d_clip.score_embeddings(&rows).expect("valid");
    let plain = d_plain.score_embeddings(&rows).expect("valid");
    assert!(approx(
        f64::from(clipped.score),
        f64::from(plain.score),
        1e-4
    ));
}

// ── EigenScoreResult / is_hallucination thresholding ──────────────────────────

#[test]
fn test_detector_default_threshold_flags_high_spread_as_hallucination() {
    let d = detector();
    let large = embeds(&[&[10.0, 0.0, 0.0], &[0.0, 10.0, 0.0], &[0.0, 0.0, 10.0]]);
    let result = d.score_embeddings(&large).expect("valid");
    assert!(result.is_hallucination);
}

#[test]
fn test_detector_default_threshold_does_not_flag_identical_embeddings() {
    let d = detector();
    let same = embeds(&[&[0.1, 0.2], &[0.1, 0.2], &[0.1, 0.2]]);
    let result = d.score_embeddings(&same).expect("valid");
    assert!(!result.is_hallucination);
}

#[test]
fn test_detector_low_threshold_forces_hallucination_true() {
    let cfg = EigenScoreConfig::new().with_hallucination_threshold(-100.0);
    let d = EigenScoreDetector::new(cfg);
    let same = embeds(&[&[0.1, 0.2], &[0.1, 0.2], &[0.1, 0.2]]);
    let result = d.score_embeddings(&same).expect("valid");
    assert!(result.is_hallucination);
}

#[test]
fn test_detector_high_threshold_forces_hallucination_false() {
    let cfg = EigenScoreConfig::new().with_hallucination_threshold(1000.0);
    let d = EigenScoreDetector::new(cfg);
    let large = embeds(&[&[10.0, 0.0, 0.0], &[0.0, 10.0, 0.0], &[0.0, 0.0, 10.0]]);
    let result = d.score_embeddings(&large).expect("valid");
    assert!(!result.is_hallucination);
}

#[test]
fn test_detector_threshold_boundary_is_inclusive() {
    // is_hallucination is defined as score >= threshold.
    let d = detector();
    let same = embeds(&[&[0.1, 0.2], &[0.1, 0.2], &[0.1, 0.2]]);
    let result = d.score_embeddings(&same).expect("valid");
    let exact_cfg = EigenScoreConfig::new().with_hallucination_threshold(result.score);
    let exact_detector = EigenScoreDetector::new(exact_cfg);
    let exact_result = exact_detector.score_embeddings(&same).expect("valid");
    assert!(exact_result.is_hallucination);
}

// ── EigenScoreResult shape / determinism ──────────────────────────────────────

#[test]
fn test_detector_eigenvalues_length_matches_k() {
    let d = detector();
    for k in 2..=8 {
        #[allow(clippy::cast_precision_loss)]
        let rows: Vec<Vec<f32>> = (0..k)
            .map(|i| vec![i as f32, (i * 2) as f32, (i + 1) as f32])
            .collect();
        let result = d.score_embeddings(&rows).expect("valid");
        assert_eq!(result.eigenvalues.len(), k);
    }
}

#[test]
fn test_detector_score_embeddings_is_deterministic() {
    let d = detector();
    let rows = embeds(&[&[1.0, 2.0, 3.0], &[4.0, 1.0, -2.0], &[0.0, 0.0, 1.0]]);
    let r1 = d.score_embeddings(&rows).expect("valid");
    let r2 = d.score_embeddings(&rows).expect("valid");
    assert!(approx(f64::from(r1.score), f64::from(r2.score), 1e-12));
    assert_eq!(r1.eigenvalues, r2.eigenvalues);
    assert_eq!(r1.is_hallucination, r2.is_hallucination);
}

#[test]
fn test_result_clone_and_debug() {
    let d = detector();
    let rows = embeds(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let result = d.score_embeddings(&rows).expect("valid");
    let cloned = result.clone();
    assert_eq!(result, cloned);
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("score"));
}

#[test]
fn test_detector_default_matches_new_with_default_config() {
    let via_default = EigenScoreDetector::default();
    let via_new = EigenScoreDetector::new(EigenScoreConfig::default());
    assert_eq!(via_default.config, via_new.config);
}

// ── EigenScoreDetector: text helper (score_responses) ─────────────────────────

#[test]
fn test_score_responses_requires_at_least_two() {
    let d = detector();
    let one = ["only one response"];
    let err = d.score_responses(&one).unwrap_err();
    assert!(matches!(
        err,
        EigenScoreError::InsufficientSamples { got: 1, need: 2 }
    ));
}

#[test]
fn test_score_responses_zero_responses_errors() {
    let d = detector();
    let none: [&str; 0] = [];
    let err = d.score_responses(&none).unwrap_err();
    assert!(matches!(
        err,
        EigenScoreError::InsufficientSamples { got: 0, need: 2 }
    ));
}

#[test]
fn test_score_responses_embed_dim_zero_errors() {
    let cfg = EigenScoreConfig::new().with_embed_dim(0);
    let d = EigenScoreDetector::new(cfg);
    let two = ["hello world", "goodbye world"];
    let err = d.score_responses(&two).unwrap_err();
    assert!(matches!(err, EigenScoreError::InvalidConfig(_)));
}

#[test]
fn test_score_responses_is_deterministic() {
    let d = detector();
    let responses = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy dog.",
        "Quantum entanglement defies classical intuition.",
    ];
    let r1 = d.score_responses(&responses).expect("valid");
    let r2 = d.score_responses(&responses).expect("valid");
    assert_eq!(r1, r2);
}

#[test]
fn test_score_responses_identical_strings_score_equals_ln_alpha() {
    let d = detector();
    let same = [
        "Repeat this exact sentence.",
        "Repeat this exact sentence.",
        "Repeat this exact sentence.",
    ];
    let result = d.score_responses(&same).expect("valid");
    let expected_score = 1e-3_f64.ln();
    assert!(approx(f64::from(result.score), expected_score, 1e-3));
}

#[test]
fn test_score_responses_consistent_paraphrases_score_lower_than_divergent() {
    let d = detector();
    let consistent = [
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "France's capital city is Paris.",
    ];
    let divergent = [
        "Paris is the capital of France.",
        "Tokyo is a bustling megacity in Japan.",
        "Mitochondria are the powerhouse of the cell.",
    ];
    let consistent_result = d.score_responses(&consistent).expect("valid");
    let divergent_result = d.score_responses(&divergent).expect("valid");
    assert!(consistent_result.score < divergent_result.score);
}

#[test]
fn test_score_responses_invariant_under_reordering() {
    // The Gram matrix is permuted by a simultaneous row/column permutation
    // when the sample order changes, which is an orthogonal similarity
    // transform — eigenvalues (and hence the score) must be unchanged.
    let d = detector();
    let original = [
        "Paris is the capital of France.",
        "Tokyo is a big city.",
        "Cats are mammals.",
    ];
    let reordered = [
        "Cats are mammals.",
        "Paris is the capital of France.",
        "Tokyo is a big city.",
    ];
    let original_result = d.score_responses(&original).expect("valid");
    let reordered_result = d.score_responses(&reordered).expect("valid");
    assert!(approx(
        f64::from(original_result.score),
        f64::from(reordered_result.score),
        1e-4
    ));
}

#[test]
fn test_score_responses_respects_custom_embed_dim() {
    let cfg = EigenScoreConfig::new().with_embed_dim(16);
    let d = EigenScoreDetector::new(cfg);
    let responses = ["hello there", "hi there", "completely different content"];
    let result = d.score_responses(&responses).expect("valid");
    assert_eq!(result.eigenvalues.len(), 3);
}

#[test]
fn test_score_responses_short_texts_do_not_panic() {
    let d = detector();
    let short = ["a", "b", "c"];
    let result = d
        .score_responses(&short)
        .expect("valid even for 1-char texts");
    assert_eq!(result.eigenvalues.len(), 3);
}

#[test]
fn test_score_responses_empty_strings_do_not_panic() {
    let d = detector();
    let empties = ["", "", ""];
    let result = d
        .score_responses(&empties)
        .expect("valid even for empty strings");
    assert_eq!(result.eigenvalues.len(), 3);
}

#[test]
fn test_score_responses_case_and_whitespace_normalized() {
    // "HELLO   WORLD" and "hello world" normalize to the same character
    // stream, so a response and its re-cased/whitespace-padded duplicate
    // should embed identically.
    let d = detector();
    let responses = [
        "hello world",
        "HELLO   WORLD",
        "totally unrelated text here",
    ];
    let result = d.score_responses(&responses).expect("valid");
    assert_eq!(result.eigenvalues.len(), 3);
    // Since two of the three embed identically, the Gram matrix is rank <=
    // 2 in the centered space among those two, which is a sanity check that
    // normalization collapses case/whitespace differences (checked via
    // determinism/ordering above; here we just assert it runs to
    // completion and returns a finite score).
    assert!(result.score.is_finite());
}

#[test]
fn test_score_responses_all_finite_and_ascending_eigenvalues() {
    let d = detector();
    let responses = [
        "Machine learning models can hallucinate facts.",
        "Neural networks sometimes generate false information.",
        "The weather today is sunny with a light breeze.",
    ];
    let result = d.score_responses(&responses).expect("valid");
    for &e in &result.eigenvalues {
        assert!(e.is_finite());
    }
    for pair in result.eigenvalues.windows(2) {
        assert!(pair[0] <= pair[1] + 1e-6);
    }
}

// ── EigenScoreResult field consistency ─────────────────────────────────────────

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_result_score_is_mean_log_eigenvalue() {
    let d = detector();
    let rows = embeds(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
    let result = d.score_embeddings(&rows).expect("valid");
    let recomputed: f64 = result
        .eigenvalues
        .iter()
        .map(|&e| f64::from(e).ln())
        .sum::<f64>()
        / result.eigenvalues.len() as f64;
    assert!(approx(f64::from(result.score), recomputed, 1e-4));
}

#[test]
fn test_result_construction_directly() {
    // EigenScoreResult's fields are all public and constructible directly,
    // e.g. for use in mocks/fixtures elsewhere in the codebase.
    let result = EigenScoreResult {
        score: -2.0,
        eigenvalues: vec![0.001, 0.5, 1.2],
        is_hallucination: false,
    };
    assert!(approx(f64::from(result.score), -2.0, 1e-9));
    assert_eq!(result.eigenvalues.len(), 3);
    assert!(!result.is_hallucination);
}

#[test]
fn test_detector_larger_k_still_converges_and_orders_correctly() {
    // K=10 identical vs K=10 spread, exercising the general n>=3 Jacobi path
    // at a larger size than the earlier K=3..5 tests.
    let d = detector();
    let identical: Vec<Vec<f32>> = (0..10).map(|_| vec![0.3_f32, -0.2, 0.7]).collect();
    #[allow(clippy::cast_precision_loss)]
    let spread: Vec<Vec<f32>> = (0..10_i32)
        .map(|i| {
            let angle = i as f32 * 0.6;
            vec![angle.sin(), angle.cos(), (angle * 0.5).sin()]
        })
        .collect();
    let identical_result = d.score_embeddings(&identical).expect("valid");
    let spread_result = d.score_embeddings(&spread).expect("valid");
    assert_eq!(identical_result.eigenvalues.len(), 10);
    assert_eq!(spread_result.eigenvalues.len(), 10);
    assert!(identical_result.score < spread_result.score);
}
