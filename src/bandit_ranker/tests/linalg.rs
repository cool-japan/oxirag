#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::manual_midpoint,
    clippy::doc_markdown
)]
//! Tests for contextual-bandit online ranking.
//! Tests for the dense linear-algebra basics and for the test-only Gauss–Jordan
//! oracle that the Sherman–Morrison correctness check is measured against.

use super::max_abs_diff;
use crate::bandit_ranker::linalg::{
    LinalgError, dot, gauss_jordan_inverse, lower_triangular_mat_vec, mat_vec,
    quadratic_form_nonnegative, scale_matrix, scaled_identity, symmetrize,
};

// ═════════════════════════════════════════════════════════════════════════════
// linalg: basics
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn scaled_identity_builds_the_right_matrix() {
    assert_eq!(
        scaled_identity(3, 2.0),
        vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
    );
    assert_eq!(scaled_identity(1, 5.0), vec![5.0]);
}

#[test]
fn dot_and_mat_vec_agree_with_hand_computation() {
    assert_eq!(
        dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).expect("same length"),
        32.0
    );
    assert!(matches!(
        dot(&[1.0], &[1.0, 2.0]),
        Err(LinalgError::DimensionMismatch { .. })
    ));

    // [[1, 2], [3, 4]] * [5, 6] = [1*5 + 2*6, 3*5 + 4*6] = [17, 39].
    let product = mat_vec(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0], 2).expect("well-formed");
    assert_eq!(product, vec![17.0, 39.0]);

    assert!(matches!(
        mat_vec(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2),
        Err(LinalgError::DimensionMismatch { .. })
    ));
}

#[test]
fn quadratic_form_clamps_tiny_negatives_to_zero() {
    // A matrix whose exact quadratic form at this x is zero, but whose *computed*
    // value can land a few ulps either side of it. The clamp must guarantee the
    // caller never sees a negative it would then take the square root of.
    let matrix = [1.0, 0.0, 0.0, 1.0];
    let form = quadratic_form_nonnegative(&matrix, &[0.0, 0.0], 2).expect("well-formed");
    assert_eq!(form, 0.0);
    assert!(form.sqrt().is_finite());

    // A deliberately indefinite matrix produces a genuinely negative exact form.
    // The clamp still fires -- which is the correct *defensive* behaviour: the
    // production path only ever passes an SPD inverse here, and a NaN score would
    // be strictly worse than a zero bonus.
    let indefinite = [-1.0, 0.0, 0.0, -1.0];
    let form = quadratic_form_nonnegative(&indefinite, &[1.0, 1.0], 2).expect("well-formed");
    assert_eq!(
        form, 0.0,
        "a negative form must be clamped, never square-rooted"
    );

    // The usual case is untouched: x^T I x = ||x||^2.
    let form = quadratic_form_nonnegative(&matrix, &[3.0, 4.0], 2).expect("well-formed");
    assert_eq!(form, 25.0);
}

#[test]
fn quadratic_form_rejects_non_finite_results() {
    let matrix = [f64::MAX, 0.0, 0.0, f64::MAX];
    let result = quadratic_form_nonnegative(&matrix, &[f64::MAX, f64::MAX], 2);
    assert!(matches!(result, Err(LinalgError::NonFinite { .. })));
}

#[test]
fn symmetrize_projects_onto_the_symmetric_matrices() {
    let mut matrix = [1.0, 2.0, 4.0, 3.0];
    symmetrize(&mut matrix, 2);
    assert_eq!(matrix, [1.0, 3.0, 3.0, 3.0]);

    // Idempotent, and a fixed point on already-symmetric input.
    let mut already = [1.0, 3.0, 3.0, 3.0];
    symmetrize(&mut already, 2);
    assert_eq!(already, [1.0, 3.0, 3.0, 3.0]);
}

#[test]
fn scale_matrix_scales_every_entry() {
    assert_eq!(scale_matrix(&[1.0, -2.0, 3.0], 2.0), vec![2.0, -4.0, 6.0]);
}

#[test]
fn lower_triangular_mat_vec_ignores_the_upper_triangle() {
    // Row-major [[2, 0], [3, 4]] with garbage above the diagonal must give the
    // same answer as a clean lower-triangular matrix, because the routine only
    // reads j <= i.
    let clean = [2.0, 0.0, 3.0, 4.0];
    let dirty = [2.0, 999.0, 3.0, 4.0];
    let z = [1.0, 1.0];
    let a = lower_triangular_mat_vec(&clean, &z, 2).expect("well-formed");
    let b = lower_triangular_mat_vec(&dirty, &z, 2).expect("well-formed");
    assert_eq!(a, vec![2.0, 7.0]);
    assert_eq!(a, b);
}

// ═════════════════════════════════════════════════════════════════════════════
// linalg: Gauss-Jordan (the test-only oracle)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn gauss_jordan_inverts_a_known_matrix() {
    // [[4, 7], [2, 6]]^-1 = [[0.6, -0.7], [-0.2, 0.4]].
    let inverse = gauss_jordan_inverse(&[4.0, 7.0, 2.0, 6.0], 2).expect("non-singular");
    let expected = [0.6, -0.7, -0.2, 0.4];
    assert!(max_abs_diff(&inverse, &expected) < 1e-12);
}

#[test]
fn gauss_jordan_needs_its_pivoting() {
    // A matrix with a zero in the (0,0) slot is only invertible by a routine that
    // *pivots*. If the oracle could not handle this it would be a bad oracle.
    let inverse = gauss_jordan_inverse(&[0.0, 1.0, 1.0, 0.0], 2).expect("non-singular");
    assert_eq!(inverse, vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn gauss_jordan_rejects_a_singular_matrix() {
    // [[1, 2], [2, 4]] has rank 1.
    let result = gauss_jordan_inverse(&[1.0, 2.0, 2.0, 4.0], 2);
    assert!(
        result.is_err(),
        "a singular matrix has no inverse to report"
    );
    assert!(matches!(
        gauss_jordan_inverse(&[], 0),
        Err(LinalgError::ZeroDimension)
    ));
}
