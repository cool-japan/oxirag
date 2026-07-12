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
//! Tests for the Cholesky factorization and its scale-aware jitter repair path.

use super::max_abs_diff;
use crate::bandit_ranker::linalg::{
    LinalgError, cholesky_lower, cholesky_with_jitter, scaled_identity,
};
use crate::bandit_ranker::rng::SplitMix64Rng;

// ═════════════════════════════════════════════════════════════════════════════
// linalg: Cholesky
// ═════════════════════════════════════════════════════════════════════════════

/// Reconstruct `L * L^T` from a lower-triangular factor.
fn reconstruct(lower: &[f64], dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = 0.0;
            for k in 0..=i.min(j) {
                acc += lower[i * dim + k] * lower[j * dim + k];
            }
            out[i * dim + j] = acc;
        }
    }
    out
}

#[test]
fn cholesky_factor_reconstructs_its_input() {
    // [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] is the classic textbook SPD
    // example, with L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]].
    let matrix = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let lower = cholesky_lower(&matrix, 3).expect("the matrix is SPD");
    assert!(max_abs_diff(&lower, &[2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]) < 1e-12);

    let reconstructed = reconstruct(&lower, 3);
    assert!(
        max_abs_diff(&reconstructed, &matrix) < 1e-12,
        "L L^T must reconstruct M"
    );

    // The strict upper triangle must be exactly zero -- `lower_triangular_mat_vec`
    // relies on it, and so does the interpretation of L as a factor.
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert_eq!(lower[i * 3 + j], 0.0);
        }
    }
}

#[test]
fn cholesky_reconstructs_random_spd_matrices() {
    // Generate M = B B^T + I, which is SPD by construction, and check the round
    // trip for a range of dimensions -- including d = 1, the degenerate case that
    // a recurrence with an inner loop is most likely to get wrong.
    let mut rng = SplitMix64Rng::new(1_234_567);
    for dim in 1..=8 {
        let b: Vec<f64> = (0..dim * dim).map(|_| rng.next_standard_normal()).collect();
        let mut matrix = scaled_identity(dim, 1.0);
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = 0.0;
                for k in 0..dim {
                    acc += b[i * dim + k] * b[j * dim + k];
                }
                matrix[i * dim + j] += acc;
            }
        }
        let lower = cholesky_lower(&matrix, dim).expect("B B^T + I is SPD");
        let reconstructed = reconstruct(&lower, dim);
        let error = max_abs_diff(&reconstructed, &matrix);
        assert!(
            error < 1e-9,
            "L L^T must reconstruct M at dim {dim}, error was {error}"
        );
    }
}

#[test]
fn cholesky_rejects_a_non_positive_definite_matrix() {
    // [[1, 2], [2, 1]] has eigenvalues 3 and -1.
    let result = cholesky_lower(&[1.0, 2.0, 2.0, 1.0], 2);
    assert!(matches!(
        result,
        Err(LinalgError::NotPositiveDefinite { index: 1, .. })
    ));

    // The zero matrix fails at the very first pivot rather than returning zeros.
    assert!(matches!(
        cholesky_lower(&[0.0, 0.0, 0.0, 0.0], 2),
        Err(LinalgError::NotPositiveDefinite { index: 0, .. })
    ));

    assert!(matches!(
        cholesky_lower(&[], 0),
        Err(LinalgError::ZeroDimension)
    ));
    assert!(matches!(
        cholesky_lower(&[f64::NAN, 0.0, 0.0, 1.0], 2),
        Err(LinalgError::NonFinite { .. })
    ));
}

#[test]
fn cholesky_jitter_path_engages_instead_of_producing_nan() {
    // An indefinite matrix: eigenvalues 3 and -1. An unguarded Cholesky would take
    // sqrt of a negative pivot and return a matrix full of NaN, which would then
    // silently poison every Thompson sample forever. The jitter path must instead
    // find a ridge that restores positive-definiteness -- and *report* how big it
    // had to be.
    let matrix = [1.0, 2.0, 2.0, 1.0];
    let (lower, jitter) = cholesky_with_jitter(&matrix, 2).expect("the ridge ladder must reach PD");

    assert!(jitter > 0.0, "the jitter path must have engaged");
    assert!(
        jitter > 1.0,
        "restoring PD requires a ridge exceeding the magnitude of the negative \
         eigenvalue (-1), got {jitter}"
    );
    assert!(
        lower.iter().all(|value| value.is_finite()),
        "the whole point is that no NaN is produced"
    );

    // L L^T must reconstruct M + jitter * I -- not M. The repair is *reported*, not
    // hidden: the caller knows exactly which matrix was factored.
    let mut expected = matrix;
    expected[0] += jitter;
    expected[3] += jitter;
    let reconstructed = reconstruct(&lower, 2);
    assert!(
        max_abs_diff(&reconstructed, &expected) < 1e-9,
        "L L^T must reconstruct M + jitter * I exactly"
    );
}

#[test]
fn cholesky_jitter_is_tiny_for_a_merely_singular_matrix() {
    // A rank-deficient PSD matrix (x x^T for x = [1, 1]) is only a hair outside the
    // cone: it needs a ridge, but a *minuscule* one. This is the realistic case --
    // a posterior that has collapsed to numerical zero in one direction -- and the
    // ridge must be small enough not to distort the sample.
    let matrix = [1.0, 1.0, 1.0, 1.0];
    let (lower, jitter) = cholesky_with_jitter(&matrix, 2).expect("a tiny ridge suffices");
    assert!(jitter > 0.0, "the jitter path must have engaged");
    assert!(
        jitter < 1e-9,
        "a merely-singular matrix must need only a rounding-scale ridge, got {jitter}"
    );
    assert!(lower.iter().all(|value| value.is_finite()));
}

#[test]
fn cholesky_jitter_does_not_perturb_an_already_positive_definite_matrix() {
    let matrix = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let (lower, jitter) = cholesky_with_jitter(&matrix, 3).expect("the matrix is SPD");
    assert_eq!(jitter, 0.0, "an SPD matrix must be factored *as given*");
    assert!(max_abs_diff(&reconstruct(&lower, 3), &matrix) < 1e-12);
}

#[test]
fn cholesky_jitter_is_scale_aware() {
    // The same singular structure at a hugely different scale must get a
    // proportionally larger ridge -- a fixed absolute ridge would be beneath the
    // rounding noise here and would never restore definiteness.
    let big = [1e8, 1e8, 1e8, 1e8];
    let (_, big_jitter) = cholesky_with_jitter(&big, 2).expect("scale-aware ridge succeeds");
    let small = [1.0, 1.0, 1.0, 1.0];
    let (_, small_jitter) = cholesky_with_jitter(&small, 2).expect("scale-aware ridge succeeds");
    assert!(
        big_jitter > small_jitter * 1e6,
        "the ridge must scale with the matrix: got {big_jitter} vs {small_jitter}"
    );
}
