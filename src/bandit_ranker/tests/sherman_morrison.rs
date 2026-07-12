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
//! The module's headline numerical test, and the guards around it.

use super::max_abs_diff;
use crate::bandit_ranker::linalg::{
    LinalgError, gauss_jordan_inverse, mat_vec, scaled_identity, sherman_morrison_update,
};
use crate::bandit_ranker::rng::SplitMix64Rng;

// ═════════════════════════════════════════════════════════════════════════════
// linalg: Sherman-Morrison  (THE HEADLINE NUMERICAL TEST)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn sherman_morrison_matches_direct_inverse_after_long_update_sequence() {
    // Maintain A^-1 incrementally, and A explicitly, through a long and
    // deliberately nasty sequence of rank-1 updates. Then invert A from scratch
    // with a *structurally different* algorithm (Gauss-Jordan with partial
    // pivoting) and demand the two agree.
    //
    // This is the load-bearing numerical test of the whole module: the production
    // path never inverts a matrix, so if Sherman-Morrison drifts, *nothing else*
    // would notice -- the bandit would keep producing plausible rankings from a
    // silently-corrupted confidence geometry.
    let dim = 6;
    let lambda = 1.0;
    let mut a_inv = scaled_identity(dim, 1.0 / lambda);
    let mut a = scaled_identity(dim, lambda);
    let mut rng = SplitMix64Rng::new(0xC0FF_EE00);

    let updates = 400;
    for step in 0..updates {
        // A mixture of update vectors chosen to stress the arithmetic:
        //  * ordinary N(0,1) directions,
        //  * *tiny* vectors (1e-6 scale), whose rank-1 contribution is at the very
        //    edge of what an f64 sum can resolve against a matrix of order 1,
        //  * *large* vectors (1e3 scale), which drive A's condition number up and
        //    A^-1's entries down toward the rounding floor,
        //  * exactly-repeated directions, which push one eigenvalue of A up fast
        //    while leaving the others alone -- the classic ill-conditioning route.
        let x: Vec<f64> = match step % 4 {
            0 => (0..dim).map(|_| rng.next_standard_normal()).collect(),
            1 => (0..dim)
                .map(|_| 1e-6 * rng.next_standard_normal())
                .collect(),
            2 => (0..dim).map(|_| 1e3 * rng.next_standard_normal()).collect(),
            _ => {
                let mut repeated = vec![0.0; dim];
                repeated[step % dim] = 1.0;
                repeated
            }
        };

        let denominator = sherman_morrison_update(&mut a_inv, &x, dim).expect("A^-1 stays SPD");
        assert!(
            denominator >= 1.0,
            "1 + x^T A^-1 x is bounded below by 1 for a positive-definite A^-1, got {denominator}"
        );

        for i in 0..dim {
            for j in 0..dim {
                a[i * dim + j] += x[i] * x[j];
            }
        }
    }

    let direct = gauss_jordan_inverse(&a, dim).expect("A is SPD, hence non-singular");
    let error = max_abs_diff(&a_inv, &direct);

    // The entries of A^-1 are small here (A has been driven to a large norm by the
    // 1e3 updates), so an *absolute* tolerance is the honest one: it is the
    // strictly harder ask.
    assert!(
        error < 1e-9,
        "incrementally maintained A^-1 drifted from the directly-computed inverse \
         by {error} after {updates} rank-1 updates (tolerance 1e-9)"
    );

    // And the maintained inverse really is an inverse: A * A^-1 = I.
    for column in 0..dim {
        let mut basis = vec![0.0; dim];
        basis[column] = 1.0;
        let round_trip =
            mat_vec(&a, &mat_vec(&a_inv, &basis, dim).expect("dims"), dim).expect("dims");
        assert!(
            max_abs_diff(&round_trip, &basis) < 1e-9,
            "A * A^-1 must reconstruct the identity, column {column} did not"
        );
    }
}

#[test]
fn sherman_morrison_keeps_the_inverse_symmetric() {
    // Without the symmetrization step, asymmetric rounding accumulates and can
    // eventually tip A^-1 out of the positive-definite cone. Exact symmetry is the
    // invariant that prevents it.
    let dim = 5;
    let mut a_inv = scaled_identity(dim, 1.0);
    let mut rng = SplitMix64Rng::new(4242);
    for _ in 0..500 {
        let x: Vec<f64> = (0..dim).map(|_| rng.next_standard_normal()).collect();
        sherman_morrison_update(&mut a_inv, &x, dim).expect("A^-1 stays SPD");
    }
    for i in 0..dim {
        for j in 0..dim {
            assert_eq!(
                a_inv[i * dim + j],
                a_inv[j * dim + i],
                "A^-1 must be *bitwise* symmetric after the symmetrization projection"
            );
        }
    }
}

#[test]
fn sherman_morrison_denominator_is_exactly_one_for_a_zero_update() {
    // x = 0 gives 1 + 0^T A^-1 0 = 1 exactly, and A + 0 0^T = A, so the inverse
    // must be untouched. This is the "zero context" path, and it must not corrupt
    // anything.
    let dim = 3;
    let mut a_inv = scaled_identity(dim, 0.5);
    let before = a_inv.clone();
    let denominator = sherman_morrison_update(&mut a_inv, &[0.0, 0.0, 0.0], dim).expect("legal");
    assert_eq!(denominator, 1.0);
    assert_eq!(a_inv, before, "a zero update must leave A^-1 bit-identical");
}

#[test]
fn sherman_morrison_denominator_never_dips_below_one_on_spd_input() {
    // The proof: A^-1 SPD => x^T A^-1 x >= 0 => denominator >= 1. Assert it across
    // a wide sweep of scales, including scales where the quadratic form underflows.
    let dim = 4;
    let mut a_inv = scaled_identity(dim, 1.0);
    let mut rng = SplitMix64Rng::new(808);
    for scale in [1e-12, 1e-6, 1.0, 1e3, 1e6] {
        for _ in 0..40 {
            let x: Vec<f64> = (0..dim)
                .map(|_| scale * rng.next_standard_normal())
                .collect();
            let denominator = sherman_morrison_update(&mut a_inv, &x, dim).expect("SPD");
            assert!(
                denominator >= 1.0,
                "denominator {denominator} fell below 1 at scale {scale}, which is impossible \
                 for a positive-definite inverse"
            );
        }
    }
}

#[test]
fn sherman_morrison_detects_a_corrupted_inverse() {
    // The guard exists to catch an *impossible* state. Hand it one: a negative
    // definite "inverse" whose quadratic form is -2, giving a denominator of -1.
    // It must refuse rather than divide and hand back garbage.
    let mut corrupt = [-2.0, 0.0, 0.0, -2.0];
    let result = sherman_morrison_update(&mut corrupt, &[1.0, 0.0], 2);
    assert!(matches!(
        result,
        Err(LinalgError::DegenerateShermanMorrison { .. })
    ));
}

#[test]
fn sherman_morrison_rejects_non_finite_input() {
    let mut a_inv = scaled_identity(2, 1.0);
    assert!(matches!(
        sherman_morrison_update(&mut a_inv, &[f64::NAN, 0.0], 2),
        Err(LinalgError::NonFinite { .. })
    ));
    assert!(matches!(
        sherman_morrison_update(&mut a_inv, &[f64::INFINITY, 0.0], 2),
        Err(LinalgError::NonFinite { .. })
    ));
    assert!(matches!(
        sherman_morrison_update(&mut a_inv, &[1.0, 2.0, 3.0], 2),
        Err(LinalgError::DimensionMismatch { .. })
    ));
}
