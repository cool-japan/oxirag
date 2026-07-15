//! Tests for the exposure primitives and group exposure metrics.

use crate::fairness_ranking::exposure::{
    ExposureMetrics, exposure_at_rank, fairness_ndcg_at_k, total_exposure,
};
use crate::fairness_ranking::types::GroupId;

#[test]
fn exposure_discount_has_the_expected_shape() {
    // Position 0 (rank 1) gets exactly 1 / log2(2) = 1.0.
    assert!((exposure_at_rank(0) - 1.0).abs() < 1e-12);
    // Rank 2 gets 1 / log2(3).
    assert!((exposure_at_rank(1) - 1.0 / 3.0_f64.log2()).abs() < 1e-12);
    // Rank 3 gets 1 / log2(4) = 0.5.
    assert!((exposure_at_rank(2) - 0.5).abs() < 1e-12);
    // Strictly decreasing.
    for i in 0..50 {
        assert!(exposure_at_rank(i) > exposure_at_rank(i + 1));
    }
    // Total over the first three positions.
    let expected = 1.0 + 1.0 / 3.0_f64.log2() + 0.5;
    assert!((total_exposure(3) - expected).abs() < 1e-12);
}

#[test]
fn two_equal_documents_receive_unequal_exposure() {
    // The impossibility that motivates amortization: whichever of two equally
    // relevant documents goes first receives 58% more attention than the other.
    let first = exposure_at_rank(0);
    let second = exposure_at_rank(1);
    assert!((first / second - 1.585).abs() < 0.01);
    assert!(first > second);
}

#[test]
fn exposure_per_relevance_is_the_disparity_axis() {
    // Two groups, three each. Group 0 (protected) is ranked last despite equal
    // per-item relevance, so it receives less exposure per unit relevance.
    let relevances = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    // Ranked order: unprotected group first (positions 0,1,2), protected last.
    let groups = vec![
        GroupId(1),
        GroupId(1),
        GroupId(1),
        GroupId(0),
        GroupId(0),
        GroupId(0),
    ];
    let metrics = ExposureMetrics::measure(&relevances, &groups, 2).expect("ok");

    let protected = metrics.group(GroupId(0)).unwrap();
    let unprotected = metrics.group(GroupId(1)).unwrap();
    assert!(unprotected.exposure > protected.exposure);

    let disparity = metrics.disparity();
    assert!(disparity.gap > 0.0, "there is a real disparity here");
    assert_eq!(disparity.most_exposed, Some(GroupId(1)));
    assert_eq!(disparity.least_exposed, Some(GroupId(0)));
    // The ratio is min/max < 1.
    assert!(disparity.ratio < 1.0);
}

#[test]
fn interleaving_reduces_the_disparity() {
    // The same six items, now interleaved, have a smaller exposure-per-relevance
    // gap than when one group is ranked entirely ahead of the other. This is the
    // property the fair ranker exploits.
    let relevances = vec![1.0; 6];
    let segregated = vec![
        GroupId(1),
        GroupId(1),
        GroupId(1),
        GroupId(0),
        GroupId(0),
        GroupId(0),
    ];
    let interleaved = vec![
        GroupId(1),
        GroupId(0),
        GroupId(1),
        GroupId(0),
        GroupId(1),
        GroupId(0),
    ];
    let gap_segregated = ExposureMetrics::measure(&relevances, &segregated, 2)
        .unwrap()
        .disparity()
        .gap;
    let gap_interleaved = ExposureMetrics::measure(&relevances, &interleaved, 2)
        .unwrap()
        .disparity()
        .gap;
    assert!(gap_interleaved < gap_segregated);
}

#[test]
fn groups_without_relevance_are_excluded_not_zeroed() {
    // A group present in the ranking but with zero total relevance has an
    // undefined exposure-per-relevance and must be excluded from the disparity,
    // and reported, rather than being given an infinity that poisons the max.
    let relevances = vec![1.0, 0.0, 1.0];
    let groups = vec![GroupId(0), GroupId(1), GroupId(0)];
    let metrics = ExposureMetrics::measure(&relevances, &groups, 2).expect("ok");
    assert_eq!(metrics.groups_without_relevance(), vec![GroupId(1)]);
    // Only group 0 is comparable, so there is no disparity (a monoculture).
    assert_eq!(metrics.disparity().gap, 0.0);
    assert!(
        metrics
            .group(GroupId(1))
            .unwrap()
            .exposure_per_relevance()
            .is_none()
    );
}

#[test]
fn negative_relevance_is_rejected() {
    let relevances = vec![1.0, -0.5];
    let groups = vec![GroupId(0), GroupId(1)];
    assert!(ExposureMetrics::measure(&relevances, &groups, 2).is_err());
}

#[test]
fn ndcg_is_one_for_the_ideal_order_and_less_otherwise() {
    let relevances = vec![3.0, 2.0, 1.0];
    assert!((fairness_ndcg_at_k(&relevances, 0) - 1.0).abs() < 1e-12);
    let reversed = vec![1.0, 2.0, 3.0];
    assert!(fairness_ndcg_at_k(&reversed, 0) < 1.0);
    // Empty / all-zero relevance is 0, not NaN.
    assert_eq!(fairness_ndcg_at_k(&[0.0, 0.0], 0), 0.0);
}
