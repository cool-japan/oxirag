#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements
)]

use crate::ares_eval::evaluator::AresEvaluator;
use crate::ares_eval::types::{AresConfig, AresError, PpiInterval};

/// Absolute tolerance for floating-point comparisons in the tests.
const EPS: f32 = 1e-5;

/// Arithmetic mean of a slice (test helper, mirrors the evaluator).
fn mean(values: &[f32]) -> f32 {
    values.iter().copied().sum::<f32>() / values.len() as f32
}

// ── AresConfig ────────────────────────────────────────────────────────────────

#[test]
fn config_default_confidence_is_0_95() {
    let config = AresConfig::default();
    assert_eq!(config.confidence, 0.95);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(AresConfig::new(), AresConfig::default());
}

#[test]
fn config_builder_sets_confidence() {
    let config = AresConfig::new().with_confidence(0.99);
    assert_eq!(config.confidence, 0.99);
}

#[test]
fn config_builder_is_chainable_and_copy() {
    let base = AresConfig::new();
    let derived = base.with_confidence(0.90);
    // `base` is still usable because AresConfig is Copy.
    assert_eq!(base.confidence, 0.95);
    assert_eq!(derived.confidence, 0.90);
}

#[test]
fn config_with_confidence_0_90() {
    let config = AresConfig::new().with_confidence(0.90);
    assert_eq!(config.confidence, 0.90);
}

// ── z_value table ─────────────────────────────────────────────────────────────

#[test]
fn z_value_for_0_95_is_1_96() {
    let evaluator = AresEvaluator::new(AresConfig::new().with_confidence(0.95));
    assert!((evaluator.z_value() - 1.960).abs() < EPS);
}

#[test]
fn z_value_for_0_90_is_1_645() {
    let evaluator = AresEvaluator::new(AresConfig::new().with_confidence(0.90));
    assert!((evaluator.z_value() - 1.645).abs() < EPS);
}

#[test]
fn z_value_for_0_99_is_2_576() {
    let evaluator = AresEvaluator::new(AresConfig::new().with_confidence(0.99));
    assert!((evaluator.z_value() - 2.576).abs() < EPS);
}

#[test]
fn z_value_default_is_1_96() {
    let evaluator = AresEvaluator::default();
    assert!((evaluator.z_value() - 1.960).abs() < EPS);
}

#[test]
fn z_value_unsupported_level_falls_back_to_1_96() {
    let evaluator = AresEvaluator::new(AresConfig::new().with_confidence(0.80));
    assert!((evaluator.z_value() - 1.960).abs() < EPS);
}

#[test]
fn z_value_another_unsupported_level_falls_back() {
    let evaluator = AresEvaluator::new(AresConfig::new().with_confidence(0.5));
    assert!((evaluator.z_value() - 1.960).abs() < EPS);
}

#[test]
fn z_value_increases_with_confidence() {
    let z90 = AresEvaluator::new(AresConfig::new().with_confidence(0.90)).z_value();
    let z95 = AresEvaluator::new(AresConfig::new().with_confidence(0.95)).z_value();
    let z99 = AresEvaluator::new(AresConfig::new().with_confidence(0.99)).z_value();
    assert!(z90 < z95);
    assert!(z95 < z99);
}

// ── ppi_estimate: point estimate ──────────────────────────────────────────────

#[test]
fn ppi_point_estimate_equals_unlabeled_mean_plus_rectifier() {
    let labeled = vec![(0.4_f32, 0.5_f32), (0.6, 0.7), (0.3, 0.4), (0.8, 0.9)];
    let unlabeled = vec![0.4_f32, 0.5, 0.6, 0.5, 0.4, 0.6];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    let rectifier = mean(&labeled.iter().map(|&(p, l)| l - p).collect::<Vec<_>>());
    let expected = mean(&unlabeled) + rectifier;
    assert!((ppi.point_estimate - expected).abs() < EPS);
}

#[test]
fn ppi_rectifier_is_positive_for_low_biased_judge() {
    // Judge under-predicts by 0.1 on every labeled pair.
    let labeled = vec![(0.4_f32, 0.5_f32), (0.6, 0.7), (0.2, 0.3)];
    let unlabeled = vec![0.5_f32, 0.5, 0.5, 0.5];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    // unlabeled mean = 0.5, rectifier = +0.1 ⇒ point ≈ 0.6.
    assert!((ppi.point_estimate - 0.6).abs() < EPS);
    assert!(ppi.point_estimate > mean(&unlabeled));
}

#[test]
fn ppi_rectifier_is_negative_for_high_biased_judge() {
    // Judge over-predicts by 0.2 on every labeled pair.
    let labeled = vec![(0.7_f32, 0.5_f32), (0.8, 0.6), (0.5, 0.3)];
    let unlabeled = vec![0.6_f32, 0.6, 0.6, 0.6];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    // unlabeled mean = 0.6, rectifier = -0.2 ⇒ point ≈ 0.4.
    assert!((ppi.point_estimate - 0.4).abs() < EPS);
    assert!(ppi.point_estimate < mean(&unlabeled));
}

#[test]
fn ppi_biased_judge_corrects_toward_truth() {
    // Ground-truth rate on the labeled set is high (0.9 mean of labels), but the
    // judge systematically under-predicts. The rectifier should pull the
    // unlabeled-based estimate upward, toward the true rate.
    let labeled = vec![(0.6_f32, 0.9_f32), (0.7, 1.0), (0.5, 0.8)];
    let unlabeled = vec![0.6_f32, 0.7, 0.5, 0.6, 0.7];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    let raw_unlabeled_mean = mean(&unlabeled);
    assert!(ppi.point_estimate > raw_unlabeled_mean);
}

#[test]
fn ppi_perfect_judge_has_zero_rectifier() {
    // prediction == label on every labeled pair ⇒ rectifier ≈ 0.
    let labeled = vec![(0.3_f32, 0.3_f32), (0.7, 0.7), (1.0, 1.0), (0.0, 0.0)];
    let unlabeled = vec![0.2_f32, 0.4, 0.6, 0.8, 0.5];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    // point_estimate should equal the unlabeled mean (rectifier ≈ 0).
    assert!((ppi.point_estimate - mean(&unlabeled)).abs() < EPS);
}

#[test]
fn ppi_perfect_judge_variance_comes_only_from_unlabeled() {
    // Residuals are all zero ⇒ residual variance is zero ⇒ the entire PPI
    // half-width is driven by the unlabeled spread.
    let labeled = vec![(0.5_f32, 0.5_f32), (0.5, 0.5)];
    let unlabeled = vec![0.0_f32, 1.0, 0.0, 1.0]; // mean 0.5, nonzero spread

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    assert!(ppi.half_width > 0.0);
    // Compare against a hand-computed variance: var(unlabeled)/n_unlabeled.
    // unlabeled var = 0.25, n = 4 ⇒ variance = 0.0625, sqrt = 0.25.
    let expected_half = 1.960 * 0.25;
    assert!((ppi.half_width - expected_half).abs() < EPS);
}

#[test]
fn ppi_handles_labels_in_unit_interval_not_just_binary() {
    // Labels need not be 0/1; graded labels in [0,1] are supported.
    let labeled = vec![(0.45_f32, 0.5_f32), (0.55, 0.6), (0.35, 0.4)];
    let unlabeled = vec![0.5_f32, 0.5, 0.5];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    // rectifier = +0.05, unlabeled mean = 0.5 ⇒ point = 0.55.
    assert!((ppi.point_estimate - 0.55).abs() < EPS);
}

#[test]
fn ppi_constant_unlabeled_gives_zero_unlabeled_variance_term() {
    // A constant unlabeled set contributes no variance from the unlabeled term;
    // only the labeled residual term remains.
    let labeled = vec![(0.4_f32, 0.6_f32), (0.5, 0.4)]; // residuals: +0.2, -0.1
    let unlabeled = vec![0.5_f32, 0.5, 0.5, 0.5, 0.5];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    // residual mean = 0.05, residual var = mean((0.15)^2,(−0.15)^2)=0.0225.
    // n_labeled = 2 ⇒ variance = 0.0225/2 = 0.01125, sqrt ≈ 0.106066.
    let expected_half = 1.960 * (0.01125_f32).sqrt();
    assert!((ppi.half_width - expected_half).abs() < 1e-4);
}

// ── ppi_estimate: interval ordering / sanity ──────────────────────────────────

#[test]
fn ppi_interval_is_ordered() {
    let labeled = vec![(0.4_f32, 0.5_f32), (0.6, 0.7), (0.3, 0.5)];
    let unlabeled = vec![0.3_f32, 0.5, 0.7, 0.4, 0.6];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    assert!(ppi.ci_low <= ppi.point_estimate);
    assert!(ppi.point_estimate <= ppi.ci_high);
    assert!(ppi.half_width >= 0.0);
}

#[test]
fn ppi_ci_edges_match_point_plus_minus_half_width() {
    let labeled = vec![(0.4_f32, 0.55_f32), (0.6, 0.7)];
    let unlabeled = vec![0.3_f32, 0.5, 0.7];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    assert!((ppi.ci_low - (ppi.point_estimate - ppi.half_width)).abs() < EPS);
    assert!((ppi.ci_high - (ppi.point_estimate + ppi.half_width)).abs() < EPS);
}

#[test]
fn ppi_interval_fields_are_finite() {
    let labeled = vec![(0.1_f32, 0.2_f32), (0.9, 0.8), (0.5, 0.6)];
    let unlabeled = vec![0.2_f32, 0.4, 0.6, 0.8];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    assert!(ppi.point_estimate.is_finite());
    assert!(ppi.ci_low.is_finite());
    assert!(ppi.ci_high.is_finite());
    assert!(ppi.half_width.is_finite());
}

#[test]
fn ppi_zero_variance_collapses_interval_to_point() {
    // Perfect judge (zero residuals) AND constant unlabeled set ⇒ zero variance.
    let labeled = vec![(0.5_f32, 0.5_f32), (0.5, 0.5)];
    let unlabeled = vec![0.5_f32, 0.5, 0.5];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    assert!((ppi.half_width - 0.0).abs() < EPS);
    assert!((ppi.ci_low - ppi.point_estimate).abs() < EPS);
    assert!((ppi.ci_high - ppi.point_estimate).abs() < EPS);
}

#[test]
fn ppi_single_labeled_pair_works() {
    let labeled = vec![(0.4_f32, 0.6_f32)];
    let unlabeled = vec![0.4_f32, 0.5, 0.6];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    // rectifier = +0.2, unlabeled mean = 0.5 ⇒ point = 0.7.
    assert!((ppi.point_estimate - 0.7).abs() < EPS);
    // Single labeled pair ⇒ residual var = 0; constant-ish unlabeled spread.
    assert!(ppi.half_width >= 0.0);
}

#[test]
fn ppi_single_unlabeled_prediction_works() {
    let labeled = vec![(0.4_f32, 0.5_f32), (0.6, 0.7)];
    let unlabeled = vec![0.5_f32];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();

    // unlabeled var = 0 (single point) ⇒ variance from residuals only.
    assert!(ppi.point_estimate.is_finite());
    assert!(ppi.half_width >= 0.0);
}

#[test]
fn ppi_larger_unlabeled_set_does_not_increase_half_width() {
    // Adding more unlabeled samples with the same spread shrinks (or keeps) the
    // unlabeled variance term because of the /n_unlabeled factor.
    let labeled = vec![(0.5_f32, 0.5_f32), (0.5, 0.5)];
    let small = vec![0.0_f32, 1.0];
    let large = vec![0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let evaluator = AresEvaluator::default();
    let hw_small = evaluator.ppi_estimate(&labeled, &small).unwrap().half_width;
    let hw_large = evaluator.ppi_estimate(&labeled, &large).unwrap().half_width;

    assert!(hw_large <= hw_small + EPS);
}

// ── ppi vs classical: tightness ───────────────────────────────────────────────

#[test]
fn ppi_half_width_tighter_than_classical_for_informative_judge() {
    // Constructed case: a perfect judge (zero residuals) plus a CONSTANT large
    // unlabeled set ⇒ PPI variance is zero, so PPI half-width is zero. The
    // classical labeled-only estimate over varying labels has positive variance.
    let labeled = vec![(1.0_f32, 1.0_f32), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
    let unlabeled = vec![0.5_f32; 100]; // many constant unlabeled predictions
    let labels: Vec<f32> = labeled.iter().map(|&(_, l)| l).collect();

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();
    let classical = evaluator.classical_estimate(&labels).unwrap();

    assert!(ppi.half_width < classical.half_width);
    // PPI half-width collapses to (near) zero here.
    assert!(ppi.half_width < EPS);
    // Classical still has a real interval from the 0/1 label spread.
    assert!(classical.half_width > 0.0);
}

#[test]
fn ppi_and_classical_both_finite_and_ordered() {
    let labeled = vec![(0.4_f32, 0.5_f32), (0.6, 0.7), (0.3, 0.4), (0.8, 0.9)];
    let unlabeled = vec![0.3_f32, 0.5, 0.7, 0.4, 0.6, 0.5];
    let labels: Vec<f32> = labeled.iter().map(|&(_, l)| l).collect();

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();
    let classical = evaluator.classical_estimate(&labels).unwrap();

    for interval in [ppi, classical] {
        assert!(interval.point_estimate.is_finite());
        assert!(interval.ci_low.is_finite());
        assert!(interval.ci_high.is_finite());
        assert!(interval.half_width.is_finite());
        assert!(interval.ci_low <= interval.point_estimate);
        assert!(interval.point_estimate <= interval.ci_high);
        assert!(interval.half_width >= 0.0);
    }
}

#[test]
fn ppi_tighter_with_large_low_variance_unlabeled_set() {
    // Perfect judge + many low-spread unlabeled predictions ⇒ tiny PPI variance.
    let labeled = vec![(0.5_f32, 0.5_f32), (0.6, 0.6), (0.4, 0.4)];
    let mut unlabeled = Vec::new();
    for _ in 0..200 {
        unlabeled.push(0.49_f32);
        unlabeled.push(0.51_f32);
    }
    let labels: Vec<f32> = vec![0.5, 0.6, 0.4];

    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();
    let classical = evaluator.classical_estimate(&labels).unwrap();

    assert!(ppi.half_width < classical.half_width);
}

// ── classical_estimate ────────────────────────────────────────────────────────

#[test]
fn classical_point_estimate_equals_label_mean() {
    let labels = vec![1.0_f32, 0.0, 1.0, 1.0, 0.0];
    let evaluator = AresEvaluator::default();
    let classical = evaluator.classical_estimate(&labels).unwrap();
    assert!((classical.point_estimate - mean(&labels)).abs() < EPS);
}

#[test]
fn classical_point_estimate_for_graded_labels() {
    let labels = vec![0.2_f32, 0.4, 0.6, 0.8];
    let evaluator = AresEvaluator::default();
    let classical = evaluator.classical_estimate(&labels).unwrap();
    assert!((classical.point_estimate - 0.5).abs() < EPS);
}

#[test]
fn classical_all_ones_has_zero_half_width() {
    let labels = vec![1.0_f32, 1.0, 1.0, 1.0];
    let evaluator = AresEvaluator::default();
    let classical = evaluator.classical_estimate(&labels).unwrap();
    assert!((classical.point_estimate - 1.0).abs() < EPS);
    assert!((classical.half_width - 0.0).abs() < EPS);
}

#[test]
fn classical_all_zeros_has_zero_half_width() {
    let labels = vec![0.0_f32, 0.0, 0.0];
    let evaluator = AresEvaluator::default();
    let classical = evaluator.classical_estimate(&labels).unwrap();
    assert!((classical.point_estimate - 0.0).abs() < EPS);
    assert!((classical.half_width - 0.0).abs() < EPS);
}

#[test]
fn classical_half_width_matches_formula() {
    // labels = [1,0,1,0]: mean 0.5, population var 0.25, n 4 ⇒ var/n = 0.0625.
    // half_width = 1.96 * sqrt(0.0625) = 1.96 * 0.25 = 0.49.
    let labels = vec![1.0_f32, 0.0, 1.0, 0.0];
    let evaluator = AresEvaluator::default();
    let classical = evaluator.classical_estimate(&labels).unwrap();
    assert!((classical.half_width - (1.960 * 0.25)).abs() < EPS);
}

#[test]
fn classical_interval_is_ordered() {
    let labels = vec![1.0_f32, 0.0, 1.0, 1.0, 0.0, 1.0];
    let evaluator = AresEvaluator::default();
    let classical = evaluator.classical_estimate(&labels).unwrap();
    assert!(classical.ci_low <= classical.point_estimate);
    assert!(classical.point_estimate <= classical.ci_high);
    assert!(classical.half_width >= 0.0);
}

#[test]
fn classical_uses_confidence_level_for_width() {
    let labels = vec![1.0_f32, 0.0, 1.0, 0.0];
    let e95 = AresEvaluator::new(AresConfig::new().with_confidence(0.95));
    let e99 = AresEvaluator::new(AresConfig::new().with_confidence(0.99));
    let w95 = e95.classical_estimate(&labels).unwrap().half_width;
    let w99 = e99.classical_estimate(&labels).unwrap().half_width;
    // Higher confidence ⇒ wider interval.
    assert!(w99 > w95);
}

#[test]
fn classical_single_label_has_zero_width() {
    let labels = vec![0.7_f32];
    let evaluator = AresEvaluator::default();
    let classical = evaluator.classical_estimate(&labels).unwrap();
    assert!((classical.point_estimate - 0.7).abs() < EPS);
    assert!((classical.half_width - 0.0).abs() < EPS);
}

// ── errors ────────────────────────────────────────────────────────────────────

#[test]
fn ppi_empty_labeled_errors() {
    let evaluator = AresEvaluator::default();
    let labeled: Vec<(f32, f32)> = Vec::new();
    let unlabeled = vec![0.5_f32, 0.5];
    let err = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap_err();
    assert_eq!(err, AresError::EmptyLabeled);
}

#[test]
fn ppi_empty_unlabeled_errors() {
    let evaluator = AresEvaluator::default();
    let labeled = vec![(0.4_f32, 0.5_f32)];
    let unlabeled: Vec<f32> = Vec::new();
    let err = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap_err();
    assert_eq!(err, AresError::EmptyUnlabeled);
}

#[test]
fn ppi_empty_labeled_takes_precedence_over_empty_unlabeled() {
    let evaluator = AresEvaluator::default();
    let labeled: Vec<(f32, f32)> = Vec::new();
    let unlabeled: Vec<f32> = Vec::new();
    let err = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap_err();
    assert_eq!(err, AresError::EmptyLabeled);
}

#[test]
fn classical_empty_labels_errors() {
    let evaluator = AresEvaluator::default();
    let labels: Vec<f32> = Vec::new();
    let err = evaluator.classical_estimate(&labels).unwrap_err();
    assert_eq!(err, AresError::EmptyLabeled);
}

#[test]
fn error_display_messages() {
    assert_eq!(AresError::EmptyLabeled.to_string(), "labeled set is empty");
    assert_eq!(
        AresError::EmptyUnlabeled.to_string(),
        "unlabeled set is empty"
    );
}

// ── PpiInterval helpers ───────────────────────────────────────────────────────

#[test]
fn interval_width_is_twice_half_width() {
    let interval = PpiInterval {
        point_estimate: 0.5,
        ci_low: 0.4,
        ci_high: 0.6,
        half_width: 0.1,
    };
    assert!((interval.width() - 0.2).abs() < EPS);
}

#[test]
fn interval_contains_inside_value() {
    let interval = PpiInterval {
        point_estimate: 0.5,
        ci_low: 0.4,
        ci_high: 0.6,
        half_width: 0.1,
    };
    assert!(interval.contains(0.5));
    assert!(interval.contains(0.4));
    assert!(interval.contains(0.6));
}

#[test]
fn interval_does_not_contain_outside_value() {
    let interval = PpiInterval {
        point_estimate: 0.5,
        ci_low: 0.4,
        ci_high: 0.6,
        half_width: 0.1,
    };
    assert!(!interval.contains(0.3));
    assert!(!interval.contains(0.7));
}

#[test]
fn ppi_point_estimate_lies_within_its_own_interval() {
    let labeled = vec![(0.4_f32, 0.5_f32), (0.6, 0.7)];
    let unlabeled = vec![0.3_f32, 0.5, 0.7];
    let evaluator = AresEvaluator::default();
    let ppi = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();
    assert!(ppi.contains(ppi.point_estimate));
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn ppi_is_deterministic() {
    let labeled = vec![(0.4_f32, 0.5_f32), (0.6, 0.7), (0.3, 0.4)];
    let unlabeled = vec![0.3_f32, 0.5, 0.7, 0.4, 0.6];
    let evaluator = AresEvaluator::default();
    let a = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();
    let b = evaluator.ppi_estimate(&labeled, &unlabeled).unwrap();
    assert_eq!(a, b);
}

#[test]
fn classical_is_deterministic() {
    let labels = vec![1.0_f32, 0.0, 1.0, 1.0, 0.0];
    let evaluator = AresEvaluator::default();
    let a = evaluator.classical_estimate(&labels).unwrap();
    let b = evaluator.classical_estimate(&labels).unwrap();
    assert_eq!(a, b);
}

#[test]
fn z_value_is_deterministic() {
    let evaluator = AresEvaluator::new(AresConfig::new().with_confidence(0.99));
    assert_eq!(evaluator.z_value(), evaluator.z_value());
}

#[test]
fn ppi_independent_of_labeled_order() {
    let labeled_a = vec![(0.4_f32, 0.5_f32), (0.6, 0.7), (0.3, 0.4)];
    let labeled_b = vec![(0.3_f32, 0.4_f32), (0.4, 0.5), (0.6, 0.7)];
    let unlabeled = vec![0.5_f32, 0.5, 0.5, 0.5];
    let evaluator = AresEvaluator::default();
    let a = evaluator.ppi_estimate(&labeled_a, &unlabeled).unwrap();
    let b = evaluator.ppi_estimate(&labeled_b, &unlabeled).unwrap();
    // Mean-based statistics are invariant to ordering.
    assert!((a.point_estimate - b.point_estimate).abs() < EPS);
    assert!((a.half_width - b.half_width).abs() < EPS);
}

#[test]
fn evaluator_default_matches_new_with_default_config() {
    let a = AresEvaluator::default();
    let b = AresEvaluator::new(AresConfig::default());
    let labeled = vec![(0.4_f32, 0.5_f32), (0.6, 0.7)];
    let unlabeled = vec![0.5_f32, 0.5];
    assert_eq!(
        a.ppi_estimate(&labeled, &unlabeled).unwrap(),
        b.ppi_estimate(&labeled, &unlabeled).unwrap()
    );
}
