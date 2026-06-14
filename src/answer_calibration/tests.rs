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
    clippy::doc_markdown
)]
//! Tests for the `answer_calibration` module.

use super::calibrator::AnswerCalibrator;
use super::metrics::{
    brier_score, compute_metrics, expected_calibration_error, maximum_calibration_error,
    reliability_bins,
};
use super::types::{
    AnswerCalibrationError, AnswerCalibratorConfig, CalibrationMetrics, ConfidenceSignals,
    ReliabilityBin,
};

const EPS: f32 = 1e-5;

// ── Config defaults & builders ────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = AnswerCalibratorConfig::default();
    assert!((cfg.verbalized_weight - 0.34).abs() < EPS);
    assert!((cfg.agreement_weight - 0.33).abs() < EPS);
    assert!((cfg.support_weight - 0.33).abs() < EPS);
    assert_eq!(cfg.num_bins, 10);
}

#[test]
fn config_new_equals_default() {
    let a = AnswerCalibratorConfig::new();
    let b = AnswerCalibratorConfig::default();
    assert_eq!(a.verbalized_weight, b.verbalized_weight);
    assert_eq!(a.agreement_weight, b.agreement_weight);
    assert_eq!(a.support_weight, b.support_weight);
    assert_eq!(a.num_bins, b.num_bins);
}

#[test]
fn config_builder_verbalized_weight() {
    let cfg = AnswerCalibratorConfig::new().with_verbalized_weight(0.5);
    assert_eq!(cfg.verbalized_weight, 0.5);
}

#[test]
fn config_builder_agreement_weight() {
    let cfg = AnswerCalibratorConfig::new().with_agreement_weight(0.6);
    assert_eq!(cfg.agreement_weight, 0.6);
}

#[test]
fn config_builder_support_weight() {
    let cfg = AnswerCalibratorConfig::new().with_support_weight(0.7);
    assert_eq!(cfg.support_weight, 0.7);
}

#[test]
fn config_builder_num_bins() {
    let cfg = AnswerCalibratorConfig::new().with_num_bins(20);
    assert_eq!(cfg.num_bins, 20);
}

#[test]
fn config_builder_chained() {
    let cfg = AnswerCalibratorConfig::new()
        .with_verbalized_weight(0.1)
        .with_agreement_weight(0.2)
        .with_support_weight(0.3)
        .with_num_bins(5);
    assert_eq!(cfg.verbalized_weight, 0.1);
    assert_eq!(cfg.agreement_weight, 0.2);
    assert_eq!(cfg.support_weight, 0.3);
    assert_eq!(cfg.num_bins, 5);
}

// ── ConfidenceSignals constructors & builders ─────────────────────────────────

#[test]
fn signals_new_no_verbalized() {
    let s = ConfidenceSignals::new(0.5, 0.4);
    assert!(s.verbalized.is_none());
    assert_eq!(s.agreement, 0.5);
    assert_eq!(s.support, 0.4);
}

#[test]
fn signals_default_is_zeroed() {
    let s = ConfidenceSignals::default();
    assert!(s.verbalized.is_none());
    assert_eq!(s.agreement, 0.0);
    assert_eq!(s.support, 0.0);
}

#[test]
fn signals_with_verbalized() {
    let s = ConfidenceSignals::new(0.5, 0.4).with_verbalized(0.9);
    assert_eq!(s.verbalized, Some(0.9));
}

#[test]
fn signals_with_agreement_and_support() {
    let s = ConfidenceSignals::default()
        .with_agreement(0.7)
        .with_support(0.6);
    assert_eq!(s.agreement, 0.7);
    assert_eq!(s.support, 0.6);
}

// ── extract_verbalized: percentages ───────────────────────────────────────────

fn calibrator() -> AnswerCalibrator {
    AnswerCalibrator::new(AnswerCalibratorConfig::default())
}

#[test]
fn extract_percentage_90() {
    let c = calibrator();
    let v = c.extract_verbalized("I am 90% sure").unwrap();
    assert!((v - 0.9).abs() < EPS);
}

#[test]
fn extract_percentage_with_decimal() {
    let c = calibrator();
    let v = c.extract_verbalized("Confidence: 87.5%").unwrap();
    assert!((v - 0.875).abs() < EPS);
}

#[test]
fn extract_percentage_with_space() {
    let c = calibrator();
    let v = c.extract_verbalized("about 50 % likely").unwrap();
    assert!((v - 0.5).abs() < EPS);
}

#[test]
fn extract_percentage_100() {
    let c = calibrator();
    let v = c.extract_verbalized("100% certain").unwrap();
    assert!((v - 1.0).abs() < EPS);
}

#[test]
fn extract_percentage_zero() {
    let c = calibrator();
    let v = c.extract_verbalized("0% chance").unwrap();
    assert!((v - 0.0).abs() < EPS);
}

#[test]
fn extract_percentage_over_100_clamped() {
    let c = calibrator();
    // 150% should clamp to 1.0.
    let v = c.extract_verbalized("150% confident").unwrap();
    assert!((v - 1.0).abs() < EPS);
}

#[test]
fn extract_percentage_takes_precedence_over_words() {
    let c = calibrator();
    // Has both "maybe" and a percentage: percentage wins.
    let v = c.extract_verbalized("maybe around 70%").unwrap();
    assert!((v - 0.7).abs() < EPS);
}

#[test]
fn extract_lone_percent_sign_is_none() {
    let c = calibrator();
    // A '%' without a preceding number is not a signal.
    assert!(
        c.extract_verbalized("100 percent but written as % alone")
            .is_none()
    );
}

// ── extract_verbalized: hedge & confident words ───────────────────────────────

#[test]
fn extract_hedge_maybe_low() {
    let c = calibrator();
    let v = c.extract_verbalized("Maybe the answer is Paris").unwrap();
    assert!(v < 0.5);
    assert!((v - 0.3).abs() < EPS);
}

#[test]
fn extract_hedge_possibly_low() {
    let c = calibrator();
    let v = c.extract_verbalized("Possibly it is Rome").unwrap();
    assert!(v < 0.5);
}

#[test]
fn extract_hedge_i_think_low() {
    let c = calibrator();
    let v = c.extract_verbalized("I think it is Berlin").unwrap();
    assert!(v < 0.5);
}

#[test]
fn extract_confident_definitely_high() {
    let c = calibrator();
    let v = c.extract_verbalized("It is definitely Paris").unwrap();
    assert!(v > 0.5);
    assert!((v - 0.9).abs() < EPS);
}

#[test]
fn extract_confident_certainly_high() {
    let c = calibrator();
    let v = c
        .extract_verbalized("Certainly the capital of France")
        .unwrap();
    assert!(v > 0.5);
}

#[test]
fn extract_no_signal_is_none() {
    let c = calibrator();
    assert!(
        c.extract_verbalized("The capital of France is Paris")
            .is_none()
    );
}

#[test]
fn extract_empty_text_is_none() {
    let c = calibrator();
    assert!(c.extract_verbalized("").is_none());
}

#[test]
fn extract_word_boundary_not_substring() {
    let c = calibrator();
    // "unlikely" should not match the hedge word "likely".
    assert!(
        c.extract_verbalized("That is unlikely a coincidence here")
            .is_none()
    );
}

#[test]
fn extract_conflicting_words_average() {
    let c = calibrator();
    // Both confident and hedge present → averaged to the middle.
    let v = c.extract_verbalized("definitely, but maybe not").unwrap();
    assert!((v - 0.6).abs() < EPS);
}

#[test]
fn extract_case_insensitive() {
    let c = calibrator();
    let v = c.extract_verbalized("DEFINITELY correct").unwrap();
    assert!((v - 0.9).abs() < EPS);
}

// ── estimate_confidence ───────────────────────────────────────────────────────

#[test]
fn estimate_all_signals_present_weighted() {
    let c = calibrator();
    let signals = ConfidenceSignals::new(0.6, 0.9).with_verbalized(0.3);
    // weights 0.34/0.33/0.33 → (0.34*0.3 + 0.33*0.6 + 0.33*0.9)/1.0
    let expected = (0.34 * 0.3 + 0.33 * 0.6 + 0.33 * 0.9) / (0.34 + 0.33 + 0.33);
    let got = c.estimate_confidence(&signals);
    assert!((got - expected).abs() < EPS);
}

#[test]
fn estimate_in_unit_range() {
    let c = calibrator();
    let signals = ConfidenceSignals::new(0.5, 0.5).with_verbalized(0.5);
    let got = c.estimate_confidence(&signals);
    assert!((0.0..=1.0).contains(&got));
}

#[test]
fn estimate_all_equal_signals_returns_same() {
    let c = calibrator();
    let signals = ConfidenceSignals::new(0.7, 0.7).with_verbalized(0.7);
    let got = c.estimate_confidence(&signals);
    assert!((got - 0.7).abs() < EPS);
}

#[test]
fn estimate_missing_verbalized_redistributes() {
    let c = calibrator();
    // Without verbalized, weight redistributes over agreement & support only.
    let signals = ConfidenceSignals::new(0.4, 0.8);
    let expected = (0.33 * 0.4 + 0.33 * 0.8) / (0.33 + 0.33);
    let got = c.estimate_confidence(&signals);
    assert!((got - expected).abs() < EPS);
    // Equivalent to the simple mean of the two when weights are equal.
    assert!((got - 0.6).abs() < EPS);
}

#[test]
fn estimate_missing_verbalized_not_treated_as_zero() {
    let c = calibrator();
    // If verbalized were treated as 0.0 the blend would drop well below 0.6.
    let with_missing = c.estimate_confidence(&ConfidenceSignals::new(0.6, 0.6));
    assert!((with_missing - 0.6).abs() < EPS);
    let as_zero = c.estimate_confidence(&ConfidenceSignals::new(0.6, 0.6).with_verbalized(0.0));
    assert!(as_zero < with_missing);
}

#[test]
fn estimate_clamps_out_of_range_signals() {
    let c = calibrator();
    let signals = ConfidenceSignals::new(2.0, -1.0).with_verbalized(5.0);
    let got = c.estimate_confidence(&signals);
    // Clamped: agreement=1.0, support=0.0, verbalized=1.0.
    assert!((0.0..=1.0).contains(&got));
    let expected = (0.34 * 1.0 + 0.33 * 1.0 + 0.33 * 0.0) / 1.0;
    assert!((got - expected).abs() < EPS);
}

#[test]
fn estimate_zero_weights_fallback_mean_with_verbalized() {
    let cfg = AnswerCalibratorConfig::new()
        .with_verbalized_weight(0.0)
        .with_agreement_weight(0.0)
        .with_support_weight(0.0);
    let c = AnswerCalibrator::new(cfg);
    let signals = ConfidenceSignals::new(0.3, 0.9).with_verbalized(0.6);
    let got = c.estimate_confidence(&signals);
    assert!((got - 0.6).abs() < EPS); // mean of 0.6, 0.3, 0.9
}

#[test]
fn estimate_zero_weights_fallback_mean_without_verbalized() {
    let cfg = AnswerCalibratorConfig::new()
        .with_agreement_weight(0.0)
        .with_support_weight(0.0);
    let c = AnswerCalibrator::new(cfg);
    let signals = ConfidenceSignals::new(0.2, 0.8);
    let got = c.estimate_confidence(&signals);
    assert!((got - 0.5).abs() < EPS);
}

#[test]
fn estimate_weight_emphasis_shifts_result() {
    // Heavily weight support; result should approach the support value.
    let cfg = AnswerCalibratorConfig::new()
        .with_verbalized_weight(0.0)
        .with_agreement_weight(0.0)
        .with_support_weight(1.0);
    let c = AnswerCalibrator::new(cfg);
    let signals = ConfidenceSignals::new(0.1, 0.95).with_verbalized(0.1);
    let got = c.estimate_confidence(&signals);
    assert!((got - 0.95).abs() < EPS);
}

// ── brier_score ───────────────────────────────────────────────────────────────

#[test]
fn brier_perfect_predictions_zero() {
    let data = [(1.0_f32, true), (0.0, false), (1.0, true), (0.0, false)];
    let b = brier_score(&data).unwrap();
    assert!(b.abs() < EPS);
}

#[test]
fn brier_worst_predictions_one() {
    let data = [(0.0_f32, true), (1.0, false)];
    let b = brier_score(&data).unwrap();
    assert!((b - 1.0).abs() < EPS);
}

#[test]
fn brier_in_unit_range() {
    let data = [
        (0.7_f32, true),
        (0.3, false),
        (0.5, true),
        (0.9, false),
        (0.2, true),
    ];
    let b = brier_score(&data).unwrap();
    assert!((0.0..=1.0).contains(&b));
}

#[test]
fn brier_half_confidence_quarter_loss() {
    // Every prediction at 0.5 → squared error 0.25 regardless of outcome.
    let data = [(0.5_f32, true), (0.5, false), (0.5, true)];
    let b = brier_score(&data).unwrap();
    assert!((b - 0.25).abs() < EPS);
}

#[test]
fn brier_empty_errors() {
    let data: [(f32, bool); 0] = [];
    assert!(matches!(
        brier_score(&data),
        Err(AnswerCalibrationError::EmptyData)
    ));
}

#[test]
fn brier_clamps_inputs() {
    // Confidence 2.0 clamped to 1.0 → perfect on a correct answer.
    let data = [(2.0_f32, true), (-1.0, false)];
    let b = brier_score(&data).unwrap();
    assert!(b.abs() < EPS);
}

// ── ECE / MCE ─────────────────────────────────────────────────────────────────

#[test]
fn ece_perfectly_calibrated_zero() {
    // Every populated bin has avg confidence exactly equal to its accuracy:
    //   bin [0.5,0.6): conf 0.5, acc 0.5  → gap 0
    //   bin [0.0,0.1): conf 0.0, acc 0.0  → gap 0
    //   bin [1.0]:     conf 1.0, acc 1.0  → gap 0
    let calibrated = [
        (0.5_f32, true),
        (0.5, false),
        (0.0, false),
        (0.0, false),
        (1.0, true),
        (1.0, true),
    ];
    let ece = expected_calibration_error(&calibrated, 10).unwrap();
    assert!(ece.abs() < EPS);
}

#[test]
fn ece_overconfident_half() {
    // All predictions confidence 1.0 but only 50% correct → ECE ≈ 0.5.
    let data = [
        (1.0_f32, true),
        (1.0, false),
        (1.0, true),
        (1.0, false),
        (1.0, true),
        (1.0, false),
    ];
    let ece = expected_calibration_error(&data, 10).unwrap();
    assert!((ece - 0.5).abs() < EPS);
}

#[test]
fn mce_ge_ece_on_overconfident() {
    let data = [(1.0_f32, true), (1.0, false), (0.6, false), (0.6, false)];
    let ece = expected_calibration_error(&data, 10).unwrap();
    let mce = maximum_calibration_error(&data, 10).unwrap();
    assert!(mce >= ece - EPS);
}

#[test]
fn mce_single_bin_equals_ece() {
    // All in one bin → MCE equals ECE.
    let data = [
        (0.95_f32, true),
        (0.95, false),
        (0.95, false),
        (0.95, false),
    ];
    let ece = expected_calibration_error(&data, 10).unwrap();
    let mce = maximum_calibration_error(&data, 10).unwrap();
    assert!((mce - ece).abs() < EPS);
}

#[test]
fn ece_empty_errors() {
    let data: [(f32, bool); 0] = [];
    assert!(matches!(
        expected_calibration_error(&data, 10),
        Err(AnswerCalibrationError::EmptyData)
    ));
}

#[test]
fn mce_empty_errors() {
    let data: [(f32, bool); 0] = [];
    assert!(matches!(
        maximum_calibration_error(&data, 10),
        Err(AnswerCalibrationError::EmptyData)
    ));
}

#[test]
fn ece_well_calibrated_below_threshold() {
    let data = [(0.5_f32, true), (0.5, false), (1.0, true), (0.0, false)];
    let ece = expected_calibration_error(&data, 10).unwrap();
    assert!(ece < 0.1);
}

// ── reliability_bins ──────────────────────────────────────────────────────────

#[test]
fn bins_partition_unit_interval() {
    let data = [(0.1_f32, true), (0.5, false), (0.9, true)];
    let bins = reliability_bins(&data, 10).unwrap();
    assert_eq!(bins.len(), 10);
    assert!((bins[0].lower - 0.0).abs() < EPS);
    assert!((bins.last().unwrap().upper - 1.0).abs() < EPS);
    // Contiguous: each bin's upper equals the next bin's lower.
    for w in bins.windows(2) {
        assert!((w[0].upper - w[1].lower).abs() < EPS);
    }
}

#[test]
fn bins_counts_sum_to_n() {
    let data = [
        (0.05_f32, true),
        (0.15, false),
        (0.25, true),
        (0.95, true),
        (1.0, false),
        (0.5, true),
    ];
    let bins = reliability_bins(&data, 10).unwrap();
    let total: usize = bins.iter().map(|b| b.count).sum();
    assert_eq!(total, data.len());
}

#[test]
fn bins_confidence_one_lands_in_last_bin() {
    let data = [(1.0_f32, true)];
    let bins = reliability_bins(&data, 10).unwrap();
    assert_eq!(bins[9].count, 1);
    assert_eq!(bins[9].accuracy, 1.0);
}

#[test]
fn bins_avg_confidence_and_accuracy_correct() {
    // Two predictions in the [0.8,0.9) bin: confidences 0.8 and 0.84, one correct.
    let data = [(0.8_f32, true), (0.84, false)];
    let bins = reliability_bins(&data, 10).unwrap();
    let bin = &bins[8]; // [0.8, 0.9)
    assert_eq!(bin.count, 2);
    assert!((bin.avg_confidence - 0.82).abs() < EPS);
    assert!((bin.accuracy - 0.5).abs() < EPS);
}

#[test]
fn bins_empty_bin_has_zero_fields() {
    let data = [(0.95_f32, true)];
    let bins = reliability_bins(&data, 10).unwrap();
    // Bin 0 ([0,0.1)) is empty.
    assert_eq!(bins[0].count, 0);
    assert_eq!(bins[0].avg_confidence, 0.0);
    assert_eq!(bins[0].accuracy, 0.0);
    assert_eq!(bins[0].gap(), 0.0);
}

#[test]
fn bins_gap_matches_abs_difference() {
    let data = [(0.9_f32, true), (0.9, false), (0.9, false), (0.9, false)];
    let bins = reliability_bins(&data, 10).unwrap();
    let bin = &bins[9];
    // conf 0.9, acc 0.25 → gap 0.65.
    assert!((bin.gap() - 0.65).abs() < EPS);
}

#[test]
fn bins_num_bins_clamped_to_one() {
    let data = [(0.3_f32, true), (0.7, false)];
    let bins = reliability_bins(&data, 0).unwrap();
    assert_eq!(bins.len(), 1);
    assert_eq!(bins[0].count, 2);
    assert!((bins[0].lower - 0.0).abs() < EPS);
    assert!((bins[0].upper - 1.0).abs() < EPS);
}

#[test]
fn bins_empty_errors() {
    let data: [(f32, bool); 0] = [];
    assert!(matches!(
        reliability_bins(&data, 10),
        Err(AnswerCalibrationError::EmptyData)
    ));
}

#[test]
fn bins_custom_count() {
    let data = [(0.25_f32, true), (0.75, false)];
    let bins = reliability_bins(&data, 4).unwrap();
    assert_eq!(bins.len(), 4);
    assert_eq!(bins[1].count, 1); // [0.25, 0.5)
    assert_eq!(bins[3].count, 1); // [0.75, 1.0]
}

// ── compute_metrics ───────────────────────────────────────────────────────────

#[test]
fn metrics_combines_all() {
    let data = [(0.9_f32, true), (0.8, true), (0.2, false), (0.1, false)];
    let m = compute_metrics(&data, 10).unwrap();
    assert!((0.0..=1.0).contains(&m.ece));
    assert!((0.0..=1.0).contains(&m.mce));
    assert!((0.0..=1.0).contains(&m.brier));
    assert_eq!(m.bins.len(), 10);
}

#[test]
fn metrics_total_count_helper() {
    let data = [(0.3_f32, true), (0.6, false), (0.9, true)];
    let m = compute_metrics(&data, 10).unwrap();
    assert_eq!(m.total_count(), 3);
}

#[test]
fn metrics_mce_ge_ece() {
    let data = [(1.0_f32, false), (1.0, false), (0.6, true), (0.4, true)];
    let m = compute_metrics(&data, 10).unwrap();
    assert!(m.mce >= m.ece - EPS);
}

#[test]
fn metrics_brier_matches_free_fn() {
    let data = [(0.7_f32, true), (0.3, false), (0.5, true)];
    let m = compute_metrics(&data, 10).unwrap();
    let b = brier_score(&data).unwrap();
    assert!((m.brier - b).abs() < EPS);
}

#[test]
fn metrics_ece_matches_free_fn() {
    let data = [(0.7_f32, true), (0.3, false), (0.5, true), (0.9, false)];
    let m = compute_metrics(&data, 8).unwrap();
    let ece = expected_calibration_error(&data, 8).unwrap();
    assert!((m.ece - ece).abs() < EPS);
}

#[test]
fn metrics_well_calibrated_flag() {
    let perfect = [(1.0_f32, true), (0.0, false), (0.5, true), (0.5, false)];
    let m = compute_metrics(&perfect, 10).unwrap();
    assert!(m.is_well_calibrated());

    let overconfident = [(1.0_f32, false), (1.0, false), (1.0, true), (1.0, false)];
    let m2 = compute_metrics(&overconfident, 10).unwrap();
    assert!(!m2.is_well_calibrated());
}

#[test]
fn metrics_empty_errors() {
    let data: [(f32, bool); 0] = [];
    assert!(matches!(
        compute_metrics(&data, 10),
        Err(AnswerCalibrationError::EmptyData)
    ));
}

// ── calibrate & fit_platt ─────────────────────────────────────────────────────

#[test]
fn calibrate_default_is_sigmoid() {
    let c = calibrator();
    // Default a=1, b=0 → calibrate(r) = sigmoid(r).
    let got = c.calibrate(0.0);
    assert!((got - 0.5).abs() < EPS);
}

#[test]
fn calibrate_monotonic_in_raw_default() {
    let c = calibrator();
    let lo = c.calibrate(0.2);
    let hi = c.calibrate(0.8);
    assert!(hi > lo);
}

#[test]
fn calibrate_clamps_input() {
    let c = calibrator();
    let a = c.calibrate(-1.0);
    let b = c.calibrate(0.0);
    assert!((a - b).abs() < EPS);
    let x = c.calibrate(2.0);
    let y = c.calibrate(1.0);
    assert!((x - y).abs() < EPS);
}

#[test]
fn fit_platt_empty_errors() {
    let mut c = calibrator();
    let data: [(f32, bool); 0] = [];
    assert!(matches!(
        c.fit_platt(&data),
        Err(AnswerCalibrationError::EmptyData)
    ));
}

#[test]
fn fit_platt_produces_monotonic_mapping() {
    let mut c = calibrator();
    // Higher raw confidence strongly co-occurs with correctness.
    let mut data = Vec::new();
    for i in 0..50 {
        let raw = i as f32 / 50.0;
        let correct = raw > 0.5;
        data.push((raw, correct));
    }
    c.fit_platt(&data).unwrap();
    // After fitting, higher raw → higher calibrated.
    let lo = c.calibrate(0.2);
    let mid = c.calibrate(0.5);
    let hi = c.calibrate(0.8);
    assert!(lo < mid);
    assert!(mid < hi);
    // Slope should be positive.
    assert!(c.platt_a > 0.0);
}

#[test]
fn fit_platt_lowers_overconfident() {
    let mut c = calibrator();
    // High raw confidence but only half correct → calibration should pull down.
    let data = [
        (0.9_f32, false),
        (0.9, true),
        (0.9, false),
        (0.9, true),
        (0.9, false),
        (0.9, true),
    ];
    c.fit_platt(&data).unwrap();
    let calibrated = c.calibrate(0.9);
    // 50% base rate at this confidence → calibrated should be near 0.5, below 0.9.
    assert!(calibrated < 0.9);
    assert!((0.0..=1.0).contains(&calibrated));
}

#[test]
fn fit_platt_deterministic() {
    let data = [
        (0.9_f32, true),
        (0.8, false),
        (0.3, false),
        (0.6, true),
        (0.1, false),
    ];
    let mut a = calibrator();
    let mut b = calibrator();
    a.fit_platt(&data).unwrap();
    b.fit_platt(&data).unwrap();
    assert_eq!(a.platt_a, b.platt_a);
    assert_eq!(a.platt_b, b.platt_b);
}

#[test]
fn fit_platt_output_in_unit_range() {
    let mut c = calibrator();
    let data = [(0.95_f32, true), (0.05, false), (0.7, true), (0.4, false)];
    c.fit_platt(&data).unwrap();
    for raw in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let v = c.calibrate(raw);
        assert!((0.0..=1.0).contains(&v));
    }
}

// ── Determinism of estimation & metrics ───────────────────────────────────────

#[test]
fn estimate_confidence_deterministic() {
    let c = calibrator();
    let signals = ConfidenceSignals::new(0.42, 0.73).with_verbalized(0.55);
    let a = c.estimate_confidence(&signals);
    let b = c.estimate_confidence(&signals);
    assert_eq!(a, b);
}

#[test]
fn compute_metrics_deterministic() {
    let data = [
        (0.9_f32, true),
        (0.7, false),
        (0.3, true),
        (0.5, false),
        (0.95, true),
    ];
    let m1 = compute_metrics(&data, 10).unwrap();
    let m2 = compute_metrics(&data, 10).unwrap();
    assert_eq!(m1.ece, m2.ece);
    assert_eq!(m1.mce, m2.mce);
    assert_eq!(m1.brier, m2.brier);
    assert_eq!(m1.bins.len(), m2.bins.len());
}

#[test]
fn extract_verbalized_deterministic() {
    let c = calibrator();
    let a = c.extract_verbalized("I am 73% sure");
    let b = c.extract_verbalized("I am 73% sure");
    assert_eq!(a, b);
}

// ── Default impls & smoke ─────────────────────────────────────────────────────

#[test]
fn calibrator_default_params() {
    let c = AnswerCalibrator::default();
    assert_eq!(c.platt_a, 1.0);
    assert_eq!(c.platt_b, 0.0);
    assert_eq!(c.config.num_bins, 10);
}

#[test]
fn reliability_bin_gap_empty_zero() {
    let bin = ReliabilityBin {
        lower: 0.0,
        upper: 0.1,
        count: 0,
        avg_confidence: 0.0,
        accuracy: 0.0,
    };
    assert_eq!(bin.gap(), 0.0);
}

#[test]
fn metrics_struct_fields_accessible() {
    let m = CalibrationMetrics {
        ece: 0.05,
        mce: 0.1,
        brier: 0.2,
        bins: Vec::new(),
    };
    assert!(m.is_well_calibrated());
    assert_eq!(m.total_count(), 0);
}
