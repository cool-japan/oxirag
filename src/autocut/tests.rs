//! Tests for the `autocut` module.
#![allow(clippy::float_cmp)]

use crate::autocut::cutter::AutoCutter;
use crate::autocut::types::{AutoCutConfig, AutoCutError, AutoCutStrategy};
use crate::types::{Document, DocumentId, SearchResult};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_result(id: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(format!("content for {id}")).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

/// A sequence with one clear discontinuity between index 2 (0.84) and 3 (0.30).
fn one_gap_scores() -> Vec<f32> {
    vec![0.9, 0.85, 0.84, 0.30, 0.28]
}

// ── AutoCutStrategy ───────────────────────────────────────────────────────────

#[test]
fn test_strategy_default_is_jumps_one() {
    assert_eq!(
        AutoCutStrategy::default(),
        AutoCutStrategy::Jumps(1),
        "default strategy should be Jumps(1)"
    );
}

#[test]
fn test_strategy_as_str_jumps() {
    assert_eq!(
        AutoCutStrategy::Jumps(2).as_str(),
        "jumps",
        "Jumps.as_str() should be 'jumps'"
    );
}

#[test]
fn test_strategy_as_str_relative_threshold() {
    assert_eq!(
        AutoCutStrategy::RelativeThreshold(0.5).as_str(),
        "relative_threshold",
        "RelativeThreshold.as_str() should be 'relative_threshold'"
    );
}

#[test]
fn test_strategy_as_str_std_dev() {
    assert_eq!(
        AutoCutStrategy::StdDev(1.0).as_str(),
        "std_dev",
        "StdDev.as_str() should be 'std_dev'"
    );
}

#[test]
fn test_strategy_as_str_knee() {
    assert_eq!(
        AutoCutStrategy::Knee.as_str(),
        "knee",
        "Knee.as_str() should be 'knee'"
    );
}

// ── AutoCutConfig defaults ─────────────────────────────────────────────────────

#[test]
fn test_config_default_strategy_is_jumps_one() {
    assert_eq!(
        AutoCutConfig::default().strategy,
        AutoCutStrategy::Jumps(1),
        "default config strategy should be Jumps(1)"
    );
}

#[test]
fn test_config_default_sensitivity_is_one() {
    assert_eq!(
        AutoCutConfig::default().sensitivity,
        1.0,
        "default sensitivity should be 1.0"
    );
}

#[test]
fn test_config_default_min_keep_is_one() {
    assert_eq!(
        AutoCutConfig::default().min_keep,
        1,
        "default min_keep should be 1"
    );
}

#[test]
fn test_config_default_max_keep_is_zero() {
    assert_eq!(
        AutoCutConfig::default().max_keep,
        0,
        "default max_keep should be 0 (no upper bound)"
    );
}

#[test]
fn test_config_new_equals_default_strategy() {
    assert_eq!(
        AutoCutConfig::new().strategy,
        AutoCutConfig::default().strategy,
        "new() should match default() strategy"
    );
}

// ── AutoCutConfig builders ─────────────────────────────────────────────────────

#[test]
fn test_config_with_strategy_builder() {
    let cfg = AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee);
    assert_eq!(
        cfg.strategy,
        AutoCutStrategy::Knee,
        "with_strategy should set Knee"
    );
}

#[test]
fn test_config_with_sensitivity_builder() {
    let cfg = AutoCutConfig::new().with_sensitivity(2.5);
    assert_eq!(cfg.sensitivity, 2.5, "with_sensitivity should set 2.5");
}

#[test]
fn test_config_with_min_keep_builder() {
    let cfg = AutoCutConfig::new().with_min_keep(3);
    assert_eq!(cfg.min_keep, 3, "with_min_keep should set 3");
}

#[test]
fn test_config_with_max_keep_builder() {
    let cfg = AutoCutConfig::new().with_max_keep(7);
    assert_eq!(cfg.max_keep, 7, "with_max_keep should set 7");
}

#[test]
fn test_config_builder_chaining() {
    let cfg = AutoCutConfig::new()
        .with_strategy(AutoCutStrategy::StdDev(1.0))
        .with_sensitivity(1.5)
        .with_min_keep(2)
        .with_max_keep(9);
    assert_eq!(cfg.max_keep, 9, "chained max_keep should be 9");
}

// ── keep_count: Jumps ──────────────────────────────────────────────────────────

#[test]
fn test_keep_count_jumps_one_clear_gap() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Jumps(1)));
    // Big gap is between index 2 (0.84) and 3 (0.30) → keep 3.
    assert_eq!(
        cutter.keep_count(&one_gap_scores()),
        3,
        "Jumps(1) should keep the 3 results before the single big gap"
    );
}

#[test]
fn test_keep_count_jumps_two_cuts_at_second_jump() {
    // Two clear gaps: between 0.80/0.50 (idx1) and 0.49/0.20 (idx3).
    let scores = vec![0.81, 0.80, 0.50, 0.49, 0.20, 0.19];
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Jumps(2)));
    assert_eq!(
        cutter.keep_count(&scores),
        4,
        "Jumps(2) should cut after the 2nd significant jump (keep 4)"
    );
}

#[test]
fn test_keep_count_jumps_sorts_unsorted_input() {
    // Same multiset as one_gap_scores() but shuffled.
    let scores = vec![0.28, 0.84, 0.30, 0.9, 0.85];
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Jumps(1)));
    assert_eq!(
        cutter.keep_count(&scores),
        3,
        "keep_count should sort descending first, yielding 3"
    );
}

#[test]
fn test_keep_count_jumps_sensitivity_high_keeps_more() {
    // With a very high sensitivity no gap exceeds the (scaled) mean, so keep all.
    let cutter = AutoCutter::new(
        AutoCutConfig::new()
            .with_strategy(AutoCutStrategy::Jumps(1))
            .with_sensitivity(100.0),
    );
    assert_eq!(
        cutter.keep_count(&one_gap_scores()),
        5,
        "very high sensitivity should detect no jump and keep all 5"
    );
}

#[test]
fn test_keep_count_jumps_zero_keeps_all() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Jumps(0)));
    assert_eq!(
        cutter.keep_count(&one_gap_scores()),
        5,
        "Jumps(0) means cut after the 0th jump → keep all"
    );
}

// ── keep_count: RelativeThreshold ──────────────────────────────────────────────

#[test]
fn test_keep_count_relative_threshold_keeps_above_ratio() {
    // top = 0.9; r = 0.9 → cutoff 0.81 → keep 0.9, 0.85, 0.84 = 3.
    let cutter = AutoCutter::new(
        AutoCutConfig::new().with_strategy(AutoCutStrategy::RelativeThreshold(0.9)),
    );
    assert_eq!(
        cutter.keep_count(&one_gap_scores()),
        3,
        "RelativeThreshold(0.9) should keep scores >= 0.81"
    );
}

#[test]
fn test_keep_count_relative_threshold_strict_keeps_top_only() {
    // r = 1.0 → cutoff equals top → only scores == top survive (one item).
    let scores = vec![0.9, 0.7, 0.5];
    let cutter = AutoCutter::new(
        AutoCutConfig::new().with_strategy(AutoCutStrategy::RelativeThreshold(1.0)),
    );
    assert_eq!(
        cutter.keep_count(&scores),
        1,
        "RelativeThreshold(1.0) should keep only the top-scoring result"
    );
}

#[test]
fn test_keep_count_relative_threshold_zero_keeps_all() {
    let cutter = AutoCutter::new(
        AutoCutConfig::new().with_strategy(AutoCutStrategy::RelativeThreshold(0.0)),
    );
    assert_eq!(
        cutter.keep_count(&one_gap_scores()),
        5,
        "RelativeThreshold(0.0) should keep all results"
    );
}

// ── keep_count: StdDev ─────────────────────────────────────────────────────────

#[test]
fn test_keep_count_std_dev_cuts_at_outlier_gap() {
    // One gap is a clear outlier; StdDev(1.0) should cut right after it.
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::StdDev(1.0)));
    assert_eq!(
        cutter.keep_count(&one_gap_scores()),
        3,
        "StdDev(1.0) should cut at the outlier gap (keep 3)"
    );
}

#[test]
fn test_keep_count_std_dev_high_s_keeps_all() {
    // A huge multiplier means no gap exceeds mean + s*std → keep all.
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::StdDev(50.0)));
    assert_eq!(
        cutter.keep_count(&one_gap_scores()),
        5,
        "StdDev with very high s should keep all results"
    );
}

#[test]
fn test_keep_count_std_dev_uniform_gaps_keeps_all() {
    // Equal gaps → zero std → no gap strictly exceeds the mean → keep all.
    let scores = vec![1.0, 0.75, 0.5, 0.25, 0.0];
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::StdDev(1.0)));
    assert_eq!(
        cutter.keep_count(&scores),
        5,
        "uniform gaps yield zero variance; StdDev should keep all"
    );
}

// ── keep_count: Knee ───────────────────────────────────────────────────────────

#[test]
fn test_keep_count_knee_cuts_at_largest_gap() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    assert_eq!(
        cutter.keep_count(&one_gap_scores()),
        3,
        "Knee should keep up to (and including) the item before the biggest gap"
    );
}

#[test]
fn test_keep_count_knee_picks_first_of_tied_largest_gaps() {
    // Exact (binary-representable) ties: gaps are all 0.5; Knee keeps the first
    // (earliest) discontinuity, so the cut lands after the very first item.
    let scores = vec![1.5, 1.0, 0.5, 0.0];
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    assert_eq!(
        cutter.keep_count(&scores),
        1,
        "Knee should cut at the first of equal largest gaps (keep 1)"
    );
}

// ── clamp: min_keep / max_keep ─────────────────────────────────────────────────

#[test]
fn test_min_keep_floor_enforced() {
    // RelativeThreshold(1.0) would keep 1, but min_keep=3 raises it.
    let scores = vec![0.9, 0.8, 0.7, 0.6];
    let cutter = AutoCutter::new(
        AutoCutConfig::new()
            .with_strategy(AutoCutStrategy::RelativeThreshold(1.0))
            .with_min_keep(3),
    );
    assert_eq!(
        cutter.keep_count(&scores),
        3,
        "min_keep=3 should raise a would-be keep of 1 up to 3"
    );
}

#[test]
fn test_min_keep_capped_by_available_count() {
    // min_keep larger than the input must not exceed the available count.
    let scores = vec![0.9, 0.5];
    let cutter = AutoCutter::new(
        AutoCutConfig::new()
            .with_strategy(AutoCutStrategy::Knee)
            .with_min_keep(10),
    );
    assert_eq!(
        cutter.keep_count(&scores),
        2,
        "min_keep beyond input size should clamp to the available count"
    );
}

#[test]
fn test_max_keep_ceiling_enforced() {
    // No jumps → Jumps would keep all 5, but max_keep=2 caps it.
    let scores = vec![1.0, 0.75, 0.5, 0.25, 0.0];
    let cutter = AutoCutter::new(
        AutoCutConfig::new()
            .with_strategy(AutoCutStrategy::Jumps(1))
            .with_max_keep(2),
    );
    assert_eq!(
        cutter.keep_count(&scores),
        2,
        "max_keep=2 should cap a would-be keep of 5 down to 2"
    );
}

#[test]
fn test_max_keep_zero_means_no_upper_bound() {
    let scores = vec![1.0, 0.75, 0.5, 0.25, 0.0];
    let cutter = AutoCutter::new(
        AutoCutConfig::new()
            .with_strategy(AutoCutStrategy::Jumps(1))
            .with_max_keep(0),
    );
    assert_eq!(
        cutter.keep_count(&scores),
        5,
        "max_keep=0 should impose no upper bound (keep all 5)"
    );
}

// ── edge cases ─────────────────────────────────────────────────────────────────

#[test]
fn test_keep_count_empty_is_zero() {
    let cutter = AutoCutter::default();
    assert_eq!(
        cutter.keep_count(&[]),
        0,
        "empty input should yield keep_count 0"
    );
}

#[test]
fn test_keep_count_single_keeps_one() {
    let cutter = AutoCutter::default();
    assert_eq!(
        cutter.keep_count(&[0.42]),
        1,
        "single score should keep exactly 1"
    );
}

#[test]
fn test_keep_count_all_equal_keeps_all() {
    let scores = vec![0.5, 0.5, 0.5, 0.5];
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    assert_eq!(
        cutter.keep_count(&scores),
        4,
        "all-equal scores have no jumps; Knee should keep all"
    );
}

#[test]
fn test_keep_count_all_equal_jumps_keeps_all() {
    let scores = vec![0.7, 0.7, 0.7];
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Jumps(1)));
    assert_eq!(
        cutter.keep_count(&scores),
        3,
        "all-equal scores have no significant jump; Jumps should keep all"
    );
}

#[test]
fn test_keep_count_all_equal_clamped_to_max_keep() {
    let scores = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    let cutter = AutoCutter::new(
        AutoCutConfig::new()
            .with_strategy(AutoCutStrategy::Jumps(1))
            .with_max_keep(2),
    );
    assert_eq!(
        cutter.keep_count(&scores),
        2,
        "all-equal scores keep all but should still respect max_keep=2"
    );
}

// ── cut() ──────────────────────────────────────────────────────────────────────

#[test]
fn test_cut_empty_yields_empty() {
    let cutter = AutoCutter::default();
    let out = cutter.cut(&[]);
    assert!(out.is_empty(), "cutting an empty slice should yield empty");
}

#[test]
fn test_cut_keeps_expected_count() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    let results: Vec<SearchResult> = one_gap_scores()
        .iter()
        .enumerate()
        .map(|(i, &s)| make_result(&format!("d{i}"), s))
        .collect();
    let out = cutter.cut(&results);
    assert_eq!(
        out.len(),
        3,
        "cut() should keep 3 results for the one-gap set"
    );
}

#[test]
fn test_cut_sorts_descending_before_truncating() {
    // Input is unsorted; the top kept item must be the highest score.
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    let results = vec![
        make_result("mid", 0.84),
        make_result("low", 0.30),
        make_result("top", 0.9),
        make_result("second", 0.85),
        make_result("lowest", 0.28),
    ];
    let out = cutter.cut(&results);
    assert_eq!(
        out[0].document.id.as_str(),
        "top",
        "cut() should sort descending so the top result is first"
    );
}

#[test]
fn test_cut_reranks_zero_indexed_consecutive() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Jumps(1)));
    let results: Vec<SearchResult> = one_gap_scores()
        .iter()
        .enumerate()
        .map(|(i, &s)| make_result(&format!("d{i}"), s))
        .collect();
    let out = cutter.cut(&results);
    let ranks: Vec<usize> = out.iter().map(|r| r.rank).collect();
    assert_eq!(
        ranks,
        vec![0, 1, 2],
        "cut() should re-rank kept items as 0..n"
    );
}

#[test]
fn test_cut_preserves_document_contents() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    let results = vec![
        make_result("alpha", 0.9),
        make_result("beta", 0.85),
        make_result("gamma", 0.84),
        make_result("delta", 0.30),
    ];
    let out = cutter.cut(&results);
    assert_eq!(
        out[0].document.content, "content for alpha",
        "cut() should preserve the original SearchResult document contents"
    );
}

#[test]
fn test_cut_single_result_kept() {
    let cutter = AutoCutter::default();
    let out = cutter.cut(&[make_result("only", 0.7)]);
    assert_eq!(
        out.len(),
        1,
        "cutting a single result should keep it (min_keep=1)"
    );
}

#[test]
fn test_cut_all_equal_keeps_all() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Jumps(1)));
    let results = vec![
        make_result("a", 0.5),
        make_result("b", 0.5),
        make_result("c", 0.5),
    ];
    let out = cutter.cut(&results);
    assert_eq!(out.len(), 3, "all-equal scores should keep every result");
}

// ── cut_with_report() ──────────────────────────────────────────────────────────

#[test]
fn test_report_kept_matches_output_len() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    let results: Vec<SearchResult> = one_gap_scores()
        .iter()
        .enumerate()
        .map(|(i, &s)| make_result(&format!("d{i}"), s))
        .collect();
    let (out, report) = cutter.cut_with_report(&results);
    assert_eq!(
        report.kept,
        out.len(),
        "report.kept should equal the kept output length"
    );
}

#[test]
fn test_report_total_is_input_len() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    let results: Vec<SearchResult> = one_gap_scores()
        .iter()
        .enumerate()
        .map(|(i, &s)| make_result(&format!("d{i}"), s))
        .collect();
    let (_, report) = cutter.cut_with_report(&results);
    assert_eq!(
        report.total, 5,
        "report.total should equal the input length (5)"
    );
}

#[test]
fn test_report_cut_score_is_first_dropped() {
    // Knee keeps 3; the first dropped score is 0.30.
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    let results: Vec<SearchResult> = one_gap_scores()
        .iter()
        .enumerate()
        .map(|(i, &s)| make_result(&format!("d{i}"), s))
        .collect();
    let (_, report) = cutter.cut_with_report(&results);
    assert_eq!(
        report.cut_score,
        Some(0.30),
        "report.cut_score should be the score of the first dropped result"
    );
}

#[test]
fn test_report_cut_score_none_when_nothing_dropped() {
    let cutter = AutoCutter::new(
        AutoCutConfig::new().with_strategy(AutoCutStrategy::RelativeThreshold(0.0)),
    );
    let results = vec![make_result("a", 0.9), make_result("b", 0.8)];
    let (_, report) = cutter.cut_with_report(&results);
    assert_eq!(
        report.cut_score, None,
        "report.cut_score should be None when no results are dropped"
    );
}

#[test]
fn test_report_largest_gap_value() {
    // Largest gap in one_gap_scores() is 0.84 - 0.30 = 0.54.
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    let results: Vec<SearchResult> = one_gap_scores()
        .iter()
        .enumerate()
        .map(|(i, &s)| make_result(&format!("d{i}"), s))
        .collect();
    let (_, report) = cutter.cut_with_report(&results);
    assert!(
        (report.largest_gap - 0.54).abs() < 1e-6,
        "report.largest_gap should be ~0.54, got {}",
        report.largest_gap
    );
}

#[test]
fn test_report_empty_input_fields() {
    let cutter = AutoCutter::default();
    let (_, report) = cutter.cut_with_report(&[]);
    assert_eq!(report.kept, 0, "empty input report should have kept == 0");
}

// ── validate / try_new ─────────────────────────────────────────────────────────

#[test]
fn test_validate_ok_for_valid_range() {
    let cfg = AutoCutConfig::new().with_min_keep(2).with_max_keep(5);
    assert!(
        cfg.validate().is_ok(),
        "min_keep <= max_keep should validate OK"
    );
}

#[test]
fn test_validate_ok_when_max_keep_zero() {
    // max_keep == 0 disables the upper bound, so any min_keep is valid.
    let cfg = AutoCutConfig::new().with_min_keep(100).with_max_keep(0);
    assert!(
        cfg.validate().is_ok(),
        "max_keep=0 should make any min_keep valid"
    );
}

#[test]
fn test_try_new_invalid_keep_range_errors() {
    let cfg = AutoCutConfig::new().with_min_keep(5).with_max_keep(3);
    let err = AutoCutter::try_new(cfg).unwrap_err();
    assert!(
        matches!(err, AutoCutError::InvalidKeepRange { min: 5, max: 3 }),
        "try_new should return InvalidKeepRange for min > max"
    );
}

#[test]
fn test_try_new_ok_for_valid_config() {
    let cfg = AutoCutConfig::new().with_min_keep(1).with_max_keep(4);
    assert!(
        AutoCutter::try_new(cfg).is_ok(),
        "try_new should succeed for a valid keep range"
    );
}

#[test]
fn test_error_display_message_descriptive() {
    let err = AutoCutError::InvalidKeepRange { min: 7, max: 2 };
    let msg = err.to_string();
    assert!(
        msg.contains('7') && msg.contains('2'),
        "InvalidKeepRange display should mention both bounds, got: {msg}"
    );
}

// ── determinism ────────────────────────────────────────────────────────────────

#[test]
fn test_keep_count_deterministic() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Jumps(1)));
    let scores = one_gap_scores();
    let a = cutter.keep_count(&scores);
    let b = cutter.keep_count(&scores);
    assert_eq!(a, b, "keep_count should be deterministic across calls");
}

#[test]
fn test_cut_deterministic_ordering() {
    let cutter = AutoCutter::new(AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee));
    let results = vec![
        make_result("a", 0.9),
        make_result("b", 0.85),
        make_result("c", 0.84),
        make_result("d", 0.30),
    ];
    let first: Vec<String> = cutter
        .cut(&results)
        .iter()
        .map(|r| r.document.id.as_str().to_string())
        .collect();
    let second: Vec<String> = cutter
        .cut(&results)
        .iter()
        .map(|r| r.document.id.as_str().to_string())
        .collect();
    assert_eq!(
        first, second,
        "cut() should produce a deterministic ordering"
    );
}

// ── AutoCutter construction ────────────────────────────────────────────────────

#[test]
fn test_cutter_new_stores_config() {
    let cfg = AutoCutConfig::new().with_strategy(AutoCutStrategy::Knee);
    let cutter = AutoCutter::new(cfg);
    assert_eq!(
        cutter.config.strategy,
        AutoCutStrategy::Knee,
        "new() should store the provided config"
    );
}

#[test]
fn test_cutter_default_uses_default_config() {
    let cutter = AutoCutter::default();
    assert_eq!(
        cutter.config.strategy,
        AutoCutStrategy::Jumps(1),
        "default cutter should use the default Jumps(1) strategy"
    );
}
