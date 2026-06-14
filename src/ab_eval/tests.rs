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
    clippy::too_many_lines
)]

use crate::ab_eval::evaluator::AbEvaluator;
use crate::ab_eval::types::{AbConfig, AbError, AbWinner};

// ── Config: defaults & builders ───────────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = AbConfig::default();
    assert_eq!(cfg.bootstrap_samples, 1000);
    assert_eq!(cfg.tie_margin, 0.0);
    assert_eq!(cfg.confidence, 0.95);
}

#[test]
fn config_new_matches_default() {
    let cfg = AbConfig::new();
    let def = AbConfig::default();
    assert_eq!(cfg.bootstrap_samples, def.bootstrap_samples);
    assert_eq!(cfg.tie_margin, def.tie_margin);
    assert_eq!(cfg.confidence, def.confidence);
}

#[test]
fn config_builder_bootstrap_samples() {
    let cfg = AbConfig::new().with_bootstrap_samples(250);
    assert_eq!(cfg.bootstrap_samples, 250);
}

#[test]
fn config_builder_tie_margin() {
    let cfg = AbConfig::new().with_tie_margin(0.05);
    assert_eq!(cfg.tie_margin, 0.05);
}

#[test]
fn config_builder_confidence() {
    let cfg = AbConfig::new().with_confidence(0.9);
    assert_eq!(cfg.confidence, 0.9);
}

#[test]
fn config_builder_chained() {
    let cfg = AbConfig::new()
        .with_bootstrap_samples(500)
        .with_tie_margin(0.1)
        .with_confidence(0.99);
    assert_eq!(cfg.bootstrap_samples, 500);
    assert_eq!(cfg.tie_margin, 0.1);
    assert_eq!(cfg.confidence, 0.99);
}

#[test]
fn config_clone_independent() {
    let cfg = AbConfig::new().with_confidence(0.8);
    let cloned = cfg.clone();
    assert_eq!(cloned.confidence, 0.8);
}

// ── Evaluator construction ────────────────────────────────────────────────────

#[test]
fn evaluator_new_holds_config() {
    let cfg = AbConfig::new().with_bootstrap_samples(123);
    let ev = AbEvaluator::new(cfg);
    assert_eq!(ev.config.bootstrap_samples, 123);
}

#[test]
fn evaluator_default_uses_default_config() {
    let ev = AbEvaluator::default();
    assert_eq!(ev.config.bootstrap_samples, 1000);
    assert_eq!(ev.config.confidence, 0.95);
}

// ── A clearly better ──────────────────────────────────────────────────────────

#[test]
fn a_clearly_better_mean_diff_one() {
    let a = vec![1.0f32; 8];
    let b = vec![0.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!((r.mean_diff - 1.0).abs() < 1e-6);
}

#[test]
fn a_clearly_better_all_wins() {
    let a = vec![1.0f32; 8];
    let b = vec![0.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.wins_a, 8);
    assert_eq!(r.wins_b, 0);
    assert_eq!(r.ties, 0);
}

#[test]
fn a_clearly_better_winner_a() {
    let a = vec![1.0f32; 8];
    let b = vec![0.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.winner, Some(AbWinner::A));
}

#[test]
fn a_clearly_better_ci_low_positive() {
    let a = vec![1.0f32; 8];
    let b = vec![0.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(r.ci_low > 0.0);
}

#[test]
fn a_clearly_better_significant() {
    let a = vec![1.0f32; 8];
    let b = vec![0.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(r.is_significant());
}

#[test]
fn a_clearly_better_means() {
    let a = vec![1.0f32; 8];
    let b = vec![0.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!((r.mean_a - 1.0).abs() < 1e-6);
    assert!((r.mean_b - 0.0).abs() < 1e-6);
}

// ── B clearly better ──────────────────────────────────────────────────────────

#[test]
fn b_clearly_better_winner_b() {
    let a = vec![0.0f32; 8];
    let b = vec![1.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.winner, Some(AbWinner::B));
}

#[test]
fn b_clearly_better_ci_high_negative() {
    let a = vec![0.0f32; 8];
    let b = vec![1.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(r.ci_high < 0.0);
}

#[test]
fn b_clearly_better_mean_diff_negative() {
    let a = vec![0.0f32; 8];
    let b = vec![1.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!((r.mean_diff + 1.0).abs() < 1e-6);
}

#[test]
fn b_clearly_better_all_wins_b() {
    let a = vec![0.0f32; 8];
    let b = vec![1.0f32; 8];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.wins_b, 8);
    assert_eq!(r.wins_a, 0);
}

// ── Equal scores ──────────────────────────────────────────────────────────────

#[test]
fn equal_scores_mean_diff_zero() {
    let a = vec![0.5f32; 10];
    let b = vec![0.5f32; 10];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!((r.mean_diff).abs() < 1e-6);
}

#[test]
fn equal_scores_winner_none() {
    let a = vec![0.5f32; 10];
    let b = vec![0.5f32; 10];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.winner, None);
}

#[test]
fn equal_scores_p_value_high() {
    let a = vec![0.5f32; 10];
    let b = vec![0.5f32; 10];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    // Every resample mean is exactly 0, so it is both <= 0 and >= 0:
    // p = 2 * min(1, 1) = 2 clamped to 1.
    assert!(r.p_value > 0.99);
}

#[test]
fn equal_scores_all_ties() {
    let a = vec![0.5f32; 10];
    let b = vec![0.5f32; 10];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.ties, 10);
    assert_eq!(r.wins_a, 0);
    assert_eq!(r.wins_b, 0);
}

#[test]
fn equal_scores_not_significant() {
    let a = vec![0.5f32; 10];
    let b = vec![0.5f32; 10];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(!r.is_significant());
}

#[test]
fn equal_scores_ci_brackets_zero() {
    let a = vec![0.5f32; 10];
    let b = vec![0.5f32; 10];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(r.ci_low <= 0.0);
    assert!(r.ci_high >= 0.0);
}

// ── Tie margin behaviour ──────────────────────────────────────────────────────

#[test]
fn tie_margin_classifies_near_equal_as_tie() {
    // Differences of +/- 0.05, all within a 0.1 margin.
    let a = vec![0.55f32, 0.45, 0.52, 0.48];
    let b = vec![0.50f32, 0.50, 0.50, 0.50];
    let ev = AbEvaluator::new(AbConfig::default().with_tie_margin(0.1));
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.ties, 4);
    assert_eq!(r.wins_a, 0);
    assert_eq!(r.wins_b, 0);
}

#[test]
fn tie_margin_zero_counts_small_diffs_as_wins() {
    let a = vec![0.55f32, 0.45, 0.52, 0.48];
    let b = vec![0.50f32, 0.50, 0.50, 0.50];
    let ev = AbEvaluator::new(AbConfig::default().with_tie_margin(0.0));
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.wins_a, 2);
    assert_eq!(r.wins_b, 2);
    assert_eq!(r.ties, 0);
}

#[test]
fn tie_margin_exact_boundary_is_tie() {
    // diff exactly equal to the margin is NOT a strict win, so it is a tie.
    // Use exactly representable f32 values so the difference is exactly 0.25.
    let a = vec![0.75f32];
    let b = vec![0.5f32];
    let ev = AbEvaluator::new(AbConfig::default().with_tie_margin(0.25));
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.ties, 1);
    assert_eq!(r.wins_a, 0);
}

#[test]
fn tie_margin_just_above_boundary_is_win() {
    // diff of 0.5 strictly exceeds a 0.25 margin (both exactly representable).
    let a = vec![1.0f32];
    let b = vec![0.5f32];
    let ev = AbEvaluator::new(AbConfig::default().with_tie_margin(0.25));
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.wins_a, 1);
    assert_eq!(r.ties, 0);
}

#[test]
fn tie_margin_mixed_classification() {
    // diffs: +0.2 (win A), -0.2 (win B), +0.01 (tie within 0.05)
    let a = vec![0.7f32, 0.3, 0.51];
    let b = vec![0.5f32, 0.5, 0.50];
    let ev = AbEvaluator::new(AbConfig::default().with_tie_margin(0.05));
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.wins_a, 1);
    assert_eq!(r.wins_b, 1);
    assert_eq!(r.ties, 1);
}

// ── Mean correctness ──────────────────────────────────────────────────────────

#[test]
fn means_computed_correctly() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.0f32, 1.0, 2.0, 1.0];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!((r.mean_a - 2.5).abs() < 1e-6);
    assert!((r.mean_b - 1.0).abs() < 1e-6);
}

#[test]
fn mean_diff_equals_mean_a_minus_mean_b() {
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!((r.mean_diff - (r.mean_a - r.mean_b)).abs() < 1e-5);
}

#[test]
fn total_equals_input_length() {
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6, 0.3];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5, 0.9];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.total(), 6);
}

// ── Interval & p-value invariants ─────────────────────────────────────────────

#[test]
fn ci_low_le_ci_high() {
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6, 0.3, 0.8, 0.55];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5, 0.9, 0.2, 0.45];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(r.ci_low <= r.ci_high);
}

#[test]
fn p_value_in_unit_range() {
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6, 0.3, 0.8, 0.55];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5, 0.9, 0.2, 0.45];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(r.p_value >= 0.0);
    assert!(r.p_value <= 1.0);
}

#[test]
fn ci_low_le_ci_high_many_inputs() {
    let ev = AbEvaluator::new(AbConfig::default());
    for seed in 0..20usize {
        let a: Vec<f32> = (0..12)
            .map(|i| ((i * 7 + seed) % 11) as f32 / 10.0)
            .collect();
        let b: Vec<f32> = (0..12)
            .map(|i| ((i * 5 + seed) % 11) as f32 / 10.0)
            .collect();
        let r = ev.compare(&a, &b).unwrap();
        assert!(r.ci_low <= r.ci_high, "seed {seed}");
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "seed {seed}");
    }
}

#[test]
fn p_value_significant_when_strongly_separated() {
    let a = vec![1.0f32; 12];
    let b = vec![0.0f32; 12];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    // No resample mean can be <= 0, so p collapses to ~0.
    assert!(r.p_value < 0.05);
}

#[test]
fn higher_confidence_widens_interval() {
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6, 0.3, 0.8, 0.55, 0.1, 0.9];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5, 0.9, 0.2, 0.45, 0.7, 0.2];
    let narrow = AbEvaluator::new(AbConfig::default().with_confidence(0.80));
    let wide = AbEvaluator::new(AbConfig::default().with_confidence(0.99));
    let rn = narrow.compare(&a, &b).unwrap();
    let rw = wide.compare(&a, &b).unwrap();
    let narrow_width = rn.ci_high - rn.ci_low;
    let wide_width = rw.ci_high - rw.ci_low;
    assert!(wide_width >= narrow_width);
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn length_mismatch_error() {
    let a = vec![0.5f32, 0.6, 0.7];
    let b = vec![0.5f32, 0.6];
    let ev = AbEvaluator::new(AbConfig::default());
    let err = ev.compare(&a, &b).unwrap_err();
    match err {
        AbError::LengthMismatch { a, b } => {
            assert_eq!(a, 3);
            assert_eq!(b, 2);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn length_mismatch_error_message() {
    let a = vec![0.5f32, 0.6, 0.7];
    let b = vec![0.5f32, 0.6];
    let ev = AbEvaluator::new(AbConfig::default());
    let err = ev.compare(&a, &b).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains('3'));
    assert!(msg.contains('2'));
    assert!(msg.contains("equal length"));
}

#[test]
fn empty_scores_error() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let ev = AbEvaluator::new(AbConfig::default());
    let err = ev.compare(&a, &b).unwrap_err();
    assert!(matches!(err, AbError::EmptyScores));
}

#[test]
fn empty_scores_error_message() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let ev = AbEvaluator::new(AbConfig::default());
    let err = ev.compare(&a, &b).unwrap_err();
    assert!(err.to_string().contains("empty"));
}

#[test]
fn length_mismatch_takes_precedence_over_empty() {
    // One empty, one not: the length check fires first.
    let a: Vec<f32> = vec![];
    let b = vec![0.5f32];
    let ev = AbEvaluator::new(AbConfig::default());
    let err = ev.compare(&a, &b).unwrap_err();
    assert!(matches!(err, AbError::LengthMismatch { a: 0, b: 1 }));
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn determinism_identical_results_twice() {
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6, 0.3, 0.8, 0.55, 0.1, 0.95];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5, 0.9, 0.2, 0.45, 0.7, 0.20];
    let ev = AbEvaluator::new(AbConfig::default());
    let r1 = ev.compare(&a, &b).unwrap();
    let r2 = ev.compare(&a, &b).unwrap();
    assert_eq!(r1.mean_a.to_bits(), r2.mean_a.to_bits());
    assert_eq!(r1.mean_b.to_bits(), r2.mean_b.to_bits());
    assert_eq!(r1.mean_diff.to_bits(), r2.mean_diff.to_bits());
    assert_eq!(r1.ci_low.to_bits(), r2.ci_low.to_bits());
    assert_eq!(r1.ci_high.to_bits(), r2.ci_high.to_bits());
    assert_eq!(r1.p_value.to_bits(), r2.p_value.to_bits());
    assert_eq!(r1.wins_a, r2.wins_a);
    assert_eq!(r1.wins_b, r2.wins_b);
    assert_eq!(r1.ties, r2.ties);
    assert_eq!(r1.winner, r2.winner);
}

#[test]
fn determinism_separate_evaluators() {
    let a = vec![0.3f32, 0.7, 0.5, 0.9, 0.1, 0.6, 0.4];
    let b = vec![0.6f32, 0.2, 0.5, 0.1, 0.8, 0.3, 0.7];
    let ev1 = AbEvaluator::new(AbConfig::default());
    let ev2 = AbEvaluator::new(AbConfig::default());
    let r1 = ev1.compare(&a, &b).unwrap();
    let r2 = ev2.compare(&a, &b).unwrap();
    assert_eq!(r1.ci_low.to_bits(), r2.ci_low.to_bits());
    assert_eq!(r1.ci_high.to_bits(), r2.ci_high.to_bits());
    assert_eq!(r1.p_value.to_bits(), r2.p_value.to_bits());
}

#[test]
fn determinism_across_sample_counts_changes_result() {
    // Sanity: different bootstrap sizes generally produce different intervals,
    // confirming the bootstrap actually depends on the sample loop.
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6, 0.3, 0.8, 0.55];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5, 0.9, 0.2, 0.45];
    let ev_small = AbEvaluator::new(AbConfig::default().with_bootstrap_samples(50));
    let ev_large = AbEvaluator::new(AbConfig::default().with_bootstrap_samples(2000));
    let rs = ev_small.compare(&a, &b).unwrap();
    let rl = ev_large.compare(&a, &b).unwrap();
    // Both remain valid intervals regardless.
    assert!(rs.ci_low <= rs.ci_high);
    assert!(rl.ci_low <= rl.ci_high);
}

#[test]
fn determinism_resample_repeatable_with_small_bootstrap() {
    let a = vec![0.8f32, 0.1, 0.6, 0.3, 0.9];
    let b = vec![0.2f32, 0.7, 0.4, 0.6, 0.1];
    let ev = AbEvaluator::new(AbConfig::default().with_bootstrap_samples(17));
    let r1 = ev.compare(&a, &b).unwrap();
    let r2 = ev.compare(&a, &b).unwrap();
    assert_eq!(r1.ci_low.to_bits(), r2.ci_low.to_bits());
    assert_eq!(r1.ci_high.to_bits(), r2.ci_high.to_bits());
}

// ── Single-element & small inputs ─────────────────────────────────────────────

#[test]
fn single_element_a_wins() {
    let a = vec![1.0f32];
    let b = vec![0.0f32];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.wins_a, 1);
    assert!((r.mean_diff - 1.0).abs() < 1e-6);
    // Every resample of a single difference is that difference.
    assert!((r.ci_low - 1.0).abs() < 1e-6);
    assert!((r.ci_high - 1.0).abs() < 1e-6);
    assert_eq!(r.winner, Some(AbWinner::A));
}

#[test]
fn single_element_equal_no_winner() {
    let a = vec![0.5f32];
    let b = vec![0.5f32];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.ties, 1);
    assert_eq!(r.winner, None);
}

#[test]
fn two_element_inputs_valid() {
    let a = vec![0.8f32, 0.6];
    let b = vec![0.2f32, 0.1];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.wins_a, 2);
    assert!(r.ci_low <= r.ci_high);
}

// ── Zero bootstrap fallback ───────────────────────────────────────────────────

#[test]
fn zero_bootstrap_falls_back_to_point_estimate() {
    let a = vec![0.9f32, 0.2, 0.7, 0.4];
    let b = vec![0.1f32, 0.8, 0.3, 0.5];
    let ev = AbEvaluator::new(AbConfig::default().with_bootstrap_samples(0));
    let r = ev.compare(&a, &b).unwrap();
    assert!((r.ci_low - r.mean_diff).abs() < 1e-6);
    assert!((r.ci_high - r.mean_diff).abs() < 1e-6);
    assert_eq!(r.p_value, 1.0);
}

#[test]
fn zero_bootstrap_winner_follows_mean_diff_sign() {
    // With ci_low == ci_high == mean_diff > 0, winner is A.
    let a = vec![1.0f32, 1.0, 1.0];
    let b = vec![0.0f32, 0.0, 0.0];
    let ev = AbEvaluator::new(AbConfig::default().with_bootstrap_samples(0));
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.winner, Some(AbWinner::A));
}

// ── Winner enum & misc ────────────────────────────────────────────────────────

#[test]
fn winner_enum_equality() {
    assert_eq!(AbWinner::A, AbWinner::A);
    assert_ne!(AbWinner::A, AbWinner::B);
}

#[test]
fn winner_enum_copy() {
    let w = AbWinner::A;
    let copied = w;
    assert_eq!(w, copied);
}

#[test]
fn result_clone_preserves_fields() {
    let a = vec![0.9f32, 0.2, 0.7];
    let b = vec![0.1f32, 0.8, 0.3];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    let cloned = r.clone();
    assert_eq!(r.wins_a, cloned.wins_a);
    assert_eq!(r.ci_low.to_bits(), cloned.ci_low.to_bits());
    assert_eq!(r.winner, cloned.winner);
}

#[test]
fn mostly_a_better_yields_winner_a() {
    // A beats B on most queries with a clear gap.
    let a = vec![0.9f32, 0.85, 0.8, 0.95, 0.7, 0.88, 0.82, 0.9, 0.86, 0.91];
    let b = vec![0.3f32, 0.35, 0.4, 0.25, 0.5, 0.3, 0.38, 0.2, 0.33, 0.29];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.winner, Some(AbWinner::A));
    assert!(r.ci_low > 0.0);
    assert!(r.wins_a > r.wins_b);
}

#[test]
fn mixed_results_no_clear_winner() {
    // Alternating small advantages cancel out around zero.
    let a = vec![0.5f32, 0.6, 0.4, 0.55, 0.45, 0.5, 0.52, 0.48];
    let b = vec![0.5f32, 0.4, 0.6, 0.45, 0.55, 0.5, 0.48, 0.52];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!((r.mean_diff).abs() < 0.05);
    assert!(r.ci_low <= 0.0 && r.ci_high >= 0.0);
    assert_eq!(r.winner, None);
}

#[test]
fn negative_scores_handled() {
    // Scores need not be in [0, 1]; differences still drive everything.
    let a = vec![-0.5f32, -0.2, -0.8, -0.1];
    let b = vec![-1.0f32, -0.9, -1.2, -0.7];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(r.mean_diff > 0.0);
    assert_eq!(r.wins_a, 4);
    assert_eq!(r.winner, Some(AbWinner::A));
}

#[test]
fn ci_bounds_within_diff_range() {
    // Bootstrap means are averages of differences, so the CI cannot exceed the
    // min/max of the individual differences.
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6, 0.3];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5, 0.9];
    let diffs: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let min_d = diffs.iter().copied().fold(f32::INFINITY, f32::min);
    let max_d = diffs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert!(r.ci_low >= min_d - 1e-5);
    assert!(r.ci_high <= max_d + 1e-5);
}

#[test]
fn wins_and_ties_sum_to_n() {
    let a = vec![0.9f32, 0.5, 0.7, 0.4, 0.6, 0.3, 0.5];
    let b = vec![0.1f32, 0.5, 0.3, 0.5, 0.5, 0.9, 0.5];
    let ev = AbEvaluator::new(AbConfig::default());
    let r = ev.compare(&a, &b).unwrap();
    assert_eq!(r.wins_a + r.wins_b + r.ties, 7);
}

#[test]
fn p_value_symmetric_under_swap() {
    // Swapping A and B negates differences; the two-sided p-value is unchanged.
    let a = vec![0.9f32, 0.2, 0.7, 0.4, 0.6, 0.3, 0.8];
    let b = vec![0.1f32, 0.8, 0.3, 0.5, 0.5, 0.9, 0.2];
    let ev = AbEvaluator::new(AbConfig::default());
    let r_ab = ev.compare(&a, &b).unwrap();
    let r_ba = ev.compare(&b, &a).unwrap();
    assert_eq!(r_ab.p_value.to_bits(), r_ba.p_value.to_bits());
}

#[test]
fn swap_negates_mean_diff_and_swaps_wins() {
    let a = vec![0.9f32, 0.2, 0.7, 0.4];
    let b = vec![0.1f32, 0.8, 0.3, 0.5];
    let ev = AbEvaluator::new(AbConfig::default());
    let r_ab = ev.compare(&a, &b).unwrap();
    let r_ba = ev.compare(&b, &a).unwrap();
    assert!((r_ab.mean_diff + r_ba.mean_diff).abs() < 1e-5);
    assert_eq!(r_ab.wins_a, r_ba.wins_b);
    assert_eq!(r_ab.wins_b, r_ba.wins_a);
}
