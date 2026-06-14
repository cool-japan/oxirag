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

//! Tests for the `abstention` module.

use super::policy::AbstentionPolicy;
use super::types::{AbstentionConfig, AbstentionDecision, AbstentionError, RiskCoverage};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// A policy with default configuration.
fn default_policy() -> AbstentionPolicy {
    AbstentionPolicy::new(AbstentionConfig::default())
}

// ── AbstentionDecision ──────────────────────────────────────────────────────

#[test]
fn decision_as_str_answer() {
    assert_eq!(AbstentionDecision::Answer.as_str(), "answer");
}

#[test]
fn decision_as_str_abstain() {
    assert_eq!(AbstentionDecision::Abstain.as_str(), "abstain");
}

#[test]
fn decision_display_matches_as_str() {
    assert_eq!(format!("{}", AbstentionDecision::Answer), "answer");
    assert_eq!(format!("{}", AbstentionDecision::Abstain), "abstain");
}

#[test]
fn decision_is_answer_is_abstain() {
    assert!(AbstentionDecision::Answer.is_answer());
    assert!(!AbstentionDecision::Answer.is_abstain());
    assert!(AbstentionDecision::Abstain.is_abstain());
    assert!(!AbstentionDecision::Abstain.is_answer());
}

#[test]
fn decision_equality_and_copy() {
    let a = AbstentionDecision::Answer;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(AbstentionDecision::Answer, AbstentionDecision::Abstain);
}

// ── AbstentionConfig defaults & builders ────────────────────────────────────

#[test]
fn config_defaults() {
    let cfg = AbstentionConfig::default();
    assert_eq!(cfg.confidence_threshold, 0.5);
    assert_eq!(cfg.min_support, 0.2);
    assert_eq!(cfg.support_weight, 0.4);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(AbstentionConfig::new(), AbstentionConfig::default());
}

#[test]
fn config_builder_confidence_threshold() {
    let cfg = AbstentionConfig::new().with_confidence_threshold(0.75);
    assert_eq!(cfg.confidence_threshold, 0.75);
    // Other fields untouched.
    assert_eq!(cfg.min_support, 0.2);
    assert_eq!(cfg.support_weight, 0.4);
}

#[test]
fn config_builder_min_support() {
    let cfg = AbstentionConfig::new().with_min_support(0.5);
    assert_eq!(cfg.min_support, 0.5);
    assert_eq!(cfg.confidence_threshold, 0.5);
}

#[test]
fn config_builder_support_weight() {
    let cfg = AbstentionConfig::new().with_support_weight(0.9);
    assert_eq!(cfg.support_weight, 0.9);
}

#[test]
fn config_builder_chaining() {
    let cfg = AbstentionConfig::new()
        .with_confidence_threshold(0.6)
        .with_min_support(0.3)
        .with_support_weight(0.5);
    assert_eq!(cfg.confidence_threshold, 0.6);
    assert_eq!(cfg.min_support, 0.3);
    assert_eq!(cfg.support_weight, 0.5);
}

// ── assess: Answer cases ────────────────────────────────────────────────────

#[test]
fn assess_high_confidence_high_support_answers() {
    let policy = default_policy();
    let a = policy.assess(0.9, 0.8);
    assert_eq!(a.decision, AbstentionDecision::Answer);
    assert!(a.is_answer());
    assert!(!a.is_abstain());
    assert!(a.reason.starts_with("answer"));
}

#[test]
fn assess_records_clamped_inputs() {
    let policy = default_policy();
    let a = policy.assess(1.5, -0.3);
    assert_eq!(a.confidence, 1.0);
    assert_eq!(a.support, 0.0);
}

#[test]
fn assess_combined_in_unit_range() {
    let policy = default_policy();
    for &(c, s) in &[(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (2.0, -1.0)] {
        let a = policy.assess(c, s);
        assert!(
            a.combined >= 0.0 && a.combined <= 1.0,
            "combined out of range"
        );
    }
}

#[test]
fn assess_answer_at_full_confidence_and_support() {
    let policy = default_policy();
    let a = policy.assess(1.0, 1.0);
    assert_eq!(a.decision, AbstentionDecision::Answer);
    assert_eq!(a.combined, 1.0);
}

// ── assess: low confidence ⇒ Abstain ────────────────────────────────────────

#[test]
fn assess_low_confidence_abstains() {
    let policy = default_policy();
    // High support but very low confidence: combined = 0.4*0.9 + 0.6*0.05 = 0.39 < 0.5.
    let a = policy.assess(0.05, 0.9);
    assert_eq!(a.decision, AbstentionDecision::Abstain);
    assert!(a.is_abstain());
    assert!(a.reason.contains("confidence_threshold"));
}

#[test]
fn assess_zero_confidence_zero_support_abstains() {
    let policy = default_policy();
    let a = policy.assess(0.0, 0.0);
    assert_eq!(a.decision, AbstentionDecision::Abstain);
    assert_eq!(a.combined, 0.0);
}

// ── assess: support below min_support ⇒ Abstain even at high confidence ──────

#[test]
fn assess_support_below_min_abstains_even_at_high_confidence() {
    let policy = default_policy();
    // confidence is maximal but support 0.1 < min_support 0.2.
    let a = policy.assess(1.0, 0.1);
    assert_eq!(a.decision, AbstentionDecision::Abstain);
    assert!(a.reason.contains("min_support"));
}

#[test]
fn assess_support_gate_dominates_confidence() {
    // Even with combined above the confidence threshold, the support gate refuses.
    let policy = AbstentionPolicy::new(
        AbstentionConfig::new()
            .with_support_weight(0.0) // combined == confidence
            .with_min_support(0.5),
    );
    let a = policy.assess(1.0, 0.3);
    // combined == 1.0 >= 0.5, but support 0.3 < 0.5.
    assert_eq!(a.combined, 1.0);
    assert_eq!(a.decision, AbstentionDecision::Abstain);
    assert!(a.reason.contains("min_support"));
}

#[test]
fn assess_support_exactly_at_min_support_is_allowed() {
    let policy = default_policy();
    // support == min_support (0.2) and high confidence ⇒ support gate passes.
    let a = policy.assess(1.0, 0.2);
    // combined = 0.4*0.2 + 0.6*1.0 = 0.68 >= 0.5.
    assert_eq!(a.decision, AbstentionDecision::Answer);
}

// ── combined formula respects support_weight ────────────────────────────────

#[test]
fn combined_formula_default_weight() {
    let policy = default_policy(); // support_weight = 0.4
    let a = policy.assess(0.6, 0.9);
    // 0.4*0.9 + 0.6*0.6 = 0.36 + 0.36 = 0.72.
    assert!((a.combined - 0.72).abs() < 1e-6);
}

#[test]
fn combined_formula_weight_zero_is_pure_confidence() {
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_support_weight(0.0));
    let a = policy.assess(0.3, 0.95);
    assert!((a.combined - 0.3).abs() < 1e-6);
}

#[test]
fn combined_formula_weight_one_is_pure_support() {
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_support_weight(1.0));
    let a = policy.assess(0.95, 0.3);
    assert!((a.combined - 0.3).abs() < 1e-6);
}

#[test]
fn combined_formula_half_weight_is_average() {
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_support_weight(0.5));
    let a = policy.assess(0.4, 0.8);
    assert!((a.combined - 0.6).abs() < 1e-6);
}

#[test]
fn combined_formula_higher_support_weight_favours_support() {
    let low_w = AbstentionPolicy::new(AbstentionConfig::new().with_support_weight(0.2));
    let high_w = AbstentionPolicy::new(AbstentionConfig::new().with_support_weight(0.8));
    // Support far exceeds confidence; weighting it more raises combined.
    let a_low = low_w.assess(0.2, 0.9);
    let a_high = high_w.assess(0.2, 0.9);
    assert!(a_high.combined > a_low.combined);
}

#[test]
fn assess_support_weight_clamped_above_one() {
    // An out-of-range weight is clamped to 1.0 (pure support), never panics.
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_support_weight(5.0));
    let a = policy.assess(0.9, 0.25);
    assert!((a.combined - 0.25).abs() < 1e-6);
}

// ── threshold boundary flips the decision ───────────────────────────────────

#[test]
fn assess_threshold_boundary_flips_decision() {
    // support_weight 0.0 ⇒ combined == confidence; min_support 0.0 so support never gates.
    let cfg = AbstentionConfig::new()
        .with_support_weight(0.0)
        .with_min_support(0.0)
        .with_confidence_threshold(0.5);
    let policy = AbstentionPolicy::new(cfg);

    // Exactly at threshold ⇒ Answer (>= is inclusive).
    assert_eq!(policy.decide(0.5, 0.5), AbstentionDecision::Answer);
    // Just below ⇒ Abstain.
    assert_eq!(policy.decide(0.499, 0.5), AbstentionDecision::Abstain);
    // Just above ⇒ Answer.
    assert_eq!(policy.decide(0.501, 0.5), AbstentionDecision::Answer);
}

#[test]
fn assess_combined_exactly_at_threshold_answers() {
    let policy = default_policy(); // threshold 0.5
    // 0.4*0.5 + 0.6*0.5 = 0.5 exactly.
    let a = policy.assess(0.5, 0.5);
    assert_eq!(a.combined, 0.5);
    assert_eq!(a.decision, AbstentionDecision::Answer);
}

// ── decide() convenience wrapper ────────────────────────────────────────────

#[test]
fn decide_matches_assess_decision() {
    let policy = default_policy();
    for &(c, s) in &[(0.9, 0.9), (0.1, 0.1), (1.0, 0.1), (0.05, 0.95)] {
        assert_eq!(policy.decide(c, s), policy.assess(c, s).decision);
    }
}

#[test]
fn decide_high_signals_answers() {
    assert_eq!(
        default_policy().decide(0.9, 0.8),
        AbstentionDecision::Answer
    );
}

#[test]
fn decide_low_signals_abstains() {
    assert_eq!(
        default_policy().decide(0.1, 0.1),
        AbstentionDecision::Abstain
    );
}

// ── risk_coverage: threshold 0 ⇒ coverage 1.0 ───────────────────────────────

#[test]
fn risk_coverage_threshold_zero_full_coverage() {
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_confidence_threshold(0.0));
    let scored = vec![(0.9_f32, true), (0.2, false), (0.5, true)];
    let rc = policy.risk_coverage(&scored).unwrap();
    assert_eq!(rc.coverage, 1.0);
    assert_eq!(rc.answered, 3);
    assert_eq!(rc.total, 3);
}

#[test]
fn risk_coverage_curve_first_point_full_coverage() {
    let policy = default_policy();
    let scored = vec![(0.9_f32, true), (0.7, true), (0.3, false), (0.1, false)];
    let curve = policy.risk_coverage_curve(&scored, 6);
    assert_eq!(curve.first().unwrap().coverage, 1.0);
}

// ── risk_coverage: risk == error rate among answered ────────────────────────

#[test]
fn risk_coverage_risk_is_error_rate_among_answered() {
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_confidence_threshold(0.5));
    // Answered (conf >= 0.5): (0.9,true), (0.6,false), (0.5,true). 1 wrong of 3.
    let scored = vec![
        (0.9_f32, true),
        (0.6, false),
        (0.5, true),
        (0.4, false), // not answered
        (0.1, true),  // not answered
    ];
    let rc = policy.risk_coverage(&scored).unwrap();
    assert_eq!(rc.answered, 3);
    assert_eq!(rc.total, 5);
    assert!((rc.coverage - 3.0 / 5.0).abs() < 1e-6);
    assert!((rc.risk - 1.0 / 3.0).abs() < 1e-6);
    assert_eq!(rc.wrong(), 1);
}

#[test]
fn risk_coverage_all_correct_zero_risk() {
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_confidence_threshold(0.0));
    let scored = vec![(0.9_f32, true), (0.8, true), (0.7, true)];
    let rc = policy.risk_coverage(&scored).unwrap();
    assert_eq!(rc.risk, 0.0);
    assert_eq!(rc.wrong(), 0);
}

#[test]
fn risk_coverage_all_wrong_full_risk() {
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_confidence_threshold(0.0));
    let scored = vec![(0.9_f32, false), (0.8, false)];
    let rc = policy.risk_coverage(&scored).unwrap();
    assert_eq!(rc.risk, 1.0);
    assert_eq!(rc.wrong(), 2);
}

#[test]
fn risk_coverage_nothing_answered_zero_risk_zero_coverage() {
    // Threshold above every confidence ⇒ nothing answered.
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_confidence_threshold(0.99));
    let scored = vec![(0.5_f32, true), (0.3, false)];
    let rc = policy.risk_coverage(&scored).unwrap();
    assert_eq!(rc.answered, 0);
    assert_eq!(rc.coverage, 0.0);
    assert_eq!(rc.risk, 0.0);
    assert_eq!(rc.wrong(), 0);
}

#[test]
fn risk_coverage_clamps_confidence() {
    // Confidence above 1.0 is clamped; still answered at threshold 1.0.
    let policy = AbstentionPolicy::new(AbstentionConfig::new().with_confidence_threshold(1.0));
    let scored = vec![(1.5_f32, true), (0.9, false)];
    let rc = policy.risk_coverage(&scored).unwrap();
    // Only the clamped-to-1.0 entry meets threshold 1.0.
    assert_eq!(rc.answered, 1);
    assert_eq!(rc.risk, 0.0);
}

// ── raising the threshold lowers coverage and (sensibly) lowers risk ─────────

#[test]
fn risk_coverage_raising_threshold_lowers_coverage() {
    let policy = default_policy();
    // Confidence-sorted dataset where higher confidence ⇒ more often correct.
    let scored = vec![
        (0.95_f32, true),
        (0.85, true),
        (0.75, true),
        (0.55, false),
        (0.35, false),
        (0.15, false),
    ];
    let low = policy.risk_coverage_at(&scored, 0.1).unwrap();
    let mid = policy.risk_coverage_at(&scored, 0.5).unwrap();
    let high = policy.risk_coverage_at(&scored, 0.8).unwrap();
    assert!(low.coverage >= mid.coverage);
    assert!(mid.coverage >= high.coverage);
    assert_eq!(low.coverage, 1.0);
}

#[test]
fn risk_coverage_raising_threshold_lowers_risk_on_sensible_data() {
    let policy = default_policy();
    // A well-calibrated dataset: correctness concentrated at high confidence.
    let scored = vec![
        (0.95_f32, true),
        (0.90, true),
        (0.80, true),
        (0.60, false),
        (0.40, false),
        (0.20, false),
    ];
    let low = policy.risk_coverage_at(&scored, 0.1).unwrap(); // all 6, 3 wrong ⇒ 0.5
    let high = policy.risk_coverage_at(&scored, 0.7).unwrap(); // top 3, 0 wrong ⇒ 0.0
    assert!(high.risk <= low.risk);
    assert!((low.risk - 0.5).abs() < 1e-6);
    assert_eq!(high.risk, 0.0);
}

// ── risk_coverage_curve: length == steps & monotone non-increasing coverage ──

#[test]
fn risk_coverage_curve_length_equals_steps() {
    let policy = default_policy();
    let scored = vec![(0.9_f32, true), (0.5, false), (0.2, true)];
    for steps in [1_usize, 2, 5, 10, 25] {
        let curve = policy.risk_coverage_curve(&scored, steps);
        assert_eq!(
            curve.len(),
            steps,
            "curve length mismatch for steps={steps}"
        );
    }
}

#[test]
fn risk_coverage_curve_zero_steps_clamped_to_one() {
    let policy = default_policy();
    let scored = vec![(0.9_f32, true)];
    let curve = policy.risk_coverage_curve(&scored, 0);
    assert_eq!(curve.len(), 1);
    assert_eq!(curve[0].coverage, 1.0);
}

#[test]
fn risk_coverage_curve_coverage_monotonically_non_increasing() {
    let policy = default_policy();
    let scored = vec![
        (0.95_f32, true),
        (0.80, true),
        (0.65, false),
        (0.50, true),
        (0.30, false),
        (0.10, false),
    ];
    let curve = policy.risk_coverage_curve(&scored, 11);
    for window in curve.windows(2) {
        assert!(
            window[1].coverage <= window[0].coverage + 1e-6,
            "coverage increased: {} -> {}",
            window[0].coverage,
            window[1].coverage
        );
    }
}

#[test]
fn risk_coverage_curve_last_point_lowest_coverage() {
    let policy = default_policy();
    let scored = vec![(0.9_f32, true), (0.6, false), (0.3, true), (0.1, false)];
    let curve = policy.risk_coverage_curve(&scored, 8);
    let first = curve.first().unwrap().coverage;
    let last = curve.last().unwrap().coverage;
    assert_eq!(first, 1.0);
    assert!(last <= first);
}

#[test]
fn risk_coverage_curve_total_constant_across_points() {
    let policy = default_policy();
    let scored = vec![(0.9_f32, true), (0.5, false), (0.2, true)];
    let curve = policy.risk_coverage_curve(&scored, 5);
    for point in &curve {
        assert_eq!(point.total, 3);
    }
}

#[test]
fn risk_coverage_curve_single_step_full_coverage() {
    let policy = default_policy();
    let scored = vec![(0.9_f32, true), (0.1, false)];
    let curve = policy.risk_coverage_curve(&scored, 1);
    assert_eq!(curve.len(), 1);
    assert_eq!(curve[0].coverage, 1.0);
    assert_eq!(curve[0].answered, 2);
}

// ── EmptyData error ─────────────────────────────────────────────────────────

#[test]
fn risk_coverage_empty_data_errors() {
    let policy = default_policy();
    let scored: Vec<(f32, bool)> = Vec::new();
    let err = policy.risk_coverage(&scored).unwrap_err();
    assert!(matches!(err, AbstentionError::EmptyData));
}

#[test]
fn risk_coverage_at_empty_data_errors() {
    let policy = default_policy();
    let scored: Vec<(f32, bool)> = Vec::new();
    assert!(matches!(
        policy.risk_coverage_at(&scored, 0.5),
        Err(AbstentionError::EmptyData)
    ));
}

#[test]
fn empty_data_error_message() {
    let err = AbstentionError::EmptyData;
    assert_eq!(err.to_string(), "dataset is empty");
}

#[test]
fn risk_coverage_curve_empty_data_returns_empty_vec() {
    let policy = default_policy();
    let scored: Vec<(f32, bool)> = Vec::new();
    let curve = policy.risk_coverage_curve(&scored, 5);
    assert!(curve.is_empty());
}

// ── RiskCoverage::wrong reconstruction ──────────────────────────────────────

#[test]
fn risk_coverage_wrong_reconstructs_count() {
    let rc = RiskCoverage {
        coverage: 0.5,
        risk: 0.25,
        answered: 4,
        total: 8,
    };
    assert_eq!(rc.wrong(), 1);
}

#[test]
fn risk_coverage_wrong_zero_when_none_answered() {
    let rc = RiskCoverage {
        coverage: 0.0,
        risk: 0.0,
        answered: 0,
        total: 5,
    };
    assert_eq!(rc.wrong(), 0);
}

// ── Determinism ─────────────────────────────────────────────────────────────

#[test]
fn assess_is_deterministic() {
    let policy = default_policy();
    let first = policy.assess(0.73, 0.41);
    for _ in 0..50 {
        let again = policy.assess(0.73, 0.41);
        assert_eq!(first, again);
    }
}

#[test]
fn risk_coverage_is_deterministic() {
    let policy = default_policy();
    let scored = vec![(0.9_f32, true), (0.6, false), (0.4, true), (0.2, false)];
    let first = policy.risk_coverage(&scored).unwrap();
    for _ in 0..50 {
        let again = policy.risk_coverage(&scored).unwrap();
        assert_eq!(first, again);
    }
}

#[test]
fn risk_coverage_curve_is_deterministic() {
    let policy = default_policy();
    let scored = vec![(0.9_f32, true), (0.5, false), (0.3, true), (0.1, false)];
    let first = policy.risk_coverage_curve(&scored, 7);
    let again = policy.risk_coverage_curve(&scored, 7);
    assert_eq!(first.len(), again.len());
    for (a, b) in first.iter().zip(again.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn policy_clone_behaves_identically() {
    let policy = default_policy();
    let cloned = policy.clone();
    let scored = vec![(0.9_f32, true), (0.4, false)];
    assert_eq!(
        policy.risk_coverage(&scored).unwrap(),
        cloned.risk_coverage(&scored).unwrap()
    );
    assert_eq!(policy.assess(0.8, 0.5), cloned.assess(0.8, 0.5));
}

// ── Assessment field consistency ────────────────────────────────────────────

#[test]
fn assessment_reason_non_empty() {
    let policy = default_policy();
    assert!(!policy.assess(0.9, 0.8).reason.is_empty());
    assert!(!policy.assess(0.1, 0.1).reason.is_empty());
}

#[test]
fn assessment_answer_reason_mentions_answer() {
    let policy = default_policy();
    let a = policy.assess(0.95, 0.9);
    assert!(a.reason.starts_with("answer"));
}

#[test]
fn assessment_abstain_reason_mentions_abstain() {
    let policy = default_policy();
    let a = policy.assess(0.0, 0.0);
    assert!(a.reason.starts_with("abstain"));
}

#[test]
fn assess_support_gate_checked_before_confidence_gate() {
    // When both gates fail, the support gate is reported first.
    let policy = default_policy();
    let a = policy.assess(0.0, 0.05); // support 0.05 < 0.2 and combined low
    assert!(a.reason.contains("min_support"));
}
