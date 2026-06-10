//! Tests for the `trust_score` module.

use crate::consistency_checker::types::{ConflictType, ConsistencyReport, Inconsistency};
use crate::hallucination_detector::types::{ClaimSupport, HallucinationReport};
use crate::trust_score::{TrustComponents, TrustConfig, TrustError, TrustScore, TrustScorer};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_clean_hallucination() -> HallucinationReport {
    HallucinationReport::new(vec![ClaimSupport {
        claim: "Test claim".to_string(),
        support_score: 0.8,
        supporting_sources: vec!["s1".to_string()],
        is_hallucination: false,
    }])
}

fn make_consistent() -> ConsistencyReport {
    ConsistencyReport::new(vec![])
}

fn make_high_hallucination() -> HallucinationReport {
    HallucinationReport::new(vec![
        ClaimSupport {
            claim: "Hallucinated claim".to_string(),
            support_score: 0.05,
            supporting_sources: vec![],
            is_hallucination: true,
        },
        ClaimSupport {
            claim: "Another hallucination".to_string(),
            support_score: 0.1,
            supporting_sources: vec![],
            is_hallucination: true,
        },
    ])
}

fn make_inconsistent() -> ConsistencyReport {
    ConsistencyReport::new(vec![
        Inconsistency {
            claim_a: "a".to_string(),
            claim_b: "b".to_string(),
            conflict_type: ConflictType::Temporal,
            confidence: 0.8,
        },
        Inconsistency {
            claim_a: "c".to_string(),
            claim_b: "d".to_string(),
            conflict_type: ConflictType::Numerical,
            confidence: 0.6,
        },
    ])
}

// ── TrustComponents ───────────────────────────────────────────────────────────

#[test]
fn test_components_default_sum() {
    let c = TrustComponents::default();
    assert!((c.sum() - 1.0).abs() < 0.001);
    assert!(c.is_valid());
}

#[test]
fn test_components_normalize() {
    let c = TrustComponents {
        grounding: 2.0,
        consistency: 2.0,
        source_quality: 2.0,
        completeness: 2.0,
    };
    let n = c.normalize();
    assert!((n.sum() - 1.0).abs() < 0.001);
    assert!((n.grounding - 0.25).abs() < 0.001);
}

#[test]
fn test_components_is_valid() {
    let c = TrustComponents::default();
    assert!(c.is_valid());
}

#[test]
fn test_components_is_invalid() {
    let c = TrustComponents {
        grounding: 0.5,
        consistency: 0.5,
        source_quality: 0.5,
        completeness: 0.5,
    };
    assert!(!c.is_valid());
}

// ── TrustScore ────────────────────────────────────────────────────────────────

#[test]
fn test_trust_score_is_trustworthy() {
    let c = TrustComponents::default();
    let ts = TrustScore::new(0.8, 0.7, c);
    assert!(ts.is_trustworthy(0.6));
    assert!(!ts.is_trustworthy(0.9));
}

#[test]
fn test_trust_score_label() {
    let c = TrustComponents::default();
    assert_eq!(TrustScore::new(0.8, 0.8, c).label(), "high");
    assert_eq!(TrustScore::new(0.5, 0.5, c).label(), "medium");
    assert_eq!(TrustScore::new(0.2, 0.2, c).label(), "low");
}

// ── TrustConfig ───────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = TrustConfig::default();
    assert!(cfg.weights.is_valid());
    assert!((cfg.min_trustworthy - 0.6).abs() < f32::EPSILON);
}

#[test]
fn test_config_builders() {
    let weights = TrustComponents::default()
        .with_grounding(0.4)
        .with_consistency(0.3)
        .with_source_quality(0.2)
        .with_completeness(0.1);
    let cfg = TrustConfig::default()
        .with_weights(weights)
        .with_min_trustworthy(0.75);
    assert!((cfg.min_trustworthy - 0.75).abs() < f32::EPSILON);
    assert!((cfg.weights.grounding - 0.4).abs() < f32::EPSILON);
}

// ── TrustScorer errors ────────────────────────────────────────────────────────

#[test]
fn test_scorer_insufficient_data() {
    let scorer = TrustScorer::default();
    let hall = make_clean_hallucination();
    let cons = make_consistent();
    let result = scorer.score("", "", 0, &hall, &cons);
    assert!(matches!(result, Err(TrustError::InsufficientData)));
}

// ── Perfect scores ────────────────────────────────────────────────────────────

#[test]
fn test_scorer_all_perfect_scores() {
    let scorer = TrustScorer::default();
    let hall = make_clean_hallucination();
    let cons = make_consistent();
    let result = scorer
        .score(
            "Rust is a systems programming language.",
            "Rust language",
            5,
            &hall,
            &cons,
        )
        .expect("score should succeed");
    assert!(result.overall > 0.5);
    assert!(result.overall <= 1.0);
}

// ── Zero grounding ────────────────────────────────────────────────────────────

#[test]
fn test_scorer_zero_grounding() {
    let scorer = TrustScorer::default();
    let hall = make_high_hallucination(); // hallucination_rate = 1.0 → grounding = 0.0
    let cons = make_consistent();
    let result = scorer
        .score("Some answer text here.", "query", 3, &hall, &cons)
        .expect("score should succeed");
    // With grounding=0 the overall score should be significantly reduced
    assert!(result.overall < 0.75);
}

// ── High hallucination rate ───────────────────────────────────────────────────

#[test]
fn test_scorer_high_hallucination_rate() {
    let scorer = TrustScorer::default();
    let hall = make_high_hallucination();
    let cons = make_consistent();
    let result = scorer
        .score(
            "Some completely made up answer.",
            "real query",
            2,
            &hall,
            &cons,
        )
        .expect("score should succeed");
    assert!(result.components.grounding < 0.5);
}

// ── Inconsistent answer ───────────────────────────────────────────────────────

#[test]
fn test_scorer_inconsistent_answer() {
    let scorer = TrustScorer::default();
    let hall = make_clean_hallucination();
    let cons = make_inconsistent(); // score < 1.0
    let result = scorer
        .score("Answer with inconsistencies.", "query", 3, &hall, &cons)
        .expect("score should succeed");
    assert!(result.components.consistency < 1.0);
}

// ── Source coverage ───────────────────────────────────────────────────────────

#[test]
fn test_scorer_good_source_coverage() {
    let scorer = TrustScorer::default();
    let hall = make_clean_hallucination();
    let cons = make_consistent();
    let result = scorer
        .score("answer", "q", 5, &hall, &cons)
        .expect("score should succeed");
    assert!((result.components.source_quality - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_scorer_poor_source_coverage() {
    let scorer = TrustScorer::default();
    let hall = make_clean_hallucination();
    let cons = make_consistent();
    let result = scorer
        .score("answer", "q", 1, &hall, &cons)
        .expect("score should succeed");
    assert!((result.components.source_quality - 0.2).abs() < 0.001);
}

// ── Completeness ──────────────────────────────────────────────────────────────

#[test]
fn test_scorer_completeness_coverage() {
    let scorer = TrustScorer::default();
    let hall = make_clean_hallucination();
    let cons = make_consistent();
    let result = scorer
        .score(
            "Rust is a programming language used for systems development.",
            "Rust programming language",
            3,
            &hall,
            &cons,
        )
        .expect("score should succeed");
    // The answer contains all query tokens → completeness should be high
    assert!(result.components.completeness > 0.5);
}

// ── Score range ───────────────────────────────────────────────────────────────

#[test]
fn test_scorer_score_range_always_01() {
    let scorer = TrustScorer::default();
    let hall = make_clean_hallucination();
    let cons = make_consistent();
    let result = scorer
        .score("any answer", "any query", 3, &hall, &cons)
        .expect("score should succeed");
    assert!((0.0..=1.0).contains(&result.overall));
    assert!((0.0..=1.0).contains(&result.confidence));
}

// ── Simple estimate ───────────────────────────────────────────────────────────

#[test]
fn test_scorer_simple_estimate() {
    let scorer = TrustScorer::default();
    let score = scorer.score_simple("Some answer text.", 3);
    assert!((0.0..=1.0).contains(&score));
}

// ── Weighted combination ──────────────────────────────────────────────────────

#[test]
fn test_scorer_weighted_combination() {
    // Heavily weight grounding so that a clean report boosts the score
    let weights = TrustComponents {
        grounding: 0.7,
        consistency: 0.1,
        source_quality: 0.1,
        completeness: 0.1,
    };
    let cfg = TrustConfig::default().with_weights(weights);
    let scorer = TrustScorer::new(cfg);

    let hall_clean = make_clean_hallucination();
    let hall_bad = make_high_hallucination();
    let cons = make_consistent();

    let good = scorer
        .score("answer", "q", 2, &hall_clean, &cons)
        .expect("score should succeed");
    let bad = scorer
        .score("answer", "q", 2, &hall_bad, &cons)
        .expect("score should succeed");

    assert!(good.overall > bad.overall);
}

// ── Error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    let e = TrustError::InsufficientData;
    assert!(!e.to_string().is_empty());
}
