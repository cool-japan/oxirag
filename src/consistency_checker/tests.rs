//! Tests for the `consistency_checker` module.

use crate::consistency_checker::{
    ConflictType, ConsistencyChecker, ConsistencyConfig, ConsistencyError, ConsistencyReport,
    Inconsistency,
};

// ── ConflictType ──────────────────────────────────────────────────────────────

#[test]
fn test_conflict_type_as_str() {
    assert_eq!(ConflictType::Numerical.as_str(), "numerical");
    assert_eq!(ConflictType::Temporal.as_str(), "temporal");
    assert_eq!(ConflictType::Negation.as_str(), "negation");
    assert_eq!(ConflictType::Factual.as_str(), "factual");
}

// ── Inconsistency ─────────────────────────────────────────────────────────────

#[test]
fn test_inconsistency_description() {
    let inc = Inconsistency {
        claim_a: "a".to_string(),
        claim_b: "b".to_string(),
        conflict_type: ConflictType::Negation,
        confidence: 0.7,
    };
    let desc = inc.description();
    assert!(desc.contains("conflict"));
}

// ── ConsistencyReport ─────────────────────────────────────────────────────────

#[test]
fn test_report_new_empty_consistent() {
    let report = ConsistencyReport::new(vec![]);
    assert!(report.is_consistent);
    assert!((report.score - 1.0).abs() < f32::EPSILON);
    assert_eq!(report.inconsistency_count(), 0);
}

#[test]
fn test_report_new_with_inconsistencies() {
    let incs = vec![
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
    ];
    let report = ConsistencyReport::new(incs);
    assert!(!report.is_consistent);
    assert_eq!(report.inconsistency_count(), 2);
    // score = 1.0 - (2/10) = 0.8
    assert!((report.score - 0.8).abs() < 0.001);
}

#[test]
fn test_report_score_range() {
    // 10 inconsistencies → score = 0.0 (clamped)
    let incs: Vec<Inconsistency> = (0..10)
        .map(|i| Inconsistency {
            claim_a: format!("claim {i}"),
            claim_b: format!("claim {}", i + 1),
            conflict_type: ConflictType::Factual,
            confidence: 0.9,
        })
        .collect();
    let report = ConsistencyReport::new(incs);
    assert!(report.score >= 0.0);
    assert!(report.score <= 1.0);
    assert!((report.score - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_report_has_numerical_conflicts() {
    let incs = vec![Inconsistency {
        claim_a: "x".to_string(),
        claim_b: "y".to_string(),
        conflict_type: ConflictType::Numerical,
        confidence: 0.6,
    }];
    let report = ConsistencyReport::new(incs);
    assert!(report.has_numerical_conflicts());
    assert!(!report.has_temporal_conflicts());
}

#[test]
fn test_report_has_temporal_conflicts() {
    let incs = vec![Inconsistency {
        claim_a: "x".to_string(),
        claim_b: "y".to_string(),
        conflict_type: ConflictType::Temporal,
        confidence: 0.8,
    }];
    let report = ConsistencyReport::new(incs);
    assert!(report.has_temporal_conflicts());
    assert!(!report.has_numerical_conflicts());
}

// ── ConsistencyChecker errors ─────────────────────────────────────────────────

#[test]
fn test_checker_empty_error() {
    let checker = ConsistencyChecker::default();
    let result = checker.check("");
    assert!(matches!(result, Err(ConsistencyError::EmptyInput)));
}

#[test]
fn test_checker_too_few_sentences() {
    let checker = ConsistencyChecker::default();
    // Single sentence without terminal period — or a single sentence
    let result = checker.check("Just one sentence with no period at end");
    assert!(matches!(result, Err(ConsistencyError::TooFewSentences)));
}

// ── Consistent text ───────────────────────────────────────────────────────────

#[test]
fn test_checker_consistent_text() {
    let checker = ConsistencyChecker::default();
    let text = "The sun rises in the east. The sun sets in the west.";
    let report = checker.check(text).expect("check should succeed");
    // These two sentences should not conflict
    assert!(report.inconsistency_count() == 0 || report.score > 0.5);
}

// ── Negation conflict ─────────────────────────────────────────────────────────

#[test]
fn test_checker_negation_conflict() {
    let checker = ConsistencyChecker::default();
    // Two sentences with shared nouns; one negates the other
    let text = "The database stores every transaction record securely. The database does not store every transaction record securely.";
    let report = checker.check(text).expect("check should succeed");
    // The negation should be detected
    let has_negation = report
        .inconsistencies
        .iter()
        .any(|i| i.conflict_type == ConflictType::Negation);
    assert!(has_negation);
}

// ── Numerical conflict ────────────────────────────────────────────────────────

#[test]
fn test_checker_numerical_conflict() {
    let checker = ConsistencyChecker::default();
    let text = "The bridge spans 500 meters across the river valley. The bridge spans 2000 meters across the river valley.";
    let report = checker.check(text).expect("check should succeed");
    assert!(report.has_numerical_conflicts());
}

// ── Temporal conflict ─────────────────────────────────────────────────────────

#[test]
fn test_checker_temporal_conflict() {
    let checker = ConsistencyChecker::default();
    let text = "The company was founded in 1990 by its original shareholders. The company was founded in 2005 by its original shareholders.";
    let report = checker.check(text).expect("check should succeed");
    assert!(report.has_temporal_conflicts());
}

// ── No shared nouns ───────────────────────────────────────────────────────────

#[test]
fn test_checker_no_shared_nouns_no_conflict() {
    let checker = ConsistencyChecker::default();
    // Sentences about completely different topics; no shared long tokens
    let text = "Apples grow on trees in orchards. Submarines navigate underwater depths.";
    let report = checker.check(text).expect("check should succeed");
    // No shared meaningful nouns → no conflict
    assert_eq!(report.inconsistency_count(), 0);
}

// ── Multiple conflicts ────────────────────────────────────────────────────────

#[test]
fn test_checker_multiple_conflicts() {
    let checker = ConsistencyChecker::default();
    // Three sentences: two pairs can conflict
    let text = concat!(
        "The server processes 100 requests per second under normal load. ",
        "The server processes 500 requests per second under normal load. ",
        "The server processes 1000 requests per second under normal load."
    );
    let report = checker.check(text).expect("check should succeed");
    assert!(report.inconsistency_count() >= 1);
}

// ── Confidence filter ─────────────────────────────────────────────────────────

#[test]
fn test_checker_min_confidence_filter() {
    // Use a very high min_confidence that should filter out numerical conflicts (0.4/0.6)
    let cfg = ConsistencyConfig::default().with_min_confidence(0.9);
    let checker = ConsistencyChecker::new(cfg);
    let text = "The bridge spans 500 meters across the river valley. The bridge spans 2000 meters across the river valley.";
    let report = checker.check(text).expect("check should succeed");
    // All conflicts have confidence ≤ 0.8, so with min_confidence=0.9, all filtered out
    assert_eq!(report.inconsistency_count(), 0);
}

// ── Config ────────────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = ConsistencyConfig::default();
    assert!((cfg.min_confidence - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_config_builder() {
    let cfg = ConsistencyConfig::default().with_min_confidence(0.75);
    assert!((cfg.min_confidence - 0.75).abs() < f32::EPSILON);
}

// ── Error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    let e1 = ConsistencyError::EmptyInput;
    let e2 = ConsistencyError::TooFewSentences;
    assert!(!e1.to_string().is_empty());
    assert!(!e2.to_string().is_empty());
}
