//! Tests for the incremental consistency checker.

#![cfg(test)]

use super::checker::IncrementalConsistencyChecker;
use super::types::{ClaimConflict, ConflictType, ConsistencyResult, Resolution};
use crate::types::{CausalStrength, ClaimStructure, ComparisonOp, LogicalClaim, Modality, TimeRelation};

fn make_predicate_claim(
    id: &str,
    subject: &str,
    predicate: &str,
    object: Option<&str>,
) -> LogicalClaim {
    let mut claim = LogicalClaim::new(
        format!("{subject} {predicate} {}", object.unwrap_or("")),
        ClaimStructure::Predicate {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.map(ToString::to_string),
        },
    );
    claim.id = id.to_string();
    claim
}

fn make_comparison_claim(id: &str, left: &str, op: ComparisonOp, right: &str) -> LogicalClaim {
    let mut claim = LogicalClaim::new(
        format!("{left} {} {right}", op.to_smtlib()),
        ClaimStructure::Comparison {
            left: left.to_string(),
            operator: op,
            right: right.to_string(),
        },
    );
    claim.id = id.to_string();
    claim
}

fn make_temporal_claim(
    id: &str,
    event: &str,
    relation: TimeRelation,
    reference: &str,
) -> LogicalClaim {
    let mut claim = LogicalClaim::new(
        format!("{event} {} {reference}", relation.to_smtlib()),
        ClaimStructure::Temporal {
            event: event.to_string(),
            time_relation: relation,
            reference: reference.to_string(),
        },
    );
    claim.id = id.to_string();
    claim
}

#[test]
fn test_add_consistent_claims() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));
    let claim2 = make_predicate_claim("c2", "grass", "is", Some("green"));

    let result1 = checker.add_claim(claim1);
    assert!(result1.is_consistent());

    let result2 = checker.add_claim(claim2);
    assert!(result2.is_consistent());

    assert_eq!(checker.claim_count(), 2);
    assert!(checker.get_conflicts().is_empty());
}

#[test]
fn test_detect_direct_contradiction() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));

    let negated_structure = ClaimStructure::Not(Box::new(ClaimStructure::Predicate {
        subject: "sky".to_string(),
        predicate: "is".to_string(),
        object: Some("blue".to_string()),
    }));
    let mut claim2 = LogicalClaim::new("sky is not blue", negated_structure);
    claim2.id = "c2".to_string();

    checker.add_claim(claim1);
    let result = checker.add_claim(claim2);

    assert!(result.is_inconsistent());
    let conflicts = result.conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(
        conflicts[0].conflict_type,
        ConflictType::DirectContradiction
    );
}

#[test]
fn test_detect_predicate_conflict() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_predicate_claim("c1", "cat", "is", Some("alive"));
    let claim2 = make_predicate_claim("c2", "cat", "is", Some("dead"));

    checker.add_claim(claim1);
    let result = checker.add_claim(claim2);

    assert!(result.is_inconsistent());
    let conflicts = result.conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].conflict_type, ConflictType::PredicateConflict);
}

#[test]
fn test_detect_comparison_conflict() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_comparison_claim("c1", "A", ComparisonOp::GreaterThan, "B");
    let claim2 = make_comparison_claim("c2", "A", ComparisonOp::LessThan, "B");

    checker.add_claim(claim1);
    let result = checker.add_claim(claim2);

    assert!(result.is_inconsistent());
    let conflicts = result.conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].conflict_type, ConflictType::ComparisonConflict);
}

#[test]
fn test_detect_circular_comparison() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_comparison_claim("c1", "A", ComparisonOp::GreaterThan, "B");
    let claim2 = make_comparison_claim("c2", "B", ComparisonOp::GreaterThan, "A");

    checker.add_claim(claim1);
    let result = checker.add_claim(claim2);

    assert!(result.is_inconsistent());
    let conflicts = result.conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].conflict_type, ConflictType::ComparisonConflict);
}

#[test]
fn test_detect_temporal_conflict() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_temporal_claim("c1", "event_a", TimeRelation::Before, "event_b");
    let claim2 = make_temporal_claim("c2", "event_b", TimeRelation::Before, "event_a");

    checker.add_claim(claim1);
    let result = checker.add_claim(claim2);

    assert!(result.is_inconsistent());
    let conflicts = result.conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].conflict_type, ConflictType::TemporalConflict);
}

#[test]
fn test_detect_opposite_temporal_relations() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_temporal_claim("c1", "meeting", TimeRelation::Before, "lunch");
    let claim2 = make_temporal_claim("c2", "meeting", TimeRelation::After, "lunch");

    checker.add_claim(claim1);
    let result = checker.add_claim(claim2);

    assert!(result.is_inconsistent());
    let conflicts = result.conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].conflict_type, ConflictType::TemporalConflict);
}

#[test]
fn test_detect_modal_conflict() {
    let mut checker = IncrementalConsistencyChecker::new();

    let inner = ClaimStructure::Raw("it will rain".to_string());

    let mut claim1 = LogicalClaim::new(
        "it must rain",
        ClaimStructure::Modal {
            claim: Box::new(inner.clone()),
            modality: Modality::Necessary,
        },
    );
    claim1.id = "c1".to_string();

    let mut claim2 = LogicalClaim::new(
        "it is unlikely to rain",
        ClaimStructure::Modal {
            claim: Box::new(inner),
            modality: Modality::Unlikely,
        },
    );
    claim2.id = "c2".to_string();

    checker.add_claim(claim1);
    let result = checker.add_claim(claim2);

    assert!(result.is_inconsistent());
    let conflicts = result.conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].conflict_type, ConflictType::ModalConflict);
}

#[test]
fn test_detect_causal_conflict() {
    let mut checker = IncrementalConsistencyChecker::new();

    let cause_a = ClaimStructure::Raw("event_a".to_string());
    let effect_b = ClaimStructure::Raw("event_b".to_string());

    let mut claim1 = LogicalClaim::new(
        "A causes B",
        ClaimStructure::Causal {
            cause: Box::new(cause_a.clone()),
            effect: Box::new(effect_b.clone()),
            strength: CausalStrength::Direct,
        },
    );
    claim1.id = "c1".to_string();

    let mut claim2 = LogicalClaim::new(
        "B causes A",
        ClaimStructure::Causal {
            cause: Box::new(effect_b),
            effect: Box::new(cause_a),
            strength: CausalStrength::Direct,
        },
    );
    claim2.id = "c2".to_string();

    checker.add_claim(claim1);
    let result = checker.add_claim(claim2);

    assert!(result.is_inconsistent());
    let conflicts = result.conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].conflict_type, ConflictType::CausalConflict);
}

#[test]
fn test_check_new_claim_without_adding() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));
    checker.add_claim(claim1);

    // Check a contradicting claim without adding it
    let claim2 = make_predicate_claim("c2", "sky", "is", Some("green"));
    let result = checker.check_new_claim(&claim2);

    assert!(result.is_inconsistent());

    // Verify it wasn't added
    assert_eq!(checker.claim_count(), 1);
    assert!(checker.get_claim("c2").is_none());
}

#[test]
fn test_remove_claim() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));
    let claim2 = make_predicate_claim("c2", "sky", "is", Some("green"));

    checker.add_claim(claim1);
    checker.add_claim(claim2);

    assert!(!checker.get_conflicts().is_empty());

    // Remove one of the conflicting claims
    checker.remove_claim("c2");

    assert_eq!(checker.claim_count(), 1);
    assert!(checker.get_conflicts().is_empty());
    assert!(checker.check_consistency().is_consistent());
}

#[test]
fn test_suggest_resolution() {
    let mut checker = IncrementalConsistencyChecker::new();

    let mut claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));
    claim1.confidence = 0.9;

    let mut claim2 = make_predicate_claim("c2", "sky", "is", Some("green"));
    claim2.confidence = 0.5;

    checker.add_claim(claim1);
    checker.add_claim(claim2);

    let conflicts = checker.get_conflicts();
    assert_eq!(conflicts.len(), 1);

    let resolutions = checker.suggest_resolution(&conflicts[0]);
    assert!(!resolutions.is_empty());

    // Should suggest prioritizing by confidence
    let has_priority_resolution = resolutions.iter().any(|r| {
        matches!(r, Resolution::PrioritizeByCofidence { keep_claim_id, remove_claim_id }
            if keep_claim_id == "c1" && remove_claim_id == "c2")
    });
    assert!(has_priority_resolution);
}

#[test]
fn test_clear() {
    let mut checker = IncrementalConsistencyChecker::new();

    checker.add_claim(make_predicate_claim("c1", "sky", "is", Some("blue")));
    checker.add_claim(make_predicate_claim("c2", "grass", "is", Some("green")));

    assert_eq!(checker.claim_count(), 2);

    checker.clear();

    assert_eq!(checker.claim_count(), 0);
    assert!(checker.get_conflicts().is_empty());
}

#[test]
fn test_consistency_result_methods() {
    let consistent = ConsistencyResult::Consistent;
    assert!(consistent.is_consistent());
    assert!(!consistent.is_inconsistent());
    assert!(consistent.conflicts().is_empty());

    let conflict = ClaimConflict::new("c1", "c2", ConflictType::DirectContradiction, "test");
    let inconsistent = ConsistencyResult::Inconsistent(vec![conflict.clone()]);
    assert!(!inconsistent.is_consistent());
    assert!(inconsistent.is_inconsistent());
    assert_eq!(inconsistent.conflicts().len(), 1);

    let unknown = ConsistencyResult::Unknown;
    assert!(!unknown.is_consistent());
    assert!(!unknown.is_inconsistent());
    assert!(unknown.conflicts().is_empty());
}

#[test]
fn test_conflict_severity() {
    let conflict = ClaimConflict::new("c1", "c2", ConflictType::DirectContradiction, "test");
    assert!((conflict.severity - 1.0).abs() < f32::EPSILON);

    let conflict_with_severity = conflict.with_severity(0.5);
    assert!((conflict_with_severity.severity - 0.5).abs() < f32::EPSILON);

    // Test clamping
    let clamped = ClaimConflict::new("c1", "c2", ConflictType::SemanticContradiction, "test")
        .with_severity(1.5);
    assert!((clamped.severity - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_conflict_type_severity() {
    assert!((ConflictType::DirectContradiction.default_severity() - 1.0).abs() < f32::EPSILON);
    assert!(
        ConflictType::SemanticContradiction.default_severity()
            > ConflictType::ModalConflict.default_severity()
    );
}

#[test]
fn test_duplicate_claim_handling() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim = make_predicate_claim("c1", "sky", "is", Some("blue"));
    checker.add_claim(claim.clone());

    // Adding same claim again should not create duplicates
    let result = checker.add_claim(claim);
    assert!(result.is_consistent());
    assert_eq!(checker.claim_count(), 1);
}

#[test]
fn test_opposite_predicates() {
    assert!(IncrementalConsistencyChecker::are_opposite_predicates(
        "is", "is not"
    ));
    assert!(IncrementalConsistencyChecker::are_opposite_predicates(
        "true", "false"
    ));
    assert!(IncrementalConsistencyChecker::are_opposite_predicates(
        "alive", "dead"
    ));
    assert!(!IncrementalConsistencyChecker::are_opposite_predicates(
        "is", "has"
    ));
}

#[test]
fn test_get_claim() {
    let mut checker = IncrementalConsistencyChecker::new();

    let claim = make_predicate_claim("c1", "sky", "is", Some("blue"));
    checker.add_claim(claim);

    assert!(checker.get_claim("c1").is_some());
    assert!(checker.get_claim("nonexistent").is_none());
}
