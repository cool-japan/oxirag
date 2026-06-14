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
//! Tests for the `knowledge_conflict` module.

use std::collections::HashMap;

use crate::knowledge_conflict::{
    ConflictDetector, ConflictKind, ConflictPolicy, ConflictResolution, ConflictResolver,
    KnowledgeConflictConfig, KnowledgeConflictError, PassageConflict,
};
use crate::types::{Document, DocumentId};

// ── helpers ─────────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

fn ages(pairs: &[(&str, f64)]) -> HashMap<DocumentId, f64> {
    pairs
        .iter()
        .map(|(id, age)| (DocumentId::from_string(*id), *age))
        .collect()
}

fn authorities(pairs: &[(&str, f32)]) -> HashMap<DocumentId, f32> {
    pairs
        .iter()
        .map(|(id, a)| (DocumentId::from_string(*id), *a))
        .collect()
}

fn empty_ages() -> HashMap<DocumentId, f64> {
    HashMap::new()
}

fn empty_auth() -> HashMap<DocumentId, f32> {
    HashMap::new()
}

// ── ConflictPolicy ──────────────────────────────────────────────────────────────

#[test]
fn test_policy_default_is_majority() {
    assert_eq!(ConflictPolicy::default(), ConflictPolicy::Majority);
}

#[test]
fn test_policy_as_str() {
    assert_eq!(ConflictPolicy::Recency.as_str(), "recency");
    assert_eq!(ConflictPolicy::Authority.as_str(), "authority");
    assert_eq!(ConflictPolicy::Majority.as_str(), "majority");
}

#[test]
fn test_policy_copy_eq() {
    let p = ConflictPolicy::Authority;
    let q = p;
    assert_eq!(p, q);
}

// ── ConflictKind ────────────────────────────────────────────────────────────────

#[test]
fn test_kind_as_str() {
    assert_eq!(ConflictKind::Negation.as_str(), "negation");
    assert_eq!(ConflictKind::Numeric.as_str(), "numeric");
    assert_eq!(ConflictKind::Temporal.as_str(), "temporal");
}

#[test]
fn test_kind_equality() {
    assert_eq!(ConflictKind::Numeric, ConflictKind::Numeric);
    assert_ne!(ConflictKind::Numeric, ConflictKind::Temporal);
}

// ── KnowledgeConflictConfig ─────────────────────────────────────────────────────

#[test]
fn test_config_default() {
    let config = KnowledgeConflictConfig::default();
    assert_eq!(config.policy, ConflictPolicy::Majority);
    assert_eq!(config.min_shared_terms, 2);
}

#[test]
fn test_config_new_matches_default() {
    let a = KnowledgeConflictConfig::new();
    let b = KnowledgeConflictConfig::default();
    assert_eq!(a.policy, b.policy);
    assert_eq!(a.min_shared_terms, b.min_shared_terms);
}

#[test]
fn test_config_with_policy() {
    let config = KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency);
    assert_eq!(config.policy, ConflictPolicy::Recency);
    assert_eq!(config.min_shared_terms, 2);
}

#[test]
fn test_config_with_min_shared_terms() {
    let config = KnowledgeConflictConfig::new().with_min_shared_terms(4);
    assert_eq!(config.min_shared_terms, 4);
    assert_eq!(config.policy, ConflictPolicy::Majority);
}

#[test]
fn test_config_builder_chained() {
    let config = KnowledgeConflictConfig::new()
        .with_policy(ConflictPolicy::Authority)
        .with_min_shared_terms(1);
    assert_eq!(config.policy, ConflictPolicy::Authority);
    assert_eq!(config.min_shared_terms, 1);
}

#[test]
fn test_config_clone() {
    let config = KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency);
    let cloned = config.clone();
    assert_eq!(cloned.policy, config.policy);
    assert_eq!(cloned.min_shared_terms, config.min_shared_terms);
}

// ── PassageConflict ─────────────────────────────────────────────────────────────

#[test]
fn test_passage_conflict_new() {
    let pc = PassageConflict::new(0, 1, "a claim", "b claim", ConflictKind::Negation);
    assert_eq!(pc.passage_a, 0);
    assert_eq!(pc.passage_b, 1);
    assert_eq!(pc.claim_a, "a claim");
    assert_eq!(pc.claim_b, "b claim");
    assert_eq!(pc.kind, ConflictKind::Negation);
}

#[test]
fn test_passage_conflict_equality() {
    let a = PassageConflict::new(0, 1, "x", "y", ConflictKind::Numeric);
    let b = PassageConflict::new(0, 1, "x", "y", ConflictKind::Numeric);
    assert_eq!(a, b);
}

// ── ConflictResolution ──────────────────────────────────────────────────────────

#[test]
fn test_resolution_new() {
    let r = ConflictResolution::new(0, vec![1], ConflictPolicy::Majority, "because");
    assert_eq!(r.winner, 0);
    assert_eq!(r.losers, vec![1]);
    assert_eq!(r.policy, ConflictPolicy::Majority);
    assert_eq!(r.rationale, "because");
}

// ── ConflictDetector: defaults ──────────────────────────────────────────────────

#[test]
fn test_detector_default() {
    let detector = ConflictDetector::default();
    assert_eq!(detector.config.min_shared_terms, 2);
    assert_eq!(detector.config.policy, ConflictPolicy::Majority);
}

#[test]
fn test_detector_new_holds_config() {
    let config = KnowledgeConflictConfig::new().with_min_shared_terms(3);
    let detector = ConflictDetector::new(config);
    assert_eq!(detector.config.min_shared_terms, 3);
}

// ── ConflictDetector: negation ──────────────────────────────────────────────────

#[test]
fn test_detect_negation_conflict() {
    let docs = vec![
        doc("a", "The new drug treatment was approved by regulators."),
        doc(
            "b",
            "The new drug treatment was not approved by regulators.",
        ),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].kind, ConflictKind::Negation);
    assert_eq!(conflicts[0].passage_a, 0);
    assert_eq!(conflicts[0].passage_b, 1);
}

#[test]
fn test_detect_negation_contraction() {
    let docs = vec![
        doc("a", "The bridge project remains fully funded this year."),
        doc("b", "The bridge project isn't fully funded this year."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].kind, ConflictKind::Negation);
}

#[test]
fn test_detect_no_negation_when_both_negated() {
    let docs = vec![
        doc("a", "The merger was not completed last quarter."),
        doc("b", "The merger was not completed last quarter."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

// ── ConflictDetector: numeric ───────────────────────────────────────────────────

#[test]
fn test_detect_numeric_conflict() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].kind, ConflictKind::Numeric);
}

#[test]
fn test_detect_numeric_agree_no_conflict() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 5 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

#[test]
fn test_detect_numeric_decimal_conflict() {
    let docs = vec![
        doc("a", "The reported failure rate stands at 2.5 percent."),
        doc("b", "The reported failure rate stands at 7.5 percent."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].kind, ConflictKind::Numeric);
}

// ── ConflictDetector: temporal ──────────────────────────────────────────────────

#[test]
fn test_detect_temporal_conflict() {
    let docs = vec![
        doc(
            "a",
            "The company headquarters building was founded in 1990.",
        ),
        doc(
            "b",
            "The company headquarters building was founded in 2005.",
        ),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].kind, ConflictKind::Temporal);
}

#[test]
fn test_detect_temporal_same_year_no_conflict() {
    let docs = vec![
        doc(
            "a",
            "The company headquarters building was founded in 1990.",
        ),
        doc(
            "b",
            "The company headquarters building was founded in 1990.",
        ),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

#[test]
fn test_detect_temporal_not_classified_as_numeric() {
    // A pure year difference must be Temporal, never Numeric.
    let docs = vec![
        doc("a", "The peace treaty agreement was signed in 1815."),
        doc("b", "The peace treaty agreement was signed in 1919."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].kind, ConflictKind::Temporal);
}

// ── ConflictDetector: no conflict / subject differences ─────────────────────────

#[test]
fn test_detect_no_conflict_different_subjects() {
    let docs = vec![
        doc(
            "a",
            "The river delta region flooded during the heavy storm.",
        ),
        doc("b", "The mountain ski resort opened earlier than usual."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

#[test]
fn test_detect_no_conflict_when_claims_agree() {
    let docs = vec![
        doc(
            "a",
            "The spacecraft successfully entered the planetary orbit.",
        ),
        doc(
            "b",
            "The spacecraft successfully entered the planetary orbit.",
        ),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

#[test]
fn test_detect_numeric_different_subjects_no_conflict() {
    // Numbers differ but the sentences share no subject vocabulary.
    let docs = vec![
        doc("a", "Apples cost 5 dollars."),
        doc("b", "Bicycles weigh 8 kilograms."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

// ── ConflictDetector: corpus size and checked variant ───────────────────────────

#[test]
fn test_detect_empty_corpus_returns_empty() {
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&[]);
    assert!(conflicts.is_empty());
}

#[test]
fn test_detect_single_doc_returns_empty() {
    let docs = vec![doc("a", "The population is 5 million people.")];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

#[test]
fn test_detect_checked_errors_on_empty() {
    let detector = ConflictDetector::default();
    let result = detector.detect_checked(&[]);
    assert!(matches!(result, Err(KnowledgeConflictError::EmptyCorpus)));
}

#[test]
fn test_detect_checked_ok_on_nonempty() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let result = detector.detect_checked(&docs);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_detect_checked_single_doc_ok_empty() {
    let docs = vec![doc("a", "Only one document here.")];
    let detector = ConflictDetector::default();
    let result = detector.detect_checked(&docs);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_error_display() {
    let err = KnowledgeConflictError::EmptyCorpus;
    assert_eq!(err.to_string(), "corpus is empty");
}

// ── ConflictDetector: min_shared_terms gating ───────────────────────────────────

#[test]
fn test_min_shared_terms_gating_blocks() {
    // The two sentences share exactly one content token ("voltage"); the
    // default gate of 2 blocks the otherwise-numeric mismatch.
    let docs = vec![
        doc("a", "Voltage climbed 5 yesterday."),
        doc("b", "Voltage dropped 8 afterwards."),
    ];
    let detector = ConflictDetector::default();
    assert_eq!(detector.config.min_shared_terms, 2);
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

#[test]
fn test_min_shared_terms_gating_allows_with_lower_threshold() {
    // Same single-shared-token sentences, now permitted by a threshold of 1.
    let docs = vec![
        doc("a", "Voltage climbed 5 yesterday."),
        doc("b", "Voltage dropped 8 afterwards."),
    ];
    let config = KnowledgeConflictConfig::new().with_min_shared_terms(1);
    let detector = ConflictDetector::new(config);
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].kind, ConflictKind::Numeric);
}

#[test]
fn test_min_shared_terms_high_threshold_blocks_real_conflict() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    // Require more shared terms than the sentences actually share.
    let config = KnowledgeConflictConfig::new().with_min_shared_terms(20);
    let detector = ConflictDetector::new(config);
    let conflicts = detector.detect(&docs);
    assert!(conflicts.is_empty());
}

// ── ConflictDetector: multiple passages / multiple conflicts ────────────────────

#[test]
fn test_detect_across_three_passages() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
        doc("c", "The city population is 12 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    // (0,1), (0,2), (1,2) all disagree numerically.
    assert_eq!(conflicts.len(), 3);
    assert!(conflicts.iter().all(|c| c.kind == ConflictKind::Numeric));
}

#[test]
fn test_detect_indices_are_ordered() {
    let docs = vec![
        doc("a", "The reactor core temperature reached 5 units."),
        doc("b", "The reactor core temperature reached 8 units."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert!(conflicts[0].passage_a < conflicts[0].passage_b);
}

// ── ConflictResolver: defaults ──────────────────────────────────────────────────

#[test]
fn test_resolver_default() {
    let resolver = ConflictResolver::default();
    assert_eq!(resolver.config.policy, ConflictPolicy::Majority);
}

#[test]
fn test_resolver_new_holds_config() {
    let config = KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency);
    let resolver = ConflictResolver::new(config);
    assert_eq!(resolver.config.policy, ConflictPolicy::Recency);
}

// ── ConflictResolver: Majority ──────────────────────────────────────────────────

#[test]
fn test_resolve_majority_a_side_wins() {
    // claim_a ("Northgate stadium ...") is corroborated by two further passages
    // that share its distinctive subject; claim_b ("Riverside arena ...") has no
    // external corroboration, so side A wins 2 vs 0.
    let docs = vec![
        doc("a", "Northgate stadium seats 5 thousand."),
        doc("b", "Riverside arena seats 8 thousand."),
        doc("c", "Northgate stadium hosts events."),
        doc("d", "Northgate stadium opened recently."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    // Find the (0,1) conflict between the two seating claims.
    let conflict = conflicts
        .iter()
        .find(|c| c.passage_a == 0 && c.passage_b == 1)
        .expect("expected a 0-1 conflict");
    assert_eq!(conflict.kind, ConflictKind::Numeric);

    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Majority));
    let resolution = resolver.resolve(conflict, &docs, &empty_ages(), &empty_auth());
    // Side A (passage 0) is corroborated by passages c and d; B by none.
    assert_eq!(resolution.winner, 0);
    assert_eq!(resolution.losers, vec![1]);
    assert_eq!(resolution.policy, ConflictPolicy::Majority);
    assert!(!resolution.rationale.is_empty());
}

#[test]
fn test_resolve_majority_b_side_wins() {
    // Mirror image: side B ("Riverside arena ...") is the corroborated one.
    let docs = vec![
        doc("a", "Northgate stadium seats 5 thousand."),
        doc("b", "Riverside arena seats 8 thousand."),
        doc("c", "Riverside arena hosts events."),
        doc("d", "Riverside arena opened recently."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    let conflict = conflicts
        .iter()
        .find(|c| c.passage_a == 0 && c.passage_b == 1)
        .expect("expected a 0-1 conflict");
    assert_eq!(conflict.kind, ConflictKind::Numeric);

    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Majority));
    let resolution = resolver.resolve(conflict, &docs, &empty_ages(), &empty_auth());
    // Passage 1's "Riverside arena" subject is corroborated by c and d; A by none.
    assert_eq!(resolution.winner, 1);
    assert_eq!(resolution.losers, vec![0]);
}

#[test]
fn test_resolve_majority_tie_favours_a() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Majority));
    let resolution = resolver.resolve(&conflicts[0], &docs, &empty_ages(), &empty_auth());
    // No external corroboration on either side: tie favours passage_a.
    assert_eq!(resolution.winner, 0);
    assert_eq!(resolution.losers, vec![1]);
}

// ── ConflictResolver: Authority ─────────────────────────────────────────────────

#[test]
fn test_resolve_authority_higher_wins() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);

    let resolver = ConflictResolver::new(
        KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Authority),
    );
    let authority = authorities(&[("a", 0.3), ("b", 0.9)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &empty_ages(), &authority);
    assert_eq!(resolution.winner, 1);
    assert_eq!(resolution.losers, vec![0]);
    assert_eq!(resolution.policy, ConflictPolicy::Authority);
    assert!(!resolution.rationale.is_empty());
}

#[test]
fn test_resolve_authority_a_wins() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);

    let resolver = ConflictResolver::new(
        KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Authority),
    );
    let authority = authorities(&[("a", 0.95), ("b", 0.2)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &empty_ages(), &authority);
    assert_eq!(resolution.winner, 0);
    assert_eq!(resolution.losers, vec![1]);
}

#[test]
fn test_resolve_authority_missing_entry_loses() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);

    let resolver = ConflictResolver::new(
        KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Authority),
    );
    // Only B has an authority score; A defaults to the minimum and loses.
    let authority = authorities(&[("b", 0.1)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &empty_ages(), &authority);
    assert_eq!(resolution.winner, 1);
}

#[test]
fn test_resolve_authority_tie_favours_a() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);

    let resolver = ConflictResolver::new(
        KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Authority),
    );
    let authority = authorities(&[("a", 0.5), ("b", 0.5)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &empty_ages(), &authority);
    assert_eq!(resolution.winner, 0);
}

// ── ConflictResolver: Recency ───────────────────────────────────────────────────

#[test]
fn test_resolve_recency_lower_age_wins() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);

    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency));
    // B is much newer (lower age).
    let ages_map = ages(&[("a", 800.0), ("b", 30.0)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &ages_map, &empty_auth());
    assert_eq!(resolution.winner, 1);
    assert_eq!(resolution.losers, vec![0]);
    assert_eq!(resolution.policy, ConflictPolicy::Recency);
    assert!(!resolution.rationale.is_empty());
}

#[test]
fn test_resolve_recency_a_newer_wins() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);

    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency));
    let ages_map = ages(&[("a", 10.0), ("b", 900.0)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &ages_map, &empty_auth());
    assert_eq!(resolution.winner, 0);
    assert_eq!(resolution.losers, vec![1]);
}

#[test]
fn test_resolve_recency_missing_age_loses() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);

    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency));
    // Only B has a recorded age; A defaults to +inf and loses.
    let ages_map = ages(&[("b", 500.0)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &ages_map, &empty_auth());
    assert_eq!(resolution.winner, 1);
}

#[test]
fn test_resolve_recency_tie_favours_a() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);

    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency));
    let ages_map = ages(&[("a", 100.0), ("b", 100.0)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &ages_map, &empty_auth());
    assert_eq!(resolution.winner, 0);
}

// ── ConflictResolver: resolve_all ───────────────────────────────────────────────

#[test]
fn test_resolve_all_preserves_order_and_count() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
        doc("c", "The city population is 12 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 3);

    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency));
    let ages_map = ages(&[("a", 30.0), ("b", 60.0), ("c", 90.0)]);
    let resolutions = resolver.resolve_all(&conflicts, &docs, &ages_map, &empty_auth());
    assert_eq!(resolutions.len(), 3);
    // For every pair, the lower-indexed (newer) passage wins under these ages.
    for (conflict, resolution) in conflicts.iter().zip(&resolutions) {
        assert_eq!(resolution.winner, conflict.passage_a);
    }
}

#[test]
fn test_resolve_all_empty_conflicts() {
    let docs = vec![doc("a", "Nothing conflicting here at all today.")];
    let resolver = ConflictResolver::default();
    let resolutions = resolver.resolve_all(&[], &docs, &empty_ages(), &empty_auth());
    assert!(resolutions.is_empty());
}

// ── rationale / policy field correctness ────────────────────────────────────────

#[test]
fn test_rationale_non_empty_all_policies() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    let ages_map = ages(&[("a", 10.0), ("b", 20.0)]);
    let authority = authorities(&[("a", 0.4), ("b", 0.6)]);

    for policy in [
        ConflictPolicy::Recency,
        ConflictPolicy::Authority,
        ConflictPolicy::Majority,
    ] {
        let resolver = ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(policy));
        let resolution = resolver.resolve(&conflicts[0], &docs, &ages_map, &authority);
        assert!(!resolution.rationale.is_empty());
        assert_eq!(resolution.policy, policy);
    }
}

#[test]
fn test_resolution_policy_matches_config() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    let resolver = ConflictResolver::new(
        KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Authority),
    );
    let authority = authorities(&[("a", 0.1), ("b", 0.2)]);
    let resolution = resolver.resolve(&conflicts[0], &docs, &empty_ages(), &authority);
    assert_eq!(resolution.policy, ConflictPolicy::Authority);
}

// ── determinism ─────────────────────────────────────────────────────────────────

#[test]
fn test_detect_is_deterministic() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
        doc("c", "The city population is 12 million people."),
    ];
    let detector = ConflictDetector::default();
    let first = detector.detect(&docs);
    let second = detector.detect(&docs);
    assert_eq!(first, second);
}

#[test]
fn test_resolve_is_deterministic() {
    let docs = vec![
        doc("a", "The city population is 5 million people."),
        doc("b", "The city population is 8 million people."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    let resolver =
        ConflictResolver::new(KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Majority));
    let first = resolver.resolve(&conflicts[0], &docs, &empty_ages(), &empty_auth());
    let second = resolver.resolve(&conflicts[0], &docs, &empty_ages(), &empty_auth());
    assert_eq!(first, second);
}

#[test]
fn test_resolve_all_is_deterministic() {
    let docs = vec![
        doc("a", "The reactor output level reached 5 units."),
        doc("b", "The reactor output level reached 8 units."),
        doc("c", "The reactor output level reached 11 units."),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    let resolver = ConflictResolver::new(
        KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Authority),
    );
    let authority = authorities(&[("a", 0.2), ("b", 0.5), ("c", 0.8)]);
    let first = resolver.resolve_all(&conflicts, &docs, &empty_ages(), &authority);
    let second = resolver.resolve_all(&conflicts, &docs, &empty_ages(), &authority);
    assert_eq!(first, second);
}

// ── end-to-end ──────────────────────────────────────────────────────────────────

#[test]
fn test_end_to_end_detect_then_resolve_recency() {
    let docs = vec![
        doc(
            "old",
            "The legislative budget proposal allocated 5 billion dollars.",
        ),
        doc(
            "new",
            "The legislative budget proposal allocated 9 billion dollars.",
        ),
    ];
    let config = KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency);
    let detector = ConflictDetector::new(config.clone());
    let conflicts = detector.detect(&docs);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].kind, ConflictKind::Numeric);

    let resolver = ConflictResolver::new(config);
    let ages_map = ages(&[("old", 1000.0), ("new", 5.0)]);
    let resolutions = resolver.resolve_all(&conflicts, &docs, &ages_map, &empty_auth());
    assert_eq!(resolutions.len(), 1);
    assert_eq!(resolutions[0].winner, 1);
}

#[test]
fn test_end_to_end_mixed_conflict_kinds() {
    let docs = vec![
        doc(
            "a",
            "The flagship product launch happened in 2018. The product was praised widely.",
        ),
        doc(
            "b",
            "The flagship product launch happened in 2021. The product was not praised at all.",
        ),
    ];
    let detector = ConflictDetector::default();
    let conflicts = detector.detect(&docs);
    // Expect at least one temporal and one negation conflict between the docs.
    assert!(conflicts.iter().any(|c| c.kind == ConflictKind::Temporal));
    assert!(conflicts.iter().any(|c| c.kind == ConflictKind::Negation));
}
