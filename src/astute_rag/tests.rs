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
//! Tests for the `astute_rag` module.

use crate::astute_rag::consolidator::AstuteConsolidator;
use crate::astute_rag::types::{
    AstuteConfig, AstuteError, ConsolidatedKnowledge, InternalKnowledge, KnowledgeConflict,
    KnowledgeSource, KnowledgeStatement, MockInternalKnowledge,
};
use crate::types::{Document, DocumentId};

// ── helpers ──────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

fn default_consolidator() -> AstuteConsolidator {
    AstuteConsolidator::new(AstuteConfig::default())
}

// ── AstuteConfig: defaults & builders ──────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = AstuteConfig::default();
    assert_eq!(cfg.reliability_threshold, 0.5);
    assert!(cfg.prefer_external);
}

#[test]
fn config_new_matches_default() {
    let a = AstuteConfig::new();
    let b = AstuteConfig::default();
    assert_eq!(a.reliability_threshold, b.reliability_threshold);
    assert_eq!(a.prefer_external, b.prefer_external);
}

#[test]
fn config_with_reliability_threshold() {
    let cfg = AstuteConfig::new().with_reliability_threshold(0.8);
    assert_eq!(cfg.reliability_threshold, 0.8);
    assert!(cfg.prefer_external);
}

#[test]
fn config_with_prefer_external() {
    let cfg = AstuteConfig::new().with_prefer_external(false);
    assert!(!cfg.prefer_external);
    assert_eq!(cfg.reliability_threshold, 0.5);
}

#[test]
fn config_builders_chain() {
    let cfg = AstuteConfig::new()
        .with_reliability_threshold(0.3)
        .with_prefer_external(false);
    assert_eq!(cfg.reliability_threshold, 0.3);
    assert!(!cfg.prefer_external);
}

#[test]
fn config_is_clone() {
    let cfg = AstuteConfig::new().with_reliability_threshold(0.7);
    let cloned = cfg.clone();
    assert_eq!(cloned.reliability_threshold, 0.7);
}

// ── KnowledgeSource ────────────────────────────────────────────────────────────

#[test]
fn source_as_str() {
    assert_eq!(KnowledgeSource::Internal.as_str(), "internal");
    assert_eq!(
        KnowledgeSource::External(DocumentId::from_string("x")).as_str(),
        "external"
    );
}

#[test]
fn source_is_internal_external() {
    assert!(KnowledgeSource::Internal.is_internal());
    assert!(!KnowledgeSource::Internal.is_external());
    let ext = KnowledgeSource::External(DocumentId::from_string("x"));
    assert!(ext.is_external());
    assert!(!ext.is_internal());
}

#[test]
fn source_document_id() {
    assert!(KnowledgeSource::Internal.document_id().is_none());
    let id = DocumentId::from_string("doc-7");
    let ext = KnowledgeSource::External(id.clone());
    assert_eq!(ext.document_id(), Some(&id));
}

#[test]
fn source_equality() {
    let a = KnowledgeSource::External(DocumentId::from_string("d"));
    let b = KnowledgeSource::External(DocumentId::from_string("d"));
    assert_eq!(a, b);
    assert_ne!(a, KnowledgeSource::Internal);
}

// ── KnowledgeStatement ─────────────────────────────────────────────────────────

#[test]
fn statement_new_and_accessors() {
    let s = KnowledgeStatement::new("hello", KnowledgeSource::Internal, 0.9);
    assert_eq!(s.content, "hello");
    assert!(s.is_internal());
    assert_eq!(s.reliability, 0.9);
}

#[test]
fn statement_internal_constructor() {
    let s = KnowledgeStatement::internal("parametric fact", 1.0);
    assert!(s.is_internal());
    assert!(s.source.is_internal());
}

#[test]
fn statement_external_constructor() {
    let s = KnowledgeStatement::external("retrieved fact", DocumentId::from_string("d1"), 0.5);
    assert!(!s.is_internal());
    assert_eq!(s.source.document_id(), Some(&DocumentId::from_string("d1")));
}

// ── KnowledgeConflict ──────────────────────────────────────────────────────────

#[test]
fn conflict_new_and_fields() {
    let c = KnowledgeConflict::new("a", "b", KnowledgeSource::Internal);
    assert_eq!(c.internal, "a");
    assert_eq!(c.external, "b");
    assert!(!c.resolved_external());
}

#[test]
fn conflict_resolved_external() {
    let c = KnowledgeConflict::new(
        "a",
        "b",
        KnowledgeSource::External(DocumentId::from_string("d")),
    );
    assert!(c.resolved_external());
}

// ── ConsolidatedKnowledge ──────────────────────────────────────────────────────

#[test]
fn consolidated_counts() {
    let stmts = vec![
        KnowledgeStatement::internal("a", 1.0),
        KnowledgeStatement::external("b", DocumentId::from_string("d"), 0.5),
    ];
    let conflicts = vec![KnowledgeConflict::new("a", "b", KnowledgeSource::Internal)];
    let ck = ConsolidatedKnowledge::new(stmts, conflicts, "answer");
    assert_eq!(ck.statement_count(), 2);
    assert_eq!(ck.conflict_count(), 1);
    assert!(ck.has_conflicts());
    assert_eq!(ck.answer, "answer");
}

#[test]
fn consolidated_no_conflicts() {
    let ck = ConsolidatedKnowledge::new(vec![], vec![], "x");
    assert!(!ck.has_conflicts());
    assert_eq!(ck.conflict_count(), 0);
}

// ── MockInternalKnowledge ──────────────────────────────────────────────────────

#[test]
fn mock_recall_returns_statements() {
    let mock = MockInternalKnowledge::new(vec!["one".to_string(), "two".to_string()]);
    let recalled = mock.recall("anything");
    assert_eq!(recalled, vec!["one".to_string(), "two".to_string()]);
}

#[test]
fn mock_recall_ignores_query() {
    let mock = MockInternalKnowledge::new(vec!["fixed".to_string()]);
    assert_eq!(mock.recall("query a"), mock.recall("query b"));
}

#[test]
fn mock_with_statement_builder() {
    let mock = MockInternalKnowledge::default()
        .with_statement("a")
        .with_statement("b");
    assert_eq!(mock.statements.len(), 2);
    assert_eq!(mock.recall("q").len(), 2);
}

#[test]
fn mock_default_is_empty() {
    let mock = MockInternalKnowledge::default();
    assert!(mock.recall("q").is_empty());
}

// ── external_statements: extraction & corroboration ─────────────────────────────

#[test]
fn external_statements_extracts_sentences() {
    let c = default_consolidator();
    let docs = vec![doc(
        "d1",
        "Paris is the capital of France. The Seine flows through Paris.",
    )];
    let stmts = c.external_statements(&docs);
    assert_eq!(stmts.len(), 2);
    assert!(stmts.iter().all(|s| !s.is_internal()));
}

#[test]
fn external_statements_tags_source_id() {
    let c = default_consolidator();
    let docs = vec![doc("doc-abc", "Water boils at 100 degrees.")];
    let stmts = c.external_statements(&docs);
    assert_eq!(stmts.len(), 1);
    assert_eq!(
        stmts[0].source.document_id(),
        Some(&DocumentId::from_string("doc-abc"))
    );
}

#[test]
fn external_statements_empty_docs() {
    let c = default_consolidator();
    let stmts = c.external_statements(&[]);
    assert!(stmts.is_empty());
}

#[test]
fn external_statements_skips_subjectless_sentences() {
    let c = default_consolidator();
    // "It is so." → no content-bearing subject terms.
    let docs = vec![doc("d1", "It is so.")];
    let stmts = c.external_statements(&docs);
    assert!(stmts.is_empty());
}

#[test]
fn external_statements_single_doc_reliability() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The mountain is tall.")];
    let stmts = c.external_statements(&docs);
    // Single doc: corroboration = 1/1 = 1.0 (its own support).
    assert_eq!(stmts.len(), 1);
    assert_eq!(stmts[0].reliability, 1.0);
}

#[test]
fn external_statements_corroboration_boosts_reliability() {
    let c = default_consolidator();
    // Claim about "comet halley 1986" appears verbatim in two of three docs.
    let docs = vec![
        doc("d1", "Comet Halley appeared in 1986."),
        doc("d2", "Comet Halley appeared in 1986 across the sky."),
        doc("d3", "Bananas are yellow fruit."),
    ];
    let stmts = c.external_statements(&docs);
    // Find the statement from d1 about the comet.
    let halley = stmts
        .iter()
        .find(|s| s.content.to_lowercase().contains("comet") && s.content.contains("1986"))
        .expect("halley statement present");
    // Supported by d1 (self) and d2 → 2/3 ≈ 0.667, higher than an unsupported 1/3.
    assert!(halley.reliability > 0.6);
}

#[test]
fn external_statements_uncorroborated_low_reliability() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "Zorblax invented the quantum widget."),
        doc("d2", "Apples grow on trees."),
        doc("d3", "Rivers flow to the sea."),
    ];
    let stmts = c.external_statements(&docs);
    let zorblax = stmts
        .iter()
        .find(|s| s.content.to_lowercase().contains("zorblax"))
        .expect("zorblax statement present");
    // Only supported by its own doc → 1/3 ≈ 0.333.
    assert!(zorblax.reliability < 0.5);
}

#[test]
fn external_statements_more_corroboration_means_higher_reliability() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "The bridge spans the wide river valley."),
        doc("d2", "The bridge spans the wide river valley too."),
        doc("d3", "The bridge spans the wide river valley as well."),
        doc("d4", "Cats sleep often."),
    ];
    let bridge = c
        .external_statements(&docs)
        .into_iter()
        .find(|s| s.content.to_lowercase().contains("bridge"))
        .expect("bridge statement present");
    // Supported by d1, d2, d3 → 3/4 = 0.75.
    assert!(bridge.reliability >= 0.7);
}

// ── detect_conflict ─────────────────────────────────────────────────────────────

#[test]
fn detect_conflict_negation_mismatch_true() {
    let c = default_consolidator();
    assert!(c.detect_conflict(
        "The vaccine is effective against the virus.",
        "The vaccine is not effective against the virus."
    ));
}

#[test]
fn detect_conflict_number_mismatch_true() {
    let c = default_consolidator();
    assert!(c.detect_conflict(
        "The distance to the station is 20 kilometers.",
        "The distance to the station is 50 kilometers."
    ));
}

#[test]
fn detect_conflict_agreement_false() {
    let c = default_consolidator();
    assert!(!c.detect_conflict(
        "The capital of France is Paris.",
        "Paris is the capital of France."
    ));
}

#[test]
fn detect_conflict_no_shared_subject_false() {
    let c = default_consolidator();
    // Negation present, but no shared subject term → not a conflict.
    assert!(!c.detect_conflict("The ocean is deep.", "The mountain is not tall."));
}

#[test]
fn detect_conflict_same_number_false() {
    let c = default_consolidator();
    assert!(!c.detect_conflict(
        "The tower is 300 meters tall.",
        "The famous tower reaches 300 meters."
    ));
}

#[test]
fn detect_conflict_both_negated_false() {
    let c = default_consolidator();
    // Both negated → no XOR mismatch, and no numbers → no conflict.
    assert!(!c.detect_conflict(
        "The bridge is not unsafe today.",
        "The bridge is not closed today."
    ));
}

#[test]
fn detect_conflict_shared_subject_one_negated() {
    let c = default_consolidator();
    assert!(c.detect_conflict("Pluto is a planet.", "Pluto is not a planet."));
}

#[test]
fn detect_conflict_unrelated_no_negation_no_number_false() {
    let c = default_consolidator();
    assert!(!c.detect_conflict("The library opens early.", "The garden blooms in spring."));
}

// ── consolidate: merging non-conflicting knowledge ──────────────────────────────

#[test]
fn consolidate_merges_internal_and_external() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The Seine flows through Paris.")];
    let internal = vec!["Paris is the capital of France.".to_string()];
    let result = c
        .consolidate("tell me about Paris", &internal, &docs)
        .expect("consolidation succeeds");
    // Both the internal claim and the external statement survive (no conflict).
    assert!(
        result
            .statements
            .iter()
            .any(KnowledgeStatement::is_internal)
    );
    assert!(result.statements.iter().any(|s| !s.is_internal()));
    assert!(!result.has_conflicts());
}

#[test]
fn consolidate_keeps_all_external_when_no_internal() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The river is long. The mountain is high.")];
    let result = c.consolidate("geography", &[], &docs).expect("succeeds");
    assert_eq!(result.statements.len(), 2);
    assert!(result.statements.iter().all(|s| !s.is_internal()));
    assert!(!result.has_conflicts());
}

#[test]
fn consolidate_keeps_internal_when_no_docs() {
    let c = default_consolidator();
    let internal = vec!["The sun is a star.".to_string()];
    let result = c
        .consolidate("astronomy", &internal, &[])
        .expect("succeeds");
    assert_eq!(result.statements.len(), 1);
    assert!(result.statements[0].is_internal());
}

#[test]
fn consolidate_unrelated_internal_and_external_both_kept() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "Photosynthesis converts sunlight to energy.")];
    let internal = vec!["The economy grew last quarter.".to_string()];
    let result = c.consolidate("mixed topics", &internal, &docs).unwrap();
    assert!(!result.has_conflicts());
    assert!(result.statements.len() >= 2);
}

// ── consolidate: conflict resolution (prefer_external default) ───────────────────

#[test]
fn consolidate_conflict_resolves_to_external_by_default() {
    let c = default_consolidator();
    // External claim corroborated across two docs (reliability >= 0.5).
    let docs = vec![
        doc("d1", "The tower was completed in 1889."),
        doc("d2", "The tower was completed in 1889 indeed."),
    ];
    let internal = vec!["The tower was completed in 1989.".to_string()];
    let result = c
        .consolidate("when was the tower completed", &internal, &docs)
        .unwrap();
    assert!(result.has_conflicts());
    assert!(result.conflicts[0].resolved_external());
    // The winning statement should be the external one (1889).
    assert!(
        result
            .statements
            .iter()
            .any(|s| { !s.is_internal() && s.content.contains("1889") })
    );
}

#[test]
fn consolidate_conflict_negation_resolves_external() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "The treatment is not effective for this disease."),
        doc(
            "d2",
            "The treatment is not effective for this disease at all.",
        ),
    ];
    let internal = vec!["The treatment is effective for this disease.".to_string()];
    let result = c
        .consolidate("treatment efficacy", &internal, &docs)
        .unwrap();
    assert!(result.has_conflicts());
    assert!(result.conflicts[0].resolved_external());
}

#[test]
fn consolidate_conflict_records_both_claims() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "The population is 8 million people."),
        doc("d2", "The population is 8 million people now."),
    ];
    let internal = vec!["The population is 5 million people.".to_string()];
    let result = c.consolidate("population", &internal, &docs).unwrap();
    assert_eq!(result.conflicts.len(), 1);
    assert!(result.conflicts[0].internal.contains('5'));
    assert!(result.conflicts[0].external.contains('8'));
}

// ── consolidate: prefer_external = false flips low-reliability case ──────────────

#[test]
fn consolidate_prefer_external_false_flips_to_internal() {
    // Uncorroborated external claim (single doc among several unrelated) →
    // with prefer_external=false, internal wins.
    let c = AstuteConsolidator::new(AstuteConfig::new().with_prefer_external(false));
    let docs = vec![
        doc("d1", "The artifact is 12 centuries old."),
        doc("d2", "Birds migrate south in winter."),
        doc("d3", "Salt dissolves in water."),
    ];
    let internal = vec!["The artifact is 9 centuries old.".to_string()];
    let result = c.consolidate("artifact age", &internal, &docs).unwrap();
    assert!(result.has_conflicts());
    assert!(!result.conflicts[0].resolved_external());
    // Internal statement (9) should be among the winners.
    assert!(
        result
            .statements
            .iter()
            .any(|s| { s.is_internal() && s.content.contains('9') })
    );
}

#[test]
fn consolidate_prefer_external_false_still_external_when_uncorroborated_loses() {
    // With prefer_external=false, the external claim never wins a conflict,
    // regardless of reliability.
    let c = AstuteConsolidator::new(AstuteConfig::new().with_prefer_external(false));
    let docs = vec![
        doc("d1", "The river is 100 kilometers long."),
        doc("d2", "The river is 100 kilometers long for sure."),
    ];
    let internal = vec!["The river is 200 kilometers long.".to_string()];
    let result = c.consolidate("river length", &internal, &docs).unwrap();
    assert!(result.has_conflicts());
    assert!(!result.conflicts[0].resolved_external());
}

#[test]
fn consolidate_low_reliability_external_loses_even_with_prefer_external() {
    // prefer_external=true but the external claim is uncorroborated AND below a
    // raised reliability threshold → internal wins.
    let cfg = AstuteConfig::new().with_reliability_threshold(0.9);
    let c = AstuteConsolidator::new(cfg);
    let docs = vec![
        doc("d1", "The crystal weighs 7 grams."),
        doc("d2", "Trees lose leaves in autumn."),
        doc("d3", "The kettle boils quickly."),
    ];
    let internal = vec!["The crystal weighs 3 grams.".to_string()];
    let result = c.consolidate("crystal weight", &internal, &docs).unwrap();
    assert!(result.has_conflicts());
    // External reliability ≈ 1/3 < 0.9 → internal wins.
    assert!(!result.conflicts[0].resolved_external());
}

#[test]
fn consolidate_corroborated_external_wins_above_threshold() {
    // High threshold, but corroboration pushes reliability above it.
    let cfg = AstuteConfig::new().with_reliability_threshold(0.6);
    let c = AstuteConsolidator::new(cfg);
    let docs = vec![
        doc("d1", "The eclipse occurred in 2017."),
        doc("d2", "The eclipse occurred in 2017 over the country."),
        doc("d3", "The eclipse occurred in 2017 as recorded."),
    ];
    let internal = vec!["The eclipse occurred in 1999.".to_string()];
    let result = c.consolidate("eclipse year", &internal, &docs).unwrap();
    assert!(result.has_conflicts());
    // Reliability 3/3 = 1.0 >= 0.6 → external wins.
    assert!(result.conflicts[0].resolved_external());
}

// ── answer synthesis ────────────────────────────────────────────────────────────

#[test]
fn answer_is_non_empty() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The lake is deep.")];
    let result = c.consolidate("lake", &[], &docs).unwrap();
    assert!(!result.answer.is_empty());
}

#[test]
fn answer_mentions_query() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The forest is dense.")];
    let result = c.consolidate("describe the forest", &[], &docs).unwrap();
    assert!(result.answer.contains("describe the forest"));
}

#[test]
fn answer_attributes_sources() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The canyon is vast.")];
    let internal = vec!["The desert is dry.".to_string()];
    let result = c.consolidate("nature", &internal, &docs).unwrap();
    assert!(result.answer.contains("[internal]"));
    assert!(result.answer.contains("[external]"));
}

#[test]
fn answer_synthesized_from_winners() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "The comet passed in 1986."),
        doc("d2", "The comet passed in 1986 brightly."),
    ];
    let internal = vec!["The comet passed in 1066.".to_string()];
    let result = c.consolidate("comet", &internal, &docs).unwrap();
    // External (1986) wins → answer contains the winning external value.
    assert!(result.answer.contains("1986"));
    // The losing internal value should not appear as a winning statement line.
    assert!(
        !result
            .answer
            .contains("[internal] The comet passed in 1066")
    );
}

#[test]
fn answer_includes_external_winner_attribution() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "The summit is 4000 meters."),
        doc("d2", "The summit is 4000 meters tall."),
    ];
    let internal = vec!["The summit is 2000 meters.".to_string()];
    let result = c.consolidate("summit height", &internal, &docs).unwrap();
    assert!(result.answer.contains("[external]"));
    assert!(result.answer.contains("4000"));
}

// ── run() with InternalKnowledge ────────────────────────────────────────────────

#[test]
fn run_uses_model_recall() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The planet has rings.")];
    let model = MockInternalKnowledge::new(vec!["The planet is gaseous.".to_string()]);
    let result = c.run("planet facts", &model, &docs).unwrap();
    assert!(
        result
            .statements
            .iter()
            .any(KnowledgeStatement::is_internal)
    );
    assert!(result.statements.iter().any(|s| !s.is_internal()));
}

#[test]
fn run_with_empty_model_keeps_external() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The valley is green.")];
    let model = MockInternalKnowledge::default();
    let result = c.run("valley", &model, &docs).unwrap();
    assert!(result.statements.iter().all(|s| !s.is_internal()));
    assert!(!result.statements.is_empty());
}

#[test]
fn run_detects_conflict_via_model() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "The element melts at 1500 degrees."),
        doc("d2", "The element melts at 1500 degrees consistently."),
    ];
    let model = MockInternalKnowledge::new(vec!["The element melts at 900 degrees.".to_string()]);
    let result = c.run("melting point", &model, &docs).unwrap();
    assert!(result.has_conflicts());
}

// ── error cases ─────────────────────────────────────────────────────────────────

#[test]
fn consolidate_empty_query_errors() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "Something useful.")];
    let err = c.consolidate("", &["fact".to_string()], &docs).unwrap_err();
    assert!(matches!(err, AstuteError::EmptyQuery));
}

#[test]
fn consolidate_whitespace_query_errors() {
    let c = default_consolidator();
    let err = c
        .consolidate("   \t  ", &["fact".to_string()], &[])
        .unwrap_err();
    assert!(matches!(err, AstuteError::EmptyQuery));
}

#[test]
fn consolidate_no_knowledge_errors() {
    let c = default_consolidator();
    let err = c.consolidate("a real query", &[], &[]).unwrap_err();
    assert!(matches!(err, AstuteError::NoKnowledge));
}

#[test]
fn consolidate_blank_internal_and_no_docs_errors() {
    let c = default_consolidator();
    // Internal claims are all blank → treated as no internal knowledge.
    let err = c
        .consolidate("query", &["   ".to_string(), String::new()], &[])
        .unwrap_err();
    assert!(matches!(err, AstuteError::NoKnowledge));
}

#[test]
fn consolidate_subjectless_docs_only_errors() {
    let c = default_consolidator();
    // Document yields no extractable statements and there is no internal claim.
    let docs = vec![doc("d1", "It is.")];
    let err = c.consolidate("query", &[], &docs).unwrap_err();
    assert!(matches!(err, AstuteError::NoKnowledge));
}

#[test]
fn error_display_messages() {
    assert_eq!(
        AstuteError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(
        AstuteError::NoKnowledge.to_string(),
        "no knowledge to consolidate"
    );
}

// ── determinism ─────────────────────────────────────────────────────────────────

#[test]
fn consolidate_is_deterministic() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "The reactor outputs 500 megawatts."),
        doc("d2", "The reactor outputs 500 megawatts steadily."),
        doc("d3", "Clouds drift across the sky."),
    ];
    let internal = vec![
        "The reactor outputs 300 megawatts.".to_string(),
        "The facility is secure.".to_string(),
    ];
    let r1 = c.consolidate("reactor", &internal, &docs).unwrap();
    let r2 = c.consolidate("reactor", &internal, &docs).unwrap();
    assert_eq!(r1.answer, r2.answer);
    assert_eq!(r1.statements.len(), r2.statements.len());
    assert_eq!(r1.conflicts.len(), r2.conflicts.len());
    assert_eq!(
        r1.conflicts[0].resolved_external(),
        r2.conflicts[0].resolved_external()
    );
}

#[test]
fn external_statements_is_deterministic() {
    let c = default_consolidator();
    let docs = vec![
        doc("d1", "Alpha beta gamma. Delta epsilon zeta."),
        doc("d2", "Alpha beta gamma again."),
    ];
    let a = c.external_statements(&docs);
    let b = c.external_statements(&docs);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.content, y.content);
        assert_eq!(x.reliability, y.reliability);
    }
}

#[test]
fn run_is_deterministic() {
    let c = default_consolidator();
    let docs = vec![doc("d1", "The galaxy is spiral. The stars are bright.")];
    let model = MockInternalKnowledge::new(vec!["The universe is vast.".to_string()]);
    let r1 = c.run("cosmos", &model, &docs).unwrap();
    let r2 = c.run("cosmos", &model, &docs).unwrap();
    assert_eq!(r1.answer, r2.answer);
}

// ── consolidator construction ───────────────────────────────────────────────────

#[test]
fn consolidator_default_uses_default_config() {
    let c = AstuteConsolidator::default();
    assert_eq!(c.config.reliability_threshold, 0.5);
    assert!(c.config.prefer_external);
}

#[test]
fn consolidator_new_stores_config() {
    let cfg = AstuteConfig::new().with_reliability_threshold(0.42);
    let c = AstuteConsolidator::new(cfg);
    assert_eq!(c.config.reliability_threshold, 0.42);
}

#[test]
fn consolidator_is_clone() {
    let c = default_consolidator();
    let cloned = c.clone();
    assert_eq!(
        cloned.config.reliability_threshold,
        c.config.reliability_threshold
    );
}

// ── integration: multi-source reconciliation ────────────────────────────────────

#[test]
fn integration_mixed_conflicts_and_agreements() {
    let c = default_consolidator();
    let docs = vec![
        doc(
            "d1",
            "The capital was founded in 1850. The city has 2 million residents.",
        ),
        doc("d2", "The capital was founded in 1850 historically."),
    ];
    let internal = vec![
        "The capital was founded in 1850.".to_string(), // agrees
        "The city has 5 million residents.".to_string(), // conflicts (2 vs 5)
    ];
    let result = c.consolidate("capital city", &internal, &docs).unwrap();
    // Exactly one conflict: the residents count.
    assert_eq!(result.conflicts.len(), 1);
    assert!(result.conflicts[0].external.contains('2'));
    assert!(!result.answer.is_empty());
}

#[test]
fn integration_all_external_corroborated() {
    let c = default_consolidator();
    // Identical claims in every doc → all subject terms corroborated everywhere.
    let docs = vec![
        doc("d1", "The species is endangered worldwide."),
        doc("d2", "The species is endangered worldwide."),
        doc("d3", "The species is endangered worldwide."),
    ];
    let result = c.consolidate("conservation status", &[], &docs).unwrap();
    assert!(!result.has_conflicts());
    // Every external statement about the species should be highly reliable.
    let species: Vec<_> = result
        .statements
        .iter()
        .filter(|s| s.content.to_lowercase().contains("species"))
        .collect();
    assert!(!species.is_empty());
    // Each claim is supported by all three docs → reliability 3/3 = 1.0.
    assert!(species.iter().all(|s| s.reliability >= 0.9));
}
