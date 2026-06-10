//! Tests for the `knowledge_graph_qa` module.

use crate::layer4_graph::types::{EntityType, GraphEntity, GraphRelationship, RelationshipType};

use super::engine::KgqaEngine;
use super::types::{KgqaAnswer, KgqaConfig, KgqaError};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_entity(name: &str, id: &str) -> GraphEntity {
    GraphEntity::new(name, EntityType::Concept).with_id(id)
}

fn make_entity_typed(name: &str, id: &str, et: EntityType) -> GraphEntity {
    GraphEntity::new(name, et).with_id(id)
}

fn make_rel(src: &str, tgt: &str, rt: RelationshipType) -> GraphRelationship {
    GraphRelationship::new(src, tgt, rt)
}

// ── config tests ──────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = KgqaConfig::default();
    assert_eq!(cfg.max_hops, 2);
    assert_eq!(cfg.top_k, 5);
    assert!((cfg.min_confidence - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_config_builders() {
    let cfg = KgqaConfig::new()
        .with_max_hops(3)
        .with_top_k(10)
        .with_min_confidence(0.1);
    assert_eq!(cfg.max_hops, 3);
    assert_eq!(cfg.top_k, 10);
    assert!((cfg.min_confidence - 0.1).abs() < f32::EPSILON);
}

// ── KgqaAnswer tests ──────────────────────────────────────────────────────────

#[test]
fn test_answer_is_confident() {
    let answer = KgqaAnswer::new("Some answer.".to_string(), vec![], vec![], 0.75);
    assert!(answer.is_confident(0.5));
    assert!(!answer.is_confident(0.9));
}

#[test]
fn test_answer_has_triples() {
    use crate::fact_triple::Triple;
    let triple = Triple::new("Rust", "uses", "LLVM", 0.7);
    let with_triples = KgqaAnswer::new("answer".to_string(), vec![], vec![triple], 0.5);
    let without_triples = KgqaAnswer::new("answer".to_string(), vec![], vec![], 0.5);
    assert!(with_triples.has_triples());
    assert!(!without_triples.has_triples());
}

// ── engine guard tests ────────────────────────────────────────────────────────

#[test]
fn test_engine_empty_query_error() {
    let engine = KgqaEngine::default();
    let entities = vec![make_entity("Rust", "e1")];
    let result = engine.answer("", &entities, &[]);
    assert!(matches!(result, Err(KgqaError::EmptyQuery)));
}

#[test]
fn test_engine_no_entities_error() {
    let engine = KgqaEngine::default();
    let result = engine.answer("What is Rust?", &[], &[]);
    assert!(matches!(result, Err(KgqaError::NoEntities)));
}

// ── engine answer tests ───────────────────────────────────────────────────────

#[test]
fn test_engine_single_entity_no_rels() {
    let engine = KgqaEngine::default();
    let entities = vec![make_entity("Rust", "e1")];
    let result = engine.answer("What is Rust?", &entities, &[]);
    assert!(result.is_ok());
    let ans = result.expect("should succeed");
    assert!(!ans.answer.is_empty());
    assert!(!ans.evidence_entities.is_empty());
}

#[test]
fn test_engine_entity_match_exact() {
    let engine = KgqaEngine::default();
    let entities = vec![make_entity("Rust", "e1"), make_entity("Python", "e2")];
    let result = engine.answer("Tell me about Rust", &entities, &[]);
    assert!(result.is_ok());
    let ans = result.expect("should succeed");
    // "Rust" must be in evidence (matched by query).
    assert!(ans.evidence_entities.iter().any(|n| n == "Rust"));
}

#[test]
fn test_engine_entity_match_substring() {
    let engine = KgqaEngine::default();
    let entities = vec![make_entity("Rust", "e1")];
    let result = engine.answer("I love Rust programming", &entities, &[]);
    assert!(result.is_ok());
    let ans = result.expect("should succeed");
    assert!(ans.evidence_entities.contains(&"Rust".to_string()));
}

#[test]
fn test_engine_two_entity_relationship() {
    let engine = KgqaEngine::default();
    let entities = vec![make_entity("Rust", "e1"), make_entity("LLVM", "e2")];
    let rels = vec![make_rel("e1", "e2", RelationshipType::Uses)];
    let result = engine.answer("How does Rust work?", &entities, &rels);
    assert!(result.is_ok());
    let ans = result.expect("should succeed");
    assert!(ans.answer.contains("Rust") || ans.answer.contains("LLVM"));
}

#[test]
fn test_engine_bfs_hop_expands_neighbors() {
    let engine = KgqaEngine::new(KgqaConfig::new().with_max_hops(2).with_top_k(10));
    let entities = vec![
        make_entity("Rust", "e1"),
        make_entity("LLVM", "e2"),
        make_entity("Clang", "e3"),
    ];
    let rels = vec![
        make_rel("e1", "e2", RelationshipType::Uses),
        make_rel("e2", "e3", RelationshipType::RelatedTo),
    ];
    let result = engine.answer("Tell me about Rust", &entities, &rels);
    assert!(result.is_ok());
    let ans = result.expect("should succeed");
    // BFS from Rust should reach LLVM (hop 1) and Clang (hop 2).
    assert!(ans.evidence_entities.len() >= 2);
}

#[test]
fn test_engine_answer_not_empty() {
    let engine = KgqaEngine::default();
    let entities = vec![make_entity("Rust", "e1")];
    let ans = engine
        .answer("What is Rust?", &entities, &[])
        .expect("should succeed");
    assert!(!ans.answer.is_empty());
}

#[test]
fn test_engine_evidence_entities_populated() {
    let engine = KgqaEngine::default();
    let entities = vec![make_entity("Rust", "e1"), make_entity("LLVM", "e2")];
    let rels = vec![make_rel("e1", "e2", RelationshipType::Uses)];
    let ans = engine
        .answer("Rust uses LLVM", &entities, &rels)
        .expect("should succeed");
    assert!(!ans.evidence_entities.is_empty());
}

#[test]
fn test_engine_low_min_confidence_more_triples() {
    let engine_low = KgqaEngine::new(KgqaConfig::new().with_min_confidence(0.0));
    let engine_high = KgqaEngine::new(KgqaConfig::new().with_min_confidence(0.99));
    let entities = vec![make_entity("Rust", "e1"), make_entity("LLVM", "e2")];
    let rels = vec![make_rel("e1", "e2", RelationshipType::Uses)];

    let ans_low = engine_low
        .answer("Rust uses LLVM", &entities, &rels)
        .expect("should succeed");
    let ans_high = engine_high
        .answer("Rust uses LLVM", &entities, &rels)
        .expect("should succeed");

    // Low threshold should allow more (or equal) triples through.
    assert!(ans_low.evidence_triples.len() >= ans_high.evidence_triples.len());
}

#[test]
fn test_engine_unknown_query_still_answers() {
    // Query that matches no entity → best-guess fallback path.
    let engine = KgqaEngine::default();
    let entities = vec![
        make_entity_typed("AlphaGo", "e1", EntityType::Technology),
        make_entity_typed("DeepMind", "e2", EntityType::Organization),
    ];
    let result = engine.answer("What is quantum computing?", &entities, &[]);
    // Best-guess fallback must not error.
    assert!(result.is_ok());
    let ans = result.expect("should succeed");
    assert!(!ans.answer.is_empty());
}

#[test]
fn test_kgqa_answer_confidence_range() {
    let engine = KgqaEngine::default();
    let entities = vec![make_entity("Rust", "e1")];
    let ans = engine
        .answer("What is Rust?", &entities, &[])
        .expect("should succeed");
    assert!((0.0..=1.0).contains(&ans.confidence));
}

// ── error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    assert_eq!(KgqaError::EmptyQuery.to_string(), "Query must not be empty");
    assert_eq!(KgqaError::NoEntities.to_string(), "No entities in graph");
    assert_eq!(KgqaError::NoAnswer.to_string(), "No answer found");
}
