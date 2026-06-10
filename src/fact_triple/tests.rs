//! Tests for the `fact_triple` module.

use super::types::{Triple, TripleConfig, TripleError, TripleExtractor, TripleStore};

// ── config tests ──────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = TripleConfig::default();
    assert!((cfg.min_confidence - 0.5).abs() < f32::EPSILON);
    assert_eq!(cfg.max_per_sentence, 3);
}

#[test]
fn test_config_builders() {
    let cfg = TripleConfig::new()
        .with_min_confidence(0.3)
        .with_max_per_sentence(5);
    assert!((cfg.min_confidence - 0.3).abs() < f32::EPSILON);
    assert_eq!(cfg.max_per_sentence, 5);
}

// ── Triple tests ──────────────────────────────────────────────────────────────

#[test]
fn test_triple_new() {
    let t = Triple::new("Rust", "uses", "LLVM", 0.8);
    assert_eq!(t.subject, "Rust");
    assert_eq!(t.predicate, "uses");
    assert_eq!(t.object, "LLVM");
    assert!((t.confidence - 0.8).abs() < f32::EPSILON);
    assert!(t.source_id.is_none());
}

#[test]
fn test_triple_to_string() {
    let t = Triple::new("Rust", "uses", "LLVM", 0.8);
    assert_eq!(t.to_string(), "Rust USES LLVM");
}

#[test]
fn test_triple_with_source() {
    let t = Triple::new("Rust", "uses", "LLVM", 0.8).with_source("doc1");
    assert_eq!(t.source_id.as_deref(), Some("doc1"));
}

// ── TripleStore tests ─────────────────────────────────────────────────────────

#[test]
fn test_store_empty() {
    let store = TripleStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
}

#[test]
fn test_store_add_and_len() {
    let mut store = TripleStore::new();
    store.add(Triple::new("Rust", "uses", "LLVM", 0.7));
    store.add(Triple::new("Python", "is", "language", 0.9));
    assert_eq!(store.len(), 2);
    assert!(!store.is_empty());
}

#[test]
fn test_store_find_by_subject() {
    let mut store = TripleStore::new();
    store.add(Triple::new("Rust", "uses", "LLVM", 0.7));
    store.add(Triple::new("Python", "is", "language", 0.9));
    let hits = store.find_by_subject("rust");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].object, "LLVM");
}

#[test]
fn test_store_find_by_predicate() {
    let mut store = TripleStore::new();
    store.add(Triple::new("Rust", "uses", "LLVM", 0.7));
    store.add(Triple::new("Python", "uses", "CPython", 0.8));
    store.add(Triple::new("Go", "is", "compiled", 0.9));
    let hits = store.find_by_predicate("uses");
    assert_eq!(hits.len(), 2);
}

#[test]
fn test_store_match_query() {
    let mut store = TripleStore::new();
    store.add(Triple::new("Rust", "uses", "LLVM", 0.7));
    store.add(Triple::new("Python", "is", "dynamic", 0.9));
    let hits = store.match_query("Rust LLVM");
    // Should match the Rust triple on both tokens
    assert!(!hits.is_empty());
    assert!(hits.iter().any(|t| t.subject == "Rust"));
}

// ── extractor tests ───────────────────────────────────────────────────────────

#[test]
fn test_extractor_empty_input() {
    let extractor = TripleExtractor::default();
    let result = extractor.extract("", None);
    assert!(result.is_empty());
}

#[test]
fn test_extractor_simple_sentence() {
    let extractor = TripleExtractor::default();
    let result = extractor.extract("Rust uses LLVM as its backend.", None);
    // Should extract at least one triple with "uses" as predicate.
    assert!(!result.is_empty());
    let triple = &result[0];
    assert!(triple.predicate.eq_ignore_ascii_case("uses"));
}

#[test]
fn test_extractor_multiple_sentences() {
    let extractor = TripleExtractor::default();
    let text = "Rust uses LLVM. Python is dynamic. Go was created by Google.";
    let result = extractor.extract(text, None);
    // Should produce at least 2 triples from the 3 sentences.
    assert!(result.len() >= 2);
}

#[test]
fn test_extractor_known_verb_higher_confidence() {
    let extractor = TripleExtractor::new(TripleConfig::new().with_min_confidence(0.0));
    let result = extractor.extract("Rust uses LLVM.", None);
    assert!(!result.is_empty());
    // "uses" is in COMMON_VERBS → confidence 0.7
    assert!((result[0].confidence - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_extractor_min_confidence_filter() {
    // Set threshold above known-verb confidence; no triples should survive.
    let extractor = TripleExtractor::new(TripleConfig::new().with_min_confidence(0.9));
    let result = extractor.extract("Rust uses LLVM.", None);
    assert!(result.is_empty());
}

#[test]
fn test_extractor_max_per_sentence() {
    // Even with a very verbose sentence, at most 1 triple is emitted per sentence
    // because the heuristic finds one verb pivot.
    let extractor = TripleExtractor::new(TripleConfig::new().with_max_per_sentence(1));
    let result = extractor.extract("Rust uses LLVM.", None);
    // Should not exceed max_per_sentence.
    assert!(result.len() <= 1);
}

#[test]
fn test_extract_store() {
    let extractor = TripleExtractor::default();
    let store = extractor.extract_store("Rust uses LLVM.", Some("doc1"));
    assert!(!store.is_empty());
    // All triples should have source_id set.
    for t in &store.triples {
        assert_eq!(t.source_id.as_deref(), Some("doc1"));
    }
}

// ── error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    assert_eq!(
        TripleError::EmptyInput.to_string(),
        "Input must not be empty"
    );
    assert_eq!(TripleError::NoParseable.to_string(), "No parseable triples");
}
