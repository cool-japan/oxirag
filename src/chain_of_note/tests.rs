//! Tests for the `chain_of_note` module.

use crate::types::{Document, DocumentId, SearchResult};

use super::engine::{ChainOfNoteEngine, extract_note, jaccard, sentence_split};
use super::types::{ChainOfNoteError, DocumentNote, NoteChain, NoteConfig};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

fn make_doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

// ── Config tests ──────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = NoteConfig::default();
    assert_eq!(cfg.max_notes_per_doc, 3);
    assert_eq!(cfg.max_note_length, 150);
    assert!((cfg.relevance_threshold - 0.1).abs() < 1e-5);
    assert_eq!(cfg.top_k, 5);
}

#[test]
fn test_config_builders() {
    let cfg = NoteConfig::default()
        .with_max_notes_per_doc(5)
        .with_max_note_length(200)
        .with_relevance_threshold(0.2)
        .with_top_k(10);
    assert_eq!(cfg.max_notes_per_doc, 5);
    assert_eq!(cfg.max_note_length, 200);
    assert!((cfg.relevance_threshold - 0.2).abs() < 1e-5);
    assert_eq!(cfg.top_k, 10);
}

// ── DocumentNote tests ────────────────────────────────────────────────────────

#[test]
fn test_document_note_is_relevant() {
    let note = DocumentNote {
        doc_id: "d1".to_string(),
        note: "Some note text".to_string(),
        relevance_score: 0.5,
    };
    assert!(note.is_relevant(0.1));
    assert!(note.is_relevant(0.5));
    assert!(!note.is_relevant(0.6));
}

// ── NoteChain tests ───────────────────────────────────────────────────────────

#[test]
fn test_note_chain_is_empty() {
    let chain = NoteChain {
        notes: vec![],
        synthesized: "answer".to_string(),
        query: "query".to_string(),
    };
    assert!(chain.is_empty());

    let chain2 = NoteChain {
        notes: vec![DocumentNote {
            doc_id: "d1".to_string(),
            note: "note".to_string(),
            relevance_score: 0.5,
        }],
        synthesized: "answer".to_string(),
        query: "query".to_string(),
    };
    assert!(!chain2.is_empty());
}

#[test]
fn test_note_chain_relevant_count() {
    let chain = NoteChain {
        notes: vec![
            DocumentNote {
                doc_id: "d1".to_string(),
                note: "a".to_string(),
                relevance_score: 0.8,
            },
            DocumentNote {
                doc_id: "d2".to_string(),
                note: "b".to_string(),
                relevance_score: 0.05,
            },
            DocumentNote {
                doc_id: "d3".to_string(),
                note: "c".to_string(),
                relevance_score: 0.3,
            },
        ],
        synthesized: String::new(),
        query: "q".to_string(),
    };
    assert_eq!(chain.relevant_count(0.1), 2);
    assert_eq!(chain.relevant_count(0.5), 1);
    assert_eq!(chain.relevant_count(0.9), 0);
}

// ── Engine error tests ────────────────────────────────────────────────────────

#[test]
fn test_engine_empty_query_error() {
    let engine = ChainOfNoteEngine::default();
    let results = vec![make_result("d1", "content", 0.9)];
    let err = engine.process("", &results).unwrap_err();
    assert!(matches!(err, ChainOfNoteError::EmptyQuery));
}

#[test]
fn test_engine_empty_documents_error() {
    let engine = ChainOfNoteEngine::default();
    let err = engine.process("rust query", &[]).unwrap_err();
    assert!(matches!(err, ChainOfNoteError::EmptyDocuments));
}

// ── Engine functional tests ───────────────────────────────────────────────────

#[test]
fn test_engine_single_doc() {
    let engine = ChainOfNoteEngine::default();
    let results = vec![make_result(
        "d1",
        "Rust is a systems programming language. It ensures memory safety.",
        0.9,
    )];
    let chain = engine.process("rust programming", &results).unwrap();
    assert!(!chain.synthesized.is_empty());
    assert_eq!(chain.query, "rust programming");
}

#[test]
fn test_engine_multiple_docs() {
    let engine = ChainOfNoteEngine::default();
    let results = vec![
        make_result(
            "d1",
            "Rust programming language ensures memory safety through ownership.",
            0.9,
        ),
        make_result(
            "d2",
            "Rust programming uses borrowing and lifetimes to prevent data races.",
            0.8,
        ),
        make_result(
            "d3",
            "Rust programming has a package manager called Cargo.",
            0.7,
        ),
    ];
    let chain = engine.process("rust programming", &results).unwrap();
    assert!(!chain.synthesized.is_empty());
    assert!(chain.synthesized.contains("Based on retrieved documents"));
}

#[test]
fn test_engine_relevance_threshold_filters() {
    // Use very high threshold so notes get filtered out
    let cfg = NoteConfig::default().with_relevance_threshold(0.99);
    let engine = ChainOfNoteEngine::new(cfg);

    let results = vec![make_result(
        "d1",
        "Completely unrelated topic xyz abc.",
        0.9,
    )];
    let chain = engine.process("rust", &results).unwrap();
    // All notes should be filtered out with high threshold
    assert_eq!(chain.relevant_count(0.99), 0);
}

#[test]
fn test_engine_note_length_capped() {
    let cfg = NoteConfig::default()
        .with_max_note_length(20)
        .with_relevance_threshold(0.0);
    let engine = ChainOfNoteEngine::new(cfg);

    let long_content = "Rust is an amazing programming language with many features for safety.";
    let results = vec![make_result("d1", long_content, 0.9)];
    let chain = engine.process("rust", &results).unwrap();
    for note in &chain.notes {
        assert!(note.note.len() <= 20, "Note too long: {}", note.note.len());
    }
}

#[test]
fn test_engine_synthesized_not_empty() {
    let engine = ChainOfNoteEngine::default();
    let results = vec![make_result(
        "d1",
        "Rust language programming systems memory safety.",
        0.9,
    )];
    let chain = engine.process("rust language", &results).unwrap();
    assert!(!chain.synthesized.is_empty());
}

#[test]
fn test_engine_top_k_limit() {
    let cfg = NoteConfig::default()
        .with_top_k(2)
        .with_relevance_threshold(0.0);
    let engine = ChainOfNoteEngine::new(cfg);

    let results = vec![
        make_result("d1", "First document about rust language.", 0.9),
        make_result("d2", "Second document about rust language.", 0.8),
        make_result("d3", "Third document about rust language.", 0.7),
        make_result("d4", "Fourth document about rust language.", 0.6),
    ];
    let chain = engine.process("rust language", &results).unwrap();
    // Should only process top_k=2 documents
    assert!(chain.notes.len() <= 2);
}

#[test]
fn test_engine_process_documents() {
    let engine = ChainOfNoteEngine::default();
    let docs = vec![
        make_doc("d1", "Rust programming language memory safety ownership."),
        make_doc("d2", "Rust programming uses lifetimes for safety."),
    ];
    let chain = engine.process_documents("rust programming", &docs).unwrap();
    assert!(!chain.synthesized.is_empty());
}

#[test]
fn test_engine_all_low_relevance_still_returns_chain() {
    // Even with all-low relevance notes, a chain should still be returned
    let cfg = NoteConfig::default().with_relevance_threshold(0.0);
    let engine = ChainOfNoteEngine::new(cfg);

    let results = vec![
        make_result("d1", "completely unrelated xyz abc.", 0.1),
        make_result("d2", "another unrelated topic xyz.", 0.1),
    ];
    let chain = engine.process("rust programming", &results).unwrap();
    assert!(!chain.synthesized.is_empty());
}

// ── Helper unit tests ─────────────────────────────────────────────────────────

#[test]
fn test_jaccard_helper() {
    // Identical strings → jaccard = 1.0
    let score = jaccard("rust programming", "rust programming");
    assert!((score - 1.0).abs() < 1e-5, "Expected 1.0, got {score}");

    // No overlap → jaccard = 0.0
    let score2 = jaccard("apple banana", "cat dog");
    assert!(score2 < 1e-5, "Expected 0.0, got {score2}");

    // Partial overlap
    let score3 = jaccard("rust programming language", "rust language");
    assert!(
        score3 > 0.0 && score3 < 1.0,
        "Expected partial overlap, got {score3}"
    );

    // Both empty
    let score4 = jaccard("", "");
    assert!(score4 < 1e-5, "Expected 0.0 for empty inputs, got {score4}");
}

#[test]
fn test_sentence_split() {
    let text = "Rust is safe. It prevents bugs. Memory management is explicit.";
    let sentences = sentence_split(text);
    assert!(
        sentences.len() >= 2,
        "Expected multiple sentences, got {}",
        sentences.len()
    );
    // All sentences should be non-empty
    for s in &sentences {
        assert!(!s.is_empty());
    }
}

#[test]
fn test_extract_note_basic() {
    let content = "Rust programming ensures memory safety. Ownership prevents leaks.";
    let (note, score) = extract_note(content, "rust programming", 3, 200);
    assert!(!note.is_empty());
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn test_error_display() {
    let e1 = ChainOfNoteError::EmptyQuery;
    assert!(e1.to_string().contains("empty"));

    let e2 = ChainOfNoteError::EmptyDocuments;
    assert!(e2.to_string().contains("documents"));
}
