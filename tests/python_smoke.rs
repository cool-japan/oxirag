//! Rust-side compile test for the PyO3 bridge.
//!
//! This file verifies that all PyO3-annotated types compile correctly with the
//! `python` feature enabled. Full runtime tests (requiring an embedded Python
//! interpreter) must be run through maturin/pytest instead:
//!
//! ```bash
//! maturin develop --features python
//! pytest python/tests/
//! ```
//!
//! The tests here exercise only pure-Rust paths (struct construction, `__repr__`,
//! and builder chaining) that do not require the Python GIL to be held.
#![cfg(not(target_arch = "wasm32"))]
#![cfg(feature = "python")]

use oxirag::python::builder::PyPipelineBuilder;
use oxirag::python::types::{PyDocument, PyQuery};

// ────────────────────────────────────────────────────────────────────────────
// Builder compile + repr tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_builder_default_repr() {
    let builder = PyPipelineBuilder::new();
    let repr = builder.__repr__();
    assert!(
        repr.contains("384"),
        "default dimension should be 384: {repr}"
    );
    assert!(
        repr.contains("10"),
        "default max_results should be 10: {repr}"
    );
}

#[test]
fn test_builder_chaining() {
    let builder = PyPipelineBuilder::new()
        .with_dimension(128)
        .with_max_results(5)
        .with_fast_path(false)
        .with_fast_path_threshold(0.8);

    let repr = builder.__repr__();
    assert!(repr.contains("128"), "dimension should be 128: {repr}");
    assert!(repr.contains('5'), "max_results should be 5: {repr}");
}

// ────────────────────────────────────────────────────────────────────────────
// PyDocument tests (pure Rust, no GIL needed)
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_document_construction() {
    let doc = PyDocument::new("test content".to_string(), Some("Test Title".to_string()));
    assert_eq!(doc.content(), "test content");
    assert_eq!(doc.title(), Some("Test Title"));
    assert!(!doc.id().is_empty(), "id should not be empty");
    assert!(doc.source().is_none(), "source should be None");
}

#[test]
fn test_document_no_title() {
    let doc = PyDocument::new("content only".to_string(), None);
    assert_eq!(doc.content(), "content only");
    assert!(doc.title().is_none());
}

#[test]
fn test_document_with_metadata_no_panic() {
    let doc = PyDocument::new("content".to_string(), None);
    let doc2 = doc.with_metadata("key".to_string(), "val".to_string());
    assert_eq!(doc2.content(), "content");
}

#[test]
fn test_document_with_source() {
    let doc =
        PyDocument::new("content".to_string(), None).with_source("https://example.com".to_string());
    assert_eq!(doc.source(), Some("https://example.com"));
}

#[test]
fn test_document_with_title() {
    let doc = PyDocument::new("content".to_string(), None).with_title("Added Title".to_string());
    assert_eq!(doc.title(), Some("Added Title"));
}

#[test]
fn test_document_repr() {
    let doc = PyDocument::new("Hello, world!".to_string(), None);
    let repr = doc.__repr__();
    assert!(repr.contains("Document"), "repr: {repr}");
    assert!(
        repr.contains("Hello"),
        "repr should contain content preview: {repr}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// PyQuery tests (pure Rust, no GIL needed)
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_query_construction() {
    let q = PyQuery::new("What is Rust?".to_string());
    assert_eq!(q.text(), "What is Rust?");
    assert_eq!(q.top_k(), 10); // default
    assert!(q.min_score().is_none());
}

#[test]
fn test_query_with_top_k() {
    let q = PyQuery::new("test".to_string()).with_top_k(5);
    assert_eq!(q.top_k(), 5);
}

#[test]
fn test_query_with_min_score() {
    let q = PyQuery::new("test".to_string()).with_min_score(0.3);
    assert!(
        (q.min_score().expect("min_score should be Some") - 0.3_f32).abs() < 1e-6,
        "min_score mismatch"
    );
}

#[test]
fn test_query_repr() {
    let q = PyQuery::new("What is Rust?".to_string()).with_top_k(3);
    let repr = q.__repr__();
    assert!(repr.contains("Query"), "repr: {repr}");
    assert!(repr.contains("What is Rust?"), "repr: {repr}");
    assert!(repr.contains('3'), "repr: {repr}");
}
