//! Tests for the `context_compression` module.

use crate::types::{Document, DocumentId, SearchResult};

use super::compressor::{ExtractiveCompressor, MockCompressor, RedundancyFilter};
use super::types::{CompressionConfig, CompressionError, ContextCompressor};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

// ── CompressionConfig tests ───────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = CompressionConfig::default();
    assert_eq!(cfg.token_budget, 512);
    assert!((cfg.relevance_threshold - 0.1).abs() < 1e-5);
    assert!((cfg.redundancy_threshold - 0.8).abs() < 1e-5);
}

#[test]
fn test_config_builders() {
    let cfg = CompressionConfig::default()
        .with_token_budget(256)
        .with_relevance_threshold(0.3)
        .with_redundancy_threshold(0.7);
    assert_eq!(cfg.token_budget, 256);
    assert!((cfg.relevance_threshold - 0.3).abs() < 1e-5);
    assert!((cfg.redundancy_threshold - 0.7).abs() < 1e-5);
}

// ── RedundancyFilter tests ────────────────────────────────────────────────────

#[test]
fn test_redundancy_filter_no_duplicates() {
    let f = RedundancyFilter::new();
    let sentences = vec![
        "Rust is fast".to_string(),
        "Python is flexible".to_string(),
        "Go is concurrent".to_string(),
    ];
    let kept = f.filter(&sentences, 0.8);
    assert_eq!(kept.len(), 3);
}

#[test]
fn test_redundancy_filter_exact_duplicates() {
    let f = RedundancyFilter::new();
    let sentences = vec![
        "Rust is fast and safe".to_string(),
        "Rust is fast and safe".to_string(),
    ];
    let kept = f.filter(&sentences, 0.8);
    assert_eq!(kept.len(), 1);
}

#[test]
fn test_redundancy_filter_near_duplicates() {
    let f = RedundancyFilter::new();
    let sentences = vec![
        "Rust programming language is fast".to_string(),
        "Rust programming language is fast and efficient".to_string(),
    ];
    // High threshold → both kept
    let kept_high = f.filter(&sentences, 0.95);
    assert_eq!(kept_high.len(), 2);
    // Lower threshold → one removed
    let kept_low = f.filter(&sentences, 0.5);
    assert_eq!(kept_low.len(), 1);
}

#[test]
fn test_redundancy_filter_empty() {
    let f = RedundancyFilter::new();
    let kept = f.filter(&[], 0.8);
    assert!(kept.is_empty());
}

// ── ExtractiveCompressor tests ────────────────────────────────────────────────

#[test]
fn test_extractor_empty_sources() {
    let comp = ExtractiveCompressor::new();
    let cfg = CompressionConfig::default();
    let err = comp.compress("rust", &[], &cfg).expect_err("should fail");
    assert!(matches!(err, CompressionError::EmptyInput));
}

#[test]
fn test_extractor_basic() {
    let comp = ExtractiveCompressor::new();
    let cfg = CompressionConfig::default().with_relevance_threshold(0.0);
    let sources = vec![make_result(
        "d1",
        "Rust is fast. Rust is safe. Rust is reliable.",
        0.9,
    )];
    let ctx = comp.compress("rust", &sources, &cfg).expect("ok");
    assert!(!ctx.text.is_empty(), "compressed text should not be empty");
    assert!(ctx.original_tokens > 0);
}

#[test]
fn test_extractor_budget_zero_gives_empty() {
    let comp = ExtractiveCompressor::new();
    let cfg = CompressionConfig::default()
        .with_token_budget(0)
        .with_relevance_threshold(0.0);
    let sources = vec![make_result("d1", "Rust is fast. Rust is safe.", 0.9)];
    let ctx = comp.compress("rust", &sources, &cfg).expect("ok");
    assert!(ctx.text.is_empty(), "zero budget → empty compression");
    assert_eq!(ctx.compressed_tokens, 0);
}

#[test]
fn test_extractor_token_budget_respected() {
    let comp = ExtractiveCompressor::new();
    let cfg = CompressionConfig::default()
        .with_token_budget(5)
        .with_relevance_threshold(0.0);
    let sources = vec![make_result(
        "d1",
        "Rust is fast and memory safe. Python is a dynamic language. Go is concurrent.",
        0.9,
    )];
    let ctx = comp.compress("language", &sources, &cfg).expect("ok");
    assert!(
        ctx.compressed_tokens <= 5,
        "should respect budget: {} tokens",
        ctx.compressed_tokens
    );
}

#[test]
fn test_extractor_ratio_range() {
    let comp = ExtractiveCompressor::new();
    let cfg = CompressionConfig::default().with_relevance_threshold(0.0);
    let sources = vec![make_result(
        "d1",
        "Rust is fast. Python is slow. Go is okay.",
        0.9,
    )];
    let ctx = comp.compress("rust", &sources, &cfg).expect("ok");
    assert!(
        (0.0..=1.0).contains(&ctx.ratio) || ctx.ratio <= 1.0,
        "ratio should be ≤ 1.0"
    );
}

#[test]
fn test_extractor_is_empty_check() {
    let comp = ExtractiveCompressor::new();
    let cfg = CompressionConfig::default()
        .with_token_budget(0)
        .with_relevance_threshold(0.0);
    let sources = vec![make_result("d1", "content here", 0.9)];
    let ctx = comp.compress("query", &sources, &cfg).expect("ok");
    assert!(ctx.is_empty());
}

#[test]
fn test_extractor_multiple_sources() {
    let comp = ExtractiveCompressor::new();
    let cfg = CompressionConfig::default().with_relevance_threshold(0.0);
    let sources = vec![
        make_result("d1", "Rust is a systems language.", 0.9),
        make_result("d2", "Rust has zero-cost abstractions.", 0.8),
        make_result("d3", "Rust prevents data races.", 0.7),
    ];
    let ctx = comp.compress("rust safety", &sources, &cfg).expect("ok");
    assert!(!ctx.text.is_empty());
}

#[test]
fn test_extractor_high_relevance_threshold_may_empty() {
    let comp = ExtractiveCompressor::new();
    let cfg = CompressionConfig::default().with_relevance_threshold(0.99);
    let sources = vec![make_result(
        "d1",
        "completely unrelated random words here",
        0.9,
    )];
    let ctx = comp
        .compress("rust programming", &sources, &cfg)
        .expect("ok");
    // May be empty or not — no panic expected
    assert!((0.0..=1.0).contains(&ctx.ratio) || ctx.ratio == 0.0);
}

// ── MockCompressor tests ──────────────────────────────────────────────────────

#[test]
fn test_mock_compressor_returns_fixed_output() {
    let comp = MockCompressor::new("compressed output text");
    let cfg = CompressionConfig::default();
    let sources = vec![make_result("d1", "original content", 0.9)];
    let ctx = comp.compress("query", &sources, &cfg).expect("ok");
    assert_eq!(ctx.text, "compressed output text");
    assert!((ctx.ratio - 1.0).abs() < 1e-5);
}

#[test]
fn test_mock_compressor_empty_sources() {
    let comp = MockCompressor::new("output");
    let cfg = CompressionConfig::default();
    let err = comp.compress("query", &[], &cfg).expect_err("should fail");
    assert!(matches!(err, CompressionError::EmptyInput));
}
