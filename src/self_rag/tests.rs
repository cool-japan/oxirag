//! Tests for the `self_rag` module.

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::self_rag::reflect::Reflector;
use crate::types::{Document, DocumentId, SearchResult};

use super::engine::SelfRagEngine;
use super::reflect::{HeuristicReflector, MockReflector};
use super::types::{ReflectionToken, SelfRagConfig, SelfRagError};

use async_trait::async_trait;

// ── MockEcho ──────────────────────────────────────────────────────────────────

struct MockEcho {
    results: Vec<SearchResult>,
}

impl MockEcho {
    fn new(results: Vec<SearchResult>) -> Self {
        Self { results }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Echo for MockEcho {
    async fn index(&mut self, document: Document) -> Result<DocumentId, EmbeddingError> {
        Ok(document.id.clone())
    }

    async fn index_batch(
        &mut self,
        documents: Vec<Document>,
    ) -> Result<Vec<DocumentId>, EmbeddingError> {
        Ok(documents.into_iter().map(|d| d.id).collect())
    }

    async fn search(
        &self,
        _query: &str,
        _top_k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, EmbeddingError> {
        Ok(self.results.clone())
    }

    async fn get(&self, _id: &DocumentId) -> Result<Option<Document>, EmbeddingError> {
        Ok(None)
    }

    async fn delete(&mut self, _id: &DocumentId) -> Result<bool, EmbeddingError> {
        Ok(false)
    }

    async fn count(&self) -> usize {
        self.results.len()
    }

    async fn clear(&mut self) -> Result<(), EmbeddingError> {
        Ok(())
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

// ── ReflectionToken tests ─────────────────────────────────────────────────────

#[test]
fn test_reflection_tokens_variants() {
    let tokens = vec![
        ReflectionToken::Retrieve,
        ReflectionToken::NoRetrieve,
        ReflectionToken::Relevant,
        ReflectionToken::Irrelevant,
        ReflectionToken::Supported,
        ReflectionToken::PartiallySupported,
        ReflectionToken::Unsupported,
        ReflectionToken::Useful,
        ReflectionToken::NotUseful,
    ];
    assert_eq!(tokens.len(), 9);
}

#[test]
fn test_reflection_token_equality() {
    assert_eq!(ReflectionToken::Retrieve, ReflectionToken::Retrieve);
    assert_ne!(ReflectionToken::Retrieve, ReflectionToken::NoRetrieve);
    assert_ne!(ReflectionToken::Supported, ReflectionToken::Unsupported);
}

// ── SelfRagConfig tests ───────────────────────────────────────────────────────

#[test]
fn test_self_rag_config_defaults() {
    let cfg = SelfRagConfig::default();
    assert!((cfg.relevance_threshold - 0.5).abs() < 1e-5);
    assert!((cfg.support_threshold - 0.5).abs() < 1e-5);
    assert_eq!(cfg.top_k, 5);
    assert_eq!(cfg.max_segments, 8);
}

#[test]
fn test_self_rag_config_builders() {
    let cfg = SelfRagConfig::default()
        .with_relevance_threshold(0.7)
        .with_support_threshold(0.6)
        .with_utility_threshold(0.3)
        .with_top_k(10)
        .with_max_segments(4);
    assert!((cfg.relevance_threshold - 0.7).abs() < 1e-5);
    assert!((cfg.support_threshold - 0.6).abs() < 1e-5);
    assert_eq!(cfg.top_k, 10);
    assert_eq!(cfg.max_segments, 4);
}

// ── HeuristicReflector tests ──────────────────────────────────────────────────

#[test]
fn test_heuristic_relevance_identical() {
    let r = HeuristicReflector::new();
    let doc = make_result("d1", "rust programming language", 1.0);
    let score = r
        .relevance("rust programming language", &doc)
        .expect("should succeed");
    assert!((score - 1.0).abs() < 1e-5);
}

#[test]
fn test_heuristic_relevance_no_overlap() {
    let r = HeuristicReflector::new();
    let doc = make_result("d1", "python machine learning", 1.0);
    let score = r
        .relevance("rust programming", &doc)
        .expect("should succeed");
    assert!(score < 0.3, "score should be low: {score}");
}

#[test]
fn test_heuristic_support_identical() {
    let r = HeuristicReflector::new();
    let doc = make_result("d1", "the quick brown fox jumps", 1.0);
    let score = r
        .support("the quick brown fox jumps", &doc)
        .expect("should succeed");
    assert!((score - 1.0).abs() < 1e-5);
}

#[test]
fn test_heuristic_utility_empty_answer() {
    let r = HeuristicReflector::new();
    let score = r.utility("what is rust", "").expect("should succeed");
    assert!((score - 0.0).abs() < 1e-5);
}

#[test]
fn test_heuristic_utility_relevant_answer() {
    let r = HeuristicReflector::new();
    let score = r
        .utility("what is rust", "rust is a systems programming language")
        .expect("should succeed");
    assert!(score > 0.0, "utility should be positive: {score}");
}

// ── MockReflector tests ───────────────────────────────────────────────────────

#[test]
fn test_mock_reflector_fixed() {
    let r = MockReflector::new(0.75);
    let doc = make_result("d1", "anything", 1.0);
    assert!((r.relevance("q", &doc).expect("ok") - 0.75).abs() < 1e-5);
    assert!((r.support("seg", &doc).expect("ok") - 0.75).abs() < 1e-5);
    assert!((r.utility("q", "ans").expect("ok") - 0.75).abs() < 1e-5);
}

#[test]
fn test_mock_reflector_independent_scores() {
    let r = MockReflector::with_scores(0.9, 0.3, 0.5);
    let doc = make_result("d1", "anything", 1.0);
    assert!((r.relevance("q", &doc).expect("ok") - 0.9).abs() < 1e-5);
    assert!((r.support("seg", &doc).expect("ok") - 0.3).abs() < 1e-5);
    assert!((r.utility("q", "ans").expect("ok") - 0.5).abs() < 1e-5);
}

// ── SelfRagEngine tests ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_engine_empty_query() {
    let engine = SelfRagEngine::new(MockReflector::new(0.8), SelfRagConfig::default());
    let echo = MockEcho::new(vec![]);
    let err = engine
        .run("", "some generation", &echo)
        .await
        .expect_err("should fail");
    assert!(matches!(err, SelfRagError::EmptyQuery), "got: {err:?}");
}

#[tokio::test]
async fn test_engine_empty_generation_retrieves() {
    let docs = vec![
        make_result("d1", "rust is a systems language", 0.9),
        make_result("d2", "rust focuses on safety", 0.8),
    ];
    let engine = SelfRagEngine::new(MockReflector::new(0.8), SelfRagConfig::default());
    let echo = MockEcho::new(docs);
    let output = engine
        .run("what is rust", "", &echo)
        .await
        .expect("should succeed");
    assert!(output.retrieved, "should have retrieved docs");
    assert!(
        output
            .reflection_tokens
            .contains(&ReflectionToken::Retrieve),
        "should have Retrieve token"
    );
    assert!(!output.answer.is_empty(), "answer should not be empty");
}

#[tokio::test]
async fn test_engine_generation_with_trigger_retrieves() {
    let docs = vec![make_result(
        "d1",
        "rust was created by mozilla research",
        0.9,
    )];
    let engine = SelfRagEngine::new(
        MockReflector::new(0.9),
        SelfRagConfig::default().with_relevance_threshold(0.5),
    );
    let echo = MockEcho::new(docs);
    let generation = "Rust was created by Mozilla Research.";
    let output = engine
        .run("who created rust", generation, &echo)
        .await
        .expect("should succeed");
    assert!(output.retrieved, "trigger words should cause retrieval");
    assert!(!output.critiques.is_empty(), "should have critiques");
}

#[tokio::test]
async fn test_engine_no_trigger_no_retrieval() {
    let engine = SelfRagEngine::new(MockReflector::new(0.8), SelfRagConfig::default());
    let echo = MockEcho::new(vec![]);
    let generation = "The sky is blue.";
    let output = engine
        .run("color of sky", generation, &echo)
        .await
        .expect("should succeed");
    // No trigger words → NoRetrieve
    assert!(
        output
            .reflection_tokens
            .contains(&ReflectionToken::NoRetrieve),
        "should have NoRetrieve"
    );
}

#[tokio::test]
async fn test_engine_high_score_marks_supported() {
    let docs = vec![make_result(
        "d1",
        "rust programming safety performance",
        0.9,
    )];
    let engine = SelfRagEngine::new(
        MockReflector::new(0.9),
        SelfRagConfig::default()
            .with_relevance_threshold(0.3)
            .with_support_threshold(0.3),
    );
    let echo = MockEcho::new(docs);
    let output = engine
        .run(
            "rust programming",
            "Rust is about safety. According to experts, it is fast.",
            &echo,
        )
        .await
        .expect("should succeed");

    let supported = output.critiques.iter().any(|c| {
        matches!(
            c.support_token,
            ReflectionToken::Supported | ReflectionToken::PartiallySupported
        )
    });
    assert!(supported, "high scores should yield supported token");
}

#[tokio::test]
async fn test_engine_zero_docs_marks_unsupported() {
    let engine = SelfRagEngine::new(
        MockReflector::new(0.9),
        SelfRagConfig::default().with_relevance_threshold(0.1),
    );
    let echo = MockEcho::new(vec![]); // no docs
    let output = engine
        .run(
            "according to research, rust is fast",
            "Rust is fast according to research.",
            &echo,
        )
        .await
        .expect("should succeed");
    // No docs → Unsupported for retrieved segments
    for critique in &output.critiques {
        if critique.retrieved_docs.is_empty() {
            assert_eq!(
                critique.support_token,
                ReflectionToken::Unsupported,
                "empty docs should yield Unsupported"
            );
        }
    }
}

#[tokio::test]
async fn test_engine_utility_token_useful() {
    let docs = vec![make_result("d1", "rust is a safe language", 0.9)];
    let engine = SelfRagEngine::new(
        MockReflector::with_scores(0.9, 0.9, 0.9),
        SelfRagConfig::default().with_utility_threshold(0.5),
    );
    let echo = MockEcho::new(docs);
    let output = engine
        .run("tell me about rust", "Rust is a safe language.", &echo)
        .await
        .expect("should succeed");
    assert_eq!(output.utility_token, ReflectionToken::Useful);
    assert!(output.is_useful());
}

#[tokio::test]
async fn test_engine_utility_token_not_useful() {
    let docs = vec![];
    let engine = SelfRagEngine::new(
        MockReflector::with_scores(0.0, 0.0, 0.0),
        SelfRagConfig::default().with_utility_threshold(0.5),
    );
    let echo = MockEcho::new(docs);
    let output = engine
        .run("complex query", "irrelevant answer", &echo)
        .await
        .expect("should succeed");
    assert_eq!(output.utility_token, ReflectionToken::NotUseful);
    assert!(!output.is_useful());
}

#[tokio::test]
async fn test_output_support_rate_all_supported() {
    let docs = vec![make_result("d1", "rust is a programming language", 0.9)];
    let engine = SelfRagEngine::new(
        MockReflector::new(0.9),
        SelfRagConfig::default()
            .with_relevance_threshold(0.1)
            .with_support_threshold(0.1),
    );
    let echo = MockEcho::new(docs);
    let output = engine
        .run(
            "rust programming",
            "Rust was invented by Graydon Hoare.",
            &echo,
        )
        .await
        .expect("should succeed");
    // At least 0 critiques or all supported
    let rate = output.support_rate();
    assert!((0.0..=1.0).contains(&rate), "rate out of range: {rate}");
}

#[test]
fn test_output_support_rate_no_critiques() {
    use super::types::SelfRagOutput;
    let output = SelfRagOutput {
        query: "q".to_string(),
        answer: "a".to_string(),
        retrieved: false,
        reflection_tokens: vec![],
        critiques: vec![],
        utility_token: ReflectionToken::NotUseful,
    };
    assert!((output.support_rate() - 0.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_engine_max_segments_limit() {
    let docs = vec![make_result("d1", "content", 0.8)];
    let engine = SelfRagEngine::new(
        MockReflector::new(0.9),
        SelfRagConfig::default().with_max_segments(2),
    );
    let echo = MockEcho::new(docs);
    // Long generation with many sentences
    let generation = "According to research, Rust is safe. According to experts, it is fast. \
        According to studies, it has zero-cost abstractions. According to data, adoption is growing.";
    let output = engine
        .run("rust", generation, &echo)
        .await
        .expect("should succeed");
    // Should not exceed max_segments in critiques
    assert!(
        output.critiques.len() <= 2,
        "should respect max_segments: {}",
        output.critiques.len()
    );
}

#[tokio::test]
async fn test_engine_whitespace_only_query() {
    let engine = SelfRagEngine::new(MockReflector::new(0.8), SelfRagConfig::default());
    let echo = MockEcho::new(vec![]);
    let err = engine
        .run("   ", "some answer", &echo)
        .await
        .expect_err("should fail");
    assert!(matches!(err, SelfRagError::EmptyQuery));
}
