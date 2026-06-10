//! Tests for the `query_decomposition` module.

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::types::{Document, DocumentId, SearchResult};

use super::decomposer::QueryDecomposer;
use super::engine::QueryDecompositionEngine;
use super::types::{DecompositionConfig, DecompositionStrategy, QueryDecompositionError};

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

// ── DecompositionStrategy tests ───────────────────────────────────────────────

#[test]
fn test_strategy_as_str() {
    assert_eq!(DecompositionStrategy::Parallel.as_str(), "parallel");
    assert_eq!(DecompositionStrategy::LeastToMost.as_str(), "least-to-most");
    assert_eq!(DecompositionStrategy::StepBack.as_str(), "step-back");
}

#[test]
fn test_strategy_default() {
    assert_eq!(
        DecompositionStrategy::default(),
        DecompositionStrategy::Parallel
    );
}

// ── DecompositionConfig tests ─────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = DecompositionConfig::default();
    assert_eq!(cfg.max_sub_questions, 4);
    assert_eq!(cfg.top_k, 5);
    assert!((cfg.rrf_k - 60.0).abs() < 1e-5);
    assert_eq!(cfg.strategy, DecompositionStrategy::Parallel);
}

#[test]
fn test_config_builders() {
    let cfg = DecompositionConfig::default()
        .with_strategy(DecompositionStrategy::LeastToMost)
        .with_max_sub_questions(2)
        .with_top_k(10)
        .with_rrf_k(30.0);
    assert_eq!(cfg.max_sub_questions, 2);
    assert_eq!(cfg.top_k, 10);
    assert!((cfg.rrf_k - 30.0).abs() < 1e-5);
    assert_eq!(cfg.strategy, DecompositionStrategy::LeastToMost);
}

// ── QueryDecomposer tests ─────────────────────────────────────────────────────

#[test]
fn test_decomposer_parallel_conjunction() {
    let d = QueryDecomposer::new(4);
    let parts = d.decompose(
        "What is Rust and what is Go",
        DecompositionStrategy::Parallel,
    );
    assert!(
        !parts.is_empty(),
        "should produce at least one sub-question"
    );
    assert!(parts.iter().all(|p| !p.text.is_empty()));
}

#[test]
fn test_decomposer_parallel_no_split() {
    let d = QueryDecomposer::new(4);
    let parts = d.decompose("what is rust", DecompositionStrategy::Parallel);
    assert_eq!(parts.len(), 1, "no split points → whole query");
    assert_eq!(parts[0].text, "what is rust");
    assert_eq!(parts[0].index, 0);
}

#[test]
fn test_decomposer_parallel_vs_split() {
    let d = QueryDecomposer::new(4);
    let parts = d.decompose("Rust vs Go performance", DecompositionStrategy::Parallel);
    assert!(!parts.is_empty());
}

#[test]
fn test_decomposer_max_respected() {
    let d = QueryDecomposer::new(1);
    let parts = d.decompose("compare A and B and C", DecompositionStrategy::Parallel);
    assert!(parts.len() <= 1);
}

#[test]
fn test_decomposer_least_to_most_how() {
    let d = QueryDecomposer::new(4);
    let parts = d.decompose(
        "how does Rust manage memory",
        DecompositionStrategy::LeastToMost,
    );
    assert!(!parts.is_empty());
    // Last element should be the original query
    assert!(parts.last().expect("non-empty").text.contains("how") || parts.len() == 1);
}

#[test]
fn test_decomposer_step_back_generates_abstract() {
    let d = QueryDecomposer::new(4);
    let parts = d.decompose(
        "how does async-await work in Rust",
        DecompositionStrategy::StepBack,
    );
    assert!(
        parts.len() >= 2,
        "step-back should prepend an abstract question"
    );
    assert!(
        parts[0].text.contains("general"),
        "first question should be abstract: {}",
        parts[0].text
    );
}

#[test]
fn test_decomposer_indices_sequential() {
    let d = QueryDecomposer::new(4);
    let parts = d.decompose("A and B", DecompositionStrategy::Parallel);
    for (i, p) in parts.iter().enumerate() {
        assert_eq!(p.index, i);
    }
}

// ── QueryDecompositionEngine tests ────────────────────────────────────────────

#[tokio::test]
async fn test_engine_empty_query() {
    let engine = QueryDecompositionEngine::new(DecompositionConfig::default());
    let echo = MockEcho::new(vec![]);
    let err = engine.run("", &echo).await.expect_err("should fail");
    assert!(matches!(err, QueryDecompositionError::EmptyQuery));
}

#[tokio::test]
async fn test_engine_whitespace_query() {
    let engine = QueryDecompositionEngine::new(DecompositionConfig::default());
    let echo = MockEcho::new(vec![]);
    let err = engine.run("  ", &echo).await.expect_err("should fail");
    assert!(matches!(err, QueryDecompositionError::EmptyQuery));
}

#[tokio::test]
async fn test_engine_simple_query() {
    let docs = vec![
        make_result("d1", "Rust is a systems language", 0.9),
        make_result("d2", "Rust is memory-safe", 0.8),
    ];
    let engine = QueryDecompositionEngine::new(DecompositionConfig::default());
    let echo = MockEcho::new(docs);
    let result = engine
        .run("what is rust", &echo)
        .await
        .expect("should succeed");
    assert_eq!(result.original_query, "what is rust");
    assert!(!result.sub_questions.is_empty());
    assert!(!result.final_answer.is_empty());
}

#[tokio::test]
async fn test_engine_conjunction_query() {
    let docs = vec![make_result("d1", "relevant content", 0.9)];
    let engine = QueryDecompositionEngine::new(DecompositionConfig::default());
    let echo = MockEcho::new(docs);
    let result = engine
        .run("what is Rust and what is Go", &echo)
        .await
        .expect("should succeed");
    assert!(!result.sub_questions.is_empty());
    assert!(!result.sub_answers.is_empty());
    assert_eq!(result.sub_questions.len(), result.sub_answers.len());
}

#[tokio::test]
async fn test_engine_fused_results_non_empty() {
    let docs = vec![
        make_result("d1", "content a", 0.9),
        make_result("d2", "content b", 0.7),
    ];
    let engine = QueryDecompositionEngine::new(DecompositionConfig::default());
    let echo = MockEcho::new(docs);
    let result = engine.run("A and B query", &echo).await.expect("ok");
    // fused_results should contain at least the docs provided
    assert!(!result.fused_results.is_empty());
}

#[tokio::test]
async fn test_engine_no_results() {
    let engine = QueryDecompositionEngine::new(DecompositionConfig::default());
    let echo = MockEcho::new(vec![]);
    let result = engine.run("what is X", &echo).await.expect("ok");
    assert!(
        !result.final_answer.is_empty(),
        "fallback answer when no docs"
    );
}

#[tokio::test]
async fn test_engine_step_back_strategy() {
    let docs = vec![make_result("d1", "general principles of programming", 0.8)];
    let cfg = DecompositionConfig::default().with_strategy(DecompositionStrategy::StepBack);
    let engine = QueryDecompositionEngine::new(cfg);
    let echo = MockEcho::new(docs);
    let result = engine
        .run("how does ownership work in Rust", &echo)
        .await
        .expect("ok");
    assert!(
        result.sub_questions.len() >= 2,
        "step-back should yield 2+ sub-questions"
    );
}

#[tokio::test]
async fn test_engine_least_to_most_strategy() {
    let docs = vec![make_result("d1", "memory management details", 0.8)];
    let cfg = DecompositionConfig::default().with_strategy(DecompositionStrategy::LeastToMost);
    let engine = QueryDecompositionEngine::new(cfg);
    let echo = MockEcho::new(docs);
    let result = engine
        .run("how does Rust manage memory safely", &echo)
        .await
        .expect("ok");
    assert!(!result.sub_questions.is_empty());
    assert!(!result.final_answer.is_empty());
}

#[tokio::test]
async fn test_engine_decomposed_query_is_empty_check() {
    let docs = vec![make_result("d1", "content", 0.9)];
    let engine = QueryDecompositionEngine::new(DecompositionConfig::default());
    let echo = MockEcho::new(docs);
    let result = engine.run("simple query", &echo).await.expect("ok");
    assert!(!result.is_empty(), "should have sub-questions");
}
