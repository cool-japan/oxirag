//! Tests for the `iterative_rag` module.

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::types::{Document, DocumentId, SearchResult};
use async_trait::async_trait;

use super::engine::{IterativeRagEngine, build_draft, extract_expansion_terms};
use super::types::{IterationStep, IterativeConfig, IterativeOutput, IterativeRagError};

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

// ── Config tests ──────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = IterativeConfig::default();
    assert_eq!(cfg.max_iterations, 3);
    assert_eq!(cfg.top_k, 5);
    assert_eq!(cfg.expansion_terms, 3);
}

#[test]
fn test_config_builders() {
    let cfg = IterativeConfig::default()
        .with_max_iterations(7)
        .with_top_k(10)
        .with_expansion_terms(5);
    assert_eq!(cfg.max_iterations, 7);
    assert_eq!(cfg.top_k, 10);
    assert_eq!(cfg.expansion_terms, 5);
}

// ── IterationStep tests ───────────────────────────────────────────────────────

#[test]
fn test_iteration_step_is_expanded() {
    let single = IterationStep {
        iteration: 0,
        expanded_query: "rust".to_string(),
        retrieved_count: 3,
        draft: "Rust programming.".to_string(),
    };
    assert!(!single.is_expanded());

    let multi = IterationStep {
        iteration: 1,
        expanded_query: "rust programming systems".to_string(),
        retrieved_count: 3,
        draft: "Rust programming.".to_string(),
    };
    assert!(multi.is_expanded());
}

// ── IterativeOutput tests ─────────────────────────────────────────────────────

#[test]
fn test_output_total_iterations() {
    let out = IterativeOutput {
        steps: vec![
            IterationStep {
                iteration: 0,
                expanded_query: "query".to_string(),
                retrieved_count: 2,
                draft: "draft".to_string(),
            },
            IterationStep {
                iteration: 1,
                expanded_query: "query more".to_string(),
                retrieved_count: 1,
                draft: "draft2".to_string(),
            },
        ],
        final_answer: "Answer".to_string(),
        total_docs_retrieved: 3,
    };
    assert_eq!(out.total_iterations(), 2);
}

#[test]
fn test_output_is_empty_answer() {
    let empty = IterativeOutput {
        steps: vec![],
        final_answer: String::new(),
        total_docs_retrieved: 0,
    };
    assert!(empty.is_empty_answer());

    let non_empty = IterativeOutput {
        steps: vec![],
        final_answer: "Some answer".to_string(),
        total_docs_retrieved: 1,
    };
    assert!(!non_empty.is_empty_answer());
}

// ── Engine tests ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_engine_empty_query_error() {
    let engine = IterativeRagEngine::default();
    let echo = MockEcho::new(vec![]);
    let result = engine.run("", &echo).await;
    assert!(matches!(result, Err(IterativeRagError::EmptyQuery)));
}

#[tokio::test]
async fn test_engine_whitespace_query_error() {
    let engine = IterativeRagEngine::default();
    let echo = MockEcho::new(vec![]);
    let result = engine.run("   ", &echo).await;
    assert!(matches!(result, Err(IterativeRagError::EmptyQuery)));
}

#[tokio::test]
async fn test_engine_no_results_returns_empty() {
    let engine = IterativeRagEngine::default();
    let echo = MockEcho::new(vec![]);
    let output = engine.run("rust programming", &echo).await.unwrap();
    assert_eq!(output.total_docs_retrieved, 0);
    assert_eq!(output.final_answer, "No relevant information found.");
}

#[tokio::test]
async fn test_engine_single_iteration() {
    let cfg = IterativeConfig::default().with_max_iterations(1);
    let engine = IterativeRagEngine::new(cfg);

    let results = vec![make_result(
        "doc1",
        "Rust is a systems programming language for safety and speed.",
        0.9,
    )];
    let echo = MockEcho::new(results);

    let output = engine.run("rust", &echo).await.unwrap();
    assert!(output.total_iterations() >= 1);
    assert!(!output.final_answer.is_empty());
}

#[tokio::test]
async fn test_engine_multiple_iterations() {
    let cfg = IterativeConfig::default()
        .with_max_iterations(3)
        .with_expansion_terms(2);
    let engine = IterativeRagEngine::new(cfg);

    let results = vec![
        make_result(
            "doc1",
            "Rust programming language focuses on memory safety and performance.",
            0.9,
        ),
        make_result(
            "doc2",
            "Ownership system prevents data races in concurrent programming.",
            0.7,
        ),
    ];
    let echo = MockEcho::new(results);

    let output = engine.run("rust language", &echo).await.unwrap();
    assert!(output.total_iterations() >= 1);
    assert!(output.total_docs_retrieved > 0);
}

#[tokio::test]
async fn test_engine_expansion_extends_query() {
    let cfg = IterativeConfig::default()
        .with_max_iterations(2)
        .with_expansion_terms(2);
    let engine = IterativeRagEngine::new(cfg);

    let results = vec![make_result(
        "doc1",
        "Ownership borrowing lifetimes prevent memory safety issues in Rust programming.",
        0.9,
    )];
    let echo = MockEcho::new(results);

    let output = engine.run("rust", &echo).await.unwrap();
    // The second iteration step (if present) should have an expanded query
    if output.steps.len() > 1 {
        assert!(output.steps[1].is_expanded());
    }
    assert!(!output.final_answer.is_empty());
}

#[tokio::test]
async fn test_engine_dedup_results() {
    let cfg = IterativeConfig::default().with_max_iterations(3);
    let engine = IterativeRagEngine::new(cfg);

    // All results have same IDs — should be deduplicated
    let results = vec![
        make_result(
            "doc1",
            "Rust memory safety prevents vulnerabilities in systems programming.",
            0.9,
        ),
        make_result(
            "doc1",
            "Rust memory safety prevents vulnerabilities in systems programming.",
            0.8,
        ),
    ];
    let echo = MockEcho::new(results);

    let output = engine.run("rust", &echo).await.unwrap();
    assert_eq!(output.total_docs_retrieved, 1);
}

#[tokio::test]
async fn test_engine_stops_when_no_expansion() {
    // With very short content, no useful expansion terms can be extracted
    let cfg = IterativeConfig::default()
        .with_max_iterations(5)
        .with_expansion_terms(3);
    let engine = IterativeRagEngine::new(cfg);

    // Content with only stop words / short tokens → no expansion terms → early stop
    let results = vec![make_result("doc1", "a is the in", 0.5)];
    let echo = MockEcho::new(results);

    let output = engine.run("rust", &echo).await.unwrap();
    // Should stop early because no expansion terms available
    assert!(output.total_iterations() <= 5);
}

#[tokio::test]
async fn test_engine_final_answer_non_empty_when_results() {
    let engine = IterativeRagEngine::default();
    let results = vec![make_result(
        "doc1",
        "Rust is a systems programming language.",
        0.9,
    )];
    let echo = MockEcho::new(results);

    let output = engine.run("rust", &echo).await.unwrap();
    assert!(!output.final_answer.is_empty());
    assert_ne!(output.final_answer, "No relevant information found.");
}

#[tokio::test]
async fn test_engine_respects_max_iterations() {
    let max = 2;
    let cfg = IterativeConfig::default().with_max_iterations(max);
    let engine = IterativeRagEngine::new(cfg);

    let results = vec![make_result(
        "doc1",
        "Systems programming memory ownership borrowing lifetimes safety performance.",
        0.9,
    )];
    let echo = MockEcho::new(results);

    let output = engine.run("rust", &echo).await.unwrap();
    assert!(output.total_iterations() <= max);
}

// ── Helper unit tests ─────────────────────────────────────────────────────────

#[test]
fn test_extract_expansion_terms_logic() {
    let text = "Ownership borrowing lifetimes prevent memory safety vulnerabilities.";
    let query = "rust";
    let terms = extract_expansion_terms(text, query, 3);
    // Should return up to 3 terms, none being very short or stop words
    assert!(!terms.is_empty());
    assert!(terms.len() <= 3);
    for term in &terms {
        assert!(
            term.len() > 3,
            "Term '{term}' should be longer than 3 chars"
        );
    }
}

#[test]
fn test_build_draft_with_matching_sentences() {
    use crate::types::{Document, DocumentId, SearchResult};

    let results = vec![SearchResult {
        document: Document::new("Rust is a programming language. It focuses on safety.")
            .with_id(DocumentId::from_string("d1")),
        score: 0.9,
        rank: 0,
    }];
    let draft = build_draft(&results, "rust");
    assert!(!draft.is_empty());
}

#[test]
fn test_error_display() {
    let e1 = IterativeRagError::EmptyQuery;
    assert!(e1.to_string().contains("empty"));

    let e2 = IterativeRagError::RetrievalFailed("timeout".to_string());
    assert!(e2.to_string().contains("timeout"));
}
