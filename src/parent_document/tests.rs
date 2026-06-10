//! Tests for the `parent_document` module.

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::types::{Document, DocumentId, SearchResult};

use super::retriever::ParentDocumentRetriever;
use super::types::{
    ExpandedResult, ParentChildIndex, ParentDocumentConfig, ParentDocumentError, WindowConfig,
};

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

fn make_doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: make_doc(id, content),
        score,
        rank: 0,
    }
}

fn make_child_result(id: &str, parent_id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content)
            .with_id(DocumentId::from_string(id))
            .with_metadata("parent_id", parent_id),
        score,
        rank: 0,
    }
}

// ── ParentChildIndex tests ────────────────────────────────────────────────────

#[test]
fn test_index_empty() {
    let idx = ParentChildIndex::new();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
}

#[test]
fn test_index_register_and_lookup() {
    let mut idx = ParentChildIndex::new();
    let parent = make_doc("p1", "Parent content");
    let children = vec![make_doc("c1", "chunk 1"), make_doc("c2", "chunk 2")];
    idx.register(parent, &children);
    assert_eq!(idx.len(), 1);
    let found = idx.parent_of("c1");
    assert!(found.is_some());
    assert_eq!(found.unwrap().id.as_str(), "p1");
}

#[test]
fn test_index_unknown_child_returns_none() {
    let idx = ParentChildIndex::new();
    assert!(idx.parent_of("nonexistent").is_none());
}

#[test]
fn test_index_multiple_parents() {
    let mut idx = ParentChildIndex::new();
    idx.register(make_doc("p1", "Parent 1"), &[make_doc("c1", "chunk")]);
    idx.register(make_doc("p2", "Parent 2"), &[make_doc("c2", "chunk")]);
    assert_eq!(idx.len(), 2);
    assert_eq!(idx.parent_of("c1").unwrap().id.as_str(), "p1");
    assert_eq!(idx.parent_of("c2").unwrap().id.as_str(), "p2");
}

// ── ParentDocumentConfig tests ────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = ParentDocumentConfig::default();
    assert_eq!(cfg.top_k, 5);
    assert!(cfg.return_parent);
}

#[test]
fn test_config_builders() {
    let cfg = ParentDocumentConfig::default()
        .with_top_k(10)
        .with_return_parent(false);
    assert_eq!(cfg.top_k, 10);
    assert!(!cfg.return_parent);
}

// ── WindowConfig tests ────────────────────────────────────────────────────────

#[test]
fn test_window_config_default() {
    let w = WindowConfig::default();
    assert_eq!(w.window_size, 0);
}

#[test]
fn test_window_config_builder() {
    let w = WindowConfig::default().with_window_size(2);
    assert_eq!(w.window_size, 2);
}

// ── ParentDocumentRetriever tests ─────────────────────────────────────────────

#[tokio::test]
async fn test_retriever_empty_query_error() {
    let idx = ParentChildIndex::new();
    let cfg = ParentDocumentConfig::default();
    let retriever = ParentDocumentRetriever::new(idx, cfg);
    let echo = MockEcho::new(vec![]);
    let err = retriever.run("", &echo).await.expect_err("should fail");
    assert!(matches!(err, ParentDocumentError::EmptyQuery));
}

#[tokio::test]
async fn test_retriever_whitespace_query_error() {
    let idx = ParentChildIndex::new();
    let cfg = ParentDocumentConfig::default();
    let retriever = ParentDocumentRetriever::new(idx, cfg);
    let echo = MockEcho::new(vec![]);
    let err = retriever.run("  ", &echo).await.expect_err("should fail");
    assert!(matches!(err, ParentDocumentError::EmptyQuery));
}

#[tokio::test]
async fn test_retriever_no_results_returns_empty() {
    let idx = ParentChildIndex::new();
    let cfg = ParentDocumentConfig::default();
    let retriever = ParentDocumentRetriever::new(idx, cfg);
    let echo = MockEcho::new(vec![]);
    let results = retriever.run("query", &echo).await.expect("ok");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_retriever_return_parent_false_returns_children() {
    let idx = ParentChildIndex::new();
    let cfg = ParentDocumentConfig::default().with_return_parent(false);
    let retriever = ParentDocumentRetriever::new(idx, cfg);
    let echo = MockEcho::new(vec![make_result("child1", "content", 0.9)]);
    let results = retriever.run("query", &echo).await.expect("ok");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].parent.id.as_str(), "child1");
}

#[tokio::test]
async fn test_retriever_expands_to_parent() {
    let mut idx = ParentChildIndex::new();
    let parent = make_doc("p1", "Full parent content with much more detail.");
    let child = make_doc("c1", "small chunk");
    idx.register(parent, &[child]);

    let cfg = ParentDocumentConfig::default();
    let retriever = ParentDocumentRetriever::new(idx, cfg);
    let echo = MockEcho::new(vec![make_result("c1", "small chunk", 0.95)]);
    let results = retriever.run("query", &echo).await.expect("ok");
    // c1's parent_id is registered in index, should be expanded
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].parent.id.as_str(), "p1");
}

#[tokio::test]
async fn test_retriever_metadata_parent_id_expansion() {
    let idx = ParentChildIndex::new(); // empty — uses metadata fallback

    let cfg = ParentDocumentConfig::default();
    let retriever = ParentDocumentRetriever::new(idx, cfg);
    // Child carries parent_id in metadata but parent is not in index → skipped
    let child = make_child_result("c1", "p1", "chunk content", 0.9);
    let echo = MockEcho::new(vec![child]);
    let results = retriever.run("query", &echo).await.expect("ok");
    // Parent p1 is not in index → skipped (empty result)
    assert!(results.is_empty(), "parent not in index should be skipped");
}

#[tokio::test]
async fn test_retriever_dedup_parents() {
    let mut idx = ParentChildIndex::new();
    let parent = make_doc("p1", "Parent content");
    let c1 = make_doc("c1", "chunk 1");
    let c2 = make_doc("c2", "chunk 2");
    idx.register(parent, &[c1, c2]);

    let cfg = ParentDocumentConfig::default();
    let retriever = ParentDocumentRetriever::new(idx, cfg);
    // Both c1 and c2 match — should dedup to one parent result
    let echo = MockEcho::new(vec![
        make_result("c1", "chunk 1", 0.9),
        make_result("c2", "chunk 2", 0.8),
    ]);
    let results = retriever.run("query", &echo).await.expect("ok");
    assert_eq!(results.len(), 1, "two children of same parent → one result");
    assert_eq!(results[0].matched_children.len(), 2);
}

#[tokio::test]
async fn test_retriever_score_is_best_child_score() {
    let mut idx = ParentChildIndex::new();
    let parent = make_doc("p1", "Parent");
    let c1 = make_doc("c1", "c1");
    let c2 = make_doc("c2", "c2");
    idx.register(parent, &[c1, c2]);

    let cfg = ParentDocumentConfig::default();
    let retriever = ParentDocumentRetriever::new(idx, cfg);
    let echo = MockEcho::new(vec![
        make_result("c1", "c1", 0.7),
        make_result("c2", "c2", 0.95),
    ]);
    let results = retriever.run("query", &echo).await.expect("ok");
    assert!(
        (results[0].score - 0.95).abs() < 1e-5,
        "score should be max child score"
    );
}

#[test]
fn test_expanded_result_child_hit_count() {
    let r = ExpandedResult {
        parent: make_doc("p1", "content"),
        matched_children: vec![DocumentId::from_string("c1"), DocumentId::from_string("c2")],
        score: 0.9,
    };
    assert_eq!(r.child_hit_count(), 2);
}
