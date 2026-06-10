//! Tests for the `multi_hop` module.

use async_trait::async_trait;

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::layer4_graph::types::{EntityType, GraphEntity, GraphRelationship, RelationshipType};
use crate::types::{Document, DocumentId, SearchResult};

use super::traversal::{MultiHopRetriever, detect_entity_mentions, expand_hop};
use super::types::{HopConfig, HopState, MultiHopError, MultiHopResult};

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

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_result(content: &str, id: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

fn make_entity(name: &str, id: &str) -> GraphEntity {
    GraphEntity::new(name, EntityType::Concept).with_id(id)
}

fn make_rel(src: &str, tgt: &str) -> GraphRelationship {
    GraphRelationship::new(src, tgt, RelationshipType::RelatedTo)
}

// ── config tests ──────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = HopConfig::default();
    assert_eq!(cfg.max_hops, 3);
    assert_eq!(cfg.entities_per_hop, 5);
    assert_eq!(cfg.top_k, 5);
}

#[test]
fn test_config_builders() {
    let cfg = HopConfig::new()
        .with_max_hops(2)
        .with_entities_per_hop(4)
        .with_top_k(10);
    assert_eq!(cfg.max_hops, 2);
    assert_eq!(cfg.entities_per_hop, 4);
    assert_eq!(cfg.top_k, 10);
}

// ── entity mention tests ──────────────────────────────────────────────────────

#[test]
fn test_entity_mention_detection_exact_match() {
    let entities = vec![make_entity("Rust", "e1"), make_entity("Python", "e2")];
    let mentions = detect_entity_mentions("Tell me about Rust", &entities);
    assert_eq!(mentions.len(), 1);
    assert_eq!(mentions[0].text, "Rust");
    assert_eq!(mentions[0].matched_id.as_deref(), Some("e1"));
}

#[test]
fn test_entity_mention_no_match() {
    let entities = vec![make_entity("Rust", "e1")];
    let mentions = detect_entity_mentions("What is Python?", &entities);
    assert!(mentions.is_empty());
}

#[test]
fn test_entity_mention_case_insensitive() {
    let entities = vec![make_entity("Rust", "e1")];
    let mentions = detect_entity_mentions("RUST is great", &entities);
    assert_eq!(mentions.len(), 1);
    assert_eq!(mentions[0].matched_id.as_deref(), Some("e1"));
}

// ── expand_hop tests ──────────────────────────────────────────────────────────

#[test]
fn test_expand_hop_follows_source_to_target() {
    let rels = vec![make_rel("e1", "e2")];
    let expanded = expand_hop(&["e1".to_string()], &rels, 5);
    assert_eq!(expanded, vec!["e2".to_string()]);
}

#[test]
fn test_expand_hop_follows_target_to_source() {
    let rels = vec![make_rel("e2", "e1")];
    // e1 is the current entity; e2 should be reachable via the reverse edge.
    let expanded = expand_hop(&["e1".to_string()], &rels, 5);
    assert_eq!(expanded, vec!["e2".to_string()]);
}

#[test]
fn test_expand_hop_no_neighbors() {
    let rels = vec![make_rel("e3", "e4")];
    let expanded = expand_hop(&["e1".to_string()], &rels, 5);
    assert!(expanded.is_empty());
}

#[test]
fn test_expand_hop_skips_current_entities() {
    let rels = vec![make_rel("e1", "e2"), make_rel("e2", "e1")];
    // Both e1 and e2 are current; nothing should expand.
    let expanded = expand_hop(&["e1".to_string(), "e2".to_string()], &rels, 5);
    assert!(expanded.is_empty());
}

// ── retriever async tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_retriever_empty_query_error() {
    let retriever = MultiHopRetriever::default();
    let echo = MockEcho::new(vec![]);
    let result = retriever.run("", &[], &[], &echo).await;
    assert!(matches!(result, Err(MultiHopError::EmptyQuery)));
}

#[tokio::test]
async fn test_retriever_no_entities_error() {
    let retriever = MultiHopRetriever::default();
    let echo = MockEcho::new(vec![]);
    let result = retriever.run("What is Rust?", &[], &[], &echo).await;
    assert!(matches!(result, Err(MultiHopError::NoEntitiesFound)));
}

#[tokio::test]
async fn test_retriever_single_hop() {
    let entities = vec![make_entity("Rust", "e1")];
    let echo = MockEcho::new(vec![make_result("Rust is a systems language.", "d1", 0.9)]);
    let retriever = MultiHopRetriever::new(HopConfig::new().with_max_hops(1));

    let result = retriever
        .run("Tell me about Rust", &entities, &[], &echo)
        .await;
    assert!(result.is_ok());
    let mh = result.expect("multi-hop should succeed");
    assert_eq!(mh.hops_used, 1);
    assert!(mh.has_answer());
}

#[tokio::test]
async fn test_retriever_multi_hop() {
    let entities = vec![make_entity("Rust", "e1"), make_entity("LLVM", "e2")];
    let rels = vec![make_rel("e1", "e2")];
    let echo = MockEcho::new(vec![make_result("Rust uses LLVM.", "d1", 0.9)]);
    let retriever = MultiHopRetriever::new(HopConfig::new().with_max_hops(2));

    let result = retriever
        .run("Tell me about Rust", &entities, &rels, &echo)
        .await;
    assert!(result.is_ok());
    let mh = result.expect("multi-hop should succeed");
    assert!(mh.hops_used >= 1);
}

#[tokio::test]
async fn test_retriever_no_results_returns_empty_answer() {
    let entities = vec![make_entity("Rust", "e1")];
    let echo = MockEcho::new(vec![]);
    let retriever = MultiHopRetriever::new(HopConfig::new().with_max_hops(1));

    let result = retriever
        .run("Tell me about Rust", &entities, &[], &echo)
        .await;
    assert!(result.is_ok());
    let mh = result.expect("multi-hop should succeed");
    assert!(!mh.has_answer());
}

// ── MultiHopResult tests ──────────────────────────────────────────────────────

#[test]
fn test_multi_hop_result_total_docs() {
    let trace = vec![
        HopState::new(0, vec![], vec!["d1".to_string(), "d2".to_string()]),
        HopState::new(1, vec![], vec!["d2".to_string(), "d3".to_string()]),
    ];
    let r = MultiHopResult::new("answer".to_string(), trace, 2);
    // d1, d2, d3 are unique
    assert_eq!(r.total_docs(), 3);
}

#[test]
fn test_multi_hop_result_has_answer() {
    let r_yes = MultiHopResult::new("some answer".to_string(), vec![], 0);
    let r_no = MultiHopResult::new("   ".to_string(), vec![], 0);
    assert!(r_yes.has_answer());
    assert!(!r_no.has_answer());
}

// ── HopState tests ────────────────────────────────────────────────────────────

#[test]
fn test_hop_state_fields() {
    let state = HopState::new(
        2,
        vec!["e1".to_string()],
        vec!["d1".to_string(), "d2".to_string()],
    );
    assert_eq!(state.hop, 2);
    assert_eq!(state.entity_ids.len(), 1);
    assert_eq!(state.doc_ids.len(), 2);
}

// ── error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    assert_eq!(
        MultiHopError::EmptyQuery.to_string(),
        "Query must not be empty"
    );
    assert_eq!(
        MultiHopError::NoEntitiesFound.to_string(),
        "No entities matched the query"
    );
    let msg = MultiHopError::RetrievalFailed("timeout".to_string()).to_string();
    assert!(msg.contains("timeout"));
}
