//! Tests for the advanced retrieval module.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::advanced_retrieval::rag_fusion::reciprocal_rank_fusion;
use crate::advanced_retrieval::{
    AdvancedRetrievalError, HydeConfig, HydeRetrieval, MmrConfig, MmrReranker, RagFusion,
    RagFusionConfig, RetrievalConfig, RetrievalStrategy,
};
use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::types::{Document, DocumentId, SearchResult};

// ── MockEcho ──────────────────────────────────────────────────────────────────

/// A minimal [`Echo`] implementation for tests.
///
/// Every call to [`search`] returns `self.results.clone()` regardless of the
/// query string, which is sufficient for smoke-testing the higher-level
/// retrieval strategies.
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

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32, rank: usize) -> SearchResult {
    SearchResult::new(Document::new(content).with_id(id), score, rank)
}

fn unit_emb(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    if axis < dim {
        v[axis] = 1.0;
    }
    v
}

// ── RetrievalConfig ───────────────────────────────────────────────────────────

#[test]
fn test_retrieval_config_defaults() {
    let cfg = RetrievalConfig::default();
    assert_eq!(cfg.top_k, 10);
    assert!(cfg.min_score.is_none());
}

#[test]
fn test_retrieval_config_builders() {
    let cfg = RetrievalConfig::default().with_top_k(5).with_min_score(0.7);
    assert_eq!(cfg.top_k, 5);
    assert_eq!(cfg.min_score, Some(0.7));
}

#[test]
fn test_retrieval_strategy_variants() {
    // Just ensure the enum variants exist and are comparable.
    assert_ne!(RetrievalStrategy::RagFusion, RetrievalStrategy::Hyde);
    assert_ne!(RetrievalStrategy::Hyde, RetrievalStrategy::Mmr);
    assert_ne!(RetrievalStrategy::RagFusion, RetrievalStrategy::Mmr);
}

// ── AdvancedRetrievalError ────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    let e = AdvancedRetrievalError::Embedding("dim mismatch".into());
    assert!(e.to_string().contains("Embedding"));
    assert!(e.to_string().contains("dim mismatch"));

    let e = AdvancedRetrievalError::Search("timeout".into());
    assert!(e.to_string().contains("Search"));

    let e = AdvancedRetrievalError::Internal("oops".into());
    assert!(e.to_string().contains("Internal"));
}

// ── RagFusionConfig ───────────────────────────────────────────────────────────

#[test]
fn test_rag_fusion_config_defaults() {
    let cfg = RagFusionConfig::default();
    assert_eq!(cfg.num_queries, 4);
    assert!((cfg.rrf_k - 60.0).abs() < 1e-4);
    assert_eq!(cfg.top_k, 10);
}

#[test]
fn test_rag_fusion_config_builders() {
    let cfg = RagFusionConfig::default()
        .with_num_queries(6)
        .with_rrf_k(30.0)
        .with_top_k(5);
    assert_eq!(cfg.num_queries, 6);
    assert!((cfg.rrf_k - 30.0).abs() < 1e-4);
    assert_eq!(cfg.top_k, 5);
}

// ── RagFusion::generate_variants ─────────────────────────────────────────────

#[test]
fn test_generate_variants_count() {
    let fusion = RagFusion::new(RagFusionConfig::default());
    let variants = fusion.generate_variants("What is Rust?");
    assert_eq!(
        variants.len(),
        4,
        "Should generate exactly num_queries variants"
    );
}

#[test]
fn test_generate_variants_first_is_original() {
    let fusion = RagFusion::new(RagFusionConfig::default());
    let query = "How does Rust handle memory safety?";
    let variants = fusion.generate_variants(query);
    assert_eq!(variants[0], query);
}

#[test]
fn test_generate_variants_all_non_empty() {
    let fusion = RagFusion::new(RagFusionConfig::default());
    let variants = fusion.generate_variants("Explain async-await in Rust");
    for (i, v) in variants.iter().enumerate() {
        assert!(!v.is_empty(), "Variant {i} is empty");
    }
}

#[test]
fn test_generate_variants_num_queries_one() {
    let cfg = RagFusionConfig::default().with_num_queries(1);
    let fusion = RagFusion::new(cfg);
    let variants = fusion.generate_variants("only one");
    assert_eq!(variants.len(), 1);
    assert_eq!(variants[0], "only one");
}

#[test]
fn test_generate_variants_large_num_queries() {
    let cfg = RagFusionConfig::default().with_num_queries(10);
    let fusion = RagFusion::new(cfg);
    let variants = fusion.generate_variants("What is deep learning?");
    assert_eq!(variants.len(), 10);
    for v in &variants {
        assert!(!v.is_empty());
    }
}

#[test]
fn test_generate_variants_single_word_query() {
    let fusion = RagFusion::new(RagFusionConfig::default());
    let variants = fusion.generate_variants("Rust");
    assert_eq!(variants.len(), 4);
    assert_eq!(variants[0], "Rust");
    for v in &variants {
        assert!(!v.is_empty());
    }
}

// ── reciprocal_rank_fusion ────────────────────────────────────────────────────

#[test]
fn test_rrf_empty_input() {
    let result = reciprocal_rank_fusion(&[], 60.0);
    assert!(result.is_empty());
}

#[test]
fn test_rrf_single_list() {
    let list = vec![
        make_result("a", "doc a", 0.9, 0),
        make_result("b", "doc b", 0.8, 1),
    ];
    let fused = reciprocal_rank_fusion(&[list], 60.0);
    assert_eq!(fused.len(), 2);
    // Doc "a" is rank 1 in the single list → higher RRF score than "b" (rank 2).
    assert_eq!(fused[0].document.id.as_str(), "a");
}

#[test]
fn test_rrf_deduplication() {
    // Same doc appears in both lists — scores should be summed.
    let list1 = vec![make_result("shared", "shared doc", 0.9, 0)];
    let list2 = vec![make_result("shared", "shared doc", 0.7, 0)];
    let fused = reciprocal_rank_fusion(&[list1, list2], 60.0);
    assert_eq!(fused.len(), 1, "Duplicate doc should be deduplicated");
    // Score = 1/(60+1) + 1/(60+1) = 2/61
    let expected = 2.0_f32 / 61.0;
    assert!((fused[0].score - expected).abs() < 1e-5);
}

#[test]
fn test_rrf_known_scores() {
    // Doc "a" at rank 1 in list 0, rank 2 in list 1.
    // Doc "b" at rank 2 in list 0, rank 1 in list 1.
    let list0 = vec![make_result("a", "a", 0.9, 0), make_result("b", "b", 0.8, 1)];
    let list1 = vec![
        make_result("b", "b", 0.85, 0),
        make_result("a", "a", 0.75, 1),
    ];
    let fused = reciprocal_rank_fusion(&[list0, list1], 60.0);

    // a: 1/61 + 1/62 ≈ 0.016393 + 0.016129 ≈ 0.032522
    // b: 1/62 + 1/61 ≈ same → tie → alphabetical order
    assert_eq!(fused.len(), 2);
    let score_a: f32 = 1.0 / 61.0 + 1.0 / 62.0;
    let score_b: f32 = 1.0 / 62.0 + 1.0 / 61.0;
    // Both scores are equal; order is alphabetical by doc id ("a" < "b").
    assert!((fused[0].score - score_a).abs() < 1e-4);
    assert!((fused[1].score - score_b).abs() < 1e-4);
}

#[test]
fn test_rrf_result_ranks_reassigned() {
    let list = vec![
        make_result("x", "x", 0.9, 99),
        make_result("y", "y", 0.5, 77),
    ];
    let fused = reciprocal_rank_fusion(&[list], 60.0);
    assert_eq!(fused[0].rank, 0);
    assert_eq!(fused[1].rank, 1);
}

// ── RagFusion::retrieve ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_rag_fusion_retrieve_smoke_test() {
    let mock_results = vec![
        make_result("d1", "first doc", 0.9, 0),
        make_result("d2", "second doc", 0.8, 1),
    ];
    let echo = MockEcho::new(mock_results);

    let fusion = RagFusion::new(RagFusionConfig::default().with_top_k(5));
    let results = fusion
        .retrieve("test query", &echo)
        .await
        .expect("retrieve should succeed");

    // MockEcho returns the same 2 results for every variant; after dedup we
    // still have exactly 2 unique documents.
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_rag_fusion_retrieve_empty_results() {
    let echo = MockEcho::new(vec![]);
    let fusion = RagFusion::new(RagFusionConfig::default());
    let results = fusion
        .retrieve("query", &echo)
        .await
        .expect("retrieve should succeed");
    assert!(results.is_empty());
}

// ── HydeConfig ────────────────────────────────────────────────────────────────

#[test]
fn test_hyde_config_defaults() {
    let cfg = HydeConfig::default();
    assert_eq!(cfg.top_k, 10);
    assert_eq!(cfg.hypothetical_prefix, "Answer to: ");
}

#[test]
fn test_hyde_config_builders() {
    let cfg = HydeConfig::default()
        .with_top_k(3)
        .with_hypothetical_prefix("In response to: ");
    assert_eq!(cfg.top_k, 3);
    assert_eq!(cfg.hypothetical_prefix, "In response to: ");
}

// ── HydeRetrieval::generate_hypothetical_doc ─────────────────────────────────

#[test]
fn test_hyde_generate_non_empty() {
    let hyde = HydeRetrieval::new(HydeConfig::default());
    let doc = hyde.generate_hypothetical_doc("What is Rust?");
    assert!(!doc.is_empty());
}

#[test]
fn test_hyde_generate_contains_query() {
    let hyde = HydeRetrieval::new(HydeConfig::default());
    let query = "What is Rust programming language?";
    let doc = hyde.generate_hypothetical_doc(query);
    assert!(
        doc.contains(query),
        "Hypothetical doc should contain the original query"
    );
}

#[test]
fn test_hyde_generate_contains_prefix() {
    let cfg = HydeConfig::default().with_hypothetical_prefix("ANSWER: ");
    let hyde = HydeRetrieval::new(cfg);
    let doc = hyde.generate_hypothetical_doc("some query");
    assert!(
        doc.starts_with("ANSWER: "),
        "Doc should start with the configured prefix"
    );
}

#[test]
fn test_hyde_generate_longer_than_query() {
    let hyde = HydeRetrieval::new(HydeConfig::default());
    let query = "What is machine learning?";
    let doc = hyde.generate_hypothetical_doc(query);
    assert!(
        doc.len() > query.len(),
        "Hypothetical doc should be longer than the query"
    );
}

#[test]
fn test_hyde_generate_contains_keywords() {
    let hyde = HydeRetrieval::new(HydeConfig::default());
    // Query has content words: "machine", "learning", "algorithms"
    let doc = hyde.generate_hypothetical_doc("What are machine learning algorithms?");
    // At least one of the content words should appear in the expansion.
    let has_keyword =
        doc.contains("machine") || doc.contains("learning") || doc.contains("algorithms");
    assert!(has_keyword, "Doc should contain query keywords; got: {doc}");
}

// ── HydeRetrieval::retrieve ───────────────────────────────────────────────────

#[tokio::test]
async fn test_hyde_retrieve_smoke_test() {
    let mock_results = vec![make_result("h1", "hypothetical match", 0.88, 0)];
    let echo = MockEcho::new(mock_results);

    let hyde = HydeRetrieval::new(HydeConfig::default());
    let results = hyde
        .retrieve("How does a compiler work?", &echo)
        .await
        .expect("retrieve should succeed");
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_hyde_retrieve_empty_echo() {
    let echo = MockEcho::new(vec![]);
    let hyde = HydeRetrieval::new(HydeConfig::default());
    let results = hyde
        .retrieve("query", &echo)
        .await
        .expect("retrieve should succeed");
    assert!(results.is_empty());
}

// ── MmrConfig ────────────────────────────────────────────────────────────────

#[test]
fn test_mmr_config_defaults() {
    let cfg = MmrConfig::default();
    assert!((cfg.lambda - 0.5).abs() < 1e-4);
    assert_eq!(cfg.top_k, 5);
}

#[test]
fn test_mmr_config_builders() {
    let cfg = MmrConfig::default().with_lambda(0.7).with_top_k(3);
    assert!((cfg.lambda - 0.7).abs() < 1e-4);
    assert_eq!(cfg.top_k, 3);
}

// ── MmrReranker::rerank ───────────────────────────────────────────────────────

#[test]
fn test_mmr_empty_candidates() {
    let reranker = MmrReranker::new(MmrConfig::default());
    let result = reranker.rerank(&[], &[1.0, 0.0], &HashMap::new());
    assert!(result.is_empty());
}

#[test]
fn test_mmr_returns_top_k() {
    let reranker = MmrReranker::new(MmrConfig::default().with_top_k(2));
    let candidates = vec![
        make_result("a", "a", 0.9, 0),
        make_result("b", "b", 0.8, 1),
        make_result("c", "c", 0.7, 2),
    ];
    let q_emb = vec![1.0_f32, 0.0];
    let doc_embeddings: HashMap<String, Vec<f32>> = [
        ("a".to_string(), vec![1.0_f32, 0.0]),
        ("b".to_string(), vec![0.9_f32, 0.1_f32.sqrt()]),
        ("c".to_string(), vec![0.0_f32, 1.0]),
    ]
    .into_iter()
    .collect();

    let result = reranker.rerank(&candidates, &q_emb, &doc_embeddings);
    assert_eq!(result.len(), 2, "Should return top_k=2 results");
}

#[test]
fn test_mmr_diversity_beats_redundancy() {
    // Query embedding: [1, 0]
    // doc A: [1, 0]  — maximally relevant to query (sim = 1.0)
    // doc B: [1, 0]  — identical to A (sim_to_query = 1.0, sim_to_A = 1.0)
    // doc C: [0.5, 0.866] — has moderate relevance (sim_to_query ≈ 0.5) but
    //                        is diverse (sim_to_A ≈ 0.5)
    //
    // After selecting A:
    //   MMR(B) = λ * 1.0 - (1-λ) * 1.0  = 0.5 - 0.5 = 0.0
    //   MMR(C) = λ * 0.5 - (1-λ) * 0.5  = 0.25 - 0.25 = 0.0
    //
    // The scores tie, so let's make the diversification clearer:
    // Use lambda=0.7 (more relevance weight) and give B a big redundancy:
    //   MMR(B) = 0.7*1.0 - 0.3*1.0 = 0.4
    //   MMR(C) = 0.7*0.3 - 0.3*0.0 = 0.21  (C is orthogonal to A → diversity bonus)
    //
    // Actually the clearest proof is: make B *less* relevant than C but also
    // redundant, so B's redundancy penalty is larger than its relevance.
    //
    // Setup:
    //   q: [1, 0]
    //   A: [1, 0]  sim_q=1.0
    //   B: [0.95, 0] sim_q≈0.95, sim_A≈1.0 (almost identical to A)
    //   C: [0, 1]  sim_q=0.0, sim_A=0.0 (orthogonal to A → zero redundancy)
    //
    // With lambda=0.3 (diversity dominant):
    //   MMR(B) = 0.3*0.95 - 0.7*1.0 = 0.285 - 0.7 = -0.415
    //   MMR(C) = 0.3*0.0  - 0.7*0.0 = 0.0
    // → C wins because it has zero redundancy.
    let reranker = MmrReranker::new(MmrConfig::default().with_lambda(0.3).with_top_k(2));

    let candidates = vec![
        make_result("a", "doc a", 0.9, 0),
        make_result("b", "doc b", 0.85, 1), // near-duplicate of A, high relevance
        make_result("c", "doc c", 0.1, 2),  // orthogonal to A, low relevance
    ];

    let q_emb = vec![1.0_f32, 0.0];
    let doc_embeddings: HashMap<String, Vec<f32>> = [
        ("a".to_string(), vec![1.0_f32, 0.0]),
        ("b".to_string(), vec![0.95_f32, 0.0]), // very close to A
        ("c".to_string(), vec![0.0_f32, 1.0]),  // orthogonal to both A and B
    ]
    .into_iter()
    .collect();

    let result = reranker.rerank(&candidates, &q_emb, &doc_embeddings);
    assert_eq!(result.len(), 2);

    // First should be A (most relevant).
    assert_eq!(
        result[0].document.id.as_str(),
        "a",
        "First result should be A"
    );

    // With diversity-dominant lambda=0.3: after selecting A,
    // B has high redundancy (sim_B_A≈1.0) → MMR(B)= 0.3*0.95 - 0.7*1.0 ≈ -0.415
    // C has zero redundancy (sim_C_A=0.0)  → MMR(C)= 0.3*0.0 - 0.7*0.0 = 0.0
    // So C > B and C should be selected.
    assert_eq!(
        result[1].document.id.as_str(),
        "c",
        "Second result should be C (diverse), not B (redundant)"
    );
}

#[test]
fn test_mmr_pure_relevance_lambda_one() {
    // lambda=1.0 → pure relevance → original order preserved
    let reranker = MmrReranker::new(MmrConfig::default().with_lambda(1.0).with_top_k(3));

    let candidates = vec![
        make_result("a", "a", 0.9, 0),
        make_result("b", "b", 0.8, 1),
        make_result("c", "c", 0.7, 2),
    ];

    let q_emb = vec![1.0_f32, 0.0, 0.0];
    let doc_embeddings: HashMap<String, Vec<f32>> = [
        ("a".to_string(), vec![1.0_f32, 0.0, 0.0]),
        ("b".to_string(), vec![0.8_f32, 0.6, 0.0]),
        ("c".to_string(), vec![0.6_f32, 0.8, 0.0]),
    ]
    .into_iter()
    .collect();

    let result = reranker.rerank(&candidates, &q_emb, &doc_embeddings);
    assert_eq!(result[0].document.id.as_str(), "a");
    assert_eq!(result[1].document.id.as_str(), "b");
    assert_eq!(result[2].document.id.as_str(), "c");
}

#[test]
fn test_mmr_ranks_reassigned_from_zero() {
    let reranker = MmrReranker::new(MmrConfig::default().with_top_k(2));
    let candidates = vec![
        make_result("x", "x", 0.9, 99),
        make_result("y", "y", 0.5, 44),
    ];
    let q_emb = unit_emb(2, 0);
    let doc_embeddings: HashMap<String, Vec<f32>> = [
        ("x".to_string(), unit_emb(2, 0)),
        ("y".to_string(), unit_emb(2, 1)),
    ]
    .into_iter()
    .collect();

    let result = reranker.rerank(&candidates, &q_emb, &doc_embeddings);
    assert_eq!(result[0].rank, 0);
    assert_eq!(result[1].rank, 1);
}

#[test]
fn test_mmr_missing_embedding_fallback() {
    // Documents with no entry in doc_embeddings → original score used for relevance.
    let reranker = MmrReranker::new(MmrConfig::default().with_top_k(2));
    let candidates = vec![make_result("a", "a", 0.9, 0), make_result("b", "b", 0.3, 1)];
    let q_emb = vec![1.0_f32];
    let doc_embeddings: HashMap<String, Vec<f32>> = HashMap::new(); // empty

    let result = reranker.rerank(&candidates, &q_emb, &doc_embeddings);
    // Should not panic and should return top_k results.
    assert_eq!(result.len(), 2);
}

// ── cosine similarity (via MmrReranker internals via rerank) ─────────────────

#[test]
fn test_mmr_cosine_orthogonal_gives_zero_redundancy() {
    // With lambda=0.5 and orthogonal doc embeddings, redundancy = 0 for all
    // pairs → selection order is purely relevance-driven.
    let reranker = MmrReranker::new(MmrConfig::default().with_lambda(0.5).with_top_k(3));

    // All three docs are mutually orthogonal.
    let candidates = vec![
        make_result("a", "a", 0.9, 0),
        make_result("b", "b", 0.7, 1),
        make_result("c", "c", 0.5, 2),
    ];
    let q_emb = unit_emb(3, 0);
    let doc_embeddings: HashMap<String, Vec<f32>> = [
        ("a".to_string(), unit_emb(3, 0)),
        ("b".to_string(), unit_emb(3, 1)),
        ("c".to_string(), unit_emb(3, 2)),
    ]
    .into_iter()
    .collect();

    let result = reranker.rerank(&candidates, &q_emb, &doc_embeddings);
    // First should be 'a' (most relevant to query axis 0).
    assert_eq!(result[0].document.id.as_str(), "a");
}
