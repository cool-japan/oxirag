//! Tests for the `corrective_rag` module.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::types::{Document, DocumentId, SearchResult};

use super::engine::CorrectiveRagEngine;
use super::grader::{HeuristicRetrievalGrader, MockRetrievalGrader, RetrievalGrader};
use super::strip::{KnowledgeRefiner, QueryRefiner};
use super::types::{
    CorrectiveAction, CorrectiveRagError, CragConfig, KnowledgeStrip, RetrievalGrade,
};

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
    async fn index(&mut self, _doc: Document) -> Result<DocumentId, EmbeddingError> {
        Ok(DocumentId::new())
    }

    async fn index_batch(
        &mut self,
        docs: Vec<Document>,
    ) -> Result<Vec<DocumentId>, EmbeddingError> {
        Ok(docs.iter().map(|_| DocumentId::new()).collect())
    }

    async fn search(
        &self,
        _query: &str,
        top_k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, EmbeddingError> {
        Ok(self.results.iter().take(top_k).cloned().collect())
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

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from(id)),
        score,
        rank: 0,
    }
}

// ── Config/enums ──────────────────────────────────────────────────────────────

#[test]
fn test_crag_config_defaults() {
    let cfg = CragConfig::default();
    assert!((cfg.upper_threshold - 0.7).abs() < 1e-5);
    assert!((cfg.lower_threshold - 0.3).abs() < 1e-5);
    assert_eq!(cfg.max_corrections, 2);
    assert!(cfg.enable_knowledge_strip);
    assert!((cfg.strip_relevance_threshold - 0.5).abs() < 1e-5);
    assert!((cfg.mmr_lambda - 0.5).abs() < 1e-5);
    assert_eq!(cfg.recomposition_dim, 256);
    assert_eq!(cfg.top_k, 5);
}

#[test]
fn test_crag_config_builders() {
    let cfg = CragConfig::default()
        .with_upper_threshold(0.9)
        .with_lower_threshold(0.1)
        .with_max_corrections(3)
        .with_knowledge_strip(false)
        .with_strip_relevance_threshold(0.4)
        .with_mmr_lambda(0.8)
        .with_recomposition_dim(128)
        .with_top_k(10);

    assert!((cfg.upper_threshold - 0.9).abs() < 1e-5);
    assert!((cfg.lower_threshold - 0.1).abs() < 1e-5);
    assert_eq!(cfg.max_corrections, 3);
    assert!(!cfg.enable_knowledge_strip);
    assert!((cfg.strip_relevance_threshold - 0.4).abs() < 1e-5);
    assert!((cfg.mmr_lambda - 0.8).abs() < 1e-5);
    assert_eq!(cfg.recomposition_dim, 128);
    assert_eq!(cfg.top_k, 10);
}

#[test]
fn test_retrieval_grade_from_score_at_upper() {
    let cfg = CragConfig::default(); // upper=0.7
    let grade = RetrievalGrade::from_score(0.7, &cfg);
    assert_eq!(grade, RetrievalGrade::Correct);
}

#[test]
fn test_retrieval_grade_from_score_at_lower() {
    let cfg = CragConfig::default(); // lower=0.3
    let grade = RetrievalGrade::from_score(0.3, &cfg);
    assert_eq!(grade, RetrievalGrade::Incorrect);
}

#[test]
fn test_retrieval_grade_from_score_between() {
    let cfg = CragConfig::default();
    let grade = RetrievalGrade::from_score(0.5, &cfg);
    assert_eq!(grade, RetrievalGrade::Ambiguous);
}

#[test]
fn test_retrieval_grade_correct_beats_boundary_when_equal_thresholds() {
    // When lower == upper, score exactly on boundary → Correct.
    let cfg = CragConfig::default()
        .with_upper_threshold(0.5)
        .with_lower_threshold(0.5);
    assert_eq!(
        RetrievalGrade::from_score(0.5, &cfg),
        RetrievalGrade::Correct
    );
}

#[test]
fn test_retrieval_grade_as_str() {
    assert_eq!(RetrievalGrade::Correct.as_str(), "correct");
    assert_eq!(RetrievalGrade::Ambiguous.as_str(), "ambiguous");
    assert_eq!(RetrievalGrade::Incorrect.as_str(), "incorrect");
}

#[test]
fn test_corrective_action_eq() {
    assert_eq!(CorrectiveAction::UseAsIs, CorrectiveAction::UseAsIs);
    assert_ne!(CorrectiveAction::UseAsIs, CorrectiveAction::Rewrite);
    assert_ne!(CorrectiveAction::Refine, CorrectiveAction::Augment);
}

#[test]
fn test_invalid_thresholds_detect() {
    // lower > upper is caught at run time inside the engine.
    let cfg = CragConfig::default()
        .with_lower_threshold(0.8)
        .with_upper_threshold(0.2);
    // We just verify the error variant can be constructed and displayed.
    let err = CorrectiveRagError::InvalidThresholds {
        lower: cfg.lower_threshold,
        upper: cfg.upper_threshold,
    };
    let s = err.to_string();
    assert!(s.contains("Invalid thresholds"));
}

// ── HeuristicRetrievalGrader ──────────────────────────────────────────────────

#[tokio::test]
async fn test_heuristic_grader_high_overlap_is_correct() {
    let grader = HeuristicRetrievalGrader::new();
    let cfg = CragConfig::default();
    // Query and doc share many tokens → high Jaccard → Correct.
    let result = make_result("a", "rust programming language safety performance", 0.9);
    let score = grader
        .grade("rust programming language safety", &result)
        .await
        .expect("grade should succeed");
    let grade = RetrievalGrade::from_score(score, &cfg);
    assert_eq!(grade, RetrievalGrade::Correct, "score={score}");
}

#[tokio::test]
async fn test_heuristic_grader_low_overlap_is_incorrect() {
    let grader = HeuristicRetrievalGrader::new();
    let cfg = CragConfig::default();
    // Completely different tokens → Jaccard ~= 0 → Incorrect.
    let result = make_result("b", "apple banana cherry tropical", 0.1);
    let score = grader
        .grade("quantum mechanics physics", &result)
        .await
        .expect("grade should succeed");
    let grade = RetrievalGrade::from_score(score, &cfg);
    assert_eq!(grade, RetrievalGrade::Incorrect, "score={score}");
}

#[tokio::test]
async fn test_heuristic_grader_mid_overlap_is_ambiguous() {
    let grader = HeuristicRetrievalGrader::new();
    let cfg = CragConfig::default()
        .with_upper_threshold(0.6)
        .with_lower_threshold(0.1);
    // Some overlap but not full → mid Jaccard.
    let result = make_result("c", "rust language systems memory", 0.5);
    let score = grader
        .grade("rust memory safety systems programming", &result)
        .await
        .expect("grade should succeed");
    // Score is between 0.1 and 0.6 for partial overlap.
    let grade = RetrievalGrade::from_score(score, &cfg);
    // Partial overlap should not be Incorrect (score > 0).
    assert_ne!(grade, RetrievalGrade::Incorrect, "score={score}");
}

#[tokio::test]
async fn test_heuristic_grader_empty_doc() {
    let grader = HeuristicRetrievalGrader::new();
    let result = make_result("d", "", 0.0);
    let score = grader
        .grade("something", &result)
        .await
        .expect("grade should succeed");
    assert!((score - 0.0).abs() < 1e-5, "empty doc → 0.0");
}

#[tokio::test]
async fn test_heuristic_grader_with_min_overlap() {
    let grader = HeuristicRetrievalGrader::new().with_min_overlap(0.2);
    assert!((grader.min_overlap - 0.2).abs() < 1e-5);
    let result = make_result("e", "hello world", 0.5);
    // Should still return a f32 result regardless of min_overlap.
    let score = grader
        .grade("hello world", &result)
        .await
        .expect("grade should succeed");
    assert!(score > 0.0);
}

#[tokio::test]
async fn test_heuristic_grader_grade_all_batch() {
    let grader = HeuristicRetrievalGrader::new();
    let cfg = CragConfig::default();
    let results = vec![
        make_result("r1", "rust programming systems language", 0.9),
        make_result("r2", "banana tropical fruit", 0.1),
    ];
    let graded_docs = grader
        .grade_all("rust systems language programming", &results, &cfg)
        .await
        .expect("grade_all should succeed");
    assert_eq!(graded_docs.len(), 2);
    assert_eq!(graded_docs[0].grade, RetrievalGrade::Correct);
    assert_eq!(graded_docs[1].grade, RetrievalGrade::Incorrect);
}

// ── MockRetrievalGrader ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_mock_grader_fixed_score() {
    let grader = MockRetrievalGrader::new(0.9);
    let result = make_result("x", "anything", 0.5);
    let score = grader.grade("query", &result).await.expect("should grade");
    assert!((score - 0.9).abs() < 1e-5);
}

#[tokio::test]
async fn test_mock_grader_with_scores_per_id() {
    let mut map = HashMap::new();
    map.insert("doc1".to_string(), 0.8_f32);
    map.insert("doc2".to_string(), 0.2_f32);
    let grader = MockRetrievalGrader::with_scores(map);

    let r1 = make_result("doc1", "content", 0.0);
    let r2 = make_result("doc2", "content", 0.0);
    let r3 = make_result("doc3", "content", 0.0); // not in map → 0.0

    let s1 = grader.grade("q", &r1).await.expect("should grade");
    let s2 = grader.grade("q", &r2).await.expect("should grade");
    let s3 = grader.grade("q", &r3).await.expect("should grade");

    assert!((s1 - 0.8).abs() < 1e-5);
    assert!((s2 - 0.2).abs() < 1e-5);
    assert!((s3 - 0.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_mock_grader_grade_all() {
    let grader = MockRetrievalGrader::new(0.75);
    let cfg = CragConfig::default();
    let results = vec![
        make_result("a", "doc a", 0.0),
        make_result("b", "doc b", 0.0),
    ];
    let graded_docs = grader
        .grade_all("query", &results, &cfg)
        .await
        .expect("grade_all ok");
    assert_eq!(graded_docs.len(), 2);
    for g in &graded_docs {
        assert!((g.relevance - 0.75).abs() < 1e-5);
        assert_eq!(g.grade, RetrievalGrade::Correct);
    }
}

// ── KnowledgeRefiner ──────────────────────────────────────────────────────────

#[test]
fn test_refiner_decompose_splits_into_sentences() {
    let refiner = KnowledgeRefiner::new();
    let cfg = CragConfig::default().with_strip_relevance_threshold(0.0); // keep all
    let doc = make_result("d", "Hello world. Goodbye world.", 0.8);
    let strips = refiner.decompose("hello", &doc, &cfg);
    assert_eq!(strips.len(), 2, "Should split into 2 sentences");
}

#[test]
fn test_refiner_decompose_keeps_above_threshold() {
    let refiner = KnowledgeRefiner::new();
    let cfg = CragConfig::default().with_strip_relevance_threshold(0.0); // keep all
    let doc = make_result("d", "rust programming language.", 0.8);
    let strips = refiner.decompose("rust programming", &doc, &cfg);
    assert!(!strips.is_empty());
    assert!(strips.iter().any(|s| s.kept));
}

#[test]
fn test_refiner_decompose_drops_below_threshold() {
    let refiner = KnowledgeRefiner::new();
    let cfg = CragConfig::default().with_strip_relevance_threshold(0.99); // nearly nothing passes
    let doc = make_result("d", "apple banana cherry tropical fruit.", 0.5);
    let strips = refiner.decompose("quantum physics", &doc, &cfg);
    assert!(
        strips.iter().all(|s| !s.kept),
        "All strips should be dropped"
    );
}

#[test]
fn test_refiner_recompose_preserves_document_id() {
    let refiner = KnowledgeRefiner::new();
    let strips = vec![
        KnowledgeStrip {
            source_id: DocumentId::from("my-id"),
            text: "sentence one".to_string(),
            relevance: 0.8,
            kept: true,
        },
        KnowledgeStrip {
            source_id: DocumentId::from("my-id"),
            text: "sentence two".to_string(),
            relevance: 0.6,
            kept: true,
        },
    ];
    let results = refiner.recompose(&strips);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.id.as_str(), "my-id");
}

#[test]
fn test_refiner_refine_across_docs() {
    let refiner = KnowledgeRefiner::new();
    let cfg = CragConfig::default().with_strip_relevance_threshold(0.0);
    let docs = vec![
        make_result("d1", "rust programming. memory safety.", 0.9),
        make_result("d2", "banana cherry tropical.", 0.4),
    ];
    let kept = refiner.refine("rust", &docs, &cfg);
    // All sentences from both docs should be kept (threshold=0).
    assert!(!kept.is_empty());
    assert!(kept.iter().all(|s| s.kept));
}

#[test]
fn test_refiner_recompose_returns_empty_when_no_kept() {
    let refiner = KnowledgeRefiner::new();
    let strips = vec![KnowledgeStrip {
        source_id: DocumentId::from("x"),
        text: "discarded".to_string(),
        relevance: 0.1,
        kept: false, // not kept
    }];
    let results = refiner.recompose(&strips);
    assert!(results.is_empty(), "No kept strips → empty recompose");
}

#[test]
fn test_refiner_recompose_joins_kept_strips() {
    let refiner = KnowledgeRefiner::new();
    let strips = vec![
        KnowledgeStrip {
            source_id: DocumentId::from("id1"),
            text: "first sentence".to_string(),
            relevance: 0.7,
            kept: true,
        },
        KnowledgeStrip {
            source_id: DocumentId::from("id1"),
            text: "second sentence".to_string(),
            relevance: 0.8,
            kept: true,
        },
    ];
    let results = refiner.recompose(&strips);
    assert_eq!(results.len(), 1);
    assert!(results[0].document.content.contains("first sentence"));
    assert!(results[0].document.content.contains("second sentence"));
    // Average score = (0.7 + 0.8) / 2 = 0.75
    assert!((results[0].score - 0.75).abs() < 1e-4);
}

#[test]
fn test_lexical_embedding_determinism_via_rewriter() {
    // Verify that the QueryRefiner produces stable output across calls.
    let refiner = QueryRefiner::new();
    let strips = vec![
        KnowledgeStrip {
            source_id: DocumentId::new(),
            text: "rust programming language".to_string(),
            relevance: 0.9,
            kept: true,
        },
        KnowledgeStrip {
            source_id: DocumentId::new(),
            text: "rust programming safety".to_string(),
            relevance: 0.8,
            kept: true,
        },
    ];
    let r1 = refiner.rewrite("what is rust", &strips);
    let r2 = refiner.rewrite("what is rust", &strips);
    assert_eq!(r1, r2, "Rewrite should be deterministic");
}

// ── QueryRefiner ──────────────────────────────────────────────────────────────

#[test]
fn test_query_refiner_appends_salient_terms() {
    let refiner = QueryRefiner::new();
    // Two strips share "programming" and "rust" but query only has "rust".
    let strips = vec![
        KnowledgeStrip {
            source_id: DocumentId::new(),
            text: "rust programming language".to_string(),
            relevance: 0.9,
            kept: true,
        },
        KnowledgeStrip {
            source_id: DocumentId::new(),
            text: "rust programming systems".to_string(),
            relevance: 0.8,
            kept: true,
        },
    ];
    let rewritten = refiner.rewrite("rust", &strips);
    // "programming" appears in 2 strips and is not in the original query.
    assert!(
        rewritten.contains("programming"),
        "Expected 'programming' in rewrite, got: {rewritten}"
    );
}

#[test]
fn test_query_refiner_never_empty_guard() {
    let refiner = QueryRefiner::new();
    // No strips at all — should return original query unchanged.
    let result = refiner.rewrite("original query", &[]);
    assert_eq!(result, "original query");
}

#[test]
fn test_query_refiner_empty_strips_returns_original() {
    let refiner = QueryRefiner::new();
    let strips: Vec<KnowledgeStrip> = Vec::new();
    let result = refiner.rewrite("my question", &strips);
    assert_eq!(result, "my question");
}

#[test]
fn test_query_refiner_different_query_different_result() {
    let refiner = QueryRefiner::new();
    let strips = vec![
        KnowledgeStrip {
            source_id: DocumentId::new(),
            text: "deep learning neural networks".to_string(),
            relevance: 0.9,
            kept: true,
        },
        KnowledgeStrip {
            source_id: DocumentId::new(),
            text: "deep learning convolutional networks".to_string(),
            relevance: 0.8,
            kept: true,
        },
    ];
    let r1 = refiner.rewrite("deep learning", &strips);
    let r2 = refiner.rewrite("what is machine learning", &strips);
    // The base query differs, so rewrites should differ.
    assert_ne!(r1, r2);
}

// ── CorrectiveRagEngine::run ──────────────────────────────────────────────────

#[tokio::test]
async fn test_engine_all_correct_uses_as_is() {
    let grader = MockRetrievalGrader::new(0.9); // above upper=0.7 → Correct
    let config = CragConfig::default();
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![
        make_result("a", "rust programming language", 0.9),
        make_result("b", "rust memory safety", 0.85),
    ];
    let echo = MockEcho::new(results);
    let output = engine.run("rust", &echo).await.expect("run ok");

    assert!(
        output.actions_taken.contains(&CorrectiveAction::UseAsIs),
        "Expected UseAsIs, got: {:?}",
        output.actions_taken
    );
    assert!(!output.refined_results.is_empty());
}

#[tokio::test]
async fn test_engine_all_incorrect_triggers_rewrite() {
    let grader = MockRetrievalGrader::new(0.1); // below lower=0.3 → Incorrect
    let config = CragConfig::default().with_max_corrections(1);
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![make_result("a", "unrelated content here", 0.1)];
    let echo = MockEcho::new(results);
    let output = engine.run("rust programming", &echo).await.expect("run ok");

    assert!(
        output.actions_taken.contains(&CorrectiveAction::Rewrite),
        "Expected Rewrite action, got: {:?}",
        output.actions_taken
    );
    assert!(output.correction_rounds >= 1);
}

#[tokio::test]
async fn test_engine_ambiguous_triggers_refine() {
    let grader = MockRetrievalGrader::new(0.5); // between 0.3 and 0.7 → Ambiguous
    let config = CragConfig::default();
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![make_result(
        "x",
        "partially relevant content about rust",
        0.5,
    )];
    let echo = MockEcho::new(results);
    let output = engine.run("rust safety", &echo).await.expect("run ok");

    assert!(
        output.actions_taken.iter().any(|a| matches!(
            a,
            CorrectiveAction::Refine | CorrectiveAction::UseAsIs | CorrectiveAction::Augment
        )),
        "Expected a non-Rewrite action, got: {:?}",
        output.actions_taken
    );
}

#[tokio::test]
async fn test_engine_max_corrections_cap() {
    let grader = MockRetrievalGrader::new(0.0); // always Incorrect
    let config = CragConfig::default().with_max_corrections(2);
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![make_result("z", "irrelevant doc", 0.0)];
    let echo = MockEcho::new(results);
    let output = engine.run("query", &echo).await.expect("run ok");

    assert!(
        output.correction_rounds <= 2,
        "Should not exceed max_corrections=2, got {}",
        output.correction_rounds
    );
}

#[tokio::test]
async fn test_engine_zero_results_guard() {
    let grader = MockRetrievalGrader::new(0.9);
    let config = CragConfig::default().with_max_corrections(1);
    let engine = CorrectiveRagEngine::new(grader, config);

    let echo = MockEcho::new(vec![]); // always returns empty
    let output = engine.run("any query", &echo).await.expect("run ok");

    // No results → empty refined_results and we exited gracefully.
    assert!(output.refined_results.is_empty());
}

#[tokio::test]
async fn test_engine_empty_query_error() {
    let grader = MockRetrievalGrader::new(0.9);
    let engine = CorrectiveRagEngine::new(grader, CragConfig::default());
    let echo = MockEcho::new(vec![]);

    let err = engine.run("   ", &echo).await.expect_err("should fail");
    assert!(
        matches!(err, CorrectiveRagError::EmptyQuery),
        "Expected EmptyQuery, got: {err:?}"
    );
}

#[tokio::test]
async fn test_engine_invalid_thresholds_error() {
    let grader = MockRetrievalGrader::new(0.9);
    let config = CragConfig::default()
        .with_lower_threshold(0.9)
        .with_upper_threshold(0.1); // lower > upper
    let engine = CorrectiveRagEngine::new(grader, config);
    let echo = MockEcho::new(vec![]);

    let err = engine
        .run("query", &echo)
        .await
        .expect_err("should fail with invalid thresholds");
    assert!(
        matches!(err, CorrectiveRagError::InvalidThresholds { .. }),
        "Expected InvalidThresholds, got: {err:?}"
    );
}

#[tokio::test]
async fn test_engine_recomposition_uses_mmr_not_empty_map() {
    // With real lexical embeddings the MMR reranker should produce a ranked output.
    let grader = MockRetrievalGrader::new(0.9);
    let config = CragConfig::default().with_mmr_lambda(0.5).with_top_k(3);
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![
        make_result("a", "rust programming systems memory safety", 0.9),
        make_result("b", "python scripting dynamic typing", 0.8),
        make_result("c", "rust cargo package manager", 0.7),
    ];
    let echo = MockEcho::new(results);
    let output = engine.run("rust programming", &echo).await.expect("run ok");

    // Should have results and ranks should be re-assigned from 0.
    assert!(!output.refined_results.is_empty());
    assert_eq!(
        output.refined_results[0].rank, 0,
        "First result should have rank 0"
    );
}

#[tokio::test]
async fn test_engine_knowledge_strip_on() {
    let grader = MockRetrievalGrader::new(0.9);
    let config = CragConfig::default()
        .with_knowledge_strip(true)
        .with_strip_relevance_threshold(0.0); // keep all strips
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![make_result(
        "d",
        "rust programming language. memory safety systems.",
        0.9,
    )];
    let echo = MockEcho::new(results);
    let output = engine.run("rust", &echo).await.expect("run ok");

    // Knowledge strip is enabled so strips should be populated.
    assert!(
        !output.knowledge_strips.is_empty(),
        "Knowledge strips should be populated"
    );
}

#[tokio::test]
async fn test_engine_knowledge_strip_off() {
    let grader = MockRetrievalGrader::new(0.9);
    let config = CragConfig::default().with_knowledge_strip(false);
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![make_result("e", "some content here", 0.9)];
    let echo = MockEcho::new(results);
    let output = engine.run("content", &echo).await.expect("run ok");

    // No strips when disabled.
    assert!(
        output.knowledge_strips.is_empty(),
        "No strips when disabled"
    );
}

#[tokio::test]
async fn test_engine_final_query_set_correctly() {
    let grader = MockRetrievalGrader::new(0.9);
    let engine = CorrectiveRagEngine::new(grader, CragConfig::default());
    let results = vec![make_result("f", "some doc", 0.9)];
    let echo = MockEcho::new(results);
    let output = engine.run("my query", &echo).await.expect("run ok");

    assert!(!output.final_query.is_empty());
    // When no rewrite happens, query should be the original.
    assert!(output.final_query.starts_with("my query"));
}

#[tokio::test]
async fn test_engine_actions_taken_recorded() {
    let grader = MockRetrievalGrader::new(0.9);
    let engine = CorrectiveRagEngine::new(grader, CragConfig::default());
    let results = vec![make_result("g", "relevant doc", 0.9)];
    let echo = MockEcho::new(results);
    let output = engine.run("query", &echo).await.expect("run ok");

    assert!(
        !output.actions_taken.is_empty(),
        "actions_taken should be recorded"
    );
}

#[tokio::test]
async fn test_engine_correction_rounds_correct() {
    let grader = MockRetrievalGrader::new(0.9); // always Correct → 0 rounds
    let engine = CorrectiveRagEngine::new(grader, CragConfig::default());
    let results = vec![make_result("h", "correct doc", 0.9)];
    let echo = MockEcho::new(results);
    let output = engine.run("query", &echo).await.expect("run ok");

    // No rewrite needed → 0 rounds.
    assert_eq!(output.correction_rounds, 0);
}

#[tokio::test]
async fn test_engine_refined_results_non_empty_when_input_non_empty() {
    let grader = MockRetrievalGrader::new(0.9);
    let engine = CorrectiveRagEngine::new(grader, CragConfig::default());
    let results = vec![
        make_result("i1", "first document content", 0.9),
        make_result("i2", "second document content", 0.85),
    ];
    let echo = MockEcho::new(results);
    let output = engine.run("document", &echo).await.expect("run ok");

    assert!(!output.refined_results.is_empty());
}

#[tokio::test]
async fn test_engine_grades_populated() {
    let grader = MockRetrievalGrader::new(0.8);
    let engine = CorrectiveRagEngine::new(grader, CragConfig::default());
    let results = vec![make_result("j", "content here", 0.8)];
    let echo = MockEcho::new(results);
    let output = engine.run("content", &echo).await.expect("run ok");

    assert!(
        !output.grades.is_empty(),
        "grades should be populated after a successful run"
    );
}

#[tokio::test]
async fn test_engine_discard_like_behavior_all_filtered() {
    // Use a very high strip threshold so all strips get discarded → fallback to top-1.
    let grader = MockRetrievalGrader::new(0.9);
    let config = CragConfig::default()
        .with_knowledge_strip(true)
        .with_strip_relevance_threshold(1.0); // nothing passes
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![make_result("k", "some content that does not match", 0.9)];
    let echo = MockEcho::new(results);
    let output = engine
        .run("entirely different topic", &echo)
        .await
        .expect("run ok");

    // Engine should not panic and should return something via fallback.
    assert!(
        !output.refined_results.is_empty(),
        "Fallback should produce results"
    );
}

#[tokio::test]
async fn test_engine_augment_keeps_correct() {
    // Mix: "correct_doc" gets 0.9 (Correct), "wrong_doc" gets 0.1 (Incorrect).
    let mut scores = HashMap::new();
    scores.insert("correct_doc".to_string(), 0.9_f32);
    scores.insert("wrong_doc".to_string(), 0.1_f32);
    let grader = MockRetrievalGrader::with_scores(scores);
    let config = CragConfig::default().with_knowledge_strip(false);
    let engine = CorrectiveRagEngine::new(grader, config);

    let results = vec![
        make_result("correct_doc", "rust programming safety", 0.9),
        make_result("wrong_doc", "banana tropical unrelated", 0.1),
    ];
    let echo = MockEcho::new(results);
    let output = engine.run("rust safety", &echo).await.expect("run ok");

    // Augment action should be recorded.
    assert!(
        output.actions_taken.contains(&CorrectiveAction::Augment),
        "Expected Augment action, got: {:?}",
        output.actions_taken
    );
    // The correct doc should appear in refined results.
    let ids: Vec<&str> = output
        .refined_results
        .iter()
        .map(|r| r.document.id.as_str())
        .collect();
    assert!(
        ids.contains(&"correct_doc"),
        "correct_doc should be in refined results; got: {ids:?}"
    );
}
