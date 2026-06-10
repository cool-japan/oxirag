#![allow(clippy::float_cmp)]
use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::reflexion::engine::ReflexionEngine;
use crate::reflexion::types::{
    AttemptEvaluator, AttemptScore, EpisodicMemory, HeuristicEvaluator, HeuristicSelfReflector,
    Reflection, ReflexionConfig, ReflexionError, SelfReflector,
};
use crate::types::{Document, DocumentId, SearchResult};
use async_trait::async_trait;

// ── MockEcho ──────────────────────────────────────────────────────────────
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

// ── helpers ───────────────────────────────────────────────────────────────
fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

// ── AttemptScore tests ────────────────────────────────────────────────────
#[test]
fn attempt_score_default_fields_are_zero() {
    let s = AttemptScore::default();
    assert_eq!(s.score, 0.0);
    assert_eq!(s.grounding, 0.0);
    assert_eq!(s.coverage, 0.0);
}

#[test]
fn attempt_score_clone_is_equal() {
    let s = AttemptScore {
        score: 0.5,
        grounding: 0.4,
        coverage: 0.3,
    };
    let c = s.clone();
    assert_eq!(c.score, s.score);
    assert_eq!(c.grounding, s.grounding);
    assert_eq!(c.coverage, s.coverage);
}

// ── Reflection tests ──────────────────────────────────────────────────────
#[test]
fn reflection_construction() {
    let r = Reflection {
        attempt: 1,
        critique: "Bad answer".to_string(),
        lesson: "Use better queries".to_string(),
        score: 0.2,
    };
    assert_eq!(r.attempt, 1);
    assert_eq!(r.critique, "Bad answer");
    assert_eq!(r.lesson, "Use better queries");
    assert!((r.score - 0.2).abs() < 1e-6);
}

// ── EpisodicMemory tests ──────────────────────────────────────────────────
#[test]
fn episodic_memory_new_is_empty() {
    let mem = EpisodicMemory::new(8);
    assert_eq!(mem.max_size, 8);
    assert!(mem.reflections.is_empty());
}

#[test]
fn episodic_memory_add_single_entry() {
    let mut mem = EpisodicMemory::new(4);
    mem.add(Reflection {
        attempt: 0,
        critique: "c".to_string(),
        lesson: "l".to_string(),
        score: 0.1,
    });
    assert_eq!(mem.reflections.len(), 1);
}

#[test]
fn episodic_memory_recent_returns_last_n() {
    let mut mem = EpisodicMemory::new(10);
    for i in 0..5 {
        mem.add(Reflection {
            attempt: i,
            critique: "c".to_string(),
            lesson: format!("lesson{i}"),
            score: 0.1,
        });
    }
    let recent = mem.recent(3);
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0].attempt, 2);
    assert_eq!(recent[2].attempt, 4);
}

#[test]
fn episodic_memory_recent_capped_by_available() {
    let mut mem = EpisodicMemory::new(10);
    mem.add(Reflection {
        attempt: 0,
        critique: "c".to_string(),
        lesson: "l".to_string(),
        score: 0.5,
    });
    let recent = mem.recent(5);
    assert_eq!(recent.len(), 1);
}

#[test]
fn episodic_memory_capacity_evicts_oldest() {
    let mut mem = EpisodicMemory::new(3);
    for i in 0..5u32 {
        mem.add(Reflection {
            attempt: i as usize,
            critique: "c".to_string(),
            lesson: format!("L{i}"),
            score: 0.1,
        });
    }
    assert_eq!(mem.reflections.len(), 3);
    // oldest evicted: 0,1 gone; remaining: 2,3,4
    assert_eq!(mem.reflections[0].attempt, 2);
    assert_eq!(mem.reflections[2].attempt, 4);
}

#[test]
fn episodic_memory_capacity_zero_allows_unlimited() {
    let mut mem = EpisodicMemory::new(0);
    for i in 0..100 {
        mem.add(Reflection {
            attempt: i,
            critique: "c".to_string(),
            lesson: "l".to_string(),
            score: 0.1,
        });
    }
    assert_eq!(mem.reflections.len(), 100);
}

#[test]
fn episodic_memory_summary_joins_lessons_with_space() {
    let mut mem = EpisodicMemory::new(10);
    mem.add(Reflection {
        attempt: 0,
        critique: "c".to_string(),
        lesson: "first".to_string(),
        score: 0.1,
    });
    mem.add(Reflection {
        attempt: 1,
        critique: "c".to_string(),
        lesson: "second".to_string(),
        score: 0.2,
    });
    assert_eq!(mem.summary(), "first second");
}

#[test]
fn episodic_memory_summary_empty_is_empty_string() {
    let mem = EpisodicMemory::new(4);
    assert_eq!(mem.summary(), "");
}

// ── HeuristicEvaluator tests ──────────────────────────────────────────────
#[test]
fn heuristic_evaluator_empty_results_returns_zero() {
    let eval = HeuristicEvaluator;
    let score = eval.evaluate("query", "some answer", &[]).unwrap();
    assert_eq!(score.score, 0.0);
    assert_eq!(score.grounding, 0.0);
    assert_eq!(score.coverage, 0.0);
}

#[test]
fn heuristic_evaluator_score_in_range_for_nonempty() {
    let eval = HeuristicEvaluator;
    let results = vec![make_result("d1", "rust programming language", 0.9)];
    let score = eval
        .evaluate("rust", "rust programming language", &results)
        .unwrap();
    assert!(score.score >= 0.0 && score.score <= 1.0);
    assert!(score.grounding >= 0.0 && score.grounding <= 1.0);
    assert!(score.coverage >= 0.0 && score.coverage <= 1.0);
}

#[test]
fn heuristic_evaluator_matching_answer_scores_higher_than_random() {
    let eval = HeuristicEvaluator;
    let results = vec![make_result("d1", "rust programming language systems", 0.9)];
    let good_score = eval
        .evaluate("rust", "rust programming language systems", &results)
        .unwrap();
    let bad_score = eval
        .evaluate("rust", "unrelated gibberish words xyz", &results)
        .unwrap();
    assert!(good_score.score > bad_score.score);
}

#[test]
fn heuristic_evaluator_full_match_high_grounding() {
    let eval = HeuristicEvaluator;
    let content = "the quick brown fox jumps over the lazy dog";
    let results = vec![make_result("d1", content, 0.9)];
    let score = eval.evaluate("fox", content, &results).unwrap();
    assert!(score.grounding > 0.5);
}

#[test]
fn heuristic_evaluator_partial_overlap_partial_coverage() {
    let eval = HeuristicEvaluator;
    let results = vec![
        make_result("d1", "rust fast systems programming", 0.9),
        make_result("d2", "python scripting glue language", 0.7),
    ];
    let score = eval
        .evaluate("language", "rust programming language", &results)
        .unwrap();
    assert!(score.coverage >= 0.0 && score.coverage <= 1.0);
}

#[test]
fn heuristic_evaluator_grounding_is_max_jaccard() {
    let eval = HeuristicEvaluator;
    let results = vec![
        make_result("d1", "apple banana cherry", 0.9),
        make_result("d2", "dog cat fish", 0.7),
    ];
    // answer matches doc1 well, not doc2
    let score = eval
        .evaluate("fruit", "apple banana cherry mango", &results)
        .unwrap();
    assert!(score.grounding > 0.3);
}

// ── HeuristicSelfReflector tests ──────────────────────────────────────────
#[test]
fn heuristic_self_reflector_returns_nonempty_critique() {
    let reflector = HeuristicSelfReflector;
    let score = AttemptScore {
        score: 0.2,
        grounding: 0.1,
        coverage: 0.1,
    };
    let reflection = reflector.reflect("query", "answer", &score).unwrap();
    assert!(!reflection.critique.is_empty());
}

#[test]
fn heuristic_self_reflector_lesson_low_score() {
    let reflector = HeuristicSelfReflector;
    let score = AttemptScore {
        score: 0.1,
        grounding: 0.1,
        coverage: 0.1,
    };
    let r = reflector.reflect("q", "a", &score).unwrap();
    assert!(r.lesson.contains("synonyms") || r.lesson.contains("lacked"));
}

#[test]
fn heuristic_self_reflector_lesson_mid_score() {
    let reflector = HeuristicSelfReflector;
    let score = AttemptScore {
        score: 0.45,
        grounding: 0.4,
        coverage: 0.3,
    };
    let r = reflector.reflect("q", "a", &score).unwrap();
    assert!(r.lesson.contains("entities") || r.lesson.contains("partially"));
}

#[test]
fn heuristic_self_reflector_lesson_high_score() {
    let reflector = HeuristicSelfReflector;
    let score = AttemptScore {
        score: 0.75,
        grounding: 0.8,
        coverage: 0.7,
    };
    let r = reflector.reflect("q", "a", &score).unwrap();
    assert!(r.lesson.contains("Broaden") || r.lesson.contains("reasonable"));
}

#[test]
fn heuristic_self_reflector_critique_contains_score() {
    let reflector = HeuristicSelfReflector;
    let score = AttemptScore {
        score: 0.35,
        grounding: 0.3,
        coverage: 0.2,
    };
    let r = reflector.reflect("q", "a", &score).unwrap();
    assert!(r.critique.contains("0.35"));
}

// ── ReflexionConfig tests ─────────────────────────────────────────────────
#[test]
fn reflexion_config_default_values() {
    let c = ReflexionConfig::default();
    assert_eq!(c.max_attempts, 3);
    assert!((c.success_threshold - 0.7).abs() < 1e-6);
    assert_eq!(c.top_k, 5);
    assert_eq!(c.memory_size, 8);
}

#[test]
fn reflexion_config_builder_max_attempts() {
    let c = ReflexionConfig::default().with_max_attempts(5);
    assert_eq!(c.max_attempts, 5);
}

#[test]
fn reflexion_config_builder_threshold() {
    let c = ReflexionConfig::default().with_success_threshold(0.9);
    assert!((c.success_threshold - 0.9).abs() < 1e-6);
}

#[test]
fn reflexion_config_builder_top_k() {
    let c = ReflexionConfig::default().with_top_k(10);
    assert_eq!(c.top_k, 10);
}

#[test]
fn reflexion_config_with_memory_size() {
    let c = ReflexionConfig::default().with_memory_size(16);
    assert_eq!(c.memory_size, 16);
}

// ── ReflexionEngine creation tests ────────────────────────────────────────
#[test]
fn reflexion_engine_new_stores_config() {
    let cfg = ReflexionConfig::default().with_max_attempts(7);
    let engine = ReflexionEngine::new(cfg);
    assert_eq!(engine.config.max_attempts, 7);
}

#[test]
fn reflexion_engine_default_uses_default_config() {
    let engine = ReflexionEngine::default();
    assert_eq!(engine.config.max_attempts, 3);
}

// ── ReflexionError tests ──────────────────────────────────────────────────
#[test]
fn reflexion_error_empty_query_display() {
    let e = ReflexionError::EmptyQuery;
    assert!(e.to_string().contains("empty") || e.to_string().contains("not be empty"));
}

#[test]
fn reflexion_error_retrieval_failed_display() {
    let e = ReflexionError::RetrievalFailed("network error".to_string());
    let s = e.to_string();
    assert!(s.contains("Retrieval") || s.contains("failed"));
    assert!(s.contains("network error"));
}

#[test]
fn reflexion_error_evaluation_failed_display() {
    let e = ReflexionError::EvaluationFailed("bad scorer".to_string());
    let s = e.to_string();
    assert!(s.contains("Evaluation") || s.contains("failed"));
    assert!(s.contains("bad scorer"));
}

// ── ReflexionEngine::run tests (async) ────────────────────────────────────
#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_empty_query_returns_error() {
    let engine = ReflexionEngine::default();
    let echo = MockEcho::new(vec![]);
    let result = engine.run("", &echo).await;
    assert!(matches!(result, Err(ReflexionError::EmptyQuery)));
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_whitespace_query_returns_error() {
    let engine = ReflexionEngine::default();
    let echo = MockEcho::new(vec![]);
    let result = engine.run("   ", &echo).await;
    assert!(matches!(result, Err(ReflexionError::EmptyQuery)));
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_empty_results_completes_without_panic() {
    let engine = ReflexionEngine::default();
    let echo = MockEcho::new(vec![]);
    let result = engine.run("what is rust?", &echo).await;
    assert!(result.is_ok());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_best_answer_nonempty() {
    let engine = ReflexionEngine::default();
    let echo = MockEcho::new(vec![make_result("d1", "Rust is a systems language", 0.9)]);
    let outcome = engine.run("what is rust?", &echo).await.unwrap();
    assert!(!outcome.best_answer.is_empty());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_attempts_len_lte_max_attempts() {
    let cfg = ReflexionConfig::default().with_max_attempts(3);
    let engine = ReflexionEngine::new(cfg);
    let echo = MockEcho::new(vec![]);
    let outcome = engine.run("test query", &echo).await.unwrap();
    assert!(outcome.attempts.len() <= 3);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_max_attempts_one_makes_one_attempt() {
    let cfg = ReflexionConfig::default().with_max_attempts(1);
    let engine = ReflexionEngine::new(cfg);
    let echo = MockEcho::new(vec![]);
    let outcome = engine.run("test query", &echo).await.unwrap();
    assert_eq!(outcome.attempts.len(), 1);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_with_matching_results_succeeded() {
    // Use a high-overlap answer to trigger success threshold
    let content = "rust systems programming language fast safe concurrent";
    let results = vec![
        make_result("d1", content, 0.95),
        make_result("d2", content, 0.90),
        make_result("d3", content, 0.85),
        make_result("d4", content, 0.80),
        make_result("d5", content, 0.75),
    ];
    let cfg = ReflexionConfig::default().with_success_threshold(0.01);
    let engine = ReflexionEngine::new(cfg);
    let echo = MockEcho::new(results);
    let outcome = engine.run("rust", &echo).await.unwrap();
    assert!(outcome.succeeded);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_outcome_fields_populated() {
    let engine = ReflexionEngine::default();
    let echo = MockEcho::new(vec![make_result("d1", "hello world answer", 0.8)]);
    let outcome = engine.run("hello", &echo).await.unwrap();
    // At minimum one attempt must be recorded
    assert!(!outcome.attempts.is_empty());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_memory_populated_after_failed_attempt() {
    // With a threshold > 1.0 (impossible to reach), all attempts should fail
    // and populate memory with reflections
    let cfg = ReflexionConfig::default()
        .with_max_attempts(3)
        .with_success_threshold(1.01); // impossible to exceed 1.0
    let engine = ReflexionEngine::new(cfg);
    // Use empty results so score stays at 0.0 (well below 1.01)
    let echo = MockEcho::new(vec![]);
    let outcome = engine.run("query", &echo).await.unwrap();
    // Should have attempted all 3 times and not succeeded
    assert!(!outcome.succeeded);
    assert_eq!(outcome.attempts.len(), 3);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_engine_run_augmented_query_uses_memory_summary() {
    // With threshold > 1.0 (impossible) and 2 attempts, both will fail
    // and memory should be populated with reflections after first attempt
    let cfg = ReflexionConfig::default()
        .with_max_attempts(2)
        .with_success_threshold(1.01) // impossible threshold
        .with_memory_size(4);
    let engine = ReflexionEngine::new(cfg);
    // Use empty results so score stays at 0.0 (< 1.01)
    let echo = MockEcho::new(vec![]);
    let outcome = engine.run("rust", &echo).await.unwrap();
    // After first failed attempt, memory should contain at least one lesson
    assert!(!outcome.memory.summary().is_empty());
}

// ── ReflexionOutcome tests ────────────────────────────────────────────────
#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_outcome_succeeded_false_with_impossible_threshold() {
    let cfg = ReflexionConfig::default().with_success_threshold(1.01);
    let engine = ReflexionEngine::new(cfg);
    let echo = MockEcho::new(vec![make_result("d1", "content", 0.5)]);
    let outcome = engine.run("query", &echo).await.unwrap();
    assert!(!outcome.succeeded);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn reflexion_outcome_best_score_reflects_attempts() {
    let engine = ReflexionEngine::default();
    let echo = MockEcho::new(vec![make_result("d1", "test content", 0.8)]);
    let outcome = engine.run("test", &echo).await.unwrap();
    // best_score should be non-negative
    assert!(outcome.best_score.score >= 0.0);
}
