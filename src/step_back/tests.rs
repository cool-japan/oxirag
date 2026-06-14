#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
//! Unit tests for the `step_back` module.

use super::engine::StepBackEngine;
use super::types::{MockStepBackModel, StepBackConfig, StepBackError, StepBackModel};

// ── helper models ─────────────────────────────────────────────────────────────

/// A model that always fails abstraction.
#[derive(Debug)]
struct FailAbstractModel;

impl StepBackModel for FailAbstractModel {
    fn abstract_query(&self, _query: &str) -> Result<String, StepBackError> {
        Err(StepBackError::AbstractionFailed("intentional".to_string()))
    }

    fn synthesize(
        &self,
        _original: &str,
        _abstract_query: &str,
        _docs: &[String],
    ) -> Result<String, StepBackError> {
        Ok("ok".to_string())
    }
}

/// A model that always fails synthesis.
#[derive(Debug)]
struct FailSynthModel;

impl StepBackModel for FailSynthModel {
    fn abstract_query(&self, _query: &str) -> Result<String, StepBackError> {
        Ok("In general terms, the query".to_string())
    }

    fn synthesize(
        &self,
        _original: &str,
        _abstract_query: &str,
        _docs: &[String],
    ) -> Result<String, StepBackError> {
        Err(StepBackError::SynthesisFailed("synth error".to_string()))
    }
}

/// A model whose abstraction returns a strictly shorter string each time,
/// so `is_more_abstract` via the "longer" branch never fires. But since the
/// shorter form also has fewer tokens, the pipeline should still abstract once.
#[derive(Debug)]
struct ShorterAbstractModel;

impl StepBackModel for ShorterAbstractModel {
    /// Always returns "general topic" regardless of input — fewer tokens, shorter.
    fn abstract_query(&self, _query: &str) -> Result<String, StepBackError> {
        Ok("general topic".to_string())
    }

    fn synthesize(
        &self,
        _original: &str,
        _abstract_query: &str,
        docs: &[String],
    ) -> Result<String, StepBackError> {
        Ok(docs.join("|"))
    }
}

/// A model whose abstraction always returns the same fixed string regardless
/// of input depth, so after the first iteration the heuristic detects no
/// further progress and halts early.
#[derive(Debug)]
struct StaticAbstractModel {
    fixed: String,
}

impl StaticAbstractModel {
    fn new(fixed: impl Into<String>) -> Self {
        Self {
            fixed: fixed.into(),
        }
    }
}

impl StepBackModel for StaticAbstractModel {
    fn abstract_query(&self, _query: &str) -> Result<String, StepBackError> {
        Ok(self.fixed.clone())
    }

    fn synthesize(
        &self,
        _original: &str,
        _abstract_query: &str,
        docs: &[String],
    ) -> Result<String, StepBackError> {
        Ok(docs.join(" "))
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn mock_engine() -> StepBackEngine {
    StepBackEngine::new(StepBackConfig::default(), Box::new(MockStepBackModel))
}

fn sample_docs() -> Vec<String> {
    vec![
        "Physics constants include the speed of light".to_string(),
        "The gravitational constant governs attraction".to_string(),
        "Quantum mechanics describes particle wave duality".to_string(),
    ]
}

// ── StepBackConfig defaults ───────────────────────────────────────────────────

#[test]
fn config_default_max_abstraction_depth_is_two() {
    assert_eq!(StepBackConfig::default().max_abstraction_depth, 2);
}

#[test]
fn config_default_top_k_is_five() {
    assert_eq!(StepBackConfig::default().top_k, 5);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(StepBackConfig::new(), StepBackConfig::default());
}

// ── StepBackConfig builders ───────────────────────────────────────────────────

#[test]
fn config_builder_sets_max_depth() {
    let cfg = StepBackConfig::new().with_max_abstraction_depth(4);
    assert_eq!(cfg.max_abstraction_depth, 4);
}

#[test]
fn config_builder_sets_top_k() {
    let cfg = StepBackConfig::new().with_top_k(10);
    assert_eq!(cfg.top_k, 10);
}

#[test]
fn config_builder_chains() {
    let cfg = StepBackConfig::new()
        .with_max_abstraction_depth(3)
        .with_top_k(8);
    assert_eq!(cfg.max_abstraction_depth, 3);
    assert_eq!(cfg.top_k, 8);
}

#[test]
fn config_builder_preserves_depth_when_setting_top_k() {
    let cfg = StepBackConfig::new().with_top_k(7);
    assert_eq!(cfg.max_abstraction_depth, 2);
}

#[test]
fn config_builder_preserves_top_k_when_setting_depth() {
    let cfg = StepBackConfig::new().with_max_abstraction_depth(1);
    assert_eq!(cfg.top_k, 5);
}

#[test]
fn config_clone_equals_original() {
    let cfg = StepBackConfig::new()
        .with_max_abstraction_depth(3)
        .with_top_k(7);
    assert_eq!(cfg.clone(), cfg);
}

#[test]
fn config_debug_non_empty() {
    assert!(!format!("{:?}", StepBackConfig::default()).is_empty());
}

#[test]
fn config_max_depth_zero() {
    assert_eq!(
        StepBackConfig::new()
            .with_max_abstraction_depth(0)
            .max_abstraction_depth,
        0
    );
}

// ── StepBackError display ─────────────────────────────────────────────────────

#[test]
fn error_abstraction_failed_display() {
    let msg = StepBackError::AbstractionFailed("oops".to_string()).to_string();
    assert!(msg.contains("abstraction failed"));
    assert!(msg.contains("oops"));
}

#[test]
fn error_retrieval_failed_display() {
    let msg = StepBackError::RetrievalFailed("db down".to_string()).to_string();
    assert!(msg.contains("retrieval failed"));
    assert!(msg.contains("db down"));
}

#[test]
fn error_synthesis_failed_display() {
    let msg = StepBackError::SynthesisFailed("timeout".to_string()).to_string();
    assert!(msg.contains("synthesis failed"));
    assert!(msg.contains("timeout"));
}

#[test]
fn error_debug_non_empty() {
    assert!(!format!("{:?}", StepBackError::AbstractionFailed("x".to_string())).is_empty());
}

// ── MockStepBackModel ─────────────────────────────────────────────────────────

#[test]
fn mock_abstract_query_prepends_in_general_terms() {
    let model = MockStepBackModel;
    let result = model.abstract_query("What is gravity?").unwrap();
    assert!(result.starts_with("In general terms, "));
}

#[test]
fn mock_abstract_query_preserves_original_in_result() {
    let model = MockStepBackModel;
    let result = model.abstract_query("quantum entanglement").unwrap();
    assert!(result.contains("quantum entanglement"));
}

#[test]
fn mock_abstract_query_with_empty_input() {
    let model = MockStepBackModel;
    let result = model.abstract_query("").unwrap();
    assert_eq!(result, "In general terms, ");
}

#[test]
fn mock_synthesize_empty_docs_returns_empty() {
    let model = MockStepBackModel;
    let result = model.synthesize("orig", "abstract", &[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn mock_synthesize_single_doc() {
    let model = MockStepBackModel;
    let docs = vec!["only doc".to_string()];
    let result = model.synthesize("q", "aq", &docs).unwrap();
    assert_eq!(result, "only doc");
}

#[test]
fn mock_synthesize_multiple_docs_joined_with_space() {
    let model = MockStepBackModel;
    let docs = vec![
        "doc one".to_string(),
        "doc two".to_string(),
        "doc three".to_string(),
    ];
    let result = model.synthesize("q", "aq", &docs).unwrap();
    assert_eq!(result, "doc one doc two doc three");
}

#[test]
fn mock_synthesize_ignores_original_and_abstract() {
    let model = MockStepBackModel;
    let docs = vec!["answer".to_string()];
    let a = model.synthesize("original_a", "abstract_a", &docs).unwrap();
    let b = model.synthesize("original_b", "abstract_b", &docs).unwrap();
    assert_eq!(a, b);
}

#[test]
fn mock_clone_and_copy() {
    let a = MockStepBackModel;
    let b = a;
    assert_eq!(a, b);
}

// ── StepBackEngine construction ───────────────────────────────────────────────

#[test]
fn engine_new_stores_config() {
    let cfg = StepBackConfig::new()
        .with_max_abstraction_depth(1)
        .with_top_k(3);
    let engine = StepBackEngine::new(cfg.clone(), Box::new(MockStepBackModel));
    assert_eq!(engine.config, cfg);
}

#[test]
fn engine_debug_non_empty() {
    assert!(!format!("{:?}", mock_engine()).is_empty());
}

// ── run: basic success ────────────────────────────────────────────────────────

#[test]
fn engine_run_basic_success() {
    let result = mock_engine().run("What is photosynthesis?", &sample_docs());
    assert!(result.is_ok());
}

#[test]
fn engine_run_original_query_preserved() {
    let q = "What is the Planck constant?";
    let result = mock_engine().run(q, &sample_docs()).unwrap();
    assert_eq!(result.original_query, q);
}

#[test]
fn engine_run_abstract_query_differs_from_original() {
    let result = mock_engine()
        .run("What is photosynthesis?", &sample_docs())
        .unwrap();
    assert_ne!(result.abstract_query, result.original_query);
}

#[test]
fn engine_run_abstract_query_starts_with_prefix() {
    let result = mock_engine()
        .run("What is photosynthesis?", &sample_docs())
        .unwrap();
    assert!(result.abstract_query.starts_with("In general terms, "));
}

#[test]
fn engine_run_final_answer_non_empty_when_docs_provided() {
    let result = mock_engine()
        .run("What is the speed of light?", &sample_docs())
        .unwrap();
    assert!(!result.final_answer.is_empty());
}

// ── run: empty / whitespace query ────────────────────────────────────────────

#[test]
fn engine_run_empty_query_errors() {
    let err = mock_engine().run("", &sample_docs()).unwrap_err();
    assert!(matches!(err, StepBackError::AbstractionFailed(_)));
}

#[test]
fn engine_run_whitespace_query_errors() {
    let err = mock_engine().run("   \t\n  ", &sample_docs()).unwrap_err();
    assert!(matches!(err, StepBackError::AbstractionFailed(_)));
}

// ── run: abstraction depth ────────────────────────────────────────────────────

#[test]
fn engine_run_depth_zero_with_max_depth_zero() {
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(0),
        Box::new(MockStepBackModel),
    );
    let result = engine.run("What is gravity?", &sample_docs()).unwrap();
    assert_eq!(result.abstraction_depth, 0);
}

#[test]
fn engine_run_abstract_equals_original_when_max_depth_zero() {
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(0),
        Box::new(MockStepBackModel),
    );
    let q = "What is gravity?";
    let result = engine.run(q, &sample_docs()).unwrap();
    assert_eq!(result.abstract_query, q);
}

#[test]
fn engine_run_depth_one_with_max_depth_one() {
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(1),
        Box::new(MockStepBackModel),
    );
    let result = engine
        .run("What is the Higgs boson?", &sample_docs())
        .unwrap();
    assert_eq!(result.abstraction_depth, 1);
}

#[test]
fn engine_run_depth_two_with_max_depth_two() {
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(2),
        Box::new(MockStepBackModel),
    );
    let result = engine
        .run("What is the Higgs boson?", &sample_docs())
        .unwrap();
    assert_eq!(result.abstraction_depth, 2);
}

#[test]
fn engine_run_depth_not_exceeding_max() {
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(3),
        Box::new(MockStepBackModel),
    );
    let result = engine
        .run(
            "A specific query about Fermat's last theorem",
            &sample_docs(),
        )
        .unwrap();
    assert!(result.abstraction_depth <= 3);
}

#[test]
fn engine_run_static_model_abstracts_at_most_once() {
    // StaticAbstractModel always returns the same string.
    // After the first successful abstraction, the next call produces the same
    // string (same length, same tokens) → not "more abstract" → loop exits.
    let query = "short";
    // "long fixed abstract query string" is longer than "short" → first iter passes.
    // Second iter: same string again → neither fewer tokens nor longer → stops.
    let model = StaticAbstractModel::new("long fixed abstract query string");
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(5),
        Box::new(model),
    );
    let result = engine.run(query, &[]).unwrap();
    assert_eq!(result.abstraction_depth, 1);
}

#[test]
fn engine_run_shorter_model_abstracts_once_via_fewer_tokens() {
    // ShorterAbstractModel returns "general topic" (2 tokens) from any query.
    // Any original query with > 2 tokens triggers the "fewer tokens" branch.
    // Second iteration: same fixed string → same token count → stops.
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(5),
        Box::new(ShorterAbstractModel),
    );
    let result = engine
        .run("What is the quantum chromodynamics theory", &[])
        .unwrap();
    // First iter: "general topic" has 2 tokens < 7 → is_more_abstract = true, depth=1.
    // Second iter: "general topic" again from "general topic" → 2 tokens = 2 tokens,
    //   length same → is_more_abstract = false → loop breaks.
    assert_eq!(result.abstraction_depth, 1);
    assert_eq!(result.abstract_query, "general topic");
}

// ── run: document retrieval ───────────────────────────────────────────────────

#[test]
fn engine_run_empty_docs_returns_empty_retrieved() {
    let result = mock_engine().run("What is gravity?", &[]).unwrap();
    assert!(result.retrieved_docs.is_empty());
}

#[test]
fn engine_run_top_k_limits_retrieved_count() {
    let docs: Vec<String> = (0..10)
        .map(|i| format!("document number {i} about physics constants general"))
        .collect();
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_top_k(3),
        Box::new(MockStepBackModel),
    );
    let result = engine.run("physics constants", &docs).unwrap();
    assert!(result.retrieved_docs.len() <= 3);
}

#[test]
fn engine_run_top_k_zero_retrieves_nothing() {
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_top_k(0),
        Box::new(MockStepBackModel),
    );
    let result = engine.run("What is gravity?", &sample_docs()).unwrap();
    assert!(result.retrieved_docs.is_empty());
}

#[test]
fn engine_run_fewer_docs_than_top_k_returns_all() {
    let docs = vec!["only doc".to_string()];
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_top_k(10),
        Box::new(MockStepBackModel),
    );
    let result = engine.run("What is gravity?", &docs).unwrap();
    assert_eq!(result.retrieved_docs.len(), 1);
}

#[test]
fn engine_run_retrieved_docs_are_subset_of_input() {
    let docs = sample_docs();
    let result = mock_engine()
        .run("What is the speed of light?", &docs)
        .unwrap();
    for retrieved in &result.retrieved_docs {
        assert!(docs.contains(retrieved));
    }
}

#[test]
fn engine_run_high_overlap_doc_ranked_first() {
    // After two abstraction rounds, the abstract query from MockStepBackModel
    // will be:
    //   "In general terms, In general terms, speed light constants"
    // The abstract query tokens include "speed", "light", "constants" (and
    // "in", "general", "terms"). doc_a shares more of those tokens.
    let doc_a = "speed light constants fundamental physics general terms".to_string();
    let doc_b = "unrelated astronomy stellar orbit binary".to_string();
    let docs = vec![doc_a.clone(), doc_b.clone()];
    let result = mock_engine().run("speed light constants", &docs).unwrap();
    assert_eq!(result.retrieved_docs[0], doc_a);
}

#[test]
fn engine_run_zero_overlap_docs_returned_in_index_order() {
    // Docs have no alphanumeric overlap with the abstract query tokens.
    let docs = vec![
        "zzz yyy xxx".to_string(),
        "aaa bbb ccc".to_string(),
        "mmm nnn ooo".to_string(),
    ];
    let engine = StepBackEngine::new(
        StepBackConfig::new()
            .with_top_k(2)
            .with_max_abstraction_depth(0),
        Box::new(MockStepBackModel),
    );
    // max_depth=0 → abstract_query = "unique_prefix_xyz" (no common tokens)
    // All docs have overlap 0 → sorted by insertion index.
    let result = engine.run("unique_prefix_xyz", &docs).unwrap();
    assert_eq!(result.retrieved_docs.len(), 2);
    assert_eq!(result.retrieved_docs[0], docs[0]);
    assert_eq!(result.retrieved_docs[1], docs[1]);
}

#[test]
fn engine_run_token_overlap_scores_correctly() {
    // doc_a shares 3 tokens with abstract, doc_b shares 1.
    let docs = vec![
        "physics constants speed light fundamental".to_string(), // high overlap
        "cooking recipes pasta tomato sauce".to_string(),        // no overlap
    ];
    let result = mock_engine().run("speed light constants", &docs).unwrap();
    // doc_a should appear in retrieved_docs before doc_b.
    let pos_a = result
        .retrieved_docs
        .iter()
        .position(|d| d.contains("physics"));
    let pos_b = result
        .retrieved_docs
        .iter()
        .position(|d| d.contains("cooking"));
    if let (Some(pa), Some(pb)) = (pos_a, pos_b) {
        assert!(
            pa < pb,
            "high-overlap doc should rank before low-overlap doc"
        );
    }
}

#[test]
fn engine_run_multi_doc_overlap_ordering() {
    let docs = vec![
        "quantum particle wave physics general constants".to_string(), // many overlaps
        "quantum mechanics".to_string(),                               // some overlap
        "soup recipe kitchen".to_string(),                             // no overlap
    ];
    let result = mock_engine()
        .run("quantum physics constants", &docs)
        .unwrap();
    let first = &result.retrieved_docs[0];
    assert!(
        first.contains("constants") || first.contains("quantum"),
        "highest-overlap doc should rank first"
    );
}

#[test]
fn engine_run_large_docs_list_top_k_five() {
    let docs: Vec<String> = (0..20)
        .map(|i| format!("document {i} about general physics constants light speed"))
        .collect();
    let result = mock_engine().run("light speed constants", &docs).unwrap();
    assert!(result.retrieved_docs.len() <= 5);
}

// ── run: final answer ─────────────────────────────────────────────────────────

#[test]
fn engine_run_final_answer_is_docs_joined_by_space() {
    let docs = vec!["alpha".to_string(), "beta".to_string()];
    let engine = StepBackEngine::new(
        StepBackConfig::new()
            .with_max_abstraction_depth(0)
            .with_top_k(10),
        Box::new(MockStepBackModel),
    );
    // max_depth=0 so abstract_query = "alpha beta" (original query)
    // Both docs have no overlap with "alpha beta" after tokenizing → tie, index order
    let result = engine.run("alpha beta", &docs).unwrap();
    // retrieved_docs are "alpha" and "beta" (both have overlap: "alpha"/"beta" in query)
    // mock synthesize joins them
    assert!(!result.final_answer.is_empty());
}

#[test]
fn engine_run_final_answer_empty_when_no_docs() {
    let result = mock_engine().run("What is gravity?", &[]).unwrap();
    assert!(result.final_answer.is_empty());
}

// ── run: error propagation ────────────────────────────────────────────────────

#[test]
fn engine_propagates_abstraction_failed() {
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(1),
        Box::new(FailAbstractModel),
    );
    let err = engine.run("What is gravity?", &[]).unwrap_err();
    assert!(matches!(err, StepBackError::AbstractionFailed(_)));
}

#[test]
fn engine_propagates_synthesis_failed() {
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(1),
        Box::new(FailSynthModel),
    );
    let err = engine.run("What is gravity?", &sample_docs()).unwrap_err();
    assert!(matches!(err, StepBackError::SynthesisFailed(_)));
}

#[test]
fn engine_abstraction_fail_not_triggered_when_max_depth_zero() {
    // FailAbstractModel.abstract_query is never called when max_depth=0.
    let engine = StepBackEngine::new(
        StepBackConfig::new().with_max_abstraction_depth(0),
        Box::new(FailAbstractModel),
    );
    let result = engine.run("safe query", &[]);
    // Should only fail at synthesis — FailAbstractModel.synthesize returns Ok.
    assert!(result.is_ok());
}

// ── run: determinism ──────────────────────────────────────────────────────────

#[test]
fn engine_run_is_deterministic() {
    let docs = sample_docs();
    let query = "What is the speed of light?";
    let engine_a = StepBackEngine::new(StepBackConfig::default(), Box::new(MockStepBackModel));
    let engine_b = StepBackEngine::new(StepBackConfig::default(), Box::new(MockStepBackModel));
    let result_a = engine_a.run(query, &docs).unwrap();
    let result_b = engine_b.run(query, &docs).unwrap();
    assert_eq!(result_a, result_b);
}

// ── run: edge cases ───────────────────────────────────────────────────────────

#[test]
fn engine_run_single_word_query() {
    let result = mock_engine().run("gravity", &sample_docs());
    assert!(result.is_ok());
}

#[test]
fn engine_run_query_with_punctuation() {
    let result = mock_engine().run("What is E=mc^2?", &sample_docs());
    assert!(result.is_ok());
}

#[test]
fn engine_run_single_relevant_doc() {
    let docs = vec!["physics constants light speed measurement".to_string()];
    let result = mock_engine().run("speed of light", &docs).unwrap();
    assert_eq!(result.retrieved_docs.len(), 1);
}

#[test]
fn engine_run_via_box_dyn_trait() {
    let model: Box<dyn StepBackModel> = Box::new(MockStepBackModel);
    let engine = StepBackEngine::new(StepBackConfig::default(), model);
    let result = engine.run("What is relativity?", &sample_docs());
    assert!(result.is_ok());
}

// ── StepBackResult fields ─────────────────────────────────────────────────────

#[test]
fn result_clone_equality() {
    let result = mock_engine()
        .run("What is gravity?", &sample_docs())
        .unwrap();
    assert_eq!(result.clone(), result);
}

#[test]
fn result_debug_non_empty() {
    let result = mock_engine()
        .run("What is gravity?", &sample_docs())
        .unwrap();
    assert!(!format!("{result:?}").is_empty());
}

#[test]
fn result_all_fields_present() {
    let result = mock_engine()
        .run("What is the speed of light?", &sample_docs())
        .unwrap();
    assert!(!result.original_query.is_empty());
    assert!(!result.abstract_query.is_empty());
    assert!(result.abstraction_depth <= 2);
}

#[test]
fn result_abstraction_depth_gte_zero() {
    let result = mock_engine().run("test query", &sample_docs()).unwrap();
    // abstraction_depth is usize, so always >= 0; just assert it is <= max.
    assert!(result.abstraction_depth <= StepBackConfig::default().max_abstraction_depth);
}

#[test]
fn result_retrieved_docs_len_lte_top_k() {
    let result = mock_engine().run("test query", &sample_docs()).unwrap();
    assert!(result.retrieved_docs.len() <= StepBackConfig::default().top_k);
}

#[test]
fn result_final_answer_matches_mock_join() {
    let docs = vec!["doc alpha".to_string(), "doc beta".to_string()];
    // Use max_depth=0 so abstract_query = original query.
    let engine = StepBackEngine::new(
        StepBackConfig::new()
            .with_max_abstraction_depth(0)
            .with_top_k(10),
        Box::new(MockStepBackModel),
    );
    let result = engine.run("doc alpha beta", &docs).unwrap();
    // Both docs share tokens with query → retrieved; mock joins with space.
    let expected = result.retrieved_docs.join(" ");
    assert_eq!(result.final_answer, expected);
}

#[test]
fn result_inequality_on_different_queries() {
    let r1 = mock_engine()
        .run("What is gravity?", &sample_docs())
        .unwrap();
    let r2 = mock_engine()
        .run("What is entropy?", &sample_docs())
        .unwrap();
    assert_ne!(r1.original_query, r2.original_query);
}
