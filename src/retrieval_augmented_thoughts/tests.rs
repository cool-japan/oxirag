//! Unit tests for the `retrieval_augmented_thoughts` module.

use crate::retrieval_augmented_thoughts::engine::RatEngine;
use crate::retrieval_augmented_thoughts::types::{
    MockRatGenerator, MockRatRetriever, RatConfig, RatError, RatGenerator, RatRetriever,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn engine_default(
    generator: MockRatGenerator,
    retriever: MockRatRetriever,
) -> RatEngine<MockRatGenerator, MockRatRetriever> {
    RatEngine::new(RatConfig::default(), generator, retriever)
}

fn two_step_generator() -> MockRatGenerator {
    MockRatGenerator::new(vec![
        "The Eiffel Tower is in Berlin.".to_string(),
        "It was completed in 1990.".to_string(),
    ])
}

fn eiffel_retriever() -> MockRatRetriever {
    MockRatRetriever::new(vec![
        "The Eiffel Tower is located in Paris, France.".to_string(),
        "Construction of the Eiffel Tower finished in 1889.".to_string(),
    ])
}

// ── RatConfig: defaults ───────────────────────────────────────────────────────

#[test]
fn config_default_max_thoughts_is_five() {
    assert_eq!(RatConfig::default().max_thoughts, 5);
}

#[test]
fn config_default_top_k_is_three() {
    assert_eq!(RatConfig::default().top_k, 3);
}

#[test]
fn config_default_min_thought_len_is_zero() {
    assert_eq!(RatConfig::default().min_thought_len, 0);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(RatConfig::new(), RatConfig::default());
}

// ── RatConfig: builders ───────────────────────────────────────────────────────

#[test]
fn config_builder_sets_max_thoughts() {
    assert_eq!(RatConfig::new().with_max_thoughts(9).max_thoughts, 9);
}

#[test]
fn config_builder_sets_top_k() {
    assert_eq!(RatConfig::new().with_top_k(7).top_k, 7);
}

#[test]
fn config_builder_sets_min_thought_len() {
    assert_eq!(
        RatConfig::new().with_min_thought_len(12).min_thought_len,
        12
    );
}

#[test]
fn config_builder_chains() {
    let cfg = RatConfig::new()
        .with_max_thoughts(2)
        .with_top_k(1)
        .with_min_thought_len(4);
    assert_eq!(cfg.max_thoughts, 2);
    assert_eq!(cfg.top_k, 1);
    assert_eq!(cfg.min_thought_len, 4);
}

#[test]
fn config_clone_equals_original() {
    let cfg = RatConfig::new().with_max_thoughts(3);
    assert_eq!(cfg.clone(), cfg);
}

// ── RatConfig: validate ───────────────────────────────────────────────────────

#[test]
fn config_validate_default_is_ok() {
    assert!(RatConfig::default().validate().is_ok());
}

#[test]
fn config_validate_zero_max_thoughts_errors() {
    let cfg = RatConfig::new().with_max_thoughts(0);
    assert!(matches!(
        cfg.validate().unwrap_err(),
        RatError::InvalidConfig(_)
    ));
}

#[test]
fn config_validate_zero_top_k_is_ok() {
    // top_k == 0 is a legal degenerate case: no retrieval augmentation.
    let cfg = RatConfig::new().with_top_k(0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn config_validate_large_min_thought_len_is_ok() {
    let cfg = RatConfig::new().with_min_thought_len(usize::MAX);
    assert!(cfg.validate().is_ok());
}

// ── MockRatGenerator ──────────────────────────────────────────────────────────

#[test]
fn mock_generator_new_stores_drafts() {
    let mock_gen = MockRatGenerator::new(vec!["a".to_string(), "b".to_string()]);
    assert_eq!(mock_gen.drafts, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn mock_generator_empty_has_no_drafts() {
    assert!(MockRatGenerator::empty().drafts.is_empty());
}

#[test]
fn mock_generator_default_has_no_drafts() {
    assert!(MockRatGenerator::default().drafts.is_empty());
}

#[test]
fn mock_generator_draft_cot_ignores_question() {
    let mock_gen = MockRatGenerator::new(vec!["fixed".to_string()]);
    assert_eq!(
        mock_gen.draft_cot("q1"),
        mock_gen.draft_cot("a completely different q")
    );
}

#[test]
fn mock_generator_draft_cot_returns_script() {
    let mock_gen = two_step_generator();
    assert_eq!(
        mock_gen.draft_cot("q"),
        vec![
            "The Eiffel Tower is in Berlin.".to_string(),
            "It was completed in 1990.".to_string(),
        ]
    );
}

#[test]
fn mock_generator_revise_empty_passages_returns_draft_unchanged() {
    let mock_gen = MockRatGenerator::empty();
    assert_eq!(mock_gen.revise("q", &[], "draft text", &[]), "draft text");
}

#[test]
fn mock_generator_revise_with_passages_appends_confirmation() {
    let mock_gen = MockRatGenerator::empty();
    let revised = mock_gen.revise("q", &[], "draft", &["evidence one".to_string()]);
    assert_eq!(revised, "draft (confirmed by: evidence one)");
}

#[test]
fn mock_generator_revise_joins_multiple_passages() {
    let mock_gen = MockRatGenerator::empty();
    let revised = mock_gen.revise("q", &[], "draft", &["p1".to_string(), "p2".to_string()]);
    assert_eq!(revised, "draft (confirmed by: p1; p2)");
}

#[test]
fn mock_generator_revise_ignores_prior_and_question() {
    let mock_gen = MockRatGenerator::empty();
    let a = mock_gen.revise("q1", &["prior".to_string()], "draft", &[]);
    let b = mock_gen.revise("q2", &[], "draft", &[]);
    assert_eq!(a, b);
}

#[test]
fn mock_generator_finalize_joins_thoughts() {
    let mock_gen = MockRatGenerator::empty();
    let out = mock_gen.finalize("q", &["t1".to_string(), "t2".to_string()]);
    assert_eq!(out, "Final answer based on: t1 -> t2");
}

#[test]
fn mock_generator_finalize_empty_thoughts() {
    let mock_gen = MockRatGenerator::empty();
    assert_eq!(mock_gen.finalize("q", &[]), "Final answer based on: ");
}

// ── MockRatRetriever ──────────────────────────────────────────────────────────

#[test]
fn mock_retriever_new_stores_corpus() {
    let r = MockRatRetriever::new(vec!["p1".to_string()]);
    assert_eq!(r.corpus, vec!["p1".to_string()]);
}

#[test]
fn mock_retriever_empty_has_no_corpus() {
    assert!(MockRatRetriever::empty().corpus.is_empty());
}

#[test]
fn mock_retriever_default_has_no_corpus() {
    assert!(MockRatRetriever::default().corpus.is_empty());
}

#[test]
fn mock_retriever_empty_corpus_returns_nothing() {
    let r = MockRatRetriever::empty();
    assert!(r.retrieve("anything", 3).is_empty());
}

#[test]
fn mock_retriever_zero_top_k_returns_nothing() {
    let r = eiffel_retriever();
    assert!(r.retrieve("Eiffel Tower", 0).is_empty());
}

#[test]
fn mock_retriever_no_overlap_returns_nothing() {
    let r = eiffel_retriever();
    assert!(r.retrieve("zzz qqq wholly unrelated", 3).is_empty());
}

#[test]
fn mock_retriever_empty_query_returns_nothing() {
    let r = eiffel_retriever();
    assert!(r.retrieve("", 3).is_empty());
}

#[test]
fn mock_retriever_ranks_by_overlap_descending() {
    let r = MockRatRetriever::new(vec![
        "cats and dogs are pets".to_string(),
        "cats are wonderful furry cats cats".to_string(),
    ]);
    // "cats" appears in both; second passage also shares "wonderful"-adjacent
    // overlap is symmetric here, so use a query that clearly favors index 1.
    let results = r.retrieve("cats furry wonderful", 2);
    assert_eq!(results[0], "cats are wonderful furry cats cats");
}

#[test]
fn mock_retriever_respects_top_k_limit() {
    let r = MockRatRetriever::new(vec![
        "alpha shared".to_string(),
        "beta shared".to_string(),
        "gamma shared".to_string(),
    ]);
    let results = r.retrieve("shared", 2);
    assert_eq!(results.len(), 2);
}

#[test]
fn mock_retriever_returns_all_when_top_k_exceeds_matches() {
    let r = MockRatRetriever::new(vec!["shared one".to_string(), "shared two".to_string()]);
    let results = r.retrieve("shared", 10);
    assert_eq!(results.len(), 2);
}

#[test]
fn mock_retriever_ties_broken_by_corpus_order() {
    let r = MockRatRetriever::new(vec!["shared alpha".to_string(), "shared beta".to_string()]);
    let results = r.retrieve("shared", 2);
    assert_eq!(
        results,
        vec!["shared alpha".to_string(), "shared beta".to_string()]
    );
}

#[test]
fn mock_retriever_is_case_insensitive() {
    let r = MockRatRetriever::new(vec!["Eiffel Tower".to_string()]);
    let results = r.retrieve("eiffel tower", 1);
    assert_eq!(results, vec!["Eiffel Tower".to_string()]);
}

#[test]
fn mock_retriever_strips_punctuation() {
    let r = MockRatRetriever::new(vec!["Paris, France.".to_string()]);
    let results = r.retrieve("Where is Paris?", 1);
    assert_eq!(results, vec!["Paris, France.".to_string()]);
}

#[test]
fn mock_retriever_eiffel_example_returns_both_passages() {
    let r = eiffel_retriever();
    let results = r.retrieve("Where and when was the Eiffel Tower built?", 3);
    assert_eq!(results.len(), 2);
}

// ── RatEngine: input validation ───────────────────────────────────────────────

#[test]
fn run_empty_question_errors() {
    let err = engine_default(two_step_generator(), eiffel_retriever())
        .run("")
        .unwrap_err();
    assert!(matches!(err, RatError::EmptyQuestion));
}

#[test]
fn run_whitespace_question_errors() {
    let err = engine_default(two_step_generator(), eiffel_retriever())
        .run("   \t\n ")
        .unwrap_err();
    assert!(matches!(err, RatError::EmptyQuestion));
}

#[test]
fn run_empty_question_error_message() {
    let err = engine_default(two_step_generator(), eiffel_retriever())
        .run("")
        .unwrap_err();
    assert_eq!(err.to_string(), "question must not be empty");
}

#[test]
fn run_invalid_config_errors() {
    let cfg = RatConfig::new().with_max_thoughts(0);
    let engine = RatEngine::new(cfg, two_step_generator(), eiffel_retriever());
    let err = engine.run("a real question").unwrap_err();
    assert!(matches!(err, RatError::InvalidConfig(_)));
}

#[test]
fn run_invalid_config_error_message() {
    let cfg = RatConfig::new().with_max_thoughts(0);
    let engine = RatEngine::new(cfg, two_step_generator(), eiffel_retriever());
    let err = engine.run("a real question").unwrap_err();
    assert!(err.to_string().contains("max_thoughts"));
}

#[test]
fn run_no_thoughts_errors() {
    let engine = engine_default(MockRatGenerator::empty(), eiffel_retriever());
    let err = engine
        .run("a question with no drafted thoughts")
        .unwrap_err();
    assert!(matches!(err, RatError::NoThoughts));
}

#[test]
fn run_no_thoughts_error_message() {
    let engine = engine_default(MockRatGenerator::empty(), eiffel_retriever());
    let err = engine.run("q").unwrap_err();
    assert_eq!(err.to_string(), "no thoughts were drafted");
}

// ── RatEngine: trace shape ────────────────────────────────────────────────────

#[test]
fn run_trace_records_trimmed_question() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("  a question with padding  ").unwrap();
    assert_eq!(trace.question, "a question with padding");
}

#[test]
fn run_trace_has_expected_thought_count() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("Where was the tower built?").unwrap();
    assert_eq!(trace.thoughts.len(), 2);
}

#[test]
fn run_trace_num_thoughts_matches_len() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("q").unwrap();
    assert_eq!(trace.num_thoughts(), trace.thoughts.len());
}

#[test]
fn run_trace_thought_indices_are_sequential() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("q").unwrap();
    let indices: Vec<usize> = trace.thoughts.iter().map(|t| t.index).collect();
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn run_trace_final_answer_not_empty() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("q").unwrap();
    assert!(!trace.final_answer.is_empty());
}

#[test]
fn run_trace_final_answer_equals_finalize_output() {
    let generator = two_step_generator();
    let engine = engine_default(generator.clone(), eiffel_retriever());
    let trace = engine.run("q").unwrap();
    let revised_owned: Vec<String> = trace.thoughts.iter().map(|t| t.revised.clone()).collect();
    let expected = generator.finalize("q", &revised_owned);
    assert_eq!(trace.final_answer, expected);
}

#[test]
fn run_trace_revised_thoughts_matches_thoughts_revised_field() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("q").unwrap();
    let expected: Vec<&str> = trace.thoughts.iter().map(|t| t.revised.as_str()).collect();
    assert_eq!(trace.revised_thoughts(), expected);
}

#[test]
fn run_trace_truncates_to_max_thoughts() {
    let generator = MockRatGenerator::new(vec![
        "d0".to_string(),
        "d1".to_string(),
        "d2".to_string(),
        "d3".to_string(),
    ]);
    let cfg = RatConfig::new().with_max_thoughts(2);
    let engine = RatEngine::new(cfg, generator, MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert_eq!(trace.thoughts.len(), 2);
}

#[test]
fn run_trace_truncation_keeps_first_steps() {
    let generator =
        MockRatGenerator::new(vec!["d0".to_string(), "d1".to_string(), "d2".to_string()]);
    let cfg = RatConfig::new().with_max_thoughts(1);
    let engine = RatEngine::new(cfg, generator, MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert_eq!(trace.thoughts[0].draft, "d0");
}

#[test]
fn run_single_thought_chain() {
    let generator = MockRatGenerator::new(vec!["only thought".to_string()]);
    let engine = engine_default(generator, MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert_eq!(trace.thoughts.len(), 1);
    assert_eq!(trace.thoughts[0].draft, "only thought");
}

// ── RatEngine: loop correctness (retrieval query construction) ───────────────

#[test]
fn run_first_step_query_has_no_prior() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("Where is the tower?").unwrap();
    assert_eq!(
        trace.thoughts[0].retrieval_query,
        "Where is the tower? The Eiffel Tower is in Berlin."
    );
}

#[test]
fn run_second_step_query_includes_prior_revised_thought() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("Where is the tower?").unwrap();
    let prior_revised = &trace.thoughts[0].revised;
    assert!(
        trace.thoughts[1]
            .retrieval_query
            .contains(prior_revised.as_str())
    );
}

#[test]
fn run_second_step_query_includes_question() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("Where is the tower?").unwrap();
    assert!(
        trace.thoughts[1]
            .retrieval_query
            .starts_with("Where is the tower?")
    );
}

#[test]
fn run_second_step_query_includes_own_draft() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("Where is the tower?").unwrap();
    assert!(
        trace.thoughts[1]
            .retrieval_query
            .ends_with("It was completed in 1990.")
    );
}

#[test]
fn run_uses_draft_not_revised_text_for_own_step_query() {
    // Step 0's own query must use its *draft* text, not any already-revised
    // text (there is none yet at step 0).
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine.run("q").unwrap();
    assert!(
        trace.thoughts[0]
            .retrieval_query
            .ends_with("The Eiffel Tower is in Berlin.")
    );
}

// ── RatEngine: revision uses retrieval ────────────────────────────────────────

#[test]
fn run_revision_incorporates_retrieved_passages() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine
        .run("Where and when was the Eiffel Tower built?")
        .unwrap();
    assert!(trace.thoughts[0].revised.contains("Paris"));
}

#[test]
fn run_revision_differs_from_draft_when_passages_found() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine
        .run("Where and when was the Eiffel Tower built?")
        .unwrap();
    assert!(trace.thoughts[0].was_revised());
}

#[test]
fn run_thought_has_passages_when_retrieval_hits() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine
        .run("Where and when was the Eiffel Tower built?")
        .unwrap();
    assert!(trace.thoughts[0].has_passages());
}

#[test]
fn run_no_retrieval_hits_leaves_thought_unrevised() {
    let generator = MockRatGenerator::new(vec!["zzz qqq no overlap at all".to_string()]);
    let engine = engine_default(generator, eiffel_retriever());
    let trace = engine.run("zzz qqq totally unrelated query").unwrap();
    assert!(!trace.thoughts[0].was_revised());
    assert_eq!(trace.thoughts[0].draft, trace.thoughts[0].revised);
}

#[test]
fn run_empty_retriever_corpus_leaves_thoughts_unrevised() {
    let engine = engine_default(two_step_generator(), MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert!(trace.thoughts.iter().all(|t| !t.was_revised()));
}

#[test]
fn run_empty_retriever_corpus_yields_no_passages() {
    let engine = engine_default(two_step_generator(), MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert!(trace.thoughts.iter().all(|t| !t.has_passages()));
}

#[test]
fn run_retrieval_query_recorded_even_without_hits() {
    let engine = engine_default(two_step_generator(), MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert!(!trace.thoughts[0].retrieval_query.is_empty());
}

// ── RatEngine: min_thought_len skip behaviour ────────────────────────────────

#[test]
fn run_short_thought_below_min_len_is_skipped() {
    let generator = MockRatGenerator::new(vec![
        ".".to_string(),
        "a long draft thought here".to_string(),
    ]);
    let cfg = RatConfig::new().with_min_thought_len(10);
    let engine = RatEngine::new(cfg, generator, eiffel_retriever());
    let trace = engine.run("q").unwrap();
    assert_eq!(trace.thoughts[0].retrieval_query, "");
    assert!(trace.thoughts[0].passages.is_empty());
    assert_eq!(trace.thoughts[0].draft, trace.thoughts[0].revised);
}

#[test]
fn run_long_thought_above_min_len_is_processed() {
    let generator = MockRatGenerator::new(vec![
        ".".to_string(),
        "a long enough draft thought".to_string(),
    ]);
    let cfg = RatConfig::new().with_min_thought_len(10);
    let engine = RatEngine::new(cfg, generator, eiffel_retriever());
    let trace = engine.run("q").unwrap();
    assert!(!trace.thoughts[1].retrieval_query.is_empty());
}

#[test]
fn run_min_thought_len_zero_processes_all() {
    let generator = MockRatGenerator::new(vec![".".to_string()]);
    let cfg = RatConfig::new().with_min_thought_len(0);
    let engine = RatEngine::new(cfg, generator, eiffel_retriever());
    let trace = engine.run("q").unwrap();
    assert!(!trace.thoughts[0].retrieval_query.is_empty());
}

#[test]
fn run_min_thought_len_boundary_is_inclusive() {
    // A draft of exactly `min_thought_len` bytes should be processed (not
    // skipped): the skip condition is a strict "<", not "<=".
    let generator = MockRatGenerator::new(vec!["12345".to_string()]);
    let cfg = RatConfig::new().with_min_thought_len(5);
    let engine = RatEngine::new(cfg, generator, MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert!(!trace.thoughts[0].retrieval_query.is_empty());
}

// ── RatEngine: top_k wiring ────────────────────────────────────────────────────

#[test]
fn run_top_k_limits_passages_per_thought() {
    let generator = MockRatGenerator::new(vec!["shared".to_string()]);
    let retriever = MockRatRetriever::new(vec![
        "shared one".to_string(),
        "shared two".to_string(),
        "shared three".to_string(),
    ]);
    let cfg = RatConfig::new().with_top_k(1);
    let engine = RatEngine::new(cfg, generator, retriever);
    let trace = engine.run("shared").unwrap();
    assert_eq!(trace.thoughts[0].passages.len(), 1);
}

#[test]
fn run_zero_top_k_yields_no_passages() {
    let generator = MockRatGenerator::new(vec!["shared".to_string()]);
    let cfg = RatConfig::new().with_top_k(0);
    let engine = RatEngine::new(cfg, generator, eiffel_retriever());
    let trace = engine.run("shared").unwrap();
    assert!(trace.thoughts[0].passages.is_empty());
}

// ── RatEngine: determinism ────────────────────────────────────────────────────

#[test]
fn run_is_deterministic() {
    let a = engine_default(two_step_generator(), eiffel_retriever())
        .run("Where and when was the Eiffel Tower built?")
        .unwrap();
    let b = engine_default(two_step_generator(), eiffel_retriever())
        .run("Where and when was the Eiffel Tower built?")
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn run_is_deterministic_with_empty_retriever() {
    let a = engine_default(two_step_generator(), MockRatRetriever::empty())
        .run("q")
        .unwrap();
    let b = engine_default(two_step_generator(), MockRatRetriever::empty())
        .run("q")
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn run_repeated_calls_on_same_engine_agree() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let a = engine.run("q").unwrap();
    let b = engine.run("q").unwrap();
    assert_eq!(a, b);
}

// ── RatEngine: construction ────────────────────────────────────────────────────

#[test]
fn engine_new_stores_config() {
    let cfg = RatConfig::new().with_max_thoughts(9);
    let engine = RatEngine::new(cfg.clone(), two_step_generator(), eiffel_retriever());
    assert_eq!(engine.config, cfg);
}

// ── RatEngine: trait objects (dyn) compile and run ────────────────────────────

#[test]
fn run_accepts_trait_objects() {
    let generator = two_step_generator();
    let retriever = eiffel_retriever();
    let generator_ref: &dyn RatGenerator = &generator;
    let retriever_ref: &dyn RatRetriever = &retriever;
    let engine = RatEngine::new(RatConfig::default(), generator_ref, retriever_ref);
    let trace = engine
        .run("Where and when was the Eiffel Tower built?")
        .unwrap();
    assert!(!trace.final_answer.is_empty());
}

#[test]
fn run_trait_object_matches_owned_result() {
    let generator = two_step_generator();
    let retriever = eiffel_retriever();
    let owned = engine_default(generator.clone(), retriever.clone())
        .run("q")
        .unwrap();

    let generator_ref: &dyn RatGenerator = &generator;
    let retriever_ref: &dyn RatRetriever = &retriever;
    let via_dyn = RatEngine::new(RatConfig::default(), generator_ref, retriever_ref)
        .run("q")
        .unwrap();

    assert_eq!(owned, via_dyn);
}

// ── RatThought helpers ────────────────────────────────────────────────────────

#[test]
fn thought_was_revised_true_when_texts_differ() {
    let engine = engine_default(two_step_generator(), eiffel_retriever());
    let trace = engine
        .run("Where and when was the Eiffel Tower built?")
        .unwrap();
    assert!(trace.thoughts[0].was_revised());
}

#[test]
fn thought_was_revised_false_when_texts_match() {
    let engine = engine_default(two_step_generator(), MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert!(!trace.thoughts[0].was_revised());
}

#[test]
fn thought_has_passages_false_when_empty() {
    let engine = engine_default(two_step_generator(), MockRatRetriever::empty());
    let trace = engine.run("q").unwrap();
    assert!(!trace.thoughts[0].has_passages());
}
