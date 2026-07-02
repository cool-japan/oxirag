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
//! Unit tests for the `buffer_of_thoughts` module.

use crate::buffer_of_thoughts::engine::{BotEngine, ThoughtBuffer};
use crate::buffer_of_thoughts::types::{
    self, BotConfig, BotError, BotGenerator, DEFAULT_TEMPLATE_TEXT, DistillAction, EvictionPolicy,
    MockBotGenerator, ProblemSignature, SolveOutcome, detect_operation_tags,
    detect_structural_features, extract_numbers, instantiate_template,
};

// ── Sample problems (hand-analyzed signatures — see comments) ─────────────────
//
// PROBLEM_A / PROBLEM_B: both arithmetic "sum of two numbers" questions.
// Identical operation tags {"arithmetic"} and identical structural features
// {"has_numbers", "is_question", "medium_form", "multi_clause"} (the "and"
// joining the two operands triggers `multi_clause`), so they are maximally
// similar (1.0) even though every number differs.
const PROBLEM_A: &str = "What is the sum of 12 and 7?";
const PROBLEM_B: &str = "What is the sum of 30 and 5?";

// PROBLEM_C: a causal "why" question with no numbers — structurally and
// operationally unrelated to A/B (similarity ~0.06, far below any reasonable
// threshold).
const PROBLEM_C: &str = "Why does ice float on water?";

// PROBLEM_D: a combinatorial counting question — also unrelated to A/B/C.
const PROBLEM_D: &str = "How many ways can 3 books be arranged on a shelf?";

fn engine_with(config: BotConfig) -> BotEngine<MockBotGenerator> {
    BotEngine::new(config, MockBotGenerator)
}

fn default_engine() -> BotEngine<MockBotGenerator> {
    engine_with(BotConfig::default())
}

// ── ProblemSignature: signature scheme ─────────────────────────────────────────

#[test]
fn signature_detects_arithmetic_operation_tag() {
    let sig = ProblemSignature::compute(PROBLEM_A);
    assert_eq!(sig.operation_tags, vec!["arithmetic".to_string()]);
}

#[test]
fn signature_detects_causal_operation_tag() {
    let sig = ProblemSignature::compute(PROBLEM_C);
    assert_eq!(sig.operation_tags, vec!["causal".to_string()]);
}

#[test]
fn signature_detects_combinatorial_operation_tag() {
    let sig = ProblemSignature::compute(PROBLEM_D);
    assert_eq!(sig.operation_tags, vec!["combinatorial".to_string()]);
}

#[test]
fn signature_falls_back_to_generic_tag() {
    let sig = ProblemSignature::compute("The cat sat on the mat.");
    assert_eq!(sig.operation_tags, vec!["generic".to_string()]);
}

#[test]
fn signature_structural_features_for_arithmetic_question() {
    let sig = ProblemSignature::compute(PROBLEM_A);
    assert_eq!(
        sig.structural_features,
        vec![
            "has_numbers".to_string(),
            "is_question".to_string(),
            "medium_form".to_string(),
            "multi_clause".to_string(),
        ]
    );
}

#[test]
fn signature_structural_features_short_generic_sentence() {
    let sig = ProblemSignature::compute("The cat sat on the mat.");
    assert_eq!(sig.structural_features, vec!["short_form".to_string()]);
}

#[test]
fn signature_is_question_without_question_mark() {
    let sig = ProblemSignature::compute("How many apples are there");
    assert!(sig.structural_features.contains(&"is_question".to_string()));
}

#[test]
fn signature_detects_negation() {
    let sig = ProblemSignature::compute("This statement is not true.");
    assert!(sig.structural_features.contains(&"negation".to_string()));
}

#[test]
fn signature_detects_list_like() {
    let sig = ProblemSignature::compute("Rank apples, bananas, cherries, and dates by weight.");
    assert!(sig.structural_features.contains(&"list_like".to_string()));
}

#[test]
fn signature_long_form_bucket() {
    let long = "This is a rather long problem statement that contains many words \
                 in order to push the tokenized length comfortably past fifteen \
                 tokens so that the long form bucket is selected deterministically.";
    let sig = ProblemSignature::compute(long);
    assert!(sig.structural_features.contains(&"long_form".to_string()));
}

#[test]
fn signature_compute_never_stores_literal_text() {
    // The signature type has no field capable of holding the raw problem
    // text — this is a structural guarantee, checked here by construction:
    // two different-but-same-kind problems must produce equal signatures.
    let sig_a = ProblemSignature::compute(PROBLEM_A);
    let sig_b = ProblemSignature::compute(PROBLEM_B);
    assert_eq!(sig_a, sig_b);
}

// ── ProblemSignature: similarity ────────────────────────────────────────────────

#[test]
fn similarity_self_is_one() {
    let sig = ProblemSignature::compute(PROBLEM_A);
    assert_eq!(sig.similarity(&sig), 1.0);
}

#[test]
fn similarity_same_kind_different_values_is_one() {
    let sig_a = ProblemSignature::compute(PROBLEM_A);
    let sig_b = ProblemSignature::compute(PROBLEM_B);
    assert_eq!(sig_a.similarity(&sig_b), 1.0);
}

#[test]
fn similarity_is_symmetric() {
    let sig_a = ProblemSignature::compute(PROBLEM_A);
    let sig_c = ProblemSignature::compute(PROBLEM_C);
    assert_eq!(sig_a.similarity(&sig_c), sig_c.similarity(&sig_a));
}

#[test]
fn similarity_unrelated_problems_is_low() {
    let sig_a = ProblemSignature::compute(PROBLEM_A);
    let sig_c = ProblemSignature::compute(PROBLEM_C);
    assert!(sig_a.similarity(&sig_c) < 0.2);
}

#[test]
fn similarity_shares_some_structural_overlap_but_no_operation_overlap() {
    let sig_a = ProblemSignature::compute(PROBLEM_A);
    let sig_d = ProblemSignature::compute(PROBLEM_D);
    let sim = sig_a.similarity(&sig_d);
    assert!(sim > 0.0);
    assert!(sim < 0.5);
}

#[test]
fn similarity_both_empty_tag_sets_defined_as_one() {
    let empty = ProblemSignature {
        operation_tags: Vec::new(),
        structural_features: Vec::new(),
    };
    assert_eq!(empty.similarity(&empty), 1.0);
}

#[test]
fn similarity_weights_sum_to_one() {
    assert_eq!(
        ProblemSignature::OPERATION_WEIGHT + ProblemSignature::STRUCTURAL_WEIGHT,
        1.0
    );
}

#[test]
fn canonical_format_contains_tags_and_features() {
    let sig = ProblemSignature::compute(PROBLEM_A);
    let canonical = sig.canonical();
    assert!(canonical.contains("arithmetic"));
    assert!(canonical.contains("has_numbers"));
    assert!(canonical.starts_with("op:["));
}

// ── Pure helper functions ───────────────────────────────────────────────────────

#[test]
fn extract_numbers_finds_integers() {
    assert_eq!(
        extract_numbers(PROBLEM_A),
        vec!["12".to_string(), "7".to_string()]
    );
}

#[test]
fn extract_numbers_empty_when_no_digits() {
    assert!(extract_numbers(PROBLEM_C).is_empty());
}

#[test]
fn extract_numbers_handles_decimals() {
    assert_eq!(
        extract_numbers("The value is 3.14 approximately."),
        vec!["3.14".to_string()]
    );
}

#[test]
fn detect_operation_tags_direct_arithmetic() {
    let tags = detect_operation_tags(PROBLEM_A);
    assert!(tags.contains("arithmetic"));
}

#[test]
fn detect_structural_features_direct_has_numbers() {
    let features = detect_structural_features(PROBLEM_A);
    assert!(features.contains("has_numbers"));
}

#[test]
fn instantiate_template_substitutes_problem_placeholder() {
    let result = instantiate_template("Solve: {problem}", "2+2");
    assert_eq!(result, "Solve: 2+2");
}

#[test]
fn instantiate_template_substitutes_operand_placeholders() {
    let result = instantiate_template("Combine {op1} and {op2}.", "add 12 and 7");
    assert_eq!(result, "Combine 12 and 7.");
}

#[test]
fn instantiate_template_fills_missing_operands_with_placeholder_marker() {
    let result = instantiate_template("Combine {op1} and {op2}.", "just 5 apples");
    assert_eq!(result, "Combine 5 and ?.");
}

#[test]
fn instantiate_template_leaves_text_without_placeholders_untouched() {
    let result = instantiate_template("A fixed, generic strategy.", "irrelevant problem");
    assert_eq!(result, "A fixed, generic strategy.");
}

#[test]
fn types_module_reexports_default_template_text() {
    // Sanity check the const is reachable both via the module path used in
    // engine.rs and via the `types::` re-export used here.
    assert_eq!(types::DEFAULT_TEMPLATE_TEXT, DEFAULT_TEMPLATE_TEXT);
    assert!(DEFAULT_TEMPLATE_TEXT.contains("{problem}"));
}

// ── BotConfig ────────────────────────────────────────────────────────────────────

#[test]
fn config_default_similarity_threshold_is_half() {
    assert_eq!(BotConfig::default().similarity_threshold, 0.5);
}

#[test]
fn config_default_max_buffer_size_is_two_hundred() {
    assert_eq!(BotConfig::default().max_buffer_size, 200);
}

#[test]
fn config_default_eviction_policy_is_lru() {
    assert_eq!(BotConfig::default().eviction_policy, EvictionPolicy::Lru);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(BotConfig::new(), BotConfig::default());
}

#[test]
fn config_with_similarity_threshold_clamps_above_one() {
    let cfg = BotConfig::new().with_similarity_threshold(5.0);
    assert_eq!(cfg.similarity_threshold, 1.0);
}

#[test]
fn config_with_similarity_threshold_clamps_below_zero() {
    let cfg = BotConfig::new().with_similarity_threshold(-3.0);
    assert_eq!(cfg.similarity_threshold, 0.0);
}

#[test]
fn config_with_max_buffer_size_clamps_zero_to_one() {
    let cfg = BotConfig::new().with_max_buffer_size(0);
    assert_eq!(cfg.max_buffer_size, 1);
}

#[test]
fn config_with_eviction_policy_sets_value() {
    let cfg = BotConfig::new().with_eviction_policy(EvictionPolicy::LowestSuccessRate);
    assert_eq!(cfg.eviction_policy, EvictionPolicy::LowestSuccessRate);
}

#[test]
fn config_builder_chain_preserves_all_fields() {
    let cfg = BotConfig::new()
        .with_similarity_threshold(0.8)
        .with_max_buffer_size(10)
        .with_eviction_policy(EvictionPolicy::LowestSuccessRate);
    assert_eq!(cfg.similarity_threshold, 0.8);
    assert_eq!(cfg.max_buffer_size, 10);
    assert_eq!(cfg.eviction_policy, EvictionPolicy::LowestSuccessRate);
}

// ── MockBotGenerator ─────────────────────────────────────────────────────────────

#[test]
fn mock_generator_generate_includes_scaffold_and_problem() {
    let generator = MockBotGenerator;
    let answer = generator.generate("2+2?", "Step 1: add.");
    assert!(answer.contains("Step 1: add."));
    assert!(answer.contains("2+2?"));
}

#[test]
fn mock_generator_distill_includes_operand_placeholders_matching_number_count() {
    let generator = MockBotGenerator;
    let sig = ProblemSignature::compute(PROBLEM_A);
    let template = generator.distill(PROBLEM_A, "19", &sig);
    assert!(template.contains("{op1}"));
    assert!(template.contains("{op2}"));
    assert!(!template.contains("{op3}"));
    assert!(template.contains("{problem}"));
    assert!(template.contains("arithmetic"));
}

#[test]
fn mock_generator_distill_without_numbers_describes_relevant_values() {
    let generator = MockBotGenerator;
    let sig = ProblemSignature::compute(PROBLEM_C);
    let template = generator.distill(PROBLEM_C, "because density", &sig);
    assert!(template.contains("relevant values"));
    assert!(template.contains("causal"));
    assert!(template.contains("{problem}"));
}

// ── ThoughtBuffer ────────────────────────────────────────────────────────────────

#[test]
fn buffer_new_is_empty() {
    let buffer = ThoughtBuffer::new();
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);
}

#[test]
fn buffer_retrieve_best_on_empty_buffer_is_none() {
    let buffer = ThoughtBuffer::new();
    let sig = ProblemSignature::compute(PROBLEM_A);
    assert!(buffer.retrieve_best(&sig, 0.0).is_none());
}

#[test]
fn buffer_insert_grows_len() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default();
    let sig = ProblemSignature::compute(PROBLEM_A);
    let id = buffer.insert(sig, "Step 1: {problem}".to_string(), &config);
    assert_eq!(buffer.len(), 1);
    assert!(buffer.get(id).is_some());
}

#[test]
fn buffer_inserted_template_starts_with_full_success_rate() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default();
    let sig = ProblemSignature::compute(PROBLEM_A);
    let id = buffer.insert(sig, "Step 1: {problem}".to_string(), &config);
    let template = buffer.get(id).expect("just inserted");
    assert_eq!(template.usage_count, 1);
    assert_eq!(template.success_count, 1);
    assert_eq!(template.success_rate, 1.0);
}

#[test]
fn buffer_reinforce_success_updates_counts() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default();
    let sig = ProblemSignature::compute(PROBLEM_A);
    let id = buffer.insert(sig, "Step 1: {problem}".to_string(), &config);

    buffer.reinforce(id, true).expect("template exists");
    let template = buffer.get(id).expect("still present");
    assert_eq!(template.usage_count, 2);
    assert_eq!(template.success_count, 2);
    assert_eq!(template.success_rate, 1.0);
}

#[test]
fn buffer_reinforce_failure_lowers_success_rate() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default();
    let sig = ProblemSignature::compute(PROBLEM_A);
    let id = buffer.insert(sig, "Step 1: {problem}".to_string(), &config);

    buffer.reinforce(id, false).expect("template exists");
    let template = buffer.get(id).expect("still present");
    assert_eq!(template.usage_count, 2);
    assert_eq!(template.success_count, 1);
    assert_eq!(template.success_rate, 0.5);
}

#[test]
fn buffer_reinforce_missing_id_errors() {
    let mut buffer = ThoughtBuffer::new();
    let err = buffer.reinforce(999, true).unwrap_err();
    assert!(matches!(err, BotError::TemplateNotFound(999)));
}

#[test]
fn buffer_retrieve_best_respects_threshold() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default();
    buffer.insert(
        ProblemSignature::compute(PROBLEM_A),
        "Step 1: {problem}".to_string(),
        &config,
    );

    let query = ProblemSignature::compute(PROBLEM_C);
    assert!(buffer.retrieve_best(&query, 0.5).is_none());
    assert!(buffer.retrieve_best(&query, 0.0).is_some());
}

#[test]
fn buffer_retrieve_best_finds_matching_signature() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default();
    let id = buffer.insert(
        ProblemSignature::compute(PROBLEM_A),
        "Step 1: {problem}".to_string(),
        &config,
    );

    let query = ProblemSignature::compute(PROBLEM_B);
    let (matched_id, sim) = buffer.retrieve_best(&query, 0.5).expect("should match");
    assert_eq!(matched_id, id);
    assert_eq!(sim, 1.0);
}

#[test]
fn buffer_eviction_lru_evicts_oldest_when_full() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default()
        .with_max_buffer_size(2)
        .with_eviction_policy(EvictionPolicy::Lru);

    let id_a = buffer.insert(
        ProblemSignature::compute(PROBLEM_A),
        "arithmetic: {problem}".to_string(),
        &config,
    );
    let id_c = buffer.insert(
        ProblemSignature::compute(PROBLEM_C),
        "causal: {problem}".to_string(),
        &config,
    );
    assert_eq!(buffer.len(), 2);

    // Third insert exceeds capacity — the least-recently-used (id_a, the
    // first ever inserted, never reinforced) should be evicted.
    let id_d = buffer.insert(
        ProblemSignature::compute(PROBLEM_D),
        "combinatorial: {problem}".to_string(),
        &config,
    );

    assert_eq!(buffer.len(), 2);
    assert!(buffer.get(id_a).is_none());
    assert!(buffer.get(id_c).is_some());
    assert!(buffer.get(id_d).is_some());
}

#[test]
fn buffer_eviction_lru_spares_recently_reinforced_template() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default()
        .with_max_buffer_size(2)
        .with_eviction_policy(EvictionPolicy::Lru);

    let id_a = buffer.insert(
        ProblemSignature::compute(PROBLEM_A),
        "arithmetic: {problem}".to_string(),
        &config,
    );
    let id_c = buffer.insert(
        ProblemSignature::compute(PROBLEM_C),
        "causal: {problem}".to_string(),
        &config,
    );

    // Touch id_a so it becomes the most-recently-used entry; id_c is now the
    // least-recently-used one.
    buffer.reinforce(id_a, true).expect("exists");

    let id_d = buffer.insert(
        ProblemSignature::compute(PROBLEM_D),
        "combinatorial: {problem}".to_string(),
        &config,
    );

    assert!(buffer.get(id_a).is_some());
    assert!(buffer.get(id_c).is_none());
    assert!(buffer.get(id_d).is_some());
}

#[test]
fn buffer_eviction_lowest_success_rate_evicts_worst_performer() {
    let mut buffer = ThoughtBuffer::new();
    let config = BotConfig::default()
        .with_max_buffer_size(2)
        .with_eviction_policy(EvictionPolicy::LowestSuccessRate);

    let id_a = buffer.insert(
        ProblemSignature::compute(PROBLEM_A),
        "arithmetic: {problem}".to_string(),
        &config,
    );
    let id_c = buffer.insert(
        ProblemSignature::compute(PROBLEM_C),
        "causal: {problem}".to_string(),
        &config,
    );

    // Degrade id_a's success rate to 0.5 (1/2), well below id_c's 1.0.
    buffer.reinforce(id_a, false).expect("exists");

    let id_d = buffer.insert(
        ProblemSignature::compute(PROBLEM_D),
        "combinatorial: {problem}".to_string(),
        &config,
    );

    assert_eq!(buffer.len(), 2);
    assert!(
        buffer.get(id_a).is_none(),
        "lowest success_rate must be evicted"
    );
    assert!(buffer.get(id_c).is_some());
    assert!(buffer.get(id_d).is_some());
}

// ── BotEngine: solve() ────────────────────────────────────────────────────────

#[test]
fn solve_rejects_empty_problem() {
    let engine = default_engine();
    let err = engine.solve("   ").unwrap_err();
    assert!(matches!(err, BotError::EmptyProblem));
}

#[test]
fn solve_on_empty_buffer_has_no_match() {
    let engine = default_engine();
    let result = engine.solve(PROBLEM_A).expect("valid problem");
    assert!(result.matched_template_id.is_none());
    assert!(result.similarity.is_none());
}

#[test]
fn solve_on_empty_buffer_uses_default_scaffold() {
    let engine = default_engine();
    let result = engine.solve(PROBLEM_A).expect("valid problem");
    assert!(
        result
            .instantiated_scaffold
            .contains("Step 1: Read the problem carefully")
    );
    assert!(result.instantiated_scaffold.contains(PROBLEM_A));
}

#[test]
fn solve_does_not_mutate_buffer() {
    let engine = default_engine();
    let _ = engine.solve(PROBLEM_A).expect("valid problem");
    assert!(engine.buffer.is_empty());
}

// ── BotEngine: the grow-then-reuse lifecycle (core value proposition) ─────────

#[test]
fn end_to_end_grow_then_reuse() {
    let mut engine = default_engine();
    assert!(engine.buffer.is_empty());

    // Step 1: solve a fresh problem. Nothing is buffered, so BoT falls back
    // to the generic scaffold.
    let first = engine.solve(PROBLEM_A).expect("valid problem");
    assert!(first.matched_template_id.is_none());

    // Step 2: the caller judges the answer correct — distill a brand-new
    // template. The buffer grows from 0 to 1.
    let action = engine
        .record_outcome(&first, SolveOutcome::Success)
        .expect("distillation succeeds");
    let distilled_id = match action {
        DistillAction::Distilled(id) => id,
        other => panic!("expected Distilled, got {other:?}"),
    };
    assert_eq!(engine.buffer.len(), 1);
    let stored = engine.buffer.get(distilled_id).expect("just distilled");
    assert!(stored.template_text.contains("{problem}"));
    assert!(
        stored
            .signature
            .operation_tags
            .contains(&"arithmetic".to_string())
    );

    // Step 3: a *structurally similar but textually different* problem now
    // arrives. It must retrieve and reuse the freshly-distilled template
    // instead of falling back to the generic scaffold.
    let second = engine.solve(PROBLEM_B).expect("valid problem");
    assert_eq!(second.matched_template_id, Some(distilled_id));
    assert_eq!(second.similarity, Some(1.0));

    // The instantiated scaffold must reflect PROBLEM_B's own numbers, not
    // PROBLEM_A's — proving genuine per-problem instantiation, not verbatim
    // reuse of the first answer.
    assert!(second.instantiated_scaffold.contains("30"));
    assert!(second.instantiated_scaffold.contains('5'));
    assert!(!second.instantiated_scaffold.contains("12"));

    // Step 4: record success for the reused template — it is reinforced,
    // not re-distilled, so the buffer does not grow further.
    let action2 = engine
        .record_outcome(&second, SolveOutcome::Success)
        .expect("reinforcement succeeds");
    assert_eq!(action2, DistillAction::Reinforced(distilled_id));
    assert_eq!(engine.buffer.len(), 1);

    let reused = engine.buffer.get(distilled_id).expect("still present");
    assert_eq!(reused.usage_count, 2);
    assert_eq!(reused.success_count, 2);
    assert_eq!(reused.success_rate, 1.0);
}

#[test]
fn solve_and_record_matches_manual_solve_then_record() {
    let mut engine = default_engine();
    let (result, action) = engine
        .solve_and_record(PROBLEM_A, |_| SolveOutcome::Success)
        .expect("solves and records");
    assert!(result.matched_template_id.is_none());
    assert!(matches!(action, DistillAction::Distilled(_)));
    assert_eq!(engine.buffer.len(), 1);
}

#[test]
fn reinforcement_on_failure_lowers_success_rate_via_engine() {
    let mut engine = default_engine();
    let (_, action) = engine
        .solve_and_record(PROBLEM_A, |_| SolveOutcome::Success)
        .expect("first distills");
    let id = match action {
        DistillAction::Distilled(id) => id,
        other => panic!("expected Distilled, got {other:?}"),
    };

    let (_, action2) = engine
        .solve_and_record(PROBLEM_B, |_| SolveOutcome::Failure)
        .expect("second reuses and records failure");
    assert_eq!(action2, DistillAction::Reinforced(id));

    let template = engine.buffer.get(id).expect("present");
    assert_eq!(template.usage_count, 2);
    assert_eq!(template.success_count, 1);
    assert_eq!(template.success_rate, 0.5);
}

// ── BotEngine: no-match fallback honesty ──────────────────────────────────────

#[test]
fn no_match_fallback_reports_none_rather_than_forcing_a_bad_match() {
    let mut engine = default_engine();
    engine
        .solve_and_record(PROBLEM_A, |_| SolveOutcome::Success)
        .expect("distills an arithmetic template");
    assert_eq!(engine.buffer.len(), 1);

    // PROBLEM_C is structurally unrelated to the buffered arithmetic
    // template — retrieval must honestly report "no match".
    let result = engine.solve(PROBLEM_C).expect("valid problem");
    assert!(result.matched_template_id.is_none());
    assert!(result.similarity.is_none());
    assert!(
        result
            .instantiated_scaffold
            .contains("Step 1: Read the problem carefully")
    );
}

#[test]
fn unmatched_failed_solve_is_skipped_not_distilled() {
    let mut engine = default_engine();
    let (_, action) = engine
        .solve_and_record(PROBLEM_C, |_| SolveOutcome::Failure)
        .expect("records without erroring");
    assert_eq!(action, DistillAction::Skipped);
    assert!(engine.buffer.is_empty());
}

// ── BotEngine: eviction under max buffer size (end-to-end via the engine) ────

#[test]
fn engine_eviction_caps_buffer_at_max_size() {
    let mut engine = engine_with(
        BotConfig::default()
            .with_max_buffer_size(2)
            .with_eviction_policy(EvictionPolicy::Lru),
    );

    let (_, a1) = engine
        .solve_and_record(PROBLEM_A, |_| SolveOutcome::Success)
        .expect("distills");
    let (_, a2) = engine
        .solve_and_record(PROBLEM_C, |_| SolveOutcome::Success)
        .expect("distills");
    let (_, a3) = engine
        .solve_and_record(PROBLEM_D, |_| SolveOutcome::Success)
        .expect("distills");

    let id1 = match a1 {
        DistillAction::Distilled(id) => id,
        other => panic!("expected Distilled, got {other:?}"),
    };
    let id2 = match a2 {
        DistillAction::Distilled(id) => id,
        other => panic!("expected Distilled, got {other:?}"),
    };
    let id3 = match a3 {
        DistillAction::Distilled(id) => id,
        other => panic!("expected Distilled, got {other:?}"),
    };

    assert_eq!(engine.buffer.len(), 2);
    assert!(
        engine.buffer.get(id1).is_none(),
        "oldest template should be evicted"
    );
    assert!(engine.buffer.get(id2).is_some());
    assert!(engine.buffer.get(id3).is_some());
}

// ── BotEngine: stale matched-template id ──────────────────────────────────────

#[test]
fn record_outcome_errors_when_matched_template_was_evicted_meanwhile() {
    let mut engine = engine_with(BotConfig::default().with_max_buffer_size(1));

    let (first, action) = engine
        .solve_and_record(PROBLEM_A, |_| SolveOutcome::Success)
        .expect("distills");
    let stale_id = match action {
        DistillAction::Distilled(id) => id,
        other => panic!("expected Distilled, got {other:?}"),
    };

    // Re-solving the exact same problem retrieves the template we just
    // distilled; keep this stale solve result around without recording it
    // yet.
    let stale_result = engine
        .solve(PROBLEM_A)
        .expect("matches the buffered template");
    assert_eq!(stale_result.matched_template_id, Some(stale_id));
    assert_eq!(first.matched_template_id, None);

    // A completely different successful solve now evicts the only buffer
    // slot (max_buffer_size = 1) to make room for its own distilled
    // template.
    engine
        .solve_and_record(PROBLEM_C, |_| SolveOutcome::Success)
        .expect("distills, evicting the arithmetic template");
    assert!(engine.buffer.get(stale_id).is_none());

    // Recording the outcome for the now-stale result must fail loudly
    // instead of silently doing nothing.
    let err = engine
        .record_outcome(&stale_result, SolveOutcome::Success)
        .unwrap_err();
    assert!(matches!(err, BotError::TemplateNotFound(id) if id == stale_id));
}

// ── BotEngine: generator returning an empty distilled template ───────────────

struct EmptyDistillGenerator;

impl BotGenerator for EmptyDistillGenerator {
    fn generate(&self, problem: &str, instantiated_scaffold: &str) -> String {
        MockBotGenerator.generate(problem, instantiated_scaffold)
    }

    fn distill(&self, _problem: &str, _answer: &str, _signature: &ProblemSignature) -> String {
        "   ".to_string()
    }
}

#[test]
fn record_outcome_rejects_empty_distilled_template_text() {
    let mut engine = BotEngine::new(BotConfig::default(), EmptyDistillGenerator);
    let result = engine.solve(PROBLEM_A).expect("valid problem");
    let err = engine
        .record_outcome(&result, SolveOutcome::Success)
        .unwrap_err();
    assert!(matches!(err, BotError::EmptyTemplateText));
    assert!(engine.buffer.is_empty());
}

// ── BotEngine: Default ──────────────────────────────────────────────────────────

#[test]
fn engine_default_has_empty_buffer_and_default_config() {
    let engine = BotEngine::<MockBotGenerator>::default();
    assert!(engine.buffer.is_empty());
    assert_eq!(engine.config, BotConfig::default());
}

// ── SolveOutcome / DistillAction / BotError ───────────────────────────────────

#[test]
fn solve_outcome_success_and_failure_are_distinct() {
    assert_ne!(SolveOutcome::Success, SolveOutcome::Failure);
}

#[test]
fn distill_action_variants_compare_by_id() {
    assert_eq!(DistillAction::Distilled(1), DistillAction::Distilled(1));
    assert_ne!(DistillAction::Distilled(1), DistillAction::Distilled(2));
    assert_ne!(DistillAction::Distilled(1), DistillAction::Reinforced(1));
    assert_eq!(DistillAction::Skipped, DistillAction::Skipped);
}

#[test]
fn bot_error_messages_are_descriptive() {
    assert_eq!(
        BotError::EmptyProblem.to_string(),
        "problem must not be empty"
    );
    assert_eq!(
        BotError::EmptyTemplateText.to_string(),
        "distilled template text must not be empty"
    );
    assert_eq!(
        BotError::TemplateNotFound(42).to_string(),
        "no thought template found in the buffer with id 42"
    );
}

// ── EvictionPolicy ───────────────────────────────────────────────────────────────

#[test]
fn eviction_policy_default_is_lru() {
    assert_eq!(EvictionPolicy::default(), EvictionPolicy::Lru);
}
