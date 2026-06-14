//! Unit tests for the `self_ask` module.

use crate::self_ask::engine::SelfAskEngine;
use crate::self_ask::types::{
    FollowUp, MockSelfAskModel, MockSubAnswerer, SelfAskConfig, SelfAskError, SelfAskModel,
    SubAnswerer,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn engine_default() -> SelfAskEngine {
    SelfAskEngine::new(SelfAskConfig::default())
}

// ── SelfAskConfig ───────────────────────────────────────────────────────────

#[test]
fn config_default_max_follow_ups_is_five() {
    assert_eq!(SelfAskConfig::default().max_follow_ups, 5);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SelfAskConfig::new(), SelfAskConfig::default());
}

#[test]
fn config_builder_sets_max_follow_ups() {
    assert_eq!(
        SelfAskConfig::new().with_max_follow_ups(3).max_follow_ups,
        3
    );
}

#[test]
fn config_builder_zero_max_follow_ups() {
    assert_eq!(
        SelfAskConfig::new().with_max_follow_ups(0).max_follow_ups,
        0
    );
}

#[test]
fn config_clone_equals_original() {
    let cfg = SelfAskConfig::new().with_max_follow_ups(7);
    assert_eq!(cfg.clone(), cfg);
}

// ── FollowUp ────────────────────────────────────────────────────────────────

#[test]
fn follow_up_new_sets_question() {
    assert_eq!(FollowUp::new("q", "a").question, "q");
}

#[test]
fn follow_up_new_sets_answer() {
    assert_eq!(FollowUp::new("q", "a").answer, "a");
}

#[test]
fn follow_up_equality() {
    assert_eq!(FollowUp::new("q", "a"), FollowUp::new("q", "a"));
}

// ── MockSelfAskModel ─────────────────────────────────────────────────────────

#[test]
fn mock_model_immediate_has_no_follow_ups() {
    assert!(MockSelfAskModel::immediate("done").follow_ups.is_empty());
}

#[test]
fn mock_model_immediate_returns_none_at_start() {
    let model = MockSelfAskModel::immediate("done");
    assert_eq!(model.next_follow_up("q", &[]), None);
}

#[test]
fn mock_model_returns_first_scripted_question() {
    let model = MockSelfAskModel::new(vec!["f1".to_string()], "final");
    assert_eq!(model.next_follow_up("q", &[]), Some("f1".to_string()));
}

#[test]
fn mock_model_returns_question_by_history_index() {
    let model = MockSelfAskModel::new(vec!["f1".to_string(), "f2".to_string()], "final");
    let history = vec![FollowUp::new("f1", "a1")];
    assert_eq!(model.next_follow_up("q", &history), Some("f2".to_string()));
}

#[test]
fn mock_model_exhausted_returns_none() {
    let model = MockSelfAskModel::new(vec!["f1".to_string()], "final");
    let history = vec![FollowUp::new("f1", "a1")];
    assert_eq!(model.next_follow_up("q", &history), None);
}

#[test]
fn mock_model_compose_final_returns_scripted_answer() {
    let model = MockSelfAskModel::immediate("the answer");
    assert_eq!(model.compose_final("q", &[]), "the answer");
}

#[test]
fn mock_model_compose_final_ignores_history() {
    let model = MockSelfAskModel::new(vec!["f1".to_string()], "fixed");
    let history = vec![FollowUp::new("f1", "a1")];
    assert_eq!(model.compose_final("q", &history), "fixed");
}

// ── MockSubAnswerer ──────────────────────────────────────────────────────────

#[test]
fn mock_answerer_echo_returns_input() {
    assert_eq!(MockSubAnswerer::echo().answer("hello"), "hello");
}

#[test]
fn mock_answerer_default_echoes() {
    assert_eq!(MockSubAnswerer::default().answer("xyz"), "xyz");
}

#[test]
fn mock_answerer_matches_substring() {
    let answerer = MockSubAnswerer::new(vec![("born".to_string(), "1970".to_string())]);
    assert_eq!(answerer.answer("When was he born?"), "1970");
}

#[test]
fn mock_answerer_no_match_echoes() {
    let answerer = MockSubAnswerer::new(vec![("born".to_string(), "1970".to_string())]);
    assert_eq!(answerer.answer("Who directed it?"), "Who directed it?");
}

#[test]
fn mock_answerer_first_match_wins() {
    let answerer = MockSubAnswerer::new(vec![
        ("o".to_string(), "first".to_string()),
        ("born".to_string(), "second".to_string()),
    ]);
    assert_eq!(answerer.answer("born"), "first");
}

// ── run: validation ──────────────────────────────────────────────────────────

#[test]
fn run_empty_question_errors() {
    let model = MockSelfAskModel::immediate("x");
    let answerer = MockSubAnswerer::echo();
    let err = engine_default().run("", &model, &answerer).unwrap_err();
    assert!(matches!(err, SelfAskError::EmptyQuestion));
}

#[test]
fn run_whitespace_question_errors() {
    let model = MockSelfAskModel::immediate("x");
    let answerer = MockSubAnswerer::echo();
    let err = engine_default().run("   ", &model, &answerer).unwrap_err();
    assert!(matches!(err, SelfAskError::EmptyQuestion));
}

#[test]
fn run_error_display_message() {
    let model = MockSelfAskModel::immediate("x");
    let answerer = MockSubAnswerer::echo();
    let err = engine_default().run("", &model, &answerer).unwrap_err();
    assert_eq!(err.to_string(), "question must not be empty");
}

// ── run: zero follow-up case ─────────────────────────────────────────────────

#[test]
fn run_zero_follow_ups_has_no_hops() {
    let model = MockSelfAskModel::immediate("final");
    let answerer = MockSubAnswerer::echo();
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    assert_eq!(trace.num_hops, 0);
}

#[test]
fn run_zero_follow_ups_empty_history() {
    let model = MockSelfAskModel::immediate("final");
    let answerer = MockSubAnswerer::echo();
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    assert!(trace.follow_ups.is_empty());
}

#[test]
fn run_zero_follow_ups_uses_compose_final() {
    let model = MockSelfAskModel::immediate("final answer");
    let answerer = MockSubAnswerer::echo();
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    assert_eq!(trace.final_answer, "final answer");
}

// ── run: single follow-up chain ──────────────────────────────────────────────

#[test]
fn run_single_follow_up_one_hop() {
    let model = MockSelfAskModel::new(vec!["who?".to_string()], "final");
    let answerer = MockSubAnswerer::new(vec![("who".to_string(), "Alice".to_string())]);
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    assert_eq!(trace.num_hops, 1);
}

#[test]
fn run_single_follow_up_records_question() {
    let model = MockSelfAskModel::new(vec!["who?".to_string()], "final");
    let answerer = MockSubAnswerer::new(vec![("who".to_string(), "Alice".to_string())]);
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    assert_eq!(trace.follow_ups[0].question, "who?");
}

#[test]
fn run_single_follow_up_records_answer() {
    let model = MockSelfAskModel::new(vec!["who?".to_string()], "final");
    let answerer = MockSubAnswerer::new(vec![("who".to_string(), "Alice".to_string())]);
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    assert_eq!(trace.follow_ups[0].answer, "Alice");
}

// ── run: multi-hop (3 follow-ups) ────────────────────────────────────────────

fn three_hop_model() -> MockSelfAskModel {
    MockSelfAskModel::new(
        vec![
            "step one?".to_string(),
            "step two?".to_string(),
            "step three?".to_string(),
        ],
        "final composed",
    )
}

fn three_hop_answerer() -> MockSubAnswerer {
    MockSubAnswerer::new(vec![
        ("one".to_string(), "A1".to_string()),
        ("two".to_string(), "A2".to_string()),
        ("three".to_string(), "A3".to_string()),
    ])
}

#[test]
fn run_multi_hop_has_three_hops() {
    let trace = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    assert_eq!(trace.num_hops, 3);
}

#[test]
fn run_multi_hop_records_three_follow_ups() {
    let trace = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    assert_eq!(trace.follow_ups.len(), 3);
}

#[test]
fn run_multi_hop_questions_in_order() {
    let trace = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    let questions: Vec<&str> = trace
        .follow_ups
        .iter()
        .map(|f| f.question.as_str())
        .collect();
    assert_eq!(questions, vec!["step one?", "step two?", "step three?"]);
}

#[test]
fn run_multi_hop_answers_in_order() {
    let trace = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    let answers: Vec<&str> = trace.follow_ups.iter().map(|f| f.answer.as_str()).collect();
    assert_eq!(answers, vec!["A1", "A2", "A3"]);
}

#[test]
fn run_multi_hop_first_answer_correct() {
    let trace = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    assert_eq!(trace.follow_ups[0].answer, "A1");
}

#[test]
fn run_multi_hop_last_answer_correct() {
    let trace = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    assert_eq!(trace.follow_ups[2].answer, "A3");
}

#[test]
fn run_multi_hop_final_answer() {
    let trace = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    assert_eq!(trace.final_answer, "final composed");
}

// ── run: max_follow_ups cap ──────────────────────────────────────────────────

/// Model that always wants another follow-up (never returns `None`).
struct UnboundedModel;

impl SelfAskModel for UnboundedModel {
    fn next_follow_up(&self, _question: &str, history: &[FollowUp]) -> Option<String> {
        Some(format!("follow up {}", history.len()))
    }

    fn compose_final(&self, _question: &str, history: &[FollowUp]) -> String {
        format!("stopped after {}", history.len())
    }
}

#[test]
fn run_cap_halts_at_max_follow_ups() {
    let cfg = SelfAskConfig::new().with_max_follow_ups(4);
    let trace = SelfAskEngine::new(cfg)
        .run("q", &UnboundedModel, &MockSubAnswerer::echo())
        .unwrap();
    assert_eq!(trace.num_hops, 4);
}

#[test]
fn run_cap_records_exactly_max_follow_ups() {
    let cfg = SelfAskConfig::new().with_max_follow_ups(2);
    let trace = SelfAskEngine::new(cfg)
        .run("q", &UnboundedModel, &MockSubAnswerer::echo())
        .unwrap();
    assert_eq!(trace.follow_ups.len(), 2);
}

#[test]
fn run_cap_zero_yields_no_hops() {
    let cfg = SelfAskConfig::new().with_max_follow_ups(0);
    let trace = SelfAskEngine::new(cfg)
        .run("q", &UnboundedModel, &MockSubAnswerer::echo())
        .unwrap();
    assert_eq!(trace.num_hops, 0);
}

#[test]
fn run_cap_zero_uses_compose_final() {
    let cfg = SelfAskConfig::new().with_max_follow_ups(0);
    let trace = SelfAskEngine::new(cfg)
        .run("q", &UnboundedModel, &MockSubAnswerer::echo())
        .unwrap();
    assert_eq!(trace.final_answer, "stopped after 0");
}

#[test]
fn run_cap_one_allows_single_hop() {
    let cfg = SelfAskConfig::new().with_max_follow_ups(1);
    let trace = SelfAskEngine::new(cfg)
        .run("q", &UnboundedModel, &MockSubAnswerer::echo())
        .unwrap();
    assert_eq!(trace.num_hops, 1);
}

// ── run: SubAnswerer wiring ───────────────────────────────────────────────────

#[test]
fn run_subanswerer_answer_wired_into_follow_up() {
    let model = MockSelfAskModel::new(vec!["capital of France?".to_string()], "Paris");
    let answerer = MockSubAnswerer::new(vec![("France".to_string(), "Paris".to_string())]);
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    assert_eq!(trace.follow_ups[0].answer, "Paris");
}

#[test]
fn run_subanswerer_echo_wires_question_as_answer() {
    let model = MockSelfAskModel::new(vec!["echo me".to_string()], "final");
    let answerer = MockSubAnswerer::echo();
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    assert_eq!(trace.follow_ups[0].answer, "echo me");
}

// ── run: final answer == compose_final ───────────────────────────────────────

#[test]
fn run_final_answer_equals_compose_final_output() {
    let model = MockSelfAskModel::new(vec!["f1".to_string()], "composed output");
    let answerer = MockSubAnswerer::echo();
    let trace = engine_default()
        .run("a real question", &model, &answerer)
        .unwrap();
    let expected = model.compose_final("a real question", &trace.follow_ups);
    assert_eq!(trace.final_answer, expected);
}

// ── run: num_hops invariant ──────────────────────────────────────────────────

#[test]
fn run_num_hops_equals_follow_ups_len() {
    let trace = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    assert_eq!(trace.num_hops, trace.follow_ups.len());
}

// ── run: determinism ─────────────────────────────────────────────────────────

#[test]
fn run_is_deterministic() {
    let a = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    let b = engine_default()
        .run("q", &three_hop_model(), &three_hop_answerer())
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn run_determinism_zero_hops() {
    let model = MockSelfAskModel::immediate("same");
    let answerer = MockSubAnswerer::echo();
    let a = engine_default().run("q", &model, &answerer).unwrap();
    let b = engine_default().run("q", &model, &answerer).unwrap();
    assert_eq!(a, b);
}

// ── MockSelfAskModel consumes scripted questions in order (via run) ───────────

#[test]
fn run_consumes_scripted_questions_in_order() {
    let model = MockSelfAskModel::new(vec!["first".to_string(), "second".to_string()], "final");
    let answerer = MockSubAnswerer::echo();
    let trace = engine_default().run("q", &model, &answerer).unwrap();
    let questions: Vec<&str> = trace
        .follow_ups
        .iter()
        .map(|f| f.question.as_str())
        .collect();
    assert_eq!(questions, vec!["first", "second"]);
}

#[test]
fn run_stops_when_script_exhausted_before_cap() {
    // Two scripted follow-ups but a generous cap: should stop at 2.
    let cfg = SelfAskConfig::new().with_max_follow_ups(10);
    let model = MockSelfAskModel::new(vec!["a".to_string(), "b".to_string()], "final");
    let trace = SelfAskEngine::new(cfg)
        .run("q", &model, &MockSubAnswerer::echo())
        .unwrap();
    assert_eq!(trace.num_hops, 2);
}

// ── engine construction ──────────────────────────────────────────────────────

#[test]
fn engine_default_uses_default_config() {
    assert_eq!(SelfAskEngine::default().config, SelfAskConfig::default());
}

#[test]
fn engine_new_stores_config() {
    let cfg = SelfAskConfig::new().with_max_follow_ups(9);
    assert_eq!(SelfAskEngine::new(cfg.clone()).config, cfg);
}

// ── trait objects (dyn) compile and run ──────────────────────────────────────

#[test]
fn run_accepts_trait_objects() {
    let model: &dyn SelfAskModel = &MockSelfAskModel::immediate("dyn final");
    let answerer: &dyn SubAnswerer = &MockSubAnswerer::echo();
    let trace = engine_default().run("q", model, answerer).unwrap();
    assert_eq!(trace.final_answer, "dyn final");
}
