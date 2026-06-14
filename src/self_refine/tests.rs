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
//! Unit tests for the `self_refine` module.

use crate::self_refine::engine::SelfRefineEngine;
use crate::self_refine::types::{
    Feedback, MockRefiner, RefineStep, Refiner, SelfRefineConfig, SelfRefineError, SelfRefineOutput,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn engine_default() -> SelfRefineEngine {
    SelfRefineEngine::new(SelfRefineConfig::default())
}

/// A refiner whose feedback never stops and never reaches threshold, refining
/// to `"refined-N"` where N counts how many times `refine` was called (encoded
/// in the output suffix so the mock stays stateless).
struct NeverStopsRefiner;

impl Refiner for NeverStopsRefiner {
    fn initial(&self, _task: &str) -> String {
        "draft-0".to_string()
    }

    fn feedback(&self, _task: &str, _output: &str) -> Feedback {
        // Low score, never stops.
        Feedback::new("keep going", 0.1, false)
    }

    fn refine(&self, _task: &str, output: &str, _feedback: &Feedback) -> String {
        // Parse the trailing number and increment it.
        let n: usize = output
            .rsplit('-')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        format!("draft-{}", n + 1)
    }
}

// ── Feedback ──────────────────────────────────────────────────────────────────

#[test]
fn feedback_new_sets_critique() {
    assert_eq!(Feedback::new("c", 0.5, false).critique, "c");
}

#[test]
fn feedback_new_sets_score() {
    assert_eq!(Feedback::new("c", 0.5, false).score, 0.5);
}

#[test]
fn feedback_new_sets_stop() {
    assert!(Feedback::new("c", 0.5, true).stop);
}

#[test]
fn feedback_clone_equals_original() {
    let fb = Feedback::new("critique", 0.7, true);
    assert_eq!(fb.clone(), fb);
}

// ── SelfRefineConfig: defaults ─────────────────────────────────────────────────

#[test]
fn config_default_max_iterations_is_four() {
    assert_eq!(SelfRefineConfig::default().max_iterations, 4);
}

#[test]
fn config_default_score_threshold_is_point_nine() {
    assert_eq!(SelfRefineConfig::default().score_threshold, 0.9);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SelfRefineConfig::new(), SelfRefineConfig::default());
}

// ── SelfRefineConfig: builders ─────────────────────────────────────────────────

#[test]
fn config_builder_sets_max_iterations() {
    assert_eq!(
        SelfRefineConfig::new()
            .with_max_iterations(7)
            .max_iterations,
        7
    );
}

#[test]
fn config_builder_zero_max_iterations() {
    assert_eq!(
        SelfRefineConfig::new()
            .with_max_iterations(0)
            .max_iterations,
        0
    );
}

#[test]
fn config_builder_sets_score_threshold() {
    assert_eq!(
        SelfRefineConfig::new()
            .with_score_threshold(0.75)
            .score_threshold,
        0.75
    );
}

#[test]
fn config_builder_chains_both() {
    let cfg = SelfRefineConfig::new()
        .with_max_iterations(2)
        .with_score_threshold(0.6);
    assert_eq!(cfg.max_iterations, 2);
    assert_eq!(cfg.score_threshold, 0.6);
}

#[test]
fn config_builder_preserves_other_field_on_max_iterations() {
    let cfg = SelfRefineConfig::new().with_max_iterations(3);
    assert_eq!(cfg.score_threshold, 0.9);
}

#[test]
fn config_builder_preserves_other_field_on_threshold() {
    let cfg = SelfRefineConfig::new().with_score_threshold(0.5);
    assert_eq!(cfg.max_iterations, 4);
}

#[test]
fn config_clone_equals_original() {
    let cfg = SelfRefineConfig::new().with_max_iterations(9);
    assert_eq!(cfg.clone(), cfg);
}

// ── MockRefiner: construction ──────────────────────────────────────────────────

#[test]
fn mock_new_sets_initial() {
    let r = MockRefiner::new("start", vec![]);
    assert_eq!(r.initial, "start");
}

#[test]
fn mock_new_sets_steps_len() {
    let r = MockRefiner::new(
        "start",
        vec![(Feedback::new("c", 0.5, false), "x".to_string())],
    );
    assert_eq!(r.steps.len(), 1);
}

#[test]
fn mock_good_enough_initial() {
    let r = MockRefiner::good_enough("answer", 0.95);
    assert_eq!(r.initial, "answer");
}

#[test]
fn mock_good_enough_feedback_stops() {
    let r = MockRefiner::good_enough("answer", 0.95);
    assert!(r.feedback("task", "answer").stop);
}

#[test]
fn mock_good_enough_feedback_score() {
    let r = MockRefiner::good_enough("answer", 0.92);
    assert_eq!(r.feedback("task", "answer").score, 0.92);
}

// ── MockRefiner: trait behavior ────────────────────────────────────────────────

#[test]
fn mock_initial_returns_configured() {
    let r = MockRefiner::new("the start", vec![]);
    assert_eq!(r.initial("task"), "the start");
}

#[test]
fn mock_feedback_for_initial_uses_first_step() {
    let r = MockRefiner::new(
        "v0",
        vec![(Feedback::new("improve", 0.3, false), "v1".to_string())],
    );
    assert_eq!(r.feedback("task", "v0").critique, "improve");
}

#[test]
fn mock_refine_for_initial_uses_first_refinement() {
    let r = MockRefiner::new(
        "v0",
        vec![(Feedback::new("improve", 0.3, false), "v1".to_string())],
    );
    let fb = r.feedback("task", "v0");
    assert_eq!(r.refine("task", "v0", &fb), "v1");
}

#[test]
fn mock_feedback_for_refinement_uses_next_step() {
    let r = MockRefiner::new(
        "v0",
        vec![
            (Feedback::new("c0", 0.3, false), "v1".to_string()),
            (Feedback::new("c1", 0.95, true), "v2".to_string()),
        ],
    );
    assert_eq!(r.feedback("task", "v1").critique, "c1");
}

#[test]
fn mock_feedback_past_script_is_terminal_stop() {
    let r = MockRefiner::new("v0", vec![]);
    let fb = r.feedback("task", "unknown output");
    assert!(fb.stop);
}

#[test]
fn mock_feedback_past_script_is_terminal_score_one() {
    let r = MockRefiner::new("v0", vec![]);
    let fb = r.feedback("task", "unknown output");
    assert_eq!(fb.score, 1.0);
}

#[test]
fn mock_refine_past_script_echoes_output() {
    let r = MockRefiner::new("v0", vec![]);
    let fb = Feedback::new("c", 0.5, false);
    assert_eq!(r.refine("task", "anything", &fb), "anything");
}

#[test]
fn mock_clone_equals_original() {
    let r = MockRefiner::new("x", vec![(Feedback::new("c", 0.5, false), "y".to_string())]);
    assert_eq!(r.clone(), r);
}

// ── run: validation ────────────────────────────────────────────────────────────

#[test]
fn run_empty_task_errors() {
    let r = MockRefiner::good_enough("x", 1.0);
    let err = engine_default().run("", &r).unwrap_err();
    assert!(matches!(err, SelfRefineError::EmptyTask));
}

#[test]
fn run_whitespace_task_errors() {
    let r = MockRefiner::good_enough("x", 1.0);
    let err = engine_default().run("   \t\n", &r).unwrap_err();
    assert!(matches!(err, SelfRefineError::EmptyTask));
}

#[test]
fn run_error_display_message() {
    let r = MockRefiner::good_enough("x", 1.0);
    let err = engine_default().run("", &r).unwrap_err();
    assert_eq!(err.to_string(), "task must not be empty");
}

// ── run: initial output produced ───────────────────────────────────────────────

#[test]
fn run_records_initial_as_first_step_output() {
    let r = MockRefiner::good_enough("my initial output", 1.0);
    let out = engine_default().run("task", &r).unwrap();
    assert_eq!(out.steps[0].output, "my initial output");
}

#[test]
fn run_initial_good_enough_single_iteration() {
    let r = MockRefiner::good_enough("done", 1.0);
    let out = engine_default().run("task", &r).unwrap();
    assert_eq!(out.iterations, 1);
}

#[test]
fn run_initial_good_enough_final_is_initial() {
    let r = MockRefiner::good_enough("done", 1.0);
    let out = engine_default().run("task", &r).unwrap();
    assert_eq!(out.final_output, "done");
}

// ── run: feedback → refine loop improves output ────────────────────────────────

fn improving_refiner() -> MockRefiner {
    MockRefiner::new(
        "draft",
        vec![
            (
                Feedback::new("too short", 0.3, false),
                "better draft".to_string(),
            ),
            (Feedback::new("good now", 0.95, true), String::new()),
        ],
    )
}

#[test]
fn run_loop_improves_output() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.final_output, "better draft");
}

#[test]
fn run_loop_does_two_iterations() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.iterations, 2);
}

#[test]
fn run_loop_first_step_is_initial() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.steps[0].output, "draft");
}

#[test]
fn run_loop_second_step_is_refinement() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.steps[1].output, "better draft");
}

#[test]
fn run_loop_first_step_feedback_critique() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.steps[0].feedback.critique, "too short");
}

#[test]
fn run_loop_second_step_feedback_critique() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.steps[1].feedback.critique, "good now");
}

// ── run: stop on score_threshold ───────────────────────────────────────────────

#[test]
fn run_stops_on_score_threshold_default() {
    // Second feedback has score 0.9 (== threshold) but stop == false.
    let r = MockRefiner::new(
        "v0",
        vec![
            (Feedback::new("c0", 0.4, false), "v1".to_string()),
            (Feedback::new("c1", 0.9, false), "v2".to_string()),
        ],
    );
    let out = engine_default().run("task", &r).unwrap();
    assert_eq!(out.iterations, 2);
}

#[test]
fn run_threshold_does_not_refine_further() {
    let r = MockRefiner::new(
        "v0",
        vec![
            (Feedback::new("c0", 0.4, false), "v1".to_string()),
            (Feedback::new("c1", 0.9, false), "v2".to_string()),
        ],
    );
    let out = engine_default().run("task", &r).unwrap();
    // v2 never produced because threshold met at v1's feedback.
    assert_eq!(out.final_output, "v1");
}

#[test]
fn run_custom_threshold_stops_early() {
    // Threshold lowered to 0.5: first feedback at 0.6 already stops.
    let cfg = SelfRefineConfig::new().with_score_threshold(0.5);
    let r = MockRefiner::new(
        "v0",
        vec![(Feedback::new("ok", 0.6, false), "v1".to_string())],
    );
    let out = SelfRefineEngine::new(cfg).run("task", &r).unwrap();
    assert_eq!(out.iterations, 1);
}

#[test]
fn run_score_just_below_threshold_continues() {
    // 0.89 < 0.9 → continues to refine.
    let r = MockRefiner::new(
        "v0",
        vec![
            (Feedback::new("c0", 0.89, false), "v1".to_string()),
            (Feedback::new("c1", 1.0, false), "v2".to_string()),
        ],
    );
    let out = engine_default().run("task", &r).unwrap();
    assert_eq!(out.iterations, 2);
}

#[test]
fn run_score_exactly_threshold_stops() {
    let cfg = SelfRefineConfig::new().with_score_threshold(0.8);
    let r = MockRefiner::new(
        "v0",
        vec![(Feedback::new("c0", 0.8, false), "v1".to_string())],
    );
    let out = SelfRefineEngine::new(cfg).run("task", &r).unwrap();
    assert_eq!(out.iterations, 1);
}

// ── run: stop on feedback.stop ─────────────────────────────────────────────────

#[test]
fn run_stops_on_feedback_stop_flag_low_score() {
    // stop == true even though score is low.
    let r = MockRefiner::new(
        "v0",
        vec![(Feedback::new("done anyway", 0.1, true), "v1".to_string())],
    );
    let out = engine_default().run("task", &r).unwrap();
    assert_eq!(out.iterations, 1);
}

#[test]
fn run_stop_flag_keeps_current_output() {
    let r = MockRefiner::new(
        "v0",
        vec![(Feedback::new("done", 0.1, true), "v1".to_string())],
    );
    let out = engine_default().run("task", &r).unwrap();
    assert_eq!(out.final_output, "v0");
}

#[test]
fn run_stop_flag_on_second_iteration() {
    let r = MockRefiner::new(
        "v0",
        vec![
            (Feedback::new("c0", 0.2, false), "v1".to_string()),
            (Feedback::new("c1", 0.2, true), "v2".to_string()),
        ],
    );
    let out = engine_default().run("task", &r).unwrap();
    assert_eq!(out.iterations, 2);
    assert_eq!(out.final_output, "v1");
}

// ── run: max_iterations cap ────────────────────────────────────────────────────

#[test]
fn run_cap_halts_at_max_iterations() {
    let cfg = SelfRefineConfig::new().with_max_iterations(3);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    assert_eq!(out.iterations, 3);
}

#[test]
fn run_cap_records_exactly_max_steps() {
    let cfg = SelfRefineConfig::new().with_max_iterations(5);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    assert_eq!(out.steps.len(), 5);
}

#[test]
fn run_cap_default_is_four() {
    // Default config, never-stopping refiner → exactly 4 iterations.
    let out = engine_default().run("task", &NeverStopsRefiner).unwrap();
    assert_eq!(out.iterations, 4);
}

#[test]
fn run_cap_zero_yields_no_iterations() {
    let cfg = SelfRefineConfig::new().with_max_iterations(0);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    assert_eq!(out.iterations, 0);
}

#[test]
fn run_cap_zero_final_output_is_initial() {
    let cfg = SelfRefineConfig::new().with_max_iterations(0);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &MockRefiner::good_enough("the-initial", 1.0))
        .unwrap();
    assert_eq!(out.final_output, "the-initial");
}

#[test]
fn run_cap_zero_has_empty_steps() {
    let cfg = SelfRefineConfig::new().with_max_iterations(0);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    assert!(out.steps.is_empty());
}

#[test]
fn run_cap_one_single_step() {
    let cfg = SelfRefineConfig::new().with_max_iterations(1);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    assert_eq!(out.iterations, 1);
}

#[test]
fn run_cap_progresses_outputs_when_never_stopping() {
    // With 3 iterations the recorded outputs should be draft-0, draft-1, draft-2.
    let cfg = SelfRefineConfig::new().with_max_iterations(3);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    let outputs: Vec<&str> = out.steps.iter().map(|s| s.output.as_str()).collect();
    assert_eq!(outputs, vec!["draft-0", "draft-1", "draft-2"]);
}

// ── run: steps recorded with feedback ──────────────────────────────────────────

#[test]
fn run_steps_pair_output_with_feedback() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    let step: &RefineStep = &out.steps[0];
    assert_eq!(step.output, "draft");
    assert_eq!(step.feedback.critique, "too short");
}

#[test]
fn run_step_feedback_score_recorded() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.steps[0].feedback.score, 0.3);
}

#[test]
fn run_step_feedback_stop_recorded() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert!(!out.steps[0].feedback.stop);
    assert!(out.steps[1].feedback.stop);
}

// ── run: final_output == last step output ──────────────────────────────────────

#[test]
fn run_final_output_equals_last_step_output() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    let last = out.steps.last().unwrap();
    assert_eq!(out.final_output, last.output);
}

#[test]
fn run_final_output_equals_last_step_when_capped() {
    let cfg = SelfRefineConfig::new().with_max_iterations(3);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    let last = out.steps.last().unwrap();
    assert_eq!(out.final_output, last.output);
}

// ── run: iterations count correct ──────────────────────────────────────────────

#[test]
fn run_iterations_equals_steps_len() {
    let out = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.iterations, out.steps.len());
}

#[test]
fn run_iterations_equals_steps_len_capped() {
    let cfg = SelfRefineConfig::new().with_max_iterations(2);
    let out = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    assert_eq!(out.iterations, out.steps.len());
}

// ── run: SelfRefineOutput shape ────────────────────────────────────────────────

#[test]
fn run_output_clone_equals_original() {
    let out: SelfRefineOutput = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(out.clone(), out);
}

// ── run: determinism ───────────────────────────────────────────────────────────

#[test]
fn run_is_deterministic() {
    let a = engine_default().run("task", &improving_refiner()).unwrap();
    let b = engine_default().run("task", &improving_refiner()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn run_is_deterministic_when_capped() {
    let cfg = SelfRefineConfig::new().with_max_iterations(4);
    let a = SelfRefineEngine::new(cfg.clone())
        .run("task", &NeverStopsRefiner)
        .unwrap();
    let b = SelfRefineEngine::new(cfg)
        .run("task", &NeverStopsRefiner)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn run_deterministic_good_enough() {
    let r = MockRefiner::good_enough("x", 0.99);
    let a = engine_default().run("task", &r).unwrap();
    let b = engine_default().run("task", &r).unwrap();
    assert_eq!(a, b);
}

// ── engine construction ────────────────────────────────────────────────────────

#[test]
fn engine_default_uses_default_config() {
    assert_eq!(
        SelfRefineEngine::default().config,
        SelfRefineConfig::default()
    );
}

#[test]
fn engine_new_stores_config() {
    let cfg = SelfRefineConfig::new().with_max_iterations(9);
    assert_eq!(SelfRefineEngine::new(cfg.clone()).config, cfg);
}

#[test]
fn engine_clone_equals_original() {
    let e = SelfRefineEngine::new(SelfRefineConfig::new().with_max_iterations(2));
    assert_eq!(e.clone().config, e.config);
}

// ── trait objects (dyn) compile and run ────────────────────────────────────────

#[test]
fn run_accepts_trait_object() {
    let refiner: &dyn Refiner = &MockRefiner::good_enough("dyn final", 1.0);
    let out = engine_default().run("task", refiner).unwrap();
    assert_eq!(out.final_output, "dyn final");
}
