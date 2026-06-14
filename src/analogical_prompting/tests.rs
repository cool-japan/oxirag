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
//! Unit tests for the `analogical_prompting` module.

use std::cell::RefCell;

use crate::analogical_prompting::engine::AnalogicalEngine;
use crate::analogical_prompting::types::{
    AnalogicalConfig, AnalogicalError, AnalogicalModel, Exemplar, MockAnalogicalModel,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn sample_exemplars() -> Vec<Exemplar> {
    vec![
        Exemplar::new("What is 2 + 3?", "2 + 3 = 5"),
        Exemplar::new("What is 10 + 4?", "10 + 4 = 14"),
        Exemplar::new("What is 1 + 1?", "1 + 1 = 2"),
        Exemplar::new("What is 6 + 6?", "6 + 6 = 12"),
    ]
}

fn sample_model() -> MockAnalogicalModel {
    MockAnalogicalModel::new(
        sample_exemplars(),
        "Addition combines two numbers into their sum.",
        "7 + 8 = 15",
    )
}

/// A recording model that captures exactly what `solve` was handed by the engine.
struct RecordingModel {
    exemplars: Vec<Exemplar>,
    knowledge: String,
    answer: String,
    seen_exemplars: RefCell<Vec<Exemplar>>,
    seen_knowledge: RefCell<String>,
    seen_problem: RefCell<String>,
    exemplar_requests: RefCell<Vec<usize>>,
    knowledge_calls: RefCell<usize>,
}

impl RecordingModel {
    fn new(
        exemplars: Vec<Exemplar>,
        knowledge: impl Into<String>,
        answer: impl Into<String>,
    ) -> Self {
        Self {
            exemplars,
            knowledge: knowledge.into(),
            answer: answer.into(),
            seen_exemplars: RefCell::new(Vec::new()),
            seen_knowledge: RefCell::new(String::new()),
            seen_problem: RefCell::new(String::new()),
            exemplar_requests: RefCell::new(Vec::new()),
            knowledge_calls: RefCell::new(0),
        }
    }
}

impl AnalogicalModel for RecordingModel {
    fn generate_exemplars(&self, _problem: &str, n: usize) -> Vec<Exemplar> {
        self.exemplar_requests.borrow_mut().push(n);
        self.exemplars.iter().take(n).cloned().collect()
    }

    fn generate_knowledge(&self, _problem: &str) -> String {
        *self.knowledge_calls.borrow_mut() += 1;
        self.knowledge.clone()
    }

    fn solve(&self, problem: &str, exemplars: &[Exemplar], knowledge: &str) -> String {
        *self.seen_problem.borrow_mut() = problem.to_string();
        *self.seen_exemplars.borrow_mut() = exemplars.to_vec();
        *self.seen_knowledge.borrow_mut() = knowledge.to_string();
        self.answer.clone()
    }
}

// ── AnalogicalConfig defaults ───────────────────────────────────────────────

#[test]
fn config_default_num_exemplars_is_three() {
    assert_eq!(AnalogicalConfig::default().num_exemplars, 3);
}

#[test]
fn config_default_use_knowledge_is_true() {
    assert!(AnalogicalConfig::default().use_knowledge);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(AnalogicalConfig::new(), AnalogicalConfig::default());
}

// ── AnalogicalConfig builders ───────────────────────────────────────────────

#[test]
fn config_builder_sets_num_exemplars() {
    assert_eq!(
        AnalogicalConfig::new().with_num_exemplars(5).num_exemplars,
        5
    );
}

#[test]
fn config_builder_zero_num_exemplars() {
    assert_eq!(
        AnalogicalConfig::new().with_num_exemplars(0).num_exemplars,
        0
    );
}

#[test]
fn config_builder_sets_use_knowledge_false() {
    assert!(
        !AnalogicalConfig::new()
            .with_use_knowledge(false)
            .use_knowledge
    );
}

#[test]
fn config_builder_sets_use_knowledge_true() {
    assert!(
        AnalogicalConfig::new()
            .with_use_knowledge(false)
            .with_use_knowledge(true)
            .use_knowledge
    );
}

#[test]
fn config_builders_chain() {
    let cfg = AnalogicalConfig::new()
        .with_num_exemplars(7)
        .with_use_knowledge(false);
    assert_eq!(cfg.num_exemplars, 7);
    assert!(!cfg.use_knowledge);
}

#[test]
fn config_builder_preserves_other_field_num() {
    let cfg = AnalogicalConfig::new().with_num_exemplars(9);
    assert!(cfg.use_knowledge);
}

#[test]
fn config_builder_preserves_other_field_knowledge() {
    let cfg = AnalogicalConfig::new().with_use_knowledge(false);
    assert_eq!(cfg.num_exemplars, 3);
}

#[test]
fn config_clone_equals_original() {
    let cfg = AnalogicalConfig::new().with_num_exemplars(4);
    assert_eq!(cfg.clone(), cfg);
}

#[test]
fn config_debug_is_non_empty() {
    assert!(!format!("{:?}", AnalogicalConfig::default()).is_empty());
}

// ── Exemplar ──────────────────────────────────────────────────────────────────

#[test]
fn exemplar_new_sets_problem() {
    assert_eq!(Exemplar::new("p", "s").problem, "p");
}

#[test]
fn exemplar_new_sets_solution() {
    assert_eq!(Exemplar::new("p", "s").solution, "s");
}

#[test]
fn exemplar_clone_equals_original() {
    let e = Exemplar::new("p", "s");
    assert_eq!(e.clone(), e);
}

#[test]
fn exemplar_inequality() {
    assert_ne!(Exemplar::new("a", "b"), Exemplar::new("a", "c"));
}

// ── MockAnalogicalModel ───────────────────────────────────────────────────────

#[test]
fn mock_generate_exemplars_takes_n() {
    let model = sample_model();
    assert_eq!(model.generate_exemplars("q", 2).len(), 2);
}

#[test]
fn mock_generate_exemplars_caps_at_available() {
    let model = sample_model();
    // Asking for more than scripted yields all four scripted exemplars.
    assert_eq!(model.generate_exemplars("q", 99).len(), 4);
}

#[test]
fn mock_generate_exemplars_zero() {
    let model = sample_model();
    assert!(model.generate_exemplars("q", 0).is_empty());
}

#[test]
fn mock_generate_exemplars_preserves_order() {
    let model = sample_model();
    let exemplars = model.generate_exemplars("q", 2);
    assert_eq!(exemplars[0].problem, "What is 2 + 3?");
    assert_eq!(exemplars[1].problem, "What is 10 + 4?");
}

#[test]
fn mock_generate_knowledge_returns_scripted() {
    let model = sample_model();
    assert_eq!(
        model.generate_knowledge("q"),
        "Addition combines two numbers into their sum."
    );
}

#[test]
fn mock_solve_returns_scripted_answer() {
    let model = sample_model();
    assert_eq!(model.solve("q", &[], ""), "7 + 8 = 15");
}

#[test]
fn mock_solve_ignores_arguments() {
    let model = sample_model();
    let with_args = model.solve("p", &sample_exemplars(), "k");
    let without_args = model.solve("", &[], "");
    assert_eq!(with_args, without_args);
}

#[test]
fn mock_clone_equals_original() {
    let model = sample_model();
    assert_eq!(model.clone(), model);
}

// ── AnalogicalEngine construction ─────────────────────────────────────────────

#[test]
fn engine_new_stores_config() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(2));
    assert_eq!(engine.config.num_exemplars, 2);
}

#[test]
fn engine_default_uses_default_config() {
    let engine = AnalogicalEngine::default();
    assert_eq!(engine.config, AnalogicalConfig::default());
}

#[test]
fn engine_clone_equals_original() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(2));
    assert_eq!(engine.clone().config, engine.config);
}

// ── run: exemplar generation ──────────────────────────────────────────────────

#[test]
fn run_generates_up_to_num_exemplars() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(2));
    let output = engine.run("What is 7 + 8?", &sample_model()).unwrap();
    assert_eq!(output.exemplars.len(), 2);
}

#[test]
fn run_exemplars_never_exceed_num_exemplars() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(1));
    let output = engine.run("What is 7 + 8?", &sample_model()).unwrap();
    assert!(output.exemplars.len() <= 1);
}

#[test]
fn run_caps_exemplars_when_model_overproduces() {
    // Even if the model ignores `n` and returns everything, the engine truncates.
    struct Greedy(Vec<Exemplar>);
    impl AnalogicalModel for Greedy {
        fn generate_exemplars(&self, _problem: &str, _n: usize) -> Vec<Exemplar> {
            self.0.clone()
        }
        fn generate_knowledge(&self, _problem: &str) -> String {
            String::new()
        }
        fn solve(&self, _problem: &str, _exemplars: &[Exemplar], _knowledge: &str) -> String {
            "ok".to_string()
        }
    }
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(2));
    let model = Greedy(sample_exemplars());
    let output = engine.run("problem", &model).unwrap();
    assert_eq!(output.exemplars.len(), 2);
}

#[test]
fn run_requests_configured_exemplar_count() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(2));
    let model = RecordingModel::new(sample_exemplars(), "k", "a");
    engine.run("problem", &model).unwrap();
    assert_eq!(model.exemplar_requests.borrow().as_slice(), &[2]);
}

#[test]
fn run_zero_exemplars_yields_empty() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(0));
    let output = engine.run("problem", &sample_model()).unwrap();
    assert!(output.exemplars.is_empty());
}

#[test]
fn run_preserves_exemplar_order() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(3));
    let output = engine.run("problem", &sample_model()).unwrap();
    assert_eq!(output.exemplars[0].problem, "What is 2 + 3?");
    assert_eq!(output.exemplars[2].problem, "What is 1 + 1?");
}

// ── run: knowledge handling ───────────────────────────────────────────────────

#[test]
fn run_use_knowledge_true_populates_knowledge() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let output = engine.run("problem", &sample_model()).unwrap();
    assert_eq!(
        output.knowledge,
        "Addition combines two numbers into their sum."
    );
}

#[test]
fn run_use_knowledge_false_yields_empty_knowledge() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_use_knowledge(false));
    let output = engine.run("problem", &sample_model()).unwrap();
    assert!(output.knowledge.is_empty());
}

#[test]
fn run_use_knowledge_false_skips_generate_knowledge() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_use_knowledge(false));
    let model = RecordingModel::new(sample_exemplars(), "k", "a");
    engine.run("problem", &model).unwrap();
    assert_eq!(*model.knowledge_calls.borrow(), 0);
}

#[test]
fn run_use_knowledge_true_calls_generate_knowledge_once() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let model = RecordingModel::new(sample_exemplars(), "k", "a");
    engine.run("problem", &model).unwrap();
    assert_eq!(*model.knowledge_calls.borrow(), 1);
}

// ── run: solve receives generated context ─────────────────────────────────────

#[test]
fn run_solve_receives_generated_exemplars() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(2));
    let model = RecordingModel::new(sample_exemplars(), "k", "a");
    let output = engine.run("problem", &model).unwrap();
    // solve must have been handed exactly the exemplars exposed in the output.
    assert_eq!(*model.seen_exemplars.borrow(), output.exemplars);
    assert_eq!(model.seen_exemplars.borrow().len(), 2);
}

#[test]
fn run_solve_receives_truncated_exemplars() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(1));
    let model = RecordingModel::new(sample_exemplars(), "k", "a");
    engine.run("problem", &model).unwrap();
    assert_eq!(model.seen_exemplars.borrow().len(), 1);
    assert_eq!(model.seen_exemplars.borrow()[0].problem, "What is 2 + 3?");
}

#[test]
fn run_solve_receives_knowledge_when_enabled() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let model = RecordingModel::new(sample_exemplars(), "deep knowledge", "a");
    engine.run("problem", &model).unwrap();
    assert_eq!(*model.seen_knowledge.borrow(), "deep knowledge");
}

#[test]
fn run_solve_receives_empty_knowledge_when_disabled() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_use_knowledge(false));
    let model = RecordingModel::new(sample_exemplars(), "deep knowledge", "a");
    engine.run("problem", &model).unwrap();
    assert_eq!(*model.seen_knowledge.borrow(), "");
}

#[test]
fn run_solve_receives_original_problem() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let model = RecordingModel::new(sample_exemplars(), "k", "a");
    engine.run("the target problem", &model).unwrap();
    assert_eq!(*model.seen_problem.borrow(), "the target problem");
}

// ── run: answer ───────────────────────────────────────────────────────────────

#[test]
fn run_answer_is_non_empty() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let output = engine.run("problem", &sample_model()).unwrap();
    assert!(!output.answer.is_empty());
}

#[test]
fn run_answer_matches_model_output() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let output = engine.run("problem", &sample_model()).unwrap();
    assert_eq!(output.answer, "7 + 8 = 15");
}

#[test]
fn run_answer_uses_solve_result_not_knowledge() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let model = MockAnalogicalModel::new(sample_exemplars(), "knowledge text", "answer text");
    let output = engine.run("problem", &model).unwrap();
    assert_eq!(output.answer, "answer text");
    assert_ne!(output.answer, output.knowledge);
}

// ── AnalogicalOutput wiring ───────────────────────────────────────────────────

#[test]
fn output_fields_all_wired() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(2));
    let output = engine.run("problem", &sample_model()).unwrap();
    assert_eq!(output.exemplars.len(), 2);
    assert!(!output.knowledge.is_empty());
    assert!(!output.answer.is_empty());
}

#[test]
fn output_clone_equals_original() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let output = engine.run("problem", &sample_model()).unwrap();
    assert_eq!(output.clone(), output);
}

#[test]
fn output_exemplar_contents_match_model() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(1));
    let output = engine.run("problem", &sample_model()).unwrap();
    assert_eq!(output.exemplars[0].solution, "2 + 3 = 5");
}

// ── run: empty-problem error ──────────────────────────────────────────────────

#[test]
fn run_empty_problem_errors() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let err = engine.run("", &sample_model()).unwrap_err();
    assert!(matches!(err, AnalogicalError::EmptyProblem));
}

#[test]
fn run_whitespace_problem_errors() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let err = engine.run("   \t\n  ", &sample_model()).unwrap_err();
    assert!(matches!(err, AnalogicalError::EmptyProblem));
}

#[test]
fn run_empty_problem_does_not_call_model() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let model = RecordingModel::new(sample_exemplars(), "k", "a");
    let _ = engine.run("", &model);
    assert!(model.exemplar_requests.borrow().is_empty());
    assert_eq!(*model.knowledge_calls.borrow(), 0);
}

#[test]
fn error_display_message() {
    assert_eq!(
        AnalogicalError::EmptyProblem.to_string(),
        "problem must not be empty"
    );
}

#[test]
fn error_debug_is_non_empty() {
    assert!(!format!("{:?}", AnalogicalError::EmptyProblem).is_empty());
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn run_is_deterministic_with_mock() {
    let engine = AnalogicalEngine::new(AnalogicalConfig::default());
    let first = engine.run("problem", &sample_model()).unwrap();
    let second = engine.run("problem", &sample_model()).unwrap();
    assert_eq!(first, second);
}

#[test]
fn run_deterministic_across_engine_instances() {
    let model = sample_model();
    let a = AnalogicalEngine::new(AnalogicalConfig::default())
        .run("problem", &model)
        .unwrap();
    let b = AnalogicalEngine::new(AnalogicalConfig::default())
        .run("problem", &model)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn run_via_dyn_trait_object() {
    // The generic bound accepts `?Sized`, so a trait object works too.
    let engine = AnalogicalEngine::new(AnalogicalConfig::new().with_num_exemplars(2));
    let model: Box<dyn AnalogicalModel> = Box::new(sample_model());
    let output = engine.run("problem", model.as_ref()).unwrap();
    assert_eq!(output.exemplars.len(), 2);
    assert_eq!(output.answer, "7 + 8 = 15");
}
