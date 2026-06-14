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
//! Unit tests for the `self_discover` module.

use crate::self_discover::{
    engine::SelfDiscoverEngine,
    types::{
        MODULE_ANALOGY, MODULE_CAUSAL_REASONING, MODULE_COMPARATIVE_ANALYSIS,
        MODULE_CRITICAL_THINKING, MODULE_DECOMPOSITION, MODULE_DEDUCTIVE_REASONING,
        MODULE_HYPOTHESIS_TESTING, MODULE_STEP_BY_STEP, MockSelfDiscoverModel, ReasoningModule,
        ReasoningStructure, SelfDiscoverConfig, SelfDiscoverError, SelfDiscoverModel,
    },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sample_docs() -> Vec<String> {
    vec![
        "First document about reasoning.".to_string(),
        "Second document about logic.".to_string(),
        "Third document about evidence.".to_string(),
    ]
}

fn make_engine() -> SelfDiscoverEngine<MockSelfDiscoverModel> {
    SelfDiscoverEngine::new(SelfDiscoverConfig::default(), MockSelfDiscoverModel)
}

fn all_modules() -> Vec<ReasoningModule> {
    SelfDiscoverEngine::<MockSelfDiscoverModel>::built_in_modules()
}

// ── SelfDiscoverConfig ────────────────────────────────────────────────────────

#[test]
fn config_default_max_modules_is_three() {
    assert_eq!(SelfDiscoverConfig::default().max_modules, 3);
}

#[test]
fn config_default_top_k_docs_is_five() {
    assert_eq!(SelfDiscoverConfig::default().top_k_docs, 5);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SelfDiscoverConfig::new(), SelfDiscoverConfig::default());
}

#[test]
fn config_with_max_modules_sets_value() {
    assert_eq!(
        SelfDiscoverConfig::default()
            .with_max_modules(7)
            .max_modules,
        7
    );
}

#[test]
fn config_with_top_k_docs_sets_value() {
    assert_eq!(
        SelfDiscoverConfig::default().with_top_k_docs(10).top_k_docs,
        10
    );
}

#[test]
fn config_builder_chain_preserves_both_fields() {
    let cfg = SelfDiscoverConfig::new()
        .with_max_modules(2)
        .with_top_k_docs(8);
    assert_eq!(cfg.max_modules, 2);
    assert_eq!(cfg.top_k_docs, 8);
}

#[test]
fn config_can_be_cloned() {
    let cfg = SelfDiscoverConfig::default().with_max_modules(4);
    assert_eq!(cfg.clone(), cfg);
}

// ── ReasoningModule ───────────────────────────────────────────────────────────

#[test]
fn reasoning_module_new_sets_name() {
    assert_eq!(
        ReasoningModule::new("critical_thinking", "Evaluate.").name,
        "critical_thinking"
    );
}

#[test]
fn reasoning_module_new_sets_description() {
    assert_eq!(
        ReasoningModule::new("analogy", "Use analogies.").description,
        "Use analogies."
    );
}

#[test]
fn reasoning_module_clone_equals_original() {
    let m = ReasoningModule::new("step_by_step", "Break problems.");
    assert_eq!(m.clone(), m);
}

#[test]
fn reasoning_module_eq_same_fields() {
    let a = ReasoningModule::new("decomposition", "Partition.");
    let b = ReasoningModule::new("decomposition", "Partition.");
    assert_eq!(a, b);
}

#[test]
fn reasoning_module_ne_different_name() {
    assert_ne!(
        ReasoningModule::new("analogy", "Same."),
        ReasoningModule::new("deduction", "Same.")
    );
}

#[test]
fn reasoning_module_ne_different_description() {
    assert_ne!(
        ReasoningModule::new("analogy", "A."),
        ReasoningModule::new("analogy", "B.")
    );
}

// ── ReasoningStructure ────────────────────────────────────────────────────────

#[test]
fn reasoning_structure_default_is_empty() {
    let rs = ReasoningStructure::default();
    assert!(rs.modules.is_empty() && rs.plan.is_empty());
}

#[test]
fn reasoning_structure_can_be_cloned() {
    let rs = ReasoningStructure {
        modules: vec![ReasoningModule::new("analogy", "desc")],
        plan: vec!["Step 1".to_string()],
    };
    assert_eq!(rs.clone(), rs);
}

#[test]
fn reasoning_structure_eq() {
    let a = ReasoningStructure {
        modules: vec![],
        plan: vec!["A".to_string()],
    };
    let b = ReasoningStructure {
        modules: vec![],
        plan: vec!["A".to_string()],
    };
    assert_eq!(a, b);
}

// ── SelfDiscoverError ─────────────────────────────────────────────────────────

#[test]
fn error_selection_failed_display() {
    let err = SelfDiscoverError::SelectionFailed("no modules".to_string());
    assert!(err.to_string().contains("selection failed"));
    assert!(err.to_string().contains("no modules"));
}

#[test]
fn error_adaptation_failed_display() {
    let err = SelfDiscoverError::AdaptationFailed("bad".to_string());
    assert!(err.to_string().contains("adaptation failed"));
    assert!(err.to_string().contains("bad"));
}

#[test]
fn error_implementation_failed_display() {
    let err = SelfDiscoverError::ImplementationFailed("crash".to_string());
    assert!(err.to_string().contains("implementation failed"));
    assert!(err.to_string().contains("crash"));
}

#[test]
fn error_selection_failed_is_debug() {
    let err = SelfDiscoverError::SelectionFailed("x".to_string());
    assert!(format!("{err:?}").contains("SelectionFailed"));
}

// ── Built-in module bank ──────────────────────────────────────────────────────

#[test]
fn built_in_modules_count_is_eight() {
    assert_eq!(all_modules().len(), 8);
}

#[test]
fn built_in_modules_contains_critical_thinking() {
    assert!(
        all_modules()
            .iter()
            .any(|m| m.name == MODULE_CRITICAL_THINKING)
    );
}

#[test]
fn built_in_modules_contains_deductive_reasoning() {
    assert!(
        all_modules()
            .iter()
            .any(|m| m.name == MODULE_DEDUCTIVE_REASONING)
    );
}

#[test]
fn built_in_modules_contains_analogy() {
    assert!(all_modules().iter().any(|m| m.name == MODULE_ANALOGY));
}

#[test]
fn built_in_modules_contains_causal_reasoning() {
    assert!(
        all_modules()
            .iter()
            .any(|m| m.name == MODULE_CAUSAL_REASONING)
    );
}

#[test]
fn built_in_modules_contains_step_by_step() {
    assert!(all_modules().iter().any(|m| m.name == MODULE_STEP_BY_STEP));
}

#[test]
fn built_in_modules_contains_comparative_analysis() {
    assert!(
        all_modules()
            .iter()
            .any(|m| m.name == MODULE_COMPARATIVE_ANALYSIS)
    );
}

#[test]
fn built_in_modules_contains_hypothesis_testing() {
    assert!(
        all_modules()
            .iter()
            .any(|m| m.name == MODULE_HYPOTHESIS_TESTING)
    );
}

#[test]
fn built_in_modules_contains_decomposition() {
    assert!(all_modules().iter().any(|m| m.name == MODULE_DECOMPOSITION));
}

#[test]
fn built_in_modules_all_have_nonempty_descriptions() {
    assert!(all_modules().iter().all(|m| !m.description.is_empty()));
}

// ── MockSelfDiscoverModel::select ─────────────────────────────────────────────

#[test]
fn mock_select_returns_ok() {
    assert!(MockSelfDiscoverModel.select("test", &all_modules()).is_ok());
}

#[test]
fn mock_select_returns_all_module_indices() {
    let modules = all_modules();
    let indices = MockSelfDiscoverModel.select("test", &modules).unwrap();
    assert_eq!(indices.len(), modules.len());
}

#[test]
fn mock_select_indices_all_within_bounds() {
    let modules = all_modules();
    let indices = MockSelfDiscoverModel.select("logic", &modules).unwrap();
    assert!(indices.iter().all(|&i| i < modules.len()));
}

#[test]
fn mock_select_empty_query_tie_broken_by_original_order() {
    let modules = all_modules();
    let indices = MockSelfDiscoverModel.select("", &modules).unwrap();
    assert_eq!(indices[0], 0);
    assert_eq!(indices[1], 1);
    assert_eq!(indices[2], 2);
}

#[test]
fn mock_select_empty_modules_returns_empty() {
    assert!(
        MockSelfDiscoverModel
            .select("some query", &[])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mock_select_no_duplicate_indices() {
    let modules = all_modules();
    let indices = MockSelfDiscoverModel.select("reasoning", &modules).unwrap();
    let unique: std::collections::HashSet<usize> = indices.iter().copied().collect();
    assert_eq!(unique.len(), indices.len());
}

#[test]
fn mock_select_single_module_bank_returns_zero() {
    let modules = vec![ReasoningModule::new("single", "Only one.")];
    assert_eq!(
        MockSelfDiscoverModel.select("only", &modules).unwrap(),
        vec![0]
    );
}

#[test]
fn mock_select_decomposition_ranked_first_by_overlap() {
    // "partition", "components", "independently", "integrate" all appear in
    // the decomposition description exclusively among the 8 modules.
    let query = "partition components independently integrate";
    let modules = all_modules();
    let indices = MockSelfDiscoverModel.select(query, &modules).unwrap();
    let decomp = modules
        .iter()
        .position(|m| m.name == MODULE_DECOMPOSITION)
        .unwrap();
    assert_eq!(indices[0], decomp);
}

#[test]
fn mock_select_hypothesis_testing_ranked_first_by_overlap() {
    // "formulate", "testable", "hypotheses", "beliefs" concentrate in
    // hypothesis_testing.
    let query = "formulate testable hypotheses evidence beliefs";
    let modules = all_modules();
    let indices = MockSelfDiscoverModel.select(query, &modules).unwrap();
    let hypo = modules
        .iter()
        .position(|m| m.name == MODULE_HYPOTHESIS_TESTING)
        .unwrap();
    assert_eq!(indices[0], hypo);
}

#[test]
fn mock_select_causal_reasoning_ranked_first_by_overlap() {
    // "cause", "effect", "causation", "correlation" are unique to
    // causal_reasoning.
    let query = "identify cause effect causation correlation";
    let modules = all_modules();
    let indices = MockSelfDiscoverModel.select(query, &modules).unwrap();
    let causal = modules
        .iter()
        .position(|m| m.name == MODULE_CAUSAL_REASONING)
        .unwrap();
    assert_eq!(indices[0], causal);
}

// ── MockSelfDiscoverModel::adapt_and_implement ────────────────────────────────

#[test]
fn mock_adapt_plan_has_one_step_per_module() {
    let modules = vec![
        ReasoningModule::new("a", "A."),
        ReasoningModule::new("b", "B."),
    ];
    let (structure, _) = MockSelfDiscoverModel
        .adapt_and_implement("q", &modules, &[])
        .unwrap();
    assert_eq!(structure.plan.len(), 2);
}

#[test]
fn mock_adapt_plan_includes_module_names() {
    let modules = vec![ReasoningModule::new("critical_thinking", "Evaluate.")];
    let (structure, _) = MockSelfDiscoverModel
        .adapt_and_implement("test", &modules, &[])
        .unwrap();
    assert!(structure.plan[0].contains("critical_thinking"));
}

#[test]
fn mock_adapt_answer_joins_docs_with_space() {
    let modules = vec![ReasoningModule::new("x", "y")];
    let docs = vec!["doc1".to_string(), "doc2".to_string()];
    let (_, answer) = MockSelfDiscoverModel
        .adapt_and_implement("q", &modules, &docs)
        .unwrap();
    assert_eq!(answer, "doc1 doc2");
}

#[test]
fn mock_adapt_empty_docs_answer_is_empty() {
    let modules = vec![ReasoningModule::new("x", "y")];
    let (_, answer) = MockSelfDiscoverModel
        .adapt_and_implement("q", &modules, &[])
        .unwrap();
    assert!(answer.is_empty());
}

#[test]
fn mock_adapt_structure_modules_match_input() {
    let modules = vec![
        ReasoningModule::new("analogy", "A."),
        ReasoningModule::new("decomposition", "D."),
    ];
    let (structure, _) = MockSelfDiscoverModel
        .adapt_and_implement("q", &modules, &[])
        .unwrap();
    assert_eq!(structure.modules, modules);
}

#[test]
fn mock_adapt_single_doc_answer_equals_doc() {
    let modules = vec![ReasoningModule::new("x", "y")];
    let (_, answer) = MockSelfDiscoverModel
        .adapt_and_implement("q", &modules, &["only doc".to_string()])
        .unwrap();
    assert_eq!(answer, "only doc");
}

// ── SelfDiscoverEngine::run ───────────────────────────────────────────────────

#[test]
fn engine_run_returns_correct_query_field() {
    let result = make_engine()
        .run("test query here", &sample_docs())
        .unwrap();
    assert_eq!(result.query, "test query here");
}

#[test]
fn engine_run_selected_modules_not_empty() {
    assert!(
        !make_engine()
            .run("test", &sample_docs())
            .unwrap()
            .selected_modules
            .is_empty()
    );
}

#[test]
fn engine_run_reasoning_structure_plan_not_empty() {
    assert!(
        !make_engine()
            .run("query", &sample_docs())
            .unwrap()
            .reasoning_structure
            .plan
            .is_empty()
    );
}

#[test]
fn engine_run_answer_contains_doc_content() {
    let docs = vec!["unique_xyz_content".to_string()];
    assert!(
        make_engine()
            .run("query", &docs)
            .unwrap()
            .answer
            .contains("unique_xyz_content")
    );
}

#[test]
fn engine_run_max_modules_one_selects_exactly_one() {
    let engine = SelfDiscoverEngine::new(
        SelfDiscoverConfig::default().with_max_modules(1),
        MockSelfDiscoverModel,
    );
    assert_eq!(
        engine
            .run("query", &sample_docs())
            .unwrap()
            .selected_modules
            .len(),
        1
    );
}

#[test]
fn engine_run_max_modules_five_selects_exactly_five() {
    let engine = SelfDiscoverEngine::new(
        SelfDiscoverConfig::default().with_max_modules(5),
        MockSelfDiscoverModel,
    );
    assert_eq!(
        engine
            .run("query", &sample_docs())
            .unwrap()
            .selected_modules
            .len(),
        5
    );
}

#[test]
fn engine_run_max_modules_larger_than_bank_caps_at_bank_size() {
    let engine = SelfDiscoverEngine::new(
        SelfDiscoverConfig::default().with_max_modules(100),
        MockSelfDiscoverModel,
    );
    assert_eq!(
        engine
            .run("query", &sample_docs())
            .unwrap()
            .selected_modules
            .len(),
        8
    );
}

#[test]
fn engine_run_empty_docs_returns_ok() {
    assert!(make_engine().run("test query", &[]).is_ok());
}

#[test]
fn engine_run_single_doc_answer_equals_doc() {
    let result = make_engine()
        .run("query", &["only one doc".to_string()])
        .unwrap();
    assert_eq!(result.answer, "only one doc");
}

#[test]
fn engine_run_structure_modules_match_selected() {
    let result = make_engine()
        .run("decompose logic", &sample_docs())
        .unwrap();
    assert_eq!(result.reasoning_structure.modules, result.selected_modules);
}

#[test]
fn engine_run_plan_len_equals_selected_len() {
    let result = make_engine().run("query", &sample_docs()).unwrap();
    assert_eq!(
        result.reasoning_structure.plan.len(),
        result.selected_modules.len()
    );
}

// ── top_k_docs ────────────────────────────────────────────────────────────────

#[test]
fn engine_top_k_docs_limits_docs_forwarded() {
    let engine = SelfDiscoverEngine::new(
        SelfDiscoverConfig::default().with_top_k_docs(2),
        MockSelfDiscoverModel,
    );
    let docs: Vec<String> = (0..10).map(|i| format!("doc{i}")).collect();
    assert_eq!(engine.run("query", &docs).unwrap().answer, "doc0 doc1");
}

#[test]
fn engine_top_k_docs_fewer_than_limit_uses_all() {
    let engine = SelfDiscoverEngine::new(
        SelfDiscoverConfig::default().with_top_k_docs(10),
        MockSelfDiscoverModel,
    );
    let docs = vec!["a".to_string(), "b".to_string()];
    assert_eq!(engine.run("query", &docs).unwrap().answer, "a b");
}

#[test]
fn engine_top_k_docs_exact_limit_uses_all() {
    let engine = SelfDiscoverEngine::new(
        SelfDiscoverConfig::default().with_top_k_docs(3),
        MockSelfDiscoverModel,
    );
    let docs: Vec<String> = (0..3).map(|i| format!("doc{i}")).collect();
    assert_eq!(engine.run("query", &docs).unwrap().answer, "doc0 doc1 doc2");
}

// ── Error propagation ─────────────────────────────────────────────────────────

struct FailingSelectModel;

impl SelfDiscoverModel for FailingSelectModel {
    fn select(
        &self,
        _query: &str,
        _modules: &[ReasoningModule],
    ) -> Result<Vec<usize>, SelfDiscoverError> {
        Err(SelfDiscoverError::SelectionFailed(
            "intentional".to_string(),
        ))
    }
    fn adapt_and_implement(
        &self,
        _query: &str,
        _modules: &[ReasoningModule],
        _docs: &[String],
    ) -> Result<(ReasoningStructure, String), SelfDiscoverError> {
        Ok((ReasoningStructure::default(), String::new()))
    }
}

struct FailingAdaptModel;

impl SelfDiscoverModel for FailingAdaptModel {
    fn select(
        &self,
        _query: &str,
        modules: &[ReasoningModule],
    ) -> Result<Vec<usize>, SelfDiscoverError> {
        Ok((0..modules.len().min(3)).collect())
    }
    fn adapt_and_implement(
        &self,
        _query: &str,
        _modules: &[ReasoningModule],
        _docs: &[String],
    ) -> Result<(ReasoningStructure, String), SelfDiscoverError> {
        Err(SelfDiscoverError::AdaptationFailed(
            "intentional".to_string(),
        ))
    }
}

#[test]
fn error_select_propagated_from_engine() {
    let engine = SelfDiscoverEngine::new(SelfDiscoverConfig::default(), FailingSelectModel);
    assert!(matches!(
        engine.run("q", &[]).unwrap_err(),
        SelfDiscoverError::SelectionFailed(_)
    ));
}

#[test]
fn error_select_message_preserved() {
    let engine = SelfDiscoverEngine::new(SelfDiscoverConfig::default(), FailingSelectModel);
    assert!(
        engine
            .run("q", &[])
            .unwrap_err()
            .to_string()
            .contains("intentional")
    );
}

#[test]
fn error_adapt_propagated_from_engine() {
    let engine = SelfDiscoverEngine::new(SelfDiscoverConfig::default(), FailingAdaptModel);
    assert!(matches!(
        engine.run("q", &[]).unwrap_err(),
        SelfDiscoverError::AdaptationFailed(_)
    ));
}

#[test]
fn error_adapt_message_preserved() {
    let engine = SelfDiscoverEngine::new(SelfDiscoverConfig::default(), FailingAdaptModel);
    assert!(
        engine
            .run("q", &[])
            .unwrap_err()
            .to_string()
            .contains("intentional")
    );
}

// ── Module-name constants ─────────────────────────────────────────────────────

#[test]
fn module_constant_critical_thinking() {
    assert_eq!(MODULE_CRITICAL_THINKING, "critical_thinking");
}

#[test]
fn module_constant_deductive_reasoning() {
    assert_eq!(MODULE_DEDUCTIVE_REASONING, "deductive_reasoning");
}

#[test]
fn module_constant_analogy() {
    assert_eq!(MODULE_ANALOGY, "analogy");
}

#[test]
fn module_constant_causal_reasoning() {
    assert_eq!(MODULE_CAUSAL_REASONING, "causal_reasoning");
}

#[test]
fn module_constant_step_by_step() {
    assert_eq!(MODULE_STEP_BY_STEP, "step_by_step");
}

#[test]
fn module_constant_hypothesis_testing() {
    assert_eq!(MODULE_HYPOTHESIS_TESTING, "hypothesis_testing");
}

#[test]
fn module_constant_decomposition() {
    assert_eq!(MODULE_DECOMPOSITION, "decomposition");
}

// ── SelfDiscoverResult fields ─────────────────────────────────────────────────

#[test]
fn result_query_field_preserved() {
    let q = "How do we reason about causation?";
    assert_eq!(make_engine().run(q, &sample_docs()).unwrap().query, q);
}

#[test]
fn result_selected_modules_within_max() {
    assert!(
        make_engine()
            .run("q", &sample_docs())
            .unwrap()
            .selected_modules
            .len()
            <= 3
    );
}

#[test]
fn result_answer_contains_all_doc_tokens() {
    let docs = vec!["alpha".to_string(), "beta".to_string()];
    let result = make_engine().run("q", &docs).unwrap();
    assert!(result.answer.contains("alpha") && result.answer.contains("beta"));
}

#[test]
fn result_plan_steps_contain_module_name() {
    let result = make_engine().run("query", &sample_docs()).unwrap();
    for (step, module) in result
        .reasoning_structure
        .plan
        .iter()
        .zip(result.selected_modules.iter())
    {
        assert!(step.contains(&module.name));
    }
}

#[test]
fn result_can_be_cloned() {
    let result = make_engine().run("query", &sample_docs()).unwrap();
    let cloned = result.clone();
    assert_eq!(result.query, cloned.query);
    assert_eq!(result.answer, cloned.answer);
}
