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
    clippy::items_after_statements,
    clippy::uninlined_format_args
)]
//! Unit tests for the `least_to_most` module.

use crate::least_to_most::engine::LtmEngine;
use crate::least_to_most::types::{
    LtmConfig, LtmError, LtmSolver, MockLtmSolver, SubProblem, tokenize,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_engine(config: LtmConfig) -> LtmEngine {
    LtmEngine::new(config, Box::new(MockLtmSolver))
}

fn default_engine() -> LtmEngine {
    make_engine(LtmConfig::default())
}

fn sample_docs() -> Vec<String> {
    vec![
        "Paris is the capital of France.".to_string(),
        "France is a country in Europe.".to_string(),
        "Europe has many countries.".to_string(),
        "The Eiffel Tower is in Paris.".to_string(),
        "Rust is a systems programming language.".to_string(),
    ]
}

// ── LtmConfig defaults ────────────────────────────────────────────────────────

#[test]
fn config_default_max_subproblems_is_five() {
    assert_eq!(LtmConfig::default().max_subproblems, 5);
}

#[test]
fn config_default_max_context_docs_is_three() {
    assert_eq!(LtmConfig::default().max_context_docs, 3);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(LtmConfig::new(), LtmConfig::default());
}

#[test]
fn config_builder_sets_max_subproblems() {
    assert_eq!(
        LtmConfig::new().with_max_subproblems(10).max_subproblems,
        10
    );
}

#[test]
fn config_builder_sets_max_context_docs() {
    assert_eq!(
        LtmConfig::new().with_max_context_docs(7).max_context_docs,
        7
    );
}

#[test]
fn config_builder_chain() {
    let cfg = LtmConfig::new()
        .with_max_subproblems(2)
        .with_max_context_docs(1);
    assert_eq!(cfg.max_subproblems, 2);
    assert_eq!(cfg.max_context_docs, 1);
}

#[test]
fn config_builder_preserves_other_field_subproblems() {
    let cfg = LtmConfig::new().with_max_subproblems(8);
    assert_eq!(cfg.max_context_docs, 3);
}

#[test]
fn config_builder_preserves_other_field_docs() {
    let cfg = LtmConfig::new().with_max_context_docs(1);
    assert_eq!(cfg.max_subproblems, 5);
}

#[test]
fn config_clone_equals_original() {
    let cfg = LtmConfig::new().with_max_subproblems(4);
    assert_eq!(cfg.clone(), cfg);
}

#[test]
fn config_debug_is_non_empty() {
    assert!(!format!("{:?}", LtmConfig::default()).is_empty());
}

// ── SubProblem ────────────────────────────────────────────────────────────────

#[test]
fn subproblem_new_stores_question() {
    let sp = SubProblem::new("q", "a", 0);
    assert_eq!(sp.question, "q");
}

#[test]
fn subproblem_new_stores_answer() {
    let sp = SubProblem::new("q", "a", 0);
    assert_eq!(sp.answer, "a");
}

#[test]
fn subproblem_new_stores_step_index() {
    let sp = SubProblem::new("q", "a", 7);
    assert_eq!(sp.step_index, 7);
}

#[test]
fn subproblem_clone_equals_original() {
    let sp = SubProblem::new("q", "a", 1);
    assert_eq!(sp.clone(), sp);
}

#[test]
fn subproblem_inequality_answer() {
    assert_ne!(SubProblem::new("q", "a", 0), SubProblem::new("q", "b", 0));
}

#[test]
fn subproblem_inequality_step_index() {
    assert_ne!(SubProblem::new("q", "a", 0), SubProblem::new("q", "a", 1));
}

#[test]
fn subproblem_debug_is_non_empty() {
    assert!(!format!("{:?}", SubProblem::new("q", "a", 0)).is_empty());
}

// ── MockLtmSolver: decompose ──────────────────────────────────────────────────

#[test]
fn mock_decompose_single_question_no_conjunction() {
    let s = MockLtmSolver;
    let parts = s.decompose("What is the capital of France?").unwrap();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0], "What is the capital of France?");
}

#[test]
fn mock_decompose_splits_on_and() {
    let s = MockLtmSolver;
    let parts = s.decompose("alpha and beta").unwrap();
    assert_eq!(parts, vec!["alpha", "beta"]);
}

#[test]
fn mock_decompose_splits_on_then() {
    let s = MockLtmSolver;
    let parts = s.decompose("step one then step two").unwrap();
    assert_eq!(parts, vec!["step one", "step two"]);
}

#[test]
fn mock_decompose_splits_on_also() {
    let s = MockLtmSolver;
    let parts = s.decompose("part A also part B").unwrap();
    assert_eq!(parts, vec!["part A", "part B"]);
}

#[test]
fn mock_decompose_case_insensitive_and() {
    let s = MockLtmSolver;
    let parts = s.decompose("A AND B").unwrap();
    assert_eq!(parts, vec!["A", "B"]);
}

#[test]
fn mock_decompose_case_insensitive_then() {
    let s = MockLtmSolver;
    let parts = s.decompose("first THEN second").unwrap();
    assert_eq!(parts, vec!["first", "second"]);
}

#[test]
fn mock_decompose_multiple_conjunctions() {
    let s = MockLtmSolver;
    let parts = s.decompose("a and b then c").unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], "a");
    assert_eq!(parts[1], "b");
    assert_eq!(parts[2], "c");
}

#[test]
fn mock_decompose_no_conjunction_returns_original_query() {
    let s = MockLtmSolver;
    let q = "What is quantum computing?";
    let parts = s.decompose(q).unwrap();
    assert_eq!(parts, vec![q]);
}

#[test]
fn mock_decompose_trims_segments() {
    let s = MockLtmSolver;
    let parts = s.decompose("  foo  and   bar  ").unwrap();
    for p in &parts {
        assert_eq!(p.trim(), p.as_str(), "segment should be trimmed: {p:?}");
    }
}

// ── MockLtmSolver: solve_step ─────────────────────────────────────────────────

#[test]
fn mock_solve_step_no_docs_no_prior() {
    let s = MockLtmSolver;
    let ans = s.solve_step("q", &[], &[]).unwrap();
    assert_eq!(ans, "Answer: q");
}

#[test]
fn mock_solve_step_contains_question() {
    let s = MockLtmSolver;
    let ans = s.solve_step("What is 2+2?", &[], &[]).unwrap();
    assert!(ans.contains("What is 2+2?"));
}

#[test]
fn mock_solve_step_includes_docs() {
    let s = MockLtmSolver;
    let d = vec!["doc content".to_string()];
    let ans = s.solve_step("q", &[], &d).unwrap();
    assert!(ans.contains("doc content"));
}

#[test]
fn mock_solve_step_includes_prior_answers() {
    let s = MockLtmSolver;
    let prior = vec![SubProblem::new("q0", "prior answer text", 0)];
    let ans = s.solve_step("q1", &prior, &[]).unwrap();
    assert!(ans.contains("prior answer text"));
}

#[test]
fn mock_solve_step_docs_before_prior_answers() {
    let s = MockLtmSolver;
    let d = vec!["doc_content".to_string()];
    let prior = vec![SubProblem::new("q0", "prior_ans", 0)];
    let ans = s.solve_step("q1", &prior, &d).unwrap();
    let doc_pos = ans.find("doc_content").unwrap();
    let prior_pos = ans.find("prior_ans").unwrap();
    assert!(
        doc_pos < prior_pos,
        "docs must precede prior answers in output"
    );
}

#[test]
fn mock_solve_step_context_prefix_when_docs_present() {
    let s = MockLtmSolver;
    let d = vec!["some doc".to_string()];
    let ans = s.solve_step("q", &[], &d).unwrap();
    assert!(
        ans.starts_with("Context:"),
        "expected 'Context:' prefix, got: {ans}"
    );
}

// ── MockLtmSolver derives ─────────────────────────────────────────────────────

#[test]
fn mock_solver_debug_is_non_empty() {
    assert!(!format!("{:?}", MockLtmSolver).is_empty());
}

#[test]
fn mock_solver_clone_equals_original() {
    let s = MockLtmSolver;
    assert_eq!(s.clone(), s);
}

#[test]
fn mock_solver_default_equals_new() {
    assert_eq!(MockLtmSolver::default(), MockLtmSolver::new());
}

// ── LtmEngine construction ────────────────────────────────────────────────────

#[test]
fn engine_new_stores_config() {
    let cfg = LtmConfig::new().with_max_subproblems(2);
    let e = LtmEngine::new(cfg.clone(), Box::new(MockLtmSolver));
    assert_eq!(e.config, cfg);
}

#[test]
fn engine_debug_is_non_empty() {
    assert!(!format!("{:?}", default_engine()).is_empty());
}

// ── LtmEngine: basic run ──────────────────────────────────────────────────────

#[test]
fn run_single_subproblem_basic() {
    let e = default_engine();
    let res = e.run("What is Rust?", &[]).unwrap();
    assert_eq!(res.original_query, "What is Rust?");
    assert_eq!(res.subproblems.len(), 1);
    assert_eq!(res.subproblems[0].step_index, 0);
    assert!(!res.final_answer.is_empty());
}

#[test]
fn run_two_subproblems_and_conjunction() {
    let e = default_engine();
    let res = e.run("part one and part two", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 2);
}

#[test]
fn run_three_subproblems_then_conjunction() {
    let e = default_engine();
    let res = e.run("step A then step B then step C", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 3);
}

#[test]
fn run_final_answer_equals_last_subproblem_answer() {
    let e = default_engine();
    let res = e.run("a and b and c", &[]).unwrap();
    let last = res.subproblems.last().unwrap();
    assert_eq!(res.final_answer, last.answer);
}

#[test]
fn run_original_query_is_stored() {
    let e = default_engine();
    let q = "What is France and what is its capital?";
    let res = e.run(q, &[]).unwrap();
    assert_eq!(res.original_query, q);
}

// ── LtmEngine: step_index ordering (0-indexed) ────────────────────────────────

#[test]
fn run_step_indices_are_zero_based() {
    let e = default_engine();
    let res = e.run("a and b and c", &[]).unwrap();
    for (i, sp) in res.subproblems.iter().enumerate() {
        assert_eq!(sp.step_index, i, "step_index mismatch at position {i}");
    }
}

#[test]
fn run_step_index_zero_for_single_subproblem() {
    let e = default_engine();
    let res = e.run("What is Rust?", &[]).unwrap();
    assert_eq!(res.subproblems[0].step_index, 0);
}

// ── LtmEngine: subproblem limit ───────────────────────────────────────────────

#[test]
fn run_subproblem_limit_exceeded_returns_error() {
    let e = make_engine(LtmConfig::new().with_max_subproblems(2));
    // "a and b and c" → 3 subproblems > 2 → error
    let err = e.run("a and b and c", &[]).unwrap_err();
    assert!(
        matches!(err, LtmError::SubproblemLimitExceeded(_)),
        "expected SubproblemLimitExceeded, got: {err:?}"
    );
}

#[test]
fn run_subproblem_limit_exceeded_count_in_error() {
    let e = make_engine(LtmConfig::new().with_max_subproblems(2));
    let err = e.run("a and b and c", &[]).unwrap_err();
    if let LtmError::SubproblemLimitExceeded(count) = err {
        assert_eq!(count, 3);
    } else {
        panic!("expected SubproblemLimitExceeded");
    }
}

#[test]
fn run_at_limit_exactly_succeeds() {
    // max_subproblems = 3, query yields exactly 3 → no error
    let e = make_engine(LtmConfig::new().with_max_subproblems(3));
    let res = e.run("a and b and c", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 3);
}

#[test]
fn run_below_limit_succeeds() {
    let e = make_engine(LtmConfig::new().with_max_subproblems(5));
    let res = e.run("a and b", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 2);
}

#[test]
fn run_limit_one_single_subproblem_succeeds() {
    let e = make_engine(LtmConfig::new().with_max_subproblems(1));
    let res = e.run("What is Rust?", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 1);
}

#[test]
fn run_limit_one_two_subproblems_errors() {
    let e = make_engine(LtmConfig::new().with_max_subproblems(1));
    let err = e.run("a and b", &[]).unwrap_err();
    assert!(matches!(err, LtmError::SubproblemLimitExceeded(2)));
}

// ── LtmEngine: prior answers propagation ──────────────────────────────────────

#[test]
fn run_second_step_receives_first_answer() {
    let e = default_engine();
    let res = e.run("first and second", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 2);
    let first_ans = &res.subproblems[0].answer;
    let second_ans = &res.subproblems[1].answer;
    assert!(
        second_ans.contains(first_ans.as_str()),
        "second step answer should include first answer; second={second_ans:?}"
    );
}

#[test]
fn run_third_step_accumulates_both_prior_answers() {
    let e = default_engine();
    let res = e.run("a and b and c", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 3);
    let third_ans = &res.subproblems[2].answer;
    assert!(
        third_ans.contains(res.subproblems[0].answer.as_str()),
        "third step should include first answer"
    );
    assert!(
        third_ans.contains(res.subproblems[1].answer.as_str()),
        "third step should include second answer"
    );
}

#[test]
fn run_prior_answer_content_preserved_verbatim() {
    // The first step with no docs produces "Answer: a" which must appear in step 2.
    let e = default_engine();
    let res = e.run("a and b", &[]).unwrap();
    let first_ans = "Answer: a";
    assert_eq!(res.subproblems[0].answer, first_ans);
    assert!(res.subproblems[1].answer.contains(first_ans));
}

// ── LtmEngine: empty docs ─────────────────────────────────────────────────────

#[test]
fn run_empty_docs_succeeds() {
    let e = default_engine();
    let res = e.run("What is France?", &[]).unwrap();
    assert!(!res.final_answer.is_empty());
}

#[test]
fn run_empty_docs_produces_plain_answer() {
    let e = default_engine();
    let res = e.run("What is France?", &[]).unwrap();
    // No docs, no prior answers → "Answer: <question>"
    assert_eq!(res.subproblems[0].answer, "Answer: What is France?");
}

// ── LtmEngine: context document selection ─────────────────────────────────────

#[test]
fn run_with_docs_includes_context_in_answer() {
    let e = make_engine(LtmConfig::new().with_max_context_docs(2));
    let d = sample_docs();
    let res = e.run("What is the capital of France?", &d).unwrap();
    let ans = &res.subproblems[0].answer;
    assert!(
        ans.contains("France") || ans.contains("Paris") || ans.contains("capital"),
        "expected selected context in answer, got: {ans}"
    );
}

#[test]
fn run_max_context_docs_zero_selects_no_docs() {
    let e = make_engine(LtmConfig::new().with_max_context_docs(0));
    let d = sample_docs();
    let res = e.run("What is France?", &d).unwrap();
    // 0 context docs + no prior → "Answer: <question>"
    assert_eq!(res.subproblems[0].answer, "Answer: What is France?");
}

#[test]
fn run_context_selection_by_token_overlap() {
    // Doc "cat dog" overlaps with "what about cat" on "cat"
    // Doc "apple banana" has no overlap with "what about cat"
    let docs = vec!["cat dog".to_string(), "apple banana".to_string()];
    let e = make_engine(LtmConfig::new().with_max_context_docs(1));
    let res = e.run("what about cat", &docs).unwrap();
    let ans = &res.subproblems[0].answer;
    assert!(
        ans.contains("cat dog"),
        "highest-overlap doc should be selected; answer={ans}"
    );
}

#[test]
fn run_context_limited_to_max_context_docs() {
    // Provide 5 docs; with max_context_docs=2, at most 2 should appear.
    let e = make_engine(LtmConfig::new().with_max_context_docs(2));
    let d = sample_docs(); // 5 docs
    let res = e.run("Paris France", &d).unwrap();
    let ans = &res.subproblems[0].answer;
    // Count how many of the 5 docs appear in the answer.
    let doc_count = d.iter().filter(|doc| ans.contains(doc.as_str())).count();
    assert!(
        doc_count <= 2,
        "at most 2 docs should appear in answer, found {doc_count}; answer={ans}"
    );
}

// ── LtmError display ──────────────────────────────────────────────────────────

#[test]
fn error_decomposition_failed_display() {
    let e = LtmError::DecompositionFailed("bad query".to_string());
    let s = e.to_string();
    assert!(s.contains("bad query"), "display: {s}");
    assert!(s.contains("decomposition failed"), "display: {s}");
}

#[test]
fn error_solve_failed_display() {
    let e = LtmError::SolveFailed("timeout".to_string());
    let s = e.to_string();
    assert!(s.contains("timeout"), "display: {s}");
    assert!(s.contains("solve failed"), "display: {s}");
}

#[test]
fn error_subproblem_limit_exceeded_display() {
    let e = LtmError::SubproblemLimitExceeded(10);
    let s = e.to_string();
    assert!(s.contains("10"), "display: {s}");
    assert!(s.contains("sub-problems"), "display: {s}");
}

#[test]
fn error_decomposition_failed_debug() {
    assert!(!format!("{:?}", LtmError::DecompositionFailed("x".to_string())).is_empty());
}

#[test]
fn error_solve_failed_debug() {
    assert!(!format!("{:?}", LtmError::SolveFailed("r".to_string())).is_empty());
}

#[test]
fn error_subproblem_limit_exceeded_debug() {
    assert!(!format!("{:?}", LtmError::SubproblemLimitExceeded(3)).is_empty());
}

// ── Custom solver: propagation of errors ──────────────────────────────────────

#[derive(Debug)]
struct FailingAtStepSolver {
    fail_at: usize,
}

impl LtmSolver for FailingAtStepSolver {
    fn decompose(&self, _query: &str) -> Result<Vec<String>, LtmError> {
        Ok(vec![
            "step 0".to_string(),
            "step 1".to_string(),
            "step 2".to_string(),
        ])
    }

    fn solve_step(
        &self,
        _question: &str,
        prior_answers: &[SubProblem],
        _docs: &[String],
    ) -> Result<String, LtmError> {
        // current step index = number of prior answers already solved
        let current = prior_answers.len();
        if current == self.fail_at {
            return Err(LtmError::SolveFailed(format!(
                "injected failure at {current}"
            )));
        }
        Ok(format!("answer_for_step_{current}"))
    }
}

#[test]
fn run_propagates_solve_failed_at_step_zero() {
    let e = LtmEngine::new(
        LtmConfig::default(),
        Box::new(FailingAtStepSolver { fail_at: 0 }),
    );
    let err = e.run("q", &[]).unwrap_err();
    assert!(matches!(err, LtmError::SolveFailed(_)));
}

#[test]
fn run_propagates_solve_failed_at_step_one() {
    let e = LtmEngine::new(
        LtmConfig::default(),
        Box::new(FailingAtStepSolver { fail_at: 1 }),
    );
    let err = e.run("q", &[]).unwrap_err();
    assert!(matches!(err, LtmError::SolveFailed(_)));
}

#[derive(Debug)]
struct DecompFailSolver;

impl LtmSolver for DecompFailSolver {
    fn decompose(&self, _query: &str) -> Result<Vec<String>, LtmError> {
        Err(LtmError::DecompositionFailed("cannot parse".to_string()))
    }

    fn solve_step(
        &self,
        _question: &str,
        _prior_answers: &[SubProblem],
        _docs: &[String],
    ) -> Result<String, LtmError> {
        Ok("never called".to_string())
    }
}

#[test]
fn run_propagates_decomposition_failed() {
    let e = LtmEngine::new(LtmConfig::default(), Box::new(DecompFailSolver));
    let err = e.run("some query", &[]).unwrap_err();
    assert!(matches!(err, LtmError::DecompositionFailed(_)));
}

// ── tokenize helper ───────────────────────────────────────────────────────────

#[test]
fn tokenize_lowercase() {
    let tokens = tokenize("Hello World");
    assert!(tokens.iter().all(|t| t == &t.to_lowercase()));
}

#[test]
fn tokenize_filters_single_char_tokens() {
    let tokens = tokenize("a be the cat");
    // "a" has len 1 → filtered; "be"/"the"/"cat" have len ≥ 2 → kept.
    assert!(!tokens.contains(&"a".to_string()));
    assert!(tokens.contains(&"be".to_string()));
}

#[test]
fn tokenize_splits_on_non_alphanumeric() {
    let tokens = tokenize("hello-world foo.bar");
    assert!(tokens.contains(&"hello".to_string()));
    assert!(tokens.contains(&"world".to_string()));
    assert!(tokens.contains(&"foo".to_string()));
    assert!(tokens.contains(&"bar".to_string()));
}

#[test]
fn tokenize_empty_string_yields_empty() {
    assert!(tokenize("").is_empty());
}

#[test]
fn tokenize_only_separators_yields_empty() {
    assert!(tokenize("---...---").is_empty());
}

// ── LtmResult ─────────────────────────────────────────────────────────────────

#[test]
fn result_clone_equals_original() {
    let e = default_engine();
    let res = e.run("a and b", &[]).unwrap();
    assert_eq!(res.clone(), res);
}

#[test]
fn result_debug_is_non_empty() {
    let e = default_engine();
    let res = e.run("a and b", &[]).unwrap();
    assert!(!format!("{:?}", res).is_empty());
}

#[test]
fn result_subproblems_len_matches_decomposition() {
    let e = default_engine();
    let res = e.run("one and two and three", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 3);
}

// ── dyn trait object ──────────────────────────────────────────────────────────

#[test]
fn run_via_dyn_solver_in_engine() {
    let solver: Box<dyn LtmSolver> = Box::new(MockLtmSolver);
    let e = LtmEngine::new(LtmConfig::default(), solver);
    let res = e.run("What is Rust?", &[]).unwrap();
    assert_eq!(res.subproblems.len(), 1);
    assert!(!res.final_answer.is_empty());
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn run_is_deterministic() {
    let d = sample_docs();
    let a = LtmEngine::new(LtmConfig::default(), Box::new(MockLtmSolver))
        .run("What is France and what is Paris?", &d)
        .unwrap();
    let b = LtmEngine::new(LtmConfig::default(), Box::new(MockLtmSolver))
        .run("What is France and what is Paris?", &d)
        .unwrap();
    assert_eq!(a, b);
}
