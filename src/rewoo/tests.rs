#![allow(clippy::too_many_lines)]

use std::cell::RefCell;

use super::engine::{
    MockRewooGenerator, MockRewooPlanSource, MockRewooRetriever, RewooGenerator, RewooPipeline,
    RewooPlanSource, RewooPlanner, RewooRetriever, RewooSolver, RewooWorker, evaluate_expression,
    substitute_placeholders,
};
use super::types::{
    PlaceholderVar, RewooAction, RewooConfig, RewooError, RewooEvidence, RewooPlan, RewooStep,
};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn var(n: usize) -> PlaceholderVar {
    PlaceholderVar::new(n)
}

fn search(text: impl Into<String>) -> RewooAction {
    RewooAction::Search(text.into())
}

fn compute(text: impl Into<String>) -> RewooAction {
    RewooAction::Compute(text.into())
}

fn evidence_of(pairs: &[(usize, &str)]) -> RewooEvidence {
    let mut evidence = RewooEvidence::new();
    for (n, text) in pairs {
        evidence.insert(var(*n), (*text).to_string());
    }
    evidence
}

// ── PlaceholderVar ─────────────────────────────────────────────────────────────

#[test]
fn placeholder_var_display_renders_hash_e_n() {
    assert_eq!(var(1).to_string(), "#E1");
    assert_eq!(var(10).to_string(), "#E10");
    assert_eq!(var(999).to_string(), "#E999");
}

#[test]
fn placeholder_var_index_roundtrips() {
    assert_eq!(var(7).index(), 7);
}

#[test]
fn placeholder_var_parse_token_accepts_valid_tokens() {
    assert_eq!(PlaceholderVar::parse_token("#E1"), Some(var(1)));
    assert_eq!(PlaceholderVar::parse_token("#E10"), Some(var(10)));
    assert_eq!(PlaceholderVar::parse_token("#E0"), Some(var(0)));
}

#[test]
fn placeholder_var_parse_token_rejects_malformed_tokens() {
    assert_eq!(PlaceholderVar::parse_token("#E"), None, "no digits");
    assert_eq!(PlaceholderVar::parse_token("#e1"), None, "lowercase e");
    assert_eq!(PlaceholderVar::parse_token("E1"), None, "missing hash");
    assert_eq!(
        PlaceholderVar::parse_token("#E1x"),
        None,
        "trailing garbage"
    );
    assert_eq!(PlaceholderVar::parse_token("#E1.5"), None, "decimal point");
    assert_eq!(PlaceholderVar::parse_token(""), None, "empty string");
}

#[test]
fn placeholder_var_ord_is_numeric_not_lexicographic() {
    // A plain string comparison would put "#E10" before "#E2" (since '1' < '2'
    // as characters); the wrapped-usize Ord must not make that mistake.
    assert!(var(2) < var(10));
    assert!(var(9) < var(10));
    let mut vars = vec![var(10), var(2), var(1), var(9)];
    vars.sort();
    assert_eq!(vars, vec![var(1), var(2), var(9), var(10)]);
}

// ── substitute_placeholders ────────────────────────────────────────────────────

#[test]
fn substitute_placeholders_replaces_single_token() {
    let evidence = evidence_of(&[(1, "Paris")]);
    let result = substitute_placeholders("The capital is #E1.", &evidence).unwrap();
    assert_eq!(result, "The capital is Paris.");
}

#[test]
fn substitute_placeholders_no_tokens_returns_text_unchanged() {
    let evidence = RewooEvidence::new();
    let result = substitute_placeholders("no placeholders here", &evidence).unwrap();
    assert_eq!(result, "no placeholders here");
}

#[test]
fn substitute_placeholders_disambiguates_e1_from_e10() {
    let evidence = evidence_of(&[(1, "A"), (10, "B")]);
    let result = substitute_placeholders("value is #E10 not #E1", &evidence).unwrap();
    // A naive `str::replace("#E1", "A")` would corrupt "#E10" into "A0",
    // producing "value is A0 not A". The correct, maximal-munch result reads
    // "#E10" as a single whole token.
    assert_eq!(result, "value is B not A");
}

#[test]
fn substitute_placeholders_handles_adjacent_tokens_without_separator() {
    let evidence = evidence_of(&[(1, "A"), (10, "B")]);
    assert_eq!(substitute_placeholders("#E1#E10", &evidence).unwrap(), "AB");
    assert_eq!(substitute_placeholders("#E10#E1", &evidence).unwrap(), "BA");
}

#[test]
fn substitute_placeholders_leaves_non_placeholder_hash_e_text_alone() {
    let evidence = RewooEvidence::new();
    let result = substitute_placeholders("#E is not a token, nor is #East", &evidence).unwrap();
    assert_eq!(result, "#E is not a token, nor is #East");
}

#[test]
fn substitute_placeholders_returns_error_for_unresolved_reference() {
    let evidence = evidence_of(&[(1, "Paris")]);
    let err = substitute_placeholders("population of #E2", &evidence).unwrap_err();
    assert_eq!(err, RewooError::UnresolvedPlaceholder(var(2)));
}

#[test]
fn substitute_placeholders_multiple_occurrences_of_same_var() {
    let evidence = evidence_of(&[(1, "X")]);
    let result = substitute_placeholders("#E1 and #E1 again", &evidence).unwrap();
    assert_eq!(result, "X and X again");
}

// ── evaluate_expression ────────────────────────────────────────────────────────

#[test]
fn evaluate_expression_respects_operator_precedence() {
    assert_eq!(evaluate_expression("2 + 3 * 4").unwrap(), "14");
}

#[test]
fn evaluate_expression_respects_parentheses() {
    assert_eq!(evaluate_expression("(2 + 3) * 4").unwrap(), "20");
}

#[test]
fn evaluate_expression_formats_integral_results_without_decimal() {
    assert_eq!(evaluate_expression("9 / 3").unwrap(), "3");
}

#[test]
fn evaluate_expression_formats_fractional_results() {
    assert_eq!(evaluate_expression("10 / 4").unwrap(), "2.5");
}

#[test]
fn evaluate_expression_handles_unary_minus() {
    assert_eq!(evaluate_expression("-5 + 3").unwrap(), "-2");
}

#[test]
fn evaluate_expression_handles_double_minus() {
    assert_eq!(evaluate_expression("3 - -2").unwrap(), "5");
}

#[test]
fn evaluate_expression_handles_nested_parentheses() {
    assert_eq!(evaluate_expression("((1 + 2) * (3 + 4))").unwrap(), "21");
}

#[test]
fn evaluate_expression_rejects_division_by_zero() {
    let err = evaluate_expression("1 / 0").unwrap_err();
    assert!(matches!(err, RewooError::ComputeFailed { .. }));
}

#[test]
fn evaluate_expression_rejects_empty_expression() {
    assert!(evaluate_expression("").is_err());
    assert!(evaluate_expression("   ").is_err());
}

#[test]
fn evaluate_expression_rejects_malformed_input() {
    assert!(evaluate_expression("2 +").is_err(), "trailing operator");
    assert!(evaluate_expression("2 3").is_err(), "missing operator");
    assert!(evaluate_expression("(2 + 3").is_err(), "unbalanced paren");
    assert!(
        evaluate_expression("2 + 3)").is_err(),
        "stray closing paren"
    );
    assert!(evaluate_expression("2 $ 3").is_err(), "unknown character");
}

#[test]
fn evaluate_expression_after_placeholder_substitution() {
    let evidence = evidence_of(&[(1, "5")]);
    let substituted = substitute_placeholders("#E1 + 10", &evidence).unwrap();
    assert_eq!(evaluate_expression(&substituted).unwrap(), "15");
}

// ── RewooPlan / RewooStep / RewooAction ────────────────────────────────────────

#[test]
fn plan_len_is_empty_and_with_step() {
    let plan = RewooPlan::new("q", Vec::new());
    assert!(plan.is_empty());
    assert_eq!(plan.len(), 0);

    let plan = plan.with_step(RewooStep::new("r", search("s"), var(1)));
    assert!(!plan.is_empty());
    assert_eq!(plan.len(), 1);
}

#[test]
fn action_raw_text_returns_inner_string_for_both_variants() {
    assert_eq!(search("hello").raw_text(), "hello");
    assert_eq!(compute("1 + 1").raw_text(), "1 + 1");
}

// ── RewooEvidence ─────────────────────────────────────────────────────────────

#[test]
fn evidence_insert_get_len_is_empty() {
    let mut evidence = RewooEvidence::new();
    assert!(evidence.is_empty());
    assert_eq!(evidence.get(var(1)), None);

    evidence.insert(var(1), "Paris");
    assert!(!evidence.is_empty());
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence.get(var(1)), Some("Paris"));
}

#[test]
fn evidence_insert_overwrites_prior_value() {
    let mut evidence = RewooEvidence::new();
    evidence.insert(var(1), "first");
    evidence.insert(var(1), "second");
    assert_eq!(evidence.get(var(1)), Some("second"));
    assert_eq!(evidence.len(), 1);
}

#[test]
fn evidence_iter_yields_ascending_placeholder_order() {
    let evidence = evidence_of(&[(10, "B"), (1, "A"), (2, "C")]);
    let pairs: Vec<(PlaceholderVar, &str)> = evidence.iter().collect();
    assert_eq!(pairs, vec![(var(1), "A"), (var(2), "C"), (var(10), "B")]);
}

// ── RewooConfig ────────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = RewooConfig::default();
    assert_eq!(config.max_steps, 16);
    assert!(config.validate_plan_structure);
}

#[test]
fn config_builders_chain() {
    let config = RewooConfig::new()
        .with_max_steps(3)
        .with_validate_plan_structure(false);
    assert_eq!(config.max_steps, 3);
    assert!(!config.validate_plan_structure);
}

// ── MockRewooRetriever / MockRewooPlanSource / MockRewooGenerator ─────────────

#[test]
fn mock_retriever_matches_substring_mapping() {
    let retriever = MockRewooRetriever::new(vec![("capital".to_string(), "Paris".to_string())]);
    assert_eq!(retriever.retrieve("capital of France"), "Paris");
}

#[test]
fn mock_retriever_echoes_query_when_no_mapping_matches() {
    let retriever = MockRewooRetriever::echo();
    assert_eq!(retriever.retrieve("anything"), "anything");
}

#[test]
fn mock_plan_source_matches_substring_mapping_and_falls_back_to_default() {
    let matched_plan = RewooPlan::new("france", vec![RewooStep::new("r", search("s"), var(1))]);
    let default_plan = RewooPlan::new("default", Vec::new());
    let source = MockRewooPlanSource::new(vec![("France".to_string(), matched_plan.clone())])
        .with_default(default_plan.clone());

    assert_eq!(source.generate_plan("capital of France?"), matched_plan);
    assert_eq!(source.generate_plan("unrelated query"), default_plan);
}

#[test]
fn mock_plan_source_single_ignores_query() {
    let plan = RewooPlan::new("q", vec![RewooStep::new("r", search("s"), var(1))]);
    let source = MockRewooPlanSource::single(plan.clone());
    assert_eq!(source.generate_plan("query A"), plan);
    assert_eq!(source.generate_plan("query B"), plan);
}

#[test]
fn mock_generator_fills_query_and_plan_placeholders() {
    let generator = MockRewooGenerator::new("Q: {query}\nEvidence: {plan}");
    let out = generator.generate("What is Rust?", "#E1: a language");
    assert!(out.contains("Q: What is Rust?"));
    assert!(out.contains("Evidence: #E1: a language"));
}

#[test]
fn mock_generator_default_echoes_substituted_plan() {
    let generator = MockRewooGenerator::default();
    assert_eq!(
        generator.generate("ignored", "the plan text"),
        "the plan text"
    );
}

// ── RewooPlanner ──────────────────────────────────────────────────────────────

#[test]
fn planner_rejects_empty_query() {
    let planner = RewooPlanner::new(RewooConfig::default());
    let source = MockRewooPlanSource::default();
    let err = planner.plan("   ", &source).unwrap_err();
    assert_eq!(err, RewooError::EmptyQuery);
}

#[test]
fn planner_rejects_plan_too_long() {
    let steps = (1..=5)
        .map(|n| RewooStep::new("r", search("s"), var(n)))
        .collect();
    let plan = RewooPlan::new("q", steps);
    let source = MockRewooPlanSource::single(plan);
    let planner = RewooPlanner::new(RewooConfig::default().with_max_steps(3));

    let err = planner.plan("q", &source).unwrap_err();
    assert_eq!(err, RewooError::PlanTooLong { len: 5, max: 3 });
}

#[test]
fn planner_rejects_duplicate_placeholder_definition() {
    let plan = RewooPlan::new(
        "q",
        vec![
            RewooStep::new("r1", search("a"), var(1)),
            RewooStep::new("r2", search("b"), var(1)),
        ],
    );
    let source = MockRewooPlanSource::single(plan);
    let planner = RewooPlanner::new(RewooConfig::default());

    let err = planner.plan("q", &source).unwrap_err();
    assert_eq!(err, RewooError::DuplicatePlaceholder(var(1)));
}

#[test]
fn planner_detects_forward_reference_when_structural_validation_enabled() {
    // Step 0 references #E2, which is only ever (going to be) defined by
    // step 1 — a forward reference that must be rejected eagerly, before any
    // evidence is gathered.
    let plan = RewooPlan::new(
        "q",
        vec![
            RewooStep::new("uses #E2 too early", search("about #E2"), var(1)),
            RewooStep::new("defines #E2", search("b"), var(2)),
        ],
    );
    let source = MockRewooPlanSource::single(plan);
    let planner = RewooPlanner::new(RewooConfig::default());

    let err = planner.plan("q", &source).unwrap_err();
    assert_eq!(
        err,
        RewooError::ForwardReference {
            step: 0,
            var: var(2)
        }
    );
}

#[test]
fn planner_allows_forward_reference_through_when_structural_validation_disabled() {
    let plan = RewooPlan::new(
        "q",
        vec![
            RewooStep::new("uses #E2 too early", search("about #E2"), var(1)),
            RewooStep::new("defines #E2", search("b"), var(2)),
        ],
    );
    let source = MockRewooPlanSource::single(plan);
    let planner = RewooPlanner::new(RewooConfig::default().with_validate_plan_structure(false));

    // The malformed plan passes the (disabled) eager check; it will only be
    // caught later, at worker execution time.
    let result = planner.plan("q", &source);
    assert!(result.is_ok());
}

#[test]
fn planner_calls_plan_source_exactly_once_regardless_of_plan_length() {
    struct CountingPlanSource {
        calls: RefCell<usize>,
        plan: RewooPlan,
    }
    impl RewooPlanSource for CountingPlanSource {
        fn generate_plan(&self, _query: &str) -> RewooPlan {
            *self.calls.borrow_mut() += 1;
            self.plan.clone()
        }
    }

    let steps = (1..=8)
        .map(|n| RewooStep::new("r", search(format!("s{n}")), var(n)))
        .collect();
    let source = CountingPlanSource {
        calls: RefCell::new(0),
        plan: RewooPlan::new("q", steps),
    };
    let planner = RewooPlanner::new(RewooConfig::default());

    let plan = planner.plan("an eight-step query", &source).unwrap();
    assert_eq!(plan.len(), 8);
    assert_eq!(*source.calls.borrow(), 1);
}

// ── RewooWorker ───────────────────────────────────────────────────────────────

#[test]
fn worker_resolves_multi_step_plan_with_placeholder_chaining() {
    let plan = RewooPlan::new(
        "What is the population of France's capital, in thousands?",
        vec![
            RewooStep::new(
                "Find the capital of France.",
                search("capital of France"),
                var(1),
            ),
            RewooStep::new(
                "Find the population of the capital found in #E1.",
                search("population of #E1"),
                var(2),
            ),
            RewooStep::new(
                "Convert the population in #E2 to thousands.",
                compute("#E2 / 1000"),
                var(3),
            ),
        ],
    );

    let retriever = MockRewooRetriever::new(vec![
        ("capital of France".to_string(), "Paris".to_string()),
        ("population of Paris".to_string(), "2141000".to_string()),
    ]);

    let worker = RewooWorker::new(RewooConfig::default());
    let evidence = worker.run(&plan, &retriever).unwrap();

    assert_eq!(evidence.get(var(1)), Some("Paris"));
    assert_eq!(evidence.get(var(2)), Some("2141000"));
    assert_eq!(evidence.get(var(3)), Some("2141"));
}

#[test]
fn worker_empty_plan_returns_empty_evidence_without_panicking() {
    let plan = RewooPlan::new("q", Vec::new());
    let retriever = MockRewooRetriever::echo();
    let worker = RewooWorker::new(RewooConfig::default());

    let evidence = worker.run(&plan, &retriever).unwrap();
    assert!(evidence.is_empty());
}

#[test]
fn worker_rejects_forward_reference_eagerly_as_error_not_panic() {
    // Step 0's action references #E2, which no earlier step defines (indeed
    // no step defines it at all here) — with eager structural validation on
    // (the default), the worker must reject this before doing any work, and
    // must return an Err, never panic.
    let plan = RewooPlan::new(
        "q",
        vec![RewooStep::new(
            "references an unresolved placeholder",
            search("info about #E2"),
            var(1),
        )],
    );
    let retriever = MockRewooRetriever::echo();
    let worker = RewooWorker::new(RewooConfig::default());

    let err = worker.run(&plan, &retriever).unwrap_err();
    assert_eq!(
        err,
        RewooError::ForwardReference {
            step: 0,
            var: var(2)
        }
    );
}

#[test]
fn worker_rejects_unresolved_placeholder_at_runtime_when_eager_validation_disabled() {
    // Same malformed plan as above, but with eager structural validation
    // switched off: the runtime substitution backstop inside `run` must still
    // catch it and return an error, never panic or silently proceed.
    let plan = RewooPlan::new(
        "q",
        vec![RewooStep::new(
            "references an unresolved placeholder",
            search("info about #E2"),
            var(1),
        )],
    );
    let retriever = MockRewooRetriever::echo();
    let worker = RewooWorker::new(RewooConfig::default().with_validate_plan_structure(false));

    let err = worker.run(&plan, &retriever).unwrap_err();
    assert_eq!(err, RewooError::UnresolvedPlaceholder(var(2)));
}

#[test]
fn worker_rejects_duplicate_placeholder_definition() {
    let plan = RewooPlan::new(
        "q",
        vec![
            RewooStep::new("r1", search("a"), var(1)),
            RewooStep::new("r2", search("b"), var(1)),
        ],
    );
    let retriever = MockRewooRetriever::echo();
    let worker = RewooWorker::new(RewooConfig::default());

    let err = worker.run(&plan, &retriever).unwrap_err();
    assert_eq!(err, RewooError::DuplicatePlaceholder(var(1)));
}

#[test]
fn worker_rejects_plan_exceeding_max_steps() {
    let steps = (1..=4)
        .map(|n| RewooStep::new("r", search("s"), var(n)))
        .collect();
    let plan = RewooPlan::new("q", steps);
    let retriever = MockRewooRetriever::echo();
    let worker = RewooWorker::new(RewooConfig::default().with_max_steps(2));

    let err = worker.run(&plan, &retriever).unwrap_err();
    assert_eq!(err, RewooError::PlanTooLong { len: 4, max: 2 });
}

#[test]
fn worker_compute_action_propagates_evaluation_error() {
    let plan = RewooPlan::new(
        "q",
        vec![RewooStep::new("divide by zero", compute("1 / 0"), var(1))],
    );
    let retriever = MockRewooRetriever::echo();
    let worker = RewooWorker::new(RewooConfig::default());

    let err = worker.run(&plan, &retriever).unwrap_err();
    assert!(matches!(err, RewooError::ComputeFailed { .. }));
}

// ── RewooSolver ───────────────────────────────────────────────────────────────

#[test]
fn solver_rejects_empty_query() {
    let plan = RewooPlan::new("q", Vec::new());
    let evidence = RewooEvidence::new();
    let generator = MockRewooGenerator::default();
    let solver = RewooSolver::new(RewooConfig::default());

    let err = solver
        .solve("  ", &plan, &evidence, &generator)
        .unwrap_err();
    assert_eq!(err, RewooError::EmptyQuery);
}

#[test]
fn solver_substitutes_evidence_defined_anywhere_in_the_plan() {
    // Unlike the worker's action-resolution step, the solver runs *after*
    // the whole plan has executed, so a step's reasoning text may reference
    // a placeholder defined by a *later* step without error.
    let plan = RewooPlan::new(
        "q",
        vec![
            RewooStep::new("mentions #E2 early", search("a"), var(1)),
            RewooStep::new("defines #E2", search("b"), var(2)),
        ],
    );
    let evidence = evidence_of(&[(1, "first"), (2, "second")]);
    let generator = MockRewooGenerator::default();
    let solver = RewooSolver::new(RewooConfig::default());

    let answer = solver.solve("q", &plan, &evidence, &generator).unwrap();
    assert!(answer.contains("second"));
}

#[test]
fn solver_returns_unresolved_placeholder_when_evidence_missing_entry() {
    let plan = RewooPlan::new(
        "q",
        vec![RewooStep::new("references #E1", search("a"), var(1))],
    );
    let evidence = RewooEvidence::new(); // #E1 was never resolved
    let generator = MockRewooGenerator::default();
    let solver = RewooSolver::new(RewooConfig::default());

    let err = solver.solve("q", &plan, &evidence, &generator).unwrap_err();
    assert_eq!(err, RewooError::UnresolvedPlaceholder(var(1)));
}

#[test]
fn solver_handles_empty_plan_without_panicking() {
    let plan = RewooPlan::new("q", Vec::new());
    let evidence = RewooEvidence::new();
    let generator = MockRewooGenerator::new("Answer to {query}: [{plan}]");
    let solver = RewooSolver::new(RewooConfig::default());

    let answer = solver.solve("q", &plan, &evidence, &generator).unwrap();
    assert_eq!(answer, "Answer to q: []");
}

// ── Efficiency property: exactly one planning call, one retrieval per Search ──

#[test]
fn worker_makes_exactly_one_retrieval_call_per_search_step_and_planner_is_never_recalled() {
    struct CountingPlanSource {
        calls: RefCell<usize>,
        plan: RewooPlan,
    }
    impl RewooPlanSource for CountingPlanSource {
        fn generate_plan(&self, _query: &str) -> RewooPlan {
            *self.calls.borrow_mut() += 1;
            self.plan.clone()
        }
    }

    struct CountingRetriever {
        calls: RefCell<usize>,
    }
    impl RewooRetriever for CountingRetriever {
        fn retrieve(&self, query: &str) -> String {
            *self.calls.borrow_mut() += 1;
            format!("evidence for {query}")
        }
    }

    // Four steps: three Search actions and one Compute action, so the
    // retrieval-call count must land on exactly 3, not 4.
    let plan = RewooPlan::new(
        "multi-part query",
        vec![
            RewooStep::new("r1", search("s1"), var(1)),
            RewooStep::new("r2", search("s2 #E1"), var(2)),
            RewooStep::new("r3", compute("1 + 1"), var(3)),
            RewooStep::new("r4", search("s4 #E2 #E3"), var(4)),
        ],
    );
    let plan_source = CountingPlanSource {
        calls: RefCell::new(0),
        plan,
    };
    let retriever = CountingRetriever {
        calls: RefCell::new(0),
    };

    let config = RewooConfig::default();
    let planner = RewooPlanner::new(config.clone());
    let resolved_plan = planner.plan("multi-part query", &plan_source).unwrap();
    assert_eq!(
        *plan_source.calls.borrow(),
        1,
        "planning must happen exactly once, no matter how many steps the plan has"
    );

    let worker = RewooWorker::new(config);
    let evidence = worker.run(&resolved_plan, &retriever).unwrap();
    assert_eq!(evidence.len(), 4);
    assert_eq!(
        *retriever.calls.borrow(),
        3,
        "exactly one retrieval call per Action::Search step"
    );

    // Crucially: running the worker must not have triggered any additional
    // planning calls — planning and evidence-gathering are never interleaved.
    assert_eq!(
        *plan_source.calls.borrow(),
        1,
        "the worker must never re-invoke the planner mid-execution"
    );
}

#[test]
fn planner_call_count_is_independent_of_plan_length() {
    // Same assertion as above, but scanning plan lengths 0..=6 to make sure
    // "exactly one call" holds regardless of how many steps are produced —
    // contrasting with a ReAct-style loop, whose call count grows linearly
    // with the number of steps.
    for step_count in 0..=6usize {
        struct CountingPlanSource {
            calls: RefCell<usize>,
            plan: RewooPlan,
        }
        impl RewooPlanSource for CountingPlanSource {
            fn generate_plan(&self, _query: &str) -> RewooPlan {
                *self.calls.borrow_mut() += 1;
                self.plan.clone()
            }
        }

        let steps = (1..=step_count)
            .map(|n| RewooStep::new("r", search(format!("s{n}")), var(n)))
            .collect();
        let source = CountingPlanSource {
            calls: RefCell::new(0),
            plan: RewooPlan::new("q", steps),
        };
        let planner = RewooPlanner::new(RewooConfig::default().with_max_steps(10));

        let plan = planner.plan("q", &source).unwrap();
        assert_eq!(plan.len(), step_count);
        assert_eq!(*source.calls.borrow(), 1);
    }
}

// ── RewooPipeline ─────────────────────────────────────────────────────────────

#[test]
fn pipeline_runs_plan_resolve_solve_end_to_end() {
    let plan = RewooPlan::new(
        "capital of France?",
        vec![RewooStep::new(
            "Look up the capital of France.",
            search("capital of France"),
            var(1),
        )],
    );
    let plan_source = MockRewooPlanSource::single(plan);
    let retriever =
        MockRewooRetriever::new(vec![("capital of France".to_string(), "Paris".to_string())]);
    let generator = MockRewooGenerator::default();

    let pipeline = RewooPipeline::new(RewooConfig::default());
    let outcome = pipeline
        .run("capital of France?", &plan_source, &retriever, &generator)
        .unwrap();

    assert_eq!(outcome.query, "capital of France?");
    assert_eq!(outcome.plan.len(), 1);
    assert_eq!(outcome.evidence.get(var(1)), Some("Paris"));
    assert!(outcome.answer.contains("Paris"));
}

#[test]
fn pipeline_propagates_planner_error() {
    let plan_source = MockRewooPlanSource::default();
    let retriever = MockRewooRetriever::echo();
    let generator = MockRewooGenerator::default();
    let pipeline = RewooPipeline::new(RewooConfig::default());

    let err = pipeline
        .run("   ", &plan_source, &retriever, &generator)
        .unwrap_err();
    assert_eq!(err, RewooError::EmptyQuery);
}

#[test]
fn pipeline_propagates_worker_error() {
    let plan = RewooPlan::new(
        "q",
        vec![RewooStep::new("divide by zero", compute("1 / 0"), var(1))],
    );
    let plan_source = MockRewooPlanSource::single(plan);
    let retriever = MockRewooRetriever::echo();
    let generator = MockRewooGenerator::default();
    let pipeline = RewooPipeline::new(RewooConfig::default());

    let err = pipeline
        .run("q", &plan_source, &retriever, &generator)
        .unwrap_err();
    assert!(matches!(err, RewooError::ComputeFailed { .. }));
}

#[test]
fn pipeline_propagates_solver_error_when_reasoning_references_undefined_placeholder() {
    // The plan's single step defines #E1, but its *reasoning* text
    // (evaluated only by the solver, after the worker has already finished)
    // references #E2, which nothing in the plan ever defines. The worker
    // succeeds; the solver must then fail cleanly instead of panicking.
    let plan = RewooPlan::new(
        "q",
        vec![RewooStep::new(
            "refers to #E2 which nothing defines",
            search("a"),
            var(1),
        )],
    );
    let plan_source = MockRewooPlanSource::single(plan);
    let retriever = MockRewooRetriever::echo();
    let generator = MockRewooGenerator::default();
    let pipeline = RewooPipeline::new(RewooConfig::default());

    let err = pipeline
        .run("q", &plan_source, &retriever, &generator)
        .unwrap_err();
    assert_eq!(err, RewooError::UnresolvedPlaceholder(var(2)));
}

// ── RewooError Display ────────────────────────────────────────────────────────

#[test]
fn error_display_messages_are_human_readable() {
    assert_eq!(
        RewooError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(
        RewooError::PlanTooLong { len: 5, max: 3 }.to_string(),
        "plan has 5 steps, exceeding the configured maximum of 3"
    );
    assert_eq!(
        RewooError::DuplicatePlaceholder(var(2)).to_string(),
        "placeholder #E2 is defined by more than one step"
    );
    assert_eq!(
        RewooError::UnresolvedPlaceholder(var(3)).to_string(),
        "placeholder #E3 is referenced before it has been resolved"
    );
}
