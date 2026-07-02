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
    clippy::no_effect_underscore_binding
)]

//! Tests for the `erag` module.

use std::collections::HashMap;

use super::correlation::{kendall_tau, spearman_rho};
use super::evaluator::ERagEvaluator;
use super::metrics::{MockDownstreamTask, RougeLiteUtility};
use super::types::{
    AggregationMethod, DownstreamTask, ERagCase, ERagConfig, ERagError, ERagReport, PerDocScore,
    UtilityMetric,
};

// ── Test-only helper task/metric implementations ─────────────────────────────

/// A [`DownstreamTask`] that simply concatenates its context with `" | "`, so
/// tests can control the exact task output (and therefore drive a
/// [`LookupUtility`] or [`FixedUtility`] deterministically).
#[derive(Debug, Clone, Copy)]
struct IdentityTask;

impl DownstreamTask for IdentityTask {
    fn run(&self, _query: &str, context: &[String]) -> String {
        context.join(" | ")
    }
}

/// A [`UtilityMetric`] that ignores its inputs and always returns a fixed
/// score, letting tests pin every per-document and end-to-end score to a
/// known constant.
#[derive(Debug, Clone, Copy)]
struct FixedUtility(f32);

impl UtilityMetric for FixedUtility {
    fn score(&self, _output: &str, _gold: &str) -> f32 {
        self.0
    }
}

/// A [`UtilityMetric`] that scores by exact-string lookup, letting tests
/// assign a distinct known score to each possible task output.
#[derive(Debug, Clone)]
struct LookupUtility(HashMap<String, f32>);

impl UtilityMetric for LookupUtility {
    fn score(&self, output: &str, _gold: &str) -> f32 {
        self.0.get(output).copied().unwrap_or(0.0)
    }
}

fn default_evaluator() -> ERagEvaluator {
    ERagEvaluator::new(ERagConfig::default())
}

// ── RougeLiteUtility ──────────────────────────────────────────────────────────

#[test]
fn rouge_perfect_match() {
    let utility = RougeLiteUtility::new();
    assert!((utility.score("the cat sat", "the cat sat") - 1.0).abs() < 1e-6);
}

#[test]
fn rouge_no_overlap() {
    let utility = RougeLiteUtility::new();
    assert_eq!(utility.score("hello world", "foo bar"), 0.0);
}

#[test]
fn rouge_partial_precision_recall() {
    let utility = RougeLiteUtility::new();
    let score = utility.score("the the the", "the");
    assert!((score - 0.5).abs() < 1e-6);
}

#[test]
fn rouge_empty_output() {
    let utility = RougeLiteUtility::new();
    assert_eq!(utility.score("", "gold text"), 0.0);
}

#[test]
fn rouge_empty_gold() {
    let utility = RougeLiteUtility::new();
    assert_eq!(utility.score("output text", ""), 0.0);
}

#[test]
fn rouge_both_empty() {
    let utility = RougeLiteUtility::new();
    assert_eq!(utility.score("", ""), 0.0);
}

#[test]
fn rouge_case_insensitive() {
    let utility = RougeLiteUtility::new();
    assert!((utility.score("THE CAT", "the cat") - 1.0).abs() < 1e-6);
}

#[test]
fn rouge_punctuation_ignored() {
    let utility = RougeLiteUtility::new();
    assert!((utility.score("the cat, sat.", "the cat sat") - 1.0).abs() < 1e-6);
}

#[test]
fn rouge_score_bounded_in_unit_interval() {
    let utility = RougeLiteUtility::new();
    let samples = [
        ("a quick brown fox", "a slow brown dog"),
        ("completely different text here", "totally unrelated words"),
        (
            "shared token overlap example",
            "shared token overlap example plus more",
        ),
    ];
    for (output, gold) in samples {
        let score = utility.score(output, gold);
        assert!((0.0..=1.0).contains(&score), "score {score} out of range");
    }
}

#[test]
fn rouge_default_and_new_equivalent() {
    let a = RougeLiteUtility::new();
    let b = RougeLiteUtility::default();
    assert_eq!(a.score("x y z", "x y"), b.score("x y z", "x y"));
}

// ── MockDownstreamTask ────────────────────────────────────────────────────────

#[test]
fn mock_task_single_sentence_doc_returns_whole_doc() {
    let task = MockDownstreamTask::new();
    let docs = vec!["A simple sentence with no punctuation".to_string()];
    let output = task.run("simple", &docs);
    assert_eq!(output, "A simple sentence with no punctuation");
}

#[test]
fn mock_task_multi_sentence_picks_highest_overlap() {
    let task = MockDownstreamTask::new();
    let docs = vec!["Cats are mammals. Dogs are loyal companions.".to_string()];
    let output = task.run("Which animals are loyal?", &docs);
    assert_eq!(output, "Dogs are loyal companions");
}

#[test]
fn mock_task_tie_breaks_to_earliest_sentence() {
    let task = MockDownstreamTask::new();
    let docs = vec!["Red apples are sweet. Green apples are sweet too.".to_string()];
    let output = task.run("apples sweet", &docs);
    assert_eq!(output, "Red apples are sweet");
}

#[test]
fn mock_task_joins_multiple_docs_with_space() {
    let task = MockDownstreamTask::new();
    let docs = vec![
        "First document here".to_string(),
        "Second document here".to_string(),
    ];
    let output = task.run("document", &docs);
    assert_eq!(output, "First document here Second document here");
}

#[test]
fn mock_task_empty_context_returns_empty_string() {
    let task = MockDownstreamTask::new();
    let docs: Vec<String> = vec![];
    assert_eq!(task.run("query", &docs), "");
}

#[test]
fn mock_task_blank_doc_contributes_nothing() {
    let task = MockDownstreamTask::new();
    let docs = vec!["   ".to_string(), "Real content here".to_string()];
    let output = task.run("content", &docs);
    assert_eq!(output, "Real content here");
}

#[test]
fn mock_task_default_and_new_equivalent() {
    let a = MockDownstreamTask::new();
    let b = MockDownstreamTask::default();
    let docs = vec!["Some content.".to_string()];
    assert_eq!(a.run("content", &docs), b.run("content", &docs));
}

#[test]
fn mock_task_case_insensitive_overlap_matching() {
    let task = MockDownstreamTask::new();
    let docs = vec!["APPLES are great. Oranges are okay.".to_string()];
    let output = task.run("apples", &docs);
    assert_eq!(output, "APPLES are great");
}

#[test]
fn mock_task_ignores_punctuation_when_tokenizing_for_overlap() {
    let task = MockDownstreamTask::new();
    let docs = vec!["Hello, world! This is great.".to_string()];
    let output = task.run("world", &docs);
    assert_eq!(output, "Hello, world");
}

// ── AggregationMethod / ERagConfig ────────────────────────────────────────────

#[test]
fn aggregation_method_default_is_mean() {
    assert_eq!(AggregationMethod::default(), AggregationMethod::Mean);
}

#[test]
fn config_default_values() {
    let cfg = ERagConfig::default();
    assert_eq!(cfg.aggregation, AggregationMethod::Mean);
    assert!((cfg.ndcg_log_base - 2.0).abs() < 1e-6);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(ERagConfig::new(), ERagConfig::default());
}

#[test]
fn config_with_aggregation_builder() {
    let cfg = ERagConfig::new().with_aggregation(AggregationMethod::Max);
    assert_eq!(cfg.aggregation, AggregationMethod::Max);
}

#[test]
fn config_with_ndcg_log_base_builder() {
    let cfg = ERagConfig::new().with_ndcg_log_base(3.0);
    assert!((cfg.ndcg_log_base - 3.0).abs() < 1e-6);
}

#[test]
fn config_validate_default_ok() {
    assert!(ERagConfig::default().validate().is_ok());
}

#[test]
fn config_validate_log_base_one_fails() {
    let cfg = ERagConfig::new().with_ndcg_log_base(1.0);
    assert!(matches!(cfg.validate(), Err(ERagError::InvalidConfig(_))));
}

#[test]
fn config_validate_log_base_negative_fails() {
    let cfg = ERagConfig::new().with_ndcg_log_base(-1.0);
    assert!(matches!(cfg.validate(), Err(ERagError::InvalidConfig(_))));
}

#[test]
fn config_validate_log_base_nan_fails() {
    let cfg = ERagConfig::new().with_ndcg_log_base(f32::NAN);
    assert!(matches!(cfg.validate(), Err(ERagError::InvalidConfig(_))));
}

#[test]
fn config_validate_log_base_infinite_fails() {
    let cfg = ERagConfig::new().with_ndcg_log_base(f32::INFINITY);
    assert!(matches!(cfg.validate(), Err(ERagError::InvalidConfig(_))));
}

#[test]
fn config_validate_log_base_large_ok() {
    let cfg = ERagConfig::new().with_ndcg_log_base(10.0);
    assert!(cfg.validate().is_ok());
}

// ── ERagError ─────────────────────────────────────────────────────────────────

#[test]
fn error_display_empty_query() {
    assert_eq!(ERagError::EmptyQuery.to_string(), "query is empty");
}

#[test]
fn error_display_empty_docs() {
    assert_eq!(
        ERagError::EmptyDocs.to_string(),
        "retrieved document set is empty"
    );
}

#[test]
fn error_display_empty_gold() {
    assert_eq!(ERagError::EmptyGold.to_string(), "gold answer is empty");
}

#[test]
fn error_display_invalid_config() {
    let err = ERagError::InvalidConfig("bad value".to_string());
    assert_eq!(err.to_string(), "invalid eRAG configuration: bad value");
}

#[test]
fn error_display_insufficient_cases() {
    let err = ERagError::InsufficientCases { got: 1, need: 2 };
    assert_eq!(
        err.to_string(),
        "insufficient cases for correlation: got 1, need at least 2"
    );
}

#[test]
fn error_equality() {
    assert_eq!(ERagError::EmptyQuery, ERagError::EmptyQuery);
    assert_ne!(ERagError::EmptyQuery, ERagError::EmptyDocs);
}

#[test]
fn error_clone() {
    let err = ERagError::InvalidConfig("x".to_string());
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

// ── PerDocScore / ERagReport / ERagCase ───────────────────────────────────────

#[test]
fn per_doc_score_fields() {
    let score = PerDocScore {
        doc_index: 2,
        utility: 0.5,
    };
    assert_eq!(score.doc_index, 2);
    assert!((score.utility - 0.5).abs() < 1e-6);
}

#[test]
fn report_best_doc_returns_max() {
    let report = ERagReport {
        per_doc_scores: vec![
            PerDocScore {
                doc_index: 0,
                utility: 0.2,
            },
            PerDocScore {
                doc_index: 1,
                utility: 0.9,
            },
            PerDocScore {
                doc_index: 2,
                utility: 0.5,
            },
        ],
        aggregated_score: 0.0,
        end_to_end_score: 0.0,
    };
    assert_eq!(report.best_doc().unwrap().doc_index, 1);
}

#[test]
fn report_best_doc_tie_breaks_to_lowest_index() {
    let report = ERagReport {
        per_doc_scores: vec![
            PerDocScore {
                doc_index: 0,
                utility: 0.7,
            },
            PerDocScore {
                doc_index: 1,
                utility: 0.7,
            },
        ],
        aggregated_score: 0.0,
        end_to_end_score: 0.0,
    };
    assert_eq!(report.best_doc().unwrap().doc_index, 0);
}

#[test]
fn report_best_doc_empty_returns_none() {
    let report = ERagReport {
        per_doc_scores: vec![],
        aggregated_score: 0.0,
        end_to_end_score: 0.0,
    };
    assert!(report.best_doc().is_none());
}

#[test]
fn case_new_constructs_fields() {
    let case = ERagCase::new("q", "gold", vec!["doc".to_string()]);
    assert_eq!(case.query, "q");
    assert_eq!(case.gold_answer, "gold");
    assert_eq!(case.retrieved_docs, vec!["doc".to_string()]);
}

// ── ERagEvaluator::evaluate — input validation ────────────────────────────────

#[test]
fn evaluate_errors_on_empty_query() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec!["doc".to_string()];
    let err = evaluator
        .evaluate("", "gold", &docs, &task, &utility)
        .unwrap_err();
    assert_eq!(err, ERagError::EmptyQuery);
}

#[test]
fn evaluate_errors_on_whitespace_query() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec!["doc".to_string()];
    let err = evaluator
        .evaluate("   ", "gold", &docs, &task, &utility)
        .unwrap_err();
    assert_eq!(err, ERagError::EmptyQuery);
}

#[test]
fn evaluate_errors_on_empty_docs() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs: Vec<String> = vec![];
    let err = evaluator
        .evaluate("query", "gold", &docs, &task, &utility)
        .unwrap_err();
    assert_eq!(err, ERagError::EmptyDocs);
}

#[test]
fn evaluate_errors_on_empty_gold() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec!["doc".to_string()];
    let err = evaluator
        .evaluate("query", "", &docs, &task, &utility)
        .unwrap_err();
    assert_eq!(err, ERagError::EmptyGold);
}

#[test]
fn evaluate_errors_on_whitespace_gold() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec!["doc".to_string()];
    let err = evaluator
        .evaluate("query", "   ", &docs, &task, &utility)
        .unwrap_err();
    assert_eq!(err, ERagError::EmptyGold);
}

#[test]
fn evaluate_errors_on_invalid_config() {
    let evaluator = ERagEvaluator::new(ERagConfig::default().with_ndcg_log_base(1.0));
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec!["doc".to_string()];
    let err = evaluator
        .evaluate("query", "gold", &docs, &task, &utility)
        .unwrap_err();
    assert!(matches!(err, ERagError::InvalidConfig(_)));
}

// ── ERagEvaluator::evaluate — per-document scoring correctness ───────────────

#[test]
fn evaluate_per_doc_scores_length_matches_docs() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec!["a.".to_string(), "b.".to_string(), "c.".to_string()];
    let report = evaluator
        .evaluate("q", "gold", &docs, &task, &utility)
        .unwrap();
    assert_eq!(report.per_doc_scores.len(), docs.len());
}

#[test]
fn evaluate_per_doc_indices_are_sequential() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec!["a.".to_string(), "b.".to_string(), "c.".to_string()];
    let report = evaluator
        .evaluate("q", "gold", &docs, &task, &utility)
        .unwrap();
    for (i, score) in report.per_doc_scores.iter().enumerate() {
        assert_eq!(score.doc_index, i);
    }
}

#[test]
fn evaluate_relevant_doc_scores_higher_than_irrelevant() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec![
        "The Eiffel Tower is located in Paris, France.".to_string(),
        "Bananas are a good source of potassium.".to_string(),
    ];
    let report = evaluator
        .evaluate(
            "Where is the Eiffel Tower?",
            "The Eiffel Tower is in Paris.",
            &docs,
            &task,
            &utility,
        )
        .unwrap();
    assert!(report.per_doc_scores[0].utility > report.per_doc_scores[1].utility);
    assert!(report.per_doc_scores[1].utility.abs() < 1e-6);
}

#[test]
fn evaluate_end_to_end_score_differs_from_per_doc_aggregate() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let docs = vec![
        "The Eiffel Tower is located in Paris, France.".to_string(),
        "Bananas are a good source of potassium.".to_string(),
    ];
    let report = evaluator
        .evaluate(
            "Where is the Eiffel Tower?",
            "The Eiffel Tower is in Paris.",
            &docs,
            &task,
            &utility,
        )
        .unwrap();
    assert!((report.end_to_end_score - report.aggregated_score).abs() > 1e-3);
}

// ── ERagEvaluator::evaluate — aggregation methods ─────────────────────────────

#[test]
fn evaluate_aggregation_mean_matches_manual_average() {
    let evaluator =
        ERagEvaluator::new(ERagConfig::default().with_aggregation(AggregationMethod::Mean));
    let task = IdentityTask;
    let utility = FixedUtility(0.6);
    let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let report = evaluator
        .evaluate("q", "gold", &docs, &task, &utility)
        .unwrap();
    assert!((report.aggregated_score - 0.6).abs() < 1e-6);
}

#[test]
fn evaluate_aggregation_sum_matches_total() {
    let evaluator =
        ERagEvaluator::new(ERagConfig::default().with_aggregation(AggregationMethod::Sum));
    let task = IdentityTask;
    let utility = FixedUtility(0.6);
    let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let report = evaluator
        .evaluate("q", "gold", &docs, &task, &utility)
        .unwrap();
    assert!((report.aggregated_score - 1.8).abs() < 1e-5);
}

#[test]
fn evaluate_aggregation_max_matches_highest() {
    let evaluator =
        ERagEvaluator::new(ERagConfig::default().with_aggregation(AggregationMethod::Max));
    let task = IdentityTask;
    let mut lookup = HashMap::new();
    lookup.insert("a".to_string(), 0.2_f32);
    lookup.insert("b".to_string(), 0.9_f32);
    lookup.insert("c".to_string(), 0.5_f32);
    let utility = LookupUtility(lookup);
    let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let report = evaluator
        .evaluate("q", "gold", &docs, &task, &utility)
        .unwrap();
    assert!((report.aggregated_score - 0.9).abs() < 1e-6);
}

#[test]
fn evaluate_aggregation_ndcg_weighted_equal_scores_returns_that_score() {
    let evaluator =
        ERagEvaluator::new(ERagConfig::default().with_aggregation(AggregationMethod::NdcgWeighted));
    let task = IdentityTask;
    let utility = FixedUtility(0.6);
    let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let report = evaluator
        .evaluate("q", "gold", &docs, &task, &utility)
        .unwrap();
    assert!((report.aggregated_score - 0.6).abs() < 1e-5);
}

#[test]
fn evaluate_aggregation_ndcg_weighted_differing_scores_exceeds_mean() {
    let mut lookup = HashMap::new();
    lookup.insert("a".to_string(), 0.9_f32);
    lookup.insert("b".to_string(), 0.3_f32);
    let docs = vec!["a".to_string(), "b".to_string()];

    let mean_evaluator =
        ERagEvaluator::new(ERagConfig::default().with_aggregation(AggregationMethod::Mean));
    let ndcg_evaluator =
        ERagEvaluator::new(ERagConfig::default().with_aggregation(AggregationMethod::NdcgWeighted));
    let task = IdentityTask;
    let utility = LookupUtility(lookup);

    let mean_report = mean_evaluator
        .evaluate("q", "gold", &docs, &task, &utility)
        .unwrap();
    let ndcg_report = ndcg_evaluator
        .evaluate("q", "gold", &docs, &task, &utility)
        .unwrap();

    assert!(ndcg_report.aggregated_score > mean_report.aggregated_score);
    assert!(ndcg_report.aggregated_score < 0.9 + 1e-6);
}

#[test]
fn evaluate_single_doc_all_aggregations_equal_that_doc_score() {
    let task = IdentityTask;
    let utility = FixedUtility(0.42);
    let docs = vec!["only".to_string()];
    for method in [
        AggregationMethod::Mean,
        AggregationMethod::Max,
        AggregationMethod::Sum,
        AggregationMethod::NdcgWeighted,
    ] {
        let evaluator = ERagEvaluator::new(ERagConfig::default().with_aggregation(method));
        let report = evaluator
            .evaluate("q", "gold", &docs, &task, &utility)
            .unwrap();
        assert!(
            (report.aggregated_score - 0.42).abs() < 1e-5,
            "{method:?}: got {}",
            report.aggregated_score
        );
    }
}

// ── ERagEvaluator::evaluate_batch ─────────────────────────────────────────────

#[test]
fn evaluate_batch_errors_on_zero_cases() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let err = evaluator.evaluate_batch(&[], &task, &utility).unwrap_err();
    assert_eq!(err, ERagError::InsufficientCases { got: 0, need: 2 });
}

#[test]
fn evaluate_batch_errors_on_one_case() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let cases = vec![ERagCase::new("q", "gold", vec!["doc".to_string()])];
    let err = evaluator
        .evaluate_batch(&cases, &task, &utility)
        .unwrap_err();
    assert_eq!(err, ERagError::InsufficientCases { got: 1, need: 2 });
}

#[test]
fn evaluate_batch_reports_len_matches_cases() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let cases = vec![
        ERagCase::new("q1", "gold1", vec!["doc a.".to_string()]),
        ERagCase::new("q2", "gold2", vec!["doc b.".to_string()]),
        ERagCase::new("q3", "gold3", vec!["doc c.".to_string()]),
    ];
    let batch = evaluator.evaluate_batch(&cases, &task, &utility).unwrap();
    assert_eq!(batch.per_case_reports.len(), cases.len());
}

#[test]
fn evaluate_batch_propagates_case_error() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let cases = vec![
        ERagCase::new("q1", "gold1", vec!["doc a.".to_string()]),
        ERagCase::new("q2", "", vec!["doc b.".to_string()]),
    ];
    let err = evaluator
        .evaluate_batch(&cases, &task, &utility)
        .unwrap_err();
    assert_eq!(err, ERagError::EmptyGold);
}

#[test]
fn evaluate_batch_kendall_and_spearman_in_valid_range() {
    let evaluator = default_evaluator();
    let task = MockDownstreamTask::new();
    let utility = RougeLiteUtility::new();
    let cases = vec![
        ERagCase::new(
            "Where is the Eiffel Tower?",
            "The Eiffel Tower is in Paris.",
            vec![
                "The Eiffel Tower is located in Paris, France.".to_string(),
                "Bananas are tasty.".to_string(),
            ],
        ),
        ERagCase::new(
            "What do bananas contain?",
            "Bananas contain potassium.",
            vec![
                "Bananas are a good source of potassium.".to_string(),
                "The moon orbits the Earth.".to_string(),
            ],
        ),
        ERagCase::new(
            "Where do penguins live?",
            "Penguins live in Antarctica.",
            vec![
                "Penguins live in Antarctica and other cold regions.".to_string(),
                "Coffee is a popular beverage.".to_string(),
            ],
        ),
    ];
    let batch = evaluator.evaluate_batch(&cases, &task, &utility).unwrap();
    assert!((-1.0..=1.0).contains(&batch.kendall_tau));
    assert!((-1.0..=1.0).contains(&batch.spearman_rho));
}

// ── Kendall's tau-b ───────────────────────────────────────────────────────────

#[test]
fn kendall_tau_perfect_agreement() {
    let xs = [1.0, 2.0, 3.0, 4.0];
    let ys = [10.0, 20.0, 30.0, 40.0];
    assert!((kendall_tau(&xs, &ys) - 1.0).abs() < 1e-6);
}

#[test]
fn kendall_tau_perfect_disagreement() {
    let xs = [1.0, 2.0, 3.0, 4.0];
    let ys = [4.0, 3.0, 2.0, 1.0];
    assert!((kendall_tau(&xs, &ys) - (-1.0)).abs() < 1e-6);
}

#[test]
fn kendall_tau_no_correlation() {
    let xs = [1.0, 2.0, 3.0, 4.0];
    let ys = [2.0, 4.0, 1.0, 3.0];
    assert!(kendall_tau(&xs, &ys).abs() < 1e-6);
}

#[test]
fn kendall_tau_with_ties_is_finite() {
    let xs = [1.0, 1.0, 2.0, 2.0];
    let ys = [1.0, 2.0, 1.0, 2.0];
    let tau = kendall_tau(&xs, &ys);
    assert!(tau.is_finite());
    assert!(tau.abs() < 1e-6);
}

#[test]
fn kendall_tau_all_x_tied_returns_zero() {
    let xs = [5.0, 5.0, 5.0];
    let ys = [1.0, 2.0, 3.0];
    assert_eq!(kendall_tau(&xs, &ys), 0.0);
}

#[test]
fn kendall_tau_mismatched_lengths_returns_zero() {
    let xs = [1.0, 2.0, 3.0];
    let ys = [1.0, 2.0];
    assert_eq!(kendall_tau(&xs, &ys), 0.0);
}

#[test]
fn kendall_tau_too_few_elements_returns_zero() {
    let xs = [1.0];
    let ys = [1.0];
    assert_eq!(kendall_tau(&xs, &ys), 0.0);
}

#[test]
fn kendall_tau_empty_returns_zero() {
    let xs: [f32; 0] = [];
    let ys: [f32; 0] = [];
    assert_eq!(kendall_tau(&xs, &ys), 0.0);
}

// ── Spearman's rho ────────────────────────────────────────────────────────────

#[test]
fn spearman_rho_perfect_agreement() {
    let xs = [1.0, 2.0, 3.0, 4.0];
    let ys = [10.0, 20.0, 30.0, 40.0];
    assert!((spearman_rho(&xs, &ys) - 1.0).abs() < 1e-6);
}

#[test]
fn spearman_rho_perfect_disagreement() {
    let xs = [1.0, 2.0, 3.0, 4.0];
    let ys = [4.0, 3.0, 2.0, 1.0];
    assert!((spearman_rho(&xs, &ys) - (-1.0)).abs() < 1e-6);
}

#[test]
fn spearman_rho_no_correlation() {
    let xs = [1.0, 2.0, 3.0, 4.0];
    let ys = [2.0, 4.0, 1.0, 3.0];
    assert!(spearman_rho(&xs, &ys).abs() < 1e-6);
}

#[test]
fn spearman_rho_with_ties() {
    let xs = [1.0, 1.0, 2.0, 2.0];
    let ys = [1.0, 2.0, 1.0, 2.0];
    assert!(spearman_rho(&xs, &ys).abs() < 1e-6);
}

#[test]
fn spearman_rho_zero_variance_returns_zero() {
    let xs = [3.0, 3.0, 3.0];
    let ys = [1.0, 2.0, 3.0];
    assert_eq!(spearman_rho(&xs, &ys), 0.0);
}

#[test]
fn spearman_rho_mismatched_lengths_returns_zero() {
    let xs = [1.0, 2.0, 3.0];
    let ys = [1.0, 2.0];
    assert_eq!(spearman_rho(&xs, &ys), 0.0);
}

#[test]
fn spearman_rho_too_few_elements_returns_zero() {
    let xs = [1.0];
    let ys = [2.0];
    assert_eq!(spearman_rho(&xs, &ys), 0.0);
}

#[test]
fn spearman_rho_stays_within_bounds_under_fp_rounding_noise() {
    // These particular values are close enough to a perfectly monotone
    // relationship that naive f32 Pearson-on-ranks arithmetic can overshoot
    // 1.0 by a single rounding-error ULP; `pearson`'s clamp must absorb that.
    let xs = [0.428_571_43_f32, 0.2_f32, 0.333_333_34_f32];
    let ys = [0.705_882_4_f32, 0.266_666_68_f32, 0.470_588_27_f32];
    let rho = spearman_rho(&xs, &ys);
    assert!((-1.0..=1.0).contains(&rho));
    assert!((rho - 1.0).abs() < 1e-6);
}
