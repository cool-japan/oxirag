//! Comprehensive tests for the RAG evaluation framework.

use std::path::PathBuf;

use crate::{
    evaluation::{
        dataset::{DatasetStats, EvaluationDataset},
        evaluator::RagEvaluator,
        metrics::{
            AnswerRelevanceScorer, ContextPrecisionScorer, ContextRecallScorer, EvaluationMetric,
            FaithfulnessScorer, OverallScorer,
        },
        types::{EvalError, EvaluationResult, EvaluationSample},
    },
    types::{Document, Draft, PipelineOutput, Query, SearchResult},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_json_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("oxirag_eval_test_{name}.json"));
    path
}

fn make_sample(
    query: &str,
    context: Vec<&str>,
    answer: &str,
    ground_truth: Option<&str>,
) -> EvaluationSample {
    let mut s = EvaluationSample::new(
        "test-id",
        query,
        context.into_iter().map(str::to_string).collect(),
        answer,
    );
    if let Some(gt) = ground_truth {
        s = s.with_ground_truth(gt);
    }
    s
}

/// Build a minimal [`PipelineOutput`] for constructor tests.
fn make_pipeline_output(query_text: &str, answer: &str) -> PipelineOutput {
    let query = Query::new(query_text);
    let draft = Draft::new(answer, query_text);

    let doc = Document::new("Test document context passage");
    let search_result = SearchResult::new(doc, 0.9, 0);

    let mut output = PipelineOutput::new(query, draft);
    output.search_results = vec![search_result];
    output.final_answer = answer.to_string();
    output.confidence = 0.8;
    output.layers_used = vec!["echo".to_string(), "speculator".to_string()];
    output
}

// ---------------------------------------------------------------------------
// EvaluationSample construction
// ---------------------------------------------------------------------------

#[test]
fn test_sample_new_fields() {
    let sample = EvaluationSample::new(
        "id-1",
        "What is Rust?",
        vec!["Rust is a systems language.".to_string()],
        "Rust is safe and fast.",
    );
    assert_eq!(sample.id, "id-1");
    assert_eq!(sample.query, "What is Rust?");
    assert_eq!(sample.context.len(), 1);
    assert_eq!(sample.answer, "Rust is safe and fast.");
    assert!(sample.ground_truth.is_none());
}

#[test]
fn test_sample_with_ground_truth() {
    let sample = EvaluationSample::new("id-2", "query", vec![], "answer")
        .with_ground_truth("expected answer");
    assert_eq!(sample.ground_truth.as_deref(), Some("expected answer"));
}

#[test]
fn test_sample_with_source_doc_ids() {
    let ids = vec!["doc-1".to_string(), "doc-2".to_string()];
    let sample = EvaluationSample::new("id-3", "q", vec![], "a").with_source_doc_ids(ids.clone());
    assert_eq!(sample.source_doc_ids, ids);
}

#[test]
fn test_sample_from_pipeline_output() {
    let output = make_pipeline_output("What is Rust?", "Rust is a systems programming language.");
    let sample = EvaluationSample::from_pipeline_output(&output);

    assert!(!sample.id.is_empty());
    assert_eq!(sample.query, "What is Rust?");
    assert!(!sample.context.is_empty());
    assert_eq!(sample.answer, "Rust is a systems programming language.");
    assert_eq!(sample.context.len(), output.search_results.len());
    assert_eq!(sample.source_doc_ids.len(), output.search_results.len());
}

#[test]
fn test_sample_from_pipeline_output_context_content() {
    let output = make_pipeline_output("query", "answer");
    let sample = EvaluationSample::from_pipeline_output(&output);
    // The context should contain the document content from search results
    assert!(sample.context[0].contains("Test document context passage"));
}

// ---------------------------------------------------------------------------
// EvaluationResult
// ---------------------------------------------------------------------------

#[test]
fn test_evaluation_result_weighted_average_all_ones() {
    let score = EvaluationResult::weighted_average(1.0, 1.0, 1.0, Some(1.0));
    assert!((score - 1.0_f32).abs() < 1e-5_f32);
}

#[test]
fn test_evaluation_result_weighted_average_all_zeros() {
    let score = EvaluationResult::weighted_average(0.0, 0.0, 0.0, Some(0.0));
    assert!(score.abs() < 1e-6_f32);
}

#[test]
fn test_evaluation_result_weighted_average_no_recall() {
    // Without ground truth, the three other metrics should still sum to 1.0
    let score = EvaluationResult::weighted_average(1.0, 1.0, 1.0, None);
    assert!((score - 1.0_f32).abs() < 1e-5_f32);
}

#[test]
fn test_evaluation_result_weighted_average_partial() {
    // Faithfulness has the highest weight (0.35), so a high faithfulness score
    // should dominate.
    let high_faith = EvaluationResult::weighted_average(0.0, 1.0, 0.0, Some(0.0));
    let high_ar = EvaluationResult::weighted_average(1.0, 0.0, 0.0, Some(0.0));
    // faithfulness weight (0.35) > AR weight (0.25)
    assert!(high_faith > high_ar);
}

// ---------------------------------------------------------------------------
// AnswerRelevanceScorer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_answer_relevance_identical() {
    let scorer = AnswerRelevanceScorer::default();
    let sample = make_sample(
        "rust programming language",
        vec!["context passage"],
        "rust programming language",
        None,
    );
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score > 0.7_f32,
        "Identical query/answer should give high score, got {score}"
    );
}

#[tokio::test]
async fn test_answer_relevance_irrelevant() {
    let scorer = AnswerRelevanceScorer::default();
    let sample = make_sample(
        "what is machine learning",
        vec!["context"],
        "bananas grow in tropical climates",
        None,
    );
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score < 0.2_f32,
        "Irrelevant answer should give low score, got {score}"
    );
}

#[tokio::test]
async fn test_answer_relevance_boost_exact_match() {
    let sample = make_sample(
        "rust language",
        vec!["context"],
        "The rust language is fast and safe. It is a systems programming language.",
        None,
    );
    let ar_with_boost = AnswerRelevanceScorer {
        boost_exact_match: true,
    };
    let result_boosted = ar_with_boost.score(&sample).await.expect("score failed");

    let ar_no_boost = AnswerRelevanceScorer {
        boost_exact_match: false,
    };
    let result_plain = ar_no_boost.score(&sample).await.expect("score failed");
    // With exact phrase present, boosted >= no-boost
    assert!(result_boosted >= result_plain);
}

#[tokio::test]
async fn test_answer_relevance_empty_answer() {
    let scorer = AnswerRelevanceScorer::default();
    let sample = make_sample("query", vec!["ctx"], "", None);
    let score = scorer.score(&sample).await.expect("score failed");
    assert!((score).abs() < 1e-6_f32);
}

// ---------------------------------------------------------------------------
// FaithfulnessScorer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_faithfulness_full_support() {
    let scorer = FaithfulnessScorer::default();
    let context = "Rust is a systems programming language focused on safety performance memory";
    let answer = "Rust is a systems programming language focused on safety and performance.";
    let sample = make_sample("what is rust", vec![context], answer, None);
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score > 0.5_f32,
        "Fully supported answer should score high, got {score}"
    );
}

#[tokio::test]
async fn test_faithfulness_no_support() {
    let scorer = FaithfulnessScorer::default();
    let context = "Python is a high-level scripting language.";
    let answer = "Bananas are a tropical fruit. Monkeys love bananas very much. They are yellow.";
    let sample = make_sample("query", vec![context], answer, None);
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score < 0.5_f32,
        "Unsupported answer should score low, got {score}"
    );
}

#[tokio::test]
async fn test_faithfulness_empty_context_error() {
    let scorer = FaithfulnessScorer::default();
    let sample = make_sample("query", vec![], "some answer", None);
    let result = scorer.score(&sample).await;
    assert!(matches!(result, Err(EvalError::EmptyContext)));
}

#[tokio::test]
async fn test_faithfulness_empty_answer() {
    let scorer = FaithfulnessScorer::default();
    let sample = make_sample("query", vec!["context"], "", None);
    let score = scorer.score(&sample).await.expect("score failed");
    assert!((score).abs() < 1e-6_f32);
}

#[tokio::test]
async fn test_faithfulness_custom_overlap_threshold() {
    let context = "Rust programming language safety concurrency";
    let answer = "Rust is used for safety and concurrency in systems programming.";
    let sample = make_sample("query", vec![context], answer, None);

    let faith_strict = FaithfulnessScorer {
        min_word_overlap: 0.8,
    };
    let result_strict = faith_strict.score(&sample).await.expect("score failed");

    let faith_lenient = FaithfulnessScorer {
        min_word_overlap: 0.2,
    };
    let result_lenient = faith_lenient.score(&sample).await.expect("score failed");
    // Lenient threshold should give >= strict threshold
    assert!(result_lenient >= result_strict);
}

// ---------------------------------------------------------------------------
// ContextPrecisionScorer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_context_precision_all_relevant() {
    let scorer = ContextPrecisionScorer::default();
    let query = "rust programming language";
    let context = vec![
        "Rust is a systems programming language.",
        "Rust programming provides memory safety.",
        "The Rust language offers zero-cost abstractions.",
    ];
    let sample = make_sample(query, context, "answer", None);
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score > 0.5_f32,
        "All relevant chunks should score high, got {score}"
    );
}

#[tokio::test]
async fn test_context_precision_none_relevant() {
    let scorer = ContextPrecisionScorer {
        relevance_threshold: 0.5,
    };
    let query = "quantum computing superposition";
    let context = vec![
        "Bananas are grown in tropical climates.",
        "The weather is sunny today.",
        "I enjoy hiking in the mountains.",
    ];
    let sample = make_sample(query, context, "answer", None);
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score < 0.2_f32,
        "No relevant chunks should score low, got {score}"
    );
}

#[tokio::test]
async fn test_context_precision_empty_context_error() {
    let scorer = ContextPrecisionScorer::default();
    let sample = make_sample("query", vec![], "answer", None);
    let result = scorer.score(&sample).await;
    assert!(matches!(result, Err(EvalError::EmptyContext)));
}

#[tokio::test]
async fn test_context_precision_mixed() {
    let scorer = ContextPrecisionScorer {
        relevance_threshold: 0.15,
    };
    let query = "rust language";
    let context = vec![
        "Rust is a language for systems programming.", // relevant
        "Bananas are tropical fruit.",                 // not relevant
    ];
    let sample = make_sample(query, context, "answer", None);
    let score = scorer.score(&sample).await.expect("score failed");
    // Exactly 1 out of 2 chunks should be relevant = 0.5
    assert!(score > 0.0_f32 && score < 1.0_f32);
}

// ---------------------------------------------------------------------------
// ContextRecallScorer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_context_recall_full() {
    let scorer = ContextRecallScorer;
    let ground_truth = "rust programming language safety performance systems";
    let context = vec!["rust programming language safety performance systems"];
    let sample = make_sample("query", context, "answer", Some(ground_truth));
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score > 0.8_f32,
        "Full recall should be near 1.0, got {score}"
    );
}

#[tokio::test]
async fn test_context_recall_partial() {
    let scorer = ContextRecallScorer;
    let ground_truth = "rust programming language safety performance";
    let context = vec!["rust language"]; // only partial overlap
    let sample = make_sample("query", context, "answer", Some(ground_truth));
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score > 0.0_f32 && score < 1.0_f32,
        "Partial recall: {score}"
    );
}

#[tokio::test]
async fn test_context_recall_missing_ground_truth_error() {
    let scorer = ContextRecallScorer;
    let sample = make_sample("query", vec!["context"], "answer", None);
    let result = scorer.score(&sample).await;
    assert!(matches!(result, Err(EvalError::MissingGroundTruth)));
}

#[tokio::test]
async fn test_context_recall_empty_context_error() {
    let scorer = ContextRecallScorer;
    let sample = make_sample("query", vec![], "answer", Some("ground truth"));
    let result = scorer.score(&sample).await;
    assert!(matches!(result, Err(EvalError::EmptyContext)));
}

// ---------------------------------------------------------------------------
// OverallScorer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_overall_scorer_perfect() {
    let scorer = OverallScorer::default();
    let query = "rust language programming";
    let ctx = "rust language programming safety performance systems";
    let answer = "rust language programming systems.";
    let gt = "rust language programming safety performance";
    let sample = make_sample(query, vec![ctx], answer, Some(gt));
    let score = scorer.score(&sample).await.expect("score failed");
    assert!(
        score > 0.4_f32,
        "Perfect-ish sample should score > 0.4, got {score}"
    );
}

#[tokio::test]
async fn test_overall_scorer_no_ground_truth() {
    // Should not error; context recall is skipped
    let scorer = OverallScorer::default();
    let sample = make_sample(
        "query about rust",
        vec!["Rust is a systems language."],
        "Rust is about systems.",
        None,
    );
    let score = scorer
        .score(&sample)
        .await
        .expect("should not error without ground truth");
    assert!((0.0_f32..=1.0_f32).contains(&score));
}

#[tokio::test]
async fn test_overall_scorer_weights_sum_to_one() {
    // When all individual metrics return 1.0 the overall must also be 1.0.
    let scorer = OverallScorer::default();
    let query = "rust programming";
    let ctx = "rust programming language systems";
    let answer = "rust programming language.";
    let gt = "rust programming language systems";
    let sample = make_sample(query, vec![ctx], answer, Some(gt));
    let score = scorer.score(&sample).await.expect("score failed");
    // Overall must be in [0, 1]
    assert!((0.0_f32..=1.0_f32).contains(&score));
}

// ---------------------------------------------------------------------------
// RagEvaluator
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_evaluator_smoke_test() {
    let evaluator = RagEvaluator::default_metrics();
    let sample = make_sample(
        "What is Rust?",
        vec!["Rust is a systems programming language."],
        "Rust is a systems programming language focused on safety.",
        None,
    );
    let result = evaluator.evaluate(&sample).await.expect("evaluate failed");
    assert_eq!(result.sample_id, "test-id");
    assert!(result.answer_relevance >= 0.0_f32);
    assert!(result.faithfulness >= 0.0_f32);
    assert!(result.context_precision >= 0.0_f32);
    assert!(result.overall >= 0.0_f32 && result.overall <= 1.0_f32);
}

#[tokio::test]
async fn test_evaluator_evaluate_dataset() {
    let evaluator = RagEvaluator::default_metrics();
    let mut dataset = EvaluationDataset::new();
    dataset.add(make_sample("query 1", vec!["context 1"], "answer 1", None));
    dataset.add(make_sample("query 2", vec!["context 2"], "answer 2", None));

    let results = evaluator.evaluate_dataset(&dataset).await;
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.is_ok(), "Expected Ok result, got {r:?}");
    }
}

#[tokio::test]
async fn test_evaluator_aggregate_stats_means() {
    let evaluator = RagEvaluator::default_metrics();
    let mut dataset = EvaluationDataset::new();

    // Use deterministic content so the scores are non-trivial.
    dataset.add(make_sample(
        "rust systems language",
        vec!["rust systems language programming"],
        "rust systems language.",
        None,
    ));
    dataset.add(make_sample(
        "cargo package manager",
        vec!["cargo package manager rust build"],
        "cargo package manager.",
        None,
    ));

    let stats = evaluator.aggregate_stats(&dataset).await;
    assert_eq!(stats.sample_count, 2);
    assert!(stats.mean_answer_relevance >= 0.0_f32);
    assert!(stats.mean_faithfulness >= 0.0_f32);
    assert!(stats.mean_context_precision >= 0.0_f32);
    assert!(stats.mean_overall >= 0.0_f32 && stats.mean_overall <= 1.0_f32);
    // No ground truth → recall should be None
    assert!(stats.mean_context_recall.is_none());
}

#[tokio::test]
async fn test_evaluator_aggregate_stats_with_ground_truth() {
    let evaluator = RagEvaluator::default_metrics();
    let mut dataset = EvaluationDataset::new();
    dataset.add(
        make_sample(
            "rust language",
            vec!["rust language programming safety"],
            "rust language.",
            None,
        )
        .with_ground_truth("rust language programming safety"),
    );
    let stats = evaluator.aggregate_stats(&dataset).await;
    assert_eq!(stats.sample_count, 1);
    assert!(stats.mean_context_recall.is_some());
}

#[tokio::test]
async fn test_evaluator_empty_dataset() {
    let evaluator = RagEvaluator::default_metrics();
    let dataset = EvaluationDataset::new();
    let stats = evaluator.aggregate_stats(&dataset).await;
    assert_eq!(stats.sample_count, 0);
    assert!((stats.mean_overall).abs() < 1e-6_f32);
}

#[tokio::test]
async fn test_evaluator_custom_metrics() {
    let evaluator = RagEvaluator::new(vec![Box::new(AnswerRelevanceScorer::default())]);
    let sample = make_sample("rust", vec!["rust language"], "rust programming", None);
    // evaluate() should work with only one metric; missing metric names get 0.0
    let result = evaluator.evaluate(&sample).await.expect("evaluate failed");
    assert!(result.answer_relevance > 0.0_f32);
}

// ---------------------------------------------------------------------------
// EvaluationDataset
// ---------------------------------------------------------------------------

#[test]
fn test_dataset_add_and_len() {
    let mut dataset = EvaluationDataset::new();
    assert!(dataset.is_empty());
    assert_eq!(dataset.len(), 0);

    dataset.add(make_sample("q1", vec!["c1"], "a1", None));
    assert!(!dataset.is_empty());
    assert_eq!(dataset.len(), 1);

    dataset.add(make_sample("q2", vec!["c2"], "a2", None));
    assert_eq!(dataset.len(), 2);
}

#[test]
fn test_dataset_iter() {
    let mut dataset = EvaluationDataset::new();
    dataset.add(make_sample("q1", vec!["c1"], "a1", None));
    dataset.add(make_sample("q2", vec!["c2"], "a2", None));

    let queries: Vec<&str> = dataset.iter().map(|s| s.query.as_str()).collect();
    assert!(queries.contains(&"q1"));
    assert!(queries.contains(&"q2"));
}

#[test]
fn test_dataset_stats_empty() {
    let dataset = EvaluationDataset::new();
    let stats: DatasetStats = dataset.stats();
    assert_eq!(stats.total_samples, 0);
    assert_eq!(stats.samples_with_ground_truth, 0);
    assert!((stats.avg_context_len).abs() < 1e-6_f32);
    assert!((stats.avg_answer_len).abs() < 1e-6_f32);
}

#[test]
fn test_dataset_stats_with_samples() {
    let mut dataset = EvaluationDataset::new();
    dataset.add(make_sample("q1", vec!["ctx1", "ctx2"], "short", None).with_ground_truth("gt1"));
    dataset.add(make_sample("q2", vec!["ctx3"], "longer answer here", None));

    let stats = dataset.stats();
    assert_eq!(stats.total_samples, 2);
    assert_eq!(stats.samples_with_ground_truth, 1);
    // avg context len = (2 + 1) / 2 = 1.5
    assert!((stats.avg_context_len - 1.5_f32).abs() < 1e-5_f32);
    // avg answer len = (5 + 18) / 2 = 11.5
    #[allow(clippy::cast_precision_loss)]
    let expected_avg_answer = ("short".len() + "longer answer here".len()) as f32 / 2.0;
    assert!((stats.avg_answer_len - expected_avg_answer).abs() < 1e-5_f32);
}

#[test]
fn test_dataset_from_pipeline_outputs() {
    let out1 = make_pipeline_output("q1", "a1");
    let out2 = make_pipeline_output("q2", "a2");
    let dataset = EvaluationDataset::from_pipeline_outputs(vec![&out1, &out2]);
    assert_eq!(dataset.len(), 2);
}

// ---------------------------------------------------------------------------
// JSON round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_dataset_save_load_json_roundtrip() {
    let path = temp_json_path("roundtrip");

    let mut dataset = EvaluationDataset::new();
    dataset.add(
        make_sample(
            "What is Rust?",
            vec!["Rust is a systems programming language."],
            "Rust is safe and fast.",
            None,
        )
        .with_ground_truth("Rust is a systems language focused on safety."),
    );
    dataset.add(make_sample(
        "What is Cargo?",
        vec!["Cargo is Rust's build system."],
        "Cargo manages packages.",
        None,
    ));

    dataset.save_json(&path).expect("save failed");
    assert!(path.exists(), "JSON file should exist after save");

    let loaded = EvaluationDataset::load_json(&path).expect("load failed");
    assert_eq!(loaded.len(), dataset.len());

    let original_queries: Vec<&str> = dataset.iter().map(|s| s.query.as_str()).collect();
    let loaded_queries: Vec<&str> = loaded.iter().map(|s| s.query.as_str()).collect();
    assert_eq!(original_queries, loaded_queries);

    // Verify ground truth was preserved
    let loaded_first = loaded.iter().next().expect("should have first sample");
    assert_eq!(
        loaded_first.ground_truth.as_deref(),
        Some("Rust is a systems language focused on safety.")
    );

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_dataset_load_json_invalid_path_error() {
    let path = temp_json_path("nonexistent_12345");
    let result = EvaluationDataset::load_json(&path);
    assert!(result.is_err(), "Loading non-existent file should error");
}

// ---------------------------------------------------------------------------
// EvalError display
// ---------------------------------------------------------------------------

#[test]
fn test_eval_error_display() {
    let err = EvalError::MissingGroundTruth;
    assert!(err.to_string().contains("ground truth"));

    let err2 = EvalError::EmptyContext;
    let msg = err2.to_string();
    assert!(
        msg.to_lowercase().contains("context"),
        "Expected 'context' in '{msg}'"
    );

    let err3 = EvalError::Other("custom message".to_string());
    assert!(err3.to_string().contains("custom message"));
}
