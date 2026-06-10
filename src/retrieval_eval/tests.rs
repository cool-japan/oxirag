use crate::retrieval_eval::metrics::{
    average_precision, dcg_at_k, f1_at_k, hit_rate_at_k, mrr, ndcg_at_k, precision_at_k,
    recall_at_k, reciprocal_rank,
};
use crate::retrieval_eval::types::{
    AggregateScores, Qrels, RelevanceJudgment, RetrievalEvalConfig, RetrievalEvalError,
    RetrievalEvaluator, RetrievalScores,
};
use crate::types::{Document, DocumentId, SearchResult};

// ── Helpers ───────────────────────────────────────────────────────────────

fn make_result(id: &str, score: f32, rank: usize) -> SearchResult {
    SearchResult::new(
        Document::new(format!("content for {id}")).with_id(DocumentId::from_string(id)),
        score,
        rank,
    )
}

fn qrels_binary(ids: &[&str]) -> Qrels {
    Qrels::from_judgments(
        ids.iter()
            .map(|id| RelevanceJudgment::relevant(id.to_string()))
            .collect(),
    )
}

// ── RelevanceJudgment ─────────────────────────────────────────────────────

#[test]
fn relevance_judgment_relevant_gain_is_one() {
    let j = RelevanceJudgment::relevant("d1");
    assert!((j.gain - 1.0_f32).abs() < 1e-6);
}

#[test]
fn relevance_judgment_relevant_doc_id() {
    let j = RelevanceJudgment::relevant("doc42");
    assert_eq!(j.doc_id, "doc42");
}

#[test]
fn relevance_judgment_graded_gain() {
    let j = RelevanceJudgment::graded("d1", 2.5);
    assert!((j.gain - 2.5_f32).abs() < 1e-6);
}

#[test]
fn relevance_judgment_graded_doc_id() {
    let j = RelevanceJudgment::graded("myDoc", 0.75);
    assert_eq!(j.doc_id, "myDoc");
}

// ── Qrels ─────────────────────────────────────────────────────────────────

#[test]
fn qrels_from_judgments_builds_map() {
    let q = qrels_binary(&["d1", "d2"]);
    assert!((q.gain("d1") - 1.0_f32).abs() < 1e-6);
    assert!((q.gain("d2") - 1.0_f32).abs() < 1e-6);
}

#[test]
fn qrels_is_relevant_true_at_threshold() {
    let q = qrels_binary(&["d1"]);
    assert!(q.is_relevant("d1", 1.0));
}

#[test]
fn qrels_is_relevant_false_above_gain() {
    let q = qrels_binary(&["d1"]); // gain = 1.0
    assert!(!q.is_relevant("d1", 2.0));
}

#[test]
fn qrels_is_relevant_false_missing() {
    let q = qrels_binary(&["d1"]);
    assert!(!q.is_relevant("missing", 1.0));
}

#[test]
fn qrels_gain_returns_correct_value() {
    let j = Qrels::from_judgments(vec![RelevanceJudgment::graded("d1", 3.0)]);
    assert!((j.gain("d1") - 3.0_f32).abs() < 1e-6);
}

#[test]
fn qrels_gain_missing_returns_zero() {
    let q = qrels_binary(&["d1"]);
    assert!((q.gain("missing") - 0.0_f32).abs() < 1e-6);
}

#[test]
fn qrels_is_relevant_graded_threshold_satisfied() {
    let j = Qrels::from_judgments(vec![RelevanceJudgment::graded("d1", 2.0)]);
    assert!(j.is_relevant("d1", 2.0));
    assert!(!j.is_relevant("d1", 2.1));
}

// ── precision_at_k ────────────────────────────────────────────────────────

#[test]
fn precision_at_k_zero_k_returns_zero() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let p = precision_at_k(&results, &q, 0, 1.0);
    assert!((p - 0.0_f32).abs() < 1e-6);
}

#[test]
fn precision_at_k_all_relevant_perfect() {
    let results = vec![
        make_result("d1", 0.9, 0),
        make_result("d2", 0.8, 1),
        make_result("d3", 0.7, 2),
    ];
    let q = qrels_binary(&["d1", "d2", "d3"]);
    let p = precision_at_k(&results, &q, 3, 1.0);
    assert!((p - 1.0_f32).abs() < 1e-6);
}

#[test]
fn precision_at_k_one_of_three_relevant() {
    let results = vec![
        make_result("d1", 0.9, 0),
        make_result("d2", 0.8, 1),
        make_result("d3", 0.7, 2),
    ];
    let q = qrels_binary(&["d1"]);
    let p = precision_at_k(&results, &q, 3, 1.0);
    assert!((p - (1.0_f32 / 3.0_f32)).abs() < 1e-6);
}

#[test]
fn precision_at_k_none_relevant_zero() {
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d99"]);
    let p = precision_at_k(&results, &q, 2, 1.0);
    assert!((p - 0.0_f32).abs() < 1e-6);
}

#[test]
fn precision_at_k_truncates_at_results_length() {
    // k=10 but only 2 results: should use min(k, len)
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d1"]);
    // k=10 but iter().take(10) on 2 items gives 2, so 1 hit / 10 = 0.1
    let p = precision_at_k(&results, &q, 10, 1.0);
    assert!((p - 0.1_f32).abs() < 1e-6);
}

#[test]
fn precision_at_k_k_equals_one_first_relevant() {
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d1"]);
    let p = precision_at_k(&results, &q, 1, 1.0);
    assert!((p - 1.0_f32).abs() < 1e-6);
}

// ── recall_at_k ───────────────────────────────────────────────────────────

#[test]
fn recall_at_k_zero_relevant_docs_returns_zero() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = Qrels::default(); // empty
    let r = recall_at_k(&results, &q, 5, 1.0);
    assert!((r - 0.0_f32).abs() < 1e-6);
}

#[test]
fn recall_at_k_found_all_is_one() {
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d1", "d2"]);
    let r = recall_at_k(&results, &q, 2, 1.0);
    assert!((r - 1.0_f32).abs() < 1e-6);
}

#[test]
fn recall_at_k_found_one_of_two() {
    let results = vec![make_result("d1", 0.9, 0), make_result("d99", 0.8, 1)];
    let q = qrels_binary(&["d1", "d2"]);
    let r = recall_at_k(&results, &q, 2, 1.0);
    assert!((r - 0.5_f32).abs() < 1e-6);
}

#[test]
fn recall_at_k_k_smaller_than_results_caps_found() {
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d1", "d2"]);
    // k=1 so only d1 is seen
    let r = recall_at_k(&results, &q, 1, 1.0);
    assert!((r - 0.5_f32).abs() < 1e-6);
}

// ── f1_at_k ───────────────────────────────────────────────────────────────

#[test]
fn f1_at_k_perfect_p_and_r_is_one() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let f = f1_at_k(&results, &q, 1, 1.0);
    assert!((f - 1.0_f32).abs() < 1e-6);
}

#[test]
fn f1_at_k_zero_precision_returns_zero() {
    let results = vec![make_result("d99", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let f = f1_at_k(&results, &q, 1, 1.0);
    assert!((f - 0.0_f32).abs() < 1e-6);
}

#[test]
fn f1_at_k_equal_precision_recall_half() {
    // P=0.5, R=0.5 → F1 = 0.5
    let results = vec![make_result("d1", 0.9, 0), make_result("d99", 0.8, 1)];
    let q = qrels_binary(&["d1", "d2"]);
    let f = f1_at_k(&results, &q, 2, 1.0);
    assert!((f - 0.5_f32).abs() < 1e-6);
}

#[test]
fn f1_at_k_no_relevant_returns_zero() {
    let results = vec![make_result("x", 0.5, 0)];
    let q = Qrels::default();
    let f = f1_at_k(&results, &q, 1, 1.0);
    assert!((f - 0.0_f32).abs() < 1e-6);
}

// ── hit_rate_at_k ─────────────────────────────────────────────────────────

#[test]
fn hit_rate_at_k_relevant_at_pos0_k1_is_one() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let h = hit_rate_at_k(&results, &q, 1, 1.0);
    assert!((h - 1.0_f32).abs() < 1e-6);
}

#[test]
fn hit_rate_at_k_relevant_at_pos2_k1_is_zero() {
    let results = vec![
        make_result("d99", 0.9, 0),
        make_result("d98", 0.8, 1),
        make_result("d1", 0.7, 2),
    ];
    let q = qrels_binary(&["d1"]);
    let h = hit_rate_at_k(&results, &q, 1, 1.0);
    assert!((h - 0.0_f32).abs() < 1e-6);
}

#[test]
fn hit_rate_at_k_all_irrelevant_is_zero() {
    let results = vec![make_result("x1", 0.9, 0), make_result("x2", 0.8, 1)];
    let q = qrels_binary(&["d1"]);
    let h = hit_rate_at_k(&results, &q, 2, 1.0);
    assert!((h - 0.0_f32).abs() < 1e-6);
}

#[test]
fn hit_rate_at_k_relevant_at_boundary_k() {
    let results = vec![make_result("d99", 0.9, 0), make_result("d1", 0.8, 1)];
    let q = qrels_binary(&["d1"]);
    let h = hit_rate_at_k(&results, &q, 2, 1.0);
    assert!((h - 1.0_f32).abs() < 1e-6);
}

// ── reciprocal_rank ───────────────────────────────────────────────────────

#[test]
fn reciprocal_rank_first_is_one() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let rr = reciprocal_rank(&results, &q, 1.0);
    assert!((rr - 1.0_f32).abs() < 1e-6);
}

#[test]
fn reciprocal_rank_second_is_half() {
    let results = vec![make_result("d99", 0.9, 0), make_result("d1", 0.8, 1)];
    let q = qrels_binary(&["d1"]);
    let rr = reciprocal_rank(&results, &q, 1.0);
    assert!((rr - 0.5_f32).abs() < 1e-6);
}

#[test]
fn reciprocal_rank_fourth_is_quarter() {
    let results = vec![
        make_result("d99", 0.9, 0),
        make_result("d98", 0.85, 1),
        make_result("d97", 0.8, 2),
        make_result("d1", 0.7, 3),
    ];
    let q = qrels_binary(&["d1"]);
    let rr = reciprocal_rank(&results, &q, 1.0);
    assert!((rr - 0.25_f32).abs() < 1e-6);
}

#[test]
fn reciprocal_rank_no_relevant_is_zero() {
    let results = vec![make_result("x1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let rr = reciprocal_rank(&results, &q, 1.0);
    assert!((rr - 0.0_f32).abs() < 1e-6);
}

// ── mrr ───────────────────────────────────────────────────────────────────

#[test]
fn mrr_empty_list_returns_zero() {
    let pairs: Vec<(&[SearchResult], &Qrels)> = vec![];
    let m = mrr(&pairs, 1.0);
    assert!((m - 0.0_f32).abs() < 1e-6);
}

#[test]
fn mrr_single_query_rr_one() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let pairs = vec![(results.as_slice(), &q)];
    let m = mrr(&pairs, 1.0);
    assert!((m - 1.0_f32).abs() < 1e-6);
}

#[test]
fn mrr_two_queries_rr_one_and_half() {
    let r1 = vec![make_result("d1", 1.0, 0)];
    let q1 = qrels_binary(&["d1"]);
    let r2 = vec![make_result("x", 0.9, 0), make_result("d2", 0.8, 1)];
    let q2 = qrels_binary(&["d2"]);
    let pairs = vec![(r1.as_slice(), &q1), (r2.as_slice(), &q2)];
    let m = mrr(&pairs, 1.0);
    // (1.0 + 0.5) / 2 = 0.75
    assert!((m - 0.75_f32).abs() < 1e-6);
}

// ── average_precision ─────────────────────────────────────────────────────

#[test]
fn average_precision_no_relevant_is_zero() {
    let results = vec![make_result("x1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let ap = average_precision(&results, &q, 1.0);
    assert!((ap - 0.0_f32).abs() < 1e-6);
}

#[test]
fn average_precision_empty_qrels_is_zero() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = Qrels::default();
    let ap = average_precision(&results, &q, 1.0);
    assert!((ap - 0.0_f32).abs() < 1e-6);
}

#[test]
fn average_precision_perfect_ranking_is_one() {
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d1", "d2"]);
    let ap = average_precision(&results, &q, 1.0);
    // P@1 = 1.0, P@2 = 1.0 → AP = (1.0 + 1.0) / 2 = 1.0
    assert!((ap - 1.0_f32).abs() < 1e-6);
}

#[test]
fn average_precision_two_relevant_at_positions_0_and_2() {
    // Relevant at idx 0 and 2 (3 results total, 2 relevant)
    // P@1 = 1/1, P@3 = 2/3 → AP = (1.0 + 2/3) / 2 = (5/3) / 2 = 5/6
    let results = vec![
        make_result("d1", 0.9, 0),
        make_result("x", 0.8, 1),
        make_result("d2", 0.7, 2),
    ];
    let q = qrels_binary(&["d1", "d2"]);
    let ap = average_precision(&results, &q, 1.0);
    let expected = f32::midpoint(1.0_f32, 2.0_f32 / 3.0_f32);
    assert!((ap - expected).abs() < 1e-6);
}

#[test]
fn average_precision_all_irrelevant_is_zero() {
    let results = vec![make_result("x1", 0.9, 0), make_result("x2", 0.8, 1)];
    let q = qrels_binary(&["d1"]);
    let ap = average_precision(&results, &q, 1.0);
    assert!((ap - 0.0_f32).abs() < 1e-6);
}

// ── dcg_at_k ──────────────────────────────────────────────────────────────

#[test]
fn dcg_at_k_zero_k_returns_zero() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let d = dcg_at_k(&results, &q, 0, 2.0);
    assert!((d - 0.0_f32).abs() < 1e-6);
}

#[test]
fn dcg_at_k_log_base_one_returns_zero() {
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    // log_base <= 1.0 guard
    let d = dcg_at_k(&results, &q, 1, 1.0);
    assert!((d - 0.0_f32).abs() < 1e-6);
}

#[test]
fn dcg_at_k_single_relevant_gain_one_at_pos0() {
    // (2^1 - 1) / log2(2) = 1.0 / 1.0 = 1.0
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let d = dcg_at_k(&results, &q, 1, 2.0);
    assert!((d - 1.0_f32).abs() < 1e-6);
}

#[test]
fn dcg_at_k_graded_gain_two_at_pos0() {
    // (2^2 - 1) / log2(2) = 3.0 / 1.0 = 3.0
    let results = vec![make_result("d1", 1.0, 0)];
    let q = Qrels::from_judgments(vec![RelevanceJudgment::graded("d1", 2.0)]);
    let d = dcg_at_k(&results, &q, 1, 2.0);
    assert!((d - 3.0_f32).abs() < 1e-6);
}

#[test]
fn dcg_at_k_no_relevant_is_zero() {
    let results = vec![make_result("x1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let d = dcg_at_k(&results, &q, 1, 2.0);
    assert!((d - 0.0_f32).abs() < 1e-6);
}

#[test]
fn dcg_at_k_two_relevant_additive() {
    // pos0: (2^1 - 1) / log2(2) = 1.0
    // pos1: (2^1 - 1) / log2(3) ≈ 1.0 / 1.585 ≈ 0.631
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d1", "d2"]);
    let d = dcg_at_k(&results, &q, 2, 2.0);
    let expected = 1.0_f32 + 1.0_f32 / 3.0_f32.log(2.0);
    assert!((d - expected).abs() < 1e-5);
}

// ── ndcg_at_k ─────────────────────────────────────────────────────────────

#[test]
fn ndcg_at_k_perfect_ranking_is_one() {
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d1", "d2"]);
    let n = ndcg_at_k(&results, &q, 2, 2.0);
    assert!((n - 1.0_f32).abs() < 1e-5);
}

#[test]
fn ndcg_at_k_no_relevant_is_zero() {
    let results = vec![make_result("x1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let n = ndcg_at_k(&results, &q, 1, 2.0);
    assert!((n - 0.0_f32).abs() < 1e-6);
}

#[test]
fn ndcg_at_k_actual_zero_guard() {
    // When actual DCG == 0 (no relevant in results), must return 0
    let results = vec![make_result("x1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let n = ndcg_at_k(&results, &q, 5, 2.0);
    assert!((n - 0.0_f32).abs() < 1e-6);
}

#[test]
fn ndcg_at_k_value_in_unit_interval() {
    let results = vec![
        make_result("d2", 0.9, 0), // suboptimal order
        make_result("d1", 0.8, 1),
    ];
    let q = Qrels::from_judgments(vec![
        RelevanceJudgment::graded("d1", 2.0),
        RelevanceJudgment::graded("d2", 1.0),
    ]);
    let n = ndcg_at_k(&results, &q, 2, 2.0);
    assert!((0.0..=1.0).contains(&n));
}

#[test]
fn ndcg_at_k_single_relevant_equals_one() {
    // Only one relevant doc and it's first; ideal == actual
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let n = ndcg_at_k(&results, &q, 1, 2.0);
    assert!((n - 1.0_f32).abs() < 1e-6);
}

// ── RetrievalEvalConfig ───────────────────────────────────────────────────

#[test]
fn retrieval_eval_config_default_values() {
    let cfg = RetrievalEvalConfig::default();
    assert_eq!(cfg.k, 10);
    assert!((cfg.relevance_threshold - 1.0_f32).abs() < 1e-6);
    assert!((cfg.log_base - 2.0_f32).abs() < 1e-6);
}

#[test]
fn retrieval_eval_config_with_k() {
    let cfg = RetrievalEvalConfig::default().with_k(5);
    assert_eq!(cfg.k, 5);
}

#[test]
fn retrieval_eval_config_with_relevance_threshold() {
    let cfg = RetrievalEvalConfig::default().with_relevance_threshold(0.5);
    assert!((cfg.relevance_threshold - 0.5_f32).abs() < 1e-6);
}

#[test]
fn retrieval_eval_config_with_log_base() {
    let cfg = RetrievalEvalConfig::default().with_log_base(10.0);
    assert!((cfg.log_base - 10.0_f32).abs() < 1e-6);
}

// ── RetrievalEvalError display ────────────────────────────────────────────

#[test]
fn retrieval_eval_error_empty_results_display() {
    let e = RetrievalEvalError::EmptyResults;
    assert!(!e.to_string().is_empty());
}

#[test]
fn retrieval_eval_error_empty_qrels_display() {
    let e = RetrievalEvalError::EmptyQrels;
    assert!(!e.to_string().is_empty());
}

#[test]
fn retrieval_eval_error_invalid_k_display() {
    let e = RetrievalEvalError::InvalidK;
    assert!(!e.to_string().is_empty());
}

// ── RetrievalEvaluator ────────────────────────────────────────────────────

#[test]
fn evaluator_evaluate_empty_results_error() {
    let ev = RetrievalEvaluator::default();
    let q = qrels_binary(&["d1"]);
    let res = ev.evaluate(&[], &q, 5);
    assert!(matches!(res, Err(RetrievalEvalError::EmptyResults)));
}

#[test]
fn evaluator_evaluate_empty_qrels_error() {
    let ev = RetrievalEvaluator::default();
    let results = vec![make_result("d1", 1.0, 0)];
    let q = Qrels::default();
    let res = ev.evaluate(&results, &q, 5);
    assert!(matches!(res, Err(RetrievalEvalError::EmptyQrels)));
}

#[test]
fn evaluator_evaluate_zero_k_error() {
    let ev = RetrievalEvaluator::default();
    let results = vec![make_result("d1", 1.0, 0)];
    let q = qrels_binary(&["d1"]);
    let res = ev.evaluate(&results, &q, 0);
    assert!(matches!(res, Err(RetrievalEvalError::InvalidK)));
}

#[test]
fn evaluator_evaluate_happy_path_all_fields_in_range() {
    let ev = RetrievalEvaluator::default();
    let results = vec![make_result("d1", 0.9, 0), make_result("d2", 0.8, 1)];
    let q = qrels_binary(&["d1", "d2"]);
    let scores: RetrievalScores = ev.evaluate(&results, &q, 2).unwrap();
    assert!((0.0..=1.0).contains(&scores.precision_at_k));
    assert!((0.0..=1.0).contains(&scores.recall_at_k));
    assert!((0.0..=1.0).contains(&scores.f1_at_k));
    assert!((0.0..=1.0).contains(&scores.hit_rate_at_k));
    assert!((0.0..=1.0).contains(&scores.reciprocal_rank));
    assert!((0.0..=1.0).contains(&scores.average_precision));
    assert!(scores.dcg_at_k >= 0.0);
    assert!((0.0..=1.0).contains(&scores.ndcg_at_k));
    assert_eq!(scores.k, 2);
}

#[test]
fn evaluator_evaluate_batch_empty_returns_default() {
    let ev = RetrievalEvaluator::default();
    let agg: AggregateScores = ev.evaluate_batch(&[]).unwrap();
    assert_eq!(agg.query_count, 0);
}

#[test]
fn evaluator_evaluate_batch_two_queries_count() {
    let ev = RetrievalEvaluator::new(RetrievalEvalConfig::default().with_k(2));
    let r1 = vec![make_result("d1", 1.0, 0)];
    let q1 = qrels_binary(&["d1"]);
    let r2 = vec![make_result("d2", 1.0, 0)];
    let q2 = qrels_binary(&["d2"]);
    let agg = ev
        .evaluate_batch(&[(r1.as_slice(), &q1), (r2.as_slice(), &q2)])
        .unwrap();
    assert_eq!(agg.query_count, 2);
}

#[test]
fn evaluator_evaluate_batch_aggregates_in_unit_interval() {
    let ev = RetrievalEvaluator::new(RetrievalEvalConfig::default().with_k(2));
    let r1 = vec![make_result("d1", 1.0, 0)];
    let q1 = qrels_binary(&["d1"]);
    let r2 = vec![make_result("d2", 1.0, 0)];
    let q2 = qrels_binary(&["d2"]);
    let agg = ev
        .evaluate_batch(&[(r1.as_slice(), &q1), (r2.as_slice(), &q2)])
        .unwrap();
    assert!((0.0..=1.0).contains(&agg.map));
    assert!((0.0..=1.0).contains(&agg.mrr));
    assert!((0.0..=1.0).contains(&agg.mean_ndcg_at_k));
}

#[test]
fn evaluator_new_stores_config() {
    let cfg = RetrievalEvalConfig::default().with_k(7);
    let ev = RetrievalEvaluator::new(cfg);
    assert_eq!(ev.config.k, 7);
}
