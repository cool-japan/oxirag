//! Tests for `ListwiseReranker` sliding-window behavior.
use super::make_result;
use crate::listwise_rerank::reranker::ListwiseReranker;
use crate::listwise_rerank::types::{ListwiseError, ListwiseJudge, WindowConfig};
use crate::types::{Document, SearchResult};

// ── rerank: errors ───────────────────────────────────────────────────────────

#[test]
fn test_rerank_empty_query_errors() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![make_result("a", "hello world", 0.5, 0)];
    let err = reranker.rerank("", &results);
    assert!(
        matches!(err, Err(ListwiseError::EmptyQuery)),
        "empty query should yield EmptyQuery error"
    );
}

#[test]
fn test_rerank_whitespace_query_errors() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![make_result("a", "hello world", 0.5, 0)];
    let err = reranker.rerank("   ", &results);
    assert!(
        matches!(err, Err(ListwiseError::EmptyQuery)),
        "whitespace-only query should yield EmptyQuery error"
    );
}

#[test]
fn test_rerank_empty_candidates_errors() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results: Vec<SearchResult> = vec![];
    let err = reranker.rerank("query", &results);
    assert!(
        matches!(err, Err(ListwiseError::EmptyCandidates)),
        "empty candidates should yield EmptyCandidates error"
    );
}

// ── rerank: core behavior ────────────────────────────────────────────────────

#[test]
fn test_rerank_moves_back_relevant_doc_to_top() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "cooking recipes for dinner", 0.9, 0),
        make_result("b", "gardening tips for spring", 0.8, 1),
        make_result("c", "weather forecast tomorrow", 0.7, 2),
        make_result("d", "travel guide to italy", 0.6, 3),
        make_result(
            "e",
            "rust programming language memory safety ownership borrow checker",
            0.5,
            4,
        ),
    ];
    let ranked = reranker
        .rerank("rust programming language memory safety", &results)
        .unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "e",
        "clearly-most-relevant back doc should reach rank 0"
    );
}

#[test]
fn test_rerank_back_relevant_doc_original_rank_preserved() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "alpha unrelated", 0.9, 0),
        make_result("b", "beta unrelated", 0.8, 1),
        make_result("c", "gamma unrelated", 0.7, 2),
        make_result("d", "delta unrelated", 0.6, 3),
        make_result("e", "neural transformer attention embeddings", 0.5, 4),
    ];
    let ranked = reranker
        .rerank("neural transformer attention", &results)
        .unwrap();
    assert_eq!(
        ranked[0].original_rank, 4,
        "moved doc retains its original rank of 4"
    );
}

#[test]
fn test_rerank_new_ranks_are_contiguous_from_zero() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "alpha beta", 0.9, 0),
        make_result("b", "gamma delta", 0.8, 1),
        make_result("c", "epsilon zeta", 0.7, 2),
        make_result("d", "eta theta", 0.6, 3),
        make_result("e", "iota kappa", 0.5, 4),
    ];
    let ranked = reranker.rerank("gamma delta", &results).unwrap();
    let ranks: Vec<usize> = ranked.iter().map(|r| r.new_rank).collect();
    assert_eq!(ranks, vec![0, 1, 2, 3, 4], "new_rank is 0..k contiguous");
}

#[test]
fn test_rerank_preserves_all_candidates_when_no_top_n() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "alpha beta", 0.9, 0),
        make_result("b", "gamma delta", 0.8, 1),
        make_result("c", "epsilon zeta", 0.7, 2),
    ];
    let ranked = reranker.rerank("gamma", &results).unwrap();
    assert_eq!(ranked.len(), 3, "all candidates returned when top_n is 0");
}

#[test]
fn test_rerank_original_score_preserved() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "unrelated alpha", 0.91, 0),
        make_result("b", "matching keyword target document", 0.42, 1),
    ];
    let ranked = reranker
        .rerank("matching keyword target", &results)
        .unwrap();
    let top = ranked
        .iter()
        .find(|r| r.document.id.as_str() == "b")
        .unwrap();
    assert!(
        (top.original_score - 0.42).abs() < 1e-6,
        "original_score is carried through unchanged"
    );
}

// ── rerank: window edge cases ────────────────────────────────────────────────

#[test]
fn test_rerank_window_ge_n_single_pass_correct() {
    let config = WindowConfig::new().with_window_size(10).with_step(2);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "totally off topic cooking", 0.9, 0),
        make_result("b", "also off topic gardening", 0.8, 1),
        make_result("c", "graph theory vertices edges adjacency", 0.7, 2),
    ];
    let ranked = reranker
        .rerank("graph theory vertices edges", &results)
        .unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "c",
        "window >= n still ranks the relevant doc first in a single window"
    );
}

#[test]
fn test_rerank_window_ge_n_returns_all() {
    let config = WindowConfig::new().with_window_size(100);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "alpha beta", 0.9, 0),
        make_result("b", "gamma delta", 0.8, 1),
    ];
    let ranked = reranker.rerank("alpha", &results).unwrap();
    assert_eq!(ranked.len(), 2, "window >= n returns every candidate");
}

#[test]
fn test_rerank_step_zero_terminates_and_orders() {
    let config = WindowConfig::new().with_window_size(3).with_step(0);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "off topic one", 0.9, 0),
        make_result("b", "off topic two", 0.8, 1),
        make_result("c", "off topic three", 0.7, 2),
        make_result("d", "off topic four", 0.6, 3),
        make_result(
            "e",
            "supervised learning classification regression labels",
            0.5,
            4,
        ),
    ];
    let ranked = reranker
        .rerank("supervised learning classification", &results)
        .unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "e",
        "step=0 is guarded (terminates) and still bubbles relevant doc up"
    );
}

#[test]
fn test_rerank_step_zero_returns_all_candidates() {
    let config = WindowConfig::new().with_window_size(2).with_step(0);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "alpha beta", 0.9, 0),
        make_result("b", "gamma delta", 0.8, 1),
        make_result("c", "epsilon zeta", 0.7, 2),
        make_result("d", "eta theta", 0.6, 3),
    ];
    let ranked = reranker.rerank("alpha", &results).unwrap();
    assert_eq!(
        ranked.len(),
        4,
        "step=0 guarded run still returns all candidates"
    );
}

#[test]
fn test_rerank_small_window_multi_pass() {
    let config = WindowConfig::new().with_window_size(2).with_step(1);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "noise one", 0.9, 0),
        make_result("b", "noise two", 0.8, 1),
        make_result("c", "noise three", 0.7, 2),
        make_result(
            "d",
            "distributed systems consensus raft paxos replication",
            0.6,
            3,
        ),
    ];
    let ranked = reranker
        .rerank("distributed systems consensus raft", &results)
        .unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "d",
        "small window with multiple passes bubbles the relevant doc to the top"
    );
}

#[test]
fn test_rerank_relevant_in_middle_window_reaches_top() {
    let config = WindowConfig::new().with_window_size(3).with_step(2);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "noise alpha", 0.9, 0),
        make_result("b", "noise beta", 0.8, 1),
        make_result("c", "noise gamma", 0.7, 2),
        make_result(
            "d",
            "convolutional neural network image recognition pooling",
            0.6,
            3,
        ),
        make_result("e", "noise epsilon", 0.5, 4),
    ];
    let ranked = reranker
        .rerank("convolutional neural network image", &results)
        .unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "d",
        "relevant doc in a middle window propagates to the front across passes"
    );
}

// ── rerank: top_n truncation ─────────────────────────────────────────────────

#[test]
fn test_rerank_top_n_truncates_length() {
    let config = WindowConfig::new().with_top_n(2);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "alpha beta", 0.9, 0),
        make_result("b", "gamma delta", 0.8, 1),
        make_result("c", "epsilon zeta", 0.7, 2),
        make_result("d", "eta theta", 0.6, 3),
    ];
    let ranked = reranker.rerank("alpha", &results).unwrap();
    assert_eq!(ranked.len(), 2, "top_n truncates the output to 2");
}

#[test]
fn test_rerank_top_n_larger_than_n_returns_all() {
    let config = WindowConfig::new().with_top_n(100);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "alpha beta", 0.9, 0),
        make_result("b", "gamma delta", 0.8, 1),
    ];
    let ranked = reranker.rerank("alpha", &results).unwrap();
    assert_eq!(
        ranked.len(),
        2,
        "top_n larger than n returns all candidates"
    );
}

#[test]
fn test_rerank_top_n_keeps_top_ranked() {
    let config = WindowConfig::new().with_window_size(5).with_top_n(1);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "irrelevant alpha", 0.9, 0),
        make_result("b", "irrelevant beta", 0.8, 1),
        make_result("c", "bayesian inference posterior prior likelihood", 0.7, 2),
    ];
    let ranked = reranker
        .rerank("bayesian inference posterior", &results)
        .unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "c",
        "top_n keeps the highest-ranked candidate"
    );
}

// ── rerank: single & degenerate inputs ───────────────────────────────────────

#[test]
fn test_rerank_single_candidate_unchanged() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![make_result("only", "the one and only document", 0.5, 0)];
    let ranked = reranker.rerank("document", &results).unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "only",
        "single-candidate list returns that candidate"
    );
}

#[test]
fn test_rerank_single_candidate_new_rank_zero() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![make_result("only", "the one and only document", 0.5, 0)];
    let ranked = reranker.rerank("document", &results).unwrap();
    assert_eq!(ranked[0].new_rank, 0, "single candidate gets new_rank 0");
}

#[test]
fn test_rerank_all_equal_scores_stable_order() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "identical body text", 0.5, 0),
        make_result("b", "identical body text", 0.5, 1),
        make_result("c", "identical body text", 0.5, 2),
        make_result("d", "identical body text", 0.5, 3),
    ];
    let ranked = reranker.rerank("identical body", &results).unwrap();
    let ids: Vec<&str> = ranked.iter().map(|r| r.document.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["a", "b", "c", "d"],
        "all-equal scores preserve the input order"
    );
}

// ── rerank: determinism ──────────────────────────────────────────────────────

#[test]
fn test_rerank_is_deterministic() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "vector databases similarity search indexes", 0.9, 0),
        make_result("b", "cooking pasta italian cuisine", 0.8, 1),
        make_result(
            "c",
            "vector embeddings approximate nearest neighbor",
            0.7,
            2,
        ),
        make_result("d", "weather rain forecast", 0.6, 3),
    ];
    let first: Vec<usize> = reranker
        .rerank("vector similarity search embeddings", &results)
        .unwrap()
        .iter()
        .map(|r| r.original_rank)
        .collect();
    let second: Vec<usize> = reranker
        .rerank("vector similarity search embeddings", &results)
        .unwrap()
        .iter()
        .map(|r| r.original_rank)
        .collect();
    assert_eq!(first, second, "reranking twice yields identical orderings");
}

// ── rerank_to_results ────────────────────────────────────────────────────────

#[test]
fn test_rerank_to_results_assigns_new_rank() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "irrelevant alpha", 0.9, 0),
        make_result(
            "b",
            "tokenization subword byte pair encoding vocabulary",
            0.5,
            1,
        ),
    ];
    let out = reranker
        .rerank_to_results("tokenization subword byte pair", &results)
        .unwrap();
    assert_eq!(
        out[0].rank, 0,
        "rerank_to_results sets rank 0 on the top result"
    );
}

#[test]
fn test_rerank_to_results_preserves_score() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "off topic alpha", 0.91, 0),
        make_result(
            "b",
            "reinforcement learning reward policy gradient",
            0.37,
            1,
        ),
    ];
    let out = reranker
        .rerank_to_results("reinforcement learning reward policy", &results)
        .unwrap();
    let top = out.iter().find(|r| r.document.id.as_str() == "b").unwrap();
    assert!(
        (top.score - 0.37).abs() < 1e-6,
        "rerank_to_results keeps the original score"
    );
}

#[test]
fn test_rerank_to_results_empty_query_errors() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![make_result("a", "hello world", 0.5, 0)];
    let out = reranker.rerank_to_results("", &results);
    assert!(
        matches!(out, Err(ListwiseError::EmptyQuery)),
        "rerank_to_results propagates EmptyQuery"
    );
}

#[test]
fn test_rerank_to_results_top_n_truncates() {
    let config = WindowConfig::new().with_top_n(1);
    let reranker = ListwiseReranker::new(config);
    let results = vec![
        make_result("a", "alpha beta", 0.9, 0),
        make_result("b", "gamma delta", 0.8, 1),
        make_result("c", "epsilon zeta", 0.7, 2),
    ];
    let out = reranker.rerank_to_results("alpha", &results).unwrap();
    assert_eq!(out.len(), 1, "rerank_to_results honors top_n truncation");
}

#[test]
fn test_rerank_to_results_count_matches_input() {
    let reranker = ListwiseReranker::new(WindowConfig::default());
    let results = vec![
        make_result("a", "alpha beta", 0.9, 0),
        make_result("b", "gamma delta", 0.8, 1),
        make_result("c", "epsilon zeta", 0.7, 2),
    ];
    let out = reranker.rerank_to_results("gamma", &results).unwrap();
    assert_eq!(
        out.len(),
        3,
        "rerank_to_results returns one entry per candidate"
    );
}

// ── custom judge ─────────────────────────────────────────────────────────────

/// A judge that simply reverses the window order, for testing `with_judge`.
struct ReverseJudge;

impl ListwiseJudge for ReverseJudge {
    fn permute(&self, _query: &str, docs: &[Document]) -> Vec<usize> {
        (0..docs.len()).rev().collect()
    }
}

#[test]
fn test_with_judge_uses_custom_judge() {
    let config = WindowConfig::new().with_window_size(10).with_step(2);
    let reranker = ListwiseReranker::with_judge(config, ReverseJudge);
    let results = vec![
        make_result("a", "first", 0.9, 0),
        make_result("b", "second", 0.8, 1),
        make_result("c", "third", 0.7, 2),
    ];
    let ranked = reranker.rerank("anything", &results).unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "c",
        "custom reverse judge places the last input first"
    );
}

#[test]
fn test_with_judge_permutation_is_valid() {
    let config = WindowConfig::new().with_window_size(3).with_step(1);
    let reranker = ListwiseReranker::with_judge(config, ReverseJudge);
    let results = vec![
        make_result("a", "first", 0.9, 0),
        make_result("b", "second", 0.8, 1),
        make_result("c", "third", 0.7, 2),
        make_result("d", "fourth", 0.6, 3),
    ];
    let ranked = reranker.rerank("anything", &results).unwrap();
    let mut ids: Vec<usize> = ranked.iter().map(|r| r.original_rank).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![0, 1, 2, 3],
        "custom-judge rerank keeps every original index exactly once"
    );
}

// ── default impl ─────────────────────────────────────────────────────────────

#[test]
fn test_default_reranker_uses_default_config() {
    let reranker = ListwiseReranker::default();
    assert_eq!(
        reranker.config.window_size, 4,
        "Default reranker uses the default window size"
    );
}

#[test]
fn test_default_reranker_reranks() {
    let reranker = ListwiseReranker::default();
    let results = vec![
        make_result("a", "off topic", 0.9, 0),
        make_result(
            "b",
            "semantic search dense retrieval embeddings vectors",
            0.5,
            1,
        ),
    ];
    let ranked = reranker
        .rerank("semantic search dense retrieval", &results)
        .unwrap();
    assert_eq!(
        ranked[0].document.id.as_str(),
        "b",
        "Default reranker orders the relevant doc first"
    );
}
