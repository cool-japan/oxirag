//! Tests for `WindowConfig` and `LexicalListwiseJudge`.
use super::doc;
use crate::listwise_rerank::judge::LexicalListwiseJudge;
use crate::listwise_rerank::types::{ListwiseError, ListwiseJudge, ListwiseResult, WindowConfig};

// ── WindowConfig defaults ────────────────────────────────────────────────────

#[test]
fn test_window_config_default_window_size() {
    let c = WindowConfig::default();
    assert_eq!(c.window_size, 4, "default window_size should be 4");
}

#[test]
fn test_window_config_default_step() {
    let c = WindowConfig::default();
    assert_eq!(c.step, 2, "default step should be 2");
}

#[test]
fn test_window_config_default_top_n() {
    let c = WindowConfig::default();
    assert_eq!(c.top_n, 0, "default top_n should be 0");
}

#[test]
fn test_window_config_new_matches_default() {
    let c = WindowConfig::new();
    assert_eq!(c.window_size, 4, "new() window_size should match default");
}

// ── WindowConfig builders ────────────────────────────────────────────────────

#[test]
fn test_window_config_with_window_size() {
    let c = WindowConfig::new().with_window_size(8);
    assert_eq!(c.window_size, 8, "with_window_size should set window_size");
}

#[test]
fn test_window_config_with_step() {
    let c = WindowConfig::new().with_step(3);
    assert_eq!(c.step, 3, "with_step should set step");
}

#[test]
fn test_window_config_with_top_n() {
    let c = WindowConfig::new().with_top_n(5);
    assert_eq!(c.top_n, 5, "with_top_n should set top_n");
}

#[test]
fn test_window_config_builder_chain() {
    let c = WindowConfig::new()
        .with_window_size(6)
        .with_step(3)
        .with_top_n(2);
    assert_eq!(c.step, 3, "chained builder should preserve step");
}

#[test]
fn test_window_config_effective_window_clamps_zero() {
    let c = WindowConfig::new().with_window_size(0);
    assert_eq!(c.effective_window(), 1, "effective_window clamps 0 to 1");
}

#[test]
fn test_window_config_effective_step_clamps_zero() {
    let c = WindowConfig::new().with_step(0);
    assert_eq!(c.effective_step(), 1, "effective_step clamps 0 to 1");
}

#[test]
fn test_window_config_effective_step_preserves_nonzero() {
    let c = WindowConfig::new().with_step(4);
    assert_eq!(c.effective_step(), 4, "effective_step preserves nonzero");
}

// ── LexicalListwiseJudge ─────────────────────────────────────────────────────

#[test]
fn test_judge_new_is_default() {
    let j = LexicalListwiseJudge::new();
    let perm = j.permute("anything", &[doc("anything here")]);
    assert_eq!(perm.len(), 1, "judge over one doc returns one index");
}

#[test]
fn test_judge_permute_is_valid_permutation() {
    let j = LexicalListwiseJudge::new();
    let docs = vec![
        doc("alpha beta gamma"),
        doc("delta epsilon"),
        doc("zeta eta theta"),
    ];
    let mut perm = j.permute("beta gamma", &docs);
    perm.sort_unstable();
    assert_eq!(perm, vec![0, 1, 2], "permutation is a bijection of 0..len");
}

#[test]
fn test_judge_permute_length_matches_input() {
    let j = LexicalListwiseJudge::new();
    let docs = vec![
        doc("one two"),
        doc("three four"),
        doc("five six"),
        doc("seven eight"),
    ];
    let perm = j.permute("three four", &docs);
    assert_eq!(
        perm.len(),
        docs.len(),
        "permutation length equals input length"
    );
}

#[test]
fn test_judge_permute_no_duplicate_indices() {
    let j = LexicalListwiseJudge::new();
    let docs = vec![
        doc("aa bb"),
        doc("cc dd"),
        doc("ee ff"),
        doc("gg hh"),
        doc("ii jj"),
    ];
    let perm = j.permute("cc dd", &docs);
    let mut seen = perm.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        perm.len(),
        "permutation has no duplicate indices"
    );
}

#[test]
fn test_judge_permute_ranks_relevant_first() {
    let j = LexicalListwiseJudge::new();
    let docs = vec![
        doc("completely unrelated text about cooking"),
        doc("machine learning neural networks deep models"),
    ];
    let perm = j.permute("machine learning neural networks", &docs);
    assert_eq!(
        perm[0], 1,
        "most relevant doc index should be ordered first"
    );
}

#[test]
fn test_judge_permute_empty_query_stable() {
    let j = LexicalListwiseJudge::new();
    let docs = vec![doc("alpha"), doc("beta"), doc("gamma")];
    let perm = j.permute("", &docs);
    assert_eq!(
        perm,
        vec![0, 1, 2],
        "empty query yields stable identity order"
    );
}

#[test]
fn test_judge_permute_all_equal_is_stable_identity() {
    let j = LexicalListwiseJudge::new();
    let docs = vec![
        doc("same words here"),
        doc("same words here"),
        doc("same words here"),
    ];
    let perm = j.permute("same words", &docs);
    assert_eq!(
        perm,
        vec![0, 1, 2],
        "equal scores keep input order (stable)"
    );
}

#[test]
fn test_judge_score_zero_for_empty_doc() {
    let j = LexicalListwiseJudge::new();
    let idf = std::collections::HashMap::new();
    let s = j.score("query terms", &doc(""), &idf);
    assert_eq!(s, 0.0, "score for empty doc content is 0.0");
}

#[test]
fn test_judge_score_zero_for_empty_query() {
    let j = LexicalListwiseJudge::new();
    let idf = std::collections::HashMap::new();
    let s = j.score("", &doc("some content here"), &idf);
    assert_eq!(s, 0.0, "score for empty query is 0.0");
}

#[test]
fn test_judge_score_higher_for_more_overlap() {
    let j = LexicalListwiseJudge::new();
    let idf = std::collections::HashMap::new();
    let high = j.score("apple banana cherry", &doc("apple banana cherry"), &idf);
    let low = j.score("apple banana cherry", &doc("apple orange"), &idf);
    assert!(high > low, "more overlapping doc scores higher");
}

// ── error display ────────────────────────────────────────────────────────────

#[test]
fn test_error_display_empty_query() {
    let e = ListwiseError::EmptyQuery;
    assert_eq!(
        e.to_string(),
        "query must not be empty",
        "EmptyQuery message"
    );
}

#[test]
fn test_error_display_empty_candidates() {
    let e = ListwiseError::EmptyCandidates;
    assert_eq!(
        e.to_string(),
        "candidates must not be empty",
        "EmptyCandidates message"
    );
}

// ── ListwiseResult shape ─────────────────────────────────────────────────────

#[test]
fn test_listwise_result_original_rank_field() {
    let r = ListwiseResult {
        document: doc("content"),
        original_score: 0.5,
        original_rank: 3,
        new_rank: 1,
    };
    assert_eq!(r.original_rank, 3, "ListwiseResult exposes original_rank");
}

#[test]
fn test_listwise_result_new_rank_field() {
    let r = ListwiseResult {
        document: doc("content"),
        original_score: 0.5,
        original_rank: 3,
        new_rank: 1,
    };
    assert_eq!(r.new_rank, 1, "ListwiseResult exposes new_rank");
}
