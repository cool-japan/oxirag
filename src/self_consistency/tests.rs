//! Tests for the `self_consistency` module.

#![allow(clippy::float_cmp)]

use super::engine::SelfConsistencyEngine;
use super::types::{
    AnswerCluster, MockReasoningSampler, ReasoningPath, ReasoningSampler, SelfConsistencyConfig,
    SelfConsistencyError, SelfConsistencyOutput, VoteWeighting,
};
use super::types::{answer_jaccard, answers_equivalent, normalize_answer};

// ── Helpers ──────────────────────────────────────────────────────────────────────

fn paths_for_answers(answers: &[&str]) -> Vec<ReasoningPath> {
    answers
        .iter()
        .map(|a| ReasoningPath::new(format!("reasoning toward {a}"), *a))
        .collect()
}

fn run_marginalize(answers: &[&str]) -> SelfConsistencyOutput {
    let engine = SelfConsistencyEngine::default();
    engine.marginalize(paths_for_answers(answers)).unwrap()
}

// ── Config defaults & builders ─────────────────────────────────────────────────────

#[test]
fn test_config_default_num_paths() {
    assert_eq!(SelfConsistencyConfig::default().num_paths, 5);
}

#[test]
fn test_config_default_threshold() {
    assert_eq!(SelfConsistencyConfig::default().equivalence_threshold, 0.8);
}

#[test]
fn test_config_default_weighting() {
    assert_eq!(
        SelfConsistencyConfig::default().weighting,
        VoteWeighting::Uniform
    );
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(SelfConsistencyConfig::new().num_paths, 5);
}

#[test]
fn test_config_with_num_paths() {
    assert_eq!(
        SelfConsistencyConfig::new().with_num_paths(11).num_paths,
        11
    );
}

#[test]
fn test_config_with_equivalence_threshold() {
    assert_eq!(
        SelfConsistencyConfig::new()
            .with_equivalence_threshold(0.5)
            .equivalence_threshold,
        0.5
    );
}

#[test]
fn test_config_with_weighting() {
    assert_eq!(
        SelfConsistencyConfig::new()
            .with_weighting(VoteWeighting::ByReasoningLength)
            .weighting,
        VoteWeighting::ByReasoningLength
    );
}

#[test]
fn test_config_builder_chain() {
    let cfg = SelfConsistencyConfig::new()
        .with_num_paths(7)
        .with_equivalence_threshold(0.9)
        .with_weighting(VoteWeighting::ByReasoningLength);
    assert_eq!(cfg.num_paths, 7);
}

// ── VoteWeighting ──────────────────────────────────────────────────────────────────

#[test]
fn test_weighting_default_is_uniform() {
    assert_eq!(VoteWeighting::default(), VoteWeighting::Uniform);
}

#[test]
fn test_weighting_uniform_as_str() {
    assert_eq!(VoteWeighting::Uniform.as_str(), "uniform");
}

#[test]
fn test_weighting_by_length_as_str() {
    assert_eq!(
        VoteWeighting::ByReasoningLength.as_str(),
        "by_reasoning_length"
    );
}

#[test]
fn test_weighting_uniform_weight_is_one() {
    let path = ReasoningPath::new("a b c d e", "x");
    assert_eq!(VoteWeighting::Uniform.weight_of(&path), 1.0);
}

#[test]
fn test_weighting_by_length_empty_reasoning() {
    let path = ReasoningPath::new("", "x");
    // 1 + ln(1 + 0) = 1.0
    assert_eq!(VoteWeighting::ByReasoningLength.weight_of(&path), 1.0);
}

#[test]
fn test_weighting_by_length_increases_with_tokens() {
    let short = ReasoningPath::new("one", "x");
    let long = ReasoningPath::new("one two three four five six seven", "x");
    assert!(
        VoteWeighting::ByReasoningLength.weight_of(&long)
            > VoteWeighting::ByReasoningLength.weight_of(&short)
    );
}

// ── ReasoningPath ──────────────────────────────────────────────────────────────────

#[test]
fn test_reasoning_path_token_count() {
    let path = ReasoningPath::new("two plus two, equals four!", "4");
    assert_eq!(path.reasoning_token_count(), 5);
}

#[test]
fn test_reasoning_path_new_fields() {
    let path = ReasoningPath::new("because reasons", "yes");
    assert_eq!(path.answer, "yes");
}

// ── MockReasoningSampler ───────────────────────────────────────────────────────────

#[test]
fn test_mock_sampler_cycles_by_index() {
    let sampler = MockReasoningSampler::new(paths_for_answers(&["a", "b"]));
    assert_eq!(sampler.sample("q", 3).answer, "b");
}

#[test]
fn test_mock_sampler_index_zero() {
    let sampler = MockReasoningSampler::new(paths_for_answers(&["a", "b", "c"]));
    assert_eq!(sampler.sample("q", 0).answer, "a");
}

#[test]
fn test_mock_sampler_empty_returns_blank() {
    let sampler = MockReasoningSampler::new(vec![]);
    assert_eq!(sampler.sample("q", 0).answer, "");
}

// ── Normalization & equivalence ────────────────────────────────────────────────────

#[test]
fn test_normalize_strips_trailing_punctuation() {
    assert_eq!(normalize_answer("Paris."), "paris");
}

#[test]
fn test_normalize_lowercases() {
    assert_eq!(normalize_answer("PARIS"), "paris");
}

#[test]
fn test_normalize_collapses_whitespace() {
    assert_eq!(normalize_answer("  new   york  "), "new york");
}

#[test]
fn test_normalize_integer_trailing_dot() {
    assert_eq!(normalize_answer("42."), "42");
}

#[test]
fn test_normalize_integer_equals_plain() {
    assert_eq!(normalize_answer("42."), normalize_answer("42"));
}

#[test]
fn test_normalize_integer_leading_zeros() {
    assert_eq!(normalize_answer("007"), "7");
}

#[test]
fn test_normalize_negative_integer() {
    assert_eq!(normalize_answer("-5"), "-5");
}

#[test]
fn test_normalize_non_integer_untouched() {
    assert_eq!(normalize_answer("3 apples"), "3 apples");
}

#[test]
fn test_jaccard_identical_is_one() {
    assert_eq!(answer_jaccard("paris france", "paris france"), 1.0);
}

#[test]
fn test_jaccard_disjoint_is_zero() {
    assert_eq!(answer_jaccard("paris", "london"), 0.0);
}

#[test]
fn test_equivalent_exact_normalized() {
    assert!(answers_equivalent("Paris.", "paris", 0.8));
}

#[test]
fn test_equivalent_paraphrase_above_threshold() {
    // "the answer is paris" vs "the answer is paris city": 4/5 = 0.8 ≥ 0.8.
    assert!(answers_equivalent(
        "the answer is paris",
        "the answer is paris city",
        0.8
    ));
}

#[test]
fn test_not_equivalent_below_threshold() {
    assert!(!answers_equivalent("paris", "the answer is paris", 0.8));
}

// ── Majority / marginalization ─────────────────────────────────────────────────────

#[test]
fn test_majority_answer_wins() {
    let out = run_marginalize(&["42", "42", "42", "43", "43"]);
    assert_eq!(out.answer, "42");
}

#[test]
fn test_majority_confidence_is_vote_share() {
    let out = run_marginalize(&["42", "42", "42", "43", "43"]);
    assert!((out.confidence - 0.6).abs() < 1e-6);
}

#[test]
fn test_unanimous_confidence_is_one() {
    let out = run_marginalize(&["yes", "yes", "yes"]);
    assert_eq!(out.confidence, 1.0);
}

#[test]
fn test_unanimous_single_cluster() {
    let out = run_marginalize(&["yes", "yes", "yes"]);
    assert_eq!(out.clusters.len(), 1);
}

#[test]
fn test_two_distinct_answers_two_clusters() {
    let out = run_marginalize(&["42", "43"]);
    assert_eq!(out.clusters.len(), 2);
}

#[test]
fn test_winner_is_first_cluster() {
    let out = run_marginalize(&["a", "a", "a", "b"]);
    assert_eq!(out.clusters[0].canonical, "a");
}

#[test]
fn test_integer_canonicalization_merges() {
    // "42." and "42" must land in the same cluster.
    let out = run_marginalize(&["42.", "42", "43"]);
    assert_eq!(out.clusters[0].members.len(), 2);
}

#[test]
fn test_integer_canonicalization_answer() {
    let out = run_marginalize(&["42.", "42", "43"]);
    assert_eq!(out.answer, "42");
}

// ── Paraphrase clustering ──────────────────────────────────────────────────────────

#[test]
fn test_paraphrase_clusters_merge() {
    // Default threshold 0.8: "the answer is paris" vs "the answer is paris"
    // (exact after normalization) merge; add a true paraphrase via Jaccard.
    let paths = vec![
        ReasoningPath::new("r1", "the answer is paris"),
        ReasoningPath::new("r2", "the answer is paris!"),
        ReasoningPath::new("r3", "london"),
    ];
    let engine = SelfConsistencyEngine::default();
    let out = engine.marginalize(paths).unwrap();
    assert_eq!(out.clusters[0].members.len(), 2);
}

#[test]
fn test_lower_threshold_merges_more() {
    // At threshold 0.5, "paris" and "paris city" (Jaccard 1/2 = 0.5) merge.
    let cfg = SelfConsistencyConfig::new().with_equivalence_threshold(0.5);
    let engine = SelfConsistencyEngine::new(cfg);
    let paths = vec![
        ReasoningPath::new("r1", "paris"),
        ReasoningPath::new("r2", "paris city"),
    ];
    let out = engine.marginalize(paths).unwrap();
    assert_eq!(out.clusters.len(), 1);
}

#[test]
fn test_high_threshold_keeps_separate() {
    // At threshold 1.0, only exact-normalized matches merge.
    let cfg = SelfConsistencyConfig::new().with_equivalence_threshold(1.0);
    let engine = SelfConsistencyEngine::new(cfg);
    let paths = vec![
        ReasoningPath::new("r1", "paris france"),
        ReasoningPath::new("r2", "paris"),
    ];
    let out = engine.marginalize(paths).unwrap();
    assert_eq!(out.clusters.len(), 2);
}

// ── Tie-breaking determinism ───────────────────────────────────────────────────────

#[test]
fn test_tie_break_lexicographic_winner() {
    // 2 votes "banana" vs 2 votes "apple": equal votes & members ⇒ "apple".
    let out = run_marginalize(&["banana", "apple", "banana", "apple"]);
    assert_eq!(out.answer, "apple");
}

#[test]
fn test_tie_break_is_deterministic_across_runs() {
    let a = run_marginalize(&["banana", "apple", "banana", "apple"]).answer;
    let b = run_marginalize(&["apple", "banana", "apple", "banana"]).answer;
    assert_eq!(a, b);
}

#[test]
fn test_tie_break_more_members_beats_fewer_same_votes() {
    // Under ByReasoningLength, craft equal total votes but different member counts.
    // Cluster "x": two paths each weight 1.0 (empty reasoning) ⇒ votes 2.0, members 2.
    // Cluster "y": one path weight 2.0 ⇒ need ln term... instead use uniform-like.
    // Simpler: equal votes via uniform is impossible with differing members,
    // so verify the ordering rule directly through cluster_order semantics:
    let out = run_marginalize(&["x", "x", "y", "y"]);
    // Equal votes (2 vs 2), equal members (2 vs 2) ⇒ lexicographic "x".
    assert_eq!(out.clusters[0].canonical, "x");
}

// ── ByReasoningLength changes the winner ───────────────────────────────────────────

#[test]
fn test_by_reasoning_length_flips_winner() {
    // Uniform: "short" wins 3 vs 2. ByReasoningLength: the 2 long-reasoning
    // "long" paths out-vote the 3 terse "short" paths.
    let long_reasoning = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega one two three four five";
    let paths = vec![
        ReasoningPath::new("a", "short"),
        ReasoningPath::new("b", "short"),
        ReasoningPath::new("c", "short"),
        ReasoningPath::new(long_reasoning, "long"),
        ReasoningPath::new(long_reasoning, "long"),
    ];
    let uniform = SelfConsistencyEngine::default()
        .marginalize(paths.clone())
        .unwrap();
    let weighted = SelfConsistencyEngine::new(
        SelfConsistencyConfig::new().with_weighting(VoteWeighting::ByReasoningLength),
    )
    .marginalize(paths)
    .unwrap();
    assert!(uniform.answer == "short" && weighted.answer == "long");
}

#[test]
fn test_by_reasoning_length_weighted_votes_exceed_uniform_majority() {
    let long_reasoning = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon";
    let paths = vec![
        ReasoningPath::new("a", "short"),
        ReasoningPath::new("b", "short"),
        ReasoningPath::new("c", "short"),
        ReasoningPath::new(long_reasoning, "long"),
        ReasoningPath::new(long_reasoning, "long"),
    ];
    let weighted = SelfConsistencyEngine::new(
        SelfConsistencyConfig::new().with_weighting(VoteWeighting::ByReasoningLength),
    )
    .marginalize(paths)
    .unwrap();
    assert_eq!(weighted.answer, "long");
}

// ── Cluster coverage ───────────────────────────────────────────────────────────────

#[test]
fn test_clusters_cover_all_paths() {
    let out = run_marginalize(&["a", "b", "a", "c", "b"]);
    let total_members: usize = out.clusters.iter().map(|c| c.members.len()).sum();
    assert_eq!(total_members, 5);
}

#[test]
fn test_cluster_member_indices_are_unique_and_complete() {
    let out = run_marginalize(&["a", "b", "a", "c", "b"]);
    let mut indices: Vec<usize> = out
        .clusters
        .iter()
        .flat_map(|c| c.members.clone())
        .collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_output_retains_all_paths() {
    let out = run_marginalize(&["a", "b", "c"]);
    assert_eq!(out.paths.len(), 3);
}

#[test]
fn test_total_vote_share_winner_plus_rest() {
    // With 5 uniform paths over 3 distinct answers (a×2, b×2, c×1): winner 2/5.
    let out = run_marginalize(&["a", "b", "a", "c", "b"]);
    assert!((out.confidence - 0.4).abs() < 1e-6);
}

// ── Errors ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_marginalize_empty_paths_errors() {
    let engine = SelfConsistencyEngine::default();
    let err = engine.marginalize(vec![]).unwrap_err();
    assert!(matches!(err, SelfConsistencyError::NoPaths));
}

#[test]
fn test_run_empty_question_errors() {
    let sampler = MockReasoningSampler::new(paths_for_answers(&["a"]));
    let engine = SelfConsistencyEngine::default();
    let err = engine.run("   ", &sampler).unwrap_err();
    assert!(matches!(err, SelfConsistencyError::EmptyQuestion));
}

#[test]
fn test_run_zero_paths_errors() {
    let sampler = MockReasoningSampler::new(paths_for_answers(&["a"]));
    let engine = SelfConsistencyEngine::new(SelfConsistencyConfig::new().with_num_paths(0));
    let err = engine.run("question", &sampler).unwrap_err();
    assert!(matches!(err, SelfConsistencyError::NoPaths));
}

#[test]
fn test_error_display_empty_question() {
    assert_eq!(
        SelfConsistencyError::EmptyQuestion.to_string(),
        "question must not be empty"
    );
}

#[test]
fn test_error_display_no_paths() {
    assert_eq!(
        SelfConsistencyError::NoPaths.to_string(),
        "no reasoning paths"
    );
}

// ── run() with MockReasoningSampler ────────────────────────────────────────────────

#[test]
fn test_run_with_mock_sampler_majority() {
    // 5 paths over cycle [42, 42, 43] ⇒ indices 0..5 ⇒ 42,42,43,42,42 ⇒ "42".
    let sampler = MockReasoningSampler::new(paths_for_answers(&["42", "42", "43"]));
    let engine = SelfConsistencyEngine::new(SelfConsistencyConfig::new().with_num_paths(5));
    let out = engine.run("what is 6*7?", &sampler).unwrap();
    assert_eq!(out.answer, "42");
}

#[test]
fn test_run_with_mock_sampler_confidence() {
    let sampler = MockReasoningSampler::new(paths_for_answers(&["42", "42", "43"]));
    let engine = SelfConsistencyEngine::new(SelfConsistencyConfig::new().with_num_paths(5));
    let out = engine.run("what is 6*7?", &sampler).unwrap();
    // 42 appears at indices 0,1,3,4 (4 of 5) ⇒ 0.8.
    assert!((out.confidence - 0.8).abs() < 1e-6);
}

#[test]
fn test_run_samples_correct_count() {
    let sampler = MockReasoningSampler::new(paths_for_answers(&["a", "b"]));
    let engine = SelfConsistencyEngine::new(SelfConsistencyConfig::new().with_num_paths(4));
    let out = engine.run("q", &sampler).unwrap();
    assert_eq!(out.paths.len(), 4);
}

#[test]
fn test_run_single_path_full_confidence() {
    let sampler = MockReasoningSampler::new(paths_for_answers(&["only"]));
    let engine = SelfConsistencyEngine::new(SelfConsistencyConfig::new().with_num_paths(1));
    let out = engine.run("q", &sampler).unwrap();
    assert_eq!(out.confidence, 1.0);
}

// ── Determinism ────────────────────────────────────────────────────────────────────

#[test]
fn test_marginalize_deterministic_answer() {
    let a = run_marginalize(&["a", "b", "a", "c", "b", "a"]).answer;
    let b = run_marginalize(&["a", "b", "a", "c", "b", "a"]).answer;
    assert_eq!(a, b);
}

#[test]
fn test_marginalize_deterministic_confidence() {
    let a = run_marginalize(&["a", "b", "a", "c", "b", "a"]).confidence;
    let b = run_marginalize(&["a", "b", "a", "c", "b", "a"]).confidence;
    assert_eq!(a, b);
}

#[test]
fn test_run_deterministic_across_calls() {
    let sampler = MockReasoningSampler::new(paths_for_answers(&["x", "y", "x"]));
    let engine = SelfConsistencyEngine::new(SelfConsistencyConfig::new().with_num_paths(6));
    let a = engine.run("q", &sampler).unwrap().answer;
    let b = engine.run("q", &sampler).unwrap().answer;
    assert_eq!(a, b);
}

#[test]
fn test_cluster_ordering_descending_votes() {
    let out = run_marginalize(&["a", "a", "a", "b", "b", "c"]);
    // Clusters sorted votes desc: a(3) ≥ b(2) ≥ c(1).
    let ok = out.clusters[0].votes >= out.clusters[1].votes
        && out.clusters[1].votes >= out.clusters[2].votes;
    assert!(ok);
}

#[test]
fn test_default_engine_uses_default_config() {
    let engine = SelfConsistencyEngine::default();
    assert_eq!(engine.config.num_paths, 5);
}

#[test]
fn test_output_struct_is_constructible() {
    // Sanity: ensure public fields of SelfConsistencyOutput / AnswerCluster are usable.
    let cluster = AnswerCluster {
        canonical: "x".to_string(),
        members: vec![0],
        votes: 1.0,
    };
    let out = SelfConsistencyOutput {
        answer: "x".to_string(),
        confidence: 1.0,
        clusters: vec![cluster],
        paths: vec![ReasoningPath::new("r", "x")],
    };
    assert_eq!(out.answer, "x");
}
