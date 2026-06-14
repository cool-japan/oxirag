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

use std::cell::Cell;

use super::reranker::PairwiseReranker;
use super::types::{
    MockPairwiseComparer, PairwiseComparer, PairwiseConfig, PairwiseError, PairwiseHit,
    PairwiseScoredPair,
};

// ── helper comparers ──────────────────────────────────────────────────────────

/// Always picks doc_a (returns 0).
#[derive(Debug)]
struct AlwaysPickA;

impl PairwiseComparer for AlwaysPickA {
    fn compare(&self, _: &str, _: &str, _: &str) -> usize {
        0
    }
}

/// Always picks doc_b (returns 1).
#[derive(Debug)]
struct AlwaysPickB;

impl PairwiseComparer for AlwaysPickB {
    fn compare(&self, _: &str, _: &str, _: &str) -> usize {
        1
    }
}

/// Returns a preset sequence of winner indices, cycling when exhausted.
#[derive(Debug)]
struct PresetComparer {
    results: Vec<usize>,
    idx: Cell<usize>,
}

impl PresetComparer {
    fn new(results: Vec<usize>) -> Self {
        Self {
            results,
            idx: Cell::new(0),
        }
    }
}

impl PairwiseComparer for PresetComparer {
    fn compare(&self, _: &str, _: &str, _: &str) -> usize {
        let i = self.idx.get();
        let result = self.results[i % self.results.len()];
        self.idx.set(i + 1);
        result
    }
}

// ── helper builders ────────────────────────────────────────────────────────────

fn mock_reranker(prefer_longer: bool, rounds: usize) -> PairwiseReranker {
    PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(rounds),
        Box::new(MockPairwiseComparer::new(prefer_longer)),
    )
}

fn d(id: &str, content: &str) -> (String, String) {
    (id.to_string(), content.to_string())
}

// ── error cases ───────────────────────────────────────────────────────────────

#[test]
fn empty_corpus_returns_empty_corpus_error() {
    let r = mock_reranker(false, 1);
    assert_eq!(
        r.rerank("q", &[], 1).unwrap_err(),
        PairwiseError::EmptyCorpus
    );
}

#[test]
fn invalid_k_zero_returns_invalid_k_error() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "alpha beta")];
    assert_eq!(
        r.rerank("alpha", &docs, 0).unwrap_err(),
        PairwiseError::InvalidK(0)
    );
}

#[test]
fn invalid_k_error_carries_the_k_value() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "alpha beta")];
    let Err(PairwiseError::InvalidK(v)) = r.rerank("alpha", &docs, 0) else {
        panic!("expected InvalidK");
    };
    assert_eq!(v, 0);
}

#[test]
fn tournament_pairs_empty_corpus_error() {
    let r = mock_reranker(false, 1);
    assert_eq!(
        r.tournament_pairs("q", &[]),
        Err(PairwiseError::EmptyCorpus)
    );
}

// ── single-document corpus ────────────────────────────────────────────────────

#[test]
fn single_doc_returns_one_hit() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("only", "rust memory")];
    let hits = r.rerank("rust", &docs, 1).unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn single_doc_score_is_zero() {
    let r = mock_reranker(false, 2);
    let docs = vec![d("only", "rust memory")];
    let hits = r.rerank("rust", &docs, 1).unwrap();
    assert_eq!(hits[0].score, 0.0_f32);
}

#[test]
fn single_doc_wins_is_zero() {
    let r = mock_reranker(false, 3);
    let docs = vec![d("only", "rust memory")];
    let hits = r.rerank("rust", &docs, 1).unwrap();
    assert_eq!(hits[0].wins, 0);
}

#[test]
fn single_doc_id_is_correct() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("sole_doc", "alpha")];
    let hits = r.rerank("alpha", &docs, 1).unwrap();
    assert_eq!(hits[0].id, "sole_doc");
}

// ── k handling ────────────────────────────────────────────────────────────────

#[test]
fn k_greater_than_corpus_returns_all_docs() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "rust"), d("b", "python"), d("c", "java")];
    let hits = r.rerank("rust", &docs, 99).unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn k_equal_to_corpus_size() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "rust"), d("b", "python")];
    let hits = r.rerank("rust", &docs, 2).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn k_equals_one_returns_top_doc_only() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("a", "rust memory ownership"),
        d("b", "banana smoothie"),
        d("c", "rust systems"),
    ];
    let hits = r.rerank("rust memory", &docs, 1).unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn k_of_two_returns_two_hits() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "rust"), d("b", "python"), d("c", "java")];
    let hits = r.rerank("rust", &docs, 2).unwrap();
    assert_eq!(hits.len(), 2);
}

// ── ordering ──────────────────────────────────────────────────────────────────

#[test]
fn results_sorted_by_wins_descending() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("a", "rust memory ownership"),
        d("b", "banana smoothie"),
        d("c", "rust systems"),
    ];
    let hits = r.rerank("rust memory systems", &docs, 3).unwrap();
    assert!(hits[0].wins >= hits[1].wins);
    assert!(hits[1].wins >= hits[2].wins);
}

#[test]
fn first_result_is_most_relevant() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("best", "rust memory ownership safety"),
        d("worst", "banana"),
    ];
    let hits = r.rerank("rust memory ownership", &docs, 2).unwrap();
    assert_eq!(hits[0].id, "best");
}

#[test]
fn last_result_has_fewest_wins_among_returned() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("a", "rust memory ownership"),
        d("b", "rust"),
        d("c", "banana smoothie"),
    ];
    let hits = r.rerank("rust memory", &docs, 3).unwrap();
    assert!(hits[0].wins >= hits[2].wins);
}

// ── two-document cases ────────────────────────────────────────────────────────

#[test]
fn two_docs_higher_overlap_wins() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("good", "rust memory ownership"),
        d("bad", "banana smoothie recipe"),
    ];
    let hits = r.rerank("rust memory", &docs, 2).unwrap();
    assert_eq!(hits[0].id, "good");
}

#[test]
fn two_docs_winner_has_score_one() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("winner", "rust memory ownership"),
        d("loser", "banana smoothie"),
    ];
    let hits = r.rerank("rust memory", &docs, 1).unwrap();
    assert_eq!(hits[0].score, 1.0_f32);
}

#[test]
fn two_docs_loser_has_score_zero() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("winner", "rust memory ownership"),
        d("loser", "banana smoothie"),
    ];
    let hits = r.rerank("rust memory", &docs, 2).unwrap();
    assert_eq!(hits[1].score, 0.0_f32);
}

#[test]
fn two_docs_winner_has_one_win() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("winner", "rust memory ownership"), d("loser", "banana")];
    let hits = r.rerank("rust memory", &docs, 2).unwrap();
    assert_eq!(hits[0].wins, 1);
    assert_eq!(hits[1].wins, 0);
}

// ── win counting ──────────────────────────────────────────────────────────────

#[test]
fn win_counts_correct_one_round_three_docs() {
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(1),
        Box::new(AlwaysPickA),
    );
    // Always-A: doc0 beats doc1 and doc2; doc1 beats doc2.
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z")];
    let hits = r.rerank("q", &docs, 3).unwrap();
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[0].wins, 2);
    assert_eq!(hits[1].id, "b");
    assert_eq!(hits[1].wins, 1);
    assert_eq!(hits[2].id, "c");
    assert_eq!(hits[2].wins, 0);
}

#[test]
fn win_counts_correct_always_pick_b_four_docs() {
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(1),
        Box::new(AlwaysPickB),
    );
    // AlwaysPickB: for pair (i,j), j always wins.
    // doc3 beats everyone (3 wins), doc2 gets 2, doc1 gets 1, doc0 gets 0.
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z"), d("d", "w")];
    let hits = r.rerank("q", &docs, 4).unwrap();
    assert_eq!(hits[0].id, "d");
    assert_eq!(hits[0].wins, 3);
    assert_eq!(hits[3].id, "a");
    assert_eq!(hits[3].wins, 0);
}

#[test]
fn total_wins_equals_n_choose_2_times_rounds() {
    let rounds = 2_usize;
    let r = mock_reranker(false, rounds);
    let docs = vec![
        d("a", "rust"),
        d("b", "python"),
        d("c", "java"),
        d("d", "go"),
    ];
    let n = docs.len();
    let hits = r.rerank("rust", &docs, n).unwrap();
    let total_wins: usize = hits.iter().map(|h| h.wins).sum();
    // Each round produces n*(n-1)/2 comparisons, each with exactly one winner.
    let expected = rounds * n * (n - 1) / 2;
    assert_eq!(total_wins, expected);
}

// ── multiple rounds ───────────────────────────────────────────────────────────

#[test]
fn two_rounds_doubles_wins_vs_one_round() {
    let docs = vec![d("a", "rust memory ownership"), d("b", "banana")];
    let r1 = mock_reranker(false, 1);
    let r2 = mock_reranker(false, 2);
    let h1 = r1.rerank("rust memory", &docs, 2).unwrap();
    let h2 = r2.rerank("rust memory", &docs, 2).unwrap();
    // Same relative order but double the wins.
    assert_eq!(h1[0].wins * 2, h2[0].wins);
}

#[test]
fn multiple_rounds_preserve_ranking_order() {
    let docs = vec![
        d("a", "rust memory ownership"),
        d("b", "rust"),
        d("c", "banana smoothie"),
    ];
    for rounds in [1, 2, 3, 5] {
        let r = mock_reranker(false, rounds);
        let hits = r.rerank("rust memory", &docs, 3).unwrap();
        // Relative order must be consistent regardless of round count.
        assert_eq!(hits[0].id, "a", "rounds={rounds}");
        assert_eq!(hits[2].id, "c", "rounds={rounds}");
    }
}

#[test]
fn three_rounds_win_count_is_three_times_one_round() {
    let docs = vec![
        d("winner", "rust memory ownership safety"),
        d("loser", "banana"),
    ];
    let r1 = mock_reranker(false, 1);
    let r3 = mock_reranker(false, 3);
    let h1 = r1.rerank("rust memory", &docs, 1).unwrap();
    let h3 = r3.rerank("rust memory", &docs, 1).unwrap();
    assert_eq!(h1[0].wins * 3, h3[0].wins);
}

// ── config ────────────────────────────────────────────────────────────────────

#[test]
fn config_default_tournament_rounds_is_2() {
    assert_eq!(PairwiseConfig::default().tournament_rounds, 2);
}

#[test]
fn config_default_prefer_longer_is_false() {
    assert!(!PairwiseConfig::default().prefer_longer);
}

#[test]
fn config_builder_sets_tournament_rounds() {
    let cfg = PairwiseConfig::new().with_tournament_rounds(7);
    assert_eq!(cfg.tournament_rounds, 7);
}

#[test]
fn config_builder_sets_prefer_longer() {
    let cfg = PairwiseConfig::new().with_prefer_longer(true);
    assert!(cfg.prefer_longer);
}

#[test]
fn config_effective_rounds_is_at_least_1() {
    let cfg = PairwiseConfig::new().with_tournament_rounds(0);
    assert_eq!(cfg.effective_rounds(), 1);
}

#[test]
fn config_zero_rounds_treated_as_one_round() {
    let cfg_zero = PairwiseConfig::new().with_tournament_rounds(0);
    let cfg_one = PairwiseConfig::new().with_tournament_rounds(1);
    let docs = vec![d("a", "rust memory"), d("b", "banana")];
    let r0 = PairwiseReranker::new(cfg_zero, Box::new(MockPairwiseComparer::new(false)));
    let r1 = PairwiseReranker::new(cfg_one, Box::new(MockPairwiseComparer::new(false)));
    let h0 = r0.rerank("rust", &docs, 2).unwrap();
    let h1 = r1.rerank("rust", &docs, 2).unwrap();
    assert_eq!(h0[0].id, h1[0].id);
    assert_eq!(h0[0].wins, h1[0].wins);
}

#[test]
fn config_clone_works() {
    let cfg = PairwiseConfig::new()
        .with_tournament_rounds(5)
        .with_prefer_longer(true);
    let cloned = cfg.clone();
    assert_eq!(cloned.tournament_rounds, 5);
    assert!(cloned.prefer_longer);
}

// ── prefer_longer tiebreak ────────────────────────────────────────────────────

#[test]
fn prefer_longer_false_doc_a_wins_on_equal_overlap() {
    // Both docs have exactly one token overlapping with query "rust".
    let comp = MockPairwiseComparer::new(false);
    let result = comp.compare("rust", "rust alpha", "rust beta");
    assert_eq!(
        result, 0,
        "doc_a should win when prefer_longer=false on tie"
    );
}

#[test]
fn prefer_longer_true_longer_doc_wins_on_tie() {
    let comp = MockPairwiseComparer::new(true);
    // Same overlap; doc_b is longer.
    let result = comp.compare("rust", "rust", "rust long content here");
    assert_eq!(result, 1, "longer doc_b should win when prefer_longer=true");
}

#[test]
fn prefer_longer_true_equal_length_doc_a_wins() {
    let comp = MockPairwiseComparer::new(true);
    let result = comp.compare("rust", "rust ab", "rust cd");
    assert_eq!(result, 0, "equal length tie: doc_a wins");
}

#[test]
fn prefer_longer_false_reranker_tiebreak_stable() {
    // All docs have the same overlap — doc at lower original index wins.
    let r = mock_reranker(false, 1);
    let docs = vec![d("first", "rust"), d("second", "rust"), d("third", "rust")];
    let hits = r.rerank("rust", &docs, 3).unwrap();
    // first beats second and third; second beats third.
    assert_eq!(hits[0].id, "first");
    assert_eq!(hits[1].id, "second");
    assert_eq!(hits[2].id, "third");
}

// ── score formula ─────────────────────────────────────────────────────────────

#[test]
fn score_is_in_range_zero_to_one() {
    let r = mock_reranker(false, 2);
    let docs = vec![
        d("a", "rust memory ownership"),
        d("b", "banana"),
        d("c", "python"),
    ];
    let hits = r.rerank("rust memory", &docs, 3).unwrap();
    for hit in &hits {
        assert!(hit.score >= 0.0_f32, "score must be >= 0");
        assert!(hit.score <= 1.0_f32, "score must be <= 1");
    }
}

#[test]
fn score_equals_wins_over_max_wins() {
    let rounds = 2_usize;
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(rounds),
        Box::new(AlwaysPickA),
    );
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z")];
    let n = docs.len();
    let hits = r.rerank("q", &docs, n).unwrap();
    let max_wins = rounds * (n - 1);
    for hit in &hits {
        let expected = hit.wins as f32 / max_wins as f32;
        assert!((hit.score - expected).abs() < 1e-6_f32);
    }
}

#[test]
fn winner_of_all_matches_has_score_one() {
    let rounds = 3_usize;
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(rounds),
        Box::new(AlwaysPickA),
    );
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z")];
    let hits = r.rerank("q", &docs, 1).unwrap();
    assert_eq!(hits[0].score, 1.0_f32);
}

#[test]
fn loser_of_all_matches_has_score_zero() {
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(1),
        Box::new(AlwaysPickA),
    );
    // With AlwaysPickA, last doc (highest index) loses every match.
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z")];
    let hits = r.rerank("q", &docs, 3).unwrap();
    assert_eq!(hits.last().unwrap().score, 0.0_f32);
}

// ── hit field correctness ─────────────────────────────────────────────────────

#[test]
fn hit_id_matches_input_doc_id() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("my_doc_42", "rust memory"), d("other", "banana")];
    let hits = r.rerank("rust", &docs, 1).unwrap();
    assert_eq!(hits[0].id, "my_doc_42");
}

#[test]
fn hit_content_matches_input_doc_content() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "the content string here"), d("b", "other")];
    let hits = r.rerank("content", &docs, 1).unwrap();
    assert_eq!(hits[0].content, "the content string here");
}

#[test]
fn hit_wins_field_is_consistent_with_score() {
    let rounds = 1_usize;
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(rounds),
        Box::new(AlwaysPickB),
    );
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z")];
    let n = docs.len();
    let hits = r.rerank("q", &docs, n).unwrap();
    let max_wins = rounds * (n - 1);
    for hit in &hits {
        let reconstructed = hit.wins as f32 / max_wins as f32;
        assert!((hit.score - reconstructed).abs() < 1e-6_f32);
    }
}

#[test]
fn hit_clone_works() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "rust memory")];
    let hits = r.rerank("rust", &docs, 1).unwrap();
    let cloned: Vec<PairwiseHit> = hits.clone();
    assert_eq!(cloned[0].id, hits[0].id);
}

// ── PairwiseScoredPair ────────────────────────────────────────────────────────

#[test]
fn scored_pair_has_correct_doc_indices() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "rust memory"), d("b", "banana")];
    let pairs = r.tournament_pairs("rust", &docs).unwrap();
    assert_eq!(pairs[0].doc_a_idx, 0);
    assert_eq!(pairs[0].doc_b_idx, 1);
}

#[test]
fn scored_pair_winner_idx_is_zero_or_one() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "rust"), d("b", "python"), d("c", "java")];
    let pairs = r.tournament_pairs("rust memory", &docs).unwrap();
    for pair in &pairs {
        assert!(pair.winner_idx == 0 || pair.winner_idx == 1);
    }
}

#[test]
fn scored_pair_count_is_n_choose_2_per_round() {
    let r = mock_reranker(false, 1);
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z"), d("d", "w")];
    let n = docs.len();
    let pairs = r.tournament_pairs("q", &docs).unwrap();
    assert_eq!(pairs.len(), n * (n - 1) / 2);
}

#[test]
fn scored_pair_equality_works() {
    let p1 = PairwiseScoredPair {
        doc_a_idx: 0,
        doc_b_idx: 1,
        winner_idx: 0,
    };
    let p2 = PairwiseScoredPair {
        doc_a_idx: 0,
        doc_b_idx: 1,
        winner_idx: 0,
    };
    assert_eq!(p1, p2);
}

#[test]
fn scored_pair_winner_reflects_comparer_result() {
    // AlwaysPickB → winner_idx always 1 (doc_b wins)
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(1),
        Box::new(AlwaysPickB),
    );
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z")];
    let pairs = r.tournament_pairs("q", &docs).unwrap();
    for pair in &pairs {
        assert_eq!(pair.winner_idx, 1);
    }
}

// ── MockPairwiseComparer ──────────────────────────────────────────────────────

#[test]
fn mock_comparer_higher_overlap_wins() {
    let comp = MockPairwiseComparer::new(false);
    // doc_a has 2 overlapping tokens, doc_b has 0
    assert_eq!(
        comp.compare("rust memory", "rust memory safety", "banana"),
        0
    );
}

#[test]
fn mock_comparer_b_wins_when_higher_overlap() {
    let comp = MockPairwiseComparer::new(false);
    assert_eq!(
        comp.compare("rust memory", "banana", "rust memory safety"),
        1
    );
}

#[test]
fn mock_comparer_no_overlap_doc_a_wins_on_tie() {
    let comp = MockPairwiseComparer::new(false);
    // Both have zero overlap: doc_a wins
    assert_eq!(comp.compare("zzz", "banana", "mango"), 0);
}

#[test]
fn mock_comparer_debug_format_includes_prefer_longer() {
    let comp = MockPairwiseComparer::new(true);
    let debug_str = format!("{comp:?}");
    assert!(debug_str.contains("prefer_longer"));
}

#[test]
fn mock_comparer_default_has_prefer_longer_false() {
    let comp = MockPairwiseComparer::default();
    assert!(!comp.prefer_longer);
}

// ── custom comparers ──────────────────────────────────────────────────────────

#[test]
fn always_pick_a_comparer_doc_a_always_wins() {
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(1),
        Box::new(AlwaysPickA),
    );
    let docs = vec![d("first", "x"), d("second", "y"), d("third", "z")];
    let hits = r.rerank("q", &docs, 1).unwrap();
    assert_eq!(hits[0].id, "first");
}

#[test]
fn always_pick_b_comparer_last_doc_wins() {
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(1),
        Box::new(AlwaysPickB),
    );
    let docs = vec![d("first", "x"), d("second", "y"), d("last", "z")];
    let hits = r.rerank("q", &docs, 1).unwrap();
    assert_eq!(hits[0].id, "last");
}

// ── tiebreak: stable original-index ordering ──────────────────────────────────

#[test]
fn all_tied_wins_preserves_original_index_order() {
    // PresetComparer cycle [1, 0, 1] produces wins = [1, 1, 1] for n=3.
    // Pairs: (0,1)→1  (0,2)→0  (1,2)→1
    // wins[1]++ then wins[0]++ then wins[2]++  ⟹  [1, 1, 1].
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(1),
        Box::new(PresetComparer::new(vec![1, 0, 1])),
    );
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z")];
    let hits = r.rerank("q", &docs, 3).unwrap();
    assert_eq!(hits[0].wins, 1);
    assert_eq!(hits[1].wins, 1);
    assert_eq!(hits[2].wins, 1);
    // Tiebreak: lower original index first.
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[1].id, "b");
    assert_eq!(hits[2].id, "c");
}

#[test]
fn partial_tie_stable_within_same_win_bucket() {
    // Pairs for n=4: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) → 6 pairs
    // Preset [1,0,1,0,0,0] → winner sequence per pair:
    //   (0,1)→1: wins[1]++
    //   (0,2)→0: wins[0]++
    //   (0,3)→1: wins[3]++
    //   (1,2)→0: wins[1]++
    //   (1,3)→0: wins[1]++
    //   (2,3)→0: wins[2]++
    // wins = [1, 3, 1, 1]
    // sorted desc by wins, tiebreak by index: doc1(3), doc0(1), doc2(1), doc3(1)
    let r = PairwiseReranker::new(
        PairwiseConfig::new().with_tournament_rounds(1),
        Box::new(PresetComparer::new(vec![1, 0, 1, 0, 0, 0])),
    );
    let docs = vec![d("a", "x"), d("b", "y"), d("c", "z"), d("d", "w")];
    let hits = r.rerank("q", &docs, 4).unwrap();
    assert_eq!(hits[0].id, "b"); // 3 wins
    assert_eq!(hits[0].wins, 3);
    // Remaining three all have 1 win — sorted by original index
    assert_eq!(hits[1].id, "a");
    assert_eq!(hits[2].id, "c");
    assert_eq!(hits[3].id, "d");
}

// ── error display ─────────────────────────────────────────────────────────────

#[test]
fn pairwise_error_display_empty_corpus() {
    let msg = format!("{}", PairwiseError::EmptyCorpus);
    assert!(msg.contains("empty") || msg.contains("corpus"));
}

#[test]
fn pairwise_error_display_invalid_k() {
    let msg = format!("{}", PairwiseError::InvalidK(0));
    assert!(msg.contains('1') || msg.contains('0'));
}

#[test]
fn pairwise_error_equality() {
    assert_eq!(PairwiseError::EmptyCorpus, PairwiseError::EmptyCorpus);
    assert_eq!(PairwiseError::InvalidK(0), PairwiseError::InvalidK(0));
    assert_ne!(PairwiseError::EmptyCorpus, PairwiseError::InvalidK(0));
}

// ── reranker debug ────────────────────────────────────────────────────────────

#[test]
fn reranker_debug_format_does_not_panic() {
    let r = mock_reranker(false, 1);
    let _ = format!("{r:?}");
}

// ── edge: identical content ───────────────────────────────────────────────────

#[test]
fn rerank_with_identical_docs_returns_stable_order() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("x", "rust memory"),
        d("y", "rust memory"),
        d("z", "rust memory"),
    ];
    let hits = r.rerank("rust memory", &docs, 3).unwrap();
    // All identical → prefer_longer=false → doc_a always wins on ties.
    assert_eq!(hits[0].id, "x");
    assert_eq!(hits[1].id, "y");
    assert_eq!(hits[2].id, "z");
}

// ── larger corpus ─────────────────────────────────────────────────────────────

#[test]
fn rerank_five_docs_top_two_are_most_relevant() {
    let r = mock_reranker(false, 2);
    let docs = vec![
        d("best", "rust memory ownership safety concurrency"),
        d("good", "rust memory ownership"),
        d("mid", "rust memory"),
        d("low", "rust"),
        d("worst", "banana smoothie recipe"),
    ];
    let hits = r.rerank("rust memory ownership safety", &docs, 2).unwrap();
    assert_eq!(hits[0].id, "best");
    assert_eq!(hits[1].id, "good");
}

#[test]
fn rerank_five_docs_all_returned_when_k_5() {
    let r = mock_reranker(false, 1);
    let docs = vec![
        d("a", "rust"),
        d("b", "python"),
        d("c", "java"),
        d("d", "go"),
        d("e", "cpp"),
    ];
    let hits = r.rerank("rust", &docs, 5).unwrap();
    assert_eq!(hits.len(), 5);
}

#[test]
fn rerank_wins_sum_correct_for_five_docs_two_rounds() {
    let rounds = 2_usize;
    let r = mock_reranker(false, rounds);
    let docs = vec![
        d("a", "rust"),
        d("b", "python"),
        d("c", "java"),
        d("d", "go"),
        d("e", "cpp"),
    ];
    let n = docs.len();
    let hits = r.rerank("rust", &docs, n).unwrap();
    let total: usize = hits.iter().map(|h| h.wins).sum();
    assert_eq!(total, rounds * n * (n - 1) / 2);
}
