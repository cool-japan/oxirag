#![allow(clippy::float_cmp, clippy::similar_names, clippy::cast_precision_loss)]

use super::engine::SetwiseReranker;
use super::types::{
    SetwiseCandidate, SetwiseComparison, SetwiseConfig, SetwiseError, SetwiseRankEntry,
    SetwiseRanking, SetwiseSortStrategy,
};

// ── test helpers ──────────────────────────────────────────────────────────────

/// Deterministic multi-topic corpus of `n` candidates, cycling through a
/// fixed set of topics so relevance is clearly separable for a topical
/// query.
fn topical_corpus(n: usize) -> Vec<SetwiseCandidate> {
    const TOPICS: [&str; 10] = [
        "rust programming systems memory safety ownership borrow checker",
        "python data science machine learning pandas numpy scikit",
        "javascript frontend react vue framework typescript",
        "banana smoothie recipe kitchen cooking blender yogurt",
        "quantum physics particle entanglement theory relativity",
        "ancient rome empire civilization history senate legion",
        "basketball sports team championship playoffs tournament",
        "jazz piano guitar composition music orchestra",
        "ocean marine biology coral reef ecosystem plankton",
        "algebra calculus geometry proof mathematics theorem",
    ];
    (0..n)
        .map(|i| {
            let topic = TOPICS[i % TOPICS.len()];
            SetwiseCandidate::new(format!("doc{i}"), format!("{topic} filler passage {i}"))
        })
        .collect()
}

/// Brute-force full sort by [`SetwiseReranker::score`] — an oracle that is
/// entirely independent of the heapsort/bubblesort schedules, used to check
/// their top-`m` outputs against.
fn brute_force_top_m(
    reranker: &SetwiseReranker,
    query: &str,
    candidates: &[SetwiseCandidate],
    top_m: usize,
) -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, reranker.score(query, &c.content)))
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored
        .into_iter()
        .take(top_m)
        .map(|(i, _)| candidates[i].id.clone())
        .collect()
}

fn heap_reranker(comparison_size: usize) -> SetwiseReranker {
    SetwiseReranker::new(
        SetwiseConfig::new()
            .with_comparison_size(comparison_size)
            .with_strategy(SetwiseSortStrategy::Heapsort),
    )
}

fn bubble_reranker(comparison_size: usize) -> SetwiseReranker {
    SetwiseReranker::new(
        SetwiseConfig::new()
            .with_comparison_size(comparison_size)
            .with_strategy(SetwiseSortStrategy::Bubblesort),
    )
}

// ── SetwiseCandidate ──────────────────────────────────────────────────────────

#[test]
fn setwise_candidate_new_stores_id_and_content() {
    let c = SetwiseCandidate::new("id1", "hello world");
    assert_eq!(c.id, "id1");
    assert_eq!(c.content, "hello world");
}

#[test]
fn setwise_candidate_equality() {
    let a = SetwiseCandidate::new("x", "text");
    let b = SetwiseCandidate::new("x", "text");
    let c = SetwiseCandidate::new("y", "text");
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ── SetwiseConfig ─────────────────────────────────────────────────────────────

#[test]
fn setwise_config_default_values() {
    let cfg = SetwiseConfig::default();
    assert_eq!(cfg.comparison_size, 4);
    assert_eq!(cfg.strategy, SetwiseSortStrategy::Heapsort);
    assert_eq!(cfg.top_m, 10);
    assert!((cfg.semantic_weight - 0.6).abs() < f32::EPSILON);
    assert!((cfg.lexical_weight - 0.4).abs() < f32::EPSILON);
    assert_eq!(cfg.embedding_dim, 128);
}

#[test]
fn setwise_config_builder_chain_overrides_all_fields() {
    let cfg = SetwiseConfig::new()
        .with_comparison_size(5)
        .with_strategy(SetwiseSortStrategy::Bubblesort)
        .with_top_m(7)
        .with_semantic_weight(0.3)
        .with_lexical_weight(0.7)
        .with_embedding_dim(32);
    assert_eq!(cfg.comparison_size, 5);
    assert_eq!(cfg.strategy, SetwiseSortStrategy::Bubblesort);
    assert_eq!(cfg.top_m, 7);
    assert!((cfg.semantic_weight - 0.3).abs() < f32::EPSILON);
    assert!((cfg.lexical_weight - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.embedding_dim, 32);
}

#[test]
fn setwise_config_new_equals_default() {
    assert_eq!(SetwiseConfig::new(), SetwiseConfig::default());
}

// ── SetwiseSortStrategy ───────────────────────────────────────────────────────

#[test]
fn setwise_sort_strategy_as_str() {
    assert_eq!(SetwiseSortStrategy::Heapsort.as_str(), "heapsort");
    assert_eq!(SetwiseSortStrategy::Bubblesort.as_str(), "bubblesort");
}

#[test]
fn setwise_sort_strategy_default_is_heapsort() {
    assert_eq!(
        SetwiseSortStrategy::default(),
        SetwiseSortStrategy::Heapsort
    );
}

// ── SetwiseComparison ─────────────────────────────────────────────────────────

#[test]
fn setwise_comparison_best_returns_first_ranked() {
    let comparison = SetwiseComparison {
        indices: vec![3, 1, 4],
        ranked: vec![4, 3, 1],
        scores: vec![0.1, 0.2, 0.9],
    };
    assert_eq!(comparison.best(), Some(4));
    assert_eq!(comparison.len(), 3);
    assert!(!comparison.is_empty());
}

#[test]
fn setwise_comparison_empty_has_no_best() {
    let comparison = SetwiseComparison {
        indices: vec![],
        ranked: vec![],
        scores: vec![],
    };
    assert_eq!(comparison.best(), None);
    assert!(comparison.is_empty());
    assert_eq!(comparison.len(), 0);
}

// ── SetwiseRanking / SetwiseRankEntry ─────────────────────────────────────────

#[test]
fn setwise_ranking_len_is_empty_and_ids() {
    let ranking = SetwiseRanking {
        entries: vec![
            SetwiseRankEntry {
                candidate: SetwiseCandidate::new("a", "x"),
                score: 0.9,
                original_index: 2,
                new_rank: 0,
            },
            SetwiseRankEntry {
                candidate: SetwiseCandidate::new("b", "y"),
                score: 0.5,
                original_index: 0,
                new_rank: 1,
            },
        ],
    };
    assert_eq!(ranking.len(), 2);
    assert!(!ranking.is_empty());
    assert_eq!(ranking.ids(), vec!["a", "b"]);
}

#[test]
fn setwise_ranking_default_is_empty() {
    let ranking = SetwiseRanking::default();
    assert!(ranking.is_empty());
    assert_eq!(ranking.len(), 0);
    assert!(ranking.ids().is_empty());
}

// ── SetwiseError ──────────────────────────────────────────────────────────────

#[test]
fn setwise_error_display_messages() {
    assert_eq!(
        SetwiseError::EmptyCandidates.to_string(),
        "candidates must not be empty"
    );
    assert_eq!(
        SetwiseError::InvalidComparisonSize(1).to_string(),
        "comparison_size must be at least 2, got 1"
    );
    assert_eq!(
        SetwiseError::InvalidTopM {
            top_m: 0,
            candidate_count: 5
        }
        .to_string(),
        "top_m must be at least 1 and at most the candidate count (5), got 0"
    );
}

#[test]
fn setwise_error_equality() {
    assert_eq!(SetwiseError::EmptyCandidates, SetwiseError::EmptyCandidates);
    assert_ne!(
        SetwiseError::InvalidComparisonSize(1),
        SetwiseError::InvalidComparisonSize(0)
    );
}

// ── compare_set primitive ─────────────────────────────────────────────────────

#[test]
fn compare_set_ranks_a_set_correctly_by_relevance() {
    let reranker = heap_reranker(4);
    let candidates = vec![
        SetwiseCandidate::new("a", "rust systems programming memory safety"),
        SetwiseCandidate::new("b", "banana smoothie recipe"),
        SetwiseCandidate::new("c", "rust ownership borrow checker"),
    ];
    let comparison = reranker.compare_set("rust memory safety systems", &candidates, &[0, 1, 2]);
    assert_eq!(comparison.indices, vec![0, 1, 2]);
    // "a" shares every query token; it must be the set's best candidate.
    assert_eq!(comparison.best(), Some(0));
    let mut sorted_ranked = comparison.ranked.clone();
    sorted_ranked.sort_unstable();
    assert_eq!(sorted_ranked, vec![0, 1, 2]);
}

#[test]
fn compare_set_ties_broken_by_ascending_original_index() {
    let reranker = heap_reranker(4);
    let candidates = vec![
        SetwiseCandidate::new("a", "same content here"),
        SetwiseCandidate::new("b", "same content here"),
        SetwiseCandidate::new("c", "same content here"),
    ];
    let comparison = reranker.compare_set("same content here", &candidates, &[2, 0, 1]);
    assert_eq!(comparison.ranked, vec![0, 1, 2]);
}

#[test]
fn compare_set_empty_indices_returns_empty_comparison() {
    let reranker = heap_reranker(4);
    let candidates = topical_corpus(3);
    let comparison = reranker.compare_set("query", &candidates, &[]);
    assert!(comparison.is_empty());
    assert_eq!(comparison.best(), None);
}

#[test]
fn compare_set_single_index_returns_trivial_ranking() {
    let reranker = heap_reranker(4);
    let candidates = topical_corpus(3);
    let comparison = reranker.compare_set("query", &candidates, &[1]);
    assert_eq!(comparison.ranked, vec![1]);
    assert_eq!(comparison.best(), Some(1));
}

#[test]
fn compare_set_at_k_equals_2_behaves_as_a_binary_comparison() {
    let reranker = heap_reranker(2);
    let candidates = vec![
        SetwiseCandidate::new("a", "rust memory safety systems"),
        SetwiseCandidate::new("b", "banana smoothie recipe"),
    ];
    let comparison = reranker.compare_set("rust memory safety systems", &candidates, &[0, 1]);
    assert_eq!(comparison.len(), 2);
    assert_eq!(comparison.best(), Some(0));
    let score_a = reranker.score("rust memory safety systems", &candidates[0].content);
    let score_b = reranker.score("rust memory safety systems", &candidates[1].content);
    assert!(score_a > score_b);
}

// ── rerank validation / error paths ───────────────────────────────────────────

#[test]
fn rerank_errors_on_empty_candidates() {
    let reranker = heap_reranker(4);
    let err = reranker.rerank("query", &[], 1).unwrap_err();
    assert_eq!(err, SetwiseError::EmptyCandidates);
}

#[test]
fn rerank_errors_on_comparison_size_zero() {
    let reranker = SetwiseReranker::new(SetwiseConfig::new().with_comparison_size(0));
    let candidates = topical_corpus(3);
    let err = reranker.rerank("query", &candidates, 1).unwrap_err();
    assert_eq!(err, SetwiseError::InvalidComparisonSize(0));
}

#[test]
fn rerank_errors_on_comparison_size_one() {
    let reranker = SetwiseReranker::new(SetwiseConfig::new().with_comparison_size(1));
    let candidates = topical_corpus(3);
    let err = reranker.rerank("query", &candidates, 1).unwrap_err();
    assert_eq!(err, SetwiseError::InvalidComparisonSize(1));
}

#[test]
fn rerank_errors_on_top_m_zero() {
    let reranker = heap_reranker(4);
    let candidates = topical_corpus(5);
    let err = reranker.rerank("query", &candidates, 0).unwrap_err();
    assert_eq!(
        err,
        SetwiseError::InvalidTopM {
            top_m: 0,
            candidate_count: 5
        }
    );
}

#[test]
fn rerank_errors_on_top_m_exceeds_candidate_count() {
    let reranker = heap_reranker(4);
    let candidates = topical_corpus(5);
    let err = reranker.rerank("query", &candidates, 6).unwrap_err();
    assert_eq!(
        err,
        SetwiseError::InvalidTopM {
            top_m: 6,
            candidate_count: 5
        }
    );
}

// ── heapsort correctness ───────────────────────────────────────────────────────

#[test]
fn heapsort_top_m_matches_brute_force_small_corpus() {
    let reranker = heap_reranker(3);
    let candidates = topical_corpus(9);
    let query = "rust programming memory safety";
    let result = reranker.rerank(query, &candidates, 3).unwrap();
    let expected = brute_force_top_m(&reranker, query, &candidates, 3);
    assert_eq!(
        result.ranking.ids(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

#[test]
fn heapsort_top_m_matches_brute_force_larger_corpus() {
    let reranker = heap_reranker(4);
    let candidates = topical_corpus(37);
    let query = "ocean marine biology coral reef";
    let result = reranker.rerank(query, &candidates, 8).unwrap();
    let expected = brute_force_top_m(&reranker, query, &candidates, 8);
    assert_eq!(
        result.ranking.ids(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

#[test]
fn heapsort_full_ranking_matches_brute_force_full_sort() {
    let reranker = heap_reranker(3);
    let candidates = topical_corpus(15);
    let query = "jazz piano guitar composition";
    let result = reranker.rerank(query, &candidates, 15).unwrap();
    let expected = brute_force_top_m(&reranker, query, &candidates, 15);
    assert_eq!(
        result.ranking.ids(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

// ── bubblesort correctness ──────────────────────────────────────────────────────

#[test]
fn bubblesort_top_m_matches_brute_force_small_corpus() {
    let reranker = bubble_reranker(3);
    let candidates = topical_corpus(9);
    let query = "rust programming memory safety";
    let result = reranker.rerank(query, &candidates, 3).unwrap();
    let expected = brute_force_top_m(&reranker, query, &candidates, 3);
    assert_eq!(
        result.ranking.ids(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

#[test]
fn bubblesort_top_m_matches_brute_force_larger_corpus() {
    let reranker = bubble_reranker(4);
    let candidates = topical_corpus(37);
    let query = "ancient rome empire civilization";
    let result = reranker.rerank(query, &candidates, 8).unwrap();
    let expected = brute_force_top_m(&reranker, query, &candidates, 8);
    assert_eq!(
        result.ranking.ids(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

#[test]
fn bubblesort_full_ranking_matches_brute_force_full_sort() {
    let reranker = bubble_reranker(3);
    let candidates = topical_corpus(15);
    let query = "algebra calculus geometry proof";
    let result = reranker.rerank(query, &candidates, 15).unwrap();
    let expected = brute_force_top_m(&reranker, query, &candidates, 15);
    assert_eq!(
        result.ranking.ids(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

// ── cross-strategy agreement ────────────────────────────────────────────────────

#[test]
fn heapsort_and_bubblesort_agree_on_top_m() {
    let candidates = topical_corpus(25);
    let query = "basketball sports championship playoffs";
    let heap_result = heap_reranker(4).rerank(query, &candidates, 6).unwrap();
    let bubble_result = bubble_reranker(4).rerank(query, &candidates, 6).unwrap();
    assert_eq!(heap_result.ranking.ids(), bubble_result.ranking.ids());
}

#[test]
fn heapsort_and_bubblesort_agree_across_multiple_k() {
    let candidates = topical_corpus(20);
    let query = "python data science machine learning";
    for k in [2, 3, 4, 5, 6] {
        let heap_result = heap_reranker(k).rerank(query, &candidates, 5).unwrap();
        let bubble_result = bubble_reranker(k).rerank(query, &candidates, 5).unwrap();
        assert_eq!(
            heap_result.ranking.ids(),
            bubble_result.ranking.ids(),
            "mismatch at k={k}"
        );
    }
}

// ── efficiency (comparison count) ───────────────────────────────────────────────

#[test]
fn heapsort_comparison_count_beats_pairwise_bound() {
    let candidates = topical_corpus(40);
    let query = "quantum physics particle entanglement";
    let result = heap_reranker(4).rerank(query, &candidates, 10).unwrap();
    let pairwise_bound = candidates.len() * (candidates.len() - 1) / 2;
    assert!(
        result.comparison_count < pairwise_bound,
        "{} !< {pairwise_bound}",
        result.comparison_count
    );
}

#[test]
fn bubblesort_comparison_count_beats_pairwise_bound() {
    let candidates = topical_corpus(40);
    let query = "quantum physics particle entanglement";
    let result = bubble_reranker(4).rerank(query, &candidates, 10).unwrap();
    let pairwise_bound = candidates.len() * (candidates.len() - 1) / 2;
    assert!(
        result.comparison_count < pairwise_bound,
        "{} !< {pairwise_bound}",
        result.comparison_count
    );
}

#[test]
fn heapsort_comparison_count_beats_pairwise_bound_full_sort() {
    let candidates = topical_corpus(50);
    let query = "history ancient rome";
    let result = heap_reranker(4).rerank(query, &candidates, 50).unwrap();
    let pairwise_bound = candidates.len() * (candidates.len() - 1) / 2;
    assert!(
        result.comparison_count < pairwise_bound,
        "{} !< {pairwise_bound}",
        result.comparison_count
    );
}

#[test]
fn bubblesort_comparison_count_beats_pairwise_bound_moderate_top_m() {
    let candidates = topical_corpus(50);
    let query = "history ancient rome";
    let result = bubble_reranker(4).rerank(query, &candidates, 15).unwrap();
    let pairwise_bound = candidates.len() * (candidates.len() - 1) / 2;
    assert!(
        result.comparison_count < pairwise_bound,
        "{} !< {pairwise_bound}",
        result.comparison_count
    );
}

// ── k=2 degeneracy ───────────────────────────────────────────────────────────────

#[test]
fn bubblesort_k2_full_sort_matches_exact_pairwise_comparison_count() {
    let candidates = topical_corpus(12);
    let query = "javascript frontend react framework";
    let result = bubble_reranker(2).rerank(query, &candidates, 12).unwrap();
    let n = candidates.len();
    // At k=2 every window step is a genuine 2-way comparison, and a full
    // sort via `top_m == n` passes visits exactly the same n*(n-1)/2 pairs a
    // pairwise round-robin tournament would.
    assert_eq!(result.comparison_count, n * (n - 1) / 2);
}

#[test]
fn heapsort_k2_still_produces_correct_top_m() {
    let reranker = heap_reranker(2);
    let candidates = topical_corpus(16);
    let query = "mathematics algebra calculus theorem";
    let result = reranker.rerank(query, &candidates, 4).unwrap();
    let expected = brute_force_top_m(&reranker, query, &candidates, 4);
    assert_eq!(
        result.ranking.ids(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert_eq!(result.comparison_size, 2);
}

// ── determinism & stable ties ─────────────────────────────────────────────────

#[test]
fn heapsort_rerank_is_deterministic() {
    let reranker = heap_reranker(4);
    let candidates = topical_corpus(18);
    let query = "ocean marine biology";
    let r1 = reranker.rerank(query, &candidates, 5).unwrap();
    let r2 = reranker.rerank(query, &candidates, 5).unwrap();
    assert_eq!(r1.ranking.ids(), r2.ranking.ids());
    assert_eq!(r1.comparison_count, r2.comparison_count);
}

#[test]
fn bubblesort_rerank_is_deterministic() {
    let reranker = bubble_reranker(4);
    let candidates = topical_corpus(18);
    let query = "ocean marine biology";
    let r1 = reranker.rerank(query, &candidates, 5).unwrap();
    let r2 = reranker.rerank(query, &candidates, 5).unwrap();
    assert_eq!(r1.ranking.ids(), r2.ranking.ids());
    assert_eq!(r1.comparison_count, r2.comparison_count);
}

#[test]
fn ties_broken_by_ascending_original_index_when_query_is_empty() {
    // An empty query tokenises to nothing, so every candidate scores 0.0 —
    // a total tie across the whole corpus.
    let reranker = heap_reranker(3);
    let candidates = topical_corpus(10);
    let result = reranker.rerank("", &candidates, 10).unwrap();
    let expected: Vec<&str> = (0..10).map(|i| candidates[i].id.as_str()).collect();
    assert_eq!(result.ranking.ids(), expected);
}

#[test]
fn ties_broken_stably_when_candidates_are_identical_bubblesort() {
    let reranker = bubble_reranker(3);
    let candidates: Vec<SetwiseCandidate> = (0..10)
        .map(|i| SetwiseCandidate::new(format!("doc{i}"), "same passage same passage"))
        .collect();
    let result = reranker.rerank("same passage", &candidates, 10).unwrap();
    let expected: Vec<&str> = (0..10).map(|i| candidates[i].id.as_str()).collect();
    assert_eq!(result.ranking.ids(), expected);
}

// ── misc: rerank_default, score, ranking bookkeeping ──────────────────────────

#[test]
fn rerank_default_uses_config_top_m() {
    let reranker = SetwiseReranker::new(SetwiseConfig::new().with_top_m(3));
    let candidates = topical_corpus(10);
    let result = reranker
        .rerank_default("rust programming", &candidates)
        .unwrap();
    assert_eq!(result.ranking.len(), 3);
}

#[test]
fn rerank_single_candidate_returns_it() {
    let reranker = heap_reranker(4);
    let candidates = vec![SetwiseCandidate::new("only", "the one and only document")];
    let result = reranker.rerank("document", &candidates, 1).unwrap();
    assert_eq!(result.ranking.len(), 1);
    assert_eq!(result.ranking.entries[0].candidate.id, "only");
}

#[test]
fn rerank_result_reports_strategy_and_comparison_size() {
    let candidates = topical_corpus(6);
    let heap_result = heap_reranker(3).rerank("rust", &candidates, 2).unwrap();
    assert_eq!(heap_result.strategy, SetwiseSortStrategy::Heapsort);
    assert_eq!(heap_result.comparison_size, 3);

    let bubble_result = bubble_reranker(5).rerank("rust", &candidates, 2).unwrap();
    assert_eq!(bubble_result.strategy, SetwiseSortStrategy::Bubblesort);
    assert_eq!(bubble_result.comparison_size, 5);
}

#[test]
fn comparison_size_clamped_to_candidate_count_when_k_exceeds_n() {
    let candidates = topical_corpus(3);
    let reranker = heap_reranker(10);
    let result = reranker.rerank("rust", &candidates, 3).unwrap();
    assert_eq!(result.comparison_size, 3);
}

#[test]
fn score_is_deterministic() {
    let reranker = heap_reranker(4);
    let s1 = reranker.score("rust memory safety", "rust systems programming memory");
    let s2 = reranker.score("rust memory safety", "rust systems programming memory");
    assert!((s1 - s2).abs() < f32::EPSILON);
}

#[test]
fn score_higher_for_lexically_closer_text() {
    let reranker = heap_reranker(4);
    let relevant = reranker.score(
        "rust memory safety systems",
        "rust systems memory safety ownership",
    );
    let irrelevant = reranker.score(
        "rust memory safety systems",
        "banana smoothie recipe kitchen",
    );
    assert!(relevant > irrelevant);
}

#[test]
fn ranking_scores_are_descending_within_top_m() {
    let reranker = heap_reranker(4);
    let candidates = topical_corpus(20);
    let result = reranker
        .rerank("python data science machine learning", &candidates, 6)
        .unwrap();
    for pair in result.ranking.entries.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

#[test]
fn ranking_new_rank_and_original_index_are_consistent() {
    let reranker = bubble_reranker(3);
    let candidates = topical_corpus(12);
    let result = reranker
        .rerank("history ancient rome empire", &candidates, 5)
        .unwrap();
    for (expected_rank, entry) in result.ranking.entries.iter().enumerate() {
        assert_eq!(entry.new_rank, expected_rank);
        assert_eq!(entry.candidate, candidates[entry.original_index]);
    }
}

#[test]
fn setwise_reranker_default_uses_default_config() {
    let reranker = SetwiseReranker::default();
    assert_eq!(reranker.config, SetwiseConfig::default());
}

#[test]
fn rerank_top_m_equal_to_candidate_count_returns_full_permutation() {
    let reranker = bubble_reranker(3);
    let candidates = topical_corpus(11);
    let result = reranker
        .rerank("ocean marine biology coral", &candidates, 11)
        .unwrap();
    let mut ids: Vec<&str> = result.ranking.ids();
    ids.sort_unstable();
    let mut expected: Vec<&str> = candidates.iter().map(|c| c.id.as_str()).collect();
    expected.sort_unstable();
    assert_eq!(ids, expected);
}

#[test]
fn heapsort_top_m_one_returns_the_single_best_candidate() {
    let reranker = heap_reranker(4);
    let candidates = topical_corpus(30);
    let query = "algebra calculus geometry proof mathematics";
    let result = reranker.rerank(query, &candidates, 1).unwrap();
    let expected = brute_force_top_m(&reranker, query, &candidates, 1);
    assert_eq!(result.ranking.ids(), vec![expected[0].as_str()]);
}
