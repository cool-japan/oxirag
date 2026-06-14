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
    clippy::doc_markdown
)]
//! Tests for the `diversity_rank` module.

use crate::diversity_rank::ranker::{DiversityRanker, embed};
use crate::diversity_rank::types::{
    DiversityConfig, DiversityRankError, DiversitySelection, SimilarityKind,
};
use crate::types::{Document, DocumentId, SearchResult};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

fn ranker_k(k: usize) -> DiversityRanker {
    DiversityRanker::new(DiversityConfig::default().with_k(k))
}

/// A near-duplicate pair (a, b) plus a clearly distinct item (c).
fn duplicate_and_distinct() -> Vec<SearchResult> {
    vec![
        make_result("a", "rust async tokio runtime futures executor", 0.90),
        make_result("b", "rust async tokio runtime futures executor", 0.88),
        make_result("c", "python pandas numpy dataframe analytics", 0.50),
    ]
}

// ── config defaults ──────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = DiversityConfig::default();
    assert_eq!(cfg.k, 5);
    assert_eq!(cfg.quality_weight, 1.0);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.similarity, SimilarityKind::Cosine);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(DiversityConfig::new(), DiversityConfig::default());
}

#[test]
fn similarity_kind_default_is_cosine() {
    assert_eq!(SimilarityKind::default(), SimilarityKind::Cosine);
}

// ── config builders ──────────────────────────────────────────────────────────

#[test]
fn builder_with_k() {
    let cfg = DiversityConfig::new().with_k(11);
    assert_eq!(cfg.k, 11);
}

#[test]
fn builder_with_quality_weight() {
    let cfg = DiversityConfig::new().with_quality_weight(2.5);
    assert_eq!(cfg.quality_weight, 2.5);
}

#[test]
fn builder_with_dim() {
    let cfg = DiversityConfig::new().with_dim(64);
    assert_eq!(cfg.dim, 64);
}

#[test]
fn builder_with_similarity() {
    let cfg = DiversityConfig::new().with_similarity(SimilarityKind::Cosine);
    assert_eq!(cfg.similarity, SimilarityKind::Cosine);
}

#[test]
fn builders_chain() {
    let cfg = DiversityConfig::new()
        .with_k(3)
        .with_quality_weight(0.5)
        .with_dim(32)
        .with_similarity(SimilarityKind::Cosine);
    assert_eq!(cfg.k, 3);
    assert_eq!(cfg.quality_weight, 0.5);
    assert_eq!(cfg.dim, 32);
    assert_eq!(cfg.similarity, SimilarityKind::Cosine);
}

#[test]
fn config_clone_eq() {
    let cfg = DiversityConfig::new().with_k(7);
    assert_eq!(cfg.clone(), cfg);
}

#[test]
fn ranker_exposes_config() {
    let ranker = ranker_k(4);
    assert_eq!(ranker.config().k, 4);
}

// ── cosine similarity ────────────────────────────────────────────────────────

#[test]
fn cosine_identical_is_one() {
    let v = vec![0.3_f32, 0.4, 0.5];
    assert!((DiversityRanker::cosine(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_orthogonal_is_zero() {
    let a = vec![1.0_f32, 0.0];
    let b = vec![0.0_f32, 1.0];
    assert!(DiversityRanker::cosine(&a, &b).abs() < 1e-6);
}

#[test]
fn cosine_mismatched_length_is_zero() {
    assert_eq!(DiversityRanker::cosine(&[1.0, 2.0], &[1.0]), 0.0);
}

#[test]
fn cosine_empty_is_zero() {
    assert_eq!(DiversityRanker::cosine(&[], &[]), 0.0);
}

#[test]
fn cosine_zero_vector_is_zero() {
    let a = vec![0.0_f32, 0.0];
    let b = vec![1.0_f32, 1.0];
    assert_eq!(DiversityRanker::cosine(&a, &b), 0.0);
}

#[test]
fn cosine_opposite_is_negative_one() {
    let a = vec![1.0_f32, 0.0];
    let b = vec![-1.0_f32, 0.0];
    assert!((DiversityRanker::cosine(&a, &b) + 1.0).abs() < 1e-6);
}

#[test]
fn cosine_is_clamped() {
    // Even with scaling the result must stay within [-1, 1].
    let a = vec![2.0_f32, 1.0, 3.0];
    let b = vec![4.0_f32, 2.0, 6.0]; // exactly parallel
    let c = DiversityRanker::cosine(&a, &b);
    assert!(c <= 1.0 && c >= -1.0);
    assert!((c - 1.0).abs() < 1e-6);
}

// ── embed ────────────────────────────────────────────────────────────────────

#[test]
fn embed_dim_zero_is_empty() {
    assert!(embed("anything", 0).is_empty());
}

#[test]
fn embed_length_matches_dim() {
    assert_eq!(embed("hello world", 64).len(), 64);
}

#[test]
fn embed_is_l2_normalised() {
    let v = embed("rust programming language systems", 128);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn embed_is_deterministic() {
    assert_eq!(
        embed("deterministic text here", 96),
        embed("deterministic text here", 96)
    );
}

#[test]
fn embed_identical_text_cosine_is_one() {
    let a = embed("alpha beta gamma delta", 128);
    let b = embed("alpha beta gamma delta", 128);
    assert!((DiversityRanker::cosine(&a, &b) - 1.0).abs() < 1e-6);
}

#[test]
fn embed_empty_text_is_zero_vector() {
    let v = embed("", 32);
    assert_eq!(v.len(), 32);
    assert!(v.iter().all(|&x| x == 0.0));
}

// ── select: basic bounds ─────────────────────────────────────────────────────

#[test]
fn select_empty_returns_empty() {
    let ranker = ranker_k(5);
    assert!(ranker.select(&[]).is_empty());
}

#[test]
fn select_returns_at_most_k() {
    let ranker = ranker_k(2);
    let results = vec![
        make_result("a", "alpha topic one", 0.9),
        make_result("b", "beta topic two", 0.8),
        make_result("c", "gamma topic three", 0.7),
        make_result("d", "delta topic four", 0.6),
    ];
    assert!(ranker.select(&results).len() <= 2);
    assert_eq!(ranker.select(&results).len(), 2);
}

#[test]
fn select_k_larger_than_candidates() {
    let ranker = ranker_k(10);
    let results = vec![
        make_result("a", "alpha unique terms here", 0.9),
        make_result("b", "beta separate words there", 0.8),
    ];
    assert_eq!(ranker.select(&results).len(), 2);
}

#[test]
fn select_single_candidate() {
    let ranker = ranker_k(5);
    let results = vec![make_result("a", "lonely document", 0.5)];
    let out = ranker.select(&results);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].document.id.as_str(), "a");
}

#[test]
fn select_reranks_from_zero() {
    let ranker = ranker_k(3);
    let results = vec![
        make_result("a", "alpha one distinct", 0.9),
        make_result("b", "beta two separate", 0.8),
        make_result("c", "gamma three different", 0.7),
    ];
    let out = ranker.select(&results);
    for (i, r) in out.iter().enumerate() {
        assert_eq!(r.rank, i);
    }
}

// ── select: first pick is highest quality ────────────────────────────────────

#[test]
fn select_first_is_highest_quality() {
    let ranker = ranker_k(3);
    let results = vec![
        make_result("low", "alpha terms one", 0.20),
        make_result("high", "beta terms two", 0.95),
        make_result("mid", "gamma terms three", 0.55),
    ];
    let out = ranker.select(&results);
    assert_eq!(out[0].document.id.as_str(), "high");
}

#[test]
fn select_with_embeddings_first_is_highest_quality() {
    let ranker = ranker_k(2);
    let qualities = vec![0.1, 0.9, 0.4];
    let embeddings = vec![
        vec![1.0_f32, 0.0, 0.0],
        vec![0.0_f32, 1.0, 0.0],
        vec![0.0_f32, 0.0, 1.0],
    ];
    let sel = ranker
        .select_with_embeddings(&qualities, &embeddings, 2)
        .unwrap();
    assert_eq!(sel[0], 1);
}

#[test]
fn select_first_quality_tie_picks_lowest_index() {
    let ranker = ranker_k(1);
    let qualities = vec![0.8, 0.8, 0.8];
    let embeddings = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0], vec![1.0_f32, 1.0]];
    let sel = ranker
        .select_with_embeddings(&qualities, &embeddings, 1)
        .unwrap();
    assert_eq!(sel, vec![0]);
}

// ── select: diversity beats pure top-k ───────────────────────────────────────

#[test]
fn select_second_pick_is_distinct_not_duplicate() {
    // a and b are textual near-duplicates; c is distinct but lower scoring.
    // Pure top-k would pick a then b; diversity must pick a then c.
    let ranker = ranker_k(2);
    let results = duplicate_and_distinct();
    let out = ranker.select(&results);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].document.id.as_str(), "a");
    assert_eq!(
        out[1].document.id.as_str(),
        "c",
        "diversity must beat top-k"
    );
}

#[test]
fn select_with_embeddings_avoids_redundant_pick() {
    // index 0 highest quality; index 1 is its near-duplicate; index 2 distinct.
    let ranker = ranker_k(2);
    let qualities = vec![1.0, 0.95, 0.6];
    let embeddings = vec![
        vec![1.0_f32, 0.0, 0.0],
        vec![0.99_f32, 0.01, 0.0],
        vec![0.0_f32, 0.0, 1.0],
    ];
    let sel = ranker
        .select_with_embeddings(&qualities, &embeddings, 2)
        .unwrap();
    assert_eq!(sel[0], 0);
    assert_eq!(
        sel[1], 2,
        "should skip the redundant high-quality duplicate"
    );
}

#[test]
fn select_high_quality_duplicate_still_loses_to_distinct() {
    // Even when the duplicate scores higher than the distinct item, the product
    // penalty should still favour diversity.
    let ranker = ranker_k(2);
    let results = vec![
        make_result("a", "machine learning neural network training", 0.90),
        make_result("b", "machine learning neural network training", 0.85),
        make_result("c", "gardening flowers soil seeds water", 0.40),
    ];
    let out = ranker.select(&results);
    assert_eq!(out[1].document.id.as_str(), "c");
}

#[test]
fn select_full_k_returns_all_when_distinct() {
    let ranker = ranker_k(3);
    let results = vec![
        make_result("a", "alpha unique alpha", 0.9),
        make_result("b", "beta separate beta", 0.8),
        make_result("c", "gamma distinct gamma", 0.7),
    ];
    let out = ranker.select(&results);
    let ids: Vec<_> = out
        .iter()
        .map(|r| r.document.id.as_str().to_string())
        .collect();
    assert_eq!(out.len(), 3);
    assert!(ids.contains(&"a".to_string()));
    assert!(ids.contains(&"b".to_string()));
    assert!(ids.contains(&"c".to_string()));
}

// ── select: all-identical embeddings → quality order ─────────────────────────

#[test]
fn select_identical_embeddings_returns_k_by_quality() {
    // All identical content → identical embeddings → no diversity signal.
    // Must still return k items ordered by quality (greedy first pick + product
    // gains where the only signal left is quality²).
    let ranker = ranker_k(3);
    let results = vec![
        make_result("a", "same same same words", 0.30),
        make_result("b", "same same same words", 0.90),
        make_result("c", "same same same words", 0.60),
        make_result("d", "same same same words", 0.10),
    ];
    let out = ranker.select(&results);
    // Spec: all-identical embeddings still return k items, ordered by quality.
    assert_eq!(out.len(), 3);
    let ids: Vec<&str> = out.iter().map(|r| r.document.id.as_str()).collect();
    // Descending quality: b (0.90), c (0.60), a (0.30); d (0.10) drops out.
    assert_eq!(ids, vec!["b", "c", "a"]);
}

#[test]
fn select_with_identical_embeddings_explicit() {
    let ranker = ranker_k(3);
    let qualities = vec![0.3, 0.9, 0.6, 0.1];
    let emb = vec![1.0_f32, 1.0, 1.0];
    let embeddings = vec![emb.clone(), emb.clone(), emb.clone(), emb];
    let sel = ranker
        .select_with_embeddings(&qualities, &embeddings, 3)
        .unwrap();
    // First pick is the highest quality (index 1); identical embeddings give
    // zero volume, so remaining slots are filled by descending quality.
    assert_eq!(sel, vec![1, 2, 0]);
}

#[test]
fn select_identical_embeddings_fills_k_by_quality_when_volume_zero() {
    // With identical embeddings every post-first product gain is zero, so the
    // ranker falls back to descending-quality fill to still reach k.
    let ranker = ranker_k(4);
    let qualities = vec![0.5, 0.9, 0.7];
    let emb = vec![1.0_f32, 0.0];
    let embeddings = vec![emb.clone(), emb.clone(), emb];
    let sel = ranker
        .select_with_embeddings(&qualities, &embeddings, 4)
        .unwrap();
    // k=4 capped to 3 candidates; descending quality → 1 (0.9), 2 (0.7), 0 (0.5).
    assert_eq!(sel, vec![1, 2, 0]);
}

// ── diversity_score ──────────────────────────────────────────────────────────

#[test]
fn diversity_score_orthogonal_is_one() {
    let ranker = ranker_k(2);
    let qualities = vec![1.0, 1.0];
    let embeddings = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let sel = ranker.select_detailed(&qualities, &embeddings, 2).unwrap();
    assert_eq!(sel.selected.len(), 2);
    assert!((sel.diversity_score - 1.0).abs() < 1e-6);
}

#[test]
fn diversity_score_single_item_is_zero() {
    let ranker = ranker_k(1);
    let qualities = vec![1.0, 1.0];
    let embeddings = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let sel = ranker.select_detailed(&qualities, &embeddings, 1).unwrap();
    assert_eq!(sel.selected.len(), 1);
    assert_eq!(sel.diversity_score, 0.0);
}

#[test]
fn diversity_score_higher_than_naive_top_k_on_redundant_set() {
    // Build a redundant set: two near-duplicates (high quality) + one distinct.
    let results = duplicate_and_distinct();
    let embeddings: Vec<Vec<f32>> = results
        .iter()
        .map(|r| embed(&r.document.content, 128))
        .collect();
    let qualities: Vec<f32> = results.iter().map(|r| r.score).collect();
    let ranker = ranker_k(2);

    // DPP selection.
    let dpp = ranker.select_detailed(&qualities, &embeddings, 2).unwrap();

    // Naive top-k by quality = indices 0 and 1 (the duplicate pair).
    let naive = vec![0usize, 1usize];
    let naive_score = pairwise_dissimilarity(&naive, &embeddings);

    assert!(
        dpp.diversity_score > naive_score,
        "DPP diversity {} should exceed naive top-k {}",
        dpp.diversity_score,
        naive_score
    );
}

#[test]
fn diversity_score_in_unit_range_for_nonneg_embeddings() {
    let results = vec![
        make_result("a", "alpha words here", 0.9),
        make_result("b", "beta words there", 0.8),
        make_result("c", "gamma words elsewhere", 0.7),
    ];
    let embeddings: Vec<Vec<f32>> = results
        .iter()
        .map(|r| embed(&r.document.content, 128))
        .collect();
    let qualities: Vec<f32> = results.iter().map(|r| r.score).collect();
    let ranker = ranker_k(3);
    let sel = ranker.select_detailed(&qualities, &embeddings, 3).unwrap();
    assert!(sel.diversity_score >= 0.0 && sel.diversity_score <= 1.0 + 1e-6);
}

/// Local re-implementation of mean pairwise dissimilarity for assertions.
fn pairwise_dissimilarity(indices: &[usize], embeddings: &[Vec<f32>]) -> f32 {
    if indices.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut pairs = 0u32;
    for (i, &a) in indices.iter().enumerate() {
        for &b in &indices[i + 1..] {
            total += 1.0 - DiversityRanker::cosine(&embeddings[a], &embeddings[b]);
            pairs += 1;
        }
    }
    total / pairs as f32
}

// ── error handling ───────────────────────────────────────────────────────────

#[test]
fn select_with_embeddings_empty_errors() {
    let ranker = ranker_k(5);
    let err = ranker.select_with_embeddings(&[], &[], 5).unwrap_err();
    assert!(matches!(err, DiversityRankError::EmptyCandidates));
}

#[test]
fn select_with_embeddings_length_mismatch_errors() {
    let ranker = ranker_k(5);
    let qualities = vec![0.5, 0.6];
    let embeddings = vec![vec![1.0_f32, 0.0]];
    let err = ranker
        .select_with_embeddings(&qualities, &embeddings, 5)
        .unwrap_err();
    assert!(matches!(err, DiversityRankError::LengthMismatch));
}

#[test]
fn select_detailed_propagates_empty_error() {
    let ranker = ranker_k(5);
    let err = ranker.select_detailed(&[], &[], 5).unwrap_err();
    assert!(matches!(err, DiversityRankError::EmptyCandidates));
}

#[test]
fn select_detailed_propagates_length_mismatch() {
    let ranker = ranker_k(5);
    let err = ranker
        .select_detailed(&[0.1, 0.2, 0.3], &[vec![1.0_f32]], 5)
        .unwrap_err();
    assert!(matches!(err, DiversityRankError::LengthMismatch));
}

#[test]
fn error_display_messages() {
    assert_eq!(
        DiversityRankError::EmptyCandidates.to_string(),
        "candidates must not be empty"
    );
    assert_eq!(
        DiversityRankError::LengthMismatch.to_string(),
        "qualities/embeddings length mismatch"
    );
}

#[test]
fn select_with_embeddings_zero_k_returns_empty() {
    let ranker = ranker_k(0);
    let qualities = vec![0.5, 0.6];
    let embeddings = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let sel = ranker
        .select_with_embeddings(&qualities, &embeddings, 0)
        .unwrap();
    assert!(sel.is_empty());
}

// ── determinism ──────────────────────────────────────────────────────────────

#[test]
fn select_is_deterministic() {
    let ranker = ranker_k(3);
    let results = duplicate_and_distinct();
    let a: Vec<_> = ranker
        .select(&results)
        .iter()
        .map(|r| r.document.id.as_str().to_string())
        .collect();
    let b: Vec<_> = ranker
        .select(&results)
        .iter()
        .map(|r| r.document.id.as_str().to_string())
        .collect();
    assert_eq!(a, b);
}

#[test]
fn select_with_embeddings_is_deterministic() {
    let ranker = ranker_k(3);
    let qualities = vec![0.9, 0.8, 0.7, 0.6];
    let embeddings = vec![
        vec![1.0_f32, 0.0, 0.0],
        vec![0.0_f32, 1.0, 0.0],
        vec![0.0_f32, 0.0, 1.0],
        vec![0.5_f32, 0.5, 0.0],
    ];
    let a = ranker
        .select_with_embeddings(&qualities, &embeddings, 3)
        .unwrap();
    let b = ranker
        .select_with_embeddings(&qualities, &embeddings, 3)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn select_detailed_is_deterministic() {
    let ranker = ranker_k(3);
    let qualities = vec![0.9, 0.8, 0.7];
    let embeddings = vec![
        vec![1.0_f32, 0.0, 0.0],
        vec![0.0_f32, 1.0, 0.0],
        vec![0.0_f32, 0.0, 1.0],
    ];
    let a = ranker.select_detailed(&qualities, &embeddings, 3).unwrap();
    let b = ranker.select_detailed(&qualities, &embeddings, 3).unwrap();
    assert_eq!(a, b);
}

// ── quality fallback (all-equal scores) ──────────────────────────────────────

#[test]
fn select_all_equal_scores_uses_rank_fallback() {
    // All scores equal (zero) → rank-based fallback means earliest stays first.
    let ranker = ranker_k(2);
    let results = vec![
        make_result("first", "alpha distinct one", 0.0),
        make_result("second", "beta distinct two", 0.0),
        make_result("third", "gamma distinct three", 0.0),
    ];
    let out = ranker.select(&results);
    assert_eq!(out[0].document.id.as_str(), "first");
}

#[test]
fn select_all_equal_nonzero_scores_uses_rank_fallback() {
    let ranker = ranker_k(1);
    let results = vec![
        make_result("first", "alpha terms", 0.5),
        make_result("second", "beta terms", 0.5),
    ];
    let out = ranker.select(&results);
    assert_eq!(out[0].document.id.as_str(), "first");
}

// ── quality weight influence ─────────────────────────────────────────────────

#[test]
fn high_quality_weight_still_selects_first_by_quality() {
    let ranker = DiversityRanker::new(DiversityConfig::new().with_k(2).with_quality_weight(5.0));
    let qualities = vec![0.2, 0.9, 0.5];
    let embeddings = vec![
        vec![1.0_f32, 0.0, 0.0],
        vec![0.0_f32, 1.0, 0.0],
        vec![0.0_f32, 0.0, 1.0],
    ];
    let sel = ranker
        .select_with_embeddings(&qualities, &embeddings, 2)
        .unwrap();
    assert_eq!(sel[0], 1);
}

#[test]
fn negative_quality_is_clamped_to_nonnegative() {
    // Negative qualities are clamped; a positive-quality item must win first.
    let ranker = ranker_k(1);
    let qualities = vec![-0.5, 0.1];
    let embeddings = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let sel = ranker
        .select_with_embeddings(&qualities, &embeddings, 1)
        .unwrap();
    assert_eq!(sel, vec![1]);
}

// ── selection / config struct plumbing ───────────────────────────────────────

#[test]
fn diversity_selection_new_fields() {
    let sel = DiversitySelection::new(vec![0, 2, 5], 0.42);
    assert_eq!(sel.selected, vec![0, 2, 5]);
    assert_eq!(sel.diversity_score, 0.42);
}

#[test]
fn diversity_selection_clone_eq() {
    let sel = DiversitySelection::new(vec![1, 3], 0.7);
    assert_eq!(sel.clone(), sel);
}

#[test]
fn ranker_clone_preserves_config() {
    let ranker = ranker_k(9);
    let cloned = ranker.clone();
    assert_eq!(cloned.config().k, 9);
}

// ── custom dim end-to-end ────────────────────────────────────────────────────

#[test]
fn select_with_small_dim_still_works() {
    let ranker = DiversityRanker::new(DiversityConfig::new().with_k(2).with_dim(16));
    let results = duplicate_and_distinct();
    let out = ranker.select(&results);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].document.id.as_str(), "a");
}

#[test]
fn select_detailed_full_set_distinct_high_score() {
    let ranker = ranker_k(3);
    let results = vec![
        make_result("a", "ocean waves tide salt", 0.9),
        make_result("b", "mountain rock cliff snow", 0.8),
        make_result("c", "desert sand dune heat", 0.7),
    ];
    let embeddings: Vec<Vec<f32>> = results
        .iter()
        .map(|r| embed(&r.document.content, 128))
        .collect();
    let qualities: Vec<f32> = results.iter().map(|r| r.score).collect();
    let sel = ranker.select_detailed(&qualities, &embeddings, 3).unwrap();
    assert_eq!(sel.selected.len(), 3);
    // Three lexically distinct docs → high diversity.
    assert!(sel.diversity_score > 0.5);
}
