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

use crate::retrieval_diversity::DiversityScorer;
use crate::retrieval_diversity::metrics::{
    alpha_ndcg, cosine, embed, intra_list_diversity, subtopic_recall,
};
use crate::retrieval_diversity::types::{
    DiversityMetrics, RetrievalDiversityConfig, RetrievalDiversityError,
};

// ── Helpers ───────────────────────────────────────────────────────────────

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

fn scorer() -> DiversityScorer {
    DiversityScorer::new(RetrievalDiversityConfig::default())
}

// ── RetrievalDiversityConfig ──────────────────────────────────────────────

#[test]
fn config_default_alpha() {
    assert!(approx(RetrievalDiversityConfig::default().alpha, 0.5));
}

#[test]
fn config_default_dim() {
    assert_eq!(RetrievalDiversityConfig::default().dim, 128);
}

#[test]
fn config_new_matches_default() {
    assert_eq!(
        RetrievalDiversityConfig::new(),
        RetrievalDiversityConfig::default()
    );
}

#[test]
fn config_with_alpha_sets_value() {
    let c = RetrievalDiversityConfig::new().with_alpha(0.25);
    assert!(approx(c.alpha, 0.25));
}

#[test]
fn config_with_alpha_clamps_high() {
    let c = RetrievalDiversityConfig::new().with_alpha(5.0);
    assert!(approx(c.alpha, 1.0));
}

#[test]
fn config_with_alpha_clamps_low() {
    let c = RetrievalDiversityConfig::new().with_alpha(-3.0);
    assert!(approx(c.alpha, 0.0));
}

#[test]
fn config_with_dim_sets_value() {
    let c = RetrievalDiversityConfig::new().with_dim(64);
    assert_eq!(c.dim, 64);
}

#[test]
fn config_builders_chain() {
    let c = RetrievalDiversityConfig::new().with_alpha(0.3).with_dim(32);
    assert!(approx(c.alpha, 0.3));
    assert_eq!(c.dim, 32);
}

#[test]
fn config_clone_eq() {
    let c = RetrievalDiversityConfig::new().with_dim(16);
    assert_eq!(c.clone(), c);
}

// ── DiversityMetrics ───────────────────────────────────────────────────────

#[test]
fn metrics_new_fields() {
    let m = DiversityMetrics::new(0.1, 0.2, 0.3);
    assert!(approx(m.ild, 0.1));
    assert!(approx(m.s_recall, 0.2));
    assert!(approx(m.alpha_ndcg, 0.3));
}

#[test]
fn metrics_default_zero() {
    let m = DiversityMetrics::default();
    assert!(approx(m.ild, 0.0));
    assert!(approx(m.s_recall, 0.0));
    assert!(approx(m.alpha_ndcg, 0.0));
}

// ── embed / cosine ─────────────────────────────────────────────────────────

#[test]
fn embed_zero_dim_is_empty() {
    assert!(embed("hello world", 0).is_empty());
}

#[test]
fn embed_is_l2_normalised() {
    let v = embed("rust async tokio runtime", 128);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(approx(norm, 1.0));
}

#[test]
fn embed_is_deterministic() {
    assert_eq!(embed("alpha beta gamma", 64), embed("alpha beta gamma", 64));
}

#[test]
fn cosine_identical_is_one() {
    let v = embed("rust async tokio", 128);
    assert!(approx(cosine(&v, &v), 1.0));
}

#[test]
fn cosine_disjoint_vocab_is_zero() {
    let a = embed("rust async tokio runtime", 256);
    let b = embed("python pandas dataframe numpy", 256);
    assert!(approx(cosine(&a, &b), 0.0));
}

#[test]
fn cosine_length_mismatch_is_zero() {
    assert!(approx(cosine(&[1.0, 0.0], &[1.0]), 0.0));
}

#[test]
fn cosine_empty_is_zero() {
    let empty: [f32; 0] = [];
    assert!(approx(cosine(&empty, &empty), 0.0));
}

#[test]
fn cosine_zero_vector_is_zero() {
    assert!(approx(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0));
}

// ── Intra-List Diversity ───────────────────────────────────────────────────

#[test]
fn ild_identical_texts_is_zero() {
    let texts = ["rust async tokio", "rust async tokio", "rust async tokio"];
    assert!(approx(scorer().intra_list_diversity(&texts, 3), 0.0));
}

#[test]
fn ild_orthogonal_texts_is_high() {
    let texts = [
        "rust async tokio runtime",
        "python pandas dataframe numpy",
        "java spring boot hibernate",
    ];
    let ild = scorer().intra_list_diversity(&texts, 3);
    assert!(ild > 0.99, "orthogonal ILD should be near 1.0, got {ild}");
}

#[test]
fn ild_single_text_is_zero() {
    let texts = ["only one document here"];
    assert!(approx(scorer().intra_list_diversity(&texts, 1), 0.0));
}

#[test]
fn ild_empty_is_zero() {
    let texts: [&str; 0] = [];
    assert!(approx(scorer().intra_list_diversity(&texts, 5), 0.0));
}

#[test]
fn ild_k_zero_is_zero() {
    let texts = ["rust async tokio", "python pandas frame"];
    assert!(approx(scorer().intra_list_diversity(&texts, 0), 0.0));
}

#[test]
fn ild_in_unit_range() {
    let texts = [
        "rust async tokio runtime",
        "rust pandas dataframe runtime",
        "python pandas dataframe numpy",
    ];
    let ild = scorer().intra_list_diversity(&texts, 3);
    assert!(ild >= 0.0 && ild <= 1.0);
}

#[test]
fn ild_respects_k_cutoff() {
    // Top-2 are identical (ILD 0); a divergent 3rd is excluded at k = 2.
    let texts = [
        "rust async tokio",
        "rust async tokio",
        "python pandas numpy dataframe",
    ];
    assert!(approx(scorer().intra_list_diversity(&texts, 2), 0.0));
    assert!(scorer().intra_list_diversity(&texts, 3) > 0.0);
}

#[test]
fn ild_free_fn_matches_method() {
    let texts = ["rust async tokio", "python pandas frame"];
    let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embed(t, 128)).collect();
    assert!(approx(
        intra_list_diversity(&embeddings, 2),
        scorer().intra_list_diversity(&texts, 2)
    ));
}

// ── Subtopic Recall ────────────────────────────────────────────────────────

#[test]
fn s_recall_all_covered_is_one() {
    let subtopics = vec![vec![0_usize], vec![1], vec![2]];
    assert!(approx(scorer().subtopic_recall(&subtopics, 3, 3), 1.0));
}

#[test]
fn s_recall_none_covered_is_zero() {
    // No result carries any subtopic label.
    let subtopics = vec![vec![], vec![], vec![]];
    assert!(approx(scorer().subtopic_recall(&subtopics, 3, 3), 0.0));
}

#[test]
fn s_recall_partial_fraction() {
    // Covers subtopics {0, 1} out of 4 → 0.5.
    let subtopics = vec![vec![0_usize], vec![1], vec![0]];
    assert!(approx(scorer().subtopic_recall(&subtopics, 4, 3), 0.5));
}

#[test]
fn s_recall_zero_total_is_zero() {
    let subtopics = vec![vec![0_usize], vec![1]];
    assert!(approx(scorer().subtopic_recall(&subtopics, 0, 2), 0.0));
}

#[test]
fn s_recall_respects_k_cutoff() {
    // Subtopic 2 only appears at rank 3; at k = 2 only {0, 1} are covered.
    let subtopics = vec![vec![0_usize], vec![1], vec![2]];
    assert!(approx(
        scorer().subtopic_recall(&subtopics, 3, 2),
        2.0 / 3.0
    ));
}

#[test]
fn s_recall_ignores_out_of_range_labels() {
    // Label 9 is >= total (3) and must be ignored; only {0} counts → 1/3.
    let subtopics = vec![vec![0_usize, 9]];
    assert!(approx(
        scorer().subtopic_recall(&subtopics, 3, 1),
        1.0 / 3.0
    ));
}

#[test]
fn s_recall_dedups_repeated_subtopics() {
    // Same subtopic many times still counts once → 1/2.
    let subtopics = vec![vec![0_usize], vec![0], vec![0]];
    assert!(approx(scorer().subtopic_recall(&subtopics, 2, 3), 0.5));
}

#[test]
fn s_recall_multi_label_results() {
    let subtopics = vec![vec![0_usize, 1], vec![2, 3]];
    assert!(approx(scorer().subtopic_recall(&subtopics, 4, 2), 1.0));
}

#[test]
fn s_recall_in_unit_range() {
    let subtopics = vec![vec![0_usize], vec![2], vec![1]];
    let r = scorer().subtopic_recall(&subtopics, 5, 3);
    assert!(r >= 0.0 && r <= 1.0);
}

#[test]
fn s_recall_free_fn_matches_method() {
    let subtopics = vec![vec![0_usize], vec![1]];
    assert!(approx(
        subtopic_recall(&subtopics, 3, 2),
        scorer().subtopic_recall(&subtopics, 3, 2)
    ));
}

// ── α-nDCG ─────────────────────────────────────────────────────────────────

#[test]
fn alpha_ndcg_diverse_beats_redundant() {
    // Same subtopic multiset {0, 0, 1}; only the ordering differs. The diverse
    // ordering surfaces the novel subtopic 1 earlier (rank 2) than the redundant
    // ordering (rank 3), so it must score strictly higher under the novelty
    // discount. Both normalise against the *same* ideal greedy ordering.
    let diverse = vec![vec![0_usize], vec![1], vec![0]];
    let redundant = vec![vec![0_usize], vec![0], vec![1]];
    let s = scorer();
    let d = s.alpha_ndcg(&diverse, 3);
    let r = s.alpha_ndcg(&redundant, 3);
    assert!(d > r, "diverse ranking ({d}) should beat redundant ({r})");
}

#[test]
fn alpha_ndcg_diverse_ordering_is_ideal() {
    // The diverse ordering already is the ideal greedy ordering → 1.0.
    let diverse = vec![vec![0_usize], vec![1], vec![2]];
    assert!(approx(scorer().alpha_ndcg(&diverse, 3), 1.0));
}

#[test]
fn alpha_ndcg_in_unit_range() {
    let subtopics = vec![vec![0_usize], vec![0], vec![1], vec![2], vec![1]];
    let v = scorer().alpha_ndcg(&subtopics, 5);
    assert!(v >= 0.0 && v <= 1.0);
}

#[test]
fn alpha_ndcg_redundant_below_one() {
    // A redundant-first ordering is sub-ideal when a fresh subtopic exists.
    let subtopics = vec![vec![0_usize], vec![0], vec![1]];
    let v = scorer().alpha_ndcg(&subtopics, 3);
    assert!(v < 1.0 && v > 0.0, "got {v}");
}

#[test]
fn alpha_ndcg_alpha_one_removes_discount() {
    // With alpha = 1.0 there is no novelty discount: every result with a label
    // contributes full gain, so order no longer matters and both orderings hit
    // the ideal (1.0).
    let s = DiversityScorer::new(RetrievalDiversityConfig::new().with_alpha(1.0));
    let redundant = vec![vec![0_usize], vec![0], vec![0]];
    let diverse = vec![vec![0_usize], vec![1], vec![2]];
    assert!(approx(s.alpha_ndcg(&redundant, 3), 1.0));
    assert!(approx(s.alpha_ndcg(&diverse, 3), 1.0));
}

#[test]
fn alpha_ndcg_alpha_one_order_invariant() {
    // alpha = 1.0: permuting equal single-label results leaves the score at 1.0
    // (each permutation is itself an ideal ordering under no discount).
    let s = DiversityScorer::new(RetrievalDiversityConfig::new().with_alpha(1.0));
    let a = vec![vec![0_usize], vec![1], vec![2]];
    let b = vec![vec![2_usize], vec![0], vec![1]];
    assert!(approx(s.alpha_ndcg(&a, 3), s.alpha_ndcg(&b, 3)));
}

#[test]
fn alpha_ndcg_alpha_zero_max_discount() {
    // alpha = 0.0: a repeated subtopic still earns full (1 - 0)^c = 1 each time,
    // so a redundant list reaches the ideal too. Distinct from alpha = 1 only in
    // how partial coverage is scored, exercised elsewhere.
    let s = DiversityScorer::new(RetrievalDiversityConfig::new().with_alpha(0.0));
    let redundant = vec![vec![0_usize], vec![0]];
    let v = s.alpha_ndcg(&redundant, 2);
    assert!(v >= 0.0 && v <= 1.0);
}

#[test]
fn alpha_ndcg_no_subtopics_is_zero() {
    let subtopics = vec![vec![], vec![]];
    assert!(approx(scorer().alpha_ndcg(&subtopics, 2), 0.0));
}

#[test]
fn alpha_ndcg_empty_list_is_zero() {
    let subtopics: Vec<Vec<usize>> = vec![];
    assert!(approx(scorer().alpha_ndcg(&subtopics, 5), 0.0));
}

#[test]
fn alpha_ndcg_single_result_is_one() {
    let subtopics = vec![vec![0_usize]];
    assert!(approx(scorer().alpha_ndcg(&subtopics, 1), 1.0));
}

#[test]
fn alpha_ndcg_respects_k_cutoff() {
    // At k = 1 only the first result is scored; both orderings start with a
    // single fresh subtopic, so the redundant list is already ideal at k = 1.
    let redundant = vec![vec![0_usize], vec![0], vec![0]];
    assert!(approx(scorer().alpha_ndcg(&redundant, 1), 1.0));
}

#[test]
fn alpha_ndcg_free_fn_matches_method() {
    let subtopics = vec![vec![0_usize], vec![0], vec![1]];
    assert!(approx(
        alpha_ndcg(&subtopics, 0.5, 3),
        scorer().alpha_ndcg(&subtopics, 3)
    ));
}

#[test]
fn alpha_ndcg_alpha_changes_score() {
    // alpha genuinely affects the score of a sub-ideal redundant ranking:
    // a steep discount (low alpha) and a shallow discount (high alpha) must not
    // collapse to the same value for `[[0], [0], [1]]`.
    let redundant = vec![vec![0_usize], vec![0], vec![1]];
    let low = DiversityScorer::new(RetrievalDiversityConfig::new().with_alpha(0.1));
    let high = DiversityScorer::new(RetrievalDiversityConfig::new().with_alpha(0.9));
    let v_low = low.alpha_ndcg(&redundant, 3);
    let v_high = high.alpha_ndcg(&redundant, 3);
    assert!((v_low - v_high).abs() > 1e-3, "low={v_low} high={v_high}");
}

#[test]
fn alpha_ndcg_single_subtopic_always_ideal() {
    // A fully-redundant single-subtopic ranking is its own ideal ordering, so it
    // scores 1.0 regardless of the novelty-discount parameter.
    let redundant = vec![vec![0_usize], vec![0], vec![0]];
    for &a in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let s = DiversityScorer::new(RetrievalDiversityConfig::new().with_alpha(a));
        assert!(approx(s.alpha_ndcg(&redundant, 3), 1.0), "alpha={a}");
    }
}

// ── compute ────────────────────────────────────────────────────────────────

#[test]
fn compute_returns_all_three() {
    let texts = [
        "rust async tokio",
        "python pandas frame",
        "rust async tokio",
    ];
    let subtopics = vec![vec![0_usize], vec![1], vec![0]];
    let m = scorer().compute(&texts, &subtopics, 2, 3).unwrap();
    assert!(m.ild >= 0.0 && m.ild <= 1.0);
    assert!(approx(m.s_recall, 1.0));
    assert!(m.alpha_ndcg >= 0.0 && m.alpha_ndcg <= 1.0);
}

#[test]
fn compute_matches_individual_methods() {
    let texts = ["rust async tokio", "python pandas frame"];
    let subtopics = vec![vec![0_usize], vec![1]];
    let s = scorer();
    let m = s.compute(&texts, &subtopics, 2, 2).unwrap();
    assert!(approx(m.ild, s.intra_list_diversity(&texts, 2)));
    assert!(approx(m.s_recall, s.subtopic_recall(&subtopics, 2, 2)));
    assert!(approx(m.alpha_ndcg, s.alpha_ndcg(&subtopics, 2)));
}

#[test]
fn compute_empty_texts_errors() {
    let texts: [&str; 0] = [];
    let subtopics: Vec<Vec<usize>> = vec![];
    let err = scorer().compute(&texts, &subtopics, 1, 1).unwrap_err();
    assert!(matches!(err, RetrievalDiversityError::EmptyResults));
}

#[test]
fn compute_empty_subtopics_errors() {
    let texts = ["a doc"];
    let subtopics: Vec<Vec<usize>> = vec![];
    let err = scorer().compute(&texts, &subtopics, 1, 1).unwrap_err();
    assert!(matches!(err, RetrievalDiversityError::EmptyResults));
}

#[test]
fn compute_length_mismatch_errors() {
    let texts = ["a doc", "another doc"];
    let subtopics = vec![vec![0_usize]];
    let err = scorer().compute(&texts, &subtopics, 1, 2).unwrap_err();
    assert!(matches!(err, RetrievalDiversityError::LengthMismatch));
}

#[test]
fn compute_error_display_messages() {
    assert_eq!(
        RetrievalDiversityError::EmptyResults.to_string(),
        "results must not be empty"
    );
    assert_eq!(
        RetrievalDiversityError::LengthMismatch.to_string(),
        "texts/subtopics length mismatch"
    );
}

// ── Determinism ──────────────────────────────────────────────────────────────

#[test]
fn compute_is_deterministic() {
    let texts = [
        "rust async tokio",
        "python pandas frame",
        "java spring boot",
    ];
    let subtopics = vec![vec![0_usize], vec![1], vec![2]];
    let a = scorer().compute(&texts, &subtopics, 3, 3).unwrap();
    let b = scorer().compute(&texts, &subtopics, 3, 3).unwrap();
    assert_eq!(a, b);
}

#[test]
fn alpha_ndcg_is_deterministic() {
    let subtopics = vec![vec![0_usize], vec![1], vec![0], vec![2], vec![1]];
    let a = scorer().alpha_ndcg(&subtopics, 5);
    let b = scorer().alpha_ndcg(&subtopics, 5);
    assert!(approx(a, b));
}

#[test]
fn scorer_default_matches_config_default() {
    let s = DiversityScorer::default();
    assert_eq!(s.config, RetrievalDiversityConfig::default());
}
