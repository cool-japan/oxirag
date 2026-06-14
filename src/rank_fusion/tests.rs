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
//! Tests for multi-list rank/score fusion.

use crate::types::{Document, DocumentId, SearchResult};

use super::{FusionMethod, RankFusion, RankFusionConfig, RankFusionError, ScoreNormalization};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a `(DocumentId, score)` pair from a string id.
fn pair(id: &str, score: f32) -> (DocumentId, f32) {
    (DocumentId::from_string(id), score)
}

/// Build a `SearchResult` carrying a document with the given id.
fn result(id: &str, score: f32, rank: usize) -> SearchResult {
    let doc = Document::new(format!("content-{id}")).with_id(DocumentId::from_string(id));
    SearchResult::new(doc, score, rank)
}

/// Look up the fused score for a given id in `(DocumentId, f32)` output.
fn score_of(out: &[(DocumentId, f32)], id: &str) -> f32 {
    out.iter()
        .find(|(d, _)| d.as_str() == id)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("id {id} not found in output"))
}

/// Run an id-based fusion with the given method and no normalization.
fn fuse_raw(method: FusionMethod, lists: &[Vec<(DocumentId, f32)>]) -> Vec<(DocumentId, f32)> {
    let config = RankFusionConfig::new()
        .with_method(method)
        .with_normalization(ScoreNormalization::None);
    RankFusion::new(config).fuse_ids(lists).expect("fuse ok")
}

// ── Config defaults & builders ───────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = RankFusionConfig::default();
    assert_eq!(config.method, FusionMethod::CombSum);
    assert_eq!(config.normalization, ScoreNormalization::MinMax);
    assert!(config.weights.is_empty());
    assert_eq!(config.rrf_k, 60.0);
    assert_eq!(config.top_n, 0);
}

#[test]
fn config_new_equals_default() {
    let a = RankFusionConfig::new();
    let b = RankFusionConfig::default();
    assert_eq!(a.method, b.method);
    assert_eq!(a.normalization, b.normalization);
    assert_eq!(a.rrf_k, b.rrf_k);
    assert_eq!(a.top_n, b.top_n);
}

#[test]
fn enum_defaults() {
    assert_eq!(FusionMethod::default(), FusionMethod::CombSum);
    assert_eq!(ScoreNormalization::default(), ScoreNormalization::MinMax);
}

#[test]
fn individual_builders() {
    assert_eq!(
        RankFusionConfig::new()
            .with_method(FusionMethod::Borda)
            .method,
        FusionMethod::Borda
    );
    assert_eq!(
        RankFusionConfig::new()
            .with_normalization(ScoreNormalization::ZScore)
            .normalization,
        ScoreNormalization::ZScore
    );
    assert_eq!(
        RankFusionConfig::new().with_weights(vec![0.2, 0.8]).weights,
        vec![0.2, 0.8]
    );
    assert_eq!(RankFusionConfig::new().with_rrf_k(10.0).rrf_k, 10.0);
    assert_eq!(RankFusionConfig::new().with_top_n(3).top_n, 3);
}

#[test]
fn builder_chaining() {
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::Rrf)
        .with_normalization(ScoreNormalization::SumTo1)
        .with_weights(vec![1.0])
        .with_rrf_k(42.0)
        .with_top_n(7);
    assert_eq!(config.method, FusionMethod::Rrf);
    assert_eq!(config.normalization, ScoreNormalization::SumTo1);
    assert_eq!(config.weights, vec![1.0]);
    assert_eq!(config.rrf_k, 42.0);
    assert_eq!(config.top_n, 7);
}

#[test]
fn rank_fusion_exposes_config() {
    let config = RankFusionConfig::new().with_top_n(5);
    let fusion = RankFusion::new(config);
    assert_eq!(fusion.config().top_n, 5);
}

// ── CombSum ──────────────────────────────────────────────────────────────────

#[test]
fn combsum_sums_scores_across_lists() {
    let lists = vec![
        vec![pair("a", 0.5), pair("b", 0.3)],
        vec![pair("a", 0.4), pair("c", 0.2)],
    ];
    let out = fuse_raw(FusionMethod::CombSum, &lists);
    assert_eq!(score_of(&out, "a"), 0.9);
    assert_eq!(score_of(&out, "b"), 0.3);
    assert_eq!(score_of(&out, "c"), 0.2);
}

#[test]
fn combsum_orders_by_descending_score() {
    let lists = vec![
        vec![pair("low", 0.1), pair("high", 0.9)],
        vec![pair("mid", 0.5)],
    ];
    let out = fuse_raw(FusionMethod::CombSum, &lists);
    assert_eq!(out[0].0.as_str(), "high");
    assert_eq!(out[1].0.as_str(), "mid");
    assert_eq!(out[2].0.as_str(), "low");
}

// ── CombMNZ ──────────────────────────────────────────────────────────────────

#[test]
fn combmnz_multiplies_by_hit_count() {
    // "a" in 2 lists: (0.5 + 0.4) * 2 = 1.8
    let lists = vec![vec![pair("a", 0.5), pair("b", 0.3)], vec![pair("a", 0.4)]];
    let out = fuse_raw(FusionMethod::CombMnz, &lists);
    assert_eq!(score_of(&out, "a"), (0.5_f32 + 0.4) * 2.0);
    // "b" in 1 list: 0.3 * 1
    assert_eq!(score_of(&out, "b"), 0.3);
}

#[test]
fn combmnz_greater_than_combsum_for_multi_list_doc() {
    let lists = vec![
        vec![pair("shared", 0.5)],
        vec![pair("shared", 0.5)],
        vec![pair("shared", 0.5)],
    ];
    let sum = fuse_raw(FusionMethod::CombSum, &lists);
    let mnz = fuse_raw(FusionMethod::CombMnz, &lists);
    assert!(score_of(&mnz, "shared") > score_of(&sum, "shared"));
    // CombSum = 1.5, CombMNZ = 1.5 * 3 = 4.5
    assert_eq!(score_of(&sum, "shared"), 1.5);
    assert_eq!(score_of(&mnz, "shared"), 4.5);
}

#[test]
fn combmnz_rewards_agreement_over_single_strong_hit() {
    // doc "agree": 0.4 in two lists -> CombMNZ = 0.8 * 2 = 1.6
    // doc "solo": 0.9 in one list   -> CombMNZ = 0.9 * 1 = 0.9
    let lists = vec![
        vec![pair("agree", 0.4), pair("solo", 0.9)],
        vec![pair("agree", 0.4)],
    ];
    let out = fuse_raw(FusionMethod::CombMnz, &lists);
    assert!(score_of(&out, "agree") > score_of(&out, "solo"));
    assert_eq!(out[0].0.as_str(), "agree");
}

// ── CombANZ ──────────────────────────────────────────────────────────────────

#[test]
fn combanz_divides_by_hit_count() {
    // "a": (0.6 + 0.4) / 2 = 0.5
    let lists = vec![vec![pair("a", 0.6)], vec![pair("a", 0.4)]];
    let out = fuse_raw(FusionMethod::CombAnz, &lists);
    assert_eq!(score_of(&out, "a"), 0.5);
}

#[test]
fn combanz_is_mean_of_three() {
    // "a": (0.3 + 0.6 + 0.9) / 3 = 0.6
    let lists = vec![
        vec![pair("a", 0.3)],
        vec![pair("a", 0.6)],
        vec![pair("a", 0.9)],
    ];
    let out = fuse_raw(FusionMethod::CombAnz, &lists);
    assert!((score_of(&out, "a") - 0.6).abs() < 1e-6);
}

// ── Borda ────────────────────────────────────────────────────────────────────

#[test]
fn borda_points_on_tiny_example() {
    // List of length 3: rank0 -> 3, rank1 -> 2, rank2 -> 1
    let lists = vec![vec![pair("a", 0.0), pair("b", 0.0), pair("c", 0.0)]];
    let out = fuse_raw(FusionMethod::Borda, &lists);
    assert_eq!(score_of(&out, "a"), 3.0);
    assert_eq!(score_of(&out, "b"), 2.0);
    assert_eq!(score_of(&out, "c"), 1.0);
}

#[test]
fn borda_sums_across_lists() {
    // a: list0 rank0 (len2 -> 2) + list1 rank1 (len2 -> 1) = 3
    // b: list0 rank1 (len2 -> 1) + list1 rank0 (len2 -> 2) = 3
    let lists = vec![
        vec![pair("a", 0.0), pair("b", 0.0)],
        vec![pair("b", 0.0), pair("a", 0.0)],
    ];
    let out = fuse_raw(FusionMethod::Borda, &lists);
    assert_eq!(score_of(&out, "a"), 3.0);
    assert_eq!(score_of(&out, "b"), 3.0);
}

#[test]
fn borda_ignores_raw_scores() {
    // Scores wildly different but Borda only uses rank position.
    let lists = vec![vec![pair("top", 0.001), pair("bottom", 999.0)]];
    let out = fuse_raw(FusionMethod::Borda, &lists);
    assert_eq!(score_of(&out, "top"), 2.0);
    assert_eq!(score_of(&out, "bottom"), 1.0);
}

// ── ISR ──────────────────────────────────────────────────────────────────────

#[test]
fn isr_inverse_square_rank() {
    // rank0 -> 1/1^2 = 1, rank1 -> 1/2^2 = 0.25, rank2 -> 1/3^2 ≈ 0.1111
    let lists = vec![vec![pair("a", 0.0), pair("b", 0.0), pair("c", 0.0)]];
    let out = fuse_raw(FusionMethod::Isr, &lists);
    assert!((score_of(&out, "a") - 1.0).abs() < 1e-6);
    assert!((score_of(&out, "b") - 0.25).abs() < 1e-6);
    assert!((score_of(&out, "c") - 1.0 / 9.0).abs() < 1e-6);
    // Best rank first.
    assert_eq!(out[0].0.as_str(), "a");
}

#[test]
fn isr_sums_across_lists() {
    // "a": rank0 (1.0) + rank1 (0.25) = 1.25
    let lists = vec![
        vec![pair("a", 0.0), pair("x", 0.0)],
        vec![pair("y", 0.0), pair("a", 0.0)],
    ];
    let out = fuse_raw(FusionMethod::Isr, &lists);
    assert!((score_of(&out, "a") - 1.25).abs() < 1e-6);
}

// ── WeightedSum ──────────────────────────────────────────────────────────────

#[test]
fn weighted_sum_honors_weights() {
    // No normalization. a: 0.5*1.0 + 0.5*3.0 = 2.0
    let lists = vec![vec![pair("a", 0.5)], vec![pair("a", 0.5)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::WeightedSum)
        .with_normalization(ScoreNormalization::None)
        .with_weights(vec![1.0, 3.0]);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("fuse ok");
    assert_eq!(score_of(&out, "a"), 2.0);
}

#[test]
fn weighted_sum_zero_weight_excludes_list() {
    // List 1 weighted 0 -> contributes nothing.
    let lists = vec![vec![pair("a", 0.9)], vec![pair("a", 0.9)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::WeightedSum)
        .with_normalization(ScoreNormalization::None)
        .with_weights(vec![1.0, 0.0]);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("fuse ok");
    assert_eq!(score_of(&out, "a"), 0.9);
}

#[test]
fn weighted_sum_relative_ordering() {
    // Boost list 0 heavily so its top item wins.
    let lists = vec![vec![pair("from0", 1.0)], vec![pair("from1", 1.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::WeightedSum)
        .with_normalization(ScoreNormalization::None)
        .with_weights(vec![5.0, 1.0]);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("fuse ok");
    assert_eq!(out[0].0.as_str(), "from0");
    assert_eq!(score_of(&out, "from0"), 5.0);
    assert_eq!(score_of(&out, "from1"), 1.0);
}

#[test]
fn weighted_sum_missing_weights_errors() {
    let lists = vec![vec![pair("a", 0.5)], vec![pair("b", 0.5)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::WeightedSum)
        .with_weights(vec![1.0]); // only 1 weight for 2 lists
    let err = RankFusion::new(config).fuse_ids(&lists).unwrap_err();
    match err {
        RankFusionError::WeightMismatch { weights, lists } => {
            assert_eq!(weights, 1);
            assert_eq!(lists, 2);
        }
        other => panic!("expected WeightMismatch, got {other:?}"),
    }
}

// ── RRF ──────────────────────────────────────────────────────────────────────

#[test]
fn rrf_matches_formula() {
    // k = 60, rank0 -> 1/(60+1), rank1 -> 1/(60+2)
    let lists = vec![vec![pair("a", 0.0), pair("b", 0.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::Rrf)
        .with_rrf_k(60.0);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("fuse ok");
    assert!((score_of(&out, "a") - 1.0 / 61.0).abs() < 1e-6);
    assert!((score_of(&out, "b") - 1.0 / 62.0).abs() < 1e-6);
}

#[test]
fn rrf_sums_across_lists() {
    // "a": list0 rank0 (1/61) + list1 rank0 (1/61) = 2/61
    let lists = vec![vec![pair("a", 0.0)], vec![pair("a", 0.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::Rrf)
        .with_rrf_k(60.0);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("fuse ok");
    assert!((score_of(&out, "a") - 2.0 / 61.0).abs() < 1e-6);
}

#[test]
fn rrf_ignores_normalization() {
    // Same result regardless of normalization since RRF is rank-based.
    let lists = vec![vec![pair("a", 0.123), pair("b", 99.0)]];
    let none = {
        let c = RankFusionConfig::new()
            .with_method(FusionMethod::Rrf)
            .with_normalization(ScoreNormalization::None);
        RankFusion::new(c).fuse_ids(&lists).expect("ok")
    };
    let minmax = {
        let c = RankFusionConfig::new()
            .with_method(FusionMethod::Rrf)
            .with_normalization(ScoreNormalization::MinMax);
        RankFusion::new(c).fuse_ids(&lists).expect("ok")
    };
    assert_eq!(score_of(&none, "a"), score_of(&minmax, "a"));
    assert_eq!(score_of(&none, "b"), score_of(&minmax, "b"));
}

// ── Normalization ────────────────────────────────────────────────────────────

#[test]
fn minmax_maps_to_unit_interval() {
    // CombSum over a single min-max-normalized list.
    let lists = vec![vec![pair("hi", 10.0), pair("mid", 5.0), pair("lo", 0.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::MinMax);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    assert_eq!(score_of(&out, "hi"), 1.0);
    assert_eq!(score_of(&out, "lo"), 0.0);
    assert_eq!(score_of(&out, "mid"), 0.5);
    for (_, s) in &out {
        assert!((0.0..=1.0).contains(s));
    }
}

#[test]
fn minmax_constant_list_maps_to_one() {
    let lists = vec![vec![pair("a", 4.0), pair("b", 4.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::MinMax);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    assert_eq!(score_of(&out, "a"), 1.0);
    assert_eq!(score_of(&out, "b"), 1.0);
}

#[test]
fn sum_to_one_sums_to_one() {
    let lists = vec![vec![pair("a", 2.0), pair("b", 3.0), pair("c", 5.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::SumTo1);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    let total: f32 = out.iter().map(|(_, s)| *s).sum();
    assert!((total - 1.0).abs() < 1e-6);
    assert!((score_of(&out, "c") - 0.5).abs() < 1e-6);
}

#[test]
fn sum_to_one_zero_sum_is_identity() {
    let lists = vec![vec![pair("a", 0.0), pair("b", 0.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::SumTo1);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    assert_eq!(score_of(&out, "a"), 0.0);
    assert_eq!(score_of(&out, "b"), 0.0);
}

#[test]
fn zscore_mean_is_approximately_zero() {
    let lists = vec![vec![
        pair("a", 1.0),
        pair("b", 2.0),
        pair("c", 3.0),
        pair("d", 4.0),
    ]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::ZScore);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    let mean: f32 = out.iter().map(|(_, s)| *s).sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-6, "z-score mean should be ~0, got {mean}");
}

#[test]
fn zscore_constant_list_maps_to_zero() {
    let lists = vec![vec![pair("a", 7.0), pair("b", 7.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::ZScore);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    assert_eq!(score_of(&out, "a"), 0.0);
    assert_eq!(score_of(&out, "b"), 0.0);
}

#[test]
fn zscore_symmetric_values_have_opposite_signs() {
    let lists = vec![vec![pair("low", 0.0), pair("high", 10.0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::ZScore);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    assert!((score_of(&out, "low") + score_of(&out, "high")).abs() < 1e-6);
    assert!(score_of(&out, "high") > 0.0);
    assert!(score_of(&out, "low") < 0.0);
}

#[test]
fn none_normalization_keeps_scores() {
    let lists = vec![vec![pair("a", 0.37), pair("b", 0.91)]];
    let out = fuse_raw(FusionMethod::CombSum, &lists);
    assert_eq!(score_of(&out, "a"), 0.37);
    assert_eq!(score_of(&out, "b"), 0.91);
}

// ── Dedupe ───────────────────────────────────────────────────────────────────

#[test]
fn dedupe_merges_same_id_once() {
    let lists = vec![
        vec![pair("dup", 0.5), pair("other", 0.2)],
        vec![pair("dup", 0.5)],
    ];
    let out = fuse_raw(FusionMethod::CombSum, &lists);
    let count = out.iter().filter(|(d, _)| d.as_str() == "dup").count();
    assert_eq!(count, 1);
    assert_eq!(out.len(), 2);
    assert_eq!(score_of(&out, "dup"), 1.0);
}

#[test]
fn dedupe_within_single_list_accumulates() {
    // Same id twice within one list is accumulated (both occurrences counted).
    let lists = vec![vec![pair("a", 0.3), pair("a", 0.4)]];
    let out = fuse_raw(FusionMethod::CombSum, &lists);
    assert_eq!(out.len(), 1);
    assert!((score_of(&out, "a") - 0.7).abs() < 1e-6);
}

// ── fuse() over SearchResult ─────────────────────────────────────────────────

#[test]
fn fuse_search_results_dedupes_and_reranks() {
    let lists = vec![
        vec![result("a", 0.5, 0), result("b", 0.3, 1)],
        vec![result("a", 0.4, 0), result("c", 0.2, 1)],
    ];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None);
    let out = RankFusion::new(config).fuse(&lists).expect("fuse ok");

    assert_eq!(out.len(), 3);
    // a has highest fused score (0.9).
    assert_eq!(out[0].document.id.as_str(), "a");
    assert!((out[0].score - 0.9).abs() < 1e-6);
    // Ranks are reassigned from 0.
    assert_eq!(out[0].rank, 0);
    assert_eq!(out[1].rank, 1);
    assert_eq!(out[2].rank, 2);
}

#[test]
fn fuse_preserves_document_payload() {
    let lists = vec![vec![result("a", 0.5, 0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None);
    let out = RankFusion::new(config).fuse(&lists).expect("ok");
    assert_eq!(out[0].document.content, "content-a");
    assert_eq!(out[0].document.id.as_str(), "a");
}

#[test]
fn fuse_keys_on_document_id() {
    // Same id across two lists with different ranks -> merged once.
    let lists = vec![
        vec![result("shared", 0.6, 0)],
        vec![result("shared", 0.4, 0)],
    ];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None);
    let out = RankFusion::new(config).fuse(&lists).expect("ok");
    assert_eq!(out.len(), 1);
    assert!((out[0].score - 1.0).abs() < 1e-6);
}

#[test]
fn fuse_weighted_sum_mismatch_errors() {
    let lists = vec![vec![result("a", 0.5, 0)], vec![result("b", 0.5, 0)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::WeightedSum)
        .with_weights(vec![1.0]);
    let err = RankFusion::new(config).fuse(&lists).unwrap_err();
    assert!(matches!(err, RankFusionError::WeightMismatch { .. }));
}

// ── top_n truncation ─────────────────────────────────────────────────────────

#[test]
fn top_n_truncates_results() {
    let lists = vec![vec![
        pair("a", 0.9),
        pair("b", 0.7),
        pair("c", 0.5),
        pair("d", 0.3),
    ]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None)
        .with_top_n(2);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].0.as_str(), "a");
    assert_eq!(out[1].0.as_str(), "b");
}

#[test]
fn top_n_zero_returns_all() {
    let lists = vec![vec![pair("a", 0.9), pair("b", 0.7), pair("c", 0.5)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None)
        .with_top_n(0);
    let out = RankFusion::new(config).fuse_ids(&lists).expect("ok");
    assert_eq!(out.len(), 3);
}

#[test]
fn top_n_truncates_search_results() {
    let lists = vec![vec![
        result("a", 0.9, 0),
        result("b", 0.7, 1),
        result("c", 0.5, 2),
    ]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None)
        .with_top_n(1);
    let out = RankFusion::new(config).fuse(&lists).expect("ok");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].document.id.as_str(), "a");
}

// ── Single-list passthrough ──────────────────────────────────────────────────

#[test]
fn single_list_combsum_preserves_order() {
    let lists = vec![vec![pair("a", 0.9), pair("b", 0.5), pair("c", 0.1)]];
    let out = fuse_raw(FusionMethod::CombSum, &lists);
    assert_eq!(out[0].0.as_str(), "a");
    assert_eq!(out[1].0.as_str(), "b");
    assert_eq!(out[2].0.as_str(), "c");
}

#[test]
fn single_list_search_results_passthrough() {
    let lists = vec![vec![
        result("a", 0.9, 0),
        result("b", 0.5, 1),
        result("c", 0.1, 2),
    ]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None);
    let out = RankFusion::new(config).fuse(&lists).expect("ok");
    assert_eq!(out[0].document.id.as_str(), "a");
    assert_eq!(out[1].document.id.as_str(), "b");
    assert_eq!(out[2].document.id.as_str(), "c");
    assert_eq!(out[0].rank, 0);
    assert_eq!(out[2].rank, 2);
}

// ── Errors ───────────────────────────────────────────────────────────────────

#[test]
fn no_lists_error_fuse_ids() {
    let lists: Vec<Vec<(DocumentId, f32)>> = Vec::new();
    let fusion = RankFusion::new(RankFusionConfig::default());
    let err = fusion.fuse_ids(&lists).unwrap_err();
    assert!(matches!(err, RankFusionError::NoLists));
}

#[test]
fn no_lists_error_fuse() {
    let lists: Vec<Vec<SearchResult>> = Vec::new();
    let fusion = RankFusion::new(RankFusionConfig::default());
    let err = fusion.fuse(&lists).unwrap_err();
    assert!(matches!(err, RankFusionError::NoLists));
}

#[test]
fn no_lists_error_takes_precedence_over_weight_check() {
    // Even with WeightedSum and weights, an empty list set is NoLists.
    let lists: Vec<Vec<(DocumentId, f32)>> = Vec::new();
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::WeightedSum)
        .with_weights(vec![1.0, 2.0]);
    let err = RankFusion::new(config).fuse_ids(&lists).unwrap_err();
    assert!(matches!(err, RankFusionError::NoLists));
}

#[test]
fn weight_mismatch_error_message() {
    let err = RankFusionError::WeightMismatch {
        weights: 2,
        lists: 3,
    };
    assert_eq!(err.to_string(), "weights length 2 != list count 3");
}

#[test]
fn no_lists_error_message() {
    let err = RankFusionError::NoLists;
    assert_eq!(err.to_string(), "no input lists");
}

#[test]
fn non_weighted_methods_ignore_missing_weights() {
    // CombSum does not require weights even when lists.len() != weights.len().
    let lists = vec![vec![pair("a", 0.5)], vec![pair("b", 0.5)]];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_weights(vec![1.0]); // mismatch, but ignored
    assert!(RankFusion::new(config).fuse_ids(&lists).is_ok());
}

// ── Tie-break & determinism ──────────────────────────────────────────────────

#[test]
fn ties_broken_by_document_id_string() {
    // Equal scores -> ascending id order ("a" before "b" before "c").
    let lists = vec![vec![pair("c", 0.5), pair("a", 0.5), pair("b", 0.5)]];
    let out = fuse_raw(FusionMethod::CombSum, &lists);
    assert_eq!(out[0].0.as_str(), "a");
    assert_eq!(out[1].0.as_str(), "b");
    assert_eq!(out[2].0.as_str(), "c");
}

#[test]
fn determinism_repeated_runs_identical() {
    let lists = vec![
        vec![pair("a", 0.5), pair("b", 0.5), pair("c", 0.5)],
        vec![pair("c", 0.5)],
    ];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None);
    let fusion = RankFusion::new(config);
    let first = fusion.fuse_ids(&lists).expect("ok");
    for _ in 0..20 {
        let again = fusion.fuse_ids(&lists).expect("ok");
        assert_eq!(first.len(), again.len());
        for (a, b) in first.iter().zip(again.iter()) {
            assert_eq!(a.0.as_str(), b.0.as_str());
            assert_eq!(a.1, b.1);
        }
    }
}

#[test]
fn determinism_input_order_independent_for_combsum() {
    // CombSum is commutative; reordering lists yields the same fused scores.
    let a = vec![
        vec![pair("x", 0.4), pair("y", 0.6)],
        vec![pair("y", 0.1), pair("z", 0.9)],
    ];
    let b = vec![
        vec![pair("z", 0.9), pair("y", 0.1)],
        vec![pair("y", 0.6), pair("x", 0.4)],
    ];
    let out_a = fuse_raw(FusionMethod::CombSum, &a);
    let out_b = fuse_raw(FusionMethod::CombSum, &b);
    assert_eq!(score_of(&out_a, "x"), score_of(&out_b, "x"));
    assert_eq!(score_of(&out_a, "y"), score_of(&out_b, "y"));
    assert_eq!(score_of(&out_a, "z"), score_of(&out_b, "z"));
}

#[test]
fn fuse_and_fuse_ids_agree_on_scores() {
    // The two entry points must compute identical fused scores.
    let id_lists = vec![
        vec![pair("a", 0.5), pair("b", 0.3)],
        vec![pair("a", 0.4), pair("c", 0.2)],
    ];
    let sr_lists = vec![
        vec![result("a", 0.5, 0), result("b", 0.3, 1)],
        vec![result("a", 0.4, 0), result("c", 0.2, 1)],
    ];
    let config = RankFusionConfig::new()
        .with_method(FusionMethod::CombSum)
        .with_normalization(ScoreNormalization::None);
    let fusion = RankFusion::new(config);
    let ids = fusion.fuse_ids(&id_lists).expect("ok");
    let srs = fusion.fuse(&sr_lists).expect("ok");
    assert_eq!(ids.len(), srs.len());
    for (pair_out, sr_out) in ids.iter().zip(srs.iter()) {
        assert_eq!(pair_out.0.as_str(), sr_out.document.id.as_str());
        assert!((pair_out.1 - sr_out.score).abs() < 1e-6);
    }
}

#[test]
fn empty_inner_lists_only_yields_empty_output() {
    let lists = vec![
        Vec::<(DocumentId, f32)>::new(),
        Vec::<(DocumentId, f32)>::new(),
    ];
    let out = fuse_raw(FusionMethod::CombSum, &lists);
    assert!(out.is_empty());
}
