#![allow(clippy::float_cmp)]
//! Tests for the `cross_encoder` module.

use crate::cross_encoder::reranker::CrossEncoderReranker;
use crate::cross_encoder::scorer::LexicalCrossEncoder;
use crate::cross_encoder::types::{
    CrossEncoderConfig, CrossEncoderError, CrossEncoderScorer, FeatureWeights, InteractionFeatures,
    RerankedResult,
};
use crate::types::{Document, DocumentId, SearchResult};
use std::collections::HashMap;

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

// ── InteractionFeatures ───────────────────────────────────────────────────

#[test]
fn test_interaction_features_default_exact_match_ratio() {
    let f = InteractionFeatures::default();
    assert_eq!(
        f.exact_match_ratio, 0.0,
        "default exact_match_ratio should be 0.0"
    );
}

#[test]
fn test_interaction_features_default_term_overlap() {
    let f = InteractionFeatures::default();
    assert_eq!(f.term_overlap, 0.0, "default term_overlap should be 0.0");
}

#[test]
fn test_interaction_features_default_idf_weighted_overlap() {
    let f = InteractionFeatures::default();
    assert_eq!(
        f.idf_weighted_overlap, 0.0,
        "default idf_weighted_overlap should be 0.0"
    );
}

#[test]
fn test_interaction_features_default_query_coverage() {
    let f = InteractionFeatures::default();
    assert_eq!(
        f.query_coverage, 0.0,
        "default query_coverage should be 0.0"
    );
}

#[test]
fn test_interaction_features_default_doc_coverage() {
    let f = InteractionFeatures::default();
    assert_eq!(f.doc_coverage, 0.0, "default doc_coverage should be 0.0");
}

#[test]
fn test_interaction_features_default_bigram_match() {
    let f = InteractionFeatures::default();
    assert_eq!(
        f.ordered_bigram_match, 0.0,
        "default ordered_bigram_match should be 0.0"
    );
}

#[test]
fn test_interaction_features_default_length_ratio() {
    let f = InteractionFeatures::default();
    assert_eq!(f.length_ratio, 0.0, "default length_ratio should be 0.0");
}

#[test]
fn test_interaction_features_field_assignment() {
    let f = InteractionFeatures {
        exact_match_ratio: 0.5,
        term_overlap: 0.3,
        idf_weighted_overlap: 0.4,
        query_coverage: 0.6,
        doc_coverage: 0.2,
        ordered_bigram_match: 0.1,
        length_ratio: 0.8,
    };
    assert!(
        (f.exact_match_ratio - 0.5).abs() < 1e-6,
        "exact_match_ratio should be 0.5"
    );
    assert!(
        (f.length_ratio - 0.8).abs() < 1e-6,
        "length_ratio should be 0.8"
    );
}

// ── FeatureWeights ────────────────────────────────────────────────────────

#[test]
fn test_feature_weights_default_exact_match() {
    let w = FeatureWeights::default();
    assert!(
        (w.exact_match - 0.25).abs() < 1e-6,
        "default exact_match should be 0.25"
    );
}

#[test]
fn test_feature_weights_default_sum_approx_one() {
    let w = FeatureWeights::default();
    let sum = w.exact_match
        + w.term_overlap
        + w.idf_weighted
        + w.query_coverage
        + w.doc_coverage
        + w.bigram_match
        + w.length_ratio;
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "default weights should sum to 1.0, got {sum}"
    );
}

#[test]
fn test_feature_weights_normalize_sums_to_one() {
    let w = FeatureWeights {
        exact_match: 3.0,
        term_overlap: 2.0,
        idf_weighted: 1.0,
        query_coverage: 1.0,
        doc_coverage: 1.0,
        bigram_match: 1.0,
        length_ratio: 1.0,
    };
    let n = w.normalize();
    let sum = n.exact_match
        + n.term_overlap
        + n.idf_weighted
        + n.query_coverage
        + n.doc_coverage
        + n.bigram_match
        + n.length_ratio;
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "normalised weights should sum to 1.0, got {sum}"
    );
}

#[test]
fn test_feature_weights_normalize_zero_weights_unchanged() {
    // All weights zero → normalize leaves them at zero (no divide by zero)
    let w = FeatureWeights {
        exact_match: 0.0,
        term_overlap: 0.0,
        idf_weighted: 0.0,
        query_coverage: 0.0,
        doc_coverage: 0.0,
        bigram_match: 0.0,
        length_ratio: 0.0,
    };
    let n = w.normalize();
    let sum = n.exact_match + n.term_overlap + n.idf_weighted;
    assert_eq!(
        sum, 0.0,
        "zero-weight normalize should leave all weights at 0.0"
    );
}

#[test]
fn test_feature_weights_with_exact_match_builder() {
    let w = FeatureWeights::default().with_exact_match(0.9);
    assert!(
        (w.exact_match - 0.9).abs() < 1e-6,
        "with_exact_match should set exact_match to 0.9"
    );
}

#[test]
fn test_feature_weights_with_idf_weighted_builder() {
    let w = FeatureWeights::default().with_idf_weighted(0.7);
    assert!(
        (w.idf_weighted - 0.7).abs() < 1e-6,
        "with_idf_weighted should set idf_weighted to 0.7"
    );
}

// ── CrossEncoderConfig ────────────────────────────────────────────────────

#[test]
fn test_cross_encoder_config_default_top_n() {
    let c = CrossEncoderConfig::default();
    assert_eq!(c.top_n, 0, "default top_n should be 0 (return all)");
}

#[test]
fn test_cross_encoder_config_default_blend_alpha() {
    let c = CrossEncoderConfig::default();
    assert!(
        (c.blend_alpha - 0.4).abs() < 1e-6,
        "default blend_alpha should be 0.4"
    );
}

#[test]
fn test_cross_encoder_config_default_score_threshold() {
    let c = CrossEncoderConfig::default();
    assert_eq!(
        c.score_threshold, 0.0,
        "default score_threshold should be 0.0"
    );
}

#[test]
fn test_cross_encoder_config_with_top_n_builder() {
    let c = CrossEncoderConfig::default().with_top_n(5);
    assert_eq!(c.top_n, 5, "with_top_n should set top_n to 5");
}

#[test]
fn test_cross_encoder_config_with_blend_alpha_builder() {
    let c = CrossEncoderConfig::default().with_blend_alpha(0.8);
    assert!(
        (c.blend_alpha - 0.8).abs() < 1e-6,
        "with_blend_alpha should set blend_alpha to 0.8"
    );
}

#[test]
fn test_cross_encoder_config_with_score_threshold_builder() {
    let c = CrossEncoderConfig::default().with_score_threshold(0.3);
    assert!(
        (c.score_threshold - 0.3).abs() < 1e-6,
        "with_score_threshold should set threshold to 0.3"
    );
}

#[test]
fn test_cross_encoder_config_with_weights_builder() {
    let w = FeatureWeights::default().with_exact_match(0.5);
    let c = CrossEncoderConfig::default().with_weights(w.clone());
    assert!(
        (c.weights.exact_match - 0.5).abs() < 1e-6,
        "with_weights should set custom weights"
    );
}

// ── RerankedResult ────────────────────────────────────────────────────────

#[test]
fn test_reranked_result_field_access() {
    let doc = Document::new("test doc").with_id(DocumentId::from_string("r1"));
    let r = RerankedResult {
        document: doc.clone(),
        original_score: 0.8,
        cross_score: 0.6,
        fused_score: 0.72,
        original_rank: 2,
        new_rank: 0,
    };
    assert_eq!(r.document.id.as_str(), "r1", "document id should be r1");
    assert!(
        (r.original_score - 0.8).abs() < 1e-6,
        "original_score should be 0.8"
    );
    assert!(
        (r.cross_score - 0.6).abs() < 1e-6,
        "cross_score should be 0.6"
    );
    assert!(
        (r.fused_score - 0.72).abs() < 1e-6,
        "fused_score should be 0.72"
    );
    assert_eq!(r.original_rank, 2, "original_rank should be 2");
    assert_eq!(r.new_rank, 0, "new_rank should be 0");
}

// ── LexicalCrossEncoder ───────────────────────────────────────────────────

#[test]
fn test_lexical_cross_encoder_new_stores_weights() {
    let w = FeatureWeights::default().with_exact_match(0.5);
    let enc = LexicalCrossEncoder::new(w.clone());
    assert!(
        (enc.weights.exact_match - 0.5).abs() < 1e-6,
        "new should store provided weights"
    );
}

#[test]
fn test_extract_features_identical_query_doc_exact_match_one() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let idf: HashMap<String, f32> = HashMap::new();
    let features = enc.extract_features("rust programming", "rust programming", &idf);
    assert!(
        (features.exact_match_ratio - 1.0).abs() < 1e-5,
        "identical query and doc should yield exact_match_ratio=1.0, got {}",
        features.exact_match_ratio
    );
}

#[test]
fn test_extract_features_disjoint_exact_match_zero() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let idf: HashMap<String, f32> = HashMap::new();
    let features = enc.extract_features("alpha beta", "gamma delta", &idf);
    assert_eq!(
        features.exact_match_ratio, 0.0,
        "disjoint query and doc should yield exact_match_ratio=0.0"
    );
}

#[test]
fn test_extract_features_empty_query_returns_default() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let idf: HashMap<String, f32> = HashMap::new();
    let features = enc.extract_features("", "some content", &idf);
    assert_eq!(
        features.exact_match_ratio, 0.0,
        "empty query should yield default features"
    );
}

#[test]
fn test_extract_features_empty_doc_returns_default() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let idf: HashMap<String, f32> = HashMap::new();
    let features = enc.extract_features("some query", "", &idf);
    assert_eq!(
        features.exact_match_ratio, 0.0,
        "empty doc should yield default features"
    );
}

#[test]
fn test_extract_features_length_ratio_equal_lengths() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let idf: HashMap<String, f32> = HashMap::new();
    // "rust lang" vs "fast code" — 2 tokens each, no overlap
    let features = enc.extract_features("rust lang", "fast code", &idf);
    assert!(
        (features.length_ratio - 1.0).abs() < 1e-5,
        "equal-length query and doc should yield length_ratio=1.0, got {}",
        features.length_ratio
    );
}

#[test]
fn test_extract_features_bigram_match_identical() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let idf: HashMap<String, f32> = HashMap::new();
    let features = enc.extract_features("rust is fast", "rust is fast", &idf);
    assert!(
        (features.ordered_bigram_match - 1.0).abs() < 1e-5,
        "identical bigrams should yield ordered_bigram_match=1.0, got {}",
        features.ordered_bigram_match
    );
}

#[test]
fn test_extract_features_term_overlap_in_zero_one() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let idf: HashMap<String, f32> = HashMap::new();
    let features = enc.extract_features("machine learning rust", "rust programming", &idf);
    assert!(
        (0.0..=1.0).contains(&features.term_overlap),
        "term_overlap must be in [0,1], got {}",
        features.term_overlap
    );
}

// ── score_all error cases ─────────────────────────────────────────────────

#[test]
fn test_score_all_empty_query_returns_error() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let results = vec![make_result("d1", "content", 0.8)];
    let err = enc.score_all("", &results).unwrap_err();
    assert!(
        matches!(err, CrossEncoderError::EmptyQuery),
        "empty query should return EmptyQuery error"
    );
}

#[test]
fn test_score_all_whitespace_query_returns_error() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let results = vec![make_result("d1", "content", 0.8)];
    let err = enc.score_all("   ", &results).unwrap_err();
    assert!(
        matches!(err, CrossEncoderError::EmptyQuery),
        "whitespace-only query should return EmptyQuery error"
    );
}

#[test]
fn test_score_all_empty_results_returns_error() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let err = enc.score_all("query", &[]).unwrap_err();
    assert!(
        matches!(err, CrossEncoderError::EmptyCandidates),
        "empty results should return EmptyCandidates error"
    );
}

#[test]
fn test_score_all_happy_path_length() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let results = vec![
        make_result("d1", "rust programming language", 0.9),
        make_result("d2", "python data science", 0.7),
    ];
    let scored = enc.score_all("rust language", &results).unwrap();
    assert_eq!(
        scored.len(),
        2,
        "score_all should return same count as input"
    );
}

#[test]
fn test_score_all_cross_scores_in_zero_one() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let results = vec![
        make_result("d1", "rust programming language", 0.9),
        make_result("d2", "python data science", 0.6),
        make_result("d3", "machine learning algorithms", 0.5),
    ];
    let scored = enc.score_all("rust language", &results).unwrap();
    for (_, _, cs) in &scored {
        assert!(
            (0.0..=1.0).contains(cs),
            "cross score must be in [0,1], got {cs}"
        );
    }
}

#[test]
fn test_score_all_relevant_doc_ranked_higher() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let results = vec![
        make_result("relevant", "rust programming language systems", 0.5),
        make_result("irrelevant", "cooking recipe pasta dinner", 0.9),
    ];
    let scored = enc.score_all("rust programming", &results).unwrap();
    // Find cross scores by original index
    let cs_relevant = scored.iter().find(|(i, _, _)| *i == 0).unwrap().2;
    let cs_irrelevant = scored.iter().find(|(i, _, _)| *i == 1).unwrap().2;
    assert!(
        cs_relevant > cs_irrelevant,
        "relevant doc should have higher cross score: {cs_relevant} vs {cs_irrelevant}"
    );
}

// ── CrossEncoderScorer trait ──────────────────────────────────────────────

#[test]
fn test_cross_encoder_scorer_score_method_returns_float() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let doc = Document::new("rust programming systems language");
    let s = enc.score("rust language", &doc);
    assert!(
        (0.0..=1.0).contains(&s),
        "CrossEncoderScorer::score must return value in [0,1], got {s}"
    );
}

#[test]
fn test_cross_encoder_scorer_score_identical_query_doc() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let doc = Document::new("rust programming systems language");
    let s = enc.score("rust programming systems language", &doc);
    // Identical query and doc should yield a high score (> 0.5)
    assert!(
        s > 0.5,
        "identical query/doc should yield score > 0.5, got {s}"
    );
}

#[test]
fn test_cross_encoder_scorer_score_empty_doc() {
    let enc = LexicalCrossEncoder::new(FeatureWeights::default());
    let doc = Document::new("");
    let s = enc.score("rust", &doc);
    // Should not panic; score can be any value in [0,1]
    assert!(
        (0.0..=1.0).contains(&s),
        "empty doc score should still be in [0,1]"
    );
}

// ── CrossEncoderReranker ──────────────────────────────────────────────────

#[test]
fn test_cross_encoder_reranker_new_stores_config() {
    let cfg = CrossEncoderConfig::default().with_top_n(3);
    let reranker = CrossEncoderReranker::new(cfg);
    assert_eq!(
        reranker.config.top_n, 3,
        "new should store the provided config"
    );
}

#[test]
fn test_cross_encoder_reranker_default_top_n_zero() {
    let reranker = CrossEncoderReranker::default();
    assert_eq!(reranker.config.top_n, 0, "default top_n should be 0");
}

#[test]
fn test_reranker_empty_query_returns_error() {
    let reranker = CrossEncoderReranker::default();
    let results = vec![make_result("d1", "content", 0.8)];
    let err = reranker.rerank("", &results).unwrap_err();
    assert!(
        matches!(err, CrossEncoderError::EmptyQuery),
        "empty query should return EmptyQuery"
    );
}

#[test]
fn test_reranker_empty_results_returns_error() {
    let reranker = CrossEncoderReranker::default();
    let err = reranker.rerank("query", &[]).unwrap_err();
    assert!(
        matches!(err, CrossEncoderError::EmptyCandidates),
        "empty results should return EmptyCandidates"
    );
}

#[test]
fn test_reranker_top_n_nonzero_limits_output() {
    let cfg = CrossEncoderConfig::default().with_top_n(2);
    let reranker = CrossEncoderReranker::new(cfg);
    let results = vec![
        make_result("d1", "rust systems programming", 0.9),
        make_result("d2", "python scripting", 0.7),
        make_result("d3", "java enterprise", 0.5),
        make_result("d4", "golang concurrency", 0.4),
    ];
    let result_list = reranker.rerank("rust language", &results).unwrap();
    assert_eq!(
        result_list.len(),
        2,
        "top_n=2 should return at most 2 results"
    );
}

#[test]
fn test_reranker_top_n_zero_returns_all() {
    let cfg = CrossEncoderConfig::default().with_top_n(0);
    let reranker = CrossEncoderReranker::new(cfg);
    let results = vec![
        make_result("d1", "rust systems programming", 0.9),
        make_result("d2", "python scripting", 0.7),
        make_result("d3", "java enterprise", 0.5),
    ];
    let result_list = reranker.rerank("rust language", &results).unwrap();
    assert_eq!(result_list.len(), 3, "top_n=0 should return all results");
}

#[test]
fn test_reranker_blend_alpha_one_preserves_original_relative_order() {
    // When blend_alpha=1.0 fused = 1.0 * orig + 0.0 * cross → original scores dominate
    let cfg = CrossEncoderConfig::default().with_blend_alpha(1.0);
    let reranker = CrossEncoderReranker::new(cfg);
    let results = vec![
        make_result("first", "alpha beta gamma", 0.9),
        make_result("second", "delta epsilon zeta", 0.1),
    ];
    let result_list = reranker.rerank("alpha beta", &results).unwrap();
    assert_eq!(
        result_list[0].document.id.as_str(),
        "first",
        "blend_alpha=1.0 should put original top doc first"
    );
}

#[test]
fn test_reranker_score_threshold_filters_low_scores() {
    // Set a very high threshold so all results are filtered out
    let cfg = CrossEncoderConfig::default().with_score_threshold(999.0);
    let reranker = CrossEncoderReranker::new(cfg);
    let results = vec![
        make_result("d1", "rust systems programming", 0.9),
        make_result("d2", "python scripting", 0.7),
    ];
    let result_list = reranker.rerank("rust language", &results).unwrap();
    assert!(
        result_list.is_empty(),
        "threshold=999.0 should filter all results, got {}",
        result_list.len()
    );
}

#[test]
fn test_reranker_new_rank_is_zero_indexed_consecutive() {
    let reranker = CrossEncoderReranker::default();
    let results = vec![
        make_result("d1", "rust programming", 0.9),
        make_result("d2", "python scripting", 0.7),
        make_result("d3", "java enterprise", 0.5),
    ];
    let result_list = reranker.rerank("rust language", &results).unwrap();
    for (i, r) in result_list.iter().enumerate() {
        assert_eq!(
            r.new_rank, i,
            "new_rank should be {i} for position {i}, got {}",
            r.new_rank
        );
    }
}

#[test]
fn test_reranker_fused_score_in_zero_one_approx() {
    // With default blend_alpha=0.4 and both orig scores in [0,1] and cross in [0,1],
    // fused should remain in [0,1].
    let reranker = CrossEncoderReranker::default();
    let results = vec![
        make_result("d1", "rust systems programming language", 0.8),
        make_result("d2", "python data science machine learning", 0.6),
    ];
    let result_list = reranker.rerank("rust programming", &results).unwrap();
    for r in &result_list {
        assert!(
            (-0.1..=1.1).contains(&r.fused_score),
            "fused_score should be approximately in [0,1], got {}",
            r.fused_score
        );
    }
}

#[test]
fn test_reranker_output_contains_all_input_ids_when_top_n_zero() {
    let reranker = CrossEncoderReranker::default();
    let results = vec![
        make_result("alpha", "rust systems programming", 0.9),
        make_result("beta", "python scripting language", 0.7),
        make_result("gamma", "java enterprise patterns", 0.5),
    ];
    let result_list = reranker.rerank("programming language", &results).unwrap();
    let ids: Vec<&str> = result_list.iter().map(|r| r.document.id.as_str()).collect();
    assert!(ids.contains(&"alpha"), "output should contain alpha");
    assert!(ids.contains(&"beta"), "output should contain beta");
    assert!(ids.contains(&"gamma"), "output should contain gamma");
}

// ── CrossEncoderError display ─────────────────────────────────────────────

#[test]
fn test_error_empty_query_display() {
    let e = CrossEncoderError::EmptyQuery;
    assert!(
        e.to_string().contains("empty"),
        "EmptyQuery display should mention 'empty', got: {e}"
    );
}

#[test]
fn test_error_empty_candidates_display() {
    let e = CrossEncoderError::EmptyCandidates;
    assert!(
        e.to_string().contains("empty") || e.to_string().contains("Candidate"),
        "EmptyCandidates display should be descriptive, got: {e}"
    );
}
