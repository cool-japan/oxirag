#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    // These hand-derived CORI formula replications coincidentally divide by
    // a literal `2.0` (e.g. the I-formula numerator `(num_shards + 0.5)`
    // with `num_shards == 2` in a fixture) without being a semantic
    // "midpoint of two points" -- `f64::midpoint` would misleadingly imply
    // that reading.
    clippy::manual_midpoint
)]
//! Tests for CORI-style pre-query shard selection.
//!
//! Sections mirror the module's own structure: `ShardTermStats` /
//! `ShardDescriptor` / `ShardSelectionScore` / `ShardSelectionConfig` /
//! `ShardSelectionError` data-type tests, then the CORI `T` component
//! (`term_frequency_belief`), the CORI `I` component
//! (`inverse_shard_frequency`), their combination (`term_belief`), the
//! per-shard/candidate-set helpers, `score_shard`, the full `select`
//! pipeline, and finally the `ShardSelector` façade. Hand-calculated
//! expectations are written out from the raw CORI formula directly (not by
//! calling the engine), so each assertion is an independent check against
//! the formula in the module doc comment.

use std::collections::HashMap;

use crate::shard_selection::engine::{ShardSelectionEngine, ShardSelector};
use crate::shard_selection::types::{
    ShardDescriptor, ShardSelectionConfig, ShardSelectionError, ShardSelectionScore, ShardTermStats,
};

// ── Test helpers ─────────────────────────────────────────────────────────────

fn engine_with_default_config() -> ShardSelectionEngine {
    ShardSelectionEngine::new(ShardSelectionConfig::default())
}

fn make_shard(id: &str, doc_count: u64, avg_doc_length: f64) -> ShardDescriptor {
    ShardDescriptor::new(id, doc_count, avg_doc_length)
}

/// Approximate equality for hand-calculated `f64` assertions.
fn approx(actual: f64, expected: f64, epsilon: f64) -> bool {
    (actual - expected).abs() < epsilon
}

// ── ShardTermStats ───────────────────────────────────────────────────────────

#[test]
fn shard_term_stats_new_is_empty() {
    let stats = ShardTermStats::new();
    assert!(stats.is_empty());
    assert_eq!(stats.len(), 0);
}

#[test]
fn shard_term_stats_with_term_records_values() {
    let stats = ShardTermStats::new().with_term("x", 3, 7);
    assert_eq!(stats.document_frequency("x"), 3);
    assert_eq!(stats.collection_term_frequency("x"), 7);
    assert!(stats.contains_term("x"));
    assert!(!stats.is_empty());
}

#[test]
fn shard_term_stats_missing_term_defaults_to_zero() {
    let stats = ShardTermStats::new();
    assert_eq!(stats.document_frequency("missing"), 0);
    assert_eq!(stats.collection_term_frequency("missing"), 0);
    assert!(!stats.contains_term("missing"));
}

#[test]
fn shard_term_stats_with_term_overwrites_previous_value() {
    let stats = ShardTermStats::new()
        .with_term("x", 1, 1)
        .with_term("x", 5, 9);
    assert_eq!(stats.document_frequency("x"), 5);
    assert_eq!(stats.collection_term_frequency("x"), 9);
    assert_eq!(stats.len(), 1);
}

#[test]
fn shard_term_stats_terms_iterator_lists_every_term() {
    let stats = ShardTermStats::new()
        .with_term("a", 1, 1)
        .with_term("b", 2, 2);
    let mut terms: Vec<&str> = stats.terms().collect();
    terms.sort_unstable();
    assert_eq!(terms, vec!["a", "b"]);
}

// ── ShardDescriptor ──────────────────────────────────────────────────────────

#[test]
fn shard_descriptor_new_defaults() {
    let descriptor = make_shard("s", 10, 20.0);
    assert_eq!(descriptor.shard_id, "s");
    assert_eq!(descriptor.doc_count, 10);
    assert_eq!(descriptor.avg_doc_length, 20.0);
    assert!(descriptor.term_stats.is_empty());
}

#[test]
fn shard_descriptor_collection_word_count() {
    let descriptor = make_shard("s", 10, 20.0);
    assert_eq!(descriptor.collection_word_count(), 200.0);
}

#[test]
fn shard_descriptor_zero_doc_count_zero_word_count() {
    let descriptor = make_shard("s", 0, 999.0);
    assert_eq!(descriptor.collection_word_count(), 0.0);
}

#[test]
fn shard_descriptor_with_term_chains_multiple_terms() {
    let descriptor = make_shard("s", 10, 20.0)
        .with_term("a", 1, 2)
        .with_term("b", 3, 4);
    assert_eq!(descriptor.term_stats.len(), 2);
    assert_eq!(descriptor.term_stats.document_frequency("a"), 1);
    assert_eq!(descriptor.term_stats.collection_term_frequency("b"), 4);
}

#[test]
fn shard_descriptor_with_term_stats_replaces_wholesale() {
    let stats = ShardTermStats::new().with_term("z", 9, 9);
    let descriptor = make_shard("s", 10, 20.0)
        .with_term("a", 1, 2)
        .with_term_stats(stats);
    assert_eq!(descriptor.term_stats.len(), 1);
    assert!(!descriptor.term_stats.contains_term("a"));
    assert!(descriptor.term_stats.contains_term("z"));
}

// ── ShardSelectionScore ──────────────────────────────────────────────────────

#[test]
fn shard_selection_score_new_constructs() {
    let score = ShardSelectionScore::new("shard-x", 0.75, 3);
    assert_eq!(score.shard_id, "shard-x");
    assert_eq!(score.score, 0.75);
    assert_eq!(score.matched_term_count, 3);
}

// ── ShardSelectionConfig ─────────────────────────────────────────────────────

#[test]
fn shard_selection_config_default_values() {
    let config = ShardSelectionConfig::default();
    assert_eq!(config.frequency_saturation_constant, 50.0);
    assert_eq!(config.length_normalization_constant, 150.0);
    assert_eq!(config.belief_floor, 0.4);
    assert_eq!(config.belief_scale, 0.6);
    assert_eq!(config.default_top_k, 10);
}

#[test]
fn shard_selection_config_new_equals_default() {
    assert_eq!(ShardSelectionConfig::new(), ShardSelectionConfig::default());
}

#[test]
fn shard_selection_config_builder_overrides_every_field() {
    let config = ShardSelectionConfig::new()
        .with_frequency_saturation_constant(10.0)
        .with_length_normalization_constant(20.0)
        .with_belief_floor(0.1)
        .with_belief_scale(0.9)
        .with_default_top_k(3);
    assert_eq!(config.frequency_saturation_constant, 10.0);
    assert_eq!(config.length_normalization_constant, 20.0);
    assert_eq!(config.belief_floor, 0.1);
    assert_eq!(config.belief_scale, 0.9);
    assert_eq!(config.default_top_k, 3);
}

// ── ShardSelectionError ──────────────────────────────────────────────────────

#[test]
fn error_display_empty_shard_set() {
    assert!(
        ShardSelectionError::EmptyShardSet
            .to_string()
            .contains("empty")
    );
}

#[test]
fn error_display_empty_query_terms() {
    assert!(
        ShardSelectionError::EmptyQueryTerms
            .to_string()
            .contains("query terms")
    );
}

#[test]
fn error_display_duplicate_shard_id() {
    let error = ShardSelectionError::DuplicateShardId {
        shard_id: "dup-1".to_string(),
    };
    assert!(error.to_string().contains("dup-1"));
}

#[test]
fn error_display_invalid_shard_descriptor() {
    let error = ShardSelectionError::InvalidShardDescriptor {
        shard_id: "bad-1".to_string(),
        reason: "avg_doc_length must be finite and non-negative, got -1".to_string(),
    };
    let message = error.to_string();
    assert!(message.contains("bad-1"));
    assert!(message.contains("avg_doc_length"));
}

// ── term_frequency_belief (CORI T component) ─────────────────────────────────

#[test]
fn term_frequency_belief_small_shard_beats_large_shard_same_df() {
    let engine = engine_with_default_config();
    // Two shards, equal df, wildly different size; avgcw is their mean.
    let t_small = 5.0 / (5.0 + 50.0 + 150.0 * (1000.0 / 5500.0));
    let t_large = 5.0 / (5.0 + 50.0 + 150.0 * (10_000.0 / 5500.0));

    let actual_small = engine.term_frequency_belief(5, 1000.0, 5500.0);
    let actual_large = engine.term_frequency_belief(5, 10_000.0, 5500.0);

    assert!(approx(actual_small, t_small, 1e-9));
    assert!(approx(actual_large, t_large, 1e-9));
    assert!(
        actual_small > actual_large,
        "a small shard should score higher than a large shard for equal df: \
         small={actual_small} large={actual_large}"
    );
}

#[test]
fn term_frequency_belief_zero_document_frequency_is_zero() {
    let engine = engine_with_default_config();
    assert_eq!(engine.term_frequency_belief(0, 1000.0, 5500.0), 0.0);
    assert_eq!(engine.term_frequency_belief(0, 0.0, 0.0), 0.0);
}

#[test]
fn term_frequency_belief_zero_average_collection_word_count_guarded() {
    let engine = engine_with_default_config();
    let actual = engine.term_frequency_belief(5, 1000.0, 0.0);
    // Degenerate avgcw=0 must not produce NaN/Inf: the size ratio is treated
    // as 0, so T = 5 / (5 + 50 + 150*0) = 5 / 55.
    assert!(actual.is_finite());
    assert!(approx(actual, 5.0 / 55.0, 1e-9));
}

#[test]
fn term_frequency_belief_matches_manual_formula_general_case() {
    let engine = engine_with_default_config();
    let expected = 20.0 / (20.0 + 50.0 + 150.0 * (3000.0 / 1500.0));
    assert!(approx(
        engine.term_frequency_belief(20, 3000.0, 1500.0),
        expected,
        1e-9
    ));
}

#[test]
fn term_frequency_belief_custom_constants_disable_length_penalty() {
    let config = ShardSelectionConfig::new()
        .with_frequency_saturation_constant(10.0)
        .with_length_normalization_constant(0.0);
    let engine = ShardSelectionEngine::new(config);
    // Length penalty disabled (multiplier 0): T = df / (df + 10) regardless
    // of shard size.
    let actual = engine.term_frequency_belief(10, 999_999.0, 1.0);
    assert!(approx(actual, 0.5, 1e-9));
}

#[test]
fn term_frequency_belief_no_penalties_configured_saturates_at_one() {
    let config = ShardSelectionConfig::new()
        .with_frequency_saturation_constant(0.0)
        .with_length_normalization_constant(0.0);
    let engine = ShardSelectionEngine::new(config);
    // Both constants zeroed: T = df / df = 1.0 for any df > 0.
    let actual = engine.term_frequency_belief(7, 12_345.0, 1.0);
    assert!(approx(actual, 1.0, 1e-9));
}

// ── inverse_shard_frequency (CORI I component) ───────────────────────────────

#[test]
fn inverse_shard_frequency_rare_term_beats_common_term() {
    let engine = engine_with_default_config();
    let num_shards = 5;
    let i_rare = engine.inverse_shard_frequency(1, num_shards);
    let i_common = engine.inverse_shard_frequency(5, num_shards);

    let expected_rare = ((5.0_f64 + 0.5) / 1.0).ln() / (5.0_f64 + 1.0).ln();
    let expected_common = ((5.0_f64 + 0.5) / 5.0).ln() / (5.0_f64 + 1.0).ln();

    assert!(approx(i_rare, expected_rare, 1e-9));
    assert!(approx(i_common, expected_common, 1e-9));
    assert!(
        i_rare > i_common,
        "a term present in 1/5 shards should discriminate more than one present in 5/5: \
         rare={i_rare} common={i_common}"
    );
}

#[test]
fn inverse_shard_frequency_zero_shard_frequency_is_zero() {
    let engine = engine_with_default_config();
    assert_eq!(engine.inverse_shard_frequency(0, 10), 0.0);
}

#[test]
fn inverse_shard_frequency_zero_num_candidate_shards_is_zero() {
    let engine = engine_with_default_config();
    assert_eq!(engine.inverse_shard_frequency(3, 0), 0.0);
}

#[test]
fn inverse_shard_frequency_matches_manual_formula() {
    let engine = engine_with_default_config();
    let expected = ((10.0_f64 + 0.5) / 3.0).ln() / (10.0_f64 + 1.0).ln();
    assert!(approx(
        engine.inverse_shard_frequency(3, 10),
        expected,
        1e-9
    ));
}

#[test]
fn inverse_shard_frequency_single_shard_single_candidate() {
    let engine = engine_with_default_config();
    let expected = ((1.0_f64 + 0.5) / 1.0).ln() / (1.0_f64 + 1.0).ln();
    assert!(approx(engine.inverse_shard_frequency(1, 1), expected, 1e-9));
}

#[test]
fn inverse_shard_frequency_strictly_decreasing_in_shard_frequency() {
    let engine = engine_with_default_config();
    let num_shards = 8;
    let values: Vec<f64> = (1..=num_shards)
        .map(|shard_frequency| engine.inverse_shard_frequency(shard_frequency, num_shards))
        .collect();
    for window in values.windows(2) {
        assert!(
            window[0] > window[1],
            "I should strictly decrease as shard_frequency grows: {values:?}"
        );
    }
}

// ── term_belief (T/I combination) ────────────────────────────────────────────

#[test]
fn term_belief_combines_floor_scale_t_i() {
    let engine = engine_with_default_config();
    let actual = engine.term_belief(0.5, 0.5);
    assert!(approx(actual, 0.4 + 0.6 * 0.5 * 0.5, 1e-12));
}

#[test]
fn term_belief_absent_term_floors_at_baseline_regardless_of_i() {
    let engine = engine_with_default_config();
    // T = 0 (the term is absent from this shard) must floor the belief at
    // belief_floor even when I is large -- this is this module's chosen,
    // deterministic rule for absent-term handling.
    assert_eq!(engine.term_belief(0.0, 0.95), 0.4);
    assert_eq!(engine.term_belief(0.0, 0.0), 0.4);
    assert_eq!(engine.term_belief(0.0, 1.0), 0.4);
}

#[test]
fn term_belief_custom_floor_and_scale() {
    let config = ShardSelectionConfig::new()
        .with_belief_floor(0.1)
        .with_belief_scale(1.0);
    let engine = ShardSelectionEngine::new(config);
    let actual = engine.term_belief(0.5, 0.4);
    assert!(approx(actual, 0.1 + 1.0 * 0.5 * 0.4, 1e-12));
}

// ── average_collection_word_count / shard_frequency ──────────────────────────

#[test]
fn average_collection_word_count_hand_calculated() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("a", 10, 10.0), // cw = 100
        make_shard("b", 20, 10.0), // cw = 200
        make_shard("c", 30, 10.0), // cw = 300
    ];
    assert_eq!(engine.average_collection_word_count(&shards), 200.0);
}

#[test]
fn average_collection_word_count_empty_shards_is_zero() {
    let engine = engine_with_default_config();
    assert_eq!(engine.average_collection_word_count(&[]), 0.0);
}

#[test]
fn shard_frequency_counts_shards_with_nonzero_document_frequency() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("a", 10, 10.0).with_term("x", 1, 1),
        make_shard("b", 10, 10.0).with_term("x", 0, 0),
        make_shard("c", 10, 10.0).with_term("x", 4, 8),
        make_shard("d", 10, 10.0),
    ];
    assert_eq!(engine.shard_frequency(&shards, "x"), 2);
}

#[test]
fn shard_frequency_term_absent_everywhere_is_zero() {
    let engine = engine_with_default_config();
    let shards = vec![make_shard("a", 10, 10.0), make_shard("b", 10, 10.0)];
    assert_eq!(engine.shard_frequency(&shards, "nonexistent"), 0);
}

// ── score_shard ───────────────────────────────────────────────────────────────

#[test]
fn score_shard_multi_term_query_matches_hand_computed_average() {
    let engine = engine_with_default_config();
    let shard = make_shard("s", 100, 50.0) // cw = 5000
        .with_term("rust", 20, 40)
        .with_term("python", 5, 5);
    // Candidate-set context supplied by the caller: 4 shards total, avgcw
    // 4000; "rust" appears in 2 of the 4, "python" in only 1.
    let shard_frequency_by_term: HashMap<&str, usize> = HashMap::from([("rust", 2), ("python", 1)]);
    let query_terms = vec!["rust".to_string(), "python".to_string()];

    let score = engine.score_shard(&shard, &query_terms, &shard_frequency_by_term, 4, 4000.0);

    let t_rust = 20.0 / (20.0 + 50.0 + 150.0 * (5000.0 / 4000.0));
    let i_rust = ((4.0_f64 + 0.5) / 2.0).ln() / (4.0_f64 + 1.0).ln();
    let belief_rust = 0.4 + 0.6 * t_rust * i_rust;

    let t_python = 5.0 / (5.0 + 50.0 + 150.0 * (5000.0 / 4000.0));
    let i_python = ((4.0_f64 + 0.5) / 1.0).ln() / (4.0_f64 + 1.0).ln();
    let belief_python = 0.4 + 0.6 * t_python * i_python;

    let expected = (belief_rust + belief_python) / 2.0;
    assert!(
        approx(score.score, expected, 1e-9),
        "got {} expected {expected}",
        score.score
    );
    assert_eq!(score.matched_term_count, 2);
}

#[test]
fn score_shard_matched_term_count_counts_occurrences_with_duplicates() {
    let engine = engine_with_default_config();
    let shard = make_shard("s", 100, 50.0).with_term("x", 5, 5);
    let shard_frequency_by_term: HashMap<&str, usize> = HashMap::from([("x", 1)]);
    let query_terms = vec!["x".to_string(), "x".to_string(), "y".to_string()];
    let score = engine.score_shard(&shard, &query_terms, &shard_frequency_by_term, 1, 5000.0);
    assert_eq!(score.matched_term_count, 2);
}

#[test]
fn score_shard_empty_query_terms_returns_zero_score() {
    let engine = engine_with_default_config();
    let shard = make_shard("s", 100, 50.0).with_term("x", 5, 5);
    let shard_frequency_by_term: HashMap<&str, usize> = HashMap::new();
    let score = engine.score_shard(&shard, &[], &shard_frequency_by_term, 1, 5000.0);
    assert_eq!(score.score, 0.0);
    assert_eq!(score.matched_term_count, 0);
}

#[test]
fn score_shard_term_absent_from_this_shard_floors_even_when_other_shards_have_it() {
    let engine = engine_with_default_config();
    let shard_without_term = make_shard("empty", 100, 50.0); // no term stats recorded
    // shard_frequency = 3 out of a 5-shard candidate set: I is genuinely
    // positive here, not the degenerate all-absent zero case.
    let shard_frequency_by_term: HashMap<&str, usize> = HashMap::from([("rust", 3)]);
    let query_terms = vec!["rust".to_string()];
    let score = engine.score_shard(
        &shard_without_term,
        &query_terms,
        &shard_frequency_by_term,
        5,
        1000.0,
    );
    assert!(approx(score.score, 0.4, 1e-12));
    assert_eq!(score.matched_term_count, 0);
}

// ── select (full pipeline) ────────────────────────────────────────────────────

#[test]
fn select_ranks_shards_by_cori_score_descending() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("weak", 100, 50.0).with_term("rust", 1, 1),
        make_shard("strong", 100, 50.0).with_term("rust", 40, 90),
        make_shard("medium", 100, 50.0).with_term("rust", 10, 20),
    ];
    let query_terms = vec!["rust".to_string()];
    let ranked = engine.select(&shards, &query_terms, 10).expect("select ok");
    let ids: Vec<&str> = ranked.iter().map(|s| s.shard_id.as_str()).collect();
    assert_eq!(ids, vec!["strong", "medium", "weak"]);
    assert!(ranked[0].score > ranked[1].score);
    assert!(ranked[1].score > ranked[2].score);
}

#[test]
fn select_top_k_truncates_ranked_list() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("a", 100, 50.0).with_term("x", 5, 5),
        make_shard("b", 100, 50.0).with_term("x", 10, 10),
        make_shard("c", 100, 50.0).with_term("x", 15, 15),
        make_shard("d", 100, 50.0).with_term("x", 20, 20),
        make_shard("e", 100, 50.0).with_term("x", 25, 25),
    ];
    let query_terms = vec!["x".to_string()];
    let full = engine.select(&shards, &query_terms, 5).expect("select ok");
    let truncated = engine.select(&shards, &query_terms, 2).expect("select ok");
    assert_eq!(truncated.len(), 2);
    assert_eq!(truncated.as_slice(), &full[..2]);
}

#[test]
fn select_top_k_zero_returns_empty_ok() {
    let engine = engine_with_default_config();
    let shards = vec![make_shard("a", 10, 10.0).with_term("x", 1, 1)];
    let query_terms = vec!["x".to_string()];
    let ranked = engine
        .select(&shards, &query_terms, 0)
        .expect("top_k=0 is a valid empty-result request, not an error");
    assert!(ranked.is_empty());
}

#[test]
fn select_top_k_larger_than_shard_count_returns_all() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("a", 10, 10.0).with_term("x", 1, 1),
        make_shard("b", 10, 10.0).with_term("x", 2, 2),
    ];
    let query_terms = vec!["x".to_string()];
    let ranked = engine
        .select(&shards, &query_terms, 1000)
        .expect("select ok");
    assert_eq!(ranked.len(), 2);
}

#[test]
fn select_empty_shards_errors() {
    let engine = engine_with_default_config();
    let result = engine.select(&[], &["x".to_string()], 10);
    assert!(matches!(result, Err(ShardSelectionError::EmptyShardSet)));
}

#[test]
fn select_empty_query_terms_errors() {
    let engine = engine_with_default_config();
    let shards = vec![make_shard("a", 10, 10.0)];
    let result = engine.select(&shards, &[], 10);
    assert!(matches!(result, Err(ShardSelectionError::EmptyQueryTerms)));
}

#[test]
fn select_duplicate_shard_id_errors() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("dup", 10, 10.0).with_term("x", 1, 1),
        make_shard("dup", 20, 20.0).with_term("x", 2, 2),
    ];
    let result = engine.select(&shards, &["x".to_string()], 10);
    match result {
        Err(ShardSelectionError::DuplicateShardId { shard_id }) => assert_eq!(shard_id, "dup"),
        other => panic!("expected DuplicateShardId, got {other:?}"),
    }
}

#[test]
fn select_invalid_avg_doc_length_negative_errors() {
    let engine = engine_with_default_config();
    let shards = vec![make_shard("bad", 10, -1.0).with_term("x", 1, 1)];
    let result = engine.select(&shards, &["x".to_string()], 10);
    assert!(matches!(
        result,
        Err(ShardSelectionError::InvalidShardDescriptor { .. })
    ));
}

#[test]
fn select_invalid_avg_doc_length_nan_errors() {
    let engine = engine_with_default_config();
    let shards = vec![make_shard("bad", 10, f64::NAN).with_term("x", 1, 1)];
    let result = engine.select(&shards, &["x".to_string()], 10);
    assert!(matches!(
        result,
        Err(ShardSelectionError::InvalidShardDescriptor { .. })
    ));
}

#[test]
fn select_invalid_avg_doc_length_infinite_errors() {
    let engine = engine_with_default_config();
    let shards = vec![make_shard("bad", 10, f64::INFINITY).with_term("x", 1, 1)];
    let result = engine.select(&shards, &["x".to_string()], 10);
    assert!(matches!(
        result,
        Err(ShardSelectionError::InvalidShardDescriptor { .. })
    ));
}

#[test]
fn select_deterministic_across_repeated_calls() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("a", 120, 48.0)
            .with_term("x", 7, 9)
            .with_term("y", 3, 3),
        make_shard("b", 80, 63.0).with_term("x", 2, 2),
        make_shard("c", 200, 30.0).with_term("y", 11, 22),
    ];
    let query_terms = vec!["x".to_string(), "y".to_string()];
    let first = engine.select(&shards, &query_terms, 10).expect("select ok");
    for _ in 0..5 {
        let repeat = engine.select(&shards, &query_terms, 10).expect("select ok");
        assert_eq!(repeat, first, "repeated select() calls must be identical");
    }
}

#[test]
fn select_ties_broken_by_ascending_shard_id() {
    let engine = engine_with_default_config();
    // Deliberately out-of-order ids; none of them have any stats for the
    // query term, so every shard floors at an identical score.
    let shards = vec![
        make_shard("zeta", 100, 50.0),
        make_shard("alpha", 100, 50.0),
        make_shard("mid", 100, 50.0),
    ];
    let query_terms = vec!["nonexistent".to_string()];
    let ranked = engine.select(&shards, &query_terms, 10).expect("select ok");
    let ids: Vec<&str> = ranked.iter().map(|s| s.shard_id.as_str()).collect();
    assert_eq!(ids, vec!["alpha", "mid", "zeta"]);
    for score in &ranked {
        assert!(approx(score.score, 0.4, 1e-12));
    }
}

#[test]
fn select_small_shard_beats_large_shard_same_df_end_to_end() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("large", 100, 100.0).with_term("x", 5, 5), // cw = 10_000
        make_shard("small", 10, 100.0).with_term("x", 5, 5),  // cw = 1_000
    ];
    let query_terms = vec!["x".to_string()];
    let ranked = engine.select(&shards, &query_terms, 10).expect("select ok");
    assert_eq!(ranked[0].shard_id, "small");
    assert_eq!(ranked[1].shard_id, "large");
}

#[test]
fn select_query_terms_with_duplicates_weight_average() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("a", 100, 50.0).with_term("rust", 20, 40),
        make_shard("b", 100, 50.0).with_term("rust", 20, 40),
    ];
    // "rust" repeated 3x, "python" absent everywhere (always floors at 0.4)
    // -- the repeated occurrences should pull the average toward "rust"'s
    // belief, proportionally to how many times it was repeated.
    let query_terms = vec![
        "rust".to_string(),
        "rust".to_string(),
        "rust".to_string(),
        "python".to_string(),
    ];
    let ranked = engine.select(&shards, &query_terms, 10).expect("select ok");

    let cw = 100.0 * 50.0;
    let avgcw = cw; // both shards are identical size
    let t_rust = 20.0 / (20.0 + 50.0 + 150.0 * (cw / avgcw));
    let i_rust = ((2.0_f64 + 0.5) / 2.0).ln() / (2.0_f64 + 1.0).ln();
    let belief_rust = 0.4 + 0.6 * t_rust * i_rust;
    let expected = (3.0 * belief_rust + 0.4) / 4.0;

    for score in &ranked {
        assert!(
            approx(score.score, expected, 1e-9),
            "got {} expected {expected}",
            score.score
        );
    }
}

#[test]
fn select_single_shard_single_term_hand_computed() {
    let engine = engine_with_default_config();
    let shards = vec![make_shard("only", 40, 25.0).with_term("rust", 8, 16)];
    let query_terms = vec!["rust".to_string()];
    let ranked = engine.select(&shards, &query_terms, 10).expect("select ok");

    let cw = 40.0 * 25.0; // 1000.0; also avgcw since it is the only shard
    let t = 8.0 / (8.0 + 50.0 + 150.0 * (cw / cw));
    let i = ((1.0_f64 + 0.5) / 1.0).ln() / (1.0_f64 + 1.0).ln();
    let expected = 0.4 + 0.6 * t * i;

    assert_eq!(ranked.len(), 1);
    assert!(approx(ranked[0].score, expected, 1e-9));
}

#[test]
fn select_shard_containing_no_query_terms_gets_floor_score_end_to_end() {
    let engine = engine_with_default_config();
    let shards = vec![
        make_shard("has-term-1", 100, 50.0).with_term("rust", 10, 20),
        make_shard("has-term-2", 100, 50.0).with_term("rust", 15, 30),
        make_shard("no-term", 100, 50.0),
    ];
    let query_terms = vec!["rust".to_string()];
    let ranked = engine.select(&shards, &query_terms, 10).expect("select ok");

    let no_term_score = ranked
        .iter()
        .find(|s| s.shard_id == "no-term")
        .expect("no-term shard present in result");
    assert!(approx(no_term_score.score, 0.4, 1e-12));
    assert_eq!(no_term_score.matched_term_count, 0);

    let matched_score = ranked
        .iter()
        .find(|s| s.shard_id == "has-term-1")
        .expect("has-term-1 shard present in result");
    assert!(no_term_score.score < matched_score.score);
}

// ── ShardSelector façade ─────────────────────────────────────────────────────

#[test]
fn shard_selector_default_matches_default_config() {
    let selector = ShardSelector::default();
    assert_eq!(selector.config(), &ShardSelectionConfig::default());
}

#[test]
fn shard_selector_config_reflects_constructor_argument() {
    let config = ShardSelectionConfig::new().with_default_top_k(3);
    let selector = ShardSelector::new(config.clone());
    assert_eq!(selector.config(), &config);
}

#[test]
fn shard_selector_select_matches_engine_select() {
    let config = ShardSelectionConfig::default();
    let shards = vec![
        make_shard("a", 100, 50.0).with_term("x", 5, 5),
        make_shard("b", 100, 50.0).with_term("x", 10, 10),
    ];
    let query_terms = vec!["x".to_string()];

    let engine = ShardSelectionEngine::new(config.clone());
    let via_engine = engine.select(&shards, &query_terms, 10).expect("select ok");

    let selector = ShardSelector::new(config);
    let via_selector = selector
        .select(&shards, &query_terms, 10)
        .expect("select ok");

    assert_eq!(via_engine, via_selector);
}

#[test]
fn shard_selector_propagates_empty_shard_set_error() {
    let selector = ShardSelector::default();
    let result = selector.select(&[], &["x".to_string()], 10);
    assert!(matches!(result, Err(ShardSelectionError::EmptyShardSet)));
}
