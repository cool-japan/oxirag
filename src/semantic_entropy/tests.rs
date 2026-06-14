//! Tests for the `semantic_entropy` module.

use crate::semantic_entropy::{
    MeaningCluster, SemanticEntropyConfig, SemanticEntropyError, SemanticEntropyEstimator,
    SemanticEntropyResult,
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Absolute-difference float comparison with a fixed tolerance.
#[allow(clippy::float_cmp)]
fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// Build a `Vec<String>` from string literals.
fn answers(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

/// Default estimator (threshold 0.6, natural log).
fn estimator() -> SemanticEntropyEstimator {
    SemanticEntropyEstimator::default()
}

// ── SemanticEntropyConfig: defaults + builders ────────────────────────────────

#[test]
fn test_config_default_threshold() {
    let cfg = SemanticEntropyConfig::default();
    assert!(approx(cfg.equivalence_threshold, 0.6));
}

#[test]
fn test_config_default_use_log2_is_false() {
    let cfg = SemanticEntropyConfig::default();
    assert!(!cfg.use_log2);
}

#[test]
fn test_config_new_matches_default() {
    let cfg = SemanticEntropyConfig::new();
    assert!(approx(cfg.equivalence_threshold, 0.6));
}

#[test]
fn test_config_new_use_log2_false() {
    let cfg = SemanticEntropyConfig::new();
    assert!(!cfg.use_log2);
}

#[test]
fn test_config_with_equivalence_threshold() {
    let cfg = SemanticEntropyConfig::new().with_equivalence_threshold(0.8);
    assert!(approx(cfg.equivalence_threshold, 0.8));
}

#[test]
fn test_config_with_use_log2() {
    let cfg = SemanticEntropyConfig::new().with_use_log2(true);
    assert!(cfg.use_log2);
}

#[test]
fn test_config_builder_chaining() {
    let cfg = SemanticEntropyConfig::new()
        .with_equivalence_threshold(0.5)
        .with_use_log2(true);
    assert!(approx(cfg.equivalence_threshold, 0.5) && cfg.use_log2);
}

#[test]
fn test_config_clone() {
    let cfg = SemanticEntropyConfig::new().with_equivalence_threshold(0.42);
    let cloned = cfg.clone();
    assert!(approx(cloned.equivalence_threshold, 0.42));
}

// ── estimator construction ────────────────────────────────────────────────────

#[test]
fn test_estimator_new_keeps_config() {
    let est = SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_use_log2(true));
    assert!(est.config.use_log2);
}

#[test]
fn test_estimator_default_threshold() {
    let est = SemanticEntropyEstimator::default();
    assert!(approx(est.config.equivalence_threshold, 0.6));
}

// ── error: empty samples ──────────────────────────────────────────────────────

#[test]
fn test_estimate_empty_samples_errors() {
    let est = estimator();
    let empty: Vec<String> = Vec::new();
    assert!(matches!(
        est.estimate(&empty),
        Err(SemanticEntropyError::EmptySamples)
    ));
}

#[test]
fn test_estimate_weighted_empty_samples_errors() {
    let est = estimator();
    let empty: Vec<String> = Vec::new();
    assert!(matches!(
        est.estimate_weighted(&empty, &[]),
        Err(SemanticEntropyError::EmptySamples)
    ));
}

#[test]
fn test_is_uncertain_empty_samples_errors() {
    let est = estimator();
    let empty: Vec<String> = Vec::new();
    assert!(matches!(
        est.is_uncertain(&empty, 0.5),
        Err(SemanticEntropyError::EmptySamples)
    ));
}

// ── error: weight mismatch ────────────────────────────────────────────────────

#[test]
fn test_estimate_weighted_mismatch_errors() {
    let est = estimator();
    let ans = answers(&["a", "b", "c"]);
    assert!(matches!(
        est.estimate_weighted(&ans, &[1.0, 1.0]),
        Err(SemanticEntropyError::WeightMismatch {
            weights: 2,
            answers: 3
        })
    ));
}

#[test]
fn test_weight_mismatch_reports_lengths() {
    let est = estimator();
    let ans = answers(&["a", "b"]);
    let err = est
        .estimate_weighted(&ans, &[1.0, 1.0, 1.0, 1.0])
        .unwrap_err();
    let msg = err.to_string();
    assert_eq!(msg, "weights length 4 != answers length 2");
}

#[test]
fn test_empty_samples_error_message() {
    let err = SemanticEntropyError::EmptySamples;
    assert_eq!(err.to_string(), "no answer samples");
}

// ── all-identical answers ─────────────────────────────────────────────────────

#[test]
fn test_identical_single_cluster() {
    let est = estimator();
    let ans = answers(&["the sky is blue", "the sky is blue", "the sky is blue"]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 1);
}

#[test]
fn test_identical_entropy_zero() {
    let est = estimator();
    let ans = answers(&["the sky is blue", "the sky is blue", "the sky is blue"]);
    let result = est.estimate(&ans).unwrap();
    assert!(approx(result.entropy, 0.0));
}

#[test]
fn test_identical_normalized_zero() {
    let est = estimator();
    let ans = answers(&["the sky is blue", "the sky is blue", "the sky is blue"]);
    let result = est.estimate(&ans).unwrap();
    assert!(approx(result.normalized_entropy, 0.0));
}

#[test]
fn test_identical_not_uncertain() {
    let est = estimator();
    let ans = answers(&["the sky is blue", "the sky is blue", "the sky is blue"]);
    assert!(!est.is_uncertain(&ans, 0.5).unwrap());
}

#[test]
fn test_identical_cluster_probability_one() {
    let est = estimator();
    let ans = answers(&["alpha beta", "alpha beta", "alpha beta"]);
    let result = est.estimate(&ans).unwrap();
    assert!(approx(result.clusters[0].probability, 1.0));
}

// ── all-distinct answers ──────────────────────────────────────────────────────

#[test]
fn test_distinct_cluster_count() {
    let est = estimator();
    let ans = answers(&["alpha", "beta", "gamma", "delta"]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 4);
}

#[test]
fn test_distinct_entropy_equals_ln_k() {
    let est = estimator();
    let ans = answers(&["alpha", "beta", "gamma", "delta"]);
    let result = est.estimate(&ans).unwrap();
    assert!(approx(result.entropy, 4.0_f32.ln()));
}

#[test]
fn test_distinct_normalized_is_one() {
    let est = estimator();
    let ans = answers(&["alpha", "beta", "gamma", "delta", "epsilon"]);
    let result = est.estimate(&ans).unwrap();
    assert!(approx(result.normalized_entropy, 1.0));
}

#[test]
fn test_distinct_is_uncertain() {
    let est = estimator();
    let ans = answers(&["alpha", "beta", "gamma", "delta"]);
    assert!(est.is_uncertain(&ans, 0.5).unwrap());
}

#[test]
fn test_distinct_log2_entropy_equals_log2_k() {
    let est = SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_use_log2(true));
    let ans = answers(&["alpha", "beta", "gamma", "delta"]);
    let result = est.estimate(&ans).unwrap();
    // log2(4) == 2.0
    assert!(approx(result.entropy, 2.0));
}

// ── paraphrase clustering ─────────────────────────────────────────────────────

#[test]
fn test_paraphrase_merges_into_one_cluster() {
    let est = estimator();
    let ans = answers(&["Paris is the capital", "the capital is Paris"]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 1);
}

#[test]
fn test_paraphrase_cluster_has_two_members() {
    let est = estimator();
    let ans = answers(&["Paris is the capital", "the capital is Paris"]);
    let clusters = est.cluster_answers(&ans);
    assert_eq!(clusters[0].members.len(), 2);
}

#[test]
fn test_paraphrase_entropy_zero() {
    let est = estimator();
    let ans = answers(&["Paris is the capital", "the capital is Paris"]);
    let result = est.estimate(&ans).unwrap();
    assert!(approx(result.entropy, 0.0));
}

#[test]
fn test_paraphrase_longer_sentences_merge() {
    let est = estimator();
    let ans = answers(&[
        "Paris is the capital of France",
        "The capital of France is Paris",
    ]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 1);
}

#[test]
fn test_distinct_meaning_does_not_merge() {
    let est = estimator();
    let ans = answers(&[
        "Paris is the capital of France",
        "Berlin is the capital of Germany",
    ]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 2);
}

// ── mixed (3 same, 1 different) ───────────────────────────────────────────────

#[test]
fn test_mixed_two_clusters() {
    let est = estimator();
    let ans = answers(&[
        "the answer is forty two",
        "the answer is forty two",
        "the answer is forty two",
        "the answer is nine",
    ]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 2);
}

#[test]
fn test_mixed_entropy_between_zero_and_max() {
    let est = estimator();
    let ans = answers(&[
        "the answer is forty two",
        "the answer is forty two",
        "the answer is forty two",
        "the answer is nine",
    ]);
    let result = est.estimate(&ans).unwrap();
    let max = 2.0_f32.ln();
    assert!(result.entropy > 0.0 && result.entropy < max);
}

#[test]
fn test_mixed_entropy_exact_value() {
    let est = estimator();
    let ans = answers(&[
        "the answer is forty two",
        "the answer is forty two",
        "the answer is forty two",
        "the answer is nine",
    ]);
    let result = est.estimate(&ans).unwrap();
    // H = -(0.75 ln 0.75 + 0.25 ln 0.25)
    let expected = -(0.75_f32 * 0.75_f32.ln() + 0.25_f32 * 0.25_f32.ln());
    assert!(approx(result.entropy, expected));
}

#[test]
fn test_mixed_normalized_in_unit_range() {
    let est = estimator();
    let ans = answers(&[
        "the answer is forty two",
        "the answer is forty two",
        "the answer is forty two",
        "the answer is nine",
    ]);
    let result = est.estimate(&ans).unwrap();
    assert!(result.normalized_entropy >= 0.0 && result.normalized_entropy <= 1.0);
}

#[test]
fn test_mixed_majority_cluster_probability() {
    let est = estimator();
    let ans = answers(&[
        "the answer is forty two",
        "the answer is forty two",
        "the answer is forty two",
        "the answer is nine",
    ]);
    let result = est.estimate(&ans).unwrap();
    let max_prob = result
        .clusters
        .iter()
        .map(|c| c.probability)
        .fold(0.0_f32, f32::max);
    assert!(approx(max_prob, 0.75));
}

// ── normalized_entropy range over several inputs ──────────────────────────────

#[test]
fn test_normalized_entropy_always_in_range() {
    let est = estimator();
    let cases = [
        answers(&["a", "a"]),
        answers(&["a", "b"]),
        answers(&["a", "b", "c"]),
        answers(&["x x x", "x x x", "y", "z"]),
    ];
    let all_ok = cases.iter().all(|c| {
        let r = est.estimate(c).unwrap();
        r.normalized_entropy >= 0.0 && r.normalized_entropy <= 1.0
    });
    assert!(all_ok);
}

// ── use_log2 changes raw entropy but not normalized ───────────────────────────

#[test]
fn test_log2_changes_raw_entropy() {
    let nat = SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_use_log2(false));
    let bits = SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_use_log2(true));
    let ans = answers(&["alpha", "beta", "gamma"]);
    let h_nat = nat.estimate(&ans).unwrap().entropy;
    let h_bits = bits.estimate(&ans).unwrap().entropy;
    assert!(!approx(h_nat, h_bits));
}

#[test]
fn test_log2_preserves_normalized_entropy() {
    let nat = SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_use_log2(false));
    let bits = SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_use_log2(true));
    let ans = answers(&[
        "the answer is forty two",
        "the answer is forty two",
        "the answer is forty two",
        "the answer is nine",
    ]);
    let n_nat = nat.estimate(&ans).unwrap().normalized_entropy;
    let n_bits = bits.estimate(&ans).unwrap().normalized_entropy;
    assert!(approx(n_nat, n_bits));
}

#[test]
fn test_log2_ratio_is_ln2() {
    let nat = SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_use_log2(false));
    let bits = SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_use_log2(true));
    let ans = answers(&["alpha", "beta", "gamma"]);
    let h_nat = nat.estimate(&ans).unwrap().entropy;
    let h_bits = bits.estimate(&ans).unwrap().entropy;
    // H_nat = H_bits * ln(2)
    assert!(approx(h_nat, h_bits * std::f32::consts::LN_2));
}

// ── weighted variant skews probabilities ──────────────────────────────────────

#[test]
fn test_weighted_skews_probability() {
    let est = estimator();
    let ans = answers(&["alpha", "beta"]);
    let result = est.estimate_weighted(&ans, &[3.0, 1.0]).unwrap();
    let alpha_prob = result.clusters[0].probability;
    assert!(approx(alpha_prob, 0.75));
}

#[test]
fn test_weighted_entropy_differs_from_uniform() {
    let est = estimator();
    let ans = answers(&["alpha", "beta"]);
    let uniform = est.estimate(&ans).unwrap().entropy;
    let skewed = est.estimate_weighted(&ans, &[3.0, 1.0]).unwrap().entropy;
    assert!(!approx(uniform, skewed));
}

#[test]
fn test_weighted_uniform_matches_estimate() {
    let est = estimator();
    let ans = answers(&["alpha", "beta", "gamma"]);
    let plain = est.estimate(&ans).unwrap().entropy;
    let weighted = est
        .estimate_weighted(&ans, &[1.0, 1.0, 1.0])
        .unwrap()
        .entropy;
    assert!(approx(plain, weighted));
}

#[test]
fn test_weighted_skewed_entropy_value() {
    let est = estimator();
    let ans = answers(&["alpha", "beta"]);
    let result = est.estimate_weighted(&ans, &[3.0, 1.0]).unwrap();
    let expected = -(0.75_f32 * 0.75_f32.ln() + 0.25_f32 * 0.25_f32.ln());
    assert!(approx(result.entropy, expected));
}

#[test]
fn test_weighted_probabilities_sum_to_one() {
    let est = estimator();
    let ans = answers(&["alpha", "beta", "gamma"]);
    let result = est.estimate_weighted(&ans, &[2.0, 5.0, 3.0]).unwrap();
    let total: f32 = result.clusters.iter().map(|c| c.probability).sum();
    assert!(approx(total, 1.0));
}

// ── predictive_entropy ≥ semantic entropy ─────────────────────────────────────

#[test]
fn test_predictive_ge_semantic_on_paraphrases() {
    let est = estimator();
    let ans = answers(&[
        "Paris is the capital of France",
        "The capital of France is Paris",
        "Berlin is the capital of Germany",
    ]);
    let result = est.estimate(&ans).unwrap();
    assert!(result.predictive_entropy >= result.entropy);
}

#[test]
fn test_predictive_strictly_greater_when_merging() {
    let est = estimator();
    let ans = answers(&[
        "Paris is the capital of France",
        "The capital of France is Paris",
        "Berlin is the capital of Germany",
    ]);
    let result = est.estimate(&ans).unwrap();
    // Three samples but only two meanings ⇒ naive entropy strictly larger.
    assert!(result.predictive_entropy > result.entropy);
}

#[test]
fn test_predictive_equals_semantic_when_all_distinct() {
    let est = estimator();
    let ans = answers(&["alpha", "beta", "gamma"]);
    let result = est.estimate(&ans).unwrap();
    assert!(approx(result.predictive_entropy, result.entropy));
}

#[test]
fn test_predictive_equals_ln_n() {
    let est = estimator();
    let ans = answers(&["one", "one", "one", "two"]);
    let result = est.estimate(&ans).unwrap();
    // Naive treats each of the 4 answers as its own cluster ⇒ ln(4).
    assert!(approx(result.predictive_entropy, 4.0_f32.ln()));
}

// ── invariants: members sum, num_samples, sizes ───────────────────────────────

#[test]
fn test_members_sum_equals_num_samples() {
    let est = estimator();
    let ans = answers(&["x x x", "x x x", "y", "z", "y"]);
    let result = est.estimate(&ans).unwrap();
    let total: usize = result.clusters.iter().map(MeaningCluster::size).sum();
    assert_eq!(total, result.num_samples);
}

#[test]
fn test_num_samples_matches_input_len() {
    let est = estimator();
    let ans = answers(&["a", "b", "c", "d", "e", "f"]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_samples, 6);
}

#[test]
fn test_cluster_member_indices_cover_all() {
    let est = estimator();
    let ans = answers(&["x x x", "x x x", "y", "z"]);
    let clusters = est.cluster_answers(&ans);
    let mut indices: Vec<usize> = clusters.iter().flat_map(|c| c.members.clone()).collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![0, 1, 2, 3]);
}

#[test]
fn test_cluster_representative_is_first_member() {
    let est = estimator();
    let ans = answers(&["hello world", "hello world"]);
    let clusters = est.cluster_answers(&ans);
    assert_eq!(clusters[0].representative, "hello world");
}

#[test]
fn test_cluster_size_helper() {
    let cluster = MeaningCluster {
        representative: "r".to_string(),
        members: vec![0, 2, 5],
        probability: 0.5,
    };
    assert_eq!(cluster.size(), 3);
}

#[test]
fn test_clusters_in_first_seen_order() {
    let est = estimator();
    let ans = answers(&["zebra", "apple"]);
    let clusters = est.cluster_answers(&ans);
    assert_eq!(clusters[0].representative, "zebra");
}

// ── single sample ─────────────────────────────────────────────────────────────

#[test]
fn test_single_sample_one_cluster() {
    let est = estimator();
    let ans = answers(&["only one answer"]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 1);
}

#[test]
fn test_single_sample_entropy_zero() {
    let est = estimator();
    let ans = answers(&["only one answer"]);
    let result = est.estimate(&ans).unwrap();
    assert!(approx(result.entropy, 0.0));
}

// ── threshold sensitivity ─────────────────────────────────────────────────────

#[test]
fn test_lower_threshold_merges_more() {
    let loose =
        SemanticEntropyEstimator::new(SemanticEntropyConfig::new().with_equivalence_threshold(0.3));
    let ans = answers(&["red green blue", "red green yellow"]);
    let result = loose.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 1);
}

#[test]
fn test_higher_threshold_splits_more() {
    let strict = SemanticEntropyEstimator::new(
        SemanticEntropyConfig::new().with_equivalence_threshold(0.95),
    );
    let ans = answers(&["red green blue", "red green yellow"]);
    let result = strict.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 2);
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_determinism_entropy() {
    let est = estimator();
    let ans = answers(&[
        "Paris is the capital of France",
        "The capital of France is Paris",
        "Berlin is the capital of Germany",
        "Rome is the capital of Italy",
    ]);
    let first = est.estimate(&ans).unwrap().entropy;
    let second = est.estimate(&ans).unwrap().entropy;
    assert!(approx(first, second));
}

#[test]
fn test_determinism_cluster_count() {
    let est = estimator();
    let ans = answers(&["x x x", "x x x", "y y", "z", "y y"]);
    let first = est.cluster_answers(&ans).len();
    let second = est.cluster_answers(&ans).len();
    assert_eq!(first, second);
}

#[test]
fn test_determinism_cluster_membership() {
    let est = estimator();
    let ans = answers(&["alpha one", "alpha one", "beta two", "gamma three"]);
    let first: Vec<Vec<usize>> = est
        .cluster_answers(&ans)
        .into_iter()
        .map(|c| c.members)
        .collect();
    let second: Vec<Vec<usize>> = est
        .cluster_answers(&ans)
        .into_iter()
        .map(|c| c.members)
        .collect();
    assert_eq!(first, second);
}

// ── is_uncertain threshold boundary ───────────────────────────────────────────

#[test]
fn test_is_uncertain_high_threshold_false() {
    let est = estimator();
    let ans = answers(&["alpha", "beta"]);
    // normalized_entropy == 1.0; threshold 1.0 is not strictly exceeded.
    assert!(!est.is_uncertain(&ans, 1.0).unwrap());
}

#[test]
fn test_is_uncertain_low_threshold_true() {
    let est = estimator();
    let ans = answers(&["alpha", "beta"]);
    assert!(est.is_uncertain(&ans, 0.1).unwrap());
}

// ── result field exposure ─────────────────────────────────────────────────────

#[test]
fn test_result_clusters_len_matches_num_clusters() {
    let est = estimator();
    let ans = answers(&["a a a", "a a a", "b", "c"]);
    let result: SemanticEntropyResult = est.estimate(&ans).unwrap();
    assert_eq!(result.clusters.len(), result.num_clusters);
}

#[test]
fn test_tokenizer_ignores_punctuation_for_equivalence() {
    let est = estimator();
    // Same tokens modulo punctuation/case ⇒ a single cluster.
    let ans = answers(&["Paris, is the Capital!", "paris is the capital"]);
    let result = est.estimate(&ans).unwrap();
    assert_eq!(result.num_clusters, 1);
}
