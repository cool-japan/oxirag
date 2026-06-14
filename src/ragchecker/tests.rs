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
//! Tests for the `ragchecker` module.

use crate::ragchecker::checker::RagChecker;
use crate::ragchecker::types::{
    RagCheckResult, RagCheckerConfig, RagCheckerError, RagCheckerMetrics,
};
use crate::types::Document;

// ── helpers ──────────────────────────────────────────────────────────────────

fn docs(contents: &[&str]) -> Vec<Document> {
    contents.iter().map(|c| Document::new(*c)).collect()
}

fn default_checker() -> RagChecker {
    RagChecker::new(RagCheckerConfig::default())
}

/// Assert every metric of `m` lies within `[0.0, 1.0]`.
fn assert_all_in_unit(m: &RagCheckerMetrics) {
    for v in [
        m.claim_recall,
        m.context_precision,
        m.faithfulness,
        m.hallucination_rate,
        m.correctness,
        m.noise_sensitivity,
    ] {
        assert!(v >= 0.0 && v <= 1.0, "metric out of range: {v}");
    }
}

// ── config defaults & builder ────────────────────────────────────────────────

#[test]
fn test_config_default_threshold() {
    assert_eq!(RagCheckerConfig::default().entailment_threshold, 0.5);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(RagCheckerConfig::new(), RagCheckerConfig::default());
}

#[test]
fn test_config_builder_sets_threshold() {
    let cfg = RagCheckerConfig::new().with_entailment_threshold(0.75);
    assert_eq!(cfg.entailment_threshold, 0.75);
}

#[test]
fn test_config_builder_clamps_high() {
    assert_eq!(
        RagCheckerConfig::new()
            .with_entailment_threshold(2.0)
            .entailment_threshold,
        1.0
    );
}

#[test]
fn test_config_builder_clamps_low() {
    assert_eq!(
        RagCheckerConfig::new()
            .with_entailment_threshold(-1.0)
            .entailment_threshold,
        0.0
    );
}

#[test]
fn test_checker_default() {
    assert_eq!(RagChecker::default().config.entailment_threshold, 0.5);
}

#[test]
fn test_checker_new_stores_config() {
    let cfg = RagCheckerConfig::new().with_entailment_threshold(0.3);
    assert_eq!(RagChecker::new(cfg).config.entailment_threshold, 0.3);
}

// ── entailed() ───────────────────────────────────────────────────────────────

#[test]
fn test_entailed_full_overlap() {
    let checker = default_checker();
    assert!(checker.entailed("the eiffel tower", "the eiffel tower stands in paris"));
}

#[test]
fn test_entailed_partial_above_threshold() {
    // claim tokens {rust, language, fast}; text has rust+fast => 2/3 >= 0.5.
    assert!(default_checker().entailed("rust language fast", "rust is very fast indeed"));
}

#[test]
fn test_entailed_partial_below_threshold() {
    // claim {alpha, beta, gamma, delta}; text has only alpha => 1/4 < 0.5.
    assert!(!default_checker().entailed("alpha beta gamma delta", "alpha only here"));
}

#[test]
fn test_entailed_empty_claim_is_false() {
    assert!(!default_checker().entailed("", "anything at all here"));
}

#[test]
fn test_entailed_punctuation_only_claim_is_false() {
    // No alphanumeric tokens of length >= 2.
    assert!(!default_checker().entailed("!! ?? ..", "anything at all here"));
}

#[test]
fn test_entailed_case_insensitive() {
    assert!(default_checker().entailed("RUST Language", "the rust LANGUAGE rocks"));
}

#[test]
fn test_entailed_threshold_one_requires_all() {
    let checker = RagChecker::new(RagCheckerConfig::new().with_entailment_threshold(1.0));
    assert!(checker.entailed("rust fast", "rust is fast"));
    assert!(!checker.entailed("rust fast safe", "rust is fast"));
}

#[test]
fn test_entailed_threshold_zero_true_for_nonempty_claim() {
    // 0 / n >= 0.0 holds, so any non-empty claim is entailed regardless of text.
    let checker = RagChecker::new(RagCheckerConfig::new().with_entailment_threshold(0.0));
    assert!(checker.entailed("totally unrelated", "nothing matches"));
}

// ── claim decomposition (observed via result.*_claims) ───────────────────────

#[test]
fn test_decomposition_splits_sentences() {
    let res = default_checker()
        .check(
            "Rust is fast. Rust is safe.",
            "Rust is a language.",
            &docs(&["Rust is fast and safe and a language."]),
        )
        .unwrap();
    assert_eq!(res.response_claims.len(), 2);
}

#[test]
fn test_decomposition_splits_clauses() {
    // " and " clause boundary splits a single sentence into two claims.
    let res = default_checker()
        .check(
            "Einstein was born in Germany and developed relativity.",
            "Einstein lived in Germany.",
            &docs(&["Einstein was born in Germany and developed relativity."]),
        )
        .unwrap();
    assert_eq!(res.response_claims.len(), 2);
}

#[test]
fn test_decomposition_drops_empty_clauses() {
    let res = default_checker()
        .check(
            "Hello world... Done.",
            "Greeting message here.",
            &docs(&["Hello world done greeting message here."]),
        )
        .unwrap();
    assert!(res.response_claims.iter().all(|c| !c.trim().is_empty()));
    assert_eq!(res.response_claims.len(), 2);
}

#[test]
fn test_decomposition_gt_claims_recorded() {
    let res = default_checker()
        .check(
            "Paris is in France.",
            "Paris is the capital. Paris sits on the Seine.",
            &docs(&["Paris is in France and is the capital on the Seine."]),
        )
        .unwrap();
    assert_eq!(res.gt_claims.len(), 2);
}

// ── faithfulness ─────────────────────────────────────────────────────────────

#[test]
fn test_faithfulness_full_when_all_in_context() {
    let res = default_checker()
        .check(
            "The Eiffel Tower is in Paris. The Eiffel Tower is tall.",
            "Unrelated ground truth statement here.",
            &docs(&["The Eiffel Tower is in Paris and the Eiffel Tower is very tall."]),
        )
        .unwrap();
    assert_eq!(res.metrics.faithfulness, 1.0);
}

#[test]
fn test_faithfulness_zero_when_none_in_context() {
    let res = default_checker()
        .check(
            "Bananas grow on tropical plants.",
            "Bananas grow on tropical plants.",
            &docs(&["A completely different subject about cars and engines."]),
        )
        .unwrap();
    assert_eq!(res.metrics.faithfulness, 0.0);
}

#[test]
fn test_faithfulness_half_when_one_of_two_in_context() {
    let res = default_checker()
        .check(
            "Mercury is a planet. Pineapples are sweet fruits.",
            "Some ground truth.",
            &docs(&["Mercury is a planet closest to the sun."]),
        )
        .unwrap();
    assert_eq!(res.metrics.faithfulness, 0.5);
}

#[test]
fn test_faithfulness_any_passage_counts() {
    let res = default_checker()
        .check(
            "Quantum entanglement links particles.",
            "Ground truth statement.",
            &docs(&[
                "Totally irrelevant passage about cooking.",
                "Quantum entanglement links particles across distance.",
            ]),
        )
        .unwrap();
    assert_eq!(res.metrics.faithfulness, 1.0);
}

// ── hallucination_rate ───────────────────────────────────────────────────────

#[test]
fn test_hallucination_positive_when_neither() {
    let res = default_checker()
        .check(
            "Dragons breathe fire.",
            "Cats are mammals.",
            &docs(&["The weather today is sunny and warm."]),
        )
        .unwrap();
    assert!(res.metrics.hallucination_rate > 0.0);
    assert_eq!(res.metrics.hallucination_rate, 1.0);
}

#[test]
fn test_hallucination_zero_when_in_gt_only() {
    // Claim is in GT (not context), so it is not a hallucination.
    let res = default_checker()
        .check(
            "Volcanoes erupt molten lava.",
            "Volcanoes erupt molten lava frequently.",
            &docs(&["A passage with nothing relevant inside it."]),
        )
        .unwrap();
    assert_eq!(res.metrics.hallucination_rate, 0.0);
}

#[test]
fn test_hallucination_partial() {
    // One grounded claim, one hallucinated => 0.5.
    let res = default_checker()
        .check(
            "Mercury is a planet. Unicorns roam enchanted forests.",
            "Mercury is a planet.",
            &docs(&["Mercury is a planet nearest the sun."]),
        )
        .unwrap();
    assert_eq!(res.metrics.hallucination_rate, 0.5);
}

// ── correctness ──────────────────────────────────────────────────────────────

#[test]
fn test_correctness_full_when_response_in_gt() {
    let res = default_checker()
        .check(
            "The capital of Japan is Tokyo.",
            "The capital of Japan is Tokyo, a huge city.",
            &docs(&["Irrelevant retrieved passage about something else."]),
        )
        .unwrap();
    assert_eq!(res.metrics.correctness, 1.0);
}

#[test]
fn test_correctness_zero_when_disjoint_from_gt() {
    let res = default_checker()
        .check(
            "The capital of Japan is Tokyo.",
            "Bananas are yellow tropical fruits.",
            &docs(&["The capital of Japan is Tokyo."]),
        )
        .unwrap();
    assert_eq!(res.metrics.correctness, 0.0);
}

#[test]
fn test_correctness_half() {
    // Only the first claim is entailed by the GT.
    let res = default_checker()
        .check(
            "Saturn has many rings. Saturn has a moon named Titan.",
            "Saturn has many rings around it.",
            &docs(&["Saturn has many rings and a moon named Titan."]),
        )
        .unwrap();
    assert_eq!(res.metrics.correctness, 0.5);
}

// ── claim_recall ─────────────────────────────────────────────────────────────

#[test]
fn test_claim_recall_full() {
    let res = default_checker()
        .check(
            "Some response statement.",
            "Gravity pulls objects down. Friction slows motion.",
            &docs(&[
                "Gravity pulls objects down toward the earth.",
                "Friction slows motion between surfaces.",
            ]),
        )
        .unwrap();
    assert_eq!(res.metrics.claim_recall, 1.0);
}

#[test]
fn test_claim_recall_zero_when_context_irrelevant() {
    let res = default_checker()
        .check(
            "Some response statement.",
            "Gravity pulls objects down.",
            &docs(&["A passage about cooking pasta and sauces."]),
        )
        .unwrap();
    assert_eq!(res.metrics.claim_recall, 0.0);
}

#[test]
fn test_claim_recall_half() {
    // Only the gravity GT claim is in the context.
    let res = default_checker()
        .check(
            "Some response.",
            "Gravity pulls objects down. Friction slows motion.",
            &docs(&["Gravity pulls objects down toward earth."]),
        )
        .unwrap();
    assert_eq!(res.metrics.claim_recall, 0.5);
}

#[test]
fn test_claim_recall_no_documents() {
    let res = default_checker()
        .check("A response.", "A ground truth claim here.", &[])
        .unwrap();
    assert_eq!(res.metrics.claim_recall, 0.0);
}

// ── context_precision ────────────────────────────────────────────────────────

#[test]
fn test_context_precision_all_useful() {
    let res = default_checker()
        .check(
            "Response text.",
            "Gravity pulls objects down. Friction slows motion.",
            &docs(&[
                "Gravity pulls objects down strongly.",
                "Friction slows motion considerably.",
            ]),
        )
        .unwrap();
    assert_eq!(res.metrics.context_precision, 1.0);
}

#[test]
fn test_context_precision_half() {
    let res = default_checker()
        .check(
            "Response text.",
            "Gravity pulls objects down.",
            &docs(&[
                "Gravity pulls objects down strongly.",
                "An irrelevant passage about cooking.",
            ]),
        )
        .unwrap();
    assert_eq!(res.metrics.context_precision, 0.5);
}

#[test]
fn test_context_precision_zero_when_no_passage_supports_gt() {
    let res = default_checker()
        .check(
            "Response text.",
            "Gravity pulls objects down.",
            &docs(&["Cooking pasta requires boiling water and salt."]),
        )
        .unwrap();
    assert_eq!(res.metrics.context_precision, 0.0);
}

#[test]
fn test_context_precision_no_documents() {
    let res = default_checker()
        .check("Response.", "Ground truth claim.", &[])
        .unwrap();
    assert_eq!(res.metrics.context_precision, 0.0);
}

// ── noise_sensitivity ────────────────────────────────────────────────────────

#[test]
fn test_noise_sensitivity_positive_when_in_context_not_gt() {
    // Claim is in context but not in GT => picked up noisy context.
    let res = default_checker()
        .check(
            "Marsupials carry young in pouches.",
            "Birds have feathers and wings.",
            &docs(&["Marsupials carry young in pouches for protection."]),
        )
        .unwrap();
    assert_eq!(res.metrics.noise_sensitivity, 1.0);
}

#[test]
fn test_noise_sensitivity_zero_when_claim_in_gt() {
    // In context AND in GT => not noise.
    let res = default_checker()
        .check(
            "Marsupials carry young in pouches.",
            "Marsupials carry young in pouches always.",
            &docs(&["Marsupials carry young in pouches for protection."]),
        )
        .unwrap();
    assert_eq!(res.metrics.noise_sensitivity, 0.0);
}

#[test]
fn test_noise_sensitivity_zero_when_not_in_context() {
    // Not in context => cannot be noise (it is a hallucination instead).
    let res = default_checker()
        .check(
            "Marsupials carry young in pouches.",
            "Birds have feathers.",
            &docs(&["An unrelated passage about deep ocean currents."]),
        )
        .unwrap();
    assert_eq!(res.metrics.noise_sensitivity, 0.0);
}

#[test]
fn test_noise_sensitivity_half() {
    // Mercury: in context + GT (not noise). Marsupial: in context, not GT (noise).
    let res = default_checker()
        .check(
            "Mercury is a planet. Marsupials carry young in pouches.",
            "Mercury is a planet.",
            &docs(&[
                "Mercury is a planet near the sun.",
                "Marsupials carry young in pouches for protection.",
            ]),
        )
        .unwrap();
    assert_eq!(res.metrics.noise_sensitivity, 0.5);
}

// ── combined / cross-metric scenarios ────────────────────────────────────────

#[test]
fn test_perfect_rag_sample() {
    let res = default_checker()
        .check(
            "The Eiffel Tower is in Paris.",
            "The Eiffel Tower is in Paris.",
            &docs(&["The Eiffel Tower is in Paris, the capital of France."]),
        )
        .unwrap();
    assert_eq!(res.metrics.faithfulness, 1.0);
    assert_eq!(res.metrics.correctness, 1.0);
    assert_eq!(res.metrics.claim_recall, 1.0);
    assert_eq!(res.metrics.context_precision, 1.0);
    assert_eq!(res.metrics.hallucination_rate, 0.0);
    assert_eq!(res.metrics.noise_sensitivity, 0.0);
}

#[test]
fn test_faithful_but_noisy_sample() {
    // Grounded in context, absent from GT.
    let res = default_checker()
        .check(
            "Octopuses have eight arms.",
            "Whales are large marine mammals.",
            &docs(&["Octopuses have eight arms and three hearts."]),
        )
        .unwrap();
    assert_eq!(res.metrics.faithfulness, 1.0);
    assert_eq!(res.metrics.correctness, 0.0);
    assert_eq!(res.metrics.hallucination_rate, 0.0);
    assert_eq!(res.metrics.noise_sensitivity, 1.0);
}

#[test]
fn test_all_metrics_in_unit_interval() {
    let res = default_checker()
        .check(
            "Mercury is a planet. Dragons breathe fire. Saturn has rings.",
            "Mercury is a planet. Saturn has rings around it.",
            &docs(&[
                "Mercury is a planet near the sun.",
                "Saturn has rings of ice and rock.",
                "An irrelevant passage about gardening.",
            ]),
        )
        .unwrap();
    assert_all_in_unit(&res.metrics);
}

#[test]
fn test_metrics_in_unit_with_high_threshold() {
    let checker = RagChecker::new(RagCheckerConfig::new().with_entailment_threshold(1.0));
    let res = checker
        .check(
            "Mercury is a planet. Saturn has rings.",
            "Mercury is a planet.",
            &docs(&["Mercury is a planet.", "Saturn has rings of ice."]),
        )
        .unwrap();
    assert_all_in_unit(&res.metrics);
}

#[test]
fn test_empty_retrieved_is_allowed() {
    // No context => not faithful, but correct via GT.
    let res = default_checker()
        .check("Mercury is a planet.", "Mercury is a planet.", &[])
        .unwrap();
    assert_eq!(res.metrics.faithfulness, 0.0);
    assert_eq!(res.metrics.correctness, 1.0);
    assert_eq!(res.metrics.claim_recall, 0.0);
    assert_eq!(res.metrics.context_precision, 0.0);
}

// ── errors ───────────────────────────────────────────────────────────────────

#[test]
fn test_empty_response_error() {
    let err = default_checker()
        .check("", "Ground truth here.", &docs(&["Some passage."]))
        .unwrap_err();
    assert!(matches!(err, RagCheckerError::EmptyResponse));
}

#[test]
fn test_empty_ground_truth_error() {
    let err = default_checker()
        .check("A response.", "", &docs(&["Some passage."]))
        .unwrap_err();
    assert!(matches!(err, RagCheckerError::EmptyGroundTruth));
}

#[test]
fn test_whitespace_ground_truth_error() {
    let err = default_checker()
        .check("A response.", "  \n ", &docs(&["Some passage."]))
        .unwrap_err();
    assert!(matches!(err, RagCheckerError::EmptyGroundTruth));
}

#[test]
fn test_response_checked_before_ground_truth() {
    // Both empty: response error takes precedence.
    let err = default_checker().check("", "", &[]).unwrap_err();
    assert!(matches!(err, RagCheckerError::EmptyResponse));
}

#[test]
fn test_error_display_messages() {
    assert_eq!(
        RagCheckerError::EmptyResponse.to_string(),
        "response must not be empty"
    );
    assert_eq!(
        RagCheckerError::EmptyGroundTruth.to_string(),
        "ground truth must not be empty"
    );
}

// ── determinism & value semantics ────────────────────────────────────────────

#[test]
fn test_determinism_same_inputs_same_output() {
    let checker = default_checker();
    let response = "Mercury is a planet. Dragons breathe fire.";
    let gt = "Mercury is a planet.";
    let retrieved = docs(&["Mercury is a planet near the sun."]);
    let first = checker.check(response, gt, &retrieved).unwrap();
    let second = checker.check(response, gt, &retrieved).unwrap();
    assert_eq!(first, second);
}

#[test]
fn test_determinism_repeated_runs_metrics_stable() {
    let checker = default_checker();
    let response = "Saturn has rings. Saturn has a moon named Titan.";
    let gt = "Saturn has rings around it.";
    let retrieved = docs(&["Saturn has rings and a moon named Titan."]);
    let baseline = checker.check(response, gt, &retrieved).unwrap().metrics;
    for _ in 0..5 {
        assert_eq!(
            checker.check(response, gt, &retrieved).unwrap().metrics,
            baseline
        );
    }
}

#[test]
fn test_result_clone_and_metrics_copy() {
    let res: RagCheckResult = default_checker()
        .check(
            "Mercury is a planet.",
            "Mercury is a planet.",
            &docs(&["Mercury is a planet."]),
        )
        .unwrap();
    assert_eq!(res, res.clone());
    let m: RagCheckerMetrics = res.metrics;
    let copy = m;
    assert_eq!(m, copy);
}
