//! Unit tests for the `adaptive_rag` module.
//!
//! Tests are deterministic and follow a strict one-assertion-per-test
//! convention.

#![allow(clippy::float_cmp)]

use super::classifier::ComplexityClassifier;
use super::router::AdaptiveRagRouter;
use super::types::{AdaptiveRagConfig, AdaptiveRagError, QueryComplexity, RetrievalStrategy};

/// Builds a classifier with the default configuration.
fn classifier() -> ComplexityClassifier {
    ComplexityClassifier::new(AdaptiveRagConfig::default())
}

/// Builds a router with the default configuration.
fn router() -> AdaptiveRagRouter {
    AdaptiveRagRouter::new(AdaptiveRagConfig::default())
}

// ── Config defaults & builders ────────────────────────────────────────────────

#[test]
fn default_single_step_threshold() {
    assert_eq!(AdaptiveRagConfig::default().single_step_threshold, 0.25);
}

#[test]
fn default_multi_step_threshold() {
    assert_eq!(AdaptiveRagConfig::default().multi_step_threshold, 0.6);
}

#[test]
fn default_base_top_k() {
    assert_eq!(AdaptiveRagConfig::default().base_top_k, 5);
}

#[test]
fn default_multi_step_max_hops() {
    assert_eq!(AdaptiveRagConfig::default().multi_step_max_hops, 3);
}

#[test]
fn new_equals_default() {
    assert_eq!(AdaptiveRagConfig::new(), AdaptiveRagConfig::default());
}

#[test]
fn builder_single_step_threshold() {
    let c = AdaptiveRagConfig::new().with_single_step_threshold(0.1);
    assert_eq!(c.single_step_threshold, 0.1);
}

#[test]
fn builder_multi_step_threshold() {
    let c = AdaptiveRagConfig::new().with_multi_step_threshold(0.8);
    assert_eq!(c.multi_step_threshold, 0.8);
}

#[test]
fn builder_base_top_k() {
    let c = AdaptiveRagConfig::new().with_base_top_k(12);
    assert_eq!(c.base_top_k, 12);
}

#[test]
fn builder_multi_step_max_hops() {
    let c = AdaptiveRagConfig::new().with_multi_step_max_hops(7);
    assert_eq!(c.multi_step_max_hops, 7);
}

#[test]
fn builders_chain() {
    let c = AdaptiveRagConfig::new()
        .with_single_step_threshold(0.2)
        .with_multi_step_threshold(0.7)
        .with_base_top_k(8)
        .with_multi_step_max_hops(4);
    assert_eq!((c.base_top_k, c.multi_step_max_hops), (8, 4));
}

// ── tier_for_score thresholds ─────────────────────────────────────────────────

#[test]
fn tier_below_single_step_is_straightforward() {
    let c = AdaptiveRagConfig::default();
    assert_eq!(c.tier_for_score(0.24), QueryComplexity::Straightforward);
}

#[test]
fn tier_at_single_step_boundary_is_single_step() {
    let c = AdaptiveRagConfig::default();
    assert_eq!(c.tier_for_score(0.25), QueryComplexity::SingleStep);
}

#[test]
fn tier_just_above_single_step_is_single_step() {
    let c = AdaptiveRagConfig::default();
    assert_eq!(c.tier_for_score(0.26), QueryComplexity::SingleStep);
}

#[test]
fn tier_just_below_multi_step_is_single_step() {
    let c = AdaptiveRagConfig::default();
    assert_eq!(c.tier_for_score(0.59), QueryComplexity::SingleStep);
}

#[test]
fn tier_at_multi_step_boundary_is_multi_step() {
    let c = AdaptiveRagConfig::default();
    assert_eq!(c.tier_for_score(0.6), QueryComplexity::MultiStep);
}

#[test]
fn tier_above_multi_step_is_multi_step() {
    let c = AdaptiveRagConfig::default();
    assert_eq!(c.tier_for_score(0.95), QueryComplexity::MultiStep);
}

#[test]
fn tier_zero_is_straightforward() {
    let c = AdaptiveRagConfig::default();
    assert_eq!(c.tier_for_score(0.0), QueryComplexity::Straightforward);
}

#[test]
fn tier_one_is_multi_step() {
    let c = AdaptiveRagConfig::default();
    assert_eq!(c.tier_for_score(1.0), QueryComplexity::MultiStep);
}

#[test]
fn custom_threshold_changes_tier() {
    let c = AdaptiveRagConfig::new().with_single_step_threshold(0.5);
    assert_eq!(c.tier_for_score(0.3), QueryComplexity::Straightforward);
}

// ── Empty-query errors ────────────────────────────────────────────────────────

#[test]
fn classify_empty_string_errors() {
    let result = classifier().classify("");
    assert!(matches!(result, Err(AdaptiveRagError::EmptyQuery)));
}

#[test]
fn classify_whitespace_only_errors() {
    let result = classifier().classify("   \t\n ");
    assert!(matches!(result, Err(AdaptiveRagError::EmptyQuery)));
}

#[test]
fn route_empty_string_errors() {
    let result = router().route("");
    assert!(matches!(result, Err(AdaptiveRagError::EmptyQuery)));
}

#[test]
fn route_whitespace_only_errors() {
    let result = router().route("  ");
    assert!(matches!(result, Err(AdaptiveRagError::EmptyQuery)));
}

#[test]
fn error_display_message() {
    assert_eq!(
        AdaptiveRagError::EmptyQuery.to_string(),
        "query must not be empty"
    );
}

// ── Trivial query classification ──────────────────────────────────────────────

#[test]
fn trivial_yes_no_not_multi_step() {
    let c = classifier().classify("Is water wet?").unwrap();
    assert_ne!(c.complexity, QueryComplexity::MultiStep);
}

#[test]
fn trivial_yes_no_is_straightforward() {
    let c = classifier().classify("Is water wet?").unwrap();
    assert_eq!(c.complexity, QueryComplexity::Straightforward);
}

#[test]
fn trivial_yes_no_score_low() {
    let c = classifier().classify("Is water wet?").unwrap();
    assert!(c.score < 0.25);
}

#[test]
fn trivial_opener_fires_signal() {
    let c = classifier().classify("Are cats mammals?").unwrap();
    assert!(c.signals.iter().any(|s| s.name == "trivial_opener"));
}

#[test]
fn trivial_score_clamped_non_negative() {
    let c = classifier().classify("Did it?").unwrap();
    assert!(c.score >= 0.0);
}

#[test]
fn short_question_routes_no_retrieval() {
    let plan = router().route("Is it hot?").unwrap();
    assert_eq!(plan.strategy, RetrievalStrategy::NoRetrieval);
}

// ── Factoid (single-step) classification ──────────────────────────────────────

#[test]
fn factoid_who_wrote_is_single_step() {
    let c = classifier().classify("Who wrote Hamlet?").unwrap();
    assert_eq!(c.complexity, QueryComplexity::SingleStep);
}

#[test]
fn factoid_who_wrote_score_in_band() {
    let c = classifier().classify("Who wrote Hamlet?").unwrap();
    assert!(c.score >= 0.25 && c.score < 0.6);
}

#[test]
fn factoid_fires_wh_single_entity_signal() {
    let c = classifier().classify("Who wrote Hamlet?").unwrap();
    assert!(c.signals.iter().any(|s| s.name == "wh_single_entity"));
}

#[test]
fn factoid_what_is_capital_single_step() {
    let c = classifier()
        .classify("What is the capital of France?")
        .unwrap();
    assert_eq!(c.complexity, QueryComplexity::SingleStep);
}

#[test]
fn factoid_when_was_single_step() {
    let c = classifier().classify("When was Tokyo founded?").unwrap();
    assert_eq!(c.complexity, QueryComplexity::SingleStep);
}

#[test]
fn factoid_where_is_single_step() {
    let c = classifier().classify("Where is Everest located?").unwrap();
    assert_eq!(c.complexity, QueryComplexity::SingleStep);
}

#[test]
fn factoid_routes_single_step_strategy() {
    let plan = router().route("Who wrote Hamlet?").unwrap();
    assert_eq!(plan.strategy, RetrievalStrategy::SingleStep);
}

// ── Comparison / multi-step classification ────────────────────────────────────

#[test]
fn comparison_gdp_is_multi_step() {
    let c = classifier()
        .classify("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert_eq!(c.complexity, QueryComplexity::MultiStep);
}

#[test]
fn comparison_gdp_score_high() {
    let c = classifier()
        .classify("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert!(c.score >= 0.6);
}

#[test]
fn comparison_fires_multi_hop_cue() {
    let c = classifier()
        .classify("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert!(
        c.signals
            .iter()
            .any(|s| s.name.starts_with("multi_hop_cue"))
    );
}

#[test]
fn comparison_fires_multi_entity() {
    let c = classifier()
        .classify("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert!(c.signals.iter().any(|s| s.name == "multi_entity"));
}

#[test]
fn comparison_fires_conjunction() {
    let c = classifier()
        .classify("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert!(c.signals.iter().any(|s| s.name == "conjunction"));
}

#[test]
fn difference_between_is_multi_step() {
    let c = classifier()
        .classify("What is the difference between Rust and Go regarding memory?")
        .unwrap();
    assert_eq!(c.complexity, QueryComplexity::MultiStep);
}

#[test]
fn versus_is_multi_step() {
    let c = classifier()
        .classify("How does Python versus Java perform and scale?")
        .unwrap();
    assert_eq!(c.complexity, QueryComplexity::MultiStep);
}

#[test]
fn temporal_chain_before_after_multi_step() {
    let c = classifier()
        .classify("Which treaties came before and after the Congress of Vienna?")
        .unwrap();
    assert_eq!(c.complexity, QueryComplexity::MultiStep);
}

#[test]
fn comparison_routes_multi_step_strategy() {
    let plan = router()
        .route("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert!(matches!(plan.strategy, RetrievalStrategy::MultiStep { .. }));
}

// ── Multi-hop cue effect ──────────────────────────────────────────────────────

#[test]
fn multi_hop_cue_raises_score() {
    let base = classifier()
        .classify("Tell me about the Roman Empire")
        .unwrap()
        .score;
    let cued = classifier()
        .classify("Tell me about the Roman Empire because of trade")
        .unwrap()
        .score;
    assert!(cued > base);
}

#[test]
fn each_distinct_cue_adds_signal() {
    let c = classifier()
        .classify("Discuss the cause and compare the result")
        .unwrap();
    let cue_signals = c
        .signals
        .iter()
        .filter(|s| s.name.starts_with("multi_hop_cue"))
        .count();
    assert_eq!(cue_signals, 2);
}

// ── Multiple question marks effect ────────────────────────────────────────────

#[test]
fn multiple_question_marks_raise_score() {
    let single = classifier()
        .classify("explain photosynthesis briefly")
        .unwrap()
        .score;
    let double = classifier()
        .classify("explain photosynthesis briefly??")
        .unwrap()
        .score;
    assert!(double > single);
}

#[test]
fn multiple_question_marks_fire_signal() {
    let c = classifier().classify("how and why??").unwrap();
    assert!(c.signals.iter().any(|s| s.name == "multi_question"));
}

// ── Entity & comparative signals ──────────────────────────────────────────────

#[test]
fn two_entities_fire_multi_entity_signal() {
    let c = classifier()
        .classify("Discuss Einstein and Newton in physics")
        .unwrap();
    assert!(c.signals.iter().any(|s| s.name == "multi_entity"));
}

#[test]
fn first_token_not_counted_as_entity() {
    // "Paris" is first → excluded; only "France" remains → no multi_entity.
    let c = classifier().classify("Paris is in France").unwrap();
    assert!(!c.signals.iter().any(|s| s.name == "multi_entity"));
}

#[test]
fn superlative_est_fires_comparative_signal() {
    let c = classifier()
        .classify("which is the tallest mountain on earth")
        .unwrap();
    assert!(c.signals.iter().any(|s| s.name == "comparative"));
}

#[test]
fn comparative_word_fires_signal() {
    let c = classifier()
        .classify("which has more rainfall than the other")
        .unwrap();
    assert!(c.signals.iter().any(|s| s.name == "comparative"));
}

#[test]
fn conjunction_fires_for_or() {
    let c = classifier()
        .classify("explain photosynthesis or respiration in plants")
        .unwrap();
    assert!(c.signals.iter().any(|s| s.name == "conjunction"));
}

#[test]
fn long_query_fires_long_signal() {
    let c = classifier()
        .classify("explain the historical economic social political cultural and military effects of the industrial revolution")
        .unwrap();
    assert!(c.signals.iter().any(|s| s.name == "long_query"));
}

// ── Signals list population ───────────────────────────────────────────────────

#[test]
fn non_trivial_query_has_signals() {
    let c = classifier().classify("Compare apples and oranges").unwrap();
    assert!(!c.signals.is_empty());
}

#[test]
fn signal_weight_recorded() {
    let c = classifier().classify("Who wrote Hamlet?").unwrap();
    let signal = c
        .signals
        .iter()
        .find(|s| s.name == "wh_single_entity")
        .expect("signal present");
    assert_eq!(signal.weight, 0.40);
}

// ── Routing plan fields per tier ──────────────────────────────────────────────

#[test]
fn no_retrieval_top_k_is_zero() {
    let plan = router().route("Is it cold?").unwrap();
    assert_eq!(plan.recommended_top_k, 0);
}

#[test]
fn single_step_top_k_is_base() {
    let plan = router().route("Who wrote Hamlet?").unwrap();
    assert_eq!(plan.recommended_top_k, 5);
}

#[test]
fn multi_step_top_k_is_double_base() {
    let plan = router()
        .route("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert_eq!(plan.recommended_top_k, 10);
}

#[test]
fn multi_step_max_hops_from_config() {
    let plan = router()
        .route("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert_eq!(plan.strategy, RetrievalStrategy::MultiStep { max_hops: 3 });
}

#[test]
fn custom_base_top_k_propagates_to_plan() {
    let r = AdaptiveRagRouter::new(AdaptiveRagConfig::new().with_base_top_k(8));
    let plan = r.route("Who wrote Hamlet?").unwrap();
    assert_eq!(plan.recommended_top_k, 8);
}

#[test]
fn custom_max_hops_propagates_to_plan() {
    let r = AdaptiveRagRouter::new(AdaptiveRagConfig::new().with_multi_step_max_hops(6));
    let plan = r
        .route("Compare the GDP of France and Germany after 2010")
        .unwrap();
    assert_eq!(plan.strategy, RetrievalStrategy::MultiStep { max_hops: 6 });
}

#[test]
fn plan_complexity_matches_classification() {
    let q = "Who wrote Hamlet?";
    let plan = router().route(q).unwrap();
    let classification = classifier().classify(q).unwrap();
    assert_eq!(plan.complexity, classification.complexity);
}

// ── Strategy & complexity helpers ─────────────────────────────────────────────

#[test]
fn complexity_as_str_round_trip() {
    assert_eq!(QueryComplexity::MultiStep.as_str(), "multi_step");
}

#[test]
fn strategy_as_str_for_multi_step() {
    assert_eq!(
        RetrievalStrategy::MultiStep { max_hops: 3 }.as_str(),
        "multi_step"
    );
}

#[test]
fn complexity_display() {
    assert_eq!(format!("{}", QueryComplexity::SingleStep), "single_step");
}

#[test]
fn strategy_display_no_retrieval() {
    assert_eq!(
        format!("{}", RetrievalStrategy::NoRetrieval),
        "no_retrieval"
    );
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn classify_is_deterministic() {
    let q = "Compare the GDP of France and Germany after 2010";
    let a = classifier().classify(q).unwrap();
    let b = classifier().classify(q).unwrap();
    assert_eq!(a, b);
}

#[test]
fn route_is_deterministic() {
    let q = "Who wrote Hamlet?";
    let a = router().route(q).unwrap();
    let b = router().route(q).unwrap();
    assert_eq!(a, b);
}

#[test]
fn classify_tier_matches_full_classification() {
    let q = "Compare apples and oranges after lunch";
    let tier = classifier().classify_tier(q).unwrap();
    let full = classifier().classify(q).unwrap().complexity;
    assert_eq!(tier, full);
}
