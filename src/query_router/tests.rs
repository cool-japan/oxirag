//! Tests for the `query_router` module.
//!
//! Groups are separated by `// ── banner ──` markers and cover enums, config,
//! `IntentScores`, the heuristic classifier, the mock classifier, and the
//! full routing pipeline.

use super::classifier::{HeuristicIntentClassifier, IntentClassifier, MockIntentClassifier};
use super::router::QueryRouter;
use super::types::{
    IntentScores, QueryIntent, QueryRouterError, RouterConfig, RoutingDecision, RoutingStrategy,
};

// ── Helper ────────────────────────────────────────────────────────────────────

fn make_router() -> QueryRouter<MockIntentClassifier> {
    QueryRouter::new(
        MockIntentClassifier::new(QueryIntent::Factual, 0.8),
        RouterConfig::default(),
    )
}

// ── Enum tests ────────────────────────────────────────────────────────────────

#[test]
fn test_query_intent_as_str_factual() {
    assert_eq!(QueryIntent::Factual.as_str(), "factual");
}

#[test]
fn test_query_intent_as_str_all_variants() {
    let pairs = [
        (QueryIntent::Factual, "factual"),
        (QueryIntent::Definitional, "definitional"),
        (QueryIntent::Comparative, "comparative"),
        (QueryIntent::Navigational, "navigational"),
        (QueryIntent::MultiHop, "multi_hop"),
        (QueryIntent::Conversational, "conversational"),
        (QueryIntent::Exploratory, "exploratory"),
        (QueryIntent::Temporal, "temporal"),
        (QueryIntent::Aggregation, "aggregation"),
        (QueryIntent::Unknown, "unknown"),
    ];
    for (intent, expected) in pairs {
        assert_eq!(intent.as_str(), expected, "mismatch for {intent:?}");
    }
}

#[test]
fn test_routing_strategy_as_str_all_variants() {
    let pairs = [
        (RoutingStrategy::VectorSearch, "vector_search"),
        (RoutingStrategy::HybridSearch, "hybrid_search"),
        (RoutingStrategy::GraphSearch, "graph_search"),
        (RoutingStrategy::MultiHop, "multi_hop"),
        (RoutingStrategy::Conversational, "conversational"),
        (RoutingStrategy::DirectAnswer, "direct_answer"),
    ];
    for (strategy, expected) in pairs {
        assert_eq!(strategy.as_str(), expected, "mismatch for {strategy:?}");
    }
}

#[test]
fn test_query_intent_display() {
    assert_eq!(format!("{}", QueryIntent::Comparative), "comparative");
    assert_eq!(format!("{}", QueryIntent::Unknown), "unknown");
}

#[test]
fn test_routing_strategy_display() {
    assert_eq!(
        format!("{}", RoutingStrategy::HybridSearch),
        "hybrid_search"
    );
    assert_eq!(
        format!("{}", RoutingStrategy::DirectAnswer),
        "direct_answer"
    );
}

#[test]
fn test_query_intent_eq_and_copy() {
    let a = QueryIntent::Factual;
    let b = a; // Copy
    assert_eq!(a, b);
}

#[test]
fn test_routing_strategy_eq_and_copy() {
    let a = RoutingStrategy::GraphSearch;
    let b = a; // Copy
    assert_eq!(a, b);
}

#[test]
fn test_query_intent_ne() {
    assert_ne!(QueryIntent::Factual, QueryIntent::Comparative);
}

#[test]
fn test_routing_strategy_ne() {
    assert_ne!(RoutingStrategy::VectorSearch, RoutingStrategy::GraphSearch);
}

// ── RouterConfig defaults ──────────────────────────────────────────────────────

#[test]
fn test_router_config_default_factual_maps_to_vector_search() {
    let cfg = RouterConfig::default();
    assert_eq!(
        cfg.strategy_for(QueryIntent::Factual),
        RoutingStrategy::VectorSearch
    );
}

#[test]
fn test_router_config_default_comparative_maps_to_hybrid_search() {
    let cfg = RouterConfig::default();
    assert_eq!(
        cfg.strategy_for(QueryIntent::Comparative),
        RoutingStrategy::HybridSearch
    );
}

#[test]
fn test_router_config_default_navigational_maps_to_direct_answer() {
    let cfg = RouterConfig::default();
    assert_eq!(
        cfg.strategy_for(QueryIntent::Navigational),
        RoutingStrategy::DirectAnswer
    );
}

#[test]
fn test_router_config_default_multi_hop_maps_to_multi_hop() {
    let cfg = RouterConfig::default();
    assert_eq!(
        cfg.strategy_for(QueryIntent::MultiHop),
        RoutingStrategy::MultiHop
    );
}

#[test]
fn test_router_config_default_aggregation_maps_to_graph_search() {
    let cfg = RouterConfig::default();
    assert_eq!(
        cfg.strategy_for(QueryIntent::Aggregation),
        RoutingStrategy::GraphSearch
    );
}

#[test]
fn test_router_config_with_route_builder() {
    let cfg =
        RouterConfig::default().with_route(QueryIntent::Factual, RoutingStrategy::HybridSearch);
    assert_eq!(
        cfg.strategy_for(QueryIntent::Factual),
        RoutingStrategy::HybridSearch
    );
}

#[test]
fn test_router_config_with_fallbacks_builder() {
    let cfg = RouterConfig::default().with_fallbacks(
        RoutingStrategy::VectorSearch,
        vec![RoutingStrategy::HybridSearch],
    );
    let fallbacks = cfg
        .fallback_table
        .get(&RoutingStrategy::VectorSearch)
        .cloned()
        .unwrap_or_default();
    assert!(fallbacks.contains(&RoutingStrategy::HybridSearch));
}

#[test]
fn test_router_config_strategy_for_unknown_falls_back_to_default() {
    // Remove Unknown from the table and verify fallback to default_strategy
    let mut cfg = RouterConfig::default();
    cfg.intent_strategy_table.remove(&QueryIntent::Unknown);
    assert_eq!(cfg.strategy_for(QueryIntent::Unknown), cfg.default_strategy);
}

// ── IntentScores ───────────────────────────────────────────────────────────────

#[test]
fn test_intent_scores_sorts_descending() {
    let scores = IntentScores::new(vec![
        (QueryIntent::Factual, 0.3),
        (QueryIntent::Comparative, 0.9),
        (QueryIntent::Unknown, 0.05),
    ]);
    let top = scores.top();
    assert_eq!(top.0, QueryIntent::Comparative);
    assert!((top.1 - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_intent_scores_top_empty_guard() {
    let scores = IntentScores::new(vec![]);
    let (intent, conf) = scores.top();
    assert_eq!(intent, QueryIntent::Unknown);
    assert!((conf - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_intent_scores_confidence_of_present() {
    let scores = IntentScores::new(vec![
        (QueryIntent::Temporal, 0.75),
        (QueryIntent::Unknown, 0.05),
    ]);
    assert!((scores.confidence_of(QueryIntent::Temporal) - 0.75).abs() < f32::EPSILON);
}

#[test]
fn test_intent_scores_confidence_of_absent() {
    let scores = IntentScores::new(vec![(QueryIntent::Factual, 0.8)]);
    assert!((scores.confidence_of(QueryIntent::Temporal) - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_intent_scores_is_empty() {
    let empty = IntentScores::new(vec![]);
    assert!(empty.is_empty());
    let non_empty = IntentScores::new(vec![(QueryIntent::Unknown, 0.05)]);
    assert!(!non_empty.is_empty());
}

// ── HeuristicIntentClassifier signals ─────────────────────────────────────────

#[tokio::test]
async fn test_heuristic_definitional_signal() {
    let clf = HeuristicIntentClassifier::new();
    let scores = clf.classify("what is Rust?").await.unwrap();
    let (intent, _) = scores.top();
    assert_eq!(intent, QueryIntent::Definitional);
}

#[tokio::test]
async fn test_heuristic_comparative_signal() {
    let clf = HeuristicIntentClassifier::new();
    let scores = clf.classify("Rust vs Go performance").await.unwrap();
    let top_intent = scores.top().0;
    assert_eq!(top_intent, QueryIntent::Comparative);
}

#[tokio::test]
async fn test_heuristic_temporal_signal() {
    let clf = HeuristicIntentClassifier::new();
    let scores = clf
        .classify("when was the Eiffel Tower built?")
        .await
        .unwrap();
    let top_intent = scores.top().0;
    assert_eq!(top_intent, QueryIntent::Temporal);
}

#[tokio::test]
async fn test_heuristic_temporal_year_detection() {
    let clf = HeuristicIntentClassifier::new();
    // Contains a 4-digit year — should boost Temporal
    let scores = clf.classify("events in 1989 in Berlin").await.unwrap();
    assert!(scores.confidence_of(QueryIntent::Temporal) > 0.0);
}

#[tokio::test]
async fn test_heuristic_aggregation_signal() {
    let clf = HeuristicIntentClassifier::new();
    let scores = clf
        .classify("how many planets are in the solar system?")
        .await
        .unwrap();
    let top_intent = scores.top().0;
    assert_eq!(top_intent, QueryIntent::Aggregation);
}

#[tokio::test]
async fn test_heuristic_multi_hop_double_question() {
    let clf = HeuristicIntentClassifier::new();
    // Two question marks → multi-hop bonus
    let scores = clf
        .classify("who invented the telephone and then who commercialised it?")
        .await
        .unwrap();
    assert!(scores.confidence_of(QueryIntent::MultiHop) > 0.0);
}

#[tokio::test]
async fn test_heuristic_navigational_signal() {
    let clf = HeuristicIntentClassifier::new();
    let scores = clf.classify("go to the settings page").await.unwrap();
    let top_intent = scores.top().0;
    assert_eq!(top_intent, QueryIntent::Navigational);
}

#[tokio::test]
async fn test_heuristic_exploratory_signal() {
    let clf = HeuristicIntentClassifier::new();
    let scores = clf
        .classify("tell me about quantum computing")
        .await
        .unwrap();
    let top_intent = scores.top().0;
    assert_eq!(top_intent, QueryIntent::Exploratory);
}

#[tokio::test]
async fn test_heuristic_short_query_conversational() {
    let clf = HeuristicIntentClassifier::new();
    // Very short query — should be Conversational
    let scores = clf.classify("more details?").await.unwrap();
    let top_intent = scores.top().0;
    assert_eq!(top_intent, QueryIntent::Conversational);
}

#[tokio::test]
async fn test_heuristic_pronoun_conversational() {
    let clf = HeuristicIntentClassifier::new();
    // Starts with "they" — conversational pronoun heuristic
    let scores = clf
        .classify("they also support async/await in the latest version")
        .await
        .unwrap();
    assert!(scores.confidence_of(QueryIntent::Conversational) > 0.0);
}

#[tokio::test]
async fn test_heuristic_factual_wh_question() {
    let clf = HeuristicIntentClassifier::new();
    // "who" wh-question with no other strong signals
    let scores = clf
        .classify("who wrote the Rust programming book?")
        .await
        .unwrap();
    assert!(scores.confidence_of(QueryIntent::Factual) > 0.0);
}

#[tokio::test]
async fn test_heuristic_empty_query_returns_error() {
    let clf = HeuristicIntentClassifier::new();
    let result = clf.classify("   ").await;
    assert!(matches!(result, Err(QueryRouterError::EmptyQuery)));
}

#[tokio::test]
async fn test_heuristic_no_signal_returns_unknown() {
    let clf = HeuristicIntentClassifier::new();
    // A long neutral sentence with no strong keywords
    let scores = clf
        .classify("the quick brown fox jumped over the lazy dog in the park near the river")
        .await
        .unwrap();
    // Should not be empty; Unknown floor should always be present
    assert!(!scores.is_empty());
    assert!(scores.confidence_of(QueryIntent::Unknown) > 0.0);
}

#[tokio::test]
async fn test_heuristic_classify_top_returns_best() {
    let clf = HeuristicIntentClassifier::new();
    let (intent, conf) = clf.classify_top("compare Python vs Rust").await.unwrap();
    assert_eq!(intent, QueryIntent::Comparative);
    assert!(conf > 0.0);
}

#[tokio::test]
async fn test_heuristic_multi_intent_ranking_comparative_beats_factual() {
    let clf = HeuristicIntentClassifier::new();
    // "compare" + "vs" → strong Comparative; "what" adds weak Factual
    let scores = clf
        .classify("compare what Python vs Rust offer for systems programming")
        .await
        .unwrap();
    let comparative_score = scores.confidence_of(QueryIntent::Comparative);
    // Comparative signal should dominate
    assert!(comparative_score > 0.0);
    assert_eq!(scores.top().0, QueryIntent::Comparative);
}

// ── MockIntentClassifier ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_mock_classifier_single_intent() {
    let clf = MockIntentClassifier::new(QueryIntent::Factual, 0.9);
    let (intent, conf) = clf.classify_top("anything").await.unwrap();
    assert_eq!(intent, QueryIntent::Factual);
    assert!((conf - 0.9).abs() < f32::EPSILON);
}

#[tokio::test]
async fn test_mock_classifier_with_scores() {
    let scripted = IntentScores::new(vec![
        (QueryIntent::Aggregation, 0.85),
        (QueryIntent::Unknown, 0.05),
    ]);
    let clf = MockIntentClassifier::with_scores(scripted);
    let scores = clf.classify("ignored").await.unwrap();
    assert_eq!(scores.top().0, QueryIntent::Aggregation);
}

#[tokio::test]
async fn test_mock_classifier_deterministic() {
    let clf = MockIntentClassifier::new(QueryIntent::Temporal, 0.7);
    let r1 = clf.classify("first call").await.unwrap();
    let r2 = clf.classify("second call").await.unwrap();
    assert_eq!(r1.top(), r2.top());
}

// ── QueryRouter.route ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_router_strategy_mapping_via_mock() {
    let router = make_router(); // Factual → VectorSearch
    let decision = router.route("some query").await.unwrap();
    assert_eq!(decision.strategy, RoutingStrategy::VectorSearch);
    assert_eq!(decision.intent, QueryIntent::Factual);
}

#[tokio::test]
async fn test_router_low_confidence_uses_default() {
    let clf = MockIntentClassifier::new(QueryIntent::MultiHop, 0.1); // 0.1 < 0.3 threshold
    let router = QueryRouter::new(clf, RouterConfig::default());
    let decision = router.route("some query").await.unwrap();
    // Low confidence → should fall back to default_strategy (VectorSearch)
    assert_eq!(decision.strategy, RouterConfig::default().default_strategy);
    assert!(decision.reasoning.contains("low confidence"));
}

#[tokio::test]
async fn test_router_empty_query_error() {
    let router = make_router();
    let result = router.route("   ").await;
    assert!(matches!(result, Err(QueryRouterError::EmptyQuery)));
}

#[tokio::test]
async fn test_router_fallback_chain_not_empty() {
    let router = make_router();
    let decision = router.route("query").await.unwrap();
    assert!(!decision.fallback_strategies.is_empty());
}

#[tokio::test]
async fn test_router_top_k_override_applied() {
    let clf = MockIntentClassifier::new(QueryIntent::Aggregation, 0.9);
    let router = QueryRouter::new(clf, RouterConfig::default());
    let decision = router.route("list all items").await.unwrap();
    assert_eq!(decision.top_k_override, Some(50)); // Aggregation override
}

#[tokio::test]
async fn test_router_reasoning_contains_intent_name() {
    let router = make_router();
    let decision = router.route("what is the speed of light?").await.unwrap();
    assert!(decision.reasoning.contains("factual"));
}

#[test]
fn test_router_decide_sync() {
    let router = make_router();
    let scores = IntentScores::new(vec![
        (QueryIntent::Exploratory, 0.8),
        (QueryIntent::Unknown, 0.05),
    ]);
    let decision = router.decide("tell me about AI", &scores);
    assert_eq!(decision.strategy, RoutingStrategy::HybridSearch);
    assert_eq!(decision.intent, QueryIntent::Exploratory);
}

#[tokio::test]
async fn test_router_comparative_e2e_with_heuristic() {
    let clf = HeuristicIntentClassifier::new();
    let router = QueryRouter::new(clf, RouterConfig::default());
    let decision = router
        .route("compare Rust vs Go performance")
        .await
        .unwrap();
    assert_eq!(decision.strategy, RoutingStrategy::HybridSearch);
    assert_eq!(decision.intent, QueryIntent::Comparative);
}

#[tokio::test]
async fn test_router_unknown_intent_maps_to_vector_search() {
    // Unknown should map to VectorSearch per the default table
    let clf = MockIntentClassifier::new(QueryIntent::Unknown, 0.9);
    let router = QueryRouter::new(clf, RouterConfig::default());
    let decision = router.route("xyzzy frobnitz").await.unwrap();
    assert_eq!(decision.strategy, RoutingStrategy::VectorSearch);
}

// Additional: RoutingDecision is well-formed
#[tokio::test]
async fn test_routing_decision_fields_consistent() {
    let clf = MockIntentClassifier::new(QueryIntent::MultiHop, 0.85);
    let router = QueryRouter::new(clf, RouterConfig::default());
    let RoutingDecision {
        intent,
        intent_confidence,
        strategy,
        fallback_strategies,
        top_k_override,
        reasoning,
    } = router.route("multi hop query").await.unwrap();

    assert_eq!(intent, QueryIntent::MultiHop);
    assert!((intent_confidence - 0.85).abs() < f32::EPSILON);
    assert_eq!(strategy, RoutingStrategy::MultiHop);
    assert!(!fallback_strategies.is_empty());
    assert_eq!(top_k_override, Some(20));
    assert!(!reasoning.is_empty());
}
