//! Tests for the `semantic_router` module.

use super::router::SemanticRouter;
use super::types::{
    RouterError, RouterExample, RoutingDecision, RoutingTarget, SemanticRoutingConfig,
};

// ── RoutingTarget ─────────────────────────────────────────────────────────────

#[test]
fn test_routing_target_as_str() {
    assert_eq!(RoutingTarget::VectorSearch.as_str(), "vector_search");
    assert_eq!(RoutingTarget::GraphSearch.as_str(), "graph_search");
    assert_eq!(RoutingTarget::HybridSearch.as_str(), "hybrid_search");
    assert_eq!(RoutingTarget::MultiHop.as_str(), "multi_hop");
    assert_eq!(RoutingTarget::AgenticSearch.as_str(), "agentic_search");
    assert_eq!(RoutingTarget::DirectAnswer.as_str(), "direct_answer");
}

#[test]
fn test_routing_target_from_str() {
    assert_eq!(
        RoutingTarget::from_str("vector_search"),
        Some(RoutingTarget::VectorSearch)
    );
    assert_eq!(
        RoutingTarget::from_str("Vector"),
        Some(RoutingTarget::VectorSearch)
    );
    assert_eq!(
        RoutingTarget::from_str("GRAPH_SEARCH"),
        Some(RoutingTarget::GraphSearch)
    );
    assert_eq!(
        RoutingTarget::from_str("graph"),
        Some(RoutingTarget::GraphSearch)
    );
    assert_eq!(
        RoutingTarget::from_str("hybrid"),
        Some(RoutingTarget::HybridSearch)
    );
    assert_eq!(
        RoutingTarget::from_str("multi_hop"),
        Some(RoutingTarget::MultiHop)
    );
    assert_eq!(
        RoutingTarget::from_str("multi-hop"),
        Some(RoutingTarget::MultiHop)
    );
    assert_eq!(
        RoutingTarget::from_str("agentic"),
        Some(RoutingTarget::AgenticSearch)
    );
    assert_eq!(
        RoutingTarget::from_str("direct"),
        Some(RoutingTarget::DirectAnswer)
    );
    assert_eq!(RoutingTarget::from_str("unknown"), None);
}

#[test]
fn test_routing_target_default() {
    let target = RoutingTarget::default();
    assert_eq!(target, RoutingTarget::VectorSearch);
}

// ── RouterExample ─────────────────────────────────────────────────────────────

#[test]
fn test_router_example_new() {
    let ex = RouterExample::new("find relevant docs", RoutingTarget::VectorSearch);
    assert_eq!(ex.query, "find relevant docs");
    assert_eq!(ex.target, RoutingTarget::VectorSearch);
}

// ── RoutingDecision ───────────────────────────────────────────────────────────

#[test]
fn test_decision_is_confident() {
    let decision = RoutingDecision {
        target: RoutingTarget::VectorSearch,
        confidence: 0.8,
        reasoning: "test".to_string(),
    };
    assert!(decision.is_confident(0.5));
    assert!(decision.is_confident(0.8));
    assert!(!decision.is_confident(0.9));
}

// ── SemanticRoutingConfig ─────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let config = SemanticRoutingConfig::default();
    assert!((config.threshold - 0.5).abs() < f32::EPSILON);
    assert_eq!(config.fallback, RoutingTarget::VectorSearch);
}

#[test]
fn test_config_builders() {
    let config = SemanticRoutingConfig::default()
        .with_threshold(0.7)
        .with_fallback(RoutingTarget::HybridSearch);
    assert!((config.threshold - 0.7).abs() < f32::EPSILON);
    assert_eq!(config.fallback, RoutingTarget::HybridSearch);
}

// ── SemanticRouter error cases ────────────────────────────────────────────────

#[test]
fn test_router_no_examples_error() {
    let router = SemanticRouter::default();
    let err = router.route("what is Rust?");
    assert!(matches!(err, Err(RouterError::NoExamples)));
}

#[test]
fn test_router_empty_query_error() {
    let router = SemanticRouter::new(
        vec![RouterExample::new(
            "find documents",
            RoutingTarget::VectorSearch,
        )],
        SemanticRoutingConfig::default(),
    );
    let err = router.route("   ");
    assert!(matches!(err, Err(RouterError::EmptyQuery)));
}

// ── SemanticRouter routing ────────────────────────────────────────────────────

#[test]
fn test_router_single_example_exact_match() {
    let examples = vec![RouterExample::new(
        "find relevant documents",
        RoutingTarget::VectorSearch,
    )];
    let router = SemanticRouter::new(
        examples,
        SemanticRoutingConfig::default().with_threshold(0.0),
    );
    let decision = router
        .route("find relevant documents")
        .expect("route should succeed");
    assert_eq!(decision.target, RoutingTarget::VectorSearch);
    // Exact same text should have similarity ≈ 1.0
    assert!(decision.confidence > 0.9);
}

#[test]
fn test_router_best_match_wins() {
    let examples = vec![
        RouterExample::new("traverse the knowledge graph", RoutingTarget::GraphSearch),
        RouterExample::new(
            "find similar documents using vectors",
            RoutingTarget::VectorSearch,
        ),
    ];
    let router = SemanticRouter::new(
        examples,
        SemanticRoutingConfig::default().with_threshold(0.0),
    );
    let decision = router
        .route("traverse the knowledge graph nodes")
        .expect("route should succeed");
    // Graph example should win over vector example
    assert_eq!(decision.target, RoutingTarget::GraphSearch);
}

#[test]
fn test_router_below_threshold_uses_fallback() {
    let examples = vec![RouterExample::new(
        "zzz completely unrelated tokens qqq",
        RoutingTarget::GraphSearch,
    )];
    let config = SemanticRoutingConfig::default()
        .with_threshold(0.99) // impossibly high threshold
        .with_fallback(RoutingTarget::HybridSearch);
    let router = SemanticRouter::new(examples, config);
    let decision = router
        .route("what is the weather today")
        .expect("route should succeed");
    assert_eq!(decision.target, RoutingTarget::HybridSearch);
}

#[test]
fn test_router_fallback_never_errors() {
    let router = SemanticRouter::default(); // no examples
    let decision = router.route_with_fallback("any query at all");
    assert_eq!(decision.target, RoutingTarget::VectorSearch);
    assert!((decision.confidence).abs() < f32::EPSILON);
}

#[test]
fn test_router_add_example() {
    let mut router = SemanticRouter::default();
    assert!(router.examples.is_empty());
    router.add_example(RouterExample::new("find docs", RoutingTarget::VectorSearch));
    assert_eq!(router.examples.len(), 1);
}

#[test]
fn test_router_vector_search_route() {
    let examples = vec![
        RouterExample::new(
            "search for documents about machine learning",
            RoutingTarget::VectorSearch,
        ),
        RouterExample::new(
            "find similar items using embeddings",
            RoutingTarget::VectorSearch,
        ),
    ];
    let router = SemanticRouter::new(
        examples,
        SemanticRoutingConfig::default().with_threshold(0.0),
    );
    let decision = router
        .route("search documents about neural networks")
        .expect("route should succeed");
    assert_eq!(decision.target, RoutingTarget::VectorSearch);
}

#[test]
fn test_router_graph_search_route() {
    let examples = vec![
        RouterExample::new(
            "traverse knowledge graph entities",
            RoutingTarget::GraphSearch,
        ),
        RouterExample::new(
            "find related nodes in the graph",
            RoutingTarget::GraphSearch,
        ),
        RouterExample::new("vector similarity search", RoutingTarget::VectorSearch),
    ];
    let router = SemanticRouter::new(
        examples,
        SemanticRoutingConfig::default().with_threshold(0.0),
    );
    let decision = router
        .route("traverse graph entities nodes")
        .expect("route should succeed");
    assert_eq!(decision.target, RoutingTarget::GraphSearch);
}

#[test]
fn test_router_hybrid_route() {
    let examples = vec![
        RouterExample::new(
            "hybrid search combining vector and graph",
            RoutingTarget::HybridSearch,
        ),
        RouterExample::new(
            "use both embeddings and graph traversal",
            RoutingTarget::HybridSearch,
        ),
        RouterExample::new("pure vector search only", RoutingTarget::VectorSearch),
    ];
    let router = SemanticRouter::new(
        examples,
        SemanticRoutingConfig::default().with_threshold(0.0),
    );
    let decision = router
        .route("hybrid search combining vector and graph")
        .expect("route should succeed");
    assert_eq!(decision.target, RoutingTarget::HybridSearch);
}

#[test]
fn test_router_confidence_range_01() {
    let examples = vec![
        RouterExample::new("find documents", RoutingTarget::VectorSearch),
        RouterExample::new("graph traversal", RoutingTarget::GraphSearch),
    ];
    let router = SemanticRouter::new(
        examples,
        SemanticRoutingConfig::default().with_threshold(0.0),
    );
    let decision = router
        .route("find some documents")
        .expect("route should succeed");
    assert!((0.0..=1.0).contains(&decision.confidence));
}

// ── Error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    assert_eq!(
        RouterError::NoExamples.to_string(),
        "No routing examples provided"
    );
    assert_eq!(
        RouterError::EmptyQuery.to_string(),
        "Query must not be empty"
    );
}
