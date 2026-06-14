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
    clippy::too_many_lines
)]
//! Unit tests for the `self_route` module.
//!
//! Tests are deterministic and follow a strict one-assertion-per-test
//! convention. Where exact answerability values are required, a configuration
//! with `coverage_weight = 1.0` is used so that the blend reduces to query
//! coverage alone, making the score exactly controllable.

use crate::types::Document;

use super::router::SelfRouter;
use super::types::{RouteAssessment, RouteDecision, SelfRouteConfig, SelfRouteError};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Builds a router with the default configuration.
fn router() -> SelfRouter {
    SelfRouter::new(SelfRouteConfig::default())
}

/// Builds a document from raw content.
fn doc(content: &str) -> Document {
    Document::new(content)
}

/// A four-token query: `alpha bravo charlie delta`.
const Q4: &str = "alpha bravo charlie delta";

// ── RouteDecision ─────────────────────────────────────────────────────────────

#[test]
fn decision_rag_as_str() {
    assert_eq!(RouteDecision::Rag.as_str(), "rag");
}

#[test]
fn decision_long_context_as_str() {
    assert_eq!(RouteDecision::LongContext.as_str(), "long_context");
}

#[test]
fn decision_rag_display() {
    assert_eq!(RouteDecision::Rag.to_string(), "rag");
}

#[test]
fn decision_long_context_display() {
    assert_eq!(RouteDecision::LongContext.to_string(), "long_context");
}

#[test]
fn decision_rag_ne_long_context() {
    assert_ne!(RouteDecision::Rag, RouteDecision::LongContext);
}

#[test]
fn decision_is_copy() {
    let d = RouteDecision::Rag;
    let copy = d;
    assert_eq!(copy, d);
    assert_eq!(d, RouteDecision::Rag);
}

// ── Config defaults & builders ────────────────────────────────────────────────

#[test]
fn default_answerability_threshold() {
    assert_eq!(SelfRouteConfig::default().answerability_threshold, 0.4);
}

#[test]
fn default_coverage_weight() {
    assert_eq!(SelfRouteConfig::default().coverage_weight, 0.5);
}

#[test]
fn new_equals_default() {
    assert_eq!(SelfRouteConfig::new(), SelfRouteConfig::default());
}

#[test]
fn builder_answerability_threshold() {
    let c = SelfRouteConfig::new().with_answerability_threshold(0.7);
    assert_eq!(c.answerability_threshold, 0.7);
}

#[test]
fn builder_coverage_weight() {
    let c = SelfRouteConfig::new().with_coverage_weight(0.25);
    assert_eq!(c.coverage_weight, 0.25);
}

#[test]
fn builder_chained_threshold() {
    let c = SelfRouteConfig::new()
        .with_coverage_weight(0.3)
        .with_answerability_threshold(0.9);
    assert_eq!(c.answerability_threshold, 0.9);
}

#[test]
fn builder_chained_weight() {
    let c = SelfRouteConfig::new()
        .with_answerability_threshold(0.9)
        .with_coverage_weight(0.3);
    assert_eq!(c.coverage_weight, 0.3);
}

#[test]
fn config_is_clone() {
    let c = SelfRouteConfig::new().with_coverage_weight(0.1);
    let cloned = c.clone();
    assert_eq!(c, cloned);
}

// ── query_coverage ────────────────────────────────────────────────────────────

#[test]
fn coverage_full() {
    let docs = vec![doc("alpha bravo charlie delta")];
    assert_eq!(router().query_coverage(Q4, &docs), 1.0);
}

#[test]
fn coverage_half() {
    let docs = vec![doc("alpha bravo nothing else here")];
    assert_eq!(router().query_coverage(Q4, &docs), 0.5);
}

#[test]
fn coverage_quarter() {
    let docs = vec![doc("alpha and some unrelated words")];
    assert_eq!(router().query_coverage(Q4, &docs), 0.25);
}

#[test]
fn coverage_zero_when_no_overlap() {
    let docs = vec![doc("entirely unrelated content words")];
    assert_eq!(router().query_coverage(Q4, &docs), 0.0);
}

#[test]
fn coverage_zero_when_docs_empty() {
    let docs: Vec<Document> = Vec::new();
    assert_eq!(router().query_coverage(Q4, &docs), 0.0);
}

#[test]
fn coverage_zero_when_query_has_no_content_tokens() {
    let docs = vec![doc("alpha bravo charlie delta")];
    assert_eq!(router().query_coverage("a , . !", &docs), 0.0);
}

#[test]
fn coverage_union_across_multiple_docs() {
    let docs = vec![doc("alpha only"), doc("bravo charlie here"), doc("delta")];
    assert_eq!(router().query_coverage(Q4, &docs), 1.0);
}

#[test]
fn coverage_is_case_insensitive() {
    let docs = vec![doc("ALPHA BRAVO CHARLIE DELTA")];
    assert_eq!(router().query_coverage(Q4, &docs), 1.0);
}

#[test]
fn coverage_distinct_tokens_only() {
    // Repeated query terms collapse to distinct token set of size 1.
    let docs = vec![doc("alpha")];
    assert_eq!(router().query_coverage("alpha alpha alpha", &docs), 1.0);
}

#[test]
fn coverage_three_quarters() {
    let docs = vec![doc("alpha bravo charlie unrelated")];
    assert_eq!(router().query_coverage(Q4, &docs), 0.75);
}

// ── assess: high coverage ⇒ Rag ───────────────────────────────────────────────

#[test]
fn assess_high_coverage_routes_rag() {
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.decision, RouteDecision::Rag);
}

#[test]
fn assess_realistic_answerable_routes_rag() {
    let docs = vec![doc("The Eiffel Tower is located in Paris, France.")];
    let a = router()
        .assess("Where is the Eiffel Tower located?", &docs)
        .unwrap();
    assert_eq!(a.decision, RouteDecision::Rag);
}

#[test]
fn assess_high_coverage_answerability_value() {
    // weight 1.0 ⇒ answerability == coverage == 1.0.
    let cfg = SelfRouteConfig::new().with_coverage_weight(1.0);
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    assert_eq!(a.answerability, 1.0);
}

#[test]
fn assess_records_query_coverage() {
    let docs = vec![doc("alpha bravo nothing else")];
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.query_coverage, 0.5);
}

#[test]
fn assess_top_relevance_in_unit_range() {
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = router().assess(Q4, &docs).unwrap();
    assert!(a.top_relevance >= 0.0 && a.top_relevance <= 1.0);
}

#[test]
fn assess_top_relevance_perfect_match_is_one() {
    // Doc tokens identical to query tokens ⇒ Jaccard 1.0.
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.top_relevance, 1.0);
}

#[test]
fn assess_top_relevance_takes_best_doc() {
    // Second doc is a perfect match; top_relevance must reflect it.
    let docs = vec![
        doc("totally unrelated material text"),
        doc("alpha bravo charlie delta"),
    ];
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.top_relevance, 1.0);
}

// ── assess: poor / empty coverage ⇒ LongContext ───────────────────────────────

#[test]
fn assess_poor_coverage_routes_long_context() {
    let docs = vec![doc("alpha and unrelated filler words here")];
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.decision, RouteDecision::LongContext);
}

#[test]
fn assess_no_overlap_routes_long_context() {
    let docs = vec![doc("completely different subject matter entirely")];
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.decision, RouteDecision::LongContext);
}

#[test]
fn assess_no_overlap_answerability_zero() {
    let docs = vec![doc("completely different subject matter entirely")];
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.answerability, 0.0);
}

// ── assess: empty retrieved ⇒ LongContext, answerability 0 ─────────────────────

#[test]
fn assess_empty_retrieved_routes_long_context() {
    let docs: Vec<Document> = Vec::new();
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.decision, RouteDecision::LongContext);
}

#[test]
fn assess_empty_retrieved_answerability_zero() {
    let docs: Vec<Document> = Vec::new();
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.answerability, 0.0);
}

#[test]
fn assess_empty_retrieved_coverage_zero() {
    let docs: Vec<Document> = Vec::new();
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.query_coverage, 0.0);
}

#[test]
fn assess_empty_retrieved_top_relevance_zero() {
    let docs: Vec<Document> = Vec::new();
    let a = router().assess(Q4, &docs).unwrap();
    assert_eq!(a.top_relevance, 0.0);
}

#[test]
fn assess_empty_retrieved_reason_non_empty() {
    let docs: Vec<Document> = Vec::new();
    let a = router().assess(Q4, &docs).unwrap();
    assert!(!a.reason.is_empty());
}

// ── threshold boundary flips the decision ─────────────────────────────────────

#[test]
fn boundary_exact_threshold_routes_rag() {
    // weight 1.0 ⇒ answerability == coverage == 0.5; threshold 0.5 ⇒ 0.5 >= 0.5.
    let cfg = SelfRouteConfig::new()
        .with_coverage_weight(1.0)
        .with_answerability_threshold(0.5);
    let docs = vec![doc("alpha bravo unrelated filler")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    assert_eq!(a.decision, RouteDecision::Rag);
}

#[test]
fn boundary_just_above_threshold_routes_long_context() {
    // answerability 0.5 < threshold 0.5001 ⇒ LongContext.
    let cfg = SelfRouteConfig::new()
        .with_coverage_weight(1.0)
        .with_answerability_threshold(0.5001);
    let docs = vec![doc("alpha bravo unrelated filler")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    assert_eq!(a.decision, RouteDecision::LongContext);
}

#[test]
fn boundary_just_below_threshold_routes_rag() {
    // answerability 0.5 >= threshold 0.4999 ⇒ Rag.
    let cfg = SelfRouteConfig::new()
        .with_coverage_weight(1.0)
        .with_answerability_threshold(0.4999);
    let docs = vec![doc("alpha bravo unrelated filler")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    assert_eq!(a.decision, RouteDecision::Rag);
}

#[test]
fn boundary_flip_same_input_different_threshold() {
    let docs = vec![doc("alpha bravo unrelated filler")];
    let low = SelfRouter::new(
        SelfRouteConfig::new()
            .with_coverage_weight(1.0)
            .with_answerability_threshold(0.4),
    )
    .route(Q4, &docs)
    .unwrap();
    let high = SelfRouter::new(
        SelfRouteConfig::new()
            .with_coverage_weight(1.0)
            .with_answerability_threshold(0.6),
    )
    .route(Q4, &docs)
    .unwrap();
    assert_ne!(low, high);
}

#[test]
fn zero_threshold_always_routes_rag() {
    // Any non-empty retrieved set with threshold 0.0 ⇒ answerability >= 0.0 ⇒ Rag.
    let cfg = SelfRouteConfig::new().with_answerability_threshold(0.0);
    let docs = vec![doc("completely unrelated text")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    assert_eq!(a.decision, RouteDecision::Rag);
}

// ── coverage_weight blend respected ───────────────────────────────────────────

#[test]
fn blend_pure_coverage_weight_one() {
    // weight 1.0 ⇒ answerability == coverage (0.5), ignoring top relevance.
    let cfg = SelfRouteConfig::new().with_coverage_weight(1.0);
    let docs = vec![doc("alpha bravo unrelated filler")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    assert_eq!(a.answerability, a.query_coverage);
}

#[test]
fn blend_pure_relevance_weight_zero() {
    // weight 0.0 ⇒ answerability == top_relevance, ignoring coverage.
    let cfg = SelfRouteConfig::new().with_coverage_weight(0.0);
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    assert_eq!(a.answerability, a.top_relevance);
}

#[test]
fn blend_convex_combination_value() {
    // Single doc identical to query ⇒ coverage 1.0, top_relevance 1.0.
    // Any weight ⇒ answerability == 1.0.
    let cfg = SelfRouteConfig::new().with_coverage_weight(0.3);
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    assert_eq!(a.answerability, 1.0);
}

#[test]
fn blend_distinct_signals_weighted() {
    // coverage 1.0 (union covers all 4), top_relevance 0.25 (best single-doc
    // Jaccard: doc {alpha,charlie} vs query of 4 ⇒ 1/5 = 0.2 ... use a doc that
    // yields a clean fraction). Here best doc shares 1 of union 4 ⇒ 1/4 = 0.25.
    let cfg = SelfRouteConfig::new().with_coverage_weight(0.5);
    let docs = vec![doc("alpha"), doc("bravo"), doc("charlie"), doc("delta")];
    let a = SelfRouter::new(cfg).assess(Q4, &docs).unwrap();
    let expected = 0.5 * a.query_coverage + 0.5 * a.top_relevance;
    assert_eq!(a.answerability, expected);
}

#[test]
fn blend_weight_affects_answerability() {
    // Coverage high (1.0), top_relevance low ⇒ higher coverage_weight raises
    // answerability.
    let docs = vec![doc("alpha"), doc("bravo"), doc("charlie"), doc("delta")];
    let high_cov = SelfRouter::new(SelfRouteConfig::new().with_coverage_weight(1.0))
        .assess(Q4, &docs)
        .unwrap()
        .answerability;
    let low_cov = SelfRouter::new(SelfRouteConfig::new().with_coverage_weight(0.0))
        .assess(Q4, &docs)
        .unwrap()
        .answerability;
    assert!(high_cov > low_cov);
}

// ── reason string ─────────────────────────────────────────────────────────────

#[test]
fn reason_non_empty_for_rag() {
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = router().assess(Q4, &docs).unwrap();
    assert!(!a.reason.is_empty());
}

#[test]
fn reason_non_empty_for_long_context() {
    let docs = vec![doc("completely unrelated text content")];
    let a = router().assess(Q4, &docs).unwrap();
    assert!(!a.reason.is_empty());
}

#[test]
fn reason_mentions_decision_for_rag() {
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = router().assess(Q4, &docs).unwrap();
    assert!(a.reason.contains("rag"));
}

#[test]
fn reason_mentions_decision_for_long_context() {
    let docs = vec![doc("completely unrelated text content")];
    let a = router().assess(Q4, &docs).unwrap();
    assert!(a.reason.contains("long_context"));
}

// ── route convenience ─────────────────────────────────────────────────────────

#[test]
fn route_matches_assess_decision_rag() {
    let docs = vec![doc("alpha bravo charlie delta")];
    let r = router();
    assert_eq!(
        r.route(Q4, &docs).unwrap(),
        r.assess(Q4, &docs).unwrap().decision
    );
}

#[test]
fn route_matches_assess_decision_long_context() {
    let docs = vec![doc("completely unrelated text content")];
    let r = router();
    assert_eq!(
        r.route(Q4, &docs).unwrap(),
        r.assess(Q4, &docs).unwrap().decision
    );
}

#[test]
fn route_empty_retrieved_long_context() {
    let docs: Vec<Document> = Vec::new();
    assert_eq!(
        router().route(Q4, &docs).unwrap(),
        RouteDecision::LongContext
    );
}

// ── EmptyQuery error ──────────────────────────────────────────────────────────

#[test]
fn assess_empty_query_errors() {
    let docs = vec![doc("alpha bravo charlie delta")];
    assert!(matches!(
        router().assess("", &docs),
        Err(SelfRouteError::EmptyQuery)
    ));
}

#[test]
fn assess_whitespace_query_errors() {
    let docs = vec![doc("alpha bravo charlie delta")];
    assert!(matches!(
        router().assess("   \t\n ", &docs),
        Err(SelfRouteError::EmptyQuery)
    ));
}

#[test]
fn route_empty_query_errors() {
    let docs = vec![doc("alpha bravo charlie delta")];
    assert!(matches!(
        router().route("", &docs),
        Err(SelfRouteError::EmptyQuery)
    ));
}

#[test]
fn empty_query_with_empty_docs_still_errors() {
    // Query emptiness is checked before the empty-retrieved short-circuit.
    let docs: Vec<Document> = Vec::new();
    assert!(matches!(
        router().assess("", &docs),
        Err(SelfRouteError::EmptyQuery)
    ));
}

#[test]
fn empty_query_error_display() {
    assert_eq!(
        SelfRouteError::EmptyQuery.to_string(),
        "query must not be empty"
    );
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn assess_is_deterministic() {
    let docs = vec![doc("alpha bravo nothing else here")];
    let r = router();
    let first = r.assess(Q4, &docs).unwrap();
    let second = r.assess(Q4, &docs).unwrap();
    assert_eq!(first, second);
}

#[test]
fn coverage_is_deterministic() {
    let docs = vec![doc("alpha bravo charlie unrelated")];
    let r = router();
    assert_eq!(r.query_coverage(Q4, &docs), r.query_coverage(Q4, &docs));
}

#[test]
fn route_is_deterministic() {
    let docs = vec![doc("alpha bravo charlie delta")];
    let r = router();
    assert_eq!(r.route(Q4, &docs).unwrap(), r.route(Q4, &docs).unwrap());
}

#[test]
fn assessment_is_clone_equal() {
    let docs = vec![doc("alpha bravo charlie delta")];
    let a = router().assess(Q4, &docs).unwrap();
    let cloned: RouteAssessment = a.clone();
    assert_eq!(a, cloned);
}
