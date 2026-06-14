#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]
//! Tests for the `source_credibility` module.

use std::collections::HashMap;

use crate::source_credibility::{
    AuthoritySignals, CredibilityConfig, CredibilityScore, CredibilityScorer,
    SourceCredibilityError, SourceGraph,
};
use crate::types::{Document, DocumentId, SearchResult};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn doc(id: &str) -> Document {
    Document::new(format!("content of {id}")).with_id(id)
}

fn doc_with_source(id: &str, source: &str) -> Document {
    doc(id).with_source(source)
}

fn doc_with_author(id: &str, author: &str) -> Document {
    doc(id).with_metadata("author", author)
}

fn result(id: &str, score: f32, rank: usize) -> SearchResult {
    SearchResult::new(doc(id), score, rank)
}

/// Sum of all PageRank scores.
fn rank_sum(ranks: &HashMap<DocumentId, f32>) -> f32 {
    ranks.values().sum()
}

// ── SourceGraph: construction ───────────────────────────────────────────────────

#[test]
fn test_graph_new_is_empty() {
    let g = SourceGraph::new();
    assert_eq!(g.node_count(), 0);
    assert_eq!(g.edge_count(), 0);
    assert!(g.is_empty());
}

#[test]
fn test_graph_default_is_empty() {
    let g = SourceGraph::default();
    assert!(g.is_empty());
}

#[test]
fn test_add_node_increments_count() {
    let mut g = SourceGraph::new();
    g.add_node(DocumentId::from("a"));
    assert_eq!(g.node_count(), 1);
    assert!(!g.is_empty());
}

#[test]
fn test_add_node_is_idempotent() {
    let mut g = SourceGraph::new();
    g.add_node(DocumentId::from("a"));
    g.add_node(DocumentId::from("a"));
    assert_eq!(g.node_count(), 1);
}

#[test]
fn test_add_citation_registers_both_nodes() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edge_count(), 1);
}

#[test]
fn test_add_citation_records_edge_direction() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    let from_a = g.citations_from(&DocumentId::from("a")).unwrap();
    assert_eq!(from_a, &[DocumentId::from("b")]);
    assert!(g.citations_from(&DocumentId::from("b")).is_none());
}

#[test]
fn test_self_citation_ignored() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("a"));
    assert_eq!(g.node_count(), 1);
    assert_eq!(g.edge_count(), 0);
}

#[test]
fn test_duplicate_citation_increases_edge_count() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    assert_eq!(g.edge_count(), 2);
}

#[test]
fn test_node_count_with_mixed_adds() {
    let mut g = SourceGraph::new();
    g.add_node(DocumentId::from("x"));
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("c"));
    // nodes: x, a, b, c
    assert_eq!(g.node_count(), 4);
}

// ── PageRank: empty / single ────────────────────────────────────────────────────

#[test]
fn test_pagerank_empty_graph_returns_empty_map() {
    let g = SourceGraph::new();
    let ranks = g.pagerank(0.85, 50);
    assert!(ranks.is_empty());
}

#[test]
fn test_pagerank_checked_empty_errors() {
    let g = SourceGraph::new();
    let err = g.pagerank_checked(0.85, 50).unwrap_err();
    assert!(matches!(err, SourceCredibilityError::EmptyGraph));
}

#[test]
fn test_pagerank_checked_nonempty_ok() {
    let mut g = SourceGraph::new();
    g.add_node(DocumentId::from("a"));
    let ranks = g.pagerank_checked(0.85, 50).unwrap();
    assert_eq!(ranks.len(), 1);
}

#[test]
fn test_pagerank_single_node_is_one() {
    let mut g = SourceGraph::new();
    g.add_node(DocumentId::from("a"));
    let ranks = g.pagerank(0.85, 50);
    assert!((ranks[&DocumentId::from("a")] - 1.0).abs() < 1e-5);
}

// ── PageRank: sums to 1 / converges ─────────────────────────────────────────────

#[test]
fn test_pagerank_sums_to_one_chain() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("c"));
    let ranks = g.pagerank(0.85, 100);
    assert!((rank_sum(&ranks) - 1.0).abs() < 1e-4);
}

#[test]
fn test_pagerank_sums_to_one_cycle() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("c"));
    g.add_citation(DocumentId::from("c"), DocumentId::from("a"));
    let ranks = g.pagerank(0.85, 100);
    assert!((rank_sum(&ranks) - 1.0).abs() < 1e-4);
}

#[test]
fn test_pagerank_sums_to_one_star() {
    let mut g = SourceGraph::new();
    for src in ["a", "b", "c", "d"] {
        g.add_citation(DocumentId::from(src), DocumentId::from("hub"));
    }
    let ranks = g.pagerank(0.85, 100);
    assert!((rank_sum(&ranks) - 1.0).abs() < 1e-4);
}

#[test]
fn test_pagerank_symmetric_cycle_uniform() {
    // A 3-cycle is fully symmetric → all ranks equal.
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("c"));
    g.add_citation(DocumentId::from("c"), DocumentId::from("a"));
    let ranks = g.pagerank(0.85, 200);
    let a = ranks[&DocumentId::from("a")];
    let b = ranks[&DocumentId::from("b")];
    let c = ranks[&DocumentId::from("c")];
    assert!((a - b).abs() < 1e-4);
    assert!((b - c).abs() < 1e-4);
    assert!((a - 1.0 / 3.0).abs() < 1e-4);
}

#[test]
fn test_pagerank_converges_more_iterations_stable() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("hub"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("hub"));
    g.add_citation(DocumentId::from("c"), DocumentId::from("hub"));
    let r50 = g.pagerank(0.85, 50);
    let r200 = g.pagerank(0.85, 200);
    for (id, v) in &r50 {
        assert!((v - r200[id]).abs() < 1e-4, "non-convergent for {id}");
    }
}

// ── PageRank: more-cited node ranks higher ──────────────────────────────────────

#[test]
fn test_more_cited_node_ranks_higher() {
    // hub is cited by a, b, c; leaf is cited by none.
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("hub"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("hub"));
    g.add_citation(DocumentId::from("c"), DocumentId::from("hub"));
    g.add_node(DocumentId::from("leaf"));
    let ranks = g.pagerank(0.85, 100);
    assert!(ranks[&DocumentId::from("hub")] > ranks[&DocumentId::from("a")]);
    assert!(ranks[&DocumentId::from("hub")] > ranks[&DocumentId::from("leaf")]);
}

#[test]
fn test_more_citations_strictly_increase_rank() {
    // node y cited twice (by a and b); node z cited once (by a).
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("y"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("y"));
    g.add_citation(DocumentId::from("a"), DocumentId::from("z"));
    let ranks = g.pagerank(0.85, 100);
    assert!(ranks[&DocumentId::from("y")] > ranks[&DocumentId::from("z")]);
}

#[test]
fn test_authority_from_authoritative_citer_ranks_high() {
    // hub is highly cited; hub then cites target. target inherits authority.
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("hub"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("hub"));
    g.add_citation(DocumentId::from("c"), DocumentId::from("hub"));
    g.add_citation(DocumentId::from("hub"), DocumentId::from("target"));
    // An isolated cited-once node for comparison.
    g.add_citation(DocumentId::from("a"), DocumentId::from("plain"));
    let ranks = g.pagerank(0.85, 200);
    assert!(ranks[&DocumentId::from("target")] > ranks[&DocumentId::from("plain")]);
}

// ── PageRank: dangling nodes ────────────────────────────────────────────────────

#[test]
fn test_dangling_node_handled_sum_preserved() {
    // "sink" has no outgoing edges → dangling.
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("sink"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("sink"));
    let ranks = g.pagerank(0.85, 100);
    assert!((rank_sum(&ranks) - 1.0).abs() < 1e-4);
}

#[test]
fn test_all_dangling_nodes_uniform() {
    // No edges at all → every node dangling → uniform distribution.
    let mut g = SourceGraph::new();
    g.add_node(DocumentId::from("a"));
    g.add_node(DocumentId::from("b"));
    g.add_node(DocumentId::from("c"));
    g.add_node(DocumentId::from("d"));
    let ranks = g.pagerank(0.85, 100);
    assert!((rank_sum(&ranks) - 1.0).abs() < 1e-4);
    for v in ranks.values() {
        assert!((v - 0.25).abs() < 1e-4);
    }
}

#[test]
fn test_dangling_sink_ranks_highest() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("sink"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("sink"));
    g.add_citation(DocumentId::from("c"), DocumentId::from("sink"));
    let ranks = g.pagerank(0.85, 200);
    let sink = ranks[&DocumentId::from("sink")];
    assert!(sink > ranks[&DocumentId::from("a")]);
    assert!(sink > ranks[&DocumentId::from("b")]);
}

#[test]
fn test_pagerank_damping_clamped() {
    // damping > 1 is clamped; should still produce a valid distribution.
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    let ranks = g.pagerank(5.0, 50);
    assert!((rank_sum(&ranks) - 1.0).abs() < 1e-3);
}

#[test]
fn test_pagerank_zero_iterations_uniform() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    let ranks = g.pagerank(0.85, 0);
    // No iterations → still initial uniform.
    for v in ranks.values() {
        assert!((v - 0.5).abs() < 1e-6);
    }
}

// ── PageRank: determinism ───────────────────────────────────────────────────────

#[test]
fn test_pagerank_deterministic() {
    let mut g = SourceGraph::new();
    g.add_citation(DocumentId::from("a"), DocumentId::from("b"));
    g.add_citation(DocumentId::from("b"), DocumentId::from("c"));
    g.add_citation(DocumentId::from("c"), DocumentId::from("a"));
    g.add_citation(DocumentId::from("a"), DocumentId::from("c"));
    let r1 = g.pagerank(0.85, 100);
    let r2 = g.pagerank(0.85, 100);
    assert_eq!(r1, r2);
}

// ── Recency scoring ─────────────────────────────────────────────────────────────

#[test]
fn test_recency_age_zero_is_one() {
    let scorer = CredibilityScorer::default();
    assert!((scorer.recency_score(0.0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_recency_at_half_life_is_half() {
    let scorer = CredibilityScorer::new(CredibilityConfig::new().with_half_life_days(365.0));
    assert!((scorer.recency_score(365.0) - 0.5).abs() < 1e-5);
}

#[test]
fn test_recency_at_two_half_lives_is_quarter() {
    let scorer = CredibilityScorer::new(CredibilityConfig::new().with_half_life_days(365.0));
    assert!((scorer.recency_score(730.0) - 0.25).abs() < 1e-5);
}

#[test]
fn test_recency_decays_monotonically() {
    let scorer = CredibilityScorer::default();
    let a = scorer.recency_score(0.0);
    let b = scorer.recency_score(100.0);
    let c = scorer.recency_score(500.0);
    let d = scorer.recency_score(2000.0);
    assert!(a > b && b > c && c > d);
}

#[test]
fn test_recency_negative_age_clamped_to_one() {
    let scorer = CredibilityScorer::default();
    assert!((scorer.recency_score(-100.0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_recency_custom_half_life() {
    let scorer = CredibilityScorer::new(CredibilityConfig::new().with_half_life_days(30.0));
    assert!((scorer.recency_score(30.0) - 0.5).abs() < 1e-5);
}

#[test]
fn test_recency_zero_half_life_step() {
    let scorer = CredibilityScorer::new(CredibilityConfig::new().with_half_life_days(0.0));
    assert!((scorer.recency_score(0.0) - 1.0).abs() < 1e-6);
    assert!((scorer.recency_score(1.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_recency_in_unit_range() {
    let scorer = CredibilityScorer::default();
    for age in [0.0, 1.0, 50.0, 365.0, 10_000.0] {
        let r = scorer.recency_score(age);
        assert!((0.0..=1.0).contains(&r));
    }
}

// ── Authority scoring ───────────────────────────────────────────────────────────

#[test]
fn test_authority_trusted_source_is_high() {
    let signals = AuthoritySignals::new().with_trusted_source("nature.com");
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::new(), signals);
    let d = doc_with_source("a", "nature.com");
    assert!((scorer.authority_score(&d) - 1.0).abs() < 1e-6);
}

#[test]
fn test_authority_untrusted_source_is_zero() {
    let signals = AuthoritySignals::new().with_trusted_source("nature.com");
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::new(), signals);
    let d = doc_with_source("a", "random-blog.example");
    assert!((scorer.authority_score(&d) - 0.0).abs() < 1e-6);
}

#[test]
fn test_authority_no_source_no_author_is_zero() {
    let scorer = CredibilityScorer::default();
    let d = doc("a");
    assert!((scorer.authority_score(&d) - 0.0).abs() < 1e-6);
}

#[test]
fn test_authority_author_reputation_contributes() {
    let signals = AuthoritySignals::new().with_author("Einstein", 0.9);
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::new(), signals);
    let d = doc_with_author("a", "Einstein");
    assert!((scorer.authority_score(&d) - 0.9).abs() < 1e-6);
}

#[test]
fn test_authority_unknown_author_is_zero() {
    let signals = AuthoritySignals::new().with_author("Einstein", 0.9);
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::new(), signals);
    let d = doc_with_author("a", "Nobody");
    assert!((scorer.authority_score(&d) - 0.0).abs() < 1e-6);
}

#[test]
fn test_authority_trusted_beats_low_author() {
    // max(trusted=1.0, author=0.3) = 1.0
    let signals = AuthoritySignals::new()
        .with_trusted_source("nature.com")
        .with_author("Jr", 0.3);
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::new(), signals);
    let d = doc_with_source("a", "nature.com").with_metadata("author", "Jr");
    assert!((scorer.authority_score(&d) - 1.0).abs() < 1e-6);
}

#[test]
fn test_authority_reputation_clamped() {
    // Over-range reputation is clamped to 1.0.
    let signals = AuthoritySignals::new().with_author("X", 5.0);
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::new(), signals);
    let d = doc_with_author("a", "X");
    assert!((scorer.authority_score(&d) - 1.0).abs() < 1e-6);
}

#[test]
fn test_authority_negative_reputation_clamped() {
    let signals = AuthoritySignals::new().with_author("X", -2.0);
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::new(), signals);
    let d = doc_with_author("a", "X");
    assert!((scorer.authority_score(&d) - 0.0).abs() < 1e-6);
}

#[test]
fn test_authority_signals_lookups() {
    let signals = AuthoritySignals::new()
        .with_trusted_source("s")
        .with_author("A", 0.7);
    assert!(signals.is_trusted("s"));
    assert!(!signals.is_trusted("t"));
    assert_eq!(signals.reputation_of("A"), Some(0.7));
    assert_eq!(signals.reputation_of("B"), None);
}

#[test]
fn test_authority_signals_default_empty() {
    let signals = AuthoritySignals::default();
    assert!(signals.trusted_sources.is_empty());
    assert!(signals.author_reputation.is_empty());
}

// ── Blended score ───────────────────────────────────────────────────────────────

#[test]
fn test_score_blends_three_components() {
    let signals = AuthoritySignals::new().with_trusted_source("nature.com");
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::default(), signals);
    let d = doc_with_source("a", "nature.com");
    // pagerank=1.0, age=0 → recency=1.0, authority=1.0 → total should be ~1.0
    let s = scorer.score(&d, 1.0, 0.0);
    assert!((s.pagerank - 1.0).abs() < 1e-6);
    assert!((s.recency - 1.0).abs() < 1e-6);
    assert!((s.authority - 1.0).abs() < 1e-6);
    assert!((s.total - 1.0).abs() < 1e-5);
}

#[test]
fn test_score_weighted_blend_value() {
    // Weights 0.4 / 0.3 / 0.3 (default), pr=1, rec=0 (age=inf-ish), auth=0
    let scorer = CredibilityScorer::default();
    let d = doc("a");
    // age large → recency ~0
    let s = scorer.score(&d, 1.0, 1.0e6);
    // total = 0.4*1 + 0.3*~0 + 0.3*0 = ~0.4
    assert!((s.total - 0.4).abs() < 1e-3);
}

#[test]
fn test_score_zero_components_zero_total() {
    let scorer = CredibilityScorer::default();
    let d = doc("a");
    let s = scorer.score(&d, 0.0, 1.0e6);
    assert!(s.total < 1e-3);
}

#[test]
fn test_score_pagerank_clamped() {
    let scorer = CredibilityScorer::default();
    let d = doc("a");
    // pagerank passed as 10.0 should clamp to 1.0 in the component.
    let s = scorer.score(&d, 10.0, 0.0);
    assert!((s.pagerank - 1.0).abs() < 1e-6);
    assert!((0.0..=1.0).contains(&s.total));
}

#[test]
fn test_score_components_in_range() {
    let signals = AuthoritySignals::new().with_author("A", 0.5);
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::default(), signals);
    let d = doc_with_author("a", "A");
    let s = scorer.score(&d, 0.3, 200.0);
    for v in [s.total, s.pagerank, s.recency, s.authority] {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn test_score_weights_normalized_unnormalized_equivalent() {
    // Weights (4, 3, 3) and (0.4, 0.3, 0.3) should give identical totals.
    let cfg_a = CredibilityConfig::new()
        .with_pagerank_weight(4.0)
        .with_recency_weight(3.0)
        .with_authority_weight(3.0);
    let cfg_b = CredibilityConfig::new()
        .with_pagerank_weight(0.4)
        .with_recency_weight(0.3)
        .with_authority_weight(0.3);
    let sa = CredibilityScorer::new(cfg_a);
    let sb = CredibilityScorer::new(cfg_b);
    let d = doc("a");
    let ra = sa.score(&d, 0.8, 100.0);
    let rb = sb.score(&d, 0.8, 100.0);
    assert!((ra.total - rb.total).abs() < 1e-5);
}

#[test]
fn test_score_label_thresholds() {
    let s_high = CredibilityScore::new(0.8, 0.8, 0.8, 0.8);
    let s_med = CredibilityScore::new(0.5, 0.5, 0.5, 0.5);
    let s_low = CredibilityScore::new(0.2, 0.2, 0.2, 0.2);
    assert_eq!(s_high.label(), "high");
    assert_eq!(s_med.label(), "medium");
    assert_eq!(s_low.label(), "low");
}

#[test]
fn test_score_pagerank_weight_dominates() {
    // Heavy pagerank weight → total tracks pagerank.
    let cfg = CredibilityConfig::new()
        .with_pagerank_weight(1.0)
        .with_recency_weight(0.0)
        .with_authority_weight(0.0);
    let scorer = CredibilityScorer::new(cfg);
    let d = doc("a");
    let s = scorer.score(&d, 0.7, 1.0e6);
    assert!((s.total - 0.7).abs() < 1e-4);
}

#[test]
fn test_score_deterministic() {
    let signals = AuthoritySignals::new().with_author("A", 0.6);
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::default(), signals);
    let d = doc_with_author("a", "A");
    let s1 = scorer.score(&d, 0.5, 120.0);
    let s2 = scorer.score(&d, 0.5, 120.0);
    assert_eq!(s1, s2);
}

// ── Config builders ─────────────────────────────────────────────────────────────

#[test]
fn test_config_default_values() {
    let c = CredibilityConfig::default();
    assert!((c.pagerank_weight - 0.4).abs() < 1e-6);
    assert!((c.recency_weight - 0.3).abs() < 1e-6);
    assert!((c.authority_weight - 0.3).abs() < 1e-6);
    assert!((c.damping - 0.85).abs() < 1e-6);
    assert_eq!(c.iterations, 50);
    assert!((c.half_life_days - 365.0).abs() < 1e-6);
}

#[test]
fn test_config_new_equals_default() {
    let a = CredibilityConfig::new();
    let b = CredibilityConfig::default();
    assert!((a.pagerank_weight - b.pagerank_weight).abs() < 1e-6);
    assert_eq!(a.iterations, b.iterations);
}

#[test]
fn test_config_builders() {
    let c = CredibilityConfig::new()
        .with_pagerank_weight(0.5)
        .with_recency_weight(0.25)
        .with_authority_weight(0.25)
        .with_damping(0.9)
        .with_iterations(100)
        .with_half_life_days(180.0);
    assert!((c.pagerank_weight - 0.5).abs() < 1e-6);
    assert!((c.recency_weight - 0.25).abs() < 1e-6);
    assert!((c.authority_weight - 0.25).abs() < 1e-6);
    assert!((c.damping - 0.9).abs() < 1e-6);
    assert_eq!(c.iterations, 100);
    assert!((c.half_life_days - 180.0).abs() < 1e-6);
}

#[test]
fn test_config_weight_sum() {
    let c = CredibilityConfig::default();
    assert!((c.weight_sum() - 1.0).abs() < 1e-6);
}

#[test]
fn test_config_normalized_weights() {
    let c = CredibilityConfig::new()
        .with_pagerank_weight(2.0)
        .with_recency_weight(1.0)
        .with_authority_weight(1.0);
    let (a, b, d) = c.normalized_weights();
    assert!((a - 0.5).abs() < 1e-6);
    assert!((b - 0.25).abs() < 1e-6);
    assert!((d - 0.25).abs() < 1e-6);
    assert!((a + b + d - 1.0).abs() < 1e-6);
}

#[test]
fn test_config_zero_weights_fall_back_uniform() {
    let c = CredibilityConfig::new()
        .with_pagerank_weight(0.0)
        .with_recency_weight(0.0)
        .with_authority_weight(0.0);
    let (a, b, d) = c.normalized_weights();
    let third = 1.0 / 3.0;
    assert!((a - third).abs() < 1e-6);
    assert!((b - third).abs() < 1e-6);
    assert!((d - third).abs() < 1e-6);
}

#[test]
fn test_authority_signals_new_equals_default() {
    let a = AuthoritySignals::new();
    assert!(a.trusted_sources.is_empty());
    assert!(a.author_reputation.is_empty());
}

#[test]
fn test_scorer_default_uses_default_config() {
    let scorer = CredibilityScorer::default();
    assert_eq!(scorer.config.iterations, 50);
    assert!(scorer.signals.trusted_sources.is_empty());
}

#[test]
fn test_scorer_new_empty_signals() {
    let scorer = CredibilityScorer::new(CredibilityConfig::default());
    assert!(scorer.signals.trusted_sources.is_empty());
    assert!(scorer.signals.author_reputation.is_empty());
}

// ── Re-ranking ──────────────────────────────────────────────────────────────────

#[test]
fn test_rerank_preserves_count() {
    let scorer = CredibilityScorer::default();
    let graph = SourceGraph::new();
    let ages = HashMap::new();
    let results = vec![result("a", 0.9, 0), result("b", 0.8, 1)];
    let out = scorer.rerank(&results, &graph, &ages, 0.5);
    assert_eq!(out.len(), 2);
}

#[test]
fn test_rerank_assigns_dense_ranks() {
    let scorer = CredibilityScorer::default();
    let graph = SourceGraph::new();
    let ages = HashMap::new();
    let results = vec![
        result("a", 0.9, 0),
        result("b", 0.8, 1),
        result("c", 0.7, 2),
    ];
    let out = scorer.rerank(&results, &graph, &ages, 0.5);
    assert_eq!(out[0].rank, 0);
    assert_eq!(out[1].rank, 1);
    assert_eq!(out[2].rank, 2);
}

#[test]
fn test_rerank_sorted_descending() {
    let scorer = CredibilityScorer::default();
    let graph = SourceGraph::new();
    let ages = HashMap::new();
    let results = vec![
        result("a", 0.5, 0),
        result("b", 0.9, 1),
        result("c", 0.7, 2),
    ];
    let out = scorer.rerank(&results, &graph, &ages, 1.0);
    // relevance_weight = 1.0 → pure relevance ordering: b, c, a
    assert_eq!(out[0].document.id.as_str(), "b");
    assert_eq!(out[1].document.id.as_str(), "c");
    assert_eq!(out[2].document.id.as_str(), "a");
    assert!(out[0].score >= out[1].score && out[1].score >= out[2].score);
}

#[test]
fn test_rerank_credibility_reorders() {
    // Two equally-relevant docs; one is highly cited → should win when
    // credibility weight dominates.
    let mut graph = SourceGraph::new();
    graph.add_citation(DocumentId::from("x"), DocumentId::from("cited"));
    graph.add_citation(DocumentId::from("y"), DocumentId::from("cited"));
    graph.add_citation(DocumentId::from("z"), DocumentId::from("cited"));
    graph.add_node(DocumentId::from("plain"));

    let scorer = CredibilityScorer::default();
    let ages = HashMap::new();
    let results = vec![result("plain", 0.6, 0), result("cited", 0.6, 1)];
    // relevance_weight = 0.0 → pure credibility; cited has higher pagerank.
    let out = scorer.rerank(&results, &graph, &ages, 0.0);
    assert_eq!(out[0].document.id.as_str(), "cited");
    assert_eq!(out[1].document.id.as_str(), "plain");
}

#[test]
fn test_rerank_recency_reorders() {
    // Equal relevance; the more recent doc wins under credibility weighting.
    let scorer = CredibilityScorer::default();
    let graph = SourceGraph::new();
    let mut ages = HashMap::new();
    ages.insert(DocumentId::from("old"), 5000.0);
    ages.insert(DocumentId::from("new"), 0.0);
    let results = vec![result("old", 0.5, 0), result("new", 0.5, 1)];
    let out = scorer.rerank(&results, &graph, &ages, 0.0);
    assert_eq!(out[0].document.id.as_str(), "new");
}

#[test]
fn test_rerank_authority_reorders() {
    let signals = AuthoritySignals::new().with_trusted_source("trusted");
    let scorer = CredibilityScorer::with_signals(CredibilityConfig::default(), signals);
    let graph = SourceGraph::new();
    let ages = HashMap::new();
    let results = vec![
        SearchResult::new(doc_with_source("a", "untrusted"), 0.5, 0),
        SearchResult::new(doc_with_source("b", "trusted"), 0.5, 1),
    ];
    let out = scorer.rerank(&results, &graph, &ages, 0.0);
    assert_eq!(out[0].document.id.as_str(), "b");
}

#[test]
fn test_rerank_blends_relevance_and_credibility() {
    // Mid relevance_weight: both signals matter. High-relevance-but-low-cred vs
    // low-relevance-but-high-cred. Verify blended scores stored on results.
    let mut graph = SourceGraph::new();
    graph.add_citation(DocumentId::from("p"), DocumentId::from("hi_cred"));
    graph.add_citation(DocumentId::from("q"), DocumentId::from("hi_cred"));
    graph.add_node(DocumentId::from("hi_rel"));

    let scorer = CredibilityScorer::default();
    let ages = HashMap::new();
    let results = vec![result("hi_rel", 1.0, 0), result("hi_cred", 0.2, 1)];
    let out = scorer.rerank(&results, &graph, &ages, 0.5);
    // Scores were overwritten with blended values, both within [0, 1].
    for r in &out {
        assert!((0.0..=1.0).contains(&r.score));
    }
}

#[test]
fn test_rerank_relevance_weight_clamped() {
    let scorer = CredibilityScorer::default();
    let graph = SourceGraph::new();
    let ages = HashMap::new();
    let results = vec![result("a", 0.9, 0), result("b", 0.1, 1)];
    // relevance_weight > 1 clamped to 1 → pure relevance ordering.
    let out = scorer.rerank(&results, &graph, &ages, 5.0);
    assert_eq!(out[0].document.id.as_str(), "a");
}

#[test]
fn test_rerank_missing_age_defaults_recent() {
    // No ages provided → recency defaults to age 0 (recency=1.0) for all.
    let scorer = CredibilityScorer::default();
    let graph = SourceGraph::new();
    let ages = HashMap::new();
    let results = vec![result("a", 0.5, 0)];
    let out = scorer.rerank(&results, &graph, &ages, 0.0);
    // total = 0.4*0 + 0.3*1 + 0.3*0 = 0.3
    assert!((out[0].score - 0.3).abs() < 1e-3);
}

#[test]
fn test_rerank_empty_results() {
    let scorer = CredibilityScorer::default();
    let graph = SourceGraph::new();
    let ages = HashMap::new();
    let out = scorer.rerank(&[], &graph, &ages, 0.5);
    assert!(out.is_empty());
}

#[test]
fn test_rerank_deterministic() {
    let mut graph = SourceGraph::new();
    graph.add_citation(DocumentId::from("x"), DocumentId::from("a"));
    graph.add_citation(DocumentId::from("y"), DocumentId::from("b"));
    let scorer = CredibilityScorer::default();
    let ages = HashMap::new();
    let results = vec![
        result("a", 0.5, 0),
        result("b", 0.5, 1),
        result("c", 0.5, 2),
    ];
    let out1 = scorer.rerank(&results, &graph, &ages, 0.5);
    let out2 = scorer.rerank(&results, &graph, &ages, 0.5);
    let ids1: Vec<&str> = out1.iter().map(|r| r.document.id.as_str()).collect();
    let ids2: Vec<&str> = out2.iter().map(|r| r.document.id.as_str()).collect();
    assert_eq!(ids1, ids2);
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert_eq!(a.score, b.score);
    }
}

#[test]
fn test_rerank_tie_break_by_id() {
    // Two results identical in every scoring dimension → stable tie-break by id.
    let scorer = CredibilityScorer::default();
    let graph = SourceGraph::new();
    let ages = HashMap::new();
    let results = vec![result("zzz", 0.5, 0), result("aaa", 0.5, 1)];
    let out = scorer.rerank(&results, &graph, &ages, 1.0);
    // Equal score → "aaa" sorts before "zzz".
    assert_eq!(out[0].document.id.as_str(), "aaa");
    assert_eq!(out[1].document.id.as_str(), "zzz");
}
