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

use super::*;
use crate::types::{Document, DocumentId};

// ── Test helpers ────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

/// A deterministic stub retriever returning a fixed candidate list.
struct StubRetriever {
    name: String,
    results: Vec<(DocumentId, f32)>,
}

impl StubRetriever {
    fn new(name: &str, results: Vec<(&str, f32)>) -> Self {
        Self {
            name: name.to_string(),
            results: results
                .into_iter()
                .map(|(id, s)| (DocumentId::from_string(id), s))
                .collect(),
        }
    }
}

impl SubRetriever for StubRetriever {
    fn name(&self) -> &str {
        &self.name
    }

    fn retrieve(&self, _query: &str, top_k: usize) -> Vec<(DocumentId, f32)> {
        let mut out = self.results.clone();
        if top_k > 0 && out.len() > top_k {
            out.truncate(top_k);
        }
        out
    }
}

fn ids(ranked: &[(DocumentId, f32)]) -> Vec<String> {
    ranked.iter().map(|(id, _)| id.0.clone()).collect()
}

fn score_of(ranked: &[(DocumentId, f32)], id: &str) -> Option<f32> {
    ranked
        .iter()
        .find(|(d, _)| d.as_str() == id)
        .map(|(_, s)| *s)
}

// ── EnsembleConfig: defaults & builders ─────────────────────────────────────

#[test]
fn config_default_fusion_is_weighted_rrf() {
    let config = EnsembleConfig::default();
    assert_eq!(config.fusion, EnsembleFusion::WeightedRrf);
}

#[test]
fn config_default_rrf_k_is_sixty() {
    assert_eq!(EnsembleConfig::default().rrf_k, 60.0);
}

#[test]
fn config_default_top_n_is_zero() {
    assert_eq!(EnsembleConfig::default().top_n, 0);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(EnsembleConfig::new(), EnsembleConfig::default());
}

#[test]
fn enum_default_is_weighted_rrf() {
    assert_eq!(EnsembleFusion::default(), EnsembleFusion::WeightedRrf);
}

#[test]
fn config_with_fusion_sets_fusion() {
    let config = EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedScore);
    assert_eq!(config.fusion, EnsembleFusion::WeightedScore);
}

#[test]
fn config_with_rrf_k_sets_value() {
    let config = EnsembleConfig::new().with_rrf_k(12.5);
    assert_eq!(config.rrf_k, 12.5);
}

#[test]
fn config_with_top_n_sets_value() {
    let config = EnsembleConfig::new().with_top_n(7);
    assert_eq!(config.top_n, 7);
}

#[test]
fn config_builders_chain() {
    let config = EnsembleConfig::new()
        .with_fusion(EnsembleFusion::WeightedScore)
        .with_rrf_k(30.0)
        .with_top_n(3);
    assert_eq!(config.fusion, EnsembleFusion::WeightedScore);
    assert_eq!(config.rrf_k, 30.0);
    assert_eq!(config.top_n, 3);
}

#[test]
fn config_builders_do_not_mutate_unrelated_fields() {
    let config = EnsembleConfig::new().with_top_n(5);
    // fusion and rrf_k retain their defaults.
    assert_eq!(config.fusion, EnsembleFusion::WeightedRrf);
    assert_eq!(config.rrf_k, 60.0);
}

#[test]
fn config_is_clone_and_debug() {
    let config = EnsembleConfig::new().with_top_n(2);
    let cloned = config.clone();
    assert_eq!(config, cloned);
    let s = format!("{config:?}");
    assert!(s.contains("EnsembleConfig"));
}

#[test]
fn fusion_enum_is_copy_eq_hash_debug() {
    let a = EnsembleFusion::WeightedScore;
    let b = a; // Copy
    assert_eq!(a, b);
    let s = format!("{a:?}");
    assert!(s.contains("WeightedScore"));
}

// ── LexicalSubRetriever: overlap retrieval ──────────────────────────────────

#[test]
fn lexical_name_is_reported() {
    let r = LexicalSubRetriever::new("lex", vec![]);
    assert_eq!(r.name(), "lex");
}

#[test]
fn lexical_document_count() {
    let r = LexicalSubRetriever::new("lex", vec![doc("a", "x"), doc("b", "y")]);
    assert_eq!(r.document_count(), 2);
}

#[test]
fn lexical_default_is_empty() {
    let r = LexicalSubRetriever::default();
    assert_eq!(r.name(), "");
    assert_eq!(r.document_count(), 0);
}

#[test]
fn lexical_retrieves_by_overlap() {
    let docs = vec![
        doc("a", "the quick brown fox"),
        doc("b", "a slow green turtle"),
        doc("c", "quick brown bear"),
    ];
    let r = LexicalSubRetriever::new("lex", docs);
    let out = r.retrieve("quick brown", 10);
    let returned = ids(&out);
    // a and c share "quick" and "brown"; b shares nothing.
    assert!(returned.contains(&"a".to_string()));
    assert!(returned.contains(&"c".to_string()));
    assert!(!returned.contains(&"b".to_string()));
}

#[test]
fn lexical_higher_overlap_ranks_first() {
    let docs = vec![
        doc("a", "alpha beta gamma delta"),
        doc("b", "alpha only here"),
    ];
    let r = LexicalSubRetriever::new("lex", docs);
    let out = r.retrieve("alpha beta gamma", 10);
    // "a" overlaps 3 tokens, "b" overlaps 1.
    assert_eq!(out[0].0.as_str(), "a");
    assert!(out[0].1 > out[1].1);
}

#[test]
fn lexical_empty_query_returns_empty() {
    let r = LexicalSubRetriever::new("lex", vec![doc("a", "hello world")]);
    assert!(r.retrieve("", 10).is_empty());
    assert!(r.retrieve("   ", 10).is_empty());
}

#[test]
fn lexical_no_overlap_returns_empty() {
    let r = LexicalSubRetriever::new("lex", vec![doc("a", "completely unrelated text")]);
    assert!(r.retrieve("xyzzy plugh", 10).is_empty());
}

#[test]
fn lexical_respects_top_k() {
    let docs = vec![
        doc("a", "shared token here"),
        doc("b", "shared token there"),
        doc("c", "shared token everywhere"),
    ];
    let r = LexicalSubRetriever::new("lex", docs);
    let out = r.retrieve("shared token", 2);
    assert_eq!(out.len(), 2);
}

#[test]
fn lexical_is_case_insensitive() {
    let r = LexicalSubRetriever::new("lex", vec![doc("a", "Rust Programming")]);
    let out = r.retrieve("RUST programming", 10);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0.as_str(), "a");
}

#[test]
fn lexical_ignores_short_tokens() {
    // Single-character tokens (<2 chars) are dropped, so "a" matches nothing.
    let r = LexicalSubRetriever::new("lex", vec![doc("d", "a b c")]);
    assert!(r.retrieve("a", 10).is_empty());
}

#[test]
fn lexical_tie_break_is_deterministic_by_id() {
    let docs = vec![doc("zzz", "alpha beta"), doc("aaa", "alpha beta")];
    let r = LexicalSubRetriever::new("lex", docs);
    let out = r.retrieve("alpha beta", 10);
    // Equal overlap → ascending id tie-break, "aaa" before "zzz".
    assert_eq!(out[0].0.as_str(), "aaa");
    assert_eq!(out[1].0.as_str(), "zzz");
}

// ── add_retriever / retriever_count ─────────────────────────────────────────

#[test]
fn ensemble_starts_with_zero_retrievers() {
    let ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    assert_eq!(ensemble.retriever_count(), 0);
}

#[test]
fn add_retriever_increments_count() {
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    ensemble.add_retriever(Box::new(StubRetriever::new("r1", vec![("a", 1.0)])), 1.0);
    assert_eq!(ensemble.retriever_count(), 1);
    ensemble.add_retriever(Box::new(StubRetriever::new("r2", vec![("b", 1.0)])), 0.5);
    assert_eq!(ensemble.retriever_count(), 2);
}

#[test]
fn ensemble_config_accessor_returns_config() {
    let ensemble = EnsembleRetriever::new(EnsembleConfig::new().with_top_n(4));
    assert_eq!(ensemble.config().top_n, 4);
}

#[test]
fn add_lexical_retriever_works() {
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    let lex = LexicalSubRetriever::new("lex", vec![doc("a", "hello world")]);
    ensemble.add_retriever(Box::new(lex), 1.0);
    assert_eq!(ensemble.retriever_count(), 1);
    let out = ensemble.retrieve("hello", 10).unwrap();
    assert_eq!(out[0].0.as_str(), "a");
}

// ── Agreement reward: a doc from TWO retrievers outranks one from ONE ────────

#[test]
fn agreement_outranks_single_rrf() {
    // shared "a" appears in both at rank 0; "b" only in r1, "c" only in r2.
    let r1 = StubRetriever::new("r1", vec![("a", 1.0), ("b", 0.9)]);
    let r2 = StubRetriever::new("r2", vec![("a", 1.0), ("c", 0.9)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    ensemble.add_retriever(Box::new(r1), 1.0);
    ensemble.add_retriever(Box::new(r2), 1.0);

    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked[0].0.as_str(), "a");
    let sa = score_of(&ranked, "a").unwrap();
    let sb = score_of(&ranked, "b").unwrap();
    let sc = score_of(&ranked, "c").unwrap();
    assert!(sa > sb);
    assert!(sa > sc);
}

#[test]
fn agreement_outranks_single_weighted_score() {
    let r1 = StubRetriever::new("r1", vec![("a", 1.0), ("b", 0.8)]);
    let r2 = StubRetriever::new("r2", vec![("a", 1.0), ("c", 0.8)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedScore));
    ensemble.add_retriever(Box::new(r1), 1.0);
    ensemble.add_retriever(Box::new(r2), 1.0);

    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked[0].0.as_str(), "a");
    // "a" is normalized to 1.0 in both lists → fused 2.0; others get less.
    let sa = score_of(&ranked, "a").unwrap();
    assert!(sa > score_of(&ranked, "b").unwrap());
    assert!(sa > score_of(&ranked, "c").unwrap());
}

#[test]
fn agreement_accumulates_three_retrievers() {
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    for i in 0..3 {
        let name = format!("r{i}");
        ensemble.add_retriever(
            Box::new(StubRetriever::new(
                &name,
                vec![("shared", 1.0), ("solo", 0.5)],
            )),
            1.0,
        );
    }
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked[0].0.as_str(), "shared");
    // "shared" was hit 3 times at rank 0; "solo" 3 times at rank 1 → shared wins.
    assert!(score_of(&ranked, "shared").unwrap() > score_of(&ranked, "solo").unwrap());
}

#[test]
fn lexical_agreement_across_two_corpora() {
    let primary = vec![
        doc("a", "async tokio runtime"),
        doc("b", "ownership borrow"),
    ];
    let secondary = vec![
        doc("a", "tokio async scheduler"),
        doc("c", "python asyncio"),
    ];
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    ensemble.add_retriever(Box::new(LexicalSubRetriever::new("p", primary)), 1.0);
    ensemble.add_retriever(Box::new(LexicalSubRetriever::new("s", secondary)), 1.0);
    let ranked = ensemble.retrieve("async tokio", 10).unwrap();
    // "a" surfaced by both lexical retrievers.
    assert_eq!(ranked[0].0.as_str(), "a");
}

// ── Weights change ranking ──────────────────────────────────────────────────

#[test]
fn heavier_weight_retriever_top_doc_wins_rrf() {
    // r1's top doc is "x"; r2's top doc is "y". Heavily weight r2.
    let r1 = StubRetriever::new("r1", vec![("x", 1.0)]);
    let r2 = StubRetriever::new("r2", vec![("y", 1.0)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    ensemble.add_retriever(Box::new(r1), 1.0);
    ensemble.add_retriever(Box::new(r2), 10.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked[0].0.as_str(), "y");
}

#[test]
fn heavier_weight_retriever_top_doc_wins_weighted_score() {
    let r1 = StubRetriever::new("r1", vec![("x", 1.0), ("filler", 0.1)]);
    let r2 = StubRetriever::new("r2", vec![("y", 1.0), ("filler2", 0.1)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedScore));
    ensemble.add_retriever(Box::new(r1), 1.0);
    ensemble.add_retriever(Box::new(r2), 5.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked[0].0.as_str(), "y");
}

#[test]
fn flipping_weights_flips_winner() {
    let make = |w1: f32, w2: f32| {
        let mut e =
            EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
        e.add_retriever(Box::new(StubRetriever::new("r1", vec![("x", 1.0)])), w1);
        e.add_retriever(Box::new(StubRetriever::new("r2", vec![("y", 1.0)])), w2);
        e.retrieve("q", 10).unwrap()[0].0.0.clone()
    };
    assert_eq!(make(10.0, 1.0), "x");
    assert_eq!(make(1.0, 10.0), "y");
}

#[test]
fn zero_weight_retriever_contributes_nothing() {
    let r1 = StubRetriever::new("r1", vec![("x", 1.0)]);
    let r2 = StubRetriever::new("r2", vec![("y", 1.0)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    ensemble.add_retriever(Box::new(r1), 1.0);
    ensemble.add_retriever(Box::new(r2), 0.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    // "y" came only from the zero-weight retriever → score 0.0, ranks last.
    assert_eq!(ranked[0].0.as_str(), "x");
    assert_eq!(score_of(&ranked, "y").unwrap(), 0.0);
}

#[test]
fn weight_scales_contribution_proportionally() {
    let r1 = StubRetriever::new("r1", vec![("a", 1.0)]);
    let mut e1 =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    e1.add_retriever(Box::new(r1), 2.0);
    let s2 = score_of(&e1.retrieve("q", 10).unwrap(), "a").unwrap();

    let r2 = StubRetriever::new("r1", vec![("a", 1.0)]);
    let mut e2 =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    e2.add_retriever(Box::new(r2), 1.0);
    let s1 = score_of(&e2.retrieve("q", 10).unwrap(), "a").unwrap();

    // Weight 2.0 yields exactly double the contribution of weight 1.0.
    assert!((s2 - 2.0 * s1).abs() < 1e-6);
}

// ── WeightedScore vs WeightedRrf both run ───────────────────────────────────

#[test]
fn both_fusion_strategies_produce_results() {
    let docs1 = vec![("a", 0.9_f32), ("b", 0.4)];
    let docs2 = vec![("b", 0.8_f32), ("c", 0.2)];

    // Both strategies run and surface all three documents; "b" (in both lists)
    // always ranks at least as high as any single-retriever document.
    for fusion in [EnsembleFusion::WeightedScore, EnsembleFusion::WeightedRrf] {
        let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new().with_fusion(fusion));
        ensemble.add_retriever(Box::new(StubRetriever::new("r1", docs1.clone())), 1.0);
        ensemble.add_retriever(Box::new(StubRetriever::new("r2", docs2.clone())), 1.0);
        let ranked = ensemble.retrieve("q", 10).unwrap();
        assert_eq!(ranked.len(), 3);
        // "b" agrees across both retrievers, so it must not rank below "c"
        // (a single-retriever doc that "b" dominates on every axis).
        assert!(score_of(&ranked, "b").unwrap() >= score_of(&ranked, "c").unwrap());
    }
}

#[test]
fn weighted_rrf_top_doc_is_the_agreed_one() {
    // Under RRF, "b" (rank 1 in r1, rank 0 in r2) beats "a" (rank 0 in r1 only).
    let r1 = StubRetriever::new("r1", vec![("a", 0.9), ("b", 0.4)]);
    let r2 = StubRetriever::new("r2", vec![("b", 0.8), ("c", 0.2)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    ensemble.add_retriever(Box::new(r1), 1.0);
    ensemble.add_retriever(Box::new(r2), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked[0].0.as_str(), "b");
}

#[test]
fn weighted_score_ties_top_doc_resolved_by_id() {
    // Under WeightedScore, "a" and "b" both reach a fused score of 1.0, so the
    // ascending id tie-break makes "a" the top result.
    let r1 = StubRetriever::new("r1", vec![("a", 0.9), ("b", 0.4)]);
    let r2 = StubRetriever::new("r2", vec![("b", 0.8), ("c", 0.2)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedScore));
    ensemble.add_retriever(Box::new(r1), 1.0);
    ensemble.add_retriever(Box::new(r2), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    let sa = score_of(&ranked, "a").unwrap();
    let sb = score_of(&ranked, "b").unwrap();
    assert!((sa - 1.0).abs() < 1e-6);
    assert!((sb - 1.0).abs() < 1e-6);
    assert_eq!(ranked[0].0.as_str(), "a");
}

#[test]
fn weighted_score_normalizes_per_retriever() {
    // r1 scores in [0,1000], r2 in [0,1]. Normalization equalizes scales.
    let r1 = StubRetriever::new("r1", vec![("a", 1000.0), ("b", 0.0)]);
    let r2 = StubRetriever::new("r2", vec![("b", 1.0), ("a", 0.0)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedScore));
    ensemble.add_retriever(Box::new(r1), 1.0);
    ensemble.add_retriever(Box::new(r2), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    // Both "a" and "b" normalize to 1.0 in one list and 0.0 in the other → tie at 1.0.
    let sa = score_of(&ranked, "a").unwrap();
    let sb = score_of(&ranked, "b").unwrap();
    assert!((sa - 1.0).abs() < 1e-6);
    assert!((sb - 1.0).abs() < 1e-6);
    // Deterministic id tie-break: "a" before "b".
    assert_eq!(ranked[0].0.as_str(), "a");
}

#[test]
fn weighted_score_single_value_list_maps_to_one() {
    // A retriever returning a single doc → zero range → normalized to 1.0.
    let r1 = StubRetriever::new("r1", vec![("only", 42.0)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedScore));
    ensemble.add_retriever(Box::new(r1), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(score_of(&ranked, "only").unwrap(), 1.0);
}

#[test]
fn weighted_rrf_uses_rank_not_score() {
    // Despite a huge raw score on "b", RRF cares about rank: "a" at rank 0 wins.
    let r1 = StubRetriever::new("r1", vec![("a", 0.01), ("b", 9999.0)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    ensemble.add_retriever(Box::new(r1), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked[0].0.as_str(), "a");
}

#[test]
fn rrf_k_affects_score_magnitude() {
    let make_k = |k: f32| {
        let mut e = EnsembleRetriever::new(
            EnsembleConfig::new()
                .with_fusion(EnsembleFusion::WeightedRrf)
                .with_rrf_k(k),
        );
        e.add_retriever(Box::new(StubRetriever::new("r1", vec![("a", 1.0)])), 1.0);
        score_of(&e.retrieve("q", 10).unwrap(), "a").unwrap()
    };
    // Smaller k → larger reciprocal-rank contribution.
    assert!(make_k(1.0) > make_k(100.0));
}

// ── top_n truncation ────────────────────────────────────────────────────────

#[test]
fn top_n_truncates_results() {
    let r1 = StubRetriever::new("r1", vec![("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7)]);
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new().with_top_n(2));
    ensemble.add_retriever(Box::new(r1), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked.len(), 2);
}

#[test]
fn top_n_zero_returns_all() {
    let r1 = StubRetriever::new("r1", vec![("a", 1.0), ("b", 0.9), ("c", 0.8)]);
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new().with_top_n(0));
    ensemble.add_retriever(Box::new(r1), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked.len(), 3);
}

#[test]
fn top_n_larger_than_results_returns_all() {
    let r1 = StubRetriever::new("r1", vec![("a", 1.0), ("b", 0.9)]);
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new().with_top_n(100));
    ensemble.add_retriever(Box::new(r1), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked.len(), 2);
}

#[test]
fn top_n_keeps_highest_scoring() {
    let r1 = StubRetriever::new("r1", vec![("low", 0.1), ("high", 9.0), ("mid", 1.0)]);
    let mut ensemble = EnsembleRetriever::new(
        EnsembleConfig::new()
            .with_fusion(EnsembleFusion::WeightedScore)
            .with_top_n(1),
    );
    ensemble.add_retriever(Box::new(r1), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].0.as_str(), "high");
}

// ── Errors: NoRetrievers & EmptyQuery ───────────────────────────────────────

#[test]
fn no_retrievers_errors() {
    let ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    let err = ensemble.retrieve("q", 10).unwrap_err();
    assert!(matches!(err, EnsembleError::NoRetrievers));
}

#[test]
fn empty_query_errors() {
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    ensemble.add_retriever(Box::new(StubRetriever::new("r1", vec![("a", 1.0)])), 1.0);
    let err = ensemble.retrieve("", 10).unwrap_err();
    assert!(matches!(err, EnsembleError::EmptyQuery));
}

#[test]
fn whitespace_query_errors() {
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    ensemble.add_retriever(Box::new(StubRetriever::new("r1", vec![("a", 1.0)])), 1.0);
    let err = ensemble.retrieve("   \t\n", 10).unwrap_err();
    assert!(matches!(err, EnsembleError::EmptyQuery));
}

#[test]
fn no_retrievers_takes_precedence_over_empty_query() {
    // With no retrievers AND an empty query, NoRetrievers is reported first.
    let ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    let err = ensemble.retrieve("", 10).unwrap_err();
    assert!(matches!(err, EnsembleError::NoRetrievers));
}

#[test]
fn error_display_messages() {
    assert_eq!(
        EnsembleError::NoRetrievers.to_string(),
        "no sub-retrievers registered"
    );
    assert_eq!(
        EnsembleError::EmptyQuery.to_string(),
        "query must not be empty"
    );
}

// ── Determinism ─────────────────────────────────────────────────────────────

#[test]
fn retrieve_is_deterministic_across_runs() {
    let build = || {
        let mut e =
            EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
        e.add_retriever(
            Box::new(StubRetriever::new(
                "r1",
                vec![("a", 1.0), ("b", 0.9), ("c", 0.8)],
            )),
            1.0,
        );
        e.add_retriever(
            Box::new(StubRetriever::new(
                "r2",
                vec![("b", 1.0), ("c", 0.9), ("d", 0.8)],
            )),
            0.7,
        );
        e
    };
    let first = build().retrieve("query", 10).unwrap();
    let second = build().retrieve("query", 10).unwrap();
    assert_eq!(ids(&first), ids(&second));
    for ((_, s1), (_, s2)) in first.iter().zip(second.iter()) {
        assert_eq!(s1, s2);
    }
}

#[test]
fn equal_scores_break_ties_by_id_ascending() {
    // Two single-doc retrievers at identical rank/weight → equal RRF scores.
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    ensemble.add_retriever(
        Box::new(StubRetriever::new("r1", vec![("zebra", 1.0)])),
        1.0,
    );
    ensemble.add_retriever(
        Box::new(StubRetriever::new("r2", vec![("apple", 1.0)])),
        1.0,
    );
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(score_of(&ranked, "apple"), score_of(&ranked, "zebra"));
    assert_eq!(ranked[0].0.as_str(), "apple");
    assert_eq!(ranked[1].0.as_str(), "zebra");
}

#[test]
fn weighted_score_is_deterministic() {
    let build = || {
        let mut e = EnsembleRetriever::new(
            EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedScore),
        );
        e.add_retriever(
            Box::new(StubRetriever::new(
                "r1",
                vec![("a", 0.5), ("b", 0.5), ("c", 0.5)],
            )),
            1.0,
        );
        e
    };
    assert_eq!(
        ids(&build().retrieve("q", 10).unwrap()),
        ids(&build().retrieve("q", 10).unwrap())
    );
}

#[test]
fn retriever_returning_empty_is_handled() {
    // One retriever finds nothing; the ensemble still ranks the other's hits.
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    ensemble.add_retriever(Box::new(StubRetriever::new("empty", vec![])), 1.0);
    ensemble.add_retriever(Box::new(StubRetriever::new("r2", vec![("a", 1.0)])), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].0.as_str(), "a");
}

#[test]
fn all_retrievers_empty_yields_empty_ranking() {
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    ensemble.add_retriever(Box::new(StubRetriever::new("e1", vec![])), 1.0);
    ensemble.add_retriever(Box::new(StubRetriever::new("e2", vec![])), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert!(ranked.is_empty());
}

#[test]
fn top_k_is_forwarded_to_sub_retrievers() {
    let docs = vec![
        doc("a", "token here"),
        doc("b", "token there"),
        doc("c", "token everywhere"),
    ];
    let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
    ensemble.add_retriever(Box::new(LexicalSubRetriever::new("lex", docs)), 1.0);
    // top_k=1 → the lexical retriever returns at most one candidate.
    let ranked = ensemble.retrieve("token", 1).unwrap();
    assert_eq!(ranked.len(), 1);
}

#[test]
fn single_retriever_preserves_rank_order_rrf() {
    let r1 = StubRetriever::new("r1", vec![("first", 1.0), ("second", 0.5), ("third", 0.1)]);
    let mut ensemble =
        EnsembleRetriever::new(EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf));
    ensemble.add_retriever(Box::new(r1), 1.0);
    let ranked = ensemble.retrieve("q", 10).unwrap();
    assert_eq!(ids(&ranked), vec!["first", "second", "third"]);
}
