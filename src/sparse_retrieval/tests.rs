#![allow(clippy::float_cmp, clippy::similar_names)]
//! Tests for SPLADE-style learned sparse retrieval.

use std::collections::HashMap;

use crate::sparse_retrieval::encoder::{SparseEncoder, tokenize};
use crate::sparse_retrieval::types::{SparseConfig, SparseHit, SparseRetrievalError, SparseVector};
use crate::types::{Document, DocumentId};

// ── Fixtures ──────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

fn corpus() -> Vec<Document> {
    vec![
        doc(
            "d1",
            "Rust delivers memory safety without a garbage collector.",
        ),
        doc(
            "d2",
            "The borrow checker enforces memory safety at compile time.",
        ),
        doc(
            "d3",
            "Python is a dynamic scripting language for data science.",
        ),
        doc(
            "d4",
            "Garbage collection reclaims memory automatically in Java.",
        ),
        doc("d5", "Concurrency in Rust is fearless thanks to ownership."),
    ]
}

fn vec_from(pairs: &[(&str, f32)]) -> SparseVector {
    let mut terms = HashMap::new();
    for (term, weight) in pairs {
        terms.insert((*term).to_string(), *weight);
    }
    SparseVector::new(terms)
}

// ── Tokeniser ─────────────────────────────────────────────────────────────────

#[test]
fn tokenize_lowercases_and_filters_short_tokens() {
    let toks = tokenize("Rust IS a Fast, Safe language!");
    assert!(toks.contains(&"rust".to_string()));
    assert!(toks.contains(&"fast".to_string()));
    assert!(toks.contains(&"safe".to_string()));
    // "is" has length 2 so is retained; "a" is dropped.
    assert!(toks.contains(&"is".to_string()));
    assert!(!toks.contains(&"a".to_string()));
}

#[test]
fn tokenize_empty_text_yields_nothing() {
    assert!(tokenize("").is_empty());
    assert!(tokenize("  , . ! ").is_empty());
}

// ── SparseConfig defaults & builders ──────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = SparseConfig::default();
    assert_eq!(cfg.expansion_terms, 3);
    assert_eq!(cfg.saturation, 1.0);
    assert_eq!(cfg.min_weight, 0.01);
}

#[test]
fn config_new_matches_default() {
    assert_eq!(SparseConfig::new(), SparseConfig::default());
}

#[test]
fn config_builder_expansion_terms() {
    let cfg = SparseConfig::new().with_expansion_terms(7);
    assert_eq!(cfg.expansion_terms, 7);
    // Other fields unchanged.
    assert_eq!(cfg.saturation, 1.0);
    assert_eq!(cfg.min_weight, 0.01);
}

#[test]
fn config_builder_saturation() {
    let cfg = SparseConfig::new().with_saturation(2.5);
    assert_eq!(cfg.saturation, 2.5);
}

#[test]
fn config_builder_min_weight() {
    let cfg = SparseConfig::new().with_min_weight(0.25);
    assert_eq!(cfg.min_weight, 0.25);
}

#[test]
fn config_builders_chain() {
    let cfg = SparseConfig::new()
        .with_expansion_terms(2)
        .with_saturation(0.5)
        .with_min_weight(0.1);
    assert_eq!(cfg.expansion_terms, 2);
    assert_eq!(cfg.saturation, 0.5);
    assert_eq!(cfg.min_weight, 0.1);
}

// ── SparseVector ──────────────────────────────────────────────────────────────

#[test]
fn vector_default_is_empty() {
    let v = SparseVector::default();
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
}

#[test]
fn vector_len_and_is_empty() {
    let v = vec_from(&[("rust", 1.0), ("safe", 2.0)]);
    assert_eq!(v.len(), 2);
    assert!(!v.is_empty());
}

#[test]
fn vector_dot_identical_is_sum_of_squares() {
    let v = vec_from(&[("a", 2.0), ("b", 3.0), ("c", 1.5)]);
    let expected = 2.0 * 2.0 + 3.0 * 3.0 + 1.5 * 1.5;
    assert_eq!(v.dot(&v), expected);
}

#[test]
fn vector_dot_disjoint_is_zero() {
    let a = vec_from(&[("a", 2.0), ("b", 3.0)]);
    let b = vec_from(&[("c", 4.0), ("d", 5.0)]);
    assert_eq!(a.dot(&b), 0.0);
}

#[test]
fn vector_dot_partial_overlap() {
    let a = vec_from(&[("a", 2.0), ("shared", 3.0)]);
    let b = vec_from(&[("shared", 4.0), ("z", 5.0)]);
    assert_eq!(a.dot(&b), 3.0 * 4.0);
}

#[test]
fn vector_dot_is_symmetric() {
    let a = vec_from(&[("a", 2.0), ("b", 3.0), ("c", 4.0)]);
    let b = vec_from(&[("b", 1.0), ("c", 2.0), ("d", 9.0)]);
    assert_eq!(a.dot(&b), b.dot(&a));
}

#[test]
fn vector_dot_with_empty_is_zero() {
    let a = vec_from(&[("a", 2.0)]);
    let empty = SparseVector::default();
    assert_eq!(a.dot(&empty), 0.0);
    assert_eq!(empty.dot(&a), 0.0);
}

#[test]
fn vector_top_terms_sorted_desc() {
    let v = vec_from(&[("low", 0.5), ("high", 3.0), ("mid", 1.5)]);
    let top = v.top_terms(3);
    assert_eq!(top[0].0, "high");
    assert_eq!(top[1].0, "mid");
    assert_eq!(top[2].0, "low");
}

#[test]
fn vector_top_terms_truncates() {
    let v = vec_from(&[("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]);
    let top = v.top_terms(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].0, "d");
    assert_eq!(top[1].0, "c");
}

#[test]
fn vector_top_terms_tie_break_alpha() {
    let v = vec_from(&[("zebra", 1.0), ("apple", 1.0), ("mango", 1.0)]);
    let top = v.top_terms(3);
    // Equal weights => alphabetical order.
    assert_eq!(top[0].0, "apple");
    assert_eq!(top[1].0, "mango");
    assert_eq!(top[2].0, "zebra");
}

#[test]
fn vector_top_terms_more_than_len() {
    let v = vec_from(&[("a", 1.0), ("b", 2.0)]);
    let top = v.top_terms(10);
    assert_eq!(top.len(), 2);
}

#[test]
fn vector_top_terms_on_empty() {
    let v = SparseVector::default();
    assert!(v.top_terms(5).is_empty());
}

#[test]
fn vector_l2_norm() {
    let v = vec_from(&[("a", 3.0), ("b", 4.0)]);
    // sqrt(9 + 16) = 5.
    assert!((v.l2_norm() - 5.0).abs() < 1e-6);
}

#[test]
fn vector_l2_norm_empty_is_zero() {
    assert_eq!(SparseVector::default().l2_norm(), 0.0);
}

// ── Encoder: fitted state ─────────────────────────────────────────────────────

#[test]
fn encoder_starts_unfitted() {
    let enc = SparseEncoder::new(SparseConfig::default());
    assert!(!enc.is_fitted());
}

#[test]
fn encoder_fit_sets_fitted() {
    let mut enc = SparseEncoder::new(SparseConfig::default());
    enc.fit(&corpus());
    assert!(enc.is_fitted());
}

#[test]
fn encoder_exposes_config() {
    let cfg = SparseConfig::new().with_expansion_terms(5);
    let enc = SparseEncoder::new(cfg);
    assert_eq!(enc.config().expansion_terms, 5);
}

// ── Encoder: IDF after fit ────────────────────────────────────────────────────

#[test]
fn idf_computed_after_fit() {
    let mut enc = SparseEncoder::new(SparseConfig::default());
    enc.fit(&corpus());
    // "memory" appears in 3 of 5 docs; "python" in 1 of 5.
    let idf_memory = enc.idf_of("memory").expect("memory idf");
    let idf_python = enc.idf_of("python").expect("python idf");
    // Rarer term => higher IDF.
    assert!(idf_python > idf_memory);
}

#[test]
fn idf_formula_matches_smoothed_definition() {
    let mut enc = SparseEncoder::new(SparseConfig::default());
    enc.fit(&corpus());
    // N = 5, "python" df = 1 => ln(6/2) + 1.
    let expected = ((5.0f32 + 1.0) / (1.0 + 1.0)).ln() + 1.0;
    let idf_python = enc.idf_of("python").expect("python idf");
    assert!((idf_python - expected).abs() < 1e-5);
}

#[test]
fn idf_unknown_term_is_none() {
    let mut enc = SparseEncoder::new(SparseConfig::default());
    enc.fit(&corpus());
    assert!(enc.idf_of("nonexistentterm").is_none());
}

// ── Encoder: encode unfitted vs fitted ────────────────────────────────────────

#[test]
fn encode_unfitted_uses_default_idf_no_expansion() {
    let enc = SparseEncoder::new(SparseConfig::default());
    let v = enc.encode("rust safety");
    // Both literal tokens present, weight = ln(1 + 1*1*1) = ln(2).
    let expected = (1.0f32 + 1.0).ln();
    assert!((v.terms["rust"] - expected).abs() < 1e-6);
    assert!((v.terms["safety"] - expected).abs() < 1e-6);
    // No expansion when unfitted => exactly the two literal tokens.
    assert_eq!(v.len(), 2);
}

#[test]
fn encode_empty_text_is_empty_vector() {
    let mut enc = SparseEncoder::new(SparseConfig::default());
    enc.fit(&corpus());
    assert!(enc.encode("").is_empty());
    assert!(enc.encode(" , . ").is_empty());
}

#[test]
fn encode_fitted_weights_scale_with_idf() {
    let mut enc = SparseEncoder::new(SparseConfig::default());
    enc.fit(&corpus());
    let v = enc.encode("memory python");
    // Both appear once; python is rarer => higher idf => higher base weight.
    assert!(v.terms["python"] > v.terms["memory"]);
}

// ── Encoder: saturation monotonic & sublinear ─────────────────────────────────

#[test]
fn saturation_monotonic_in_tf() {
    // Use no expansion to isolate the base weight behaviour.
    let cfg = SparseConfig::new().with_expansion_terms(0);
    let mut enc = SparseEncoder::new(cfg);
    enc.fit(&corpus());
    let single = enc.encode("memory");
    let triple = enc.encode("memory memory memory");
    assert!(triple.terms["memory"] > single.terms["memory"]);
}

#[test]
fn saturation_is_sublinear() {
    let cfg = SparseConfig::new().with_expansion_terms(0);
    let mut enc = SparseEncoder::new(cfg);
    enc.fit(&corpus());
    let one = enc.encode("memory").terms["memory"];
    let three = enc.encode("memory memory memory").terms["memory"];
    // Sublinear: tripling tf must not triple the weight.
    assert!(three < 3.0 * one);
}

#[test]
fn saturation_multiplier_increases_weight() {
    let mut low = SparseEncoder::new(
        SparseConfig::new()
            .with_saturation(0.5)
            .with_expansion_terms(0),
    );
    let mut high = SparseEncoder::new(
        SparseConfig::new()
            .with_saturation(4.0)
            .with_expansion_terms(0),
    );
    low.fit(&corpus());
    high.fit(&corpus());
    let w_low = low.encode("memory").terms["memory"];
    let w_high = high.encode("memory").terms["memory"];
    assert!(w_high > w_low);
}

// ── Encoder: expansion ────────────────────────────────────────────────────────

#[test]
fn expansion_adds_extra_terms() {
    // No expansion: only literal "rust" survives (low min_weight to keep it).
    let mut bare = SparseEncoder::new(
        SparseConfig::new()
            .with_expansion_terms(0)
            .with_min_weight(0.0),
    );
    bare.fit(&corpus());
    let bare_v = bare.encode("rust");

    // With expansion: extra co-occurring terms appear.
    let mut expanded = SparseEncoder::new(
        SparseConfig::new()
            .with_expansion_terms(3)
            .with_min_weight(0.0),
    );
    expanded.fit(&corpus());
    let expanded_v = expanded.encode("rust");

    assert!(expanded_v.len() > bare_v.len());
    assert!(expanded_v.terms.contains_key("rust"));
}

#[test]
fn expansion_terms_are_discounted() {
    let mut enc = SparseEncoder::new(
        SparseConfig::new()
            .with_expansion_terms(5)
            .with_min_weight(0.0),
    );
    enc.fit(&corpus());
    let v = enc.encode("rust");
    let base = v.terms["rust"];
    // Every expansion term must be <= the source term's weight (strength <= 1).
    for (term, weight) in &v.terms {
        if term != "rust" {
            assert!(*weight <= base + 1e-6, "term {term} not discounted");
        }
    }
}

#[test]
fn expansion_zero_keeps_only_literal_terms() {
    let mut enc = SparseEncoder::new(
        SparseConfig::new()
            .with_expansion_terms(0)
            .with_min_weight(0.0),
    );
    enc.fit(&corpus());
    let v = enc.encode("memory safety");
    let mut keys: Vec<&String> = v.terms.keys().collect();
    keys.sort();
    assert_eq!(keys, vec![&"memory".to_string(), &"safety".to_string()]);
}

#[test]
fn expansion_respects_term_budget() {
    // Limit to a single expansion term per source term.
    let mut enc = SparseEncoder::new(
        SparseConfig::new()
            .with_expansion_terms(1)
            .with_min_weight(0.0),
    );
    enc.fit(&corpus());
    // "python" co-occurs with several terms in d3 but only 1 should be added.
    let v = enc.encode("python");
    // python itself + at most 1 expansion term.
    assert!(v.len() <= 2);
    assert!(v.terms.contains_key("python"));
}

// ── Encoder: min_weight pruning ───────────────────────────────────────────────

#[test]
fn min_weight_prunes_low_weights() {
    // A very high min_weight prunes everything.
    let mut enc = SparseEncoder::new(SparseConfig::new().with_min_weight(1000.0));
    enc.fit(&corpus());
    let v = enc.encode("memory safety");
    assert!(v.is_empty());
}

#[test]
fn min_weight_keeps_strong_weights() {
    let mut strict = SparseEncoder::new(
        SparseConfig::new()
            .with_min_weight(0.5)
            .with_expansion_terms(3),
    );
    strict.fit(&corpus());
    let v = strict.encode("python");
    // Every surviving weight must clear the threshold.
    for weight in v.terms.values() {
        assert!(*weight >= 0.5);
    }
}

#[test]
fn min_weight_higher_prunes_more() {
    let mut loose = SparseEncoder::new(
        SparseConfig::new()
            .with_min_weight(0.0)
            .with_expansion_terms(3),
    );
    let mut tight = SparseEncoder::new(
        SparseConfig::new()
            .with_min_weight(0.4)
            .with_expansion_terms(3),
    );
    loose.fit(&corpus());
    tight.fit(&corpus());
    let loose_v = loose.encode("memory");
    let tight_v = tight.encode("memory");
    assert!(tight_v.len() <= loose_v.len());
}

// ── Index: build / len / is_empty ─────────────────────────────────────────────

#[test]
fn index_new_is_empty() {
    let index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn index_build_populates_entries() {
    let mut index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    index.build(&corpus()).expect("build");
    assert_eq!(index.len(), 5);
    assert!(!index.is_empty());
    assert!(index.encoder().is_fitted());
}

#[test]
fn index_build_empty_corpus_errors() {
    let mut index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    let err = index.build(&[]).expect_err("empty corpus");
    assert_eq!(err, SparseRetrievalError::EmptyCorpus);
    assert!(index.is_empty());
}

// ── Index: search ranking ─────────────────────────────────────────────────────

#[test]
fn search_ranks_exact_match_first() {
    let mut index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    index.build(&corpus()).expect("build");
    let hits = index
        .search("dynamic scripting language", 5)
        .expect("search");
    assert!(!hits.is_empty());
    // d3 is the only doc containing "dynamic scripting language".
    assert_eq!(hits[0].id, DocumentId::from_string("d3"));
}

#[test]
fn search_top_k_truncates() {
    let mut index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    index.build(&corpus()).expect("build");
    let hits = index.search("memory", 2).expect("search");
    assert!(hits.len() <= 2);
}

#[test]
fn search_scores_sorted_descending() {
    let mut index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    index.build(&corpus()).expect("build");
    let hits = index.search("memory safety rust", 5).expect("search");
    for pair in hits.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

#[test]
fn search_returns_hit_type() {
    let mut index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    index.build(&corpus()).expect("build");
    let hits: Vec<SparseHit> = index.search("rust ownership", 3).expect("search");
    // The Rust-concurrency document should rank for "ownership".
    assert!(hits.iter().any(|h| h.id == DocumentId::from_string("d5")));
}

#[test]
fn search_unrelated_query_scores_low() {
    let mut index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    index.build(&corpus()).expect("build");
    // A token that never appears in the corpus.
    let hits = index.search("quantum", 5).expect("search");
    // Either no hits clear scoring, or all are zero.
    assert!(hits.iter().all(|h| h.score == 0.0));
}

// ── Index: error cases ────────────────────────────────────────────────────────

#[test]
fn search_before_build_errors_not_built() {
    let index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    let err = index.search("anything", 3).expect_err("not built");
    assert_eq!(err, SparseRetrievalError::NotBuilt);
}

#[test]
fn search_empty_query_errors() {
    let mut index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    index.build(&corpus()).expect("build");
    let err = index.search("   ", 3).expect_err("empty query");
    assert_eq!(err, SparseRetrievalError::EmptyQuery);
}

#[test]
fn empty_query_checked_before_not_built() {
    // Even on an unbuilt index, a blank query reports EmptyQuery first.
    let index = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    let err = index.search("", 3).expect_err("empty query");
    assert_eq!(err, SparseRetrievalError::EmptyQuery);
}

#[test]
fn error_display_messages() {
    assert_eq!(
        SparseRetrievalError::EmptyCorpus.to_string(),
        "corpus is empty"
    );
    assert_eq!(
        SparseRetrievalError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(
        SparseRetrievalError::NotBuilt.to_string(),
        "index not built"
    );
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn encode_is_deterministic() {
    let mut a = SparseEncoder::new(SparseConfig::default());
    let mut b = SparseEncoder::new(SparseConfig::default());
    a.fit(&corpus());
    b.fit(&corpus());
    let va = a.encode("rust memory safety");
    let vb = b.encode("rust memory safety");
    assert_eq!(va.terms, vb.terms);
}

#[test]
fn search_is_deterministic() {
    let mut index_a = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    let mut index_b = crate::sparse_retrieval::SparseIndex::new(SparseConfig::default());
    index_a.build(&corpus()).expect("build a");
    index_b.build(&corpus()).expect("build b");
    let hits_a = index_a.search("memory safety rust", 5).expect("search a");
    let hits_b = index_b.search("memory safety rust", 5).expect("search b");
    assert_eq!(hits_a.len(), hits_b.len());
    for (ha, hb) in hits_a.iter().zip(hits_b.iter()) {
        // Ranking is fully deterministic; scores match up to the float
        // summation order of the sparse dot product.
        assert_eq!(ha.id, hb.id);
        assert!((ha.score - hb.score).abs() < 1e-5);
    }
}

#[test]
fn fit_does_not_depend_on_corpus_ordering_for_idf() {
    let mut forward = SparseEncoder::new(SparseConfig::default());
    let mut reversed = SparseEncoder::new(SparseConfig::default());
    let mut docs = corpus();
    forward.fit(&docs);
    docs.reverse();
    reversed.fit(&docs);
    // IDF is order-independent.
    assert_eq!(forward.idf_of("memory"), reversed.idf_of("memory"));
    assert_eq!(forward.idf_of("python"), reversed.idf_of("python"));
}
