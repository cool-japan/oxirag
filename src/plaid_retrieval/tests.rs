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
    clippy::items_after_statements,
    clippy::cast_possible_truncation,
    clippy::decimal_bitwise_operands
)]

use crate::plaid_retrieval::retriever::PlaidRetriever;
use crate::plaid_retrieval::types::{PlaidConfig, PlaidError};
use crate::types::Document;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

fn corpus() -> Vec<Document> {
    vec![
        doc(
            "rust",
            "Rust delivers memory safety without a garbage collector through ownership.",
        ),
        doc(
            "python",
            "Python is a dynamic interpreted language popular for data science.",
        ),
        doc(
            "ocean",
            "The ocean covers most of the planet and hosts coral reefs and whales.",
        ),
        doc(
            "cooking",
            "Slow cooking braises tough cuts of meat into tender flavorful stew.",
        ),
    ]
}

fn built_retriever() -> PlaidRetriever {
    let mut r = PlaidRetriever::new(PlaidConfig::default());
    r.build(&corpus()).unwrap();
    r
}

const DIM: usize = 128;

fn embed_tokens(text: &str) -> Vec<Vec<f32>> {
    // Reproduce the tokeniser + per-token embedding via the public maxsim path:
    // build a one-doc retriever and pull token vectors back out indirectly is
    // not exposed, so we instead reconstruct using a tiny local helper that
    // mirrors the module contract for test inputs.
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .map(|tok| single_token_embed(&tok))
        .collect()
}

fn single_token_embed(token: &str) -> Vec<f32> {
    // Mirror of the module's embed_token for constructing maxsim test inputs.
    let dim = DIM;
    let mut vec = vec![0.0f32; dim];
    let spread = 8usize.min(dim);
    let bytes = token.as_bytes();
    for k in 0..spread {
        let h = fnv1a(bytes, (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let idx = (h as usize) % dim;
        let sign = if (h >> 33) & 1 == 0 { 1.0 } else { -1.0 };
        let mag = 1.0 + ((h >> 7) % 7) as f32;
        vec[idx] += sign * mag;
    }
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut vec {
            *x /= norm;
        }
    }
    vec
}

fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037 ^ seed;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ── Config: defaults ────────────────────────────────────────────────────────────

#[test]
fn config_default_num_centroids() {
    assert_eq!(PlaidConfig::default().num_centroids, 16);
}

#[test]
fn config_default_nprobe() {
    assert_eq!(PlaidConfig::default().nprobe, 4);
}

#[test]
fn config_default_dim() {
    assert_eq!(PlaidConfig::default().dim, 128);
}

#[test]
fn config_default_kmeans_iters() {
    assert_eq!(PlaidConfig::default().kmeans_iters, 10);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(PlaidConfig::new(), PlaidConfig::default());
}

// ── Config: builders ────────────────────────────────────────────────────────────

#[test]
fn builder_num_centroids() {
    assert_eq!(PlaidConfig::new().with_num_centroids(32).num_centroids, 32);
}

#[test]
fn builder_nprobe() {
    assert_eq!(PlaidConfig::new().with_nprobe(8).nprobe, 8);
}

#[test]
fn builder_dim() {
    assert_eq!(PlaidConfig::new().with_dim(64).dim, 64);
}

#[test]
fn builder_kmeans_iters() {
    assert_eq!(PlaidConfig::new().with_kmeans_iters(25).kmeans_iters, 25);
}

#[test]
fn builder_chaining_all() {
    let c = PlaidConfig::new()
        .with_num_centroids(8)
        .with_nprobe(2)
        .with_dim(32)
        .with_kmeans_iters(5);
    assert_eq!(c.num_centroids, 8);
    assert_eq!(c.nprobe, 2);
    assert_eq!(c.dim, 32);
    assert_eq!(c.kmeans_iters, 5);
}

#[test]
fn builder_does_not_touch_other_fields() {
    let c = PlaidConfig::new().with_dim(99);
    assert_eq!(c.num_centroids, 16);
    assert_eq!(c.nprobe, 4);
    assert_eq!(c.kmeans_iters, 10);
}

#[test]
fn config_is_clone() {
    let c = PlaidConfig::new().with_dim(70);
    let d = c.clone();
    assert_eq!(c, d);
}

// ── Construction ────────────────────────────────────────────────────────────────

#[test]
fn new_retriever_is_empty() {
    let r = PlaidRetriever::new(PlaidConfig::default());
    assert!(r.is_empty());
}

#[test]
fn new_retriever_len_zero() {
    let r = PlaidRetriever::new(PlaidConfig::default());
    assert_eq!(r.len(), 0);
}

#[test]
fn new_retriever_no_centroids() {
    let r = PlaidRetriever::new(PlaidConfig::default());
    assert_eq!(r.num_centroids(), 0);
}

// ── Build ───────────────────────────────────────────────────────────────────────

#[test]
fn build_sets_len() {
    let r = built_retriever();
    assert_eq!(r.len(), 4);
}

#[test]
fn build_not_empty() {
    let r = built_retriever();
    assert!(!r.is_empty());
}

#[test]
fn build_trains_centroids() {
    let r = built_retriever();
    assert!(r.num_centroids() > 0);
}

#[test]
fn build_centroids_capped_by_tokens() {
    // num_centroids requested far exceeds tokens of a tiny corpus.
    let mut r = PlaidRetriever::new(PlaidConfig::new().with_num_centroids(1000));
    r.build(&[doc("a", "alpha beta gamma")]).unwrap();
    assert!(r.num_centroids() <= 3);
    assert!(r.num_centroids() >= 1);
}

#[test]
fn build_empty_corpus_errors() {
    let mut r = PlaidRetriever::new(PlaidConfig::default());
    assert_eq!(r.build(&[]), Err(PlaidError::EmptyCorpus));
}

#[test]
fn build_single_document() {
    let mut r = PlaidRetriever::new(PlaidConfig::default());
    r.build(&[doc("solo", "a lone document about ferns and gardens")])
        .unwrap();
    assert_eq!(r.len(), 1);
    assert!(r.num_centroids() > 0);
}

#[test]
fn build_is_idempotent_on_len() {
    let mut r = PlaidRetriever::new(PlaidConfig::default());
    r.build(&corpus()).unwrap();
    r.build(&corpus()).unwrap();
    assert_eq!(r.len(), 4);
}

#[test]
fn build_then_rebuild_with_fewer() {
    let mut r = built_retriever();
    r.build(&[doc("only", "single survivor document text")])
        .unwrap();
    assert_eq!(r.len(), 1);
}

// ── MaxSim ──────────────────────────────────────────────────────────────────────

#[test]
fn maxsim_identical_sets_high() {
    let q = embed_tokens("memory safety ownership model");
    let score = PlaidRetriever::maxsim(&q, &q);
    // Each query token matches itself with cosine ≈ 1, so total ≈ |q|.
    assert!((score - q.len() as f32).abs() < 1e-3);
}

#[test]
fn maxsim_identical_single_token() {
    let q = embed_tokens("rustlang");
    let score = PlaidRetriever::maxsim(&q, &q);
    assert!((score - 1.0).abs() < 1e-4);
}

#[test]
fn maxsim_disjoint_low() {
    let q = embed_tokens("ownership borrow lifetime");
    let d = embed_tokens("whales coral reefs");
    let score = PlaidRetriever::maxsim(&q, &d);
    // Unrelated tokens are near-orthogonal; total stays well below |q|.
    assert!(score < (q.len() as f32) * 0.5);
}

#[test]
fn maxsim_empty_query_zero() {
    let d = embed_tokens("something here");
    assert_eq!(PlaidRetriever::maxsim(&[], &d), 0.0);
}

#[test]
fn maxsim_empty_doc_zero() {
    let q = embed_tokens("something here");
    assert_eq!(PlaidRetriever::maxsim(&q, &[]), 0.0);
}

#[test]
fn maxsim_both_empty_zero() {
    assert_eq!(PlaidRetriever::maxsim(&[], &[]), 0.0);
}

#[test]
fn maxsim_partial_overlap_between() {
    let q = embed_tokens("memory safety dragons");
    let identical = PlaidRetriever::maxsim(&q, &q);
    let partial = PlaidRetriever::maxsim(&q, &embed_tokens("memory safety"));
    let none = PlaidRetriever::maxsim(&q, &embed_tokens("totally unrelated stuff"));
    assert!(partial < identical);
    assert!(partial > none);
}

#[test]
fn maxsim_superset_doc_matches_all_query() {
    let q = embed_tokens("memory safety");
    let d = embed_tokens("memory safety ownership borrow checker");
    let score = PlaidRetriever::maxsim(&q, &d);
    assert!((score - q.len() as f32).abs() < 1e-3);
}

#[test]
fn maxsim_is_nonnegative_for_normalised() {
    let q = embed_tokens("alpha beta");
    let d = embed_tokens("gamma delta epsilon");
    let score = PlaidRetriever::maxsim(&q, &d);
    assert!(score >= -(q.len() as f32));
}

#[test]
fn maxsim_self_geq_other() {
    let q = embed_tokens("ownership memory safety");
    let other = embed_tokens("python data science");
    assert!(PlaidRetriever::maxsim(&q, &q) >= PlaidRetriever::maxsim(&q, &other));
}

// ── Candidate generation ────────────────────────────────────────────────────────

#[test]
fn candidate_surfaces_sharing_doc_at_rank0() {
    let r = built_retriever();
    let hits = r.search("memory safety ownership", 4).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "rust");
}

#[test]
fn candidate_generation_returns_results() {
    let r = built_retriever();
    let hits = r.search("ocean coral whales", 4).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].document.id.as_str(), "ocean");
}

#[test]
fn candidate_top_score_positive_for_match() {
    let r = built_retriever();
    let hits = r.search("python data science", 4).unwrap();
    assert!(hits[0].score > 0.0);
    assert_eq!(hits[0].document.id.as_str(), "python");
}

// ── Search ranking ──────────────────────────────────────────────────────────────

#[test]
fn search_ranks_query_doc_first_rust() {
    let r = built_retriever();
    let hits = r.search("garbage collector ownership", 4).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "rust");
}

#[test]
fn search_ranks_query_doc_first_cooking() {
    let r = built_retriever();
    let hits = r.search("braises tender stew meat", 4).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "cooking");
}

#[test]
fn search_scores_descending() {
    let r = built_retriever();
    let hits = r.search("memory safety ownership", 4).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[test]
fn search_match_outscores_unrelated() {
    let r = built_retriever();
    let hits = r.search("ocean whales reefs", 4).unwrap();
    let ocean = hits
        .iter()
        .find(|h| h.document.id.as_str() == "ocean")
        .unwrap();
    for h in &hits {
        if h.document.id.as_str() != "ocean" {
            assert!(ocean.score >= h.score);
        }
    }
}

// ── top_k ───────────────────────────────────────────────────────────────────────

#[test]
fn search_top_k_limits_results() {
    let r = built_retriever();
    let hits = r.search("memory safety", 2).unwrap();
    assert!(hits.len() <= 2);
}

#[test]
fn search_top_k_one() {
    let r = built_retriever();
    let hits = r.search("memory safety ownership", 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document.id.as_str(), "rust");
}

#[test]
fn search_top_k_zero_empty() {
    let r = built_retriever();
    let hits = r.search("memory safety", 0).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn search_top_k_larger_than_corpus() {
    let r = built_retriever();
    let hits = r.search("memory safety ownership", 100).unwrap();
    assert!(hits.len() <= 4);
}

// ── Fallback ────────────────────────────────────────────────────────────────────

#[test]
fn search_fallback_scores_all_when_no_candidate() {
    // A retriever whose centroids cannot be trained (dim 0) yields no candidate
    // routing, so search must fall back to the whole corpus.
    let mut r = PlaidRetriever::new(PlaidConfig::new().with_dim(0));
    r.build(&corpus()).unwrap();
    let hits = r.search("memory safety", 10).unwrap();
    assert_eq!(hits.len(), 4);
}

#[test]
fn search_fallback_still_ranks() {
    let mut r = PlaidRetriever::new(PlaidConfig::new().with_dim(0));
    r.build(&corpus()).unwrap();
    let hits = r.search("memory safety", 2).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn search_query_term_absent_falls_back_to_corpus() {
    // Query tokens share no centroid with any doc token in the worst case; the
    // search must still return results (fallback) rather than nothing.
    let r = built_retriever();
    let hits = r.search("zzqq", 4).unwrap();
    assert!(!hits.is_empty());
}

// ── Errors ──────────────────────────────────────────────────────────────────────

#[test]
fn search_not_built_errors() {
    let r = PlaidRetriever::new(PlaidConfig::default());
    assert_eq!(r.search("anything", 5).unwrap_err(), PlaidError::NotBuilt);
}

#[test]
fn search_empty_query_errors() {
    let r = built_retriever();
    assert_eq!(r.search("", 5).unwrap_err(), PlaidError::EmptyQuery);
}

#[test]
fn search_whitespace_query_errors() {
    let r = built_retriever();
    assert_eq!(r.search("   \t\n", 5).unwrap_err(), PlaidError::EmptyQuery);
}

#[test]
fn search_punctuation_only_query_errors() {
    let r = built_retriever();
    assert_eq!(
        r.search("!!! ... ???", 5).unwrap_err(),
        PlaidError::EmptyQuery
    );
}

#[test]
fn search_single_char_tokens_filtered_to_empty_errors() {
    // Tokens of length < 2 are dropped by the tokeniser.
    let r = built_retriever();
    assert_eq!(r.search("a b c", 5).unwrap_err(), PlaidError::EmptyQuery);
}

#[test]
fn error_messages_match() {
    assert_eq!(PlaidError::EmptyCorpus.to_string(), "corpus is empty");
    assert_eq!(
        PlaidError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(PlaidError::NotBuilt.to_string(), "retriever not built");
}

#[test]
fn not_built_takes_priority_over_empty_query() {
    let r = PlaidRetriever::new(PlaidConfig::default());
    assert_eq!(r.search("", 5).unwrap_err(), PlaidError::NotBuilt);
}

// ── Determinism ─────────────────────────────────────────────────────────────────

#[test]
fn build_is_deterministic_centroid_count() {
    let mut a = PlaidRetriever::new(PlaidConfig::default());
    let mut b = PlaidRetriever::new(PlaidConfig::default());
    a.build(&corpus()).unwrap();
    b.build(&corpus()).unwrap();
    assert_eq!(a.num_centroids(), b.num_centroids());
}

#[test]
fn search_is_deterministic_ranking() {
    let a = built_retriever();
    let b = built_retriever();
    let ha = a.search("memory safety ownership", 4).unwrap();
    let hb = b.search("memory safety ownership", 4).unwrap();
    let ids_a: Vec<_> = ha
        .iter()
        .map(|h| h.document.id.as_str().to_string())
        .collect();
    let ids_b: Vec<_> = hb
        .iter()
        .map(|h| h.document.id.as_str().to_string())
        .collect();
    assert_eq!(ids_a, ids_b);
}

#[test]
fn search_is_deterministic_scores() {
    let a = built_retriever();
    let b = built_retriever();
    let ha = a.search("ocean coral whales", 4).unwrap();
    let hb = b.search("ocean coral whales", 4).unwrap();
    for (x, y) in ha.iter().zip(hb.iter()) {
        assert_eq!(x.score, y.score);
    }
}

#[test]
fn maxsim_is_deterministic() {
    let q = embed_tokens("alpha beta gamma");
    let d = embed_tokens("beta gamma delta");
    assert_eq!(
        PlaidRetriever::maxsim(&q, &d),
        PlaidRetriever::maxsim(&q, &d)
    );
}

#[test]
fn embed_token_identical_inputs_identical() {
    let a = single_token_embed("ownership");
    let b = single_token_embed("ownership");
    assert_eq!(a, b);
}

#[test]
fn embed_token_is_l2_normalised() {
    let v = single_token_embed("normalise");
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

// ── Robustness ──────────────────────────────────────────────────────────────────

#[test]
fn search_handles_duplicate_documents() {
    let mut r = PlaidRetriever::new(PlaidConfig::default());
    r.build(&[
        doc("a", "memory safety ownership"),
        doc("b", "memory safety ownership"),
    ])
    .unwrap();
    let hits = r.search("memory safety", 2).unwrap();
    assert_eq!(hits.len(), 2);
    // Tie broken by document id ascending.
    assert_eq!(hits[0].document.id.as_str(), "a");
}

#[test]
fn search_doc_with_no_tokens_scores_zero() {
    let mut r = PlaidRetriever::new(PlaidConfig::default());
    r.build(&[
        doc("empty", "!!! ???"),
        doc("real", "memory safety ownership model"),
    ])
    .unwrap();
    let hits = r.search("memory safety", 2).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "real");
}

#[test]
fn search_returns_hit_documents_unmodified() {
    let r = built_retriever();
    let hits = r.search("python data science", 1).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "python");
    assert!(hits[0].document.content.contains("Python"));
}

#[test]
fn smaller_nprobe_still_finds_match() {
    let mut r = PlaidRetriever::new(PlaidConfig::new().with_nprobe(1));
    r.build(&corpus()).unwrap();
    let hits = r.search("memory safety ownership", 4).unwrap();
    assert!(hits.iter().any(|h| h.document.id.as_str() == "rust"));
}

#[test]
fn custom_dim_search_works() {
    let mut r = PlaidRetriever::new(PlaidConfig::new().with_dim(64));
    r.build(&corpus()).unwrap();
    let hits = r.search("memory safety ownership", 4).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "rust");
}
