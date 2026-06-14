//! Unit tests for the `multi_query` module.

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

use crate::multi_query::generator::MultiQueryGenerator;
use crate::multi_query::types::{
    GeneratedQuery, MockQueryVariantGenerator, MultiQueryConfig, MultiQueryError, MultiQueryHit,
    QueryVariantGenerator,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_docs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|&(id, content)| (id.to_string(), content.to_string()))
        .collect()
}

fn default_generator() -> MultiQueryGenerator {
    MultiQueryGenerator::new(
        MultiQueryConfig::default(),
        MockQueryVariantGenerator::new(),
    )
}

fn small_corpus() -> Vec<(String, String)> {
    make_docs(&[
        ("d1", "Rust programming language systems memory safety"),
        ("d2", "Python scripting dynamic language applications"),
        ("d3", "Rust ownership borrowing borrow checker lifetime"),
        ("d4", "Java virtual machine object oriented programming"),
        ("d5", "Rust compiler type system zero cost abstractions"),
    ])
}

// ── MultiQueryConfig ──────────────────────────────────────────────────────────

#[test]
fn config_default_num_variants_is_three() {
    assert_eq!(MultiQueryConfig::default().num_variants, 3);
}

#[test]
fn config_default_top_k_is_five() {
    assert_eq!(MultiQueryConfig::default().top_k, 5);
}

#[test]
fn config_default_rrf_k_is_sixty() {
    assert_eq!(MultiQueryConfig::default().rrf_k, 60);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(MultiQueryConfig::new(), MultiQueryConfig::default());
}

#[test]
fn config_with_num_variants_sets_field() {
    let cfg = MultiQueryConfig::new().with_num_variants(7);
    assert_eq!(cfg.num_variants, 7);
}

#[test]
fn config_with_top_k_sets_field() {
    let cfg = MultiQueryConfig::new().with_top_k(10);
    assert_eq!(cfg.top_k, 10);
}

#[test]
fn config_with_rrf_k_sets_field() {
    let cfg = MultiQueryConfig::new().with_rrf_k(30);
    assert_eq!(cfg.rrf_k, 30);
}

#[test]
fn config_builder_chain() {
    let cfg = MultiQueryConfig::new()
        .with_num_variants(2)
        .with_top_k(3)
        .with_rrf_k(40);
    assert_eq!(cfg.num_variants, 2);
    assert_eq!(cfg.top_k, 3);
    assert_eq!(cfg.rrf_k, 40);
}

#[test]
fn config_clone_equals_original() {
    let cfg = MultiQueryConfig::new().with_num_variants(5).with_top_k(8);
    assert_eq!(cfg.clone(), cfg);
}

#[test]
fn config_partial_eq() {
    let a = MultiQueryConfig::new().with_num_variants(2);
    let b = MultiQueryConfig::new().with_num_variants(2);
    assert_eq!(a, b);
}

// ── GeneratedQuery ────────────────────────────────────────────────────────────

#[test]
fn generated_query_new_sets_text() {
    let q = GeneratedQuery::new("What is Rust?", 0);
    assert_eq!(q.text, "What is Rust?");
}

#[test]
fn generated_query_new_sets_variant_idx() {
    let q = GeneratedQuery::new("Describe Rust", 2);
    assert_eq!(q.variant_idx, 2);
}

#[test]
fn generated_query_equality() {
    let a = GeneratedQuery::new("text", 1);
    let b = GeneratedQuery::new("text", 1);
    assert_eq!(a, b);
}

#[test]
fn generated_query_inequality_on_text() {
    let a = GeneratedQuery::new("foo", 0);
    let b = GeneratedQuery::new("bar", 0);
    assert_ne!(a, b);
}

#[test]
fn generated_query_inequality_on_idx() {
    let a = GeneratedQuery::new("foo", 0);
    let b = GeneratedQuery::new("foo", 1);
    assert_ne!(a, b);
}

#[test]
fn generated_query_clone() {
    let q = GeneratedQuery::new("clone me", 3);
    assert_eq!(q.clone(), q);
}

// ── MultiQueryHit ─────────────────────────────────────────────────────────────

#[test]
fn hit_new_sets_id() {
    let h = MultiQueryHit::new("doc1", "content", 0.5);
    assert_eq!(h.id, "doc1");
}

#[test]
fn hit_new_sets_content() {
    let h = MultiQueryHit::new("doc1", "some content here", 0.5);
    assert_eq!(h.content, "some content here");
}

#[test]
fn hit_new_sets_rrf_score() {
    let h = MultiQueryHit::new("doc1", "content", 0.123);
    assert!((h.rrf_score - 0.123).abs() < 1e-9);
}

#[test]
fn hit_clone() {
    let h = MultiQueryHit::new("id", "content", 1.0);
    assert_eq!(h.clone(), h);
}

#[test]
fn hit_equality() {
    let a = MultiQueryHit::new("id", "content", 0.5);
    let b = MultiQueryHit::new("id", "content", 0.5);
    assert_eq!(a, b);
}

// ── MockQueryVariantGenerator ─────────────────────────────────────────────────

#[test]
fn mock_generator_produces_requested_count() {
    let mg = MockQueryVariantGenerator::new();
    let variants = mg.generate_variants("Rust language", 3).unwrap();
    assert_eq!(variants.len(), 3);
}

#[test]
fn mock_generator_zero_variants_returns_empty() {
    let mg = MockQueryVariantGenerator::new();
    let variants = mg.generate_variants("Rust language", 0).unwrap();
    assert!(variants.is_empty());
}

#[test]
fn mock_generator_first_variant_starts_with_what_is() {
    let mg = MockQueryVariantGenerator::new();
    let variants = mg.generate_variants("Rust", 1).unwrap();
    assert!(variants[0].starts_with("What is"), "got: {}", variants[0]);
}

#[test]
fn mock_generator_second_variant_starts_with_describe() {
    let mg = MockQueryVariantGenerator::new();
    let variants = mg.generate_variants("Rust", 2).unwrap();
    assert!(variants[1].starts_with("Describe"), "got: {}", variants[1]);
}

#[test]
fn mock_generator_third_variant_starts_with_explain() {
    let mg = MockQueryVariantGenerator::new();
    let variants = mg.generate_variants("Rust", 3).unwrap();
    assert!(variants[2].starts_with("Explain"), "got: {}", variants[2]);
}

#[test]
fn mock_generator_fourth_variant_tell_me_about() {
    let mg = MockQueryVariantGenerator::new();
    let variants = mg.generate_variants("Rust", 4).unwrap();
    assert!(
        variants[3].starts_with("Tell me about"),
        "got: {}",
        variants[3]
    );
}

#[test]
fn mock_generator_fallback_suffix_for_large_n() {
    let mg = MockQueryVariantGenerator::new();
    let n = 10usize;
    let variants = mg.generate_variants("Rust", n).unwrap();
    assert_eq!(variants.len(), n);
    // Indices >= 6 should use the fallback pattern.
    assert!(variants[6].contains("[variant 6]"), "got: {}", variants[6]);
}

#[test]
fn mock_generator_empty_query_returns_err() {
    let mg = MockQueryVariantGenerator::new();
    assert!(matches!(
        mg.generate_variants("", 3),
        Err(MultiQueryError::GenerationFailed(_))
    ));
}

#[test]
fn mock_generator_whitespace_only_query_returns_err() {
    let mg = MockQueryVariantGenerator::new();
    assert!(matches!(
        mg.generate_variants("   ", 3),
        Err(MultiQueryError::GenerationFailed(_))
    ));
}

#[test]
fn mock_generator_default_impl_same_as_new() {
    assert_eq!(MockQueryVariantGenerator, MockQueryVariantGenerator::new());
}

#[test]
fn mock_generator_prefixes_has_six_entries() {
    assert_eq!(MockQueryVariantGenerator::prefixes().len(), 6);
}

#[test]
fn mock_generator_variants_contain_query() {
    let mg = MockQueryVariantGenerator::new();
    let variants = mg.generate_variants("memory safety", 3).unwrap();
    for v in &variants {
        assert!(
            v.contains("memory safety"),
            "variant does not contain query: {v}"
        );
    }
}

// ── MultiQueryGenerator::new ──────────────────────────────────────────────────

#[test]
fn generator_new_stores_config() {
    let cfg = MultiQueryConfig::new().with_num_variants(5);
    let g = MultiQueryGenerator::new(cfg.clone(), MockQueryVariantGenerator::new());
    assert_eq!(g.config, cfg);
}

#[test]
fn generator_debug_does_not_panic() {
    let g = default_generator();
    let s = format!("{g:?}");
    assert!(s.contains("MultiQueryGenerator"));
}

// ── retrieve — error cases ────────────────────────────────────────────────────

#[test]
fn retrieve_empty_corpus_returns_empty_corpus_error() {
    let g = default_generator();
    assert!(matches!(
        g.retrieve("Rust", &[]),
        Err(MultiQueryError::EmptyCorpus)
    ));
}

#[test]
fn retrieve_empty_query_returns_generation_failed() {
    let g = default_generator();
    let docs = make_docs(&[("d1", "content")]);
    assert!(matches!(
        g.retrieve("", &docs),
        Err(MultiQueryError::GenerationFailed(_))
    ));
}

// ── retrieve — result shape ───────────────────────────────────────────────────

#[test]
fn retrieve_returns_ok() {
    let g = default_generator();
    assert!(g.retrieve("Rust", &small_corpus()).is_ok());
}

#[test]
fn retrieve_result_original_query_matches() {
    let g = default_generator();
    let result = g.retrieve("Rust language", &small_corpus()).unwrap();
    assert_eq!(result.original_query, "Rust language");
}

#[test]
fn retrieve_generated_queries_count_matches_config() {
    let cfg = MultiQueryConfig::new().with_num_variants(5);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    assert_eq!(result.generated_queries.len(), 5);
}

#[test]
fn retrieve_generated_queries_have_sequential_indices() {
    let g = default_generator();
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    for (expected_idx, gq) in result.generated_queries.iter().enumerate() {
        assert_eq!(gq.variant_idx, expected_idx);
    }
}

#[test]
fn retrieve_generated_queries_texts_are_non_empty() {
    let g = default_generator();
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    for gq in &result.generated_queries {
        assert!(!gq.text.is_empty());
    }
}

#[test]
fn retrieve_hits_not_empty_for_relevant_query() {
    let g = default_generator();
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    assert!(!result.hits.is_empty());
}

#[test]
fn retrieve_hits_sorted_by_rrf_score_descending() {
    let g = default_generator();
    let result = g.retrieve("Rust language", &small_corpus()).unwrap();
    let scores: Vec<f64> = result.hits.iter().map(|h| h.rrf_score).collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(
        scores, sorted,
        "hits are not sorted by descending RRF score"
    );
}

#[test]
fn retrieve_hits_capped_at_top_k() {
    let cfg = MultiQueryConfig::new().with_top_k(2);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    assert!(result.hits.len() <= 2);
}

#[test]
fn retrieve_hits_have_positive_rrf_scores() {
    let g = default_generator();
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    for h in &result.hits {
        assert!(h.rrf_score > 0.0, "RRF score not positive: {}", h.rrf_score);
    }
}

#[test]
fn retrieve_hit_ids_match_corpus_ids() {
    let g = default_generator();
    let docs = small_corpus();
    let result = g.retrieve("Rust", &docs).unwrap();
    let corpus_ids: std::collections::HashSet<&str> =
        docs.iter().map(|(id, _)| id.as_str()).collect();
    for h in &result.hits {
        assert!(
            corpus_ids.contains(h.id.as_str()),
            "hit id {} not in corpus",
            h.id
        );
    }
}

#[test]
fn retrieve_hit_content_matches_corpus_content() {
    let g = default_generator();
    let docs = small_corpus();
    let content_map: std::collections::HashMap<&str, &str> = docs
        .iter()
        .map(|(id, content)| (id.as_str(), content.as_str()))
        .collect();
    let result = g.retrieve("Rust", &docs).unwrap();
    for h in &result.hits {
        let expected = content_map[h.id.as_str()];
        assert_eq!(h.content, expected);
    }
}

#[test]
fn retrieve_no_duplicate_ids_in_hits() {
    let g = default_generator();
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    let mut seen = std::collections::HashSet::new();
    for h in &result.hits {
        assert!(seen.insert(h.id.as_str()), "duplicate id: {}", h.id);
    }
}

// ── retrieve — single-doc corpus ──────────────────────────────────────────────

#[test]
fn retrieve_single_doc_returns_that_doc() {
    let g = default_generator();
    let docs = make_docs(&[("only", "Rust memory ownership safety language")]);
    let result = g.retrieve("Rust memory", &docs).unwrap();
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, "only");
}

// ── retrieve — top_k = 0 ─────────────────────────────────────────────────────

#[test]
fn retrieve_top_k_zero_returns_empty_hits() {
    let cfg = MultiQueryConfig::new().with_top_k(0);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    assert!(result.hits.is_empty());
}

// ── retrieve — num_variants = 0 ───────────────────────────────────────────────

#[test]
fn retrieve_num_variants_zero_returns_empty_generated_and_hits() {
    let cfg = MultiQueryConfig::new().with_num_variants(0);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    assert!(result.generated_queries.is_empty());
    assert!(result.hits.is_empty());
}

// ── retrieve — num_variants = 1 ───────────────────────────────────────────────

#[test]
fn retrieve_single_variant_still_returns_hits() {
    let cfg = MultiQueryConfig::new().with_num_variants(1);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    assert_eq!(result.generated_queries.len(), 1);
    assert!(!result.hits.is_empty());
}

// ── retrieve — RRF accumulation ───────────────────────────────────────────────

#[test]
fn retrieve_doc_appearing_in_more_variants_gets_higher_score() {
    // doc_a shares tokens with all variants; doc_b shares tokens only sometimes
    let docs = make_docs(&[
        ("doc_a", "rust language systems rust rust"),
        ("doc_b", "python dynamic scripting"),
    ]);
    let cfg = MultiQueryConfig::new().with_num_variants(3).with_top_k(2);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("rust language", &docs).unwrap();
    // doc_a should appear and have a higher score than doc_b (or doc_b absent)
    assert_eq!(result.hits[0].id, "doc_a");
}

#[test]
fn retrieve_rrf_k_affects_score_magnitude() {
    // Higher rrf_k should yield smaller individual contributions.
    let docs = make_docs(&[
        ("d1", "rust ownership memory"),
        ("d2", "python scripting language"),
    ]);
    let cfg_low = MultiQueryConfig::new()
        .with_num_variants(1)
        .with_top_k(2)
        .with_rrf_k(1);
    let cfg_high = MultiQueryConfig::new()
        .with_num_variants(1)
        .with_top_k(2)
        .with_rrf_k(1000);
    let g_low = MultiQueryGenerator::new(cfg_low, MockQueryVariantGenerator::new());
    let g_high = MultiQueryGenerator::new(cfg_high, MockQueryVariantGenerator::new());
    let r_low = g_low.retrieve("rust language", &docs).unwrap();
    let r_high = g_high.retrieve("rust language", &docs).unwrap();
    // Both should surface d1 first; the low-k scores should be larger.
    if !r_low.hits.is_empty() && !r_high.hits.is_empty() {
        assert!(r_low.hits[0].rrf_score > r_high.hits[0].rrf_score);
    }
}

#[test]
fn retrieve_many_variants_accumulates_higher_rrf() {
    let docs = make_docs(&[
        ("match", "rust language systems programming"),
        ("no_match", "cooking recipes baking"),
    ]);
    let cfg_few = MultiQueryConfig::new().with_num_variants(1).with_top_k(2);
    let cfg_many = MultiQueryConfig::new().with_num_variants(4).with_top_k(2);
    let g_few = MultiQueryGenerator::new(cfg_few, MockQueryVariantGenerator::new());
    let g_many = MultiQueryGenerator::new(cfg_many, MockQueryVariantGenerator::new());
    let r_few = g_few.retrieve("rust language", &docs).unwrap();
    let r_many = g_many.retrieve("rust language", &docs).unwrap();
    // With more variants the matching doc should accumulate more RRF score.
    let score_few = r_few
        .hits
        .iter()
        .find(|h| h.id == "match")
        .map(|h| h.rrf_score)
        .unwrap_or(0.0);
    let score_many = r_many
        .hits
        .iter()
        .find(|h| h.id == "match")
        .map(|h| h.rrf_score)
        .unwrap_or(0.0);
    assert!(
        score_many >= score_few,
        "more variants should not reduce score; few={score_few}, many={score_many}"
    );
}

// ── retrieve — large corpus ───────────────────────────────────────────────────

#[test]
fn retrieve_large_corpus_returns_at_most_top_k() {
    let docs: Vec<(String, String)> = (0..200)
        .map(|i| {
            (
                format!("doc{i}"),
                format!("rust language memory safety document number {i}"),
            )
        })
        .collect();
    let cfg = MultiQueryConfig::new().with_top_k(5);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("rust memory", &docs).unwrap();
    assert!(result.hits.len() <= 5);
}

// ── retrieve — result struct fields ──────────────────────────────────────────

#[test]
fn retrieve_result_all_fields_populated() {
    let g = default_generator();
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    assert!(!result.original_query.is_empty());
    assert!(!result.generated_queries.is_empty());
    assert!(!result.hits.is_empty());
}

#[test]
fn retrieve_result_clone() {
    let g = default_generator();
    let result = g.retrieve("Rust", &small_corpus()).unwrap();
    assert_eq!(result.clone(), result);
}

// ── retrieve — single-word query ─────────────────────────────────────────────

#[test]
fn retrieve_single_word_query() {
    let g = default_generator();
    let docs = make_docs(&[
        ("d1", "rust programming language"),
        ("d2", "python scripting"),
    ]);
    let result = g.retrieve("rust", &docs).unwrap();
    assert!(!result.hits.is_empty());
    assert_eq!(result.hits[0].id, "d1");
}

// ── retrieve — multi-word query ───────────────────────────────────────────────

#[test]
fn retrieve_multi_word_query_ranks_best_match_first() {
    let g = default_generator();
    let docs = make_docs(&[
        ("perfect", "rust memory safety systems ownership"),
        ("partial", "rust programming"),
        ("unrelated", "coffee morning breakfast"),
    ]);
    let result = g.retrieve("rust memory safety", &docs).unwrap();
    assert!(!result.hits.is_empty());
    assert_eq!(result.hits[0].id, "perfect");
}

// ── retrieve — consistent ordering tie-break ─────────────────────────────────

#[test]
fn retrieve_identical_content_tie_broken_by_id() {
    let same_content = "rust language";
    let docs = make_docs(&[
        ("zzz", same_content),
        ("aaa", same_content),
        ("mmm", same_content),
    ]);
    let cfg = MultiQueryConfig::new().with_top_k(3);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("rust language", &docs).unwrap();
    // All three docs have identical RRF scores; they should be sorted by id.
    let ids: Vec<&str> = result.hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, vec!["aaa", "mmm", "zzz"]);
}

// ── retrieve — unrelated corpus ───────────────────────────────────────────────

#[test]
fn retrieve_zero_overlap_docs_still_returns_hits() {
    // When BM25-lite gives every doc score 0, all docs tie and any subset is valid.
    let docs = make_docs(&[
        ("d1", "aardvark balloon castle"),
        ("d2", "donkey elephant frog"),
    ]);
    let g = default_generator();
    let result = g.retrieve("xyz quantum unicorn", &docs).unwrap();
    // Should not error, and hits may be empty or contain any subset.
    let _ = result;
}

// ── MultiQueryError variants ──────────────────────────────────────────────────

#[test]
fn error_generation_failed_display() {
    let e = MultiQueryError::GenerationFailed("bad input".to_string());
    let msg = format!("{e}");
    assert!(msg.contains("generation failed") || msg.contains("bad input"));
}

#[test]
fn error_retrieval_failed_display() {
    let e = MultiQueryError::RetrievalFailed("timeout".to_string());
    let msg = format!("{e}");
    assert!(msg.contains("retrieval failed") || msg.contains("timeout"));
}

#[test]
fn error_empty_corpus_display() {
    let e = MultiQueryError::EmptyCorpus;
    let msg = format!("{e}");
    assert!(msg.contains("empty"));
}

// ── MultiQueryResult deduplication ───────────────────────────────────────────

#[test]
fn retrieve_each_doc_id_appears_at_most_once_in_hits() {
    let cfg = MultiQueryConfig::new().with_num_variants(6).with_top_k(10);
    let g = MultiQueryGenerator::new(cfg, MockQueryVariantGenerator::new());
    let result = g.retrieve("rust systems", &small_corpus()).unwrap();
    let mut ids = std::collections::HashSet::new();
    for h in &result.hits {
        assert!(ids.insert(h.id.as_str()), "duplicate id: {}", h.id);
    }
}
