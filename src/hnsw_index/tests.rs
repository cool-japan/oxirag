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
    clippy::needless_for_each,
    clippy::uninlined_format_args,
    clippy::single_char_pattern
)]

use super::index::HnswIndex;
use super::types::{HnswConfig, HnswError};

// ── Test helpers ──────────────────────────────────────────────────────────────

/// FNV-1a pseudo-embedding: deterministic, no `rand` dependency.
///
/// Each dimension hashes the text bytes then applies full FNV-1a to each byte
/// of the dimension index `i`, ensuring the upper 32 bits of the hash vary
/// across dimensions and across different texts.
fn fnv_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    let bytes = text.as_bytes();
    for (i, slot) in v.iter_mut().enumerate() {
        let mut h: u64 = 14_695_981_039_346_656_037;
        for &b in bytes {
            h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
        }
        // Hash each byte of the dimension index so the upper 32 bits vary.
        for b in (i as u64).to_le_bytes() {
            h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
        }
        *slot = ((h >> 32) as f32) / u32::MAX as f32 * 2.0 - 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// Build a small index with `n` docs (labels "doc_0" … "doc_n-1"), dim=16.
///
/// Parameters are intentionally generous (m=16, high ef values) so that
/// the layer-0 graph is essentially complete for n≤17 and well-connected
/// for larger n — this guarantees that self-queries always return the
/// correct top-1 result.
fn small_index(n: usize) -> HnswIndex {
    let cfg = HnswConfig::new()
        .with_dim(16)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_search(50)
        .with_max_layers(6);
    let mut idx = HnswIndex::new(cfg);
    for i in 0..n {
        idx.insert(format!("doc_{i}"), fnv_embed(&format!("doc_{i}"), 16))
            .unwrap();
    }
    idx
}

// ── HnswConfig tests ──────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let c = HnswConfig::default();
    assert_eq!(c.dim, 128);
    assert_eq!(c.m, 16);
    assert_eq!(c.ef_construction, 200);
    assert_eq!(c.ef_search, 50);
    assert_eq!(c.max_layers, 6);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(HnswConfig::new(), HnswConfig::default());
}

#[test]
fn config_builder_dim() {
    assert_eq!(HnswConfig::new().with_dim(64).dim, 64);
}

#[test]
fn config_builder_m() {
    assert_eq!(HnswConfig::new().with_m(8).m, 8);
}

#[test]
fn config_builder_ef_construction() {
    assert_eq!(
        HnswConfig::new().with_ef_construction(100).ef_construction,
        100
    );
}

#[test]
fn config_builder_ef_search() {
    assert_eq!(HnswConfig::new().with_ef_search(25).ef_search, 25);
}

#[test]
fn config_builder_max_layers() {
    assert_eq!(HnswConfig::new().with_max_layers(3).max_layers, 3);
}

#[test]
fn config_builder_chain() {
    let c = HnswConfig::new()
        .with_dim(32)
        .with_m(8)
        .with_ef_construction(50)
        .with_ef_search(20)
        .with_max_layers(3);
    assert_eq!(c.dim, 32);
    assert_eq!(c.m, 8);
    assert_eq!(c.ef_construction, 50);
    assert_eq!(c.ef_search, 20);
    assert_eq!(c.max_layers, 3);
}

#[test]
fn config_validate_ok() {
    assert!(HnswConfig::new().with_dim(4).validate().is_ok());
}

#[test]
fn config_validate_zero_dim() {
    let err = HnswConfig::new().with_dim(0).validate().unwrap_err();
    assert!(matches!(err, HnswError::InvalidConfig(_)));
}

#[test]
fn config_validate_zero_m() {
    let err = HnswConfig::new()
        .with_dim(4)
        .with_m(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, HnswError::InvalidConfig(_)));
}

#[test]
fn config_validate_zero_ef_construction() {
    let err = HnswConfig::new()
        .with_dim(4)
        .with_ef_construction(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, HnswError::InvalidConfig(_)));
}

#[test]
fn config_validate_zero_ef_search() {
    let err = HnswConfig::new()
        .with_dim(4)
        .with_ef_search(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, HnswError::InvalidConfig(_)));
}

#[test]
fn config_validate_zero_max_layers() {
    let err = HnswConfig::new()
        .with_dim(4)
        .with_max_layers(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, HnswError::InvalidConfig(_)));
}

// ── len / is_empty / contains ─────────────────────────────────────────────────

#[test]
fn empty_index_len_is_zero() {
    let idx = HnswIndex::new(HnswConfig::new().with_dim(4));
    assert_eq!(idx.len(), 0);
}

#[test]
fn empty_index_is_empty_true() {
    let idx = HnswIndex::new(HnswConfig::new().with_dim(4));
    assert!(idx.is_empty());
}

#[test]
fn after_insert_is_empty_false() {
    let mut idx = HnswIndex::new(HnswConfig::new().with_dim(4));
    idx.insert("a", fnv_embed("a", 4)).unwrap();
    assert!(!idx.is_empty());
}

#[test]
fn len_grows_with_inserts() {
    let cfg = HnswConfig::new()
        .with_dim(16)
        .with_m(8)
        .with_ef_construction(20)
        .with_ef_search(10)
        .with_max_layers(4);
    let mut idx = HnswIndex::new(cfg);
    for i in 0..5 {
        idx.insert(format!("d{i}"), fnv_embed(&format!("d{i}"), 16))
            .unwrap();
        assert_eq!(idx.len(), i + 1);
    }
}

#[test]
fn contains_existing_id() {
    let idx = small_index(3);
    assert!(idx.contains("doc_0"));
    assert!(idx.contains("doc_1"));
    assert!(idx.contains("doc_2"));
}

#[test]
fn contains_missing_id() {
    let idx = small_index(3);
    assert!(!idx.contains("doc_99"));
    assert!(!idx.contains(""));
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn search_empty_index_returns_error() {
    let idx = HnswIndex::new(HnswConfig::new().with_dim(4));
    let err = idx.search(&fnv_embed("q", 4), 1).unwrap_err();
    assert_eq!(err, HnswError::EmptyIndex);
}

#[test]
fn search_k_zero_returns_error() {
    let idx = small_index(3);
    let err = idx.search(&fnv_embed("q", 16), 0).unwrap_err();
    assert_eq!(err, HnswError::InvalidK);
}

#[test]
fn search_dim_mismatch_returns_error() {
    let idx = small_index(3);
    let wrong_dim = fnv_embed("q", 8); // index expects 16
    let err = idx.search(&wrong_dim, 1).unwrap_err();
    assert!(matches!(
        err,
        HnswError::DimMismatch {
            expected: 16,
            got: 8
        }
    ));
}

#[test]
fn insert_dim_mismatch_returns_error() {
    let mut idx = HnswIndex::new(HnswConfig::new().with_dim(8));
    let err = idx.insert("a", vec![1.0; 4]).unwrap_err();
    assert!(matches!(
        err,
        HnswError::DimMismatch {
            expected: 8,
            got: 4
        }
    ));
}

#[test]
fn insert_invalid_config_dim_zero_returns_error() {
    let mut idx = HnswIndex::new(HnswConfig::new().with_dim(0));
    let err = idx.insert("a", vec![]).unwrap_err();
    assert!(matches!(err, HnswError::InvalidConfig(_)));
}

// ── Single-document index ─────────────────────────────────────────────────────

#[test]
fn single_doc_search_returns_it() {
    let cfg = HnswConfig::new()
        .with_dim(8)
        .with_m(4)
        .with_ef_construction(10)
        .with_ef_search(5);
    let mut idx = HnswIndex::new(cfg);
    idx.insert("only", fnv_embed("only", 8)).unwrap();
    let hits = idx.search(&fnv_embed("only", 8), 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "only");
}

#[test]
fn single_doc_score_is_one_for_exact_query() {
    let cfg = HnswConfig::new()
        .with_dim(8)
        .with_m(4)
        .with_ef_construction(10)
        .with_ef_search(5);
    let mut idx = HnswIndex::new(cfg);
    let v = fnv_embed("ping", 8);
    idx.insert("ping", v.clone()).unwrap();
    let hits = idx.search(&v, 1).unwrap();
    assert!(
        (hits[0].score - 1.0_f32).abs() < 1e-5,
        "score={}",
        hits[0].score
    );
}

// ── Build and search ──────────────────────────────────────────────────────────

#[test]
fn build_and_search_returns_nonempty() {
    let idx = small_index(10);
    let hits = idx.search(&fnv_embed("doc_0", 16), 3).unwrap();
    assert!(!hits.is_empty());
}

#[test]
fn exact_query_top_hit_matches_inserted_doc() {
    let idx = small_index(10);
    let q = fnv_embed("doc_3", 16);
    let hits = idx.search(&q, 1).unwrap();
    assert_eq!(hits[0].id, "doc_3");
}

#[test]
fn scores_descending_order() {
    let idx = small_index(10);
    let hits = idx.search(&fnv_embed("doc_0", 16), 5).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score, "scores out of order: {:?}", hits);
    }
}

#[test]
fn score_range_zero_to_one() {
    let idx = small_index(10);
    let hits = idx.search(&fnv_embed("doc_0", 16), 10).unwrap();
    for h in &hits {
        assert!(
            h.score >= 0.0 && h.score <= 1.0,
            "score out of [0,1]: {}",
            h.score
        );
    }
}

#[test]
fn top_hit_score_is_one_for_self_query() {
    let idx = small_index(5);
    for i in 0..5_usize {
        let label = format!("doc_{i}");
        let v = fnv_embed(&label, 16);
        let hits = idx.search(&v, 1).unwrap();
        assert_eq!(hits[0].id, label, "self-query failed for {label}");
        assert!(
            (hits[0].score - 1.0_f32).abs() < 1e-5,
            "score not ~1.0 for self-query of {label}: {}",
            hits[0].score
        );
    }
}

// ── k > corpus size ───────────────────────────────────────────────────────────

#[test]
fn k_larger_than_corpus_returns_all_docs() {
    let n = 5;
    let idx = small_index(n);
    let hits = idx.search(&fnv_embed("doc_0", 16), 100).unwrap();
    assert_eq!(hits.len(), n, "expected {n} hits, got {}", hits.len());
}

#[test]
fn k_equals_corpus_size() {
    let n = 7;
    let idx = small_index(n);
    let hits = idx.search(&fnv_embed("doc_2", 16), n).unwrap();
    assert_eq!(hits.len(), n);
}

// ── Large corpus ──────────────────────────────────────────────────────────────

#[test]
fn large_corpus_100_docs_builds_without_error() {
    let idx = small_index(100);
    assert_eq!(idx.len(), 100);
}

#[test]
fn large_corpus_100_docs_self_query_top_hit() {
    let idx = small_index(100);
    for i in [0_usize, 33, 66, 99] {
        let label = format!("doc_{i}");
        let v = fnv_embed(&label, 16);
        let hits = idx.search(&v, 1).unwrap();
        assert_eq!(hits[0].id, label, "self-query failed for {label}");
    }
}

#[test]
fn large_corpus_100_docs_scores_descending() {
    let idx = small_index(100);
    let hits = idx.search(&fnv_embed("doc_50", 16), 20).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[test]
fn large_corpus_no_duplicate_ids_in_results() {
    let idx = small_index(100);
    let hits = idx.search(&fnv_embed("doc_0", 16), 50).unwrap();
    let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), hits.len(), "duplicate ids in results");
}

// ── Various dim sizes ─────────────────────────────────────────────────────────

#[test]
fn works_with_dim_2() {
    let cfg = HnswConfig::new()
        .with_dim(2)
        .with_m(4)
        .with_ef_construction(10)
        .with_ef_search(5);
    let mut idx = HnswIndex::new(cfg);
    idx.insert("a", fnv_embed("a", 2)).unwrap();
    idx.insert("b", fnv_embed("b", 2)).unwrap();
    let hits = idx.search(&fnv_embed("a", 2), 1).unwrap();
    assert_eq!(hits[0].id, "a");
}

#[test]
fn works_with_dim_128() {
    let cfg = HnswConfig::new()
        .with_dim(128)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_search(50);
    let mut idx = HnswIndex::new(cfg);
    for i in 0..10_usize {
        idx.insert(format!("v{i}"), fnv_embed(&format!("vec_{i}"), 128))
            .unwrap();
    }
    let hits = idx.search(&fnv_embed("vec_5", 128), 1).unwrap();
    assert_eq!(hits[0].id, "v5");
}

// ── Hit fields ────────────────────────────────────────────────────────────────

#[test]
fn hit_has_id_and_score_fields() {
    let idx = small_index(5);
    let hits = idx.search(&fnv_embed("doc_0", 16), 1).unwrap();
    let h = &hits[0];
    assert!(!h.id.is_empty());
    assert!(h.score >= 0.0);
}

#[test]
fn hit_new_constructor() {
    use super::types::HnswHit;
    let h = HnswHit::new("test_id", 0.75_f32);
    assert_eq!(h.id, "test_id");
    assert!((h.score - 0.75).abs() < 1e-6);
}

// ── Score formula ─────────────────────────────────────────────────────────────

#[test]
fn score_formula_maps_cosine_to_zero_one() {
    // score = (cosine + 1.0) / 2.0
    // cosine = 1.0  → score = 1.0
    // cosine = -1.0 → score = 0.0
    // cosine = 0.0  → score = 0.5
    let cfg = HnswConfig::new()
        .with_dim(4)
        .with_m(4)
        .with_ef_construction(10)
        .with_ef_search(5);
    let mut idx = HnswIndex::new(cfg);
    // Insert a vector and query with itself → cosine should be ~1.0
    let v = vec![0.5_f32, 0.5, 0.5, 0.5]; // not unit — index handles as-is
    idx.insert("v", v.clone()).unwrap();
    let hits = idx.search(&v, 1).unwrap();
    // self-cosine of normalised-v vs normalised-v should be ~1.0 → score ~1.0
    assert!(hits[0].score > 0.99);
}

// ── Mutation after build ───────────────────────────────────────────────────────

#[test]
fn insert_after_build_increases_len() {
    let mut idx = small_index(5);
    assert_eq!(idx.len(), 5);
    idx.insert("extra", fnv_embed("extra_doc", 16)).unwrap();
    assert_eq!(idx.len(), 6);
}

#[test]
fn contains_newly_inserted() {
    let mut idx = small_index(5);
    assert!(!idx.contains("new_doc"));
    idx.insert("new_doc", fnv_embed("new_doc", 16)).unwrap();
    assert!(idx.contains("new_doc"));
}

#[test]
fn search_finds_newly_inserted() {
    let mut idx = small_index(5);
    idx.insert("fresh", fnv_embed("fresh", 16)).unwrap();
    let hits = idx.search(&fnv_embed("fresh", 16), 1).unwrap();
    assert_eq!(hits[0].id, "fresh");
}

// ── Error display ─────────────────────────────────────────────────────────────

#[test]
fn error_display_dim_mismatch() {
    let e = HnswError::DimMismatch {
        expected: 16,
        got: 8,
    };
    let s = e.to_string();
    assert!(s.contains("16") && s.contains("8"), "display: {s}");
}

#[test]
fn error_display_empty_index() {
    let s = HnswError::EmptyIndex.to_string();
    assert!(!s.is_empty());
}

#[test]
fn error_display_invalid_config() {
    let s = HnswError::InvalidConfig("bad".into()).to_string();
    assert!(s.contains("bad"));
}

#[test]
fn error_display_invalid_k() {
    let s = HnswError::InvalidK.to_string();
    assert!(!s.is_empty());
}

// ── Clone and Debug ───────────────────────────────────────────────────────────

#[test]
fn hnsw_config_clone_and_debug() {
    let c = HnswConfig::new().with_dim(4);
    let c2 = c.clone();
    assert_eq!(c, c2);
    assert!(!format!("{c:?}").is_empty());
}

#[test]
fn hnsw_index_clone() {
    let idx = small_index(5);
    let idx2 = idx.clone();
    assert_eq!(idx.len(), idx2.len());
}

#[test]
fn hnsw_index_debug() {
    let idx = small_index(2);
    assert!(!format!("{idx:?}").is_empty());
}
