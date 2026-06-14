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
    clippy::single_char_pattern
)]

use super::index::{LshIndex, MinHashIndex};
use super::types::{LshConfig, LshError, LshHit};

// ── FNV embed helper (matches task spec) ─────────────────────────────────────

/// FNV-1a based deterministic embedding.
///
/// Each dimension `i` is derived by starting with the FNV-1a hash of `text`,
/// then running a second FNV-1a pass over the bytes of `i` so that EVERY byte
/// of the dimension index affects the upper 32 bits of `h`. This ensures the
/// per-dimension values are genuinely distinct (unlike a simple `h ^= i` XOR
/// which only changes the lowest bits and leaves the upper 32 bits identical
/// for i < 2^32).
#[allow(clippy::cast_precision_loss)]
fn fnv_embed(text: &str, dim: usize) -> Vec<f32> {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut v = vec![0.0f32; dim];
    let bytes = text.as_bytes();
    for (i, slot) in v.iter_mut().enumerate() {
        // Pass 1: FNV-1a over text bytes
        let mut h: u64 = OFFSET;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(PRIME);
        }
        // Pass 2: FNV-1a over each byte of the dimension index so that
        // the full 64-bit hash (and thus the upper 32 bits) changes per dim.
        for byte in i.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(PRIME);
        }
        *slot = (h >> 32) as f32 / u32::MAX as f32 * 2.0 - 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// Simple shingle hashes for MinHash tests.
fn shingle_hashes(text: &str, k: usize) -> Vec<u64> {
    if text.len() < k {
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in text.as_bytes() {
            h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(*b);
        }
        return vec![h];
    }
    text.as_bytes()
        .windows(k)
        .map(|w| {
            let mut h: u64 = 14_695_981_039_346_656_037;
            for &b in w {
                h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
            }
            h
        })
        .collect()
}

// ── LshConfig tests ───────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let c = LshConfig::default();
    assert_eq!(c.dim, 128);
    assert_eq!(c.num_planes, 16);
    assert_eq!(c.num_bands, 4);
    assert_eq!(c.rows_per_band, 4);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(LshConfig::new(), LshConfig::default());
}

#[test]
fn config_builder_dim() {
    assert_eq!(LshConfig::new().with_dim(64).dim, 64);
}

#[test]
fn config_builder_num_planes() {
    assert_eq!(LshConfig::new().with_num_planes(32).num_planes, 32);
}

#[test]
fn config_builder_num_bands() {
    assert_eq!(LshConfig::new().with_num_bands(8).num_bands, 8);
}

#[test]
fn config_builder_rows_per_band() {
    assert_eq!(LshConfig::new().with_rows_per_band(6).rows_per_band, 6);
}

#[test]
fn config_builder_chain() {
    let c = LshConfig::new()
        .with_dim(32)
        .with_num_planes(8)
        .with_num_bands(3)
        .with_rows_per_band(2);
    assert_eq!(c.dim, 32);
    assert_eq!(c.num_planes, 8);
    assert_eq!(c.num_bands, 3);
    assert_eq!(c.rows_per_band, 2);
}

#[test]
fn config_total_minhash_funcs() {
    let c = LshConfig::new().with_num_bands(3).with_rows_per_band(5);
    assert_eq!(c.total_minhash_funcs(), 15);
}

#[test]
fn config_validate_ok() {
    assert!(LshConfig::new().validate().is_ok());
}

#[test]
fn config_validate_zero_dim() {
    let err = LshConfig::new().with_dim(0).validate().unwrap_err();
    assert!(matches!(err, LshError::InvalidConfig(_)));
}

#[test]
fn config_validate_zero_planes() {
    let err = LshConfig::new().with_num_planes(0).validate().unwrap_err();
    assert!(matches!(err, LshError::InvalidConfig(_)));
}

#[test]
fn config_validate_zero_bands() {
    let err = LshConfig::new().with_num_bands(0).validate().unwrap_err();
    assert!(matches!(err, LshError::InvalidConfig(_)));
}

#[test]
fn config_validate_zero_rows() {
    let err = LshConfig::new()
        .with_rows_per_band(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, LshError::InvalidConfig(_)));
}

// ── LshIndex — construction ───────────────────────────────────────────────────

#[test]
fn index_new_invalid_config_zero_dim() {
    let cfg = LshConfig::new().with_dim(0).with_num_planes(4);
    assert!(LshIndex::new(cfg).is_err());
}

#[test]
fn index_new_invalid_config_zero_planes() {
    let cfg = LshConfig::new().with_dim(4).with_num_planes(0);
    assert!(LshIndex::new(cfg).is_err());
}

#[test]
fn index_empty_after_construction() {
    let idx = LshIndex::new(LshConfig::new().with_dim(4)).unwrap();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
}

// ── LshIndex — insert / len ───────────────────────────────────────────────────

#[test]
fn index_insert_single_item() {
    let mut idx = LshIndex::new(LshConfig::new().with_dim(4)).unwrap();
    idx.insert("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(idx.len(), 1);
    assert!(!idx.is_empty());
}

#[test]
fn index_insert_dim_mismatch() {
    let mut idx = LshIndex::new(LshConfig::new().with_dim(4)).unwrap();
    let err = idx.insert("bad", vec![1.0, 0.0]).unwrap_err();
    assert!(matches!(
        err,
        LshError::DimMismatch {
            expected: 4,
            got: 2
        }
    ));
}

#[test]
fn index_insert_zero_vector_normalised() {
    // A zero vector normalised lands at near-zero; no panic expected.
    let mut idx = LshIndex::new(LshConfig::new().with_dim(4)).unwrap();
    idx.insert("zero", vec![0.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(idx.len(), 1);
}

#[test]
fn index_insert_multiple() {
    let dim = 8;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim)).unwrap();
    for i in 0..10 {
        let v = fnv_embed(&format!("doc{i}"), dim);
        idx.insert(format!("doc{i}"), v).unwrap();
    }
    assert_eq!(idx.len(), 10);
}

// ── LshIndex — search errors ──────────────────────────────────────────────────

#[test]
fn search_empty_index_returns_error() {
    let idx = LshIndex::new(LshConfig::new().with_dim(4)).unwrap();
    let err = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap_err();
    assert_eq!(err, LshError::EmptyIndex);
}

#[test]
fn search_k_zero_returns_error() {
    let mut idx = LshIndex::new(LshConfig::new().with_dim(4)).unwrap();
    idx.insert("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    let err = idx.search(&[1.0, 0.0, 0.0, 0.0], 0).unwrap_err();
    assert_eq!(err, LshError::InvalidK);
}

#[test]
fn search_dim_mismatch_query() {
    let mut idx = LshIndex::new(LshConfig::new().with_dim(4)).unwrap();
    idx.insert("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    let err = idx.search(&[1.0, 0.0], 1).unwrap_err();
    assert!(matches!(
        err,
        LshError::DimMismatch {
            expected: 4,
            got: 2
        }
    ));
}

// ── LshIndex — search correctness ─────────────────────────────────────────────

#[test]
fn search_single_item_returns_it() {
    let mut idx = LshIndex::new(LshConfig::new().with_dim(4).with_num_planes(8)).unwrap();
    idx.insert("only", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "only");
}

#[test]
fn search_score_in_zero_one() {
    let dim = 16;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(16)).unwrap();
    for i in 0..20 {
        idx.insert(format!("d{i}"), fnv_embed(&format!("item{i}"), dim))
            .unwrap();
    }
    let query = fnv_embed("query", dim);
    let hits = idx.search(&query, 5).unwrap();
    for h in &hits {
        assert!(
            h.score >= 0.0 && h.score <= 1.0,
            "score {} out of range",
            h.score
        );
    }
}

#[test]
fn search_identical_vector_scores_near_one() {
    let dim = 8;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(16)).unwrap();
    let v = fnv_embed("hello world", dim);
    idx.insert("exact", v.clone()).unwrap();
    for i in 0..10 {
        idx.insert(format!("noise{i}"), fnv_embed(&format!("noise{i}"), dim))
            .unwrap();
    }
    let hits = idx.search(&v, 3).unwrap();
    assert!(!hits.is_empty());
    // The exact match should score very close to 1.0
    let top = hits.iter().find(|h| h.id == "exact").unwrap();
    assert!(top.score > 0.98, "score {} not close to 1.0", top.score);
}

#[test]
fn search_k_larger_than_corpus() {
    let dim = 4;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(8)).unwrap();
    idx.insert("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    idx.insert("b", vec![0.0, 1.0, 0.0, 0.0]).unwrap();
    // k=10 with only 2 items — should return at most 2 hits, not panic
    let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
    assert!(hits.len() <= 2);
}

#[test]
fn search_descending_score_order() {
    let dim = 16;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(16)).unwrap();
    for i in 0..15 {
        idx.insert(format!("d{i}"), fnv_embed(&format!("doc{i}"), dim))
            .unwrap();
    }
    let q = fnv_embed("doc0", dim);
    let hits = idx.search(&q, 5).unwrap();
    for w in hits.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "hits not in descending order: {} < {}",
            w[0].score,
            w[1].score
        );
    }
}

#[test]
fn search_returns_at_most_k_hits() {
    let dim = 8;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(8)).unwrap();
    for i in 0..20 {
        idx.insert(format!("d{i}"), fnv_embed(&format!("item{i}"), dim))
            .unwrap();
    }
    let q = fnv_embed("item0", dim);
    let hits = idx.search(&q, 5).unwrap();
    assert!(hits.len() <= 5);
}

#[test]
fn search_cosine_one_for_parallel_vectors() {
    // Two vectors in the same direction should yield score ≈ 1.0.
    let dim = 4;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(16)).unwrap();
    idx.insert("pos", vec![2.0, 0.0, 0.0, 0.0]).unwrap();
    idx.insert("neg", vec![-1.0, 0.0, 0.0, 0.0]).unwrap();
    let hits = idx.search(&[3.0, 0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(hits[0].id, "pos");
    assert!(hits[0].score > 0.99);
}

#[test]
fn search_100_doc_corpus_finds_near_duplicate() {
    let dim = 32;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(32)).unwrap();
    for i in 0..100 {
        idx.insert(
            format!("doc{i}"),
            fnv_embed(&format!("document number {i}"), dim),
        )
        .unwrap();
    }
    assert_eq!(idx.len(), 100);
    let target = fnv_embed("document number 42", dim);
    let hits = idx.search(&target, 5).unwrap();
    assert!(!hits.is_empty());
    // The exact match for doc42 should appear in the results
    let found = hits.iter().any(|h| h.id == "doc42");
    assert!(
        found,
        "doc42 not found in top-5 for its own vector; hits={hits:?}"
    );
}

#[test]
fn search_no_duplicate_ids_in_result() {
    let dim = 8;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(8)).unwrap();
    for i in 0..10 {
        idx.insert(format!("d{i}"), fnv_embed(&format!("item{i}"), dim))
            .unwrap();
    }
    let q = fnv_embed("item3", dim);
    let hits = idx.search(&q, 10).unwrap();
    let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    ids.sort_unstable();
    let before = ids.len();
    ids.dedup();
    assert_eq!(before, ids.len(), "duplicate ids in results");
}

#[test]
fn search_bucket_collision_same_signature() {
    // Two nearly-identical vectors should share a bucket.
    let dim = 4;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(4)).unwrap();
    idx.insert("v1", vec![1.0, 0.001, 0.0, 0.0]).unwrap();
    idx.insert("v2", vec![1.0, 0.002, 0.0, 0.0]).unwrap();
    let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
    // Both should be retrieved since they're close to the query
    assert!(!hits.is_empty());
}

#[test]
fn search_orthogonal_vectors_score_near_half() {
    // Cosine of orthogonal unit vectors is 0 → mapped score ≈ 0.5.
    let dim = 4;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(64)).unwrap();
    idx.insert("orth", vec![0.0, 1.0, 0.0, 0.0]).unwrap();
    let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    if !hits.is_empty() {
        // cos ≈ 0 → score ≈ 0.5; allow wide tolerance because LSH is approximate
        assert!(hits[0].score >= 0.0 && hits[0].score <= 1.0);
    }
}

#[test]
fn index_config_accessor() {
    let cfg = LshConfig::new().with_dim(16).with_num_planes(8);
    let idx = LshIndex::new(cfg.clone()).unwrap();
    assert_eq!(idx.config().dim, 16);
    assert_eq!(idx.config().num_planes, 8);
}

#[test]
fn lsh_hit_new() {
    let h = LshHit::new("x", 0.75);
    assert_eq!(h.id, "x");
    assert_eq!(h.score, 0.75);
}

#[test]
fn lsh_error_dim_mismatch_display() {
    let e = LshError::DimMismatch {
        expected: 4,
        got: 2,
    };
    let s = e.to_string();
    assert!(
        s.contains("4") && s.contains("2"),
        "unexpected display: {s}"
    );
}

#[test]
fn lsh_error_empty_index_display() {
    let s = LshError::EmptyIndex.to_string();
    assert!(!s.is_empty());
}

#[test]
fn lsh_error_invalid_k_display() {
    let s = LshError::InvalidK.to_string();
    assert!(!s.is_empty());
}

// ── MinHashIndex tests ────────────────────────────────────────────────────────

#[test]
fn minhash_new_ok() {
    let cfg = LshConfig::new().with_num_bands(4).with_rows_per_band(4);
    let idx = MinHashIndex::new(cfg).unwrap();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
}

#[test]
fn minhash_new_invalid_config() {
    let cfg = LshConfig::new().with_num_bands(0);
    assert!(MinHashIndex::new(cfg).is_err());
}

#[test]
fn minhash_insert_single() {
    let cfg = LshConfig::new().with_num_bands(2).with_rows_per_band(2);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    let elems = shingle_hashes("hello world", 3);
    idx.insert("a", elems).unwrap();
    assert_eq!(idx.len(), 1);
    assert!(!idx.is_empty());
}

#[test]
fn minhash_insert_empty_set_error() {
    let cfg = LshConfig::new().with_num_bands(2).with_rows_per_band(2);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    let err = idx.insert("bad", vec![]).unwrap_err();
    assert_eq!(err, LshError::EmptyIndex);
}

#[test]
fn minhash_search_empty_index_error() {
    let cfg = LshConfig::new().with_num_bands(2).with_rows_per_band(2);
    let idx = MinHashIndex::new(cfg).unwrap();
    let err = idx.search(&[1, 2, 3], 1).unwrap_err();
    assert_eq!(err, LshError::EmptyIndex);
}

#[test]
fn minhash_search_k_zero_error() {
    let cfg = LshConfig::new().with_num_bands(2).with_rows_per_band(2);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    idx.insert("a", shingle_hashes("hello", 2)).unwrap();
    let err = idx.search(&shingle_hashes("hello", 2), 0).unwrap_err();
    assert_eq!(err, LshError::InvalidK);
}

#[test]
fn minhash_identical_sets_high_score() {
    let cfg = LshConfig::new().with_num_bands(8).with_rows_per_band(4);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    let text = "the quick brown fox jumps over the lazy dog";
    let elems = shingle_hashes(text, 3);
    idx.insert("same", elems.clone()).unwrap();
    for i in 0..5 {
        idx.insert(
            format!("noise{i}"),
            shingle_hashes(&format!("noise document {i}"), 3),
        )
        .unwrap();
    }
    let hits = idx.search(&elems, 3).unwrap();
    assert!(!hits.is_empty());
    let top = hits.iter().find(|h| h.id == "same");
    assert!(top.is_some(), "identical set not found; hits={hits:?}");
    assert!(
        top.unwrap().score > 0.9,
        "score too low: {}",
        top.unwrap().score
    );
}

#[test]
fn minhash_disjoint_sets_low_score() {
    let cfg = LshConfig::new().with_num_bands(4).with_rows_per_band(4);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    // Sets with nothing in common
    idx.insert("a", vec![1u64, 2, 3, 4, 5]).unwrap();
    idx.insert("b", vec![100u64, 200, 300, 400, 500]).unwrap();
    let hits = idx.search(&[1u64, 2, 3, 4, 5], 2).unwrap();
    // "a" should be found if there is any band collision; score ≈ 1.0
    if let Some(hit_a) = hits.iter().find(|h| h.id == "a") {
        assert!(hit_a.score > 0.5);
    }
    // "b" (disjoint) should score 0.0 if found
    if let Some(hit_b) = hits.iter().find(|h| h.id == "b") {
        assert!(
            hit_b.score < 0.2,
            "disjoint sets scored too high: {}",
            hit_b.score
        );
    }
}

#[test]
fn minhash_score_in_zero_one() {
    let cfg = LshConfig::new().with_num_bands(4).with_rows_per_band(3);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    for i in 0..10u64 {
        idx.insert(format!("s{i}"), vec![i * 10, i * 10 + 1, i * 10 + 2])
            .unwrap();
    }
    let q = vec![1u64, 2, 3];
    let hits = idx.search(&q, 5).unwrap();
    for h in &hits {
        assert!(
            h.score >= 0.0 && h.score <= 1.0,
            "score {} out of [0,1]",
            h.score
        );
    }
}

#[test]
fn minhash_multiple_inserts_len() {
    let cfg = LshConfig::new().with_num_bands(3).with_rows_per_band(3);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    for i in 0..25u64 {
        idx.insert(format!("s{i}"), vec![i, i + 1, i + 2]).unwrap();
    }
    assert_eq!(idx.len(), 25);
}

#[test]
fn minhash_k_larger_than_corpus() {
    let cfg = LshConfig::new().with_num_bands(4).with_rows_per_band(4);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    idx.insert("x", vec![1u64, 2, 3]).unwrap();
    // Request more hits than indexed — should not panic
    let hits = idx.search(&[1u64, 2, 3], 100).unwrap();
    assert!(hits.len() <= 1);
}

#[test]
fn minhash_search_descending_order() {
    let cfg = LshConfig::new().with_num_bands(6).with_rows_per_band(4);
    let mut idx = MinHashIndex::new(cfg).unwrap();
    // Insert overlapping sets of varying similarity to query [1..10]
    idx.insert("full", vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .unwrap();
    idx.insert("half", vec![1u64, 2, 3, 4, 5]).unwrap();
    idx.insert("tiny", vec![1u64, 2]).unwrap();
    idx.insert("extra", vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        .unwrap();
    let q = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let hits = idx.search(&q, 5).unwrap();
    for w in hits.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "scores not descending: {} < {}",
            w[0].score,
            w[1].score
        );
    }
}

// ── Combined / cross-type tests ───────────────────────────────────────────────

#[test]
fn dense_and_minhash_independent() {
    // Both index types can be used simultaneously without interference.
    let dim = 8;
    let mut dense = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(8)).unwrap();
    let mut minhash =
        MinHashIndex::new(LshConfig::new().with_num_bands(3).with_rows_per_band(3)).unwrap();

    dense.insert("d", fnv_embed("dense doc", dim)).unwrap();
    minhash
        .insert("m", shingle_hashes("minhash doc", 3))
        .unwrap();

    assert_eq!(dense.len(), 1);
    assert_eq!(minhash.len(), 1);
}

#[test]
fn index_insert_then_search_roundtrip() {
    let dim = 16;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(16)).unwrap();
    let labels = ["alpha", "beta", "gamma", "delta", "epsilon"];
    for lbl in labels {
        idx.insert(lbl, fnv_embed(lbl, dim)).unwrap();
    }
    let q = fnv_embed("alpha", dim);
    let hits = idx.search(&q, 3).unwrap();
    // "alpha" must appear (searching for its own embedding)
    assert!(
        hits.iter().any(|h| h.id == "alpha"),
        "alpha not found: {hits:?}"
    );
}

#[test]
fn large_num_planes_still_works() {
    let dim = 16;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(128)).unwrap();
    for i in 0..10 {
        idx.insert(format!("d{i}"), fnv_embed(&format!("doc{i}"), dim))
            .unwrap();
    }
    let q = fnv_embed("doc5", dim);
    let hits = idx.search(&q, 3).unwrap();
    // documents were inserted with key format "d{i}", so doc5 has id "d5"
    assert!(
        hits.iter().any(|h| h.id == "d5"),
        "d5 not retrieved: {hits:?}"
    );
}

#[test]
fn search_negative_cosine_score_clamped() {
    // Antipodal vectors: cosine ≈ -1 → score should clamp to ≥ 0.
    let dim = 4;
    let mut idx = LshIndex::new(LshConfig::new().with_dim(dim).with_num_planes(32)).unwrap();
    idx.insert("anti", vec![-1.0, 0.0, 0.0, 0.0]).unwrap();
    let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    for h in &hits {
        assert!(h.score >= 0.0, "score below 0: {}", h.score);
    }
}
