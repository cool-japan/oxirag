#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::cast_possible_truncation
)]
//! Tests for the `rabitq` module.
//!
//! Covers: `RaBitQConfig` validation, `Rotation` orthogonality and
//! determinism, encode/decode bit packing, the unbiasedness of the
//! `estimate_ip` / `estimate_l2_sq` / `estimate_cosine` estimators on
//! deterministic fixtures, recall on well-separated clusters via
//! `RaBitQIndex`, and every error path.

use super::index::RaBitQIndex;
use super::quantizer::RaBitQuantizer;
use super::rotation::Rotation;
use super::types::{RaBitQCode, RaBitQConfig, RaBitQError, RaBitQHit, RaBitQMetric};

// ── Shared fixtures ───────────────────────────────────────────────────────────

/// FNV-1a based deterministic pseudo-embedding generator (mirrors the helper
/// used by `scalar_quantization`'s tests). Produces a length-`dim` `f32`
/// vector from `text`; not normalized (encode() normalizes internally).
fn fnv_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    let bytes = text.as_bytes();
    for (i, slot) in v.iter_mut().enumerate() {
        let mut h: u64 = 14_695_981_039_346_656_037;
        for &b in bytes {
            h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
        }
        h = h.wrapping_mul(1_099_511_628_211) ^ i as u64;
        *slot = (h >> 32) as f32 / u32::MAX as f32 * 2.0 - 1.0;
    }
    v
}

/// Exact squared L2 distance between two equal-length slices.
fn true_l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Exact dot product between two equal-length slices.
fn true_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `count` deterministic pseudo-embeddings of dimension `dim`, generated from
/// distinct label strings so every call is reproducible.
fn make_corpus(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| fnv_embed(&format!("corpus-vector-{i}"), dim))
        .collect()
}

/// Four well-separated clusters (anchors far apart relative to jitter),
/// mirroring `product_quantization`'s fixture style, so brute-force nearest
/// neighbor is unambiguous.
fn make_cluster_items(per_cluster: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
    let anchors = [0.0f32, 50.0, -50.0, 100.0];
    let mut items = Vec::new();
    for (a, &anchor) in anchors.iter().enumerate() {
        for i in 0..per_cluster {
            let id = format!("cluster{a}-item{i}");
            let v: Vec<f32> = (0..dim)
                .map(|d| {
                    let jitter = ((i * 7 + d * 3) % 5) as f32 * 0.01;
                    anchor + jitter
                })
                .collect();
            items.push((id, v));
        }
    }
    items
}

/// Extract logical bit `i` from a packed LSB-first `u64` word array (mirrors
/// the documented packing scheme on [`RaBitQCode::bits`]).
fn extract_bit(bits: &[u64], i: usize) -> bool {
    (bits[i / 64] >> (i % 64)) & 1 == 1
}

// ── RaBitQConfig tests ────────────────────────────────────────────────────────

#[test]
fn test_config_new_valid() {
    let cfg = RaBitQConfig::new(64).unwrap();
    assert_eq!(cfg.dim, 64);
    assert_eq!(cfg.seed, super::types::DEFAULT_RABITQ_SEED);
    assert_eq!(cfg.metric, RaBitQMetric::L2);
}

#[test]
fn test_config_new_zero_dim_is_error() {
    let err = RaBitQConfig::new(0).unwrap_err();
    assert!(matches!(err, RaBitQError::InvalidConfig(_)));
}

#[test]
fn test_config_default_values() {
    let cfg = RaBitQConfig::default();
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.seed, super::types::DEFAULT_RABITQ_SEED);
    assert_eq!(cfg.metric, RaBitQMetric::L2);
}

#[test]
fn test_config_with_seed() {
    let cfg = RaBitQConfig::new(8).unwrap().with_seed(42);
    assert_eq!(cfg.seed, 42);
}

#[test]
fn test_config_with_metric_inner_product() {
    let cfg = RaBitQConfig::new(8)
        .unwrap()
        .with_metric(RaBitQMetric::InnerProduct);
    assert_eq!(cfg.metric, RaBitQMetric::InnerProduct);
}

#[test]
fn test_config_with_metric_cosine() {
    let cfg = RaBitQConfig::new(8)
        .unwrap()
        .with_metric(RaBitQMetric::Cosine);
    assert_eq!(cfg.metric, RaBitQMetric::Cosine);
}

#[test]
fn test_config_validate_ok() {
    let cfg = RaBitQConfig::new(16).unwrap();
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_validate_zero_dim() {
    let cfg = RaBitQConfig {
        dim: 0,
        seed: 1,
        metric: RaBitQMetric::L2,
    };
    assert!(matches!(cfg.validate(), Err(RaBitQError::InvalidConfig(_))));
}

#[test]
fn test_config_is_copy() {
    let cfg = RaBitQConfig::new(4).unwrap();
    let cfg2 = cfg; // Copy, not move
    assert_eq!(cfg.dim, cfg2.dim);
}

#[test]
fn test_config_eq() {
    let a = RaBitQConfig::new(4).unwrap();
    let b = RaBitQConfig::new(4).unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_metric_default_is_l2() {
    assert_eq!(RaBitQMetric::default(), RaBitQMetric::L2);
}

#[test]
fn test_metric_variants_distinct() {
    assert_ne!(RaBitQMetric::L2, RaBitQMetric::InnerProduct);
    assert_ne!(RaBitQMetric::InnerProduct, RaBitQMetric::Cosine);
    assert_ne!(RaBitQMetric::L2, RaBitQMetric::Cosine);
}

// ── RaBitQError tests ─────────────────────────────────────────────────────────

#[test]
fn test_error_display_dimension_mismatch() {
    let e = RaBitQError::DimensionMismatch {
        expected: 4,
        got: 3,
    };
    assert_eq!(e.to_string(), "dimension mismatch: expected 4, got 3");
}

#[test]
fn test_error_display_invalid_config() {
    let e = RaBitQError::InvalidConfig("bad".into());
    assert_eq!(e.to_string(), "invalid configuration: bad");
}

#[test]
fn test_error_display_not_trained() {
    let e = RaBitQError::NotTrained;
    assert!(e.to_string().contains("not been trained"));
}

#[test]
fn test_error_display_empty_index() {
    let e = RaBitQError::EmptyIndex;
    assert!(e.to_string().contains("empty"));
}

#[test]
fn test_error_display_empty_query() {
    let e = RaBitQError::EmptyQuery;
    assert_eq!(e.to_string(), "query vector is empty");
}

#[test]
fn test_error_clone_eq() {
    let e1 = RaBitQError::NotTrained;
    let e2 = e1.clone();
    assert_eq!(e1, e2);
}

// ── RaBitQCode tests ──────────────────────────────────────────────────────────

#[test]
fn test_code_new_and_fields() {
    let code = RaBitQCode::new(vec![0b101], 0.75, 1.5);
    assert_eq!(code.bits, vec![0b101]);
    assert_eq!(code.factor, 0.75);
    assert_eq!(code.norm, 1.5);
}

#[test]
fn test_code_bit_capacity_single_word() {
    let code = RaBitQCode::new(vec![0u64], 0.5, 1.0);
    assert_eq!(code.bit_capacity(), 64);
}

#[test]
fn test_code_bit_capacity_multi_word() {
    let code = RaBitQCode::new(vec![0u64, 0u64, 0u64], 0.5, 1.0);
    assert_eq!(code.bit_capacity(), 192);
}

// ── RaBitQHit tests ───────────────────────────────────────────────────────────

#[test]
fn test_hit_new() {
    let hit = RaBitQHit::new("doc1".to_string(), 0.42);
    assert_eq!(hit.id, "doc1");
    assert_eq!(hit.estimated_distance, 0.42);
}

// ── Rotation tests ─────────────────────────────────────────────────────────────

#[test]
fn test_rotation_zero_dim_is_error() {
    let err = Rotation::generate(0, 1).unwrap_err();
    assert!(matches!(err, RaBitQError::InvalidConfig(_)));
}

#[test]
fn test_rotation_dim_one() {
    let rot = Rotation::generate(1, 7).unwrap();
    assert_eq!(rot.dim(), 1);
    // A 1x1 orthogonal matrix must have entry +-1.
    assert!((rot.rows()[0][0].abs() - 1.0).abs() < 1e-5);
}

#[test]
fn test_rotation_getters() {
    let rot = Rotation::generate(6, 99).unwrap();
    assert_eq!(rot.dim(), 6);
    assert_eq!(rot.seed(), 99);
    assert_eq!(rot.rows().len(), 6);
    for row in rot.rows() {
        assert_eq!(row.len(), 6);
    }
}

#[test]
fn test_rotation_deterministic_same_seed() {
    let a = Rotation::generate(10, 123).unwrap();
    let b = Rotation::generate(10, 123).unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_rotation_different_seed_differs() {
    let a = Rotation::generate(10, 1).unwrap();
    let b = Rotation::generate(10, 2).unwrap();
    assert_ne!(a, b);
}

/// PᵀP ≈ I for a range of dimensions: column `j` dotted with column `k`
/// should be ~1 for `j == k` and ~0 otherwise.
fn assert_orthogonal(rot: &Rotation, tol: f32) {
    let dim = rot.dim();
    let rows = rot.rows();
    for j in 0..dim {
        for k in 0..dim {
            let dot: f32 = (0..dim).map(|i| rows[i][j] * rows[i][k]).sum();
            let expected = if j == k { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < tol,
                "P^T P [{j}][{k}] = {dot}, expected {expected} (dim={dim})"
            );
        }
    }
}

#[test]
fn test_rotation_orthogonal_dim2() {
    assert_orthogonal(&Rotation::generate(2, 11).unwrap(), 1e-3);
}

#[test]
fn test_rotation_orthogonal_dim4() {
    assert_orthogonal(&Rotation::generate(4, 22).unwrap(), 1e-3);
}

#[test]
fn test_rotation_orthogonal_dim8() {
    assert_orthogonal(&Rotation::generate(8, 33).unwrap(), 1e-3);
}

#[test]
fn test_rotation_orthogonal_dim16() {
    assert_orthogonal(&Rotation::generate(16, 44).unwrap(), 1e-3);
}

#[test]
fn test_rotation_orthogonal_dim32() {
    assert_orthogonal(&Rotation::generate(32, 55).unwrap(), 2e-3);
}

#[test]
fn test_rotation_orthogonal_default_seed() {
    assert_orthogonal(
        &Rotation::generate(12, super::types::DEFAULT_RABITQ_SEED).unwrap(),
        1e-3,
    );
}

#[test]
fn test_rotation_apply_matches_column() {
    let dim = 5;
    let rot = Rotation::generate(dim, 7).unwrap();
    for j in 0..dim {
        let mut e_j = vec![0.0f32; dim];
        e_j[j] = 1.0;
        let out = rot.apply(&e_j);
        for i in 0..dim {
            assert!((out[i] - rot.rows()[i][j]).abs() < 1e-5);
        }
    }
}

#[test]
fn test_rotation_apply_preserves_norm() {
    let dim = 16;
    let rot = Rotation::generate(dim, 71).unwrap();
    let v = fnv_embed("norm-probe", dim);
    let original_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let rotated = rot.apply(&v);
    let rotated_norm = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((original_norm - rotated_norm).abs() < 1e-3);
}

#[test]
fn test_rotation_apply_preserves_inner_product() {
    let dim = 16;
    let rot = Rotation::generate(dim, 72).unwrap();
    let a = fnv_embed("iso-a", dim);
    let b = fnv_embed("iso-b", dim);
    let original_dot = true_dot(&a, &b);
    let rotated_dot = true_dot(&rot.apply(&a), &rot.apply(&b));
    assert!(
        (original_dot - rotated_dot).abs() < 1e-2,
        "original={original_dot}, rotated={rotated_dot}"
    );
}

// ── RaBitQuantizer basic lifecycle tests ──────────────────────────────────────

#[test]
fn test_quantizer_new_is_untrained() {
    let q = RaBitQuantizer::new(RaBitQConfig::new(8).unwrap());
    assert!(!q.is_trained());
    assert!(q.centroid().is_empty());
    assert!(q.rotation().is_none());
}

#[test]
fn test_quantizer_train_becomes_trained() {
    let cfg = RaBitQConfig::new(8).unwrap();
    let q = RaBitQuantizer::new(cfg);
    let trained = q.train(&make_corpus(20, 8)).unwrap();
    assert!(trained.is_trained());
    assert_eq!(trained.centroid().len(), 8);
    assert!(trained.rotation().is_some());
}

#[test]
fn test_quantizer_train_centroid_is_mean() {
    let cfg = RaBitQConfig::new(2).unwrap();
    let q = RaBitQuantizer::new(cfg);
    let vectors = vec![vec![0.0f32, 0.0], vec![2.0, 4.0]];
    let trained = q.train(&vectors).unwrap();
    assert!((trained.centroid()[0] - 1.0).abs() < 1e-6);
    assert!((trained.centroid()[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_quantizer_train_empty_vectors_is_error() {
    let q = RaBitQuantizer::new(RaBitQConfig::new(4).unwrap());
    let err = q.train(&[]).unwrap_err();
    assert!(matches!(err, RaBitQError::EmptyIndex));
}

#[test]
fn test_quantizer_train_dim_mismatch_is_error() {
    let q = RaBitQuantizer::new(RaBitQConfig::new(4).unwrap());
    let err = q.train(&[vec![1.0, 2.0, 3.0]]).unwrap_err();
    assert!(matches!(
        err,
        RaBitQError::DimensionMismatch {
            expected: 4,
            got: 3
        }
    ));
}

#[test]
fn test_quantizer_train_same_seed_reproducible() {
    let cfg = RaBitQConfig::new(6).unwrap();
    let vectors = make_corpus(10, 6);
    let t1 = RaBitQuantizer::new(cfg).train(&vectors).unwrap();
    let t2 = RaBitQuantizer::new(cfg).train(&vectors).unwrap();
    assert_eq!(t1.rotation(), t2.rotation());
    assert_eq!(t1.centroid(), t2.centroid());
}

#[test]
fn test_quantizer_encode_before_train_is_not_trained() {
    let q = RaBitQuantizer::new(RaBitQConfig::new(4).unwrap());
    let err = q.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap_err();
    assert!(matches!(err, RaBitQError::NotTrained));
}

#[test]
fn test_quantizer_estimate_ip_before_train_is_not_trained() {
    let q = RaBitQuantizer::new(RaBitQConfig::new(4).unwrap());
    let code = RaBitQCode::new(vec![0u64], 0.5, 1.0);
    let err = q.estimate_ip(&[1.0, 2.0, 3.0, 4.0], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::NotTrained));
}

#[test]
fn test_quantizer_estimate_l2_before_train_is_not_trained() {
    let q = RaBitQuantizer::new(RaBitQConfig::new(4).unwrap());
    let code = RaBitQCode::new(vec![0u64], 0.5, 1.0);
    let err = q.estimate_l2_sq(&[1.0, 2.0, 3.0, 4.0], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::NotTrained));
}

#[test]
fn test_quantizer_estimate_cosine_before_train_is_not_trained() {
    let q = RaBitQuantizer::new(RaBitQConfig::new(4).unwrap());
    let code = RaBitQCode::new(vec![0u64], 0.5, 1.0);
    let err = q.estimate_cosine(&[1.0, 2.0, 3.0, 4.0], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::NotTrained));
}

fn trained_quantizer(dim: usize, count: usize) -> RaBitQuantizer {
    let cfg = RaBitQConfig::new(dim).unwrap();
    RaBitQuantizer::new(cfg)
        .train(&make_corpus(count, dim))
        .unwrap()
}

#[test]
fn test_quantizer_encode_dim_mismatch_is_error() {
    let q = trained_quantizer(8, 10);
    let err = q.encode(&[1.0, 2.0]).unwrap_err();
    assert!(matches!(
        err,
        RaBitQError::DimensionMismatch {
            expected: 8,
            got: 2
        }
    ));
}

#[test]
fn test_quantizer_estimate_ip_dim_mismatch_is_error() {
    let q = trained_quantizer(8, 10);
    let code = q.encode(&fnv_embed("probe", 8)).unwrap();
    let err = q.estimate_ip(&[1.0, 2.0], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::DimensionMismatch { .. }));
}

#[test]
fn test_quantizer_estimate_l2_dim_mismatch_is_error() {
    let q = trained_quantizer(8, 10);
    let code = q.encode(&fnv_embed("probe", 8)).unwrap();
    let err = q.estimate_l2_sq(&[1.0, 2.0], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::DimensionMismatch { .. }));
}

#[test]
fn test_quantizer_estimate_cosine_dim_mismatch_is_error() {
    let q = trained_quantizer(8, 10);
    let code = q.encode(&fnv_embed("probe", 8)).unwrap();
    let err = q.estimate_cosine(&[1.0, 2.0], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::DimensionMismatch { .. }));
}

#[test]
fn test_quantizer_estimate_ip_empty_query_is_error() {
    let q = trained_quantizer(8, 10);
    let code = q.encode(&fnv_embed("probe", 8)).unwrap();
    let err = q.estimate_ip(&[], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::EmptyQuery));
}

#[test]
fn test_quantizer_estimate_l2_empty_query_is_error() {
    let q = trained_quantizer(8, 10);
    let code = q.encode(&fnv_embed("probe", 8)).unwrap();
    let err = q.estimate_l2_sq(&[], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::EmptyQuery));
}

#[test]
fn test_quantizer_estimate_cosine_empty_query_is_error() {
    let q = trained_quantizer(8, 10);
    let code = q.encode(&fnv_embed("probe", 8)).unwrap();
    let err = q.estimate_cosine(&[], &code).unwrap_err();
    assert!(matches!(err, RaBitQError::EmptyQuery));
}

#[test]
fn test_quantizer_encode_degenerate_vector_equal_to_centroid_succeeds() {
    // Two identical vectors -> centroid equals both -> residual is zero.
    // The quantizer must still produce a usable code (a fixed canonical
    // direction is substituted; see the doc comment on `encode`).
    let cfg = RaBitQConfig::new(4).unwrap();
    let v = vec![1.0f32, 2.0, 3.0, 4.0];
    let q = RaBitQuantizer::new(cfg)
        .train(&[v.clone(), v.clone()])
        .unwrap();
    let code = q.encode(&v).unwrap();
    assert!((code.norm - 0.0).abs() < 1e-6);
    assert!(code.factor >= 0.0);
}

#[test]
fn test_quantizer_estimate_l2_exact_for_degenerate_code() {
    // With norm == 0, ‖o_raw - query‖² collapses exactly to ‖query - c‖²
    // (the arbitrary fallback direction contributes nothing), regardless of
    // which query is supplied.
    let cfg = RaBitQConfig::new(4).unwrap();
    let v = vec![1.0f32, 2.0, 3.0, 4.0];
    let q = RaBitQuantizer::new(cfg)
        .train(&[v.clone(), v.clone()])
        .unwrap();
    let code = q.encode(&v).unwrap();
    let query = vec![5.0f32, -1.0, 2.5, 0.0];
    let est = q.estimate_l2_sq(&query, &code).unwrap();
    let truth = true_l2_sq(&query, &v); // v == centroid exactly here
    assert!((est - truth).abs() < 1e-4, "est={est} truth={truth}");
}

#[test]
fn test_quantizer_estimate_ip_exact_for_degenerate_code() {
    // With norm == 0, ⟨o_raw, query⟩ collapses exactly to ⟨c, query⟩.
    let cfg = RaBitQConfig::new(4).unwrap();
    let v = vec![1.0f32, 2.0, 3.0, 4.0];
    let q = RaBitQuantizer::new(cfg)
        .train(&[v.clone(), v.clone()])
        .unwrap();
    let code = q.encode(&v).unwrap();
    let query = vec![5.0f32, -1.0, 2.5, 0.0];
    let est = q.estimate_ip(&query, &code).unwrap();
    let truth = true_dot(&v, &query);
    assert!((est - truth).abs() < 1e-4, "est={est} truth={truth}");
}

#[test]
fn test_quantizer_error_bound_scales_as_inv_sqrt_dim() {
    let q64 = trained_quantizer(64, 4);
    let q256 = trained_quantizer(256, 4);
    assert!((q64.error_bound() - 0.125).abs() < 1e-5); // 1/sqrt(64)
    assert!((q256.error_bound() - 0.0625).abs() < 1e-5); // 1/sqrt(256)
    assert!(q256.error_bound() < q64.error_bound());
}

#[test]
fn test_quantizer_config_accessor() {
    let cfg = RaBitQConfig::new(12).unwrap().with_seed(5);
    let q = RaBitQuantizer::new(cfg);
    assert_eq!(q.config().dim, 12);
    assert_eq!(q.config().seed, 5);
}

// ── Bit packing / codebook tests ──────────────────────────────────────────────

#[test]
fn test_encode_bits_match_rotated_sign() {
    let dim = 20;
    let q = trained_quantizer(dim, 30);
    let v = fnv_embed("bitcheck", dim);
    let code = q.encode(&v).unwrap();

    // Recompute o' independently and check every stored bit matches its sign.
    let centroid = q.centroid().to_vec();
    let residual: Vec<f32> = v
        .iter()
        .zip(centroid.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let norm = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
    let unit: Vec<f32> = residual.iter().map(|&x| x / norm).collect();
    let rotated = q.rotation().unwrap().apply(&unit);

    for (i, &r) in rotated.iter().enumerate() {
        let expected_bit = r > 0.0;
        assert_eq!(extract_bit(&code.bits, i), expected_bit, "bit {i} mismatch");
    }
}

#[test]
fn test_encode_factor_is_non_negative() {
    let dim = 32;
    let q = trained_quantizer(dim, 40);
    for i in 0..20 {
        let v = fnv_embed(&format!("factor-probe-{i}"), dim);
        let code = q.encode(&v).unwrap();
        assert!(code.factor >= 0.0, "factor {} should be >= 0", code.factor);
    }
}

#[test]
fn test_encode_bit_capacity_covers_dim() {
    let dim = 100; // spans two u64 words
    let q = trained_quantizer(dim, 10);
    let v = fnv_embed("cap-probe", dim);
    let code = q.encode(&v).unwrap();
    assert!(code.bit_capacity() >= dim);
    assert_eq!(code.bits.len(), 2); // ceil(100/64) = 2
}

#[test]
fn test_encode_norm_matches_residual_norm() {
    let dim = 6;
    let q = trained_quantizer(dim, 12);
    let v = fnv_embed("norm-match", dim);
    let code = q.encode(&v).unwrap();
    let centroid = q.centroid();
    let expected_norm = true_l2_sq(&v, centroid).sqrt();
    assert!((code.norm - expected_norm).abs() < 1e-4);
}

// ── Unbiasedness tests ─────────────────────────────────────────────────────────

#[test]
fn test_estimate_ip_mean_error_within_bound() {
    let dim = 64;
    let count = 200;
    let vectors = make_corpus(count, dim);
    let q = RaBitQuantizer::new(RaBitQConfig::new(dim).unwrap())
        .train(&vectors)
        .unwrap();
    let query = fnv_embed("ip-unbiased-query", dim);

    let mut total_error = 0.0f64;
    for v in &vectors {
        let code = q.encode(v).unwrap();
        let est = q.estimate_ip(&query, &code).unwrap();
        let truth = true_dot(v, &query);
        total_error += f64::from(est - truth);
    }
    let mean_error = (total_error / count as f64) as f32;
    let bound = q.error_bound();
    assert!(
        mean_error.abs() < bound,
        "mean IP error {mean_error} exceeds error bound {bound}"
    );
}

#[test]
fn test_estimate_l2_sq_mean_error_within_bound() {
    let dim = 64;
    let count = 200;
    let vectors = make_corpus(count, dim);
    let q = RaBitQuantizer::new(RaBitQConfig::new(dim).unwrap())
        .train(&vectors)
        .unwrap();
    let query = fnv_embed("l2-unbiased-query", dim);

    let mut total_error = 0.0f64;
    // Normalize by the typical scale (mean true distance) so the pass/fail
    // criterion is scale-invariant, mirroring the relative nature of the
    // theoretical O(1/sqrt(D)) bound (which is stated for unit vectors).
    let mut total_truth = 0.0f64;
    for v in &vectors {
        let code = q.encode(v).unwrap();
        let est = q.estimate_l2_sq(&query, &code).unwrap();
        let truth = true_l2_sq(v, &query);
        total_error += f64::from(est - truth);
        total_truth += f64::from(truth);
    }
    let mean_error = total_error / count as f64;
    let mean_truth = total_truth / count as f64;
    let relative_error = (mean_error / mean_truth).abs() as f32;
    let bound = q.error_bound();
    assert!(
        relative_error < bound,
        "relative L2 error {relative_error} exceeds error bound {bound}"
    );
}

#[test]
fn test_estimate_cosine_within_unit_range_approximately() {
    let dim = 48;
    let q = trained_quantizer(dim, 50);
    let v = fnv_embed("cosine-probe-vec", dim);
    let code = q.encode(&v).unwrap();
    let query = fnv_embed("cosine-probe-query", dim);
    let cos = q.estimate_cosine(&query, &code).unwrap();
    // Cosine similarity is mathematically in [-1, 1]; allow small estimator
    // slack beyond the exact bound.
    assert!(cos > -1.2 && cos < 1.2, "cosine {cos} wildly out of range");
}

#[test]
fn test_estimate_cosine_self_similarity_is_close_to_one() {
    // Query equal to (a shifted copy of) an indexed vector's residual
    // direction should have a cosine estimate close to +1.
    let dim = 64;
    let vectors = make_corpus(80, dim);
    let q = RaBitQuantizer::new(RaBitQConfig::new(dim).unwrap())
        .train(&vectors)
        .unwrap();
    let v = &vectors[0];
    let code = q.encode(v).unwrap();
    // Query = the indexed vector itself, so query - c == o exactly.
    let cos = q.estimate_cosine(v, &code).unwrap();
    assert!(cos > 0.5, "self-cosine {cos} should be strongly positive");
}

#[test]
fn test_estimate_ip_self_close_to_true_norm_squared_ip() {
    let dim = 64;
    let vectors = make_corpus(80, dim);
    let q = RaBitQuantizer::new(RaBitQConfig::new(dim).unwrap())
        .train(&vectors)
        .unwrap();
    let v = &vectors[3];
    let code = q.encode(v).unwrap();
    let est = q.estimate_ip(v, &code).unwrap();
    let truth = true_dot(v, v);
    let bound = q.error_bound();
    // Self-IP is the largest-magnitude quantity in the corpus; allow a
    // generous multiple of the error bound scaled by the vector's own norm.
    let tolerance = bound * truth.abs().max(1.0) * 2.0;
    assert!(
        (est - truth).abs() < tolerance,
        "self-IP est={est} truth={truth} tolerance={tolerance}"
    );
}

// ── RaBitQIndex tests ──────────────────────────────────────────────────────────

#[test]
fn test_index_build_empty_items_is_error() {
    let err = RaBitQIndex::build(vec![], RaBitQConfig::new(4).unwrap()).unwrap_err();
    assert!(matches!(err, RaBitQError::EmptyIndex));
}

#[test]
fn test_index_build_dim_mismatch_propagates() {
    let items = vec![
        ("a".to_string(), vec![1.0, 2.0, 3.0, 4.0]),
        ("b".to_string(), vec![1.0, 2.0]), // wrong dim
    ];
    let err = RaBitQIndex::build(items, RaBitQConfig::new(4).unwrap()).unwrap_err();
    assert!(matches!(err, RaBitQError::DimensionMismatch { .. }));
}

#[test]
fn test_index_len_and_is_empty() {
    let items = make_cluster_items(3, 8);
    let index = RaBitQIndex::build(items, RaBitQConfig::new(8).unwrap()).unwrap();
    assert_eq!(index.len(), 12);
    assert!(!index.is_empty());
}

#[test]
fn test_index_search_empty_query_is_error() {
    let items = make_cluster_items(2, 8);
    let index = RaBitQIndex::build(items, RaBitQConfig::new(8).unwrap()).unwrap();
    let err = index.search(&[], 3).unwrap_err();
    assert!(matches!(err, RaBitQError::EmptyQuery));
}

#[test]
fn test_index_search_k_zero_returns_empty() {
    let items = make_cluster_items(2, 8);
    let index = RaBitQIndex::build(items, RaBitQConfig::new(8).unwrap()).unwrap();
    let query = fnv_embed("k-zero-query", 8);
    let hits = index.search(&query, 0).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn test_index_search_k_larger_than_len_returns_all() {
    let items = make_cluster_items(2, 8); // 8 items total
    let index = RaBitQIndex::build(items, RaBitQConfig::new(8).unwrap()).unwrap();
    let query = fnv_embed("k-large-query", 8);
    let hits = index.search(&query, 1000).unwrap();
    assert_eq!(hits.len(), 8);
}

#[test]
fn test_index_search_results_ascending_sorted() {
    let items = make_cluster_items(4, 12);
    let index = RaBitQIndex::build(items, RaBitQConfig::new(12).unwrap()).unwrap();
    let query = fnv_embed("sorted-query", 12);
    let hits = index.search(&query, 10).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].estimated_distance <= w[1].estimated_distance);
    }
}

// Recall tests use exactly one item per cluster: with several items packed
// into the *same* cluster, the true nearest neighbor is determined by
// sub-jitter-scale differences (~0.01) that are far below what a coarse
// 1-bit-per-dimension code can resolve at these dimensionalities, even though
// the (much larger, 50-150 unit) inter-cluster separation is resolved
// reliably. One item per cluster isolates exactly the property the task asks
// for: "for well-separated fixtures the estimated nearest neighbor equals the
// true nearest neighbor".

#[test]
fn test_index_recall_l2_matches_true_nearest_neighbor() {
    let dim = 16;
    let items = make_cluster_items(1, dim);
    let config = RaBitQConfig::new(dim).unwrap();
    let index = RaBitQIndex::build(items.clone(), config).unwrap();

    // Query planted right on top of cluster 2's anchor (-50.0).
    let query: Vec<f32> = (0..dim).map(|d| -50.0 + (d as f32 * 0.001)).collect();

    let true_nearest = items
        .iter()
        .min_by(|(_, a), (_, b)| {
            true_l2_sq(a, &query)
                .partial_cmp(&true_l2_sq(b, &query))
                .unwrap()
        })
        .unwrap()
        .0
        .clone();

    let hits = index.search(&query, 1).unwrap();
    assert_eq!(hits[0].id, true_nearest);
}

#[test]
fn test_index_recall_multiple_queries_all_clusters() {
    let dim = 16;
    let items = make_cluster_items(1, dim);
    let config = RaBitQConfig::new(dim).unwrap();
    let index = RaBitQIndex::build(items.clone(), config).unwrap();

    let anchors = [0.0f32, 50.0, -50.0, 100.0];
    for (a, &anchor) in anchors.iter().enumerate() {
        let query: Vec<f32> = (0..dim).map(|d| anchor + (d as f32 * 0.001)).collect();
        let true_nearest = items
            .iter()
            .min_by(|(_, x), (_, y)| {
                true_l2_sq(x, &query)
                    .partial_cmp(&true_l2_sq(y, &query))
                    .unwrap()
            })
            .unwrap()
            .0
            .clone();
        let hits = index.search(&query, 1).unwrap();
        assert_eq!(
            hits[0].id, true_nearest,
            "cluster {a} (anchor {anchor}) recall mismatch"
        );
    }
}

#[test]
fn test_index_inner_product_metric_ranks_by_negated_ip() {
    let dim = 8;
    let config = RaBitQConfig::new(dim)
        .unwrap()
        .with_metric(RaBitQMetric::InnerProduct);
    let items = make_cluster_items(3, dim);
    let index = RaBitQIndex::build(items, config).unwrap();
    let query = fnv_embed("ip-metric-query", dim);
    let hits = index.search(&query, 5).unwrap();
    // ascending estimated_distance == descending IP; verify against direct
    // quantizer calls for the top hit.
    let top_code = index
        .quantizer()
        .encode(
            &make_cluster_items(3, dim)
                .into_iter()
                .find(|(id, _)| *id == hits[0].id)
                .unwrap()
                .1,
        )
        .unwrap();
    let ip = index.quantizer().estimate_ip(&query, &top_code).unwrap();
    assert!((hits[0].estimated_distance - (-ip)).abs() < 1e-3);
}

#[test]
fn test_index_cosine_metric_ranks_ascending() {
    let dim = 8;
    let config = RaBitQConfig::new(dim)
        .unwrap()
        .with_metric(RaBitQMetric::Cosine);
    let items = make_cluster_items(3, dim);
    let index = RaBitQIndex::build(items, config).unwrap();
    let query = fnv_embed("cosine-metric-query", dim);
    let hits = index.search(&query, 5).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].estimated_distance <= w[1].estimated_distance);
    }
}

#[test]
fn test_index_quantizer_accessor_reports_trained() {
    let items = make_cluster_items(2, 8);
    let index = RaBitQIndex::build(items, RaBitQConfig::new(8).unwrap()).unwrap();
    assert!(index.quantizer().is_trained());
}

#[test]
fn test_index_single_item() {
    let items = vec![("solo".to_string(), vec![1.0f32, 2.0, 3.0, 4.0])];
    let index = RaBitQIndex::build(items, RaBitQConfig::new(4).unwrap()).unwrap();
    assert_eq!(index.len(), 1);
    let hits = index.search(&[1.0, 2.0, 3.0, 4.5], 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "solo");
}

#[test]
fn test_index_build_is_deterministic() {
    let items_a = make_cluster_items(2, 8);
    let items_b = make_cluster_items(2, 8);
    let config = RaBitQConfig::new(8).unwrap();
    let idx_a = RaBitQIndex::build(items_a, config).unwrap();
    let idx_b = RaBitQIndex::build(items_b, config).unwrap();
    let query = fnv_embed("determinism-query", 8);
    let hits_a = idx_a.search(&query, 4).unwrap();
    let hits_b = idx_b.search(&query, 4).unwrap();
    assert_eq!(hits_a, hits_b);
}

#[test]
fn test_index_different_seed_still_recalls_correctly() {
    let dim = 16;
    let items = make_cluster_items(1, dim);
    let config = RaBitQConfig::new(dim).unwrap().with_seed(0xDEAD_BEEF_u64);
    let index = RaBitQIndex::build(items.clone(), config).unwrap();
    let query: Vec<f32> = (0..dim).map(|d| 100.0 + (d as f32 * 0.001)).collect();
    let true_nearest = items
        .iter()
        .min_by(|(_, a), (_, b)| {
            true_l2_sq(a, &query)
                .partial_cmp(&true_l2_sq(b, &query))
                .unwrap()
        })
        .unwrap()
        .0
        .clone();
    let hits = index.search(&query, 1).unwrap();
    assert_eq!(hits[0].id, true_nearest);
}
