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
    clippy::single_char_pattern,
    clippy::absurd_extreme_comparisons
)]
//! Tests for the `scalar_quantization` module.
//!
//! Covers: `SqConfig` validation, `ScalarQuantizer` calibration, int8
//! encode/decode roundtrips, binary encoding, Hamming distance, asymmetric
//! dot product accuracy and reconstruction error bounds.

use super::{BinaryVector, QuantizedVector, ScalarQuantizer, SqConfig, SqError};

// ── Shared helpers ────────────────────────────────────────────────────────────

/// FNV-1a based deterministic embedding generator.
///
/// Produces a unit-normalised `f32` vector of length `dim` from `text`.
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
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// Build a calibrated `ScalarQuantizer` from FNV embeddings.
fn calibrated_sq(dim: usize, texts: &[&str]) -> ScalarQuantizer {
    let cfg = SqConfig::new().with_dim(dim).with_bits(8);
    let mut sq = ScalarQuantizer::new(cfg);
    let samples: Vec<Vec<f32>> = texts.iter().map(|t| fnv_embed(t, dim)).collect();
    sq.calibrate(&samples).unwrap();
    sq
}

/// Explicit dot product of two equal-length slices.
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── SqConfig tests ────────────────────────────────────────────────────────────

#[test]
fn test_config_default_values() {
    let cfg = SqConfig::default();
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.bits, 8);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(SqConfig::new(), SqConfig::default());
}

#[test]
fn test_config_with_dim() {
    let cfg = SqConfig::new().with_dim(64);
    assert_eq!(cfg.dim, 64);
    assert_eq!(cfg.bits, 8); // unchanged
}

#[test]
fn test_config_with_bits_1() {
    let cfg = SqConfig::new().with_bits(1);
    assert_eq!(cfg.bits, 1);
}

#[test]
fn test_config_with_bits_8() {
    let cfg = SqConfig::new().with_bits(8);
    assert_eq!(cfg.bits, 8);
}

#[test]
fn test_config_builder_chain() {
    let cfg = SqConfig::new().with_dim(32).with_bits(1);
    assert_eq!(cfg.dim, 32);
    assert_eq!(cfg.bits, 1);
}

#[test]
fn test_config_validate_ok_bits8() {
    let cfg = SqConfig::new().with_dim(4).with_bits(8);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_validate_ok_bits1() {
    let cfg = SqConfig::new().with_dim(4).with_bits(1);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_validate_zero_dim_errors() {
    let cfg = SqConfig::new().with_dim(0).with_bits(8);
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, SqError::InvalidConfig(_)));
}

#[test]
fn test_config_validate_bits_2_errors() {
    let cfg = SqConfig::new().with_dim(4).with_bits(2);
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, SqError::InvalidConfig(_)));
}

#[test]
fn test_config_validate_bits_4_errors() {
    let cfg = SqConfig::new().with_dim(4).with_bits(4);
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, SqError::InvalidConfig(_)));
}

#[test]
fn test_config_validate_bits_7_errors() {
    let cfg = SqConfig::new().with_dim(4).with_bits(7);
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, SqError::InvalidConfig(_)));
}

#[test]
fn test_config_validate_bits_0_errors() {
    let cfg = SqConfig::new().with_dim(4).with_bits(0);
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, SqError::InvalidConfig(_)));
}

// ── ScalarQuantizer construction ──────────────────────────────────────────────

#[test]
fn test_quantizer_new_uncalibrated() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    assert!(!sq.is_calibrated());
    assert!(sq.dim_min().is_empty());
    assert!(sq.dim_scale().is_empty());
}

#[test]
fn test_quantizer_config_accessor() {
    let cfg = SqConfig::new().with_dim(16).with_bits(1);
    let sq = ScalarQuantizer::new(cfg.clone());
    assert_eq!(sq.config().dim, 16);
    assert_eq!(sq.config().bits, 1);
}

// ── Calibration ───────────────────────────────────────────────────────────────

#[test]
fn test_calibrate_single_sample_succeeds() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(4));
    let result = sq.calibrate(&[vec![0.0, 1.0, -1.0, 0.5]]);
    assert!(result.is_ok());
    assert!(sq.is_calibrated());
}

#[test]
fn test_calibrate_multiple_samples_succeeds() {
    let samples: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32; 8]).collect();
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    assert!(sq.calibrate(&samples).is_ok());
    assert!(sq.is_calibrated());
}

#[test]
fn test_calibrate_populates_dim_min_and_scale() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(3));
    sq.calibrate(&[vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 3.0]])
        .unwrap();
    assert_eq!(sq.dim_min().len(), 3);
    assert_eq!(sq.dim_scale().len(), 3);
    assert_eq!(sq.dim_min()[0], -1.0);
    assert_eq!(sq.dim_min()[1], 0.0);
    assert_eq!(sq.dim_min()[2], 1.0);
}

#[test]
fn test_calibrate_scale_equals_range_over_255() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    sq.calibrate(&[vec![0.0], vec![255.0]]).unwrap();
    // scale should be (255 - 0) / 255 = 1.0
    let scale = sq.dim_scale()[0];
    assert!((scale - 1.0).abs() < 1e-5);
}

#[test]
fn test_calibrate_empty_set_errors() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(4));
    let err = sq.calibrate(&[]).unwrap_err();
    assert_eq!(err, SqError::EmptyTrainingSet);
}

#[test]
fn test_calibrate_dim_mismatch_errors() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(4));
    let err = sq.calibrate(&[vec![1.0, 2.0, 3.0]]).unwrap_err();
    assert!(matches!(
        err,
        SqError::DimMismatch {
            expected: 4,
            got: 3
        }
    ));
}

#[test]
fn test_calibrate_invalid_config_errors() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(0));
    let err = sq.calibrate(&[]).unwrap_err();
    assert!(matches!(err, SqError::InvalidConfig(_)));
}

#[test]
fn test_calibrate_can_be_called_twice() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(2));
    sq.calibrate(&[vec![0.0, 0.0], vec![1.0, 1.0]]).unwrap();
    // Second calibration with wider range replaces the first.
    sq.calibrate(&[vec![-10.0, -10.0], vec![10.0, 10.0]])
        .unwrap();
    assert_eq!(sq.dim_min()[0], -10.0);
    assert!(sq.is_calibrated());
}

#[test]
fn test_calibrate_constant_dimension_uses_tiny_scale() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(2));
    // Second dimension is constant across all samples.
    sq.calibrate(&[vec![0.0, 5.0], vec![1.0, 5.0], vec![2.0, 5.0]])
        .unwrap();
    // Zero-range dim → tiny scale floor, not NaN.
    let scale = sq.dim_scale()[1];
    assert!(scale > 0.0);
    assert!(scale.is_finite());
}

// ── Encode ────────────────────────────────────────────────────────────────────

#[test]
fn test_encode_before_calibrate_errors() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(4));
    let err = sq.encode(&[0.0; 4]).unwrap_err();
    assert_eq!(err, SqError::NotCalibrated);
}

#[test]
fn test_encode_dim_mismatch_errors() {
    let sq = calibrated_sq(4, &["a", "b"]);
    let err = sq.encode(&[0.0; 3]).unwrap_err();
    assert!(matches!(
        err,
        SqError::DimMismatch {
            expected: 4,
            got: 3
        }
    ));
}

#[test]
fn test_encode_produces_correct_length() {
    let sq = calibrated_sq(8, &["hello", "world", "rust"]);
    let v = fnv_embed("test", 8);
    let qv = sq.encode(&v).unwrap();
    assert_eq!(qv.data.len(), 8);
}

#[test]
fn test_encode_all_codes_in_i8_range() {
    let sq = calibrated_sq(32, &["foo", "bar", "baz", "qux"]);
    let v = fnv_embed("query", 32);
    let qv = sq.encode(&v).unwrap();
    for &code in &qv.data {
        assert!(code >= i8::MIN && code <= i8::MAX);
    }
}

#[test]
fn test_encode_min_value_maps_to_minus128() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    sq.calibrate(&[vec![-1.0], vec![1.0]]).unwrap();
    let qv = sq.encode(&[-1.0]).unwrap();
    assert_eq!(qv.data[0], -128_i8);
}

#[test]
fn test_encode_max_value_maps_to_127() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    sq.calibrate(&[vec![-1.0], vec![1.0]]).unwrap();
    let qv = sq.encode(&[1.0]).unwrap();
    assert_eq!(qv.data[0], 127_i8);
}

#[test]
fn test_encode_value_below_range_clamped_to_minus128() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    sq.calibrate(&[vec![0.0], vec![10.0]]).unwrap();
    // Value well below the calibration min.
    let qv = sq.encode(&[-100.0]).unwrap();
    assert_eq!(qv.data[0], -128_i8);
}

#[test]
fn test_encode_value_above_range_clamped_to_127() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    sq.calibrate(&[vec![0.0], vec![10.0]]).unwrap();
    // Value well above the calibration max.
    let qv = sq.encode(&[1000.0]).unwrap();
    assert_eq!(qv.data[0], 127_i8);
}

#[test]
fn test_encode_zero_vector_in_range() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(3));
    sq.calibrate(&[vec![-1.0, -1.0, -1.0], vec![1.0, 1.0, 1.0]])
        .unwrap();
    // Zero lies at the midpoint of [-1, 1] → code ≈ -1 or 0
    let qv = sq.encode(&[0.0, 0.0, 0.0]).unwrap();
    for &code in &qv.data {
        assert!(code == -1 || code == 0);
    }
}

#[test]
fn test_encode_deterministic_same_result() {
    let sq = calibrated_sq(16, &["stable", "output", "test"]);
    let v = fnv_embed("determinism", 16);
    let qv1 = sq.encode(&v).unwrap();
    let qv2 = sq.encode(&v).unwrap();
    assert_eq!(qv1.data, qv2.data);
}

#[test]
fn test_encode_scale_field_positive() {
    let sq = calibrated_sq(8, &["a", "b"]);
    let v = fnv_embed("x", 8);
    let qv = sq.encode(&v).unwrap();
    assert!(qv.scale > 0.0);
    assert!(qv.scale.is_finite());
}

// ── Decode ────────────────────────────────────────────────────────────────────

#[test]
fn test_decode_uncalibrated_returns_empty() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(4));
    let qv = QuantizedVector {
        data: vec![0_i8; 4],
        scale: 1.0,
        offset: 0.0,
    };
    let result = sq.decode(&qv);
    assert!(result.is_empty());
}

#[test]
fn test_decode_produces_correct_length() {
    let sq = calibrated_sq(8, &["hello", "world"]);
    let v = fnv_embed("decode_len", 8);
    let qv = sq.encode(&v).unwrap();
    let decoded = sq.decode(&qv);
    assert_eq!(decoded.len(), 8);
}

#[test]
fn test_decode_wrong_length_qv_returns_empty() {
    let sq = calibrated_sq(8, &["a", "b"]);
    let qv = QuantizedVector {
        data: vec![0_i8; 5], // wrong length
        scale: 1.0,
        offset: 0.0,
    };
    let result = sq.decode(&qv);
    assert!(result.is_empty());
}

#[test]
fn test_decode_min_code_reconstructs_near_min() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    sq.calibrate(&[vec![-2.0], vec![2.0]]).unwrap();
    let qv = sq.encode(&[-2.0]).unwrap();
    let decoded = sq.decode(&qv);
    assert!((decoded[0] - (-2.0)).abs() < 0.02);
}

#[test]
fn test_decode_max_code_reconstructs_near_max() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    sq.calibrate(&[vec![-2.0], vec![2.0]]).unwrap();
    let qv = sq.encode(&[2.0]).unwrap();
    let decoded = sq.decode(&qv);
    assert!((decoded[0] - 2.0).abs() < 0.02);
}

// ── Encode / decode roundtrip ─────────────────────────────────────────────────

#[test]
fn test_encode_decode_roundtrip_unit_norm_vectors() {
    let sq = calibrated_sq(16, &["alpha", "beta", "gamma", "delta", "epsilon"]);
    let v = fnv_embed("roundtrip", 16);
    let qv = sq.encode(&v).unwrap();
    let decoded = sq.decode(&qv);
    // Maximum error per dimension: scale / 2
    let max_scale = sq
        .dim_scale()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    for (&orig, &rec) in v.iter().zip(decoded.iter()) {
        assert!(
            (orig - rec).abs() <= max_scale + 1e-6,
            "orig={orig}, rec={rec}, max_scale={max_scale}"
        );
    }
}

#[test]
fn test_encode_decode_reconstruction_error_within_half_step() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(4));
    sq.calibrate(&[vec![0.0; 4], vec![1.0; 4]]).unwrap();
    let v = vec![0.1, 0.4, 0.7, 0.9];
    let qv = sq.encode(&v).unwrap();
    let decoded = sq.decode(&qv);
    // step = 1.0/255 ≈ 0.00392; allow half-step + epsilon
    let half_step = 1.0_f32 / 255.0 / 2.0 + 1e-5;
    for (&orig, &rec) in v.iter().zip(decoded.iter()) {
        assert!(
            (orig - rec).abs() <= half_step,
            "orig={orig}, rec={rec}, half_step={half_step}"
        );
    }
}

#[test]
fn test_roundtrip_many_fnv_vectors() {
    let texts = [
        "the quick brown fox",
        "jumped over the lazy dog",
        "rust is a systems language",
        "scalar quantization test",
    ];
    let sq = calibrated_sq(32, &texts);
    for t in &texts {
        let v = fnv_embed(t, 32);
        let qv = sq.encode(&v).unwrap();
        let decoded = sq.decode(&qv);
        let max_scale = sq
            .dim_scale()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        for (&orig, &rec) in v.iter().zip(decoded.iter()) {
            assert!((orig - rec).abs() <= max_scale + 1e-5);
        }
    }
}

#[test]
fn test_roundtrip_l2_error_bounded() {
    let sq = calibrated_sq(
        64,
        &[
            "l2 error test alpha",
            "l2 error test beta",
            "l2 error test gamma",
        ],
    );
    let v = fnv_embed("l2 error query", 64);
    let qv = sq.encode(&v).unwrap();
    let decoded = sq.decode(&qv);
    let l2_err: f32 = v
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    // For unit vectors in [-1,1] encoded at 8 bits, expect < 0.1 L2 error.
    assert!(l2_err < 0.1, "L2 error {l2_err} too large");
}

// ── Binary encoding ───────────────────────────────────────────────────────────

#[test]
fn test_encode_binary_all_positive() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    let bv = sq.encode_binary(&[1.0, 0.5, 0.1, 2.0, 0.01, 3.0, 0.001, 100.0]);
    // All positive → all bits set → 0xFF
    assert_eq!(bv.data[0], 0xFF);
}

#[test]
fn test_encode_binary_all_negative() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    let bv = sq.encode_binary(&[-1.0, -0.5, -0.1, -2.0, -0.01, -3.0, -0.001, -100.0]);
    // All non-positive → all bits clear → 0x00
    assert_eq!(bv.data[0], 0x00);
}

#[test]
fn test_encode_binary_zero_is_not_set() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    let bv = sq.encode_binary(&[0.0; 8]);
    // Threshold is strictly > 0; zero maps to 0.
    assert_eq!(bv.data[0], 0x00);
}

#[test]
fn test_encode_binary_mixed() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    // Alternating +/-: bits 0,2,4,6 set = 0b10101010 = 0xAA
    let bv = sq.encode_binary(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    assert_eq!(bv.data[0], 0xAA);
}

#[test]
fn test_encode_binary_dim_field() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(13));
    let bv = sq.encode_binary(&vec![1.0; 13]);
    assert_eq!(bv.dim, 13);
}

#[test]
fn test_encode_binary_byte_length_aligned() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    let bv = sq.encode_binary(&[1.0; 8]);
    assert_eq!(bv.data.len(), 1); // 8 bits → 1 byte
}

#[test]
fn test_encode_binary_byte_length_unaligned() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(9));
    let bv = sq.encode_binary(&[1.0; 9]);
    assert_eq!(bv.data.len(), 2); // 9 bits → 2 bytes
}

#[test]
fn test_encode_binary_packing_msb_first() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    // Only bit 0 (MSB of byte 0) set.
    let mut v = [0.0_f32; 8];
    v[0] = 1.0;
    let bv = sq.encode_binary(&v);
    assert_eq!(bv.data[0], 0b1000_0000);
}

#[test]
fn test_encode_binary_second_byte() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(16));
    // Only dim 8 (first bit of second byte) set.
    let mut v = [0.0_f32; 16];
    v[8] = 1.0;
    let bv = sq.encode_binary(&v);
    assert_eq!(bv.data[0], 0x00);
    assert_eq!(bv.data[1], 0b1000_0000);
}

#[test]
fn test_encode_binary_empty_vector() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(0));
    let bv = sq.encode_binary(&[]);
    assert!(bv.data.is_empty());
    assert_eq!(bv.dim, 0);
}

#[test]
fn test_encode_binary_single_dim_positive() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    let bv = sq.encode_binary(&[0.1]);
    // Bit 7 of byte 0 should be set (MSB-first, dim 0 → position 7).
    assert_eq!(bv.data[0], 0b1000_0000);
}

#[test]
fn test_encode_binary_lsb_of_first_byte() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    // Only dim 7 (LSB of byte 0) set.
    let mut v = [0.0_f32; 8];
    v[7] = 1.0;
    let bv = sq.encode_binary(&v);
    assert_eq!(bv.data[0], 0b0000_0001);
}

#[test]
fn test_encode_binary_does_not_require_calibration() {
    // encode_binary works even before calibrate.
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(4));
    let bv = sq.encode_binary(&[1.0, -1.0, 1.0, -1.0]);
    assert_eq!(bv.dim, 4);
}

// ── Hamming distance ──────────────────────────────────────────────────────────

#[test]
fn test_hamming_identical_is_zero() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    let bv = sq.encode_binary(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    assert_eq!(sq.hamming_distance(&bv, &bv), 0);
}

#[test]
fn test_hamming_all_different_8_bits() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    let a = BinaryVector {
        data: vec![0xFF],
        dim: 8,
    };
    let b = BinaryVector {
        data: vec![0x00],
        dim: 8,
    };
    assert_eq!(sq.hamming_distance(&a, &b), 8);
}

#[test]
fn test_hamming_one_bit_different() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    let a = BinaryVector {
        data: vec![0b1000_0000],
        dim: 8,
    };
    let b = BinaryVector {
        data: vec![0b0000_0000],
        dim: 8,
    };
    assert_eq!(sq.hamming_distance(&a, &b), 1);
}

#[test]
fn test_hamming_known_value() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    // 0xAA = 10101010, 0x55 = 01010101 → XOR = 0xFF → 8 bits differ
    let a = BinaryVector {
        data: vec![0xAA],
        dim: 8,
    };
    let b = BinaryVector {
        data: vec![0x55],
        dim: 8,
    };
    assert_eq!(sq.hamming_distance(&a, &b), 8);
}

#[test]
fn test_hamming_symmetric() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(8));
    let a = sq.encode_binary(&[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]);
    let b = sq.encode_binary(&[-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    assert_eq!(sq.hamming_distance(&a, &b), sq.hamming_distance(&b, &a));
}

#[test]
fn test_hamming_multi_byte() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(16));
    // All bits set vs. all bits clear across two bytes → 16 differences.
    let a = BinaryVector {
        data: vec![0xFF, 0xFF],
        dim: 16,
    };
    let b = BinaryVector {
        data: vec![0x00, 0x00],
        dim: 16,
    };
    assert_eq!(sq.hamming_distance(&a, &b), 16);
}

#[test]
fn test_hamming_triangle_inequality() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(32));
    let va = fnv_embed("triangle_a", 32);
    let vb = fnv_embed("triangle_b", 32);
    let vc = fnv_embed("triangle_c", 32);
    let ba = sq.encode_binary(&va);
    let bb = sq.encode_binary(&vb);
    let bc = sq.encode_binary(&vc);
    let dab = sq.hamming_distance(&ba, &bb);
    let dbc = sq.hamming_distance(&bb, &bc);
    let dac = sq.hamming_distance(&ba, &bc);
    assert!(dac <= dab + dbc, "triangle inequality violated");
}

#[test]
fn test_hamming_fnv_different_texts_nonzero() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(64));
    let ba = sq.encode_binary(&fnv_embed("alice", 64));
    let bb = sq.encode_binary(&fnv_embed("bob", 64));
    // Different texts produce different binary codes with high probability.
    let dist = sq.hamming_distance(&ba, &bb);
    assert!(dist > 0, "distinct texts should differ in at least one bit");
}

// ── Asymmetric dot product ────────────────────────────────────────────────────

#[test]
fn test_asymmetric_dot_uncalibrated_returns_zero() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(4));
    let qv = QuantizedVector {
        data: vec![0_i8; 4],
        scale: 1.0,
        offset: 0.0,
    };
    let dot = sq.asymmetric_dot(&[1.0; 4], &qv);
    assert_eq!(dot, 0.0);
}

#[test]
fn test_asymmetric_dot_length_mismatch_returns_zero() {
    let sq = calibrated_sq(4, &["a", "b"]);
    let v = fnv_embed("v", 4);
    let qv = sq.encode(&v).unwrap();
    // Query length ≠ qv length.
    let dot = sq.asymmetric_dot(&[1.0; 5], &qv);
    assert_eq!(dot, 0.0);
}

#[test]
fn test_asymmetric_dot_equals_explicit_dot_on_decoded() {
    let sq = calibrated_sq(16, &["doc1", "doc2", "doc3", "doc4"]);
    let v = fnv_embed("query_for_dot", 16);
    let q = fnv_embed("the_query", 16);
    let qv = sq.encode(&v).unwrap();
    let decoded = sq.decode(&qv);

    let adot = sq.asymmetric_dot(&q, &qv);
    let exact_dot = dot_f32(&q, &decoded);
    // The two should be identical (same arithmetic).
    assert!(
        (adot - exact_dot).abs() < 1e-4,
        "asymmetric_dot={adot}, explicit_dot={exact_dot}"
    );
}

#[test]
fn test_asymmetric_dot_approximates_true_dot() {
    let sq = calibrated_sq(
        32,
        &[
            "approx dot test a",
            "approx dot test b",
            "approx dot test c",
        ],
    );
    let v = fnv_embed("database vector", 32);
    let q = fnv_embed("float query", 32);
    let qv = sq.encode(&v).unwrap();

    let adot = sq.asymmetric_dot(&q, &qv);
    let true_dot = dot_f32(&q, &v);
    // Encoding introduces at most O(max_scale * sqrt(dim)) error in dot.
    let max_scale = sq.dim_scale().iter().copied().fold(0.0_f32, f32::max);
    let error_bound = max_scale * (32.0_f32).sqrt() + 1e-5;
    assert!(
        (adot - true_dot).abs() <= error_bound,
        "adot={adot}, true_dot={true_dot}, error_bound={error_bound}"
    );
}

#[test]
fn test_asymmetric_dot_zero_query_returns_near_zero() {
    let sq = calibrated_sq(8, &["a", "b"]);
    let v = fnv_embed("v", 8);
    let qv = sq.encode(&v).unwrap();
    let dot = sq.asymmetric_dot(&[0.0; 8], &qv);
    assert!(dot.abs() < 1e-6, "zero query should give ~0 dot, got {dot}");
}

#[test]
fn test_asymmetric_dot_cosine_approximation_for_unit_vectors() {
    let texts = ["cosine a", "cosine b", "cosine c", "cosine d", "cosine e"];
    let sq = calibrated_sq(32, &texts);
    let q = fnv_embed("cosine query", 32);
    let v = fnv_embed("cosine database", 32);
    let qv = sq.encode(&v).unwrap();

    let adot = sq.asymmetric_dot(&q, &qv);
    let true_cosine = dot_f32(&q, &v); // both already unit-normalised by fnv_embed
    // Expect the approximation to be within 0.05 of the true cosine.
    assert!(
        (adot - true_cosine).abs() < 0.05,
        "adot={adot}, true_cosine={true_cosine}"
    );
}

// ── QuantizedVector and BinaryVector field invariants ─────────────────────────

#[test]
fn test_quantized_vector_scale_positive_after_encode() {
    let sq = calibrated_sq(4, &["p", "q"]);
    let qv = sq.encode(&fnv_embed("r", 4)).unwrap();
    assert!(qv.scale > 0.0 && qv.scale.is_finite());
}

#[test]
fn test_quantized_vector_offset_finite() {
    let sq = calibrated_sq(4, &["p", "q"]);
    let qv = sq.encode(&fnv_embed("r", 4)).unwrap();
    assert!(qv.offset.is_finite());
}

#[test]
fn test_binary_vector_dim_field_preserved() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(13));
    let bv = sq.encode_binary(&vec![-1.0_f32; 13]);
    assert_eq!(bv.dim, 13);
    assert_eq!(bv.data.len(), 2); // ceil(13/8) = 2
}

// ── Misc / edge case tests ────────────────────────────────────────────────────

#[test]
fn test_encode_fnv_vectors_no_panic() {
    let sq = calibrated_sq(
        128,
        &[
            "no panic test a",
            "no panic test b",
            "no panic test c",
            "no panic test d",
        ],
    );
    for i in 0..10 {
        let v = fnv_embed(&format!("vector {i}"), 128);
        sq.encode(&v).unwrap();
    }
}

#[test]
fn test_encode_and_encode_binary_consistent_on_same_vector() {
    let sq = calibrated_sq(8, &["consistent a", "consistent b"]);
    let v = fnv_embed("consistency", 8);
    let qv = sq.encode(&v).unwrap();
    let bv = sq.encode_binary(&v);
    // Both should succeed and have proper lengths.
    assert_eq!(qv.data.len(), 8);
    assert_eq!(bv.dim, 8);
    assert_eq!(bv.data.len(), 1);
}

#[test]
fn test_hamming_empty_vectors() {
    let sq = ScalarQuantizer::new(SqConfig::new().with_dim(0));
    let a = BinaryVector {
        data: vec![],
        dim: 0,
    };
    let b = BinaryVector {
        data: vec![],
        dim: 0,
    };
    assert_eq!(sq.hamming_distance(&a, &b), 0);
}

#[test]
fn test_calibrate_single_dim_vector() {
    let mut sq = ScalarQuantizer::new(SqConfig::new().with_dim(1));
    sq.calibrate(&[vec![0.0], vec![1.0]]).unwrap();
    let qv = sq.encode(&[0.5]).unwrap();
    let decoded = sq.decode(&qv);
    assert!((decoded[0] - 0.5).abs() < 0.01);
}

#[test]
fn test_sq_error_display_not_calibrated() {
    let err = SqError::NotCalibrated;
    let s = format!("{err}");
    assert!(s.contains("calibrate"));
}

#[test]
fn test_sq_error_display_dim_mismatch() {
    let err = SqError::DimMismatch {
        expected: 8,
        got: 4,
    };
    let s = format!("{err}");
    assert!(s.contains("8") && s.contains("4"));
}

#[test]
fn test_sq_error_display_empty_training_set() {
    let err = SqError::EmptyTrainingSet;
    let s = format!("{err}");
    assert!(s.contains("calibration") || s.contains("sample"));
}

#[test]
fn test_encode_large_dim_no_panic() {
    let texts: Vec<String> = (0..10).map(|i| format!("large {i}")).collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let sq = calibrated_sq(256, &text_refs);
    let v = fnv_embed("large query", 256);
    let qv = sq.encode(&v).unwrap();
    assert_eq!(qv.data.len(), 256);
}

#[test]
fn test_roundtrip_consistent_across_multiple_calls() {
    let sq = calibrated_sq(16, &["stable1", "stable2", "stable3"]);
    let v = fnv_embed("multi_call", 16);
    let qv1 = sq.encode(&v).unwrap();
    let decoded1 = sq.decode(&qv1);
    let qv2 = sq.encode(&v).unwrap();
    let decoded2 = sq.decode(&qv2);
    // Multiple encode→decode calls should give identical results.
    for (&d1, &d2) in decoded1.iter().zip(decoded2.iter()) {
        assert_eq!(d1, d2);
    }
}
