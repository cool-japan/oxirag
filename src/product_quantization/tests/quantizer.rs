//! Tests for [`ProductQuantizer`]: training, encode/decode and ADC.

use super::{make_vectors, small_config, sq_l2};
use crate::product_quantization::quantizer::ProductQuantizer;
use crate::product_quantization::types::{PqCode, PqConfig, PqError};

// ── Untrained state ───────────────────────────────────────────────────────────

#[test]
fn test_quantizer_new_untrained() {
    let q = ProductQuantizer::new(small_config());
    assert!(!q.is_trained());
    assert!(q.codebooks().is_empty());
    assert_eq!(q.num_subspaces(), 2);
}

#[test]
fn test_quantizer_config_accessor() {
    let q = ProductQuantizer::new(small_config());
    assert_eq!(q.config().dim, 8);
    assert_eq!(q.config().num_subspaces, 2);
}

// ── Training ──────────────────────────────────────────────────────────────────

#[test]
fn test_train_sets_trained_flag() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(20, 8);
    q.train(&vectors).expect("train should succeed");
    assert!(q.is_trained());
}

#[test]
fn test_train_builds_m_codebooks() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(20, 8);
    q.train(&vectors).expect("train should succeed");
    // M = num_subspaces = 2 codebooks.
    assert_eq!(q.codebooks().len(), 2);
}

#[test]
fn test_train_codebooks_have_k_centroids() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let k = small_config().codebook_size(); // 4
    for codebook in q.codebooks() {
        assert_eq!(codebook.len(), k, "each codebook must hold K centroids");
    }
}

#[test]
fn test_train_centroids_have_subspace_dim() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let sub_dim = small_config().subspace_dim(); // 4
    for codebook in q.codebooks() {
        for centroid in codebook {
            assert_eq!(centroid.len(), sub_dim);
        }
    }
}

#[test]
fn test_train_larger_config() {
    let cfg = PqConfig::new()
        .with_dim(16)
        .with_num_subspaces(4)
        .with_codebook_bits(3);
    let mut q = ProductQuantizer::new(cfg.clone());
    let vectors = make_vectors(60, 16);
    q.train(&vectors).expect("train should succeed");
    assert_eq!(q.codebooks().len(), 4);
    for codebook in q.codebooks() {
        assert_eq!(codebook.len(), cfg.codebook_size()); // 8
        for c in codebook {
            assert_eq!(c.len(), 4);
        }
    }
}

#[test]
fn test_train_more_centroids_than_points() {
    // K = 4 but only 3 distinct points; codebook still padded to K.
    let cfg = small_config();
    let mut q = ProductQuantizer::new(cfg.clone());
    let vectors = vec![vec![0.0; 8], vec![1.0; 8], vec![2.0; 8]];
    q.train(&vectors).expect("train should succeed");
    for codebook in q.codebooks() {
        assert_eq!(codebook.len(), cfg.codebook_size());
    }
}

// ── Training errors ───────────────────────────────────────────────────────────

#[test]
fn test_train_empty_set_errors() {
    let mut q = ProductQuantizer::new(small_config());
    let err = q.train(&[]).unwrap_err();
    assert_eq!(err, PqError::EmptyTrainingSet);
    assert!(!q.is_trained());
}

#[test]
fn test_train_invalid_config_errors() {
    let cfg = PqConfig::new().with_dim(10).with_num_subspaces(3);
    let mut q = ProductQuantizer::new(cfg);
    let vectors = vec![vec![0.0; 10]; 5];
    assert!(matches!(
        q.train(&vectors),
        Err(PqError::InvalidConfig { .. })
    ));
}

#[test]
fn test_train_dim_mismatch_errors() {
    let mut q = ProductQuantizer::new(small_config());
    // Vectors are length 6, config dim is 8.
    let vectors = vec![vec![0.0; 6]; 5];
    assert_eq!(q.train(&vectors).unwrap_err(), PqError::DimMismatch);
}

// ── Encode ────────────────────────────────────────────────────────────────────

#[test]
fn test_encode_produces_num_subspaces_codes() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(20, 8);
    q.train(&vectors).expect("train should succeed");
    let code = q.encode(&vectors[0]).expect("encode should succeed");
    assert_eq!(code.len(), 2);
}

#[test]
fn test_encode_codes_within_codebook_range() {
    let cfg = small_config();
    let mut q = ProductQuantizer::new(cfg.clone());
    let vectors = make_vectors(30, 8);
    q.train(&vectors).expect("train should succeed");
    let k = cfg.codebook_size();
    for v in &vectors {
        let code = q.encode(v).expect("encode should succeed");
        for &c in &code.codes {
            assert!(usize::from(c) < k, "code index {c} out of range {k}");
        }
    }
}

#[test]
fn test_encode_before_train_errors() {
    let q = ProductQuantizer::new(small_config());
    assert_eq!(q.encode(&vec![0.0; 8]).unwrap_err(), PqError::NotTrained);
}

#[test]
fn test_encode_dim_mismatch_errors() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(20, 8);
    q.train(&vectors).expect("train should succeed");
    assert_eq!(q.encode(&vec![0.0; 6]).unwrap_err(), PqError::DimMismatch);
}

#[test]
fn test_encode_picks_nearest_centroid() {
    // A vector encodes to the centroid nearest to each of its subvectors.
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let sub_dim = small_config().subspace_dim();
    for v in vectors.iter().take(8) {
        let code = q.encode(v).expect("encode should succeed");
        for (s, &ci) in code.codes.iter().enumerate() {
            let start = s * sub_dim;
            let sub = &v[start..start + sub_dim];
            let codebook = &q.codebooks()[s];
            let chosen = sq_l2(sub, &codebook[ci as usize]);
            for centroid in codebook {
                assert!(
                    chosen <= sq_l2(sub, centroid) + 1e-6,
                    "encode did not pick the nearest centroid"
                );
            }
        }
    }
}

// ── Decode ────────────────────────────────────────────────────────────────────

#[test]
fn test_decode_length_matches_dim() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(20, 8);
    q.train(&vectors).expect("train should succeed");
    let code = q.encode(&vectors[0]).expect("encode should succeed");
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), 8);
}

#[test]
fn test_decode_untrained_returns_zeros() {
    let q = ProductQuantizer::new(small_config());
    let code = PqCode::new(vec![0, 0]);
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), 8);
    assert!(decoded.iter().all(|&x| x == 0.0));
}

#[test]
fn test_decode_reconstructs_centroids() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(20, 8);
    q.train(&vectors).expect("train should succeed");
    let code = q.encode(&vectors[0]).expect("encode should succeed");
    let decoded = q.decode(&code);
    let sub_dim = small_config().subspace_dim();
    // Each decoded subvector must equal the selected centroid exactly.
    for (s, &ci) in code.codes.iter().enumerate() {
        let start = s * sub_dim;
        assert_eq!(
            &decoded[start..start + sub_dim],
            &q.codebooks()[s][ci as usize][..]
        );
    }
}

// ── Round-trip & reconstruction ───────────────────────────────────────────────

#[test]
fn test_encode_decode_roundtrip_stable() {
    // Encoding a decoded vector reproduces the same code (idempotent re-encode).
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    for v in vectors.iter().take(10) {
        let code = q.encode(v).expect("encode should succeed");
        let decoded = q.decode(&code);
        let recode = q.encode(&decoded).expect("re-encode should succeed");
        assert_eq!(code, recode, "round-trip code changed");
    }
}

#[test]
fn test_reconstruction_error_small() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let mut total_err = 0.0f32;
    for v in &vectors {
        let code = q.encode(v).expect("encode should succeed");
        let decoded = q.decode(&code);
        total_err += sq_l2(v, &decoded);
    }
    let mean_err = total_err / vectors.len() as f32;
    // Tight, well-separated clusters reconstruct with near-zero residual.
    assert!(mean_err < 1.0, "reconstruction error too high: {mean_err}");
}

// ── ADC ───────────────────────────────────────────────────────────────────────

#[test]
fn test_adc_equals_decoded_distance() {
    // ADC distance must equal the true squared-L2 distance to decode(code).
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let query = vectors[3].clone();
    for v in vectors.iter().take(12) {
        let code = q.encode(v).expect("encode should succeed");
        let adc = q
            .asymmetric_distance(&query, &code)
            .expect("adc should succeed");
        let decoded = q.decode(&code);
        let truth = sq_l2(&query, &decoded);
        assert!(
            (adc - truth).abs() < 1e-3,
            "ADC {adc} != decoded distance {truth}"
        );
    }
}

#[test]
fn test_adc_approximates_true_distance_same_cluster() {
    // Within a well-reconstructed cluster, ADC tracks the true squared-L2
    // closely: query and compared vectors share an anchor, so the quantization
    // residual folded into ADC is small in absolute terms.
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let query = vectors[0].clone(); // cluster 0
    // Indices 0, 4, 8, 12, 16 all share cluster 0 (i % 4 == 0).
    for &idx in &[0usize, 4, 8, 12, 16] {
        let v = &vectors[idx];
        let code = q.encode(v).expect("encode should succeed");
        let adc = q
            .asymmetric_distance(&query, &code)
            .expect("adc should succeed");
        let true_dist = sq_l2(&query, v);
        assert!(
            (adc - true_dist).abs() < 1.0,
            "ADC {adc} far from true {true_dist} (idx {idx})"
        );
    }
}

#[test]
fn test_adc_preserves_distance_ordering() {
    // The defining property of ADC for retrieval: a query ranks a near vector
    // below a far vector, matching the true squared-L2 ordering.
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let query = vectors[0].clone(); // cluster 0
    let near = &vectors[4]; // cluster 0 — close
    let far = &vectors[1]; // cluster 20 — far
    assert!(sq_l2(&query, near) < sq_l2(&query, far));

    let near_code = q.encode(near).expect("encode should succeed");
    let far_code = q.encode(far).expect("encode should succeed");
    let near_adc = q
        .asymmetric_distance(&query, &near_code)
        .expect("adc should succeed");
    let far_adc = q
        .asymmetric_distance(&query, &far_code)
        .expect("adc should succeed");
    assert!(
        near_adc < far_adc,
        "ADC broke distance ordering: near {near_adc} >= far {far_adc}"
    );
}

#[test]
fn test_adc_self_distance_near_zero() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let v = &vectors[0];
    let code = q.encode(v).expect("encode should succeed");
    let adc = q.asymmetric_distance(v, &code).expect("adc should succeed");
    // Distance from a vector to its own centroid is the (small) residual.
    assert!(adc < 1.0, "self ADC unexpectedly large: {adc}");
}

#[test]
fn test_adc_before_train_errors() {
    let q = ProductQuantizer::new(small_config());
    let code = PqCode::new(vec![0, 0]);
    assert_eq!(
        q.asymmetric_distance(&vec![0.0; 8], &code).unwrap_err(),
        PqError::NotTrained
    );
}

#[test]
fn test_adc_query_dim_mismatch_errors() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(20, 8);
    q.train(&vectors).expect("train should succeed");
    let code = q.encode(&vectors[0]).expect("encode should succeed");
    assert_eq!(
        q.asymmetric_distance(&vec![0.0; 6], &code).unwrap_err(),
        PqError::DimMismatch
    );
}

#[test]
fn test_adc_code_length_mismatch_errors() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(20, 8);
    q.train(&vectors).expect("train should succeed");
    // Code with wrong number of subspaces (3 instead of 2).
    let bad = PqCode::new(vec![0, 0, 0]);
    assert_eq!(
        q.asymmetric_distance(&vectors[0], &bad).unwrap_err(),
        PqError::DimMismatch
    );
}

// ── Distance tables ───────────────────────────────────────────────────────────

#[test]
fn test_distance_tables_shape() {
    let cfg = small_config();
    let mut q = ProductQuantizer::new(cfg.clone());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let tables = q.distance_tables(&vectors[0]).expect("tables should build");
    assert_eq!(tables.len(), cfg.num_subspaces);
    for row in &tables {
        assert_eq!(row.len(), cfg.codebook_size());
    }
}

#[test]
fn test_distance_from_tables_matches_adc() {
    let mut q = ProductQuantizer::new(small_config());
    let vectors = make_vectors(40, 8);
    q.train(&vectors).expect("train should succeed");
    let query = &vectors[2];
    let tables = q.distance_tables(query).expect("tables should build");
    for v in vectors.iter().take(10) {
        let code = q.encode(v).expect("encode should succeed");
        let from_tables = ProductQuantizer::distance_from_tables(&tables, &code);
        let adc = q
            .asymmetric_distance(query, &code)
            .expect("adc should succeed");
        assert!((from_tables - adc).abs() < 1e-4);
    }
}

#[test]
fn test_distance_tables_untrained_errors() {
    let q = ProductQuantizer::new(small_config());
    assert_eq!(
        q.distance_tables(&vec![0.0; 8]).unwrap_err(),
        PqError::NotTrained
    );
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_training_deterministic() {
    let vectors = make_vectors(50, 8);
    let mut q1 = ProductQuantizer::new(small_config());
    let mut q2 = ProductQuantizer::new(small_config());
    q1.train(&vectors).expect("train should succeed");
    q2.train(&vectors).expect("train should succeed");
    assert_eq!(
        q1.codebooks(),
        q2.codebooks(),
        "identical inputs must yield identical codebooks"
    );
}

#[test]
fn test_encoding_deterministic() {
    let vectors = make_vectors(50, 8);
    let mut q1 = ProductQuantizer::new(small_config());
    let mut q2 = ProductQuantizer::new(small_config());
    q1.train(&vectors).expect("train should succeed");
    q2.train(&vectors).expect("train should succeed");
    for v in &vectors {
        assert_eq!(q1.encode(v).expect("encode"), q2.encode(v).expect("encode"));
    }
}

#[test]
fn test_training_deterministic_across_runs() {
    // Re-training the same quantizer twice reproduces the codebooks.
    let vectors = make_vectors(30, 8);
    let mut q = ProductQuantizer::new(small_config());
    q.train(&vectors).expect("train should succeed");
    let first = q.codebooks().to_vec();
    q.train(&vectors).expect("re-train should succeed");
    assert_eq!(first, q.codebooks());
}
