#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::useless_vec,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
//! Tests for the `residual_vq` module.
//!
//! Covers configuration validation, the residual cascade's monotone error
//! guarantee, encode/decode round-trip quality, additive reconstruction, the
//! asymmetric additive-lookup distance identity, symmetric-vs-asymmetric
//! accuracy, greedy-vs-beam encoding, recall against a brute-force baseline and
//! the `num_stages == 1` / small-training-set edge cases.

use super::index::ResidualVqIndex;
use super::quantizer::ResidualQuantizer;
use super::types::{ResidualCode, ResidualVqConfig, ResidualVqError};
use crate::types::DocumentId;

// ── Deterministic data helpers ──────────────────────────────────────────────────

/// FNV-1a hash of a sequence of `u64` words — deterministic pseudo-randomness
/// for building reproducible test data without the `rand` crate.
fn fnv_words(words: &[u64]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &w in words {
        for &b in &w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
    }
    h
}

/// Map a hash to `[0, 1)`.
fn unit01(h: u64) -> f32 {
    (h % 1_000_003) as f32 / 1_000_003.0
}

/// Build `num_clusters * per_cluster` vectors of length `dim`, drawn from
/// well-separated cluster centres with small deterministic jitter.
fn cluster_data(num_clusters: usize, per_cluster: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(num_clusters * per_cluster);
    for c in 0..num_clusters {
        let center: Vec<f32> = (0..dim)
            .map(|d| unit01(fnv_words(&[0xC0, c as u64, d as u64])) * 10.0 - 5.0)
            .collect();
        for p in 0..per_cluster {
            let v: Vec<f32> = center
                .iter()
                .enumerate()
                .map(|(d, &cv)| {
                    cv + (unit01(fnv_words(&[0x0E, c as u64, p as u64, d as u64])) * 0.4 - 0.2)
                })
                .collect();
            out.push(v);
        }
    }
    out
}

/// Squared L2 distance.
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Squared L2 norm.
fn norm_sq(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

/// Brute-force exact top-k indices by squared L2.
fn brute_force_topk(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut d: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (sq_l2(query, v), i))
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    d.into_iter().take(k).map(|(_, i)| i).collect()
}

/// A moderately sized, well-clustered config used across several tests.
fn base_config() -> ResidualVqConfig {
    ResidualVqConfig::new()
        .with_dim(16)
        .with_num_stages(4)
        .with_codebook_size(8)
        .with_max_kmeans_iterations(15)
}

// ── Config validation ───────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let c = ResidualVqConfig::new();
    assert_eq!(c.num_stages, 4);
    assert_eq!(c.codebook_size, 256);
    assert_eq!(c.dim, 128);
    assert_eq!(c.beam_width, 1);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_rejects_zero_stages() {
    let c = ResidualVqConfig::new().with_num_stages(0);
    assert!(matches!(
        c.validate(),
        Err(ResidualVqError::InvalidConfig { .. })
    ));
}

#[test]
fn test_config_rejects_zero_codebook_size() {
    let c = ResidualVqConfig::new().with_codebook_size(0);
    assert!(matches!(
        c.validate(),
        Err(ResidualVqError::InvalidConfig { .. })
    ));
}

#[test]
fn test_config_rejects_zero_dim() {
    let c = ResidualVqConfig::new().with_dim(0);
    assert!(matches!(
        c.validate(),
        Err(ResidualVqError::InvalidConfig { .. })
    ));
}

#[test]
fn test_config_rejects_zero_beam_width() {
    let c = ResidualVqConfig::new().with_beam_width(0);
    assert!(matches!(
        c.validate(),
        Err(ResidualVqError::InvalidConfig { .. })
    ));
}

// ── Untrained state ─────────────────────────────────────────────────────────────

#[test]
fn test_untrained_quantizer_errors() {
    let q = ResidualQuantizer::new(base_config());
    assert!(!q.is_trained());
    assert!(q.codebooks().is_empty());
    assert!(q.stage_errors().is_empty());
    assert!(q.total_error().is_none());
    assert!(matches!(
        q.encode(&vec![0.0; 16]),
        Err(ResidualVqError::NotTrained)
    ));
    assert!(matches!(
        q.inner_product_tables(&vec![0.0; 16]),
        Err(ResidualVqError::NotTrained)
    ));
    // Decode of an untrained quantizer yields zeros of the configured length.
    let zeros = q.decode(&ResidualCode::new(vec![0, 0, 0, 0]));
    assert_eq!(zeros, vec![0.0f32; 16]);
}

#[test]
fn test_train_empty_set_errors() {
    let mut q = ResidualQuantizer::new(base_config());
    assert!(matches!(
        q.train(&[]),
        Err(ResidualVqError::EmptyTrainingSet)
    ));
}

#[test]
fn test_train_dim_mismatch_errors() {
    let mut q = ResidualQuantizer::new(base_config());
    let bad = vec![vec![0.0f32; 8], vec![0.0f32; 8]];
    assert!(matches!(
        q.train(&bad),
        Err(ResidualVqError::DimMismatch {
            expected: 16,
            got: 8
        })
    ));
}

// ── Training shape ──────────────────────────────────────────────────────────────

#[test]
fn test_train_builds_m_codebooks_of_k() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(6, 20, 16);
    q.train(&data).expect("train");
    assert!(q.is_trained());
    assert_eq!(q.codebooks().len(), 4, "one codebook per stage");
    for codebook in q.codebooks() {
        assert_eq!(codebook.len(), 8, "each stage holds K codewords");
        for codeword in codebook {
            assert_eq!(codeword.len(), 16, "each codeword is full-dim");
        }
    }
    assert_eq!(q.stage_errors().len(), 4);
}

#[test]
fn test_training_is_deterministic() {
    let data = cluster_data(5, 15, 16);
    let mut a = ResidualQuantizer::new(base_config());
    let mut b = ResidualQuantizer::new(base_config());
    a.train(&data).expect("train a");
    b.train(&data).expect("train b");
    assert_eq!(
        a.codebooks(),
        b.codebooks(),
        "training must be reproducible"
    );
    assert_eq!(a.stage_errors(), b.stage_errors());
}

// ── Monotone reconstruction error (key correctness property) ─────────────────────

#[test]
fn test_stage_errors_non_increasing() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(8, 25, 16);
    q.train(&data).expect("train");

    let errors = q.stage_errors();
    assert_eq!(errors.len(), 4);

    // Every stage may only reduce (never increase) the total residual energy.
    for w in errors.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "stage error must be non-increasing: {} !<= {}",
            w[1],
            w[0]
        );
    }

    // The first stage may not exceed the raw input energy either.
    let input_energy: f64 = data.iter().map(|v| f64::from(norm_sq(v))).sum();
    assert!(errors[0] <= input_energy + 1e-6);
}

#[test]
fn test_more_stages_lower_error() {
    let data = cluster_data(6, 20, 16);
    let mut short = ResidualQuantizer::new(base_config().with_num_stages(2));
    let mut long = ResidualQuantizer::new(base_config().with_num_stages(6));
    short.train(&data).expect("train short");
    long.train(&data).expect("train long");
    let short_err = short.total_error().expect("short err");
    let long_err = long.total_error().expect("long err");
    assert!(
        long_err <= short_err + 1e-6,
        "more stages must not increase error: {long_err} vs {short_err}"
    );
}

// ── Encode / decode round-trip ──────────────────────────────────────────────────

#[test]
fn test_encode_produces_one_index_per_stage() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(6, 20, 16);
    q.train(&data).expect("train");
    let code = q.encode(&data[0]).expect("encode");
    assert_eq!(code.len(), 4);
    for &ci in &code.indices {
        assert!(ci < 8, "each index must be within codebook size");
    }
}

#[test]
fn test_decode_is_additive_sum_of_codewords() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(6, 20, 16);
    q.train(&data).expect("train");
    let code = q.encode(&data[3]).expect("encode");
    let decoded = q.decode(&code);

    // Manually add up the selected codewords across stages.
    let mut manual = vec![0.0f32; 16];
    for (m, codebook) in q.codebooks().iter().enumerate() {
        let cw = &codebook[code.indices[m]];
        for (a, b) in manual.iter_mut().zip(cw.iter()) {
            *a += b;
        }
    }
    for (d, m) in decoded.iter().zip(manual.iter()) {
        assert!((d - m).abs() < 1e-5);
    }
}

#[test]
fn test_reconstruction_quality() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(8, 25, 16);
    q.train(&data).expect("train");

    let mut total_err = 0.0f32;
    let mut total_energy = 0.0f32;
    for v in &data {
        let code = q.encode(v).expect("encode");
        total_err += q.reconstruction_error(v, &code).expect("recon error");
        total_energy += norm_sq(v);
    }
    // RVQ over well-separated clusters should capture almost all the energy.
    assert!(
        total_err < 0.05 * total_energy,
        "reconstruction error {total_err} should be tiny vs energy {total_energy}"
    );
}

// ── Asymmetric distance identity ────────────────────────────────────────────────

#[test]
fn test_asymmetric_distance_matches_exact_reconstruction_distance() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(6, 20, 16);
    q.train(&data).expect("train");

    // A query that is not in the training set.
    let query: Vec<f32> = (0..16)
        .map(|d| unit01(fnv_words(&[0xEE, d as u64])) * 2.0 - 1.0)
        .collect();

    for v in data.iter().take(30) {
        let code = q.encode(v).expect("encode");
        let asym = q.asymmetric_distance(&query, &code).expect("asym");
        let exact = sq_l2(&query, &q.decode(&code));
        assert!(
            (asym - exact).abs() <= 1e-2 + 1e-3 * exact.abs(),
            "asymmetric {asym} must equal exact reconstruction distance {exact}"
        );
    }
}

#[test]
fn test_distance_from_tables_matches_asymmetric() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(6, 20, 16);
    q.train(&data).expect("train");

    let query = data[7].clone();
    let tables = q.inner_product_tables(&query).expect("tables");
    let query_norm_sq = ResidualQuantizer::query_norm_sq(&query);

    for v in data.iter().take(20) {
        let code = q.encode(v).expect("encode");
        let recon_norm_sq = q.reconstruction_norm_sq(&code);
        let table_dist = ResidualQuantizer::distance_from_ip_tables(
            &tables,
            &code,
            query_norm_sq,
            recon_norm_sq,
        );
        let direct = q.asymmetric_distance(&query, &code).expect("asym");
        assert!((table_dist - direct).abs() <= 1e-2 + 1e-3 * direct.abs());
    }
}

// ── Symmetric vs asymmetric accuracy ────────────────────────────────────────────

#[test]
fn test_asymmetric_at_least_as_accurate_as_symmetric() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(8, 20, 16);
    q.train(&data).expect("train");

    let mut asym_error = 0.0f32;
    let mut sym_error = 0.0f32;
    // Use several data points as queries against many database vectors.
    for qi in (0..data.len()).step_by(11) {
        let query = &data[qi];
        let query_code = q.encode(query).expect("encode query");
        for v in data.iter().take(40) {
            let code = q.encode(v).expect("encode");
            let truth = sq_l2(query, v);
            let asym = q.asymmetric_distance(query, &code).expect("asym");
            let sym = q.symmetric_distance(&query_code, &code).expect("sym");
            asym_error += (asym - truth).abs();
            sym_error += (sym - truth).abs();
        }
    }
    // Keeping the query full-precision should be no worse, on aggregate, than
    // quantizing both sides.
    assert!(
        asym_error <= sym_error + 1e-3,
        "asymmetric aggregate error {asym_error} should not exceed symmetric {sym_error}"
    );
}

// ── Greedy vs beam encoding ─────────────────────────────────────────────────────

#[test]
fn test_beam_beats_greedy_on_crafted_codebooks() {
    // Two stages, two codewords each, dim 2. Greedy commits to the locally best
    // stage-0 codeword and pays for it; beam explores the alternative and wins.
    let config = ResidualVqConfig::new()
        .with_dim(2)
        .with_num_stages(2)
        .with_codebook_size(2);
    let codebooks = vec![
        vec![vec![1.1f32, 0.0], vec![0.5f32, 0.0]],
        vec![vec![0.5f32, 0.0], vec![0.0f32, 0.0]],
    ];
    let q = ResidualQuantizer::from_codebooks(config, codebooks).expect("from_codebooks");

    let x = vec![1.0f32, 0.0];
    let greedy = q.encode_greedy(&x).expect("greedy");
    let beam = q.encode_beam(&x, 2).expect("beam");

    let greedy_err = q.reconstruction_error(&x, &greedy).expect("greedy err");
    let beam_err = q.reconstruction_error(&x, &beam).expect("beam err");

    assert_eq!(greedy.indices, vec![0, 1]);
    assert_eq!(beam.indices, vec![1, 0]);
    assert!(
        beam_err < greedy_err,
        "beam error {beam_err} must be strictly below greedy {greedy_err}"
    );
    assert!(beam_err < 1e-6, "beam reconstructs x exactly here");
}

#[test]
fn test_beam_never_worse_than_greedy_on_trained_data() {
    let mut q = ResidualQuantizer::new(base_config().with_codebook_size(6));
    let data = cluster_data(10, 20, 16);
    q.train(&data).expect("train");

    let mut greedy_total = 0.0f32;
    let mut beam_total = 0.0f32;
    for v in &data {
        let g = q.encode_greedy(v).expect("greedy");
        let b = q.encode_beam(v, 4).expect("beam");
        greedy_total += q.reconstruction_error(v, &g).expect("g err");
        beam_total += q.reconstruction_error(v, &b).expect("b err");
    }
    assert!(
        beam_total <= greedy_total + 1e-4,
        "beam total {beam_total} must not exceed greedy total {greedy_total}"
    );
}

#[test]
fn test_encode_width_one_equals_greedy() {
    let mut q = ResidualQuantizer::new(base_config());
    let data = cluster_data(6, 20, 16);
    q.train(&data).expect("train");
    for v in data.iter().take(20) {
        let via_encode = q.encode_beam(v, 1).expect("width1");
        let greedy = q.encode_greedy(v).expect("greedy");
        assert_eq!(via_encode.indices, greedy.indices);
    }
}

// ── Edge cases ──────────────────────────────────────────────────────────────────

#[test]
fn test_single_stage_is_plain_kmeans() {
    let config = base_config().with_num_stages(1);
    let mut q = ResidualQuantizer::new(config);
    let data = cluster_data(5, 20, 16);
    q.train(&data).expect("train");
    assert_eq!(q.codebooks().len(), 1);
    assert_eq!(q.stage_errors().len(), 1);

    let code = q.encode(&data[0]).expect("encode");
    assert_eq!(code.len(), 1);
    // Reconstruction is exactly the single chosen codeword (nearest centroid).
    let decoded = q.decode(&code);
    let centroid = &q.codebooks()[0][code.indices[0]];
    for (d, c) in decoded.iter().zip(centroid.iter()) {
        assert!((d - c).abs() < 1e-6);
    }
}

#[test]
fn test_training_set_smaller_than_k() {
    // Only 3 vectors but K = 8: k-means must clamp to the available points and
    // pad the codebook so every index stays valid.
    let config = ResidualVqConfig::new()
        .with_dim(4)
        .with_num_stages(2)
        .with_codebook_size(8)
        .with_max_kmeans_iterations(5);
    let mut q = ResidualQuantizer::new(config);
    let data = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.0f32, 1.0, 0.0, 0.0],
        vec![0.0f32, 0.0, 1.0, 0.0],
    ];
    q.train(&data).expect("train");
    for codebook in q.codebooks() {
        assert_eq!(codebook.len(), 8, "codebook padded to K");
    }
    // Every vector still encodes and decodes.
    for v in &data {
        let code = q.encode(v).expect("encode");
        assert_eq!(code.len(), 2);
        let err = q.reconstruction_error(v, &code).expect("err");
        assert!(err.is_finite());
    }
    // Monotone property still holds on tiny data.
    for w in q.stage_errors().windows(2) {
        assert!(w[1] <= w[0] + 1e-6);
    }
}

#[test]
fn test_from_codebooks_validation() {
    let config = ResidualVqConfig::new()
        .with_dim(2)
        .with_num_stages(2)
        .with_codebook_size(2);

    // Wrong number of stages.
    let bad_stages = vec![vec![vec![0.0f32, 0.0], vec![1.0, 1.0]]];
    assert!(matches!(
        ResidualQuantizer::from_codebooks(config.clone(), bad_stages),
        Err(ResidualVqError::InvalidConfig { .. })
    ));

    // Wrong codeword dimensionality.
    let bad_dim = vec![
        vec![vec![0.0f32, 0.0, 0.0], vec![1.0, 1.0, 1.0]],
        vec![vec![0.0f32, 0.0, 0.0], vec![1.0, 1.0, 1.0]],
    ];
    assert!(matches!(
        ResidualQuantizer::from_codebooks(config, bad_dim),
        Err(ResidualVqError::DimMismatch { .. })
    ));
}

// ── Index build + search ────────────────────────────────────────────────────────

#[test]
fn test_index_empty_build_errors() {
    let mut index = ResidualVqIndex::new(base_config());
    assert!(index.is_empty());
    assert!(matches!(
        index.build(&[]),
        Err(ResidualVqError::EmptyTrainingSet)
    ));
}

#[test]
fn test_index_search_before_build_errors() {
    let index = ResidualVqIndex::new(base_config());
    assert!(matches!(
        index.search(&vec![0.0; 16], 5),
        Err(ResidualVqError::NotTrained)
    ));
}

#[test]
fn test_index_search_dim_mismatch_errors() {
    let mut index = ResidualVqIndex::new(base_config());
    let data = cluster_data(4, 10, 16);
    let items: Vec<(DocumentId, Vec<f32>)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (DocumentId::from(format!("d{i}")), v.clone()))
        .collect();
    index.build(&items).expect("build");
    assert!(matches!(
        index.search(&vec![0.0; 8], 5),
        Err(ResidualVqError::DimMismatch { .. })
    ));
}

#[test]
fn test_index_recall_vs_brute_force() {
    let data = cluster_data(10, 20, 16);
    let items: Vec<(DocumentId, Vec<f32>)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (DocumentId::from(format!("doc{i}")), v.clone()))
        .collect();

    let mut index = ResidualVqIndex::new(base_config());
    index.build(&items).expect("build");
    assert_eq!(index.len(), data.len());

    let top_k = 5;
    let mut hits = 0usize;
    for (qi, v) in data.iter().enumerate() {
        // Query is the exact database vector; its true nearest neighbour is
        // itself (distance 0). Check the RVQ index retrieves it in the top-k.
        let truth = brute_force_topk(v, &data, 1)[0];
        assert_eq!(truth, qi);
        let results = index.search(v, top_k).expect("search");
        let ids: Vec<&str> = results.iter().map(|h| h.id.as_str()).collect();
        if ids.contains(&format!("doc{qi}").as_str()) {
            hits += 1;
        }
    }
    let recall = hits as f32 / data.len() as f32;
    assert!(
        recall >= 0.95,
        "recall@{top_k} was {recall}, expected >= 0.95"
    );
}

#[test]
fn test_index_search_orders_ascending_and_truncates() {
    let data = cluster_data(6, 15, 16);
    let items: Vec<(DocumentId, Vec<f32>)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (DocumentId::from(format!("v{i}")), v.clone()))
        .collect();
    let mut index = ResidualVqIndex::new(base_config());
    index.build(&items).expect("build");

    let results = index.search(&data[0], 5).expect("search");
    assert_eq!(results.len(), 5, "top_k truncation");
    for w in results.windows(2) {
        assert!(w[0].distance <= w[1].distance, "ascending distance order");
    }
}

#[test]
fn test_index_symmetric_search_runs() {
    let data = cluster_data(6, 15, 16);
    let items: Vec<(DocumentId, Vec<f32>)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (DocumentId::from(format!("s{i}")), v.clone()))
        .collect();
    let mut index = ResidualVqIndex::new(base_config());
    index.build(&items).expect("build");

    let results = index
        .search_symmetric(&data[0], 3)
        .expect("symmetric search");
    assert_eq!(results.len(), 3);
    for w in results.windows(2) {
        assert!(w[0].distance <= w[1].distance);
    }
}
