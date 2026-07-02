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
//! Tests for the `anisotropic_vq` module.
//!
//! Covers: `AnisotropicVqConfig` validation and builders; the parallel /
//! orthogonal residual decomposition (orthogonality, exact reconstruction, the
//! zero-vector convention) and its agreement with the weighted loss; the
//! weighted-least-squares centroid update (recovers the plain mean at
//! multiplier `1.0`, and provably reduces the parallel residual as the
//! multiplier grows — the defining property of anisotropic VQ); weighted-Lloyd
//! loss-trajectory monotonicity; encode / decode round-trips; asymmetric
//! score identities; end-to-end recall against a brute-force baseline for both
//! metrics and for product-style (multi-subspace) codebooks; determinism; and
//! every error path plus the tricky edge cases (single training vector,
//! all-identical vectors, zero vectors).

use super::index::AnisotropicVqIndex;
use super::quantizer::{AnisotropicQuantizer, anisotropic_loss, decompose_residual};
use super::types::{
    AnisotropicCode, AnisotropicVqConfig, AnisotropicVqError, AnisotropicVqHit,
    AnisotropicVqMetric, DEFAULT_ANISOTROPIC_VQ_SEED, MAX_CODEBOOK_SIZE,
};
use crate::types::DocumentId;

// ── Shared fixtures ───────────────────────────────────────────────────────────

/// FNV-1a + splitmix64 deterministic pseudo-embedding generator. Produces a
/// length-`dim` `f32` vector from `text`, entries in `[-1, 1)`; fully
/// reproducible.
///
/// The text is hashed with FNV-1a, then each component mixes in the position
/// index and runs a full splitmix64 finalizer. The strong finalizer is
/// essential: extracting only the high 32 bits of a *plain* FNV hash lets a
/// single-byte (or index) change reach only the mid bits, collapsing every
/// component — and every vector — to nearly the same value (and direction),
/// which would silently make anisotropic and isotropic quantization coincide.
fn fnv_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut base: u64 = 14_695_981_039_346_656_037;
    for &b in text.as_bytes() {
        base ^= u64::from(b);
        base = base.wrapping_mul(1_099_511_628_211);
    }
    (0..dim)
        .map(|i| {
            let mut h = base ^ (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            h ^= h >> 30;
            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            h ^= h >> 27;
            h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
            h ^= h >> 31;
            (h >> 32) as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect()
}

/// `count` deterministic pseudo-embeddings of dimensionality `dim`.
fn make_corpus(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| fnv_embed(&format!("corpus-vector-{i}"), dim))
        .collect()
}

/// Exact inner product between two equal-length slices (f64 accumulation).
fn true_dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// Exact squared L2 distance between two equal-length slices (f64).
fn true_l2_sq(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum()
}

/// Euclidean norm of a slice (f64).
fn norm(a: &[f32]) -> f64 {
    a.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

/// Sum of squared entries of a slice (f64).
fn sum_sq(a: &[f32]) -> f64 {
    a.iter().map(|&x| f64::from(x) * f64::from(x)).sum()
}

// ── Config ──────────────────────────────────────────────────────────────────

#[test]
fn config_default_is_valid() {
    let config = AnisotropicVqConfig::default();
    assert_eq!(config.dim, 128);
    assert_eq!(config.num_codewords, 256);
    assert_eq!(config.num_subspaces, 1);
    assert_eq!(config.parallel_weight_multiplier, 4.0);
    assert_eq!(config.orthogonal_weight(), 1.0);
    assert_eq!(config.metric, AnisotropicVqMetric::InnerProduct);
    assert_eq!(config.seed, DEFAULT_ANISOTROPIC_VQ_SEED);
    assert!(config.validate().is_ok());
    assert_eq!(config.subspace_dim(), 128);
}

#[test]
fn config_builders_chain() {
    let config = AnisotropicVqConfig::new()
        .with_dim(64)
        .with_num_codewords(32)
        .with_num_subspaces(4)
        .with_parallel_weight_multiplier(8.0)
        .with_max_iterations(7)
        .with_convergence_tolerance(1e-6)
        .with_seed(42)
        .with_metric(AnisotropicVqMetric::L2);
    assert_eq!(config.dim, 64);
    assert_eq!(config.num_codewords, 32);
    assert_eq!(config.num_subspaces, 4);
    assert_eq!(config.parallel_weight_multiplier, 8.0);
    assert_eq!(config.max_iterations, 7);
    assert_eq!(config.convergence_tolerance, 1e-6);
    assert_eq!(config.seed, 42);
    assert_eq!(config.metric, AnisotropicVqMetric::L2);
    assert_eq!(config.subspace_dim(), 16);
    assert!(config.validate().is_ok());
}

#[test]
fn config_validation_rejects_bad_fields() {
    let bad = [
        AnisotropicVqConfig::new().with_dim(0),
        AnisotropicVqConfig::new().with_num_subspaces(0),
        AnisotropicVqConfig::new()
            .with_dim(10)
            .with_num_subspaces(3),
        AnisotropicVqConfig::new().with_num_codewords(0),
        AnisotropicVqConfig::new().with_num_codewords(MAX_CODEBOOK_SIZE + 1),
        AnisotropicVqConfig::new().with_parallel_weight_multiplier(0.5),
        AnisotropicVqConfig::new().with_parallel_weight_multiplier(f64::NAN),
        AnisotropicVqConfig::new().with_max_iterations(0),
        AnisotropicVqConfig::new().with_convergence_tolerance(-1.0),
        AnisotropicVqConfig::new().with_convergence_tolerance(f64::INFINITY),
    ];
    for config in bad {
        assert!(
            matches!(config.validate(), Err(AnisotropicVqError::InvalidConfig(_))),
            "expected InvalidConfig for {config:?}"
        );
    }
}

#[test]
fn config_multiplier_one_is_valid() {
    // multiplier == 1.0 recovers isotropic k-means and must be accepted.
    assert!(
        AnisotropicVqConfig::new()
            .with_parallel_weight_multiplier(1.0)
            .validate()
            .is_ok()
    );
}

// ── Residual decomposition ────────────────────────────────────────────────────

#[test]
fn decomposition_is_orthogonal_and_sums_to_residual() {
    let dim = 12;
    for t in 0..25 {
        let x = fnv_embed(&format!("decomp-x-{t}"), dim);
        let q = fnv_embed(&format!("decomp-q-{t}"), dim);
        let (parallel, orthogonal) = decompose_residual(&x, &q);

        // e_∥ + e_⊥ == e (exactly, up to f32 rounding).
        for i in 0..dim {
            let e = x[i] - q[i];
            assert!(
                (parallel[i] + orthogonal[i] - e).abs() < 1e-5,
                "components must sum to the residual"
            );
        }
        // e_∥ · e_⊥ ≈ 0.
        let dot = true_dot(&parallel, &orthogonal);
        let scale = (norm(&parallel) * norm(&orthogonal)).max(1e-9);
        assert!(
            dot.abs() / scale < 1e-4,
            "parallel and orthogonal components must be orthogonal (dot={dot})"
        );
    }
}

#[test]
fn decomposition_parallel_is_along_x_direction() {
    let dim = 8;
    let x = fnv_embed("along-x", dim);
    let q = fnv_embed("along-q", dim);
    let (parallel, _orthogonal) = decompose_residual(&x, &q);
    // The parallel component must be a scalar multiple of x, i.e. cross terms
    // parallel[i]*x[j] - parallel[j]*x[i] vanish.
    for i in 0..dim {
        for j in 0..dim {
            let cross =
                f64::from(parallel[i]) * f64::from(x[j]) - f64::from(parallel[j]) * f64::from(x[i]);
            assert!(
                cross.abs() < 1e-4,
                "parallel component must be collinear with x"
            );
        }
    }
}

#[test]
fn decomposition_zero_vector_has_no_parallel_component() {
    let dim = 6;
    let x = vec![0.0_f32; dim];
    let q = fnv_embed("zero-q", dim);
    let (parallel, orthogonal) = decompose_residual(&x, &q);
    for i in 0..dim {
        assert_eq!(parallel[i], 0.0, "zero vector has no parallel direction");
        // whole residual (= -q) is reported as orthogonal
        assert!((orthogonal[i] - (x[i] - q[i])).abs() < 1e-6);
    }
}

#[test]
fn anisotropic_loss_matches_weighted_decomposition() {
    let dim = 10;
    for t in 0..20 {
        let x = fnv_embed(&format!("loss-x-{t}"), dim);
        let q = fnv_embed(&format!("loss-q-{t}"), dim);
        for &mult in &[1.0_f64, 2.5, 4.0, 9.0] {
            let (parallel, orthogonal) = decompose_residual(&x, &q);
            let expected = mult * sum_sq(&parallel) + 1.0 * sum_sq(&orthogonal);
            let actual = anisotropic_loss(&x, &q, mult);
            let rel = (actual - expected).abs() / expected.abs().max(1e-9);
            assert!(rel < 1e-4, "loss identity mismatch: {actual} vs {expected}");
        }
    }
}

#[test]
fn anisotropic_loss_at_multiplier_one_equals_squared_l2() {
    let dim = 9;
    let x = fnv_embed("mse-x", dim);
    let q = fnv_embed("mse-q", dim);
    let loss = anisotropic_loss(&x, &q, 1.0);
    let l2 = true_l2_sq(&x, &q);
    assert!((loss - l2).abs() / l2.abs().max(1e-9) < 1e-5);
}

// ── Weighted-least-squares centroid update ────────────────────────────────────

#[test]
fn single_cluster_update_at_multiplier_one_is_the_mean() {
    // With K = 1 and multiplier == 1.0 the weighted-least-squares solution
    // (Σ W_x) q = Σ W_x x reduces to the plain arithmetic mean.
    let dim = 5;
    let vectors: Vec<Vec<f32>> = (0..7)
        .map(|i| fnv_embed(&format!("mean-{i}"), dim))
        .collect();
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(1)
        .with_parallel_weight_multiplier(1.0)
        .with_max_iterations(20);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();

    let codeword = &quantizer.codebooks()[0][0];
    let mut mean = vec![0.0f64; dim];
    for v in &vectors {
        for (m, &x) in mean.iter_mut().zip(v.iter()) {
            *m += f64::from(x);
        }
    }
    for m in &mut mean {
        *m /= vectors.len() as f64;
    }
    for i in 0..dim {
        assert!(
            (f64::from(codeword[i]) - mean[i]).abs() < 1e-4,
            "isotropic K=1 codeword must be the mean"
        );
    }
}

#[test]
fn single_cluster_solves_the_normal_equations() {
    // The learned codeword must satisfy (Σ W_x) q = Σ W_x x to f32 precision.
    let dim = 6;
    let mult = 5.0_f64;
    let h_orth = 1.0_f64;
    let vectors: Vec<Vec<f32>> = (0..9)
        .map(|i| fnv_embed(&format!("normal-eq-{i}"), dim))
        .collect();
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(1)
        .with_parallel_weight_multiplier(mult)
        .with_max_iterations(30);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();
    let q: Vec<f64> = quantizer.codebooks()[0][0]
        .iter()
        .map(|&x| f64::from(x))
        .collect();

    // Accumulate A = Σ W_x and b = Σ W_x x in f64.
    let mut a = vec![vec![0.0f64; dim]; dim];
    let mut b = vec![0.0f64; dim];
    for v in &vectors {
        let n = norm(v);
        let unit: Vec<f64> = v.iter().map(|&x| f64::from(x) / n).collect();
        for r in 0..dim {
            a[r][r] += h_orth;
            b[r] += h_orth * f64::from(v[r]);
            for c in 0..dim {
                a[r][c] += (mult - h_orth) * unit[r] * unit[c];
            }
            b[r] += (mult - h_orth) * n * unit[r];
        }
    }
    // residual = A q - b must be (near) zero.
    let mut resid_sq = 0.0f64;
    let mut b_sq = 0.0f64;
    for r in 0..dim {
        let mut aq = 0.0f64;
        for c in 0..dim {
            aq += a[r][c] * q[c];
        }
        let e = aq - b[r];
        resid_sq += e * e;
        b_sq += b[r] * b[r];
    }
    let rel = resid_sq.sqrt() / b_sq.sqrt().max(1e-9);
    assert!(rel < 1e-3, "normal-equation residual too large: {rel}");
}

#[test]
fn higher_multiplier_reduces_parallel_residual() {
    // Pareto monotonicity: if q1 minimizes w1·P + O and q2 minimizes w2·P + O
    // with w1 > w2, then P(q1) <= P(q2), where P is the summed parallel residual
    // energy and O the orthogonal energy over a shared single cluster (K = 1).
    //
    // For a *clear* margin the data must be (a) unit-norm — otherwise P is
    // dominated by irreducible norm mismatch that no single centroid can fix,
    // swamping the anisotropic effect in floating-point noise — and (b)
    // anisotropically distributed in *direction*: here the directions form a
    // cone around a shared base axis, so the parallel-optimal centroid genuinely
    // differs from the isotropic mean.
    let dim = 6;
    let base = {
        let raw = fnv_embed("pareto-base-axis", dim);
        let n = norm(&raw) as f32;
        raw.iter().map(|&x| x / n).collect::<Vec<f32>>()
    };
    let vectors: Vec<Vec<f32>> = (0..16)
        .map(|i| {
            let noise = fnv_embed(&format!("pareto-noise-{i}"), dim);
            // A direction inside a cone around `base`, then unit-normalized.
            let dir: Vec<f32> = base
                .iter()
                .zip(noise.iter())
                .map(|(&b, &z)| b + 0.6 * z)
                .collect();
            let n = norm(&dir) as f32;
            dir.iter().map(|&x| x / n).collect()
        })
        .collect();

    let parallel_energy = |mult: f64| -> f64 {
        let config = AnisotropicVqConfig::new()
            .with_dim(dim)
            .with_num_codewords(1)
            .with_parallel_weight_multiplier(mult)
            .with_max_iterations(40);
        let mut quantizer = AnisotropicQuantizer::new(config);
        quantizer.train(&vectors).unwrap();
        let codeword = quantizer.codebooks()[0][0].clone();
        vectors
            .iter()
            .map(|v| {
                let (parallel, _) = decompose_residual(v, &codeword);
                sum_sq(&parallel)
            })
            .sum::<f64>()
    };

    let p_iso = parallel_energy(1.0);
    let p_mid = parallel_energy(4.0);
    let p_aniso = parallel_energy(16.0);
    // Monotone non-increasing parallel energy as the multiplier grows (relative
    // slack absorbs f32 codeword-rounding noise).
    assert!(
        p_mid <= p_iso * (1.0 + 1e-4),
        "P(4) {p_mid} must be <= P(1) {p_iso}"
    );
    assert!(
        p_aniso <= p_mid * (1.0 + 1e-4),
        "P(16) {p_aniso} must be <= P(4) {p_mid}"
    );
    // And a clear (>= 0.5%) reduction from isotropic to strongly anisotropic.
    assert!(
        p_aniso < p_iso * 0.995,
        "anisotropic weighting must reduce parallel error: P(16)={p_aniso} vs P(1)={p_iso}"
    );
}

// ── Weighted-Lloyd convergence ────────────────────────────────────────────────

#[test]
fn loss_trajectory_is_non_increasing() {
    let dim = 8;
    let vectors = make_corpus(60, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(8)
        .with_parallel_weight_multiplier(4.0)
        .with_max_iterations(30)
        .with_convergence_tolerance(0.0); // run the full budget
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();

    let trajectory = quantizer.loss_trajectory();
    assert!(!trajectory.is_empty(), "trajectory must be recorded");
    for pair in trajectory.windows(2) {
        let (prev, next) = (pair[0], pair[1]);
        let slack = prev.abs().max(1.0) * 1e-3;
        assert!(
            next <= prev + slack,
            "weighted-Lloyd loss must be non-increasing: {prev} -> {next}"
        );
    }
    // Training must make genuine progress on non-trivial data.
    assert!(
        *trajectory.last().unwrap() < trajectory[0],
        "final loss must be below the first"
    );
    assert!(*trajectory.last().unwrap() >= 0.0, "loss is non-negative");
}

#[test]
fn product_style_trajectory_is_non_increasing() {
    let dim = 12;
    let vectors = make_corpus(50, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_subspaces(3)
        .with_num_codewords(6)
        .with_parallel_weight_multiplier(6.0)
        .with_max_iterations(25)
        .with_convergence_tolerance(0.0);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();
    let trajectory = quantizer.loss_trajectory();
    assert!(!trajectory.is_empty());
    for pair in trajectory.windows(2) {
        let slack = pair[0].abs().max(1.0) * 1e-3;
        assert!(
            pair[1] <= pair[0] + slack,
            "combined trajectory must not increase"
        );
    }
}

// ── Encode / decode ───────────────────────────────────────────────────────────

#[test]
fn encode_decode_shapes_and_indices() {
    let dim = 12;
    let vectors = make_corpus(40, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_subspaces(3)
        .with_num_codewords(5);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();

    let code = quantizer.encode(&vectors[0]).unwrap();
    assert_eq!(code.len(), 3, "one index per subspace");
    assert!(!code.is_empty());
    for &idx in &code.indices {
        assert!((idx as usize) < 5, "index within codebook size");
    }
    let decoded = quantizer.decode(&code);
    assert_eq!(decoded.len(), dim);
}

#[test]
fn decode_untrained_is_zero() {
    let config = AnisotropicVqConfig::new().with_dim(4);
    let quantizer = AnisotropicQuantizer::new(config);
    let code = AnisotropicCode::new(vec![0]);
    assert_eq!(quantizer.decode(&code), vec![0.0; 4]);
}

#[test]
fn reconstruction_quality_is_high_with_full_codebook() {
    // With K == N the codebook can (near-)exactly represent every vector.
    let dim = 8;
    let vectors = make_corpus(40, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(vectors.len())
        .with_max_iterations(50);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();

    let mut total_err = 0.0f64;
    let mut total_norm = 0.0f64;
    for v in &vectors {
        let code = quantizer.encode(v).unwrap();
        let decoded = quantizer.decode(&code);
        total_err += true_l2_sq(v, &decoded).sqrt();
        total_norm += norm(v);
    }
    let relative = total_err / total_norm.max(1e-9);
    assert!(
        relative < 0.1,
        "relative reconstruction error too high: {relative}"
    );
}

// ── Asymmetric score identities ───────────────────────────────────────────────

#[test]
fn estimated_inner_product_equals_decode_inner_product() {
    let dim = 10;
    let vectors = make_corpus(45, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_subspaces(2)
        .with_num_codewords(8);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();

    let query = fnv_embed("ip-query", dim);
    for v in vectors.iter().take(10) {
        let code = quantizer.encode(v).unwrap();
        let estimated = quantizer.estimate_inner_product(&query, &code).unwrap();
        let decoded = quantizer.decode(&code);
        let exact = true_dot(&query, &decoded);
        assert!(
            (f64::from(estimated) - exact).abs() < 1e-3,
            "asymmetric IP must equal ⟨query, decode(code)⟩"
        );
    }
}

#[test]
fn estimated_l2_equals_decode_l2() {
    let dim = 10;
    let vectors = make_corpus(45, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_subspaces(2)
        .with_num_codewords(8);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();

    let query = fnv_embed("l2-query", dim);
    for v in vectors.iter().take(10) {
        let code = quantizer.encode(v).unwrap();
        let estimated = quantizer.estimate_l2_sq(&query, &code).unwrap();
        let decoded = quantizer.decode(&code);
        let exact = true_l2_sq(&query, &decoded);
        assert!(
            (f64::from(estimated) - exact).abs() / exact.abs().max(1e-6) < 1e-3,
            "asymmetric L2 must equal ‖query − decode(code)‖²"
        );
    }
}

// ── Index build / search ──────────────────────────────────────────────────────

fn labelled_items(vectors: &[Vec<f32>]) -> Vec<(DocumentId, Vec<f32>)> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (DocumentId(format!("doc-{i}")), v.clone()))
        .collect()
}

/// Brute-force exact top-k document ids for a query under a metric on the
/// *original* (un-quantized) vectors.
fn brute_force_topk(
    items: &[(DocumentId, Vec<f32>)],
    query: &[f32],
    k: usize,
    metric: AnisotropicVqMetric,
) -> Vec<DocumentId> {
    let mut scored: Vec<(f64, DocumentId)> = items
        .iter()
        .map(|(id, v)| {
            let distance = match metric {
                AnisotropicVqMetric::InnerProduct => -true_dot(query, v),
                AnisotropicVqMetric::L2 => true_l2_sq(query, v),
            };
            (distance, id.clone())
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.0.cmp(&b.1.0))
    });
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn recall_at_k(approx: &[AnisotropicVqHit], exact: &[DocumentId]) -> f64 {
    let hit_ids: std::collections::HashSet<&String> = approx.iter().map(|h| &h.id.0).collect();
    let found = exact.iter().filter(|id| hit_ids.contains(&id.0)).count();
    found as f64 / exact.len().max(1) as f64
}

#[test]
fn index_recall_inner_product_matches_brute_force() {
    let dim = 8;
    let vectors = make_corpus(40, dim);
    let items = labelled_items(&vectors);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(vectors.len())
        .with_parallel_weight_multiplier(4.0)
        .with_max_iterations(50)
        .with_metric(AnisotropicVqMetric::InnerProduct);
    let index = AnisotropicVqIndex::build_from(&items, config).unwrap();
    assert_eq!(index.len(), vectors.len());
    assert!(!index.is_empty());

    let k = 5;
    let mut total_recall = 0.0;
    let queries = 12usize;
    for t in 0..queries {
        let query = fnv_embed(&format!("ip-search-query-{t}"), dim);
        let hits = index.search(&query, k).unwrap();
        assert!(hits.len() <= k);
        // hits are sorted ascending by estimated_distance
        for pair in hits.windows(2) {
            assert!(pair[0].estimated_distance <= pair[1].estimated_distance);
        }
        let exact = brute_force_topk(&items, &query, k, AnisotropicVqMetric::InnerProduct);
        total_recall += recall_at_k(&hits, &exact);
    }
    let avg = total_recall / queries as f64;
    assert!(
        avg >= 0.7,
        "average inner-product recall@{k} too low: {avg}"
    );
}

#[test]
fn index_recall_l2_and_self_retrieval() {
    let dim = 8;
    let vectors = make_corpus(40, dim);
    let items = labelled_items(&vectors);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(vectors.len())
        .with_max_iterations(50)
        .with_metric(AnisotropicVqMetric::L2);
    let index = AnisotropicVqIndex::build_from(&items, config).unwrap();

    // Self-retrieval: a stored vector must retrieve itself as the top hit.
    for j in [0usize, 7, 19, 33] {
        let hits = index.search(&vectors[j], 1).unwrap();
        assert_eq!(hits[0].id.0, format!("doc-{j}"), "self-retrieval failed");
    }

    // Recall against exact L2 top-k for perturbed queries.
    let k = 5;
    let mut total_recall = 0.0;
    let queries = 10usize;
    for t in 0..queries {
        let query = fnv_embed(&format!("l2-search-query-{t}"), dim);
        let hits = index.search(&query, k).unwrap();
        let exact = brute_force_topk(&items, &query, k, AnisotropicVqMetric::L2);
        total_recall += recall_at_k(&hits, &exact);
    }
    let avg = total_recall / queries as f64;
    assert!(avg >= 0.7, "average L2 recall@{k} too low: {avg}");
}

#[test]
fn product_style_index_search_works() {
    let dim = 16;
    let vectors = make_corpus(48, dim);
    let items = labelled_items(&vectors);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_subspaces(4)
        .with_num_codewords(16)
        .with_parallel_weight_multiplier(4.0)
        .with_max_iterations(30)
        .with_metric(AnisotropicVqMetric::InnerProduct);
    let index = AnisotropicVqIndex::build_from(&items, config).unwrap();

    let k = 5;
    let mut total_recall = 0.0;
    let queries = 10usize;
    for t in 0..queries {
        let query = fnv_embed(&format!("prod-query-{t}"), dim);
        let hits = index.search(&query, k).unwrap();
        assert!(hits.len() <= k);
        let exact = brute_force_topk(&items, &query, k, AnisotropicVqMetric::InnerProduct);
        total_recall += recall_at_k(&hits, &exact);
    }
    let avg = total_recall / queries as f64;
    // Product quantization is lossier; require a modest but real recall.
    assert!(avg >= 0.4, "product-style recall@{k} too low: {avg}");
}

#[test]
fn index_add_appends_after_build() {
    let dim = 6;
    let vectors = make_corpus(20, dim);
    let items = labelled_items(&vectors);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(16)
        .with_max_iterations(20)
        .with_metric(AnisotropicVqMetric::L2);
    let mut index = AnisotropicVqIndex::build_from(&items, config).unwrap();
    let before = index.len();

    let extra = fnv_embed("appended-doc", dim);
    index.add(DocumentId("extra".into()), &extra).unwrap();
    assert_eq!(index.len(), before + 1);

    // The appended vector becomes searchable: querying with it, the appended
    // document is retrieved (a full-index scan is used to avoid relying on
    // codeword-collision tie-breaking with K < N).
    let hits = index.search(&extra, index.len()).unwrap();
    assert!(
        hits.iter().any(|h| h.id.0 == "extra"),
        "appended document must be searchable"
    );
    // Its estimated distance to itself must be among the smallest (top quartile).
    let rank = hits
        .iter()
        .position(|h| h.id.0 == "extra")
        .expect("extra present");
    assert!(
        rank < hits.len() / 4 + 1,
        "appended vector should rank near the top"
    );
}

#[test]
fn hit_similarity_is_negated_distance() {
    let hit = AnisotropicVqHit::new(DocumentId("h".into()), -3.5);
    assert_eq!(hit.similarity(), 3.5);
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn training_is_deterministic() {
    let dim = 8;
    let vectors = make_corpus(35, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_subspaces(2)
        .with_num_codewords(8)
        .with_parallel_weight_multiplier(4.0)
        .with_max_iterations(20);

    let mut a = AnisotropicQuantizer::new(config);
    a.train(&vectors).unwrap();
    let mut b = AnisotropicQuantizer::new(config);
    b.train(&vectors).unwrap();

    assert_eq!(a.codebooks(), b.codebooks(), "codebooks must be identical");
    assert_eq!(a.loss_trajectory(), b.loss_trajectory());
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn single_training_vector_reconstructs_exactly() {
    let dim = 5;
    let only = fnv_embed("solo", dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(8)
        .with_parallel_weight_multiplier(4.0);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(std::slice::from_ref(&only)).unwrap();

    let code = quantizer.encode(&only).unwrap();
    let decoded = quantizer.decode(&code);
    for i in 0..dim {
        assert!(
            (decoded[i] - only[i]).abs() < 1e-4,
            "single vector must reconstruct exactly"
        );
    }
    // Every recorded loss is (near) zero.
    for &loss in quantizer.loss_trajectory() {
        assert!(loss < 1e-6, "single-vector cluster has zero loss");
    }
}

#[test]
fn all_identical_vectors() {
    let dim = 6;
    let base = fnv_embed("identical", dim);
    let vectors = vec![base.clone(); 12];
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(4)
        .with_parallel_weight_multiplier(4.0)
        .with_max_iterations(20);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();

    let code = quantizer.encode(&base).unwrap();
    let decoded = quantizer.decode(&code);
    for i in 0..dim {
        assert!((decoded[i] - base[i]).abs() < 1e-4);
    }
    for pair in quantizer.loss_trajectory().windows(2) {
        assert!(pair[1] <= pair[0] + 1e-6);
    }
}

#[test]
fn zero_vectors_are_handled_isotropically() {
    // A corpus mixing zero vectors and normal vectors must train and encode
    // without panicking (zero vectors have no direction -> isotropic weight).
    let dim = 5;
    let mut vectors = make_corpus(10, dim);
    vectors.push(vec![0.0; dim]);
    vectors.push(vec![0.0; dim]);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(6)
        .with_parallel_weight_multiplier(4.0)
        .with_max_iterations(20);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();

    let zero = vec![0.0f32; dim];
    let code = quantizer.encode(&zero).unwrap();
    let decoded = quantizer.decode(&code);
    assert_eq!(decoded.len(), dim);
    // The zero-vector loss reduces to plain squared L2 to its codeword.
    let loss = anisotropic_loss(&zero, &decoded, 4.0);
    let l2 = true_l2_sq(&zero, &decoded);
    assert!((loss - l2).abs() < 1e-6);
}

// ── Error paths ───────────────────────────────────────────────────────────────

#[test]
fn train_empty_set_errors() {
    let config = AnisotropicVqConfig::new().with_dim(4);
    let mut quantizer = AnisotropicQuantizer::new(config);
    assert_eq!(
        quantizer.train(&[]),
        Err(AnisotropicVqError::EmptyTrainingSet)
    );
}

#[test]
fn train_dim_mismatch_errors() {
    let config = AnisotropicVqConfig::new().with_dim(4);
    let mut quantizer = AnisotropicQuantizer::new(config);
    let vectors = vec![vec![1.0, 2.0, 3.0]]; // dim 3, not 4
    assert_eq!(
        quantizer.train(&vectors),
        Err(AnisotropicVqError::DimMismatch {
            expected: 4,
            actual: 3
        })
    );
}

#[test]
fn train_invalid_config_errors() {
    let config = AnisotropicVqConfig::new()
        .with_dim(10)
        .with_num_subspaces(3);
    let mut quantizer = AnisotropicQuantizer::new(config);
    let vectors = make_corpus(5, 10);
    assert!(matches!(
        quantizer.train(&vectors),
        Err(AnisotropicVqError::InvalidConfig(_))
    ));
}

#[test]
fn encode_before_train_errors() {
    let config = AnisotropicVqConfig::new().with_dim(4);
    let quantizer = AnisotropicQuantizer::new(config);
    assert_eq!(
        quantizer.encode(&[1.0, 2.0, 3.0, 4.0]),
        Err(AnisotropicVqError::NotTrained)
    );
}

#[test]
fn encode_dim_mismatch_errors() {
    let dim = 4;
    let vectors = make_corpus(6, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(4);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();
    assert_eq!(
        quantizer.encode(&[1.0, 2.0]),
        Err(AnisotropicVqError::DimMismatch {
            expected: 4,
            actual: 2
        })
    );
}

#[test]
fn tables_before_train_error() {
    let config = AnisotropicVqConfig::new().with_dim(4);
    let quantizer = AnisotropicQuantizer::new(config);
    let query = vec![0.0; 4];
    assert_eq!(
        quantizer.inner_product_tables(&query),
        Err(AnisotropicVqError::NotTrained)
    );
    assert_eq!(
        quantizer.l2_tables(&query),
        Err(AnisotropicVqError::NotTrained)
    );
}

#[test]
fn index_build_empty_errors() {
    let config = AnisotropicVqConfig::new().with_dim(4);
    let mut index = AnisotropicVqIndex::new(config);
    assert_eq!(index.build(&[]), Err(AnisotropicVqError::EmptyTrainingSet));
}

#[test]
fn index_search_before_build_errors() {
    let config = AnisotropicVqConfig::new().with_dim(4);
    let index = AnisotropicVqIndex::new(config);
    assert_eq!(
        index.search(&[1.0, 2.0, 3.0, 4.0], 3),
        Err(AnisotropicVqError::NotTrained)
    );
}

#[test]
fn index_search_dim_mismatch_errors() {
    let dim = 4;
    let vectors = make_corpus(8, dim);
    let items = labelled_items(&vectors);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_codewords(4);
    let index = AnisotropicVqIndex::build_from(&items, config).unwrap();
    assert_eq!(
        index.search(&[1.0, 2.0], 3),
        Err(AnisotropicVqError::DimMismatch {
            expected: 4,
            actual: 2
        })
    );
}

#[test]
fn estimate_wrong_code_length_errors() {
    let dim = 8;
    let vectors = make_corpus(20, dim);
    let config = AnisotropicVqConfig::new()
        .with_dim(dim)
        .with_num_subspaces(2)
        .with_num_codewords(4);
    let mut quantizer = AnisotropicQuantizer::new(config);
    quantizer.train(&vectors).unwrap();
    let query = fnv_embed("q", dim);
    let bad_code = AnisotropicCode::new(vec![0]); // needs 2 subspaces
    assert!(matches!(
        quantizer.estimate_inner_product(&query, &bad_code),
        Err(AnisotropicVqError::DimMismatch { .. })
    ));
    assert!(matches!(
        quantizer.estimate_l2_sq(&query, &bad_code),
        Err(AnisotropicVqError::DimMismatch { .. })
    ));
}
