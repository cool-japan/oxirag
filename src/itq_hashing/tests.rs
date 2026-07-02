#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::doc_markdown
)]
//! Tests for the `itq_hashing` module.
//!
//! Covers: `ItqConfig` validation and builders; `ItqError` display; `ItqCode`
//! bit access / signs / Hamming distance; the pure-Rust linear algebra
//! (Jacobi eigenvectors, square SVD including the rank-deficient completion
//! path, orthogonal Procrustes, deterministic random orthogonal matrices);
//! the ITQ hasher (PCA sanity, learned-rotation orthogonality, monotonic
//! convergence of the quantization objective, deterministic encoding, the
//! `sign(0) := +1` convention, `k == D` and `k > D` edges, tiny training
//! sets); and `ItqIndex` build/search recall against a brute-force baseline
//! on synthetic clustered data plus every error path.

use std::cmp::Ordering;

use super::hasher::ItqHasher;
use super::index::ItqIndex;
use super::linalg::{
    dot, jacobi_symmetric, mat_mul_abt, orthogonal_procrustes, random_orthogonal, svd_square,
};
use super::types::{DEFAULT_ITQ_SEED, ItqCode, ItqConfig, ItqError, ItqHit};

// ── shared numeric helpers ────────────────────────────────────────────────────

/// Dense `a · b` product for two row-major matrices.
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let inner = b.len();
    let m = if inner == 0 { 0 } else { b[0].len() };
    let mut out = vec![vec![0.0_f64; m]; n];
    for i in 0..n {
        for l in 0..inner {
            let ail = a[i][l];
            for j in 0..m {
                out[i][j] += ail * b[l][j];
            }
        }
    }
    out
}

/// Transpose a row-major matrix.
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut out = vec![vec![0.0_f64; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j][i] = a[i][j];
        }
    }
    out
}

/// Assert that `m` is (approximately) the identity within `tol`.
fn assert_identity(m: &[Vec<f64>], tol: f64) {
    let n = m.len();
    for i in 0..n {
        assert_eq!(m[i].len(), n, "matrix must be square to be the identity");
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (m[i][j] - expected).abs() < tol,
                "identity mismatch at ({i},{j}): {} vs {expected}",
                m[i][j]
            );
        }
    }
}

/// Assert an `n x n` matrix is orthogonal: `M Mᵀ ≈ I`.
fn assert_orthogonal(m: &[Vec<f64>], tol: f64) {
    let product = matmul(m, &transpose(m));
    assert_identity(&product, tol);
}

/// Matrix–vector product `m · v` for a row-major matrix.
fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

/// Exact squared L2 distance between two equal-length slices.
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Deterministic FNV-1a pseudo-embedding of a label, length `dim`, in
/// `[-1, 1]`.
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

/// A deterministic set of well-separated, block-structured clusters: cluster
/// `c` places `magnitude` in a disjoint block of `block` dimensions, plus a
/// tiny per-coordinate jitter, so brute-force nearest neighbor is unambiguous.
fn make_clusters(
    n_clusters: usize,
    per_cluster: usize,
    block: usize,
    magnitude: f32,
) -> Vec<(String, Vec<f32>)> {
    let dim = n_clusters * block;
    let mut items = Vec::new();
    for c in 0..n_clusters {
        for i in 0..per_cluster {
            let id = format!("cluster{c}-item{i}");
            let mut v = vec![0.0_f32; dim];
            for d in 0..dim {
                let jitter = ((i * 7 + d * 3) % 5) as f32 * 0.001;
                if (c * block..c * block + block).contains(&d) {
                    v[d] = magnitude + jitter;
                } else {
                    v[d] = jitter;
                }
            }
            items.push((id, v));
        }
    }
    items
}

/// Brute-force top-`k` item ids by ascending exact L2 distance.
fn brute_force_topk(items: &[(String, Vec<f32>)], query: &[f32], k: usize) -> Vec<String> {
    let mut scored: Vec<(f32, String)> = items
        .iter()
        .map(|(id, v)| (l2_sq(query, v), id.clone()))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(_, id)| id).collect()
}

// ── ItqConfig ─────────────────────────────────────────────────────────────────

#[test]
fn test_config_new_valid() {
    let cfg = ItqConfig::new(16).unwrap();
    assert_eq!(cfg.num_bits, 16);
    assert_eq!(cfg.max_iterations, 50);
    assert_eq!(cfg.tolerance, 1e-7);
    assert_eq!(cfg.seed, DEFAULT_ITQ_SEED);
    assert_eq!(cfg.jacobi_max_sweeps, 100);
    assert_eq!(cfg.jacobi_tol, 1e-10);
}

#[test]
fn test_config_new_zero_bits_is_error() {
    let err = ItqConfig::new(0).unwrap_err();
    assert!(matches!(err, ItqError::InvalidConfig(_)));
}

#[test]
fn test_config_builders() {
    let cfg = ItqConfig::new(8)
        .unwrap()
        .with_max_iterations(10)
        .with_tolerance(1e-4)
        .with_seed(42)
        .with_jacobi_max_sweeps(25)
        .with_jacobi_tol(1e-8);
    assert_eq!(cfg.max_iterations, 10);
    assert_eq!(cfg.tolerance, 1e-4);
    assert_eq!(cfg.seed, 42);
    assert_eq!(cfg.jacobi_max_sweeps, 25);
    assert_eq!(cfg.jacobi_tol, 1e-8);
}

#[test]
fn test_config_is_copy() {
    let cfg = ItqConfig::new(4).unwrap();
    let cfg2 = cfg; // Copy, not move.
    assert_eq!(cfg.num_bits, cfg2.num_bits);
}

#[test]
fn test_config_validate_ok() {
    assert!(ItqConfig::new(16).unwrap().validate().is_ok());
}

#[test]
fn test_config_validate_zero_max_iterations() {
    let cfg = ItqConfig::new(4).unwrap().with_max_iterations(0);
    assert!(matches!(cfg.validate(), Err(ItqError::InvalidConfig(_))));
}

#[test]
fn test_config_validate_zero_jacobi_sweeps() {
    let cfg = ItqConfig::new(4).unwrap().with_jacobi_max_sweeps(0);
    assert!(matches!(cfg.validate(), Err(ItqError::InvalidConfig(_))));
}

#[test]
fn test_config_validate_negative_tolerance() {
    let cfg = ItqConfig::new(4).unwrap().with_tolerance(-1.0);
    assert!(matches!(cfg.validate(), Err(ItqError::InvalidConfig(_))));
}

#[test]
fn test_config_validate_nan_jacobi_tol() {
    let cfg = ItqConfig::new(4).unwrap().with_jacobi_tol(f64::NAN);
    assert!(matches!(cfg.validate(), Err(ItqError::InvalidConfig(_))));
}

// ── ItqError display ──────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    assert_eq!(
        ItqError::InvalidConfig("bad".into()).to_string(),
        "invalid configuration: bad"
    );
    assert_eq!(
        ItqError::DimensionMismatch {
            expected: 4,
            got: 3
        }
        .to_string(),
        "dimension mismatch: expected 4, got 3"
    );
    assert!(
        ItqError::NumBitsExceedsDimension {
            num_bits: 8,
            dim: 4
        }
        .to_string()
        .contains("exceeds input dimensionality")
    );
    assert!(ItqError::EmptyDataset.to_string().contains("empty"));
    assert_eq!(ItqError::EmptyQuery.to_string(), "query vector is empty");
    assert!(ItqError::NonFinite.to_string().contains("non-finite"));
    assert!(
        ItqError::Numerical("boom".into())
            .to_string()
            .contains("boom")
    );
}

#[test]
fn test_error_clone_eq() {
    let e = ItqError::EmptyDataset;
    assert_eq!(e.clone(), e);
}

// ── ItqCode ───────────────────────────────────────────────────────────────────

#[test]
fn test_code_bit_and_signs() {
    // bits 0 and 2 set out of 4 => +1, -1, +1, -1.
    let code = ItqCode::new(vec![0b0101], 4);
    assert!(code.bit(0));
    assert!(!code.bit(1));
    assert!(code.bit(2));
    assert!(!code.bit(3));
    assert_eq!(code.as_signs(), vec![1, -1, 1, -1]);
}

#[test]
fn test_code_bit_out_of_range_is_false() {
    let code = ItqCode::new(vec![0b1111], 4);
    assert!(!code.bit(4));
    assert!(!code.bit(1000));
}

#[test]
fn test_code_hamming_distance() {
    let a = ItqCode::new(vec![0b0000], 4);
    let b = ItqCode::new(vec![0b1011], 4);
    assert_eq!(a.hamming_distance(&b), 3);
    assert_eq!(a.hamming_distance(&a), 0);
    assert_eq!(a.hamming_distance(&b), b.hamming_distance(&a));
}

#[test]
fn test_code_multiword_hamming() {
    let a = ItqCode::new(vec![0u64, 0u64], 100);
    let b = ItqCode::new(vec![u64::MAX, 0b111u64], 100);
    assert_eq!(a.hamming_distance(&b), 64 + 3);
}

#[test]
fn test_hit_new() {
    let h = ItqHit::new("x".to_string(), 7);
    assert_eq!(h.id, "x");
    assert_eq!(h.hamming_distance, 7);
}

// ── linalg: Jacobi eigensolver ────────────────────────────────────────────────

#[test]
fn test_jacobi_identity() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let (vals, vecs) = jacobi_symmetric(&a, 100, 1e-12).unwrap();
    assert_eq!(vals.len(), 2);
    assert!((vals[0] - 1.0).abs() < 1e-12);
    assert!((vals[1] - 1.0).abs() < 1e-12);
    assert_orthogonal(&vecs, 1e-12);
}

#[test]
fn test_jacobi_1x1() {
    let a = vec![vec![3.5]];
    let (vals, vecs) = jacobi_symmetric(&a, 100, 1e-12).unwrap();
    assert_eq!(vals, vec![3.5]);
    assert_eq!(vecs, vec![vec![1.0]]);
}

#[test]
fn test_jacobi_known_2x2_descending() {
    // [[2,1],[1,2]] has eigenvalues 3 and 1.
    let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
    let (vals, _vecs) = jacobi_symmetric(&a, 100, 1e-12).unwrap();
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_jacobi_eigenvectors_satisfy_av_equals_lambda_v() {
    // A random-ish symmetric 3x3.
    let a = vec![
        vec![4.0, 1.0, 2.0],
        vec![1.0, 3.0, 0.5],
        vec![2.0, 0.5, 5.0],
    ];
    let (vals, vecs) = jacobi_symmetric(&a, 200, 1e-14).unwrap();
    let n = 3;
    // Eigenvectors are orthonormal.
    assert_orthogonal(&vecs, 1e-9);
    // Descending eigenvalues.
    for w in vals.windows(2) {
        assert!(w[0] >= w[1] - 1e-12);
    }
    // A v_j ≈ lambda_j v_j.
    for j in 0..n {
        let v_col: Vec<f64> = (0..n).map(|i| vecs[i][j]).collect();
        let av = mat_vec(&a, &v_col);
        for i in 0..n {
            assert!(
                (av[i] - vals[j] * v_col[i]).abs() < 1e-8,
                "eigenpair {j} mismatch at {i}"
            );
        }
    }
}

#[test]
fn test_jacobi_non_square_errors() {
    let a = vec![vec![1.0, 0.0], vec![0.0]];
    assert!(matches!(
        jacobi_symmetric(&a, 10, 1e-10),
        Err(ItqError::Numerical(_))
    ));
}

#[test]
fn test_jacobi_non_finite_errors() {
    let a = vec![vec![f64::NAN, 0.0], vec![0.0, 1.0]];
    assert!(matches!(
        jacobi_symmetric(&a, 10, 1e-10),
        Err(ItqError::NonFinite)
    ));
}

// ── linalg: square SVD ────────────────────────────────────────────────────────

/// Reconstruct `U diag(sigma) Vᵀ` from an SVD triple.
fn svd_reconstruct(u: &[Vec<f64>], sigma: &[f64], v: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = u.len();
    let mut out = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += u[i][l] * sigma[l] * v[j][l];
            }
            out[i][j] = acc;
        }
    }
    out
}

#[test]
fn test_svd_diagonal() {
    let m = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
    let (u, sigma, v) = svd_square(&m, 200, 1e-14).unwrap();
    // Singular values descending: [3, 2].
    assert!((sigma[0] - 3.0).abs() < 1e-9);
    assert!((sigma[1] - 2.0).abs() < 1e-9);
    assert_orthogonal(&u, 1e-9);
    assert_orthogonal(&v, 1e-9);
    let recon = svd_reconstruct(&u, &sigma, &v);
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon[i][j] - m[i][j]).abs() < 1e-8);
        }
    }
}

#[test]
fn test_svd_general_reconstruct() {
    let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let (u, sigma, v) = svd_square(&m, 300, 1e-14).unwrap();
    assert_orthogonal(&u, 1e-8);
    assert_orthogonal(&v, 1e-8);
    for w in sigma.windows(2) {
        assert!(w[0] >= w[1] - 1e-12);
        assert!(w[1] >= -1e-12);
    }
    let recon = svd_reconstruct(&u, &sigma, &v);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (recon[i][j] - m[i][j]).abs() < 1e-7,
                "reconstruction mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_svd_rank_deficient_completion() {
    // Rank-1 matrix: M^T M has a zero eigenvalue => the U completion path runs.
    let m = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    let (u, sigma, v) = svd_square(&m, 300, 1e-14).unwrap();
    // One singular value ~5, the other ~0.
    assert!((sigma[0] - 5.0).abs() < 1e-7);
    assert!(sigma[1].abs() < 1e-6);
    // U remains genuinely orthogonal even though a column was completed.
    assert_orthogonal(&u, 1e-8);
    assert_orthogonal(&v, 1e-8);
    let recon = svd_reconstruct(&u, &sigma, &v);
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon[i][j] - m[i][j]).abs() < 1e-7);
        }
    }
}

#[test]
fn test_svd_zero_matrix() {
    let m = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let (u, sigma, v) = svd_square(&m, 100, 1e-14).unwrap();
    assert!(sigma.iter().all(|&s| s.abs() < 1e-9));
    assert_orthogonal(&u, 1e-9);
    assert_orthogonal(&v, 1e-9);
}

#[test]
fn test_svd_low_rank_high_dim_completion_orthogonal() {
    // A badly-scaled, rank-3 10x10 matrix: exactly the structure produced by
    // ITQ's Procrustes step on low-effective-rank data. Seven singular values
    // are ~0, so the U-completion path must fill seven orthonormal columns
    // (which requires reorthogonalized Gram–Schmidt to stay orthogonal).
    let k = 10;
    // Three orthogonal-ish "signal" directions with huge magnitudes.
    let dirs = [
        (
            3000.0_f64,
            [1.0, 0.5, -0.3, 0.2, 0.0, 0.1, -0.4, 0.0, 0.6, -0.2],
        ),
        (
            2000.0_f64,
            [-0.2, 1.0, 0.4, -0.5, 0.3, 0.0, 0.1, 0.2, -0.1, 0.4],
        ),
        (
            1500.0_f64,
            [0.1, -0.3, 1.0, 0.6, -0.2, 0.5, 0.0, -0.4, 0.2, 0.1],
        ),
    ];
    let mut m = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for (mag, vec) in &dirs {
                acc += mag * vec[i] * vec[j];
            }
            m[i][j] = acc;
        }
    }
    let (u, sigma, v) = svd_square(&m, 100, 1e-12).unwrap();
    assert_orthogonal(&u, 1e-8);
    assert_orthogonal(&v, 1e-8);
    // Reconstruction still holds.
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += u[i][l] * sigma[l] * v[j][l];
            }
            assert!((acc - m[i][j]).abs() < 1e-6, "recon mismatch at ({i},{j})");
        }
    }
    // The Procrustes rotation of such an M is genuinely orthogonal.
    let r = orthogonal_procrustes(&m, 100, 1e-12).unwrap();
    assert_orthogonal(&r, 1e-8);
}

// ── linalg: orthogonal Procrustes ─────────────────────────────────────────────

#[test]
fn test_procrustes_recovers_rotation() {
    // If M is itself an (orthogonal) rotation, the Procrustes solution is M.
    let theta = 0.5_f64;
    let m = vec![
        vec![theta.cos(), -theta.sin()],
        vec![theta.sin(), theta.cos()],
    ];
    let r = orthogonal_procrustes(&m, 200, 1e-14).unwrap();
    assert_orthogonal(&r, 1e-9);
    for i in 0..2 {
        for j in 0..2 {
            assert!((r[i][j] - m[i][j]).abs() < 1e-7);
        }
    }
}

#[test]
fn test_procrustes_is_orthogonal_for_general_matrix() {
    let m = vec![vec![1.5, -0.3], vec![0.7, 2.1]];
    let r = orthogonal_procrustes(&m, 300, 1e-14).unwrap();
    assert_orthogonal(&r, 1e-9);
}

#[test]
fn test_mat_mul_abt_and_dot() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    // (A Bᵀ)[0][0] = 1*5 + 2*6 = 17.
    let prod = mat_mul_abt(&a, &b);
    assert_eq!(prod[0][0], 17.0);
    assert_eq!(prod[1][1], 3.0 * 7.0 + 4.0 * 8.0);
    assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
}

// ── linalg: deterministic random orthogonal ───────────────────────────────────

#[test]
fn test_random_orthogonal_is_orthogonal() {
    for k in 1..=8 {
        let r = random_orthogonal(k, 12345).unwrap();
        assert_eq!(r.len(), k);
        assert_orthogonal(&r, 1e-9);
    }
}

#[test]
fn test_random_orthogonal_determinism() {
    let a = random_orthogonal(6, 777).unwrap();
    let b = random_orthogonal(6, 777).unwrap();
    assert_eq!(a, b);
    let c = random_orthogonal(6, 778).unwrap();
    assert_ne!(a, c);
}

#[test]
fn test_random_orthogonal_zero_dim_errors() {
    assert!(matches!(
        random_orthogonal(0, 1),
        Err(ItqError::InvalidConfig(_))
    ));
}

// ── hasher: training, PCA, learned rotation ───────────────────────────────────

#[test]
fn test_train_basic_shapes() {
    let vectors: Vec<Vec<f32>> = (0..40).map(|i| fnv_embed(&format!("v{i}"), 12)).collect();
    let cfg = ItqConfig::new(8).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    assert_eq!(hasher.dim(), 12);
    assert_eq!(hasher.num_bits(), 8);
    assert_eq!(hasher.mean().len(), 12);
    assert_eq!(hasher.components().len(), 8);
    assert!(hasher.components().iter().all(|c| c.len() == 12));
    assert_eq!(hasher.rotation().len(), 8);
    assert!(!hasher.loss_history().is_empty());
}

#[test]
fn test_learned_rotation_is_orthogonal() {
    let vectors: Vec<Vec<f32>> = (0..60)
        .map(|i| fnv_embed(&format!("row-{i}"), 16))
        .collect();
    let cfg = ItqConfig::new(10).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    // The whole point: R is a genuine learned orthogonal matrix, R Rᵀ ≈ I.
    assert_orthogonal(hasher.rotation(), 1e-8);
}

#[test]
fn test_learned_rotation_orthogonal_on_low_rank_data() {
    // Clustered data has low effective rank (4 clusters => ~3 between-cluster
    // dims after centering), so the Procrustes M is rank-deficient and heavily
    // exercises the SVD completion path. R must still be exactly orthogonal.
    let items = make_clusters(4, 12, 4, 30.0);
    let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
    let cfg = ItqConfig::new(10).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    assert_orthogonal(hasher.rotation(), 1e-8);
}

#[test]
fn test_pca_components_orthonormal() {
    let vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| fnv_embed(&format!("pca-{i}"), 10))
        .collect();
    let cfg = ItqConfig::new(6).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    // Vᵀ V ≈ I over the k retained components.
    let comps = hasher.components();
    for a in 0..comps.len() {
        for b in 0..comps.len() {
            let d = dot(&comps[a], &comps[b]);
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!(
                (d - expected).abs() < 1e-7,
                "component inner product ({a},{b}) = {d}"
            );
        }
    }
}

#[test]
fn test_pca_top_component_aligns_with_dominant_variance() {
    // Variance concentrated along axis 0; the first PC must align with it.
    let vectors: Vec<Vec<f32>> = (0..21)
        .map(|i| vec![(i as f32) - 10.0, 0.0005 * ((i % 3) as f32)])
        .collect();
    let cfg = ItqConfig::new(2).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    let top = &hasher.components()[0];
    assert!(
        top[0].abs() > 0.999,
        "top PC should align with axis 0, got {top:?}"
    );
    assert!(
        top[1].abs() < 0.05,
        "top PC axis-1 leakage too large: {top:?}"
    );
}

#[test]
fn test_objective_monotonically_non_increasing() {
    let items = make_clusters(4, 12, 4, 30.0);
    let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
    let cfg = ItqConfig::new(10).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    let history = hasher.loss_history();
    assert!(!history.is_empty());
    for w in history.windows(2) {
        // Each closed-form sub-step is a global minimizer, so the objective
        // cannot increase (a tiny slack absorbs floating-point noise).
        assert!(
            w[1] <= w[0] + 1e-6,
            "objective increased: {} -> {}",
            w[0],
            w[1]
        );
    }
    // The learned rotation should improve on (or match) the very first value.
    assert!(*history.last().unwrap() <= history[0] + 1e-6);
}

#[test]
fn test_encode_is_deterministic() {
    let vectors: Vec<Vec<f32>> = (0..30).map(|i| fnv_embed(&format!("d{i}"), 8)).collect();
    let cfg = ItqConfig::new(6).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    let q = fnv_embed("query", 8);
    let a = hasher.encode(&q).unwrap();
    let b = hasher.encode(&q).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.num_bits, 6);
    assert_eq!(a.bits.len(), 1);
}

#[test]
fn test_encode_zero_projection_sign_zero_convention() {
    // A dataset symmetric about the origin has an exact-zero mean (in f64 *and*
    // f32), so encoding the zero vector yields an exact-zero rotated projection.
    // The `sign(0) := +1` convention then sets every bit.
    let vectors = vec![
        vec![1.0_f32, 2.0, 3.0],
        vec![-1.0_f32, -2.0, -3.0],
        vec![2.0_f32, -1.0, 1.0],
        vec![-2.0_f32, 1.0, -1.0],
    ];
    let cfg = ItqConfig::new(3).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    // Mean must be exactly zero for the convention to be exercised cleanly.
    assert!(hasher.mean().iter().all(|&m| m == 0.0));
    let code = hasher.encode(&[0.0_f32, 0.0, 0.0]).unwrap();
    for i in 0..code.num_bits {
        assert!(code.bit(i), "bit {i} should be set for the zero projection");
    }
    assert_eq!(code.as_signs(), vec![1, 1, 1]);
}

#[test]
fn test_project_matches_rotation_dimension() {
    let vectors: Vec<Vec<f32>> = (0..20).map(|i| fnv_embed(&format!("p{i}"), 9)).collect();
    let cfg = ItqConfig::new(5).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    let t = hasher.project(&fnv_embed("z", 9)).unwrap();
    assert_eq!(t.len(), 5);
}

#[test]
fn test_encode_dimension_mismatch() {
    let vectors: Vec<Vec<f32>> = (0..10).map(|i| fnv_embed(&format!("e{i}"), 8)).collect();
    let hasher = ItqHasher::train(&vectors, &ItqConfig::new(4).unwrap()).unwrap();
    let err = hasher.encode(&[1.0, 2.0, 3.0]).unwrap_err();
    assert!(matches!(
        err,
        ItqError::DimensionMismatch {
            expected: 8,
            got: 3
        }
    ));
}

#[test]
fn test_encode_non_finite() {
    let vectors: Vec<Vec<f32>> = (0..10).map(|i| fnv_embed(&format!("f{i}"), 4)).collect();
    let hasher = ItqHasher::train(&vectors, &ItqConfig::new(3).unwrap()).unwrap();
    let err = hasher.encode(&[1.0, f32::INFINITY, 0.0, 0.0]).unwrap_err();
    assert!(matches!(err, ItqError::NonFinite));
}

// ── hasher: edge cases ────────────────────────────────────────────────────────

#[test]
fn test_train_empty_dataset() {
    let vectors: Vec<Vec<f32>> = Vec::new();
    assert!(matches!(
        ItqHasher::train(&vectors, &ItqConfig::new(4).unwrap()),
        Err(ItqError::EmptyDataset)
    ));
}

#[test]
fn test_train_inconsistent_dims() {
    let vectors = vec![vec![1.0_f32, 2.0], vec![1.0_f32, 2.0, 3.0]];
    let err = ItqHasher::train(&vectors, &ItqConfig::new(2).unwrap()).unwrap_err();
    assert!(matches!(err, ItqError::DimensionMismatch { .. }));
}

#[test]
fn test_train_non_finite() {
    let vectors = vec![vec![1.0_f32, f32::NAN], vec![0.0_f32, 1.0]];
    assert!(matches!(
        ItqHasher::train(&vectors, &ItqConfig::new(2).unwrap()),
        Err(ItqError::NonFinite)
    ));
}

#[test]
fn test_num_bits_equal_to_dim_ok() {
    // k == D: full-rank rotation, no dimensionality reduction.
    let vectors: Vec<Vec<f32>> = (0..30).map(|i| fnv_embed(&format!("k{i}"), 6)).collect();
    let cfg = ItqConfig::new(6).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    assert_eq!(hasher.num_bits(), 6);
    assert_orthogonal(hasher.rotation(), 1e-8);
    let code = hasher.encode(&fnv_embed("q", 6)).unwrap();
    assert_eq!(code.num_bits, 6);
}

#[test]
fn test_num_bits_greater_than_dim_errors() {
    let vectors: Vec<Vec<f32>> = (0..10).map(|i| fnv_embed(&format!("g{i}"), 4)).collect();
    let cfg = ItqConfig::new(5).unwrap();
    let err = ItqHasher::train(&vectors, &cfg).unwrap_err();
    assert!(matches!(
        err,
        ItqError::NumBitsExceedsDimension {
            num_bits: 5,
            dim: 4
        }
    ));
}

#[test]
fn test_tiny_training_set_single_vector() {
    // n = 1: covariance is all zeros; everything degenerate but must not panic.
    let vectors = vec![vec![1.0_f32, 2.0, 3.0, 4.0]];
    let cfg = ItqConfig::new(3).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    assert_orthogonal(hasher.rotation(), 1e-8);
    let code = hasher.encode(&vectors[0]).unwrap();
    assert_eq!(code.num_bits, 3);
    // Centered single sample => zero projection => sign(0) := +1 everywhere.
    for i in 0..3 {
        assert!(code.bit(i));
    }
}

#[test]
fn test_tiny_training_set_two_vectors() {
    let vectors = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0_f32, 1.0, 0.0]];
    let cfg = ItqConfig::new(3).unwrap();
    let hasher = ItqHasher::train(&vectors, &cfg).unwrap();
    assert_orthogonal(hasher.rotation(), 1e-8);
    // Both training vectors encode without error.
    for v in &vectors {
        assert_eq!(hasher.encode(v).unwrap().num_bits, 3);
    }
}

// ── ItqIndex ──────────────────────────────────────────────────────────────────

#[test]
fn test_index_build_and_accessors() {
    let items = make_clusters(3, 5, 3, 20.0);
    let index = ItqIndex::build(items, ItqConfig::new(6).unwrap()).unwrap();
    assert_eq!(index.len(), 15);
    assert!(!index.is_empty());
    assert_eq!(index.hasher().num_bits(), 6);
}

#[test]
fn test_index_build_empty() {
    let items: Vec<(String, Vec<f32>)> = Vec::new();
    assert!(matches!(
        ItqIndex::build(items, ItqConfig::new(4).unwrap()),
        Err(ItqError::EmptyDataset)
    ));
}

#[test]
fn test_index_search_empty_query() {
    let items = make_clusters(2, 4, 3, 15.0);
    let index = ItqIndex::build(items, ItqConfig::new(4).unwrap()).unwrap();
    assert!(matches!(index.search(&[], 3), Err(ItqError::EmptyQuery)));
}

#[test]
fn test_index_search_dimension_mismatch() {
    let items = make_clusters(2, 4, 3, 15.0);
    let index = ItqIndex::build(items, ItqConfig::new(4).unwrap()).unwrap();
    let err = index.search(&[1.0, 2.0], 3).unwrap_err();
    assert!(matches!(err, ItqError::DimensionMismatch { .. }));
}

#[test]
fn test_index_search_truncates_to_k() {
    let items = make_clusters(3, 5, 3, 20.0);
    let index = ItqIndex::build(items, ItqConfig::new(6).unwrap()).unwrap();
    let query = fnv_embed("anything", 9);
    let hits = index.search(&query, 4).unwrap();
    assert_eq!(hits.len(), 4);
    // Non-decreasing Hamming order.
    for w in hits.windows(2) {
        assert!(w[0].hamming_distance <= w[1].hamming_distance);
    }
}

#[test]
fn test_index_recall_vs_brute_force() {
    let n_clusters = 4;
    let per_cluster = 6;
    let block = 3;
    let magnitude = 25.0_f32;
    let items = make_clusters(n_clusters, per_cluster, block, magnitude);
    let dim = n_clusters * block;
    let index = ItqIndex::build(items.clone(), ItqConfig::new(8).unwrap()).unwrap();

    let mut total_recall = 0.0_f64;
    for c in 0..n_clusters {
        // Query at the cluster centroid (tiny offset).
        let mut query = vec![0.0_f32; dim];
        for d in c * block..c * block + block {
            query[d] = magnitude + 0.0005;
        }

        let brute = brute_force_topk(&items, &query, per_cluster);
        let hits = index.search(&query, per_cluster).unwrap();
        let hit_ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();

        // The nearest neighbor must be from the query's own cluster.
        assert!(
            hit_ids[0].starts_with(&format!("cluster{c}-")),
            "top hit {:?} not in cluster {c}",
            hit_ids[0]
        );

        let intersection = hit_ids.iter().filter(|id| brute.contains(id)).count();
        total_recall += intersection as f64 / per_cluster as f64;
    }
    let mean_recall = total_recall / n_clusters as f64;
    assert!(
        mean_recall >= 0.9,
        "mean recall@{per_cluster} too low: {mean_recall}"
    );
}

#[test]
fn test_index_same_cluster_codes_identical() {
    let items = make_clusters(3, 4, 4, 40.0);
    let index = ItqIndex::build(items, ItqConfig::new(8).unwrap()).unwrap();
    let dim = 12;
    // Two vectors in the same cluster (block 0) should hash identically.
    let mut a = vec![0.0_f32; dim];
    let mut b = vec![0.0_f32; dim];
    for d in 0..4 {
        a[d] = 40.0;
        b[d] = 40.02;
    }
    let ca = index.hasher().encode(&a).unwrap();
    let cb = index.hasher().encode(&b).unwrap();
    assert_eq!(ca.hamming_distance(&cb), 0);
}

#[test]
fn test_index_determinism() {
    let items = make_clusters(3, 4, 3, 20.0);
    let cfg = ItqConfig::new(6).unwrap();
    let index_a = ItqIndex::build(items.clone(), cfg).unwrap();
    let index_b = ItqIndex::build(items.clone(), cfg).unwrap();
    let query = fnv_embed("determinism", 9);
    let ha = index_a.search(&query, 5).unwrap();
    let hb = index_b.search(&query, 5).unwrap();
    assert_eq!(ha, hb);
}

#[test]
fn test_index_search_k_zero_returns_empty() {
    let items = make_clusters(2, 3, 3, 15.0);
    let index = ItqIndex::build(items, ItqConfig::new(4).unwrap()).unwrap();
    let hits = index.search(&fnv_embed("q", 6), 0).unwrap();
    assert!(hits.is_empty());
}
