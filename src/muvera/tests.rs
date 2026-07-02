#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::useless_vec,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::type_complexity
)]
//! Tests for the `muvera` module.
//!
//! Coverage: `MuveraConfig` validation and derived-dimension arithmetic; FDE
//! dimension equalling `r_reps * 2^k_sim * inner_dim` (with and without inner
//! projection); byte-exact determinism; the empty-document-cell fill rule and
//! its Hamming tie-break; FDE dot-product ordering matching exact Chamfer
//! ordering on unambiguous hand-built sets; variance reduction from more
//! repetitions; the approximate-vs-exact re-rank divergence; and every error
//! path.

use super::fde::{MuveraEncoder, nearest_occupied_bucket};
use super::index::MuveraIndex;
use super::types::{
    DEFAULT_MUVERA_SEED, MAX_K_SIM, MuveraConfig, MuveraDocument, MuveraError, MuveraSimilarity,
};

// ── Shared fixtures ──────────────────────────────────────────────────────────

/// Deterministic pseudo-random, L2-normalised vector derived from `tag`.
///
/// An FNV-1a hash of `tag` seeds an independent `splitmix64` stream — one draw
/// per coordinate — so distinct tags yield genuinely diverse (near-orthogonal)
/// directions rather than the near-parallel vectors a single hash-per-index
/// scheme produces.
fn pseudo_vec(tag: &str, dim: usize) -> Vec<f32> {
    let mut seed: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in tag.as_bytes() {
        seed ^= u64::from(b);
        seed = seed.wrapping_mul(0x0100_0000_01b3);
    }
    let mut v = vec![0.0f32; dim];
    for slot in &mut v {
        seed = seed
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .rotate_left(31)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = seed ^ (seed >> 31);
        *slot = (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
    }
    normalize(v)
}

/// A deterministic multi-vector set of `n_tokens` tokens.
fn multi_set(tag: &str, n_tokens: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n_tokens)
        .map(|t| pseudo_vec(&format!("{tag}-{t}"), dim))
        .collect()
}

/// L2-normalise a vector (no-op for a zero vector).
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

/// The `i`-th standard basis vector of dimension `dim`.
fn basis(i: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[i] = 1.0;
    v
}

/// Number of pairs `(i < j)` that `exact` and `approx` order in strictly
/// opposite directions — a Kendall-style ranking-error count.
fn discordant_pairs(exact: &[f32], approx: &[f32]) -> usize {
    let mut count = 0;
    for i in 0..exact.len() {
        for j in (i + 1)..exact.len() {
            let de = exact[i] - exact[j];
            let da = approx[i] - approx[j];
            if (de > 0.0 && da < 0.0) || (de < 0.0 && da > 0.0) {
                count += 1;
            }
        }
    }
    count
}

/// Return `true` when every element of `block` is exactly zero.
fn is_zero_block(block: &[f32]) -> bool {
    block.iter().all(|&x| x == 0.0)
}

// ── MuveraConfig: validation ─────────────────────────────────────────────────

#[test]
fn config_default_is_valid() {
    let config = MuveraConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.dim, 128);
    assert_eq!(config.k_sim, 4);
    assert_eq!(config.d_proj, 16);
    assert_eq!(config.r_reps, 8);
    assert!(config.fill_empty);
    assert!(!config.rerank);
    assert_eq!(config.seed, DEFAULT_MUVERA_SEED);
    assert_eq!(config.top_k, 10);
}

#[test]
fn config_new_sets_dim_keeps_defaults() {
    let config = MuveraConfig::new(32).unwrap();
    assert_eq!(config.dim, 32);
    assert_eq!(config.k_sim, 4);
    assert_eq!(config.r_reps, 8);
}

#[test]
fn config_new_rejects_zero_dim() {
    let err = MuveraConfig::new(0).unwrap_err();
    assert!(matches!(err, MuveraError::InvalidConfig(_)));
}

#[test]
fn config_validate_rejects_zero_k_sim() {
    let config = MuveraConfig::new(8).unwrap().with_k_sim(0);
    assert!(matches!(
        config.validate(),
        Err(MuveraError::InvalidConfig(_))
    ));
}

#[test]
fn config_validate_rejects_k_sim_over_max() {
    let config = MuveraConfig::new(8).unwrap().with_k_sim(MAX_K_SIM + 1);
    assert!(matches!(
        config.validate(),
        Err(MuveraError::InvalidConfig(_))
    ));
}

#[test]
fn config_validate_accepts_k_sim_at_max_boundary() {
    // At the boundary the config is structurally valid; only the (huge)
    // encoder allocation would be expensive, so we validate without building.
    let config = MuveraConfig::new(1)
        .unwrap()
        .with_k_sim(MAX_K_SIM)
        .with_d_proj(0)
        .with_r_reps(1);
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_rejects_zero_r_reps() {
    let config = MuveraConfig::new(8).unwrap().with_r_reps(0);
    assert!(matches!(
        config.validate(),
        Err(MuveraError::InvalidConfig(_))
    ));
}

#[test]
fn config_builders_set_all_fields() {
    let config = MuveraConfig::new(16)
        .unwrap()
        .with_k_sim(3)
        .with_d_proj(8)
        .with_r_reps(5)
        .with_fill_empty(false)
        .with_seed(42)
        .with_rerank(true)
        .with_rerank_depth(7)
        .with_top_k(3);
    assert_eq!(config.k_sim, 3);
    assert_eq!(config.d_proj, 8);
    assert_eq!(config.r_reps, 5);
    assert!(!config.fill_empty);
    assert_eq!(config.seed, 42);
    assert!(config.rerank);
    assert_eq!(config.rerank_depth, 7);
    assert_eq!(config.top_k, 3);
}

// ── MuveraConfig: derived dimensions ─────────────────────────────────────────

#[test]
fn config_num_buckets_is_two_to_k_sim() {
    assert_eq!(MuveraConfig::new(4).unwrap().with_k_sim(1).num_buckets(), 2);
    assert_eq!(MuveraConfig::new(4).unwrap().with_k_sim(3).num_buckets(), 8);
    assert_eq!(
        MuveraConfig::new(4).unwrap().with_k_sim(6).num_buckets(),
        64
    );
}

#[test]
fn config_inner_dim_with_projection_is_d_proj() {
    let config = MuveraConfig::new(32).unwrap().with_d_proj(8);
    assert_eq!(config.inner_dim(), 8);
}

#[test]
fn config_inner_dim_without_projection_is_dim() {
    let config = MuveraConfig::new(32).unwrap().with_d_proj(0);
    assert_eq!(config.inner_dim(), 32);
}

#[test]
fn config_expected_dim_with_projection() {
    // r_reps * 2^k_sim * d_proj
    let config = MuveraConfig::new(8)
        .unwrap()
        .with_k_sim(3)
        .with_d_proj(4)
        .with_r_reps(2);
    assert_eq!(config.expected_dim(), 2 * 8 * 4);
}

#[test]
fn config_expected_dim_without_projection() {
    // r_reps * 2^k_sim * dim
    let config = MuveraConfig::new(8)
        .unwrap()
        .with_k_sim(3)
        .with_d_proj(0)
        .with_r_reps(2);
    assert_eq!(config.expected_dim(), 2 * 8 * 8);
}

// ── FDE dimension ────────────────────────────────────────────────────────────

#[test]
fn fde_dim_equals_formula_with_projection() {
    let config = MuveraConfig::new(6)
        .unwrap()
        .with_k_sim(2)
        .with_d_proj(5)
        .with_r_reps(3);
    let encoder = MuveraEncoder::new(config).unwrap();
    let fde = encoder.encode_document(&multi_set("doc", 7, 6)).unwrap();
    assert_eq!(fde.dim(), 3 * 4 * 5);
    assert_eq!(fde.dim(), config.expected_dim());
}

#[test]
fn fde_dim_equals_formula_without_projection() {
    let config = MuveraConfig::new(6)
        .unwrap()
        .with_k_sim(2)
        .with_d_proj(0)
        .with_r_reps(3);
    let encoder = MuveraEncoder::new(config).unwrap();
    let fde = encoder.encode_document(&multi_set("doc", 7, 6)).unwrap();
    assert_eq!(fde.dim(), 3 * 4 * 6);
    assert_eq!(fde.dim(), config.expected_dim());
}

#[test]
fn fde_num_blocks_and_block_dim() {
    let config = MuveraConfig::new(6)
        .unwrap()
        .with_k_sim(2)
        .with_d_proj(5)
        .with_r_reps(3);
    let encoder = MuveraEncoder::new(config).unwrap();
    let fde = encoder.encode_document(&multi_set("doc", 4, 6)).unwrap();
    assert_eq!(fde.block_dim(), 5);
    assert_eq!(fde.num_blocks(), 3 * 4);
    // Each block is retrievable and of the right length.
    for b in 0..fde.num_blocks() {
        assert_eq!(fde.block(b).unwrap().len(), 5);
    }
    assert!(fde.block(fde.num_blocks()).is_none());
}

#[test]
fn encoder_expected_dim_matches_encoding() {
    let config = MuveraConfig::new(10).unwrap().with_k_sim(3).with_d_proj(4);
    let encoder = MuveraEncoder::new(config).unwrap();
    let q = encoder.encode_query(&multi_set("q", 3, 10)).unwrap();
    let d = encoder.encode_document(&multi_set("d", 9, 10)).unwrap();
    assert_eq!(encoder.expected_dim(), q.dim());
    assert_eq!(q.dim(), d.dim());
}

#[test]
fn fde_query_and_document_share_dimension() {
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap()).unwrap();
    let q = encoder.encode_query(&multi_set("q", 2, 8)).unwrap();
    let d = encoder.encode_document(&multi_set("d", 12, 8)).unwrap();
    assert_eq!(q.dim(), d.dim());
    assert!(!q.is_empty());
}

// ── Determinism ──────────────────────────────────────────────────────────────

#[test]
fn determinism_same_config_same_document() {
    let config = MuveraConfig::new(12).unwrap().with_k_sim(3).with_d_proj(6);
    let doc = multi_set("stable", 9, 12);
    let a = MuveraEncoder::new(config)
        .unwrap()
        .encode_document(&doc)
        .unwrap();
    let b = MuveraEncoder::new(config)
        .unwrap()
        .encode_document(&doc)
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.as_slice(), b.as_slice());
}

#[test]
fn determinism_same_config_same_query() {
    let config = MuveraConfig::new(12).unwrap().with_k_sim(3).with_d_proj(6);
    let q = multi_set("q", 4, 12);
    let a = MuveraEncoder::new(config)
        .unwrap()
        .encode_query(&q)
        .unwrap();
    let b = MuveraEncoder::new(config)
        .unwrap()
        .encode_query(&q)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn determinism_no_projection_is_reproducible() {
    let config = MuveraConfig::new(9).unwrap().with_k_sim(2).with_d_proj(0);
    let doc = multi_set("np", 6, 9);
    let a = MuveraEncoder::new(config)
        .unwrap()
        .encode_document(&doc)
        .unwrap();
    let b = MuveraEncoder::new(config)
        .unwrap()
        .encode_document(&doc)
        .unwrap();
    assert_eq!(a.as_slice(), b.as_slice());
}

#[test]
fn determinism_different_seed_changes_encoding() {
    let doc = multi_set("seedy", 8, 12);
    let a = MuveraEncoder::new(MuveraConfig::new(12).unwrap().with_seed(1))
        .unwrap()
        .encode_document(&doc)
        .unwrap();
    let b = MuveraEncoder::new(MuveraConfig::new(12).unwrap().with_seed(2))
        .unwrap()
        .encode_document(&doc)
        .unwrap();
    assert_ne!(a.as_slice(), b.as_slice());
}

#[test]
fn determinism_token_bucket_is_stable() {
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap().with_k_sim(4)).unwrap();
    let token = pseudo_vec("bucketed", 8);
    let first = encoder.token_bucket(&token, 0).unwrap();
    let again = encoder.token_bucket(&token, 0).unwrap();
    assert_eq!(first, again);
    assert!(first < 16); // k_sim = 4 → 16 buckets
}

#[test]
fn token_bucket_scalar_multiples_share_bucket() {
    // A positive scalar multiple never flips any hyperplane sign.
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap().with_k_sim(4)).unwrap();
    let token = pseudo_vec("scaled", 8);
    let scaled: Vec<f32> = token.iter().map(|x| x * 0.01).collect();
    for rep in 0..encoder.config().r_reps {
        assert_eq!(
            encoder.token_bucket(&token, rep).unwrap(),
            encoder.token_bucket(&scaled, rep).unwrap()
        );
    }
}

// ── Approximation quality: FDE ordering matches Chamfer ──────────────────────

#[test]
fn fde_ranks_matching_document_over_orthogonal_one() {
    // Query = {e0}. Doc "hit" contains e0 (identical to query); doc "miss" has
    // only vectors orthogonal to e0. Chamfer(hit) = 1 > Chamfer(miss) = 0, and
    // the FDE dot product must agree because "miss"'s cells are orthogonal to
    // e0 (whether or not empty-fill borrows them).
    let dim = 8;
    let config = MuveraConfig::new(dim)
        .unwrap()
        .with_k_sim(4)
        .with_d_proj(0)
        .with_r_reps(4);
    let mut index = MuveraIndex::new(config).unwrap();
    index
        .add("hit", vec![basis(0, dim), basis(3, dim)])
        .unwrap();
    index
        .add("miss", vec![basis(1, dim), basis(2, dim)])
        .unwrap();

    let query = vec![basis(0, dim)];
    let hits = index.search(&query, 2).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "hit");
    assert!(hits[0].score > hits[1].score);
    // "miss" is exactly orthogonal → exactly zero approximate score.
    assert_eq!(hits[1].score, 0.0);
}

#[test]
fn fde_picks_unambiguous_maxsim_winner_among_many() {
    // Four documents built from disjoint basis-vector pairs; a query equal to
    // one document's basis vector must rank that document first.
    let dim = 8;
    let config = MuveraConfig::new(dim)
        .unwrap()
        .with_k_sim(4)
        .with_d_proj(0)
        .with_r_reps(4);
    let mut index = MuveraIndex::new(config).unwrap();
    index.add("d0", vec![basis(0, dim), basis(4, dim)]).unwrap();
    index.add("d1", vec![basis(1, dim), basis(5, dim)]).unwrap();
    index.add("d2", vec![basis(2, dim), basis(6, dim)]).unwrap();
    index.add("d3", vec![basis(3, dim), basis(7, dim)]).unwrap();

    for (winner, axis) in [("d0", 0), ("d1", 1), ("d2", 2), ("d3", 3)] {
        let query = vec![basis(axis, dim)];
        let hits = index.search(&query, 4).unwrap();
        assert_eq!(hits[0].doc_id, winner, "axis {axis} should pick {winner}");
        assert!(hits[0].score > 0.0);
    }
}

#[test]
fn fde_ordering_agrees_with_chamfer_on_basis_documents() {
    // With orthogonal basis documents the FDE ordering is exact: verify the
    // full ranking matches the exact-Chamfer ranking.
    let dim = 6;
    let config = MuveraConfig::new(dim)
        .unwrap()
        .with_k_sim(3)
        .with_d_proj(0)
        .with_r_reps(3);
    let encoder = MuveraEncoder::new(config).unwrap();
    let docs = [
        ("a", vec![basis(0, dim)]),
        ("b", vec![basis(1, dim)]),
        ("c", vec![basis(2, dim)]),
    ];
    // Query aligned mostly with basis 0, a little with basis 1.
    let query = vec![normalize(vec![3.0, 1.0, 0.0, 0.0, 0.0, 0.0])];
    let q_fde = encoder.encode_query(&query).unwrap();

    let mut exact = Vec::new();
    let mut approx = Vec::new();
    for (_, tokens) in &docs {
        exact.push(encoder.chamfer(&query, tokens).unwrap());
        let d_fde = encoder.encode_document(tokens).unwrap();
        approx.push(q_fde.dot(&d_fde));
    }
    // "a" beats "b" beats "c" under both measures.
    assert!(exact[0] > exact[1] && exact[1] > exact[2]);
    assert_eq!(discordant_pairs(&exact, &approx), 0);
}

#[test]
fn chamfer_similarity_matches_manual_computation() {
    let dim = 4;
    let encoder = MuveraEncoder::new(MuveraConfig::new(dim).unwrap()).unwrap();
    let query = vec![basis(0, dim), basis(1, dim)];
    let doc = vec![
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.8, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    // max over doc of <e0, .> = 0.9 ; max over doc of <e1, .> = 0.8
    let expected = 0.9 + 0.8;
    let got = encoder.chamfer(&query, &doc).unwrap();
    assert!((got - expected).abs() < 1e-6);
}

#[test]
fn more_repetitions_reduce_ranking_errors() {
    // Coarse SimHash + a lossy inner projection makes a single repetition a
    // noisy Chamfer estimator; averaging over many repetitions must reduce the
    // aggregate ranking error against exact Chamfer.
    let dim = 16;
    let n_docs = 24;
    let n_queries = 24;
    let base = MuveraConfig::new(dim).unwrap().with_k_sim(3).with_d_proj(8);

    let docs: Vec<Vec<Vec<f32>>> = (0..n_docs)
        .map(|d| multi_set(&format!("D{d}"), 6, dim))
        .collect();
    let queries: Vec<Vec<Vec<f32>>> = (0..n_queries)
        .map(|q| multi_set(&format!("Q{q}"), 3, dim))
        .collect();

    let errors_for = |r_reps: usize| -> usize {
        let encoder = MuveraEncoder::new(base.with_r_reps(r_reps)).unwrap();
        let doc_fdes: Vec<_> = docs
            .iter()
            .map(|d| encoder.encode_document(d).unwrap())
            .collect();
        let mut total = 0;
        for query in &queries {
            let q_fde = encoder.encode_query(query).unwrap();
            let mut exact = Vec::new();
            let mut approx = Vec::new();
            for (doc, d_fde) in docs.iter().zip(doc_fdes.iter()) {
                exact.push(encoder.chamfer(query, doc).unwrap());
                approx.push(q_fde.dot(d_fde));
            }
            total += discordant_pairs(&exact, &approx);
        }
        total
    };

    let few = errors_for(1);
    let many = errors_for(32);
    assert!(few > 0, "single repetition should make ranking errors");
    assert!(
        many < few,
        "more repetitions should reduce ranking errors: 32-rep {many} vs 1-rep {few}"
    );
}

// ── Empty-cell fill behaviour ────────────────────────────────────────────────

#[test]
fn fill_populates_every_block_when_enabled() {
    // A document whose tokens all point along e0 occupies a single SimHash
    // cell; with fill enabled every block is populated, with it disabled the
    // other cells stay zero.
    let dim = 4;
    let doc = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![2.0, 0.0, 0.0, 0.0],
        vec![3.0, 0.0, 0.0, 0.0],
    ];

    let filled = MuveraEncoder::new(
        MuveraConfig::new(dim)
            .unwrap()
            .with_k_sim(2)
            .with_d_proj(0)
            .with_r_reps(1)
            .with_fill_empty(true),
    )
    .unwrap()
    .encode_document(&doc)
    .unwrap();
    let zero_filled = (0..filled.num_blocks())
        .filter(|&b| is_zero_block(filled.block(b).unwrap()))
        .count();
    assert_eq!(zero_filled, 0, "fill should leave no empty block");

    let unfilled = MuveraEncoder::new(
        MuveraConfig::new(dim)
            .unwrap()
            .with_k_sim(2)
            .with_d_proj(0)
            .with_r_reps(1)
            .with_fill_empty(false),
    )
    .unwrap()
    .encode_document(&doc)
    .unwrap();
    let zero_unfilled = (0..unfilled.num_blocks())
        .filter(|&b| is_zero_block(unfilled.block(b).unwrap()))
        .count();
    // B = 4 cells, only one occupied → three empty when fill is off.
    assert_eq!(zero_unfilled, 3);
}

#[test]
fn fill_borrows_single_nonempty_centroid_into_all_blocks() {
    let dim = 4;
    // Tokens all along e0 → one cell, centroid = mean = [2,0,0,0].
    let doc = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![2.0, 0.0, 0.0, 0.0],
        vec![3.0, 0.0, 0.0, 0.0],
    ];
    let fde = MuveraEncoder::new(
        MuveraConfig::new(dim)
            .unwrap()
            .with_k_sim(2)
            .with_d_proj(0)
            .with_r_reps(1)
            .with_fill_empty(true),
    )
    .unwrap()
    .encode_document(&doc)
    .unwrap();
    let centroid = vec![2.0f32, 0.0, 0.0, 0.0];
    for b in 0..fde.num_blocks() {
        assert_eq!(fde.block(b).unwrap(), centroid.as_slice());
    }
}

#[test]
fn without_fill_empty_blocks_are_zero() {
    let dim = 4;
    let doc = vec![vec![1.0, 0.0, 0.0, 0.0], vec![2.0, 0.0, 0.0, 0.0]];
    let fde = MuveraEncoder::new(
        MuveraConfig::new(dim)
            .unwrap()
            .with_k_sim(2)
            .with_d_proj(0)
            .with_r_reps(1)
            .with_fill_empty(false),
    )
    .unwrap()
    .encode_document(&doc)
    .unwrap();
    let non_zero = (0..fde.num_blocks())
        .filter(|&b| !is_zero_block(fde.block(b).unwrap()))
        .count();
    assert_eq!(non_zero, 1);
}

// ── nearest_occupied_bucket: the fill rule in isolation ──────────────────────

#[test]
fn nearest_bucket_none_when_all_empty() {
    let occupied = [false, false, false, false];
    assert_eq!(nearest_occupied_bucket(2, &occupied, 2), None);
}

#[test]
fn nearest_bucket_picks_only_occupied() {
    // Only bucket 0 (00) occupied; target 3 (11) at Hamming distance 2.
    let occupied = [true, false, false, false];
    assert_eq!(nearest_occupied_bucket(3, &occupied, 2), Some(0));
}

#[test]
fn nearest_bucket_minimises_hamming_distance() {
    // Buckets 0 (00) and 2 (10) occupied; target 1 (01):
    // Hamming(01,00)=1, Hamming(01,10)=2 → nearest is 0.
    let occupied = [true, false, true, false];
    assert_eq!(nearest_occupied_bucket(1, &occupied, 2), Some(0));
}

#[test]
fn nearest_bucket_tie_breaks_to_lowest_index() {
    // Buckets 1 (01) and 2 (10) occupied; target 0 (00) is Hamming distance 1
    // from both → tie broken to the lower index, 1.
    let occupied = [false, true, true, false];
    assert_eq!(nearest_occupied_bucket(0, &occupied, 2), Some(1));
    // target 3 (11) is likewise distance 1 from both → still 1.
    assert_eq!(nearest_occupied_bucket(3, &occupied, 2), Some(1));
}

#[test]
fn nearest_bucket_returns_target_when_occupied() {
    let occupied = [true, true, false, false];
    assert_eq!(nearest_occupied_bucket(0, &occupied, 2), Some(0));
}

// ── Exact re-rank ────────────────────────────────────────────────────────────

/// Fixtures that deterministically make the FDE (average-based) ordering
/// disagree with exact Chamfer: `dilute` has a strong token diluted by three
/// weak parallel copies (so its cell centroid is small), while `single`'s one
/// token is a better *centroid* match yet a worse *max* match.
fn rerank_fixtures() -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let u = normalize(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let weak: Vec<f32> = u.iter().map(|x| x * 0.02).collect();
    // dilute: one exact-match token + three tiny parallel copies.
    let dilute = vec![u.clone(), weak.clone(), weak.clone(), weak.clone()];
    // single: one token with cosine ≈ 0.588 to u.
    let t = normalize(vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let single = vec![t];
    let query = vec![u];
    (query, dilute, single)
}

#[test]
fn rerank_flips_approximate_ordering_to_match_chamfer() {
    let (query, dilute, single) = rerank_fixtures();
    let dim = 8;

    // Approximate: no re-rank. The diluted-centroid document scores lower.
    let approx_index = MuveraIndex::build(
        MuveraConfig::new(dim)
            .unwrap()
            .with_k_sim(2)
            .with_d_proj(0)
            .with_r_reps(1)
            .with_rerank(false),
        [
            ("dilute".to_string(), dilute.clone()),
            ("single".to_string(), single.clone()),
        ],
    )
    .unwrap();
    let approx = approx_index.search(&query, 2).unwrap();
    assert_eq!(
        approx[0].doc_id, "single",
        "FDE ranks the centroid match first"
    );

    // Exact re-rank: the true MaxSim winner (dilute, which contains an exact
    // match) is restored to the top.
    let exact_index = MuveraIndex::build(
        MuveraConfig::new(dim)
            .unwrap()
            .with_k_sim(2)
            .with_d_proj(0)
            .with_r_reps(1)
            .with_rerank(true),
        [
            ("dilute".to_string(), dilute.clone()),
            ("single".to_string(), single.clone()),
        ],
    )
    .unwrap();
    let exact = exact_index.search(&query, 2).unwrap();
    assert_eq!(
        exact[0].doc_id, "dilute",
        "exact Chamfer restores the max match"
    );
    assert_ne!(
        approx[0].doc_id, exact[0].doc_id,
        "re-rank changed the order"
    );
}

#[test]
fn rerank_scores_equal_exact_chamfer() {
    let (query, dilute, single) = rerank_fixtures();
    let dim = 8;
    let index = MuveraIndex::build(
        MuveraConfig::new(dim)
            .unwrap()
            .with_k_sim(2)
            .with_d_proj(0)
            .with_rerank(true),
        [
            ("dilute".to_string(), dilute.clone()),
            ("single".to_string(), single.clone()),
        ],
    )
    .unwrap();
    let hits = index.search(&query, 2).unwrap();
    for hit in &hits {
        let expected = index.exact_similarity(&query, &hit.doc_id).unwrap();
        assert!((hit.score - expected).abs() < 1e-6);
    }
    // The top exact score is the exact-match document's 1.0.
    assert!((hits[0].score - 1.0).abs() < 1e-6);
}

#[test]
fn rerank_depth_limits_candidate_pool() {
    // Stage 1 ranks the diluted-centroid document *below* "single"; a depth-1
    // re-rank window therefore only reconsiders "single", so the exact-better
    // but approx-worse "dilute" is pruned and cannot be promoted — whereas a
    // full-corpus re-rank (depth 0) restores it.
    let (query, dilute, single) = rerank_fixtures();
    let dim = 8;
    let base = MuveraConfig::new(dim)
        .unwrap()
        .with_k_sim(2)
        .with_d_proj(0)
        .with_r_reps(1)
        .with_rerank(true);

    let shallow = MuveraIndex::build(
        base.with_rerank_depth(1),
        [
            ("dilute".to_string(), dilute.clone()),
            ("single".to_string(), single.clone()),
        ],
    )
    .unwrap();
    let shallow_hits = shallow.search(&query, 1).unwrap();
    assert_eq!(shallow_hits.len(), 1);
    assert_eq!(
        shallow_hits[0].doc_id, "single",
        "depth-1 window prunes dilute"
    );

    let full = MuveraIndex::build(
        base.with_rerank_depth(0),
        [
            ("dilute".to_string(), dilute.clone()),
            ("single".to_string(), single.clone()),
        ],
    )
    .unwrap();
    let full_hits = full.search(&query, 1).unwrap();
    assert_eq!(
        full_hits[0].doc_id, "dilute",
        "full re-rank restores the max match"
    );
}

#[test]
fn search_default_uses_config_top_k() {
    let dim = 6;
    let config = MuveraConfig::new(dim).unwrap().with_top_k(2);
    let mut index = MuveraIndex::new(config).unwrap();
    for d in 0..5 {
        index
            .add(format!("d{d}"), multi_set(&format!("d{d}"), 4, dim))
            .unwrap();
    }
    let hits = index.search_default(&multi_set("q", 3, dim)).unwrap();
    assert_eq!(hits.len(), 2);
}

// ── Index basics ─────────────────────────────────────────────────────────────

#[test]
fn index_starts_empty() {
    let index = MuveraIndex::new(MuveraConfig::new(8).unwrap()).unwrap();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn index_add_grows_length() {
    let dim = 8;
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    index.add("a", multi_set("a", 3, dim)).unwrap();
    index.add("b", multi_set("b", 5, dim)).unwrap();
    assert_eq!(index.len(), 2);
    assert!(!index.is_empty());
}

#[test]
fn index_contains_and_document_accessors() {
    let dim = 8;
    let tokens = multi_set("a", 3, dim);
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    index.add("a", tokens.clone()).unwrap();
    assert!(index.contains("a"));
    assert!(!index.contains("z"));
    let doc = index.document("a").unwrap();
    assert_eq!(doc.id, "a");
    assert_eq!(doc.token_vectors, tokens);
    assert!(index.document("z").is_none());
}

#[test]
fn index_encoding_accessor_matches_encoder() {
    let dim = 8;
    let tokens = multi_set("a", 3, dim);
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    index.add("a", tokens.clone()).unwrap();
    let stored = index.encoding("a").unwrap();
    let recomputed = index.encoder().encode_document(&tokens).unwrap();
    assert_eq!(stored.as_slice(), recomputed.as_slice());
    assert!(index.encoding("z").is_none());
}

#[test]
fn index_build_from_iterator() {
    let dim = 8;
    let index = MuveraIndex::build(
        MuveraConfig::new(dim).unwrap(),
        [
            ("x".to_string(), multi_set("x", 3, dim)),
            ("y".to_string(), multi_set("y", 4, dim)),
        ],
    )
    .unwrap();
    assert_eq!(index.len(), 2);
    assert!(index.contains("x") && index.contains("y"));
}

#[test]
fn search_truncates_to_k() {
    let dim = 8;
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    for d in 0..6 {
        index
            .add(format!("d{d}"), multi_set(&format!("d{d}"), 4, dim))
            .unwrap();
    }
    let hits = index.search(&multi_set("q", 3, dim), 3).unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn search_results_are_descending_by_score() {
    let dim = 10;
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    for d in 0..8 {
        index
            .add(format!("d{d}"), multi_set(&format!("d{d}"), 5, dim))
            .unwrap();
    }
    let hits = index.search(&multi_set("q", 4, dim), 8).unwrap();
    for pair in hits.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

#[test]
fn exact_similarity_reports_chamfer() {
    let dim = 4;
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    let doc = vec![vec![0.9, 0.1, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    index.add("d", doc).unwrap();
    let query = vec![basis(0, dim), basis(1, dim)];
    let got = index.exact_similarity(&query, "d").unwrap();
    assert!((got - (0.9 + 1.0)).abs() < 1e-6);
}

// ── MuveraDocument / MuveraSimilarity value types ────────────────────────────

#[test]
fn muvera_document_len_and_empty() {
    let doc = MuveraDocument::new("id", vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(doc.id, "id");
    assert_eq!(doc.len(), 2);
    assert!(!doc.is_empty());
    assert!(MuveraDocument::new("e", Vec::new()).is_empty());
}

#[test]
fn muvera_similarity_constructor() {
    let hit = MuveraSimilarity::new("doc", 1.5);
    assert_eq!(hit.doc_id, "doc");
    assert_eq!(hit.score, 1.5);
}

#[test]
fn fixed_dim_encoding_dot_is_symmetric() {
    let encoder =
        MuveraEncoder::new(MuveraConfig::new(6).unwrap().with_k_sim(2).with_d_proj(0)).unwrap();
    let a = encoder.encode_document(&multi_set("a", 4, 6)).unwrap();
    let b = encoder.encode_document(&multi_set("b", 4, 6)).unwrap();
    assert!((a.dot(&b) - b.dot(&a)).abs() < 1e-6);
}

// ── Error paths ──────────────────────────────────────────────────────────────

#[test]
fn encode_query_rejects_empty_set() {
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap()).unwrap();
    assert_eq!(
        encoder.encode_query(&[]),
        Err(MuveraError::EmptyMultiVector)
    );
}

#[test]
fn encode_document_rejects_empty_set() {
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap()).unwrap();
    assert_eq!(
        encoder.encode_document(&[]),
        Err(MuveraError::EmptyMultiVector)
    );
}

#[test]
fn encode_rejects_dimension_mismatch() {
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap()).unwrap();
    let bad = vec![vec![1.0f32; 7]]; // expected 8
    let err = encoder.encode_document(&bad).unwrap_err();
    assert_eq!(
        err,
        MuveraError::DimensionMismatch {
            expected: 8,
            got: 7
        }
    );
}

#[test]
fn token_bucket_rejects_dimension_mismatch() {
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap()).unwrap();
    let err = encoder.token_bucket(&[1.0, 2.0], 0).unwrap_err();
    assert_eq!(
        err,
        MuveraError::DimensionMismatch {
            expected: 8,
            got: 2
        }
    );
}

#[test]
fn token_bucket_rejects_out_of_range_repetition() {
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap().with_r_reps(2)).unwrap();
    let token = pseudo_vec("x", 8);
    assert!(matches!(
        encoder.token_bucket(&token, 5),
        Err(MuveraError::InvalidConfig(_))
    ));
}

#[test]
fn chamfer_rejects_empty_sets() {
    let encoder = MuveraEncoder::new(MuveraConfig::new(8).unwrap()).unwrap();
    let good = multi_set("g", 3, 8);
    assert_eq!(
        encoder.chamfer(&[], &good),
        Err(MuveraError::EmptyMultiVector)
    );
    assert_eq!(
        encoder.chamfer(&good, &[]),
        Err(MuveraError::EmptyMultiVector)
    );
}

#[test]
fn search_rejects_empty_index() {
    let index = MuveraIndex::new(MuveraConfig::new(8).unwrap()).unwrap();
    let err = index.search(&multi_set("q", 3, 8), 5).unwrap_err();
    assert_eq!(err, MuveraError::EmptyIndex);
}

#[test]
fn search_rejects_empty_query() {
    let dim = 8;
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    index.add("a", multi_set("a", 3, dim)).unwrap();
    assert_eq!(index.search(&[], 5), Err(MuveraError::EmptyMultiVector));
}

#[test]
fn search_rejects_query_dimension_mismatch() {
    let dim = 8;
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    index.add("a", multi_set("a", 3, dim)).unwrap();
    let bad_query = vec![vec![1.0f32; 4]];
    let err = index.search(&bad_query, 5).unwrap_err();
    assert_eq!(
        err,
        MuveraError::DimensionMismatch {
            expected: 8,
            got: 4
        }
    );
}

#[test]
fn add_rejects_duplicate_id() {
    let dim = 8;
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    index.add("a", multi_set("a", 3, dim)).unwrap();
    let err = index.add("a", multi_set("a2", 3, dim)).unwrap_err();
    assert_eq!(err, MuveraError::DuplicateId("a".to_string()));
}

#[test]
fn add_rejects_empty_document() {
    let mut index = MuveraIndex::new(MuveraConfig::new(8).unwrap()).unwrap();
    assert_eq!(
        index.add("a", Vec::new()),
        Err(MuveraError::EmptyMultiVector)
    );
}

#[test]
fn add_rejects_dimension_mismatch() {
    let mut index = MuveraIndex::new(MuveraConfig::new(8).unwrap()).unwrap();
    let err = index.add("a", vec![vec![1.0f32; 3]]).unwrap_err();
    assert_eq!(
        err,
        MuveraError::DimensionMismatch {
            expected: 8,
            got: 3
        }
    );
}

#[test]
fn build_propagates_duplicate_id() {
    let dim = 8;
    let err = MuveraIndex::build(
        MuveraConfig::new(dim).unwrap(),
        [
            ("dup".to_string(), multi_set("a", 3, dim)),
            ("dup".to_string(), multi_set("b", 3, dim)),
        ],
    )
    .unwrap_err();
    assert_eq!(err, MuveraError::DuplicateId("dup".to_string()));
}

#[test]
fn exact_similarity_rejects_unknown_id() {
    let dim = 8;
    let mut index = MuveraIndex::new(MuveraConfig::new(dim).unwrap()).unwrap();
    index.add("a", multi_set("a", 3, dim)).unwrap();
    assert_eq!(
        index.exact_similarity(&multi_set("q", 3, dim), "missing"),
        Err(MuveraError::EmptyIndex)
    );
}

#[test]
fn encoder_new_rejects_invalid_config() {
    let bad = MuveraConfig::new(8).unwrap().with_k_sim(0);
    assert!(matches!(
        MuveraEncoder::new(bad),
        Err(MuveraError::InvalidConfig(_))
    ));
}

#[test]
fn index_new_rejects_invalid_config() {
    let bad = MuveraConfig::new(0);
    assert!(bad.is_err());
    let bad2 = MuveraConfig::new(8).unwrap().with_r_reps(0);
    assert!(matches!(
        MuveraIndex::new(bad2),
        Err(MuveraError::InvalidConfig(_))
    ));
}
