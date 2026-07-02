//! Comprehensive tests for the `rp_tree_index` module.
//!
//! Covers the configuration/type surface, the deterministic `splitmix64`
//! generator, single-tree construction (termination, leaf-size and depth
//! bounds, degenerate all-identical input, determinism), the forest, and
//! forest search (exact-vs-brute-force equivalence, determinism, and the
//! "more trees ⇒ higher recall" property).

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::items_after_statements
)]

use super::tree::{RpTree, SplitMix64, tree_seed};
use super::types::{
    RpTreeConfig, RpTreeError, RpTreeHit, RpTreeHyperplane, RpTreeMetric, RpTreeNode,
};
use super::{RpTreeForest, RpTreeIndex};

// ── Test helpers ─────────────────────────────────────────────────────────────

/// Deterministic pseudo-random embedding of `text` into `dim` dimensions, each
/// component well-distributed in `[-1, 1)`.
///
/// A per-string FNV-1a base hash is combined with the dimension index and run
/// through a `splitmix64` finaliser, so — crucially — every component varies
/// with the dimension (a naive `hash >> 32` would collapse all components to a
/// single value and make every vector parallel).
fn embed(text: &str, dim: usize) -> Vec<f32> {
    let mut base: u64 = 14_695_981_039_346_656_037;
    for &b in text.as_bytes() {
        base = base.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
    }
    (0..dim)
        .map(|i| {
            let mut z = base.wrapping_add(
                (i as u64)
                    .wrapping_add(1)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Top 24 bits mapped uniformly into [-1, 1).
            (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Build a corpus of `n` distinct `(id, vector)` points with the given prefix.
fn corpus(prefix: &str, n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
    (0..n)
        .map(|i| {
            let id = format!("{prefix}{i}");
            let v = embed(&id, dim);
            (id, v)
        })
        .collect()
}

/// Extract the raw vectors of a corpus (for direct single-tree construction).
fn vectors_of(points: &[(String, Vec<f32>)]) -> Vec<Vec<f32>> {
    points.iter().map(|(_, v)| v.clone()).collect()
}

/// Brute-force top-`k` ids, using the exact same score and tie-break
/// (descending score, then ascending insertion index) as the forest ranker.
fn brute_force(
    points: &[(String, Vec<f32>)],
    query: &[f32],
    k: usize,
    metric: RpTreeMetric,
) -> Vec<String> {
    let q = metric.preprocess(query.to_vec());
    let mut scored: Vec<(f32, usize)> = points
        .iter()
        .enumerate()
        .map(|(i, (_, v))| (metric.score(&q, &metric.preprocess(v.clone())), i))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.truncate(k);
    scored
        .into_iter()
        .map(|(_, i)| points[i].0.clone())
        .collect()
}

// ── RpTreeMetric ──────────────────────────────────────────────────────────────

#[test]
fn metric_default_is_cosine() {
    assert_eq!(RpTreeMetric::default(), RpTreeMetric::Cosine);
}

#[test]
fn metric_is_cosine_is_l2() {
    assert!(RpTreeMetric::Cosine.is_cosine());
    assert!(!RpTreeMetric::Cosine.is_l2());
    assert!(RpTreeMetric::L2.is_l2());
    assert!(!RpTreeMetric::L2.is_cosine());
}

#[test]
fn metric_preprocess_cosine_normalizes() {
    let out = RpTreeMetric::Cosine.preprocess(vec![3.0, 4.0]);
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "expected unit length, got {norm}"
    );
}

#[test]
fn metric_preprocess_l2_is_identity() {
    let input = vec![3.0, 4.0, 5.0];
    assert_eq!(RpTreeMetric::L2.preprocess(input.clone()), input);
}

#[test]
fn metric_score_cosine_in_range_and_self_is_one() {
    let a = RpTreeMetric::Cosine.preprocess(vec![1.0, 2.0, 3.0]);
    let self_score = RpTreeMetric::Cosine.score(&a, &a);
    assert!((self_score - 1.0).abs() < 1e-5);
    let opposite = RpTreeMetric::Cosine.preprocess(vec![-1.0, -2.0, -3.0]);
    let opp_score = RpTreeMetric::Cosine.score(&a, &opposite);
    assert!((0.0..=1.0).contains(&opp_score));
    assert!(opp_score < self_score);
}

#[test]
fn metric_score_l2_in_range_and_self_is_one() {
    let a = vec![1.0, 2.0, 3.0];
    let self_score = RpTreeMetric::L2.score(&a, &a);
    assert!((self_score - 1.0).abs() < 1e-6);
    let far = vec![100.0, 100.0, 100.0];
    let far_score = RpTreeMetric::L2.score(&a, &far);
    assert!(far_score > 0.0 && far_score < self_score);
}

// ── RpTreeConfig ──────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = RpTreeConfig::default();
    assert_eq!(cfg.n_trees, 10);
    assert_eq!(cfg.leaf_size, 16);
    assert_eq!(cfg.search_k, 100);
    assert_eq!(cfg.max_depth, 32);
    assert_eq!(cfg.metric, RpTreeMetric::Cosine);
    assert_eq!(cfg.dim, 128);
}

#[test]
fn config_builder_chain() {
    let cfg = RpTreeConfig::new()
        .with_dim(7)
        .with_n_trees(3)
        .with_leaf_size(5)
        .with_search_k(42)
        .with_max_depth(9)
        .with_seed(1234)
        .with_metric(RpTreeMetric::L2);
    assert_eq!(cfg.dim, 7);
    assert_eq!(cfg.n_trees, 3);
    assert_eq!(cfg.leaf_size, 5);
    assert_eq!(cfg.search_k, 42);
    assert_eq!(cfg.max_depth, 9);
    assert_eq!(cfg.seed, 1234);
    assert_eq!(cfg.metric, RpTreeMetric::L2);
}

#[test]
fn config_validate_ok() {
    assert!(RpTreeConfig::default().validate().is_ok());
}

#[test]
fn config_validate_rejects_zero_dim() {
    let err = RpTreeConfig::new().with_dim(0).validate().unwrap_err();
    assert!(matches!(err, RpTreeError::InvalidConfig(_)));
}

#[test]
fn config_validate_rejects_zero_n_trees() {
    let err = RpTreeConfig::new().with_n_trees(0).validate().unwrap_err();
    assert!(matches!(err, RpTreeError::InvalidConfig(_)));
}

#[test]
fn config_validate_rejects_zero_leaf_size() {
    let err = RpTreeConfig::new()
        .with_leaf_size(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, RpTreeError::InvalidConfig(_)));
}

#[test]
fn config_validate_rejects_zero_search_k() {
    let err = RpTreeConfig::new().with_search_k(0).validate().unwrap_err();
    assert!(matches!(err, RpTreeError::InvalidConfig(_)));
}

#[test]
fn config_validate_rejects_zero_max_depth() {
    let err = RpTreeConfig::new()
        .with_max_depth(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, RpTreeError::InvalidConfig(_)));
}

// ── RpTreeHyperplane / RpTreeNode / RpTreeHit / RpTreeError ─────────────────────

#[test]
fn hyperplane_margin_positive_and_negative() {
    // Plane x = 0.5 : normal (1,0), bias 0.5.
    let hp = RpTreeHyperplane::new(vec![1.0, 0.0], 0.5);
    assert!(hp.margin(&[1.0, 9.0]) > 0.0);
    assert!(hp.margin(&[0.0, 9.0]) < 0.0);
    assert!((hp.margin(&[0.5, 9.0]) - 0.0).abs() < 1e-6);
}

#[test]
fn hyperplane_degenerate_margin_always_zero() {
    let hp = RpTreeHyperplane::degenerate(4);
    assert!(hp.normal.iter().all(|&x| x == 0.0));
    assert_eq!(hp.margin(&[1.0, -2.0, 3.0, 4.0]), 0.0);
    assert_eq!(hp.margin(&[9.0, 9.0, 9.0, 9.0]), 0.0);
}

#[test]
fn node_is_leaf_and_leaf_len() {
    let leaf = RpTreeNode::Leaf { ids: vec![1, 2, 3] };
    assert!(leaf.is_leaf());
    assert_eq!(leaf.leaf_len(), 3);

    let internal = RpTreeNode::Internal {
        hyperplane: RpTreeHyperplane::degenerate(2),
        left: 0,
        right: 1,
    };
    assert!(!internal.is_leaf());
    assert_eq!(internal.leaf_len(), 0);
}

#[test]
fn hit_new_fields() {
    let hit = RpTreeHit::new("doc", 0.75);
    assert_eq!(hit.id, "doc");
    assert_eq!(hit.score, 0.75);
}

#[test]
fn error_display_messages() {
    let dim = RpTreeError::DimMismatch {
        expected: 4,
        got: 3,
    };
    assert!(dim.to_string().contains("expected 4"));
    assert_eq!(RpTreeError::EmptyIndex.to_string(), "index is empty");
    assert_eq!(RpTreeError::InvalidK.to_string(), "k must be >= 1");
    assert!(RpTreeError::EmptyPoints.to_string().contains("empty"));
    assert!(
        RpTreeError::InvalidConfig("x".into())
            .to_string()
            .contains('x')
    );
}

// ── SplitMix64 / tree_seed ────────────────────────────────────────────────────

#[test]
fn splitmix64_is_deterministic() {
    let mut a = SplitMix64::new(99);
    let mut b = SplitMix64::new(99);
    let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
    let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
    assert_eq!(seq_a, seq_b);
    // Distinct seeds produce distinct streams.
    let mut c = SplitMix64::new(100);
    let seq_c: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
    assert_ne!(seq_a, seq_c);
}

#[test]
fn splitmix64_index_within_bound() {
    let mut rng = SplitMix64::new(7);
    for _ in 0..1000 {
        assert!(rng.index(5) < 5);
    }
    assert_eq!(rng.index(0), 0);
}

#[test]
fn tree_seed_distinct_per_tree_but_reproducible() {
    let seeds: Vec<u64> = (0..8).map(|t| tree_seed(0x_5EED_C0DE, t)).collect();
    // All eight per-tree seeds are distinct.
    for i in 0..seeds.len() {
        for j in (i + 1)..seeds.len() {
            assert_ne!(seeds[i], seeds[j], "trees {i} and {j} share a seed");
        }
    }
    // Reproducible from the same forest seed.
    let again: Vec<u64> = (0..8).map(|t| tree_seed(0x_5EED_C0DE, t)).collect();
    assert_eq!(seeds, again);
    // A different forest seed shifts the whole schedule.
    let other: Vec<u64> = (0..8).map(|t| tree_seed(1, t)).collect();
    assert_ne!(seeds, other);
}

// ── Single-tree construction ──────────────────────────────────────────────────

#[test]
fn tree_single_leaf_when_within_leaf_size() {
    let points = corpus("s", 4, 6);
    let vectors = vectors_of(&points);
    let cfg = RpTreeConfig::new()
        .with_dim(6)
        .with_leaf_size(8)
        .with_max_depth(32);
    let tree = RpTree::build((0..4).collect(), &vectors, &cfg, 1).unwrap();
    // n <= leaf_size ⇒ the root itself is the single leaf.
    assert_eq!(tree.node_count(), 1);
    assert!(tree.nodes()[tree.root()].is_leaf());
    assert_eq!(tree.observed_depth(), 0);
}

#[test]
fn tree_two_points_produce_valid_structure() {
    let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let cfg = RpTreeConfig::new()
        .with_dim(3)
        .with_leaf_size(1)
        .with_max_depth(8);
    let tree = RpTree::build(vec![0, 1], &vectors, &cfg, 3).unwrap();
    // Distinct points, leaf_size 1 ⇒ one internal node with two singleton leaves.
    assert_eq!(tree.collect_all_ids(), vec![0, 1]);
    assert!(tree.leaf_sizes().iter().all(|&s| s <= 1));
}

#[test]
fn tree_every_leaf_within_leaf_size() {
    let points = corpus("leaf", 64, 8);
    let vectors = vectors_of(&points);
    let leaf_size = 8;
    // max_depth is set far above any possible path length so the depth cap
    // never fires and every leaf must reach the leaf_size bound.
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_leaf_size(leaf_size)
        .with_max_depth(256);
    let tree = RpTree::build((0..64).collect(), &vectors, &cfg, 42).unwrap();
    for size in tree.leaf_sizes() {
        assert!(size <= leaf_size, "leaf of size {size} exceeds {leaf_size}");
    }
}

#[test]
fn tree_all_points_present_and_unique() {
    let points = corpus("all", 50, 8);
    let vectors = vectors_of(&points);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_leaf_size(4)
        .with_max_depth(64);
    let tree = RpTree::build((0..50).collect(), &vectors, &cfg, 11).unwrap();
    let all = tree.collect_all_ids();
    let expected: Vec<usize> = (0..50).collect();
    assert_eq!(
        all, expected,
        "every point appears exactly once across leaves"
    );
}

#[test]
fn tree_is_full_binary_node_count_invariant() {
    let points = corpus("fb", 40, 8);
    let vectors = vectors_of(&points);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_leaf_size(3)
        .with_max_depth(64);
    let tree = RpTree::build((0..40).collect(), &vectors, &cfg, 5).unwrap();
    let leaves = tree.leaf_sizes().len();
    // A full binary tree (every internal node has exactly two children) has
    // `2 * leaves - 1` nodes in total.
    assert_eq!(tree.node_count(), 2 * leaves - 1);
}

#[test]
fn tree_depth_never_exceeds_max_depth() {
    let points = corpus("depth", 128, 8);
    let vectors = vectors_of(&points);
    for &max_depth in &[1usize, 2, 4, 8, 16] {
        let cfg = RpTreeConfig::new()
            .with_dim(8)
            .with_leaf_size(1)
            .with_max_depth(max_depth);
        let tree = RpTree::build((0..128).collect(), &vectors, &cfg, 77).unwrap();
        assert!(
            tree.observed_depth() <= max_depth,
            "observed depth {} exceeds cap {max_depth}",
            tree.observed_depth()
        );
        // No points lost regardless of the cap.
        assert_eq!(tree.collect_all_ids(), (0..128).collect::<Vec<_>>());
    }
}

#[test]
fn tree_leaf_size_one_forces_maximal_splitting() {
    let points = corpus("split", 16, 8);
    let vectors = vectors_of(&points);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_leaf_size(1)
        .with_max_depth(64);
    let tree = RpTree::build((0..16).collect(), &vectors, &cfg, 9).unwrap();
    // Distinct points fully partitioned ⇒ each leaf holds exactly one point.
    let sizes = tree.leaf_sizes();
    assert_eq!(sizes.len(), 16);
    assert!(sizes.iter().all(|&s| s == 1));
    assert!(tree.node_count() > 1, "expected internal nodes");
}

#[test]
fn tree_small_max_depth_forces_early_leaves() {
    let points = corpus("cap", 64, 8);
    let vectors = vectors_of(&points);
    let max_depth = 3;
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_leaf_size(1)
        .with_max_depth(max_depth);
    let tree = RpTree::build((0..64).collect(), &vectors, &cfg, 21).unwrap();
    assert!(tree.observed_depth() <= max_depth);
    // 64 points cannot fit into at most 2^3 = 8 singleton leaves, so the cap
    // must have produced at least one over-sized leaf.
    let biggest = tree.leaf_sizes().into_iter().max().unwrap_or(0);
    assert!(biggest > 1, "depth cap should force over-sized leaves");
    assert_eq!(tree.collect_all_ids(), (0..64).collect::<Vec<_>>());
}

#[test]
fn tree_identical_points_terminate_without_overflow() {
    // 32 identical vectors: no geometric split exists, so the degenerate
    // median-by-rank fallback must still terminate the recursion.
    let dim = 8;
    let vectors = vec![vec![0.5f32; dim]; 32];
    let max_depth = 16;
    let cfg = RpTreeConfig::new()
        .with_dim(dim)
        .with_leaf_size(1)
        .with_max_depth(max_depth);
    let tree = RpTree::build((0..32).collect(), &vectors, &cfg, 3).unwrap();
    assert_eq!(tree.collect_all_ids(), (0..32).collect::<Vec<_>>());
    assert!(tree.observed_depth() <= max_depth);
    // Halving 32 identical points reaches singleton leaves at depth 5.
    assert!(tree.leaf_sizes().iter().all(|&s| s <= 1));
}

#[test]
fn tree_identical_points_respect_small_depth_cap() {
    let dim = 4;
    let vectors = vec![vec![-1.0f32, 2.0, -3.0, 4.0]; 40];
    let max_depth = 3;
    let cfg = RpTreeConfig::new()
        .with_dim(dim)
        .with_leaf_size(1)
        .with_max_depth(max_depth);
    let tree = RpTree::build((0..40).collect(), &vectors, &cfg, 8).unwrap();
    assert!(tree.observed_depth() <= max_depth);
    assert_eq!(tree.collect_all_ids(), (0..40).collect::<Vec<_>>());
    let biggest = tree.leaf_sizes().into_iter().max().unwrap_or(0);
    assert!(biggest > 1);
}

#[test]
fn tree_construction_is_deterministic_for_same_seed() {
    let points = corpus("det", 60, 8);
    let vectors = vectors_of(&points);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_leaf_size(2)
        .with_max_depth(32);
    let a = RpTree::build((0..60).collect(), &vectors, &cfg, 555).unwrap();
    let b = RpTree::build((0..60).collect(), &vectors, &cfg, 555).unwrap();
    assert_eq!(a.nodes(), b.nodes(), "same seed must reproduce the tree");
}

#[test]
fn tree_different_seeds_produce_different_structures() {
    let points = corpus("diff", 48, 8);
    let vectors = vectors_of(&points);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_leaf_size(2)
        .with_max_depth(32);
    let a = RpTree::build((0..48).collect(), &vectors, &cfg, 1).unwrap();
    let b = RpTree::build((0..48).collect(), &vectors, &cfg, 2).unwrap();
    assert_ne!(
        a.nodes(),
        b.nodes(),
        "different seeds should yield different splits"
    );
    // Yet both remain internally valid partitions of the same point set.
    assert_eq!(a.collect_all_ids(), (0..48).collect::<Vec<_>>());
    assert_eq!(b.collect_all_ids(), (0..48).collect::<Vec<_>>());
}

// ── RpTreeForest construction ─────────────────────────────────────────────────

#[test]
fn forest_builds_requested_number_of_trees() {
    let points = corpus("f", 30, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(7)
        .with_leaf_size(4);
    let forest = RpTreeForest::build(points, cfg).unwrap();
    assert_eq!(forest.num_trees(), 7);
    assert_eq!(forest.trees().len(), 7);
    assert_eq!(forest.len(), 30);
    assert!(!forest.is_empty());
}

#[test]
fn forest_build_rejects_empty_points() {
    let cfg = RpTreeConfig::new().with_dim(8);
    let err = RpTreeForest::build(Vec::new(), cfg).unwrap_err();
    assert_eq!(err, RpTreeError::EmptyPoints);
}

#[test]
fn forest_build_rejects_dim_mismatch() {
    let cfg = RpTreeConfig::new().with_dim(4);
    let points = vec![
        ("ok".to_string(), vec![1.0, 2.0, 3.0, 4.0]),
        ("bad".to_string(), vec![1.0, 2.0]),
    ];
    let err = RpTreeForest::build(points, cfg).unwrap_err();
    assert_eq!(
        err,
        RpTreeError::DimMismatch {
            expected: 4,
            got: 2
        }
    );
}

#[test]
fn forest_build_rejects_invalid_config() {
    let cfg = RpTreeConfig::new().with_dim(0);
    let points = corpus("bad", 3, 1);
    let err = RpTreeForest::build(points, cfg).unwrap_err();
    assert!(matches!(err, RpTreeError::InvalidConfig(_)));
}

#[test]
fn forest_every_tree_covers_all_points() {
    let points = corpus("cov", 40, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(5)
        .with_leaf_size(3);
    let forest = RpTreeForest::build(points, cfg).unwrap();
    for tree in forest.trees() {
        assert_eq!(tree.collect_all_ids(), (0..40).collect::<Vec<_>>());
    }
}

#[test]
fn forest_is_deterministic_for_same_config() {
    let points = corpus("fd", 40, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(4)
        .with_leaf_size(2);
    let a = RpTreeForest::build(points.clone(), cfg.clone()).unwrap();
    let b = RpTreeForest::build(points, cfg).unwrap();
    for (ta, tb) in a.trees().iter().zip(b.trees().iter()) {
        assert_eq!(ta.nodes(), tb.nodes());
    }
}

#[test]
fn forest_trees_are_independent() {
    let points = corpus("ind", 40, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(4)
        .with_leaf_size(2);
    let forest = RpTreeForest::build(points, cfg).unwrap();
    let trees = forest.trees();
    // With four different derived seeds at least one pair of trees must differ.
    let mut any_different = false;
    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            if trees[i].nodes() != trees[j].nodes() {
                any_different = true;
            }
        }
    }
    assert!(any_different, "trees should not all be identical");
}

// ── RpTreeIndex basics ────────────────────────────────────────────────────────

#[test]
fn index_build_len_and_num_trees() {
    let points = corpus("i", 25, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(6)
        .with_leaf_size(4);
    let index = RpTreeIndex::build(points, cfg).unwrap();
    assert_eq!(index.len(), 25);
    assert_eq!(index.num_trees(), 6);
    assert!(!index.is_empty());
    assert_eq!(index.config().dim, 8);
    assert_eq!(index.forest().len(), 25);
}

#[test]
fn search_rejects_zero_k() {
    let points = corpus("z", 10, 8);
    let cfg = RpTreeConfig::new().with_dim(8).with_leaf_size(2);
    let index = RpTreeIndex::build(points, cfg).unwrap();
    let err = index.search(&embed("z0", 8), 0).unwrap_err();
    assert_eq!(err, RpTreeError::InvalidK);
}

#[test]
fn search_rejects_query_dim_mismatch() {
    let points = corpus("d", 10, 8);
    let cfg = RpTreeConfig::new().with_dim(8).with_leaf_size(2);
    let index = RpTreeIndex::build(points, cfg).unwrap();
    let err = index.search(&[1.0, 2.0, 3.0], 3).unwrap_err();
    assert_eq!(
        err,
        RpTreeError::DimMismatch {
            expected: 8,
            got: 3
        }
    );
}

#[test]
fn search_returns_at_most_k_hits() {
    let points = corpus("k", 40, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(4)
        .with_leaf_size(2)
        .with_search_k(50);
    let index = RpTreeIndex::build(points, cfg).unwrap();
    let hits = index.search(&embed("k3", 8), 5).unwrap();
    assert!(hits.len() <= 5);
    assert!(!hits.is_empty());
}

#[test]
fn search_k_larger_than_point_count_is_bounded() {
    let points = corpus("b", 6, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(3)
        .with_leaf_size(2)
        .with_search_k(100);
    let index = RpTreeIndex::build(points, cfg).unwrap();
    let hits = index.search(&embed("b1", 8), 100).unwrap();
    // Never more than the number of indexed points.
    assert_eq!(hits.len(), 6);
}

#[test]
fn search_finds_query_that_is_an_indexed_point() {
    let points = corpus("self", 30, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(8)
        .with_leaf_size(1)
        .with_search_k(200)
        .with_metric(RpTreeMetric::Cosine);
    let index = RpTreeIndex::build(points.clone(), cfg).unwrap();
    // Querying with an indexed vector must return that point first.
    let (id, vector) = &points[7];
    let hits = index.search(vector, 1).unwrap();
    assert_eq!(&hits[0].id, id);
    assert!((hits[0].score - 1.0).abs() < 1e-4);
}

#[test]
fn search_scores_are_descending() {
    let points = corpus("desc", 40, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(6)
        .with_leaf_size(2)
        .with_search_k(200);
    let index = RpTreeIndex::build(points, cfg).unwrap();
    let hits = index.search(&embed("desc5", 8), 10).unwrap();
    for pair in hits.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "scores not descending: {} then {}",
            pair[0].score,
            pair[1].score
        );
    }
}

#[test]
fn search_returned_ids_are_all_from_the_corpus() {
    let points = corpus("valid", 30, 8);
    let ids: std::collections::HashSet<String> = points.iter().map(|(id, _)| id.clone()).collect();
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(4)
        .with_leaf_size(2)
        .with_search_k(50);
    let index = RpTreeIndex::build(points, cfg).unwrap();
    let hits = index.search(&embed("valid2", 8), 8).unwrap();
    for hit in hits {
        assert!(ids.contains(&hit.id));
    }
}

// ── Exact-vs-brute-force equivalence ──────────────────────────────────────────

#[test]
fn search_exact_matches_bruteforce_cosine() {
    let points = corpus("bf", 40, 6);
    // search_k far above the corpus size ⇒ every candidate is collected, so the
    // exact re-ranking must reproduce a full brute-force scan.
    let cfg = RpTreeConfig::new()
        .with_dim(6)
        .with_n_trees(3)
        .with_leaf_size(2)
        .with_search_k(10_000)
        .with_metric(RpTreeMetric::Cosine);
    let index = RpTreeIndex::build(points.clone(), cfg).unwrap();

    for q in ["bf0", "bf17", "bf39", "unseen-query", "another"] {
        let query = embed(q, 6);
        let got: Vec<String> = index
            .search(&query, 5)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let want = brute_force(&points, &query, 5, RpTreeMetric::Cosine);
        assert_eq!(got, want, "mismatch for query {q}");
    }
}

#[test]
fn search_exact_matches_bruteforce_l2() {
    let points = corpus("bfl", 35, 6);
    let cfg = RpTreeConfig::new()
        .with_dim(6)
        .with_n_trees(3)
        .with_leaf_size(2)
        .with_search_k(10_000)
        .with_metric(RpTreeMetric::L2);
    let index = RpTreeIndex::build(points.clone(), cfg).unwrap();

    for q in ["bfl0", "bfl20", "bfl34", "l2-unseen"] {
        let query = embed(q, 6);
        let got: Vec<String> = index
            .search(&query, 4)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let want = brute_force(&points, &query, 4, RpTreeMetric::L2);
        assert_eq!(got, want, "L2 mismatch for query {q}");
    }
}

#[test]
fn search_single_tree_full_budget_equals_bruteforce() {
    // Even a single tree, given an unlimited candidate budget, is exact.
    let points = corpus("st", 24, 6);
    let cfg = RpTreeConfig::new()
        .with_dim(6)
        .with_n_trees(1)
        .with_leaf_size(1)
        .with_search_k(10_000)
        .with_metric(RpTreeMetric::Cosine);
    let index = RpTreeIndex::build(points.clone(), cfg).unwrap();
    let query = embed("st-query", 6);
    let got: Vec<String> = index
        .search(&query, 6)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    let want = brute_force(&points, &query, 6, RpTreeMetric::Cosine);
    assert_eq!(got, want);
}

// ── Determinism of search ─────────────────────────────────────────────────────

#[test]
fn search_is_deterministic() {
    let points = corpus("sd", 50, 8);
    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(6)
        .with_leaf_size(2)
        .with_search_k(20)
        .with_seed(2024);
    let index_a = RpTreeIndex::build(points.clone(), cfg.clone()).unwrap();
    let index_b = RpTreeIndex::build(points, cfg).unwrap();
    let query = embed("sd-q", 8);
    let a = index_a.search(&query, 7).unwrap();
    let b = index_b.search(&query, 7).unwrap();
    assert_eq!(a.len(), b.len());
    for (ha, hb) in a.iter().zip(b.iter()) {
        assert_eq!(ha.id, hb.id);
        assert_eq!(ha.score, hb.score);
    }
}

// ── The core "why multiple trees" property ────────────────────────────────────

#[test]
fn recall_improves_with_more_trees() {
    let dim = 8;
    let metric = RpTreeMetric::Cosine;
    let points = corpus("c", 80, dim);
    let queries: Vec<Vec<f32>> = (0..40).map(|j| embed(&format!("q{j}"), dim)).collect();

    // Ground truth top-1 for each query.
    let truth: Vec<String> = queries
        .iter()
        .map(|q| brute_force(&points, q, 1, metric)[0].clone())
        .collect();

    // A deliberately small candidate budget so a single partition frequently
    // separates the query from its true nearest neighbour.
    let base = RpTreeConfig::new()
        .with_dim(dim)
        .with_leaf_size(1)
        .with_search_k(4)
        .with_metric(metric)
        .with_seed(1234);

    let count_recall = |n_trees: usize| -> usize {
        let cfg = base.clone().with_n_trees(n_trees);
        let index = RpTreeIndex::build(points.clone(), cfg).unwrap();
        queries
            .iter()
            .zip(truth.iter())
            .filter(|(q, t)| {
                index
                    .search(q, 1)
                    .is_ok_and(|hits| hits.first().map(|h| &h.id) == Some(*t))
            })
            .count()
    };

    let recall_1 = count_recall(1);
    let recall_8 = count_recall(8);

    // The whole point of a forest: more independent trees find the true nearest
    // neighbour strictly more often on this adversarial, low-budget setup.
    assert!(
        recall_8 > recall_1,
        "expected more trees to improve recall, got recall_1={recall_1}, recall_8={recall_8}"
    );
}

#[test]
fn recall_is_monotone_nondecreasing_across_tree_counts() {
    let dim = 8;
    let metric = RpTreeMetric::Cosine;
    let points = corpus("m", 80, dim);
    let queries: Vec<Vec<f32>> = (0..40).map(|j| embed(&format!("mq{j}"), dim)).collect();
    let truth: Vec<String> = queries
        .iter()
        .map(|q| brute_force(&points, q, 1, metric)[0].clone())
        .collect();

    let base = RpTreeConfig::new()
        .with_dim(dim)
        .with_leaf_size(1)
        .with_search_k(4)
        .with_metric(metric)
        .with_seed(4242);

    let count_recall = |n_trees: usize| -> usize {
        let cfg = base.clone().with_n_trees(n_trees);
        let index = RpTreeIndex::build(points.clone(), cfg).unwrap();
        queries
            .iter()
            .zip(truth.iter())
            .filter(|(q, t)| {
                index
                    .search(q, 1)
                    .is_ok_and(|hits| hits.first().map(|h| &h.id) == Some(*t))
            })
            .count()
    };

    let r1 = count_recall(1);
    let r4 = count_recall(4);
    let r16 = count_recall(16);
    assert!(r4 >= r1, "recall dropped from 1→4 trees: {r1} then {r4}");
    assert!(r16 >= r4, "recall dropped from 4→16 trees: {r4} then {r16}");
    assert!(r16 > r1, "16 trees should beat 1 tree: {r1} vs {r16}");
}

// ── Temp-dir persistence smoke test (house convention) ────────────────────────

#[test]
fn index_survives_serialised_roundtrip_via_tempdir() {
    // The index is not persisted directly, but exercising a temp-dir path keeps
    // the module aligned with the house test convention and verifies that a
    // rebuilt index over the same points reproduces identical search results.
    let dir = std::env::temp_dir().join("rp_tree_index_roundtrip");
    let _ = std::fs::create_dir_all(&dir);
    let manifest = dir.join("points.txt");

    use std::fmt::Write as _;
    let points = corpus("rt", 30, 8);
    let mut serialised = String::new();
    for (id, v) in &points {
        let coords: Vec<String> = v.iter().map(std::string::ToString::to_string).collect();
        writeln!(serialised, "{id}\t{}", coords.join(",")).unwrap();
    }
    std::fs::write(&manifest, &serialised).unwrap();

    let read_back = std::fs::read_to_string(&manifest).unwrap();
    let reloaded: Vec<(String, Vec<f32>)> = read_back
        .lines()
        .map(|line| {
            let mut parts = line.split('\t');
            let id = parts.next().unwrap().to_string();
            let coords = parts
                .next()
                .unwrap()
                .split(',')
                .map(|s| s.parse::<f32>().unwrap())
                .collect();
            (id, coords)
        })
        .collect();

    let cfg = RpTreeConfig::new()
        .with_dim(8)
        .with_n_trees(4)
        .with_leaf_size(2)
        .with_search_k(200);
    let a = RpTreeIndex::build(points, cfg.clone()).unwrap();
    let b = RpTreeIndex::build(reloaded, cfg).unwrap();

    let query = embed("rt-query", 8);
    let ha: Vec<String> = a
        .search(&query, 5)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    let hb: Vec<String> = b
        .search(&query, 5)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(ha, hb);

    let _ = std::fs::remove_dir_all(&dir);
}
