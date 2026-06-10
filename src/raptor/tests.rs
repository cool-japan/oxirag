//! Tests for the `raptor` module.

use super::cluster::{agglomerative, embed};
use super::tree::RaptorBuilder;
use super::types::{ClusterStrategy, RaptorConfig, RaptorError};

// ── Embedding tests ───────────────────────────────────────────────────────────

#[test]
fn test_embed_returns_correct_dim() {
    let emb = embed("hello world foo bar", 128);
    assert_eq!(emb.len(), 128);
}

#[test]
fn test_embed_l2_normalised() {
    let emb = embed("some text here", 64);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5 || norm < 1e-5,
        "embedding should be unit-normalised or zero: norm={norm}"
    );
}

#[test]
fn test_embed_empty_text_zero_vec() {
    let emb = embed("", 64);
    assert_eq!(emb.len(), 64);
    assert!(emb.iter().all(|x| *x == 0.0));
}

#[test]
fn test_embed_same_text_same_vector() {
    let a = embed("rust programming language", 128);
    let b = embed("rust programming language", 128);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-10);
    }
}

#[test]
fn test_embed_different_texts_different_vectors() {
    let a = embed("rust programming", 128);
    let b = embed("python data science", 128);
    let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    assert!(
        diff > 0.0,
        "different texts should produce different embeddings"
    );
}

// ── Clustering tests ──────────────────────────────────────────────────────────

#[test]
fn test_agglomerative_empty() {
    let clusters = agglomerative(&[], 3);
    assert!(clusters.is_empty());
}

#[test]
fn test_agglomerative_single_item() {
    let emb = vec![embed("hello", 32)];
    let clusters = agglomerative(&emb, 2);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0], vec![0]);
}

#[test]
fn test_agglomerative_all_items_assigned() {
    let texts = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let embs: Vec<_> = texts.iter().map(|t| embed(t, 64)).collect();
    let clusters = agglomerative(&embs, 2);
    let total: usize = clusters.iter().map(Vec::len).sum();
    assert_eq!(total, 5, "all items should be assigned");
}

#[test]
fn test_agglomerative_cluster_size_limit() {
    let texts: Vec<_> = (0..10).map(|i| format!("text {i}")).collect();
    let embs: Vec<_> = texts.iter().map(|t| embed(t, 32)).collect();
    let clusters = agglomerative(&embs, 3);
    // With target_size=3, should produce roughly ceil(10/3)=4 clusters
    assert!(!clusters.is_empty());
    let total: usize = clusters.iter().map(Vec::len).sum();
    assert_eq!(total, 10);
}

// ── ClusterStrategy tests ─────────────────────────────────────────────────────

#[test]
fn test_cluster_strategy_as_str() {
    assert_eq!(ClusterStrategy::Agglomerative.as_str(), "agglomerative");
    assert_eq!(ClusterStrategy::KMeansLite.as_str(), "kmeans_lite");
}

#[test]
fn test_cluster_strategy_default() {
    assert_eq!(ClusterStrategy::default(), ClusterStrategy::Agglomerative);
}

// ── RaptorConfig tests ────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = RaptorConfig::default();
    assert_eq!(cfg.cluster_size, 4);
    assert_eq!(cfg.max_levels, 4);
    assert_eq!(cfg.dim, 256);
    assert_eq!(cfg.summary_sentences, 3);
    assert_eq!(cfg.cluster_strategy, ClusterStrategy::Agglomerative);
}

#[test]
fn test_config_builders() {
    let cfg = RaptorConfig::default()
        .with_cluster_size(2)
        .with_max_levels(3)
        .with_dim(128)
        .with_summary_sentences(2)
        .with_cluster_strategy(ClusterStrategy::KMeansLite);
    assert_eq!(cfg.cluster_size, 2);
    assert_eq!(cfg.max_levels, 3);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.summary_sentences, 2);
    assert_eq!(cfg.cluster_strategy, ClusterStrategy::KMeansLite);
}

// ── RaptorBuilder tests ───────────────────────────────────────────────────────

#[test]
fn test_builder_empty_input_error() {
    let builder = RaptorBuilder::new();
    let cfg = RaptorConfig::default();
    let err = builder.build(&[], &cfg).expect_err("should fail");
    assert!(matches!(err, RaptorError::EmptyInput));
}

#[test]
fn test_builder_single_text() {
    let builder = RaptorBuilder::new();
    let cfg = RaptorConfig::default().with_dim(64);
    let texts = vec!["Rust is a systems programming language.".to_string()];
    let tree = builder.build(&texts, &cfg).expect("ok");
    assert_eq!(tree.len(), 1, "single text → single leaf node");
    assert_eq!(tree.nodes[0].level, 0);
    assert!(tree.nodes[0].is_leaf());
}

#[test]
fn test_builder_two_texts_single_level() {
    let builder = RaptorBuilder::new();
    let cfg = RaptorConfig::default().with_dim(64).with_cluster_size(2);
    let texts = vec![
        "Rust is fast.".to_string(),
        "Python is flexible.".to_string(),
    ];
    let tree = builder.build(&texts, &cfg).expect("ok");
    assert!(tree.len() >= 2, "should have at least the leaf nodes");
}

#[test]
fn test_builder_multi_level_tree() {
    let builder = RaptorBuilder::new();
    let cfg = RaptorConfig::default()
        .with_dim(64)
        .with_cluster_size(2)
        .with_max_levels(4);
    let texts: Vec<_> = (0..8)
        .map(|i| format!("Document {i}: Rust is a safe fast language."))
        .collect();
    let tree = builder.build(&texts, &cfg).expect("ok");
    assert!(!tree.is_empty());
    assert!(tree.levels.contains(&0), "should have leaf level");
}

#[test]
fn test_builder_all_items_appear_in_leaves() {
    let builder = RaptorBuilder::new();
    let cfg = RaptorConfig::default().with_dim(64);
    let texts: Vec<_> = (0..5).map(|i| format!("text {i}")).collect();
    let tree = builder.build(&texts, &cfg).expect("ok");
    let leaves: Vec<_> = tree.nodes_at_level(0);
    assert_eq!(
        leaves.len(),
        5,
        "all input texts should appear as leaf nodes"
    );
}

#[test]
fn test_collapsed_retrieval_top_k() {
    let builder = RaptorBuilder::new();
    let cfg = RaptorConfig::default().with_dim(64);
    let texts: Vec<_> = (0..6)
        .map(|i| format!("chunk {i} about Rust safety"))
        .collect();
    let tree = builder.build(&texts, &cfg).expect("ok");
    let query_emb = super::cluster::embed("Rust safety", 64);
    let top = tree.collapsed_retrieval(&query_emb, 3);
    assert!(top.len() <= 3, "should return at most top_k nodes");
    assert!(!top.is_empty());
}

#[test]
fn test_tree_nodes_at_level() {
    let builder = RaptorBuilder::new();
    let cfg = RaptorConfig::default().with_dim(64);
    let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let tree = builder.build(&texts, &cfg).expect("ok");
    let level_0 = tree.nodes_at_level(0);
    assert_eq!(level_0.len(), 3);
}

#[test]
fn test_raptor_error_display() {
    let e = RaptorError::EmptyInput;
    assert!(!e.to_string().is_empty());
}
