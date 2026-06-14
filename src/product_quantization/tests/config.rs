//! Tests for [`PqConfig`], [`PqError`], [`PqCode`] and [`PqHit`].

use super::small_config;
use crate::product_quantization::types::{PqCode, PqConfig, PqError, PqHit};
use crate::types::DocumentId;

// ── PqConfig: defaults ────────────────────────────────────────────────────────

#[test]
fn test_config_default_values() {
    let cfg = PqConfig::default();
    assert_eq!(cfg.num_subspaces, 4);
    assert_eq!(cfg.codebook_bits, 8);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.kmeans_iters, 10);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(PqConfig::new(), PqConfig::default());
}

#[test]
fn test_config_codebook_size() {
    assert_eq!(PqConfig::default().codebook_size(), 256);
    assert_eq!(PqConfig::new().with_codebook_bits(4).codebook_size(), 16);
    assert_eq!(PqConfig::new().with_codebook_bits(1).codebook_size(), 2);
}

#[test]
fn test_config_subspace_dim() {
    assert_eq!(PqConfig::default().subspace_dim(), 32);
    assert_eq!(small_config().subspace_dim(), 4);
}

#[test]
fn test_config_subspace_dim_zero_subspaces() {
    let cfg = PqConfig::new().with_num_subspaces(0);
    assert_eq!(cfg.subspace_dim(), 0);
}

// ── PqConfig: builders ────────────────────────────────────────────────────────

#[test]
fn test_config_builders_chain() {
    let cfg = PqConfig::new()
        .with_num_subspaces(8)
        .with_codebook_bits(6)
        .with_dim(64)
        .with_kmeans_iters(25);
    assert_eq!(cfg.num_subspaces, 8);
    assert_eq!(cfg.codebook_bits, 6);
    assert_eq!(cfg.dim, 64);
    assert_eq!(cfg.kmeans_iters, 25);
}

#[test]
fn test_config_builder_num_subspaces() {
    assert_eq!(PqConfig::new().with_num_subspaces(16).num_subspaces, 16);
}

#[test]
fn test_config_builder_dim() {
    assert_eq!(PqConfig::new().with_dim(512).dim, 512);
}

#[test]
fn test_config_clone_eq() {
    let cfg = small_config();
    assert_eq!(cfg.clone(), cfg);
}

// ── PqConfig: validate ────────────────────────────────────────────────────────

#[test]
fn test_validate_default_ok() {
    assert!(PqConfig::default().validate().is_ok());
}

#[test]
fn test_validate_small_ok() {
    assert!(small_config().validate().is_ok());
}

#[test]
fn test_validate_rejects_indivisible_dim() {
    let cfg = PqConfig::new().with_dim(10).with_num_subspaces(3);
    assert_eq!(
        cfg.validate(),
        Err(PqError::InvalidConfig {
            dim: 10,
            subspaces: 3
        })
    );
}

#[test]
fn test_validate_rejects_dim_not_multiple() {
    let cfg = PqConfig::new().with_dim(7).with_num_subspaces(2);
    assert!(matches!(cfg.validate(), Err(PqError::InvalidConfig { .. })));
}

#[test]
fn test_validate_rejects_zero_subspaces() {
    let cfg = PqConfig::new().with_num_subspaces(0).with_dim(8);
    assert!(matches!(cfg.validate(), Err(PqError::InvalidConfig { .. })));
}

#[test]
fn test_validate_rejects_too_many_bits() {
    let cfg = PqConfig::new().with_codebook_bits(9);
    assert_eq!(cfg.validate(), Err(PqError::DimMismatch));
}

#[test]
fn test_validate_accepts_eight_bits() {
    let cfg = PqConfig::new()
        .with_dim(16)
        .with_num_subspaces(4)
        .with_codebook_bits(8);
    assert!(cfg.validate().is_ok());
}

// ── Error Display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display_invalid_config() {
    let err = PqError::InvalidConfig {
        dim: 10,
        subspaces: 3,
    };
    assert_eq!(err.to_string(), "dim 10 not divisible by num_subspaces 3");
}

#[test]
fn test_error_display_variants() {
    assert_eq!(PqError::DimMismatch.to_string(), "vector dim mismatch");
    assert_eq!(PqError::NotTrained.to_string(), "quantizer not trained");
    assert_eq!(
        PqError::EmptyTrainingSet.to_string(),
        "training set is empty"
    );
}

// ── PqCode ────────────────────────────────────────────────────────────────────

#[test]
fn test_pqcode_new_len() {
    let code = PqCode::new(vec![1, 2, 3]);
    assert_eq!(code.len(), 3);
    assert!(!code.is_empty());
    assert_eq!(code.codes, vec![1, 2, 3]);
}

#[test]
fn test_pqcode_empty() {
    let code = PqCode::new(Vec::new());
    assert!(code.is_empty());
    assert_eq!(code.len(), 0);
}

// ── PqHit ─────────────────────────────────────────────────────────────────────

#[test]
fn test_pqhit_new() {
    let hit = PqHit::new(DocumentId::from_string("x"), 1.5);
    assert_eq!(hit.id, DocumentId::from_string("x"));
    assert_eq!(hit.distance, 1.5);
}
