//! Tests for [`PqIndex`]: build and ADC search.

use super::{make_vectors, small_config};
use crate::product_quantization::index::PqIndex;
use crate::product_quantization::types::{PqConfig, PqError};
use crate::types::DocumentId;

/// Items as `(DocumentId, vector)` pairs with stable ids `doc-{i}`.
fn make_items(count: usize, dim: usize) -> Vec<(DocumentId, Vec<f32>)> {
    make_vectors(count, dim)
        .into_iter()
        .enumerate()
        .map(|(i, v)| (DocumentId::from_string(format!("doc-{i}")), v))
        .collect()
}

// ── Construction & metadata ───────────────────────────────────────────────────

#[test]
fn test_index_new_empty() {
    let index = PqIndex::new(small_config());
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert!(!index.quantizer().is_trained());
}

#[test]
fn test_index_build_len() {
    let mut index = PqIndex::new(small_config());
    let items = make_items(20, 8);
    index.build(&items).expect("build should succeed");
    assert_eq!(index.len(), 20);
    assert!(!index.is_empty());
    assert!(index.quantizer().is_trained());
}

#[test]
fn test_index_build_trains_quantizer() {
    let mut index = PqIndex::new(small_config());
    let items = make_items(30, 8);
    index.build(&items).expect("build should succeed");
    assert_eq!(index.quantizer().codebooks().len(), 2);
}

#[test]
fn test_index_rebuild_replaces() {
    let mut index = PqIndex::new(small_config());
    index
        .build(&make_items(20, 8))
        .expect("build should succeed");
    assert_eq!(index.len(), 20);
    index
        .build(&make_items(12, 8))
        .expect("rebuild should succeed");
    assert_eq!(index.len(), 12);
}

// ── Build errors ──────────────────────────────────────────────────────────────

#[test]
fn test_index_build_empty_errors() {
    let mut index = PqIndex::new(small_config());
    assert_eq!(index.build(&[]).unwrap_err(), PqError::EmptyTrainingSet);
}

#[test]
fn test_index_build_invalid_config_errors() {
    let cfg = PqConfig::new().with_dim(10).with_num_subspaces(3);
    let mut index = PqIndex::new(cfg);
    let items = vec![(DocumentId::from_string("a"), vec![0.0; 10]); 5];
    assert!(matches!(
        index.build(&items),
        Err(PqError::InvalidConfig { .. })
    ));
}

#[test]
fn test_index_build_dim_mismatch_errors() {
    let mut index = PqIndex::new(small_config());
    let items = vec![(DocumentId::from_string("a"), vec![0.0; 6]); 5];
    assert_eq!(index.build(&items).unwrap_err(), PqError::DimMismatch);
}

// ── Search ────────────────────────────────────────────────────────────────────

#[test]
fn test_search_before_build_errors() {
    let index = PqIndex::new(small_config());
    assert_eq!(
        index.search(&vec![0.0; 8], 3).unwrap_err(),
        PqError::NotTrained
    );
}

#[test]
fn test_search_dim_mismatch_errors() {
    let mut index = PqIndex::new(small_config());
    index
        .build(&make_items(20, 8))
        .expect("build should succeed");
    assert_eq!(
        index.search(&vec![0.0; 6], 3).unwrap_err(),
        PqError::DimMismatch
    );
}

#[test]
fn test_search_respects_top_k() {
    let mut index = PqIndex::new(small_config());
    index
        .build(&make_items(20, 8))
        .expect("build should succeed");
    let hits = index
        .search(&make_vectors(1, 8)[0], 5)
        .expect("search should succeed");
    assert_eq!(hits.len(), 5);
}

#[test]
fn test_search_top_k_clamped_to_len() {
    let mut index = PqIndex::new(small_config());
    index
        .build(&make_items(7, 8))
        .expect("build should succeed");
    let hits = index
        .search(&make_vectors(1, 8)[0], 100)
        .expect("search should succeed");
    assert_eq!(hits.len(), 7);
}

#[test]
fn test_search_ascending_distance() {
    let mut index = PqIndex::new(small_config());
    index
        .build(&make_items(20, 8))
        .expect("build should succeed");
    let hits = index
        .search(&make_vectors(20, 8)[10], 10)
        .expect("search should succeed");
    for w in hits.windows(2) {
        assert!(
            w[0].distance <= w[1].distance,
            "hits not sorted ascending: {} > {}",
            w[0].distance,
            w[1].distance
        );
    }
}

#[test]
fn test_search_exact_query_rank_zero() {
    // An exact copy of a training vector should rank that vector at position 0.
    // Use one item per distinct cluster so the target is unambiguously nearest
    // (each item encodes to a unique codeword).
    let mut index = PqIndex::new(small_config());
    let items: Vec<(DocumentId, Vec<f32>)> = [0.0f32, 20.0, 40.0, 60.0]
        .iter()
        .enumerate()
        .map(|(i, &base)| (DocumentId::from_string(format!("c-{i}")), vec![base; 8]))
        .collect();
    index.build(&items).expect("build should succeed");
    let target_idx = 2usize;
    let query = items[target_idx].1.clone();
    let hits = index.search(&query, 1).expect("search should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].id, items[target_idx].0,
        "nearest hit was not the exact training vector"
    );
}

#[test]
fn test_search_exact_query_distance_small() {
    let mut index = PqIndex::new(small_config());
    let items = make_items(24, 8);
    index.build(&items).expect("build should succeed");
    let query = items[3].1.clone();
    let hits = index.search(&query, 1).expect("search should succeed");
    // Rank-0 distance is the (tiny) quantization residual of the exact vector.
    assert!(hits[0].distance < 1.0, "rank-0 distance too large");
}

#[test]
fn test_search_distinct_clusters_rank_zero() {
    // Build over two well-separated clusters; query near one returns its member.
    let cfg = PqConfig::new()
        .with_dim(4)
        .with_num_subspaces(2)
        .with_codebook_bits(1);
    let mut index = PqIndex::new(cfg);
    let items = vec![
        (DocumentId::from_string("low"), vec![0.0, 0.0, 0.0, 0.0]),
        (
            DocumentId::from_string("high"),
            vec![50.0, 50.0, 50.0, 50.0],
        ),
    ];
    index.build(&items).expect("build should succeed");
    let hits = index
        .search(&vec![49.0, 49.0, 49.0, 49.0], 2)
        .expect("search should succeed");
    assert_eq!(hits[0].id, DocumentId::from_string("high"));
}

#[test]
fn test_search_deterministic() {
    let mut index = PqIndex::new(small_config());
    let items = make_items(20, 8);
    index.build(&items).expect("build should succeed");
    let query = make_vectors(20, 8)[7].clone();
    let a = index.search(&query, 5).expect("search should succeed");
    let b = index.search(&query, 5).expect("search should succeed");
    assert_eq!(a, b);
}
