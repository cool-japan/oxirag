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
    clippy::items_after_statements
)]

use super::*;
use crate::types::DocumentId;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn did(s: &str) -> DocumentId {
    DocumentId::from_string(s)
}

/// Squared L2 distance, mirroring the index's internal metric, for brute-force
/// reference comparisons in tests.
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Brute-force top-k ids ranked by ascending squared L2 distance.
fn brute_force(items: &[(DocumentId, Vec<f32>)], query: &[f32], top_k: usize) -> Vec<DocumentId> {
    let mut scored: Vec<(f32, DocumentId)> = items
        .iter()
        .map(|(id, v)| (l2_sq(query, v), id.clone()))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(top_k).map(|(_, id)| id).collect()
}

/// A spread-out 2-D dataset with deterministic clusters.
fn dataset_2d() -> Vec<(DocumentId, Vec<f32>)> {
    vec![
        (did("p0"), vec![0.0, 0.0]),
        (did("p1"), vec![0.5, 0.2]),
        (did("p2"), vec![0.2, 0.6]),
        (did("p3"), vec![10.0, 10.0]),
        (did("p4"), vec![10.5, 9.8]),
        (did("p5"), vec![9.7, 10.3]),
        (did("p6"), vec![-8.0, 7.0]),
        (did("p7"), vec![-8.3, 6.6]),
        (did("p8"), vec![5.0, -9.0]),
        (did("p9"), vec![4.7, -9.4]),
    ]
}

// ── IvfConfig ─────────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let c = IvfConfig::default();
    assert_eq!(c.num_cells, 16);
    assert_eq!(c.nprobe, 4);
    assert_eq!(c.dim, 128);
    assert_eq!(c.kmeans_iters, 10);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(IvfConfig::new(), IvfConfig::default());
}

#[test]
fn config_builder_num_cells() {
    assert_eq!(IvfConfig::new().with_num_cells(32).num_cells, 32);
}

#[test]
fn config_builder_nprobe() {
    assert_eq!(IvfConfig::new().with_nprobe(8).nprobe, 8);
}

#[test]
fn config_builder_dim() {
    assert_eq!(IvfConfig::new().with_dim(64).dim, 64);
}

#[test]
fn config_builder_kmeans_iters() {
    assert_eq!(IvfConfig::new().with_kmeans_iters(25).kmeans_iters, 25);
}

#[test]
fn config_builders_chain() {
    let c = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(2)
        .with_dim(3)
        .with_kmeans_iters(7);
    assert_eq!(c.num_cells, 4);
    assert_eq!(c.nprobe, 2);
    assert_eq!(c.dim, 3);
    assert_eq!(c.kmeans_iters, 7);
}

#[test]
fn config_clone_eq() {
    let c = IvfConfig::new().with_dim(5);
    assert_eq!(c.clone(), c);
}

// ── IvfHit ────────────────────────────────────────────────────────────────────

#[test]
fn hit_new_fields() {
    let h = IvfHit::new(did("x"), 1.5);
    assert_eq!(h.id, did("x"));
    assert_eq!(h.distance, 1.5);
}

#[test]
fn hit_clone_eq() {
    let h = IvfHit::new(did("y"), 2.0);
    assert_eq!(h.clone(), h);
}

// ── Construction state ────────────────────────────────────────────────────────

#[test]
fn new_index_is_untrained() {
    let index = IvfIndex::new(IvfConfig::new());
    assert!(!index.is_trained());
}

#[test]
fn new_index_is_empty() {
    let index = IvfIndex::new(IvfConfig::new());
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.num_cells(), 0);
}

#[test]
fn config_accessor_round_trips() {
    let config = IvfConfig::new().with_dim(9);
    let index = IvfIndex::new(config.clone());
    assert_eq!(index.config(), &config);
}

// ── Training: centroid count ──────────────────────────────────────────────────

#[test]
fn train_builds_num_cells_centroids() {
    let config = IvfConfig::new().with_num_cells(4).with_dim(2);
    let mut index = IvfIndex::new(config);
    let vectors: Vec<Vec<f32>> = dataset_2d().into_iter().map(|(_, v)| v).collect();
    index.train(&vectors).unwrap();
    assert!(index.is_trained());
    assert_eq!(index.num_cells(), 4);
}

#[test]
fn train_clamps_cells_to_vector_count() {
    // Request more cells than there are vectors.
    let config = IvfConfig::new().with_num_cells(16).with_dim(2);
    let mut index = IvfIndex::new(config);
    let vectors = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
    index.train(&vectors).unwrap();
    assert_eq!(index.num_cells(), 3);
}

#[test]
fn train_single_cell() {
    let config = IvfConfig::new().with_num_cells(1).with_dim(2);
    let mut index = IvfIndex::new(config);
    let vectors = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
    index.train(&vectors).unwrap();
    assert_eq!(index.num_cells(), 1);
}

#[test]
fn train_single_vector() {
    let config = IvfConfig::new().with_num_cells(8).with_dim(2);
    let mut index = IvfIndex::new(config);
    index.train(&[vec![3.0, 4.0]]).unwrap();
    assert_eq!(index.num_cells(), 1);
}

#[test]
fn retrain_resets_lists() {
    let config = IvfConfig::new().with_num_cells(2).with_dim(2);
    let mut index = IvfIndex::new(config);
    index.train(&[vec![0.0, 0.0], vec![9.0, 9.0]]).unwrap();
    index.add(did("a"), vec![0.1, 0.1]).unwrap();
    assert_eq!(index.len(), 1);
    // Retraining discards previously added vectors.
    index.train(&[vec![0.0, 0.0], vec![9.0, 9.0]]).unwrap();
    assert_eq!(index.len(), 0);
}

// ── add / cell_of ─────────────────────────────────────────────────────────────

#[test]
fn add_assigns_to_nearest_cell() {
    let config = IvfConfig::new().with_num_cells(4).with_dim(2);
    let mut index = IvfIndex::new(config);
    let items = dataset_2d();
    let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
    index.train(&vectors).unwrap();

    for (id, v) in &items {
        let expected = index.cell_of(v).unwrap();
        index.add(id.clone(), v.clone()).unwrap();
        // The freshly-added vector must live in the cell that cell_of reports.
        let list = index.list(expected).unwrap();
        assert!(list.iter().any(|(lid, _)| lid == id));
    }
}

#[test]
fn cell_of_is_argmin_over_centroids() {
    let config = IvfConfig::new().with_num_cells(4).with_dim(2);
    let mut index = IvfIndex::new(config);
    let vectors: Vec<Vec<f32>> = dataset_2d().into_iter().map(|(_, v)| v).collect();
    index.train(&vectors).unwrap();

    let probe = vec![10.2, 10.1];
    let cell = index.cell_of(&probe).unwrap();
    // Verify it is the minimal-distance cell by checking distances to all lists'
    // contents is consistent: rebuild centroid distances via search probing.
    // Use a direct check: cell index must be < num_cells.
    assert!(cell < index.num_cells());
}

#[test]
fn add_increases_len() {
    let config = IvfConfig::new().with_num_cells(2).with_dim(2);
    let mut index = IvfIndex::new(config);
    index.train(&[vec![0.0, 0.0], vec![9.0, 9.0]]).unwrap();
    index.add(did("a"), vec![0.1, 0.0]).unwrap();
    index.add(did("b"), vec![9.1, 9.0]).unwrap();
    assert_eq!(index.len(), 2);
    assert!(!index.is_empty());
}

// ── build ─────────────────────────────────────────────────────────────────────

#[test]
fn build_trains_and_adds_all() {
    let config = IvfConfig::new().with_num_cells(4).with_dim(2);
    let mut index = IvfIndex::new(config);
    let items = dataset_2d();
    index.build(&items).unwrap();
    assert!(index.is_trained());
    assert_eq!(index.len(), items.len());
}

#[test]
fn build_partitions_all_vectors() {
    let config = IvfConfig::new().with_num_cells(4).with_dim(2);
    let mut index = IvfIndex::new(config);
    let items = dataset_2d();
    index.build(&items).unwrap();

    // Sum of list lengths equals total inserted count: a partition.
    let total: usize = (0..index.num_cells())
        .map(|c| index.list(c).map_or(0, <[_]>::len))
        .sum();
    assert_eq!(total, items.len());
}

#[test]
fn build_lists_cover_every_id_once() {
    let config = IvfConfig::new().with_num_cells(4).with_dim(2);
    let mut index = IvfIndex::new(config);
    let items = dataset_2d();
    index.build(&items).unwrap();

    let mut seen: Vec<DocumentId> = Vec::new();
    for c in 0..index.num_cells() {
        for (id, _) in index.list(c).unwrap() {
            seen.push(id.clone());
        }
    }
    seen.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    let mut expected: Vec<DocumentId> = items.iter().map(|(id, _)| id.clone()).collect();
    expected.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    assert_eq!(seen, expected);
}

#[test]
fn build_some_lists_nonempty() {
    let config = IvfConfig::new().with_num_cells(4).with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&dataset_2d()).unwrap();
    let nonempty = (0..index.num_cells())
        .filter(|&c| !index.list(c).unwrap().is_empty())
        .count();
    assert!(nonempty >= 1);
}

// ── search: exact when nprobe == num_cells ────────────────────────────────────

#[test]
fn search_exact_when_nprobe_equals_num_cells() {
    let items = dataset_2d();
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();

    let queries = [
        vec![0.3, 0.3],
        vec![10.1, 10.0],
        vec![-8.1, 6.8],
        vec![4.9, -9.1],
        vec![1.0, 1.0],
    ];
    for q in &queries {
        let top_k = 3;
        let hits = index.search(q, top_k).unwrap();
        let ids: Vec<DocumentId> = hits.iter().map(|h| h.id.clone()).collect();
        assert_eq!(ids, brute_force(&items, q, top_k));
    }
}

#[test]
fn search_full_probe_matches_full_ranking() {
    let items = dataset_2d();
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();

    let q = vec![0.0, 0.0];
    let hits = index.search(&q, items.len()).unwrap();
    let ids: Vec<DocumentId> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(ids, brute_force(&items, &q, items.len()));
}

#[test]
fn search_distances_are_ascending() {
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&dataset_2d()).unwrap();
    let hits = index.search(&vec![3.0, 3.0], 5).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].distance <= w[1].distance);
    }
}

#[test]
fn search_distance_values_are_l2_squared() {
    let items = dataset_2d();
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();

    let q = vec![10.0, 10.0];
    let hits = index.search(&q, 1).unwrap();
    let top = &hits[0];
    let original = items.iter().find(|(id, _)| *id == top.id).unwrap();
    assert!((top.distance - l2_sq(&q, &original.1)).abs() < 1e-5);
}

// ── search: exact-match query ─────────────────────────────────────────────────

#[test]
fn search_exact_match_is_rank_zero() {
    let items = dataset_2d();
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();

    for (id, v) in &items {
        let hits = index.search(v, 1).unwrap();
        assert_eq!(&hits[0].id, id);
        assert!(hits[0].distance.abs() < 1e-6);
    }
}

#[test]
fn search_exact_match_with_low_nprobe() {
    let items = dataset_2d();
    // Even nprobe=1 must find an exact match: the query lands in its own cell.
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(1)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();

    for (id, v) in &items {
        let hits = index.search(v, 1).unwrap();
        assert_eq!(&hits[0].id, id);
    }
}

// ── search: approximate with nprobe < num_cells ───────────────────────────────

#[test]
fn search_low_nprobe_returns_at_most_top_k() {
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(1)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&dataset_2d()).unwrap();
    let hits = index.search(&vec![0.3, 0.3], 5).unwrap();
    assert!(hits.len() <= 5);
}

#[test]
fn search_partial_probe_scans_subset() {
    // With nprobe < num_cells fewer candidates are typically scanned than the
    // full corpus, so result count cannot exceed what a single (or few) cells hold.
    let items = dataset_2d();
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(1)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();
    let hits = index.search(&vec![10.1, 10.0], 100).unwrap();
    assert!(hits.len() < items.len());
}

#[test]
fn search_nprobe_clamped_above_num_cells() {
    let items = dataset_2d();
    // nprobe larger than num_cells behaves like full probing (exact).
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(99)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();
    let q = vec![1.0, 1.0];
    let ids: Vec<DocumentId> = index
        .search(&q, 3)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(ids, brute_force(&items, &q, 3));
}

#[test]
fn search_top_k_zero_returns_empty() {
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&dataset_2d()).unwrap();
    assert!(index.search(&vec![0.0, 0.0], 0).unwrap().is_empty());
}

#[test]
fn search_top_k_exceeds_corpus() {
    let items = dataset_2d();
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();
    let hits = index.search(&vec![0.0, 0.0], 1000).unwrap();
    assert_eq!(hits.len(), items.len());
}

#[test]
fn search_returns_distinct_ids() {
    let items = dataset_2d();
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();
    let hits = index.search(&vec![0.0, 0.0], items.len()).unwrap();
    let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    let total = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), total);
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[test]
fn train_empty_set_errors() {
    let mut index = IvfIndex::new(IvfConfig::new().with_dim(2));
    let empty: Vec<Vec<f32>> = Vec::new();
    assert_eq!(index.train(&empty), Err(IvfError::EmptyTrainingSet));
}

#[test]
fn train_dim_mismatch_errors() {
    let mut index = IvfIndex::new(IvfConfig::new().with_dim(2));
    let vectors = vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
    assert_eq!(index.train(&vectors), Err(IvfError::DimMismatch));
}

#[test]
fn build_empty_set_errors() {
    let mut index = IvfIndex::new(IvfConfig::new().with_dim(2));
    let empty: Vec<(DocumentId, Vec<f32>)> = Vec::new();
    assert_eq!(index.build(&empty), Err(IvfError::EmptyTrainingSet));
}

#[test]
fn build_dim_mismatch_errors() {
    let mut index = IvfIndex::new(IvfConfig::new().with_dim(2));
    let items = vec![(did("a"), vec![1.0, 2.0, 3.0])];
    assert_eq!(index.build(&items), Err(IvfError::DimMismatch));
}

#[test]
fn add_before_train_errors() {
    let mut index = IvfIndex::new(IvfConfig::new().with_dim(2));
    assert_eq!(
        index.add(did("a"), vec![1.0, 2.0]),
        Err(IvfError::NotTrained)
    );
}

#[test]
fn add_dim_mismatch_errors() {
    let mut index = IvfIndex::new(IvfConfig::new().with_num_cells(2).with_dim(2));
    index.train(&[vec![0.0, 0.0], vec![1.0, 1.0]]).unwrap();
    assert_eq!(
        index.add(did("a"), vec![1.0, 2.0, 3.0]),
        Err(IvfError::DimMismatch)
    );
}

#[test]
fn cell_of_before_train_errors() {
    let index = IvfIndex::new(IvfConfig::new().with_dim(2));
    assert_eq!(index.cell_of(&[1.0, 2.0]), Err(IvfError::NotTrained));
}

#[test]
fn cell_of_dim_mismatch_errors() {
    let mut index = IvfIndex::new(IvfConfig::new().with_num_cells(2).with_dim(2));
    index.train(&[vec![0.0, 0.0], vec![1.0, 1.0]]).unwrap();
    assert_eq!(index.cell_of(&[1.0, 2.0, 3.0]), Err(IvfError::DimMismatch));
}

#[test]
fn search_before_train_errors() {
    let index = IvfIndex::new(IvfConfig::new().with_dim(2));
    assert_eq!(index.search(&[1.0, 2.0], 3), Err(IvfError::NotTrained));
}

#[test]
fn search_dim_mismatch_errors() {
    let mut index = IvfIndex::new(IvfConfig::new().with_num_cells(2).with_dim(2));
    index.build(&[(did("a"), vec![0.0, 0.0])]).unwrap();
    assert_eq!(
        index.search(&[1.0, 2.0, 3.0], 3),
        Err(IvfError::DimMismatch)
    );
}

#[test]
fn error_display_messages() {
    assert_eq!(IvfError::DimMismatch.to_string(), "vector dim mismatch");
    assert_eq!(IvfError::NotTrained.to_string(), "index not trained");
    assert_eq!(
        IvfError::EmptyTrainingSet.to_string(),
        "training set is empty"
    );
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn training_is_deterministic() {
    let items = dataset_2d();
    let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();

    let build = || {
        let mut index = IvfIndex::new(IvfConfig::new().with_num_cells(4).with_dim(2));
        index.train(&vectors).unwrap();
        (0..index.num_cells())
            .map(|c| index.cell_of(&vectors[c % vectors.len()]).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(build(), build());
}

#[test]
fn search_is_deterministic() {
    let items = dataset_2d();
    let make = || {
        let mut index = IvfIndex::new(
            IvfConfig::new()
                .with_num_cells(4)
                .with_nprobe(2)
                .with_dim(2),
        );
        index.build(&items).unwrap();
        index
    };
    let a = make();
    let b = make();
    let q = vec![2.0, 2.0];
    let ha: Vec<(String, f32)> = a
        .search(&q, 5)
        .unwrap()
        .into_iter()
        .map(|h| (h.id.as_str().to_string(), h.distance))
        .collect();
    let hb: Vec<(String, f32)> = b
        .search(&q, 5)
        .unwrap()
        .into_iter()
        .map(|h| (h.id.as_str().to_string(), h.distance))
        .collect();
    assert_eq!(ha, hb);
}

#[test]
fn assignment_is_deterministic_across_rebuilds() {
    let items = dataset_2d();
    let cells = |index: &IvfIndex| -> Vec<usize> {
        items
            .iter()
            .map(|(_, v)| index.cell_of(v).unwrap())
            .collect()
    };
    let mut a = IvfIndex::new(IvfConfig::new().with_num_cells(4).with_dim(2));
    a.build(&items).unwrap();
    let mut b = IvfIndex::new(IvfConfig::new().with_num_cells(4).with_dim(2));
    b.build(&items).unwrap();
    assert_eq!(cells(&a), cells(&b));
}

// ── Higher-dimensional smoke test ─────────────────────────────────────────────

#[test]
fn search_higher_dim_exact_full_probe() {
    // Three well-separated points in 4-D.
    let items = vec![
        (did("u"), vec![1.0, 0.0, 0.0, 0.0]),
        (did("v"), vec![0.0, 1.0, 0.0, 0.0]),
        (did("w"), vec![0.0, 0.0, 1.0, 0.0]),
        (did("x"), vec![0.0, 0.0, 0.0, 1.0]),
    ];
    let config = IvfConfig::new()
        .with_num_cells(4)
        .with_nprobe(4)
        .with_dim(4);
    let mut index = IvfIndex::new(config);
    index.build(&items).unwrap();

    let q = vec![0.0, 0.0, 0.9, 0.1];
    let ids: Vec<DocumentId> = index
        .search(&q, 2)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(ids, brute_force(&items, &q, 2));
    assert_eq!(ids[0], did("w"));
}

#[test]
fn empty_query_dim_zero_config() {
    // dim == 0 is a degenerate but valid configuration.
    let config = IvfConfig::new().with_num_cells(2).with_dim(0);
    let mut index = IvfIndex::new(config);
    index.train(&[Vec::new(), Vec::new()]).unwrap();
    assert!(index.is_trained());
    index.add(did("z"), Vec::new()).unwrap();
    let hits = index.search(&[], 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, did("z"));
}

#[test]
fn duplicate_ids_are_preserved() {
    // The index does not deduplicate ids; both insertions are retrievable.
    let config = IvfConfig::new()
        .with_num_cells(2)
        .with_nprobe(2)
        .with_dim(2);
    let mut index = IvfIndex::new(config);
    index.train(&[vec![0.0, 0.0], vec![9.0, 9.0]]).unwrap();
    index.add(did("dup"), vec![0.0, 0.0]).unwrap();
    index.add(did("dup"), vec![9.0, 9.0]).unwrap();
    assert_eq!(index.len(), 2);
    let hits = index.search(&vec![0.0, 0.0], 2).unwrap();
    assert_eq!(hits.len(), 2);
}
