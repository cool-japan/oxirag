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
    clippy::items_after_statements,
    clippy::needless_for_each,
    clippy::uninlined_format_args,
    clippy::single_char_pattern
)]

use super::graph::VamanaGraph;
use super::index::DiskAnnIndex;
use super::types::{DiskAnnConfig, DiskAnnError, DiskAnnHit, DiskAnnMetric};

// ── Test helpers ──────────────────────────────────────────────────────────────

/// FNV-1a pseudo-embedding: deterministic, no `rand` dependency.
///
/// Components are *not* pre-normalised — `DiskAnnMetric::Cosine` normalises
/// internally, so this lets tests exercise that behaviour directly.
fn fnv_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    let bytes = text.as_bytes();
    for (i, slot) in v.iter_mut().enumerate() {
        let mut h: u64 = 14_695_981_039_346_656_037;
        for &b in bytes {
            h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
        }
        for b in (i as u64).to_le_bytes() {
            h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
        }
        *slot = ((h >> 32) as f32) / u32::MAX as f32 * 2.0 - 1.0;
    }
    v
}

/// Unit-normalised FNV-1a pseudo-embedding, for tests that need a metric-
/// independent guarantee that the self vector is the unique global optimum
/// (relevant for the raw `Dot` metric, where un-normalised magnitude can
/// otherwise make a *different* vector score higher than the true self
/// match).
fn fnv_embed_unit(text: &str, dim: usize) -> Vec<f32> {
    let mut v = fnv_embed(text, dim);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Build an index of `n` docs (labels `"doc_0"` … `"doc_{n-1}"`), dim 16,
/// with generous `R`/`L` so recall on self-queries is essentially exact.
fn build_corpus(n: usize, metric: DiskAnnMetric) -> DiskAnnIndex {
    let embed = if metric == DiskAnnMetric::Dot {
        fnv_embed_unit
    } else {
        fnv_embed
    };
    let items: Vec<(String, Vec<f32>)> = (0..n)
        .map(|i| {
            let label = format!("doc_{i}");
            let vector = embed(&label, 16);
            (label, vector)
        })
        .collect();
    let config = DiskAnnConfig::new()
        .with_max_degree(16)
        .with_search_list_size(64)
        .with_alpha(1.2)
        .with_metric(metric);
    DiskAnnIndex::build(items, config).unwrap()
}

// ── DiskAnnConfig tests ───────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let c = DiskAnnConfig::default();
    assert_eq!(c.max_degree, 32);
    assert_eq!(c.search_list_size, 64);
    assert!((c.alpha - 1.2).abs() < 1e-6);
    assert_eq!(c.metric, DiskAnnMetric::Cosine);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(DiskAnnConfig::new(), DiskAnnConfig::default());
}

#[test]
fn config_builder_max_degree() {
    assert_eq!(DiskAnnConfig::new().with_max_degree(8).max_degree, 8);
}

#[test]
fn config_builder_search_list_size() {
    assert_eq!(
        DiskAnnConfig::new()
            .with_search_list_size(100)
            .search_list_size,
        100
    );
}

#[test]
fn config_builder_alpha() {
    let c = DiskAnnConfig::new().with_alpha(1.5);
    assert!((c.alpha - 1.5).abs() < 1e-6);
}

#[test]
fn config_builder_metric() {
    let c = DiskAnnConfig::new().with_metric(DiskAnnMetric::L2);
    assert_eq!(c.metric, DiskAnnMetric::L2);
}

#[test]
fn config_builder_chain() {
    let c = DiskAnnConfig::new()
        .with_max_degree(10)
        .with_search_list_size(40)
        .with_alpha(1.3)
        .with_metric(DiskAnnMetric::Dot);
    assert_eq!(c.max_degree, 10);
    assert_eq!(c.search_list_size, 40);
    assert!((c.alpha - 1.3).abs() < 1e-6);
    assert_eq!(c.metric, DiskAnnMetric::Dot);
}

#[test]
fn config_validate_ok() {
    assert!(DiskAnnConfig::new().validate().is_ok());
}

#[test]
fn config_validate_zero_max_degree() {
    let err = DiskAnnConfig::new()
        .with_max_degree(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, DiskAnnError::InvalidConfig(_)));
}

#[test]
fn config_validate_zero_search_list_size() {
    let err = DiskAnnConfig::new()
        .with_search_list_size(0)
        .validate()
        .unwrap_err();
    assert!(matches!(err, DiskAnnError::InvalidConfig(_)));
}

#[test]
fn config_validate_alpha_below_one() {
    let err = DiskAnnConfig::new().with_alpha(0.5).validate().unwrap_err();
    assert!(matches!(err, DiskAnnError::InvalidConfig(_)));
}

#[test]
fn config_validate_alpha_nan() {
    let err = DiskAnnConfig::new()
        .with_alpha(f32::NAN)
        .validate()
        .unwrap_err();
    assert!(matches!(err, DiskAnnError::InvalidConfig(_)));
}

#[test]
fn config_validate_alpha_infinite() {
    let err = DiskAnnConfig::new()
        .with_alpha(f32::INFINITY)
        .validate()
        .unwrap_err();
    assert!(matches!(err, DiskAnnError::InvalidConfig(_)));
}

#[test]
fn config_validate_alpha_exactly_one_is_ok() {
    assert!(DiskAnnConfig::new().with_alpha(1.0).validate().is_ok());
}

#[test]
fn config_clone_and_debug() {
    let c = DiskAnnConfig::new().with_max_degree(4);
    let c2 = c;
    assert_eq!(c, c2);
    assert!(!format!("{c:?}").is_empty());
}

// ── DiskAnnMetric tests ───────────────────────────────────────────────────────

#[test]
fn metric_default_is_cosine() {
    assert_eq!(DiskAnnMetric::default(), DiskAnnMetric::Cosine);
}

#[test]
fn cosine_identical_vectors() {
    let a = vec![1.0_f32, 2.0, 3.0];
    assert!((DiskAnnMetric::Cosine.distance(&a, &a)).abs() < 1e-5);
    assert!((DiskAnnMetric::Cosine.score(&a, &a) - 1.0).abs() < 1e-5);
}

#[test]
fn cosine_orthogonal_vectors() {
    let a = vec![1.0_f32, 0.0];
    let b = vec![0.0_f32, 1.0];
    assert!((DiskAnnMetric::Cosine.score(&a, &b)).abs() < 1e-6);
    assert!((DiskAnnMetric::Cosine.distance(&a, &b) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_opposite_vectors() {
    let a = vec![1.0_f32, 0.0];
    let b = vec![-1.0_f32, 0.0];
    assert!((DiskAnnMetric::Cosine.score(&a, &b) - (-1.0)).abs() < 1e-6);
    assert!((DiskAnnMetric::Cosine.distance(&a, &b) - 2.0).abs() < 1e-6);
}

#[test]
fn cosine_zero_vector_returns_zero_score() {
    let a = vec![0.0_f32, 0.0];
    let b = vec![1.0_f32, 1.0];
    assert_eq!(DiskAnnMetric::Cosine.score(&a, &b), 0.0);
}

#[test]
fn l2_identical_vectors_zero_distance() {
    let a = vec![5.0_f32, -3.0, 2.0];
    assert!(DiskAnnMetric::L2.distance(&a, &a).abs() < 1e-5);
    assert!(DiskAnnMetric::L2.score(&a, &a).abs() < 1e-5);
}

#[test]
fn l2_known_distance_3_4_5_triangle() {
    let a = vec![0.0_f32, 0.0];
    let b = vec![3.0_f32, 4.0];
    assert!((DiskAnnMetric::L2.distance(&a, &b) - 5.0).abs() < 1e-5);
    assert!((DiskAnnMetric::L2.score(&a, &b) - (-5.0)).abs() < 1e-5);
}

#[test]
fn dot_known_value() {
    let a = vec![1.0_f32, 2.0];
    let b = vec![3.0_f32, 4.0];
    assert!((DiskAnnMetric::Dot.score(&a, &b) - 11.0).abs() < 1e-5);
}

#[test]
fn dot_distance_is_negative_dot_product() {
    let a = vec![1.0_f32, 2.0];
    let b = vec![3.0_f32, 4.0];
    assert!((DiskAnnMetric::Dot.distance(&a, &b) - (-11.0)).abs() < 1e-5);
}

#[test]
fn metric_debug_and_eq() {
    assert_eq!(DiskAnnMetric::Cosine, DiskAnnMetric::Cosine);
    assert_ne!(DiskAnnMetric::Cosine, DiskAnnMetric::L2);
    assert!(!format!("{:?}", DiskAnnMetric::Dot).is_empty());
}

// ── DiskAnnHit tests ──────────────────────────────────────────────────────────

#[test]
fn hit_new_constructor() {
    let h = DiskAnnHit::new("test_id", 0.75_f32);
    assert_eq!(h.id, "test_id");
    assert!((h.score - 0.75).abs() < 1e-6);
}

#[test]
fn hit_clone_and_debug() {
    let h = DiskAnnHit::new("a", 1.0);
    let h2 = h.clone();
    assert_eq!(h, h2);
    assert!(!format!("{h:?}").is_empty());
}

// ── DiskAnnError tests ────────────────────────────────────────────────────────

#[test]
fn error_display_empty_index() {
    assert!(!DiskAnnError::EmptyIndex.to_string().is_empty());
}

#[test]
fn error_display_dimension_mismatch() {
    let e = DiskAnnError::DimensionMismatch {
        expected: 16,
        got: 8,
    };
    let s = e.to_string();
    assert!(s.contains("16") && s.contains('8'), "display: {s}");
}

#[test]
fn error_display_invalid_config() {
    let s = DiskAnnError::InvalidConfig("bad alpha".into()).to_string();
    assert!(s.contains("bad alpha"));
}

#[test]
fn error_display_empty_query() {
    assert!(!DiskAnnError::EmptyQuery.to_string().is_empty());
}

#[test]
fn error_eq_and_clone() {
    let e1 = DiskAnnError::EmptyIndex;
    let e2 = e1.clone();
    assert_eq!(e1, e2);
    assert_ne!(DiskAnnError::EmptyIndex, DiskAnnError::EmptyQuery);
}

// ── Build error paths ─────────────────────────────────────────────────────────

#[test]
fn build_empty_items_returns_empty_index_error() {
    let err = DiskAnnIndex::build(Vec::new(), DiskAnnConfig::new()).unwrap_err();
    assert_eq!(err, DiskAnnError::EmptyIndex);
}

#[test]
fn build_dim_mismatch_returns_error() {
    let items = vec![
        ("a".to_string(), vec![1.0, 2.0, 3.0]),
        ("b".to_string(), vec![1.0, 2.0]),
    ];
    let err = DiskAnnIndex::build(items, DiskAnnConfig::new()).unwrap_err();
    assert!(matches!(
        err,
        DiskAnnError::DimensionMismatch {
            expected: 3,
            got: 2
        }
    ));
}

#[test]
fn build_zero_dim_vector_returns_invalid_config() {
    let items = vec![("a".to_string(), vec![])];
    let err = DiskAnnIndex::build(items, DiskAnnConfig::new()).unwrap_err();
    assert!(matches!(err, DiskAnnError::InvalidConfig(_)));
}

#[test]
fn build_invalid_config_max_degree_zero_propagates() {
    let items = vec![("a".to_string(), vec![1.0, 2.0])];
    let cfg = DiskAnnConfig::new().with_max_degree(0);
    let err = DiskAnnIndex::build(items, cfg).unwrap_err();
    assert!(matches!(err, DiskAnnError::InvalidConfig(_)));
}

#[test]
fn build_invalid_config_alpha_too_small_propagates() {
    let items = vec![("a".to_string(), vec![1.0, 2.0])];
    let cfg = DiskAnnConfig::new().with_alpha(0.1);
    let err = DiskAnnIndex::build(items, cfg).unwrap_err();
    assert!(matches!(err, DiskAnnError::InvalidConfig(_)));
}

#[test]
fn build_single_item_succeeds() {
    let items = vec![("only".to_string(), fnv_embed("only", 8))];
    let idx = DiskAnnIndex::build(items, DiskAnnConfig::new().with_max_degree(4)).unwrap();
    assert_eq!(idx.len(), 1);
    assert!(!idx.is_empty());
    assert_eq!(idx.dim(), 8);
}

// ── Index accessors ───────────────────────────────────────────────────────────

#[test]
fn len_matches_item_count() {
    let idx = build_corpus(7, DiskAnnMetric::Cosine);
    assert_eq!(idx.len(), 7);
}

#[test]
fn is_empty_false_after_build() {
    let idx = build_corpus(3, DiskAnnMetric::Cosine);
    assert!(!idx.is_empty());
}

#[test]
fn dim_matches_vector_length() {
    let idx = build_corpus(4, DiskAnnMetric::Cosine);
    assert_eq!(idx.dim(), 16);
}

#[test]
fn contains_existing_id() {
    let idx = build_corpus(5, DiskAnnMetric::Cosine);
    assert!(idx.contains("doc_0"));
    assert!(idx.contains("doc_4"));
}

#[test]
fn contains_missing_id() {
    let idx = build_corpus(5, DiskAnnMetric::Cosine);
    assert!(!idx.contains("doc_99"));
    assert!(!idx.contains(""));
}

#[test]
fn config_accessor_returns_built_config() {
    let items = vec![("a".to_string(), vec![1.0, 2.0])];
    let cfg = DiskAnnConfig::new().with_max_degree(4).with_alpha(1.4);
    let idx = DiskAnnIndex::build(items, cfg).unwrap();
    assert_eq!(idx.config().max_degree, 4);
    assert!((idx.config().alpha - 1.4).abs() < 1e-6);
}

#[test]
fn graph_accessor_len_matches_index_len() {
    let idx = build_corpus(9, DiskAnnMetric::Cosine);
    assert_eq!(idx.graph().len(), idx.len());
}

#[test]
fn index_clone_and_debug() {
    let idx = build_corpus(5, DiskAnnMetric::Cosine);
    let idx2 = idx.clone();
    assert_eq!(idx.len(), idx2.len());
    assert!(!format!("{idx:?}").is_empty());
}

// ── Search error paths ────────────────────────────────────────────────────────

#[test]
fn search_on_empty_index_returns_error() {
    let idx = DiskAnnIndex::empty_for_test(DiskAnnConfig::new(), 4);
    let err = idx.search(&[0.0, 0.0, 0.0, 0.0], 1).unwrap_err();
    assert_eq!(err, DiskAnnError::EmptyIndex);
}

#[test]
fn search_empty_query_returns_error() {
    let idx = build_corpus(4, DiskAnnMetric::Cosine);
    let err = idx.search(&[], 1).unwrap_err();
    assert_eq!(err, DiskAnnError::EmptyQuery);
}

#[test]
fn search_dim_mismatch_returns_error() {
    let idx = build_corpus(4, DiskAnnMetric::Cosine);
    let err = idx.search(&fnv_embed("q", 8), 1).unwrap_err();
    assert!(matches!(
        err,
        DiskAnnError::DimensionMismatch {
            expected: 16,
            got: 8
        }
    ));
}

#[test]
fn search_k_zero_returns_empty_vec_not_error() {
    let idx = build_corpus(4, DiskAnnMetric::Cosine);
    let hits = idx.search(&fnv_embed("doc_0", 16), 0).unwrap();
    assert!(hits.is_empty());
}

// ── Build & search: recall on self-queries ────────────────────────────────────

#[test]
fn exact_query_top_hit_matches_inserted_doc_cosine() {
    let idx = build_corpus(10, DiskAnnMetric::Cosine);
    let q = fnv_embed("doc_3", 16);
    let hits = idx.search(&q, 1).unwrap();
    assert_eq!(hits[0].id, "doc_3");
}

#[test]
fn exact_query_top_hit_matches_inserted_doc_l2() {
    let idx = build_corpus(10, DiskAnnMetric::L2);
    let q = fnv_embed("doc_3", 16);
    let hits = idx.search(&q, 1).unwrap();
    assert_eq!(hits[0].id, "doc_3");
}

#[test]
fn exact_query_top_hit_matches_inserted_doc_dot() {
    let idx = build_corpus(10, DiskAnnMetric::Dot);
    let q = fnv_embed_unit("doc_3", 16);
    let hits = idx.search(&q, 1).unwrap();
    assert_eq!(hits[0].id, "doc_3");
}

#[test]
fn top_hit_score_is_one_for_self_query_cosine() {
    let idx = build_corpus(5, DiskAnnMetric::Cosine);
    for i in 0..5_usize {
        let label = format!("doc_{i}");
        let q = fnv_embed(&label, 16);
        let hits = idx.search(&q, 1).unwrap();
        assert_eq!(hits[0].id, label, "self-query failed for {label}");
        assert!(
            (hits[0].score - 1.0_f32).abs() < 1e-4,
            "score not ~1.0 for {label}: {}",
            hits[0].score
        );
    }
}

#[test]
fn top_hit_score_is_near_zero_l2_for_self_query() {
    let idx = build_corpus(5, DiskAnnMetric::L2);
    for i in 0..5_usize {
        let label = format!("doc_{i}");
        let q = fnv_embed(&label, 16);
        let hits = idx.search(&q, 1).unwrap();
        assert_eq!(hits[0].id, label, "self-query failed for {label}");
        assert!(
            hits[0].score.abs() < 1e-4,
            "score not ~0.0 for {label}: {}",
            hits[0].score
        );
    }
}

#[test]
fn scores_descending_order_cosine() {
    let idx = build_corpus(20, DiskAnnMetric::Cosine);
    let hits = idx.search(&fnv_embed("doc_0", 16), 10).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score, "scores out of order: {:?}", hits);
    }
}

#[test]
fn scores_descending_order_l2() {
    let idx = build_corpus(20, DiskAnnMetric::L2);
    let hits = idx.search(&fnv_embed("doc_0", 16), 10).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score, "scores out of order: {:?}", hits);
    }
}

#[test]
fn scores_descending_order_dot() {
    let idx = build_corpus(20, DiskAnnMetric::Dot);
    let hits = idx.search(&fnv_embed_unit("doc_0", 16), 10).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score, "scores out of order: {:?}", hits);
    }
}

#[test]
fn k_larger_than_corpus_returns_all_docs() {
    let n = 6;
    let idx = build_corpus(n, DiskAnnMetric::Cosine);
    let hits = idx.search(&fnv_embed("doc_0", 16), 100).unwrap();
    assert_eq!(hits.len(), n);
}

#[test]
fn k_equals_corpus_size() {
    let n = 8;
    let idx = build_corpus(n, DiskAnnMetric::Cosine);
    let hits = idx.search(&fnv_embed("doc_2", 16), n).unwrap();
    assert_eq!(hits.len(), n);
}

#[test]
fn no_duplicate_ids_in_results() {
    let idx = build_corpus(30, DiskAnnMetric::Cosine);
    let hits = idx.search(&fnv_embed("doc_0", 16), 15).unwrap();
    let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), hits.len(), "duplicate ids in results");
}

// ── Large corpus (exercises the approximate-medoid path) ──────────────────────

#[test]
fn large_corpus_builds_without_error() {
    let idx = build_corpus(300, DiskAnnMetric::Cosine);
    assert_eq!(idx.len(), 300);
}

#[test]
fn large_corpus_self_query_top_hit() {
    let idx = build_corpus(300, DiskAnnMetric::Cosine);
    for i in [0_usize, 50, 150, 260, 299] {
        let label = format!("doc_{i}");
        let hits = idx.search(&fnv_embed(&label, 16), 1).unwrap();
        assert_eq!(hits[0].id, label, "self-query failed for {label}");
    }
}

// ── Various dimensions ─────────────────────────────────────────────────────────

#[test]
fn works_with_dim_1() {
    // Use L2 explicitly: with the default Cosine metric a zero vector's
    // similarity is ill-defined (falls back to 0.0), which would make this
    // fixture's absolute-distance intuition misleading.
    let items = vec![
        ("a".to_string(), vec![0.0_f32]),
        ("b".to_string(), vec![10.0_f32]),
    ];
    let cfg = DiskAnnConfig::new()
        .with_max_degree(2)
        .with_metric(DiskAnnMetric::L2);
    let idx = DiskAnnIndex::build(items, cfg).unwrap();
    let hits = idx.search(&[0.5_f32], 1).unwrap();
    assert_eq!(hits[0].id, "a");
}

#[test]
fn works_with_dim_2() {
    let cfg = DiskAnnConfig::new()
        .with_max_degree(4)
        .with_search_list_size(8);
    let mut items = Vec::new();
    for i in 0..6_usize {
        items.push((format!("v{i}"), fnv_embed(&format!("v{i}"), 2)));
    }
    let idx = DiskAnnIndex::build(items, cfg).unwrap();
    let hits = idx.search(&fnv_embed("v3", 2), 1).unwrap();
    assert_eq!(hits[0].id, "v3");
}

#[test]
fn works_with_dim_128() {
    let cfg = DiskAnnConfig::new()
        .with_max_degree(16)
        .with_search_list_size(50);
    let items: Vec<(String, Vec<f32>)> = (0..10_usize)
        .map(|i| (format!("v{i}"), fnv_embed(&format!("vec_{i}"), 128)))
        .collect();
    let idx = DiskAnnIndex::build(items, cfg).unwrap();
    let hits = idx.search(&fnv_embed("vec_5", 128), 1).unwrap();
    assert_eq!(hits[0].id, "v5");
}

// ── Structural invariants ──────────────────────────────────────────────────────

#[test]
fn every_node_out_degree_at_most_max_degree() {
    let idx = build_corpus(40, DiskAnnMetric::Cosine);
    let r = idx.config().max_degree;
    for i in 0..idx.len() {
        assert!(
            idx.graph().neighbors_of(i).len() <= r,
            "node {i} exceeds max_degree {r}: {} neighbors",
            idx.graph().neighbors_of(i).len()
        );
    }
}

#[test]
fn no_self_loops_in_graph() {
    let idx = build_corpus(40, DiskAnnMetric::Cosine);
    for i in 0..idx.len() {
        assert!(
            !idx.graph().neighbors_of(i).contains(&i),
            "node {i} has a self-loop"
        );
    }
}

#[test]
fn neighbor_indices_are_in_bounds() {
    let idx = build_corpus(40, DiskAnnMetric::L2);
    let n = idx.len();
    for i in 0..n {
        for &nb in idx.graph().neighbors_of(i) {
            assert!(nb < n, "neighbor index {nb} out of bounds for n={n}");
        }
    }
}

// ── Determinism / reproducibility ─────────────────────────────────────────────

#[test]
fn build_is_deterministic_graph_structure() {
    let items: Vec<(String, Vec<f32>)> = (0..25_usize)
        .map(|i| (format!("doc_{i}"), fnv_embed(&format!("doc_{i}"), 16)))
        .collect();
    let cfg = DiskAnnConfig::new()
        .with_max_degree(8)
        .with_search_list_size(20);

    let idx_a = DiskAnnIndex::build(items.clone(), cfg).unwrap();
    let idx_b = DiskAnnIndex::build(items, cfg).unwrap();

    assert_eq!(idx_a.graph().medoid(), idx_b.graph().medoid());
    for i in 0..idx_a.len() {
        assert_eq!(
            idx_a.graph().neighbors_of(i),
            idx_b.graph().neighbors_of(i),
            "graph diverged at node {i}"
        );
    }
}

#[test]
fn build_is_deterministic_search_results() {
    let items: Vec<(String, Vec<f32>)> = (0..25_usize)
        .map(|i| (format!("doc_{i}"), fnv_embed(&format!("doc_{i}"), 16)))
        .collect();
    let cfg = DiskAnnConfig::new()
        .with_max_degree(8)
        .with_search_list_size(20);

    let idx_a = DiskAnnIndex::build(items.clone(), cfg).unwrap();
    let idx_b = DiskAnnIndex::build(items, cfg).unwrap();

    let q = fnv_embed("doc_7", 16);
    let hits_a = idx_a.search(&q, 5).unwrap();
    let hits_b = idx_b.search(&q, 5).unwrap();
    assert_eq!(hits_a, hits_b);
}

// ── Medoid computation ─────────────────────────────────────────────────────────

#[test]
fn medoid_is_the_central_point_on_a_line() {
    let items: Vec<(String, Vec<f32>)> = (0..5_usize)
        .map(|i| (format!("p{i}"), vec![i as f32]))
        .collect();
    let cfg = DiskAnnConfig::new()
        .with_max_degree(4)
        .with_search_list_size(8)
        .with_metric(DiskAnnMetric::L2);
    let idx = DiskAnnIndex::build(items, cfg).unwrap();
    assert_eq!(idx.graph().medoid(), 2, "expected the middle point p2");
}

// ── Exact small fixtures: L2 ───────────────────────────────────────────────────

#[test]
fn l2_fixture_exact_top2_ranking() {
    let items = vec![
        ("a".to_string(), vec![0.0_f32, 0.0]),
        ("b".to_string(), vec![1.0_f32, 0.0]),
        ("c".to_string(), vec![10.0_f32, 10.0]),
        ("d".to_string(), vec![11.0_f32, 10.0]),
        ("e".to_string(), vec![10.0_f32, 11.0]),
    ];
    let cfg = DiskAnnConfig::new()
        .with_max_degree(4)
        .with_search_list_size(8)
        .with_metric(DiskAnnMetric::L2);
    let idx = DiskAnnIndex::build(items, cfg).unwrap();

    let hits = idx.search(&[0.2_f32, 0.0], 2).unwrap();
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[1].id, "b");
    assert!((hits[0].score - (-0.2)).abs() < 1e-4, "{}", hits[0].score);
    assert!((hits[1].score - (-0.8)).abs() < 1e-4, "{}", hits[1].score);
}

// ── Exact small fixtures: Cosine (directional ranking) ────────────────────────

#[test]
fn cosine_fixture_exact_directional_ranking() {
    let items = vec![
        ("east".to_string(), vec![1.0_f32, 0.0]),
        ("northeast".to_string(), vec![1.0_f32, 1.0]),
        ("north".to_string(), vec![0.0_f32, 1.0]),
        ("west".to_string(), vec![-1.0_f32, 0.0]),
    ];
    let cfg = DiskAnnConfig::new()
        .with_max_degree(8)
        .with_search_list_size(8)
        .with_metric(DiskAnnMetric::Cosine);
    let idx = DiskAnnIndex::build(items, cfg).unwrap();

    let hits = idx.search(&[1.0_f32, 0.01], 4).unwrap();
    let order: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(order, vec!["east", "northeast", "north", "west"]);
}

#[test]
fn cosine_ignores_magnitude_ties() {
    let items = vec![
        ("big".to_string(), vec![5.0_f32, 0.0]),
        ("small".to_string(), vec![1.0_f32, 0.0]),
        ("orth".to_string(), vec![0.0_f32, 5.0]),
    ];
    let cfg = DiskAnnConfig::new()
        .with_max_degree(4)
        .with_search_list_size(8)
        .with_metric(DiskAnnMetric::Cosine);
    let idx = DiskAnnIndex::build(items, cfg).unwrap();

    let hits = idx.search(&[1.0_f32, 0.0], 3).unwrap();
    assert_eq!(hits[2].id, "orth");
    assert!((hits[2].score).abs() < 1e-5);
    assert!((hits[0].score - 1.0).abs() < 1e-4);
    assert!((hits[1].score - 1.0).abs() < 1e-4);
}

// ── Exact small fixtures: Dot (magnitude matters) ─────────────────────────────

#[test]
fn dot_fixture_prioritizes_magnitude() {
    let items = vec![
        ("big".to_string(), vec![5.0_f32, 0.0]),
        ("small".to_string(), vec![1.0_f32, 0.0]),
        ("orth".to_string(), vec![0.0_f32, 5.0]),
    ];
    let cfg = DiskAnnConfig::new()
        .with_max_degree(4)
        .with_search_list_size(8)
        .with_metric(DiskAnnMetric::Dot);
    let idx = DiskAnnIndex::build(items, cfg).unwrap();

    let hits = idx.search(&[1.0_f32, 0.0], 3).unwrap();
    let order: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(order, vec!["big", "small", "orth"]);
    assert!((hits[0].score - 5.0).abs() < 1e-4);
    assert!((hits[1].score - 1.0).abs() < 1e-4);
    assert!((hits[2].score).abs() < 1e-4);
}

// ── VamanaGraph direct tests ───────────────────────────────────────────────────

#[test]
fn vamana_graph_new_basic_accessors() {
    let g = VamanaGraph::new(vec![vec![1, 2], vec![0], vec![0]], 0);
    assert_eq!(g.len(), 3);
    assert!(!g.is_empty());
    assert_eq!(g.medoid(), 0);
    assert_eq!(g.neighbors_of(0), &[1, 2]);
    assert_eq!(g.neighbors_of(1), &[0]);
}

#[test]
fn vamana_graph_empty() {
    let g = VamanaGraph::new(Vec::new(), 0);
    assert_eq!(g.len(), 0);
    assert!(g.is_empty());
}

#[test]
fn vamana_graph_clone_and_debug() {
    let g = VamanaGraph::new(vec![vec![1], vec![0]], 0);
    let g2 = g.clone();
    assert_eq!(g.len(), g2.len());
    assert!(!format!("{g:?}").is_empty());
}

#[test]
fn vamana_graph_from_built_index_matches_len() {
    let idx = build_corpus(12, DiskAnnMetric::Cosine);
    assert_eq!(idx.graph().len(), 12);
    assert!(idx.graph().medoid() < 12);
}
