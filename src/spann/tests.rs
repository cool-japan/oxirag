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
    clippy::no_effect_underscore_binding
)]

//! Tests for the `spann` module.

use super::SpannIndex;
use super::cluster::{balanced_kmeans, boundary_replicas, metric_distance, metric_score};
use super::types::{Posting, SpannConfig, SpannError, SpannHit, SpannMetric, SpannStats};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Find the group index (position in `members`) containing global index `gi`.
fn group_of(members: &[Vec<usize>], gi: usize) -> usize {
    members
        .iter()
        .position(|g| g.contains(&gi))
        .expect("global index must belong to exactly one group")
}

/// Brute-force top-k ids ranked by descending `metric_score`, ties broken by
/// ascending id, mirroring `SpannIndex::search`'s own ranking convention.
fn brute_force_ids(
    items: &[(String, Vec<f32>)],
    query: &[f32],
    metric: SpannMetric,
    k: usize,
) -> Vec<String> {
    let mut scored: Vec<(f32, String)> = items
        .iter()
        .map(|(id, v)| (metric_score(query, v, metric), id.clone()))
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn origin_and_far_fixture() -> Vec<(String, Vec<f32>)> {
    vec![
        ("o0".to_string(), vec![0.0, 0.0]),
        ("o1".to_string(), vec![0.0, 1.0]),
        ("o2".to_string(), vec![1.0, 0.0]),
        ("f0".to_string(), vec![10.0, 10.0]),
        ("f1".to_string(), vec![10.0, 11.0]),
        ("f2".to_string(), vec![11.0, 10.0]),
    ]
}

// ── SpannMetric ───────────────────────────────────────────────────────────────

#[test]
fn metric_default_is_cosine() {
    assert_eq!(SpannMetric::default(), SpannMetric::Cosine);
}

#[test]
fn metric_variants_distinct() {
    assert_ne!(SpannMetric::Cosine, SpannMetric::L2);
    assert_ne!(SpannMetric::L2, SpannMetric::Dot);
    assert_ne!(SpannMetric::Cosine, SpannMetric::Dot);
}

// ── SpannConfig ───────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let c = SpannConfig::default();
    assert_eq!(c.num_postings, 16);
    assert_eq!(c.replica_count, 8);
    assert_eq!(c.boundary_epsilon, 0.1);
    assert_eq!(c.posting_limit, 100);
    assert_eq!(c.nprobe, 8);
    assert_eq!(c.max_iters, 25);
    assert_eq!(c.metric, SpannMetric::Cosine);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SpannConfig::new(), SpannConfig::default());
}

#[test]
fn config_with_num_postings() {
    assert_eq!(SpannConfig::new().with_num_postings(4).num_postings, 4);
}

#[test]
fn config_with_replica_count() {
    assert_eq!(SpannConfig::new().with_replica_count(3).replica_count, 3);
}

#[test]
fn config_with_boundary_epsilon() {
    assert_eq!(
        SpannConfig::new()
            .with_boundary_epsilon(0.25)
            .boundary_epsilon,
        0.25
    );
}

#[test]
fn config_with_posting_limit() {
    assert_eq!(SpannConfig::new().with_posting_limit(5).posting_limit, 5);
}

#[test]
fn config_with_nprobe() {
    assert_eq!(SpannConfig::new().with_nprobe(2).nprobe, 2);
}

#[test]
fn config_with_max_iters() {
    assert_eq!(SpannConfig::new().with_max_iters(50).max_iters, 50);
}

#[test]
fn config_with_metric() {
    assert_eq!(
        SpannConfig::new().with_metric(SpannMetric::L2).metric,
        SpannMetric::L2
    );
}

#[test]
fn config_validate_ok_default() {
    assert!(SpannConfig::default().validate().is_ok());
}

#[test]
fn config_validate_err_num_postings_zero() {
    let c = SpannConfig::new().with_num_postings(0);
    assert!(matches!(c.validate(), Err(SpannError::InvalidConfig(_))));
}

#[test]
fn config_validate_err_replica_count_zero() {
    let c = SpannConfig::new().with_replica_count(0);
    assert!(matches!(c.validate(), Err(SpannError::InvalidConfig(_))));
}

#[test]
fn config_validate_err_posting_limit_zero() {
    let c = SpannConfig::new().with_posting_limit(0);
    assert!(matches!(c.validate(), Err(SpannError::InvalidConfig(_))));
}

#[test]
fn config_validate_err_nprobe_zero() {
    let c = SpannConfig::new().with_nprobe(0);
    assert!(matches!(c.validate(), Err(SpannError::InvalidConfig(_))));
}

#[test]
fn config_validate_err_max_iters_zero() {
    let c = SpannConfig::new().with_max_iters(0);
    assert!(matches!(c.validate(), Err(SpannError::InvalidConfig(_))));
}

#[test]
fn config_validate_err_epsilon_negative() {
    let c = SpannConfig::new().with_boundary_epsilon(-0.1);
    assert!(matches!(c.validate(), Err(SpannError::InvalidConfig(_))));
}

#[test]
fn config_validate_err_epsilon_nan() {
    let c = SpannConfig::new().with_boundary_epsilon(f32::NAN);
    assert!(matches!(c.validate(), Err(SpannError::InvalidConfig(_))));
}

#[test]
fn config_validate_err_epsilon_infinite() {
    let c = SpannConfig::new().with_boundary_epsilon(f32::INFINITY);
    assert!(matches!(c.validate(), Err(SpannError::InvalidConfig(_))));
}

// ── Posting / SpannHit / SpannStats / SpannError ─────────────────────────────

#[test]
fn posting_fields_accessible() {
    let p = Posting {
        centroid: vec![1.0, 2.0],
        member_ids: vec!["a".to_string()],
    };
    assert_eq!(p.centroid, vec![1.0, 2.0]);
    assert_eq!(p.member_ids, vec!["a".to_string()]);
}

#[test]
fn spann_hit_fields_accessible() {
    let h = SpannHit {
        id: "x".to_string(),
        score: 0.5,
    };
    assert_eq!(h.id, "x");
    assert_eq!(h.score, 0.5);
}

#[test]
fn spann_stats_fields_accessible() {
    let s = SpannStats {
        num_postings: 3,
        total_replicas: 10,
        avg_posting_len: 3.333,
        max_posting_len: 5,
        min_posting_len: 2,
    };
    assert_eq!(s.num_postings, 3);
    assert_eq!(s.total_replicas, 10);
    assert_eq!(s.max_posting_len, 5);
    assert_eq!(s.min_posting_len, 2);
}

#[test]
fn error_display_empty_index() {
    assert_eq!(SpannError::EmptyIndex.to_string(), "index is empty");
}

#[test]
fn error_display_dimension_mismatch() {
    let e = SpannError::DimensionMismatch {
        expected: 3,
        got: 5,
    };
    assert_eq!(
        e.to_string(),
        "vector dimension mismatch: expected 3, got 5"
    );
}

#[test]
fn error_display_invalid_config() {
    let e = SpannError::InvalidConfig("bad".to_string());
    assert_eq!(e.to_string(), "invalid configuration: bad");
}

#[test]
fn error_display_empty_query() {
    assert_eq!(SpannError::EmptyQuery.to_string(), "query is empty");
}

// ── metric_distance / metric_score ────────────────────────────────────────────

#[test]
fn metric_distance_cosine_identical_is_zero() {
    let a = vec![1.0, 2.0, 3.0];
    assert!(metric_distance(&a, &a, SpannMetric::Cosine).abs() < 1e-5);
}

#[test]
fn metric_distance_cosine_orthogonal_is_one() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert!((metric_distance(&a, &b, SpannMetric::Cosine) - 1.0).abs() < 1e-5);
}

#[test]
fn metric_distance_l2_matches_euclidean() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    assert!((metric_distance(&a, &b, SpannMetric::L2) - 5.0).abs() < 1e-5);
}

#[test]
fn metric_distance_dot_is_negated_dot_product() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    // dot = 1*3 + 2*4 = 11
    assert!((metric_distance(&a, &b, SpannMetric::Dot) - (-11.0)).abs() < 1e-5);
}

#[test]
fn metric_score_cosine_identical_is_one() {
    let a = vec![1.0, 2.0, 3.0];
    assert!((metric_score(&a, &a, SpannMetric::Cosine) - 1.0).abs() < 1e-5);
}

#[test]
fn metric_score_l2_bounded_in_zero_one() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let s = metric_score(&a, &b, SpannMetric::L2);
    assert!(s > 0.0 && s < 1.0);
    assert!((s - 1.0 / 6.0).abs() < 1e-5);
}

#[test]
fn metric_score_dot_matches_raw_dot_product() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    assert!((metric_score(&a, &b, SpannMetric::Dot) - 11.0).abs() < 1e-5);
}

#[test]
fn metric_distance_zero_vector_cosine_is_safe() {
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 2.0];
    // Cosine similarity of a zero vector is defined as 0.0, so distance is 1.0.
    assert!((metric_distance(&a, &b, SpannMetric::Cosine) - 1.0).abs() < 1e-5);
}

// ── balanced_kmeans ────────────────────────────────────────────────────────────

#[test]
fn balanced_kmeans_empty_vectors_returns_empty() {
    let result = balanced_kmeans(&[], 4, 100, 25, SpannMetric::L2);
    assert!(result.centroids.is_empty());
    assert!(result.members.is_empty());
}

#[test]
fn balanced_kmeans_basic_two_clusters_separate_correctly() {
    let items = origin_and_far_fixture();
    let vecs: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
    let result = balanced_kmeans(&vecs, 2, 100, 25, SpannMetric::L2);

    assert_eq!(result.centroids.len(), 2);
    // The three "origin" points (0, 1, 2) share a group...
    let g0 = group_of(&result.members, 0);
    assert_eq!(group_of(&result.members, 1), g0);
    assert_eq!(group_of(&result.members, 2), g0);
    // ...and the three "far" points (3, 4, 5) share a different group.
    let g3 = group_of(&result.members, 3);
    assert_eq!(group_of(&result.members, 4), g3);
    assert_eq!(group_of(&result.members, 5), g3);
    assert_ne!(g0, g3);
}

#[test]
fn balanced_kmeans_clamps_num_postings_to_n() {
    let vecs = vec![vec![0.0, 0.0], vec![5.0, 5.0], vec![9.0, 1.0]];
    let result = balanced_kmeans(&vecs, 10, 100, 25, SpannMetric::L2);
    assert_eq!(result.centroids.len(), 3);
    for group in &result.members {
        assert_eq!(group.len(), 1);
    }
}

#[test]
fn balanced_kmeans_no_split_needed_under_limit() {
    let items = origin_and_far_fixture();
    let vecs: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
    let result = balanced_kmeans(&vecs, 2, 100, 25, SpannMetric::L2);
    assert_eq!(result.centroids.len(), 2);
}

#[test]
fn balanced_kmeans_splits_oversized_cluster() {
    let items = origin_and_far_fixture();
    let vecs: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
    // Force everything into one initial cluster, but posting_limit=3 requires
    // the 6-member cluster to be split.
    let result = balanced_kmeans(&vecs, 1, 3, 25, SpannMetric::L2);

    assert_eq!(result.centroids.len(), 2);
    for group in &result.members {
        assert!(group.len() <= 3);
    }
    // The natural sub-structure (origin cluster vs far cluster) should still
    // be recovered by the forced split.
    let g0 = group_of(&result.members, 0);
    assert_eq!(group_of(&result.members, 1), g0);
    assert_eq!(group_of(&result.members, 2), g0);
    let g3 = group_of(&result.members, 3);
    assert_eq!(group_of(&result.members, 4), g3);
    assert_eq!(group_of(&result.members, 5), g3);
    assert_ne!(g0, g3);
}

#[test]
fn balanced_kmeans_recursive_split_keeps_all_lists_under_limit() {
    // Four distinct, well-separated micro-clusters of three points each,
    // forced into a single initial posting with a tight posting_limit.
    let mut vecs = Vec::new();
    for &(cx, cy) in &[(0.0, 0.0), (50.0, 0.0), (0.0, 50.0), (50.0, 50.0)] {
        vecs.push(vec![cx, cy]);
        vecs.push(vec![cx + 1.0, cy]);
        vecs.push(vec![cx, cy + 1.0]);
    }
    let result = balanced_kmeans(&vecs, 1, 4, 25, SpannMetric::L2);
    for group in &result.members {
        assert!(group.len() <= 4, "oversized group: {group:?}");
    }
    assert_eq!(
        result.members.iter().map(Vec::len).sum::<usize>(),
        vecs.len()
    );
}

#[test]
fn balanced_kmeans_identical_points_cannot_split_below_limit() {
    let vecs = vec![vec![1.0, 1.0]; 5];
    let result = balanced_kmeans(&vecs, 1, 2, 25, SpannMetric::L2);
    // k-means cannot separate identical points, so the oversized cluster is
    // honestly left as-is rather than looping forever or fabricating a split.
    let total: usize = result.members.iter().map(Vec::len).sum();
    assert_eq!(total, 5);
}

#[test]
fn balanced_kmeans_deterministic_repeated_calls() {
    let items = origin_and_far_fixture();
    let vecs: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
    let r1 = balanced_kmeans(&vecs, 2, 100, 25, SpannMetric::L2);
    let r2 = balanced_kmeans(&vecs, 2, 100, 25, SpannMetric::L2);
    assert_eq!(r1.centroids, r2.centroids);
    assert_eq!(r1.members, r2.members);
}

#[test]
fn balanced_kmeans_all_members_accounted_for() {
    let items = origin_and_far_fixture();
    let vecs: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
    let result = balanced_kmeans(&vecs, 3, 100, 25, SpannMetric::Cosine);
    let total: usize = result.members.iter().map(Vec::len).sum();
    assert_eq!(total, vecs.len());
}

// ── boundary_replicas / RNG pruning ────────────────────────────────────────────

#[test]
fn boundary_replicas_empty_centroids_returns_empty() {
    let accepted = boundary_replicas(&[1.0, 2.0], &[], 0.1, 8, SpannMetric::L2);
    assert!(accepted.is_empty());
}

#[test]
fn boundary_replicas_nearest_always_included() {
    let point = vec![0.0, 0.0];
    let centroids = vec![vec![1.0, 0.0], vec![100.0, 100.0]];
    let accepted = boundary_replicas(&point, &centroids, 0.0, 8, SpannMetric::L2);
    assert!(accepted.contains(&0));
}

#[test]
fn boundary_replicas_zero_epsilon_excludes_non_tied_neighbours() {
    let point = vec![0.0, 0.0];
    let centroids = vec![vec![1.0, 0.0], vec![2.0, 0.0]];
    let accepted = boundary_replicas(&point, &centroids, 0.0, 8, SpannMetric::L2);
    // With epsilon = 0, only centroids at *exactly* the nearest distance
    // qualify; centroid 1 (farther) must not be included.
    assert_eq!(accepted, vec![0]);
}

#[test]
fn boundary_replicas_large_epsilon_includes_multiple() {
    // c0 and c1 sit in different directions from `point` (neither dominates
    // the other under the RNG rule), while c2 is far enough to stay outside
    // even a generous epsilon window.
    let point = vec![0.0, 0.0];
    let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.05], vec![100.0, 0.0]];
    let accepted = boundary_replicas(&point, &centroids, 1.0, 8, SpannMetric::L2);
    assert!(accepted.contains(&0));
    assert!(accepted.len() >= 2);
    assert!(!accepted.contains(&2));
}

#[test]
fn boundary_replicas_capped_at_replica_count() {
    let point = vec![0.0, 0.0];
    // Five centroids all within a huge epsilon window, but replica_count
    // caps how many raw candidates are even considered before RNG pruning.
    let centroids = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.1],
        vec![1.0, 0.2],
        vec![1.0, 0.3],
        vec![1.0, 0.4],
    ];
    let accepted = boundary_replicas(&point, &centroids, 1000.0, 2, SpannMetric::L2);
    assert!(accepted.len() <= 2);
}

#[test]
fn boundary_replicas_rng_prunes_dominated_candidate() {
    // 1-D layout on the x-axis: point at 0, centroids at 1, 2 and 10.
    let point = vec![0.0];
    let centroids = vec![vec![1.0], vec![2.0], vec![10.0]];
    // Huge epsilon so every centroid initially qualifies as a boundary
    // candidate; replica_count large enough not to cap anything.
    let accepted = boundary_replicas(&point, &centroids, 1000.0, 8, SpannMetric::L2);
    // c1 (dist 2.0) is dominated by c0 (dist(c0,c1)=1.0 < dist(point,c1)=2.0).
    // c2 (dist 10.0) is dominated by c0 (dist(c0,c2)=9.0 < dist(point,c2)=10.0).
    assert_eq!(accepted, vec![0]);
}

#[test]
fn boundary_replicas_rng_keeps_non_dominated_candidate() {
    // Candidate centroid lies in a direction where the nearest centroid is
    // far from it relative to the point, so it survives the RNG rule.
    let point = vec![0.0, 0.0];
    let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.05]];
    let accepted = boundary_replicas(&point, &centroids, 1000.0, 8, SpannMetric::L2);
    assert_eq!(accepted.len(), 2);
    assert!(accepted.contains(&0));
    assert!(accepted.contains(&1));
}

#[test]
fn boundary_replicas_single_centroid() {
    let point = vec![0.0, 0.0];
    let centroids = vec![vec![5.0, 5.0]];
    let accepted = boundary_replicas(&point, &centroids, 0.5, 8, SpannMetric::L2);
    assert_eq!(accepted, vec![0]);
}

// ── SpannIndex::build ──────────────────────────────────────────────────────────

#[test]
fn build_empty_items_errors() {
    let err = SpannIndex::build(Vec::new(), SpannConfig::default()).unwrap_err();
    assert_eq!(err, SpannError::EmptyIndex);
}

#[test]
fn build_dimension_mismatch_errors() {
    let items = vec![
        ("a".to_string(), vec![1.0, 0.0]),
        ("b".to_string(), vec![1.0, 0.0, 0.0]),
    ];
    let err = SpannIndex::build(items, SpannConfig::default()).unwrap_err();
    assert_eq!(
        err,
        SpannError::DimensionMismatch {
            expected: 2,
            got: 3
        }
    );
}

#[test]
fn build_invalid_config_propagates() {
    let items = vec![("a".to_string(), vec![1.0])];
    let config = SpannConfig::new().with_num_postings(0);
    let err = SpannIndex::build(items, config).unwrap_err();
    assert!(matches!(err, SpannError::InvalidConfig(_)));
}

#[test]
fn build_basic_len_dim_num_postings() {
    let items = origin_and_far_fixture();
    let config = SpannConfig::new().with_num_postings(2).with_nprobe(2);
    let index = SpannIndex::build(items, config).unwrap();
    assert_eq!(index.len(), 6);
    assert_eq!(index.dim(), 2);
    assert_eq!(index.num_postings(), 2);
    assert!(!index.is_empty());
}

#[test]
fn build_num_postings_clamped_when_greater_than_n() {
    let items = vec![
        ("a".to_string(), vec![0.0, 0.0]),
        ("b".to_string(), vec![1.0, 1.0]),
    ];
    let config = SpannConfig::new().with_num_postings(50).with_nprobe(50);
    let index = SpannIndex::build(items, config).unwrap();
    assert_eq!(index.num_postings(), 2);
}

#[test]
fn build_stats_total_at_least_len() {
    let items = origin_and_far_fixture();
    let config = SpannConfig::new().with_num_postings(2).with_nprobe(2);
    let index = SpannIndex::build(items, config).unwrap();
    let stats = index.stats();
    assert_eq!(stats.num_postings, 2);
    assert!(stats.total_replicas >= index.len());
    assert!(stats.max_posting_len >= stats.min_posting_len);
    assert!((stats.avg_posting_len - stats.total_replicas as f32 / 2.0).abs() < 1e-5);
}

#[test]
fn build_single_point() {
    let items = vec![("only".to_string(), vec![1.0, 2.0, 3.0])];
    let index = SpannIndex::build(items, SpannConfig::default()).unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(index.num_postings(), 1);
    let hits = index.search(&[1.0, 2.0, 3.0], 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "only");
}

#[test]
fn build_duplicate_ids_last_write_wins() {
    let items = vec![
        ("x".to_string(), vec![0.0, 0.0]),
        ("x".to_string(), vec![5.0, 5.0]),
    ];
    let config = SpannConfig::new().with_num_postings(1).with_nprobe(1);
    let index = SpannIndex::build(items, config).unwrap();
    assert_eq!(index.len(), 1);
    let hits = index.search(&[5.0, 5.0], 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "x");
    // The stored vector is the last occurrence, so scoring against [5,5]
    // should yield a near-perfect match.
    assert!(hits[0].score > 0.99);
}

#[test]
fn build_config_accessor_roundtrip() {
    let items = vec![("a".to_string(), vec![1.0, 2.0])];
    let config = SpannConfig::new().with_num_postings(1).with_nprobe(1);
    let index = SpannIndex::build(items, config.clone()).unwrap();
    assert_eq!(index.config(), &config);
}

#[test]
fn build_postings_accessor_lengths_consistent() {
    let items = origin_and_far_fixture();
    let config = SpannConfig::new().with_num_postings(2).with_nprobe(2);
    let index = SpannIndex::build(items, config).unwrap();
    assert_eq!(index.postings().len(), index.num_postings());
    for posting in index.postings() {
        assert_eq!(posting.centroid.len(), index.dim());
    }
}

// ── SpannIndex::search ────────────────────────────────────────────────────────

#[test]
fn search_exact_recall_cosine_full_probe() {
    let items = origin_and_far_fixture();
    let n = items.len();
    let config = SpannConfig::new()
        .with_num_postings(3)
        .with_nprobe(n) // full scan: guarantees exact recall
        .with_metric(SpannMetric::Cosine);
    let index = SpannIndex::build(items.clone(), config).unwrap();

    let query = vec![0.0, 0.9];
    let expected = brute_force_ids(&items, &query, SpannMetric::Cosine, 3);
    let got: Vec<String> = index
        .search(&query, 3)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn search_exact_recall_l2_full_probe() {
    let items = origin_and_far_fixture();
    let n = items.len();
    let config = SpannConfig::new()
        .with_num_postings(3)
        .with_nprobe(n)
        .with_metric(SpannMetric::L2);
    let index = SpannIndex::build(items.clone(), config).unwrap();

    let query = vec![10.5, 10.5];
    let expected = brute_force_ids(&items, &query, SpannMetric::L2, 2);
    let got: Vec<String> = index
        .search(&query, 2)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(got, expected);
    // [10.5, 10.5] is exactly equidistant from f0/f1/f2 under L2, so the
    // ascending-id tie-break must pick "f0" first.
    assert_eq!(got[0], "f0".to_string());
}

#[test]
fn search_exact_recall_dot_full_probe() {
    let items = origin_and_far_fixture();
    let n = items.len();
    let config = SpannConfig::new()
        .with_num_postings(3)
        .with_nprobe(n)
        .with_metric(SpannMetric::Dot);
    let index = SpannIndex::build(items.clone(), config).unwrap();

    let query = vec![1.0, 1.0];
    let expected = brute_force_ids(&items, &query, SpannMetric::Dot, 4);
    let got: Vec<String> = index
        .search(&query, 4)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn search_k_zero_returns_empty() {
    let items = origin_and_far_fixture();
    let index = SpannIndex::build(items, SpannConfig::default()).unwrap();
    let hits = index.search(&[0.0, 0.0], 0).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn search_k_greater_than_len_returns_all() {
    let items = origin_and_far_fixture();
    let n = items.len();
    let config = SpannConfig::new().with_num_postings(2).with_nprobe(n);
    let index = SpannIndex::build(items, config).unwrap();
    let hits = index.search(&[0.0, 0.0], 1000).unwrap();
    assert_eq!(hits.len(), n);
}

#[test]
fn search_query_dimension_mismatch_errors() {
    let items = origin_and_far_fixture();
    let index = SpannIndex::build(items, SpannConfig::default()).unwrap();
    let err = index.search(&[1.0, 2.0, 3.0], 1).unwrap_err();
    assert_eq!(
        err,
        SpannError::DimensionMismatch {
            expected: 2,
            got: 3
        }
    );
}

#[test]
fn search_query_empty_errors() {
    let items = origin_and_far_fixture();
    let index = SpannIndex::build(items, SpannConfig::default()).unwrap();
    let err = index.search(&[], 1).unwrap_err();
    assert_eq!(err, SpannError::EmptyQuery);
}

#[test]
fn search_dim_zero_config_allows_empty_query() {
    let items = vec![("a".to_string(), Vec::new()), ("b".to_string(), Vec::new())];
    let config = SpannConfig::new().with_num_postings(1).with_nprobe(1);
    let index = SpannIndex::build(items, config).unwrap();
    assert_eq!(index.dim(), 0);
    let hits = index.search(&[], 2).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn search_nprobe_clamped_to_num_postings() {
    let items = origin_and_far_fixture();
    let config = SpannConfig::new().with_num_postings(2).with_nprobe(1000);
    let index = SpannIndex::build(items, config).unwrap();
    // Should not panic and should still be able to return every point.
    let hits = index.search(&[0.0, 0.0], 10).unwrap();
    assert_eq!(hits.len(), 6);
}

#[test]
fn search_dedup_replicated_point_returned_once() {
    // A point exactly midway between two centroids should be replicated
    // into both postings, but search must report it only once.
    let items = vec![
        ("left".to_string(), vec![0.0, 0.0]),
        ("mid".to_string(), vec![5.0, 0.0]),
        ("right".to_string(), vec![10.0, 0.0]),
    ];
    let config = SpannConfig::new()
        .with_num_postings(2)
        .with_nprobe(2)
        .with_replica_count(8)
        .with_boundary_epsilon(10.0)
        .with_metric(SpannMetric::L2);
    let index = SpannIndex::build(items, config).unwrap();

    let hits = index.search(&[5.0, 0.0], 10).unwrap();
    let ids: Vec<&String> = hits.iter().map(|h| &h.id).collect();
    let unique: std::collections::HashSet<&&String> = ids.iter().collect();
    assert_eq!(ids.len(), unique.len(), "duplicate id in search results");
}

#[test]
fn search_scores_descending_order() {
    let items = origin_and_far_fixture();
    let n = items.len();
    let config = SpannConfig::new().with_num_postings(2).with_nprobe(n);
    let index = SpannIndex::build(items, config).unwrap();
    let hits = index.search(&[0.0, 0.0], 6).unwrap();
    for pair in hits.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

#[test]
fn search_ties_broken_by_id_ascending() {
    // Two points equidistant from the query under L2.
    let items = vec![
        ("b".to_string(), vec![1.0, 0.0]),
        ("a".to_string(), vec![-1.0, 0.0]),
    ];
    let config = SpannConfig::new()
        .with_num_postings(1)
        .with_nprobe(1)
        .with_metric(SpannMetric::L2);
    let index = SpannIndex::build(items, config).unwrap();
    let hits = index.search(&[0.0, 0.0], 2).unwrap();
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[1].id, "b");
}

#[test]
fn search_build_and_search_all_metrics_smoke() {
    for metric in [SpannMetric::Cosine, SpannMetric::L2, SpannMetric::Dot] {
        let items = origin_and_far_fixture();
        let config = SpannConfig::new()
            .with_num_postings(2)
            .with_nprobe(2)
            .with_metric(metric);
        let index = SpannIndex::build(items, config).unwrap();
        let hits = index.search(&[0.0, 0.0], 3).unwrap();
        assert!(!hits.is_empty());
        assert!(hits.len() <= 3);
    }
}

#[test]
fn search_never_returns_more_than_k() {
    let items = origin_and_far_fixture();
    let config = SpannConfig::new().with_num_postings(2).with_nprobe(6);
    let index = SpannIndex::build(items, config).unwrap();
    for k in 0..8 {
        let hits = index.search(&[0.0, 0.0], k).unwrap();
        assert!(hits.len() <= k);
    }
}

#[test]
fn search_low_nprobe_still_returns_some_hits() {
    let items = origin_and_far_fixture();
    let config = SpannConfig::new().with_num_postings(2).with_nprobe(1);
    let index = SpannIndex::build(items, config).unwrap();
    let hits = index.search(&[0.0, 0.0], 3).unwrap();
    assert!(!hits.is_empty());
}
