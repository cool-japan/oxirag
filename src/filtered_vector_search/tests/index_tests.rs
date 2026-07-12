#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]
//! Tests for the index's mechanics, its error surface, and its edge cases.

use std::collections::HashSet;

use super::super::index::{FilteredVectorIndex, FilteredVectorRecord};
use super::super::predicate::FilterPredicate;
use super::super::types::{
    FilterStrategy, FilteredDistanceMetric, FilteredMetadata, FilteredSearchConfig,
    FilteredSearchError,
};
use super::{
    BUCKET_CARDINALITY, CORPUS_DIMENSION, CORPUS_SIZE, brute_force_ground_truth, build_corpus,
    build_queries, mean_recall, selectivity_predicate, standard_index,
};

// ═══════════════════════════════════════════════════════════════════════════
// Index mechanics and edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn index_rejects_a_zero_dimension() {
    assert!(matches!(
        FilteredVectorIndex::with_default_config(0),
        Err(FilteredSearchError::EmptyVector)
    ));
}

#[test]
fn index_rejects_bad_inserts() {
    let mut index = FilteredVectorIndex::with_default_config(3).expect("valid dimension");
    index
        .insert("a", vec![1.0, 0.0, 0.0], FilteredMetadata::new())
        .expect("well-formed");

    assert!(matches!(
        index.insert("b", vec![1.0, 0.0], FilteredMetadata::new()),
        Err(FilteredSearchError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        index.insert("c", vec![1.0, f32::NAN, 0.0], FilteredMetadata::new()),
        Err(FilteredSearchError::NonFiniteVector { component: 1, .. })
    ));
    assert!(matches!(
        index.insert("a", vec![0.0, 1.0, 0.0], FilteredMetadata::new()),
        Err(FilteredSearchError::DuplicateId(_))
    ));
    assert_eq!(index.len(), 1);
}

#[test]
fn index_rejects_bad_queries() {
    let corpus = build_corpus(64, CORPUS_DIMENSION, 3);
    let index = standard_index(&corpus);
    let predicate = FilterPredicate::all();

    assert!(matches!(
        index.search(&[0.0; 4], 5, &predicate),
        Err(FilteredSearchError::DimensionMismatch { expected: 16, .. })
    ));

    let mut bad_query = vec![0.1_f32; CORPUS_DIMENSION];
    bad_query[7] = f32::INFINITY;
    assert!(matches!(
        index.search(&bad_query, 5, &predicate),
        Err(FilteredSearchError::NonFiniteVector { component: 7, .. })
    ));

    let malformed = FilterPredicate::range_inclusive("bucket", 10.0, 1.0);
    let query = vec![0.1_f32; CORPUS_DIMENSION];
    assert!(matches!(
        index.search(&query, 5, &malformed),
        Err(FilteredSearchError::InvalidPredicate { .. })
    ));
    assert!(index.exact_search(&query, 5, &malformed).is_err());
    assert!(index.estimate_selectivity(&malformed).is_err());
}

#[test]
fn empty_index_returns_no_hits() {
    let index = FilteredVectorIndex::with_default_config(4).expect("valid dimension");
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.dimension(), 4);

    let query = vec![1.0, 0.0, 0.0, 0.0];
    for strategy in [
        FilterStrategy::Auto,
        FilterStrategy::PreFilter,
        FilterStrategy::PostFilter,
        FilterStrategy::InFilter,
    ] {
        let (hits, stats) = index
            .search_with_strategy(&query, 10, &FilterPredicate::all(), strategy)
            .expect("an empty index is not an error");
        assert!(hits.is_empty(), "{strategy:?} must return nothing");
        assert_eq!(stats.nodes_visited, 0);
    }
    assert!(
        index
            .exact_search(&query, 10, &FilterPredicate::all())
            .expect("valid")
            .is_empty()
    );
}

#[test]
fn predicate_matching_zero_documents_returns_no_hits() {
    let corpus = build_corpus(512, CORPUS_DIMENSION, 5);
    let index = standard_index(&corpus);
    let query = build_queries(1, CORPUS_DIMENSION, 99).remove(0);

    // Two ways of matching nothing: a value that does not occur, and the
    // vacuously-false empty disjunction.
    for predicate in [
        FilterPredicate::eq("lang", "klingon"),
        FilterPredicate::none(),
        FilterPredicate::and([
            FilterPredicate::eq("lang", "ja"),
            FilterPredicate::eq("lang", "en"),
        ]),
    ] {
        assert_eq!(corpus.true_match_count(&predicate), 0);
        for strategy in [
            FilterStrategy::Auto,
            FilterStrategy::PreFilter,
            FilterStrategy::PostFilter,
            FilterStrategy::InFilter,
        ] {
            let (hits, _) = index
                .search_with_strategy(&query, 10, &predicate, strategy)
                .expect("valid");
            assert!(
                hits.is_empty(),
                "{strategy:?} returned {} hits for a predicate matching nothing",
                hits.len()
            );
        }
    }
}

#[test]
fn predicate_matching_all_documents_behaves_like_an_unconstrained_search() {
    let corpus = build_corpus(512, CORPUS_DIMENSION, 5);
    let index = standard_index(&corpus);
    let queries = build_queries(8, CORPUS_DIMENSION, 77);
    let predicate = FilterPredicate::all();
    assert_eq!(corpus.true_match_count(&predicate), 512);

    for strategy in [
        FilterStrategy::PreFilter,
        FilterStrategy::PostFilter,
        FilterStrategy::InFilter,
    ] {
        let recall = mean_recall(&index, &corpus, &queries, 10, &predicate, strategy);
        assert!(
            recall >= 0.9,
            "{strategy:?} recall {recall:.3} under an unconstrained predicate"
        );
    }

    // The planner should route an unconstrained predicate to `PostFilter`
    // (selectivity 1.0), where the over-fetch is free.
    let (strategy, estimate) = index.plan(&predicate, 10).expect("valid");
    assert_eq!(estimate.selectivity, 1.0);
    assert_eq!(strategy, FilterStrategy::PostFilter);
}

#[test]
fn k_larger_than_the_matching_set_returns_the_whole_matching_set() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 7);
    let index = standard_index(&corpus);
    let query = build_queries(1, CORPUS_DIMENSION, 31).remove(0);

    // `bucket == 3` matches exactly 20 records; ask for 100.
    let predicate = FilterPredicate::eq("bucket", 3_i64);
    let expected = corpus.true_match_count(&predicate);
    assert_eq!(expected, CORPUS_SIZE / BUCKET_CARDINALITY);

    let (hits, stats) = index
        .search_with_strategy(&query, 100, &predicate, FilterStrategy::PreFilter)
        .expect("valid");
    assert_eq!(hits.len(), expected);
    assert_eq!(stats.matched_candidates, expected);

    // ...and the ranks are dense and ordered.
    for (position, hit) in hits.iter().enumerate() {
        assert_eq!(hit.rank, position);
    }

    let exact = index.exact_search(&query, 100, &predicate).expect("valid");
    assert_eq!(exact.len(), expected);
}

#[test]
fn top_k_of_zero_returns_nothing() {
    let corpus = build_corpus(128, CORPUS_DIMENSION, 9);
    let index = standard_index(&corpus);
    let query = build_queries(1, CORPUS_DIMENSION, 2).remove(0);
    let (hits, _) = index
        .search_with_stats(&query, 0, &FilterPredicate::all())
        .expect("valid");
    assert!(hits.is_empty());
}

#[test]
fn hits_are_ordered_and_scored_consistently() {
    let corpus = build_corpus(512, CORPUS_DIMENSION, 13);
    let index = standard_index(&corpus);
    let query = build_queries(1, CORPUS_DIMENSION, 4).remove(0);

    let hits = index
        .search(&query, 10, &FilterPredicate::exists("lang"))
        .expect("valid");
    assert_eq!(hits.len(), 10);
    for window in hits.windows(2) {
        // Distance ascends, score descends: the two conventions must never
        // disagree.
        assert!(window[0].distance <= window[1].distance);
        assert!(window[0].score >= window[1].score);
    }
    for hit in &hits {
        // Cosine: score == 1 - distance.
        assert!((hit.score - (1.0 - hit.distance)).abs() < 1e-6);
        assert!(index.metadata_of(&hit.id).is_some());
    }
}

#[test]
fn euclidean_metric_ranks_and_scores_correctly() {
    let config = FilteredSearchConfig {
        metric: FilteredDistanceMetric::Euclidean,
        ..FilteredSearchConfig::default()
    };
    let records = vec![
        FilteredVectorRecord::new(
            "near",
            vec![1.0, 0.0],
            FilteredMetadata::new().with("k", 1_i64),
        ),
        FilteredVectorRecord::new(
            "far",
            vec![5.0, 0.0],
            FilteredMetadata::new().with("k", 1_i64),
        ),
    ];
    let index = FilteredVectorIndex::build(2, config, records).expect("valid");

    let hits = index
        .search(&[0.0, 0.0], 2, &FilterPredicate::eq("k", 1_i64))
        .expect("valid");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "near");
    assert!((hits[0].distance - 1.0).abs() < 1e-6);
    assert!((hits[1].distance - 5.0).abs() < 1e-6);
    // Euclidean scores are negated distances, so larger is still better.
    assert!((hits[0].score + 1.0).abs() < 1e-6);
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn zero_norm_vectors_are_orthogonal_to_everything_under_cosine() {
    let records = vec![
        FilteredVectorRecord::new("zero", vec![0.0, 0.0], FilteredMetadata::new()),
        FilteredVectorRecord::new("unit", vec![1.0, 0.0], FilteredMetadata::new()),
    ];
    let index =
        FilteredVectorIndex::build(2, FilteredSearchConfig::default(), records).expect("valid");
    let hits = index
        .search(&[1.0, 0.0], 2, &FilterPredicate::all())
        .expect("valid");
    assert_eq!(hits[0].id, "unit");
    assert!((hits[0].distance).abs() < 1e-6);
    assert_eq!(hits[1].id, "zero");
    assert!((hits[1].distance - 1.0).abs() < 1e-6);
}

#[test]
fn incremental_inserts_are_queryable_without_an_explicit_optimize() {
    let corpus = build_corpus(400, CORPUS_DIMENSION, 17);
    let mut index =
        FilteredVectorIndex::new(CORPUS_DIMENSION, FilteredSearchConfig::default()).expect("valid");
    for record in corpus.records() {
        index.insert_record(record).expect("well-formed");
    }
    assert_eq!(index.len(), 400);

    // Ranges must stay exact even before `optimize` has merged the pending
    // numeric values into the sorted arrays.
    let query = build_queries(1, CORPUS_DIMENSION, 8).remove(0);
    let predicate = selectivity_predicate(10);
    let (hits, _) = index
        .search_with_strategy(&query, 5, &predicate, FilterStrategy::PreFilter)
        .expect("valid");
    let truth = brute_force_ground_truth(&corpus, &query, 5, &predicate);
    assert_eq!(
        hits.iter().map(|hit| hit.id.clone()).collect::<Vec<_>>(),
        truth
    );

    // `optimize` is idempotent and must not change PreFilter's exact answer.
    index.optimize();
    index.optimize();
    let (hits, _) = index
        .search_with_strategy(&query, 5, &predicate, FilterStrategy::PreFilter)
        .expect("valid");
    assert_eq!(
        hits.iter().map(|hit| hit.id.clone()).collect::<Vec<_>>(),
        truth
    );
}

#[test]
fn graph_respects_its_degree_bound_and_reaches_every_node() {
    let corpus = build_corpus(512, CORPUS_DIMENSION, 19);
    let index = standard_index(&corpus);
    let graph = index.graph();
    let degree = index.config().graph_degree;

    for node in 0..512_u32 {
        let neighbors = graph.neighbors_of(node);
        assert!(
            neighbors.len() <= degree,
            "node {node} has {} neighbors, exceeding the degree bound {degree}",
            neighbors.len()
        );
        assert!(
            !neighbors.contains(&node),
            "node {node} is its own neighbor"
        );
        assert!(
            !neighbors.is_empty(),
            "node {node} is isolated; greedy search could never reach it"
        );
    }

    // The whole graph must be reachable from the medoid, or a traversal seeded
    // there simply cannot find some of the data — filtered or not.
    let mut reached: HashSet<u32> = HashSet::new();
    let mut frontier = vec![graph.medoid()];
    reached.insert(graph.medoid());
    while let Some(node) = frontier.pop() {
        for &neighbor in graph.neighbors_of(node) {
            if reached.insert(neighbor) {
                frontier.push(neighbor);
            }
        }
    }
    assert_eq!(
        reached.len(),
        512,
        "only {} of 512 nodes are reachable from the medoid",
        reached.len()
    );
}

#[test]
fn index_is_deterministic() {
    let corpus = build_corpus(300, CORPUS_DIMENSION, 23);
    let query = build_queries(1, CORPUS_DIMENSION, 24).remove(0);
    let predicate = selectivity_predicate(20);

    let first = standard_index(&corpus);
    let second = standard_index(&corpus);
    for strategy in [
        FilterStrategy::Auto,
        FilterStrategy::PreFilter,
        FilterStrategy::PostFilter,
        FilterStrategy::InFilter,
    ] {
        let (left, left_stats) = first
            .search_with_strategy(&query, 10, &predicate, strategy)
            .expect("valid");
        let (right, right_stats) = second
            .search_with_strategy(&query, 10, &predicate, strategy)
            .expect("valid");
        assert_eq!(left, right, "{strategy:?} is not deterministic");
        assert_eq!(left_stats, right_stats);
    }
}
