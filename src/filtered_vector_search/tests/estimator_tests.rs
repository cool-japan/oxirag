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
//! Tests for the histogram, the per-attribute statistics, the selectivity
//! estimator, the strategy selector, and configuration validation.

use super::super::predicate::FilterPredicate;
use super::super::selectivity::{
    FilteredAttributeStats, FilteredNumericHistogram, SelectivityEstimator, StrategySelector,
};
use super::super::types::{
    AttrValue, FilterBound, FilterStrategy, FilteredDistanceMetric, FilteredSearchConfig,
    FilteredSearchError, SelectivityEstimate,
};
use super::{
    CORPUS_DIMENSION, CORPUS_SIZE, SELECTIVITY_TOLERANCE, build_corpus, selectivity_predicate,
    standard_index,
};

// ═══════════════════════════════════════════════════════════════════════════
// Histogram
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn histogram_empty() {
    let histogram = FilteredNumericHistogram::new(8);
    assert_eq!(histogram.total(), 0);
    assert_eq!(histogram.minimum(), None);
    assert_eq!(histogram.maximum(), None);
    let (count, exact) = histogram.estimate_range(FilterBound::Unbounded, FilterBound::Unbounded);
    assert_eq!(count, 0.0);
    assert!(exact);
}

#[test]
fn histogram_rebins_on_domain_growth() {
    let mut histogram = FilteredNumericHistogram::new(4);
    for value in [10.0, 20.0, 30.0, 40.0] {
        histogram.observe(value);
    }
    assert_eq!(histogram.total(), 4);
    assert_eq!(histogram.minimum(), Some(10.0));
    assert_eq!(histogram.maximum(), Some(40.0));
    assert_eq!(histogram.buckets().iter().sum::<u64>(), 4);

    // A value far outside the domain forces an exact re-bin; every observation
    // must survive it.
    histogram.observe(1000.0);
    assert_eq!(histogram.total(), 5);
    assert_eq!(histogram.maximum(), Some(1000.0));
    assert_eq!(histogram.buckets().iter().sum::<u64>(), 5);
}

#[test]
fn histogram_ignores_non_finite_observations() {
    let mut histogram = FilteredNumericHistogram::new(4);
    histogram.observe(1.0);
    histogram.observe(f64::NAN);
    histogram.observe(f64::INFINITY);
    // An infinity would collapse the domain and destroy every other estimate.
    assert_eq!(histogram.total(), 1);
    assert_eq!(histogram.maximum(), Some(1.0));
}

#[test]
fn histogram_exactness_is_reported_honestly() {
    let mut histogram = FilteredNumericHistogram::new(4);
    for value in 0..100 {
        histogram.observe(f64::from(value));
    }

    // Entirely below the domain: provably zero, therefore exact.
    let (count, exact) =
        histogram.estimate_range(FilterBound::Unbounded, FilterBound::Exclusive(-10.0));
    assert_eq!(count, 0.0);
    assert!(exact);

    // Covering the whole domain: provably everything, therefore exact.
    let (count, exact) = histogram.estimate_range(FilterBound::Unbounded, FilterBound::Unbounded);
    assert_eq!(count, 100.0);
    assert!(exact);

    // Slicing through a non-empty bucket: the uniformity assumption is in play,
    // and the estimate says so.
    let (count, exact) =
        histogram.estimate_range(FilterBound::Unbounded, FilterBound::Exclusive(30.0));
    assert!(!exact);
    assert!((count - 30.0).abs() < 5.0, "estimate {count} far from 30");
}

/// The integer-point apportioning rule, isolated.
///
/// This is the fix that makes the *planner* work, not merely the estimator look
/// tidy: with continuous apportioning, a 1%-selective integer predicate on the
/// module's fixture estimates at 1.29%, which sails past a `prefilter_threshold`
/// of 1% and routes the query to the approximate traversal instead of the exact
/// scan it deserves. A third of a percentage point of estimator error, and the
/// planner picks the wrong plan.
#[test]
fn histogram_apportions_integer_attributes_by_integer_points() {
    // The exact shape of the fixture: `bucket` uniform over 0..=99, twenty
    // records per value, thirty-two equi-width buckets.
    let mut histogram = FilteredNumericHistogram::new(32);
    for value in 0..100_i64 {
        for _ in 0..20 {
            histogram.observe(value as f64);
        }
    }
    assert!(histogram.is_integral());
    assert_eq!(histogram.total(), 2_000);

    // Bucket width is 99/32 = 3.09375, so bucket 0 spans [0, 3.09) and holds the
    // integers {0, 1, 2, 3} — 80 records. `bucket < 1` admits exactly one of
    // those four integer points.
    //
    // Continuous apportioning would give 80 * (1 / 3.09375) = 25.9. Integer
    // apportioning gives 80 * (1 / 4) = 20 — the true answer.
    let (count, exact) =
        histogram.estimate_range(FilterBound::Unbounded, FilterBound::Exclusive(1.0));
    assert_eq!(count, 20.0);
    // ...though it still rests on uniformity *across those four integer points*,
    // and reports as much.
    assert!(!exact);

    // `bucket < 10` needs no assumption at all: the integers 0..=9 fill buckets
    // 0, 1 and 2 exactly, so every bucket is wholly in or wholly out.
    let (count, exact) =
        histogram.estimate_range(FilterBound::Unbounded, FilterBound::Exclusive(10.0));
    assert_eq!(count, 200.0);
    assert!(exact);

    // Bound strictness is honored exactly on the integer path, where it really
    // does change which values are admitted.
    let (inclusive, _) =
        histogram.estimate_range(FilterBound::Inclusive(10.0), FilterBound::Inclusive(19.0));
    let (exclusive, _) =
        histogram.estimate_range(FilterBound::Exclusive(9.0), FilterBound::Exclusive(20.0));
    assert_eq!(inclusive, 200.0, "the ten integers 10..=19");
    assert_eq!(
        exclusive, inclusive,
        "(9, 20) and [10, 19] admit the same integers"
    );
}

#[test]
fn histogram_uses_continuous_apportioning_for_float_attributes() {
    let mut histogram = FilteredNumericHistogram::new(10);
    // 1000 values evenly spread over [0, 1): decidedly not integers.
    for step in 0..1_000 {
        histogram.observe(f64::from(step) / 1_000.0);
    }
    assert!(!histogram.is_integral());

    let (count, _) =
        histogram.estimate_range(FilterBound::Inclusive(0.0), FilterBound::Exclusive(0.25));
    assert!(
        (count - 250.0).abs() <= 5.0,
        "continuous interpolation should land near 250, got {count}"
    );
}

#[test]
fn histogram_degenerate_domain() {
    let mut histogram = FilteredNumericHistogram::new(8);
    for _ in 0..10 {
        histogram.observe(5.0);
    }
    // Every observation is the same value: the answer is all-or-nothing, and
    // exactly so.
    let (count, exact) =
        histogram.estimate_range(FilterBound::Inclusive(5.0), FilterBound::Inclusive(5.0));
    assert_eq!(count, 10.0);
    assert!(exact);

    let (count, exact) =
        histogram.estimate_range(FilterBound::Exclusive(5.0), FilterBound::Unbounded);
    assert_eq!(count, 0.0);
    assert!(exact);
}

// ═══════════════════════════════════════════════════════════════════════════
// Attribute statistics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn attribute_stats_exact_value_counts() {
    let mut stats = FilteredAttributeStats::new(8, 64);
    for _ in 0..7 {
        stats.observe(&AttrValue::Str("ja".into()));
    }
    for _ in 0..3 {
        stats.observe(&AttrValue::Str("en".into()));
    }
    assert_eq!(stats.present_count(), 10);
    assert!(!stats.is_saturated());
    assert_eq!(stats.tracked_distinct(), 2);

    let (count, exact) = stats.estimate_eq(&AttrValue::Str("ja".into()));
    assert_eq!(count, 7.0);
    assert!(exact);

    // Unsaturated + absent value => provably zero.
    let (count, exact) = stats.estimate_eq(&AttrValue::Str("fr".into()));
    assert_eq!(count, 0.0);
    assert!(exact);

    // `Ne` is presence-scoped, so it complements within the present set.
    let (count, exact) = stats.estimate_ne(&AttrValue::Str("ja".into()));
    assert_eq!(count, 3.0);
    assert!(exact);

    let (count, exact) =
        stats.estimate_in(&[AttrValue::Str("ja".into()), AttrValue::Str("en".into())]);
    assert_eq!(count, 10.0);
    assert!(exact);
}

#[test]
fn attribute_stats_saturation_falls_back_honestly() {
    // Cap at 4 distinct values; feed it 10.
    let mut stats = FilteredAttributeStats::new(8, 4);
    for value in 0..10_i64 {
        for _ in 0..2 {
            stats.observe(&AttrValue::Int(value));
        }
    }
    assert!(stats.is_saturated());
    assert_eq!(stats.tracked_distinct(), 4);
    assert_eq!(stats.present_count(), 20);

    // A tracked value keeps an exact count even after saturation.
    let (count, exact) = stats.estimate_eq(&AttrValue::Int(0));
    assert_eq!(count, 2.0);
    assert!(exact);

    // An untracked value gets the average-multiplicity heuristic, and the
    // estimate declares itself inexact rather than pretending.
    let (count, exact) = stats.estimate_eq(&AttrValue::Int(9));
    assert!(!exact);
    assert!(count > 0.0, "the heuristic must not report a false zero");
    assert!(count <= 12.0);
}

#[test]
fn attribute_stats_point_range_uses_value_counts_not_the_histogram() {
    let mut stats = FilteredAttributeStats::new(8, 64);
    for _ in 0..5 {
        stats.observe(&AttrValue::Int(42));
    }
    for value in 0..50_i64 {
        stats.observe(&AttrValue::Int(value));
    }

    // A zero-width range is a point query. Interpolating over zero width would
    // yield zero regardless of the true count; the value counts must answer it.
    let (count, exact) =
        stats.estimate_range(FilterBound::Inclusive(42.0), FilterBound::Inclusive(42.0));
    assert_eq!(count, 6.0, "5 extra + 1 from the 0..50 sweep");
    assert!(exact);
}

// ═══════════════════════════════════════════════════════════════════════════
// Selectivity estimator
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn estimator_is_exact_on_equality_and_existence() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 11);
    let mut estimator = SelectivityEstimator::new(32, 4_096);
    for metadata in &corpus.metadata {
        estimator.observe(metadata);
    }
    assert_eq!(estimator.total(), CORPUS_SIZE);
    assert_eq!(
        estimator.attribute_names(),
        vec![
            "bucket".to_string(),
            "lang".to_string(),
            "public".to_string(),
            "score".to_string(),
            "tag".to_string()
        ]
    );

    // Equality is served from exact value counts, so it is not an estimate at
    // all — it is a count.
    let predicate = FilterPredicate::eq("lang", "ja");
    let estimate = estimator.estimate(&predicate);
    assert_eq!(estimate.selectivity, corpus.true_selectivity(&predicate));
    assert!(estimate.exact_leaves);
    assert!(!estimate.independence_assumed);
    assert!(estimate.is_exact());

    // So is existence.
    let predicate = FilterPredicate::exists("tag");
    let estimate = estimator.estimate(&predicate);
    assert_eq!(estimate.selectivity, corpus.true_selectivity(&predicate));
    assert!(estimate.is_exact());

    // An attribute the estimator has never seen is present in *no* record — a
    // conclusion, not a guess.
    let predicate = FilterPredicate::exists("never_indexed");
    let estimate = estimator.estimate(&predicate);
    assert_eq!(estimate.selectivity, 0.0);
    assert!(estimate.exact_leaves);
}

#[test]
fn estimator_accuracy_across_predicate_shapes() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 11);
    let index = standard_index(&corpus);

    let cases: Vec<(&str, FilterPredicate)> = vec![
        ("range 1%", selectivity_predicate(1)),
        ("range 10%", selectivity_predicate(10)),
        ("range 50%", selectivity_predicate(50)),
        ("range 90%", selectivity_predicate(90)),
        (
            "float range",
            FilterPredicate::range(
                "score",
                FilterBound::Inclusive(0.25),
                FilterBound::Exclusive(0.75),
            ),
        ),
        ("eq", FilterPredicate::eq("lang", "ja")),
        ("ne", FilterPredicate::ne("lang", "ja")),
        ("in", FilterPredicate::in_set("lang", ["ja", "en"])),
        ("exists", FilterPredicate::exists("tag")),
        (
            "not",
            FilterPredicate::not(FilterPredicate::eq("lang", "ja")),
        ),
        (
            "and",
            FilterPredicate::and([selectivity_predicate(10), FilterPredicate::eq("lang", "ja")]),
        ),
        (
            "or",
            FilterPredicate::or([selectivity_predicate(10), FilterPredicate::eq("lang", "ja")]),
        ),
        (
            "nested",
            FilterPredicate::and([
                FilterPredicate::or([
                    FilterPredicate::eq("lang", "ja"),
                    FilterPredicate::eq("lang", "en"),
                ]),
                FilterPredicate::not(selectivity_predicate(50)),
            ]),
        ),
    ];

    for (label, predicate) in cases {
        let truth = corpus.true_selectivity(&predicate);
        let estimate = index
            .estimate_selectivity(&predicate)
            .expect("predicate is well-formed");
        let error = (estimate.selectivity - truth).abs();
        assert!(
            error <= SELECTIVITY_TOLERANCE,
            "{label}: estimated {:.4}, true {truth:.4}, error {error:.4} exceeds {SELECTIVITY_TOLERANCE}",
            estimate.selectivity
        );
        assert_eq!(
            estimate.estimated_matches,
            estimate.selectivity * CORPUS_SIZE as f64
        );
        assert_eq!(estimate.total_vectors, CORPUS_SIZE);
    }
}

#[test]
fn estimator_flags_its_own_assumptions() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 11);
    let index = standard_index(&corpus);

    // `bucket < 10` looks like it must be approximate — a continuous equi-width
    // histogram over 0..=99 with 32 buckets has boundaries at 3.09, 6.19, 9.28,
    // ... none of which is 10. But the attribute is an *integer* attribute, so
    // buckets are apportioned by the integer points they contain, and the
    // integers 0..=9 happen to fill buckets 0, 1 and 2 exactly. Every bucket is
    // therefore wholly inside or wholly outside the range, no uniformity
    // assumption is needed, and the estimate is not an estimate at all.
    let aligned = index
        .estimate_selectivity(&selectivity_predicate(10))
        .expect("valid");
    assert!(
        aligned.exact_leaves,
        "integer-point apportioning makes this range provably exact"
    );
    assert_eq!(
        aligned.selectivity,
        corpus.true_selectivity(&selectivity_predicate(10))
    );
    assert!(aligned.is_exact());

    // `bucket < 1` genuinely slices bucket 0 (which holds the integers
    // {0, 1, 2, 3}, of which the range admits only one). The uniformity
    // assumption is in play — over the four integer points rather than over the
    // bucket's continuous length — and the estimate says so, even though on this
    // uniformly-distributed fixture it happens to land on the exact answer.
    let sliced = index
        .estimate_selectivity(&selectivity_predicate(1))
        .expect("valid");
    assert!(
        !sliced.exact_leaves,
        "a partially-covered non-empty bucket must not claim exactness"
    );
    assert!(!sliced.independence_assumed);

    // A disjunction of equalities on the *same* attribute is mutually exclusive,
    // so the estimator adds the counts and assumes nothing.
    let disjunction = index
        .estimate_selectivity(&FilterPredicate::or([
            FilterPredicate::eq("lang", "ja"),
            FilterPredicate::eq("lang", "en"),
        ]))
        .expect("valid");
    assert!(!disjunction.independence_assumed);
    assert!(disjunction.is_exact());
    assert_eq!(
        disjunction.selectivity, 0.5,
        "two of four balanced languages"
    );

    // A conjunction folds two estimates under independence, and says so.
    let conjunction = index
        .estimate_selectivity(&FilterPredicate::and([
            FilterPredicate::eq("lang", "ja"),
            FilterPredicate::eq("public", true),
        ]))
        .expect("valid");
    assert!(conjunction.exact_leaves, "both leaves are exact counts");
    assert!(conjunction.independence_assumed);
    assert!(!conjunction.is_exact());
}

#[test]
fn estimator_handles_empty_corpus() {
    let estimator = SelectivityEstimator::new(8, 16);
    assert_eq!(estimator.total(), 0);
    let estimate = estimator.estimate(&FilterPredicate::eq("lang", "ja"));
    assert_eq!(estimate.selectivity, 0.0);
    assert_eq!(estimate.estimated_matches, 0.0);
    // `all()` is vacuously true, but a vacuous truth over nothing still matches
    // nothing.
    let estimate = estimator.estimate(&FilterPredicate::all());
    assert_eq!(estimate.selectivity, 1.0);
    assert_eq!(estimate.estimated_matches, 0.0);
}

#[test]
fn selectivity_estimate_clamps_and_reports() {
    let estimate = SelectivityEstimate::new(1.5, 100, true, false);
    assert_eq!(estimate.selectivity, 1.0);
    assert_eq!(estimate.estimated_matches, 100.0);

    let estimate = SelectivityEstimate::new(-0.5, 100, true, false);
    assert_eq!(estimate.selectivity, 0.0);

    // A non-finite intermediate degrades to "assume everything matches", which
    // is the safe direction: it never picks a strategy that loses recall.
    let estimate = SelectivityEstimate::new(f64::NAN, 100, true, false);
    assert_eq!(estimate.selectivity, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Strategy selector
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn selector_picks_the_expected_strategy_in_each_regime() {
    let config = FilteredSearchConfig::default();
    let selector = StrategySelector::from_config(&config);
    let total = 1_000_000;

    // Low selectivity, few enough matches to scan => exact scan wins outright.
    let low = SelectivityEstimate::new(0.001, total, true, false);
    assert_eq!(selector.select(&low, 10), FilterStrategy::PreFilter);

    // Middling => ACORN.
    for selectivity in [0.05, 0.1, 0.3, 0.5, 0.79] {
        let estimate = SelectivityEstimate::new(selectivity, total, true, false);
        assert_eq!(
            selector.select(&estimate, 10),
            FilterStrategy::InFilter,
            "selectivity {selectivity} should route to InFilter"
        );
    }

    // High => the over-fetch is nearly free.
    for selectivity in [0.8, 0.95, 1.0] {
        let estimate = SelectivityEstimate::new(selectivity, total, true, false);
        assert_eq!(
            selector.select(&estimate, 10),
            FilterStrategy::PostFilter,
            "selectivity {selectivity} should route to PostFilter"
        );
    }
}

#[test]
fn selector_gates_prefilter_on_the_absolute_match_count_too() {
    let config = FilteredSearchConfig::default();
    let selector = StrategySelector::from_config(&config);

    // 1% of a thousand vectors is ten records: scan them.
    let small = SelectivityEstimate::new(0.01, 1_000, true, false);
    assert_eq!(selector.select(&small, 10), FilterStrategy::PreFilter);

    // 1% of a billion vectors is ten *million* records. The ratio is identical;
    // the right plan is not. A ratio-only gate would happily scan them.
    let huge = SelectivityEstimate::new(0.01, 1_000_000_000, true, false);
    assert_eq!(selector.select(&huge, 10), FilterStrategy::InFilter);
}

#[test]
fn selector_prefers_the_exact_scan_when_k_exceeds_the_matching_set() {
    let config = FilteredSearchConfig::default();
    let selector = StrategySelector::from_config(&config);

    // 50% selectivity would normally mean `InFilter` — but if the caller wants
    // more results than there are matches, the whole matching set must be
    // examined regardless, and an approximate traversal of it can only lose
    // recall for nothing.
    let estimate = SelectivityEstimate::new(0.5, 40, true, false);
    assert_eq!(selector.select(&estimate, 100), FilterStrategy::PreFilter);
    // With a small `k` the same estimate routes to the traversal.
    assert_eq!(selector.select(&estimate, 5), FilterStrategy::InFilter);
}

#[test]
fn selector_never_returns_auto() {
    let config = FilteredSearchConfig::default();
    let selector = StrategySelector::from_config(&config);
    for step in 0..=100 {
        let estimate = SelectivityEstimate::new(f64::from(step) / 100.0, 100_000, true, false);
        assert_ne!(selector.select(&estimate, 10), FilterStrategy::Auto);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn default_config_is_valid() {
    let config = FilteredSearchConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.max_visits(), 64 * 16);
    assert_eq!(config.matching_reserve(), 16);
    assert_eq!(config.metric, FilteredDistanceMetric::Cosine);
}

#[test]
fn config_rejects_a_selector_that_could_route_into_postfilters_failure_regime() {
    // `PostFilter` needs `s >= 1 / m` for its over-fetch to be able to contain
    // `k` matches. A `postfilter_threshold` below that break-even would let the
    // planner choose it in the regime where it silently truncates.
    let config = FilteredSearchConfig {
        post_filter_multiplier: 10,
        postfilter_threshold: 0.05,
        ..FilteredSearchConfig::default()
    };
    let error = config.validate().expect_err("must be rejected");
    assert!(matches!(
        error,
        FilteredSearchError::InvalidConfig { ref field, .. } if field == "postfilter_threshold"
    ));

    // Raising the multiplier to match makes the same threshold sound again.
    let config = FilteredSearchConfig {
        post_filter_multiplier: 20,
        postfilter_threshold: 0.05,
        ..FilteredSearchConfig::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn config_rejects_out_of_range_parameters() {
    let cases: Vec<(&str, FilteredSearchConfig)> = vec![
        (
            "graph_degree",
            FilteredSearchConfig {
                graph_degree: 0,
                ..Default::default()
            },
        ),
        (
            "build_beam_width",
            FilteredSearchConfig {
                graph_degree: 32,
                build_beam_width: 16,
                ..Default::default()
            },
        ),
        (
            "prune_alpha",
            FilteredSearchConfig {
                prune_alpha: 0.5,
                ..Default::default()
            },
        ),
        (
            "search_beam_width",
            FilteredSearchConfig {
                search_beam_width: 0,
                ..Default::default()
            },
        ),
        (
            "seed_count",
            FilteredSearchConfig {
                seed_count: 0,
                ..Default::default()
            },
        ),
        (
            "neighbor_expansion_gamma",
            FilteredSearchConfig {
                neighbor_expansion_gamma: 0,
                ..Default::default()
            },
        ),
        (
            "min_predicate_neighbors",
            FilteredSearchConfig {
                min_predicate_neighbors: 0,
                ..Default::default()
            },
        ),
        (
            "matching_reserve_ratio",
            FilteredSearchConfig {
                matching_reserve_ratio: 1.0,
                ..Default::default()
            },
        ),
        (
            "max_visit_multiplier",
            FilteredSearchConfig {
                max_visit_multiplier: 0,
                ..Default::default()
            },
        ),
        (
            "post_filter_multiplier",
            FilteredSearchConfig {
                post_filter_multiplier: 0,
                ..Default::default()
            },
        ),
        (
            "prefilter_threshold",
            FilteredSearchConfig {
                prefilter_threshold: 0.9,
                postfilter_threshold: 0.8,
                ..Default::default()
            },
        ),
        (
            "histogram_buckets",
            FilteredSearchConfig {
                histogram_buckets: 0,
                ..Default::default()
            },
        ),
        (
            "max_tracked_values",
            FilteredSearchConfig {
                max_tracked_values: 0,
                ..Default::default()
            },
        ),
    ];

    for (field, config) in cases {
        let error = config
            .validate()
            .expect_err(&format!("{field} must be rejected"));
        match error {
            FilteredSearchError::InvalidConfig {
                field: reported, ..
            } => assert_eq!(reported, field),
            other => panic!("{field}: expected InvalidConfig, got {other:?}"),
        }
    }
}
