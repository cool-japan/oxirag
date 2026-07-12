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
//! The measurements that justify the module: `PreFilter`'s exactness,
//! `InFilter`'s recall across selectivity regimes, and the headline collapse of
//! a bounded-over-fetch post-filter.

use super::super::index::{FilteredVectorIndex, FilteredVectorRecord};
use super::super::predicate::FilterPredicate;
use super::super::types::{
    FilterBound, FilterStrategy, FilteredDistanceMetric, FilteredHit, FilteredMetadata,
    FilteredSearchConfig,
};
use super::{
    CORPUS_DIMENSION, CORPUS_SIZE, SELECTIVITY_TOLERANCE, brute_force_ground_truth, build_corpus,
    build_queries, cosine_distance, mean_recall, selectivity_predicate, standard_index,
};

// ═══════════════════════════════════════════════════════════════════════════
// PreFilter exactness
// ═══════════════════════════════════════════════════════════════════════════

/// `PreFilter` claims to be *exact*, not merely accurate. This asserts equality
/// — same ids, same order, same distances — against a brute-force ground truth
/// written independently in this file.
///
/// It is a real test of the inverted-index resolver, not a tautology: the
/// resolver walks posting lists, sorted numeric arrays, set intersections,
/// unions and complements to produce a candidate set (which it is allowed to
/// over-approximate), and the strategy then verifies and scans it. The ground
/// truth does none of that — it scans all 2,000 records with its own distance
/// function. Any bug in the resolver that *drops* a candidate shows up here
/// immediately.
#[test]
fn prefilter_returns_exactly_the_brute_force_top_k() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 29);
    let index = standard_index(&corpus);
    let queries = build_queries(12, CORPUS_DIMENSION, 30);

    // Exercise every resolver path: posting lists (Eq/In/Ne), the sorted numeric
    // array (Range), the present list (Exists), intersection (And), union (Or),
    // and complementation (Not) — plus a mixed And whose second conjunct the
    // resolver can serve while the first widens it.
    let predicates: Vec<(&str, FilterPredicate)> = vec![
        ("eq", FilterPredicate::eq("lang", "ja")),
        ("ne", FilterPredicate::ne("lang", "ja")),
        ("in", FilterPredicate::in_set("lang", ["ja", "fr"])),
        ("exists", FilterPredicate::exists("tag")),
        ("range", selectivity_predicate(10)),
        (
            "float range",
            FilterPredicate::range(
                "score",
                FilterBound::Inclusive(0.2),
                FilterBound::Exclusive(0.6),
            ),
        ),
        (
            "and",
            FilterPredicate::and([FilterPredicate::eq("lang", "ja"), selectivity_predicate(50)]),
        ),
        (
            "or",
            FilterPredicate::or([FilterPredicate::eq("lang", "de"), selectivity_predicate(5)]),
        ),
        ("not", FilterPredicate::not(FilterPredicate::exists("tag"))),
        (
            "not-of-and",
            FilterPredicate::not(FilterPredicate::and([
                FilterPredicate::eq("lang", "ja"),
                FilterPredicate::eq("public", true),
            ])),
        ),
        (
            "nested",
            FilterPredicate::and([
                FilterPredicate::or([
                    FilterPredicate::eq("lang", "ja"),
                    FilterPredicate::eq("lang", "en"),
                ]),
                FilterPredicate::not(selectivity_predicate(30)),
                FilterPredicate::exists("score"),
            ]),
        ),
        ("all", FilterPredicate::all()),
    ];

    for (label, predicate) in &predicates {
        assert!(
            corpus.true_match_count(predicate) > 0,
            "{label}: fixture bug — the predicate matches nothing"
        );
        for query in &queries {
            let (hits, _) = index
                .search_with_strategy(query, 10, predicate, FilterStrategy::PreFilter)
                .expect("valid");
            let truth = brute_force_ground_truth(&corpus, query, 10, predicate);
            let returned: Vec<String> = hits.iter().map(|hit| hit.id.clone()).collect();
            assert_eq!(
                returned, truth,
                "{label}: PreFilter is not exact (it claims to be)"
            );

            // Distances must agree too, not just the identity of the hits.
            for hit in &hits {
                let position = corpus
                    .ids
                    .iter()
                    .position(|id| *id == hit.id)
                    .expect("hit id is in the corpus");
                let expected = cosine_distance(query, &corpus.vectors[position]);
                assert!(
                    (hit.distance - expected).abs() < 1e-6,
                    "{label}: distance {} != {expected}",
                    hit.distance
                );
            }
        }
    }
}

#[test]
fn prefilter_survives_a_predicate_the_inverted_index_cannot_resolve() {
    // `Not` over an `And` with an unresolvable conjunct cannot be complemented
    // safely, so the resolver must decline and the strategy must fall back to a
    // full scan — still exact, merely slower. We force the "cannot resolve" path
    // by capping the posting lists at a single distinct value per attribute.
    let corpus = build_corpus(600, CORPUS_DIMENSION, 31);
    let config = FilteredSearchConfig {
        max_postings_per_attr: 1,
        ..FilteredSearchConfig::default()
    };
    let index =
        FilteredVectorIndex::build(CORPUS_DIMENSION, config, corpus.records()).expect("valid");
    let queries = build_queries(6, CORPUS_DIMENSION, 32);

    let predicate = FilterPredicate::not(FilterPredicate::eq("lang", "ja"));
    for query in &queries {
        let (hits, _) = index
            .search_with_strategy(query, 10, &predicate, FilterStrategy::PreFilter)
            .expect("valid");
        let truth = brute_force_ground_truth(&corpus, query, 10, &predicate);
        assert_eq!(
            hits.iter().map(|hit| hit.id.clone()).collect::<Vec<_>>(),
            truth,
            "PreFilter must stay exact when the resolver declines to answer"
        );
    }
}

#[test]
fn exact_search_agrees_with_the_independent_ground_truth() {
    let corpus = build_corpus(600, CORPUS_DIMENSION, 33);
    let index = standard_index(&corpus);
    let queries = build_queries(6, CORPUS_DIMENSION, 34);
    let predicate = FilterPredicate::in_set("lang", ["ja", "en"]);

    for query in &queries {
        let hits = index.exact_search(query, 10, &predicate).expect("valid");
        let truth = brute_force_ground_truth(&corpus, query, 10, &predicate);
        assert_eq!(
            hits.iter().map(|hit| hit.id.clone()).collect::<Vec<_>>(),
            truth
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// InFilter recall
// ═══════════════════════════════════════════════════════════════════════════

/// Recall@10 of the ACORN traversal against the independent brute-force ground
/// truth, at 1%, 10%, 50% and 90% selectivity.
///
/// The floors below are deliberately generous relative to what the algorithm
/// actually achieves (the assertion message prints the measurement), because the
/// point is to fail loudly on a *regression*, not to pin a number that a
/// beneficial change to the graph would break. What matters is the shape: recall
/// stays high across three orders of magnitude of selectivity, which is exactly
/// what post-filtering cannot do.
#[test]
fn infilter_recall_across_selectivity_regimes() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 41);
    let index = standard_index(&corpus);
    let queries = build_queries(20, CORPUS_DIMENSION, 42);

    for (percent, floor) in [(1_usize, 0.80), (10, 0.85), (50, 0.90), (90, 0.90)] {
        let predicate = selectivity_predicate(percent);
        // The balanced labels make this exact, not approximate.
        assert_eq!(
            corpus.true_match_count(&predicate),
            CORPUS_SIZE * percent / 100
        );

        let recall = mean_recall(
            &index,
            &corpus,
            &queries,
            10,
            &predicate,
            FilterStrategy::InFilter,
        );
        assert!(
            recall >= floor,
            "InFilter recall@10 at {percent}% selectivity is {recall:.3}, below the {floor} floor"
        );
    }
}

#[test]
fn infilter_two_hop_expansion_is_adaptive() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 43);
    let index = standard_index(&corpus);
    let query = build_queries(1, CORPUS_DIMENSION, 44).remove(0);

    // At 1% selectivity, a degree-32 node has ~0.32 matching neighbors — far
    // below the `R / gamma` threshold — so essentially every expansion must
    // route through failing neighbors, or the traversal would dead-end.
    let (_, sparse) = index
        .search_with_strategy(
            &query,
            10,
            &selectivity_predicate(1),
            FilterStrategy::InFilter,
        )
        .expect("valid");
    assert!(
        sparse.two_hop_expansions > 0,
        "a 1%-selective predicate must trigger the two-hop step"
    );

    // At 90% selectivity, a node has ~29 matching neighbors — well above the
    // threshold — so the predicate subgraph navigates on its own and the
    // two-hop tax must be skipped entirely.
    let (_, dense) = index
        .search_with_strategy(
            &query,
            10,
            &selectivity_predicate(90),
            FilterStrategy::InFilter,
        )
        .expect("valid");
    assert_eq!(
        dense.two_hop_expansions, 0,
        "a 90%-selective predicate must not pay the two-hop tax"
    );

    // ...and the cost profile follows — but *where* the extra cost lands is the
    // interesting part, and it is not where one would guess.
    //
    // The two-hop tax is paid almost entirely in **predicate evaluations** (and
    // in walking neighbor lists that are already in cache), not in graph hops:
    // the number of nodes actually expanded is governed by the beam and is
    // essentially identical in both regimes. The two-hop step reaches through a
    // failing neighbor, tests the predicate on its neighbors, and computes a
    // distance only for the ones that *match* — so at 1% selectivity it tests a
    // great many nodes and buys a distance computation for hardly any of them.
    //
    // This is exactly why `neighbor_expansion_gamma` is cheap to raise, and why
    // the default is 16 rather than something timid: the marginal cost of a
    // larger gamma is predicate work, which is memoized, rather than arithmetic.
    assert!(
        sparse.predicate_evaluations > dense.predicate_evaluations,
        "the two-hop step must cost predicate work: sparse {} vs dense {}",
        sparse.predicate_evaluations,
        dense.predicate_evaluations
    );
    assert!(
        sparse.two_hop_expansions >= sparse.nodes_visited / 2,
        "at 1% selectivity nearly every expansion should route through a failing \
         neighbor: {} two-hops over {} visits",
        sparse.two_hop_expansions,
        sparse.nodes_visited
    );
}

#[test]
fn infilter_hits_always_satisfy_the_predicate() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 45);
    let index = standard_index(&corpus);
    let queries = build_queries(10, CORPUS_DIMENSION, 46);

    // Soundness is unconditional — it holds even in the regimes where recall is
    // poor and even for strategies that return short.
    for percent in [1_usize, 5, 25, 75] {
        let predicate = selectivity_predicate(percent);
        for query in &queries {
            for strategy in [
                FilterStrategy::InFilter,
                FilterStrategy::PostFilter,
                FilterStrategy::PreFilter,
            ] {
                let (hits, _) = index
                    .search_with_strategy(query, 10, &predicate, strategy)
                    .expect("valid");
                for hit in &hits {
                    let metadata = index.metadata_of(&hit.id).expect("hit is indexed");
                    assert!(
                        predicate.matches(metadata),
                        "{strategy:?} returned {} which fails the predicate",
                        hit.id
                    );
                }
            }
        }
    }
}

#[test]
fn infilter_converges_to_exactness_as_the_budget_grows() {
    // The recall argument in the module docs claims that `InFilter`'s recall is
    // a function of *budget*, not a structural ceiling — that routing nodes let
    // the frontier reach the whole connected component, so a large enough beam
    // and visit ceiling recover the exact answer. This tests that claim.
    let corpus = build_corpus(1_000, CORPUS_DIMENSION, 47);
    let queries = build_queries(6, CORPUS_DIMENSION, 48);
    let predicate = selectivity_predicate(2);

    let generous = FilteredSearchConfig {
        search_beam_width: 512,
        max_visit_multiplier: 8,
        neighbor_expansion_gamma: 8,
        ..FilteredSearchConfig::default()
    };
    let index =
        FilteredVectorIndex::build(CORPUS_DIMENSION, generous, corpus.records()).expect("valid");

    let recall = mean_recall(
        &index,
        &corpus,
        &queries,
        10,
        &predicate,
        FilterStrategy::InFilter,
    );
    assert!(
        recall >= 0.99,
        "with a generous budget InFilter should be essentially exact, got {recall:.3}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// THE HEADLINE MEASUREMENT
// ═══════════════════════════════════════════════════════════════════════════

/// **The reason this module exists.**
///
/// At 1% selectivity, a naive post-filter with a bounded over-fetch of
/// `k * m = 10 * 10 = 100` candidates can, in expectation, contain only
/// `s * k * m = 0.01 * 100 = 1` matching vector. It therefore returns roughly
/// *one* of the ten correct answers — a recall around `0.1`, and often `0.0`.
/// This is not an implementation weakness that a bigger beam would fix: the 100
/// candidates were ranked without reference to the predicate at all, so the
/// matching vectors are simply not in the list that was fetched.
///
/// `InFilter` and `PreFilter`, on the same index, same queries, same predicate,
/// against the same independently-computed ground truth, return nearly all of
/// them.
#[test]
fn headline_postfilter_collapses_where_infilter_and_prefilter_hold() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 51);
    let index = standard_index(&corpus);
    let queries = build_queries(20, CORPUS_DIMENSION, 52);
    let top_k = 10;

    let predicate = selectivity_predicate(1);
    let matching = corpus.true_match_count(&predicate);
    assert_eq!(matching, 20, "1% of 2,000 records");
    // There are 20 matches and we ask for 10, so a perfect answer exists and has
    // exactly 10 entries — recall is measured against a full-size ground truth,
    // never against a degenerate one.
    assert!(matching >= top_k);

    let post = mean_recall(
        &index,
        &corpus,
        &queries,
        top_k,
        &predicate,
        FilterStrategy::PostFilter,
    );
    let acorn = mean_recall(
        &index,
        &corpus,
        &queries,
        top_k,
        &predicate,
        FilterStrategy::InFilter,
    );
    let pre = mean_recall(
        &index,
        &corpus,
        &queries,
        top_k,
        &predicate,
        FilterStrategy::PreFilter,
    );

    // The predicted collapse: recall <= s * m = 0.01 * 10 = 0.1, plus slack for
    // the fact that the *closest* matches are slightly likelier than uniform to
    // appear near the top of an unfiltered ranking.
    assert!(
        post <= 0.30,
        "PostFilter recall@10 at 1% selectivity is {post:.3}; the whole premise of \
         this module is that a bounded over-fetch collapses here. If this now \
         passes, the fixture has acquired a correlation between the predicate and \
         the vector geometry and no longer measures what it claims to."
    );

    assert!(
        acorn >= 0.80,
        "InFilter recall@10 at 1% selectivity is {acorn:.3}, expected >= 0.80"
    );
    assert_eq!(
        pre, 1.0,
        "PreFilter is exact by construction; it must recall every ground-truth hit"
    );

    // And the point, stated as an inequality: the predicate-aware traversal is
    // not marginally better than the naive baseline, it is categorically better.
    assert!(
        acorn > post * 2.5,
        "InFilter {acorn:.3} vs PostFilter {post:.3}: the gap should be a chasm, not a nudge"
    );

    // Finally, the planner must not need to be told any of this: given a
    // 1%-selective predicate it should pick the exact scan on its own.
    let (chosen, estimate) = index.plan(&predicate, top_k).expect("valid");
    assert_eq!(
        chosen,
        FilterStrategy::PreFilter,
        "the planner should route a {:.3}-selectivity predicate to the exact scan",
        estimate.selectivity
    );
}

/// The complementary half of the headline: `PostFilter` is not *bad*, it is
/// *mis-applied*. In the regime the planner actually gives it — high
/// selectivity — it is both correct and cheap, and the planner picks it.
#[test]
fn postfilter_is_fine_in_the_regime_the_planner_gives_it() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 53);
    let index = standard_index(&corpus);
    let queries = build_queries(20, CORPUS_DIMENSION, 54);

    let predicate = selectivity_predicate(90);
    let recall = mean_recall(
        &index,
        &corpus,
        &queries,
        10,
        &predicate,
        FilterStrategy::PostFilter,
    );
    assert!(
        recall >= 0.90,
        "PostFilter recall@10 at 90% selectivity is {recall:.3}; here the over-fetch is nearly free"
    );

    let (chosen, _) = index.plan(&predicate, 10).expect("valid");
    assert_eq!(chosen, FilterStrategy::PostFilter);
}

/// The cost side of the same coin: at low selectivity `PreFilter` is not merely
/// exact, it is *cheaper* — which is why the planner's choice is not a
/// recall-for-speed trade at all.
#[test]
fn prefilter_is_cheaper_than_infilter_at_low_selectivity() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 55);
    let index = standard_index(&corpus);
    let query = build_queries(1, CORPUS_DIMENSION, 56).remove(0);
    let predicate = selectivity_predicate(1);

    let (_, pre) = index
        .search_with_strategy(&query, 10, &predicate, FilterStrategy::PreFilter)
        .expect("valid");
    let (_, acorn) = index
        .search_with_strategy(&query, 10, &predicate, FilterStrategy::InFilter)
        .expect("valid");

    // The exact scan touches ~20 vectors; the traversal touches hundreds.
    assert_eq!(pre.distance_computations, 20);
    assert_eq!(pre.nodes_visited, 0, "PreFilter never touches the graph");
    assert!(
        acorn.distance_computations > pre.distance_computations,
        "InFilter {} vs PreFilter {} distance computations",
        acorn.distance_computations,
        pre.distance_computations
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Planning integration and statistics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn auto_strategy_reports_the_plan_it_chose() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 57);
    let index = standard_index(&corpus);
    let query = build_queries(1, CORPUS_DIMENSION, 58).remove(0);

    for (percent, expected) in [
        (1_usize, FilterStrategy::PreFilter),
        (30, FilterStrategy::InFilter),
        (95, FilterStrategy::PostFilter),
    ] {
        let predicate = selectivity_predicate(percent);
        let (hits, stats) = index
            .search_with_stats(&query, 10, &predicate)
            .expect("valid");
        assert_eq!(
            stats.strategy, expected,
            "at {percent}% selectivity the planner chose {:?}",
            stats.strategy
        );
        assert_ne!(stats.strategy, FilterStrategy::Auto);
        let selectivity = stats
            .selectivity
            .expect("Auto must report the estimate that drove its choice");
        assert!((selectivity.selectivity - percent as f64 / 100.0).abs() <= SELECTIVITY_TOLERANCE);
        assert_eq!(hits.len(), 10);

        // Explicit strategies do not run the estimator at all.
        let (_, explicit) = index
            .search_with_strategy(&query, 10, &predicate, expected)
            .expect("valid");
        assert!(explicit.selectivity.is_none());
    }
}

#[test]
fn stats_count_real_work() {
    let corpus = build_corpus(CORPUS_SIZE, CORPUS_DIMENSION, 59);
    let index = standard_index(&corpus);
    let query = build_queries(1, CORPUS_DIMENSION, 60).remove(0);
    let predicate = selectivity_predicate(20);

    let (_, acorn) = index
        .search_with_strategy(&query, 10, &predicate, FilterStrategy::InFilter)
        .expect("valid");
    assert!(acorn.nodes_visited > 0);
    assert!(acorn.distance_computations > 0);
    assert!(acorn.candidates_considered > 0);
    assert!(acorn.matched_candidates > 0);
    // Memoization means the predicate is evaluated at most once per distinct
    // node; if this ever exceeded the corpus size, the cache would be broken.
    assert!(acorn.predicate_evaluations <= CORPUS_SIZE);
    // Every candidate reached the pool, so every candidate was tested.
    assert!(acorn.predicate_evaluations >= acorn.matched_candidates);
    // The visit ceiling is respected.
    assert!(acorn.nodes_visited <= index.config().max_visits());

    let (_, post) = index
        .search_with_strategy(&query, 10, &predicate, FilterStrategy::PostFilter)
        .expect("valid");
    // PostFilter over-fetches `k * m` and tests exactly those.
    assert!(post.predicate_evaluations <= 10 * index.config().post_filter_multiplier);
    assert_eq!(post.two_hop_expansions, 0, "only ACORN does two-hop work");

    let (_, pre) = index
        .search_with_strategy(&query, 10, &predicate, FilterStrategy::PreFilter)
        .expect("valid");
    // The inverted index resolved the range exactly, so PreFilter tested only
    // the ~400 candidates it handed back — not all 2,000 records.
    assert_eq!(pre.candidates_considered, 400);
    assert_eq!(pre.predicate_evaluations, 400);
    assert_eq!(pre.matched_candidates, 400);
    assert_eq!(pre.distance_computations, 400);
    assert_eq!(pre.two_hop_expansions, 0);
}

#[test]
fn record_and_hit_constructors() {
    let record = FilteredVectorRecord::new(
        "x",
        vec![1.0, 2.0],
        FilteredMetadata::new().with("a", 1_i64),
    );
    assert_eq!(record.id, "x");
    assert_eq!(record.vector, vec![1.0, 2.0]);

    let hit = FilteredHit::new("x", 0.25, 0.75, 3);
    assert_eq!(hit.id, "x");
    assert_eq!(hit.distance, 0.25);
    assert_eq!(hit.score, 0.75);
    assert_eq!(hit.rank, 3);

    assert_eq!(FilterStrategy::InFilter.name(), "in_filter");
    assert!(FilterStrategy::PreFilter.is_exact());
    assert!(!FilterStrategy::InFilter.is_exact());
    assert_eq!(FilteredDistanceMetric::Cosine.name(), "cosine");
}

#[test]
fn index_exposes_its_estimator_and_node_mapping() {
    let corpus = build_corpus(128, CORPUS_DIMENSION, 61);
    let index = standard_index(&corpus);
    assert_eq!(index.estimator().total(), 128);
    assert!(index.estimator().attribute("lang").is_some());
    assert!(index.estimator().attribute("nope").is_none());
    assert_eq!(index.node_of("doc-00000"), Some(0));
    assert_eq!(index.node_of("missing"), None);
    assert!(index.metadata_of("doc-00007").is_some());
    assert!(index.metadata_of("missing").is_none());
}
