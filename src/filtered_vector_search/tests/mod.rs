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
//! Tests for predicate-constrained approximate nearest-neighbor search.
//!
//! The suite is organized to mirror the module: the attribute model, the
//! predicate AST and its evaluator, the histogram and the selectivity
//! estimator, the strategy selector, configuration validation, the index's
//! mechanics and edge cases — and then the three measurements that are the
//! entire reason the module exists:
//!
//! 1. [`prefilter_returns_exactly_the_brute_force_top_k`] — `PreFilter` is
//!    exact, asserted as *equality* against an independently-written brute-force
//!    ground truth (not against the index's own `exact_search`, which would be
//!    circular).
//! 2. [`infilter_recall_across_selectivity_regimes`] — ACORN's recall against
//!    that same ground truth at 1%, 10%, 50% and 90% selectivity.
//! 3. [`headline_postfilter_collapses_where_infilter_and_prefilter_hold`] — the
//!    headline: at 1% selectivity a bounded-over-fetch post-filter returns
//!    almost nothing, while `InFilter` and `PreFilter` return almost everything.
//!
//! # Why the ground truth is written twice
//!
//! Every recall number below is measured against [`brute_force_ground_truth`],
//! which is implemented *here*, from the raw corpus, with its own distance
//! function — it never calls into the index. Comparing the index against its own
//! `exact_search` would prove only that two code paths agree, which is precisely
//! the kind of tautology that makes a benchmark worthless. The one thing shared
//! is [`FilterPredicate::matches`] itself, which is the *definition* of the
//! predicate rather than an implementation of the search.

use std::collections::HashSet;

use super::graph::SplitMix64;
use super::index::{FilteredVectorIndex, FilteredVectorRecord};
use super::predicate::FilterPredicate;
use super::types::{
    FilterBound, FilterStrategy, FilteredHit, FilteredMetadata, FilteredSearchConfig,
};

mod estimator_tests;
mod index_tests;
mod predicate_tests;
mod recall_tests;

// ═══════════════════════════════════════════════════════════════════════════
// Test corpus and ground truth
// ═══════════════════════════════════════════════════════════════════════════

const CORPUS_SIZE: usize = 2_000;
const CORPUS_DIMENSION: usize = 16;
const CLUSTER_COUNT: usize = 16;
const BUCKET_CARDINALITY: usize = 100;
const LANGUAGES: [&str; 4] = ["en", "ja", "fr", "de"];

/// The raw corpus, held independently of any index so that ground truth can be
/// computed without consulting one.
struct TestCorpus {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    metadata: Vec<FilteredMetadata>,
}

impl TestCorpus {
    fn records(&self) -> Vec<FilteredVectorRecord> {
        (0..self.ids.len())
            .map(|index| {
                FilteredVectorRecord::new(
                    self.ids[index].clone(),
                    self.vectors[index].clone(),
                    self.metadata[index].clone(),
                )
            })
            .collect()
    }

    /// The exact number of records satisfying `predicate` — a plain scan, so it
    /// is the truth by definition.
    fn true_match_count(&self, predicate: &FilterPredicate) -> usize {
        self.metadata
            .iter()
            .filter(|metadata| predicate.matches(metadata))
            .count()
    }

    fn true_selectivity(&self, predicate: &FilterPredicate) -> f64 {
        self.true_match_count(predicate) as f64 / self.ids.len() as f64
    }
}

/// Deterministically shuffle `0..count` labels so that each label value occurs
/// exactly `count / cardinality` times, *independently* of a record's position
/// (and therefore of its vector).
///
/// Independence matters: the whole `PostFilter` failure argument assumes the
/// predicate is uncorrelated with the query's proximity ranking. If labels were
/// assigned by position and vectors were clustered by position, the corpus would
/// quietly hand `PostFilter` a correlation it does not deserve, and the headline
/// measurement would be measuring the fixture rather than the algorithm.
fn balanced_labels(count: usize, cardinality: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut labels: Vec<usize> = (0..count).map(|index| index % cardinality).collect();
    rng.shuffle(&mut labels);
    labels
}

/// Build a clustered corpus with exactly-balanced, position-independent
/// attributes.
///
/// - `bucket`: `Int` in `0..100`, exactly `CORPUS_SIZE / 100` records each — so
///   `bucket < t` matches *exactly* `t%` of the corpus, with no sampling noise
///   to muddy a recall measurement.
/// - `lang`: `Str`, exactly a quarter of the corpus each.
/// - `score`: `Float` in `[0, 1)`.
/// - `public`: `Bool`.
/// - `tag`: `Str`, present on only 40% of records, so `Exists` / `Ne` / `Not`
///   have something to disagree about.
fn build_corpus(size: usize, dimension: usize, seed: u64) -> TestCorpus {
    let mut geometry_rng = SplitMix64::new(seed);
    let mut label_rng = SplitMix64::new(seed ^ 0xa5a5_a5a5_a5a5_a5a5);

    let centers: Vec<Vec<f32>> = (0..CLUSTER_COUNT)
        .map(|_| {
            (0..dimension)
                .map(|_| geometry_rng.next_f32() * 2.0 - 1.0)
                .collect()
        })
        .collect();

    let buckets = balanced_labels(size, BUCKET_CARDINALITY, &mut label_rng);
    let languages = balanced_labels(size, LANGUAGES.len(), &mut label_rng);
    let tagged = balanced_labels(size, 5, &mut label_rng);

    let mut ids = Vec::with_capacity(size);
    let mut vectors = Vec::with_capacity(size);
    let mut metadata = Vec::with_capacity(size);

    for index in 0..size {
        let cluster = geometry_rng.next_bounded(CLUSTER_COUNT);
        let vector: Vec<f32> = (0..dimension)
            .map(|axis| centers[cluster][axis] + (geometry_rng.next_f32() - 0.5) * 0.7)
            .collect();

        let mut record = FilteredMetadata::new()
            .with("bucket", buckets[index] as i64)
            .with("lang", LANGUAGES[languages[index]])
            .with("score", f64::from(geometry_rng.next_f32()))
            .with("public", index % 3 == 0);
        // Exactly 40% carry `tag` (labels 0 and 1 of 5).
        if tagged[index] < 2 {
            record.insert("tag", if tagged[index] == 0 { "alpha" } else { "beta" });
        }

        ids.push(format!("doc-{index:05}"));
        vectors.push(vector);
        metadata.push(record);
    }

    TestCorpus {
        ids,
        vectors,
        metadata,
    }
}

/// Cosine distance, reimplemented here so that ground truth never borrows the
/// index's arithmetic. Matches [`FilteredDistanceMetric::Cosine`]'s definition:
/// `1 - cos(a, b)`, with a zero-norm vector defined to be orthogonal to
/// everything.
fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    let left_norm: f32 = left.iter().map(|a| a * a).sum::<f32>().sqrt();
    let right_norm: f32 = right.iter().map(|b| b * b).sum::<f32>().sqrt();
    if left_norm <= 0.0 || right_norm <= 0.0 {
        return 1.0;
    }
    (1.0 - dot / (left_norm * right_norm)).clamp(0.0, 2.0)
}

/// The true top-`k` ids among the records satisfying `predicate`, best first.
///
/// A plain scan over the raw corpus. Ties break by ascending id, which — because
/// the ids are zero-padded and assigned in insertion order — agrees with the
/// index's internal ascending-node-id tie-break.
fn brute_force_ground_truth(
    corpus: &TestCorpus,
    query: &[f32],
    top_k: usize,
    predicate: &FilterPredicate,
) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = Vec::new();
    for index in 0..corpus.ids.len() {
        if !predicate.matches(&corpus.metadata[index]) {
            continue;
        }
        scored.push((
            cosine_distance(query, &corpus.vectors[index]),
            corpus.ids[index].as_str(),
        ));
    }
    scored.sort_by(|(left_distance, left_id), (right_distance, right_id)| {
        left_distance
            .partial_cmp(right_distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left_id.cmp(right_id))
    });
    scored
        .into_iter()
        .take(top_k)
        .map(|(_, id)| id.to_string())
        .collect()
}

/// `|returned ∩ truth| / |truth|`, or `1.0` when there is nothing to recall.
fn recall_at_k(returned: &[FilteredHit], truth: &[String]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let found: HashSet<&str> = returned.iter().map(|hit| hit.id.as_str()).collect();
    let hits = truth
        .iter()
        .filter(|id| found.contains(id.as_str()))
        .count();
    hits as f64 / truth.len() as f64
}

/// Deterministic query vectors drawn from the same clustered geometry as the
/// corpus, so that queries land *inside* the data rather than in empty space.
fn build_queries(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64::new(seed);
    (0..count)
        .map(|_| (0..dimension).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect()
}

/// `bucket < threshold`, which — thanks to the balanced labels — matches
/// *exactly* `threshold`% of the corpus.
fn selectivity_predicate(percent: usize) -> FilterPredicate {
    FilterPredicate::range(
        "bucket",
        FilterBound::Unbounded,
        FilterBound::Exclusive(percent as f64),
    )
}

/// Mean recall@k of one strategy over a set of queries, measured against the
/// independent brute-force ground truth.
fn mean_recall(
    index: &FilteredVectorIndex,
    corpus: &TestCorpus,
    queries: &[Vec<f32>],
    top_k: usize,
    predicate: &FilterPredicate,
    strategy: FilterStrategy,
) -> f64 {
    let mut total = 0.0;
    for query in queries {
        let (hits, _) = index
            .search_with_strategy(query, top_k, predicate, strategy)
            .expect("query and predicate are well-formed");
        let truth = brute_force_ground_truth(corpus, query, top_k, predicate);
        total += recall_at_k(&hits, &truth);
    }
    total / queries.len() as f64
}

fn standard_index(corpus: &TestCorpus) -> FilteredVectorIndex {
    FilteredVectorIndex::build(
        CORPUS_DIMENSION,
        FilteredSearchConfig::default(),
        corpus.records(),
    )
    .expect("the synthetic corpus is well-formed")
}

/// The documented accuracy contract of the estimator on this corpus: the
/// estimated selectivity is within **2 percentage points** of the truth for
/// every predicate shape exercised below.
///
/// The error has exactly two sources, both of which the module names openly: the
/// within-bucket-uniformity assumption of the equi-width histogram (a range
/// endpoint landing inside a bucket), and the independence assumption of
/// `And`/`Or`. There is no third.
const SELECTIVITY_TOLERANCE: f64 = 0.02;
