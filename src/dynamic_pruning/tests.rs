#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::needless_range_loop
)]
//! Tests for exact top-k early termination.
//!
//! The suite is organised around one load-bearing claim: **`Wand`,
//! `BlockMaxWand` and `MaxScore` return exactly what `Exhaustive` returns**,
//! and they get there by doing measurably less work. Everything else here
//! exists to support or to attack that claim.
//!
//! * `bm25` / `bounds` — the ceilings really are ceilings: no posting outruns
//!   its block's max, no block outruns its term's global max, no document
//!   outruns the sum of its terms' maxima. If these fail, every skipping proof
//!   in the module collapses.
//! * `equivalence` — randomised corpora (deterministic `SplitMix64`, no `rand`
//!   dependency) over Zipf-skewed vocabularies, varying corpus size, query
//!   length, `k` and block size, plus corpora engineered to be *saturated with
//!   ties*, which is exactly where a naive implementation silently diverges.
//! * `pruning_happens` — a strategy that returns the right answer while
//!   touching every posting is a failure. These tests assert real, measured
//!   skipping, and the headline ordering
//!   `BlockMaxWand < Wand ≪ Exhaustive` in postings scored.
//! * `edge_cases` — single-term queries, absent terms, `k` beyond the match
//!   count, the empty index, and an all-identical corpus where every score ties.

use std::collections::{HashMap, HashSet};

use crate::dynamic_pruning::cursor::{RankedEntry, ScoreAccumulator, TopKHeap, can_reach};
use crate::dynamic_pruning::index::{DynamicPruningIndex, bm25_idf, bm25_term_score};
use crate::dynamic_pruning::types::{
    DynamicPruningError, PruningConfig, PruningSearchResult, PruningStrategy,
};

/// The tolerance at which two scores from two strategies are considered equal.
/// The implementation actually guarantees bit-for-bit equality (see the
/// canonical-summation-order argument in `cursor.rs`), and
/// `equivalence::scores_are_bit_identical_across_strategies` asserts exactly
/// that; this looser figure is the *documented contract*.
const SCORE_TOLERANCE: f64 = 1e-6;

// ── Deterministic PRNG ───────────────────────────────────────────────────────

/// `SplitMix64` — a tiny, high-quality, fully deterministic generator, written
/// out here because the crate forbids a `rand` dependency and because a
/// reproducible seed is precisely what a randomised-equivalence test needs: a
/// failure must be replayable from its seed alone.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, bound)`. `bound` must be non-zero.
    fn below(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "bound must be non-zero");
        (self.next_u64() % bound as u64) as usize
    }

    /// Uniform in `[low, high]`.
    fn range(&mut self, low: usize, high: usize) -> usize {
        assert!(low <= high, "empty range");
        low + self.below(high - low + 1)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        // 53 significant bits, the full mantissa of an f64.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ── Corpus generation ────────────────────────────────────────────────────────

/// How a randomised corpus is shaped.
#[derive(Debug, Clone, Copy)]
struct CorpusShape {
    document_count: usize,
    vocabulary_size: usize,
    min_length: usize,
    max_length: usize,
    /// Zipf exponent. `0.0` is a uniform vocabulary; `1.0`+ produces the sharp
    /// head-and-tail skew of real text, where a handful of terms occur in most
    /// documents and most terms occur in a handful. Skew is what makes pruning
    /// pay, and what makes its bugs visible.
    zipf_exponent: f64,
}

/// A vocabulary whose term frequencies follow a Zipf-like law, sampled from a
/// precomputed cumulative distribution.
struct ZipfVocabulary {
    terms: Vec<String>,
    cumulative: Vec<f64>,
}

impl ZipfVocabulary {
    fn new(size: usize, exponent: f64) -> Self {
        let terms: Vec<String> = (0..size).map(|index| format!("t{index:04}")).collect();
        let weights: Vec<f64> = (0..size)
            .map(|index| 1.0 / ((index + 1) as f64).powf(exponent))
            .collect();
        let total: f64 = weights.iter().sum();
        let mut cumulative = Vec::with_capacity(size);
        let mut running = 0.0;
        for weight in &weights {
            running += weight / total;
            cumulative.push(running);
        }
        Self { terms, cumulative }
    }

    fn sample(&self, rng: &mut SplitMix64) -> &str {
        let draw = rng.unit();
        let index = self
            .cumulative
            .partition_point(|&bound| bound < draw)
            .min(self.terms.len() - 1);
        &self.terms[index]
    }
}

fn build_random_index(
    shape: CorpusShape,
    config: PruningConfig,
    seed: u64,
) -> (DynamicPruningIndex, ZipfVocabulary) {
    let mut rng = SplitMix64::new(seed);
    let vocabulary = ZipfVocabulary::new(shape.vocabulary_size, shape.zipf_exponent);
    let mut index =
        DynamicPruningIndex::new(config).expect("randomised test configuration must be valid");

    for document in 0..shape.document_count {
        let length = rng.range(shape.min_length, shape.max_length);
        let terms: Vec<String> = (0..length)
            .map(|_| vocabulary.sample(&mut rng).to_string())
            .collect();
        index
            .add_document(&format!("d{document:05}"), &terms)
            .expect("generated documents are non-empty and uniquely named");
    }
    index.build();
    (index, vocabulary)
}

/// Draw a query by sampling the vocabulary, mixing high-frequency head terms
/// with low-frequency tail terms (the case dynamic pruning is built for) and
/// occasionally an out-of-vocabulary term.
fn random_query(vocabulary: &ZipfVocabulary, length: usize, rng: &mut SplitMix64) -> Vec<String> {
    (0..length)
        .map(|_| {
            if rng.below(20) == 0 {
                // 5% out-of-vocabulary: a term with no posting list at all.
                format!("absent{}", rng.below(1000))
            } else {
                vocabulary.sample(rng).to_string()
            }
        })
        .collect()
}

// ── Cross-strategy comparison ────────────────────────────────────────────────

/// Run every strategy and assert they all agree with `Exhaustive`, exactly.
///
/// Returns the four results, keyed by strategy, so callers can go on to make
/// claims about the *work* each of them did.
fn assert_all_strategies_agree(
    index: &DynamicPruningIndex,
    query: &[String],
    top_k: usize,
    context: &str,
) -> HashMap<PruningStrategy, PruningSearchResult> {
    let mut results = HashMap::new();
    for strategy in PruningStrategy::all() {
        let result = index
            .search_top_k(query, strategy, top_k)
            .unwrap_or_else(|error| panic!("{context}: {strategy:?} failed: {error}"));
        results.insert(strategy, result);
    }

    let baseline = results
        .get(&PruningStrategy::Exhaustive)
        .expect("exhaustive result was just inserted");

    for strategy in PruningStrategy::all() {
        if strategy == PruningStrategy::Exhaustive {
            continue;
        }
        let candidate = results
            .get(&strategy)
            .expect("every strategy's result was just inserted");

        assert_eq!(
            candidate.hits.len(),
            baseline.hits.len(),
            "{context}: {strategy:?} returned {} hits, exhaustive returned {}",
            candidate.hits.len(),
            baseline.hits.len()
        );

        for (rank, (got, want)) in candidate.hits.iter().zip(baseline.hits.iter()).enumerate() {
            assert_eq!(
                got.document_id, want.document_id,
                "{context}: {strategy:?} put {} at rank {rank}, exhaustive put {} \
                 (scores {} vs {}); the pruning is not exact",
                got.document_id, want.document_id, got.score, want.score
            );
            assert_eq!(
                got.ordinal, want.ordinal,
                "{context}: {strategy:?} disagreed on the ordinal at rank {rank}"
            );
            assert!(
                (got.score - want.score).abs() <= SCORE_TOLERANCE,
                "{context}: {strategy:?} scored {} as {}, exhaustive scored it {}",
                got.document_id,
                got.score,
                want.score
            );
        }

        // Every strategy must account for exactly the same universe of work.
        assert_eq!(
            candidate.stats.total_postings, baseline.stats.total_postings,
            "{context}: {strategy:?} disagreed on the query's posting count"
        );
        assert_eq!(
            candidate.stats.postings_scored + candidate.stats.postings_skipped,
            candidate.stats.total_postings,
            "{context}: {strategy:?} scored + skipped must equal total"
        );
        assert!(
            candidate.stats.postings_scored <= baseline.stats.postings_scored,
            "{context}: {strategy:?} scored {} postings, more than exhaustive's {}",
            candidate.stats.postings_scored,
            baseline.stats.postings_scored
        );
    }

    assert_eq!(
        baseline.stats.postings_skipped, 0,
        "{context}: the exhaustive baseline must never skip anything"
    );
    assert_eq!(
        baseline.stats.postings_scored, baseline.stats.total_postings,
        "{context}: the exhaustive baseline must score every posting"
    );

    results
}

// ── bm25 ─────────────────────────────────────────────────────────────────────

mod bm25 {
    use super::{bm25_idf, bm25_term_score};

    #[test]
    fn idf_is_positive_even_for_a_term_in_every_document() {
        // The `1 +` guard in ln(1 + (N - df + 0.5)/(df + 0.5)) is not cosmetic:
        // a negative contribution would destroy the monotonicity the ceilings
        // depend on, because adding a term could then *lower* a score.
        let idf = bm25_idf(100, 100);
        assert!(idf > 0.0, "idf must stay positive, got {idf}");
    }

    #[test]
    fn idf_decreases_with_document_frequency() {
        let rare = bm25_idf(1000, 1);
        let common = bm25_idf(1000, 900);
        assert!(rare > common, "rare {rare} must outrank common {common}");
    }

    #[test]
    fn term_score_increases_with_term_frequency() {
        let idf = bm25_idf(1000, 10);
        let low = bm25_term_score(idf, 1, 100, 100.0, 1.2, 0.75);
        let high = bm25_term_score(idf, 5, 100, 100.0, 1.2, 0.75);
        assert!(high > low, "tf=5 ({high}) must beat tf=1 ({low})");
    }

    #[test]
    fn term_score_decreases_with_document_length() {
        let idf = bm25_idf(1000, 10);
        let short = bm25_term_score(idf, 3, 50, 100.0, 1.2, 0.75);
        let long = bm25_term_score(idf, 3, 400, 100.0, 1.2, 0.75);
        assert!(short > long, "short {short} must beat long {long}");
    }

    #[test]
    fn term_score_saturates_towards_idf_times_k1_plus_one() {
        let idf = bm25_idf(1000, 10);
        let k1 = 1.2;
        let ceiling = idf * (k1 + 1.0);
        let huge = bm25_term_score(idf, 100_000, 100, 100.0, k1, 0.75);
        assert!(huge < ceiling, "BM25 must stay below idf*(k1+1)");
        assert!(
            huge > ceiling * 0.999,
            "with tf = 100000 the score should be all but saturated"
        );
    }

    #[test]
    fn b_zero_disables_length_normalisation() {
        let idf = bm25_idf(1000, 10);
        let short = bm25_term_score(idf, 3, 10, 100.0, 1.2, 0.0);
        let long = bm25_term_score(idf, 3, 900, 100.0, 1.2, 0.0);
        assert_eq!(short, long, "with b = 0 the length must not matter at all");
    }

    #[test]
    fn hand_computed_score_matches() {
        // N = 3, df = 2, tf = 2, |d| = 3, avgdl = 8/3, k1 = 1.2, b = 0.75.
        let idf: f64 = (1.0f64 + (3.0 - 2.0 + 0.5) / (2.0 + 0.5)).ln();
        let expected_idf = 1.6f64.ln();
        assert!((idf - expected_idf).abs() < 1e-12);

        let avgdl = 8.0 / 3.0;
        let denominator = 2.0 + 1.2 * (1.0 - 0.75 + 0.75 * (3.0 / avgdl));
        let expected = idf * (2.0 * 2.2) / denominator;
        let actual = bm25_term_score(idf, 2, 3, avgdl, 1.2, 0.75);
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }
}

// ── bounds ───────────────────────────────────────────────────────────────────

mod bounds {
    use std::collections::HashMap;

    use super::{
        CorpusShape, DynamicPruningIndex, PruningConfig, PruningStrategy, SplitMix64,
        assert_all_strategies_agree, build_random_index, random_query,
    };
    use crate::dynamic_pruning::PruningPostingList;

    /// Every stored posting score must lie at or below its block's max, and
    /// every block's max at or below the term's global max. These hold *exactly*
    /// in f64 -- both ceilings are maxima over the very same stored values, not
    /// analytic over-estimates -- so the assertions use no tolerance at all.
    #[test]
    fn no_posting_ever_exceeds_its_block_max_or_its_term_max() {
        let shape = CorpusShape {
            document_count: 400,
            vocabulary_size: 60,
            min_length: 4,
            max_length: 40,
            zipf_exponent: 1.1,
        };
        let config = PruningConfig {
            block_size: 8,
            ..PruningConfig::default()
        };
        let (index, vocabulary) = build_random_index(shape, config, 0x5EED_0001);

        let mut checked_blocks = 0usize;
        let mut checked_postings = 0usize;
        for term in &vocabulary.terms {
            let Some(list) = index.posting_list(term) else {
                continue;
            };
            assert!(
                !list.blocks().is_empty(),
                "a non-empty list must carry block metadata"
            );

            let mut covered = 0usize;
            for block in list.blocks() {
                checked_blocks += 1;
                assert!(
                    block.max_score() <= list.max_score(),
                    "block max {} exceeded term max {}",
                    block.max_score(),
                    list.max_score()
                );
                assert_eq!(
                    block.start(),
                    covered,
                    "blocks must tile the list contiguously"
                );
                covered = block.end();

                for position in block.start()..block.end() {
                    let posting = list
                        .posting(position)
                        .expect("position lies inside the list");
                    checked_postings += 1;
                    assert!(
                        posting.score <= block.max_score(),
                        "posting score {} exceeded block max {}",
                        posting.score,
                        block.max_score()
                    );
                    assert!(
                        posting.score <= list.max_score(),
                        "posting score {} exceeded term max {}",
                        posting.score,
                        list.max_score()
                    );
                    assert!(
                        posting.document_ordinal <= block.max_document_ordinal(),
                        "posting ordinal exceeded the block's max ordinal"
                    );
                    assert!(
                        posting.document_ordinal >= block.first_document_ordinal(),
                        "posting ordinal preceded the block's first ordinal"
                    );
                }
            }
            assert_eq!(covered, list.len(), "blocks must cover the whole list");

            // Each ceiling must be *attained*, not merely respected: a ceiling
            // that is never touched is a loose bound and would prune less than
            // it should.
            let attained = list
                .scores()
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(
                attained,
                list.max_score(),
                "the term's global ceiling must be attained by some posting"
            );
        }
        assert!(checked_blocks > 50, "the fixture must exercise many blocks");
        assert!(
            checked_postings > 1000,
            "the fixture must exercise many postings"
        );
    }

    #[test]
    fn posting_lists_are_strictly_ascending_in_document_ordinal() {
        let shape = CorpusShape {
            document_count: 300,
            vocabulary_size: 40,
            min_length: 3,
            max_length: 25,
            zipf_exponent: 0.9,
        };
        let (index, vocabulary) = build_random_index(shape, PruningConfig::default(), 0x5EED_0002);

        for term in &vocabulary.terms {
            let Some(list) = index.posting_list(term) else {
                continue;
            };
            let ordinals = list.document_ordinals();
            for window in ordinals.windows(2) {
                assert!(
                    window[0] < window[1],
                    "posting lists must be strictly ascending: {} then {}",
                    window[0],
                    window[1]
                );
            }
            // And each block's max ordinal is its last -- the property that lets
            // a shallow advance identify "the only block that could contain d".
            for block in list.blocks() {
                assert_eq!(block.max_document_ordinal(), ordinals[block.end() - 1]);
                assert_eq!(block.first_document_ordinal(), ordinals[block.start()]);
            }
        }
    }

    /// The bound that WAND's pivot rule actually leans on: a document's true
    /// score never exceeds the sum of the global ceilings of the query terms it
    /// contains. Checked against every document in the corpus, for many queries.
    #[test]
    fn no_document_score_exceeds_the_sum_of_its_terms_ceilings() {
        let shape = CorpusShape {
            document_count: 500,
            vocabulary_size: 80,
            min_length: 5,
            max_length: 35,
            zipf_exponent: 1.0,
        };
        let (index, vocabulary) = build_random_index(shape, PruningConfig::default(), 0x5EED_0003);
        let mut rng = SplitMix64::new(0xC0FFEE);

        for _ in 0..40 {
            let query = random_query(&vocabulary, rng.range(2, 7), &mut rng);

            // Fold duplicate query terms exactly as the engine does (a term
            // occurring twice has twice the ceiling), then take each distinct
            // term's global ceiling.
            let mut ceilings: HashMap<String, f64> = HashMap::new();
            for term in &query {
                let per_occurrence = index
                    .posting_list(term)
                    .map_or(0.0, PruningPostingList::max_score);
                *ceilings.entry(term.clone()).or_insert(0.0) += per_occurrence;
            }

            // Take the true top-k (large k, so we see many real scores) and
            // check each against the ceiling of the terms it actually contains.
            let result = index
                .search_top_k(&query, PruningStrategy::Exhaustive, 500)
                .expect("query is well formed");

            for hit in &result.hits {
                let mut ceiling = 0.0f64;
                for (term, term_ceiling) in &ceilings {
                    let contains = index
                        .posting_list(term)
                        .is_some_and(|list| list.document_ordinals().contains(&hit.ordinal));
                    if contains {
                        ceiling += *term_ceiling;
                    }
                }
                assert!(
                    hit.score <= ceiling + 1e-9,
                    "document {} scored {} but its terms' ceilings sum to only {}",
                    hit.document_id,
                    hit.score,
                    ceiling
                );
                assert!(hit.score > 0.0, "a matching document must score above zero");
            }
        }

        // And the same corpus must, of course, still answer exactly.
        let mut rng = SplitMix64::new(0xBEEF);
        let query = random_query(&vocabulary, 4, &mut rng);
        assert_all_strategies_agree(&index, &query, 10, "bounds fixture");
    }

    /// Sanity: a `DynamicPruningIndex` reports the corpus statistics BM25 is
    /// normalising against.
    #[test]
    fn corpus_statistics_are_exposed_and_correct() {
        let mut index =
            DynamicPruningIndex::new(PruningConfig::default()).expect("default config is valid");
        index.add_document("a", ["x", "y"]).expect("valid document");
        index
            .add_document("b", ["x", "y", "z", "z"])
            .expect("valid document");
        index.build();

        assert_eq!(index.document_count(), 2);
        assert_eq!(index.term_count(), 3);
        assert_eq!(index.average_document_length(), 3.0);
        assert_eq!(index.document_length(0), Some(2));
        assert_eq!(index.document_length(1), Some(4));
        assert_eq!(index.ordinal_of("b"), Some(1));
        assert_eq!(index.document_id(1), Some("b"));

        let zeds = index.posting_list("z").expect("z is indexed");
        assert_eq!(zeds.len(), 1);
        let posting = zeds.posting(0).expect("one posting");
        assert_eq!(posting.document_ordinal, 1);
        assert_eq!(posting.term_frequency, 2);
    }
}

// ── equivalence ──────────────────────────────────────────────────────────────

mod equivalence {
    use super::{
        CorpusShape, DynamicPruningIndex, PruningConfig, PruningStrategy, SplitMix64,
        assert_all_strategies_agree, build_random_index, random_query,
    };

    /// The headline test. Many seeds, many corpus shapes, many block sizes, many
    /// query lengths, many `k` -- and in every single combination all three
    /// pruning strategies must reproduce the exhaustive ranking exactly.
    #[test]
    fn all_strategies_match_exhaustive_across_randomised_corpora() {
        let shapes = [
            CorpusShape {
                document_count: 60,
                vocabulary_size: 12,
                min_length: 2,
                max_length: 8,
                zipf_exponent: 0.0, // uniform vocabulary
            },
            CorpusShape {
                document_count: 250,
                vocabulary_size: 40,
                min_length: 3,
                max_length: 30,
                zipf_exponent: 0.7,
            },
            CorpusShape {
                document_count: 900,
                vocabulary_size: 120,
                min_length: 6,
                max_length: 60,
                zipf_exponent: 1.3, // sharply skewed, like real text
            },
            CorpusShape {
                document_count: 400,
                vocabulary_size: 6, // tiny vocabulary => huge, dense lists
                min_length: 4,
                max_length: 20,
                zipf_exponent: 0.4,
            },
        ];
        let block_sizes = [1usize, 2, 7, 64, 128, 4096];
        let top_ks = [1usize, 2, 3, 5, 10, 25, 100];

        let mut comparisons = 0usize;
        for (shape_index, shape) in shapes.iter().enumerate() {
            for &block_size in &block_sizes {
                let config = PruningConfig {
                    block_size,
                    ..PruningConfig::default()
                };
                let seed = 0xA5A5_0000 ^ ((shape_index as u64) << 16) ^ block_size as u64;
                let (index, vocabulary) = build_random_index(*shape, config, seed);

                let mut rng = SplitMix64::new(seed ^ 0xDEAD_BEEF);
                for trial in 0..12 {
                    let query_length = rng.range(1, 8);
                    let query = random_query(&vocabulary, query_length, &mut rng);
                    let top_k = top_ks[rng.below(top_ks.len())];
                    let context = format!(
                        "shape {shape_index}, block_size {block_size}, trial {trial}, \
                         k {top_k}, query {query:?}"
                    );
                    assert_all_strategies_agree(&index, &query, top_k, &context);
                    comparisons += 1;
                }
            }
        }
        assert!(
            comparisons >= 250,
            "the sweep must actually be broad; ran only {comparisons}"
        );
    }

    /// Ties are where naive implementations silently diverge, so build a corpus
    /// *saturated* with them: a tiny vocabulary and a single document length,
    /// so vast numbers of documents share a bit-for-bit identical score. Only a
    /// deterministic tie-break rule -- descending score, then *ascending
    /// ordinal* -- can make four different traversal orders agree here.
    #[test]
    fn strategies_agree_on_corpora_saturated_with_ties() {
        // Every document is one of 8 distinct term-multisets, repeated 40 times,
        // all of the same length. Documents sharing a multiset score identically
        // for every query, to the last bit.
        let patterns: [&[&str]; 8] = [
            &["a", "b", "c", "d"],
            &["a", "a", "b", "c"],
            &["b", "b", "c", "d"],
            &["a", "b", "b", "d"],
            &["c", "c", "d", "d"],
            &["a", "c", "c", "d"],
            &["a", "b", "c", "c"],
            &["b", "c", "d", "d"],
        ];

        for &block_size in &[1usize, 3, 64] {
            for &top_k in &[1usize, 2, 5, 9, 17, 40, 100, 320] {
                let config = PruningConfig {
                    block_size,
                    ..PruningConfig::default()
                };
                let mut index = DynamicPruningIndex::new(config).expect("valid config");
                for repetition in 0..40 {
                    for (pattern_index, pattern) in patterns.iter().enumerate() {
                        let id = format!("r{repetition:02}p{pattern_index}");
                        index.add_document(&id, *pattern).expect("valid document");
                    }
                }
                index.build();

                for query in [
                    vec!["a".to_string()],
                    vec!["a".to_string(), "b".to_string()],
                    vec!["a".to_string(), "b".to_string(), "c".to_string()],
                    vec![
                        "a".to_string(),
                        "b".to_string(),
                        "c".to_string(),
                        "d".to_string(),
                    ],
                    vec!["d".to_string(), "a".to_string()],
                ] {
                    let context =
                        format!("ties: block_size {block_size}, k {top_k}, query {query:?}");
                    let results = assert_all_strategies_agree(&index, &query, top_k, &context);

                    // And the tie-break really is "lowest ordinal wins": within
                    // any run of equal scores, ordinals must ascend.
                    let hits = &results
                        .get(&PruningStrategy::Exhaustive)
                        .expect("exhaustive ran")
                        .hits;
                    for window in hits.windows(2) {
                        let (left, right) = (&window[0], &window[1]);
                        assert!(
                            left.score > right.score
                                || (left.score == right.score && left.ordinal < right.ordinal),
                            "{context}: ranking violated the total order at {left:?} / {right:?}"
                        );
                    }
                }
            }
        }
    }

    /// The canonical-summation-order machinery promises more than the documented
    /// `1e-6`: it promises *bit-for-bit* identical scores across strategies.
    /// Hold it to that, because it is exactly what keeps tied documents from
    /// being reordered.
    #[test]
    fn scores_are_bit_identical_across_strategies() {
        let shape = CorpusShape {
            document_count: 700,
            vocabulary_size: 50,
            min_length: 5,
            max_length: 45,
            zipf_exponent: 1.1,
        };
        let (index, vocabulary) = build_random_index(shape, PruningConfig::default(), 0x1234_5678);
        let mut rng = SplitMix64::new(0x8765_4321);

        let mut compared = 0usize;
        for _ in 0..60 {
            let query = random_query(&vocabulary, rng.range(2, 6), &mut rng);
            let baseline = index
                .search_top_k(&query, PruningStrategy::Exhaustive, 20)
                .expect("valid query");
            for strategy in [
                PruningStrategy::Wand,
                PruningStrategy::BlockMaxWand,
                PruningStrategy::MaxScore,
            ] {
                let pruned = index
                    .search_top_k(&query, strategy, 20)
                    .expect("valid query");
                for (got, want) in pruned.hits.iter().zip(baseline.hits.iter()) {
                    assert_eq!(
                        got.score.to_bits(),
                        want.score.to_bits(),
                        "{strategy:?} scored {} as {:?}, exhaustive as {:?}; the canonical \
                         summation order is not being honoured",
                        got.document_id,
                        got.score,
                        want.score
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 500, "must actually have compared many scores");
    }

    /// Duplicate query terms are folded into one cursor with a `qtf` weight.
    /// Every ceiling is scaled by the same weight, so the pruning must stay
    /// exact -- and the repeated term must genuinely count twice.
    #[test]
    fn repeated_query_terms_are_weighted_and_still_exact() {
        let shape = CorpusShape {
            document_count: 300,
            vocabulary_size: 25,
            min_length: 4,
            max_length: 20,
            zipf_exponent: 0.8,
        };
        let (index, vocabulary) = build_random_index(shape, PruningConfig::default(), 0x00DD_BA11);

        let head = vocabulary.terms[0].clone();
        let tail = vocabulary.terms[20].clone();

        let once = vec![head.clone(), tail.clone()];
        let twice = vec![head.clone(), tail.clone(), head.clone()];

        assert_all_strategies_agree(&index, &once, 10, "duplicate terms: once");
        let doubled = assert_all_strategies_agree(&index, &twice, 10, "duplicate terms: twice");

        // Repeating a term must change the ranking's arithmetic, not just be
        // silently dropped: the top hit's score must move.
        let single = index
            .search_top_k(&once, PruningStrategy::Exhaustive, 10)
            .expect("valid query");
        let double = doubled
            .get(&PruningStrategy::Exhaustive)
            .expect("exhaustive ran");
        assert!(
            single.hits[0].score != double.hits[0].score,
            "a repeated query term must actually be weighted"
        );

        // The posting universe is unchanged: duplicates are folded, not appended.
        assert_eq!(
            single.stats.total_postings, double.stats.total_postings,
            "duplicate query terms must fold into a single cursor"
        );
    }

    /// Adding a document invalidates every ceiling (idf and avgdl are
    /// corpus-wide), so a stale index must refuse to answer rather than answer
    /// wrongly -- and after a rebuild, all four strategies must agree again.
    #[test]
    fn rebuilding_after_mutation_keeps_every_strategy_exact() {
        let mut index =
            DynamicPruningIndex::new(PruningConfig::default()).expect("default config is valid");
        let mut rng = SplitMix64::new(0xFEED_FACE);
        let vocabulary = super::ZipfVocabulary::new(20, 1.0);

        for round in 0..5 {
            for document in 0..50 {
                let length = rng.range(3, 15);
                let terms: Vec<String> = (0..length)
                    .map(|_| vocabulary.sample(&mut rng).to_string())
                    .collect();
                index
                    .add_document(&format!("r{round}d{document:03}"), &terms)
                    .expect("valid document");
            }
            assert!(!index.is_built(), "adding a document must dirty the index");
            let query = random_query(&vocabulary, 3, &mut rng);
            assert!(
                matches!(
                    index.search(&query),
                    Err(super::DynamicPruningError::IndexNotBuilt)
                ),
                "a dirty index must refuse to answer"
            );

            index.build();
            assert!(index.is_built());
            assert_all_strategies_agree(&index, &query, 7, &format!("round {round}"));
        }
        assert_eq!(index.document_count(), 250);
    }
}

// ── pruning_happens ──────────────────────────────────────────────────────────

mod pruning_happens {
    use super::{
        CorpusShape, PruningConfig, PruningStrategy, SplitMix64, assert_all_strategies_agree,
        build_random_index,
    };

    /// A corpus with the head-and-tail skew of real text, plus a query mixing a
    /// very common term with rarer ones. This is the setting dynamic pruning
    /// exists for, and it is where the counters must show the family ordering:
    ///
    /// ```text
    /// postings_scored:  BlockMaxWand  <  Wand  <<  Exhaustive
    /// ```
    #[test]
    fn block_max_wand_scores_strictly_fewer_postings_than_wand_on_a_skewed_corpus() {
        let shape = CorpusShape {
            document_count: 5000,
            vocabulary_size: 200,
            min_length: 10,
            max_length: 80,
            zipf_exponent: 1.4,
        };
        let config = PruningConfig {
            block_size: 32,
            ..PruningConfig::default()
        };
        let (index, vocabulary) = build_random_index(shape, config, 0x9E37_79B9);

        // t0000 is the Zipf head (in nearly every document); t0040 / t0090 are
        // progressively rarer.
        let query = vec![
            vocabulary.terms[0].clone(),
            vocabulary.terms[40].clone(),
            vocabulary.terms[90].clone(),
        ];

        let results = assert_all_strategies_agree(&index, &query, 10, "skewed corpus");

        let exhaustive = results[&PruningStrategy::Exhaustive].stats;
        let wand = results[&PruningStrategy::Wand].stats;
        let block_max = results[&PruningStrategy::BlockMaxWand].stats;
        let max_score = results[&PruningStrategy::MaxScore].stats;

        // Every pruning strategy must actually prune.
        assert!(
            wand.postings_skipped > 0,
            "WAND skipped nothing at all: {wand:?}"
        );
        assert!(
            block_max.postings_skipped > 0,
            "BlockMaxWand skipped nothing at all: {block_max:?}"
        );
        assert!(
            max_score.postings_skipped > 0,
            "MaxScore skipped nothing at all: {max_score:?}"
        );

        // ... and materially, not marginally.
        assert!(
            wand.postings_scored * 2 < exhaustive.postings_scored,
            "WAND scored {} of {} postings -- less than a 2x saving is not pruning",
            wand.postings_scored,
            exhaustive.postings_scored
        );
        assert!(
            block_max.postings_scored * 2 < exhaustive.postings_scored,
            "BlockMaxWand scored {} of {} postings",
            block_max.postings_scored,
            exhaustive.postings_scored
        );
        assert!(
            max_score.postings_scored * 2 < exhaustive.postings_scored,
            "MaxScore scored {} of {} postings",
            max_score.postings_scored,
            exhaustive.postings_scored
        );

        // The headline: block-max metadata buys strictly more skipping than the
        // global per-term ceilings alone.
        assert!(
            block_max.postings_scored < wand.postings_scored,
            "BlockMaxWand scored {} postings, WAND scored {} -- block-max bought nothing",
            block_max.postings_scored,
            wand.postings_scored
        );
        assert!(
            block_max.full_evaluations < wand.full_evaluations,
            "BlockMaxWand fully evaluated {} documents, WAND {}",
            block_max.full_evaluations,
            wand.full_evaluations
        );
        assert!(
            block_max.blocks_skipped > 0,
            "BlockMaxWand must record whole-block skips"
        );

        // Only BlockMaxWand consults block-level ceilings.
        assert_eq!(exhaustive.blocks_skipped, 0);
        assert_eq!(wand.blocks_skipped, 0);
        assert_eq!(max_score.blocks_skipped, 0);

        // Exhaustive is, by construction, the pathological case.
        assert_eq!(exhaustive.postings_scored, exhaustive.total_postings);
        assert_eq!(exhaustive.skip_ratio(), 0.0);
        assert!(block_max.skip_ratio() > 0.5, "{block_max:?}");
    }

    /// The saving must survive a broad sweep, not just one lucky fixture: over
    /// many skewed corpora and many queries, every pruning strategy must skip,
    /// and BlockMaxWand must never do *more* full evaluations than WAND.
    #[test]
    fn pruning_is_pervasive_not_incidental() {
        let shape = CorpusShape {
            document_count: 2000,
            vocabulary_size: 150,
            min_length: 8,
            max_length: 60,
            zipf_exponent: 1.2,
        };
        let config = PruningConfig {
            block_size: 32,
            ..PruningConfig::default()
        };
        let (index, vocabulary) = build_random_index(shape, config, 0x1357_9BDF);
        let mut rng = SplitMix64::new(0x2468_ACE0);

        let mut wand_total = 0u64;
        let mut block_max_total = 0u64;
        let mut exhaustive_total = 0u64;
        let mut max_score_total = 0u64;
        let mut block_max_beat_wand = 0usize;
        let mut trials = 0usize;

        for _ in 0..40 {
            // Always include the Zipf head, which is what makes exhaustive
            // scanning ruinous and pruning worthwhile.
            let mut query = vec![vocabulary.terms[0].clone()];
            for _ in 0..rng.range(1, 3) {
                query.push(vocabulary.terms[rng.range(1, 149)].clone());
            }
            let top_k = [5usize, 10, 20][rng.below(3)];
            let context = format!("sweep: k {top_k}, query {query:?}");
            let results = assert_all_strategies_agree(&index, &query, top_k, &context);

            let exhaustive = results[&PruningStrategy::Exhaustive].stats;
            let wand = results[&PruningStrategy::Wand].stats;
            let block_max = results[&PruningStrategy::BlockMaxWand].stats;
            let max_score = results[&PruningStrategy::MaxScore].stats;

            assert!(wand.postings_skipped > 0, "{context}: WAND skipped nothing");
            assert!(
                block_max.postings_skipped > 0,
                "{context}: BlockMaxWand skipped nothing"
            );
            assert!(
                max_score.postings_skipped > 0,
                "{context}: MaxScore skipped nothing"
            );
            assert!(
                block_max.full_evaluations <= wand.full_evaluations,
                "{context}: block-max ceilings are tighter than global ones, so BlockMaxWand \
                 can never evaluate more documents than WAND ({} vs {})",
                block_max.full_evaluations,
                wand.full_evaluations
            );
            if block_max.postings_scored < wand.postings_scored {
                block_max_beat_wand += 1;
            }

            exhaustive_total += exhaustive.postings_scored;
            wand_total += wand.postings_scored;
            block_max_total += block_max.postings_scored;
            max_score_total += max_score.postings_scored;
            trials += 1;
        }

        assert_eq!(trials, 40);
        assert!(
            block_max_total < wand_total,
            "aggregate: BlockMaxWand {block_max_total} must beat WAND {wand_total}"
        );
        assert!(
            wand_total < exhaustive_total,
            "aggregate: WAND {wand_total} must beat exhaustive {exhaustive_total}"
        );
        assert!(
            max_score_total < exhaustive_total,
            "aggregate: MaxScore {max_score_total} must beat exhaustive {exhaustive_total}"
        );
        assert!(
            block_max_beat_wand * 2 > trials,
            "block-max must win the majority of queries outright, won {block_max_beat_wand}/{trials}"
        );
    }

    /// Smaller blocks give tighter ceilings and therefore more skipping. If this
    /// ordering ever inverted, the block metadata would not be doing what the
    /// bound argument claims.
    #[test]
    fn tighter_blocks_prune_harder() {
        let shape = CorpusShape {
            document_count: 3000,
            vocabulary_size: 100,
            min_length: 10,
            max_length: 50,
            zipf_exponent: 1.3,
        };

        let mut scored_by_block_size = Vec::new();
        for block_size in [8usize, 64, 1024] {
            let config = PruningConfig {
                block_size,
                ..PruningConfig::default()
            };
            let (index, vocabulary) = build_random_index(shape, config, 0xB10C_5123);
            let query = vec![
                vocabulary.terms[0].clone(),
                vocabulary.terms[25].clone(),
                vocabulary.terms[60].clone(),
            ];
            let results = assert_all_strategies_agree(
                &index,
                &query,
                10,
                &format!("block sweep {block_size}"),
            );
            scored_by_block_size.push((
                block_size,
                results[&PruningStrategy::BlockMaxWand]
                    .stats
                    .postings_scored,
            ));
        }

        let tight = scored_by_block_size[0].1;
        let medium = scored_by_block_size[1].1;
        let loose = scored_by_block_size[2].1;
        assert!(
            tight <= medium && medium <= loose,
            "tighter blocks must prune at least as hard: {scored_by_block_size:?}"
        );
        assert!(
            tight < loose,
            "block size 8 must strictly beat block size 1024: {scored_by_block_size:?}"
        );
    }

    /// A smaller `k` keeps the threshold higher and therefore prunes harder.
    #[test]
    fn smaller_k_prunes_harder() {
        let shape = CorpusShape {
            document_count: 4000,
            vocabulary_size: 120,
            min_length: 10,
            max_length: 60,
            zipf_exponent: 1.35,
        };
        let config = PruningConfig {
            block_size: 32,
            ..PruningConfig::default()
        };
        let (index, vocabulary) = build_random_index(shape, config, 0x0E2E_5EED);
        let query = vec![
            vocabulary.terms[0].clone(),
            vocabulary.terms[30].clone(),
            vocabulary.terms[70].clone(),
        ];

        let small = index
            .search_top_k(&query, PruningStrategy::BlockMaxWand, 1)
            .expect("valid query");
        let large = index
            .search_top_k(&query, PruningStrategy::BlockMaxWand, 200)
            .expect("valid query");

        assert!(
            small.stats.postings_scored < large.stats.postings_scored,
            "k=1 scored {} postings, k=200 scored {} -- a tighter threshold must prune harder",
            small.stats.postings_scored,
            large.stats.postings_scored
        );
    }
}

// ── edge_cases ───────────────────────────────────────────────────────────────

mod edge_cases {
    use super::{
        DynamicPruningError, DynamicPruningIndex, PruningConfig, PruningStrategy,
        assert_all_strategies_agree,
    };

    fn tiny_index(block_size: usize) -> DynamicPruningIndex {
        let config = PruningConfig {
            block_size,
            ..PruningConfig::default()
        };
        let mut index = DynamicPruningIndex::new(config).expect("valid config");
        index
            .add_document("alpha", ["rust", "memory", "safety"])
            .expect("valid");
        index
            .add_document("beta", ["rust", "rust", "speed"])
            .expect("valid");
        index
            .add_document("gamma", ["python", "safety"])
            .expect("valid");
        index
            .add_document("delta", ["rust", "safety", "safety", "speed"])
            .expect("valid");
        index.build();
        index
    }

    #[test]
    fn single_term_query() {
        let index = tiny_index(2);
        let query = vec!["rust".to_string()];
        for top_k in [1usize, 2, 3, 10] {
            assert_all_strategies_agree(&index, &query, top_k, "single-term query");
        }
        let result = index
            .search_with(&query, PruningStrategy::MaxScore)
            .expect("valid");
        // Three documents contain "rust".
        assert_eq!(result.len(), 3);
        assert_eq!(result.stats.total_postings, 3);
    }

    #[test]
    fn term_absent_from_the_index_contributes_nothing() {
        let index = tiny_index(4);

        // A query made *entirely* of absent terms matches nothing.
        let missing = vec!["haskell".to_string(), "monad".to_string()];
        for strategy in PruningStrategy::all() {
            let result = index
                .search_with(&missing, strategy)
                .expect("an absent term is not an error");
            assert!(result.is_empty(), "{strategy:?} must return no hits");
            assert_eq!(result.stats.total_postings, 0);
            assert_eq!(result.stats.postings_scored, 0);
            assert_eq!(result.stats.postings_skipped, 0);
        }

        // A query mixing present and absent terms must behave exactly as if the
        // absent one were not there.
        let mixed = vec!["rust".to_string(), "haskell".to_string()];
        let only = vec!["rust".to_string()];
        assert_all_strategies_agree(&index, &mixed, 4, "mixed present/absent");

        let with_absent = index
            .search_with(&mixed, PruningStrategy::BlockMaxWand)
            .expect("valid");
        let without = index
            .search_with(&only, PruningStrategy::BlockMaxWand)
            .expect("valid");
        assert_eq!(with_absent.document_ids(), without.document_ids());
        assert_eq!(
            with_absent.hits[0].score.to_bits(),
            without.hits[0].score.to_bits()
        );
    }

    #[test]
    fn k_larger_than_the_number_of_matching_documents() {
        let index = tiny_index(64);
        let query = vec!["python".to_string()];
        let results = assert_all_strategies_agree(&index, &query, 1000, "k > matches");
        for strategy in PruningStrategy::all() {
            assert_eq!(
                results[&strategy].len(),
                1,
                "{strategy:?}: only one document contains \"python\""
            );
            // The heap never fills, so the threshold never exists and nothing
            // can be pruned -- every strategy degenerates to the full scan.
            assert_eq!(results[&strategy].stats.postings_skipped, 0);
        }
    }

    #[test]
    fn empty_index_answers_with_no_hits() {
        let mut index =
            DynamicPruningIndex::new(PruningConfig::default()).expect("default config is valid");
        index.build();
        assert_eq!(index.document_count(), 0);
        assert_eq!(index.term_count(), 0);
        assert_eq!(index.average_document_length(), 0.0);

        for strategy in PruningStrategy::all() {
            let result = index
                .search_with(&["anything".to_string()], strategy)
                .expect("searching an empty index is legitimate");
            assert!(result.is_empty(), "{strategy:?} must return nothing");
            assert_eq!(result.stats.total_postings, 0);
            assert_eq!(result.stats.full_evaluations, 0);
            assert_eq!(result.stats.skip_ratio(), 0.0);
        }
    }

    /// Every document is a verbatim copy of every other, so every score is
    /// bit-for-bit identical. The answer is then decided *entirely* by the
    /// tie-break rule: the k lowest ordinals, in ascending order. Any strategy
    /// whose traversal order leaked into the ranking would fail here.
    #[test]
    fn every_document_scoring_identically() {
        for block_size in [1usize, 5, 64] {
            let config = PruningConfig {
                block_size,
                ..PruningConfig::default()
            };
            let mut index = DynamicPruningIndex::new(config).expect("valid config");
            for document in 0..50 {
                index
                    .add_document(&format!("d{document:02}"), ["same", "same", "words"])
                    .expect("valid document");
            }
            index.build();

            let query = vec!["same".to_string(), "words".to_string()];
            for top_k in [1usize, 3, 12, 50, 80] {
                let context = format!("identical scores: block_size {block_size}, k {top_k}");
                let results = assert_all_strategies_agree(&index, &query, top_k, &context);
                let hits = &results[&PruningStrategy::Exhaustive].hits;
                assert_eq!(hits.len(), top_k.min(50), "{context}");

                let first = hits[0].score;
                for (rank, hit) in hits.iter().enumerate() {
                    assert_eq!(
                        hit.score.to_bits(),
                        first.to_bits(),
                        "{context}: all scores must be bit-identical"
                    );
                    assert_eq!(
                        hit.ordinal, rank as u32,
                        "{context}: the tie-break must hand back the lowest ordinals, in order"
                    );
                }
            }
        }
    }

    #[test]
    fn a_single_document_corpus() {
        let mut index =
            DynamicPruningIndex::new(PruningConfig::default()).expect("default config is valid");
        index.add_document("only", ["one", "two"]).expect("valid");
        index.build();

        // With N = df = 1, idf = ln(1 + 0.5/1.5) = ln(4/3) > 0 -- still positive,
        // which is exactly what the `1 +` guard is for.
        let list = index.posting_list("one").expect("indexed");
        assert!(
            list.idf() > 0.0,
            "idf must not collapse on a single document"
        );
        assert!(list.max_score() > 0.0);
        assert_eq!(list.blocks().len(), 1);

        let results = assert_all_strategies_agree(
            &index,
            &["one".to_string(), "two".to_string()],
            5,
            "single document",
        );
        assert_eq!(results[&PruningStrategy::Wand].len(), 1);
    }

    #[test]
    fn block_size_one_makes_every_posting_its_own_block() {
        let config = PruningConfig {
            block_size: 1,
            ..PruningConfig::default()
        };
        let mut index = DynamicPruningIndex::new(config).expect("valid config");
        for document in 0..20 {
            index
                .add_document(&format!("d{document}"), ["shared", "unique"])
                .expect("valid");
        }
        index.build();

        let list = index.posting_list("shared").expect("indexed");
        assert_eq!(list.blocks().len(), list.len());
        for block in list.blocks() {
            assert_eq!(block.len(), 1);
            assert!(!block.is_empty());
            assert_eq!(block.first_document_ordinal(), block.max_document_ordinal());
        }
        assert_all_strategies_agree(&index, &["shared".to_string()], 3, "block_size 1");
    }

    #[test]
    fn block_size_larger_than_every_list() {
        let config = PruningConfig {
            block_size: 4096,
            ..PruningConfig::default()
        };
        let mut index = DynamicPruningIndex::new(config).expect("valid config");
        for document in 0..30 {
            index
                .add_document(&format!("d{document}"), ["a", "b", "b"])
                .expect("valid");
        }
        index.build();

        let list = index.posting_list("b").expect("indexed");
        assert_eq!(
            list.blocks().len(),
            1,
            "one block must swallow the whole list"
        );
        assert_eq!(list.blocks()[0].max_score(), list.max_score());
        assert_all_strategies_agree(
            &index,
            &["a".to_string(), "b".to_string()],
            5,
            "block_size 4096",
        );
    }

    // ── error surface ────────────────────────────────────────────────────────

    #[test]
    fn duplicate_document_id_is_rejected() {
        let mut index =
            DynamicPruningIndex::new(PruningConfig::default()).expect("default config is valid");
        index.add_document("a", ["x"]).expect("valid");
        let error = index.add_document("a", ["y"]).expect_err("duplicate");
        assert!(matches!(
            error,
            DynamicPruningError::DuplicateDocumentId { ref document_id } if document_id == "a"
        ));
        assert_eq!(index.document_count(), 1, "the failure must not mutate");
    }

    #[test]
    fn empty_document_is_rejected() {
        let mut index =
            DynamicPruningIndex::new(PruningConfig::default()).expect("default config is valid");
        let empty: [&str; 0] = [];
        let error = index.add_document("a", empty).expect_err("empty");
        assert!(matches!(
            error,
            DynamicPruningError::EmptyDocument { ref document_id } if document_id == "a"
        ));
        // A document of nothing but empty strings is equally empty.
        let error = index.add_document("b", ["", ""]).expect_err("empty");
        assert!(matches!(error, DynamicPruningError::EmptyDocument { .. }));
        assert_eq!(index.document_count(), 0);
    }

    #[test]
    fn empty_query_is_rejected() {
        let index = tiny_index(8);
        let empty: [String; 0] = [];
        assert!(matches!(
            index.search(&empty),
            Err(DynamicPruningError::EmptyQuery)
        ));
        assert!(matches!(
            index.search(&[String::new()]),
            Err(DynamicPruningError::EmptyQuery)
        ));
    }

    #[test]
    fn zero_top_k_is_rejected() {
        let index = tiny_index(8);
        assert!(matches!(
            index.search_top_k(&["rust".to_string()], PruningStrategy::Wand, 0),
            Err(DynamicPruningError::InvalidTopK)
        ));

        let bad = PruningConfig {
            top_k: 0,
            ..PruningConfig::default()
        };
        assert!(matches!(
            DynamicPruningIndex::new(bad),
            Err(DynamicPruningError::InvalidTopK)
        ));
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let zero_block = PruningConfig {
            block_size: 0,
            ..PruningConfig::default()
        };
        assert!(matches!(
            DynamicPruningIndex::new(zero_block),
            Err(DynamicPruningError::InvalidBlockSize)
        ));

        for bad_k1 in [-1.0, f64::NAN, f64::INFINITY] {
            let config = PruningConfig {
                bm25_k1: bad_k1,
                ..PruningConfig::default()
            };
            assert!(
                matches!(
                    DynamicPruningIndex::new(config),
                    Err(DynamicPruningError::InvalidBm25Parameter {
                        name: "bm25_k1",
                        ..
                    })
                ),
                "k1 = {bad_k1} must be rejected"
            );
        }

        for bad_b in [-0.1, 1.1, f64::NAN] {
            let config = PruningConfig {
                bm25_b: bad_b,
                ..PruningConfig::default()
            };
            assert!(
                matches!(
                    DynamicPruningIndex::new(config),
                    Err(DynamicPruningError::InvalidBm25Parameter { name: "bm25_b", .. })
                ),
                "b = {bad_b} must be rejected"
            );
        }
    }

    #[test]
    fn searching_a_dirty_index_is_an_error_not_a_stale_answer() {
        let mut index = tiny_index(8);
        index.add_document("epsilon", ["rust"]).expect("valid");
        assert!(!index.is_built());
        assert!(matches!(
            index.search(&["rust".to_string()]),
            Err(DynamicPruningError::IndexNotBuilt)
        ));
        index.build();
        assert!(index.search(&["rust".to_string()]).is_ok());
    }

    #[test]
    fn extreme_bm25_parameters_stay_exact() {
        // b = 0 (no length normalisation) and k1 = 0 (no tf saturation at all:
        // every posting of a term scores identically, which manufactures ties on
        // a grand scale).
        for (k1, b) in [(0.0, 0.0), (0.0, 1.0), (3.0, 1.0), (1.2, 0.0)] {
            let config = PruningConfig {
                bm25_k1: k1,
                bm25_b: b,
                block_size: 4,
                ..PruningConfig::default()
            };
            let mut index = DynamicPruningIndex::new(config).expect("valid config");
            for document in 0..40 {
                let mut terms = vec!["common".to_string()];
                if document % 3 == 0 {
                    terms.push("third".to_string());
                }
                if document % 7 == 0 {
                    terms.push("seventh".to_string());
                    terms.push("seventh".to_string());
                }
                for _ in 0..(document % 5) {
                    terms.push("filler".to_string());
                }
                index
                    .add_document(&format!("d{document:02}"), &terms)
                    .expect("valid document");
            }
            index.build();

            assert_all_strategies_agree(
                &index,
                &[
                    "common".to_string(),
                    "third".to_string(),
                    "seventh".to_string(),
                ],
                6,
                &format!("k1 {k1}, b {b}"),
            );
        }
    }
}

// ── internals ────────────────────────────────────────────────────────────────

mod internals {
    use super::{RankedEntry, ScoreAccumulator, TopKHeap, can_reach};

    #[test]
    fn ranked_entry_orders_by_score_then_ascending_ordinal() {
        let better_score = RankedEntry {
            score: 2.0,
            document_ordinal: 99,
        };
        let worse_score = RankedEntry {
            score: 1.0,
            document_ordinal: 0,
        };
        assert!(better_score > worse_score, "score dominates the ordinal");

        let early = RankedEntry {
            score: 1.0,
            document_ordinal: 3,
        };
        let late = RankedEntry {
            score: 1.0,
            document_ordinal: 4,
        };
        assert!(early > late, "on a tie, the lower ordinal wins");
        assert_ne!(early, late, "distinct ordinals must never compare equal");
    }

    #[test]
    fn top_k_heap_keeps_the_best_k_and_reports_the_threshold() {
        let mut heap = TopKHeap::new(3);
        assert_eq!(heap.threshold(), None, "an unfilled heap prunes nothing");

        for (score, ordinal) in [(1.0, 0u32), (5.0, 1), (3.0, 2)] {
            assert!(heap.offer(RankedEntry {
                score,
                document_ordinal: ordinal
            }));
        }
        assert_eq!(
            heap.threshold(),
            Some(1.0),
            "the k-th best is the threshold"
        );

        // Strictly worse: rejected.
        assert!(!heap.offer(RankedEntry {
            score: 0.5,
            document_ordinal: 3
        }));
        // Strictly better: admitted, and the threshold rises.
        assert!(heap.offer(RankedEntry {
            score: 4.0,
            document_ordinal: 4
        }));
        assert_eq!(heap.threshold(), Some(3.0));

        // Tied with the threshold but with a *lower* ordinal: must displace it,
        // or the tie-break rule would not be honoured.
        assert!(heap.offer(RankedEntry {
            score: 3.0,
            document_ordinal: 1
        }));

        let ranked = heap.into_ranked();
        let ordinals: Vec<u32> = ranked.iter().map(|entry| entry.document_ordinal).collect();
        assert_eq!(ordinals, vec![1, 4, 1]);
        let scores: Vec<f64> = ranked.iter().map(|entry| entry.score).collect();
        assert_eq!(scores, vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn top_k_heap_rejects_a_tie_with_a_higher_ordinal() {
        let mut heap = TopKHeap::new(1);
        assert!(heap.offer(RankedEntry {
            score: 2.0,
            document_ordinal: 7
        }));
        assert!(
            !heap.offer(RankedEntry {
                score: 2.0,
                document_ordinal: 8
            }),
            "a later document must not displace an equally-scoring earlier one"
        );
        assert!(heap.offer(RankedEntry {
            score: 2.0,
            document_ordinal: 6
        }));
        assert_eq!(heap.threshold(), Some(2.0));
    }

    #[test]
    fn can_reach_never_prunes_an_unfilled_heap() {
        assert!(can_reach(0.0, None));
        assert!(can_reach(f64::NEG_INFINITY, None));
    }

    #[test]
    fn can_reach_admits_an_exact_tie_and_rejects_a_clear_loss() {
        assert!(
            can_reach(5.0, Some(5.0)),
            "a ceiling exactly at the threshold must not be pruned: the document \
             could tie on score and win on ordinal"
        );
        assert!(can_reach(5.000_000_001, Some(5.0)));
        assert!(!can_reach(4.9, Some(5.0)), "a clear loss must be pruned");
    }

    #[test]
    fn can_reach_absorbs_summation_rounding() {
        // A ceiling a few ulps below the threshold must still be admitted: the
        // sum of ceilings and the sum of scores round independently, and the
        // ceiling only *appears* smaller.
        let theta = 12.345_678_9_f64;
        let a_hair_under = theta - 4.0 * f64::EPSILON * theta;
        assert!(a_hair_under < theta, "the fixture must actually be under");
        assert!(
            can_reach(a_hair_under, Some(theta)),
            "rounding slack must be absorbed, or exactness is lost"
        );
    }

    #[test]
    fn score_accumulator_sums_in_query_order_with_exact_zero_padding() {
        let mut accumulator = ScoreAccumulator::new(4);
        assert_eq!(accumulator.total(), 0.0);

        accumulator.set(2, 0.25);
        accumulator.set(0, 0.5);
        assert_eq!(accumulator.total(), 0.75);

        // Writing the same slots in the opposite order must give a bit-identical
        // total: the sum is over slots, not over the order they were written in.
        let mut mirror = ScoreAccumulator::new(4);
        mirror.set(0, 0.5);
        mirror.set(2, 0.25);
        assert_eq!(mirror.total().to_bits(), accumulator.total().to_bits());

        accumulator.reset();
        assert_eq!(accumulator.total(), 0.0);
    }

    /// The property that makes zero-padding safe: adding an exact `+0.0` cannot
    /// perturb a non-negative running sum by a single bit.
    #[test]
    fn adding_zero_is_bit_preserving_for_non_negative_sums() {
        for value in [0.0f64, 1.0, 0.1, 1e-300, 1e300, f64::MIN_POSITIVE] {
            assert_eq!((value + 0.0).to_bits(), value.to_bits(), "value {value}");
        }
    }
}

/// A last, blunt guard: whatever else changes, the four strategies must remain
/// mutually indistinguishable in their answers over a broad, deterministic
/// sweep. Kept separate from the `equivalence` module's shaped fixtures so that
/// a regression shows up here even if a fixture is ever weakened.
#[test]
fn exactness_holds_over_a_broad_deterministic_sweep() {
    let mut seen_documents: HashSet<String> = HashSet::new();
    let mut rng = SplitMix64::new(0x0BAD_C0DE);
    let mut total_queries = 0usize;
    let mut total_skipped = 0u64;

    for seed in 0..8u64 {
        let shape = CorpusShape {
            document_count: 120 + (seed as usize) * 90,
            vocabulary_size: 15 + (seed as usize) * 9,
            min_length: 3,
            max_length: 12 + (seed as usize) * 5,
            zipf_exponent: 0.2 * seed as f64,
        };
        let config = PruningConfig {
            block_size: 1 + (seed as usize) * 11,
            bm25_k1: 0.6 + 0.3 * seed as f64,
            bm25_b: (0.1 * seed as f64).min(1.0),
            ..PruningConfig::default()
        };
        let (index, vocabulary) = build_random_index(shape, config, 0xFACE_0000 ^ seed);

        for _ in 0..15 {
            let query = random_query(&vocabulary, rng.range(1, 6), &mut rng);
            let top_k = rng.range(1, 30);
            let context = format!("sweep seed {seed}, k {top_k}, query {query:?}");
            let results = assert_all_strategies_agree(&index, &query, top_k, &context);
            for hit in &results[&PruningStrategy::BlockMaxWand].hits {
                seen_documents.insert(hit.document_id.clone());
            }
            total_skipped += results[&PruningStrategy::BlockMaxWand]
                .stats
                .postings_skipped;
            total_queries += 1;
        }
    }

    assert_eq!(total_queries, 120);
    assert!(
        seen_documents.len() > 50,
        "the sweep must actually return varied documents"
    );
    assert!(
        total_skipped > 0,
        "across 120 queries, block-max pruning must have skipped *something*"
    );
}
