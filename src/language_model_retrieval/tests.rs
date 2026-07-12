#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    // Every hand-computed expectation below re-derives a formula from the
    // module documentation using raw arithmetic. Rewriting `a * b + c` as
    // `a.mul_add(b, c)` (or `(x + y) / 2.0` as `f64::midpoint`) would make the
    // test read differently from the formula it is checking, which is exactly
    // what these tests exist to prevent.
    clippy::suboptimal_flops,
    clippy::manual_midpoint
)]
//! Tests for probabilistic language-model retrieval.
//!
//! The ground-truth sections re-derive every expected value from the raw
//! formulas in the module documentation, written out as explicit arithmetic —
//! they never call the code under test to produce an expectation. Where the
//! closed form is exact (the query-likelihood scores reduce to sums of logs of
//! rationals) the literal decimal is asserted as well, so a refactor that
//! changed both the implementation and the "hand" computation in the same way
//! would still be caught.
//!
//! The reference corpus throughout is deliberately tiny:
//!
//! ```text
//! d1 = "apple banana apple"       -> apple:2 banana:1          |d1|=3  |d1|_u=2
//! d2 = "banana cherry"            -> banana:1 cherry:1         |d2|=2  |d2|_u=2
//! d3 = "apple cherry cherry date" -> apple:1 cherry:2 date:1   |d3|=4  |d3|_u=3
//!
//! N = 3   |C| = 9   avgdl = 3
//! cf: apple=3 banana=2 cherry=3 date=1        (3+2+3+1 = 9, as it must)
//! P(w|C): apple=1/3 banana=2/9 cherry=1/3 date=1/9
//! ```

use std::collections::HashMap;
use std::f64::consts::{LN_2, LOG2_E, TAU};

use crate::language_model_retrieval::dfr::{dph_term_score, pl2_term_score};
use crate::language_model_retrieval::index::LmRetrievalIndex;
use crate::language_model_retrieval::rm3::{lm_log_sum_exp, lm_query_posteriors};
use crate::language_model_retrieval::smoothing::LmCollectionModel;
use crate::language_model_retrieval::types::{
    DfrModel, LM_MIN_PROBABILITY, LmHit, LmRetrievalConfig, LmRetrievalError, LmScoringModel,
    LmSmoothing, Rm3Config, lm_tokenize,
};

// ── Test helpers ─────────────────────────────────────────────────────────────

const TIGHT: f64 = 1e-12;
const HAND: f64 = 1e-6;

#[track_caller]
fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: got {actual}, expected {expected} (tolerance {tolerance}, delta {})",
        (actual - expected).abs()
    );
}

/// The reference corpus of the module header.
fn fruit_index(smoothing: LmSmoothing) -> LmRetrievalIndex {
    build_fruit(LmRetrievalConfig::query_likelihood(smoothing))
}

fn fruit_dfr_index(model: DfrModel) -> LmRetrievalIndex {
    build_fruit(LmRetrievalConfig::dfr(model))
}

fn build_fruit(config: LmRetrievalConfig) -> LmRetrievalIndex {
    let mut index = LmRetrievalIndex::new(config).expect("configuration is valid");
    index
        .add_document("d1", "apple banana apple")
        .expect("unique id");
    index
        .add_document("d2", "banana cherry")
        .expect("unique id");
    index
        .add_document("d3", "apple cherry cherry date")
        .expect("unique id");
    index.build().expect("configuration is valid");
    index
}

#[track_caller]
fn score_of(hits: &[LmHit], document_id: &str) -> f64 {
    hits.iter()
        .find(|hit| hit.document_id == document_id)
        .unwrap_or_else(|| panic!("{document_id} is missing from the hit list"))
        .score
}

fn ranking(hits: &[LmHit]) -> Vec<&str> {
    hits.iter().map(|hit| hit.document_id.as_str()).collect()
}

/// `SplitMix64` — a deterministic seeded generator for the stress corpus. The
/// crate forbids `rand`, and a stress test that is not reproducible is not a
/// test.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("a usize bound fits in u64");
        usize::try_from(self.next_u64() % bound).expect("a value below the bound fits in usize")
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

#[test]
fn tokenizer_splits_on_non_alphanumerics_and_lowercases() {
    assert_eq!(
        lm_tokenize("Renal-Failure, kidney!", true),
        vec![
            "renal".to_string(),
            "failure".to_string(),
            "kidney".to_string()
        ]
    );
    assert_eq!(
        lm_tokenize("Renal Failure", false),
        vec!["Renal".to_string(), "Failure".to_string()]
    );
    assert!(lm_tokenize("   !!!  ", true).is_empty());
}

// ── Collection model ─────────────────────────────────────────────────────────

#[test]
fn collection_model_is_the_maximum_likelihood_estimate_over_the_concatenated_corpus() {
    let index = fruit_index(LmSmoothing::default());

    assert_eq!(index.num_documents(), 3);
    assert_eq!(index.total_tokens(), 9);
    assert_eq!(index.vocabulary_size(), 4);
    assert_close(index.average_document_length(), 3.0, TIGHT, "avgdl");

    assert_eq!(index.collection_frequency("apple"), 3);
    assert_eq!(index.collection_frequency("banana"), 2);
    assert_eq!(index.collection_frequency("cherry"), 3);
    assert_eq!(index.collection_frequency("date"), 1);
    assert_eq!(index.document_frequency("apple"), 2);
    assert_eq!(index.document_frequency("date"), 1);

    assert_close(
        index.collection_probability("apple"),
        3.0 / 9.0,
        TIGHT,
        "P(apple|C)",
    );
    assert_close(
        index.collection_probability("banana"),
        2.0 / 9.0,
        TIGHT,
        "P(banana|C)",
    );
    assert_close(
        index.collection_probability("date"),
        1.0 / 9.0,
        TIGHT,
        "P(date|C)",
    );

    // The collection distribution over the known vocabulary sums to exactly 1.
    let mass: f64 = ["apple", "banana", "cherry", "date"]
        .iter()
        .map(|term| index.collection_probability(term))
        .sum();
    assert_close(mass, 1.0, TIGHT, "collection model normalization");
}

#[test]
fn out_of_vocabulary_terms_get_a_positive_floor_below_the_rarest_seen_term() {
    let index = fruit_index(LmSmoothing::default());

    // 1 / (|C| + 1) = 1 / 10.
    let floor = index.collection_probability("zzz_unseen");
    assert_close(floor, 1.0 / 10.0, TIGHT, "OOV floor");

    // Strictly below 1/|C|, the probability of a term seen exactly once.
    assert!(floor < 1.0 / 9.0);
    assert!(floor < index.collection_probability("date"));
    assert!(floor > 0.0);
    assert!(floor.ln().is_finite());
}

#[test]
fn empty_collection_degenerates_finitely_rather_than_to_negative_infinity() {
    let frequencies: HashMap<String, u64> = HashMap::new();
    let collection = LmCollectionModel::new(&frequencies, 0);

    // 1 / (0 + 1) = 1. Meaningless, but finite -- which is the contract.
    assert_close(
        collection.probability("anything"),
        1.0,
        TIGHT,
        "empty-collection floor",
    );
    assert!(collection.probability("anything").ln().is_finite());
}

// ── Ground truth: Dirichlet ──────────────────────────────────────────────────

#[test]
fn dirichlet_query_likelihood_matches_hand_computation() {
    let mu = 2.0;
    let index = fruit_index(LmSmoothing::Dirichlet { mu });
    let hits = index
        .search("apple cherry", 3)
        .expect("built, non-empty query");
    assert_eq!(hits.len(), 3);

    // P(w|D) = (tf + mu * P(w|C)) / (|D| + mu),  P(apple|C) = P(cherry|C) = 3/9.
    let p_apple = 3.0 / 9.0;
    let p_cherry = 3.0 / 9.0;

    //   d1: |D|=3, tf(apple)=2, tf(cherry)=0
    //     P(apple |d1) = (2 + 2*(1/3)) / 5 =  (8/3)/5 =  8/15
    //     P(cherry|d1) = (0 + 2*(1/3)) / 5 =  (2/3)/5 =  2/15
    let d1 = ((2.0 + mu * p_apple) / (3.0 + mu)).ln() + ((0.0 + mu * p_cherry) / (3.0 + mu)).ln();
    //   d2: |D|=2, tf(apple)=0, tf(cherry)=1
    //     P(apple |d2) = (0 + 2/3) / 4 = 1/6
    //     P(cherry|d2) = (1 + 2/3) / 4 = 5/12
    let d2 = ((0.0 + mu * p_apple) / (2.0 + mu)).ln() + ((1.0 + mu * p_cherry) / (2.0 + mu)).ln();
    //   d3: |D|=4, tf(apple)=1, tf(cherry)=2
    //     P(apple |d3) = (1 + 2/3) / 6 = 5/18
    //     P(cherry|d3) = (2 + 2/3) / 6 = 4/9
    let d3 = ((1.0 + mu * p_apple) / (4.0 + mu)).ln() + ((2.0 + mu * p_cherry) / (4.0 + mu)).ln();

    assert_close(score_of(&hits, "d1"), d1, TIGHT, "Dirichlet d1");
    assert_close(score_of(&hits, "d2"), d2, TIGHT, "Dirichlet d2");
    assert_close(score_of(&hits, "d3"), d3, TIGHT, "Dirichlet d3");

    // The same values, worked out to closed form off the machine:
    //   d1 = ln(8/15) + ln(2/15) = -0.628608659... + -2.014903020... = -2.643511679965
    //   d2 = ln(1/6)  + ln(5/12) = -1.791759469... + -0.875468737... = -2.667228206582
    //   d3 = ln(5/18) + ln(4/9)  = -1.280933845... + -0.810930216... = -2.091864061678
    assert_close(
        score_of(&hits, "d1"),
        -2.643_511_679_965,
        HAND,
        "Dirichlet d1 literal",
    );
    assert_close(
        score_of(&hits, "d2"),
        -2.667_228_206_582,
        HAND,
        "Dirichlet d2 literal",
    );
    assert_close(
        score_of(&hits, "d3"),
        -2.091_864_061_678,
        HAND,
        "Dirichlet d3 literal",
    );

    assert_eq!(ranking(&hits), vec!["d3", "d1", "d2"]);
}

// ── Ground truth: Jelinek-Mercer ─────────────────────────────────────────────

#[test]
fn jelinek_mercer_query_likelihood_matches_hand_computation() {
    let lambda = 0.5;
    let index = fruit_index(LmSmoothing::JelinekMercer { lambda });
    let hits = index
        .search("apple cherry", 3)
        .expect("built, non-empty query");

    // P(w|D) = (1 - lambda) * tf/|D| + lambda * P(w|C)
    let p_apple = 3.0 / 9.0;
    let p_cherry = 3.0 / 9.0;

    //   d1: 0.5*(2/3) + 0.5*(1/3) = 1/2 ; 0.5*0     + 0.5*(1/3) = 1/6
    let d1 = ((1.0 - lambda) * (2.0 / 3.0) + lambda * p_apple).ln()
        + ((1.0 - lambda) * (0.0 / 3.0) + lambda * p_cherry).ln();
    //   d2: 0.5*0     + 0.5*(1/3) = 1/6 ; 0.5*(1/2) + 0.5*(1/3) = 5/12
    let d2 = ((1.0 - lambda) * (0.0 / 2.0) + lambda * p_apple).ln()
        + ((1.0 - lambda) * (1.0 / 2.0) + lambda * p_cherry).ln();
    //   d3: 0.5*(1/4) + 0.5*(1/3) = 7/24; 0.5*(2/4) + 0.5*(1/3) = 5/12
    let d3 = ((1.0 - lambda) * (1.0 / 4.0) + lambda * p_apple).ln()
        + ((1.0 - lambda) * (2.0 / 4.0) + lambda * p_cherry).ln();

    assert_close(score_of(&hits, "d1"), d1, TIGHT, "JM d1");
    assert_close(score_of(&hits, "d2"), d2, TIGHT, "JM d2");
    assert_close(score_of(&hits, "d3"), d3, TIGHT, "JM d3");

    //   d1 = ln(1/2) + ln(1/6)  = -ln 12          = -2.484906649788
    //   d2 = ln(1/6) + ln(5/12)                   = -2.667228206582
    //   d3 = ln(7/24) + ln(5/12)                  = -2.107612418647
    assert_close(
        score_of(&hits, "d1"),
        -2.484_906_649_788,
        HAND,
        "JM d1 literal",
    );
    assert_close(
        score_of(&hits, "d2"),
        -2.667_228_206_582,
        HAND,
        "JM d2 literal",
    );
    assert_close(
        score_of(&hits, "d3"),
        -2.107_612_418_647,
        HAND,
        "JM d3 literal",
    );

    assert_eq!(ranking(&hits), vec!["d3", "d1", "d2"]);
}

// ── Ground truth: absolute discounting ───────────────────────────────────────

#[test]
fn absolute_discounting_query_likelihood_matches_hand_computation() {
    let delta = 0.5;
    let index = fruit_index(LmSmoothing::AbsoluteDiscounting { delta });
    let hits = index
        .search("apple cherry", 3)
        .expect("built, non-empty query");

    // P(w|D) = max(tf - delta, 0)/|D| + (delta * |D|_unique / |D|) * P(w|C)
    let p_apple = 3.0 / 9.0;
    let p_cherry = 3.0 / 9.0;

    //   d1: |D|=3, |D|_u=2 -> escape = 0.5*2/3 = 1/3
    //     P(apple |d1) = (2-0.5)/3 + (1/3)*(1/3) = 0.5     + 1/9 = 11/18
    //     P(cherry|d1) = 0         + (1/3)*(1/3)               = 1/9
    let escape_d1 = delta * 2.0 / 3.0;
    let d1 = (((2.0 - delta).max(0.0) / 3.0) + escape_d1 * p_apple).ln()
        + (((0.0 - delta).max(0.0) / 3.0) + escape_d1 * p_cherry).ln();
    //   d2: |D|=2, |D|_u=2 -> escape = 0.5*2/2 = 1/2
    //     P(apple |d2) = 0         + 0.5*(1/3)               = 1/6
    //     P(cherry|d2) = (1-0.5)/2 + 0.5*(1/3) = 0.25 + 1/6  = 5/12
    let escape_d2 = delta * 2.0 / 2.0;
    let d2 = (((0.0 - delta).max(0.0) / 2.0) + escape_d2 * p_apple).ln()
        + (((1.0 - delta).max(0.0) / 2.0) + escape_d2 * p_cherry).ln();
    //   d3: |D|=4, |D|_u=3 -> escape = 0.5*3/4 = 3/8
    //     P(apple |d3) = (1-0.5)/4 + 0.375*(1/3) = 0.125 + 0.125 = 1/4
    //     P(cherry|d3) = (2-0.5)/4 + 0.375*(1/3) = 0.375 + 0.125 = 1/2
    let escape_d3 = delta * 3.0 / 4.0;
    let d3 = (((1.0 - delta).max(0.0) / 4.0) + escape_d3 * p_apple).ln()
        + (((2.0 - delta).max(0.0) / 4.0) + escape_d3 * p_cherry).ln();

    assert_close(score_of(&hits, "d1"), d1, TIGHT, "AD d1");
    assert_close(score_of(&hits, "d2"), d2, TIGHT, "AD d2");
    assert_close(score_of(&hits, "d3"), d3, TIGHT, "AD d3");

    //   d1 = ln(11/18) + ln(1/9)  = -2.689701062434
    //   d2 = ln(1/6)   + ln(5/12) = -2.667228206582
    //   d3 = ln(1/4)   + ln(1/2)  = -ln 8 = -2.079441541680
    assert_close(
        score_of(&hits, "d1"),
        -2.689_701_062_434,
        HAND,
        "AD d1 literal",
    );
    assert_close(
        score_of(&hits, "d2"),
        -2.667_228_206_582,
        HAND,
        "AD d2 literal",
    );
    assert_close(
        score_of(&hits, "d3"),
        -2.079_441_541_680,
        HAND,
        "AD d3 literal",
    );

    // Note the ranking: absolute discounting puts d2 *above* d1, where
    // Dirichlet and Jelinek-Mercer both put d1 above d2. The smoothing scheme
    // is the retrieval model.
    assert_eq!(ranking(&hits), vec!["d3", "d2", "d1"]);
}

// ── Normalization ────────────────────────────────────────────────────────────

#[test]
fn every_smoothing_scheme_produces_a_proper_distribution() {
    let vocabulary = [
        ("apple", 3.0),
        ("banana", 2.0),
        ("cherry", 3.0),
        ("date", 1.0),
    ];
    let documents = [
        // (tf(apple), tf(banana), tf(cherry), tf(date), |D|, |D|_unique)
        (2_u64, 1_u64, 0_u64, 0_u64, 3_u64, 2_u64),
        (0, 1, 1, 0, 2, 2),
        (1, 0, 2, 1, 4, 3),
        // The empty document: |D| = 0 backs off wholly to the collection model.
        (0, 0, 0, 0, 0, 0),
    ];

    for smoothing in [
        LmSmoothing::Dirichlet { mu: 2.0 },
        LmSmoothing::Dirichlet { mu: 2000.0 },
        LmSmoothing::JelinekMercer { lambda: 0.5 },
        LmSmoothing::JelinekMercer { lambda: 1.0 },
        LmSmoothing::AbsoluteDiscounting { delta: 0.5 },
        LmSmoothing::AbsoluteDiscounting { delta: 1.0 },
    ] {
        for (apple, banana, cherry, date, length, unique) in documents {
            let frequencies = [apple, banana, cherry, date];
            let mass: f64 = vocabulary
                .iter()
                .zip(frequencies)
                .map(|((_, cf), tf)| smoothing.document_probability(tf, length, unique, cf / 9.0))
                .sum();
            assert_close(
                mass,
                1.0,
                1e-9,
                &format!("normalization of {smoothing:?} on |D|={length}"),
            );
        }
    }
}

#[test]
fn absolute_discounting_normalization_is_why_delta_must_not_exceed_one() {
    // delta > 1 truncates singleton terms at zero (max(1 - delta, 0) = 0) while
    // still handing out delta * |D|_u / |D| of escape mass, so the "distribution"
    // would sum to more than one. The configuration is rejected outright.
    let error = LmSmoothing::AbsoluteDiscounting { delta: 1.5 }
        .validate()
        .expect_err("delta > 1 must be rejected");
    assert!(matches!(
        error,
        LmRetrievalError::InvalidParameter { ref parameter, .. } if parameter == "delta"
    ));

    // At the boundary delta = 1 it still normalizes exactly: every observed term
    // is discounted to tf - 1 and the whole |D|_u / |D| goes to the background.
    let mass: f64 = [
        (2_u64, 3.0 / 9.0),
        (1, 2.0 / 9.0),
        (0, 3.0 / 9.0),
        (0, 1.0 / 9.0),
    ]
    .iter()
    .map(|(tf, p)| {
        LmSmoothing::AbsoluteDiscounting { delta: 1.0 }.document_probability(*tf, 3, 2, *p)
    })
    .sum();
    assert_close(mass, 1.0, 1e-12, "delta = 1 normalization");
}

// ── Smoothing limits ─────────────────────────────────────────────────────────

#[test]
fn dirichlet_approaches_the_unsmoothed_mle_as_mu_goes_to_zero() {
    // tf = 2, |D| = 3 -> MLE is 2/3.
    let p_collection = 3.0 / 9.0;
    for mu in [1.0, 1e-3, 1e-6, 1e-9] {
        let probability = LmSmoothing::Dirichlet { mu }.document_probability(2, 3, 2, p_collection);
        let error = (probability - 2.0 / 3.0).abs();
        assert!(
            error < 0.2 * mu.max(1e-12) + 1e-12 || mu >= 1.0,
            "mu = {mu} should approach the MLE, got {probability}"
        );
    }
    let nearly_unsmoothed =
        LmSmoothing::Dirichlet { mu: 1e-9 }.document_probability(2, 3, 2, p_collection);
    assert_close(nearly_unsmoothed, 2.0 / 3.0, 1e-9, "mu -> 0 is the MLE");

    // At mu = 0 exactly, the MLE assigns zero to an absent term. The clamp
    // keeps the logarithm finite instead of emitting -inf.
    let absent = LmSmoothing::Dirichlet { mu: 0.0 }.document_probability(0, 3, 2, p_collection);
    assert_close(absent, LM_MIN_PROBABILITY, TIGHT, "mu = 0, absent term");
    assert!(absent.ln().is_finite());
    // ...and the present term is untouched by the clamp.
    let present = LmSmoothing::Dirichlet { mu: 0.0 }.document_probability(2, 3, 2, p_collection);
    assert_close(present, 2.0 / 3.0, TIGHT, "mu = 0, present term");
}

#[test]
fn dirichlet_approaches_the_collection_model_as_mu_goes_to_infinity() {
    let p_collection = 3.0 / 9.0;
    let heavily_smoothed =
        LmSmoothing::Dirichlet { mu: 1e12 }.document_probability(2, 3, 2, p_collection);
    assert_close(heavily_smoothed, p_collection, 1e-9, "mu -> inf is P(w|C)");

    // And so every document's score collapses to the same value.
    let index = fruit_index(LmSmoothing::Dirichlet { mu: 1e12 });
    let hits = index
        .search("apple cherry", 3)
        .expect("built, non-empty query");
    let spread = score_of(&hits, "d3") - score_of(&hits, "d2");
    assert!(
        spread.abs() < 1e-6,
        "mu = 1e12 should flatten the ranking; spread was {spread}"
    );
}

#[test]
fn jelinek_mercer_at_lambda_one_is_the_pure_collection_model_and_every_document_ties() {
    let index = fruit_index(LmSmoothing::JelinekMercer { lambda: 1.0 });
    let hits = index
        .search("apple cherry", 3)
        .expect("built, non-empty query");

    // P(w|D) = P(w|C) for every D, so the score is the same for every document:
    // ln(1/3) + ln(1/3) = -2 * ln 3.
    let expected = (3.0_f64 / 9.0).ln() + (3.0_f64 / 9.0).ln();
    for hit in &hits {
        assert_close(hit.score, expected, TIGHT, "lambda = 1 collapses to P(.|C)");
    }
    assert_close(score_of(&hits, "d1"), -2.0 * 3.0_f64.ln(), TIGHT, "-2 ln 3");
}

#[test]
fn jelinek_mercer_at_lambda_zero_is_the_unsmoothed_mle() {
    let p_collection = 3.0 / 9.0;
    let smoothing = LmSmoothing::JelinekMercer { lambda: 0.0 };
    assert_close(
        smoothing.document_probability(2, 3, 2, p_collection),
        2.0 / 3.0,
        TIGHT,
        "lambda = 0, present term",
    );
    let absent = smoothing.document_probability(0, 3, 2, p_collection);
    assert_close(absent, LM_MIN_PROBABILITY, TIGHT, "lambda = 0, absent term");
    assert!(absent.ln().is_finite());
}

#[test]
fn dirichlet_and_jelinek_mercer_produce_different_rankings_on_skewed_lengths() {
    // The classic divergence. Two documents contain both query terms:
    //   long_doc : |D| = 100, tf(alpha) = tf(beta) = 10  -> relative freq 0.10
    //   short_doc: |D| =  10, tf(alpha) = tf(beta) =  2  -> relative freq 0.20
    // Dirichlet's smoothing weight mu/(|D|+mu) shrinks with length, so a long
    // document leans on its own (large, absolute) counts and wins. Jelinek-
    // Mercer's weight is the constant lambda, so it compares *relative*
    // frequencies -- and the short document wins. Same corpus, same query,
    // opposite answers.
    let build = |config: LmRetrievalConfig| {
        let mut index = LmRetrievalIndex::new(config).expect("valid configuration");
        let long_text = format!(
            "{}{}{}",
            "alpha ".repeat(10),
            "beta ".repeat(10),
            "filler ".repeat(80)
        );
        index.add_document("long_doc", &long_text).expect("unique");
        index
            .add_document(
                "short_doc",
                "alpha alpha beta beta gamma gamma gamma gamma gamma gamma",
            )
            .expect("unique");
        index
            .add_document("noise_doc", &"delta ".repeat(200))
            .expect("unique");
        index.build().expect("valid configuration");
        index
    };

    let dirichlet = build(LmRetrievalConfig::query_likelihood(
        LmSmoothing::Dirichlet { mu: 2000.0 },
    ));
    let jelinek = build(LmRetrievalConfig::query_likelihood(
        LmSmoothing::JelinekMercer { lambda: 0.5 },
    ));

    assert_eq!(dirichlet.total_tokens(), 310);
    assert_eq!(dirichlet.collection_frequency("alpha"), 12);

    let dirichlet_hits = dirichlet.search("alpha beta", 3).expect("built");
    let jelinek_hits = jelinek.search("alpha beta", 3).expect("built");

    assert_eq!(
        ranking(&dirichlet_hits),
        vec!["long_doc", "short_doc", "noise_doc"]
    );
    assert_eq!(
        ranking(&jelinek_hits),
        vec!["short_doc", "long_doc", "noise_doc"]
    );
    assert_ne!(ranking(&dirichlet_hits), ranking(&jelinek_hits));

    // Hand check of the two top scores, from the raw formulas.
    // P(alpha|C) = P(beta|C) = 12/310.
    let p: f64 = 12.0 / 310.0;
    let mu: f64 = 2000.0;
    let dirichlet_long = 2.0 * ((10.0 + mu * p) / (100.0 + mu)).ln();
    let dirichlet_short = 2.0 * ((2.0 + mu * p) / (10.0 + mu)).ln();
    assert!(dirichlet_long > dirichlet_short);
    assert_close(
        score_of(&dirichlet_hits, "long_doc"),
        dirichlet_long,
        TIGHT,
        "Dirichlet long_doc",
    );

    let lambda: f64 = 0.5;
    let jelinek_long = 2.0 * ((1.0 - lambda) * (10.0 / 100.0) + lambda * p).ln();
    let jelinek_short = 2.0 * ((1.0 - lambda) * (2.0 / 10.0) + lambda * p).ln();
    assert!(jelinek_short > jelinek_long);
    assert_close(
        score_of(&jelinek_hits, "short_doc"),
        jelinek_short,
        TIGHT,
        "JM short_doc",
    );
}

// ── Ground truth: PL2 ────────────────────────────────────────────────────────

#[test]
fn pl2_matches_hand_computation() {
    let c = 1.0;
    let index = fruit_dfr_index(DfrModel::Pl2 { c });
    let hits = index.search("apple", 3).expect("built, non-empty query");

    // DFR scores only the posting list: d2 has no "apple".
    assert_eq!(hits.len(), 2);
    assert_eq!(ranking(&hits), vec!["d1", "d3"]);

    // lambda_w = cf(apple)/N = 3/3 = 1; avgdl = 3.
    let lambda = 3.0 / 3.0;
    let avgdl = 3.0;

    //   d1: tf = 2, dl = 3 -> tfn = 2 * log2(1 + 1*3/3) = 2 * log2(2) = 2
    let tfn_d1 = 2.0 * (1.0 + c * avgdl / 3.0).log2();
    assert_close(tfn_d1, 2.0, TIGHT, "tfn(d1)");
    let d1 = (tfn_d1 * (tfn_d1 / lambda).log2()
        + (lambda - tfn_d1) * LOG2_E
        + 0.5 * (TAU * tfn_d1).log2())
        / (tfn_d1 + 1.0);

    //   d3: tf = 1, dl = 4 -> tfn = log2(1 + 3/4) = log2(1.75) = 0.8073549220576
    let tfn_d3 = 1.0 * (1.0 + c * avgdl / 4.0).log2();
    assert_close(tfn_d3, 0.807_354_922_057_6, HAND, "tfn(d3)");
    let d3 = (tfn_d3 * (tfn_d3 / lambda).log2()
        + (lambda - tfn_d3) * LOG2_E
        + 0.5 * (TAU * tfn_d3).log2())
        / (tfn_d3 + 1.0);

    assert_close(score_of(&hits, "d1"), d1, TIGHT, "PL2 d1");
    assert_close(score_of(&hits, "d3"), d3, TIGHT, "PL2 d3");

    // Worked out off the machine:
    //   d1 = (2*1 + (1-2)*1.442695041 + 0.5*log2(4*pi)) / 3
    //      = (2 - 1.442695041 + 1.825748050) / 3 = 2.383053009 / 3
    assert_close(
        score_of(&hits, "d1"),
        0.794_351_007_949,
        HAND,
        "PL2 d1 literal",
    );
    assert_close(
        score_of(&hits, "d3"),
        0.663_988_531_151,
        HAND,
        "PL2 d3 literal",
    );

    // The free function agrees with the index (it is what the index calls).
    assert_close(
        pl2_term_score(1.0, 2, 3, avgdl, 3, 3, c),
        d1,
        TIGHT,
        "pl2_term_score d1",
    );
}

#[test]
fn pl2_normalized_term_frequency_is_the_normalization_2_formula() {
    let model = DfrModel::Pl2 { c: 1.0 };
    // tfn = tf * log2(1 + c * avgdl / dl); at dl = c*avgdl the factor is exactly 1.
    assert_close(
        model.normalized_term_frequency(3, 100, 100.0),
        3.0 * 2.0_f64.log2(),
        TIGHT,
        "tfn at dl = avgdl",
    );
    // Long documents are normalized down towards zero.
    let long = model.normalized_term_frequency(3, 10_000, 100.0);
    assert!(
        long > 0.0 && long < 0.05,
        "tfn on a very long document: {long}"
    );
    // DPH has no such quantity.
    assert_close(
        DfrModel::Dph.normalized_term_frequency(3, 100, 100.0),
        0.0,
        TIGHT,
        "DPH has no tfn",
    );
}

// ── Ground truth: DPH ────────────────────────────────────────────────────────

#[test]
fn dph_matches_hand_computation() {
    let index = fruit_dfr_index(DfrModel::Dph);
    let hits = index.search("apple", 3).expect("built, non-empty query");

    assert_eq!(hits.len(), 2);
    assert_eq!(ranking(&hits), vec!["d1", "d3"]);

    // score = qtf * ((1-f)^2 / (tf+1)) * ( tf*log2((tf/dl)*(N/cf)) + 0.5*log2(2*pi*tf*(1-f)) )
    // N = 3, cf(apple) = 3.
    let n: f64 = 3.0;
    let cf: f64 = 3.0;

    //   d1: tf = 2, dl = 3 -> f = 2/3, 1-f = 1/3
    let f_d1 = 2.0 / 3.0;
    let d1 = ((1.0 - f_d1) * (1.0 - f_d1) / (2.0 + 1.0))
        * (2.0 * ((2.0 / 3.0) * (n / cf)).log2() + 0.5 * (TAU * 2.0 * (1.0 - f_d1)).log2());
    //   d3: tf = 1, dl = 4 -> f = 1/4, 1-f = 3/4
    let f_d3 = 1.0 / 4.0;
    let d3 = ((1.0 - f_d3) * (1.0 - f_d3) / (1.0 + 1.0))
        * (1.0 * ((1.0 / 4.0) * (n / cf)).log2() + 0.5 * (TAU * 1.0 * (1.0 - f_d3)).log2());

    assert_close(score_of(&hits, "d1"), d1, TIGHT, "DPH d1");
    assert_close(score_of(&hits, "d3"), d3, TIGHT, "DPH d3");

    // Worked out off the machine:
    //   d1 = (1/9)/3 * (2*log2(2/3) + 0.5*log2(4*pi/3))
    //      = 0.037037037 * (-1.169925001 + 1.033276893) = -0.005061414336
    //   d3 = (9/16)/2 * (log2(1/4)   + 0.5*log2(3*pi/2))
    //      = 0.28125     * (-2.0        + 1.118307838)   = -0.247998005129
    assert_close(
        score_of(&hits, "d1"),
        -0.005_061_414_336,
        HAND,
        "DPH d1 literal",
    );
    assert_close(
        score_of(&hits, "d3"),
        -0.247_998_005_129,
        HAND,
        "DPH d3 literal",
    );

    assert_close(
        dph_term_score(1.0, 2, 3, 3, 3),
        d1,
        TIGHT,
        "dph_term_score d1",
    );
}

// ── DFR singularities ────────────────────────────────────────────────────────

#[test]
fn dfr_contributes_exactly_zero_for_a_term_the_document_does_not_contain() {
    // tf = 0 sends log2(tfn) and log2(2*pi*tf) to -inf in both formulas. The
    // contribution is defined to be 0: DFR weighs *observed* divergence.
    assert_eq!(pl2_term_score(1.0, 0, 10, 5.0, 20, 100, 1.0), 0.0);
    assert_eq!(dph_term_score(1.0, 0, 10, 20, 100), 0.0);
    assert!(pl2_term_score(1.0, 0, 10, 5.0, 20, 100, 1.0).is_finite());
    assert!(dph_term_score(1.0, 0, 10, 20, 100).is_finite());
}

#[test]
fn dph_is_exactly_zero_when_the_document_is_nothing_but_the_query_term() {
    // tf = dl -> f = 1 -> the prefactor (1-f)^2 is 0 while the Stirling term
    // 0.5*log2(2*pi*tf*(1-f)) is -inf: a 0 * (-inf) indeterminate form. The
    // limit is exactly 0 because u^2 * log(u) -> 0, and 0 is what is returned.
    assert_eq!(dph_term_score(1.0, 1, 1, 10, 100), 0.0);
    assert_eq!(dph_term_score(1.0, 7, 7, 40, 100), 0.0);
    assert!(dph_term_score(1.0, 7, 7, 40, 100).is_finite());

    // Just short of the singularity the score is finite and small in magnitude
    // (the u^2 factor is already crushing it).
    let nearly = dph_term_score(1.0, 999, 1000, 5000, 100);
    assert!(nearly.is_finite(), "tf = dl - 1 must stay finite");
    assert!(nearly.abs() < 1.0, "u^2 should dominate: {nearly}");

    // ...and an index reproduces it end to end.
    let mut index =
        LmRetrievalIndex::new(LmRetrievalConfig::dfr(DfrModel::Dph)).expect("valid configuration");
    index
        .add_document("pure", "solo solo solo")
        .expect("unique");
    index
        .add_document("mixed", "solo other other")
        .expect("unique");
    index.build().expect("valid configuration");
    let hits = index.search("solo", 5).expect("built");
    assert_close(score_of(&hits, "pure"), 0.0, TIGHT, "tf = dl gives 0");
    assert!(hits.iter().all(|hit| hit.score.is_finite()));
}

#[test]
fn pl2_survives_a_vanishing_normalized_term_frequency() {
    // dl astronomically larger than c * avgdl drives tfn below PL2_MIN_TFN.
    // The naive Stirling formula would return -inf there.
    let vanishing = pl2_term_score(1.0, 1, 1_000_000_000_000, 100.0, 5, 1000, 1.0);
    assert!(
        vanishing.is_finite(),
        "a vanishing tfn must not produce -inf: {vanishing}"
    );
    assert!(
        vanishing < 0.0,
        "a vanishing tfn is a heavy penalty: {vanishing}"
    );

    // Even an absurdly small c (valid, but useless) stays finite.
    let degenerate = pl2_term_score(1.0, 3, 10, 5.0, 20, 100, 1e-300);
    assert!(degenerate.is_finite(), "c -> 0 must not produce -inf");

    // And an invalid c is refused before it can ever be scored.
    assert!(DfrModel::Pl2 { c: 0.0 }.validate().is_err());
    assert!(DfrModel::Pl2 { c: -1.0 }.validate().is_err());
    assert!(DfrModel::Pl2 { c: f64::NAN }.validate().is_err());
    assert!(DfrModel::Dph.validate().is_ok());
}

// ── Numerical stability under stress ─────────────────────────────────────────

#[test]
fn no_model_ever_produces_a_non_finite_score() {
    // A deliberately hostile corpus: empty documents, single-token documents,
    // documents that are a single term repeated (tf = dl), a very long
    // document, and pseudo-random filler. Queried with in-vocabulary terms,
    // out-of-vocabulary terms, and a mixture.
    let vocabulary = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    ];

    let build = |config: LmRetrievalConfig| {
        let mut index = LmRetrievalIndex::new(config).expect("valid configuration");
        index.add_document("empty", "").expect("unique");
        index.add_document("single", "alpha").expect("unique");
        index
            .add_document("all_one_term", "beta beta beta beta")
            .expect("unique");
        index
            .add_document("very_long", &"gamma ".repeat(5000))
            .expect("unique");
        index
            .add_document("punctuation", "!!! ??? ...")
            .expect("unique");

        let mut rng = SplitMix64::new(0x0BAD_C0FF_EE0D_DF00);
        for i in 0..40 {
            let length = rng.below(13);
            let text: Vec<&str> = (0..length)
                .map(|_| vocabulary[rng.below(vocabulary.len())])
                .collect();
            index
                .add_document(format!("rnd{i}"), &text.join(" "))
                .expect("unique");
        }
        index.build().expect("valid configuration");
        index
    };

    let configs = [
        LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: 0.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: 1e-9 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: 2000.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: 1e12 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::JelinekMercer { lambda: 0.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::JelinekMercer { lambda: 0.5 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::JelinekMercer { lambda: 1.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::AbsoluteDiscounting { delta: 0.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::AbsoluteDiscounting { delta: 0.5 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::AbsoluteDiscounting { delta: 1.0 }),
        LmRetrievalConfig::dfr(DfrModel::Pl2 { c: 1e-6 }),
        LmRetrievalConfig::dfr(DfrModel::Pl2 { c: 1.0 }),
        LmRetrievalConfig::dfr(DfrModel::Pl2 { c: 100.0 }),
        LmRetrievalConfig::dfr(DfrModel::Dph),
    ];

    let queries = [
        "alpha",
        "beta beta beta",
        "gamma delta epsilon",
        "zzz_out_of_vocabulary",
        "alpha zzz_out_of_vocabulary qqq_also_unseen",
        "eta theta zeta alpha beta gamma delta epsilon",
    ];

    for config in configs {
        let index = build(config);
        for query in queries {
            let hits = index.search(query, 100).expect("built, non-empty query");
            for hit in &hits {
                assert!(
                    hit.score.is_finite(),
                    "{config:?} scored {query:?} on {} as {}",
                    hit.document_id,
                    hit.score
                );
            }
            // Scores must be sorted descending -- which they cannot be if any
            // of them is NaN, since NaN poisons every comparison.
            for pair in hits.windows(2) {
                assert!(
                    pair[0].score >= pair[1].score,
                    "{config:?} produced an unsorted ranking for {query:?}"
                );
            }

            // RM3 over the same hostile corpus.
            let rm3 = Rm3Config::new(5, 10, 0.5);
            let expanded = index.expand_query_rm3(query, &rm3).expect("built");
            for term in &expanded.terms {
                assert!(
                    term.weight.is_finite() && term.weight > 0.0,
                    "{config:?} produced weight {} for {:?}",
                    term.weight,
                    term.term
                );
            }
            if !expanded.is_empty() {
                assert_close(
                    expanded.total_weight(),
                    1.0,
                    1e-9,
                    &format!("{config:?} expanded-query normalization"),
                );
            }
            for hit in index.search_rm3(query, &rm3, 10).expect("built") {
                assert!(hit.score.is_finite(), "{config:?} RM3 score was not finite");
            }
        }
    }
}

#[test]
fn an_empty_document_backs_off_entirely_to_the_collection_model() {
    let mut index = LmRetrievalIndex::new(LmRetrievalConfig::query_likelihood(
        LmSmoothing::Dirichlet { mu: 2.0 },
    ))
    .expect("valid configuration");
    index.add_document("empty", "").expect("unique");
    index
        .add_document("full", "apple apple banana")
        .expect("unique");
    index.build().expect("valid configuration");

    let stats = index.document_stats("empty").expect("indexed");
    assert_eq!(stats.length, 0);
    assert_eq!(stats.unique_terms, 0);

    // P(w | empty) = P(w | C) exactly, for every scheme.
    for smoothing in [
        LmSmoothing::Dirichlet { mu: 2.0 },
        LmSmoothing::Dirichlet { mu: 0.0 },
        LmSmoothing::JelinekMercer { lambda: 0.3 },
        LmSmoothing::AbsoluteDiscounting { delta: 0.7 },
    ] {
        let probability = smoothing.document_probability(0, 0, 0, 2.0 / 3.0);
        assert_close(
            probability,
            2.0 / 3.0,
            TIGHT,
            &format!("{smoothing:?} on an empty document"),
        );
    }

    let hits = index.search("apple", 2).expect("built");
    assert!(hits.iter().all(|hit| hit.score.is_finite()));
    // ln(P(apple|C)) = ln(2/3); the empty document scores exactly the prior.
    assert_close(
        score_of(&hits, "empty"),
        (2.0_f64 / 3.0).ln(),
        TIGHT,
        "empty document scores the collection model",
    );
}

// ── Log-sum-exp ──────────────────────────────────────────────────────────────

#[test]
fn log_sum_exp_is_exact_on_easy_inputs() {
    assert_close(lm_log_sum_exp(&[0.0]), 0.0, TIGHT, "logsumexp([0])");
    assert_close(
        lm_log_sum_exp(&[0.0, 0.0]),
        2.0_f64.ln(),
        TIGHT,
        "logsumexp([0,0])",
    );
    assert_close(
        lm_log_sum_exp(&[1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln()]),
        6.0_f64.ln(),
        TIGHT,
        "logsumexp of ln(1), ln(2), ln(3)",
    );
    assert_eq!(lm_log_sum_exp(&[]), f64::NEG_INFINITY);
}

#[test]
fn naive_exponentiation_of_a_query_likelihood_score_underflows_to_zero() {
    // This is the trap the whole of `rm3`'s posterior machinery exists to
    // avoid. It is not hypothetical: -1000 is an ordinary query-likelihood
    // score for a long query.
    assert_eq!((-1000.0_f64).exp(), 0.0);
    assert_eq!((-100_000.0_f64).exp(), 0.0);
    // A naive softmax over these scores divides 0 by 0.
    let naive_sum: f64 = [-1000.0_f64, -1001.0, -1002.0]
        .iter()
        .map(|s| s.exp())
        .sum();
    assert_eq!(naive_sum, 0.0);
    assert!((0.0_f64 / naive_sum).is_nan());
}

#[test]
fn log_sum_exp_posteriors_survive_extreme_negative_scores() {
    // The same posterior, computed from scores offset by 100 000.
    let ordinary = lm_query_posteriors(&[0.0, -1.0, -2.0]);
    let extreme = lm_query_posteriors(&[-100_000.0, -100_001.0, -100_002.0]);

    assert_eq!(extreme.len(), 3);
    for (index, (a, b)) in ordinary.iter().zip(extreme.iter()).enumerate() {
        assert!(*b > 0.0, "posterior {index} collapsed to zero");
        assert!(b.is_finite(), "posterior {index} was not finite");
        assert_close(*b, *a, 1e-12, "shift invariance of the posterior");
    }
    assert_close(
        extreme.iter().sum::<f64>(),
        1.0,
        1e-12,
        "posteriors sum to 1",
    );

    // exp(0) / (exp(0) + exp(-1) + exp(-2)) = 1 / (1 + e^-1 + e^-2)
    let expected_first = 1.0 / (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp());
    assert_close(extreme[0], expected_first, 1e-12, "leading posterior");

    // Log-sum-exp itself keeps the offset.
    assert_close(
        lm_log_sum_exp(&[-100_000.0, -100_000.0]),
        -100_000.0 + 2.0_f64.ln(),
        1e-9,
        "logsumexp of two equal extreme scores",
    );

    // Degenerate inputs.
    assert!(lm_query_posteriors(&[]).is_empty());
    let uniform = lm_query_posteriors(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
    assert_close(uniform[0], 0.5, TIGHT, "all -inf falls back to uniform");
    assert_close(uniform.iter().sum::<f64>(), 1.0, TIGHT, "uniform sums to 1");
}

// ── RM3 ──────────────────────────────────────────────────────────────────────

/// The `RM3` corpus.
///
/// ```text
/// d1 = renal renal kidney kidney kidney failure     |D| = 6
/// d2 = renal kidney kidney kidney disease           |D| = 5
/// d3 = kidney kidney kidney transplant surgery      |D| = 5   <- no "renal"
/// d4 = automobile engine transmission gearbox       |D| = 4
/// d5 = weather forecast rain snow                   |D| = 4
///
/// N = 5   |C| = 24   cf(renal) = 3   cf(kidney) = 9
/// ```
///
/// The query "renal" cannot find `d3`, which never uses the word — but the two
/// documents it *does* find are saturated with "kidney", and `RM3` reads that
/// straight off their language models.
fn renal_index(mu: f64) -> LmRetrievalIndex {
    let mut index = LmRetrievalIndex::new(LmRetrievalConfig::query_likelihood(
        LmSmoothing::Dirichlet { mu },
    ))
    .expect("valid configuration");
    index
        .add_document("d1", "renal renal kidney kidney kidney failure")
        .expect("unique");
    index
        .add_document("d2", "renal kidney kidney kidney disease")
        .expect("unique");
    index
        .add_document("d3", "kidney kidney kidney transplant surgery")
        .expect("unique");
    index
        .add_document("d4", "automobile engine transmission gearbox")
        .expect("unique");
    index
        .add_document("d5", "weather forecast rain snow")
        .expect("unique");
    index.build().expect("valid configuration");
    index
}

#[test]
fn rm1_relevance_model_is_a_normalized_distribution() {
    let index = renal_index(10.0);
    let config = Rm3Config::new(2, 4, 0.5);
    let model = index.relevance_model("renal", &config).expect("built");

    // Closed form: sum over the vocabulary of S(w) + K.
    assert_close(
        model.total_probability_mass(),
        1.0,
        1e-12,
        "RM1 closed-form normalization",
    );

    // Brute force: materialize P(w|R) over every term in the vocabulary and add
    // it up. This is the check the closed form is a shortcut for.
    let distribution = index.materialize_relevance_model(&model);
    assert_eq!(distribution.len(), index.vocabulary_size());
    let mass: f64 = distribution.iter().map(|(_, p)| *p).sum();
    assert_close(mass, 1.0, 1e-9, "RM1 materialized normalization");
    assert!(distribution.iter().all(|(_, p)| *p > 0.0 && p.is_finite()));

    // The posteriors P(D|Q) over the feedback set are a distribution too.
    let posteriors = model.document_posteriors();
    assert_eq!(posteriors.len(), 2);
    assert_eq!(posteriors[0].0, "d1");
    assert_eq!(posteriors[1].0, "d2");
    assert_close(
        posteriors.iter().map(|(_, p)| *p).sum::<f64>(),
        1.0,
        1e-12,
        "P(D|Q) sums to 1",
    );

    // Hand computation of the posteriors.
    //   score(d1) = ln((2 + 10*(3/24)) / (6+10)) = ln(3.25/16) = -1.593934...
    //   score(d2) = ln((1 + 10*(3/24)) / (5+10)) = ln(2.25/15) = -1.897120...
    //   pi_1 = e^s1 / (e^s1 + e^s2) = 0.203125 / (0.203125 + 0.15) = 0.575221...
    let s1 = ((2.0_f64 + 10.0 * (3.0 / 24.0)) / 16.0).ln();
    let s2 = ((1.0_f64 + 10.0 * (3.0 / 24.0)) / 15.0).ln();
    let pi1 = s1.exp() / (s1.exp() + s2.exp());
    assert_close(posteriors[0].1, pi1, TIGHT, "pi(d1)");
    assert_close(posteriors[0].1, 0.575_221, HAND, "pi(d1) literal");
    assert_close(posteriors[1].1, 0.424_779, HAND, "pi(d2) literal");
}

#[test]
fn rm1_weights_match_hand_computation() {
    let index = renal_index(10.0);
    let config = Rm3Config::new(2, 4, 0.5);
    let model = index.relevance_model("renal", &config).expect("built");

    let s1 = ((2.0_f64 + 10.0 * (3.0 / 24.0)) / 16.0).ln();
    let s2 = ((1.0_f64 + 10.0 * (3.0 / 24.0)) / 15.0).ln();
    let pi1 = s1.exp() / (s1.exp() + s2.exp());
    let pi2 = 1.0 - pi1;

    // Dirichlet: S_D(w) = tf / (|D| + mu),  K_D = mu / (|D| + mu).
    let k = pi1 * (10.0 / 16.0) + pi2 * (10.0 / 15.0);
    assert_close(model.background_coefficient(), k, TIGHT, "K");
    assert_close(model.background_coefficient(), 0.642_699, HAND, "K literal");

    // P(kidney|R) = [pi1*3/16 + pi2*3/15] + K * (9/24)
    let p_kidney = (pi1 * 3.0 / 16.0 + pi2 * 3.0 / 15.0) + k * (9.0 / 24.0);
    // P(renal|R)  = [pi1*2/16 + pi2*1/15] + K * (3/24)
    let p_renal = (pi1 * 2.0 / 16.0 + pi2 * 1.0 / 15.0) + k * (3.0 / 24.0);

    let weights: HashMap<&str, f64> = model
        .weights()
        .iter()
        .map(|(term, weight)| (term.as_str(), *weight))
        .collect();

    assert_close(weights["kidney"], p_kidney, TIGHT, "P(kidney|R)");
    assert_close(weights["renal"], p_renal, TIGHT, "P(renal|R)");
    assert_close(weights["kidney"], 0.433_822, HAND, "P(kidney|R) literal");
    assert_close(weights["renal"], 0.180_559, HAND, "P(renal|R) literal");

    // Only the feedback documents' terms are listed; "transplant" (d3) is not
    // in the feedback set, so it carries no foreground mass -- but it still has
    // a well-defined, non-zero relevance probability of K * P(w|C).
    assert!(!weights.contains_key("transplant"));
    assert_close(
        model.probability("transplant", index.collection_probability("transplant")),
        k * (1.0 / 24.0),
        TIGHT,
        "P(transplant|R) = K * P(w|C)",
    );
    assert_close(
        model.foreground("transplant"),
        0.0,
        TIGHT,
        "S(transplant) = 0",
    );

    // "kidney" -- absent from the query -- is the heaviest term in the model.
    assert_eq!(model.weights()[0].0, "kidney");
}

#[test]
fn rm3_surfaces_a_term_the_query_never_used_and_retrieves_the_document_it_missed() {
    let index = renal_index(10.0);
    let config = Rm3Config::new(2, 4, 0.5);

    // The original query cannot reach d3: it does not contain "renal", so under
    // Dirichlet smoothing it scores below the two documents that are about cars
    // and weather (both of which are shorter, and so are smoothed less harshly).
    let original = index.search("renal", 3).expect("built");
    assert_eq!(ranking(&original), vec!["d1", "d2", "d4"]);
    assert!(!original.iter().any(|hit| hit.document_id == "d3"));

    let expanded = index.expand_query_rm3("renal", &config).expect("built");

    // "kidney" is surfaced with real weight -- a quarter of the whole query.
    assert!(expanded.contains("kidney"));
    assert!(
        expanded.weight_of("kidney") > 0.1,
        "kidney weight was only {}",
        expanded.weight_of("kidney")
    );
    assert_close(
        expanded.total_weight(),
        1.0,
        TIGHT,
        "expanded query is normalized",
    );

    // Hand computation of every expanded weight.
    //   P'(w) = alpha * P_ML(w|Q) + (1 - alpha) * P(w|R),  alpha = 0.5,
    //   P_ML(renal|Q) = 1 (the query is one term).
    let s1 = ((2.0_f64 + 10.0 * (3.0 / 24.0)) / 16.0).ln();
    let s2 = ((1.0_f64 + 10.0 * (3.0 / 24.0)) / 15.0).ln();
    let pi1 = s1.exp() / (s1.exp() + s2.exp());
    let pi2 = 1.0 - pi1;
    let k = pi1 * (10.0 / 16.0) + pi2 * (10.0 / 15.0);

    let p_renal = (pi1 * 2.0 / 16.0 + pi2 * 1.0 / 15.0) + k * (3.0 / 24.0);
    let p_kidney = (pi1 * 3.0 / 16.0 + pi2 * 3.0 / 15.0) + k * (9.0 / 24.0);
    let p_failure = (pi1 * 1.0 / 16.0) + k * (1.0 / 24.0);
    let p_disease = (pi2 * 1.0 / 15.0) + k * (1.0 / 24.0);

    let raw_renal = 0.5 * 1.0 + 0.5 * p_renal;
    let raw_kidney = 0.5 * p_kidney;
    let raw_failure = 0.5 * p_failure;
    let raw_disease = 0.5 * p_disease;
    let total = raw_renal + raw_kidney + raw_failure + raw_disease;

    assert_eq!(expanded.len(), 4);
    assert_close(
        expanded.weight_of("renal"),
        raw_renal / total,
        TIGHT,
        "w(renal)",
    );
    assert_close(
        expanded.weight_of("kidney"),
        raw_kidney / total,
        TIGHT,
        "w(kidney)",
    );
    assert_close(
        expanded.weight_of("failure"),
        raw_failure / total,
        TIGHT,
        "w(failure)",
    );
    assert_close(
        expanded.weight_of("disease"),
        raw_disease / total,
        TIGHT,
        "w(disease)",
    );

    assert_close(
        expanded.weight_of("renal"),
        0.681_534,
        HAND,
        "w(renal) literal",
    );
    assert_close(
        expanded.weight_of("kidney"),
        0.250_444,
        HAND,
        "w(kidney) literal",
    );
    assert_close(
        expanded.weight_of("failure"),
        0.036_214,
        HAND,
        "w(failure) literal",
    );
    assert_close(
        expanded.weight_of("disease"),
        0.031_808,
        HAND,
        "w(disease) literal",
    );

    // The expansion win: d3 -- which the original query could not reach at all
    // -- is now the third hit.
    let after = index.search_rm3("renal", &config, 3).expect("built");
    assert_eq!(ranking(&after), vec!["d1", "d2", "d3"]);
    assert!(after.iter().any(|hit| hit.document_id == "d3"));
    assert!(after.iter().all(|hit| hit.score.is_finite()));
}

#[test]
fn rm3_at_alpha_one_reduces_exactly_to_the_original_query() {
    let index = renal_index(10.0);

    // A multi-term query, so that P_ML(w|Q) is not trivially 1.
    let expanded = index
        .expand_query_rm3("renal renal failure", &Rm3Config::new(2, 10, 1.0))
        .expect("built");

    // alpha = 1 annihilates (1 - alpha) * P(w|R) exactly, so no feedback term
    // can be selected however large fb_terms is, and the weights are P_ML.
    assert_eq!(expanded.len(), 2);
    assert_close(
        expanded.weight_of("renal"),
        2.0 / 3.0,
        TIGHT,
        "P_ML(renal|Q)",
    );
    assert_close(
        expanded.weight_of("failure"),
        1.0 / 3.0,
        TIGHT,
        "P_ML(failure|Q)",
    );
    assert!(!expanded.contains("kidney"));
    assert_close(
        expanded.total_weight(),
        1.0,
        TIGHT,
        "alpha = 1 is normalized",
    );

    // And the ranking is the one the unexpanded query produces.
    let plain = index.search("renal renal failure", 5).expect("built");
    let rm3 = index
        .search_rm3("renal renal failure", &Rm3Config::new(2, 10, 1.0), 5)
        .expect("built");
    assert_eq!(ranking(&plain), ranking(&rm3));
}

#[test]
fn rm1_posteriors_do_not_collapse_when_query_likelihoods_are_extreme() {
    let index = renal_index(10.0);

    // A 600-fold query. score_QL(d1) = 600 * ln(0.203125) = -956.4, and
    // exp(-956.4) is *exactly zero* in f64 -- a naive softmax would return NaN
    // for every feedback document, and the relevance model would be destroyed.
    let query = "renal ".repeat(600);
    let s1 = 600.0 * ((2.0_f64 + 10.0 * (3.0 / 24.0)) / 16.0).ln();
    assert!(
        s1 < -900.0,
        "the test corpus must actually produce extreme scores"
    );
    assert_eq!(
        s1.exp(),
        0.0,
        "the naive computation must genuinely underflow"
    );

    let model = index
        .relevance_model(&query, &Rm3Config::new(3, 10, 0.5))
        .expect("built");

    let posteriors = model.document_posteriors();
    assert_eq!(posteriors.len(), 3);
    assert!(posteriors.iter().all(|(_, p)| p.is_finite()));
    assert_close(
        posteriors.iter().map(|(_, p)| *p).sum::<f64>(),
        1.0,
        1e-12,
        "extreme posteriors still sum to 1",
    );
    // With a 600-fold query the likelihood ratio is astronomical, so the
    // posterior collapses onto the best document -- correctly, and *finitely*.
    assert!(
        posteriors[0].1 > 0.99,
        "expected the posterior to concentrate on d1, got {}",
        posteriors[0].1
    );
    assert_close(
        model.total_probability_mass(),
        1.0,
        1e-9,
        "still a distribution",
    );
    assert!(
        model
            .weights()
            .iter()
            .all(|(_, w)| w.is_finite() && *w > 0.0)
    );
}

#[test]
fn rm3_feedback_set_is_chosen_by_the_configured_model_but_weighted_by_query_likelihood() {
    // A DFR index still produces a coherent relevance model: DPH picks the
    // feedback documents, query likelihood weights them (a DFR divergence is
    // not a log-probability, so exponentiating it would mean nothing).
    let mut index =
        LmRetrievalIndex::new(LmRetrievalConfig::dfr(DfrModel::Dph)).expect("valid configuration");
    index
        .add_document("d1", "renal renal kidney kidney kidney failure")
        .expect("unique");
    index
        .add_document("d2", "renal kidney kidney kidney disease")
        .expect("unique");
    index
        .add_document("d3", "kidney kidney kidney transplant surgery")
        .expect("unique");
    index.build().expect("valid configuration");

    let model = index
        .relevance_model("renal", &Rm3Config::new(2, 5, 0.5))
        .expect("built");

    // DPH only scores the posting list of "renal", so the feedback set is
    // {d1, d2} -- d3 is not a candidate at all.
    let posteriors = model.document_posteriors();
    assert_eq!(posteriors.len(), 2);
    assert!(posteriors.iter().all(|(id, _)| id != "d3"));
    assert_close(
        posteriors.iter().map(|(_, p)| *p).sum::<f64>(),
        1.0,
        1e-12,
        "posteriors sum to 1",
    );
    // ...and the weights are still query likelihoods, so still a distribution.
    assert_close(
        model.total_probability_mass(),
        1.0,
        1e-9,
        "still a distribution",
    );
    assert!(model.weights().iter().any(|(term, _)| term == "kidney"));
}

// ── API surface and errors ───────────────────────────────────────────────────

#[test]
fn searching_an_unbuilt_index_is_an_error() {
    let mut index = LmRetrievalIndex::new(LmRetrievalConfig::default()).expect("valid");
    index.add_document("d1", "apple").expect("unique");
    assert!(!index.is_built());
    assert_eq!(index.search("apple", 5), Err(LmRetrievalError::NotBuilt));

    index.build().expect("valid");
    assert!(index.is_built());
    assert!(index.search("apple", 5).is_ok());

    // Adding a document invalidates the derived statistics.
    index.add_document("d2", "banana").expect("unique");
    assert!(!index.is_built());
    assert_eq!(index.search("apple", 5), Err(LmRetrievalError::NotBuilt));
    assert_eq!(
        index.relevance_model("apple", &Rm3Config::default()),
        Err(LmRetrievalError::NotBuilt)
    );

    index.build().expect("valid");
    assert_eq!(index.total_tokens(), 2);
}

#[test]
fn duplicate_document_ids_are_rejected() {
    let mut index = LmRetrievalIndex::new(LmRetrievalConfig::default()).expect("valid");
    index.add_document("d1", "apple").expect("unique");
    assert_eq!(
        index.add_document("d1", "banana"),
        Err(LmRetrievalError::DuplicateDocumentId {
            document_id: "d1".to_string()
        })
    );
}

#[test]
fn an_empty_query_is_an_error() {
    let index = fruit_index(LmSmoothing::default());
    assert_eq!(index.search("", 5), Err(LmRetrievalError::EmptyQuery));
    assert_eq!(
        index.search("!!! ???", 5),
        Err(LmRetrievalError::EmptyQuery)
    );
    assert_eq!(
        index.search_weighted(&[], 5),
        Err(LmRetrievalError::EmptyQuery)
    );
}

#[test]
fn invalid_hyper_parameters_are_rejected_at_construction() {
    let invalid = [
        LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: -1.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: f64::NAN }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: f64::INFINITY }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::JelinekMercer { lambda: -0.1 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::JelinekMercer { lambda: 1.1 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::AbsoluteDiscounting { delta: 1.5 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::AbsoluteDiscounting { delta: -0.5 }),
        LmRetrievalConfig::dfr(DfrModel::Pl2 { c: 0.0 }),
        LmRetrievalConfig::dfr(DfrModel::Pl2 { c: -3.0 }),
    ];
    for config in invalid {
        assert!(
            LmRetrievalIndex::new(config).is_err(),
            "{config:?} should have been rejected"
        );
    }

    // Valid boundary values are accepted.
    for config in [
        LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: 0.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::JelinekMercer { lambda: 0.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::JelinekMercer { lambda: 1.0 }),
        LmRetrievalConfig::query_likelihood(LmSmoothing::AbsoluteDiscounting { delta: 1.0 }),
        LmRetrievalConfig::dfr(DfrModel::Dph),
    ] {
        assert!(LmRetrievalIndex::new(config).is_ok(), "{config:?} is valid");
    }
}

#[test]
fn invalid_rm3_configurations_are_rejected() {
    let index = fruit_index(LmSmoothing::default());
    for config in [
        Rm3Config::new(0, 5, 0.5),
        Rm3Config::new(5, 0, 0.5),
        Rm3Config::new(5, 5, -0.1),
        Rm3Config::new(5, 5, 1.1),
        Rm3Config::new(5, 5, f64::NAN),
    ] {
        assert!(
            index.relevance_model("apple", &config).is_err(),
            "{config:?} should have been rejected"
        );
    }
    assert!(Rm3Config::default().validate().is_ok());
}

#[test]
fn config_builders_and_defaults_behave() {
    let config = LmRetrievalConfig::default()
        .with_model(LmScoringModel::Dfr(DfrModel::Dph))
        .with_smoothing(LmSmoothing::JelinekMercer { lambda: 0.2 })
        .with_lowercase(false);
    assert_eq!(config.model, LmScoringModel::Dfr(DfrModel::Dph));
    assert_eq!(config.smoothing, LmSmoothing::JelinekMercer { lambda: 0.2 });
    assert!(!config.lowercase);
    assert!(config.validate().is_ok());

    // The default is Dirichlet(2000) query likelihood, per Zhai & Lafferty.
    assert_eq!(
        LmRetrievalConfig::default().smoothing,
        LmSmoothing::Dirichlet { mu: 2000.0 }
    );
    assert_eq!(
        LmRetrievalConfig::default().model,
        LmScoringModel::QueryLikelihood
    );
    // A DFR configuration still carries a smoothing scheme, because RM3 needs
    // one whatever produced the first-pass ranking.
    assert_eq!(
        LmRetrievalConfig::dfr(DfrModel::Dph).smoothing,
        LmSmoothing::default()
    );
}

#[test]
fn top_k_and_empty_index_edge_cases() {
    let index = fruit_index(LmSmoothing::default());
    assert!(index.search("apple", 0).expect("built").is_empty());
    assert_eq!(index.search("apple", 1).expect("built").len(), 1);
    assert_eq!(index.search("apple", 100).expect("built").len(), 3);

    let mut empty = LmRetrievalIndex::new(LmRetrievalConfig::default()).expect("valid");
    empty.build().expect("valid");
    assert!(empty.is_empty());
    assert_eq!(empty.num_documents(), 0);
    assert!(empty.search("apple", 5).expect("built").is_empty());
    assert_close(
        empty.average_document_length(),
        0.0,
        TIGHT,
        "avgdl of nothing",
    );

    // An out-of-vocabulary-only query against a DFR index has no candidates at
    // all -- an empty result, not an error and not a NaN.
    let dfr = fruit_dfr_index(DfrModel::Dph);
    assert!(dfr.search("zzz_unseen", 5).expect("built").is_empty());
    // Query likelihood, by contrast, scores every document (all identically,
    // since the term is unseen everywhere).
    let ql = fruit_index(LmSmoothing::Dirichlet { mu: 2.0 });
    let hits = ql.search("zzz_unseen", 5).expect("built");
    assert_eq!(hits.len(), 3);
    assert!(hits.iter().all(|hit| hit.score.is_finite()));
}

#[test]
fn ranks_are_dense_and_ties_break_deterministically() {
    let index = fruit_index(LmSmoothing::default());
    let hits = index.search("apple cherry", 3).expect("built");
    for (position, hit) in hits.iter().enumerate() {
        assert_eq!(hit.rank, position);
    }

    // Two identical documents tie exactly and are ordered by document id.
    let mut tied = LmRetrievalIndex::new(LmRetrievalConfig::default()).expect("valid");
    tied.add_document("zeta", "apple apple").expect("unique");
    tied.add_document("alpha", "apple apple").expect("unique");
    tied.build().expect("valid");
    let hits = tied.search("apple", 2).expect("built");
    assert_eq!(ranking(&hits), vec!["alpha", "zeta"]);
    assert_close(
        hits[0].score,
        hits[1].score,
        TIGHT,
        "identical documents tie",
    );
}

#[test]
fn document_statistics_are_exact() {
    let index = fruit_index(LmSmoothing::default());
    let d3 = index.document_stats("d3").expect("indexed");
    assert_eq!(d3.length, 4);
    assert_eq!(d3.unique_terms, 3);
    assert_eq!(d3.term_frequency("cherry"), 2);
    assert_eq!(d3.term_frequency("apple"), 1);
    assert_eq!(d3.term_frequency("banana"), 0);
    assert!(index.document_stats("nonexistent").is_none());
    assert_eq!(index.documents().len(), 3);
}

#[test]
fn pre_tokenized_documents_bypass_the_tokenizer() {
    let mut index = LmRetrievalIndex::new(LmRetrievalConfig::query_likelihood(
        LmSmoothing::Dirichlet { mu: 1.0 },
    ))
    .expect("valid");
    let terms = vec!["Renal-Failure".to_string(), "KIDNEY".to_string()];
    index.add_document_terms("stemmed", &terms).expect("unique");
    index.add_document_terms("blank", &[]).expect("unique");
    index.build().expect("valid");

    assert_eq!(index.collection_frequency("Renal-Failure"), 1);
    assert_eq!(index.document_stats("blank").expect("indexed").length, 0);
    assert_eq!(index.total_tokens(), 2);
}

#[test]
fn ln_2_and_log2_agree_with_the_dfr_derivation() {
    // The DFR code computes log2 via ln_1p / LN_2 for accuracy at small
    // arguments; confirm that equals the direct log2 where both are well
    // conditioned, so the hand-computed expectations above are comparable.
    for x in [0.5_f64, 1.0, 3.0, 1e-6] {
        assert_close(
            x.ln_1p() / LN_2,
            (1.0 + x).log2(),
            1e-12,
            "ln_1p / LN_2 == log2(1+x)",
        );
    }
    // And that it is *more* accurate than the naive form at tiny arguments,
    // where (1.0 + x) rounds to exactly 1.0 and log2 returns 0.
    let tiny = 1e-20_f64;
    assert_eq!((1.0 + tiny).log2(), 0.0);
    assert!(tiny.ln_1p() / LN_2 > 0.0);
}
