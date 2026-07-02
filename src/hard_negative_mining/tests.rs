//! Comprehensive tests for the `hard_negative_mining` module.

#![allow(clippy::too_many_lines)]
#![allow(clippy::float_cmp)]
#![allow(clippy::unreadable_literal)]

use crate::hard_negative_mining::engine::HardNegativeMiner;
use crate::hard_negative_mining::miner::{cosine, embed};
use crate::hard_negative_mining::types::{
    EmbeddingVersion, HardNegativeConfig, HardNegativeDocument, HardNegativeError,
    HardNegativePositivePair, MiningRound,
};
use crate::types::DocumentId;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn doc(id: &str, text: &str) -> HardNegativeDocument {
    HardNegativeDocument::new(id, text)
}

fn pair(query: &str, positive: &str) -> HardNegativePositivePair {
    HardNegativePositivePair::new(query, positive)
}

fn id(value: &str) -> DocumentId {
    DocumentId::from_string(value)
}

/// L2 norm of a vector, for unit-norm assertions.
fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// A corpus with strictly graded token overlap against the query
/// `"one two three four five"`: `c5` shares all five, `c4` four, …, `c0` none,
/// plus six fully-disjoint filler documents to give a scrambling version room
/// to reorder the ranking.
fn graded_corpus() -> Vec<HardNegativeDocument> {
    vec![
        doc("c5", "one two three four five"),
        doc("c4", "one two three four zulu"),
        doc("c3", "one two three yankee zulu"),
        doc("c2", "one two xray yankee zulu"),
        doc("c1", "one whiskey xray yankee zulu"),
        doc("c0", "victor whiskey xray yankee zulu"),
        doc("e1", "papa quebec romeo sierra tango"),
        doc("e2", "alpha bravo charlie delta echo"),
        doc("e3", "foxtrot golf hotel india juliet"),
        doc("e4", "kilo lima mike november oscar"),
        doc("e5", "aaa bbb ccc ddd eee"),
        doc("e6", "fff ggg hhh iii jjj"),
    ]
}

fn graded_positives() -> Vec<HardNegativePositivePair> {
    vec![pair("one two three four five", "c5")]
}

// ── Embedding primitives ──────────────────────────────────────────────────────

#[test]
fn embed_is_deterministic() {
    let version = EmbeddingVersion::new(7, 0.4);
    let a = embed("hard negative mining is fun", version, 64);
    let b = embed("hard negative mining is fun", version, 64);
    assert_eq!(a, b);
}

#[test]
fn embed_canonical_ignores_seed() {
    // At drift 0.0 the seed is irrelevant: the pure content embedding.
    let a = embed("the quick brown fox", EmbeddingVersion::new(0, 0.0), 128);
    let b = embed(
        "the quick brown fox",
        EmbeddingVersion::new(987654321, 0.0),
        128,
    );
    assert_eq!(a, b);
    assert_eq!(
        a,
        embed("the quick brown fox", EmbeddingVersion::canonical(), 128)
    );
}

#[test]
fn embed_differs_by_seed_under_drift() {
    let a = embed("the quick brown fox", EmbeddingVersion::new(1, 0.5), 128);
    let b = embed("the quick brown fox", EmbeddingVersion::new(2, 0.5), 128);
    assert_ne!(a, b);
}

#[test]
fn embed_differs_by_drift() {
    let a = embed("the quick brown fox", EmbeddingVersion::new(5, 0.0), 128);
    let b = embed("the quick brown fox", EmbeddingVersion::new(5, 0.9), 128);
    assert_ne!(a, b);
}

#[test]
fn embed_is_unit_norm_for_nonempty_text() {
    let v = embed("alpha beta gamma delta", EmbeddingVersion::new(3, 0.3), 64);
    assert!((norm(&v) - 1.0).abs() < 1e-5, "norm was {}", norm(&v));
}

#[test]
fn embed_zero_dim_is_empty() {
    assert!(embed("anything", EmbeddingVersion::canonical(), 0).is_empty());
}

#[test]
fn cosine_of_identical_text_is_one() {
    let v = embed("alpha beta gamma", EmbeddingVersion::canonical(), 256);
    assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
}

#[test]
fn cosine_of_disjoint_text_is_low() {
    let a = embed("alpha beta gamma", EmbeddingVersion::canonical(), 256);
    let b = embed("zulu yankee xray", EmbeddingVersion::canonical(), 256);
    assert!(cosine(&a, &b) < 0.1, "cosine was {}", cosine(&a, &b));
}

#[test]
fn cosine_mismatched_lengths_is_zero() {
    assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0, 0.0]), 0.0);
}

#[test]
fn content_similarity_orders_by_token_overlap() {
    let version = EmbeddingVersion::canonical();
    let q = embed("apple banana cherry", version, 256);
    let high = embed("apple banana cherry", version, 256);
    let mid = embed("apple banana date", version, 256);
    let low = embed("elderberry fig grape", version, 256);
    let s_high = cosine(&q, &high);
    let s_mid = cosine(&q, &mid);
    let s_low = cosine(&q, &low);
    assert!(s_high > s_mid, "{s_high} !> {s_mid}");
    assert!(s_mid > s_low, "{s_mid} !> {s_low}");
}

// ── EmbeddingVersion ──────────────────────────────────────────────────────────

#[test]
fn embedding_version_constructors() {
    let v = EmbeddingVersion::new(42, 0.25);
    assert_eq!(v.seed, 42);
    assert_eq!(v.drift, 0.25);

    let canonical = EmbeddingVersion::canonical();
    assert_eq!(canonical.seed, 0);
    assert_eq!(canonical.drift, 0.0);
    assert_eq!(EmbeddingVersion::default(), canonical);

    assert_eq!(v.with_seed(7).seed, 7);
    assert_eq!(v.with_drift(0.9).drift, 0.9);
}

#[test]
fn embedding_version_effective_drift_clamps() {
    assert_eq!(EmbeddingVersion::new(0, -1.0).effective_drift(), 0.0);
    assert_eq!(EmbeddingVersion::new(0, 2.0).effective_drift(), 1.0);
    assert_eq!(EmbeddingVersion::new(0, 0.5).effective_drift(), 0.5);
}

// ── Config ────────────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let c = HardNegativeConfig::default();
    assert_eq!(c.negatives_per_query, 5);
    assert_eq!(c.hard_rank_cutoff, 10);
    assert_eq!(c.embedding_dim, 64);
    assert!(c.exclude_positive);
}

#[test]
fn config_builders() {
    let c = HardNegativeConfig::new()
        .with_negatives_per_query(3)
        .with_hard_rank_cutoff(7)
        .with_embedding_dim(32)
        .with_exclude_positive(false);
    assert_eq!(c.negatives_per_query, 3);
    assert_eq!(c.hard_rank_cutoff, 7);
    assert_eq!(c.embedding_dim, 32);
    assert!(!c.exclude_positive);
}

#[test]
fn config_validate_ok() {
    assert!(HardNegativeConfig::default().validate().is_ok());
}

#[test]
fn config_validate_zero_negatives() {
    let c = HardNegativeConfig::new().with_negatives_per_query(0);
    assert_eq!(
        c.validate().unwrap_err(),
        HardNegativeError::ZeroNegativesPerQuery
    );
}

#[test]
fn config_validate_zero_cutoff() {
    let c = HardNegativeConfig::new().with_hard_rank_cutoff(0);
    assert_eq!(
        c.validate().unwrap_err(),
        HardNegativeError::ZeroHardRankCutoff
    );
}

#[test]
fn config_validate_zero_dim() {
    let c = HardNegativeConfig::new().with_embedding_dim(0);
    assert_eq!(
        c.validate().unwrap_err(),
        HardNegativeError::ZeroEmbeddingDim
    );
}

// ── Small data-structure helpers ──────────────────────────────────────────────

#[test]
fn document_and_pair_constructors() {
    let d = doc("x", "hello world");
    assert_eq!(d.id, id("x"));
    assert_eq!(d.text, "hello world");

    let p = pair("who?", "x");
    assert_eq!(p.query, "who?");
    assert_eq!(p.positive_id, id("x"));
}

#[test]
fn query_result_negative_ids_and_contains() {
    let corpus = vec![
        doc("p", "alpha beta gamma"),
        doc("n1", "alpha beta delta"),
        doc("n2", "alpha epsilon zeta"),
    ];
    let positives = vec![pair("alpha beta gamma", "p")];
    let config = HardNegativeConfig::new()
        .with_embedding_dim(256)
        .with_hard_rank_cutoff(3)
        .with_negatives_per_query(2);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    let qr = round.query_result("alpha beta gamma").unwrap();
    assert_eq!(
        qr.negative_ids(),
        qr.negatives
            .iter()
            .map(|s| s.doc_id.clone())
            .collect::<Vec<_>>()
    );
    assert!(!qr.contains_negative(&id("p")));
    for negative in &qr.negatives {
        assert!(qr.contains_negative(&negative.doc_id));
    }
}

#[test]
fn mining_round_helpers() {
    let corpus = vec![
        doc("p", "alpha beta gamma"),
        doc("n1", "alpha beta delta"),
        doc("n2", "alpha beta epsilon"),
    ];
    let positives = vec![pair("alpha beta gamma", "p")];
    let config = HardNegativeConfig::new()
        .with_embedding_dim(256)
        .with_hard_rank_cutoff(3)
        .with_negatives_per_query(2);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    let expected_total: usize = round.results.iter().map(|r| r.negatives.len()).sum();
    assert_eq!(round.total_negatives(), expected_total);
    assert!(round.query_result("no such query").is_none());
    assert!(round.negative_ids("no such query").is_none());
    assert!(round.negative_ids("alpha beta gamma").is_some());
}

// ── Core mining behaviour ─────────────────────────────────────────────────────

#[test]
fn mine_excludes_true_positive_by_default() {
    let corpus = graded_corpus();
    let positives = graded_positives();
    let config = HardNegativeConfig::default().with_embedding_dim(256);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    for result in &round.results {
        assert!(
            !result.contains_negative(&result.positive_id),
            "positive {} was mined as its own negative",
            result.positive_id
        );
    }
}

#[test]
fn hard_negative_beats_easy_negative() {
    // Crafted corpus: `d_hard` shares every query token (ranked just below the
    // identical positive); `d_easy` shares none (ranked far below the cutoff).
    // Only the hard negative must be selected.
    let corpus = vec![
        doc("d_pos", "alpha beta gamma"),
        doc("d_hard", "alpha beta gamma delta"),
        doc("d_easy", "zulu yankee xray whiskey"),
    ];
    let positives = vec![pair("alpha beta gamma", "d_pos")];
    let config = HardNegativeConfig::new()
        .with_negatives_per_query(1)
        .with_hard_rank_cutoff(2)
        .with_embedding_dim(256);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    let qr = round.query_result("alpha beta gamma").unwrap();

    assert_eq!(qr.positive_rank, 1, "identical positive must rank first");
    assert_eq!(qr.negatives.len(), 1);
    assert_eq!(qr.negatives[0].doc_id, id("d_hard"));
    assert!(
        !qr.contains_negative(&id("d_easy")),
        "the disjoint easy negative must never be mined"
    );
    assert!(!qr.contains_negative(&id("d_pos")));
    // The hard negative outranks (is more similar than) the easy one.
    assert!(qr.negatives[0].similarity > 0.5);
}

#[test]
fn mean_positive_rank_is_hand_checkable() {
    let corpus = vec![
        doc("a1", "apple banana cherry"),
        doc("a2", "apple banana"),
        doc("a3", "date elderberry fig"),
    ];
    // Query 1's positive is the identical doc (rank 1); query 2 ("apple") puts
    // its labelled positive `a3` (which lacks "apple") at rank 3.
    let positives = vec![pair("apple banana cherry", "a1"), pair("apple", "a3")];
    let config = HardNegativeConfig::new()
        .with_embedding_dim(256)
        .with_hard_rank_cutoff(3)
        .with_negatives_per_query(2);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    assert_eq!(
        round
            .query_result("apple banana cherry")
            .unwrap()
            .positive_rank,
        1
    );
    assert_eq!(round.query_result("apple").unwrap().positive_rank, 3);
    assert_eq!(round.mean_positive_rank, 2.0);
}

#[test]
fn budget_n_is_respected() {
    let corpus = graded_corpus();
    let positives = graded_positives();
    let config = HardNegativeConfig::new()
        .with_embedding_dim(256)
        .with_hard_rank_cutoff(12)
        .with_negatives_per_query(2);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    for result in &round.results {
        assert!(result.negatives.len() <= 2, "budget n=2 violated");
    }
}

#[test]
fn hard_rank_cutoff_excludes_low_ranked() {
    let corpus = graded_corpus();
    let positives = graded_positives();
    // Cutoff of 3: only ranks 1..=3 are eligible. The positive c5 is rank 1, so
    // at most ranks 2 and 3 can be mined.
    let config = HardNegativeConfig::new()
        .with_embedding_dim(256)
        .with_hard_rank_cutoff(3)
        .with_negatives_per_query(10);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    let qr = round.query_result("one two three four five").unwrap();
    for negative in &qr.negatives {
        assert!(
            negative.rank <= 3,
            "mined a negative at rank {} beyond cutoff",
            negative.rank
        );
    }
    // Ranks 2 and 3 (docs c4, c3), positive c5 is rank 1.
    assert_eq!(qr.negatives.len(), 2);
    assert_eq!(qr.negatives[0].doc_id, id("c4"));
    assert_eq!(qr.negatives[1].doc_id, id("c3"));
}

#[test]
fn negatives_are_ordered_best_first() {
    let corpus = graded_corpus();
    let positives = graded_positives();
    let config = HardNegativeConfig::new()
        .with_embedding_dim(256)
        .with_hard_rank_cutoff(12)
        .with_negatives_per_query(4);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    let qr = round.query_result("one two three four five").unwrap();
    for window in qr.negatives.windows(2) {
        assert!(
            window[0].rank < window[1].rank,
            "negatives not strictly rank-ordered"
        );
        assert!(window[0].similarity >= window[1].similarity);
    }
}

#[test]
fn n_larger_than_corpus_is_bounded_not_error() {
    let corpus = vec![
        doc("p", "alpha beta gamma"),
        doc("n1", "alpha beta delta"),
        doc("n2", "alpha epsilon"),
    ];
    let positives = vec![pair("alpha beta gamma", "p")];
    let config = HardNegativeConfig::new()
        .with_embedding_dim(256)
        .with_hard_rank_cutoff(100)
        .with_negatives_per_query(100);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    let qr = round.query_result("alpha beta gamma").unwrap();
    // Three docs, one of which is the excluded positive: at most two negatives.
    assert_eq!(qr.negatives.len(), 2);
}

#[test]
fn single_document_corpus_yields_no_negatives() {
    let corpus = vec![doc("only", "alpha beta gamma")];
    let positives = vec![pair("alpha beta gamma", "only")];
    let config = HardNegativeConfig::default().with_embedding_dim(256);
    let round = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    let qr = round.query_result("alpha beta gamma").unwrap();
    assert_eq!(qr.positive_rank, 1);
    assert!(qr.negatives.is_empty());
}

#[test]
fn exclude_positive_false_can_include_positive() {
    let corpus = vec![doc("p", "alpha beta gamma"), doc("n", "alpha beta delta")];
    let positives = vec![pair("alpha beta gamma", "p")];
    // With exclusion off and the positive ranked first, it is emitted as a
    // negative; with exclusion on (the default), it is not.
    let off = HardNegativeConfig::new()
        .with_embedding_dim(256)
        .with_hard_rank_cutoff(2)
        .with_negatives_per_query(2)
        .with_exclude_positive(false);
    let round_off =
        HardNegativeMiner::mine_round(&off, &positives, &corpus, EmbeddingVersion::canonical(), 0)
            .unwrap();
    assert!(
        round_off
            .query_result("alpha beta gamma")
            .unwrap()
            .contains_negative(&id("p"))
    );

    let on = off.with_exclude_positive(true);
    let round_on =
        HardNegativeMiner::mine_round(&on, &positives, &corpus, EmbeddingVersion::canonical(), 0)
            .unwrap();
    assert!(
        !round_on
            .query_result("alpha beta gamma")
            .unwrap()
            .contains_negative(&id("p"))
    );
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn mining_is_fully_deterministic() {
    let corpus = graded_corpus();
    let positives = graded_positives();
    let config = HardNegativeConfig::default().with_embedding_dim(256);
    let version = EmbeddingVersion::new(31, 0.4);
    let a = HardNegativeMiner::mine_round(&config, &positives, &corpus, version, 0).unwrap();
    let b = HardNegativeMiner::mine_round(&config, &positives, &corpus, version, 0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn stateless_mine_round_matches_instance_mine() {
    let corpus = graded_corpus();
    let positives = graded_positives();
    let config = HardNegativeConfig::default().with_embedding_dim(256);
    let version = EmbeddingVersion::canonical();

    let stateless =
        HardNegativeMiner::mine_round(&config, &positives, &corpus, version, 0).unwrap();

    let mut miner = HardNegativeMiner::new(config, positives, corpus).unwrap();
    let stateful = miner.mine(version).unwrap();

    assert_eq!(stateless, stateful);
}

// ── Refresh / staleness ───────────────────────────────────────────────────────

#[test]
fn refresh_without_prior_round_errors() {
    let mut miner = HardNegativeMiner::new(
        HardNegativeConfig::default(),
        graded_positives(),
        graded_corpus(),
    )
    .unwrap();
    assert_eq!(
        miner.refresh(EmbeddingVersion::new(1, 0.5)).unwrap_err(),
        HardNegativeError::NoPriorRound
    );
}

#[test]
fn refresh_appends_round_and_staleness() {
    let config = HardNegativeConfig::default().with_embedding_dim(256);
    let mut miner = HardNegativeMiner::new(config, graded_positives(), graded_corpus()).unwrap();

    let round0 = miner.mine(EmbeddingVersion::canonical()).unwrap();
    assert_eq!(round0.round_index, 0);
    assert_eq!(miner.round_count(), 1);
    assert!(miner.latest_staleness().is_none());

    let round1 = miner.refresh(EmbeddingVersion::new(123, 0.6)).unwrap();
    assert_eq!(round1.round_index, 1);
    assert_eq!(miner.round_count(), 2);
    assert_eq!(miner.staleness_log().len(), 1);
    let staleness = miner.latest_staleness().unwrap();
    assert_eq!(staleness.previous_round_index, 0);
    assert_eq!(staleness.current_round_index, 1);
    assert_eq!(staleness.current_version, EmbeddingVersion::new(123, 0.6));
}

#[test]
fn refresh_same_version_is_unchanged() {
    let config = HardNegativeConfig::default().with_embedding_dim(256);
    let mut miner = HardNegativeMiner::new(config, graded_positives(), graded_corpus()).unwrap();
    let version = EmbeddingVersion::new(55, 0.5);
    miner.mine(version).unwrap();
    miner.refresh(version).unwrap();

    let staleness = miner.latest_staleness().unwrap();
    assert_eq!(staleness.mean_jaccard, 1.0);
    assert!(staleness.is_unchanged());
    assert_eq!(staleness.staleness(), 0.0);
    for overlap in &staleness.per_query {
        assert_eq!(overlap.jaccard, 1.0);
        assert_eq!(overlap.added, 0);
        assert_eq!(overlap.removed, 0);
    }
    assert_eq!(staleness.mean_positive_rank_delta, 0.0);
}

#[test]
fn refresh_under_scrambling_version_changes_the_set() {
    let config = HardNegativeConfig::new()
        .with_embedding_dim(64)
        .with_hard_rank_cutoff(12)
        .with_negatives_per_query(4);
    let mut miner = HardNegativeMiner::new(config, graded_positives(), graded_corpus()).unwrap();
    miner.mine(EmbeddingVersion::canonical()).unwrap();
    miner.refresh(EmbeddingVersion::new(9_999, 0.9)).unwrap();

    let staleness = miner.latest_staleness().unwrap();
    assert!(
        staleness.mean_jaccard < 1.0,
        "a scrambling refresh must change the mined set (jaccard {})",
        staleness.mean_jaccard
    );
    assert!(staleness.staleness() > 0.0);
    assert!(!staleness.is_unchanged());
}

#[test]
fn staleness_low_drift_high_overlap_vs_scramble_low_overlap() {
    let config = HardNegativeConfig::new()
        .with_embedding_dim(64)
        .with_hard_rank_cutoff(12)
        .with_negatives_per_query(4);

    // Barely-perturbing version: tiny drift, so the well-separated base ranking
    // is preserved and the mined set is unchanged (high overlap).
    let mut miner_small =
        HardNegativeMiner::new(config.clone(), graded_positives(), graded_corpus()).unwrap();
    miner_small.mine(EmbeddingVersion::canonical()).unwrap();
    miner_small
        .refresh(EmbeddingVersion::new(4242, 0.02))
        .unwrap();
    let j_small = miner_small.latest_staleness().unwrap().mean_jaccard;

    // Scrambling version: heavy drift with an unrelated seed, so the ranking is
    // reordered and the mined set turns over (low overlap).
    let mut miner_big =
        HardNegativeMiner::new(config, graded_positives(), graded_corpus()).unwrap();
    miner_big.mine(EmbeddingVersion::canonical()).unwrap();
    miner_big
        .refresh(EmbeddingVersion::new(9_999, 0.95))
        .unwrap();
    let j_big = miner_big.latest_staleness().unwrap().mean_jaccard;

    assert!(
        j_small > j_big,
        "small-drift overlap ({j_small}) must exceed scramble overlap ({j_big})"
    );
    assert!(
        j_small >= 0.75,
        "small drift should barely change the set (jaccard {j_small})"
    );
    assert!(
        j_big < 0.5,
        "scramble should mostly turn the set over (jaccard {j_big})"
    );
}

#[test]
fn round_history_accumulates() {
    let config = HardNegativeConfig::default().with_embedding_dim(256);
    let mut miner = HardNegativeMiner::new(config, graded_positives(), graded_corpus()).unwrap();
    miner.mine(EmbeddingVersion::canonical()).unwrap();
    miner.refresh(EmbeddingVersion::new(1, 0.3)).unwrap();
    miner.refresh(EmbeddingVersion::new(2, 0.6)).unwrap();
    miner.refresh(EmbeddingVersion::new(3, 0.9)).unwrap();

    assert_eq!(miner.round_count(), 4);
    assert_eq!(miner.history().len(), 4);
    assert_eq!(miner.staleness_log().len(), 3);
    for (i, round) in miner.history().iter().enumerate() {
        assert_eq!(round.round_index, i);
    }
    assert_eq!(miner.latest_round().unwrap().round_index, 3);
}

#[test]
fn staleness_per_query_counts_are_consistent() {
    let config = HardNegativeConfig::new()
        .with_embedding_dim(64)
        .with_hard_rank_cutoff(12)
        .with_negatives_per_query(4);
    let mut miner = HardNegativeMiner::new(config, graded_positives(), graded_corpus()).unwrap();
    let round0 = miner.mine(EmbeddingVersion::canonical()).unwrap();
    let round1 = miner.refresh(EmbeddingVersion::new(9_999, 0.9)).unwrap();

    let staleness = miner.latest_staleness().unwrap();
    for overlap in &staleness.per_query {
        let prev = round0.query_result(&overlap.query).unwrap().negatives.len();
        let cur = round1.query_result(&overlap.query).unwrap().negatives.len();
        // retained + added == current-set size; retained + removed == prev size.
        assert_eq!(overlap.retained + overlap.added, cur);
        assert_eq!(overlap.retained + overlap.removed, prev);
    }
}

#[test]
fn staleness_associated_fn_matches_refresh_log() {
    let config = HardNegativeConfig::new()
        .with_embedding_dim(64)
        .with_hard_rank_cutoff(12)
        .with_negatives_per_query(4);
    let mut miner = HardNegativeMiner::new(config, graded_positives(), graded_corpus()).unwrap();
    let round0 = miner.mine(EmbeddingVersion::canonical()).unwrap();
    let round1 = miner.refresh(EmbeddingVersion::new(7, 0.7)).unwrap();

    let direct = HardNegativeMiner::staleness(&round0, &round1);
    let logged = miner.latest_staleness().unwrap();
    assert_eq!(direct.mean_jaccard, logged.mean_jaccard);
    assert_eq!(direct.previous_round_index, logged.previous_round_index);
    assert_eq!(direct.current_round_index, logged.current_round_index);
}

#[test]
fn mean_positive_rank_delta_matches_round_difference() {
    let config = HardNegativeConfig::new()
        .with_embedding_dim(64)
        .with_hard_rank_cutoff(12);
    let mut miner = HardNegativeMiner::new(config, graded_positives(), graded_corpus()).unwrap();
    let round0 = miner.mine(EmbeddingVersion::canonical()).unwrap();
    let round1 = miner.refresh(EmbeddingVersion::new(9_999, 0.95)).unwrap();
    let staleness = miner.latest_staleness().unwrap();
    assert_eq!(
        staleness.mean_positive_rank_delta,
        round1.mean_positive_rank - round0.mean_positive_rank
    );
}

// ── Construction / validation errors ──────────────────────────────────────────

#[test]
fn new_empty_corpus_errors() {
    let err = HardNegativeMiner::new(
        HardNegativeConfig::default(),
        vec![pair("q", "x")],
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(err, HardNegativeError::EmptyCorpus);
}

#[test]
fn new_empty_positives_errors() {
    let err = HardNegativeMiner::new(
        HardNegativeConfig::default(),
        Vec::new(),
        vec![doc("x", "alpha")],
    )
    .unwrap_err();
    assert_eq!(err, HardNegativeError::EmptyPositives);
}

#[test]
fn new_duplicate_document_id_errors() {
    let err = HardNegativeMiner::new(
        HardNegativeConfig::default(),
        vec![pair("q", "dup")],
        vec![doc("dup", "alpha"), doc("dup", "beta")],
    )
    .unwrap_err();
    assert_eq!(err, HardNegativeError::DuplicateDocumentId(id("dup")));
}

#[test]
fn new_unknown_positive_errors() {
    let err = HardNegativeMiner::new(
        HardNegativeConfig::default(),
        vec![pair("q", "missing")],
        vec![doc("present", "alpha")],
    )
    .unwrap_err();
    assert_eq!(
        err,
        HardNegativeError::UnknownPositiveDocument(id("missing"))
    );
}

#[test]
fn new_empty_query_errors() {
    let err = HardNegativeMiner::new(
        HardNegativeConfig::default(),
        vec![pair("   ", "x")],
        vec![doc("x", "alpha")],
    )
    .unwrap_err();
    assert_eq!(err, HardNegativeError::EmptyQuery);
}

#[test]
fn mine_round_rejects_zero_config() {
    let corpus = vec![doc("x", "alpha")];
    let positives = vec![pair("q", "x")];
    let bad = HardNegativeConfig::new().with_negatives_per_query(0);
    assert_eq!(
        HardNegativeMiner::mine_round(&bad, &positives, &corpus, EmbeddingVersion::canonical(), 0)
            .unwrap_err(),
        HardNegativeError::ZeroNegativesPerQuery
    );
}

#[test]
fn error_display_messages_are_specific() {
    assert!(
        HardNegativeError::EmptyCorpus
            .to_string()
            .contains("corpus is empty")
    );
    assert!(
        HardNegativeError::NoPriorRound
            .to_string()
            .contains("no prior mining round")
    );
    let dup = HardNegativeError::DuplicateDocumentId(id("abc"));
    assert!(dup.to_string().contains("abc"));
    let unknown = HardNegativeError::UnknownPositiveDocument(id("zzz"));
    assert!(unknown.to_string().contains("zzz"));
}

#[test]
fn mining_round_type_is_cloneable_and_comparable() {
    let corpus = graded_corpus();
    let positives = graded_positives();
    let config = HardNegativeConfig::default().with_embedding_dim(256);
    let round: MiningRound = HardNegativeMiner::mine_round(
        &config,
        &positives,
        &corpus,
        EmbeddingVersion::canonical(),
        0,
    )
    .unwrap();
    let cloned = round.clone();
    assert_eq!(round, cloned);
}

#[test]
fn accessors_expose_bound_inputs() {
    let corpus = graded_corpus();
    let positives = graded_positives();
    let config = HardNegativeConfig::default().with_embedding_dim(128);
    let miner = HardNegativeMiner::new(config.clone(), positives.clone(), corpus.clone()).unwrap();
    assert_eq!(miner.config(), &config);
    assert_eq!(miner.corpus(), corpus.as_slice());
    assert_eq!(miner.positives(), positives.as_slice());
    assert_eq!(miner.round_count(), 0);
    assert!(miner.latest_round().is_none());
}
