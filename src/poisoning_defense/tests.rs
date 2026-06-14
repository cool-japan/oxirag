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

use super::detector::PoisoningDetector;
use super::types::{PoisonAssessment, PoisonConfig, PoisonError};
use crate::types::{Document, SearchResult};

// ── helpers ─────────────────────────────────────────────────────────────────────

/// Build a [`SearchResult`] from raw content with a fixed score and rank.
fn result(content: &str, score: f32, rank: usize) -> SearchResult {
    SearchResult::new(Document::new(content), score, rank)
}

/// A small, varied corpus of natural-looking passages used as background.
fn normal_corpus() -> Vec<SearchResult> {
    vec![
        result(
            "The Eiffel Tower is a wrought iron lattice tower located in Paris France.",
            0.91,
            0,
        ),
        result(
            "Construction of the Eiffel Tower finished in eighteen eighty nine for the world fair.",
            0.82,
            1,
        ),
        result(
            "Gustave Eiffel designed the famous Paris landmark that still draws many visitors.",
            0.74,
            2,
        ),
    ]
}

// ── PoisonConfig: defaults ────────────────────────────────────────────────────────

#[test]
fn config_default_stuffing_threshold() {
    assert_eq!(PoisonConfig::default().stuffing_threshold, 0.35);
}

#[test]
fn config_default_diversity_threshold() {
    assert_eq!(PoisonConfig::default().diversity_threshold, 0.4);
}

#[test]
fn config_default_risk_threshold() {
    assert_eq!(PoisonConfig::default().risk_threshold, 0.5);
}

#[test]
fn config_new_matches_default() {
    assert_eq!(PoisonConfig::new(), PoisonConfig::default());
}

// ── PoisonConfig: builders ────────────────────────────────────────────────────────

#[test]
fn config_with_stuffing_threshold() {
    let c = PoisonConfig::new().with_stuffing_threshold(0.7);
    assert_eq!(c.stuffing_threshold, 0.7);
}

#[test]
fn config_with_diversity_threshold() {
    let c = PoisonConfig::new().with_diversity_threshold(0.25);
    assert_eq!(c.diversity_threshold, 0.25);
}

#[test]
fn config_with_risk_threshold() {
    let c = PoisonConfig::new().with_risk_threshold(0.8);
    assert_eq!(c.risk_threshold, 0.8);
}

#[test]
fn config_builders_chain() {
    let c = PoisonConfig::new()
        .with_stuffing_threshold(0.6)
        .with_diversity_threshold(0.3)
        .with_risk_threshold(0.9);
    assert_eq!(c.stuffing_threshold, 0.6);
    assert_eq!(c.diversity_threshold, 0.3);
    assert_eq!(c.risk_threshold, 0.9);
}

#[test]
fn config_builders_do_not_disturb_other_fields() {
    let c = PoisonConfig::new().with_risk_threshold(0.99);
    assert_eq!(c.stuffing_threshold, 0.35);
    assert_eq!(c.diversity_threshold, 0.4);
}

#[test]
fn config_clone_eq() {
    let c = PoisonConfig::new().with_stuffing_threshold(0.5);
    assert_eq!(c.clone(), c);
}

// ── detector construction ─────────────────────────────────────────────────────────

#[test]
fn detector_new_keeps_config() {
    let cfg = PoisonConfig::new().with_risk_threshold(0.42);
    let d = PoisoningDetector::new(cfg.clone());
    assert_eq!(*d.config(), cfg);
}

#[test]
fn detector_default_uses_default_config() {
    let d = PoisoningDetector::default();
    assert_eq!(*d.config(), PoisonConfig::default());
}

// ── stuffing_score ────────────────────────────────────────────────────────────────

#[test]
fn stuffing_high_for_keyword_stuffed_passage() {
    let d = PoisoningDetector::default();
    let query = "best laptop";
    // Query terms repeated many times — classic stuffing.
    let text = "best laptop best laptop best laptop best laptop best laptop best laptop";
    let s = d.stuffing_score(query, text);
    assert!(s > 0.9, "stuffing score should be near 1.0, got {s}");
}

#[test]
fn stuffing_low_for_natural_passage() {
    let d = PoisoningDetector::default();
    let query = "eiffel tower";
    let text =
        "The Eiffel Tower is a wrought iron lattice tower located in the city of Paris France.";
    let s = d.stuffing_score(query, text);
    assert!(s < 0.35, "natural prose should have low stuffing, got {s}");
}

#[test]
fn stuffing_zero_for_empty_query() {
    let d = PoisoningDetector::default();
    assert_eq!(d.stuffing_score("", "some text here today"), 0.0);
}

#[test]
fn stuffing_zero_for_empty_text() {
    let d = PoisoningDetector::default();
    assert_eq!(d.stuffing_score("query terms", ""), 0.0);
}

#[test]
fn stuffing_zero_when_no_overlap() {
    let d = PoisoningDetector::default();
    let s = d.stuffing_score("apple banana", "completely different words appear here");
    assert_eq!(s, 0.0);
}

#[test]
fn stuffing_in_unit_range() {
    let d = PoisoningDetector::default();
    let s = d.stuffing_score("alpha beta", "alpha beta gamma delta alpha");
    assert!(s >= 0.0 && s <= 1.0);
}

#[test]
fn stuffing_counts_repetition_not_just_presence() {
    let d = PoisoningDetector::default();
    let once = d.stuffing_score("term", "term word other thing here now then");
    let many = d.stuffing_score("term", "term term term term term term term term");
    assert!(
        many > once,
        "repetition must raise density: {many} vs {once}"
    );
}

#[test]
fn stuffing_full_density_when_all_tokens_are_query_terms() {
    let d = PoisoningDetector::default();
    let s = d.stuffing_score("foo bar", "foo bar foo bar");
    assert_eq!(s, 1.0);
}

// ── diversity_score ───────────────────────────────────────────────────────────────

#[test]
fn diversity_low_for_repeated_single_word() {
    let d = PoisoningDetector::default();
    let text = "spam spam spam spam spam spam spam spam spam spam";
    let s = d.diversity_score(text);
    assert!(
        s < 0.4,
        "single repeated word should be low diversity, got {s}"
    );
}

#[test]
fn diversity_high_for_varied_text() {
    let d = PoisoningDetector::default();
    let text = "every single token within this particular sentence happens to be unique";
    let s = d.diversity_score(text);
    assert!(s > 0.9, "all-distinct text should be near 1.0, got {s}");
}

#[test]
fn diversity_one_for_all_distinct() {
    let d = PoisoningDetector::default();
    assert_eq!(d.diversity_score("alpha beta gamma delta"), 1.0);
}

#[test]
fn diversity_empty_text_is_one() {
    let d = PoisoningDetector::default();
    assert_eq!(d.diversity_score(""), 1.0);
}

#[test]
fn diversity_half_when_each_token_appears_twice() {
    let d = PoisoningDetector::default();
    // four tokens, two distinct -> ratio 0.5
    let s = d.diversity_score("one two one two");
    assert_eq!(s, 0.5);
}

#[test]
fn diversity_in_unit_range() {
    let d = PoisoningDetector::default();
    let s = d.diversity_score("aa bb aa cc bb aa");
    assert!(s >= 0.0 && s <= 1.0);
}

#[test]
fn diversity_decreases_with_more_repetition() {
    let d = PoisoningDetector::default();
    let some = d.diversity_score("red green blue red");
    let lots = d.diversity_score("red red red red red green");
    assert!(
        lots < some,
        "more repetition lowers diversity: {lots} vs {some}"
    );
}

// ── anomaly_score ─────────────────────────────────────────────────────────────────

#[test]
fn anomaly_high_for_outlier_contradicting_consensus() {
    let d = PoisoningDetector::default();
    let corpus = vec![
        "The Eiffel Tower is located in Paris France a famous landmark.",
        "Paris France hosts the Eiffel Tower a wrought iron lattice structure.",
        "Visitors to Paris France admire the tall iron Eiffel Tower landmark.",
    ];
    // Outlier shares essentially no vocabulary with the consensus subject.
    let outlier = "Quantum chromodynamics describes interactions among subatomic colored quarks.";
    let s = d.anomaly_score(outlier, &corpus);
    assert!(
        s > 0.8,
        "off-topic outlier should be highly anomalous, got {s}"
    );
}

#[test]
fn anomaly_low_for_consensus_member() {
    let d = PoisoningDetector::default();
    let corpus = vec![
        "The Eiffel Tower is located in Paris France a famous iron landmark.",
        "Paris France hosts the Eiffel Tower a wrought iron lattice landmark.",
        "The Eiffel Tower in Paris France is a famous iron lattice landmark.",
    ];
    let member = "The Eiffel Tower is a famous iron lattice landmark in Paris France.";
    let s = d.anomaly_score(member, &corpus);
    assert!(
        s < 0.6,
        "on-topic passage should be less anomalous, got {s}"
    );
}

#[test]
fn anomaly_zero_with_no_other_passages() {
    let d = PoisoningDetector::default();
    assert_eq!(d.anomaly_score("anything here", &[]), 0.0);
}

#[test]
fn anomaly_skips_identical_self_entry() {
    let d = PoisoningDetector::default();
    // Only entry equals the target -> nothing to compare against -> 0.0
    let text = "self referential passage content";
    let s = d.anomaly_score(text, &[text]);
    assert_eq!(s, 0.0);
}

#[test]
fn anomaly_outlier_higher_than_member() {
    let d = PoisoningDetector::default();
    let corpus = vec![
        "machine learning models train on large labelled datasets every day",
        "training large machine learning models needs labelled datasets and compute",
        "labelled datasets help machine learning models learn useful representations",
    ];
    let member = "machine learning models need labelled datasets to train well";
    let outlier = "the ancient roman aqueducts carried fresh water across long valleys";
    let m = d.anomaly_score(member, &corpus);
    let o = d.anomaly_score(outlier, &corpus);
    assert!(o > m, "outlier must exceed member anomaly: {o} vs {m}");
}

#[test]
fn anomaly_in_unit_range() {
    let d = PoisoningDetector::default();
    let corpus = vec!["alpha beta gamma", "beta gamma delta"];
    let s = d.anomaly_score("gamma delta epsilon", &corpus);
    assert!(s >= 0.0 && s <= 1.0);
}

#[test]
fn anomaly_zero_when_identical_to_all_others() {
    let d = PoisoningDetector::default();
    let corpus = vec!["red green blue circle", "red green blue circle"];
    // identical content -> jaccard 1.0 with each non-self neighbour -> anomaly 0.
    let s = d.anomaly_score("red green blue circle", &corpus);
    assert_eq!(s, 0.0);
}

// ── assess: keyword-stuffed passage ────────────────────────────────────────────────

#[test]
fn assess_flags_keyword_stuffed_passage() {
    let d = PoisoningDetector::default();
    let query = "wireless headphones review";
    let stuffed = "wireless headphones review wireless headphones review wireless headphones review wireless headphones review";
    let corpus = vec![
        "These wireless headphones offer crisp sound and a comfortable padded fit.",
        "Our review found the battery on these wireless headphones lasts all day.",
    ];
    let a = d.assess(query, stuffed, &corpus);
    assert!(a.stuffing_score >= d.config().stuffing_threshold);
    assert!(a.is_poisoned, "stuffed passage must be flagged: {a:?}");
}

#[test]
fn assess_stuffed_passage_has_high_stuffing_score() {
    let d = PoisoningDetector::default();
    let query = "cheap flights";
    let stuffed = "cheap flights cheap flights cheap flights cheap flights cheap flights";
    let a = d.assess(query, stuffed, &[]);
    assert!(a.stuffing_score > 0.9, "got {}", a.stuffing_score);
}

// ── assess: low-diversity passage ──────────────────────────────────────────────────

#[test]
fn assess_flags_low_diversity_passage() {
    let d = PoisoningDetector::default();
    let query = "weather forecast";
    let low_div = "buy buy buy buy buy buy buy buy buy buy buy buy buy buy buy buy";
    let corpus = vec![
        "Tomorrow brings light rain across the northern hills with a cool breeze.",
        "The weekend forecast shows clear skies and mild afternoon temperatures.",
    ];
    let a = d.assess(query, low_div, &corpus);
    assert!(a.diversity_score <= d.config().diversity_threshold);
    assert!(
        a.is_poisoned,
        "low-diversity passage must be flagged: {a:?}"
    );
}

#[test]
fn assess_low_diversity_passage_has_low_diversity_score() {
    let d = PoisoningDetector::default();
    let a = d.assess("topic", "loop loop loop loop loop loop loop loop", &[]);
    assert!(a.diversity_score < 0.4, "got {}", a.diversity_score);
}

// ── assess: normal relevant passage ────────────────────────────────────────────────

#[test]
fn assess_normal_passage_not_poisoned() {
    let d = PoisoningDetector::default();
    let query = "eiffel tower location";
    let corpus = normal_corpus();
    let corpus_texts: Vec<&str> = corpus.iter().map(|r| r.document.content.as_str()).collect();
    let normal = "The Eiffel Tower stands in the heart of Paris France beside the river Seine.";
    let a = d.assess(query, normal, &corpus_texts);
    assert!(!a.is_poisoned, "normal passage flagged poisoned: {a:?}");
}

#[test]
fn assess_normal_passage_low_risk() {
    let d = PoisoningDetector::default();
    let query = "eiffel tower";
    let corpus = normal_corpus();
    let corpus_texts: Vec<&str> = corpus.iter().map(|r| r.document.content.as_str()).collect();
    let normal = "The Eiffel Tower was completed in eighteen eighty nine in Paris France.";
    let a = d.assess(query, normal, &corpus_texts);
    assert!(
        a.risk < d.config().risk_threshold,
        "risk too high: {}",
        a.risk
    );
}

// ── assess: outlier ────────────────────────────────────────────────────────────────

#[test]
fn assess_outlier_has_high_anomaly() {
    let d = PoisoningDetector::default();
    let query = "eiffel tower";
    let corpus = normal_corpus();
    let corpus_texts: Vec<&str> = corpus.iter().map(|r| r.document.content.as_str()).collect();
    let outlier =
        "Photosynthesis converts sunlight carbon dioxide and water into glucose inside leaves.";
    let a = d.assess(query, outlier, &corpus_texts);
    assert!(
        a.anomaly_score > 0.8,
        "outlier anomaly too low: {}",
        a.anomaly_score
    );
}

// ── assess: risk blend / structure ─────────────────────────────────────────────────

#[test]
fn assess_risk_in_unit_range() {
    let d = PoisoningDetector::default();
    let a = d.assess(
        "alpha",
        "alpha beta gamma delta",
        &["beta gamma delta epsilon"],
    );
    assert!(a.risk >= 0.0 && a.risk <= 1.0);
}

#[test]
fn assess_all_scores_in_unit_range() {
    let d = PoisoningDetector::default();
    let a = d.assess(
        "machine learning",
        "machine learning systems learn machine learning patterns",
        &["data pipelines feed machine learning systems daily"],
    );
    assert!(a.stuffing_score >= 0.0 && a.stuffing_score <= 1.0);
    assert!(a.diversity_score >= 0.0 && a.diversity_score <= 1.0);
    assert!(a.anomaly_score >= 0.0 && a.anomaly_score <= 1.0);
}

#[test]
fn assess_index_is_zero() {
    let d = PoisoningDetector::default();
    let a = d.assess(
        "query",
        "normal varied passage about many distinct topics here",
        &[],
    );
    assert_eq!(a.index, 0);
}

#[test]
fn assess_high_risk_blend_flags_without_hard_trip() {
    // A passage that is moderately stuffed AND moderately anomalous can clear the
    // risk threshold even if no single hard rule trips. Lower the risk threshold
    // and raise the hard thresholds so only the blend can fire.
    let cfg = PoisonConfig::new()
        .with_stuffing_threshold(0.99)
        .with_diversity_threshold(0.01)
        .with_risk_threshold(0.4);
    let d = PoisoningDetector::new(cfg);
    let query = "alpha beta";
    let text = "alpha beta zzz qqq alpha beta www";
    let corpus = vec!["totally unrelated vocabulary nowhere near the passage tokens"];
    let a = d.assess(query, text, &corpus);
    assert!(a.stuffing_score < 0.99);
    assert!(a.diversity_score > 0.01);
    assert!(a.risk >= 0.4, "blend should reach threshold: {}", a.risk);
    assert!(a.is_poisoned);
}

#[test]
fn assess_stuffing_hard_trip_overrides_low_risk_threshold_config() {
    // Even with risk_threshold raised to 1.0, a stuffing hard-trip flags.
    let cfg = PoisonConfig::new().with_risk_threshold(1.0);
    let d = PoisoningDetector::new(cfg);
    let a = d.assess("xx yy", "xx yy xx yy xx yy xx yy xx yy xx yy", &[]);
    assert!(a.risk < 1.0);
    assert!(a.is_poisoned, "stuffing hard-trip should flag: {a:?}");
}

// ── scan ────────────────────────────────────────────────────────────────────────

#[test]
fn scan_returns_one_assessment_per_result() {
    let d = PoisoningDetector::default();
    let results = normal_corpus();
    let out = d.scan("eiffel tower", &results);
    assert_eq!(out.len(), results.len());
}

#[test]
fn scan_assessments_have_sequential_indices() {
    let d = PoisoningDetector::default();
    let results = normal_corpus();
    let out = d.scan("eiffel tower", &results);
    for (i, a) in out.iter().enumerate() {
        assert_eq!(a.index, i);
    }
}

#[test]
fn scan_empty_results_yields_empty() {
    let d = PoisoningDetector::default();
    let out = d.scan("query", &[]);
    assert!(out.is_empty());
}

#[test]
fn scan_detects_injected_poison_among_normal() {
    let d = PoisoningDetector::default();
    let mut results = normal_corpus();
    // Inject a keyword-stuffed adversarial passage at the front.
    results.insert(
        0,
        result(
            "eiffel tower eiffel tower eiffel tower eiffel tower eiffel tower eiffel tower",
            0.99,
            0,
        ),
    );
    let out = d.scan("eiffel tower", &results);
    assert!(
        out[0].is_poisoned,
        "injected passage should be flagged: {:?}",
        out[0]
    );
}

#[test]
fn scan_leaves_genuine_passages_unflagged() {
    let d = PoisoningDetector::default();
    let results = normal_corpus();
    let out = d.scan("eiffel tower paris", &results);
    assert!(
        out.iter().all(|a| !a.is_poisoned),
        "genuine corpus should be clean: {out:?}"
    );
}

// ── filter ────────────────────────────────────────────────────────────────────────

#[test]
fn filter_removes_poisoned_passage() {
    let d = PoisoningDetector::default();
    let mut results = normal_corpus();
    results.insert(
        0,
        result(
            "eiffel tower eiffel tower eiffel tower eiffel tower eiffel tower eiffel tower",
            0.99,
            0,
        ),
    );
    let original = results.len();
    let kept = d.filter("eiffel tower", &results).unwrap();
    assert!(kept.len() < original, "poison should be dropped");
    assert!(
        kept.iter()
            .all(|r| !r.document.content.starts_with("eiffel tower eiffel tower")),
        "stuffed passage must be gone"
    );
}

#[test]
fn filter_reranks_survivors_from_zero() {
    let d = PoisoningDetector::default();
    let mut results = normal_corpus();
    results.insert(
        0,
        result("spam spam spam spam spam spam spam spam spam spam", 0.99, 0),
    );
    let kept = d.filter("eiffel tower", &results).unwrap();
    for (i, r) in kept.iter().enumerate() {
        assert_eq!(r.rank, i, "rank must be reassigned 0..");
    }
}

#[test]
fn filter_keeps_all_when_nothing_poisoned() {
    let d = PoisoningDetector::default();
    let results = normal_corpus();
    let kept = d.filter("eiffel tower paris", &results).unwrap();
    assert_eq!(kept.len(), results.len());
}

#[test]
fn filter_preserves_relative_order_of_survivors() {
    let d = PoisoningDetector::default();
    let mut results = normal_corpus();
    // Insert poison in the middle.
    results.insert(
        1,
        result("ad ad ad ad ad ad ad ad ad ad ad ad ad ad ad ad", 0.5, 1),
    );
    let kept = d.filter("eiffel tower", &results).unwrap();
    // Survivors should be the three original passages, in order.
    assert_eq!(kept.len(), 3);
    assert!(
        kept[0]
            .document
            .content
            .starts_with("The Eiffel Tower is a wrought")
    );
    assert!(
        kept[1]
            .document
            .content
            .starts_with("Construction of the Eiffel")
    );
    assert!(
        kept[2]
            .document
            .content
            .starts_with("Gustave Eiffel designed")
    );
}

#[test]
fn filter_empty_results_is_ok_empty() {
    let d = PoisoningDetector::default();
    let kept = d.filter("query", &[]).unwrap();
    assert!(kept.is_empty());
}

#[test]
fn filter_empty_results_does_not_error_on_empty_query() {
    let d = PoisoningDetector::default();
    // Empty results short-circuits before query validation.
    let kept = d.filter("", &[]).unwrap();
    assert!(kept.is_empty());
}

// ── EmptyQuery error ──────────────────────────────────────────────────────────────

#[test]
fn filter_empty_query_errors() {
    let d = PoisoningDetector::default();
    let results = normal_corpus();
    assert_eq!(d.filter("", &results).unwrap_err(), PoisonError::EmptyQuery);
}

#[test]
fn filter_whitespace_query_errors() {
    let d = PoisoningDetector::default();
    let results = normal_corpus();
    assert_eq!(
        d.filter("   \t  ", &results).unwrap_err(),
        PoisonError::EmptyQuery
    );
}

#[test]
fn empty_query_error_message() {
    assert_eq!(
        PoisonError::EmptyQuery.to_string(),
        "query must not be empty"
    );
}

// ── determinism ────────────────────────────────────────────────────────────────────

#[test]
fn assess_is_deterministic() {
    let d = PoisoningDetector::default();
    let query = "eiffel tower";
    let text = "The Eiffel Tower is a wrought iron lattice tower in Paris France.";
    let corpus = vec!["Paris France hosts many landmarks including the iron tower."];
    let a = d.assess(query, text, &corpus);
    let b = d.assess(query, text, &corpus);
    assert_eq!(a, b);
}

#[test]
fn scan_is_deterministic() {
    let d = PoisoningDetector::default();
    let results = normal_corpus();
    let a = d.scan("eiffel tower", &results);
    let b = d.scan("eiffel tower", &results);
    assert_eq!(a, b);
}

#[test]
fn filter_is_deterministic() {
    let d = PoisoningDetector::default();
    let mut results = normal_corpus();
    results.insert(0, result("xx xx xx xx xx xx xx xx xx xx xx xx", 0.9, 0));
    let a = d.filter("eiffel tower", &results).unwrap();
    let b = d.filter("eiffel tower", &results).unwrap();
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.document.content, rb.document.content);
        assert_eq!(ra.rank, rb.rank);
    }
}

#[test]
fn scores_are_deterministic_across_calls() {
    let d = PoisoningDetector::default();
    let s1 = d.stuffing_score("a b", "a b c a b c");
    let s2 = d.stuffing_score("a b", "a b c a b c");
    let v1 = d.diversity_score("a b c a b c");
    let v2 = d.diversity_score("a b c a b c");
    let n1 = d.anomaly_score("a b c", &["a b c", "x y z"]);
    let n2 = d.anomaly_score("a b c", &["a b c", "x y z"]);
    assert_eq!(s1, s2);
    assert_eq!(v1, v2);
    assert_eq!(n1, n2);
}

// ── assessment value semantics ──────────────────────────────────────────────────────

#[test]
fn assessment_clone_eq() {
    let a = PoisonAssessment {
        index: 3,
        stuffing_score: 0.5,
        diversity_score: 0.6,
        anomaly_score: 0.7,
        risk: 0.55,
        is_poisoned: true,
    };
    assert_eq!(a.clone(), a);
}

#[test]
fn config_is_accessible_from_detector() {
    let cfg = PoisonConfig::new().with_stuffing_threshold(0.5);
    let d = PoisoningDetector::new(cfg);
    assert_eq!(d.config().stuffing_threshold, 0.5);
}
