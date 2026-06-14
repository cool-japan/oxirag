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

//! Tests for the `query_difficulty` module.

use super::predictor::DifficultyPredictor;
use super::types::{DifficultyBand, DifficultyConfig, DifficultyError, NEGATION_WORDS, STOP_WORDS};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn default_predictor() -> DifficultyPredictor {
    DifficultyPredictor::new(DifficultyConfig::default())
}

// ── DifficultyConfig ──────────────────────────────────────────────────────────

#[test]
fn config_defaults_negation_weight() {
    assert_eq!(DifficultyConfig::default().negation_weight, 0.2);
}

#[test]
fn config_defaults_multi_hop_weight() {
    assert_eq!(DifficultyConfig::default().multi_hop_weight, 0.3);
}

#[test]
fn config_defaults_ambiguity_weight() {
    assert_eq!(DifficultyConfig::default().ambiguity_weight, 0.2);
}

#[test]
fn config_defaults_length_weight() {
    assert_eq!(DifficultyConfig::default().length_weight, 0.15);
}

#[test]
fn config_defaults_rare_word_weight() {
    assert_eq!(DifficultyConfig::default().rare_word_weight, 0.15);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(DifficultyConfig::new(), DifficultyConfig::default());
}

#[test]
fn config_builder_negation_weight() {
    let cfg = DifficultyConfig::new().with_negation_weight(0.5);
    assert_eq!(cfg.negation_weight, 0.5);
    // Other fields untouched.
    assert_eq!(cfg.multi_hop_weight, 0.3);
}

#[test]
fn config_builder_multi_hop_weight() {
    let cfg = DifficultyConfig::new().with_multi_hop_weight(0.6);
    assert_eq!(cfg.multi_hop_weight, 0.6);
}

#[test]
fn config_builder_ambiguity_weight() {
    let cfg = DifficultyConfig::new().with_ambiguity_weight(0.1);
    assert_eq!(cfg.ambiguity_weight, 0.1);
}

#[test]
fn config_builder_length_weight() {
    let cfg = DifficultyConfig::new().with_length_weight(0.0);
    assert_eq!(cfg.length_weight, 0.0);
}

#[test]
fn config_builder_rare_word_weight() {
    let cfg = DifficultyConfig::new().with_rare_word_weight(0.0);
    assert_eq!(cfg.rare_word_weight, 0.0);
}

#[test]
fn config_weights_default_sum_to_one() {
    let cfg = DifficultyConfig::default();
    let total = cfg.negation_weight
        + cfg.multi_hop_weight
        + cfg.ambiguity_weight
        + cfg.length_weight
        + cfg.rare_word_weight;
    assert!((total - 1.0_f32).abs() < 1e-5, "weights sum to {total}");
}

// ── DifficultyBand ────────────────────────────────────────────────────────────

#[test]
fn band_easy_as_str() {
    assert_eq!(DifficultyBand::Easy.as_str(), "easy");
}

#[test]
fn band_medium_as_str() {
    assert_eq!(DifficultyBand::Medium.as_str(), "medium");
}

#[test]
fn band_hard_as_str() {
    assert_eq!(DifficultyBand::Hard.as_str(), "hard");
}

#[test]
fn band_ambiguous_as_str() {
    assert_eq!(DifficultyBand::Ambiguous.as_str(), "ambiguous");
}

#[test]
fn band_display_matches_as_str() {
    for band in [
        DifficultyBand::Easy,
        DifficultyBand::Medium,
        DifficultyBand::Hard,
        DifficultyBand::Ambiguous,
    ] {
        assert_eq!(format!("{band}"), band.as_str());
    }
}

#[test]
fn band_equality() {
    assert_eq!(DifficultyBand::Easy, DifficultyBand::Easy);
    assert_eq!(DifficultyBand::Ambiguous, DifficultyBand::Ambiguous);
}

#[test]
fn band_inequality() {
    assert_ne!(DifficultyBand::Easy, DifficultyBand::Hard);
    assert_ne!(DifficultyBand::Medium, DifficultyBand::Ambiguous);
}

#[test]
fn band_clone() {
    let b = DifficultyBand::Hard;
    assert_eq!(b.clone(), DifficultyBand::Hard);
}

// ── DifficultyError ───────────────────────────────────────────────────────────

#[test]
fn error_empty_query_display() {
    let err = DifficultyError::EmptyQuery;
    assert_eq!(err.to_string(), "query is empty");
}

#[test]
fn error_prediction_failed_display() {
    let err = DifficultyError::PredictionFailed("NaN weight".to_string());
    assert!(err.to_string().contains("NaN weight"));
}

#[test]
fn error_equality() {
    assert_eq!(DifficultyError::EmptyQuery, DifficultyError::EmptyQuery);
}

// ── Empty / blank query ───────────────────────────────────────────────────────

#[test]
fn empty_query_returns_error() {
    let p = default_predictor();
    assert_eq!(p.predict(""), Err(DifficultyError::EmptyQuery));
}

#[test]
fn whitespace_only_returns_error() {
    let p = default_predictor();
    assert_eq!(p.predict("   "), Err(DifficultyError::EmptyQuery));
}

#[test]
fn tab_only_returns_error() {
    let p = default_predictor();
    assert_eq!(p.predict("\t\n"), Err(DifficultyError::EmptyQuery));
}

// ── Band assignment — Easy ────────────────────────────────────────────────────

#[test]
fn easy_query_band() {
    // Short, direct factoid with no signals → Easy.
    let result = default_predictor().predict("What is Rust?").unwrap();
    assert_eq!(result.band, DifficultyBand::Easy);
}

#[test]
fn easy_query_score_below_threshold() {
    let result = default_predictor().predict("What is Rust?").unwrap();
    assert!(
        result.score < 0.3,
        "expected score < 0.3, got {}",
        result.score
    );
}

#[test]
fn single_word_query_is_easy() {
    let result = default_predictor().predict("Rust").unwrap();
    assert_eq!(result.band, DifficultyBand::Easy);
}

// ── Band assignment — Medium ──────────────────────────────────────────────────

#[test]
fn medium_band_multi_hop_query() {
    // Multi-hop but no negation/ambiguity; moderate length → Medium.
    let q = "What are the similarities and differences between Rust and Python programming?";
    let result = default_predictor().predict(q).unwrap();
    assert_eq!(result.band, DifficultyBand::Medium);
}

#[test]
fn medium_band_score_in_range() {
    let q = "What are the similarities and differences between Rust and Python programming?";
    let result = default_predictor().predict(q).unwrap();
    assert!(
        result.score >= 0.3 && result.score < 0.55,
        "expected [0.3, 0.55), got {}",
        result.score
    );
}

// ── Band assignment — Hard ────────────────────────────────────────────────────

#[test]
fn hard_band_negation_plus_multi_hop() {
    // Negation + multi-hop + long sentence → Hard.
    let q =
        "It does not work and furthermore it fails for multiple reasons in production environments";
    let result = default_predictor().predict(q).unwrap();
    assert_eq!(result.band, DifficultyBand::Hard);
}

#[test]
fn hard_band_score_in_range() {
    let q =
        "It does not work and furthermore it fails for multiple reasons in production environments";
    let result = default_predictor().predict(q).unwrap();
    assert!(
        result.score >= 0.55 && result.score < 0.75,
        "expected [0.55, 0.75), got {}",
        result.score
    );
}

// ── Band assignment — Ambiguous ───────────────────────────────────────────────

#[test]
fn ambiguous_band_all_signals() {
    // Negation + multi-hop + ambiguity + long → Ambiguous.
    let q = "It cannot be confirmed and furthermore it is neither clear nor obvious which one is correct or maybe it depends";
    let result = default_predictor().predict(q).unwrap();
    assert_eq!(result.band, DifficultyBand::Ambiguous);
}

#[test]
fn ambiguous_band_score_in_range() {
    let q = "It cannot be confirmed and furthermore it is neither clear nor obvious which one is correct or maybe it depends";
    let result = default_predictor().predict(q).unwrap();
    assert!(
        result.score >= 0.75,
        "expected >= 0.75, got {}",
        result.score
    );
}

// ── Score always in [0, 1] ────────────────────────────────────────────────────

#[test]
fn score_bounded_easy() {
    let r = default_predictor().predict("What is Rust?").unwrap();
    assert!(r.score >= 0.0 && r.score <= 1.0);
}

#[test]
fn score_bounded_complex() {
    let q = "It cannot be confirmed and furthermore it is neither clear nor obvious which one is correct or maybe it depends";
    let r = default_predictor().predict(q).unwrap();
    assert!(r.score >= 0.0 && r.score <= 1.0);
}

#[test]
fn score_bounded_negation_only() {
    let r = default_predictor().predict("never").unwrap();
    assert!(r.score >= 0.0 && r.score <= 1.0);
}

// ── DifficultyScore fields ────────────────────────────────────────────────────

#[test]
fn score_query_field_preserved() {
    let q = "What is Rust?";
    let result = default_predictor().predict(q).unwrap();
    // predict trims whitespace; the trimmed form is the stored query.
    assert_eq!(result.query, q.trim());
}

#[test]
fn score_query_field_trimmed() {
    let result = default_predictor().predict("  What is Rust?  ").unwrap();
    assert_eq!(result.query, "What is Rust?");
}

// ── DifficultySignals — negation ──────────────────────────────────────────────

#[test]
fn signals_no_negation_simple() {
    let r = default_predictor().predict("What is Rust?").unwrap();
    assert!(!r.signals.has_negation);
}

#[test]
fn signals_negation_not() {
    let r = default_predictor()
        .predict("do not restart the system")
        .unwrap();
    assert!(r.signals.has_negation);
}

#[test]
fn signals_negation_no() {
    let r = default_predictor()
        .predict("there is no solution available")
        .unwrap();
    assert!(r.signals.has_negation);
}

#[test]
fn signals_negation_never() {
    let r = default_predictor()
        .predict("Rust never panics in safe code")
        .unwrap();
    assert!(r.signals.has_negation);
}

#[test]
fn signals_negation_cannot() {
    let r = default_predictor()
        .predict("the system cannot process requests")
        .unwrap();
    assert!(r.signals.has_negation);
}

#[test]
fn signals_negation_without() {
    let r = default_predictor()
        .predict("Rust without unsafe blocks")
        .unwrap();
    assert!(r.signals.has_negation);
}

#[test]
fn signals_negation_isnt() {
    let r = default_predictor()
        .predict("it isn't working correctly")
        .unwrap();
    assert!(r.signals.has_negation);
}

#[test]
fn signals_negation_neither_nor() {
    let r = default_predictor()
        .predict("neither this nor that is acceptable")
        .unwrap();
    assert!(r.signals.has_negation);
}

#[test]
fn signals_negation_word_boundary_notable() {
    // "notable" contains "not" but not at a word boundary — must NOT fire.
    let r = default_predictor()
        .predict("notable achievements in computing")
        .unwrap();
    assert!(!r.signals.has_negation);
}

// ── DifficultySignals — multi-hop ────────────────────────────────────────────

#[test]
fn signals_no_multi_hop_simple() {
    let r = default_predictor()
        .predict("What is memory safety?")
        .unwrap();
    assert!(!r.signals.multi_hop_indicator);
}

#[test]
fn signals_multi_hop_and() {
    let r = default_predictor()
        .predict("What is Rust and what is Python?")
        .unwrap();
    assert!(r.signals.multi_hop_indicator);
}

#[test]
fn signals_multi_hop_also() {
    let r = default_predictor()
        .predict("Rust is fast also it is memory safe")
        .unwrap();
    assert!(r.signals.multi_hop_indicator);
}

#[test]
fn signals_multi_hop_furthermore() {
    let r = default_predictor()
        .predict("it compiles successfully furthermore it runs fast")
        .unwrap();
    assert!(r.signals.multi_hop_indicator);
}

#[test]
fn signals_multi_hop_between() {
    let r = default_predictor()
        .predict("What is the difference between Rust and C++?")
        .unwrap();
    assert!(r.signals.multi_hop_indicator);
}

#[test]
fn signals_multi_hop_both() {
    let r = default_predictor()
        .predict("both Rust and Python support async programming")
        .unwrap();
    assert!(r.signals.multi_hop_indicator);
}

#[test]
fn signals_multi_hop_whereas() {
    let r = default_predictor()
        .predict("Rust is safe whereas C allows raw pointers")
        .unwrap();
    assert!(r.signals.multi_hop_indicator);
}

#[test]
fn signals_multi_hop_relationship_between() {
    let r = default_predictor()
        .predict("Explain the relationship between ownership and memory safety in Rust")
        .unwrap();
    assert!(r.signals.multi_hop_indicator);
}

// ── DifficultySignals — ambiguity ─────────────────────────────────────────────

#[test]
fn signals_no_ambiguity_simple() {
    // "What is" — "what" is NOT in AMBIGUITY_MARKERS.
    let r = default_predictor()
        .predict("What is memory safety?")
        .unwrap();
    assert!(!r.signals.ambiguity_marker);
}

#[test]
fn signals_ambiguity_who() {
    let r = default_predictor()
        .predict("Who designed the Rust language?")
        .unwrap();
    assert!(r.signals.ambiguity_marker);
}

#[test]
fn signals_ambiguity_or() {
    let r = default_predictor()
        .predict("Is Rust fast or slow?")
        .unwrap();
    assert!(r.signals.ambiguity_marker);
}

#[test]
fn signals_ambiguity_maybe() {
    let r = default_predictor()
        .predict("Maybe the answer is yes")
        .unwrap();
    assert!(r.signals.ambiguity_marker);
}

#[test]
fn signals_ambiguity_unclear() {
    let r = default_predictor()
        .predict("The semantics are unclear")
        .unwrap();
    assert!(r.signals.ambiguity_marker);
}

#[test]
fn signals_ambiguity_vague() {
    let r = default_predictor()
        .predict("The specification is vague")
        .unwrap();
    assert!(r.signals.ambiguity_marker);
}

#[test]
fn signals_ambiguity_perhaps() {
    let r = default_predictor()
        .predict("Perhaps the algorithm is optimal")
        .unwrap();
    assert!(r.signals.ambiguity_marker);
}

#[test]
fn signals_ambiguity_either() {
    let r = default_predictor()
        .predict("either Rust or Go would work here")
        .unwrap();
    assert!(r.signals.ambiguity_marker);
}

#[test]
fn signals_ambiguity_word_boundary_or_in_horror() {
    // "horror" contains "or" but not at a word boundary — must NOT fire.
    let r = default_predictor()
        .predict("the horror is unimaginable")
        .unwrap();
    assert!(!r.signals.ambiguity_marker);
}

// ── DifficultySignals — token_count ──────────────────────────────────────────

#[test]
fn signals_token_count_single_word() {
    let r = default_predictor().predict("Rust").unwrap();
    assert_eq!(r.signals.token_count, 1);
}

#[test]
fn signals_token_count_known_query() {
    // "What is Rust and Python?" → 5 alphanumeric tokens (the "?" is dropped).
    let r = default_predictor()
        .predict("What is Rust and Python?")
        .unwrap();
    assert_eq!(r.signals.token_count, 5);
}

#[test]
fn signals_token_count_punct_ignored() {
    // Punctuation is a split point and single-char leftovers are dropped.
    let r = default_predictor()
        .predict("fast, safe, concurrent")
        .unwrap();
    // tokens: ["fast", "safe", "concurrent"] — the "," splits but leaves no
    // single char because both sides are ≥2 chars.
    assert_eq!(r.signals.token_count, 3);
}

// ── DifficultySignals — rare_word_ratio ──────────────────────────────────────

#[test]
fn signals_rare_word_ratio_all_stop_words() {
    // All tokens are stop words → ratio = 0.0.
    let r = default_predictor().predict("is the in by for").unwrap();
    // Tokens: ["is", "the", "in", "by", "for"] — all in STOP_WORDS.
    assert_eq!(r.signals.rare_word_ratio, 0.0_f32);
}

#[test]
fn signals_rare_word_ratio_all_domain_terms() {
    // All tokens are domain-specific (not in STOP_WORDS) → ratio = 1.0.
    let r = default_predictor()
        .predict("Rust ownership borrowing concurrency performance")
        .unwrap();
    assert_eq!(r.signals.rare_word_ratio, 1.0_f32);
}

#[test]
fn signals_rare_word_ratio_in_zero_one() {
    let r = default_predictor()
        .predict("What is the difference between Rust and Python?")
        .unwrap();
    assert!(
        r.signals.rare_word_ratio >= 0.0 && r.signals.rare_word_ratio <= 1.0,
        "rare_word_ratio out of [0,1]: {}",
        r.signals.rare_word_ratio
    );
}

// ── Score monotonicity ────────────────────────────────────────────────────────

#[test]
fn score_higher_with_negation() {
    let without = default_predictor()
        .predict("Rust is fast and efficient for systems programming")
        .unwrap();
    let with_neg = default_predictor()
        .predict("Rust is not fast and not efficient for systems programming")
        .unwrap();
    assert!(
        with_neg.score > without.score,
        "expected negation to raise score: {} vs {}",
        with_neg.score,
        without.score
    );
}

#[test]
fn score_higher_with_multi_hop() {
    let simple = default_predictor().predict("What is Rust?").unwrap();
    let multi = default_predictor()
        .predict("What is the difference between Rust and Python?")
        .unwrap();
    assert!(
        multi.score > simple.score,
        "expected multi-hop to raise score: {} vs {}",
        multi.score,
        simple.score
    );
}

#[test]
fn score_higher_with_more_tokens_same_signals() {
    // No signals fire in either query; only length and rare_word_ratio vary.
    let short_q = default_predictor().predict("Rust concurrency").unwrap();
    let long_q = default_predictor()
        .predict("Rust concurrency memory safety performance ownership borrowing lifetime async")
        .unwrap();
    assert!(
        long_q.score > short_q.score,
        "expected longer query to have higher score: {} vs {}",
        long_q.score,
        short_q.score
    );
}

// ── STOP_WORDS invariants ─────────────────────────────────────────────────────

#[test]
fn stop_words_count_at_least_fifty() {
    assert!(
        STOP_WORDS.len() >= 50,
        "expected ≥50 stop words, found {}",
        STOP_WORDS.len()
    );
}

#[test]
fn stop_words_contains_common_articles() {
    assert!(STOP_WORDS.contains(&"a"));
    assert!(STOP_WORDS.contains(&"an"));
    assert!(STOP_WORDS.contains(&"the"));
}

// ── NEGATION_WORDS invariants ─────────────────────────────────────────────────

#[test]
fn negation_words_contains_not() {
    assert!(NEGATION_WORDS.contains(&"not"));
}

#[test]
fn negation_words_contains_never() {
    assert!(NEGATION_WORDS.contains(&"never"));
}

// ── Predictor clone produces identical results ────────────────────────────────

#[test]
fn predictor_clone_identical_results() {
    let p = default_predictor();
    let p2 = p.clone();
    let q =
        "It does not work and furthermore it fails for multiple reasons in production environments";
    let r1 = p.predict(q).unwrap();
    let r2 = p2.predict(q).unwrap();
    assert_eq!(r1.score, r2.score);
    assert_eq!(r1.band, r2.band);
    assert_eq!(r1.signals, r2.signals);
}

// ── Signals struct field accessibility ───────────────────────────────────────

#[test]
fn signals_fields_all_accessible() {
    let r = default_predictor().predict("What is Rust?").unwrap();
    let _has_neg: bool = r.signals.has_negation;
    let _multi: bool = r.signals.multi_hop_indicator;
    let _ambig: bool = r.signals.ambiguity_marker;
    let _count: usize = r.signals.token_count;
    let _ratio: f32 = r.signals.rare_word_ratio;
}

#[test]
fn difficulty_score_fields_all_accessible() {
    let r = default_predictor().predict("What is Rust?").unwrap();
    let _q: &str = &r.query;
    let _s: f32 = r.score;
    let _b: &DifficultyBand = &r.band;
    let _sig = &r.signals;
}

// ── Custom config ─────────────────────────────────────────────────────────────

#[test]
fn zero_weights_gives_near_zero_score() {
    // With all weights zero the score should be 0.0 (no contribution from any signal).
    let cfg = DifficultyConfig {
        negation_weight: 0.0,
        multi_hop_weight: 0.0,
        ambiguity_weight: 0.0,
        length_weight: 0.0,
        rare_word_weight: 0.0,
    };
    let p = DifficultyPredictor::new(cfg);
    let r = p
        .predict("complex multi-hop query with negation not working")
        .unwrap();
    assert_eq!(r.score, 0.0_f32);
    assert_eq!(r.band, DifficultyBand::Easy);
}

#[test]
fn boosted_negation_weight_amplifies_score() {
    // With a much higher negation weight the negated query should outscore the
    // default predictor's assessment of the same query.
    let default_r = default_predictor()
        .predict("Rust cannot guarantee safety without the borrow checker")
        .unwrap();
    let boosted_cfg = DifficultyConfig::new().with_negation_weight(0.9);
    let boosted_r = DifficultyPredictor::new(boosted_cfg)
        .predict("Rust cannot guarantee safety without the borrow checker")
        .unwrap();
    assert!(
        boosted_r.score > default_r.score,
        "boosted negation weight should increase score"
    );
}
