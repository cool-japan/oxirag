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
    clippy::too_many_lines
)]

use super::analyzer::FreshnessAnalyzer;
use super::types::{FreshConfig, FreshError, TimeSensitivity};
use crate::types::{Document, SearchResult};

// ── helpers ───────────────────────────────────────────────────────────────────

fn analyzer() -> FreshnessAnalyzer {
    FreshnessAnalyzer::new(FreshConfig::new())
}

fn result(content: &str, score: f32, rank: usize) -> SearchResult {
    SearchResult::new(Document::new(content), score, rank)
}

fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

// ── FreshConfig defaults & builders ─────────────────────────────────────────────

#[test]
fn config_default_values() {
    let c = FreshConfig::default();
    assert_eq!(c.half_life_days, 180.0);
    assert_eq!(c.staleness_threshold_days, 30.0);
    assert_eq!(c.freshness_weight, 0.5);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(FreshConfig::new(), FreshConfig::default());
}

#[test]
fn config_builder_half_life() {
    let c = FreshConfig::new().with_half_life_days(90.0);
    assert_eq!(c.half_life_days, 90.0);
    // others unchanged
    assert_eq!(c.staleness_threshold_days, 30.0);
    assert_eq!(c.freshness_weight, 0.5);
}

#[test]
fn config_builder_staleness_threshold() {
    let c = FreshConfig::new().with_staleness_threshold_days(7.0);
    assert_eq!(c.staleness_threshold_days, 7.0);
}

#[test]
fn config_builder_freshness_weight() {
    let c = FreshConfig::new().with_freshness_weight(0.9);
    assert_eq!(c.freshness_weight, 0.9);
}

#[test]
fn config_builders_chain() {
    let c = FreshConfig::new()
        .with_half_life_days(10.0)
        .with_staleness_threshold_days(5.0)
        .with_freshness_weight(0.25);
    assert_eq!(c.half_life_days, 10.0);
    assert_eq!(c.staleness_threshold_days, 5.0);
    assert_eq!(c.freshness_weight, 0.25);
}

#[test]
fn analyzer_default_uses_default_config() {
    let a = FreshnessAnalyzer::default();
    assert_eq!(*a.config(), FreshConfig::default());
}

#[test]
fn analyzer_exposes_config() {
    let a = FreshnessAnalyzer::new(FreshConfig::new().with_half_life_days(42.0));
    assert_eq!(a.config().half_life_days, 42.0);
}

// ── TimeSensitivity helpers ─────────────────────────────────────────────────────

#[test]
fn sensitivity_labels() {
    assert_eq!(TimeSensitivity::Static.label(), "static");
    assert_eq!(TimeSensitivity::SlowChanging.label(), "slow_changing");
    assert_eq!(TimeSensitivity::FastChanging.label(), "fast_changing");
}

#[test]
fn sensitivity_base_demand_ordering() {
    assert_eq!(TimeSensitivity::Static.base_demand(), 0.0);
    assert!(TimeSensitivity::Static.base_demand() < TimeSensitivity::SlowChanging.base_demand());
    assert!(
        TimeSensitivity::SlowChanging.base_demand() < TimeSensitivity::FastChanging.base_demand()
    );
}

#[test]
fn sensitivity_default_is_static() {
    assert_eq!(TimeSensitivity::default(), TimeSensitivity::Static);
}

// ── classify: fast-changing ─────────────────────────────────────────────────────

#[test]
fn classify_latest_iphone_price_is_fast_changing() {
    let a = analyzer();
    let assessment = a.classify("What is the latest iPhone price?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
    assert!(assessment.freshness_demand > 0.7);
    assert!(!assessment.signals.is_empty());
    assert!(assessment.is_time_sensitive());
}

#[test]
fn classify_current_stock_price_is_fast_changing() {
    let a = analyzer();
    let assessment = a
        .classify("What is the current stock price of Apple?")
        .unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
    assert!(assessment.signals.iter().any(|s| s == "domain:stock"));
}

#[test]
fn classify_weather_is_fast_changing() {
    let a = analyzer();
    let assessment = a.classify("What is the weather in Tokyo?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
}

#[test]
fn classify_today_marker_is_fast_changing() {
    let a = analyzer();
    let assessment = a.classify("Who is playing today?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
    assert!(assessment.signals.iter().any(|s| s == "marker:today"));
}

#[test]
fn classify_now_marker_is_fast_changing() {
    let a = analyzer();
    let assessment = a.classify("What time is it now?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
}

#[test]
fn classify_news_domain_is_fast_changing() {
    let a = analyzer();
    let assessment = a.classify("Latest news about the election").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
    assert!(assessment.signals.iter().any(|s| s == "domain:news"));
}

#[test]
fn classify_score_domain_is_fast_changing() {
    let a = analyzer();
    let assessment = a.classify("What was the score of the game?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
}

#[test]
fn classify_fast_changing_demand_is_high() {
    let a = analyzer();
    let assessment = a.classify("current weather and stock price now").unwrap();
    // many fast signals → demand saturates high
    assert!(assessment.freshness_demand >= 0.8);
    assert!(assessment.freshness_demand <= 1.0);
}

// ── classify: static ─────────────────────────────────────────────────────────────

#[test]
fn classify_who_wrote_hamlet_is_static() {
    let a = analyzer();
    let assessment = a.classify("Who wrote Hamlet?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::Static);
    assert_eq!(assessment.freshness_demand, 0.0);
    assert!(assessment.signals.is_empty());
    assert!(!assessment.is_time_sensitive());
}

#[test]
fn classify_speed_of_light_is_static() {
    let a = analyzer();
    let assessment = a.classify("What is the speed of light?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::Static);
    assert_eq!(assessment.freshness_demand, 0.0);
}

#[test]
fn classify_capital_of_france_is_static() {
    let a = analyzer();
    let assessment = a.classify("What is the capital of France?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::Static);
}

#[test]
fn classify_math_fact_is_static() {
    let a = analyzer();
    let assessment = a.classify("How many sides does a triangle have?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::Static);
    assert!(assessment.signals.is_empty());
}

// ── classify: slow-changing ─────────────────────────────────────────────────────

#[test]
fn classify_recent_marker_is_slow_changing() {
    let a = analyzer();
    let assessment = a
        .classify("What are recent developments in fusion?")
        .unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::SlowChanging);
    assert!(assessment.freshness_demand > 0.0);
    assert!(assessment.freshness_demand < 0.8);
}

#[test]
fn classify_newest_comparative_is_slow_changing() {
    let a = analyzer();
    let assessment = a.classify("Who is the newest member of the band?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::SlowChanging);
    assert!(assessment.signals.iter().any(|s| s == "comparative:newest"));
}

#[test]
fn classify_modern_marker_is_slow_changing() {
    let a = analyzer();
    let assessment = a.classify("What is modern architecture?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::SlowChanging);
}

// ── classify: explicit year & phrases ───────────────────────────────────────────

#[test]
fn classify_explicit_year_raises_sensitivity() {
    let a = analyzer();
    let with_year = a.classify("Who won the championship in 2024?").unwrap();
    let without_year = a.classify("Who won the championship?").unwrap();
    assert_eq!(without_year.sensitivity, TimeSensitivity::Static);
    assert_eq!(with_year.sensitivity, TimeSensitivity::SlowChanging);
    assert!(with_year.signals.iter().any(|s| s == "year:2024"));
    assert!(with_year.freshness_demand > without_year.freshness_demand);
}

#[test]
fn classify_this_year_phrase_raises_sensitivity() {
    let a = analyzer();
    let assessment = a.classify("Which movies came out this year?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::SlowChanging);
    assert!(assessment.signals.iter().any(|s| s == "phrase:this_year"));
}

#[test]
fn classify_non_year_number_not_treated_as_year() {
    let a = analyzer();
    // 42 and 1500 (out of 1900..=2099) must not fire a year signal alone
    let assessment = a.classify("What is 42 divided by 6?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::Static);
    assert!(!assessment.signals.iter().any(|s| s.starts_with("year:")));
}

#[test]
fn classify_year_out_of_range_ignored() {
    let a = analyzer();
    let assessment = a.classify("What happened in 1066?").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::Static);
}

#[test]
fn classify_fast_overrides_recency() {
    let a = analyzer();
    // recency marker "latest" + fast domain "price" → FastChanging wins
    let assessment = a.classify("latest price").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
}

#[test]
fn classify_is_case_insensitive() {
    let a = analyzer();
    let lower = a.classify("latest iphone price").unwrap();
    let upper = a.classify("LATEST IPHONE PRICE").unwrap();
    assert_eq!(lower.sensitivity, upper.sensitivity);
    assert_eq!(lower.freshness_demand, upper.freshness_demand);
    assert_eq!(lower.signals, upper.signals);
}

#[test]
fn classify_punctuation_does_not_break_tokens() {
    let a = analyzer();
    let assessment = a.classify("price?! ...now,").unwrap();
    assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
}

#[test]
fn classify_demand_in_unit_interval() {
    let a = analyzer();
    for q in [
        "Who wrote Hamlet?",
        "recent news",
        "current stock price now today weather",
        "What about 2024?",
    ] {
        let d = a.classify(q).unwrap().freshness_demand;
        assert!((0.0..=1.0).contains(&d), "demand {d} out of range for {q}");
    }
}

// ── classify: errors ─────────────────────────────────────────────────────────────

#[test]
fn classify_empty_query_errors() {
    let a = analyzer();
    assert_eq!(a.classify("").unwrap_err(), FreshError::EmptyQuery);
}

#[test]
fn classify_whitespace_query_errors() {
    let a = analyzer();
    assert_eq!(a.classify("   \t\n").unwrap_err(), FreshError::EmptyQuery);
}

#[test]
fn classify_punctuation_only_query_errors() {
    let a = analyzer();
    assert_eq!(a.classify("?!.,-").unwrap_err(), FreshError::EmptyQuery);
}

// ── freshness_score: decay ───────────────────────────────────────────────────────

#[test]
fn freshness_age_zero_is_one() {
    let a = analyzer();
    assert_eq!(a.freshness_score(0.0), 1.0);
}

#[test]
fn freshness_age_half_life_is_half() {
    let a = analyzer();
    let s = a.freshness_score(180.0);
    assert!(approx(s, 0.5, 1e-6), "expected 0.5, got {s}");
}

#[test]
fn freshness_age_two_half_lives_is_quarter() {
    let a = analyzer();
    let s = a.freshness_score(360.0);
    assert!(approx(s, 0.25, 1e-6), "expected 0.25, got {s}");
}

#[test]
fn freshness_custom_half_life() {
    let a = FreshnessAnalyzer::new(FreshConfig::new().with_half_life_days(30.0));
    assert!(approx(a.freshness_score(30.0), 0.5, 1e-6));
    assert!(approx(a.freshness_score(60.0), 0.25, 1e-6));
}

#[test]
fn freshness_monotonically_decreasing() {
    let a = analyzer();
    let mut prev = a.freshness_score(0.0);
    for age in [1.0, 10.0, 50.0, 180.0, 365.0, 1000.0] {
        let cur = a.freshness_score(age);
        assert!(cur < prev, "freshness not decreasing at age {age}");
        prev = cur;
    }
}

#[test]
fn freshness_negative_age_clamped_to_one() {
    let a = analyzer();
    assert_eq!(a.freshness_score(-100.0), 1.0);
}

#[test]
fn freshness_in_unit_interval() {
    let a = analyzer();
    for age in [0.0, 5.0, 180.0, 10_000.0] {
        let s = a.freshness_score(age);
        assert!(s > 0.0 && s <= 1.0, "freshness {s} out of (0,1] at {age}");
    }
}

#[test]
fn freshness_zero_half_life_disables_decay() {
    let a = FreshnessAnalyzer::new(FreshConfig::new().with_half_life_days(0.0));
    assert_eq!(a.freshness_score(1000.0), 1.0);
}

// ── is_stale ─────────────────────────────────────────────────────────────────────

#[test]
fn is_stale_fast_changing_old_answer_true() {
    let a = analyzer();
    assert!(
        a.is_stale("What is the current stock price?", 45.0)
            .unwrap()
    );
}

#[test]
fn is_stale_fast_changing_fresh_answer_false() {
    let a = analyzer();
    assert!(!a.is_stale("What is the current stock price?", 5.0).unwrap());
}

#[test]
fn is_stale_static_query_never_stale() {
    let a = analyzer();
    assert!(!a.is_stale("Who wrote Hamlet?", 100_000.0).unwrap());
}

#[test]
fn is_stale_slow_changing_never_stale() {
    let a = analyzer();
    assert!(!a.is_stale("recent developments", 9999.0).unwrap());
}

#[test]
fn is_stale_boundary_not_strictly_greater_false() {
    let a = analyzer();
    // exactly at threshold (30.0) is NOT stale (strict >)
    assert!(!a.is_stale("current price", 30.0).unwrap());
}

#[test]
fn is_stale_just_over_boundary_true() {
    let a = analyzer();
    assert!(a.is_stale("current price", 30.001).unwrap());
}

#[test]
fn is_stale_custom_threshold() {
    let a = FreshnessAnalyzer::new(FreshConfig::new().with_staleness_threshold_days(7.0));
    assert!(a.is_stale("weather now", 8.0).unwrap());
    assert!(!a.is_stale("weather now", 6.0).unwrap());
}

#[test]
fn is_stale_empty_query_errors() {
    let a = analyzer();
    assert_eq!(a.is_stale("", 10.0).unwrap_err(), FreshError::EmptyQuery);
}

// ── rerank ───────────────────────────────────────────────────────────────────────

#[test]
fn rerank_promotes_fresher_for_fast_changing() {
    let a = analyzer();
    // Two equally-relevant docs; second is much fresher and should rise.
    let results = vec![result("stale", 0.8, 0), result("fresh", 0.8, 1)];
    let ages = vec![3650.0, 0.0]; // ~10 years vs brand new
    let out = a.rerank("current stock price", &results, &ages).unwrap();
    assert_eq!(out[0].document.content, "fresh");
    assert_eq!(out[1].document.content, "stale");
    // ranks re-numbered
    assert_eq!(out[0].rank, 0);
    assert_eq!(out[1].rank, 1);
}

#[test]
fn rerank_identity_for_static_query() {
    let a = analyzer();
    let results = vec![
        result("a", 0.9, 0),
        result("b", 0.5, 1),
        result("c", 0.7, 2),
    ];
    let ages = vec![1000.0, 0.0, 500.0];
    let out = a.rerank("Who wrote Hamlet?", &results, &ages).unwrap();
    // order preserved exactly, scores unchanged
    assert_eq!(out[0].document.content, "a");
    assert_eq!(out[1].document.content, "b");
    assert_eq!(out[2].document.content, "c");
    assert_eq!(out[0].score, 0.9);
    assert_eq!(out[1].score, 0.5);
    assert_eq!(out[2].score, 0.7);
}

#[test]
fn rerank_static_keeps_original_ranks() {
    let a = analyzer();
    let results = vec![result("a", 0.9, 0), result("b", 0.5, 1)];
    let ages = vec![0.0, 9999.0];
    let out = a.rerank("capital of France", &results, &ages).unwrap();
    assert_eq!(out[0].rank, 0);
    assert_eq!(out[1].rank, 1);
}

#[test]
fn rerank_blends_score_with_freshness() {
    let a = analyzer();
    // single doc; verify the exact blended score
    let results = vec![result("only", 0.6, 0)];
    let ages = vec![180.0]; // freshness 0.5
    let assessment = a.classify("current price").unwrap();
    let weight = assessment.freshness_demand * a.config().freshness_weight;
    let expected = (1.0 - weight) * 0.6 + weight * 0.5;
    let out = a.rerank("current price", &results, &ages).unwrap();
    assert!(approx(out[0].score, expected, 1e-6), "got {}", out[0].score);
}

#[test]
fn rerank_weaker_demand_blends_less() {
    let a = analyzer();
    // slow-changing query has lower demand than fast-changing → freshness moves
    // the score less. Compare blended weights via single-doc scores.
    let results = vec![result("only", 0.2, 0)];
    let fresh_ages = vec![0.0]; // freshness 1.0 — pulls score up
    let slow = a.rerank("recent topic", &results, &fresh_ages).unwrap()[0].score;
    let fast = a.rerank("current price", &results, &fresh_ages).unwrap()[0].score;
    // both pulled up toward 1.0, fast more so
    assert!(fast > slow, "fast {fast} should exceed slow {slow}");
    assert!(slow > 0.2);
}

#[test]
fn rerank_does_not_promote_fresh_when_relevance_dominates() {
    let a = analyzer();
    // Highly relevant stale doc vs barely-relevant fresh doc; for a slow-changing
    // query the relevance gap should still keep the relevant doc on top.
    let results = vec![result("relevant", 0.95, 0), result("fresh", 0.05, 1)];
    let ages = vec![3650.0, 0.0];
    let out = a.rerank("recent overview", &results, &ages).unwrap();
    assert_eq!(out[0].document.content, "relevant");
}

#[test]
fn rerank_empty_results_with_empty_ages_ok() {
    let a = analyzer();
    let out = a.rerank("current price", &[], &[]).unwrap();
    assert!(out.is_empty());
}

#[test]
fn rerank_length_mismatch_errors() {
    let a = analyzer();
    let results = vec![result("a", 0.5, 0), result("b", 0.5, 1)];
    let ages = vec![1.0]; // too few
    let err = a.rerank("current price", &results, &ages).unwrap_err();
    assert_eq!(
        err,
        FreshError::LengthMismatch {
            ages: 1,
            results: 2,
        }
    );
}

#[test]
fn rerank_length_mismatch_too_many_ages() {
    let a = analyzer();
    let results = vec![result("a", 0.5, 0)];
    let ages = vec![1.0, 2.0, 3.0];
    let err = a.rerank("current price", &results, &ages).unwrap_err();
    assert_eq!(
        err,
        FreshError::LengthMismatch {
            ages: 3,
            results: 1,
        }
    );
}

#[test]
fn rerank_empty_query_errors() {
    let a = analyzer();
    let results = vec![result("a", 0.5, 0)];
    let ages = vec![1.0];
    assert_eq!(
        a.rerank("", &results, &ages).unwrap_err(),
        FreshError::EmptyQuery
    );
}

#[test]
fn rerank_empty_query_checked_before_length() {
    let a = analyzer();
    // Empty query should surface EmptyQuery even if lengths also mismatch.
    let results = vec![result("a", 0.5, 0)];
    let ages: Vec<f64> = vec![];
    assert_eq!(
        a.rerank("", &results, &ages).unwrap_err(),
        FreshError::EmptyQuery
    );
}

// ── error formatting ─────────────────────────────────────────────────────────────

#[test]
fn error_display_messages() {
    assert_eq!(
        FreshError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(
        FreshError::LengthMismatch {
            ages: 2,
            results: 3,
        }
        .to_string(),
        "ages length 2 != results length 3"
    );
}

// ── determinism ──────────────────────────────────────────────────────────────────

#[test]
fn classify_is_deterministic() {
    let a = analyzer();
    let q = "What is the latest iPhone price in 2024?";
    let first = a.classify(q).unwrap();
    for _ in 0..32 {
        let again = a.classify(q).unwrap();
        assert_eq!(first.sensitivity, again.sensitivity);
        assert_eq!(first.freshness_demand, again.freshness_demand);
        assert_eq!(first.signals, again.signals);
    }
}

#[test]
fn freshness_score_is_deterministic() {
    let a = analyzer();
    let first = a.freshness_score(123.4);
    for _ in 0..16 {
        assert_eq!(a.freshness_score(123.4), first);
    }
}

#[test]
fn rerank_is_deterministic() {
    let a = analyzer();
    let results = vec![
        result("a", 0.8, 0),
        result("b", 0.7, 1),
        result("c", 0.6, 2),
    ];
    let ages = vec![100.0, 0.0, 500.0];
    let first = a.rerank("current price", &results, &ages).unwrap();
    for _ in 0..16 {
        let again = a.rerank("current price", &results, &ages).unwrap();
        let fa: Vec<_> = first.iter().map(|r| &r.document.content).collect();
        let ag: Vec<_> = again.iter().map(|r| &r.document.content).collect();
        assert_eq!(fa, ag);
        for (x, y) in first.iter().zip(again.iter()) {
            assert_eq!(x.score, y.score);
        }
    }
}

#[test]
fn is_stale_is_deterministic() {
    let a = analyzer();
    let first = a.is_stale("current price", 45.0).unwrap();
    for _ in 0..16 {
        assert_eq!(a.is_stale("current price", 45.0).unwrap(), first);
    }
}
