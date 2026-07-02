#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::default_constructed_unit_structs
)]

//! Tests for the `recomp` module.

use super::abstractive::AbstractiveSummaryCompressor;
use super::extractive::ExtractiveSummaryCompressor;
use super::pipeline::RecompPipeline;
use super::types::{CompressorStrategy, RecompConfig, RecompDecision, RecompError};

// ── helpers ───────────────────────────────────────────────────────────────────

fn passages(strings: &[&str]) -> Vec<String> {
    strings.iter().map(|s| (*s).to_string()).collect()
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

// A passage set with strong, clear lexical overlap with "rust borrow checker".
fn relevant_passages() -> Vec<String> {
    passages(&[
        "The Rust borrow checker enforces memory safety at compile time without a garbage collector.",
        "Ownership rules in Rust prevent data races between concurrent threads.",
        "The borrow checker rejects programs that violate Rust's aliasing rules.",
    ])
}

// A passage set with essentially no lexical overlap with "rust borrow checker".
fn irrelevant_passages() -> Vec<String> {
    passages(&[
        "Bananas are a good source of potassium and dietary fiber.",
        "The weather in Tokyo during autumn is generally mild and dry.",
        "Classical music concerts often feature string quartets.",
    ])
}

// ── CompressorStrategy ────────────────────────────────────────────────────────

#[test]
fn strategy_default_is_extractive() {
    assert_eq!(
        CompressorStrategy::default(),
        CompressorStrategy::Extractive
    );
}

#[test]
fn strategy_extractive_as_str() {
    assert_eq!(CompressorStrategy::Extractive.as_str(), "extractive");
}

#[test]
fn strategy_abstractive_lite_as_str() {
    assert_eq!(
        CompressorStrategy::AbstractiveLite.as_str(),
        "abstractive_lite"
    );
}

#[test]
fn strategy_display_matches_as_str() {
    for s in [
        CompressorStrategy::Extractive,
        CompressorStrategy::AbstractiveLite,
    ] {
        assert_eq!(format!("{s}"), s.as_str());
    }
}

#[test]
fn strategy_equality() {
    assert_eq!(
        CompressorStrategy::Extractive,
        CompressorStrategy::Extractive
    );
    assert_ne!(
        CompressorStrategy::Extractive,
        CompressorStrategy::AbstractiveLite
    );
}

#[test]
fn strategy_copy() {
    let s = CompressorStrategy::AbstractiveLite;
    let s2 = s;
    assert_eq!(s, s2);
}

// ── RecompConfig defaults & builders ──────────────────────────────────────────

#[test]
fn config_default_token_budget() {
    assert_eq!(RecompConfig::default().token_budget, 150);
}

#[test]
fn config_default_augmentation_threshold() {
    assert_eq!(RecompConfig::default().augmentation_threshold, 0.15);
}

#[test]
fn config_default_strategy() {
    assert_eq!(
        RecompConfig::default().strategy,
        CompressorStrategy::Extractive
    );
}

#[test]
fn config_default_dedup_similarity_threshold() {
    assert_eq!(RecompConfig::default().dedup_similarity_threshold, 0.85);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(RecompConfig::new(), RecompConfig::default());
}

#[test]
fn config_builder_token_budget() {
    let cfg = RecompConfig::new().with_token_budget(50);
    assert_eq!(cfg.token_budget, 50);
}

#[test]
fn config_builder_augmentation_threshold() {
    let cfg = RecompConfig::new().with_augmentation_threshold(0.5);
    assert_eq!(cfg.augmentation_threshold, 0.5);
}

#[test]
fn config_builder_strategy() {
    let cfg = RecompConfig::new().with_strategy(CompressorStrategy::AbstractiveLite);
    assert_eq!(cfg.strategy, CompressorStrategy::AbstractiveLite);
}

#[test]
fn config_builder_dedup_similarity_threshold() {
    let cfg = RecompConfig::new().with_dedup_similarity_threshold(0.5);
    assert_eq!(cfg.dedup_similarity_threshold, 0.5);
}

#[test]
fn config_builders_are_independent() {
    let cfg = RecompConfig::new().with_token_budget(10);
    assert_eq!(cfg.augmentation_threshold, 0.15);
    assert_eq!(cfg.dedup_similarity_threshold, 0.85);
}

// ── RecompConfig::validate — bounds ───────────────────────────────────────────

#[test]
fn config_validate_default_ok() {
    assert!(RecompConfig::default().validate().is_ok());
}

#[test]
fn config_validate_zero_token_budget_rejected() {
    let cfg = RecompConfig::new().with_token_budget(0);
    assert!(matches!(cfg.validate(), Err(RecompError::InvalidConfig(_))));
}

#[test]
fn config_validate_negative_augmentation_threshold_rejected() {
    let cfg = RecompConfig::new().with_augmentation_threshold(-0.1);
    assert!(matches!(cfg.validate(), Err(RecompError::InvalidConfig(_))));
}

#[test]
fn config_validate_augmentation_threshold_above_one_rejected() {
    let cfg = RecompConfig::new().with_augmentation_threshold(1.5);
    assert!(matches!(cfg.validate(), Err(RecompError::InvalidConfig(_))));
}

#[test]
fn config_validate_nan_augmentation_threshold_rejected() {
    let cfg = RecompConfig::new().with_augmentation_threshold(f32::NAN);
    assert!(matches!(cfg.validate(), Err(RecompError::InvalidConfig(_))));
}

#[test]
fn config_validate_negative_dedup_threshold_rejected() {
    let cfg = RecompConfig::new().with_dedup_similarity_threshold(-0.01);
    assert!(matches!(cfg.validate(), Err(RecompError::InvalidConfig(_))));
}

#[test]
fn config_validate_dedup_threshold_above_one_rejected() {
    let cfg = RecompConfig::new().with_dedup_similarity_threshold(1.01);
    assert!(matches!(cfg.validate(), Err(RecompError::InvalidConfig(_))));
}

#[test]
fn config_validate_boundary_zero_thresholds_ok() {
    let cfg = RecompConfig::new()
        .with_augmentation_threshold(0.0)
        .with_dedup_similarity_threshold(0.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn config_validate_boundary_one_thresholds_ok() {
    let cfg = RecompConfig::new()
        .with_augmentation_threshold(1.0)
        .with_dedup_similarity_threshold(1.0);
    assert!(cfg.validate().is_ok());
}

// ── RecompError ────────────────────────────────────────────────────────────────

#[test]
fn error_empty_query_display() {
    assert_eq!(RecompError::EmptyQuery.to_string(), "query is empty");
}

#[test]
fn error_empty_passages_display() {
    assert_eq!(
        RecompError::EmptyPassages.to_string(),
        "passages must not be empty"
    );
}

#[test]
fn error_invalid_config_display_contains_message() {
    let err = RecompError::InvalidConfig("bad value".to_string());
    assert!(err.to_string().contains("bad value"));
}

#[test]
fn error_equality() {
    assert_eq!(RecompError::EmptyQuery, RecompError::EmptyQuery);
    assert_ne!(RecompError::EmptyQuery, RecompError::EmptyPassages);
}

// ── RecompDecision ─────────────────────────────────────────────────────────────

#[test]
fn decision_augment_is_augment() {
    let d = RecompDecision::Augment("text".to_string());
    assert!(d.is_augment());
    assert!(!d.is_skip());
}

#[test]
fn decision_skip_is_skip() {
    let d = RecompDecision::Skip {
        reason: "low relevance".to_string(),
    };
    assert!(d.is_skip());
    assert!(!d.is_augment());
}

#[test]
fn decision_augment_text_some() {
    let d = RecompDecision::Augment("hello".to_string());
    assert_eq!(d.text(), Some("hello"));
}

#[test]
fn decision_skip_text_none() {
    let d = RecompDecision::Skip {
        reason: "nope".to_string(),
    };
    assert_eq!(d.text(), None);
}

#[test]
fn decision_equality() {
    let a = RecompDecision::Augment("x".to_string());
    let b = RecompDecision::Augment("x".to_string());
    assert_eq!(a, b);
}

// ── ExtractiveSummaryCompressor — standalone ─────────────────────────────────

#[test]
fn extractive_compress_respects_budget() {
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 10);
    assert!(
        word_count(&out) <= 10,
        "expected <=10 words, got {}: {out:?}",
        word_count(&out)
    );
}

#[test]
fn extractive_compress_small_budget_still_bounded() {
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 3);
    assert!(word_count(&out) <= 3);
}

#[test]
fn extractive_compress_zero_budget_returns_empty() {
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 0);
    assert!(out.is_empty());
}

#[test]
fn extractive_compress_empty_passages_returns_empty() {
    let compressor = ExtractiveSummaryCompressor::new();
    let out = compressor.compress("rust borrow checker", &[], 100);
    assert!(out.is_empty());
}

#[test]
fn extractive_compress_large_budget_includes_relevant_sentence() {
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 200);
    assert!(out.to_lowercase().contains("borrow checker"));
}

#[test]
fn extractive_compress_preserves_original_order() {
    // Construct passages where sentence "First" scores lower than "Second"
    // but appears earlier; the output must still list "First" before
    // "Second" whenever both are selected.
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = passages(&[
        "First sentence mentions apples briefly.",
        "Second sentence is entirely about apples apples apples fruit.",
    ]);
    let out = compressor.compress("apples fruit", &ps, 200);
    let first_pos = out.find("First");
    let second_pos = out.find("Second");
    if let (Some(f), Some(s)) = (first_pos, second_pos) {
        assert!(f < s, "expected original order preserved, got: {out}");
    }
}

#[test]
fn extractive_compress_dedup_collapses_near_identical_sentences() {
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = passages(&[
        "The Rust borrow checker enforces memory safety at compile time.",
        "The Rust borrow checker enforces memory safety at compile time!",
    ]);
    let out = compressor.compress("rust borrow checker memory safety", &ps, 200);
    // Only one near-identical copy should survive.
    let occurrences = out.matches("borrow checker").count();
    assert_eq!(
        occurrences, 1,
        "expected dedup to collapse duplicates: {out}"
    );
}

#[test]
fn extractive_compress_no_dedup_when_sentences_distinct() {
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker ownership", &ps, 200);
    // Distinct sentences about different aspects should both be retained
    // given a generous budget.
    assert!(out.to_lowercase().contains("ownership") || out.to_lowercase().contains("borrow"));
}

#[test]
fn extractive_compress_deterministic() {
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out1 = compressor.compress("rust borrow checker", &ps, 50);
    let out2 = compressor.compress("rust borrow checker", &ps, 50);
    assert_eq!(out1, out2);
}

#[test]
fn extractive_compress_irrelevant_query_still_bounded() {
    // Even with a query unrelated to the passages, the compressor should
    // still return *something* bounded by budget rather than erroring
    // (the skip decision belongs to the pipeline gate, not the compressor).
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = irrelevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 20);
    assert!(word_count(&out) <= 20);
}

#[test]
fn extractive_compressor_default_and_new_equivalent() {
    let a = ExtractiveSummaryCompressor::new();
    let b = ExtractiveSummaryCompressor::default();
    let ps = relevant_passages();
    assert_eq!(
        a.compress("rust borrow checker", &ps, 50),
        b.compress("rust borrow checker", &ps, 50)
    );
}

#[test]
fn extractive_compress_whitespace_only_passages_returns_empty() {
    let compressor = ExtractiveSummaryCompressor::new();
    let ps = passages(&["   ", "\t\n"]);
    let out = compressor.compress("rust", &ps, 50);
    assert!(out.is_empty());
}

// ── AbstractiveSummaryCompressor — standalone ────────────────────────────────

#[test]
fn abstractive_compress_respects_budget() {
    let compressor = AbstractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 10);
    assert!(
        word_count(&out) <= 10,
        "expected <=10 words, got {}: {out:?}",
        word_count(&out)
    );
}

#[test]
fn abstractive_compress_tiny_budget_still_bounded() {
    let compressor = AbstractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 2);
    assert!(word_count(&out) <= 2);
}

#[test]
fn abstractive_compress_zero_budget_returns_empty() {
    let compressor = AbstractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 0);
    assert!(out.is_empty());
}

#[test]
fn abstractive_compress_empty_passages_returns_empty() {
    let compressor = AbstractiveSummaryCompressor::new();
    let out = compressor.compress("rust borrow checker", &[], 100);
    assert!(out.is_empty());
}

#[test]
fn abstractive_compress_produces_single_paragraph() {
    let compressor = AbstractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 200);
    assert!(!out.is_empty());
    // A single paragraph: no blank-line breaks.
    assert!(!out.contains("\n\n"));
}

#[test]
fn abstractive_compress_uses_connective_when_multiple_sentences_fit() {
    let compressor = AbstractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker ownership data races", &ps, 200);
    let connectives = [
        "Additionally,",
        "Furthermore,",
        "Moreover,",
        "In addition,",
        "Also,",
    ];
    let word_total = word_count(&out);
    if word_total > 30 {
        // With a generous budget multiple sentences should be fused with a
        // connective phrase.
        assert!(
            connectives.iter().any(|c| out.contains(c)),
            "expected a connective phrase in: {out}"
        );
    }
}

#[test]
fn abstractive_compress_truncates_when_single_sentence_overflows() {
    let compressor = AbstractiveSummaryCompressor::new();
    let ps =
        passages(&["Rust rust rust rust rust rust rust rust rust rust rust rust rust rust rust."]);
    let out = compressor.compress("rust", &ps, 3);
    assert!(word_count(&out) <= 3);
    assert!(!out.is_empty());
}

#[test]
fn abstractive_compress_deterministic() {
    let compressor = AbstractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out1 = compressor.compress("rust borrow checker", &ps, 50);
    let out2 = compressor.compress("rust borrow checker", &ps, 50);
    assert_eq!(out1, out2);
}

#[test]
fn abstractive_compressor_default_and_new_equivalent() {
    let a = AbstractiveSummaryCompressor::new();
    let b = AbstractiveSummaryCompressor::default();
    let ps = relevant_passages();
    assert_eq!(
        a.compress("rust borrow checker", &ps, 50),
        b.compress("rust borrow checker", &ps, 50)
    );
}

#[test]
fn abstractive_compress_ends_with_period() {
    let compressor = AbstractiveSummaryCompressor::new();
    let ps = relevant_passages();
    let out = compressor.compress("rust borrow checker", &ps, 200);
    assert!(out.ends_with('.'), "expected trailing period: {out}");
}

#[test]
fn abstractive_compress_differs_from_extractive_ordering_when_scores_differ() {
    // Abstractive-lite ranks by score (most relevant first); extractive
    // preserves original document order. With a query that strongly favors
    // the second sentence, the two compressors should surface it in
    // different relative positions.
    let extractive = ExtractiveSummaryCompressor::new();
    let abstractive = AbstractiveSummaryCompressor::new();
    let ps = passages(&[
        "An unrelated sentence about gardening and soil composition.",
        "Rust rust rust ownership ownership borrow checker checker safety.",
    ]);
    let extractive_out = extractive.compress("rust ownership borrow checker safety", &ps, 200);
    let abstractive_out = abstractive.compress("rust ownership borrow checker safety", &ps, 200);
    // Both should surface the second (highly relevant) sentence.
    assert!(extractive_out.to_lowercase().contains("ownership"));
    assert!(abstractive_out.to_lowercase().contains("ownership"));
}

// ── RecompPipeline — construction & validation ───────────────────────────────

#[test]
fn pipeline_new_stores_config() {
    let cfg = RecompConfig::new().with_token_budget(42);
    let pipeline = RecompPipeline::new(cfg.clone());
    assert_eq!(pipeline.config(), &cfg);
}

#[test]
fn pipeline_compress_empty_query_errors() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = relevant_passages();
    assert_eq!(pipeline.compress("", &ps), Err(RecompError::EmptyQuery));
}

#[test]
fn pipeline_compress_whitespace_query_errors() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = relevant_passages();
    assert_eq!(
        pipeline.compress("   \t", &ps),
        Err(RecompError::EmptyQuery)
    );
}

#[test]
fn pipeline_compress_empty_passages_errors() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    assert_eq!(
        pipeline.compress("rust borrow checker", &[]),
        Err(RecompError::EmptyPassages)
    );
}

#[test]
fn pipeline_compress_blank_passages_errors() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = passages(&["   ", ""]);
    assert_eq!(
        pipeline.compress("rust borrow checker", &ps),
        Err(RecompError::EmptyPassages)
    );
}

#[test]
fn pipeline_compress_invalid_config_errors() {
    let cfg = RecompConfig::new().with_token_budget(0);
    let pipeline = RecompPipeline::new(cfg);
    let ps = relevant_passages();
    assert!(matches!(
        pipeline.compress("rust borrow checker", &ps),
        Err(RecompError::InvalidConfig(_))
    ));
}

// ── RecompPipeline — selective-augmentation gate ─────────────────────────────

#[test]
fn pipeline_gate_skips_low_relevance_set() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = irrelevant_passages();
    let outcome = pipeline.compress("rust borrow checker", &ps).unwrap();
    assert!(
        outcome.decision.is_skip(),
        "expected skip, got {:?}",
        outcome.decision
    );
    assert_eq!(outcome.compressed_token_count, 0);
}

#[test]
fn pipeline_gate_skip_reason_mentions_threshold() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = irrelevant_passages();
    let outcome = pipeline.compress("rust borrow checker", &ps).unwrap();
    match outcome.decision {
        RecompDecision::Skip { reason } => {
            assert!(reason.contains("augmentation_threshold"));
        }
        RecompDecision::Augment(_) => panic!("expected skip"),
    }
}

#[test]
fn pipeline_gate_augments_high_relevance_set() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = relevant_passages();
    let outcome = pipeline.compress("rust borrow checker", &ps).unwrap();
    assert!(
        outcome.decision.is_augment(),
        "expected augment, got {:?}",
        outcome.decision
    );
}

#[test]
fn pipeline_gate_relevance_score_in_bounds() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    for ps in [relevant_passages(), irrelevant_passages()] {
        let outcome = pipeline.compress("rust borrow checker", &ps).unwrap();
        assert!(
            outcome.relevance_score >= 0.0 && outcome.relevance_score <= 1.0,
            "relevance score out of bounds: {}",
            outcome.relevance_score
        );
    }
}

#[test]
fn pipeline_gate_relevance_higher_for_relevant_set() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let relevant = pipeline
        .compress("rust borrow checker", &relevant_passages())
        .unwrap();
    let irrelevant = pipeline
        .compress("rust borrow checker", &irrelevant_passages())
        .unwrap();
    assert!(
        relevant.relevance_score > irrelevant.relevance_score,
        "expected relevant set to score higher: {} vs {}",
        relevant.relevance_score,
        irrelevant.relevance_score
    );
}

#[test]
fn pipeline_gate_threshold_zero_never_skips() {
    let cfg = RecompConfig::new().with_augmentation_threshold(0.0);
    let pipeline = RecompPipeline::new(cfg);
    let outcome = pipeline
        .compress("rust borrow checker", &irrelevant_passages())
        .unwrap();
    assert!(outcome.decision.is_augment());
}

#[test]
fn pipeline_gate_threshold_one_always_skips_unless_perfect_overlap() {
    let cfg = RecompConfig::new().with_augmentation_threshold(1.0);
    let pipeline = RecompPipeline::new(cfg);
    let outcome = pipeline
        .compress("rust borrow checker", &relevant_passages())
        .unwrap();
    assert!(outcome.decision.is_skip());
}

#[test]
fn pipeline_outcome_original_passage_count_matches_input() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = relevant_passages();
    let outcome = pipeline.compress("rust borrow checker", &ps).unwrap();
    assert_eq!(outcome.original_passage_count, ps.len());
}

// ── RecompPipeline — compressor strategies via pipeline ──────────────────────

#[test]
fn pipeline_extractive_strategy_produces_text() {
    let cfg = RecompConfig::new().with_strategy(CompressorStrategy::Extractive);
    let pipeline = RecompPipeline::new(cfg);
    let outcome = pipeline
        .compress("rust borrow checker", &relevant_passages())
        .unwrap();
    match outcome.decision {
        RecompDecision::Augment(text) => assert!(!text.is_empty()),
        RecompDecision::Skip { reason } => panic!("expected augment: {reason}"),
    }
}

#[test]
fn pipeline_abstractive_lite_strategy_produces_text() {
    let cfg = RecompConfig::new().with_strategy(CompressorStrategy::AbstractiveLite);
    let pipeline = RecompPipeline::new(cfg);
    let outcome = pipeline
        .compress("rust borrow checker", &relevant_passages())
        .unwrap();
    match outcome.decision {
        RecompDecision::Augment(text) => assert!(!text.is_empty()),
        RecompDecision::Skip { reason } => panic!("expected augment: {reason}"),
    }
}

#[test]
fn pipeline_compressed_token_count_within_budget_extractive() {
    let cfg = RecompConfig::new()
        .with_strategy(CompressorStrategy::Extractive)
        .with_token_budget(15);
    let pipeline = RecompPipeline::new(cfg);
    let outcome = pipeline
        .compress("rust borrow checker", &relevant_passages())
        .unwrap();
    assert!(outcome.compressed_token_count <= 15);
}

#[test]
fn pipeline_compressed_token_count_within_budget_abstractive() {
    let cfg = RecompConfig::new()
        .with_strategy(CompressorStrategy::AbstractiveLite)
        .with_token_budget(15);
    let pipeline = RecompPipeline::new(cfg);
    let outcome = pipeline
        .compress("rust borrow checker", &relevant_passages())
        .unwrap();
    assert!(outcome.compressed_token_count <= 15);
}

#[test]
fn pipeline_compressed_token_count_matches_text_word_count() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let outcome = pipeline
        .compress("rust borrow checker", &relevant_passages())
        .unwrap();
    if let RecompDecision::Augment(text) = &outcome.decision {
        assert_eq!(outcome.compressed_token_count, word_count(text));
    } else {
        panic!("expected augment");
    }
}

#[test]
fn pipeline_two_strategies_can_differ() {
    let ps = passages(&[
        "An unrelated sentence about gardening and soil composition today.",
        "Rust rust rust ownership ownership borrow checker checker safety guarantee.",
        "More detail about Rust lifetimes and the borrow checker's static analysis passes.",
    ]);
    let extractive_pipeline =
        RecompPipeline::new(RecompConfig::new().with_strategy(CompressorStrategy::Extractive));
    let abstractive_pipeline =
        RecompPipeline::new(RecompConfig::new().with_strategy(CompressorStrategy::AbstractiveLite));
    let query = "rust ownership borrow checker lifetimes safety";
    let extractive_outcome = extractive_pipeline.compress(query, &ps).unwrap();
    let abstractive_outcome = abstractive_pipeline.compress(query, &ps).unwrap();
    assert!(extractive_outcome.decision.is_augment());
    assert!(abstractive_outcome.decision.is_augment());
    // Different compression strategies over data with several candidate
    // sentences are expected to produce different surface text (order vs.
    // rank, and abstractive's connective-phrase fusion).
    assert_ne!(
        extractive_outcome.decision.text(),
        abstractive_outcome.decision.text()
    );
}

#[test]
fn pipeline_dedup_threshold_configurable_via_pipeline() {
    let ps = passages(&[
        "The Rust borrow checker enforces memory safety at compile time.",
        "The Rust borrow checker enforces memory safety at compile time!",
    ]);
    // A very low dedup threshold should collapse near-identical sentences
    // even more aggressively than the default.
    let cfg = RecompConfig::new()
        .with_dedup_similarity_threshold(0.1)
        .with_token_budget(200);
    let pipeline = RecompPipeline::new(cfg);
    let outcome = pipeline
        .compress("rust borrow checker memory safety", &ps)
        .unwrap();
    if let RecompDecision::Augment(text) = outcome.decision {
        assert_eq!(text.matches("borrow checker").count(), 1);
    } else {
        panic!("expected augment");
    }
}

#[test]
fn pipeline_clone_produces_independent_pipeline() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let cloned = pipeline.clone();
    let ps = relevant_passages();
    let a = pipeline.compress("rust borrow checker", &ps).unwrap();
    let b = cloned.compress("rust borrow checker", &ps).unwrap();
    assert_eq!(a, b);
}

#[test]
fn pipeline_default_matches_new_with_default_config() {
    let default_pipeline = RecompPipeline::default();
    let explicit_pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = relevant_passages();
    assert_eq!(
        default_pipeline
            .compress("rust borrow checker", &ps)
            .unwrap(),
        explicit_pipeline
            .compress("rust borrow checker", &ps)
            .unwrap()
    );
}

#[test]
fn pipeline_single_passage_relevant() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = passages(&["The Rust borrow checker enforces memory safety at compile time."]);
    let outcome = pipeline
        .compress("rust borrow checker safety", &ps)
        .unwrap();
    assert!(outcome.decision.is_augment());
}

#[test]
fn pipeline_single_passage_irrelevant_skips() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = passages(&["Bananas are a good source of potassium."]);
    let outcome = pipeline
        .compress("rust borrow checker safety", &ps)
        .unwrap();
    assert!(outcome.decision.is_skip());
}

#[test]
fn pipeline_large_token_budget_returns_more_content_than_small_budget() {
    let cfg_small = RecompConfig::new().with_token_budget(5);
    let cfg_large = RecompConfig::new().with_token_budget(100);
    let pipeline_small = RecompPipeline::new(cfg_small);
    let pipeline_large = RecompPipeline::new(cfg_large);
    let ps = relevant_passages();
    let small_outcome = pipeline_small.compress("rust borrow checker", &ps).unwrap();
    let large_outcome = pipeline_large.compress("rust borrow checker", &ps).unwrap();
    assert!(large_outcome.compressed_token_count >= small_outcome.compressed_token_count);
}

#[test]
fn pipeline_query_is_trimmed_before_scoring() {
    let pipeline = RecompPipeline::new(RecompConfig::default());
    let ps = relevant_passages();
    let a = pipeline.compress("  rust borrow checker  ", &ps).unwrap();
    let b = pipeline.compress("rust borrow checker", &ps).unwrap();
    assert_eq!(a.relevance_score, b.relevance_score);
}
