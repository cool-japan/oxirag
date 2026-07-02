#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]

//! Tests for the `selfcheckgpt` module.

use super::scorer::SelfCheckScorer;
use super::types::{SelfCheckConfig, SelfCheckError, SelfCheckVariant, SentenceCheck};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn samples_consistent() -> Vec<String> {
    vec![
        "Paris is the capital of France and a major cultural center.".to_string(),
        "The capital of France is Paris, known for its cultural heritage.".to_string(),
        "Paris, the capital city of France, is famous worldwide.".to_string(),
    ]
}

fn scorer_with(variant: SelfCheckVariant) -> SelfCheckScorer {
    SelfCheckScorer::new(SelfCheckConfig::new().with_variant(variant))
}

// ── SelfCheckVariant ──────────────────────────────────────────────────────────

#[test]
fn variant_default_is_ngram() {
    assert_eq!(SelfCheckVariant::default(), SelfCheckVariant::NGram);
}

#[test]
fn variant_as_str_ngram() {
    assert_eq!(SelfCheckVariant::NGram.as_str(), "ngram");
}

#[test]
fn variant_as_str_nli_lite() {
    assert_eq!(SelfCheckVariant::NliLite.as_str(), "nli_lite");
}

#[test]
fn variant_as_str_qa_lite() {
    assert_eq!(SelfCheckVariant::QaLite.as_str(), "qa_lite");
}

#[test]
fn variant_display_matches_as_str() {
    for variant in [
        SelfCheckVariant::NGram,
        SelfCheckVariant::NliLite,
        SelfCheckVariant::QaLite,
    ] {
        assert_eq!(format!("{variant}"), variant.as_str());
    }
}

#[test]
fn variant_equality() {
    assert_eq!(SelfCheckVariant::NGram, SelfCheckVariant::NGram);
    assert_ne!(SelfCheckVariant::NGram, SelfCheckVariant::NliLite);
}

#[test]
fn variant_copy_clone() {
    let v = SelfCheckVariant::QaLite;
    let v2 = v;
    assert_eq!(v, v2);
    assert_eq!(v.clone(), SelfCheckVariant::QaLite);
}

// ── SelfCheckConfig defaults & builders ────────────────────────────────────────

#[test]
fn config_default_variant() {
    assert_eq!(SelfCheckConfig::default().variant, SelfCheckVariant::NGram);
}

#[test]
fn config_default_ngram_size() {
    assert_eq!(SelfCheckConfig::default().ngram_size, 3);
}

#[test]
fn config_default_hallucination_threshold() {
    assert_eq!(SelfCheckConfig::default().hallucination_threshold, 0.5);
}

#[test]
fn config_default_min_samples() {
    assert_eq!(SelfCheckConfig::default().min_samples, 3);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SelfCheckConfig::new(), SelfCheckConfig::default());
}

#[test]
fn config_builder_variant() {
    let cfg = SelfCheckConfig::new().with_variant(SelfCheckVariant::NliLite);
    assert_eq!(cfg.variant, SelfCheckVariant::NliLite);
}

#[test]
fn config_builder_ngram_size() {
    let cfg = SelfCheckConfig::new().with_ngram_size(2);
    assert_eq!(cfg.ngram_size, 2);
}

#[test]
fn config_builder_hallucination_threshold() {
    let cfg = SelfCheckConfig::new().with_hallucination_threshold(0.7);
    assert_eq!(cfg.hallucination_threshold, 0.7);
}

#[test]
fn config_builder_min_samples() {
    let cfg = SelfCheckConfig::new().with_min_samples(5);
    assert_eq!(cfg.min_samples, 5);
}

#[test]
fn config_builders_are_independent() {
    let cfg = SelfCheckConfig::new().with_ngram_size(4);
    assert_eq!(cfg.hallucination_threshold, 0.5);
    assert_eq!(cfg.min_samples, 3);
}

#[test]
fn config_clone_equals_original() {
    let cfg = SelfCheckConfig::new().with_ngram_size(5);
    assert_eq!(cfg.clone(), cfg);
}

// ── SelfCheckConfig::validate ──────────────────────────────────────────────────

#[test]
fn validate_default_ok() {
    assert!(SelfCheckConfig::default().validate().is_ok());
}

#[test]
fn validate_zero_ngram_size_errors() {
    let cfg = SelfCheckConfig::new().with_ngram_size(0);
    assert!(matches!(
        cfg.validate(),
        Err(SelfCheckError::InvalidConfig(_))
    ));
}

#[test]
fn validate_negative_threshold_errors() {
    let cfg = SelfCheckConfig::new().with_hallucination_threshold(-0.1);
    assert!(matches!(
        cfg.validate(),
        Err(SelfCheckError::InvalidConfig(_))
    ));
}

#[test]
fn validate_threshold_above_one_errors() {
    let cfg = SelfCheckConfig::new().with_hallucination_threshold(1.1);
    assert!(matches!(
        cfg.validate(),
        Err(SelfCheckError::InvalidConfig(_))
    ));
}

#[test]
fn validate_nan_threshold_errors() {
    let cfg = SelfCheckConfig::new().with_hallucination_threshold(f32::NAN);
    assert!(matches!(
        cfg.validate(),
        Err(SelfCheckError::InvalidConfig(_))
    ));
}

#[test]
fn validate_zero_min_samples_errors() {
    let cfg = SelfCheckConfig::new().with_min_samples(0);
    assert!(matches!(
        cfg.validate(),
        Err(SelfCheckError::InvalidConfig(_))
    ));
}

#[test]
fn validate_threshold_boundary_zero_ok() {
    let cfg = SelfCheckConfig::new().with_hallucination_threshold(0.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_threshold_boundary_one_ok() {
    let cfg = SelfCheckConfig::new().with_hallucination_threshold(1.0);
    assert!(cfg.validate().is_ok());
}

// ── SelfCheckError ─────────────────────────────────────────────────────────────

#[test]
fn error_empty_response_display() {
    assert_eq!(
        SelfCheckError::EmptyResponse.to_string(),
        "main response is empty"
    );
}

#[test]
fn error_insufficient_samples_display() {
    let err = SelfCheckError::InsufficientSamples { got: 1, need: 3 };
    assert_eq!(
        err.to_string(),
        "insufficient samples: got 1, need at least 3"
    );
}

#[test]
fn error_invalid_config_display() {
    let err = SelfCheckError::InvalidConfig("bad".to_string());
    assert_eq!(err.to_string(), "invalid config: bad");
}

#[test]
fn error_equality() {
    assert_eq!(SelfCheckError::EmptyResponse, SelfCheckError::EmptyResponse);
    assert_eq!(
        SelfCheckError::InsufficientSamples { got: 1, need: 3 },
        SelfCheckError::InsufficientSamples { got: 1, need: 3 }
    );
    assert_ne!(
        SelfCheckError::InsufficientSamples { got: 1, need: 3 },
        SelfCheckError::InsufficientSamples { got: 2, need: 3 }
    );
}

#[test]
fn error_clone() {
    let err = SelfCheckError::InvalidConfig("x".to_string());
    assert_eq!(err.clone(), err);
}

// ── Scorer: empty-response error ───────────────────────────────────────────────

#[test]
fn score_empty_string_errors() {
    let scorer = SelfCheckScorer::default();
    let result = scorer.score("", &samples_consistent());
    assert_eq!(result, Err(SelfCheckError::EmptyResponse));
}

#[test]
fn score_whitespace_only_errors() {
    let scorer = SelfCheckScorer::default();
    let result = scorer.score("   \n\t  ", &samples_consistent());
    assert_eq!(result, Err(SelfCheckError::EmptyResponse));
}

// ── Scorer: insufficient-samples error ─────────────────────────────────────────

#[test]
fn score_insufficient_samples_errors() {
    let scorer = SelfCheckScorer::default();
    let samples = vec!["only one sample".to_string()];
    let result = scorer.score("Paris is the capital of France.", &samples);
    assert_eq!(
        result,
        Err(SelfCheckError::InsufficientSamples { got: 1, need: 3 })
    );
}

#[test]
fn score_zero_samples_errors() {
    let scorer = SelfCheckScorer::default();
    let result = scorer.score("Paris is the capital of France.", &[]);
    assert_eq!(
        result,
        Err(SelfCheckError::InsufficientSamples { got: 0, need: 3 })
    );
}

#[test]
fn score_exact_min_samples_ok() {
    let scorer = SelfCheckScorer::default();
    let result = scorer.score("Paris is the capital of France.", &samples_consistent());
    assert!(result.is_ok());
}

#[test]
fn score_custom_min_samples_respected() {
    let scorer = SelfCheckScorer::new(SelfCheckConfig::new().with_min_samples(5));
    let result = scorer.score("Paris is the capital of France.", &samples_consistent());
    assert_eq!(
        result,
        Err(SelfCheckError::InsufficientSamples { got: 3, need: 5 })
    );
}

// ── Scorer: invalid config propagation ─────────────────────────────────────────

#[test]
fn score_invalid_config_errors() {
    let scorer = SelfCheckScorer::new(SelfCheckConfig::new().with_ngram_size(0));
    let result = scorer.score("Paris is the capital of France.", &samples_consistent());
    assert!(matches!(result, Err(SelfCheckError::InvalidConfig(_))));
}

// ── Scorer: sentence splitting ─────────────────────────────────────────────────

#[test]
fn score_splits_multiple_sentences() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. It has a famous tower.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert_eq!(result.sentence_checks.len(), 2);
}

#[test]
fn score_single_sentence_no_terminal_punctuation() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert_eq!(result.sentence_checks.len(), 1);
    assert_eq!(
        result.sentence_checks[0].sentence,
        "Paris is the capital of France"
    );
}

#[test]
fn score_sentence_text_preserved() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert_eq!(
        result.sentence_checks[0].sentence,
        "Paris is the capital of France."
    );
}

// ── Scorer: overall_score & variant_used ───────────────────────────────────────

#[test]
fn overall_score_is_mean_of_sentence_scores() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    let mean: f32 = result
        .sentence_checks
        .iter()
        .map(|c| c.inconsistency_score)
        .sum::<f32>()
        / result.sentence_checks.len() as f32;
    assert!((result.overall_score - mean).abs() < 1e-6);
}

#[test]
fn variant_used_matches_config_ngram() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let result = scorer
        .score("Paris is the capital of France.", &samples_consistent())
        .unwrap();
    assert_eq!(result.variant_used, SelfCheckVariant::NGram);
}

#[test]
fn variant_used_matches_config_nli_lite() {
    let scorer = scorer_with(SelfCheckVariant::NliLite);
    let result = scorer
        .score("Paris is the capital of France.", &samples_consistent())
        .unwrap();
    assert_eq!(result.variant_used, SelfCheckVariant::NliLite);
}

#[test]
fn variant_used_matches_config_qa_lite() {
    let scorer = scorer_with(SelfCheckVariant::QaLite);
    let result = scorer
        .score("Paris is the capital of France.", &samples_consistent())
        .unwrap();
    assert_eq!(result.variant_used, SelfCheckVariant::QaLite);
}

// ── Scorer: all scores bounded to [0, 1] ───────────────────────────────────────

#[test]
fn ngram_scores_bounded() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    for check in &result.sentence_checks {
        assert!((0.0..=1.0).contains(&check.inconsistency_score));
    }
}

#[test]
fn nli_lite_scores_bounded() {
    let scorer = scorer_with(SelfCheckVariant::NliLite);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    for check in &result.sentence_checks {
        assert!((0.0..=1.0).contains(&check.inconsistency_score));
    }
}

#[test]
fn qa_lite_scores_bounded() {
    let scorer = scorer_with(SelfCheckVariant::QaLite);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    for check in &result.sentence_checks {
        assert!((0.0..=1.0).contains(&check.inconsistency_score));
    }
}

// ── NGram variant: direction of the signal ─────────────────────────────────────

#[test]
fn ngram_consistent_sentence_scores_lower_than_fabricated() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert!(
        result.sentence_checks[0].inconsistency_score
            < result.sentence_checks[1].inconsistency_score,
        "consistent={}, fabricated={}",
        result.sentence_checks[0].inconsistency_score,
        result.sentence_checks[1].inconsistency_score
    );
}

#[test]
fn ngram_fabricated_sentence_flagged_hallucination() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert!(result.sentence_checks[1].is_hallucination);
}

#[test]
fn ngram_verbatim_repeated_sentence_scores_low() {
    // A sentence that is an exact verbatim match to every sample should have
    // a low inconsistency score.
    let samples = vec![
        "The Eiffel Tower is located in Paris.".to_string(),
        "The Eiffel Tower is located in Paris.".to_string(),
        "The Eiffel Tower is located in Paris.".to_string(),
    ];
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let result = scorer
        .score("The Eiffel Tower is located in Paris.", &samples)
        .unwrap();
    assert!(result.sentence_checks[0].inconsistency_score < 0.3);
}

#[test]
fn ngram_wholly_unique_sentence_scores_high() {
    let samples = vec![
        "The Eiffel Tower is located in Paris.".to_string(),
        "Big Ben is located in London.".to_string(),
        "The Colosseum is located in Rome.".to_string(),
    ];
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let result = scorer
        .score("Quantum dolphins orbit the moon nightly.", &samples)
        .unwrap();
    assert!(result.sentence_checks[0].inconsistency_score > 0.5);
}

#[test]
fn ngram_respects_custom_ngram_size() {
    let scorer = SelfCheckScorer::new(
        SelfCheckConfig::new()
            .with_variant(SelfCheckVariant::NGram)
            .with_ngram_size(1),
    );
    let result = scorer
        .score("Paris is the capital of France.", &samples_consistent())
        .unwrap();
    assert!((0.0..=1.0).contains(&result.sentence_checks[0].inconsistency_score));
}

#[test]
fn ngram_short_sentence_below_ngram_size_neutral() {
    // A single-token sentence cannot form a trigram; expect the documented
    // neutral fallback of 0.5 rather than a fabricated extreme value.
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let result = scorer.score("Yes.", &samples_consistent()).unwrap();
    assert_eq!(result.sentence_checks[0].inconsistency_score, 0.5);
}

// ── NliLite variant: direction of the signal ───────────────────────────────────

#[test]
fn nli_lite_consistent_sentence_scores_lower_than_fabricated() {
    let scorer = scorer_with(SelfCheckVariant::NliLite);
    let main = "Paris is the capital of France. Quantum dolphins orbit the moon nightly.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert!(
        result.sentence_checks[0].inconsistency_score
            < result.sentence_checks[1].inconsistency_score,
        "consistent={}, fabricated={}",
        result.sentence_checks[0].inconsistency_score,
        result.sentence_checks[1].inconsistency_score
    );
}

#[test]
fn nli_lite_fully_corroborated_sentence_scores_zero() {
    let samples = vec![
        "The Eiffel Tower is located in Paris.".to_string(),
        "The Eiffel Tower is located in Paris.".to_string(),
        "The Eiffel Tower is located in Paris.".to_string(),
    ];
    let scorer = scorer_with(SelfCheckVariant::NliLite);
    let result = scorer
        .score("The Eiffel Tower is located in Paris.", &samples)
        .unwrap();
    assert_eq!(result.sentence_checks[0].inconsistency_score, 0.0);
}

#[test]
fn nli_lite_fully_unsupported_sentence_scores_one() {
    let samples = vec![
        "The Eiffel Tower is located in Paris.".to_string(),
        "Big Ben is located in London.".to_string(),
        "The Colosseum is located in Rome.".to_string(),
    ];
    let scorer = scorer_with(SelfCheckVariant::NliLite);
    let result = scorer
        .score("Quantum dolphins orbit the moon nightly.", &samples)
        .unwrap();
    assert_eq!(result.sentence_checks[0].inconsistency_score, 1.0);
}

// ── QaLite variant: direction of the signal ────────────────────────────────────

#[test]
fn qa_lite_consistent_sentence_scores_lower_than_fabricated() {
    let scorer = scorer_with(SelfCheckVariant::QaLite);
    let main = "The capital of France is Paris. The capital of France is Atlantis.";
    let samples = vec![
        "The capital of France is Paris.".to_string(),
        "France's capital city is Paris.".to_string(),
        "Paris serves as the capital of France.".to_string(),
    ];
    let result = scorer.score(main, &samples).unwrap();
    assert!(
        result.sentence_checks[0].inconsistency_score
            < result.sentence_checks[1].inconsistency_score,
        "consistent={}, fabricated={}",
        result.sentence_checks[0].inconsistency_score,
        result.sentence_checks[1].inconsistency_score
    );
}

#[test]
fn qa_lite_keyword_present_in_all_samples_scores_zero() {
    let samples = vec![
        "The capital of France is Paris.".to_string(),
        "France's capital city is Paris.".to_string(),
        "Paris serves as the capital of France.".to_string(),
    ];
    let scorer = scorer_with(SelfCheckVariant::QaLite);
    let result = scorer
        .score("The capital of France is Paris.", &samples)
        .unwrap();
    assert_eq!(result.sentence_checks[0].inconsistency_score, 0.0);
}

#[test]
fn qa_lite_keyword_absent_from_all_samples_scores_one() {
    let samples = vec![
        "The capital of France is Paris.".to_string(),
        "France's capital city is Paris.".to_string(),
        "Paris serves as the capital of France.".to_string(),
    ];
    let scorer = scorer_with(SelfCheckVariant::QaLite);
    let result = scorer
        .score("The capital of France is Atlantis.", &samples)
        .unwrap();
    assert_eq!(result.sentence_checks[0].inconsistency_score, 1.0);
}

#[test]
fn qa_lite_all_function_words_neutral() {
    let scorer = scorer_with(SelfCheckVariant::QaLite);
    // "It is." tokenises to ["it", "is"], both function words: no keyword.
    let result = scorer.score("It is.", &samples_consistent()).unwrap();
    assert_eq!(result.sentence_checks[0].inconsistency_score, 0.5);
}

// ── Threshold boundary flagging ────────────────────────────────────────────────

#[test]
fn threshold_boundary_flags_when_score_equals_threshold() {
    // Craft 4 samples where exactly half agree on the keyword, giving a QA-lite
    // score of exactly 0.5, matching the default threshold via `>=`.
    let samples = vec![
        "The Eiffel Tower is a landmark.".to_string(),
        "The Eiffel Tower is a landmark.".to_string(),
        "A famous landmark stands in the city.".to_string(),
        "A famous landmark stands in the city.".to_string(),
    ];
    let scorer =
        SelfCheckScorer::new(SelfCheckConfig::new().with_variant(SelfCheckVariant::QaLite));
    let result = scorer.score("The Eiffel Tower is here.", &samples).unwrap();
    assert_eq!(result.sentence_checks[0].inconsistency_score, 0.5);
    assert!(result.sentence_checks[0].is_hallucination);
}

#[test]
fn threshold_boundary_below_threshold_not_flagged() {
    let samples = vec![
        "The Eiffel Tower is a landmark.".to_string(),
        "The Eiffel Tower is a landmark.".to_string(),
        "The Eiffel Tower is a landmark.".to_string(),
        "A famous landmark stands in the city.".to_string(),
    ];
    let scorer =
        SelfCheckScorer::new(SelfCheckConfig::new().with_variant(SelfCheckVariant::QaLite));
    let result = scorer.score("The Eiffel Tower is here.", &samples).unwrap();
    assert_eq!(result.sentence_checks[0].inconsistency_score, 0.25);
    assert!(!result.sentence_checks[0].is_hallucination);
}

#[test]
fn threshold_custom_low_flags_more() {
    let scorer = SelfCheckScorer::new(
        SelfCheckConfig::new()
            .with_variant(SelfCheckVariant::QaLite)
            .with_hallucination_threshold(0.1),
    );
    let samples = vec![
        "The Eiffel Tower is a landmark.".to_string(),
        "The Eiffel Tower is a landmark.".to_string(),
        "The Eiffel Tower is a landmark.".to_string(),
        "A famous landmark stands in the city.".to_string(),
    ];
    let result = scorer.score("The Eiffel Tower is here.", &samples).unwrap();
    // 0.25 >= 0.1 -> flagged.
    assert!(result.sentence_checks[0].is_hallucination);
}

#[test]
fn threshold_custom_high_flags_fewer() {
    let scorer = SelfCheckScorer::new(
        SelfCheckConfig::new()
            .with_variant(SelfCheckVariant::QaLite)
            .with_hallucination_threshold(0.9),
    );
    let samples = vec![
        "The Eiffel Tower is a landmark.".to_string(),
        "The Eiffel Tower is a landmark.".to_string(),
        "The Eiffel Tower is a landmark.".to_string(),
        "A famous landmark stands in the city.".to_string(),
    ];
    let result = scorer.score("The Eiffel Tower is here.", &samples).unwrap();
    // 0.25 < 0.9 -> not flagged.
    assert!(!result.sentence_checks[0].is_hallucination);
}

// ── Determinism ────────────────────────────────────────────────────────────────

#[test]
fn ngram_scoring_is_deterministic() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let r1 = scorer.score(main, &samples_consistent()).unwrap();
    let r2 = scorer.score(main, &samples_consistent()).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn nli_lite_scoring_is_deterministic() {
    let scorer = scorer_with(SelfCheckVariant::NliLite);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let r1 = scorer.score(main, &samples_consistent()).unwrap();
    let r2 = scorer.score(main, &samples_consistent()).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn qa_lite_scoring_is_deterministic() {
    let scorer = scorer_with(SelfCheckVariant::QaLite);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let r1 = scorer.score(main, &samples_consistent()).unwrap();
    let r2 = scorer.score(main, &samples_consistent()).unwrap();
    assert_eq!(r1, r2);
}

// ── SelfCheckScore helper methods ───────────────────────────────────────────────

#[test]
fn hallucination_count_counts_flagged_sentences() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert_eq!(result.hallucination_count(), 1);
}

#[test]
fn is_clean_true_when_no_hallucinations() {
    let samples = vec![
        "The Eiffel Tower is located in Paris.".to_string(),
        "The Eiffel Tower is located in Paris.".to_string(),
        "The Eiffel Tower is located in Paris.".to_string(),
    ];
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let result = scorer
        .score("The Eiffel Tower is located in Paris.", &samples)
        .unwrap();
    assert!(result.is_clean());
}

#[test]
fn is_clean_false_when_hallucinations_present() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert!(!result.is_clean());
}

// ── SentenceCheck field accessibility & construction ───────────────────────────

#[test]
fn sentence_check_fields_accessible() {
    let check = SentenceCheck {
        sentence: "hello".to_string(),
        inconsistency_score: 0.42,
        is_hallucination: false,
    };
    assert_eq!(check.sentence, "hello");
    assert_eq!(check.inconsistency_score, 0.42);
    assert!(!check.is_hallucination);
}

#[test]
fn sentence_check_clone_eq() {
    let check = SentenceCheck {
        sentence: "hello".to_string(),
        inconsistency_score: 0.42,
        is_hallucination: false,
    };
    assert_eq!(check.clone(), check);
}

// ── SelfCheckScorer defaults & config field access ─────────────────────────────

#[test]
fn scorer_default_uses_default_config() {
    let scorer = SelfCheckScorer::default();
    assert_eq!(scorer.config, SelfCheckConfig::default());
}

#[test]
fn scorer_new_preserves_config() {
    let cfg = SelfCheckConfig::new().with_min_samples(7);
    let scorer = SelfCheckScorer::new(cfg.clone());
    assert_eq!(scorer.config, cfg);
}

#[test]
fn scorer_clone_identical_results() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let scorer2 = scorer.clone();
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200.";
    let r1 = scorer.score(main, &samples_consistent()).unwrap();
    let r2 = scorer2.score(main, &samples_consistent()).unwrap();
    assert_eq!(r1, r2);
}

// ── Multi-sentence document with a mix of scores ───────────────────────────────

#[test]
fn multi_sentence_document_mixed_hallucination_flags() {
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let main = "Paris is the capital of France. Paris was founded by unicorns in 1200. \
                Paris is a major cultural center of France.";
    let result = scorer.score(main, &samples_consistent()).unwrap();
    assert_eq!(result.sentence_checks.len(), 3);
    assert!(!result.sentence_checks[0].is_hallucination);
    assert!(result.sentence_checks[1].is_hallucination);
}

// ── Longer sample sets ──────────────────────────────────────────────────────────

#[test]
fn works_with_more_than_min_samples() {
    let mut samples = samples_consistent();
    samples.push("Paris is famous for the Louvre museum.".to_string());
    samples.push("Many tourists visit Paris every year.".to_string());
    let scorer = scorer_with(SelfCheckVariant::NGram);
    let result = scorer
        .score("Paris is the capital of France.", &samples)
        .unwrap();
    assert_eq!(result.sentence_checks.len(), 1);
}
