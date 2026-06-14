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

use crate::faithfulness_eval::evaluator::FaithfulnessEvaluator;
use crate::faithfulness_eval::types::{
    ClaimEntailment, FaithfulnessConfig, FaithfulnessError, FaithfulnessScore,
};

/// Absolute tolerance used in floating-point assertions.
const EPS: f32 = 1e-5;

// ── FaithfulnessConfig ────────────────────────────────────────────────────────

#[test]
fn config_default_min_entailment_score_is_0_5() {
    let cfg = FaithfulnessConfig::default();
    assert!((cfg.min_entailment_score - 0.5).abs() < EPS);
}

#[test]
fn config_default_claim_split_tokens_are_four_punctuation_marks() {
    let cfg = FaithfulnessConfig::default();
    let tokens = &cfg.claim_split_tokens;
    assert_eq!(tokens.len(), 4);
    assert!(tokens.contains(&".".to_owned()));
    assert!(tokens.contains(&"!".to_owned()));
    assert!(tokens.contains(&"?".to_owned()));
    assert!(tokens.contains(&";".to_owned()));
}

#[test]
fn config_new_equals_default() {
    assert_eq!(FaithfulnessConfig::new(), FaithfulnessConfig::default());
}

#[test]
fn config_builder_with_min_entailment_score() {
    let cfg = FaithfulnessConfig::new().with_min_entailment_score(0.8);
    assert!((cfg.min_entailment_score - 0.8).abs() < EPS);
}

#[test]
fn config_builder_with_claim_split_tokens() {
    let cfg =
        FaithfulnessConfig::new().with_claim_split_tokens(vec!["|".to_owned(), ",".to_owned()]);
    assert_eq!(cfg.claim_split_tokens, vec!["|", ","]);
}

#[test]
fn config_is_clone() {
    let original = FaithfulnessConfig::new().with_min_entailment_score(0.7);
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn config_is_debug() {
    let cfg = FaithfulnessConfig::default();
    let repr = format!("{cfg:?}");
    assert!(repr.contains("FaithfulnessConfig"));
}

#[test]
fn config_split_tokens_count_is_four_by_default() {
    let cfg = FaithfulnessConfig::default();
    assert_eq!(cfg.claim_split_tokens.len(), 4);
}

// ── FaithfulnessError ─────────────────────────────────────────────────────────

#[test]
fn error_empty_answer_display() {
    assert_eq!(
        FaithfulnessError::EmptyAnswer.to_string(),
        "answer is empty"
    );
}

#[test]
fn error_empty_context_display() {
    assert_eq!(
        FaithfulnessError::EmptyContext.to_string(),
        "context is empty"
    );
}

#[test]
fn error_evaluation_failed_display_includes_message() {
    let err = FaithfulnessError::EvaluationFailed("bad input".to_owned());
    assert_eq!(err.to_string(), "evaluation failed: bad input");
}

#[test]
fn error_empty_answer_is_partial_eq() {
    assert_eq!(
        FaithfulnessError::EmptyAnswer,
        FaithfulnessError::EmptyAnswer
    );
}

#[test]
fn error_clone_is_equal() {
    let err = FaithfulnessError::EvaluationFailed("oops".to_owned());
    assert_eq!(err.clone(), err);
}

#[test]
fn error_is_debug() {
    let err = FaithfulnessError::EmptyContext;
    let repr = format!("{err:?}");
    assert!(repr.contains("EmptyContext"));
}

// ── ClaimEntailment and FaithfulnessScore structs ────────────────────────────

#[test]
fn claim_entailment_fields_are_accessible() {
    let ce = ClaimEntailment {
        claim: "test claim".to_owned(),
        entailment_score: 0.75,
        is_entailed: true,
        supporting_passage: Some("some passage".to_owned()),
    };
    assert_eq!(ce.claim, "test claim");
    assert!((ce.entailment_score - 0.75).abs() < EPS);
    assert!(ce.is_entailed);
    assert_eq!(ce.supporting_passage.as_deref(), Some("some passage"));
}

#[test]
fn faithfulness_score_fields_are_accessible() {
    let fs = FaithfulnessScore {
        answer: "my answer".to_owned(),
        claims: vec![],
        faithfulness: 1.0,
        entailed_count: 0,
        total_claims: 0,
    };
    assert_eq!(fs.answer, "my answer");
    assert!((fs.faithfulness - 1.0).abs() < EPS);
    assert_eq!(fs.entailed_count, 0);
    assert_eq!(fs.total_claims, 0);
}

#[test]
fn claim_entailment_clone_and_debug() {
    let ce = ClaimEntailment {
        claim: "hello world".to_owned(),
        entailment_score: 0.5,
        is_entailed: true,
        supporting_passage: None,
    };
    let cloned = ce.clone();
    assert_eq!(ce, cloned);
    let repr = format!("{ce:?}");
    assert!(repr.contains("ClaimEntailment"));
}

// ── split_into_claims: basic splitting ───────────────────────────────────────

#[test]
fn claim_split_single_sentence_no_separator_yields_one_claim() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("The sky is blue today");
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0], "The sky is blue today");
}

#[test]
fn claim_split_period_separator_yields_two_claims() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("The sky is blue. Water is wet.");
    assert_eq!(claims.len(), 2);
}

#[test]
fn claim_split_semicolon_separator_works() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("First claim; Second claim here");
    assert_eq!(claims.len(), 2);
}

#[test]
fn claim_split_exclamation_separator_works() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("Amazing result! Another finding here");
    assert_eq!(claims.len(), 2);
}

#[test]
fn claim_split_question_mark_separator_works() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("Is the sky blue? It appears so today");
    assert_eq!(claims.len(), 2);
}

#[test]
fn claim_split_mixed_separators_work() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("First sentence. Second one! Third here? Fourth now");
    assert_eq!(claims.len(), 4);
}

// ── split_into_claims: filtering ─────────────────────────────────────────────

#[test]
fn claim_split_short_single_word_fragment_filtered() {
    // "OK" alone has only 1 token → filtered.
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("OK. The sky is blue today.");
    // "OK" has 1 token (filtered); "The sky is blue today" has 5 (kept).
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0], "The sky is blue today");
}

#[test]
fn claim_split_empty_answer_yields_empty_vec() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("");
    assert!(claims.is_empty());
}

#[test]
fn claim_split_only_separators_yields_empty_vec() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("...!?;");
    assert!(claims.is_empty());
}

#[test]
fn claim_split_trims_leading_trailing_whitespace() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator.split_into_claims("Hello world. Foo bar.");
    for claim in &claims {
        assert_eq!(claim.as_str(), claim.trim());
    }
}

#[test]
fn claim_split_custom_separator_works() {
    let cfg = FaithfulnessConfig::new().with_claim_split_tokens(vec!["|".to_owned()]);
    let evaluator = FaithfulnessEvaluator::new(cfg);
    let claims = evaluator.split_into_claims("First claim | Second claim here");
    assert_eq!(claims.len(), 2);
}

// ── entailment_score: core scoring ───────────────────────────────────────────

#[test]
fn entailment_score_perfect_overlap_is_one() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // Claim tokens: {dogs, bark, loud}; passage contains all three.
    let (score, _) = evaluator.entailment_score("dogs bark loud", &["dogs bark loud in the park"]);
    assert!((score - 1.0).abs() < EPS);
}

#[test]
fn entailment_score_zero_overlap_is_zero() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // Completely disjoint vocabularies.
    let (score, passage) = evaluator.entailment_score(
        "quantum physics explains atoms",
        &["horses gallop through meadows swiftly"],
    );
    assert!((score - 0.0).abs() < EPS);
    assert!(passage.is_none());
}

#[test]
fn entailment_score_partial_overlap_between_zero_and_one() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // claim tokens = {the, sky, turns, crimson}; passage has {the, sky} but not {turns, crimson}.
    let (score, _) = evaluator.entailment_score(
        "the sky turns crimson",
        &["the sky is typically blue in daytime"],
    );
    assert!(score > 0.0);
    assert!(score < 1.0);
}

#[test]
fn entailment_score_empty_passages_returns_zero_and_none() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let (score, passage) = evaluator.entailment_score("dogs bark loud", &[]);
    assert!((score - 0.0).abs() < EPS);
    assert!(passage.is_none());
}

#[test]
fn entailment_score_supporting_passage_returned_on_overlap() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let (score, passage) = evaluator.entailment_score(
        "Paris is the capital",
        &["Paris serves as the capital city of France"],
    );
    assert!(score > 0.0);
    assert!(passage.is_some());
}

#[test]
fn entailment_score_no_supporting_passage_when_zero_overlap() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let (_, passage) =
        evaluator.entailment_score("plutonium is nutritious", &["elephants roam vast savannas"]);
    assert!(passage.is_none());
}

#[test]
fn entailment_score_best_passage_selected_from_multiple() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // claim tokens: {rust, is, systems, programming, language}  ("a" filtered: len 1)
    let claim = "rust is systems programming language";
    let passages: &[&str] = &[
        "java is an object oriented language",
        "rust is fast and safe systems language",
        "fish swim underwater silently",
    ];
    let (score, passage) = evaluator.entailment_score(claim, passages);
    // Passage 2 has more claim tokens than passage 1 or 3.
    assert!(score > 0.0);
    let supporting = passage.expect("best passage should be Some");
    assert!(supporting.contains("rust") || supporting.contains("systems"));
}

#[test]
fn entailment_score_is_case_insensitive() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // Tokenizer lowercases both claim and passage, so case must not matter.
    let (score_lower, _) = evaluator.entailment_score(
        "water boils at high temperature",
        &["Water Boils At High Temperature always"],
    );
    let (score_upper, _) = evaluator.entailment_score(
        "WATER BOILS AT HIGH TEMPERATURE",
        &["water boils at high temperature always"],
    );
    assert!((score_lower - score_upper).abs() < EPS);
}

// ── evaluate(): faithful answers ─────────────────────────────────────────────

#[test]
fn evaluate_fully_faithful_answer_scores_one() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The sun provides light. Plants use sunlight.";
    let context = vec![
        "The sun is our nearest star that provides light. Plants use sunlight for photosynthesis."
            .to_owned(),
    ];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert!((result.faithfulness - 1.0).abs() < EPS);
    assert_eq!(result.entailed_count, result.total_claims);
}

#[test]
fn evaluate_partial_faithfulness_returns_correct_ratio() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // First claim entailed; second has no shared vocabulary with context.
    let answer = "The moon orbits Earth. Bananas radioactive metals explode.";
    let context =
        vec!["The moon is Earth's natural satellite and orbits Earth monthly.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, 2);
    assert_eq!(result.entailed_count, 1);
    assert!((result.faithfulness - 0.5).abs() < EPS);
}

#[test]
fn evaluate_fully_hallucinated_answer_scores_zero() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // Completely different vocabulary between answer and context.
    let answer = "Bananas radioactive metals explode. Computers spontaneously levitate.";
    let context =
        vec!["Fruit trees grow in orchards. Gardens need careful pruning seasonally.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert!((result.faithfulness - 0.0).abs() < EPS);
    assert_eq!(result.entailed_count, 0);
}

#[test]
fn evaluate_single_faithful_claim_scores_one() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The ocean covers most of the Earth surface.";
    let context = vec!["The ocean covers most of the Earth surface area today.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, 1);
    assert_eq!(result.entailed_count, 1);
    assert!((result.faithfulness - 1.0).abs() < EPS);
}

#[test]
fn evaluate_single_hallucinated_claim_scores_zero() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "Plutonium edible nutritious daily vitamins.";
    let context = vec!["Nuclear plants convert uranium into electricity safely.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, 1);
    assert_eq!(result.entailed_count, 0);
    assert!((result.faithfulness - 0.0).abs() < EPS);
}

#[test]
fn evaluate_multiple_passages_uses_best_matching() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "Rust is systems programming language.";
    let context = vec![
        "Java is an object oriented language.".to_owned(),
        "Rust is fast safe systems language.".to_owned(),
        "Fish swim underwater silently.".to_owned(),
    ];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, 1);
    assert_eq!(result.entailed_count, 1);
    assert!((result.faithfulness - 1.0).abs() < EPS);
}

// ── evaluate(): structural correctness ───────────────────────────────────────

#[test]
fn evaluate_entailed_count_matches_is_entailed_flags() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The sky is blue. Water is wet. Cats purr softly.";
    let context = vec!["The sky appears blue. Water is wet. Cats purr when content.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    let manual_count = result.claims.iter().filter(|c| c.is_entailed).count();
    assert_eq!(result.entailed_count, manual_count);
}

#[test]
fn evaluate_total_claims_matches_claims_vec_length() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "First fact here. Second fact here. Third fact here.";
    let context = vec!["First fact here. Second fact here.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, result.claims.len());
}

#[test]
fn evaluate_faithfulness_equals_entailed_over_total() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "Alpha beta gamma. Delta epsilon zeta.";
    let context = vec!["Alpha beta gamma works well here.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    let expected = result.entailed_count as f32 / result.total_claims as f32;
    assert!((result.faithfulness - expected).abs() < EPS);
}

#[test]
fn evaluate_answer_field_preserved_in_result() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The sky is blue. Water is wet.";
    let context = vec!["The sky is blue. Water is wet.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.answer, answer);
}

#[test]
fn evaluate_entailed_claim_has_supporting_passage() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The sun provides warmth energy.";
    let context = vec!["The sun provides warmth and energy to Earth.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, 1);
    let claim = &result.claims[0];
    assert!(claim.is_entailed);
    assert!(claim.supporting_passage.is_some());
}

#[test]
fn evaluate_hallucinated_claim_has_no_supporting_passage() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // Completely disjoint vocabulary.
    let answer = "Penguins combustion ignite spontaneously.";
    let context = vec!["Butterflies migrate southward during autumn.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, 1);
    let claim = &result.claims[0];
    assert!(!claim.is_entailed);
    assert!(claim.supporting_passage.is_none());
}

#[test]
fn evaluate_per_claim_entailment_score_in_range() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The sky is blue. Bananas spontaneously combust magically.";
    let context = vec!["The sky is typically blue during daytime hours.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    for claim in &result.claims {
        assert!(claim.entailment_score >= 0.0);
        assert!(claim.entailment_score <= 1.0);
    }
}

#[test]
fn evaluate_claims_vec_not_empty_for_valid_answer() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The moon orbits Earth every month.";
    let context = vec!["The moon orbits Earth monthly in a regular cycle.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert!(!result.claims.is_empty());
}

// ── evaluate(): error conditions ─────────────────────────────────────────────

#[test]
fn evaluate_empty_answer_returns_empty_answer_error() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let err = evaluator
        .evaluate("", &vec!["some context".to_owned()])
        .unwrap_err();
    assert_eq!(err, FaithfulnessError::EmptyAnswer);
}

#[test]
fn evaluate_whitespace_only_answer_returns_empty_answer_error() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let err = evaluator
        .evaluate("   \t\n  ", &vec!["some context".to_owned()])
        .unwrap_err();
    assert_eq!(err, FaithfulnessError::EmptyAnswer);
}

#[test]
fn evaluate_empty_context_returns_empty_context_error() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let err = evaluator.evaluate("The sky is blue.", &[]).unwrap_err();
    assert_eq!(err, FaithfulnessError::EmptyContext);
}

#[test]
fn evaluate_empty_answer_takes_precedence_over_empty_context() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let err = evaluator.evaluate("", &[]).unwrap_err();
    assert_eq!(err, FaithfulnessError::EmptyAnswer);
}

// ── evaluate(): threshold and configuration effects ───────────────────────────

#[test]
fn evaluate_high_threshold_reduces_entailed_count() {
    // With threshold 0.5 several claims are entailed; with 0.99 none are.
    let answer = "Paris is famous worldwide. London is large. Tokyo is crowded.";
    let context = vec!["Paris is a city in France. London is located in England.".to_owned()];

    let low_cfg = FaithfulnessConfig::new().with_min_entailment_score(0.3);
    let high_cfg = FaithfulnessConfig::new().with_min_entailment_score(0.99);

    let low_eval = FaithfulnessEvaluator::new(low_cfg);
    let high_eval = FaithfulnessEvaluator::new(high_cfg);

    let low_result = low_eval.evaluate(answer, &context).unwrap();
    let high_result = high_eval.evaluate(answer, &context).unwrap();

    // More claims must be entailed at the lower threshold.
    assert!(low_result.entailed_count >= high_result.entailed_count);
}

#[test]
fn evaluate_low_threshold_increases_entailed_count() {
    let answer = "The sky turns orange. Leaves change colour. Wind blows gently.";
    let context = vec!["The sky turns orange at dusk. Autumn leaves change colour.".to_owned()];

    let threshold_low = FaithfulnessConfig::new().with_min_entailment_score(0.1);
    let threshold_high = FaithfulnessConfig::new().with_min_entailment_score(0.9);

    let low_result = FaithfulnessEvaluator::new(threshold_low)
        .evaluate(answer, &context)
        .unwrap();
    let high_result = FaithfulnessEvaluator::new(threshold_high)
        .evaluate(answer, &context)
        .unwrap();

    assert!(low_result.entailed_count >= high_result.entailed_count);
}

#[test]
fn evaluate_threshold_at_exact_score_is_entailed() {
    // Build a claim with exactly 2/4 = 0.5 coverage so threshold boundary works.
    // claim = "word one more test" → tokens {word, one, more, test}
    // passage = "word one data info" → tokens {word, one, data, info}
    // intersection = {word, one} → 2/4 = 0.5
    let cfg_05 = FaithfulnessConfig::new().with_min_entailment_score(0.5);
    let evaluator = FaithfulnessEvaluator::new(cfg_05);
    let (score, _) = evaluator.entailment_score(
        "word one more test",
        &["word one data info facts stuff here"],
    );
    // Coverage should be 0.5 exactly (2 of 4 tokens matched).
    assert!((score - 0.5).abs() < EPS);
    // Is_entailed = score >= threshold → 0.5 >= 0.5 → true.
    let context = vec!["word one data info facts stuff here".to_owned()];
    let result = evaluator.evaluate("word one more test", &context).unwrap();
    assert!(result.claims[0].is_entailed);
}

#[test]
fn evaluate_no_claims_after_filtering_faithfulness_is_zero() {
    // An answer composed only of punctuation yields zero claims → faithfulness = 0.0.
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let context = vec!["some context passage text here".to_owned()];
    // "...!?;" splits at each char; each fragment is empty → 0 token-count → filtered.
    let result = evaluator.evaluate("...!?;", &context).unwrap();
    assert_eq!(result.total_claims, 0);
    assert!((result.faithfulness - 0.0).abs() < EPS);
}

#[test]
fn evaluate_is_deterministic() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The sky is blue. Water is wet. Cats purr softly.";
    let context = vec!["The sky is blue. Water is wet. Cats purr.".to_owned()];
    let first = evaluator.evaluate(answer, &context).unwrap();
    let second = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(first.faithfulness, second.faithfulness);
    assert_eq!(first.entailed_count, second.entailed_count);
    assert_eq!(first.total_claims, second.total_claims);
}

// ── per-claim field verification ──────────────────────────────────────────────

#[test]
fn evaluate_claim_entailment_score_matches_entailment_score_method() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "The moon orbits Earth every month.";
    let context = vec!["The moon orbits Earth monthly in a regular cycle.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    let claim = &result.claims[0];
    // Re-compute manually and compare.
    let passages: Vec<&str> = context.iter().map(String::as_str).collect();
    let (expected_score, _) = evaluator.entailment_score(&claim.claim, &passages);
    assert!((claim.entailment_score - expected_score).abs() < EPS);
}

#[test]
fn evaluate_claim_is_entailed_respects_threshold() {
    let cfg = FaithfulnessConfig::new().with_min_entailment_score(0.9);
    let evaluator = FaithfulnessEvaluator::new(cfg);
    let answer = "The sky is blue today.";
    // Partial vocabulary overlap — unlikely to reach 0.9 threshold.
    let context = vec!["The sky appears somewhat blue on clear days.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    let claim = &result.claims[0];
    // is_entailed must match score >= threshold.
    assert_eq!(claim.is_entailed, claim.entailment_score >= 0.9);
}

#[test]
fn evaluate_supporting_passage_text_comes_from_context() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let ctx_text = "The sun provides light and warmth.";
    let answer = "The sun provides warmth light.";
    let context = vec![ctx_text.to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, 1);
    if let Some(ref sp) = result.claims[0].supporting_passage {
        assert_eq!(sp.as_str(), ctx_text);
    }
}

// ── additional coverage tests ─────────────────────────────────────────────────

#[test]
fn entailment_score_multiple_passages_returns_max() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // Claim: "water boils heat" → tokens {water, boils, heat}
    // passage A: "water flows cold" → intersection {water} → 1/3 ≈ 0.333
    // passage B: "water boils heat always" → intersection {water, boils, heat} → 3/3 = 1.0
    let (score, passage) = evaluator.entailment_score(
        "water boils heat",
        &["water flows cold river", "water boils heat always"],
    );
    assert!((score - 1.0).abs() < EPS);
    assert_eq!(passage.as_deref(), Some("water boils heat always"));
}

#[test]
fn evaluate_all_claims_have_claim_string_set() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "Alpha beta gamma. Delta epsilon zeta.";
    let context = vec!["Alpha beta gamma delta epsilon.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    for claim in &result.claims {
        assert!(!claim.claim.is_empty());
    }
}

#[test]
fn evaluate_faithfulness_one_when_all_entailed() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "Dogs bark loudly. Cats meow softly.";
    let context = vec!["Dogs bark loudly when alarmed. Cats meow softly for attention.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert!((result.faithfulness - 1.0).abs() < EPS);
}

#[test]
fn evaluate_faithfulness_zero_when_none_entailed() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // Completely different vocabulary for all claims.
    let answer = "Uranium combustion ignites spontaneously. Penguins fly supersonic speeds.";
    let context = vec!["Butterflies migrate southward. Gardens bloom springtime.".to_owned()];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert!((result.faithfulness - 0.0).abs() < EPS);
}

#[test]
fn claim_split_three_sentences_yields_three_claims() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let claims = evaluator
        .split_into_claims("First sentence here. Second sentence here. Third sentence here.");
    assert_eq!(claims.len(), 3);
}

#[test]
fn entailment_score_short_claim_one_unique_token_handled() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    // "go run" → tokens {go, run} (each length >= 2)
    let (score, _) = evaluator.entailment_score("go run", &["go run fast now"]);
    assert!((score - 1.0).abs() < EPS);
}

#[test]
fn evaluate_three_context_passages_best_used() {
    let evaluator = FaithfulnessEvaluator::new(FaithfulnessConfig::default());
    let answer = "Photosynthesis converts sunlight into energy.";
    let context = vec![
        "Respiration consumes oxygen.".to_owned(),
        "Photosynthesis uses sunlight and converts it into chemical energy.".to_owned(),
        "Transpiration releases water vapour.".to_owned(),
    ];
    let result = evaluator.evaluate(answer, &context).unwrap();
    assert_eq!(result.total_claims, 1);
    assert_eq!(result.entailed_count, 1);
    let sp = result.claims[0]
        .supporting_passage
        .as_ref()
        .expect("should have passage");
    assert!(sp.contains("Photosynthesis"));
}
