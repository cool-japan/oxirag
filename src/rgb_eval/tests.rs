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
use crate::rgb_eval::evaluator::RgbEvaluator;
use crate::rgb_eval::types::{RgbAbility, RgbConfig, RgbError, RgbScores, RgbTestCase};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn eval_default() -> RgbEvaluator {
    RgbEvaluator::new(RgbConfig::default())
}

fn noise_case(query: &str, gold: &str) -> RgbTestCase {
    RgbTestCase::new(
        query,
        gold,
        vec![
            "Some unrelated noise about the weather.".to_string(),
            format!("The relevant fact: {gold}."),
        ],
        RgbAbility::NoiseRobustness,
    )
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

// ── RgbConfig: defaults & builders ──────────────────────────────────────────

#[test]
fn config_default_match_threshold() {
    assert!(approx(RgbConfig::default().match_threshold, 0.5));
}

#[test]
fn config_default_rejection_phrases() {
    let cfg = RgbConfig::default();
    assert_eq!(cfg.rejection_phrases.len(), 4);
    assert!(cfg.rejection_phrases.contains(&"cannot answer".to_string()));
    assert!(
        cfg.rejection_phrases
            .contains(&"not enough information".to_string())
    );
    assert!(cfg.rejection_phrases.contains(&"no answer".to_string()));
    assert!(cfg.rejection_phrases.contains(&"insufficient".to_string()));
}

#[test]
fn config_new_equals_default() {
    let a = RgbConfig::new();
    let b = RgbConfig::default();
    assert!(approx(a.match_threshold, b.match_threshold));
    assert_eq!(a.rejection_phrases, b.rejection_phrases);
}

#[test]
fn config_with_match_threshold() {
    let cfg = RgbConfig::new().with_match_threshold(0.8);
    assert!(approx(cfg.match_threshold, 0.8));
}

#[test]
fn config_with_rejection_phrases() {
    let cfg = RgbConfig::new().with_rejection_phrases(vec!["unknown".to_string()]);
    assert_eq!(cfg.rejection_phrases, vec!["unknown".to_string()]);
}

#[test]
fn config_builders_chain() {
    let cfg = RgbConfig::new()
        .with_match_threshold(0.3)
        .with_rejection_phrases(vec!["nope".to_string(), "dunno".to_string()]);
    assert!(approx(cfg.match_threshold, 0.3));
    assert_eq!(cfg.rejection_phrases.len(), 2);
}

// ── RgbAbility ──────────────────────────────────────────────────────────────

#[test]
fn ability_labels_are_stable() {
    assert_eq!(RgbAbility::NoiseRobustness.label(), "noise_robustness");
    assert_eq!(RgbAbility::NegativeRejection.label(), "negative_rejection");
    assert_eq!(
        RgbAbility::InformationIntegration.label(),
        "information_integration"
    );
    assert_eq!(
        RgbAbility::CounterfactualRobustness.label(),
        "counterfactual_robustness"
    );
}

// ── RgbTestCase construction ────────────────────────────────────────────────

#[test]
fn test_case_new_defaults() {
    let case = RgbTestCase::new(
        "q",
        "gold",
        vec!["ctx".to_string()],
        RgbAbility::NoiseRobustness,
    );
    assert_eq!(case.query, "q");
    assert_eq!(case.gold_answer, "gold");
    assert!(case.sub_answers.is_empty());
    assert!(case.has_answer);
    assert_eq!(case.ability, RgbAbility::NoiseRobustness);
}

#[test]
fn test_case_with_sub_answers() {
    let case = RgbTestCase::new("q", "gold", vec![], RgbAbility::InformationIntegration)
        .with_sub_answers(vec!["alpha".to_string(), "beta".to_string()]);
    assert_eq!(case.sub_answers.len(), 2);
}

#[test]
fn test_case_with_has_answer_false() {
    let case =
        RgbTestCase::new("q", "gold", vec![], RgbAbility::NegativeRejection).with_has_answer(false);
    assert!(!case.has_answer);
    assert_eq!(case.ability, RgbAbility::NegativeRejection);
}

// ── is_correct_answer: overlap threshold ────────────────────────────────────

#[test]
fn correct_answer_exact_match() {
    let ev = eval_default();
    assert!(ev.is_correct_answer("Paris", "Paris"));
}

#[test]
fn correct_answer_full_gold_coverage() {
    let ev = eval_default();
    // All gold tokens present in a longer answer.
    assert!(ev.is_correct_answer("The capital is Paris France", "Paris France"));
}

#[test]
fn correct_answer_half_coverage_meets_threshold() {
    let ev = eval_default();
    // gold has 2 tokens, answer covers 1 => 0.5 >= 0.5.
    assert!(ev.is_correct_answer("Paris", "Paris London"));
}

#[test]
fn correct_answer_below_threshold_fails() {
    let ev = eval_default();
    // gold has 3 tokens, answer covers 1 => 0.33 < 0.5.
    assert!(!ev.is_correct_answer("alpha", "alpha beta gamma"));
}

#[test]
fn correct_answer_no_overlap_fails() {
    let ev = eval_default();
    assert!(!ev.is_correct_answer("completely different", "Paris France"));
}

#[test]
fn correct_answer_empty_gold_fails() {
    let ev = eval_default();
    assert!(!ev.is_correct_answer("anything", ""));
}

#[test]
fn correct_answer_case_insensitive() {
    let ev = eval_default();
    assert!(ev.is_correct_answer("PARIS france", "paris FRANCE"));
}

#[test]
fn correct_answer_high_threshold_requires_all() {
    let ev = RgbEvaluator::new(RgbConfig::new().with_match_threshold(1.0));
    assert!(ev.is_correct_answer("alpha beta", "alpha beta"));
    assert!(!ev.is_correct_answer("alpha", "alpha beta"));
}

#[test]
fn correct_answer_low_threshold_lenient() {
    let ev = RgbEvaluator::new(RgbConfig::new().with_match_threshold(0.1));
    // 1 of 4 gold tokens => 0.25 >= 0.1.
    assert!(ev.is_correct_answer("alpha", "alpha beta gamma delta"));
}

// ── is_rejection: phrase detection ──────────────────────────────────────────

#[test]
fn rejection_detects_cannot_answer() {
    let ev = eval_default();
    assert!(ev.is_rejection("I cannot answer this question."));
}

#[test]
fn rejection_detects_not_enough_information() {
    let ev = eval_default();
    assert!(ev.is_rejection("There is not enough information to respond."));
}

#[test]
fn rejection_detects_no_answer() {
    let ev = eval_default();
    assert!(ev.is_rejection("Sorry, no answer is available."));
}

#[test]
fn rejection_detects_insufficient() {
    let ev = eval_default();
    assert!(ev.is_rejection("The context is insufficient for this query."));
}

#[test]
fn rejection_case_insensitive() {
    let ev = eval_default();
    assert!(ev.is_rejection("I CANNOT ANSWER that."));
}

#[test]
fn rejection_false_for_plain_answer() {
    let ev = eval_default();
    assert!(!ev.is_rejection("The capital of France is Paris."));
}

#[test]
fn rejection_custom_phrase() {
    let ev = RgbEvaluator::new(
        RgbConfig::new().with_rejection_phrases(vec!["i don't know".to_string()]),
    );
    assert!(ev.is_rejection("Honestly, I don't know."));
    assert!(!ev.is_rejection("I cannot answer.")); // default phrase no longer configured
}

#[test]
fn rejection_empty_phrase_list() {
    let ev = RgbEvaluator::new(RgbConfig::new().with_rejection_phrases(vec![]));
    assert!(!ev.is_rejection("I cannot answer."));
}

// ── NegativeRejection cases ─────────────────────────────────────────────────

#[test]
fn negative_rejection_correct_when_rejecting() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "What is the population of Atlantis?",
        "(no answer exists)",
        vec!["Atlantis is a fictional place.".to_string()],
        RgbAbility::NegativeRejection,
    )
    .with_has_answer(false);
    assert!(ev.evaluate_case(&case, "I cannot answer this from the context."));
}

#[test]
fn negative_rejection_wrong_when_answering() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "What is the population of Atlantis?",
        "(no answer exists)",
        vec!["Atlantis is a fictional place.".to_string()],
        RgbAbility::NegativeRejection,
    )
    .with_has_answer(false);
    // System hallucinates a confident answer => wrong.
    assert!(!ev.evaluate_case(&case, "The population is ten million."));
}

#[test]
fn negative_rejection_ignores_gold_overlap() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "q",
        "cannot answer", // even if gold overlaps, only rejection counts
        vec![],
        RgbAbility::NegativeRejection,
    );
    // A plain answer that happens to overlap gold tokens is still wrong.
    assert!(!ev.evaluate_case(&case, "the answer is forty two"));
    // A rejection is correct.
    assert!(ev.evaluate_case(&case, "I cannot answer that."));
}

// ── InformationIntegration cases ────────────────────────────────────────────

#[test]
fn integration_correct_when_all_sub_answers_covered() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "Who founded the two companies?",
        "Alice founded Acme and Bob founded Beta",
        vec![
            "Acme was founded by Alice.".to_string(),
            "Beta was founded by Bob.".to_string(),
        ],
        RgbAbility::InformationIntegration,
    )
    .with_sub_answers(vec!["Alice Acme".to_string(), "Bob Beta".to_string()]);
    let answer = "Alice founded Acme, and Bob founded Beta.";
    assert!(ev.evaluate_case(&case, answer));
}

#[test]
fn integration_wrong_when_one_sub_answer_missing() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "Who founded the two companies?",
        "Alice founded Acme and Bob founded Beta",
        vec![],
        RgbAbility::InformationIntegration,
    )
    .with_sub_answers(vec!["Alice Acme".to_string(), "Bob Beta".to_string()]);
    // Only the first fact is covered.
    let answer = "Alice founded Acme.";
    assert!(!ev.evaluate_case(&case, answer));
}

#[test]
fn integration_wrong_when_no_sub_answers_covered() {
    let ev = eval_default();
    let case = RgbTestCase::new("q", "gold", vec![], RgbAbility::InformationIntegration)
        .with_sub_answers(vec!["alpha one".to_string(), "beta two".to_string()]);
    assert!(!ev.evaluate_case(&case, "completely unrelated text here"));
}

#[test]
fn integration_rejection_is_wrong() {
    let ev = eval_default();
    let case = RgbTestCase::new("q", "gold", vec![], RgbAbility::InformationIntegration)
        .with_sub_answers(vec!["alpha".to_string()]);
    // Even if a sub-answer token appears, a rejection answer is wrong.
    assert!(!ev.evaluate_case(&case, "alpha — but I cannot answer fully."));
}

#[test]
fn integration_falls_back_to_gold_without_sub_answers() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "q",
        "Paris France",
        vec![],
        RgbAbility::InformationIntegration,
    );
    assert!(ev.evaluate_case(&case, "The answer is Paris France."));
    assert!(!ev.evaluate_case(&case, "Something else entirely here."));
}

// ── NoiseRobustness cases ───────────────────────────────────────────────────

#[test]
fn noise_robustness_correct_despite_noise() {
    let ev = eval_default();
    let case = noise_case("What is the capital of France?", "Paris France");
    assert!(ev.evaluate_case(&case, "Ignoring the noise, the capital is Paris France."));
}

#[test]
fn noise_robustness_wrong_on_bad_answer() {
    let ev = eval_default();
    let case = noise_case("What is the capital of France?", "Paris France");
    assert!(!ev.evaluate_case(&case, "The capital is Berlin Germany."));
}

#[test]
fn noise_robustness_rejection_is_wrong() {
    let ev = eval_default();
    let case = noise_case("What is the capital of France?", "Paris France");
    // The answer IS in context, so rejecting is wrong.
    assert!(!ev.evaluate_case(&case, "I cannot answer due to insufficient context."));
}

// ── CounterfactualRobustness cases ──────────────────────────────────────────

#[test]
fn counterfactual_correct_on_true_gold() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci",
        vec!["A false note claims Pablo Picasso painted the Mona Lisa.".to_string()],
        RgbAbility::CounterfactualRobustness,
    );
    // Correct = TRUE gold answer, resisting the counterfactual.
    assert!(ev.evaluate_case(&case, "It was painted by Leonardo da Vinci."));
}

#[test]
fn counterfactual_wrong_when_repeating_counterfactual() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci",
        vec!["A false note claims Pablo Picasso painted the Mona Lisa.".to_string()],
        RgbAbility::CounterfactualRobustness,
    );
    // System repeats the planted counterfactual => does not match gold => wrong.
    assert!(!ev.evaluate_case(&case, "It was painted by Pablo Picasso."));
}

#[test]
fn counterfactual_rejection_is_wrong() {
    let ev = eval_default();
    let case = RgbTestCase::new(
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci",
        vec!["A false note claims Picasso painted it.".to_string()],
        RgbAbility::CounterfactualRobustness,
    );
    assert!(!ev.evaluate_case(&case, "There is no answer I can give."));
}

// ── evaluate: per-ability scores ────────────────────────────────────────────

#[test]
fn evaluate_all_correct_overall_one() {
    let ev = eval_default();
    let cases = vec![
        noise_case("q1", "Paris"),
        RgbTestCase::new("q2", "gold", vec![], RgbAbility::NegativeRejection)
            .with_has_answer(false),
        RgbTestCase::new("q3", "gold", vec![], RgbAbility::InformationIntegration)
            .with_sub_answers(vec!["alpha".to_string(), "beta".to_string()]),
        RgbTestCase::new(
            "q4",
            "Leonardo",
            vec![],
            RgbAbility::CounterfactualRobustness,
        ),
    ];
    let answers = vec![
        "Paris".to_string(),
        "I cannot answer.".to_string(),
        "alpha and beta".to_string(),
        "Leonardo".to_string(),
    ];
    let scores = ev.evaluate(&cases, &answers).expect("evaluate");
    assert!(approx(scores.noise_robustness, 1.0));
    assert!(approx(scores.negative_rejection, 1.0));
    assert!(approx(scores.information_integration, 1.0));
    assert!(approx(scores.counterfactual_robustness, 1.0));
    assert!(approx(scores.overall, 1.0));
}

#[test]
fn evaluate_all_wrong_overall_zero() {
    let ev = eval_default();
    let cases = vec![
        noise_case("q1", "Paris"),
        RgbTestCase::new("q2", "gold", vec![], RgbAbility::NegativeRejection)
            .with_has_answer(false),
    ];
    let answers = vec![
        "Berlin".to_string(),                  // wrong noise answer
        "The population is huge.".to_string(), // should have rejected
    ];
    let scores = ev.evaluate(&cases, &answers).expect("evaluate");
    assert!(approx(scores.noise_robustness, 0.0));
    assert!(approx(scores.negative_rejection, 0.0));
    assert!(approx(scores.overall, 0.0));
}

#[test]
fn evaluate_per_ability_half_accuracy() {
    let ev = eval_default();
    // Two noise cases: one correct, one wrong => 0.5.
    let cases = vec![noise_case("q1", "Paris"), noise_case("q2", "London")];
    let answers = vec!["Paris".to_string(), "Tokyo".to_string()];
    let scores = ev.evaluate(&cases, &answers).expect("evaluate");
    assert!(approx(scores.noise_robustness, 0.5));
    // Absent abilities are zero.
    assert!(approx(scores.negative_rejection, 0.0));
    assert!(approx(scores.information_integration, 0.0));
    assert!(approx(scores.counterfactual_robustness, 0.0));
    // Overall = mean of present abilities only (just noise) => 0.5.
    assert!(approx(scores.overall, 0.5));
}

#[test]
fn evaluate_overall_is_mean_of_present_abilities() {
    let ev = eval_default();
    // Noise: 1.0 (1/1), Negative: 0.0 (0/1). Overall = (1.0 + 0.0) / 2 = 0.5.
    let cases = vec![
        noise_case("q1", "Paris"),
        RgbTestCase::new("q2", "gold", vec![], RgbAbility::NegativeRejection)
            .with_has_answer(false),
    ];
    let answers = vec!["Paris".to_string(), "It is fifty.".to_string()];
    let scores = ev.evaluate(&cases, &answers).expect("evaluate");
    assert!(approx(scores.noise_robustness, 1.0));
    assert!(approx(scores.negative_rejection, 0.0));
    assert!(approx(scores.overall, 0.5));
}

#[test]
fn evaluate_overall_excludes_absent_abilities() {
    let ev = eval_default();
    // Only counterfactual cases present, all correct => overall = 1.0, not 0.25.
    let cases = vec![
        RgbTestCase::new(
            "q1",
            "Leonardo",
            vec![],
            RgbAbility::CounterfactualRobustness,
        ),
        RgbTestCase::new("q2", "Newton", vec![], RgbAbility::CounterfactualRobustness),
    ];
    let answers = vec!["Leonardo".to_string(), "Newton".to_string()];
    let scores = ev.evaluate(&cases, &answers).expect("evaluate");
    assert!(approx(scores.counterfactual_robustness, 1.0));
    assert!(approx(scores.overall, 1.0));
    assert!(approx(scores.noise_robustness, 0.0));
}

#[test]
fn evaluate_three_abilities_overall() {
    let ev = eval_default();
    // noise 1.0, integration 0.0, counterfactual 1.0 => overall = 2/3.
    let cases = vec![
        noise_case("q1", "Paris"),
        RgbTestCase::new("q2", "gold", vec![], RgbAbility::InformationIntegration)
            .with_sub_answers(vec!["alpha".to_string(), "beta".to_string()]),
        RgbTestCase::new("q3", "Newton", vec![], RgbAbility::CounterfactualRobustness),
    ];
    let answers = vec![
        "Paris".to_string(),
        "only alpha".to_string(), // beta missing => integration wrong
        "Newton".to_string(),
    ];
    let scores = ev.evaluate(&cases, &answers).expect("evaluate");
    assert!(approx(scores.noise_robustness, 1.0));
    assert!(approx(scores.information_integration, 0.0));
    assert!(approx(scores.counterfactual_robustness, 1.0));
    assert!(approx(scores.overall, 2.0 / 3.0));
}

// ── evaluate: errors ────────────────────────────────────────────────────────

#[test]
fn evaluate_empty_cases_errors() {
    let ev = eval_default();
    let err = ev.evaluate(&[], &[]).unwrap_err();
    assert!(matches!(err, RgbError::EmptyCases));
}

#[test]
fn evaluate_length_mismatch_errors() {
    let ev = eval_default();
    let cases = vec![noise_case("q1", "Paris")];
    let answers = vec!["Paris".to_string(), "extra".to_string()];
    let err = ev.evaluate(&cases, &answers).unwrap_err();
    match err {
        RgbError::LengthMismatch { answers, cases } => {
            assert_eq!(answers, 2);
            assert_eq!(cases, 1);
        }
        RgbError::EmptyCases => panic!("expected LengthMismatch"),
    }
}

#[test]
fn error_display_messages() {
    let empty = RgbError::EmptyCases;
    assert_eq!(empty.to_string(), "no test cases");
    let mismatch = RgbError::LengthMismatch {
        answers: 3,
        cases: 2,
    };
    assert_eq!(mismatch.to_string(), "answers length 3 != cases length 2");
}

// ── RgbScores ───────────────────────────────────────────────────────────────

#[test]
fn scores_default_is_zero() {
    let s = RgbScores::default();
    assert!(approx(s.noise_robustness, 0.0));
    assert!(approx(s.negative_rejection, 0.0));
    assert!(approx(s.information_integration, 0.0));
    assert!(approx(s.counterfactual_robustness, 0.0));
    assert!(approx(s.overall, 0.0));
}

// ── Determinism ─────────────────────────────────────────────────────────────

#[test]
fn evaluate_is_deterministic() {
    let ev = eval_default();
    let cases = vec![
        noise_case("q1", "Paris France"),
        RgbTestCase::new("q2", "gold", vec![], RgbAbility::NegativeRejection)
            .with_has_answer(false),
        RgbTestCase::new("q3", "gold", vec![], RgbAbility::InformationIntegration)
            .with_sub_answers(vec!["alpha one".to_string(), "beta two".to_string()]),
        RgbTestCase::new(
            "q4",
            "Leonardo da Vinci",
            vec![],
            RgbAbility::CounterfactualRobustness,
        ),
    ];
    let answers = vec![
        "The capital is Paris France.".to_string(),
        "I cannot answer.".to_string(),
        "alpha one and beta two".to_string(),
        "Leonardo da Vinci painted it.".to_string(),
    ];
    let first = ev.evaluate(&cases, &answers).expect("first");
    for _ in 0..5 {
        let again = ev.evaluate(&cases, &answers).expect("again");
        assert_eq!(first, again);
    }
}

// ── Evaluator construction ──────────────────────────────────────────────────

#[test]
fn evaluator_new_stores_config() {
    let ev = RgbEvaluator::new(RgbConfig::new().with_match_threshold(0.7));
    assert!(approx(ev.config.match_threshold, 0.7));
}

#[test]
fn evaluator_default_matches_default_config() {
    let ev = RgbEvaluator::default();
    assert!(approx(ev.config.match_threshold, 0.5));
    assert_eq!(ev.config.rejection_phrases.len(), 4);
}
