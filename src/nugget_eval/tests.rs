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
//! Tests for the `nugget_eval` module.

use crate::nugget_eval::extractor::{HeuristicNuggetExtractor, NuggetExtractor};
use crate::nugget_eval::scorer::NuggetScorer;
use crate::nugget_eval::types::{
    Nugget, NuggetConfig, NuggetEvalError, NuggetImportance, NuggetScore,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn default_scorer() -> NuggetScorer<HeuristicNuggetExtractor> {
    NuggetScorer::new(NuggetConfig::default())
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

// ── NuggetConfig defaults ─────────────────────────────────────────────────────

#[test]
fn test_config_default_coverage_threshold() {
    assert_eq!(NuggetConfig::default().coverage_threshold, 0.5);
}

#[test]
fn test_config_default_vital_weight() {
    assert_eq!(NuggetConfig::default().vital_weight, 1.0);
}

#[test]
fn test_config_default_okay_weight() {
    assert_eq!(NuggetConfig::default().okay_weight, 0.5);
}

#[test]
fn test_config_default_min_nugget_tokens() {
    assert_eq!(NuggetConfig::default().min_nugget_tokens, 2);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(NuggetConfig::new(), NuggetConfig::default());
}

// ── NuggetConfig builders ─────────────────────────────────────────────────────

#[test]
fn test_config_with_coverage_threshold() {
    let cfg = NuggetConfig::new().with_coverage_threshold(0.8);
    assert_eq!(cfg.coverage_threshold, 0.8);
}

#[test]
fn test_config_with_vital_weight() {
    let cfg = NuggetConfig::new().with_vital_weight(2.0);
    assert_eq!(cfg.vital_weight, 2.0);
}

#[test]
fn test_config_with_okay_weight() {
    let cfg = NuggetConfig::new().with_okay_weight(0.25);
    assert_eq!(cfg.okay_weight, 0.25);
}

#[test]
fn test_config_with_min_nugget_tokens() {
    let cfg = NuggetConfig::new().with_min_nugget_tokens(3);
    assert_eq!(cfg.min_nugget_tokens, 3);
}

#[test]
fn test_config_builder_chaining() {
    let cfg = NuggetConfig::new()
        .with_coverage_threshold(0.6)
        .with_vital_weight(1.5)
        .with_okay_weight(0.4)
        .with_min_nugget_tokens(4);
    assert_eq!(cfg.coverage_threshold, 0.6);
    assert_eq!(cfg.vital_weight, 1.5);
    assert_eq!(cfg.okay_weight, 0.4);
    assert_eq!(cfg.min_nugget_tokens, 4);
}

#[test]
fn test_config_weight_for_vital() {
    let cfg = NuggetConfig::default();
    assert_eq!(cfg.weight_for(NuggetImportance::Vital), 1.0);
}

#[test]
fn test_config_weight_for_okay() {
    let cfg = NuggetConfig::default();
    assert_eq!(cfg.weight_for(NuggetImportance::Okay), 0.5);
}

// ── Nugget constructors ───────────────────────────────────────────────────────

#[test]
fn test_nugget_new() {
    let n = Nugget::new("the sky is blue", NuggetImportance::Okay);
    assert_eq!(n.text, "the sky is blue");
    assert_eq!(n.importance, NuggetImportance::Okay);
}

#[test]
fn test_nugget_vital_helper() {
    let n = Nugget::vital("Paris is the capital");
    assert_eq!(n.importance, NuggetImportance::Vital);
    assert!(n.is_vital());
}

#[test]
fn test_nugget_okay_helper() {
    let n = Nugget::okay("it is nice");
    assert_eq!(n.importance, NuggetImportance::Okay);
    assert!(!n.is_vital());
}

// ── Extractor: produces at least one nugget ───────────────────────────────────

#[test]
fn test_extractor_produces_at_least_one_nugget() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("Rust is a systems programming language");
    assert!(!nuggets.is_empty());
}

#[test]
fn test_extractor_splits_multiple_clauses() {
    let ext = HeuristicNuggetExtractor::default();
    // Three clauses separated by punctuation.
    let nuggets = ext.extract("Cats are mammals. Dogs are mammals. Fish are aquatic.");
    assert_eq!(nuggets.len(), 3);
}

#[test]
fn test_extractor_splits_on_comma() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("apples are red, bananas are yellow");
    assert_eq!(nuggets.len(), 2);
}

#[test]
fn test_extractor_splits_on_and() {
    let ext = HeuristicNuggetExtractor::default();
    // " and " should act as a clause boundary.
    let nuggets = ext.extract("water boils at high temperature and ice melts when warm");
    assert_eq!(nuggets.len(), 2);
}

#[test]
fn test_extractor_splits_on_semicolon() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("first fact holds true; second fact also holds");
    assert_eq!(nuggets.len(), 2);
}

#[test]
fn test_extractor_splits_on_question_mark() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("what is rust? rust is a language");
    assert_eq!(nuggets.len(), 2);
}

#[test]
fn test_extractor_splits_on_exclamation() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("rust is fast! rust is safe");
    assert_eq!(nuggets.len(), 2);
}

// ── Vital marking (entity / number) ───────────────────────────────────────────

#[test]
fn test_extractor_marks_entity_as_vital() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("Mozilla created the language");
    assert_eq!(nuggets.len(), 1);
    assert_eq!(nuggets[0].importance, NuggetImportance::Vital);
}

#[test]
fn test_extractor_marks_number_as_vital() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("the release happened in 2015 year");
    assert_eq!(nuggets.len(), 1);
    assert_eq!(nuggets[0].importance, NuggetImportance::Vital);
}

#[test]
fn test_extractor_marks_plain_clause_as_okay() {
    let ext = HeuristicNuggetExtractor::default();
    // No capitalized token, no digit.
    let nuggets = ext.extract("the sky appears blue");
    assert_eq!(nuggets.len(), 1);
    assert_eq!(nuggets[0].importance, NuggetImportance::Okay);
}

#[test]
fn test_extractor_mixed_importance() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("the sky appears blue, Einstein won a prize");
    assert_eq!(nuggets.len(), 2);
    assert_eq!(nuggets[0].importance, NuggetImportance::Okay);
    assert_eq!(nuggets[1].importance, NuggetImportance::Vital);
}

#[test]
fn test_extractor_number_with_units_is_vital() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("the distance was 42km long");
    assert_eq!(nuggets.len(), 1);
    assert_eq!(nuggets[0].importance, NuggetImportance::Vital);
}

// ── min_nugget_tokens drops tiny clauses ──────────────────────────────────────

#[test]
fn test_extractor_drops_tiny_clauses() {
    let ext = HeuristicNuggetExtractor::default();
    // "ok" has one token; the longer clause survives.
    let nuggets = ext.extract("ok, the experiment succeeded clearly");
    assert_eq!(nuggets.len(), 1);
    assert!(nuggets[0].text.contains("experiment"));
}

#[test]
fn test_extractor_single_token_clause_dropped() {
    let ext = HeuristicNuggetExtractor::default();
    let nuggets = ext.extract("hi");
    assert!(nuggets.is_empty());
}

#[test]
fn test_extractor_min_tokens_three() {
    let ext = HeuristicNuggetExtractor::with_min_tokens(3);
    // Two tokens -> dropped; four tokens -> kept.
    let nuggets = ext.extract("two words, this clause has four tokens");
    assert_eq!(nuggets.len(), 1);
    assert!(nuggets[0].text.contains("four"));
}

#[test]
fn test_extractor_min_tokens_accessor() {
    let ext = HeuristicNuggetExtractor::with_min_tokens(5);
    assert_eq!(ext.min_nugget_tokens(), 5);
}

#[test]
fn test_extractor_default_min_tokens_accessor() {
    assert_eq!(HeuristicNuggetExtractor::default().min_nugget_tokens(), 2);
}

#[test]
fn test_extractor_short_tokens_filtered() {
    let ext = HeuristicNuggetExtractor::default();
    // Single-letter tokens (len < 2) do not count toward the threshold.
    let nuggets = ext.extract("a b c");
    assert!(nuggets.is_empty());
}

#[test]
fn test_extractor_empty_reference_yields_no_nuggets() {
    let ext = HeuristicNuggetExtractor::default();
    assert!(ext.extract("").is_empty());
}

// ── is_covered threshold ──────────────────────────────────────────────────────

#[test]
fn test_is_covered_full_overlap() {
    let scorer = default_scorer();
    let nugget = Nugget::okay("memory safety guarantee");
    assert!(scorer.is_covered(&nugget, "rust provides a memory safety guarantee"));
}

#[test]
fn test_is_covered_below_threshold() {
    let scorer = default_scorer();
    // Three tokens; only "memory" present -> 1/3 < 0.5.
    let nugget = Nugget::okay("memory safety guarantee");
    assert!(!scorer.is_covered(&nugget, "memory only here"));
}

#[test]
fn test_is_covered_exactly_at_threshold() {
    let scorer = default_scorer();
    // Two tokens, one present -> 0.5 >= 0.5 (default threshold).
    let nugget = Nugget::okay("alpha beta");
    assert!(scorer.is_covered(&nugget, "alpha gamma delta"));
}

#[test]
fn test_is_covered_high_threshold_rejects_partial() {
    let scorer = NuggetScorer::new(NuggetConfig::new().with_coverage_threshold(0.9));
    // Two tokens, one present -> 0.5 < 0.9.
    let nugget = Nugget::okay("alpha beta");
    assert!(!scorer.is_covered(&nugget, "alpha gamma"));
}

#[test]
fn test_is_covered_zero_threshold_always_true() {
    let scorer = NuggetScorer::new(NuggetConfig::new().with_coverage_threshold(0.0));
    let nugget = Nugget::okay("alpha beta");
    // 0/2 = 0.0 >= 0.0.
    assert!(scorer.is_covered(&nugget, "nothing matches here"));
}

#[test]
fn test_is_covered_empty_nugget_never_covered() {
    let scorer = NuggetScorer::new(NuggetConfig::new().with_coverage_threshold(0.0));
    // Punctuation-only nugget tokenizes to nothing.
    let nugget = Nugget::okay("...");
    assert!(!scorer.is_covered(&nugget, "anything at all"));
}

#[test]
fn test_is_covered_case_insensitive() {
    let scorer = default_scorer();
    let nugget = Nugget::vital("Paris France");
    assert!(scorer.is_covered(&nugget, "the city of paris is in france"));
}

// ── Full coverage ⇒ 1.0 ───────────────────────────────────────────────────────

#[test]
fn test_full_coverage_coverage_one() {
    let scorer = default_scorer();
    let nuggets = vec![
        Nugget::okay("the sky appears blue"),
        Nugget::okay("grass appears green"),
    ];
    let answer = "the sky appears blue while the grass appears green";
    let score = scorer.score_with_nuggets(&nuggets, answer);
    assert!(approx(score.coverage, 1.0));
}

#[test]
fn test_full_coverage_weighted_one() {
    let scorer = default_scorer();
    let nuggets = vec![
        Nugget::vital("Mozilla project"),
        Nugget::okay("the sky appears blue"),
    ];
    let answer = "the Mozilla project and the sky appears blue";
    let score = scorer.score_with_nuggets(&nuggets, answer);
    assert!(approx(score.weighted_score, 1.0));
}

#[test]
fn test_full_coverage_all_indices_covered() {
    let scorer = default_scorer();
    let nuggets = vec![Nugget::okay("alpha beta"), Nugget::okay("gamma delta")];
    let answer = "alpha beta gamma delta";
    let score = scorer.score_with_nuggets(&nuggets, answer);
    assert_eq!(score.covered, vec![0, 1]);
    assert!(score.missed.is_empty());
}

// ── Zero coverage ⇒ 0.0 ───────────────────────────────────────────────────────

#[test]
fn test_zero_coverage_coverage_zero() {
    let scorer = default_scorer();
    let nuggets = vec![Nugget::okay("alpha beta"), Nugget::okay("gamma delta")];
    let score = scorer.score_with_nuggets(&nuggets, "completely unrelated text");
    assert!(approx(score.coverage, 0.0));
}

#[test]
fn test_zero_coverage_weighted_zero() {
    let scorer = default_scorer();
    let nuggets = vec![Nugget::vital("alpha beta"), Nugget::okay("gamma delta")];
    let score = scorer.score_with_nuggets(&nuggets, "nothing relevant whatsoever");
    assert!(approx(score.weighted_score, 0.0));
}

#[test]
fn test_zero_coverage_all_missed() {
    let scorer = default_scorer();
    let nuggets = vec![Nugget::okay("alpha beta"), Nugget::okay("gamma delta")];
    let score = scorer.score_with_nuggets(&nuggets, "unrelated content");
    assert!(score.covered.is_empty());
    assert_eq!(score.missed, vec![0, 1]);
}

// ── Partial coverage fraction ─────────────────────────────────────────────────

#[test]
fn test_partial_coverage_half() {
    let scorer = default_scorer();
    let nuggets = vec![Nugget::okay("alpha beta"), Nugget::okay("gamma delta")];
    // Only first nugget's tokens appear.
    let score = scorer.score_with_nuggets(&nuggets, "alpha beta only");
    assert!(approx(score.coverage, 0.5));
}

#[test]
fn test_partial_coverage_one_third() {
    let scorer = default_scorer();
    let nuggets = vec![
        Nugget::okay("alpha beta"),
        Nugget::okay("gamma delta"),
        Nugget::okay("epsilon zeta"),
    ];
    let score = scorer.score_with_nuggets(&nuggets, "alpha beta present");
    assert!(approx(score.coverage, 1.0 / 3.0));
}

#[test]
fn test_partial_coverage_indices() {
    let scorer = default_scorer();
    let nuggets = vec![
        Nugget::okay("alpha beta"),
        Nugget::okay("gamma delta"),
        Nugget::okay("epsilon zeta"),
    ];
    // First and third covered.
    let score = scorer.score_with_nuggets(&nuggets, "alpha beta and epsilon zeta");
    assert_eq!(score.covered, vec![0, 2]);
    assert_eq!(score.missed, vec![1]);
}

// ── vital_coverage computed separately ────────────────────────────────────────

#[test]
fn test_vital_coverage_separate_from_overall() {
    let scorer = default_scorer();
    let nuggets = vec![
        Nugget::vital("Mozilla project"),
        Nugget::okay("the sky appears blue"),
    ];
    // Only the vital nugget is covered.
    let score = scorer.score_with_nuggets(&nuggets, "the Mozilla project shipped");
    assert!(approx(score.vital_coverage, 1.0));
    assert!(approx(score.coverage, 0.5));
}

#[test]
fn test_vital_coverage_partial() {
    let scorer = default_scorer();
    let nuggets = vec![
        Nugget::vital("Paris France"),
        Nugget::vital("Berlin Germany"),
    ];
    let score = scorer.score_with_nuggets(&nuggets, "paris is in france");
    assert!(approx(score.vital_coverage, 0.5));
}

#[test]
fn test_vital_coverage_zero_when_no_vital_nuggets() {
    let scorer = default_scorer();
    let nuggets = vec![Nugget::okay("alpha beta"), Nugget::okay("gamma delta")];
    let score = scorer.score_with_nuggets(&nuggets, "alpha beta gamma delta");
    // No vital nuggets -> vital_coverage defined as 0.0.
    assert!(approx(score.vital_coverage, 0.0));
    assert!(approx(score.coverage, 1.0));
}

#[test]
fn test_vital_coverage_missed_vital() {
    let scorer = default_scorer();
    let nuggets = vec![
        Nugget::vital("Tokyo Japan"),
        Nugget::okay("the food is tasty"),
    ];
    // Cover the okay one only.
    let score = scorer.score_with_nuggets(&nuggets, "the food is tasty indeed");
    assert!(approx(score.vital_coverage, 0.0));
    assert!(approx(score.coverage, 0.5));
}

// ── weighted_score uses vital/okay weights ────────────────────────────────────

#[test]
fn test_weighted_score_vital_only_covered() {
    let scorer = default_scorer();
    // vital weight 1.0, okay weight 0.5; only vital covered.
    let nuggets = vec![Nugget::vital("alpha beta"), Nugget::okay("gamma delta")];
    let score = scorer.score_with_nuggets(&nuggets, "alpha beta only");
    // covered weight = 1.0; total weight = 1.5 -> 0.6667.
    assert!(approx(score.weighted_score, 1.0 / 1.5));
}

#[test]
fn test_weighted_score_okay_only_covered() {
    let scorer = default_scorer();
    let nuggets = vec![Nugget::vital("alpha beta"), Nugget::okay("gamma delta")];
    let score = scorer.score_with_nuggets(&nuggets, "gamma delta only");
    // covered weight = 0.5; total weight = 1.5 -> 0.3333.
    assert!(approx(score.weighted_score, 0.5 / 1.5));
}

#[test]
fn test_weighted_score_custom_weights() {
    let cfg = NuggetConfig::new()
        .with_vital_weight(3.0)
        .with_okay_weight(1.0);
    let scorer = NuggetScorer::new(cfg);
    let nuggets = vec![Nugget::vital("alpha beta"), Nugget::okay("gamma delta")];
    let score = scorer.score_with_nuggets(&nuggets, "alpha beta covered");
    // covered weight = 3.0; total = 4.0 -> 0.75.
    assert!(approx(score.weighted_score, 0.75));
}

#[test]
fn test_weighted_score_differs_from_coverage() {
    let scorer = default_scorer();
    let nuggets = vec![Nugget::vital("alpha beta"), Nugget::okay("gamma delta")];
    let score = scorer.score_with_nuggets(&nuggets, "alpha beta present");
    // coverage = 0.5 but weighted = 1.0/1.5 != 0.5.
    assert!(approx(score.coverage, 0.5));
    assert!(!approx(score.weighted_score, 0.5));
}

// ── empty nugget slice ────────────────────────────────────────────────────────

#[test]
fn test_score_with_empty_nuggets_all_zero() {
    let scorer = default_scorer();
    let score = scorer.score_with_nuggets(&[], "any answer");
    assert!(approx(score.coverage, 0.0));
    assert!(approx(score.vital_coverage, 0.0));
    assert!(approx(score.weighted_score, 0.0));
    assert!(score.covered.is_empty());
    assert!(score.missed.is_empty());
}

// ── score() end-to-end + error ────────────────────────────────────────────────

#[test]
fn test_score_end_to_end() {
    let scorer = default_scorer();
    let reference = "Rust was created by Mozilla, and it guarantees memory safety";
    let answer = "Rust, made at Mozilla, guarantees memory safety";
    let score = scorer
        .score(reference, answer)
        .expect("non-empty reference");
    assert!(score.coverage > 0.0);
}

#[test]
fn test_score_empty_reference_errors() {
    let scorer = default_scorer();
    let err = scorer.score("", "some answer").unwrap_err();
    assert!(matches!(err, NuggetEvalError::EmptyReference));
}

#[test]
fn test_score_whitespace_reference_errors() {
    let scorer = default_scorer();
    let err = scorer.score("   \t\n  ", "some answer").unwrap_err();
    assert!(matches!(err, NuggetEvalError::EmptyReference));
}

#[test]
fn test_score_error_display() {
    let err = NuggetEvalError::EmptyReference;
    assert_eq!(err.to_string(), "reference must not be empty");
}

#[test]
fn test_score_empty_answer_is_zero_coverage() {
    let scorer = default_scorer();
    let score = scorer
        .score("the sky appears blue", "")
        .expect("non-empty reference");
    assert!(approx(score.coverage, 0.0));
}

// ── with_extractor + custom extractor ─────────────────────────────────────────

struct FixedExtractor;
impl NuggetExtractor for FixedExtractor {
    fn extract(&self, _reference: &str) -> Vec<Nugget> {
        vec![Nugget::vital("alpha beta"), Nugget::okay("gamma delta")]
    }
}

#[test]
fn test_with_extractor_uses_custom_nuggets() {
    let scorer = NuggetScorer::with_extractor(NuggetConfig::default(), FixedExtractor);
    let score = scorer
        .score("ignored reference text", "alpha beta gamma delta")
        .expect("non-empty reference");
    assert!(approx(score.coverage, 1.0));
}

#[test]
fn test_with_extractor_respects_config_weights() {
    let cfg = NuggetConfig::new().with_okay_weight(0.0);
    let scorer = NuggetScorer::with_extractor(cfg, FixedExtractor);
    let score = scorer
        .score("ignored", "alpha beta only")
        .expect("non-empty reference");
    // vital covered (weight 1.0), okay missed; total weight = 1.0 + 0.0 = 1.0.
    assert!(approx(score.weighted_score, 1.0));
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_extractor_deterministic() {
    let ext = HeuristicNuggetExtractor::default();
    let reference = "Mozilla shipped Rust in 2015, and the sky appears blue";
    let a = ext.extract(reference);
    let b = ext.extract(reference);
    assert_eq!(a, b);
}

#[test]
fn test_score_deterministic() {
    let scorer = default_scorer();
    let reference = "Mozilla shipped Rust in 2015, and the sky appears blue";
    let answer = "Mozilla shipped Rust in 2015";
    let s1 = scorer.score(reference, answer).expect("ok");
    let s2 = scorer.score(reference, answer).expect("ok");
    assert_eq!(s1, s2);
}

#[test]
fn test_nugget_score_clone_eq() {
    let score = NuggetScore {
        coverage: 0.5,
        vital_coverage: 1.0,
        weighted_score: 0.6,
        covered: vec![0],
        missed: vec![1],
    };
    assert_eq!(score.clone(), score);
}

// ── scorer construction wires extractor min tokens ────────────────────────────

#[test]
fn test_new_scorer_wires_min_tokens_into_extractor() {
    let cfg = NuggetConfig::new().with_min_nugget_tokens(5);
    let scorer = NuggetScorer::new(cfg);
    assert_eq!(scorer.extractor.min_nugget_tokens(), 5);
}

#[test]
fn test_score_respects_min_nugget_tokens() {
    // With a high min, the short "ok" clause is dropped, only the long one remains.
    let cfg = NuggetConfig::new().with_min_nugget_tokens(4);
    let scorer = NuggetScorer::new(cfg);
    let score = scorer
        .score(
            "ok fine, this clause has many qualifying tokens here",
            "this clause has many qualifying tokens here",
        )
        .expect("non-empty reference");
    // Exactly one nugget survives and it is covered.
    assert!(approx(score.coverage, 1.0));
    assert_eq!(score.covered.len(), 1);
}
