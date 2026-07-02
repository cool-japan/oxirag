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
//! Tests for the `crud_rag` module.

use crate::crud_rag::engine::CrudRagHarness;
use crate::crud_rag::types::{
    CrudCase, CrudMetric, CrudOperation, CrudRagConfig, CrudRagError, CrudScore,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn default_harness() -> CrudRagHarness {
    CrudRagHarness::new(CrudRagConfig::default())
}

fn score_of(scores: &[CrudScore], metric: CrudMetric) -> f32 {
    scores
        .iter()
        .find(|s| s.metric == metric)
        .map(|s| s.value)
        .expect("metric should be present")
}

fn combined_of(scores: &[CrudScore]) -> f32 {
    score_of(scores, CrudMetric::Combined)
}

// ── CrudOperation ─────────────────────────────────────────────────────────────

#[test]
fn operation_labels_are_stable() {
    assert_eq!(CrudOperation::Create.label(), "create");
    assert_eq!(CrudOperation::Read.label(), "read");
    assert_eq!(CrudOperation::Update.label(), "update");
    assert_eq!(CrudOperation::Delete.label(), "delete");
}

#[test]
fn operation_all_returns_four_in_order() {
    let all = CrudOperation::all();
    assert_eq!(
        all,
        [
            CrudOperation::Create,
            CrudOperation::Read,
            CrudOperation::Update,
            CrudOperation::Delete
        ]
    );
}

#[test]
fn operation_display_matches_label() {
    assert_eq!(CrudOperation::Create.to_string(), "create");
    assert_eq!(CrudOperation::Delete.to_string(), "delete");
}

// ── CrudMetric ───────────────────────────────────────────────────────────────

#[test]
fn metric_labels_are_stable() {
    assert_eq!(CrudMetric::RougeL.label(), "rouge_l");
    assert_eq!(CrudMetric::Bleu.label(), "bleu");
    assert_eq!(CrudMetric::ExactMatch.label(), "exact_match");
    assert_eq!(CrudMetric::TokenF1.label(), "token_f1");
    assert_eq!(
        CrudMetric::CorrectionSimilarity.label(),
        "correction_similarity"
    );
    assert_eq!(CrudMetric::ErrorRemoval.label(), "error_removal");
    assert_eq!(
        CrudMetric::CorrectSpanIntroduced.label(),
        "correct_span_introduced"
    );
    assert_eq!(CrudMetric::Coverage.label(), "coverage");
    assert_eq!(CrudMetric::Redundancy.label(), "redundancy");
    assert_eq!(CrudMetric::Combined.label(), "combined");
}

// ── CrudCase builders ────────────────────────────────────────────────────────

#[test]
fn case_new_defaults() {
    let case = CrudCase::new("id-1", CrudOperation::Read, "output text");
    assert_eq!(case.id, "id-1");
    assert_eq!(case.operation, CrudOperation::Read);
    assert_eq!(case.output, "output text");
    assert!(case.references.is_empty());
    assert!(case.key_points.is_empty());
    assert!(case.error_span.is_none());
    assert!(case.correct_span.is_none());
}

#[test]
fn case_with_reference_appends() {
    let case = CrudCase::new("id", CrudOperation::Read, "out")
        .with_reference("r1")
        .with_reference("r2");
    assert_eq!(case.references, vec!["r1".to_string(), "r2".to_string()]);
}

#[test]
fn case_with_references_replaces() {
    let case = CrudCase::new("id", CrudOperation::Read, "out")
        .with_reference("r1")
        .with_references(vec!["r2".to_string(), "r3".to_string()]);
    assert_eq!(case.references, vec!["r2".to_string(), "r3".to_string()]);
}

#[test]
fn case_with_key_points() {
    let case = CrudCase::new("id", CrudOperation::Delete, "out")
        .with_key_points(vec!["kp1".to_string(), "kp2".to_string()]);
    assert_eq!(case.key_points, vec!["kp1".to_string(), "kp2".to_string()]);
}

#[test]
fn case_with_error_and_correct_span() {
    let case = CrudCase::new("id", CrudOperation::Update, "out")
        .with_error_span("wrong")
        .with_correct_span("right");
    assert_eq!(case.error_span, Some("wrong".to_string()));
    assert_eq!(case.correct_span, Some("right".to_string()));
}

#[test]
fn case_builder_chaining() {
    let case = CrudCase::new("id", CrudOperation::Update, "out")
        .with_reference("ref")
        .with_error_span("e")
        .with_correct_span("c");
    assert_eq!(case.references, vec!["ref".to_string()]);
    assert_eq!(case.error_span, Some("e".to_string()));
    assert_eq!(case.correct_span, Some("c".to_string()));
}

// ── CrudRagConfig ────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = CrudRagConfig::default();
    assert_eq!(cfg.bleu_ngram_order, 4);
    assert!(approx(cfg.rouge_beta, 1.0));
    assert!(cfg.lowercase);
    assert!(cfg.strip_punctuation);
    assert!(approx(cfg.redundancy_penalty_weight, 0.5));
    assert_eq!(cfg.enabled_operations.len(), 4);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(CrudRagConfig::new(), CrudRagConfig::default());
}

#[test]
fn config_with_bleu_ngram_order_clamps_to_min_1() {
    let cfg = CrudRagConfig::new().with_bleu_ngram_order(0);
    assert_eq!(cfg.bleu_ngram_order, 1);
    let cfg = CrudRagConfig::new().with_bleu_ngram_order(2);
    assert_eq!(cfg.bleu_ngram_order, 2);
}

#[test]
fn config_with_rouge_beta() {
    let cfg = CrudRagConfig::new().with_rouge_beta(2.0);
    assert!(approx(cfg.rouge_beta, 2.0));
}

#[test]
fn config_with_lowercase_and_strip_punctuation() {
    let cfg = CrudRagConfig::new()
        .with_lowercase(false)
        .with_strip_punctuation(false);
    assert!(!cfg.lowercase);
    assert!(!cfg.strip_punctuation);
}

#[test]
fn config_with_redundancy_penalty_weight() {
    let cfg = CrudRagConfig::new().with_redundancy_penalty_weight(0.9);
    assert!(approx(cfg.redundancy_penalty_weight, 0.9));
}

#[test]
fn config_with_enabled_operations() {
    let cfg = CrudRagConfig::new().with_enabled_operations(vec![CrudOperation::Read]);
    assert_eq!(cfg.enabled_operations, vec![CrudOperation::Read]);
}

#[test]
fn config_is_enabled_true_and_false() {
    let cfg = CrudRagConfig::new().with_enabled_operations(vec![CrudOperation::Read]);
    assert!(cfg.is_enabled(CrudOperation::Read));
    assert!(!cfg.is_enabled(CrudOperation::Create));
}

// ── Create: ROUGE-L ──────────────────────────────────────────────────────────

#[test]
fn rouge_l_f1_hand_verified_prefix_case() {
    // candidate is an exact 3-token prefix of a 6-token reference:
    // lcs = 3, P = 3/3 = 1.0, R = 3/6 = 0.5, F1 = 2*1*0.5/1.5 = 2/3.
    let harness = default_harness();
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat")
        .with_reference("the cat sat on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::RougeL), 2.0 / 3.0));
}

#[test]
fn rouge_l_identical_strings_is_one() {
    let harness = default_harness();
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat on the mat")
        .with_reference("the cat sat on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::RougeL), 1.0));
}

#[test]
fn rouge_l_no_overlap_is_zero() {
    let harness = default_harness();
    let case =
        CrudCase::new("c", CrudOperation::Create, "alpha beta").with_reference("gamma delta");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::RougeL), 0.0));
}

#[test]
fn rouge_l_beta_zero_is_precision_only() {
    // beta = 0 collapses the F-beta combination to plain precision.
    let harness = CrudRagHarness::new(CrudRagConfig::new().with_rouge_beta(0.0));
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat")
        .with_reference("the cat sat on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::RougeL), 1.0));
}

#[test]
fn rouge_l_beta_two_favors_recall() {
    // P = 1.0, R = 0.5, beta = 2 => F = 5*R*P / (R + 4*P) = 2.5/4.5 = 5/9.
    let harness = CrudRagHarness::new(CrudRagConfig::new().with_rouge_beta(2.0));
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat")
        .with_reference("the cat sat on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::RougeL), 5.0 / 9.0));
}

// ── Create: BLEU ─────────────────────────────────────────────────────────────

#[test]
fn bleu_identical_strings_is_one() {
    let harness = default_harness();
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat on the mat")
        .with_reference("the cat sat on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::Bleu), 1.0));
}

#[test]
fn bleu_brevity_penalty_hand_verified() {
    // candidate is an exact 3-token prefix of a 6-token reference, so every
    // clipped n-gram precision up to order 3 is 1.0 and the score reduces to
    // the brevity penalty alone: exp(1 - 6/3) = exp(-1).
    let harness = default_harness();
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat")
        .with_reference("the cat sat on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    #[allow(clippy::cast_possible_truncation)]
    let expected = std::f64::consts::E.recip() as f32;
    assert!(approx(score_of(&scores, CrudMetric::Bleu), expected));
}

#[test]
fn bleu_brevity_penalty_is_one_when_candidate_not_shorter() {
    // candidate (6 tokens) is longer than reference (3 tokens), so the
    // brevity penalty is exactly 1.0 regardless of the length difference;
    // capping the order at 1 isolates it from n-gram-order precision decay:
    // p1 = 3/6 = 0.5 ("the": clipped to 1 of 2 occurrences, "cat"/"sat":
    // 1 each, "on"/"mat": 0 each), so BLEU = BP * p1 = 1.0 * 0.5 = 0.5.
    let harness = CrudRagHarness::new(CrudRagConfig::new().with_bleu_ngram_order(1));
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat on the mat")
        .with_reference("the cat sat");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::Bleu), 0.5));
}

#[test]
fn bleu_zero_when_high_order_ngram_missing() {
    // Order 4 (default): the 4-grams never match (one interior token
    // differs), so the geometric mean collapses to 0.
    let harness = default_harness();
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat on the roof")
        .with_reference("the cat sat near the roof");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::Bleu), 0.0));
}

#[test]
fn bleu_ngram_order_config_changes_score() {
    // Same pair as `bleu_zero_when_high_order_ngram_missing`, but capping
    // the order at 2 avoids the mismatching 3-/4-grams entirely:
    // p1 = 5/6, p2 = 3/5, geo_mean = sqrt(1/2), BP = 1.0 (equal lengths).
    let harness = CrudRagHarness::new(CrudRagConfig::new().with_bleu_ngram_order(2));
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat on the roof")
        .with_reference("the cat sat near the roof");
    let scores = harness.evaluate_case(&case).unwrap();
    let expected = 0.5f32.sqrt();
    assert!(approx(score_of(&scores, CrudMetric::Bleu), expected));
}

#[test]
fn create_case_combined_is_mean_of_rouge_and_bleu() {
    let harness = default_harness();
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat")
        .with_reference("the cat sat on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    let rouge = score_of(&scores, CrudMetric::RougeL);
    let bleu = score_of(&scores, CrudMetric::Bleu);
    assert!(approx(combined_of(&scores), f32::midpoint(rouge, bleu)));
}

#[test]
fn create_multi_reference_best_of() {
    let harness = default_harness();
    let case = CrudCase::new("c", CrudOperation::Create, "the cat sat on the mat")
        .with_reference("completely unrelated text")
        .with_reference("the cat sat on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::RougeL), 1.0));
    assert!(approx(score_of(&scores, CrudMetric::Bleu), 1.0));
}

// ── Read: exact match + token F1 ─────────────────────────────────────────────

#[test]
fn read_exact_match_true() {
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "Paris").with_reference("Paris");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::ExactMatch), 1.0));
}

#[test]
fn read_exact_match_false() {
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "London").with_reference("Paris");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::ExactMatch), 0.0));
}

#[test]
fn read_token_f1_partial_overlap_hand_verified() {
    // Both 6 tokens, differ in exactly one ("sat" vs "lay"): overlap = 5,
    // precision = recall = 5/6, so F1 = 5/6.
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "the cat sat on the mat")
        .with_reference("the cat lay on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::TokenF1), 5.0 / 6.0));
}

#[test]
fn read_token_f1_zero_no_overlap() {
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "foo bar").with_reference("baz qux");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::TokenF1), 0.0));
}

#[test]
fn read_combined_is_half_em_half_f1() {
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "the cat sat on the mat")
        .with_reference("the cat lay on the mat");
    let scores = harness.evaluate_case(&case).unwrap();
    let em = score_of(&scores, CrudMetric::ExactMatch);
    let f1 = score_of(&scores, CrudMetric::TokenF1);
    assert!(approx(combined_of(&scores), 0.5 * em + 0.5 * f1));
}

#[test]
fn read_multi_reference_best_of() {
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "Paris")
        .with_reference("London")
        .with_reference("Paris");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::ExactMatch), 1.0));
    assert!(approx(score_of(&scores, CrudMetric::TokenF1), 1.0));
}

// ── Update: correction similarity + error removal + span introduction ───────

fn update_case(id: &str, output: &str) -> CrudCase {
    CrudCase::new(id, CrudOperation::Update, output)
        .with_reference("The event was held in Lyon in 1998.")
        .with_error_span("Paris")
        .with_correct_span("Lyon")
}

#[test]
fn update_rewards_good_correction() {
    let harness = default_harness();
    let case = update_case("u1", "The event was held in Lyon in 1998.");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(combined_of(&scores), 1.0));
}

#[test]
fn update_penalizes_bad_correction() {
    let harness = default_harness();
    let case = update_case("u2", "The event was held in Paris in 1998.");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(combined_of(&scores), 0.4375));
}

#[test]
fn update_good_correction_beats_bad_correction() {
    let harness = default_harness();
    let good = harness
        .evaluate_case(&update_case("u1", "The event was held in Lyon in 1998."))
        .unwrap();
    let bad = harness
        .evaluate_case(&update_case("u2", "The event was held in Paris in 1998."))
        .unwrap();
    assert!(combined_of(&good) > combined_of(&bad));
}

#[test]
fn update_error_removal_flag() {
    let harness = default_harness();
    let removed = harness
        .evaluate_case(&update_case("u1", "The event was held in Lyon in 1998."))
        .unwrap();
    let kept = harness
        .evaluate_case(&update_case("u2", "The event was held in Paris in 1998."))
        .unwrap();
    assert!(approx(score_of(&removed, CrudMetric::ErrorRemoval), 1.0));
    assert!(approx(score_of(&kept, CrudMetric::ErrorRemoval), 0.0));
}

#[test]
fn update_correct_span_introduced_flag() {
    let harness = default_harness();
    let introduced = harness
        .evaluate_case(&update_case("u1", "The event was held in Lyon in 1998."))
        .unwrap();
    let missing = harness
        .evaluate_case(&update_case("u2", "The event was held in Paris in 1998."))
        .unwrap();
    assert!(approx(
        score_of(&introduced, CrudMetric::CorrectSpanIntroduced),
        1.0
    ));
    assert!(approx(
        score_of(&missing, CrudMetric::CorrectSpanIntroduced),
        0.0
    ));
}

#[test]
fn update_partial_correction_scores_between_good_and_bad() {
    // Error span removed, but the wrong replacement ("Berlin") is
    // introduced instead of the correct one ("Lyon"): 0.5*0.875 + 0.25*1 +
    // 0.25*0 = 0.6875, strictly between the fully-bad (0.4375) and
    // fully-good (1.0) cases.
    let harness = default_harness();
    let partial = harness
        .evaluate_case(&update_case("u3", "The event was held in Berlin in 1998."))
        .unwrap();
    let bad = harness
        .evaluate_case(&update_case("u2", "The event was held in Paris in 1998."))
        .unwrap();
    let good = harness
        .evaluate_case(&update_case("u1", "The event was held in Lyon in 1998."))
        .unwrap();
    assert!(approx(combined_of(&partial), 0.6875));
    assert!(combined_of(&bad) < combined_of(&partial));
    assert!(combined_of(&partial) < combined_of(&good));
}

#[test]
fn update_missing_error_span_error() {
    let harness = default_harness();
    let case = CrudCase::new("u", CrudOperation::Update, "out")
        .with_reference("ref")
        .with_correct_span("Lyon");
    let err = harness.evaluate_case(&case).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::MissingErrorSpan { case_id } if case_id == "u"
    ));
}

#[test]
fn update_missing_correct_span_error() {
    let harness = default_harness();
    let case = CrudCase::new("u", CrudOperation::Update, "out")
        .with_reference("ref")
        .with_error_span("Paris");
    let err = harness.evaluate_case(&case).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::MissingCorrectSpan { case_id } if case_id == "u"
    ));
}

#[test]
fn update_missing_reference_error() {
    let harness = default_harness();
    let case = CrudCase::new("u", CrudOperation::Update, "out")
        .with_error_span("Paris")
        .with_correct_span("Lyon");
    let err = harness.evaluate_case(&case).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::MissingReference { case_id, operation }
            if case_id == "u" && operation == CrudOperation::Update
    ));
}

// ── Delete: coverage + redundancy ────────────────────────────────────────────

#[test]
fn delete_coverage_and_redundancy_hand_verified() {
    // summary_content = {paris, capital, london}; kp1 = {paris, capital}
    // (covered), kp2 requires {london, city} (not covered, "city" absent):
    // coverage = 1/2. Sentences: [s1, s1, s3] with jaccard(s1,s1)=1,
    // jaccard(s1,s3)=0 twice => mean redundancy = 1/3. combined =
    // 0.5 - 0.5*(1/3) = 1/3.
    let harness = default_harness();
    let case = CrudCase::new(
        "d",
        CrudOperation::Delete,
        "Paris is the capital. Paris is the capital. London is big.",
    )
    .with_key_points(vec![
        "Paris is the capital".to_string(),
        "London is a big city".to_string(),
    ]);
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::Coverage), 0.5));
    assert!(approx(score_of(&scores, CrudMetric::Redundancy), 1.0 / 3.0));
    assert!(approx(combined_of(&scores), 1.0 / 3.0));
}

#[test]
fn delete_full_coverage_no_redundancy() {
    let harness = default_harness();
    let case = CrudCase::new(
        "d",
        CrudOperation::Delete,
        "The company was founded in Boston. It now serves clients worldwide.",
    )
    .with_key_points(vec![
        "The company was founded in Boston".to_string(),
        "It serves clients worldwide".to_string(),
    ]);
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::Coverage), 1.0));
    assert!(approx(score_of(&scores, CrudMetric::Redundancy), 0.0));
    assert!(approx(combined_of(&scores), 1.0));
}

#[test]
fn delete_redundancy_penalizes_repetition() {
    let harness = default_harness();
    let key_points = vec![
        "The company was founded in Boston".to_string(),
        "It serves clients worldwide".to_string(),
    ];
    let no_repeat = CrudCase::new(
        "d1",
        CrudOperation::Delete,
        "The company was founded in Boston. It now serves clients worldwide.",
    )
    .with_key_points(key_points.clone());
    let repeated = CrudCase::new(
        "d2",
        CrudOperation::Delete,
        "The company was founded in Boston. The company was founded in Boston. It now serves clients worldwide.",
    )
    .with_key_points(key_points);

    let no_repeat_scores = harness.evaluate_case(&no_repeat).unwrap();
    let repeated_scores = harness.evaluate_case(&repeated).unwrap();

    // Both cover every key point equally well ...
    assert!(approx(
        score_of(&no_repeat_scores, CrudMetric::Coverage),
        1.0
    ));
    assert!(approx(
        score_of(&repeated_scores, CrudMetric::Coverage),
        1.0
    ));
    // ... but the repeated summary is penalized for redundancy.
    assert!(approx(
        score_of(&repeated_scores, CrudMetric::Redundancy),
        1.0 / 3.0
    ));
    assert!(combined_of(&repeated_scores) < combined_of(&no_repeat_scores));
    assert!(approx(combined_of(&repeated_scores), 5.0 / 6.0));
}

#[test]
fn delete_redundancy_penalty_weight_is_configurable() {
    let key_points = vec![
        "The company was founded in Boston".to_string(),
        "It serves clients worldwide".to_string(),
    ];
    let case = CrudCase::new(
        "d",
        CrudOperation::Delete,
        "The company was founded in Boston. The company was founded in Boston. It now serves clients worldwide.",
    )
    .with_key_points(key_points);

    let zero_weight = CrudRagHarness::new(CrudRagConfig::new().with_redundancy_penalty_weight(0.0));
    let full_weight = CrudRagHarness::new(CrudRagConfig::new().with_redundancy_penalty_weight(1.0));

    let zero_scores = zero_weight.evaluate_case(&case).unwrap();
    let full_scores = full_weight.evaluate_case(&case).unwrap();
    assert!(approx(combined_of(&zero_scores), 1.0));
    assert!(combined_of(&full_scores) < combined_of(&zero_scores));
}

#[test]
fn delete_missing_key_points_error() {
    let harness = default_harness();
    let case = CrudCase::new("d", CrudOperation::Delete, "some summary text");
    let err = harness.evaluate_case(&case).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::MissingKeyPoints { case_id } if case_id == "d"
    ));
}

// ── harness dispatch + aggregation ───────────────────────────────────────────

#[test]
fn evaluate_case_dispatches_by_operation() {
    let harness = default_harness();
    let create = harness
        .evaluate_case(
            &CrudCase::new("c", CrudOperation::Create, "the cat sat").with_reference("the cat"),
        )
        .unwrap();
    assert!(create.iter().all(|s| s.operation == CrudOperation::Create));

    let read = harness
        .evaluate_case(&CrudCase::new("r", CrudOperation::Read, "Paris").with_reference("Paris"))
        .unwrap();
    assert!(read.iter().all(|s| s.operation == CrudOperation::Read));
}

#[test]
fn evaluate_read_only_batch_mean_hand_verified() {
    // case1: EM=1.0,F1=1.0 -> combined=1.0. case2: EM=0.0,F1=0.0 (no
    // overlap) -> combined=0.0. mean = 0.5, and it is the only operation
    // present so overall = 0.5 too.
    let harness = default_harness();
    let cases = vec![
        CrudCase::new("r1", CrudOperation::Read, "Paris").with_reference("Paris"),
        CrudCase::new("r2", CrudOperation::Read, "London").with_reference("Paris"),
    ];
    let report = harness.evaluate(&cases).unwrap();
    assert!(approx(report.read_mean, 0.5));
    assert!(approx(report.overall, 0.5));
    assert!(approx(report.create_mean, 0.0));
    assert!(approx(report.update_mean, 0.0));
    assert!(approx(report.delete_mean, 0.0));
}

#[test]
fn evaluate_mixed_operation_batch_cross_check() {
    let harness = default_harness();
    let cases = vec![
        CrudCase::new("c1", CrudOperation::Create, "the cat sat")
            .with_reference("the cat sat on the mat"),
        CrudCase::new("r1", CrudOperation::Read, "Paris").with_reference("Paris"),
        CrudCase::new("r2", CrudOperation::Read, "London").with_reference("Paris"),
        update_case("u1", "The event was held in Lyon in 1998."),
        CrudCase::new(
            "d1",
            CrudOperation::Delete,
            "The company was founded in Boston. It now serves clients worldwide.",
        )
        .with_key_points(vec![
            "The company was founded in Boston".to_string(),
            "It serves clients worldwide".to_string(),
        ]),
    ];

    let c1 = combined_of(&harness.evaluate_case(&cases[0]).unwrap());
    let r1 = combined_of(&harness.evaluate_case(&cases[1]).unwrap());
    let r2 = combined_of(&harness.evaluate_case(&cases[2]).unwrap());
    let u1 = combined_of(&harness.evaluate_case(&cases[3]).unwrap());
    let d1 = combined_of(&harness.evaluate_case(&cases[4]).unwrap());

    let report = harness.evaluate(&cases).unwrap();

    assert!(approx(report.create_mean, c1));
    assert!(approx(report.read_mean, f32::midpoint(r1, r2)));
    assert!(approx(report.update_mean, u1));
    assert!(approx(report.delete_mean, d1));

    let expected_overall =
        (report.create_mean + report.read_mean + report.update_mean + report.delete_mean) / 4.0;
    assert!(approx(report.overall, expected_overall));

    assert_eq!(report.total_count(), 5);
    assert_eq!(report.count_for(CrudOperation::Read), 2);
    assert_eq!(report.count_for(CrudOperation::Create), 1);
    assert_eq!(report.scores.len(), 3 + 3 + 3 + 4 + 3); // create+read+read+update+delete
}

#[test]
fn evaluate_report_mean_for_matches_fields() {
    let harness = default_harness();
    let cases = vec![
        CrudCase::new("c1", CrudOperation::Create, "the cat sat")
            .with_reference("the cat sat on the mat"),
        CrudCase::new("r1", CrudOperation::Read, "Paris").with_reference("Paris"),
    ];
    let report = harness.evaluate(&cases).unwrap();
    assert!(approx(
        report.mean_for(CrudOperation::Create),
        report.create_mean
    ));
    assert!(approx(
        report.mean_for(CrudOperation::Read),
        report.read_mean
    ));
    assert!(approx(
        report.mean_for(CrudOperation::Update),
        report.update_mean
    ));
    assert!(approx(
        report.mean_for(CrudOperation::Delete),
        report.delete_mean
    ));
}

#[test]
fn evaluate_is_deterministic() {
    let harness = default_harness();
    let cases = vec![
        CrudCase::new("c1", CrudOperation::Create, "the cat sat")
            .with_reference("the cat sat on the mat"),
        CrudCase::new("r1", CrudOperation::Read, "Paris").with_reference("Paris"),
        update_case("u1", "The event was held in Lyon in 1998."),
        CrudCase::new(
            "d1",
            CrudOperation::Delete,
            "The company was founded in Boston. It now serves clients worldwide.",
        )
        .with_key_points(vec!["The company was founded in Boston".to_string()]),
    ];
    let report1 = harness.evaluate(&cases).unwrap();
    let report2 = harness.evaluate(&cases).unwrap();
    assert_eq!(report1, report2);
}

// ── error paths ──────────────────────────────────────────────────────────────

#[test]
fn evaluate_empty_cases_error() {
    let harness = default_harness();
    let err = harness.evaluate(&[]).unwrap_err();
    assert!(matches!(err, CrudRagError::EmptyCases));
}

#[test]
fn evaluate_empty_output_error() {
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "   ").with_reference("Paris");
    let err = harness.evaluate_case(&case).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::EmptyOutput { case_id } if case_id == "r"
    ));
}

#[test]
fn evaluate_create_missing_reference_error() {
    let harness = default_harness();
    let case = CrudCase::new("c", CrudOperation::Create, "some output");
    let err = harness.evaluate_case(&case).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::MissingReference { case_id, operation }
            if case_id == "c" && operation == CrudOperation::Create
    ));
}

#[test]
fn evaluate_read_missing_reference_error() {
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "some output");
    let err = harness.evaluate_case(&case).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::MissingReference { case_id, operation }
            if case_id == "r" && operation == CrudOperation::Read
    ));
}

#[test]
fn evaluate_operation_disabled_error() {
    let harness = CrudRagHarness::new(
        CrudRagConfig::new()
            .with_enabled_operations(vec![CrudOperation::Read, CrudOperation::Update]),
    );
    let case = CrudCase::new("c", CrudOperation::Create, "out").with_reference("ref");
    let err = harness.evaluate_case(&case).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::OperationDisabled { case_id, operation }
            if case_id == "c" && operation == CrudOperation::Create
    ));
}

#[test]
fn evaluate_stops_at_first_invalid_case() {
    let harness = default_harness();
    let cases = vec![
        CrudCase::new("r1", CrudOperation::Read, "Paris").with_reference("Paris"),
        CrudCase::new("r2", CrudOperation::Read, "no reference here"),
    ];
    let err = harness.evaluate(&cases).unwrap_err();
    assert!(matches!(
        err,
        CrudRagError::MissingReference { case_id, .. } if case_id == "r2"
    ));
}

// ── normalization ────────────────────────────────────────────────────────────

#[test]
fn normalization_strip_punctuation_affects_exact_match() {
    let case = CrudCase::new("r", CrudOperation::Read, "Paris.").with_reference("Paris");

    let stripped = default_harness();
    let stripped_scores = stripped.evaluate_case(&case).unwrap();
    assert!(approx(
        score_of(&stripped_scores, CrudMetric::ExactMatch),
        1.0
    ));

    let not_stripped = CrudRagHarness::new(CrudRagConfig::new().with_strip_punctuation(false));
    let not_stripped_scores = not_stripped.evaluate_case(&case).unwrap();
    assert!(approx(
        score_of(&not_stripped_scores, CrudMetric::ExactMatch),
        0.0
    ));
}

#[test]
fn normalization_lowercase_affects_exact_match() {
    let case = CrudCase::new("r", CrudOperation::Read, "PARIS").with_reference("paris");

    let lowered = default_harness();
    let lowered_scores = lowered.evaluate_case(&case).unwrap();
    assert!(approx(
        score_of(&lowered_scores, CrudMetric::ExactMatch),
        1.0
    ));

    let not_lowered = CrudRagHarness::new(CrudRagConfig::new().with_lowercase(false));
    let not_lowered_scores = not_lowered.evaluate_case(&case).unwrap();
    assert!(approx(
        score_of(&not_lowered_scores, CrudMetric::ExactMatch),
        0.0
    ));
}

#[test]
fn normalization_default_is_case_and_punctuation_insensitive() {
    let harness = default_harness();
    let case = CrudCase::new("r", CrudOperation::Read, "PARIS!").with_reference("paris");
    let scores = harness.evaluate_case(&case).unwrap();
    assert!(approx(score_of(&scores, CrudMetric::ExactMatch), 1.0));
}
