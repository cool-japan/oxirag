use crate::llm_judge::judge::LlmJudge;
use crate::llm_judge::types::{
    Criterion, CriterionScores, HeuristicJudge, JudgeContext, JudgeMode, JudgeModel,
    LlmJudgeConfig, LlmJudgeError, PairwiseVerdict, PointwiseVerdict, Pref, Rubric,
};
use crate::types::{Document, DocumentId, SearchResult};

// ── helpers ───────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str) -> SearchResult {
    SearchResult::new(
        Document::new(content).with_id(DocumentId::from_string(id)),
        1.0,
        0,
    )
}

fn default_ctx() -> JudgeContext {
    JudgeContext::new()
}

// ── JudgeMode ─────────────────────────────────────────────────────────────

#[test]
fn judge_mode_default_is_pointwise() {
    assert_eq!(JudgeMode::default(), JudgeMode::Pointwise);
}

#[test]
fn judge_mode_variants_equality() {
    assert_eq!(JudgeMode::Pairwise, JudgeMode::Pairwise);
    assert_eq!(JudgeMode::Reference, JudgeMode::Reference);
    assert_ne!(JudgeMode::Pointwise, JudgeMode::Pairwise);
}

// ── Criterion ─────────────────────────────────────────────────────────────

#[test]
fn criterion_new_fields() {
    let c = Criterion::new("relevance", 0.5, "How relevant?");
    assert_eq!(c.name, "relevance");
    assert!((c.weight - 0.5_f32).abs() < 1e-6);
    assert_eq!(c.description, "How relevant?");
}

#[test]
fn criterion_zero_weight_allowed() {
    let c = Criterion::new("none", 0.0, "unused");
    assert!((c.weight - 0.0_f32).abs() < 1e-6);
}

// ── Rubric ────────────────────────────────────────────────────────────────

#[test]
fn rubric_new_empty_criteria() {
    let r = Rubric::new(vec![]);
    assert_eq!(r.criteria.len(), 0);
}

#[test]
fn rubric_relevance_helpfulness_groundedness_has_three_criteria() {
    let r = Rubric::relevance_helpfulness_groundedness();
    assert_eq!(r.criteria.len(), 3);
}

#[test]
fn rubric_relevance_helpfulness_groundedness_criterion_names() {
    let r = Rubric::relevance_helpfulness_groundedness();
    let names: Vec<&str> = r.criteria.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"relevance"));
    assert!(names.contains(&"helpfulness"));
    assert!(names.contains(&"groundedness"));
}

#[test]
fn rubric_correctness_has_three_criteria() {
    let r = Rubric::correctness();
    assert_eq!(r.criteria.len(), 3);
}

#[test]
fn rubric_correctness_criterion_names() {
    let r = Rubric::correctness();
    let names: Vec<&str> = r.criteria.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"accuracy"));
    assert!(names.contains(&"completeness"));
    assert!(names.contains(&"precision"));
}

#[test]
fn rubric_normalize_weights_sum_to_one() {
    let r = Rubric::relevance_helpfulness_groundedness().normalize();
    let sum: f32 = r.criteria.iter().map(|c| c.weight).sum();
    assert!((sum - 1.0_f32).abs() < 1e-5);
}

#[test]
fn rubric_normalize_zero_weights_no_panic() {
    let r = Rubric::new(vec![
        Criterion::new("a", 0.0, "desc"),
        Criterion::new("b", 0.0, "desc"),
    ]);
    let normalized = r.normalize();
    // weights remain 0, no panic
    for c in &normalized.criteria {
        assert!((c.weight - 0.0_f32).abs() < 1e-6);
    }
}

#[test]
fn rubric_normalize_single_criterion_weight_one() {
    let r = Rubric::new(vec![Criterion::new("a", 3.0, "desc")]).normalize();
    assert!((r.criteria[0].weight - 1.0_f32).abs() < 1e-6);
}

// ── JudgeContext ──────────────────────────────────────────────────────────

#[test]
fn judge_context_new_empty() {
    let ctx = JudgeContext::new();
    assert!(ctx.sources.is_empty());
    assert!(ctx.reference.is_none());
}

#[test]
fn judge_context_with_sources() {
    let src = make_result("s1", "source content");
    let ctx = JudgeContext::new().with_sources(vec![src]);
    assert_eq!(ctx.sources.len(), 1);
}

#[test]
fn judge_context_with_reference() {
    let ctx = JudgeContext::new().with_reference("reference answer");
    assert_eq!(ctx.reference.as_deref(), Some("reference answer"));
}

// ── Pref ──────────────────────────────────────────────────────────────────

#[test]
fn pref_variants_equality() {
    assert_eq!(Pref::A, Pref::A);
    assert_eq!(Pref::B, Pref::B);
    assert_eq!(Pref::Tie, Pref::Tie);
    assert_ne!(Pref::A, Pref::B);
    assert_ne!(Pref::A, Pref::Tie);
}

// ── LlmJudgeConfig ────────────────────────────────────────────────────────

#[test]
fn llm_judge_config_default_tie_margin() {
    let cfg = LlmJudgeConfig::default();
    assert!((cfg.tie_margin - 0.05_f32).abs() < 1e-6);
}

#[test]
fn llm_judge_config_default_pointwise_scale() {
    let cfg = LlmJudgeConfig::default();
    assert!((cfg.pointwise_scale - 5.0_f32).abs() < 1e-6);
}

#[test]
fn llm_judge_config_with_rubric() {
    let r = Rubric::correctness();
    let cfg = LlmJudgeConfig::default().with_rubric(r);
    assert_eq!(cfg.rubric.criteria.len(), 3);
}

#[test]
fn llm_judge_config_with_tie_margin() {
    let cfg = LlmJudgeConfig::default().with_tie_margin(0.1);
    assert!((cfg.tie_margin - 0.1_f32).abs() < 1e-6);
}

#[test]
fn llm_judge_config_with_pointwise_scale() {
    let cfg = LlmJudgeConfig::default().with_pointwise_scale(10.0);
    assert!((cfg.pointwise_scale - 10.0_f32).abs() < 1e-6);
}

// ── HeuristicJudge ────────────────────────────────────────────────────────

#[test]
fn heuristic_judge_empty_answer_error() {
    let j = HeuristicJudge;
    let res = j.grade("query", "", &default_ctx());
    assert!(matches!(res, Err(LlmJudgeError::EmptyAnswer)));
}

#[test]
fn heuristic_judge_whitespace_only_answer_error() {
    let j = HeuristicJudge;
    let res = j.grade("query", "   ", &default_ctx());
    assert!(matches!(res, Err(LlmJudgeError::EmptyAnswer)));
}

#[test]
fn heuristic_judge_non_empty_overall_in_unit_interval() {
    let j = HeuristicJudge;
    let scores = j
        .grade(
            "What is Rust?",
            "Rust is a systems programming language.",
            &default_ctx(),
        )
        .unwrap();
    assert!((0.0..=1.0).contains(&scores.overall));
}

#[test]
fn heuristic_judge_identical_query_answer_high_relevance() {
    let j = HeuristicJudge;
    let text = "rust systems language programming";
    let scores = j.grade(text, text, &default_ctx()).unwrap();
    // Jaccard(text, text) = 1.0 → relevance = 1.0
    let relevance = scores.scores.get("relevance").copied().unwrap_or(0.0);
    assert!((relevance - 1.0_f32).abs() < 1e-6);
}

#[test]
fn heuristic_judge_with_matching_source_groundedness_positive() {
    let j = HeuristicJudge;
    let content = "rust programming systems";
    let src = make_result("s1", content);
    let ctx = JudgeContext::new().with_sources(vec![src]);
    let scores = j.grade("query", content, &ctx).unwrap();
    let g = scores.scores.get("groundedness").copied().unwrap_or(0.0);
    assert!(g > 0.0);
}

#[test]
fn heuristic_judge_scores_contain_relevance_key() {
    let j = HeuristicJudge;
    let scores = j.grade("query", "some answer", &default_ctx()).unwrap();
    assert!(scores.scores.contains_key("relevance"));
}

#[test]
fn heuristic_judge_scores_contain_groundedness_key() {
    let j = HeuristicJudge;
    let scores = j.grade("query", "some answer", &default_ctx()).unwrap();
    assert!(scores.scores.contains_key("groundedness"));
}

#[test]
fn heuristic_judge_long_answer_lower_conciseness() {
    let j = HeuristicJudge;
    let short = "concise answer";
    // Build a >300-word answer
    let long: String = "word ".repeat(350);
    let short_scores = j.grade("query", short, &default_ctx()).unwrap();
    let long_scores = j.grade("query", &long, &default_ctx()).unwrap();
    let s_conc = short_scores
        .scores
        .get("conciseness")
        .copied()
        .unwrap_or(0.0);
    let l_conc = long_scores
        .scores
        .get("conciseness")
        .copied()
        .unwrap_or(0.0);
    assert!(s_conc > l_conc);
}

// ── LlmJudge ─────────────────────────────────────────────────────────────

#[test]
fn llm_judge_new_stores_config() {
    let cfg = LlmJudgeConfig::default().with_tie_margin(0.2);
    let judge = LlmJudge::new(cfg);
    assert!((judge.config.tie_margin - 0.2_f32).abs() < 1e-6);
}

#[test]
fn llm_judge_default_created() {
    let judge = LlmJudge::default();
    assert!((judge.config.pointwise_scale - 5.0_f32).abs() < 1e-6);
}

#[test]
fn score_pointwise_empty_query_error() {
    let judge = LlmJudge::default();
    let res = judge.score_pointwise("", "some answer", &default_ctx());
    assert!(matches!(res, Err(LlmJudgeError::EmptyQuery)));
}

#[test]
fn score_pointwise_empty_answer_error() {
    let judge = LlmJudge::default();
    let res = judge.score_pointwise("query", "", &default_ctx());
    assert!(matches!(res, Err(LlmJudgeError::EmptyAnswer)));
}

#[test]
fn score_pointwise_empty_rubric_error() {
    let cfg = LlmJudgeConfig::default().with_rubric(Rubric::new(vec![]));
    let judge = LlmJudge::new(cfg);
    let res = judge.score_pointwise("query", "answer", &default_ctx());
    assert!(matches!(res, Err(LlmJudgeError::EmptyRubric)));
}

#[test]
fn score_pointwise_score_in_scale_range() {
    let judge = LlmJudge::default();
    let v: PointwiseVerdict = judge
        .score_pointwise(
            "What is Rust?",
            "Rust is a systems language.",
            &default_ctx(),
        )
        .unwrap();
    let scale = judge.config.pointwise_scale;
    assert!((0.0..=scale).contains(&v.score));
}

#[test]
fn score_pointwise_score_non_negative() {
    let judge = LlmJudge::default();
    let v = judge
        .score_pointwise("query", "answer", &default_ctx())
        .unwrap();
    assert!(v.score >= 0.0);
}

#[test]
fn score_pointwise_rationale_non_empty() {
    let judge = LlmJudge::default();
    let v = judge
        .score_pointwise("query", "answer", &default_ctx())
        .unwrap();
    assert!(!v.rationale.is_empty());
}

#[test]
fn score_pointwise_with_correctness_rubric() {
    let cfg = LlmJudgeConfig::default().with_rubric(Rubric::correctness());
    let judge = LlmJudge::new(cfg);
    let v = judge
        .score_pointwise("query", "correct answer here", &default_ctx())
        .unwrap();
    assert!(v.score >= 0.0);
}

// ── compare ───────────────────────────────────────────────────────────────

#[test]
fn compare_empty_query_error() {
    let judge = LlmJudge::default();
    let res = judge.compare("", "a1", "a2", &default_ctx());
    assert!(matches!(res, Err(LlmJudgeError::EmptyQuery)));
}

#[test]
fn compare_identical_answers_tie() {
    let judge = LlmJudge::default();
    let v: PairwiseVerdict = judge
        .compare(
            "query",
            "same answer text",
            "same answer text",
            &default_ctx(),
        )
        .unwrap();
    assert_eq!(v.winner, Pref::Tie);
}

#[test]
fn compare_a_better_than_b() {
    // A perfectly matches the query; B is irrelevant noise
    let judge = LlmJudge::new(LlmJudgeConfig::default().with_tie_margin(0.0));
    let query = "rust programming language systems";
    let answer_a = "rust programming language systems"; // same tokens
    let answer_b = "banana pineapple mango fruit tropical"; // no overlap
    let v = judge
        .compare(query, answer_a, answer_b, &default_ctx())
        .unwrap();
    assert_eq!(v.winner, Pref::A);
}

#[test]
fn compare_b_better_than_a() {
    let judge = LlmJudge::new(LlmJudgeConfig::default().with_tie_margin(0.0));
    let query = "rust programming language systems";
    let answer_a = "banana pineapple mango fruit tropical";
    let answer_b = "rust programming language systems";
    let v = judge
        .compare(query, answer_a, answer_b, &default_ctx())
        .unwrap();
    assert_eq!(v.winner, Pref::B);
}

#[test]
fn compare_margin_reflects_score_difference() {
    let judge = LlmJudge::default();
    let v = judge
        .compare("rust", "rust language", "rust language", &default_ctx())
        .unwrap();
    // Identical scores → margin should be 0
    assert!((v.margin - 0.0_f32).abs() < 1e-5);
}

#[test]
fn compare_within_tie_margin_gives_tie() {
    // Default tie_margin = 0.05; use a large margin so scores are very close
    let judge = LlmJudge::new(LlmJudgeConfig::default().with_tie_margin(1.0));
    let v = judge
        .compare("query", "answer one", "answer two", &default_ctx())
        .unwrap();
    assert_eq!(v.winner, Pref::Tie);
}

// ── score_with_reference ──────────────────────────────────────────────────

#[test]
fn score_with_reference_attaches_reference_to_context() {
    let judge = LlmJudge::default();
    let v = judge
        .score_with_reference("query", "answer", "reference answer", &default_ctx())
        .unwrap();
    // Should succeed and return a PointwiseVerdict
    assert!(v.score >= 0.0);
}

#[test]
fn score_with_reference_returns_same_type_as_pointwise() {
    let judge = LlmJudge::default();
    let v: PointwiseVerdict = judge
        .score_with_reference("query", "answer text", "reference text", &default_ctx())
        .unwrap();
    assert!(!v.rationale.is_empty());
}

#[test]
fn score_with_reference_empty_answer_error() {
    let judge = LlmJudge::default();
    let res = judge.score_with_reference("query", "", "reference", &default_ctx());
    assert!(matches!(res, Err(LlmJudgeError::EmptyAnswer)));
}

// ── LlmJudgeError display ─────────────────────────────────────────────────

#[test]
fn llm_judge_error_empty_answer_display() {
    let e = LlmJudgeError::EmptyAnswer;
    assert!(!e.to_string().is_empty());
}

#[test]
fn llm_judge_error_empty_rubric_display() {
    let e = LlmJudgeError::EmptyRubric;
    assert!(!e.to_string().is_empty());
}

#[test]
fn llm_judge_error_empty_query_display() {
    let e = LlmJudgeError::EmptyQuery;
    assert!(!e.to_string().is_empty());
}

// ── CriterionScores ───────────────────────────────────────────────────────

#[test]
fn criterion_scores_default_overall_zero() {
    let cs = CriterionScores::default();
    assert!((cs.overall - 0.0_f32).abs() < 1e-6);
}

#[test]
fn criterion_scores_default_scores_empty() {
    let cs = CriterionScores::default();
    assert!(cs.scores.is_empty());
}

// ── PointwiseVerdict score non-negative ───────────────────────────────────

#[test]
fn pointwise_verdict_per_criterion_has_relevance_after_grading() {
    let judge = LlmJudge::default();
    let v = judge
        .score_pointwise("rust programming", "rust language", &default_ctx())
        .unwrap();
    assert!(v.per_criterion.scores.contains_key("relevance"));
}
