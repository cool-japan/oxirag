#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
//! Tests for the `extractive_qa` module.
//!
//! Coverage strategy: adversarial "type A beats type B" claims are checked
//! two ways — precisely, via direct [`ExtractiveQaEngine::score_span`] calls
//! on two hand-picked candidates (so the assertion does not depend on
//! reasoning about every candidate a full passage could generate), and
//! qualitatively, via end-to-end [`ExtractiveQaEngine::answer`] calls that
//! assert a *property* of the winning text (contains a digit, is
//! capitalized, etc.) rather than an exact string. Component-isolation tests
//! (context overlap, length prior) hold two of the three `SpanScore` signals
//! fixed and vary only the third, so the differentiator under test is
//! unambiguous.

use super::engine::ExtractiveQaEngine;
use super::types::{
    AnswerSpan, ExtractiveQaConfig, ExtractiveQaError, ExtractiveQaResult, QuestionType,
    SpanCandidate, SpanScore,
};

// ── test fixtures ─────────────────────────────────────────────────────────────

/// tokens: 0=Isaac 1=Newton 2=formulated 3=the 4=theory 5=in 6=1687 7=during
/// 8=a 9=plague 10=year. (11 tokens)
const NEWTON_PASSAGE: &str = "Isaac Newton formulated the theory in 1687 during a plague year.";

/// tokens: 0=Officials 1=met 2=near 3=Geneva 4=to 5=discuss 6=the 7=treaty.
/// (8 tokens)
const GENEVA_PASSAGE: &str = "Officials met near Geneva to discuss the treaty.";

/// tokens: 0=The 1=team 2=found 3=ancient 4=pottery 5=inside 6=a 7=remote
/// 8=cave 9=last 10=Tuesday. (11 tokens)
const CAVE_PASSAGE: &str = "The team found ancient pottery inside a remote cave last Tuesday.";

/// tokens: 0=The 1=committee 2=approve 3=zzqux 4=after 5=lengthy 6=debate
/// 7=today. 8=Random 9=filler 10=banter 11=chatter 12=zzqux 13=nonsense
/// 14=babble 15=here. (16 tokens)
const COMMITTEE_PASSAGE: &str = "The committee approve zzqux after lengthy debate today. \
     Random filler banter chatter zzqux nonsense babble here.";

/// tokens: 0=The 1=necklace 2=once 3=belonged 4=to 5=Queen 6=Victoria 7=and
/// 8=cost 9=$500. (10 tokens)
const NECKLACE_PASSAGE: &str = "The necklace once belonged to Queen Victoria and cost $500.";

fn engine() -> ExtractiveQaEngine {
    ExtractiveQaEngine::new(ExtractiveQaConfig::default())
}

fn engine_with(config: ExtractiveQaConfig) -> ExtractiveQaEngine {
    ExtractiveQaEngine::new(config)
}

// ── QuestionType classification ─────────────────────────────────────────────

#[test]
fn test_classify_who() {
    assert_eq!(
        engine().classify_question("Who discovered radium?"),
        QuestionType::Who
    );
}

#[test]
fn test_classify_what() {
    assert_eq!(
        engine().classify_question("What is the capital of France?"),
        QuestionType::What
    );
}

#[test]
fn test_classify_when() {
    assert_eq!(
        engine().classify_question("When did the war end?"),
        QuestionType::When
    );
}

#[test]
fn test_classify_where() {
    assert_eq!(
        engine().classify_question("Where is the Eiffel Tower located?"),
        QuestionType::Where
    );
}

#[test]
fn test_classify_how_many() {
    assert_eq!(
        engine().classify_question("How many people attended the event?"),
        QuestionType::HowMany
    );
}

#[test]
fn test_classify_how_much() {
    assert_eq!(
        engine().classify_question("How much did the ticket cost?"),
        QuestionType::HowMuch
    );
}

#[test]
fn test_classify_why() {
    assert_eq!(
        engine().classify_question("Why did the treaty fail?"),
        QuestionType::Why
    );
}

#[test]
fn test_classify_which() {
    assert_eq!(
        engine().classify_question("Which country won the war?"),
        QuestionType::Which
    );
}

#[test]
fn test_classify_other_no_keyword() {
    assert_eq!(
        engine().classify_question("Tell me about the history of Rome."),
        QuestionType::Other
    );
}

#[test]
fn test_classify_bare_how_is_other() {
    // "how" alone (without "many"/"much") is not a keyword for any type.
    assert_eq!(
        engine().classify_question("How did the war end?"),
        QuestionType::Other
    );
}

#[test]
fn test_classify_leftmost_match_wins() {
    // "who" (position 0) precedes "what" (later) -> Who.
    assert_eq!(
        engine().classify_question("Who knows what happened?"),
        QuestionType::Who
    );
}

#[test]
fn test_classify_leftmost_match_wins_reversed() {
    // "what" (position 0) precedes "who" (later) -> What.
    assert_eq!(
        engine().classify_question("What did who say?"),
        QuestionType::What
    );
}

#[test]
fn test_classify_case_insensitive() {
    assert_eq!(
        engine().classify_question("WHO directed the film?"),
        QuestionType::Who
    );
}

#[test]
fn test_classify_empty_question_is_other() {
    assert_eq!(engine().classify_question(""), QuestionType::Other);
}

#[test]
fn test_classify_whitespace_question_is_other() {
    assert_eq!(engine().classify_question("   "), QuestionType::Other);
}

#[test]
fn test_classify_overridable_keyword_single_type() {
    let config = ExtractiveQaConfig::default()
        .with_keywords_for(QuestionType::Why, vec!["wherefore".to_string()]);
    let eng = engine_with(config);
    assert_eq!(
        eng.classify_question("Wherefore art thou Romeo?"),
        QuestionType::Why
    );
}

#[test]
fn test_classify_overridable_keyword_does_not_disturb_other_types() {
    let config = ExtractiveQaConfig::default()
        .with_keywords_for(QuestionType::Why, vec!["wherefore".to_string()]);
    let eng = engine_with(config);
    assert_eq!(eng.classify_question("Who is Romeo?"), QuestionType::Who);
}

#[test]
fn test_classify_with_type_keywords_replaces_whole_table() {
    let config = ExtractiveQaConfig::default()
        .with_type_keywords(vec![(QuestionType::Who, vec!["who".to_string()])]);
    let eng = engine_with(config);
    // "what" is no longer recognised after the full table replacement.
    assert_eq!(eng.classify_question("What is this?"), QuestionType::Other);
    assert_eq!(eng.classify_question("Who is this?"), QuestionType::Who);
}

#[test]
fn test_with_keywords_for_appends_when_type_absent() {
    let config = ExtractiveQaConfig::default()
        .with_type_keywords(Vec::new())
        .with_keywords_for(QuestionType::Who, vec!["who".to_string()]);
    assert_eq!(config.type_keywords.len(), 1);
    assert_eq!(config.type_keywords[0].0, QuestionType::Who);
}

#[test]
fn test_classify_question_is_deterministic() {
    let eng = engine();
    let first = eng.classify_question("Who formulated the theory?");
    for _ in 0..5 {
        assert_eq!(eng.classify_question("Who formulated the theory?"), first);
    }
}

// ── QuestionType::as_str ─────────────────────────────────────────────────────

#[test]
fn test_question_type_as_str_all_variants() {
    let cases = [
        (QuestionType::Who, "who"),
        (QuestionType::What, "what"),
        (QuestionType::When, "when"),
        (QuestionType::Where, "where"),
        (QuestionType::HowMany, "how_many"),
        (QuestionType::HowMuch, "how_much"),
        (QuestionType::Why, "why"),
        (QuestionType::Which, "which"),
        (QuestionType::Other, "other"),
    ];
    for (question_type, expected) in cases {
        assert_eq!(question_type.as_str(), expected);
    }
}

// ── SpanCandidate / AnswerSpan ───────────────────────────────────────────────

#[test]
fn test_span_candidate_new_and_fields() {
    let candidate = SpanCandidate::new(2, 5, "hello world foo");
    assert_eq!(candidate.start_token, 2);
    assert_eq!(candidate.end_token, 5);
    assert_eq!(candidate.text, "hello world foo");
}

#[test]
fn test_span_candidate_len_tokens() {
    assert_eq!(SpanCandidate::new(2, 5, "hello world foo").len_tokens(), 3);
}

#[test]
fn test_span_candidate_len_tokens_saturates_on_malformed_bounds() {
    assert_eq!(SpanCandidate::new(5, 2, "malformed").len_tokens(), 0);
}

#[test]
fn test_answer_span_len_tokens() {
    let span = AnswerSpan {
        text: "Marie Curie".to_string(),
        start_token: 0,
        end_token: 2,
        question_type: QuestionType::Who,
        score: SpanScore {
            type_match: 1.0,
            context_overlap: 0.5,
            length_prior: 0.9,
            total: 0.8,
        },
    };
    assert_eq!(span.len_tokens(), 2);
}

// ── ExtractiveQaConfig: defaults ─────────────────────────────────────────────

#[test]
fn test_config_default_max_span_tokens() {
    assert_eq!(ExtractiveQaConfig::default().max_span_tokens, 10);
}

#[test]
fn test_config_default_context_window_tokens() {
    assert_eq!(ExtractiveQaConfig::default().context_window_tokens, 6);
}

#[test]
fn test_config_default_weights() {
    let config = ExtractiveQaConfig::default();
    assert_eq!(config.weight_type_match, 0.5);
    assert_eq!(config.weight_context_overlap, 0.3);
    assert_eq!(config.weight_length_prior, 0.2);
}

#[test]
fn test_config_default_length_prior_decay() {
    assert_eq!(ExtractiveQaConfig::default().length_prior_decay, 0.15);
}

#[test]
fn test_config_default_type_keywords_cover_eight_types() {
    let config = ExtractiveQaConfig::default();
    let covered: std::collections::HashSet<QuestionType> =
        config.type_keywords.iter().map(|(t, _)| *t).collect();
    for expected in [
        QuestionType::Who,
        QuestionType::What,
        QuestionType::When,
        QuestionType::Where,
        QuestionType::HowMany,
        QuestionType::HowMuch,
        QuestionType::Why,
        QuestionType::Which,
    ] {
        assert!(covered.contains(&expected), "missing {expected:?}");
    }
    assert!(!covered.contains(&QuestionType::Other));
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(ExtractiveQaConfig::new(), ExtractiveQaConfig::default());
}

// ── ExtractiveQaConfig: builders ─────────────────────────────────────────────

#[test]
fn test_config_with_max_span_tokens() {
    assert_eq!(
        ExtractiveQaConfig::new()
            .with_max_span_tokens(4)
            .max_span_tokens,
        4
    );
}

#[test]
fn test_config_with_context_window_tokens() {
    assert_eq!(
        ExtractiveQaConfig::new()
            .with_context_window_tokens(2)
            .context_window_tokens,
        2
    );
}

#[test]
fn test_config_with_weight_type_match() {
    assert_eq!(
        ExtractiveQaConfig::new()
            .with_weight_type_match(0.9)
            .weight_type_match,
        0.9
    );
}

#[test]
fn test_config_with_weight_context_overlap() {
    assert_eq!(
        ExtractiveQaConfig::new()
            .with_weight_context_overlap(0.1)
            .weight_context_overlap,
        0.1
    );
}

#[test]
fn test_config_with_weight_length_prior() {
    assert_eq!(
        ExtractiveQaConfig::new()
            .with_weight_length_prior(0.05)
            .weight_length_prior,
        0.05
    );
}

#[test]
fn test_config_with_length_prior_decay() {
    assert_eq!(
        ExtractiveQaConfig::new()
            .with_length_prior_decay(0.5)
            .length_prior_decay,
        0.5
    );
}

#[test]
fn test_config_builder_chain() {
    let config = ExtractiveQaConfig::new()
        .with_max_span_tokens(3)
        .with_context_window_tokens(2)
        .with_weight_type_match(0.6)
        .with_weight_context_overlap(0.25)
        .with_weight_length_prior(0.15)
        .with_length_prior_decay(0.2);
    assert_eq!(config.max_span_tokens, 3);
    assert_eq!(config.context_window_tokens, 2);
    assert_eq!(config.weight_type_match, 0.6);
    assert_eq!(config.weight_context_overlap, 0.25);
    assert_eq!(config.weight_length_prior, 0.15);
    assert_eq!(config.length_prior_decay, 0.2);
}

// ── generate_candidates ──────────────────────────────────────────────────────

#[test]
fn test_generate_candidates_empty_passage_errors() {
    assert_eq!(
        engine().generate_candidates(""),
        Err(ExtractiveQaError::EmptyPassage)
    );
}

#[test]
fn test_generate_candidates_whitespace_only_passage_errors() {
    assert_eq!(
        engine().generate_candidates("   \n\t  "),
        Err(ExtractiveQaError::EmptyPassage)
    );
}

#[test]
fn test_generate_candidates_single_token() {
    let candidates = engine().generate_candidates("Hello").unwrap();
    assert_eq!(candidates, vec![SpanCandidate::new(0, 1, "Hello")]);
}

#[test]
fn test_generate_candidates_full_coverage_when_cap_not_binding() {
    // 3 tokens, cap >= 3: every contiguous span is generated: 3+2+1 = 6.
    let candidates = engine().generate_candidates("A B C").unwrap();
    assert_eq!(candidates.len(), 6);
}

#[test]
fn test_generate_candidates_respects_max_span_tokens_cap() {
    let config = ExtractiveQaConfig::new().with_max_span_tokens(3);
    let passage = "one two three four five six seven eight nine ten";
    let candidates = engine_with(config).generate_candidates(passage).unwrap();
    assert!(!candidates.is_empty());
    assert!(candidates.iter().all(|c| c.len_tokens() <= 3));
}

#[test]
fn test_generate_candidates_cap_zero_yields_no_candidates() {
    let config = ExtractiveQaConfig::new().with_max_span_tokens(0);
    let candidates = engine_with(config)
        .generate_candidates("some passage text")
        .unwrap();
    assert!(candidates.is_empty());
}

#[test]
fn test_generate_candidates_bounded_count_formula() {
    // 5 tokens, cap 3: sum_{start=0}^{4} min(3, 5-start) = 3+3+3+2+1 = 12.
    let config = ExtractiveQaConfig::new().with_max_span_tokens(3);
    let candidates = engine_with(config)
        .generate_candidates("t0 t1 t2 t3 t4")
        .unwrap();
    assert_eq!(candidates.len(), 12);
}

#[test]
fn test_generate_candidates_no_duplicate_bounds() {
    let candidates = engine().generate_candidates("A B C D").unwrap();
    let mut seen = std::collections::HashSet::new();
    for c in &candidates {
        assert!(
            seen.insert((c.start_token, c.end_token)),
            "duplicate span bounds {:?}..{:?}",
            c.start_token,
            c.end_token
        );
    }
}

#[test]
fn test_generate_candidates_text_matches_token_join() {
    let candidates = engine().generate_candidates("alpha beta gamma").unwrap();
    let hit = candidates
        .iter()
        .find(|c| c.start_token == 1 && c.end_token == 3)
        .unwrap();
    assert_eq!(hit.text, "beta gamma");
}

// ── score_span: validation errors ───────────────────────────────────────────

#[test]
fn test_score_span_empty_question_errors() {
    let candidate = SpanCandidate::new(0, 1, "Hello");
    assert_eq!(
        engine().score_span("", "Hello world", &candidate),
        Err(ExtractiveQaError::EmptyQuestion)
    );
}

#[test]
fn test_score_span_empty_passage_errors() {
    let candidate = SpanCandidate::new(0, 1, "Hello");
    assert_eq!(
        engine().score_span("Who?", "", &candidate),
        Err(ExtractiveQaError::EmptyPassage)
    );
}

#[test]
fn test_score_span_invalid_bounds_start_ge_end() {
    let candidate = SpanCandidate::new(2, 2, "");
    assert_eq!(
        engine().score_span("Who?", "Hello world foo", &candidate),
        Err(ExtractiveQaError::InvalidSpanBounds {
            start_token: 2,
            end_token: 2,
            passage_tokens: 3,
        })
    );
}

#[test]
fn test_score_span_invalid_bounds_end_exceeds_passage() {
    let candidate = SpanCandidate::new(0, 10, "out of range");
    assert_eq!(
        engine().score_span("Who?", "Hello world foo", &candidate),
        Err(ExtractiveQaError::InvalidSpanBounds {
            start_token: 0,
            end_token: 10,
            passage_tokens: 3,
        })
    );
}

// ── score_span: question-type content-match, direct pairwise comparisons ───

#[test]
fn test_type_match_who_capitalized_beats_numeric() {
    let eng = engine();
    let question = "Who formulated the theory?";
    let capitalized = SpanCandidate::new(0, 2, "Isaac Newton");
    let numeric = SpanCandidate::new(6, 7, "1687");

    let score_capitalized = eng
        .score_span(question, NEWTON_PASSAGE, &capitalized)
        .unwrap();
    let score_numeric = eng.score_span(question, NEWTON_PASSAGE, &numeric).unwrap();

    assert_eq!(score_capitalized.type_match, 1.0);
    assert_eq!(score_numeric.type_match, 0.0);
    assert!(score_capitalized.total > score_numeric.total);
}

#[test]
fn test_type_match_how_many_numeric_beats_capitalized() {
    let eng = engine();
    let question = "How many works did Newton write before 1687?";
    let numeric = SpanCandidate::new(6, 7, "1687");
    let capitalized = SpanCandidate::new(0, 2, "Isaac Newton");

    let score_numeric = eng.score_span(question, NEWTON_PASSAGE, &numeric).unwrap();
    let score_capitalized = eng
        .score_span(question, NEWTON_PASSAGE, &capitalized)
        .unwrap();

    assert_eq!(score_numeric.type_match, 1.0);
    assert_eq!(score_capitalized.type_match, 0.0);
    assert!(score_numeric.total > score_capitalized.total);
}

#[test]
fn test_type_match_when_year_beats_capitalized() {
    let eng = engine();
    let question = "When was the theory formulated?";
    let year = SpanCandidate::new(6, 7, "1687");
    let capitalized = SpanCandidate::new(0, 2, "Isaac Newton");

    let score_year = eng.score_span(question, NEWTON_PASSAGE, &year).unwrap();
    let score_capitalized = eng
        .score_span(question, NEWTON_PASSAGE, &capitalized)
        .unwrap();

    assert_eq!(score_year.type_match, 1.0);
    assert_eq!(score_capitalized.type_match, 0.0);
    assert!(score_year.total > score_capitalized.total);
}

#[test]
fn test_type_match_where_prepositional_context_beats_plain_capitalized() {
    let eng = engine();
    let question = "Where was the meeting held?";
    let after_near = SpanCandidate::new(3, 4, "Geneva");
    let sentence_initial_capital = SpanCandidate::new(0, 1, "Officials");

    let score_geneva = eng
        .score_span(question, GENEVA_PASSAGE, &after_near)
        .unwrap();
    let score_officials = eng
        .score_span(question, GENEVA_PASSAGE, &sentence_initial_capital)
        .unwrap();

    // Both spans are equally capitalized; only the preposition-preceded
    // span gets the extra prepositional-phrase-context bonus.
    assert_eq!(score_geneva.type_match, 1.0);
    assert_eq!(score_officials.type_match, 0.5);
    assert!(score_geneva.total > score_officials.total);
}

#[test]
fn test_type_match_what_lexical_overlap_span_matching_question_wins() {
    let eng = engine();
    let question = "What did the scientists discover in the cave?";
    let matching = SpanCandidate::new(8, 9, "cave");
    let unrelated = SpanCandidate::new(10, 11, "Tuesday.");

    let score_matching = eng.score_span(question, CAVE_PASSAGE, &matching).unwrap();
    let score_unrelated = eng.score_span(question, CAVE_PASSAGE, &unrelated).unwrap();

    assert!(score_matching.type_match > score_unrelated.type_match);
    assert!(score_matching.total > score_unrelated.total);
}

#[test]
fn test_type_match_why_uses_lexical_overlap_like_what() {
    let eng = engine();
    let question = "Why was the pottery ancient?";
    let matching = SpanCandidate::new(3, 5, "ancient pottery");
    let unrelated = SpanCandidate::new(9, 11, "last Tuesday.");

    let score_matching = eng.score_span(question, CAVE_PASSAGE, &matching).unwrap();
    let score_unrelated = eng.score_span(question, CAVE_PASSAGE, &unrelated).unwrap();

    assert_eq!(score_matching.type_match, 1.0);
    assert_eq!(score_unrelated.type_match, 0.0);
}

// ── context overlap: isolation ───────────────────────────────────────────────

#[test]
fn test_context_overlap_prefers_span_in_question_relevant_context() {
    let eng = engine();
    let question = "What did the committee approve?";
    let near_relevant_context = SpanCandidate::new(3, 4, "zzqux"); // near "committee approve"
    let near_irrelevant_context = SpanCandidate::new(12, 13, "zzqux"); // near unrelated filler

    let score_near = eng
        .score_span(question, COMMITTEE_PASSAGE, &near_relevant_context)
        .unwrap();
    let score_far = eng
        .score_span(question, COMMITTEE_PASSAGE, &near_irrelevant_context)
        .unwrap();

    // Identical span text and length -> identical type_match (both spans
    // are the literal word "zzqux", which itself echoes no question content
    // word) and identical length_prior. Only the surrounding context
    // differs, so it alone must explain the total difference.
    assert_eq!(score_near.type_match, score_far.type_match);
    assert_eq!(score_near.length_prior, score_far.length_prior);
    assert!(score_near.context_overlap > score_far.context_overlap);
    assert!(score_near.total > score_far.total);
}

// ── length prior ──────────────────────────────────────────────────────────────

#[test]
fn test_length_prior_isolation_via_empty_question_content() {
    let eng = engine();
    // "is"/"that" carry no usable content words (stopwords/too-short), so
    // question_content is empty and both type_match (the `What`
    // lexical-overlap fallback) and context_overlap are identically 0.0 for
    // *every* candidate -- isolating length_prior as the sole determinant
    // of `total`.
    let question = "What is that?";
    assert_eq!(eng.classify_question(question), QuestionType::What);

    let short = SpanCandidate::new(0, 1, "Isaac");
    let long = SpanCandidate::new(0, 7, "Isaac Newton formulated the theory in 1687");

    let score_short = eng.score_span(question, NEWTON_PASSAGE, &short).unwrap();
    let score_long = eng.score_span(question, NEWTON_PASSAGE, &long).unwrap();

    assert_eq!(score_short.type_match, 0.0);
    assert_eq!(score_long.type_match, 0.0);
    assert_eq!(score_short.context_overlap, 0.0);
    assert_eq!(score_long.context_overlap, 0.0);
    assert!(score_short.length_prior > score_long.length_prior);
    assert!(score_short.total > score_long.total);
}

#[test]
fn test_length_prior_penalizes_engulfing_long_span_around_correct_short_answer() {
    let eng = engine();
    let question = "Who formulated the theory?";
    let short_correct_answer = SpanCandidate::new(0, 2, "Isaac Newton");
    let engulfing_long_span =
        SpanCandidate::new(0, 7, "Isaac Newton formulated the theory in 1687");

    let score_short = eng
        .score_span(question, NEWTON_PASSAGE, &short_correct_answer)
        .unwrap();
    let score_long = eng
        .score_span(question, NEWTON_PASSAGE, &engulfing_long_span)
        .unwrap();

    assert!(
        score_short.length_prior > score_long.length_prior,
        "a length-2 span should score a higher length prior than a length-7 span"
    );
    assert!(score_short.total > score_long.total);
}

#[test]
fn test_length_prior_strictly_decreases_with_length() {
    let eng = engine();
    let question = "What is that?"; // empty question_content -> isolates length_prior
    let mut previous = f32::INFINITY;
    for len in 1..=8usize {
        let candidate = SpanCandidate::new(0, len, "span");
        let score = eng
            .score_span(question, NEWTON_PASSAGE, &candidate)
            .unwrap();
        assert!(
            score.length_prior < previous,
            "length_prior should strictly decrease as span length grows (len={len})"
        );
        previous = score.length_prior;
    }
}

// ── score breakdown sanity ───────────────────────────────────────────────────

#[test]
fn test_score_total_matches_weighted_formula() {
    let config = ExtractiveQaConfig::default();
    let eng = engine_with(config.clone());
    let candidate = SpanCandidate::new(0, 2, "Isaac Newton");
    let score = eng
        .score_span("Who formulated the theory?", NEWTON_PASSAGE, &candidate)
        .unwrap();

    let expected_total = config.weight_type_match * score.type_match
        + config.weight_context_overlap * score.context_overlap
        + config.weight_length_prior * score.length_prior;

    assert!((score.total - expected_total).abs() < 1e-5);
}

// ── answer(): end-to-end adversarial (qualitative) ──────────────────────────

#[test]
fn test_answer_who_selects_capitalized_over_numeric_end_to_end() {
    let answer = engine()
        .answer("Who formulated the theory?", NEWTON_PASSAGE)
        .unwrap();
    assert_eq!(answer.question_type, QuestionType::Who);
    assert!(
        answer.text.chars().any(char::is_uppercase),
        "expected a capitalized answer, got {:?}",
        answer.text
    );
    assert!(
        !answer.text.chars().any(|c| c.is_ascii_digit()),
        "expected no digits in a Who answer, got {:?}",
        answer.text
    );
}

#[test]
fn test_answer_how_many_selects_numeric_over_capitalized_end_to_end() {
    let passage = "Isaac Newton wrote 3 major works before 1687.";
    let answer = engine()
        .answer("How many major works did Newton write?", passage)
        .unwrap();
    assert_eq!(answer.question_type, QuestionType::HowMany);
    assert!(
        answer.text.chars().any(|c| c.is_ascii_digit()),
        "expected a numeric answer, got {:?}",
        answer.text
    );
}

#[test]
fn test_answer_how_much_selects_numeric_over_capitalized_end_to_end() {
    let answer = engine()
        .answer("How much did the necklace cost?", NECKLACE_PASSAGE)
        .unwrap();
    assert_eq!(answer.question_type, QuestionType::HowMuch);
    assert!(
        answer.text.chars().any(|c| c.is_ascii_digit()),
        "expected a numeric answer, got {:?}",
        answer.text
    );
}

#[test]
fn test_answer_when_selects_year_end_to_end() {
    let answer = engine()
        .answer("When did Newton formulate the theory?", NEWTON_PASSAGE)
        .unwrap();
    assert_eq!(answer.question_type, QuestionType::When);
    assert!(
        answer.text.chars().any(|c| c.is_ascii_digit()),
        "expected a date/year answer, got {:?}",
        answer.text
    );
}

#[test]
fn test_answer_where_selects_location_after_preposition_end_to_end() {
    let answer = engine()
        .answer("Where was the meeting held?", GENEVA_PASSAGE)
        .unwrap();
    assert_eq!(answer.question_type, QuestionType::Where);
    assert!(
        answer.text.contains("Geneva"),
        "expected Geneva in the answer, got {:?}",
        answer.text
    );
}

// ── answer() / top_k_spans(): error paths ───────────────────────────────────

#[test]
fn test_answer_empty_question_errors() {
    assert_eq!(
        engine().answer("", NEWTON_PASSAGE),
        Err(ExtractiveQaError::EmptyQuestion)
    );
}

#[test]
fn test_answer_whitespace_question_errors() {
    assert_eq!(
        engine().answer("   ", NEWTON_PASSAGE),
        Err(ExtractiveQaError::EmptyQuestion)
    );
}

#[test]
fn test_answer_empty_passage_errors() {
    assert_eq!(
        engine().answer("Who?", ""),
        Err(ExtractiveQaError::EmptyPassage)
    );
}

#[test]
fn test_answer_whitespace_passage_errors() {
    assert_eq!(
        engine().answer("Who?", "\n\t "),
        Err(ExtractiveQaError::EmptyPassage)
    );
}

#[test]
fn test_answer_no_viable_candidates_when_max_span_tokens_zero() {
    let config = ExtractiveQaConfig::new().with_max_span_tokens(0);
    let result = engine_with(config).answer("Who?", NEWTON_PASSAGE);
    assert_eq!(
        result,
        Err(ExtractiveQaError::NoViableCandidates {
            max_span_tokens: 0,
            passage_tokens: 11,
        })
    );
}

#[test]
fn test_top_k_spans_empty_question_errors() {
    assert_eq!(
        engine().top_k_spans("", NEWTON_PASSAGE, 3),
        Err(ExtractiveQaError::EmptyQuestion)
    );
}

#[test]
fn test_top_k_spans_empty_passage_errors() {
    assert_eq!(
        engine().top_k_spans("Who?", "", 3),
        Err(ExtractiveQaError::EmptyPassage)
    );
}

#[test]
fn test_top_k_spans_no_viable_candidates_when_max_span_tokens_zero() {
    let config = ExtractiveQaConfig::new().with_max_span_tokens(0);
    let result = engine_with(config).top_k_spans("Who?", NEWTON_PASSAGE, 3);
    assert_eq!(
        result,
        Err(ExtractiveQaError::NoViableCandidates {
            max_span_tokens: 0,
            passage_tokens: 11,
        })
    );
}

// ── top_k_spans(): structural properties ────────────────────────────────────

#[test]
fn test_top_k_spans_first_matches_answer() {
    let eng = engine();
    let question = "Who formulated the theory?";
    let via_answer = eng.answer(question, NEWTON_PASSAGE).unwrap();
    let via_top_k = eng.top_k_spans(question, NEWTON_PASSAGE, 5).unwrap();
    assert_eq!(via_top_k[0], via_answer);
}

#[test]
fn test_top_k_spans_descending_order() {
    let results = engine()
        .top_k_spans("Who formulated the theory?", NEWTON_PASSAGE, 10)
        .unwrap();
    assert!(results.len() > 1);
    for pair in results.windows(2) {
        assert!(pair[0].score.total >= pair[1].score.total);
    }
}

#[test]
fn test_top_k_spans_no_duplicate_offsets() {
    let results = engine()
        .top_k_spans("Who formulated the theory?", NEWTON_PASSAGE, 50)
        .unwrap();
    let mut seen = std::collections::HashSet::new();
    for r in &results {
        assert!(
            seen.insert((r.start_token, r.end_token)),
            "duplicate span offsets in top_k_spans"
        );
    }
}

#[test]
fn test_top_k_spans_respects_k() {
    let results = engine()
        .top_k_spans("Who formulated the theory?", NEWTON_PASSAGE, 4)
        .unwrap();
    assert_eq!(results.len(), 4);
}

#[test]
fn test_top_k_spans_k_zero_returns_empty() {
    let results = engine()
        .top_k_spans("Who formulated the theory?", NEWTON_PASSAGE, 0)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_top_k_spans_k_larger_than_candidates_returns_all() {
    let config = ExtractiveQaConfig::new().with_max_span_tokens(3);
    let candidate_count = engine_with(config.clone())
        .generate_candidates("one two three")
        .unwrap()
        .len();
    let results = engine_with(config)
        .top_k_spans("What is that?", "one two three", 1_000)
        .unwrap();
    assert_eq!(results.len(), candidate_count);
}

#[test]
fn test_top_k_spans_all_share_same_question_type() {
    let results = engine()
        .top_k_spans("Who formulated the theory?", NEWTON_PASSAGE, 20)
        .unwrap();
    assert!(results.iter().all(|r| r.question_type == QuestionType::Who));
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_answer_is_deterministic() {
    let eng = engine();
    let a = eng
        .answer("Who formulated the theory?", NEWTON_PASSAGE)
        .unwrap();
    let b = eng
        .answer("Who formulated the theory?", NEWTON_PASSAGE)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_top_k_spans_is_deterministic() {
    let eng = engine();
    let a = eng
        .top_k_spans("Who formulated the theory?", NEWTON_PASSAGE, 5)
        .unwrap();
    let b = eng
        .top_k_spans("Who formulated the theory?", NEWTON_PASSAGE, 5)
        .unwrap();
    assert_eq!(a, b);
}

// ── answer text / passage consistency ────────────────────────────────────────

#[test]
fn test_answer_text_is_a_verbatim_token_span_of_the_passage() {
    let eng = engine();
    let answer = eng
        .answer("Who formulated the theory?", NEWTON_PASSAGE)
        .unwrap();
    let passage_tokens: Vec<&str> = NEWTON_PASSAGE.split_whitespace().collect();
    let expected = passage_tokens[answer.start_token..answer.end_token].join(" ");
    assert_eq!(answer.text, expected);
}

#[test]
fn test_answer_span_question_type_matches_classify_question() {
    let eng = engine();
    let question = "Who formulated the theory?";
    let answer = eng.answer(question, NEWTON_PASSAGE).unwrap();
    assert_eq!(answer.question_type, eng.classify_question(question));
}

#[test]
fn test_answer_respects_configured_max_span_tokens() {
    let config = ExtractiveQaConfig::new().with_max_span_tokens(2);
    let answer = engine_with(config)
        .answer("Who formulated the theory?", NEWTON_PASSAGE)
        .unwrap();
    assert!(answer.len_tokens() <= 2);
}

// ── ExtractiveQaResult alias sanity ──────────────────────────────────────────

#[test]
fn test_extractive_qa_result_alias_is_usable() {
    let ok: ExtractiveQaResult<i32> = Ok(42);
    let err: ExtractiveQaResult<i32> = Err(ExtractiveQaError::EmptyQuestion);
    assert_eq!(ok, Ok(42));
    assert!(err.is_err());
}
