#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]
//! Tests for the `multi_agent_debate` module.

use std::cell::RefCell;

use super::engine::{
    DebateEngine, MockDebateJudge, MockDebatePersona, content_terms, fnv1a, jaccard, salient_term,
};
use super::types::{
    DebateArgument, DebateConfig, DebateError, DebateJudge, DebateJudgeWeights, DebateParticipant,
    DebatePersona, DebatePositionScore, DebateResult, DebateRound, DebateVerdict,
};

// ── Test helpers ─────────────────────────────────────────────────────────────

/// Build a [`DebateArgument`] with every bookkeeping field spelled out, for
/// hand-crafted [`DebateJudge`] transcripts.
fn arg(
    persona_index: usize,
    position: &str,
    round: usize,
    text: &str,
    confidence: f32,
) -> DebateArgument {
    DebateArgument {
        persona_index,
        position: position.to_string(),
        round,
        text: text.to_string(),
        confidence,
    }
}

/// A minimal, non-zero [`DebatePositionScore`] for [`DebateVerdict`]-level
/// tests that don't care about the exact scoring math.
fn sample_score(persona_index: usize, position: &str, total_score: f32) -> DebatePositionScore {
    DebatePositionScore {
        persona_index,
        position: position.to_string(),
        argument_count: 1,
        normalized_argument_count: 1.0,
        average_confidence: 0.5,
        rebuttal_engagement: 0.5,
        total_score,
    }
}

/// Two-participant helper for [`DebateEngine`] integration tests.
fn two_participants<'a>(
    persona_a: &'a dyn DebatePersona,
    position_a: &'a str,
    persona_b: &'a dyn DebatePersona,
    position_b: &'a str,
) -> [DebateParticipant<'a>; 2] {
    [
        DebateParticipant::new(persona_a, position_a),
        DebateParticipant::new(persona_b, position_b),
    ]
}

/// Records every `prior_arguments` slice it was called with (length and full
/// snapshot), so tests can verify the engine hands each persona the
/// **complete** prior transcript rather than just its own history.
#[derive(Default)]
struct RecordingDebatePersona {
    seen_lengths: RefCell<Vec<usize>>,
    seen_snapshots: RefCell<Vec<Vec<DebateArgument>>>,
}

impl DebatePersona for RecordingDebatePersona {
    fn argue(
        &self,
        _question: &str,
        position: &str,
        prior_arguments: &[DebateArgument],
    ) -> Result<DebateArgument, DebateError> {
        self.seen_lengths.borrow_mut().push(prior_arguments.len());
        self.seen_snapshots
            .borrow_mut()
            .push(prior_arguments.to_vec());
        Ok(DebateArgument::new(format!("argument for {position}"), 0.5))
    }
}

/// Returns confidences from a fixed, caller-supplied script (indexed by the
/// number of the persona's own prior arguments, i.e. its round), for tests
/// that need to control the exact confidence trajectory driving early stop.
struct ScriptedDebatePersona {
    confidences: Vec<f32>,
}

impl DebatePersona for ScriptedDebatePersona {
    fn argue(
        &self,
        _question: &str,
        position: &str,
        prior_arguments: &[DebateArgument],
    ) -> Result<DebateArgument, DebateError> {
        let round = prior_arguments
            .iter()
            .filter(|a| a.position == position)
            .count();
        let confidence = if self.confidences.is_empty() {
            0.5
        } else {
            self.confidences[round % self.confidences.len()]
        };
        Ok(DebateArgument::new(
            format!("scripted round {round} for {position}"),
            confidence,
        ))
    }
}

/// A [`DebatePersona`] that always fails, for error-propagation tests.
struct FailingDebatePersona;

impl DebatePersona for FailingDebatePersona {
    fn argue(
        &self,
        _question: &str,
        _position: &str,
        _prior_arguments: &[DebateArgument],
    ) -> Result<DebateArgument, DebateError> {
        Err(DebateError::PersonaGenerationFailed {
            reason: "boom".to_string(),
        })
    }
}

/// A [`DebateJudge`] that always fails, for error-propagation tests.
struct FailingDebateJudge;

impl DebateJudge for FailingDebateJudge {
    fn judge(
        &self,
        _question: &str,
        _transcript: &[DebateRound],
    ) -> Result<DebateVerdict, DebateError> {
        Err(DebateError::JudgeFailed {
            reason: "boom".to_string(),
        })
    }
}

/// A [`DebatePersona`] that returns deliberately wrong bookkeeping fields
/// (but valid text), to verify [`DebateEngine::run`] overwrites them with
/// ground truth regardless.
struct SloppyDebatePersona;

impl DebatePersona for SloppyDebatePersona {
    fn argue(
        &self,
        _question: &str,
        position: &str,
        _prior_arguments: &[DebateArgument],
    ) -> Result<DebateArgument, DebateError> {
        Ok(DebateArgument {
            persona_index: 999,
            position: "WRONG POSITION".to_string(),
            round: 999,
            text: format!("sloppy argument for {position}"),
            confidence: 0.5,
        })
    }
}

/// A [`DebatePersona`] that returns an out-of-`[0, 1]`-range confidence
/// (bypassing [`DebateArgument::new`]'s own clamp), to verify
/// [`DebateEngine::run`] defensively clamps it anyway.
struct OutOfRangeConfidencePersona {
    confidence: f32,
}

impl DebatePersona for OutOfRangeConfidencePersona {
    fn argue(
        &self,
        _question: &str,
        position: &str,
        _prior_arguments: &[DebateArgument],
    ) -> Result<DebateArgument, DebateError> {
        Ok(DebateArgument {
            persona_index: 0,
            position: position.to_string(),
            round: 0,
            text: "raw confidence".to_string(),
            confidence: self.confidence,
        })
    }
}

// ── DebateConfig ─────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = DebateConfig::default();
    assert_eq!(config.max_rounds, 3);
    assert!(!config.early_stop);
    assert_eq!(config.flat_confidence_window, 2);
    assert_eq!(config.flat_confidence_epsilon, 0.02);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(DebateConfig::new(), DebateConfig::default());
}

#[test]
fn config_with_max_rounds() {
    assert_eq!(DebateConfig::new().with_max_rounds(7).max_rounds, 7);
}

#[test]
fn config_with_early_stop() {
    assert!(DebateConfig::new().with_early_stop(true).early_stop);
}

#[test]
fn config_with_flat_confidence_window() {
    assert_eq!(
        DebateConfig::new()
            .with_flat_confidence_window(5)
            .flat_confidence_window,
        5
    );
}

#[test]
fn config_with_flat_confidence_epsilon() {
    assert_eq!(
        DebateConfig::new()
            .with_flat_confidence_epsilon(0.1)
            .flat_confidence_epsilon,
        0.1
    );
}

#[test]
fn config_with_judge_weights() {
    let weights = DebateJudgeWeights::new(1.0, 2.0, 3.0);
    assert_eq!(
        DebateConfig::new()
            .with_judge_weights(weights)
            .judge_weights,
        weights
    );
}

#[test]
fn config_builder_chain() {
    let config = DebateConfig::new()
        .with_max_rounds(4)
        .with_early_stop(true)
        .with_flat_confidence_window(3)
        .with_flat_confidence_epsilon(0.05);
    assert_eq!(config.max_rounds, 4);
    assert!(config.early_stop);
    assert_eq!(config.flat_confidence_window, 3);
    assert_eq!(config.flat_confidence_epsilon, 0.05);
}

// ── DebateJudgeWeights ───────────────────────────────────────────────────────

#[test]
fn judge_weights_default() {
    let weights = DebateJudgeWeights::default();
    assert_eq!(weights.argument_count_weight, 0.2);
    assert_eq!(weights.confidence_weight, 0.3);
    assert_eq!(weights.rebuttal_weight, 0.5);
}

#[test]
fn judge_weights_rebuttal_is_dominant_by_default() {
    let weights = DebateJudgeWeights::default();
    assert!(weights.rebuttal_weight > weights.confidence_weight);
    assert!(weights.confidence_weight > weights.argument_count_weight);
}

#[test]
fn judge_weights_new() {
    let weights = DebateJudgeWeights::new(1.0, 2.0, 3.0);
    assert_eq!(weights.argument_count_weight, 1.0);
    assert_eq!(weights.confidence_weight, 2.0);
    assert_eq!(weights.rebuttal_weight, 3.0);
}

#[test]
fn judge_weights_with_argument_count_weight() {
    assert_eq!(
        DebateJudgeWeights::default()
            .with_argument_count_weight(9.0)
            .argument_count_weight,
        9.0
    );
}

#[test]
fn judge_weights_with_confidence_weight() {
    assert_eq!(
        DebateJudgeWeights::default()
            .with_confidence_weight(9.0)
            .confidence_weight,
        9.0
    );
}

#[test]
fn judge_weights_with_rebuttal_weight() {
    assert_eq!(
        DebateJudgeWeights::default()
            .with_rebuttal_weight(9.0)
            .rebuttal_weight,
        9.0
    );
}

// ── DebateArgument / DebateRound / DebateParticipant ────────────────────────

#[test]
fn debate_argument_new_defaults() {
    let argument = DebateArgument::new("hello", 0.5);
    assert_eq!(argument.persona_index, 0);
    assert_eq!(argument.position, "");
    assert_eq!(argument.round, 0);
    assert_eq!(argument.text, "hello");
    assert_eq!(argument.confidence, 0.5);
}

#[test]
fn debate_argument_new_clamps_high_confidence() {
    assert_eq!(DebateArgument::new("x", 5.0).confidence, 1.0);
}

#[test]
fn debate_argument_new_clamps_low_confidence() {
    assert_eq!(DebateArgument::new("x", -3.0).confidence, 0.0);
}

#[test]
fn debate_round_new_is_empty() {
    let round = DebateRound::new(2);
    assert_eq!(round.round_index, 2);
    assert!(round.arguments.is_empty());
}

#[test]
fn debate_round_argument_for_finds_position() {
    let mut round = DebateRound::new(0);
    round.arguments.push(arg(0, "pro", 0, "t", 0.5));
    assert_eq!(round.argument_for("pro").unwrap().persona_index, 0);
    assert!(round.argument_for("con").is_none());
}

#[test]
fn debate_participant_new_stores_fields() {
    let persona = MockDebatePersona::new("X");
    let participant = DebateParticipant::new(&persona, "stance");
    assert_eq!(participant.position, "stance");
}

// ── DebateError ──────────────────────────────────────────────────────────────

#[test]
fn error_empty_question_display() {
    assert_eq!(
        DebateError::EmptyQuestion.to_string(),
        "question must not be empty"
    );
}

#[test]
fn error_too_few_personas_display() {
    assert_eq!(
        DebateError::TooFewPersonas { count: 1 }.to_string(),
        "a debate requires at least 2 personas, got 1"
    );
}

#[test]
fn error_zero_rounds_display() {
    assert_eq!(
        DebateError::ZeroRounds.to_string(),
        "max_rounds must be at least 1, got 0"
    );
}

#[test]
fn error_empty_position_display() {
    assert_eq!(
        DebateError::EmptyPosition { index: 3 }.to_string(),
        "participant 3 has an empty position"
    );
}

#[test]
fn error_duplicate_position_display() {
    assert_eq!(
        DebateError::DuplicatePosition {
            position: "pro".to_string()
        }
        .to_string(),
        "position \"pro\" is assigned to more than one participant"
    );
}

#[test]
fn error_persona_generation_failed_display() {
    assert_eq!(
        DebateError::PersonaGenerationFailed {
            reason: "timeout".to_string()
        }
        .to_string(),
        "persona failed to produce an argument: timeout"
    );
}

#[test]
fn error_judge_failed_display() {
    assert_eq!(
        DebateError::JudgeFailed {
            reason: "oops".to_string()
        }
        .to_string(),
        "judge failed to produce a verdict: oops"
    );
}

#[test]
fn error_empty_transcript_display() {
    assert_eq!(
        DebateError::EmptyTranscript.to_string(),
        "cannot judge an empty transcript (no rounds)"
    );
}

#[test]
fn error_is_clone_and_partial_eq() {
    let first = DebateError::ZeroRounds;
    let second = first.clone();
    assert_eq!(first, second);
    assert_ne!(DebateError::ZeroRounds, DebateError::EmptyQuestion);
}

// ── DebateVerdict / DebateResult ─────────────────────────────────────────────

#[test]
fn verdict_score_for_finds_position() {
    let verdict = DebateVerdict {
        winning_position: "pro".to_string(),
        winning_persona_index: 0,
        rationale: "r".to_string(),
        scores: vec![sample_score(0, "pro", 0.9), sample_score(1, "con", 0.4)],
    };
    assert_eq!(verdict.score_for("pro").unwrap().persona_index, 0);
    assert_eq!(verdict.score_for("con").unwrap().persona_index, 1);
    assert!(verdict.score_for("missing").is_none());
}

#[test]
fn result_all_arguments_flattens_transcript() {
    let round0 = DebateRound {
        round_index: 0,
        arguments: vec![DebateArgument::new("a", 0.5), DebateArgument::new("b", 0.5)],
    };
    let round1 = DebateRound {
        round_index: 1,
        arguments: vec![DebateArgument::new("c", 0.5)],
    };
    let result = DebateResult {
        question: "q".to_string(),
        transcript: vec![round0, round1],
        verdict: DebateVerdict {
            winning_position: "pro".to_string(),
            winning_persona_index: 0,
            rationale: "r".to_string(),
            scores: vec![],
        },
        rounds_run: 2,
        stopped_early: false,
        early_stop_reason: None,
    };
    let all = result.all_arguments();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].text, "a");
    assert_eq!(all[2].text, "c");
}

#[test]
fn result_winning_position_shortcut() {
    let result = DebateResult {
        question: "q".to_string(),
        transcript: vec![],
        verdict: DebateVerdict {
            winning_position: "pro".to_string(),
            winning_persona_index: 0,
            rationale: "r".to_string(),
            scores: vec![],
        },
        rounds_run: 0,
        stopped_early: false,
        early_stop_reason: None,
    };
    assert_eq!(result.winning_position(), "pro");
}

#[test]
fn debate_position_score_fields() {
    let score = sample_score(2, "neutral", 0.75);
    assert_eq!(score.persona_index, 2);
    assert_eq!(score.position, "neutral");
    assert_eq!(score.total_score, 0.75);
}

#[test]
fn debate_verdict_clone_eq() {
    let verdict = DebateVerdict {
        winning_position: "pro".to_string(),
        winning_persona_index: 0,
        rationale: "r".to_string(),
        scores: vec![],
    };
    assert_eq!(verdict.clone(), verdict);
}

// ── lexical helpers ──────────────────────────────────────────────────────────

#[test]
fn fnv1a_is_deterministic() {
    assert_eq!(fnv1a(b"hello world"), fnv1a(b"hello world"));
}

#[test]
fn fnv1a_differs_for_different_input() {
    assert_ne!(fnv1a(b"hello"), fnv1a(b"world"));
}

#[test]
fn fnv1a_empty_input_is_offset_basis() {
    assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
}

#[test]
fn content_terms_filters_stopwords_and_short_tokens() {
    let terms = content_terms("The cat is on a mat, and it is red");
    assert!(!terms.contains("the"));
    assert!(!terms.contains("is"));
    assert!(!terms.contains("on"));
    assert!(!terms.contains("a"));
    assert!(terms.contains("cat"));
    assert!(terms.contains("red"));
}

#[test]
fn content_terms_lowercases() {
    let terms = content_terms("CATS Dogs");
    assert!(terms.contains("cats"));
    assert!(terms.contains("dogs"));
}

#[test]
fn jaccard_identical_sets_is_one() {
    let a = content_terms("cats dogs birds");
    let b = content_terms("cats dogs birds");
    assert_eq!(jaccard(&a, &b), 1.0);
}

#[test]
fn jaccard_disjoint_sets_is_zero() {
    let a = content_terms("cats dogs");
    let b = content_terms("rockets planets");
    assert_eq!(jaccard(&a, &b), 0.0);
}

#[test]
fn jaccard_both_empty_is_zero() {
    let a = content_terms("a an is");
    let b = content_terms("");
    assert_eq!(jaccard(&a, &b), 0.0);
}

#[test]
fn jaccard_partial_overlap() {
    let a = content_terms("cats dogs birds");
    let b = content_terms("cats fish reptiles");
    assert_eq!(jaccard(&a, &b), 1.0 / 5.0);
}

#[test]
fn salient_term_is_deterministic() {
    assert_eq!(
        salient_term("cats dogs birds fly high"),
        salient_term("cats dogs birds fly high")
    );
}

#[test]
fn salient_term_none_when_no_content_terms() {
    assert_eq!(salient_term("a an is"), None);
}

// ── MockDebatePersona ────────────────────────────────────────────────────────

#[test]
fn mock_persona_round_zero_has_no_opponent_reference() {
    let persona = MockDebatePersona::new("A");
    let argument = persona
        .argue("What is best?", "cats are best", &[])
        .unwrap();
    assert_eq!(argument.round, 0);
    assert!(argument.confidence >= 0.0 && argument.confidence <= 1.0);
    assert!(!argument.text.is_empty());
}

#[test]
fn mock_persona_computes_round_from_own_prior_arguments() {
    let persona = MockDebatePersona::new("A");
    let prior = vec![arg(0, "cats", 0, "t0", 0.5), arg(1, "dogs", 0, "t1", 0.5)];
    let argument = persona.argue("q", "cats", &prior).unwrap();
    assert_eq!(argument.round, 1);
}

#[test]
fn mock_persona_reacts_to_opponent_salient_term() {
    let persona_a = MockDebatePersona::new("A");
    let opponent_arg = arg(
        1,
        "the panspermia hypothesis explains life",
        0,
        "the panspermia hypothesis explains life on earth well",
        0.5,
    );
    let own_arg = arg(
        0,
        "the mitochondria hypothesis explains life",
        0,
        "opening",
        0.5,
    );
    let prior = vec![own_arg, opponent_arg.clone()];
    let argument = persona_a
        .argue("q", "the mitochondria hypothesis explains life", &prior)
        .unwrap();
    let salient = salient_term(&opponent_arg.text).unwrap();
    assert!(argument.text.to_lowercase().contains(&salient));
}

#[test]
fn mock_persona_determinism() {
    let persona = MockDebatePersona::new("A");
    let prior = vec![arg(1, "dogs", 0, "dogs are loyal", 0.5)];
    let first = persona.argue("q", "cats", &prior).unwrap();
    let second = persona.argue("q", "cats", &prior).unwrap();
    assert_eq!(first.text, second.text);
    assert_eq!(first.confidence, second.confidence);
}

#[test]
fn mock_persona_confidence_in_bounds() {
    let persona = MockDebatePersona::new("");
    for question in ["", "a", "what is the meaning of life and the universe"] {
        let argument = persona
            .argue(question, "some fairly long position statement here", &[])
            .unwrap();
        assert!(argument.confidence >= 0.0 && argument.confidence <= 1.0);
    }
}

#[test]
fn mock_persona_empty_label_round_zero_uses_generic_phrasing() {
    let persona = MockDebatePersona::default();
    let argument = persona.argue("q", "cats are best", &[]).unwrap();
    assert!(argument.text.starts_with("Opening position on"));
}

#[test]
fn mock_persona_labeled_round_zero_includes_label() {
    let persona = MockDebatePersona::new("Advocate");
    let argument = persona.argue("q", "cats are best", &[]).unwrap();
    assert!(argument.text.starts_with("Advocate opens on"));
}

#[test]
fn mock_persona_labeled_rebuttal_includes_label() {
    let persona = MockDebatePersona::new("Advocate");
    let prior = vec![arg(1, "dogs", 0, "dogs are loyal companions", 0.5)];
    let argument = persona.argue("q", "cats", &prior).unwrap();
    assert!(argument.text.starts_with("Advocate rebuts persona 1"));
}

#[test]
fn mock_persona_different_opponents_yield_different_text() {
    let persona = MockDebatePersona::new("A");
    let prior1 = vec![arg(
        1,
        "dogs",
        0,
        "dogs are extremely loyal companions",
        0.5,
    )];
    let prior2 = vec![arg(
        1,
        "birds",
        0,
        "birds can fly gracefully through skies",
        0.5,
    )];
    let first = persona.argue("q", "cats", &prior1).unwrap();
    let second = persona.argue("q", "cats", &prior2).unwrap();
    assert_ne!(first.text, second.text);
}

#[test]
fn mock_persona_last_opposing_skips_own_position() {
    let persona = MockDebatePersona::new("A");
    let prior = vec![arg(0, "cats", 0, "t0", 0.5)];
    let argument = persona.argue("q", "cats", &prior).unwrap();
    assert!(argument.text.starts_with("A opens on"));
}

// ── MockDebateJudge ──────────────────────────────────────────────────────────

#[test]
fn judge_empty_question_errors() {
    let judge = MockDebateJudge::default();
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![arg(0, "pro", 0, "t", 0.5)],
    }];
    assert_eq!(
        judge.judge("", &transcript),
        Err(DebateError::EmptyQuestion)
    );
}

#[test]
fn judge_empty_transcript_errors() {
    let judge = MockDebateJudge::default();
    assert_eq!(judge.judge("q", &[]), Err(DebateError::EmptyTranscript));
}

#[test]
fn judge_transcript_with_only_empty_rounds_errors() {
    let judge = MockDebateJudge::default();
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![],
    }];
    assert_eq!(
        judge.judge("q", &transcript),
        Err(DebateError::EmptyTranscript)
    );
}

#[test]
fn judge_prefers_higher_confidence_when_other_dims_tied() {
    let judge = MockDebateJudge::new(DebateJudgeWeights::new(0.0, 1.0, 0.0));
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![
            arg(0, "pro", 0, "alpha beta gamma", 0.9),
            arg(1, "con", 0, "alpha beta gamma", 0.2),
        ],
    }];
    let verdict = judge.judge("q", &transcript).unwrap();
    assert_eq!(verdict.winning_position, "pro");
}

#[test]
fn judge_prefers_more_arguments_when_other_dims_tied() {
    let judge = MockDebateJudge::new(DebateJudgeWeights::new(1.0, 0.0, 0.0));
    let transcript = vec![
        DebateRound {
            round_index: 0,
            arguments: vec![arg(0, "pro", 0, "a", 0.5), arg(1, "con", 0, "a", 0.5)],
        },
        DebateRound {
            round_index: 1,
            arguments: vec![arg(0, "pro", 1, "a", 0.5)],
        },
    ];
    let verdict = judge.judge("q", &transcript).unwrap();
    assert_eq!(verdict.winning_position, "pro");
}

#[test]
fn judge_favors_better_rebuttal_engagement() {
    // Isolate rebuttal_weight: argument count and confidence are identical
    // for both sides, so only rebuttal engagement can decide the winner.
    let judge = MockDebateJudge::new(DebateJudgeWeights::new(0.0, 0.0, 1.0));
    let transcript = vec![
        DebateRound {
            round_index: 0,
            arguments: vec![
                arg(0, "pro", 0, "renewable solar wind power grows fast", 0.6),
                arg(1, "con", 0, "fossil coal oil gas remain dominant", 0.6),
            ],
        },
        DebateRound {
            round_index: 1,
            arguments: vec![
                // pro directly echoes con's round-0 vocabulary: sharp rebuttal.
                arg(
                    0,
                    "pro",
                    1,
                    "fossil coal oil gas cannot compete on cost anymore",
                    0.6,
                ),
                // con ignores pro's round-0 point entirely: no engagement.
                arg(
                    1,
                    "con",
                    1,
                    "economic growth continues steadily worldwide",
                    0.6,
                ),
            ],
        },
    ];
    let verdict = judge.judge("Will renewables win?", &transcript).unwrap();
    assert_eq!(verdict.winning_position, "pro");
    let pro_score = verdict.score_for("pro").unwrap();
    let con_score = verdict.score_for("con").unwrap();
    assert!(pro_score.rebuttal_engagement > con_score.rebuttal_engagement);
    assert_eq!(pro_score.rebuttal_engagement, 0.4);
    assert_eq!(con_score.rebuttal_engagement, 0.0);
}

#[test]
fn judge_weight_changes_can_flip_winner() {
    let transcript = vec![
        DebateRound {
            round_index: 0,
            arguments: vec![
                arg(0, "pro", 0, "solar wind renewable power", 0.9),
                arg(1, "con", 0, "coal gas fossil fuel", 0.9),
            ],
        },
        DebateRound {
            round_index: 1,
            arguments: vec![
                // High confidence, but no engagement with con's round-0 point.
                arg(0, "pro", 1, "batteries storage improve rapidly", 0.9),
                // Low confidence, but heavily echoes pro's round-0 vocabulary.
                arg(1, "con", 1, "solar wind renewable power intermittent", 0.1),
            ],
        },
    ];

    let confidence_judge = MockDebateJudge::new(DebateJudgeWeights::new(0.0, 1.0, 0.0));
    let by_confidence = confidence_judge.judge("q", &transcript).unwrap();
    assert_eq!(by_confidence.winning_position, "pro");

    let rebuttal_judge = MockDebateJudge::new(DebateJudgeWeights::new(0.0, 0.0, 1.0));
    let by_rebuttal = rebuttal_judge.judge("q", &transcript).unwrap();
    assert_eq!(by_rebuttal.winning_position, "con");
}

#[test]
fn judge_scores_sorted_descending() {
    let judge = MockDebateJudge::default();
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![
            arg(0, "pro", 0, "alpha beta gamma delta", 0.9),
            arg(1, "con", 0, "x", 0.1),
        ],
    }];
    let verdict = judge.judge("q", &transcript).unwrap();
    assert!(verdict.scores[0].total_score >= verdict.scores[1].total_score);
}

#[test]
fn judge_winning_persona_index_matches_top_score() {
    let judge = MockDebateJudge::default();
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![
            arg(0, "pro", 0, "alpha beta gamma delta", 0.9),
            arg(1, "con", 0, "x", 0.1),
        ],
    }];
    let verdict = judge.judge("q", &transcript).unwrap();
    assert_eq!(
        verdict.winning_persona_index,
        verdict.scores[0].persona_index
    );
    assert_eq!(verdict.winning_position, verdict.scores[0].position);
}

#[test]
fn judge_tie_breaks_by_ascending_persona_index() {
    let judge = MockDebateJudge::default();
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![
            arg(0, "pro", 0, "alpha beta", 0.5),
            arg(1, "con", 0, "alpha beta", 0.5),
        ],
    }];
    let verdict = judge.judge("q", &transcript).unwrap();
    assert_eq!(verdict.winning_persona_index, 0);
}

#[test]
fn judge_rationale_is_nonempty_and_traceable() {
    let judge = MockDebateJudge::default();
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![
            arg(0, "pro", 0, "alpha beta gamma", 0.9),
            arg(1, "con", 0, "x", 0.1),
        ],
    }];
    let verdict = judge.judge("the exact question text", &transcript).unwrap();
    assert!(!verdict.rationale.is_empty());
    assert!(verdict.rationale.contains("the exact question text"));
    assert!(verdict.rationale.contains("pro"));
    assert!(verdict.rationale.contains("After 1 round"));
}

#[test]
fn judge_rationale_mentions_runner_up_when_present() {
    let judge = MockDebateJudge::default();
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![
            arg(0, "pro", 0, "alpha beta gamma", 0.9),
            arg(1, "con", 0, "x", 0.1),
        ],
    }];
    let verdict = judge.judge("q", &transcript).unwrap();
    assert!(verdict.rationale.contains("runner-up"));
}

#[test]
fn judge_rebuttal_engagement_zero_when_no_preceding_opponent() {
    let judge = MockDebateJudge::default();
    let transcript = vec![DebateRound {
        round_index: 0,
        arguments: vec![
            arg(0, "pro", 0, "alpha beta", 0.9),
            arg(1, "con", 0, "gamma delta", 0.5),
        ],
    }];
    let verdict = judge.judge("q", &transcript).unwrap();
    assert_eq!(verdict.score_for("pro").unwrap().rebuttal_engagement, 0.0);
    assert_eq!(verdict.score_for("con").unwrap().rebuttal_engagement, 0.0);
}

#[test]
fn judge_argument_count_reflects_transcript() {
    let judge = MockDebateJudge::default();
    let transcript = vec![
        DebateRound {
            round_index: 0,
            arguments: vec![arg(0, "pro", 0, "a", 0.5), arg(1, "con", 0, "a", 0.5)],
        },
        DebateRound {
            round_index: 1,
            arguments: vec![arg(0, "pro", 1, "b", 0.5), arg(1, "con", 1, "b", 0.5)],
        },
    ];
    let verdict = judge.judge("q", &transcript).unwrap();
    assert_eq!(verdict.score_for("pro").unwrap().argument_count, 2);
    assert_eq!(verdict.score_for("con").unwrap().argument_count, 2);
}

// ── DebateEngine ─────────────────────────────────────────────────────────────

#[test]
fn engine_two_persona_debate_runs_configured_rounds() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "cats are best", &persona_b, "dogs are best");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(4));
    let judge = MockDebateJudge::default();
    let result = engine
        .run("Which pet is best?", &participants, &judge)
        .unwrap();
    assert_eq!(result.transcript.len(), 4);
    assert_eq!(result.rounds_run, 4);
    for round in &result.transcript {
        assert_eq!(round.arguments.len(), 2);
    }
}

#[test]
fn engine_three_persona_debate_each_gets_a_turn_every_round() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let persona_c = MockDebatePersona::new("C");
    let participants = [
        DebateParticipant::new(&persona_a, "position alpha"),
        DebateParticipant::new(&persona_b, "position beta"),
        DebateParticipant::new(&persona_c, "position gamma"),
    ];
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(3));
    let judge = MockDebateJudge::default();
    let result = engine
        .run("Three way question?", &participants, &judge)
        .unwrap();
    assert_eq!(result.transcript.len(), 3);
    for round in &result.transcript {
        assert_eq!(round.arguments.len(), 3);
        let mut indices: Vec<usize> = round.arguments.iter().map(|a| a.persona_index).collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1, 2]);
    }
}

#[test]
fn engine_determinism_same_config_and_mocks_twice() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "cats are best", &persona_b, "dogs are best");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(3));
    let judge = MockDebateJudge::default();
    let first = engine
        .run("Which pet is best?", &participants, &judge)
        .unwrap();
    let second = engine
        .run("Which pet is best?", &participants, &judge)
        .unwrap();
    assert_eq!(first, second);
}

#[test]
fn engine_passes_full_transcript_length_growing_each_round() {
    let recorder = RecordingDebatePersona::default();
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&recorder, "alpha", &persona_b, "beta");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(3));
    let judge = MockDebateJudge::default();
    let _ = engine.run("q", &participants, &judge).unwrap();
    let lengths = recorder.seen_lengths.borrow();
    assert_eq!(*lengths, vec![0, 2, 4]);
}

#[test]
fn engine_prior_arguments_include_opponent_entries() {
    let recorder = RecordingDebatePersona::default();
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&recorder, "alpha", &persona_b, "beta");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(2));
    let judge = MockDebateJudge::default();
    let _ = engine.run("q", &participants, &judge).unwrap();
    let snapshots = recorder.seen_snapshots.borrow();
    assert!(snapshots[1].iter().any(|a| a.position == "beta"));
}

#[test]
fn engine_mock_persona_rebuttal_engages_with_opponent_text() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(
        &persona_a,
        "the mitochondria hypothesis explains eukaryotic life",
        &persona_b,
        "the panspermia hypothesis explains eukaryotic life",
    );
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(2));
    let judge = MockDebateJudge::default();
    let result = engine
        .run("What explains eukaryotic life?", &participants, &judge)
        .unwrap();

    let round0_a_text = &result.transcript[0].arguments[0].text;
    let expected_salient = salient_term(round0_a_text).expect("round-0 text has content terms");
    let round1_b_text = result.transcript[1].arguments[1].text.to_lowercase();
    assert!(round1_b_text.contains(&expected_salient));
}

#[test]
fn engine_error_empty_question() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "x", &persona_b, "y");
    let engine = DebateEngine::default();
    let judge = MockDebateJudge::default();
    assert_eq!(
        engine.run("   ", &participants, &judge),
        Err(DebateError::EmptyQuestion)
    );
}

#[test]
fn engine_error_too_few_personas_zero() {
    let engine = DebateEngine::default();
    let judge = MockDebateJudge::default();
    let participants: [DebateParticipant<'_>; 0] = [];
    assert_eq!(
        engine.run("q", &participants, &judge),
        Err(DebateError::TooFewPersonas { count: 0 })
    );
}

#[test]
fn engine_error_too_few_personas_one() {
    let persona_a = MockDebatePersona::new("A");
    let engine = DebateEngine::default();
    let judge = MockDebateJudge::default();
    let participants = [DebateParticipant::new(&persona_a, "x")];
    assert_eq!(
        engine.run("q", &participants, &judge),
        Err(DebateError::TooFewPersonas { count: 1 })
    );
}

#[test]
fn engine_error_zero_rounds() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "x", &persona_b, "y");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(0));
    let judge = MockDebateJudge::default();
    assert_eq!(
        engine.run("q", &participants, &judge),
        Err(DebateError::ZeroRounds)
    );
}

#[test]
fn engine_error_empty_position() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "   ", &persona_b, "y");
    let engine = DebateEngine::default();
    let judge = MockDebateJudge::default();
    assert_eq!(
        engine.run("q", &participants, &judge),
        Err(DebateError::EmptyPosition { index: 0 })
    );
}

#[test]
fn engine_error_duplicate_position() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "same", &persona_b, "same");
    let engine = DebateEngine::default();
    let judge = MockDebateJudge::default();
    assert_eq!(
        engine.run("q", &participants, &judge),
        Err(DebateError::DuplicatePosition {
            position: "same".to_string()
        })
    );
}

#[test]
fn engine_persona_error_propagates() {
    let failing = FailingDebatePersona;
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&failing, "x", &persona_b, "y");
    let engine = DebateEngine::default();
    let judge = MockDebateJudge::default();
    assert_eq!(
        engine.run("q", &participants, &judge),
        Err(DebateError::PersonaGenerationFailed {
            reason: "boom".to_string()
        })
    );
}

#[test]
fn engine_judge_error_propagates() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "x", &persona_b, "y");
    let engine = DebateEngine::default();
    let judge = FailingDebateJudge;
    assert_eq!(
        engine.run("q", &participants, &judge),
        Err(DebateError::JudgeFailed {
            reason: "boom".to_string()
        })
    );
}

#[test]
fn engine_overwrites_sloppy_persona_bookkeeping() {
    let sloppy = SloppyDebatePersona;
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&sloppy, "alpha", &persona_b, "beta");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(1));
    let judge = MockDebateJudge::default();
    let result = engine.run("q", &participants, &judge).unwrap();
    let sloppy_arg = &result.transcript[0].arguments[0];
    assert_eq!(sloppy_arg.persona_index, 0);
    assert_eq!(sloppy_arg.position, "alpha");
    assert_eq!(sloppy_arg.round, 0);
}

#[test]
fn engine_clamps_out_of_range_confidence_high() {
    let persona_a = OutOfRangeConfidencePersona { confidence: 5.0 };
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "alpha", &persona_b, "beta");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(1));
    let judge = MockDebateJudge::default();
    let result = engine.run("q", &participants, &judge).unwrap();
    assert_eq!(result.transcript[0].arguments[0].confidence, 1.0);
}

#[test]
fn engine_clamps_out_of_range_confidence_low() {
    let persona_a = OutOfRangeConfidencePersona { confidence: -3.0 };
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "alpha", &persona_b, "beta");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(1));
    let judge = MockDebateJudge::default();
    let result = engine.run("q", &participants, &judge).unwrap();
    assert_eq!(result.transcript[0].arguments[0].confidence, 0.0);
}

#[test]
fn engine_early_stop_disabled_runs_full_rounds() {
    let scripted_a = ScriptedDebatePersona {
        confidences: vec![0.9, 0.5, 0.5, 0.5, 0.5],
    };
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&scripted_a, "alpha", &persona_b, "beta");
    let engine = DebateEngine::new(
        DebateConfig::new()
            .with_max_rounds(5)
            .with_early_stop(false),
    );
    let judge = MockDebateJudge::default();
    let result = engine.run("q", &participants, &judge).unwrap();
    assert_eq!(result.rounds_run, 5);
    assert!(!result.stopped_early);
    assert!(result.early_stop_reason.is_none());
}

#[test]
fn engine_early_stop_triggers_on_flat_confidence() {
    // Round 0: 0.9 (baseline). Round 1: 0.5 (decline -> streak 1). Round 2:
    // 0.5 (flat -> streak 2, meets the default window of 2 -> stop).
    let scripted_a = ScriptedDebatePersona {
        confidences: vec![0.9, 0.5, 0.5, 0.5, 0.5],
    };
    let scripted_b = ScriptedDebatePersona {
        confidences: vec![0.1, 0.4, 0.7, 0.8, 0.95],
    };
    let participants = two_participants(&scripted_a, "alpha", &scripted_b, "beta");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(5).with_early_stop(true));
    let judge = MockDebateJudge::default();
    let result = engine.run("q", &participants, &judge).unwrap();
    assert!(result.stopped_early);
    assert_eq!(result.rounds_run, 3);
    assert!(result.early_stop_reason.is_some());
}

#[test]
fn engine_early_stop_does_not_trigger_when_confidence_keeps_increasing() {
    let scripted_a = ScriptedDebatePersona {
        confidences: vec![0.1, 0.3, 0.5, 0.7, 0.9],
    };
    let scripted_b = ScriptedDebatePersona {
        confidences: vec![0.05, 0.25, 0.45, 0.65, 0.85],
    };
    let participants = two_participants(&scripted_a, "alpha", &scripted_b, "beta");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(5).with_early_stop(true));
    let judge = MockDebateJudge::default();
    let result = engine.run("q", &participants, &judge).unwrap();
    assert!(!result.stopped_early);
    assert_eq!(result.rounds_run, 5);
    assert!(result.early_stop_reason.is_none());
}

#[test]
fn engine_early_stop_reason_mentions_offending_persona() {
    let scripted_a = ScriptedDebatePersona {
        confidences: vec![0.9, 0.5, 0.5, 0.5, 0.5],
    };
    let scripted_b = ScriptedDebatePersona {
        confidences: vec![0.1, 0.4, 0.7, 0.8, 0.95],
    };
    let participants =
        two_participants(&scripted_a, "alpha-position", &scripted_b, "beta-position");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(5).with_early_stop(true));
    let judge = MockDebateJudge::default();
    let result = engine.run("q", &participants, &judge).unwrap();
    let reason = result.early_stop_reason.unwrap();
    assert!(reason.contains("alpha-position"));
}

#[test]
fn engine_verdict_rounds_match_transcript_length() {
    let persona_a = MockDebatePersona::new("A");
    let persona_b = MockDebatePersona::new("B");
    let participants = two_participants(&persona_a, "x", &persona_b, "y");
    let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(3));
    let judge = MockDebateJudge::default();
    let result = engine.run("q", &participants, &judge).unwrap();
    assert_eq!(result.rounds_run, result.transcript.len());
}

#[test]
fn engine_default_engine_uses_default_config() {
    let engine = DebateEngine::default();
    assert_eq!(engine.config, DebateConfig::default());
}

#[test]
fn engine_new_stores_config() {
    let config = DebateConfig::new().with_max_rounds(9);
    let engine = DebateEngine::new(config.clone());
    assert_eq!(engine.config, config);
}
