#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]

//! Tests for the `chainpoll` module.

use super::scorer::ChainPollScorer;
use super::types::{
    ChainOfThoughtJudge, ChainPollConfig, ChainPollError, MockChainPollJudge, PollFormulation,
    PollFraming,
};

// ── test helpers ────────────────────────────────────────────────────────────

const CONTEXT: &str = "The Eiffel Tower was completed in 1889 and is located in Paris, France.";
const GROUNDED_CLAIM: &str = "The Eiffel Tower was completed in 1889 in Paris, France.";
const FABRICATED_CLAIM: &str = "Kangaroos are native to the Amazon rainforest.";

fn mock_scorer() -> ChainPollScorer {
    ChainPollScorer::new(
        ChainPollConfig::default(),
        Box::new(MockChainPollJudge::default()),
    )
}

/// A [`ChainOfThoughtJudge`] driven entirely by a script keyed on formulation
/// index, so tests can engineer an exact vote split independent of the
/// (lexical-heuristic) [`MockChainPollJudge`].
///
/// `supported_by_index[i]` is the desired **un-inverted** "claim supported"
/// answer for formulation `i`; [`ScriptedJudge`] encodes it as whatever raw
/// verdict that formulation's [`PollFraming`] would produce for it (since
/// [`PollFraming::resolve_supported`] is its own inverse).
#[derive(Debug, Clone)]
struct ScriptedJudge {
    supported_by_index: Vec<bool>,
}

impl ScriptedJudge {
    fn new(supported_by_index: Vec<bool>) -> Self {
        Self { supported_by_index }
    }
}

impl ChainOfThoughtJudge for ScriptedJudge {
    fn judge(
        &self,
        claim: &str,
        context: &str,
        formulation: &PollFormulation,
    ) -> Result<(bool, String), ChainPollError> {
        if claim.trim().is_empty() {
            return Err(ChainPollError::EmptyClaim);
        }
        if context.trim().is_empty() {
            return Err(ChainPollError::EmptyContext);
        }
        let supported = self
            .supported_by_index
            .get(formulation.index)
            .copied()
            .unwrap_or(true);
        let raw_verdict = formulation.framing.resolve_supported(supported);
        Ok((
            raw_verdict,
            format!(
                "scripted verdict for formulation {}: supported={supported}",
                formulation.index
            ),
        ))
    }
}

fn scripted_scorer(supported_by_index: Vec<bool>) -> ChainPollScorer {
    let n = supported_by_index.len();
    ChainPollScorer::new(
        ChainPollConfig::new().with_num_formulations(n),
        Box::new(ScriptedJudge::new(supported_by_index)),
    )
}

// ── PollFraming ─────────────────────────────────────────────────────────────

#[test]
fn framing_direct_resolves_identity() {
    assert!(PollFraming::Direct.resolve_supported(true));
    assert!(!PollFraming::Direct.resolve_supported(false));
}

#[test]
fn framing_inverted_resolves_negation() {
    assert!(!PollFraming::Inverted.resolve_supported(true));
    assert!(PollFraming::Inverted.resolve_supported(false));
}

#[test]
fn framing_resolve_is_involution() {
    for framing in [PollFraming::Direct, PollFraming::Inverted] {
        for raw in [true, false] {
            let supported = framing.resolve_supported(raw);
            assert_eq!(framing.resolve_supported(supported), raw);
        }
    }
}

#[test]
fn framing_display() {
    assert_eq!(PollFraming::Direct.to_string(), "direct");
    assert_eq!(PollFraming::Inverted.to_string(), "inverted");
}

#[test]
fn framing_equality_and_copy() {
    assert_eq!(PollFraming::Direct, PollFraming::Direct);
    assert_ne!(PollFraming::Direct, PollFraming::Inverted);
    let f = PollFraming::Inverted;
    let g = f;
    assert_eq!(f, g);
}

// ── ChainPollConfig ──────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = ChainPollConfig::default();
    assert_eq!(config.num_formulations, 5);
    assert!((config.hallucination_threshold - 0.5).abs() < f32::EPSILON);
}

#[test]
fn config_builder_overrides() {
    let config = ChainPollConfig::new()
        .with_num_formulations(3)
        .with_hallucination_threshold(0.7);
    assert_eq!(config.num_formulations, 3);
    assert!((config.hallucination_threshold - 0.7).abs() < f32::EPSILON);
}

#[test]
fn config_validate_rejects_zero_formulations() {
    let config = ChainPollConfig::new().with_num_formulations(0);
    assert!(matches!(
        config.validate(),
        Err(ChainPollError::InvalidConfig(_))
    ));
}

#[test]
fn config_validate_rejects_too_many_formulations() {
    let config =
        ChainPollConfig::new().with_num_formulations(ChainPollConfig::MAX_FORMULATIONS + 1);
    assert!(matches!(
        config.validate(),
        Err(ChainPollError::InvalidConfig(_))
    ));
}

#[test]
fn config_validate_accepts_max_formulations() {
    let config = ChainPollConfig::new().with_num_formulations(ChainPollConfig::MAX_FORMULATIONS);
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_rejects_out_of_range_threshold() {
    let too_high = ChainPollConfig::new().with_hallucination_threshold(1.5);
    assert!(matches!(
        too_high.validate(),
        Err(ChainPollError::InvalidConfig(_))
    ));

    let too_low = ChainPollConfig::new().with_hallucination_threshold(-0.1);
    assert!(matches!(
        too_low.validate(),
        Err(ChainPollError::InvalidConfig(_))
    ));
}

#[test]
fn config_validate_rejects_non_finite_threshold() {
    let config = ChainPollConfig::new().with_hallucination_threshold(f32::NAN);
    assert!(matches!(
        config.validate(),
        Err(ChainPollError::InvalidConfig(_))
    ));
}

// ── formulation generation ────────────────────────────────────────────────────

#[test]
fn generate_formulations_yields_configured_count() {
    let scorer = ChainPollScorer::new(
        ChainPollConfig::new().with_num_formulations(4),
        Box::new(MockChainPollJudge::default()),
    );
    let formulations = scorer
        .generate_formulations(GROUNDED_CLAIM, CONTEXT)
        .expect("valid inputs");
    assert_eq!(formulations.len(), 4);
    for (i, f) in formulations.iter().enumerate() {
        assert_eq!(f.index, i);
        assert!(f.prompt.contains(GROUNDED_CLAIM));
        assert!(f.prompt.contains(CONTEXT));
    }
}

#[test]
fn generate_formulations_is_deterministic() {
    let scorer = mock_scorer();
    let first = scorer
        .generate_formulations(GROUNDED_CLAIM, CONTEXT)
        .expect("valid inputs");
    let second = scorer
        .generate_formulations(GROUNDED_CLAIM, CONTEXT)
        .expect("valid inputs");
    assert_eq!(first, second);
}

#[test]
fn generate_formulations_prefix_is_stable_across_n() {
    let small = ChainPollScorer::new(
        ChainPollConfig::new().with_num_formulations(3),
        Box::new(MockChainPollJudge::default()),
    )
    .generate_formulations(GROUNDED_CLAIM, CONTEXT)
    .expect("valid inputs");

    let large = ChainPollScorer::new(
        ChainPollConfig::new().with_num_formulations(ChainPollConfig::MAX_FORMULATIONS),
        Box::new(MockChainPollJudge::default()),
    )
    .generate_formulations(GROUNDED_CLAIM, CONTEXT)
    .expect("valid inputs");

    assert_eq!(&large[..3], small.as_slice());
}

#[test]
fn generate_formulations_includes_both_framings_by_default() {
    let scorer = mock_scorer();
    let formulations = scorer
        .generate_formulations(GROUNDED_CLAIM, CONTEXT)
        .expect("valid inputs");
    assert!(
        formulations
            .iter()
            .any(|f| f.framing == PollFraming::Direct)
    );
    assert!(
        formulations
            .iter()
            .any(|f| f.framing == PollFraming::Inverted)
    );
}

#[test]
fn generate_formulations_rejects_empty_claim() {
    let scorer = mock_scorer();
    assert_eq!(
        scorer.generate_formulations("   ", CONTEXT),
        Err(ChainPollError::EmptyClaim)
    );
}

#[test]
fn generate_formulations_rejects_empty_context() {
    let scorer = mock_scorer();
    assert_eq!(
        scorer.generate_formulations(GROUNDED_CLAIM, "  "),
        Err(ChainPollError::EmptyContext)
    );
}

#[test]
fn generate_formulations_rejects_invalid_config() {
    let scorer = ChainPollScorer::new(
        ChainPollConfig::new().with_num_formulations(0),
        Box::new(MockChainPollJudge::default()),
    );
    assert!(matches!(
        scorer.generate_formulations(GROUNDED_CLAIM, CONTEXT),
        Err(ChainPollError::InvalidConfig(_))
    ));
}

#[test]
fn formulation_bank_boundaries_match_max_formulations() {
    // At the boundary, exactly MAX_FORMULATIONS formulations are produced...
    let at_max = ChainPollScorer::new(
        ChainPollConfig::new().with_num_formulations(ChainPollConfig::MAX_FORMULATIONS),
        Box::new(MockChainPollJudge::default()),
    )
    .generate_formulations(GROUNDED_CLAIM, CONTEXT)
    .expect("valid inputs");
    assert_eq!(at_max.len(), ChainPollConfig::MAX_FORMULATIONS);

    // ...and one formulation beyond that is rejected by config validation
    // rather than silently truncated or panicking.
    let over_max = ChainPollScorer::new(
        ChainPollConfig::new().with_num_formulations(ChainPollConfig::MAX_FORMULATIONS + 1),
        Box::new(MockChainPollJudge::default()),
    )
    .generate_formulations(GROUNDED_CLAIM, CONTEXT);
    assert!(matches!(over_max, Err(ChainPollError::InvalidConfig(_))));
}

// ── poll: grounded vs fabricated claims ────────────────────────────────────────

#[test]
fn poll_supported_claim_scores_near_zero() {
    let result = mock_scorer()
        .poll(GROUNDED_CLAIM, CONTEXT)
        .expect("valid inputs");
    assert!(!result.is_hallucination);
    assert!(
        result.hallucination_score < 0.2,
        "score = {}",
        result.hallucination_score
    );
    assert_eq!(result.hallucinated_count(), 0);
    assert_eq!(result.supported_count(), result.num_formulations());
}

#[test]
fn poll_contradicted_claim_scores_near_one() {
    let result = mock_scorer()
        .poll(FABRICATED_CLAIM, CONTEXT)
        .expect("valid inputs");
    assert!(result.is_hallucination);
    assert!(
        result.hallucination_score > 0.8,
        "score = {}",
        result.hallucination_score
    );
    assert_eq!(result.supported_count(), 0);
    assert_eq!(result.hallucinated_count(), result.num_formulations());
}

#[test]
fn poll_rejects_empty_claim() {
    let err = mock_scorer().poll("", CONTEXT).unwrap_err();
    assert_eq!(err, ChainPollError::EmptyClaim);
}

#[test]
fn poll_rejects_empty_context() {
    let err = mock_scorer().poll(GROUNDED_CLAIM, "").unwrap_err();
    assert_eq!(err, ChainPollError::EmptyContext);
}

// ── calibration: unanimous / narrow-majority / monotonic ───────────────────────

#[test]
fn calibration_unanimous_supported_scores_zero() {
    let scorer = scripted_scorer(vec![true; 5]);
    let result = scorer.poll("claim", "context").expect("valid inputs");
    assert_eq!(result.hallucination_score, 0.0);
    assert!(!result.is_hallucination);
    assert_eq!(result.confidence, 1.0);
}

#[test]
fn calibration_unanimous_hallucinated_scores_one() {
    let scorer = scripted_scorer(vec![false; 5]);
    let result = scorer.poll("claim", "context").expect("valid inputs");
    assert_eq!(result.hallucination_score, 1.0);
    assert!(result.is_hallucination);
    assert_eq!(result.confidence, 1.0);
}

#[test]
fn calibration_narrow_majority_scores_near_half() {
    // 3 supported / 2 hallucinated: a narrow majority favoring "supported".
    let scorer = scripted_scorer(vec![true, true, true, false, false]);
    let result = scorer.poll("claim", "context").expect("valid inputs");
    assert!((result.hallucination_score - 0.4).abs() < 1e-6);
    assert!(!result.is_hallucination);
    assert!(result.hallucination_score > 0.2 && result.hallucination_score < 0.8);
    assert!(result.confidence < 0.5);

    // 2 supported / 3 hallucinated: a narrow majority favoring "hallucinated".
    let scorer = scripted_scorer(vec![true, true, false, false, false]);
    let result = scorer.poll("claim", "context").expect("valid inputs");
    assert!((result.hallucination_score - 0.6).abs() < 1e-6);
    assert!(result.is_hallucination);
    assert!(result.hallucination_score > 0.2 && result.hallucination_score < 0.8);
    assert!(result.confidence < 0.5);
}

#[test]
fn calibration_tie_has_zero_confidence() {
    let scorer = scripted_scorer(vec![true, true, false, false]);
    let result = scorer.poll("claim", "context").expect("valid inputs");
    assert_eq!(result.hallucination_score, 0.5);
    assert_eq!(result.confidence, 0.0);
}

#[test]
fn calibration_monotonic_in_vote_fraction() {
    // Sweep the number of "hallucinated" votes out of 5 formulations from 0
    // to 5 and check the score matches the exact vote fraction and strictly
    // increases as more formulations vote "hallucinated".
    let mut previous_score = -1.0_f32;
    for hallucinated in 0..=5usize {
        let supported = 5 - hallucinated;
        let mut votes = vec![true; supported];
        votes.extend(std::iter::repeat_n(false, hallucinated));
        let scorer = scripted_scorer(votes);
        let result = scorer.poll("claim", "context").expect("valid inputs");

        let expected = hallucinated as f32 / 5.0;
        assert!(
            (result.hallucination_score - expected).abs() < 1e-6,
            "hallucinated={hallucinated} expected={expected} got={}",
            result.hallucination_score
        );
        assert!(
            result.hallucination_score > previous_score,
            "score did not strictly increase at hallucinated={hallucinated}"
        );
        previous_score = result.hallucination_score;
    }
}

// ── inversion correctness ───────────────────────────────────────────────────

#[test]
fn inverted_formulation_is_un_inverted_before_voting() {
    // All formulations "vote supported": every un-inverted verdict must be
    // `true`, but the *raw* verdict for an Inverted formulation must be the
    // literal negation, `false`.
    let scorer = scripted_scorer(vec![true; 5]);
    let result = scorer.poll("claim", "context").expect("valid inputs");

    let inverted_verdicts: Vec<_> = result
        .formulation_verdicts
        .iter()
        .filter(|v| v.formulation.framing == PollFraming::Inverted)
        .collect();
    assert!(
        !inverted_verdicts.is_empty(),
        "expected an inverted formulation in the default bank"
    );
    for v in &inverted_verdicts {
        assert!(v.supported, "un-inverted verdict must be supported=true");
        assert!(
            !v.raw_verdict,
            "raw verdict for an Inverted formulation voting supported must be false"
        );
    }

    let direct_verdicts: Vec<_> = result
        .formulation_verdicts
        .iter()
        .filter(|v| v.formulation.framing == PollFraming::Direct)
        .collect();
    assert!(!direct_verdicts.is_empty());
    for v in &direct_verdicts {
        assert!(v.supported);
        assert!(
            v.raw_verdict,
            "raw verdict for a Direct formulation voting supported must be true"
        );
    }
}

#[test]
fn inverted_formulation_un_inverted_when_hallucinated() {
    let scorer = scripted_scorer(vec![false; 5]);
    let result = scorer.poll("claim", "context").expect("valid inputs");

    for v in result
        .formulation_verdicts
        .iter()
        .filter(|v| v.formulation.framing == PollFraming::Inverted)
    {
        assert!(!v.supported);
        assert!(
            v.raw_verdict,
            "raw verdict for an Inverted formulation voting hallucinated must be true"
        );
    }
    for v in result
        .formulation_verdicts
        .iter()
        .filter(|v| v.formulation.framing == PollFraming::Direct)
    {
        assert!(!v.supported);
        assert!(
            !v.raw_verdict,
            "raw verdict for a Direct formulation voting hallucinated must be false"
        );
    }
}

// ── per-formulation breakdown ─────────────────────────────────────────────────

#[test]
fn formulation_breakdown_matches_script() {
    let script = vec![true, false, true, false, true];
    let scorer = scripted_scorer(script.clone());
    let result = scorer.poll("claim", "context").expect("valid inputs");

    assert_eq!(result.formulation_verdicts.len(), script.len());
    for (i, verdict) in result.formulation_verdicts.iter().enumerate() {
        assert_eq!(verdict.formulation.index, i);
        assert_eq!(verdict.supported, script[i]);
        assert!(!verdict.reasoning.is_empty());
    }
    assert_eq!(result.supported_count(), 3);
    assert_eq!(result.hallucinated_count(), 2);
    assert_eq!(
        result.supported_count() + result.hallucinated_count(),
        result.num_formulations()
    );
}

#[test]
fn formulation_breakdown_preserves_unique_template_ids_and_order() {
    let result = mock_scorer()
        .poll(GROUNDED_CLAIM, CONTEXT)
        .expect("valid inputs");
    let ids: Vec<&str> = result
        .formulation_verdicts
        .iter()
        .map(|v| v.formulation.template_id)
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "template ids must be unique");

    for (i, verdict) in result.formulation_verdicts.iter().enumerate() {
        assert_eq!(verdict.formulation.index, i);
    }
}

// ── MockChainPollJudge ─────────────────────────────────────────────────────────

#[test]
fn mock_judge_default_threshold() {
    assert!((MockChainPollJudge::default().support_threshold - 0.15).abs() < f32::EPSILON);
}

#[test]
fn mock_judge_custom_threshold_flips_verdict() {
    // With the default threshold this pair is judged unsupported; a
    // near-zero threshold should flip that to "supported".
    let scorer = ChainPollScorer::new(
        ChainPollConfig::default(),
        Box::new(MockChainPollJudge::new(0.0)),
    );
    let result = scorer
        .poll(FABRICATED_CLAIM, CONTEXT)
        .expect("valid inputs");
    assert!(!result.is_hallucination);
}

#[test]
fn mock_judge_rejects_empty_claim() {
    let judge = MockChainPollJudge::default();
    let formulation = PollFormulation {
        index: 0,
        template_id: "direct_support",
        framing: PollFraming::Direct,
        prompt: "prompt".to_string(),
    };
    assert_eq!(
        judge.judge("", CONTEXT, &formulation),
        Err(ChainPollError::EmptyClaim)
    );
}

#[test]
fn mock_judge_rejects_empty_context() {
    let judge = MockChainPollJudge::default();
    let formulation = PollFormulation {
        index: 0,
        template_id: "direct_support",
        framing: PollFraming::Direct,
        prompt: "prompt".to_string(),
    };
    assert_eq!(
        judge.judge(GROUNDED_CLAIM, "", &formulation),
        Err(ChainPollError::EmptyContext)
    );
}

#[test]
fn mock_judge_reasoning_mentions_template_and_verdict() {
    let judge = MockChainPollJudge::default();
    let formulation = PollFormulation {
        index: 1,
        template_id: "inverted_contradiction",
        framing: PollFraming::Inverted,
        prompt: "prompt".to_string(),
    };
    let (raw_verdict, reasoning) = judge
        .judge(GROUNDED_CLAIM, CONTEXT, &formulation)
        .expect("valid inputs");
    assert!(reasoning.contains("inverted_contradiction"));
    assert!(reasoning.contains(if raw_verdict { "yes" } else { "no" }));
}

// ── ChainPollScorer defaults ────────────────────────────────────────────────────

#[test]
fn scorer_default_uses_mock_judge_and_default_config() {
    let scorer = ChainPollScorer::default();
    assert_eq!(scorer.config, ChainPollConfig::default());
    let result = scorer.poll(GROUNDED_CLAIM, CONTEXT).expect("valid inputs");
    assert_eq!(result.num_formulations(), 5);
}

// ── ChainPollError ───────────────────────────────────────────────────────────

#[test]
fn error_display_messages() {
    assert_eq!(
        ChainPollError::EmptyClaim.to_string(),
        "claim must not be empty"
    );
    assert_eq!(
        ChainPollError::EmptyContext.to_string(),
        "context must not be empty"
    );
    assert_eq!(
        ChainPollError::InvalidConfig("bad".to_string()).to_string(),
        "invalid config: bad"
    );
    assert_eq!(
        ChainPollError::JudgeFailed("boom".to_string()).to_string(),
        "chain-of-thought judge failed: boom"
    );
}

#[test]
fn error_equality() {
    assert_eq!(ChainPollError::EmptyClaim, ChainPollError::EmptyClaim);
    assert_ne!(ChainPollError::EmptyClaim, ChainPollError::EmptyContext);
}
