//! [`ChainPollScorer`]: deterministic multi-formulation prompt polling plus
//! majority-vote aggregation (Friel & Sanyal, 2023).
//!
//! This file owns the entire mechanism that makes `ChainPoll` what it is:
//!
//! 1. `FORMULATION_BANK` -- a fixed, ordered bank of
//!    [`ChainPollConfig::MAX_FORMULATIONS`] prompt templates, each a
//!    structurally distinct phrasing/ordering/perspective of "is this claim
//!    grounded in the context?" paired with its [`PollFraming`]. No
//!    randomness is involved anywhere in this file: the same claim, context,
//!    and `num_formulations` always yield the same formulations in the same
//!    order.
//! 2. [`ChainPollScorer::generate_formulations`] -- renders the first `N`
//!    bank entries into concrete [`PollFormulation`]s for a given claim and
//!    context.
//! 3. [`ChainPollScorer::poll`] -- dispatches each formulation to the
//!    configured [`ChainOfThoughtJudge`], un-inverts every raw verdict via
//!    its formulation's [`PollFraming`], and majority-votes the un-inverted
//!    verdicts into a [`ChainPollResult`].

use super::types::{
    ChainOfThoughtJudge, ChainPollConfig, ChainPollError, ChainPollFormulationVerdict,
    ChainPollResult, MockChainPollJudge, PollFormulation, PollFraming,
};

// ── formulation bank ──────────────────────────────────────────────────────────

/// Signature shared by every formulation-rendering function in
/// [`FORMULATION_BANK`]: `(claim, context) -> rendered prompt text`.
type RenderFn = fn(&str, &str) -> String;

/// Direct framing: asks outright whether the context supports the claim.
fn render_direct_support(claim: &str, context: &str) -> String {
    format!(
        "Read the context and the claim below. Think step by step about whether the context \
         provides factual support for the claim, then give a final YES or NO answer.\n\n\
         Context:\n{context}\n\nClaim:\n{claim}\n\n\
         Question: Is the claim factually supported by the context?"
    )
}

/// Inverted framing: asks whether the context contradicts the claim, so a
/// "yes" here means the claim is *not* supported.
fn render_inverted_contradiction(claim: &str, context: &str) -> String {
    format!(
        "Read the context and the claim below. Reason step by step about whether the context \
         contradicts the claim, then give a final YES or NO answer.\n\n\
         Context:\n{context}\n\nClaim:\n{claim}\n\n\
         Question: Does the context contradict this claim?"
    )
}

/// Direct framing: asks for an expert-judgment perspective on accuracy.
fn render_expert_judgment(claim: &str, context: &str) -> String {
    format!(
        "Context:\n{context}\n\nClaim:\n{claim}\n\n\
         Think step by step, then answer YES or NO: given the context, would a subject-matter \
         expert consider the claim accurate?"
    )
}

/// Direct framing: context-first ordering, "verifiable from" phrasing.
fn render_context_first_grounding(claim: &str, context: &str) -> String {
    format!(
        "Context:\n{context}\n\nNow consider this claim: {claim}\n\n\
         Reasoning step by step, decide whether the claim above can be verified directly from \
         the context above, and answer YES or NO."
    )
}

/// Inverted framing: claim-first ordering, asks about failure-to-support or
/// contradiction.
fn render_claim_first_inverted(claim: &str, context: &str) -> String {
    format!(
        "Claim: {claim}\n\nContext:\n{context}\n\n\
         Step by step, determine whether anything in the context above fails to support or \
         actively contradicts the claim. Answer YES or NO."
    )
}

/// Direct framing: second-person "fact-checker" perspective.
fn render_second_person_direct(claim: &str, context: &str) -> String {
    format!(
        "You are a careful fact-checker. You are given a context and a claim.\n\n\
         Context:\n{context}\n\nClaim:\n{claim}\n\n\
         Reasoning step by step and based solely on the context, would you say the claim holds \
         true? Answer YES or NO."
    )
}

/// Inverted framing: third-person "skeptical reader" perspective, asks
/// whether the claim would be flagged.
fn render_reader_perspective_inverted(claim: &str, context: &str) -> String {
    format!(
        "Imagine a skeptical reader fact-checking the claim below against the context.\n\n\
         Context:\n{context}\n\nClaim:\n{claim}\n\n\
         Thinking step by step, would that reader flag the claim as unsupported or fabricated? \
         Answer YES or NO."
    )
}

/// Direct framing: entailment phrasing.
fn render_entailment_direct(claim: &str, context: &str) -> String {
    format!(
        "Context:\n{context}\n\nClaim:\n{claim}\n\n\
         Step by step, decide: does the context entail (fully support) the claim? Answer YES \
         or NO."
    )
}

/// The fixed, deterministic bank of `(template_id, framing, render_fn)`
/// triples that [`ChainPollScorer::generate_formulations`] draws from.
///
/// Order is significant and stable: `generate_formulations` always takes the
/// first `num_formulations` entries, so growing or shrinking
/// [`ChainPollConfig::num_formulations`] only adds or removes formulations
/// from the tail -- it never reorders or replaces earlier ones. The length of
/// this bank must equal [`ChainPollConfig::MAX_FORMULATIONS`] (enforced by
/// the `formulation_bank_boundaries_match_max_formulations` test).
const FORMULATION_BANK: [(&str, PollFraming, RenderFn); ChainPollConfig::MAX_FORMULATIONS] = [
    ("direct_support", PollFraming::Direct, render_direct_support),
    (
        "inverted_contradiction",
        PollFraming::Inverted,
        render_inverted_contradiction,
    ),
    (
        "expert_judgment",
        PollFraming::Direct,
        render_expert_judgment,
    ),
    (
        "context_first_grounding",
        PollFraming::Direct,
        render_context_first_grounding,
    ),
    (
        "claim_first_inverted",
        PollFraming::Inverted,
        render_claim_first_inverted,
    ),
    (
        "second_person_direct",
        PollFraming::Direct,
        render_second_person_direct,
    ),
    (
        "reader_perspective_inverted",
        PollFraming::Inverted,
        render_reader_perspective_inverted,
    ),
    (
        "entailment_direct",
        PollFraming::Direct,
        render_entailment_direct,
    ),
];

// ── ChainPollScorer ────────────────────────────────────────────────────────────

/// `ChainPoll`'s deterministic multi-formulation polling orchestrator (Friel
/// & Sanyal, 2023).
///
/// Owns a [`ChainPollConfig`] and a pluggable [`ChainOfThoughtJudge`]. Calling
/// [`ChainPollScorer::poll`] generates `N` deterministic prompt formulations
/// for a claim, judges each one, un-inverts every raw verdict, and
/// majority-votes the result into a [`ChainPollResult`].
#[derive(Debug)]
pub struct ChainPollScorer {
    /// Configuration controlling the number of formulations and the
    /// majority-vote threshold.
    pub config: ChainPollConfig,
    /// The chain-of-thought judge dispatched for every formulation.
    pub judge: Box<dyn ChainOfThoughtJudge>,
}

impl ChainPollScorer {
    /// Construct a new scorer with the given configuration and judge.
    #[must_use]
    pub fn new(config: ChainPollConfig, judge: Box<dyn ChainOfThoughtJudge>) -> Self {
        Self { config, judge }
    }

    /// Generate the `N` = [`ChainPollConfig::num_formulations`] deterministic
    /// prompt formulations for `claim` against `context`.
    ///
    /// Pure and deterministic: the same claim, context, and configuration
    /// always produce byte-identical formulations in the same order. No
    /// sampling or randomness is involved anywhere in this function -- this
    /// is what distinguishes `ChainPoll`'s "polling" from every
    /// stochastic-resampling technique in this crate.
    ///
    /// # Errors
    ///
    /// - [`ChainPollError::EmptyClaim`] -- `claim` is blank.
    /// - [`ChainPollError::EmptyContext`] -- `context` is blank.
    /// - [`ChainPollError::InvalidConfig`] -- `self.config` fails
    ///   [`ChainPollConfig::validate`].
    ///
    /// # Panics
    ///
    /// Does not panic in practice. [`ChainPollConfig::validate`] (called
    /// above the slicing below) guarantees
    /// `num_formulations <= ChainPollConfig::MAX_FORMULATIONS`, and the
    /// `formulation_bank_boundaries_match_max_formulations` test enforces
    /// that [`ChainPollConfig::MAX_FORMULATIONS`] equals
    /// `FORMULATION_BANK`'s length, so the slice below is always in
    /// bounds.
    pub fn generate_formulations(
        &self,
        claim: &str,
        context: &str,
    ) -> Result<Vec<PollFormulation>, ChainPollError> {
        if claim.trim().is_empty() {
            return Err(ChainPollError::EmptyClaim);
        }
        if context.trim().is_empty() {
            return Err(ChainPollError::EmptyContext);
        }
        self.config.validate()?;

        let formulations = FORMULATION_BANK[..self.config.num_formulations]
            .iter()
            .copied()
            .enumerate()
            .map(|(index, (template_id, framing, render))| PollFormulation {
                index,
                template_id,
                framing,
                prompt: render(claim, context),
            })
            .collect();

        Ok(formulations)
    }

    /// Poll `claim` against `context`: generate the deterministic
    /// formulations, judge each with the configured
    /// [`ChainOfThoughtJudge`], un-invert every raw verdict via its
    /// formulation's [`PollFraming`], and majority-vote the un-inverted
    /// verdicts into a [`ChainPollResult`].
    ///
    /// # Errors
    ///
    /// - [`ChainPollError::EmptyClaim`] / [`ChainPollError::EmptyContext`] /
    ///   [`ChainPollError::InvalidConfig`] -- propagated from
    ///   [`ChainPollScorer::generate_formulations`].
    /// - Any error returned by the configured [`ChainOfThoughtJudge`] is
    ///   propagated as-is, aborting the poll on the first failing
    ///   formulation.
    pub fn poll(&self, claim: &str, context: &str) -> Result<ChainPollResult, ChainPollError> {
        let formulations = self.generate_formulations(claim, context)?;

        let mut formulation_verdicts = Vec::with_capacity(formulations.len());
        for formulation in formulations {
            let (raw_verdict, reasoning) = self.judge.judge(claim, context, &formulation)?;
            let supported = formulation.framing.resolve_supported(raw_verdict);
            formulation_verdicts.push(ChainPollFormulationVerdict {
                formulation,
                raw_verdict,
                supported,
                reasoning,
            });
        }

        debug_assert_eq!(formulation_verdicts.len(), self.config.num_formulations);

        let total = formulation_verdicts.len();
        let hallucinated_votes = formulation_verdicts.iter().filter(|v| !v.supported).count();

        #[allow(clippy::cast_precision_loss)]
        let hallucination_score = if total == 0 {
            0.0_f32
        } else {
            hallucinated_votes as f32 / total as f32
        };

        let confidence = ((hallucination_score - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        let is_hallucination = hallucination_score >= self.config.hallucination_threshold;

        Ok(ChainPollResult {
            claim: claim.to_string(),
            is_hallucination,
            hallucination_score,
            confidence,
            formulation_verdicts,
        })
    }
}

impl Default for ChainPollScorer {
    fn default() -> Self {
        Self::new(
            ChainPollConfig::default(),
            Box::new(MockChainPollJudge::default()),
        )
    }
}
