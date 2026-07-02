//! Types and traits for the `chainpoll` module.
//!
//! Defines the deterministic [`PollFormulation`] and [`PollFraming`] that
//! distinguish `ChainPoll`'s *prompt-formulation* polling from every
//! sampling-based technique in this crate, the pluggable
//! [`ChainOfThoughtJudge`] trait (with the deterministic
//! [`MockChainPollJudge`] proxy for tests), and the aggregate
//! [`ChainPollResult`].

use std::collections::HashSet;
use std::fmt;

use thiserror::Error;

// ── PollFraming ──────────────────────────────────────────────────────────────

/// How a [`PollFormulation`]'s raw yes/no verdict maps onto "the claim is
/// supported by the context".
///
/// `ChainPoll` formulations are not all phrased as "is this claim true?" --
/// some are deliberately phrased as the *opposite* question (e.g. "does the
/// context **contradict** this claim?") to diversify the prompt surface.
/// [`PollFraming`] records which direction a given formulation's raw yes/no
/// answer runs in, so aggregation can correctly un-invert it before voting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PollFraming {
    /// A raw `true` ("yes") verdict means the claim IS supported by the
    /// context. No inversion is needed.
    Direct,
    /// A raw `true` ("yes") verdict means the claim is NOT supported by (or
    /// is contradicted by) the context. Must be inverted (`!raw_verdict`) to
    /// recover "claim is supported".
    Inverted,
}

impl PollFraming {
    /// Resolve a raw yes/no `verdict` -- the literal answer to this
    /// formulation's question -- into "the claim is supported by the
    /// context", inverting when `self` is [`PollFraming::Inverted`].
    ///
    /// This transform is its own inverse: applying it twice recovers the
    /// original value, so the same function also converts a "claim
    /// supported" answer back into the raw verdict this framing would have
    /// produced for it.
    #[must_use]
    pub fn resolve_supported(&self, raw_verdict: bool) -> bool {
        match self {
            Self::Direct => raw_verdict,
            Self::Inverted => !raw_verdict,
        }
    }
}

impl fmt::Display for PollFraming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Direct => "direct",
            Self::Inverted => "inverted",
        })
    }
}

// ── PollFormulation ──────────────────────────────────────────────────────────

/// One deterministic prompt-formulation variant of the chain-of-thought
/// hallucination-check question for a single claim.
///
/// `ChainPoll`'s defining mechanism is polling *structurally distinct
/// prompts* -- different phrasings, orderings, and perspectives of the same
/// underlying support question -- rather than re-sampling one fixed prompt at
/// nonzero temperature. Every formulation for a given claim/context pair is
/// produced by
/// [`ChainPollScorer::generate_formulations`](crate::chainpoll::scorer::ChainPollScorer::generate_formulations)
/// from a fixed, ordered bank of templates: the same inputs always yield the
/// same `N` formulations, in the same order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PollFormulation {
    /// Position of this formulation within the fixed formulation bank
    /// (`0`-based), stable across calls for the same claim/context.
    pub index: usize,
    /// Stable identifier for the template that produced this formulation
    /// (e.g. `"direct_support"`, `"inverted_contradiction"`).
    pub template_id: &'static str,
    /// How this formulation's raw yes/no verdict maps to "claim supported".
    pub framing: PollFraming,
    /// The fully-rendered chain-of-thought prompt text, with the claim and
    /// context already interpolated.
    pub prompt: String,
}

// ── ChainOfThoughtJudge ───────────────────────────────────────────────────────

/// A pluggable chain-of-thought yes/no judge for one poll formulation.
///
/// Implementations answer the question posed by `formulation` (a specific
/// phrasing of "is this claim grounded in the context?") for the given
/// `claim` and `context`, returning the *raw* (not yet un-inverted) yes/no
/// verdict together with the reasoning that led to it. Callers wire this
/// trait up to an actual LLM chain-of-thought call in production;
/// [`MockChainPollJudge`] is a deterministic lexical stand-in for tests.
///
/// # Object Safety
///
/// This trait is object-safe and is stored as `Box<dyn ChainOfThoughtJudge>`
/// by [`ChainPollScorer`](crate::chainpoll::scorer::ChainPollScorer).
pub trait ChainOfThoughtJudge: fmt::Debug {
    /// Judge whether `claim` is grounded in `context` under the specific
    /// question posed by `formulation`.
    ///
    /// Returns `(raw_verdict, reasoning)`: `raw_verdict` is the literal
    /// yes/no answer to `formulation`'s question (before applying
    /// [`PollFraming::resolve_supported`]), and `reasoning` is a
    /// human-readable chain-of-thought trace supporting that answer.
    ///
    /// # Errors
    ///
    /// Returns a [`ChainPollError`] when the judge cannot produce a verdict
    /// (e.g. empty inputs, or an underlying model/transport failure in a
    /// real LLM-backed implementation).
    fn judge(
        &self,
        claim: &str,
        context: &str,
        formulation: &PollFormulation,
    ) -> Result<(bool, String), ChainPollError>;
}

// ── MockChainPollJudge ─────────────────────────────────────────────────────────

/// Deterministic lexical-overlap [`ChainOfThoughtJudge`] used for tests and
/// prototyping.
///
/// Computes the Jaccard token overlap between `claim` and `context`; the
/// underlying "is the claim supported" heuristic is
/// `overlap >= support_threshold`. That heuristic answer is identical for
/// every formulation of a given claim/context pair -- formulation phrasing
/// only changes *how the answer is encoded* (via [`PollFraming`]) and the
/// generated reasoning text, never the underlying judgment -- so this mock
/// still faithfully exercises the un-inversion path without pretending to
/// model a real LLM's phrasing-sensitivity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MockChainPollJudge {
    /// Minimum Jaccard token overlap between `claim` and `context` required
    /// to consider the claim supported. Default: `0.15`.
    pub support_threshold: f32,
}

impl Default for MockChainPollJudge {
    fn default() -> Self {
        Self {
            support_threshold: 0.15,
        }
    }
}

impl MockChainPollJudge {
    /// Construct a mock judge with an explicit overlap threshold.
    #[must_use]
    pub fn new(support_threshold: f32) -> Self {
        Self { support_threshold }
    }
}

/// Tokenize `text`: split on non-alphanumeric boundaries, lowercase, and drop
/// empty fragments.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard similarity between two token sets. Returns `0.0` when both are
/// empty.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let score = intersection as f32 / union as f32;
    score
}

impl ChainOfThoughtJudge for MockChainPollJudge {
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

        let overlap = jaccard(&tokenize(claim), &tokenize(context));
        let claim_supported = overlap >= self.support_threshold;
        let raw_verdict = formulation.framing.resolve_supported(claim_supported);

        let reasoning = format!(
            "[{}, {} framing] claim/context lexical overlap = {overlap:.2} (threshold {:.2}); \
             the claim is {} by the context, so the raw answer to this formulation's question is {}.",
            formulation.template_id,
            formulation.framing,
            self.support_threshold,
            if claim_supported {
                "supported"
            } else {
                "not supported"
            },
            if raw_verdict { "yes" } else { "no" },
        );

        Ok((raw_verdict, reasoning))
    }
}

// ── ChainPollConfig ────────────────────────────────────────────────────────────

/// Configuration for [`ChainPollScorer`](crate::chainpoll::scorer::ChainPollScorer).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainPollConfig {
    /// Number of deterministic prompt formulations to poll per claim. Must
    /// be in `1..=ChainPollConfig::MAX_FORMULATIONS`. Default: `5`.
    pub num_formulations: usize,
    /// Hallucination-score threshold (in `[0.0, 1.0]`) at or above which the
    /// majority verdict is [`ChainPollResult::is_hallucination`]. Default:
    /// `0.5` (a plain majority).
    pub hallucination_threshold: f32,
}

impl Default for ChainPollConfig {
    fn default() -> Self {
        Self {
            num_formulations: 5,
            hallucination_threshold: 0.5,
        }
    }
}

impl ChainPollConfig {
    /// Number of templates in the built-in deterministic formulation bank;
    /// the maximum valid value for [`ChainPollConfig::num_formulations`].
    pub const MAX_FORMULATIONS: usize = 8;

    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of prompt formulations to poll.
    #[must_use]
    pub fn with_num_formulations(mut self, num_formulations: usize) -> Self {
        self.num_formulations = num_formulations;
        self
    }

    /// Set the majority hallucination threshold.
    #[must_use]
    pub fn with_hallucination_threshold(mut self, hallucination_threshold: f32) -> Self {
        self.hallucination_threshold = hallucination_threshold;
        self
    }

    /// Validate this configuration's numeric invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ChainPollError::InvalidConfig`] when:
    /// - `num_formulations == 0`;
    /// - `num_formulations > ChainPollConfig::MAX_FORMULATIONS`;
    /// - `hallucination_threshold` is outside `[0.0, 1.0]` or non-finite.
    pub fn validate(&self) -> Result<(), ChainPollError> {
        if self.num_formulations == 0 {
            return Err(ChainPollError::InvalidConfig(
                "num_formulations must be at least 1".to_string(),
            ));
        }
        if self.num_formulations > Self::MAX_FORMULATIONS {
            return Err(ChainPollError::InvalidConfig(format!(
                "num_formulations must be at most {} (the size of the built-in deterministic \
                 formulation bank), got {}",
                Self::MAX_FORMULATIONS,
                self.num_formulations
            )));
        }
        if !self.hallucination_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.hallucination_threshold)
        {
            return Err(ChainPollError::InvalidConfig(format!(
                "hallucination_threshold must be a finite value in [0.0, 1.0], got {}",
                self.hallucination_threshold
            )));
        }
        Ok(())
    }
}

// ── ChainPollFormulationVerdict ─────────────────────────────────────────────────

/// The per-formulation outcome recorded inside a [`ChainPollResult`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChainPollFormulationVerdict {
    /// The formulation that was judged.
    pub formulation: PollFormulation,
    /// The raw yes/no verdict, as literally answered to this formulation's
    /// question (before un-inverting via [`PollFraming::resolve_supported`]).
    pub raw_verdict: bool,
    /// `true` when this formulation, after un-inverting, considers the claim
    /// supported by the context; `false` when it votes hallucinated.
    pub supported: bool,
    /// The chain-of-thought reasoning text produced by the judge.
    pub reasoning: String,
}

// ── ChainPollResult ──────────────────────────────────────────────────────────

/// The aggregated outcome of polling `N` deterministic prompt formulations
/// for a single claim.
///
/// Produced by [`ChainPollScorer::poll`](crate::chainpoll::scorer::ChainPollScorer::poll).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainPollResult {
    /// The claim that was polled.
    pub claim: String,
    /// Majority verdict: `true` when `hallucination_score >=
    /// `[`ChainPollConfig::hallucination_threshold`].
    pub is_hallucination: bool,
    /// Fraction of formulations (in `[0.0, 1.0]`) whose un-inverted verdict
    /// voted "hallucinated" / unsupported. `0.0` means unanimous support,
    /// `1.0` means unanimous hallucination, and values near `0.5` mean an
    /// evenly split poll.
    pub hallucination_score: f32,
    /// Calibrated confidence in the majority verdict, in `[0.0, 1.0]`:
    /// `2 * |hallucination_score - 0.5|`. `0.0` at an exact tie, `1.0` for a
    /// unanimous poll.
    pub confidence: f32,
    /// Per-formulation breakdown, in formulation order, for introspection.
    pub formulation_verdicts: Vec<ChainPollFormulationVerdict>,
}

impl ChainPollResult {
    /// Number of formulations polled.
    #[must_use]
    pub fn num_formulations(&self) -> usize {
        self.formulation_verdicts.len()
    }

    /// Number of formulations whose un-inverted verdict supported the claim.
    #[must_use]
    pub fn supported_count(&self) -> usize {
        self.formulation_verdicts
            .iter()
            .filter(|v| v.supported)
            .count()
    }

    /// Number of formulations whose un-inverted verdict voted hallucinated.
    #[must_use]
    pub fn hallucinated_count(&self) -> usize {
        self.formulation_verdicts
            .iter()
            .filter(|v| !v.supported)
            .count()
    }
}

// ── ChainPollError ───────────────────────────────────────────────────────────

/// Errors produced by the `chainpoll` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ChainPollError {
    /// The claim text was empty or contained only whitespace.
    #[error("claim must not be empty")]
    EmptyClaim,
    /// The context text was empty or contained only whitespace.
    #[error("context must not be empty")]
    EmptyContext,
    /// The supplied [`ChainPollConfig`] failed validation.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// A [`ChainOfThoughtJudge`] implementation failed to produce a verdict.
    #[error("chain-of-thought judge failed: {0}")]
    JudgeFailed(String),
}
