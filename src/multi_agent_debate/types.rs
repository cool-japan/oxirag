//! Core data types, configuration, error enum, and the pluggable
//! [`DebatePersona`] / [`DebateJudge`] traits for the `multi_agent_debate`
//! module.
//!
//! The debate loop (round orchestration), the deterministic
//! [`MockDebatePersona`](crate::multi_agent_debate::MockDebatePersona) and
//! [`MockDebateJudge`](crate::multi_agent_debate::MockDebateJudge)
//! implementations, and the judging heuristics all live in
//! [`super::engine`].

use thiserror::Error;

// ── DebateError ──────────────────────────────────────────────────────────────

/// Errors produced by the `multi_agent_debate` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DebateError {
    /// The question was empty or contained only whitespace.
    #[error("question must not be empty")]
    EmptyQuestion,

    /// Fewer than two participants were supplied; a debate requires at
    /// least two (possibly opposing) positions.
    #[error("a debate requires at least 2 personas, got {count}")]
    TooFewPersonas {
        /// The number of participants actually supplied.
        count: usize,
    },

    /// [`DebateConfig::max_rounds`] was `0`.
    #[error("max_rounds must be at least 1, got 0")]
    ZeroRounds,

    /// A participant's assigned position was empty or whitespace-only.
    #[error("participant {index} has an empty position")]
    EmptyPosition {
        /// The zero-based index (into the participant list) of the
        /// offending participant.
        index: usize,
    },

    /// Two or more participants were assigned the exact same position
    /// text.
    ///
    /// Positions double as participant identity within a transcript (see
    /// [`DebatePersona::argue`]), so duplicate positions would leave a
    /// participant unable to distinguish its own prior arguments from an
    /// opponent's.
    #[error("position {position:?} is assigned to more than one participant")]
    DuplicatePosition {
        /// The position text shared by more than one participant.
        position: String,
    },

    /// A [`DebatePersona`] implementation failed to produce an argument.
    ///
    /// Provided for custom (non-mock) implementations backed by a real
    /// model or service, so a failure (timeout, refusal, malformed
    /// response, ...) can be reported without extending this enum.
    #[error("persona failed to produce an argument: {reason}")]
    PersonaGenerationFailed {
        /// A human-readable reason for the failure.
        reason: String,
    },

    /// A [`DebateJudge`] implementation failed to produce a verdict.
    #[error("judge failed to produce a verdict: {reason}")]
    JudgeFailed {
        /// A human-readable reason for the failure.
        reason: String,
    },

    /// A [`DebateJudge`] was asked to judge a transcript with zero rounds.
    #[error("cannot judge an empty transcript (no rounds)")]
    EmptyTranscript,
}

// ── DebateJudgeWeights ───────────────────────────────────────────────────────

/// Relative weighting of the three scoring dimensions
/// [`MockDebateJudge`](crate::multi_agent_debate::MockDebateJudge) combines
/// into each side's [`DebatePositionScore::total_score`]: normalized
/// argument count, average self-reported confidence, and rebuttal
/// engagement (lexical overlap with the immediately preceding opposing
/// argument).
///
/// The weights need not sum to `1.0` —
/// [`MockDebateJudge`](crate::multi_agent_debate::MockDebateJudge) applies
/// them as raw linear coefficients over inputs that are each already
/// normalized to `[0.0, 1.0]`, so `total_score` itself lands in
/// `[0.0, argument_count_weight + confidence_weight + rebuttal_weight]`.
///
/// This is deliberately a plain, freestanding config struct rather than a
/// field consulted directly by [`DebateEngine`](crate::multi_agent_debate::DebateEngine), since judging strategy is
/// owned by whichever [`DebateJudge`] implementation is supplied to
/// [`DebateEngine::run`](crate::multi_agent_debate::DebateEngine::run) (mirroring how
/// [`crate::searchain::MockSearchainGenerator`] owns its own
/// `AnswerAssemblyStrategy` rather than reading it from
/// `crate::searchain::SearChainConfig`): a
/// [`MockDebateJudge`](crate::multi_agent_debate::MockDebateJudge) takes its
/// own `DebateJudgeWeights` directly, and [`DebateConfig::judge_weights`]
/// simply carries the *recommended* defaults a caller may thread through
/// when constructing one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebateJudgeWeights {
    /// Weight applied to each side's argument count, normalized against the
    /// highest count among all sides in the transcript. Defaults to `0.2`.
    pub argument_count_weight: f32,
    /// Weight applied to each side's average self-reported confidence
    /// (already in `[0.0, 1.0]`). Defaults to `0.3`.
    pub confidence_weight: f32,
    /// Weight applied to each side's rebuttal engagement: how directly its
    /// arguments engaged with the immediately preceding opposing argument
    /// (already in `[0.0, 1.0]`). Defaults to `0.5`, the dominant term,
    /// since directly engaging with an opponent is the defining trait of
    /// adversarial debate — as opposed to independently-sampled reasoning
    /// (see the [module-level documentation](crate::multi_agent_debate)).
    pub rebuttal_weight: f32,
}

impl Default for DebateJudgeWeights {
    fn default() -> Self {
        Self {
            argument_count_weight: 0.2,
            confidence_weight: 0.3,
            rebuttal_weight: 0.5,
        }
    }
}

impl DebateJudgeWeights {
    /// Create a new set of weights.
    #[must_use]
    pub fn new(argument_count_weight: f32, confidence_weight: f32, rebuttal_weight: f32) -> Self {
        Self {
            argument_count_weight,
            confidence_weight,
            rebuttal_weight,
        }
    }

    /// Set the argument-count weight.
    #[must_use]
    pub fn with_argument_count_weight(mut self, weight: f32) -> Self {
        self.argument_count_weight = weight;
        self
    }

    /// Set the average-confidence weight.
    #[must_use]
    pub fn with_confidence_weight(mut self, weight: f32) -> Self {
        self.confidence_weight = weight;
        self
    }

    /// Set the rebuttal-engagement weight.
    #[must_use]
    pub fn with_rebuttal_weight(mut self, weight: f32) -> Self {
        self.rebuttal_weight = weight;
        self
    }
}

// ── DebateConfig ─────────────────────────────────────────────────────────────

/// Configuration for [`DebateEngine`](crate::multi_agent_debate::DebateEngine).
///
/// Participants (personas paired with their positions) are supplied
/// per-call to [`DebateEngine::run`](crate::multi_agent_debate::DebateEngine::run)
/// as `&[DebateParticipant]`, not stored here — mirroring the
/// caller-supplies-executor convention used throughout this crate (e.g.
/// `crate::searchain::SearChainEngine::run`).
#[derive(Debug, Clone, PartialEq)]
pub struct DebateConfig {
    /// Number of debate rounds to run — each round, every participant
    /// argues exactly once — unless early stopping cuts the debate short.
    /// Must be at least `1`. Defaults to `3`.
    pub max_rounds: usize,
    /// Whether the debate may stop before `max_rounds` when a convergence
    /// signal fires (see [`DebateConfig::flat_confidence_window`]).
    /// Defaults to `false`.
    pub early_stop: bool,
    /// Number of consecutive round-over-round comparisons in which a
    /// participant's self-reported confidence must fail to improve (by more
    /// than [`DebateConfig::flat_confidence_epsilon`]) before early stopping
    /// fires for that participant. Only consulted when
    /// [`DebateConfig::early_stop`] is `true`. Defaults to `2`.
    pub flat_confidence_window: usize,
    /// The maximum increase in confidence, from one round to the next, that
    /// still counts as "flat or declining" for the purposes of
    /// [`DebateConfig::flat_confidence_window`]. Defaults to `0.02`.
    pub flat_confidence_epsilon: f32,
    /// Recommended judge scoring weights. Not consulted by
    /// [`DebateEngine`](crate::multi_agent_debate::DebateEngine) directly
    /// (the judge is a caller-supplied, opaque [`DebateJudge`]); provided so
    /// a caller can build a
    /// [`MockDebateJudge`](crate::multi_agent_debate::MockDebateJudge)
    /// consistent with the rest of the configuration, e.g.
    /// `MockDebateJudge::new(config.judge_weights)`. Defaults to
    /// [`DebateJudgeWeights::default`].
    pub judge_weights: DebateJudgeWeights,
}

impl Default for DebateConfig {
    fn default() -> Self {
        Self {
            max_rounds: 3,
            early_stop: false,
            flat_confidence_window: 2,
            flat_confidence_epsilon: 0.02,
            judge_weights: DebateJudgeWeights::default(),
        }
    }
}

impl DebateConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of debate rounds.
    #[must_use]
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    /// Enable or disable convergence-based early stopping.
    #[must_use]
    pub fn with_early_stop(mut self, early_stop: bool) -> Self {
        self.early_stop = early_stop;
        self
    }

    /// Set the consecutive-round window for the flat-confidence early-stop
    /// signal.
    #[must_use]
    pub fn with_flat_confidence_window(mut self, flat_confidence_window: usize) -> Self {
        self.flat_confidence_window = flat_confidence_window;
        self
    }

    /// Set the flat-confidence tolerance.
    #[must_use]
    pub fn with_flat_confidence_epsilon(mut self, flat_confidence_epsilon: f32) -> Self {
        self.flat_confidence_epsilon = flat_confidence_epsilon;
        self
    }

    /// Set the recommended judge scoring weights.
    #[must_use]
    pub fn with_judge_weights(mut self, judge_weights: DebateJudgeWeights) -> Self {
        self.judge_weights = judge_weights;
        self
    }
}

// ── DebateArgument ───────────────────────────────────────────────────────────

/// A single argument produced by one participant during one round of a
/// debate.
///
/// `persona_index`, `position`, and `round` are bookkeeping fields owned by
/// [`DebateEngine`](crate::multi_agent_debate::DebateEngine): after calling
/// [`DebatePersona::argue`], the engine always overwrites these three
/// fields on the returned value with the ground-truth values for the call
/// that was actually made, regardless of what the implementation returned,
/// so a transcript's bookkeeping stays trustworthy even for careless
/// [`DebatePersona`] implementations. Implementations are responsible for
/// `text` and `confidence` only — see [`DebateArgument::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct DebateArgument {
    /// Zero-based index, into the participant list a debate was run with,
    /// of the persona that produced this argument.
    pub persona_index: usize,
    /// The position (stance) this argument was made in support of.
    pub position: String,
    /// Zero-based round number this argument was produced in.
    pub round: usize,
    /// The argument's text.
    pub text: String,
    /// The persona's self-reported confidence in this argument, in
    /// `[0.0, 1.0]`.
    pub confidence: f32,
}

impl DebateArgument {
    /// Create a new argument with placeholder bookkeeping fields
    /// (`persona_index: 0`, `position: String::new()`, `round: 0`).
    ///
    /// [`DebatePersona::argue`] implementations can build their return
    /// value with this and leave the bookkeeping fields at their
    /// placeholder values: [`DebateEngine`](crate::multi_agent_debate::DebateEngine)
    /// overwrites them unconditionally after the call (see the struct-level
    /// documentation). `confidence` is clamped into `[0.0, 1.0]`.
    #[must_use]
    pub fn new(text: impl Into<String>, confidence: f32) -> Self {
        Self {
            persona_index: 0,
            position: String::new(),
            round: 0,
            text: text.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

// ── DebateRound ──────────────────────────────────────────────────────────────

/// All participants' arguments from a single round of a debate.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DebateRound {
    /// Zero-based round number.
    pub round_index: usize,
    /// Each participant's argument for this round, in participant-list
    /// order (i.e. `arguments[i].persona_index == i` for a well-formed
    /// round produced by [`DebateEngine`](crate::multi_agent_debate::DebateEngine)).
    pub arguments: Vec<DebateArgument>,
}

impl DebateRound {
    /// Create a new, empty round.
    #[must_use]
    pub fn new(round_index: usize) -> Self {
        Self {
            round_index,
            arguments: Vec::new(),
        }
    }

    /// The argument made in support of `position` during this round, if
    /// any.
    #[must_use]
    pub fn argument_for(&self, position: &str) -> Option<&DebateArgument> {
        self.arguments.iter().find(|a| a.position == position)
    }
}

// ── DebatePositionScore ──────────────────────────────────────────────────────

/// A [`DebateJudge`]'s scoring breakdown for a single position, as computed
/// by [`MockDebateJudge`](crate::multi_agent_debate::MockDebateJudge).
///
/// Custom [`DebateJudge`] implementations are free to compute
/// [`DebateVerdict::scores`] by an entirely different method; this struct's
/// fields describe
/// [`MockDebateJudge`](crate::multi_agent_debate::MockDebateJudge)'s
/// particular heuristics specifically.
#[derive(Debug, Clone, PartialEq)]
pub struct DebatePositionScore {
    /// The persona index this score belongs to.
    pub persona_index: usize,
    /// The position text this score belongs to.
    pub position: String,
    /// Number of arguments this side made across the transcript.
    pub argument_count: usize,
    /// This side's argument count, normalized against the highest argument
    /// count among all sides in the transcript (`[0.0, 1.0]`).
    pub normalized_argument_count: f32,
    /// Average self-reported confidence across this side's arguments
    /// (`[0.0, 1.0]`).
    pub average_confidence: f32,
    /// How directly this side's arguments engaged with the immediately
    /// preceding opposing argument, averaged over every argument that had
    /// one to engage with (`[0.0, 1.0]`; `0.0` when this side never had an
    /// opposing argument to respond to, e.g. it only ever spoke in round
    /// `0`).
    pub rebuttal_engagement: f32,
    /// The weighted combination of the three fields above, per the judge's
    /// [`DebateJudgeWeights`].
    pub total_score: f32,
}

// ── DebateVerdict ────────────────────────────────────────────────────────────

/// A [`DebateJudge`]'s decision after reviewing a complete debate
/// transcript: the winning position, a human-readable rationale, and a
/// per-position score breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct DebateVerdict {
    /// The winning position's text.
    pub winning_position: String,
    /// The winning position's persona index.
    pub winning_persona_index: usize,
    /// A human-readable rationale for the verdict, referencing the
    /// transcript it was derived from (round count, the winning position,
    /// and its scoring breakdown).
    pub rationale: String,
    /// Every position's scoring breakdown, sorted by descending total
    /// score (`scores[0]` is the winner).
    pub scores: Vec<DebatePositionScore>,
}

impl DebateVerdict {
    /// The scoring breakdown for `position`, if it took part in the debate.
    #[must_use]
    pub fn score_for(&self, position: &str) -> Option<&DebatePositionScore> {
        self.scores.iter().find(|s| s.position == position)
    }
}

// ── DebateResult ─────────────────────────────────────────────────────────────

/// The complete output of a
/// [`DebateEngine`](crate::multi_agent_debate::DebateEngine) run.
#[derive(Debug, Clone, PartialEq)]
pub struct DebateResult {
    /// The original question the debate addressed.
    pub question: String,
    /// The full multi-round transcript, in round order.
    pub transcript: Vec<DebateRound>,
    /// The judge's final verdict.
    pub verdict: DebateVerdict,
    /// Number of rounds actually run. Less than or equal to
    /// `DebateConfig::max_rounds`; strictly less than it only when early
    /// stopping fired.
    pub rounds_run: usize,
    /// `true` when the debate stopped before `DebateConfig::max_rounds` due
    /// to the convergence signal firing.
    pub stopped_early: bool,
    /// A human-readable explanation of why the debate stopped early, or
    /// `None` when it was not stopped early.
    pub early_stop_reason: Option<String>,
}

impl DebateResult {
    /// Every argument from every round, in round-then-participant order.
    #[must_use]
    pub fn all_arguments(&self) -> Vec<&DebateArgument> {
        self.transcript
            .iter()
            .flat_map(|round| round.arguments.iter())
            .collect()
    }

    /// Shortcut for `self.verdict.winning_position`.
    #[must_use]
    pub fn winning_position(&self) -> &str {
        &self.verdict.winning_position
    }
}

// ── DebateParticipant ────────────────────────────────────────────────────────

/// One debate participant: a pluggable [`DebatePersona`] paired with the
/// position (stance) it argues for.
///
/// Participants are supplied per-call to
/// [`DebateEngine::run`](crate::multi_agent_debate::DebateEngine::run) as a
/// borrowed slice — using a trait object (rather than a single generic type
/// parameter shared by every participant) so that heterogeneous pluggable
/// implementations can sit side by side in one debate, without requiring
/// every persona to share a single concrete type (mirroring
/// `crate::ensemble_retriever::EnsembleRetriever`'s use of
/// `Box<dyn SubRetriever>` for the same reason).
#[derive(Clone, Copy)]
pub struct DebateParticipant<'a> {
    /// The persona arguing this participant's position.
    pub persona: &'a dyn DebatePersona,
    /// The position (stance) this participant argues for. Must be
    /// non-empty and distinct from every other participant's position in
    /// the same debate.
    pub position: &'a str,
}

impl<'a> DebateParticipant<'a> {
    /// Create a new participant.
    #[must_use]
    pub fn new(persona: &'a dyn DebatePersona, position: &'a str) -> Self {
        Self { persona, position }
    }
}

// ── DebatePersona ────────────────────────────────────────────────────────────

/// A pluggable participant in a multi-agent debate.
///
/// Implementations produce one new [`DebateArgument`] per call, given the
/// question, the position they are arguing for, and the **complete**
/// transcript of every argument made so far by every participant —
/// including their own prior arguments and every opponent's —
/// [`DebateEngine::run`](crate::multi_agent_debate::DebateEngine::run) never
/// truncates or filters this history down to just the caller's own side.
/// This is the central mechanic that distinguishes adversarial debate from
/// `crate::self_consistency`'s independently-sampled reasoning chains: a
/// `DebatePersona` can read, quote, and directly rebut what an opponent
/// just argued (see the [module-level documentation](crate::multi_agent_debate)).
///
/// Implementations should use `position` — rather than trying to infer a
/// numeric identity — to tell their own prior arguments apart from an
/// opponent's within `prior_arguments`, by comparing against
/// [`DebateArgument::position`]; positions are validated to be unique
/// within a debate by
/// [`DebateEngine::run`](crate::multi_agent_debate::DebateEngine::run). The
/// [`DebateArgument::persona_index`] field on the *returned* argument (along
/// with `position` and `round`) is bookkeeping the engine always overwrites
/// with ground truth after the call — see [`DebateArgument`]'s
/// documentation — so an implementation only needs to get `text` and
/// `confidence` right (e.g. via [`DebateArgument::new`]).
///
/// [`MockDebatePersona`](crate::multi_agent_debate::MockDebatePersona)
/// provides a deterministic implementation for tests.
pub trait DebatePersona {
    /// Produce this participant's next argument.
    ///
    /// `prior_arguments` holds every argument made so far, by every
    /// participant, across every completed round — see the trait-level
    /// documentation.
    ///
    /// # Errors
    ///
    /// Implementations may return [`DebateError::PersonaGenerationFailed`]
    /// (or any other [`DebateError`] variant that genuinely applies) when
    /// they cannot produce an argument.
    fn argue(
        &self,
        question: &str,
        position: &str,
        prior_arguments: &[DebateArgument],
    ) -> Result<DebateArgument, DebateError>;
}

// ── DebateJudge ──────────────────────────────────────────────────────────────

/// A pluggable judge that reviews a complete debate transcript and decides
/// a winner.
///
/// Unlike a [`DebatePersona`], a `DebateJudge` is consulted exactly once per
/// debate, after every round has been played out, and never argues for a
/// position itself — it only evaluates.
///
/// [`MockDebateJudge`](crate::multi_agent_debate::MockDebateJudge) provides
/// a deterministic implementation for tests.
pub trait DebateJudge {
    /// Review `transcript` (every round of the debate, in order) and decide
    /// a winning position.
    ///
    /// # Errors
    ///
    /// Implementations may return [`DebateError::JudgeFailed`],
    /// [`DebateError::EmptyTranscript`], or any other [`DebateError`]
    /// variant that genuinely applies.
    fn judge(
        &self,
        question: &str,
        transcript: &[DebateRound],
    ) -> Result<DebateVerdict, DebateError>;
}
