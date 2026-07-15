//! Core types for the `process_reward_model` module.
//!
//! This file defines the data model a process reward model (`PRM`) works on —
//! reasoning steps and trajectories, per-step labels and scores, rollout records,
//! configuration, and errors — together with the three traits the engine is
//! written against: [`RolloutPolicy`] (the stochastic step completer),
//! [`ProcessRewardModel`] (scores *every* step), and [`OutcomeRewardModel`]
//! (scores *only* the final answer). It also ships [`PrmRng`], the module's
//! deterministic `SplitMix64` generator.

use thiserror::Error;

// ── PrmRng ────────────────────────────────────────────────────────────────────

/// The 64-bit odd golden-ratio increment of the `SplitMix64` Weyl sequence.
const PRM_GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// First avalanche multiplier of the `SplitMix64` finalizer.
const PRM_MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second avalanche multiplier of the `SplitMix64` finalizer.
const PRM_MIX_2: u64 = 0x94D0_49BB_1331_11EB;

/// A deterministic, dependency-free `SplitMix64` pseudo-random generator.
///
/// Monte Carlo rollout labelling is only defined relative to a source of
/// randomness. This crate takes no dependency on `rand` (see the project's
/// `SciRS2` policy), so — following the convention already set by
/// `bandit_ranker` and `click_model`, each of which ships its own prefixed
/// generator — this module hand-rolls `SplitMix64`. Seeded once, it reproduces
/// its entire stream exactly, which is what makes the labeller's determinism
/// contract meaningful: the same [`PrmConfig::seed`] replayed against the same
/// trajectory yields byte-identical labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrmRng {
    /// The Weyl-sequence state, advanced by [`PRM_GOLDEN_GAMMA`] on every draw.
    state: u64,
}

impl PrmRng {
    /// Create a generator seeded with `seed`.
    ///
    /// `SplitMix64` has no degenerate seeds, so `0` is a perfectly ordinary seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next uniformly distributed `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(PRM_GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(PRM_MIX_1);
        z = (z ^ (z >> 27)).wrapping_mul(PRM_MIX_2);
        z ^ (z >> 31)
    }

    /// Draw a uniform `f64` in the half-open interval `[0, 1)`.
    ///
    /// Uses the top 53 bits of a `u64` draw — exactly the width of an `f64`
    /// significand — so every representable multiple of `2^-53` is hit with equal
    /// probability and the conversion carries no rounding bias.
    pub fn next_f64(&mut self) -> f64 {
        /// `2^53`, the number of representable mantissa steps in `[0, 1)`.
        const TWO_POW_53: f64 = 9_007_199_254_740_992.0;
        #[allow(clippy::cast_precision_loss)]
        let mantissa = (self.next_u64() >> 11) as f64;
        mantissa / TWO_POW_53
    }

    /// Draw a Bernoulli outcome that is `true` with probability `probability`.
    ///
    /// Values outside `[0, 1]` are clamped. The draw succeeds exactly when the
    /// underlying uniform lands below `probability`, so over many draws the
    /// success rate converges to `probability`.
    pub fn bernoulli(&mut self, probability: f64) -> bool {
        self.next_f64() < probability.clamp(0.0, 1.0)
    }
}

impl Default for PrmRng {
    /// A generator seeded with `0`; deliberately *not* entropy-seeded, so the
    /// module's reproducibility contract holds by default.
    fn default() -> Self {
        Self::new(0)
    }
}

// ── ReasoningStep ─────────────────────────────────────────────────────────────

/// One intermediate step of a reasoning trajectory.
///
/// A step is the atom a `PRM` assigns a reward to. It is deliberately just text:
/// the module makes no assumption about whether a step is a sentence, an
/// equation, or a tool call — only that a trajectory is an *ordered* sequence of
/// them ending in a final answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningStep {
    /// The textual content of the step.
    pub text: String,
}

impl ReasoningStep {
    /// Create a step from any string-like value.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// Borrow the step's text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl From<&str> for ReasoningStep {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

impl From<String> for ReasoningStep {
    fn from(text: String) -> Self {
        Self { text }
    }
}

// ── PrmTrajectory ─────────────────────────────────────────────────────────────

/// A full reasoning trajectory: an ordered list of steps plus the final answer
/// the chain arrives at.
///
/// The [`answer`](Self::answer) is what an [`OutcomeRewardModel`] scores; the
/// [`steps`](Self::steps) are what a [`ProcessRewardModel`] scores. Two
/// trajectories can share an identical `answer` yet differ in `steps` — the case
/// that motivates the whole module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrmTrajectory {
    /// The ordered reasoning steps.
    pub steps: Vec<ReasoningStep>,
    /// The final answer the trajectory reaches.
    pub answer: String,
}

impl PrmTrajectory {
    /// Assemble a trajectory from steps and a final answer.
    #[must_use]
    pub fn new(steps: Vec<ReasoningStep>, answer: impl Into<String>) -> Self {
        Self {
            steps,
            answer: answer.into(),
        }
    }

    /// Assemble a trajectory from step texts and a final answer.
    #[must_use]
    pub fn from_texts<I, S>(steps: I, answer: impl Into<String>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            steps: steps.into_iter().map(ReasoningStep::new).collect(),
            answer: answer.into(),
        }
    }

    /// The number of steps in the trajectory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the trajectory has no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ── PrmLabelKind ──────────────────────────────────────────────────────────────

/// Which `Math-Shepherd` label variant a step reward is derived from.
///
/// Both are computed from the same rollout batch; this only selects which scalar
/// the [`ProcessRewardModel`] emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrmLabelKind {
    /// The *soft* label: the fraction of rollouts that reach the correct answer.
    #[default]
    Soft,
    /// The *hard* label: `1.0` if *any* rollout reaches the correct answer, else
    /// `0.0`.
    Hard,
}

// ── StepLabel ─────────────────────────────────────────────────────────────────

/// The `Math-Shepherd` automatic label for a single step, produced by Monte
/// Carlo rollout from that step.
///
/// A step's quality is estimated by completing the solution many times *starting
/// from that step* and measuring how often those completions reach the correct
/// final answer. Both label variants are retained: [`soft_label`](Self::soft_label)
/// is the fraction, [`hard_label`](Self::hard_label) is whether the fraction is
/// non-zero.
#[derive(Debug, Clone, PartialEq)]
pub struct StepLabel {
    /// The index (into the trajectory) of the labelled step.
    pub step_index: usize,
    /// How many rollouts were drawn from this step.
    pub num_rollouts: usize,
    /// How many of those rollouts reached the correct final answer.
    pub num_correct: usize,
    /// The soft label: `num_correct / num_rollouts`, in `[0, 1]`.
    pub soft_label: f32,
    /// The hard label: `true` when at least one rollout reached the answer.
    pub hard_label: bool,
}

impl StepLabel {
    /// The scalar reward this label yields under `kind`.
    ///
    /// [`PrmLabelKind::Soft`] returns the fraction; [`PrmLabelKind::Hard`] returns
    /// `1.0` or `0.0`.
    #[must_use]
    pub fn value(&self, kind: PrmLabelKind) -> f32 {
        match kind {
            PrmLabelKind::Soft => self.soft_label,
            PrmLabelKind::Hard => {
                if self.hard_label {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// ── StepScore ─────────────────────────────────────────────────────────────────

/// A scalar reward assigned to one step of a trajectory.
///
/// This is the output row of a [`ProcessRewardModel`]: the step's position and
/// its reward in `[0, 1]`. The lowest-scoring step of a trajectory is where the
/// `PRM` localizes the weakest link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepScore {
    /// The index (into the trajectory) of the scored step.
    pub step_index: usize,
    /// The step's reward, in `[0, 1]`.
    pub score: f32,
}

impl StepScore {
    /// Create a step score.
    #[must_use]
    pub fn new(step_index: usize, score: f32) -> Self {
        Self { step_index, score }
    }
}

// ── StepAggregation ───────────────────────────────────────────────────────────

/// How per-step scores collapse into one trajectory-level `PRM` score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepAggregation {
    /// The minimum step score — the weakest link governs the trajectory. This is
    /// the default and the aggregation that most sharply penalizes a single bad
    /// step.
    #[default]
    Min,
    /// The product of the step scores. Every step must be good for the trajectory
    /// to score well, and the penalty compounds.
    Product,
    /// The last step's score only — the reward of the final reasoning move.
    LastStep,
}

impl StepAggregation {
    /// Collapse `scores` into a single scalar.
    ///
    /// An empty slice yields `0.0` (a trajectory with no steps has no reward). The
    /// product folds left in index order so the result is bit-reproducible.
    #[must_use]
    pub fn aggregate(self, scores: &[StepScore]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        match self {
            Self::Min => scores.iter().fold(f32::INFINITY, |acc, s| acc.min(s.score)),
            Self::Product => scores.iter().fold(1.0, |acc, s| acc * s.score),
            Self::LastStep => scores.last().map_or(0.0, |s| s.score),
        }
    }
}

// ── Rollout ───────────────────────────────────────────────────────────────────

/// One sampled continuation of a reasoning prefix to a final answer.
///
/// [`reached_correct`](Self::reached_correct) is set by the *labeller* by
/// comparing [`answer`](Self::answer) to its gold answer under answer
/// equivalence — it is not the policy's own opinion, so correctness is judged
/// from a single, central definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rollout {
    /// The sampled steps that continue the prefix.
    pub continuation: Vec<ReasoningStep>,
    /// The final answer the continuation reaches.
    pub answer: String,
    /// Whether the answer matched the gold answer (set by the labeller).
    pub reached_correct: bool,
}

// ── RolloutResult ─────────────────────────────────────────────────────────────

/// The batch of rollouts drawn from one step prefix.
///
/// Retaining the individual [`Rollout`]s (not just their count) lets a caller
/// inspect *why* a step scored the way it did; [`label`](Self::label) reduces the
/// batch to the compact [`StepLabel`].
#[derive(Debug, Clone, PartialEq)]
pub struct RolloutResult {
    /// The index (into the trajectory) of the step these rollouts start from.
    pub step_index: usize,
    /// The individual rollouts.
    pub rollouts: Vec<Rollout>,
}

impl RolloutResult {
    /// How many rollouts reached the correct answer.
    #[must_use]
    pub fn num_correct(&self) -> usize {
        self.rollouts.iter().filter(|r| r.reached_correct).count()
    }

    /// How many rollouts were drawn.
    #[must_use]
    pub fn num_rollouts(&self) -> usize {
        self.rollouts.len()
    }

    /// Reduce the batch to its [`StepLabel`].
    ///
    /// The soft label is `num_correct / num_rollouts` (or `0.0` for an empty
    /// batch); the hard label is whether any rollout reached the answer.
    #[must_use]
    pub fn label(&self) -> StepLabel {
        let num_rollouts = self.num_rollouts();
        let num_correct = self.num_correct();
        let soft_label = if num_rollouts == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let value = num_correct as f32 / num_rollouts as f32;
            value
        };
        StepLabel {
            step_index: self.step_index,
            num_rollouts,
            num_correct,
            soft_label,
            hard_label: num_correct > 0,
        }
    }
}

// ── RankedTrajectory ──────────────────────────────────────────────────────────

/// A single candidate in a best-of-`N` ranking, with its aggregated reward and
/// the per-step scores that produced it.
///
/// [`index`](Self::index) is the candidate's original position in the input pool.
/// It doubles as the stable identity used to break exact score ties (lower index
/// wins), so a ranking is fully deterministic.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedTrajectory {
    /// The candidate's original position in the input pool.
    pub index: usize,
    /// The aggregated trajectory reward it was ranked by.
    pub aggregate_score: f32,
    /// The per-step scores (empty when ranked by an [`OutcomeRewardModel`], which
    /// does not score steps).
    pub step_scores: Vec<StepScore>,
    /// The candidate's final answer.
    pub answer: String,
}

impl RankedTrajectory {
    /// The index of the lowest-scoring step — where a [`ProcessRewardModel`]
    /// localizes the weakest link.
    ///
    /// Returns `None` when there are no step scores (e.g. an outcome-only
    /// ranking). Ties resolve to the earliest step, deterministically.
    #[must_use]
    pub fn weakest_step(&self) -> Option<usize> {
        self.step_scores
            .iter()
            .min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.step_index.cmp(&b.step_index))
            })
            .map(|s| s.step_index)
    }
}

// ── BestOfNResult ─────────────────────────────────────────────────────────────

/// The outcome of a best-of-`N` reranking: the candidates sorted best-first and
/// the original index of the winner.
#[derive(Debug, Clone, PartialEq)]
pub struct BestOfNResult {
    /// The candidates, sorted by descending aggregate score then ascending index.
    pub ranked: Vec<RankedTrajectory>,
    /// The original index of the top-ranked candidate.
    pub best_index: usize,
}

impl BestOfNResult {
    /// The winning candidate, or `None` if the pool was empty.
    #[must_use]
    pub fn best(&self) -> Option<&RankedTrajectory> {
        self.ranked.first()
    }
}

// ── PrmConfig ─────────────────────────────────────────────────────────────────

/// Configuration for [`MonteCarloProcessReward`](super::engine::MonteCarloProcessReward).
#[derive(Debug, Clone, PartialEq)]
pub struct PrmConfig {
    /// The number of rollouts `R` drawn per step. Larger `R` tightens the Monte
    /// Carlo estimate of each label. Defaults to `32`.
    pub num_rollouts: usize,
    /// The seed for the rollout generator. Defaults to `0`.
    pub seed: u64,
    /// Which label variant the step reward uses. Defaults to
    /// [`PrmLabelKind::Soft`].
    pub label_kind: PrmLabelKind,
    /// The default aggregation for a trajectory-level score. Defaults to
    /// [`StepAggregation::Min`].
    pub aggregation: StepAggregation,
    /// The answer-equivalence threshold in `[0, 1]` deciding whether a rollout
    /// reached the gold answer. Defaults to `0.6`.
    pub equivalence_threshold: f32,
}

impl Default for PrmConfig {
    fn default() -> Self {
        Self {
            num_rollouts: 32,
            seed: 0,
            label_kind: PrmLabelKind::Soft,
            aggregation: StepAggregation::Min,
            equivalence_threshold: 0.6,
        }
    }
}

impl PrmConfig {
    /// Set the number of rollouts per step.
    #[must_use]
    pub fn with_num_rollouts(mut self, num_rollouts: usize) -> Self {
        self.num_rollouts = num_rollouts;
        self
    }

    /// Set the rollout seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the label variant.
    #[must_use]
    pub fn with_label_kind(mut self, label_kind: PrmLabelKind) -> Self {
        self.label_kind = label_kind;
        self
    }

    /// Set the default aggregation.
    #[must_use]
    pub fn with_aggregation(mut self, aggregation: StepAggregation) -> Self {
        self.aggregation = aggregation;
        self
    }

    /// Set the answer-equivalence threshold.
    #[must_use]
    pub fn with_equivalence_threshold(mut self, equivalence_threshold: f32) -> Self {
        self.equivalence_threshold = equivalence_threshold;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PrmError::ZeroRollouts`] when `num_rollouts` is `0`, and
    /// [`PrmError::InvalidThreshold`] when `equivalence_threshold` is not in
    /// `[0, 1]`.
    pub fn validate(&self) -> Result<(), PrmError> {
        if self.num_rollouts == 0 {
            return Err(PrmError::ZeroRollouts);
        }
        if !(0.0..=1.0).contains(&self.equivalence_threshold) {
            return Err(PrmError::InvalidThreshold(self.equivalence_threshold));
        }
        Ok(())
    }
}

// ── PrmError ──────────────────────────────────────────────────────────────────

/// Errors from the `process_reward_model` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum PrmError {
    /// The reasoning trajectory had no steps.
    #[error("reasoning trajectory has no steps")]
    EmptyTrajectory,
    /// The gold answer was empty after trimming.
    #[error("gold answer must not be empty")]
    EmptyGoldAnswer,
    /// The configured number of rollouts was zero.
    #[error("number of rollouts must be positive")]
    ZeroRollouts,
    /// The candidate pool for a best-of-`N` reranking was empty.
    #[error("candidate pool for best-of-N reranking is empty")]
    EmptyCandidatePool,
    /// The answer-equivalence threshold was outside `[0, 1]`.
    #[error("equivalence threshold {0} is not in [0, 1]")]
    InvalidThreshold(f32),
}

// ── Traits ────────────────────────────────────────────────────────────────────

/// Samples continuations of a reasoning prefix to a final answer.
///
/// This is the stochastic environment the `Math-Shepherd` labeller queries: given
/// the steps fixed so far, sample one way the solution *could* finish. In a
/// production system this wraps a language model; the module ships a deterministic
/// lexical default,
/// [`LexicalRolloutPolicy`](super::engine::LexicalRolloutPolicy).
pub trait RolloutPolicy {
    /// Sample one continuation of `prefix` (the steps fixed so far) for `query`,
    /// drawing all randomness from `rng`.
    ///
    /// The returned [`Rollout::reached_correct`] is advisory: the labeller
    /// re-judges correctness against its own gold answer.
    fn rollout(&self, query: &str, prefix: &[ReasoningStep], rng: &mut PrmRng) -> Rollout;
}

/// Assigns a reward to *each* step of a reasoning trajectory.
///
/// This is the process reward model proper. Where an [`OutcomeRewardModel`]
/// returns one number for the whole answer, this returns one [`StepScore`] per
/// step, so credit (or blame) can be assigned to the exact reasoning move
/// responsible.
pub trait ProcessRewardModel {
    /// Assign a reward to each step of `steps` given `query`, one [`StepScore`]
    /// per input step, in order.
    fn score_steps(&self, query: &str, steps: &[ReasoningStep]) -> Vec<StepScore>;

    /// Collapse the per-step scores into one trajectory-level reward under
    /// `aggregation`.
    #[must_use]
    fn score_trajectory(
        &self,
        query: &str,
        steps: &[ReasoningStep],
        aggregation: StepAggregation,
    ) -> f32 {
        aggregation.aggregate(&self.score_steps(query, steps))
    }
}

/// Scores *only* the final answer, ignoring how it was reached.
///
/// This is the baseline the module exists to beat: two trajectories with the same
/// final answer are, to an outcome model, indistinguishable — even if one reached
/// it through a broken intermediate step.
pub trait OutcomeRewardModel {
    /// Score the final `answer` to `query`, in `[0, 1]`.
    fn score_answer(&self, query: &str, answer: &str) -> f32;
}
