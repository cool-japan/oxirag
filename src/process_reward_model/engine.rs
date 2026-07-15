//! The `process_reward_model` engine: `Math-Shepherd` automatic step labelling by
//! Monte Carlo rollout, the concrete process/outcome reward models, the shipped
//! lexical rollout policy, and best-of-`N` reranking.
//!
//! The core idea is credit assignment. An outcome reward model looks at the final
//! answer and stops. A process reward model asks a harder question at every step:
//! *from here, how likely is the solution to finish correctly?* `Math-Shepherd`
//! answers it without any human step annotations, by completing the solution many
//! times from each step and counting how often those completions land on the gold
//! answer (Wang et al., 2024). The fraction is the step's soft label; whether the
//! fraction is non-zero is its hard label.

use std::collections::HashSet;

use super::types::{
    BestOfNResult, OutcomeRewardModel, PrmConfig, PrmError, PrmRng, PrmTrajectory,
    ProcessRewardModel, RankedTrajectory, ReasoningStep, Rollout, RolloutPolicy, RolloutResult,
    StepAggregation, StepLabel, StepScore,
};

// ── Answer equivalence ────────────────────────────────────────────────────────

/// Tokenize `text` into a lowercase set of alphanumeric tokens.
fn token_set(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard similarity `|a ∩ b| / |a ∪ b|` between two token sets.
///
/// Returns `0.0` when both sets are empty (no shared meaning can be asserted).
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    #[allow(clippy::cast_precision_loss)]
    let score = intersection as f32 / union as f32;
    score
}

/// Directional containment `|a ∩ b| / |a|` — the fraction of `a` covered by `b`.
///
/// Returns `0.0` when `a` is empty.
fn containment(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    #[allow(clippy::cast_precision_loss)]
    let score = intersection as f32 / a.len() as f32;
    score
}

/// A graded answer similarity in `[0, 1]`, used for partial outcome credit.
///
/// This is the token-Jaccard of the two answers — `1.0` for an exact token match,
/// falling toward `0.0` as they diverge.
fn answer_similarity(a: &str, b: &str) -> f32 {
    jaccard(&token_set(a), &token_set(b))
}

/// Whether two answers are *equivalent* — Jaccard `>= threshold` and both
/// directional containments `>= threshold` (mutual containment is high).
///
/// This is the hand-rolled, embedder-free stand-in for "did this rollout reach
/// the gold answer", in the spirit of the clustering used elsewhere in the crate
/// but kept internal so the module carries no cross-feature dependency.
fn answers_equivalent(a: &str, b: &str, threshold: f32) -> bool {
    let set_a = token_set(a);
    let set_b = token_set(b);
    jaccard(&set_a, &set_b) >= threshold
        && containment(&set_a, &set_b) >= threshold
        && containment(&set_b, &set_a) >= threshold
}

// ── MonteCarloProcessReward ───────────────────────────────────────────────────

/// The `Math-Shepherd` process reward model: it labels every step of a trajectory
/// by Monte Carlo rollout against a known gold answer, then exposes those labels
/// as per-step rewards.
///
/// It owns a [`RolloutPolicy`] (the step completer) and a gold answer. For each
/// step `i` it fixes the prefix `steps[0..=i]`, draws
/// [`PrmConfig::num_rollouts`] completions, and counts how many reach the gold
/// answer under answer equivalence. The count becomes the step's [`StepLabel`],
/// and [`PrmConfig::label_kind`] selects whether the emitted [`StepScore`] is the
/// soft fraction or the hard indicator.
///
/// Every labelling run reseeds its generator from [`PrmConfig::seed`], so scoring
/// the same trajectory twice yields byte-identical labels.
#[derive(Debug, Clone)]
pub struct MonteCarloProcessReward<P: RolloutPolicy> {
    /// The rollout policy that completes a prefix to a final answer.
    policy: P,
    /// The gold answer a rollout must reach to count as correct.
    gold_answer: String,
    /// Rollout count, seed, label variant, and equivalence threshold.
    config: PrmConfig,
}

impl<P: RolloutPolicy> MonteCarloProcessReward<P> {
    /// Build a process reward model from a `policy`, a `gold_answer`, and a
    /// `config`.
    ///
    /// # Errors
    ///
    /// Returns [`PrmError::ZeroRollouts`] or [`PrmError::InvalidThreshold`] if the
    /// config is invalid (see [`PrmConfig::validate`]), and
    /// [`PrmError::EmptyGoldAnswer`] if `gold_answer` is blank.
    pub fn new(
        policy: P,
        gold_answer: impl Into<String>,
        config: PrmConfig,
    ) -> Result<Self, PrmError> {
        config.validate()?;
        let gold_answer = gold_answer.into();
        if gold_answer.trim().is_empty() {
            return Err(PrmError::EmptyGoldAnswer);
        }
        Ok(Self {
            policy,
            gold_answer,
            config,
        })
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &PrmConfig {
        &self.config
    }

    /// Borrow the gold answer.
    #[must_use]
    pub fn gold_answer(&self) -> &str {
        &self.gold_answer
    }

    /// Draw one rollout batch from `prefix`, judging each completion against the
    /// gold answer.
    fn rollout_batch(
        &self,
        query: &str,
        prefix: &[ReasoningStep],
        step_index: usize,
        rng: &mut PrmRng,
    ) -> RolloutResult {
        let mut rollouts = Vec::with_capacity(self.config.num_rollouts);
        for _ in 0..self.config.num_rollouts {
            let raw = self.policy.rollout(query, prefix, rng);
            let reached_correct = answers_equivalent(
                &raw.answer,
                &self.gold_answer,
                self.config.equivalence_threshold,
            );
            rollouts.push(Rollout {
                continuation: raw.continuation,
                answer: raw.answer,
                reached_correct,
            });
        }
        RolloutResult {
            step_index,
            rollouts,
        }
    }

    /// Label every step of `steps` by Monte Carlo rollout (`Math-Shepherd`).
    ///
    /// Returns one [`StepLabel`] per step, in order. The generator is seeded once
    /// from [`PrmConfig::seed`] and threaded through the steps in order, so the
    /// result is deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`PrmError::EmptyTrajectory`] when `steps` is empty.
    pub fn label_trajectory(
        &self,
        query: &str,
        steps: &[ReasoningStep],
    ) -> Result<Vec<StepLabel>, PrmError> {
        Ok(self
            .rollout_trajectory(query, steps)?
            .iter()
            .map(RolloutResult::label)
            .collect())
    }

    /// Draw the full rollout batches for every step of `steps`, retaining the
    /// individual rollouts.
    ///
    /// Same ordering and determinism as [`label_trajectory`](Self::label_trajectory);
    /// use this when the individual completions matter, not just their counts.
    ///
    /// # Errors
    ///
    /// Returns [`PrmError::EmptyTrajectory`] when `steps` is empty.
    pub fn rollout_trajectory(
        &self,
        query: &str,
        steps: &[ReasoningStep],
    ) -> Result<Vec<RolloutResult>, PrmError> {
        if steps.is_empty() {
            return Err(PrmError::EmptyTrajectory);
        }
        let mut rng = PrmRng::new(self.config.seed);
        let mut batches = Vec::with_capacity(steps.len());
        for i in 0..steps.len() {
            let Some(prefix) = steps.get(..=i) else {
                continue;
            };
            batches.push(self.rollout_batch(query, prefix, i, &mut rng));
        }
        Ok(batches)
    }
}

impl<P: RolloutPolicy> ProcessRewardModel for MonteCarloProcessReward<P> {
    /// Label the trajectory and emit each step's reward under
    /// [`PrmConfig::label_kind`]. An empty trajectory yields no scores.
    fn score_steps(&self, query: &str, steps: &[ReasoningStep]) -> Vec<StepScore> {
        match self.label_trajectory(query, steps) {
            Ok(labels) => labels
                .iter()
                .map(|label| StepScore {
                    step_index: label.step_index,
                    score: label.value(self.config.label_kind),
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

// ── MatchOutcomeReward ────────────────────────────────────────────────────────

/// The outcome-only baseline reward model: it scores the final answer against a
/// gold answer and never looks at the reasoning.
///
/// An exact (equivalent) answer scores `1.0`; otherwise the score is the graded
/// token-Jaccard similarity to the gold answer. Because it depends only on the
/// answer, two trajectories that reach the *same* answer receive the *same*
/// score — which is exactly the blind spot a [`ProcessRewardModel`] fills.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchOutcomeReward {
    /// The gold answer to compare against.
    gold_answer: String,
    /// The equivalence threshold for a full-credit match.
    equivalence_threshold: f32,
}

impl MatchOutcomeReward {
    /// Build an outcome reward model from a `gold_answer` and an equivalence
    /// `threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`PrmError::InvalidThreshold`] when `threshold` is not in `[0, 1]`,
    /// and [`PrmError::EmptyGoldAnswer`] when `gold_answer` is blank.
    pub fn new(gold_answer: impl Into<String>, threshold: f32) -> Result<Self, PrmError> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(PrmError::InvalidThreshold(threshold));
        }
        let gold_answer = gold_answer.into();
        if gold_answer.trim().is_empty() {
            return Err(PrmError::EmptyGoldAnswer);
        }
        Ok(Self {
            gold_answer,
            equivalence_threshold: threshold,
        })
    }

    /// Borrow the gold answer.
    #[must_use]
    pub fn gold_answer(&self) -> &str {
        &self.gold_answer
    }
}

impl OutcomeRewardModel for MatchOutcomeReward {
    fn score_answer(&self, _query: &str, answer: &str) -> f32 {
        if answers_equivalent(answer, &self.gold_answer, self.equivalence_threshold) {
            1.0
        } else {
            answer_similarity(answer, &self.gold_answer)
        }
    }
}

// ── LexicalRolloutPolicy ──────────────────────────────────────────────────────

/// The lower clamp on the lexical reach probability — no prefix is treated as
/// hopeless.
const LEXICAL_REACH_FLOOR: f64 = 0.05;
/// The upper clamp on the lexical reach probability — no prefix is treated as
/// certain.
const LEXICAL_REACH_CEIL: f64 = 0.95;
/// The reach probability assigned to an empty prefix, which has no coherence to
/// measure.
const LEXICAL_REACH_BASE: f64 = 0.5;

/// The module's shipped default rollout policy: a deterministic lexical
/// heuristic, in the spirit of the `Heuristic*` evaluators in `tree_of_thought`.
///
/// It calls no language model. Instead it models "how likely is this prefix to
/// finish correctly" by a *coherence* signal: the mean token-Jaccard of the most
/// recent step against the query and the earlier steps. A step that stays on
/// topic (shares vocabulary with the query and the reasoning so far) is treated
/// as more likely to complete correctly; an off-topic step is treated as less
/// likely. The signal drives a Bernoulli success draw, so a batch of rollouts
/// recovers the coherence as a soft label.
///
/// This is a *reach model based on the current step*: a later on-topic step can
/// recover the reach probability after an off-topic one, which is what lets the
/// process reward dip at a bad step and rise again afterward. Replace it with an
/// language-model-backed [`RolloutPolicy`] for real use.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalRolloutPolicy {
    /// The answer emitted when a rollout "succeeds".
    gold_answer: String,
    /// The answer emitted when a rollout "fails".
    wrong_answer: String,
}

impl LexicalRolloutPolicy {
    /// Build a policy that emits `gold_answer` on success and `wrong_answer` on
    /// failure.
    #[must_use]
    pub fn new(gold_answer: impl Into<String>, wrong_answer: impl Into<String>) -> Self {
        Self {
            gold_answer: gold_answer.into(),
            wrong_answer: wrong_answer.into(),
        }
    }

    /// The reach probability of a `prefix` under `query`: the mean token-Jaccard
    /// of the last step against the query and the earlier steps, clamped away from
    /// `0` and `1`.
    #[must_use]
    pub fn reach_probability(&self, query: &str, prefix: &[ReasoningStep]) -> f64 {
        let Some((last, earlier)) = prefix.split_last() else {
            return LEXICAL_REACH_BASE;
        };
        let last_set = token_set(&last.text);
        let mut total = f64::from(jaccard(&last_set, &token_set(query)));
        let mut count = 1_usize;
        for step in earlier {
            total += f64::from(jaccard(&last_set, &token_set(&step.text)));
            count += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = total / count as f64;
        mean.clamp(LEXICAL_REACH_FLOOR, LEXICAL_REACH_CEIL)
    }
}

impl RolloutPolicy for LexicalRolloutPolicy {
    fn rollout(&self, query: &str, prefix: &[ReasoningStep], rng: &mut PrmRng) -> Rollout {
        let reach = self.reach_probability(query, prefix);
        let success = rng.bernoulli(reach);
        let answer = if success {
            self.gold_answer.clone()
        } else {
            self.wrong_answer.clone()
        };
        let continuation = vec![ReasoningStep::new(format!("conclude: {answer}"))];
        Rollout {
            continuation,
            answer,
            reached_correct: success,
        }
    }
}

// ── BestOfN ───────────────────────────────────────────────────────────────────

/// A total order over ranked candidates: aggregate score descending, then
/// original index ascending. The index tie-break makes exact ties deterministic
/// (the earliest candidate wins) — a `max_by_key` here would silently return the
/// *last* maximal element instead.
fn ranked_order(a: &RankedTrajectory, b: &RankedTrajectory) -> std::cmp::Ordering {
    b.aggregate_score
        .partial_cmp(&a.aggregate_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.index.cmp(&b.index))
}

/// Best-of-`N` reranking: score `N` candidate trajectories and return them
/// ranked best-first.
///
/// The same pool can be ranked two ways — by a [`ProcessRewardModel`] (aggregating
/// per-step rewards) or by an [`OutcomeRewardModel`] (the final answer only) — so
/// the two can be compared directly. When a flawed intermediate step separates
/// two otherwise-equal answers, the process ranking prefers the clean trajectory
/// while the outcome ranking cannot tell them apart.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BestOfN {
    /// How per-step rewards collapse into the trajectory score used for ranking.
    aggregation: StepAggregation,
}

impl BestOfN {
    /// Build a reranker that aggregates per-step rewards with `aggregation`.
    #[must_use]
    pub fn new(aggregation: StepAggregation) -> Self {
        Self { aggregation }
    }

    /// The aggregation used for process ranking.
    #[must_use]
    pub fn aggregation(&self) -> StepAggregation {
        self.aggregation
    }

    /// Rank `trajectories` by aggregated *process* reward.
    ///
    /// Each candidate is scored per-step by `model` and reduced with this
    /// reranker's aggregation; ties on the aggregate resolve to the earlier
    /// candidate.
    ///
    /// # Errors
    ///
    /// Returns [`PrmError::EmptyCandidatePool`] when `trajectories` is empty.
    pub fn rank_process<M: ProcessRewardModel>(
        &self,
        query: &str,
        trajectories: &[PrmTrajectory],
        model: &M,
    ) -> Result<BestOfNResult, PrmError> {
        if trajectories.is_empty() {
            return Err(PrmError::EmptyCandidatePool);
        }
        let mut ranked: Vec<RankedTrajectory> = trajectories
            .iter()
            .enumerate()
            .map(|(index, trajectory)| {
                let step_scores = model.score_steps(query, &trajectory.steps);
                let aggregate_score = self.aggregation.aggregate(&step_scores);
                RankedTrajectory {
                    index,
                    aggregate_score,
                    step_scores,
                    answer: trajectory.answer.clone(),
                }
            })
            .collect();
        ranked.sort_by(ranked_order);
        let best_index = ranked.first().map_or(0, |r| r.index);
        Ok(BestOfNResult { ranked, best_index })
    }

    /// Rank `trajectories` by *outcome* reward — the final answer only.
    ///
    /// Step scores are left empty (an outcome model does not score steps); ties on
    /// the answer score resolve to the earlier candidate.
    ///
    /// # Errors
    ///
    /// Returns [`PrmError::EmptyCandidatePool`] when `trajectories` is empty.
    pub fn rank_outcome<O: OutcomeRewardModel>(
        &self,
        query: &str,
        trajectories: &[PrmTrajectory],
        model: &O,
    ) -> Result<BestOfNResult, PrmError> {
        if trajectories.is_empty() {
            return Err(PrmError::EmptyCandidatePool);
        }
        let mut ranked: Vec<RankedTrajectory> = trajectories
            .iter()
            .enumerate()
            .map(|(index, trajectory)| RankedTrajectory {
                index,
                aggregate_score: model.score_answer(query, &trajectory.answer),
                step_scores: Vec::new(),
                answer: trajectory.answer.clone(),
            })
            .collect();
        ranked.sort_by(ranked_order);
        let best_index = ranked.first().map_or(0, |r| r.index);
        Ok(BestOfNResult { ranked, best_index })
    }
}
