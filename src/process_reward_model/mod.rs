//! Step-level (process) reward modelling with `Math-Shepherd` automatic labelling
//! and best-of-`N` reranking.
//!
//! A *process* reward model (`PRM`) scores every intermediate step of a chain of
//! reasoning, not just the answer at the end. This matters because a chain can
//! reach the right answer through a wrong step — a lucky cancellation, an
//! arithmetic error that a second error undoes — and an answer-only reward has no
//! way to see it. The `PRM` does: it assigns a reward to each step, so blame lands
//! on the exact move that went wrong.
//!
//! Distinct from `llm_judge`, `trust_score`, and `chainpoll`, which score a
//! **finished answer** as one unit: this module assigns a reward to **each
//! intermediate reasoning step**, so a chain that reaches a right answer by a
//! wrong step is penalised where it went wrong.
//!
//! # What it does
//!
//! 1. **`Math-Shepherd` automatic step labelling by Monte Carlo rollout.**
//!    [`MonteCarloProcessReward`] labels a step by *completing the solution from
//!    that step* many times and measuring how often the completions reach the gold
//!    answer. The fraction is the step's **soft** label; whether the fraction is
//!    non-zero is its **hard** label. No human step annotations are needed — the
//!    labels are recovered from rollouts alone (Wang et al., 2024).
//! 2. **Step-score aggregation.** [`StepAggregation`] collapses the per-step
//!    rewards of a trajectory into one number — the [`Min`](StepAggregation::Min)
//!    (weakest link), the [`Product`](StepAggregation::Product), or the
//!    [`LastStep`](StepAggregation::LastStep).
//! 3. **Best-of-`N` reranking.** [`BestOfN`] ranks `N` candidate trajectories by
//!    aggregated process reward, and can rank the same pool by an
//!    [`OutcomeRewardModel`] (final answer only) for comparison. Where a flawed
//!    step separates two trajectories that reach the *same* answer, the process
//!    ranking prefers the clean one and points at the offending step; the outcome
//!    ranking cannot tell them apart.
//!
//! # Determinism
//!
//! Rollout sampling draws from [`PrmRng`], a seeded `SplitMix64` generator. A
//! given [`PrmConfig::seed`] replayed against the same trajectory yields
//! byte-identical labels and rankings.
//!
//! # Quick start
//!
//! ```
//! # #[cfg(feature = "process-reward-model")]
//! # {
//! use oxirag::process_reward_model::{
//!     BestOfN, LexicalRolloutPolicy, MatchOutcomeReward, MonteCarloProcessReward, PrmConfig,
//!     PrmTrajectory, StepAggregation,
//! };
//!
//! // A rollout policy that reaches "42" more often when the reasoning stays on
//! // topic; the process reward model labels each step against the gold answer.
//! let query = "compute the value forty two";
//! let policy = LexicalRolloutPolicy::new("42", "0");
//! let config = PrmConfig::default().with_num_rollouts(200).with_seed(7);
//! let prm = MonteCarloProcessReward::new(policy, "42", config).expect("valid config");
//!
//! // Two trajectories, same final answer. They share the first step; the noisy
//! // one then goes off topic, which the process reward model penalizes.
//! let clean = PrmTrajectory::from_texts(["compute the value", "the value forty two"], "42");
//! let noisy = PrmTrajectory::from_texts(["compute the value", "xyz abc def ghi"], "42");
//! let pool = [clean, noisy];
//!
//! let by_process = BestOfN::new(StepAggregation::Min)
//!     .rank_process(query, &pool, &prm)
//!     .expect("non-empty pool");
//! assert_eq!(by_process.best_index, 0); // the clean trajectory wins
//!
//! let outcome = MatchOutcomeReward::new("42", 0.6).expect("valid gold");
//! let by_outcome = BestOfN::new(StepAggregation::Min)
//!     .rank_outcome(query, &pool, &outcome)
//!     .expect("non-empty pool");
//! // Both reach "42", so the outcome model scores them equally.
//! assert_eq!(by_outcome.ranked[0].aggregate_score, by_outcome.ranked[1].aggregate_score);
//! # }
//! ```

pub mod engine;
pub mod types;

#[cfg(test)]
mod tests;

pub use engine::{BestOfN, LexicalRolloutPolicy, MatchOutcomeReward, MonteCarloProcessReward};
pub use types::{
    BestOfNResult, OutcomeRewardModel, PrmConfig, PrmError, PrmLabelKind, PrmRng, PrmTrajectory,
    ProcessRewardModel, RankedTrajectory, ReasoningStep, Rollout, RolloutPolicy, RolloutResult,
    StepAggregation, StepLabel, StepScore,
};
