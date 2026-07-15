//! Tests for `process_reward_model`.
//!
//! The suite is measurement-driven, not self-consistent:
//!
//! - `soft_label_recovers_known_probability` — a rollout policy that reaches the
//!   answer with a *known* probability `p`; the estimated soft label must land
//!   within the `Hoeffding` tolerance of `p`, and that tolerance must tighten as
//!   the rollout count grows.
//! - `aggregation_matches_hand_computation` — min / product / last computed by
//!   hand and demanded back exactly.
//! - `prm_localizes_flawed_step_where_outcome_cannot` — the headline: two
//!   trajectories with the *same* final answer, one via a flawed step. The outcome
//!   model scores them equally; the process reward model ranks the clean one
//!   higher and points at the exact offending step.
//! - `best_of_n_selects_known_best_with_tie_break` — a pool with a known winner
//!   and an exact tie, resolved by the documented index tie-break.
//! - `labels_and_ranking_are_deterministic` — same seed, byte-identical results.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::doc_markdown
)]

use super::engine::{BestOfN, LexicalRolloutPolicy, MatchOutcomeReward, MonteCarloProcessReward};
use super::types::{
    OutcomeRewardModel, PrmConfig, PrmError, PrmLabelKind, PrmRng, PrmTrajectory,
    ProcessRewardModel, ReasoningStep, Rollout, RolloutPolicy, StepAggregation, StepScore,
};

// ── Test fixtures ─────────────────────────────────────────────────────────────

/// Extract the number following `tag` in `text` (e.g. `"p=0.70"` with tag `"p="`
/// yields `0.70`), falling back to `default` when it is absent or unparseable.
fn parse_tagged(text: &str, tag: &str, default: f64) -> f64 {
    text.split(tag)
        .nth(1)
        .map(|rest| {
            rest.chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect::<String>()
        })
        .and_then(|num| num.parse::<f64>().ok())
        .unwrap_or(default)
}

/// A rollout policy whose success probability is written *into the last step*:
/// a step tagged `p=0.70` reaches the gold answer with probability exactly `0.70`.
///
/// It draws a single uniform per rollout, so a batch of `R` rollouts is `R`
/// independent `Bernoulli(p)` trials — which is what makes the `Hoeffding`
/// convergence test exact.
#[derive(Debug, Clone)]
struct EncodedProbabilityPolicy {
    gold: String,
    wrong: String,
}

impl EncodedProbabilityPolicy {
    fn new(gold: impl Into<String>, wrong: impl Into<String>) -> Self {
        Self {
            gold: gold.into(),
            wrong: wrong.into(),
        }
    }
}

impl RolloutPolicy for EncodedProbabilityPolicy {
    fn rollout(&self, _query: &str, prefix: &[ReasoningStep], rng: &mut PrmRng) -> Rollout {
        let p = prefix
            .last()
            .map_or(0.5, |step| parse_tagged(&step.text, "p=", 0.5));
        let success = rng.next_f64() < p;
        let answer = if success {
            self.gold.clone()
        } else {
            self.wrong.clone()
        };
        Rollout {
            continuation: Vec::new(),
            answer,
            reached_correct: success,
        }
    }
}

/// A deterministic process reward model whose per-step scores are written *into
/// the step text*: a step tagged `s=0.30` scores exactly `0.30`. Used where exact
/// per-step control (and exact ties) is needed.
#[derive(Debug, Clone)]
struct FixedStepScorer;

impl ProcessRewardModel for FixedStepScorer {
    fn score_steps(&self, _query: &str, steps: &[ReasoningStep]) -> Vec<StepScore> {
        steps
            .iter()
            .enumerate()
            .map(|(index, step)| StepScore::new(index, parse_tagged(&step.text, "s=", 0.0) as f32))
            .collect()
    }
}

/// The `Hoeffding` two-sided tolerance `sqrt(ln(2/delta) / (2R))`.
fn hoeffding_bound(delta: f64, rollouts: usize) -> f64 {
    ((2.0 / delta).ln() / (2.0 * rollouts as f64)).sqrt()
}

/// Soft label of a single-step trajectory under a fresh seeded model.
fn single_step_soft(step_text: &str, gold: &str, rollouts: usize, seed: u64) -> f64 {
    let config = PrmConfig::default()
        .with_num_rollouts(rollouts)
        .with_seed(seed);
    let policy = EncodedProbabilityPolicy::new(gold, "0");
    let prm = MonteCarloProcessReward::new(policy, gold, config).unwrap();
    let steps = [ReasoningStep::new(step_text)];
    let labels = prm.label_trajectory("q", &steps).unwrap();
    f64::from(labels[0].soft_label)
}

// ── (a) Known generating process ──────────────────────────────────────────────

#[test]
fn soft_label_recovers_known_probability() {
    let gold = "42";
    let p = 0.70_f64;
    let step = "first p=0.70";
    let delta = 0.01;
    let seed = 0xC0FFEE;

    let small_r = 500;
    let large_r = 5000;

    let est_small = single_step_soft(step, gold, small_r, seed);
    let est_large = single_step_soft(step, gold, large_r, seed);

    let bound_small = hoeffding_bound(delta, small_r);
    let bound_large = hoeffding_bound(delta, large_r);

    // The estimate lands within the Monte Carlo tolerance of the true p...
    assert!(
        (est_small - p).abs() <= bound_small,
        "R={small_r}: est={est_small} p={p} bound={bound_small}"
    );
    assert!(
        (est_large - p).abs() <= bound_large,
        "R={large_r}: est={est_large} p={p} bound={bound_large}"
    );
    // ...and that tolerance genuinely tightens as R grows.
    assert!(
        bound_large < bound_small,
        "bound must shrink: {bound_large} !< {bound_small}"
    );
}

#[test]
fn hard_and_soft_labels_at_extremes() {
    let gold = "42";
    let rollouts = 256;

    // p = 1.0: every rollout reaches the answer.
    let all = single_step_full("sure p=1.0", gold, rollouts, 1);
    assert_eq!(all.num_rollouts, rollouts);
    assert_eq!(all.num_correct, rollouts);
    assert_eq!(all.soft_label, 1.0);
    assert!(all.hard_label);

    // p = 0.0: none do.
    let none = single_step_full("never p=0.0", gold, rollouts, 1);
    assert_eq!(none.num_correct, 0);
    assert_eq!(none.soft_label, 0.0);
    assert!(!none.hard_label);
}

/// Full label of a single-step trajectory (helper for the extremes test).
fn single_step_full(
    step_text: &str,
    gold: &str,
    rollouts: usize,
    seed: u64,
) -> super::types::StepLabel {
    let config = PrmConfig::default()
        .with_num_rollouts(rollouts)
        .with_seed(seed);
    let policy = EncodedProbabilityPolicy::new(gold, "0");
    let prm = MonteCarloProcessReward::new(policy, gold, config).unwrap();
    let steps = [ReasoningStep::new(step_text)];
    prm.label_trajectory("q", &steps).unwrap().remove(0)
}

// ── (b) Aggregation arithmetic ────────────────────────────────────────────────

#[test]
fn aggregation_matches_hand_computation() {
    let scores = [
        StepScore::new(0, 0.9),
        StepScore::new(1, 0.3),
        StepScore::new(2, 0.8),
    ];

    // min([0.9, 0.3, 0.8]) = 0.3
    assert_eq!(StepAggregation::Min.aggregate(&scores), 0.3_f32);

    // product = 0.9 * 0.3 * 0.8 = 0.216 (folded in index order, bit-reproducible)
    let expected_product = 1.0_f32 * 0.9 * 0.3 * 0.8;
    assert_eq!(
        StepAggregation::Product.aggregate(&scores),
        expected_product
    );
    assert!((StepAggregation::Product.aggregate(&scores) - 0.216).abs() < 1e-6);

    // last = 0.8
    assert_eq!(StepAggregation::LastStep.aggregate(&scores), 0.8_f32);

    // A single-step trajectory: all three collapse to that step.
    let one = [StepScore::new(0, 0.42)];
    assert_eq!(StepAggregation::Min.aggregate(&one), 0.42);
    assert_eq!(StepAggregation::Product.aggregate(&one), 0.42);
    assert_eq!(StepAggregation::LastStep.aggregate(&one), 0.42);

    // Empty is defined as 0.0.
    assert_eq!(StepAggregation::Min.aggregate(&[]), 0.0);
    assert_eq!(StepAggregation::Product.aggregate(&[]), 0.0);
    assert_eq!(StepAggregation::LastStep.aggregate(&[]), 0.0);
}

// ── (c) PRM beats outcome-only where it must ──────────────────────────────────

#[test]
fn prm_localizes_flawed_step_where_outcome_cannot() {
    let gold = "42";
    let query = "q";
    let rollouts = 4000;
    let seed = 0x5EED_1234;

    let config = PrmConfig::default()
        .with_num_rollouts(rollouts)
        .with_seed(seed)
        .with_aggregation(StepAggregation::Min);
    let prm = MonteCarloProcessReward::new(EncodedProbabilityPolicy::new(gold, "0"), gold, config)
        .unwrap();

    // Same final answer; the flawed trajectory dips at step 1 and recovers.
    let clean = PrmTrajectory::from_texts(["a p=0.90", "b p=0.90", "c p=0.90"], gold);
    let flawed = PrmTrajectory::from_texts(["a p=0.90", "b p=0.10", "c p=0.90"], gold);

    let clean_scores = prm.score_steps(query, &clean.steps);
    let flawed_scores = prm.score_steps(query, &flawed.steps);

    // Localization: the flaw shows up at step 1 and nowhere else.
    assert!(
        flawed_scores[1].score < 0.2,
        "flawed step 1: {flawed_scores:?}"
    );
    assert!(
        flawed_scores[0].score > 0.8,
        "flawed step 0: {flawed_scores:?}"
    );
    assert!(
        flawed_scores[2].score > 0.8,
        "flawed step 2: {flawed_scores:?}"
    );
    for score in &clean_scores {
        assert!(score.score > 0.8, "clean scores: {clean_scores:?}");
    }

    let clean_min = StepAggregation::Min.aggregate(&clean_scores);
    let flawed_min = StepAggregation::Min.aggregate(&flawed_scores);
    assert!(
        clean_min > flawed_min,
        "process reward must separate them: clean_min={clean_min} flawed_min={flawed_min}"
    );

    // Best-of-N by process reward ranks clean strictly above flawed, and the
    // flawed candidate's weakest step is exactly index 1.
    let pool = [clean.clone(), flawed.clone()];
    let by_process = BestOfN::new(StepAggregation::Min)
        .rank_process(query, &pool, &prm)
        .unwrap();
    assert_eq!(by_process.best_index, 0);
    assert!(by_process.ranked[0].aggregate_score > by_process.ranked[1].aggregate_score);
    let flawed_ranked = by_process.ranked.iter().find(|r| r.index == 1).unwrap();
    assert_eq!(flawed_ranked.weakest_step(), Some(1));

    // The outcome model is blind to the flaw: both reach "42", both score 1.0.
    let outcome = MatchOutcomeReward::new(gold, 0.6).unwrap();
    let by_outcome = BestOfN::new(StepAggregation::Min)
        .rank_outcome(query, &pool, &outcome)
        .unwrap();
    assert_eq!(
        by_outcome.ranked[0].aggregate_score,
        by_outcome.ranked[1].aggregate_score
    );
    assert_eq!(outcome.score_answer(query, &clean.answer), 1.0);
    assert_eq!(outcome.score_answer(query, &flawed.answer), 1.0);

    println!("clean step scores : {clean_scores:?}");
    println!("flawed step scores: {flawed_scores:?}");
    println!("clean_min={clean_min}  flawed_min={flawed_min}");
}

// ── (d) Best-of-N selects the known-best ──────────────────────────────────────

#[test]
fn best_of_n_selects_known_best_with_tie_break() {
    let query = "q";
    let scorer = FixedStepScorer;

    // Min-aggregates: t0=0.90, t1=0.30, t2=0.95, t3=0.95 (t3 ties t2 at a higher
    // index, so the index tie-break must keep t2 ahead of t3).
    let pool = [
        PrmTrajectory::from_texts(["s=0.90", "s=0.90"], "A"),
        PrmTrajectory::from_texts(["s=0.90", "s=0.30"], "B"),
        PrmTrajectory::from_texts(["s=0.95", "s=0.95"], "C"),
        PrmTrajectory::from_texts(["s=0.95", "s=0.97"], "D"),
    ];

    let result = BestOfN::new(StepAggregation::Min)
        .rank_process(query, &pool, &scorer)
        .unwrap();

    assert_eq!(result.best_index, 2);
    let order: Vec<usize> = result.ranked.iter().map(|r| r.index).collect();
    assert_eq!(order, vec![2, 3, 0, 1]);
    assert_eq!(result.ranked[0].aggregate_score, 0.95_f32);
    assert_eq!(result.ranked[1].aggregate_score, 0.95_f32);
    assert_eq!(result.ranked[2].aggregate_score, 0.90_f32);
    assert_eq!(result.ranked[3].aggregate_score, 0.30_f32);
}

// ── (e) Determinism ───────────────────────────────────────────────────────────

#[test]
fn labels_and_ranking_are_deterministic() {
    let gold = "42";
    let config = PrmConfig::default().with_num_rollouts(300).with_seed(999);
    let steps = PrmTrajectory::from_texts(["a p=0.80", "b p=0.40"], gold).steps;

    let prm1 = MonteCarloProcessReward::new(
        EncodedProbabilityPolicy::new(gold, "0"),
        gold,
        config.clone(),
    )
    .unwrap();
    let prm2 = MonteCarloProcessReward::new(EncodedProbabilityPolicy::new(gold, "0"), gold, config)
        .unwrap();

    // Same seed => identical labels, bit for bit.
    let labels1 = prm1.label_trajectory("q", &steps).unwrap();
    let labels2 = prm2.label_trajectory("q", &steps).unwrap();
    assert_eq!(labels1, labels2);

    // Scoring the same trajectory twice on one instance is reproducible.
    let scores_a = prm1.score_steps("q", &steps);
    let scores_b = prm1.score_steps("q", &steps);
    assert_eq!(scores_a, scores_b);

    // And so is the whole best-of-N ranking.
    let pool = [
        PrmTrajectory::from_texts(["a p=0.80", "b p=0.40"], gold),
        PrmTrajectory::from_texts(["a p=0.90", "b p=0.90"], gold),
    ];
    let rank1 = BestOfN::new(StepAggregation::Min)
        .rank_process("q", &pool, &prm1)
        .unwrap();
    let rank2 = BestOfN::new(StepAggregation::Min)
        .rank_process("q", &pool, &prm2)
        .unwrap();
    assert_eq!(rank1, rank2);
}

// ── Supporting behaviour ──────────────────────────────────────────────────────

#[test]
fn hard_label_kind_scores_are_binary() {
    let gold = "42";
    let config = PrmConfig::default()
        .with_num_rollouts(100)
        .with_seed(3)
        .with_label_kind(PrmLabelKind::Hard);
    let prm = MonteCarloProcessReward::new(EncodedProbabilityPolicy::new(gold, "0"), gold, config)
        .unwrap();

    let some = [ReasoningStep::new("x p=0.50")];
    assert_eq!(prm.score_steps("q", &some)[0].score, 1.0);

    let none = [ReasoningStep::new("x p=0.00")];
    assert_eq!(prm.score_steps("q", &none)[0].score, 0.0);
}

#[test]
fn rollout_trajectory_retains_individual_rollouts() {
    let gold = "42";
    let rollouts = 50;
    let config = PrmConfig::default()
        .with_num_rollouts(rollouts)
        .with_seed(11);
    let prm = MonteCarloProcessReward::new(EncodedProbabilityPolicy::new(gold, "0"), gold, config)
        .unwrap();
    let steps = [ReasoningStep::new("x p=0.60")];

    let batches = prm.rollout_trajectory("q", &steps).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rollouts(), rollouts);

    let label = batches[0].label();
    assert_eq!(label.num_rollouts, rollouts);
    let counted = batches[0]
        .rollouts
        .iter()
        .filter(|r| r.reached_correct)
        .count();
    assert_eq!(label.num_correct, counted);
    assert!((f64::from(label.soft_label) - counted as f64 / rollouts as f64).abs() < 1e-6);

    // The default score_trajectory aggregates the per-step scores.
    let agg = prm.score_trajectory("q", &steps, StepAggregation::LastStep);
    assert_eq!(agg, prm.score_steps("q", &steps).last().unwrap().score);
}

#[test]
fn lexical_policy_rewards_coherent_steps() {
    let gold = "answer forty two";
    let query = "forty two is the answer";
    let policy = LexicalRolloutPolicy::new(gold, "unrelated garbage");

    let coherent = [ReasoningStep::new("the answer is forty two")];
    let incoherent = [ReasoningStep::new("banana orange kiwi")];

    // The shipped policy's reach probability follows lexical coherence.
    let p_coherent = policy.reach_probability(query, &coherent);
    let p_incoherent = policy.reach_probability(query, &incoherent);
    assert!(
        p_coherent > p_incoherent,
        "coherent={p_coherent} incoherent={p_incoherent}"
    );

    // And a batch of rollouts recovers that ordering as a soft label.
    let config = PrmConfig::default().with_num_rollouts(2000).with_seed(4);
    let prm = MonteCarloProcessReward::new(policy, gold, config).unwrap();
    let coherent_soft = prm.label_trajectory(query, &coherent).unwrap()[0].soft_label;
    let incoherent_soft = prm.label_trajectory(query, &incoherent).unwrap()[0].soft_label;
    assert!(
        coherent_soft > incoherent_soft,
        "coherent_soft={coherent_soft} incoherent_soft={incoherent_soft}"
    );
}

#[test]
fn prm_rng_is_deterministic_and_ranged() {
    let mut a = PrmRng::new(123);
    let mut b = PrmRng::new(123);
    for _ in 0..64 {
        assert_eq!(a.next_u64(), b.next_u64());
    }

    let mut c = PrmRng::new(7);
    for _ in 0..1000 {
        let u = c.next_f64();
        assert!((0.0..1.0).contains(&u));
    }

    assert!(!PrmRng::new(1).bernoulli(0.0));
    assert!(PrmRng::new(1).bernoulli(1.0));
}

#[test]
fn error_paths_are_reported() {
    // Config validation.
    assert_eq!(
        PrmConfig::default().with_num_rollouts(0).validate(),
        Err(PrmError::ZeroRollouts)
    );
    assert_eq!(
        PrmConfig::default()
            .with_equivalence_threshold(1.5)
            .validate(),
        Err(PrmError::InvalidThreshold(1.5))
    );

    // Model construction.
    let bad_config = PrmConfig::default().with_num_rollouts(0);
    assert_eq!(
        MonteCarloProcessReward::new(LexicalRolloutPolicy::new("a", "b"), "a", bad_config)
            .unwrap_err(),
        PrmError::ZeroRollouts
    );
    assert_eq!(
        MonteCarloProcessReward::new(
            LexicalRolloutPolicy::new("a", "b"),
            "   ",
            PrmConfig::default()
        )
        .unwrap_err(),
        PrmError::EmptyGoldAnswer
    );

    // Empty trajectory: an error from the labeller, an empty score vector from
    // the trait method.
    let prm = MonteCarloProcessReward::new(
        LexicalRolloutPolicy::new("a", "b"),
        "a",
        PrmConfig::default(),
    )
    .unwrap();
    assert_eq!(
        prm.label_trajectory("q", &[]).unwrap_err(),
        PrmError::EmptyTrajectory
    );
    assert!(prm.score_steps("q", &[]).is_empty());

    // Outcome model construction.
    assert_eq!(
        MatchOutcomeReward::new("g", 2.0).unwrap_err(),
        PrmError::InvalidThreshold(2.0)
    );
    assert_eq!(
        MatchOutcomeReward::new("", 0.5).unwrap_err(),
        PrmError::EmptyGoldAnswer
    );

    // Empty best-of-N pool.
    let empty: [PrmTrajectory; 0] = [];
    assert_eq!(
        BestOfN::new(StepAggregation::Min)
            .rank_process("q", &empty, &FixedStepScorer)
            .unwrap_err(),
        PrmError::EmptyCandidatePool
    );
}
