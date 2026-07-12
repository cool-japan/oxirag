#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::manual_midpoint,
    clippy::doc_markdown
)]
//! Tests for contextual-bandit online ranking.
//!
//! The sections mirror the module: the PRNG, the dense linear algebra (including
//! the **headline** Sherman–Morrison-vs-Gauss–Jordan correctness check), the data
//! types, the shared per-arm model, then the three policies, then the evaluation
//! machinery.
//!
//! Four tests carry the module's actual claims, and they are worth calling out
//! because none of them is satisfiable by an algorithm that merely *compiles*:
//!
//! * `sherman_morrison_matches_direct_inverse_after_long_update_sequence` — the
//!   incrementally maintained `A^-1` is checked, after hundreds of rank-1 updates,
//!   against an independently-implemented Gauss–Jordan inverse of the design matrix
//!   `A` accumulated alongside it. Two structurally different computations of the
//!   same object must agree to ~1e-9. A drifting or subtly-wrong update is caught
//!   here and nowhere else.
//! * `linucb_recovers_ground_truth_parameters` — with rewards generated from a known
//!   `theta*`, the ridge estimate must converge to it. This is the check that the
//!   *estimation* is right, independent of whether the *exploration* is.
//! * `linucb_regret_is_sublinear` — average regret must fall across successive time
//!   windows and `Regret(2T)` must be less than `2 * Regret(T)`. Linear regret means
//!   the bandit is not learning, and it is invisible in every other statistic.
//! * `exploration_beats_greedy_on_deceptive_arm` — on a problem engineered so that
//!   pure greedy locks onto a mediocre arm on round one and never looks back,
//!   LinUCB and Thompson sampling must beat both it *and* uniform random. This is
//!   what demonstrates the exploration term is doing real work rather than being
//!   decoration.
//!
//! The assertions themselves are split by concern into the submodules below;
//! the shared simulation harness (a ground-truth linear environment, the two
//! context generators, and the simulation/baseline runners) lives here so every
//! submodule measures the same policies against the same worlds.

mod cholesky;
mod edge_cases;
mod evaluation;
mod linalg;
mod model;
mod policies;
mod rng;
mod sherman_morrison;

use crate::bandit_ranker::evaluation::{BanditLoggedEvent, BanditRegretTracker};
use crate::bandit_ranker::rng::SplitMix64Rng;
use crate::bandit_ranker::types::{BanditContext, BanditRanker, BanditResult};

// ═════════════════════════════════════════════════════════════════════════════
// Shared simulation harness
// ═════════════════════════════════════════════════════════════════════════════

/// A ground-truth disjoint-linear environment: every arm has a true `theta*`, and
/// the reward of pulling arm `a` in context `x` is `theta*_a^T x + noise`.
///
/// The *expected* reward `theta*_a^T x` is what regret is measured against; the
/// noisy realization is what the policy actually observes. Keeping the two
/// separate is the whole reason a simulated environment can measure regret at all.
struct LinearEnvironment {
    thetas: Vec<(String, Vec<f64>)>,
    noise_std: f64,
}

impl LinearEnvironment {
    fn new(thetas: Vec<(&str, Vec<f64>)>, noise_std: f64) -> Self {
        Self {
            thetas: thetas
                .into_iter()
                .map(|(id, theta)| (id.to_string(), theta))
                .collect(),
            noise_std,
        }
    }

    fn arm_ids(&self) -> Vec<String> {
        self.thetas.iter().map(|(id, _)| id.clone()).collect()
    }

    /// `theta*_a^T x` — the arm's true mean reward on this context.
    fn expected_reward(&self, arm_id: &str, x: &[f64]) -> f64 {
        let theta = &self
            .thetas
            .iter()
            .find(|(id, _)| id == arm_id)
            .expect("environment must know the arm being queried")
            .1;
        theta.iter().zip(x).map(|(t, xi)| t * xi).sum()
    }

    /// The oracle's payoff: the best mean reward available on this context.
    fn best_expected_reward(&self, x: &[f64]) -> f64 {
        self.thetas
            .iter()
            .map(|(id, _)| self.expected_reward(id, x))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// A noisy realization of the reward — what the policy actually sees.
    fn sample_reward(&self, arm_id: &str, x: &[f64], rng: &mut SplitMix64Rng) -> f64 {
        self.expected_reward(arm_id, x) + self.noise_std * rng.next_standard_normal()
    }
}

/// A context with every feature drawn from `N(0, 1)`.
///
/// The natural choice for the estimation tests: the resulting design matrix
/// concentrates around `T * I`, which is well conditioned, so any failure to
/// recover `theta*` is the algorithm's fault and not the geometry's.
fn gaussian_context(rng: &mut SplitMix64Rng, dim: usize) -> BanditContext {
    let features: Vec<f64> = (0..dim).map(|_| rng.next_standard_normal()).collect();
    BanditContext::new(features).expect("standard normals are finite")
}

/// A context with every feature drawn from `U(0, 1)`.
///
/// Non-negative by construction, which is what makes the deceptive-arm trap work:
/// an arm with an all-positive `theta*` pays a strictly positive reward on *every*
/// context, so a greedy policy that tries it once will never see a reason to try
/// anything else (every rival's untrained estimate is exactly `0`).
fn uniform_context(rng: &mut SplitMix64Rng, dim: usize) -> BanditContext {
    let features: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();
    BanditContext::new(features).expect("uniforms are finite")
}

/// Run `policy` against `environment` for `rounds` rounds, tracking regret.
///
/// Returns the regret tracker; the policy is left in its post-run state so a
/// caller can inspect what it learned.
fn run_simulation(
    policy: &mut dyn BanditRanker,
    environment: &LinearEnvironment,
    rounds: usize,
    dim: usize,
    context_seed: u64,
    noise_seed: u64,
    gaussian: bool,
) -> BanditResult<BanditRegretTracker> {
    let mut context_rng = SplitMix64Rng::new(context_seed);
    let mut noise_rng = SplitMix64Rng::new(noise_seed);
    let mut tracker = BanditRegretTracker::new();

    for _ in 0..rounds {
        let context = if gaussian {
            gaussian_context(&mut context_rng, dim)
        } else {
            uniform_context(&mut context_rng, dim)
        };
        let ranking = policy.select(&context)?;
        let chosen = ranking
            .top_arm_id()
            .expect("select never returns an empty ranking")
            .to_string();

        let x = context.features();
        tracker.record(
            environment.best_expected_reward(x),
            environment.expected_reward(&chosen, x),
        );

        let reward = environment.sample_reward(&chosen, x, &mut noise_rng);
        policy.update(&chosen, &context, reward)?;
    }
    Ok(tracker)
}

/// The mean *expected* (noise-free) reward the policy earned per round.
fn mean_policy_value(tracker: &BanditRegretTracker) -> f64 {
    tracker.policy_reward() / tracker.rounds() as f64
}

/// A "uniformly random arm every round" baseline, measured on the same
/// environment with the same context stream.
fn run_uniform_baseline(
    environment: &LinearEnvironment,
    rounds: usize,
    dim: usize,
    context_seed: u64,
    choice_seed: u64,
    gaussian: bool,
) -> BanditRegretTracker {
    let mut context_rng = SplitMix64Rng::new(context_seed);
    let mut choice_rng = SplitMix64Rng::new(choice_seed);
    let mut tracker = BanditRegretTracker::new();
    let arm_ids = environment.arm_ids();

    for _ in 0..rounds {
        let context = if gaussian {
            gaussian_context(&mut context_rng, dim)
        } else {
            uniform_context(&mut context_rng, dim)
        };
        let index = choice_rng
            .next_usize_below(arm_ids.len())
            .expect("environment has at least one arm");
        let x = context.features();
        tracker.record(
            environment.best_expected_reward(x),
            environment.expected_reward(&arm_ids[index], x),
        );
    }
    tracker
}

/// The largest absolute difference between two equal-length slices.
fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len(), "slices must be the same length");
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Generate a log of uniformly-random actions — the precondition that makes
/// replay evaluation unbiased.
fn generate_uniform_log(
    environment: &LinearEnvironment,
    events: usize,
    dim: usize,
    context_seed: u64,
    action_seed: u64,
    noise_seed: u64,
) -> Vec<BanditLoggedEvent> {
    let mut context_rng = SplitMix64Rng::new(context_seed);
    let mut action_rng = SplitMix64Rng::new(action_seed);
    let mut noise_rng = SplitMix64Rng::new(noise_seed);
    let arm_ids = environment.arm_ids();

    (0..events)
        .map(|_| {
            let context = uniform_context(&mut context_rng, dim);
            let index = action_rng
                .next_usize_below(arm_ids.len())
                .expect("at least one arm");
            let arm_id = arm_ids[index].clone();
            let reward = environment.sample_reward(&arm_id, context.features(), &mut noise_rng);
            BanditLoggedEvent::new(context, arm_id, reward)
        })
        .collect()
}
