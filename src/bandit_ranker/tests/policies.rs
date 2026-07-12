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
//! Tests for the three policies: parameter recovery, sublinear regret, the
//! deceptive-arm trap that separates exploration from greed, the analytic
//! posterior covariance, and the ε schedules.

use super::{
    LinearEnvironment, gaussian_context, max_abs_diff, mean_policy_value, run_simulation,
    run_uniform_baseline,
};
use crate::bandit_ranker::epsilon_greedy::EpsilonGreedyRanker;
use crate::bandit_ranker::linalg::scale_matrix;
use crate::bandit_ranker::linucb::LinUcbRanker;
use crate::bandit_ranker::rng::SplitMix64Rng;
use crate::bandit_ranker::thompson::ThompsonSamplingRanker;
use crate::bandit_ranker::types::{
    BanditArm, BanditConfig, BanditContext, BanditError, BanditRanker,
};

// ═════════════════════════════════════════════════════════════════════════════
// LinUCB
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn linucb_recovers_ground_truth_parameters() {
    // Rewards are generated from a known theta*. The ridge estimate must converge
    // to it: this is the check that the *estimation* is correct, entirely separately
    // from whether the *exploration* is.
    let truth = vec![0.70, -0.30, 0.50, 0.20];
    let dim = truth.len();
    let environment = LinearEnvironment::new(vec![("only", truth.clone())], 0.05);

    let config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_alpha(1.0);
    let mut ranker = LinUcbRanker::from_ids(config, ["only"]).expect("one arm");

    let mut context_rng = SplitMix64Rng::new(11);
    let mut noise_rng = SplitMix64Rng::new(22);

    let mut error_at_200 = f64::NAN;
    for round in 1..=5_000 {
        let context = gaussian_context(&mut context_rng, dim);
        let ranking = ranker.select(&context).expect("one arm exists");
        let chosen = ranking.top_arm_id().expect("non-empty").to_string();
        assert_eq!(chosen, "only");

        let reward = environment.sample_reward(&chosen, context.features(), &mut noise_rng);
        ranker.update(&chosen, &context, reward).expect("known arm");

        if round == 200 {
            error_at_200 = max_abs_diff(ranker.theta("only").expect("known arm"), &truth);
        }
    }

    let error_final = max_abs_diff(ranker.theta("only").expect("known arm"), &truth);
    assert!(
        error_final < 0.01,
        "theta_hat must converge to theta*; max |theta_hat - theta*| was {error_final}"
    );
    assert!(
        error_final < error_at_200,
        "the estimate must *improve* with data: {error_at_200} at t=200 vs {error_final} at t=5000"
    );
}

#[test]
fn linucb_recovers_parameters_in_one_dimension() {
    // d = 1 is the degenerate case every matrix routine is most likely to fumble.
    let environment = LinearEnvironment::new(vec![("only", vec![0.8])], 0.05);
    let config = BanditConfig::with_dimension(1).expect("valid");
    let mut ranker = LinUcbRanker::from_ids(config, ["only"]).expect("one arm");

    let mut context_rng = SplitMix64Rng::new(3);
    let mut noise_rng = SplitMix64Rng::new(4);
    for _ in 0..3_000 {
        let context = gaussian_context(&mut context_rng, 1);
        ranker.select(&context).expect("one arm exists");
        let reward = environment.sample_reward("only", context.features(), &mut noise_rng);
        ranker.update("only", &context, reward).expect("known arm");
    }

    let theta = ranker.theta("only").expect("known arm");
    assert_eq!(theta.len(), 1);
    assert!(
        (theta[0] - 0.8).abs() < 0.01,
        "d = 1 must converge just like any other dimension, got {}",
        theta[0]
    );
}

#[test]
fn linucb_regret_is_sublinear() {
    // THE defining property. A bandit with linear regret is not learning -- and it
    // is indistinguishable from a working one by every other statistic it reports.
    let environment = LinearEnvironment::new(
        vec![
            ("a", vec![0.50, -0.20, 0.10, 0.00, 0.30]),
            ("b", vec![-0.30, 0.60, 0.00, 0.20, -0.10]),
            ("c", vec![0.10, 0.10, 0.70, -0.40, 0.00]),
            ("d", vec![0.00, -0.50, 0.20, 0.50, 0.40]),
        ],
        0.10,
    );
    let dim = 5;
    let rounds = 8_000;

    let config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_alpha(0.5);
    let mut ranker = LinUcbRanker::from_ids(config, environment.arm_ids()).expect("distinct ids");
    let tracker = run_simulation(&mut ranker, &environment, rounds, dim, 101, 202, true)
        .expect("simulation runs cleanly");

    // 1. Average regret must fall, window over window, and approach zero.
    let windows = tracker.window_average_regret(4);
    assert_eq!(windows.len(), 4);
    for pair in windows.windows(2) {
        assert!(
            pair[1] < pair[0],
            "average regret must strictly decrease across successive windows, got {windows:?}"
        );
    }
    assert!(
        windows[3] < windows[0] * 0.1,
        "the final window's average regret must be an order of magnitude below the \
         first window's, got {windows:?}"
    );
    assert!(
        windows[3] < 0.02,
        "average regret must approach ~0, final window was {}",
        windows[3]
    );

    // 2. Regret(2T) < 2 * Regret(T): the curve is concave, i.e. genuinely sublinear.
    //    A linear-regret policy would satisfy Regret(2T) ~ 2 * Regret(T) exactly.
    let half = rounds / 2;
    let regret_t = tracker.cumulative_regret_at(half);
    let regret_2t = tracker.cumulative_regret_at(rounds);
    assert!(
        regret_2t < 2.0 * regret_t,
        "Regret(2T) = {regret_2t} must be strictly less than 2 * Regret(T) = {}",
        2.0 * regret_t
    );
    // Meaningfully so, not just by a rounding error.
    assert!(
        regret_2t < 1.35 * regret_t,
        "sublinearity must be substantial: Regret(2T) = {regret_2t}, Regret(T) = {regret_t}"
    );

    // 3. And the cumulative curve is monotone non-decreasing by construction.
    let curve = tracker.cumulative_curve();
    assert_eq!(curve.len(), rounds);
    for pair in curve.windows(2) {
        assert!(pair[1] >= pair[0], "cumulative regret can never decrease");
    }
}

#[test]
fn thompson_regret_is_sublinear() {
    let environment = LinearEnvironment::new(
        vec![
            ("a", vec![0.50, -0.20, 0.10, 0.00, 0.30]),
            ("b", vec![-0.30, 0.60, 0.00, 0.20, -0.10]),
            ("c", vec![0.10, 0.10, 0.70, -0.40, 0.00]),
            ("d", vec![0.00, -0.50, 0.20, 0.50, 0.40]),
        ],
        0.10,
    );
    let dim = 5;
    let rounds = 8_000;

    let config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_exploration_variance(0.15)
        .set_seed(77);
    let mut ranker =
        ThompsonSamplingRanker::from_ids(config, environment.arm_ids()).expect("distinct ids");
    let tracker = run_simulation(&mut ranker, &environment, rounds, dim, 101, 202, true)
        .expect("simulation runs cleanly");

    let windows = tracker.window_average_regret(4);
    for pair in windows.windows(2) {
        assert!(
            pair[1] < pair[0],
            "Thompson's average regret must strictly decrease across windows, got {windows:?}"
        );
    }
    let half = rounds / 2;
    assert!(
        tracker.cumulative_regret_at(rounds) < 2.0 * tracker.cumulative_regret_at(half),
        "Thompson sampling must also have sublinear regret"
    );

    // A healthy run should never have needed the Cholesky repair path.
    assert_eq!(
        ranker.jitter_events(),
        0,
        "the maintained inverses stayed comfortably positive definite"
    );
}

#[test]
fn exploration_beats_greedy_on_deceptive_arm() {
    // The trap. Contexts are non-negative, so the "deceptive" arm -- registered
    // *first*, with an all-positive theta* -- pays a strictly positive reward on
    // every single context.
    //
    // A pure-greedy policy starts with every theta_hat = 0, so every arm scores
    // exactly 0 and the tie breaks to registration order: it pulls "deceptive".
    // That pull returns a positive reward, so "deceptive" now scores *above* zero
    // while every rival is still stuck at exactly zero. Greedy therefore pulls
    // "deceptive" again. And again. Forever. It never discovers that "optimal" pays
    // three times as much, because it never looks -- and it has no mechanism that
    // could ever make it look.
    //
    // This is not a contrived pathology; it is the single most common way a
    // naively-deployed online ranker fails, and it is *invisible* from the outside:
    // greedy's model fits its own data beautifully, its mean reward is respectable,
    // and its rankings look entirely sensible.
    let environment = LinearEnvironment::new(
        vec![
            ("deceptive", vec![0.30, 0.30, 0.30, 0.30]),
            ("optimal", vec![0.90, 0.90, 0.90, 0.90]),
            ("poor", vec![0.05, 0.05, 0.05, 0.05]),
        ],
        0.05,
    );
    let dim = 4;
    let rounds = 3_000;

    // Pure greedy: epsilon = 0, no exploration of any kind.
    let greedy_config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_epsilon(0.0)
        .set_decay_epsilon(false);
    let mut greedy =
        EpsilonGreedyRanker::from_ids(greedy_config, environment.arm_ids()).expect("distinct ids");
    let greedy_tracker = run_simulation(&mut greedy, &environment, rounds, dim, 5, 6, false)
        .expect("simulation runs cleanly");

    // It locked on, exactly as predicted.
    let greedy_stats = greedy.stats();
    assert_eq!(
        greedy.exploration_rounds(),
        0,
        "epsilon = 0 must never explore"
    );
    assert_eq!(
        greedy_stats.most_pulled_arm(),
        Some("deceptive"),
        "greedy must have locked onto the deceptive arm"
    );
    assert!(
        greedy_stats.concentration() > 0.999,
        "greedy must have pulled essentially nothing else, got {}",
        greedy_stats.concentration()
    );
    assert_eq!(
        greedy_stats.arms[1].pulls, 0,
        "greedy never even *tried* the optimal arm"
    );

    // Uniform random: the do-nothing baseline.
    let uniform_tracker = run_uniform_baseline(&environment, rounds, dim, 5, 9, false);

    // LinUCB.
    let linucb_config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_alpha(0.8);
    let mut linucb =
        LinUcbRanker::from_ids(linucb_config, environment.arm_ids()).expect("distinct ids");
    let linucb_tracker = run_simulation(&mut linucb, &environment, rounds, dim, 5, 6, false)
        .expect("simulation runs cleanly");

    // Thompson sampling.
    let thompson_config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_exploration_variance(0.30)
        .set_seed(1234);
    let mut thompson = ThompsonSamplingRanker::from_ids(thompson_config, environment.arm_ids())
        .expect("distinct ids");
    let thompson_tracker = run_simulation(&mut thompson, &environment, rounds, dim, 5, 6, false)
        .expect("simulation runs cleanly");

    let greedy_value = mean_policy_value(&greedy_tracker);
    let uniform_value = mean_policy_value(&uniform_tracker);
    let linucb_value = mean_policy_value(&linucb_tracker);
    let thompson_value = mean_policy_value(&thompson_tracker);

    assert!(
        linucb_value > greedy_value,
        "LinUCB ({linucb_value}) must escape the trap that pure greedy ({greedy_value}) fell into"
    );
    assert!(
        linucb_value > uniform_value,
        "LinUCB ({linucb_value}) must beat uniform random ({uniform_value})"
    );
    assert!(
        thompson_value > greedy_value,
        "Thompson ({thompson_value}) must escape the trap that pure greedy ({greedy_value}) fell into"
    );
    assert!(
        thompson_value > uniform_value,
        "Thompson ({thompson_value}) must beat uniform random ({uniform_value})"
    );

    // And they found the *actually* optimal arm, rather than merely not-the-worst.
    assert_eq!(linucb.stats().most_pulled_arm(), Some("optimal"));
    assert_eq!(thompson.stats().most_pulled_arm(), Some("optimal"));

    // Greedy's regret, meanwhile, is *linear*: its average regret does not fall.
    let greedy_windows = greedy_tracker.window_average_regret(4);
    assert!(
        greedy_windows[3] > 0.5 * greedy_windows[0],
        "greedy's regret is linear -- its average regret plateaus rather than \
         decaying, got {greedy_windows:?}"
    );
    let linucb_windows = linucb_tracker.window_average_regret(4);
    assert!(
        linucb_windows[3] < 0.1 * linucb_windows[0],
        "LinUCB's regret is sublinear -- its average regret collapses, got {linucb_windows:?}"
    );
}

#[test]
fn linucb_ranks_every_arm_not_just_the_best() {
    let config = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_alpha(0.5);
    let mut ranker = LinUcbRanker::from_ids(config, ["a", "b", "c"]).expect("distinct ids");
    let context = BanditContext::new(vec![1.0, 1.0]).expect("finite");

    for _ in 0..30 {
        ranker.update("a", &context, 0.1).expect("known arm");
        ranker.update("b", &context, 0.9).expect("known arm");
        ranker.update("c", &context, 0.5).expect("known arm");
    }

    let ranking = ranker.select(&context).expect("arms exist");
    assert_eq!(ranking.len(), 3, "a *ranker* returns the whole ordering");
    assert!(!ranking.is_empty());
    assert!(
        !ranking.explored,
        "LinUCB has no separate exploration branch"
    );
    assert_eq!(ranking.arm_ids(), vec!["b", "c", "a"]);
    for (position, arm) in ranking.ranked.iter().enumerate() {
        assert_eq!(arm.rank, position);
    }
    // Scores must be non-increasing.
    for pair in ranking.ranked.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
    // And the score really is mean + alpha * width.
    let bonus = ranker.exploration_bonus("b", &context).expect("known arm");
    let top = ranking.top().expect("non-empty");
    assert!((top.score - top.mean_estimate - bonus).abs() < 1e-12);
    assert!(bonus > 0.0);
}

#[test]
fn linucb_exploration_bonus_decays_with_evidence() {
    let config = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_alpha(1.0);
    let mut ranker = LinUcbRanker::from_ids(config, ["a"]).expect("one arm");
    let context = BanditContext::new(vec![1.0, 0.0]).expect("finite");

    let before = ranker.exploration_bonus("a", &context).expect("known arm");
    for _ in 0..400 {
        ranker.update("a", &context, 1.0).expect("known arm");
    }
    let after = ranker.exploration_bonus("a", &context).expect("known arm");

    assert!(
        (before - 1.0).abs() < 1e-12,
        "the prior bonus is alpha * sqrt(1/lambda)"
    );
    // Exactly alpha / sqrt(1 + 400).
    assert!((after - (1.0_f64 / 401.0).sqrt()).abs() < 1e-12);
    assert!(after < before / 19.0, "the bonus must anneal on its own");

    assert!(matches!(
        ranker.exploration_bonus("nope", &context),
        Err(BanditError::UnknownArm { .. })
    ));
}

#[test]
fn linucb_with_alpha_zero_is_pure_greedy() {
    let config = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_alpha(0.0);
    let mut ranker = LinUcbRanker::from_ids(config, ["a", "b"]).expect("distinct ids");
    let context = BanditContext::new(vec![1.0, 1.0]).expect("finite");
    ranker.update("a", &context, 1.0).expect("known arm");

    let ranking = ranker.select(&context).expect("arms exist");
    let top = ranking.top().expect("non-empty");
    assert_eq!(
        top.score, top.mean_estimate,
        "alpha = 0 deletes the exploration term entirely"
    );
    assert_eq!(
        ranker.exploration_bonus("b", &context).expect("known arm"),
        0.0
    );
}

#[test]
fn linucb_can_add_an_arm_mid_flight_and_will_go_and_try_it() {
    let config = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_alpha(1.0);
    let mut ranker = LinUcbRanker::from_ids(config, ["incumbent"]).expect("one arm");
    let context = BanditContext::new(vec![1.0, 0.0]).expect("finite");

    for _ in 0..500 {
        ranker
            .update("incumbent", &context, 0.4)
            .expect("known arm");
    }
    // The incumbent's estimate is ~0.4 with an essentially-vanished bonus.
    let incumbent_score = ranker
        .select(&context)
        .expect("arms exist")
        .top()
        .expect("non-empty")
        .score;

    ranker
        .add_arm(BanditArm::new("newcomer"))
        .expect("fresh id");
    let ranking = ranker.select(&context).expect("arms exist");

    // The newcomer has theta_hat = 0 but the *maximal* confidence width, so
    // optimism sends LinUCB straight at it. This is exactly the behaviour you want
    // when a new retrieval strategy ships into a live ranker -- and note that
    // epsilon-greedy, whose newcomer would score a flat 0, would *not* do this.
    assert_eq!(
        ranking.top_arm_id(),
        Some("newcomer"),
        "an untried arm must be optimistically preferred (incumbent scored {incumbent_score})"
    );
    assert!(matches!(
        ranker.add_arm(BanditArm::new("newcomer")),
        Err(BanditError::DuplicateArm { .. })
    ));
}

// ═════════════════════════════════════════════════════════════════════════════
// Thompson sampling
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn thompson_posterior_draws_have_the_analytic_covariance() {
    // The sampler is checked *directly* against N(theta_hat, v^2 A^-1), not merely
    // inferred from downstream behaviour: a sampler with the wrong covariance still
    // produces a plausible-looking bandit, just a badly calibrated one.
    let dim = 3;
    let posterior_scale = 0.4;
    let config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_exploration_variance(posterior_scale)
        .set_seed(555);
    let mut ranker = ThompsonSamplingRanker::from_ids(config, ["a"]).expect("one arm");

    // Give the arm some asymmetric evidence so A^-1 is a non-trivial full matrix
    // rather than a multiple of the identity.
    let mut rng = SplitMix64Rng::new(31);
    for _ in 0..40 {
        let x: Vec<f64> = (0..dim).map(|_| rng.next_standard_normal()).collect();
        let context = BanditContext::new(x).expect("finite");
        ranker
            .update("a", &context, rng.next_standard_normal())
            .expect("known arm");
    }

    let theta_hat = ranker.theta("a").expect("known arm").to_vec();
    let a_inv = ranker
        .arms()
        .get("a")
        .expect("known arm")
        .design_inverse()
        .to_vec();

    let draws = 200_000;
    let mut mean = vec![0.0; dim];
    let mut covariance = vec![0.0; dim * dim];
    for _ in 0..draws {
        let sample = ranker.sample_theta("a").expect("known arm");
        let deviation: Vec<f64> = sample.iter().zip(&theta_hat).map(|(s, m)| s - m).collect();
        for i in 0..dim {
            mean[i] += sample[i];
            for j in 0..dim {
                covariance[i * dim + j] += deviation[i] * deviation[j];
            }
        }
    }
    let n = f64::from(draws);
    for value in &mut mean {
        *value /= n;
    }
    for value in &mut covariance {
        *value /= n;
    }

    // The posterior mean is theta_hat.
    assert!(
        max_abs_diff(&mean, &theta_hat) < 0.01,
        "the sample mean must be theta_hat; got {mean:?} vs {theta_hat:?}"
    );

    // And the posterior covariance is v^2 A^-1 -- exactly, up to Monte Carlo error.
    let expected = scale_matrix(&a_inv, posterior_scale * posterior_scale);
    let error = max_abs_diff(&covariance, &expected);
    assert!(
        error < 0.01,
        "the sample covariance must be v^2 A^-1; max entry error {error}"
    );

    // No jitter was needed on a healthy posterior.
    assert_eq!(ranker.jitter_events(), 0);
}

#[test]
fn thompson_exploration_is_proportional_to_uncertainty() {
    // The property that makes posterior sampling *work*: the spread of the sampled
    // score in direction x is v * sqrt(x^T A^-1 x) -- i.e. it is *exactly* LinUCB's
    // exploration bonus, scaled. Where the model is ignorant, the draws are wild;
    // where it is confident, they collapse onto the mean.
    let dim = 2;
    let config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_exploration_variance(0.5)
        .set_seed(909);
    let mut ranker = ThompsonSamplingRanker::from_ids(config, ["a"]).expect("one arm");

    let observed = BanditContext::new(vec![1.0, 0.0]).expect("finite");
    let unobserved = BanditContext::new(vec![0.0, 1.0]).expect("finite");

    for _ in 0..500 {
        ranker.update("a", &observed, 1.0).expect("known arm");
    }

    // Empirical spread of theta_tilde^T x in each direction.
    let mut spread = |x: &BanditContext| -> f64 {
        let draws = 40_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..draws {
            let theta = ranker.sample_theta("a").expect("known arm");
            let score: f64 = theta.iter().zip(x.features()).map(|(t, xi)| t * xi).sum();
            sum += score;
            sum_sq += score * score;
        }
        let n = f64::from(draws);
        let mean = sum / n;
        (sum_sq / n - mean * mean).sqrt()
    };

    let spread_observed = spread(&observed);
    let spread_unobserved = spread(&unobserved);

    // Analytically: 0.5 / sqrt(501) ~ 0.0223 vs 0.5 / sqrt(1) = 0.5.
    let expected_observed = 0.5 / 501.0_f64.sqrt();
    assert!(
        (spread_observed - expected_observed).abs() < 0.005,
        "spread in the observed direction should be v/sqrt(1 + n) = {expected_observed}, got {spread_observed}"
    );
    assert!(
        (spread_unobserved - 0.5).abs() < 0.02,
        "spread in the unobserved direction should still be the prior v = 0.5, got {spread_unobserved}"
    );
    assert!(
        spread_unobserved > spread_observed * 15.0,
        "exploration must remain concentrated where the model is still ignorant"
    );
}

#[test]
fn thompson_is_deterministic_given_its_seed() {
    // Two arms with *identical* theta*, so the choice on every round is decided
    // purely by posterior noise. Any difference in the RNG shows up immediately as
    // a different action sequence -- which is exactly what makes this a sharp test
    // of determinism rather than a test of "both converged to the same arm".
    let environment = LinearEnvironment::new(
        vec![("twin-a", vec![0.5, 0.5]), ("twin-b", vec![0.5, 0.5])],
        0.20,
    );

    let trajectory = |seed: u64| -> (Vec<String>, Vec<f64>) {
        let config = BanditConfig::with_dimension(2)
            .expect("valid")
            .set_exploration_variance(0.3)
            .set_seed(seed);
        let mut ranker =
            ThompsonSamplingRanker::from_ids(config, environment.arm_ids()).expect("distinct ids");
        let mut context_rng = SplitMix64Rng::new(1);
        let mut noise_rng = SplitMix64Rng::new(2);
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        for _ in 0..200 {
            let context = gaussian_context(&mut context_rng, 2);
            let ranking = ranker.select(&context).expect("arms exist");
            let chosen = ranking.top_arm_id().expect("non-empty").to_string();
            let reward = environment.sample_reward(&chosen, context.features(), &mut noise_rng);
            ranker.update(&chosen, &context, reward).expect("known arm");
            actions.push(chosen);
            rewards.push(reward);
        }
        (actions, rewards)
    };

    let (actions_a, rewards_a) = trajectory(42);
    let (actions_b, rewards_b) = trajectory(42);
    assert_eq!(
        actions_a, actions_b,
        "the same seed must replay the same actions"
    );
    assert_eq!(rewards_a, rewards_b, "...and, therefore, the same rewards");

    let (actions_c, _) = trajectory(43);
    assert_ne!(
        actions_a, actions_c,
        "a different seed must explore a different trajectory"
    );
}

#[test]
fn thompson_ranks_every_arm() {
    let config = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_exploration_variance(0.1)
        .set_seed(3);
    let mut ranker =
        ThompsonSamplingRanker::from_ids(config, ["a", "b", "c"]).expect("distinct ids");
    let context = BanditContext::new(vec![1.0, 0.0]).expect("finite");
    let ranking = ranker.select(&context).expect("arms exist");

    assert_eq!(ranking.len(), 3);
    assert!(!ranking.explored);
    for pair in ranking.ranked.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
    for (position, arm) in ranking.ranked.iter().enumerate() {
        assert_eq!(arm.rank, position);
        assert_eq!(arm.mean_estimate, 0.0, "untrained arms have theta_hat = 0");
        assert!(arm.score.is_finite());
    }
    assert_eq!(ranker.policy_name(), "thompson-sampling");
    assert_eq!(ranker.dimension(), 2);
    assert!(matches!(
        ranker.sample_theta("nope"),
        Err(BanditError::UnknownArm { .. })
    ));
}

// ═════════════════════════════════════════════════════════════════════════════
// Epsilon-greedy
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn epsilon_zero_never_explores_and_epsilon_one_always_does() {
    let context = BanditContext::new(vec![1.0, 1.0]).expect("finite");

    let never = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_epsilon(0.0)
        .set_decay_epsilon(false);
    let mut never_ranker = EpsilonGreedyRanker::from_ids(never, ["a", "b"]).expect("distinct ids");
    for _ in 0..1_000 {
        let ranking = never_ranker.select(&context).expect("arms exist");
        assert!(!ranking.explored);
    }
    assert_eq!(never_ranker.exploration_rounds(), 0);
    assert_eq!(never_ranker.rounds(), 1_000);
    assert_eq!(never_ranker.current_epsilon(), 0.0);

    let always = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_epsilon(1.0)
        .set_decay_epsilon(false);
    let mut always_ranker =
        EpsilonGreedyRanker::from_ids(always, ["a", "b"]).expect("distinct ids");
    for _ in 0..1_000 {
        let ranking = always_ranker.select(&context).expect("arms exist");
        assert!(ranking.explored);
    }
    assert_eq!(always_ranker.exploration_rounds(), 1_000);
    // Both extremes are *exact*, not approximate: next_f64 is uniform on the
    // half-open [0, 1), so no draw is < 0 and every draw is < 1.
}

#[test]
fn epsilon_explores_at_the_configured_rate() {
    let context = BanditContext::new(vec![1.0, 1.0]).expect("finite");
    let config = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_epsilon(0.25)
        .set_decay_epsilon(false)
        .set_seed(17);
    let mut ranker = EpsilonGreedyRanker::from_ids(config, ["a", "b", "c"]).expect("distinct ids");

    let rounds = 100_000;
    for _ in 0..rounds {
        ranker.select(&context).expect("arms exist");
    }
    let rate = ranker.exploration_rounds() as f64 / f64::from(rounds);
    assert!(
        (rate - 0.25).abs() < 0.01,
        "the exploration rate must match epsilon, got {rate}"
    );
}

#[test]
fn epsilon_decay_schedule_follows_one_over_sqrt_t() {
    let config = BanditConfig::with_dimension(1)
        .expect("valid")
        .set_epsilon(0.5)
        .set_decay_epsilon(true);
    let mut ranker = EpsilonGreedyRanker::from_ids(config, ["a"]).expect("one arm");
    let context = BanditContext::new(vec![1.0]).expect("finite");

    // Round 1 uses eps_0 exactly.
    assert!((ranker.current_epsilon() - 0.5).abs() < 1e-12);
    ranker.select(&context).expect("arms exist");
    // Round 2 uses eps_0 / sqrt(2).
    assert!((ranker.current_epsilon() - 0.5 / 2.0_f64.sqrt()).abs() < 1e-12);

    // 1 select already happened; 99 more brings the count to 100.
    for _ in 0..99 {
        ranker.select(&context).expect("arms exist");
    }
    assert_eq!(ranker.rounds(), 100);
    // So the *next* round is t = 101, and uses eps_0 / sqrt(101).
    assert!((ranker.current_epsilon() - 0.5 / 101.0_f64.sqrt()).abs() < 1e-12);
    assert!(
        ranker.current_epsilon() < 0.05,
        "the schedule must actually anneal"
    );
}

#[test]
fn epsilon_decay_gives_sublinear_regret_where_a_constant_epsilon_does_not() {
    // The textbook result made concrete: a *constant* epsilon pays its exploration
    // tax forever, so its regret is linear. The 1/sqrt(t) schedule anneals, so its
    // regret is sublinear. Same policy, same model, same environment, same seed --
    // the schedule is the *only* difference.
    let environment = LinearEnvironment::new(
        vec![
            ("good", vec![0.90, 0.90, 0.90]),
            ("mid", vec![0.40, 0.40, 0.40]),
            ("bad", vec![0.05, 0.05, 0.05]),
        ],
        0.05,
    );
    let dim = 3;
    let rounds = 8_000;

    let constant_config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_epsilon(0.30)
        .set_decay_epsilon(false)
        .set_seed(8);
    let mut constant = EpsilonGreedyRanker::from_ids(constant_config, environment.arm_ids())
        .expect("distinct ids");
    let constant_tracker = run_simulation(&mut constant, &environment, rounds, dim, 5, 6, false)
        .expect("simulation runs cleanly");

    let decaying_config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_epsilon(0.30)
        .set_decay_epsilon(true)
        .set_seed(8);
    let mut decaying = EpsilonGreedyRanker::from_ids(decaying_config, environment.arm_ids())
        .expect("distinct ids");
    let decaying_tracker = run_simulation(&mut decaying, &environment, rounds, dim, 5, 6, false)
        .expect("simulation runs cleanly");

    let constant_windows = constant_tracker.window_average_regret(4);
    let decaying_windows = decaying_tracker.window_average_regret(4);

    // The constant schedule plateaus: its last window is no better than ~its first.
    assert!(
        constant_windows[3] > 0.7 * constant_windows[0],
        "a constant epsilon must keep paying its tax: {constant_windows:?}"
    );
    // The decaying schedule collapses.
    assert!(
        decaying_windows[3] < 0.35 * decaying_windows[0],
        "a 1/sqrt(t) schedule must anneal its regret away: {decaying_windows:?}"
    );
    assert!(
        decaying_tracker.cumulative_regret() < constant_tracker.cumulative_regret(),
        "the decaying schedule must accumulate strictly less regret in total"
    );
}

#[test]
fn epsilon_greedy_is_deterministic_given_its_seed() {
    let environment =
        LinearEnvironment::new(vec![("a", vec![0.6, 0.2]), ("b", vec![0.2, 0.6])], 0.10);

    let trajectory = |seed: u64| -> Vec<String> {
        let config = BanditConfig::with_dimension(2)
            .expect("valid")
            .set_epsilon(0.4)
            .set_seed(seed);
        let mut ranker =
            EpsilonGreedyRanker::from_ids(config, environment.arm_ids()).expect("distinct ids");
        let mut context_rng = SplitMix64Rng::new(1);
        let mut noise_rng = SplitMix64Rng::new(2);
        let mut actions = Vec::new();
        for _ in 0..300 {
            let context = gaussian_context(&mut context_rng, 2);
            let ranking = ranker.select(&context).expect("arms exist");
            let chosen = ranking.top_arm_id().expect("non-empty").to_string();
            let reward = environment.sample_reward(&chosen, context.features(), &mut noise_rng);
            ranker.update(&chosen, &context, reward).expect("known arm");
            actions.push(chosen);
        }
        actions
    };

    assert_eq!(trajectory(100), trajectory(100));
    assert_ne!(trajectory(100), trajectory(101));
}

#[test]
fn epsilon_exploration_produces_a_permutation_not_a_truncation() {
    // The exploration branch shuffles; it must not drop, duplicate, or corrupt arms.
    let config = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_epsilon(1.0)
        .set_decay_epsilon(false)
        .set_seed(64);
    let mut ranker =
        EpsilonGreedyRanker::from_ids(config, ["a", "b", "c", "d"]).expect("distinct ids");
    let context = BanditContext::new(vec![1.0, 1.0]).expect("finite");

    let mut leader_counts = [0_u32; 4];
    let trials = 40_000;
    for _ in 0..trials {
        let ranking = ranker.select(&context).expect("arms exist");
        assert!(ranking.explored);
        assert_eq!(ranking.len(), 4);
        let mut ids = ranking.arm_ids();
        for (position, arm) in ranking.ranked.iter().enumerate() {
            assert_eq!(
                arm.rank, position,
                "ranks must be renumbered after the shuffle"
            );
        }
        let leader = match ids[0] {
            "a" => 0,
            "b" => 1,
            "c" => 2,
            "d" => 3,
            other => panic!("unexpected arm {other}"),
        };
        leader_counts[leader] += 1;
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec!["a", "b", "c", "d"],
            "every arm must still be present"
        );
    }
    // The top slot -- the arm that actually gets served -- is uniform over the arms,
    // which is precisely classical epsilon-greedy's exploration step.
    for count in leader_counts {
        let ratio = f64::from(count) / f64::from(trials);
        assert!(
            (ratio - 0.25).abs() < 0.015,
            "exploration must be uniform over arms, got {ratio}"
        );
    }
}

#[test]
fn epsilon_greedy_exploit_branch_ranks_by_mean_estimate() {
    let config = BanditConfig::with_dimension(2)
        .expect("valid")
        .set_epsilon(0.0)
        .set_decay_epsilon(false);
    let mut ranker = EpsilonGreedyRanker::from_ids(config, ["a", "b"]).expect("distinct ids");
    let context = BanditContext::new(vec![1.0, 0.0]).expect("finite");

    for _ in 0..20 {
        ranker.update("a", &context, 0.2).expect("known arm");
        ranker.update("b", &context, 0.9).expect("known arm");
    }
    let ranking = ranker.select(&context).expect("arms exist");
    let top = ranking.top().expect("non-empty");
    assert_eq!(top.arm_id, "b");
    assert_eq!(
        top.score, top.mean_estimate,
        "the greedy score *is* the mean estimate -- no bonus of any kind"
    );
    assert_eq!(ranker.policy_name(), "epsilon-greedy");
    assert_eq!(ranker.arm_ids(), vec!["a", "b"]);
    assert!(ranker.theta("a").is_some());
    assert!(ranker.theta("nope").is_none());
}
