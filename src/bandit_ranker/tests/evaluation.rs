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
//! Tests for [`BanditRegretTracker`] and for the replay-based off-policy
//! evaluator.

use super::{
    LinearEnvironment, generate_uniform_log, max_abs_diff, mean_policy_value, run_simulation,
};
use crate::bandit_ranker::epsilon_greedy::EpsilonGreedyRanker;
use crate::bandit_ranker::evaluation::{
    BanditLoggedEvent, BanditOffPolicyEstimate, BanditOffPolicyEvaluator, BanditRegretTracker,
};
use crate::bandit_ranker::linucb::LinUcbRanker;
use crate::bandit_ranker::thompson::ThompsonSamplingRanker;
use crate::bandit_ranker::types::{BanditConfig, BanditContext, BanditError, BanditRanker};

// ═════════════════════════════════════════════════════════════════════════════
// Regret tracker
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn regret_tracker_accumulates_the_gap() {
    let mut tracker = BanditRegretTracker::new();
    assert!(tracker.is_empty());
    assert_eq!(tracker.rounds(), 0);
    assert_eq!(tracker.cumulative_regret(), 0.0);
    assert_eq!(tracker.average_regret(), 0.0);
    assert!(tracker.window_average_regret(3).is_empty());

    tracker.record(1.0, 0.4);
    tracker.record(2.0, 1.5);
    tracker.record(1.0, 1.0);

    assert_eq!(tracker.rounds(), 3);
    assert!((tracker.cumulative_regret() - (0.6 + 0.5 + 0.0)).abs() < 1e-12);
    assert!((tracker.average_regret() - 1.1 / 3.0).abs() < 1e-12);
    assert_eq!(tracker.instantaneous_regrets(), &[0.6, 0.5, 0.0]);
    assert!((tracker.policy_reward() - 2.9).abs() < 1e-12);
    assert!((tracker.oracle_reward() - 4.0).abs() < 1e-12);

    assert!((tracker.cumulative_regret_at(1) - 0.6).abs() < 1e-12);
    assert!((tracker.cumulative_regret_at(2) - 1.1).abs() < 1e-12);
    assert!(
        (tracker.cumulative_regret_at(99) - 1.1).abs() < 1e-12,
        "saturates"
    );
    assert_eq!(tracker.cumulative_regret_at(0), 0.0);

    let curve = tracker.cumulative_curve();
    assert!(max_abs_diff(&curve, &[0.6, 1.1, 1.1]) < 1e-12);
}

#[test]
fn regret_tracker_clamps_a_negative_gap_to_zero() {
    // The oracle maximizes over the same arm set the policy chose from, so
    // `optimal >= actual` holds by definition. A negative difference can only be
    // the caller's floating-point noise -- and letting it accumulate would let a
    // long run drift its regret *downward*, which is meaningless.
    let mut tracker = BanditRegretTracker::new();
    tracker.record(1.0, 1.0 + 1e-16);
    assert_eq!(tracker.cumulative_regret(), 0.0);
    assert_eq!(tracker.instantaneous_regrets(), &[0.0]);
}

#[test]
fn regret_of_a_single_arm_bandit_is_exactly_zero() {
    // With one arm the policy has no choice, so it cannot possibly be wrong: the
    // oracle's arm and the policy's arm are the same arm, every round.
    let environment = LinearEnvironment::new(vec![("only", vec![0.7, -0.2, 0.4])], 0.1);
    let dim = 3;

    let config = BanditConfig::with_dimension(dim).expect("valid");
    let mut ranker = LinUcbRanker::from_ids(config, ["only"]).expect("one arm");
    let tracker = run_simulation(&mut ranker, &environment, 500, dim, 1, 2, true)
        .expect("simulation runs cleanly");

    assert_eq!(tracker.rounds(), 500);
    assert_eq!(
        tracker.cumulative_regret(),
        0.0,
        "a one-armed bandit has *exactly* zero regret, not approximately zero"
    );
    assert_eq!(tracker.average_regret(), 0.0);
    for regret in tracker.instantaneous_regrets() {
        assert_eq!(*regret, 0.0);
    }
    assert!(
        (tracker.policy_reward() - tracker.oracle_reward()).abs() < 1e-12,
        "the policy *is* the oracle"
    );
}

#[test]
fn regret_windows_drop_the_remainder_rather_than_shortening_the_last_window() {
    // 10 rounds into 3 windows: 3 per window, and the 10th round is dropped, so all
    // three averages are computed over the same denominator and are comparable.
    let mut tracker = BanditRegretTracker::new();
    for value in [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 100.0] {
        tracker.record(value, 0.0);
    }
    let windows = tracker.window_average_regret(3);
    assert!(max_abs_diff(&windows, &[1.0, 2.0, 3.0]) < 1e-12);
    assert!(tracker.window_average_regret(0).is_empty());
    assert!(
        tracker.window_average_regret(11).is_empty(),
        "more windows than rounds is not a meaningful split"
    );
    assert_eq!(tracker.window_average_regret(10).len(), 10);
}

// ═════════════════════════════════════════════════════════════════════════════
// Off-policy replay evaluation
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn off_policy_replay_estimates_the_true_online_value() {
    // Replay a LinUCB policy against a log of uniformly-random actions, and compare
    // the estimate against what the *same* policy actually earns when run online
    // against the same environment for the same horizon. If the unbiasedness
    // argument is right, they must agree.
    let environment = LinearEnvironment::new(
        vec![
            ("a", vec![0.80, 0.10, 0.20, 0.10]),
            ("b", vec![0.20, 0.70, 0.10, 0.20]),
            ("c", vec![0.10, 0.10, 0.30, 0.30]),
        ],
        0.10,
    );
    let dim = 4;
    let log_size = 60_000;
    let log = generate_uniform_log(&environment, log_size, dim, 71, 72, 73);

    let config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_alpha(0.5);
    let mut replay_policy =
        LinUcbRanker::from_ids(config.clone(), environment.arm_ids()).expect("distinct ids");
    let estimate = BanditOffPolicyEvaluator::new()
        .evaluate(&mut replay_policy, &log)
        .expect("some events are accepted");

    // Acceptance is a *constant* 1/K, whatever the policy does -- that is the whole
    // basis of the unbiasedness argument, and it is directly observable.
    assert_eq!(estimate.num_arms, 3);
    assert_eq!(estimate.total_events, log_size);
    assert!((estimate.expected_acceptance_rate() - 1.0 / 3.0).abs() < 1e-12);
    assert!(
        (estimate.acceptance_rate - 1.0 / 3.0).abs() < 0.02,
        "a uniform 3-arm log must be accepted at ~1/3, got {}",
        estimate.acceptance_rate
    );
    assert_eq!(estimate.effective_sample_size(), estimate.accepted_events);
    assert!(
        estimate.accepted_events > 18_000,
        "roughly log_size / K events should survive, got {}",
        estimate.accepted_events
    );

    // Now the ground truth: run the same policy online for the same horizon.
    let horizon = estimate.accepted_events;
    let mut online_policy =
        LinUcbRanker::from_ids(config, environment.arm_ids()).expect("distinct ids");
    let online_tracker = run_simulation(
        &mut online_policy,
        &environment,
        horizon,
        dim,
        81,
        82,
        false,
    )
    .expect("simulation runs cleanly");
    let online_value = mean_policy_value(&online_tracker);

    assert!(
        (estimate.estimated_value - online_value).abs() < 0.05,
        "the replay estimate ({}) must match the policy's true online value ({online_value})",
        estimate.estimated_value
    );

    // And the replayed policy learned the same thing the online one did.
    assert_eq!(
        replay_policy.stats().most_pulled_arm(),
        online_policy.stats().most_pulled_arm()
    );
}

#[test]
fn off_policy_replay_beats_the_logging_policy_it_replays() {
    // A sanity check with real content: the logging policy is uniform random, so its
    // own value is the *average* arm's value. A learned policy replayed against that
    // log must be estimated as strictly better -- otherwise the estimator is not
    // measuring the evaluated policy at all.
    let environment = LinearEnvironment::new(
        vec![
            ("good", vec![0.9, 0.9]),
            ("mid", vec![0.4, 0.4]),
            ("bad", vec![0.1, 0.1]),
        ],
        0.05,
    );
    let dim = 2;
    let log = generate_uniform_log(&environment, 30_000, dim, 1, 2, 3);

    let logging_value: f64 = log.iter().map(|event| event.reward).sum::<f64>() / log.len() as f64;

    let config = BanditConfig::with_dimension(dim)
        .expect("valid")
        .set_alpha(0.5);
    let mut policy = LinUcbRanker::from_ids(config, environment.arm_ids()).expect("distinct ids");
    let estimate = BanditOffPolicyEvaluator::new()
        .evaluate(&mut policy, &log)
        .expect("some events accepted");

    assert!(
        estimate.estimated_value > logging_value + 0.2,
        "the learned policy ({}) must be estimated well above the uniform logging \
         policy ({logging_value})",
        estimate.estimated_value
    );
    assert_eq!(policy.stats().most_pulled_arm(), Some("good"));
}

#[test]
fn off_policy_replay_honours_its_horizon() {
    let environment =
        LinearEnvironment::new(vec![("a", vec![0.5, 0.5]), ("b", vec![0.2, 0.2])], 0.05);
    let log = generate_uniform_log(&environment, 4_000, 2, 9, 10, 11);

    let config = BanditConfig::with_dimension(2).expect("valid");
    let mut policy = LinUcbRanker::from_ids(config, environment.arm_ids()).expect("distinct ids");
    let estimate = BanditOffPolicyEvaluator::new()
        .with_horizon(100)
        .evaluate(&mut policy, &log)
        .expect("100 events are reachable");

    assert_eq!(estimate.accepted_events, 100);
    assert!(
        estimate.total_events < 4_000,
        "the evaluator must stop early once the horizon is met, examined {}",
        estimate.total_events
    );
    assert_eq!(
        policy.stats().total_pulls,
        100,
        "only accepted events teach the policy"
    );
}

#[test]
fn off_policy_replay_reports_when_nothing_is_accepted() {
    // A log whose actions the policy never chooses yields no sample at all. Returning
    // a `0.0` "estimate" would be a fabrication; the evaluator says so instead.
    let context = BanditContext::new(vec![1.0]).expect("finite");
    let log = vec![
        BanditLoggedEvent::new(context.clone(), "b", 1.0),
        BanditLoggedEvent::new(context, "b", 1.0),
    ];

    // Only arm "a" is registered, so the policy can never choose "b".
    let config = BanditConfig::with_dimension(1).expect("valid");
    let mut policy = LinUcbRanker::from_ids(config, ["a"]).expect("one arm");
    let result = BanditOffPolicyEvaluator::new().evaluate(&mut policy, &log);

    assert!(matches!(
        result,
        Err(BanditError::NoAcceptedEvents { total: 2 })
    ));
}

#[test]
fn off_policy_replay_propagates_a_dimension_mismatch() {
    let log = vec![BanditLoggedEvent::new(
        BanditContext::new(vec![1.0, 2.0, 3.0]).expect("finite"),
        "a",
        1.0,
    )];
    let config = BanditConfig::with_dimension(2).expect("valid");
    let mut policy = LinUcbRanker::from_ids(config, ["a"]).expect("one arm");
    assert!(matches!(
        BanditOffPolicyEvaluator::new().evaluate(&mut policy, &log),
        Err(BanditError::DimensionMismatch { .. })
    ));
}

#[test]
fn off_policy_evaluator_works_through_a_trait_object_for_every_policy() {
    // The evaluator takes `&mut dyn BanditRanker`, so one implementation serves all
    // three policies -- and any policy a downstream crate writes.
    let environment =
        LinearEnvironment::new(vec![("a", vec![0.7, 0.2]), ("b", vec![0.2, 0.7])], 0.05);
    let log = generate_uniform_log(&environment, 6_000, 2, 21, 22, 23);
    let config = BanditConfig::with_dimension(2).expect("valid").set_seed(5);

    let mut linucb =
        LinUcbRanker::from_ids(config.clone(), environment.arm_ids()).expect("distinct ids");
    let mut thompson = ThompsonSamplingRanker::from_ids(config.clone(), environment.arm_ids())
        .expect("distinct ids");
    let mut greedy =
        EpsilonGreedyRanker::from_ids(config, environment.arm_ids()).expect("distinct ids");

    let policies: Vec<&mut dyn BanditRanker> = vec![&mut linucb, &mut thompson, &mut greedy];
    let evaluator = BanditOffPolicyEvaluator::new();
    let estimates: Vec<BanditOffPolicyEstimate> = policies
        .into_iter()
        .map(|policy| {
            evaluator
                .evaluate(policy, &log)
                .expect("events are accepted")
        })
        .collect();

    for estimate in &estimates {
        assert_eq!(estimate.num_arms, 2);
        // Every policy sees the same ~1/K acceptance rate: it depends on the *log*,
        // not on the policy.
        assert!((estimate.acceptance_rate - 0.5).abs() < 0.03);
        assert!(estimate.estimated_value.is_finite());
        assert!(estimate.accepted_events > 2_500);
    }
}
