//! Measuring a bandit: [`BanditRegretTracker`] (online) and
//! [`BanditOffPolicyEvaluator`] (offline replay).
//!
//! A bandit cannot be evaluated the way a ranker or a classifier can, and the two
//! types here exist because of that. There is no held-out test set: the policy's
//! *actions* determine which rewards it ever gets to see, so its own behaviour is
//! entangled with its own evaluation. The two standard escapes are:
//!
//! * **Regret**, when the truth is known (a simulator, a synthetic benchmark, a
//!   replayed environment with a ground-truth reward model): compare what the
//!   policy earned against what an oracle *would* have earned, round by round.
//! * **Replay / rejection sampling**, when the truth is *not* known but a log of
//!   uniformly-random actions exists: [`BanditOffPolicyEvaluator`], which turns
//!   that log into an unbiased estimate of a policy's online value without ever
//!   deploying it.

use super::types::{BanditContext, BanditError, BanditRanker, BanditResult};
use serde::{Deserialize, Serialize};

// ── BanditRegretTracker ──────────────────────────────────────────────────────

/// Cumulative regret against the best arm available on each round.
///
/// # What regret is
///
/// On round `t` the environment presents a context `x_t`. Each arm `a` has some
/// true expected reward `mu_a(x_t)`. The policy pulls `a_t`. Its **instantaneous
/// regret** is what it left on the table:
///
/// ```text
/// r_t  =  max_a mu_a(x_t)  -  mu_{a_t}(x_t)   >=  0
/// ```
///
/// and its **cumulative regret** is `Regret(T) = sum_{t=1..T} r_t`. Note both
/// terms are *expected* rewards, not the noisy realized ones: regret measures the
/// quality of the *decision*, not the luck of the draw. A policy that picks the
/// best arm and then gets an unlucky sample has zero regret on that round, and
/// rightly so.
///
/// # Why sublinearity is the property that matters
///
/// The **only** thing that separates a bandit from a random number generator is
/// that its regret grows *sublinearly*:
///
/// ```text
/// Regret(T) / T  ->  0     as T -> infinity
/// ```
///
/// Read it as: the policy's average per-round payoff converges to the *oracle's*
/// average per-round payoff. It is "no-regret" — asymptotically it loses nothing
/// to a strategy that knew the true parameters all along.
///
/// A policy with **linear** regret has `Regret(T) / T -> c > 0`: it keeps paying
/// a fixed tax per round, forever, and never converges to the oracle. That is what
/// a broken bandit looks like from the outside — and, crucially, it is *not*
/// obvious from anything else. A locked-on greedy policy still returns rankings,
/// still learns a model, still reports plausible-looking statistics, and still
/// earns a perfectly respectable-looking mean reward. The single measurement that
/// exposes it is the shape of this curve. [`BanditRegretTracker::window_average_regret`]
/// is the shape test: split the run into equal windows and check the average
/// regret *falls* across them. If it plateaus, the policy is not learning, and no
/// adjustment to the threshold will change that fact.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "bandit-ranker")]
/// # {
/// use oxirag::bandit_ranker::BanditRegretTracker;
///
/// let mut tracker = BanditRegretTracker::new();
/// // A policy that gets steadily better: it forgoes 0.5, then 0.2, then 0.0.
/// tracker.record(1.0, 0.5);
/// tracker.record(1.0, 0.8);
/// tracker.record(1.0, 1.0);
///
/// assert!((tracker.cumulative_regret() - 0.7).abs() < 1e-12);
/// assert_eq!(tracker.rounds(), 3);
/// # }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BanditRegretTracker {
    /// Per-round instantaneous regret, in order.
    instantaneous: Vec<f64>,
    /// Running total of `instantaneous`, so `cumulative_regret` is `O(1)`.
    cumulative: f64,
    /// Running total of the *realized* expected reward the policy earned.
    policy_reward: f64,
    /// Running total of the oracle's expected reward.
    oracle_reward: f64,
}

impl BanditRegretTracker {
    /// A tracker with no rounds recorded.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one round: the best arm's expected reward and the pulled arm's
    /// expected reward.
    ///
    /// The instantaneous regret is `max(0, optimal - actual)`. The clamp is not a
    /// fudge: `optimal >= actual` holds **by definition** (the oracle maximizes
    /// over the same set the policy chose from), so a negative difference can only
    /// be floating-point noise from the caller computing the two through slightly
    /// different arithmetic paths — and letting that noise accumulate into the
    /// cumulative sum would let a long run drift its regret *downward*, which is
    /// meaningless. A caller who passes a genuinely larger `actual` than `optimal`
    /// has a bug in their oracle, and this clamp is the honest floor of `0` rather
    /// than a silent negative credit.
    pub fn record(&mut self, optimal_expected_reward: f64, actual_expected_reward: f64) {
        let regret = (optimal_expected_reward - actual_expected_reward).max(0.0);
        self.instantaneous.push(regret);
        self.cumulative += regret;
        self.policy_reward += actual_expected_reward;
        self.oracle_reward += optimal_expected_reward;
    }

    /// How many rounds have been recorded.
    #[must_use]
    pub fn rounds(&self) -> usize {
        self.instantaneous.len()
    }

    /// Whether no rounds have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instantaneous.is_empty()
    }

    /// `Regret(T)` — the total reward the policy forfeited.
    #[must_use]
    pub fn cumulative_regret(&self) -> f64 {
        self.cumulative
    }

    /// `Regret(T) / T` — the quantity that must tend to zero. `0.0` before the
    /// first round.
    #[must_use]
    pub fn average_regret(&self) -> f64 {
        if self.instantaneous.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)] // Round counts stay far below 2^53.
        {
            self.cumulative / self.instantaneous.len() as f64
        }
    }

    /// The per-round instantaneous regrets, in order.
    #[must_use]
    pub fn instantaneous_regrets(&self) -> &[f64] {
        &self.instantaneous
    }

    /// `Regret(t)` — cumulative regret over the *first* `t` rounds, saturating at
    /// the number of rounds actually recorded.
    ///
    /// This is what a sublinearity check like `Regret(2T) < 2 * Regret(T)` is
    /// written in terms of.
    #[must_use]
    pub fn cumulative_regret_at(&self, rounds: usize) -> f64 {
        let end = rounds.min(self.instantaneous.len());
        self.instantaneous[..end].iter().sum()
    }

    /// The full cumulative-regret curve, `[Regret(1), Regret(2), ..., Regret(T)]`
    /// — the thing you plot.
    #[must_use]
    pub fn cumulative_curve(&self) -> Vec<f64> {
        let mut running = 0.0;
        self.instantaneous
            .iter()
            .map(|regret| {
                running += regret;
                running
            })
            .collect()
    }

    /// Split the run into `windows` contiguous equal-length windows and return
    /// the *average* regret within each.
    ///
    /// **This is the sublinearity test.** For a no-regret policy the sequence
    /// falls, because `Regret(T) = o(T)` forces the per-round regret to decay; for
    /// a policy with linear regret it plateaus at a positive constant, because it
    /// is paying the same tax every round. The averages are the honest statistic
    /// to compare (a *cumulative* curve rises for any policy, including a good
    /// one, and so tells you nothing on its own).
    ///
    /// Any remainder rounds — when `windows` does not divide `T` — are dropped
    /// from the *last* window rather than being folded into a short final window,
    /// so every returned average is computed over exactly the same number of
    /// rounds and the comparison between them is apples-to-apples.
    ///
    /// Returns an empty vector if `windows` is zero or exceeds the number of
    /// recorded rounds.
    #[must_use]
    pub fn window_average_regret(&self, windows: usize) -> Vec<f64> {
        if windows == 0 || windows > self.instantaneous.len() {
            return Vec::new();
        }
        let window_size = self.instantaneous.len() / windows;
        #[allow(clippy::cast_precision_loss)] // Window sizes stay far below 2^53.
        let divisor = window_size as f64;
        (0..windows)
            .map(|index| {
                let start = index * window_size;
                let end = start + window_size;
                self.instantaneous[start..end].iter().sum::<f64>() / divisor
            })
            .collect()
    }

    /// The total *expected* reward the policy earned.
    #[must_use]
    pub fn policy_reward(&self) -> f64 {
        self.policy_reward
    }

    /// The total expected reward an oracle would have earned on the same rounds.
    #[must_use]
    pub fn oracle_reward(&self) -> f64 {
        self.oracle_reward
    }
}

// ── BanditLoggedEvent ────────────────────────────────────────────────────────

/// One row of an exploration log: the context that was seen, the action that was
/// taken, and the reward that came back.
///
/// For [`BanditOffPolicyEvaluator`] to be *unbiased*, the `arm_id` in each event
/// must have been chosen **uniformly at random**, independently of the context and
/// of everything the logging system knew. That is a real operational cost — it
/// means running a small uniformly-random exploration bucket in production — and
/// it is the price of being able to evaluate *any future policy* offline, for
/// free, forever, against that log. It is a bargain, and it is why the technique
/// is standard practice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditLoggedEvent {
    /// The context the logging policy saw.
    pub context: BanditContext,
    /// The arm the logging policy pulled — which **must** have been drawn
    /// uniformly at random over the arm set.
    pub arm_id: String,
    /// The reward that followed.
    pub reward: f64,
}

impl BanditLoggedEvent {
    /// Assemble a logged event.
    #[must_use]
    pub fn new(context: BanditContext, arm_id: impl Into<String>, reward: f64) -> Self {
        Self {
            context,
            arm_id: arm_id.into(),
            reward,
        }
    }
}

// ── BanditOffPolicyEstimate ──────────────────────────────────────────────────

/// The result of an off-policy replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditOffPolicyEstimate {
    /// The unbiased estimate of the policy's per-round value: the mean reward
    /// over the *accepted* events.
    pub estimated_value: f64,
    /// How many logged events were accepted (i.e. the policy's top-ranked arm
    /// matched the logged arm).
    pub accepted_events: usize,
    /// How many logged events were examined in total.
    pub total_events: usize,
    /// `accepted_events / total_events`.
    ///
    /// Diagnostic, and an important one. For a `K`-arm uniformly-random log this
    /// should sit at approximately `1 / K` **regardless of the policy being
    /// evaluated** — the logging policy's coin does not know or care what the
    /// evaluated policy chose. A rate far from `1 / K` is therefore evidence that
    /// the log was *not* uniformly random, which silently invalidates the
    /// unbiasedness argument; see [`BanditOffPolicyEstimate::expected_acceptance_rate`].
    pub acceptance_rate: f64,
    /// The number of arms the evaluated policy had, so a caller can compare
    /// `acceptance_rate` against `1 / num_arms` without recomputing it.
    pub num_arms: usize,
    /// The sum of the accepted rewards.
    pub total_reward: f64,
}

impl BanditOffPolicyEstimate {
    /// `1 / num_arms` — the acceptance rate a genuinely uniform log must produce.
    ///
    /// Compare against [`BanditOffPolicyEstimate::acceptance_rate`]. A large
    /// discrepancy means the log's actions were *not* uniform, and the estimate
    /// should not be trusted: the replay estimator has no way to detect that on
    /// its own, so this is the check that has to be made by hand.
    #[must_use]
    pub fn expected_acceptance_rate(&self) -> f64 {
        if self.num_arms == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)] // Arm counts are tiny.
        {
            1.0 / self.num_arms as f64
        }
    }

    /// The effective sample size of the estimate — just `accepted_events`, named
    /// for what it is.
    ///
    /// Replay throws away roughly a `1 - 1/K` fraction of the log, so a `K`-arm
    /// evaluation over `N` logged rows yields an estimate with the precision of
    /// about `N / K` samples. That factor of `K` is the entire cost of the method,
    /// and it is worth stating loudly: it is *unbiased*, not *free*.
    #[must_use]
    pub fn effective_sample_size(&self) -> usize {
        self.accepted_events
    }
}

// ── BanditOffPolicyEvaluator ─────────────────────────────────────────────────

/// Unbiased offline evaluation of a bandit policy by **replay** (Li, Chu,
/// Langford & Wang, 2011, "Unbiased Offline Evaluation of Contextual-Bandit-Based
/// News Article Recommendation Algorithms").
///
/// # The method
///
/// Walk the logged stream in order. For each event, ask the policy what it would
/// do on that context. If its top-ranked arm **matches** the logged arm, *accept*
/// the event: reveal the logged reward to the policy (letting it learn, exactly as
/// it would online) and count the reward. If it does not match, **discard** the
/// event entirely — the policy is told nothing, learns nothing, and its state is
/// untouched, as though the event had never happened.
///
/// The estimate is the mean reward over accepted events.
///
/// # Why this is unbiased
///
/// The claim is that the accepted subsequence is distributed *exactly* as a real
/// online run of the policy against the same environment. Here is why.
///
/// Consider the policy's state after `k` accepted events. It is a deterministic
/// (or, for a randomized policy, an independently-randomized) function of those
/// `k` accepted events alone — the discarded ones left no trace, by construction.
/// Now feed it the next logged event, with context `x` drawn from the environment
/// and logged action `a_log` drawn **uniformly** over the `K` arms, independently
/// of `x` and of everything else. The policy names some arm `a_pi`. Then:
///
/// ```text
/// P( accept | x, policy state, a_pi )  =  P( a_log = a_pi )  =  1 / K
/// ```
///
/// — **a constant.** It does not depend on the context, it does not depend on the
/// policy's state, and, decisively, it does not depend on *which arm the policy
/// chose*. This is exactly what uniform logging buys, and it is the crux of the
/// whole argument: an acceptance rule whose probability depended on the chosen arm
/// would preferentially retain some of the policy's decisions over others and so
/// distort the very distribution being estimated.
///
/// Because the acceptance probability is a constant, conditioning on acceptance
/// changes *nothing* about the joint law of `(x, a_pi, reward)`. By Bayes' rule the
/// conditional law of the context given acceptance is the same as its marginal, and
/// the reward is drawn from the environment's true `P(r | x, a_pi)` either way. So
/// each accepted event is a faithful draw from the policy's own online interaction
/// distribution, the state it carries forward is the state it would have carried
/// forward online, and by induction the whole accepted subsequence is a genuine
/// sample path of the policy. The mean of its rewards is therefore an unbiased
/// estimate of the policy's per-round value.
///
/// # What it costs
///
/// Acceptance keeps a `1 / K` fraction of the log, so `N` logged rows buy an
/// evaluation of horizon `≈ N / K`. The estimator is unbiased but *lossy*; see
/// [`BanditOffPolicyEstimate::effective_sample_size`].
///
/// # What breaks it
///
/// A log whose actions were *not* uniform (say, collected under a previous
/// production policy) violates the constancy of `P(accept | ...)` and biases the
/// estimate toward whatever that previous policy preferred — and the estimator
/// cannot tell. [`BanditOffPolicyEstimate::acceptance_rate`] against
/// [`BanditOffPolicyEstimate::expected_acceptance_rate`] is the one sanity check
/// available; it is necessary but not sufficient, and it is why
/// [`BanditLoggedEvent`] says what it says.
#[derive(Debug, Clone, Copy, Default)]
pub struct BanditOffPolicyEvaluator {
    /// Stop after this many *accepted* events, if set. Useful for holding the
    /// evaluation horizon fixed across policies that would otherwise consume
    /// different amounts of the log.
    horizon: Option<usize>,
}

impl BanditOffPolicyEvaluator {
    /// An evaluator that consumes the whole log.
    #[must_use]
    pub fn new() -> Self {
        Self { horizon: None }
    }

    /// Stop once `horizon` events have been accepted.
    #[must_use]
    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.horizon = Some(horizon);
        self
    }

    /// Replay `events` against `policy` and return an unbiased estimate of the
    /// policy's per-round value.
    ///
    /// `policy` is **mutated**: it learns from every accepted event, exactly as it
    /// would online. Pass a freshly-constructed policy unless you specifically
    /// want to evaluate a warm-started one.
    ///
    /// # Errors
    ///
    /// * [`BanditError::NoAcceptedEvents`] if not a single logged action ever
    ///   matched the policy's choice — there is no sample to average, and
    ///   returning a `0.0` "estimate" would be a fabrication.
    /// * Whatever [`BanditRanker::select`] / [`BanditRanker::update`] return
    ///   (dimension mismatches, unknown arms, numerical failures).
    pub fn evaluate(
        &self,
        policy: &mut dyn BanditRanker,
        events: &[BanditLoggedEvent],
    ) -> BanditResult<BanditOffPolicyEstimate> {
        let num_arms = policy.arm_ids().len();
        let mut accepted = 0_usize;
        let mut examined = 0_usize;
        let mut total_reward = 0.0;

        for event in events {
            if self.horizon.is_some_and(|horizon| accepted >= horizon) {
                break;
            }
            examined += 1;

            let ranking = policy.select(&event.context)?;
            let Some(chosen) = ranking.top_arm_id() else {
                // `select` errors on an empty arm set rather than returning an
                // empty ranking, so this is unreachable — but it is a `let-else`
                // rather than an `expect`, because "unreachable" is a claim about
                // today's code and a panic would be a claim about all future code.
                return Err(BanditError::NoArms);
            };

            if chosen == event.arm_id {
                // Accept: the policy's decision coincided with the logged
                // uniformly-random one, so this event is a faithful sample of what
                // the policy would have experienced online.
                let arm_id = event.arm_id.clone();
                policy.update(&arm_id, &event.context, event.reward)?;
                accepted += 1;
                total_reward += event.reward;
            }
            // Discard otherwise: the policy is told nothing and its state is left
            // exactly as it was. This is what keeps the retained subsequence a
            // genuine sample path.
        }

        if accepted == 0 {
            return Err(BanditError::NoAcceptedEvents { total: examined });
        }

        #[allow(clippy::cast_precision_loss)] // Event counts stay far below 2^53.
        let (estimated_value, acceptance_rate) = {
            let accepted_f64 = accepted as f64;
            let examined_f64 = examined as f64;
            (
                total_reward / accepted_f64,
                if examined == 0 {
                    0.0
                } else {
                    accepted_f64 / examined_f64
                },
            )
        };

        Ok(BanditOffPolicyEstimate {
            estimated_value,
            accepted_events: accepted,
            total_events: examined,
            acceptance_rate,
            num_arms,
            total_reward,
        })
    }
}
