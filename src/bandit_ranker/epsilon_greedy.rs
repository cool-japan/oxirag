//! [`EpsilonGreedyRanker`] — the simplest exploration strategy there is, and the
//! one that shows most clearly *why* the other two are worth their complexity.
//!
//! # The rule
//!
//! ```text
//! with probability eps_t :  rank the arms uniformly at random
//! otherwise              :  rank the arms by  theta_hat_a^T x   (pure greedy)
//! ```
//!
//! Because this module is a *ranker* rather than an arm-picker, the exploration
//! branch produces a uniformly random **permutation** (a Fisher–Yates shuffle)
//! rather than a single random arm. The two coincide where it matters: the first
//! element of a uniformly random permutation is uniform over the arms, so
//! `select(...).ranked[0]` is exactly the classical ε-greedy action — "a uniformly
//! random arm with probability `eps_t`, the greedy arm otherwise" — while the
//! rest of the list stays a usable ordering for a caller that must fill several
//! slots. [`BanditRanking::explored`] reports which branch was taken.
//!
//! # The decay schedule, and why a *constant* ε is broken
//!
//! With a **constant** ε, the policy explores at a fixed rate forever. Even after
//! it has learned everything there is to learn, it still throws away a fraction
//! `eps * (1 - 1/K)` of its rounds on arms it *knows* are worse, paying the
//! suboptimality gap on each. Its regret is therefore
//!
//! ```text
//! Regret(T)  ~  eps * (1 - 1/K) * average_gap * T   =   Theta(T)
//! ```
//!
//! — **linear**. Average regret converges to a positive constant, not to zero. A
//! constant-ε bandit is not a learning algorithm in the regret sense at all; it
//! is a learning algorithm bolted to a permanent tax.
//!
//! The fix is to let ε *anneal*. With [`BanditConfig::decay_epsilon`] set, this
//! ranker uses
//!
//! ```text
//! eps_t = min(1, eps_0 / sqrt(t))
//! ```
//!
//! where `t` is the number of `select` calls so far (1-based). The exploration
//! cost then telescopes into
//!
//! ```text
//! sum_{t=1..T} eps_0 / sqrt(t)  ~  2 * eps_0 * sqrt(T)   =   O(sqrt(T))
//! ```
//!
//! which is sublinear — average regret does go to zero. (The classical
//! `eps_t ~ 1/t` schedule of Auer, Cesa-Bianchi & Fischer, 2002 gives an even
//! better `O(log T)` *if* its constant is tuned to the unknown gap; the
//! `1/sqrt(t)` schedule is the standard gap-free choice, and it is the one that
//! degrades gracefully when you cannot tune it.)
//!
//! # And why it still loses to `LinUCB` and Thompson
//!
//! Even annealed, ε-greedy explores **blindly**: when it explores, it picks
//! uniformly, spending exactly as much effort on an arm it has pulled ten
//! thousand times as on one it has never seen, and exactly as much on a context
//! direction it understands perfectly as on one it has never observed. `LinUCB` and
//! Thompson sampling explore **where the uncertainty is** — their exploration
//! term is a function of `A_a^-1`, which encodes *which arm* and *which
//! direction* remain unknown. That is the entire difference, and it is why both
//! of them beat this ranker on the module's benchmark problems while using the
//! same underlying model.
//!
//! ε-greedy is nonetheless kept, and kept honest, for two reasons: it is the
//! baseline that makes the others' advantage *measurable*, and its `eps = 0`
//! configuration is a pure-greedy policy — the deceptive-arm trap that the
//! module's tests use to prove exploration is doing real work.

use super::model::{BanditArmSet, assign_ranks, rank_by_score_desc};
use super::rng::SplitMix64Rng;
use super::types::{
    BanditArm, BanditConfig, BanditContext, BanditError, BanditRankedArm, BanditRanker,
    BanditRanking, BanditResult, BanditStats,
};

/// ε-greedy over the same disjoint linear model `LinUCB` and Thompson sampling
/// use, with an optional `eps_0 / sqrt(t)` decay schedule.
///
/// See the [module documentation](self) for the schedule, the linear-regret
/// pathology of a constant ε, and why uniform exploration is strictly weaker than
/// uncertainty-directed exploration.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "bandit-ranker")]
/// # {
/// use oxirag::bandit_ranker::{
///     BanditConfig, BanditContext, BanditRanker, EpsilonGreedyRanker,
/// };
///
/// // eps = 0 is pure greedy: it never takes the exploration branch.
/// let config = BanditConfig::with_dimension(2)
///     .expect("dimension 2 is valid")
///     .set_epsilon(0.0);
/// let mut greedy = EpsilonGreedyRanker::from_ids(config, ["a", "b"])
///     .expect("two distinct arm ids");
///
/// let context = BanditContext::new(vec![1.0, 1.0]).expect("finite features");
/// let ranking = greedy.select(&context).expect("arms exist");
/// assert!(!ranking.explored);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct EpsilonGreedyRanker {
    /// The per-arm ridge models.
    arms: BanditArmSet,
    /// The configuration this ranker was built with.
    config: BanditConfig,
    /// The deterministic RNG behind the ε coin and the exploration shuffle.
    rng: SplitMix64Rng,
    /// How many times [`BanditRanker::select`] has been called. Drives the decay
    /// schedule; incremented on *selection*, not on update, because it is
    /// selections that spend the exploration budget.
    rounds: u64,
    /// How many of those rounds took the exploration branch.
    exploration_rounds: u64,
}

impl EpsilonGreedyRanker {
    /// Build an ε-greedy ranker over `arms`, seeded from [`BanditConfig::seed`].
    ///
    /// # Errors
    ///
    /// * [`BanditError::InvalidConfig`] if `config` is out of range (notably:
    ///   `epsilon` must lie in `[0, 1]`).
    /// * [`BanditError::DuplicateArm`] if two arms share an id.
    pub fn new(
        config: BanditConfig,
        arms: impl IntoIterator<Item = BanditArm>,
    ) -> BanditResult<Self> {
        config.validate()?;
        let arm_set = BanditArmSet::new(arms, config.dimension, config.ridge_lambda)?;
        let rng = SplitMix64Rng::new(config.seed);
        Ok(Self {
            arms: arm_set,
            config,
            rng,
            rounds: 0,
            exploration_rounds: 0,
        })
    }

    /// Build an ε-greedy ranker over bare arm ids.
    ///
    /// # Errors
    ///
    /// As [`EpsilonGreedyRanker::new`].
    pub fn from_ids(
        config: BanditConfig,
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> BanditResult<Self> {
        Self::new(config, ids.into_iter().map(BanditArm::new))
    }

    /// Register a new arm mid-flight.
    ///
    /// Note the asymmetry with `LinUCB` and Thompson sampling, and it is a real
    /// weakness of this policy: a brand-new arm gets *no* special attention here.
    /// Its `theta_hat` is zero, so greedy will simply ignore it unless every
    /// incumbent has a negative estimate, and it will only ever be tried by the
    /// blind uniform coin. The uncertainty-aware policies, by contrast, see a new
    /// arm's maximal confidence width and go and look at it immediately.
    ///
    /// # Errors
    ///
    /// [`BanditError::DuplicateArm`] if the id is already registered.
    pub fn add_arm(&mut self, arm: BanditArm) -> BanditResult<()> {
        self.arms.add(arm)
    }

    /// The configuration this ranker was built with.
    #[must_use]
    pub fn config(&self) -> &BanditConfig {
        &self.config
    }

    /// The per-arm models, in registration order.
    #[must_use]
    pub fn arms(&self) -> &BanditArmSet {
        &self.arms
    }

    /// How many selections have been made.
    #[must_use]
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// How many of those selections took the exploration branch.
    #[must_use]
    pub fn exploration_rounds(&self) -> u64 {
        self.exploration_rounds
    }

    /// The exploration probability that *would* be used on the next selection.
    ///
    /// With decay enabled this is `min(1, eps_0 / sqrt(t))` for the next round
    /// `t = rounds + 1` (so the very first selection uses exactly `eps_0`, and it
    /// falls from there); without decay it is the constant `eps_0`.
    ///
    /// The `min(1, ...)` clamp is defensive rather than necessary — `eps_0` is
    /// validated into `[0, 1]` and `sqrt(t) >= 1`, so the quotient can never
    /// exceed `eps_0` — but a schedule that could silently exceed a probability
    /// of one is the kind of bug that produces "why does it explore forever?"
    /// with no visible cause, so the clamp stays.
    #[must_use]
    pub fn current_epsilon(&self) -> f64 {
        if !self.config.decay_epsilon {
            return self.config.epsilon;
        }
        #[allow(clippy::cast_precision_loss)] // Round counts stay far below 2^53.
        let next_round = (self.rounds + 1) as f64;
        (self.config.epsilon / next_round.sqrt()).min(1.0)
    }
}

impl BanditRanker for EpsilonGreedyRanker {
    fn select(&mut self, context: &BanditContext) -> BanditResult<BanditRanking> {
        if self.arms.is_empty() {
            return Err(BanditError::NoArms);
        }
        context.require_dimension(self.config.dimension)?;

        let epsilon = self.current_epsilon();
        self.rounds += 1;

        // Every arm is scored by its plain exploitation estimate, whichever
        // branch we take: even on an exploration round the caller deserves to see
        // what the model *believed*, and `mean_estimate` is what the ranking would
        // have been sorted by.
        let x = context.features();
        let mut scored = Vec::with_capacity(self.arms.len());
        for model in self.arms.models() {
            let mean_estimate = model.mean_estimate(x)?;
            scored.push(BanditRankedArm {
                arm_id: model.arm_id().to_string(),
                score: mean_estimate,
                mean_estimate,
                rank: 0,
            });
        }

        // The coin. `next_f64` is uniform on [0, 1), so `draw < epsilon` fires
        // with probability exactly `epsilon` — and, importantly, *never* fires
        // when `epsilon == 0.0` (no draw is < 0) and *always* fires when
        // `epsilon == 1.0` (every draw is < 1, because the interval is
        // half-open). Both extremes are therefore exact rather than approximate,
        // which is what lets the tests assert them as hard equalities.
        let draw = self.rng.next_f64();
        let explored = draw < epsilon;

        if explored {
            self.exploration_rounds += 1;
            // Uniformly random permutation: its first element is uniform over the
            // arms, which is precisely classical ε-greedy's exploration step.
            self.rng.shuffle(&mut scored);
            assign_ranks(&mut scored);
        } else {
            rank_by_score_desc(&mut scored);
        }

        Ok(BanditRanking {
            ranked: scored,
            explored,
        })
    }

    fn update(&mut self, arm_id: &str, context: &BanditContext, reward: f64) -> BanditResult<()> {
        context.require_dimension(self.config.dimension)?;
        self.arms
            .get_mut(arm_id)?
            .update(context.features(), reward)
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn arm_ids(&self) -> Vec<&str> {
        self.arms.arm_ids()
    }

    fn theta(&self, arm_id: &str) -> Option<&[f64]> {
        self.arms
            .get(arm_id)
            .map(super::model::BanditLinearModel::theta)
    }

    fn stats(&self) -> BanditStats {
        self.arms.stats(self.policy_name())
    }

    fn policy_name(&self) -> &'static str {
        "epsilon-greedy"
    }
}
