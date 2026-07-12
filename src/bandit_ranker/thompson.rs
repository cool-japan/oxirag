//! [`ThompsonSamplingRanker`] — linear-Gaussian Thompson sampling (Thompson,
//! 1933; Agrawal & Goyal, 2013, "Thompson Sampling for Contextual Bandits with
//! Linear Payoffs").
//!
//! # The idea: act *as if* a random draw from your beliefs were the truth
//!
//! `LinUCB` explores by being *optimistic*. Thompson sampling explores by being
//! *honest about its own uncertainty and then committing to a guess*. On each
//! round it draws one plausible parameter vector from each arm's posterior,
//! pretends that draw is the truth, and acts greedily on it:
//!
//! ```text
//! theta_tilde_a  ~  N( theta_hat_a ,  v^2 * A_a^-1 )        (one draw per arm)
//! score(a | x)   =  theta_tilde_a^T x
//! ```
//!
//! # Why this is exploration — and exactly the right amount of it
//!
//! An arm is chosen on a round precisely when its *sampled* score is the highest.
//! By construction the sampled score for arm `a` in context `x` is itself normal:
//!
//! ```text
//! theta_tilde_a^T x  ~  N( theta_hat_a^T x ,  v^2 * (x^T A_a^-1 x) )
//! ```
//!
//! — mean equal to the exploitation estimate, and **standard deviation
//! `v * sqrt(x^T A_a^-1 x)`, which is `v / alpha` times `LinUCB`'s exploration
//! bonus**. The two algorithms are reading the *same* uncertainty geometry out of
//! the *same* matrix `A^-1`; `LinUCB` adds a fixed multiple of it to the score,
//! while Thompson sampling adds a *random* multiple, drawn from a standard normal.
//!
//! The consequence is the property that makes Thompson sampling work: **an arm is
//! played with probability equal to the posterior probability that it is
//! optimal.** A barely-tried arm has a fat posterior, so its sampled score
//! frequently overshoots the leader's and it gets pulled. A thoroughly-tried arm
//! has a needle-thin posterior; if it is genuinely worse, it essentially never
//! overshoots again, and the policy stops paying for it. Exploration therefore
//! anneals *automatically*, in proportion to the remaining doubt, with no
//! schedule and no `alpha` — and the resulting regret is `O(d * sqrt(T) * log T)`,
//! the same order as `LinUCB`'s.
//!
//! Note that `A^-1` shrinks in exactly the directions that have been *observed*,
//! since each pull adds `x x^T` to `A`. So the posterior can stay wide in an
//! unexplored *direction* even for a heavily-pulled arm: the uncertainty being
//! sampled is directional, not merely a per-arm scalar. That is the difference
//! between a *contextual* bandit and `K` independent ones.
//!
//! # Drawing the sample: why Cholesky
//!
//! Sampling `N(mu, Sigma)` in `d` dimensions reduces to sampling `d` independent
//! standard normals and *correlating* them. If `L L^T = Sigma`, then for
//! `z ~ N(0, I)`:
//!
//! ```text
//! Cov( mu + L z )  =  L * Cov(z) * L^T  =  L I L^T  =  L L^T  =  Sigma
//! ```
//!
//! so `theta_tilde = theta_hat + L z` has exactly the desired distribution (and
//! is normal, being an affine map of a normal). The Cholesky factor is precisely
//! such an `L`, it is the cheapest one to compute (`O(d^3 / 3)`, no eigenvalues),
//! and its triangularity makes `L z` an `O(d^2 / 2)` back-substitution rather than
//! a full matrix–vector product.
//!
//! The covariance factored here is `v^2 * A^-1`. Rather than factoring
//! `v^2 * A^-1` directly, this implementation factors `A^-1` and scales the
//! resulting factor by `v`: since `(v L)(v L)^T = v^2 L L^T = v^2 A^-1`, the two
//! are identical in exact arithmetic, but scaling the *factor* keeps the matrix
//! handed to the factorization independent of `v`, so a large or small `v` cannot
//! itself push the input toward the edge of the positive-definite cone.
//!
//! `A^-1` is positive definite by the induction in [`super::linalg`], so the
//! factorization should always succeed. "Should" is doing real work in that
//! sentence, though: after tens of thousands of rank-1 updates the smallest
//! eigenvalue of `A^-1` can be genuinely tiny — a direction observed thousands of
//! times *has* an almost-zero posterior variance — and at that point rounding
//! error is the same size as the eigenvalue, so the computed matrix can land a
//! few ulps outside the cone. [`super::linalg::cholesky_with_jitter`] handles
//! that by adding a scale-aware ridge and reporting how large a ridge it needed;
//! [`ThompsonSamplingRanker::jitter_events`] counts how often that path has
//! engaged, so an operator can *see* whether their bandit is numerically healthy
//! instead of guessing.

use super::linalg::{cholesky_with_jitter, lower_triangular_mat_vec, scale_matrix};
use super::model::{BanditArmSet, rank_by_score_desc};
use super::rng::SplitMix64Rng;
use super::types::{
    BanditArm, BanditConfig, BanditContext, BanditError, BanditRankedArm, BanditRanker,
    BanditRanking, BanditResult, BanditStats,
};

/// Linear-Gaussian Thompson sampling: draw a parameter vector per arm from its
/// posterior, then rank by the sampled scores.
///
/// See the [module documentation](self) for the posterior, the Cholesky draw, and
/// why sampling yields exploration proportional to uncertainty.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "bandit-ranker")]
/// # {
/// use oxirag::bandit_ranker::{
///     BanditConfig, BanditContext, BanditRanker, ThompsonSamplingRanker,
/// };
///
/// let config = BanditConfig::with_dimension(2)
///     .expect("dimension 2 is valid")
///     .set_exploration_variance(0.2)
///     .set_seed(7);
/// let mut ranker = ThompsonSamplingRanker::from_ids(config, ["dense", "bm25"])
///     .expect("two distinct arm ids");
///
/// let context = BanditContext::new(vec![1.0, 0.0]).expect("finite features");
/// for _ in 0..200 {
///     let ranking = ranker.select(&context).expect("arms exist");
///     let arm = ranking.top_arm_id().expect("non-empty ranking").to_string();
///     let reward = if arm == "dense" { 1.0 } else { 0.0 };
///     ranker.update(&arm, &context, reward).expect("known arm");
/// }
///
/// // The posterior has concentrated: "dense" now wins essentially every draw.
/// let stats = ranker.stats();
/// assert_eq!(stats.most_pulled_arm(), Some("dense"));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ThompsonSamplingRanker {
    /// The per-arm ridge models — identical machinery to `LinUCB`'s; only the
    /// scoring rule differs.
    arms: BanditArmSet,
    /// The configuration this ranker was built with.
    config: BanditConfig,
    /// The deterministic RNG the posterior draws come from.
    rng: SplitMix64Rng,
    /// How many times the Cholesky jitter path has engaged. `0` on a healthy
    /// bandit; a rising count means the maintained inverses are drifting out of
    /// the positive-definite cone and the model deserves a look.
    jitter_events: u64,
}

impl ThompsonSamplingRanker {
    /// Build a Thompson-sampling ranker over `arms`, seeded from
    /// [`BanditConfig::seed`].
    ///
    /// # Errors
    ///
    /// * [`BanditError::InvalidConfig`] if `config` is out of range (notably:
    ///   `exploration_variance` must be strictly positive — a zero-variance
    ///   posterior is a point mass, which would silently turn this into a greedy
    ///   policy).
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
            jitter_events: 0,
        })
    }

    /// Build a Thompson-sampling ranker over bare arm ids.
    ///
    /// # Errors
    ///
    /// As [`ThompsonSamplingRanker::new`].
    pub fn from_ids(
        config: BanditConfig,
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> BanditResult<Self> {
        Self::new(config, ids.into_iter().map(BanditArm::new))
    }

    /// Register a new arm mid-flight. Its posterior starts at the prior — the
    /// widest it will ever be — so it will win draws often until it has earned
    /// its narrowness.
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

    /// How many times the Cholesky jitter path has engaged since construction.
    ///
    /// Expected to stay at `0`. A non-zero value is not an error — the jitter is
    /// a *correct* repair, and the module documents why — but it is a signal that
    /// some arm's posterior has collapsed to numerical zero in some direction.
    #[must_use]
    pub fn jitter_events(&self) -> u64 {
        self.jitter_events
    }

    /// Draw one parameter vector `theta_tilde ~ N(theta_hat, v^2 A^-1)` for the
    /// arm named `arm_id`.
    ///
    /// Advances the RNG. Exposed because it is the one operation whose
    /// distribution the module's tests need to check *directly* (a sampler whose
    /// covariance is wrong would still produce a plausible-looking bandit, just a
    /// badly-calibrated one — so it is verified against the analytic covariance
    /// rather than inferred from downstream behaviour).
    ///
    /// # Errors
    ///
    /// * [`BanditError::UnknownArm`] if `arm_id` is not registered.
    /// * [`BanditError::Linalg`] if the posterior covariance cannot be factored
    ///   even after the full jitter escalation.
    pub fn sample_theta(&mut self, arm_id: &str) -> BanditResult<Vec<f64>> {
        let dimension = self.config.dimension;
        let posterior_scale = self.config.exploration_variance;

        // Factor A^-1 (not v^2 A^-1) and scale the *factor* by v. Identical in
        // exact arithmetic, since (v L)(v L)^T = v^2 L L^T; keeps the matrix
        // handed to the factorization independent of v.
        let (lower, jitter, theta_hat) = {
            let model = self
                .arms
                .get(arm_id)
                .ok_or_else(|| BanditError::UnknownArm {
                    arm_id: arm_id.to_string(),
                })?;
            let (lower, jitter) = cholesky_with_jitter(model.design_inverse(), dimension)?;
            (lower, jitter, model.theta().to_vec())
        };
        let scaled_lower = scale_matrix(&lower, posterior_scale);

        let z: Vec<f64> = (0..dimension)
            .map(|_| self.rng.next_standard_normal())
            .collect();
        let perturbation = lower_triangular_mat_vec(&scaled_lower, &z, dimension)?;

        if jitter > 0.0 {
            self.jitter_events += 1;
        }
        Ok(theta_hat
            .iter()
            .zip(&perturbation)
            .map(|(mean, delta)| mean + delta)
            .collect())
    }
}

impl BanditRanker for ThompsonSamplingRanker {
    fn select(&mut self, context: &BanditContext) -> BanditResult<BanditRanking> {
        if self.arms.is_empty() {
            return Err(BanditError::NoArms);
        }
        context.require_dimension(self.config.dimension)?;

        // Collect the ids first: `sample_theta` needs `&mut self` (it advances
        // the RNG), so we cannot hold a borrow of `self.arms` across the loop.
        let arm_ids: Vec<String> = self
            .arms
            .arm_ids()
            .into_iter()
            .map(ToString::to_string)
            .collect();

        let mut scored = Vec::with_capacity(arm_ids.len());
        for arm_id in arm_ids {
            let sampled_theta = self.sample_theta(&arm_id)?;
            let model = self
                .arms
                .get(&arm_id)
                .ok_or_else(|| BanditError::UnknownArm {
                    arm_id: arm_id.clone(),
                })?;
            let mean_estimate = model.mean_estimate(context.features())?;
            let score: f64 = sampled_theta
                .iter()
                .zip(context.features())
                .map(|(theta, x)| theta * x)
                .sum();
            scored.push(BanditRankedArm {
                arm_id,
                score,
                mean_estimate,
                rank: 0,
            });
        }
        rank_by_score_desc(&mut scored);
        Ok(BanditRanking {
            ranked: scored,
            // As with LinUCB: the randomness *is* the score, not a separate
            // branch. Every Thompson round explores and exploits at once.
            explored: false,
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
        "thompson-sampling"
    }
}
