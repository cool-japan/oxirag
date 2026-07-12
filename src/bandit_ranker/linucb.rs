//! [`LinUcbRanker`] — `LinUCB` with disjoint arms (Li, Chu, Langford & Schapire,
//! 2010, "A Contextual-Bandit Approach to Personalized News Article
//! Recommendation").
//!
//! # The algorithm in one line
//!
//! Score each arm by an **optimistic** estimate of its reward — the top of a
//! confidence interval rather than its centre — and act on the best of those:
//!
//! ```text
//! score(a | x) = theta_hat_a^T x  +  alpha * sqrt(x^T A_a^-1 x)
//!                \-------------/     \-----------------------/
//!                 exploitation:        exploration: how *wide* the
//!                 what we believe      interval is in the direction x
//! ```
//!
//! # Why optimism works ("optimism in the face of uncertainty")
//!
//! The ridge estimate `theta_hat_a` concentrates around the true `theta*_a`, and
//! standard self-normalized martingale bounds (Abbasi-Yadkori et al., 2011) give,
//! with probability at least `1 - delta`, simultaneously for all `x` and all `t`:
//!
//! ```text
//! | theta_hat_a^T x  -  theta*_a^T x |  <=  alpha * sqrt(x^T A_a^-1 x)
//! ```
//!
//! for `alpha` of order `sqrt(d log(t / delta))`. So the `LinUCB` score is, on that
//! high-probability event, an **upper bound** on the arm's true mean reward. Now
//! consider what happens when the policy picks arm `a_t` over the true best arm
//! `a*`. Because it picked `a_t`, `score(a_t) >= score(a*)`; because the bound
//! holds, `score(a*) >= mu*(x_t)`, the true best mean. Chaining:
//!
//! ```text
//! mu*(x_t) - mu(a_t, x_t)  <=  score(a_t) - mu(a_t, x_t)  <=  2 * alpha * w_t
//! ```
//!
//! where `w_t = sqrt(x_t^T A_{a_t}^-1 x_t)` is the width of the *chosen* arm in
//! the *observed* direction. That is the crux: **the per-round regret is bounded
//! by the confidence width of the arm the policy actually pulled**. A mistake is
//! only possible where the model is uncertain — and pulling an arm is exactly
//! what *reduces* its uncertainty in that direction, because the pull adds
//! `x x^T` to `A_a`.
//!
//! # Why the regret is *sublinear*
//!
//! The widths cannot stay large: the elliptical-potential (a.k.a.
//! determinant-trace) lemma bounds `sum_{t<=T} min(1, w_t^2)` by
//! `O(d log T)`, because each pull increases `det(A_a)` and the determinant
//! cannot grow forever without the widths collapsing. Cauchy–Schwarz then turns
//! the sum of widths into
//!
//! ```text
//! Regret(T)  <=  2 * alpha * sum_t w_t  <=  2 * alpha * sqrt(T * sum_t w_t^2)
//!            =   O( d * sqrt(T) * log T )
//! ```
//!
//! which is `o(T)`. **This is the whole point.** Average regret `Regret(T) / T`
//! goes to zero, so the policy's per-round payoff converges to the *oracle's*.
//! It is also exactly what the module's headline regret test measures: if
//! average regret over successive windows does not fall, the algorithm is not
//! learning, and no amount of loosening the assertion would make it so.
//!
//! Contrast a constant-ε ε-greedy, which explores at a fixed rate forever and
//! therefore pays `Theta(epsilon * gap * T)` — *linear* regret. `LinUCB`'s
//! exploration is self-annealing: the bonus `alpha * sqrt(x^T A^-1 x)` shrinks on
//! its own, at exactly the rate the evidence accumulates, with no schedule to
//! tune.
//!
//! # The `alpha = 0` degenerate case
//!
//! Setting `alpha = 0` deletes the exploration term and leaves pure greedy
//! exploitation. This is deliberately *allowed*, because it is the most
//! instructive baseline the module has: greedy can lock permanently onto a
//! mediocre arm whose early rewards happened to beat an untrained rival's
//! `theta_hat^T x = 0`, never sample the rival again, and so never discover it
//! was better — linear regret, from a policy whose only sin is not exploring.
//! The module's tests construct precisely that trap and check `LinUCB` escapes it.

use super::model::{BanditArmSet, rank_by_score_desc};
use super::types::{
    BanditArm, BanditConfig, BanditContext, BanditError, BanditRankedArm, BanditRanker,
    BanditRanking, BanditResult, BanditStats,
};

/// `LinUCB` with disjoint arms: optimistic linear scoring with an incrementally
/// maintained inverse.
///
/// See the [module documentation](self) for the algorithm, the optimism
/// argument, and the sublinear-regret proof sketch.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "bandit-ranker")]
/// # {
/// use oxirag::bandit_ranker::{BanditConfig, BanditContext, BanditRanker, LinUcbRanker};
///
/// let config = BanditConfig::with_dimension(2).expect("dimension 2 is valid").set_alpha(1.0);
/// let mut ranker = LinUcbRanker::from_ids(config, ["dense", "bm25"])
///     .expect("two distinct arm ids");
///
/// let context = BanditContext::new(vec![1.0, 0.0]).expect("finite features");
///
/// // Untrained: both arms score 0 + the same bonus, so the ranking is the
/// // registration order — but every arm is ranked, and every one is explorable.
/// let ranking = ranker.select(&context).expect("arms exist");
/// assert_eq!(ranking.len(), 2);
///
/// // Teach it that "dense" pays off on this context.
/// for _ in 0..50 {
///     ranker.update("dense", &context, 1.0).expect("known arm");
///     ranker.update("bm25", &context, 0.0).expect("known arm");
/// }
/// let ranking = ranker.select(&context).expect("arms exist");
/// assert_eq!(ranking.top_arm_id(), Some("dense"));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct LinUcbRanker {
    /// The per-arm ridge models.
    arms: BanditArmSet,
    /// The configuration this ranker was built with.
    config: BanditConfig,
}

impl LinUcbRanker {
    /// Build a `LinUCB` ranker over `arms`.
    ///
    /// The arm set may be **empty** at construction — arms can be registered
    /// later with [`LinUcbRanker::add_arm`], which is what a live deployment does
    /// when a new retrieval strategy ships. [`BanditRanker::select`] is what
    /// refuses to operate on an empty arm set, because *that* is the point at
    /// which "which action?" becomes an unanswerable question.
    ///
    /// # Errors
    ///
    /// * [`BanditError::InvalidConfig`] if `config` is out of range.
    /// * [`BanditError::DuplicateArm`] if two arms share an id.
    pub fn new(
        config: BanditConfig,
        arms: impl IntoIterator<Item = BanditArm>,
    ) -> BanditResult<Self> {
        config.validate()?;
        let arm_set = BanditArmSet::new(arms, config.dimension, config.ridge_lambda)?;
        Ok(Self {
            arms: arm_set,
            config,
        })
    }

    /// Build a `LinUCB` ranker over bare arm ids.
    ///
    /// # Errors
    ///
    /// As [`LinUcbRanker::new`].
    pub fn from_ids(
        config: BanditConfig,
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> BanditResult<Self> {
        Self::new(config, ids.into_iter().map(BanditArm::new))
    }

    /// Register a new arm mid-flight. It starts untrained, hence with the widest
    /// possible confidence bonus — so `LinUCB` will go and try it.
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

    /// The exploration bonus `alpha * sqrt(x^T A_a^-1 x)` for one arm — the part
    /// of the score that is *not* belief.
    ///
    /// Exposed because it is the single most diagnostic number a `LinUCB`
    /// deployment can look at: it should be large for fresh arms and for unusual
    /// contexts, and it should decay toward zero, per arm and per direction, as
    /// evidence accumulates. A bonus that never decays means the contexts are not
    /// spanning the feature space; a bonus that collapses immediately means
    /// `alpha` is too small and the ranker is effectively greedy.
    ///
    /// # Errors
    ///
    /// * [`BanditError::UnknownArm`] if `arm_id` is not registered.
    /// * [`BanditError::DimensionMismatch`] on a context of the wrong dimension.
    /// * [`BanditError::Linalg`] if the quadratic form is non-finite.
    pub fn exploration_bonus(&self, arm_id: &str, context: &BanditContext) -> BanditResult<f64> {
        context.require_dimension(self.config.dimension)?;
        let model = self
            .arms
            .get(arm_id)
            .ok_or_else(|| BanditError::UnknownArm {
                arm_id: arm_id.to_string(),
            })?;
        Ok(self.config.alpha * model.confidence_width(context.features())?)
    }
}

impl BanditRanker for LinUcbRanker {
    fn select(&mut self, context: &BanditContext) -> BanditResult<BanditRanking> {
        if self.arms.is_empty() {
            return Err(BanditError::NoArms);
        }
        context.require_dimension(self.config.dimension)?;
        let x = context.features();

        let mut scored = Vec::with_capacity(self.arms.len());
        for model in self.arms.models() {
            let mean_estimate = model.mean_estimate(x)?;
            let width = model.confidence_width(x)?;
            scored.push(BanditRankedArm {
                arm_id: model.arm_id().to_string(),
                score: mean_estimate + self.config.alpha * width,
                mean_estimate,
                rank: 0,
            });
        }
        rank_by_score_desc(&mut scored);
        Ok(BanditRanking {
            ranked: scored,
            // LinUCB never takes a separate "exploration branch": its optimism is
            // fused into the score itself. See `BanditRanking::explored`.
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
        "linucb"
    }
}
