//! Public data types and the [`BanditRanker`] trait shared by all three
//! policies.
//!
//! An **arm** ([`BanditArm`]) is a re-usable *action* the ranker can take —
//! concretely, in a RAG system, a retrieval strategy (dense / BM25 / hybrid), a
//! reranker, a prompt template, or a candidate document to place in the top
//! slot. A **context** ([`BanditContext`]) is the feature vector describing the
//! *current* request (query embedding statistics, query length, user segment,
//! …). A **reward** is whatever the deployment can actually observe after the
//! fact: a click, a thumbs-up, a downstream answer-quality score.
//!
//! The contract every policy in this module implements is
//! [`BanditRanker`]: given a context, produce a *full ranking* of the arms
//! ([`BanditRanking`]); later, when a reward for one of them is observed, fold
//! it back in with [`BanditRanker::update`].

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::linalg::LinalgError;

// ── BanditError ──────────────────────────────────────────────────────────────

/// Errors produced by the `bandit_ranker` module.
#[derive(Debug, Error)]
pub enum BanditError {
    /// A ranker was asked to [`BanditRanker::select`] with no arms registered.
    /// There is no such thing as "the best of nothing", so this is an error
    /// rather than an empty ranking: a caller that silently received an empty
    /// list would have no action to take and no reward to report.
    #[error("bandit has no arms to rank")]
    NoArms,
    /// [`BanditRanker::update`] named an arm the ranker does not know about.
    #[error("unknown bandit arm: {arm_id}")]
    UnknownArm {
        /// The arm id that was not registered.
        arm_id: String,
    },
    /// Two [`BanditArm`]s with the same id were supplied to one ranker, which
    /// would make the arm set ambiguous (which of the two do you update?).
    #[error("duplicate bandit arm id: {arm_id}")]
    DuplicateArm {
        /// The id that appeared more than once.
        arm_id: String,
    },
    /// A [`BanditContext`]'s feature count did not match the ranker's configured
    /// [`BanditConfig::dimension`].
    #[error("context dimension mismatch: ranker expects {expected}, context has {actual}")]
    DimensionMismatch {
        /// The dimension the ranker was configured with.
        expected: usize,
        /// The dimension of the offending context.
        actual: usize,
    },
    /// A configuration value was outside its admissible range.
    #[error("invalid bandit configuration: {reason}")]
    InvalidConfig {
        /// A human-readable explanation of what was invalid.
        reason: String,
    },
    /// A context feature, or an observed reward, was `NaN` or infinite.
    ///
    /// Rejected up front and loudly: a single non-finite value folded into an
    /// arm's `A^-1` or `b` would contaminate that arm's state *permanently*,
    /// and every subsequent score for that arm would be `NaN` — a failure that
    /// is nearly impossible to diagnose after the fact.
    #[error("non-finite {what}: {value}")]
    NonFinite {
        /// Which quantity was non-finite.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// A numerical kernel failed. In practice this means the maintained inverse
    /// has been corrupted (see [`LinalgError`]).
    #[error("bandit linear algebra failure: {0}")]
    Linalg(#[from] LinalgError),
    /// [`super::BanditOffPolicyEvaluator`] finished a replay without accepting a
    /// single logged event, so there is nothing to average and no estimate to
    /// report.
    #[error(
        "off-policy replay accepted 0 of {total} logged events; \
         no unbiased estimate can be formed"
    )]
    NoAcceptedEvents {
        /// How many logged events were examined.
        total: usize,
    },
}

/// Convenience alias for this module's fallible return type.
pub type BanditResult<T> = Result<T, BanditError>;

// ── BanditArm ────────────────────────────────────────────────────────────────

/// One action the bandit can take, identified by a stable string id.
///
/// In a RAG deployment an arm is typically a *retrieval or generation
/// configuration* — "dense-only", "hybrid-rrf", "hyde-then-rerank" — that the
/// system can choose per query and whose payoff it can later observe. It is
/// deliberately just an id plus an optional human label: everything the bandit
/// learns about an arm lives in the ranker's per-arm linear model, not here, so
/// arms stay cheap to clone and trivially serializable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BanditArm {
    /// Stable identifier, unique within one ranker.
    pub id: String,
    /// Optional human-readable description, carried through to
    /// [`BanditArmStats`] for reporting.
    pub label: Option<String>,
}

impl BanditArm {
    /// Create an arm with the given id and no label.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
        }
    }

    /// Attach a human-readable label, returning `self` for chaining.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ── BanditContext ────────────────────────────────────────────────────────────

/// The feature vector `x` describing the request currently being served.
///
/// Validated on construction: every feature must be finite. A `NaN` slipping
/// into a context is the single most destructive input this module can receive
/// (it would be absorbed into an arm's `A^-1` and `b` and never wash out), so it
/// is rejected at the boundary rather than deep inside a matrix kernel.
///
/// A **zero** context is explicitly *legal*, not an error: it is the honest
/// encoding of "no features available for this request". Every downstream
/// formula degrades gracefully — `theta^T 0 = 0`, the exploration bonus
/// `sqrt(0^T A^-1 0) = 0`, and the Sherman–Morrison update `A + 0 0^T = A` is a
/// no-op with denominator exactly `1` — so a zero context produces a tied
/// ranking broken by the arms' stable insertion order, and learns nothing. That
/// is the correct behaviour, and it is tested.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditContext {
    /// The features, `d` of them.
    features: Vec<f64>,
    /// Optional provenance tag (a query id, a request id) carried for logging;
    /// never read by any policy.
    query_id: Option<String>,
}

impl BanditContext {
    /// Build a context from a feature vector.
    ///
    /// # Errors
    ///
    /// [`BanditError::NonFinite`] if any feature is `NaN` or infinite.
    pub fn new(features: impl Into<Vec<f64>>) -> BanditResult<Self> {
        let features = features.into();
        for &value in &features {
            if !value.is_finite() {
                return Err(BanditError::NonFinite {
                    what: "context feature",
                    value,
                });
            }
        }
        Ok(Self {
            features,
            query_id: None,
        })
    }

    /// Attach a provenance tag, returning `self` for chaining.
    #[must_use]
    pub fn with_query_id(mut self, query_id: impl Into<String>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    /// The feature vector.
    #[must_use]
    pub fn features(&self) -> &[f64] {
        &self.features
    }

    /// The number of features, `d`.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.features.len()
    }

    /// The provenance tag, if one was attached.
    #[must_use]
    pub fn query_id(&self) -> Option<&str> {
        self.query_id.as_deref()
    }

    /// Check this context against a ranker's configured dimension.
    ///
    /// # Errors
    ///
    /// [`BanditError::DimensionMismatch`] if the lengths differ.
    pub fn require_dimension(&self, expected: usize) -> BanditResult<()> {
        if self.features.len() == expected {
            Ok(())
        } else {
            Err(BanditError::DimensionMismatch {
                expected,
                actual: self.features.len(),
            })
        }
    }
}

// ── BanditConfig ─────────────────────────────────────────────────────────────

/// Configuration shared by all three policies.
///
/// Each policy reads the subset of fields that is meaningful to it — `LinUCB` uses
/// `alpha`, Thompson sampling uses `exploration_variance`, ε-greedy uses
/// `epsilon` / `decay_epsilon` — while `dimension`, `ridge_lambda` and `seed`
/// govern the linear model and the RNG that all of them share. Keeping one
/// config type means the three policies are drop-in substitutable in a
/// deployment, which is the whole point of hiding them behind [`BanditRanker`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditConfig {
    /// The context dimension `d`. Must be at least `1`.
    pub dimension: usize,
    /// `LinUCB`'s exploration coefficient `alpha`, the multiplier on the
    /// confidence-width term `sqrt(x^T A^-1 x)`.
    ///
    /// The theory (Li et al., 2010) sets `alpha = 1 + sqrt(ln(2 / delta) / 2)`
    /// for a `1 - delta` confidence bound, which for the usual `delta` lands
    /// somewhere in `[1, 2.5]`; in practice it is tuned, and *smaller* values
    /// than theory suggests usually win, because the theoretical constant is
    /// worst-case over adversarial contexts. `alpha = 0` degenerates `LinUCB` into
    /// pure greedy exploitation — which is a legal (and, in the tests, a
    /// deliberately *instructive*) configuration.
    pub alpha: f64,
    /// Thompson sampling's posterior scale `v`: the posterior is
    /// `N(theta_hat, v^2 A^-1)`.
    ///
    /// Agrawal & Goyal (2013) derive `v = R * sqrt(24 / eps * d * ln(1 / delta))`
    /// for `R`-sub-Gaussian rewards; as with `alpha`, deployments tune it, and a
    /// smaller `v` means a tighter posterior and less exploration. Must be
    /// strictly positive — `v = 0` would collapse the posterior to a point mass
    /// and turn Thompson sampling into greedy.
    pub exploration_variance: f64,
    /// ε-greedy's exploration probability at `t = 1`, in `[0, 1]`.
    pub epsilon: f64,
    /// Whether ε decays as `epsilon_t = epsilon / sqrt(t)`.
    ///
    /// A *constant* ε has linear regret: it keeps paying `epsilon * gap` forever,
    /// so `regret(T) ~ epsilon * gap * T`. The decaying schedule is what makes
    /// ε-greedy a *learning* algorithm — see [`super::EpsilonGreedyRanker`].
    pub decay_epsilon: bool,
    /// The ridge `lambda` in `A_a = lambda * I`, which is both the regularizer of
    /// the per-arm ridge regression and the prior precision of the Bayesian view.
    /// Must be strictly positive: it is what makes `A` positive *definite* rather
    /// than merely semi-definite from round one, and hence what makes `A^-1` exist
    /// before a single reward has been seen.
    pub ridge_lambda: f64,
    /// Seed for the module's [`super::SplitMix64Rng`]. Same seed, same contexts,
    /// same rewards ⇒ byte-identical trajectory.
    pub seed: u64,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            dimension: 8,
            alpha: 1.0,
            exploration_variance: 0.25,
            epsilon: 0.1,
            decay_epsilon: true,
            ridge_lambda: 1.0,
            seed: 0,
        }
    }
}

impl BanditConfig {
    /// A configuration for contexts of dimension `dimension`, everything else
    /// defaulted.
    ///
    /// # Errors
    ///
    /// [`BanditError::InvalidConfig`] if `dimension` is zero.
    pub fn with_dimension(dimension: usize) -> BanditResult<Self> {
        let config = Self {
            dimension,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Set `LinUCB`'s `alpha`, returning `self` for chaining.
    #[must_use]
    pub fn set_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set Thompson sampling's posterior scale `v`, returning `self` for
    /// chaining.
    #[must_use]
    pub fn set_exploration_variance(mut self, exploration_variance: f64) -> Self {
        self.exploration_variance = exploration_variance;
        self
    }

    /// Set ε-greedy's exploration probability, returning `self` for chaining.
    #[must_use]
    pub fn set_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Enable or disable the `1 / sqrt(t)` ε decay, returning `self` for
    /// chaining.
    #[must_use]
    pub fn set_decay_epsilon(mut self, decay_epsilon: bool) -> Self {
        self.decay_epsilon = decay_epsilon;
        self
    }

    /// Set the ridge `lambda`, returning `self` for chaining.
    #[must_use]
    pub fn set_ridge_lambda(mut self, ridge_lambda: f64) -> Self {
        self.ridge_lambda = ridge_lambda;
        self
    }

    /// Set the RNG seed, returning `self` for chaining.
    #[must_use]
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Check every field is in range.
    ///
    /// # Errors
    ///
    /// [`BanditError::InvalidConfig`] naming the first offending field.
    pub fn validate(&self) -> BanditResult<()> {
        if self.dimension == 0 {
            return Err(BanditError::InvalidConfig {
                reason: "dimension must be at least 1".to_string(),
            });
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(BanditError::InvalidConfig {
                reason: format!("alpha must be finite and non-negative, got {}", self.alpha),
            });
        }
        if !self.exploration_variance.is_finite() || self.exploration_variance <= 0.0 {
            return Err(BanditError::InvalidConfig {
                reason: format!(
                    "exploration_variance must be finite and strictly positive, got {}",
                    self.exploration_variance
                ),
            });
        }
        if !self.epsilon.is_finite() || !(0.0..=1.0).contains(&self.epsilon) {
            return Err(BanditError::InvalidConfig {
                reason: format!("epsilon must lie in [0, 1], got {}", self.epsilon),
            });
        }
        if !self.ridge_lambda.is_finite() || self.ridge_lambda <= 0.0 {
            return Err(BanditError::InvalidConfig {
                reason: format!(
                    "ridge_lambda must be finite and strictly positive, got {}",
                    self.ridge_lambda
                ),
            });
        }
        Ok(())
    }
}

// ── BanditRankedArm / BanditRanking ──────────────────────────────────────────

/// One arm's position in a [`BanditRanking`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditRankedArm {
    /// The arm's id.
    pub arm_id: String,
    /// The policy's score for this arm on this context.
    ///
    /// Its meaning depends on the policy — `LinUCB` reports the upper confidence
    /// bound `theta^T x + alpha * sqrt(x^T A^-1 x)`, Thompson sampling reports
    /// the *sampled* score `theta_tilde^T x`, ε-greedy reports the plain mean
    /// estimate `theta_hat^T x` — but in every case a *higher* score means the
    /// policy prefers the arm more, and the ranking is sorted by it (except in
    /// an ε-greedy exploration round; see [`BanditRanking::explored`]).
    pub score: f64,
    /// The policy's *exploitation* component `theta_hat^T x` — the current best
    /// point estimate of this arm's mean reward on this context, with no
    /// exploration term of any kind mixed in.
    ///
    /// Reported alongside `score` so a caller can *see* how much of a decision
    /// was driven by exploration rather than by belief. For ε-greedy in an
    /// exploit round this equals `score`; for `LinUCB` the difference
    /// `score - mean_estimate` is exactly `alpha * sqrt(x^T A^-1 x)`.
    pub mean_estimate: f64,
    /// Zero-based position in the ranking; `0` is the arm the policy would act
    /// on.
    pub rank: usize,
}

/// A full ranking of every arm for one context — the output of
/// [`BanditRanker::select`].
///
/// This module is a *ranker*, not merely an arm-picker: a RAG system that has to
/// fill several result slots, or that wants a fallback if its first choice
/// errors, needs the whole ordering, not just the argmax. `ranked[0]` is
/// nevertheless exactly the arm a classical bandit would pull, and it is what
/// [`super::BanditOffPolicyEvaluator`] matches against a logged action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditRanking {
    /// Every arm, best first.
    pub ranked: Vec<BanditRankedArm>,
    /// Whether this decision was an **exploration** round.
    ///
    /// Only ε-greedy ever sets this: it is `true` exactly when the coin came up
    /// "explore" and the ranking is a uniformly random permutation rather than
    /// the greedy order. `LinUCB` and Thompson sampling always report `false` —
    /// not because they do not explore (they very much do), but because their
    /// exploration is *baked into the score itself* rather than being a separate
    /// randomized branch. There is no round in which `LinUCB` "is exploring" as
    /// opposed to "is exploiting"; it is always doing both at once, which is
    /// precisely why it beats ε-greedy.
    pub explored: bool,
}

impl BanditRanking {
    /// The top-ranked arm — the one a caller should act on.
    ///
    /// `None` only if the ranking is empty, which [`BanditRanker::select`] never
    /// returns (it errors with [`BanditError::NoArms`] instead).
    #[must_use]
    pub fn top(&self) -> Option<&BanditRankedArm> {
        self.ranked.first()
    }

    /// The id of the top-ranked arm.
    #[must_use]
    pub fn top_arm_id(&self) -> Option<&str> {
        self.ranked.first().map(|arm| arm.arm_id.as_str())
    }

    /// The arm ids in ranked order.
    #[must_use]
    pub fn arm_ids(&self) -> Vec<&str> {
        self.ranked.iter().map(|arm| arm.arm_id.as_str()).collect()
    }

    /// How many arms were ranked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranked.len()
    }

    /// Whether the ranking is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranked.is_empty()
    }
}

// ── BanditStats ──────────────────────────────────────────────────────────────

/// Per-arm summary of what a ranker has learned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditArmStats {
    /// The arm's id.
    pub arm_id: String,
    /// The arm's human-readable label, if it was given one.
    pub label: Option<String>,
    /// How many rewards have been folded into this arm.
    pub pulls: u64,
    /// The sum of those rewards.
    pub total_reward: f64,
    /// Their mean, or `0.0` for an arm that has never been pulled.
    pub mean_reward: f64,
    /// The arm's current ridge-regression coefficients
    /// `theta_hat = A^-1 b`.
    pub theta: Vec<f64>,
}

/// A snapshot of a ranker's state — useful for dashboards, for debugging a
/// deployment that has locked onto one arm, and for asserting in tests that
/// exploration actually happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditStats {
    /// Which policy produced this snapshot.
    pub policy: String,
    /// The context dimension `d`.
    pub dimension: usize,
    /// Total rewards folded in across all arms.
    pub total_pulls: u64,
    /// Sum of every reward ever observed.
    pub total_reward: f64,
    /// Mean reward per pull, or `0.0` before the first pull.
    pub mean_reward: f64,
    /// Per-arm detail, in the arms' registration order.
    pub arms: Vec<BanditArmStats>,
}

impl BanditStats {
    /// The id of the arm pulled most often — the arm the policy has, in
    /// practice, converged on. `None` if there are no arms.
    ///
    /// Ties break to the **earliest-registered** arm, matching the tie-break the
    /// rankers themselves use. (Note this is deliberately *not*
    /// [`Iterator::max_by_key`], which returns the **last** maximal element and
    /// would therefore disagree with the ranking on an all-zero-pull bandit.)
    #[must_use]
    pub fn most_pulled_arm(&self) -> Option<&str> {
        let mut best: Option<&BanditArmStats> = None;
        for arm in &self.arms {
            if best.is_none_or(|current| arm.pulls > current.pulls) {
                best = Some(arm);
            }
        }
        best.map(|arm| arm.arm_id.as_str())
    }

    /// The fraction of pulls that went to the *most*-pulled arm, in `[0, 1]`.
    ///
    /// A useful one-number diagnostic: near `1 / num_arms` means the policy is
    /// still exploring broadly; near `1.0` means it has committed. A policy that
    /// commits *immediately* is the pathology the module's "deceptive arm" test
    /// exists to catch.
    #[must_use]
    pub fn concentration(&self) -> f64 {
        if self.total_pulls == 0 {
            return 0.0;
        }
        let most = self.arms.iter().map(|arm| arm.pulls).max().unwrap_or(0);
        #[allow(clippy::cast_precision_loss)] // Pull counts far below 2^53.
        {
            most as f64 / self.total_pulls as f64
        }
    }
}

// ── BanditRanker ─────────────────────────────────────────────────────────────

/// The contract every policy in this module satisfies: rank the arms for a
/// context, then learn from an observed reward.
///
/// Deliberately object-safe (`&mut self` methods, no generics, no `Self`-typed
/// returns) so that [`super::BanditOffPolicyEvaluator`] can replay a logged
/// stream against `&mut dyn BanditRanker` — i.e. so that a *single* off-policy
/// evaluator works for all three policies, and for any policy a downstream crate
/// writes.
///
/// # Why `select` takes `&mut self`
///
/// Thompson sampling *draws* from a posterior and ε-greedy *flips a coin*; both
/// advance an RNG, and both must do so as part of choosing. A `&self` signature
/// would force interior mutability (and, with it, either a lock or an
/// unsound `Cell` across threads) to hide the fact that selection is a
/// state-advancing operation. It is more honest to say so in the type.
pub trait BanditRanker {
    /// Rank every arm for `context`, best first.
    ///
    /// # Errors
    ///
    /// * [`BanditError::NoArms`] if no arms are registered.
    /// * [`BanditError::DimensionMismatch`] if the context's dimension does not
    ///   match the ranker's.
    /// * [`BanditError::Linalg`] if the maintained inverse has been corrupted.
    fn select(&mut self, context: &BanditContext) -> BanditResult<BanditRanking>;

    /// Fold an observed `reward` for `arm_id` on `context` back into the model.
    ///
    /// # Errors
    ///
    /// * [`BanditError::UnknownArm`] if `arm_id` is not registered.
    /// * [`BanditError::DimensionMismatch`] on a context of the wrong dimension.
    /// * [`BanditError::NonFinite`] if `reward` is `NaN` or infinite.
    /// * [`BanditError::Linalg`] if the Sherman–Morrison update fails.
    fn update(&mut self, arm_id: &str, context: &BanditContext, reward: f64) -> BanditResult<()>;

    /// The context dimension `d` this ranker was configured for.
    fn dimension(&self) -> usize;

    /// The registered arm ids, in registration order.
    fn arm_ids(&self) -> Vec<&str>;

    /// The current coefficient estimate `theta_hat = A^-1 b` for one arm, or
    /// `None` if the arm is not registered.
    fn theta(&self, arm_id: &str) -> Option<&[f64]>;

    /// A snapshot of everything the ranker has learned.
    fn stats(&self) -> BanditStats;

    /// The policy's name, for reporting.
    fn policy_name(&self) -> &'static str;
}

/// Validate a reward value before it is folded into an arm's state.
///
/// # Errors
///
/// [`BanditError::NonFinite`] if `reward` is `NaN` or infinite.
pub(crate) fn check_reward(reward: f64) -> BanditResult<()> {
    if reward.is_finite() {
        Ok(())
    } else {
        Err(BanditError::NonFinite {
            what: "reward",
            value: reward,
        })
    }
}
