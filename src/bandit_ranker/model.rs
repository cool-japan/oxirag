//! [`BanditLinearModel`] — the per-arm ridge-regression state that all three
//! policies share, and [`BanditArmSet`], the ordered collection of them.
//!
//! Every policy in this module is a *disjoint linear* bandit: it assumes each
//! arm `a` has its own unknown coefficient vector `theta*_a` and that the
//! expected reward of pulling `a` in context `x` is `theta*_a^T x`. "Disjoint"
//! means the arms share no parameters, so each one carries its own independent
//! model — this struct. `LinUCB`, Thompson sampling and ε-greedy differ *only* in
//! how they turn that model into a score; the estimation machinery underneath is
//! literally identical, so it lives here once.
//!
//! # The model
//!
//! Arm `a` accumulates
//!
//! ```text
//! A_a = lambda * I + sum over its pulls of  x x^T        (d x d)
//! b_a =             sum over its pulls of  r * x         (d)
//! ```
//!
//! and estimates `theta_hat_a = A_a^-1 b_a`. This is exactly **ridge regression**
//! of the observed rewards on the observed contexts, with regularizer `lambda`:
//! minimizing `sum (r_i - theta^T x_i)^2 + lambda ||theta||^2` has the normal
//! equations `(lambda I + X^T X) theta = X^T y`, whose left- and right-hand sides
//! are precisely `A_a` and `b_a`.
//!
//! It is *also*, read the other way, the **posterior** of a Bayesian linear model
//! with prior `theta ~ N(0, (1 / lambda) I)` and Gaussian likelihood: the
//! posterior mean is `A^-1 b` and the posterior covariance is proportional to
//! `A^-1`. That dual reading is not a curiosity — it is what makes the *same*
//! state serve `LinUCB` (which reads `A^-1` as a confidence ellipsoid) and Thompson
//! sampling (which reads it as a covariance to sample from).
//!
//! # What is actually stored
//!
//! The inverse `A^-1` is maintained **incrementally** by
//! [`super::linalg::sherman_morrison_update`] and is *never* recomputed from
//! `A`. `A` itself is stored too, but only for observability
//! ([`BanditLinearModel::design_matrix`]) and for the module's headline test,
//! which reconstructs `A^-1` from it with an independent Gauss–Jordan inverse and
//! checks the incremental one has not drifted. Nothing in the scoring path ever
//! reads `A`.
//!
//! `theta_hat` is recomputed as `A^-1 b` after every update rather than being
//! re-derived on every score, because updates are rare (one per observed reward)
//! and scores are frequent (one per arm per request), and because `A^-1 b` is
//! `O(d^2)` either way.

use std::collections::HashMap;

use super::linalg::{
    mat_vec, quadratic_form_nonnegative, scaled_identity, sherman_morrison_update,
};
use super::types::{
    BanditArm, BanditArmStats, BanditError, BanditRankedArm, BanditResult, BanditStats,
    check_reward,
};

/// Sort scored arms best-first, then number them `0, 1, 2, ...`.
///
/// The sort is **stable** and compares with [`f64::total_cmp`], which matters
/// twice over:
///
/// * `total_cmp` gives a *total* order on `f64` — unlike `partial_cmp`, it cannot
///   return `None`, so there is no `unwrap` to write and no panic to hit. (The
///   scores here are finite by construction, but a comparator that is total by
///   type rather than by argument is a comparator that cannot rot.)
/// * *Stability* means equally-scored arms keep their registration order. That is
///   what makes an untrained ranker — every `theta_hat = 0`, hence every score
///   `0` — return a deterministic ranking instead of an arbitrary one, and it is
///   what makes the "greedy locks onto the deceptive arm" scenario reproducible
///   rather than a coin flip on the first round.
pub(crate) fn rank_by_score_desc(arms: &mut [BanditRankedArm]) {
    arms.sort_by(|left, right| right.score.total_cmp(&left.score));
    assign_ranks(arms);
}

/// Number arms `0, 1, 2, ...` in their current order, leaving that order alone.
///
/// Used by ε-greedy's exploration branch, where the order is a uniformly random
/// permutation and must *not* be re-sorted by score.
pub(crate) fn assign_ranks(arms: &mut [BanditRankedArm]) {
    for (position, arm) in arms.iter_mut().enumerate() {
        arm.rank = position;
    }
}

/// The ridge-regression state of a single arm.
#[derive(Debug, Clone)]
pub struct BanditLinearModel {
    /// The arm this model belongs to.
    arm: BanditArm,
    /// The context dimension `d`.
    dimension: usize,
    /// `A = lambda * I + sum x x^T`, row-major `d x d`.
    ///
    /// Kept for observability and for cross-checking the incremental inverse;
    /// **never** read by the scoring path.
    design: Vec<f64>,
    /// `A^-1`, row-major `d x d`, maintained incrementally by Sherman–Morrison.
    /// This is the one the policies actually use.
    design_inverse: Vec<f64>,
    /// `b = sum r * x`.
    response: Vec<f64>,
    /// `theta_hat = A^-1 b`, refreshed after every update.
    theta: Vec<f64>,
    /// How many rewards have been folded in.
    pulls: u64,
    /// The sum of those rewards.
    total_reward: f64,
}

impl BanditLinearModel {
    /// A fresh model for `arm`: `A = lambda I`, `b = 0`, `theta_hat = 0`.
    ///
    /// Note that `A^-1` is initialized directly to `(1 / lambda) I` — the
    /// inverse of the initial `A` is known in closed form, so this ranker never
    /// performs a matrix inversion, not even once at startup. That is not a
    /// micro-optimization; it is the invariant that lets the module ship without
    /// a general inverse in the production path at all.
    #[must_use]
    pub fn new(arm: BanditArm, dimension: usize, ridge_lambda: f64) -> Self {
        Self {
            arm,
            dimension,
            design: scaled_identity(dimension, ridge_lambda),
            design_inverse: scaled_identity(dimension, 1.0 / ridge_lambda),
            response: vec![0.0; dimension],
            theta: vec![0.0; dimension],
            pulls: 0,
            total_reward: 0.0,
        }
    }

    /// The arm this model belongs to.
    #[must_use]
    pub fn arm(&self) -> &BanditArm {
        &self.arm
    }

    /// The arm's id.
    #[must_use]
    pub fn arm_id(&self) -> &str {
        &self.arm.id
    }

    /// `theta_hat = A^-1 b`, the current point estimate of this arm's
    /// coefficients.
    #[must_use]
    pub fn theta(&self) -> &[f64] {
        &self.theta
    }

    /// The maintained inverse `A^-1`, row-major.
    #[must_use]
    pub fn design_inverse(&self) -> &[f64] {
        &self.design_inverse
    }

    /// The design matrix `A = lambda I + sum x x^T`, row-major.
    ///
    /// Exposed for observability and for the module's Sherman–Morrison
    /// correctness test, which inverts this independently and compares against
    /// [`BanditLinearModel::design_inverse`]. Not read by any policy.
    #[must_use]
    pub fn design_matrix(&self) -> &[f64] {
        &self.design
    }

    /// How many rewards have been folded into this arm.
    #[must_use]
    pub fn pulls(&self) -> u64 {
        self.pulls
    }

    /// The sum of the rewards folded into this arm.
    #[must_use]
    pub fn total_reward(&self) -> f64 {
        self.total_reward
    }

    /// The mean reward this arm has actually paid out, or `0.0` if it has never
    /// been pulled.
    #[must_use]
    pub fn mean_reward(&self) -> f64 {
        if self.pulls == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)] // Pull counts stay far below 2^53.
            {
                self.total_reward / self.pulls as f64
            }
        }
    }

    /// The **exploitation** score `theta_hat^T x`: the model's best point
    /// estimate of the reward this arm would pay in context `x`.
    ///
    /// # Errors
    ///
    /// [`BanditError::Linalg`] on a dimension mismatch.
    pub fn mean_estimate(&self, x: &[f64]) -> BanditResult<f64> {
        let mut acc = 0.0;
        if x.len() != self.dimension {
            return Err(BanditError::DimensionMismatch {
                expected: self.dimension,
                actual: x.len(),
            });
        }
        for (theta_i, x_i) in self.theta.iter().zip(x) {
            acc += theta_i * x_i;
        }
        Ok(acc)
    }

    /// The **confidence width** `sqrt(x^T A^-1 x)` in the direction `x` — the
    /// standard deviation (up to the noise scale) of the estimate
    /// `theta_hat^T x`.
    ///
    /// This single quantity is the exploration signal shared by two of the three
    /// policies: `LinUCB` *adds* `alpha` times it to the score, while Thompson
    /// sampling draws a parameter whose induced score has exactly this standard
    /// deviation (times `v`). Its geometry is the reason bandits work at all:
    /// `A^-1` shrinks fastest in the directions that have been *observed* most,
    /// so the width is large precisely where the model is ignorant, and it decays
    /// like `1 / sqrt(n)` in directions that have been seen `n` times.
    ///
    /// The quadratic form is clamped at zero before the `sqrt`
    /// ([`quadratic_form_nonnegative`]), because rounding can drag a
    /// mathematically-non-negative but genuinely tiny value below zero and a
    /// `NaN` score would poison the whole ranking.
    ///
    /// # Errors
    ///
    /// [`BanditError::Linalg`] on a dimension mismatch or a non-finite form.
    pub fn confidence_width(&self, x: &[f64]) -> BanditResult<f64> {
        let form = quadratic_form_nonnegative(&self.design_inverse, x, self.dimension)?;
        Ok(form.sqrt())
    }

    /// Fold an observed reward into this arm: `A += x x^T`, `b += r x`, then
    /// refresh `theta_hat = A^-1 b`.
    ///
    /// The inverse is updated by **Sherman–Morrison in `O(d^2)`**, not by
    /// re-inverting `A`. See [`super::linalg`] for the derivation and for why the
    /// update's denominator can never vanish.
    ///
    /// A **zero** context is a legitimate no-op: `A + 0 0^T = A`, `b + r * 0 = b`,
    /// the Sherman–Morrison denominator is exactly `1`, and the arm's estimate is
    /// unchanged — the pull is still *counted*, because it happened, but it
    /// carried no information.
    ///
    /// # Errors
    ///
    /// * [`BanditError::NonFinite`] if `reward` is `NaN`/infinite.
    /// * [`BanditError::DimensionMismatch`] if `x` is the wrong length.
    /// * [`BanditError::Linalg`] if the rank-1 inverse update fails, which for a
    ///   healthy positive-definite `A^-1` it cannot.
    pub fn update(&mut self, x: &[f64], reward: f64) -> BanditResult<()> {
        check_reward(reward)?;
        if x.len() != self.dimension {
            return Err(BanditError::DimensionMismatch {
                expected: self.dimension,
                actual: x.len(),
            });
        }

        // The incremental inverse first: it is the only step that can fail, and
        // failing it *before* mutating `design` / `response` keeps the model's
        // three pieces mutually consistent even on the error path.
        sherman_morrison_update(&mut self.design_inverse, x, self.dimension)?;

        // A += x x^T.
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                self.design[i * self.dimension + j] += x[i] * x[j];
            }
        }
        // b += r x.
        for (b_i, x_i) in self.response.iter_mut().zip(x) {
            *b_i += reward * x_i;
        }

        // theta_hat = A^-1 b.
        self.theta = mat_vec(&self.design_inverse, &self.response, self.dimension)?;

        self.pulls += 1;
        self.total_reward += reward;
        Ok(())
    }

    /// This arm's contribution to a [`BanditStats`] snapshot.
    #[must_use]
    pub fn stats(&self) -> BanditArmStats {
        BanditArmStats {
            arm_id: self.arm.id.clone(),
            label: self.arm.label.clone(),
            pulls: self.pulls,
            total_reward: self.total_reward,
            mean_reward: self.mean_reward(),
            theta: self.theta.clone(),
        }
    }
}

/// An ordered set of per-arm models, with `O(1)` lookup by arm id.
///
/// Registration order is preserved and is *load-bearing*: it is the tie-break
/// for equally-scored arms, which makes a ranking over an untrained model (all
/// `theta_hat = 0`, hence all scores `0`) deterministic rather than
/// implementation-defined. That determinism is not cosmetic — it is what makes
/// the "pure greedy locks onto the deceptive arm" test *reproducible*, because
/// greedy's very first choice is decided entirely by this tie-break.
#[derive(Debug, Clone)]
pub struct BanditArmSet {
    /// The models, in registration order.
    models: Vec<BanditLinearModel>,
    /// `arm_id -> index into models`.
    index: HashMap<String, usize>,
    /// The context dimension `d`.
    dimension: usize,
    /// The ridge `lambda` new arms are initialized with.
    ridge_lambda: f64,
}

impl BanditArmSet {
    /// Build an arm set from `arms`.
    ///
    /// # Errors
    ///
    /// [`BanditError::DuplicateArm`] if two arms share an id.
    pub fn new(
        arms: impl IntoIterator<Item = BanditArm>,
        dimension: usize,
        ridge_lambda: f64,
    ) -> BanditResult<Self> {
        let mut set = Self {
            models: Vec::new(),
            index: HashMap::new(),
            dimension,
            ridge_lambda,
        };
        for arm in arms {
            set.add(arm)?;
        }
        Ok(set)
    }

    /// Register a new arm, which starts from the untrained prior.
    ///
    /// Arms may be added *online*, after the bandit has been running: a new arm
    /// begins with `theta_hat = 0` and the widest possible confidence interval,
    /// so `LinUCB` and Thompson sampling will both go and try it — which is
    /// precisely the behaviour you want when a new retrieval strategy is
    /// deployed into a live ranker.
    ///
    /// # Errors
    ///
    /// [`BanditError::DuplicateArm`] if the id is already registered.
    pub fn add(&mut self, arm: BanditArm) -> BanditResult<()> {
        if self.index.contains_key(&arm.id) {
            return Err(BanditError::DuplicateArm {
                arm_id: arm.id.clone(),
            });
        }
        self.index.insert(arm.id.clone(), self.models.len());
        self.models.push(BanditLinearModel::new(
            arm,
            self.dimension,
            self.ridge_lambda,
        ));
        Ok(())
    }

    /// The models, in registration order.
    #[must_use]
    pub fn models(&self) -> &[BanditLinearModel] {
        &self.models
    }

    /// How many arms are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether no arms are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// The registered arm ids, in registration order.
    #[must_use]
    pub fn arm_ids(&self) -> Vec<&str> {
        self.models.iter().map(BanditLinearModel::arm_id).collect()
    }

    /// The model for `arm_id`, or `None`.
    #[must_use]
    pub fn get(&self, arm_id: &str) -> Option<&BanditLinearModel> {
        self.index.get(arm_id).map(|&i| &self.models[i])
    }

    /// A mutable handle on the model for `arm_id`.
    ///
    /// # Errors
    ///
    /// [`BanditError::UnknownArm`] if the id is not registered.
    pub fn get_mut(&mut self, arm_id: &str) -> BanditResult<&mut BanditLinearModel> {
        let index = *self
            .index
            .get(arm_id)
            .ok_or_else(|| BanditError::UnknownArm {
                arm_id: arm_id.to_string(),
            })?;
        Ok(&mut self.models[index])
    }

    /// Assemble a [`BanditStats`] snapshot for a policy named `policy`.
    #[must_use]
    pub fn stats(&self, policy: &str) -> BanditStats {
        let arms: Vec<BanditArmStats> = self.models.iter().map(BanditLinearModel::stats).collect();
        let total_pulls: u64 = arms.iter().map(|arm| arm.pulls).sum();
        let total_reward: f64 = arms.iter().map(|arm| arm.total_reward).sum();
        #[allow(clippy::cast_precision_loss)] // Pull counts stay far below 2^53.
        let mean_reward = if total_pulls == 0 {
            0.0
        } else {
            total_reward / total_pulls as f64
        };
        BanditStats {
            policy: policy.to_string(),
            dimension: self.dimension,
            total_pulls,
            total_reward,
            mean_reward,
            arms,
        }
    }
}
