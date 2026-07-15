//! `DELTR`: disparate exposure in learning to rank (Zehlike & Castillo, `WWW`
//! 2020).
//!
//! `FA*IR` ([`fair`](super::fair)) repairs a ranking *after* the scores exist. It
//! cannot repair the scorer. If the training labels themselves are discriminatory
//! — if the judgments that a model learns from systematically under-rate one
//! group — then every ranking that model produces will under-expose that group,
//! and post-processing every one of them is treating the symptom. `DELTR` treats
//! the cause: it puts the exposure disparity **into the training objective**, so
//! the learned scorer is one whose rankings do not need repairing.
//!
//! # The objective
//!
//! For each training query, with predicted scores `s = X w` and target relevances
//! `y`:
//!
//! ```text
//! L(w) = L_listwise(w)  +  gamma * U(w)
//! ```
//!
//! **The listwise term** is `ListNet`'s top-1 cross-entropy (Cao et al., 2007).
//! Both the predictions and the labels are turned into distributions over "which
//! document belongs first",
//!
//! ```text
//! P(i) = softmax(s)_i ,      T(i) = softmax(y)_i
//! ```
//!
//! and the loss is their cross-entropy `-sum_i T(i) log P(i)`. Written with the
//! log-sum-exp `Z = log sum_j exp(s_j)` it becomes `Z - sum_i T(i) s_i`, which is
//! how it is evaluated here: no logarithm of a possibly-zero probability is ever
//! taken.
//!
//! **The exposure term** is the disparity between what the two groups are
//! predicted to receive:
//!
//! ```text
//! Exposure(G) = (1 / |G|) * sum_{i in G} P(i)
//! U(w)        = max(0, Exposure(G_unprotected) - Exposure(G_protected))^2
//! ```
//!
//! Two design decisions inside that formula carry the whole method.
//!
//! *Why `P(i)` and not `1 / log2(1 + rank_i)`.* The exposure a document really
//! receives is a function of its **rank**, and rank is a step function of the
//! scores: its gradient is zero almost everywhere and undefined at the ties. You
//! cannot descend on it. `P(i)` — the top-1 probability, which under the
//! Plackett–Luce model *is* the probability that document `i` is placed first — is
//! the standard smooth surrogate: it is large exactly when a document is scored to
//! the top, it is differentiable everywhere, and it moves in the same direction as
//! the true exposure. The disparity this module *reports* is always the real
//! position-discounted one ([`ExposureMetrics`](super::ExposureMetrics)); the
//! disparity it *descends on* is this surrogate. Conflating the two would be a
//! subtle way to lie about what was optimized.
//!
//! *Why `max(0, ...)` and not `|...|`.* The penalty is **one-sided**. It fires
//! only when the *non-protected* group is over-exposed, and is exactly zero when
//! the protected group is doing better than parity. This is not an oversight: a
//! two-sided penalty would actively push a deserving protected group back *down*
//! toward the disparity line, which is not what anybody means by anti-discrimination.
//!
//! # The gradient
//!
//! Differentiating a softmax through a group mean is where an implementation
//! quietly goes wrong, so it is written out. With `S_g = sum_{i in G_g} P(i)` and
//! `n_g = |G_g|`, and using `d P(i) / d s_j = P(i) (delta_ij - P(j))`:
//!
//! ```text
//! d L_listwise / d s_j = P(j) - T(j)
//!
//! d Exposure(G_g) / d s_j = (P(j) / n_g) * ( [j in G_g] - S_g )
//!
//! d U / d s_j = 2 * max(0, D) * ( d Exposure(G_0) / d s_j
//!                              -  d Exposure(G_1) / d s_j )
//!
//! d L / d w = X^T * ( d L / d s )
//! ```
//!
//! Note the `- S_g` term: raising `s_j` raises `P(j)` but *lowers* every other
//! `P(i)`, so a document's score affects its own group's mean exposure through
//! both channels, and an implementation that keeps only the `[j in G_g]` term has
//! a gradient that is wrong for every document and plausible for all of them. The
//! module's tests check the analytic gradient against central finite differences
//! of the loss, which is the only thing that catches this.
//!
//! # Optimization
//!
//! Batch gradient descent, with an optional ridge and an early stop on the
//! gradient norm. The weights are initialized from a small seeded interval rather
//! than zeros — see [`FairnessRng`] — so a run is replayable
//! from [`DeltrConfig::seed`].

use serde::{Deserialize, Serialize};

use super::rng::FairnessRng;
use super::types::{FairnessError, FairnessResult};

/// Hyper-parameters for [`DeltrModel`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltrConfig {
    /// The weight `gamma` on the exposure-disparity penalty. `0.0` reduces the
    /// model to plain `ListNet` — which is exactly the **ablation baseline** this
    /// module measures the fair model against, and the reason `gamma` is a
    /// parameter and not a constant.
    pub gamma: f64,
    /// Gradient-descent step size.
    pub learning_rate: f64,
    /// Maximum number of full-batch iterations.
    pub iterations: usize,
    /// Ridge coefficient on the weights. `0.0` disables it.
    pub l2_regularization: f64,
    /// Stop early once the gradient's Euclidean norm falls below this.
    pub tolerance: f64,
    /// Seed for the weight initialization.
    pub seed: u64,
    /// Half-width of the uniform interval the initial weights are drawn from.
    pub init_scale: f64,
}

impl Default for DeltrConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            learning_rate: 0.1,
            iterations: 500,
            l2_regularization: 0.0,
            tolerance: 1e-9,
            seed: 0,
            init_scale: 0.01,
        }
    }
}

/// One training query: a feature matrix, the target relevances, and the protected
/// flags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltrSample {
    /// Row-major `n x dim` feature matrix.
    features: Vec<f64>,
    /// The `n` target relevances.
    relevances: Vec<f64>,
    /// Whether each of the `n` documents belongs to the protected group.
    protected: Vec<bool>,
    /// The feature dimension.
    dim: usize,
}

impl DeltrSample {
    /// Build a training query.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::EmptyRanking`] — no documents, or `dim == 0`.
    /// * [`FairnessError::DimensionMismatch`] — `features` is not `n * dim` long,
    ///   or `relevances` / `protected` are not `n` long.
    /// * [`FairnessError::NonFinite`] — a feature or relevance is `NaN` or
    ///   infinite.
    pub fn new(
        features: Vec<f64>,
        relevances: Vec<f64>,
        protected: Vec<bool>,
        dim: usize,
    ) -> FairnessResult<Self> {
        let n = relevances.len();
        if n == 0 || dim == 0 {
            return Err(FairnessError::EmptyRanking { action: "train on" });
        }
        if features.len() != n * dim {
            return Err(FairnessError::DimensionMismatch {
                expected: n * dim,
                actual: features.len(),
            });
        }
        if protected.len() != n {
            return Err(FairnessError::DimensionMismatch {
                expected: n,
                actual: protected.len(),
            });
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(FairnessError::NonFinite {
                what: "DELTR feature matrix",
            });
        }
        if relevances.iter().any(|value| !value.is_finite()) {
            return Err(FairnessError::NonFinite {
                what: "DELTR target relevances",
            });
        }

        Ok(Self {
            features,
            relevances,
            protected,
            dim,
        })
    }

    /// How many documents the query has.
    #[must_use]
    pub fn len(&self) -> usize {
        self.relevances.len()
    }

    /// Whether the query has no documents. It never does —
    /// [`DeltrSample::new`] rejects that — but the accessor exists so `len` does
    /// not stand alone.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.relevances.is_empty()
    }

    /// The feature dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The row-major feature matrix.
    #[must_use]
    pub fn features(&self) -> &[f64] {
        &self.features
    }

    /// The target relevances.
    #[must_use]
    pub fn relevances(&self) -> &[f64] {
        &self.relevances
    }

    /// The protected flags.
    #[must_use]
    pub fn protected(&self) -> &[bool] {
        &self.protected
    }

    /// Document `index`'s feature row.
    #[must_use]
    pub fn row(&self, index: usize) -> &[f64] {
        let start = index * self.dim;
        &self.features[start..start + self.dim]
    }
}

/// The loss at one iteration, broken into the terms that make it up.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeltrLoss {
    /// Which iteration this was measured at, `0` being the initial weights.
    pub iteration: usize,
    /// `listwise + gamma * exposure_disparity + l2 * ||w||^2`.
    pub total: f64,
    /// The mean `ListNet` cross-entropy over the training queries.
    pub listwise: f64,
    /// The mean **surrogate** disparity penalty `U`, *before* being scaled by
    /// `gamma`. Reported separately so that a caller can see the fairness term
    /// fall while the listwise term rises, which is what a working `DELTR` run
    /// looks like.
    pub exposure_disparity: f64,
    /// The Euclidean norm of the gradient at these weights.
    pub gradient_norm: f64,
}

/// A linear `DELTR` scorer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltrModel {
    /// The weight vector.
    weights: Vec<f64>,
    /// The feature dimension.
    dim: usize,
    /// The hyper-parameters.
    config: DeltrConfig,
    /// Per-iteration loss trace.
    history: Vec<DeltrLoss>,
}

impl DeltrModel {
    /// Create a model over `dim` features, with weights drawn from
    /// `[-init_scale, init_scale)` using [`DeltrConfig::seed`].
    ///
    /// # Errors
    ///
    /// * [`FairnessError::EmptyRanking`] — `dim == 0`.
    /// * [`FairnessError::InvalidConfig`] — a non-finite or non-positive learning
    ///   rate, a negative `gamma`, or a negative ridge.
    pub fn new(dim: usize, config: DeltrConfig) -> FairnessResult<Self> {
        if dim == 0 {
            return Err(FairnessError::EmptyRanking {
                action: "build a DELTR model over",
            });
        }
        if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
            return Err(FairnessError::InvalidConfig {
                reason: format!("learning rate {} must be positive", config.learning_rate),
            });
        }
        if !config.gamma.is_finite() || config.gamma < 0.0 {
            return Err(FairnessError::InvalidConfig {
                reason: format!("gamma {} must be non-negative", config.gamma),
            });
        }
        if !config.l2_regularization.is_finite() || config.l2_regularization < 0.0 {
            return Err(FairnessError::InvalidConfig {
                reason: format!(
                    "l2 regularization {} must be non-negative",
                    config.l2_regularization
                ),
            });
        }

        let mut rng = FairnessRng::new(config.seed);
        let scale = config.init_scale.abs();
        let weights = (0..dim).map(|_| rng.next_range(-scale, scale)).collect();

        Ok(Self {
            weights,
            dim,
            config,
            history: Vec::new(),
        })
    }

    /// The weight vector.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The feature dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The hyper-parameters.
    #[must_use]
    pub fn config(&self) -> &DeltrConfig {
        &self.config
    }

    /// The per-iteration loss trace from the last [`DeltrModel::train`].
    #[must_use]
    pub fn history(&self) -> &[DeltrLoss] {
        &self.history
    }

    /// Overwrite the weights.
    ///
    /// Exists so that the analytic gradient can be checked against finite
    /// differences of the loss, which needs to evaluate the loss at perturbed
    /// weights without retraining.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::DimensionMismatch`] — wrong length.
    /// * [`FairnessError::NonFinite`] — a `NaN` or infinite weight.
    pub fn set_weights(&mut self, weights: Vec<f64>) -> FairnessResult<()> {
        if weights.len() != self.dim {
            return Err(FairnessError::DimensionMismatch {
                expected: self.dim,
                actual: weights.len(),
            });
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(FairnessError::NonFinite {
                what: "DELTR weight vector",
            });
        }
        self.weights = weights;
        Ok(())
    }

    /// Score one feature row.
    ///
    /// # Errors
    ///
    /// [`FairnessError::DimensionMismatch`] if `features` is not `dim` long.
    pub fn score(&self, features: &[f64]) -> FairnessResult<f64> {
        if features.len() != self.dim {
            return Err(FairnessError::DimensionMismatch {
                expected: self.dim,
                actual: features.len(),
            });
        }
        Ok(dot(&self.weights, features))
    }

    /// Score every document of a query.
    ///
    /// # Errors
    ///
    /// [`FairnessError::DimensionMismatch`] if the sample's dimension differs from
    /// the model's.
    pub fn scores(&self, sample: &DeltrSample) -> FairnessResult<Vec<f64>> {
        if sample.dim != self.dim {
            return Err(FairnessError::DimensionMismatch {
                expected: self.dim,
                actual: sample.dim,
            });
        }
        Ok((0..sample.len())
            .map(|index| dot(&self.weights, sample.row(index)))
            .collect())
    }

    /// Rank a query's documents by descending score, breaking ties by index so
    /// the ordering is deterministic.
    ///
    /// # Errors
    ///
    /// As [`DeltrModel::scores`].
    pub fn rank(&self, sample: &DeltrSample) -> FairnessResult<Vec<usize>> {
        let scores = self.scores(sample)?;
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&left, &right| {
            scores[right]
                .partial_cmp(&scores[left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(&right))
        });
        Ok(order)
    }

    /// The **signed** surrogate exposure gap of a query at the current weights:
    /// `Exposure(unprotected) - Exposure(protected)`.
    ///
    /// Positive means the non-protected group is over-exposed — the direction the
    /// penalty fires in. Exactly `0.0` when either group is absent, since there is
    /// then no between-group disparity to have.
    ///
    /// # Errors
    ///
    /// As [`DeltrModel::scores`].
    pub fn exposure_gap(&self, sample: &DeltrSample) -> FairnessResult<f64> {
        let scores = self.scores(sample)?;
        let probabilities = softmax(&scores);
        Ok(group_exposure_gap(&probabilities, sample.protected()).unwrap_or(0.0))
    }

    /// The loss at the current weights.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::EmptyRanking`] — no training queries.
    /// * [`FairnessError::DimensionMismatch`] — a sample's dimension differs from
    ///   the model's.
    /// * [`FairnessError::NonFinite`] — the loss overflowed, which means the step
    ///   size has diverged; reported rather than returned as a plausible number.
    pub fn loss(&self, samples: &[DeltrSample]) -> FairnessResult<DeltrLoss> {
        let (loss, _) = self.loss_and_gradient(samples)?;
        Ok(loss)
    }

    /// The analytic gradient of the loss at the current weights.
    ///
    /// # Errors
    ///
    /// As [`DeltrModel::loss`].
    pub fn gradient(&self, samples: &[DeltrSample]) -> FairnessResult<Vec<f64>> {
        let (_, gradient) = self.loss_and_gradient(samples)?;
        Ok(gradient)
    }

    /// The loss and its gradient, computed together because they share every
    /// intermediate.
    ///
    /// # Errors
    ///
    /// As [`DeltrModel::loss`].
    #[allow(clippy::cast_precision_loss)] // Document and query counts.
    pub fn loss_and_gradient(
        &self,
        samples: &[DeltrSample],
    ) -> FairnessResult<(DeltrLoss, Vec<f64>)> {
        if samples.is_empty() {
            return Err(FairnessError::EmptyRanking {
                action: "compute a DELTR loss over",
            });
        }

        let mut listwise_total = 0.0f64;
        let mut disparity_total = 0.0f64;
        let mut gradient = vec![0.0f64; self.dim];

        for sample in samples {
            if sample.dim != self.dim {
                return Err(FairnessError::DimensionMismatch {
                    expected: self.dim,
                    actual: sample.dim,
                });
            }
            let n = sample.len();
            let scores = self.scores(sample)?;

            // ListNet: cross-entropy between softmax(y) and softmax(s), evaluated
            // as `Z - sum_i T_i s_i` so that no log of a probability is taken.
            let normalizer = log_sum_exp(&scores);
            let predicted = softmax(&scores);
            let target = softmax(sample.relevances());
            let target_score: f64 = target
                .iter()
                .zip(&scores)
                .map(|(weight, score)| weight * score)
                .sum();
            listwise_total += normalizer - target_score;

            // d L_listwise / d s_j.
            let mut score_gradient: Vec<f64> = (0..n)
                .map(|index| predicted[index] - target[index])
                .collect();

            // The one-sided disparate-exposure penalty and its gradient, folded
            // into the score gradient (the returned gradient already carries the
            // `2 * gap * gamma` factor; the returned value is the unweighted
            // `max(0, gap)^2`, scaled by `gamma` once, below, in `total`).
            let (penalty, penalty_gradient) =
                exposure_penalty(&predicted, sample.protected(), self.config.gamma);
            disparity_total += penalty;
            for (slot, addition) in score_gradient.iter_mut().zip(&penalty_gradient) {
                *slot += addition;
            }

            // Chain back to the weights: d L / d w = X^T (d L / d s).
            for (index, &coefficient) in score_gradient.iter().enumerate() {
                let row = sample.row(index);
                for (slot, feature) in gradient.iter_mut().zip(row) {
                    *slot += coefficient * feature;
                }
            }
        }

        let queries = samples.len() as f64;
        listwise_total /= queries;
        disparity_total /= queries;
        for slot in &mut gradient {
            *slot /= queries;
        }

        // The ridge is applied to the averaged objective, so that its strength does
        // not silently depend on how many queries happened to be in the batch.
        let ridge = self.config.l2_regularization;
        let mut total = listwise_total + self.config.gamma * disparity_total;
        if ridge > 0.0 {
            let squared_norm: f64 = self.weights.iter().map(|value| value * value).sum();
            total += ridge * squared_norm;
            for (slot, weight) in gradient.iter_mut().zip(&self.weights) {
                *slot += 2.0 * ridge * weight;
            }
        }

        if !total.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(FairnessError::NonFinite {
                what: "DELTR loss or gradient (the optimization has diverged)",
            });
        }

        let gradient_norm = gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();

        Ok((
            DeltrLoss {
                iteration: 0,
                total,
                listwise: listwise_total,
                exposure_disparity: disparity_total,
                gradient_norm,
            },
            gradient,
        ))
    }

    /// Fit the model by batch gradient descent, returning the loss trace.
    ///
    /// Stops early once the gradient norm falls below [`DeltrConfig::tolerance`].
    ///
    /// # Errors
    ///
    /// As [`DeltrModel::loss`]. A divergent step size surfaces as
    /// [`FairnessError::NonFinite`] rather than as a model full of `NaN` weights.
    pub fn train(&mut self, samples: &[DeltrSample]) -> FairnessResult<&[DeltrLoss]> {
        self.history.clear();
        self.history.reserve(self.config.iterations);

        for iteration in 0..self.config.iterations {
            let (mut loss, gradient) = self.loss_and_gradient(samples)?;
            loss.iteration = iteration;
            let norm = loss.gradient_norm;
            self.history.push(loss);

            if norm < self.config.tolerance {
                break;
            }
            for (weight, slope) in self.weights.iter_mut().zip(&gradient) {
                *weight -= self.config.learning_rate * slope;
            }
            if self.weights.iter().any(|value| !value.is_finite()) {
                return Err(FairnessError::NonFinite {
                    what: "DELTR weights after a gradient step (the step size has diverged)",
                });
            }
        }

        Ok(&self.history)
    }
}

/// The dot product of two equal-length vectors.
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// `log sum_i exp(x_i)`, shifted by the maximum so that no term overflows.
fn log_sum_exp(values: &[f64]) -> f64 {
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return peak;
    }
    let sum: f64 = values.iter().map(|value| (value - peak).exp()).sum();
    peak + sum.ln()
}

/// The softmax of `values`, shifted by the maximum so that no term overflows.
///
/// An empty input gives an empty output; a degenerate input whose exponentials all
/// underflow gives the uniform distribution, which is the limit the softmax
/// approaches and the only answer that keeps the result a distribution.
fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    #[allow(clippy::cast_precision_loss)] // Documents in one query.
    let uniform = 1.0 / (values.len() as f64);
    if !peak.is_finite() {
        return vec![uniform; values.len()];
    }
    let exponentials: Vec<f64> = values.iter().map(|value| (value - peak).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return vec![uniform; values.len()];
    }
    exponentials.iter().map(|value| value / total).collect()
}

/// The one-sided disparate-exposure penalty of one query and its gradient with
/// respect to the scores.
///
/// Returns `(value, gradient)` where `value` is the **unweighted** `max(0, gap)^2`
/// — the caller scales it by `gamma` once, when forming the total loss — and
/// `gradient` is `d (gamma * max(0, gap)^2) / d s`, already carrying the
/// `2 * gap * gamma` factor. When either group is absent, or the non-protected
/// group is not over-exposed, the penalty is inactive: the value is `0.0` and the
/// gradient is all zeros. The gradient of the group-mean softmax is derived in the
/// [module documentation](self); the `- S_g` cross-terms are what an easy-but-wrong
/// implementation drops.
fn exposure_penalty(predicted: &[f64], protected: &[bool], gamma: f64) -> (f64, Vec<f64>) {
    let n = predicted.len();
    let mut gradient = vec![0.0f64; n];

    let Some(gap) = group_exposure_gap(predicted, protected) else {
        return (0.0, gradient);
    };
    if gap <= 0.0 {
        return (0.0, gradient);
    }

    // Both counts are strictly positive here: `group_exposure_gap` returns `None`
    // when either group is empty, so this point is unreachable with a zero size.
    let protected_count = protected.iter().filter(|flag| **flag).count();
    #[allow(clippy::cast_precision_loss)] // Documents in one query.
    let protected_size = protected_count as f64;
    #[allow(clippy::cast_precision_loss)]
    let unprotected_size = (n - protected_count) as f64;
    let protected_mass: f64 = predicted
        .iter()
        .zip(protected)
        .filter(|(_, flag)| **flag)
        .map(|(mass, _)| *mass)
        .sum();
    let unprotected_mass: f64 = predicted
        .iter()
        .zip(protected)
        .filter(|(_, flag)| !**flag)
        .map(|(mass, _)| *mass)
        .sum();

    let outer = 2.0 * gap * gamma;
    for ((slot, &mass), &in_protected) in gradient.iter_mut().zip(predicted).zip(protected) {
        // d Exposure(G) / d s = (P / |G|) * ([in G] - S_G).
        let unprotected_term =
            (mass / unprotected_size) * (f64::from(u8::from(!in_protected)) - unprotected_mass);
        let protected_term =
            (mass / protected_size) * (f64::from(u8::from(in_protected)) - protected_mass);
        *slot = outer * (unprotected_term - protected_term);
    }

    (gap * gap, gradient)
}

/// `Exposure(unprotected) - Exposure(protected)`, the signed surrogate disparity.
///
/// `None` when either group is empty: with only one group present there is no
/// between-group disparity, and inventing one — by treating the absent group's
/// mean as zero — would make the penalty fire on every single-group query in the
/// training set.
#[allow(clippy::cast_precision_loss)] // Documents in one query.
fn group_exposure_gap(probabilities: &[f64], protected: &[bool]) -> Option<f64> {
    let mut protected_mass = 0.0f64;
    let mut unprotected_mass = 0.0f64;
    let mut protected_count = 0usize;
    let mut unprotected_count = 0usize;

    for (&mass, &flag) in probabilities.iter().zip(protected) {
        if flag {
            protected_mass += mass;
            protected_count += 1;
        } else {
            unprotected_mass += mass;
            unprotected_count += 1;
        }
    }
    if protected_count == 0 || unprotected_count == 0 {
        return None;
    }

    Some(unprotected_mass / (unprotected_count as f64) - protected_mass / (protected_count as f64))
}
