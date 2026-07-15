//! [`LinearProbe`] — the binary linear classifier this module fits on contrastive
//! activations, and the two ways of fitting it.
//!
//! # Why this is written here rather than imported
//!
//! There is no logistic regression, and no linear classifier of any kind,
//! anywhere else in this crate. `ITI` needs one per attention head — hundreds of
//! them, each over a few dozen `head_dim`-wide examples — so this file states
//! one: a convex objective, a step size that provably decreases it, and a
//! closed-form alternative for the case where you do not want to optimize at all.
//!
//! # The objective
//!
//! With labels `y in {0, 1}` and the ±1 encoding `m = 2y - 1`, the mean logistic
//! loss with an `L2` penalty on the weights is
//!
//! ```text
//! F(w, b) = (1/n) * sum_i  softplus(-m_i * (w . x_i + b))  +  (lambda / 2) * ||w||^2
//! ```
//!
//! which is convex and differentiable, with gradient
//!
//! ```text
//! dF/dw = (1/n) * sum_i (sigmoid(z_i) - y_i) * x_i  +  lambda * w
//! dF/db = (1/n) * sum_i (sigmoid(z_i) - y_i)                    ,  z_i = w . x_i + b
//! ```
//!
//! The bias is **not** penalized. Shrinking an intercept toward zero is a claim
//! about where the activations sit in absolute terms, and nobody making that
//! claim means to.
//!
//! # The step size is derived, not tuned
//!
//! `F` is `L`-smooth with
//!
//! ```text
//! L  <=  ||X~||_F^2 / (4n)  +  lambda
//! ```
//!
//! where `X~` is the design matrix with a column of ones appended for the bias.
//! (The Hessian of the mean logistic loss is `(1/n) sum_i s_i (1 - s_i) x~_i x~_i^T`,
//! and `s(1-s) <= 1/4`; the spectral norm of the resulting Gram matrix is at most
//! its Frobenius norm.) Gradient descent with `step = 1 / L` therefore satisfies
//! the descent lemma, and the loss is **monotonically non-increasing** — an
//! invariant strong enough to catch a wrong gradient, and one this module's tests
//! assert on every iterate.
//!
//! There is no learning rate to tune, and no schedule to get wrong. The bound is
//! crude — `||X~||_F^2` over-estimates `||X~||_2^2` by up to a factor of
//! `rank(X~)` — so convergence is slower than an optimal step would give. That is
//! the right trade for a few dozen `head_dim`-wide examples: the whole fit costs
//! microseconds, and a step size that is *provably* safe is worth far more here
//! than one that is merely fast.

// Sample counts and dimensions are averaged over, so they become `f64`. A
// contrastive dataset with more than 2^53 examples is not a thing.
#![allow(clippy::cast_precision_loss)]

use super::types::{
    ProbeAccuracy, ProbeMethod, SteeringConfig, SteeringError, SteeringResult, SteeringVector,
};

/// The logistic function, evaluated without overflowing.
///
/// `1 / (1 + e^-z)` overflows for very negative `z`; `e^z / (1 + e^z)` overflows
/// for very positive `z`. Branching on the sign uses whichever form has a bounded
/// exponent, so the result is finite for every finite input.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let exponential = z.exp();
        exponential / (1.0 + exponential)
    }
}

/// `log(1 + e^t)`, evaluated without overflowing.
///
/// The identity `softplus(t) = max(t, 0) + log(1 + e^-|t|)` keeps the exponent
/// non-positive, so `e^-|t|` is in `(0, 1]` and `ln_1p` of it is exact to the last
/// bit for small arguments.
fn softplus(t: f64) -> f64 {
    t.max(0.0) + (-t.abs()).exp().ln_1p()
}

/// A binary linear classifier over one activation site: `logit(x) = w . x + b`.
///
/// Both [`ProbeMethod`]s produce one of these, and — this is the point of the
/// design — for both of them the **weight vector is the steering direction**:
///
/// * [`ProbeMethod::Logistic`]: `w` is the gradient-descent solution, so
///   `w / ||w||` is the discriminative direction.
/// * [`ProbeMethod::MassMean`]: `w` is *defined* as `mean(pos) - mean(neg)`, and
///   `b` is chosen to put the decision boundary at the midpoint of the two class
///   means, so `w / ||w||` is the mass-mean direction and the classifier is the
///   nearest-centroid rule.
///
/// So [`LinearProbe::weight_vector`] is the direction, whichever method produced
/// the probe, and the engine never has to branch on the method to find it.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearProbe {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearProbe {
    /// Wrap a weight vector and a bias.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::EmptyVector`] when `weights` is empty, and
    /// [`SteeringError::NonFinite`] when any weight, or the bias, is `NaN` or
    /// infinite.
    pub fn new(weights: Vec<f64>, bias: f64) -> SteeringResult<Self> {
        if weights.is_empty() {
            return Err(SteeringError::EmptyVector {
                what: "probe weights",
            });
        }
        for (index, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(SteeringError::NonFinite {
                    what: "probe weight",
                    index,
                    value: weight,
                });
            }
        }
        if !bias.is_finite() {
            return Err(SteeringError::NonFinite {
                what: "probe bias",
                index: 0,
                value: bias,
            });
        }
        Ok(Self { weights, bias })
    }

    /// The weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The bias.
    #[must_use]
    pub const fn bias(&self) -> f64 {
        self.bias
    }

    /// The width of the activation this probe reads.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.weights.len()
    }

    /// The weights as a [`SteeringVector`] — **unnormalized**.
    ///
    /// This is the raw steering direction, whose norm is meaningful for
    /// [`ProbeMethod::MassMean`] (it is the distance between the class means) and
    /// arbitrary for [`ProbeMethod::Logistic`] (it is set by the `L2` penalty).
    /// Normalize it before use as an `ITI` direction.
    ///
    /// # Errors
    ///
    /// Cannot fail for a probe built through [`LinearProbe::new`], which already
    /// rejects empty and non-finite weights; the `Result` exists because
    /// [`SteeringVector`] enforces those invariants at its own boundary rather
    /// than trusting its caller.
    pub fn weight_vector(&self) -> SteeringResult<SteeringVector> {
        SteeringVector::new(self.weights.clone())
    }

    /// `w . x + b`.
    ///
    /// The `f32 -> f64` promotion is exact; the accumulation is what needs the
    /// wider type.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::DimensionMismatch`] when `activation` is the
    /// wrong width.
    pub fn logit(&self, activation: &[f32]) -> SteeringResult<f64> {
        if activation.len() != self.weights.len() {
            return Err(SteeringError::DimensionMismatch {
                what: "probe input",
                expected: self.weights.len(),
                actual: activation.len(),
            });
        }
        let dot: f64 = self
            .weights
            .iter()
            .zip(activation)
            .map(|(w, &x)| w * f64::from(x))
            .sum();
        Ok(dot + self.bias)
    }

    /// `sigmoid(w . x + b)` — the probability the probe assigns to the positive
    /// class.
    ///
    /// # Errors
    ///
    /// Propagates [`LinearProbe::logit`].
    pub fn probability(&self, activation: &[f32]) -> SteeringResult<f64> {
        self.logit(activation).map(sigmoid)
    }

    /// Whether the probe calls `activation` positive.
    ///
    /// The decision rule is `logit > 0`, so the boundary itself is classified
    /// **negative**. That is an arbitrary choice, but it is a *total* one, and it
    /// matters in exactly one case that is not arbitrary at all: a **degenerate**
    /// probe (`w = 0, b = 0`, which is what fitting a head whose two classes have
    /// identical activations produces) assigns `logit == 0` to everything,
    /// predicts a single class, and therefore scores **exactly chance** on a
    /// balanced set. That is the honest score for a head that carries no signal,
    /// and it is what keeps such a head from floating to the top of the ranking on
    /// a coin-flip.
    ///
    /// # Errors
    ///
    /// Propagates [`LinearProbe::logit`].
    pub fn predict(&self, activation: &[f32]) -> SteeringResult<bool> {
        self.logit(activation).map(|z| z > 0.0)
    }

    /// The fraction of `positive` and `negative` the probe classifies correctly.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NoPairs`] when both sets are empty, and
    /// propagates [`LinearProbe::logit`].
    pub fn accuracy(&self, positive: &[&[f32]], negative: &[&[f32]]) -> SteeringResult<f64> {
        let total = positive.len() + negative.len();
        if total == 0 {
            return Err(SteeringError::NoPairs);
        }
        let mut correct = 0usize;
        for activation in positive {
            if self.predict(activation)? {
                correct += 1;
            }
        }
        for activation in negative {
            if !self.predict(activation)? {
                correct += 1;
            }
        }
        Ok(correct as f64 / total as f64)
    }

    /// The mean logistic loss of this probe on a labelled set, with the same `L2`
    /// penalty the fit used.
    ///
    /// Reported for both methods, so that a mass-mean probe and a logistic probe
    /// on the same head carry *comparable* numbers.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NoPairs`] when both sets are empty, and propagates
    /// [`LinearProbe::logit`].
    pub fn logistic_loss(
        &self,
        positive: &[&[f32]],
        negative: &[&[f32]],
        l2_penalty: f64,
    ) -> SteeringResult<f64> {
        let total = positive.len() + negative.len();
        if total == 0 {
            return Err(SteeringError::NoPairs);
        }
        let mut sum = 0.0;
        for activation in positive {
            // label +1: softplus(-z)
            sum += softplus(-self.logit(activation)?);
        }
        for activation in negative {
            // label -1: softplus(+z)
            sum += softplus(self.logit(activation)?);
        }
        let penalty = 0.5 * l2_penalty * self.weights.iter().map(|w| w * w).sum::<f64>();
        Ok(sum / total as f64 + penalty)
    }

    /// Fit by **gradient descent on the logistic loss**, with the provably-safe
    /// step size derived in the [module documentation](self).
    ///
    /// Returns the fitted probe, the loss at every iterate (of length
    /// `iterations + 1`, and **non-increasing** by construction), and whether the
    /// gradient's infinity-norm fell below `tolerance` before the iteration budget
    /// ran out.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NoPairs`] when either class is empty (a
    /// one-class "classifier" is not a classifier),
    /// [`SteeringError::DimensionMismatch`] when the activations disagree on
    /// width, [`SteeringError::NonFinite`] when any activation is not finite, and
    /// [`SteeringError::ProbeDiverged`] if the loss ever leaves the reals — which
    /// the derived step size makes impossible, and which is checked anyway
    /// because a silent `NaN` here would become a silent `NaN` steering vector.
    pub fn fit_logistic(
        positive: &[&[f32]],
        negative: &[&[f32]],
        l2_penalty: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> SteeringResult<(Self, Vec<f64>, bool)> {
        let dim = check_two_class_input(positive, negative)?;
        let count = positive.len() + negative.len();
        let count_f = count as f64;

        // The design matrix, promoted once, with its ±1 label. Promotion of `f32`
        // to `f64` is exact.
        let mut samples: Vec<(Vec<f64>, f64)> = Vec::with_capacity(count);
        for activation in positive {
            samples.push((activation.iter().map(|&x| f64::from(x)).collect(), 1.0));
        }
        for activation in negative {
            samples.push((activation.iter().map(|&x| f64::from(x)).collect(), 0.0));
        }

        // `||X~||_F^2 = sum_i (||x_i||^2 + 1)`: the `+1` per row is the bias column.
        let frobenius_squared: f64 = samples
            .iter()
            .map(|(x, _)| x.iter().map(|v| v * v).sum::<f64>() + 1.0)
            .sum();
        // `L <= ||X~||_F^2 / (4n) + lambda`, and `frobenius_squared >= n`, so
        // `lipschitz >= 1/4 > 0`: the step size can never divide by zero.
        let lipschitz = frobenius_squared / (4.0 * count_f) + l2_penalty;
        let step = 1.0 / lipschitz;

        let mut weights = vec![0.0_f64; dim];
        let mut bias = 0.0_f64;
        let mut loss_history: Vec<f64> = Vec::with_capacity(max_iterations + 1);
        let mut converged = false;
        let mut iterations = 0usize;

        loop {
            let mut loss = 0.0_f64;
            let mut gradient = vec![0.0_f64; dim];
            let mut gradient_bias = 0.0_f64;

            for (activation, label) in &samples {
                let z: f64 = weights
                    .iter()
                    .zip(activation)
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias;
                // `softplus(-m z)` with `m = 2y - 1`: for `y = 1` this is
                // `softplus(-z)`, for `y = 0` it is `softplus(z)`.
                loss += softplus(label.mul_add(-2.0, 1.0) * z);
                let residual = sigmoid(z) - label;
                for (slot, x) in gradient.iter_mut().zip(activation) {
                    *slot += residual * x;
                }
                gradient_bias += residual;
            }

            loss /= count_f;
            gradient_bias /= count_f;
            for (slot, weight) in gradient.iter_mut().zip(&weights) {
                *slot = *slot / count_f + l2_penalty * weight;
            }
            loss += 0.5 * l2_penalty * weights.iter().map(|w| w * w).sum::<f64>();

            if !loss.is_finite() {
                return Err(SteeringError::ProbeDiverged {
                    iteration: iterations,
                    loss,
                });
            }
            loss_history.push(loss);

            let gradient_norm = gradient
                .iter()
                .map(|g| g.abs())
                .fold(gradient_bias.abs(), f64::max);
            if gradient_norm <= tolerance {
                converged = true;
                break;
            }
            if iterations >= max_iterations {
                break;
            }

            for (weight, g) in weights.iter_mut().zip(&gradient) {
                *weight -= step * g;
            }
            bias -= step * gradient_bias;
            iterations += 1;
        }

        Ok((Self::new(weights, bias)?, loss_history, converged))
    }

    /// Fit in **closed form**: `w = mean(positive) - mean(negative)`, with the
    /// bias placing the boundary at the midpoint of the two class means.
    ///
    /// No optimization, no hyper-parameters, nothing to converge. The resulting
    /// classifier is nearest-centroid: `logit(x) = w . (x - (mu_pos + mu_neg) / 2)`,
    /// which is positive exactly when `x` is closer to `mu_pos` than to `mu_neg`
    /// *along `w`*.
    ///
    /// When the two class means coincide the weights are the zero vector, the
    /// probe is degenerate, and it will score exactly chance — see
    /// [`LinearProbe::predict`]. That is not a failure to report; it is the
    /// correct description of a head that does not distinguish the classes at all,
    /// and the engine relies on it to keep such a head out of the top-`K`.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NoPairs`] when either class is empty,
    /// [`SteeringError::DimensionMismatch`] when the activations disagree on
    /// width, and [`SteeringError::NonFinite`] when any activation is not finite.
    pub fn fit_mass_mean(positive: &[&[f32]], negative: &[&[f32]]) -> SteeringResult<Self> {
        let dim = check_two_class_input(positive, negative)?;
        let positive_mean = class_mean(positive, dim);
        let negative_mean = class_mean(negative, dim);

        let weights: Vec<f64> = positive_mean
            .iter()
            .zip(&negative_mean)
            .map(|(p, n)| p - n)
            .collect();
        // b = -w . midpoint, so that logit(x) = w . (x - midpoint).
        let bias: f64 = -weights
            .iter()
            .zip(positive_mean.iter().zip(&negative_mean))
            .map(|(w, (p, n))| w * 0.5 * (p + n))
            .sum::<f64>();

        Self::new(weights, bias)
    }

    /// Fit a probe on the training split and score it on **both** splits.
    ///
    /// This is the module's single probe-fitting entry point, and it takes the
    /// validation split as an argument rather than leaving it to the caller for a
    /// reason: a [`SteeringProbeResult`] without a validation accuracy would still
    /// *have* a validation-accuracy field, and whatever went in it would be a
    /// number nobody measured. Head selection ranks on that field.
    ///
    /// # Errors
    ///
    /// Propagates [`LinearProbe::fit_logistic`] / [`LinearProbe::fit_mass_mean`]
    /// and [`LinearProbe::accuracy`]; in particular, returns
    /// [`SteeringError::NoPairs`] when either split is missing a class.
    pub fn fit(
        method: ProbeMethod,
        train_positive: &[&[f32]],
        train_negative: &[&[f32]],
        validation_positive: &[&[f32]],
        validation_negative: &[&[f32]],
        config: &SteeringConfig,
    ) -> SteeringResult<SteeringProbeResult> {
        let (probe, loss_history, converged) = match method {
            ProbeMethod::Logistic => LinearProbe::fit_logistic(
                train_positive,
                train_negative,
                config.l2_penalty,
                config.max_iterations,
                config.tolerance,
            )?,
            ProbeMethod::MassMean => {
                let probe = LinearProbe::fit_mass_mean(train_positive, train_negative)?;
                // Closed form: one "iterate", and its loss is reported so that a
                // mass-mean probe is comparable to a logistic one on the same head.
                let loss =
                    probe.logistic_loss(train_positive, train_negative, config.l2_penalty)?;
                (probe, vec![loss], true)
            }
        };

        let accuracy = ProbeAccuracy {
            train: probe.accuracy(train_positive, train_negative)?,
            validation: probe.accuracy(validation_positive, validation_negative)?,
        };
        let iterations = loss_history.len().saturating_sub(1);

        Ok(SteeringProbeResult {
            probe,
            accuracy,
            iterations,
            converged,
            loss_history,
        })
    }
}

/// The outcome of fitting one probe: the probe itself, what it scored, and what
/// the optimizer did to get there.
///
/// Named `SteeringProbeResult` rather than `ProbeResult` because the crate's flat
/// prelude already exports a `ProbeResult` from `membership_inference`, which is a
/// completely unrelated object (the outcome of a membership-inference attack).
#[derive(Debug, Clone, PartialEq)]
pub struct SteeringProbeResult {
    /// The fitted classifier. Its weight vector is the steering direction.
    pub probe: LinearProbe,
    /// What it scored on the training and validation splits.
    pub accuracy: ProbeAccuracy,
    /// How many gradient steps were taken (`0` for [`ProbeMethod::MassMean`],
    /// which takes none).
    pub iterations: usize,
    /// Whether the gradient's infinity-norm fell below the tolerance before the
    /// iteration budget ran out. A probe that did *not* converge is still
    /// returned — with its accuracy honestly measured — because on a
    /// well-separated head the logistic loss has no finite minimizer and the
    /// weights grow without bound, so "did not converge" is the *expected*
    /// outcome there, not a failure.
    pub converged: bool,
    /// The loss at every iterate, `iterations + 1` long. Non-increasing by
    /// construction; the module's tests assert exactly that.
    pub loss_history: Vec<f64>,
}

impl SteeringProbeResult {
    /// The loss at the final iterate.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::EmptyVector`] when the loss history is empty,
    /// which [`LinearProbe::fit`] never produces.
    pub fn final_loss(&self) -> SteeringResult<f64> {
        self.loss_history
            .last()
            .copied()
            .ok_or(SteeringError::EmptyVector {
                what: "probe loss history",
            })
    }
}

/// Validate a two-class activation set and return its common width.
fn check_two_class_input(positive: &[&[f32]], negative: &[&[f32]]) -> SteeringResult<usize> {
    if positive.is_empty() || negative.is_empty() {
        return Err(SteeringError::NoPairs);
    }
    let dim = positive[0].len();
    if dim == 0 {
        return Err(SteeringError::EmptyVector {
            what: "probe activation",
        });
    }
    for (what, class) in [("positive", positive), ("negative", negative)] {
        for activation in class {
            if activation.len() != dim {
                return Err(SteeringError::DimensionMismatch {
                    what: "probe activation",
                    expected: dim,
                    actual: activation.len(),
                });
            }
            for (index, &value) in activation.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SteeringError::NonFinite {
                        what: if what == "positive" {
                            "positive activation"
                        } else {
                            "negative activation"
                        },
                        index,
                        value: f64::from(value),
                    });
                }
            }
        }
    }
    Ok(dim)
}

/// The componentwise mean of a class's activations, in `f64`.
fn class_mean(class: &[&[f32]], dim: usize) -> Vec<f64> {
    let mut mean = vec![0.0_f64; dim];
    for activation in class {
        for (slot, &value) in mean.iter_mut().zip(*activation) {
            *slot += f64::from(value);
        }
    }
    let count = class.len() as f64;
    for slot in &mut mean {
        *slot /= count;
    }
    mean
}
