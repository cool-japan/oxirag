//! Pillar 2 — a trained binary quality classifier over hashed text features.
//!
//! No classifier exists anywhere else in the crate; heuristic rules
//! ([`super::heuristics`]) are cheap but brittle (a well-formed document can
//! still be low-quality prose, and a rule an author didn't anticipate can slip
//! through). [`QualityClassifier`] instead *learns* a decision boundary from
//! labelled examples: text is reduced to a fixed-width vector of hashed
//! bag-of-words/bigram term frequencies (the "hashing trick" — no vocabulary
//! to build or ship), and a logistic-regression weight vector is fit to it by
//! full-batch gradient descent on the (convex) binary cross-entropy loss.
//!
//! The training loop mirrors `crate::learning_to_rank::train`'s numerically
//! stable `sigmoid`/`softplus` primitives and its "record loss, then step,
//! stop early on convergence" structure, adapted from that module's *pairwise*
//! objective (`P(a≻b) = sigmoid(w·(x_a - x_b))`) to a genuine *binary*
//! objective (`P(good) = sigmoid(w·x + bias)`) with its own bias term.
//! Training is fully deterministic: the same [`ClassifierConfig`] and
//! examples (in the same order) always produce byte-identical weights.

use super::rng::{CurationRng, fnv1a};
use super::types::CurationError;

// ── text -> hashed features ──────────────────────────────────────────────────

/// Split `text` into lowercase alphanumeric tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Hash `bytes` into a bucket index in `[0, num_buckets)`.
#[allow(clippy::cast_possible_truncation)]
fn hash_bucket(bytes: &[u8], num_buckets: usize) -> usize {
    (fnv1a(bytes) % num_buckets as u64) as usize
}

/// Reduce `text` to a dense, hashed bag-of-words (and, when `use_bigrams`,
/// bag-of-bigrams) term-frequency vector of length `num_buckets`.
///
/// Each token (and each adjacent token pair, if bigrams are enabled) is hashed
/// via `FNV-1a` into a bucket and increments that bucket's count; the vector
/// is then normalised by token count so a feature vector reflects *frequency*
/// rather than raw length, keeping short and long documents comparable. Two
/// distinct tokens that collide into the same bucket are indistinguishable to
/// the classifier — an accepted, standard trade-off of the hashing trick that
/// shrinks arbitrarily large vocabularies to a fixed dimension with no
/// vocabulary table to build or ship.
#[allow(clippy::cast_precision_loss)]
fn hashed_term_frequencies(text: &str, num_buckets: usize, use_bigrams: bool) -> Vec<f64> {
    let buckets_len = num_buckets.max(1);
    let mut buckets = vec![0.0f64; buckets_len];
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return buckets;
    }

    for tok in &tokens {
        buckets[hash_bucket(tok.as_bytes(), buckets_len)] += 1.0;
    }
    if use_bigrams {
        for pair in tokens.windows(2) {
            let joined = format!("{} {}", pair[0], pair[1]);
            buckets[hash_bucket(joined.as_bytes(), buckets_len)] += 1.0;
        }
    }

    let n = tokens.len() as f64;
    for b in &mut buckets {
        *b /= n;
    }
    buckets
}

// ── numerically stable primitives (adapted from learning_to_rank) ──────────

/// Numerically stable logistic sigmoid, `1 / (1 + e^{-x})`.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable softplus, `ln(1 + e^x)`.
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Dot product over the common prefix of `weights` and `features` (never
/// panics on a length mismatch, though in practice both are always
/// `num_buckets` long).
fn dot(weights: &[f64], features: &[f64]) -> f64 {
    weights
        .iter()
        .zip(features.iter())
        .map(|(w, f)| w * f)
        .sum()
}

/// Deterministically initialise `dim` weights. `scale == 0.0` starts every
/// weight at zero (a valid start for this convex objective, and the default);
/// otherwise weight `i` is drawn uniformly from `(-scale, scale)` off `rng`.
fn init_weights(dim: usize, rng: &mut CurationRng, scale: f64) -> Vec<f64> {
    if scale == 0.0 {
        return vec![0.0; dim];
    }
    (0..dim)
        .map(|_| (rng.next_f64() * 2.0 - 1.0) * scale)
        .collect()
}

/// Mean binary cross-entropy loss and its gradient (w.r.t. `weights` and
/// `bias`) at the current parameters, including an `L2` penalty on `weights`
/// only (the bias is conventionally left unregularised).
fn loss_and_gradient(
    weights: &[f64],
    bias: f64,
    features: &[Vec<f64>],
    labels: &[f64],
    l2: f64,
) -> (f64, Vec<f64>, f64) {
    let dim = weights.len();
    let mut gradient = vec![0.0f64; dim];
    let mut bias_gradient = 0.0f64;
    let mut loss = 0.0f64;
    #[allow(clippy::cast_precision_loss)]
    let n = features.len() as f64;

    for (x, &y) in features.iter().zip(labels.iter()) {
        let z = dot(weights, x) + bias;
        // Stable BCE: softplus(z) - y*z == y*softplus(-z) + (1-y)*softplus(z).
        loss += softplus(z) - y * z;
        let residual = sigmoid(z) - y;
        for (g, xi) in gradient.iter_mut().zip(x.iter()) {
            *g += residual * xi;
        }
        bias_gradient += residual;
    }

    if n > 0.0 {
        loss /= n;
        for g in &mut gradient {
            *g /= n;
        }
        bias_gradient /= n;
    }

    if l2 > 0.0 {
        let mut penalty = 0.0f64;
        for (g, w) in gradient.iter_mut().zip(weights.iter()) {
            *g += l2 * w;
            penalty += w * w;
        }
        loss += 0.5 * l2 * penalty;
    }

    (loss, gradient, bias_gradient)
}

// ── ClassifierConfig ─────────────────────────────────────────────────────────

/// Hyperparameters for [`QualityClassifier::train`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierConfig {
    /// Width of the hashed feature vector. Defaults to `128`.
    pub num_buckets: usize,
    /// Whether adjacent-token bigrams are hashed in addition to unigrams.
    /// Defaults to `true`.
    pub use_bigrams: bool,
    /// Gradient-descent step size. Must be finite and strictly positive.
    /// Defaults to `0.5`.
    pub learning_rate: f64,
    /// Maximum number of full-batch epochs. Must be greater than zero.
    /// Defaults to `300`.
    pub epochs: usize,
    /// Early-stop tolerance: training halts once the absolute change in epoch
    /// loss drops below this value. Defaults to `1e-7`.
    pub tolerance: f64,
    /// `L2` regularisation coefficient applied to `weights` (not the bias).
    /// Defaults to `0.001`.
    pub l2_regularization: f64,
    /// Magnitude of deterministic uniform weight initialisation; `0.0` (the
    /// default) starts every weight at zero.
    pub weight_init_scale: f64,
    /// Seed for weight initialisation when `weight_init_scale > 0.0`.
    pub seed: u64,
    /// Minimum predicted probability for [`QualityClassifier::predict`] to
    /// report `true`. Must lie in `[0.0, 1.0]`. Defaults to `0.5`.
    pub decision_threshold: f64,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            num_buckets: 128,
            use_bigrams: true,
            learning_rate: 0.5,
            epochs: 300,
            tolerance: 1e-7,
            l2_regularization: 0.001,
            weight_init_scale: 0.0,
            seed: 0x5EED_C0DE_1234_5678,
            decision_threshold: 0.5,
        }
    }
}

impl ClassifierConfig {
    /// Construct a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hashed feature width.
    #[must_use]
    pub fn with_num_buckets(mut self, num_buckets: usize) -> Self {
        self.num_buckets = num_buckets;
        self
    }

    /// Set whether bigram features are hashed in addition to unigrams.
    #[must_use]
    pub fn with_use_bigrams(mut self, use_bigrams: bool) -> Self {
        self.use_bigrams = use_bigrams;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of epochs.
    #[must_use]
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set the early-stop convergence tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the `L2` regularisation coefficient.
    #[must_use]
    pub fn with_l2_regularization(mut self, l2_regularization: f64) -> Self {
        self.l2_regularization = l2_regularization;
        self
    }

    /// Set the weight-initialisation scale.
    #[must_use]
    pub fn with_weight_init_scale(mut self, weight_init_scale: f64) -> Self {
        self.weight_init_scale = weight_init_scale;
        self
    }

    /// Set the weight-initialisation seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the decision threshold.
    #[must_use]
    pub fn with_decision_threshold(mut self, decision_threshold: f64) -> Self {
        self.decision_threshold = decision_threshold;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::InvalidConfig`] when `num_buckets` is zero,
    /// `epochs` is zero, or any of `learning_rate`, `tolerance`,
    /// `l2_regularization`, `weight_init_scale`, `decision_threshold` is
    /// non-finite or out of its valid range.
    pub fn validate(&self) -> Result<(), CurationError> {
        if self.num_buckets == 0 {
            return Err(CurationError::InvalidConfig(
                "num_buckets must be > 0".into(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(CurationError::InvalidConfig(format!(
                "learning_rate must be finite and > 0, got {}",
                self.learning_rate
            )));
        }
        if self.epochs == 0 {
            return Err(CurationError::InvalidConfig("epochs must be > 0".into()));
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(CurationError::InvalidConfig(format!(
                "tolerance must be finite and >= 0, got {}",
                self.tolerance
            )));
        }
        if !self.l2_regularization.is_finite() || self.l2_regularization < 0.0 {
            return Err(CurationError::InvalidConfig(format!(
                "l2_regularization must be finite and >= 0, got {}",
                self.l2_regularization
            )));
        }
        if !self.weight_init_scale.is_finite() || self.weight_init_scale < 0.0 {
            return Err(CurationError::InvalidConfig(format!(
                "weight_init_scale must be finite and >= 0, got {}",
                self.weight_init_scale
            )));
        }
        if !self.decision_threshold.is_finite() || !(0.0..=1.0).contains(&self.decision_threshold) {
            return Err(CurationError::InvalidConfig(format!(
                "decision_threshold must be finite and within [0, 1], got {}",
                self.decision_threshold
            )));
        }
        Ok(())
    }
}

// ── ClassifierExample ────────────────────────────────────────────────────────

/// One labelled training or evaluation example for [`QualityClassifier`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierExample {
    /// The document text.
    pub text: String,
    /// `true` when this example is a positive ("good"/admit) instance.
    pub label: bool,
}

impl ClassifierExample {
    /// Construct a positive ("good") example.
    #[must_use]
    pub fn good(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            label: true,
        }
    }

    /// Construct a negative ("bad") example.
    #[must_use]
    pub fn bad(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            label: false,
        }
    }
}

/// Shuffle `examples` with `rng` and split into `(train, test)` by
/// `train_fraction` (clamped to `[0, 1]`).
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
pub fn train_test_split(
    examples: &[ClassifierExample],
    train_fraction: f64,
    rng: &mut CurationRng,
) -> (Vec<ClassifierExample>, Vec<ClassifierExample>) {
    let mut indices: Vec<usize> = (0..examples.len()).collect();
    rng.shuffle(&mut indices);
    let fraction = train_fraction.clamp(0.0, 1.0);
    #[allow(clippy::cast_possible_truncation)]
    let split_at = ((examples.len() as f64) * fraction).round() as usize;
    let split_at = split_at.min(examples.len());
    let train = indices[..split_at]
        .iter()
        .map(|&i| examples[i].clone())
        .collect();
    let test = indices[split_at..]
        .iter()
        .map(|&i| examples[i].clone())
        .collect();
    (train, test)
}

// ── ClassifierEvaluation ─────────────────────────────────────────────────────

/// Held-out evaluation metrics for a [`QualityClassifier`], computed by
/// [`QualityClassifier::evaluate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifierEvaluation {
    /// Number of examples evaluated.
    pub n: usize,
    /// Fraction of examples correctly classified.
    pub accuracy: f64,
    /// Of the examples predicted positive, the fraction that truly are.
    pub precision: f64,
    /// Of the examples that are truly positive, the fraction predicted so.
    pub recall: f64,
    /// The harmonic mean of precision and recall.
    pub f1: f64,
}

// ── QualityClassifier ────────────────────────────────────────────────────────

/// A trained binary logistic-regression quality classifier over hashed
/// bag-of-words/bigram text features.
///
/// See the [module documentation](self) for the feature representation and
/// training objective. `weights.len() == num_buckets`.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityClassifier {
    /// One fitted weight per hashed feature bucket.
    pub weights: Vec<f64>,
    /// The fitted bias (intercept) term.
    pub bias: f64,
    /// The hashed feature width this classifier expects (`weights.len()`).
    pub num_buckets: usize,
    /// Whether this classifier hashes bigram features in addition to
    /// unigrams (must match at inference time; stored so callers never have
    /// to remember it separately).
    pub use_bigrams: bool,
    /// The minimum predicted probability [`QualityClassifier::predict`]
    /// treats as positive.
    pub decision_threshold: f64,
    /// The number of epochs actually executed (`<= ClassifierConfig::epochs`,
    /// fewer when early stopping triggered).
    pub epochs_run: usize,
    /// The loss evaluated at the final weights.
    pub final_loss: f64,
    /// `true` when training stopped early on the convergence tolerance rather
    /// than exhausting the epoch budget.
    pub converged: bool,
    /// Per-epoch loss recorded *before* each weight update, in order.
    pub loss_history: Vec<f64>,
}

impl QualityClassifier {
    /// Fit a classifier to `examples` under `config` via full-batch gradient
    /// descent on the binary cross-entropy loss (see the
    /// [module documentation](self)).
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::InvalidConfig`] when `config` fails
    /// [`ClassifierConfig::validate`], or [`CurationError::EmptyTrainingSet`]
    /// when `examples` is empty.
    pub fn train(
        examples: &[ClassifierExample],
        config: &ClassifierConfig,
    ) -> Result<Self, CurationError> {
        config.validate()?;
        if examples.is_empty() {
            return Err(CurationError::EmptyTrainingSet);
        }

        let num_buckets = config.num_buckets.max(1);
        let features: Vec<Vec<f64>> = examples
            .iter()
            .map(|e| hashed_term_frequencies(&e.text, num_buckets, config.use_bigrams))
            .collect();
        let labels: Vec<f64> = examples
            .iter()
            .map(|e| if e.label { 1.0 } else { 0.0 })
            .collect();

        let mut rng = CurationRng::new(config.seed);
        let mut weights = init_weights(num_buckets, &mut rng, config.weight_init_scale);
        let mut bias = 0.0f64;
        let mut loss_history: Vec<f64> = Vec::with_capacity(config.epochs);
        let mut previous_loss = f64::INFINITY;
        let mut converged = false;
        let mut epochs_run = 0usize;

        for epoch in 0..config.epochs {
            let (loss, grad_w, grad_b) =
                loss_and_gradient(&weights, bias, &features, &labels, config.l2_regularization);
            loss_history.push(loss);
            epochs_run = epoch + 1;

            if (previous_loss - loss).abs() < config.tolerance {
                converged = true;
                break;
            }
            previous_loss = loss;

            for (w, g) in weights.iter_mut().zip(grad_w.iter()) {
                *w -= config.learning_rate * g;
            }
            bias -= config.learning_rate * grad_b;
        }

        let final_loss =
            loss_and_gradient(&weights, bias, &features, &labels, config.l2_regularization).0;

        Ok(Self {
            weights,
            bias,
            num_buckets,
            use_bigrams: config.use_bigrams,
            decision_threshold: config.decision_threshold,
            epochs_run,
            final_loss,
            converged,
            loss_history,
        })
    }

    /// Predict the probability that `text` is a positive ("good") example.
    #[must_use]
    pub fn predict_probability(&self, text: &str) -> f64 {
        let features = hashed_term_frequencies(text, self.num_buckets, self.use_bigrams);
        sigmoid(dot(&self.weights, &features) + self.bias)
    }

    /// `true` when [`Self::predict_probability`] is at or above
    /// [`Self::decision_threshold`].
    #[must_use]
    pub fn predict(&self, text: &str) -> bool {
        self.predict_probability(text) >= self.decision_threshold
    }

    /// Evaluate this classifier against labelled `examples`, computing
    /// accuracy, precision, recall, and `F1`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&self, examples: &[ClassifierExample]) -> ClassifierEvaluation {
        let mut true_positive = 0usize;
        let mut true_negative = 0usize;
        let mut false_positive = 0usize;
        let mut false_negative = 0usize;
        for example in examples {
            let predicted = self.predict(&example.text);
            match (predicted, example.label) {
                (true, true) => true_positive += 1,
                (true, false) => false_positive += 1,
                (false, true) => false_negative += 1,
                (false, false) => true_negative += 1,
            }
        }
        let n = examples.len();
        let accuracy = if n == 0 {
            0.0
        } else {
            (true_positive + true_negative) as f64 / n as f64
        };
        let precision = if true_positive + false_positive == 0 {
            0.0
        } else {
            true_positive as f64 / (true_positive + false_positive) as f64
        };
        let recall = if true_positive + false_negative == 0 {
            0.0
        } else {
            true_positive as f64 / (true_positive + false_negative) as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        ClassifierEvaluation {
            n,
            accuracy,
            precision,
            recall,
            f1,
        }
    }
}
