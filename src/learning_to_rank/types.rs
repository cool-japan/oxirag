//! Types for the `learning_to_rank` module.
//!
//! This file holds the configuration ([`LtrConfig`]), the data structures that
//! flow through training and inference ([`LtrFeatureVector`], [`LtrDocument`],
//! [`LtrTrainingPair`], [`LtrTrainingSet`], [`LtrModel`]), the error enum
//! ([`LtrError`]), and the crate-local result alias ([`LtrResult`]).
//!
//! Unlike `cross_encoder`, whose `FeatureWeights` are fixed defaults over
//! *text-interaction* features, the weights carried by an [`LtrModel`] here are
//! **fitted** from labelled preference data (see `train.rs`).

use thiserror::Error;

// ── Feature layout constants ─────────────────────────────────────────────────

/// Index of the (saturated) `BM25` relevance signal in the default feature
/// layout produced by [`crate::learning_to_rank::LtrFeatureExtractor`].
pub const LTR_FEATURE_BM25: usize = 0;
/// Index of the recency signal (newer documents score closer to `1.0`).
pub const LTR_FEATURE_RECENCY: usize = 1;
/// Index of the query/document pseudo-embedding cosine-similarity signal.
pub const LTR_FEATURE_EMBEDDING_SIM: usize = 2;
/// Index of the popularity signal (more-popular documents score higher).
pub const LTR_FEATURE_POPULARITY: usize = 3;
/// Index of the document-length signal.
pub const LTR_FEATURE_LENGTH: usize = 4;
/// Dimension of the default feature layout emitted by the extractor.
pub const LTR_DEFAULT_FEATURE_DIM: usize = 5;

// ── LtrResult ────────────────────────────────────────────────────────────────

/// Convenience alias for a fallible `learning_to_rank` operation.
pub type LtrResult<T> = std::result::Result<T, LtrError>;

// ── LtrError ─────────────────────────────────────────────────────────────────

/// Errors produced by the `learning_to_rank` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LtrError {
    /// The training set contained no preference pairs.
    #[error("training set must contain at least one preference pair")]
    EmptyTrainingSet,
    /// The configured feature dimension was zero.
    #[error("feature dimension must be greater than zero")]
    ZeroFeatureDimension,
    /// The configured epoch count was zero.
    #[error("epoch count must be greater than zero")]
    ZeroEpochs,
    /// The learning rate was not a finite, strictly-positive number.
    #[error("learning rate must be finite and strictly positive, got {0}")]
    InvalidLearningRate(f64),
    /// The convergence tolerance was not finite and non-negative.
    #[error("convergence tolerance must be finite and non-negative, got {0}")]
    InvalidTolerance(f64),
    /// The `L2` regularisation coefficient was not finite and non-negative.
    #[error("l2 regularization must be finite and non-negative, got {0}")]
    InvalidRegularization(f64),
    /// The weight-initialisation scale was not finite and non-negative.
    #[error("weight init scale must be finite and non-negative, got {0}")]
    InvalidInitScale(f64),
    /// A feature vector's dimension did not match the expected dimension.
    #[error("feature vector dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch {
        /// The dimension the vector was expected to have.
        expected: usize,
        /// The dimension the vector actually had.
        found: usize,
    },
    /// A feature vector contained a non-finite (`NaN`/`inf`) value.
    #[error("feature vector contains a non-finite value at index {index}")]
    NonFiniteFeature {
        /// The index of the offending feature value.
        index: usize,
    },
    /// A preference label fell outside the accepted `[0.0, 1.0]` range or was
    /// non-finite.
    #[error("preference label must be finite and within [0.0, 1.0], got {0}")]
    InvalidLabel(f64),
}

// ── LtrFeatureVector ─────────────────────────────────────────────────────────

/// A fixed-order numeric vector of retrieval signals for a `(query, document)`
/// pair.
///
/// The default layout emitted by
/// [`crate::learning_to_rank::LtrFeatureExtractor`] is
/// `[bm25, recency, embedding_similarity, popularity, length]` (see the
/// `LTR_FEATURE_*` constants), but the training loop and model are agnostic to
/// the meaning of each dimension — they operate on any fixed-width vector of
/// finite `f64` values.
#[derive(Debug, Clone, PartialEq)]
pub struct LtrFeatureVector {
    /// The ordered feature values.
    pub values: Vec<f64>,
}

impl LtrFeatureVector {
    /// Wrap an owned vector of feature values.
    #[must_use]
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Build a feature vector by copying `values`.
    #[must_use]
    pub fn from_slice(values: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
        }
    }

    /// Build a feature vector in the default 5-signal layout.
    #[must_use]
    pub fn from_signals(
        bm25: f64,
        recency: f64,
        embedding_similarity: f64,
        popularity: f64,
        length: f64,
    ) -> Self {
        Self {
            values: vec![bm25, recency, embedding_similarity, popularity, length],
        }
    }

    /// The number of feature dimensions.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Return `true` when the vector has no dimensions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrow the underlying feature values.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Return the value at `index`, or `None` when out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    /// Dot product of these features with a weight slice, taken over the
    /// common prefix of both (so a correct-dimension pair yields the exact
    /// linear score and a mismatch never panics).
    #[must_use]
    pub fn dot(&self, weights: &[f64]) -> f64 {
        self.values
            .iter()
            .zip(weights.iter())
            .map(|(f, w)| f * w)
            .sum()
    }

    /// Validate that the vector has exactly `expected_dim` dimensions and that
    /// every value is finite.
    ///
    /// # Errors
    ///
    /// Returns [`LtrError::DimensionMismatch`] when the dimension differs from
    /// `expected_dim`, or [`LtrError::NonFiniteFeature`] when any value is
    /// `NaN` or infinite.
    pub fn validate(&self, expected_dim: usize) -> LtrResult<()> {
        if self.values.len() != expected_dim {
            return Err(LtrError::DimensionMismatch {
                expected: expected_dim,
                found: self.values.len(),
            });
        }
        for (index, value) in self.values.iter().enumerate() {
            if !value.is_finite() {
                return Err(LtrError::NonFiniteFeature { index });
            }
        }
        Ok(())
    }
}

// ── LtrDocument ──────────────────────────────────────────────────────────────

/// A raw candidate document, together with the metadata the feature extractor
/// turns into numeric retrieval signals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LtrDocument {
    /// A stable identifier for the document.
    pub id: String,
    /// The document body used for lexical (`BM25`) and semantic (pseudo-
    /// embedding) signals.
    pub content: String,
    /// A monotonic timestamp (e.g. seconds since the Unix epoch). Larger means
    /// more recent; used against a reference time to derive the recency
    /// signal.
    pub timestamp: i64,
    /// An opaque popularity count (clicks, views, inbound links, …).
    pub popularity: u64,
}

impl LtrDocument {
    /// Construct a document with a zero timestamp and zero popularity.
    #[must_use]
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            timestamp: 0,
            popularity: 0,
        }
    }

    /// Set the document timestamp (used for the recency signal).
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: i64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set the document popularity count.
    #[must_use]
    pub fn with_popularity(mut self, popularity: u64) -> Self {
        self.popularity = popularity;
        self
    }
}

// ── LtrTrainingPair ──────────────────────────────────────────────────────────

/// A single pairwise preference observation.
///
/// `label` is the *target probability that `doc_a` should rank above `doc_b`*:
/// `1.0` means "`doc_a` strictly preferred", `0.0` means "`doc_b` strictly
/// preferred", and `0.5` encodes a tie. Both feature vectors must share the
/// model's feature dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct LtrTrainingPair {
    /// The query these two documents were retrieved for. Pairs are only
    /// comparable within the same query; the id records that grouping.
    pub query_id: u64,
    /// Features of the document hypothesised to rank higher.
    pub doc_a: LtrFeatureVector,
    /// Features of the document hypothesised to rank lower.
    pub doc_b: LtrFeatureVector,
    /// Target probability in `[0.0, 1.0]` that `doc_a` outranks `doc_b`.
    pub label: f64,
}

impl LtrTrainingPair {
    /// Construct a pair with an explicit target probability `label`.
    #[must_use]
    pub fn new(
        query_id: u64,
        doc_a: LtrFeatureVector,
        doc_b: LtrFeatureVector,
        label: f64,
    ) -> Self {
        Self {
            query_id,
            doc_a,
            doc_b,
            label,
        }
    }

    /// Construct a pair asserting that `higher` should rank strictly above
    /// `lower` (label `1.0`).
    #[must_use]
    pub fn preferred(query_id: u64, higher: LtrFeatureVector, lower: LtrFeatureVector) -> Self {
        Self::new(query_id, higher, lower, 1.0)
    }

    /// The signed feature difference `doc_a - doc_b`, taken over the common
    /// prefix of the two vectors. This is the quantity the pairwise model
    /// scores.
    #[must_use]
    pub fn difference(&self) -> Vec<f64> {
        self.doc_a
            .values
            .iter()
            .zip(self.doc_b.values.iter())
            .map(|(a, b)| a - b)
            .collect()
    }

    /// Validate that both vectors have `expected_dim` finite dimensions and
    /// that `label` is finite and within `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Propagates [`LtrFeatureVector::validate`] errors for either vector and
    /// returns [`LtrError::InvalidLabel`] when the label is out of range.
    pub fn validate(&self, expected_dim: usize) -> LtrResult<()> {
        self.doc_a.validate(expected_dim)?;
        self.doc_b.validate(expected_dim)?;
        if !self.label.is_finite() || !(0.0..=1.0).contains(&self.label) {
            return Err(LtrError::InvalidLabel(self.label));
        }
        Ok(())
    }
}

// ── LtrTrainingSet ───────────────────────────────────────────────────────────

/// A collection of pairwise preference observations used to fit an
/// [`LtrModel`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LtrTrainingSet {
    /// The preference pairs, in a stable, caller-supplied order (which the
    /// deterministic training loop preserves).
    pub pairs: Vec<LtrTrainingPair>,
}

impl LtrTrainingSet {
    /// Construct an empty training set.
    #[must_use]
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Construct a training set from a vector of pairs.
    #[must_use]
    pub fn from_pairs(pairs: Vec<LtrTrainingPair>) -> Self {
        Self { pairs }
    }

    /// Append a preference pair, returning `self` for chaining.
    #[must_use]
    pub fn with_pair(mut self, pair: LtrTrainingPair) -> Self {
        self.pairs.push(pair);
        self
    }

    /// Append a preference pair in place.
    pub fn push(&mut self, pair: LtrTrainingPair) {
        self.pairs.push(pair);
    }

    /// The number of preference pairs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Return `true` when the set holds no pairs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The feature dimension of the first pair's `doc_a`, or `None` when the
    /// set is empty.
    #[must_use]
    pub fn feature_dim(&self) -> Option<usize> {
        self.pairs.first().map(|p| p.doc_a.dim())
    }

    /// Validate every pair against `expected_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`LtrError::EmptyTrainingSet`] when there are no pairs, or the
    /// first per-pair validation error otherwise (see
    /// [`LtrTrainingPair::validate`]).
    pub fn validate(&self, expected_dim: usize) -> LtrResult<()> {
        if self.pairs.is_empty() {
            return Err(LtrError::EmptyTrainingSet);
        }
        for pair in &self.pairs {
            pair.validate(expected_dim)?;
        }
        Ok(())
    }
}

// ── LtrConfig ────────────────────────────────────────────────────────────────

/// Configuration for the pairwise `RankNet`-style training loop.
#[derive(Debug, Clone, PartialEq)]
pub struct LtrConfig {
    /// Gradient-descent step size. Must be finite and strictly positive.
    pub learning_rate: f64,
    /// Maximum number of full-batch epochs. Must be greater than zero.
    pub epochs: usize,
    /// Early-stop tolerance: training halts once the absolute change in epoch
    /// loss drops below this value. Must be finite and non-negative.
    pub tolerance: f64,
    /// The number of feature dimensions (and therefore weights). Must be
    /// greater than zero.
    pub feature_dim: usize,
    /// Seed used to derive deterministic initial weights when
    /// `weight_init_scale > 0.0`.
    pub seed: u64,
    /// `L2` regularisation coefficient applied to the weights each step. Must
    /// be finite and non-negative. Defaults to `0.0` (no regularisation).
    pub l2_regularization: f64,
    /// Magnitude of the deterministic uniform weight initialisation. `0.0`
    /// (the default) yields an all-zero start, which is both deterministic and
    /// a valid initialisation for this convex objective.
    pub weight_init_scale: f64,
}

impl Default for LtrConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            epochs: 200,
            tolerance: 1e-6,
            feature_dim: LTR_DEFAULT_FEATURE_DIM,
            seed: 0x1234_5678_9ABC_DEF0,
            l2_regularization: 0.0,
            weight_init_scale: 0.0,
        }
    }
}

impl LtrConfig {
    /// Construct a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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

    /// Set the feature dimension.
    #[must_use]
    pub fn with_feature_dim(mut self, feature_dim: usize) -> Self {
        self.feature_dim = feature_dim;
        self
    }

    /// Set the weight-initialisation seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
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

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns the matching `LtrError` variant for a non-positive/non-finite
    /// learning rate, zero epochs, a negative/non-finite tolerance, a zero
    /// feature dimension, or a negative/non-finite regularisation or
    /// initialisation scale.
    pub fn validate(&self) -> LtrResult<()> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(LtrError::InvalidLearningRate(self.learning_rate));
        }
        if self.epochs == 0 {
            return Err(LtrError::ZeroEpochs);
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(LtrError::InvalidTolerance(self.tolerance));
        }
        if self.feature_dim == 0 {
            return Err(LtrError::ZeroFeatureDimension);
        }
        if !self.l2_regularization.is_finite() || self.l2_regularization < 0.0 {
            return Err(LtrError::InvalidRegularization(self.l2_regularization));
        }
        if !self.weight_init_scale.is_finite() || self.weight_init_scale < 0.0 {
            return Err(LtrError::InvalidInitScale(self.weight_init_scale));
        }
        Ok(())
    }
}

// ── LtrModel ─────────────────────────────────────────────────────────────────

/// A trained pairwise ranking model: the fitted weight vector plus feature and
/// training metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct LtrModel {
    /// One fitted weight per feature dimension.
    pub weights: Vec<f64>,
    /// The feature dimension this model expects.
    pub feature_dim: usize,
    /// The number of epochs actually executed (`<= LtrConfig::epochs`, fewer
    /// when early stopping triggered).
    pub epochs_run: usize,
    /// The loss evaluated at the final weights.
    pub final_loss: f64,
    /// `true` when training stopped early on the convergence tolerance rather
    /// than exhausting the epoch budget.
    pub converged: bool,
    /// The per-epoch loss values recorded *before* each weight update, in
    /// order. Non-increasing for a suitably small learning rate.
    pub loss_history: Vec<f64>,
}

impl LtrModel {
    /// Score a feature vector as the linear combination `w · features`.
    ///
    /// The dot product is taken over the common prefix of the weights and the
    /// features (so it never panics); for a correctly-dimensioned input this
    /// is the exact model score. Use [`LtrModel::try_score`] to reject a
    /// mismatched dimension explicitly.
    #[must_use]
    pub fn score(&self, features: &LtrFeatureVector) -> f64 {
        features.dot(&self.weights)
    }

    /// Score a feature vector after validating its dimension and finiteness.
    ///
    /// # Errors
    ///
    /// Returns [`LtrError::DimensionMismatch`] or
    /// [`LtrError::NonFiniteFeature`] via [`LtrFeatureVector::validate`].
    pub fn try_score(&self, features: &LtrFeatureVector) -> LtrResult<f64> {
        features.validate(self.feature_dim)?;
        Ok(self.score(features))
    }

    /// The weight at `index`, or `None` when out of range.
    #[must_use]
    pub fn weight(&self, index: usize) -> Option<f64> {
        self.weights.get(index).copied()
    }
}
