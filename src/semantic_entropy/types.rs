//! Types for the `semantic_entropy` module.

use thiserror::Error;

// ── MeaningCluster ────────────────────────────────────────────────────────────

/// A cluster of sampled answers that are mutually semantically equivalent.
///
/// Each cluster groups answers judged to share the same *meaning* under the
/// bidirectional-entailment stand-in (high token-set Jaccard plus high mutual
/// containment).  The cluster's [`probability`](Self::probability) is the share
/// of total sample weight assigned to it and is used to compute the semantic
/// entropy over the cluster distribution.
#[derive(Debug, Clone)]
pub struct MeaningCluster {
    /// The representative answer for this cluster (the first member added).
    pub representative: String,
    /// Indices (into the original answer slice) of the answers in this cluster.
    pub members: Vec<usize>,
    /// Probability mass `weight(cluster) / total_weight`, in `[0.0, 1.0]`.
    pub probability: f32,
}

impl MeaningCluster {
    /// Number of sampled answers contained in this cluster.
    #[must_use]
    pub fn size(&self) -> usize {
        self.members.len()
    }
}

// ── SemanticEntropyResult ─────────────────────────────────────────────────────

/// Outcome of estimating semantic entropy over a set of sampled answers.
///
/// Produced by
/// [`SemanticEntropyEstimator::estimate`](super::estimator::SemanticEntropyEstimator::estimate).
///
/// Low [`entropy`](Self::entropy) / [`normalized_entropy`](Self::normalized_entropy)
/// indicates the model is confident and consistent (the samples collapse into a
/// few meanings); high values indicate uncertainty and a higher risk of
/// hallucination.
#[derive(Debug, Clone)]
pub struct SemanticEntropyResult {
    /// Discrete semantic entropy `H = -Σ p_c·log(p_c)` over the cluster
    /// distribution.  Uses natural log unless `use_log2` is set on the config,
    /// in which case base-2 is used.
    pub entropy: f32,
    /// [`entropy`](Self::entropy) divided by `log(num_clusters)`, in `[0.0, 1.0]`.
    ///
    /// `0.0` when there is a single cluster (no uncertainty).
    pub normalized_entropy: f32,
    /// Number of distinct meaning clusters discovered.
    pub num_clusters: usize,
    /// Total number of answer samples that were clustered.
    pub num_samples: usize,
    /// The discovered meaning clusters.
    pub clusters: Vec<MeaningCluster>,
    /// Naive lexical entropy: `H` computed treating **each answer as its own
    /// cluster** (an upper-bound baseline).  Always `>=` [`entropy`](Self::entropy)
    /// on the same sample set, so comparing the two reveals how much semantic
    /// clustering reduced the apparent uncertainty.
    pub predictive_entropy: f32,
}

// ── SemanticEntropyConfig ─────────────────────────────────────────────────────

/// Configuration for
/// [`SemanticEntropyEstimator`](super::estimator::SemanticEntropyEstimator).
#[derive(Debug, Clone)]
pub struct SemanticEntropyConfig {
    /// Minimum token-set Jaccard **and** minimum mutual containment required for
    /// two answers to be deemed semantically equivalent.
    ///
    /// Default: `0.6`.
    pub equivalence_threshold: f32,
    /// When `true`, entropy is computed in bits (base-2 log); otherwise in nats
    /// (natural log).  Normalisation cancels the base, so this does not affect
    /// [`SemanticEntropyResult::normalized_entropy`].
    ///
    /// Default: `false`.
    pub use_log2: bool,
}

impl Default for SemanticEntropyConfig {
    fn default() -> Self {
        Self {
            equivalence_threshold: 0.6,
            use_log2: false,
        }
    }
}

impl SemanticEntropyConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the semantic-equivalence threshold (Jaccard and mutual containment).
    #[must_use]
    pub fn with_equivalence_threshold(mut self, equivalence_threshold: f32) -> Self {
        self.equivalence_threshold = equivalence_threshold;
        self
    }

    /// Choose base-2 (`true`) or natural (`false`) logarithm for entropy.
    #[must_use]
    pub fn with_use_log2(mut self, use_log2: bool) -> Self {
        self.use_log2 = use_log2;
        self
    }
}

// ── SemanticEntropyError ──────────────────────────────────────────────────────

/// Errors produced by
/// [`SemanticEntropyEstimator`](super::estimator::SemanticEntropyEstimator).
#[derive(Debug, Error)]
pub enum SemanticEntropyError {
    /// No answer samples were supplied.
    #[error("no answer samples")]
    EmptySamples,
    /// The supplied weight slice length did not match the answer slice length.
    #[error("weights length {weights} != answers length {answers}")]
    WeightMismatch {
        /// Length of the supplied weight slice.
        weights: usize,
        /// Length of the supplied answer slice.
        answers: usize,
    },
}
