//! Core semantic-entropy logic: tokenisation, bidirectional-entailment
//! clustering, and entropy over the cluster distribution.

use std::collections::HashSet;

use super::types::{
    MeaningCluster, SemanticEntropyConfig, SemanticEntropyError, SemanticEntropyResult,
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Tokenize `text` into a lowercase set of alphanumeric tokens.
///
/// Splits on any non-alphanumeric character and drops empty fragments.
fn token_set(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard similarity between two token sets.
///
/// Returns `0.0` when both sets are empty (no shared meaning can be asserted).
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let score = intersection as f32 / union as f32;
    score
}

/// Directional containment `|a ∩ b| / |a|` — the fraction of `a` covered by `b`.
///
/// Returns `0.0` when `a` is empty.
fn containment(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    #[allow(clippy::cast_precision_loss)]
    let score = intersection as f32 / a.len() as f32;
    score
}

/// Bidirectional-entailment stand-in: two token sets are *semantically
/// equivalent* when their Jaccard is `>= threshold` **and** both directional
/// containments are `>= threshold` (mutual containment is high).
fn semantically_equivalent(a: &HashSet<String>, b: &HashSet<String>, threshold: f32) -> bool {
    jaccard(a, b) >= threshold && containment(a, b) >= threshold && containment(b, a) >= threshold
}

/// Shannon entropy `-Σ p·log(p)` over a probability slice.
///
/// Uses base-2 logarithm when `use_log2`, otherwise natural log.  Zero-mass
/// entries contribute nothing (`0·log 0 := 0`).
fn shannon_entropy(probs: &[f32], use_log2: bool) -> f32 {
    let mut h = 0.0_f32;
    for &p in probs {
        if p > 0.0 {
            let log_p = if use_log2 { p.log2() } else { p.ln() };
            h -= p * log_p;
        }
    }
    // Guard against a tiny negative value from floating-point rounding.
    if h < 0.0 { 0.0 } else { h }
}

/// Natural-or-base-2 logarithm of a count, used as the entropy normaliser.
fn log_count(count: usize, use_log2: bool) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let c = count as f32;
    if use_log2 { c.log2() } else { c.ln() }
}

// ── SemanticEntropyEstimator ──────────────────────────────────────────────────

/// Estimates uncertainty by clustering sampled answers by meaning and computing
/// the entropy of the resulting cluster distribution (Kuhn et al. 2023).
///
/// Unlike pairwise conflict typing or claim-support scoring, this estimator asks
/// a single question: *how many distinct meanings did the model produce, and how
/// is probability mass spread across them?*  A confident model collapses its
/// samples into one meaning (entropy `0.0`); an uncertain one spreads them across
/// many (entropy approaching `log(k)`).
#[derive(Debug, Clone)]
pub struct SemanticEntropyEstimator {
    /// Configuration controlling the equivalence threshold and log base.
    pub config: SemanticEntropyConfig,
}

impl SemanticEntropyEstimator {
    /// Construct an estimator with the given configuration.
    #[must_use]
    pub fn new(config: SemanticEntropyConfig) -> Self {
        Self { config }
    }

    /// Cluster `answers` into meaning groups by greedy single-linkage over the
    /// bidirectional-entailment stand-in.
    ///
    /// Each answer joins the first existing cluster whose representative it is
    /// semantically equivalent to; otherwise it seeds a new cluster.  The
    /// returned clusters carry uniform-weight probabilities (each member counts
    /// as `1`); use [`estimate`](Self::estimate) for the full result and
    /// [`estimate_weighted`](Self::estimate_weighted) for custom weights.
    ///
    /// The result is deterministic: clusters are returned in first-seen order
    /// and members in ascending index order.
    #[must_use]
    pub fn cluster_answers(&self, answers: &[String]) -> Vec<MeaningCluster> {
        let weights = vec![1.0_f32; answers.len()];
        self.cluster_internal(answers, &weights)
    }

    /// Cluster answers using explicit per-answer weights.
    ///
    /// `weights[i]` is the sample weight of `answers[i]`; cluster probabilities
    /// are the summed member weights divided by the total weight.
    fn cluster_internal(&self, answers: &[String], weights: &[f32]) -> Vec<MeaningCluster> {
        let threshold = self.config.equivalence_threshold;
        let token_sets: Vec<HashSet<String>> = answers.iter().map(|a| token_set(a)).collect();

        // Greedy single-linkage: each cluster keyed by its representative index.
        let mut rep_sets: Vec<&HashSet<String>> = Vec::new();
        let mut member_lists: Vec<Vec<usize>> = Vec::new();
        let mut representatives: Vec<String> = Vec::new();

        for (i, set_i) in token_sets.iter().enumerate() {
            let mut placed = false;
            for (cluster_idx, rep_set) in rep_sets.iter().enumerate() {
                if semantically_equivalent(set_i, rep_set, threshold) {
                    member_lists[cluster_idx].push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                rep_sets.push(set_i);
                member_lists.push(vec![i]);
                representatives.push(answers[i].clone());
            }
        }

        let total_weight: f32 = weights.iter().copied().sum();
        let mut clusters = Vec::with_capacity(member_lists.len());
        for (members, representative) in member_lists.into_iter().zip(representatives) {
            let cluster_weight: f32 = members.iter().map(|&idx| weights[idx]).sum();
            let probability = if total_weight > 0.0 {
                cluster_weight / total_weight
            } else {
                0.0
            };
            clusters.push(MeaningCluster {
                representative,
                members,
                probability,
            });
        }
        clusters
    }

    /// Estimate semantic entropy over uniformly weighted `answers`.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticEntropyError::EmptySamples`] when `answers` is empty.
    pub fn estimate(
        &self,
        answers: &[String],
    ) -> Result<SemanticEntropyResult, SemanticEntropyError> {
        let weights = vec![1.0_f32; answers.len()];
        self.estimate_weighted(answers, &weights)
    }

    /// Estimate semantic entropy over `answers` with explicit per-answer weights.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticEntropyError::EmptySamples`] when `answers` is empty.
    /// Returns [`SemanticEntropyError::WeightMismatch`] when `weights.len()` does
    /// not equal `answers.len()`.
    pub fn estimate_weighted(
        &self,
        answers: &[String],
        weights: &[f32],
    ) -> Result<SemanticEntropyResult, SemanticEntropyError> {
        if answers.is_empty() {
            return Err(SemanticEntropyError::EmptySamples);
        }
        if weights.len() != answers.len() {
            return Err(SemanticEntropyError::WeightMismatch {
                weights: weights.len(),
                answers: answers.len(),
            });
        }

        let use_log2 = self.config.use_log2;
        let clusters = self.cluster_internal(answers, weights);

        // Discrete semantic entropy over the cluster probability distribution.
        let probs: Vec<f32> = clusters.iter().map(|c| c.probability).collect();
        let entropy = shannon_entropy(&probs, use_log2);
        let num_clusters = clusters.len();
        let normalized_entropy = if num_clusters > 1 {
            let denom = log_count(num_clusters, use_log2);
            if denom > 0.0 {
                (entropy / denom).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Naive lexical (predictive) entropy: each answer is its own cluster.
        let total_weight: f32 = weights.iter().copied().sum();
        let naive_probs: Vec<f32> = if total_weight > 0.0 {
            weights.iter().map(|&w| w / total_weight).collect()
        } else {
            Vec::new()
        };
        let predictive_entropy = shannon_entropy(&naive_probs, use_log2);

        Ok(SemanticEntropyResult {
            entropy,
            normalized_entropy,
            num_clusters,
            num_samples: answers.len(),
            clusters,
            predictive_entropy,
        })
    }

    /// Returns `true` when the answer set is uncertain: the
    /// [`normalized_entropy`](SemanticEntropyResult::normalized_entropy) exceeds
    /// `threshold`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`estimate`](Self::estimate).
    pub fn is_uncertain(
        &self,
        answers: &[String],
        threshold: f32,
    ) -> Result<bool, SemanticEntropyError> {
        let result = self.estimate(answers)?;
        Ok(result.normalized_entropy > threshold)
    }
}

impl Default for SemanticEntropyEstimator {
    fn default() -> Self {
        Self::new(SemanticEntropyConfig::default())
    }
}
