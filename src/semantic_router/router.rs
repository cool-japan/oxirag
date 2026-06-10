//! FNV-1a pseudo-embedding semantic router implementation.

use super::types::{RouterError, RouterExample, RoutingDecision, SemanticRoutingConfig};

// ── FNV-1a pseudo-embeddings ──────────────────────────────────────────────────

/// Compute a pseudo-embedding for `text` using FNV-1a hashing into a bucket
/// histogram of size `dim`, then L2-normalise the result.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut buckets = vec![0u64; dim];
    for token in text.split_whitespace() {
        let mut h: u64 = FNV_OFFSET;
        for b in token.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    let norm: f64 = buckets
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm < 1e-10 {
        return vec![0.0; dim];
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    buckets.iter().map(|x| (*x as f64 / norm) as f32).collect()
}

/// Compute the cosine similarity between two equal-length slices.
///
/// The result is clamped to `[-1.0, 1.0]` to compensate for floating-point
/// rounding in the L2 normalisation step.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(-1.0, 1.0)
}

// ── SemanticRouter ────────────────────────────────────────────────────────────

/// Embedding-based retrieval-strategy router.
///
/// Uses FNV-1a bucket histograms as lightweight pseudo-embeddings and selects
/// the routing target whose labelled example is most similar to the query.
pub struct SemanticRouter {
    /// Labelled examples used to route queries.
    pub examples: Vec<RouterExample>,
    /// Configuration (threshold, fallback).
    pub config: SemanticRoutingConfig,
}

impl SemanticRouter {
    /// Embedding dimension used for pseudo-vectors.
    const DIM: usize = 64;

    /// Create a new [`SemanticRouter`] with the given examples and config.
    #[must_use]
    pub fn new(examples: Vec<RouterExample>, config: SemanticRoutingConfig) -> Self {
        Self { examples, config }
    }

    /// Route a `query` to the best-matching retrieval strategy.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::EmptyQuery`] when `query` is blank, or
    /// [`RouterError::NoExamples`] when no labelled examples have been provided.
    pub fn route(&self, query: &str) -> Result<RoutingDecision, RouterError> {
        if query.trim().is_empty() {
            return Err(RouterError::EmptyQuery);
        }
        if self.examples.is_empty() {
            return Err(RouterError::NoExamples);
        }

        let query_vec = embed(query, Self::DIM);

        let (best_idx, best_sim) = self
            .examples
            .iter()
            .enumerate()
            .map(|(i, ex)| {
                let ex_vec = embed(&ex.query, Self::DIM);
                (i, cosine(&query_vec, &ex_vec))
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));

        let best_example = &self.examples[best_idx];

        if best_sim >= self.config.threshold {
            Ok(RoutingDecision {
                target: best_example.target.clone(),
                confidence: best_sim,
                reasoning: format!(
                    "Matched example '{}' with similarity {best_sim:.2}",
                    best_example.query
                ),
            })
        } else {
            Ok(RoutingDecision {
                target: self.config.fallback.clone(),
                confidence: best_sim,
                reasoning: format!(
                    "Matched example '{}' with similarity {best_sim:.2}",
                    best_example.query
                ),
            })
        }
    }

    /// Route a `query`, never returning an error.
    ///
    /// If no examples are available or the query is empty the configured
    /// fallback target is returned with confidence `0.0`.
    #[must_use]
    pub fn route_with_fallback(&self, query: &str) -> RoutingDecision {
        if query.trim().is_empty() || self.examples.is_empty() {
            return RoutingDecision {
                target: self.config.fallback.clone(),
                confidence: 0.0,
                reasoning: "No examples or empty query — using fallback".to_string(),
            };
        }
        self.route(query).unwrap_or_else(|_| RoutingDecision {
            target: self.config.fallback.clone(),
            confidence: 0.0,
            reasoning: "Routing failed — using fallback".to_string(),
        })
    }

    /// Append a labelled example to the router.
    pub fn add_example(&mut self, example: RouterExample) {
        self.examples.push(example);
    }
}

impl Default for SemanticRouter {
    fn default() -> Self {
        Self::new(Vec::new(), SemanticRoutingConfig::default())
    }
}
