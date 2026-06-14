//! Nested embedding encoder for Matryoshka Representation Learning.
//!
//! Builds a deterministic FNV-1a token-histogram embedding and applies a
//! per-dimension importance decay so that earlier dimensions dominate. Because
//! the front of the vector carries the most information, any prefix obtained by
//! [`MatryoshkaEmbedding::truncate`] is itself a usable, coarser embedding.

use super::types::MatryoshkaConfig;

// ── MatryoshkaEmbedding ───────────────────────────────────────────────────────

/// A nested ("Russian doll") embedding whose prefixes are valid embeddings.
///
/// The full vector is stored L2-normalised. Truncating the first `dim`
/// components and re-normalising yields a coarser embedding of the same input.
#[derive(Debug, Clone, PartialEq)]
pub struct MatryoshkaEmbedding {
    /// The full, L2-normalised embedding vector.
    pub full: Vec<f32>,
}

impl MatryoshkaEmbedding {
    /// Number of components in the full embedding.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.full.len()
    }

    /// Return the first `min(dim, dims())` components, re-normalised to unit
    /// length.
    ///
    /// This realises the nested-prefix property: the result is a valid
    /// lower-dimensional embedding. Because each prefix is independently
    /// re-normalised, `truncate(d1)` and `truncate(d2)` share the prefix
    /// relationship of the underlying unnormalised vector.
    ///
    /// A `dim` of `0` yields an empty vector; a zero-magnitude prefix is
    /// returned as zeros of the requested length.
    #[must_use]
    pub fn truncate(&self, dim: usize) -> Vec<f32> {
        let n = dim.min(self.full.len());
        let mut prefix = self.full[..n].to_vec();
        let norm: f32 = prefix.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut prefix {
                *x /= norm;
            }
        }
        prefix
    }
}

// ── MatryoshkaEncoder ─────────────────────────────────────────────────────────

/// Encodes text into nested [`MatryoshkaEmbedding`] vectors.
///
/// The pipeline is fully deterministic: tokenise → FNV-1a bucket histogram of
/// `full_dim` → multiply dimension `i` by `decay.powi(i)` → L2-normalise.
#[derive(Debug, Clone)]
pub struct MatryoshkaEncoder {
    /// Encoder configuration.
    pub config: MatryoshkaConfig,
}

impl MatryoshkaEncoder {
    /// Create a new encoder from the given configuration.
    #[must_use]
    pub fn new(config: MatryoshkaConfig) -> Self {
        Self { config }
    }

    /// Encode `text` into a nested embedding of `config.full_dim` components.
    ///
    /// The returned [`MatryoshkaEmbedding::full`] vector is L2-normalised (or
    /// all-zeros when the text contains no usable tokens).
    #[must_use]
    pub fn encode(&self, text: &str) -> MatryoshkaEmbedding {
        let dim = self.config.full_dim.max(1);
        let mut buckets = vec![0.0f32; dim];
        for token in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(str::to_lowercase)
        {
            // Deterministic hash: FNV-1a.
            let mut h: u64 = 14_695_981_039_346_656_037;
            for b in token.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(1_099_511_628_211);
            }
            #[allow(clippy::cast_possible_truncation)]
            let idx = (h as usize) % dim;
            buckets[idx] += 1.0;
        }

        // Per-dimension importance decay: earlier dimensions dominate so that
        // truncated prefixes remain meaningful.
        let mut weight = 1.0f32;
        for bucket in &mut buckets {
            *bucket *= weight;
            weight *= self.config.decay;
        }

        // L2-normalise the full vector.
        let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut buckets {
                *x /= norm;
            }
        }

        MatryoshkaEmbedding { full: buckets }
    }

    /// Cosine similarity between two equal-length slices.
    ///
    /// The inputs are assumed to be L2-normalised, so this returns their dot
    /// product clamped to `[-1.0, 1.0]`. Mismatched or empty slices score `0.0`.
    #[must_use]
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        dot.clamp(-1.0, 1.0)
    }
}
