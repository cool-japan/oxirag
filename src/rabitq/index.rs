//! An in-memory `RaBitQ` index: build a corpus of quantized codes and run
//! brute-force top-k search over estimated distances.

use super::quantizer::RaBitQuantizer;
use super::types::{RaBitQCode, RaBitQConfig, RaBitQError, RaBitQHit, RaBitQMetric};

// ── RaBitQIndex ───────────────────────────────────────────────────────────────

/// An in-memory index of `RaBitQ`-quantized vectors.
///
/// [`build`](Self::build) trains the internal [`RaBitQuantizer`] on the
/// supplied vectors (computing the centroid and rotation) and stores one
/// [`RaBitQCode`] per item. [`search`](Self::search) scores the query against
/// every stored code according to the configured [`RaBitQMetric`] and returns
/// the closest matches in ascending `estimated_distance` order.
#[derive(Debug, Clone)]
pub struct RaBitQIndex {
    quantizer: RaBitQuantizer,
    entries: Vec<(String, RaBitQCode)>,
}

impl RaBitQIndex {
    /// Train on `items` and encode every vector, producing a fully built
    /// index.
    ///
    /// # Errors
    ///
    /// - [`RaBitQError::EmptyIndex`] when `items` is empty.
    /// - [`RaBitQError::InvalidConfig`] / [`RaBitQError::DimensionMismatch`]
    ///   propagated from [`RaBitQuantizer::train`] and
    ///   [`RaBitQuantizer::encode`].
    pub fn build(
        items: Vec<(String, Vec<f32>)>,
        config: RaBitQConfig,
    ) -> Result<Self, RaBitQError> {
        if items.is_empty() {
            return Err(RaBitQError::EmptyIndex);
        }
        let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
        let quantizer = RaBitQuantizer::new(config).train(&vectors)?;

        let mut entries = Vec::with_capacity(items.len());
        for (id, v) in items {
            let code = quantizer.encode(&v)?;
            entries.push((id, code));
        }
        Ok(Self { quantizer, entries })
    }

    /// Borrow the underlying (trained) quantizer.
    #[must_use]
    pub fn quantizer(&self) -> &RaBitQuantizer {
        &self.quantizer
    }

    /// Number of indexed items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no items are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Search the index for the `k` nearest items to `query`, ranked by the
    /// configured [`RaBitQMetric`]:
    ///
    /// - [`RaBitQMetric::L2`]: ascending [`RaBitQuantizer::estimate_l2_sq`].
    /// - [`RaBitQMetric::InnerProduct`]: ascending negated
    ///   [`RaBitQuantizer::estimate_ip`] (so higher inner product ranks first).
    /// - [`RaBitQMetric::Cosine`]: ascending negated
    ///   [`RaBitQuantizer::estimate_cosine`] (so higher similarity ranks first).
    ///
    /// # Errors
    ///
    /// - [`RaBitQError::EmptyQuery`] when `query` is empty.
    /// - [`RaBitQError::EmptyIndex`] when the index holds no items.
    /// - [`RaBitQError::DimensionMismatch`] when `query.len()` does not match
    ///   the configured dimensionality.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<RaBitQHit>, RaBitQError> {
        if query.is_empty() {
            return Err(RaBitQError::EmptyQuery);
        }
        if self.entries.is_empty() {
            return Err(RaBitQError::EmptyIndex);
        }

        let metric = self.quantizer.config().metric;
        let mut hits = Vec::with_capacity(self.entries.len());
        for (id, code) in &self.entries {
            let distance = match metric {
                RaBitQMetric::L2 => self.quantizer.estimate_l2_sq(query, code)?,
                RaBitQMetric::InnerProduct => -self.quantizer.estimate_ip(query, code)?,
                RaBitQMetric::Cosine => -self.quantizer.estimate_cosine(query, code)?,
            };
            hits.push(RaBitQHit::new(id.clone(), distance));
        }

        hits.sort_by(|a, b| {
            a.estimated_distance
                .partial_cmp(&b.estimated_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }
}
