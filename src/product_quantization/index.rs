//! A product-quantized vector index with ADC search.

use super::quantizer::ProductQuantizer;
use super::types::{PqCode, PqConfig, PqError, PqHit};
use crate::types::DocumentId;

// ── PqIndex ───────────────────────────────────────────────────────────────────

/// An in-memory index of product-quantized vectors.
///
/// [`build`](Self::build) trains the internal
/// [`ProductQuantizer`] on the supplied vectors and
/// stores one [`PqCode`] per document. [`search`](Self::search) scores the
/// query against every stored code using table-driven Asymmetric Distance
/// Computation and returns the closest matches in ascending distance.
#[derive(Debug, Clone)]
pub struct PqIndex {
    quantizer: ProductQuantizer,
    codes: Vec<(DocumentId, PqCode)>,
}

impl PqIndex {
    /// Create an empty, untrained index for the given configuration.
    #[must_use]
    pub fn new(config: PqConfig) -> Self {
        Self {
            quantizer: ProductQuantizer::new(config),
            codes: Vec::new(),
        }
    }

    /// Borrow the underlying quantizer.
    #[must_use]
    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.quantizer
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Return `true` when no documents are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Train on the supplied vectors, then encode and store every item.
    ///
    /// Re-building replaces any previously indexed codes.
    ///
    /// # Errors
    ///
    /// - [`PqError::EmptyTrainingSet`] when `items` is empty.
    /// - [`PqError::InvalidConfig`] / [`PqError::DimMismatch`] propagated from
    ///   [`ProductQuantizer::train`] and [`ProductQuantizer::encode`].
    pub fn build(&mut self, items: &[(DocumentId, Vec<f32>)]) -> Result<(), PqError> {
        if items.is_empty() {
            return Err(PqError::EmptyTrainingSet);
        }
        let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
        self.quantizer.train(&vectors)?;

        let mut codes = Vec::with_capacity(items.len());
        for (id, vector) in items {
            let code = self.quantizer.encode(vector)?;
            codes.push((id.clone(), code));
        }
        self.codes = codes;
        Ok(())
    }

    /// Search the index for the `top_k` nearest documents to `query`.
    ///
    /// Builds the ADC distance tables once for the query, scores every stored
    /// code, and returns hits sorted by ascending (squared-L2) distance.
    ///
    /// # Errors
    ///
    /// - [`PqError::NotTrained`] when the index has not been
    ///   [`build`](Self::build)-ed.
    /// - [`PqError::DimMismatch`] when `query.len()` does not match the
    ///   configured dimensionality.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<PqHit>, PqError> {
        if !self.quantizer.is_trained() {
            return Err(PqError::NotTrained);
        }
        let tables = self.quantizer.distance_tables(query)?;

        let mut hits: Vec<PqHit> = self
            .codes
            .iter()
            .map(|(id, code)| {
                let distance = ProductQuantizer::distance_from_tables(&tables, code);
                PqHit::new(id.clone(), distance)
            })
            .collect();

        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}
