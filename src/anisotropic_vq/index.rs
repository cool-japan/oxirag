//! An in-memory anisotropic-VQ index: build a corpus of quantized codes and run
//! top-k search over asymmetrically-estimated scores.

use super::quantizer::AnisotropicQuantizer;
use super::types::{
    AnisotropicCode, AnisotropicVqConfig, AnisotropicVqError, AnisotropicVqHit, AnisotropicVqMetric,
};
use crate::types::DocumentId;

// ── AnisotropicVqIndex ────────────────────────────────────────────────────────

/// An in-memory index of anisotropically-quantized vectors.
///
/// [`build`](Self::build) trains the internal [`AnisotropicQuantizer`] on the
/// supplied vectors and stores one [`AnisotropicCode`] per document.
/// [`add`](Self::add) appends a single document to an already-trained index, and
/// [`search`](Self::search) scores the query against every stored code by
/// asymmetric lookup, returning the closest matches in ascending
/// `estimated_distance` order (best first).
#[derive(Debug, Clone)]
pub struct AnisotropicVqIndex {
    quantizer: AnisotropicQuantizer,
    entries: Vec<(DocumentId, AnisotropicCode)>,
}

impl AnisotropicVqIndex {
    /// Create an empty, untrained index for the given configuration.
    #[must_use]
    pub fn new(config: AnisotropicVqConfig) -> Self {
        Self {
            quantizer: AnisotropicQuantizer::new(config),
            entries: Vec::new(),
        }
    }

    /// Borrow the underlying quantizer.
    #[must_use]
    pub fn quantizer(&self) -> &AnisotropicQuantizer {
        &self.quantizer
    }

    /// The ranking metric this index was configured with.
    #[must_use]
    pub fn metric(&self) -> AnisotropicVqMetric {
        self.quantizer.config().metric
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no documents are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Train on the supplied items, then encode and store every one.
    ///
    /// Re-building replaces any previously indexed codes.
    ///
    /// # Errors
    ///
    /// - [`AnisotropicVqError::EmptyTrainingSet`] when `items` is empty.
    /// - [`AnisotropicVqError::InvalidConfig`] / [`AnisotropicVqError::DimMismatch`]
    ///   propagated from [`AnisotropicQuantizer::train`] and
    ///   [`AnisotropicQuantizer::encode`].
    pub fn build(&mut self, items: &[(DocumentId, Vec<f32>)]) -> Result<(), AnisotropicVqError> {
        if items.is_empty() {
            return Err(AnisotropicVqError::EmptyTrainingSet);
        }
        let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
        self.quantizer.train(&vectors)?;

        let mut entries = Vec::with_capacity(items.len());
        for (id, vector) in items {
            let code = self.quantizer.encode(vector)?;
            entries.push((id.clone(), code));
        }
        self.entries = entries;
        Ok(())
    }

    /// Convenience constructor: build an index in one call from `items` and a
    /// `config`.
    ///
    /// # Errors
    ///
    /// As for [`build`](Self::build).
    pub fn build_from(
        items: &[(DocumentId, Vec<f32>)],
        config: AnisotropicVqConfig,
    ) -> Result<Self, AnisotropicVqError> {
        let mut index = Self::new(config);
        index.build(items)?;
        Ok(index)
    }

    /// Encode and append a single document to an already-trained index.
    ///
    /// # Errors
    ///
    /// - [`AnisotropicVqError::NotTrained`] when the index has not been
    ///   [`build`](Self::build)-ed.
    /// - [`AnisotropicVqError::DimMismatch`] when `vector.len()` does not match
    ///   the configured dimensionality.
    pub fn add(&mut self, id: DocumentId, vector: &[f32]) -> Result<(), AnisotropicVqError> {
        let code = self.quantizer.encode(vector)?;
        self.entries.push((id, code));
        Ok(())
    }

    /// Search the index for the `top_k` closest documents to `query`.
    ///
    /// Builds the asymmetric lookup tables once for the query (inner-product or
    /// squared-L2 depending on the configured [`AnisotropicVqMetric`]), scores
    /// every stored code, and returns hits sorted by ascending
    /// `estimated_distance` (best match first).
    ///
    /// # Errors
    ///
    /// - [`AnisotropicVqError::NotTrained`] when the index has not been
    ///   [`build`](Self::build)-ed.
    /// - [`AnisotropicVqError::DimMismatch`] when `query.len()` does not match
    ///   the configured dimensionality.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<AnisotropicVqHit>, AnisotropicVqError> {
        if !self.quantizer.is_trained() {
            return Err(AnisotropicVqError::NotTrained);
        }
        let metric = self.quantizer.config().metric;
        let tables = match metric {
            AnisotropicVqMetric::InnerProduct => self.quantizer.inner_product_tables(query)?,
            AnisotropicVqMetric::L2 => self.quantizer.l2_tables(query)?,
        };

        let mut hits: Vec<AnisotropicVqHit> = self
            .entries
            .iter()
            .map(|(id, code)| {
                let distance =
                    AnisotropicQuantizer::ranked_distance_from_tables(&tables, code, metric);
                AnisotropicVqHit::new(id.clone(), distance)
            })
            .collect();

        hits.sort_by(|a, b| {
            a.estimated_distance
                .partial_cmp(&b.estimated_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.0.cmp(&b.id.0))
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}
