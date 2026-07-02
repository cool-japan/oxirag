//! A residual-quantized vector index with additive-lookup asymmetric search.

use super::quantizer::ResidualQuantizer;
use super::types::{ResidualCode, ResidualVqConfig, ResidualVqError, ResidualVqHit};
use crate::types::DocumentId;

/// One stored database entry: a document id, its residual code, and the
/// precomputed squared norm of its reconstruction (the `‖x̂‖²` term reused by
/// every asymmetric distance estimate).
#[derive(Debug, Clone)]
struct IndexEntry {
    id: DocumentId,
    code: ResidualCode,
    recon_norm_sq: f32,
}

/// Squared Euclidean (L2) distance between two equal-length slices.
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ── ResidualVqIndex ─────────────────────────────────────────────────────────────

/// An in-memory index of residual-quantized vectors.
///
/// [`build`](Self::build) trains the internal [`ResidualQuantizer`] on the
/// supplied vectors, encodes every item to a [`ResidualCode`], and caches each
/// reconstruction's squared norm. [`search`](Self::search) scores the query
/// against every stored code using table-driven asymmetric distance
/// computation — precompute per-query inner-product tables once, then estimate
/// each database distance with `M` lookups plus the cached norm — and returns
/// the closest matches in ascending distance.
#[derive(Debug, Clone)]
pub struct ResidualVqIndex {
    quantizer: ResidualQuantizer,
    entries: Vec<IndexEntry>,
}

impl ResidualVqIndex {
    /// Create an empty, untrained index for the given configuration.
    #[must_use]
    pub fn new(config: ResidualVqConfig) -> Self {
        Self {
            quantizer: ResidualQuantizer::new(config),
            entries: Vec::new(),
        }
    }

    /// Borrow the underlying quantizer.
    #[must_use]
    pub fn quantizer(&self) -> &ResidualQuantizer {
        &self.quantizer
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

    /// Train on the supplied vectors, then encode and store every item.
    ///
    /// Encoding uses the configured
    /// [`beam_width`](ResidualVqConfig::beam_width). Re-building replaces any
    /// previously indexed codes.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::EmptyTrainingSet`] when `items` is empty.
    /// - [`ResidualVqError::InvalidConfig`] / [`ResidualVqError::DimMismatch`]
    ///   propagated from [`ResidualQuantizer::train`] and
    ///   [`ResidualQuantizer::encode`].
    pub fn build(&mut self, items: &[(DocumentId, Vec<f32>)]) -> Result<(), ResidualVqError> {
        if items.is_empty() {
            return Err(ResidualVqError::EmptyTrainingSet);
        }
        let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
        self.quantizer.train(&vectors)?;

        let mut entries = Vec::with_capacity(items.len());
        for (id, vector) in items {
            let code = self.quantizer.encode(vector)?;
            let recon_norm_sq = self.quantizer.reconstruction_norm_sq(&code);
            entries.push(IndexEntry {
                id: id.clone(),
                code,
                recon_norm_sq,
            });
        }
        self.entries = entries;
        Ok(())
    }

    /// Search for the `top_k` nearest documents to `query` (asymmetric).
    ///
    /// Builds the per-query inner-product tables once, then scores every stored
    /// code as `‖q‖² + ‖x̂‖² − 2 Σₘ ⟨q, codewordₘ⟩` using `M` table lookups plus
    /// the cached reconstruction norm — no database vector is reconstructed at
    /// query time. Hits are returned sorted by ascending (squared-L2) distance.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when the index has not been
    ///   [`build`](Self::build)-ed.
    /// - [`ResidualVqError::DimMismatch`] when `query.len()` does not match the
    ///   configured dimensionality.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<ResidualVqHit>, ResidualVqError> {
        if !self.quantizer.is_trained() {
            return Err(ResidualVqError::NotTrained);
        }
        let tables = self.quantizer.inner_product_tables(query)?;
        let query_norm_sq = ResidualQuantizer::query_norm_sq(query);

        let mut hits: Vec<ResidualVqHit> = self
            .entries
            .iter()
            .map(|entry| {
                let distance = ResidualQuantizer::distance_from_ip_tables(
                    &tables,
                    &entry.code,
                    query_norm_sq,
                    entry.recon_norm_sq,
                );
                ResidualVqHit::new(entry.id.clone(), distance)
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

    /// Search for the `top_k` nearest documents to `query` (symmetric).
    ///
    /// Encodes the query, reconstructs it, and compares against every stored
    /// reconstruction. Both sides carry quantization error, so this is usually
    /// less accurate than [`search`](Self::search); it is provided for
    /// comparison and for callers that only keep quantized queries.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when the index has not been
    ///   [`build`](Self::build)-ed.
    /// - [`ResidualVqError::DimMismatch`] when `query.len()` does not match the
    ///   configured dimensionality.
    pub fn search_symmetric(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<ResidualVqHit>, ResidualVqError> {
        if !self.quantizer.is_trained() {
            return Err(ResidualVqError::NotTrained);
        }
        let query_code = self.quantizer.encode(query)?;
        let query_recon = self.quantizer.decode(&query_code);

        let mut hits: Vec<ResidualVqHit> = self
            .entries
            .iter()
            .map(|entry| {
                let recon = self.quantizer.decode(&entry.code);
                let distance = squared_l2(&query_recon, &recon);
                ResidualVqHit::new(entry.id.clone(), distance)
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
