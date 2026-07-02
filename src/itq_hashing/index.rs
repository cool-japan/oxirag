//! An in-memory ITQ index: train a hasher on a corpus, encode every vector to
//! a binary code, and run brute-force top-`k` search ranked by Hamming
//! distance.

use super::hasher::ItqHasher;
use super::types::{ItqCode, ItqConfig, ItqError, ItqHit};

// ── ItqIndex ──────────────────────────────────────────────────────────────────

/// An in-memory index of ITQ binary codes.
///
/// [`build`](Self::build) trains an internal [`ItqHasher`] on the supplied
/// vectors (learning the PCA basis and the binary-coding rotation) and stores
/// one [`ItqCode`] per item. [`search`](Self::search) encodes the query and
/// ranks every stored code by ascending Hamming distance.
#[derive(Debug, Clone)]
pub struct ItqIndex {
    hasher: ItqHasher,
    entries: Vec<(String, ItqCode)>,
}

impl ItqIndex {
    /// Train on `items` and encode every vector, producing a fully built index.
    ///
    /// # Errors
    ///
    /// - [`ItqError::EmptyDataset`] when `items` is empty.
    /// - Any error propagated from [`ItqHasher::train`] (invalid config,
    ///   dimension mismatch, non-finite input, `num_bits > D`, or an internal
    ///   numerical failure) and from [`ItqHasher::encode`].
    pub fn build(items: Vec<(String, Vec<f32>)>, config: ItqConfig) -> Result<Self, ItqError> {
        if items.is_empty() {
            return Err(ItqError::EmptyDataset);
        }
        let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
        let hasher = ItqHasher::train(&vectors, &config)?;

        let mut entries = Vec::with_capacity(items.len());
        for (id, v) in items {
            let code = hasher.encode(&v)?;
            entries.push((id, code));
        }
        Ok(Self { hasher, entries })
    }

    /// Borrow the underlying (trained) hasher.
    #[must_use]
    pub fn hasher(&self) -> &ItqHasher {
        &self.hasher
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

    /// Search the index for the `k` nearest items to `query`, ranked by
    /// ascending Hamming distance between binary codes.
    ///
    /// Ties in Hamming distance are broken by the order in which items were
    /// supplied to [`build`](Self::build).
    ///
    /// # Errors
    ///
    /// - [`ItqError::EmptyQuery`] when `query` is empty.
    /// - [`ItqError::EmptyDataset`] when the index holds no items.
    /// - [`ItqError::DimensionMismatch`] when `query.len()` does not match the
    ///   trained dimensionality.
    /// - [`ItqError::NonFinite`] when `query` holds a `NaN`/infinite value.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<ItqHit>, ItqError> {
        if query.is_empty() {
            return Err(ItqError::EmptyQuery);
        }
        if self.entries.is_empty() {
            return Err(ItqError::EmptyDataset);
        }

        let query_code = self.hasher.encode(query)?;
        let mut hits: Vec<ItqHit> = self
            .entries
            .iter()
            .map(|(id, code)| ItqHit::new(id.clone(), code.hamming_distance(&query_code)))
            .collect();

        hits.sort_by_key(|hit| hit.hamming_distance);
        hits.truncate(k);
        Ok(hits)
    }
}
