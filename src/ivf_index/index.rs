//! The IVF index itself: coarse quantizer training, inverted lists and search.
//!
//! The index implements an Inverted-File (IVF) approximate-nearest-neighbour
//! structure. Training runs deterministic Lloyd k-means with a *spread*
//! initialisation to produce `num_cells` centroids; every indexed vector is then
//! filed into the inverted list of its nearest centroid. A query locates the
//! `nprobe` nearest centroids and scans only those lists, ranking candidates by
//! squared L2 distance.

use crate::types::DocumentId;

use super::types::{IvfConfig, IvfError, IvfHit};

/// An Inverted-File (IVF) approximate-nearest-neighbour index.
///
/// # Lifecycle
///
/// 1. Construct with [`IvfIndex::new`].
/// 2. Train the coarse quantizer with [`IvfIndex::train`] (or use
///    [`IvfIndex::build`] to train and populate in one call).
/// 3. Insert vectors with [`IvfIndex::add`].
/// 4. Query with [`IvfIndex::search`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "ivf-index")] {
/// use oxirag::ivf_index::{IvfConfig, IvfIndex};
/// use oxirag::types::DocumentId;
///
/// let config = IvfConfig::new().with_num_cells(2).with_nprobe(2).with_dim(2);
/// let mut index = IvfIndex::new(config);
/// index
///     .build(&[
///         (DocumentId::from_string("a"), vec![0.0, 0.0]),
///         (DocumentId::from_string("b"), vec![10.0, 10.0]),
///     ])
///     .unwrap();
///
/// let hits = index.search(&[0.1, 0.1], 1).unwrap();
/// assert_eq!(hits[0].id, DocumentId::from_string("a"));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct IvfIndex {
    config: IvfConfig,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<(DocumentId, Vec<f32>)>>,
    trained: bool,
}

impl IvfIndex {
    /// Create a new, untrained index with the given configuration.
    #[must_use]
    pub fn new(config: IvfConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            lists: Vec::new(),
            trained: false,
        }
    }

    /// Borrow the configuration backing this index.
    #[must_use]
    pub fn config(&self) -> &IvfConfig {
        &self.config
    }

    /// Number of coarse-quantizer cells actually built.
    ///
    /// This equals `config.num_cells` after a normal training run, but may be
    /// smaller when the training set contained fewer distinct vectors than
    /// requested cells. Returns `0` before training.
    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.centroids.len()
    }

    /// Whether the coarse quantizer has been trained.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Total number of vectors stored across all inverted lists.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lists.iter().map(Vec::len).sum()
    }

    /// Whether the index currently holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the inverted list for a given cell, if it exists.
    #[must_use]
    pub fn list(&self, cell: usize) -> Option<&[(DocumentId, Vec<f32>)]> {
        self.lists.get(cell).map(Vec::as_slice)
    }

    /// Train the coarse quantizer on `vectors`.
    ///
    /// Runs deterministic Lloyd k-means with a spread initialisation to build up
    /// to `config.num_cells` centroids and allocates the (empty) inverted lists.
    /// Any vectors previously added are discarded.
    ///
    /// # Errors
    ///
    /// Returns [`IvfError::EmptyTrainingSet`] when `vectors` is empty, or
    /// [`IvfError::DimMismatch`] when any vector's length differs from
    /// `config.dim`.
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<(), IvfError> {
        if vectors.is_empty() {
            return Err(IvfError::EmptyTrainingSet);
        }
        for v in vectors {
            if v.len() != self.config.dim {
                return Err(IvfError::DimMismatch);
            }
        }

        let k = self.config.num_cells.max(1).min(vectors.len());
        self.centroids = kmeans(vectors, k, self.config.kmeans_iters, self.config.dim);
        self.lists = vec![Vec::new(); self.centroids.len()];
        self.trained = true;
        Ok(())
    }

    /// File `vector` into the inverted list of its nearest centroid.
    ///
    /// # Errors
    ///
    /// Returns [`IvfError::NotTrained`] if [`train`](IvfIndex::train) has not run,
    /// or [`IvfError::DimMismatch`] when `vector.len() != config.dim`.
    pub fn add(&mut self, id: DocumentId, vector: Vec<f32>) -> Result<(), IvfError> {
        let cell = self.cell_of(&vector)?;
        self.lists[cell].push((id, vector));
        Ok(())
    }

    /// Train on, then add, every item in `items`.
    ///
    /// Equivalent to calling [`train`](IvfIndex::train) with the item vectors
    /// followed by [`add`](IvfIndex::add) for each item.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`train`](IvfIndex::train) or
    /// [`add`](IvfIndex::add).
    pub fn build(&mut self, items: &[(DocumentId, Vec<f32>)]) -> Result<(), IvfError> {
        let vectors: Vec<Vec<f32>> = items.iter().map(|(_, v)| v.clone()).collect();
        self.train(&vectors)?;
        for (id, vector) in items {
            self.add(id.clone(), vector.clone())?;
        }
        Ok(())
    }

    /// Return the index of the nearest centroid (cell) to `vector`.
    ///
    /// # Errors
    ///
    /// Returns [`IvfError::NotTrained`] if the quantizer is untrained, or
    /// [`IvfError::DimMismatch`] when `vector.len() != config.dim`.
    pub fn cell_of(&self, vector: &[f32]) -> Result<usize, IvfError> {
        if !self.trained {
            return Err(IvfError::NotTrained);
        }
        if vector.len() != self.config.dim {
            return Err(IvfError::DimMismatch);
        }
        Ok(nearest_centroid(vector, &self.centroids))
    }

    /// Search for the `top_k` nearest neighbours of `query`.
    ///
    /// Locates the `nprobe` nearest centroids, scans only their inverted lists
    /// and returns the matches with the smallest squared L2 distance in ascending
    /// order. At most `top_k` hits are returned; fewer if the probed lists hold
    /// fewer candidates.
    ///
    /// # Errors
    ///
    /// Returns [`IvfError::NotTrained`] if the quantizer is untrained, or
    /// [`IvfError::DimMismatch`] when `query.len() != config.dim`.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<IvfHit>, IvfError> {
        if !self.trained {
            return Err(IvfError::NotTrained);
        }
        if query.len() != self.config.dim {
            return Err(IvfError::DimMismatch);
        }
        if top_k == 0 || self.centroids.is_empty() {
            return Ok(Vec::new());
        }

        let probe_cells = self.nearest_cells(query);

        let mut candidates: Vec<IvfHit> = Vec::new();
        for cell in probe_cells {
            for (id, vector) in &self.lists[cell] {
                candidates.push(IvfHit::new(id.clone(), l2_squared(query, vector)));
            }
        }

        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
        Ok(candidates)
    }

    /// Return the indices of the `nprobe` nearest centroids to `query`,
    /// nearest first. `nprobe` is clamped to the number of available cells.
    fn nearest_cells(&self, query: &[f32]) -> Vec<usize> {
        let nprobe = self.config.nprobe.max(1).min(self.centroids.len());
        let mut scored: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(ci, c)| (l2_squared(query, c), ci))
            .collect();
        scored.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        scored.into_iter().take(nprobe).map(|(_, ci)| ci).collect()
    }
}

// ── Distance helpers ──────────────────────────────────────────────────────────

/// Squared L2 distance between two equal-length vectors.
///
/// Mismatched lengths fall back to the shared prefix, which never occurs on the
/// validated paths inside this module.
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Index of the centroid nearest to `vector` by squared L2 distance.
///
/// Ties are broken by the lower centroid index for determinism.
fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (ci, c) in centroids.iter().enumerate() {
        let d = l2_squared(vector, c);
        if d < best_dist {
            best_dist = d;
            best = ci;
        }
    }
    best
}

// ── Deterministic k-means ─────────────────────────────────────────────────────

/// Deterministic Lloyd k-means producing `k` centroids of dimension `dim`.
///
/// Initialisation is a *spread* seeding: the `k` seeds are drawn at evenly
/// spaced positions across the (input order of the) training set, giving a
/// reproducible, well-separated starting configuration without any randomness.
/// Assignment ties and empty clusters are resolved deterministically so repeated
/// runs on identical input yield identical centroids.
fn kmeans(vectors: &[Vec<f32>], k: usize, iters: usize, dim: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let k = k.max(1).min(n);

    // Spread initialisation: evenly spaced seeds across the input order.
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|ci| {
            let idx = if k == 1 { 0 } else { (ci * (n - 1)) / (k - 1) };
            vectors[idx.min(n - 1)].clone()
        })
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..iters.max(1) {
        // Assignment step.
        let mut changed = false;
        for (i, v) in vectors.iter().enumerate() {
            let best = nearest_centroid(v, &centroids);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update step.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &ci) in assignments.iter().enumerate() {
            for d in 0..dim {
                sums[ci][d] += vectors[i][d];
            }
            counts[ci] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                #[allow(clippy::cast_precision_loss)]
                let cnt = counts[ci] as f32;
                for x in &mut sums[ci] {
                    *x /= cnt;
                }
                centroids[ci].clone_from(&sums[ci]);
            }
            // Empty clusters keep their previous centroid (deterministic).
        }
    }

    centroids
}
