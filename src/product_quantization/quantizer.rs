//! The product quantizer: codebook training, encoding, decoding and ADC.

use super::types::{PqCode, PqConfig, PqError};

// ── Distance helper ───────────────────────────────────────────────────────────

/// Squared Euclidean (L2) distance between two equal-length slices.
///
/// Lengths are assumed equal by the caller; only the shared prefix is summed.
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ── ProductQuantizer ──────────────────────────────────────────────────────────

/// Product quantizer over `M = num_subspaces` subspaces.
///
/// After [`train`](Self::train), `codebooks[m][k]` holds the `k`-th centroid of
/// subspace `m`; each centroid is a `subspace_dim`-length vector. A vector is
/// [`encode`](Self::encode)d to one centroid index per subspace and scored
/// against a query via Asymmetric Distance Computation
/// ([`asymmetric_distance`](Self::asymmetric_distance)).
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    config: PqConfig,
    /// Codebooks shaped `M × K × subspace_dim`; empty until trained.
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Create an untrained quantizer for the given configuration.
    ///
    /// The configuration is validated lazily on [`train`](Self::train); this
    /// constructor never fails.
    #[must_use]
    pub fn new(config: PqConfig) -> Self {
        Self {
            config,
            codebooks: Vec::new(),
        }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &PqConfig {
        &self.config
    }

    /// Return `true` once [`train`](Self::train) has produced codebooks.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        !self.codebooks.is_empty()
    }

    /// Number of subspaces (`M`).
    #[must_use]
    pub fn num_subspaces(&self) -> usize {
        self.config.num_subspaces
    }

    /// Borrow the trained codebooks (`M × K × subspace_dim`).
    ///
    /// Empty until [`train`](Self::train) has run.
    #[must_use]
    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    /// Train one codebook per subspace via deterministic k-means-lite.
    ///
    /// Each subspace is clustered into `K = 2^codebook_bits` centroids. Initial
    /// centroids are seeded by spreading sample indices evenly across the input
    /// (never randomly) so that identical inputs yield identical codebooks.
    ///
    /// # Errors
    ///
    /// - [`PqError::InvalidConfig`] / [`PqError::DimMismatch`] when the
    ///   configuration is invalid (see [`PqConfig::validate`]).
    /// - [`PqError::EmptyTrainingSet`] when `vectors` is empty.
    /// - [`PqError::DimMismatch`] when any vector length differs from
    ///   [`PqConfig::dim`].
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<(), PqError> {
        self.config.validate()?;
        if vectors.is_empty() {
            return Err(PqError::EmptyTrainingSet);
        }
        for v in vectors {
            if v.len() != self.config.dim {
                return Err(PqError::DimMismatch);
            }
        }

        let m = self.config.num_subspaces;
        let sub_dim = self.config.subspace_dim();
        let k = self.config.codebook_size();

        let mut codebooks = Vec::with_capacity(m);
        for s in 0..m {
            let start = s * sub_dim;
            let end = start + sub_dim;
            // Project every training vector onto this subspace.
            let sub_vectors: Vec<&[f32]> = vectors.iter().map(|v| &v[start..end]).collect();
            let centroids =
                Self::train_subspace(&sub_vectors, k, sub_dim, self.config.kmeans_iters);
            codebooks.push(centroids);
        }
        self.codebooks = codebooks;
        Ok(())
    }

    /// Run deterministic k-means-lite on one subspace, returning `k` centroids.
    fn train_subspace(
        sub_vectors: &[&[f32]],
        k: usize,
        sub_dim: usize,
        iters: usize,
    ) -> Vec<Vec<f32>> {
        let n = sub_vectors.len();
        // Effective centroid count: never exceed the number of points.
        let effective_k = k.min(n).max(1);

        // Deterministic seeding: spread indices evenly across the sample set.
        let mut centroids: Vec<Vec<f32>> = (0..effective_k)
            .map(|ci| {
                let idx = if effective_k == 1 {
                    0
                } else {
                    (ci * (n - 1)) / (effective_k - 1)
                };
                sub_vectors[idx].to_vec()
            })
            .collect();

        let mut assignments = vec![0usize; n];

        for _ in 0..iters {
            // ── Assignment step ──
            let mut changed = false;
            for (i, sv) in sub_vectors.iter().enumerate() {
                let mut best = 0usize;
                let mut best_dist = f32::INFINITY;
                for (ci, centroid) in centroids.iter().enumerate() {
                    let dist = squared_l2(sv, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best = ci;
                    }
                }
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }

            // ── Update step ──
            let mut sums = vec![vec![0.0f32; sub_dim]; effective_k];
            let mut counts = vec![0usize; effective_k];
            for (i, sv) in sub_vectors.iter().enumerate() {
                let ci = assignments[i];
                counts[ci] += 1;
                for (acc, &val) in sums[ci].iter_mut().zip(sv.iter()) {
                    *acc += val;
                }
            }
            for ci in 0..effective_k {
                if counts[ci] > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    let inv = 1.0 / counts[ci] as f32;
                    for (slot, &acc) in centroids[ci].iter_mut().zip(sums[ci].iter()) {
                        *slot = acc * inv;
                    }
                }
                // Empty centroid: leave it where it was (deterministic).
            }
        }

        // Pad up to the full codebook size so every code index is valid.
        // Duplicated trailing centroids never win the nearest-centroid race
        // against the originals, keeping encoding deterministic.
        while centroids.len() < k {
            let fill = centroids[centroids.len() % effective_k].clone();
            centroids.push(fill);
        }
        centroids
    }

    /// Encode a vector to one centroid index per subspace (nearest centroid).
    ///
    /// # Errors
    ///
    /// - [`PqError::NotTrained`] when called before [`train`](Self::train).
    /// - [`PqError::DimMismatch`] when `vector.len() != dim`.
    pub fn encode(&self, vector: &[f32]) -> Result<PqCode, PqError> {
        if !self.is_trained() {
            return Err(PqError::NotTrained);
        }
        if vector.len() != self.config.dim {
            return Err(PqError::DimMismatch);
        }

        let sub_dim = self.config.subspace_dim();
        let mut codes = Vec::with_capacity(self.config.num_subspaces);
        for (s, codebook) in self.codebooks.iter().enumerate() {
            let start = s * sub_dim;
            let sub = &vector[start..start + sub_dim];
            let mut best = 0usize;
            let mut best_dist = f32::INFINITY;
            for (ci, centroid) in codebook.iter().enumerate() {
                let dist = squared_l2(sub, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best = ci;
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            codes.push(best as u8);
        }
        Ok(PqCode::new(codes))
    }

    /// Reconstruct an approximate vector by concatenating the selected
    /// centroids.
    ///
    /// When the quantizer is untrained or `code` has fewer entries than there
    /// are subspaces, the corresponding output region is left as zeros.
    #[must_use]
    pub fn decode(&self, code: &PqCode) -> Vec<f32> {
        let sub_dim = self.config.subspace_dim();
        let mut out = vec![0.0f32; self.config.dim];
        if !self.is_trained() {
            return out;
        }
        for (s, codebook) in self.codebooks.iter().enumerate() {
            let Some(&ci) = code.codes.get(s) else {
                break;
            };
            let centroid = &codebook[ci as usize];
            let start = s * sub_dim;
            out[start..start + sub_dim].copy_from_slice(centroid);
        }
        out
    }

    /// Asymmetric Distance Computation (ADC): squared-L2 distance between a
    /// full-precision `query` and a quantized `code`.
    ///
    /// Precomputes, per subspace, the squared distance from the query
    /// subvector to every centroid, then sums the looked-up distance for each
    /// code entry. The result equals the squared L2 distance between `query`
    /// and `decode(code)`.
    ///
    /// # Errors
    ///
    /// - [`PqError::NotTrained`] when called before [`train`](Self::train).
    /// - [`PqError::DimMismatch`] when `query.len() != dim` or `code` does not
    ///   span every subspace.
    pub fn asymmetric_distance(&self, query: &[f32], code: &PqCode) -> Result<f32, PqError> {
        if !self.is_trained() {
            return Err(PqError::NotTrained);
        }
        if query.len() != self.config.dim || code.codes.len() != self.config.num_subspaces {
            return Err(PqError::DimMismatch);
        }

        let sub_dim = self.config.subspace_dim();
        let mut total = 0.0f32;
        for (s, codebook) in self.codebooks.iter().enumerate() {
            let start = s * sub_dim;
            let sub_query = &query[start..start + sub_dim];
            // Distance table lookup for this subspace's selected centroid.
            let ci = code.codes[s] as usize;
            total += squared_l2(sub_query, &codebook[ci]);
        }
        Ok(total)
    }

    /// Build the ADC distance tables for `query`: `tables[m][k]` is the squared
    /// distance from the query's `m`-th subvector to centroid `k` of subspace
    /// `m`.
    ///
    /// # Errors
    ///
    /// - [`PqError::NotTrained`] when called before [`train`](Self::train).
    /// - [`PqError::DimMismatch`] when `query.len() != dim`.
    pub fn distance_tables(&self, query: &[f32]) -> Result<Vec<Vec<f32>>, PqError> {
        if !self.is_trained() {
            return Err(PqError::NotTrained);
        }
        if query.len() != self.config.dim {
            return Err(PqError::DimMismatch);
        }

        let sub_dim = self.config.subspace_dim();
        let mut tables = Vec::with_capacity(self.config.num_subspaces);
        for (s, codebook) in self.codebooks.iter().enumerate() {
            let start = s * sub_dim;
            let sub_query = &query[start..start + sub_dim];
            let row: Vec<f32> = codebook
                .iter()
                .map(|centroid| squared_l2(sub_query, centroid))
                .collect();
            tables.push(row);
        }
        Ok(tables)
    }

    /// Sum precomputed [`distance_tables`](Self::distance_tables) for a `code`.
    ///
    /// This is the table-driven core of ADC; callers that score many codes
    /// against one query build the tables once and reuse them here.
    #[must_use]
    pub fn distance_from_tables(tables: &[Vec<f32>], code: &PqCode) -> f32 {
        tables
            .iter()
            .zip(code.codes.iter())
            .map(|(row, &ci)| row[ci as usize])
            .sum()
    }
}
