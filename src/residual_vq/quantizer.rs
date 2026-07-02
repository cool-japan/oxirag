//! The residual quantizer: cascaded k-means training, greedy / beam encoding,
//! additive reconstruction and additive-lookup distance estimation.
//!
//! # The residual cascade
//!
//! Training fits `M = num_stages` codebooks *sequentially*. Stage `0` runs
//! deterministic Lloyd k-means on the training vectors themselves; every
//! vector is then replaced by its residual (input minus the nearest stage-0
//! centroid). Stage `m` runs k-means on the residuals produced by stage
//! `m - 1`, and so on. Because each stage encodes only what the previous
//! stages failed to capture, a vector is reconstructed as the **additive sum**
//! of one codeword per stage — a much finer approximation than a single
//! codebook of the same size, at the cost of sequential (rather than parallel)
//! encoding.
//!
//! # Monotone reconstruction error
//!
//! [`train`](ResidualQuantizer::train) records the total sum of squared
//! residual norms after every stage in [`stage_errors`](ResidualQuantizer::stage_errors).
//! This sequence is guaranteed non-increasing: each stage's centroids are the
//! means of their assigned residual clusters, and by the variance
//! decomposition `Σ‖r − mean‖² = Σ‖r‖² − Σ nₖ‖meanₖ‖² ≤ Σ‖r‖²`, so a stage can
//! never increase the total residual energy.
//!
//! # Distance estimation
//!
//! Two schemes are provided. The *symmetric* form reconstructs both operands
//! and compares the reconstructions. The *asymmetric* form keeps the query at
//! full precision: it precomputes, per stage, the inner product of the query
//! against every codeword (a lookup table), and estimates the squared-L2
//! distance to any encoded database vector as
//! `‖q‖² + ‖x̂‖² − 2 Σₘ ⟨q, codebookₘ[cₘ]⟩`. The middle term `‖x̂‖²` depends
//! only on the code and is precomputed once per database vector, so scoring a
//! code costs `M` table lookups plus one stored scalar — the whole efficiency
//! point of the technique.

use super::types::{ResidualCode, ResidualVqConfig, ResidualVqError};

// ── FNV-1a deterministic seeding ────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// FNV-1a 64-bit hash of raw bytes.
///
/// Used only to derive a deterministic pseudo-random ordering for k-means seed
/// selection — never for cryptographic purposes. The same bytes always hash to
/// the same value, so training needs no `rand` dependency to be reproducible.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Deterministic FNV-seeded ordering key for the residual at index `idx`.
///
/// Mixes a per-stage `salt`, the residual's position and its full bit pattern
/// so that seed selection is stable across repeated training runs on identical
/// input while still varying with content and stage.
fn fnv_seed_key(vector: &[f32], idx: usize, salt: u64) -> u64 {
    let mut bytes = Vec::with_capacity(16 + vector.len() * 4);
    bytes.extend_from_slice(&salt.to_le_bytes());
    bytes.extend_from_slice(&(idx as u64).to_le_bytes());
    for x in vector {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    fnv1a(&bytes)
}

// ── Vector primitives ───────────────────────────────────────────────────────────

/// Squared Euclidean (L2) distance between two equal-length slices (`f32`).
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Squared L2 norm of a slice (`f32`).
fn norm_sq(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

/// Squared L2 norm accumulated in `f64` for numerically stable error tracking.
fn norm_sq_f64(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x) * f64::from(x)).sum()
}

/// Inner product of two equal-length slices (`f32`).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Index of the centroid nearest to `vector` by squared L2. Ties break toward
/// the lower centroid index for determinism.
fn nearest(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (ci, c) in centroids.iter().enumerate() {
        let dist = squared_l2(vector, c);
        if dist < best_dist {
            best_dist = dist;
            best = ci;
        }
    }
    best
}

// ── ResidualQuantizer ───────────────────────────────────────────────────────────

/// Residual (multi-stage) vector quantizer over `M = num_stages` codebooks.
///
/// After [`train`](Self::train), `codebooks[m][k]` holds the `k`-th codeword of
/// stage `m`; every codeword is a full `dim`-length vector. A vector is
/// [`encode`](Self::encode)d to one codeword index per stage and reconstructed
/// by summing the selected codewords ([`decode`](Self::decode)).
#[derive(Debug, Clone)]
pub struct ResidualQuantizer {
    config: ResidualVqConfig,
    /// Codebooks shaped `M × K × dim`; empty until trained.
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Total sum of squared residual norms after each stage (length `M` once
    /// trained via [`train`](Self::train); empty when built from raw codebooks).
    stage_errors: Vec<f64>,
}

impl ResidualQuantizer {
    /// Create an untrained quantizer for the given configuration.
    ///
    /// The configuration is validated lazily on [`train`](Self::train); this
    /// constructor never fails.
    #[must_use]
    pub fn new(config: ResidualVqConfig) -> Self {
        Self {
            config,
            codebooks: Vec::new(),
            stage_errors: Vec::new(),
        }
    }

    /// Build a trained quantizer directly from precomputed codebooks.
    ///
    /// Useful for reloading persisted codebooks or constructing a quantizer
    /// with hand-specified codewords. The returned quantizer reports no
    /// [`stage_errors`](Self::stage_errors) (those are produced only by
    /// [`train`](Self::train)).
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::InvalidConfig`] when the configuration is invalid,
    ///   when `codebooks.len()` differs from `num_stages`, or when a stage does
    ///   not hold exactly `codebook_size` codewords.
    /// - [`ResidualVqError::DimMismatch`] when any codeword length differs from
    ///   `dim`.
    pub fn from_codebooks(
        config: ResidualVqConfig,
        codebooks: Vec<Vec<Vec<f32>>>,
    ) -> Result<Self, ResidualVqError> {
        config.validate()?;
        if codebooks.len() != config.num_stages {
            return Err(ResidualVqError::InvalidConfig {
                reason: "codebook count must equal num_stages",
            });
        }
        for codebook in &codebooks {
            if codebook.len() != config.codebook_size {
                return Err(ResidualVqError::InvalidConfig {
                    reason: "each stage must hold codebook_size codewords",
                });
            }
            for codeword in codebook {
                if codeword.len() != config.dim {
                    return Err(ResidualVqError::DimMismatch {
                        expected: config.dim,
                        got: codeword.len(),
                    });
                }
            }
        }
        Ok(Self {
            config,
            codebooks,
            stage_errors: Vec::new(),
        })
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &ResidualVqConfig {
        &self.config
    }

    /// Return `true` once training (or [`from_codebooks`](Self::from_codebooks))
    /// has produced codebooks.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        !self.codebooks.is_empty()
    }

    /// Number of sequential stages (`M`).
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.config.num_stages
    }

    /// Number of codewords per stage codebook (`K`).
    #[must_use]
    pub fn codebook_size(&self) -> usize {
        self.config.codebook_size
    }

    /// Input dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Borrow the trained codebooks (`M × K × dim`).
    ///
    /// Empty until [`train`](Self::train) or
    /// [`from_codebooks`](Self::from_codebooks).
    #[must_use]
    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    /// Total sum of squared residual norms after each stage.
    ///
    /// Length equals `num_stages` once [`train`](Self::train) has run, and the
    /// sequence is guaranteed non-increasing. Empty for a quantizer built via
    /// [`from_codebooks`](Self::from_codebooks).
    #[must_use]
    pub fn stage_errors(&self) -> &[f64] {
        &self.stage_errors
    }

    /// Total reconstruction error over the training set after the final stage,
    /// or `None` if the quantizer was not produced by [`train`](Self::train).
    #[must_use]
    pub fn total_error(&self) -> Option<f64> {
        self.stage_errors.last().copied()
    }

    // ── Training ──────────────────────────────────────────────────────────────

    /// Train the `M` codebooks over the residual cascade.
    ///
    /// Stage `0` clusters `vectors`; every subsequent stage clusters the
    /// residuals left by the preceding stage. Centroids are initialised with
    /// deterministic FNV-1a-seeded selection (never randomly), so identical
    /// input yields identical codebooks. The per-stage total squared residual
    /// norm is recorded in [`stage_errors`](Self::stage_errors).
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::InvalidConfig`] when the configuration is invalid
    ///   (see [`ResidualVqConfig::validate`]).
    /// - [`ResidualVqError::EmptyTrainingSet`] when `vectors` is empty.
    /// - [`ResidualVqError::DimMismatch`] when any vector length differs from
    ///   `dim`.
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<(), ResidualVqError> {
        self.config.validate()?;
        if vectors.is_empty() {
            return Err(ResidualVqError::EmptyTrainingSet);
        }
        for v in vectors {
            if v.len() != self.config.dim {
                return Err(ResidualVqError::DimMismatch {
                    expected: self.config.dim,
                    got: v.len(),
                });
            }
        }

        let stages = self.config.num_stages;
        let k = self.config.codebook_size;
        let dim = self.config.dim;
        let iters = self.config.max_kmeans_iterations;

        // Working residual set — starts as a copy of the training vectors.
        let mut residuals: Vec<Vec<f32>> = vectors.to_vec();
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(stages);
        let mut stage_errors: Vec<f64> = Vec::with_capacity(stages);

        for stage in 0..stages {
            let salt = self.config.seed ^ (stage as u64).wrapping_mul(FNV_PRIME);
            let (centroids, assignment) = kmeans(&residuals, k, dim, iters, salt);

            // Subtract the assigned centroid from each residual and accumulate
            // the resulting total squared residual norm.
            let mut sse = 0.0f64;
            for (i, residual) in residuals.iter_mut().enumerate() {
                let centroid = &centroids[assignment[i]];
                for (r, c) in residual.iter_mut().zip(centroid.iter()) {
                    *r -= c;
                }
                sse += norm_sq_f64(residual);
            }

            codebooks.push(centroids);
            stage_errors.push(sse);
        }

        self.codebooks = codebooks;
        self.stage_errors = stage_errors;
        Ok(())
    }

    // ── Encoding ────────────────────────────────────────────────────────────────

    /// Encode a vector to one codeword index per stage.
    ///
    /// Uses the configured [`beam_width`](ResidualVqConfig::beam_width): a width
    /// of `1` is greedy per-stage assignment; larger widths run beam search.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when called before training.
    /// - [`ResidualVqError::DimMismatch`] when `vector.len() != dim`.
    pub fn encode(&self, vector: &[f32]) -> Result<ResidualCode, ResidualVqError> {
        self.encode_beam(vector, self.config.beam_width)
    }

    /// Encode a vector greedily: at each stage pick the single nearest codeword
    /// to the current residual, subtract it, and continue.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when called before training.
    /// - [`ResidualVqError::DimMismatch`] when `vector.len() != dim`.
    pub fn encode_greedy(&self, vector: &[f32]) -> Result<ResidualCode, ResidualVqError> {
        self.check_query(vector)?;
        let (indices, _) = self.greedy_path(vector);
        Ok(ResidualCode::new(indices))
    }

    /// Encode a vector with beam search of the given `beam_width`.
    ///
    /// Beam search keeps the `beam_width` lowest-error partial codes at each
    /// stage instead of committing to the single greedy choice, which is
    /// provably suboptimal for the joint multi-stage reconstruction. The result
    /// is warm-started with the greedy solution, so it never yields a *higher*
    /// reconstruction error than [`encode_greedy`](Self::encode_greedy). A
    /// `beam_width` of `0` or `1` reduces exactly to greedy encoding.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when called before training.
    /// - [`ResidualVqError::DimMismatch`] when `vector.len() != dim`.
    pub fn encode_beam(
        &self,
        vector: &[f32],
        beam_width: usize,
    ) -> Result<ResidualCode, ResidualVqError> {
        self.check_query(vector)?;
        let (greedy_indices, greedy_error) = self.greedy_path(vector);
        if beam_width <= 1 {
            return Ok(ResidualCode::new(greedy_indices));
        }
        let (beam_indices, beam_error) = self.beam_path(vector, beam_width);
        if beam_error < greedy_error {
            Ok(ResidualCode::new(beam_indices))
        } else {
            Ok(ResidualCode::new(greedy_indices))
        }
    }

    /// Greedy cascade: returns the chosen indices and the final squared
    /// residual norm (the reconstruction error). Assumes a trained quantizer
    /// and a dimension-checked `vector`.
    fn greedy_path(&self, vector: &[f32]) -> (Vec<usize>, f32) {
        let mut residual = vector.to_vec();
        let mut indices = Vec::with_capacity(self.codebooks.len());
        for codebook in &self.codebooks {
            let best = nearest(&residual, codebook);
            for (r, c) in residual.iter_mut().zip(codebook[best].iter()) {
                *r -= c;
            }
            indices.push(best);
        }
        let error = norm_sq(&residual);
        (indices, error)
    }

    /// Beam-search cascade: returns the lowest-error indices found and their
    /// final squared residual norm. Assumes a trained quantizer, a
    /// dimension-checked `vector` and `beam_width >= 1`.
    fn beam_path(&self, vector: &[f32], beam_width: usize) -> (Vec<usize>, f32) {
        // Each beam entry is (partial code, current residual, error).
        let mut beam: Vec<(Vec<usize>, Vec<f32>, f32)> =
            vec![(Vec::new(), vector.to_vec(), norm_sq(vector))];

        for codebook in &self.codebooks {
            let mut expansions: Vec<(Vec<usize>, Vec<f32>, f32)> =
                Vec::with_capacity(beam.len() * codebook.len());
            for (code, residual, _) in &beam {
                for (ci, codeword) in codebook.iter().enumerate() {
                    let new_residual: Vec<f32> = residual
                        .iter()
                        .zip(codeword.iter())
                        .map(|(r, c)| r - c)
                        .collect();
                    let error = norm_sq(&new_residual);
                    let mut new_code = code.clone();
                    new_code.push(ci);
                    expansions.push((new_code, new_residual, error));
                }
            }
            // Keep the lowest-error partial codes; ties break on the code itself
            // for deterministic results.
            expansions.sort_by(|a, b| {
                a.2.partial_cmp(&b.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            expansions.truncate(beam_width.max(1));
            beam = expansions;
        }

        beam.into_iter().next().map_or_else(
            || (vec![0usize; self.codebooks.len()], f32::INFINITY),
            |(code, _, error)| (code, error),
        )
    }

    // ── Decoding / reconstruction ─────────────────────────────────────────────

    /// Reconstruct an approximate vector by summing the selected codewords
    /// across all stages.
    ///
    /// Returns an all-zero vector when the quantizer is untrained. Missing or
    /// out-of-range code entries contribute nothing (their stage is skipped),
    /// so a short code reconstructs only its leading stages.
    #[must_use]
    pub fn decode(&self, code: &ResidualCode) -> Vec<f32> {
        let mut out = vec![0.0f32; self.config.dim];
        if !self.is_trained() {
            return out;
        }
        for (m, codebook) in self.codebooks.iter().enumerate() {
            let Some(&ci) = code.indices.get(m) else {
                break;
            };
            if let Some(codeword) = codebook.get(ci) {
                for (o, c) in out.iter_mut().zip(codeword.iter()) {
                    *o += c;
                }
            }
        }
        out
    }

    /// Squared L2 norm of the reconstruction of `code`.
    ///
    /// This is the `‖x̂‖²` term of asymmetric distance estimation; it depends
    /// only on the code, so an index precomputes it once per database vector.
    #[must_use]
    pub fn reconstruction_norm_sq(&self, code: &ResidualCode) -> f32 {
        norm_sq(&self.decode(code))
    }

    /// Squared-L2 reconstruction error `‖vector − decode(code)‖²`.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when called before training.
    /// - [`ResidualVqError::DimMismatch`] when `vector.len() != dim`.
    pub fn reconstruction_error(
        &self,
        vector: &[f32],
        code: &ResidualCode,
    ) -> Result<f32, ResidualVqError> {
        self.check_query(vector)?;
        Ok(squared_l2(vector, &self.decode(code)))
    }

    // ── Distance estimation ───────────────────────────────────────────────────

    /// Build the asymmetric inner-product tables for `query`.
    ///
    /// `tables[m][k]` is `⟨query, codebook[m][k]⟩`. Combined with a code, a
    /// query norm and a reconstruction norm, these give the squared-L2 distance
    /// via [`distance_from_ip_tables`](Self::distance_from_ip_tables) without
    /// reconstructing the database vector at query time.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when called before training.
    /// - [`ResidualVqError::DimMismatch`] when `query.len() != dim`.
    pub fn inner_product_tables(&self, query: &[f32]) -> Result<Vec<Vec<f32>>, ResidualVqError> {
        self.check_query(query)?;
        let tables = self
            .codebooks
            .iter()
            .map(|codebook| codebook.iter().map(|cw| dot(query, cw)).collect())
            .collect();
        Ok(tables)
    }

    /// Squared L2 norm of a query slice — the constant `‖q‖²` term shared by all
    /// asymmetric distance estimates for a given query.
    #[must_use]
    pub fn query_norm_sq(query: &[f32]) -> f32 {
        norm_sq(query)
    }

    /// Combine precomputed inner-product tables with a code into a squared-L2
    /// distance estimate.
    ///
    /// Computes `query_norm_sq + recon_norm_sq − 2 Σₘ tables[m][code[m]]`, which
    /// equals `‖q − x̂‖²` exactly when `recon_norm_sq` is the true `‖x̂‖²`
    /// (as produced by [`reconstruction_norm_sq`](Self::reconstruction_norm_sq)).
    /// Out-of-range stages contribute nothing.
    #[must_use]
    pub fn distance_from_ip_tables(
        tables: &[Vec<f32>],
        code: &ResidualCode,
        query_norm_sq: f32,
        recon_norm_sq: f32,
    ) -> f32 {
        let mut ip = 0.0f32;
        for (m, &ci) in code.indices.iter().enumerate() {
            if let Some(v) = tables.get(m).and_then(|row| row.get(ci)).copied() {
                ip += v;
            }
        }
        query_norm_sq + recon_norm_sq - 2.0 * ip
    }

    /// Asymmetric squared-L2 distance between a full-precision `query` and an
    /// encoded database vector.
    ///
    /// Keeps the query at full precision and uses the additive inner-product
    /// decomposition, so the result equals `‖query − decode(code)‖²` exactly
    /// (up to floating-point rounding) — the only approximation is the
    /// database-side quantization.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when called before training.
    /// - [`ResidualVqError::DimMismatch`] when `query.len() != dim`.
    pub fn asymmetric_distance(
        &self,
        query: &[f32],
        code: &ResidualCode,
    ) -> Result<f32, ResidualVqError> {
        self.check_query(query)?;
        let recon = self.decode(code);
        let query_norm = norm_sq(query);
        let recon_norm = norm_sq(&recon);
        let mut ip = 0.0f32;
        for (m, codebook) in self.codebooks.iter().enumerate() {
            if let Some(codeword) = code.indices.get(m).and_then(|&ci| codebook.get(ci)) {
                ip += dot(query, codeword);
            }
        }
        Ok(query_norm + recon_norm - 2.0 * ip)
    }

    /// Symmetric squared-L2 distance between two encoded vectors.
    ///
    /// Reconstructs both codes and compares the reconstructions; both operands
    /// carry quantization error. Cheaper to reason about than the asymmetric
    /// form but generally less accurate for a full-precision query.
    ///
    /// # Errors
    ///
    /// - [`ResidualVqError::NotTrained`] when called before training.
    pub fn symmetric_distance(
        &self,
        code_a: &ResidualCode,
        code_b: &ResidualCode,
    ) -> Result<f32, ResidualVqError> {
        if !self.is_trained() {
            return Err(ResidualVqError::NotTrained);
        }
        Ok(squared_l2(&self.decode(code_a), &self.decode(code_b)))
    }

    /// Shared validation for query-shaped inputs.
    fn check_query(&self, vector: &[f32]) -> Result<(), ResidualVqError> {
        if !self.is_trained() {
            return Err(ResidualVqError::NotTrained);
        }
        if vector.len() != self.config.dim {
            return Err(ResidualVqError::DimMismatch {
                expected: self.config.dim,
                got: vector.len(),
            });
        }
        Ok(())
    }
}

// ── Deterministic k-means ────────────────────────────────────────────────────────

/// Pick `k` FNV-seeded, deterministically spread seed indices out of `data`.
///
/// Every candidate is keyed by [`fnv_seed_key`] (salt + position + content),
/// the candidates are sorted by that key, and `k` seeds are chosen at evenly
/// spaced positions across the hash-sorted order. This yields a reproducible
/// pseudo-random spread without any `rand` dependency.
fn seed_indices(data: &[Vec<f32>], k: usize, salt: u64) -> Vec<usize> {
    let n = data.len();
    let mut keyed: Vec<(u64, usize)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (fnv_seed_key(v, i, salt), i))
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    (0..k)
        .map(|ci| {
            let pos = if k <= 1 { 0 } else { (ci * (n - 1)) / (k - 1) };
            keyed[pos.min(n - 1)].1
        })
        .collect()
}

/// Recompute every non-empty cluster's centroid as the mean of its members.
///
/// Accumulates in `f64` for stability, then stores back as `f32`. Empty
/// clusters retain their previous centroid (deterministic), and contribute
/// nothing to the reconstruction-error bound because no residual is assigned to
/// them.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn update_means(
    centroids: &mut [Vec<f32>],
    data: &[Vec<f32>],
    assignment: &[usize],
    effective_k: usize,
    dim: usize,
) {
    let mut sums = vec![vec![0.0f64; dim]; effective_k];
    let mut counts = vec![0usize; effective_k];
    for (i, v) in data.iter().enumerate() {
        let c = assignment[i];
        counts[c] += 1;
        for (acc, &x) in sums[c].iter_mut().zip(v.iter()) {
            *acc += f64::from(x);
        }
    }
    for c in 0..effective_k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f64;
            for (slot, &acc) in centroids[c].iter_mut().zip(sums[c].iter()) {
                *slot = (acc * inv) as f32;
            }
        }
    }
}

/// Run deterministic Lloyd k-means on `data`, returning `k` centroids (padded
/// with duplicates when there are fewer than `k` points) and an assignment of
/// every point to a centroid index.
///
/// The returned centroids are the means of the final assignment's clusters, so
/// the total squared residual `Σ‖data[i] − centroids[assignment[i]]‖²` never
/// exceeds `Σ‖data[i]‖²` — the property that makes the residual cascade's total
/// error monotone non-increasing across stages.
fn kmeans(
    data: &[Vec<f32>],
    k: usize,
    dim: usize,
    iters: usize,
    salt: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let n = data.len();
    // Never request more centroids than points; always at least one.
    let effective_k = k.min(n).max(1);

    let seeds = seed_indices(data, effective_k, salt);
    let mut centroids: Vec<Vec<f32>> = seeds.iter().map(|&i| data[i].clone()).collect();
    let mut assignment = vec![0usize; n];

    for _ in 0..iters.max(1) {
        let mut changed = false;
        for (i, v) in data.iter().enumerate() {
            let best = nearest(v, &centroids);
            if assignment[i] != best {
                assignment[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        update_means(&mut centroids, data, &assignment, effective_k, dim);
    }

    // Final assign + mean update guarantees the centroids are exactly the means
    // of the final assignment regardless of how the loop terminated, which the
    // monotone-error property depends on.
    for (i, v) in data.iter().enumerate() {
        assignment[i] = nearest(v, &centroids);
    }
    update_means(&mut centroids, data, &assignment, effective_k, dim);

    // Pad up to the full codebook size with duplicated centroids so every
    // stored index is valid. Duplicates never win the nearest-centroid race
    // against their originals (ties break to the lower index), keeping encoding
    // deterministic.
    while centroids.len() < k {
        let fill = centroids[centroids.len() % effective_k].clone();
        centroids.push(fill);
    }

    (centroids, assignment)
}
