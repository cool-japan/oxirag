//! The anisotropic quantizer: weighted-loss codebook training (a Lloyd variant
//! whose centroid update solves a per-cluster weighted least-squares system),
//! encoding, decoding and asymmetric query scoring.
//!
//! # The anisotropic loss
//!
//! For a training vector `x` (unit direction `x̂ = x/‖x‖`) quantized to codeword
//! `q`, split the residual `e = x − q` into a component parallel to `x̂` and a
//! component orthogonal to it:
//!
//! ```text
//! e_∥ = (e · x̂) x̂        e_⊥ = e − e_∥
//! ```
//!
//! and penalize them unequally:
//!
//! ```text
//! L(x, q) = h_∥ ‖e_∥‖² + h_⊥ ‖e_⊥‖²
//!         = h_⊥ ‖e‖² + (h_∥ − h_⊥)(x̂ · e)²          (identical, cheaper form)
//!         = (x − q)ᵀ W_x (x − q),  W_x = h_∥ x̂x̂ᵀ + h_⊥(I − x̂x̂ᵀ)
//! ```
//!
//! # Weighted Lloyd iteration
//!
//! * **Assignment.** Each `x` is assigned to the codeword minimizing the
//!   *weighted* loss above (not plain L2), because the weight matrix `W_x`
//!   depends on `x`'s own direction.
//! * **Update.** The new codeword of a cluster is **not** the plain mean; it is
//!   the minimizer of `Σ_x (x − q)ᵀ W_x (x − q)`. Differentiating and setting
//!   the gradient to zero gives the normal equations
//!
//!   ```text
//!   (Σ_x W_x) q* = Σ_x W_x x
//!   ```
//!
//!   a small `d×d` symmetric-positive-definite linear system solved here by
//!   Gaussian elimination with partial pivoting (all accumulation in `f64`).
//!
//! Because each step is the exact minimizer of a shared objective, the total
//! weighted loss is non-increasing across iterations — [`AnisotropicQuantizer`]
//! records the per-iteration loss so callers (and tests) can verify monotone
//! convergence.

use super::types::{AnisotropicCode, AnisotropicVqConfig, AnisotropicVqError, AnisotropicVqMetric};

// ── Deterministic FNV-1a pseudo-random stream (no `rand` dependency) ──────────

/// FNV-1a offset basis.
const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
/// FNV-1a prime.
const FNV_PRIME: u64 = 1_099_511_628_211;

/// FNV-1a 64-bit hash of the little-endian byte concatenation of
/// `(seed, index)`.
fn fnv1a_u64(seed: u64, index: u64) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for &b in seed.to_le_bytes().iter().chain(index.to_le_bytes().iter()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// FNV-1a 64-bit hash mixing a seed with a vector's raw bit pattern and its
/// index, used to derive a deterministic pseudo-random ordering key.
fn fnv_seed_key(seed: u64, vector: &[f32], index: usize) -> u64 {
    let mut h = FNV_OFFSET_BASIS ^ seed;
    for &value in vector {
        for b in value.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    for b in (index as u64).to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Map a 64-bit hash to a half-open uniform `f64` in `[0, 1)`.
fn hash_to_unit(h: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let v = (h >> 11) as f64 / ((1u64 << 53) as f64);
    v
}

// ── Small dense linear algebra (pure Rust, f64) ──────────────────────────────

/// Solve the `n×n` linear system `a · x = b` by Gaussian elimination with
/// partial pivoting. `a` is row-major (`a[r][c]`). Both `a` and `b` are consumed
/// (mutated in place). Returns `None` if the system is (numerically) singular.
///
/// Index-based loops are the clearest expression of dense elimination here, so
/// `needless_range_loop` is allowed locally.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot: pick the row with the largest magnitude in this column.
        let mut pivot = col;
        let mut max_abs = a[col][col].abs();
        for r in (col + 1)..n {
            let candidate = a[r][col].abs();
            if candidate > max_abs {
                max_abs = candidate;
                pivot = r;
            }
        }
        if max_abs < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let pivot_val = a[col][col];
        for r in (col + 1)..n {
            let factor = a[r][col] / pivot_val;
            if factor != 0.0 {
                for c in col..n {
                    a[r][c] -= factor * a[col][c];
                }
                b[r] -= factor * b[col];
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= a[i][j] * x[j];
        }
        x[i] = acc / a[i][i];
    }
    Some(x)
}

// ── Direction helper ─────────────────────────────────────────────────────────

/// The squared Euclidean norm and, when non-degenerate, the unit direction of a
/// vector (all in `f64`).
///
/// Returns `(norm, None)` when `‖v‖` is below a numerical floor — a zero (or
/// effectively zero) vector has no meaningful direction, so it is treated
/// isotropically (`W_x = h_⊥ I`) everywhere in this module.
fn direction(v: &[f32]) -> (f64, Option<Vec<f64>>) {
    let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let norm = norm_sq.sqrt();
    if norm < 1e-12 {
        (norm, None)
    } else {
        let unit = v.iter().map(|&x| f64::from(x) / norm).collect();
        (norm, Some(unit))
    }
}

/// Weighted anisotropic loss `L(x, q)` for a point whose precomputed unit
/// direction is `unit` (`None` for a zero vector, handled isotropically).
///
/// Uses the cheap identity
/// `L = h_⊥ ‖e‖² + (h_∥ − h_⊥)(x̂ · e)²` with `e = x − q`.
fn weighted_loss(
    x: &[f32],
    unit: Option<&[f64]>,
    q: &[f32],
    h_parallel: f64,
    h_orthogonal: f64,
) -> f64 {
    let mut e_norm_sq = 0.0f64;
    let mut parallel_dot = 0.0f64;
    match unit {
        Some(u) => {
            for ((&xi, &qi), &ui) in x.iter().zip(q.iter()).zip(u.iter()) {
                let e = f64::from(xi) - f64::from(qi);
                e_norm_sq += e * e;
                parallel_dot += ui * e;
            }
        }
        None => {
            for (&xi, &qi) in x.iter().zip(q.iter()) {
                let e = f64::from(xi) - f64::from(qi);
                e_norm_sq += e * e;
            }
        }
    }
    let parallel_norm_sq = parallel_dot * parallel_dot;
    h_orthogonal * e_norm_sq + (h_parallel - h_orthogonal) * parallel_norm_sq
}

// ── Public residual-decomposition / loss helpers ─────────────────────────────

/// Decompose the reconstruction residual `e = x − q` into its component
/// **parallel** to `x`'s own direction and its **orthogonal** complement.
///
/// Returns `(e_parallel, e_orthogonal)` with `e_parallel + e_orthogonal == e`
/// (exactly, up to floating-point rounding) and `e_parallel · e_orthogonal ≈ 0`.
/// When `x` is the zero vector its direction is undefined; the parallel
/// component is then defined to be zero and the whole residual is reported as
/// orthogonal.
///
/// The shorter of the two slices determines the output length (the public API
/// validates equal lengths before calling this).
#[must_use]
pub fn decompose_residual(x: &[f32], q: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let len = x.len().min(q.len());
    let residual: Vec<f64> = (0..len)
        .map(|i| f64::from(x[i]) - f64::from(q[i]))
        .collect();
    let (_, unit) = direction(&x[..len]);

    if let Some(u) = unit {
        let dot: f64 = u
            .iter()
            .zip(residual.iter())
            .map(|(&ui, &ei)| ui * ei)
            .sum();
        let mut parallel = Vec::with_capacity(len);
        let mut orthogonal = Vec::with_capacity(len);
        for (&ui, &ei) in u.iter().zip(residual.iter()) {
            let par = dot * ui;
            #[allow(clippy::cast_possible_truncation)]
            {
                parallel.push(par as f32);
                orthogonal.push((ei - par) as f32);
            }
        }
        (parallel, orthogonal)
    } else {
        let parallel = vec![0.0f32; len];
        #[allow(clippy::cast_possible_truncation)]
        let orthogonal = residual.iter().map(|&e| e as f32).collect();
        (parallel, orthogonal)
    }
}

/// The anisotropic weighted loss `L(x, q) = h_∥ ‖e_∥‖² + h_⊥ ‖e_⊥‖²` with
/// `h_⊥ = 1.0` and `h_∥ = parallel_multiplier`, computed directly from `x` and
/// `q` (the unit direction of `x` is derived internally).
///
/// The shorter of the two slices determines the length used.
#[must_use]
pub fn anisotropic_loss(x: &[f32], q: &[f32], parallel_multiplier: f64) -> f64 {
    let len = x.len().min(q.len());
    let (_, unit) = direction(&x[..len]);
    weighted_loss(
        &x[..len],
        unit.as_deref(),
        &q[..len],
        parallel_multiplier,
        1.0,
    )
}

// ── Per-point cached geometry ─────────────────────────────────────────────────

/// A training point's cached norm and (optional) unit direction, computed once
/// per `train` call and reused across every iteration.
struct PointGeometry {
    norm: f64,
    unit: Option<Vec<f64>>,
}

// ── AnisotropicQuantizer ──────────────────────────────────────────────────────

/// A `ScaNN`-style anisotropic vector quantizer.
///
/// After [`train`](Self::train), `codebooks()[m][k]` holds the `k`-th codeword
/// of subspace `m` (a `subspace_dim`-length vector). A vector is
/// [`encode`](Self::encode)d to the codeword minimizing its *anisotropic*
/// reconstruction loss per subspace, and scored against a query by asymmetric
/// lookup ([`inner_product_tables`](Self::inner_product_tables) /
/// [`l2_tables`](Self::l2_tables)).
#[derive(Debug, Clone)]
pub struct AnisotropicQuantizer {
    config: AnisotropicVqConfig,
    /// Codebooks shaped `num_subspaces × K × subspace_dim`; empty until trained.
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Total weighted loss after each weighted-Lloyd iteration (summed across
    /// subspaces); empty until trained. Non-increasing by construction.
    loss_trajectory: Vec<f64>,
}

impl AnisotropicQuantizer {
    /// Create an untrained quantizer for the given configuration.
    ///
    /// The configuration is validated lazily on [`train`](Self::train); this
    /// constructor never fails.
    #[must_use]
    pub fn new(config: AnisotropicVqConfig) -> Self {
        Self {
            config,
            codebooks: Vec::new(),
            loss_trajectory: Vec::new(),
        }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &AnisotropicVqConfig {
        &self.config
    }

    /// Return `true` once [`train`](Self::train) has produced codebooks.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        !self.codebooks.is_empty()
    }

    /// Borrow the trained codebooks (`num_subspaces × K × subspace_dim`).
    ///
    /// Empty until [`train`](Self::train) has run.
    #[must_use]
    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    /// Borrow the recorded per-iteration total weighted loss.
    ///
    /// The sequence is non-increasing (a fundamental property of weighted
    /// Lloyd, where both the assignment and the weighted-least-squares update
    /// minimize a shared objective). Empty until [`train`](Self::train) has run.
    #[must_use]
    pub fn loss_trajectory(&self) -> &[f64] {
        &self.loss_trajectory
    }

    /// Number of subspaces.
    #[must_use]
    pub fn num_subspaces(&self) -> usize {
        self.config.num_subspaces
    }

    /// Train one anisotropic codebook per subspace via weighted Lloyd.
    ///
    /// # Errors
    ///
    /// - [`AnisotropicVqError::InvalidConfig`] when the configuration is invalid
    ///   (see [`AnisotropicVqConfig::validate`]).
    /// - [`AnisotropicVqError::EmptyTrainingSet`] when `vectors` is empty.
    /// - [`AnisotropicVqError::DimMismatch`] when any vector length differs from
    ///   [`AnisotropicVqConfig::dim`].
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<(), AnisotropicVqError> {
        self.config.validate()?;
        if vectors.is_empty() {
            return Err(AnisotropicVqError::EmptyTrainingSet);
        }
        for v in vectors {
            if v.len() != self.config.dim {
                return Err(AnisotropicVqError::DimMismatch {
                    expected: self.config.dim,
                    actual: v.len(),
                });
            }
        }

        let num_subspaces = self.config.num_subspaces;
        let sub_dim = self.config.subspace_dim();
        let k = self.config.num_codewords;
        let h_parallel = self.config.parallel_weight_multiplier;
        let h_orthogonal = self.config.orthogonal_weight();

        let mut codebooks = Vec::with_capacity(num_subspaces);
        let mut per_subspace_traj: Vec<Vec<f64>> = Vec::with_capacity(num_subspaces);

        for s in 0..num_subspaces {
            let start = s * sub_dim;
            let end = start + sub_dim;
            let sub_vectors: Vec<Vec<f32>> =
                vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // Fold the subspace index into the seed so each subspace gets an
            // independent-but-deterministic k-means++ initialization.
            let subspace_seed = fnv1a_u64(self.config.seed, s as u64);
            let (codebook, trajectory) = Self::train_subspace(
                &sub_vectors,
                k,
                sub_dim,
                h_parallel,
                h_orthogonal,
                self.config.max_iterations,
                self.config.convergence_tolerance,
                subspace_seed,
            );
            codebooks.push(codebook);
            per_subspace_traj.push(trajectory);
        }

        self.codebooks = codebooks;
        self.loss_trajectory = Self::combine_trajectories(&per_subspace_traj);
        Ok(())
    }

    /// Align per-subspace loss trajectories (which may have differing lengths
    /// due to independent early stopping) to a common length by right-padding
    /// each with its final value, then sum element-wise.
    ///
    /// Padding with the *last* (smallest) value keeps every padded trajectory
    /// non-increasing, so the element-wise sum is non-increasing too.
    fn combine_trajectories(per_subspace: &[Vec<f64>]) -> Vec<f64> {
        let max_len = per_subspace.iter().map(Vec::len).max().unwrap_or(0);
        if max_len == 0 {
            return Vec::new();
        }
        let mut combined = vec![0.0f64; max_len];
        for traj in per_subspace {
            let last = traj.last().copied().unwrap_or(0.0);
            for (i, slot) in combined.iter_mut().enumerate() {
                *slot += traj.get(i).copied().unwrap_or(last);
            }
        }
        combined
    }

    /// Run weighted Lloyd on one subspace, returning `(codebook, trajectory)`.
    ///
    /// The returned codebook always has exactly `k` entries; if the number of
    /// distinct points is smaller than `k` the surplus slots are filled with
    /// duplicate codewords (which never win an assignment race against the
    /// originals, keeping encoding deterministic).
    #[allow(clippy::too_many_arguments)]
    fn train_subspace(
        sub_vectors: &[Vec<f32>],
        k: usize,
        sub_dim: usize,
        h_parallel: f64,
        h_orthogonal: f64,
        max_iterations: usize,
        convergence_tolerance: f64,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<f64>) {
        let n = sub_vectors.len();
        let effective_k = k.min(n).max(1);

        // Precompute per-point geometry once.
        let geometry: Vec<PointGeometry> = sub_vectors
            .iter()
            .map(|v| {
                let (norm, unit) = direction(v);
                PointGeometry { norm, unit }
            })
            .collect();

        let mut centroids = Self::kmeans_pp_init(
            sub_vectors,
            &geometry,
            effective_k,
            h_parallel,
            h_orthogonal,
            seed,
        );

        // Initial assignment against the k-means++ seed centroids, and its loss.
        // Recording this *before* the first update lets the trajectory expose the
        // improvement the very first weighted-least-squares update produces (and
        // guarantees at least one update is applied even when the seed assignment
        // is already stable, e.g. the single-cluster `K = 1` case).
        let mut assignment: Vec<usize> = sub_vectors
            .iter()
            .enumerate()
            .map(|(i, sv)| {
                Self::assign_point(
                    sv,
                    geometry[i].unit.as_deref(),
                    &centroids,
                    h_parallel,
                    h_orthogonal,
                )
            })
            .collect();
        let mut trajectory = vec![Self::total_loss(
            sub_vectors,
            &geometry,
            &centroids,
            &assignment,
            h_parallel,
            h_orthogonal,
        )];
        let mut prev_loss = f64::INFINITY;

        for _ in 0..max_iterations.max(1) {
            // ── Weighted-least-squares update step ──
            Self::update_centroids(
                &mut centroids,
                sub_vectors,
                &geometry,
                &assignment,
                sub_dim,
                h_parallel,
                h_orthogonal,
            );

            // Total weighted loss under the (assignment, updated centroids) pair.
            // Non-increasing relative to the previously recorded loss because the
            // update minimizes the objective for the fixed assignment.
            let loss = Self::total_loss(
                sub_vectors,
                &geometry,
                &centroids,
                &assignment,
                h_parallel,
                h_orthogonal,
            );
            trajectory.push(loss);

            // ── Assignment step (weighted argmin) against the updated codebook ──
            let mut changed = false;
            for (i, sv) in sub_vectors.iter().enumerate() {
                let best = Self::assign_point(
                    sv,
                    geometry[i].unit.as_deref(),
                    &centroids,
                    h_parallel,
                    h_orthogonal,
                );
                if assignment[i] != best {
                    assignment[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
            if prev_loss.is_finite() {
                let denom = prev_loss.abs().max(1e-12);
                let relative_improvement = (prev_loss - loss) / denom;
                if relative_improvement < convergence_tolerance {
                    break;
                }
            }
            prev_loss = loss;
        }

        // Pad the codebook up to the full size `k` with duplicates.
        while centroids.len() < k {
            let fill = centroids[centroids.len() % effective_k].clone();
            centroids.push(fill);
        }

        (centroids, trajectory)
    }

    /// Deterministic k-means++ initialization using the *anisotropic* distance
    /// as the D² weighting, driven by an FNV-1a pseudo-random stream.
    fn kmeans_pp_init(
        sub_vectors: &[Vec<f32>],
        geometry: &[PointGeometry],
        effective_k: usize,
        h_parallel: f64,
        h_orthogonal: f64,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let n = sub_vectors.len();
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(effective_k);
        let mut chosen = vec![false; n];

        // First center: the point with the smallest deterministic FNV key.
        let first = (0..n)
            .min_by_key(|&i| (fnv_seed_key(seed, &sub_vectors[i], i), i))
            .unwrap_or(0);
        centroids.push(sub_vectors[first].clone());
        chosen[first] = true;

        // Nearest-center D² for every point (weighted).
        let mut nearest_d2: Vec<f64> = (0..n)
            .map(|i| {
                weighted_loss(
                    &sub_vectors[i],
                    geometry[i].unit.as_deref(),
                    &sub_vectors[first],
                    h_parallel,
                    h_orthogonal,
                )
            })
            .collect();

        for step in 1..effective_k {
            let total: f64 = nearest_d2.iter().sum();
            let pick = if total <= 1e-18 {
                // All remaining points coincide (numerically) with the chosen
                // centers: fall back to the smallest-FNV-key unchosen point.
                (0..n)
                    .filter(|&i| !chosen[i])
                    .min_by_key(|&i| (fnv_seed_key(seed, &sub_vectors[i], i), i))
                    .unwrap_or(first)
            } else {
                // D²-weighted deterministic sample.
                let target = hash_to_unit(fnv1a_u64(seed, step as u64)) * total;
                let mut acc = 0.0f64;
                let mut candidate = first;
                for (i, &d2) in nearest_d2.iter().enumerate() {
                    acc += d2;
                    if acc >= target && !chosen[i] {
                        candidate = i;
                        break;
                    }
                    // Track the last unchosen point as a safe fallback for the
                    // case where floating-point summation stops just short.
                    if !chosen[i] {
                        candidate = i;
                    }
                }
                candidate
            };

            chosen[pick] = true;
            centroids.push(sub_vectors[pick].clone());

            // Update nearest-center distances against the newly added center.
            let new_center = &sub_vectors[pick];
            for (i, d2) in nearest_d2.iter_mut().enumerate() {
                if chosen[i] {
                    *d2 = 0.0;
                    continue;
                }
                let candidate_d2 = weighted_loss(
                    &sub_vectors[i],
                    geometry[i].unit.as_deref(),
                    new_center,
                    h_parallel,
                    h_orthogonal,
                );
                if candidate_d2 < *d2 {
                    *d2 = candidate_d2;
                }
            }
        }

        centroids
    }

    /// Index of the codeword minimizing the weighted loss for one point.
    /// Ties are broken toward the lowest index (deterministic).
    fn assign_point(
        x: &[f32],
        unit: Option<&[f64]>,
        centroids: &[Vec<f32>],
        h_parallel: f64,
        h_orthogonal: f64,
    ) -> usize {
        let mut best = 0usize;
        let mut best_loss = f64::INFINITY;
        for (ci, centroid) in centroids.iter().enumerate() {
            let loss = weighted_loss(x, unit, centroid, h_parallel, h_orthogonal);
            if loss < best_loss {
                best_loss = loss;
                best = ci;
            }
        }
        best
    }

    /// Weighted-least-squares centroid update: for each cluster, solve
    /// `(Σ_x W_x) q* = Σ_x W_x x`. Empty clusters (and the rare numerically
    /// singular system) retain their previous centroid, so the total loss never
    /// increases and repeated runs are deterministic.
    fn update_centroids(
        centroids: &mut [Vec<f32>],
        sub_vectors: &[Vec<f32>],
        geometry: &[PointGeometry],
        assignment: &[usize],
        sub_dim: usize,
        h_parallel: f64,
        h_orthogonal: f64,
    ) {
        let k = centroids.len();
        // Per-cluster normal-equation accumulators: `a[ci]` = Σ W_x, `b[ci]` =
        // Σ W_x x, `counts[ci]` = cluster size.
        let mut a_mats: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0f64; sub_dim]; sub_dim]; k];
        let mut b_vecs: Vec<Vec<f64>> = vec![vec![0.0f64; sub_dim]; k];
        let mut counts = vec![0usize; k];

        let coef = h_parallel - h_orthogonal;

        for (i, sv) in sub_vectors.iter().enumerate() {
            let ci = assignment[i];
            counts[ci] += 1;
            let a = &mut a_mats[ci];
            let b = &mut b_vecs[ci];

            // A += W_x = h_⊥ I + (h_∥ − h_⊥) x̂ x̂ᵀ  (diagonal isotropic part)
            for (r, row) in a.iter_mut().enumerate() {
                row[r] += h_orthogonal;
            }
            // b += W_x x = h_⊥ x + (h_∥ − h_⊥)(x̂ · x) x̂  (isotropic part)
            for (slot, &xr) in b.iter_mut().zip(sv.iter()) {
                *slot += h_orthogonal * f64::from(xr);
            }

            if let Some(u) = geometry[i].unit.as_deref() {
                // (x̂ · x) = ‖x‖ = geometry[i].norm, so the parallel part of
                // `W_x x` is (h_∥ − h_⊥)·‖x‖·x̂.
                let norm = geometry[i].norm;
                for (r, row) in a.iter_mut().enumerate() {
                    let ur = u[r];
                    for (c, slot) in row.iter_mut().enumerate() {
                        *slot += coef * ur * u[c];
                    }
                    b[r] += coef * norm * ur;
                }
            }
        }

        for ci in 0..k {
            if counts[ci] == 0 {
                continue; // keep previous centroid for empty clusters
            }
            let a = std::mem::take(&mut a_mats[ci]);
            let b = std::mem::take(&mut b_vecs[ci]);
            if let Some(solution) = solve_linear_system(a, b) {
                #[allow(clippy::cast_possible_truncation)]
                for (slot, &value) in centroids[ci].iter_mut().zip(solution.iter()) {
                    *slot = value as f32;
                }
            }
            // Singular system: leave the previous centroid in place.
        }
    }

    /// Total weighted loss over all points under the current assignment.
    fn total_loss(
        sub_vectors: &[Vec<f32>],
        geometry: &[PointGeometry],
        centroids: &[Vec<f32>],
        assignment: &[usize],
        h_parallel: f64,
        h_orthogonal: f64,
    ) -> f64 {
        sub_vectors
            .iter()
            .enumerate()
            .map(|(i, sv)| {
                weighted_loss(
                    sv,
                    geometry[i].unit.as_deref(),
                    &centroids[assignment[i]],
                    h_parallel,
                    h_orthogonal,
                )
            })
            .sum()
    }

    /// Encode a vector to one codeword index per subspace, choosing the codeword
    /// that minimizes the point's *anisotropic* reconstruction loss (the same
    /// objective training optimized).
    ///
    /// # Errors
    ///
    /// - [`AnisotropicVqError::NotTrained`] before [`train`](Self::train).
    /// - [`AnisotropicVqError::DimMismatch`] when `vector.len() != dim`.
    pub fn encode(&self, vector: &[f32]) -> Result<AnisotropicCode, AnisotropicVqError> {
        if !self.is_trained() {
            return Err(AnisotropicVqError::NotTrained);
        }
        if vector.len() != self.config.dim {
            return Err(AnisotropicVqError::DimMismatch {
                expected: self.config.dim,
                actual: vector.len(),
            });
        }

        let sub_dim = self.config.subspace_dim();
        let h_parallel = self.config.parallel_weight_multiplier;
        let h_orthogonal = self.config.orthogonal_weight();

        let mut indices = Vec::with_capacity(self.config.num_subspaces);
        for (s, codebook) in self.codebooks.iter().enumerate() {
            let start = s * sub_dim;
            let sub = &vector[start..start + sub_dim];
            let (_, unit) = direction(sub);
            let best = Self::assign_point(sub, unit.as_deref(), codebook, h_parallel, h_orthogonal);
            #[allow(clippy::cast_possible_truncation)]
            indices.push(best as u16);
        }
        Ok(AnisotropicCode::new(indices))
    }

    /// Reconstruct an approximate vector by concatenating the selected
    /// codewords.
    ///
    /// When the quantizer is untrained, or `code` has fewer entries than there
    /// are subspaces, the corresponding output region is left as zeros.
    #[must_use]
    pub fn decode(&self, code: &AnisotropicCode) -> Vec<f32> {
        let sub_dim = self.config.subspace_dim();
        let mut out = vec![0.0f32; self.config.dim];
        if !self.is_trained() {
            return out;
        }
        for (s, codebook) in self.codebooks.iter().enumerate() {
            let Some(&ci) = code.indices.get(s) else {
                break;
            };
            let idx = (ci as usize).min(codebook.len().saturating_sub(1));
            let start = s * sub_dim;
            out[start..start + sub_dim].copy_from_slice(&codebook[idx]);
        }
        out
    }

    /// Build the per-subspace inner-product lookup tables for `query`:
    /// `tables[m][k] = ⟨query_m, codebook[m][k]⟩`.
    ///
    /// The estimated inner product of `query` with a code is then the sum of the
    /// per-subspace looked-up entries.
    ///
    /// # Errors
    ///
    /// - [`AnisotropicVqError::NotTrained`] before [`train`](Self::train).
    /// - [`AnisotropicVqError::DimMismatch`] when `query.len() != dim`.
    pub fn inner_product_tables(&self, query: &[f32]) -> Result<Vec<Vec<f32>>, AnisotropicVqError> {
        self.build_tables(query, TableKind::InnerProduct)
    }

    /// Build the per-subspace squared-L2 lookup tables for `query`:
    /// `tables[m][k] = ‖query_m − codebook[m][k]‖²`.
    ///
    /// # Errors
    ///
    /// - [`AnisotropicVqError::NotTrained`] before [`train`](Self::train).
    /// - [`AnisotropicVqError::DimMismatch`] when `query.len() != dim`.
    pub fn l2_tables(&self, query: &[f32]) -> Result<Vec<Vec<f32>>, AnisotropicVqError> {
        self.build_tables(query, TableKind::L2)
    }

    fn build_tables(
        &self,
        query: &[f32],
        kind: TableKind,
    ) -> Result<Vec<Vec<f32>>, AnisotropicVqError> {
        if !self.is_trained() {
            return Err(AnisotropicVqError::NotTrained);
        }
        if query.len() != self.config.dim {
            return Err(AnisotropicVqError::DimMismatch {
                expected: self.config.dim,
                actual: query.len(),
            });
        }

        let sub_dim = self.config.subspace_dim();
        let mut tables = Vec::with_capacity(self.config.num_subspaces);
        for (s, codebook) in self.codebooks.iter().enumerate() {
            let start = s * sub_dim;
            let sub_query = &query[start..start + sub_dim];
            let row: Vec<f32> = codebook
                .iter()
                .map(|centroid| match kind {
                    TableKind::InnerProduct => inner_product(sub_query, centroid),
                    TableKind::L2 => squared_l2(sub_query, centroid),
                })
                .collect();
            tables.push(row);
        }
        Ok(tables)
    }

    /// Sum precomputed lookup-table entries for a `code`.
    ///
    /// This is the table-driven core of asymmetric scoring: callers that score
    /// many codes against one query build the tables once and reuse them here.
    #[must_use]
    pub fn score_from_tables(tables: &[Vec<f32>], code: &AnisotropicCode) -> f32 {
        tables
            .iter()
            .zip(code.indices.iter())
            .map(|(row, &ci)| {
                let idx = (ci as usize).min(row.len().saturating_sub(1));
                row[idx]
            })
            .sum()
    }

    /// Estimated inner product `⟨query, decode(code)⟩` via asymmetric lookup.
    ///
    /// # Errors
    ///
    /// As for [`inner_product_tables`](Self::inner_product_tables), plus
    /// [`AnisotropicVqError::DimMismatch`] when `code` does not span every
    /// subspace.
    pub fn estimate_inner_product(
        &self,
        query: &[f32],
        code: &AnisotropicCode,
    ) -> Result<f32, AnisotropicVqError> {
        if code.indices.len() != self.config.num_subspaces {
            return Err(AnisotropicVqError::DimMismatch {
                expected: self.config.num_subspaces,
                actual: code.indices.len(),
            });
        }
        let tables = self.inner_product_tables(query)?;
        Ok(Self::score_from_tables(&tables, code))
    }

    /// Estimated squared-L2 distance `‖query − decode(code)‖²` via asymmetric
    /// lookup.
    ///
    /// # Errors
    ///
    /// As for [`l2_tables`](Self::l2_tables), plus
    /// [`AnisotropicVqError::DimMismatch`] when `code` does not span every
    /// subspace.
    pub fn estimate_l2_sq(
        &self,
        query: &[f32],
        code: &AnisotropicCode,
    ) -> Result<f32, AnisotropicVqError> {
        if code.indices.len() != self.config.num_subspaces {
            return Err(AnisotropicVqError::DimMismatch {
                expected: self.config.num_subspaces,
                actual: code.indices.len(),
            });
        }
        let tables = self.l2_tables(query)?;
        Ok(Self::score_from_tables(&tables, code))
    }

    /// Estimated "smaller is closer" distance for `code` under `metric`, taken
    /// from precomputed `tables` built for the matching metric.
    ///
    /// For [`InnerProduct`](AnisotropicVqMetric::InnerProduct) the inner-product
    /// sum is negated so ascending order surfaces the best matches first.
    #[must_use]
    pub fn ranked_distance_from_tables(
        tables: &[Vec<f32>],
        code: &AnisotropicCode,
        metric: AnisotropicVqMetric,
    ) -> f32 {
        let score = Self::score_from_tables(tables, code);
        match metric {
            AnisotropicVqMetric::InnerProduct => -score,
            AnisotropicVqMetric::L2 => score,
        }
    }
}

/// Which quantity a lookup table stores.
#[derive(Debug, Clone, Copy)]
enum TableKind {
    InnerProduct,
    L2,
}

/// Inner product of two equal-length slices (shared prefix only).
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared Euclidean distance between two equal-length slices (shared prefix).
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}
