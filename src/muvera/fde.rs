//! The Fixed Dimensional Encoding (FDE) construction — the algorithmic core of
//! MUVERA (Dhulipala et al., 2024).
//!
//! # What the FDE does
//!
//! A multi-vector set `X = {x_1, …, x_m} ⊂ ℝ^d` (the token embeddings of a
//! document or a query) is mapped to a single vector `FDE(X) ∈ ℝ^D` so that,
//! for a query set `Q` and a document set `D`,
//!
//! ```text
//! ⟨FDE_query(Q), FDE_doc(D)⟩  ≈  Σ_{q∈Q} max_{d∈D} ⟨q, d⟩   (= Chamfer / MaxSim)
//! ```
//!
//! # Construction (per repetition, then concatenated)
//!
//! 1. **`SimHash` space partition.** Draw `k_sim` deterministic Gaussian
//!    hyperplanes. A token `x`'s signs against them form a `k_sim`-bit cell id
//!    in `{0, …, 2^k_sim − 1}`, partitioning `ℝ^d` into `B = 2^k_sim` cells.
//! 2. **Per-cell aggregation.** For the **query** side, each cell holds the
//!    **sum** of the query tokens that fall in it; for the **document** side,
//!    each cell holds the **average** (centroid) of the document tokens that
//!    fall in it. Empty document cells optionally borrow the centroid of the
//!    Hamming-nearest non-empty cell (see `nearest_occupied_bucket`). This
//!    sum-vs-average asymmetry is exactly what makes the dot product telescope
//!    into `Σ_{q∈Q} ⟨q, centroid(cell(q))⟩ ≈ Σ_{q∈Q} max_{d∈D} ⟨q, d⟩`.
//! 3. **Inner projection.** Each `d`-dimensional cell block is optionally
//!    projected to `d_proj` dimensions by a deterministic Rademacher (`±1`)
//!    Johnson–Lindenstrauss matrix scaled by `1/√d_proj`, which preserves
//!    inner products in expectation.
//! 4. **Repetitions.** Steps 1–3 are repeated `r_reps` times with independent
//!    seeds and the results **concatenated**, averaging out the variance of a
//!    single `SimHash` partition.
//!
//! The final dimension is `r_reps · B · inner_dim`, where `inner_dim = d_proj`
//! when projection is enabled and `inner_dim = d` otherwise.
//!
//! # Determinism
//!
//! No random-number crate is used. A `splitmix64` stream expands the master
//! seed into an independent 64-bit sub-seed for every (repetition, purpose)
//! pair; hyperplane entries are standard-normal samples produced by FNV-1a
//! hashing followed by the Box–Muller transform; projection signs are single
//! bits drawn from an FNV-1a stream. The same configuration therefore always
//! reproduces byte-identical hyperplanes, projections, and FDEs.

use super::types::{MuveraConfig, MuveraError, MuveraResult};

// ── Deterministic pseudo-random primitives ───────────────────────────────────

/// One step of the `splitmix64` generator: maps a 64-bit state to a
/// well-mixed 64-bit output. Used to expand the master seed into independent
/// per-(repetition, purpose) sub-seeds.
#[inline]
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive an independent sub-seed for a given repetition and purpose from the
/// master `seed`, by folding both into a `splitmix64` stream.
#[inline]
fn derive_seed(seed: u64, rep: usize, purpose: u64) -> u64 {
    let mixed = seed
        .wrapping_add(splitmix64((rep as u64).wrapping_add(1)))
        .wrapping_add(splitmix64(purpose));
    splitmix64(mixed)
}

/// FNV-1a 64-bit hash (offset basis `14695981039346656037`, prime
/// `1099511628211`) of the little-endian bytes of `(seed, index)`.
#[inline]
fn fnv1a_u64(seed: u64, index: u64) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in seed.to_le_bytes().iter().chain(index.to_le_bytes().iter()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Map a 64-bit hash to an open-interval uniform `f64` in `(0, 1)`.
///
/// The open interval avoids exactly `0.0` (which would make `ln(u1)` diverge
/// in the Box–Muller transform) and exactly `1.0`.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn hash_to_unit_open(h: u64) -> f64 {
    let v = (h as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    v.clamp(1e-15, 1.0 - 1e-15)
}

/// The `k`-th (0-indexed) standard-normal pseudo-random sample, deterministic
/// in `(seed, k)`, via the Box–Muller transform. Each uniform pair
/// `(u1, u2)` yields one normal pair `(z0, z1)`; `k` selects which element of
/// pair `k / 2` to return.
#[inline]
fn box_muller_sample(seed: u64, k: u64) -> f64 {
    let pair = k / 2;
    let u1 = hash_to_unit_open(fnv1a_u64(seed, 2 * pair));
    let u2 = hash_to_unit_open(fnv1a_u64(seed, 2 * pair + 1));
    let radius = (-2.0 * u1.ln()).sqrt();
    if k.is_multiple_of(2) {
        radius * (2.0 * std::f64::consts::PI * u2).cos()
    } else {
        radius * (2.0 * std::f64::consts::PI * u2).sin()
    }
}

/// The deterministic Rademacher sign (`+1.0` or `-1.0`) for projection entry
/// `index` under `seed`.
///
/// The sign is read from the top bit of a `splitmix64`-mixed FNV-1a hash. The
/// extra `splitmix64` avalanche is essential: the *low* bits of a raw FNV-1a
/// hash are poorly distributed and, for an even token dimension, the entry sign
/// `sign(out · dim + j)` degenerates into a function of `j` alone — every row of
/// the projection matrix becomes identical and the matrix collapses to rank 1,
/// destroying the Johnson–Lindenstrauss property. Mixing first and taking the
/// most-significant bit yields independent, unbiased signs.
#[inline]
fn rademacher_sign(seed: u64, index: u64) -> f32 {
    let mixed = splitmix64(fnv1a_u64(seed, index));
    if (mixed >> 63) & 1 == 0 { 1.0 } else { -1.0 }
}

// Distinct `purpose` tags keep the hyperplane and projection seed streams
// independent within a repetition.
const PURPOSE_HYPERPLANE: u64 = 0x0000_0000_0000_0001;
const PURPOSE_PROJECTION: u64 = 0x0000_0000_0000_0002;

// ── Empty-cell fill ──────────────────────────────────────────────────────────

/// Index of the occupied cell whose `k_sim`-bit id is Hamming-nearest to
/// `target`, ties broken by the lowest cell index. Returns `None` when no cell
/// is occupied.
///
/// This is the document-side rule for populating an empty `SimHash` cell: the
/// missing cell borrows the centroid of the non-empty cell closest to it in
/// the `SimHash` bit space, which is the cell most likely to contain a token
/// similar to any query token that would hash to the empty cell.
#[must_use]
pub(crate) fn nearest_occupied_bucket(
    target: usize,
    occupied: &[bool],
    k_sim: u32,
) -> Option<usize> {
    let mut best: Option<(u32, usize)> = None;
    for (idx, &is_occupied) in occupied.iter().enumerate() {
        if !is_occupied {
            continue;
        }
        // Hamming distance between the two cell ids, restricted to k_sim bits.
        let mask = if k_sim >= usize::BITS {
            usize::MAX
        } else {
            (1usize << k_sim) - 1
        };
        let distance = ((target ^ idx) & mask).count_ones();
        match best {
            Some((best_dist, _)) if distance >= best_dist => {}
            _ => best = Some((distance, idx)),
        }
    }
    best.map(|(_, idx)| idx)
}

// ── FixedDimEncoding ─────────────────────────────────────────────────────────

/// A Fixed Dimensional Encoding: the single dense vector that a multi-vector
/// set is reduced to.
///
/// Its layout is `r_reps` contiguous repetition segments, each holding `B`
/// contiguous cell blocks of `inner_dim` values, so the total length is
/// `r_reps · B · inner_dim` (see [`MuveraConfig::expected_dim`]). Two FDEs
/// built by the same [`MuveraEncoder`] are directly comparable by dot product;
/// [`dot`](Self::dot) computes that comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedDimEncoding {
    /// The dense encoding values.
    values: Vec<f32>,
    /// The dimensionality of one per-cell block (`inner_dim`), recorded so the
    /// FDE can be sliced back into blocks for inspection.
    block_dim: usize,
}

impl FixedDimEncoding {
    /// Construct an encoding from its raw values and per-cell block dimension.
    #[must_use]
    pub(crate) fn new(values: Vec<f32>, block_dim: usize) -> Self {
        Self { values, block_dim }
    }

    /// Borrow the dense encoding values.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// The encoding length `r_reps · B · inner_dim`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Return `true` when the encoding has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The dimensionality of a single per-cell block (`inner_dim`).
    #[must_use]
    pub fn block_dim(&self) -> usize {
        self.block_dim
    }

    /// The number of per-cell blocks in the encoding (`r_reps · B`).
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.values.len().checked_div(self.block_dim).unwrap_or(0)
    }

    /// Borrow the `index`-th per-cell block, or `None` when out of range.
    #[must_use]
    pub fn block(&self, index: usize) -> Option<&[f32]> {
        if self.block_dim == 0 {
            return None;
        }
        let start = index.checked_mul(self.block_dim)?;
        let end = start.checked_add(self.block_dim)?;
        self.values.get(start..end)
    }

    /// The dot product `⟨self, other⟩`, the single-vector proxy for the
    /// Chamfer similarity between the two underlying multi-vector sets.
    ///
    /// Only the shared prefix is used when the two encodings differ in length;
    /// for FDEs produced by the same encoder the lengths always match.
    #[must_use]
    pub fn dot(&self, other: &FixedDimEncoding) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

// ── Exact Chamfer / MaxSim ───────────────────────────────────────────────────

/// The exact Chamfer / `MaxSim` similarity `Σ_{q∈Q} max_{d∈D} ⟨q, d⟩` between a
/// query set and a document set. Returns `0.0` when either set is empty.
///
/// This is the ground-truth quantity the FDE dot product approximates, and the
/// score used by the optional exact re-rank.
#[must_use]
pub(crate) fn chamfer_similarity(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for q in query {
        let mut best = f32::NEG_INFINITY;
        for d in doc {
            let dot: f32 = q.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
            if dot > best {
                best = dot;
            }
        }
        if best > f32::NEG_INFINITY {
            total += best;
        }
    }
    total
}

// ── MuveraEncoder ────────────────────────────────────────────────────────────

/// Encodes multi-vector sets into [`FixedDimEncoding`]s.
///
/// Construction precomputes and stores every `SimHash` hyperplane and inner
/// projection matrix from the configuration's seed, so encoding is a fast,
/// allocation-light pass and is fully deterministic:
/// [`encode_query`](Self::encode_query) and
/// [`encode_document`](Self::encode_document) applied to the same input under
/// the same configuration always return identical encodings.
///
/// The query and document paths share the *same* hyperplanes and projections
/// per repetition — only the per-cell aggregation differs (sum for the query,
/// average-plus-optional-fill for the document), which is what makes the dot
/// product of the two encodings approximate Chamfer.
#[derive(Debug, Clone)]
pub struct MuveraEncoder {
    config: MuveraConfig,
    /// `hyperplanes[rep][plane]` is a `dim`-length hyperplane normal.
    hyperplanes: Vec<Vec<Vec<f32>>>,
    /// `projections[rep][out]` is a `dim`-length Rademacher row; empty when
    /// projection is disabled (`d_proj == 0`).
    projections: Vec<Vec<Vec<f32>>>,
}

impl MuveraEncoder {
    /// Build an encoder from `config`, precomputing all hyperplanes and
    /// projection matrices.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::InvalidConfig`] when
    /// [`MuveraConfig::validate`] rejects the configuration.
    pub fn new(config: MuveraConfig) -> MuveraResult<Self> {
        config.validate()?;
        let dim = config.dim;
        let k_sim = config.k_sim as usize;
        let hyperplanes: Vec<Vec<Vec<f32>>> = (0..config.r_reps)
            .map(|rep| {
                let seed = derive_seed(config.seed, rep, PURPOSE_HYPERPLANE);
                (0..k_sim)
                    .map(|plane| {
                        (0..dim)
                            .map(|j| {
                                #[allow(clippy::cast_possible_truncation)]
                                let idx = (plane * dim + j) as u64;
                                #[allow(clippy::cast_possible_truncation)]
                                let entry = box_muller_sample(seed, idx) as f32;
                                entry
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let projections: Vec<Vec<Vec<f32>>> = if config.d_proj == 0 {
            Vec::new()
        } else {
            (0..config.r_reps)
                .map(|rep| {
                    let seed = derive_seed(config.seed, rep, PURPOSE_PROJECTION);
                    (0..config.d_proj)
                        .map(|out| {
                            (0..dim)
                                .map(|j| {
                                    #[allow(clippy::cast_possible_truncation)]
                                    let idx = (out * dim + j) as u64;
                                    rademacher_sign(seed, idx)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        };

        Ok(Self {
            config,
            hyperplanes,
            projections,
        })
    }

    /// Borrow the encoder's configuration.
    #[must_use]
    pub fn config(&self) -> &MuveraConfig {
        &self.config
    }

    /// The length every FDE produced by this encoder will have
    /// (`r_reps · 2^k_sim · inner_dim`).
    #[must_use]
    pub fn expected_dim(&self) -> usize {
        self.config.expected_dim()
    }

    /// The `SimHash` cell id (in `{0, …, 2^k_sim − 1}`) that `token` falls into
    /// for repetition `rep`.
    ///
    /// Bit `p` of the id is set when the token's dot product with hyperplane
    /// `p` is non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::DimensionMismatch`] when `token.len()` differs
    /// from [`MuveraConfig::dim`], or [`MuveraError::InvalidConfig`] when `rep`
    /// is out of range.
    pub fn token_bucket(&self, token: &[f32], rep: usize) -> MuveraResult<usize> {
        if token.len() != self.config.dim {
            return Err(MuveraError::DimensionMismatch {
                expected: self.config.dim,
                got: token.len(),
            });
        }
        let planes = self.hyperplanes.get(rep).ok_or_else(|| {
            MuveraError::InvalidConfig(format!(
                "repetition {rep} out of range (r_reps = {})",
                self.config.r_reps
            ))
        })?;
        Ok(Self::bucket_of(planes, token))
    }

    /// The `SimHash` cell id of `token` against a repetition's `planes`.
    #[inline]
    fn bucket_of(planes: &[Vec<f32>], token: &[f32]) -> usize {
        let mut bucket = 0usize;
        for (p, plane) in planes.iter().enumerate() {
            let dot: f32 = plane.iter().zip(token.iter()).map(|(h, x)| h * x).sum();
            if dot >= 0.0 {
                bucket |= 1usize << p;
            }
        }
        bucket
    }

    /// Project a `dim`-length cell block down to `inner_dim` via the
    /// repetition's Rademacher matrix (scaled by `1/√d_proj`), or return the
    /// block unchanged when projection is disabled.
    #[inline]
    fn project(&self, block: &[f32], rep: usize) -> Vec<f32> {
        if self.config.d_proj == 0 {
            return block.to_vec();
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0f32 / (self.config.d_proj as f32).sqrt();
        self.projections[rep]
            .iter()
            .map(|row| {
                let s: f32 = row.iter().zip(block.iter()).map(|(w, x)| w * x).sum();
                s * scale
            })
            .collect()
    }

    /// Validate that a multi-vector set is non-empty and every token has the
    /// configured dimensionality.
    fn validate_set(&self, tokens: &[Vec<f32>]) -> MuveraResult<()> {
        if tokens.is_empty() {
            return Err(MuveraError::EmptyMultiVector);
        }
        for token in tokens {
            if token.len() != self.config.dim {
                return Err(MuveraError::DimensionMismatch {
                    expected: self.config.dim,
                    got: token.len(),
                });
            }
        }
        Ok(())
    }

    /// Encode a **query** multi-vector set. Each `SimHash` cell aggregates its
    /// tokens by **sum**; empty cells stay zero (only the document side fills).
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::EmptyMultiVector`] when `tokens` is empty, or
    /// [`MuveraError::DimensionMismatch`] when any token's length differs from
    /// [`MuveraConfig::dim`].
    pub fn encode_query(&self, tokens: &[Vec<f32>]) -> MuveraResult<FixedDimEncoding> {
        self.validate_set(tokens)?;
        let dim = self.config.dim;
        let buckets = self.config.num_buckets();
        let mut values = Vec::with_capacity(self.expected_dim());

        for rep in 0..self.config.r_reps {
            let planes = &self.hyperplanes[rep];
            let mut cells = vec![vec![0.0f32; dim]; buckets];
            for token in tokens {
                let bucket = Self::bucket_of(planes, token);
                for (slot, &value) in cells[bucket].iter_mut().zip(token.iter()) {
                    *slot += value;
                }
            }
            for cell in &cells {
                values.extend(self.project(cell, rep));
            }
        }

        Ok(FixedDimEncoding::new(values, self.config.inner_dim()))
    }

    /// Encode a **document** multi-vector set. Each `SimHash` cell aggregates its
    /// tokens by **average** (centroid); when
    /// [`MuveraConfig::fill_empty`] is set, empty cells borrow the centroid of
    /// the Hamming-nearest non-empty cell so all `B` blocks are populated.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::EmptyMultiVector`] when `tokens` is empty, or
    /// [`MuveraError::DimensionMismatch`] when any token's length differs from
    /// [`MuveraConfig::dim`].
    pub fn encode_document(&self, tokens: &[Vec<f32>]) -> MuveraResult<FixedDimEncoding> {
        self.validate_set(tokens)?;
        let dim = self.config.dim;
        let buckets = self.config.num_buckets();
        let mut values = Vec::with_capacity(self.expected_dim());

        for rep in 0..self.config.r_reps {
            let planes = &self.hyperplanes[rep];
            let mut cells = vec![vec![0.0f32; dim]; buckets];
            let mut counts = vec![0usize; buckets];
            for token in tokens {
                let bucket = Self::bucket_of(planes, token);
                for (slot, &value) in cells[bucket].iter_mut().zip(token.iter()) {
                    *slot += value;
                }
                counts[bucket] += 1;
            }

            let mut occupied = vec![false; buckets];
            for bucket in 0..buckets {
                if counts[bucket] > 0 {
                    occupied[bucket] = true;
                    #[allow(clippy::cast_precision_loss)]
                    let count = counts[bucket] as f32;
                    for slot in &mut cells[bucket] {
                        *slot /= count;
                    }
                }
            }

            if self.config.fill_empty {
                for bucket in 0..buckets {
                    if occupied[bucket] {
                        continue;
                    }
                    if let Some(src) = nearest_occupied_bucket(bucket, &occupied, self.config.k_sim)
                    {
                        let borrowed = cells[src].clone();
                        cells[bucket] = borrowed;
                    }
                }
            }

            for cell in &cells {
                values.extend(self.project(cell, rep));
            }
        }

        Ok(FixedDimEncoding::new(values, self.config.inner_dim()))
    }

    /// The exact Chamfer / `MaxSim` similarity `Σ_{q∈Q} max_{d∈D} ⟨q, d⟩`
    /// between a query set and a document set — the ground truth the FDE dot
    /// product approximates.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::EmptyMultiVector`] when either set is empty, or
    /// [`MuveraError::DimensionMismatch`] when any token's length differs from
    /// [`MuveraConfig::dim`].
    pub fn chamfer(&self, query: &[Vec<f32>], doc: &[Vec<f32>]) -> MuveraResult<f32> {
        self.validate_set(query)?;
        self.validate_set(doc)?;
        Ok(chamfer_similarity(query, doc))
    }
}
