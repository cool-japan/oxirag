//! The `RaBitQ` quantizer: training (centroid + rotation), 1-bit encoding, and
//! the unbiased inner-product / L2 estimator.
//!
//! ## The estimator
//!
//! For a fixed indexed vector, let `o_unit = (o_raw - c)/‖o_raw - c‖` be its
//! unit centroid residual, `o' = P · o_unit` its rotation, `b_i = 1[o'_i > 0]`
//! the stored sign bits and `x̄_i = (2 b_i - 1)/√D` the recovered codebook
//! vertex. `RaBitQ`'s key identity is that for *any* vector `v` (not just the
//! vector that was quantized), rotating `v` with the *same* `P` and taking
//! the ratio
//!
//! ```text
//! est(v) = ⟨x̄, P·v⟩ / ⟨x̄, o'⟩
//! ```
//!
//! is an unbiased estimator of `⟨o_unit, v⟩`, with estimation error
//! concentrating at scale `O(1/√(D-1))` (Gao & Long, Theorem 3.3). The
//! denominator `⟨x̄, o'⟩` is exactly the `factor` stored in [`RaBitQCode`],
//! precomputed once at encode time; the numerator is computed fresh for
//! whatever `v` the caller asks about, using the *same* random rotation `P`
//! (this is what makes the estimator well-defined for arbitrary `v`, not only
//! for `v = o_unit` itself — the isometry `⟨P a, P b⟩ = ⟨a, b⟩` holds for any
//! `a, b`, in particular for `a = o_unit, b = v`).
//!
//! This module instantiates `est(v)` twice, with two different choices of `v`:
//!
//! - `v = query` (the raw query, uncentered) → estimates `⟨o_unit, query⟩`,
//!   used by [`estimate_ip`](RaBitQuantizer::estimate_ip) together with the
//!   exact cross term `⟨c, query⟩` to reconstruct `⟨o_raw, query⟩`.
//! - `v = query - c` (the centered residual query) → estimates
//!   `⟨o_unit, query - c⟩`, used by
//!   [`estimate_l2_sq`](RaBitQuantizer::estimate_l2_sq) together with the
//!   identity `‖o_raw - query‖² = ‖o - (query-c)‖²` (the centroid cancels
//!   exactly in a difference of raw vectors) to reconstruct the *exact* raw
//!   squared L2 distance up to the estimator's own approximation error.

use super::rotation::Rotation;
use super::types::{RaBitQCode, RaBitQConfig, RaBitQError};

// ── Bit packing helpers ───────────────────────────────────────────────────────

/// Pack a slice of booleans into `u64` words, LSB-first (bit `i` lives in
/// word `i / 64` at bit position `i % 64`).
fn pack_bits(signs: &[bool]) -> Vec<u64> {
    let num_words = signs.len().div_ceil(64);
    let mut words = vec![0u64; num_words];
    for (i, &s) in signs.iter().enumerate() {
        if s {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    words
}

/// Read bit `i` from a packed LSB-first `u64` word array. Out-of-range `i`
/// (beyond the packed word count) reads as `false`.
fn unpack_bit(bits: &[u64], i: usize) -> bool {
    let word = bits.get(i / 64).copied().unwrap_or(0);
    (word >> (i % 64)) & 1 == 1
}

/// The recovered codebook component `x̄_i = (2·b_i - 1)/√D`, where `b_i` is
/// [`unpack_bit`]`(bits, i)` and `inv_sqrt_dim = 1/√D`.
fn x_bar_component(bits: &[u64], i: usize, inv_sqrt_dim: f32) -> f32 {
    if unpack_bit(bits, i) {
        inv_sqrt_dim
    } else {
        -inv_sqrt_dim
    }
}

/// `⟨x̄, v⟩ = Σ_i x̄_i · v_i`, computed on the fly from the packed `bits`
/// without ever materializing `x̄` as a `Vec`.
fn dot_with_bits(bits: &[u64], v: &[f32], inv_sqrt_dim: f32) -> f32 {
    v.iter()
        .enumerate()
        .map(|(i, &x)| x_bar_component(bits, i, inv_sqrt_dim) * x)
        .sum()
}

// ── RaBitQuantizer ────────────────────────────────────────────────────────────

/// `RaBitQ` quantizer: holds the dataset centroid `c`, the random rotation `P`,
/// and the configuration.
///
/// # Workflow
///
/// 1. Create with [`RaBitQuantizer::new`] and a validated [`RaBitQConfig`].
/// 2. Call [`train`](Self::train) with a representative set of vectors to
///    compute the centroid and derive the rotation; this returns a *new*,
///    fully trained [`RaBitQuantizer`].
/// 3. Use [`encode`](Self::encode) to quantize vectors, and
///    [`estimate_ip`](Self::estimate_ip) / [`estimate_l2_sq`](Self::estimate_l2_sq) /
///    [`estimate_cosine`](Self::estimate_cosine) to score a query against a
///    stored [`RaBitQCode`].
#[derive(Debug, Clone)]
pub struct RaBitQuantizer {
    config: RaBitQConfig,
    /// Dataset centroid `c`; empty until [`train`](Self::train) has run.
    centroid: Vec<f32>,
    /// Random rotation `P`; `None` until [`train`](Self::train) has run.
    rotation: Option<Rotation>,
}

impl RaBitQuantizer {
    /// Create a new, untrained quantizer for the given configuration.
    ///
    /// This constructor never fails; [`RaBitQConfig`] is validated by its own
    /// constructor and re-checked in [`train`](Self::train).
    #[must_use]
    pub fn new(config: RaBitQConfig) -> Self {
        Self {
            config,
            centroid: Vec::new(),
            rotation: None,
        }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &RaBitQConfig {
        &self.config
    }

    /// `true` once [`train`](Self::train) has produced a centroid and rotation.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.rotation.is_some()
    }

    /// Borrow the dataset centroid (empty until trained).
    #[must_use]
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Borrow the trained rotation, if any.
    #[must_use]
    pub fn rotation(&self) -> Option<&Rotation> {
        self.rotation.as_ref()
    }

    /// Train on a representative set of vectors: computes the centroid `c` as
    /// their arithmetic mean, then derives the random rotation `P` from
    /// `(config.dim, config.seed)`. Returns a new, fully trained
    /// [`RaBitQuantizer`] (this quantizer's own configuration is left
    /// unmodified; the trained state lives in the returned value).
    ///
    /// # Errors
    ///
    /// - [`RaBitQError::InvalidConfig`] when [`RaBitQConfig::validate`] fails.
    /// - [`RaBitQError::EmptyIndex`] when `vectors` is empty.
    /// - [`RaBitQError::DimensionMismatch`] when any vector's length differs
    ///   from [`RaBitQConfig::dim`].
    pub fn train(&self, vectors: &[Vec<f32>]) -> Result<Self, RaBitQError> {
        self.config.validate()?;
        if vectors.is_empty() {
            return Err(RaBitQError::EmptyIndex);
        }
        let dim = self.config.dim;
        for v in vectors {
            if v.len() != dim {
                return Err(RaBitQError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        let mut centroid = vec![0.0f32; dim];
        for v in vectors {
            for (acc, &x) in centroid.iter_mut().zip(v.iter()) {
                *acc += x;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let inv_n = 1.0f32 / vectors.len() as f32;
        for acc in &mut centroid {
            *acc *= inv_n;
        }

        let rotation = Rotation::generate(dim, self.config.seed)?;

        Ok(Self {
            config: self.config,
            centroid,
            rotation: Some(rotation),
        })
    }

    /// Quantize a single raw vector `v` to a [`RaBitQCode`].
    ///
    /// Computes the centroid residual `o = v - c`, its norm `‖o‖`, the unit
    /// direction `o_unit = o/‖o‖`, rotates it (`o' = P · o_unit`), extracts the
    /// sign bits `b_i = 1[o'_i > 0]`, and stores `factor = ⟨x̄, o'⟩` alongside
    /// `norm = ‖o‖`.
    ///
    /// ## Degenerate residual (`v` coincides with the centroid)
    ///
    /// When `‖o‖ ≈ 0` the residual direction is mathematically undefined, but
    /// this must still produce a usable code (e.g. a single-item index has
    /// its lone vector *equal* to the centroid by construction). In that case
    /// a fixed canonical direction (the first standard basis vector `e_0`) is
    /// substituted for `o_unit`. This substitution is provably inconsequential:
    /// every downstream estimate multiplies the estimated cosine by the
    /// *stored, unaltered* `norm ≈ 0`, so the arbitrary direction contributes
    /// exactly `0` to [`estimate_ip`](Self::estimate_ip) and
    /// [`estimate_l2_sq`](Self::estimate_l2_sq) regardless of which direction
    /// was chosen.
    ///
    /// # Errors
    ///
    /// - [`RaBitQError::NotTrained`] when called before [`train`](Self::train).
    /// - [`RaBitQError::DimensionMismatch`] when `v.len() != config.dim`.
    pub fn encode(&self, v: &[f32]) -> Result<RaBitQCode, RaBitQError> {
        let rotation = self.rotation.as_ref().ok_or(RaBitQError::NotTrained)?;
        let dim = self.config.dim;
        if v.len() != dim {
            return Err(RaBitQError::DimensionMismatch {
                expected: dim,
                got: v.len(),
            });
        }

        let residual: Vec<f32> = v
            .iter()
            .zip(self.centroid.iter())
            .map(|(&x, &c)| x - c)
            .collect();
        let norm = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = if norm < 1e-9 {
            // Degenerate direction: substitute e_0. Provably harmless — see
            // the doc comment above — because `norm` (stored unmodified) is
            // what scales this direction's contribution to zero everywhere
            // it is used downstream.
            let mut e0 = vec![0.0f32; dim];
            e0[0] = 1.0;
            e0
        } else {
            residual.iter().map(|&x| x / norm).collect()
        };
        let rotated = rotation.apply(&unit);
        let signs: Vec<bool> = rotated.iter().map(|&x| x > 0.0).collect();
        let bits = pack_bits(&signs);

        #[allow(clippy::cast_precision_loss)]
        let inv_sqrt_dim = 1.0f32 / (dim as f32).sqrt();
        let factor = dot_with_bits(&bits, &rotated, inv_sqrt_dim);

        Ok(RaBitQCode::new(bits, factor, norm))
    }

    /// The theoretical error-bound scale for the unbiased estimator,
    /// `1/√D` (Gao & Long's analysis shows the estimation error concentrates
    /// at scale `O(1/√(D-1))` with high probability; we expose the simpler
    /// `1/√D` as a practical confidence-radius heuristic).
    ///
    /// Callers can treat `est ± error_bound()` as an approximate confidence
    /// band around any value returned by [`estimate_ip`](Self::estimate_ip),
    /// [`estimate_l2_sq`](Self::estimate_l2_sq) or
    /// [`estimate_cosine`](Self::estimate_cosine) (the latter two being
    /// derived from the same underlying cosine estimator, scaled by exactly
    /// known quantities).
    #[must_use]
    pub fn error_bound(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let dim = self.config.dim as f32;
        1.0 / dim.sqrt()
    }

    /// Core estimator: `est(v) = ⟨x̄, P·v⟩ / ⟨x̄, o'⟩`, an unbiased estimate of
    /// `⟨o_unit, v⟩` for the vector `v` supplied by the caller (already in
    /// original, un-rotated coordinates).
    ///
    /// Returns `0.0` for the measure-zero degenerate case `factor ≈ 0` (no
    /// directional signal can be extracted from such a code).
    fn estimate_unit_projection(&self, code: &RaBitQCode, v: &[f32]) -> Result<f32, RaBitQError> {
        let rotation = self.rotation.as_ref().ok_or(RaBitQError::NotTrained)?;
        let dim = self.config.dim;
        if v.len() != dim {
            return Err(RaBitQError::DimensionMismatch {
                expected: dim,
                got: v.len(),
            });
        }
        let rotated_v = rotation.apply(v);
        #[allow(clippy::cast_precision_loss)]
        let inv_sqrt_dim = 1.0f32 / (dim as f32).sqrt();
        let numerator = dot_with_bits(&code.bits, &rotated_v, inv_sqrt_dim);
        if code.factor.abs() < 1e-12 {
            return Ok(0.0);
        }
        Ok(numerator / code.factor)
    }

    /// Estimate the raw inner product `⟨o_raw, query⟩`.
    ///
    /// Uses `est(query) ≈ ⟨o_unit, query⟩` (the core estimator applied
    /// directly to the *uncentered* query) together with the exact identity
    ///
    /// ```text
    /// ⟨o_raw, query⟩ = ⟨c + ‖o‖·o_unit, query⟩ = ⟨c, query⟩ + ‖o‖ · ⟨o_unit, query⟩
    /// ```
    ///
    /// where `⟨c, query⟩` is computed exactly (both are full-precision) and
    /// `‖o‖` is the code's stored `norm`.
    ///
    /// # Errors
    ///
    /// - [`RaBitQError::NotTrained`] when called before [`train`](Self::train).
    /// - [`RaBitQError::EmptyQuery`] when `query` is empty.
    /// - [`RaBitQError::DimensionMismatch`] when `query.len() != config.dim`.
    pub fn estimate_ip(&self, query: &[f32], code: &RaBitQCode) -> Result<f32, RaBitQError> {
        if query.is_empty() {
            return Err(RaBitQError::EmptyQuery);
        }
        if !self.is_trained() {
            return Err(RaBitQError::NotTrained);
        }
        let est_cos_raw = self.estimate_unit_projection(code, query)?;
        let centroid_dot: f32 = self
            .centroid
            .iter()
            .zip(query.iter())
            .map(|(&c, &q)| c * q)
            .sum();
        Ok(centroid_dot + code.norm * est_cos_raw)
    }

    /// Estimate the squared Euclidean distance `‖o_raw - query‖²`.
    ///
    /// Uses `est(query - c) ≈ ⟨o_unit, query - c⟩` together with the exact
    /// identity (the centroid cancels in a difference of two raw vectors)
    ///
    /// ```text
    /// ‖o_raw - query‖² = ‖o - (query - c)‖²
    ///                  = ‖o‖² + ‖query - c‖² - 2·‖o‖·⟨o_unit, query - c⟩
    /// ```
    ///
    /// The result is clamped to `0.0` to guard against tiny negative values
    /// from floating-point / estimator error.
    ///
    /// # Errors
    ///
    /// - [`RaBitQError::NotTrained`] when called before [`train`](Self::train).
    /// - [`RaBitQError::EmptyQuery`] when `query` is empty.
    /// - [`RaBitQError::DimensionMismatch`] when `query.len() != config.dim`.
    pub fn estimate_l2_sq(&self, query: &[f32], code: &RaBitQCode) -> Result<f32, RaBitQError> {
        if query.is_empty() {
            return Err(RaBitQError::EmptyQuery);
        }
        if !self.is_trained() {
            return Err(RaBitQError::NotTrained);
        }
        if query.len() != self.config.dim {
            return Err(RaBitQError::DimensionMismatch {
                expected: self.config.dim,
                got: query.len(),
            });
        }
        let residual: Vec<f32> = query
            .iter()
            .zip(self.centroid.iter())
            .map(|(&q, &c)| q - c)
            .collect();
        let est_cos_res = self.estimate_unit_projection(code, &residual)?;
        let q_res_sq: f32 = residual.iter().map(|x| x * x).sum();
        let dist = code.norm.mul_add(code.norm, q_res_sq) - 2.0 * code.norm * est_cos_res;
        Ok(dist.max(0.0))
    }

    /// Estimate the cosine similarity between the *centroid residuals*
    /// `(o_raw - c)` and `(query - c)`:
    ///
    /// ```text
    /// cos(o_raw - c, query - c) = ⟨o_unit, query - c⟩ / ‖query - c‖
    /// ```
    ///
    /// Returns `0.0` when `‖query - c‖` is (numerically) zero, since the
    /// residual direction of such a query is undefined.
    ///
    /// # Errors
    ///
    /// - [`RaBitQError::NotTrained`] when called before [`train`](Self::train).
    /// - [`RaBitQError::EmptyQuery`] when `query` is empty.
    /// - [`RaBitQError::DimensionMismatch`] when `query.len() != config.dim`.
    pub fn estimate_cosine(&self, query: &[f32], code: &RaBitQCode) -> Result<f32, RaBitQError> {
        if query.is_empty() {
            return Err(RaBitQError::EmptyQuery);
        }
        if !self.is_trained() {
            return Err(RaBitQError::NotTrained);
        }
        if query.len() != self.config.dim {
            return Err(RaBitQError::DimensionMismatch {
                expected: self.config.dim,
                got: query.len(),
            });
        }
        let residual: Vec<f32> = query
            .iter()
            .zip(self.centroid.iter())
            .map(|(&q, &c)| q - c)
            .collect();
        let residual_norm = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
        if residual_norm < 1e-9 {
            return Ok(0.0);
        }
        let est_cos_res = self.estimate_unit_projection(code, &residual)?;
        Ok(est_cos_res / residual_norm)
    }
}
