//! Core types for the `rabitq` module: configuration, error type, the packed
//! quantized code record and the search-hit record.
//!
//! ## Notation (following Gao & Long, *`RaBitQ`: Quantizing High-Dimensional
//! Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor
//! Search*, SIGMOD 2024)
//!
//! | Symbol | Meaning |
//! |--------|---------|
//! | `D` | Dimensionality of the (residual) vectors |
//! | `c` | Dataset centroid |
//! | `o_raw` | A raw database vector |
//! | `o = o_raw - c` | Centroid residual |
//! | `o_unit = o / ‖o‖` | Unit-normalized residual |
//! | `P` | Deterministic pseudo-random `D×D` orthogonal (rotation) matrix |
//! | `o' = P · o_unit` | Rotated unit residual |
//! | `b_i = 1[o'_i > 0]` | Per-dimension quantization bit |
//! | `x̄_i = (2·b_i - 1)/√D` | Quantized codebook vertex (unit norm) |
//! | `⟨x̄, o'⟩` | Per-vector normalization factor stored alongside the code |

use thiserror::Error;

// ── DEFAULT_RABITQ_SEED ───────────────────────────────────────────────────────

/// The default, fixed seed used to deterministically derive the random
/// orthogonal rotation matrix `P` when a caller does not override
/// [`RaBitQConfig::with_seed`].
///
/// Fixed (not time- or entropy-derived) so that repeated `train` calls with an
/// identical configuration reproduce a byte-identical rotation and, therefore,
/// byte-identical codes.
pub const DEFAULT_RABITQ_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

// ── RaBitQMetric ──────────────────────────────────────────────────────────────

/// Distance metric used to rank [`RaBitQIndex`](super::index::RaBitQIndex)
/// search results.
///
/// Regardless of metric, [`RaBitQHit::estimated_distance`] follows the
/// "smaller is closer" convention: for [`InnerProduct`](Self::InnerProduct)
/// and [`Cosine`](Self::Cosine) the stored value is the *negated* similarity
/// so that ascending sort still surfaces the best matches first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RaBitQMetric {
    /// Squared Euclidean (L2) distance between the query and the raw indexed
    /// vector, reconstructed exactly (up to estimator error) via
    /// [`RaBitQuantizer::estimate_l2_sq`](super::quantizer::RaBitQuantizer::estimate_l2_sq).
    /// Smaller is closer. This is the default metric.
    #[default]
    L2,
    /// Estimated raw inner product `⟨o_raw, query⟩` via
    /// [`RaBitQuantizer::estimate_ip`](super::quantizer::RaBitQuantizer::estimate_ip).
    /// Larger is closer.
    InnerProduct,
    /// Cosine similarity between the *centroid residuals* `(o_raw - c)` and
    /// `(query - c)` via
    /// [`RaBitQuantizer::estimate_cosine`](super::quantizer::RaBitQuantizer::estimate_cosine).
    /// Larger is closer.
    Cosine,
}

// ── RaBitQConfig ──────────────────────────────────────────────────────────────

/// Configuration for a [`RaBitQuantizer`](super::quantizer::RaBitQuantizer) /
/// [`RaBitQIndex`](super::index::RaBitQIndex).
///
/// # Defaults
///
/// | Field | Default |
/// |-------|---------|
/// | [`seed`](Self::seed) | [`DEFAULT_RABITQ_SEED`] |
/// | [`metric`](Self::metric) | [`RaBitQMetric::L2`] |
///
/// `dim` has no sensible default (it must match the embedding dimensionality
/// of the caller's vectors), so it is supplied to the validating constructor
/// [`RaBitQConfig::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaBitQConfig {
    /// Dimensionality of the input vectors. Must be greater than zero.
    pub dim: usize,
    /// Seed for the deterministic pseudo-random rotation matrix `P`.
    ///
    /// Defaults to [`DEFAULT_RABITQ_SEED`]. Any two [`RaBitQuantizer`]s
    /// trained with the same `(dim, seed)` pair derive an identical `P`.
    ///
    /// [`RaBitQuantizer`]: super::quantizer::RaBitQuantizer
    pub seed: u64,
    /// Distance metric used by [`RaBitQIndex::search`](super::index::RaBitQIndex::search).
    ///
    /// Defaults to [`RaBitQMetric::L2`].
    pub metric: RaBitQMetric,
}

impl RaBitQConfig {
    /// Create a new configuration for `dim`-dimensional vectors, with the
    /// default seed ([`DEFAULT_RABITQ_SEED`]) and metric
    /// ([`RaBitQMetric::L2`]).
    ///
    /// # Errors
    ///
    /// Returns [`RaBitQError::InvalidConfig`] when `dim == 0`.
    pub fn new(dim: usize) -> Result<Self, RaBitQError> {
        if dim == 0 {
            return Err(RaBitQError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }
        Ok(Self {
            dim,
            seed: DEFAULT_RABITQ_SEED,
            metric: RaBitQMetric::L2,
        })
    }

    /// Override the rotation seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Override the ranking metric.
    #[must_use]
    pub fn with_metric(mut self, metric: RaBitQMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RaBitQError::InvalidConfig`] when `dim == 0`. Useful for
    /// re-validating a configuration that was constructed by hand (e.g. via
    /// struct-update syntax) rather than through [`RaBitQConfig::new`].
    pub fn validate(&self) -> Result<(), RaBitQError> {
        if self.dim == 0 {
            return Err(RaBitQError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

impl Default for RaBitQConfig {
    /// A convenience default for `dim = 128`. `dim` is always non-zero here,
    /// so this never panics and mirrors [`RaBitQConfig::new`]'s validation
    /// trivially.
    fn default() -> Self {
        Self {
            dim: 128,
            seed: DEFAULT_RABITQ_SEED,
            metric: RaBitQMetric::L2,
        }
    }
}

// ── RaBitQError ───────────────────────────────────────────────────────────────

/// Errors produced by the `rabitq` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum RaBitQError {
    /// A vector's length did not match the configured dimensionality.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The expected dimensionality (from [`RaBitQConfig::dim`]).
        expected: usize,
        /// The actual length of the offending vector.
        got: usize,
    },
    /// The supplied configuration is invalid (e.g. `dim == 0`, or a rotation
    /// seed/dim pair produced a degenerate basis).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// An operation requiring a trained quantizer was attempted before
    /// [`RaBitQuantizer::train`](super::quantizer::RaBitQuantizer::train).
    #[error("quantizer has not been trained — call train() first")]
    NotTrained,
    /// [`RaBitQIndex::build`](super::index::RaBitQIndex::build) was called
    /// with no items, or [`RaBitQIndex::search`](super::index::RaBitQIndex::search)
    /// was called on an empty index.
    #[error("index is empty — build() requires at least one item")]
    EmptyIndex,
    /// A query vector was empty (zero-length).
    #[error("query vector is empty")]
    EmptyQuery,
}

// ── RaBitQCode ────────────────────────────────────────────────────────────────

/// A single `RaBitQ`-quantized vector: a packed 1-bit code plus the two scalars
/// needed to reconstruct inner-product / L2 estimates against the raw vector.
///
/// # Fields
///
/// - [`bits`](Self::bits): the packed sign pattern `b_i = 1[o'_i > 0]` of the
///   rotated unit residual `o' = P · o_unit`, packed LSB-first into `u64`
///   words (bit `i` lives in word `i / 64` at bit position `i % 64`). The
///   codebook vector is recovered on demand as `x̄_i = (2·b_i - 1)/√D`; it is
///   never itself stored.
/// - [`factor`](Self::factor): `⟨x̄, o'⟩`, the cosine between the quantized
///   codebook vertex and the true rotated unit residual. This is always
///   non-negative (each term `x̄_i · o'_i = |o'_i| / √D ≥ 0`) and is the
///   normalization the unbiased estimator divides by.
/// - [`norm`](Self::norm): `‖o_raw - c‖`, the Euclidean length of the
///   centroid residual, needed to rescale the unit-sphere estimate back to
///   the original vector's magnitude.
#[derive(Debug, Clone, PartialEq)]
pub struct RaBitQCode {
    /// Packed 1-bit sign code, LSB-first, `⌈D/64⌉` words long.
    pub bits: Vec<u64>,
    /// `⟨x̄, o'⟩`: normalization factor (always `>= 0`).
    pub factor: f32,
    /// `‖o_raw - c‖`: centroid-residual norm.
    pub norm: f32,
}

impl RaBitQCode {
    /// Construct a code from its raw parts.
    #[must_use]
    pub fn new(bits: Vec<u64>, factor: f32, norm: f32) -> Self {
        Self { bits, factor, norm }
    }

    /// Maximum number of dimensions this code's packed [`bits`](Self::bits)
    /// can represent (`64 * bits.len()`).
    #[must_use]
    pub fn bit_capacity(&self) -> usize {
        self.bits.len() * 64
    }
}

// ── RaBitQHit ─────────────────────────────────────────────────────────────────

/// A single ranked result from [`RaBitQIndex::search`](super::index::RaBitQIndex::search).
#[derive(Debug, Clone, PartialEq)]
pub struct RaBitQHit {
    /// Identifier of the matched item.
    pub id: String,
    /// The metric-dependent estimated distance (see [`RaBitQMetric`] for the
    /// "smaller is closer" sign convention). Smaller values are always
    /// ranked first by [`RaBitQIndex::search`](super::index::RaBitQIndex::search).
    pub estimated_distance: f32,
}

impl RaBitQHit {
    /// Construct a new hit.
    #[must_use]
    pub fn new(id: String, estimated_distance: f32) -> Self {
        Self {
            id,
            estimated_distance,
        }
    }
}
