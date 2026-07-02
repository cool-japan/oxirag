//! Core types for the `anisotropic_vq` module: configuration, error type, the
//! quantized code record, the ranked search-hit record and the ranking metric.
//!
//! ## Notation (following Guo, Sun, Lindgren, Geng, Simcha, Chern & Kumar,
//! *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*,
//! ICML 2020 — a.k.a. `ScaNN`)
//!
//! | Symbol | Meaning |
//! |--------|---------|
//! | `x` | A raw database (training) vector |
//! | `x̂ = x / ‖x‖` | Unit direction of `x` (undefined for `x = 0`) |
//! | `q` | The codeword (centroid) a vector is quantized to |
//! | `e = x − q` | Reconstruction residual |
//! | `e_∥ = (e · x̂) x̂` | Component of the residual **parallel** to `x̂` |
//! | `e_⊥ = e − e_∥` | Component of the residual **orthogonal** to `x̂` |
//! | `h_∥` | Weight on the parallel residual (`parallel_weight_multiplier`) |
//! | `h_⊥` | Weight on the orthogonal residual (fixed at `1.0`) |
//! | `W_x = h_∥ x̂x̂ᵀ + h_⊥(I − x̂x̂ᵀ)` | Per-point anisotropic weight matrix |

use thiserror::Error;

// ── DEFAULT_ANISOTROPIC_VQ_SEED ─────────────────────────────────────────────────

/// The default, fixed seed used to deterministically drive k-means++ codeword
/// initialization when a caller does not override
/// [`AnisotropicVqConfig::with_seed`].
///
/// Fixed (not time- or entropy-derived) so that repeated `train` calls with an
/// identical configuration reproduce a byte-identical codebook and, therefore,
/// byte-identical codes. The value is the golden-ratio odd constant used
/// throughout this crate's deterministic hashing.
pub const DEFAULT_ANISOTROPIC_VQ_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// The largest codebook size representable as a [`u16`] codeword index.
///
/// A code stores one [`u16`] index per subspace, so the number of codewords
/// per subspace must not exceed `65536` (indices `0..=65535`).
pub const MAX_CODEBOOK_SIZE: usize = (u16::MAX as usize) + 1;

// ── AnisotropicVqMetric ─────────────────────────────────────────────────────────

/// Distance metric used to rank
/// [`AnisotropicVqIndex`](super::index::AnisotropicVqIndex) search results.
///
/// Regardless of metric, [`AnisotropicVqHit::estimated_distance`] follows the
/// crate-wide "smaller is closer" convention: for
/// [`InnerProduct`](Self::InnerProduct) the stored value is the **negated**
/// estimated inner product so that an ascending sort still surfaces the best
/// (largest inner product) matches first.
///
/// Note that the metric only affects *query-time scoring*. The anisotropic
/// reweighting shapes the learned codebook itself and is applied identically
/// regardless of the ranking metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnisotropicVqMetric {
    /// Maximum inner-product search (MIPS). The estimated inner product
    /// `⟨query, x⟩ ≈ ⟨query, decode(code)⟩` is computed by asymmetric lookup;
    /// larger is closer. This is the metric anisotropic VQ is designed for and
    /// is the default.
    #[default]
    InnerProduct,
    /// Squared Euclidean (L2) distance between the query and the reconstructed
    /// (decoded) vector, computed by asymmetric lookup; smaller is closer.
    L2,
}

// ── AnisotropicVqConfig ─────────────────────────────────────────────────────────

/// Configuration for an
/// [`AnisotropicQuantizer`](super::quantizer::AnisotropicQuantizer) /
/// [`AnisotropicVqIndex`](super::index::AnisotropicVqIndex).
///
/// # The `parallel_weight_multiplier` knob
///
/// This is the defining parameter of anisotropic vector quantization. It sets
/// `h_∥ = parallel_weight_multiplier` while `h_⊥` is fixed at `1.0`, so a value
/// of `1.0` recovers ordinary (isotropic) MSE k-means and larger values
/// penalize residual error *parallel* to each vector's own direction more
/// heavily than *orthogonal* error.
///
/// **Why penalize the parallel component more?** For inner-product / cosine
/// ranking, the queries that rank a database vector `x` highly point in roughly
/// the same direction as `x`. An error component along `x̂` therefore distorts
/// the estimated score for exactly those high-scoring queries, whereas an
/// orthogonal error mostly averages out. `ScaNN` shows that reshaping the
/// quantization loss this way substantially improves MIPS recall at a fixed
/// bitrate.
///
/// **Choice of default (`4.0`).** `ScaNN` derives an *optimal* weight ratio that
/// grows with dimensionality; there is no single universal constant. `4.0` is a
/// deliberately moderate, principled default: it penalizes parallel error four
/// times more than orthogonal error — enough to meaningfully bias the codebook
/// toward preserving inner products, yet not so aggressive that orthogonal
/// fidelity collapses (which would hurt recall for queries only loosely aligned
/// with the database vector). For high-dimensional embeddings (hundreds of
/// dims) a larger value (roughly `8`–`16`) typically tracks the paper's optimal
/// ratio more closely; the knob is exposed precisely so callers can tune it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnisotropicVqConfig {
    /// Dimensionality of the input vectors. Must be greater than zero and an
    /// integer multiple of [`num_subspaces`](Self::num_subspaces).
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Number of codewords per subspace (`K`). Each subspace learns its own
    /// codebook of this many centroids. Must be in `1..=65536`.
    ///
    /// Defaults to `256`.
    pub num_codewords: usize,
    /// Number of contiguous subspaces the input is split into. `1` is a single
    /// full-vector codebook; values `> 1` enable product-style anisotropic
    /// quantization, learning one anisotropic codebook per subvector (each
    /// subvector reweighted by *its own* direction). Must evenly divide
    /// [`dim`](Self::dim).
    ///
    /// Defaults to `1`.
    pub num_subspaces: usize,
    /// Parallel-residual weight `h_∥` (with `h_⊥ = 1.0`). Must be finite and
    /// `>= 1.0`. See the type-level documentation for the rationale and the
    /// choice of default.
    ///
    /// Defaults to `4.0`.
    pub parallel_weight_multiplier: f64,
    /// Maximum number of weighted Lloyd (k-means) iterations per subspace.
    /// Must be `>= 1`.
    ///
    /// Defaults to `25`.
    pub max_iterations: usize,
    /// Relative-improvement threshold for early stopping: training halts once
    /// `(prev_loss - loss) / |prev_loss|` drops below this value. Must be
    /// finite and `>= 0.0` (use `0.0` to run the full iteration budget).
    ///
    /// Defaults to `1e-4`.
    pub convergence_tolerance: f64,
    /// Seed for deterministic k-means++ codeword initialization.
    ///
    /// Defaults to [`DEFAULT_ANISOTROPIC_VQ_SEED`].
    pub seed: u64,
    /// Metric used to rank search hits.
    ///
    /// Defaults to [`AnisotropicVqMetric::InnerProduct`].
    pub metric: AnisotropicVqMetric,
}

impl Default for AnisotropicVqConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            num_codewords: 256,
            num_subspaces: 1,
            parallel_weight_multiplier: 4.0,
            max_iterations: 25,
            convergence_tolerance: 1e-4,
            seed: DEFAULT_ANISOTROPIC_VQ_SEED,
            metric: AnisotropicVqMetric::InnerProduct,
        }
    }
}

impl AnisotropicVqConfig {
    /// Create a new configuration with the [default](Self::default) values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input vector dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the number of codewords per subspace (`K`).
    #[must_use]
    pub fn with_num_codewords(mut self, num_codewords: usize) -> Self {
        self.num_codewords = num_codewords;
        self
    }

    /// Set the number of subspaces (`1` = single full-vector codebook).
    #[must_use]
    pub fn with_num_subspaces(mut self, num_subspaces: usize) -> Self {
        self.num_subspaces = num_subspaces;
        self
    }

    /// Set the parallel-residual weight multiplier `h_∥`.
    #[must_use]
    pub fn with_parallel_weight_multiplier(mut self, multiplier: f64) -> Self {
        self.parallel_weight_multiplier = multiplier;
        self
    }

    /// Set the maximum number of weighted k-means iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the relative-improvement early-stopping threshold.
    #[must_use]
    pub fn with_convergence_tolerance(mut self, tolerance: f64) -> Self {
        self.convergence_tolerance = tolerance;
        self
    }

    /// Set the deterministic initialization seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the ranking metric.
    #[must_use]
    pub fn with_metric(mut self, metric: AnisotropicVqMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Length of each subvector (`dim / num_subspaces`).
    ///
    /// Returns `0` when `num_subspaces` is `0` (a configuration rejected by
    /// [`validate`](Self::validate)) to avoid a division-by-zero panic.
    #[must_use]
    pub fn subspace_dim(&self) -> usize {
        self.dim.checked_div(self.num_subspaces).unwrap_or(0)
    }

    /// The orthogonal weight `h_⊥`, fixed at `1.0`.
    #[must_use]
    pub fn orthogonal_weight(&self) -> f64 {
        1.0
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AnisotropicVqError::InvalidConfig`] when any field is out of
    /// range: `dim == 0`, `num_subspaces == 0`, `dim` not divisible by
    /// `num_subspaces`, `num_codewords` outside `1..=65536`,
    /// `parallel_weight_multiplier` not finite or `< 1.0`, `max_iterations == 0`,
    /// or `convergence_tolerance` not finite or `< 0.0`.
    pub fn validate(&self) -> Result<(), AnisotropicVqError> {
        if self.dim == 0 {
            return Err(AnisotropicVqError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }
        if self.num_subspaces == 0 {
            return Err(AnisotropicVqError::InvalidConfig(
                "num_subspaces must be greater than zero".into(),
            ));
        }
        if !self.dim.is_multiple_of(self.num_subspaces) {
            return Err(AnisotropicVqError::InvalidConfig(format!(
                "dim {} not divisible by num_subspaces {}",
                self.dim, self.num_subspaces
            )));
        }
        if self.num_codewords == 0 || self.num_codewords > MAX_CODEBOOK_SIZE {
            return Err(AnisotropicVqError::InvalidConfig(format!(
                "num_codewords {} must be in 1..={MAX_CODEBOOK_SIZE}",
                self.num_codewords
            )));
        }
        if !self.parallel_weight_multiplier.is_finite() || self.parallel_weight_multiplier < 1.0 {
            return Err(AnisotropicVqError::InvalidConfig(format!(
                "parallel_weight_multiplier {} must be finite and >= 1.0",
                self.parallel_weight_multiplier
            )));
        }
        if self.max_iterations == 0 {
            return Err(AnisotropicVqError::InvalidConfig(
                "max_iterations must be greater than zero".into(),
            ));
        }
        if !self.convergence_tolerance.is_finite() || self.convergence_tolerance < 0.0 {
            return Err(AnisotropicVqError::InvalidConfig(format!(
                "convergence_tolerance {} must be finite and >= 0.0",
                self.convergence_tolerance
            )));
        }
        Ok(())
    }
}

// ── AnisotropicCode ─────────────────────────────────────────────────────────────

/// An anisotropically-quantized vector: one codeword index per subspace.
///
/// The [`indices`](Self::indices) length equals
/// [`num_subspaces`](AnisotropicVqConfig::num_subspaces); for a single
/// full-vector codebook it holds exactly one entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnisotropicCode {
    /// Per-subspace codeword indices.
    pub indices: Vec<u16>,
}

impl AnisotropicCode {
    /// Create a new code from raw codeword indices.
    #[must_use]
    pub fn new(indices: Vec<u16>) -> Self {
        Self { indices }
    }

    /// Number of subspaces (codewords) this code spans.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Return `true` when the code holds no subspaces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

// ── AnisotropicVqHit ────────────────────────────────────────────────────────────

/// A single ranked search result from an
/// [`AnisotropicVqIndex`](super::index::AnisotropicVqIndex).
#[derive(Debug, Clone, PartialEq)]
pub struct AnisotropicVqHit {
    /// Identifier of the matched document.
    pub id: crate::types::DocumentId,
    /// Estimated distance under the configured
    /// [`AnisotropicVqMetric`]; **smaller is closer** (for
    /// [`InnerProduct`](AnisotropicVqMetric::InnerProduct) this is the negated
    /// estimated inner product).
    pub estimated_distance: f32,
}

impl AnisotropicVqHit {
    /// Create a new hit.
    #[must_use]
    pub fn new(id: crate::types::DocumentId, estimated_distance: f32) -> Self {
        Self {
            id,
            estimated_distance,
        }
    }

    /// The estimated *similarity* (larger is closer), i.e. the sign-flipped
    /// [`estimated_distance`](Self::estimated_distance).
    ///
    /// For [`InnerProduct`](AnisotropicVqMetric::InnerProduct) this recovers the
    /// estimated inner product; for
    /// [`L2`](AnisotropicVqMetric::L2) it is the negated squared distance.
    #[must_use]
    pub fn similarity(&self) -> f32 {
        -self.estimated_distance
    }
}

// ── AnisotropicVqError ──────────────────────────────────────────────────────────

/// Errors from the `anisotropic_vq` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AnisotropicVqError {
    /// A configuration field was out of range; the message names the offender.
    #[error("invalid anisotropic-vq configuration: {0}")]
    InvalidConfig(String),
    /// A vector did not match the configured [`dim`](AnisotropicVqConfig::dim).
    #[error("vector dim mismatch: expected {expected}, got {actual}")]
    DimMismatch {
        /// The configured dimensionality.
        expected: usize,
        /// The dimensionality actually supplied.
        actual: usize,
    },
    /// An operation requiring a trained quantizer was attempted before
    /// training.
    #[error("quantizer not trained")]
    NotTrained,
    /// Training (or index construction) was requested with an empty vector set.
    #[error("training set is empty")]
    EmptyTrainingSet,
}
