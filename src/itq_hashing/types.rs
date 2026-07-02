//! Core types for the `itq_hashing` module: configuration, error type, the
//! packed binary code record and the search-hit record.
//!
//! ## Notation (following Gong & Lazebnik, *Iterative Quantization: A
//! Procrustean Approach to Learning Binary Codes*, CVPR 2011)
//!
//! | Symbol | Meaning |
//! |--------|---------|
//! | `X` | Training matrix (`n` rows, `D` columns) |
//! | `mu` | Per-column mean of `X` (the centering vector) |
//! | `V` | `D x k` matrix of the top-`k` PCA eigenvectors |
//! | `Z = X_centered · V` | `n x k` PCA projection of the training data |
//! | `R` | Learned `k x k` orthogonal rotation |
//! | `B = sign(Z · R)` | `n x k` binary code matrix in `{-1, +1}` |
//! | `k` | Number of code bits (`num_bits`) |

use thiserror::Error;

// ── DEFAULT_ITQ_SEED ──────────────────────────────────────────────────────────

/// The default, fixed seed used to deterministically derive the initial random
/// orthogonal rotation `R` in the alternating-minimization loop when a caller
/// does not override [`ItqConfig::with_seed`].
///
/// Fixed (not time- or entropy-derived) so that repeated `train` calls with an
/// identical configuration and identical training data reproduce a
/// byte-identical rotation and, therefore, byte-identical codes.
pub const DEFAULT_ITQ_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

// ── ItqConfig ─────────────────────────────────────────────────────────────────

/// Configuration for an [`ItqHasher`](super::hasher::ItqHasher) /
/// [`ItqIndex`](super::index::ItqIndex).
///
/// # Defaults
///
/// | Field | Default |
/// |-------|---------|
/// | [`max_iterations`](Self::max_iterations) | `50` |
/// | [`tolerance`](Self::tolerance) | `1e-7` |
/// | [`seed`](Self::seed) | [`DEFAULT_ITQ_SEED`] |
/// | [`jacobi_max_sweeps`](Self::jacobi_max_sweeps) | `100` |
/// | [`jacobi_tol`](Self::jacobi_tol) | `1e-10` |
///
/// `num_bits` has no sensible default (it is the code length and must be no
/// larger than the input dimensionality), so it is supplied to the validating
/// constructor [`ItqConfig::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItqConfig {
    /// Number of binary code bits `k`. This is also the PCA target rank: the
    /// hasher keeps the top-`num_bits` principal components. Must be greater
    /// than zero and no larger than the training vectors' dimensionality `D`.
    pub num_bits: usize,
    /// Maximum number of alternating-minimization iterations. Must be at least
    /// one. Default: `50`.
    pub max_iterations: usize,
    /// Convergence tolerance on the *decrease* of the Frobenius quantization
    /// objective between consecutive iterations: the loop stops once
    /// `loss_prev - loss_current < tolerance`. Must be finite and
    /// non-negative. Default: `1e-7`.
    pub tolerance: f64,
    /// Seed for the deterministic pseudo-random initial orthogonal rotation.
    /// Defaults to [`DEFAULT_ITQ_SEED`].
    pub seed: u64,
    /// Maximum number of cyclic-Jacobi sweeps used by the internal symmetric
    /// eigensolver (for both PCA and the per-iteration SVD). Must be at least
    /// one. Default: `100`.
    pub jacobi_max_sweeps: usize,
    /// Off-diagonal convergence tolerance for the internal cyclic-Jacobi
    /// eigensolver. Must be finite and non-negative. Default: `1e-10`.
    pub jacobi_tol: f64,
}

impl ItqConfig {
    /// Create a new configuration for `num_bits` code bits, with the default
    /// iteration budget, tolerances and seed.
    ///
    /// # Errors
    ///
    /// Returns [`ItqError::InvalidConfig`] when `num_bits == 0`.
    pub fn new(num_bits: usize) -> Result<Self, ItqError> {
        if num_bits == 0 {
            return Err(ItqError::InvalidConfig(
                "num_bits must be greater than zero".into(),
            ));
        }
        Ok(Self {
            num_bits,
            max_iterations: 50,
            tolerance: 1e-7,
            seed: DEFAULT_ITQ_SEED,
            jacobi_max_sweeps: 100,
            jacobi_tol: 1e-10,
        })
    }

    /// Override the maximum number of alternating-minimization iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Override the convergence tolerance on the objective decrease.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Override the seed for the initial pseudo-random rotation.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Override the maximum number of cyclic-Jacobi sweeps.
    #[must_use]
    pub fn with_jacobi_max_sweeps(mut self, jacobi_max_sweeps: usize) -> Self {
        self.jacobi_max_sweeps = jacobi_max_sweeps;
        self
    }

    /// Override the cyclic-Jacobi off-diagonal convergence tolerance.
    #[must_use]
    pub fn with_jacobi_tol(mut self, jacobi_tol: f64) -> Self {
        self.jacobi_tol = jacobi_tol;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ItqError::InvalidConfig`] when `num_bits == 0`,
    /// `max_iterations == 0`, `jacobi_max_sweeps == 0`, or when `tolerance` /
    /// `jacobi_tol` is negative or non-finite.
    pub fn validate(&self) -> Result<(), ItqError> {
        if self.num_bits == 0 {
            return Err(ItqError::InvalidConfig(
                "num_bits must be greater than zero".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(ItqError::InvalidConfig(
                "max_iterations must be at least one".into(),
            ));
        }
        if self.jacobi_max_sweeps == 0 {
            return Err(ItqError::InvalidConfig(
                "jacobi_max_sweeps must be at least one".into(),
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(ItqError::InvalidConfig(format!(
                "tolerance must be a non-negative finite value, got {}",
                self.tolerance
            )));
        }
        if !self.jacobi_tol.is_finite() || self.jacobi_tol < 0.0 {
            return Err(ItqError::InvalidConfig(format!(
                "jacobi_tol must be a non-negative finite value, got {}",
                self.jacobi_tol
            )));
        }
        Ok(())
    }
}

// ── ItqError ──────────────────────────────────────────────────────────────────

/// Errors produced by the `itq_hashing` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ItqError {
    /// The supplied configuration is invalid (e.g. `num_bits == 0`).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// A vector's length did not match the configured/trained dimensionality.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The expected dimensionality.
        expected: usize,
        /// The actual length of the offending vector.
        got: usize,
    },
    /// `num_bits` exceeded the training vectors' dimensionality `D`. PCA can
    /// produce at most `D` principal components, so the code length cannot be
    /// larger than `D`.
    #[error(
        "num_bits ({num_bits}) exceeds input dimensionality ({dim}); PCA yields at most D components"
    )]
    NumBitsExceedsDimension {
        /// The requested number of code bits.
        num_bits: usize,
        /// The training vectors' dimensionality.
        dim: usize,
    },
    /// [`ItqHasher::train`](super::hasher::ItqHasher::train) /
    /// [`ItqIndex::build`](super::index::ItqIndex::build) was called with no
    /// training vectors.
    #[error("training dataset is empty — at least one vector is required")]
    EmptyDataset,
    /// A query vector was empty (zero-length).
    #[error("query vector is empty")]
    EmptyQuery,
    /// A `NaN` or infinite value was encountered where a finite value was
    /// required.
    #[error("encountered a non-finite (NaN or infinite) value")]
    NonFinite,
    /// A numerical routine (eigensolver / SVD / basis completion) failed in a
    /// way that should not occur for well-formed finite inputs, with a
    /// human-readable explanation.
    #[error("numerical failure: {0}")]
    Numerical(String),
}

// ── ItqCode ───────────────────────────────────────────────────────────────────

/// A single ITQ binary code: the sign pattern of the rotated PCA projection,
/// packed one bit per dimension.
///
/// Bit `i` is set (`1`) when the `i`-th rotated coordinate is
/// non-negative (`>= 0`), and clear (`0`) when it is negative — that is, a set
/// bit encodes the `+1` symbol and a clear bit the `-1` symbol of the code
/// matrix `B = sign(Z · R)`. The `sign(0) := +1` convention is applied at the
/// threshold. Bits are packed LSB-first into `u64` words: logical bit `i`
/// lives in word `i / 64` at bit position `i % 64`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItqCode {
    /// Packed 1-bit sign code, LSB-first, `ceil(num_bits / 64)` words long.
    pub bits: Vec<u64>,
    /// The logical number of bits `k` represented by this code (may be smaller
    /// than `64 * bits.len()`; any surplus high bits are always zero).
    pub num_bits: usize,
}

impl ItqCode {
    /// Construct a code from its packed words and logical bit count.
    #[must_use]
    pub fn new(bits: Vec<u64>, num_bits: usize) -> Self {
        Self { bits, num_bits }
    }

    /// Read logical bit `i` (`true` == set == the `+1` symbol).
    ///
    /// Returns `false` for indices at or beyond [`num_bits`](Self::num_bits).
    #[must_use]
    pub fn bit(&self, i: usize) -> bool {
        if i >= self.num_bits {
            return false;
        }
        (self.bits[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Expand the code to its `{-1, +1}` symbol vector (length
    /// [`num_bits`](Self::num_bits)): a set bit maps to `+1`, a clear bit to
    /// `-1`.
    #[must_use]
    pub fn as_signs(&self) -> Vec<i8> {
        (0..self.num_bits)
            .map(|i| if self.bit(i) { 1 } else { -1 })
            .collect()
    }

    /// Hamming distance to `other`: the number of differing logical bits.
    ///
    /// Computed as the total population count of the word-wise XOR. When the
    /// two codes share the same [`num_bits`](Self::num_bits) (the invariant
    /// maintained by [`ItqIndex`](super::index::ItqIndex)) this is the exact
    /// Hamming distance; unused high bits are always zero on both sides so they
    /// never contribute.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

// ── ItqHit ────────────────────────────────────────────────────────────────────

/// A single ranked result from [`ItqIndex::search`](super::index::ItqIndex::search).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItqHit {
    /// Identifier of the matched item.
    pub id: String,
    /// Hamming distance between the query code and this item's code (smaller is
    /// closer). Results are ranked ascending by this value.
    pub hamming_distance: u32,
}

impl ItqHit {
    /// Construct a new hit.
    #[must_use]
    pub fn new(id: String, hamming_distance: u32) -> Self {
        Self {
            id,
            hamming_distance,
        }
    }
}
