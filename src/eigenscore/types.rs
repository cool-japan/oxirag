//! Types for the `eigenscore` module.

use thiserror::Error;

// ── EigenScoreConfig ──────────────────────────────────────────────────────────

/// Configuration for [`EigenScoreDetector`](super::detector::EigenScoreDetector).
///
/// See the [module docs](super) for the full formula. In short: given `K`
/// sampled-response embeddings, the detector centers them, forms the `K x K`
/// Gram matrix, regularizes it into a covariance-like matrix `Sigma`, and
/// reports the mean log-eigenvalue of `Sigma` as the `EigenScore`.
#[derive(Debug, Clone, PartialEq)]
pub struct EigenScoreConfig {
    /// Regularization `alpha` added to the diagonal of the `(1/K)`-scaled Gram
    /// matrix before eigen-decomposition: `Sigma = (1/K) * G + alpha * I_K`.
    ///
    /// Keeps `Sigma` strictly positive-definite (so `ln(lambda_i)` stays
    /// finite) even when the `K` embeddings are exactly collinear or
    /// identical, in which case `G` is singular.
    ///
    /// Default: `1e-3`.
    pub regularization: f32,
    /// Optional feature-clipping quantile `p` in `[0.0, 0.5)`.
    ///
    /// When set, every embedding coordinate is clipped to the empirical
    /// `[p, 1-p]` quantile band computed across the `K` samples *before*
    /// centering. This mirrors INSIDE's test-time feature-clipping trick,
    /// which suppresses a handful of outlier dimensions from dominating the
    /// covariance spectrum.
    ///
    /// Default: `None` (no clipping).
    pub clip_percentile: Option<f32>,
    /// Maximum number of full cyclic-Jacobi sweeps run by the eigensolver.
    ///
    /// Default: `100`.
    pub jacobi_max_sweeps: usize,
    /// Convergence tolerance on the off-diagonal Frobenius norm of the
    /// working matrix; sweeping stops early once the norm drops below this
    /// value.
    ///
    /// Default: `1e-10`.
    pub jacobi_tol: f64,
    /// Score threshold: [`EigenScoreResult::is_hallucination`] is `true`
    /// whenever the computed score is `>=` this value.
    ///
    /// Higher `EigenScore` means the `K` sampled responses embed into a
    /// higher-entropy (more spread out / less self-consistent) distribution,
    /// which INSIDE associates with a higher hallucination risk.
    ///
    /// Default: `-3.0`.
    pub hallucination_threshold: f32,
    /// Pseudo-embedding dimension used by
    /// [`EigenScoreDetector::score_responses`](super::detector::EigenScoreDetector::score_responses)
    /// when deriving embeddings directly from response strings.
    ///
    /// Default: `64`.
    pub embed_dim: usize,
}

impl Default for EigenScoreConfig {
    fn default() -> Self {
        Self {
            regularization: 1e-3,
            clip_percentile: None,
            jacobi_max_sweeps: 100,
            jacobi_tol: 1e-10,
            hallucination_threshold: -3.0,
            embed_dim: 64,
        }
    }
}

impl EigenScoreConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the regularization `alpha`.
    #[must_use]
    pub fn with_regularization(mut self, regularization: f32) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the optional feature-clipping quantile.
    #[must_use]
    pub fn with_clip_percentile(mut self, clip_percentile: Option<f32>) -> Self {
        self.clip_percentile = clip_percentile;
        self
    }

    /// Set the maximum number of cyclic-Jacobi sweeps.
    #[must_use]
    pub fn with_jacobi_max_sweeps(mut self, jacobi_max_sweeps: usize) -> Self {
        self.jacobi_max_sweeps = jacobi_max_sweeps;
        self
    }

    /// Set the Jacobi off-diagonal convergence tolerance.
    #[must_use]
    pub fn with_jacobi_tol(mut self, jacobi_tol: f64) -> Self {
        self.jacobi_tol = jacobi_tol;
        self
    }

    /// Set the hallucination-decision threshold.
    #[must_use]
    pub fn with_hallucination_threshold(mut self, hallucination_threshold: f32) -> Self {
        self.hallucination_threshold = hallucination_threshold;
        self
    }

    /// Set the pseudo-embedding dimension used by the text helper.
    #[must_use]
    pub fn with_embed_dim(mut self, embed_dim: usize) -> Self {
        self.embed_dim = embed_dim;
        self
    }
}

// ── EigenScoreResult ──────────────────────────────────────────────────────────

/// Outcome of scoring a set of sampled-response embeddings.
///
/// Produced by
/// [`EigenScoreDetector::score_embeddings`](super::detector::EigenScoreDetector::score_embeddings)
/// and
/// [`EigenScoreDetector::score_responses`](super::detector::EigenScoreDetector::score_responses).
#[derive(Debug, Clone, PartialEq)]
pub struct EigenScoreResult {
    /// The `EigenScore`: `(1/K) * sum_i ln(lambda_i)`, the mean log-eigenvalue of
    /// the regularized covariance `Sigma`. This is (up to an additive
    /// dimension/constant term) the differential entropy of a Gaussian with
    /// covariance `Sigma` — low values indicate a tight, self-consistent
    /// response distribution; high values indicate a spread-out, likely
    /// hallucinated one.
    pub score: f32,
    /// The eigenvalues of `Sigma`, ascending, as computed by the cyclic-Jacobi
    /// eigensolver. Always has length `K` (the number of sampled responses).
    pub eigenvalues: Vec<f32>,
    /// Whether [`score`](Self::score) meets or exceeds the configured
    /// [`EigenScoreConfig::hallucination_threshold`].
    pub is_hallucination: bool,
}

// ── EigenScoreError ───────────────────────────────────────────────────────────

/// Errors produced by [`EigenScoreDetector`](super::detector::EigenScoreDetector)
/// and by [`symmetric_eigenvalues`](super::jacobi::symmetric_eigenvalues).
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EigenScoreError {
    /// Fewer than the required number of samples were supplied.
    ///
    /// `EigenScore` needs at least two sampled responses to form a non-trivial
    /// Gram matrix.
    #[error("insufficient samples: got {got}, need at least {need}")]
    InsufficientSamples {
        /// The number of samples that were actually supplied.
        got: usize,
        /// The minimum number of samples required.
        need: usize,
    },
    /// A row/embedding had an unexpected length.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The expected dimension (inferred from the first row/embedding).
        expected: usize,
        /// The actual dimension encountered.
        got: usize,
    },
    /// The configuration (or an intermediate computation it produced) is
    /// invalid, with a human-readable explanation.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// A `NaN` or infinite value was encountered where a finite value was
    /// required.
    #[error("encountered a non-finite (NaN or infinite) value")]
    NonFinite,
}
