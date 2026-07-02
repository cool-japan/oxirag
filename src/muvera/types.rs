//! Core types for the `muvera` module: configuration, the multi-vector
//! document record, the search-hit record, the error type, and the crate-local
//! `Result` alias.
//!
//! # MUVERA in one paragraph
//!
//! MUVERA (Dhulipala et al., 2024, *MUVERA: Multi-Vector Retrieval via Fixed
//! Dimensional Encodings*) turns a *set* of token vectors — a multi-vector
//! document or query, as produced by a late-interaction model such as
//! `ColBERT` — into a **single** fixed-dimensional vector, the *Fixed
//! Dimensional Encoding* (FDE), such that the ordinary dot product of two FDEs
//! approximates the Chamfer / `MaxSim` similarity
//! `Σ_{q∈Q} max_{d∈D} ⟨q, d⟩` between the underlying sets. Multi-vector
//! retrieval is thereby reduced to a single-vector maximum-inner-product search
//! (MIPS) with no per-query `MaxSim` scan.
//!
//! See [`crate::muvera::fde`] for the encoding construction and
//! [`crate::muvera::index`] for the store + search.

use thiserror::Error;

// ── DEFAULT_MUVERA_SEED ──────────────────────────────────────────────────────

/// The default, fixed master seed used to deterministically derive every
/// `SimHash` hyperplane and inner-projection matrix when a caller does not
/// override [`MuveraConfig::with_seed`].
///
/// Fixed (not time- or entropy-derived) so that two encoders built from the
/// same configuration reproduce byte-identical hyperplanes, projections, and
/// therefore byte-identical FDEs.
pub const DEFAULT_MUVERA_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Hard upper bound on [`MuveraConfig::k_sim`].
///
/// The number of `SimHash` partition cells is `B = 2^k_sim`; the FDE dimension
/// grows as `r_reps · B · inner_dim`. This cap keeps `B` (and the FDE
/// dimension) from exploding — `2^20 ≈ 1_048_576` cells is already far beyond
/// any realistic configuration (typical `k_sim` is `3..=8`) and guards against
/// shift overflow and pathological allocations.
pub const MAX_K_SIM: u32 = 20;

// ── MuveraResult ─────────────────────────────────────────────────────────────

/// Convenience alias for a [`Result`] whose error is [`MuveraError`].
pub type MuveraResult<T> = Result<T, MuveraError>;

// ── MuveraError ──────────────────────────────────────────────────────────────

/// Errors produced by the `muvera` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MuveraError {
    /// The supplied configuration is invalid (for example `dim == 0`,
    /// `k_sim == 0`, `k_sim > MAX_K_SIM`, `r_reps == 0`, or an FDE dimension
    /// that overflows `usize`).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// A token vector's length did not match the configured token
    /// dimensionality [`MuveraConfig::dim`].
    #[error("dimension mismatch: expected token vectors of length {expected}, got {got}")]
    DimensionMismatch {
        /// The expected token-vector length (from [`MuveraConfig::dim`]).
        expected: usize,
        /// The actual length of the offending token vector.
        got: usize,
    },
    /// A multi-vector set (document or query) contained zero token vectors.
    ///
    /// A set with no tokens has no meaningful FDE — the average / sum
    /// aggregation is undefined — so encoding is refused rather than silently
    /// returning a zero vector.
    #[error("multi-vector set is empty — at least one token vector is required")]
    EmptyMultiVector,
    /// A search or exact re-rank was attempted on an index that holds no
    /// documents.
    #[error("index is empty — add at least one document before searching")]
    EmptyIndex,
    /// [`MuveraIndex::add`](crate::muvera::index::MuveraIndex::add) was called
    /// with an id that is already present in the index.
    #[error("duplicate document id: {0}")]
    DuplicateId(String),
}

// ── MuveraConfig ─────────────────────────────────────────────────────────────

/// Configuration for a [`MuveraEncoder`](crate::muvera::fde::MuveraEncoder) and
/// [`MuveraIndex`](crate::muvera::index::MuveraIndex).
///
/// # Fields & defaults
///
/// | Field | Meaning | Default |
/// |-------|---------|---------|
/// | [`dim`](Self::dim) | Token-vector dimensionality `d` | `128` |
/// | [`k_sim`](Self::k_sim) | `SimHash` hyperplane count; `B = 2^k_sim` cells | `4` |
/// | [`d_proj`](Self::d_proj) | Per-cell inner projection dim (`0` = none) | `16` |
/// | [`r_reps`](Self::r_reps) | Independent repetitions, concatenated | `8` |
/// | [`fill_empty`](Self::fill_empty) | Fill empty document cells (Hamming-nearest) | `true` |
/// | [`seed`](Self::seed) | Master seed for hyperplanes & projections | [`DEFAULT_MUVERA_SEED`] |
/// | [`rerank`](Self::rerank) | Exact-Chamfer re-rank at search time | `false` |
/// | [`rerank_depth`](Self::rerank_depth) | Candidates to exact-re-rank (`0` = whole corpus) | `0` |
/// | [`top_k`](Self::top_k) | Default result count for convenience search | `10` |
///
/// The final FDE dimension is `r_reps · 2^k_sim · inner_dim`, where
/// `inner_dim = d_proj` when `d_proj > 0` and `inner_dim = dim` otherwise
/// (see [`Self::expected_dim`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MuveraConfig {
    /// Dimensionality `d` of every input token vector. Must be greater than
    /// zero.
    pub dim: usize,
    /// Number of `SimHash` hyperplanes drawn per repetition. Each token's signs
    /// against these `k_sim` hyperplanes form a `k_sim`-bit cell id, so the
    /// latent space is partitioned into `B = 2^k_sim` cells. Must satisfy
    /// `1 <= k_sim <= MAX_K_SIM`.
    pub k_sim: u32,
    /// Target dimensionality of the per-cell inner (Johnson–Lindenstrauss)
    /// projection. `0` disables projection, leaving each cell block at the
    /// full token dimensionality `dim`.
    pub d_proj: usize,
    /// Number of independent `SimHash`+projection repetitions whose FDEs are
    /// concatenated to reduce the estimator's variance. Must be greater than
    /// zero.
    pub r_reps: usize,
    /// When `true`, every empty document cell borrows the centroid of the
    /// Hamming-nearest non-empty cell so that all `B` blocks are populated;
    /// this asymmetry (document side fills, query side does not) is what makes
    /// the FDE dot product approximate Chamfer. When `false`, empty document
    /// cells stay zero.
    pub fill_empty: bool,
    /// Master seed from which every hyperplane and projection matrix is
    /// deterministically derived. Two configurations that differ only in
    /// `seed` produce different — but each individually reproducible —
    /// encodings.
    pub seed: u64,
    /// When `true`, [`MuveraIndex::search`](crate::muvera::index::MuveraIndex::search)
    /// re-ranks its top FDE candidates by the exact Chamfer similarity before
    /// truncating to `k`.
    pub rerank: bool,
    /// How many top-scoring (by approximate FDE dot product) candidates the
    /// exact re-rank reconsiders. `0` means "re-rank the whole corpus"; any
    /// positive value is clamped up to the requested `k` so the final top-`k`
    /// is never starved. Ignored when [`rerank`](Self::rerank) is `false`.
    pub rerank_depth: usize,
    /// Default number of results returned by
    /// [`MuveraIndex::search_default`](crate::muvera::index::MuveraIndex::search_default).
    pub top_k: usize,
}

impl Default for MuveraConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            k_sim: 4,
            d_proj: 16,
            r_reps: 8,
            fill_empty: true,
            seed: DEFAULT_MUVERA_SEED,
            rerank: false,
            rerank_depth: 0,
            top_k: 10,
        }
    }
}

impl MuveraConfig {
    /// Create a configuration for `dim`-dimensional token vectors, with every
    /// other field taken from [`MuveraConfig::default`].
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::InvalidConfig`] when the resulting configuration
    /// fails [`validate`](Self::validate) (for example `dim == 0`).
    pub fn new(dim: usize) -> MuveraResult<Self> {
        let config = Self {
            dim,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Override the `SimHash` hyperplane count `k_sim` (`B = 2^k_sim` cells).
    #[must_use]
    pub fn with_k_sim(mut self, k_sim: u32) -> Self {
        self.k_sim = k_sim;
        self
    }

    /// Override the per-cell inner projection dimension (`0` disables it).
    #[must_use]
    pub fn with_d_proj(mut self, d_proj: usize) -> Self {
        self.d_proj = d_proj;
        self
    }

    /// Override the number of repetitions.
    #[must_use]
    pub fn with_r_reps(mut self, r_reps: usize) -> Self {
        self.r_reps = r_reps;
        self
    }

    /// Toggle the empty-document-cell fill behaviour.
    #[must_use]
    pub fn with_fill_empty(mut self, fill_empty: bool) -> Self {
        self.fill_empty = fill_empty;
        self
    }

    /// Override the master seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Toggle exact-Chamfer re-ranking at search time.
    #[must_use]
    pub fn with_rerank(mut self, rerank: bool) -> Self {
        self.rerank = rerank;
        self
    }

    /// Set how many FDE candidates the exact re-rank reconsiders (`0` = whole
    /// corpus).
    #[must_use]
    pub fn with_rerank_depth(mut self, rerank_depth: usize) -> Self {
        self.rerank_depth = rerank_depth;
        self
    }

    /// Set the default result count for the convenience search.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Number of `SimHash` partition cells `B = 2^k_sim`.
    ///
    /// Assumes [`validate`](Self::validate) has established
    /// `k_sim <= MAX_K_SIM`, so the shift never overflows.
    #[must_use]
    pub fn num_buckets(&self) -> usize {
        1usize << self.k_sim
    }

    /// The dimensionality of a single per-cell block after optional
    /// projection: `d_proj` when projection is enabled, otherwise the full
    /// token dimensionality `dim`.
    #[must_use]
    pub fn inner_dim(&self) -> usize {
        if self.d_proj == 0 {
            self.dim
        } else {
            self.d_proj
        }
    }

    /// The exact length every [`FixedDimEncoding`](crate::muvera::fde::FixedDimEncoding)
    /// produced under this configuration will have:
    /// `r_reps · 2^k_sim · inner_dim`.
    ///
    /// Assumes [`validate`](Self::validate) has confirmed the product does not
    /// overflow `usize`.
    #[must_use]
    pub fn expected_dim(&self) -> usize {
        self.r_reps * self.num_buckets() * self.inner_dim()
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::InvalidConfig`] when any of the following hold:
    /// `dim == 0`; `k_sim == 0`; `k_sim > MAX_K_SIM`; `r_reps == 0`; or the
    /// FDE dimension `r_reps · 2^k_sim · inner_dim` overflows `usize`.
    pub fn validate(&self) -> MuveraResult<()> {
        if self.dim == 0 {
            return Err(MuveraError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }
        if self.k_sim == 0 {
            return Err(MuveraError::InvalidConfig(
                "k_sim must be greater than zero".into(),
            ));
        }
        if self.k_sim > MAX_K_SIM {
            return Err(MuveraError::InvalidConfig(format!(
                "k_sim ({}) must not exceed MAX_K_SIM ({MAX_K_SIM})",
                self.k_sim
            )));
        }
        if self.r_reps == 0 {
            return Err(MuveraError::InvalidConfig(
                "r_reps must be greater than zero".into(),
            ));
        }
        // k_sim <= MAX_K_SIM (<= 20) guarantees the shift fits in usize on all
        // supported targets.
        let buckets = 1usize << self.k_sim;
        let overflow = || {
            MuveraError::InvalidConfig(format!(
                "FDE dimension r_reps ({}) * 2^k_sim ({buckets}) * inner_dim ({}) overflows usize",
                self.r_reps,
                self.inner_dim()
            ))
        };
        self.r_reps
            .checked_mul(buckets)
            .and_then(|v| v.checked_mul(self.inner_dim()))
            .ok_or_else(overflow)?;
        Ok(())
    }
}

// ── MuveraDocument ───────────────────────────────────────────────────────────

/// A document expressed as a *set* (multi-vector) of token embeddings, together
/// with a stable identifier.
///
/// This is the user-facing input to
/// [`MuveraIndex::add`](crate::muvera::index::MuveraIndex::add): the index
/// computes and stores each document's Fixed Dimensional Encoding internally
/// while retaining the original `token_vectors` so an optional exact-Chamfer
/// re-rank can recompute the true `MaxSim` at query time.
#[derive(Debug, Clone, PartialEq)]
pub struct MuveraDocument {
    /// Stable identifier, unique within an index.
    pub id: String,
    /// The document's token vectors. Every vector must have length
    /// [`MuveraConfig::dim`].
    pub token_vectors: Vec<Vec<f32>>,
}

impl MuveraDocument {
    /// Construct a document from an id and its token vectors.
    #[must_use]
    pub fn new(id: impl Into<String>, token_vectors: Vec<Vec<f32>>) -> Self {
        Self {
            id: id.into(),
            token_vectors,
        }
    }

    /// Number of token vectors in the document.
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_vectors.len()
    }

    /// Return `true` when the document has no token vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_vectors.is_empty()
    }
}

// ── MuveraSimilarity ─────────────────────────────────────────────────────────

/// A single ranked search result: a document id paired with its similarity to
/// the query.
///
/// The [`score`](Self::score) is the approximate FDE dot product
/// (proportional to the estimated Chamfer similarity) when re-ranking is off,
/// or the exact Chamfer / `MaxSim` similarity when
/// [`MuveraConfig::rerank`] is enabled. Larger is more similar; results are
/// returned in descending score order.
#[derive(Debug, Clone, PartialEq)]
pub struct MuveraSimilarity {
    /// Identifier of the matched document.
    pub doc_id: String,
    /// Similarity score (larger is more similar).
    pub score: f32,
}

impl MuveraSimilarity {
    /// Construct a similarity result.
    #[must_use]
    pub fn new(doc_id: impl Into<String>, score: f32) -> Self {
        Self {
            doc_id: doc_id.into(),
            score,
        }
    }
}
