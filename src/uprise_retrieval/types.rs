//! Types for the `uprise_retrieval` module.

use thiserror::Error;

// ── PromptExemplar ────────────────────────────────────────────────────────────

/// A single reusable prompt/exemplar unit stored in a cross-task
/// [`UpriseIndex`](super::UpriseIndex).
///
/// Pairs a caller-supplied [`task_label`](Self::task_label) — the task the
/// exemplar was *authored* for, e.g. `"sentiment_classification"`,
/// `"qa"`, `"summarization"` — with the reusable
/// [`prompt_text`](Self::prompt_text) (the instruction/demonstration itself)
/// and a running [`outcome_quality`](Self::outcome_quality) signal describing
/// how well this prompt has performed historically when reused.
///
/// `task_label` is bookkeeping metadata only: [`UpriseRetriever`
/// ](super::UpriseRetriever) never matches on it when scoring candidates for
/// a new query — that is precisely what makes retrieval "universal" /
/// cross-task in the UPRISE sense. It exists so callers can inspect
/// provenance, or explicitly exclude same-task exemplars via
/// [`UpriseRetriever::retrieve_excluding_task`
/// ](super::UpriseRetriever::retrieve_excluding_task) to measure pure
/// cross-task transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptExemplar {
    /// Caller-supplied label identifying which task this exemplar was
    /// authored for (e.g. `"qa"`, `"summarization"`). Purely descriptive
    /// metadata — retrieval never filters or matches on it unless the caller
    /// explicitly asks to (see [`UpriseRetriever::retrieve_excluding_task`
    /// ](super::UpriseRetriever::retrieve_excluding_task)).
    pub task_label: String,
    /// The reusable instruction/exemplar text (e.g. a task instruction, or an
    /// instruction plus a worked few-shot example).
    pub prompt_text: String,
    /// Historical performance signal in `[0.0, 1.0]`. Higher means the prompt
    /// has performed well when previously reused; updated over time via
    /// [`UpriseIndex::update_outcome`](super::UpriseIndex::update_outcome).
    pub outcome_quality: f64,
    /// Optional precomputed FNV-1a pseudo-embedding of
    /// [`prompt_text`](Self::prompt_text).
    ///
    /// Left empty by default. [`UpriseRetriever`](super::UpriseRetriever)
    /// computes one on demand at its configured
    /// [`UpriseConfig::dim`](UpriseConfig::dim) whenever this field's length
    /// does not already match that dimension — mirroring the
    /// `Demonstration` pattern used by `prompt_optimization`, so callers with
    /// an expensive/precise embedder can pre-populate this field to skip the
    /// pseudo-embedding fallback.
    pub embedding: Vec<f32>,
}

impl PromptExemplar {
    /// Create a new exemplar with no precomputed embedding.
    ///
    /// # Errors
    ///
    /// Returns [`UpriseError::InvalidOutcomeQuality`] when `outcome_quality`
    /// is not finite or lies outside `[0.0, 1.0]`.
    pub fn new(
        task_label: impl Into<String>,
        prompt_text: impl Into<String>,
        outcome_quality: f64,
    ) -> Result<Self, UpriseError> {
        if !is_valid_unit_interval(outcome_quality) {
            return Err(UpriseError::InvalidOutcomeQuality(outcome_quality));
        }
        Ok(Self {
            task_label: task_label.into(),
            prompt_text: prompt_text.into(),
            outcome_quality,
            embedding: Vec::new(),
        })
    }

    /// Attach a precomputed embedding (builder).
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }
}

/// Return `true` when `value` is finite and lies in `[0.0, 1.0]`.
pub(crate) fn is_valid_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

// ── UpriseHit ─────────────────────────────────────────────────────────────────

/// A single scored retrieval result returned by [`UpriseRetriever`
/// ](super::UpriseRetriever).
///
/// Carries a snapshot of the source [`PromptExemplar`]'s fields alongside the
/// two scores that produced its rank: the raw embedding
/// [`similarity`](Self::similarity) to the query, and the outcome-weighted
/// [`combined_score`](Self::combined_score) that [`UpriseRetriever::retrieve`
/// ](super::UpriseRetriever::retrieve) actually ranks by.
#[derive(Debug, Clone, PartialEq)]
pub struct UpriseHit {
    /// The source exemplar's task label (metadata only; see
    /// [`PromptExemplar::task_label`]).
    pub task_label: String,
    /// The source exemplar's reusable prompt text.
    pub prompt_text: String,
    /// The source exemplar's historical outcome-quality signal, as stored at
    /// scoring time.
    pub outcome_quality: f64,
    /// Cosine similarity in `[-1.0, 1.0]` (in practice `[0.0, 1.0]` for the
    /// non-negative FNV-1a bucket histograms used here) between the query
    /// embedding and the exemplar embedding.
    pub similarity: f64,
    /// The outcome-weighted score this hit was ranked by:
    /// `similarity_weight * similarity + outcome_weight * outcome_quality`
    /// (see [`UpriseConfig::combined_score`]).
    pub combined_score: f64,
}

// ── UpriseConfig ──────────────────────────────────────────────────────────────

/// Configuration for [`UpriseRetriever`](super::UpriseRetriever) and
/// [`UpriseIndex::update_outcome`](super::UpriseIndex::update_outcome).
#[derive(Debug, Clone, PartialEq)]
pub struct UpriseConfig {
    /// Maximum number of exemplars to return from a retrieval call. Defaults
    /// to `5`.
    pub top_k: usize,
    /// Weight applied to embedding similarity in the outcome-weighted
    /// reranking formula. Defaults to `0.6`.
    pub similarity_weight: f64,
    /// Weight applied to historical outcome quality in the outcome-weighted
    /// reranking formula. Defaults to `0.4`.
    ///
    /// Raising this relative to [`similarity_weight`](Self::similarity_weight)
    /// makes retrieval favour exemplars with a strong track record even when
    /// they are only moderately similar to the query — the core UPRISE
    /// "universal" behaviour this module implements.
    pub outcome_weight: f64,
    /// Smoothing rate `alpha` used by the exponential moving average (EMA)
    /// outcome update in `(0.0, 1.0]`. Defaults to `0.3`.
    ///
    /// `alpha` near `1.0` makes the stored outcome quality track the newest
    /// feedback almost exclusively; `alpha` near `0.0` makes it barely move.
    pub ema_alpha: f64,
    /// Dimensionality of the FNV-1a pseudo-embeddings used to compare the
    /// query against exemplar prompt text. Defaults to `128`.
    pub dim: usize,
}

impl Default for UpriseConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            similarity_weight: 0.6,
            outcome_weight: 0.4,
            ema_alpha: 0.3,
            dim: 128,
        }
    }
}

impl UpriseConfig {
    /// Set [`top_k`](Self::top_k) (builder).
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set [`similarity_weight`](Self::similarity_weight) (builder).
    #[must_use]
    pub fn with_similarity_weight(mut self, weight: f64) -> Self {
        self.similarity_weight = weight;
        self
    }

    /// Set [`outcome_weight`](Self::outcome_weight) (builder).
    #[must_use]
    pub fn with_outcome_weight(mut self, weight: f64) -> Self {
        self.outcome_weight = weight;
        self
    }

    /// Set [`ema_alpha`](Self::ema_alpha) (builder).
    #[must_use]
    pub fn with_ema_alpha(mut self, ema_alpha: f64) -> Self {
        self.ema_alpha = ema_alpha;
        self
    }

    /// Set [`dim`](Self::dim) (builder).
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Combine a similarity score and an outcome-quality score into the
    /// single scalar [`UpriseRetriever`](super::UpriseRetriever) ranks by:
    ///
    /// ```text
    /// combined_score = similarity_weight * similarity + outcome_weight * outcome_quality
    /// ```
    #[must_use]
    pub fn combined_score(&self, similarity: f64, outcome_quality: f64) -> f64 {
        self.similarity_weight
            .mul_add(similarity, self.outcome_weight * outcome_quality)
    }
}

// ── UpriseError ───────────────────────────────────────────────────────────────

/// Errors produced by the `uprise_retrieval` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum UpriseError {
    /// The supplied query was empty (after trimming).
    #[error("query must not be empty")]
    EmptyQuery,
    /// The [`UpriseIndex`](super::UpriseIndex) held no exemplars at all.
    #[error("prompt exemplar index is empty")]
    EmptyIndex,
    /// The index held exemplars, but none remained eligible after applying a
    /// filter/exclusion predicate.
    #[error("no eligible exemplars remain after filtering")]
    NoEligibleExemplars,
    /// An `outcome_quality` or outcome-update `new_signal` value was not
    /// finite or fell outside `[0.0, 1.0]`.
    #[error("outcome quality must be finite and within [0.0, 1.0], got {0}")]
    InvalidOutcomeQuality(f64),
    /// An EMA smoothing rate was not finite or fell outside `(0.0, 1.0]`.
    #[error("ema_alpha must be finite and within (0.0, 1.0], got {0}")]
    InvalidEmaAlpha(f64),
    /// An exemplar index passed to
    /// [`UpriseIndex::update_outcome`](super::UpriseIndex::update_outcome)
    /// was out of bounds (or, from
    /// [`UpriseIndex::update_outcome_for`](super::UpriseIndex::update_outcome_for),
    /// no exemplar matched the given `task_label`/`prompt_text` pair).
    #[error("exemplar index {0} out of bounds")]
    IndexOutOfBounds(usize),
}
