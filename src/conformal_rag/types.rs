//! Types for the `conformal_rag` module.
//!
//! These types describe **split conformal prediction** for RAG: a
//! distribution-free wrapper that turns any underlying nonconformity score into
//! a *prediction set* carrying a finite-sample marginal coverage guarantee
//! `P(true_label in prediction_set) >= 1 - alpha`.
//!
//! They are **distinct** from the heuristic answer-or-refuse signals of the
//! `abstention` module: abstention thresholds a blended confidence margin with
//! *no* statistical guarantee, whereas here the threshold is a calibration-set
//! quantile with a *proven* coverage guarantee under exchangeability.

use thiserror::Error;

// ── ConformalError ──────────────────────────────────────────────────────────

/// Errors produced by the `conformal_rag` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConformalError {
    /// The target miscoverage rate `alpha` was not in the open interval
    /// `(0, 1)`.
    #[error("alpha must be in the open interval (0, 1), got {0}")]
    InvalidAlpha(String),
    /// The calibration set was empty; at least one calibration score is
    /// required to calibrate a threshold.
    #[error("calibration set is empty: at least one nonconformity score is required")]
    EmptyCalibration,
    /// A calibration nonconformity score was `NaN` or infinite. Calibration
    /// requires finite scores so the order statistics are well defined.
    #[error("calibration score at index {index} is non-finite (NaN or infinite)")]
    NonFiniteScore {
        /// The position of the offending score within the calibration slice.
        index: usize,
    },
    /// A group key mapped to an empty calibration set in a Mondrian
    /// (group-conditional) calibration.
    #[error("group '{group}' has no calibration scores")]
    EmptyGroup {
        /// The group key that had no scores.
        group: String,
    },
}

// ── ConformalConfig ─────────────────────────────────────────────────────────

/// Configuration for split conformal calibration.
///
/// The single governing parameter is [`alpha`](Self::alpha), the target
/// *miscoverage* rate. A calibrated [`ConformalCalibrator`] guarantees marginal
/// coverage `>= 1 - alpha`: the true label lands inside the prediction set with
/// probability at least `1 - alpha`, provided the calibration and test data are
/// exchangeable.
///
/// [`ConformalCalibrator`]: super::calibrator::ConformalCalibrator
#[derive(Debug, Clone, PartialEq)]
pub struct ConformalConfig {
    /// Target miscoverage rate in the open interval `(0, 1)`. The coverage
    /// guarantee is `1 - alpha`. Default `0.1` (i.e. 90% coverage).
    pub alpha: f64,
    /// Minimum number of calibration scores a Mondrian group must contribute
    /// before it earns its own per-group threshold. Groups below this size fall
    /// back to the pooled global threshold, which trades some group-conditional
    /// tightness for a more stable estimate. Default `1` (every non-empty group
    /// gets its own threshold). Ignored by the global [`ConformalCalibrator`].
    ///
    /// [`ConformalCalibrator`]: super::calibrator::ConformalCalibrator
    pub min_group_size: usize,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            min_group_size: 1,
        }
    }
}

impl ConformalConfig {
    /// Creates a new [`ConformalConfig`] with default values (`alpha = 0.1`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the target miscoverage rate `alpha`.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the minimum group size for Mondrian (group-conditional)
    /// calibration.
    #[must_use]
    pub fn with_min_group_size(mut self, min_group_size: usize) -> Self {
        self.min_group_size = min_group_size;
        self
    }

    /// Validates that [`alpha`](Self::alpha) lies in the open interval
    /// `(0, 1)`.
    ///
    /// # Errors
    ///
    /// Returns [`ConformalError::InvalidAlpha`] if `alpha` is `NaN`, not
    /// finite, or outside `(0, 1)`.
    pub fn validate(&self) -> Result<(), ConformalError> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(ConformalError::InvalidAlpha(self.alpha.to_string()));
        }
        Ok(())
    }
}

// ── NonconformityScorer ─────────────────────────────────────────────────────

/// A nonconformity score `s(query, candidate) -> f64`.
///
/// The score measures how *unusual* or *wrong* a candidate answer/passage is
/// for a query — **higher means more nonconforming** (less plausible). Conformal
/// prediction is agnostic to how this number is produced: it can wrap an
/// LLM-judge score, a retrieval-relevance score, or any bespoke closure. The
/// only requirement for the coverage guarantee is that calibration and test
/// scores be exchangeable draws from the same distribution.
///
/// A common RAG instantiation is `s = 1 - relevance_or_confidence`, so that a
/// perfectly relevant candidate has `s = 0` and an irrelevant one has `s = 1`.
///
/// Any closure `Fn(&str, &str) -> f64` implements this trait, so callers can
/// plug in their own scorer inline without defining a type; the default
/// [`LexicalOverlapScorer`] is provided as a working, dependency-free example.
pub trait NonconformityScorer {
    /// Returns the nonconformity of `candidate` as an answer/passage for
    /// `query`. Higher is more nonconforming.
    fn nonconformity(&self, query: &str, candidate: &str) -> f64;
}

impl<F> NonconformityScorer for F
where
    F: Fn(&str, &str) -> f64,
{
    fn nonconformity(&self, query: &str, candidate: &str) -> f64 {
        self(query, candidate)
    }
}

// ── LexicalOverlapScorer ────────────────────────────────────────────────────

/// A default, dependency-free [`NonconformityScorer`] based on lexical token
/// overlap.
///
/// It tokenizes `query` and `candidate` into lower-cased alphanumeric tokens,
/// computes their Jaccard similarity `|intersection| / |union|` in `[0, 1]`, and
/// returns the nonconformity `s = 1 - jaccard`. A candidate that shares all of
/// the query's vocabulary scores `s = 0` (maximally conforming); a candidate
/// with no shared tokens scores `s = 1` (maximally nonconforming). When both
/// texts are empty the union is empty and the score is defined as `1.0` (nothing
/// to be relevant to ⇒ maximally nonconforming).
///
/// This mirrors the spirit of the crate's other `Mock*`/heuristic scorers: it is
/// deterministic, needs no model, and exists so the pipeline can be exercised
/// end-to-end. Production callers supply a real scorer (an LLM judge, a
/// cross-encoder, etc.) via the [`NonconformityScorer`] trait.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LexicalOverlapScorer;

impl LexicalOverlapScorer {
    /// Creates a new [`LexicalOverlapScorer`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Tokenizes `text` into a sorted, de-duplicated set of lower-cased
    /// alphanumeric tokens.
    fn tokens(text: &str) -> Vec<String> {
        let mut toks: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect();
        toks.sort_unstable();
        toks.dedup();
        toks
    }
}

impl NonconformityScorer for LexicalOverlapScorer {
    fn nonconformity(&self, query: &str, candidate: &str) -> f64 {
        let q = Self::tokens(query);
        let c = Self::tokens(candidate);
        if q.is_empty() && c.is_empty() {
            return 1.0;
        }
        // Both token vecs are sorted+deduped; walk them to count intersection.
        let mut i = 0usize;
        let mut j = 0usize;
        let mut inter = 0usize;
        while i < q.len() && j < c.len() {
            match q[i].cmp(&c[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    inter += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        let union = q.len() + c.len() - inter;
        if union == 0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let jaccard = inter as f64 / union as f64;
        (1.0 - jaccard).clamp(0.0, 1.0)
    }
}

// ── PredictionKind ──────────────────────────────────────────────────────────

/// The qualitative shape of a conformal [`PredictionSet`].
///
/// | Kind | Members | RAG interpretation |
/// |------|---------|--------------------|
/// | [`Abstain`](PredictionKind::Abstain) | 0 | Nothing passes calibrated scrutiny ⇒ refuse to answer |
/// | [`Singleton`](PredictionKind::Singleton) | 1 | Exactly one candidate is admissible ⇒ answer with it |
/// | [`Ambiguous`](PredictionKind::Ambiguous) | >= 2 | A calibrated ambiguity set ⇒ pick top-1 or surface the choices |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictionKind {
    /// The prediction set is empty: no candidate has nonconformity at or below
    /// the calibrated threshold. The calibrated advice is to abstain.
    Abstain,
    /// The prediction set contains exactly one candidate.
    Singleton,
    /// The prediction set contains two or more candidates.
    Ambiguous,
}

impl PredictionKind {
    /// Derives the [`PredictionKind`] from the number of admitted members.
    #[must_use]
    pub fn from_len(len: usize) -> Self {
        match len {
            0 => Self::Abstain,
            1 => Self::Singleton,
            _ => Self::Ambiguous,
        }
    }

    /// Returns a stable, lower-case string identifier for the kind.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Abstain => "abstain",
            Self::Singleton => "singleton",
            Self::Ambiguous => "ambiguous",
        }
    }
}

impl std::fmt::Display for PredictionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── ConformalMember ─────────────────────────────────────────────────────────

/// A single candidate admitted into a conformal [`PredictionSet`].
///
/// Members carry their original position in the caller's candidate list, their
/// computed nonconformity score, and (when available) the candidate text.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformalMember {
    /// The index of this candidate within the caller-supplied candidate slice.
    pub index: usize,
    /// The nonconformity score `s(query, candidate)` for this candidate. It is
    /// guaranteed to be at or below the calibrated threshold.
    pub nonconformity: f64,
    /// The candidate text, when the prediction set was produced from candidate
    /// strings; `None` when produced directly from precomputed scores.
    pub label: Option<String>,
}

// ── PredictionSet ───────────────────────────────────────────────────────────

/// The output of [`ConformalCalibrator::predict_set`]: the set of candidates
/// whose nonconformity falls at or below the calibrated threshold, plus the
/// derived [`PredictionKind`].
///
/// Members are sorted by ascending nonconformity, so [`best`](Self::best) is the
/// most conforming (most plausible) admitted candidate — the natural top-1 pick
/// when the caller wants a single answer out of an ambiguity set.
///
/// [`ConformalCalibrator::predict_set`]: super::calibrator::ConformalCalibrator::predict_set
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionSet {
    /// The admitted candidates, sorted by ascending nonconformity (most
    /// conforming first).
    pub members: Vec<ConformalMember>,
    /// The qualitative shape of the set (abstain / singleton / ambiguous).
    pub kind: PredictionKind,
    /// The calibrated threshold that produced this set. `f64::INFINITY` means
    /// the calibrated threshold admits everything (see
    /// [`ConformalCalibrator`](super::calibrator::ConformalCalibrator) for when
    /// that happens).
    pub threshold: f64,
    /// The total number of candidates that were considered (admitted plus
    /// rejected).
    pub total_candidates: usize,
}

impl PredictionSet {
    /// Returns `true` if the set is empty ⇒ the calibrated advice is to
    /// abstain.
    #[must_use]
    pub fn is_abstain(&self) -> bool {
        matches!(self.kind, PredictionKind::Abstain)
    }

    /// Returns `true` if the set contains exactly one candidate.
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        matches!(self.kind, PredictionKind::Singleton)
    }

    /// Returns `true` if the set contains two or more candidates.
    #[must_use]
    pub fn is_ambiguous(&self) -> bool {
        matches!(self.kind, PredictionKind::Ambiguous)
    }

    /// The number of admitted candidates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` if no candidate was admitted (equivalent to
    /// [`is_abstain`](Self::is_abstain)).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The most conforming (lowest-nonconformity) admitted candidate, i.e. the
    /// natural top-1 pick, or `None` when the set is empty (abstain).
    #[must_use]
    pub fn best(&self) -> Option<&ConformalMember> {
        self.members.first()
    }
}
