//! Core data types for the `click_model` module: the click-log data model
//! ([`ClickSession`], [`ClickLog`], [`ClickImpression`]), the shared EM
//! configuration ([`ClickModelConfig`]), the fit report
//! ([`ClickModelFit`]), the [`ClickModel`] trait every click model
//! implements, this module's error type ([`ClickModelError`]), and the
//! position-discount function ([`position_gain`]) the counterfactual
//! estimators score rankings with.
//!
//! # Rank conventions
//!
//! Ranks in this module are **zero-based positions** into
//! [`ClickSession::ranked_doc_ids`]: rank `0` is the top of the result page.
//! The discount [`position_gain`] therefore uses `1 / log2(2 + rank)`, which
//! is exactly the classical DCG discount `1 / log2(1 + one_based_rank)`.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── ClickModelError ──────────────────────────────────────────────────────────

/// Errors produced by the `click_model` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ClickModelError {
    /// A model was asked to fit an empty [`ClickLog`]. There is nothing to
    /// estimate from zero sessions, and every parameter would be its prior —
    /// silently returning the initialisation would be a fabrication, so this
    /// is an error.
    #[error("click log is empty: nothing to fit")]
    EmptyLog,
    /// A [`ClickSession`] carried zero documents. A zero-document impression
    /// list cannot be examined, cannot be clicked, and contributes no
    /// likelihood term, so it is rejected at validation time rather than
    /// silently skipped.
    #[error("session {session_index} contains zero documents")]
    EmptySession {
        /// Index of the offending session within the log.
        session_index: usize,
    },
    /// The `clicks` vector's length did not match `ranked_doc_ids`. Clicks are
    /// aligned to the ranking *by position*, so the two must have identical
    /// length.
    #[error(
        "session {session_index}: clicks length {click_count} does not match \
         ranked_doc_ids length {doc_count}"
    )]
    ClickLengthMismatch {
        /// Index of the offending session within the log.
        session_index: usize,
        /// The number of ranked documents in the session.
        doc_count: usize,
        /// The number of click flags supplied.
        click_count: usize,
    },
    /// The optional `examinations` vector's length did not match
    /// `ranked_doc_ids`.
    #[error(
        "session {session_index}: examinations length {examination_count} does \
         not match ranked_doc_ids length {doc_count}"
    )]
    ExaminationLengthMismatch {
        /// Index of the offending session within the log.
        session_index: usize,
        /// The number of ranked documents in the session.
        doc_count: usize,
        /// The number of examination flags supplied.
        examination_count: usize,
    },
    /// A session declared a click at a position it also declared *unexamined*.
    /// Every click model in this module (and every click model in the
    /// literature) asserts `click ⇒ examined`; a log that violates it is
    /// internally inconsistent, not merely unusual.
    #[error("session {session_index}: rank {rank} is clicked but flagged as not examined")]
    ClickWithoutExamination {
        /// Index of the offending session within the log.
        session_index: usize,
        /// The zero-based rank at which the inconsistency occurs.
        rank: usize,
    },
    /// The same document id appeared at two different ranks within one
    /// session. Ranks within a session must address *distinct* documents,
    /// otherwise a single document has two examination events and two
    /// attractiveness draws in the same impression list, which none of these
    /// models are defined for.
    #[error("session {session_index}: document {doc_id} appears at more than one rank")]
    DuplicateDocumentInSession {
        /// Index of the offending session within the log.
        session_index: usize,
        /// The document id that appeared more than once.
        doc_id: String,
    },
    /// A fitted model was queried about a document it never saw during
    /// fitting, so it holds no attractiveness (or satisfaction) estimate for
    /// it. Returning a prior would be indistinguishable from a real estimate,
    /// so this is an error.
    #[error("document {doc_id} was not present in the log this model was fitted on")]
    UnknownDocument {
        /// The unknown document id.
        doc_id: String,
    },
    /// A fitted model (or a [`crate::click_model::PropensityEstimates`]) was
    /// queried about a rank deeper than anything it ever observed.
    #[error("rank {rank} is beyond the fitted depth {depth}")]
    UnknownRank {
        /// The requested zero-based rank.
        rank: usize,
        /// The number of ranks the model actually observed (valid ranks are
        /// `0..depth`).
        depth: usize,
    },
    /// A model method that requires fitted parameters was called before
    /// [`ClickModel::fit`].
    #[error("{model} has not been fitted yet")]
    NotFitted {
        /// The name of the model that was queried.
        model: &'static str,
    },
    /// A configuration value was outside its admissible range (see
    /// [`ClickModelConfig::validate`]).
    #[error("invalid click-model configuration: {reason}")]
    InvalidConfig {
        /// A human-readable explanation of what was invalid.
        reason: String,
    },
    /// A supplied propensity / examination curve contained a value that cannot
    /// be inverted into an importance weight (non-finite, non-positive, or
    /// greater than one).
    #[error("invalid propensity at rank {rank}: {value} ({reason})")]
    InvalidPropensity {
        /// The zero-based rank whose propensity was rejected.
        rank: usize,
        /// The offending value.
        value: f64,
        /// Why it was rejected.
        reason: &'static str,
    },
    /// A feature vector supplied to [`crate::click_model::DebiasedRanker`] had
    /// a different dimensionality from the others, or was empty.
    #[error(
        "invalid feature vector for document {doc_id}: expected dimension {expected}, got {actual}"
    )]
    FeatureDimensionMismatch {
        /// The document whose feature vector was malformed.
        doc_id: String,
        /// The dimensionality established by the first document.
        expected: usize,
        /// The dimensionality actually supplied.
        actual: usize,
    },
    /// A [`crate::click_model::DebiasedRanker`] was asked to fit a log that
    /// yields no usable (clicked, not-clicked) preference pair at all — every
    /// session was either fully clicked or entirely unclicked, so there is no
    /// preference signal of any kind.
    #[error("click log yields no (clicked, unclicked) preference pairs")]
    NoPreferencePairs,
}

/// Convenience alias for this module's fallible return type.
pub type ClickModelResult<T> = Result<T, ClickModelError>;

// ── position_gain ────────────────────────────────────────────────────────────

/// The DCG position discount of a **zero-based** rank:
/// `gain(rank) = 1 / log2(2 + rank)`.
///
/// This is exactly the classical `1 / log2(1 + r)` discount written for
/// one-based ranks `r = rank + 1`, so `gain(0) = 1 / log2(2) = 1.0`,
/// `gain(1) = 1 / log2(3) ≈ 0.6309`, and so on. It is the utility a
/// counterfactual estimator credits a *relevant* document with when a
/// candidate policy places it at `rank`.
#[must_use]
pub fn position_gain(rank: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let rank = rank as f64;
    1.0 / (rank + 2.0).log2()
}

// ── ClickSession ─────────────────────────────────────────────────────────────

/// One logged impression list: the ranking a user was shown for a query, and
/// which of its positions they clicked.
///
/// `clicks[i]` is aligned **by rank position** to `ranked_doc_ids[i]`, so
/// `clicks[0]` is the click on the top result. `examinations` is the optional
/// ground-truth examination indicator — see below, it changes what the
/// doubly-robust estimator can guarantee.
///
/// # The `examinations` field, and why it matters
///
/// In ordinary click logs, examination is **latent**: a zero click is
/// ambiguous between *"the user never looked at that position"* and *"the user
/// looked and was not interested"*. That ambiguity is the entire reason click
/// models exist, and it is what makes classical doubly-robust estimation
/// awkward here (see the module documentation).
///
/// Some real logging systems *do* record examination — viewport / impression
/// beacons, eye-tracking studies, or a simulator that knows the truth. When
/// they do, set `examinations`, and
/// [`crate::click_model::DoublyRobustEstimator`] becomes *genuinely* doubly
/// robust: unbiased if **either** the propensity model **or** the reward model
/// is correct. Without it, the estimator falls back on the click model's
/// examination *posterior*, which retains unbiasedness only on the
/// propensity-correct branch.
///
/// A session is rejected at validation time if it claims a click at a position
/// it also flags as unexamined (`click ⇒ examined` is an axiom of every model
/// here).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClickSession {
    /// The query this ranking answered. Used to key
    /// [`crate::click_model::ClickRankingPolicy`] scores; the click models
    /// themselves parameterise attractiveness **per document**, so they ignore
    /// it.
    pub query_id: String,
    /// The documents shown, in the order they were shown. Index `i` is
    /// zero-based rank `i`.
    pub ranked_doc_ids: Vec<String>,
    /// `clicks[i] == true` iff the document at rank `i` was clicked.
    pub clicks: Vec<bool>,
    /// Optional ground-truth examination indicators, aligned by rank. `None`
    /// (the usual case) means examination is latent.
    pub examinations: Option<Vec<bool>>,
}

impl ClickSession {
    /// Build a session whose examination indicators are latent.
    #[must_use]
    pub fn new(
        query_id: impl Into<String>,
        ranked_doc_ids: Vec<String>,
        clicks: Vec<bool>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            ranked_doc_ids,
            clicks,
            examinations: None,
        }
    }

    /// Attach ground-truth examination indicators (see the type-level docs for
    /// what this buys you).
    #[must_use]
    pub fn with_examinations(mut self, examinations: Vec<bool>) -> Self {
        self.examinations = Some(examinations);
        self
    }

    /// The number of ranks in this session's impression list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranked_doc_ids.len()
    }

    /// Whether the impression list is empty (rejected by validation).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranked_doc_ids.is_empty()
    }

    /// The zero-based rank of the **first** click, or `None` if the session
    /// has no click at all. This is the cascade model's stopping position.
    #[must_use]
    pub fn first_click_rank(&self) -> Option<usize> {
        self.clicks.iter().position(|&clicked| clicked)
    }

    /// The zero-based rank of the **last** click, or `None` if the session has
    /// no click at all. This is the deepest rank the DBN knows for certain was
    /// examined.
    #[must_use]
    pub fn last_click_rank(&self) -> Option<usize> {
        self.clicks.iter().rposition(|&clicked| clicked)
    }

    /// How many positions in this session were clicked.
    #[must_use]
    pub fn click_count(&self) -> usize {
        self.clicks.iter().filter(|&&clicked| clicked).count()
    }

    /// Validate this session in isolation. `session_index` is only used to
    /// label errors.
    ///
    /// Checks, in order: non-empty impression list; `clicks` length matches;
    /// `examinations` length matches (when present); every click is at an
    /// examined position (when examinations are present); and no document id
    /// occupies two ranks.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::EmptySession`], [`ClickModelError::ClickLengthMismatch`],
    /// [`ClickModelError::ExaminationLengthMismatch`],
    /// [`ClickModelError::ClickWithoutExamination`] or
    /// [`ClickModelError::DuplicateDocumentInSession`], for the first rule the
    /// session breaks.
    pub fn validate(&self, session_index: usize) -> ClickModelResult<()> {
        if self.ranked_doc_ids.is_empty() {
            return Err(ClickModelError::EmptySession { session_index });
        }
        if self.clicks.len() != self.ranked_doc_ids.len() {
            return Err(ClickModelError::ClickLengthMismatch {
                session_index,
                doc_count: self.ranked_doc_ids.len(),
                click_count: self.clicks.len(),
            });
        }
        if let Some(examinations) = &self.examinations {
            if examinations.len() != self.ranked_doc_ids.len() {
                return Err(ClickModelError::ExaminationLengthMismatch {
                    session_index,
                    doc_count: self.ranked_doc_ids.len(),
                    examination_count: examinations.len(),
                });
            }
            for (rank, (&clicked, &examined)) in
                self.clicks.iter().zip(examinations.iter()).enumerate()
            {
                if clicked && !examined {
                    return Err(ClickModelError::ClickWithoutExamination {
                        session_index,
                        rank,
                    });
                }
            }
        }
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for doc_id in &self.ranked_doc_ids {
            if !seen.insert(doc_id.as_str()) {
                return Err(ClickModelError::DuplicateDocumentInSession {
                    session_index,
                    doc_id: doc_id.clone(),
                });
            }
        }
        Ok(())
    }
}

// ── ClickImpression ──────────────────────────────────────────────────────────

/// A single `(session, rank, document, click)` row — the unit every estimator
/// in this module accumulates over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClickImpression<'a> {
    /// The index of the session this impression belongs to within its
    /// [`ClickLog`].
    pub session_index: usize,
    /// The query the session answered.
    pub query_id: &'a str,
    /// The document shown.
    pub doc_id: &'a str,
    /// The zero-based rank it was shown at.
    pub rank: usize,
    /// Whether it was clicked.
    pub clicked: bool,
    /// The ground-truth examination indicator, when the session carries one.
    pub examined: Option<bool>,
}

// ── ClickLog ─────────────────────────────────────────────────────────────────

/// A validated collection of [`ClickSession`]s.
///
/// Every constructor validates: a `ClickLog` that exists is a log every model
/// in this module can consume without further checks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClickLog {
    sessions: Vec<ClickSession>,
}

impl ClickLog {
    /// An empty log. (Empty logs are legal *objects*; fitting one is an
    /// error — see [`ClickModelError::EmptyLog`].)
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: Vec::new(),
        }
    }

    /// Validate and take ownership of a batch of sessions.
    ///
    /// # Errors
    ///
    /// Returns the first [`ClickSession::validate`] failure, labelled with the
    /// offending session's index.
    pub fn from_sessions(sessions: Vec<ClickSession>) -> ClickModelResult<Self> {
        for (session_index, session) in sessions.iter().enumerate() {
            session.validate(session_index)?;
        }
        Ok(Self { sessions })
    }

    /// Validate and append one session.
    ///
    /// # Errors
    ///
    /// Returns the [`ClickSession::validate`] failure, if any; the log is left
    /// unchanged in that case.
    pub fn push(&mut self, session: ClickSession) -> ClickModelResult<()> {
        session.validate(self.sessions.len())?;
        self.sessions.push(session);
        Ok(())
    }

    /// The validated sessions, in insertion order.
    #[must_use]
    pub fn sessions(&self) -> &[ClickSession] {
        &self.sessions
    }

    /// The number of sessions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether the log holds no sessions at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// The deepest rank depth in the log, i.e. `max(session.len())`. Valid
    /// ranks across the whole log are `0..depth()`.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.sessions
            .iter()
            .map(ClickSession::len)
            .max()
            .unwrap_or(0)
    }

    /// Every distinct document id in the log, in **sorted** order.
    ///
    /// Sorted rather than hash order so that every parameter vector this
    /// module builds is indexed deterministically, and every floating-point
    /// accumulation happens in the same order on every run.
    #[must_use]
    pub fn doc_ids(&self) -> Vec<String> {
        let unique: BTreeSet<&str> = self
            .sessions
            .iter()
            .flat_map(|session| session.ranked_doc_ids.iter().map(String::as_str))
            .collect();
        unique.into_iter().map(str::to_owned).collect()
    }

    /// Iterate over every `(session, rank, document, click)` row in the log, in
    /// session order and then rank order.
    pub fn impressions(&self) -> impl Iterator<Item = ClickImpression<'_>> + '_ {
        self.sessions
            .iter()
            .enumerate()
            .flat_map(|(session_index, session)| {
                session
                    .ranked_doc_ids
                    .iter()
                    .enumerate()
                    .map(move |(rank, doc_id)| ClickImpression {
                        session_index,
                        query_id: session.query_id.as_str(),
                        doc_id: doc_id.as_str(),
                        rank,
                        clicked: session.clicks[rank],
                        examined: session
                            .examinations
                            .as_ref()
                            .map(|examinations| examinations[rank]),
                    })
            })
    }

    /// The total number of impressions (documents shown) across all sessions.
    #[must_use]
    pub fn impression_count(&self) -> usize {
        self.sessions.iter().map(ClickSession::len).sum()
    }

    /// The total number of clicks across all sessions.
    #[must_use]
    pub fn click_count(&self) -> usize {
        self.sessions.iter().map(ClickSession::click_count).sum()
    }

    /// Build a sub-log from session indices (used, for example, to measure an
    /// estimator's sampling variance across repeated sub-samples).
    ///
    /// Indices outside `0..len()` are silently skipped — the result is always a
    /// valid log, because every session in it was already validated.
    #[must_use]
    pub fn subset(&self, session_indices: &[usize]) -> Self {
        let sessions = session_indices
            .iter()
            .filter_map(|&index| self.sessions.get(index).cloned())
            .collect();
        Self { sessions }
    }
}

// ── ClickModelConfig ─────────────────────────────────────────────────────────

/// Shared configuration for all three click models' EM loops, plus the
/// propensity clipping floor the counterfactual estimators inherit.
///
/// # Why parameters are clamped away from `0` and `1`
///
/// Every model here evaluates `ln(p)` and `ln(1 - p)` for probabilities `p`
/// built out of the parameters. A parameter that reaches exactly `0` or `1`
/// makes some likelihood term `ln(0) = -inf`, and — worse — makes the E-step's
/// Bayes denominator `1 - γ·α` collapse to zero, producing `0/0 = NaN` and
/// pinning the parameter there forever. [`ClickModelConfig::parameter_floor`]
/// confines every parameter to `[floor, 1 - floor]`.
///
/// This clamping does **not** break EM's monotonicity guarantee. Each
/// parameter's M-step objective is a Bernoulli log-likelihood
/// `n₁·ln θ + n₀·ln(1 - θ)`, which is strictly concave; maximising it over the
/// closed interval `[floor, 1 - floor]` instead of over `(0, 1)` just means
/// projecting the unconstrained maximiser onto that interval. Since the
/// previous iterate also lives in the interval, the constrained maximiser is
/// still at least as good, so `Q(θ_{t+1} | θ_t) ≥ Q(θ_t | θ_t)` and hence the
/// observed-data log-likelihood is still monotone non-decreasing. The same
/// argument covers [`ClickModelConfig::anchor_top_examination`], which fixes
/// `γ₀ = 1` — a feasible set of one point that the previous iterate is already
/// in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClickModelConfig {
    /// Hard cap on EM iterations. Default `100`.
    pub max_iterations: usize,
    /// Convergence tolerance on the **absolute change in log-likelihood**
    /// between consecutive iterations. Default `1e-6`.
    pub tolerance: f64,
    /// Parameters are confined to `[parameter_floor, 1 - parameter_floor]`.
    /// Default `1e-6`.
    pub parameter_floor: f64,
    /// The propensity clipping floor inherited by
    /// [`crate::click_model::PropensityEstimates`]: no examination probability
    /// used as an inverse-propensity weight is ever allowed below this, so no
    /// importance weight ever exceeds `1 / propensity_clip`. Default `0.01`
    /// (a weight cap of `100`).
    pub propensity_clip: f64,
    /// Initial value for every document's attractiveness `α_d`. Default `0.2`.
    pub initial_attractiveness: f64,
    /// Initial value for every document's satisfaction `σ_d` (DBN only).
    /// Default `0.5`.
    pub initial_satisfaction: f64,
    /// Initial value for the DBN's global persistence `γ`. Default `0.9`.
    pub initial_persistence: f64,
    /// The PBM's examination curve is initialised as
    /// `γ_r = initial_examination_decay^r`, a geometric decay with rank.
    /// Default `0.9`. Note this automatically satisfies `γ₀ = 1`, matching
    /// [`ClickModelConfig::anchor_top_examination`].
    pub initial_examination_decay: f64,
    /// Fix the PBM's top-rank examination probability at `γ₀ = 1` ("the user
    /// always looks at the first result").
    ///
    /// **This is not cosmetic — it is what makes the PBM identifiable at
    /// all.** The PBM likelihood depends on `γ_r` and `α_d` only through their
    /// product `γ_r · α_d`, so `(c·γ, α/c)` has exactly the same likelihood as
    /// `(γ, α)` for any scale `c`. Anchoring one parameter pins that scale.
    /// With `γ₀ = 1` fixed, EM recovers the true `γ` and `α` (up to sampling
    /// noise) whenever the ground truth also satisfies `γ₀ = 1`; if the truth
    /// has `γ₀ = g < 1`, EM recovers `γ_r / g` and `α_d · g`, which is the most
    /// the data can possibly tell you. Default `true`.
    pub anchor_top_examination: bool,
}

impl Default for ClickModelConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            parameter_floor: 1e-6,
            propensity_clip: 0.01,
            initial_attractiveness: 0.2,
            initial_satisfaction: 0.5,
            initial_persistence: 0.9,
            initial_examination_decay: 0.9,
            anchor_top_examination: true,
        }
    }
}

impl ClickModelConfig {
    /// A configuration with the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the EM iteration cap.
    #[must_use]
    pub fn with_max_iterations(mut self, value: usize) -> Self {
        self.max_iterations = value;
        self
    }

    /// Override the log-likelihood convergence tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, value: f64) -> Self {
        self.tolerance = value;
        self
    }

    /// Override the parameter clamping floor.
    #[must_use]
    pub fn with_parameter_floor(mut self, value: f64) -> Self {
        self.parameter_floor = value;
        self
    }

    /// Override the propensity clipping floor.
    #[must_use]
    pub fn with_propensity_clip(mut self, value: f64) -> Self {
        self.propensity_clip = value;
        self
    }

    /// Override the initial attractiveness.
    #[must_use]
    pub fn with_initial_attractiveness(mut self, value: f64) -> Self {
        self.initial_attractiveness = value;
        self
    }

    /// Override the initial satisfaction (DBN).
    #[must_use]
    pub fn with_initial_satisfaction(mut self, value: f64) -> Self {
        self.initial_satisfaction = value;
        self
    }

    /// Override the initial persistence (DBN).
    #[must_use]
    pub fn with_initial_persistence(mut self, value: f64) -> Self {
        self.initial_persistence = value;
        self
    }

    /// Override the geometric decay used to initialise the PBM examination
    /// curve.
    #[must_use]
    pub fn with_initial_examination_decay(mut self, value: f64) -> Self {
        self.initial_examination_decay = value;
        self
    }

    /// Turn the `γ₀ = 1` identifiability anchor on or off. Turning it **off**
    /// leaves the PBM's overall scale undetermined — read
    /// [`ClickModelConfig::anchor_top_examination`] before doing so.
    #[must_use]
    pub fn with_anchor_top_examination(mut self, value: bool) -> Self {
        self.anchor_top_examination = value;
        self
    }

    /// Clamp `value` into `[parameter_floor, 1 - parameter_floor]`.
    #[must_use]
    pub fn clamp_parameter(&self, value: f64) -> f64 {
        if value.is_nan() {
            return self.parameter_floor;
        }
        value.clamp(self.parameter_floor, 1.0 - self.parameter_floor)
    }

    /// Check that every configured value is usable.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::InvalidConfig`] when `max_iterations` is zero, when
    /// `tolerance` is negative or non-finite, when `parameter_floor` or
    /// `propensity_clip` is outside `(0, 0.5)` / `(0, 1]`, or when any initial
    /// parameter is outside `(0, 1)`.
    pub fn validate(&self) -> ClickModelResult<()> {
        let invalid = |reason: &str| ClickModelError::InvalidConfig {
            reason: reason.to_owned(),
        };
        if self.max_iterations == 0 {
            return Err(invalid("max_iterations must be at least 1"));
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(invalid("tolerance must be finite and non-negative"));
        }
        if !self.parameter_floor.is_finite()
            || self.parameter_floor <= 0.0
            || self.parameter_floor >= 0.5
        {
            return Err(invalid("parameter_floor must lie strictly inside (0, 0.5)"));
        }
        if !self.propensity_clip.is_finite()
            || self.propensity_clip <= 0.0
            || self.propensity_clip > 1.0
        {
            return Err(invalid("propensity_clip must lie inside (0, 1]"));
        }
        for (name, value) in [
            ("initial_attractiveness", self.initial_attractiveness),
            ("initial_satisfaction", self.initial_satisfaction),
            ("initial_persistence", self.initial_persistence),
            ("initial_examination_decay", self.initial_examination_decay),
        ] {
            if !value.is_finite() || value <= 0.0 || value >= 1.0 {
                return Err(invalid(&format!(
                    "{name} must lie strictly inside (0, 1), got {value}"
                )));
            }
        }
        Ok(())
    }
}

// ── ClickModelFit ────────────────────────────────────────────────────────────

/// The report an EM run hands back: how many iterations it took, whether it hit
/// the tolerance, and the **complete log-likelihood trace**.
///
/// `log_likelihood_history[0]` is the log-likelihood *at initialisation*, before
/// any EM step; `log_likelihood_history[i]` is the log-likelihood after `i` EM
/// steps. Expectation-Maximisation guarantees this sequence is monotone
/// non-decreasing — [`ClickModelFit::is_monotone`] checks it, and every model in
/// this module is tested against it, because a decrease is *proof* of a bug in
/// the E-step or the M-step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClickModelFit {
    /// The number of EM steps actually performed.
    pub iterations: usize,
    /// Whether the run stopped because the log-likelihood change fell below
    /// [`ClickModelConfig::tolerance`] (as opposed to exhausting
    /// [`ClickModelConfig::max_iterations`]).
    pub converged: bool,
    /// The log-likelihood at initialisation, then after each EM step.
    pub log_likelihood_history: Vec<f64>,
    /// The log-likelihood of the returned parameters.
    pub final_log_likelihood: f64,
}

impl ClickModelFit {
    /// Whether the log-likelihood trace never decreased by more than
    /// `epsilon` (a small allowance for floating-point summation noise; pass
    /// `0.0` for an exact check).
    ///
    /// **Beware the scale.** A log-likelihood is a *sum* over impressions, so a
    /// large log has a large `|LL|`, and its unavoidable floating-point rounding
    /// error grows in proportion. An absolute `epsilon` that is sharp for a
    /// 300-session log is meaningless for a 100,000-session one. Prefer
    /// [`ClickModelFit::is_monotone_relative`] whenever the log is big.
    #[must_use]
    pub fn is_monotone(&self, epsilon: f64) -> bool {
        self.log_likelihood_history
            .windows(2)
            .all(|pair| pair[1] >= pair[0] - epsilon)
    }

    /// Whether the log-likelihood trace never decreased by more than
    /// `relative_epsilon · max(1, |LL|)` — the scale-aware version of
    /// [`ClickModelFit::is_monotone`], and the one to reach for on a real log.
    ///
    /// Even with the compensated (Kahan) summation this module uses, evaluating a
    /// log-likelihood over `n` impressions carries a rounding error on the order
    /// of `machine-epsilon · |LL|` (≈ `2.2e-16 · |LL|`). A `relative_epsilon` of
    /// `1e-10` therefore leaves a safety factor of roughly a million over the
    /// *numerics*, while remaining orders of magnitude tighter than any decrease
    /// a genuinely wrong E-step or M-step could hide behind: a broken derivation
    /// moves the likelihood by a fraction of *itself*, not by its last bits.
    #[must_use]
    pub fn is_monotone_relative(&self, relative_epsilon: f64) -> bool {
        self.log_likelihood_history.windows(2).all(|pair| {
            let tolerance = relative_epsilon * pair[0].abs().max(1.0);
            pair[1] >= pair[0] - tolerance
        })
    }

    /// The largest single-step *decrease* in the log-likelihood trace, or `0.0`
    /// if it never decreased. Anything materially above zero is an EM bug.
    #[must_use]
    pub fn worst_decrease(&self) -> f64 {
        self.log_likelihood_history
            .windows(2)
            .map(|pair| pair[0] - pair[1])
            .fold(0.0_f64, f64::max)
    }

    /// Total improvement from initialisation to the final parameters.
    #[must_use]
    pub fn total_improvement(&self) -> f64 {
        match (
            self.log_likelihood_history.first(),
            self.log_likelihood_history.last(),
        ) {
            (Some(&first), Some(&last)) => last - first,
            _ => 0.0,
        }
    }
}

// ── ClickModel trait ─────────────────────────────────────────────────────────

/// The common surface of every click model in this module.
///
/// All three models are *generative* models of the click vector, all three are
/// fitted by (a special case of) Expectation-Maximisation, and all three expose
/// a marginal **examination curve** — the probability that the document at a
/// given rank is looked at. That curve is precisely what
/// [`crate::click_model::PropensityEstimates`] turns into inverse-propensity
/// weights, which is how a click model becomes a debiasing tool rather than
/// merely a descriptive one.
pub trait ClickModel {
    /// Fit the model's parameters to `log` by EM, replacing any previous fit.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::EmptyLog`] for a log with no sessions, and
    /// [`ClickModelError::InvalidConfig`] if the model's configuration is
    /// unusable.
    fn fit(&mut self, log: &ClickLog) -> ClickModelResult<ClickModelFit>;

    /// The model's marginal probability that `doc_id` is **clicked** when shown
    /// at zero-based `rank`.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before [`ClickModel::fit`],
    /// [`ClickModelError::UnknownDocument`] for a document the fit never saw,
    /// and [`ClickModelError::UnknownRank`] for a rank deeper than the fit ever
    /// saw.
    fn predict_click_prob(&self, doc_id: &str, rank: usize) -> ClickModelResult<f64>;

    /// The model's log-likelihood of an arbitrary log under its current
    /// parameters. Pass the training log to reproduce
    /// [`ClickModelFit::final_log_likelihood`]; pass a held-out log to compare
    /// models.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before [`ClickModel::fit`], and
    /// [`ClickModelError::UnknownDocument`] / [`ClickModelError::UnknownRank`]
    /// if `log` mentions documents or ranks the fit never saw.
    fn log_likelihood(&self, log: &ClickLog) -> ClickModelResult<f64>;

    /// The fitted **marginal examination probability** at each zero-based rank:
    /// entry `r` is the probability that whatever document sits at rank `r` is
    /// looked at.
    ///
    /// For the PBM this *is* the parameter `γ_r`. For the cascade model and the
    /// DBN, examination depends on the documents above, so this is the model's
    /// examination probability averaged over the rankings in the log it was
    /// fitted on. Empty before a fit.
    fn examination_curve(&self) -> &[f64];

    /// A short, stable name for the model, used in error messages.
    fn model_name(&self) -> &'static str;
}

// ── shared fitting helpers ───────────────────────────────────────────────────

/// Build the sorted document vocabulary of a log plus its reverse index.
///
/// The vector is sorted, so parameter vectors are laid out identically on every
/// run and every floating-point accumulation happens in the same order.
pub(crate) fn document_vocabulary(log: &ClickLog) -> (Vec<String>, HashMap<String, usize>) {
    let doc_ids = log.doc_ids();
    let index = doc_ids
        .iter()
        .enumerate()
        .map(|(index, doc_id)| (doc_id.clone(), index))
        .collect();
    (doc_ids, index)
}

/// A Kahan–Babuška compensated accumulator.
///
/// A log-likelihood here is a sum of tens or hundreds of thousands of `ln`
/// terms. Naive left-to-right summation of `n` terms accumulates a rounding
/// error that grows like `n · ε · Σ|xᵢ|`, which for a 100,000-impression log is
/// large enough to *swamp* the final, tiny improvements EM makes as it
/// converges — and to make the log-likelihood trace appear to go **downhill**
/// when the mathematics guarantees it cannot. The monotonicity of that trace is
/// this module's central correctness invariant, so it is worth spending a few
/// extra flops to keep it observable: compensated summation reduces the error to
/// `O(ε · |Σ xᵢ|)`, independent of `n`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KahanSum {
    total: f64,
    compensation: f64,
}

impl KahanSum {
    /// A zeroed accumulator.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add one term, carrying the rounding error forward.
    pub(crate) fn add(&mut self, value: f64) {
        let corrected = value - self.compensation;
        let candidate = self.total + corrected;
        // `(candidate − total)` is the part of `corrected` that actually landed;
        // whatever is left over is the error we carry into the next term.
        self.compensation = (candidate - self.total) - corrected;
        self.total = candidate;
    }

    /// The compensated total.
    pub(crate) fn total(self) -> f64 {
        self.total
    }
}

/// `ln(p)` with `p` already guaranteed strictly positive by the parameter
/// clamp — this exists so every likelihood term in the module goes through one
/// documented, defensively-floored code path.
pub(crate) fn safe_ln(value: f64) -> f64 {
    // Every caller passes a probability built from parameters clamped into
    // `[floor, 1 - floor]`, so `value` is bounded below by `floor²` (≈ 1e-12 at
    // the default floor) and `ln` is finite. The clamp below is a belt-and-
    // braces guard against a caller-supplied propensity curve that slipped
    // through validation; it never engages on the fitted paths.
    value.max(f64::MIN_POSITIVE).ln()
}
