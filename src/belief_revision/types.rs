//! Types for the `belief_revision` module: the configuration, the hypothesis
//! and evidence data structures, the belief-state distribution, the error
//! enum, and the run-result struct.
//!
//! The mathematical machinery that operates on these types lives in
//! [`super::bayes`]; this file is concerned with data and validation-free
//! read-out accessors. [`BeliefState`] deliberately stores its distribution
//! in **log-probability space** and exposes [`BeliefState::posterior`] as the
//! public linear read-out, mirroring how the update math is carried out.

use std::collections::HashSet;

use thiserror::Error;

/// The default likelihood floor used by [`EvidenceUpdate::from_text_support`]
/// when the caller-supplied floor is out of range.
pub(crate) const DEFAULT_MIN_LIKELIHOOD: f64 = 1e-6;

// ── tokenization (self-contained lexical helpers) ────────────────────────────
//
// Each module in this crate owns its own small lexical toolkit rather than
// sharing private helpers across module boundaries.

/// Split `text` on non-alphanumeric boundaries, lowercase, and keep non-empty
/// fragments.
fn tokenize_lower(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The distinct content-token vocabulary of `text` (tokens of at least three
/// characters), used by the text-support likelihood heuristic.
fn content_terms(text: &str) -> HashSet<String> {
    tokenize_lower(text)
        .into_iter()
        .filter(|t| t.chars().count() >= 3)
        .collect()
}

// ── BeliefRevisionError ──────────────────────────────────────────────────────

/// Errors produced by the `belief_revision` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BeliefRevisionError {
    /// The hypothesis set was empty; at least one hypothesis is required to
    /// form a distribution.
    #[error("hypothesis set is empty: at least one hypothesis is required")]
    EmptyHypothesisSet,
    /// Two hypotheses shared the same id; hypothesis ids must be unique so the
    /// posterior is unambiguous.
    #[error("duplicate hypothesis id '{id}': hypothesis ids must be unique")]
    DuplicateHypothesisId {
        /// The id that appeared more than once.
        id: String,
    },
    /// A caller-supplied prior had a different length than the hypothesis set.
    #[error("prior length {found} does not match the number of hypotheses {expected}")]
    PriorLengthMismatch {
        /// The number of hypotheses (the required prior length).
        expected: usize,
        /// The length of the prior actually supplied.
        found: usize,
    },
    /// A caller-supplied prior did not sum to `1.0` within the configured
    /// tolerance.
    #[error("prior must sum to 1.0 within tolerance, but sums to {sum}")]
    InvalidPriorSum {
        /// The observed sum of the prior entries.
        sum: f64,
    },
    /// A caller-supplied prior contained a negative entry.
    #[error("prior entry for hypothesis '{id}' is negative ({value})")]
    NegativePrior {
        /// The id of the hypothesis with the negative prior.
        id: String,
        /// The offending value.
        value: f64,
    },
    /// A caller-supplied prior contained a non-finite (`NaN` or infinite)
    /// entry.
    #[error("prior entry for hypothesis '{id}' is not finite (NaN or infinite)")]
    NonFinitePrior {
        /// The id of the hypothesis with the non-finite prior.
        id: String,
    },
    /// An evidence update carried no likelihoods at all, so it conveys no
    /// information.
    #[error("evidence '{description}' carries no likelihoods")]
    EmptyEvidence {
        /// The description of the empty evidence update.
        description: String,
    },
    /// A strict-mode evidence update did not provide a likelihood for one of
    /// the hypotheses.
    #[error("evidence is missing a likelihood for hypothesis '{id}'")]
    MissingLikelihood {
        /// The id of the hypothesis whose likelihood was missing.
        id: String,
    },
    /// An evidence update referenced a hypothesis id that is not in the belief
    /// state's hypothesis set.
    #[error("evidence references unknown hypothesis '{id}'")]
    UnknownHypothesis {
        /// The unknown hypothesis id.
        id: String,
    },
    /// An evidence likelihood was non-finite (`NaN` or infinite).
    #[error("likelihood for hypothesis '{id}' is not finite (NaN or infinite)")]
    NonFiniteLikelihood {
        /// The id of the hypothesis with the non-finite likelihood.
        id: String,
    },
    /// An evidence likelihood was negative; likelihoods are `P(evidence | hypothesis)`
    /// and must be non-negative.
    #[error("likelihood for hypothesis '{id}' is negative ({value})")]
    NegativeLikelihood {
        /// The id of the hypothesis with the negative likelihood.
        id: String,
        /// The offending value.
        value: f64,
    },
    /// The engine configuration was invalid.
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// A human-readable description of what was wrong.
        reason: String,
    },
}

// ── BeliefRevisionConfig ─────────────────────────────────────────────────────

/// Configuration for [`super::engine::BeliefRevisionEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionConfig {
    /// Minimum likelihood floor (epsilon). Every per-hypothesis likelihood is
    /// clamped up to this value before its logarithm is taken, so a single
    /// zero-likelihood observation cannot drive a hypothesis to log-probability
    /// `-inf` — a permanent, unrecoverable "impossible" state that no later
    /// favorable evidence could climb back out of. Must lie in the open
    /// interval `(0, 1)`. Default `1e-6`.
    pub min_likelihood: f64,
    /// Absolute tolerance used when validating that a caller-supplied prior
    /// sums to `1.0`. Must be finite and non-negative. Default `1e-6`.
    pub prior_sum_tolerance: f64,
    /// Whether every evidence update must specify a likelihood for *every*
    /// hypothesis (and reference no unknown hypothesis). In strict mode a
    /// missing or unknown likelihood is a hard error; in lenient mode a missing
    /// likelihood defaults to a neutral `1.0` (uninformative) and a likelihood
    /// for an unknown hypothesis is ignored. Default `true`.
    pub strict: bool,
}

impl Default for BeliefRevisionConfig {
    fn default() -> Self {
        Self {
            min_likelihood: DEFAULT_MIN_LIKELIHOOD,
            prior_sum_tolerance: 1e-6,
            strict: true,
        }
    }
}

impl BeliefRevisionConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum likelihood floor (epsilon).
    #[must_use]
    pub fn with_min_likelihood(mut self, min_likelihood: f64) -> Self {
        self.min_likelihood = min_likelihood;
        self
    }

    /// Set the absolute tolerance for prior-sum validation.
    #[must_use]
    pub fn with_prior_sum_tolerance(mut self, prior_sum_tolerance: f64) -> Self {
        self.prior_sum_tolerance = prior_sum_tolerance;
        self
    }

    /// Set the strictness of evidence-likelihood coverage validation.
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BeliefRevisionError::InvalidConfig`] if `min_likelihood` is
    /// not in the open interval `(0, 1)`, or if `prior_sum_tolerance` is
    /// negative or non-finite.
    pub fn validate(&self) -> Result<(), BeliefRevisionError> {
        if !self.min_likelihood.is_finite()
            || self.min_likelihood <= 0.0
            || self.min_likelihood >= 1.0
        {
            return Err(BeliefRevisionError::InvalidConfig {
                reason: format!(
                    "min_likelihood must be in (0, 1), got {}",
                    self.min_likelihood
                ),
            });
        }
        if !self.prior_sum_tolerance.is_finite() || self.prior_sum_tolerance < 0.0 {
            return Err(BeliefRevisionError::InvalidConfig {
                reason: format!(
                    "prior_sum_tolerance must be finite and non-negative, got {}",
                    self.prior_sum_tolerance
                ),
            });
        }
        Ok(())
    }
}

// ── BeliefHypothesis ─────────────────────────────────────────────────────────

/// A single candidate hypothesis (answer) that the belief state assigns a
/// probability to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefHypothesis {
    /// Stable, unique identifier for this hypothesis (used to key likelihoods).
    pub id: String,
    /// Human-readable description of the hypothesis. Used as the text signal by
    /// the [`EvidenceUpdate::from_text_support`] likelihood heuristic.
    pub description: String,
}

impl BeliefHypothesis {
    /// Construct a hypothesis from an id and a description.
    #[must_use]
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
        }
    }
}

// ── EvidenceUpdate ───────────────────────────────────────────────────────────

/// One piece of retrieved evidence, carrying a likelihood
/// `P(evidence | hypothesis)` for each hypothesis.
///
/// Likelihoods are stored as an ordered `(hypothesis_id, likelihood)` list
/// (rather than a hash map) so the update is fully deterministic. Callers can
/// supply likelihoods explicitly (via [`with_likelihood`](Self::with_likelihood)
/// or [`from_likelihoods`](Self::from_likelihoods)) or derive them from the
/// evidence text with the deterministic lexical heuristic
/// [`from_text_support`](Self::from_text_support).
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceUpdate {
    /// The evidence text or a short label for it.
    pub description: String,
    /// Per-hypothesis likelihoods `P(evidence | hypothesis)` as
    /// `(hypothesis_id, likelihood)` pairs.
    pub likelihoods: Vec<(String, f64)>,
}

impl EvidenceUpdate {
    /// Create an evidence update with a description and no likelihoods yet.
    #[must_use]
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            likelihoods: Vec::new(),
        }
    }

    /// Builder: attach a likelihood `P(evidence | hypothesis)` for one
    /// hypothesis.
    #[must_use]
    pub fn with_likelihood(mut self, hypothesis_id: impl Into<String>, likelihood: f64) -> Self {
        self.likelihoods.push((hypothesis_id.into(), likelihood));
        self
    }

    /// Create an evidence update from a description and a full list of
    /// `(hypothesis_id, likelihood)` pairs.
    #[must_use]
    pub fn from_likelihoods(
        description: impl Into<String>,
        likelihoods: Vec<(String, f64)>,
    ) -> Self {
        Self {
            description: description.into(),
            likelihoods,
        }
    }

    /// Look up the likelihood for a hypothesis id, if present (the first
    /// matching entry wins).
    #[must_use]
    pub fn likelihood_for(&self, hypothesis_id: &str) -> Option<f64> {
        self.likelihoods
            .iter()
            .find(|(id, _)| id == hypothesis_id)
            .map(|(_, value)| *value)
    }

    /// Derive per-hypothesis likelihoods from the evidence text using a
    /// deterministic lexical-overlap heuristic.
    ///
    /// For each hypothesis, the overlap ratio is the fraction of the
    /// hypothesis description's content terms (tokens of at least three
    /// characters) that also appear in the evidence text, in `[0, 1]`. The
    /// likelihood is `floor + (1 - floor) * ratio`, so it lies in
    /// `[floor, 1]`: a hypothesis whose description is well supported by the
    /// evidence gets a likelihood near `1`, one sharing no vocabulary gets the
    /// `floor`, and no hypothesis ever gets exactly `0` (which would trigger
    /// the `log(0) = -inf` trap). `floor` is `min_likelihood` when it lies in
    /// `(0, 1)`, otherwise [`DEFAULT_MIN_LIKELIHOOD`].
    #[must_use]
    pub fn from_text_support(
        description: impl Into<String>,
        hypotheses: &[BeliefHypothesis],
        min_likelihood: f64,
    ) -> Self {
        let description = description.into();
        let evidence_terms = content_terms(&description);
        let floor = if min_likelihood > 0.0 && min_likelihood < 1.0 {
            min_likelihood
        } else {
            DEFAULT_MIN_LIKELIHOOD
        };
        let likelihoods = hypotheses
            .iter()
            .map(|hypothesis| {
                let hypothesis_terms = content_terms(&hypothesis.description);
                let ratio = if hypothesis_terms.is_empty() {
                    0.0
                } else {
                    let shared = hypothesis_terms
                        .iter()
                        .filter(|term| evidence_terms.contains(term.as_str()))
                        .count();
                    #[allow(clippy::cast_precision_loss)]
                    let ratio = shared as f64 / hypothesis_terms.len() as f64;
                    ratio
                };
                let likelihood = floor + (1.0 - floor) * ratio;
                (hypothesis.id.clone(), likelihood)
            })
            .collect();
        Self {
            description,
            likelihoods,
        }
    }
}

// ── BeliefState ──────────────────────────────────────────────────────────────

/// A probability distribution over a fixed set of [`BeliefHypothesis`]es,
/// stored internally in **log-probability space**.
///
/// The internal `log_probs` are kept normalized so that
/// `logsumexp(log_probs) == 0`, i.e. the exponentiated distribution sums to
/// `1`. Storing logs (rather than linear probabilities) is what makes the
/// state numerically robust under many sequential Bayesian updates: each
/// update *adds* a log-likelihood instead of *multiplying* a small
/// probability, so the running values never underflow to zero. The public
/// [`posterior`](Self::posterior) accessor converts back to normalized linear
/// probabilities on demand.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefState {
    hypotheses: Vec<BeliefHypothesis>,
    log_probs: Vec<f64>,
    updates_applied: usize,
}

impl BeliefState {
    /// Construct a state from hypotheses and their (assumed-normalized)
    /// log-probabilities. Internal constructor used by the engine.
    pub(crate) fn new_internal(hypotheses: Vec<BeliefHypothesis>, log_probs: Vec<f64>) -> Self {
        Self {
            hypotheses,
            log_probs,
            updates_applied: 0,
        }
    }

    /// Mutable access to the internal log-probabilities, for the engine's
    /// in-place update.
    pub(crate) fn log_probs_mut(&mut self) -> &mut [f64] {
        &mut self.log_probs
    }

    /// Increment the applied-update counter.
    pub(crate) fn increment_updates(&mut self) {
        self.updates_applied += 1;
    }

    /// The hypotheses this state is defined over, in their original order.
    #[must_use]
    pub fn hypotheses(&self) -> &[BeliefHypothesis] {
        &self.hypotheses
    }

    /// The number of hypotheses.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    /// `true` when there are no hypotheses (never the case for a state built
    /// by the engine, which rejects an empty hypothesis set).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty()
    }

    /// The number of evidence updates applied to this state so far.
    #[must_use]
    pub fn updates_applied(&self) -> usize {
        self.updates_applied
    }

    /// The raw internal log-probabilities (normalized so that their
    /// `logsumexp` is `0`), aligned with [`hypotheses`](Self::hypotheses).
    #[must_use]
    pub fn log_probs(&self) -> &[f64] {
        &self.log_probs
    }

    /// The posterior as normalized **linear** probabilities, as
    /// `(hypothesis_id, probability)` pairs aligned with the hypothesis order.
    ///
    /// The probabilities are non-negative and sum to `1.0` within
    /// floating-point tolerance.
    #[must_use]
    pub fn posterior(&self) -> Vec<(String, f64)> {
        let linear = super::bayes::log_probs_to_linear(&self.log_probs);
        self.hypotheses
            .iter()
            .zip(linear)
            .map(|(hypothesis, probability)| (hypothesis.id.clone(), probability))
            .collect()
    }

    /// The posterior as normalized **log** probabilities, as
    /// `(hypothesis_id, log_probability)` pairs. Each `log_probability` is
    /// `<= 0` and `exp` of the whole vector sums to `1`.
    #[must_use]
    pub fn log_posterior(&self) -> Vec<(String, f64)> {
        let z = super::bayes::logsumexp(&self.log_probs);
        self.hypotheses
            .iter()
            .zip(&self.log_probs)
            .map(|(hypothesis, &log_prob)| {
                let normalized = if z.is_finite() {
                    log_prob - z
                } else {
                    log_prob
                };
                (hypothesis.id.clone(), normalized)
            })
            .collect()
    }

    /// The linear posterior probability of a hypothesis by id, or `None` when
    /// no hypothesis has that id.
    #[must_use]
    pub fn probability_of(&self, hypothesis_id: &str) -> Option<f64> {
        let index = self
            .hypotheses
            .iter()
            .position(|hypothesis| hypothesis.id == hypothesis_id)?;
        let linear = super::bayes::log_probs_to_linear(&self.log_probs);
        linear.get(index).copied()
    }

    /// The index of the most likely hypothesis (argmax of the posterior), or
    /// `None` when the state is empty. Ties are resolved deterministically in
    /// favor of the lowest index.
    #[must_use]
    pub fn most_likely_index(&self) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (index, &log_prob) in self.log_probs.iter().enumerate() {
            let is_better = match best {
                None => true,
                Some((_, best_log_prob)) => log_prob > best_log_prob,
            };
            if is_better {
                best = Some((index, log_prob));
            }
        }
        best.map(|(index, _)| index)
    }

    /// The Shannon entropy (in nats) of the posterior — `0` when one
    /// hypothesis holds all the mass, `ln(k)` when the `k` hypotheses are
    /// equiprobable.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        super::bayes::entropy_nats(&self.log_probs)
    }
}

// ── BeliefRevisionResult ─────────────────────────────────────────────────────

/// The outcome of a full [`super::engine::BeliefRevisionEngine::run`]: the
/// final belief state plus convenient derived summaries.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionResult {
    /// The final belief state after applying every evidence update.
    pub state: BeliefState,
    /// The single most likely hypothesis (argmax of the final posterior).
    pub most_likely: BeliefHypothesis,
    /// The number of evidence updates that were applied.
    pub updates_applied: usize,
    /// Shannon entropy (in nats) of the final posterior distribution.
    pub entropy: f64,
}

impl BeliefRevisionResult {
    /// The final posterior as normalized linear `(hypothesis_id, probability)`
    /// pairs.
    #[must_use]
    pub fn posterior(&self) -> Vec<(String, f64)> {
        self.state.posterior()
    }

    /// The final linear posterior probability of a hypothesis by id.
    #[must_use]
    pub fn probability_of(&self, hypothesis_id: &str) -> Option<f64> {
        self.state.probability_of(hypothesis_id)
    }
}
