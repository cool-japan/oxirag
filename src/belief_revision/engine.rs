//! [`BeliefRevisionEngine`] — the belief-state lifecycle: initialize a prior,
//! apply a sequence of evidence updates, and query the posterior.
//!
//! The engine is a thin, deterministic driver over the log-space math in
//! [`super::bayes`]. It validates every input (hypothesis set, prior, and each
//! evidence update's likelihoods) and never panics: all failure modes surface
//! as a [`BeliefRevisionError`].

use super::bayes::{self, LikelihoodRatio};
use super::types::{
    BeliefHypothesis, BeliefRevisionConfig, BeliefRevisionError, BeliefRevisionResult, BeliefState,
    EvidenceUpdate,
};

/// Drives the Bayesian belief-revision lifecycle: [`init`](Self::init) builds
/// a belief state from a hypothesis set and an optional prior;
/// [`update`](Self::update) and [`update_sequence`](Self::update_sequence)
/// fold retrieved evidence into it via genuine log-space Bayesian updates; and
/// [`posterior`](Self::posterior) / [`most_likely`](Self::most_likely) read it
/// back out.
#[derive(Debug, Clone, Default)]
pub struct BeliefRevisionEngine {
    /// Configuration for this engine.
    pub config: BeliefRevisionConfig,
}

impl BeliefRevisionEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: BeliefRevisionConfig) -> Self {
        Self { config }
    }

    /// The engine's configuration.
    #[must_use]
    pub fn config(&self) -> &BeliefRevisionConfig {
        &self.config
    }

    /// Initialize a belief state over `hypotheses` with an optional `prior`.
    ///
    /// When `prior` is `None` a **uniform** prior is used (every hypothesis
    /// gets probability `1 / n`). When `prior` is `Some(p)`, `p` is interpreted
    /// positionally, aligned with `hypotheses`, and must be a valid probability
    /// distribution: the same length as `hypotheses`, all entries finite and
    /// non-negative, and summing to `1.0` within
    /// [`BeliefRevisionConfig::prior_sum_tolerance`]. A prior entry of exactly
    /// `0` is permitted but is floored to [`BeliefRevisionConfig::min_likelihood`]
    /// in log space so it can later recover.
    ///
    /// # Errors
    ///
    /// - [`BeliefRevisionError::InvalidConfig`] if the configuration is invalid.
    /// - [`BeliefRevisionError::EmptyHypothesisSet`] if `hypotheses` is empty.
    /// - [`BeliefRevisionError::DuplicateHypothesisId`] if two hypotheses share
    ///   an id.
    /// - [`BeliefRevisionError::PriorLengthMismatch`] if a supplied prior's
    ///   length differs from the hypothesis count.
    /// - [`BeliefRevisionError::NonFinitePrior`] / [`BeliefRevisionError::NegativePrior`]
    ///   / [`BeliefRevisionError::InvalidPriorSum`] if a supplied prior is not a
    ///   valid probability distribution.
    pub fn init(
        &self,
        hypotheses: Vec<BeliefHypothesis>,
        prior: Option<&[f64]>,
    ) -> Result<BeliefState, BeliefRevisionError> {
        self.config.validate()?;

        if hypotheses.is_empty() {
            return Err(BeliefRevisionError::EmptyHypothesisSet);
        }

        // Reject duplicate ids so the posterior is unambiguous.
        for (i, hypothesis) in hypotheses.iter().enumerate() {
            if hypotheses[..i]
                .iter()
                .any(|earlier| earlier.id == hypothesis.id)
            {
                return Err(BeliefRevisionError::DuplicateHypothesisId {
                    id: hypothesis.id.clone(),
                });
            }
        }

        let n = hypotheses.len();
        let mut log_probs = match prior {
            None => {
                #[allow(clippy::cast_precision_loss)]
                let uniform_log = (1.0 / n as f64).ln();
                vec![uniform_log; n]
            }
            Some(prior_values) => {
                if prior_values.len() != n {
                    return Err(BeliefRevisionError::PriorLengthMismatch {
                        expected: n,
                        found: prior_values.len(),
                    });
                }
                let mut sum = 0.0;
                for (value, hypothesis) in prior_values.iter().zip(&hypotheses) {
                    if !value.is_finite() {
                        return Err(BeliefRevisionError::NonFinitePrior {
                            id: hypothesis.id.clone(),
                        });
                    }
                    if *value < 0.0 {
                        return Err(BeliefRevisionError::NegativePrior {
                            id: hypothesis.id.clone(),
                            value: *value,
                        });
                    }
                    sum += *value;
                }
                if (sum - 1.0).abs() > self.config.prior_sum_tolerance {
                    return Err(BeliefRevisionError::InvalidPriorSum { sum });
                }
                prior_values
                    .iter()
                    .map(|&value| bayes::log_likelihood_floored(value, self.config.min_likelihood))
                    .collect()
            }
        };

        // Ensure the state starts exactly normalized (matters after flooring a
        // prior that contained zeros).
        bayes::normalize_log_probs(&mut log_probs);
        Ok(BeliefState::new_internal(hypotheses, log_probs))
    }

    /// Apply a single evidence update to `state`, in place.
    ///
    /// Each hypothesis's log-probability gains `ln(P(evidence | hypothesis))`
    /// (with the likelihood floored to [`BeliefRevisionConfig::min_likelihood`]),
    /// and the distribution is then renormalized via a numerically stable
    /// `logsumexp`. In strict mode every hypothesis must have a likelihood and
    /// the update may reference no unknown hypothesis; in lenient mode a
    /// missing likelihood defaults to a neutral `1.0` and an unknown one is
    /// ignored.
    ///
    /// # Errors
    ///
    /// - [`BeliefRevisionError::EmptyEvidence`] if the update carries no
    ///   likelihoods.
    /// - [`BeliefRevisionError::UnknownHypothesis`] (strict mode) if the update
    ///   references a hypothesis not in `state`.
    /// - [`BeliefRevisionError::MissingLikelihood`] (strict mode) if a
    ///   hypothesis has no likelihood.
    /// - [`BeliefRevisionError::NonFiniteLikelihood`] / [`BeliefRevisionError::NegativeLikelihood`]
    ///   if a likelihood is `NaN`/infinite or negative.
    pub fn update(
        &self,
        state: &mut BeliefState,
        evidence: &EvidenceUpdate,
    ) -> Result<(), BeliefRevisionError> {
        if evidence.likelihoods.is_empty() {
            return Err(BeliefRevisionError::EmptyEvidence {
                description: evidence.description.clone(),
            });
        }

        // In strict mode, an evidence likelihood for an unknown hypothesis is an
        // error (a mismatched likelihood-map-vs-hypothesis-set).
        if self.config.strict {
            for (id, _) in &evidence.likelihoods {
                if !state
                    .hypotheses()
                    .iter()
                    .any(|hypothesis| &hypothesis.id == id)
                {
                    return Err(BeliefRevisionError::UnknownHypothesis { id: id.clone() });
                }
            }
        }

        let mut log_likelihoods = Vec::with_capacity(state.len());
        for hypothesis in state.hypotheses() {
            let likelihood = match evidence.likelihood_for(&hypothesis.id) {
                Some(value) => value,
                None if self.config.strict => {
                    return Err(BeliefRevisionError::MissingLikelihood {
                        id: hypothesis.id.clone(),
                    });
                }
                // Lenient mode: a missing likelihood is uninformative.
                None => 1.0,
            };
            if !likelihood.is_finite() {
                return Err(BeliefRevisionError::NonFiniteLikelihood {
                    id: hypothesis.id.clone(),
                });
            }
            if likelihood < 0.0 {
                return Err(BeliefRevisionError::NegativeLikelihood {
                    id: hypothesis.id.clone(),
                    value: likelihood,
                });
            }
            log_likelihoods.push(bayes::log_likelihood_floored(
                likelihood,
                self.config.min_likelihood,
            ));
        }

        bayes::bayes_update_log(state.log_probs_mut(), &log_likelihoods);
        state.increment_updates();
        Ok(())
    }

    /// Apply a sequence of evidence updates to `state`, in order.
    ///
    /// An empty slice is a valid no-op that leaves `state` unchanged. If any
    /// update fails, the error is returned immediately and `state` reflects the
    /// updates applied up to (but not including) the failing one.
    ///
    /// # Errors
    ///
    /// Any error from [`update`](Self::update).
    pub fn update_sequence(
        &self,
        state: &mut BeliefState,
        evidence: &[EvidenceUpdate],
    ) -> Result<(), BeliefRevisionError> {
        for update in evidence {
            self.update(state, update)?;
        }
        Ok(())
    }

    /// The posterior of `state` as normalized linear `(hypothesis_id, probability)`
    /// pairs. Delegates to [`BeliefState::posterior`].
    #[must_use]
    #[allow(clippy::unused_self)]
    pub fn posterior(&self, state: &BeliefState) -> Vec<(String, f64)> {
        state.posterior()
    }

    /// The most likely hypothesis in `state` (argmax of the posterior).
    ///
    /// # Errors
    ///
    /// [`BeliefRevisionError::EmptyHypothesisSet`] if `state` has no
    /// hypotheses (never the case for a state built by [`init`](Self::init)).
    #[allow(clippy::unused_self)]
    pub fn most_likely(
        &self,
        state: &BeliefState,
    ) -> Result<BeliefHypothesis, BeliefRevisionError> {
        state
            .most_likely_index()
            .and_then(|index| state.hypotheses().get(index).cloned())
            .ok_or(BeliefRevisionError::EmptyHypothesisSet)
    }

    /// Compute the pairwise [`LikelihoodRatio`] (log Bayes factor) that
    /// `evidence` induces between two hypotheses, `numerator_id` over
    /// `denominator_id`.
    ///
    /// This is the log-odds view of the update: the returned ratio's
    /// [`log_ratio`](LikelihoodRatio::log_ratio) is exactly what an
    /// [`update`](Self::update) with this evidence adds to the log-odds of the
    /// numerator hypothesis against the denominator.
    ///
    /// # Errors
    ///
    /// - [`BeliefRevisionError::UnknownHypothesis`] if either id is not in
    ///   `state`.
    /// - [`BeliefRevisionError::MissingLikelihood`] (strict mode) if `evidence`
    ///   omits a likelihood for either hypothesis.
    /// - [`BeliefRevisionError::NonFiniteLikelihood`] / [`BeliefRevisionError::NegativeLikelihood`]
    ///   if a likelihood is `NaN`/infinite or negative.
    pub fn likelihood_ratio(
        &self,
        state: &BeliefState,
        evidence: &EvidenceUpdate,
        numerator_id: &str,
        denominator_id: &str,
    ) -> Result<LikelihoodRatio, BeliefRevisionError> {
        for id in [numerator_id, denominator_id] {
            if !state
                .hypotheses()
                .iter()
                .any(|hypothesis| hypothesis.id == id)
            {
                return Err(BeliefRevisionError::UnknownHypothesis { id: id.to_string() });
            }
        }

        let numerator_likelihood = self.resolve_likelihood(evidence, numerator_id)?;
        let denominator_likelihood = self.resolve_likelihood(evidence, denominator_id)?;

        Ok(LikelihoodRatio::from_likelihoods(
            numerator_id.to_string(),
            denominator_id.to_string(),
            numerator_likelihood,
            denominator_likelihood,
            self.config.min_likelihood,
        ))
    }

    /// Resolve and validate a single hypothesis's likelihood from an evidence
    /// update, honoring strict/lenient mode.
    fn resolve_likelihood(
        &self,
        evidence: &EvidenceUpdate,
        hypothesis_id: &str,
    ) -> Result<f64, BeliefRevisionError> {
        let likelihood = match evidence.likelihood_for(hypothesis_id) {
            Some(value) => value,
            None if self.config.strict => {
                return Err(BeliefRevisionError::MissingLikelihood {
                    id: hypothesis_id.to_string(),
                });
            }
            None => 1.0,
        };
        if !likelihood.is_finite() {
            return Err(BeliefRevisionError::NonFiniteLikelihood {
                id: hypothesis_id.to_string(),
            });
        }
        if likelihood < 0.0 {
            return Err(BeliefRevisionError::NegativeLikelihood {
                id: hypothesis_id.to_string(),
                value: likelihood,
            });
        }
        Ok(likelihood)
    }

    /// Run the full lifecycle: initialize from `hypotheses`/`prior`, apply the
    /// whole `evidence` sequence, and package the final posterior into a
    /// [`BeliefRevisionResult`].
    ///
    /// # Errors
    ///
    /// Any error from [`init`](Self::init) or [`update`](Self::update).
    pub fn run(
        &self,
        hypotheses: Vec<BeliefHypothesis>,
        prior: Option<&[f64]>,
        evidence: &[EvidenceUpdate],
    ) -> Result<BeliefRevisionResult, BeliefRevisionError> {
        let mut state = self.init(hypotheses, prior)?;
        self.update_sequence(&mut state, evidence)?;
        let most_likely = self.most_likely(&state)?;
        let entropy = state.entropy();
        let updates_applied = state.updates_applied();
        Ok(BeliefRevisionResult {
            state,
            most_likely,
            updates_applied,
            entropy,
        })
    }
}
