//! The **cascade model** (Craswell, Zoeter, Taylor & Ramsey, 2008) — a click
//! model with *no free examination parameters at all*, because examination is
//! entirely determined by what happened above.
//!
//! # The model
//!
//! The user scans the result page strictly top-down. At each position they
//! examine the document; with probability `α_d` they find it attractive and
//! click; and **the moment they click, they leave**. So examination is not a
//! per-rank coin flip (as in the PBM) — it is a *deterministic consequence* of
//! the ranking and of the clicks above:
//!
//! ```text
//! P(click at rank r) = α_{d_r} · Π_{j < r} (1 − α_{d_j})
//! ```
//!
//! Two consequences follow immediately, and they drive the whole
//! implementation:
//!
//! 1. **A cascade session has at most one click.** A log with multi-click
//!    sessions is not *impossible* to consume — it is simply outside the
//!    model's support, and this implementation handles it by using only the
//!    prefix up to and including the first click (see below), which is exactly
//!    the part of the session the model can speak about.
//! 2. **Positions strictly below the first click were never examined.** They
//!    contribute *nothing* — no click evidence, no no-click evidence, no
//!    likelihood term. Including them would be a bug: it would count a
//!    never-seen document as "shown and rejected" and drive its attractiveness
//!    to zero. [`CascadeClickModel`] excludes them, and the module's tests
//!    assert directly that appending arbitrary documents below the first click
//!    changes neither the log-likelihood nor a single fitted parameter.
//!
//! # The examined set
//!
//! An impression at rank `r` in a session is **examined** iff
//!
//! ```text
//! r ≤ (rank of the first click)      … if the session has any click
//! always                             … if the session has no click at all
//! ```
//!
//! (The first click's own position *is* examined — it is where the click
//! happened.)
//!
//! # Fitting: EM that finishes in one step
//!
//! Restricted to the examined set, the cascade likelihood is
//!
//! ```text
//! L = Π_sessions Π_{r examined} α_{d_r}^{c_r} · (1 − α_{d_r})^{1 − c_r}
//! ```
//!
//! — a product of plain Bernoulli terms. The examined set depends only on the
//! **observed** clicks, never on the parameters, so *the cascade model has no
//! latent variables to integrate out*. Its "E-step" is trivial (the examined
//! set is already known), and the M-step is the closed-form MLE
//!
//! ```text
//! α_d = (clicks on d) / (examined impressions of d).
//! ```
//!
//! This implementation still runs the same iterate-and-trace loop as the other
//! two models — the first iteration jumps straight to the global maximum and
//! every subsequent one is a fixed point — so that the log-likelihood trace,
//! the monotonicity guarantee, and the [`ClickModelFit`] report all mean the
//! same thing across all three models.
//!
//! # Documents that were never examined
//!
//! A document that only ever appeared *below* a first click has zero examined
//! impressions and therefore **no information at all** about its
//! attractiveness. Rather than let it silently keep whatever the arithmetic
//! produced (`0/0`), this model leaves it at
//! [`ClickModelConfig::initial_attractiveness`] and names it in
//! [`CascadeClickModel::unidentified_doc_ids`]. That is the documented
//! fallback: the value is the prior, and the model says so out loud.

use std::collections::HashMap;

use super::types::{
    ClickLog, ClickModel, ClickModelConfig, ClickModelError, ClickModelFit, ClickModelResult,
    ClickSession, KahanSum, document_vocabulary, safe_ln,
};

/// The number of ranks in `session` that the cascade model considers examined:
/// everything down to and including the first click, or the whole list when
/// there is no click.
///
/// This is the single rule the entire module hangs on, so it is a named,
/// separately-testable function rather than an inline expression.
#[must_use]
pub fn cascade_examined_depth(session: &ClickSession) -> usize {
    match session.first_click_rank() {
        Some(rank) => rank + 1,
        None => session.len(),
    }
}

// ── CascadeClickModel ────────────────────────────────────────────────────────

/// The cascade model, fitted by maximum likelihood over the examined prefix of
/// every session.
///
/// ```
/// # #[cfg(feature = "click-model")]
/// # {
/// use oxirag::click_model::{
///     CascadeClickModel, ClickLog, ClickModel, ClickModelConfig, ClickSession,
/// };
///
/// // Every session shows the same three documents. "top" is clicked in half
/// // the sessions; whenever it is, the user leaves and never sees "deep".
/// let mut sessions = Vec::new();
/// for index in 0..100 {
///     let docs = vec!["top".to_owned(), "middle".to_owned(), "deep".to_owned()];
///     let top_clicked = index % 2 == 0;
///     // Of the 50 sessions that get past "top", 20 click "middle".
///     let middle_clicked = !top_clicked && index % 5 == 1;
///     sessions.push(ClickSession::new("q", docs, vec![top_clicked, middle_clicked, false]));
/// }
/// let log = ClickLog::from_sessions(sessions).expect("valid log");
///
/// let mut model = CascadeClickModel::new(ClickModelConfig::default());
/// let fit = model.fit(&log).expect("fit succeeds");
/// assert!(fit.is_monotone(1e-9));
///
/// // "top" is examined in all 100 sessions and clicked in 50 → α = 0.5.
/// let top = model.attractiveness("top").expect("known document");
/// assert!((top - 0.5).abs() < 1e-9, "top α = {top}");
///
/// // "middle" is examined only in the 50 sessions where "top" was NOT clicked,
/// // and is clicked in 10 of them → α = 0.2. A naive click-through rate over
/// // all 100 impressions would have said 0.1 — exactly half the truth, because
/// // it counted 50 impressions the user never even saw.
/// let middle = model.attractiveness("middle").expect("known document");
/// assert!((middle - 0.2).abs() < 1e-9, "middle α = {middle}");
///
/// // The exact cascade prediction for the logged ranking.
/// let probs = model
///     .session_click_probabilities(&["top".to_owned(), "middle".to_owned()])
///     .expect("known documents");
/// assert!((probs[0] - 0.5).abs() < 1e-9);          // α_top
/// assert!((probs[1] - 0.5 * 0.2).abs() < 1e-9);    // (1 − α_top) · α_middle
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CascadeClickModel {
    config: ClickModelConfig,
    /// `α_d`, indexed by position in `doc_ids`.
    attractiveness: Vec<f64>,
    /// How many *examined* impressions each document had in the fitted log.
    examined_impressions: Vec<u64>,
    doc_ids: Vec<String>,
    doc_index: HashMap<String, usize>,
    /// The marginal examination probability at each rank, averaged over the
    /// rankings of the log this model was fitted on.
    examination_curve: Vec<f64>,
    fitted: bool,
}

impl CascadeClickModel {
    /// A fresh, unfitted model.
    #[must_use]
    pub fn new(config: ClickModelConfig) -> Self {
        Self {
            config,
            attractiveness: Vec::new(),
            examined_impressions: Vec::new(),
            doc_ids: Vec::new(),
            doc_index: HashMap::new(),
            examination_curve: Vec::new(),
            fitted: false,
        }
    }

    /// The configuration this model fits with.
    #[must_use]
    pub fn config(&self) -> &ClickModelConfig {
        &self.config
    }

    /// The sorted document vocabulary of the fitted log.
    #[must_use]
    pub fn doc_ids(&self) -> &[String] {
        &self.doc_ids
    }

    /// The fitted attractiveness `α_d` of one document.
    ///
    /// For a document with zero examined impressions this is the *prior*
    /// ([`ClickModelConfig::initial_attractiveness`]), not an estimate — see
    /// [`CascadeClickModel::unidentified_doc_ids`].
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit, and
    /// [`ClickModelError::UnknownDocument`] for a document the fit never saw.
    pub fn attractiveness(&self, doc_id: &str) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        Ok(self.attractiveness[self.lookup_doc(doc_id)?])
    }

    /// Every fitted attractiveness, keyed by document id.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit.
    pub fn attractiveness_map(&self) -> ClickModelResult<HashMap<String, f64>> {
        self.ensure_fitted()?;
        Ok(self
            .doc_ids
            .iter()
            .cloned()
            .zip(self.attractiveness.iter().copied())
            .collect())
    }

    /// How many examined impressions a document had — the effective sample size
    /// behind its attractiveness estimate. Zero means the estimate is the prior.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit, and
    /// [`ClickModelError::UnknownDocument`] for an unknown document.
    pub fn examined_impressions(&self, doc_id: &str) -> ClickModelResult<u64> {
        self.ensure_fitted()?;
        Ok(self.examined_impressions[self.lookup_doc(doc_id)?])
    }

    /// The documents that were shown but **never examined** — every one of their
    /// impressions sat below a first click. Their attractiveness is the prior,
    /// because the log contains literally zero information about them.
    #[must_use]
    pub fn unidentified_doc_ids(&self) -> Vec<&str> {
        self.doc_ids
            .iter()
            .zip(self.examined_impressions.iter())
            .filter(|&(_, &count)| count == 0)
            .map(|(doc_id, _)| doc_id.as_str())
            .collect()
    }

    /// The exact cascade click probability of every position of a *specific*
    /// ranking: `α_{d_r} · Π_{j<r} (1 − α_{d_j})`.
    ///
    /// This — not [`ClickModel::predict_click_prob`] — is the cascade model's
    /// real prediction, because in a cascade the click probability at a rank
    /// depends on *which* documents sit above it, not merely on how deep it is.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit, and
    /// [`ClickModelError::UnknownDocument`] for any unknown document.
    pub fn session_click_probabilities(
        &self,
        ranked_doc_ids: &[String],
    ) -> ClickModelResult<Vec<f64>> {
        self.ensure_fitted()?;
        let mut survival = 1.0_f64;
        let mut probabilities = Vec::with_capacity(ranked_doc_ids.len());
        for doc_id in ranked_doc_ids {
            let alpha = self.attractiveness[self.lookup_doc(doc_id)?];
            probabilities.push(survival * alpha);
            survival *= 1.0 - alpha;
        }
        Ok(probabilities)
    }

    /// The exact cascade *examination* probability of every position of a
    /// specific ranking: `Π_{j<r} (1 − α_{d_j})` (so position 0 is always `1`).
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit, and
    /// [`ClickModelError::UnknownDocument`] for any unknown document.
    pub fn session_examination_probabilities(
        &self,
        ranked_doc_ids: &[String],
    ) -> ClickModelResult<Vec<f64>> {
        self.ensure_fitted()?;
        let mut survival = 1.0_f64;
        let mut probabilities = Vec::with_capacity(ranked_doc_ids.len());
        for doc_id in ranked_doc_ids {
            probabilities.push(survival);
            survival *= 1.0 - self.attractiveness[self.lookup_doc(doc_id)?];
        }
        Ok(probabilities)
    }

    fn ensure_fitted(&self) -> ClickModelResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(ClickModelError::NotFitted {
                model: "CascadeClickModel",
            })
        }
    }

    fn lookup_doc(&self, doc_id: &str) -> ClickModelResult<usize> {
        self.doc_index
            .get(doc_id)
            .copied()
            .ok_or_else(|| ClickModelError::UnknownDocument {
                doc_id: doc_id.to_owned(),
            })
    }

    /// The cascade log-likelihood of `log` under `attractiveness`, summing
    /// **only over examined impressions**.
    fn log_likelihood_with(&self, log: &ClickLog, attractiveness: &[f64]) -> ClickModelResult<f64> {
        let mut total = KahanSum::new();
        for session in log.sessions() {
            let examined_depth = cascade_examined_depth(session);
            for rank in 0..examined_depth {
                let alpha = attractiveness[self.lookup_doc(&session.ranked_doc_ids[rank])?];
                total.add(if session.clicks[rank] {
                    safe_ln(alpha)
                } else {
                    safe_ln(1.0 - alpha)
                });
            }
        }
        Ok(total.total())
    }

    /// The closed-form M-step: `α_d = clicks(d) / examined(d)` over the examined
    /// set. Returns the new parameters and the per-document examined counts.
    fn maximisation_step(&self, log: &ClickLog) -> (Vec<f64>, Vec<u64>) {
        let doc_count = self.doc_ids.len();
        let mut clicks = vec![0_u64; doc_count];
        let mut examined = vec![0_u64; doc_count];

        for session in log.sessions() {
            let examined_depth = cascade_examined_depth(session);
            for rank in 0..examined_depth {
                let Some(&doc) = self.doc_index.get(session.ranked_doc_ids[rank].as_str()) else {
                    continue;
                };
                examined[doc] += 1;
                if session.clicks[rank] {
                    clicks[doc] += 1;
                }
            }
        }

        let mut attractiveness = self.attractiveness.clone();
        for doc in 0..doc_count {
            if examined[doc] == 0 {
                // Zero information: keep the prior and report it via
                // `unidentified_doc_ids`. Never `0 / 0`.
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let rate = clicks[doc] as f64 / examined[doc] as f64;
            attractiveness[doc] = self.config.clamp_parameter(rate);
        }
        (attractiveness, examined)
    }

    /// The marginal examination probability at each rank, averaged over the
    /// rankings actually present in the fitted log.
    fn compute_examination_curve(&self, log: &ClickLog, depth: usize) -> Vec<f64> {
        let mut sums = vec![0.0_f64; depth];
        let mut counts = vec![0_u64; depth];
        for session in log.sessions() {
            let mut survival = 1.0_f64;
            for (rank, doc_id) in session.ranked_doc_ids.iter().enumerate() {
                sums[rank] += survival;
                counts[rank] += 1;
                let Some(&doc) = self.doc_index.get(doc_id.as_str()) else {
                    continue;
                };
                survival *= 1.0 - self.attractiveness[doc];
            }
        }
        (0..depth)
            .map(|rank| {
                if counts[rank] == 0 {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let count = counts[rank] as f64;
                    sums[rank] / count
                }
            })
            .collect()
    }
}

impl ClickModel for CascadeClickModel {
    fn fit(&mut self, log: &ClickLog) -> ClickModelResult<ClickModelFit> {
        self.config.validate()?;
        if log.is_empty() {
            return Err(ClickModelError::EmptyLog);
        }

        let depth = log.depth();
        let (doc_ids, doc_index) = document_vocabulary(log);
        self.attractiveness = vec![
            self.config
                .clamp_parameter(self.config.initial_attractiveness);
            doc_ids.len()
        ];
        self.examined_impressions = vec![0; doc_ids.len()];
        self.doc_ids = doc_ids;
        self.doc_index = doc_index;

        let mut previous = self.log_likelihood_with(log, &self.attractiveness)?;
        let mut history = vec![previous];
        let mut iterations = 0_usize;
        let mut converged = false;

        for _ in 0..self.config.max_iterations {
            let (attractiveness, examined) = self.maximisation_step(log);
            self.attractiveness = attractiveness;
            self.examined_impressions = examined;
            iterations += 1;

            let current = self.log_likelihood_with(log, &self.attractiveness)?;
            history.push(current);
            let delta = current - previous;
            previous = current;
            if delta.abs() < self.config.tolerance {
                converged = true;
                break;
            }
        }

        self.fitted = true;
        self.examination_curve = self.compute_examination_curve(log, depth);

        Ok(ClickModelFit {
            iterations,
            converged,
            final_log_likelihood: previous,
            log_likelihood_history: history,
        })
    }

    /// The cascade model's marginal click probability for `doc_id` at `rank`:
    /// `α_doc · Ê(rank)`, where `Ê(rank)` is the model's marginal examination
    /// probability at that rank **averaged over the rankings in the fitted
    /// log**.
    ///
    /// The averaging is unavoidable and is stated here rather than hidden: in a
    /// cascade, whether rank `r` is examined depends on the *specific*
    /// documents above it, so there is no such thing as "the" click probability
    /// at a rank without naming them. When you know the ranking, use
    /// [`CascadeClickModel::session_click_probabilities`], which is exact.
    fn predict_click_prob(&self, doc_id: &str, rank: usize) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        let examination =
            *self
                .examination_curve
                .get(rank)
                .ok_or(ClickModelError::UnknownRank {
                    rank,
                    depth: self.examination_curve.len(),
                })?;
        let alpha = self.attractiveness[self.lookup_doc(doc_id)?];
        Ok(alpha * examination)
    }

    fn log_likelihood(&self, log: &ClickLog) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        self.log_likelihood_with(log, &self.attractiveness)
    }

    fn examination_curve(&self) -> &[f64] {
        &self.examination_curve
    }

    fn model_name(&self) -> &'static str {
        "CascadeClickModel"
    }
}
