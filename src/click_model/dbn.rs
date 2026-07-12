//! The **Dynamic Bayesian Network** click model (Chapelle & Zhang, 2009) and
//! its Expectation-Maximisation fit.
//!
//! # The model
//!
//! The DBN is the cascade model with the one thing the cascade model gets
//! wrong: a click is not the end of the story. A user who clicks a result and
//! finds it *useless* comes back and keeps scanning. So the DBN splits
//! relevance in two:
//!
//! - **attractiveness** `α_d` — the probability the *snippet* earns a click,
//! - **satisfaction** `σ_d` — the probability the *landing page* ends the
//!   search,
//!
//! and adds one global **persistence** `γ`, the probability a still-unsatisfied
//! user bothers to look at the next position at all.
//!
//! For a document `d_r` at zero-based rank `r`, with `E_r` the examination
//! indicator (`E₀ = 1`: the top result is always examined):
//!
//! ```text
//! C_r = E_r ∧ A_r,        A_r ~ Bernoulli(α_{d_r})
//! S_r ~ Bernoulli(σ_{d_r}),  drawn only when C_r = 1
//! E_{r+1} = 1  ⟺  E_r = 1  ∧  ¬(C_r ∧ S_r)  ∧  Bernoulli(γ)
//! ```
//!
//! in words: *you can only click what you looked at; a satisfying click ends
//! the session; anything else, you continue with probability `γ`.*
//!
//! Two consequences make the inference tractable, and they are worth stating
//! before any algebra:
//!
//! - **Examination is monotone.** `E_r = 0 ⇒ E_j = 0` for all `j > r`. So a
//!   session is fully described by the last examined rank.
//! - **Everything above the last click is *known*.** If the deepest click is at
//!   rank `k`, then `E_r = 1` for every `r ≤ k` — with certainty, no inference
//!   required. And for `r < k` with a click, the user demonstrably *continued*,
//!   so `S_r = 0` and the persistence coin came up heads. **The only latent
//!   quantities in a DBN session are `S_k` (was the last click satisfying?) and
//!   the tail below `k`.**
//!
//! Unlike the PBM, the DBN needs no identifiability anchor: `E₀ = 1` is part of
//! the model, so rank 0 is a direct Bernoulli observation of `α_{d₀}` and the
//! scale is pinned by construction.
//!
//! # The backward recursion `W`
//!
//! For the tail — the positions at or below `k + 1`, all of which we observed
//! *not* clicked — define
//!
//! ```text
//! W_r = P(no clicks at ranks r … n−1 | E_r = 1)
//! ```
//!
//! Given `E_r = 1` and no click at `r`, the document must have been
//! unattractive (`1 − α_{d_r}`); with no click there is no satisfaction draw, so
//! the user continues with probability `γ` (and then must produce no clicks from
//! `r + 1` on, i.e. `W_{r+1}`) or abandons with probability `1 − γ` (and then
//! produces no clicks trivially). Hence
//!
//! ```text
//! W_n     = 1                                            (past the last rank)
//! W_r     = (1 − α_{d_r}) · [ γ · W_{r+1} + (1 − γ) ]     for r < n
//! ```
//!
//! Note `W_r ≥ (1 − α)·(1 − γ) ≥ floor²` for clamped parameters, so it never
//! underflows to zero and `ln W_r` is always finite — regardless of how long the
//! ranking is.
//!
//! # The forward recursion `f`
//!
//! Let `k` be the **last** clicked rank (`k = −1` when the session has no
//! click). Define `f_r` = the probability of *reaching* rank `r` in the
//! examined state, starting from the certainty at `k`, excluding everything
//! already accounted for above `k`:
//!
//! ```text
//! f_{k+1} = (1 − σ_{d_k}) · γ         if k ≥ 0   (unsatisfied by the last click, then continued)
//! f_0     = 1                          if k = −1  (E₀ = 1 by definition)
//! f_{r+1} = f_r · (1 − α_{d_r}) · γ    for r ≥ max(k + 1, 0)
//! ```
//!
//! # The session likelihood
//!
//! Split the session at `k`. Everything above `k` is deterministic given the
//! clicks; everything at or below `k + 1` is the `W` tail. With
//!
//! ```text
//! D = σ_{d_k} + (1 − σ_{d_k}) · [ γ · W_{k+1} + (1 − γ) ]      (k ≥ 0)
//! D = W_0                                                       (k = −1)
//! ```
//!
//! the observed-data likelihood of one session is
//!
//! ```text
//! L = [ Π_{r ≤ k} α_{d_r}^{c_r} (1 − α_{d_r})^{1 − c_r} ]      snippet draws, all examined
//!     · γ^k                                                     the user continued past every r < k
//!     · [ Π_{r < k, c_r = 1} (1 − σ_{d_r}) ]                    every click before the last one dissatisfied
//!     · D                                                       the last click's satisfaction, then the tail
//! ```
//!
//! and `L = W_0` when there is no click. Sanity check: when the last click is at
//! the *last* rank (`k = n − 1`), `W_{k+1} = W_n = 1`, so
//! `D = σ + (1 − σ)·[γ + 1 − γ] = 1` — a click at the bottom of the page tells
//! you **nothing** about satisfaction, because there was nothing left to
//! continue to. The model says so, exactly, and the tests assert it.
//!
//! # The E-step
//!
//! ```text
//! P(E_r = 1 | data) = 1                     for r ≤ k
//!                   = f_r · W_r / D          for r > k
//!
//! P(S_r = 1 | data) = 0                     for a click at r < k  (the user continued!)
//!                   = σ_{d_k} / D            for the last click, r = k
//!
//! P(A_r = 1 | data) = c_r                                for r ≤ k  (examined: the click IS the answer)
//!                   = (1 − P(E_r = 1 | data)) · α_{d_r}  for r > k
//! ```
//!
//! The last line is the DBN's version of the PBM's "a no-click deep down is
//! barely evidence": below the last click, `A_r` can only be `1` if the position
//! was never examined (otherwise it would have been clicked), and *conditional*
//! on never being examined `A_r` keeps its prior `α_{d_r}` exactly, because an
//! unexamined attractiveness draw influences nothing else in the session.
//!
//! # The M-step
//!
//! The complete-data log-likelihood separates into one Bernoulli term per
//! parameter, so every parameter is the mean of its posteriors:
//!
//! ```text
//! α_d ← mean of P(A_r = 1 | data)  over ALL impressions of d
//! σ_d ← mean of P(S_r = 1 | data)  over CLICKED impressions of d
//! γ   ← E[# persistence coins that came up heads] / E[# persistence coins flipped]
//! ```
//!
//! The persistence coin at rank `r` is flipped iff the user was there (`E_r = 1`)
//! and was not satisfied by a click there (`¬(C_r ∧ S_r)`) — and only for
//! `r < n − 1`, since the coin after the last rank of the page decides nothing
//! observable. It comes up heads iff `E_{r+1} = 1`. So
//!
//! ```text
//! heads ← Σ_{r=0}^{n−2} P(E_{r+1} = 1 | data)
//! flips ← Σ_{r=0}^{n−2} [ P(E_r = 1 | data) − c_r · P(S_r = 1 | data) ]
//! ```
//!
//! (using `E[1{flip} · E_{r+1}] = E[E_{r+1}]`, because `E_{r+1} = 1` already
//! implies the coin was flipped and won).

use std::collections::HashMap;

use super::types::{
    ClickLog, ClickModel, ClickModelConfig, ClickModelError, ClickModelFit, ClickModelResult,
    ClickSession, KahanSum, document_vocabulary, safe_ln,
};

/// The output of one DBN M-step: a full parameter set plus the per-document
/// clicked-impression counts (the sample size behind each satisfaction
/// estimate).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DbnParameters {
    /// `α_d`, indexed by position in the model's document vocabulary.
    pub attractiveness: Vec<f64>,
    /// `σ_d`, indexed the same way.
    pub satisfaction: Vec<f64>,
    /// The global persistence `γ`.
    pub persistence: f64,
    /// How many clicked impressions each document had.
    pub clicked_impressions: Vec<u64>,
}

/// The per-session posteriors produced by one DBN E-step. Exposed to the module
/// (not to the world) so the tests can check the recursions directly.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DbnSessionPosterior {
    /// `P(E_r = 1 | data)` for every rank.
    pub examined: Vec<f64>,
    /// `P(S_r = 1 | data)` for every rank; `0` at unclicked ranks (where `S` is
    /// not even drawn).
    pub satisfied: Vec<f64>,
    /// `P(A_r = 1 | data)` for every rank.
    pub attractive: Vec<f64>,
    /// This session's observed-data log-likelihood under the current parameters.
    pub log_likelihood: f64,
}

// ── DbnClickModel ────────────────────────────────────────────────────────────

/// The Dynamic Bayesian Network click model, fitted by Expectation-Maximisation.
///
/// See the module-level documentation of [`crate::click_model::dbn`] for the
/// full derivation of the forward (`f`) / backward (`W`) recursions, the E-step
/// posteriors, and the M-step.
///
/// ```
/// # #[cfg(feature = "click-model")]
/// # {
/// use oxirag::click_model::{
///     ClickLogSimulator, ClickModel, ClickModelConfig, ClickSimulationModel, DbnClickModel,
/// };
///
/// // Ground truth. "d0" is a classic clickbait snippet: highly attractive,
/// // rarely satisfying. "d1" is the opposite.
/// let true_attractiveness = vec![0.7, 0.35, 0.2];
/// let true_satisfaction = vec![0.2, 0.8, 0.5];
/// let true_persistence = 0.85;
///
/// let log = ClickLogSimulator::new(vec!["d0".to_owned(), "d1".to_owned(), "d2".to_owned()])
///     .with_attractiveness(true_attractiveness.clone())
///     .with_satisfaction(true_satisfaction.clone())
///     .with_persistence(true_persistence)
///     .with_slate_size(3)
///     .with_seed(7)
///     .simulate(ClickSimulationModel::Dbn, 8_000)
///     .expect("simulation succeeds");
///
/// let mut model = DbnClickModel::new(ClickModelConfig::default().with_max_iterations(300));
/// let fit = model.fit(&log).expect("fit succeeds");
/// assert!(fit.is_monotone(1e-9));
///
/// // All three parameter families are recovered from clicks alone.
/// for (index, (&alpha, &sigma)) in true_attractiveness
///     .iter()
///     .zip(true_satisfaction.iter())
///     .enumerate()
/// {
///     let doc = format!("d{index}");
///     let fitted_alpha = model.attractiveness(&doc).expect("known doc");
///     let fitted_sigma = model.satisfaction(&doc).expect("known doc");
///     assert!((fitted_alpha - alpha).abs() < 0.05, "α{index}: {fitted_alpha} vs {alpha}");
///     assert!((fitted_sigma - sigma).abs() < 0.05, "σ{index}: {fitted_sigma} vs {sigma}");
/// }
/// assert!((model.persistence() - true_persistence).abs() < 0.05);
///
/// // The DBN tells clickbait from quality: d0 wins on clicks, d1 on satisfaction.
/// assert!(model.attractiveness("d0").expect("known") > model.attractiveness("d1").expect("known"));
/// assert!(model.satisfaction("d0").expect("known") < model.satisfaction("d1").expect("known"));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DbnClickModel {
    config: ClickModelConfig,
    /// `α_d`, indexed by position in `doc_ids`.
    attractiveness: Vec<f64>,
    /// `σ_d`, indexed by position in `doc_ids`.
    satisfaction: Vec<f64>,
    /// The global persistence `γ`.
    persistence: f64,
    /// How many *clicked* impressions each document had — the sample size behind
    /// its satisfaction estimate.
    clicked_impressions: Vec<u64>,
    doc_ids: Vec<String>,
    doc_index: HashMap<String, usize>,
    /// Marginal examination probability at each rank, averaged over the rankings
    /// of the fitted log.
    examination_curve: Vec<f64>,
    fitted: bool,
}

impl DbnClickModel {
    /// A fresh, unfitted model.
    #[must_use]
    pub fn new(config: ClickModelConfig) -> Self {
        let persistence = config.clamp_parameter(config.initial_persistence);
        Self {
            config,
            attractiveness: Vec::new(),
            satisfaction: Vec::new(),
            persistence,
            clicked_impressions: Vec::new(),
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

    /// The fitted global persistence `γ` — the probability an unsatisfied user
    /// scans one position further.
    #[must_use]
    pub fn persistence(&self) -> f64 {
        self.persistence
    }

    /// The fitted attractiveness `α_d` (probability the *snippet* is clicked
    /// when examined).
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit, and
    /// [`ClickModelError::UnknownDocument`] for an unknown document.
    pub fn attractiveness(&self, doc_id: &str) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        Ok(self.attractiveness[self.lookup_doc(doc_id)?])
    }

    /// The fitted satisfaction `σ_d` (probability the *landing page* ends the
    /// session, given a click).
    ///
    /// A document that was never clicked carries **no** satisfaction
    /// information; its value is the prior
    /// ([`ClickModelConfig::initial_satisfaction`]) and it is named in
    /// [`DbnClickModel::unidentified_satisfaction_doc_ids`].
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit, and
    /// [`ClickModelError::UnknownDocument`] for an unknown document.
    pub fn satisfaction(&self, doc_id: &str) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        Ok(self.satisfaction[self.lookup_doc(doc_id)?])
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

    /// The documents that were never clicked, and whose satisfaction is
    /// therefore the prior rather than an estimate.
    #[must_use]
    pub fn unidentified_satisfaction_doc_ids(&self) -> Vec<&str> {
        self.doc_ids
            .iter()
            .zip(self.clicked_impressions.iter())
            .filter(|&(_, &count)| count == 0)
            .map(|(doc_id, _)| doc_id.as_str())
            .collect()
    }

    /// The DBN's exact **examination** probability at every position of a
    /// specific ranking: `e₀ = 1` and
    /// `e_{r+1} = e_r · γ · (1 − α_{d_r}·σ_{d_r})`.
    ///
    /// (Given examination, the user leaves for good only by clicking *and* being
    /// satisfied — probability `α·σ` — otherwise they continue with probability
    /// `γ`.)
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
        let mut probabilities = Vec::with_capacity(ranked_doc_ids.len());
        let mut examination = 1.0_f64;
        for doc_id in ranked_doc_ids {
            let doc = self.lookup_doc(doc_id)?;
            probabilities.push(examination);
            examination *=
                self.persistence * (1.0 - self.attractiveness[doc] * self.satisfaction[doc]);
        }
        Ok(probabilities)
    }

    /// The DBN's exact **click** probability at every position of a specific
    /// ranking: `e_r · α_{d_r}`.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit, and
    /// [`ClickModelError::UnknownDocument`] for any unknown document.
    pub fn session_click_probabilities(
        &self,
        ranked_doc_ids: &[String],
    ) -> ClickModelResult<Vec<f64>> {
        let examinations = self.session_examination_probabilities(ranked_doc_ids)?;
        ranked_doc_ids
            .iter()
            .zip(examinations)
            .map(|(doc_id, examination)| {
                let doc = self.lookup_doc(doc_id)?;
                Ok(examination * self.attractiveness[doc])
            })
            .collect()
    }

    fn ensure_fitted(&self) -> ClickModelResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(ClickModelError::NotFitted {
                model: "DbnClickModel",
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

    /// The backward recursion `W_r = P(no clicks at r..n−1 | E_r = 1)`.
    ///
    /// Returns `n + 1` values, with `W[n] = 1`. Only entries strictly below the
    /// last click are meaningful (above it, the "no clicks" premise is false);
    /// the E-step only ever reads those.
    pub(crate) fn backward_no_click(
        docs: &[usize],
        persistence: f64,
        attractiveness: &[f64],
    ) -> Vec<f64> {
        let n = docs.len();
        let mut w = vec![1.0_f64; n + 1];
        for rank in (0..n).rev() {
            let alpha = attractiveness[docs[rank]];
            w[rank] = (1.0 - alpha) * (persistence * w[rank + 1] + (1.0 - persistence));
        }
        w
    }

    /// The full E-step for one session: the `W` backward pass, the `f` forward
    /// pass, the three posterior vectors, and the session's log-likelihood.
    ///
    /// A pure function of the supplied parameters — it reads nothing from the
    /// model's own state, which is what lets the tests drive it with
    /// hand-computed `α`, `σ` and `γ` and check the recursions directly.
    pub(crate) fn session_posterior(
        session: &ClickSession,
        docs: &[usize],
        attractiveness: &[f64],
        satisfaction: &[f64],
        persistence: f64,
    ) -> DbnSessionPosterior {
        let n = docs.len();
        let w = Self::backward_no_click(docs, persistence, attractiveness);
        // `last_click` as a signed index: −1 when the session has no click at
        // all, which lets the "r > k" branches below cover every rank uniformly.
        let last_click: isize = session
            .last_click_rank()
            .map_or(-1, |rank| isize::try_from(rank).unwrap_or(isize::MAX));

        // ── the normaliser D, and the forward pass seed ──────────────────────
        let (denominator, mut forward, forward_start) = if last_click < 0 {
            // No click: E₀ = 1 with certainty, and the whole session is tail.
            (w[0], 1.0_f64, 0_usize)
        } else {
            #[allow(clippy::cast_sign_loss)]
            let k = last_click as usize;
            let sigma = satisfaction[docs[k]];
            let continued = persistence * w[k + 1] + (1.0 - persistence);
            let denominator = sigma + (1.0 - sigma) * continued;
            // Reaching rank k+1 requires: not satisfied by the click at k, then
            // a winning persistence coin.
            ((denominator), (1.0 - sigma) * persistence, k + 1)
        };

        let mut examined = vec![0.0_f64; n];
        let mut satisfied = vec![0.0_f64; n];
        let mut attractive = vec![0.0_f64; n];

        // Everything at or above the last click is known with certainty.
        for rank in 0..forward_start.min(n) {
            examined[rank] = 1.0;
            attractive[rank] = f64::from(u8::from(session.clicks[rank]));
        }
        if last_click >= 0 {
            #[allow(clippy::cast_sign_loss)]
            let k = last_click as usize;
            // The *last* click is the only one whose satisfaction is in doubt:
            // every earlier click was demonstrably followed by more browsing.
            satisfied[k] = satisfaction[docs[k]] / denominator;
        }

        // ── the tail: r > k ──────────────────────────────────────────────────
        for rank in forward_start..n {
            let examined_posterior = (forward * w[rank] / denominator).clamp(0.0, 1.0);
            examined[rank] = examined_posterior;
            // Below the last click every click is 0, so A_r = 1 is only possible
            // if the position was never examined — and then A keeps its prior.
            attractive[rank] = (1.0 - examined_posterior) * attractiveness[docs[rank]];
            forward *= (1.0 - attractiveness[docs[rank]]) * persistence;
        }

        DbnSessionPosterior {
            examined,
            satisfied,
            attractive,
            log_likelihood: Self::session_log_likelihood(
                session,
                docs,
                &w,
                last_click,
                attractiveness,
                satisfaction,
                persistence,
            ),
        }
    }

    /// The observed-data log-likelihood of one session — the `L` of the module
    /// docs, in logs.
    #[allow(clippy::too_many_arguments)]
    fn session_log_likelihood(
        session: &ClickSession,
        docs: &[usize],
        w: &[f64],
        last_click: isize,
        attractiveness: &[f64],
        satisfaction: &[f64],
        persistence: f64,
    ) -> f64 {
        if last_click < 0 {
            return safe_ln(w[0]);
        }
        #[allow(clippy::cast_sign_loss)]
        let k = last_click as usize;

        let mut total = 0.0_f64;
        for rank in 0..=k {
            let alpha = attractiveness[docs[rank]];
            total += if session.clicks[rank] {
                safe_ln(alpha)
            } else {
                safe_ln(1.0 - alpha)
            };
            if rank < k && session.clicks[rank] {
                // A click before the last one: the user was NOT satisfied,
                // otherwise they would have stopped.
                total += safe_ln(1.0 - satisfaction[docs[rank]]);
            }
        }
        // The persistence coin came up heads at every rank 0..k−1 — that is how
        // the user got as far as k in the first place.
        #[allow(clippy::cast_precision_loss)]
        let heads = k as f64;
        total += heads * safe_ln(persistence);

        let sigma = satisfaction[docs[k]];
        let continued = persistence * w[k + 1] + (1.0 - persistence);
        total += safe_ln(sigma + (1.0 - sigma) * continued);
        total
    }

    /// Map a session's document ids onto vocabulary indices.
    fn session_docs(&self, session: &ClickSession) -> ClickModelResult<Vec<usize>> {
        session
            .ranked_doc_ids
            .iter()
            .map(|doc_id| self.lookup_doc(doc_id))
            .collect()
    }

    /// The full log-likelihood of `log` under the supplied parameters.
    fn log_likelihood_with(
        &self,
        log: &ClickLog,
        attractiveness: &[f64],
        satisfaction: &[f64],
        persistence: f64,
    ) -> ClickModelResult<f64> {
        let mut total = KahanSum::new();
        for session in log.sessions() {
            let docs = self.session_docs(session)?;
            let w = Self::backward_no_click(&docs, persistence, attractiveness);
            let last_click: isize = session
                .last_click_rank()
                .map_or(-1, |rank| isize::try_from(rank).unwrap_or(isize::MAX));
            total.add(Self::session_log_likelihood(
                session,
                &docs,
                &w,
                last_click,
                attractiveness,
                satisfaction,
                persistence,
            ));
        }
        Ok(total.total())
    }

    /// One EM step: the E-step posteriors accumulated straight into the M-step's
    /// sufficient statistics, and the resulting parameters.
    fn em_step(&self, log: &ClickLog) -> ClickModelResult<DbnParameters> {
        let doc_count = self.doc_ids.len();
        let mut attractive_sum = vec![0.0_f64; doc_count];
        let mut attractive_count = vec![0_u64; doc_count];
        let mut satisfied_sum = vec![0.0_f64; doc_count];
        let mut clicked_count = vec![0_u64; doc_count];
        let mut persistence_heads = 0.0_f64;
        let mut persistence_flips = 0.0_f64;

        for session in log.sessions() {
            let docs = self.session_docs(session)?;
            let posterior = Self::session_posterior(
                session,
                &docs,
                &self.attractiveness,
                &self.satisfaction,
                self.persistence,
            );
            let n = docs.len();

            for (rank, &doc) in docs.iter().enumerate() {
                attractive_sum[doc] += posterior.attractive[rank];
                attractive_count[doc] += 1;
                if session.clicks[rank] {
                    satisfied_sum[doc] += posterior.satisfied[rank];
                    clicked_count[doc] += 1;
                }
            }

            // Persistence: one coin per rank that has a successor.
            for rank in 0..n.saturating_sub(1) {
                persistence_heads += posterior.examined[rank + 1];
                let satisfied_here = if session.clicks[rank] {
                    posterior.satisfied[rank]
                } else {
                    0.0
                };
                persistence_flips += posterior.examined[rank] - satisfied_here;
            }
        }

        let mut attractiveness = self.attractiveness.clone();
        let mut satisfaction = self.satisfaction.clone();
        for doc in 0..doc_count {
            if attractive_count[doc] > 0 {
                #[allow(clippy::cast_precision_loss)]
                let count = attractive_count[doc] as f64;
                attractiveness[doc] = self.config.clamp_parameter(attractive_sum[doc] / count);
            }
            if clicked_count[doc] > 0 {
                #[allow(clippy::cast_precision_loss)]
                let count = clicked_count[doc] as f64;
                satisfaction[doc] = self.config.clamp_parameter(satisfied_sum[doc] / count);
            }
            // A document with zero clicked impressions keeps the satisfaction
            // prior — there is no evidence, and inventing one would be a lie.
        }

        let persistence = if persistence_flips > 0.0 {
            self.config
                .clamp_parameter(persistence_heads / persistence_flips)
        } else {
            // Every session was a single-rank page: no persistence coin was ever
            // flipped, so γ keeps its prior.
            self.persistence
        };

        Ok(DbnParameters {
            attractiveness,
            satisfaction,
            persistence,
            clicked_impressions: clicked_count,
        })
    }

    /// The marginal examination probability per rank, averaged over the rankings
    /// of the fitted log.
    fn compute_examination_curve(&self, log: &ClickLog, depth: usize) -> Vec<f64> {
        let mut sums = vec![0.0_f64; depth];
        let mut counts = vec![0_u64; depth];
        for session in log.sessions() {
            let mut examination = 1.0_f64;
            for (rank, doc_id) in session.ranked_doc_ids.iter().enumerate() {
                sums[rank] += examination;
                counts[rank] += 1;
                let Some(&doc) = self.doc_index.get(doc_id.as_str()) else {
                    continue;
                };
                examination *=
                    self.persistence * (1.0 - self.attractiveness[doc] * self.satisfaction[doc]);
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

impl ClickModel for DbnClickModel {
    fn fit(&mut self, log: &ClickLog) -> ClickModelResult<ClickModelFit> {
        self.config.validate()?;
        if log.is_empty() {
            return Err(ClickModelError::EmptyLog);
        }

        let depth = log.depth();
        let (doc_ids, doc_index) = document_vocabulary(log);
        let doc_count = doc_ids.len();

        self.attractiveness = vec![
            self.config
                .clamp_parameter(self.config.initial_attractiveness);
            doc_count
        ];
        self.satisfaction = vec![
            self.config
                .clamp_parameter(self.config.initial_satisfaction);
            doc_count
        ];
        self.persistence = self.config.clamp_parameter(self.config.initial_persistence);
        self.clicked_impressions = vec![0; doc_count];
        self.doc_ids = doc_ids;
        self.doc_index = doc_index;

        let mut previous = self.log_likelihood_with(
            log,
            &self.attractiveness,
            &self.satisfaction,
            self.persistence,
        )?;
        let mut history = vec![previous];
        let mut iterations = 0_usize;
        let mut converged = false;

        for _ in 0..self.config.max_iterations {
            let parameters = self.em_step(log)?;
            self.attractiveness = parameters.attractiveness;
            self.satisfaction = parameters.satisfaction;
            self.persistence = parameters.persistence;
            self.clicked_impressions = parameters.clicked_impressions;
            iterations += 1;

            let current = self.log_likelihood_with(
                log,
                &self.attractiveness,
                &self.satisfaction,
                self.persistence,
            )?;
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

    /// The DBN's marginal click probability for `doc_id` at `rank`:
    /// `α_doc · Ê(rank)`, where `Ê(rank)` is the model's examination probability
    /// at that rank **averaged over the rankings in the fitted log** — because,
    /// exactly as in the cascade model, DBN examination depends on which
    /// documents sit above. When you know the ranking, use
    /// [`DbnClickModel::session_click_probabilities`], which is exact.
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
        Ok(self.attractiveness[self.lookup_doc(doc_id)?] * examination)
    }

    fn log_likelihood(&self, log: &ClickLog) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        self.log_likelihood_with(
            log,
            &self.attractiveness,
            &self.satisfaction,
            self.persistence,
        )
    }

    fn examination_curve(&self) -> &[f64] {
        &self.examination_curve
    }

    fn model_name(&self) -> &'static str {
        "DbnClickModel"
    }
}
