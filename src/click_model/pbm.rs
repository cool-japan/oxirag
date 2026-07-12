//! The **Position-Based Model** (PBM) and its Expectation-Maximisation fit.
//!
//! # The model
//!
//! The PBM is the simplest model that takes the *examination hypothesis*
//! seriously. For a document `d` shown at zero-based rank `r`, it posits two
//! independent latent Bernoulli variables:
//!
//! ```text
//! E ~ Bernoulli(γ_r)     "the user examined position r"   (depends only on the rank)
//! A ~ Bernoulli(α_d)     "the document is attractive"     (depends only on the document)
//! C = E ∧ A              "a click happens iff both"
//! ```
//!
//! so the only thing ever *observed*, the click, has
//!
//! ```text
//! P(C = 1 | d, r) = γ_r · α_d.
//! ```
//!
//! `γ_r` is the **propensity** — the probability that an item at rank `r` even
//! gets a chance to be clicked. `α_d` is the **attractiveness** — the
//! position-free relevance signal we actually want. A raw click-through rate
//! conflates the two; the PBM's whole job is to separate them.
//!
//! # Identifiability: why `γ₀ = 1` is not optional
//!
//! The likelihood only ever sees the *product* `γ_r · α_d`. Scale every
//! examination probability up by `c` and every attractiveness down by `c` and
//! the likelihood is **bit-for-bit identical**. The PBM is therefore identified
//! only up to that one global scale, and no amount of data fixes it. The
//! standard remedy — used here, controlled by
//! [`ClickModelConfig::anchor_top_examination`] — is to *anchor* the scale by
//! declaring `γ₀ = 1`: the top result is always examined. With one parameter
//! pinned, the scale `c` is forced to `1` and everything else is identified.
//!
//! A second, subtler identifiability question survives the anchor: the products
//! only connect a rank to a document if that `(rank, document)` cell was
//! actually observed. Ranks and documents therefore form a bipartite graph, and
//! the anchor only pins the scale of the connected **component** that contains
//! rank `0`. Any other component floats freely. [`PositionBasedModel`] computes
//! this with a union-find during fitting and reports it as
//! [`PbmIdentifiability`] — so a caller can find out, honestly, which of their
//! parameters the data actually determined.
//!
//! # The E-step, derived
//!
//! Let `p = γ_r · α_d`. The complete data is `(E, A)`; the observed data is
//! `C`. EM needs `P(E = 1 | C)` and `P(A = 1 | C)` under the current
//! parameters.
//!
//! **Clicked (`C = 1`).** `C = E ∧ A`, so a click forces both:
//!
//! ```text
//! P(E = 1 | C = 1) = 1,        P(A = 1 | C = 1) = 1.
//! ```
//!
//! **Not clicked (`C = 0`).** Now `(E, A) ≠ (1, 1)`, and the remaining three
//! configurations keep their prior mass, renormalised by
//! `P(C = 0) = 1 − γ_r·α_d`. By Bayes:
//!
//! ```text
//! P(E=1, A=0 | C=0) = P(C=0 | E=1,A=0)·P(E=1)·P(A=0) / P(C=0)
//!                   = 1 · γ_r · (1 − α_d) / (1 − γ_r·α_d)
//!
//! P(E=0, A=1 | C=0) = (1 − γ_r) · α_d       / (1 − γ_r·α_d)
//!
//! P(E=0, A=0 | C=0) = (1 − γ_r) · (1 − α_d) / (1 − γ_r·α_d)
//!
//! P(E=1, A=1 | C=0) = 0                     (it would have produced a click)
//! ```
//!
//! (The three add to `[γ_r − γ_r·α_d + α_d − γ_r·α_d + 1 − α_d − γ_r + γ_r·α_d]
//! / (1 − γ_r·α_d) = 1` — as they must.) Marginalising:
//!
//! ```text
//! P(E = 1 | C = 0) = γ_r · (1 − α_d) / (1 − γ_r · α_d)
//! P(A = 1 | C = 0) = α_d · (1 − γ_r) / (1 − γ_r · α_d)
//! ```
//!
//! Both are in `[0, 1]`: e.g. `γ(1−α) ≤ 1 − γα ⟺ γ ≤ 1`. Read them out loud and
//! they are exactly the intuition: *a no-click at the top of the page (`γ_r`
//! near 1) is strong evidence the document was unattractive; a no-click at the
//! bottom (`γ_r` near 0) is barely evidence of anything, and the document keeps
//! most of its prior attractiveness.*
//!
//! # The M-step, derived
//!
//! The complete-data log-likelihood factorises completely (`C` is a
//! deterministic function of `E` and `A`, contributing no parameter-dependent
//! term):
//!
//! ```text
//! Q(θ | θ_t) = Σ_impressions [ Ê·ln γ_r + (1−Ê)·ln(1−γ_r)
//!                            + Â·ln α_d + (1−Â)·ln(1−α_d) ]
//! ```
//!
//! with `Ê = P(E=1 | C)` and `Â = P(A=1 | C)` from the E-step. This is a sum of
//! independent Bernoulli log-likelihoods, one per parameter, so it maximises
//! coordinate-wise at the **posterior mean**:
//!
//! ```text
//! γ_r ← mean of Ê over every impression at rank r
//! α_d ← mean of Â over every impression of document d
//! ```
//!
//! Note that `Â` is averaged over **all** impressions of `d`, including the
//! ones the posterior says were probably never examined — those contribute
//! `P(E=0 | C=0)·α_d`, i.e. they hand the document back (a fraction of) its own
//! prior. That is not a fudge; it is what the derivation says, and it is exactly
//! why the PBM does not punish a good document for sitting at rank 9.

use std::collections::HashMap;

use super::types::{
    ClickImpression, ClickLog, ClickModel, ClickModelConfig, ClickModelError, ClickModelFit,
    ClickModelResult, KahanSum, document_vocabulary, safe_ln,
};

// ── PbmIdentifiability ───────────────────────────────────────────────────────

/// An honest report of which PBM parameters the fitted log actually determined.
///
/// The PBM's likelihood depends on `(γ_r, α_d)` only through their products, so
/// the parameters are identified only up to one global scale *per connected
/// component* of the bipartite `rank ↔ document` co-occurrence graph. The
/// `γ₀ = 1` anchor fixes the scale of the component containing rank `0`; every
/// other component keeps a free scale, and its fitted values are whatever the EM
/// initialisation happened to lead to. They still reproduce the observed
/// click-through rates exactly — the *products* are right — but the split of a
/// product into "examination" and "attractiveness" is arbitrary there.
///
/// The canonical way to land in this situation is a document that only ever
/// appeared at a single rank, at which no other document ever appeared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PbmIdentifiability {
    /// Whether the `γ₀ = 1` anchor was enabled at all. If it was not, *nothing*
    /// is identified (the whole model floats on one free scale).
    pub anchored: bool,
    /// `true` iff every rank and every document lies in the anchored component.
    pub fully_identified: bool,
    /// The number of connected components in the `rank ↔ document` graph.
    pub component_count: usize,
    /// Ranks whose `γ_r` is *not* pinned by the anchor, sorted ascending.
    pub unanchored_ranks: Vec<usize>,
    /// Documents whose `α_d` is *not* pinned by the anchor, sorted.
    pub unanchored_doc_ids: Vec<String>,
}

impl PbmIdentifiability {
    /// A one-line, human-readable summary — handy to log after a fit.
    #[must_use]
    pub fn summary(&self) -> String {
        if !self.anchored {
            return "unanchored PBM: γ and α are determined only up to one global scale".to_owned();
        }
        if self.fully_identified {
            return format!(
                "fully identified: all ranks and documents are in the anchored component ({} component)",
                self.component_count
            );
        }
        format!(
            "partially identified: {} components; {} rank(s) and {} document(s) lie outside the anchored component and keep a free scale",
            self.component_count,
            self.unanchored_ranks.len(),
            self.unanchored_doc_ids.len()
        )
    }
}

/// Union-find `find` with path halving, over the bipartite `rank ↔ document`
/// co-occurrence graph used by [`PositionBasedModel::identifiability`].
fn find_root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]]; // path halving
        node = parent[node];
    }
    node
}

// ── PositionBasedModel ───────────────────────────────────────────────────────

/// The Position-Based Model, fitted by Expectation-Maximisation.
///
/// See the module-level documentation of [`crate::click_model::pbm`] for the
/// full E-step / M-step derivation.
///
/// ```
/// # #[cfg(feature = "click-model")]
/// # {
/// use oxirag::click_model::{
///     ClickLogSimulator, ClickModel, ClickModelConfig, ClickSimulationModel,
///     PositionBasedModel,
/// };
///
/// // Ground truth: a steep examination curve and four documents of very
/// // different quality. The simulator draws clicks from exactly the PBM this
/// // model assumes, so EM should be able to invert it.
/// let true_examination = vec![1.0, 0.65, 0.4, 0.22];
/// let true_attractiveness = vec![0.75, 0.5, 0.3, 0.1];
/// let log = ClickLogSimulator::new(vec![
///         "d0".to_owned(), "d1".to_owned(), "d2".to_owned(), "d3".to_owned(),
///     ])
///     .with_attractiveness(true_attractiveness.clone())
///     .with_examination(true_examination.clone())
///     .with_slate_size(4)
///     .with_seed(20_240_711)
///     .simulate(ClickSimulationModel::PositionBased, 6_000)
///     .expect("simulation succeeds");
///
/// let mut model = PositionBasedModel::new(
///     ClickModelConfig::default().with_max_iterations(200),
/// );
/// let fit = model.fit(&log).expect("fit succeeds");
///
/// // EM never went downhill — that is the correctness invariant.
/// assert!(fit.is_monotone(1e-9));
/// // The `γ₀ = 1` anchor is what makes the scale identifiable at all.
/// assert_eq!(model.examination_curve()[0], 1.0);
///
/// // Both latent curves are recovered from clicks alone.
/// for (rank, &truth) in true_examination.iter().enumerate() {
///     let fitted = model.examination_curve()[rank];
///     assert!((fitted - truth).abs() < 0.05, "γ{rank}: {fitted} vs {truth}");
/// }
/// for (index, &truth) in true_attractiveness.iter().enumerate() {
///     let fitted = model.attractiveness(&format!("d{index}")).expect("known doc");
///     assert!((fitted - truth).abs() < 0.05, "α{index}: {fitted} vs {truth}");
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PositionBasedModel {
    config: ClickModelConfig,
    /// `γ_r`, indexed by zero-based rank.
    examination: Vec<f64>,
    /// `α_d`, indexed by position in `doc_ids`.
    attractiveness: Vec<f64>,
    /// The sorted document vocabulary of the log this model was fitted on.
    doc_ids: Vec<String>,
    /// Reverse index into `doc_ids`.
    doc_index: HashMap<String, usize>,
    identifiability: Option<PbmIdentifiability>,
    fitted: bool,
}

impl PositionBasedModel {
    /// A fresh, unfitted model.
    #[must_use]
    pub fn new(config: ClickModelConfig) -> Self {
        Self {
            config,
            examination: Vec::new(),
            attractiveness: Vec::new(),
            doc_ids: Vec::new(),
            doc_index: HashMap::new(),
            identifiability: None,
            fitted: false,
        }
    }

    /// The configuration this model fits with.
    #[must_use]
    pub fn config(&self) -> &ClickModelConfig {
        &self.config
    }

    /// The fitted examination probabilities `γ_r`, indexed by zero-based rank.
    /// These are the raw, **unclipped** propensities — see
    /// [`crate::click_model::PropensityEstimates`] for the clipped, invertible
    /// view.
    #[must_use]
    pub fn examination(&self) -> &[f64] {
        &self.examination
    }

    /// The sorted document vocabulary of the fitted log.
    #[must_use]
    pub fn doc_ids(&self) -> &[String] {
        &self.doc_ids
    }

    /// The fitted attractiveness `α_d` of one document.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit, and
    /// [`ClickModelError::UnknownDocument`] for a document the fit never saw.
    pub fn attractiveness(&self, doc_id: &str) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        let index = self.lookup_doc(doc_id)?;
        Ok(self.attractiveness[index])
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

    /// What the data actually determined — see [`PbmIdentifiability`].
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] before a fit.
    pub fn identifiability(&self) -> ClickModelResult<&PbmIdentifiability> {
        self.identifiability
            .as_ref()
            .ok_or(ClickModelError::NotFitted {
                model: "PositionBasedModel",
            })
    }

    fn ensure_fitted(&self) -> ClickModelResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(ClickModelError::NotFitted {
                model: "PositionBasedModel",
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

    fn lookup_rank(&self, rank: usize) -> ClickModelResult<f64> {
        self.examination
            .get(rank)
            .copied()
            .ok_or(ClickModelError::UnknownRank {
                rank,
                depth: self.examination.len(),
            })
    }

    /// The log-likelihood of `log` under the *supplied* parameter vectors —
    /// the workhorse behind both [`ClickModel::log_likelihood`] and the EM
    /// loop's monotonicity trace.
    fn log_likelihood_with(
        &self,
        log: &ClickLog,
        examination: &[f64],
        attractiveness: &[f64],
    ) -> ClickModelResult<f64> {
        let mut total = KahanSum::new();
        for impression in log.impressions() {
            let ClickImpression {
                doc_id,
                rank,
                clicked,
                ..
            } = impression;
            let gamma = *examination.get(rank).ok_or(ClickModelError::UnknownRank {
                rank,
                depth: examination.len(),
            })?;
            let doc = self.lookup_doc(doc_id)?;
            let alpha = attractiveness[doc];
            let click_prob = gamma * alpha;
            total.add(if clicked {
                safe_ln(click_prob)
            } else {
                safe_ln(1.0 - click_prob)
            });
        }
        Ok(total.total())
    }

    /// One full EM step: E-step posteriors accumulated straight into the
    /// M-step's sufficient statistics.
    fn em_step(&self, log: &ClickLog) -> (Vec<f64>, Vec<f64>) {
        let depth = self.examination.len();
        let doc_count = self.attractiveness.len();

        // Σ Ê and count, per rank; Σ Â and count, per document.
        let mut examined_sum = vec![0.0_f64; depth];
        let mut examined_count = vec![0_u64; depth];
        let mut attractive_sum = vec![0.0_f64; doc_count];
        let mut attractive_count = vec![0_u64; doc_count];

        for impression in log.impressions() {
            let rank = impression.rank;
            let Some(&doc) = self.doc_index.get(impression.doc_id) else {
                continue;
            };
            let gamma = self.examination[rank];
            let alpha = self.attractiveness[doc];

            // ── E-step ───────────────────────────────────────────────────────
            let (examined_posterior, attractive_posterior) = if impression.clicked {
                // A click forces E = 1 and A = 1.
                (1.0, 1.0)
            } else {
                // Bayes over the three surviving configurations; the
                // denominator P(C = 0) = 1 − γ·α is bounded below by
                // `parameter_floor` because α ≤ 1 − floor and γ ≤ 1.
                let no_click = 1.0 - gamma * alpha;
                (
                    gamma * (1.0 - alpha) / no_click,
                    alpha * (1.0 - gamma) / no_click,
                )
            };

            examined_sum[rank] += examined_posterior;
            examined_count[rank] += 1;
            attractive_sum[doc] += attractive_posterior;
            attractive_count[doc] += 1;
        }

        // ── M-step: each parameter is the mean of its posteriors ─────────────
        let mut examination = self.examination.clone();
        for rank in 0..depth {
            if self.config.anchor_top_examination && rank == 0 {
                // Fixed at 1 by the identifiability anchor: its feasible set is
                // a single point, which the previous iterate already occupies,
                // so Q cannot decrease.
                continue;
            }
            if examined_count[rank] == 0 {
                continue; // no evidence: leave the parameter where it was.
            }
            #[allow(clippy::cast_precision_loss)]
            let count = examined_count[rank] as f64;
            examination[rank] = self.config.clamp_parameter(examined_sum[rank] / count);
        }

        let mut attractiveness = self.attractiveness.clone();
        for doc in 0..doc_count {
            if attractive_count[doc] == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let count = attractive_count[doc] as f64;
            attractiveness[doc] = self.config.clamp_parameter(attractive_sum[doc] / count);
        }

        (examination, attractiveness)
    }

    /// Union-find over the bipartite `rank ↔ document` co-occurrence graph, to
    /// determine which parameters the `γ₀ = 1` anchor actually pins.
    fn analyse_identifiability(&self, log: &ClickLog, depth: usize) -> PbmIdentifiability {
        let doc_count = self.doc_ids.len();
        // Nodes: `0..depth` are ranks, `depth..depth + doc_count` are documents.
        let mut parent: Vec<usize> = (0..depth + doc_count).collect();

        for impression in log.impressions() {
            let Some(&doc) = self.doc_index.get(impression.doc_id) else {
                continue;
            };
            let rank_root = find_root(&mut parent, impression.rank);
            let doc_root = find_root(&mut parent, depth + doc);
            if rank_root != doc_root {
                parent[doc_root] = rank_root;
            }
        }

        let mut roots: Vec<usize> = (0..depth + doc_count)
            .map(|node| find_root(&mut parent, node))
            .collect();
        roots.sort_unstable();
        roots.dedup();
        let component_count = roots.len();

        let anchored = self.config.anchor_top_examination;
        if !anchored || depth == 0 {
            return PbmIdentifiability {
                anchored,
                fully_identified: false,
                component_count,
                unanchored_ranks: (0..depth).collect(),
                unanchored_doc_ids: self.doc_ids.clone(),
            };
        }

        let anchor_root = find_root(&mut parent, 0);
        let unanchored_ranks: Vec<usize> = (0..depth)
            .filter(|&rank| find_root(&mut parent, rank) != anchor_root)
            .collect();
        let unanchored_doc_ids: Vec<String> = (0..doc_count)
            .filter(|&doc| find_root(&mut parent, depth + doc) != anchor_root)
            .map(|doc| self.doc_ids[doc].clone())
            .collect();

        PbmIdentifiability {
            anchored,
            fully_identified: unanchored_ranks.is_empty() && unanchored_doc_ids.is_empty(),
            component_count,
            unanchored_ranks,
            unanchored_doc_ids,
        }
    }
}

impl ClickModel for PositionBasedModel {
    fn fit(&mut self, log: &ClickLog) -> ClickModelResult<ClickModelFit> {
        self.config.validate()?;
        if log.is_empty() {
            return Err(ClickModelError::EmptyLog);
        }

        let depth = log.depth();
        let (doc_ids, doc_index) = document_vocabulary(log);
        let doc_count = doc_ids.len();

        // ── initialisation ───────────────────────────────────────────────────
        // γ_r = decay^r, which already satisfies γ₀ = 1; the anchor then holds
        // it there for the whole run.
        let mut examination = Vec::with_capacity(depth);
        for rank in 0..depth {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let power = rank as i32;
            let raw = self.config.initial_examination_decay.powi(power);
            let value = if self.config.anchor_top_examination && rank == 0 {
                1.0
            } else {
                self.config.clamp_parameter(raw)
            };
            examination.push(value);
        }
        let attractiveness = vec![
            self.config
                .clamp_parameter(self.config.initial_attractiveness);
            doc_count
        ];

        self.doc_ids = doc_ids;
        self.doc_index = doc_index;
        self.examination = examination;
        self.attractiveness = attractiveness;

        // ── EM ───────────────────────────────────────────────────────────────
        let mut previous =
            self.log_likelihood_with(log, &self.examination, &self.attractiveness)?;
        let mut history = vec![previous];
        let mut iterations = 0_usize;
        let mut converged = false;

        for _ in 0..self.config.max_iterations {
            let (examination, attractiveness) = self.em_step(log);
            self.examination = examination;
            self.attractiveness = attractiveness;
            iterations += 1;

            let current = self.log_likelihood_with(log, &self.examination, &self.attractiveness)?;
            history.push(current);
            let delta = current - previous;
            previous = current;
            if delta.abs() < self.config.tolerance {
                converged = true;
                break;
            }
        }

        self.fitted = true;
        self.identifiability = Some(self.analyse_identifiability(log, depth));

        Ok(ClickModelFit {
            iterations,
            converged,
            final_log_likelihood: previous,
            log_likelihood_history: history,
        })
    }

    fn predict_click_prob(&self, doc_id: &str, rank: usize) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        let gamma = self.lookup_rank(rank)?;
        let alpha = self.attractiveness[self.lookup_doc(doc_id)?];
        Ok(gamma * alpha)
    }

    fn log_likelihood(&self, log: &ClickLog) -> ClickModelResult<f64> {
        self.ensure_fitted()?;
        self.log_likelihood_with(log, &self.examination, &self.attractiveness)
    }

    fn examination_curve(&self) -> &[f64] {
        &self.examination
    }

    fn model_name(&self) -> &'static str {
        "PositionBasedModel"
    }
}
