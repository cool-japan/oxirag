//! Propensities and counterfactual (off-policy) estimators: turning a fitted
//! click model into an **unbiased** answer to the question *"how good would a
//! ranking policy I have never deployed have been?"*.
//!
//! # The propensity
//!
//! A fitted click model hands back an examination curve — for the PBM, the
//! parameter `γ_r` itself. That curve **is** the propensity: the probability
//! that a document placed at rank `r` was given a chance to be clicked. Every
//! estimator below is built on the single identity
//!
//! ```text
//! E[ click | doc d at rank r ] = γ_r · α_d      ⟹      E[ click / γ_r ] = α_d
//! ```
//!
//! — *divide a click by the probability of it being seen at all, and you get an
//! unbiased estimate of relevance.* This is Horvitz–Thompson estimation, and
//! examination is the missing-data mechanism.
//!
//! # Propensity clipping, and why it is not optional
//!
//! The importance weight is `1 / γ_r`, so a rank whose examination probability
//! is estimated at `0.001` multiplies a single lucky click by `1000`. The
//! estimator stays *unbiased* — but its **variance** explodes, and in a finite
//! sample it is dominated by whichever rare deep click happened to land.
//! [`PropensityEstimates`] therefore floors every propensity at
//! [`ClickModelConfig::propensity_clip`](super::types::ClickModelConfig::propensity_clip)
//! (default `0.01`, a weight cap of `100`).
//!
//! Clipping trades a little bias for a lot of variance: a clipped rank's clicks
//! are now *under*-weighted, so the estimate of a policy that promotes documents
//! from those ranks is biased **downwards** — conservatively. Track
//! [`CounterfactualEstimate::clipped_impression_count`] and
//! [`CounterfactualEstimate::effective_sample_size`] to see how much of the
//! estimate rests on how few observations.
//!
//! # The three estimators
//!
//! Write `w = 1 / γ_r` for the importance weight of a click logged at rank `r`,
//! and `g(d) = 1 / log2(2 + rank_π(d))` for the gain a candidate policy `π`
//! would earn by placing `d` where it places it.
//!
//! **IPS** — inverse propensity scoring:
//!
//! ```text
//! V_IPS = (1/|S|) · Σ_sessions Σ_{d clicked} g(d) · w(d)
//! ```
//!
//! Unbiased for `(1/|S|)·Σ_s Σ_d g(d)·α_d` whenever the propensities are
//! correct. Its weakness is entirely variance: the random *number and depth* of
//! clicks makes `Σ w` swing wildly from sample to sample.
//!
//! **SNIPS** — self-normalised IPS:
//!
//! ```text
//! V_SNIPS = Σ g(d)·w(d) / Σ w(d)
//! ```
//!
//! Divide by the realised total weight instead of by the session count. The
//! fluctuation in `Σ w` now appears in *both* numerator and denominator and
//! largely cancels, which is why SNIPS is markedly more stable. The price is a
//! small bias — it is a **ratio** of two random variables, and
//! `E[X/Y] ≠ E[X]/E[Y]` — that vanishes as `O(1/n)`. In practice the variance
//! reduction dwarfs the bias, and the module's tests measure exactly that
//! trade-off. Note SNIPS reports a *per-clicked-impression* gain, not a
//! per-session total, so it lives on a different scale to IPS by construction.
//!
//! **Doubly robust** — see [`DoublyRobustEstimator`], which needs a section of
//! its own, because the honest story here is more interesting than the textbook
//! one.

use std::collections::HashMap;

use super::types::{
    ClickLog, ClickModel, ClickModelError, ClickModelResult, ClickSession, position_gain,
};

// ── PropensityEstimates ──────────────────────────────────────────────────────

/// A clipped, invertible examination curve — the bridge between a fitted click
/// model and every counterfactual estimator in this module.
#[derive(Debug, Clone, PartialEq)]
pub struct PropensityEstimates {
    examination: Vec<f64>,
    clip_floor: f64,
}

impl PropensityEstimates {
    /// Wrap a raw examination curve with a clipping floor.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::InvalidPropensity`] if any entry is non-finite,
    /// non-positive, or greater than `1`, or if `clip_floor` is outside
    /// `(0, 1]`.
    pub fn new(examination: Vec<f64>, clip_floor: f64) -> ClickModelResult<Self> {
        if !clip_floor.is_finite() || clip_floor <= 0.0 || clip_floor > 1.0 {
            return Err(ClickModelError::InvalidPropensity {
                rank: 0,
                value: clip_floor,
                reason: "clip floor must lie inside (0, 1]",
            });
        }
        for (rank, &value) in examination.iter().enumerate() {
            if !value.is_finite() {
                return Err(ClickModelError::InvalidPropensity {
                    rank,
                    value,
                    reason: "examination probability must be finite",
                });
            }
            if value <= 0.0 || value > 1.0 {
                return Err(ClickModelError::InvalidPropensity {
                    rank,
                    value,
                    reason: "examination probability must lie inside (0, 1]",
                });
            }
        }
        Ok(Self {
            examination,
            clip_floor,
        })
    }

    /// Take the examination curve from any fitted [`ClickModel`].
    ///
    /// # Errors
    ///
    /// [`ClickModelError::NotFitted`] if the model has not been fitted, and
    /// [`ClickModelError::InvalidPropensity`] if its curve is unusable.
    pub fn from_click_model(model: &dyn ClickModel, clip_floor: f64) -> ClickModelResult<Self> {
        let curve = model.examination_curve();
        if curve.is_empty() {
            return Err(ClickModelError::NotFitted {
                model: model.model_name(),
            });
        }
        Self::new(curve.to_vec(), clip_floor)
    }

    /// The "no position bias at all" baseline: every rank has propensity `1`, so
    /// every importance weight is `1` and every estimator below degenerates to
    /// its **naive** counterpart. Useful precisely as a control — it is how the
    /// module's tests demonstrate that the debiasing, and not something else,
    /// is what recovers the true ranking.
    #[must_use]
    pub fn uniform(depth: usize) -> Self {
        Self {
            examination: vec![1.0; depth],
            clip_floor: 1.0,
        }
    }

    /// The number of ranks this curve covers.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.examination.len()
    }

    /// The clipping floor.
    #[must_use]
    pub fn clip_floor(&self) -> f64 {
        self.clip_floor
    }

    /// The **unclipped** examination probability at `rank`, or `None` beyond the
    /// curve's depth.
    #[must_use]
    pub fn raw_examination(&self, rank: usize) -> Option<f64> {
        self.examination.get(rank).copied()
    }

    /// The clipped propensity at `rank`.
    ///
    /// Ranks beyond the curve's depth fall back on the clipping floor. That is
    /// the *most pessimistic* propensity this estimator will admit, and so the
    /// *largest* importance weight it will produce — a deliberately loud
    /// fallback rather than a quiet one. Use
    /// [`PropensityEstimates::try_propensity`] if you would rather it were an
    /// error.
    #[must_use]
    pub fn propensity(&self, rank: usize) -> f64 {
        self.examination
            .get(rank)
            .copied()
            .unwrap_or(self.clip_floor)
            .max(self.clip_floor)
    }

    /// The clipped propensity at `rank`, or an error beyond the curve's depth.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::UnknownRank`] when `rank >= depth()`.
    pub fn try_propensity(&self, rank: usize) -> ClickModelResult<f64> {
        self.examination
            .get(rank)
            .copied()
            .map(|value| value.max(self.clip_floor))
            .ok_or(ClickModelError::UnknownRank {
                rank,
                depth: self.examination.len(),
            })
    }

    /// The inverse-propensity (importance) weight `1 / propensity(rank)`, capped
    /// at `1 / clip_floor` by construction.
    #[must_use]
    pub fn importance_weight(&self, rank: usize) -> f64 {
        1.0 / self.propensity(rank)
    }

    /// Whether the clipping floor actually binds at `rank` — i.e. whether the
    /// fitted examination probability there was *below* the floor and the
    /// estimator is therefore deliberately under-weighting that rank's clicks.
    #[must_use]
    pub fn is_clipped(&self, rank: usize) -> bool {
        self.examination
            .get(rank)
            .is_none_or(|&value| value < self.clip_floor)
    }

    /// How many ranks in the curve are clipped.
    #[must_use]
    pub fn clipped_rank_count(&self) -> usize {
        (0..self.examination.len())
            .filter(|&rank| self.is_clipped(rank))
            .count()
    }
}

// ── ClickRankingPolicy ───────────────────────────────────────────────────────

/// A candidate ranking policy: anything that can score a `(query, document)`
/// pair. Higher scores rank first; ties fall back on the logged order, so
/// ranking is fully deterministic.
///
/// This is the "new policy" whose value the counterfactual estimators estimate
/// **without ever deploying it**.
pub trait ClickRankingPolicy {
    /// The policy's score for `doc_id` under `query_id`. Higher is better.
    fn score(&self, query_id: &str, doc_id: &str) -> f64;
}

/// The zero-based position each of a session's logged documents would occupy
/// under `policy`, indexed by their *logged* rank.
///
/// Ties are broken by logged order, so the mapping is a deterministic
/// permutation.
#[must_use]
pub fn policy_positions(policy: &dyn ClickRankingPolicy, session: &ClickSession) -> Vec<usize> {
    let n = session.ranked_doc_ids.len();
    let scores: Vec<f64> = session
        .ranked_doc_ids
        .iter()
        .map(|doc_id| policy.score(&session.query_id, doc_id))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(&right))
    });
    let mut positions = vec![0_usize; n];
    for (new_rank, &logged_rank) in order.iter().enumerate() {
        positions[logged_rank] = new_rank;
    }
    positions
}

/// A ranking policy backed by a plain lookup table of document scores, with an
/// optional per-query override layer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TableClickPolicy {
    doc_scores: HashMap<String, f64>,
    query_doc_scores: HashMap<String, HashMap<String, f64>>,
    default_score: f64,
}

impl TableClickPolicy {
    /// An empty table; every document scores `default_score` (`0` unless
    /// overridden).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a table from a document → score map.
    #[must_use]
    pub fn from_doc_scores(doc_scores: HashMap<String, f64>) -> Self {
        Self {
            doc_scores,
            query_doc_scores: HashMap::new(),
            default_score: 0.0,
        }
    }

    /// Set one document's query-independent score.
    #[must_use]
    pub fn with_doc_score(mut self, doc_id: impl Into<String>, score: f64) -> Self {
        self.doc_scores.insert(doc_id.into(), score);
        self
    }

    /// Set one `(query, document)` score, overriding the query-independent one.
    #[must_use]
    pub fn with_query_doc_score(
        mut self,
        query_id: impl Into<String>,
        doc_id: impl Into<String>,
        score: f64,
    ) -> Self {
        self.query_doc_scores
            .entry(query_id.into())
            .or_default()
            .insert(doc_id.into(), score);
        self
    }

    /// Set the score used for documents that are in neither table.
    #[must_use]
    pub fn with_default_score(mut self, score: f64) -> Self {
        self.default_score = score;
        self
    }
}

impl ClickRankingPolicy for TableClickPolicy {
    fn score(&self, query_id: &str, doc_id: &str) -> f64 {
        if let Some(score) = self
            .query_doc_scores
            .get(query_id)
            .and_then(|scores| scores.get(doc_id))
        {
            return *score;
        }
        self.doc_scores
            .get(doc_id)
            .copied()
            .unwrap_or(self.default_score)
    }
}

// ── naive baseline ───────────────────────────────────────────────────────────

/// The naive, **biased** baseline every serious click-model discussion starts by
/// discrediting: raw click-through rate, `clicks(d) / impressions(d)`.
///
/// It answers *"how often was this clicked?"*, which is not the question. A
/// mediocre document parked at rank 0 out-scores an excellent one buried at rank
/// 9 every single time, because CTR silently attributes the top slot's
/// examination advantage to the document sitting in it. Recovering the true
/// order from the same log is exactly what [`crate::click_model::DebiasedRanker`]
/// and the estimators here are for, and the module's tests demonstrate the
/// reversal end to end.
#[must_use]
pub fn naive_click_through_rates(log: &ClickLog) -> HashMap<String, f64> {
    let mut clicks: HashMap<&str, u64> = HashMap::new();
    let mut impressions: HashMap<&str, u64> = HashMap::new();
    for impression in log.impressions() {
        *impressions.entry(impression.doc_id).or_insert(0) += 1;
        if impression.clicked {
            *clicks.entry(impression.doc_id).or_insert(0) += 1;
        }
    }
    impressions
        .into_iter()
        .map(|(doc_id, shown)| {
            let clicked = clicks.get(doc_id).copied().unwrap_or(0);
            #[allow(clippy::cast_precision_loss)]
            let rate = clicked as f64 / shown as f64;
            (doc_id.to_owned(), rate)
        })
        .collect()
}

// ── ClickRewardModel ─────────────────────────────────────────────────────────

/// The regression baseline `ŷ(d)` the doubly-robust estimator leans on: a
/// direct, model-based guess at a document's relevance, made *without* any
/// importance weighting.
///
/// On its own this is the **direct method**, and on its own it is exactly as
/// trustworthy as the model that produced it — a systematically wrong `ŷ` is a
/// systematically wrong estimate, with no data-driven correction anywhere. Its
/// value is that it has *low variance* (no `1/γ` anywhere), which is precisely
/// the property IPS lacks. Doubly-robust estimation is the marriage of the two.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClickRewardModel {
    doc_reward: HashMap<String, f64>,
    default_reward: f64,
}

impl ClickRewardModel {
    /// A reward model that predicts `value` for every document.
    ///
    /// `ClickRewardModel::constant(0.0)` is the *degenerate* baseline that makes
    /// the doubly-robust estimator collapse exactly onto IPS — the module tests
    /// assert that identity, because it is the cleanest possible check that the
    /// DR algebra is right.
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self {
            doc_reward: HashMap::new(),
            default_reward: value,
        }
    }

    /// A reward model from an explicit table, with a fallback for documents it
    /// does not name.
    #[must_use]
    pub fn from_table(doc_reward: HashMap<String, f64>, default_reward: f64) -> Self {
        Self {
            doc_reward,
            default_reward,
        }
    }

    /// Fit a per-document reward predictor from the log itself, by *averaging
    /// the propensity-corrected clicks* of each document and shrinking that
    /// average toward the global mean.
    ///
    /// `ŷ(d) = clamp₀¹( (Σ_i c_i/γ_{r_i} + κ·ȳ) / (n_d + κ) )`
    ///
    /// where `κ = prior_strength` (an equivalent-sample-size prior) and `ȳ` is
    /// the pooled IPS estimate over the whole log. Shrinkage matters: a document
    /// seen twice, clicked once, at a clipped deep rank would otherwise be
    /// handed a reward of `1.0` on the strength of a single observation, and the
    /// direct-method half of the DR estimator would inherit that noise wholesale.
    #[must_use]
    pub fn fit_ips(
        log: &ClickLog,
        propensities: &PropensityEstimates,
        prior_strength: f64,
    ) -> Self {
        let mut weighted_clicks: HashMap<&str, f64> = HashMap::new();
        let mut impressions: HashMap<&str, f64> = HashMap::new();
        let mut global_weighted = 0.0_f64;
        let mut global_impressions = 0.0_f64;

        for impression in log.impressions() {
            let contribution = if impression.clicked {
                propensities.importance_weight(impression.rank)
            } else {
                0.0
            };
            *weighted_clicks.entry(impression.doc_id).or_insert(0.0) += contribution;
            *impressions.entry(impression.doc_id).or_insert(0.0) += 1.0;
            global_weighted += contribution;
            global_impressions += 1.0;
        }

        let global_mean = if global_impressions > 0.0 {
            (global_weighted / global_impressions).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let prior_strength = prior_strength.max(0.0);

        let doc_reward = impressions
            .into_iter()
            .map(|(doc_id, shown)| {
                let weighted = weighted_clicks.get(doc_id).copied().unwrap_or(0.0);
                let reward = (weighted + prior_strength * global_mean) / (shown + prior_strength);
                (doc_id.to_owned(), reward.clamp(0.0, 1.0))
            })
            .collect();

        Self {
            doc_reward,
            default_reward: global_mean,
        }
    }

    /// The predicted reward of a document.
    #[must_use]
    pub fn reward(&self, doc_id: &str) -> f64 {
        self.doc_reward
            .get(doc_id)
            .copied()
            .unwrap_or(self.default_reward)
    }

    /// The fallback reward for unknown documents.
    #[must_use]
    pub fn default_reward(&self) -> f64 {
        self.default_reward
    }
}

// ── CounterfactualStrategy / CounterfactualEstimate ──────────────────────────

/// Which counterfactual estimator to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CounterfactualStrategy {
    /// Inverse propensity scoring: unbiased, high variance.
    Ips,
    /// Self-normalised IPS: slightly biased, much lower variance.
    Snips,
    /// Doubly robust: an IPS correction on top of a regression baseline.
    DoublyRobust,
}

impl CounterfactualStrategy {
    /// A short, stable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Ips => "IPS",
            Self::Snips => "SNIPS",
            Self::DoublyRobust => "DR",
        }
    }
}

/// The result of one counterfactual evaluation, with the diagnostics you need in
/// order to decide whether to believe it.
#[derive(Debug, Clone, PartialEq)]
pub struct CounterfactualEstimate {
    /// Which estimator produced this.
    pub strategy: CounterfactualStrategy,
    /// The estimated value of the evaluated policy. For
    /// [`CounterfactualStrategy::Ips`] and
    /// [`CounterfactualStrategy::DoublyRobust`] this is a *per-session*
    /// gain-weighted relevance total; for [`CounterfactualStrategy::Snips`] it is
    /// a *per-clicked-impression* average, which is a different scale (see the
    /// module docs).
    pub value: f64,
    /// How many sessions contributed.
    pub session_count: usize,
    /// How many impressions contributed.
    pub impression_count: usize,
    /// How many of those impressions were clicked — the only ones IPS and SNIPS
    /// can see at all.
    pub clicked_impression_count: usize,
    /// `Σ 1/γ_r` over clicked impressions: the total importance mass this
    /// estimate rests on.
    pub total_importance_weight: f64,
    /// The largest single importance weight used. Compare against
    /// `1 / clip_floor` to see whether clipping is binding.
    pub max_importance_weight: f64,
    /// How many contributing impressions sat at a *clipped* rank — i.e. how much
    /// of the estimate is deliberately, conservatively under-weighted.
    pub clipped_impression_count: usize,
    /// Kish's effective sample size of the importance weights,
    /// `(Σw)² / Σw²`. If this is far below
    /// [`CounterfactualEstimate::clicked_impression_count`], the estimate is
    /// really resting on a handful of heavily-weighted clicks and its variance
    /// is much larger than the raw click count suggests.
    pub effective_sample_size: f64,
}

// ── CounterfactualEstimator ──────────────────────────────────────────────────

/// The one estimator that implements all three strategies over a click log.
///
/// [`IpsEstimator`], [`SnipsEstimator`] and [`DoublyRobustEstimator`] are thin,
/// self-documenting wrappers around it.
#[derive(Debug, Clone)]
pub struct CounterfactualEstimator {
    propensities: PropensityEstimates,
    reward_model: ClickRewardModel,
    /// Per-document attractiveness (from a fitted click model), used only to
    /// impute the latent examination indicator when the log does not carry one.
    attractiveness: HashMap<String, f64>,
    default_attractiveness: f64,
}

impl CounterfactualEstimator {
    /// A new estimator over a propensity curve. Without a reward model it
    /// behaves as `ŷ ≡ 0`, which makes [`CounterfactualStrategy::DoublyRobust`]
    /// identical to [`CounterfactualStrategy::Ips`].
    #[must_use]
    pub fn new(propensities: PropensityEstimates) -> Self {
        Self {
            propensities,
            reward_model: ClickRewardModel::constant(0.0),
            attractiveness: HashMap::new(),
            default_attractiveness: 0.0,
        }
    }

    /// Attach the regression baseline used by the doubly-robust strategy.
    #[must_use]
    pub fn with_reward_model(mut self, reward_model: ClickRewardModel) -> Self {
        self.reward_model = reward_model;
        self
    }

    /// Attach the click model's fitted attractiveness, so the doubly-robust
    /// strategy can *impute* the latent examination indicator on sessions that
    /// do not carry ground-truth [`ClickSession::examinations`]. See
    /// [`DoublyRobustEstimator`] for exactly what this changes about the
    /// guarantee.
    #[must_use]
    pub fn with_attractiveness(mut self, attractiveness: HashMap<String, f64>) -> Self {
        self.default_attractiveness = if attractiveness.is_empty() {
            0.0
        } else {
            let sum: f64 = attractiveness.values().sum();
            #[allow(clippy::cast_precision_loss)]
            let count = attractiveness.len() as f64;
            sum / count
        };
        self.attractiveness = attractiveness;
        self
    }

    /// The propensity curve this estimator weights with.
    #[must_use]
    pub fn propensities(&self) -> &PropensityEstimates {
        &self.propensities
    }

    /// The regression baseline this estimator's DR strategy uses.
    #[must_use]
    pub fn reward_model(&self) -> &ClickRewardModel {
        &self.reward_model
    }

    /// The examination indicator used by the doubly-robust estimator for one
    /// impression.
    ///
    /// - Ground truth, when the session carries it: exact, and the DR estimator
    ///   is then genuinely doubly robust.
    /// - `1`, for a clicked impression: a click *proves* examination, under every
    ///   model here. No imputation needed.
    /// - Otherwise, the PBM examination posterior
    ///   `ô = γ_r·(1 − α_d) / (1 − γ_r·α_d)` — the same quantity the PBM E-step
    ///   computes. Its expectation under the true click process is
    ///   `γ·α·1 + (1 − γ·α)·γ(1 − α)/(1 − γ·α) = γ`, **exactly** the true
    ///   examination rate, whenever the click model is correct. That identity is
    ///   the entire reason the DR estimator below stays unbiased on the
    ///   propensity-correct branch.
    fn examination_indicator(
        &self,
        doc_id: &str,
        rank: usize,
        clicked: bool,
        logged: Option<bool>,
    ) -> f64 {
        if let Some(examined) = logged {
            return f64::from(u8::from(examined));
        }
        if clicked {
            return 1.0;
        }
        let gamma = self.propensities.propensity(rank);
        let alpha = self
            .attractiveness
            .get(doc_id)
            .copied()
            .unwrap_or(self.default_attractiveness);
        let denominator = 1.0 - gamma * alpha;
        if denominator <= f64::MIN_POSITIVE {
            // γ = α = 1: examination is certain and so is a click, so a
            // *non*-click is outside the model. The honest imputation is 1 (the
            // user did look); the likelihood floor makes this unreachable on any
            // fitted model, but the estimator accepts caller-supplied curves too.
            return 1.0;
        }
        (gamma * (1.0 - alpha) / denominator).clamp(0.0, 1.0)
    }

    /// Estimate the value of `policy` on `log` with the chosen strategy.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::EmptyLog`] when `log` has no sessions.
    pub fn estimate(
        &self,
        strategy: CounterfactualStrategy,
        log: &ClickLog,
        policy: &dyn ClickRankingPolicy,
    ) -> ClickModelResult<CounterfactualEstimate> {
        if log.is_empty() {
            return Err(ClickModelError::EmptyLog);
        }

        let mut value_numerator = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        let mut weight_square_sum = 0.0_f64;
        let mut max_weight = 0.0_f64;
        let mut clicked_impressions = 0_usize;
        let mut clipped_impressions = 0_usize;
        let mut impressions = 0_usize;

        for session in log.sessions() {
            let positions = policy_positions(policy, session);
            for (logged_rank, doc_id) in session.ranked_doc_ids.iter().enumerate() {
                impressions += 1;
                let clicked = session.clicks[logged_rank];
                let gain = position_gain(positions[logged_rank]);
                let propensity = self.propensities.propensity(logged_rank);
                let weight = 1.0 / propensity;

                if clicked {
                    clicked_impressions += 1;
                    weight_sum += weight;
                    weight_square_sum += weight * weight;
                    max_weight = max_weight.max(weight);
                    if self.propensities.is_clipped(logged_rank) {
                        clipped_impressions += 1;
                    }
                }

                let click = f64::from(u8::from(clicked));
                match strategy {
                    CounterfactualStrategy::Ips | CounterfactualStrategy::Snips => {
                        // Unclicked impressions contribute exactly zero, so this
                        // is the "sum over clicked impressions" of the docs.
                        value_numerator += gain * click * weight;
                    }
                    CounterfactualStrategy::DoublyRobust => {
                        let predicted = self.reward_model.reward(doc_id);
                        let examined = self.examination_indicator(
                            doc_id,
                            logged_rank,
                            clicked,
                            session
                                .examinations
                                .as_ref()
                                .map(|flags| flags[logged_rank]),
                        );
                        // ŷ + (c − ô·ŷ)/γ  — the augmented-IPW estimator of the
                        // document's relevance, scored by the new policy's gain.
                        let corrected = predicted + (click - examined * predicted) * weight;
                        value_numerator += gain * corrected;
                    }
                }
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let session_count = log.len() as f64;
        let value = match strategy {
            CounterfactualStrategy::Ips | CounterfactualStrategy::DoublyRobust => {
                value_numerator / session_count
            }
            CounterfactualStrategy::Snips => {
                if weight_sum > 0.0 {
                    value_numerator / weight_sum
                } else {
                    // No clicks anywhere: there is no importance mass to
                    // normalise by, and the honest answer is "zero observed
                    // value", not a division by zero.
                    0.0
                }
            }
        };

        let effective_sample_size = if weight_square_sum > 0.0 {
            weight_sum * weight_sum / weight_square_sum
        } else {
            0.0
        };

        Ok(CounterfactualEstimate {
            strategy,
            value,
            session_count: log.len(),
            impression_count: impressions,
            clicked_impression_count: clicked_impressions,
            total_importance_weight: weight_sum,
            max_importance_weight: max_weight,
            clipped_impression_count: clipped_impressions,
            effective_sample_size,
        })
    }
}

// ── the three named estimators ───────────────────────────────────────────────

/// Inverse-propensity-scored off-policy evaluation.
///
/// ```text
/// V_IPS = (1/|S|) · Σ_sessions Σ_{d clicked at logged rank r} gain(rank_π(d)) / γ_r
/// ```
///
/// Unbiased whenever the propensities are right, and blind to every impression
/// that was not clicked. Its variance is driven by `Σ 1/γ_r`, which is why
/// [`CounterfactualEstimate::effective_sample_size`] is worth reading.
#[derive(Debug, Clone)]
pub struct IpsEstimator {
    inner: CounterfactualEstimator,
}

impl IpsEstimator {
    /// A new IPS estimator over a propensity curve.
    #[must_use]
    pub fn new(propensities: PropensityEstimates) -> Self {
        Self {
            inner: CounterfactualEstimator::new(propensities),
        }
    }

    /// Estimate the value of `policy`.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::EmptyLog`] when `log` has no sessions.
    pub fn estimate(
        &self,
        log: &ClickLog,
        policy: &dyn ClickRankingPolicy,
    ) -> ClickModelResult<CounterfactualEstimate> {
        self.inner
            .estimate(CounterfactualStrategy::Ips, log, policy)
    }
}

/// Self-normalised IPS — the same weighted sum, divided by the realised total
/// importance weight instead of by the session count.
///
/// ```text
/// V_SNIPS = Σ gain(rank_π(d))/γ_r  ÷  Σ 1/γ_r          (both over clicked impressions)
/// ```
///
/// **The trade-off, stated plainly.** IPS's variance has two sources: which
/// documents were clicked, and *how much total weight* those clicks happened to
/// carry. A sample that happens to contain a couple of extra deep clicks gets a
/// large `Σ w` and an inflated estimate. SNIPS puts that same `Σ w` in the
/// denominator, so the fluctuation largely cancels and the estimator is far more
/// stable across samples. In exchange it is a *ratio estimator* — a ratio of two
/// random quantities — and `E[X/Y] ≠ E[X]/E[Y]`, so SNIPS carries an `O(1/n)`
/// bias that IPS does not. For any realistic log the variance reduction is worth
/// far more than the bias costs, which is why SNIPS is the usual default in
/// practice.
#[derive(Debug, Clone)]
pub struct SnipsEstimator {
    inner: CounterfactualEstimator,
}

impl SnipsEstimator {
    /// A new SNIPS estimator over a propensity curve.
    #[must_use]
    pub fn new(propensities: PropensityEstimates) -> Self {
        Self {
            inner: CounterfactualEstimator::new(propensities),
        }
    }

    /// Estimate the value of `policy`.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::EmptyLog`] when `log` has no sessions.
    pub fn estimate(
        &self,
        log: &ClickLog,
        policy: &dyn ClickRankingPolicy,
    ) -> ClickModelResult<CounterfactualEstimate> {
        self.inner
            .estimate(CounterfactualStrategy::Snips, log, policy)
    }
}

/// Doubly-robust off-policy evaluation: a regression baseline, corrected by an
/// inverse-propensity-weighted residual.
///
/// ```text
/// V_DR = (1/|S|) · Σ_sessions Σ_d gain(rank_π(d)) · [ ŷ(d) + (c − ô·ŷ(d)) / γ_r ]
/// ```
///
/// where `c` is the observed click, `ŷ(d)` the [`ClickRewardModel`]'s guess at
/// the document's relevance, `γ_r` the propensity at the *logged* rank, and `ô`
/// the examination indicator.
///
/// # Why it is doubly robust
///
/// Write the truth as `E[c | d, r] = γ_r · ρ_d` (`ρ_d` = true relevance) and
/// `E[ô] = γ_r` (an unbiased examination indicator). Then:
///
/// - **Propensity model correct, reward model arbitrary.** With the true `γ`,
///   `E[ŷ + (c − ô·ŷ)/γ] = ŷ + (γρ − γŷ)/γ = ρ`. The `ŷ` terms cancel
///   *identically* — a hopelessly wrong reward model changes the variance and
///   nothing else.
/// - **Reward model correct, propensity model arbitrary.** With `ŷ = ρ`,
///   `E[c − ô·ŷ] = γρ − γρ = 0`, so the entire correction term has expectation
///   zero **no matter what `γ̃` we divide it by**, and the estimator returns
///   `ŷ = ρ`.
///
/// Either half being right saves the whole thing. That is the promise, and both
/// halves are asserted directly in the module's tests.
///
/// # The catch that the textbook version glosses over
///
/// Both branches above need an examination indicator `ô` with `E[ô] = γ`. In
/// classical missing-data problems the missingness indicator is *observed* — you
/// know which survey responses came back. **In click data it is not.** A zero
/// click is ambiguous between "never looked" and "looked and passed", and that
/// ambiguity is precisely what click models exist to resolve. So:
///
/// - If the session carries ground-truth [`ClickSession::examinations`] (a
///   viewport beacon, an eye-tracker, a simulator), `ô` is the truth and **both**
///   robustness branches hold exactly, as stated.
/// - Otherwise this estimator imputes `ô` from the click model's own examination
///   posterior, `ô = γ_r(1 − α_d)/(1 − γ_r·α_d)` for an unclicked impression and
///   `ô = 1` for a clicked one. That imputation satisfies `E[ô] = γ` **exactly**
///   when the click model is correct — so the *propensity-correct* branch
///   survives intact, for any reward model. The *reward-correct* branch does not:
///   it needs `E[ô] = γ_true`, and a wrong click model supplies a wrong `ô`,
///   leaving a residual bias of `ρ · (γ_true − E[ô]) / γ̃`. This is stated rather
///   than swept away, because a "doubly robust" estimator that quietly has only
///   one robust branch is worse than an honest IPS.
///
/// Setting [`ClickRewardModel::constant`] to `0.0` collapses DR exactly onto
/// IPS, which the tests check as an algebraic sanity bound.
#[derive(Debug, Clone)]
pub struct DoublyRobustEstimator {
    inner: CounterfactualEstimator,
}

impl DoublyRobustEstimator {
    /// A new DR estimator over a propensity curve and a regression baseline.
    #[must_use]
    pub fn new(propensities: PropensityEstimates, reward_model: ClickRewardModel) -> Self {
        Self {
            inner: CounterfactualEstimator::new(propensities).with_reward_model(reward_model),
        }
    }

    /// Supply the click model's fitted attractiveness, so that sessions without
    /// ground-truth examinations can have `ô` imputed from the PBM posterior
    /// rather than falling back on the global mean.
    #[must_use]
    pub fn with_attractiveness(mut self, attractiveness: HashMap<String, f64>) -> Self {
        self.inner = self.inner.with_attractiveness(attractiveness);
        self
    }

    /// Estimate the value of `policy`.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::EmptyLog`] when `log` has no sessions.
    pub fn estimate(
        &self,
        log: &ClickLog,
        policy: &dyn ClickRankingPolicy,
    ) -> ClickModelResult<CounterfactualEstimate> {
        self.inner
            .estimate(CounterfactualStrategy::DoublyRobust, log, policy)
    }
}
