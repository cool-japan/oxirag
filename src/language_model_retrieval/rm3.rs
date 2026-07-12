//! `RM1` / `RM3`: relevance-model estimation and query expansion, built on the
//! query-likelihood language model.
//!
//! # `RM1` — the relevance model
//!
//! Lavrenko & Croft (2001) observed that the language-modelling framework has
//! no notion of *relevance* in it at all: query likelihood ranks by
//! `P(Q | D)`, but what we actually want is a model of the terms a *relevant*
//! document would generate. Call that unknown distribution `P(w | R)`. We
//! never observe `R`, but we can approximate it: assume the query and a
//! relevant document's terms are sampled from the same underlying model, and
//! estimate it as the query-conditioned mixture of the document models of the
//! documents that the query itself already ranks highly:
//!
//! ```text
//! P(w | R)  ≈  Σ_{D ∈ F}  P(w | D) · P(D | Q)
//! ```
//!
//! over a feedback set `F` — the top `fb_docs` documents of the first-pass
//! ranking. Two things in that formula are load-bearing, and both are what
//! distinguish `RM1` from the heuristics it is often mistaken for:
//!
//! - `P(w | D)` is the **smoothed document language model** of
//!   [`super::smoothing`] — not a raw count. Every term in the vocabulary has
//!   non-zero probability under it, and the mixture is a genuine probability
//!   distribution.
//! - `P(D | Q)` is the document's **query likelihood**, turned into a
//!   posterior over the feedback set by Bayes' rule with a uniform prior:
//!
//! ```text
//! P(D | Q)  =  exp(score_QL(D, Q))  /  Σ_{D' ∈ F} exp(score_QL(D', Q))
//! ```
//!
//! So a feedback document that the query matched *twice as well* contributes
//! twice as much of its language model to the estimate. There is no rank
//! heuristic and no hand-tuned decay anywhere.
//!
//! ## Numerical stability: `P(D | Q)` must be computed in log space
//!
//! `score_QL(D, Q) = Σ_w qtf(w) · log P(w | D)` is a **sum of logarithms of
//! probabilities**, so it is large and negative — for a twenty-term query
//! against a large collection, `-150` is unremarkable, and pathological inputs
//! reach `-10 000`. Now observe that `f64::exp(-750.0)` is exactly `0.0`
//! (the smallest normal `f64` is about `2.2e-308`, and `e^-750 ≈ 3e-326`).
//! Exponentiating those scores directly therefore does not "lose a little
//! precision" — it silently produces a vector of **all zeros**, and the
//! subsequent normalization divides `0` by `0` and yields `NaN` for every
//! feedback document. The relevance model is destroyed, and nothing warns you.
//!
//! [`lm_query_posteriors`] avoids this with the max-subtraction (log-sum-exp)
//! identity. Because
//!
//! ```text
//! exp(sᵢ) / Σⱼ exp(sⱼ)  =  exp(sᵢ - m) / Σⱼ exp(sⱼ - m)     for any m
//! ```
//!
//! — the shift cancels exactly, top and bottom — we take `m = maxⱼ sⱼ`. Every
//! exponent `sᵢ - m` is then `≤ 0`, the largest is exactly `0` (so `exp` gives
//! `1` and the denominator is at least `1`, never zero), and only the
//! *relative* differences among the scores, which are small, ever reach `exp`.
//! Scores of `-10 000` and `-10 001` produce the same posteriors as `0` and
//! `-1`, which is exactly right: a posterior over `F` cannot depend on a
//! constant offset shared by all of `F`.
//!
//! ## Computing `P(w | R)` in closed form
//!
//! Naively, `P(w | R)` has support over the *entire* vocabulary, because the
//! smoothed `P(w | D)` is non-zero even for terms `D` does not contain. That
//! is not a rounding artifact to be swept away — it is what makes `P(· | R)` a
//! proper distribution — but it appears to force an `O(|F| · |V|)` computation.
//!
//! It does not, thanks to the foreground/background decomposition of
//! [`super::smoothing`]: `P(w | D) = S_D(w) + K_D · P(w | C)`, where `S_D(w)`
//! is zero for every term `D` does not contain and `K_D` does not depend on
//! `w` at all. Substituting,
//!
//! ```text
//! P(w | R)  =  Σ_D π_D · (S_D(w) + K_D · P(w | C))
//!           =  [ Σ_D π_D · S_D(w) ]  +  [ Σ_D π_D · K_D ] · P(w | C)
//!           =        S(w)            +           K        · P(w | C)
//! ```
//!
//! with `π_D = P(D | Q)`. The first bracket is **sparse** — supported only on
//! terms the feedback documents actually contain — and the second is a single
//! **scalar**. [`LmRelevanceModel`] stores exactly that: the sparse table `S`,
//! the scalar `K`, and the posteriors. The result is exact, not an
//! approximation, and the full distribution can be materialized on demand.
//!
//! The decomposition also *proves* normalization in one line. Using
//! `Σ_{w ∈ V} S_D(w) + K_D = 1` (the identity established in
//! [`super::smoothing`]) and `Σ_{w ∈ V} P(w | C) = 1`:
//!
//! ```text
//! Σ_{w ∈ V} P(w | R)  =  Σ_{w ∈ V} S(w)  +  K
//!                     =  Σ_D π_D · Σ_{w ∈ V} S_D(w)  +  Σ_D π_D · K_D
//!                     =  Σ_D π_D · ( Σ_{w ∈ V} S_D(w) + K_D )
//!                     =  Σ_D π_D  =  1
//! ```
//!
//! [`LmRelevanceModel::total_probability_mass`] returns exactly the left-hand
//! side, computed as `Σ S(w) + K`; it is `1` for any well-formed model.
//!
//! # `RM3` — interpolating the relevance model back into the query
//!
//! `RM1` alone is a notoriously aggressive query rewriter: it will happily
//! discard the user's own terms in favour of whatever the feedback documents
//! were about, and when the first-pass ranking has drifted, so does it.
//! `RM3` (Abdul-Jaleel et al., 2004) is the fix, and it is trivially simple —
//! interpolate with the query's own maximum-likelihood model
//! `P_ML(w | Q) = qtf(w) / |Q|`:
//!
//! ```text
//! P'(w | Q)  =  α · P_ML(w | Q)  +  (1 - α) · P(w | R)
//! ```
//!
//! Take the `fb_terms` heaviest terms under `P'`, renormalize their weights to
//! sum to `1`, and score with the resulting weighted query. `α = 1` is the
//! identity (the original query, unchanged, exactly); `α = 0` is pure `RM1`.
//!
//! # How this differs from the crate's two other feedback mechanisms
//!
//! `RM3` is the third query-feedback mechanism in this crate and the only
//! *probabilistic* one. The other two are worth naming precisely, because from
//! a distance all three look like "take the top documents, harvest terms, add
//! them to the query":
//!
//! | | mechanism | what it manipulates | where the term weights come from |
//! |---|---|---|---|
//! | [`RocchioFeedbackAdjuster`](crate::relevance_feedback::RocchioFeedbackAdjuster) | vector-space centroid arithmetic | *embedding vectors* (and, in `compute_relevance_model`, raw term sets) | geometry: `q' = α·q + β·mean(positive) - γ·mean(negative)`, i.e. move the query point toward the positive centroid and away from the negative one. There is no probability anywhere; the weights are whatever the vector arithmetic leaves behind, and `β`/`γ` are tuning knobs, not estimates. |
//! | [`PseudoRelevanceFeedback`](crate::query_expansion::prf::PseudoRelevanceFeedback) | a bag of literal words | *raw term frequencies* in the top-ranked documents' text | a rank heuristic: each document contributes `1 / (i + 1) · score` per term occurrence, summed. The `1/(i+1)` decay is a plausible-looking constant with no model behind it; a term's weight is a *count*, and the result is appended to the query as plain text with no weights at all. |
//! | `RM3` (this module) | a weighted term distribution | *smoothed document language models* `P(w \| D)` | estimation: `P(w \| R) = Σ_D P(w \| D) · P(D \| Q)`, then interpolated with `P_ML(w \| Q)`. Every quantity is a probability, the feedback documents are weighted by their **query likelihood** rather than by their rank, unseen terms have well-defined non-zero probability because the document models are smoothed, and the output is a genuine distribution that sums to one. |
//!
//! The practical consequences of that last row are not cosmetic. Because `RM3`
//! weights by likelihood rather than rank, a feedback set whose top document
//! matches the query far better than the rest is dominated by that document —
//! whereas `1/(i+1)` would give it a fixed `1.0` against the runner-up's
//! `0.5` no matter how much better it was. Because `RM3` mixes *smoothed*
//! models, a term that is common in the collection is not automatically a good
//! expansion term: it has high `P(w | D)` in every document, so it also has
//! high `P(w | C)`, and the mixture does not distinguish it — whereas raw
//! frequency counting rewards it directly. And because the output is a
//! distribution rather than a word list, the expansion terms carry
//! *calibrated* weights into the second-pass scoring, instead of each being
//! silently worth exactly one query term.
//!
//! None of the three is a substitute for the others: Rocchio operates on dense
//! vectors, `PseudoRelevanceFeedback` on any [`SearchResult`](crate::types::SearchResult)
//! list from any retriever, and `RM3` only inside a language-model index that
//! can supply `P(w | D)` and `P(D | Q)`. This module does not touch either of
//! them.

use std::collections::{HashMap, HashSet};

use super::types::{Rm3ExpandedQuery, Rm3Term};

// ── Log-sum-exp ──────────────────────────────────────────────────────────────

/// `log Σᵢ exp(sᵢ)`, computed by max-subtraction so that it never overflows or
/// underflows.
///
/// Returns `f64::NEG_INFINITY` for an empty slice (the additive identity of
/// the sum inside the log), and ignores `-inf` entries (they contribute zero
/// mass). If any entry is `+inf` or `NaN` the result is not meaningful, but it
/// is still returned rather than panicking.
///
/// See the module documentation for why the naive `Σ exp(sᵢ)` cannot be used:
/// query-likelihood scores are sums of log-probabilities, and `exp` of them
/// underflows to exactly zero long before the numbers themselves become
/// unreasonable.
#[must_use]
pub fn lm_log_sum_exp(log_scores: &[f64]) -> f64 {
    if log_scores.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = log_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        // Either every score is -inf (total mass zero -> log is -inf) or some
        // score is +inf/NaN, in which case max already carries that value and
        // propagating it is the honest answer.
        return max;
    }
    let sum: f64 = log_scores.iter().map(|s| (s - max).exp()).sum();
    max + sum.ln()
}

/// Turn query-likelihood log scores into the posterior `P(D | Q)` over the
/// feedback set, normalized to sum to `1`.
///
/// This is `softmax` with the max-subtraction trick, and it is the reason
/// `RM1` survives contact with real log-likelihoods. Feeding it
/// `[-10_000.0, -10_001.0]` returns `[0.731, 0.269]`, exactly as it would for
/// `[0.0, -1.0]`; feeding the same input to a naive `exp`-then-normalize
/// returns `[NaN, NaN]`.
///
/// Degenerate inputs: an empty slice returns an empty vector; a slice in which
/// every score is `-inf` (or otherwise carries no finite maximum) falls back
/// to the uniform distribution, which is the correct posterior when the
/// likelihood carries no information.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn lm_query_posteriors(log_scores: &[f64]) -> Vec<f64> {
    if log_scores.is_empty() {
        return Vec::new();
    }
    let max = log_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        let uniform = 1.0 / log_scores.len() as f64;
        return vec![uniform; log_scores.len()];
    }
    // Every shifted exponent is <= 0, and the largest is exactly 0, so `sum`
    // is at least 1.0 and the division below can never be 0/0.
    let shifted: Vec<f64> = log_scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = shifted.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let uniform = 1.0 / log_scores.len() as f64;
        return vec![uniform; log_scores.len()];
    }
    shifted.into_iter().map(|value| value / sum).collect()
}

// ── LmRelevanceModel ─────────────────────────────────────────────────────────

/// An estimated `RM1` relevance model `P(w | R)`.
///
/// Stored in the closed form derived in the module documentation:
///
/// ```text
/// P(w | R) = S(w) + K · P(w | C)
/// ```
///
/// with `S` sparse (supported on the feedback documents' terms) and `K` a
/// scalar. This is exact, not an approximation: [`Self::weights`] gives the
/// true `P(w | R)` for every term the feedback documents contain, and
/// [`LmRetrievalIndex::materialize_relevance_model`](super::LmRetrievalIndex::materialize_relevance_model)
/// expands it to the full vocabulary when the whole distribution is wanted.
///
/// Deliberately *not* named `RelevanceModel`: the crate already has a
/// [`crate::relevance_feedback::RelevanceModel`], which is an unrelated
/// vector-space structure.
#[derive(Debug, Clone, PartialEq)]
pub struct LmRelevanceModel {
    /// `P(w | R)` for every term occurring in at least one feedback document,
    /// heaviest first (ties broken by ascending term, so the order is
    /// deterministic).
    weights: Vec<(String, f64)>,
    /// The sparse foreground table `S(w) = Σ_D π_D · S_D(w)`, indexed by term.
    sparse: HashMap<String, f64>,
    /// `Σ_{w ∈ V} S(w)` — the total foreground mass. Together with
    /// [`Self::background_coefficient`] this must sum to `1`.
    sparse_mass: f64,
    /// `K = Σ_D π_D · K_D` — the coefficient on the collection model.
    background_coefficient: f64,
    /// The feedback set, as `(document_id, P(D | Q))`, in first-pass rank
    /// order. Sums to `1`.
    document_posteriors: Vec<(String, f64)>,
}

impl LmRelevanceModel {
    /// Assemble a relevance model from its parts.
    ///
    /// `sparse` is `S(w)`, `background_coefficient` is `K`, and
    /// `collection_probability` supplies `P(w | C)` so that the public
    /// [`Self::weights`] can be materialized for the feedback terms.
    pub(super) fn from_parts<F>(
        sparse: HashMap<String, f64>,
        background_coefficient: f64,
        document_posteriors: Vec<(String, f64)>,
        collection_probability: F,
    ) -> Self
    where
        F: Fn(&str) -> f64,
    {
        let sparse_mass = sparse.values().sum();
        let mut weights: Vec<(String, f64)> = sparse
            .iter()
            .map(|(term, foreground)| {
                let probability =
                    background_coefficient.mul_add(collection_probability(term), *foreground);
                (term.clone(), probability)
            })
            .collect();
        sort_terms_by_weight(&mut weights);

        Self {
            weights,
            sparse,
            sparse_mass,
            background_coefficient,
            document_posteriors,
        }
    }

    /// `P(w | R)` for every term the feedback documents contain, heaviest
    /// first.
    ///
    /// Terms *outside* the feedback set are not listed — not because they have
    /// zero probability (they do not; they have `K · P(w | C)`), but because
    /// there are `|V|` of them and they are exactly the terms the feedback
    /// carries no evidence about. Use
    /// [`LmRetrievalIndex::materialize_relevance_model`](super::LmRetrievalIndex::materialize_relevance_model)
    /// for the full distribution.
    #[must_use]
    pub fn weights(&self) -> &[(String, f64)] {
        &self.weights
    }

    /// `P(w | R)` for a single term, exactly — including terms absent from the
    /// feedback set, whose probability is `K · P(w | C)`.
    ///
    /// `collection_probability` is `P(w | C)`, which the model does not itself
    /// store.
    #[must_use]
    pub fn probability(&self, term: &str, collection_probability: f64) -> f64 {
        let foreground = self.sparse.get(term).copied().unwrap_or(0.0);
        self.background_coefficient
            .mul_add(collection_probability, foreground)
    }

    /// The foreground table `S(w) = Σ_D π_D · S_D(w)`.
    #[must_use]
    pub fn foreground(&self, term: &str) -> f64 {
        self.sparse.get(term).copied().unwrap_or(0.0)
    }

    /// `K = Σ_D π_D · K_D`, the coefficient multiplying the collection model.
    #[must_use]
    pub fn background_coefficient(&self) -> f64 {
        self.background_coefficient
    }

    /// `Σ_{w ∈ V} S(w)`, the total foreground mass.
    #[must_use]
    pub fn sparse_mass(&self) -> f64 {
        self.sparse_mass
    }

    /// `Σ_{w ∈ V} P(w | R) = Σ_{w ∈ V} S(w) + K`.
    ///
    /// **This is `1`** for any model built from a valid feedback set — see the
    /// one-line proof in the module documentation. It is exposed because a
    /// relevance model that does not integrate to one is a bug, and a caller
    /// (or a test) should be able to say so.
    #[must_use]
    pub fn total_probability_mass(&self) -> f64 {
        self.sparse_mass + self.background_coefficient
    }

    /// The feedback set with its query-likelihood posteriors `P(D | Q)`, in
    /// first-pass rank order. The posteriors sum to `1`.
    #[must_use]
    pub fn document_posteriors(&self) -> &[(String, f64)] {
        &self.document_posteriors
    }

    /// Interpolate with the query's own maximum-likelihood model and select
    /// the `fb_terms` heaviest terms — the `RM3` step.
    ///
    /// `query_weights` is `qtf(w)` for each distinct query term; it is
    /// normalized internally into `P_ML(w | Q) = qtf(w) / |Q|`, so callers may
    /// pass raw counts or any positive weighting.
    ///
    /// `collection_probability` supplies `P(w | C)`, which the model needs in
    /// order to evaluate `P(w | R) = S(w) + K · P(w | C)` for terms *outside*
    /// the feedback set. That matters: a query term that none of the feedback
    /// documents contains still has a non-zero relevance probability
    /// (`K · P(w | C)`), and silently treating it as zero would misstate `P'`
    /// for precisely the terms the user actually typed.
    ///
    /// The candidate set is the union of the query's terms and the feedback
    /// documents' terms. Terms whose interpolated weight is not strictly
    /// positive are dropped before selection, which is what makes `α = 1` an
    /// *exact* identity: at `α = 1` the factor `(1 - α)` annihilates the
    /// relevance model exactly, every feedback-only term has `P' = 0` and
    /// cannot be selected even if `fb_terms` exceeds the query's length, and
    /// the expanded query is the original query with weights `qtf(w) / |Q|`.
    #[must_use]
    pub fn expand_query<F>(
        &self,
        query_weights: &[(String, f64)],
        alpha: f64,
        fb_terms: usize,
        collection_probability: F,
    ) -> Rm3ExpandedQuery
    where
        F: Fn(&str) -> f64,
    {
        let query_total: f64 = query_weights.iter().map(|(_, weight)| *weight).sum();

        // P_ML(w|Q) = qtf(w) / |Q|, over the query's distinct terms.
        let mut maximum_likelihood: HashMap<&str, f64> = HashMap::new();
        if query_total > 0.0 {
            for (term, weight) in query_weights {
                *maximum_likelihood.entry(term.as_str()).or_insert(0.0) += weight / query_total;
            }
        }

        // Candidate set: query terms union feedback terms.
        let mut candidates: Vec<(String, f64)> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        let candidate_terms = self
            .weights
            .iter()
            .map(|(term, _)| term.as_str())
            .chain(maximum_likelihood.keys().copied());

        for term in candidate_terms {
            if !seen.insert(term) {
                continue;
            }
            let prior = maximum_likelihood.get(term).copied().unwrap_or(0.0);
            let relevance = self.probability(term, collection_probability(term));
            let interpolated = (1.0 - alpha).mul_add(relevance, alpha * prior);
            if interpolated > 0.0 {
                candidates.push((term.to_string(), interpolated));
            }
        }

        sort_terms_by_weight(&mut candidates);
        candidates.truncate(fb_terms);

        renormalize_into_query(candidates)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Sort `(term, weight)` pairs by descending weight, breaking ties by
/// ascending term so the order is total and deterministic.
///
/// Uses [`f64::total_cmp`]: `partial_cmp` returns `None` for `NaN`, and a
/// comparator that lies about a total order is instant undefined *behaviour*
/// (not memory-unsafety, but arbitrary output) inside `sort_by`.
pub(super) fn sort_terms_by_weight(terms: &mut [(String, f64)]) {
    terms.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
}

/// Renormalize a selected term list to sum to `1` and wrap it as an
/// [`Rm3ExpandedQuery`].
///
/// If the total weight is not strictly positive (only reachable when the
/// selection is empty), the result is an empty expanded query rather than a
/// vector of `NaN`s.
pub(super) fn renormalize_into_query(terms: Vec<(String, f64)>) -> Rm3ExpandedQuery {
    let total: f64 = terms.iter().map(|(_, weight)| *weight).sum();
    if total <= 0.0 || !total.is_finite() {
        return Rm3ExpandedQuery::new(Vec::new());
    }
    Rm3ExpandedQuery::new(
        terms
            .into_iter()
            .map(|(term, weight)| Rm3Term {
                term,
                weight: weight / total,
            })
            .collect(),
    )
}
