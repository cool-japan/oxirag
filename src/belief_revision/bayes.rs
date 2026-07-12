//! Core Bayesian belief-revision mathematics, carried out in
//! log-probability / log-odds space for numerical stability.
//!
//! A Bayesian filter that ingests a long stream of evidence multiplies many
//! small probabilities together. In *linear* space that product underflows:
//! `0.1_f64.powi(400)` is smaller than the smallest positive `f64` and rounds
//! to exactly `0.0`, at which point the final normalization becomes `0 / 0`
//! and the whole distribution collapses to `NaN`. Working in log space turns
//! those products into sums, which never underflow, and the one delicate
//! step — exponentiating to renormalize — is done through a numerically
//! stable `logsumexp` that subtracts the maximum first.
//!
//! This file owns:
//!
//! * `logsumexp` — the stable `log(sum(exp(x_i)))` primitive.
//! * `log_likelihood_floored` — `ln` of a likelihood clamped to a positive
//!   floor, so a single zero-likelihood observation cannot send a hypothesis
//!   to `-inf` (a permanent, unrecoverable "impossible" state).
//! * `normalize_log_probs` / `bayes_update_log` — the in-place log-space
//!   Bayesian update: add log-likelihoods, then subtract `logsumexp` so the
//!   distribution stays a valid probability distribution.
//! * `log_probs_to_linear` / `entropy_nats` — read-out helpers.
//! * [`LikelihoodRatio`] — the pairwise log-odds (log Bayes factor) form of
//!   the update, `log(P(e | h_i) / P(e | h_j))`.

/// Numerically stable `log(sum_i exp(values[i]))`.
///
/// The naive evaluation overflows for large inputs (`exp(1000)` is `+inf`)
/// and loses all precision for very negative ones (`exp(-1000)` is `0`, whose
/// `ln` is `-inf`). Subtracting the maximum before exponentiating fixes both:
/// the largest term becomes `exp(0) = 1` and every other term is in
/// `(0, 1]`, so the sum is well scaled and the result is `max + ln(sum)`.
///
/// Edge cases:
/// * an empty slice returns `f64::NEG_INFINITY` (the log of an empty sum,
///   i.e. `log(0)`);
/// * if the maximum is `-inf` (every value is `-inf`, i.e. every term has
///   probability zero) the result is `-inf`;
/// * if the maximum is `+inf` the result is `+inf`.
///
/// The result always satisfies `result >= max(values)`.
#[must_use]
pub(crate) fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // A `-inf` maximum means every term is `-inf`; a `+inf` maximum dominates
    // the sum. Returning `max` is correct in both cases and, crucially, avoids
    // the `(-inf) - (-inf) = NaN` that the general path would produce.
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Natural log of `likelihood`, clamped up to a strictly positive floor first.
///
/// A raw likelihood of `0.0` has `ln(0) = -inf`; adding that to a
/// hypothesis's log-probability drives it to `-inf` permanently — no amount
/// of later favorable evidence (which only ever *adds* finite quantities) can
/// climb back out. Flooring the likelihood to `min_likelihood` before taking
/// the log replaces that trap with a large-but-finite negative log-likelihood
/// that a subsequent update *can* recover from.
///
/// If `min_likelihood` is itself non-positive (a misconfiguration) the floor
/// falls back to `f64::MIN_POSITIVE`, so the returned value is always finite.
#[must_use]
pub(crate) fn log_likelihood_floored(likelihood: f64, min_likelihood: f64) -> f64 {
    let floor = if min_likelihood > 0.0 {
        min_likelihood
    } else {
        f64::MIN_POSITIVE
    };
    likelihood.max(floor).ln()
}

/// Renormalize `log_probs` in place so that `logsumexp(log_probs) == 0`, i.e.
/// so the exponentiated distribution sums to `1`.
///
/// This subtracts `logsumexp(log_probs)` from every entry. When the running
/// normalizer is not finite (a degenerate all-`-inf` distribution, which the
/// likelihood floor is designed to prevent) the entries are left untouched
/// rather than turned into `NaN`.
pub(crate) fn normalize_log_probs(log_probs: &mut [f64]) {
    let z = logsumexp(log_probs);
    if z.is_finite() {
        for lp in log_probs.iter_mut() {
            *lp -= z;
        }
    }
}

/// Apply one Bayesian update in log space: add each hypothesis's
/// log-likelihood to its log-probability, then renormalize.
///
/// This is the log-space form of Bayes' rule
/// `posterior(h) ∝ prior(h) * P(e | h)`: multiplication becomes addition, and
/// the proportionality is restored by the [`normalize_log_probs`] step. The
/// two slices are walked pairwise; if their lengths differ, the shorter one
/// bounds the update (the caller guarantees equal length).
pub(crate) fn bayes_update_log(log_probs: &mut [f64], log_likelihoods: &[f64]) {
    for (lp, &ll) in log_probs.iter_mut().zip(log_likelihoods.iter()) {
        *lp += ll;
    }
    normalize_log_probs(log_probs);
}

/// Convert log-probabilities to a normalized linear probability distribution.
///
/// The result is normalized defensively — each entry is `exp(lp - logsumexp)`
/// — so it sums to `1` (within floating-point tolerance) even if the input
/// was not already normalized. A degenerate all-`-inf` input yields a uniform
/// distribution, keeping the contract that the returned vector is always a
/// valid probability distribution.
#[must_use]
pub(crate) fn log_probs_to_linear(log_probs: &[f64]) -> Vec<f64> {
    let z = logsumexp(log_probs);
    if z.is_finite() {
        return log_probs.iter().map(|&lp| (lp - z).exp()).collect();
    }
    let n = log_probs.len();
    if n == 0 {
        return Vec::new();
    }
    #[allow(clippy::cast_precision_loss)]
    let uniform = 1.0 / n as f64;
    vec![uniform; n]
}

/// Shannon entropy (in nats) of the distribution encoded by `log_probs`.
///
/// Defined as `-sum_i p_i * ln(p_i)` with `p_i` the normalized linear
/// probabilities. It is `0` when one hypothesis holds all the mass and
/// `ln(k)` when the `k` hypotheses are equiprobable, so it is a convenient
/// scalar summary of how much uncertainty remains in a belief state.
#[must_use]
pub(crate) fn entropy_nats(log_probs: &[f64]) -> f64 {
    let z = logsumexp(log_probs);
    if !z.is_finite() {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &lp in log_probs {
        let normalized = lp - z;
        let prob = normalized.exp();
        if prob > 0.0 {
            entropy -= prob * normalized;
        }
    }
    entropy
}

/// A pairwise **log-likelihood-ratio** (log Bayes factor) between two
/// hypotheses under a single piece of evidence.
///
/// This is the log-odds form of the Bayesian update. For two hypotheses
/// `h_num` and `h_den`, Bayes' rule in odds form reads
///
/// ```text
/// posterior_odds = prior_odds * (P(e | h_num) / P(e | h_den))
/// ```
///
/// Taking logs turns the multiplicative Bayes factor
/// `P(e | h_num) / P(e | h_den)` into the additive
/// `log_ratio = ln P(e | h_num) - ln P(e | h_den)` stored here, so
/// `posterior_log_odds = prior_log_odds + log_ratio` (see
/// [`apply_to_log_odds`](Self::apply_to_log_odds)). A positive `log_ratio`
/// means the evidence favors the numerator hypothesis; a negative one favors
/// the denominator; zero is uninformative between the two.
#[derive(Debug, Clone, PartialEq)]
pub struct LikelihoodRatio {
    /// Identifier of the numerator hypothesis `h_num`.
    pub numerator_id: String,
    /// Identifier of the denominator hypothesis `h_den`.
    pub denominator_id: String,
    /// The natural-log Bayes factor `ln P(e | h_num) - ln P(e | h_den)`.
    pub log_ratio: f64,
}

impl LikelihoodRatio {
    /// Construct a ratio directly from a precomputed `log_ratio`.
    #[must_use]
    pub fn new(
        numerator_id: impl Into<String>,
        denominator_id: impl Into<String>,
        log_ratio: f64,
    ) -> Self {
        Self {
            numerator_id: numerator_id.into(),
            denominator_id: denominator_id.into(),
            log_ratio,
        }
    }

    /// Construct a ratio from the two raw likelihoods
    /// `P(e | h_num)` and `P(e | h_den)`.
    ///
    /// Both likelihoods are clamped to `min_likelihood` before their logs are
    /// taken (via `log_likelihood_floored`), so a zero likelihood on either
    /// side yields a large-but-finite `log_ratio` rather than `+/-inf`.
    #[must_use]
    pub fn from_likelihoods(
        numerator_id: impl Into<String>,
        denominator_id: impl Into<String>,
        numerator_likelihood: f64,
        denominator_likelihood: f64,
        min_likelihood: f64,
    ) -> Self {
        let log_num = log_likelihood_floored(numerator_likelihood, min_likelihood);
        let log_den = log_likelihood_floored(denominator_likelihood, min_likelihood);
        Self {
            numerator_id: numerator_id.into(),
            denominator_id: denominator_id.into(),
            log_ratio: log_num - log_den,
        }
    }

    /// The Bayes factor in linear space, `exp(log_ratio) = P(e | h_num) / P(e | h_den)`.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.log_ratio.exp()
    }

    /// Apply this log Bayes factor to a prior log-odds, returning the
    /// posterior log-odds `prior_log_odds + log_ratio`.
    #[must_use]
    pub fn apply_to_log_odds(&self, prior_log_odds: f64) -> f64 {
        prior_log_odds + self.log_ratio
    }

    /// `true` when the evidence favors the numerator hypothesis
    /// (`log_ratio > 0`).
    #[must_use]
    pub fn favors_numerator(&self) -> bool {
        self.log_ratio > 0.0
    }

    /// `true` when the evidence favors the denominator hypothesis
    /// (`log_ratio < 0`).
    #[must_use]
    pub fn favors_denominator(&self) -> bool {
        self.log_ratio < 0.0
    }

    /// `true` when the evidence is exactly uninformative between the two
    /// hypotheses (`log_ratio == 0`, i.e. a Bayes factor of `1`).
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn is_neutral(&self) -> bool {
        self.log_ratio == 0.0
    }
}
