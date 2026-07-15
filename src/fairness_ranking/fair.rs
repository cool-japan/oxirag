//! `FA*IR`: ranked group fairness (Zehlike, Bonald, Sun, Capannini, Hauff,
//! Megahed, Baeza-Yates — `CIKM` 2017).
//!
//! # The criterion
//!
//! A ranking is *ranked-group-fair* with respect to a target proportion `p` and a
//! significance `alpha` if, **at every prefix**, the number of protected
//! candidates in the top-`k` is not significantly below what a `Bin(k, p)` null
//! model would produce. The decision boundary at prefix `k` is the binomial
//! quantile
//!
//! ```text
//! m_alpha(k) = min { m : P(Bin(k, p) <= m) >= alpha }
//! ```
//!
//! and the ranking is fair exactly when `count(k) >= m_alpha(k)` for every `k` in
//! `1..=K`. The vector of those quantiles is the **`m-table`**.
//!
//! Two things about `m_alpha` are worth pinning down, because both have a
//! plausible-looking wrong version.
//!
//! **The quantile is a `<=` quantile.** The equivalent strict form is
//! `m_alpha(k) = max { m : P(Bin(k, p) < m) <= alpha }`; the two agree. The
//! mixture `min { m : P(Bin(k, p) < m) >= alpha }` is off by one, and the off-by-one
//! is not cosmetic — it gives `m_alpha(1) = 1` for `p = 0.5, alpha = 0.1`, which
//! *forces the top-1 slot to be protected* when the correct answer is
//! `m_alpha(1) = 0` (the top slot is not a fairness violation, because a single
//! Bernoulli trial cannot be significantly unbalanced at `alpha = 0.1`). This
//! module's tests pin `m_alpha(1) = 0`.
//!
//! **The `m-table` is non-decreasing and 1-Lipschitz.** `m_alpha(k+1)` is either
//! `m_alpha(k)` or `m_alpha(k) + 1`, never more. Non-decreasing because
//! `P(Bin(k+1, p) <= m) <= P(Bin(k, p) <= m)` — one more trial can only make it
//! harder to stay at or below `m`. Bounded above because
//! `X_{k+1} = X_k + B` with `B` in `{0, 1}`, so `X_k <= m` implies
//! `X_{k+1} <= m + 1`, giving `P(Bin(k+1,p) <= m+1) >= P(Bin(k,p) <= m) >= alpha`.
//! That is what makes [`fair_top_k`] work: a ranking's protected count also grows
//! by at most one per position, so the table can always be *kept up with*, one
//! forced protected pick at a time, and the only way to fail is to run out of
//! protected candidates altogether.
//!
//! # The correction, which is the actual hard part
//!
//! The criterion applies the test at **every one of `K` prefixes**. Running `K`
//! tests each at level `alpha` does not give a procedure with family-wise error
//! `alpha`; it gives one with a far larger error, and a perfectly fair ranking
//! drawn from the null model would be rejected far more often than `alpha` of the
//! time. `FA*IR`'s contribution is to correct for this **exactly**, and that is
//! what [`adjusted_significance`] does.
//!
//! Under the null, the protected count is a Bernoulli(`p`) random walk
//! `count(1), ..., count(K)`, and the family-wise failure probability of a table
//! `M` is
//!
//! ```text
//! P[ there exists k with count(k) < M(k) ]
//! ```
//!
//! [`failure_probability`] computes that **exactly**, by a forward recursion over
//! the distribution of the walk that *deletes* the mass which has already
//! violated the table:
//!
//! ```text
//! f(0, 0) = 1
//! f(k, c) = [ f(k-1, c-1) * p  +  f(k-1, c) * (1 - p) ] * [ c >= M(k) ]
//! fail    = 1 - sum_c f(K, c)
//! ```
//!
//! The indicator is the whole trick: a path that dips below the table at prefix
//! `k` is killed *there*, so it cannot be resurrected by climbing back above the
//! table later — which is exactly right, because the criterion demands the
//! condition at every prefix, and one violation condemns the whole ranking. The
//! recursion is `O(K^2)` and involves no cancellation (every term is a
//! non-negative probability), so it is exact to a few ulps.
//!
//! `failure_probability` is non-decreasing in `alpha_c` — raising the per-prefix
//! significance can only raise `m-table` entries, which can only kill more paths —
//! so [`adjusted_significance`] bisects for the **largest** `alpha_c <= alpha`
//! whose family-wise failure probability is still at most `alpha`. Returning the
//! largest such value rather than the smallest matters: the correction must not
//! be more conservative than it has to be, or the test loses the power to detect
//! the discrimination it exists to detect.
//!
//! None of this can be done with a normal approximation. See
//! [`binomial`](super::binomial).

use serde::{Deserialize, Serialize};

use super::binomial::binomial_quantile;
use super::types::{FairnessError, FairnessResult};

/// How many bisection steps [`adjusted_significance`] takes.
///
/// The family-wise failure probability is a **step function** of `alpha_c` — it
/// only changes when some `m-table` entry ticks over — so bisection is converging
/// on the location of a discontinuity, not on a root of a smooth function. Sixty
/// halvings shrink the bracket by `2^-60`, far below the width of any step a
/// ranking of realistic length produces, and the loop also stops early once the
/// bracket is too narrow for `f64` to split.
const BISECTION_STEPS: usize = 60;

/// The `m-table` of `FA*IR`: the minimum number of protected candidates required
/// in each prefix of a ranking, together with the corrected significance it was
/// built from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MTable {
    /// `required[k]` is `m_alpha(k)`, the minimum protected count in the top-`k`.
    /// `required[0]` is `0` and exists only so the vector can be indexed by the
    /// 1-based prefix length. Length `k + 1`.
    required: Vec<usize>,
    /// The target proportion `p`.
    target_proportion: f64,
    /// The significance the caller asked for.
    significance: f64,
    /// The per-prefix significance actually used, after the multiple-test
    /// correction. Never greater than `significance`.
    adjusted_significance: f64,
    /// The family-wise probability that a ranking drawn from the null model fails
    /// this table at some prefix. Must not exceed `significance`; that is the
    /// property the correction exists to guarantee.
    failure_probability: f64,
}

impl MTable {
    /// Build the corrected `m-table` for a ranking of `k` positions.
    ///
    /// This is the constructor to use. It derives the adjusted per-prefix
    /// significance from `significance` by the exact multiple-test correction
    /// described in the [module documentation](self), which costs
    /// `O(BISECTION_STEPS * k^2)`; the result is cached in the returned value, so
    /// the cost is paid once per `(k, p, alpha)`.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::InvalidProportion`] — `target_proportion` outside `(0, 1)`.
    /// * [`FairnessError::InvalidSignificance`] — `significance` outside `(0, 1)`.
    /// * [`FairnessError::EmptyRanking`] — `k == 0`.
    pub fn new(k: usize, target_proportion: f64, significance: f64) -> FairnessResult<Self> {
        validate(k, target_proportion, significance)?;
        let adjusted = adjusted_significance(k, target_proportion, significance);
        let required = raw_table(k, target_proportion, adjusted);
        let failure = failure_probability(&required, target_proportion);
        Ok(Self {
            required,
            target_proportion,
            significance,
            adjusted_significance: adjusted,
            failure_probability: failure,
        })
    }

    /// Build the **uncorrected** `m-table`, applying `significance` at each
    /// prefix without adjusting for the fact that there are `k` prefixes.
    ///
    /// This is the naive procedure, and it is here so that the correction can be
    /// *measured against* it: its family-wise failure probability, reported by
    /// [`MTable::failure_probability`], overshoots `significance` — for `k = 100,
    /// p = 0.5, alpha = 0.1` it is several times the nominal level. Use
    /// [`MTable::new`] for anything real.
    ///
    /// # Errors
    ///
    /// As [`MTable::new`].
    pub fn unadjusted(k: usize, target_proportion: f64, significance: f64) -> FairnessResult<Self> {
        validate(k, target_proportion, significance)?;
        let required = raw_table(k, target_proportion, significance);
        let failure = failure_probability(&required, target_proportion);
        Ok(Self {
            required,
            target_proportion,
            significance,
            adjusted_significance: significance,
            failure_probability: failure,
        })
    }

    /// The minimum protected count required in the top-`prefix`.
    ///
    /// A `prefix` past the end of the table returns the last entry — the strictest
    /// requirement the table states — rather than zero, which would silently turn
    /// off the constraint exactly where a caller has over-run it.
    #[must_use]
    pub fn required(&self, prefix: usize) -> usize {
        match self.required.get(prefix) {
            Some(&value) => value,
            None => self.required.last().copied().unwrap_or(0),
        }
    }

    /// The whole table, indexed by prefix length; entry `0` is `0`.
    #[must_use]
    pub fn entries(&self) -> &[usize] {
        &self.required
    }

    /// The ranking length the table was built for.
    #[must_use]
    pub fn k(&self) -> usize {
        self.required.len().saturating_sub(1)
    }

    /// The target proportion `p`.
    #[must_use]
    pub fn target_proportion(&self) -> f64 {
        self.target_proportion
    }

    /// The significance the caller asked for.
    #[must_use]
    pub fn significance(&self) -> f64 {
        self.significance
    }

    /// The per-prefix significance after the multiple-test correction.
    #[must_use]
    pub fn adjusted_significance(&self) -> f64 {
        self.adjusted_significance
    }

    /// The exact probability that a ranking drawn from the `Bin(k, p)` null model
    /// violates this table at some prefix — the table's family-wise Type-I error.
    #[must_use]
    pub fn failure_probability(&self) -> f64 {
        self.failure_probability
    }

    /// How many protected candidates the table demands in the full top-`k`.
    #[must_use]
    pub fn total_protected_required(&self) -> usize {
        self.required.last().copied().unwrap_or(0)
    }
}

/// Validate the parameters shared by every `FA*IR` entry point.
fn validate(k: usize, target_proportion: f64, significance: f64) -> FairnessResult<()> {
    if k == 0 {
        return Err(FairnessError::EmptyRanking {
            action: "build an m-table for",
        });
    }
    if !target_proportion.is_finite() || target_proportion <= 0.0 || target_proportion >= 1.0 {
        return Err(FairnessError::InvalidProportion {
            value: target_proportion,
        });
    }
    if !significance.is_finite() || significance <= 0.0 || significance >= 1.0 {
        return Err(FairnessError::InvalidSignificance {
            value: significance,
        });
    }
    Ok(())
}

/// The `m-table` for a *given* per-prefix significance, with no correction:
/// `required[i] = min { m : P(Bin(i, p) <= m) >= alpha_c }`.
#[must_use]
#[allow(clippy::needless_range_loop)] // `prefix` is both the index and the trial count.
pub fn raw_table(k: usize, target_proportion: f64, per_prefix_significance: f64) -> Vec<usize> {
    let mut required = vec![0usize; k + 1];
    for prefix in 1..=k {
        required[prefix] = binomial_quantile(per_prefix_significance, prefix, target_proportion);
    }
    required
}

/// The **exact** probability that a ranking drawn from the null model — each
/// position independently protected with probability `p` — violates `required` at
/// some prefix.
///
/// `required` is indexed by prefix length, with `required[0]` ignored, exactly as
/// [`MTable::entries`] returns it.
///
/// The forward recursion is derived in the [module documentation](self). It is
/// `O(k^2)` and free of cancellation: every intermediate is a probability, all
/// non-negative, and they are only ever added or zeroed. The single subtraction
/// is the final `1 - survive`, which loses absolute precision only when the
/// failure probability is already far below any significance level anyone would
/// choose.
#[must_use]
#[allow(clippy::needless_range_loop)] // `prefix` indexes `required` and drives the walk length.
pub fn failure_probability(required: &[usize], target_proportion: f64) -> f64 {
    let k = required.len().saturating_sub(1);
    if k == 0 {
        return 0.0;
    }
    if !target_proportion.is_finite() || !(0.0..=1.0).contains(&target_proportion) {
        return f64::NAN;
    }
    let protected = target_proportion;
    let unprotected = 1.0 - target_proportion;

    // `alive[c]` = P(the walk has c protected so far AND has never yet fallen
    // below the table).
    let mut alive = vec![0.0f64; k + 1];
    alive[0] = 1.0;

    for prefix in 1..=k {
        // One Bernoulli step, in place. Descending, so `alive[c - 1]` is still the
        // *previous* prefix's value when it is read.
        for count in (0..=prefix).rev() {
            let from_protected = if count > 0 {
                alive[count - 1] * protected
            } else {
                0.0
            };
            let from_unprotected = alive[count] * unprotected;
            alive[count] = from_protected + from_unprotected;
        }
        // Delete the mass that has just violated the table. A path killed here is
        // never revived: that is what makes this a family-wise probability rather
        // than a per-prefix one.
        let floor = required[prefix].min(prefix + 1);
        for slot in alive.iter_mut().take(floor) {
            *slot = 0.0;
        }
    }

    let survive: f64 = alive.iter().sum();
    (1.0 - survive).clamp(0.0, 1.0)
}

/// The corrected per-prefix significance `alpha_c`: the **largest**
/// `alpha_c <= alpha` whose `m-table` has family-wise failure probability at most
/// `alpha`.
///
/// See the [module documentation](self) for why a correction is needed at all and
/// why "largest" is the right side of the boundary to land on. The failure
/// probability is a non-decreasing step function of `alpha_c`, with value `0` at
/// `alpha_c = 0` (the all-zero table constrains nothing), so the bracket
/// `[0, alpha]` always contains an admissible point and the bisection always has
/// somewhere to go.
///
/// When the uncorrected table already controls the family-wise error — which
/// happens for short rankings, where there are too few prefixes for the
/// multiplicity to bite — no correction is applied and `alpha` is returned
/// unchanged.
#[must_use]
pub fn adjusted_significance(k: usize, target_proportion: f64, significance: f64) -> f64 {
    let failure_at_nominal = failure_probability(
        &raw_table(k, target_proportion, significance),
        target_proportion,
    );
    if failure_at_nominal <= significance {
        return significance;
    }

    // Invariant: `admissible` always has failure probability <= significance
    // (trivially true at 0.0, where the table is all zeros), and `excessive`
    // always exceeds it.
    let mut admissible = 0.0f64;
    let mut excessive = significance;
    for _ in 0..BISECTION_STEPS {
        let midpoint = 0.5 * (admissible + excessive);
        if midpoint <= admissible || midpoint >= excessive {
            // The bracket is narrower than `f64` can split; the step has been
            // located as precisely as the type allows.
            break;
        }
        let failure = failure_probability(
            &raw_table(k, target_proportion, midpoint),
            target_proportion,
        );
        if failure <= significance {
            admissible = midpoint;
        } else {
            excessive = midpoint;
        }
    }
    admissible
}

/// The protected count in every prefix of `protected`, as
/// `counts[k] = |{ i < k : protected[i] }|`. Length `protected.len() + 1`.
#[must_use]
pub fn prefix_protected_counts(protected: &[bool]) -> Vec<usize> {
    let mut counts = vec![0usize; protected.len() + 1];
    let mut running = 0usize;
    for (index, &is_protected) in protected.iter().enumerate() {
        if is_protected {
            running += 1;
        }
        counts[index + 1] = running;
    }
    counts
}

/// The outcome of testing a ranking against an [`MTable`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FairnessAudit {
    /// Whether every prefix met its requirement.
    pub satisfied: bool,
    /// The **1-based** length of the shortest prefix that failed, if any. The
    /// shortest one is the informative one: it is where the ranking first went
    /// wrong, and everything after it is downstream of that.
    pub first_violation: Option<usize>,
    /// How many protected candidates the first failing prefix was short of.
    pub shortfall: usize,
    /// The requirement at each prefix, from the `m-table`.
    pub required: Vec<usize>,
    /// The observed protected count at each prefix.
    pub observed: Vec<usize>,
    /// The corrected per-prefix significance the table was built with.
    pub adjusted_significance: f64,
}

/// Test a ranking against an [`MTable`].
///
/// `protected` is the ranking's protected flags in **ranked order**. Only the
/// first `table.k()` positions are tested — the table is defined for exactly that
/// many prefixes, and testing further would be testing a hypothesis nobody
/// corrected for.
#[must_use]
#[allow(clippy::needless_range_loop)] // `prefix` indexes both `observed` and the table.
pub fn audit_ranking(protected: &[bool], table: &MTable) -> FairnessAudit {
    let depth = table.k().min(protected.len());
    let observed = prefix_protected_counts(&protected[..depth]);

    let mut first_violation = None;
    let mut shortfall = 0usize;
    for prefix in 1..=depth {
        let needed = table.required(prefix);
        if observed[prefix] < needed {
            first_violation = Some(prefix);
            shortfall = needed - observed[prefix];
            break;
        }
    }

    FairnessAudit {
        satisfied: first_violation.is_none(),
        first_violation,
        shortfall,
        required: table.entries()[..=depth].to_vec(),
        observed,
        adjusted_significance: table.adjusted_significance(),
    }
}

/// The `FA*IR` ranking algorithm: the **most useful** ranking of `k` positions
/// that satisfies the ranked group fairness criterion.
///
/// `protected_order` and `unprotected_order` are the two groups' candidate
/// indices, each already sorted by **descending qualification** (in-group
/// monotonicity is a hard requirement of the criterion: a fair ranking may never
/// promote a *worse* member of a group over a better one, since that would be
/// discrimination inside the group in the name of fixing discrimination between
/// groups). `scores` gives every candidate's qualification, indexed by candidate.
///
/// The algorithm is greedy, and the greed is *provably* optimal here (Zehlike et
/// al., Theorem 1): at each position, if the table's requirement is about to be
/// missed, take the best remaining protected candidate; otherwise take the better
/// of the two groups' heads. Because the `m-table` is 1-Lipschitz (see the
/// [module documentation](self)) at most one protected pick is ever forced at a
/// time, so the greedy choice never paints itself into a corner, and because
/// in-group order is fixed, the only freedom the ranking has is *when* to
/// interleave — which is what the greedy step resolves in favour of utility.
///
/// Returns the candidate indices in ranked order.
///
/// # Errors
///
/// [`FairnessError::InsufficientProtected`] when the pool has fewer protected
/// candidates than the table demands in the top-`k`. There is genuinely no fair
/// ranking in that case, and returning the closest unfair one would be exactly the
/// silent failure this module exists to prevent.
pub fn fair_top_k(
    protected_order: &[usize],
    unprotected_order: &[usize],
    scores: &[f64],
    table: &MTable,
) -> FairnessResult<Vec<usize>> {
    let k = table
        .k()
        .min(protected_order.len() + unprotected_order.len());
    let needed = table.required(k);
    if protected_order.len() < needed {
        return Err(FairnessError::InsufficientProtected {
            required: needed,
            available: protected_order.len(),
        });
    }

    let mut ranking = Vec::with_capacity(k);
    let mut next_protected = 0usize;
    let mut next_unprotected = 0usize;
    let mut protected_placed = 0usize;

    for position in 1..=k {
        let requirement = table.required(position);
        let protected_head = protected_order.get(next_protected).copied();
        let unprotected_head = unprotected_order.get(next_unprotected).copied();

        // Forced move: this prefix's requirement is about to be missed. Because
        // the table is 1-Lipschitz, at most one protected pick is ever forced at
        // a time, and the up-front check guarantees one is left — but if that
        // invariant were ever broken, say so instead of emitting a ranking that
        // quietly violates the criterion it claims to enforce.
        if protected_placed < requirement {
            let Some(candidate) = protected_head else {
                return Err(FairnessError::InsufficientProtected {
                    required: requirement,
                    available: protected_order.len(),
                });
            };
            ranking.push(candidate);
            next_protected += 1;
            protected_placed += 1;
            continue;
        }

        // Free move: take whichever head is more qualified. Ties break toward the
        // protected candidate — at equal qualification, promoting it costs no
        // utility at all and relieves pressure on every later prefix.
        match (protected_head, unprotected_head) {
            (None, None) => break,
            (Some(candidate), None) => {
                ranking.push(candidate);
                next_protected += 1;
                protected_placed += 1;
            }
            (None, Some(candidate)) => {
                ranking.push(candidate);
                next_unprotected += 1;
            }
            (Some(protected_candidate), Some(unprotected_candidate)) => {
                let protected_score = scores.get(protected_candidate).copied().unwrap_or(0.0);
                let unprotected_score = scores.get(unprotected_candidate).copied().unwrap_or(0.0);
                if protected_score >= unprotected_score {
                    ranking.push(protected_candidate);
                    next_protected += 1;
                    protected_placed += 1;
                } else {
                    ranking.push(unprotected_candidate);
                    next_unprotected += 1;
                }
            }
        }
    }

    Ok(ranking)
}
