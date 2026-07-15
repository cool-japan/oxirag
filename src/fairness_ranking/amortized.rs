//! **Amortized** equity of attention (Biega, Gummadi & Weikum, `SIGIR` 2018):
//! fairness that no single ranking can deliver, and that a *sequence* of rankings
//! can.
//!
//! # Why one ranking is not enough
//!
//! Take two documents of exactly equal relevance. One of them must go first. The
//! first position pays out `1 / log2(2) = 1.0` of attention and the second
//! `1 / log2(3) = 0.63`, so whichever document is chosen receives 58% more
//! attention than its equally-deserving twin. **No ranking of that query is fair**,
//! and no amount of cleverness inside a single ranking can make one — the
//! positions are discrete, their payouts are unequal, and somebody has to take the
//! smaller one. This is not a defect in the ranker; it is a property of ranked
//! presentation itself.
//!
//! Every other fairness construct in this module inherits that impossibility.
//! `FA*IR` ([`fair`](super::fair)) constrains one ranking. `DELTR`
//! ([`deltr`](super::deltr)) trains a scorer that produces less-unfair single
//! rankings. Neither can equalize two identical documents, because there is
//! nothing left to equalize *within*.
//!
//! Biega et al.'s move is to change the unit of accounting. Fairness is not a
//! property of a ranking; it is a property of a **sequence** of rankings, and the
//! twins are treated fairly if, over many queries, each spends about the same time
//! on top. Attention is a resource that is **amortized** across the sequence, and
//! the equity condition becomes
//!
//! ```text
//! A_i / A_j  =  R_i / R_j        for every pair of subjects i, j
//! ```
//!
//! where `A_i` is subject `i`'s **cumulative attention** and `R_i` its
//! **cumulative relevance**. Equivalently: `A_i / R_i` is the same for everyone,
//! which is precisely the exposure-per-unit-relevance parity that
//! [`ExposureMetrics::disparity`](super::ExposureMetrics::disparity) measures — now
//! measured over a lifetime rather than over a page.
//!
//! # Serving one query is an assignment problem
//!
//! At query `t`, with the cumulative state `A_i` and the updated cumulative
//! relevance `R_i' = R_i + r_i`, the ranking that best repairs the accumulated
//! inequity minimizes
//!
//! ```text
//! sum_i | A_i + a_{pos(i)} - R_i' |
//! ```
//!
//! where `a_j` is the attention paid out at position `j`. Each subject occupies
//! exactly one position, so its term depends on nothing but its own position: the
//! objective is **separable over the matched pairs**, and the whole problem is a
//! linear assignment problem with cost
//!
//! ```text
//! cost(position j, subject i) = | A_i + a_j - R_i' |
//! ```
//!
//! solved exactly by [`solve_assignment`]. Not greedily,
//! not approximately: the ranking returned is the global minimizer over all `n!/(n-k)!`
//! injections.
//!
//! ## Relevance and attention are put on one scale
//!
//! `A_i` and `R_i` are compared *directly* — subtracted from one another — so they
//! must be denominated in the same units, or the objective would be measuring an
//! arbitrary scale mismatch instead of an injustice. Both are therefore normalized
//! per query to sum to one: each query pays out one unit of attention, spread over
//! the `k` positions as `a_j / sum a`, and confers one unit of relevance, spread
//! over the `n` subjects as `r_i / sum r`. After `T` queries both cumulative totals
//! are exactly `T`, and equity means `A_i = R_i`. The *shape* of the position
//! discount is untouched by the normalization; only its total is fixed.
//!
//! ## Relevance accrues whether or not you were shown
//!
//! `R_i` is incremented for **every** subject, including the `n - k` that did not
//! make the page. That is the point: relevance is what you *deserved*, attention is
//! what you *got*, and a subject that was relevant and passed over is exactly the
//! subject the next query owes something to. Incrementing `R_i` only for the shown
//! subjects would make the metric self-fulfilling — the unshown would never be
//! owed anything, and the greedy relevance ranking would score as perfectly fair.
//!
//! ## The quality trade-off
//!
//! Pure equity would happily seat an irrelevant subject at the top to settle a
//! debt. Biega et al. bound this with an explicit `NDCG` constraint inside an
//! integer program. An `NDCG` floor is not separable over matched pairs, so adding
//! it would destroy the very structure that makes the problem an assignment
//! problem; this module takes the **Lagrangian** view instead and blends the two
//! objectives inside the cost:
//!
//! ```text
//! cost(j, i) = (1 - lambda) * | A_i + a_j - R_i' |  -  lambda * r_i * a_j
//! ```
//!
//! The second term is the negated `DCG` contribution of seating `i` at `j`, so
//! minimizing it maximizes quality. This stays a linear assignment problem, and
//! `lambda` sweeps the whole frontier: at `lambda = 0` it is Biega's unconstrained
//! equity objective, and at `lambda = 1` the rearrangement inequality makes the
//! optimal assignment **exactly the relevance sort** — which is a closed-form
//! answer this module's tests use to check the solver against.
//!
//! Being a relaxation rather than the hard constraint is a real difference from the
//! paper, and it is stated here rather than papered over: `lambda` controls the
//! trade-off, it does not *guarantee* an `NDCG` floor.

use serde::{Deserialize, Serialize};

use super::assignment::solve_assignment;
use super::exposure::{GroupExposure, exposure_at_rank, fairness_ndcg_at_k};
use super::types::{FairnessError, FairnessResult, GroupId};

/// A snapshot of cumulative equity after some number of queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExposureReport {
    /// How many queries have been served.
    pub queries: usize,
    /// Cumulative attention and relevance per group.
    pub groups: Vec<GroupExposure>,
    /// `max - min` of the per-group cumulative attention-per-unit-relevance. This
    /// is the number that must fall as the sequence lengthens; a single ranking
    /// cannot drive it to zero, and that is the entire thesis of this file.
    pub group_gap: f64,
    /// `sum_i |A_i - R_i|`, the individual `L1` inequity Biega et al. minimize.
    pub individual_inequity: f64,
    /// The mean `nDCG` of the rankings served so far, against the relevances of the
    /// query they were serving. The price paid for the equity above.
    pub mean_ndcg: f64,
}

/// Amortized equity of attention across a sequence of queries over a fixed set of
/// subjects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquityOfAttention {
    /// Each subject's group.
    groups: Vec<GroupId>,
    /// How many groups the attribute has.
    group_count: usize,
    /// How many positions each ranking has.
    positions: usize,
    /// Cumulative normalized attention per subject.
    attention: Vec<f64>,
    /// Cumulative normalized relevance per subject.
    relevance: Vec<f64>,
    /// The Lagrangian weight on the quality term, in `[0, 1]`.
    relevance_weight: f64,
    /// How many queries have been served.
    queries: usize,
    /// The sum of the served rankings' `nDCG`, for the mean in the report.
    ndcg_total: f64,
}

impl EquityOfAttention {
    /// Start a sequence over the subjects labelled by `groups`, serving
    /// `positions` slots per query.
    ///
    /// `relevance_weight` is the Lagrangian `lambda` of the
    /// [module documentation](self): `0.0` is pure equity, `1.0` is pure relevance
    /// (and reduces the solver to a relevance sort).
    ///
    /// # Errors
    ///
    /// * [`FairnessError::EmptyRanking`] — no subjects, or no positions.
    /// * [`FairnessError::UnknownGroup`] — a group label is out of range.
    /// * [`FairnessError::InvalidConfig`] — more positions than subjects (there
    ///   would be nothing to allocate to the surplus slots), or a
    ///   `relevance_weight` outside `[0, 1]`.
    pub fn new(
        groups: Vec<GroupId>,
        group_count: usize,
        positions: usize,
        relevance_weight: f64,
    ) -> FairnessResult<Self> {
        if groups.is_empty() {
            return Err(FairnessError::EmptyRanking {
                action: "amortize attention over",
            });
        }
        if positions == 0 {
            return Err(FairnessError::EmptyRanking {
                action: "allocate attention across",
            });
        }
        if positions > groups.len() {
            return Err(FairnessError::InvalidConfig {
                reason: format!(
                    "{positions} positions but only {} subjects to fill them",
                    groups.len()
                ),
            });
        }
        if !relevance_weight.is_finite() || !(0.0..=1.0).contains(&relevance_weight) {
            return Err(FairnessError::InvalidConfig {
                reason: format!("relevance weight {relevance_weight} is outside [0, 1]"),
            });
        }
        for group in &groups {
            if group.index() >= group_count {
                return Err(FairnessError::UnknownGroup {
                    group: group.index(),
                    groups: group_count,
                });
            }
        }

        let subjects = groups.len();
        Ok(Self {
            groups,
            group_count,
            positions,
            attention: vec![0.0; subjects],
            relevance: vec![0.0; subjects],
            relevance_weight,
            queries: 0,
            ndcg_total: 0.0,
        })
    }

    /// How many subjects the sequence is over.
    #[must_use]
    pub fn subjects(&self) -> usize {
        self.groups.len()
    }

    /// How many positions each ranking has.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// How many queries have been served.
    #[must_use]
    pub fn queries(&self) -> usize {
        self.queries
    }

    /// Each subject's cumulative normalized attention.
    #[must_use]
    pub fn subject_attention(&self) -> &[f64] {
        &self.attention
    }

    /// Each subject's cumulative normalized relevance.
    #[must_use]
    pub fn subject_relevance(&self) -> &[f64] {
        &self.relevance
    }

    /// Serve one query **amortizing** the accumulated inequity: solve the
    /// assignment problem of the [module documentation](self) and fold the realized
    /// attention into the cumulative state.
    ///
    /// Returns the ranking as the subject index at each of the `positions` slots.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::DimensionMismatch`] — `relevances` is not one per subject.
    /// * [`FairnessError::NonFinite`] — a relevance is `NaN` or infinite.
    /// * [`FairnessError::InvalidConfig`] — a negative relevance.
    /// * [`FairnessError::ZeroRelevance`] — every relevance is zero, so the
    ///   normalization that puts attention and relevance on one scale is undefined.
    /// * [`FairnessError::Assignment`] — the solver rejected the cost matrix.
    pub fn serve(&mut self, relevances: &[f64]) -> FairnessResult<Vec<usize>> {
        let shares = self.relevance_shares(relevances)?;
        let attention_units = self.attention_units();

        // cost(position j, subject i)
        //   = (1 - lambda) * | A_i + a_j - (R_i + r_i) |  -  lambda * r_i * a_j
        //
        // Rows are positions, columns are subjects, so `rows <= cols` as the
        // solver requires and the `n - k` unmatched subjects are simply not shown.
        let subjects = self.subjects();
        let equity_weight = 1.0 - self.relevance_weight;
        let mut cost = vec![0.0f64; self.positions * subjects];
        for (position, &attention) in attention_units.iter().enumerate() {
            for subject in 0..subjects {
                let target = self.relevance[subject] + shares[subject];
                let inequity = (self.attention[subject] + attention - target).abs();
                let quality = shares[subject] * attention;
                cost[position * subjects + subject] =
                    equity_weight * inequity - self.relevance_weight * quality;
            }
        }

        let solution = solve_assignment(&cost, self.positions, subjects)?;
        let ranking = solution.column_of_row().to_vec();
        self.commit(&ranking, &shares, &attention_units, relevances);
        Ok(ranking)
    }

    /// Serve one query by **descending relevance**, the way an ordinary ranker
    /// would, and fold the result into the same cumulative state.
    ///
    /// This is the control the amortized sequence is measured against. It is the
    /// ranking that maximizes each query's `nDCG` in isolation, and it is exactly
    /// the ranking that cannot amortize: the same subjects win every query, so their
    /// attention-per-unit-relevance stays high forever while everybody else's stays
    /// at zero. Test (e) of this module asserts precisely that divergence.
    ///
    /// # Errors
    ///
    /// As [`EquityOfAttention::serve`], minus the assignment errors (there is no
    /// assignment problem to solve).
    pub fn serve_by_relevance(&mut self, relevances: &[f64]) -> FairnessResult<Vec<usize>> {
        let shares = self.relevance_shares(relevances)?;
        let attention_units = self.attention_units();

        let mut order: Vec<usize> = (0..self.subjects()).collect();
        order.sort_by(|&left, &right| {
            relevances[right]
                .partial_cmp(&relevances[left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(&right))
        });
        order.truncate(self.positions);

        self.commit(&order, &shares, &attention_units, relevances);
        Ok(order)
    }

    /// Fold a served ranking into the cumulative state.
    fn commit(
        &mut self,
        ranking: &[usize],
        shares: &[f64],
        attention_units: &[f64],
        relevances: &[f64],
    ) {
        for (position, &subject) in ranking.iter().enumerate() {
            if let (Some(slot), Some(&attention)) = (
                self.attention.get_mut(subject),
                attention_units.get(position),
            ) {
                *slot += attention;
            }
        }
        // Relevance accrues to *everyone*, shown or not. See the module docs.
        for (slot, &share) in self.relevance.iter_mut().zip(shares) {
            *slot += share;
        }

        let served: Vec<f64> = ranking
            .iter()
            .map(|&subject| relevances.get(subject).copied().unwrap_or(0.0))
            .collect();
        // nDCG of the served page against the *full* candidate pool's ideal, so
        // that leaving a highly relevant subject off the page is charged for.
        let mut padded = served;
        padded.resize(self.positions, 0.0);
        self.ndcg_total += ndcg_against_pool(&padded, relevances, self.positions);

        self.queries += 1;
    }

    /// The per-query normalized relevance shares, summing to one.
    fn relevance_shares(&self, relevances: &[f64]) -> FairnessResult<Vec<f64>> {
        if relevances.len() != self.subjects() {
            return Err(FairnessError::DimensionMismatch {
                expected: self.subjects(),
                actual: relevances.len(),
            });
        }
        let mut total = 0.0f64;
        for &relevance in relevances {
            if !relevance.is_finite() {
                return Err(FairnessError::NonFinite {
                    what: "relevance in an amortized query",
                });
            }
            if relevance < 0.0 {
                return Err(FairnessError::InvalidConfig {
                    reason: format!("relevance {relevance} is negative"),
                });
            }
            total += relevance;
        }
        if total <= 0.0 {
            return Err(FairnessError::ZeroRelevance);
        }
        Ok(relevances.iter().map(|value| value / total).collect())
    }

    /// The per-query normalized attention payouts, summing to one, preserving the
    /// shape of `1 / log2(1 + rank)`.
    fn attention_units(&self) -> Vec<f64> {
        let raw: Vec<f64> = (0..self.positions).map(exposure_at_rank).collect();
        let total: f64 = raw.iter().sum();
        if total <= 0.0 {
            return raw;
        }
        raw.iter().map(|value| value / total).collect()
    }

    /// Cumulative attention and relevance per group.
    #[must_use]
    pub fn group_equity(&self) -> Vec<GroupExposure> {
        let mut accumulated: Vec<GroupExposure> = (0..self.group_count)
            .map(|index| GroupExposure {
                group: GroupId(index),
                members: 0,
                exposure: 0.0,
                relevance: 0.0,
            })
            .collect();
        for (subject, &group) in self.groups.iter().enumerate() {
            if let Some(slot) = accumulated.get_mut(group.index()) {
                slot.members += 1;
                slot.exposure += self.attention[subject];
                slot.relevance += self.relevance[subject];
            }
        }
        accumulated
    }

    /// `max - min` of the per-group cumulative attention-per-unit-relevance.
    ///
    /// Zero means every group has received attention in exact proportion to the
    /// relevance it earned across the whole sequence. Groups with no accumulated
    /// relevance are skipped, and with fewer than two comparable groups the gap is
    /// zero.
    #[must_use]
    pub fn group_gap(&self) -> f64 {
        let ratios: Vec<f64> = self
            .group_equity()
            .iter()
            .filter_map(GroupExposure::exposure_per_relevance)
            .collect();
        if ratios.len() < 2 {
            return 0.0;
        }
        let highest = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lowest = ratios.iter().copied().fold(f64::INFINITY, f64::min);
        highest - lowest
    }

    /// `sum_i |A_i - R_i|`: the individual `L1` inequity Biega et al. minimize.
    #[must_use]
    pub fn individual_inequity(&self) -> f64 {
        self.attention
            .iter()
            .zip(&self.relevance)
            .map(|(attention, relevance)| (attention - relevance).abs())
            .sum()
    }

    /// A snapshot of the sequence so far.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Query counts.
    pub fn report(&self) -> ExposureReport {
        let mean_ndcg = if self.queries == 0 {
            0.0
        } else {
            self.ndcg_total / (self.queries as f64)
        };
        ExposureReport {
            queries: self.queries,
            groups: self.group_equity(),
            group_gap: self.group_gap(),
            individual_inequity: self.individual_inequity(),
            mean_ndcg,
        }
    }
}

/// `nDCG` of a served page against the ideal page drawn from the **whole** subject
/// pool, so that omitting a highly relevant subject is charged for rather than
/// being invisible.
fn ndcg_against_pool(served: &[f64], pool: &[f64], positions: usize) -> f64 {
    let mut ideal = pool.to_vec();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    ideal.truncate(positions);

    let actual = super::exposure::fairness_dcg_at_k(served, positions);
    let best = super::exposure::fairness_dcg_at_k(&ideal, positions);
    if best <= 0.0 {
        // Nothing relevant anywhere; `fairness_ndcg_at_k` agrees this is zero.
        return fairness_ndcg_at_k(served, positions);
    }
    (actual / best).clamp(0.0, 1.0)
}
