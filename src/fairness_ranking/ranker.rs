//! [`FairnessRanker`]: the single-ranking entry point that applies a
//! [`FairnessPolicy`] to a scored candidate list and reports what it did to the
//! group exposure.
//!
//! This is the module's façade over `FA*IR` ([`fair`](super::fair)) and the
//! greedy proportional-exposure re-ranker. `DELTR` ([`deltr`](super::deltr)) is a
//! *scorer* rather than a re-ranker and so has its own type; the amortized
//! sequence ([`amortized`](super::amortized)) spans many queries and so has its
//! own type too. What lives here is the per-query intervention: take scores, take
//! group labels, return a fair order, and hand back a [`FairnessReport`] that
//! quantifies the exposure before and after.
//!
//! The re-ranking shape — sort by score, break ties by document id, re-assign the
//! rank field `0..n` — imitates `source_credibility::CredibilityScorer::rerank`,
//! so a `FairnessRanker` drops into the same pipeline slot as any other re-ranker.

use serde::{Deserialize, Serialize};

use crate::types::SearchResult;

use super::exposure::ExposureMetrics;
use super::fair::{FairnessAudit, MTable, audit_ranking, fair_top_k};
use super::types::{
    ExposureTarget, FairnessConfig, FairnessError, FairnessMetric, FairnessPolicy, FairnessResult,
    GroupId,
};

/// A before-and-after account of a single fairness intervention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FairnessReport {
    /// Exposure of the **unconstrained** baseline ranking (sort by score). This is
    /// the ablation baseline; every number below is relative to it.
    pub baseline: ExposureMetrics,
    /// Exposure of the ranking the policy actually produced.
    pub adjusted: ExposureMetrics,
    /// The result of the `FA*IR` audit, present only under
    /// [`FairnessPolicy::RankedGroupFairness`].
    pub audit: Option<FairnessAudit>,
    /// The baseline's exposure-per-unit-relevance disparity gap.
    pub baseline_gap: f64,
    /// The adjusted ranking's exposure-per-unit-relevance disparity gap. Under any
    /// working policy this is no larger than `baseline_gap`.
    pub adjusted_gap: f64,
    /// The baseline `nDCG` (utility to the user).
    pub baseline_ndcg: f64,
    /// The adjusted `nDCG`. The gap between this and `baseline_ndcg` is the utility
    /// price of the fairness gain.
    pub adjusted_ndcg: f64,
}

impl FairnessReport {
    /// How much the exposure disparity fell: `baseline_gap - adjusted_gap`. A
    /// positive number means the intervention helped; a negative one means it hurt,
    /// and is exactly the regression the ablation test is built to catch.
    #[must_use]
    pub fn disparity_reduction(&self) -> f64 {
        self.baseline_gap - self.adjusted_gap
    }

    /// How much utility was surrendered: `baseline_ndcg - adjusted_ndcg`.
    #[must_use]
    pub fn utility_cost(&self) -> f64 {
        self.baseline_ndcg - self.adjusted_ndcg
    }

    /// The requested single-scalar summary of the adjusted ranking.
    #[must_use]
    pub fn metric(&self, metric: FairnessMetric) -> f64 {
        match metric {
            FairnessMetric::ExposureDisparity => self.adjusted.disparity().gap,
            FairnessMetric::ExposureRatio => self.adjusted.disparity().ratio,
            FairnessMetric::DemographicParityGap => self.adjusted.demographic_parity_gap(),
            FairnessMetric::RankedGroupFairness => match &self.audit {
                Some(audit) if audit.satisfied => 1.0,
                _ => 0.0,
            },
        }
    }
}

/// Applies a [`FairnessPolicy`] to a scored candidate list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FairnessRanker {
    /// The configuration: attribute, policy, cutoff.
    config: FairnessConfig,
}

impl FairnessRanker {
    /// Build a ranker from a validated configuration.
    ///
    /// # Errors
    ///
    /// Whatever [`FairnessConfig::validate`] raises — an out-of-range proportion
    /// or significance.
    pub fn new(config: FairnessConfig) -> FairnessResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &FairnessConfig {
        &self.config
    }

    /// The effective cutoff for a candidate list of length `n`:
    /// [`FairnessConfig::top_k`] clamped into `1..=n`, with `0` meaning "all".
    fn effective_k(&self, n: usize) -> usize {
        if self.config.top_k == 0 {
            n
        } else {
            self.config.top_k.min(n)
        }
    }

    /// Re-rank raw `(score, group)` pairs, returning the fair order as indices into
    /// the input.
    ///
    /// This is the core routine; [`FairnessRanker::rerank`] wraps it for
    /// [`SearchResult`]s. `scores` and `groups` are parallel and index-aligned; the
    /// returned indices are a permutation-prefix of `0..scores.len()` of length
    /// `effective_k`.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::GroupLabelMismatch`] / [`FairnessError::UnknownGroup`] —
    ///   from validating the labels.
    /// * [`FairnessError::EmptyRanking`] — empty input.
    /// * [`FairnessError::NonFinite`] — a non-finite score.
    /// * [`FairnessError::InsufficientProtected`] — `FA*IR` cannot be satisfied by
    ///   the pool.
    /// * [`FairnessError::ZeroRelevance`] — proportional-exposure against a
    ///   relevance target when all scores are zero.
    pub fn order(&self, scores: &[f64], groups: &[GroupId]) -> FairnessResult<Vec<usize>> {
        self.config
            .attribute
            .validate_labels(groups, scores.len())?;
        if scores.is_empty() {
            return Err(FairnessError::EmptyRanking { action: "re-rank" });
        }
        if scores.iter().any(|value| !value.is_finite()) {
            return Err(FairnessError::NonFinite {
                what: "candidate score",
            });
        }
        let k = self.effective_k(scores.len());

        match &self.config.policy {
            FairnessPolicy::Unconstrained => Ok(Self::unconstrained_order(scores, k)),
            FairnessPolicy::RankedGroupFairness {
                target_proportion,
                significance,
            } => self.fair_order(scores, groups, k, *target_proportion, *significance),
            FairnessPolicy::ProportionalExposure { target } => {
                self.proportional_order(scores, groups, k, *target)
            }
        }
    }

    /// The unconstrained baseline: sort by descending score, break ties by the
    /// original index (which stands in for the deterministic document-id tie-break
    /// of the `SearchResult` path).
    fn unconstrained_order(scores: &[f64], k: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&left, &right| {
            scores[right]
                .partial_cmp(&scores[left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(&right))
        });
        order.truncate(k);
        order
    }

    /// `FA*IR` ranked group fairness. Collapses the attribute's groups to the
    /// binary "protected / not", sorts each side by descending score, and
    /// interleaves them with [`fair_top_k`] under the corrected `m-table`.
    fn fair_order(
        &self,
        scores: &[f64],
        groups: &[GroupId],
        k: usize,
        target_proportion: f64,
        significance: f64,
    ) -> FairnessResult<Vec<usize>> {
        let attribute = &self.config.attribute;

        let mut protected: Vec<usize> = (0..scores.len())
            .filter(|&index| attribute.is_protected(groups[index]))
            .collect();
        let mut unprotected: Vec<usize> = (0..scores.len())
            .filter(|&index| !attribute.is_protected(groups[index]))
            .collect();
        let by_score = |left: &usize, right: &usize| {
            scores[*right]
                .partial_cmp(&scores[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        };
        protected.sort_by(by_score);
        unprotected.sort_by(by_score);

        let table = MTable::new(k, target_proportion, significance)?;
        fair_top_k(&protected, &unprotected, scores, &table)
    }

    /// Greedy multi-group proportional-exposure re-ranking. Position by position,
    /// the group whose exposure has fallen furthest behind its target share
    /// contributes its best remaining candidate.
    fn proportional_order(
        &self,
        scores: &[f64],
        groups: &[GroupId],
        k: usize,
        target: ExposureTarget,
    ) -> FairnessResult<Vec<usize>> {
        let group_count = self.config.attribute.group_count();

        // Each group's candidates, best first.
        let mut queues: Vec<Vec<usize>> = vec![Vec::new(); group_count];
        for (index, group) in groups.iter().enumerate() {
            if let Some(queue) = queues.get_mut(group.index()) {
                queue.push(index);
            }
        }
        for queue in &mut queues {
            queue.sort_by(|&left, &right| {
                scores[right]
                    .partial_cmp(&scores[left])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.cmp(&right))
            });
        }
        let mut cursors = vec![0usize; group_count];

        // Target exposure shares.
        let targets = Self::exposure_targets(scores, groups, group_count, target)?;

        // The deficit is measured against the **whole** exposure budget the
        // ranking will hand out, not the running total awarded so far. Using the
        // running total is the myopic trap: after the top slot goes to one group,
        // every other group's "share of what has been handed out so far" makes it
        // look badly behind, so it grabs the next (still large) slots and
        // *overshoots* its true target — driving the disparity up rather than
        // down. Anchoring on the full budget lets the greedy anticipate the slots
        // still to come and stop short of overshooting.
        let total_budget: f64 = (0..k).map(super::exposure::exposure_at_rank).sum();

        let mut order = Vec::with_capacity(k);
        let mut awarded = vec![0.0f64; group_count];

        for position in 0..k {
            let slot = super::exposure::exposure_at_rank(position);

            // Pick the group with a remaining candidate whose exposure deficit —
            // its target share of the whole budget, minus what it has received so
            // far — is largest. Ties break toward the higher-scoring head, so the
            // order is deterministic.
            let mut best_group: Option<usize> = None;
            let mut best_deficit = f64::NEG_INFINITY;
            let mut best_head_score = f64::NEG_INFINITY;
            for group in 0..group_count {
                let Some(&head) = queues[group].get(cursors[group]) else {
                    continue;
                };
                let deficit = targets[group] * total_budget - awarded[group];
                let head_score = scores[head];
                let better = match best_group {
                    None => true,
                    Some(_) if deficit > best_deficit + DEFICIT_EPSILON => true,
                    Some(_) if deficit < best_deficit - DEFICIT_EPSILON => false,
                    // Deficits tied: prefer the more qualified head.
                    Some(_) => head_score > best_head_score,
                };
                if better {
                    best_group = Some(group);
                    best_deficit = deficit;
                    best_head_score = head_score;
                }
            }

            let Some(group) = best_group else {
                break;
            };
            let head = queues[group][cursors[group]];
            cursors[group] += 1;
            order.push(head);
            awarded[group] += slot;
        }

        Ok(order)
    }

    /// The target exposure share of each group, either its share of the candidate
    /// pool ([`ExposureTarget::Population`]) or of the total relevance
    /// ([`ExposureTarget::Relevance`]).
    fn exposure_targets(
        scores: &[f64],
        groups: &[GroupId],
        group_count: usize,
        target: ExposureTarget,
    ) -> FairnessResult<Vec<f64>> {
        let mut shares = vec![0.0f64; group_count];
        match target {
            ExposureTarget::Population => {
                for &group in groups {
                    if let Some(slot) = shares.get_mut(group.index()) {
                        *slot += 1.0;
                    }
                }
                #[allow(clippy::cast_precision_loss)] // Candidate counts.
                let total = groups.len() as f64;
                if total > 0.0 {
                    for slot in &mut shares {
                        *slot /= total;
                    }
                }
            }
            ExposureTarget::Relevance => {
                let mut total = 0.0f64;
                for (index, &group) in groups.iter().enumerate() {
                    let relevance = scores[index].max(0.0);
                    if let Some(slot) = shares.get_mut(group.index()) {
                        *slot += relevance;
                    }
                    total += relevance;
                }
                if total <= 0.0 {
                    return Err(FairnessError::ZeroRelevance);
                }
                for slot in &mut shares {
                    *slot /= total;
                }
            }
        }
        Ok(shares)
    }

    /// The full before-and-after [`FairnessReport`] for a `(score, group)` list,
    /// treating each candidate's score as its relevance.
    ///
    /// # Errors
    ///
    /// As [`FairnessRanker::order`].
    pub fn report(&self, scores: &[f64], groups: &[GroupId]) -> FairnessResult<FairnessReport> {
        let group_count = self.config.attribute.group_count();
        let k = self.effective_k(scores.len());

        let baseline_indices = {
            // The baseline is always the unconstrained sort, whatever the policy.
            self.config
                .attribute
                .validate_labels(groups, scores.len())?;
            if scores.is_empty() {
                return Err(FairnessError::EmptyRanking {
                    action: "report on",
                });
            }
            Self::unconstrained_order(scores, k)
        };
        let adjusted_indices = self.order(scores, groups)?;

        let baseline = Self::measure(&baseline_indices, scores, groups, group_count)?;
        let adjusted = Self::measure(&adjusted_indices, scores, groups, group_count)?;

        let baseline_relevances: Vec<f64> = baseline_indices
            .iter()
            .map(|&index| scores[index].max(0.0))
            .collect();
        let adjusted_relevances: Vec<f64> = adjusted_indices
            .iter()
            .map(|&index| scores[index].max(0.0))
            .collect();
        let baseline_ndcg = super::exposure::fairness_ndcg_at_k(&baseline_relevances, k);
        let adjusted_ndcg = super::exposure::fairness_ndcg_at_k(&adjusted_relevances, k);

        let audit = match &self.config.policy {
            FairnessPolicy::RankedGroupFairness {
                target_proportion,
                significance,
            } => {
                let protected_flags: Vec<bool> = adjusted_indices
                    .iter()
                    .map(|&index| self.config.attribute.is_protected(groups[index]))
                    .collect();
                let table = MTable::new(k, *target_proportion, *significance)?;
                Some(audit_ranking(&protected_flags, &table))
            }
            _ => None,
        };

        Ok(FairnessReport {
            baseline_gap: baseline.disparity().gap,
            adjusted_gap: adjusted.disparity().gap,
            baseline_ndcg,
            adjusted_ndcg,
            baseline,
            adjusted,
            audit,
        })
    }

    /// Measure the exposure of a ranking given as indices into the candidate list.
    fn measure(
        indices: &[usize],
        scores: &[f64],
        groups: &[GroupId],
        group_count: usize,
    ) -> FairnessResult<ExposureMetrics> {
        let relevances: Vec<f64> = indices
            .iter()
            .map(|&index| scores[index].max(0.0))
            .collect();
        let ordered_groups: Vec<GroupId> = indices.iter().map(|&index| groups[index]).collect();
        ExposureMetrics::measure(&relevances, &ordered_groups, group_count)
    }

    /// Re-rank [`SearchResult`]s under the policy, using each result's `score` as
    /// its relevance and `groups[i]` as result `i`'s group.
    ///
    /// The returned results are in fair order with their `rank` field re-assigned
    /// `0..k` and their `score` left untouched, exactly as
    /// `source_credibility`'s re-ranker leaves the blended score in place.
    ///
    /// # Errors
    ///
    /// As [`FairnessRanker::order`].
    pub fn rerank(
        &self,
        results: &[SearchResult],
        groups: &[GroupId],
    ) -> FairnessResult<Vec<SearchResult>> {
        let scores: Vec<f64> = results
            .iter()
            .map(|result| f64::from(result.score))
            .collect();
        let order = self.order(&scores, groups)?;
        Ok(order
            .into_iter()
            .enumerate()
            .map(|(rank, index)| {
                let mut result = results[index].clone();
                result.rank = rank;
                result
            })
            .collect())
    }
}

/// Two exposure deficits within this of each other are treated as tied, so that
/// floating-point noise does not make the greedy proportional re-ranker's group
/// choice depend on the last bit of a subtraction.
const DEFICIT_EPSILON: f64 = 1e-12;
