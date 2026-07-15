//! **Exposure**: the position-discounted attention a ranking hands out, and how
//! it is split between groups.
//!
//! # The primitive, and the semantic inversion that makes it new
//!
//! The discount itself is the familiar one:
//!
//! ```text
//! exposure(rank) = 1 / log2(1 + rank)      (rank is 1-based)
//! ```
//!
//! and it is already all over this crate — `retrieval_eval::dcg_at_k`,
//! `retrieval_diversity`'s `alpha-nDCG`, every `nDCG` in every evaluation module.
//! But in every one of those places it is a **gain to the user**: a weight on how
//! much a document at that rank is *worth to the person reading the page*, and
//! the quantity being accumulated is the searcher's satisfaction.
//!
//! Here it is the opposite side of the same coin. `1 / log2(1 + rank)` is read as
//! the **attention the ranking pays out** to whoever occupies that slot — a
//! *resource* being allocated, not a *utility* being collected. The user is not
//! the beneficiary; the ranked item is. Nothing about the arithmetic changes and
//! everything about what it means does, and that inversion is the whole reason
//! this module exists: once exposure is a resource, it can be distributed
//! unfairly, and asking whether it *has* been is a question no other module in
//! this crate asks.
//!
//! The contrast with `retrieval_diversity` is the sharpest available. Its
//! `alpha-nDCG` *discounts* a subtopic that has already been covered — the second
//! document about the same thing is worth less, because the user has already
//! learned it. Exposure equity wants the **exact opposite**: a group that has
//! already appeared should keep appearing, in proportion to its relevance, and a
//! ranking that shows a group once and then stops has failed. The two metrics
//! would grade the same ranking in opposite directions, and neither is wrong;
//! they are answering different questions.
//!
//! # Exposure per unit relevance
//!
//! Raw group exposure is not a fairness measure, because groups are not equally
//! relevant. A group holding nine of the ten best documents *should* get most of
//! the exposure. The quantity that controls for this is
//!
//! ```text
//! EUR(g) = exposure(g) / relevance(g)
//! ```
//!
//! — the exposure a group receives *per unit of relevance it brought* — and the
//! headline disparity is `max_g EUR(g) - min_g EUR(g)`, zero exactly when
//! attention is being allocated in proportion to merit. This is the target
//! [`EquityOfAttention`](super::EquityOfAttention) amortizes toward, and the
//! number the module's ablations report.
//!
//! A group with zero total relevance has no defined `EUR` — you cannot divide by
//! it, and "infinitely over-exposed" is not a useful thing to say about a group
//! that was never going to be shown. Such groups are excluded from the disparity
//! and counted in [`ExposureMetrics::groups_without_relevance`], loudly, rather
//! than being given a `0` or an infinity that would quietly poison the maximum.

use serde::{Deserialize, Serialize};

use super::types::{FairnessError, FairnessResult, GroupId};

/// The attention paid out at zero-based position `index`, i.e. at the 1-based
/// rank `index + 1`:
///
/// ```text
/// 1 / log2(1 + rank) = 1 / log2(2 + index)
/// ```
///
/// Position `0` (rank 1) receives exactly `1.0`, and the function is total — there
/// is no rank at which it divides by zero, which is why it takes the zero-based
/// index the rest of this crate already uses in `SearchResult::rank`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // Positions on a results page.
pub fn exposure_at_rank(index: usize) -> f64 {
    1.0 / ((index as f64) + 2.0).log2()
}

/// The total attention a ranking of `positions` slots pays out:
/// `sum_{j < positions} exposure_at_rank(j)`.
#[must_use]
pub fn total_exposure(positions: usize) -> f64 {
    (0..positions).map(exposure_at_rank).sum()
}

/// Discounted cumulative gain of `relevances`, taken **in the given order**, to
/// depth `k`.
///
/// `k == 0` means "the whole list".
#[must_use]
pub fn fairness_dcg_at_k(relevances: &[f64], k: usize) -> f64 {
    let depth = if k == 0 { relevances.len() } else { k };
    relevances
        .iter()
        .take(depth)
        .enumerate()
        .map(|(index, &relevance)| relevance * exposure_at_rank(index))
        .sum()
}

/// Normalized discounted cumulative gain of `relevances` in the given order, to
/// depth `k`, against the ideal ordering of the same relevances.
///
/// This is the **utility** side of every ablation in this module: a fairness
/// intervention buys exposure equity by moving relevant documents down, and what
/// it costs is measured here. Returns `0.0` when the ideal gain is zero (nothing
/// relevant to rank).
///
/// Named with the module prefix because `retrieval_eval::ndcg_at_k` already
/// exists and means the same thing over a different input shape; this crate's
/// prelude is flat.
#[must_use]
pub fn fairness_ndcg_at_k(relevances: &[f64], k: usize) -> f64 {
    let actual = fairness_dcg_at_k(relevances, k);
    let mut ideal_order = relevances.to_vec();
    ideal_order.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let ideal = fairness_dcg_at_k(&ideal_order, k);
    if ideal <= 0.0 {
        return 0.0;
    }
    (actual / ideal).clamp(0.0, 1.0)
}

// ── GroupExposure ────────────────────────────────────────────────────────────

/// What one group got out of a ranking: how many candidates it had, how much
/// position-discounted exposure they were allocated, and how much relevance they
/// contributed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupExposure {
    /// Which group.
    pub group: GroupId,
    /// How many of its members appear in the ranked prefix being measured.
    pub members: usize,
    /// The sum of `exposure_at_rank` over the positions its members occupy.
    pub exposure: f64,
    /// The sum of its members' relevances.
    pub relevance: f64,
}

impl GroupExposure {
    /// Exposure **per unit relevance**: `exposure / relevance`.
    ///
    /// `None` when the group's total relevance is zero, because the ratio is
    /// genuinely undefined there — see the [module documentation](self) for why
    /// this is an `Option` and not a zero.
    #[must_use]
    pub fn exposure_per_relevance(&self) -> Option<f64> {
        if self.relevance > 0.0 {
            Some(self.exposure / self.relevance)
        } else {
            None
        }
    }

    /// This group's share of `total`, or `0.0` if `total` is not positive.
    #[must_use]
    pub fn exposure_share(&self, total: f64) -> f64 {
        if total > 0.0 {
            self.exposure / total
        } else {
            0.0
        }
    }
}

// ── DisparateExposure ────────────────────────────────────────────────────────

/// The spread of exposure-per-unit-relevance across groups: who is most
/// over-exposed, who is most under-exposed, and by how much.
///
/// This is `DELTR`'s objective made explicit and measurable — the quantity a
/// disparate-exposure regularizer exists to shrink.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisparateExposure {
    /// The group with the highest exposure per unit relevance, if any group has
    /// positive relevance.
    pub most_exposed: Option<GroupId>,
    /// The group with the lowest exposure per unit relevance.
    pub least_exposed: Option<GroupId>,
    /// The highest exposure per unit relevance seen.
    pub max_exposure_per_relevance: f64,
    /// The lowest exposure per unit relevance seen.
    pub min_exposure_per_relevance: f64,
    /// `max - min`. **Zero exactly when exposure is proportional to relevance.**
    /// This is the module's headline disparity.
    pub gap: f64,
    /// `min / max`, in `[0, 1]`. `1.0` is parity. `0.0` when the least-exposed
    /// group received nothing at all.
    pub ratio: f64,
}

impl DisparateExposure {
    /// The disparity of a ranking in which no group has positive relevance:
    /// nothing to compare, so no disparity to report.
    #[must_use]
    fn degenerate() -> Self {
        Self {
            most_exposed: None,
            least_exposed: None,
            max_exposure_per_relevance: 0.0,
            min_exposure_per_relevance: 0.0,
            gap: 0.0,
            ratio: 1.0,
        }
    }
}

// ── ExposureMetrics ──────────────────────────────────────────────────────────

/// Every exposure quantity a single ranking gives rise to.
///
/// Built by [`ExposureMetrics::measure`] from a ranking (a slice of relevances
/// **in ranked order**) and the parallel group labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExposureMetrics {
    /// Per-group totals, one entry per group of the attribute, in group order.
    groups: Vec<GroupExposure>,
    /// The total exposure the ranking handed out.
    total_exposure: f64,
    /// The total relevance in the ranked prefix.
    total_relevance: f64,
    /// How many candidates were ranked.
    ranked: usize,
}

impl ExposureMetrics {
    /// Measure a ranking. `relevances` are in **ranked order** (position `0` is
    /// the top of the page) and `groups` is the parallel per-result group label
    /// slice; `group_count` is how many groups the attribute has.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::GroupLabelMismatch`] — the two slices differ in length.
    /// * [`FairnessError::UnknownGroup`] — a label is out of range.
    /// * [`FairnessError::NonFinite`] — a relevance is `NaN` or infinite.
    /// * [`FairnessError::InvalidConfig`] — a relevance is negative. A negative
    ///   relevance would make `exposure / relevance` change sign and turn the
    ///   disparity into nonsense, so it is rejected rather than absorbed.
    pub fn measure(
        relevances: &[f64],
        groups: &[GroupId],
        group_count: usize,
    ) -> FairnessResult<Self> {
        if relevances.len() != groups.len() {
            return Err(FairnessError::GroupLabelMismatch {
                results: relevances.len(),
                labels: groups.len(),
            });
        }
        for &relevance in relevances {
            if !relevance.is_finite() {
                return Err(FairnessError::NonFinite {
                    what: "relevance in exposure measurement",
                });
            }
            if relevance < 0.0 {
                return Err(FairnessError::InvalidConfig {
                    reason: format!("relevance {relevance} is negative"),
                });
            }
        }

        let mut accumulated: Vec<GroupExposure> = (0..group_count)
            .map(|index| GroupExposure {
                group: GroupId(index),
                members: 0,
                exposure: 0.0,
                relevance: 0.0,
            })
            .collect();

        let mut total_exposure = 0.0;
        let mut total_relevance = 0.0;
        for (position, (&relevance, &group)) in relevances.iter().zip(groups).enumerate() {
            let slot = accumulated
                .get_mut(group.index())
                .ok_or(FairnessError::UnknownGroup {
                    group: group.index(),
                    groups: group_count,
                })?;
            let attention = exposure_at_rank(position);
            slot.members += 1;
            slot.exposure += attention;
            slot.relevance += relevance;
            total_exposure += attention;
            total_relevance += relevance;
        }

        Ok(Self {
            groups: accumulated,
            total_exposure,
            total_relevance,
            ranked: relevances.len(),
        })
    }

    /// Per-group totals, in group order.
    #[must_use]
    pub fn groups(&self) -> &[GroupExposure] {
        &self.groups
    }

    /// One group's totals.
    #[must_use]
    pub fn group(&self, group: GroupId) -> Option<&GroupExposure> {
        self.groups.get(group.index())
    }

    /// The total exposure the ranking handed out.
    #[must_use]
    pub fn total_exposure(&self) -> f64 {
        self.total_exposure
    }

    /// The total relevance in the ranked prefix.
    #[must_use]
    pub fn total_relevance(&self) -> f64 {
        self.total_relevance
    }

    /// How many candidates were ranked.
    #[must_use]
    pub fn ranked(&self) -> usize {
        self.ranked
    }

    /// The groups whose total relevance is zero, and whose exposure per unit
    /// relevance is therefore undefined. They are excluded from
    /// [`ExposureMetrics::disparity`]; this accessor is how a caller finds out
    /// that they were.
    #[must_use]
    pub fn groups_without_relevance(&self) -> Vec<GroupId> {
        self.groups
            .iter()
            .filter(|entry| entry.members > 0 && entry.relevance <= 0.0)
            .map(|entry| entry.group)
            .collect()
    }

    /// The spread of exposure per unit relevance across the groups that have any.
    ///
    /// Groups with no members, or with zero total relevance, are skipped. With
    /// fewer than two comparable groups there is nothing to compare and the gap is
    /// zero.
    #[must_use]
    pub fn disparity(&self) -> DisparateExposure {
        let mut best: Option<(GroupId, f64)> = None;
        let mut worst: Option<(GroupId, f64)> = None;
        let mut comparable = 0usize;

        for entry in &self.groups {
            let Some(ratio) = entry.exposure_per_relevance() else {
                continue;
            };
            comparable += 1;
            if best.is_none_or(|(_, value)| ratio > value) {
                best = Some((entry.group, ratio));
            }
            if worst.is_none_or(|(_, value)| ratio < value) {
                worst = Some((entry.group, ratio));
            }
        }

        let (Some((max_group, max_value)), Some((min_group, min_value))) = (best, worst) else {
            return DisparateExposure::degenerate();
        };
        if comparable < 2 {
            // One group with relevance is not a disparity; it is a monoculture.
            return DisparateExposure {
                most_exposed: Some(max_group),
                least_exposed: Some(min_group),
                max_exposure_per_relevance: max_value,
                min_exposure_per_relevance: min_value,
                gap: 0.0,
                ratio: 1.0,
            };
        }

        let ratio = if max_value > 0.0 {
            (min_value / max_value).clamp(0.0, 1.0)
        } else {
            1.0
        };
        DisparateExposure {
            most_exposed: Some(max_group),
            least_exposed: Some(min_group),
            max_exposure_per_relevance: max_value,
            min_exposure_per_relevance: min_value,
            gap: max_value - min_value,
            ratio,
        }
    }

    /// The largest gap between a group's share of the exposure and its share of
    /// the candidate pool.
    ///
    /// **Relevance-blind by construction.** A group whose candidates are all
    /// terrible will still be scored as under-exposed by this metric, and that is
    /// what "demographic parity" means; use [`ExposureMetrics::disparity`] when
    /// merit should be controlled for.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Member counts in one ranking.
    pub fn demographic_parity_gap(&self) -> f64 {
        if self.ranked == 0 || self.total_exposure <= 0.0 {
            return 0.0;
        }
        let ranked = self.ranked as f64;
        self.groups
            .iter()
            .filter(|entry| entry.members > 0)
            .map(|entry| {
                let exposure_share = entry.exposure / self.total_exposure;
                let population_share = (entry.members as f64) / ranked;
                (exposure_share - population_share).abs()
            })
            .fold(0.0f64, f64::max)
    }
}
