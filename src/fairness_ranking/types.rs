//! The vocabulary the rest of the module is written in: groups, protected
//! attributes, policies, metrics, configuration, and errors.
//!
//! # The one modelling decision that matters
//!
//! Every algorithm here needs to know, for each candidate, **which group it
//! belongs to**. That label arrives as a [`GroupId`] per result — a slice
//! parallel to the results, exactly the shape `retrieval_diversity::alpha_ndcg`
//! uses for its subtopic labels. A [`ProtectedAttribute`] then says what those
//! group indices *mean*: their human-readable labels, and which of them are
//! protected.
//!
//! Splitting it that way is what lets one module serve two families of
//! algorithm that disagree about how many groups there are. `FA*IR` and `DELTR`
//! are **binary** — they ask "protected or not?", and the answer is
//! [`ProtectedAttribute::is_protected`], which treats the union of every
//! protected group as *the* protected class. The exposure metrics and the
//! proportional-exposure policy are **multi-group** — they compare the exposure
//! of an arbitrary number of groups against each other, and they read the
//! [`GroupId`] directly. Neither family has to pretend to be the other.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::assignment::AssignmentError;

// ── Groups ───────────────────────────────────────────────────────────────────

/// The index of a group within a [`ProtectedAttribute`].
///
/// A plain index rather than a string, so that per-result labels are a cheap
/// `&[GroupId]` slice and every group-keyed accumulator is a `Vec` rather than a
/// hash map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GroupId(pub usize);

impl GroupId {
    /// The underlying index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for GroupId {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

/// One group of a [`ProtectedAttribute`]: its identity, its human-readable
/// label, and whether it is protected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedGroup {
    /// The group's index, which is what per-result labels carry.
    pub id: GroupId,
    /// A human-readable name, for reports.
    pub label: String,
    /// Whether this group is *protected*, i.e. whether the ranked group fairness
    /// test and the disparate-exposure penalty are on its side.
    pub protected: bool,
}

/// A demographic attribute over which fairness is being measured — a name
/// (`"gender"`, `"institution tier"`, `"publisher"`) and the groups it takes.
///
/// Group `i` of the attribute is [`GroupId`]`(i)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedAttribute {
    /// The attribute's name.
    name: String,
    /// Its groups, indexed by [`GroupId`].
    groups: Vec<ProtectedGroup>,
}

impl ProtectedAttribute {
    /// Build an attribute from `(label, protected)` pairs, in group-index order.
    ///
    /// # Errors
    ///
    /// [`FairnessError::InvalidConfig`] if fewer than two groups are supplied
    /// (fairness *between* groups needs at least two of them) or if no group is
    /// marked protected (there would be nobody for the fairness constraint to
    /// protect, and every algorithm here would silently become the identity).
    pub fn new<S, I>(name: impl Into<String>, groups: I) -> FairnessResult<Self>
    where
        S: Into<String>,
        I: IntoIterator<Item = (S, bool)>,
    {
        let groups: Vec<ProtectedGroup> = groups
            .into_iter()
            .enumerate()
            .map(|(index, (label, protected))| ProtectedGroup {
                id: GroupId(index),
                label: label.into(),
                protected,
            })
            .collect();

        if groups.len() < 2 {
            return Err(FairnessError::InvalidConfig {
                reason: format!(
                    "a protected attribute needs at least two groups, got {}",
                    groups.len()
                ),
            });
        }
        if !groups.iter().any(|group| group.protected) {
            return Err(FairnessError::InvalidConfig {
                reason: "no group is marked protected: every fairness constraint would be vacuous"
                    .to_string(),
            });
        }

        Ok(Self {
            name: name.into(),
            groups,
        })
    }

    /// The common two-group case: group `0` is protected, group `1` is not.
    ///
    /// # Errors
    ///
    /// Propagates [`ProtectedAttribute::new`], which cannot actually fail for two
    /// groups one of which is protected; the `Result` is kept so callers can use
    /// one error path.
    pub fn binary(
        name: impl Into<String>,
        protected_label: impl Into<String>,
        unprotected_label: impl Into<String>,
    ) -> FairnessResult<Self> {
        Self::new(
            name,
            vec![
                (protected_label.into(), true),
                (unprotected_label.into(), false),
            ],
        )
    }

    /// The attribute's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Every group, in index order.
    #[must_use]
    pub fn groups(&self) -> &[ProtectedGroup] {
        &self.groups
    }

    /// How many groups the attribute has.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Whether `group` is protected. An out-of-range group is not protected —
    /// but [`ProtectedAttribute::validate_labels`] will have rejected it long
    /// before any algorithm asks.
    #[must_use]
    pub fn is_protected(&self, group: GroupId) -> bool {
        self.groups
            .get(group.index())
            .is_some_and(|entry| entry.protected)
    }

    /// The label of `group`, if it exists.
    #[must_use]
    pub fn label(&self, group: GroupId) -> Option<&str> {
        self.groups
            .get(group.index())
            .map(|entry| entry.label.as_str())
    }

    /// Check that `labels` is a usable per-result group assignment: the right
    /// length, and every index in range.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::GroupLabelMismatch`] — `labels.len() != expected_len`.
    /// * [`FairnessError::UnknownGroup`] — a label indexes a group this attribute
    ///   does not have.
    pub fn validate_labels(&self, labels: &[GroupId], expected_len: usize) -> FairnessResult<()> {
        if labels.len() != expected_len {
            return Err(FairnessError::GroupLabelMismatch {
                results: expected_len,
                labels: labels.len(),
            });
        }
        for label in labels {
            if label.index() >= self.groups.len() {
                return Err(FairnessError::UnknownGroup {
                    group: label.index(),
                    groups: self.groups.len(),
                });
            }
        }
        Ok(())
    }

    /// The share of `labels` that belong to a protected group — the natural
    /// choice of `FA*IR`'s target proportion `p` when the goal is for the ranking
    /// to reflect the candidate pool rather than an externally mandated quota.
    ///
    /// Returns `0.0` for an empty slice.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Counts of candidates in one ranking.
    pub fn protected_share(&self, labels: &[GroupId]) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }
        let protected = labels
            .iter()
            .filter(|label| self.is_protected(**label))
            .count();
        (protected as f64) / (labels.len() as f64)
    }
}

// ── Policy ───────────────────────────────────────────────────────────────────

/// How [`FairnessRanker`](super::FairnessRanker) should intervene on a ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FairnessPolicy {
    /// Do not intervene: sort by score, break ties by document id. This is the
    /// **ablation baseline** — the ranking every fairness intervention in this
    /// module is measured against, and the one whose exposure disparity the
    /// others must beat.
    Unconstrained,

    /// `FA*IR` ranked group fairness (Zehlike et al., `CIKM` 2017).
    ///
    /// Binary: protected (the union of the attribute's protected groups) against
    /// the rest. At every prefix `k` of the ranking, the number of protected
    /// candidates in the top-`k` must be at least the `m-table` entry
    /// `m_alpha(k)` — the adjusted binomial quantile derived in
    /// [`fair`](super::fair).
    RankedGroupFairness {
        /// The target proportion `p` of protected candidates, in `(0, 1)`. This
        /// is the success probability of the binomial null model, *not* a hard
        /// quota: the `m-table` is a statistical lower envelope around it, and a
        /// ranking may exceed `p` freely.
        target_proportion: f64,
        /// The significance level `alpha` of the ranked group fairness test, in
        /// `(0, 1)`. Applied to the whole ranking as a family: the per-prefix
        /// significance is *corrected downward* from this value, because the test
        /// is run at every prefix.
        significance: f64,
    },

    /// Greedy proportional-exposure re-ranking. Multi-group, and unlike `FA*IR`
    /// it does not need the groups to collapse to a binary.
    ///
    /// Position by position, the group whose **exposure deficit** is largest —
    /// how far its allocated position-discounted exposure has fallen behind its
    /// target share of the *whole* exposure budget — contributes its
    /// highest-scoring remaining candidate.
    ///
    /// This is a **greedy heuristic**, not a global optimizer. It drives exposure
    /// toward the target shares and, in the common case (a pool with enough
    /// candidates per group to interleave freely), reduces the disparity it
    /// targets — reliably so for [`ExposureTarget::Population`], whose objective
    /// (equal exposure per group) is what the greedy directly descends. On small
    /// or rigidly structured pools, where every group's candidate count forces a
    /// fixed number of slots, the myopic choice can *overshoot* a group's target
    /// and leave the exposure-per-relevance gap no better — occasionally worse —
    /// than the unconstrained baseline. When a *guarantee* is needed, use
    /// [`FairnessPolicy::RankedGroupFairness`], whose `FA*IR` construction is
    /// provably optimal for its criterion.
    ProportionalExposure {
        /// What each group's exposure share is measured *against*.
        target: ExposureTarget,
    },
}

/// The reference distribution a group's exposure share is compared to under
/// [`FairnessPolicy::ProportionalExposure`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExposureTarget {
    /// A group should receive exposure in proportion to its **share of the
    /// candidate pool**. This is demographic parity of exposure: it ignores
    /// relevance entirely, and will hold a slot open for a group whose
    /// candidates are all poor.
    Population,
    /// A group should receive exposure in proportion to its **share of the total
    /// relevance**. This is the merit-based reading — the same one
    /// [`EquityOfAttention`](super::EquityOfAttention) amortizes across a query
    /// sequence — and it is the target that makes exposure *per unit relevance*
    /// equal across groups.
    Relevance,
}

// ── Metric ───────────────────────────────────────────────────────────────────

/// A single scalar summarizing a [`FairnessReport`](super::FairnessReport), for
/// callers that need to rank, threshold, or log one number.
///
/// The three exposure metrics disagree on purpose, and a caller has to choose:
/// [`FairnessMetric::ExposureDisparity`] is scale-dependent and unbounded above,
/// [`FairnessMetric::ExposureRatio`] is a bounded `[0, 1]` score where `1.0` is
/// parity, and [`FairnessMetric::DemographicParityGap`] ignores relevance
/// altogether and so will call a *justified* imbalance unfair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FairnessMetric {
    /// `max - min` of the per-group exposure per unit relevance. Zero is perfect;
    /// larger is worse. The headline number for the ablations in this module.
    ExposureDisparity,
    /// `min / max` of the per-group exposure per unit relevance. `1.0` is
    /// perfect; smaller is worse. Bounded, and therefore comparable across
    /// corpora in a way the raw disparity is not.
    ExposureRatio,
    /// The largest gap between a group's share of the *exposure* and its share of
    /// the *candidate pool*. Zero is perfect. Deliberately relevance-blind.
    DemographicParityGap,
    /// `1.0` if the ranking passes the `FA*IR` ranked group fairness test at every
    /// prefix, `0.0` if it fails at any of them, and `0.0` if no ranked group
    /// fairness audit was requested.
    RankedGroupFairness,
}

// ── Config ───────────────────────────────────────────────────────────────────

/// Configuration for a [`FairnessRanker`](super::FairnessRanker).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FairnessConfig {
    /// The attribute fairness is measured over.
    pub attribute: ProtectedAttribute,
    /// The intervention to apply.
    pub policy: FairnessPolicy,
    /// How many positions the ranking has. `0` means "all the candidates".
    ///
    /// This is not cosmetic: the `m-table` is defined *for a given `k`*, and the
    /// multiple-test correction that produces it depends on how many prefixes the
    /// test is applied to. Auditing the top 10 and the top 100 of the same
    /// ranking are different tests with different decision boundaries.
    pub top_k: usize,
}

impl FairnessConfig {
    /// A configuration with the given attribute, policy, and cutoff.
    #[must_use]
    pub fn new(attribute: ProtectedAttribute, policy: FairnessPolicy, top_k: usize) -> Self {
        Self {
            attribute,
            policy,
            top_k,
        }
    }

    /// Validate the policy's numeric parameters.
    ///
    /// # Errors
    ///
    /// * [`FairnessError::InvalidProportion`] — a target proportion outside the
    ///   **open** interval `(0, 1)`. The endpoints are excluded on purpose: `p = 0`
    ///   makes the `m-table` identically zero (no constraint at all), and `p = 1`
    ///   demands that *every* position be protected.
    /// * [`FairnessError::InvalidSignificance`] — a significance outside `(0, 1)`.
    pub fn validate(&self) -> FairnessResult<()> {
        if let FairnessPolicy::RankedGroupFairness {
            target_proportion,
            significance,
        } = self.policy
        {
            if !target_proportion.is_finite()
                || target_proportion <= 0.0
                || target_proportion >= 1.0
            {
                return Err(FairnessError::InvalidProportion {
                    value: target_proportion,
                });
            }
            if !significance.is_finite() || significance <= 0.0 || significance >= 1.0 {
                return Err(FairnessError::InvalidSignificance {
                    value: significance,
                });
            }
        }
        Ok(())
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// The result type of the `fairness_ranking` module.
pub type FairnessResult<T> = Result<T, FairnessError>;

/// Errors produced by the `fairness_ranking` module.
#[derive(Debug, Error)]
pub enum FairnessError {
    /// The per-result group labels were not the same length as the results.
    #[error("group labels do not match results: {results} results, {labels} labels")]
    GroupLabelMismatch {
        /// How many results were supplied.
        results: usize,
        /// How many group labels were supplied.
        labels: usize,
    },
    /// A group label indexed a group the [`ProtectedAttribute`] does not have.
    #[error("unknown group {group}: the attribute has {groups} groups")]
    UnknownGroup {
        /// The out-of-range index.
        group: usize,
        /// How many groups the attribute actually has.
        groups: usize,
    },
    /// A target proportion `p` was outside the open interval `(0, 1)`.
    #[error("target proportion {value} is outside the open interval (0, 1)")]
    InvalidProportion {
        /// The offending value.
        value: f64,
    },
    /// A significance level `alpha` was outside the open interval `(0, 1)`.
    #[error("significance {value} is outside the open interval (0, 1)")]
    InvalidSignificance {
        /// The offending value.
        value: f64,
    },
    /// `FA*IR` needs more protected candidates than the pool contains: the
    /// `m-table` requires `required` of them in the top-`k` and only `available`
    /// exist. There is no fair ranking, and returning an unfair one silently
    /// would defeat the entire module.
    #[error(
        "ranked group fairness needs {required} protected candidates in the top-k \
         but only {available} exist"
    )]
    InsufficientProtected {
        /// How many the `m-table` demands.
        required: usize,
        /// How many the candidate pool has.
        available: usize,
    },
    /// A feature vector, weight vector, or relevance vector had the wrong length.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// The length that was required.
        expected: usize,
        /// The length that was supplied.
        actual: usize,
    },
    /// An input contained a `NaN` or an infinity. These are rejected at the
    /// boundary rather than allowed to contaminate an exposure accumulator or a
    /// gradient, from which they could never be recovered.
    #[error("non-finite value in {what}")]
    NonFinite {
        /// Which input was non-finite.
        what: &'static str,
    },
    /// A ranking or candidate list was empty.
    #[error("cannot {action} an empty ranking")]
    EmptyRanking {
        /// What was being attempted.
        action: &'static str,
    },
    /// Every relevance in a query was zero (or the total was), so relevance
    /// shares are undefined and "exposure proportional to relevance" has no
    /// meaning.
    #[error("total relevance is zero: exposure cannot be made proportional to it")]
    ZeroRelevance,
    /// A configuration value was outside its admissible range.
    #[error("invalid fairness configuration: {reason}")]
    InvalidConfig {
        /// A human-readable explanation.
        reason: String,
    },
    /// The assignment solver failed; see [`AssignmentError`].
    #[error("assignment failed: {0}")]
    Assignment(#[from] AssignmentError),
}
