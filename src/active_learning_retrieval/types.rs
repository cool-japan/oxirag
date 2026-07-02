//! Types, configuration, and errors for the `active_learning_retrieval`
//! module.
//!
//! Uncertainty-sampling active learning for retrieval-pool selection:
//!
//! - [`PoolItem`] — one unlabeled/unverified candidate carrying the raw
//!   candidate/label scores produced by some upstream scorer (e.g. the top-2
//!   or top-N relevance scores of a retrieval pass), plus an optional
//!   feature vector used only for diversity comparisons.
//! - [`UncertaintyMeasure`] — `MarginSampling` (top-1 vs. top-2 score gap) or
//!   `EntropySampling` (Shannon entropy of the whole normalized score
//!   distribution).
//! - [`UncertaintySample`] — one item's id paired with its computed
//!   uncertainty score under a chosen measure.
//! - [`SelectionBatch`] — the output of a selection pass: the chosen batch
//!   plus the full ranked pool, for transparency.
//! - [`ActiveLearningConfig`] — measure choice, batch size, and the optional
//!   greedy diversity filter's toggle/threshold.
//! - [`ActiveLearningError`] / [`ActiveLearningResult`] — the module's error
//!   type and result alias.

use thiserror::Error;

// ── PoolItem ──────────────────────────────────────────────────────────────

/// One unlabeled/unverified candidate item in an active-learning selection
/// pool.
///
/// `scores` carries the raw candidate/label scores an *existing* scorer
/// already produced for this item — for example the top-2 or top-N
/// relevance scores from one retrieval pass. This module does not compute
/// retrieval or scoring itself; it only ranks already-scored candidates by
/// how informative they would be to verify/label next.
///
/// `features` is an optional feature vector (e.g. an embedding of the
/// item's underlying content) consulted *only* by the greedy diversity
/// filter (see [`ActiveLearningConfig::diversity_enabled`]) to detect
/// near-duplicate items. It plays no role in uncertainty scoring. Left empty
/// by default; the cosine similarity between two items that both leave
/// `features` empty is always exactly `0.0` (see
/// [`ActiveLearningSelector::select_batch`](crate::active_learning_retrieval::ActiveLearningSelector::select_batch)),
/// which will not exceed a non-negative
/// [`diversity_threshold`](ActiveLearningConfig::diversity_threshold) — so,
/// with the sensible non-negative thresholds this crate defaults to,
/// omitting `features` simply opts an item out of diversity comparisons
/// rather than causing spurious matches.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolItem {
    /// Unique identifier for this candidate. Must be unique within a single
    /// pool passed to [`ActiveLearningSelector::select_batch`
    /// ](crate::active_learning_retrieval::ActiveLearningSelector::select_batch).
    pub id: String,
    /// Raw candidate/label scores from an existing scorer. Higher is
    /// assumed to mean "more relevant"/"more likely"; the scores need not
    /// be sorted, non-negative, or normalized — [`UncertaintyMeasure`]
    /// computations normalize internally. Must contain at least one
    /// element and every element must be finite (checked by
    /// [`ActiveLearningSelector::compute_uncertainty`
    /// ](crate::active_learning_retrieval::ActiveLearningSelector::compute_uncertainty)).
    pub scores: Vec<f32>,
    /// Optional feature vector used only for diversity comparisons (cosine
    /// similarity). Empty by default, meaning "no diversity signal
    /// supplied" for this item.
    pub features: Vec<f32>,
}

impl PoolItem {
    /// Create a new pool item with no feature vector (diversity comparisons
    /// involving this item will always score `0.0` similarity until
    /// [`PoolItem::with_features`] is used).
    #[must_use]
    pub fn new(id: impl Into<String>, scores: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            scores,
            features: Vec::new(),
        }
    }

    /// Attach a feature vector for diversity comparisons (builder).
    #[must_use]
    pub fn with_features(mut self, features: Vec<f32>) -> Self {
        self.features = features;
        self
    }
}

// ── UncertaintyMeasure ────────────────────────────────────────────────────

/// Which uncertainty-sampling statistic ranks the pool.
///
/// | Measure | Looks at | Signal |
/// |---------|----------|--------|
/// | [`MarginSampling`](Self::MarginSampling) | only the top-1 and top-2 candidate scores | small gap between the two best candidates ⇒ high uncertainty |
/// | [`EntropySampling`](Self::EntropySampling) | the entire normalized score distribution | probability mass spread broadly across many candidates ⇒ high uncertainty |
///
/// Both are genuinely different computations (not aliases): margin sampling
/// is blind to everything past the top two candidates, while entropy
/// sampling is driven by the whole tail of the distribution. On a
/// three-or-more-candidate item the two can and do disagree about which
/// item is more uncertain — see the crate tests for a worked example.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum UncertaintyMeasure {
    /// `1.0 - (p_top1 - p_top2)` over the item's normalized score
    /// distribution — bounded to `[0.0, 1.0]`. Higher means the top two
    /// candidates were harder to tell apart.
    #[default]
    MarginSampling,
    /// Shannon entropy `H = -Σ p_i · ln(p_i)` (natural log, i.e. nats) over
    /// the item's normalized score distribution. Ranges from `0.0` (one
    /// candidate holds all the probability mass) to `ln(n)` for `n`
    /// candidates (a perfectly uniform distribution).
    EntropySampling,
}

impl UncertaintyMeasure {
    /// Return a stable lowercase, `snake_case` string representation of the
    /// measure.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MarginSampling => "margin_sampling",
            Self::EntropySampling => "entropy_sampling",
        }
    }
}

impl std::fmt::Display for UncertaintyMeasure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── UncertaintySample ─────────────────────────────────────────────────────

/// One pool item's id paired with its computed uncertainty score under a
/// chosen [`UncertaintyMeasure`].
///
/// Higher `uncertainty_score` always means "more informative to
/// label/verify next", regardless of which measure produced it — but the
/// two measures are on different numeric scales ([`MarginSampling`
/// ](UncertaintyMeasure::MarginSampling) is bounded to `[0.0, 1.0]`;
/// [`EntropySampling`](UncertaintyMeasure::EntropySampling) is not), so
/// scores are only meaningfully compared *within* a single
/// [`SelectionBatch`] (which always reflects one consistent measure).
#[derive(Debug, Clone, PartialEq)]
pub struct UncertaintySample {
    /// The [`PoolItem::id`] this sample was computed for.
    pub item_id: String,
    /// The computed uncertainty score under the batch's [`UncertaintyMeasure`].
    pub uncertainty_score: f32,
}

// ── SelectionBatch ────────────────────────────────────────────────────────

/// The result of an [`ActiveLearningSelector::select_batch`
/// ](crate::active_learning_retrieval::ActiveLearningSelector::select_batch)
/// call.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionBatch {
    /// The items chosen for labeling/verification next: a prefix of
    /// `ranked` (after the optional greedy diversity filter has skipped any
    /// near-duplicates), in descending-uncertainty order. Contains at most
    /// `batch_size` items, and fewer whenever the pool itself has fewer
    /// items than `batch_size`, or (when the diversity filter is enabled)
    /// whenever too few sufficiently-distinct candidates exist to fill the
    /// batch.
    pub selected: Vec<UncertaintySample>,
    /// Every pool item's [`UncertaintySample`], in descending-uncertainty
    /// order, regardless of what ended up in `selected` — kept so callers
    /// can audit the full ranking a selection decision was based on.
    pub ranked: Vec<UncertaintySample>,
    /// The [`UncertaintyMeasure`] used to compute every score in this batch.
    pub measure: UncertaintyMeasure,
}

impl SelectionBatch {
    /// Return `true` when [`SelectionBatch::selected`] is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// Number of items in [`SelectionBatch::selected`].
    #[must_use]
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    /// The ids of every selected item, in order.
    #[must_use]
    pub fn selected_ids(&self) -> Vec<&str> {
        self.selected.iter().map(|s| s.item_id.as_str()).collect()
    }

    /// The single most uncertain item in the whole pool (the head of
    /// [`SelectionBatch::ranked`]), or `None` when the pool was empty.
    #[must_use]
    pub fn most_uncertain(&self) -> Option<&UncertaintySample> {
        self.ranked.first()
    }
}

// ── ActiveLearningConfig ──────────────────────────────────────────────────

/// Configuration for [`ActiveLearningSelector`
/// ](crate::active_learning_retrieval::ActiveLearningSelector).
///
/// # Fields at a glance
///
/// | Field | Meaning | Default |
/// |-------|---------|---------|
/// | [`measure`](Self::measure) | default uncertainty measure (used by [`ActiveLearningSelector::select_default_batch`](crate::active_learning_retrieval::ActiveLearningSelector::select_default_batch)) | [`MarginSampling`](UncertaintyMeasure::MarginSampling) |
/// | [`batch_size`](Self::batch_size) | default batch size | `10` |
/// | [`diversity_enabled`](Self::diversity_enabled) | apply the greedy near-duplicate filter | `false` |
/// | [`diversity_threshold`](Self::diversity_threshold) | cosine-similarity cutoff above which a candidate is skipped as a near-duplicate | `0.9` |
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveLearningConfig {
    /// The uncertainty measure [`ActiveLearningSelector::select_default_batch`
    /// ](crate::active_learning_retrieval::ActiveLearningSelector::select_default_batch)
    /// uses. [`ActiveLearningSelector::select_batch`
    /// ](crate::active_learning_retrieval::ActiveLearningSelector::select_batch)
    /// takes its own `measure` argument and ignores this field.
    pub measure: UncertaintyMeasure,
    /// The batch size [`ActiveLearningSelector::select_default_batch`
    /// ](crate::active_learning_retrieval::ActiveLearningSelector::select_default_batch)
    /// uses. [`ActiveLearningSelector::select_batch`
    /// ](crate::active_learning_retrieval::ActiveLearningSelector::select_batch)
    /// takes its own `batch_size` argument and ignores this field.
    pub batch_size: usize,
    /// When `true`, [`ActiveLearningSelector::select_batch`
    /// ](crate::active_learning_retrieval::ActiveLearningSelector::select_batch)
    /// applies a greedy filter that skips a candidate whenever its
    /// [`PoolItem::features`] are more similar (by cosine similarity) than
    /// [`diversity_threshold`](Self::diversity_threshold) to an item
    /// already chosen for the *same* batch. When `false`, selection is a
    /// plain top-`batch_size` cut of the ranked pool.
    pub diversity_enabled: bool,
    /// Cosine-similarity threshold (in `[-1.0, 1.0]`) above which two
    /// items' [`PoolItem::features`] are considered near-duplicates by the
    /// greedy diversity filter. Only consulted when
    /// [`diversity_enabled`](Self::diversity_enabled) is `true`, but always
    /// validated by [`ActiveLearningConfig::validate`] regardless, so
    /// toggling diversity on later can never silently activate a
    /// nonsensical threshold.
    pub diversity_threshold: f32,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            measure: UncertaintyMeasure::MarginSampling,
            batch_size: 10,
            diversity_enabled: false,
            diversity_threshold: 0.9,
        }
    }
}

impl ActiveLearningConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default uncertainty measure (builder).
    #[must_use]
    pub fn with_measure(mut self, measure: UncertaintyMeasure) -> Self {
        self.measure = measure;
        self
    }

    /// Set the default batch size (builder).
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable or disable the greedy diversity filter (builder).
    #[must_use]
    pub fn with_diversity_enabled(mut self, diversity_enabled: bool) -> Self {
        self.diversity_enabled = diversity_enabled;
        self
    }

    /// Set the diversity near-duplicate similarity threshold (builder).
    #[must_use]
    pub fn with_diversity_threshold(mut self, diversity_threshold: f32) -> Self {
        self.diversity_threshold = diversity_threshold;
        self
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// [`ActiveLearningError::InvalidDiversityThreshold`] if
    /// [`diversity_threshold`](Self::diversity_threshold) is not finite or
    /// lies outside `[-1.0, 1.0]` (the valid range of a cosine similarity).
    pub fn validate(&self) -> ActiveLearningResult<()> {
        if !self.diversity_threshold.is_finite()
            || !(-1.0..=1.0).contains(&self.diversity_threshold)
        {
            return Err(ActiveLearningError::InvalidDiversityThreshold(
                self.diversity_threshold,
            ));
        }
        Ok(())
    }
}

// ── ActiveLearningError / ActiveLearningResult ───────────────────────────

/// Errors produced by the `active_learning_retrieval` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ActiveLearningError {
    /// A [`PoolItem`] was submitted with an empty `scores` vector; at least
    /// one candidate score is required to compute either uncertainty
    /// measure.
    #[error(
        "pool item {item_id:?} has an empty scores vector; at least one candidate score is required"
    )]
    EmptyItemScores {
        /// The offending item's id.
        item_id: String,
    },
    /// A [`PoolItem`] carried a `NaN` or infinite score.
    #[error("pool item {item_id:?} has a non-finite score at index {index}")]
    NonFiniteScore {
        /// The offending item's id.
        item_id: String,
        /// The index of the non-finite entry within [`PoolItem::scores`].
        index: usize,
    },
    /// Two or more [`PoolItem`]s in the same pool shared an id.
    #[error("duplicate pool item id {item_id:?}")]
    DuplicateItemId {
        /// The id that appeared more than once.
        item_id: String,
    },
    /// [`ActiveLearningConfig::diversity_threshold`] was not finite or fell
    /// outside `[-1.0, 1.0]`.
    #[error("diversity threshold must be finite and within [-1.0, 1.0], got {0}")]
    InvalidDiversityThreshold(f32),
}

/// Convenient result alias for the `active_learning_retrieval` module.
pub type ActiveLearningResult<T> = Result<T, ActiveLearningError>;
