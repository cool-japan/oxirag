//! [`ActiveLearningSelector`] — uncertainty scoring, pool ranking, and
//! (optionally diversity-filtered) batch selection.

use super::types::{
    ActiveLearningConfig, ActiveLearningError, ActiveLearningResult, PoolItem, SelectionBatch,
    UncertaintyMeasure, UncertaintySample,
};

// ── probability-distribution / uncertainty math (house pattern; mirrors
//    `skr`/`diversity_rank`/`semantic_entropy` in spirit but kept private
//    and self-contained here, per this crate's convention of each module
//    owning its own small numerical toolkit rather than sharing helpers
//    across module boundaries) ─────────────────────────────────────────────

/// Convert raw candidate `scores` into a probability distribution.
///
/// Each score is clamped to `>= 0.0` (a negative "relevance" is treated as
/// no relevance) and the clamped vector is divided by its sum. When every
/// clamped score is (numerically) zero — every raw score was non-positive,
/// or they were all exactly equal after clamping in a way that still sums
/// near zero — there is no signal left to distinguish candidates, so a
/// uniform distribution over all `n` entries is returned instead of
/// dividing by zero. This also means an item whose raw scores are all
/// exactly equal (a genuine tie, the maximally ambiguous case) is correctly
/// treated as uniform even when the tied value itself is nonzero, since the
/// direct (non-fallback) division branch produces the same uniform result
/// for equal positive inputs.
///
/// Preserves the *rank order* of the input (a positive scalar division does
/// not reorder), so whichever entries were the largest raw scores remain
/// the largest probabilities — this is what lets [`margin_uncertainty`]
/// read off "top-1" and "top-2" after normalizing.
///
/// Callers must ensure `scores` is non-empty and every entry is finite; see
/// [`ActiveLearningSelector::compute_uncertainty`].
fn to_probability_distribution(scores: &[f32]) -> Vec<f32> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let clamped: Vec<f32> = scores.iter().map(|&s| s.max(0.0)).collect();
    let sum: f32 = clamped.iter().sum();
    if sum > f32::EPSILON {
        clamped.iter().map(|&c| c / sum).collect()
    } else {
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0 / n as f32;
        vec![uniform; n]
    }
}

/// Margin-sampling uncertainty for one item's raw `scores`.
///
/// `1.0 - (p_top1 - p_top2)` over the probability distribution derived by
/// [`to_probability_distribution`], where `p_top1 >= p_top2` are the two
/// largest entries. `p_top2` is `0.0` when fewer than two candidates are
/// present (a single-candidate item has nothing to be ambiguous *against*,
/// so it is maximally *un*certain — i.e. this returns `0.0`).
///
/// Bounded to `[0.0, 1.0]`: `0.0` when one candidate completely dominates
/// (`p_top1 = 1`, `p_top2 = 0`), `1.0` when the top two candidates are
/// exactly tied (maximum ambiguity, including the "no signal at all"
/// uniform fallback).
///
/// Only ever looks at the two largest probabilities — everything else in
/// the distribution's tail is ignored, which is precisely what
/// distinguishes this from [`entropy_uncertainty`].
fn margin_uncertainty(scores: &[f32]) -> f32 {
    let mut probs = to_probability_distribution(scores);
    probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top1 = probs.first().copied().unwrap_or(0.0);
    let top2 = probs.get(1).copied().unwrap_or(0.0);
    (1.0 - (top1 - top2)).clamp(0.0, 1.0)
}

/// Entropy-sampling uncertainty for one item's raw `scores`.
///
/// Shannon entropy `H = -Σ p_i · ln(p_i)` (natural log, i.e. nats) over the
/// probability distribution derived by [`to_probability_distribution`].
/// Zero-mass entries contribute nothing (`0 · ln(0) := 0`, the standard
/// convention/limit).
///
/// Ranges from `0.0` (one candidate holds all the probability mass) up to
/// `ln(n)` for `n` candidates (a perfectly uniform distribution) — *not*
/// rescaled to `[0.0, 1.0]`, so raw entropy magnitudes are only directly
/// comparable across items with the same candidate count. Unlike
/// [`margin_uncertainty`], every candidate in the distribution contributes,
/// not just the top two.
fn entropy_uncertainty(scores: &[f32]) -> f32 {
    let probs = to_probability_distribution(scores);
    let mut h = 0.0_f32;
    for &p in &probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    // Guard against a tiny negative value from floating-point rounding.
    h.max(0.0)
}

/// Cosine similarity between two vectors, clamped to `[-1.0, 1.0]`.
///
/// Returns `0.0` for mismatched-length or empty inputs and for
/// zero-magnitude vectors, rather than dividing by zero — in particular,
/// two [`PoolItem`]s that both left [`PoolItem::features`] empty are always
/// `0.0` similar, so they are never flagged as near-duplicates by the
/// diversity filter as long as its threshold is non-negative (the sensible,
/// and default, case).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

// ── ActiveLearningSelector ────────────────────────────────────────────────

/// Uncertainty-sampling active learning selector: scores a pool of
/// already-scored candidate items by informativeness, ranks it, and selects
/// the most valuable batch to verify/label next.
///
/// See the [module documentation](crate::active_learning_retrieval) for how
/// this differs from `skr`'s per-query retrieve-or-not gate and `dragin`'s
/// token-level mid-generation trigger.
#[derive(Debug, Clone, Default)]
pub struct ActiveLearningSelector {
    /// The selector's configuration.
    pub config: ActiveLearningConfig,
}

impl ActiveLearningSelector {
    /// Create a new selector.
    ///
    /// # Errors
    ///
    /// Returns an error from [`ActiveLearningConfig::validate`] if `config`
    /// is invalid.
    pub fn new(config: ActiveLearningConfig) -> ActiveLearningResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute the uncertainty score for one candidate's raw `scores` under
    /// `measure`.
    ///
    /// `item_id` is used only to attribute a validation error to the right
    /// candidate; it does not affect the computed score.
    ///
    /// # Errors
    ///
    /// - [`ActiveLearningError::EmptyItemScores`] if `scores` is empty.
    /// - [`ActiveLearningError::NonFiniteScore`] if any entry of `scores`
    ///   is `NaN` or infinite.
    pub fn compute_uncertainty(
        item_id: &str,
        scores: &[f32],
        measure: UncertaintyMeasure,
    ) -> ActiveLearningResult<f32> {
        if scores.is_empty() {
            return Err(ActiveLearningError::EmptyItemScores {
                item_id: item_id.to_string(),
            });
        }
        for (index, &score) in scores.iter().enumerate() {
            if !score.is_finite() {
                return Err(ActiveLearningError::NonFiniteScore {
                    item_id: item_id.to_string(),
                    index,
                });
            }
        }
        Ok(match measure {
            UncertaintyMeasure::MarginSampling => margin_uncertainty(scores),
            UncertaintyMeasure::EntropySampling => entropy_uncertainty(scores),
        })
    }

    /// Score, rank, and select a batch from `pool` under `measure`.
    ///
    /// 1. Every item's [`UncertaintySample`] is computed via
    ///    [`ActiveLearningSelector::compute_uncertainty`].
    /// 2. The pool is stable-sorted descending by uncertainty score (ties
    ///    keep their original `pool` order, so results are fully
    ///    deterministic) — this is [`SelectionBatch::ranked`].
    /// 3. The top `min(batch_size, pool.len())` ranked items become
    ///    [`SelectionBatch::selected`], *unless*
    ///    [`ActiveLearningConfig::diversity_enabled`] is set, in which case
    ///    a greedy filter walks the ranked list top-down and skips any
    ///    candidate whose [`PoolItem::features`] are more similar (cosine)
    ///    than [`ActiveLearningConfig::diversity_threshold`] to an item
    ///    *already selected for this batch* — moving on to consider the
    ///    next-most-uncertain candidate rather than stopping. This can
    ///    yield fewer than `batch_size` selected items when the pool does
    ///    not contain enough mutually-distinct high-uncertainty candidates;
    ///    that is reported honestly rather than backfilled with
    ///    near-duplicates just to hit the requested count.
    ///
    /// An empty `pool`, a `batch_size` of `0`, and a `batch_size` larger
    /// than `pool.len()` are all legitimate degenerate inputs, not errors:
    /// each simply yields fewer than `batch_size` selected items (see
    /// [`SelectionBatch::selected`]).
    ///
    /// # Errors
    ///
    /// - Any error from [`ActiveLearningConfig::validate`] if
    ///   [`ActiveLearningSelector::config`] is invalid.
    /// - [`ActiveLearningError::DuplicateItemId`] if two items in `pool`
    ///   share an id.
    /// - Any error from [`ActiveLearningSelector::compute_uncertainty`]
    ///   propagated from a malformed pool item.
    pub fn select_batch(
        &self,
        pool: &[PoolItem],
        batch_size: usize,
        measure: UncertaintyMeasure,
    ) -> ActiveLearningResult<SelectionBatch> {
        self.config.validate()?;
        Self::check_unique_ids(pool)?;

        // (pool_index, uncertainty_score) pairs, scored in input order.
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(pool.len());
        for (index, item) in pool.iter().enumerate() {
            let score = Self::compute_uncertainty(&item.id, &item.scores, measure)?;
            scored.push((index, score));
        }

        // Stable sort: items that tie on uncertainty keep their original
        // pool (insertion) order, so results stay fully deterministic.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let ranked: Vec<UncertaintySample> = scored
            .iter()
            .map(|&(idx, score)| UncertaintySample {
                item_id: pool[idx].id.clone(),
                uncertainty_score: score,
            })
            .collect();

        let target = batch_size.min(pool.len());
        let selected_pairs: Vec<(usize, f32)> = if target == 0 {
            Vec::new()
        } else if self.config.diversity_enabled {
            Self::greedy_diverse_select(pool, &scored, target, self.config.diversity_threshold)
        } else {
            scored.iter().take(target).copied().collect()
        };

        let selected: Vec<UncertaintySample> = selected_pairs
            .into_iter()
            .map(|(idx, score)| UncertaintySample {
                item_id: pool[idx].id.clone(),
                uncertainty_score: score,
            })
            .collect();

        Ok(SelectionBatch {
            selected,
            ranked,
            measure,
        })
    }

    /// Convenience wrapper: run [`ActiveLearningSelector::select_batch`]
    /// using [`ActiveLearningConfig::batch_size`] and
    /// [`ActiveLearningConfig::measure`] from [`ActiveLearningSelector::config`].
    ///
    /// # Errors
    ///
    /// See [`ActiveLearningSelector::select_batch`].
    pub fn select_default_batch(&self, pool: &[PoolItem]) -> ActiveLearningResult<SelectionBatch> {
        self.select_batch(pool, self.config.batch_size, self.config.measure)
    }

    /// Reject a pool containing two or more items with the same id.
    fn check_unique_ids(pool: &[PoolItem]) -> ActiveLearningResult<()> {
        let mut seen: std::collections::HashSet<&str> =
            std::collections::HashSet::with_capacity(pool.len());
        for item in pool {
            if !seen.insert(item.id.as_str()) {
                return Err(ActiveLearningError::DuplicateItemId {
                    item_id: item.id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Greedy diversity filter: walk `scored` (already sorted
    /// descending-by-uncertainty) top-down, skipping any candidate whose
    /// [`PoolItem::features`] are more similar than `diversity_threshold`
    /// to an item already accumulated in the returned selection, until
    /// `target` items have been chosen or the list is exhausted.
    ///
    /// Returns fewer than `target` pairs when not enough mutually-distinct
    /// candidates exist — deliberately does not backfill with
    /// near-duplicates to force the count up to `target`.
    fn greedy_diverse_select(
        pool: &[PoolItem],
        scored: &[(usize, f32)],
        target: usize,
        diversity_threshold: f32,
    ) -> Vec<(usize, f32)> {
        let mut chosen: Vec<(usize, f32)> = Vec::with_capacity(target);
        for &(idx, score) in scored {
            if chosen.len() >= target {
                break;
            }
            let candidate_features = &pool[idx].features;
            let too_similar = chosen.iter().any(|&(chosen_idx, _)| {
                cosine_similarity(candidate_features, &pool[chosen_idx].features)
                    > diversity_threshold
            });
            if too_similar {
                continue;
            }
            chosen.push((idx, score));
        }
        chosen
    }
}
