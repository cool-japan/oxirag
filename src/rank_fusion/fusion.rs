//! The [`RankFusion`] engine and the underlying fusion algorithms.
//!
//! [`RankFusion`] merges several ranked result lists into a single ranked list
//! using the [`FusionMethod`] and [`ScoreNormalization`] selected in its
//! [`RankFusionConfig`]. Two entry points are provided:
//!
//! * [`RankFusion::fuse`] operates on [`SearchResult`] lists, keying document
//!   identity on [`SearchResult::document`]'s [`DocumentId`].
//! * [`RankFusion::fuse_ids`] operates on `(DocumentId, score)` pairs for callers
//!   that do not have full [`Document`] payloads.
//!
//! Both deduplicate by [`DocumentId`], assign a fused score, sort by descending
//! score with a deterministic [`DocumentId`]-string tie-break, re-rank from `0`,
//! and truncate to [`RankFusionConfig::top_n`].
//!
//! [`FusionMethod`]: crate::rank_fusion::FusionMethod
//! [`ScoreNormalization`]: crate::rank_fusion::ScoreNormalization
//! [`RankFusionConfig`]: crate::rank_fusion::RankFusionConfig
//! [`Document`]: crate::types::Document
//! [`DocumentId`]: crate::types::DocumentId

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::types::{DocumentId, SearchResult};

use super::types::{FusionMethod, RankFusionConfig, RankFusionError, ScoreNormalization};

/// Engine that fuses multiple ranked result lists into a single ranked list.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rank-fusion")]
/// # {
/// use oxirag::rank_fusion::{RankFusion, RankFusionConfig, FusionMethod};
///
/// let fusion = RankFusion::new(
///     RankFusionConfig::new().with_method(FusionMethod::Rrf),
/// );
/// let lists = vec![
///     vec![("a".into(), 0.9_f32), ("b".into(), 0.4)],
///     vec![("b".into(), 0.8_f32), ("a".into(), 0.3)],
/// ];
/// let fused = fusion.fuse_ids(&lists).unwrap();
/// assert_eq!(fused.len(), 2);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct RankFusion {
    /// The configuration controlling method, normalization, weights, and limits.
    config: RankFusionConfig,
}

/// Internal accumulator for a single unique document during fusion.
struct Accumulator {
    /// The cumulative fused score so far.
    score: f32,
    /// The number of input lists in which the document has been observed.
    hits: usize,
}

impl RankFusion {
    /// Create a new fusion engine with the given configuration.
    #[must_use]
    pub fn new(config: RankFusionConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration backing this engine.
    #[must_use]
    pub fn config(&self) -> &RankFusionConfig {
        &self.config
    }

    /// Fuse multiple ranked [`SearchResult`] lists into a single ranked list.
    ///
    /// Documents are deduplicated by [`SearchResult::document`]'s [`DocumentId`].
    /// For each unique document a fused score is computed according to
    /// [`RankFusionConfig::method`], the surviving [`SearchResult`]s are sorted by
    /// descending fused score (ties broken by [`DocumentId`] string order),
    /// re-ranked from `0`, and truncated to [`RankFusionConfig::top_n`].
    ///
    /// The first encountered [`SearchResult`] for a given document id (scanning
    /// lists in order, then positions in order) supplies the [`Document`] payload
    /// carried into the output; only its `score` and `rank` fields are replaced.
    ///
    /// # Errors
    ///
    /// * [`RankFusionError::NoLists`] if `lists` is empty.
    /// * [`RankFusionError::WeightMismatch`] if the method is
    ///   [`FusionMethod::WeightedSum`] and `config.weights.len() != lists.len()`.
    ///
    /// [`Document`]: crate::types::Document
    /// [`DocumentId`]: crate::types::DocumentId
    pub fn fuse(&self, lists: &[Vec<SearchResult>]) -> Result<Vec<SearchResult>, RankFusionError> {
        self.validate(lists.len())?;

        // Normalize per-list scores, projecting each list onto (DocumentId, score).
        let normalized: Vec<Vec<(DocumentId, f32)>> = lists
            .iter()
            .map(|list| {
                let pairs: Vec<(DocumentId, f32)> = list
                    .iter()
                    .map(|sr| (sr.document.id.clone(), sr.score))
                    .collect();
                normalize_list(&pairs, self.config.normalization)
            })
            .collect();

        let fused = self.accumulate(&normalized);

        // Preserve the first-seen full Document payload per document id.
        let mut representatives: HashMap<&str, &SearchResult> = HashMap::new();
        for list in lists {
            for sr in list {
                representatives.entry(sr.document.id.as_str()).or_insert(sr);
            }
        }

        let ordered = sort_and_rank(fused);

        let mut output: Vec<SearchResult> = Vec::with_capacity(ordered.len());
        for (rank, (id, score)) in ordered.into_iter().enumerate() {
            if let Some(rep) = representatives.get(id.as_str()) {
                let mut sr = (*rep).clone();
                sr.score = score;
                sr.rank = rank;
                output.push(sr);
            }
        }

        Ok(self.truncate(output))
    }

    /// Fuse multiple ranked `(DocumentId, score)` lists into a single ranked list.
    ///
    /// Behaves exactly like [`RankFusion::fuse`] but operates on bare
    /// `(DocumentId, score)` pairs and returns the fused pairs sorted by
    /// descending score (ties broken by [`DocumentId`] string order) and
    /// truncated to [`RankFusionConfig::top_n`].
    ///
    /// # Errors
    ///
    /// * [`RankFusionError::NoLists`] if `lists` is empty.
    /// * [`RankFusionError::WeightMismatch`] if the method is
    ///   [`FusionMethod::WeightedSum`] and `config.weights.len() != lists.len()`.
    ///
    /// [`DocumentId`]: crate::types::DocumentId
    pub fn fuse_ids(
        &self,
        lists: &[Vec<(DocumentId, f32)>],
    ) -> Result<Vec<(DocumentId, f32)>, RankFusionError> {
        self.validate(lists.len())?;

        let normalized: Vec<Vec<(DocumentId, f32)>> = lists
            .iter()
            .map(|list| normalize_list(list, self.config.normalization))
            .collect();

        let fused = self.accumulate(&normalized);
        let ordered = sort_and_rank(fused);
        Ok(self.truncate(ordered))
    }

    // ── Internal helpers ───────────────────────────────────────────────────────

    /// Validate the list count against the configured method and weights.
    fn validate(&self, list_count: usize) -> Result<(), RankFusionError> {
        if list_count == 0 {
            return Err(RankFusionError::NoLists);
        }
        if self.config.method == FusionMethod::WeightedSum
            && self.config.weights.len() != list_count
        {
            return Err(RankFusionError::WeightMismatch {
                weights: self.config.weights.len(),
                lists: list_count,
            });
        }
        Ok(())
    }

    /// Accumulate per-document fused scores across all (normalized) lists.
    ///
    /// Returns `(DocumentId, fused_score)` pairs in arbitrary order.
    fn accumulate(&self, lists: &[Vec<(DocumentId, f32)>]) -> Vec<(DocumentId, f32)> {
        let mut acc: HashMap<DocumentId, Accumulator> = HashMap::new();

        for (list_index, list) in lists.iter().enumerate() {
            let list_len = list.len();
            for (rank, (id, score)) in list.iter().enumerate() {
                let contribution = self.contribution(list_index, rank, list_len, *score);
                acc.entry(id.clone())
                    .and_modify(|a| {
                        a.score += contribution;
                        a.hits += 1;
                    })
                    .or_insert(Accumulator {
                        score: contribution,
                        hits: 1,
                    });
            }
        }

        // Apply the multiplicity-dependent post-pass for CombMNZ / CombANZ.
        acc.into_iter()
            .map(|(id, a)| {
                let final_score = match self.config.method {
                    #[allow(clippy::cast_precision_loss)]
                    FusionMethod::CombMnz => a.score * a.hits as f32,
                    #[allow(clippy::cast_precision_loss)]
                    FusionMethod::CombAnz => {
                        if a.hits == 0 {
                            0.0
                        } else {
                            a.score / a.hits as f32
                        }
                    }
                    _ => a.score,
                };
                (id, final_score)
            })
            .collect()
    }

    /// Compute the contribution of a single list entry to a document's score.
    ///
    /// `list_index` selects the weight for [`FusionMethod::WeightedSum`];
    /// `rank` is the 0-based position; `list_len` is the length of the list;
    /// `score` is the already-normalized score.
    #[allow(clippy::cast_precision_loss)]
    fn contribution(&self, list_index: usize, rank: usize, list_len: usize, score: f32) -> f32 {
        let rank_1based = (rank + 1) as f32;
        match self.config.method {
            // CombMNZ / CombANZ accumulate the plain sum first; their
            // multiplicity factor is applied once in `accumulate`.
            FusionMethod::CombSum | FusionMethod::CombMnz | FusionMethod::CombAnz => score,
            FusionMethod::WeightedSum => {
                let weight = self.config.weights.get(list_index).copied().unwrap_or(0.0);
                weight * score
            }
            FusionMethod::Borda => (list_len.saturating_sub(rank)) as f32,
            FusionMethod::Isr => 1.0 / (rank_1based * rank_1based),
            FusionMethod::Rrf => 1.0 / (self.config.rrf_k + rank_1based),
        }
    }

    /// Truncate the fused output to `top_n` (0 = keep all).
    fn truncate<T>(&self, mut items: Vec<T>) -> Vec<T> {
        if self.config.top_n > 0 && items.len() > self.config.top_n {
            items.truncate(self.config.top_n);
        }
        items
    }
}

// ── Free functions ───────────────────────────────────────────────────────────

/// Normalize a single list's scores according to `normalization`.
///
/// Document ids are preserved verbatim; only scores are transformed. The
/// transformation is applied independently to this list.
#[must_use]
fn normalize_list(
    list: &[(DocumentId, f32)],
    normalization: ScoreNormalization,
) -> Vec<(DocumentId, f32)> {
    match normalization {
        ScoreNormalization::None => list.to_vec(),
        ScoreNormalization::MinMax => min_max(list),
        ScoreNormalization::ZScore => z_score(list),
        ScoreNormalization::SumTo1 => sum_to_one(list),
    }
}

/// Min-max scaling of a list's scores onto `[0, 1]`.
fn min_max(list: &[(DocumentId, f32)]) -> Vec<(DocumentId, f32)> {
    if list.is_empty() {
        return Vec::new();
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for (_, s) in list {
        if *s < min {
            min = *s;
        }
        if *s > max {
            max = *s;
        }
    }
    let range = max - min;
    list.iter()
        .map(|(id, s)| {
            let scaled = if range == 0.0 { 1.0 } else { (s - min) / range };
            (id.clone(), scaled)
        })
        .collect()
}

/// Z-score standardization of a list's scores (population standard deviation).
fn z_score(list: &[(DocumentId, f32)]) -> Vec<(DocumentId, f32)> {
    if list.is_empty() {
        return Vec::new();
    }
    #[allow(clippy::cast_precision_loss)]
    let n = list.len() as f32;
    let sum: f32 = list.iter().map(|(_, s)| *s).sum();
    let mean = sum / n;
    let variance: f32 = list
        .iter()
        .map(|(_, s)| (s - mean) * (s - mean))
        .sum::<f32>()
        / n;
    let std_dev = variance.sqrt();
    list.iter()
        .map(|(id, s)| {
            let z = if std_dev == 0.0 {
                0.0
            } else {
                (s - mean) / std_dev
            };
            (id.clone(), z)
        })
        .collect()
}

/// Scale a list's scores so they sum to `1.0`.
fn sum_to_one(list: &[(DocumentId, f32)]) -> Vec<(DocumentId, f32)> {
    let sum: f32 = list.iter().map(|(_, s)| *s).sum();
    if sum == 0.0 {
        return list.to_vec();
    }
    list.iter().map(|(id, s)| (id.clone(), s / sum)).collect()
}

/// Sort fused `(DocumentId, score)` pairs by descending score with a
/// deterministic [`DocumentId`]-string tie-break (ascending).
///
/// [`DocumentId`]: crate::types::DocumentId
#[must_use]
fn sort_and_rank(mut fused: Vec<(DocumentId, f32)>) -> Vec<(DocumentId, f32)> {
    fused.sort_by(|(a_id, a_score), (b_id, b_score)| {
        b_score
            .partial_cmp(a_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a_id.as_str().cmp(b_id.as_str()))
    });
    fused
}
