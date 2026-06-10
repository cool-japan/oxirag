//! Types for the `lost_in_middle` module.
use thiserror::Error;
// ── ReorderStrategy ───────────────────────────────────────────────────────────
/// Strategy used to reorder retrieved documents.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ReorderStrategy {
    /// Place most-relevant docs at both ends; least-relevant in the middle.
    #[default]
    Sandwich,
    /// Alternately assign best remaining results to head then tail.
    HeadTail,
    /// Simply sort descending by score (no special positioning).
    Descending,
    /// External custom ordering — results returned as-is after score sort.
    Custom,
}
impl ReorderStrategy {
    /// Human-readable label for this strategy.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Sandwich => "sandwich",
            Self::HeadTail => "head_tail",
            Self::Descending => "descending",
            Self::Custom => "custom",
        }
    }
}
// ── ReorderConfig ─────────────────────────────────────────────────────────────
/// Configuration for `LostInMiddleReorderer`.
#[derive(Debug, Clone)]
pub struct ReorderConfig {
    /// Reordering strategy. Defaults to [`ReorderStrategy::Sandwich`].
    pub strategy: ReorderStrategy,
    /// Top-k results to preserve in their original relative score order before reordering.
    ///
    /// `0` means reorder all results.
    pub preserve_top_k: usize,
}
impl Default for ReorderConfig {
    fn default() -> Self {
        Self {
            strategy: ReorderStrategy::Sandwich,
            preserve_top_k: 0,
        }
    }
}
impl ReorderConfig {
    /// Set the reordering strategy.
    #[must_use]
    pub fn with_strategy(mut self, v: ReorderStrategy) -> Self {
        self.strategy = v;
        self
    }
    /// Set the number of top results to preserve.
    #[must_use]
    pub fn with_preserve_top_k(mut self, v: usize) -> Self {
        self.preserve_top_k = v;
        self
    }
}
// ── ReorderReport ─────────────────────────────────────────────────────────────
/// Describes the reordering that was applied.
#[derive(Debug, Clone)]
pub struct ReorderReport {
    /// Document IDs in the original order.
    pub original_order: Vec<String>,
    /// Document IDs in the new order.
    pub new_order: Vec<String>,
    /// Strategy that was applied.
    pub strategy: ReorderStrategy,
}
// ── LostInMiddleReorderer ─────────────────────────────────────────────────────
/// Reorders search results to mitigate the lost-in-the-middle effect.
#[derive(Debug, Clone)]
pub struct LostInMiddleReorderer {
    /// Configuration for this reorderer.
    pub config: ReorderConfig,
}
impl LostInMiddleReorderer {
    /// Create a new reorderer with the given config.
    #[must_use]
    pub fn new(config: ReorderConfig) -> Self {
        Self { config }
    }
    /// Reorder `results` according to the configured strategy.
    ///
    /// # Errors
    ///
    /// Returns [`LostInMiddleError::EmptyResults`] when `results` is empty.
    pub fn reorder(
        &self,
        results: &[crate::types::SearchResult],
    ) -> Result<Vec<crate::types::SearchResult>, LostInMiddleError> {
        if results.is_empty() {
            return Err(LostInMiddleError::EmptyResults);
        }
        let mut sorted = results.to_vec();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = sorted.len();
        match self.config.strategy {
            ReorderStrategy::Sandwich | ReorderStrategy::HeadTail => {
                let mut out = vec![None; n];
                let mut head = 0usize;
                let mut tail = n.saturating_sub(1);
                for (i, r) in sorted.into_iter().enumerate() {
                    if i % 2 == 0 {
                        out[head] = Some(r);
                        head += 1;
                    } else {
                        out[tail] = Some(r);
                        tail = tail.saturating_sub(1);
                    }
                }
                let mut res: Vec<crate::types::SearchResult> = out.into_iter().flatten().collect();
                for (i, r) in res.iter_mut().enumerate() {
                    r.rank = i;
                }
                Ok(res)
            }
            ReorderStrategy::Descending | ReorderStrategy::Custom => {
                for (i, r) in sorted.iter_mut().enumerate() {
                    r.rank = i;
                }
                Ok(sorted)
            }
        }
    }
    /// Reorder and also return a [`ReorderReport`].
    ///
    /// # Errors
    ///
    /// Returns [`LostInMiddleError::EmptyResults`] when `results` is empty.
    pub fn reorder_with_report(
        &self,
        results: &[crate::types::SearchResult],
    ) -> Result<(Vec<crate::types::SearchResult>, ReorderReport), LostInMiddleError> {
        let orig: Vec<String> = results
            .iter()
            .map(|r| r.document.id.as_str().to_string())
            .collect();
        let reordered = self.reorder(results)?;
        let new_order: Vec<String> = reordered
            .iter()
            .map(|r| r.document.id.as_str().to_string())
            .collect();
        Ok((
            reordered,
            ReorderReport {
                original_order: orig,
                new_order,
                strategy: self.config.strategy.clone(),
            },
        ))
    }
}
impl Default for LostInMiddleReorderer {
    fn default() -> Self {
        Self::new(ReorderConfig::default())
    }
}
// ── LostInMiddleError ─────────────────────────────────────────────────────────
/// Errors from the `lost_in_middle` module.
#[derive(Debug, Error)]
pub enum LostInMiddleError {
    /// No results were provided to reorder.
    #[error("Results list must not be empty")]
    EmptyResults,
}
