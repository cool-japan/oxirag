//! The RECOMP pipeline: selective-augmentation gate followed by the
//! configured compressor.

use super::abstractive::AbstractiveSummaryCompressor;
use super::extractive::{ExtractiveSummaryCompressor, approx_token_count, jaccard, token_set};
use super::types::{CompressorStrategy, RecompConfig, RecompDecision, RecompError, RecompOutcome};

/// Computes the selective-augmentation relevance score for `passages` against
/// `query`: the mean, across all non-blank passages, of the Jaccard overlap
/// between the query's token set and each passage's token set.
///
/// A mean (rather than a max) is used so that a single highly relevant
/// passage buried among many irrelevant ones does not, by itself, force the
/// gate open — RECOMP's selective-augmentation contribution is about judging
/// whether the *retrieved set as a whole* was worth retrieving.
fn compute_relevance_score(query: &str, passages: &[String]) -> f32 {
    let query_tokens = token_set(query);
    if query_tokens.is_empty() {
        return 0.0;
    }

    let mut total = 0.0_f32;
    let mut count = 0usize;
    for passage in passages {
        if passage.trim().is_empty() {
            continue;
        }
        let passage_tokens = token_set(passage);
        total += jaccard(&query_tokens, &passage_tokens);
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let mean = total / count as f32;
        mean
    }
}

// ── RecompPipeline ─────────────────────────────────────────────────────────────

/// The RECOMP (Xu et al., ICLR 2024) compression pipeline.
///
/// On every call to [`Self::compress`], the pipeline:
///
/// 1. Validates the query, passages, and configuration.
/// 2. Computes an overall relevance score for the retrieved passage set (the
///    selective-augmentation gate).
/// 3. If the score is below
///    [`RecompConfig::augmentation_threshold`], returns
///    [`RecompDecision::Skip`] — compression is judged not worth performing.
/// 4. Otherwise, runs the configured [`CompressorStrategy`] and returns
///    [`RecompDecision::Augment`] with the compressed text.
#[derive(Debug, Clone, Default)]
pub struct RecompPipeline {
    config: RecompConfig,
    extractive: ExtractiveSummaryCompressor,
    abstractive: AbstractiveSummaryCompressor,
}

impl RecompPipeline {
    /// Creates a new [`RecompPipeline`] with the given configuration.
    #[must_use]
    pub fn new(config: RecompConfig) -> Self {
        Self {
            config,
            extractive: ExtractiveSummaryCompressor::new(),
            abstractive: AbstractiveSummaryCompressor::new(),
        }
    }

    /// Returns the pipeline's configuration.
    #[must_use]
    pub fn config(&self) -> &RecompConfig {
        &self.config
    }

    /// Runs the selective-augmentation gate and, if it passes, compresses
    /// `passages` relevant to `query` using the configured
    /// [`CompressorStrategy`].
    ///
    /// # Errors
    ///
    /// - [`RecompError::EmptyQuery`] — `query` is empty or whitespace-only.
    /// - [`RecompError::EmptyPassages`] — `passages` is empty, or every
    ///   passage is empty/whitespace-only.
    /// - [`RecompError::InvalidConfig`] — the pipeline's [`RecompConfig`]
    ///   fails [`RecompConfig::validate`].
    pub fn compress(&self, query: &str, passages: &[String]) -> Result<RecompOutcome, RecompError> {
        let trimmed_query = query.trim();
        if trimmed_query.is_empty() {
            return Err(RecompError::EmptyQuery);
        }
        if passages.is_empty() || passages.iter().all(|p| p.trim().is_empty()) {
            return Err(RecompError::EmptyPassages);
        }
        self.config.validate()?;

        let relevance_score = compute_relevance_score(trimmed_query, passages);

        if relevance_score < self.config.augmentation_threshold {
            return Ok(RecompOutcome {
                decision: RecompDecision::Skip {
                    reason: format!(
                        "relevance score {relevance_score:.3} is below augmentation_threshold \
                         {:.3}; the retrieved passages were judged low-value, so compression \
                         was skipped",
                        self.config.augmentation_threshold
                    ),
                },
                relevance_score,
                original_passage_count: passages.len(),
                compressed_token_count: 0,
            });
        }

        let compressed = match self.config.strategy {
            CompressorStrategy::Extractive => self.extractive.compress_with_dedup_threshold(
                trimmed_query,
                passages,
                self.config.token_budget,
                self.config.dedup_similarity_threshold,
            ),
            CompressorStrategy::AbstractiveLite => self.abstractive.compress_with_dedup_threshold(
                trimmed_query,
                passages,
                self.config.token_budget,
                self.config.dedup_similarity_threshold,
            ),
        };
        let compressed_token_count = approx_token_count(&compressed);

        Ok(RecompOutcome {
            decision: RecompDecision::Augment(compressed),
            relevance_score,
            original_passage_count: passages.len(),
            compressed_token_count,
        })
    }
}
