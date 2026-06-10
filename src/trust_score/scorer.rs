//! Core trust-scoring logic combining grounding, consistency, source quality, and completeness.

use super::types::{TrustComponents, TrustConfig, TrustError, TrustScore};
use crate::consistency_checker::ConsistencyReport;
use crate::hallucination_detector::HallucinationReport;

// ── component helpers ─────────────────────────────────────────────────────────

/// Derive a grounding score from a `HallucinationReport`.
///
/// `1.0 - hallucination_rate`, clamped to `[0.0, 1.0]`.
fn grounding_from_hallucination(report: &HallucinationReport) -> f32 {
    (1.0_f32 - report.hallucination_rate).clamp(0.0, 1.0)
}

/// Derive a consistency score from a `ConsistencyReport`.
///
/// Directly uses `report.score` which is already in `[0.0, 1.0]`.
fn consistency_from_report(report: &ConsistencyReport) -> f32 {
    report.score.clamp(0.0, 1.0)
}

/// Derive a source-quality score from the number of sources provided.
///
/// Up to 5 sources yields a full quality score of `1.0`; fewer scale linearly.
#[allow(clippy::cast_precision_loss)]
fn source_quality_from_count(source_count: usize) -> f32 {
    (source_count.min(5) as f32 / 5.0_f32).clamp(0.0, 1.0)
}

/// Derive a completeness score by measuring query-term coverage in the answer.
///
/// Returns the fraction of query tokens that appear (case-insensitively) in
/// `answer`.  Returns `0.5` when `query` is empty.
fn completeness_from_answer(answer: &str, query: &str) -> f32 {
    if query.trim().is_empty() {
        return 0.5;
    }

    let answer_lower = answer.to_lowercase();
    let query_tokens: Vec<String> = query
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .collect();

    if query_tokens.is_empty() {
        return 0.5;
    }

    #[allow(clippy::cast_precision_loss)]
    let matched = query_tokens
        .iter()
        .filter(|t| answer_lower.contains(t.as_str()))
        .count() as f32;

    #[allow(clippy::cast_precision_loss)]
    let total = query_tokens.len() as f32;

    (matched / total).clamp(0.0, 1.0)
}

// ── TrustScorer ───────────────────────────────────────────────────────────────

/// Composite trust scorer for RAG-generated answers.
///
/// Combines grounding (from hallucination detection), consistency, source
/// quality, and query-answer completeness into a single weighted score.
#[derive(Debug, Clone)]
pub struct TrustScorer {
    /// Configuration including per-component weights and trustworthiness threshold.
    pub config: TrustConfig,
}

impl TrustScorer {
    /// Construct a scorer with the given configuration.
    #[must_use]
    pub fn new(config: TrustConfig) -> Self {
        Self { config }
    }

    /// Compute a full trust score for the given answer and supporting reports.
    ///
    /// # Errors
    ///
    /// Returns [`TrustError::InsufficientData`] when `answer` is empty **and**
    /// `source_count` is zero simultaneously.
    pub fn score(
        &self,
        answer: &str,
        query: &str,
        source_count: usize,
        hallucination: &HallucinationReport,
        consistency: &ConsistencyReport,
    ) -> Result<TrustScore, TrustError> {
        if answer.trim().is_empty() && source_count == 0 {
            return Err(TrustError::InsufficientData);
        }

        // Raw component scores
        let grounding = grounding_from_hallucination(hallucination);
        let consistency_score = consistency_from_report(consistency);
        let source_quality = source_quality_from_count(source_count);
        let completeness = completeness_from_answer(answer, query);

        let raw_components = TrustComponents {
            grounding,
            consistency: consistency_score,
            source_quality,
            completeness,
        };

        // Normalise weights so they always sum to 1.0
        let weights = self.config.weights.normalize();

        let overall = (weights.grounding * grounding
            + weights.consistency * consistency_score
            + weights.source_quality * source_quality
            + weights.completeness * completeness)
            .clamp(0.0, 1.0);

        // Confidence = minimum of the four components (conservative estimate)
        let confidence = grounding
            .min(consistency_score)
            .min(source_quality)
            .min(completeness)
            .clamp(0.0, 1.0);

        Ok(TrustScore::new(overall, confidence, raw_components))
    }

    /// Quick trust estimate from answer length / source count alone — no ML reports needed.
    ///
    /// Returns the average of `source_quality_from_count` and
    /// `completeness_from_answer("", answer)`.
    #[must_use]
    pub fn score_simple(&self, answer: &str, source_count: usize) -> f32 {
        let sq = source_quality_from_count(source_count);
        let comp = completeness_from_answer(answer, "");
        f32::midpoint(sq, comp).clamp(0.0, 1.0)
    }
}

impl Default for TrustScorer {
    fn default() -> Self {
        Self::new(TrustConfig::default())
    }
}
