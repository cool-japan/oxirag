//! Candidate detection for distillation.
//!
//! This module provides the `CandidateDetector` which evaluates tracked patterns
//! and determines which are ready for distillation into specialized models.

use super::types::{DistillationCandidate, DistillationConfig, QueryPattern};
use std::cmp::Ordering;

/// Detects and ranks candidates ready for distillation.
///
/// The detector evaluates candidates based on frequency, confidence,
/// and quality metrics to determine distillation readiness.
#[derive(Debug, Clone)]
pub struct CandidateDetector {
    /// Configuration for detection thresholds.
    config: DistillationConfig,
}

impl CandidateDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: DistillationConfig) -> Self {
        Self { config }
    }

    /// Create a new detector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DistillationConfig::default())
    }

    /// Check if a candidate meets the frequency threshold.
    #[must_use]
    pub fn meets_frequency_threshold(&self, candidate: &DistillationCandidate) -> bool {
        candidate.frequency >= self.config.min_frequency_threshold
    }

    /// Check if a candidate has sufficient Q&A pairs.
    #[must_use]
    pub fn has_sufficient_pairs(&self, candidate: &DistillationCandidate) -> bool {
        !candidate.qa_pairs.is_empty()
    }

    /// Check if a candidate meets the confidence threshold.
    #[must_use]
    pub fn meets_confidence_threshold(&self, candidate: &DistillationCandidate) -> bool {
        candidate.avg_confidence >= self.config.similarity_threshold
    }

    /// Evaluate if a candidate is ready for distillation.
    #[must_use]
    pub fn is_ready(&self, candidate: &DistillationCandidate) -> bool {
        self.meets_frequency_threshold(candidate)
            && self.has_sufficient_pairs(candidate)
            && self.meets_confidence_threshold(candidate)
    }

    /// Calculate a priority score for a candidate.
    ///
    /// Higher scores indicate higher priority for distillation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn calculate_priority(&self, candidate: &DistillationCandidate) -> f64 {
        const FREQUENCY_WEIGHT: f64 = 0.4;
        const CONFIDENCE_WEIGHT: f64 = 0.35;
        const PAIRS_WEIGHT: f64 = 0.25;

        let frequency_score = f64::from(candidate.frequency);
        let confidence_score = f64::from(candidate.avg_confidence);
        let pairs_score = candidate.qa_pairs.len() as f64;

        (frequency_score * FREQUENCY_WEIGHT)
            + (confidence_score * 100.0 * CONFIDENCE_WEIGHT)
            + (pairs_score * PAIRS_WEIGHT)
    }

    /// Filter candidates that are ready for distillation.
    #[must_use]
    pub fn filter_ready<'a>(
        &self,
        candidates: &'a [DistillationCandidate],
    ) -> Vec<&'a DistillationCandidate> {
        candidates.iter().filter(|c| self.is_ready(c)).collect()
    }

    /// Rank candidates by priority (highest first).
    #[must_use]
    pub fn rank_by_priority<'a>(
        &self,
        candidates: &'a [DistillationCandidate],
    ) -> Vec<&'a DistillationCandidate> {
        let mut ranked: Vec<_> = candidates.iter().collect();
        ranked.sort_by(|a, b| {
            let score_a = self.calculate_priority(a);
            let score_b = self.calculate_priority(b);
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });
        ranked
    }

    /// Get the top N candidates ready for distillation, ranked by priority.
    #[must_use]
    pub fn top_ready_candidates<'a>(
        &self,
        candidates: &'a [DistillationCandidate],
        n: usize,
    ) -> Vec<&'a DistillationCandidate> {
        let ready = self.filter_ready(candidates);
        let mut ranked = ready;
        ranked.sort_by(|a, b| {
            let score_a = self.calculate_priority(a);
            let score_b = self.calculate_priority(b);
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });
        ranked.into_iter().take(n).collect()
    }

    /// Evaluate multiple candidates and update their readiness status.
    pub fn update_readiness(&self, candidates: &mut [DistillationCandidate]) {
        for candidate in candidates {
            candidate.ready_for_distillation = self.is_ready(candidate);
        }
    }

    /// Get detailed evaluation results for a candidate.
    #[must_use]
    pub fn evaluate(&self, candidate: &DistillationCandidate) -> CandidateEvaluation {
        CandidateEvaluation {
            pattern: candidate.pattern.clone(),
            frequency: candidate.frequency,
            frequency_threshold: self.config.min_frequency_threshold,
            meets_frequency: self.meets_frequency_threshold(candidate),
            qa_pairs_count: candidate.qa_pairs.len(),
            has_pairs: self.has_sufficient_pairs(candidate),
            avg_confidence: candidate.avg_confidence,
            confidence_threshold: self.config.similarity_threshold,
            meets_confidence: self.meets_confidence_threshold(candidate),
            priority_score: self.calculate_priority(candidate),
            is_ready: self.is_ready(candidate),
        }
    }

    /// Batch evaluate multiple candidates.
    #[must_use]
    pub fn batch_evaluate(&self, candidates: &[DistillationCandidate]) -> Vec<CandidateEvaluation> {
        candidates.iter().map(|c| self.evaluate(c)).collect()
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: DistillationConfig) {
        self.config = config;
    }

    /// Find candidates that are close to being ready.
    ///
    /// Returns candidates that meet at least some criteria but not all.
    #[must_use]
    pub fn find_near_ready<'a>(
        &self,
        candidates: &'a [DistillationCandidate],
    ) -> Vec<(&'a DistillationCandidate, NearReadyReason)> {
        candidates
            .iter()
            .filter_map(|c| {
                if self.is_ready(c) {
                    return None;
                }

                let reason = if !self.meets_frequency_threshold(c) {
                    Some(NearReadyReason::NeedsMoreFrequency {
                        current: c.frequency,
                        needed: self.config.min_frequency_threshold,
                    })
                } else if !self.has_sufficient_pairs(c) {
                    Some(NearReadyReason::NeedsPairs)
                } else if !self.meets_confidence_threshold(c) {
                    Some(NearReadyReason::NeedsHigherConfidence {
                        current: c.avg_confidence,
                        needed: self.config.similarity_threshold,
                    })
                } else {
                    None
                };

                reason.map(|r| (c, r))
            })
            .collect()
    }

    /// Calculate the overall readiness ratio of candidates.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn readiness_ratio(&self, candidates: &[DistillationCandidate]) -> f32 {
        if candidates.is_empty() {
            return 0.0;
        }

        let ready_count = candidates.iter().filter(|c| self.is_ready(c)).count();
        ready_count as f32 / candidates.len() as f32
    }
}

impl Default for CandidateDetector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Detailed evaluation result for a candidate.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct CandidateEvaluation {
    /// The evaluated pattern.
    pub pattern: QueryPattern,
    /// Current frequency.
    pub frequency: u32,
    /// Required frequency threshold.
    pub frequency_threshold: u32,
    /// Whether frequency threshold is met.
    pub meets_frequency: bool,
    /// Number of Q&A pairs.
    pub qa_pairs_count: usize,
    /// Whether there are sufficient pairs.
    pub has_pairs: bool,
    /// Average confidence score.
    pub avg_confidence: f32,
    /// Required confidence threshold.
    pub confidence_threshold: f32,
    /// Whether confidence threshold is met.
    pub meets_confidence: bool,
    /// Calculated priority score.
    pub priority_score: f64,
    /// Overall readiness status.
    pub is_ready: bool,
}

/// Reason why a candidate is near ready but not quite.
#[derive(Debug, Clone)]
pub enum NearReadyReason {
    /// Needs more query frequency.
    NeedsMoreFrequency {
        /// Current frequency count.
        current: u32,
        /// Required frequency count.
        needed: u32,
    },
    /// Needs Q&A pairs collected.
    NeedsPairs,
    /// Needs higher confidence score.
    NeedsHigherConfidence {
        /// Current average confidence.
        current: f32,
        /// Required confidence threshold.
        needed: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distillation::types::QAPair;

    fn create_test_candidate(
        query: &str,
        frequency: u32,
        pairs_count: usize,
        confidence: f32,
    ) -> DistillationCandidate {
        let pattern = QueryPattern::new(query);
        let mut candidate = DistillationCandidate::new(pattern.clone());

        // Set frequency (starts at 1)
        for _ in 1..frequency {
            candidate.record_occurrence();
        }

        // Add Q&A pairs
        for i in 0..pairs_count {
            let pair = QAPair::new(query, &format!("Answer {i}"), confidence, pattern.clone());
            candidate.add_qa_pair(pair, 100);
        }

        candidate
    }

    #[test]
    fn test_detector_creation() {
        let detector = CandidateDetector::with_defaults();
        assert_eq!(detector.config().min_frequency_threshold, 5);
    }

    #[test]
    fn test_meets_frequency_threshold() {
        let config = DistillationConfig {
            min_frequency_threshold: 3,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let low_freq = create_test_candidate("test", 2, 1, 0.9);
        let high_freq = create_test_candidate("test", 5, 1, 0.9);

        assert!(!detector.meets_frequency_threshold(&low_freq));
        assert!(detector.meets_frequency_threshold(&high_freq));
    }

    #[test]
    fn test_has_sufficient_pairs() {
        let detector = CandidateDetector::with_defaults();

        let no_pairs = create_test_candidate("test", 5, 0, 0.9);
        let has_pairs = create_test_candidate("test", 5, 1, 0.9);

        assert!(!detector.has_sufficient_pairs(&no_pairs));
        assert!(detector.has_sufficient_pairs(&has_pairs));
    }

    #[test]
    fn test_meets_confidence_threshold() {
        let config = DistillationConfig {
            similarity_threshold: 0.8,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let low_conf = create_test_candidate("test", 5, 1, 0.7);
        let high_conf = create_test_candidate("test", 5, 1, 0.9);

        assert!(!detector.meets_confidence_threshold(&low_conf));
        assert!(detector.meets_confidence_threshold(&high_conf));
    }

    #[test]
    fn test_is_ready() {
        let config = DistillationConfig {
            min_frequency_threshold: 3,
            similarity_threshold: 0.8,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let not_ready = create_test_candidate("test", 2, 1, 0.9);
        let ready = create_test_candidate("test", 5, 2, 0.9);

        assert!(!detector.is_ready(&not_ready));
        assert!(detector.is_ready(&ready));
    }

    #[test]
    fn test_calculate_priority() {
        let detector = CandidateDetector::with_defaults();

        let low_priority = create_test_candidate("low", 2, 1, 0.5);
        let high_priority = create_test_candidate("high", 10, 5, 0.95);

        let low_score = detector.calculate_priority(&low_priority);
        let high_score = detector.calculate_priority(&high_priority);

        assert!(high_score > low_score);
    }

    #[test]
    fn test_filter_ready() {
        let config = DistillationConfig {
            min_frequency_threshold: 3,
            similarity_threshold: 0.7,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let candidates = vec![
            create_test_candidate("ready1", 5, 2, 0.9),
            create_test_candidate("not_ready", 1, 0, 0.5),
            create_test_candidate("ready2", 4, 1, 0.8),
        ];

        let ready = detector.filter_ready(&candidates);
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn test_rank_by_priority() {
        let detector = CandidateDetector::with_defaults();

        let candidates = vec![
            create_test_candidate("low", 2, 1, 0.7),
            create_test_candidate("high", 10, 5, 0.95),
            create_test_candidate("medium", 5, 3, 0.85),
        ];

        let ranked = detector.rank_by_priority(&candidates);

        // High should be first (highest priority)
        assert!(ranked[0].frequency == 10);
        assert!(ranked[2].frequency == 2);
    }

    #[test]
    fn test_top_ready_candidates() {
        let config = DistillationConfig {
            min_frequency_threshold: 2,
            similarity_threshold: 0.6,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let candidates = vec![
            create_test_candidate("c1", 3, 2, 0.7),
            create_test_candidate("c2", 5, 3, 0.8),
            create_test_candidate("c3", 4, 2, 0.75),
            create_test_candidate("not_ready", 1, 0, 0.5),
        ];

        let top = detector.top_ready_candidates(&candidates, 2);
        assert_eq!(top.len(), 2);
        // c2 should be first (highest frequency and confidence)
        assert_eq!(top[0].frequency, 5);
    }

    #[test]
    fn test_evaluate() {
        let config = DistillationConfig {
            min_frequency_threshold: 3,
            similarity_threshold: 0.8,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let candidate = create_test_candidate("test query", 5, 2, 0.9);
        let eval = detector.evaluate(&candidate);

        assert_eq!(eval.frequency, 5);
        assert!(eval.meets_frequency);
        assert_eq!(eval.qa_pairs_count, 2);
        assert!(eval.has_pairs);
        assert!(eval.meets_confidence);
        assert!(eval.is_ready);
        assert!(eval.priority_score > 0.0);
    }

    #[test]
    fn test_find_near_ready() {
        let config = DistillationConfig {
            min_frequency_threshold: 5,
            similarity_threshold: 0.8,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let candidates = vec![
            create_test_candidate("ready", 5, 2, 0.9),
            create_test_candidate("needs_freq", 3, 2, 0.9),
            create_test_candidate("needs_pairs", 5, 0, 0.9),
            create_test_candidate("needs_conf", 5, 2, 0.5),
        ];

        let near_ready = detector.find_near_ready(&candidates);
        assert_eq!(near_ready.len(), 3);
    }

    #[test]
    fn test_readiness_ratio() {
        let config = DistillationConfig {
            min_frequency_threshold: 3,
            similarity_threshold: 0.7,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let candidates = vec![
            create_test_candidate("ready1", 5, 2, 0.9),
            create_test_candidate("ready2", 4, 1, 0.8),
            create_test_candidate("not_ready1", 1, 0, 0.5),
            create_test_candidate("not_ready2", 2, 0, 0.6),
        ];

        let ratio = detector.readiness_ratio(&candidates);
        assert!((ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_update_readiness() {
        let config = DistillationConfig {
            min_frequency_threshold: 3,
            similarity_threshold: 0.7,
            ..Default::default()
        };
        let detector = CandidateDetector::new(config);

        let mut candidates = vec![
            create_test_candidate("ready", 5, 2, 0.9),
            create_test_candidate("not_ready", 1, 0, 0.5),
        ];

        detector.update_readiness(&mut candidates);

        assert!(candidates[0].ready_for_distillation);
        assert!(!candidates[1].ready_for_distillation);
    }

    #[test]
    fn test_batch_evaluate() {
        let detector = CandidateDetector::with_defaults();

        let candidates = vec![
            create_test_candidate("c1", 3, 1, 0.8),
            create_test_candidate("c2", 5, 2, 0.9),
        ];

        let evaluations = detector.batch_evaluate(&candidates);
        assert_eq!(evaluations.len(), 2);
    }
}
