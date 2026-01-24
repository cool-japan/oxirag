//! Q&A pair collector for distillation training data.
//!
//! This module provides the `QAPairCollector` which manages the collection
//! of question-answer pairs grouped by query patterns.

use super::types::{DistillationConfig, QAPair, QueryPattern};
use std::collections::HashMap;

/// Collects and manages Q&A pairs for distillation training.
///
/// The collector groups Q&A pairs by their query patterns and provides
/// functionality for deduplication, time-based windowing, and export.
#[derive(Debug, Clone)]
pub struct QAPairCollector {
    /// Configuration for collection behavior.
    config: DistillationConfig,
    /// Map from pattern hash to collected Q&A pairs.
    pairs_by_pattern: HashMap<u64, Vec<QAPair>>,
    /// Total pairs collected across all patterns.
    total_collected: usize,
}

impl QAPairCollector {
    /// Create a new collector with the given configuration.
    #[must_use]
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            config,
            pairs_by_pattern: HashMap::new(),
            total_collected: 0,
        }
    }

    /// Create a new collector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DistillationConfig::default())
    }

    /// Collect a Q&A pair.
    ///
    /// Returns `true` if the pair was collected, `false` if it was rejected
    /// (e.g., duplicate or limit reached).
    pub fn collect(&mut self, query: &str, answer: &str, confidence: f32) -> bool {
        let pattern = QueryPattern::new(query);
        self.collect_with_pattern(query, answer, confidence, pattern)
    }

    /// Collect a Q&A pair with a pre-computed pattern.
    pub fn collect_with_pattern(
        &mut self,
        query: &str,
        answer: &str,
        confidence: f32,
        pattern: QueryPattern,
    ) -> bool {
        let hash = pattern.pattern_hash;

        // Get or create the pairs vector for this pattern
        let pairs = self.pairs_by_pattern.entry(hash).or_default();

        // Check if we've hit the limit for this pattern
        if pairs.len() >= self.config.max_qa_pairs_per_pattern {
            return false;
        }

        // Check for duplicate (same query and answer) - inline to avoid borrow issues
        let is_dup = pairs.iter().any(|p| p.query == query && p.answer == answer);
        if is_dup {
            return false;
        }

        // Create and store the pair
        let pair = QAPair::new(query, answer, confidence, pattern);
        pairs.push(pair);
        self.total_collected += 1;

        true
    }

    /// Get all Q&A pairs for a specific pattern.
    #[must_use]
    pub fn get_pairs(&self, pattern: &QueryPattern) -> Vec<QAPair> {
        self.pairs_by_pattern
            .get(&pattern.pattern_hash)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all Q&A pairs for patterns similar to the given one.
    #[must_use]
    pub fn get_similar_pairs(&self, pattern: &QueryPattern, threshold: f32) -> Vec<QAPair> {
        let mut result = Vec::new();

        for pairs in self.pairs_by_pattern.values() {
            if let Some(first) = pairs.first()
                && (first.pattern.is_similar_to(pattern, threshold)
                    || first.pattern.pattern_hash == pattern.pattern_hash)
            {
                result.extend(pairs.iter().cloned());
            }
        }

        result
    }

    /// Get all collected Q&A pairs.
    #[must_use]
    pub fn all_pairs(&self) -> Vec<QAPair> {
        self.pairs_by_pattern.values().flatten().cloned().collect()
    }

    /// Get the number of Q&A pairs for a specific pattern.
    #[must_use]
    pub fn count_for_pattern(&self, pattern: &QueryPattern) -> usize {
        self.pairs_by_pattern
            .get(&pattern.pattern_hash)
            .map_or(0, Vec::len)
    }

    /// Get the total number of collected pairs.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.total_collected
    }

    /// Get the number of unique patterns.
    #[must_use]
    pub fn pattern_count(&self) -> usize {
        self.pairs_by_pattern.len()
    }

    /// Remove expired pairs outside the collection window.
    pub fn cleanup_expired(&mut self) {
        let window = self.config.collection_window_secs;
        let mut removed = 0;

        for pairs in self.pairs_by_pattern.values_mut() {
            let before_len = pairs.len();
            pairs.retain(|p| p.is_within_window(window));
            removed += before_len - pairs.len();
        }

        // Remove empty pattern entries
        self.pairs_by_pattern.retain(|_, v| !v.is_empty());

        self.total_collected = self.total_collected.saturating_sub(removed);
    }

    /// Remove pairs with confidence below the threshold.
    pub fn filter_by_confidence(&mut self, min_confidence: f32) {
        let mut removed = 0;

        for pairs in self.pairs_by_pattern.values_mut() {
            let before_len = pairs.len();
            pairs.retain(|p| p.confidence >= min_confidence);
            removed += before_len - pairs.len();
        }

        self.pairs_by_pattern.retain(|_, v| !v.is_empty());
        self.total_collected = self.total_collected.saturating_sub(removed);
    }

    /// Get patterns that have enough pairs for training.
    #[must_use]
    pub fn patterns_with_min_pairs(&self, min_pairs: usize) -> Vec<QueryPattern> {
        self.pairs_by_pattern
            .iter()
            .filter(|(_, pairs)| pairs.len() >= min_pairs)
            .filter_map(|(_, pairs)| pairs.first().map(|p| p.pattern.clone()))
            .collect()
    }

    /// Export pairs for a pattern in a format suitable for training.
    #[must_use]
    pub fn export_for_training(&self, pattern: &QueryPattern) -> Vec<TrainingExample> {
        self.get_pairs(pattern)
            .into_iter()
            .map(|p| TrainingExample {
                input: p.query,
                output: p.answer,
                confidence: p.confidence,
            })
            .collect()
    }

    /// Export all pairs in a format suitable for training.
    #[must_use]
    pub fn export_all_for_training(&self) -> Vec<TrainingExample> {
        self.all_pairs()
            .into_iter()
            .map(|p| TrainingExample {
                input: p.query,
                output: p.answer,
                confidence: p.confidence,
            })
            .collect()
    }

    /// Clear all collected pairs.
    pub fn clear(&mut self) {
        self.pairs_by_pattern.clear();
        self.total_collected = 0;
    }

    /// Clear pairs for a specific pattern.
    pub fn clear_pattern(&mut self, pattern: &QueryPattern) {
        if let Some(pairs) = self.pairs_by_pattern.remove(&pattern.pattern_hash) {
            self.total_collected = self.total_collected.saturating_sub(pairs.len());
        }
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

    /// Calculate average confidence for a pattern.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_confidence(&self, pattern: &QueryPattern) -> f32 {
        let pairs = self.get_pairs(pattern);
        if pairs.is_empty() {
            return 0.0;
        }
        let total: f32 = pairs.iter().map(|p| p.confidence).sum();
        total / pairs.len() as f32
    }

    /// Get statistics about the collector.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn statistics(&self) -> CollectorStatistics {
        let all_pairs = self.all_pairs();
        let total_confidence: f32 = all_pairs.iter().map(|p| p.confidence).sum();
        let avg_confidence = if all_pairs.is_empty() {
            0.0
        } else {
            total_confidence / all_pairs.len() as f32
        };

        let pairs_per_pattern: Vec<usize> = self.pairs_by_pattern.values().map(Vec::len).collect();
        let avg_pairs_per_pattern = if pairs_per_pattern.is_empty() {
            0.0
        } else {
            pairs_per_pattern.iter().sum::<usize>() as f32 / pairs_per_pattern.len() as f32
        };

        CollectorStatistics {
            total_pairs: self.total_collected,
            unique_patterns: self.pairs_by_pattern.len(),
            average_confidence: avg_confidence,
            average_pairs_per_pattern: avg_pairs_per_pattern,
        }
    }
}

impl Default for QAPairCollector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// A training example extracted from a Q&A pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingExample {
    /// The input query.
    pub input: String,
    /// The expected output answer.
    pub output: String,
    /// Confidence score of this example.
    pub confidence: f32,
}

/// Statistics about the collector's state.
#[derive(Debug, Clone, Default)]
pub struct CollectorStatistics {
    /// Total number of pairs collected.
    pub total_pairs: usize,
    /// Number of unique patterns.
    pub unique_patterns: usize,
    /// Average confidence across all pairs.
    pub average_confidence: f32,
    /// Average pairs per pattern.
    pub average_pairs_per_pattern: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_creation() {
        let collector = QAPairCollector::with_defaults();
        assert_eq!(collector.total_count(), 0);
        assert_eq!(collector.pattern_count(), 0);
    }

    #[test]
    fn test_collect_single_pair() {
        let mut collector = QAPairCollector::with_defaults();

        let collected = collector.collect("What is Rust?", "A programming language.", 0.95);

        assert!(collected);
        assert_eq!(collector.total_count(), 1);
        assert_eq!(collector.pattern_count(), 1);
    }

    #[test]
    fn test_collect_multiple_pairs_same_pattern() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("What is Rust?", "A programming language.", 0.95);
        collector.collect("What is Rust?", "A systems programming language.", 0.90);
        collector.collect(
            "what is rust",
            "Rust is a language focused on safety.",
            0.85,
        );

        // Note: "What is Rust?" and "what is rust" normalize to the same pattern
        let pattern = QueryPattern::new("What is Rust?");
        assert_eq!(collector.count_for_pattern(&pattern), 3);
        assert_eq!(collector.pattern_count(), 1);
    }

    #[test]
    fn test_duplicate_rejection() {
        let mut collector = QAPairCollector::with_defaults();

        let first = collector.collect("What is Rust?", "A programming language.", 0.95);
        let duplicate = collector.collect("What is Rust?", "A programming language.", 0.95);

        assert!(first);
        assert!(!duplicate);
        assert_eq!(collector.total_count(), 1);
    }

    #[test]
    fn test_max_pairs_limit() {
        let config = DistillationConfig {
            max_qa_pairs_per_pattern: 2,
            ..Default::default()
        };
        let mut collector = QAPairCollector::new(config);

        collector.collect("test", "answer 1", 0.9);
        collector.collect("test", "answer 2", 0.9);
        let third = collector.collect("test", "answer 3", 0.9);

        assert!(!third);
        assert_eq!(collector.total_count(), 2);
    }

    #[test]
    fn test_get_pairs() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("What is Rust?", "Answer 1", 0.9);
        collector.collect("What is Rust?", "Answer 2", 0.85);

        let pattern = QueryPattern::new("What is Rust?");
        let pairs = collector.get_pairs(&pattern);

        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_filter_by_confidence() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("q1", "a1", 0.9);
        collector.collect("q1", "a2", 0.5);
        collector.collect("q1", "a3", 0.7);

        collector.filter_by_confidence(0.6);

        assert_eq!(collector.total_count(), 2);
    }

    #[test]
    fn test_patterns_with_min_pairs() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("query1", "a1", 0.9);
        collector.collect("query1", "a2", 0.9);
        collector.collect("query1", "a3", 0.9);
        collector.collect("query2", "b1", 0.9);

        let patterns = collector.patterns_with_min_pairs(3);
        assert_eq!(patterns.len(), 1);
    }

    #[test]
    fn test_export_for_training() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("What is Rust?", "A programming language.", 0.95);

        let pattern = QueryPattern::new("What is Rust?");
        let examples = collector.export_for_training(&pattern);

        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].input, "What is Rust?");
        assert_eq!(examples[0].output, "A programming language.");
    }

    #[test]
    fn test_clear() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("q1", "a1", 0.9);
        collector.collect("q2", "a2", 0.9);

        collector.clear();

        assert_eq!(collector.total_count(), 0);
        assert_eq!(collector.pattern_count(), 0);
    }

    #[test]
    fn test_clear_pattern() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("query1", "a1", 0.9);
        collector.collect("query2", "b1", 0.9);

        let pattern = QueryPattern::new("query1");
        collector.clear_pattern(&pattern);

        assert_eq!(collector.total_count(), 1);
        assert_eq!(collector.pattern_count(), 1);
    }

    #[test]
    fn test_average_confidence() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("test", "a1", 0.8);
        collector.collect("test", "a2", 0.9);
        collector.collect("test", "a3", 1.0);

        let pattern = QueryPattern::new("test");
        let avg = collector.average_confidence(&pattern);

        let expected = (0.8 + 0.9 + 1.0) / 3.0;
        assert!((avg - expected).abs() < 0.001);
    }

    #[test]
    fn test_statistics() {
        let mut collector = QAPairCollector::with_defaults();

        collector.collect("query1", "a1", 0.9);
        collector.collect("query1", "a2", 0.8);
        collector.collect("query2", "b1", 0.7);

        let stats = collector.statistics();

        assert_eq!(stats.total_pairs, 3);
        assert_eq!(stats.unique_patterns, 2);
        assert!((stats.average_confidence - 0.8).abs() < 0.01);
        assert!((stats.average_pairs_per_pattern - 1.5).abs() < 0.01);
    }
}
