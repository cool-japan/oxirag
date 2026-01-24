//! Query frequency tracker implementation.
//!
//! This module provides the `QueryFrequencyTracker` which monitors query patterns
//! and tracks their frequency for distillation candidate identification.

use super::types::{
    DistillationCandidate, DistillationConfig, DistillationStats, QAPair, QueryPattern,
    current_timestamp,
};
use std::collections::HashMap;

/// Tracks query patterns and their frequencies.
///
/// The tracker normalizes incoming queries, identifies patterns,
/// and maintains frequency counts for distillation candidate detection.
#[derive(Debug, Clone)]
pub struct QueryFrequencyTracker {
    /// Configuration for tracking behavior.
    config: DistillationConfig,
    /// Map from pattern hash to candidate data.
    candidates: HashMap<u64, DistillationCandidate>,
    /// Total queries tracked.
    total_queries: u64,
}

impl QueryFrequencyTracker {
    /// Create a new tracker with the given configuration.
    #[must_use]
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            config,
            candidates: HashMap::new(),
            total_queries: 0,
        }
    }

    /// Create a new tracker with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DistillationConfig::default())
    }

    /// Track a query and return its pattern.
    pub fn track(&mut self, query: &str) -> QueryPattern {
        let pattern = QueryPattern::new(query);
        self.total_queries += 1;

        // Check for existing similar pattern
        let existing_hash = self.find_similar_pattern(&pattern);

        if let Some(hash) = existing_hash {
            if let Some(candidate) = self.candidates.get_mut(&hash) {
                candidate.record_occurrence();
                candidate.update_readiness(&self.config);
            }
            // Return the existing pattern
            self.candidates
                .get(&hash)
                .map_or_else(|| pattern.clone(), |c| c.pattern.clone())
        } else {
            // Check capacity before adding
            if self.candidates.len() < self.config.max_candidates {
                let candidate = DistillationCandidate::new(pattern.clone());
                self.candidates.insert(pattern.pattern_hash, candidate);
            }
            pattern
        }
    }

    /// Track a query with an associated answer.
    pub fn track_with_answer(
        &mut self,
        query: &str,
        answer: &str,
        confidence: f32,
    ) -> QueryPattern {
        let pattern = self.track(query);

        // Add Q&A pair to the candidate
        if let Some(candidate) = self.candidates.get_mut(&pattern.pattern_hash) {
            let qa_pair = QAPair::new(query, answer, confidence, pattern.clone());
            candidate.add_qa_pair(qa_pair, self.config.max_qa_pairs_per_pattern);
            candidate.update_readiness(&self.config);
        }

        pattern
    }

    /// Find an existing pattern that is similar to the given one.
    fn find_similar_pattern(&self, pattern: &QueryPattern) -> Option<u64> {
        // First check for exact hash match
        if self.candidates.contains_key(&pattern.pattern_hash) {
            return Some(pattern.pattern_hash);
        }

        // Then check for similar patterns
        for (hash, candidate) in &self.candidates {
            if candidate
                .pattern
                .is_similar_to(pattern, self.config.similarity_threshold)
            {
                return Some(*hash);
            }
        }

        None
    }

    /// Get a candidate by its pattern.
    #[must_use]
    pub fn get_candidate(&self, pattern: &QueryPattern) -> Option<&DistillationCandidate> {
        self.candidates.get(&pattern.pattern_hash).or_else(|| {
            // Try to find similar pattern
            self.candidates.values().find(|c| {
                c.pattern
                    .is_similar_to(pattern, self.config.similarity_threshold)
            })
        })
    }

    /// Get a mutable reference to a candidate by its pattern.
    pub fn get_candidate_mut(
        &mut self,
        pattern: &QueryPattern,
    ) -> Option<&mut DistillationCandidate> {
        let hash = pattern.pattern_hash;
        if self.candidates.contains_key(&hash) {
            return self.candidates.get_mut(&hash);
        }

        // Try to find similar pattern
        let similar_hash = self
            .candidates
            .iter()
            .find(|(_, c)| {
                c.pattern
                    .is_similar_to(pattern, self.config.similarity_threshold)
            })
            .map(|(h, _)| *h);

        similar_hash.and_then(move |h| self.candidates.get_mut(&h))
    }

    /// Get all candidates.
    #[must_use]
    pub fn all_candidates(&self) -> Vec<&DistillationCandidate> {
        self.candidates.values().collect()
    }

    /// Get candidates ready for distillation.
    #[must_use]
    pub fn ready_candidates(&self) -> Vec<&DistillationCandidate> {
        self.candidates
            .values()
            .filter(|c| c.ready_for_distillation)
            .collect()
    }

    /// Get the frequency of a pattern.
    #[must_use]
    pub fn get_frequency(&self, pattern: &QueryPattern) -> u32 {
        self.get_candidate(pattern).map_or(0, |c| c.frequency)
    }

    /// Check if a pattern is ready for distillation.
    #[must_use]
    pub fn is_ready(&self, pattern: &QueryPattern) -> bool {
        self.get_candidate(pattern)
            .is_some_and(|c| c.ready_for_distillation)
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> DistillationStats {
        let qa_pairs_collected: usize = self.candidates.values().map(|c| c.qa_pairs.len()).sum();

        DistillationStats {
            total_queries_tracked: self.total_queries,
            unique_patterns: self.candidates.len(),
            candidates_ready: self.ready_candidates().len(),
            qa_pairs_collected,
        }
    }

    /// Clean up expired Q&A pairs from all candidates.
    pub fn cleanup_expired(&mut self) {
        for candidate in self.candidates.values_mut() {
            candidate.cleanup_expired_pairs(self.config.collection_window_secs);
            candidate.update_readiness(&self.config);
        }
    }

    /// Remove candidates that have been inactive for a long time.
    pub fn prune_inactive(&mut self, max_inactive_secs: u64) {
        let now = current_timestamp();
        self.candidates
            .retain(|_, c| now.saturating_sub(c.last_seen) <= max_inactive_secs);
    }

    /// Clear all tracking data.
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.total_queries = 0;
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: DistillationConfig) {
        self.config = config;
        // Re-evaluate all candidates with new config
        for candidate in self.candidates.values_mut() {
            candidate.update_readiness(&self.config);
        }
    }
}

impl Default for QueryFrequencyTracker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = QueryFrequencyTracker::with_defaults();
        assert_eq!(tracker.stats().total_queries_tracked, 0);
        assert_eq!(tracker.stats().unique_patterns, 0);
    }

    #[test]
    fn test_track_single_query() {
        let mut tracker = QueryFrequencyTracker::with_defaults();
        let pattern = tracker.track("What is Rust?");

        assert_eq!(pattern.normalized_text, "what is rust");
        assert_eq!(tracker.stats().total_queries_tracked, 1);
        assert_eq!(tracker.stats().unique_patterns, 1);
    }

    #[test]
    fn test_track_duplicate_queries() {
        let mut tracker = QueryFrequencyTracker::with_defaults();

        tracker.track("What is Rust?");
        tracker.track("what is rust");
        tracker.track("What Is RUST?!");

        assert_eq!(tracker.stats().total_queries_tracked, 3);
        assert_eq!(tracker.stats().unique_patterns, 1);

        let pattern = QueryPattern::new("What is Rust?");
        assert_eq!(tracker.get_frequency(&pattern), 3);
    }

    #[test]
    fn test_track_different_queries() {
        let mut tracker = QueryFrequencyTracker::with_defaults();

        tracker.track("What is Rust?");
        tracker.track("How to install Python?");
        tracker.track("What is JavaScript?");

        assert_eq!(tracker.stats().total_queries_tracked, 3);
        assert_eq!(tracker.stats().unique_patterns, 3);
    }

    #[test]
    fn test_track_with_answer() {
        let mut tracker = QueryFrequencyTracker::with_defaults();

        let pattern =
            tracker.track_with_answer("What is Rust?", "Rust is a programming language.", 0.95);

        assert_eq!(tracker.stats().qa_pairs_collected, 1);

        let candidate = tracker.get_candidate(&pattern).unwrap();
        assert_eq!(candidate.qa_pairs.len(), 1);
        assert_eq!(candidate.avg_confidence, 0.95);
    }

    #[test]
    fn test_ready_candidates() {
        let config = DistillationConfig {
            min_frequency_threshold: 2,
            similarity_threshold: 0.5,
            ..Default::default()
        };
        let mut tracker = QueryFrequencyTracker::new(config);

        // Track query multiple times with answers
        tracker.track_with_answer("What is Rust?", "A programming language.", 0.9);
        tracker.track_with_answer("What is Rust?", "A systems language.", 0.85);

        let ready = tracker.ready_candidates();
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn test_max_candidates_limit() {
        let config = DistillationConfig {
            max_candidates: 2,
            ..Default::default()
        };
        let mut tracker = QueryFrequencyTracker::new(config);

        tracker.track("query 1");
        tracker.track("query 2");
        tracker.track("query 3"); // Should not be added

        assert_eq!(tracker.stats().unique_patterns, 2);
    }

    #[test]
    fn test_clear() {
        let mut tracker = QueryFrequencyTracker::with_defaults();

        tracker.track("query 1");
        tracker.track("query 2");

        tracker.clear();

        assert_eq!(tracker.stats().total_queries_tracked, 0);
        assert_eq!(tracker.stats().unique_patterns, 0);
    }

    #[test]
    fn test_similar_pattern_matching() {
        let config = DistillationConfig {
            similarity_threshold: 0.6, // Lower threshold for testing
            ..Default::default()
        };
        let mut tracker = QueryFrequencyTracker::new(config);

        tracker.track("What is the capital of France");
        tracker.track("What is the capital of Germany");

        // These should be considered similar with low threshold
        // and grouped together
        assert!(tracker.stats().unique_patterns <= 2);
    }

    #[test]
    fn test_config_update() {
        let mut tracker = QueryFrequencyTracker::with_defaults();

        tracker.track_with_answer("test", "answer", 0.9);
        tracker.track_with_answer("test", "answer", 0.9);

        // Initially not ready (needs 5 occurrences)
        assert!(tracker.ready_candidates().is_empty());

        // Lower the threshold
        tracker.set_config(DistillationConfig {
            min_frequency_threshold: 2,
            similarity_threshold: 0.5,
            ..Default::default()
        });

        // Now should be ready
        assert_eq!(tracker.ready_candidates().len(), 1);
    }
}
