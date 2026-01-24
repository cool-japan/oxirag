//! Core types for the distillation module.
//!
//! This module provides types for on-the-fly distillation, enabling automatic
//! generation of specialized lightweight models for frequent queries.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Configuration for distillation operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Minimum frequency a pattern must reach before being considered for distillation.
    pub min_frequency_threshold: u32,
    /// Similarity threshold for pattern matching (0.0 - 1.0).
    pub similarity_threshold: f32,
    /// Maximum number of candidates to track.
    pub max_candidates: usize,
    /// Time window in seconds for collecting Q&A pairs.
    pub collection_window_secs: u64,
    /// Maximum Q&A pairs to collect per pattern.
    pub max_qa_pairs_per_pattern: usize,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            min_frequency_threshold: 5,
            similarity_threshold: 0.85,
            max_candidates: 100,
            collection_window_secs: 3600, // 1 hour
            max_qa_pairs_per_pattern: 50,
        }
    }
}

/// A normalized query pattern used for frequency tracking.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryPattern {
    /// The normalized text of the query.
    pub normalized_text: String,
    /// A hash of the normalized pattern for quick comparison.
    pub pattern_hash: u64,
    /// Optional category classification for the pattern.
    pub category: Option<String>,
}

impl QueryPattern {
    /// Create a new query pattern from a raw query string.
    #[must_use]
    pub fn new(query: &str) -> Self {
        let normalized = Self::normalize(query);
        let hash = Self::compute_hash(&normalized);
        Self {
            normalized_text: normalized,
            pattern_hash: hash,
            category: None,
        }
    }

    /// Create a new query pattern with a category.
    #[must_use]
    pub fn with_category(query: &str, category: impl Into<String>) -> Self {
        let normalized = Self::normalize(query);
        let hash = Self::compute_hash(&normalized);
        Self {
            normalized_text: normalized,
            pattern_hash: hash,
            category: Some(category.into()),
        }
    }

    /// Normalize a query string for pattern matching.
    ///
    /// This performs:
    /// - Lowercase conversion
    /// - Punctuation removal
    /// - Whitespace normalization
    /// - Trimming
    #[must_use]
    pub fn normalize(query: &str) -> String {
        query
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Compute a hash for the normalized pattern.
    fn compute_hash(normalized: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if this pattern is similar to another based on normalized text.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn is_similar_to(&self, other: &Self, threshold: f32) -> bool {
        if self.pattern_hash == other.pattern_hash {
            return true;
        }
        // Simple word overlap similarity
        let self_words: std::collections::HashSet<_> =
            self.normalized_text.split_whitespace().collect();
        let other_words: std::collections::HashSet<_> =
            other.normalized_text.split_whitespace().collect();

        if self_words.is_empty() || other_words.is_empty() {
            return false;
        }

        let intersection = self_words.intersection(&other_words).count();
        let union = self_words.union(&other_words).count();

        if union == 0 {
            return false;
        }

        let similarity = intersection as f32 / union as f32;
        similarity >= threshold
    }
}

/// A collected Question-Answer pair for distillation training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAPair {
    /// The original query text.
    pub query: String,
    /// The generated answer.
    pub answer: String,
    /// Confidence score of the answer (0.0 - 1.0).
    pub confidence: f32,
    /// The normalized pattern this Q&A belongs to.
    pub pattern: QueryPattern,
    /// Unix timestamp when this pair was collected.
    pub collected_at: u64,
}

impl QAPair {
    /// Create a new Q&A pair.
    #[must_use]
    pub fn new(query: &str, answer: &str, confidence: f32, pattern: QueryPattern) -> Self {
        Self {
            query: query.to_string(),
            answer: answer.to_string(),
            confidence,
            pattern,
            collected_at: current_timestamp(),
        }
    }

    /// Check if this Q&A pair is within the collection window.
    #[must_use]
    pub fn is_within_window(&self, window_secs: u64) -> bool {
        let now = current_timestamp();
        now.saturating_sub(self.collected_at) <= window_secs
    }
}

/// A candidate pattern ready for distillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationCandidate {
    /// The query pattern being tracked.
    pub pattern: QueryPattern,
    /// How many times this pattern has been seen.
    pub frequency: u32,
    /// Collected Q&A pairs for training.
    pub qa_pairs: Vec<QAPair>,
    /// Average confidence across all Q&A pairs.
    pub avg_confidence: f32,
    /// Unix timestamp when first seen.
    pub first_seen: u64,
    /// Unix timestamp when last seen.
    pub last_seen: u64,
    /// Whether this candidate meets distillation criteria.
    pub ready_for_distillation: bool,
}

impl DistillationCandidate {
    /// Create a new distillation candidate.
    #[must_use]
    pub fn new(pattern: QueryPattern) -> Self {
        let now = current_timestamp();
        Self {
            pattern,
            frequency: 1,
            qa_pairs: Vec::new(),
            avg_confidence: 0.0,
            first_seen: now,
            last_seen: now,
            ready_for_distillation: false,
        }
    }

    /// Record a new occurrence of this pattern.
    pub fn record_occurrence(&mut self) {
        self.frequency += 1;
        self.last_seen = current_timestamp();
    }

    /// Add a Q&A pair to this candidate.
    pub fn add_qa_pair(&mut self, pair: QAPair, max_pairs: usize) {
        if self.qa_pairs.len() < max_pairs {
            self.qa_pairs.push(pair);
            self.recalculate_avg_confidence();
        }
    }

    /// Recalculate the average confidence from all Q&A pairs.
    #[allow(clippy::cast_precision_loss)]
    fn recalculate_avg_confidence(&mut self) {
        if self.qa_pairs.is_empty() {
            self.avg_confidence = 0.0;
        } else {
            let total: f32 = self.qa_pairs.iter().map(|p| p.confidence).sum();
            self.avg_confidence = total / self.qa_pairs.len() as f32;
        }
    }

    /// Clean up Q&A pairs outside the collection window.
    pub fn cleanup_expired_pairs(&mut self, window_secs: u64) {
        self.qa_pairs
            .retain(|pair| pair.is_within_window(window_secs));
        self.recalculate_avg_confidence();
    }

    /// Check if this candidate meets the readiness criteria.
    #[must_use]
    pub fn check_readiness(&self, config: &DistillationConfig) -> bool {
        self.frequency >= config.min_frequency_threshold
            && !self.qa_pairs.is_empty()
            && self.avg_confidence >= config.similarity_threshold
    }

    /// Update the readiness status based on configuration.
    pub fn update_readiness(&mut self, config: &DistillationConfig) {
        self.ready_for_distillation = self.check_readiness(config);
    }
}

/// Statistics about distillation tracking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistillationStats {
    /// Total number of queries tracked.
    pub total_queries_tracked: u64,
    /// Number of unique patterns identified.
    pub unique_patterns: usize,
    /// Number of candidates ready for distillation.
    pub candidates_ready: usize,
    /// Total Q&A pairs collected.
    pub qa_pairs_collected: usize,
}

/// Get the current Unix timestamp in seconds.
#[must_use]
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_query_pattern_normalization() {
        let pattern = QueryPattern::new("What is the CAPITAL of France?!");
        assert_eq!(pattern.normalized_text, "what is the capital of france");
    }

    #[test]
    fn test_query_pattern_hash_consistency() {
        let pattern1 = QueryPattern::new("What is Rust?");
        let pattern2 = QueryPattern::new("what is rust");
        assert_eq!(pattern1.pattern_hash, pattern2.pattern_hash);
    }

    #[test]
    fn test_query_pattern_with_category() {
        let pattern = QueryPattern::with_category("How to install Rust?", "installation");
        assert_eq!(pattern.category, Some("installation".to_string()));
    }

    #[test]
    fn test_pattern_similarity() {
        let pattern1 = QueryPattern::new("What is the capital of France");
        let pattern2 = QueryPattern::new("What is the capital of Germany");
        assert!(pattern1.is_similar_to(&pattern2, 0.5));
        assert!(!pattern1.is_similar_to(&pattern2, 0.9));
    }

    #[test]
    fn test_qa_pair_creation() {
        let pattern = QueryPattern::new("What is Rust?");
        let pair = QAPair::new(
            "What is Rust?",
            "Rust is a systems programming language.",
            0.95,
            pattern,
        );
        assert_eq!(pair.confidence, 0.95);
        assert!(pair.collected_at > 0);
    }

    #[test]
    fn test_distillation_candidate_frequency() {
        let pattern = QueryPattern::new("test query");
        let mut candidate = DistillationCandidate::new(pattern);
        assert_eq!(candidate.frequency, 1);

        candidate.record_occurrence();
        candidate.record_occurrence();
        assert_eq!(candidate.frequency, 3);
    }

    #[test]
    fn test_distillation_candidate_avg_confidence() {
        let pattern = QueryPattern::new("test query");
        let mut candidate = DistillationCandidate::new(pattern.clone());

        candidate.add_qa_pair(QAPair::new("q1", "a1", 0.8, pattern.clone()), 10);
        candidate.add_qa_pair(QAPair::new("q2", "a2", 0.9, pattern.clone()), 10);
        candidate.add_qa_pair(QAPair::new("q3", "a3", 1.0, pattern), 10);

        let expected_avg = (0.8 + 0.9 + 1.0) / 3.0;
        assert!((candidate.avg_confidence - expected_avg).abs() < 0.001);
    }

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert_eq!(config.min_frequency_threshold, 5);
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.max_candidates, 100);
    }

    #[test]
    fn test_candidate_readiness_check() {
        let config = DistillationConfig {
            min_frequency_threshold: 3,
            similarity_threshold: 0.7,
            ..Default::default()
        };

        let pattern = QueryPattern::new("frequent query");
        let mut candidate = DistillationCandidate::new(pattern.clone());

        // Not ready: frequency too low
        assert!(!candidate.check_readiness(&config));

        candidate.record_occurrence();
        candidate.record_occurrence();

        // Not ready: no Q&A pairs
        assert!(!candidate.check_readiness(&config));

        candidate.add_qa_pair(QAPair::new("q", "a", 0.9, pattern), 10);

        // Now ready
        assert!(candidate.check_readiness(&config));
    }

    #[test]
    fn test_distillation_stats_default() {
        let stats = DistillationStats::default();
        assert_eq!(stats.total_queries_tracked, 0);
        assert_eq!(stats.unique_patterns, 0);
        assert_eq!(stats.candidates_ready, 0);
        assert_eq!(stats.qa_pairs_collected, 0);
    }
}
