//! Traits for the distillation module.
//!
//! This module defines the core trait for distillation tracking,
//! enabling different implementations of the distillation system.

use super::types::{DistillationCandidate, DistillationStats, QAPair, QueryPattern};
use crate::error::OxiRagError;
use async_trait::async_trait;

/// A trait for tracking queries and managing distillation candidates.
///
/// Implementations of this trait are responsible for:
/// - Tracking query frequency patterns
/// - Collecting Q&A pairs for training data
/// - Identifying candidates ready for distillation
/// - Managing statistics and cleanup
#[async_trait]
pub trait DistillationTracker: Send + Sync {
    /// Track a query and optionally collect a Q&A pair.
    ///
    /// # Arguments
    ///
    /// * `query` - The raw query string
    /// * `answer` - Optional answer to collect as training data
    /// * `confidence` - Confidence score of the answer (0.0 - 1.0)
    ///
    /// # Errors
    ///
    /// Returns an error if tracking fails due to storage or other issues.
    async fn track_query(
        &mut self,
        query: &str,
        answer: Option<&str>,
        confidence: f32,
    ) -> Result<(), OxiRagError>;

    /// Get all candidates that are ready for distillation.
    ///
    /// Returns candidates that meet the frequency and quality thresholds.
    async fn get_candidates(&self) -> Vec<DistillationCandidate>;

    /// Get Q&A pairs for a specific pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The query pattern to retrieve pairs for
    async fn get_qa_pairs(&self, pattern: &QueryPattern) -> Vec<QAPair>;

    /// Check if a pattern is ready for distillation.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The query pattern to check
    async fn is_ready_for_distillation(&self, pattern: &QueryPattern) -> bool;

    /// Get current distillation statistics.
    fn stats(&self) -> DistillationStats;

    /// Clear all tracking data.
    ///
    /// # Errors
    ///
    /// Returns an error if clearing fails.
    async fn clear(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// A mock implementation for testing the trait.
    struct MockTracker {
        candidates: Vec<DistillationCandidate>,
        qa_pairs: Vec<QAPair>,
        stats: DistillationStats,
    }

    #[async_trait]
    impl DistillationTracker for MockTracker {
        async fn track_query(
            &mut self,
            _query: &str,
            _answer: Option<&str>,
            _confidence: f32,
        ) -> Result<(), OxiRagError> {
            self.stats.total_queries_tracked += 1;
            Ok(())
        }

        async fn get_candidates(&self) -> Vec<DistillationCandidate> {
            self.candidates.clone()
        }

        async fn get_qa_pairs(&self, _pattern: &QueryPattern) -> Vec<QAPair> {
            self.qa_pairs.clone()
        }

        async fn is_ready_for_distillation(&self, _pattern: &QueryPattern) -> bool {
            true
        }

        fn stats(&self) -> DistillationStats {
            self.stats.clone()
        }

        async fn clear(&mut self) {
            self.candidates.clear();
            self.qa_pairs.clear();
            self.stats = DistillationStats::default();
        }
    }

    #[tokio::test]
    async fn test_mock_tracker_track_query() {
        let mut tracker = MockTracker {
            candidates: Vec::new(),
            qa_pairs: Vec::new(),
            stats: DistillationStats::default(),
        };

        tracker
            .track_query("test", Some("answer"), 0.9)
            .await
            .unwrap();
        assert_eq!(tracker.stats().total_queries_tracked, 1);
    }

    #[tokio::test]
    async fn test_mock_tracker_clear() {
        let mut tracker = MockTracker {
            candidates: vec![DistillationCandidate::new(QueryPattern::new("test"))],
            qa_pairs: Vec::new(),
            stats: DistillationStats {
                total_queries_tracked: 10,
                ..Default::default()
            },
        };

        tracker.clear().await;
        assert!(tracker.get_candidates().await.is_empty());
        assert_eq!(tracker.stats().total_queries_tracked, 0);
    }

    #[tokio::test]
    async fn test_trait_object_safety() {
        let tracker: Arc<Mutex<dyn DistillationTracker>> = Arc::new(Mutex::new(MockTracker {
            candidates: Vec::new(),
            qa_pairs: Vec::new(),
            stats: DistillationStats::default(),
        }));

        let mut guard = tracker.lock().await;
        guard
            .track_query("test", None, 0.5)
            .await
            .expect("tracking should work");
        assert_eq!(guard.stats().total_queries_tracked, 1);
    }
}
