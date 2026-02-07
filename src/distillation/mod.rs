//! On-the-fly Distillation Module for `OxiRAG`.
//!
//! This module implements Core Vision #3: On-the-fly Distillation, which
//! automatically generates specialized lightweight models for frequent queries.
//!
//! # Overview
//!
//! The distillation system works by:
//! 1. Tracking query patterns and their frequencies
//! 2. Collecting Q&A pairs as training data
//! 3. Detecting candidates ready for distillation
//! 4. Preparing training data for model specialization
//!
//! # Architecture
//!
//! ```text
//! Incoming Query
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │ QueryFrequencyTracker│  ← Normalize and track patterns
//! └──────────┬──────────┘
//!            │
//!            ▼
//! ┌─────────────────────┐
//! │   QAPairCollector   │  ← Collect training data
//! └──────────┬──────────┘
//!            │
//!            ▼
//! ┌─────────────────────┐
//! │  CandidateDetector  │  ← Identify distillation candidates
//! └──────────┬──────────┘
//!            │
//!            ▼
//!   Ready for Distillation
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use oxirag::distillation::{
//!     InMemoryDistillationTracker, DistillationConfig, DistillationTracker,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create tracker with custom config
//!     let config = DistillationConfig {
//!         min_frequency_threshold: 3,
//!         similarity_threshold: 0.85,
//!         max_candidates: 100,
//!         collection_window_secs: 3600,
//!         max_qa_pairs_per_pattern: 50,
//!     };
//!
//!     let mut tracker = InMemoryDistillationTracker::new(config);
//!
//!     // Track queries as they come in
//!     tracker.track_query("What is Rust?", Some("A systems programming language."), 0.95).await?;
//!     tracker.track_query("What is Rust?", Some("Rust is a language focused on safety."), 0.90).await?;
//!
//!     // Check for candidates ready for distillation
//!     let candidates = tracker.get_candidates().await;
//!     for candidate in candidates {
//!         if candidate.ready_for_distillation {
//!             println!("Ready: {} (freq: {})", candidate.pattern.normalized_text, candidate.frequency);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - **Query Normalization**: Queries are normalized (lowercase, punctuation removed) for pattern matching
//! - **Similarity Detection**: Similar queries are grouped together even with slight variations
//! - **Time Windows**: Q&A pairs expire after a configurable time window
//! - **Priority Ranking**: Candidates are ranked by frequency, confidence, and data quality
//! - **Deduplication**: Duplicate Q&A pairs are automatically rejected

pub mod candle_lora;
pub mod collector;
pub mod detector;
pub mod feature;
pub mod hotswap;
pub mod lora;
pub mod losses;
pub mod metrics;
pub mod progressive;
pub mod registry;
pub mod teacher_student;
pub mod tracker;
pub mod traits;
pub mod trigger;
pub mod types;

// Re-export main types
pub use candle_lora::{CandleLoraConfig, CandleLoraTrainer, TrainingMetrics};
pub use collector::{CollectorStatistics, QAPairCollector, TrainingExample};
pub use detector::{CandidateDetector, CandidateEvaluation, NearReadyReason};
pub use feature::{
    AttentionTransfer, FeatureDistillation, FeatureDistillationConfig, FeatureLoss, LayerMapping,
    MockFeatureDistillation, ProjectionType,
};
pub use hotswap::{ModelSelector, ModelSelectorBuilder, SelectionStrategy, SelectorStatistics};
pub use lora::{
    LoraConfig, LoraTrainer, LoraTrainingExample, MockLoraTrainer, TrainingJob, TrainingStatus,
};
pub use losses::{
    CombinedLoss, CosineLoss, DistillationLoss, HardTargetLoss, KLDivergenceLoss, LossConfig,
    LossType, MSELoss, SoftTargetLoss, TemperatureScaling,
};
pub use metrics::{
    ComparisonResult, ComparisonSummary, DistillationEvaluator, EvalStudentModel, EvalTeacherModel,
    EvaluationResult, EvaluatorConfig, ExtraEpochMetrics, KnowledgeTransferMetrics,
    LayerSimilarity, MetricsTracker, PlotData, TestExample, TestExampleMetadata, TrackerSummary,
    TrainingEpochMetrics,
};
pub use progressive::{
    EpochMetrics, LossWeights, MockProgressiveDistillation, ModelSize, ProgressiveConfig,
    ProgressiveDistillation, ProgressiveResult, ProgressiveScheduler, StageConfig, StageResult,
};
pub use registry::{ModelMetadata, ModelMetrics, ModelRegistry, RegistryStatistics};
pub use teacher_student::{
    DistillationMetrics, DistillationPair, DistillationStepConfig, DistillationStepResult,
    MockStudentModel, MockTeacherModel, StudentModel, TeacherModel,
};
pub use tracker::QueryFrequencyTracker;
pub use traits::DistillationTracker;
pub use trigger::{DistillationTrigger, TriggerCondition, TriggerEvaluation, TriggerStatistics};
pub use types::{
    DistillationCandidate, DistillationConfig, DistillationStats, QAPair, QueryPattern,
    current_timestamp,
};

use crate::error::OxiRagError;
use async_trait::async_trait;

/// An in-memory implementation of the `DistillationTracker` trait.
///
/// This combines the frequency tracker, collector, and detector into
/// a unified interface for distillation tracking.
#[derive(Debug, Clone)]
pub struct InMemoryDistillationTracker {
    /// The frequency tracker.
    tracker: QueryFrequencyTracker,
    /// The Q&A pair collector.
    collector: QAPairCollector,
    /// The candidate detector.
    detector: CandidateDetector,
}

impl InMemoryDistillationTracker {
    /// Create a new in-memory tracker with the given configuration.
    #[must_use]
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            tracker: QueryFrequencyTracker::new(config.clone()),
            collector: QAPairCollector::new(config.clone()),
            detector: CandidateDetector::new(config),
        }
    }

    /// Create a new in-memory tracker with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DistillationConfig::default())
    }

    /// Get the frequency tracker.
    #[must_use]
    pub fn tracker(&self) -> &QueryFrequencyTracker {
        &self.tracker
    }

    /// Get a mutable reference to the frequency tracker.
    pub fn tracker_mut(&mut self) -> &mut QueryFrequencyTracker {
        &mut self.tracker
    }

    /// Get the collector.
    #[must_use]
    pub fn collector(&self) -> &QAPairCollector {
        &self.collector
    }

    /// Get a mutable reference to the collector.
    pub fn collector_mut(&mut self) -> &mut QAPairCollector {
        &mut self.collector
    }

    /// Get the detector.
    #[must_use]
    pub fn detector(&self) -> &CandidateDetector {
        &self.detector
    }

    /// Clean up expired data.
    pub fn cleanup(&mut self) {
        self.tracker.cleanup_expired();
        self.collector.cleanup_expired();
    }

    /// Get training examples for a specific pattern.
    #[must_use]
    pub fn get_training_examples(&self, pattern: &QueryPattern) -> Vec<TrainingExample> {
        self.collector.export_for_training(pattern)
    }

    /// Get all training examples.
    #[must_use]
    pub fn get_all_training_examples(&self) -> Vec<TrainingExample> {
        self.collector.export_all_for_training()
    }
}

impl Default for InMemoryDistillationTracker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[async_trait]
impl DistillationTracker for InMemoryDistillationTracker {
    async fn track_query(
        &mut self,
        query: &str,
        answer: Option<&str>,
        confidence: f32,
    ) -> Result<(), OxiRagError> {
        let pattern = if let Some(ans) = answer {
            self.tracker.track_with_answer(query, ans, confidence)
        } else {
            self.tracker.track(query)
        };

        // Also add to collector if we have an answer
        if let Some(ans) = answer {
            self.collector
                .collect_with_pattern(query, ans, confidence, pattern);
        }

        Ok(())
    }

    async fn get_candidates(&self) -> Vec<DistillationCandidate> {
        self.tracker.all_candidates().into_iter().cloned().collect()
    }

    async fn get_qa_pairs(&self, pattern: &QueryPattern) -> Vec<QAPair> {
        self.collector.get_pairs(pattern)
    }

    async fn is_ready_for_distillation(&self, pattern: &QueryPattern) -> bool {
        self.tracker.is_ready(pattern)
    }

    fn stats(&self) -> DistillationStats {
        let tracker_stats = self.tracker.stats();
        let collector_stats = self.collector.statistics();

        DistillationStats {
            total_queries_tracked: tracker_stats.total_queries_tracked,
            unique_patterns: tracker_stats.unique_patterns,
            candidates_ready: tracker_stats.candidates_ready,
            qa_pairs_collected: collector_stats.total_pairs,
        }
    }

    async fn clear(&mut self) {
        self.tracker.clear();
        self.collector.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_tracker_creation() {
        let tracker = InMemoryDistillationTracker::with_defaults();
        let stats = tracker.stats();
        assert_eq!(stats.total_queries_tracked, 0);
    }

    #[tokio::test]
    async fn test_track_query_without_answer() {
        let mut tracker = InMemoryDistillationTracker::with_defaults();

        tracker
            .track_query("What is Rust?", None, 0.0)
            .await
            .unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.total_queries_tracked, 1);
        assert_eq!(stats.qa_pairs_collected, 0);
    }

    #[tokio::test]
    async fn test_track_query_with_answer() {
        let mut tracker = InMemoryDistillationTracker::with_defaults();

        tracker
            .track_query("What is Rust?", Some("A programming language."), 0.95)
            .await
            .unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.total_queries_tracked, 1);
        assert_eq!(stats.qa_pairs_collected, 1);
    }

    #[tokio::test]
    async fn test_get_candidates() {
        let mut tracker = InMemoryDistillationTracker::with_defaults();

        tracker.track_query("query1", None, 0.0).await.unwrap();
        tracker.track_query("query2", None, 0.0).await.unwrap();

        let candidates = tracker.get_candidates().await;
        assert_eq!(candidates.len(), 2);
    }

    #[tokio::test]
    async fn test_get_qa_pairs() {
        let mut tracker = InMemoryDistillationTracker::with_defaults();

        tracker
            .track_query("What is Rust?", Some("Answer 1"), 0.9)
            .await
            .unwrap();
        tracker
            .track_query("What is Rust?", Some("Answer 2"), 0.85)
            .await
            .unwrap();

        let pattern = QueryPattern::new("What is Rust?");
        let pairs = tracker.get_qa_pairs(&pattern).await;

        assert_eq!(pairs.len(), 2);
    }

    #[tokio::test]
    async fn test_is_ready_for_distillation() {
        let config = DistillationConfig {
            min_frequency_threshold: 2,
            similarity_threshold: 0.7,
            ..Default::default()
        };
        let mut tracker = InMemoryDistillationTracker::new(config);

        // Track same query multiple times with answers
        tracker
            .track_query("test query", Some("answer 1"), 0.9)
            .await
            .unwrap();
        tracker
            .track_query("test query", Some("answer 2"), 0.85)
            .await
            .unwrap();

        let pattern = QueryPattern::new("test query");
        assert!(tracker.is_ready_for_distillation(&pattern).await);
    }

    #[tokio::test]
    async fn test_clear() {
        let mut tracker = InMemoryDistillationTracker::with_defaults();

        tracker
            .track_query("query1", Some("answer"), 0.9)
            .await
            .unwrap();
        tracker
            .track_query("query2", Some("answer"), 0.9)
            .await
            .unwrap();

        tracker.clear().await;

        let stats = tracker.stats();
        assert_eq!(stats.total_queries_tracked, 0);
        assert_eq!(stats.qa_pairs_collected, 0);
    }

    #[tokio::test]
    async fn test_get_training_examples() {
        let mut tracker = InMemoryDistillationTracker::with_defaults();

        tracker
            .track_query("What is Rust?", Some("A programming language."), 0.95)
            .await
            .unwrap();

        let pattern = QueryPattern::new("What is Rust?");
        let examples = tracker.get_training_examples(&pattern);

        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].input, "What is Rust?");
        assert_eq!(examples[0].output, "A programming language.");
    }

    #[tokio::test]
    async fn test_cleanup() {
        let mut tracker = InMemoryDistillationTracker::with_defaults();

        tracker
            .track_query("test", Some("answer"), 0.9)
            .await
            .unwrap();

        // Cleanup should not error
        tracker.cleanup();

        // Stats should still be valid (total_queries_tracked should be 1)
        let stats = tracker.stats();
        assert_eq!(stats.total_queries_tracked, 1);
    }

    #[tokio::test]
    async fn test_multiple_patterns() {
        let mut tracker = InMemoryDistillationTracker::with_defaults();

        tracker
            .track_query("What is Rust?", Some("A language."), 0.9)
            .await
            .unwrap();
        tracker
            .track_query("What is Python?", Some("Another language."), 0.85)
            .await
            .unwrap();
        tracker
            .track_query("What is JavaScript?", Some("Yet another language."), 0.8)
            .await
            .unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.total_queries_tracked, 3);
        assert_eq!(stats.unique_patterns, 3);
    }

    #[tokio::test]
    async fn test_ready_candidates_flow() {
        let config = DistillationConfig {
            min_frequency_threshold: 3,
            similarity_threshold: 0.7,
            ..Default::default()
        };
        let mut tracker = InMemoryDistillationTracker::new(config);

        // Track same query multiple times
        for i in 0..5 {
            tracker
                .track_query("frequent query", Some(&format!("answer {i}")), 0.9)
                .await
                .unwrap();
        }

        // Track another query just once
        tracker
            .track_query("rare query", Some("answer"), 0.9)
            .await
            .unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.candidates_ready, 1);

        // Verify the right pattern is ready
        let pattern = QueryPattern::new("frequent query");
        assert!(tracker.is_ready_for_distillation(&pattern).await);

        let rare_pattern = QueryPattern::new("rare query");
        assert!(!tracker.is_ready_for_distillation(&rare_pattern).await);
    }

    #[test]
    fn test_query_pattern_exports() {
        // Test that all types are properly exported
        let _config = DistillationConfig::default();
        let _pattern = QueryPattern::new("test");
        let _stats = DistillationStats::default();
    }

    #[test]
    fn test_detector_exports() {
        let detector = CandidateDetector::with_defaults();
        let _config = detector.config();
    }

    #[test]
    fn test_collector_exports() {
        let collector = QAPairCollector::with_defaults();
        let _stats = collector.statistics();
    }
}
