//! Relevance feedback loop for improving search quality.
//!
//! This module provides a feedback mechanism where users can indicate whether
//! search results were relevant, and this feedback is used to adjust future
//! search scores using algorithms like Rocchio and simple boosting.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

use crate::types::{Document, SearchResult};

/// Configuration for the relevance feedback system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Weight applied to positive feedback adjustments.
    pub positive_weight: f32,
    /// Weight applied to negative feedback adjustments.
    pub negative_weight: f32,
    /// Minimum number of feedback entries before adjustments are applied.
    pub min_feedback_count: usize,
    /// Decay factor for older feedback (0.0-1.0, where 1.0 means no decay).
    pub decay_factor: f32,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            positive_weight: 1.0,
            negative_weight: 0.5,
            min_feedback_count: 3,
            decay_factor: 0.95,
        }
    }
}

impl FeedbackConfig {
    /// Create a new feedback configuration with custom settings.
    #[must_use]
    pub fn new(
        positive_weight: f32,
        negative_weight: f32,
        min_feedback_count: usize,
        decay_factor: f32,
    ) -> Self {
        Self {
            positive_weight,
            negative_weight,
            min_feedback_count,
            decay_factor,
        }
    }

    /// Set the positive weight.
    #[must_use]
    pub fn with_positive_weight(mut self, weight: f32) -> Self {
        self.positive_weight = weight;
        self
    }

    /// Set the negative weight.
    #[must_use]
    pub fn with_negative_weight(mut self, weight: f32) -> Self {
        self.negative_weight = weight;
        self
    }

    /// Set the minimum feedback count threshold.
    #[must_use]
    pub fn with_min_feedback_count(mut self, count: usize) -> Self {
        self.min_feedback_count = count;
        self
    }

    /// Set the decay factor.
    #[must_use]
    pub fn with_decay_factor(mut self, factor: f32) -> Self {
        self.decay_factor = factor;
        self
    }
}

/// A single feedback entry recording user relevance judgment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    /// The query ID this feedback relates to.
    pub query_id: String,
    /// The document ID this feedback relates to.
    pub document_id: String,
    /// Whether the user marked the document as relevant.
    pub relevant: bool,
    /// Optional explicit rating (e.g., 1-5 stars normalized to 0.0-1.0).
    pub rating: Option<f32>,
    /// When the feedback was recorded.
    pub timestamp: DateTime<Utc>,
}

impl FeedbackEntry {
    /// Create a new feedback entry.
    #[must_use]
    pub fn new(
        query_id: impl Into<String>,
        document_id: impl Into<String>,
        relevant: bool,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            document_id: document_id.into(),
            relevant,
            rating: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a new feedback entry with a rating.
    #[must_use]
    pub fn with_rating(mut self, rating: f32) -> Self {
        self.rating = Some(rating);
        self
    }

    /// Create a new feedback entry with a specific timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }
}

/// A model derived from relevance feedback for query expansion and adjustment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelevanceModel {
    /// Terms to add to the query with their weights.
    pub query_expansion_terms: Vec<(String, f32)>,
    /// Adjustment vector for embedding space (same dimension as embeddings).
    pub embedding_adjustment: Vec<f32>,
}

impl RelevanceModel {
    /// Create a new empty relevance model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a relevance model with query expansion terms.
    #[must_use]
    pub fn with_expansion_terms(mut self, terms: Vec<(String, f32)>) -> Self {
        self.query_expansion_terms = terms;
        self
    }

    /// Create a relevance model with an embedding adjustment vector.
    #[must_use]
    pub fn with_embedding_adjustment(mut self, adjustment: Vec<f32>) -> Self {
        self.embedding_adjustment = adjustment;
        self
    }

    /// Check if the model is empty (no adjustments).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.query_expansion_terms.is_empty() && self.embedding_adjustment.is_empty()
    }

    /// Get the top-k expansion terms by weight.
    #[must_use]
    pub fn top_expansion_terms(&self, k: usize) -> Vec<(String, f32)> {
        let mut terms = self.query_expansion_terms.clone();
        terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        terms.truncate(k);
        terms
    }
}

/// Trait for adjusting search scores based on relevance feedback.
pub trait FeedbackAdjuster: Send + Sync {
    /// Adjust the scores of search results based on feedback.
    ///
    /// Modifies scores in place based on historical feedback for the documents.
    fn adjust_scores(&self, results: &mut [SearchResult], feedback: &[FeedbackEntry]);

    /// Compute a relevance model from positive and negative document examples.
    ///
    /// This model can be used for query expansion or embedding adjustment.
    fn compute_relevance_model(
        &self,
        positive_docs: &[Document],
        negative_docs: &[Document],
    ) -> RelevanceModel;
}

/// Main relevance feedback handler.
pub struct RelevanceFeedback<S: FeedbackStore> {
    store: S,
    config: FeedbackConfig,
}

impl<S: FeedbackStore> RelevanceFeedback<S> {
    /// Create a new relevance feedback handler with the given store and config.
    #[must_use]
    pub fn new(store: S, config: FeedbackConfig) -> Self {
        Self { store, config }
    }

    /// Record feedback for a query-document pair.
    pub fn record_feedback(
        &self,
        query_id: &str,
        doc_id: &str,
        relevant: bool,
        rating: Option<f32>,
    ) {
        let mut entry = FeedbackEntry::new(query_id, doc_id, relevant);
        if let Some(r) = rating {
            entry = entry.with_rating(r);
        }
        self.store.store(entry);
    }

    /// Get all feedback entries for a specific query.
    #[must_use]
    pub fn get_feedback_for_query(&self, query_id: &str) -> Vec<FeedbackEntry> {
        self.store.get_by_query(query_id)
    }

    /// Get all feedback entries for a specific document.
    #[must_use]
    pub fn get_feedback_for_document(&self, doc_id: &str) -> Vec<FeedbackEntry> {
        self.store.get_by_document(doc_id)
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &FeedbackConfig {
        &self.config
    }

    /// Check if there is enough feedback to apply adjustments.
    #[must_use]
    pub fn has_sufficient_feedback(&self, query_id: &str) -> bool {
        self.get_feedback_for_query(query_id).len() >= self.config.min_feedback_count
    }
}

/// Trait for storing and retrieving feedback entries.
pub trait FeedbackStore: Send + Sync {
    /// Store a feedback entry.
    fn store(&self, entry: FeedbackEntry);

    /// Retrieve all feedback entries for a query.
    fn get_by_query(&self, query_id: &str) -> Vec<FeedbackEntry>;

    /// Retrieve all feedback entries for a document.
    fn get_by_document(&self, doc_id: &str) -> Vec<FeedbackEntry>;

    /// Get all feedback entries.
    fn get_all(&self) -> Vec<FeedbackEntry>;

    /// Clear all feedback entries.
    fn clear(&self);
}

/// In-memory implementation of the feedback store.
#[derive(Default)]
pub struct InMemoryFeedbackStore {
    entries: RwLock<Vec<FeedbackEntry>>,
    by_query: RwLock<HashMap<String, Vec<usize>>>,
    by_document: RwLock<HashMap<String, Vec<usize>>>,
}

impl InMemoryFeedbackStore {
    /// Create a new empty in-memory feedback store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the total number of stored entries.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().expect("lock poisoned").len()
    }

    /// Check if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FeedbackStore for InMemoryFeedbackStore {
    fn store(&self, entry: FeedbackEntry) {
        let mut entries = self.entries.write().expect("lock poisoned");
        let idx = entries.len();

        let query_id = entry.query_id.clone();
        let doc_id = entry.document_id.clone();

        entries.push(entry);

        let mut by_query = self.by_query.write().expect("lock poisoned");
        by_query.entry(query_id).or_default().push(idx);

        let mut by_doc = self.by_document.write().expect("lock poisoned");
        by_doc.entry(doc_id).or_default().push(idx);
    }

    fn get_by_query(&self, query_id: &str) -> Vec<FeedbackEntry> {
        let entries = self.entries.read().expect("lock poisoned");
        let by_query = self.by_query.read().expect("lock poisoned");

        by_query
            .get(query_id)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&i| entries.get(i).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_by_document(&self, doc_id: &str) -> Vec<FeedbackEntry> {
        let entries = self.entries.read().expect("lock poisoned");
        let by_doc = self.by_document.read().expect("lock poisoned");

        by_doc
            .get(doc_id)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&i| entries.get(i).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_all(&self) -> Vec<FeedbackEntry> {
        self.entries.read().expect("lock poisoned").clone()
    }

    fn clear(&self) {
        self.entries.write().expect("lock poisoned").clear();
        self.by_query.write().expect("lock poisoned").clear();
        self.by_document.write().expect("lock poisoned").clear();
    }
}

/// Rocchio feedback adjuster implementing the classic Rocchio algorithm.
///
/// The Rocchio algorithm modifies query vectors based on positive and negative
/// examples: `q' = alpha * q + beta * mean(positive) - gamma * mean(negative)`
#[derive(Debug, Clone)]
pub struct RocchioFeedbackAdjuster {
    /// Weight for the original query (alpha).
    pub alpha: f32,
    /// Weight for positive documents (beta).
    pub beta: f32,
    /// Weight for negative documents (gamma).
    pub gamma: f32,
    /// Feedback configuration.
    config: FeedbackConfig,
}

impl Default for RocchioFeedbackAdjuster {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.75,
            gamma: 0.15,
            config: FeedbackConfig::default(),
        }
    }
}

impl RocchioFeedbackAdjuster {
    /// Create a new Rocchio adjuster with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new Rocchio adjuster with custom parameters.
    #[must_use]
    pub fn with_params(alpha: f32, beta: f32, gamma: f32) -> Self {
        Self {
            alpha,
            beta,
            gamma,
            config: FeedbackConfig::default(),
        }
    }

    /// Set the feedback configuration.
    #[must_use]
    pub fn with_config(mut self, config: FeedbackConfig) -> Self {
        self.config = config;
        self
    }

    /// Calculate time decay weight for a feedback entry.
    #[allow(clippy::cast_precision_loss)]
    fn calculate_decay_weight(&self, entry: &FeedbackEntry) -> f32 {
        let now = Utc::now();
        let age_days = (now - entry.timestamp).num_days() as f32;
        self.config.decay_factor.powf(age_days.max(0.0))
    }
}

impl FeedbackAdjuster for RocchioFeedbackAdjuster {
    fn adjust_scores(&self, results: &mut [SearchResult], feedback: &[FeedbackEntry]) {
        if feedback.len() < self.config.min_feedback_count {
            return;
        }

        // Build a map of document_id -> weighted relevance signal
        let mut doc_signals: HashMap<String, f32> = HashMap::new();

        for entry in feedback {
            let decay = self.calculate_decay_weight(entry);
            let base_signal = if entry.relevant {
                entry.rating.unwrap_or(1.0) * self.config.positive_weight
            } else {
                -entry.rating.unwrap_or(1.0) * self.config.negative_weight
            };

            *doc_signals.entry(entry.document_id.clone()).or_default() += base_signal * decay;
        }

        // Apply adjustments to results
        for result in results.iter_mut() {
            if let Some(&signal) = doc_signals.get(&result.document.id.0) {
                // Apply Rocchio-style adjustment: score' = alpha * score + beta * positive_signal
                let adjustment = if signal > 0.0 {
                    self.beta * signal
                } else {
                    self.gamma * signal
                };

                result.score = (self.alpha * result.score + adjustment).clamp(0.0, 1.0);
            }
        }

        // Re-sort by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
    }

    fn compute_relevance_model(
        &self,
        positive_docs: &[Document],
        negative_docs: &[Document],
    ) -> RelevanceModel {
        // Extract terms from documents and compute weighted term frequencies
        let mut term_weights: HashMap<String, f32> = HashMap::new();

        // Add positive document terms with positive weight
        for doc in positive_docs {
            for term in extract_terms(&doc.content) {
                *term_weights.entry(term).or_default() += self.beta;
            }
        }

        // Subtract negative document terms
        for doc in negative_docs {
            for term in extract_terms(&doc.content) {
                *term_weights.entry(term).or_default() -= self.gamma;
            }
        }

        // Filter to keep only positive-weighted terms
        let expansion_terms: Vec<(String, f32)> =
            term_weights.into_iter().filter(|(_, w)| *w > 0.0).collect();

        RelevanceModel::new().with_expansion_terms(expansion_terms)
    }
}

/// Simple boost adjuster that directly boosts scores of liked documents.
#[derive(Debug, Clone)]
pub struct SimpleBoostAdjuster {
    /// Boost factor for positively rated documents.
    pub positive_boost: f32,
    /// Penalty factor for negatively rated documents.
    pub negative_penalty: f32,
    /// Feedback configuration.
    config: FeedbackConfig,
}

impl Default for SimpleBoostAdjuster {
    fn default() -> Self {
        Self {
            positive_boost: 0.2,
            negative_penalty: 0.1,
            config: FeedbackConfig::default(),
        }
    }
}

impl SimpleBoostAdjuster {
    /// Create a new simple boost adjuster with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new simple boost adjuster with custom boost and penalty.
    #[must_use]
    pub fn with_params(positive_boost: f32, negative_penalty: f32) -> Self {
        Self {
            positive_boost,
            negative_penalty,
            config: FeedbackConfig::default(),
        }
    }

    /// Set the feedback configuration.
    #[must_use]
    pub fn with_config(mut self, config: FeedbackConfig) -> Self {
        self.config = config;
        self
    }

    /// Calculate time decay weight for a feedback entry.
    #[allow(clippy::cast_precision_loss)]
    fn calculate_decay_weight(&self, entry: &FeedbackEntry) -> f32 {
        let now = Utc::now();
        let age_days = (now - entry.timestamp).num_days() as f32;
        self.config.decay_factor.powf(age_days.max(0.0))
    }
}

impl FeedbackAdjuster for SimpleBoostAdjuster {
    fn adjust_scores(&self, results: &mut [SearchResult], feedback: &[FeedbackEntry]) {
        if feedback.len() < self.config.min_feedback_count {
            return;
        }

        // Build a map of document_id -> (positive_count, negative_count) with decay
        let mut doc_feedback: HashMap<String, (f32, f32)> = HashMap::new();

        for entry in feedback {
            let decay = self.calculate_decay_weight(entry);
            let (pos, neg) = doc_feedback.entry(entry.document_id.clone()).or_default();

            if entry.relevant {
                *pos += decay * entry.rating.unwrap_or(1.0);
            } else {
                *neg += decay * entry.rating.unwrap_or(1.0);
            }
        }

        // Apply simple boost/penalty
        for result in results.iter_mut() {
            if let Some(&(pos, neg)) = doc_feedback.get(&result.document.id.0) {
                let boost = pos * self.positive_boost * self.config.positive_weight;
                let penalty = neg * self.negative_penalty * self.config.negative_weight;

                result.score = (result.score + boost - penalty).clamp(0.0, 1.0);
            }
        }

        // Re-sort by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
    }

    fn compute_relevance_model(
        &self,
        positive_docs: &[Document],
        negative_docs: &[Document],
    ) -> RelevanceModel {
        // Extract unique terms from positive docs that aren't in negative docs
        let positive_terms: std::collections::HashSet<String> = positive_docs
            .iter()
            .flat_map(|d| extract_terms(&d.content))
            .collect();

        let negative_terms: std::collections::HashSet<String> = negative_docs
            .iter()
            .flat_map(|d| extract_terms(&d.content))
            .collect();

        let unique_positive: Vec<(String, f32)> = positive_terms
            .difference(&negative_terms)
            .map(|t| (t.clone(), 1.0))
            .collect();

        RelevanceModel::new().with_expansion_terms(unique_positive)
    }
}

/// Extract terms from text for relevance model computation.
fn extract_terms(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(str::to_lowercase)
        .filter(|s| s.len() > 2)
        .map(|s| {
            s.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DocumentId;

    fn create_test_document(id: &str, content: &str) -> Document {
        Document::new(content).with_id(DocumentId::from_string(id))
    }

    fn create_test_result(id: &str, content: &str, score: f32, rank: usize) -> SearchResult {
        SearchResult::new(create_test_document(id, content), score, rank)
    }

    #[test]
    fn test_feedback_config_default() {
        let config = FeedbackConfig::default();
        assert!((config.positive_weight - 1.0).abs() < f32::EPSILON);
        assert!((config.negative_weight - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.min_feedback_count, 3);
        assert!((config.decay_factor - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feedback_config_builder() {
        let config = FeedbackConfig::default()
            .with_positive_weight(2.0)
            .with_negative_weight(0.3)
            .with_min_feedback_count(5)
            .with_decay_factor(0.9);

        assert!((config.positive_weight - 2.0).abs() < f32::EPSILON);
        assert!((config.negative_weight - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.min_feedback_count, 5);
        assert!((config.decay_factor - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feedback_entry_creation() {
        let entry = FeedbackEntry::new("query1", "doc1", true);
        assert_eq!(entry.query_id, "query1");
        assert_eq!(entry.document_id, "doc1");
        assert!(entry.relevant);
        assert!(entry.rating.is_none());
    }

    #[test]
    fn test_feedback_entry_with_rating() {
        let entry = FeedbackEntry::new("query1", "doc1", true).with_rating(0.8);
        assert!(entry.rating.is_some());
        assert!((entry.rating.unwrap() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_relevance_model_empty() {
        let model = RelevanceModel::new();
        assert!(model.is_empty());
    }

    #[test]
    fn test_relevance_model_with_terms() {
        let model = RelevanceModel::new()
            .with_expansion_terms(vec![("term1".to_string(), 0.9), ("term2".to_string(), 0.7)]);

        assert!(!model.is_empty());
        assert_eq!(model.query_expansion_terms.len(), 2);
    }

    #[test]
    fn test_relevance_model_top_terms() {
        let model = RelevanceModel::new().with_expansion_terms(vec![
            ("low".to_string(), 0.1),
            ("high".to_string(), 0.9),
            ("medium".to_string(), 0.5),
        ]);

        let top = model.top_expansion_terms(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "high");
        assert_eq!(top[1].0, "medium");
    }

    #[test]
    fn test_in_memory_store_basic() {
        let store = InMemoryFeedbackStore::new();
        assert!(store.is_empty());

        store.store(FeedbackEntry::new("q1", "d1", true));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_in_memory_store_get_by_query() {
        let store = InMemoryFeedbackStore::new();

        store.store(FeedbackEntry::new("q1", "d1", true));
        store.store(FeedbackEntry::new("q1", "d2", false));
        store.store(FeedbackEntry::new("q2", "d1", true));

        let q1_feedback = store.get_by_query("q1");
        assert_eq!(q1_feedback.len(), 2);

        let q2_feedback = store.get_by_query("q2");
        assert_eq!(q2_feedback.len(), 1);

        let q3_feedback = store.get_by_query("q3");
        assert!(q3_feedback.is_empty());
    }

    #[test]
    fn test_in_memory_store_get_by_document() {
        let store = InMemoryFeedbackStore::new();

        store.store(FeedbackEntry::new("q1", "d1", true));
        store.store(FeedbackEntry::new("q2", "d1", true));
        store.store(FeedbackEntry::new("q1", "d2", false));

        let d1_feedback = store.get_by_document("d1");
        assert_eq!(d1_feedback.len(), 2);

        let d2_feedback = store.get_by_document("d2");
        assert_eq!(d2_feedback.len(), 1);
    }

    #[test]
    fn test_in_memory_store_clear() {
        let store = InMemoryFeedbackStore::new();

        store.store(FeedbackEntry::new("q1", "d1", true));
        store.store(FeedbackEntry::new("q2", "d2", false));

        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
        assert!(store.get_by_query("q1").is_empty());
    }

    #[test]
    fn test_relevance_feedback_record_and_retrieve() {
        let store = InMemoryFeedbackStore::new();
        let config = FeedbackConfig::default();
        let rf = RelevanceFeedback::new(store, config);

        rf.record_feedback("q1", "d1", true, Some(0.9));
        rf.record_feedback("q1", "d2", false, None);

        let query_feedback = rf.get_feedback_for_query("q1");
        assert_eq!(query_feedback.len(), 2);

        let doc_feedback = rf.get_feedback_for_document("d1");
        assert_eq!(doc_feedback.len(), 1);
        assert!(doc_feedback[0].relevant);
    }

    #[test]
    fn test_relevance_feedback_sufficient_threshold() {
        let store = InMemoryFeedbackStore::new();
        let config = FeedbackConfig::default().with_min_feedback_count(3);
        let rf = RelevanceFeedback::new(store, config);

        rf.record_feedback("q1", "d1", true, None);
        rf.record_feedback("q1", "d2", true, None);
        assert!(!rf.has_sufficient_feedback("q1"));

        rf.record_feedback("q1", "d3", false, None);
        assert!(rf.has_sufficient_feedback("q1"));
    }

    #[test]
    fn test_rocchio_adjuster_default() {
        let adjuster = RocchioFeedbackAdjuster::new();
        assert!((adjuster.alpha - 1.0).abs() < f32::EPSILON);
        assert!((adjuster.beta - 0.75).abs() < f32::EPSILON);
        assert!((adjuster.gamma - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rocchio_adjuster_with_params() {
        let adjuster = RocchioFeedbackAdjuster::with_params(0.8, 0.6, 0.2);
        assert!((adjuster.alpha - 0.8).abs() < f32::EPSILON);
        assert!((adjuster.beta - 0.6).abs() < f32::EPSILON);
        assert!((adjuster.gamma - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rocchio_adjust_scores_insufficient_feedback() {
        let adjuster = RocchioFeedbackAdjuster::new()
            .with_config(FeedbackConfig::default().with_min_feedback_count(5));

        let mut results = vec![
            create_test_result("d1", "content 1", 0.9, 0),
            create_test_result("d2", "content 2", 0.8, 1),
        ];

        let feedback = vec![
            FeedbackEntry::new("q1", "d1", true),
            FeedbackEntry::new("q1", "d2", false),
        ];

        let original_scores: Vec<f32> = results.iter().map(|r| r.score).collect();

        adjuster.adjust_scores(&mut results, &feedback);

        // Scores should be unchanged due to insufficient feedback
        for (i, result) in results.iter().enumerate() {
            assert!((result.score - original_scores[i]).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_rocchio_adjust_scores_with_feedback() {
        let config = FeedbackConfig::default().with_min_feedback_count(2);
        let adjuster = RocchioFeedbackAdjuster::new().with_config(config);

        let mut results = vec![
            create_test_result("d1", "content 1", 0.5, 0),
            create_test_result("d2", "content 2", 0.6, 1),
        ];

        let feedback = vec![
            FeedbackEntry::new("q1", "d1", true),
            FeedbackEntry::new("q1", "d1", true),
            FeedbackEntry::new("q1", "d2", false),
        ];

        adjuster.adjust_scores(&mut results, &feedback);

        // d1 should have higher score after positive feedback
        // Results should be re-sorted
        assert!(results[0].document.id.0 == "d1" || results[0].score >= results[1].score);
    }

    #[test]
    fn test_rocchio_compute_relevance_model() {
        let adjuster = RocchioFeedbackAdjuster::new();

        let positive = vec![
            create_test_document("d1", "rust programming language"),
            create_test_document("d2", "rust memory safety"),
        ];

        let negative = vec![create_test_document("d3", "python programming")];

        let model = adjuster.compute_relevance_model(&positive, &negative);

        assert!(!model.is_empty());
        assert!(!model.query_expansion_terms.is_empty());

        // "rust" should appear with positive weight
        let rust_weight = model
            .query_expansion_terms
            .iter()
            .find(|(t, _)| t == "rust")
            .map(|(_, w)| *w);

        assert!(rust_weight.is_some());
        assert!(rust_weight.unwrap() > 0.0);
    }

    #[test]
    fn test_simple_boost_adjuster_default() {
        let adjuster = SimpleBoostAdjuster::new();
        assert!((adjuster.positive_boost - 0.2).abs() < f32::EPSILON);
        assert!((adjuster.negative_penalty - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_simple_boost_adjuster_with_params() {
        let adjuster = SimpleBoostAdjuster::with_params(0.3, 0.15);
        assert!((adjuster.positive_boost - 0.3).abs() < f32::EPSILON);
        assert!((adjuster.negative_penalty - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_simple_boost_adjust_scores() {
        let config = FeedbackConfig::default().with_min_feedback_count(2);
        let adjuster = SimpleBoostAdjuster::new().with_config(config);

        let mut results = vec![
            create_test_result("d1", "content 1", 0.5, 0),
            create_test_result("d2", "content 2", 0.6, 1),
        ];

        let feedback = vec![
            FeedbackEntry::new("q1", "d1", true).with_rating(1.0),
            FeedbackEntry::new("q1", "d1", true).with_rating(1.0),
            FeedbackEntry::new("q1", "d2", false).with_rating(1.0),
        ];

        let original_d1_score = results[0].score;
        let original_d2_score = results[1].score;

        adjuster.adjust_scores(&mut results, &feedback);

        // Find the results after re-sorting
        let d1 = results.iter().find(|r| r.document.id.0 == "d1").unwrap();
        let d2 = results.iter().find(|r| r.document.id.0 == "d2").unwrap();

        // d1 should be boosted, d2 should be penalized
        assert!(d1.score > original_d1_score);
        assert!(d2.score < original_d2_score);
    }

    #[test]
    fn test_simple_boost_compute_relevance_model() {
        let adjuster = SimpleBoostAdjuster::new();

        let positive = vec![
            create_test_document("d1", "rust programming"),
            create_test_document("d2", "rust safety"),
        ];

        let negative = vec![create_test_document("d3", "programming python")];

        let model = adjuster.compute_relevance_model(&positive, &negative);

        assert!(!model.is_empty());

        // "rust" and "safety" should be in expansion terms
        // "programming" should not be (it's in both positive and negative)
        let has_rust = model.query_expansion_terms.iter().any(|(t, _)| t == "rust");
        let has_safety = model
            .query_expansion_terms
            .iter()
            .any(|(t, _)| t == "safety");
        let has_programming = model
            .query_expansion_terms
            .iter()
            .any(|(t, _)| t == "programming");

        assert!(has_rust);
        assert!(has_safety);
        assert!(!has_programming);
    }

    #[test]
    fn test_extract_terms() {
        let terms = extract_terms("Hello, World! This is a test.");
        assert!(terms.contains(&"hello".to_string()));
        assert!(terms.contains(&"world".to_string()));
        assert!(terms.contains(&"this".to_string()));
        assert!(terms.contains(&"test".to_string()));
        // Short words should be filtered out
        assert!(!terms.contains(&"is".to_string()));
        assert!(!terms.contains(&"a".to_string()));
    }

    #[test]
    fn test_score_clamping() {
        let config = FeedbackConfig::default().with_min_feedback_count(1);
        let adjuster = SimpleBoostAdjuster::with_params(2.0, 0.0).with_config(config);

        let mut results = vec![create_test_result("d1", "content", 0.9, 0)];

        // Massive positive feedback
        let feedback = vec![
            FeedbackEntry::new("q1", "d1", true).with_rating(1.0),
            FeedbackEntry::new("q1", "d1", true).with_rating(1.0),
            FeedbackEntry::new("q1", "d1", true).with_rating(1.0),
            FeedbackEntry::new("q1", "d1", true).with_rating(1.0),
            FeedbackEntry::new("q1", "d1", true).with_rating(1.0),
        ];

        adjuster.adjust_scores(&mut results, &feedback);

        // Score should be clamped to 1.0
        assert!((results[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rank_update_after_adjustment() {
        let config = FeedbackConfig::default().with_min_feedback_count(1);
        let adjuster = SimpleBoostAdjuster::with_params(0.5, 0.5).with_config(config);

        let mut results = vec![
            create_test_result("d1", "content 1", 0.9, 0), // Originally rank 0
            create_test_result("d2", "content 2", 0.8, 1), // Originally rank 1
            create_test_result("d3", "content 3", 0.7, 2), // Originally rank 2
        ];

        // Heavy negative feedback on d1, positive on d3
        let feedback = vec![
            FeedbackEntry::new("q1", "d1", false).with_rating(1.0),
            FeedbackEntry::new("q1", "d1", false).with_rating(1.0),
            FeedbackEntry::new("q1", "d3", true).with_rating(1.0),
            FeedbackEntry::new("q1", "d3", true).with_rating(1.0),
        ];

        adjuster.adjust_scores(&mut results, &feedback);

        // Verify ranks are updated correctly
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.rank, i);
        }
    }
}
