//! Hot-swap support for runtime model switching.
//!
//! This module provides functionality for dynamically selecting and switching
//! between distilled models at runtime based on various strategies.

use super::registry::ModelRegistry;
use super::types::QueryPattern;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "native")]
use tokio::sync::RwLock;

#[cfg(not(feature = "native"))]
use std::sync::RwLock;

/// Strategy for selecting which model to use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Use the model trained on the matching query pattern.
    #[default]
    PatternMatch,
    /// Use the model with the lowest average latency.
    LowestLatency,
    /// Use the model with the highest accuracy.
    HighestAccuracy,
    /// Rotate between available models.
    RoundRobin,
    /// Use the most recently used model.
    MostRecentlyUsed,
    /// Use the most frequently used model.
    MostFrequentlyUsed,
}

impl SelectionStrategy {
    /// Get a human-readable description of this strategy.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::PatternMatch => "Select model based on query pattern match",
            Self::LowestLatency => "Select the fastest model",
            Self::HighestAccuracy => "Select the most accurate model",
            Self::RoundRobin => "Rotate between available models",
            Self::MostRecentlyUsed => "Prefer recently used models",
            Self::MostFrequentlyUsed => "Prefer frequently used models",
        }
    }
}

/// Model selector for runtime model switching.
///
/// Provides functionality to select the appropriate model based on
/// the configured strategy and query characteristics.
pub struct ModelSelector {
    /// The model registry.
    registry: Arc<RwLock<ModelRegistry>>,
    /// Fallback model to use when no suitable model is found.
    fallback_model: String,
    /// Strategy for model selection.
    selection_strategy: SelectionStrategy,
    /// Counter for round-robin selection.
    round_robin_counter: AtomicUsize,
    /// History of recently used models (for MRU strategy).
    usage_history: Arc<RwLock<VecDeque<String>>>,
    /// Maximum history size.
    max_history_size: usize,
    /// Similarity threshold for pattern matching.
    similarity_threshold: f32,
}

impl ModelSelector {
    /// Create a new model selector.
    #[must_use]
    pub fn new(registry: Arc<RwLock<ModelRegistry>>, fallback_model: impl Into<String>) -> Self {
        Self {
            registry,
            fallback_model: fallback_model.into(),
            selection_strategy: SelectionStrategy::default(),
            round_robin_counter: AtomicUsize::new(0),
            usage_history: Arc::new(RwLock::new(VecDeque::new())),
            max_history_size: 100,
            similarity_threshold: 0.8,
        }
    }

    /// Set the selection strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set the similarity threshold for pattern matching.
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum history size.
    #[must_use]
    pub fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Get the current selection strategy.
    #[must_use]
    pub fn strategy(&self) -> SelectionStrategy {
        self.selection_strategy
    }

    /// Set the selection strategy.
    pub fn set_strategy(&mut self, strategy: SelectionStrategy) {
        self.selection_strategy = strategy;
    }

    /// Get the fallback model ID.
    #[must_use]
    pub fn fallback_model(&self) -> &str {
        &self.fallback_model
    }

    /// Set the fallback model ID.
    pub fn set_fallback_model(&mut self, model_id: impl Into<String>) {
        self.fallback_model = model_id.into();
    }

    /// Select a model for the given query.
    ///
    /// This is the main entry point for model selection. It uses the configured
    /// strategy to select the most appropriate model.
    #[cfg(feature = "native")]
    pub async fn select_model(&self, query: &str) -> String {
        let registry = self.registry.read().await;
        self.select_model_internal(&registry, query)
    }

    /// Select a model for the given query (non-async version for WASM).
    ///
    /// # Panics
    ///
    /// This function will not panic as lock poisoning returns the fallback model.
    #[cfg(not(feature = "native"))]
    pub fn select_model(&self, query: &str) -> String {
        let Ok(registry) = self.registry.read() else {
            return self.fallback_model.clone();
        };
        self.select_model_internal(&registry, query)
    }

    /// Internal model selection logic.
    fn select_model_internal(&self, registry: &ModelRegistry, query: &str) -> String {
        match self.selection_strategy {
            SelectionStrategy::PatternMatch => self.select_by_pattern(registry, query),
            SelectionStrategy::LowestLatency => self.select_by_latency(registry),
            SelectionStrategy::HighestAccuracy => self.select_by_accuracy(registry),
            SelectionStrategy::RoundRobin => self.select_round_robin(registry),
            SelectionStrategy::MostRecentlyUsed => self.select_mru(registry),
            SelectionStrategy::MostFrequentlyUsed => self.select_mfu(registry),
        }
    }

    /// Select model by pattern matching.
    fn select_by_pattern(&self, registry: &ModelRegistry, query: &str) -> String {
        let pattern = QueryPattern::new(query);

        registry
            .find_by_pattern_with_threshold(&pattern, self.similarity_threshold)
            .map_or_else(|| self.fallback_model.clone(), |m| m.model_id.clone())
    }

    /// Select model with lowest latency.
    fn select_by_latency(&self, registry: &ModelRegistry) -> String {
        registry
            .find_fastest()
            .map_or_else(|| self.fallback_model.clone(), |m| m.model_id.clone())
    }

    /// Select model with highest accuracy.
    fn select_by_accuracy(&self, registry: &ModelRegistry) -> String {
        registry
            .find_most_accurate()
            .map_or_else(|| self.fallback_model.clone(), |m| m.model_id.clone())
    }

    /// Select model using round-robin.
    fn select_round_robin(&self, registry: &ModelRegistry) -> String {
        let models = registry.list_active();
        if models.is_empty() {
            return self.fallback_model.clone();
        }

        let index = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
        models[index % models.len()].model_id.clone()
    }

    /// Select most recently used model.
    fn select_mru(&self, registry: &ModelRegistry) -> String {
        #[cfg(feature = "native")]
        {
            // For async context, we can't easily access the history
            // Fall back to pattern match for simplicity
            self.select_by_accuracy(registry)
        }

        #[cfg(not(feature = "native"))]
        {
            let Ok(history) = self.usage_history.read() else {
                return self.fallback_model.clone();
            };
            if let Some(model_id) = history.front() {
                if registry.get(model_id).is_some_and(|m| m.is_active) {
                    return model_id.clone();
                }
            }
            self.fallback_model.clone()
        }
    }

    /// Select most frequently used model.
    #[allow(clippy::cast_precision_loss)]
    fn select_mfu(&self, registry: &ModelRegistry) -> String {
        registry
            .find_best(|m| m.metrics.usage_count as f64)
            .map_or_else(|| self.fallback_model.clone(), |m| m.model_id.clone())
    }

    /// Report the result of using a model.
    ///
    /// This updates metrics and usage history.
    #[cfg(feature = "native")]
    pub async fn report_result(&self, model_id: &str, latency_ms: f64, success: bool) {
        // Update registry metrics
        {
            let mut registry = self.registry.write().await;
            registry.update_metrics(model_id, latency_ms, success);
        }

        // Update usage history
        {
            let mut history = self.usage_history.write().await;
            history.push_front(model_id.to_string());
            while history.len() > self.max_history_size {
                history.pop_back();
            }
        }
    }

    /// Report the result of using a model (non-async version for WASM).
    ///
    /// Silently ignores poisoned locks to maintain fault tolerance.
    #[cfg(not(feature = "native"))]
    pub fn report_result(&self, model_id: &str, latency_ms: f64, success: bool) {
        // Update registry metrics
        if let Ok(mut registry) = self.registry.write() {
            registry.update_metrics(model_id, latency_ms, success);
        }

        // Update usage history
        if let Ok(mut history) = self.usage_history.write() {
            history.push_front(model_id.to_string());
            while history.len() > self.max_history_size {
                history.pop_back();
            }
        }
    }

    /// Get the current selection statistics.
    #[cfg(feature = "native")]
    pub async fn statistics(&self) -> SelectorStatistics {
        let registry = self.registry.read().await;
        let history = self.usage_history.read().await;

        SelectorStatistics {
            strategy: self.selection_strategy,
            fallback_model: self.fallback_model.clone(),
            total_models: registry.count(),
            active_models: registry.active_count(),
            history_size: history.len(),
            round_robin_position: self.round_robin_counter.load(Ordering::Relaxed),
        }
    }

    /// Get the current selection statistics (non-async version for WASM).
    #[cfg(not(feature = "native"))]
    pub fn statistics(&self) -> SelectorStatistics {
        let registry = self.registry.read().unwrap();
        let history = self.usage_history.read().unwrap();

        SelectorStatistics {
            strategy: self.selection_strategy,
            fallback_model: self.fallback_model.clone(),
            total_models: registry.count(),
            active_models: registry.active_count(),
            history_size: history.len(),
            round_robin_position: self.round_robin_counter.load(Ordering::Relaxed),
        }
    }

    /// Clear the usage history.
    #[cfg(feature = "native")]
    pub async fn clear_history(&self) {
        let mut history = self.usage_history.write().await;
        history.clear();
    }

    /// Clear the usage history (non-async version for WASM).
    #[cfg(not(feature = "native"))]
    pub fn clear_history(&self) {
        let mut history = self.usage_history.write().unwrap();
        history.clear();
    }

    /// Reset the round-robin counter.
    pub fn reset_round_robin(&self) {
        self.round_robin_counter.store(0, Ordering::Relaxed);
    }

    /// Get a list of recently used model IDs.
    #[cfg(feature = "native")]
    pub async fn recent_models(&self, limit: usize) -> Vec<String> {
        let history = self.usage_history.read().await;
        history.iter().take(limit).cloned().collect()
    }

    /// Get a list of recently used model IDs (non-async version for WASM).
    #[cfg(not(feature = "native"))]
    pub fn recent_models(&self, limit: usize) -> Vec<String> {
        let history = self.usage_history.read().unwrap();
        history.iter().take(limit).cloned().collect()
    }
}

impl std::fmt::Debug for ModelSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSelector")
            .field("fallback_model", &self.fallback_model)
            .field("selection_strategy", &self.selection_strategy)
            .field("similarity_threshold", &self.similarity_threshold)
            .field("max_history_size", &self.max_history_size)
            .finish_non_exhaustive()
    }
}

/// Statistics about the model selector.
#[derive(Debug, Clone)]
pub struct SelectorStatistics {
    /// Current selection strategy.
    pub strategy: SelectionStrategy,
    /// Fallback model ID.
    pub fallback_model: String,
    /// Total number of registered models.
    pub total_models: usize,
    /// Number of active models.
    pub active_models: usize,
    /// Size of usage history.
    pub history_size: usize,
    /// Current round-robin position.
    pub round_robin_position: usize,
}

/// Builder for creating `ModelSelector` instances.
#[derive(Debug)]
pub struct ModelSelectorBuilder {
    registry: Arc<RwLock<ModelRegistry>>,
    fallback_model: String,
    strategy: SelectionStrategy,
    similarity_threshold: f32,
    max_history_size: usize,
}

impl ModelSelectorBuilder {
    /// Create a new builder with required parameters.
    #[must_use]
    pub fn new(registry: Arc<RwLock<ModelRegistry>>, fallback_model: impl Into<String>) -> Self {
        Self {
            registry,
            fallback_model: fallback_model.into(),
            strategy: SelectionStrategy::default(),
            similarity_threshold: 0.8,
            max_history_size: 100,
        }
    }

    /// Set the selection strategy.
    #[must_use]
    pub fn strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the similarity threshold.
    #[must_use]
    pub fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum history size.
    #[must_use]
    pub fn max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Build the `ModelSelector`.
    #[must_use]
    pub fn build(self) -> ModelSelector {
        ModelSelector {
            registry: self.registry,
            fallback_model: self.fallback_model,
            selection_strategy: self.strategy,
            round_robin_counter: AtomicUsize::new(0),
            usage_history: Arc::new(RwLock::new(VecDeque::new())),
            max_history_size: self.max_history_size,
            similarity_threshold: self.similarity_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distillation::registry::ModelMetadata;

    fn create_test_registry() -> Arc<RwLock<ModelRegistry>> {
        Arc::new(RwLock::new(ModelRegistry::new()))
    }

    fn create_test_metadata(id: &str, pattern: &str) -> ModelMetadata {
        ModelMetadata::new(id, QueryPattern::new(pattern), "base-model")
    }

    #[test]
    fn test_selection_strategy_default() {
        assert_eq!(
            SelectionStrategy::default(),
            SelectionStrategy::PatternMatch
        );
    }

    #[test]
    fn test_selection_strategy_descriptions() {
        assert!(!SelectionStrategy::PatternMatch.description().is_empty());
        assert!(!SelectionStrategy::LowestLatency.description().is_empty());
        assert!(!SelectionStrategy::HighestAccuracy.description().is_empty());
        assert!(!SelectionStrategy::RoundRobin.description().is_empty());
    }

    #[tokio::test]
    async fn test_model_selector_creation() {
        let registry = create_test_registry();
        let selector = ModelSelector::new(registry, "fallback");

        assert_eq!(selector.fallback_model(), "fallback");
        assert_eq!(selector.strategy(), SelectionStrategy::PatternMatch);
    }

    #[tokio::test]
    async fn test_model_selector_with_strategy() {
        let registry = create_test_registry();
        let selector = ModelSelector::new(registry, "fallback")
            .with_strategy(SelectionStrategy::LowestLatency);

        assert_eq!(selector.strategy(), SelectionStrategy::LowestLatency);
    }

    #[tokio::test]
    async fn test_model_selector_fallback() {
        let registry = create_test_registry();
        let selector = ModelSelector::new(registry, "fallback");

        // Empty registry should return fallback
        let selected = selector.select_model("test query").await;
        assert_eq!(selected, "fallback");
    }

    #[tokio::test]
    async fn test_model_selector_pattern_match() {
        let registry = create_test_registry();

        // Add a model to the registry
        {
            let mut reg = registry.write().await;
            let metadata = create_test_metadata("model-1", "test query");
            reg.register(metadata).unwrap();
        }

        let selector =
            ModelSelector::new(registry, "fallback").with_strategy(SelectionStrategy::PatternMatch);

        let selected = selector.select_model("test query").await;
        assert_eq!(selected, "model-1");
    }

    #[tokio::test]
    async fn test_model_selector_lowest_latency() {
        let registry = create_test_registry();

        {
            let mut reg = registry.write().await;

            let mut fast = create_test_metadata("fast", "q1");
            fast.metrics.record_usage(10.0, true);

            let mut slow = create_test_metadata("slow", "q2");
            slow.metrics.record_usage(100.0, true);

            reg.register(fast).unwrap();
            reg.register(slow).unwrap();
        }

        let selector = ModelSelector::new(registry, "fallback")
            .with_strategy(SelectionStrategy::LowestLatency);

        let selected = selector.select_model("any query").await;
        assert_eq!(selected, "fast");
    }

    #[tokio::test]
    async fn test_model_selector_highest_accuracy() {
        let registry = create_test_registry();

        {
            let mut reg = registry.write().await;

            let mut accurate = create_test_metadata("accurate", "q1");
            accurate.metrics.accuracy = 0.95;
            accurate.metrics.usage_count = 1;

            let mut inaccurate = create_test_metadata("inaccurate", "q2");
            inaccurate.metrics.accuracy = 0.5;
            inaccurate.metrics.usage_count = 1;

            reg.register(accurate).unwrap();
            reg.register(inaccurate).unwrap();
        }

        let selector = ModelSelector::new(registry, "fallback")
            .with_strategy(SelectionStrategy::HighestAccuracy);

        let selected = selector.select_model("any query").await;
        assert_eq!(selected, "accurate");
    }

    #[tokio::test]
    async fn test_model_selector_round_robin() {
        let registry = create_test_registry();

        {
            let mut reg = registry.write().await;
            reg.register(create_test_metadata("m1", "q1")).unwrap();
            reg.register(create_test_metadata("m2", "q2")).unwrap();
        }

        let selector =
            ModelSelector::new(registry, "fallback").with_strategy(SelectionStrategy::RoundRobin);

        let first = selector.select_model("q").await;
        let second = selector.select_model("q").await;

        // Should get different models (or wrap around)
        assert!(first == "m1" || first == "m2");
        assert!(second == "m1" || second == "m2");
    }

    #[tokio::test]
    async fn test_model_selector_report_result() {
        let registry = create_test_registry();

        {
            let mut reg = registry.write().await;
            reg.register(create_test_metadata("model-1", "test"))
                .unwrap();
        }

        let selector = ModelSelector::new(registry.clone(), "fallback");

        selector.report_result("model-1", 50.0, true).await;

        {
            let reg = registry.read().await;
            let model = reg.get("model-1").unwrap();
            assert!((model.metrics.avg_latency_ms - 50.0).abs() < f64::EPSILON);
            assert_eq!(model.metrics.usage_count, 1);
        }
    }

    #[tokio::test]
    async fn test_model_selector_statistics() {
        let registry = create_test_registry();

        {
            let mut reg = registry.write().await;
            reg.register(create_test_metadata("model-1", "test"))
                .unwrap();
        }

        let selector = ModelSelector::new(registry, "fallback")
            .with_strategy(SelectionStrategy::LowestLatency);

        let stats = selector.statistics().await;
        assert_eq!(stats.strategy, SelectionStrategy::LowestLatency);
        assert_eq!(stats.total_models, 1);
        assert_eq!(stats.active_models, 1);
    }

    #[tokio::test]
    async fn test_model_selector_clear_history() {
        let registry = create_test_registry();
        let selector = ModelSelector::new(registry, "fallback");

        selector.report_result("model-1", 50.0, true).await;

        let stats = selector.statistics().await;
        assert_eq!(stats.history_size, 1);

        selector.clear_history().await;

        let stats = selector.statistics().await;
        assert_eq!(stats.history_size, 0);
    }

    #[tokio::test]
    async fn test_model_selector_reset_round_robin() {
        let registry = create_test_registry();

        {
            let mut reg = registry.write().await;
            reg.register(create_test_metadata("m1", "q1")).unwrap();
            reg.register(create_test_metadata("m2", "q2")).unwrap();
        }

        let selector =
            ModelSelector::new(registry, "fallback").with_strategy(SelectionStrategy::RoundRobin);

        selector.select_model("q").await;
        selector.select_model("q").await;

        selector.reset_round_robin();

        let stats = selector.statistics().await;
        assert_eq!(stats.round_robin_position, 0);
    }

    #[tokio::test]
    async fn test_model_selector_recent_models() {
        let registry = create_test_registry();
        let selector = ModelSelector::new(registry, "fallback").with_max_history_size(10);

        selector.report_result("model-1", 50.0, true).await;
        selector.report_result("model-2", 50.0, true).await;
        selector.report_result("model-3", 50.0, true).await;

        let recent = selector.recent_models(2).await;
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0], "model-3");
        assert_eq!(recent[1], "model-2");
    }

    #[tokio::test]
    async fn test_model_selector_builder() {
        let registry = create_test_registry();

        let selector = ModelSelectorBuilder::new(registry, "fallback")
            .strategy(SelectionStrategy::HighestAccuracy)
            .similarity_threshold(0.9)
            .max_history_size(50)
            .build();

        assert_eq!(selector.strategy(), SelectionStrategy::HighestAccuracy);
        assert!((selector.similarity_threshold - 0.9).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_model_selector_set_strategy() {
        let registry = create_test_registry();
        let mut selector = ModelSelector::new(registry, "fallback");

        selector.set_strategy(SelectionStrategy::LowestLatency);
        assert_eq!(selector.strategy(), SelectionStrategy::LowestLatency);
    }

    #[tokio::test]
    async fn test_model_selector_set_fallback() {
        let registry = create_test_registry();
        let mut selector = ModelSelector::new(registry, "old-fallback");

        selector.set_fallback_model("new-fallback");
        assert_eq!(selector.fallback_model(), "new-fallback");
    }

    #[tokio::test]
    async fn test_model_selector_mfu() {
        let registry = create_test_registry();

        {
            let mut reg = registry.write().await;

            let mut popular = create_test_metadata("popular", "q1");
            popular.metrics.usage_count = 100;

            let mut unpopular = create_test_metadata("unpopular", "q2");
            unpopular.metrics.usage_count = 10;

            reg.register(popular).unwrap();
            reg.register(unpopular).unwrap();
        }

        let selector = ModelSelector::new(registry, "fallback")
            .with_strategy(SelectionStrategy::MostFrequentlyUsed);

        let selected = selector.select_model("any query").await;
        assert_eq!(selected, "popular");
    }

    #[test]
    fn test_model_selector_debug() {
        let registry = create_test_registry();
        let selector = ModelSelector::new(registry, "fallback");

        let debug_str = format!("{selector:?}");
        assert!(debug_str.contains("ModelSelector"));
        assert!(debug_str.contains("fallback"));
    }

    #[tokio::test]
    async fn test_model_selector_history_limit() {
        let registry = create_test_registry();
        let selector = ModelSelector::new(registry, "fallback").with_max_history_size(3);

        for i in 0..10 {
            selector
                .report_result(&format!("model-{i}"), 50.0, true)
                .await;
        }

        let stats = selector.statistics().await;
        assert_eq!(stats.history_size, 3);
    }
}
