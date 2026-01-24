//! Model registry for managing distilled models.
//!
//! This module provides a registry for tracking and managing specialized
//! models created through the distillation process.

use super::types::QueryPattern;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "distillation")]
use crate::error::{DistillationError, OxiRagError};

/// Metrics for a registered model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Accuracy score (0.0 - 1.0).
    pub accuracy: f32,
    /// Total number of times this model has been used.
    pub usage_count: u64,
    /// Unix timestamp of last usage.
    pub last_used: Option<u64>,
}

impl ModelMetrics {
    /// Create new model metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update metrics after a model invocation.
    #[allow(clippy::cast_precision_loss)]
    pub fn record_usage(&mut self, latency_ms: f64, success: bool) {
        // Update running average of latency
        let old_total = self.avg_latency_ms * self.usage_count as f64;
        self.usage_count += 1;
        self.avg_latency_ms = (old_total + latency_ms) / self.usage_count as f64;

        // Update accuracy using exponential moving average
        let success_val = if success { 1.0 } else { 0.0 };
        if self.usage_count == 1 {
            self.accuracy = success_val;
        } else {
            // EMA with alpha = 0.1
            self.accuracy = 0.9 * self.accuracy + 0.1 * success_val;
        }

        self.last_used = Some(super::types::current_timestamp());
    }

    /// Check if the model has been used recently.
    #[must_use]
    pub fn is_recently_used(&self, max_age_secs: u64) -> bool {
        self.last_used.is_some_and(|last| {
            let now = super::types::current_timestamp();
            now.saturating_sub(last) <= max_age_secs
        })
    }

    /// Get the success rate as a percentage.
    #[must_use]
    pub fn success_rate_percent(&self) -> f32 {
        self.accuracy * 100.0
    }
}

/// Metadata for a registered model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique identifier for this model.
    pub model_id: String,
    /// The query pattern this model was trained for.
    pub pattern: QueryPattern,
    /// The base model this was fine-tuned from.
    pub base_model: String,
    /// Path to the `LoRA` adapter weights (if applicable).
    pub adapter_path: Option<String>,
    /// Unix timestamp when this model was created.
    pub created_at: u64,
    /// Performance metrics for this model.
    pub metrics: ModelMetrics,
    /// Whether this model is currently active/enabled.
    pub is_active: bool,
}

impl ModelMetadata {
    /// Create new model metadata.
    #[must_use]
    pub fn new(
        model_id: impl Into<String>,
        pattern: QueryPattern,
        base_model: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            pattern,
            base_model: base_model.into(),
            adapter_path: None,
            created_at: super::types::current_timestamp(),
            metrics: ModelMetrics::new(),
            is_active: true,
        }
    }

    /// Create model metadata with an adapter path.
    #[must_use]
    pub fn with_adapter(mut self, path: impl Into<String>) -> Self {
        self.adapter_path = Some(path.into());
        self
    }

    /// Set the active status.
    #[must_use]
    pub fn with_active(mut self, active: bool) -> Self {
        self.is_active = active;
        self
    }

    /// Check if this model has a `LoRA` adapter.
    #[must_use]
    pub fn has_adapter(&self) -> bool {
        self.adapter_path.is_some()
    }

    /// Get the age of this model in seconds.
    #[must_use]
    pub fn age_secs(&self) -> u64 {
        super::types::current_timestamp().saturating_sub(self.created_at)
    }

    /// Record a usage of this model.
    pub fn record_usage(&mut self, latency_ms: f64, success: bool) {
        self.metrics.record_usage(latency_ms, success);
    }

    /// Check if this model matches a given pattern.
    #[must_use]
    pub fn matches_pattern(&self, pattern: &QueryPattern, threshold: f32) -> bool {
        self.pattern.pattern_hash == pattern.pattern_hash
            || self.pattern.is_similar_to(pattern, threshold)
    }
}

/// Registry for managing distilled models.
///
/// Provides functionality for registering, querying, and managing
/// specialized models created through the distillation process.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    /// Map from model ID to metadata.
    models: HashMap<String, ModelMetadata>,
    /// Map from pattern hash to model ID for quick lookup.
    pattern_to_model: HashMap<u64, String>,
    /// Currently active model ID (if any).
    active_model: Option<String>,
}

impl ModelRegistry {
    /// Create a new empty model registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new model.
    ///
    /// # Errors
    ///
    /// Returns an error if a model with the same ID already exists.
    pub fn register(&mut self, metadata: ModelMetadata) -> Result<(), OxiRagError> {
        if self.models.contains_key(&metadata.model_id) {
            return Err(DistillationError::InvalidConfig(format!(
                "Model already registered: {}",
                metadata.model_id
            ))
            .into());
        }

        let pattern_hash = metadata.pattern.pattern_hash;
        let model_id = metadata.model_id.clone();

        self.models.insert(model_id.clone(), metadata);
        self.pattern_to_model.insert(pattern_hash, model_id);

        Ok(())
    }

    /// Unregister a model by ID.
    ///
    /// Returns the removed metadata, or None if not found.
    pub fn unregister(&mut self, model_id: &str) -> Option<ModelMetadata> {
        let metadata = self.models.remove(model_id)?;

        // Remove from pattern mapping
        self.pattern_to_model.retain(|_, id| id != model_id);

        // Clear active model if it was this one
        if self.active_model.as_deref() == Some(model_id) {
            self.active_model = None;
        }

        Some(metadata)
    }

    /// Get model metadata by ID.
    #[must_use]
    pub fn get(&self, model_id: &str) -> Option<&ModelMetadata> {
        self.models.get(model_id)
    }

    /// Get mutable model metadata by ID.
    pub fn get_mut(&mut self, model_id: &str) -> Option<&mut ModelMetadata> {
        self.models.get_mut(model_id)
    }

    /// Find a model by query pattern.
    #[must_use]
    pub fn find_by_pattern(&self, pattern: &QueryPattern) -> Option<&ModelMetadata> {
        // First try exact hash match
        if let Some(model_id) = self.pattern_to_model.get(&pattern.pattern_hash) {
            return self.models.get(model_id);
        }

        // Fall back to similarity search
        self.models
            .values()
            .find(|m| m.is_active && m.pattern.is_similar_to(pattern, 0.8))
    }

    /// Find a model by query pattern with a custom similarity threshold.
    #[must_use]
    pub fn find_by_pattern_with_threshold(
        &self,
        pattern: &QueryPattern,
        threshold: f32,
    ) -> Option<&ModelMetadata> {
        // First try exact hash match
        if let Some(model_id) = self.pattern_to_model.get(&pattern.pattern_hash) {
            return self.models.get(model_id);
        }

        // Fall back to similarity search
        self.models
            .values()
            .find(|m| m.is_active && m.pattern.is_similar_to(pattern, threshold))
    }

    /// Set the active model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model doesn't exist.
    pub fn set_active(&mut self, model_id: &str) -> Result<(), OxiRagError> {
        if !self.models.contains_key(model_id) {
            return Err(
                DistillationError::PatternNotFound(format!("Model not found: {model_id}")).into(),
            );
        }

        self.active_model = Some(model_id.to_string());
        Ok(())
    }

    /// Clear the active model.
    pub fn clear_active(&mut self) {
        self.active_model = None;
    }

    /// Get the currently active model.
    #[must_use]
    pub fn get_active(&self) -> Option<&ModelMetadata> {
        self.active_model
            .as_ref()
            .and_then(|id| self.models.get(id))
    }

    /// List all registered models.
    #[must_use]
    pub fn list_all(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    /// List only active models.
    #[must_use]
    pub fn list_active(&self) -> Vec<&ModelMetadata> {
        self.models.values().filter(|m| m.is_active).collect()
    }

    /// Update metrics for a model.
    pub fn update_metrics(&mut self, model_id: &str, latency_ms: f64, success: bool) {
        if let Some(model) = self.models.get_mut(model_id) {
            model.record_usage(latency_ms, success);
        }
    }

    /// Get the total number of registered models.
    #[must_use]
    pub fn count(&self) -> usize {
        self.models.len()
    }

    /// Get the number of active models.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.models.values().filter(|m| m.is_active).count()
    }

    /// Check if a model is registered.
    #[must_use]
    pub fn contains(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    /// Get the best model based on a scoring function.
    #[must_use]
    pub fn find_best<F>(&self, score_fn: F) -> Option<&ModelMetadata>
    where
        F: Fn(&ModelMetadata) -> f64,
    {
        self.models.values().filter(|m| m.is_active).max_by(|a, b| {
            score_fn(a)
                .partial_cmp(&score_fn(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get the fastest model (lowest average latency).
    #[must_use]
    pub fn find_fastest(&self) -> Option<&ModelMetadata> {
        self.models
            .values()
            .filter(|m| m.is_active && m.metrics.usage_count > 0)
            .min_by(|a, b| {
                a.metrics
                    .avg_latency_ms
                    .partial_cmp(&b.metrics.avg_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get the most accurate model.
    #[must_use]
    pub fn find_most_accurate(&self) -> Option<&ModelMetadata> {
        self.models
            .values()
            .filter(|m| m.is_active && m.metrics.usage_count > 0)
            .max_by(|a, b| {
                a.metrics
                    .accuracy
                    .partial_cmp(&b.metrics.accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get all models matching a base model.
    #[must_use]
    pub fn find_by_base_model(&self, base_model: &str) -> Vec<&ModelMetadata> {
        self.models
            .values()
            .filter(|m| m.base_model == base_model)
            .collect()
    }

    /// Deactivate models that haven't been used recently.
    pub fn deactivate_stale(&mut self, max_age_secs: u64) {
        for model in self.models.values_mut() {
            if !model.metrics.is_recently_used(max_age_secs) {
                model.is_active = false;
            }
        }
    }

    /// Remove all inactive models.
    pub fn prune_inactive(&mut self) {
        let inactive_ids: Vec<String> = self
            .models
            .iter()
            .filter(|(_, m)| !m.is_active)
            .map(|(id, _)| id.clone())
            .collect();

        for id in inactive_ids {
            self.unregister(&id);
        }
    }

    /// Clear all models from the registry.
    pub fn clear(&mut self) {
        self.models.clear();
        self.pattern_to_model.clear();
        self.active_model = None;
    }

    /// Get statistics about the registry.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn statistics(&self) -> RegistryStatistics {
        let total_usage: u64 = self.models.values().map(|m| m.metrics.usage_count).sum();
        let active_models = self.active_count();

        let avg_latency = if active_models > 0 {
            self.models
                .values()
                .filter(|m| m.is_active && m.metrics.usage_count > 0)
                .map(|m| m.metrics.avg_latency_ms)
                .sum::<f64>()
                / active_models as f64
        } else {
            0.0
        };

        let avg_accuracy = if active_models > 0 {
            self.models
                .values()
                .filter(|m| m.is_active && m.metrics.usage_count > 0)
                .map(|m| m.metrics.accuracy)
                .sum::<f32>()
                / active_models as f32
        } else {
            0.0
        };

        RegistryStatistics {
            total_models: self.models.len(),
            active_models,
            total_usage,
            avg_latency_ms: avg_latency,
            avg_accuracy,
        }
    }
}

/// Statistics about the model registry.
#[derive(Debug, Clone, Default)]
pub struct RegistryStatistics {
    /// Total number of registered models.
    pub total_models: usize,
    /// Number of active models.
    pub active_models: usize,
    /// Total usage count across all models.
    pub total_usage: u64,
    /// Average latency across active models.
    pub avg_latency_ms: f64,
    /// Average accuracy across active models.
    pub avg_accuracy: f32,
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    fn create_test_metadata(id: &str, pattern: &str) -> ModelMetadata {
        ModelMetadata::new(id, QueryPattern::new(pattern), "test-base-model")
    }

    #[test]
    fn test_model_metrics_default() {
        let metrics = ModelMetrics::new();
        assert!((metrics.avg_latency_ms - 0.0).abs() < f64::EPSILON);
        assert!((metrics.accuracy - 0.0).abs() < f32::EPSILON);
        assert_eq!(metrics.usage_count, 0);
    }

    #[test]
    fn test_model_metrics_record_usage() {
        let mut metrics = ModelMetrics::new();

        metrics.record_usage(100.0, true);
        assert_eq!(metrics.usage_count, 1);
        assert!((metrics.avg_latency_ms - 100.0).abs() < f64::EPSILON);
        assert!((metrics.accuracy - 1.0).abs() < f32::EPSILON);

        metrics.record_usage(200.0, false);
        assert_eq!(metrics.usage_count, 2);
        assert!((metrics.avg_latency_ms - 150.0).abs() < f64::EPSILON);
        // EMA: 0.9 * 1.0 + 0.1 * 0.0 = 0.9
        assert!((metrics.accuracy - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_model_metadata_creation() {
        let pattern = QueryPattern::new("test query");
        let metadata = ModelMetadata::new("model-1", pattern, "gpt-4");

        assert_eq!(metadata.model_id, "model-1");
        assert_eq!(metadata.base_model, "gpt-4");
        assert!(metadata.is_active);
        assert!(!metadata.has_adapter());
    }

    #[test]
    fn test_model_metadata_with_adapter() {
        let pattern = QueryPattern::new("test");
        let metadata =
            ModelMetadata::new("model-1", pattern, "base").with_adapter("/path/to/adapter");

        assert!(metadata.has_adapter());
        assert_eq!(metadata.adapter_path, Some("/path/to/adapter".to_string()));
    }

    #[test]
    fn test_model_registry_register() {
        let mut registry = ModelRegistry::new();
        let metadata = create_test_metadata("model-1", "test query");

        let result = registry.register(metadata);
        assert!(result.is_ok());
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_model_registry_duplicate_registration() {
        let mut registry = ModelRegistry::new();
        let metadata1 = create_test_metadata("model-1", "test query");
        let metadata2 = create_test_metadata("model-1", "another query");

        registry.register(metadata1).unwrap();
        let result = registry.register(metadata2);

        assert!(result.is_err());
    }

    #[test]
    fn test_model_registry_unregister() {
        let mut registry = ModelRegistry::new();
        let metadata = create_test_metadata("model-1", "test query");

        registry.register(metadata).unwrap();
        let removed = registry.unregister("model-1");

        assert!(removed.is_some());
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_model_registry_get() {
        let mut registry = ModelRegistry::new();
        let metadata = create_test_metadata("model-1", "test query");

        registry.register(metadata).unwrap();

        assert!(registry.get("model-1").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_model_registry_find_by_pattern() {
        let mut registry = ModelRegistry::new();
        let pattern = QueryPattern::new("test query");
        let metadata = ModelMetadata::new("model-1", pattern.clone(), "base");

        registry.register(metadata).unwrap();

        let found = registry.find_by_pattern(&pattern);
        assert!(found.is_some());
        assert_eq!(found.unwrap().model_id, "model-1");
    }

    #[test]
    fn test_model_registry_set_active() {
        let mut registry = ModelRegistry::new();
        let metadata = create_test_metadata("model-1", "test");

        registry.register(metadata).unwrap();
        registry.set_active("model-1").unwrap();

        assert!(registry.get_active().is_some());
        assert_eq!(registry.get_active().unwrap().model_id, "model-1");
    }

    #[test]
    fn test_model_registry_set_active_nonexistent() {
        let mut registry = ModelRegistry::new();
        let result = registry.set_active("nonexistent");

        assert!(result.is_err());
    }

    #[test]
    fn test_model_registry_list_all() {
        let mut registry = ModelRegistry::new();

        registry
            .register(create_test_metadata("model-1", "query1"))
            .unwrap();
        registry
            .register(create_test_metadata("model-2", "query2"))
            .unwrap();

        assert_eq!(registry.list_all().len(), 2);
    }

    #[test]
    fn test_model_registry_list_active() {
        let mut registry = ModelRegistry::new();

        let mut inactive = create_test_metadata("model-1", "query1");
        inactive.is_active = false;

        registry.register(inactive).unwrap();
        registry
            .register(create_test_metadata("model-2", "query2"))
            .unwrap();

        assert_eq!(registry.list_active().len(), 1);
    }

    #[test]
    fn test_model_registry_update_metrics() {
        let mut registry = ModelRegistry::new();
        let metadata = create_test_metadata("model-1", "test");

        registry.register(metadata).unwrap();
        registry.update_metrics("model-1", 50.0, true);

        let model = registry.get("model-1").unwrap();
        assert!((model.metrics.avg_latency_ms - 50.0).abs() < f64::EPSILON);
        assert_eq!(model.metrics.usage_count, 1);
    }

    #[test]
    fn test_model_registry_find_fastest() {
        let mut registry = ModelRegistry::new();

        let mut fast = create_test_metadata("fast", "query1");
        fast.metrics.record_usage(10.0, true);

        let mut slow = create_test_metadata("slow", "query2");
        slow.metrics.record_usage(100.0, true);

        registry.register(fast).unwrap();
        registry.register(slow).unwrap();

        let fastest = registry.find_fastest();
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().model_id, "fast");
    }

    #[test]
    fn test_model_registry_find_most_accurate() {
        let mut registry = ModelRegistry::new();

        let mut accurate = create_test_metadata("accurate", "query1");
        accurate.metrics.accuracy = 0.95;
        accurate.metrics.usage_count = 1;

        let mut inaccurate = create_test_metadata("inaccurate", "query2");
        inaccurate.metrics.accuracy = 0.5;
        inaccurate.metrics.usage_count = 1;

        registry.register(accurate).unwrap();
        registry.register(inaccurate).unwrap();

        let best = registry.find_most_accurate();
        assert!(best.is_some());
        assert_eq!(best.unwrap().model_id, "accurate");
    }

    #[test]
    fn test_model_registry_find_by_base_model() {
        let mut registry = ModelRegistry::new();

        let model1 = ModelMetadata::new("m1", QueryPattern::new("q1"), "gpt-4");
        let model2 = ModelMetadata::new("m2", QueryPattern::new("q2"), "gpt-4");
        let model3 = ModelMetadata::new("m3", QueryPattern::new("q3"), "llama-2");

        registry.register(model1).unwrap();
        registry.register(model2).unwrap();
        registry.register(model3).unwrap();

        let gpt4_models = registry.find_by_base_model("gpt-4");
        assert_eq!(gpt4_models.len(), 2);
    }

    #[test]
    fn test_model_registry_prune_inactive() {
        let mut registry = ModelRegistry::new();

        let mut inactive = create_test_metadata("inactive", "q1");
        inactive.is_active = false;

        registry.register(inactive).unwrap();
        registry
            .register(create_test_metadata("active", "q2"))
            .unwrap();

        assert_eq!(registry.count(), 2);

        registry.prune_inactive();

        assert_eq!(registry.count(), 1);
        assert!(registry.get("active").is_some());
    }

    #[test]
    fn test_model_registry_clear() {
        let mut registry = ModelRegistry::new();

        registry
            .register(create_test_metadata("model-1", "q1"))
            .unwrap();
        registry
            .register(create_test_metadata("model-2", "q2"))
            .unwrap();
        registry.set_active("model-1").unwrap();

        registry.clear();

        assert_eq!(registry.count(), 0);
        assert!(registry.get_active().is_none());
    }

    #[test]
    fn test_model_registry_statistics() {
        let mut registry = ModelRegistry::new();

        let mut model1 = create_test_metadata("m1", "q1");
        model1.metrics.record_usage(100.0, true);

        let mut model2 = create_test_metadata("m2", "q2");
        model2.metrics.record_usage(200.0, true);
        model2.metrics.record_usage(200.0, false);

        registry.register(model1).unwrap();
        registry.register(model2).unwrap();

        let stats = registry.statistics();
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.active_models, 2);
        assert_eq!(stats.total_usage, 3);
    }

    #[test]
    fn test_model_registry_contains() {
        let mut registry = ModelRegistry::new();
        registry
            .register(create_test_metadata("model-1", "test"))
            .unwrap();

        assert!(registry.contains("model-1"));
        assert!(!registry.contains("model-2"));
    }

    #[test]
    fn test_model_registry_active_count() {
        let mut registry = ModelRegistry::new();

        let mut inactive = create_test_metadata("inactive", "q1");
        inactive.is_active = false;

        registry.register(inactive).unwrap();
        registry
            .register(create_test_metadata("active", "q2"))
            .unwrap();

        assert_eq!(registry.count(), 2);
        assert_eq!(registry.active_count(), 1);
    }

    #[test]
    fn test_model_registry_find_best() {
        let mut registry = ModelRegistry::new();

        let mut model1 = create_test_metadata("m1", "q1");
        model1.metrics.usage_count = 100;

        let mut model2 = create_test_metadata("m2", "q2");
        model2.metrics.usage_count = 50;

        registry.register(model1).unwrap();
        registry.register(model2).unwrap();

        // Find model with highest usage
        let best = registry.find_best(|m| m.metrics.usage_count as f64);
        assert!(best.is_some());
        assert_eq!(best.unwrap().model_id, "m1");
    }

    #[test]
    fn test_model_metadata_matches_pattern() {
        let pattern = QueryPattern::new("test query");
        let metadata = ModelMetadata::new("model-1", pattern.clone(), "base");

        assert!(metadata.matches_pattern(&pattern, 0.8));

        let different = QueryPattern::new("completely different");
        assert!(!metadata.matches_pattern(&different, 0.9));
    }

    #[test]
    fn test_model_registry_clear_active() {
        let mut registry = ModelRegistry::new();
        registry
            .register(create_test_metadata("model-1", "test"))
            .unwrap();
        registry.set_active("model-1").unwrap();

        assert!(registry.get_active().is_some());

        registry.clear_active();
        assert!(registry.get_active().is_none());
    }
}
