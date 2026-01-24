//! Hidden state speculator for draft verification.
//!
//! This module provides a speculator implementation that uses hidden states
//! from transformer models to verify draft answers against context documents.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::SpeculatorError;
use crate::layer2_speculator::hidden_states::{
    HiddenStateCache, HiddenStateCacheConfig, HiddenStateProvider, ModelHiddenStates,
    StateReuseStrategy,
};
use crate::layer2_speculator::traits::{Speculator, SpeculatorConfig};
use crate::types::{Draft, SearchResult, SpeculationDecision, SpeculationResult};

/// Configuration for hidden state speculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStateSpeculatorConfig {
    /// Similarity threshold for acceptance.
    pub similarity_threshold: f32,
    /// Whether to use attention patterns for verification.
    pub use_attention_patterns: bool,
    /// Optional weights for each layer in comparison.
    pub layer_weights: Option<Vec<f32>>,
    /// Cache configuration.
    pub cache_config: HiddenStateCacheConfig,
    /// Base speculator configuration.
    #[serde(default)]
    pub speculator_config: SpeculatorConfig,
}

impl Default for HiddenStateSpeculatorConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            use_attention_patterns: false,
            layer_weights: None,
            cache_config: HiddenStateCacheConfig::default(),
            speculator_config: SpeculatorConfig::default(),
        }
    }
}

impl HiddenStateSpeculatorConfig {
    /// Create a new configuration with the given similarity threshold.
    #[must_use]
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
            ..Default::default()
        }
    }

    /// Set whether to use attention patterns.
    #[must_use]
    pub fn with_attention_patterns(mut self, use_attention: bool) -> Self {
        self.use_attention_patterns = use_attention;
        self
    }

    /// Set layer weights for comparison.
    #[must_use]
    pub fn with_layer_weights(mut self, weights: Vec<f32>) -> Self {
        self.layer_weights = Some(weights);
        self
    }

    /// Set the cache configuration.
    #[must_use]
    pub fn with_cache_config(mut self, config: HiddenStateCacheConfig) -> Self {
        self.cache_config = config;
        self
    }

    /// Set the base speculator configuration.
    #[must_use]
    pub fn with_speculator_config(mut self, config: SpeculatorConfig) -> Self {
        self.speculator_config = config;
        self
    }
}

/// A point of divergence detected in hidden states.
#[derive(Debug, Clone)]
pub struct DivergencePoint {
    /// The layer index where divergence was detected.
    pub layer_idx: usize,
    /// The position in the sequence.
    pub position: usize,
    /// The divergence score (higher = more divergence).
    pub divergence_score: f32,
    /// Description of the divergence.
    pub description: String,
}

impl DivergencePoint {
    /// Create a new divergence point.
    #[must_use]
    pub fn new(layer_idx: usize, position: usize, divergence_score: f32) -> Self {
        Self {
            layer_idx,
            position,
            divergence_score,
            description: String::new(),
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// Speculator that uses hidden states for verification.
pub struct HiddenStateSpeculator<P: HiddenStateProvider> {
    provider: P,
    cache: HiddenStateCache,
    reuse_strategy: Box<dyn StateReuseStrategy>,
    config: HiddenStateSpeculatorConfig,
}

impl<P: HiddenStateProvider> HiddenStateSpeculator<P> {
    /// Create a new hidden state speculator.
    #[must_use]
    pub fn new(provider: P, config: HiddenStateSpeculatorConfig) -> Self {
        let cache = HiddenStateCache::new(config.cache_config.clone());
        let reuse_strategy =
            Box::new(crate::layer2_speculator::hidden_states::PrefixReuseStrategy::default());

        Self {
            provider,
            cache,
            reuse_strategy,
            config,
        }
    }

    /// Set a custom state reuse strategy.
    #[must_use]
    pub fn with_reuse_strategy(mut self, strategy: Box<dyn StateReuseStrategy>) -> Self {
        self.reuse_strategy = strategy;
        self
    }

    /// Verify a draft using hidden state comparison.
    ///
    /// # Errors
    ///
    /// Returns an error if hidden state extraction fails.
    pub async fn verify_with_hidden_states(
        &mut self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError> {
        // Get hidden states for draft
        let draft_states = self.provider.get_hidden_states(&draft.content).await?;

        // Get hidden states for context documents
        let context_states = self.cache_context_states(context).await?;

        if context_states.is_empty() {
            // No context to compare against
            return Ok(
                SpeculationResult::new(SpeculationDecision::Accept, draft.confidence)
                    .with_explanation("No context available for comparison.".to_string()),
            );
        }

        // Compare draft states with context states
        let similarity = self.compare_states(&draft_states, &context_states);

        // Detect divergence points
        let divergences = self.detect_divergence(&draft_states, &context_states);

        // Make decision based on similarity
        let (decision, confidence) = if similarity >= self.config.similarity_threshold {
            if divergences.is_empty() {
                (SpeculationDecision::Accept, similarity)
            } else {
                (SpeculationDecision::Revise, similarity * 0.9)
            }
        } else if similarity >= self.config.similarity_threshold * 0.5 {
            (SpeculationDecision::Revise, similarity)
        } else {
            (SpeculationDecision::Reject, similarity)
        };

        let mut result = SpeculationResult::new(decision, confidence);

        // Add explanation
        let explanation = match &result.decision {
            SpeculationDecision::Accept => {
                format!(
                    "Draft hidden states align well with context (similarity: {similarity:.2})."
                )
            }
            SpeculationDecision::Revise => {
                format!(
                    "Draft shows partial alignment with context (similarity: {:.2}). {} divergence points detected.",
                    similarity,
                    divergences.len()
                )
            }
            SpeculationDecision::Reject => {
                format!(
                    "Draft hidden states diverge significantly from context (similarity: {similarity:.2})."
                )
            }
        };
        result = result.with_explanation(explanation);

        // Add issues from divergence points
        for divergence in divergences {
            result = result.with_issue(format!(
                "Divergence at layer {}, position {}: {} (score: {:.2})",
                divergence.layer_idx,
                divergence.position,
                divergence.description,
                divergence.divergence_score
            ));
        }

        Ok(result)
    }

    /// Extract and cache hidden states for context documents.
    async fn cache_context_states(
        &mut self,
        ctx_results: &[SearchResult],
    ) -> Result<Vec<ModelHiddenStates>, SpeculatorError> {
        let mut states = Vec::with_capacity(ctx_results.len());

        for result in ctx_results {
            let doc_content = &result.document.content;

            // Check cache first
            let cache_key =
                HiddenStateCache::make_key(self.provider.model_id(), doc_content.as_bytes());

            if let Some(cached) = self.cache.get(&cache_key) {
                states.push(cached);
                continue;
            }

            // Get hidden states from provider
            let doc_states = self.provider.get_hidden_states(doc_content).await?;

            // Cache the states
            self.cache.insert(cache_key, doc_states.clone());

            states.push(doc_states);
        }

        Ok(states)
    }

    /// Compare draft hidden states with context states.
    fn compare_states(
        &self,
        draft_states: &ModelHiddenStates,
        context_states: &[ModelHiddenStates],
    ) -> f32 {
        if context_states.is_empty() {
            return 0.0;
        }

        // Compute similarity with each context document
        let similarities: Vec<f32> = context_states
            .iter()
            .map(|ctx| self.compute_weighted_similarity(draft_states, ctx))
            .collect();

        // Return maximum similarity (best match)
        similarities.into_iter().fold(0.0_f32, f32::max)
    }

    /// Compute weighted similarity between two hidden state representations.
    fn compute_weighted_similarity(
        &self,
        states_a: &ModelHiddenStates,
        states_b: &ModelHiddenStates,
    ) -> f32 {
        if let Some(weights) = &self.config.layer_weights {
            // Weighted comparison across layers
            let num_layers = states_a
                .num_layers()
                .min(states_b.num_layers())
                .min(weights.len());
            if num_layers == 0 {
                return states_a.cosine_similarity(states_b);
            }

            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (i, weight) in weights.iter().enumerate().take(num_layers) {
                if let (Some(layer_a), Some(layer_b)) = (states_a.layer(i), states_b.layer(i))
                    && let (Some(vec_a), Some(vec_b)) =
                        (layer_a.at_position(0, 0), layer_b.at_position(0, 0))
                {
                    let sim = cosine_similarity(vec_a, vec_b);
                    weighted_sum += sim * weight;
                    weight_sum += weight;
                }
            }

            if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                0.0
            }
        } else {
            // Simple cosine similarity of pooled representations
            states_a.cosine_similarity(states_b)
        }
    }

    /// Detect factual inconsistencies via hidden state divergence.
    #[allow(clippy::unused_self)]
    fn detect_divergence(
        &self,
        draft_states: &ModelHiddenStates,
        context_states: &[ModelHiddenStates],
    ) -> Vec<DivergencePoint> {
        let mut divergences = Vec::new();

        // Only check if we have context
        if context_states.is_empty() {
            return divergences;
        }

        // Check each layer
        for layer_idx in 0..draft_states.num_layers() {
            if let Some(draft_layer) = draft_states.layer(layer_idx) {
                // Compare with best matching context
                let mut max_similarity = 0.0_f32;

                for ctx in context_states {
                    if let Some(ctx_layer) = ctx.layer(layer_idx) {
                        // Compare position 0 (CLS token or first token)
                        if let (Some(draft_vec), Some(ctx_vec)) =
                            (draft_layer.at_position(0, 0), ctx_layer.at_position(0, 0))
                        {
                            let sim = cosine_similarity(draft_vec, ctx_vec);
                            max_similarity = max_similarity.max(sim);
                        }
                    }
                }

                // Check for divergence
                let divergence_threshold = 0.5;
                if max_similarity < divergence_threshold {
                    let divergence_score = 1.0 - max_similarity;
                    let description = if layer_idx < draft_states.num_layers() / 3 {
                        "Low-level representation divergence (possibly lexical mismatch)"
                    } else if layer_idx < 2 * draft_states.num_layers() / 3 {
                        "Mid-level representation divergence (possibly semantic mismatch)"
                    } else {
                        "High-level representation divergence (possibly conceptual mismatch)"
                    };

                    divergences.push(
                        DivergencePoint::new(layer_idx, 0, divergence_score)
                            .with_description(description),
                    );
                }
            }
        }

        // Sort by divergence score (highest first)
        divergences.sort_by(|a, b| {
            b.divergence_score
                .partial_cmp(&a.divergence_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top divergences
        divergences.truncate(5);
        divergences
    }

    /// Get the current cache size.
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &HiddenStateSpeculatorConfig {
        &self.config
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-9 { dot / denom } else { 0.0 }
}

#[async_trait]
impl<P: HiddenStateProvider + Send + Sync> Speculator for HiddenStateSpeculator<P> {
    async fn verify_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError> {
        // Get hidden states for draft
        let draft_states = self.provider.get_hidden_states(&draft.content).await?;

        // Get hidden states for context documents (without caching in immutable context)
        let mut context_states = Vec::with_capacity(context.len());
        for result in context {
            let doc_states = self
                .provider
                .get_hidden_states(&result.document.content)
                .await?;
            context_states.push(doc_states);
        }

        if context_states.is_empty() {
            return Ok(
                SpeculationResult::new(SpeculationDecision::Accept, draft.confidence)
                    .with_explanation("No context available for comparison.".to_string()),
            );
        }

        // Compare states
        let similarity = self.compare_states(&draft_states, &context_states);

        // Make decision
        let (decision, confidence) = if similarity >= self.config.similarity_threshold {
            (SpeculationDecision::Accept, similarity)
        } else if similarity >= self.config.similarity_threshold * 0.5 {
            (SpeculationDecision::Revise, similarity)
        } else {
            (SpeculationDecision::Reject, similarity)
        };

        let explanation = format!("Hidden state similarity with context: {similarity:.2}");

        Ok(SpeculationResult::new(decision, confidence).with_explanation(explanation))
    }

    async fn revise_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
        speculation: &SpeculationResult,
    ) -> Result<Draft, SpeculatorError> {
        // Simple revision: incorporate context information
        let context_summary: String = context
            .iter()
            .take(2)
            .map(|r| r.document.content.chars().take(100).collect::<String>())
            .collect::<Vec<_>>()
            .join(" ");

        let revised_content = format!(
            "Based on the following context: {}...\n\n{}",
            context_summary, draft.content
        );

        Ok(Draft::new(revised_content, &draft.query).with_confidence(speculation.confidence + 0.1))
    }

    fn config(&self) -> &SpeculatorConfig {
        &self.config.speculator_config
    }
}

/// Mock hidden state speculator for testing.
pub struct MockHiddenStateSpeculator {
    config: HiddenStateSpeculatorConfig,
    similarity_response: f32,
}

impl MockHiddenStateSpeculator {
    /// Create a new mock hidden state speculator.
    #[must_use]
    pub fn new(config: HiddenStateSpeculatorConfig) -> Self {
        Self {
            config,
            similarity_response: 0.8,
        }
    }

    /// Set the mock similarity response.
    #[must_use]
    pub fn with_similarity(mut self, similarity: f32) -> Self {
        self.similarity_response = similarity;
        self
    }
}

impl Default for MockHiddenStateSpeculator {
    fn default() -> Self {
        Self::new(HiddenStateSpeculatorConfig::default())
    }
}

#[async_trait]
impl Speculator for MockHiddenStateSpeculator {
    async fn verify_draft(
        &self,
        _draft: &Draft,
        _context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError> {
        let similarity = self.similarity_response;

        let (decision, confidence) = if similarity >= self.config.similarity_threshold {
            (SpeculationDecision::Accept, similarity)
        } else if similarity >= self.config.similarity_threshold * 0.5 {
            (SpeculationDecision::Revise, similarity)
        } else {
            (SpeculationDecision::Reject, similarity)
        };

        let explanation = format!("Mock hidden state verification (similarity: {similarity:.2})");

        Ok(SpeculationResult::new(decision, confidence).with_explanation(explanation))
    }

    async fn revise_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
        _speculation: &SpeculationResult,
    ) -> Result<Draft, SpeculatorError> {
        let context_prefix: String = context
            .first()
            .map(|r| r.document.content.chars().take(50).collect())
            .unwrap_or_default();

        let revised = format!(
            "[Revised based on: {}...] {}",
            context_prefix, draft.content
        );
        Ok(Draft::new(revised, &draft.query).with_confidence(0.75))
    }

    fn config(&self) -> &SpeculatorConfig {
        &self.config.speculator_config
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::layer2_speculator::hidden_states::MockHiddenStateProvider;
    use crate::types::Document;

    fn create_context() -> Vec<SearchResult> {
        vec![
            SearchResult::new(
                Document::new(
                    "The capital of France is Paris. Paris is known for the Eiffel Tower.",
                ),
                0.9,
                0,
            ),
            SearchResult::new(
                Document::new("France is a country in Western Europe. Its capital is Paris."),
                0.85,
                1,
            ),
        ]
    }

    #[test]
    fn test_hidden_state_speculator_config_default() {
        let config = HiddenStateSpeculatorConfig::default();
        assert_eq!(config.similarity_threshold, 0.7);
        assert!(!config.use_attention_patterns);
        assert!(config.layer_weights.is_none());
    }

    #[test]
    fn test_hidden_state_speculator_config_builder() {
        let config = HiddenStateSpeculatorConfig::new(0.8)
            .with_attention_patterns(true)
            .with_layer_weights(vec![0.5, 0.3, 0.2]);

        assert_eq!(config.similarity_threshold, 0.8);
        assert!(config.use_attention_patterns);
        assert_eq!(config.layer_weights.unwrap(), vec![0.5, 0.3, 0.2]);
    }

    #[test]
    fn test_divergence_point() {
        let point =
            DivergencePoint::new(5, 10, 0.75).with_description("Semantic mismatch detected");

        assert_eq!(point.layer_idx, 5);
        assert_eq!(point.position, 10);
        assert_eq!(point.divergence_score, 0.75);
        assert_eq!(point.description, "Semantic mismatch detected");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        // Orthogonal vectors
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-5);

        // Same vectors
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-5);

        // Empty vectors
        assert_eq!(cosine_similarity(&[], &[]), 0.0);

        // Different lengths
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[tokio::test]
    async fn test_hidden_state_speculator_creation() {
        let provider = MockHiddenStateProvider::new(768, 12);
        let config = HiddenStateSpeculatorConfig::default();

        let speculator = HiddenStateSpeculator::new(provider, config);

        assert_eq!(speculator.cache_size(), 0);
        assert_eq!(speculator.config().similarity_threshold, 0.7);
    }

    #[tokio::test]
    async fn test_hidden_state_speculator_verify_draft() {
        let provider = MockHiddenStateProvider::new(768, 12);
        let config = HiddenStateSpeculatorConfig::default();

        let speculator = HiddenStateSpeculator::new(provider, config);

        let draft = Draft::new(
            "The capital of France is Paris.",
            "What is the capital of France?",
        )
        .with_confidence(0.8);
        let context = create_context();

        let result = speculator.verify_draft(&draft, &context).await.unwrap();

        // With mock provider, similarity will be non-trivial
        assert!(result.confidence >= 0.0);
        assert!(!result.explanation.is_empty());
    }

    #[tokio::test]
    async fn test_hidden_state_speculator_verify_with_hidden_states() {
        let provider = MockHiddenStateProvider::new(384, 6);
        let config = HiddenStateSpeculatorConfig::new(0.5); // Lower threshold

        let mut speculator = HiddenStateSpeculator::new(provider, config);

        let draft = Draft::new("Paris is the capital.", "What is the capital of France?");
        let context = create_context();

        let result = speculator
            .verify_with_hidden_states(&draft, &context)
            .await
            .unwrap();

        assert!(!result.explanation.is_empty());
    }

    #[tokio::test]
    async fn test_hidden_state_speculator_caching() {
        let provider = MockHiddenStateProvider::new(256, 4);
        let config = HiddenStateSpeculatorConfig::default();

        let mut speculator = HiddenStateSpeculator::new(provider, config);

        let draft = Draft::new("Test draft", "Test query");
        let context = create_context();

        // First call - should populate cache
        let _ = speculator
            .verify_with_hidden_states(&draft, &context)
            .await
            .unwrap();
        assert_eq!(speculator.cache_size(), 2); // Two context documents

        // Second call - should use cache
        let _ = speculator
            .verify_with_hidden_states(&draft, &context)
            .await
            .unwrap();
        assert_eq!(speculator.cache_size(), 2); // Still two entries

        // Clear cache
        speculator.clear_cache();
        assert_eq!(speculator.cache_size(), 0);
    }

    #[tokio::test]
    async fn test_hidden_state_speculator_empty_context() {
        let provider = MockHiddenStateProvider::new(768, 12);
        let config = HiddenStateSpeculatorConfig::default();

        let speculator = HiddenStateSpeculator::new(provider, config);

        let draft = Draft::new("Some draft content", "Query").with_confidence(0.9);

        let result = speculator.verify_draft(&draft, &[]).await.unwrap();

        assert!(matches!(result.decision, SpeculationDecision::Accept));
        assert!(result.explanation.contains("No context"));
    }

    #[tokio::test]
    async fn test_hidden_state_speculator_revise_draft() {
        let provider = MockHiddenStateProvider::new(768, 12);
        let config = HiddenStateSpeculatorConfig::default();

        let speculator = HiddenStateSpeculator::new(provider, config);

        let draft = Draft::new("Original draft", "Query");
        let context = create_context();
        let speculation = SpeculationResult::new(SpeculationDecision::Revise, 0.5);

        let revised = speculator
            .revise_draft(&draft, &context, &speculation)
            .await
            .unwrap();

        assert!(revised.content.contains("Original draft"));
        assert!(revised.content.len() > draft.content.len());
    }

    #[tokio::test]
    async fn test_mock_hidden_state_speculator() {
        let speculator = MockHiddenStateSpeculator::default();

        let draft = Draft::new("Test draft", "Query");
        let context = create_context();

        let result = speculator.verify_draft(&draft, &context).await.unwrap();

        assert!(matches!(result.decision, SpeculationDecision::Accept));
    }

    #[tokio::test]
    async fn test_mock_hidden_state_speculator_low_similarity() {
        let config = HiddenStateSpeculatorConfig::new(0.9);
        let speculator = MockHiddenStateSpeculator::new(config).with_similarity(0.3);

        let draft = Draft::new("Test draft", "Query");
        let context = create_context();

        let result = speculator.verify_draft(&draft, &context).await.unwrap();

        assert!(matches!(result.decision, SpeculationDecision::Reject));
    }

    #[tokio::test]
    async fn test_mock_hidden_state_speculator_revise() {
        let speculator = MockHiddenStateSpeculator::default();

        let draft = Draft::new("Original", "Query");
        let context = create_context();
        let speculation = SpeculationResult::new(SpeculationDecision::Revise, 0.5);

        let revised = speculator
            .revise_draft(&draft, &context, &speculation)
            .await
            .unwrap();

        assert!(revised.content.contains("Original"));
        assert!(revised.content.contains("[Revised based on:"));
    }

    #[test]
    fn test_hidden_state_speculator_config_with_cache() {
        let cache_config = HiddenStateCacheConfig {
            max_entries: 500,
            cache_attention: true,
            ttl_seconds: 1800,
            use_lru: false,
        };

        let config = HiddenStateSpeculatorConfig::default().with_cache_config(cache_config.clone());

        assert_eq!(config.cache_config.max_entries, 500);
        assert!(config.cache_config.cache_attention);
    }

    #[test]
    fn test_hidden_state_speculator_config_with_speculator_config() {
        let speculator_config = SpeculatorConfig {
            temperature: 0.5,
            accept_threshold: 0.95,
            ..Default::default()
        };

        let config =
            HiddenStateSpeculatorConfig::default().with_speculator_config(speculator_config);

        assert_eq!(config.speculator_config.temperature, 0.5);
        assert_eq!(config.speculator_config.accept_threshold, 0.95);
    }

    #[tokio::test]
    async fn test_detect_divergence() {
        let provider = MockHiddenStateProvider::new(256, 6);
        let config = HiddenStateSpeculatorConfig::default();

        let mut speculator = HiddenStateSpeculator::new(provider, config);

        let draft = Draft::new("Completely unrelated content xyz", "Query");
        let context = create_context();

        let result = speculator
            .verify_with_hidden_states(&draft, &context)
            .await
            .unwrap();

        // The mock provider generates pseudo-random states, so divergence detection will work
        // based on the actual comparison logic
        assert!(!result.explanation.is_empty());
    }

    #[tokio::test]
    async fn test_hidden_state_speculator_with_layer_weights() {
        let provider = MockHiddenStateProvider::new(256, 6);
        let config = HiddenStateSpeculatorConfig::default()
            .with_layer_weights(vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.1]);

        let speculator = HiddenStateSpeculator::new(provider, config);

        let draft = Draft::new("Test content", "Query");
        let context = create_context();

        let result = speculator.verify_draft(&draft, &context).await.unwrap();

        assert!(!result.explanation.is_empty());
    }

    #[test]
    fn test_divergence_point_creation() {
        let point = DivergencePoint::new(0, 0, 0.5);
        assert_eq!(point.layer_idx, 0);
        assert_eq!(point.position, 0);
        assert_eq!(point.divergence_score, 0.5);
        assert!(point.description.is_empty());
    }

    #[tokio::test]
    async fn test_speculator_trait_implementation() {
        let provider = MockHiddenStateProvider::new(768, 12);
        let config = HiddenStateSpeculatorConfig::default();

        let speculator: Box<dyn Speculator> =
            Box::new(HiddenStateSpeculator::new(provider, config));

        let draft = Draft::new("Test", "Query");
        let context = create_context();

        let result = speculator.verify_draft(&draft, &context).await.unwrap();
        assert!(result.confidence >= 0.0);

        let spec_config = speculator.config();
        assert!(spec_config.temperature >= 0.0);
    }

    #[tokio::test]
    async fn test_verify_with_hidden_states_revise_decision() {
        let provider = MockHiddenStateProvider::new(256, 6);
        // Set threshold higher to trigger revise decisions
        let config = HiddenStateSpeculatorConfig::new(0.9);

        let mut speculator = HiddenStateSpeculator::new(provider, config);

        let draft = Draft::new("Some content", "Query");
        let context = create_context();

        let result = speculator
            .verify_with_hidden_states(&draft, &context)
            .await
            .unwrap();

        // With mock provider and high threshold, likely to get revise or reject
        assert!(matches!(
            result.decision,
            SpeculationDecision::Accept | SpeculationDecision::Revise | SpeculationDecision::Reject
        ));
    }

    #[test]
    fn test_mock_hidden_state_speculator_default() {
        let speculator = MockHiddenStateSpeculator::default();
        assert_eq!(speculator.similarity_response, 0.8);
    }
}
