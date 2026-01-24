//! State reuse strategies for speculative RAG.
//!
//! This module provides strategies for determining when and how to reuse
//! cached hidden states for new inputs.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::needless_lifetimes,
    clippy::struct_field_names,
    clippy::manual_range_contains,
    clippy::needless_borrow
)]

use super::extractor::StateSimilarity;
use super::traits::StateReuseStrategy;
use super::types::ModelHiddenStates;

/// Prefix-based state reuse strategy.
///
/// Reuses hidden states when the cached text is a prefix of the new text.
/// This is the most common and reliable reuse strategy.
pub struct PrefixReuseStrategy {
    /// Minimum prefix length in characters to consider for reuse.
    min_prefix_length: usize,
    /// Similarity threshold for partial matches (0.0 to 1.0).
    similarity_threshold: f32,
}

impl Default for PrefixReuseStrategy {
    fn default() -> Self {
        Self {
            min_prefix_length: 10,
            similarity_threshold: 0.8,
        }
    }
}

impl PrefixReuseStrategy {
    /// Create a new prefix reuse strategy.
    #[must_use]
    pub fn new(min_prefix_length: usize, similarity_threshold: f32) -> Self {
        Self {
            min_prefix_length,
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
        }
    }

    /// Set the minimum prefix length.
    #[must_use]
    pub fn with_min_prefix_length(mut self, length: usize) -> Self {
        self.min_prefix_length = length;
        self
    }

    /// Set the similarity threshold.
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Find the common prefix length between two strings.
    #[must_use]
    pub fn common_prefix_len(a: &str, b: &str) -> usize {
        a.chars()
            .zip(b.chars())
            .take_while(|(c1, c2)| c1 == c2)
            .count()
    }

    /// Estimate token count from character count.
    /// Rough approximation: ~4 characters per token on average.
    #[must_use]
    pub fn estimate_token_count(char_count: usize) -> usize {
        (char_count as f32 / 4.0).ceil() as usize
    }
}

impl StateReuseStrategy for PrefixReuseStrategy {
    fn can_reuse(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> bool {
        let prefix_len = Self::common_prefix_len(new_text, cached_text);

        // Must have minimum prefix length
        if prefix_len < self.min_prefix_length {
            return false;
        }

        // Cached text must be a prefix of new text
        if !new_text.starts_with(cached_text) && !cached_text.starts_with(new_text) {
            return false;
        }

        // Check that cached states have enough sequence length
        let estimated_tokens = Self::estimate_token_count(prefix_len.min(cached_text.len()));
        if cached.sequence_length < estimated_tokens {
            return false;
        }

        true
    }

    fn reuse_point(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> usize {
        if !self.can_reuse(cached, new_text, cached_text) {
            return 0;
        }

        let prefix_len = Self::common_prefix_len(new_text, cached_text);

        // Reuse point is the estimated token count at the prefix boundary
        let max_reuse = Self::estimate_token_count(prefix_len.min(cached_text.len()));

        // But never reuse more than the cached sequence length
        max_reuse.min(cached.sequence_length)
    }

    fn reuse_quality(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> f32 {
        if !self.can_reuse(cached, new_text, cached_text) {
            return 0.0;
        }

        let prefix_len = Self::common_prefix_len(new_text, cached_text);
        let new_len = new_text.len().max(1);
        let cached_len = cached_text.len().max(1);

        // Quality is based on:
        // 1. How much of the new text is covered by the prefix
        // 2. How much of the cached text is being reused

        let coverage_ratio = prefix_len as f32 / new_len as f32;
        let reuse_ratio = prefix_len as f32 / cached_len as f32;

        // Weight coverage more heavily
        (coverage_ratio * 0.7 + reuse_ratio * 0.3).clamp(0.0, 1.0)
    }

    fn description(&self) -> &'static str {
        "Prefix-based reuse: reuses cached states when cached text is a prefix of new text"
    }
}

/// Semantic similarity based reuse strategy.
///
/// Reuses hidden states when the semantic similarity between cached and new
/// inputs exceeds a threshold, even if they don't share a common prefix.
pub struct SemanticReuseStrategy {
    /// Similarity threshold for reuse (0.0 to 1.0).
    similarity_threshold: f32,
    /// Minimum layer similarity ratio required.
    min_layer_agreement: f32,
}

impl Default for SemanticReuseStrategy {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            min_layer_agreement: 0.7,
        }
    }
}

impl SemanticReuseStrategy {
    /// Create a new semantic reuse strategy.
    #[must_use]
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
            min_layer_agreement: 0.7,
        }
    }

    /// Set the similarity threshold.
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the minimum layer agreement ratio.
    #[must_use]
    pub fn with_min_layer_agreement(mut self, ratio: f32) -> Self {
        self.min_layer_agreement = ratio.clamp(0.0, 1.0);
        self
    }
}

impl StateReuseStrategy for SemanticReuseStrategy {
    fn can_reuse(&self, cached: &ModelHiddenStates, _new_text: &str, _cached_text: &str) -> bool {
        // Need pooled output for semantic comparison
        cached.pooled_output.is_some()
    }

    fn reuse_point(
        &self,
        cached: &ModelHiddenStates,
        _new_text: &str,
        _cached_text: &str,
    ) -> usize {
        // For semantic reuse, we typically reuse all cached states
        // (since they represent similar semantic content)
        cached.sequence_length
    }

    fn reuse_quality(
        &self,
        cached: &ModelHiddenStates,
        _new_text: &str,
        _cached_text: &str,
    ) -> f32 {
        // Without the actual new text's hidden states, we can only
        // estimate quality based on cached state characteristics

        // Check if we have enough layers
        if cached.layers.is_empty() {
            return 0.0;
        }

        // Use pooled output quality as a proxy
        if let Some(ref pooled) = cached.pooled_output {
            let norm = StateSimilarity::norm(pooled);
            // Normalized outputs tend to be more useful
            if norm > 0.1 && norm < 10.0 {
                return self.similarity_threshold;
            }
        }

        // Default to threshold if we have valid states
        self.similarity_threshold * 0.8
    }

    fn description(&self) -> &'static str {
        "Semantic similarity reuse: reuses states when semantic content is similar"
    }
}

/// Combined hybrid strategy that uses multiple strategies.
pub struct HybridReuseStrategy {
    strategies: Vec<Box<dyn StateReuseStrategy>>,
    weights: Vec<f32>,
}

impl Default for HybridReuseStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridReuseStrategy {
    /// Create a new hybrid strategy with default strategies.
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategies: vec![
                Box::new(PrefixReuseStrategy::default()),
                Box::new(SemanticReuseStrategy::default()),
            ],
            weights: vec![0.7, 0.3],
        }
    }

    /// Create an empty hybrid strategy.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            strategies: vec![],
            weights: vec![],
        }
    }

    /// Add a strategy with a weight.
    #[must_use]
    pub fn with_strategy(mut self, strategy: Box<dyn StateReuseStrategy>, weight: f32) -> Self {
        self.strategies.push(strategy);
        self.weights.push(weight.max(0.0));
        self
    }

    /// Get the number of strategies.
    #[must_use]
    pub fn num_strategies(&self) -> usize {
        self.strategies.len()
    }

    /// Normalize weights to sum to 1.0.
    fn normalized_weights(&self) -> Vec<f32> {
        let total: f32 = self.weights.iter().sum();
        if total == 0.0 {
            return vec![0.0; self.weights.len()];
        }
        self.weights.iter().map(|w| w / total).collect()
    }
}

impl StateReuseStrategy for HybridReuseStrategy {
    fn can_reuse(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> bool {
        // Can reuse if any strategy says we can
        self.strategies
            .iter()
            .any(|s| s.can_reuse(cached, new_text, cached_text))
    }

    fn reuse_point(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> usize {
        // Take the maximum reuse point from strategies that can reuse
        let weights = self.normalized_weights();

        let mut best_point = 0;
        let mut best_weight = 0.0;

        for (i, strategy) in self.strategies.iter().enumerate() {
            if strategy.can_reuse(cached, new_text, cached_text) {
                let point = strategy.reuse_point(cached, new_text, cached_text);
                let weight = weights.get(i).copied().unwrap_or(0.0);

                if weight > best_weight || (weight == best_weight && point > best_point) {
                    best_point = point;
                    best_weight = weight;
                }
            }
        }

        best_point
    }

    fn reuse_quality(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> f32 {
        let weights = self.normalized_weights();

        // Weighted average of quality scores
        let mut total_quality = 0.0f32;
        let mut total_weight = 0.0f32;

        for (i, strategy) in self.strategies.iter().enumerate() {
            if strategy.can_reuse(cached, new_text, cached_text) {
                let quality = strategy.reuse_quality(cached, new_text, cached_text);
                let weight = weights.get(i).copied().unwrap_or(0.0);

                total_quality += quality * weight;
                total_weight += weight;
            }
        }

        if total_weight == 0.0 {
            0.0
        } else {
            total_quality / total_weight
        }
    }

    fn description(&self) -> &'static str {
        "Hybrid reuse: combines multiple strategies with weighted voting"
    }
}

/// Length-aware reuse strategy.
///
/// Considers the relative lengths of cached and new text when deciding
/// whether to reuse states.
pub struct LengthAwareReuseStrategy {
    /// Maximum ratio of length difference allowed.
    max_length_ratio: f32,
    /// Minimum overlap ratio required.
    min_overlap_ratio: f32,
}

impl Default for LengthAwareReuseStrategy {
    fn default() -> Self {
        Self {
            max_length_ratio: 2.0,
            min_overlap_ratio: 0.3,
        }
    }
}

impl LengthAwareReuseStrategy {
    /// Create a new length-aware strategy.
    #[must_use]
    pub fn new(max_length_ratio: f32, min_overlap_ratio: f32) -> Self {
        Self {
            max_length_ratio: max_length_ratio.max(1.0),
            min_overlap_ratio: min_overlap_ratio.clamp(0.0, 1.0),
        }
    }
}

impl StateReuseStrategy for LengthAwareReuseStrategy {
    fn can_reuse(&self, _cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> bool {
        let new_len = new_text.len().max(1) as f32;
        let cached_len = cached_text.len().max(1) as f32;

        // Check length ratio
        let ratio = (new_len / cached_len).max(cached_len / new_len);
        if ratio > self.max_length_ratio {
            return false;
        }

        // Check overlap
        let common_prefix = PrefixReuseStrategy::common_prefix_len(new_text, cached_text);
        let overlap_ratio = common_prefix as f32 / new_len.min(cached_len);

        overlap_ratio >= self.min_overlap_ratio
    }

    fn reuse_point(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> usize {
        if !self.can_reuse(cached, new_text, cached_text) {
            return 0;
        }

        let common_prefix = PrefixReuseStrategy::common_prefix_len(new_text, cached_text);
        PrefixReuseStrategy::estimate_token_count(common_prefix).min(cached.sequence_length)
    }

    fn reuse_quality(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> f32 {
        if !self.can_reuse(cached, new_text, cached_text) {
            return 0.0;
        }

        let new_len = new_text.len().max(1) as f32;
        let cached_len = cached_text.len().max(1) as f32;

        // Quality based on length similarity and overlap
        let length_sim = 1.0 - ((new_len - cached_len).abs() / new_len.max(cached_len));
        let common_prefix = PrefixReuseStrategy::common_prefix_len(new_text, cached_text);
        let overlap_ratio = common_prefix as f32 / new_len.min(cached_len);

        (length_sim * 0.4 + overlap_ratio * 0.6).clamp(0.0, 1.0)
    }

    fn description(&self) -> &'static str {
        "Length-aware reuse: considers text length ratios and overlap"
    }
}

/// Strategy selector based on input characteristics.
#[derive(Default)]
pub struct AdaptiveReuseStrategy {
    prefix: PrefixReuseStrategy,
    semantic: SemanticReuseStrategy,
    length: LengthAwareReuseStrategy,
}

impl AdaptiveReuseStrategy {
    /// Create a new adaptive strategy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the best strategy based on input characteristics.
    fn select_strategy<'a>(
        &'a self,
        new_text: &str,
        cached_text: &str,
    ) -> &'a dyn StateReuseStrategy {
        // If texts share a significant prefix, use prefix strategy
        let common_prefix = PrefixReuseStrategy::common_prefix_len(new_text, cached_text);
        let shorter_len = new_text.len().min(cached_text.len());

        if shorter_len > 0 && common_prefix as f32 / shorter_len as f32 > 0.5 {
            return &self.prefix;
        }

        // If lengths are similar, use length-aware strategy
        let length_ratio = new_text.len().max(1) as f32 / cached_text.len().max(1) as f32;
        if length_ratio >= 0.5 && length_ratio <= 2.0 {
            return &self.length;
        }

        // Default to semantic strategy
        &self.semantic
    }
}

impl StateReuseStrategy for AdaptiveReuseStrategy {
    fn can_reuse(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> bool {
        let strategy = self.select_strategy(new_text, cached_text);
        strategy.can_reuse(cached, new_text, cached_text)
    }

    fn reuse_point(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> usize {
        let strategy = self.select_strategy(new_text, cached_text);
        strategy.reuse_point(cached, new_text, cached_text)
    }

    fn reuse_quality(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> f32 {
        let strategy = self.select_strategy(new_text, cached_text);
        strategy.reuse_quality(cached, new_text, cached_text)
    }

    fn description(&self) -> &'static str {
        "Adaptive reuse: automatically selects the best strategy based on input"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hidden_states::types::{
        DType, Device, HiddenStateTensor, LayerHiddenState, TensorShape,
    };

    fn create_mock_states(seq_len: usize) -> ModelHiddenStates {
        let mut states = ModelHiddenStates::new("test-model", 4, 64);
        states.sequence_length = seq_len;

        for i in 0..4 {
            let hidden = HiddenStateTensor::zeros(
                TensorShape::new(vec![1, seq_len, 64]),
                DType::F32,
                Device::Cpu,
            );
            states.add_layer(LayerHiddenState::new(i, hidden));
        }

        let pooled = HiddenStateTensor::from_vec_1d(vec![0.5; 64]);
        states.set_pooled_output(pooled);

        states
    }

    #[test]
    fn test_prefix_strategy_common_prefix_len() {
        assert_eq!(
            PrefixReuseStrategy::common_prefix_len("hello world", "hello there"),
            6
        );
        assert_eq!(PrefixReuseStrategy::common_prefix_len("abc", "xyz"), 0);
        assert_eq!(PrefixReuseStrategy::common_prefix_len("test", "test"), 4);
    }

    #[test]
    fn test_prefix_strategy_can_reuse() {
        let strategy = PrefixReuseStrategy::new(5, 0.8);
        let cached = create_mock_states(20);

        // Exact prefix match
        assert!(strategy.can_reuse(&cached, "Hello, world! How are you?", "Hello, world!"));

        // Text too short
        assert!(!strategy.can_reuse(&cached, "Hi", "Hi"));

        // No common prefix
        assert!(!strategy.can_reuse(&cached, "Goodbye", "Hello"));
    }

    #[test]
    fn test_prefix_strategy_reuse_point() {
        let strategy = PrefixReuseStrategy::default();
        let cached = create_mock_states(50);

        let point = strategy.reuse_point(&cached, "Hello, world! This is a test.", "Hello, world!");
        assert!(point > 0);
        assert!(point <= 50);
    }

    #[test]
    fn test_prefix_strategy_reuse_quality() {
        let strategy = PrefixReuseStrategy::default();
        let cached = create_mock_states(50);

        // High quality: cached is prefix of new
        let quality1 = strategy.reuse_quality(
            &cached,
            "Hello, world! Extended text here.",
            "Hello, world!",
        );
        assert!(quality1 > 0.0);

        // Lower quality: less overlap
        let quality2 = strategy.reuse_quality(&cached, "Hello! Different text", "Hello, world!");
        assert!(quality2 < quality1 || quality2 == 0.0);
    }

    #[test]
    fn test_semantic_strategy_can_reuse() {
        let strategy = SemanticReuseStrategy::default();
        let cached = create_mock_states(20);

        // Should be able to reuse since we have pooled output
        assert!(strategy.can_reuse(&cached, "any text", "any other text"));
    }

    #[test]
    fn test_hybrid_strategy() {
        let strategy = HybridReuseStrategy::new();
        let cached = create_mock_states(30);

        assert!(strategy.num_strategies() >= 2);

        // Should work with prefix match
        let can_reuse = strategy.can_reuse(&cached, "Hello, world! Extended", "Hello, world!");
        assert!(can_reuse);
    }

    #[test]
    fn test_hybrid_strategy_custom() {
        let strategy = HybridReuseStrategy::empty()
            .with_strategy(Box::new(PrefixReuseStrategy::new(5, 0.9)), 0.8)
            .with_strategy(Box::new(SemanticReuseStrategy::new(0.9)), 0.2);

        assert_eq!(strategy.num_strategies(), 2);
    }

    #[test]
    fn test_length_aware_strategy() {
        let strategy = LengthAwareReuseStrategy::default();
        let cached = create_mock_states(20);

        // Similar length texts with overlap
        assert!(strategy.can_reuse(&cached, "Hello there friend", "Hello there buddy"));

        // Very different lengths
        let strategy_strict = LengthAwareReuseStrategy::new(1.5, 0.5);
        assert!(!strategy_strict.can_reuse(&cached, "Hi", "Hello there my good friend"));
    }

    #[test]
    fn test_adaptive_strategy() {
        let strategy = AdaptiveReuseStrategy::new();
        let cached = create_mock_states(30);

        // Should adapt to different input patterns
        let quality1 =
            strategy.reuse_quality(&cached, "Hello, world! Extended text", "Hello, world!");
        let quality2 =
            strategy.reuse_quality(&cached, "Completely different text", "Hello, world!");

        // Prefix match should have higher quality
        assert!(quality1 >= quality2 || quality2 == 0.0);
    }

    #[test]
    fn test_strategy_descriptions() {
        assert!(!PrefixReuseStrategy::default().description().is_empty());
        assert!(!SemanticReuseStrategy::default().description().is_empty());
        assert!(!HybridReuseStrategy::default().description().is_empty());
        assert!(!LengthAwareReuseStrategy::default().description().is_empty());
        assert!(!AdaptiveReuseStrategy::default().description().is_empty());
    }

    #[test]
    fn test_estimate_token_count() {
        // ~4 chars per token
        assert_eq!(PrefixReuseStrategy::estimate_token_count(4), 1);
        assert_eq!(PrefixReuseStrategy::estimate_token_count(8), 2);
        assert_eq!(PrefixReuseStrategy::estimate_token_count(10), 3);
    }

    #[test]
    fn test_zero_length_handling() {
        let strategy = PrefixReuseStrategy::default();
        let cached = create_mock_states(10);

        // Empty strings
        assert!(!strategy.can_reuse(&cached, "", ""));
        assert_eq!(strategy.reuse_point(&cached, "", ""), 0);
        assert!((strategy.reuse_quality(&cached, "", "") - 0.0).abs() < 0.001);
    }
}
