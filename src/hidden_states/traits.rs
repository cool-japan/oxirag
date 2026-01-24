//! Traits for hidden state providers and reuse strategies.
//!
//! This module defines the core traits for extracting and reusing hidden states
//! from transformer models.

#![allow(clippy::collapsible_if)]

use async_trait::async_trait;

use super::types::{HiddenStateConfig, ModelHiddenStates, ModelKVCache};
use crate::error::HiddenStateError;

/// Provider for extracting hidden states from transformer models.
///
/// Implementations of this trait provide the capability to extract internal
/// hidden states from model forward passes, which can be cached and reused
/// for speculative RAG operations.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to support concurrent access.
///
/// # Example
///
/// ```rust,ignore
/// use oxirag::hidden_states::{HiddenStateProvider, MockHiddenStateProvider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = MockHiddenStateProvider::new("test-model", 12, 768);
///
///     let states = provider.extract_hidden_states("Hello, world!").await.unwrap();
///     println!("Extracted {} layers", states.layers.len());
/// }
/// ```
#[async_trait]
pub trait HiddenStateProvider: Send + Sync {
    /// Extract hidden states from input text.
    ///
    /// This performs a forward pass through the model and captures the
    /// hidden states from each transformer layer.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails or state extraction is not supported.
    async fn extract_hidden_states(
        &self,
        text: &str,
    ) -> Result<ModelHiddenStates, HiddenStateError>;

    /// Extract hidden states with an existing KV cache.
    ///
    /// This is useful for incremental generation where previous context
    /// has already been processed and cached.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails or the cache is incompatible.
    async fn extract_with_kv_cache(
        &self,
        text: &str,
        past_kv: Option<&ModelKVCache>,
    ) -> Result<(ModelHiddenStates, ModelKVCache), HiddenStateError>;

    /// Get the model configuration for hidden state operations.
    fn model_config(&self) -> &HiddenStateConfig;

    /// Get the model identifier.
    fn model_id(&self) -> &str;

    /// Get the number of layers in the model.
    fn num_layers(&self) -> usize;

    /// Get the hidden dimension of the model.
    fn hidden_dim(&self) -> usize;

    /// Check if this provider supports attention weight capture.
    fn supports_attention_weights(&self) -> bool {
        self.model_config().capture_attention_weights
    }

    /// Check if this provider supports KV caching.
    fn supports_kv_cache(&self) -> bool {
        true
    }
}

/// Strategy for reusing hidden states across different inputs.
///
/// Implementations determine whether and how cached hidden states can be
/// reused for new inputs, enabling efficient speculative decoding.
pub trait StateReuseStrategy: Send + Sync {
    /// Determine if cached states can be reused for new input.
    ///
    /// # Arguments
    ///
    /// * `cached` - The cached hidden states
    /// * `new_text` - The new input text
    /// * `cached_text` - The text that produced the cached states
    ///
    /// # Returns
    ///
    /// `true` if the cached states can be reused, `false` otherwise.
    fn can_reuse(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> bool;

    /// Calculate the reuse point (how many tokens can be reused).
    ///
    /// Returns the number of tokens from the cached states that can be
    /// reused for the new input.
    fn reuse_point(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> usize;

    /// Score the quality of state reuse (0.0 to 1.0).
    ///
    /// Higher scores indicate better reuse potential:
    /// - 1.0: Perfect reuse (identical prefix)
    /// - 0.5: Partial reuse
    /// - 0.0: No reuse possible
    fn reuse_quality(&self, cached: &ModelHiddenStates, new_text: &str, cached_text: &str) -> f32;

    /// Get a description of this strategy.
    fn description(&self) -> &'static str;
}

/// A boxed state reuse strategy for dynamic dispatch.
pub type BoxedStateReuseStrategy = Box<dyn StateReuseStrategy>;

/// Extension trait for hidden state providers with additional utilities.
#[async_trait]
pub trait HiddenStateProviderExt: HiddenStateProvider {
    /// Extract hidden states and automatically cache them.
    async fn extract_and_cache(
        &self,
        text: &str,
        cache: &mut super::cache::HiddenStateCache,
    ) -> Result<ModelHiddenStates, HiddenStateError> {
        // Check cache first
        if let Some(cached) = cache.get(text) {
            return Ok(cached.states.clone());
        }

        // Extract states
        let states = self.extract_hidden_states(text).await?;

        // Cache the results
        cache.put(text.to_string(), states.clone(), None);

        Ok(states)
    }

    /// Extract states with KV cache support and automatic caching.
    async fn extract_with_caching(
        &self,
        text: &str,
        past_kv: Option<&ModelKVCache>,
        cache: &mut super::cache::HiddenStateCache,
    ) -> Result<(ModelHiddenStates, ModelKVCache), HiddenStateError> {
        // Check cache first
        if past_kv.is_none() {
            if let Some(cached) = cache.get(text) {
                if let Some(ref kv) = cached.kv_cache {
                    return Ok((cached.states.clone(), kv.clone()));
                }
            }
        }

        // Extract states
        let (states, kv) = self.extract_with_kv_cache(text, past_kv).await?;

        // Cache the results if no past KV was provided
        if past_kv.is_none() {
            cache.put(text.to_string(), states.clone(), Some(kv.clone()));
        }

        Ok((states, kv))
    }
}

// Blanket implementation for all HiddenStateProvider
impl<T: HiddenStateProvider> HiddenStateProviderExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hidden_states::extractor::MockHiddenStateProvider;

    #[tokio::test]
    async fn test_mock_provider_extract() {
        let provider = MockHiddenStateProvider::new("test-model", 12, 768);
        let states = provider
            .extract_hidden_states("Hello, world!")
            .await
            .unwrap();

        assert_eq!(states.model_id, "test-model");
        assert_eq!(states.num_layers, 12);
        assert_eq!(states.hidden_dim, 768);
    }

    #[tokio::test]
    async fn test_mock_provider_with_kv_cache() {
        let provider = MockHiddenStateProvider::new("test-model", 6, 512);
        let (states, kv) = provider
            .extract_with_kv_cache("Test input", None)
            .await
            .unwrap();

        assert_eq!(states.model_id, "test-model");
        assert_eq!(kv.model_id, "test-model");
        assert_eq!(kv.layers.len(), 6);
    }

    #[test]
    fn test_provider_trait_methods() {
        let provider = MockHiddenStateProvider::new("test-model", 12, 768);

        assert_eq!(provider.model_id(), "test-model");
        assert_eq!(provider.num_layers(), 12);
        assert_eq!(provider.hidden_dim(), 768);
        assert!(provider.supports_kv_cache());
    }
}
