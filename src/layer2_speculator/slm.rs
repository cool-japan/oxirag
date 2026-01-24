//! Small Language Model (SLM) interface for the Speculator layer.
//!
//! This module provides a generic trait for integrating small language models
//! into the speculative verification pipeline. It includes a mock implementation
//! for testing and development.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::SpeculatorError;

/// Configuration for SLM generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlmConfig {
    /// Model identifier (e.g., "microsoft/phi-2").
    pub model_id: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic, higher = more random).
    pub temperature: f32,
    /// Top-p (nucleus) sampling parameter.
    pub top_p: f32,
    /// Top-k sampling parameter.
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty).
    pub repetition_penalty: f32,
}

impl Default for SlmConfig {
    fn default() -> Self {
        Self {
            model_id: "mock-slm".to_string(),
            max_tokens: 256,
            temperature: 0.3,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
        }
    }
}

impl SlmConfig {
    /// Create a new SLM configuration with the given model ID.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Set the maximum tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p sampling.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set top-k sampling.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set repetition penalty.
    #[must_use]
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }
}

/// Result of text generation.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// The generated text.
    pub text: String,
    /// Token IDs (if available).
    pub tokens: Vec<u32>,
    /// Log probabilities for each token (if available).
    pub logprobs: Option<Vec<f32>>,
    /// Reason for generation completion.
    pub finish_reason: FinishReason,
}

impl GenerationOutput {
    /// Create a new generation output.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
            logprobs: None,
            finish_reason: FinishReason::Stop,
        }
    }

    /// Set the tokens.
    #[must_use]
    pub fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.tokens = tokens;
        self
    }

    /// Set the log probabilities.
    #[must_use]
    pub fn with_logprobs(mut self, logprobs: Vec<f32>) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    /// Set the finish reason.
    #[must_use]
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = reason;
        self
    }
}

/// Reason for generation completion.
#[derive(Debug, Clone, Default)]
pub enum FinishReason {
    /// Generation stopped normally (EOS token).
    #[default]
    Stop,
    /// Maximum token limit reached.
    MaxTokens,
    /// An error occurred during generation.
    Error(String),
}

impl FinishReason {
    /// Check if the finish reason indicates success.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Stop | Self::MaxTokens)
    }

    /// Check if the finish reason indicates an error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

/// Trait for small language models used in speculation.
#[async_trait]
pub trait SmallLanguageModel: Send + Sync {
    /// Generate text from a prompt.
    ///
    /// # Arguments
    /// * `prompt` - The input prompt for generation
    /// * `config` - Configuration for generation
    ///
    /// # Returns
    /// The generated output including text, tokens, and log probabilities.
    async fn generate(
        &self,
        prompt: &str,
        config: &SlmConfig,
    ) -> Result<GenerationOutput, SpeculatorError>;

    /// Get log probabilities for a given text.
    ///
    /// # Arguments
    /// * `text` - The text to analyze
    ///
    /// # Returns
    /// Log probabilities for each token in the text.
    async fn get_logprobs(&self, text: &str) -> Result<Vec<f32>, SpeculatorError>;

    /// Verify if draft text is likely correct given context.
    ///
    /// # Arguments
    /// * `draft` - The draft text to verify
    /// * `context` - The context for verification
    ///
    /// # Returns
    /// A confidence score (0.0 to 1.0) indicating verification confidence.
    async fn verify_text(&self, draft: &str, context: &str) -> Result<f32, SpeculatorError>;

    /// Get the model configuration.
    fn model_info(&self) -> &SlmConfig;
}

/// A mock SLM for testing without actual model inference.
pub struct MockSlm {
    config: SlmConfig,
    response_delay_ms: u64,
    /// Deterministic seed for reproducible outputs.
    seed: u64,
}

impl MockSlm {
    /// Create a new mock SLM with the given configuration.
    #[must_use]
    pub fn new(config: SlmConfig) -> Self {
        Self {
            config,
            response_delay_ms: 0,
            seed: 42,
        }
    }

    /// Set a delay for simulating inference time.
    #[must_use]
    pub fn with_delay(mut self, delay_ms: u64) -> Self {
        self.response_delay_ms = delay_ms;
        self
    }

    /// Set the random seed for reproducible outputs.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate mock tokens from text.
    #[allow(clippy::unused_self)]
    fn mock_tokenize(&self, text: &str) -> Vec<u32> {
        // Simple character-based tokenization for testing
        text.chars()
            .enumerate()
            .map(|(i, c)| {
                #[allow(clippy::cast_possible_truncation)]
                let token = (c as u32) ^ ((i as u32) % 256);
                token
            })
            .collect()
    }

    /// Generate mock log probabilities.
    #[allow(clippy::unused_self)]
    fn mock_logprobs(&self, num_tokens: usize) -> Vec<f32> {
        (0..num_tokens)
            .map(|i| {
                // Generate semi-random log probs
                #[allow(clippy::cast_precision_loss)]
                let base = -((i % 10) as f32 * 0.1 + 0.5);
                base
            })
            .collect()
    }

    /// Simulate processing delay.
    #[cfg(feature = "native")]
    async fn simulate_delay(&self) {
        if self.response_delay_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.response_delay_ms)).await;
        }
    }

    #[cfg(not(feature = "native"))]
    async fn simulate_delay(&self) {
        // No delay simulation in WASM
    }
}

impl Default for MockSlm {
    fn default() -> Self {
        Self::new(SlmConfig::default())
    }
}

#[async_trait]
impl SmallLanguageModel for MockSlm {
    async fn generate(
        &self,
        prompt: &str,
        config: &SlmConfig,
    ) -> Result<GenerationOutput, SpeculatorError> {
        self.simulate_delay().await;

        // Generate a mock response based on the prompt
        let response = if prompt.to_lowercase().contains("verify") {
            "The information appears to be consistent with the provided context. ACCEPT."
        } else if prompt.to_lowercase().contains("revise") {
            "Based on the context, here is a revised answer that addresses the identified issues."
        } else {
            "This is a mock response from the SLM for testing purposes."
        };

        // Truncate to max tokens (approximate)
        let truncated: String = response.chars().take(config.max_tokens * 4).collect();

        let tokens = self.mock_tokenize(&truncated);
        let logprobs = self.mock_logprobs(tokens.len());

        let finish_reason = if truncated.len() < response.len() {
            FinishReason::MaxTokens
        } else {
            FinishReason::Stop
        };

        Ok(GenerationOutput::new(truncated)
            .with_tokens(tokens)
            .with_logprobs(logprobs)
            .with_finish_reason(finish_reason))
    }

    async fn get_logprobs(&self, text: &str) -> Result<Vec<f32>, SpeculatorError> {
        self.simulate_delay().await;

        let tokens = self.mock_tokenize(text);
        Ok(self.mock_logprobs(tokens.len()))
    }

    async fn verify_text(&self, draft: &str, context: &str) -> Result<f32, SpeculatorError> {
        self.simulate_delay().await;

        // Simple heuristic-based verification
        let draft_lower = draft.to_lowercase();
        let context_lower = context.to_lowercase();

        // Count word overlap
        let draft_words: std::collections::HashSet<&str> = draft_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        let context_words: std::collections::HashSet<&str> = context_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        if draft_words.is_empty() || context_words.is_empty() {
            return Ok(0.5);
        }

        #[allow(clippy::cast_precision_loss)]
        let overlap = draft_words.intersection(&context_words).count() as f32;
        #[allow(clippy::cast_precision_loss)]
        let total = draft_words.len() as f32;

        let confidence = (overlap / total).clamp(0.0, 1.0) * 0.5 + 0.5;
        Ok(confidence)
    }

    fn model_info(&self) -> &SlmConfig {
        &self.config
    }
}

/// Builder for creating SLM instances.
pub struct SlmBuilder {
    config: SlmConfig,
    delay_ms: u64,
}

impl SlmBuilder {
    /// Create a new SLM builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SlmConfig::default(),
            delay_ms: 0,
        }
    }

    /// Set the model ID.
    #[must_use]
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.config.model_id = model_id.into();
        self
    }

    /// Set the maximum tokens.
    #[must_use]
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_tokens = max_tokens;
        self
    }

    /// Set the temperature.
    #[must_use]
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set top-p sampling.
    #[must_use]
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.config.top_p = top_p;
        self
    }

    /// Set top-k sampling.
    #[must_use]
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.config.top_k = top_k;
        self
    }

    /// Set repetition penalty.
    #[must_use]
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.config.repetition_penalty = penalty;
        self
    }

    /// Set simulated delay.
    #[must_use]
    pub fn delay_ms(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Build a mock SLM.
    #[must_use]
    pub fn build_mock(self) -> MockSlm {
        MockSlm::new(self.config).with_delay(self.delay_ms)
    }
}

impl Default for SlmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_slm_config_default() {
        let config = SlmConfig::default();
        assert_eq!(config.model_id, "mock-slm");
        assert_eq!(config.max_tokens, 256);
        assert_eq!(config.temperature, 0.3);
    }

    #[test]
    fn test_slm_config_builder() {
        let config = SlmConfig::new("test-model")
            .with_max_tokens(512)
            .with_temperature(0.5)
            .with_top_p(0.95)
            .with_top_k(100)
            .with_repetition_penalty(1.2);

        assert_eq!(config.model_id, "test-model");
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.top_p, 0.95);
        assert_eq!(config.top_k, 100);
        assert_eq!(config.repetition_penalty, 1.2);
    }

    #[test]
    fn test_generation_output() {
        let output = GenerationOutput::new("test output")
            .with_tokens(vec![1, 2, 3])
            .with_logprobs(vec![-0.5, -0.3, -0.2])
            .with_finish_reason(FinishReason::Stop);

        assert_eq!(output.text, "test output");
        assert_eq!(output.tokens, vec![1, 2, 3]);
        assert!(output.logprobs.is_some());
        assert!(matches!(output.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn test_finish_reason() {
        assert!(FinishReason::Stop.is_success());
        assert!(FinishReason::MaxTokens.is_success());
        assert!(!FinishReason::Stop.is_error());
        assert!(FinishReason::Error("test".to_string()).is_error());
        assert!(!FinishReason::Error("test".to_string()).is_success());
    }

    #[test]
    fn test_finish_reason_default() {
        let reason = FinishReason::default();
        assert!(matches!(reason, FinishReason::Stop));
    }

    #[test]
    fn test_mock_slm_creation() {
        let slm = MockSlm::new(SlmConfig::default());
        assert_eq!(slm.model_info().model_id, "mock-slm");
    }

    #[test]
    fn test_mock_slm_with_delay() {
        let slm = MockSlm::default().with_delay(100);
        assert_eq!(slm.response_delay_ms, 100);
    }

    #[test]
    fn test_mock_slm_with_seed() {
        let slm = MockSlm::default().with_seed(12345);
        assert_eq!(slm.seed, 12345);
    }

    #[tokio::test]
    async fn test_mock_slm_generate() {
        let slm = MockSlm::default();
        let config = SlmConfig::default();

        let result = slm.generate("verify this text", &config).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.text.is_empty());
        assert!(output.text.contains("ACCEPT"));
    }

    #[tokio::test]
    async fn test_mock_slm_generate_revise() {
        let slm = MockSlm::default();
        let config = SlmConfig::default();

        let result = slm.generate("revise this answer", &config).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.text.contains("revised"));
    }

    #[tokio::test]
    async fn test_mock_slm_get_logprobs() {
        let slm = MockSlm::default();

        let result = slm.get_logprobs("test text").await;
        assert!(result.is_ok());

        let logprobs = result.unwrap();
        assert!(!logprobs.is_empty());
        for prob in logprobs {
            assert!(prob <= 0.0); // Log probs should be negative
        }
    }

    #[tokio::test]
    async fn test_mock_slm_verify_text() {
        let slm = MockSlm::default();

        // High overlap should give high confidence
        let confidence = slm
            .verify_text("Paris is the capital", "Paris is the capital of France")
            .await
            .unwrap();
        assert!(confidence >= 0.5);

        // Low overlap should give lower confidence
        let confidence_low = slm
            .verify_text("xyz abc def", "Paris is the capital of France")
            .await
            .unwrap();
        assert!(confidence_low <= confidence);
    }

    #[tokio::test]
    async fn test_mock_slm_verify_empty() {
        let slm = MockSlm::default();

        let confidence = slm.verify_text("", "context").await.unwrap();
        assert_eq!(confidence, 0.5);

        let confidence2 = slm.verify_text("draft", "").await.unwrap();
        assert_eq!(confidence2, 0.5);
    }

    #[test]
    fn test_slm_builder() {
        let slm = SlmBuilder::new()
            .model_id("custom-model")
            .max_tokens(128)
            .temperature(0.7)
            .top_p(0.8)
            .top_k(40)
            .repetition_penalty(1.5)
            .delay_ms(50)
            .build_mock();

        let info = slm.model_info();
        assert_eq!(info.model_id, "custom-model");
        assert_eq!(info.max_tokens, 128);
        assert_eq!(info.temperature, 0.7);
        assert_eq!(slm.response_delay_ms, 50);
    }

    #[test]
    fn test_slm_builder_default() {
        let builder = SlmBuilder::default();
        let slm = builder.build_mock();
        assert_eq!(slm.model_info().model_id, "mock-slm");
    }

    #[tokio::test]
    async fn test_mock_slm_tokens() {
        let slm = MockSlm::default();
        let config = SlmConfig::default().with_max_tokens(1000);

        let result = slm.generate("test prompt", &config).await.unwrap();

        assert!(!result.tokens.is_empty());
        assert!(result.logprobs.is_some());
        assert_eq!(result.tokens.len(), result.logprobs.as_ref().unwrap().len());
    }

    #[tokio::test]
    async fn test_mock_slm_max_tokens_truncation() {
        let slm = MockSlm::default();
        let config = SlmConfig::default().with_max_tokens(5);

        let result = slm.generate("generate long text", &config).await.unwrap();

        // Should truncate to approximately max_tokens * 4 characters
        assert!(result.text.len() <= 20);
    }

    #[test]
    fn test_mock_tokenize() {
        let slm = MockSlm::default();
        let tokens = slm.mock_tokenize("hello");

        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_mock_logprobs() {
        let slm = MockSlm::default();
        let logprobs = slm.mock_logprobs(10);

        assert_eq!(logprobs.len(), 10);
        for prob in logprobs {
            assert!(prob <= 0.0);
        }
    }
}
