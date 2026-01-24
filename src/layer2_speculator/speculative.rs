//! Speculative decoding implementation using hidden states.
//!
//! This module implements the speculative decoding algorithm that uses a small
//! draft model to speculatively generate tokens, which are then verified by
//! a larger target model using hidden state comparison.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::error::SpeculatorError;
use crate::layer2_speculator::hidden_states::{
    HiddenStateCache, HiddenStateCacheConfig, HiddenStateProvider, ModelHiddenStates, ModelKVCache,
};

/// Configuration for speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeDecodingConfig {
    /// Draft model identifier.
    pub draft_model_id: String,
    /// Target model identifier.
    pub target_model_id: String,
    /// Number of speculative tokens to generate (K in the algorithm).
    pub num_speculative_tokens: usize,
    /// Temperature for sampling.
    pub temperature: f32,
    /// Acceptance threshold for probability comparison.
    pub acceptance_threshold: f32,
    /// Whether to use hidden state caching.
    pub use_hidden_state_cache: bool,
    /// Maximum number of cache entries.
    pub max_cache_entries: usize,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            draft_model_id: "draft-model".to_string(),
            target_model_id: "target-model".to_string(),
            num_speculative_tokens: 4,
            temperature: 0.7,
            acceptance_threshold: 0.8,
            use_hidden_state_cache: true,
            max_cache_entries: 1000,
        }
    }
}

impl SpeculativeDecodingConfig {
    /// Create a new speculative decoding configuration.
    #[must_use]
    pub fn new(draft_model_id: impl Into<String>, target_model_id: impl Into<String>) -> Self {
        Self {
            draft_model_id: draft_model_id.into(),
            target_model_id: target_model_id.into(),
            ..Default::default()
        }
    }

    /// Set the number of speculative tokens.
    #[must_use]
    pub fn with_num_speculative_tokens(mut self, k: usize) -> Self {
        self.num_speculative_tokens = k;
        self
    }

    /// Set the temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the acceptance threshold.
    #[must_use]
    pub fn with_acceptance_threshold(mut self, threshold: f32) -> Self {
        self.acceptance_threshold = threshold;
        self
    }

    /// Enable or disable hidden state caching.
    #[must_use]
    pub fn with_hidden_state_cache(mut self, use_cache: bool) -> Self {
        self.use_hidden_state_cache = use_cache;
        self
    }

    /// Set the maximum cache entries.
    #[must_use]
    pub fn with_max_cache_entries(mut self, max_entries: usize) -> Self {
        self.max_cache_entries = max_entries;
        self
    }
}

/// Token with its probability information.
#[derive(Debug, Clone)]
pub struct TokenWithProb {
    /// The token ID.
    pub token_id: u32,
    /// The token text (if available).
    pub token_text: String,
    /// The probability of this token.
    pub probability: f32,
    /// The log probability of this token.
    pub log_prob: f32,
}

impl TokenWithProb {
    /// Create a new token with probability.
    #[must_use]
    pub fn new(token_id: u32, token_text: impl Into<String>, probability: f32) -> Self {
        let log_prob = if probability > 0.0 {
            probability.ln()
        } else {
            f32::NEG_INFINITY
        };
        Self {
            token_id,
            token_text: token_text.into(),
            probability,
            log_prob,
        }
    }

    /// Create from token ID and log probability.
    #[must_use]
    pub fn from_log_prob(token_id: u32, token_text: impl Into<String>, log_prob: f32) -> Self {
        let probability = log_prob.exp();
        Self {
            token_id,
            token_text: token_text.into(),
            probability,
            log_prob,
        }
    }
}

/// Result of a single speculative decoding step.
#[derive(Debug, Clone)]
pub struct SpeculativeStep {
    /// The draft tokens proposed by the draft model.
    pub draft_tokens: Vec<TokenWithProb>,
    /// The tokens that were accepted.
    pub accepted_tokens: Vec<TokenWithProb>,
    /// The position where rejection occurred (if any).
    pub rejected_at: Option<usize>,
    /// Correction token from target model (if draft was rejected).
    pub target_correction: Option<TokenWithProb>,
    /// Acceptance rate for this step.
    pub acceptance_rate: f32,
    /// Hidden states from the draft model (if available).
    pub draft_hidden_states: Option<ModelHiddenStates>,
    /// Hidden states from the target model (if available).
    pub target_hidden_states: Option<ModelHiddenStates>,
}

impl SpeculativeStep {
    /// Create a new speculative step.
    #[must_use]
    pub fn new(draft_tokens: Vec<TokenWithProb>, accepted_tokens: Vec<TokenWithProb>) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let acceptance_rate = if draft_tokens.is_empty() {
            0.0
        } else {
            accepted_tokens.len() as f32 / draft_tokens.len() as f32
        };

        Self {
            draft_tokens,
            accepted_tokens,
            rejected_at: None,
            target_correction: None,
            acceptance_rate,
            draft_hidden_states: None,
            target_hidden_states: None,
        }
    }

    /// Set the rejection position.
    #[must_use]
    pub fn with_rejected_at(mut self, pos: usize) -> Self {
        self.rejected_at = Some(pos);
        self
    }

    /// Set the target correction token.
    #[must_use]
    pub fn with_correction(mut self, correction: TokenWithProb) -> Self {
        self.target_correction = Some(correction);
        self
    }

    /// Set the draft hidden states.
    #[must_use]
    pub fn with_draft_hidden_states(mut self, states: ModelHiddenStates) -> Self {
        self.draft_hidden_states = Some(states);
        self
    }

    /// Set the target hidden states.
    #[must_use]
    pub fn with_target_hidden_states(mut self, states: ModelHiddenStates) -> Self {
        self.target_hidden_states = Some(states);
        self
    }

    /// Get the total number of tokens generated in this step.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.accepted_tokens.len() + usize::from(self.target_correction.is_some())
    }
}

/// Statistics for speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total number of draft tokens generated.
    pub total_draft_tokens: u64,
    /// Number of accepted tokens.
    pub accepted_tokens: u64,
    /// Number of rejected tokens.
    pub rejected_tokens: u64,
    /// Number of correction tokens from target model.
    pub correction_tokens: u64,
    /// Average acceptance rate.
    pub avg_acceptance_rate: f32,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Total number of speculative steps.
    pub total_steps: u64,
}

impl SpeculativeStats {
    /// Update stats with a new step result.
    pub fn update(&mut self, step: &SpeculativeStep) {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.total_draft_tokens += step.draft_tokens.len() as u64;
            self.accepted_tokens += step.accepted_tokens.len() as u64;
            self.rejected_tokens += (step.draft_tokens.len() - step.accepted_tokens.len()) as u64;
        }

        if step.target_correction.is_some() {
            self.correction_tokens += 1;
        }

        self.total_steps += 1;

        // Update rolling average
        #[allow(clippy::cast_precision_loss)]
        {
            let n = self.total_steps as f32;
            self.avg_acceptance_rate =
                (self.avg_acceptance_rate * (n - 1.0) + step.acceptance_rate) / n;
        }
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Get the cache hit rate.
    #[must_use]
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                self.cache_hits as f32 / total as f32
            }
        }
    }

    /// Get the overall speedup factor.
    #[must_use]
    pub fn speedup_factor(&self) -> f32 {
        // Speedup = tokens generated / target model calls
        // Each step generates accepted_tokens + 1 (correction or verification)
        // but only requires 1 target model call
        if self.total_steps == 0 {
            1.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                let tokens_generated = self.accepted_tokens + self.correction_tokens;
                tokens_generated as f32 / self.total_steps as f32
            }
        }
    }
}

/// Output from speculative decoding generation.
#[derive(Debug, Clone)]
pub struct SpeculativeOutput {
    /// The generated text.
    pub text: String,
    /// The tokens with probabilities.
    pub tokens: Vec<TokenWithProb>,
    /// All speculative steps.
    pub steps: Vec<SpeculativeStep>,
    /// Total number of draft tokens proposed.
    pub total_draft_tokens: usize,
    /// Total number of accepted tokens.
    pub total_accepted: usize,
    /// Final hidden states from the generation.
    pub final_hidden_states: Option<ModelHiddenStates>,
}

impl SpeculativeOutput {
    /// Create a new speculative output.
    #[must_use]
    pub fn new(text: impl Into<String>, tokens: Vec<TokenWithProb>) -> Self {
        Self {
            text: text.into(),
            tokens,
            steps: Vec::new(),
            total_draft_tokens: 0,
            total_accepted: 0,
            final_hidden_states: None,
        }
    }

    /// Add a speculative step.
    #[must_use]
    pub fn with_step(mut self, step: SpeculativeStep) -> Self {
        self.total_draft_tokens += step.draft_tokens.len();
        self.total_accepted += step.accepted_tokens.len();
        self.steps.push(step);
        self
    }

    /// Set the final hidden states.
    #[must_use]
    pub fn with_hidden_states(mut self, states: ModelHiddenStates) -> Self {
        self.final_hidden_states = Some(states);
        self
    }

    /// Get the overall acceptance rate.
    #[must_use]
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                self.total_accepted as f32 / self.total_draft_tokens as f32
            }
        }
    }
}

/// Verification result from the target model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Target model probabilities for each draft position.
    pub target_probs: Vec<Vec<f32>>,
    /// Hidden states from the target model.
    pub hidden_states: ModelHiddenStates,
    /// KV cache after verification.
    pub kv_cache: ModelKVCache,
}

impl VerificationResult {
    /// Create a new verification result.
    #[must_use]
    pub fn new(
        target_probs: Vec<Vec<f32>>,
        hidden_states: ModelHiddenStates,
        kv_cache: ModelKVCache,
    ) -> Self {
        Self {
            target_probs,
            hidden_states,
            kv_cache,
        }
    }
}

/// Speculative decoder using hidden states.
pub struct SpeculativeDecoder<D, T>
where
    D: HiddenStateProvider,
    T: HiddenStateProvider,
{
    draft_model: D,
    target_model: T,
    config: SpeculativeDecodingConfig,
    #[allow(dead_code)]
    hidden_state_cache: HiddenStateCache,
    stats: SpeculativeStats,
}

impl<D, T> SpeculativeDecoder<D, T>
where
    D: HiddenStateProvider,
    T: HiddenStateProvider,
{
    /// Create a new speculative decoder.
    #[must_use]
    pub fn new(draft_model: D, target_model: T, config: SpeculativeDecodingConfig) -> Self {
        let cache_config = HiddenStateCacheConfig {
            max_entries: config.max_cache_entries,
            ..Default::default()
        };

        Self {
            draft_model,
            target_model,
            config,
            hidden_state_cache: HiddenStateCache::new(cache_config),
            stats: SpeculativeStats::default(),
        }
    }

    /// Generate tokens using speculative decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if token generation fails.
    pub async fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<SpeculativeOutput, SpeculatorError> {
        let mut generated_tokens: Vec<TokenWithProb> = Vec::new();
        let mut generated_text = String::new();
        let mut steps = Vec::new();
        let mut context = prompt.to_string();
        let mut past_kv: Option<ModelKVCache> = None;

        let mut tokens_generated = 0;

        while tokens_generated < max_tokens {
            let step = self.speculative_step(&context, past_kv.as_ref()).await?;

            // Collect accepted tokens
            for token in &step.accepted_tokens {
                generated_tokens.push(token.clone());
                generated_text.push_str(&token.token_text);
                context.push_str(&token.token_text);
            }

            // Add correction token if present
            if let Some(correction) = &step.target_correction {
                generated_tokens.push(correction.clone());
                generated_text.push_str(&correction.token_text);
                context.push_str(&correction.token_text);
            }

            tokens_generated += step.total_tokens();

            // Update KV cache from target model verification
            if let Some(states) = &step.target_hidden_states {
                past_kv = Some(ModelKVCache::new(
                    &self.config.target_model_id,
                    12, // Default num_heads
                    64, // Default head_dim
                ));
                if let Some(ref mut kv) = past_kv {
                    #[allow(clippy::cast_possible_truncation)]
                    kv.set_seq_len(states.input_tokens.len());
                }
            }

            self.stats.update(&step);
            steps.push(step);

            // Check for end of sequence (simple heuristic)
            if generated_text.ends_with('.') && generated_text.len() > 10 {
                // Could check for EOS token here
                break;
            }
        }

        let mut output = SpeculativeOutput::new(generated_text, generated_tokens);
        for step in steps {
            output = output.with_step(step);
        }

        Ok(output)
    }

    /// Perform a single speculative decoding step.
    async fn speculative_step(
        &mut self,
        context: &str,
        past_kv: Option<&ModelKVCache>,
    ) -> Result<SpeculativeStep, SpeculatorError> {
        // Draft K tokens
        let (draft_tokens, draft_states) = self
            .draft_tokens(context, self.config.num_speculative_tokens, past_kv)
            .await?;

        if draft_tokens.is_empty() {
            return Ok(SpeculativeStep::new(Vec::new(), Vec::new()));
        }

        // Verify with target model
        let verification = self.verify_tokens(context, &draft_tokens, past_kv).await?;

        // Accept/reject tokens
        let mut accepted = Vec::new();
        let mut rejected_at = None;

        for (i, draft_token) in draft_tokens.iter().enumerate() {
            let target_prob = verification
                .target_probs
                .get(i)
                .and_then(|probs| probs.get(draft_token.token_id as usize))
                .copied()
                .unwrap_or(0.0);

            if self.acceptance_criterion(draft_token.probability, target_prob) {
                accepted.push(draft_token.clone());
            } else {
                rejected_at = Some(i);
                break;
            }
        }

        let mut step = SpeculativeStep::new(draft_tokens.clone(), accepted)
            .with_draft_hidden_states(draft_states)
            .with_target_hidden_states(verification.hidden_states);

        if let Some(pos) = rejected_at {
            step = step.with_rejected_at(pos);

            // Sample correction token from target distribution
            if let Some(target_dist) = verification.target_probs.get(pos) {
                let draft_dist: Vec<f32> = (0..target_dist.len()).map(|_| 0.0).collect();
                let correction = self.sample_correction(target_dist, &draft_dist);
                step = step.with_correction(correction);
            }
        }

        Ok(step)
    }

    /// Draft K tokens using the draft model.
    async fn draft_tokens(
        &self,
        context: &str,
        k: usize,
        _past_kv: Option<&ModelKVCache>,
    ) -> Result<(Vec<TokenWithProb>, ModelHiddenStates), SpeculatorError> {
        // Get hidden states from draft model
        let states = self.draft_model.get_hidden_states(context).await?;

        // Generate K tokens (mock implementation - in practice would use the model's generate)
        let tokens: Vec<TokenWithProb> = (0..k)
            .map(|i| {
                // Generate mock tokens based on context
                let mut hasher = DefaultHasher::new();
                context.hash(&mut hasher);
                i.hash(&mut hasher);
                let hash = hasher.finish();

                #[allow(clippy::cast_possible_truncation)]
                let token_id = (hash % 50000) as u32;
                #[allow(clippy::cast_precision_loss)]
                let prob = 0.5 + (hash % 50) as f32 / 100.0;

                TokenWithProb::new(token_id, format!("[t{i}]"), prob)
            })
            .collect();

        Ok((tokens, states))
    }

    /// Verify draft tokens using the target model.
    async fn verify_tokens(
        &self,
        context: &str,
        draft_tokens: &[TokenWithProb],
        _past_kv: Option<&ModelKVCache>,
    ) -> Result<VerificationResult, SpeculatorError> {
        // Build full context with draft tokens
        let mut full_context = context.to_string();
        for token in draft_tokens {
            full_context.push_str(&token.token_text);
        }

        // Get hidden states from target model
        let states = self.target_model.get_hidden_states(&full_context).await?;

        // Generate mock probability distributions
        let target_probs: Vec<Vec<f32>> = draft_tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                let mut probs = vec![0.001; 50000];
                // Give high probability to the draft token
                #[allow(clippy::cast_precision_loss)]
                let base_prob = 0.3 + (i as f32 * 0.1).min(0.5);
                probs[token.token_id as usize] = base_prob;
                probs
            })
            .collect();

        let kv_cache = ModelKVCache::new(self.target_model.model_id(), 12, 64);

        Ok(VerificationResult::new(target_probs, states, kv_cache))
    }

    /// Determine if a draft token should be accepted.
    fn acceptance_criterion(&self, draft_prob: f32, target_prob: f32) -> bool {
        if target_prob >= draft_prob {
            // Always accept if target probability is higher
            true
        } else if draft_prob > 0.0 {
            // Probabilistic acceptance based on ratio
            let ratio = target_prob / draft_prob;
            ratio >= self.config.acceptance_threshold
        } else {
            false
        }
    }

    /// Sample a correction token when draft is rejected.
    #[allow(clippy::unused_self)]
    fn sample_correction(&self, target_probs: &[f32], draft_probs: &[f32]) -> TokenWithProb {
        // Compute adjusted distribution: max(0, target - draft)
        let mut adjusted: Vec<f32> = target_probs
            .iter()
            .zip(draft_probs.iter())
            .map(|(t, d)| (t - d).max(0.0))
            .collect();

        // Normalize
        let sum: f32 = adjusted.iter().sum();
        if sum > 0.0 {
            for p in &mut adjusted {
                *p /= sum;
            }
        }

        // Find the token with highest adjusted probability
        let (token_id, prob) = adjusted
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((0, 0.0), |(i, p)| (i, *p));

        #[allow(clippy::cast_possible_truncation)]
        TokenWithProb::new(token_id as u32, format!("[c{token_id}]"), prob)
    }

    /// Get the current statistics.
    #[must_use]
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset the statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &SpeculativeDecodingConfig {
        &self.config
    }
}

/// Mock speculative decoder for testing.
pub struct MockSpeculativeDecoder {
    config: SpeculativeDecodingConfig,
    acceptance_rate: f32,
    stats: SpeculativeStats,
}

impl MockSpeculativeDecoder {
    /// Create a new mock speculative decoder.
    #[must_use]
    pub fn new(config: SpeculativeDecodingConfig) -> Self {
        Self {
            config,
            acceptance_rate: 0.8,
            stats: SpeculativeStats::default(),
        }
    }

    /// Set the mock acceptance rate.
    #[must_use]
    pub fn with_acceptance_rate(mut self, rate: f32) -> Self {
        self.acceptance_rate = rate;
        self
    }

    /// Generate tokens using mock speculative decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    #[allow(clippy::unused_async)]
    pub async fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<SpeculativeOutput, SpeculatorError> {
        let mut tokens = Vec::new();
        let mut text = String::new();
        let mut steps = Vec::new();

        let mut remaining = max_tokens;
        let k = self.config.num_speculative_tokens;

        while remaining > 0 {
            // Generate draft tokens
            let num_draft = k.min(remaining);
            let draft_tokens: Vec<TokenWithProb> = (0..num_draft)
                .map(|i| {
                    let mut hasher = DefaultHasher::new();
                    prompt.hash(&mut hasher);
                    i.hash(&mut hasher);
                    tokens.len().hash(&mut hasher);
                    let hash = hasher.finish();

                    #[allow(clippy::cast_possible_truncation)]
                    let token_id = (hash % 1000) as u32;
                    let token_text = format!("w{token_id} ");

                    TokenWithProb::new(token_id, token_text, 0.7)
                })
                .collect();

            // Accept tokens based on acceptance rate
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let num_accepted = (draft_tokens.len() as f32 * self.acceptance_rate).round() as usize;
            let accepted: Vec<TokenWithProb> =
                draft_tokens.iter().take(num_accepted).cloned().collect();

            let mut step = SpeculativeStep::new(draft_tokens.clone(), accepted.clone());

            // Add tokens to output
            for token in &accepted {
                tokens.push(token.clone());
                text.push_str(&token.token_text);
            }

            // Add correction if not all accepted
            if num_accepted < draft_tokens.len() {
                let correction = TokenWithProb::new(999, "corr ", 0.9);
                tokens.push(correction.clone());
                text.push_str(&correction.token_text);
                step = step
                    .with_rejected_at(num_accepted)
                    .with_correction(correction);
            }

            remaining = remaining.saturating_sub(step.total_tokens());
            self.stats.update(&step);
            steps.push(step);
        }

        let mut output = SpeculativeOutput::new(text, tokens);
        for step in steps {
            output = output.with_step(step);
        }

        Ok(output)
    }

    /// Get the current statistics.
    #[must_use]
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset the statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }
}

impl Default for MockSpeculativeDecoder {
    fn default() -> Self {
        Self::new(SpeculativeDecodingConfig::default())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::layer2_speculator::hidden_states::MockHiddenStateProvider;

    #[test]
    fn test_speculative_decoding_config_default() {
        let config = SpeculativeDecodingConfig::default();
        assert_eq!(config.num_speculative_tokens, 4);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.acceptance_threshold, 0.8);
        assert!(config.use_hidden_state_cache);
    }

    #[test]
    fn test_speculative_decoding_config_builder() {
        let config = SpeculativeDecodingConfig::new("draft", "target")
            .with_num_speculative_tokens(8)
            .with_temperature(0.5)
            .with_acceptance_threshold(0.9)
            .with_hidden_state_cache(false)
            .with_max_cache_entries(500);

        assert_eq!(config.draft_model_id, "draft");
        assert_eq!(config.target_model_id, "target");
        assert_eq!(config.num_speculative_tokens, 8);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.acceptance_threshold, 0.9);
        assert!(!config.use_hidden_state_cache);
        assert_eq!(config.max_cache_entries, 500);
    }

    #[test]
    fn test_token_with_prob() {
        let token = TokenWithProb::new(42, "hello", 0.5);
        assert_eq!(token.token_id, 42);
        assert_eq!(token.token_text, "hello");
        assert_eq!(token.probability, 0.5);
        assert!((token.log_prob - (-0.693)).abs() < 0.01);
    }

    #[test]
    fn test_token_with_prob_from_log_prob() {
        let token = TokenWithProb::from_log_prob(10, "test", -1.0);
        assert_eq!(token.token_id, 10);
        assert!((token.probability - 0.368).abs() < 0.01);
        assert_eq!(token.log_prob, -1.0);
    }

    #[test]
    fn test_speculative_step() {
        let draft = vec![
            TokenWithProb::new(1, "a", 0.8),
            TokenWithProb::new(2, "b", 0.7),
            TokenWithProb::new(3, "c", 0.6),
        ];
        let accepted = vec![
            TokenWithProb::new(1, "a", 0.8),
            TokenWithProb::new(2, "b", 0.7),
        ];

        let step = SpeculativeStep::new(draft, accepted);

        assert_eq!(step.draft_tokens.len(), 3);
        assert_eq!(step.accepted_tokens.len(), 2);
        assert!((step.acceptance_rate - 0.667).abs() < 0.01);
        assert_eq!(step.total_tokens(), 2);
    }

    #[test]
    fn test_speculative_step_with_correction() {
        let draft = vec![TokenWithProb::new(1, "a", 0.8)];
        let accepted = vec![];
        let correction = TokenWithProb::new(99, "x", 0.9);

        let step = SpeculativeStep::new(draft, accepted)
            .with_rejected_at(0)
            .with_correction(correction);

        assert_eq!(step.rejected_at, Some(0));
        assert!(step.target_correction.is_some());
        assert_eq!(step.total_tokens(), 1); // Just the correction
    }

    #[test]
    fn test_speculative_stats() {
        let mut stats = SpeculativeStats::default();

        let step1 = SpeculativeStep::new(
            vec![
                TokenWithProb::new(1, "a", 0.8),
                TokenWithProb::new(2, "b", 0.7),
            ],
            vec![TokenWithProb::new(1, "a", 0.8)],
        );

        stats.update(&step1);

        assert_eq!(stats.total_draft_tokens, 2);
        assert_eq!(stats.accepted_tokens, 1);
        assert_eq!(stats.rejected_tokens, 1);
        assert_eq!(stats.total_steps, 1);
        assert_eq!(stats.avg_acceptance_rate, 0.5);
    }

    #[test]
    fn test_speculative_stats_cache() {
        let mut stats = SpeculativeStats::default();

        stats.record_cache_hit();
        stats.record_cache_hit();
        stats.record_cache_miss();

        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
        assert!((stats.cache_hit_rate() - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_speculative_stats_speedup() {
        let mut stats = SpeculativeStats::default();

        // Simulate 10 steps with 3 accepted + 1 correction each = 4 tokens per step
        for _ in 0..10 {
            stats.accepted_tokens += 3;
            stats.correction_tokens += 1;
            stats.total_steps += 1;
        }

        assert!((stats.speedup_factor() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_output() {
        let tokens = vec![
            TokenWithProb::new(1, "a", 0.8),
            TokenWithProb::new(2, "b", 0.7),
        ];
        let output = SpeculativeOutput::new("ab", tokens);

        assert_eq!(output.text, "ab");
        assert_eq!(output.tokens.len(), 2);
        assert_eq!(output.acceptance_rate(), 0.0); // No steps yet
    }

    #[test]
    fn test_speculative_output_with_step() {
        let tokens = vec![TokenWithProb::new(1, "a", 0.8)];
        let step = SpeculativeStep::new(
            vec![
                TokenWithProb::new(1, "a", 0.8),
                TokenWithProb::new(2, "b", 0.7),
            ],
            vec![TokenWithProb::new(1, "a", 0.8)],
        );

        let output = SpeculativeOutput::new("a", tokens).with_step(step);

        assert_eq!(output.total_draft_tokens, 2);
        assert_eq!(output.total_accepted, 1);
        assert_eq!(output.acceptance_rate(), 0.5);
    }

    #[test]
    fn test_verification_result() {
        let probs = vec![vec![0.1, 0.2, 0.7], vec![0.3, 0.4, 0.3]];
        let states = ModelHiddenStates::new("model", vec![], vec![1, 2]);
        let kv_cache = ModelKVCache::new("model", 12, 64);

        let result = VerificationResult::new(probs.clone(), states, kv_cache);

        assert_eq!(result.target_probs.len(), 2);
        assert_eq!(result.hidden_states.model_id, "model");
    }

    #[tokio::test]
    async fn test_speculative_decoder_creation() {
        let draft = MockHiddenStateProvider::new(256, 6);
        let target = MockHiddenStateProvider::new(512, 12);
        let config = SpeculativeDecodingConfig::default();

        let decoder = SpeculativeDecoder::new(draft, target, config);

        assert_eq!(decoder.config().num_speculative_tokens, 4);
        assert_eq!(decoder.stats().total_steps, 0);
    }

    #[tokio::test]
    async fn test_speculative_decoder_generate() {
        let draft = MockHiddenStateProvider::new(256, 6);
        let target = MockHiddenStateProvider::new(512, 12);
        let config = SpeculativeDecodingConfig::default();

        let mut decoder = SpeculativeDecoder::new(draft, target, config);
        let output = decoder.generate("Hello", 10).await.unwrap();

        assert!(!output.tokens.is_empty());
        assert!(!output.steps.is_empty());
    }

    #[tokio::test]
    async fn test_speculative_decoder_stats() {
        let draft = MockHiddenStateProvider::new(256, 6);
        let target = MockHiddenStateProvider::new(512, 12);
        let config = SpeculativeDecodingConfig::default();

        let mut decoder = SpeculativeDecoder::new(draft, target, config);
        let _ = decoder.generate("Test prompt", 8).await;

        assert!(decoder.stats().total_steps > 0);
        assert!(decoder.stats().total_draft_tokens > 0);

        decoder.reset_stats();
        assert_eq!(decoder.stats().total_steps, 0);
    }

    #[test]
    fn test_acceptance_criterion() {
        let draft = MockHiddenStateProvider::new(256, 6);
        let target = MockHiddenStateProvider::new(512, 12);
        let config = SpeculativeDecodingConfig::default().with_acceptance_threshold(0.8);

        let decoder = SpeculativeDecoder::new(draft, target, config);

        // Target prob higher than draft -> accept
        assert!(decoder.acceptance_criterion(0.5, 0.7));

        // Target prob lower but ratio >= threshold -> accept
        assert!(decoder.acceptance_criterion(0.5, 0.45)); // 0.45/0.5 = 0.9 >= 0.8

        // Target prob much lower -> reject
        assert!(!decoder.acceptance_criterion(0.5, 0.3)); // 0.3/0.5 = 0.6 < 0.8
    }

    #[tokio::test]
    async fn test_mock_speculative_decoder() {
        let config = SpeculativeDecodingConfig::default();
        let mut decoder = MockSpeculativeDecoder::new(config);

        let output = decoder.generate("Test", 10).await.unwrap();

        assert!(!output.tokens.is_empty());
        assert!(!output.text.is_empty());
    }

    #[tokio::test]
    async fn test_mock_speculative_decoder_acceptance_rate() {
        let config = SpeculativeDecodingConfig::default();
        let mut decoder = MockSpeculativeDecoder::new(config).with_acceptance_rate(0.5);

        let output = decoder.generate("Test", 20).await.unwrap();

        // With 50% acceptance, we should see some rejections
        let steps_with_corrections = output
            .steps
            .iter()
            .filter(|s| s.target_correction.is_some())
            .count();
        assert!(steps_with_corrections > 0);
    }

    #[tokio::test]
    async fn test_mock_speculative_decoder_stats() {
        let config = SpeculativeDecodingConfig::default();
        let mut decoder = MockSpeculativeDecoder::new(config);

        let _ = decoder.generate("Test", 15).await;

        assert!(decoder.stats().total_steps > 0);
        assert!(decoder.stats().total_draft_tokens > 0);

        decoder.reset_stats();
        assert_eq!(decoder.stats().total_steps, 0);
    }

    #[test]
    fn test_mock_speculative_decoder_default() {
        let decoder = MockSpeculativeDecoder::default();
        assert_eq!(decoder.config.num_speculative_tokens, 4);
    }

    #[test]
    fn test_sample_correction() {
        let draft = MockHiddenStateProvider::new(256, 6);
        let target = MockHiddenStateProvider::new(512, 12);
        let config = SpeculativeDecodingConfig::default();

        let decoder = SpeculativeDecoder::new(draft, target, config);

        let target_probs = vec![0.1, 0.3, 0.6];
        let draft_probs = vec![0.2, 0.2, 0.6];

        let correction = decoder.sample_correction(&target_probs, &draft_probs);

        // Token 1 should have highest adjusted prob (0.3 - 0.2 = 0.1 vs 0.0 and 0.0)
        assert_eq!(correction.token_id, 1);
    }

    use crate::layer2_speculator::hidden_states::ModelHiddenStates;

    #[test]
    fn test_speculative_step_with_hidden_states() {
        let draft = vec![TokenWithProb::new(1, "a", 0.8)];
        let accepted = vec![TokenWithProb::new(1, "a", 0.8)];
        let draft_states = ModelHiddenStates::new("draft", vec![], vec![1]);
        let target_states = ModelHiddenStates::new("target", vec![], vec![1]);

        let step = SpeculativeStep::new(draft, accepted)
            .with_draft_hidden_states(draft_states)
            .with_target_hidden_states(target_states);

        assert!(step.draft_hidden_states.is_some());
        assert!(step.target_hidden_states.is_some());
    }

    #[test]
    fn test_speculative_output_with_hidden_states() {
        let tokens = vec![TokenWithProb::new(1, "a", 0.8)];
        let states = ModelHiddenStates::new("model", vec![], vec![1]);

        let output = SpeculativeOutput::new("a", tokens).with_hidden_states(states);

        assert!(output.final_hidden_states.is_some());
    }

    #[test]
    fn test_token_with_prob_zero_probability() {
        let token = TokenWithProb::new(0, "", 0.0);
        assert_eq!(token.probability, 0.0);
        assert!(token.log_prob.is_infinite() && token.log_prob.is_sign_negative());
    }

    #[test]
    fn test_speculative_stats_empty() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.cache_hit_rate(), 0.0);
        assert_eq!(stats.speedup_factor(), 1.0);
    }
}
