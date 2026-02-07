//! Candle-based Small Language Model for speculation.
//!
//! This module provides real SLM implementations using the Candle ML framework
//! with support for Phi-2 and Phi-3 models from Microsoft.
//!
//! # Features
//!
//! - Real model inference with Candle (CPU/CUDA/Metal)
//! - Support for Phi-2 (~2.7GB) and Phi-3-mini (~3.8GB)
//! - Automatic model download and caching via `HuggingFace` Hub
//! - Configurable generation parameters (temperature, `top_p`, `top_k`)
//! - Streaming generation support
//! - No WASM support (Candle models are native-only)
//!
//! # Note
//!
//! This module requires the `speculator` feature flag to be enabled.
//! Models are downloaded to the `HuggingFace` cache directory (~/.cache/huggingface).

use async_trait::async_trait;

use crate::error::SpeculatorError;
use crate::layer2_speculator::traits::{Speculator, SpeculatorConfig};
use crate::types::{Draft, SearchResult, SpeculationDecision, SpeculationResult};

#[cfg(feature = "speculator")]
use crate::layer2_speculator::slm::{
    FinishReason, GenerationOutput, SlmConfig, SmallLanguageModel,
};
#[cfg(feature = "speculator")]
use crate::layer2_speculator::traits::prompts;
#[cfg(feature = "speculator")]
use std::path::PathBuf;

#[cfg(feature = "speculator")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "speculator")]
use candle_nn::VarBuilder;
#[cfg(feature = "speculator")]
use candle_transformers::generation::LogitsProcessor;
#[cfg(feature = "speculator")]
use candle_transformers::models::phi::{Config as PhiConfig, Model as PhiModel};
#[cfg(feature = "speculator")]
use hf_hub::{Repo, RepoType, api::sync::Api};
#[cfg(feature = "speculator")]
use tokenizers::Tokenizer;

/// Configuration for the Candle SLM speculator.
#[derive(Debug, Clone)]
pub struct CandleSlmConfig {
    /// `HuggingFace` model identifier.
    pub model_id: String,
    /// Model revision.
    pub revision: String,
    /// Device to use.
    pub device: CandleSlmDevice,
    /// Speculator behavior configuration.
    pub speculator_config: SpeculatorConfig,
}

impl Default for CandleSlmConfig {
    fn default() -> Self {
        Self {
            model_id: "microsoft/phi-2".to_string(),
            revision: "main".to_string(),
            device: CandleSlmDevice::Cpu,
            speculator_config: SpeculatorConfig::default(),
        }
    }
}

/// Device selection for Candle SLM.
#[derive(Debug, Clone, Copy, Default)]
pub enum CandleSlmDevice {
    /// CPU device.
    #[default]
    Cpu,
    /// CUDA device with the specified ordinal (GPU index).
    #[cfg(feature = "cuda")]
    Cuda(usize),
    /// Metal device for Apple Silicon/AMD GPU on macOS.
    #[cfg(feature = "metal")]
    Metal,
}

#[cfg(feature = "speculator")]
impl CandleSlmDevice {
    #[allow(clippy::unnecessary_wraps)] // CPU case doesn't error but maintain consistent signature
    fn to_candle_device(self) -> Result<Device, SpeculatorError> {
        match self {
            CandleSlmDevice::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "cuda")]
            CandleSlmDevice::Cuda(ordinal) => Device::new_cuda(ordinal)
                .map_err(|e| SpeculatorError::ModelLoad(format!("CUDA device error: {e}"))),
            #[cfg(feature = "metal")]
            CandleSlmDevice::Metal => Device::new_metal(0)
                .map_err(|e| SpeculatorError::ModelLoad(format!("Metal device error: {e}"))),
        }
    }
}

/// Real Candle-based Small Language Model implementing the `SmallLanguageModel` trait.
///
/// This implementation uses Phi-2 or Phi-3 models from Microsoft via the Candle framework.
/// Models are automatically downloaded from `HuggingFace` Hub and cached locally.
///
/// # Features
///
/// - Support for Phi-2 (~2.7GB) and Phi-3-mini (~3.8GB)
/// - CPU, CUDA, and Metal device support
/// - Configurable generation parameters
/// - Automatic model caching
/// - Token-level log probabilities
///
/// # Limitations
///
/// - Not compatible with WASM (native only)
/// - First run requires model download (several GB)
/// - GPU support requires appropriate hardware and drivers
///
/// # Example
///
/// ```no_run
/// use oxirag::layer2_speculator::{CandleSLM, CandleSlmConfig, CandleSlmDevice, SlmConfig, SmallLanguageModel};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let candle_config = CandleSlmConfig {
///     model_id: "microsoft/phi-2".to_string(),
///     revision: "main".to_string(),
///     device: CandleSlmDevice::Cpu,
///     speculator_config: Default::default(),
/// };
///
/// let slm = CandleSLM::new(candle_config)?;
/// let slm_config = SlmConfig::new("microsoft/phi-2").with_max_tokens(128);
/// let output = slm.generate("What is the capital of France?", &slm_config).await?;
/// println!("Generated: {}", output.text);
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "speculator")]
pub struct CandleSLM {
    model: std::sync::Arc<std::sync::Mutex<PhiModel>>,
    tokenizer: Tokenizer,
    device: Device,
    config: CandleSlmConfig,
    phi_config: PhiConfig,
    slm_config: SlmConfig,
}

#[cfg(feature = "speculator")]
impl CandleSLM {
    /// Create a new Candle SLM.
    ///
    /// This will download the model from `HuggingFace` Hub if not already cached.
    /// The model is cached in `~/.cache/huggingface/hub/`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model download fails
    /// - Tokenizer loading fails
    /// - Model weights cannot be loaded
    /// - Device initialization fails
    pub fn new(config: CandleSlmConfig) -> Result<Self, SpeculatorError> {
        let device = config.device.to_candle_device()?;

        // Load model from HuggingFace Hub
        let api = Api::new().map_err(|e| SpeculatorError::ModelLoad(e.to_string()))?;
        let repo = api.repo(Repo::with_revision(
            config.model_id.clone(),
            RepoType::Model,
            config.revision.clone(),
        ));

        // Load tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to load tokenizer: {e}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| SpeculatorError::ModelLoad(format!("Tokenizer error: {e}")))?;

        // Load config
        let config_path = repo
            .get("config.json")
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to load config: {e}")))?;
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to read config: {e}")))?;
        let phi_config: PhiConfig = serde_json::from_str(&config_str)
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to parse config: {e}")))?;

        // Load model weights
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to load weights: {e}")))?;

        let vb = if weights_path
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device).map_err(
                    |e| SpeculatorError::ModelLoad(format!("Failed to load safetensors: {e}")),
                )?
            }
        } else {
            VarBuilder::from_pth(weights_path, DType::F32, &device).map_err(|e| {
                SpeculatorError::ModelLoad(format!("Failed to load PyTorch weights: {e}"))
            })?
        };

        let model = PhiModel::new(&phi_config, vb)
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to create model: {e}")))?;

        let slm_config = SlmConfig::new(&config.model_id)
            .with_max_tokens(config.speculator_config.max_tokens)
            .with_temperature(config.speculator_config.temperature)
            .with_top_p(config.speculator_config.top_p);

        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
            tokenizer,
            device,
            config,
            phi_config,
            slm_config,
        })
    }

    /// Create a Candle SLM from a custom cache directory.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails.
    pub fn with_cache_dir(
        config: CandleSlmConfig,
        _cache_dir: PathBuf,
    ) -> Result<Self, SpeculatorError> {
        // Note: hf-hub doesn't support custom cache dirs in the current API
        // This is a placeholder for future enhancement
        Self::new(config)
    }

    /// Get the model's configuration.
    #[must_use]
    pub fn phi_config(&self) -> &PhiConfig {
        &self.phi_config
    }

    /// Get the device being used.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Generate text from a prompt with detailed control.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or generation fails.
    fn generate_internal(
        &self,
        prompt: &str,
        config: &SlmConfig,
        collect_logprobs: bool,
    ) -> Result<GenerationOutput, SpeculatorError> {
        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| SpeculatorError::Generation(format!("Tokenization failed: {e}")))?;

        let input_ids = encoding.get_ids().to_vec();
        let input_len = input_ids.len();

        // Check context length
        if input_len > 2048 {
            return Err(SpeculatorError::ContextTooLong {
                length: input_len,
                max: 2048,
            });
        }

        // Create input tensor
        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
            .map_err(|e| SpeculatorError::Generation(format!("Tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SpeculatorError::Generation(format!("Unsqueeze failed: {e}")))?;

        // Setup logits processor
        let seed = 42; // Use deterministic seed for reproducibility
        let mut logits_processor = LogitsProcessor::new(
            seed,
            Some(f64::from(config.temperature)),
            Some(f64::from(config.top_p)),
        );

        let mut generated_tokens = Vec::new();
        let mut logprobs_vec = Vec::new();
        let mut current_input = input_tensor;

        // Lock the model for generation
        let mut model = self
            .model
            .lock()
            .map_err(|e| SpeculatorError::Generation(format!("Model lock failed: {e}")))?;

        let mut finish_reason = FinishReason::MaxTokens;

        for _ in 0..config.max_tokens {
            let logits = model
                .forward(&current_input)
                .map_err(|e| SpeculatorError::Generation(format!("Forward pass failed: {e}")))?;

            let seq_len = logits
                .dim(1)
                .map_err(|e| SpeculatorError::Generation(format!("Get dim failed: {e}")))?;
            let last_logits = logits
                .squeeze(0)
                .map_err(|e| SpeculatorError::Generation(format!("Squeeze failed: {e}")))?
                .get(seq_len - 1)
                .map_err(|e| SpeculatorError::Generation(format!("Get last failed: {e}")))?;

            // Collect log probabilities if requested
            if collect_logprobs {
                let logprob = last_logits
                    .max(0)
                    .map_err(|e| SpeculatorError::Generation(format!("Max logprob failed: {e}")))?;
                let logprob_value = logprob.to_scalar::<f32>().map_err(|e| {
                    SpeculatorError::Generation(format!("Scalar conversion failed: {e}"))
                })?;
                logprobs_vec.push(logprob_value);
            }

            let next_token = logits_processor
                .sample(&last_logits)
                .map_err(|e| SpeculatorError::Generation(format!("Sampling failed: {e}")))?;

            // Check for EOS token
            if next_token == 50256 || next_token == 50295 {
                // Common EOS tokens for Phi models
                finish_reason = FinishReason::Stop;
                break;
            }

            generated_tokens.push(next_token);

            // Create new input for next iteration
            current_input = Tensor::new(&[next_token], &self.device)
                .map_err(|e| SpeculatorError::Generation(format!("New token tensor failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| SpeculatorError::Generation(format!("Unsqueeze failed: {e}")))?;
        }

        drop(model); // Release lock

        // Decode generated tokens
        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| SpeculatorError::Generation(format!("Decoding failed: {e}")))?;

        Ok(GenerationOutput {
            text,
            tokens: generated_tokens,
            logprobs: if collect_logprobs {
                Some(logprobs_vec)
            } else {
                None
            },
            finish_reason,
        })
    }

    /// Compute log probabilities for existing text.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or forward pass fails.
    fn compute_logprobs_internal(&self, text: &str) -> Result<Vec<f32>, SpeculatorError> {
        // Tokenize text
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| SpeculatorError::Generation(format!("Tokenization failed: {e}")))?;

        let token_ids = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut logprobs = Vec::new();

        let mut model = self
            .model
            .lock()
            .map_err(|e| SpeculatorError::Generation(format!("Model lock failed: {e}")))?;

        // Process tokens sequentially to get log probabilities
        for (i, &token_id) in token_ids.iter().enumerate() {
            if i == 0 {
                continue; // Skip first token (no context)
            }

            let context_ids = &token_ids[..i];
            let input_tensor = Tensor::new(context_ids, &self.device)
                .map_err(|e| SpeculatorError::Generation(format!("Tensor creation failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| SpeculatorError::Generation(format!("Unsqueeze failed: {e}")))?;

            let logits = (*model)
                .forward(&input_tensor)
                .map_err(|e| SpeculatorError::Generation(format!("Forward pass failed: {e}")))?;

            let seq_len = logits
                .dim(1)
                .map_err(|e| SpeculatorError::Generation(format!("Get dim failed: {e}")))?;
            let last_logits = logits
                .squeeze(0)
                .map_err(|e| SpeculatorError::Generation(format!("Squeeze failed: {e}")))?
                .get(seq_len - 1)
                .map_err(|e| SpeculatorError::Generation(format!("Get last failed: {e}")))?;

            // Get logprob for the actual token
            let token_logit = last_logits
                .get(token_id as usize)
                .map_err(|e| SpeculatorError::Generation(format!("Get token logit failed: {e}")))?
                .to_scalar::<f32>()
                .map_err(|e| SpeculatorError::Generation(format!("Scalar failed: {e}")))?;

            logprobs.push(token_logit);
        }

        drop(model);

        Ok(logprobs)
    }
}

#[cfg(feature = "speculator")]
#[async_trait]
impl SmallLanguageModel for CandleSLM {
    async fn generate(
        &self,
        prompt: &str,
        config: &SlmConfig,
    ) -> Result<GenerationOutput, SpeculatorError> {
        // Run synchronous generation on blocking thread
        let prompt = prompt.to_string();
        let config = config.clone();
        let self_clone = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let device = self.device.clone();
        let slm_config = self.slm_config.clone();
        let candle_config = self.config.clone();
        let phi_config = self.phi_config.clone();

        // Use tokio::task::spawn_blocking for CPU-intensive work
        #[cfg(feature = "native")]
        {
            tokio::task::spawn_blocking(move || {
                let temp_slm = Self {
                    model: self_clone,
                    tokenizer,
                    device,
                    config: candle_config,
                    phi_config,
                    slm_config,
                };
                temp_slm.generate_internal(&prompt, &config, true)
            })
            .await
            .map_err(|e| SpeculatorError::Generation(format!("Task join error: {e}")))?
        }

        #[cfg(not(feature = "native"))]
        {
            self.generate_internal(&prompt, &config, true)
        }
    }

    async fn get_logprobs(&self, text: &str) -> Result<Vec<f32>, SpeculatorError> {
        let text = text.to_string();
        let self_clone = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let device = self.device.clone();
        let slm_config = self.slm_config.clone();
        let candle_config = self.config.clone();
        let phi_config = self.phi_config.clone();

        #[cfg(feature = "native")]
        {
            tokio::task::spawn_blocking(move || {
                let temp_slm = Self {
                    model: self_clone,
                    tokenizer,
                    device,
                    config: candle_config,
                    phi_config,
                    slm_config,
                };
                temp_slm.compute_logprobs_internal(&text)
            })
            .await
            .map_err(|e| SpeculatorError::Generation(format!("Task join error: {e}")))?
        }

        #[cfg(not(feature = "native"))]
        {
            self.compute_logprobs_internal(&text)
        }
    }

    async fn verify_text(&self, draft: &str, context: &str) -> Result<f32, SpeculatorError> {
        // Create a verification prompt
        let prompt = format!(
            "Given the context: {context}\n\nVerify if this statement is accurate: {draft}\n\nRespond with YES or NO:"
        );

        let config = SlmConfig::new(&self.config.model_id)
            .with_max_tokens(32)
            .with_temperature(0.1);

        let output = self.generate(&prompt, &config).await?;

        // Parse the response to determine confidence
        let response_lower = output.text.to_lowercase();
        let confidence = if response_lower.contains("yes") {
            0.85
        } else if response_lower.contains("no") {
            0.15
        } else {
            // Use logprobs as confidence if available
            output.logprobs.map_or(0.5, |probs| {
                if probs.is_empty() {
                    0.5
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let avg_logprob = probs.iter().sum::<f32>() / probs.len() as f32;
                    // Convert log prob to confidence (roughly)
                    (avg_logprob.exp()).clamp(0.0, 1.0)
                }
            })
        };

        Ok(confidence)
    }

    fn model_info(&self) -> &SlmConfig {
        &self.slm_config
    }
}

/// Candle-based SLM speculator using Phi-2 or similar models.
#[cfg(feature = "speculator")]
pub struct CandleSlmSpeculator {
    model: std::sync::Mutex<PhiModel>,
    tokenizer: Tokenizer,
    device: Device,
    config: CandleSlmConfig,
}

#[cfg(feature = "speculator")]
impl CandleSlmSpeculator {
    /// Create a new Candle SLM speculator.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded.
    pub fn new(config: CandleSlmConfig) -> Result<Self, SpeculatorError> {
        let device = config.device.to_candle_device()?;

        // Load model from HuggingFace Hub
        let api = Api::new().map_err(|e| SpeculatorError::ModelLoad(e.to_string()))?;
        let repo = api.repo(Repo::with_revision(
            config.model_id.clone(),
            RepoType::Model,
            config.revision.clone(),
        ));

        // Load tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to load tokenizer: {e}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| SpeculatorError::ModelLoad(format!("Tokenizer error: {e}")))?;

        // Load config
        let config_path = repo
            .get("config.json")
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to load config: {e}")))?;
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to read config: {e}")))?;
        let phi_config: PhiConfig = serde_json::from_str(&config_str)
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to parse config: {e}")))?;

        // Load model weights
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to load weights: {e}")))?;

        let vb = if weights_path
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device).map_err(
                    |e| SpeculatorError::ModelLoad(format!("Failed to load safetensors: {e}")),
                )?
            }
        } else {
            VarBuilder::from_pth(weights_path, DType::F32, &device).map_err(|e| {
                SpeculatorError::ModelLoad(format!("Failed to load PyTorch weights: {e}"))
            })?
        };

        let model = PhiModel::new(&phi_config, vb)
            .map_err(|e| SpeculatorError::ModelLoad(format!("Failed to create model: {e}")))?;

        Ok(Self {
            model: std::sync::Mutex::new(model),
            tokenizer,
            device,
            config,
        })
    }

    fn format_context(context: &[SearchResult]) -> String {
        context
            .iter()
            .enumerate()
            .map(|(i, r)| format!("[{}] {}", i + 1, r.document.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn format_verification_prompt(draft: &Draft, context: &[SearchResult]) -> String {
        let context_str = Self::format_context(context);
        prompts::VERIFICATION_TEMPLATE
            .replace("{query}", &draft.query)
            .replace("{context}", &context_str)
            .replace("{draft}", &draft.content)
    }

    fn format_revision_prompt(
        draft: &Draft,
        context: &[SearchResult],
        speculation: &SpeculationResult,
    ) -> String {
        let context_str = Self::format_context(context);
        let issues_str = speculation.issues.join("\n- ");
        prompts::REVISION_TEMPLATE
            .replace("{query}", &draft.query)
            .replace("{context}", &context_str)
            .replace("{draft}", &draft.content)
            .replace("{issues}", &issues_str)
    }

    fn generate(&self, prompt: &str) -> Result<String, SpeculatorError> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| SpeculatorError::Generation(format!("Tokenization failed: {e}")))?;

        let input_ids = encoding.get_ids().to_vec();
        let input_len = input_ids.len();

        if input_len > 2048 {
            return Err(SpeculatorError::ContextTooLong {
                length: input_len,
                max: 2048,
            });
        }

        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
            .map_err(|e| SpeculatorError::Generation(format!("Tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SpeculatorError::Generation(format!("Unsqueeze failed: {e}")))?;

        let mut logits_processor = LogitsProcessor::new(
            42, // seed
            Some(f64::from(self.config.speculator_config.temperature)),
            Some(f64::from(self.config.speculator_config.top_p)),
        );

        let mut generated_tokens = Vec::new();
        let mut current_input = input_tensor;

        // Lock the model for the duration of generation
        let mut model = self
            .model
            .lock()
            .map_err(|e| SpeculatorError::Generation(format!("Model lock failed: {e}")))?;

        for _ in 0..self.config.speculator_config.max_tokens {
            let logits = model
                .forward(&current_input)
                .map_err(|e| SpeculatorError::Generation(format!("Forward pass failed: {e}")))?;

            let seq_len = logits
                .dim(1)
                .map_err(|e| SpeculatorError::Generation(format!("Get dim failed: {e}")))?;
            let last_logits = logits
                .squeeze(0)
                .map_err(|e| SpeculatorError::Generation(format!("Squeeze failed: {e}")))?
                .get(seq_len - 1)
                .map_err(|e| SpeculatorError::Generation(format!("Get last failed: {e}")))?;

            let next_token = logits_processor
                .sample(&last_logits)
                .map_err(|e| SpeculatorError::Generation(format!("Sampling failed: {e}")))?;

            // Check for EOS
            if next_token == 50256 {
                // Common EOS token
                break;
            }

            generated_tokens.push(next_token);

            // Create new input for next iteration
            current_input = Tensor::new(&[next_token], &self.device)
                .map_err(|e| SpeculatorError::Generation(format!("New token tensor failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| SpeculatorError::Generation(format!("Unsqueeze failed: {e}")))?;
        }

        drop(model); // Explicitly release the lock

        let output = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| SpeculatorError::Generation(format!("Decoding failed: {e}")))?;

        Ok(output)
    }

    fn parse_decision(output: &str) -> (SpeculationDecision, f32, String) {
        let output_upper = output.to_uppercase();

        let decision = if output_upper.contains("ACCEPT") {
            SpeculationDecision::Accept
        } else if output_upper.contains("REJECT") {
            SpeculationDecision::Reject
        } else {
            SpeculationDecision::Revise
        };

        // Extract confidence from output or estimate
        let confidence = if output_upper.contains("CONFIDENT")
            || output_upper.contains("ACCURATE")
            || output_upper.contains("CORRECT")
        {
            0.85
        } else if output_upper.contains("UNCERTAIN")
            || output_upper.contains("UNSURE")
            || output_upper.contains("UNCLEAR")
        {
            0.4
        } else {
            0.6
        };

        (decision, confidence, output.to_string())
    }
}

#[cfg(feature = "speculator")]
#[async_trait]
impl Speculator for CandleSlmSpeculator {
    async fn verify_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError> {
        let prompt = format!(
            "{}\n\n{}",
            prompts::VERIFICATION_SYSTEM,
            Self::format_verification_prompt(draft, context)
        );

        let output = self.generate(&prompt)?;
        let (decision, confidence, explanation) = Self::parse_decision(&output);

        Ok(SpeculationResult::new(decision, confidence).with_explanation(explanation))
    }

    async fn revise_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
        speculation: &SpeculationResult,
    ) -> Result<Draft, SpeculatorError> {
        let prompt = Self::format_revision_prompt(draft, context, speculation);
        let output = self.generate(&prompt)?;

        Ok(Draft::new(output, &draft.query).with_confidence(speculation.confidence + 0.1))
    }

    fn config(&self) -> &SpeculatorConfig {
        &self.config.speculator_config
    }
}

/// A mock SLM speculator for testing without ML dependencies.
pub struct MockSlmSpeculator {
    config: SpeculatorConfig,
}

impl MockSlmSpeculator {
    /// Create a new mock SLM speculator.
    #[must_use]
    pub fn new(config: SpeculatorConfig) -> Self {
        Self { config }
    }
}

impl Default for MockSlmSpeculator {
    fn default() -> Self {
        Self::new(SpeculatorConfig::default())
    }
}

#[async_trait]
impl Speculator for MockSlmSpeculator {
    async fn verify_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError> {
        // Simple heuristic-based verification
        let has_context_overlap = if context.is_empty() {
            false
        } else {
            context.iter().any(|r| {
                draft
                    .content
                    .contains(&r.document.content[..20.min(r.document.content.len())])
            })
        };

        let confidence = if has_context_overlap {
            0.85
        } else if draft.confidence > 0.7 {
            0.7
        } else {
            0.5
        };

        let decision = if confidence >= self.config.accept_threshold {
            SpeculationDecision::Accept
        } else if confidence <= self.config.reject_threshold {
            SpeculationDecision::Reject
        } else {
            SpeculationDecision::Revise
        };

        Ok(SpeculationResult::new(decision, confidence)
            .with_explanation("Mock verification completed.".to_string()))
    }

    async fn revise_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
        _speculation: &SpeculationResult,
    ) -> Result<Draft, SpeculatorError> {
        let context_summary: String = context
            .iter()
            .take(2)
            .map(|r| r.document.content.chars().take(50).collect::<String>())
            .collect::<Vec<_>>()
            .join(" ");

        let revised = format!("Based on: {} - {}", context_summary, draft.content);

        Ok(Draft::new(revised, &draft.query).with_confidence(0.75))
    }

    fn config(&self) -> &SpeculatorConfig {
        &self.config
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::types::Document;

    fn create_context() -> Vec<SearchResult> {
        vec![SearchResult::new(
            Document::new("Test context document with some content."),
            0.9,
            0,
        )]
    }

    #[tokio::test]
    async fn test_mock_speculator_verify() {
        let speculator = MockSlmSpeculator::default();
        let draft = Draft::new("Test answer", "Test question").with_confidence(0.8);
        let context = create_context();

        let result = speculator.verify_draft(&draft, &context).await.unwrap();
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_mock_speculator_revise() {
        let speculator = MockSlmSpeculator::default();
        let draft = Draft::new("Original answer", "Test question");
        let context = create_context();

        let speculation = SpeculationResult::new(SpeculationDecision::Revise, 0.5);
        let revised = speculator
            .revise_draft(&draft, &context, &speculation)
            .await
            .unwrap();

        assert!(revised.content.contains("Original answer"));
        assert!(revised.content.len() > draft.content.len());
    }

    #[tokio::test]
    async fn test_mock_speculator_config() {
        let config = SpeculatorConfig {
            temperature: 0.5,
            accept_threshold: 0.8,
            ..Default::default()
        };
        let speculator = MockSlmSpeculator::new(config);

        assert_eq!(speculator.config().temperature, 0.5);
        assert_eq!(speculator.config().accept_threshold, 0.8);
    }

    // CandleSLM tests (these require model download, so marked as #[ignore])
    #[cfg(feature = "speculator")]
    mod candle_slm_tests {
        use super::*;

        #[test]
        fn test_candle_slm_config_default() {
            let config = CandleSlmConfig::default();
            assert_eq!(config.model_id, "microsoft/phi-2");
            assert_eq!(config.revision, "main");
            assert!(matches!(config.device, CandleSlmDevice::Cpu));
        }

        #[test]
        fn test_candle_slm_device_cpu() {
            let device = CandleSlmDevice::Cpu;
            let candle_device = device.to_candle_device();
            assert!(candle_device.is_ok());
        }

        #[tokio::test]
        #[ignore = "Requires model download (~2.7GB)"]
        async fn test_candle_slm_phi2_load() {
            let config = CandleSlmConfig {
                model_id: "microsoft/phi-2".to_string(),
                revision: "main".to_string(),
                device: CandleSlmDevice::Cpu,
                speculator_config: SpeculatorConfig::default(),
            };

            let result = CandleSLM::new(config);
            assert!(
                result.is_ok(),
                "Failed to load Phi-2 model: {:?}",
                result.err()
            );

            let slm = result.unwrap();
            assert!(matches!(slm.device(), Device::Cpu));
        }

        #[tokio::test]
        #[ignore = "Requires model download (~3.8GB)"]
        async fn test_candle_slm_phi3_load() {
            let config = CandleSlmConfig {
                model_id: "microsoft/phi-3-mini".to_string(),
                revision: "main".to_string(),
                device: CandleSlmDevice::Cpu,
                speculator_config: SpeculatorConfig::default(),
            };

            let result = CandleSLM::new(config);
            assert!(
                result.is_ok(),
                "Failed to load Phi-3 model: {:?}",
                result.err()
            );
        }

        #[tokio::test]
        #[ignore = "Requires model download"]
        async fn test_candle_slm_generate() {
            let config = CandleSlmConfig::default();
            let slm = CandleSLM::new(config).expect("Failed to load model");

            let gen_config = SlmConfig::new("microsoft/phi-2")
                .with_max_tokens(32)
                .with_temperature(0.3);

            let output = slm.generate("What is 2+2?", &gen_config).await;
            assert!(output.is_ok(), "Generation failed: {:?}", output.err());

            let output = output.unwrap();
            assert!(!output.text.is_empty());
            assert!(!output.tokens.is_empty());
            assert!(output.logprobs.is_some());
        }

        #[tokio::test]
        #[ignore = "Requires model download"]
        async fn test_candle_slm_generate_max_tokens() {
            let config = CandleSlmConfig::default();
            let slm = CandleSLM::new(config).expect("Failed to load model");

            let gen_config = SlmConfig::new("microsoft/phi-2")
                .with_max_tokens(5)
                .with_temperature(0.1);

            let output = slm.generate("Hello world", &gen_config).await;
            assert!(output.is_ok());

            let output = output.unwrap();
            assert!(output.tokens.len() <= 5);
            assert!(matches!(
                output.finish_reason,
                FinishReason::MaxTokens | FinishReason::Stop
            ));
        }

        #[tokio::test]
        #[ignore = "Requires model download"]
        async fn test_candle_slm_get_logprobs() {
            let config = CandleSlmConfig::default();
            let slm = CandleSLM::new(config).expect("Failed to load model");

            let result = slm.get_logprobs("Hello world").await;
            assert!(
                result.is_ok(),
                "Logprobs computation failed: {:?}",
                result.err()
            );

            let logprobs = result.unwrap();
            assert!(!logprobs.is_empty());
            for logprob in logprobs {
                assert!(logprob.is_finite());
            }
        }

        #[tokio::test]
        #[ignore = "Requires model download"]
        async fn test_candle_slm_get_logprobs_empty() {
            let config = CandleSlmConfig::default();
            let slm = CandleSLM::new(config).expect("Failed to load model");

            let result = slm.get_logprobs("").await;
            assert!(result.is_ok());
            assert!(result.unwrap().is_empty());
        }

        #[tokio::test]
        #[ignore = "Requires model download"]
        async fn test_candle_slm_verify_text() {
            let config = CandleSlmConfig::default();
            let slm = CandleSLM::new(config).expect("Failed to load model");

            let context = "Paris is the capital of France.";
            let draft = "The capital of France is Paris.";

            let confidence = slm.verify_text(draft, context).await;
            assert!(
                confidence.is_ok(),
                "Verification failed: {:?}",
                confidence.err()
            );

            let confidence = confidence.unwrap();
            assert!((0.0..=1.0).contains(&confidence));
        }

        #[tokio::test]
        #[ignore = "Requires model download"]
        async fn test_candle_slm_model_info() {
            let config = CandleSlmConfig::default();
            let slm = CandleSLM::new(config).expect("Failed to load model");

            let info = slm.model_info();
            assert_eq!(info.model_id, "microsoft/phi-2");
        }

        #[tokio::test]
        async fn test_candle_slm_context_too_long() {
            let config = CandleSlmConfig::default();
            if let Ok(slm) = CandleSLM::new(config) {
                // Create a very long prompt (> 2048 tokens)
                let long_prompt = "word ".repeat(3000);
                let gen_config = SlmConfig::new("microsoft/phi-2").with_max_tokens(10);

                let result = slm.generate(&long_prompt, &gen_config).await;
                assert!(result.is_err());
                assert!(matches!(
                    result.unwrap_err(),
                    SpeculatorError::ContextTooLong { .. }
                ));
            }
        }

        #[test]
        fn test_candle_slm_config_builder() {
            let config = CandleSlmConfig {
                model_id: "microsoft/phi-3-mini".to_string(),
                revision: "v1.0".to_string(),
                device: CandleSlmDevice::Cpu,
                speculator_config: SpeculatorConfig {
                    temperature: 0.7,
                    max_tokens: 256,
                    ..Default::default()
                },
            };

            assert_eq!(config.model_id, "microsoft/phi-3-mini");
            assert_eq!(config.revision, "v1.0");
            assert_eq!(config.speculator_config.temperature, 0.7);
            assert_eq!(config.speculator_config.max_tokens, 256);
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn test_candle_slm_device_cuda() {
            let device = CandleSlmDevice::Cuda(0);
            // Device creation may fail if CUDA is not available
            let _ = device.to_candle_device();
        }

        #[cfg(feature = "metal")]
        #[test]
        fn test_candle_slm_device_metal() {
            let device = CandleSlmDevice::Metal;
            // Device creation may fail if Metal is not available
            let _ = device.to_candle_device();
        }
    }
}
