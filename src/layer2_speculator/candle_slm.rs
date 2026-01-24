//! Candle-based Small Language Model for speculation.

use async_trait::async_trait;

use crate::error::SpeculatorError;
use crate::layer2_speculator::traits::{Speculator, SpeculatorConfig};
use crate::types::{Draft, SearchResult, SpeculationDecision, SpeculationResult};

#[cfg(feature = "speculator")]
use crate::layer2_speculator::traits::prompts;

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
    fn to_candle_device(self) -> Device {
        match self {
            CandleSlmDevice::Cpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            CandleSlmDevice::Cuda(ordinal) => {
                Device::new_cuda(ordinal).expect("CUDA device should be available")
            }
            #[cfg(feature = "metal")]
            CandleSlmDevice::Metal => {
                Device::new_metal(0).expect("Metal device should be available")
            }
        }
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
        let device = config.device.to_candle_device();

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
}
