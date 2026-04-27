//! Candle-based hidden state provider using BERT models.
//!
//! This module provides a real Candle/HuggingFace integration for extracting
//! hidden states from BERT-class transformer models. The output hidden states
//! tensor from the model's final encoder layer is packaged into the
//! `ModelHiddenStates` abstraction used throughout `OxiRAG`.
//!
//! # Gate
//!
//! This entire module is compiled only when **both** `hidden-states` and
//! `speculator` features are enabled, because it depends on `candle_core`,
//! `candle_transformers`, `hf_hub`, and `tokenizers` — all pulled in by the
//! `speculator` feature.

#![cfg(all(feature = "hidden-states", feature = "speculator"))]

use async_trait::async_trait;

use candle_core::{DType as CandleDType, Device as CandleCoreDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

use super::traits::HiddenStateProvider;
use super::types::{
    HiddenStateConfig, HiddenStateTensor, LayerHiddenState, ModelHiddenStates, ModelKVCache,
    TensorShape,
};
use crate::error::HiddenStateError;

// ────────────────────────────────────────────────────────────────────────────
// Device abstraction
// ────────────────────────────────────────────────────────────────────────────

/// Device selection for the Candle hidden-state provider.
///
/// Mirrors the `CandleDevice` in `layer1_echo::embedding::candle` so that
/// callers who already handle that type can convert straightforwardly.
#[derive(Debug, Clone, Copy, Default)]
pub enum CandleDevice {
    /// CPU (always available).
    #[default]
    Cpu,
    /// CUDA GPU by device ordinal.  Only meaningful when the `cuda` feature is
    /// enabled; on all other platforms the build will succeed but the variant
    /// is unreachable at runtime.
    #[cfg(feature = "cuda")]
    Cuda(usize),
    /// Apple Metal (macOS only).
    #[cfg(feature = "metal")]
    Metal,
}

impl CandleDevice {
    /// Convert to the underlying `candle_core::Device`.
    fn to_candle_device(self) -> Result<CandleCoreDevice, HiddenStateError> {
        match self {
            CandleDevice::Cpu => Ok(CandleCoreDevice::Cpu),
            #[cfg(feature = "cuda")]
            CandleDevice::Cuda(ordinal) => {
                CandleCoreDevice::new_cuda(ordinal).map_err(|e| {
                    HiddenStateError::ProviderError(format!(
                        "Failed to open CUDA device {ordinal}: {e}"
                    ))
                })
            }
            #[cfg(feature = "metal")]
            CandleDevice::Metal => CandleCoreDevice::new_metal(0).map_err(|e| {
                HiddenStateError::ProviderError(format!("Failed to open Metal device: {e}"))
            }),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for constructing a [`CandleHiddenStateProvider`].
#[derive(Debug, Clone)]
pub struct CandleHiddenStateConfig {
    /// `HuggingFace` Hub model identifier (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`).
    pub model_id: String,
    /// Git revision to fetch (branch, tag, or commit SHA).  Use `"main"` for
    /// the default branch.
    pub revision: String,
    /// Compute device.
    pub device: CandleDevice,
    /// Whether the provider should capture attention weights when building
    /// `LayerHiddenState`.  BERT's public API does not return attention weights
    /// from a plain `forward` call, so this field is currently advisory only;
    /// when `true` the provider notes the request in the config but no
    /// attention tensors are stored (the field is kept for forward-compat).
    pub capture_attention_weights: bool,
    /// Maximum sequence length fed to the tokenizer / model.  Tokens beyond
    /// this limit are truncated silently.
    pub max_sequence_length: usize,
}

impl Default for CandleHiddenStateConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            revision: "main".to_string(),
            device: CandleDevice::Cpu,
            capture_attention_weights: false,
            max_sequence_length: 512,
        }
    }
}

impl CandleHiddenStateConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new(model_id: impl Into<String>, revision: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            revision: revision.into(),
            ..Default::default()
        }
    }

    /// Override the compute device.
    #[must_use]
    pub fn with_device(mut self, device: CandleDevice) -> Self {
        self.device = device;
        self
    }

    /// Enable attention weight capture (advisory — see struct docs).
    #[must_use]
    pub fn with_capture_attention_weights(mut self, capture: bool) -> Self {
        self.capture_attention_weights = capture;
        self
    }

    /// Set the maximum sequence length.
    #[must_use]
    pub fn with_max_sequence_length(mut self, len: usize) -> Self {
        self.max_sequence_length = len;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Provider struct
// ────────────────────────────────────────────────────────────────────────────

/// Candle-based hidden-state provider backed by a BERT-class encoder model.
///
/// On each call to [`extract_hidden_states`] the provider:
/// 1. Tokenises the input text.
/// 2. Builds `input_ids`, `token_type_ids`, and `position_ids` tensors on the
///    configured device.
/// 3. Runs a forward pass through the loaded `BertModel`.
/// 4. Extracts the resulting `[1, seq_len, hidden_dim]` f32 tensor.
/// 5. Packages the data into a single-layer `ModelHiddenStates`.
///
/// Only one layer of hidden states (the final encoder output) is returned.
/// This is a real Candle inference pass — not a mock — even though `num_layers`
/// reports `1`.
///
/// [`extract_hidden_states`]: CandleHiddenStateProvider::extract_hidden_states
pub struct CandleHiddenStateProvider {
    /// The loaded BERT model.
    model: BertModel,
    /// `HuggingFace` fast tokenizer.
    tokenizer: Tokenizer,
    /// The Candle device that owns the model tensors.
    device: CandleCoreDevice,
    /// Provider-level configuration surfaced through the trait.
    hidden_state_config: HiddenStateConfig,
    /// Original config used during construction.
    candle_config: CandleHiddenStateConfig,
    /// The model's hidden dimension (read from `bert_config.hidden_size`).
    hidden_dim: usize,
}

impl CandleHiddenStateProvider {
    /// Construct a new provider by downloading (or using a cached copy of) the
    /// model from `HuggingFace` Hub.
    ///
    /// # Errors
    ///
    /// Returns [`HiddenStateError::ProviderError`] if:
    /// - the `HuggingFace` API cannot be initialised,
    /// - the tokenizer, config, or weight files cannot be fetched,
    /// - weight loading or model construction fails,
    /// - or the requested device cannot be opened.
    pub fn new(candle_config: CandleHiddenStateConfig) -> Result<Self, HiddenStateError> {
        let device = candle_config.device.to_candle_device()?;

        // Initialise HuggingFace Hub client.
        let api = Api::new().map_err(|e| {
            HiddenStateError::ProviderError(format!("Failed to initialise HF Hub API: {e}"))
        })?;

        let repo = api.repo(Repo::with_revision(
            candle_config.model_id.clone(),
            RepoType::Model,
            candle_config.revision.clone(),
        ));

        // ── Tokenizer ────────────────────────────────────────────────────────
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            HiddenStateError::ProviderError(format!("Failed to fetch tokenizer.json: {e}"))
        })?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            HiddenStateError::ProviderError(format!("Failed to parse tokenizer: {e}"))
        })?;

        // ── BERT config ──────────────────────────────────────────────────────
        let config_path = repo.get("config.json").map_err(|e| {
            HiddenStateError::ProviderError(format!("Failed to fetch config.json: {e}"))
        })?;
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            HiddenStateError::ProviderError(format!("Failed to read config.json: {e}"))
        })?;
        let bert_config: BertConfig = serde_json::from_str(&config_str).map_err(|e| {
            HiddenStateError::ProviderError(format!("Failed to deserialise config.json: {e}"))
        })?;

        let hidden_dim = bert_config.hidden_size;

        // ── Model weights ────────────────────────────────────────────────────
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| {
                HiddenStateError::ProviderError(format!("Failed to fetch model weights: {e}"))
            })?;

        let vb = if weights_path
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            // SAFETY: mmap is safe here because the file is a read-only
            // safetensors blob whose lifetime is tied to the VarBuilder.
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_path],
                    CandleDType::F32,
                    &device,
                )
                .map_err(|e| {
                    HiddenStateError::ProviderError(format!(
                        "Failed to mmap safetensors weights: {e}"
                    ))
                })?
            }
        } else {
            VarBuilder::from_pth(weights_path, CandleDType::F32, &device).map_err(|e| {
                HiddenStateError::ProviderError(format!(
                    "Failed to load PyTorch weights: {e}"
                ))
            })?
        };

        let model = BertModel::load(vb, &bert_config).map_err(|e| {
            HiddenStateError::ProviderError(format!("Failed to construct BertModel: {e}"))
        })?;

        // Build provider-level config.
        let hidden_state_config = HiddenStateConfig {
            capture_attention_weights: candle_config.capture_attention_weights,
            ..HiddenStateConfig::default()
        };

        Ok(Self {
            model,
            tokenizer,
            device,
            hidden_state_config,
            candle_config,
            hidden_dim,
        })
    }

    /// Convenience constructor that takes individual string parameters.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn from_params(
        model_id: &str,
        revision: &str,
        device: CandleDevice,
    ) -> Result<Self, HiddenStateError> {
        Self::new(
            CandleHiddenStateConfig::new(model_id, revision).with_device(device),
        )
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Tokenise `text`, truncating to `max_len` tokens, and return the
    /// resulting token-ID, token-type-ID, and position-ID tensors on
    /// `self.device`.
    fn tokenise(
        &self,
        text: &str,
        max_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor, usize), HiddenStateError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| HiddenStateError::ProviderError(format!("Tokenisation failed: {e}")))?;

        let ids: Vec<u32> = encoding
            .get_ids()
            .iter()
            .copied()
            .take(max_len)
            .collect();

        let seq_len = ids.len();
        if seq_len == 0 {
            return Err(HiddenStateError::ProviderError(
                "Tokenisation produced zero tokens — input may be empty".to_string(),
            ));
        }

        let type_ids: Vec<u32> = vec![0u32; seq_len];
        let position_ids: Vec<u32> = (0u32..u32::try_from(seq_len).unwrap_or(u32::MAX)).collect();

        // Build [1, seq_len] tensors.
        let input_ids =
            Tensor::from_vec(ids, (1, seq_len), &self.device).map_err(|e| {
                HiddenStateError::ProviderError(format!("Failed to build input_ids tensor: {e}"))
            })?;

        let token_type_ids =
            Tensor::from_vec(type_ids, (1, seq_len), &self.device).map_err(|e| {
                HiddenStateError::ProviderError(format!(
                    "Failed to build token_type_ids tensor: {e}"
                ))
            })?;

        let position_ids_tensor =
            Tensor::from_vec(position_ids, (1, seq_len), &self.device).map_err(|e| {
                HiddenStateError::ProviderError(format!(
                    "Failed to build position_ids tensor: {e}"
                ))
            })?;

        Ok((input_ids, token_type_ids, position_ids_tensor, seq_len))
    }

    /// Run the BERT forward pass and return `[1, seq_len, hidden_dim]` f32
    /// data as a flat `Vec<f32>`.
    fn forward_pass(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Vec<f32>, HiddenStateError> {
        let hidden_states = self
            .model
            .forward(input_ids, token_type_ids, Some(position_ids))
            .map_err(|e| {
                HiddenStateError::ProviderError(format!("BERT forward pass failed: {e}"))
            })?;

        // Ensure we work in f32 and flatten to a contiguous Vec.
        let data = hidden_states
            .to_dtype(CandleDType::F32)
            .map_err(|e| {
                HiddenStateError::ProviderError(format!("Dtype conversion failed: {e}"))
            })?
            .flatten_all()
            .map_err(|e| {
                HiddenStateError::ProviderError(format!("Tensor flattening failed: {e}"))
            })?
            .to_vec1::<f32>()
            .map_err(|e| {
                HiddenStateError::ProviderError(format!(
                    "Failed to extract f32 data from tensor: {e}"
                ))
            })?;

        Ok(data)
    }

    /// Build a `ModelHiddenStates` from raw f32 data of shape
    /// `[1, seq_len, hidden_dim]`.
    fn build_model_hidden_states(
        &self,
        data: Vec<f32>,
        seq_len: usize,
    ) -> Result<ModelHiddenStates, HiddenStateError> {
        // Expected element count: 1 × seq_len × hidden_dim.
        let expected = seq_len * self.hidden_dim;
        if data.len() != expected {
            return Err(HiddenStateError::ProviderError(format!(
                "Hidden state data length mismatch: expected {} (seq_len={seq_len} × \
                 hidden_dim={}), got {}",
                expected,
                self.hidden_dim,
                data.len()
            )));
        }

        let shape = TensorShape::new(vec![1, seq_len, self.hidden_dim]);
        let hidden_tensor = HiddenStateTensor::from_vec(data, shape).map_err(|e| {
            HiddenStateError::ProviderError(format!(
                "Failed to construct hidden state tensor: {e}"
            ))
        })?;

        let layer = LayerHiddenState::new(0, hidden_tensor);

        let mut states =
            ModelHiddenStates::new(&self.candle_config.model_id, 1, self.hidden_dim);
        states.sequence_length = seq_len;
        states.add_layer(layer);

        Ok(states)
    }

    /// Shared extraction logic used by both trait methods.
    fn extract_sync(&self, text: &str) -> Result<ModelHiddenStates, HiddenStateError> {
        let max_len = self.candle_config.max_sequence_length;
        let (input_ids, token_type_ids, position_ids, seq_len) =
            self.tokenise(text, max_len)?;

        let data = self.forward_pass(&input_ids, &token_type_ids, &position_ids)?;
        self.build_model_hidden_states(data, seq_len)
    }

    /// Build an empty `ModelKVCache` compatible with this model.
    ///
    /// BERT is a bidirectional encoder and does not perform autoregressive
    /// decoding; therefore its KV cache is always empty.
    fn empty_kv_cache(&self) -> ModelKVCache {
        ModelKVCache {
            model_id: self.candle_config.model_id.clone(),
            layers: Vec::new(),
            max_seq_len: self.candle_config.max_sequence_length,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Default impl  (sentence-transformers/all-MiniLM-L6-v2, 384 hidden, 6 layers)
// ────────────────────────────────────────────────────────────────────────────

impl Default for CandleHiddenStateProvider {
    /// Panics if model loading fails.  Use [`CandleHiddenStateProvider::new`]
    /// for fallible construction.
    fn default() -> Self {
        Self::new(CandleHiddenStateConfig::default())
            .expect("Default CandleHiddenStateProvider model should load successfully")
    }
}

// ────────────────────────────────────────────────────────────────────────────
// HiddenStateProvider trait impl
// ────────────────────────────────────────────────────────────────────────────

#[async_trait]
impl HiddenStateProvider for CandleHiddenStateProvider {
    /// Extract hidden states from `text` via a live BERT forward pass.
    ///
    /// The returned `ModelHiddenStates` contains one layer (index 0) whose
    /// hidden-state tensor has shape `[1, seq_len, hidden_dim]`.
    ///
    /// # Errors
    ///
    /// Returns [`HiddenStateError::ProviderError`] if tokenisation or the
    /// forward pass fails.
    async fn extract_hidden_states(
        &self,
        text: &str,
    ) -> Result<ModelHiddenStates, HiddenStateError> {
        self.extract_sync(text)
    }

    /// Extract hidden states with KV-cache awareness.
    ///
    /// BERT is a non-autoregressive encoder; it has no KV cache.  This method
    /// ignores `past_kv` and returns an empty `ModelKVCache` alongside the
    /// freshly computed hidden states.
    ///
    /// # Errors
    ///
    /// Returns [`HiddenStateError::ProviderError`] if the forward pass fails.
    async fn extract_with_kv_cache(
        &self,
        text: &str,
        _past_kv: Option<&ModelKVCache>,
    ) -> Result<(ModelHiddenStates, ModelKVCache), HiddenStateError> {
        let states = self.extract_sync(text)?;
        let kv_cache = self.empty_kv_cache();
        Ok((states, kv_cache))
    }

    fn model_config(&self) -> &HiddenStateConfig {
        &self.hidden_state_config
    }

    fn model_id(&self) -> &str {
        &self.candle_config.model_id
    }

    /// Number of hidden-state layers returned per forward pass.
    ///
    /// Currently always `1`: we expose only the final encoder output.
    fn num_layers(&self) -> usize {
        1
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config-level tests (no model loading required) ────────────────────

    #[test]
    fn test_candle_hidden_state_config_default() {
        let config = CandleHiddenStateConfig::default();
        assert!(
            config.model_id.contains("MiniLM"),
            "Default model should be MiniLM-based"
        );
        assert_eq!(config.revision, "main");
        assert!(!config.capture_attention_weights);
        assert_eq!(config.max_sequence_length, 512);
    }

    #[test]
    fn test_candle_hidden_state_config_custom_model() {
        let config = CandleHiddenStateConfig {
            model_id: "BAAI/bge-base-en-v1.5".to_string(),
            revision: "main".to_string(),
            device: CandleDevice::Cpu,
            capture_attention_weights: false,
            max_sequence_length: 512,
        };
        assert!(config.model_id.contains("bge"));
        assert_eq!(config.revision, "main");
    }

    #[test]
    fn test_candle_hidden_state_config_builder() {
        let config = CandleHiddenStateConfig::new(
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "refs/pr/1",
        )
        .with_device(CandleDevice::Cpu)
        .with_capture_attention_weights(true)
        .with_max_sequence_length(256);

        assert!(config.model_id.contains("paraphrase"));
        assert_eq!(config.revision, "refs/pr/1");
        assert!(config.capture_attention_weights);
        assert_eq!(config.max_sequence_length, 256);
    }

    #[test]
    fn test_candle_device_default_is_cpu() {
        let device = CandleDevice::default();
        // Pattern-match to verify the variant without inspecting internals.
        assert!(matches!(device, CandleDevice::Cpu));
    }

    #[test]
    fn test_candle_device_to_candle_device_cpu() {
        let device = CandleDevice::Cpu;
        let result = device.to_candle_device();
        assert!(result.is_ok(), "CPU device conversion must not fail");
    }

    #[test]
    fn test_candle_hidden_state_config_three_presets() {
        // Preset 1 – default / MiniLM
        let default_cfg = CandleHiddenStateConfig::default();
        assert!(default_cfg.model_id.contains("MiniLM"));

        // Preset 2 – BGE base
        let bge_cfg = CandleHiddenStateConfig::new("BAAI/bge-base-en-v1.5", "main");
        assert!(bge_cfg.model_id.contains("bge"));

        // Preset 3 – DistilBERT
        let distil_cfg = CandleHiddenStateConfig::new("distilbert-base-uncased", "main")
            .with_max_sequence_length(128);
        assert!(distil_cfg.model_id.contains("distilbert"));
        assert_eq!(distil_cfg.max_sequence_length, 128);
    }

    #[test]
    fn test_candle_hidden_state_config_sequence_length_variants() {
        let short = CandleHiddenStateConfig::default().with_max_sequence_length(64);
        let medium = CandleHiddenStateConfig::default().with_max_sequence_length(256);
        let long = CandleHiddenStateConfig::default().with_max_sequence_length(512);

        assert_eq!(short.max_sequence_length, 64);
        assert_eq!(medium.max_sequence_length, 256);
        assert_eq!(long.max_sequence_length, 512);
    }
}
