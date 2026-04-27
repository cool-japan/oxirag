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
// Pooling strategy
// ────────────────────────────────────────────────────────────────────────────

/// Strategy for pooling token-level hidden states into a single sentence representation.
///
/// After a BERT forward pass, the output has shape `[1, seq_len, hidden_dim]`. A pooling
/// strategy collapses the `seq_len` dimension to produce a `[1, 1, hidden_dim]` tensor
/// suitable for sentence-level similarity tasks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HiddenStatePooling {
    /// Use the `[CLS]` token embedding (token index 0). Best for classification tasks
    /// and models fine-tuned with sentence-transformers.
    #[default]
    Cls,
    /// Arithmetic mean over all token positions. Robust general-purpose pooling.
    MeanPool,
    /// Element-wise maximum over token positions. Captures the most salient features.
    MaxPool,
    /// Mean pooling that ignores padding tokens (positions with id 0).
    /// Requires the raw token IDs to identify padding.
    MaskMean,
}

/// Apply pooling to convert `[1, seq_len, hidden_dim]` flat data into a `[hidden_dim]` vector.
///
/// This is a pure, free function so it can be called and tested independently of any
/// model-loading infrastructure. It is the kernel used by
/// [`CandleHiddenStateProvider::pool_hidden_states`].
///
/// # Parameters
///
/// - `data` – flat `f32` slice of length `seq_len * hidden_dim`, representing the
///   `[1, seq_len, hidden_dim]` output of a BERT encoder.
/// - `seq_len` – number of token positions in `data`.
/// - `hidden_dim` – dimensionality of each token embedding.
/// - `pooling` – which pooling strategy to apply.
/// - `token_ids` – raw token IDs used for padding detection in [`HiddenStatePooling::MaskMean`].
///   When `None`, all positions are treated as non-padding.
///
/// # Returns
///
/// A `Vec<f32>` of length `hidden_dim` containing the pooled representation.
#[must_use]
pub fn apply_hidden_state_pooling(
    data: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    pooling: HiddenStatePooling,
    token_ids: Option<&[u32]>,
) -> Vec<f32> {
    match pooling {
        HiddenStatePooling::Cls => {
            // Take the first hidden_dim elements: the [CLS] token at position 0.
            data[..hidden_dim].to_vec()
        }
        HiddenStatePooling::MeanPool => {
            let mut result = vec![0.0f32; hidden_dim];
            for t in 0..seq_len {
                let offset = t * hidden_dim;
                for (i, r) in result.iter_mut().enumerate() {
                    *r += data[offset + i];
                }
            }
            let scale = 1.0_f32 / f32::from(u16::try_from(seq_len).unwrap_or(u16::MAX));
            for v in &mut result {
                *v *= scale;
            }
            result
        }
        HiddenStatePooling::MaxPool => {
            let mut result = vec![f32::NEG_INFINITY; hidden_dim];
            for t in 0..seq_len {
                let offset = t * hidden_dim;
                for (i, r) in result.iter_mut().enumerate() {
                    *r = r.max(data[offset + i]);
                }
            }
            result
        }
        HiddenStatePooling::MaskMean => {
            // Count non-padding tokens (token_id != 0).
            let mask: Vec<bool> = token_ids.map_or_else(
                || vec![true; seq_len],
                |ids| ids.iter().map(|&id| id != 0).collect(),
            );
            let valid_count = mask.iter().filter(|&&m| m).count().max(1);
            let mut result = vec![0.0f32; hidden_dim];
            for (t, &is_valid) in mask.iter().enumerate() {
                if is_valid {
                    let offset = t * hidden_dim;
                    for (i, r) in result.iter_mut().enumerate() {
                        *r += data[offset + i];
                    }
                }
            }
            let scale = 1.0_f32 / f32::from(u16::try_from(valid_count).unwrap_or(u16::MAX));
            for v in &mut result {
                *v *= scale;
            }
            result
        }
    }
}

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
    /// Pooling strategy applied by [`CandleHiddenStateProvider::extract_sentence_embedding`].
    ///
    /// Does not affect [`extract_hidden_states`], which always returns the full
    /// `[seq_len, hidden_dim]` tensor.
    ///
    /// [`extract_hidden_states`]: CandleHiddenStateProvider::extract_hidden_states
    pub pooling: HiddenStatePooling,
}

impl Default for CandleHiddenStateConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            revision: "main".to_string(),
            device: CandleDevice::Cpu,
            capture_attention_weights: false,
            max_sequence_length: 512,
            pooling: HiddenStatePooling::Cls,
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

    /// Set the hidden-state pooling strategy.
    ///
    /// Only affects [`CandleHiddenStateProvider::extract_sentence_embedding`].
    #[must_use]
    pub fn with_pooling(mut self, pooling: HiddenStatePooling) -> Self {
        self.pooling = pooling;
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

    /// Apply the configured pooling strategy to flatten `[1, seq_len, hidden_dim]` data.
    ///
    /// Delegates to the free function [`apply_hidden_state_pooling`] using
    /// `self.hidden_dim` and `self.candle_config.pooling`.
    fn pool_hidden_states(
        &self,
        data: &[f32],
        seq_len: usize,
        token_ids: Option<&[u32]>,
    ) -> Vec<f32> {
        apply_hidden_state_pooling(
            data,
            seq_len,
            self.hidden_dim,
            self.candle_config.pooling,
            token_ids,
        )
    }

    /// Extract a pooled sentence embedding as a flat `Vec<f32>` of length `hidden_dim`.
    ///
    /// Unlike [`extract_hidden_states`] which returns the full `[seq_len, hidden_dim]` tensor,
    /// this method applies the configured [`HiddenStatePooling`] strategy to produce a
    /// single fixed-size vector suitable for semantic similarity tasks.
    ///
    /// The [`HiddenStatePooling::MaskMean`] strategy uses the raw token IDs (before tensor
    /// construction) to correctly identify and exclude padding positions.
    ///
    /// # Errors
    ///
    /// Returns [`HiddenStateError::ProviderError`] if tokenisation or model inference fails.
    ///
    /// [`extract_hidden_states`]: CandleHiddenStateProvider::extract_hidden_states
    pub fn extract_sentence_embedding(
        &self,
        text: &str,
    ) -> Result<Vec<f32>, HiddenStateError> {
        let max_len = self.candle_config.max_sequence_length;

        // Tokenise, obtaining the raw token IDs alongside the tensors.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| HiddenStateError::ProviderError(format!("Tokenisation failed: {e}")))?;

        let raw_ids: Vec<u32> = encoding
            .get_ids()
            .iter()
            .copied()
            .take(max_len)
            .collect();

        let seq_len = raw_ids.len();
        if seq_len == 0 {
            return Err(HiddenStateError::ProviderError(
                "Tokenisation produced zero tokens — input may be empty".to_string(),
            ));
        }

        let type_ids: Vec<u32> = vec![0u32; seq_len];
        let position_ids_raw: Vec<u32> =
            (0u32..u32::try_from(seq_len).unwrap_or(u32::MAX)).collect();

        let input_ids =
            Tensor::from_vec(raw_ids.clone(), (1, seq_len), &self.device).map_err(|e| {
                HiddenStateError::ProviderError(format!("Failed to build input_ids tensor: {e}"))
            })?;

        let token_type_ids =
            Tensor::from_vec(type_ids, (1, seq_len), &self.device).map_err(|e| {
                HiddenStateError::ProviderError(format!(
                    "Failed to build token_type_ids tensor: {e}"
                ))
            })?;

        let position_ids_tensor =
            Tensor::from_vec(position_ids_raw, (1, seq_len), &self.device).map_err(|e| {
                HiddenStateError::ProviderError(format!(
                    "Failed to build position_ids tensor: {e}"
                ))
            })?;

        let data =
            self.forward_pass(&input_ids, &token_type_ids, &position_ids_tensor)?;

        // Pool using raw token IDs for MaskMean padding detection.
        Ok(self.pool_hidden_states(&data, seq_len, Some(&raw_ids)))
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
            pooling: HiddenStatePooling::MeanPool,
        };
        assert!(config.model_id.contains("bge"));
        assert_eq!(config.revision, "main");
        assert_eq!(config.pooling, HiddenStatePooling::MeanPool);
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

// ────────────────────────────────────────────────────────────────────────────
// Pooling unit tests (no model loading required)
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod pooling_tests {
    use super::*;

    // Helper: build a synthetic [1, seq_len, hidden_dim] flat buffer where
    // token t has value `(t + 1) as f32` in every dimension.
    fn synthetic_data(seq_len: usize, hidden_dim: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(seq_len * hidden_dim);
        for t in 0..seq_len {
            let val = (t + 1) as f32;
            data.extend(std::iter::repeat(val).take(hidden_dim));
        }
        data
    }

    #[test]
    fn test_cls_pooling_extracts_first_token() {
        // seq_len=3, hidden_dim=4: tokens are [1.0, 1.0, 1.0, 1.0], [2.0, ...], [3.0, ...]
        let data = synthetic_data(3, 4);
        let result = apply_hidden_state_pooling(&data, 3, 4, HiddenStatePooling::Cls, None);

        assert_eq!(result.len(), 4);
        // CLS (position 0) should be 1.0 in every dimension.
        for &v in &result {
            assert!(
                (v - 1.0_f32).abs() < f32::EPSILON,
                "CLS token should be 1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_mean_pool_averages_correctly() {
        // seq_len=4, hidden_dim=2: token values 1.0, 2.0, 3.0, 4.0
        // Expected mean = (1+2+3+4)/4 = 2.5
        let data = synthetic_data(4, 2);
        let result = apply_hidden_state_pooling(&data, 4, 2, HiddenStatePooling::MeanPool, None);

        assert_eq!(result.len(), 2);
        let expected = 2.5_f32;
        for &v in &result {
            assert!(
                (v - expected).abs() < 1e-5,
                "MeanPool should yield {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_max_pool_takes_maximum() {
        // seq_len=5, hidden_dim=3: max token value = 5.0
        let data = synthetic_data(5, 3);
        let result = apply_hidden_state_pooling(&data, 5, 3, HiddenStatePooling::MaxPool, None);

        assert_eq!(result.len(), 3);
        let expected = 5.0_f32;
        for &v in &result {
            assert!(
                (v - expected).abs() < f32::EPSILON,
                "MaxPool should yield {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_mask_mean_ignores_padding() {
        // seq_len=4, hidden_dim=2.
        // Token IDs: [101, 2022, 102, 0]  — last token is PAD (id == 0).
        // Token values: 1.0, 2.0, 3.0, 4.0
        // MaskMean should average only positions 0–2: (1+2+3)/3 = 2.0
        let token_ids: Vec<u32> = vec![101, 2022, 102, 0];
        let data = synthetic_data(4, 2);

        let result = apply_hidden_state_pooling(
            &data,
            4,
            2,
            HiddenStatePooling::MaskMean,
            Some(&token_ids),
        );

        assert_eq!(result.len(), 2);
        let expected = 2.0_f32;
        for &v in &result {
            assert!(
                (v - expected).abs() < 1e-5,
                "MaskMean should yield {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_default_pooling_is_cls() {
        let default_pooling = HiddenStatePooling::default();
        assert_eq!(
            default_pooling,
            HiddenStatePooling::Cls,
            "Default pooling strategy must be CLS"
        );
    }

    #[test]
    fn test_pooling_config_builder() {
        // Verify the builder chain works for all pooling variants.
        let cls_cfg = CandleHiddenStateConfig::default().with_pooling(HiddenStatePooling::Cls);
        assert_eq!(cls_cfg.pooling, HiddenStatePooling::Cls);

        let mean_cfg =
            CandleHiddenStateConfig::default().with_pooling(HiddenStatePooling::MeanPool);
        assert_eq!(mean_cfg.pooling, HiddenStatePooling::MeanPool);

        let max_cfg = CandleHiddenStateConfig::default().with_pooling(HiddenStatePooling::MaxPool);
        assert_eq!(max_cfg.pooling, HiddenStatePooling::MaxPool);

        let mask_cfg =
            CandleHiddenStateConfig::default().with_pooling(HiddenStatePooling::MaskMean);
        assert_eq!(mask_cfg.pooling, HiddenStatePooling::MaskMean);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests
    // -----------------------------------------------------------------------

    /// `HiddenStateConfig` pooling field defaults to CLS.
    #[test]
    fn test_hidden_state_config_pooling_default() {
        let config = CandleHiddenStateConfig::default();
        assert_eq!(
            config.pooling,
            HiddenStatePooling::Cls,
            "CandleHiddenStateConfig default pooling must be Cls"
        );
    }

    /// `with_pooling` builder round-trips through all four strategies.
    #[test]
    fn test_config_with_pooling_builder_all_variants() {
        let variants = [
            HiddenStatePooling::Cls,
            HiddenStatePooling::MeanPool,
            HiddenStatePooling::MaxPool,
            HiddenStatePooling::MaskMean,
        ];
        for variant in variants {
            let config = CandleHiddenStateConfig::default().with_pooling(variant);
            assert_eq!(
                config.pooling, variant,
                "with_pooling({variant:?}) must set config.pooling"
            );
        }
    }

    /// CLS pooling on a zero-vector returns zeros (not NaN/Inf).
    #[test]
    fn test_cls_pooling_zero_data_is_finite() {
        let data = vec![0.0f32; 16];
        let result =
            apply_hidden_state_pooling(&data, 2, 8, HiddenStatePooling::Cls, None);
        assert_eq!(result.len(), 8, "CLS pooling must return hidden_dim elements");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "CLS output[{i}] must be finite, got {v}");
        }
    }

    /// MeanPool on a single-token sequence equals that token's values.
    #[test]
    fn test_mean_pool_single_token_equals_token_values() {
        // hidden_dim=4, seq_len=1
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let result =
            apply_hidden_state_pooling(&data, 1, 4, HiddenStatePooling::MeanPool, None);
        assert_eq!(result, data, "MeanPool of a single token must equal that token");
    }

    /// MaxPool on identical tokens returns the same values.
    #[test]
    fn test_max_pool_uniform_data_returns_same_value() {
        // 3 tokens, each [0.5, 0.5], hidden_dim=2
        let data = vec![0.5_f32, 0.5, 0.5, 0.5, 0.5, 0.5];
        let result =
            apply_hidden_state_pooling(&data, 3, 2, HiddenStatePooling::MaxPool, None);
        for &v in &result {
            assert!(
                (v - 0.5).abs() < f32::EPSILON,
                "MaxPool of uniform data must yield the uniform value, got {v}"
            );
        }
    }

    /// MaskMean with all padding IDs (zeros) falls back to averaging all tokens.
    #[test]
    fn test_mask_mean_all_padding_falls_back_to_all_tokens() {
        // All token_ids are 0 (padding) — the implementation must not produce NaN.
        // Expected: the implementation treats valid_count = max(0_padding_count, 1)
        // so it averages all tokens rather than dividing by zero.
        let data = synthetic_data(3, 2);
        let token_ids: Vec<u32> = vec![0, 0, 0]; // all padding
        let result = apply_hidden_state_pooling(
            &data,
            3,
            2,
            HiddenStatePooling::MaskMean,
            Some(&token_ids),
        );
        assert_eq!(result.len(), 2, "MaskMean must return hidden_dim elements");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "MaskMean output[{i}] must be finite, got {v}");
        }
    }

    /// `CandleHiddenStateConfig::new` sets model_id and revision correctly.
    #[test]
    fn test_config_new_sets_fields() {
        let config = CandleHiddenStateConfig::new("model/id", "v1.0");
        assert_eq!(config.model_id, "model/id");
        assert_eq!(config.revision, "v1.0");
    }

    /// `with_capture_attention_weights` toggles the field correctly.
    #[test]
    fn test_config_capture_attention_weights_toggle() {
        let enabled = CandleHiddenStateConfig::default().with_capture_attention_weights(true);
        assert!(enabled.capture_attention_weights);

        let disabled = CandleHiddenStateConfig::default().with_capture_attention_weights(false);
        assert!(!disabled.capture_attention_weights);
    }

    /// `with_max_sequence_length` is respected by the builder.
    #[test]
    fn test_config_max_sequence_length_variants() {
        for &len in &[64_usize, 128, 256, 512] {
            let config = CandleHiddenStateConfig::default().with_max_sequence_length(len);
            assert_eq!(
                config.max_sequence_length, len,
                "with_max_sequence_length({len}) must set the field to {len}"
            );
        }
    }
}
