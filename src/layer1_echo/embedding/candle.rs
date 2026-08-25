//! Candle-based embedding provider.

use async_trait::async_trait;

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::EmbeddingProvider;

#[cfg(feature = "speculator")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "speculator")]
use candle_nn::VarBuilder;
#[cfg(feature = "speculator")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
#[cfg(feature = "speculator")]
use hf_hub::{Repo, RepoType, api::sync::Api};
#[cfg(feature = "speculator")]
use tokenizers::Tokenizer;

/// Configuration for the Candle embedding provider.
#[derive(Debug, Clone)]
pub struct CandleEmbeddingConfig {
    /// `HuggingFace` model identifier.
    pub model_id: String,
    /// Model revision.
    pub revision: String,
    /// Device to use (CPU or CUDA).
    pub device: CandleDevice,
    /// Whether to normalize embeddings.
    pub normalize: bool,
    /// Maximum sequence length.
    pub max_length: usize,
}

impl Default for CandleEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            revision: "main".to_string(),
            device: CandleDevice::Cpu,
            normalize: true,
            max_length: 512,
        }
    }
}

impl CandleEmbeddingConfig {
    /// Create config for `BAAI/bge-base-en-v1.5` (768-dim, strong general-purpose performance).
    ///
    /// BGE-base is the recommended default for most English embedding workloads: it is
    /// significantly more accurate than small/MiniLM models while still being fast enough
    /// for CPU inference.
    #[must_use]
    pub fn bge_base_en_v15() -> Self {
        Self {
            model_id: "BAAI/bge-base-en-v1.5".to_string(),
            revision: "main".to_string(),
            max_length: 512,
            ..Default::default()
        }
    }

    /// Create config for `BAAI/bge-large-en-v1.5` (1024-dim, best accuracy).
    ///
    /// BGE-large delivers state-of-the-art retrieval quality on the MTEB benchmark.
    /// Prefer this when accuracy is more important than throughput.
    #[must_use]
    pub fn bge_large_en_v15() -> Self {
        Self {
            model_id: "BAAI/bge-large-en-v1.5".to_string(),
            revision: "main".to_string(),
            max_length: 512,
            ..Default::default()
        }
    }

    /// Create config for `sentence-transformers/all-mpnet-base-v2` (768-dim).
    ///
    /// A widely-used SBERT model fine-tuned on 1 billion+ sentence pairs.
    /// Balanced quality across semantic similarity, clustering, and retrieval tasks.
    /// Maximum input length is 384 tokens (the model's native window).
    #[must_use]
    pub fn all_mpnet_base_v2() -> Self {
        Self {
            model_id: "sentence-transformers/all-mpnet-base-v2".to_string(),
            revision: "main".to_string(),
            max_length: 384,
            ..Default::default()
        }
    }

    /// Create config for `BAAI/bge-small-en-v1.5` (384-dim, fastest inference).
    ///
    /// BGE-small is the lightest BGE model: ideal for latency-sensitive paths or
    /// edge/WASM deployments where memory footprint matters.
    #[must_use]
    pub fn bge_small_en_v15() -> Self {
        Self {
            model_id: "BAAI/bge-small-en-v1.5".to_string(),
            revision: "main".to_string(),
            max_length: 512,
            ..Default::default()
        }
    }
}

/// Device selection for Candle.
#[derive(Debug, Clone, Copy, Default)]
pub enum CandleDevice {
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
impl CandleDevice {
    fn to_candle_device(self) -> Device {
        match self {
            CandleDevice::Cpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            CandleDevice::Cuda(ordinal) => {
                Device::new_cuda(ordinal).expect("CUDA device should be available")
            }
            #[cfg(feature = "metal")]
            CandleDevice::Metal => Device::new_metal(0).expect("Metal device should be available"),
        }
    }
}

/// Candle-based embedding provider using BERT models.
#[cfg(feature = "speculator")]
pub struct CandleEmbeddingProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    config: CandleEmbeddingConfig,
    dimension: usize,
}

#[cfg(feature = "speculator")]
impl CandleEmbeddingProvider {
    /// Create a new Candle embedding provider.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded.
    pub fn new(config: CandleEmbeddingConfig) -> Result<Self, EmbeddingError> {
        let device = config.device.to_candle_device();

        // Load model from HuggingFace Hub
        let api = Api::new().map_err(|e| EmbeddingError::ModelLoad(e.to_string()))?;
        let repo = api.repo(Repo::with_revision(
            config.model_id.clone(),
            RepoType::Model,
            config.revision.clone(),
        ));

        // Load tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to load tokenizer: {e}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbeddingError::Tokenization(e.to_string()))?;

        // Load config
        let config_path = repo
            .get("config.json")
            .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to load config: {e}")))?;
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to read config: {e}")))?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to parse config: {e}")))?;

        let dimension = bert_config.hidden_size;

        // Load model weights
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to load weights: {e}")))?;

        let vb = if weights_path
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device).map_err(
                    |e| EmbeddingError::ModelLoad(format!("Failed to load safetensors: {e}")),
                )?
            }
        } else {
            VarBuilder::from_pth(weights_path, DType::F32, &device).map_err(|e| {
                EmbeddingError::ModelLoad(format!("Failed to load PyTorch weights: {e}"))
            })?
        };

        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| EmbeddingError::ModelLoad(format!("Failed to create model: {e}")))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            dimension,
        })
    }

    fn mean_pooling(
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, EmbeddingError> {
        // Expand attention mask to match hidden states dimensions
        let mask = attention_mask
            .unsqueeze(2)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to unsqueeze mask: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to convert mask dtype: {e}")))?;

        // Apply mask and sum
        let masked = hidden_states
            .broadcast_mul(&mask)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to apply mask: {e}")))?;

        let sum = masked
            .sum(1)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to sum: {e}")))?;

        let mask_sum = mask
            .sum(1)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to sum mask: {e}")))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to clamp: {e}")))?;

        sum.broadcast_div(&mask_sum)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to divide: {e}")))
    }

    fn normalize_tensor(tensor: &Tensor) -> Result<Tensor, EmbeddingError> {
        let norm = tensor
            .sqr()
            .map_err(|e| EmbeddingError::Inference(format!("Failed to square: {e}")))?
            .sum_keepdim(1)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to sum: {e}")))?
            .sqrt()
            .map_err(|e| EmbeddingError::Inference(format!("Failed to sqrt: {e}")))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to clamp: {e}")))?;

        tensor
            .broadcast_div(&norm)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to normalize: {e}")))
    }
}

#[cfg(feature = "speculator")]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EmbeddingProvider for CandleEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.embed_batch(&[text]).await?;
        results.into_iter().next().ok_or(EmbeddingError::EmptyInput)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbeddingError::Tokenization(e.to_string()))?;

        // Get max length
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.config.max_length);

        // Pad and create tensors
        let mut input_ids = Vec::new();
        let mut attention_masks = Vec::new();
        let mut token_type_ids = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();

            let mut padded_ids = ids[..ids.len().min(max_len)].to_vec();
            let mut padded_mask = mask[..mask.len().min(max_len)].to_vec();
            let mut padded_types = types[..types.len().min(max_len)].to_vec();

            padded_ids.resize(max_len, 0);
            padded_mask.resize(max_len, 0);
            padded_types.resize(max_len, 0);

            input_ids.push(padded_ids);
            attention_masks.push(padded_mask);
            token_type_ids.push(padded_types);
        }

        let batch_size = texts.len();

        let input_ids_flat: Vec<u32> = input_ids.into_iter().flatten().collect();
        let attention_mask_flat: Vec<u32> = attention_masks.into_iter().flatten().collect();
        let token_type_ids_flat: Vec<u32> = token_type_ids.into_iter().flatten().collect();

        let input_ids_tensor =
            Tensor::from_vec(input_ids_flat, (batch_size, max_len), &self.device).map_err(|e| {
                EmbeddingError::Inference(format!("Failed to create input tensor: {e}"))
            })?;

        let attention_mask_tensor =
            Tensor::from_vec(attention_mask_flat, (batch_size, max_len), &self.device).map_err(
                |e| EmbeddingError::Inference(format!("Failed to create mask tensor: {e}")),
            )?;

        let token_type_ids_tensor =
            Tensor::from_vec(token_type_ids_flat, (batch_size, max_len), &self.device).map_err(
                |e| EmbeddingError::Inference(format!("Failed to create type tensor: {e}")),
            )?;

        // Forward pass
        let hidden_states = self
            .model
            .forward(
                &input_ids_tensor,
                &token_type_ids_tensor,
                Some(&attention_mask_tensor),
            )
            .map_err(|e| EmbeddingError::Inference(format!("Forward pass failed: {e}")))?;

        // Mean pooling
        let pooled = Self::mean_pooling(&hidden_states, &attention_mask_tensor)?;

        // Normalize if configured
        let output = if self.config.normalize {
            Self::normalize_tensor(&pooled)?
        } else {
            pooled
        };

        // Convert to Vec<Vec<f32>>
        let output_vec: Vec<f32> = output
            .to_vec2()
            .map_err(|e| EmbeddingError::Inference(format!("Failed to convert output: {e}")))?
            .into_iter()
            .flatten()
            .collect();

        Ok(output_vec
            .chunks(self.dimension)
            .map(<[f32]>::to_vec)
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        &self.config.model_id
    }
}

/// A mock embedding provider that generates deterministic pseudo-random vectors for testing.
///
/// No ML model is loaded. Each text is hashed to seed a simple numeric computation that
/// produces a unit-normalised vector of the requested `dimension`. The same text always
/// produces the same vector within one process, so tests are reproducible.
///
/// This provider is WASM-compatible and has no native dependencies.
pub struct MockEmbeddingProvider {
    dimension: usize,
    model_id: String,
}

impl MockEmbeddingProvider {
    /// Create a new mock embedding provider.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model_id: "mock-embedding-model".to_string(),
        }
    }

    /// Set the model ID.
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Generate a deterministic embedding based on text hash.
    fn hash_to_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut embedding = Vec::with_capacity(self.dimension);
        let mut current_hash = hash;

        for _ in 0..self.dimension {
            // Generate pseudo-random values between -1 and 1
            #[allow(clippy::cast_precision_loss)]
            let value = ((current_hash % 10000) as f32 / 5000.0) - 1.0;
            embedding.push(value);
            current_hash = current_hash
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        embedding
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        Ok(self.hash_to_embedding(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        texts
            .iter()
            .map(|t| self.hash_to_embedding(t))
            .map(Ok)
            .collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_dimension() {
        let provider = MockEmbeddingProvider::new(384);
        assert_eq!(provider.dimension(), 384);
    }

    #[tokio::test]
    async fn test_mock_provider_embed() {
        let provider = MockEmbeddingProvider::new(128);
        let embedding = provider
            .embed("test text")
            .await
            .expect("test operation should succeed");
        assert_eq!(embedding.len(), 128);
    }

    #[tokio::test]
    async fn test_mock_provider_batch() {
        let provider = MockEmbeddingProvider::new(64);
        let embeddings = provider
            .embed_batch(&["text1", "text2", "text3"])
            .await
            .expect("test operation should succeed");
        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.len(), 64);
        }
    }

    #[tokio::test]
    async fn test_mock_provider_deterministic() {
        let provider = MockEmbeddingProvider::new(32);
        let emb1 = provider
            .embed("same text")
            .await
            .expect("test operation should succeed");
        let emb2 = provider
            .embed("same text")
            .await
            .expect("test operation should succeed");
        assert_eq!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_mock_provider_different_texts() {
        let provider = MockEmbeddingProvider::new(32);
        let emb1 = provider
            .embed("text one")
            .await
            .expect("test operation should succeed");
        let emb2 = provider
            .embed("text two")
            .await
            .expect("test operation should succeed");
        assert_ne!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_mock_provider_normalized() {
        let provider = MockEmbeddingProvider::new(64);
        let embedding = provider
            .embed("test")
            .await
            .expect("test operation should succeed");
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_mock_provider_empty_input() {
        let provider = MockEmbeddingProvider::new(32);
        let result = provider.embed("").await;
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    #[tokio::test]
    async fn test_mock_provider_empty_batch() {
        let provider = MockEmbeddingProvider::new(32);
        let result = provider.embed_batch(&[]).await;
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    // -----------------------------------------------------------------------
    // BGE / model preset tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bge_base_preset() {
        let config = CandleEmbeddingConfig::bge_base_en_v15();
        assert!(
            config.model_id.contains("bge-base"),
            "model_id should contain 'bge-base'"
        );
        assert_eq!(config.max_length, 512);
        assert!(config.normalize, "normalize should be true by default");
    }

    #[test]
    fn test_bge_large_preset() {
        let config = CandleEmbeddingConfig::bge_large_en_v15();
        assert!(
            config.model_id.contains("bge-large"),
            "model_id should contain 'bge-large'"
        );
        assert_eq!(config.max_length, 512);
        assert!(config.normalize);
    }

    #[test]
    fn test_mpnet_preset() {
        let config = CandleEmbeddingConfig::all_mpnet_base_v2();
        assert!(
            config.model_id.contains("mpnet"),
            "model_id should contain 'mpnet'"
        );
        assert_eq!(config.max_length, 384, "mpnet native window is 384 tokens");
        assert!(config.normalize);
    }

    #[test]
    fn test_bge_small_preset() {
        let config = CandleEmbeddingConfig::bge_small_en_v15();
        assert!(
            config.model_id.contains("bge-small"),
            "model_id should contain 'bge-small'"
        );
        assert_eq!(config.max_length, 512);
        assert!(config.normalize);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case and preset-coverage tests
    // -----------------------------------------------------------------------

    /// All four presets must have a non-empty `model_id` and a non-zero `max_length`.
    #[test]
    fn test_all_presets_have_valid_model_ids() {
        let presets = [
            CandleEmbeddingConfig::bge_base_en_v15(),
            CandleEmbeddingConfig::bge_large_en_v15(),
            CandleEmbeddingConfig::bge_small_en_v15(),
            CandleEmbeddingConfig::all_mpnet_base_v2(),
        ];
        for (i, config) in presets.iter().enumerate() {
            assert!(
                !config.model_id.is_empty(),
                "preset {i}: model_id must not be empty"
            );
            assert!(
                config.max_length > 0,
                "preset {i}: max_length must be non-zero, got {}",
                config.max_length
            );
        }
    }

    /// All four presets must use the "main" revision.
    #[test]
    fn test_presets_use_main_revision() {
        let presets = [
            ("bge_base", CandleEmbeddingConfig::bge_base_en_v15()),
            ("bge_large", CandleEmbeddingConfig::bge_large_en_v15()),
            ("bge_small", CandleEmbeddingConfig::bge_small_en_v15()),
            ("mpnet", CandleEmbeddingConfig::all_mpnet_base_v2()),
        ];
        for (name, config) in &presets {
            assert_eq!(
                config.revision, "main",
                "preset '{name}' must use revision 'main', got '{}'",
                config.revision
            );
        }
    }

    /// The default `CandleDevice` must be `Cpu`.
    #[test]
    fn test_device_default_is_cpu() {
        let device = CandleDevice::default();
        assert!(
            matches!(device, CandleDevice::Cpu),
            "default CandleDevice must be Cpu"
        );
    }

    /// Default config must use "main" revision and have reasonable `max_length`.
    #[test]
    fn test_default_config_revision_and_length() {
        let config = CandleEmbeddingConfig::default();
        assert_eq!(
            config.revision, "main",
            "default config revision must be 'main'"
        );
        assert!(
            config.max_length >= 128,
            "default max_length must be at least 128, got {}",
            config.max_length
        );
        assert!(config.normalize, "default config must have normalize=true");
    }

    /// Preset model IDs must differ from each other (each preset targets a distinct model).
    #[test]
    fn test_preset_model_ids_are_distinct() {
        let ids = [
            CandleEmbeddingConfig::bge_base_en_v15().model_id,
            CandleEmbeddingConfig::bge_large_en_v15().model_id,
            CandleEmbeddingConfig::bge_small_en_v15().model_id,
            CandleEmbeddingConfig::all_mpnet_base_v2().model_id,
        ];
        let mut seen = std::collections::HashSet::new();
        for id in &ids {
            assert!(
                seen.insert(id.as_str()),
                "model_id '{id}' appears in more than one preset"
            );
        }
    }

    /// `MPNet` preset has a shorter `max_length` than the BGE presets.
    #[test]
    fn test_mpnet_shorter_window_than_bge() {
        let mpnet = CandleEmbeddingConfig::all_mpnet_base_v2();
        let bge_base = CandleEmbeddingConfig::bge_base_en_v15();
        assert!(
            mpnet.max_length < bge_base.max_length,
            "MPNet native window ({}) must be shorter than BGE-base window ({})",
            mpnet.max_length,
            bge_base.max_length
        );
    }

    /// Mock provider: `embed_batch` returns one embedding per text.
    #[tokio::test]
    async fn test_mock_provider_batch_count_matches_input() {
        let provider = MockEmbeddingProvider::new(16);
        for n in [1, 3, 5, 10] {
            let texts: Vec<&str> = (0..n).map(|_| "text").collect();
            let embeddings = provider
                .embed_batch(&texts)
                .await
                .expect("embed_batch must succeed");
            assert_eq!(
                embeddings.len(),
                n,
                "embed_batch must return exactly {n} embeddings"
            );
        }
    }

    /// Mock provider `model_id` is preserved by `with_model_id`.
    #[test]
    fn test_mock_provider_with_model_id() {
        let provider = MockEmbeddingProvider::new(32).with_model_id("my-custom-model");
        assert_eq!(provider.model_id(), "my-custom-model");
    }
}
