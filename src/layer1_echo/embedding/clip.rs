//! CLIP-based multi-modal embedding provider using `candle-transformers`.
//!
//! # Feature Flag
//! Only compiled with the `multimodal` feature (implies `speculator`).
//!
//! # Supported models
//! | Preset | HF repo | Joint dim |
//! |--------|---------|-----------|
//! | `ClipPreset::VitBase32` (default) | `openai/clip-vit-base-patch32` | 512 |
//! | `ClipPreset::VitLarge14` | `openai/clip-vit-large-patch14` | 768 |
//! | `ClipPreset::VitLarge14_336` | `openai/clip-vit-large-patch14-336` | 768 |
//!
//! # Architecture
//! The CLIP model encodes text and images into a shared embedding space via projection
//! heads. For joint text+image queries the two normalised projections are averaged,
//! giving a centroid that is itself re-normalised to the unit hypersphere.
//!
//! # Loading
//! Model weights are loaded from the HuggingFace Hub via `hf-hub`. The first call
//! triggers a download; subsequent calls use the local cache in `~/.cache/huggingface`.

#![cfg(all(feature = "multimodal", not(target_arch = "wasm32")))]
#![allow(clippy::doc_markdown)]

use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{
    ClipConfig, ClipModel, text_model::ClipTextConfig, vision_model::ClipVisionConfig,
};
use hf_hub::{Repo, RepoType, api::sync::Api};
use image::{DynamicImage, ImageReader, imageops::FilterType};
use tokenizers::Tokenizer;

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::{EmbeddingInput, MultiModalEmbeddingProvider};

// ─────────────────────────────────────────────────────────────────────────────
// Preset
// ─────────────────────────────────────────────────────────────────────────────

/// Well-known CLIP model presets available on HuggingFace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClipPreset {
    /// `openai/clip-vit-base-patch32` — 512-dimensional joint embedding. Default.
    #[default]
    VitBase32,
    /// `openai/clip-vit-large-patch14` — 768-dimensional joint embedding.
    VitLarge14,
    /// `openai/clip-vit-large-patch14-336` — 768-dimensional joint embedding,
    /// higher-resolution (336 px) vision encoder.
    VitLarge14_336,
}

impl ClipPreset {
    /// HuggingFace repository identifier for this preset.
    #[must_use]
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::VitBase32 => "openai/clip-vit-base-patch32",
            Self::VitLarge14 => "openai/clip-vit-large-patch14",
            Self::VitLarge14_336 => "openai/clip-vit-large-patch14-336",
        }
    }

    /// Output dimensionality of the joint embedding space (projection head output).
    #[must_use]
    pub fn dimension(self) -> usize {
        match self {
            Self::VitBase32 => 512,
            // Both large variants project to 768-dimensional space
            Self::VitLarge14 | Self::VitLarge14_336 => 768,
        }
    }

    /// Expected input image size (square side length in pixels).
    #[must_use]
    pub fn image_size(self) -> usize {
        match self {
            Self::VitBase32 | Self::VitLarge14 => 224,
            Self::VitLarge14_336 => 336,
        }
    }

    /// Build the `candle_transformers` [`ClipConfig`] for this preset.
    #[must_use]
    pub(crate) fn clip_config(self) -> ClipConfig {
        match self {
            Self::VitBase32 => ClipConfig::vit_base_patch32(),
            Self::VitLarge14 => {
                // clip-vit-large-patch14 @ 224 px — manually constructed because
                // candle-transformers only ships the 336-px factory method.
                let text_config = ClipTextConfig {
                    vocab_size: 49408,
                    embed_dim: 768,
                    intermediate_size: 3072,
                    max_position_embeddings: 77,
                    pad_with: None,
                    num_hidden_layers: 12,
                    num_attention_heads: 12,
                    projection_dim: 768,
                    activation:
                        candle_transformers::models::clip::text_model::Activation::QuickGelu,
                };
                let vision_config = ClipVisionConfig {
                    embed_dim: 1024,
                    activation:
                        candle_transformers::models::clip::text_model::Activation::QuickGelu,
                    intermediate_size: 4096,
                    num_hidden_layers: 24,
                    num_attention_heads: 16,
                    projection_dim: 768,
                    num_channels: 3,
                    image_size: 224,
                    patch_size: 14,
                };
                ClipConfig {
                    text_config,
                    vision_config,
                    logit_scale_init_value: 2.6592,
                    image_size: 224,
                }
            }
            Self::VitLarge14_336 => {
                // Use the 336-px vision config from candle-transformers, but pair it
                // with a matching text config.
                let text_config = ClipTextConfig {
                    vocab_size: 49408,
                    embed_dim: 768,
                    intermediate_size: 3072,
                    max_position_embeddings: 77,
                    pad_with: None,
                    num_hidden_layers: 12,
                    num_attention_heads: 12,
                    projection_dim: 768,
                    activation:
                        candle_transformers::models::clip::text_model::Activation::QuickGelu,
                };
                let vision_config = ClipVisionConfig::clip_vit_large_patch14_336();
                ClipConfig {
                    text_config,
                    vision_config,
                    logit_scale_init_value: 2.6592,
                    image_size: 336,
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Image normalisation constants (ImageNet mean / std used by OpenAI CLIP)
// ─────────────────────────────────────────────────────────────────────────────

/// ImageNet RGB channel means used by OpenAI CLIP preprocessing.
const IMAGE_MEAN: [f32; 3] = [0.481_456_1, 0.457_827_5, 0.408_211];
/// ImageNet RGB channel standard deviations used by OpenAI CLIP preprocessing.
const IMAGE_STD: [f32; 3] = [0.268_629_5, 0.261_303, 0.275_777_1];

// ─────────────────────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────────────────────

/// CLIP-based multi-modal embedding provider.
///
/// Wraps a loaded `candle_transformers` CLIP model and provides the
/// [`MultiModalEmbeddingProvider`] interface so it can be used as a drop-in
/// embedding back-end inside the Echo layer.
///
/// # Construction
///
/// ```rust,ignore
/// use oxirag::layer1_echo::embedding::{CandleClipProvider, ClipPreset};
///
/// let provider = CandleClipProvider::new(ClipPreset::VitBase32)?;
/// ```
pub struct CandleClipProvider {
    /// Loaded CLIP model (owns both the text and vision encoders + projection heads).
    model: ClipModel,
    /// CLIP tokenizer (BPE, 49408-token vocabulary).
    tokenizer: Tokenizer,
    /// Candle compute device (CPU in the default configuration).
    device: Device,
    /// Output dimension of the joint embedding space.
    dim: usize,
    /// HuggingFace model identifier stored for the `model_id()` accessor.
    model_id: String,
    /// Maximum sequence length for text tokenisation (77 tokens for CLIP BPE).
    max_seq_len: usize,
    /// Target image size (square, in pixels) for preprocessing.
    image_size: usize,
}

impl CandleClipProvider {
    /// Create a `CandleClipProvider` from a well-known [`ClipPreset`].
    ///
    /// Downloads the model from HuggingFace Hub on first use (cached locally
    /// at `~/.cache/huggingface`).
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::ModelLoad`] if the HF download, tokenizer
    /// parsing, or weight-loading fails.
    pub fn new(preset: ClipPreset) -> Result<Self, EmbeddingError> {
        Self::with_model_id(preset.repo_id(), preset.dimension(), preset)
    }

    /// Create a `CandleClipProvider` with a custom HuggingFace repository and
    /// explicit output dimension.  The `preset` controls which architecture
    /// configuration is built; pass `ClipPreset::VitBase32` for the default
    /// architecture when using a fine-tuned variant of the same size.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::ModelLoad`] if any download or loading step fails.
    pub fn with_model_id(
        repo: &str,
        dim: usize,
        preset: ClipPreset,
    ) -> Result<Self, EmbeddingError> {
        let device = Device::Cpu;
        let api = Api::new().map_err(|e| EmbeddingError::ModelLoad(e.to_string()))?;

        let hf_repo = api.repo(Repo::with_revision(
            repo.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        // ── Tokenizer ──────────────────────────────────────────────────────
        let tokenizer_path = hf_repo
            .get("tokenizer.json")
            .map_err(|e| EmbeddingError::ModelLoad(format!("tokenizer.json: {e}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbeddingError::Tokenization(e.to_string()))?;

        // ── Weights ────────────────────────────────────────────────────────
        let weights_path = hf_repo
            .get("model.safetensors")
            .or_else(|_| hf_repo.get("pytorch_model.bin"))
            .map_err(|e| EmbeddingError::ModelLoad(format!("model weights: {e}")))?;

        let clip_config = preset.clip_config();

        let vb = if weights_path
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            // SAFETY: mmap is valid for the duration of `vb` and the model built from it.
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| EmbeddingError::ModelLoad(format!("safetensors mmap: {e}")))?
            }
        } else {
            VarBuilder::from_pth(weights_path, DType::F32, &device)
                .map_err(|e| EmbeddingError::ModelLoad(format!("pth load: {e}")))?
        };

        // ── Model ──────────────────────────────────────────────────────────
        let model = ClipModel::new(vb, &clip_config)
            .map_err(|e| EmbeddingError::ModelLoad(format!("ClipModel::new: {e}")))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
            model_id: repo.to_string(),
            max_seq_len: clip_config.text_config.max_position_embeddings,
            image_size: clip_config.image_size,
        })
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Tokenise `text` into a `[1, seq_len]` `u32` tensor ready for the CLIP
    /// text encoder.  Sequences longer than `max_seq_len` are truncated.
    fn tokenise(&self, text: &str) -> Result<Tensor, EmbeddingError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbeddingError::Tokenization(e.to_string()))?;

        let ids = encoding.get_ids();
        let seq_len = ids.len().min(self.max_seq_len);
        let ids_u32: Vec<u32> = ids[..seq_len].to_vec();

        Tensor::from_vec(ids_u32, (1, seq_len), &self.device)
            .map_err(|e| EmbeddingError::Inference(format!("tokenise tensor: {e}")))
    }

    /// Decode `bytes` as an image (any format the `image` crate supports),
    /// resize to `image_size × image_size`, and produce a normalised
    /// `[1, 3, H, W]` `f32` tensor suitable for the CLIP vision encoder.
    fn preprocess_image(&self, bytes: &[u8]) -> Result<Tensor, EmbeddingError> {
        // ── Decode ─────────────────────────────────────────────────────────
        let img: DynamicImage = {
            let cursor = std::io::Cursor::new(bytes);
            let reader = ImageReader::new(cursor)
                .with_guessed_format()
                .map_err(|e| EmbeddingError::Inference(format!("image format guess: {e}")))?;
            reader
                .decode()
                .map_err(|e| EmbeddingError::Inference(format!("image decode: {e}")))?
        };

        // ── Resize to model's expected square ──────────────────────────────
        let sz = u32::try_from(self.image_size)
            .map_err(|e| EmbeddingError::Inference(format!("image_size overflow: {e}")))?;
        let rgb = img.resize_exact(sz, sz, FilterType::Lanczos3).into_rgb8();

        // ── Convert to f32 in [0, 1], then normalise per-channel ───────────
        let h = rgb.height() as usize;
        let w = rgb.width() as usize;
        let raw: Vec<u8> = rgb.into_raw(); // HWC, u8

        // Reorder to CHW and apply per-channel mean/std normalisation
        let mut chw = vec![0_f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let base_hwc = (y * w + x) * 3;
                for c in 0..3_usize {
                    let val = f32::from(raw[base_hwc + c]) / 255.0;
                    chw[c * h * w + y * w + x] = (val - IMAGE_MEAN[c]) / IMAGE_STD[c];
                }
            }
        }

        // Build [1, 3, H, W] tensor
        Tensor::from_vec(chw, (1_usize, 3_usize, h, w), &self.device)
            .map_err(|e| EmbeddingError::Inference(format!("pixel tensor: {e}")))
    }

    /// Apply L2-normalisation to a `[1, D]` tensor and return `Vec<f32>`.
    fn l2_normalise_to_vec(features: &Tensor) -> Result<Vec<f32>, EmbeddingError> {
        // features: [1, D]
        let norm = features
            .sqr()
            .map_err(|e| EmbeddingError::Inference(format!("sqr: {e}")))?
            .sum_keepdim(candle_core::D::Minus1)
            .map_err(|e| EmbeddingError::Inference(format!("sum_keepdim: {e}")))?
            .sqrt()
            .map_err(|e| EmbeddingError::Inference(format!("sqrt: {e}")))?
            .clamp(1e-9_f64, f64::MAX)
            .map_err(|e| EmbeddingError::Inference(format!("clamp: {e}")))?;

        let normalised = features
            .broadcast_div(&norm)
            .map_err(|e| EmbeddingError::Inference(format!("broadcast_div: {e}")))?;

        normalised
            .squeeze(0)
            .map_err(|e| EmbeddingError::Inference(format!("squeeze: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| EmbeddingError::Inference(format!("to_vec1: {e}")))
    }

    /// Embed a single text string.
    fn embed_text_inner(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let input_ids = self.tokenise(text)?;
        let features = self
            .model
            .get_text_features(&input_ids)
            .map_err(|e| EmbeddingError::Inference(format!("get_text_features: {e}")))?;
        // features is [1, D] after the projection head
        Self::l2_normalise_to_vec(&features)
    }

    /// Embed a single image given its raw bytes.
    fn embed_image_inner(&self, bytes: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
        let pixel_values = self.preprocess_image(bytes)?;
        let features = self
            .model
            .get_image_features(&pixel_values)
            .map_err(|e| EmbeddingError::Inference(format!("get_image_features: {e}")))?;
        // features is [1, D] after the visual projection head
        Self::l2_normalise_to_vec(&features)
    }

    /// Embed a text+image pair by averaging the two unit-normalised feature
    /// vectors and then re-normalising the result.
    fn embed_joint_inner(&self, text: &str, image: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
        let text_vec = self.embed_text_inner(text)?;
        let image_vec = self.embed_image_inner(image)?;

        // Element-wise average
        let avg: Vec<f32> = text_vec
            .iter()
            .zip(image_vec.iter())
            .map(|(t, i)| (t + i) * 0.5_f32)
            .collect();

        // Re-normalise to unit sphere
        let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-9 {
            // Degenerate case: both embeddings cancelled; return the text embedding
            return Ok(text_vec);
        }
        Ok(avg.into_iter().map(|x| x / norm).collect())
    }

    /// Extract [CLS] token from a raw text-transformer output `[batch, seq, dim]`
    /// at the position of the highest token id (CLIP's `eos` convention), then
    /// squeeze to `[batch, dim]`.
    ///
    /// This is not used in normal inference (the projection head already applies
    /// this logic internally via `Module`), but exposed for debugging.
    #[allow(dead_code)]
    fn extract_eos(output: &Tensor, input_ids: &Tensor) -> Result<Tensor, EmbeddingError> {
        let indices = input_ids
            .argmax(candle_core::D::Minus1)
            .map_err(|e| EmbeddingError::Inference(format!("argmax: {e}")))?
            .to_dtype(DType::I64)
            .map_err(|e| EmbeddingError::Inference(format!("to_dtype i64: {e}")))?;

        let batch = indices
            .dims()
            .first()
            .copied()
            .ok_or_else(|| EmbeddingError::Inference("empty indices".to_string()))?;

        let mut parts = Vec::with_capacity(batch);
        for b in 0..batch {
            let seq_idx_i64 = indices
                .i(b)
                .map_err(|e| EmbeddingError::Inference(format!("index {b}: {e}")))?
                .to_scalar::<i64>()
                .map_err(|e| EmbeddingError::Inference(format!("to_scalar: {e}")))?;
            let seq_idx = usize::try_from(seq_idx_i64)
                .map_err(|e| EmbeddingError::Inference(format!("seq_idx {seq_idx_i64}: {e}")))?;
            parts.push(
                output
                    .i((b, seq_idx))
                    .map_err(|e| EmbeddingError::Inference(format!("i({b},{seq_idx}): {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| EmbeddingError::Inference(format!("unsqueeze: {e}")))?,
            );
        }
        Tensor::cat(&parts, 0).map_err(|e| EmbeddingError::Inference(format!("cat: {e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait implementation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl MultiModalEmbeddingProvider for CandleClipProvider {
    async fn embed_multi(&self, input: EmbeddingInput<'_>) -> Result<Vec<f32>, EmbeddingError> {
        match input {
            EmbeddingInput::Text(text) => self.embed_text_inner(text),
            EmbeddingInput::Image(bytes) => self.embed_image_inner(bytes),
            EmbeddingInput::TextAndImage { text, image } => self.embed_joint_inner(text, image),
        }
    }

    async fn embed_multi_batch(
        &self,
        inputs: &[EmbeddingInput<'_>],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Candle's CLIP text-path does support batching, but the vision encoder
        // and the joint path do not share a common batch size constraint.  We
        // process each input independently to keep the implementation correct
        // and simple; performance-critical callers should pre-batch at a higher
        // abstraction level.
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.embed_multi(input.clone()).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    // ── Static / compile-time tests ─────────────────────────────────────────

    #[test]
    fn test_clip_preset_variants_exist() {
        let _ = ClipPreset::VitBase32;
        let _ = ClipPreset::VitLarge14;
    }

    #[test]
    fn test_clip_preset_dimensions() {
        assert_eq!(ClipPreset::VitBase32.dimension(), 512);
        assert_eq!(ClipPreset::VitLarge14.dimension(), 768);
    }

    #[test]
    fn test_clip_preset_large_336_dimension() {
        assert_eq!(ClipPreset::VitLarge14_336.dimension(), 768);
    }

    #[test]
    fn test_clip_preset_repo_ids() {
        let repo = ClipPreset::VitBase32.repo_id();
        assert!(
            repo.contains("clip-vit-base-patch32"),
            "expected 'clip-vit-base-patch32' in repo id, got: {repo}"
        );
    }

    #[test]
    fn test_clip_preset_repo_id_large14() {
        let repo = ClipPreset::VitLarge14.repo_id();
        assert!(
            repo.contains("clip-vit-large-patch14"),
            "expected 'clip-vit-large-patch14' in repo id, got: {repo}"
        );
    }

    #[test]
    fn test_clip_preset_repo_id_large14_336() {
        let repo = ClipPreset::VitLarge14_336.repo_id();
        assert!(
            repo.contains("clip-vit-large-patch14-336"),
            "expected '336' in repo id, got: {repo}"
        );
    }

    #[test]
    fn test_clip_preset_image_sizes() {
        assert_eq!(ClipPreset::VitBase32.image_size(), 224);
        assert_eq!(ClipPreset::VitLarge14.image_size(), 224);
        assert_eq!(ClipPreset::VitLarge14_336.image_size(), 336);
    }

    #[test]
    fn test_clip_preset_default_is_vitbase32() {
        let p = ClipPreset::default();
        assert_eq!(p, ClipPreset::VitBase32);
    }

    #[test]
    fn test_clip_config_base32_image_size() {
        let cfg = ClipPreset::VitBase32.clip_config();
        assert_eq!(cfg.image_size, 224);
    }

    #[test]
    fn test_clip_config_large14_image_size() {
        let cfg = ClipPreset::VitLarge14.clip_config();
        assert_eq!(cfg.image_size, 224);
    }

    #[test]
    fn test_clip_config_large14_336_image_size() {
        let cfg = ClipPreset::VitLarge14_336.clip_config();
        assert_eq!(cfg.image_size, 336);
    }

    #[test]
    fn test_clip_config_text_projection_dim_base32() {
        let cfg = ClipPreset::VitBase32.clip_config();
        assert_eq!(cfg.text_config.projection_dim, 512);
    }

    #[test]
    fn test_clip_config_text_projection_dim_large14() {
        let cfg = ClipPreset::VitLarge14.clip_config();
        assert_eq!(cfg.text_config.projection_dim, 768);
    }

    // ── EmbeddingInput clone ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_embedding_input_clone() {
        let input = EmbeddingInput::Text("hello");
        let _cloned = input.clone();
    }

    #[tokio::test]
    async fn test_embedding_input_image_clone() {
        let bytes = [0u8, 1, 2, 3];
        let input = EmbeddingInput::Image(&bytes);
        let _cloned = input.clone();
    }

    #[tokio::test]
    async fn test_embedding_input_joint_clone() {
        let bytes = [0u8, 1, 2];
        let input = EmbeddingInput::TextAndImage {
            text: "hello",
            image: &bytes,
        };
        let _cloned = input.clone();
    }

    // ── IMAGE_MEAN / IMAGE_STD sanity ───────────────────────────────────────

    #[test]
    fn test_image_normalisation_constants_are_in_range() {
        for m in IMAGE_MEAN {
            assert!(m > 0.0 && m < 1.0, "IMAGE_MEAN out of (0,1): {m}");
        }
        for s in IMAGE_STD {
            assert!(s > 0.0 && s < 1.0, "IMAGE_STD out of (0,1): {s}");
        }
    }

    // ── Network-dependent tests — skipped unless OXIRAG_TEST_DOWNLOADS=1 ───

    #[tokio::test]
    async fn test_candle_clip_loads_if_download_enabled() {
        if std::env::var("OXIRAG_TEST_DOWNLOADS").is_err() {
            return; // Skip in CI
        }
        let provider =
            CandleClipProvider::new(ClipPreset::VitBase32).expect("should load clip model");
        assert_eq!(provider.dimension(), 512);

        let emb = provider
            .embed_multi(EmbeddingInput::Text("a photo of a cat"))
            .await
            .expect("should embed text");
        assert_eq!(emb.len(), 512);

        // L2 norm should be ~1.0
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected unit vector, got norm={norm}"
        );
    }

    #[tokio::test]
    async fn test_candle_clip_image_embedding_if_download_enabled() {
        if std::env::var("OXIRAG_TEST_DOWNLOADS").is_err() {
            return;
        }
        let provider =
            CandleClipProvider::new(ClipPreset::VitBase32).expect("should load clip model");

        // Create a minimal valid PNG (1×1 red pixel) to test the image path.
        let mut png_bytes = Vec::new();
        {
            // PNG header + IHDR + IDAT + IEND
            let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200u8, 100, 50]));
            let dyn_img = DynamicImage::ImageRgb8(img);
            let mut cursor = std::io::Cursor::new(&mut png_bytes);
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Png)
                .expect("write png");
        }

        let emb = provider
            .embed_multi(EmbeddingInput::Image(&png_bytes))
            .await
            .expect("should embed image");
        assert_eq!(emb.len(), 512);

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected unit image vector, got norm={norm}"
        );
    }

    #[tokio::test]
    async fn test_candle_clip_joint_embedding_if_download_enabled() {
        if std::env::var("OXIRAG_TEST_DOWNLOADS").is_err() {
            return;
        }
        let provider =
            CandleClipProvider::new(ClipPreset::VitBase32).expect("should load clip model");

        let mut png_bytes = Vec::new();
        {
            let img = image::RgbImage::from_pixel(4, 4, image::Rgb([100u8, 150, 200]));
            let dyn_img = DynamicImage::ImageRgb8(img);
            let mut cursor = std::io::Cursor::new(&mut png_bytes);
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Png)
                .expect("write png");
        }

        let emb = provider
            .embed_multi(EmbeddingInput::TextAndImage {
                text: "a blue object",
                image: &png_bytes,
            })
            .await
            .expect("should embed text+image");
        assert_eq!(emb.len(), 512);

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected unit joint vector, got norm={norm}"
        );
    }
}
