//! Embedding providers for the Echo layer.

mod cache;
mod candle;
#[cfg(all(feature = "multimodal", not(target_arch = "wasm32")))]
pub mod clip;

pub use self::cache::{CacheStats, CachedEmbeddingProvider, EmbeddingCacheConfig};
pub use self::candle::{CandleDevice, CandleEmbeddingConfig, MockEmbeddingProvider};

#[cfg(feature = "speculator")]
pub use self::candle::CandleEmbeddingProvider;

#[cfg(all(feature = "multimodal", not(target_arch = "wasm32")))]
pub use self::clip::{CandleClipProvider, ClipPreset};
