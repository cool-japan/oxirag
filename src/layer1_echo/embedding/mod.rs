//! Embedding providers for the Echo layer.

mod cache;
mod candle;

pub use self::cache::{CacheStats, CachedEmbeddingProvider, EmbeddingCacheConfig};
pub use self::candle::{CandleDevice, CandleEmbeddingConfig, MockEmbeddingProvider};

#[cfg(feature = "speculator")]
pub use self::candle::CandleEmbeddingProvider;
