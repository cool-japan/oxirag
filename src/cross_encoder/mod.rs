//! Pairwise cross-encoder reranking for retrieval precision.
pub mod reranker;
pub mod scorer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;
pub use reranker::CrossEncoderReranker;
pub use scorer::LexicalCrossEncoder;
pub use types::CrossEncoderScorer;
pub use types::{
    CrossEncoderConfig, CrossEncoderError, FeatureWeights, InteractionFeatures, RerankedResult,
};
