//! Pairwise cross-encoder reranking for retrieval precision.
pub mod reranker;
pub mod scorer;
#[cfg(test)]
mod tests;
pub mod types;
pub use reranker::CrossEncoderReranker;
pub use scorer::LexicalCrossEncoder;
pub use types::CrossEncoderScorer;
pub use types::{
    CrossEncoderConfig, CrossEncoderError, FeatureWeights, InteractionFeatures, RerankedResult,
};
