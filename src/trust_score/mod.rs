//! Composite trust scoring for RAG-generated answers.
pub mod scorer;
#[cfg(test)]
mod tests;
pub mod types;
pub use scorer::TrustScorer;
pub use types::{TrustComponents, TrustConfig, TrustError, TrustScore};
