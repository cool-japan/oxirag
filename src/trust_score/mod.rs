//! Composite trust scoring for RAG-generated answers.
pub mod scorer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;
pub use scorer::TrustScorer;
pub use types::{TrustComponents, TrustConfig, TrustError, TrustScore};
