//! Lexical claim-support scoring for hallucination detection.
pub mod detector;
#[cfg(test)]
mod tests;
pub mod types;
pub use detector::HallucinationDetector;
pub use types::{ClaimSupport, HallucinationConfig, HallucinationError, HallucinationReport};
