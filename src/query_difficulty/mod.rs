//! Query Difficulty Predictor: answerability estimation for adaptive routing.
//!
//! This module estimates how hard a natural-language query is to answer and
//! assigns it to one of four [`DifficultyBand`]s (`Easy`, `Medium`, `Hard`,
//! `Ambiguous`).  The resulting [`DifficultyScore`] includes a scalar score
//! in `[0.0, 1.0]` and the decomposed [`DifficultySignals`] that explain the
//! prediction, making it straightforward to use the output as a routing signal
//! for more advanced retrieval pipelines.
//!
//! ## Distinct from `adaptive_rag` and `query_router`
//!
//! | Module | Classifies | Maps to |
//! |--------|------------|---------|
//! | `query_router` | Semantic *intent* (factual / comparative / navigational) | Retrieval *modality* (vector / hybrid / graph) |
//! | `adaptive_rag` | Structural *complexity* (straightforward / single-step / multi-step) | Retrieval *depth* |
//! | **`query_difficulty`** | Predicted *answerability difficulty* (Easy → Ambiguous) | Routing priority or pipeline choice |
//!
//! ## Signal dimensions
//!
//! | Signal | Contribution |
//! |--------|-------------|
//! | Negation presence | [`DifficultyConfig::negation_weight`] (default 0.20) |
//! | Multi-hop indicators | [`DifficultyConfig::multi_hop_weight`] (default 0.30) |
//! | Ambiguity markers | [`DifficultyConfig::ambiguity_weight`] (default 0.20) |
//! | Normalised query length | [`DifficultyConfig::length_weight`] (default 0.15) |
//! | Rare-word ratio | [`DifficultyConfig::rare_word_weight`] (default 0.15) |
//!
//! ## Band thresholds
//!
//! | Band | Score range |
//! |------|-------------|
//! | [`DifficultyBand::Easy`] | `[0.0, 0.3)` |
//! | [`DifficultyBand::Medium`] | `[0.3, 0.55)` |
//! | [`DifficultyBand::Hard`] | `[0.55, 0.75)` |
//! | [`DifficultyBand::Ambiguous`] | `[0.75, 1.0]` |
//!
//! ## Example
//!
//! ```
//! use oxirag::query_difficulty::{DifficultyBand, DifficultyConfig, DifficultyPredictor};
//!
//! let predictor = DifficultyPredictor::new(DifficultyConfig::default());
//!
//! // A short, direct factoid question → Easy.
//! let result = predictor.predict("What is Rust?").unwrap();
//! assert_eq!(result.band, DifficultyBand::Easy);
//! assert!(result.score >= 0.0 && result.score <= 1.0);
//!
//! // The decomposed signals are always available.
//! assert!(!result.signals.has_negation);
//! assert!(!result.signals.multi_hop_indicator);
//! ```

pub mod predictor;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use predictor::DifficultyPredictor;
pub use types::{
    AMBIGUITY_MARKERS, DifficultyBand, DifficultyConfig, DifficultyError, DifficultyScore,
    DifficultySignals, MULTI_HOP_INDICATORS, NEGATION_WORDS, STOP_WORDS,
};
