//! Token-level context pruning (LLMLingua-style).
//!
//! Compresses a prompt by ranking individual **tokens** by importance and
//! dropping the least-important ones to hit a target compression ratio, then
//! reconstructing readable text in the original word order. This follows the
//! coarse-to-fine intuition of `LLMLingua` (Jiang et al., 2023): low-information
//! tokens — common stopwords and high-frequency filler — can be removed with
//! little loss of meaning, while rare, content-bearing, and query-relevant
//! tokens are retained.
//!
//! Unlike the sentence-extractive `context_compression` module, which selects
//! whole sentences, this module operates at the **token granularity**: it keeps
//! a fraction of the words within each passage.
//!
//! # Importance signal
//!
//! Each word's importance combines:
//!
//! | Factor | Effect |
//! |--------|--------|
//! | Inverse frequency | Rarer tokens within the context score higher |
//! | Content-ness | Stopwords are penalized |
//! | Query overlap | Tokens appearing in the query gain a bonus |
//! | Entity preservation | Capitalized entities are always kept (optional) |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "context-pruning")] {
//! use oxirag::prelude::*;
//!
//! let pruner = TokenPruner::new(PruneConfig::new().with_target_ratio(0.5));
//! let pruned = pruner.prune("the quick brown fox jumps", Some("fox")).unwrap();
//! assert!(pruned.kept_tokens <= 5);
//! # }
//! ```

pub mod pruner;
#[cfg(test)]
mod tests;
pub mod types;

pub use pruner::TokenPruner;
pub use types::{ContextPruningError, PruneConfig, PrunedContext};
