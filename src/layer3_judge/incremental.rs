//! Incremental consistency checking for logical claims.
//!
//! This module provides efficient incremental consistency checking that avoids
//! re-checking all claims on each addition. It maintains a knowledge base of
//! claims and can detect conflicts between them.

pub mod checker;
pub mod conflict;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use checker::IncrementalConsistencyChecker;
pub use types::{ClaimConflict, ConflictType, ConsistencyResult, Resolution};
