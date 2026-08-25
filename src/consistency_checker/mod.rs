//! Cross-claim consistency detection within generated answers.
pub mod checker;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;
pub use checker::ConsistencyChecker;
pub use types::{
    ConflictType, ConsistencyConfig, ConsistencyError, ConsistencyReport, Inconsistency,
};
