//! Lost-in-the-Middle reordering — places highest-relevance docs at head and tail.
#[cfg(test)]
mod tests;
pub mod types;
pub use types::{
    LostInMiddleError, LostInMiddleReorderer, ReorderConfig, ReorderReport, ReorderStrategy,
};
