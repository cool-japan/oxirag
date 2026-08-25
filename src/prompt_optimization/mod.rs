//! Prompt optimization — few-shot demo selection and prompt variant scoring.
pub mod optimizer;
pub mod selector;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;
pub use optimizer::PromptOptimizer;
pub use types::{
    DemoPool, DemoSelectionStrategy, DemoSelector, Demonstration, DevExample, OutputScorer,
    PromptOptimizationConfig, PromptOptimizationError, PromptVariant, VariantScore,
};
