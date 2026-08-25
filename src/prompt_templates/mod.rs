//! Prompt template registry — placeholder (agents fill in).
pub mod engine;
pub mod registry;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;
pub use engine::TemplateEngine;
pub use registry::{PromptRegistry, builtin_templates};
pub use types::{PromptTemplate, PromptTemplateError, RenderContext, TemplateId};
