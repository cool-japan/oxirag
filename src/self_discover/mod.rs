//! Self-Discover (Zhou et al. 2024) — structured self-discovery of reasoning
//! plans for complex queries.
//!
//! Self-Discover is a meta-reasoning framework that operates in three
//! sequential stages before answering a query:
//!
//! 1. **SELECT** — the model picks the most relevant *atomic reasoning
//!    modules* from a built-in bank (e.g., `"critical_thinking"`,
//!    `"decomposition"`, `"analogy"`).
//! 2. **ADAPT** — each selected module is rephrased to fit the specific task.
//! 3. **IMPLEMENT** — the adapted modules are assembled into a structured
//!    reasoning plan, which is then executed to produce the final answer.
//!
//! All three stages are simulated deterministically in pure Rust via the
//! [`SelfDiscoverModel`] trait. A built-in module bank of 8 named modules is
//! provided via [`SelfDiscoverEngine::built_in_modules`].
//!
//! # Example
//!
//! ```
//! use oxirag::self_discover::{
//!     MockSelfDiscoverModel, SelfDiscoverConfig, SelfDiscoverEngine,
//! };
//!
//! let config = SelfDiscoverConfig::default();
//! let engine = SelfDiscoverEngine::new(config, MockSelfDiscoverModel);
//!
//! let docs = vec![
//!     "Critical evaluation requires examining every assumption.".to_string(),
//!     "Decomposition makes large problems tractable.".to_string(),
//! ];
//! let result = engine
//!     .run("How should we evaluate complex arguments?", &docs)
//!     .unwrap();
//!
//! assert_eq!(result.query, "How should we evaluate complex arguments?");
//! assert!(!result.selected_modules.is_empty());
//! assert!(!result.reasoning_structure.plan.is_empty());
//! assert!(!result.answer.is_empty());
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::SelfDiscoverEngine;
pub use types::{
    MODULE_ANALOGY, MODULE_CAUSAL_REASONING, MODULE_COMPARATIVE_ANALYSIS, MODULE_CRITICAL_THINKING,
    MODULE_DECOMPOSITION, MODULE_DEDUCTIVE_REASONING, MODULE_HYPOTHESIS_TESTING,
    MODULE_STEP_BY_STEP, MockSelfDiscoverModel, ReasoningModule, ReasoningStructure,
    SelfDiscoverConfig, SelfDiscoverError, SelfDiscoverModel, SelfDiscoverResult,
};
