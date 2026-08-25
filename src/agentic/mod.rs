//! `ReAct` Thought→Action→Observation agentic RAG loop.
//!
//! Based on Yao et al. 2022 "`ReAct`: Synergizing Reasoning and Acting in Language Models".
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`Tool`] | Async callable invoked by the agent |
//! | [`ToolRegistry`] | Maps tool names to implementations |
//! | [`CalculatorTool`] | Built-in arithmetic tool |
//! | [`LookupTool`] | Built-in static key-value lookup |
//! | [`MockTool`] | Scripted test double |
//! | [`ReActAgent`] | `ReAct` loop: heuristic thought → action → observe |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "agentic")] {
//! use oxirag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let mut registry = ToolRegistry::new();
//! registry.register(Box::new(CalculatorTool::new()));
//! let agent = ReActAgent::new(registry, AgenticConfig::default());
//! # }
//! # }
//! ```

pub mod agent;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod tool;
pub mod types;

pub use agent::ReActAgent;
pub use tool::{CalculatorTool, LookupTool, MockTool, Tool, ToolRegistry};
pub use types::{AgentAction, AgentStep, AgentTrace, AgenticConfig, AgenticError};
