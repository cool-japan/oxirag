//! Types for the `agentic` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── AgentAction ───────────────────────────────────────────────────────────────

/// An action chosen by the `ReAct` agent during one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentAction {
    /// Search the knowledge base for relevant documents.
    Search(String),
    /// Invoke a registered tool by name with the given input string.
    UseTool {
        /// The name of the tool to invoke.
        name: String,
        /// The input string passed to the tool.
        input: String,
    },
    /// Finish the task with a final answer.
    Finish(String),
}

// ── AgentStep ─────────────────────────────────────────────────────────────────

/// One step in the `ReAct` agent trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    /// The agent's thought for this step.
    pub thought: String,
    /// The action chosen.
    pub action: AgentAction,
    /// The observation returned after executing the action.
    pub observation: String,
    /// Zero-based step index.
    pub step_index: usize,
}

// ── AgentTrace ────────────────────────────────────────────────────────────────

/// Complete trace of an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrace {
    /// Ordered list of steps taken.
    pub steps: Vec<AgentStep>,
    /// Final answer extracted from the last Finish action (empty if truncated).
    pub final_answer: String,
    /// Whether the agent reached its step limit without finishing.
    pub truncated: bool,
}

impl AgentTrace {
    /// Create an empty trace.
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            final_answer: String::new(),
            truncated: false,
        }
    }

    /// Return the number of steps taken.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Return `true` if no steps were taken.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl Default for AgentTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ── AgenticConfig ─────────────────────────────────────────────────────────────

/// Configuration for the [`crate::agentic::ReActAgent`].
#[derive(Debug, Clone)]
pub struct AgenticConfig {
    /// Maximum number of steps before the agent is forced to finish.
    ///
    /// Defaults to `6`.
    pub max_steps: usize,
    /// Number of documents to retrieve per Search action.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
    /// Whether to include search results in the observation text.
    ///
    /// Defaults to `true`.
    pub include_snippets: bool,
}

impl Default for AgenticConfig {
    fn default() -> Self {
        Self {
            max_steps: 6,
            top_k: 5,
            include_snippets: true,
        }
    }
}

impl AgenticConfig {
    /// Set the maximum number of steps.
    #[must_use]
    pub fn with_max_steps(mut self, v: usize) -> Self {
        self.max_steps = v;
        self
    }

    /// Set the retrieval top-k.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }

    /// Set whether to include snippets in observations.
    #[must_use]
    pub fn with_include_snippets(mut self, v: bool) -> Self {
        self.include_snippets = v;
        self
    }
}

// ── AgenticError ──────────────────────────────────────────────────────────────

/// Errors from the `agentic` module.
#[derive(Debug, Error)]
pub enum AgenticError {
    /// The query string was empty after trimming.
    #[error("Query must not be empty")]
    EmptyQuery,

    /// A tool named `{0}` is not registered.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// A tool invocation failed.
    #[error("Tool failed: {0}")]
    ToolFailed(String),

    /// The agent reached its maximum step limit without finishing.
    #[error("Maximum steps exceeded")]
    MaxStepsExceeded,

    /// The retrieval step failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
}
