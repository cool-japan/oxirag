//! Tool trait and built-in tool implementations for the `agentic` module.

use std::collections::HashMap;

use async_trait::async_trait;

use super::types::AgenticError;

// ── Tool trait ────────────────────────────────────────────────────────────────

/// An async tool that the `ReAct` agent can invoke by name.
///
/// Implementations must be `Send + Sync` to work safely in the multi-threaded
/// agentic engine.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The unique name of this tool (used as the key in [`ToolRegistry`]).
    fn name(&self) -> &str;

    /// A one-line description of what this tool does (shown to the planner).
    fn description(&self) -> &str;

    /// Invoke the tool with the provided `input` string.
    ///
    /// # Errors
    ///
    /// Returns [`AgenticError::ToolFailed`] when the tool cannot process the input.
    async fn invoke(&self, input: &str) -> Result<String, AgenticError>;
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry that maps tool names to [`Tool`] implementations.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool, replacing any existing tool with the same name.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Look up a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    /// List all registered tool names and descriptions.
    #[must_use]
    pub fn list_tools(&self) -> Vec<(&str, &str)> {
        let mut list: Vec<_> = self
            .tools
            .values()
            .map(|t| (t.name(), t.description()))
            .collect();
        list.sort_by_key(|(n, _)| *n);
        list
    }

    /// Return `true` if no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Return the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }
}

// ── CalculatorTool ────────────────────────────────────────────────────────────

/// A tool that evaluates simple arithmetic expressions.
///
/// Supports `+`, `-`, `*`, `/` operators over integer and decimal operands.
/// Expressions must be of the form `<number> <op> <number>`.
pub struct CalculatorTool;

impl CalculatorTool {
    /// Create a new [`CalculatorTool`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &'static str {
        "calculator"
    }

    fn description(&self) -> &'static str {
        "Evaluates a simple arithmetic expression: <number> <op> <number> (op: +, -, *, /)"
    }

    async fn invoke(&self, input: &str) -> Result<String, AgenticError> {
        let parts: Vec<&str> = input.trim().splitn(3, ' ').collect();
        if parts.len() != 3 {
            return Err(AgenticError::ToolFailed(
                "expected format: <number> <op> <number>".to_string(),
            ));
        }
        let a: f64 = parts[0]
            .parse()
            .map_err(|_| AgenticError::ToolFailed(format!("invalid number: {}", parts[0])))?;
        let b: f64 = parts[2]
            .parse()
            .map_err(|_| AgenticError::ToolFailed(format!("invalid number: {}", parts[2])))?;
        let result = match parts[1] {
            "+" => a + b,
            "-" => a - b,
            "*" => a * b,
            "/" => {
                if b == 0.0 {
                    return Err(AgenticError::ToolFailed("division by zero".to_string()));
                }
                a / b
            }
            op => {
                return Err(AgenticError::ToolFailed(format!("unknown operator: {op}")));
            }
        };
        Ok(result.to_string())
    }
}

// ── LookupTool ────────────────────────────────────────────────────────────────

/// A tool that looks up a key in a static key-value table.
pub struct LookupTool {
    table: HashMap<String, String>,
}

impl LookupTool {
    /// Create a new [`LookupTool`] from a key-value map.
    #[must_use]
    pub fn new(table: HashMap<String, String>) -> Self {
        Self { table }
    }
}

#[async_trait]
impl Tool for LookupTool {
    fn name(&self) -> &'static str {
        "lookup"
    }

    fn description(&self) -> &'static str {
        "Looks up a key in the knowledge table and returns its value"
    }

    async fn invoke(&self, input: &str) -> Result<String, AgenticError> {
        let key = input.trim().to_lowercase();
        self.table
            .get(&key)
            .cloned()
            .ok_or_else(|| AgenticError::ToolFailed(format!("key not found: {input}")))
    }
}

// ── MockTool ──────────────────────────────────────────────────────────────────

/// Scripted tool for unit tests.
///
/// Always returns the same configured output string.
pub struct MockTool {
    /// The name this tool will report.
    pub tool_name: String,
    /// Description reported by this tool.
    pub tool_desc: String,
    /// The fixed output returned by every invocation.
    pub output: String,
}

impl MockTool {
    /// Create a mock tool with the given name and fixed output.
    #[must_use]
    pub fn new(name: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            tool_name: name.into(),
            tool_desc: "mock tool".to_string(),
            output: output.into(),
        }
    }
}

#[async_trait]
impl Tool for MockTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_desc
    }

    async fn invoke(&self, _input: &str) -> Result<String, AgenticError> {
        Ok(self.output.clone())
    }
}
