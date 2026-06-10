//! Types for the `prompt_templates` module.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemplateId(pub String);
impl TemplateId {
    /// Create a new template id.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    /// Return the string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl std::fmt::Display for TemplateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl From<&str> for TemplateId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}
impl From<String> for TemplateId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// A versioned prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Template identifier.
    pub id: TemplateId,
    /// Monotonic version number.
    pub version: u32,
    /// Template body.
    pub body: String,
    /// Variables required at render time.
    pub required_vars: Vec<String>,
    /// Human-readable description.
    pub description: String,
}
impl PromptTemplate {
    /// Create a new template.
    pub fn new(id: impl Into<TemplateId>, version: u32, body: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version,
            body: body.into(),
            required_vars: Vec::new(),
            description: String::new(),
        }
    }
    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }
    /// Add a required variable.
    #[must_use]
    pub fn with_required_var(mut self, v: impl Into<String>) -> Self {
        self.required_vars.push(v.into());
        self
    }
    /// Set all required variables.
    #[must_use]
    pub fn with_required_vars(mut self, v: Vec<String>) -> Self {
        self.required_vars = v;
        self
    }
}

/// Render context: variables and boolean flags.
#[derive(Debug, Clone, Default)]
pub struct RenderContext {
    /// String variables.
    pub vars: HashMap<String, String>,
    /// Boolean flags.
    pub flags: HashMap<String, bool>,
}
impl RenderContext {
    /// Create an empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a variable.
    #[must_use]
    pub fn with_var(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.vars.insert(k.into(), v.into());
        self
    }
    /// Add a flag.
    #[must_use]
    pub fn with_flag(mut self, k: impl Into<String>, v: bool) -> Self {
        self.flags.insert(k.into(), v);
        self
    }
    /// Get a variable.
    pub fn get_var(&self, k: &str) -> Option<&str> {
        self.vars.get(k).map(String::as_str)
    }
    /// Get a flag (default false when absent).
    #[must_use]
    pub fn get_flag(&self, k: &str) -> bool {
        *self.flags.get(k).unwrap_or(&false)
    }
}

/// Errors from the `prompt_templates` module.
#[derive(Debug, thiserror::Error)]
pub enum PromptTemplateError {
    /// Template not found.
    #[error("template not found: {0}")]
    TemplateNotFound(String),
    /// Missing variable.
    #[error("missing variable: {0}")]
    MissingVariable(String),
    /// Parse error.
    #[error("parse error: {0}")]
    ParseError(String),
    /// Unclosed block.
    #[error("unclosed block: {0}")]
    UnclosedBlock(String),
    /// Version not found.
    #[error("version not found: {0}")]
    VersionNotFound(u32),
    /// Unexpected closing tag.
    #[error("unexpected closing tag: {0}")]
    UnexpectedClosingTag(String),
    /// Empty template id.
    #[error("empty template id")]
    EmptyTemplateId,
}
