//! Prompt registry (stub — agents fill in).
use super::engine::TemplateEngine;
use super::types::{PromptTemplate, PromptTemplateError, RenderContext, TemplateId};
use std::collections::HashMap;

/// Stable id for the answer-synthesis built-in.
pub const TEMPLATE_ANSWER_SYNTHESIS: &str = "answer-synthesis";
/// Stable id for the query-rewrite built-in.
pub const TEMPLATE_QUERY_REWRITE: &str = "query-rewrite";
/// Stable id for the doc-grading built-in.
pub const TEMPLATE_DOC_GRADING: &str = "doc-grading";
/// Stable id for the citation built-in.
pub const TEMPLATE_CITATION: &str = "citation";

/// Return the four built-in templates.
#[must_use]
pub fn builtin_templates() -> Vec<PromptTemplate> {
    vec![
        PromptTemplate::new(TEMPLATE_ANSWER_SYNTHESIS, 1,
            "Use the following documents to answer the question.\n\nDocuments:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:")
            .with_required_vars(vec!["context".to_string(), "question".to_string()])
            .with_description("Synthesize an answer from retrieved documents"),
        PromptTemplate::new(TEMPLATE_QUERY_REWRITE, 1,
            "Rewrite the following query to be more specific and clear.\n\nOriginal query: {{query}}\n\nRewritten query:")
            .with_required_var("query")
            .with_description("Rewrite a query for better retrieval"),
        PromptTemplate::new(TEMPLATE_DOC_GRADING, 1,
            "Grade the relevance of the following document to the query.\n\nQuery: {{query}}\n\nDocument: {{document}}\n\nRelevance (0-1):")
            .with_required_vars(vec!["query".to_string(), "document".to_string()])
            .with_description("Grade document relevance to a query"),
        PromptTemplate::new(TEMPLATE_CITATION, 1,
            "Add inline citations to the following answer based on the provided sources.\n\nAnswer: {{answer}}\n\nSources:\n{{sources}}\n\nAnswer with citations:")
            .with_required_vars(vec!["answer".to_string(), "sources".to_string()])
            .with_description("Add inline citations to a generated answer"),
    ]
}

/// Versioned prompt-template registry.
#[derive(Debug, Default)]
pub struct PromptRegistry {
    templates: HashMap<TemplateId, Vec<PromptTemplate>>,
}

impl PromptRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-loaded with the four built-in RAG templates.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        for t in builtin_templates() {
            let _ = r.register(t);
        }
        r
    }

    /// Register a template. Overwrites an existing (id, version) pair.
    ///
    /// # Errors
    /// Returns [`PromptTemplateError::EmptyTemplateId`] when the id is empty.
    pub fn register(&mut self, t: PromptTemplate) -> Result<(), PromptTemplateError> {
        if t.id.as_str().is_empty() {
            return Err(PromptTemplateError::EmptyTemplateId);
        }
        let versions = self.templates.entry(t.id.clone()).or_default();
        if let Some(pos) = versions.iter().position(|v| v.version == t.version) {
            versions[pos] = t;
        } else {
            versions.push(t);
            versions.sort_by_key(|v| v.version);
        }
        Ok(())
    }

    /// Get the latest version of a template.
    ///
    /// # Errors
    /// Returns [`PromptTemplateError::TemplateNotFound`] when the id is unknown.
    pub fn get_latest(&self, id: &TemplateId) -> Result<&PromptTemplate, PromptTemplateError> {
        self.templates
            .get(id)
            .and_then(|v| v.last())
            .ok_or_else(|| PromptTemplateError::TemplateNotFound(id.as_str().to_string()))
    }

    /// Get a specific version of a template.
    ///
    /// # Errors
    /// Returns [`PromptTemplateError::TemplateNotFound`] or [`PromptTemplateError::VersionNotFound`].
    pub fn get_version(
        &self,
        id: &TemplateId,
        version: u32,
    ) -> Result<&PromptTemplate, PromptTemplateError> {
        let versions = self
            .templates
            .get(id)
            .ok_or_else(|| PromptTemplateError::TemplateNotFound(id.as_str().to_string()))?;
        versions
            .iter()
            .find(|v| v.version == version)
            .ok_or(PromptTemplateError::VersionNotFound(version))
    }

    /// List all versions of a template.
    ///
    /// # Errors
    /// Returns [`PromptTemplateError::TemplateNotFound`] when the id is unknown.
    pub fn versions(&self, id: &TemplateId) -> Result<Vec<u32>, PromptTemplateError> {
        self.templates
            .get(id)
            .map(|v| v.iter().map(|t| t.version).collect())
            .ok_or_else(|| PromptTemplateError::TemplateNotFound(id.as_str().to_string()))
    }

    /// Render the latest version of a template.
    ///
    /// # Errors
    /// Propagates [`PromptTemplateError`] from lookup or render.
    pub fn render_latest(
        &self,
        id: &TemplateId,
        ctx: &RenderContext,
    ) -> Result<String, PromptTemplateError> {
        TemplateEngine::render(self.get_latest(id)?, ctx)
    }

    /// Return all registered template ids.
    #[must_use]
    pub fn ids(&self) -> Vec<&TemplateId> {
        self.templates.keys().collect()
    }

    /// Number of distinct template ids.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// True if no templates are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}
