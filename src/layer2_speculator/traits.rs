//! Traits for the Speculator (draft verification) layer.

use async_trait::async_trait;

use crate::error::SpeculatorError;
use crate::types::{Draft, SearchResult, SpeculationDecision, SpeculationResult};

/// Prompt templates for speculative verification.
pub mod prompts {
    /// Default system prompt for verification.
    pub const VERIFICATION_SYSTEM: &str = r"You are a verification assistant. Your task is to verify if a draft answer is accurate and well-supported by the provided context.

Analyze the draft for:
1. Factual accuracy - Is the information correct based on the context?
2. Completeness - Does it fully answer the question?
3. Consistency - Are there any contradictions?
4. Relevance - Is the answer relevant to the question?

Respond with your analysis and a decision: ACCEPT, REVISE, or REJECT.";

    /// Template for verification prompt.
    pub const VERIFICATION_TEMPLATE: &str = r"Question: {query}

Context:
{context}

Draft Answer:
{draft}

Please verify this draft answer and provide your decision.";

    /// Template for revision prompt.
    pub const REVISION_TEMPLATE: &str = r"Question: {query}

Context:
{context}

Original Draft:
{draft}

Issues Found:
{issues}

Please provide a revised answer that addresses the issues while staying faithful to the context.";
}

/// Configuration for the speculator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeculatorConfig {
    /// Temperature for generation (0.0 = deterministic, higher = more random).
    pub temperature: f32,
    /// Top-p sampling parameter.
    pub top_p: f32,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Confidence threshold for auto-accept.
    pub accept_threshold: f32,
    /// Confidence threshold for auto-reject.
    pub reject_threshold: f32,
    /// Maximum number of revision attempts.
    pub max_revisions: usize,
}

impl Default for SpeculatorConfig {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.9,
            max_tokens: 512,
            accept_threshold: 0.9,
            reject_threshold: 0.3,
            max_revisions: 2,
        }
    }
}

/// The Speculator trait for draft verification.
#[async_trait]
pub trait Speculator: Send + Sync {
    /// Verify a draft answer against the query and context.
    ///
    /// # Arguments
    /// * `draft` - The draft answer to verify
    /// * `context` - The retrieved documents providing context
    ///
    /// # Returns
    /// A speculation result with the decision and explanation.
    async fn verify_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError>;

    /// Revise a draft based on the verification result.
    ///
    /// # Arguments
    /// * `draft` - The original draft
    /// * `context` - The retrieved documents
    /// * `speculation` - The verification result with issues
    ///
    /// # Returns
    /// A revised draft addressing the issues.
    async fn revise_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
        speculation: &SpeculationResult,
    ) -> Result<Draft, SpeculatorError>;

    /// Get the configuration.
    fn config(&self) -> &SpeculatorConfig;
}

/// A simple rule-based speculator for testing.
pub struct RuleBasedSpeculator {
    config: SpeculatorConfig,
}

impl RuleBasedSpeculator {
    /// Create a new rule-based speculator.
    #[must_use]
    pub fn new(config: SpeculatorConfig) -> Self {
        Self { config }
    }

    /// Analyze a draft for common issues.
    #[allow(clippy::unused_self)]
    fn analyze_draft(&self, draft: &Draft, context: &[SearchResult]) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for empty draft
        if draft.content.trim().is_empty() {
            issues.push("Draft is empty".to_string());
            return issues;
        }

        // Check for minimum length
        if draft.content.len() < 10 {
            issues.push("Draft is too short".to_string());
        }

        // Check if draft mentions content from context
        if !context.is_empty() {
            let context_words: std::collections::HashSet<&str> = context
                .iter()
                .flat_map(|r| r.document.content.split_whitespace())
                .filter(|w| w.len() > 4)
                .collect();

            let draft_words: std::collections::HashSet<&str> = draft
                .content
                .split_whitespace()
                .filter(|w| w.len() > 4)
                .collect();

            let overlap: usize = draft_words.intersection(&context_words).count();
            #[allow(clippy::cast_precision_loss)]
            let overlap_ratio = if draft_words.is_empty() {
                0.0
            } else {
                overlap as f32 / draft_words.len() as f32
            };

            if overlap_ratio < 0.1 && !context.is_empty() {
                issues.push("Draft does not appear to use information from context".to_string());
            }
        }

        // Check for uncertainty markers
        let uncertainty_markers = [
            "maybe",
            "might",
            "possibly",
            "i think",
            "not sure",
            "uncertain",
        ];
        let draft_lower = draft.content.to_lowercase();
        for marker in &uncertainty_markers {
            if draft_lower.contains(marker) {
                issues.push(format!("Draft contains uncertainty marker: '{marker}'"));
            }
        }

        issues
    }

    /// Calculate a confidence score based on analysis.
    #[allow(clippy::cast_precision_loss, clippy::unused_self)]
    fn calculate_confidence(
        &self,
        draft: &Draft,
        context: &[SearchResult],
        issues: &[String],
    ) -> f32 {
        let mut confidence = 1.0;

        // Heavy penalty for empty drafts
        if draft.content.trim().is_empty() {
            return 0.0;
        }

        // Penalize for each issue
        confidence -= issues.len() as f32 * 0.15;

        // Boost for high draft confidence
        confidence += (draft.confidence - 0.5) * 0.2;

        // Boost for high context scores
        if !context.is_empty() {
            let avg_context_score: f32 =
                context.iter().map(|r| r.score).sum::<f32>() / context.len() as f32;
            confidence += (avg_context_score - 0.5) * 0.3;
        }

        confidence.clamp(0.0, 1.0)
    }
}

impl Default for RuleBasedSpeculator {
    fn default() -> Self {
        Self::new(SpeculatorConfig::default())
    }
}

#[async_trait]
impl Speculator for RuleBasedSpeculator {
    async fn verify_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError> {
        let issues = self.analyze_draft(draft, context);
        let confidence = self.calculate_confidence(draft, context, &issues);

        let decision = if confidence >= self.config.accept_threshold {
            SpeculationDecision::Accept
        } else if confidence <= self.config.reject_threshold {
            SpeculationDecision::Reject
        } else {
            SpeculationDecision::Revise
        };

        let explanation = match &decision {
            SpeculationDecision::Accept => "Draft appears accurate and well-supported.".to_string(),
            SpeculationDecision::Reject => {
                format!("Draft has significant issues: {}", issues.join("; "))
            }
            SpeculationDecision::Revise => {
                format!("Draft needs revision to address: {}", issues.join("; "))
            }
        };

        let mut result = SpeculationResult::new(decision, confidence).with_explanation(explanation);

        for issue in issues {
            result = result.with_issue(issue);
        }

        Ok(result)
    }

    async fn revise_draft(
        &self,
        draft: &Draft,
        context: &[SearchResult],
        speculation: &SpeculationResult,
    ) -> Result<Draft, SpeculatorError> {
        // Simple revision: append a note about the issues
        // In a real implementation, this would use an LLM
        let revision_note = if speculation.issues.is_empty() {
            String::new()
        } else {
            format!("\n\n[Revision notes: {}]", speculation.issues.join("; "))
        };

        let revised_content = if context.is_empty() {
            format!("{}{}", draft.content, revision_note)
        } else {
            // Try to incorporate context
            let context_summary: String = context
                .iter()
                .take(3)
                .map(|r| r.document.content.chars().take(100).collect::<String>())
                .collect::<Vec<_>>()
                .join(" ... ");

            format!(
                "Based on the available information: {}\n\n{}{}",
                context_summary, draft.content, revision_note
            )
        };

        Ok(Draft::new(revised_content, &draft.query).with_confidence(speculation.confidence + 0.1))
    }

    fn config(&self) -> &SpeculatorConfig {
        &self.config
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::types::Document;

    fn create_context() -> Vec<SearchResult> {
        vec![
            SearchResult::new(
                Document::new("The capital of France is Paris. It is known for the Eiffel Tower."),
                0.9,
                0,
            ),
            SearchResult::new(
                Document::new("Paris has a population of about 2 million in the city proper."),
                0.85,
                1,
            ),
        ]
    }

    #[tokio::test]
    async fn test_verify_good_draft() {
        let speculator = RuleBasedSpeculator::default();
        let draft = Draft::new(
            "The capital of France is Paris, which is famous for the Eiffel Tower.",
            "What is the capital of France?",
        )
        .with_confidence(0.9);

        let context = create_context();
        let result = speculator.verify_draft(&draft, &context).await.unwrap();

        assert!(result.confidence > 0.5);
        assert!(matches!(
            result.decision,
            SpeculationDecision::Accept | SpeculationDecision::Revise
        ));
    }

    #[tokio::test]
    async fn test_verify_empty_draft() {
        let speculator = RuleBasedSpeculator::default();
        let draft = Draft::new("", "What is the capital of France?");

        let context = create_context();
        let result = speculator.verify_draft(&draft, &context).await.unwrap();

        assert!(result.confidence < 0.5);
        assert!(!result.issues.is_empty());
    }

    #[tokio::test]
    async fn test_verify_uncertain_draft() {
        let speculator = RuleBasedSpeculator::default();
        let draft = Draft::new(
            "I think maybe the capital might be Paris, but I'm not sure.",
            "What is the capital of France?",
        );

        let context = create_context();
        let result = speculator.verify_draft(&draft, &context).await.unwrap();

        // Should detect uncertainty markers
        assert!(
            result
                .issues
                .iter()
                .any(|i| i.contains("uncertainty marker"))
        );
    }

    #[tokio::test]
    async fn test_revise_draft() {
        let speculator = RuleBasedSpeculator::default();
        let draft = Draft::new("Paris", "What is the capital of France?");

        let context = create_context();
        let speculation = speculator.verify_draft(&draft, &context).await.unwrap();
        let revised = speculator
            .revise_draft(&draft, &context, &speculation)
            .await
            .unwrap();

        assert!(revised.content.len() > draft.content.len());
    }

    #[tokio::test]
    async fn test_config() {
        let config = SpeculatorConfig {
            temperature: 0.5,
            accept_threshold: 0.85,
            ..Default::default()
        };

        let speculator = RuleBasedSpeculator::new(config);
        assert_eq!(speculator.config().temperature, 0.5);
        assert_eq!(speculator.config().accept_threshold, 0.85);
    }
}
