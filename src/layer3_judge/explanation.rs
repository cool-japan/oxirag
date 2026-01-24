//! Explanation generation for verification results.
//!
//! This module provides utilities for generating human-readable explanations
//! of verification results, including detailed reasoning and counterexamples.

use std::fmt::Write as _;

use crate::types::{
    ClaimStructure, ClaimVerificationResult, LogicalClaim, VerificationResult, VerificationStatus,
};

/// A builder for generating detailed verification explanations.
pub struct ExplanationBuilder {
    /// Include individual claim explanations.
    include_claim_details: bool,
    /// Include counterexamples for falsified claims.
    include_counterexamples: bool,
    /// Include confidence reasoning.
    include_confidence_reasoning: bool,
    /// Maximum claims to explain in detail.
    max_detailed_claims: usize,
}

impl Default for ExplanationBuilder {
    fn default() -> Self {
        Self {
            include_claim_details: true,
            include_counterexamples: true,
            include_confidence_reasoning: true,
            max_detailed_claims: 5,
        }
    }
}

impl ExplanationBuilder {
    /// Create a new explanation builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include individual claim details.
    #[must_use]
    pub fn with_claim_details(mut self, include: bool) -> Self {
        self.include_claim_details = include;
        self
    }

    /// Set whether to include counterexamples.
    #[must_use]
    pub fn with_counterexamples(mut self, include: bool) -> Self {
        self.include_counterexamples = include;
        self
    }

    /// Set whether to include confidence reasoning.
    #[must_use]
    pub fn with_confidence_reasoning(mut self, include: bool) -> Self {
        self.include_confidence_reasoning = include;
        self
    }

    /// Set the maximum number of claims to explain in detail.
    #[must_use]
    pub fn with_max_detailed_claims(mut self, max: usize) -> Self {
        self.max_detailed_claims = max;
        self
    }

    /// Generate an explanation for a verification result.
    #[must_use]
    pub fn explain(&self, result: &VerificationResult) -> String {
        let mut explanation = String::new();

        // Overall status summary
        explanation.push_str(&Self::explain_overall_status(result));
        explanation.push_str("\n\n");

        // Confidence reasoning
        if self.include_confidence_reasoning {
            explanation.push_str(&Self::explain_confidence(result));
            explanation.push_str("\n\n");
        }

        // Individual claim details
        if self.include_claim_details && !result.claim_results.is_empty() {
            explanation.push_str(&self.explain_claims(&result.claim_results));
        }

        explanation.trim().to_string()
    }

    /// Explain the overall verification status.
    fn explain_overall_status(result: &VerificationResult) -> String {
        let status_text = match result.status {
            VerificationStatus::Verified => {
                "VERIFIED: All claims in the response have been logically verified."
            }
            VerificationStatus::Falsified => {
                "FALSIFIED: One or more claims in the response are logically inconsistent or contradicted by the evidence."
            }
            VerificationStatus::Unknown => {
                "UNKNOWN: The verification could not definitively confirm or deny all claims."
            }
            VerificationStatus::Timeout => {
                "TIMEOUT: The verification process exceeded the time limit."
            }
            VerificationStatus::Error => "ERROR: An error occurred during verification.",
        };

        format!("## Verification Status\n{status_text}")
    }

    /// Explain the confidence score.
    #[allow(clippy::cast_precision_loss)]
    fn explain_confidence(result: &VerificationResult) -> String {
        let verified_count = result
            .claim_results
            .iter()
            .filter(|r| r.status == VerificationStatus::Verified)
            .count();
        let falsified_count = result
            .claim_results
            .iter()
            .filter(|r| r.status == VerificationStatus::Falsified)
            .count();
        let unknown_count = result
            .claim_results
            .iter()
            .filter(|r| r.status == VerificationStatus::Unknown)
            .count();
        let total = result.claim_results.len();

        let mut reasoning = String::from("## Confidence Analysis\n");
        let _ = writeln!(
            reasoning,
            "Overall confidence: {:.1}%",
            result.confidence * 100.0
        );

        if total > 0 {
            let _ = writeln!(
                reasoning,
                "- Verified: {verified_count}/{total} ({:.1}%)",
                (verified_count as f32 / total as f32) * 100.0
            );
            let _ = writeln!(
                reasoning,
                "- Falsified: {falsified_count}/{total} ({:.1}%)",
                (falsified_count as f32 / total as f32) * 100.0
            );
            let _ = write!(
                reasoning,
                "- Unknown: {unknown_count}/{total} ({:.1}%)",
                (unknown_count as f32 / total as f32) * 100.0
            );
        } else {
            reasoning.push_str("No claims were extracted for verification.");
        }

        reasoning
    }

    /// Explain individual claim verification results.
    fn explain_claims(&self, claim_results: &[ClaimVerificationResult]) -> String {
        let mut explanation = String::from("## Claim Details\n");

        for (i, result) in claim_results
            .iter()
            .take(self.max_detailed_claims)
            .enumerate()
        {
            let _ = write!(explanation, "\n### Claim {}\n", i + 1);
            explanation.push_str(&self.explain_single_claim(result));
        }

        if claim_results.len() > self.max_detailed_claims {
            let _ = write!(
                explanation,
                "\n... and {} more claims not shown.",
                claim_results.len() - self.max_detailed_claims
            );
        }

        explanation
    }

    /// Explain a single claim verification result.
    fn explain_single_claim(&self, result: &ClaimVerificationResult) -> String {
        let mut explanation = String::new();

        // Claim text
        let _ = writeln!(explanation, "**Text:** \"{}\"", result.claim.text);

        // Claim structure
        let _ = writeln!(
            explanation,
            "**Structure:** {}",
            Self::describe_structure(&result.claim.structure)
        );

        // Verification status
        let status_str = match result.status {
            VerificationStatus::Verified => "✓ Verified",
            VerificationStatus::Falsified => "✗ Falsified",
            VerificationStatus::Unknown => "? Unknown",
            VerificationStatus::Timeout => "⏱ Timeout",
            VerificationStatus::Error => "⚠ Error",
        };
        let _ = writeln!(explanation, "**Status:** {status_str}");

        // Explanation if available
        if self.include_counterexamples
            && let Some(exp) = &result.explanation
        {
            let _ = writeln!(explanation, "**Explanation:** {exp}");
        }

        // Duration
        let _ = write!(explanation, "**Duration:** {}ms", result.duration_ms);

        explanation
    }

    /// Describe a claim structure in human-readable form.
    fn describe_structure(structure: &ClaimStructure) -> String {
        match structure {
            ClaimStructure::Predicate {
                subject,
                predicate,
                object,
            } => {
                if let Some(obj) = object {
                    format!("Predicate: {subject} {predicate} {obj}")
                } else {
                    format!("Predicate: {subject} {predicate}")
                }
            }
            ClaimStructure::Comparison {
                left,
                operator,
                right,
            } => {
                format!("Comparison: {left} {} {right}", operator.to_smtlib())
            }
            ClaimStructure::And(claims) => {
                format!("Conjunction of {} claims", claims.len())
            }
            ClaimStructure::Or(claims) => {
                format!("Disjunction of {} claims", claims.len())
            }
            ClaimStructure::Not(_) => "Negation".to_string(),
            ClaimStructure::Implies { .. } => "Implication".to_string(),
            ClaimStructure::Quantified {
                quantifier,
                variable,
                domain,
                ..
            } => {
                format!("{} {variable} in {domain}", quantifier.to_smtlib())
            }
            ClaimStructure::Temporal {
                event,
                time_relation,
                reference,
            } => {
                format!(
                    "Temporal: {event} {} {reference}",
                    time_relation.to_smtlib()
                )
            }
            ClaimStructure::Causal { strength, .. } => {
                format!("Causal ({})", strength.to_smtlib())
            }
            ClaimStructure::Modal { modality, .. } => {
                format!("Modal ({})", modality.to_smtlib())
            }
            ClaimStructure::Raw(text) => {
                let truncated = if text.len() > 50 {
                    format!("{}...", &text[..47])
                } else {
                    text.clone()
                };
                format!("Raw: \"{truncated}\"")
            }
        }
    }
}

/// Generate a counterexample explanation for a falsified claim.
#[must_use]
pub fn generate_counterexample(claim: &LogicalClaim, context: &str) -> Option<String> {
    match &claim.structure {
        ClaimStructure::Comparison {
            left,
            operator,
            right,
        } => Some(format!(
            "The comparison '{left} {} {right}' cannot be satisfied with the given constraints.",
            operator.to_smtlib()
        )),
        ClaimStructure::Implies { .. } => Some(
            "The implication is falsified because the premise is true but the conclusion is false."
                .to_string(),
        ),
        ClaimStructure::And(claims) => Some(format!(
            "At least one of the {} conjuncts is false.",
            claims.len()
        )),
        ClaimStructure::Temporal {
            event,
            time_relation,
            reference,
        } => Some(format!(
            "The temporal relationship '{event} {} {reference}' is contradicted by the context: {context}",
            time_relation.to_smtlib()
        )),
        ClaimStructure::Causal { .. } => {
            Some("The causal relationship is not supported by the evidence.".to_string())
        }
        _ => None,
    }
}

/// Generate a quick summary explanation for a verification result.
#[must_use]
pub fn generate_summary(result: &VerificationResult) -> String {
    let verified = result
        .claim_results
        .iter()
        .filter(|r| r.status == VerificationStatus::Verified)
        .count();
    let total = result.claim_results.len();

    match result.status {
        VerificationStatus::Verified => {
            format!(
                "All {total} claims verified successfully with {:.0}% confidence.",
                result.confidence * 100.0
            )
        }
        VerificationStatus::Falsified => {
            let falsified = result
                .claim_results
                .iter()
                .filter(|r| r.status == VerificationStatus::Falsified)
                .count();
            format!(
                "{falsified} of {total} claims falsified. Only {verified} claims verified. Confidence: {:.0}%",
                result.confidence * 100.0
            )
        }
        VerificationStatus::Unknown => {
            format!(
                "Verification inconclusive. {verified} of {total} claims verified. Confidence: {:.0}%",
                result.confidence * 100.0
            )
        }
        VerificationStatus::Timeout => "Verification timed out before completion.".to_string(),
        VerificationStatus::Error => "An error occurred during verification.".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ComparisonOp;

    fn create_test_claim(text: &str, structure: ClaimStructure) -> LogicalClaim {
        LogicalClaim::new(text, structure).with_confidence(0.8)
    }

    fn create_test_result(status: VerificationStatus) -> ClaimVerificationResult {
        let claim = create_test_claim("test claim", ClaimStructure::Raw("test".to_string()));
        ClaimVerificationResult::new(claim, status).with_duration(10)
    }

    #[test]
    fn test_explanation_builder_default() {
        let builder = ExplanationBuilder::new();
        assert!(builder.include_claim_details);
        assert!(builder.include_counterexamples);
        assert!(builder.include_confidence_reasoning);
        assert_eq!(builder.max_detailed_claims, 5);
    }

    #[test]
    fn test_explanation_builder_with_options() {
        let builder = ExplanationBuilder::new()
            .with_claim_details(false)
            .with_counterexamples(false)
            .with_confidence_reasoning(false)
            .with_max_detailed_claims(3);

        assert!(!builder.include_claim_details);
        assert!(!builder.include_counterexamples);
        assert!(!builder.include_confidence_reasoning);
        assert_eq!(builder.max_detailed_claims, 3);
    }

    #[test]
    fn test_explain_verified_result() {
        let mut result = VerificationResult::new(VerificationStatus::Verified).with_confidence(1.0);
        result = result.with_claim_result(create_test_result(VerificationStatus::Verified));

        let builder = ExplanationBuilder::new();
        let explanation = builder.explain(&result);

        assert!(explanation.contains("VERIFIED"));
        assert!(explanation.contains("100.0%"));
    }

    #[test]
    fn test_explain_falsified_result() {
        let mut result =
            VerificationResult::new(VerificationStatus::Falsified).with_confidence(0.5);
        result = result.with_claim_result(create_test_result(VerificationStatus::Falsified));

        let builder = ExplanationBuilder::new();
        let explanation = builder.explain(&result);

        assert!(explanation.contains("FALSIFIED"));
    }

    #[test]
    fn test_describe_structure_predicate() {
        let structure = ClaimStructure::Predicate {
            subject: "sky".to_string(),
            predicate: "is".to_string(),
            object: Some("blue".to_string()),
        };

        let description = ExplanationBuilder::describe_structure(&structure);
        assert!(description.contains("Predicate"));
        assert!(description.contains("sky"));
    }

    #[test]
    fn test_describe_structure_comparison() {
        let structure = ClaimStructure::Comparison {
            left: "5".to_string(),
            operator: ComparisonOp::GreaterThan,
            right: "3".to_string(),
        };

        let description = ExplanationBuilder::describe_structure(&structure);
        assert!(description.contains("Comparison"));
    }

    #[test]
    fn test_generate_summary_verified() {
        let result = VerificationResult::new(VerificationStatus::Verified)
            .with_confidence(1.0)
            .with_claim_result(create_test_result(VerificationStatus::Verified));

        let summary = generate_summary(&result);
        assert!(summary.contains("verified successfully"));
    }

    #[test]
    fn test_generate_summary_falsified() {
        let result = VerificationResult::new(VerificationStatus::Falsified)
            .with_confidence(0.5)
            .with_claim_result(create_test_result(VerificationStatus::Falsified));

        let summary = generate_summary(&result);
        assert!(summary.contains("falsified"));
    }

    #[test]
    fn test_generate_counterexample_comparison() {
        let claim = create_test_claim(
            "5 > 10",
            ClaimStructure::Comparison {
                left: "5".to_string(),
                operator: ComparisonOp::GreaterThan,
                right: "10".to_string(),
            },
        );

        let counterexample = generate_counterexample(&claim, "");
        assert!(counterexample.is_some());
        assert!(counterexample.unwrap().contains("comparison"));
    }

    #[test]
    fn test_max_detailed_claims_limit() {
        let mut result = VerificationResult::new(VerificationStatus::Verified).with_confidence(1.0);

        // Add 10 claims
        for _ in 0..10 {
            result = result.with_claim_result(create_test_result(VerificationStatus::Verified));
        }

        let builder = ExplanationBuilder::new().with_max_detailed_claims(3);
        let explanation = builder.explain(&result);

        // Should mention that more claims are not shown
        assert!(explanation.contains("more claims not shown"));
    }
}
