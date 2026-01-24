//! Traits for the Judge (logic verification) layer.

use async_trait::async_trait;

use crate::error::JudgeError;
use crate::types::{
    ClaimStructure, ClaimVerificationResult, Draft, LogicalClaim, SearchResult, VerificationResult,
};

/// Trait for extracting logical claims from text.
#[async_trait]
pub trait ClaimExtractor: Send + Sync {
    /// Extract logical claims from text.
    ///
    /// # Arguments
    /// * `text` - The text to extract claims from
    /// * `max_claims` - Maximum number of claims to extract
    ///
    /// # Returns
    /// A list of extracted logical claims.
    async fn extract_claims(
        &self,
        text: &str,
        max_claims: usize,
    ) -> Result<Vec<LogicalClaim>, JudgeError>;

    /// Convert a claim to SMT-LIB format.
    ///
    /// # Errors
    ///
    /// Returns an error if the claim cannot be converted to SMT-LIB format.
    fn to_smtlib(&self, claim: &LogicalClaim) -> Result<String, JudgeError>;
}

/// Trait for SMT-based verification.
#[async_trait]
pub trait SmtVerifier: Send + Sync {
    /// Verify a single claim.
    async fn verify_claim(
        &self,
        claim: &LogicalClaim,
    ) -> Result<ClaimVerificationResult, JudgeError>;

    /// Verify multiple claims.
    async fn verify_claims(
        &self,
        claims: &[LogicalClaim],
    ) -> Result<Vec<ClaimVerificationResult>, JudgeError>;

    /// Check consistency between multiple claims.
    async fn check_consistency(&self, claims: &[LogicalClaim]) -> Result<bool, JudgeError>;
}

/// Configuration for the Judge.
#[derive(Debug, Clone)]
pub struct JudgeConfig {
    /// Timeout for SMT solving in milliseconds.
    pub timeout_ms: u64,
    /// Maximum claims to verify.
    pub max_claims: usize,
    /// Whether to check consistency.
    pub check_consistency: bool,
    /// Minimum confidence for claim extraction.
    pub min_claim_confidence: f32,
    /// Whether to generate counterexamples.
    pub generate_counterexamples: bool,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5000,
            max_claims: 10,
            check_consistency: true,
            min_claim_confidence: 0.5,
            generate_counterexamples: true,
        }
    }
}

/// The Judge trait for logic verification.
#[async_trait]
pub trait Judge: Send + Sync {
    /// Fully verify a draft against the context.
    ///
    /// This extracts claims, verifies each one, and checks consistency.
    async fn judge(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<VerificationResult, JudgeError>;

    /// Quick verification using only claim extraction and basic checks.
    async fn quick_judge(&self, draft: &Draft) -> Result<VerificationResult, JudgeError>;

    /// Get the configuration.
    fn config(&self) -> &JudgeConfig;
}

/// A simple pattern-based claim extractor.
pub struct PatternClaimExtractor {
    /// Patterns for claim extraction.
    patterns: Vec<ClaimPattern>,
}

/// A pattern for extracting claims.
#[derive(Debug, Clone)]
pub struct ClaimPattern {
    /// Pattern name.
    pub name: String,
    /// Keywords that trigger this pattern.
    pub keywords: Vec<String>,
    /// How to construct the claim structure.
    pub structure_type: PatternStructureType,
}

/// Type of structure to create from a pattern match.
#[derive(Debug, Clone)]
pub enum PatternStructureType {
    /// Create a predicate structure.
    Predicate,
    /// Create a comparison structure.
    Comparison,
    /// Create an implication structure.
    Implication,
}

impl Default for PatternClaimExtractor {
    fn default() -> Self {
        Self {
            patterns: vec![
                ClaimPattern {
                    name: "is-a".to_string(),
                    keywords: vec![
                        "is".to_string(),
                        "are".to_string(),
                        "was".to_string(),
                        "were".to_string(),
                    ],
                    structure_type: PatternStructureType::Predicate,
                },
                ClaimPattern {
                    name: "comparison".to_string(),
                    keywords: vec![
                        "greater".to_string(),
                        "less".to_string(),
                        "more".to_string(),
                        "fewer".to_string(),
                        "equal".to_string(),
                    ],
                    structure_type: PatternStructureType::Comparison,
                },
                ClaimPattern {
                    name: "conditional".to_string(),
                    keywords: vec!["if".to_string(), "when".to_string(), "then".to_string()],
                    structure_type: PatternStructureType::Implication,
                },
            ],
        }
    }
}

impl PatternClaimExtractor {
    /// Create a new pattern-based claim extractor.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom pattern.
    #[must_use]
    pub fn with_pattern(mut self, pattern: ClaimPattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    fn extract_sentence_claims(&self, sentence: &str) -> Vec<(String, ClaimStructure, f32)> {
        let mut claims = Vec::new();
        let sentence_lower = sentence.to_lowercase();

        for pattern in &self.patterns {
            for keyword in &pattern.keywords {
                if sentence_lower.contains(keyword) {
                    let structure = match pattern.structure_type {
                        PatternStructureType::Predicate => ClaimStructure::Predicate {
                            subject: sentence.to_string(),
                            predicate: keyword.clone(),
                            object: None,
                        },
                        PatternStructureType::Comparison | PatternStructureType::Implication => {
                            ClaimStructure::Raw(sentence.to_string())
                        }
                    };

                    claims.push((sentence.to_string(), structure, 0.6));
                    break;
                }
            }
        }

        claims
    }
}

#[async_trait]
impl ClaimExtractor for PatternClaimExtractor {
    async fn extract_claims(
        &self,
        text: &str,
        max_claims: usize,
    ) -> Result<Vec<LogicalClaim>, JudgeError> {
        let sentences: Vec<&str> = text
            .split(['.', '!', '?'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        let mut claims = Vec::new();

        for sentence in sentences {
            for (text, structure, confidence) in self.extract_sentence_claims(sentence) {
                if claims.len() >= max_claims {
                    break;
                }
                claims.push(LogicalClaim::new(text, structure).with_confidence(confidence));
            }
        }

        Ok(claims)
    }

    fn to_smtlib(&self, claim: &LogicalClaim) -> Result<String, JudgeError> {
        let smtlib = match &claim.structure {
            ClaimStructure::Predicate {
                subject,
                predicate,
                object,
            } => {
                let obj_str = object.as_deref().unwrap_or("true");
                format!("(assert ({predicate} {subject} {obj_str}))")
            }
            ClaimStructure::Comparison {
                left,
                operator,
                right,
            } => {
                format!("(assert ({} {} {}))", operator.to_smtlib(), left, right)
            }
            ClaimStructure::And(claims) => {
                let inner: Result<Vec<String>, _> = claims
                    .iter()
                    .map(|c| self.to_smtlib(&LogicalClaim::new("", c.clone())))
                    .collect();
                format!("(assert (and {}))", inner?.join(" "))
            }
            ClaimStructure::Or(claims) => {
                let inner: Result<Vec<String>, _> = claims
                    .iter()
                    .map(|c| self.to_smtlib(&LogicalClaim::new("", c.clone())))
                    .collect();
                format!("(assert (or {}))", inner?.join(" "))
            }
            ClaimStructure::Not(inner) => {
                let inner_smt = self.to_smtlib(&LogicalClaim::new("", *inner.clone()))?;
                format!("(assert (not {inner_smt}))")
            }
            ClaimStructure::Implies {
                premise,
                conclusion,
            } => {
                let p = self.to_smtlib(&LogicalClaim::new("", *premise.clone()))?;
                let c = self.to_smtlib(&LogicalClaim::new("", *conclusion.clone()))?;
                format!("(assert (=> {p} {c}))")
            }
            ClaimStructure::Quantified {
                quantifier,
                variable,
                domain,
                body,
            } => {
                let body_smt = self.to_smtlib(&LogicalClaim::new("", *body.clone()))?;
                format!(
                    "(assert ({} (({} {})) {}))",
                    quantifier.to_smtlib(),
                    variable,
                    domain,
                    body_smt
                )
            }
            ClaimStructure::Raw(raw) => format!("(assert {raw})"),
            ClaimStructure::Temporal {
                event,
                time_relation,
                reference,
            } => {
                format!(
                    "(assert ({} {} {}))",
                    time_relation.to_smtlib(),
                    event,
                    reference
                )
            }
            ClaimStructure::Causal {
                cause,
                effect,
                strength,
            } => {
                let cause_smt = self.to_smtlib(&LogicalClaim::new("", *cause.clone()))?;
                let effect_smt = self.to_smtlib(&LogicalClaim::new("", *effect.clone()))?;
                format!(
                    "(assert ({} {} {}))",
                    strength.to_smtlib(),
                    cause_smt,
                    effect_smt
                )
            }
            ClaimStructure::Modal { claim, modality } => {
                let claim_smt = self.to_smtlib(&LogicalClaim::new("", *claim.clone()))?;
                format!("(assert ({} {}))", modality.to_smtlib(), claim_smt)
            }
        };

        Ok(smtlib)
    }
}

#[cfg(test)]
#[allow(clippy::single_char_pattern)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pattern_extractor_basic() {
        let extractor = PatternClaimExtractor::new();
        let claims = extractor
            .extract_claims("The sky is blue. Water flows downhill.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
    }

    #[tokio::test]
    async fn test_pattern_extractor_max_claims() {
        let extractor = PatternClaimExtractor::new();
        let text = "A is B. C is D. E is F. G is H.";
        let claims = extractor.extract_claims(text, 2).await.unwrap();

        assert!(claims.len() <= 2);
    }

    #[tokio::test]
    async fn test_pattern_extractor_empty() {
        let extractor = PatternClaimExtractor::new();
        let claims = extractor.extract_claims("", 10).await.unwrap();

        assert!(claims.is_empty());
    }

    #[test]
    fn test_to_smtlib_predicate() {
        let extractor = PatternClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Predicate {
                subject: "x".to_string(),
                predicate: "positive".to_string(),
                object: None,
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("assert"));
        assert!(smt.contains("positive"));
    }

    #[test]
    fn test_to_smtlib_comparison() {
        use crate::types::ComparisonOp;

        let extractor = PatternClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Comparison {
                left: "a".to_string(),
                operator: ComparisonOp::GreaterThan,
                right: "b".to_string(),
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains(">"));
    }
}
