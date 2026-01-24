//! Advanced claim extraction from text.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::JudgeError;
use crate::layer3_judge::traits::ClaimExtractor;
use crate::types::{
    CausalStrength, ClaimStructure, ComparisonOp, LogicalClaim, Modality, TimeRelation,
};

/// An advanced claim extractor using NLP heuristics.
pub struct AdvancedClaimExtractor {
    /// Verb patterns for predicate extraction.
    predicate_verbs: Vec<String>,
    /// Comparison keywords.
    comparison_words: HashMap<String, ComparisonOp>,
    /// Conditional keywords.
    conditional_words: Vec<String>,
    /// Negation words.
    negation_words: Vec<String>,
    /// Temporal keywords mapped to time relations.
    temporal_words: HashMap<String, TimeRelation>,
    /// Causal keywords mapped to causal strength.
    causal_words: HashMap<String, CausalStrength>,
    /// Modal keywords mapped to modality.
    modal_words: HashMap<String, Modality>,
}

impl Default for AdvancedClaimExtractor {
    fn default() -> Self {
        let mut comparison_words = HashMap::new();
        comparison_words.insert("greater".to_string(), ComparisonOp::GreaterThan);
        comparison_words.insert("more".to_string(), ComparisonOp::GreaterThan);
        comparison_words.insert("higher".to_string(), ComparisonOp::GreaterThan);
        comparison_words.insert("larger".to_string(), ComparisonOp::GreaterThan);
        comparison_words.insert("less".to_string(), ComparisonOp::LessThan);
        comparison_words.insert("fewer".to_string(), ComparisonOp::LessThan);
        comparison_words.insert("smaller".to_string(), ComparisonOp::LessThan);
        comparison_words.insert("lower".to_string(), ComparisonOp::LessThan);
        comparison_words.insert("equal".to_string(), ComparisonOp::Equal);
        comparison_words.insert("same".to_string(), ComparisonOp::Equal);

        let mut temporal_words = HashMap::new();
        temporal_words.insert("before".to_string(), TimeRelation::Before);
        temporal_words.insert("prior".to_string(), TimeRelation::Before);
        temporal_words.insert("preceded".to_string(), TimeRelation::Before);
        temporal_words.insert("after".to_string(), TimeRelation::After);
        temporal_words.insert("following".to_string(), TimeRelation::After);
        temporal_words.insert("subsequently".to_string(), TimeRelation::After);
        temporal_words.insert("during".to_string(), TimeRelation::During);
        temporal_words.insert("while".to_string(), TimeRelation::During);
        temporal_words.insert("throughout".to_string(), TimeRelation::During);
        temporal_words.insert("simultaneously".to_string(), TimeRelation::Simultaneous);
        temporal_words.insert("concurrently".to_string(), TimeRelation::Simultaneous);
        temporal_words.insert("meanwhile".to_string(), TimeRelation::Simultaneous);

        let mut causal_words = HashMap::new();
        causal_words.insert("causes".to_string(), CausalStrength::Direct);
        causal_words.insert("caused".to_string(), CausalStrength::Direct);
        causal_words.insert("results".to_string(), CausalStrength::Direct);
        causal_words.insert("therefore".to_string(), CausalStrength::Direct);
        causal_words.insert("thus".to_string(), CausalStrength::Direct);
        causal_words.insert("hence".to_string(), CausalStrength::Direct);
        causal_words.insert("because".to_string(), CausalStrength::Direct);
        causal_words.insert("contributes".to_string(), CausalStrength::Indirect);
        causal_words.insert("influences".to_string(), CausalStrength::Indirect);
        causal_words.insert("affects".to_string(), CausalStrength::Indirect);
        causal_words.insert("leads".to_string(), CausalStrength::Indirect);
        causal_words.insert("correlates".to_string(), CausalStrength::Correlated);
        causal_words.insert("associated".to_string(), CausalStrength::Correlated);
        causal_words.insert("linked".to_string(), CausalStrength::Correlated);

        let mut modal_words = HashMap::new();
        modal_words.insert("might".to_string(), Modality::Possible);
        modal_words.insert("may".to_string(), Modality::Possible);
        modal_words.insert("could".to_string(), Modality::Possible);
        modal_words.insert("possibly".to_string(), Modality::Possible);
        modal_words.insert("perhaps".to_string(), Modality::Possible);
        modal_words.insert("must".to_string(), Modality::Necessary);
        modal_words.insert("necessarily".to_string(), Modality::Necessary);
        modal_words.insert("certainly".to_string(), Modality::Necessary);
        modal_words.insert("definitely".to_string(), Modality::Necessary);
        modal_words.insert("should".to_string(), Modality::Likely);
        modal_words.insert("probably".to_string(), Modality::Likely);
        modal_words.insert("likely".to_string(), Modality::Likely);
        modal_words.insert("unlikely".to_string(), Modality::Unlikely);
        modal_words.insert("improbably".to_string(), Modality::Unlikely);
        modal_words.insert("doubtfully".to_string(), Modality::Unlikely);

        Self {
            predicate_verbs: vec![
                "is".to_string(),
                "are".to_string(),
                "was".to_string(),
                "were".to_string(),
                "has".to_string(),
                "have".to_string(),
                "contains".to_string(),
                "includes".to_string(),
            ],
            comparison_words,
            conditional_words: vec![
                "if".to_string(),
                "when".to_string(),
                "whenever".to_string(),
                "unless".to_string(),
            ],
            negation_words: vec![
                "not".to_string(),
                "never".to_string(),
                "no".to_string(),
                "none".to_string(),
            ],
            temporal_words,
            causal_words,
            modal_words,
        }
    }
}

impl AdvancedClaimExtractor {
    /// Create a new advanced claim extractor.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Split text into sentences.
    #[allow(clippy::unused_self)]
    fn split_sentences(&self, text: &str) -> Vec<String> {
        text.split(['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Tokenize a sentence into words.
    #[allow(clippy::unused_self)]
    fn tokenize(&self, sentence: &str) -> Vec<String> {
        sentence
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }

    /// Check if a sentence contains negation.
    fn has_negation(&self, tokens: &[String]) -> bool {
        tokens.iter().any(|t| self.negation_words.contains(t))
    }

    /// Try to extract a predicate claim.
    #[allow(clippy::cast_precision_loss)]
    fn try_extract_predicate(
        &self,
        sentence: &str,
        tokens: &[String],
    ) -> Option<(ClaimStructure, f32)> {
        for (i, token) in tokens.iter().enumerate() {
            if self.predicate_verbs.contains(token) && i > 0 && i < tokens.len() - 1 {
                let subject = tokens[..i].join(" ");
                let predicate = token.clone();
                let object = tokens[i + 1..].join(" ");

                let structure = ClaimStructure::Predicate {
                    subject,
                    predicate,
                    object: Some(object),
                };

                // Higher confidence for longer, more specific claims
                let confidence = (0.5 + (sentence.len() as f32 / 200.0).min(0.4)).min(0.9);

                return Some((structure, confidence));
            }
        }
        None
    }

    /// Try to extract a comparison claim.
    fn try_extract_comparison(&self, tokens: &[String]) -> Option<(ClaimStructure, f32)> {
        for (i, token) in tokens.iter().enumerate() {
            if let Some(op) = self.comparison_words.get(token)
                && i > 0
                && i < tokens.len() - 1
            {
                let left = tokens[..i].join(" ");
                let right = tokens[i + 1..].join(" ");

                let structure = ClaimStructure::Comparison {
                    left,
                    operator: *op,
                    right,
                };

                return Some((structure, 0.7));
            }
        }
        None
    }

    /// Try to extract a conditional claim.
    fn try_extract_conditional(&self, sentence: &str) -> Option<(ClaimStructure, f32)> {
        let sentence_lower = sentence.to_lowercase();

        for keyword in &self.conditional_words {
            if sentence_lower.starts_with(keyword) {
                // Look for "then" or a comma to split premise/conclusion
                let rest = &sentence[keyword.len()..].trim();
                let split_point = rest.find("then").or_else(|| rest.find(','));

                if let Some(pos) = split_point {
                    let premise_text = rest[..pos].trim();
                    let conclusion_text = rest[pos..]
                        .trim_start_matches(',')
                        .trim_start_matches("then")
                        .trim();

                    let premise = ClaimStructure::Raw(premise_text.to_string());
                    let conclusion = ClaimStructure::Raw(conclusion_text.to_string());

                    let structure = ClaimStructure::Implies {
                        premise: Box::new(premise),
                        conclusion: Box::new(conclusion),
                    };

                    return Some((structure, 0.65));
                }
            }
        }
        None
    }

    /// Try to extract a temporal claim.
    fn try_extract_temporal(&self, tokens: &[String]) -> Option<(ClaimStructure, f32)> {
        for (i, token) in tokens.iter().enumerate() {
            if let Some(time_relation) = self.temporal_words.get(token)
                && i > 0
                && i < tokens.len() - 1
            {
                let event = tokens[..i].join(" ");
                let reference = tokens[i + 1..].join(" ");

                let structure = ClaimStructure::Temporal {
                    event,
                    time_relation: time_relation.clone(),
                    reference,
                };

                return Some((structure, 0.7));
            }
        }
        None
    }

    /// Try to extract a causal claim.
    fn try_extract_causal(&self, tokens: &[String]) -> Option<(ClaimStructure, f32)> {
        for (i, token) in tokens.iter().enumerate() {
            if let Some(strength) = self.causal_words.get(token)
                && i > 0
                && i < tokens.len() - 1
            {
                let cause_text = tokens[..i].join(" ");
                let effect_text = tokens[i + 1..].join(" ");

                let cause = ClaimStructure::Raw(cause_text);
                let effect = ClaimStructure::Raw(effect_text);

                let structure = ClaimStructure::Causal {
                    cause: Box::new(cause),
                    effect: Box::new(effect),
                    strength: strength.clone(),
                };

                return Some((structure, 0.65));
            }
        }
        None
    }

    /// Try to extract a modal claim.
    fn try_extract_modal(
        &self,
        sentence: &str,
        tokens: &[String],
    ) -> Option<(ClaimStructure, f32)> {
        for (i, token) in tokens.iter().enumerate() {
            if let Some(modality) = self.modal_words.get(token) {
                // Get the rest of the sentence after the modal word
                let remaining_tokens = if i < tokens.len() - 1 {
                    &tokens[i + 1..]
                } else {
                    continue;
                };

                if remaining_tokens.is_empty() {
                    continue;
                }

                // Try to extract the underlying claim from the remaining tokens
                let inner_claim = if let Some((inner_structure, _)) =
                    self.try_extract_predicate(sentence, remaining_tokens)
                {
                    inner_structure
                } else {
                    ClaimStructure::Raw(remaining_tokens.join(" "))
                };

                let structure = ClaimStructure::Modal {
                    claim: Box::new(inner_claim),
                    modality: modality.clone(),
                };

                return Some((structure, 0.6));
            }
        }
        None
    }

    /// Extract a claim from a sentence.
    fn extract_claim_from_sentence(&self, sentence: &str) -> Option<LogicalClaim> {
        let tokens = self.tokenize(sentence);

        if tokens.len() < 2 {
            return None;
        }

        let has_negation = self.has_negation(&tokens);

        // Try different extraction strategies in order of specificity
        let extraction = self
            .try_extract_conditional(sentence)
            .or_else(|| self.try_extract_modal(sentence, &tokens))
            .or_else(|| self.try_extract_causal(&tokens))
            .or_else(|| self.try_extract_temporal(&tokens))
            .or_else(|| self.try_extract_comparison(&tokens))
            .or_else(|| self.try_extract_predicate(sentence, &tokens));

        extraction.map(|(mut structure, mut confidence)| {
            // Wrap in negation if needed
            if has_negation {
                structure = ClaimStructure::Not(Box::new(structure));
                confidence *= 0.9; // Slightly lower confidence for negated claims
            }

            LogicalClaim::new(sentence, structure).with_confidence(confidence)
        })
    }
}

#[async_trait]
impl ClaimExtractor for AdvancedClaimExtractor {
    async fn extract_claims(
        &self,
        text: &str,
        max_claims: usize,
    ) -> Result<Vec<LogicalClaim>, JudgeError> {
        let sentences = self.split_sentences(text);

        let claims: Vec<LogicalClaim> = sentences
            .into_iter()
            .filter_map(|s| self.extract_claim_from_sentence(&s))
            .take(max_claims)
            .collect();

        Ok(claims)
    }

    fn to_smtlib(&self, claim: &LogicalClaim) -> Result<String, JudgeError> {
        fn structure_to_smt(structure: &ClaimStructure) -> String {
            match structure {
                ClaimStructure::Predicate {
                    subject,
                    predicate,
                    object,
                } => {
                    let subj = subject.replace(' ', "_");
                    let pred = predicate.replace(' ', "_");
                    let obj = object
                        .as_ref()
                        .map_or("true".to_string(), |o| o.replace(' ', "_"));
                    format!("({pred} |{subj}| |{obj}|)")
                }
                ClaimStructure::Comparison {
                    left,
                    operator,
                    right,
                } => {
                    let l = left.replace(' ', "_");
                    let r = right.replace(' ', "_");
                    format!("({} |{}| |{}|)", operator.to_smtlib(), l, r)
                }
                ClaimStructure::And(claims) => {
                    let parts: Vec<String> = claims.iter().map(structure_to_smt).collect();
                    format!("(and {})", parts.join(" "))
                }
                ClaimStructure::Or(claims) => {
                    let parts: Vec<String> = claims.iter().map(structure_to_smt).collect();
                    format!("(or {})", parts.join(" "))
                }
                ClaimStructure::Not(inner) => {
                    format!("(not {})", structure_to_smt(inner))
                }
                ClaimStructure::Implies {
                    premise,
                    conclusion,
                } => {
                    format!(
                        "(=> {} {})",
                        structure_to_smt(premise),
                        structure_to_smt(conclusion)
                    )
                }
                ClaimStructure::Quantified {
                    quantifier,
                    variable,
                    domain,
                    body,
                } => {
                    format!(
                        "({} (({} {})) {})",
                        quantifier.to_smtlib(),
                        variable,
                        domain,
                        structure_to_smt(body)
                    )
                }
                ClaimStructure::Temporal {
                    event,
                    time_relation,
                    reference,
                } => {
                    let ev = event.replace(' ', "_");
                    let rel = time_relation.to_smtlib();
                    let ref_str = reference.replace(' ', "_");
                    format!("({rel} |{ev}| |{ref_str}|)")
                }
                ClaimStructure::Causal {
                    cause,
                    effect,
                    strength,
                } => {
                    format!(
                        "({} {} {})",
                        strength.to_smtlib(),
                        structure_to_smt(cause),
                        structure_to_smt(effect)
                    )
                }
                ClaimStructure::Modal { claim, modality } => {
                    format!("({} {})", modality.to_smtlib(), structure_to_smt(claim))
                }
                ClaimStructure::Raw(raw) => format!("|{}|", raw.replace(' ', "_")),
            }
        }

        let smt_expr = structure_to_smt(&claim.structure);
        Ok(format!("(assert {smt_expr})"))
    }
}

#[cfg(test)]
#[allow(clippy::single_char_pattern)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extract_predicate_claim() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The capital of France is Paris.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(
            claims[0].structure,
            ClaimStructure::Predicate { .. }
        ));
    }

    #[tokio::test]
    async fn test_extract_comparison_claim() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The population is greater than one million.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
    }

    #[tokio::test]
    async fn test_extract_conditional_claim() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("If it rains, then the ground gets wet.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(
            claims[0].structure,
            ClaimStructure::Implies { .. }
        ));
    }

    #[tokio::test]
    async fn test_extract_negation() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The answer is not correct.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Not(_)));
    }

    #[tokio::test]
    async fn test_multiple_sentences() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims(
                "The sky is blue. Water is essential. If heated, water becomes steam.",
                10,
            )
            .await
            .unwrap();

        assert!(claims.len() >= 2);
    }

    #[tokio::test]
    async fn test_max_claims_limit() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("A is B. C is D. E is F. G is H.", 2)
            .await
            .unwrap();

        assert!(claims.len() <= 2);
    }

    #[test]
    fn test_to_smtlib() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Predicate {
                subject: "X".to_string(),
                predicate: "is".to_string(),
                object: Some("Y".to_string()),
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("assert"));
        assert!(smt.contains("is"));
    }

    #[test]
    fn test_to_smtlib_comparison() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Comparison {
                left: "a".to_string(),
                operator: ComparisonOp::LessThan,
                right: "b".to_string(),
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("<"));
    }

    #[test]
    fn test_to_smtlib_implies() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Implies {
                premise: Box::new(ClaimStructure::Raw("premise".to_string())),
                conclusion: Box::new(ClaimStructure::Raw("conclusion".to_string())),
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("=>"));
    }

    // Tests for temporal claims
    #[tokio::test]
    async fn test_extract_temporal_before() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The meeting started before lunch.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(
            claims[0].structure,
            ClaimStructure::Temporal { .. }
        ));
    }

    #[tokio::test]
    async fn test_extract_temporal_after() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The event occurred after the announcement.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(
            claims[0].structure,
            ClaimStructure::Temporal { .. }
        ));
    }

    #[tokio::test]
    async fn test_extract_temporal_during() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("Sales increased during the holiday season.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(
            claims[0].structure,
            ClaimStructure::Temporal { .. }
        ));
    }

    #[test]
    fn test_to_smtlib_temporal() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Temporal {
                event: "meeting".to_string(),
                time_relation: TimeRelation::Before,
                reference: "lunch".to_string(),
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("before"));
        assert!(smt.contains("meeting"));
        assert!(smt.contains("lunch"));
    }

    // Tests for causal claims
    #[tokio::test]
    async fn test_extract_causal_causes() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("Smoking causes cancer.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Causal { .. }));
    }

    #[tokio::test]
    async fn test_extract_causal_because() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The plant died because of lack of water.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Causal { .. }));
    }

    #[tokio::test]
    async fn test_extract_causal_leads() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("Exercise leads to better health.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Causal { .. }));
    }

    #[tokio::test]
    async fn test_extract_causal_correlates() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("Income correlates with education level.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Causal { .. }));
    }

    #[test]
    fn test_to_smtlib_causal() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Causal {
                cause: Box::new(ClaimStructure::Raw("smoking".to_string())),
                effect: Box::new(ClaimStructure::Raw("cancer".to_string())),
                strength: CausalStrength::Direct,
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("causes"));
        assert!(smt.contains("smoking"));
        assert!(smt.contains("cancer"));
    }

    #[test]
    fn test_to_smtlib_causal_indirect() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Causal {
                cause: Box::new(ClaimStructure::Raw("exercise".to_string())),
                effect: Box::new(ClaimStructure::Raw("health".to_string())),
                strength: CausalStrength::Indirect,
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("contributes_to"));
    }

    #[test]
    fn test_to_smtlib_causal_correlated() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Causal {
                cause: Box::new(ClaimStructure::Raw("income".to_string())),
                effect: Box::new(ClaimStructure::Raw("education".to_string())),
                strength: CausalStrength::Correlated,
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("correlates_with"));
    }

    // Tests for modal claims
    #[tokio::test]
    async fn test_extract_modal_might() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The weather might be sunny tomorrow.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Modal { .. }));
    }

    #[tokio::test]
    async fn test_extract_modal_must() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The answer must be correct.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Modal { .. }));
    }

    #[tokio::test]
    async fn test_extract_modal_probably() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The train will probably arrive on time.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Modal { .. }));
    }

    #[tokio::test]
    async fn test_extract_modal_could() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("This could be the solution.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Modal { .. }));
    }

    #[test]
    fn test_to_smtlib_modal_possible() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Modal {
                claim: Box::new(ClaimStructure::Raw("sunny".to_string())),
                modality: Modality::Possible,
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("possibly"));
        assert!(smt.contains("sunny"));
    }

    #[test]
    fn test_to_smtlib_modal_necessary() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Modal {
                claim: Box::new(ClaimStructure::Raw("correct".to_string())),
                modality: Modality::Necessary,
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("necessarily"));
    }

    #[test]
    fn test_to_smtlib_modal_likely() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Modal {
                claim: Box::new(ClaimStructure::Raw("on_time".to_string())),
                modality: Modality::Likely,
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("likely"));
    }

    #[test]
    fn test_to_smtlib_modal_unlikely() {
        let extractor = AdvancedClaimExtractor::new();
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Modal {
                claim: Box::new(ClaimStructure::Raw("failure".to_string())),
                modality: Modality::Unlikely,
            },
        );

        let smt = extractor.to_smtlib(&claim).unwrap();
        assert!(smt.contains("unlikely"));
    }

    // Test time relation enum
    #[test]
    fn test_time_relation_to_smtlib() {
        assert_eq!(TimeRelation::Before.to_smtlib(), "before");
        assert_eq!(TimeRelation::After.to_smtlib(), "after");
        assert_eq!(TimeRelation::During.to_smtlib(), "during");
        assert_eq!(TimeRelation::Simultaneous.to_smtlib(), "simultaneous");
    }

    // Test causal strength enum
    #[test]
    fn test_causal_strength_to_smtlib() {
        assert_eq!(CausalStrength::Direct.to_smtlib(), "causes");
        assert_eq!(CausalStrength::Indirect.to_smtlib(), "contributes_to");
        assert_eq!(CausalStrength::Correlated.to_smtlib(), "correlates_with");
    }

    // Test modality enum
    #[test]
    fn test_modality_to_smtlib() {
        assert_eq!(Modality::Possible.to_smtlib(), "possibly");
        assert_eq!(Modality::Necessary.to_smtlib(), "necessarily");
        assert_eq!(Modality::Likely.to_smtlib(), "likely");
        assert_eq!(Modality::Unlikely.to_smtlib(), "unlikely");
    }

    // Test nested modal with negation
    #[tokio::test]
    async fn test_extract_modal_with_negation() {
        let extractor = AdvancedClaimExtractor::new();
        let claims = extractor
            .extract_claims("The result might not be accurate.", 10)
            .await
            .unwrap();

        assert!(!claims.is_empty());
        // Should wrap modal in negation
        assert!(matches!(claims[0].structure, ClaimStructure::Not(_)));
    }

    // Test combined extraction priority
    #[tokio::test]
    async fn test_extraction_priority() {
        let extractor = AdvancedClaimExtractor::new();

        // Modal should take priority over predicate
        let claims = extractor
            .extract_claims("The system might be operational.", 10)
            .await
            .unwrap();
        assert!(!claims.is_empty());
        assert!(matches!(claims[0].structure, ClaimStructure::Modal { .. }));
    }
}
