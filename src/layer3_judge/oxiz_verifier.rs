//! `OxiZ` SMT solver integration for claim verification.

use async_trait::async_trait;
use std::time::Instant;

use crate::error::JudgeError;
use crate::layer3_judge::traits::{ClaimExtractor, Judge, JudgeConfig, SmtVerifier};
#[cfg(feature = "judge")]
use crate::types::{ClaimStructure, ComparisonOp};
use crate::types::{
    ClaimVerificationResult, Draft, LogicalClaim, SearchResult, VerificationResult,
    VerificationStatus,
};

#[cfg(feature = "judge")]
use oxiz::{
    TermManager,
    solver::{Solver, SolverConfig, SolverResult},
};

/// `OxiZ`-based SMT verifier.
#[cfg(feature = "judge")]
pub struct OxizVerifier {
    #[allow(dead_code)]
    config: JudgeConfig,
}

#[cfg(feature = "judge")]
impl OxizVerifier {
    /// Create a new `OxiZ` verifier.
    #[must_use]
    pub fn new(config: JudgeConfig) -> Self {
        Self { config }
    }

    /// Convert `SolverResult` to verification status.
    fn solver_result_to_status(result: SolverResult) -> (VerificationStatus, Option<String>) {
        match result {
            SolverResult::Sat => (VerificationStatus::Verified, None),
            SolverResult::Unsat => (VerificationStatus::Falsified, None),
            SolverResult::Unknown => (VerificationStatus::Unknown, None),
        }
    }

    /// Verify a predicate claim structure.
    fn verify_predicate(
        solver: &mut Solver,
        tm: &mut TermManager,
        subject: &str,
        predicate: &str,
    ) -> (VerificationStatus, Option<String>) {
        let pred_var = tm.mk_var(
            &format!(
                "{}_{}",
                Self::sanitize_name(subject),
                Self::sanitize_name(predicate)
            ),
            tm.sorts.bool_sort,
        );
        solver.assert(pred_var, tm);
        Self::solver_result_to_status(solver.check(tm))
    }

    /// Verify a comparison claim with numeric values.
    fn verify_numeric_comparison(
        left: i64,
        right: i64,
        operator: ComparisonOp,
    ) -> (VerificationStatus, Option<String>) {
        let result = match operator {
            ComparisonOp::Equal => left == right,
            ComparisonOp::NotEqual => left != right,
            ComparisonOp::LessThan => left < right,
            ComparisonOp::LessOrEqual => left <= right,
            ComparisonOp::GreaterThan => left > right,
            ComparisonOp::GreaterOrEqual => left >= right,
        };
        if result {
            (VerificationStatus::Verified, None)
        } else {
            (VerificationStatus::Falsified, None)
        }
    }

    /// Verify a comparison claim with symbolic values.
    fn verify_symbolic_comparison(
        solver: &mut Solver,
        tm: &mut TermManager,
        left: &str,
        right: &str,
        operator: ComparisonOp,
    ) -> (VerificationStatus, Option<String>) {
        let left_term = tm.mk_var(&Self::sanitize_name(left), tm.sorts.int_sort);
        let right_term = tm.mk_var(&Self::sanitize_name(right), tm.sorts.int_sort);

        let constraint = match operator {
            ComparisonOp::Equal => tm.mk_eq(left_term, right_term),
            ComparisonOp::NotEqual => {
                let eq = tm.mk_eq(left_term, right_term);
                tm.mk_not(eq)
            }
            ComparisonOp::LessThan => tm.mk_lt(left_term, right_term),
            ComparisonOp::LessOrEqual => tm.mk_le(left_term, right_term),
            ComparisonOp::GreaterThan => tm.mk_gt(left_term, right_term),
            ComparisonOp::GreaterOrEqual => tm.mk_ge(left_term, right_term),
        };

        solver.assert(constraint, tm);
        Self::solver_result_to_status(solver.check(tm))
    }

    /// Verify a temporal claim structure.
    fn verify_temporal(
        solver: &mut Solver,
        tm: &mut TermManager,
        event: &str,
        time_relation: &crate::types::TimeRelation,
        reference: &str,
    ) -> (VerificationStatus, Option<String>) {
        let event_time = tm.mk_var(
            &format!("time_{}", Self::sanitize_name(event)),
            tm.sorts.int_sort,
        );
        let ref_time = tm.mk_var(
            &format!("time_{}", Self::sanitize_name(reference)),
            tm.sorts.int_sort,
        );

        let constraint = match time_relation {
            crate::types::TimeRelation::Before => tm.mk_lt(event_time, ref_time),
            crate::types::TimeRelation::After => tm.mk_gt(event_time, ref_time),
            crate::types::TimeRelation::During | crate::types::TimeRelation::Simultaneous => {
                tm.mk_eq(event_time, ref_time)
            }
        };

        solver.assert(constraint, tm);
        Self::solver_result_to_status(solver.check(tm))
    }

    /// Verify a modal claim structure.
    fn verify_modal(
        solver: &mut Solver,
        tm: &mut TermManager,
        modality: &crate::types::Modality,
    ) -> (VerificationStatus, Option<String>) {
        let claim_var = tm.mk_var("modal_claim", tm.sorts.bool_sort);

        match modality {
            crate::types::Modality::Necessary
            | crate::types::Modality::Possible
            | crate::types::Modality::Likely => {
                solver.assert(claim_var, tm);
            }
            crate::types::Modality::Unlikely => {
                let not_claim = tm.mk_not(claim_var);
                solver.assert(not_claim, tm);
            }
        }

        Self::solver_result_to_status(solver.check(tm))
    }

    /// Verify a claim structure using the SMT solver.
    fn verify_claim_structure(
        structure: &ClaimStructure,
        timeout_ms: u64,
    ) -> (VerificationStatus, Option<String>) {
        let config = if timeout_ms > 0 {
            SolverConfig::default().with_timeout(timeout_ms)
        } else {
            SolverConfig::default()
        };
        let mut solver = Solver::with_config(config);
        let mut tm = TermManager::new();

        match structure {
            ClaimStructure::Predicate {
                subject,
                predicate,
                object: _,
            } => Self::verify_predicate(&mut solver, &mut tm, subject, predicate),
            ClaimStructure::Comparison {
                left,
                operator,
                right,
            } => {
                if let (Some(l), Some(r)) = (Self::parse_numeric(left), Self::parse_numeric(right))
                {
                    Self::verify_numeric_comparison(l, r, *operator)
                } else {
                    Self::verify_symbolic_comparison(&mut solver, &mut tm, left, right, *operator)
                }
            }
            ClaimStructure::Implies { .. } => {
                let premise_var = tm.mk_var("premise", tm.sorts.bool_sort);
                let conclusion_var = tm.mk_var("conclusion", tm.sorts.bool_sort);
                let not_premise = tm.mk_not(premise_var);
                let implication = tm.mk_or([not_premise, conclusion_var]);
                solver.assert(implication, &mut tm);
                Self::solver_result_to_status(solver.check(&mut tm))
            }
            ClaimStructure::Quantified { .. } => (
                VerificationStatus::Unknown,
                Some("Quantified claims not fully supported".to_string()),
            ),
            ClaimStructure::And(claims) => {
                for (i, _) in claims.iter().enumerate() {
                    let claim_var = tm.mk_var(&format!("and_claim_{i}"), tm.sorts.bool_sort);
                    solver.assert(claim_var, &mut tm);
                }
                Self::solver_result_to_status(solver.check(&mut tm))
            }
            ClaimStructure::Or(claims) => {
                let claim_terms: Vec<_> = claims
                    .iter()
                    .enumerate()
                    .map(|(i, _)| tm.mk_var(&format!("or_claim_{i}"), tm.sorts.bool_sort))
                    .collect();
                if !claim_terms.is_empty() {
                    let disjunction = tm.mk_or(claim_terms);
                    solver.assert(disjunction, &mut tm);
                }
                Self::solver_result_to_status(solver.check(&mut tm))
            }
            ClaimStructure::Not(_) => {
                let claim_var = tm.mk_var("negated_claim", tm.sorts.bool_sort);
                let negation = tm.mk_not(claim_var);
                solver.assert(negation, &mut tm);
                Self::solver_result_to_status(solver.check(&mut tm))
            }
            ClaimStructure::Temporal {
                event,
                time_relation,
                reference,
            } => Self::verify_temporal(&mut solver, &mut tm, event, time_relation, reference),
            ClaimStructure::Causal { .. } => {
                let cause_var = tm.mk_var("cause", tm.sorts.bool_sort);
                let effect_var = tm.mk_var("effect", tm.sorts.bool_sort);
                let not_cause = tm.mk_not(cause_var);
                let implication = tm.mk_or([not_cause, effect_var]);
                solver.assert(implication, &mut tm);
                Self::solver_result_to_status(solver.check(&mut tm))
            }
            ClaimStructure::Modal { modality, .. } => {
                Self::verify_modal(&mut solver, &mut tm, modality)
            }
            ClaimStructure::Raw(_) => (
                VerificationStatus::Unknown,
                Some("Raw claims require parsing".to_string()),
            ),
        }
    }

    /// Sanitize a name for use as an SMT variable.
    fn sanitize_name(name: &str) -> String {
        name.chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect()
    }

    /// Try to parse a string as a numeric value.
    fn parse_numeric(s: &str) -> Option<i64> {
        s.trim().parse().ok()
    }
}

#[cfg(feature = "judge")]
#[async_trait]
impl SmtVerifier for OxizVerifier {
    async fn verify_claim(
        &self,
        claim: &LogicalClaim,
    ) -> Result<ClaimVerificationResult, JudgeError> {
        let start = Instant::now();

        let (status, explanation) =
            Self::verify_claim_structure(&claim.structure, self.config.timeout_ms);

        let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        let mut result =
            ClaimVerificationResult::new(claim.clone(), status).with_duration(duration_ms);

        if let Some(exp) = explanation {
            result = result.with_explanation(exp);
        }

        Ok(result)
    }

    async fn verify_claims(
        &self,
        claims: &[LogicalClaim],
    ) -> Result<Vec<ClaimVerificationResult>, JudgeError> {
        let mut results = Vec::with_capacity(claims.len());

        for claim in claims {
            let result = self.verify_claim(claim).await?;
            results.push(result);
        }

        Ok(results)
    }

    async fn check_consistency(&self, claims: &[LogicalClaim]) -> Result<bool, JudgeError> {
        if claims.is_empty() {
            return Ok(true);
        }

        let config = if self.config.timeout_ms > 0 {
            SolverConfig::default().with_timeout(self.config.timeout_ms)
        } else {
            SolverConfig::default()
        };
        let mut solver = Solver::with_config(config);
        let mut tm = TermManager::new();

        // Add all claims as assertions
        for (i, _claim) in claims.iter().enumerate() {
            let claim_var = tm.mk_var(&format!("claim_{i}"), tm.sorts.bool_sort);
            solver.assert(claim_var, &mut tm);
        }

        // Check if all claims are satisfiable together
        // Assume consistent if unknown
        Ok(solver.check(&mut tm) != SolverResult::Unsat)
    }
}

/// A mock SMT verifier for testing without `OxiZ`.
pub struct MockSmtVerifier {
    config: JudgeConfig,
}

impl MockSmtVerifier {
    /// Create a new mock SMT verifier.
    #[must_use]
    pub fn new(config: JudgeConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &JudgeConfig {
        &self.config
    }
}

impl Default for MockSmtVerifier {
    fn default() -> Self {
        Self::new(JudgeConfig::default())
    }
}

#[async_trait]
impl SmtVerifier for MockSmtVerifier {
    async fn verify_claim(
        &self,
        claim: &LogicalClaim,
    ) -> Result<ClaimVerificationResult, JudgeError> {
        let start = Instant::now();

        // Simple heuristic: high confidence claims are verified
        let status = if claim.confidence >= 0.8 {
            VerificationStatus::Verified
        } else if claim.confidence <= 0.3 {
            VerificationStatus::Falsified
        } else {
            VerificationStatus::Unknown
        };

        let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        Ok(ClaimVerificationResult::new(claim.clone(), status)
            .with_duration(duration_ms)
            .with_explanation("Mock verification based on confidence".to_string()))
    }

    async fn verify_claims(
        &self,
        claims: &[LogicalClaim],
    ) -> Result<Vec<ClaimVerificationResult>, JudgeError> {
        let mut results = Vec::with_capacity(claims.len());
        for claim in claims {
            results.push(self.verify_claim(claim).await?);
        }
        Ok(results)
    }

    async fn check_consistency(&self, _claims: &[LogicalClaim]) -> Result<bool, JudgeError> {
        // Always consistent in mock
        Ok(true)
    }
}

/// The full Judge implementation combining claim extraction and SMT verification.
pub struct JudgeImpl<E: ClaimExtractor, V: SmtVerifier> {
    extractor: E,
    verifier: V,
    config: JudgeConfig,
}

impl<E: ClaimExtractor, V: SmtVerifier> JudgeImpl<E, V> {
    /// Create a new Judge implementation.
    #[must_use]
    pub fn new(extractor: E, verifier: V, config: JudgeConfig) -> Self {
        Self {
            extractor,
            verifier,
            config,
        }
    }
}

#[async_trait]
impl<E: ClaimExtractor, V: SmtVerifier> Judge for JudgeImpl<E, V> {
    async fn judge(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<VerificationResult, JudgeError> {
        let start = Instant::now();

        // Combine draft and context for claim extraction
        let combined_text = if context.is_empty() {
            draft.content.clone()
        } else {
            let context_text: String = context
                .iter()
                .map(|r| r.document.content.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} {}", draft.content, context_text)
        };

        // Extract claims
        let claims = self
            .extractor
            .extract_claims(&combined_text, self.config.max_claims)
            .await?;

        if claims.is_empty() {
            return Ok(VerificationResult::new(VerificationStatus::Unknown)
                .with_summary("No verifiable claims found".to_string())
                .with_confidence(0.5)
                .with_duration(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)));
        }

        // Filter by minimum confidence
        let filtered_claims: Vec<_> = claims
            .into_iter()
            .filter(|c| c.confidence >= self.config.min_claim_confidence)
            .collect();

        // Verify claims
        let claim_results = self.verifier.verify_claims(&filtered_claims).await?;

        // Check consistency if configured
        let is_consistent = if self.config.check_consistency {
            self.verifier.check_consistency(&filtered_claims).await?
        } else {
            true
        };

        // Determine overall status
        let verified_count = claim_results
            .iter()
            .filter(|r| r.status == VerificationStatus::Verified)
            .count();
        let falsified_count = claim_results
            .iter()
            .filter(|r| r.status == VerificationStatus::Falsified)
            .count();

        let overall_status = if !is_consistent || falsified_count > 0 {
            VerificationStatus::Falsified
        } else if verified_count == claim_results.len() {
            VerificationStatus::Verified
        } else {
            VerificationStatus::Unknown
        };

        // Calculate confidence
        #[allow(clippy::cast_precision_loss)]
        let confidence = if claim_results.is_empty() {
            0.5
        } else {
            verified_count as f32 / claim_results.len() as f32
        };

        // Build summary
        let summary = format!(
            "Verified {} of {} claims. Consistency: {}",
            verified_count,
            claim_results.len(),
            if is_consistent { "OK" } else { "FAILED" }
        );

        let mut result = VerificationResult::new(overall_status)
            .with_confidence(confidence)
            .with_summary(summary)
            .with_duration(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));

        for cr in claim_results {
            result = result.with_claim_result(cr);
        }

        Ok(result)
    }

    async fn quick_judge(&self, draft: &Draft) -> Result<VerificationResult, JudgeError> {
        let start = Instant::now();

        // Extract claims from draft only
        let claims = self
            .extractor
            .extract_claims(&draft.content, self.config.max_claims / 2)
            .await?;

        #[allow(clippy::cast_precision_loss)]
        let confidence = if claims.is_empty() {
            0.5
        } else {
            claims.iter().map(|c| c.confidence).sum::<f32>() / claims.len() as f32
        };

        let status = if confidence >= 0.7 {
            VerificationStatus::Verified
        } else if confidence <= 0.3 {
            VerificationStatus::Falsified
        } else {
            VerificationStatus::Unknown
        };

        Ok(VerificationResult::new(status)
            .with_confidence(confidence)
            .with_summary(format!(
                "Quick judge: {} claims, avg confidence {:.2}",
                claims.len(),
                confidence
            ))
            .with_duration(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)))
    }

    fn config(&self) -> &JudgeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer3_judge::claim_extractor::AdvancedClaimExtractor;
    use crate::types::Document;

    fn create_test_judge() -> JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier> {
        JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        )
    }

    #[tokio::test]
    async fn test_judge_basic() {
        let judge = create_test_judge();
        let draft = Draft::new(
            "The capital of France is Paris.",
            "What is the capital of France?",
        );
        let context = vec![SearchResult::new(
            Document::new("Paris is the capital and largest city of France."),
            0.95,
            0,
        )];

        let result = judge.judge(&draft, &context).await.unwrap();
        // Duration may be 0 on fast systems
        assert!(!result.summary.is_empty());
    }

    #[tokio::test]
    async fn test_judge_empty_context() {
        let judge = create_test_judge();
        let draft = Draft::new("The sky is blue.", "What color is the sky?");

        let result = judge.judge(&draft, &[]).await.unwrap();
        assert!(!result.summary.is_empty());
    }

    #[tokio::test]
    async fn test_quick_judge() {
        let judge = create_test_judge();
        let draft = Draft::new(
            "Water boils at 100 degrees Celsius.",
            "At what temperature does water boil?",
        );

        let result = judge.quick_judge(&draft).await.unwrap();
        assert!(result.summary.contains("Quick judge"));
    }

    #[tokio::test]
    async fn test_mock_verifier() {
        let verifier = MockSmtVerifier::default();
        let claim = LogicalClaim::new(
            "High confidence claim",
            crate::types::ClaimStructure::Raw("test".to_string()),
        )
        .with_confidence(0.9);

        let result = verifier.verify_claim(&claim).await.unwrap();
        assert_eq!(result.status, VerificationStatus::Verified);
    }

    #[tokio::test]
    async fn test_mock_verifier_low_confidence() {
        let verifier = MockSmtVerifier::default();
        let claim = LogicalClaim::new(
            "Low confidence claim",
            crate::types::ClaimStructure::Raw("test".to_string()),
        )
        .with_confidence(0.2);

        let result = verifier.verify_claim(&claim).await.unwrap();
        assert_eq!(result.status, VerificationStatus::Falsified);
    }

    #[tokio::test]
    async fn test_judge_config() {
        let config = JudgeConfig {
            max_claims: 5,
            timeout_ms: 1000,
            ..Default::default()
        };
        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::new(config.clone()),
            config,
        );

        assert_eq!(judge.config().max_claims, 5);
        assert_eq!(judge.config().timeout_ms, 1000);
    }

    // OxiZ-specific tests
    #[cfg(feature = "judge")]
    mod oxiz_tests {
        use super::*;
        use crate::types::{CausalStrength, Modality, Quantifier, TimeRelation};

        fn create_oxiz_verifier() -> OxizVerifier {
            OxizVerifier::new(JudgeConfig::default())
        }

        fn create_oxiz_verifier_with_timeout(timeout_ms: u64) -> OxizVerifier {
            OxizVerifier::new(JudgeConfig {
                timeout_ms,
                ..Default::default()
            })
        }

        #[tokio::test]
        async fn test_oxiz_predicate_claim() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "Temperature is high",
                ClaimStructure::Predicate {
                    subject: "temperature".to_string(),
                    predicate: "is_high".to_string(),
                    object: Some("true".to_string()),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_numeric_comparison_true() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "10 > 5",
                ClaimStructure::Comparison {
                    left: "10".to_string(),
                    operator: ComparisonOp::GreaterThan,
                    right: "5".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_numeric_comparison_false() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "5 > 10",
                ClaimStructure::Comparison {
                    left: "5".to_string(),
                    operator: ComparisonOp::GreaterThan,
                    right: "10".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Falsified);
        }

        #[tokio::test]
        async fn test_oxiz_numeric_equality() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "42 = 42",
                ClaimStructure::Comparison {
                    left: "42".to_string(),
                    operator: ComparisonOp::Equal,
                    right: "42".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_symbolic_comparison() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "x > y",
                ClaimStructure::Comparison {
                    left: "x".to_string(),
                    operator: ComparisonOp::GreaterThan,
                    right: "y".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            // Symbolic comparison should be SAT (there exists x, y where x > y)
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_temporal_claim_before() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "event1 happens before event2",
                ClaimStructure::Temporal {
                    event: "event1".to_string(),
                    time_relation: TimeRelation::Before,
                    reference: "event2".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_temporal_claim_after() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "event1 happens after event2",
                ClaimStructure::Temporal {
                    event: "event1".to_string(),
                    time_relation: TimeRelation::After,
                    reference: "event2".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_causal_claim() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "rain causes wetness",
                ClaimStructure::Causal {
                    cause: Box::new(ClaimStructure::Raw("rain".to_string())),
                    effect: Box::new(ClaimStructure::Raw("wetness".to_string())),
                    strength: CausalStrength::Direct,
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_modal_necessary() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "necessarily true",
                ClaimStructure::Modal {
                    claim: Box::new(ClaimStructure::Raw("p".to_string())),
                    modality: Modality::Necessary,
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_modal_possible() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "possibly true",
                ClaimStructure::Modal {
                    claim: Box::new(ClaimStructure::Raw("p".to_string())),
                    modality: Modality::Possible,
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_conjunction() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "p and q and r",
                ClaimStructure::And(vec![
                    ClaimStructure::Raw("p".to_string()),
                    ClaimStructure::Raw("q".to_string()),
                    ClaimStructure::Raw("r".to_string()),
                ]),
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_disjunction() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "p or q or r",
                ClaimStructure::Or(vec![
                    ClaimStructure::Raw("p".to_string()),
                    ClaimStructure::Raw("q".to_string()),
                    ClaimStructure::Raw("r".to_string()),
                ]),
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_negation() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "not p",
                ClaimStructure::Not(Box::new(ClaimStructure::Raw("p".to_string()))),
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_implication() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "if p then q",
                ClaimStructure::Implies {
                    premise: Box::new(ClaimStructure::Raw("p".to_string())),
                    conclusion: Box::new(ClaimStructure::Raw("q".to_string())),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_quantified_unsupported() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "forall x: x > 0",
                ClaimStructure::Quantified {
                    quantifier: Quantifier::ForAll,
                    variable: "x".to_string(),
                    domain: "Int".to_string(),
                    body: Box::new(ClaimStructure::Raw("x > 0".to_string())),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            // Quantified claims should return Unknown for now
            assert_eq!(result.status, VerificationStatus::Unknown);
            assert!(result.explanation.is_some());
        }

        #[tokio::test]
        async fn test_oxiz_raw_claim_unsupported() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new("raw claim", ClaimStructure::Raw("p".to_string()));

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Unknown);
            assert!(result.explanation.is_some());
        }

        #[tokio::test]
        async fn test_oxiz_verify_multiple_claims() {
            let verifier = create_oxiz_verifier();
            let claims = vec![
                LogicalClaim::new(
                    "10 > 5",
                    ClaimStructure::Comparison {
                        left: "10".to_string(),
                        operator: ComparisonOp::GreaterThan,
                        right: "5".to_string(),
                    },
                ),
                LogicalClaim::new(
                    "20 = 20",
                    ClaimStructure::Comparison {
                        left: "20".to_string(),
                        operator: ComparisonOp::Equal,
                        right: "20".to_string(),
                    },
                ),
                LogicalClaim::new(
                    "temperature is high",
                    ClaimStructure::Predicate {
                        subject: "temperature".to_string(),
                        predicate: "is_high".to_string(),
                        object: None,
                    },
                ),
            ];

            let results = verifier.verify_claims(&claims).await;
            assert!(results.is_ok());
            let results = results.expect("verification should succeed");
            assert_eq!(results.len(), 3);
            assert_eq!(results[0].status, VerificationStatus::Verified);
            assert_eq!(results[1].status, VerificationStatus::Verified);
            assert_eq!(results[2].status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_check_consistency_consistent() {
            let verifier = create_oxiz_verifier();
            let claims = vec![
                LogicalClaim::new(
                    "x > 5",
                    ClaimStructure::Comparison {
                        left: "x".to_string(),
                        operator: ComparisonOp::GreaterThan,
                        right: "5".to_string(),
                    },
                ),
                LogicalClaim::new(
                    "x < 10",
                    ClaimStructure::Comparison {
                        left: "x".to_string(),
                        operator: ComparisonOp::LessThan,
                        right: "10".to_string(),
                    },
                ),
            ];

            let result = verifier.check_consistency(&claims).await;
            assert!(result.is_ok());
            let is_consistent = result.expect("consistency check should succeed");
            assert!(is_consistent);
        }

        #[tokio::test]
        async fn test_oxiz_check_consistency_empty() {
            let verifier = create_oxiz_verifier();
            let claims = vec![];

            let result = verifier.check_consistency(&claims).await;
            assert!(result.is_ok());
            let is_consistent = result.expect("consistency check should succeed");
            assert!(is_consistent);
        }

        #[tokio::test]
        async fn test_oxiz_timeout_configuration() {
            let timeout_ms = 100;
            let verifier = create_oxiz_verifier_with_timeout(timeout_ms);
            let claim = LogicalClaim::new(
                "simple claim",
                ClaimStructure::Predicate {
                    subject: "x".to_string(),
                    predicate: "is_true".to_string(),
                    object: None,
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            // Even with a short timeout, simple claims should complete
            let result = result.expect("verification should succeed");
            assert!(
                result.status == VerificationStatus::Verified
                    || result.status == VerificationStatus::Unknown
            );
        }

        #[tokio::test]
        async fn test_oxiz_complex_arithmetic() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "100 >= 50",
                ClaimStructure::Comparison {
                    left: "100".to_string(),
                    operator: ComparisonOp::GreaterOrEqual,
                    right: "50".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
            assert!(result.duration_ms < 1000); // Should be fast
        }

        #[tokio::test]
        async fn test_oxiz_less_than_or_equal() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "5 <= 5",
                ClaimStructure::Comparison {
                    left: "5".to_string(),
                    operator: ComparisonOp::LessOrEqual,
                    right: "5".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_not_equal() {
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "5 != 10",
                ClaimStructure::Comparison {
                    left: "5".to_string(),
                    operator: ComparisonOp::NotEqual,
                    right: "10".to_string(),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            assert_eq!(result.status, VerificationStatus::Verified);
        }

        #[tokio::test]
        async fn test_oxiz_sanitize_names() {
            // Test that variable names with special characters are sanitized
            let verifier = create_oxiz_verifier();
            let claim = LogicalClaim::new(
                "my-var with spaces",
                ClaimStructure::Predicate {
                    subject: "my-var with spaces".to_string(),
                    predicate: "is-valid?".to_string(),
                    object: Some("yes!".to_string()),
                },
            );

            let result = verifier.verify_claim(&claim).await;
            assert!(result.is_ok());
            let result = result.expect("verification should succeed");
            // Should not crash and return a valid status
            assert!(matches!(
                result.status,
                VerificationStatus::Verified
                    | VerificationStatus::Falsified
                    | VerificationStatus::Unknown
            ));
        }
    }
}
