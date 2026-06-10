//! Chain-of-Verification engine.
use crate::advanced_retrieval::rag_fusion::reciprocal_rank_fusion;
use crate::chain_of_verification::types::{
    ChainOfVerificationError, ClaimVerdict, CoVeConfig, CoVeOutput, HeuristicQuestionPlanner,
    QuestionPlanner, VerificationAnswer,
};
use crate::types::SearchResult;

// ── helpers ───────────────────────────────────────────────────────────────────

fn jaccard(a: &str, b: &str) -> f32 {
    let ta: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let tb: std::collections::HashSet<&str> = b.split_whitespace().collect();
    if ta.is_empty() && tb.is_empty() {
        return 0.0;
    }
    let i = ta.intersection(&tb).count();
    let u = ta.union(&tb).count();
    if u == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        {
            i as f32 / u as f32
        }
    }
}

fn build_answer(claim: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No evidence found for: {claim}");
    }
    results
        .iter()
        .take(2)
        .map(|r| r.document.content.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn measure_faithfulness(answer: &str, results: &[SearchResult]) -> f32 {
    if results.is_empty() {
        return 0.0;
    }
    results
        .iter()
        .map(|r| jaccard(answer, &r.document.content))
        .fold(0.0_f32, f32::max)
}

// ── ChainOfVerificationEngine ─────────────────────────────────────────────────

/// Chain-of-Verification engine: draft → verify claims → revise.
///
/// The [`Echo`] layer is provided *per call* through the generic `run<E>` method.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
#[derive(Debug, Clone)]
pub struct ChainOfVerificationEngine {
    /// Configuration for this engine.
    pub config: CoVeConfig,
}

impl ChainOfVerificationEngine {
    /// Create a new engine with the given config.
    #[must_use]
    pub fn new(config: CoVeConfig) -> Self {
        Self { config }
    }

    /// Run Chain-of-Verification for `query`.
    ///
    /// # Errors
    ///
    /// Returns [`ChainOfVerificationError::EmptyQuery`] if `query` is empty.
    /// Returns [`ChainOfVerificationError::RetrievalFailed`] if retrieval fails.
    /// Returns [`ChainOfVerificationError::NoClaimsFound`] if no claims are extracted.
    pub async fn run<E>(
        &self,
        query: &str,
        echo: &E,
    ) -> Result<CoVeOutput, ChainOfVerificationError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(ChainOfVerificationError::EmptyQuery);
        }

        let init_results: Vec<SearchResult> = echo
            .search(query, self.config.top_k, None)
            .await
            .map_err(|e| ChainOfVerificationError::RetrievalFailed(e.to_string()))?;

        let baseline_answer = if init_results.is_empty() {
            format!("No information found for: {query}")
        } else {
            init_results
                .iter()
                .take(3)
                .map(|r| r.document.content.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };

        let planner = HeuristicQuestionPlanner::new(self.config.max_questions);
        let questions = planner.plan(&baseline_answer);
        if questions.is_empty() {
            return Err(ChainOfVerificationError::NoClaimsFound);
        }

        let baseline_faith = measure_faithfulness(&baseline_answer, &init_results);
        let mut verifications: Vec<VerificationAnswer> = Vec::new();
        let mut contradicted_claims = 0usize;

        for question in &questions {
            let vq_results: Vec<SearchResult> = echo
                .search(&question.text, self.config.top_k, None)
                .await
                .map_err(|e| ChainOfVerificationError::RetrievalFailed(e.to_string()))?;
            let answer = build_answer(&question.target_claim, &vq_results);
            let support_score = measure_faithfulness(&question.target_claim, &vq_results);
            let verdict = if support_score >= self.config.support_threshold {
                ClaimVerdict::Supported
            } else if support_score < self.config.support_threshold * 0.5 {
                ClaimVerdict::Contradicted
            } else {
                ClaimVerdict::Unverified
            };
            if verdict == ClaimVerdict::Contradicted {
                contradicted_claims += 1;
            }
            verifications.push(VerificationAnswer {
                question: question.clone(),
                answer,
                support_score,
                results: vq_results,
                verdict,
            });
        }

        let revised_answer = if self.config.revise && contradicted_claims > 0 {
            let all_results: Vec<Vec<SearchResult>> = verifications
                .iter()
                .filter(|v| v.verdict == ClaimVerdict::Supported)
                .map(|v| v.results.clone())
                .collect();
            if all_results.is_empty() {
                baseline_answer.clone()
            } else {
                let fused = reciprocal_rank_fusion(&all_results, 60.0);
                fused
                    .iter()
                    .take(3)
                    .map(|r| r.document.content.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        } else {
            baseline_answer.clone()
        };

        let revised_faith = measure_faithfulness(&revised_answer, &init_results);
        let faithfulness_delta = revised_faith - baseline_faith;

        Ok(CoVeOutput {
            baseline_answer,
            questions: questions.clone(),
            verifications,
            revised_answer,
            contradicted_claims,
            faithfulness_delta,
        })
    }
}

impl Default for ChainOfVerificationEngine {
    fn default() -> Self {
        Self::new(CoVeConfig::default())
    }
}
