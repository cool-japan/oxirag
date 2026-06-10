//! `LlmJudge` implementation.
use crate::llm_judge::types::{
    HeuristicJudge, JudgeContext, JudgeModel, LlmJudgeConfig, LlmJudgeError, PairwiseVerdict,
    PointwiseVerdict, Pref,
};

// ── LlmJudge ──────────────────────────────────────────────────────────────────

/// LLM-as-judge evaluator (heuristic implementation — no external LLM call).
#[derive(Debug, Clone)]
pub struct LlmJudge {
    /// Configuration for this judge.
    pub config: LlmJudgeConfig,
}

impl LlmJudge {
    /// Create a new judge with the given config.
    #[must_use]
    pub fn new(config: LlmJudgeConfig) -> Self {
        Self { config }
    }

    /// Score `answer` for `query` on a pointwise scale.
    ///
    /// # Errors
    ///
    /// Returns [`LlmJudgeError::EmptyAnswer`] if answer is empty.
    /// Returns [`LlmJudgeError::EmptyQuery`] if query is empty.
    /// Returns [`LlmJudgeError::EmptyRubric`] if rubric has no criteria.
    pub fn score_pointwise(
        &self,
        query: &str,
        answer: &str,
        context: &JudgeContext,
    ) -> Result<PointwiseVerdict, LlmJudgeError> {
        if query.trim().is_empty() {
            return Err(LlmJudgeError::EmptyQuery);
        }
        if answer.trim().is_empty() {
            return Err(LlmJudgeError::EmptyAnswer);
        }
        if self.config.rubric.criteria.is_empty() {
            return Err(LlmJudgeError::EmptyRubric);
        }
        let judge = HeuristicJudge;
        let scores = judge.grade(query, answer, context)?;
        let scaled = scores.overall * self.config.pointwise_scale;
        let rationale = format!(
            "Score {scaled:.2}/{:.1}: relevance={:.2}, grounding={:.2}",
            self.config.pointwise_scale,
            scores.scores.get("relevance").copied().unwrap_or(0.0),
            scores.scores.get("groundedness").copied().unwrap_or(0.0)
        );
        Ok(PointwiseVerdict {
            score: scaled,
            per_criterion: scores,
            rationale,
        })
    }

    /// Compare `answer_a` and `answer_b` and return which is preferred.
    ///
    /// # Errors
    ///
    /// Returns [`LlmJudgeError`] if either answer or query is empty.
    pub fn compare(
        &self,
        query: &str,
        answer_a: &str,
        answer_b: &str,
        context: &JudgeContext,
    ) -> Result<PairwiseVerdict, LlmJudgeError> {
        let va = self.score_pointwise(query, answer_a, context)?;
        let vb = self.score_pointwise(query, answer_b, context)?;
        let margin = (va.score - vb.score).abs();
        let winner = if margin < self.config.tie_margin * self.config.pointwise_scale {
            Pref::Tie
        } else if va.score >= vb.score {
            Pref::A
        } else {
            Pref::B
        };
        let rationale = format!("A={:.2} B={:.2} margin={margin:.2}", va.score, vb.score);
        Ok(PairwiseVerdict {
            winner,
            margin,
            rationale,
        })
    }

    /// Score `answer` against a `reference` answer.
    ///
    /// # Errors
    ///
    /// Returns [`LlmJudgeError`] if answer, query, or reference is empty.
    pub fn score_with_reference(
        &self,
        query: &str,
        answer: &str,
        reference: &str,
        context: &JudgeContext,
    ) -> Result<PointwiseVerdict, LlmJudgeError> {
        let mut ctx = context.clone();
        ctx.reference = Some(reference.to_string());
        self.score_pointwise(query, answer, &ctx)
    }
}

impl Default for LlmJudge {
    fn default() -> Self {
        Self::new(LlmJudgeConfig::default())
    }
}
