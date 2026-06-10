//! Types for the `llm_judge` module.
use std::collections::HashMap;
use thiserror::Error;
// ── JudgeMode ─────────────────────────────────────────────────────────────────
/// Evaluation mode for `LlmJudge`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum JudgeMode {
    /// Score a single answer against a rubric.
    #[default]
    Pointwise,
    /// Compare two answers and pick the better one.
    Pairwise,
    /// Score an answer against a reference answer.
    Reference,
}
// ── Criterion ─────────────────────────────────────────────────────────────────
/// A single evaluation criterion.
#[derive(Debug, Clone)]
pub struct Criterion {
    /// Short identifier (e.g. `"relevance"`).
    pub name: String,
    /// Weight of this criterion in the overall score. Normalised by [`Rubric::normalize`].
    pub weight: f32,
    /// Human-readable description of what this criterion measures.
    pub description: String,
}
impl Criterion {
    /// Create a new criterion.
    #[must_use]
    pub fn new(name: impl Into<String>, weight: f32, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            weight,
            description: description.into(),
        }
    }
}
// ── Rubric ────────────────────────────────────────────────────────────────────
/// A set of criteria defining how answers are evaluated.
#[derive(Debug, Clone, Default)]
pub struct Rubric {
    /// The evaluation criteria.
    pub criteria: Vec<Criterion>,
}
impl Rubric {
    /// Create a rubric with the given criteria.
    #[must_use]
    pub fn new(criteria: Vec<Criterion>) -> Self {
        Self { criteria }
    }
    /// Built-in: relevance + helpfulness + groundedness.
    #[must_use]
    pub fn relevance_helpfulness_groundedness() -> Self {
        Self::new(vec![
            Criterion::new(
                "relevance",
                0.35,
                "How relevant is the answer to the query?",
            ),
            Criterion::new(
                "helpfulness",
                0.35,
                "How helpful and actionable is the answer?",
            ),
            Criterion::new(
                "groundedness",
                0.30,
                "How well is the answer grounded in sources?",
            ),
        ])
    }
    /// Built-in: factual correctness rubric.
    #[must_use]
    pub fn correctness() -> Self {
        Self::new(vec![
            Criterion::new("accuracy", 0.50, "Is the answer factually accurate?"),
            Criterion::new(
                "completeness",
                0.30,
                "Does the answer cover all key points?",
            ),
            Criterion::new(
                "precision",
                0.20,
                "Is the answer concise and free of hallucinations?",
            ),
        ])
    }
    /// Normalise criterion weights so they sum to 1.0.
    #[must_use]
    pub fn normalize(mut self) -> Self {
        let s: f32 = self.criteria.iter().map(|c| c.weight).sum();
        if s > 0.0 {
            for c in &mut self.criteria {
                c.weight /= s;
            }
        }
        self
    }
}
// ── CriterionScores ───────────────────────────────────────────────────────────
/// Scores for each criterion in a rubric.
#[derive(Debug, Clone, Default)]
pub struct CriterionScores {
    /// Per-criterion scores in [0.0, 1.0].
    pub scores: HashMap<String, f32>,
    /// Weighted overall score.
    pub overall: f32,
}
// ── JudgeContext ──────────────────────────────────────────────────────────────
/// Contextual information passed to a judge.
#[derive(Debug, Clone, Default)]
pub struct JudgeContext {
    /// Source documents used for retrieval.
    pub sources: Vec<crate::types::SearchResult>,
    /// Reference answer for [`JudgeMode::Reference`] evaluation.
    pub reference: Option<String>,
}
impl JudgeContext {
    /// Create an empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Attach source documents.
    #[must_use]
    pub fn with_sources(mut self, sources: Vec<crate::types::SearchResult>) -> Self {
        self.sources = sources;
        self
    }
    /// Attach a reference answer.
    #[must_use]
    pub fn with_reference(mut self, reference: impl Into<String>) -> Self {
        self.reference = Some(reference.into());
        self
    }
}
// ── Pref ──────────────────────────────────────────────────────────────────────
/// Pairwise preference verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pref {
    /// Answer A is preferred.
    A,
    /// Answer B is preferred.
    B,
    /// Neither is clearly better.
    Tie,
}
// ── PointwiseVerdict ──────────────────────────────────────────────────────────
/// Pointwise evaluation result.
#[derive(Debug, Clone)]
pub struct PointwiseVerdict {
    /// Weighted overall score in [0.0, `pointwise_scale`].
    pub score: f32,
    /// Per-criterion scores.
    pub per_criterion: CriterionScores,
    /// Human-readable rationale.
    pub rationale: String,
}
// ── PairwiseVerdict ───────────────────────────────────────────────────────────
/// Pairwise comparison result.
#[derive(Debug, Clone)]
pub struct PairwiseVerdict {
    /// Which answer was preferred.
    pub winner: Pref,
    /// Score margin (absolute difference).
    pub margin: f32,
    /// Human-readable rationale.
    pub rationale: String,
}
// ── JudgeConfig ───────────────────────────────────────────────────────────────
/// Configuration for `LlmJudge`.
#[derive(Debug, Clone)]
pub struct LlmJudgeConfig {
    /// Rubric used for evaluation.
    pub rubric: Rubric,
    /// Score margin below which pairwise results are called a tie. Defaults to `0.05`.
    pub tie_margin: f32,
    /// Scale for pointwise scores (multiplier on `[0,1]` scores). Defaults to `5.0`.
    pub pointwise_scale: f32,
}
impl Default for LlmJudgeConfig {
    fn default() -> Self {
        Self {
            rubric: Rubric::relevance_helpfulness_groundedness(),
            tie_margin: 0.05,
            pointwise_scale: 5.0,
        }
    }
}
impl LlmJudgeConfig {
    /// Set the rubric.
    #[must_use]
    pub fn with_rubric(mut self, v: Rubric) -> Self {
        self.rubric = v;
        self
    }
    /// Set the tie margin.
    #[must_use]
    pub fn with_tie_margin(mut self, v: f32) -> Self {
        self.tie_margin = v;
        self
    }
    /// Set the pointwise scale.
    #[must_use]
    pub fn with_pointwise_scale(mut self, v: f32) -> Self {
        self.pointwise_scale = v;
        self
    }
}
// ── JudgeModel ────────────────────────────────────────────────────────────────
/// Synchronous judge that assigns criterion scores.
pub trait JudgeModel {
    /// Grade `answer` for `query` against `context`.
    ///
    /// # Errors
    ///
    /// Returns [`LlmJudgeError`] on evaluation errors.
    fn grade(
        &self,
        query: &str,
        answer: &str,
        context: &JudgeContext,
    ) -> Result<CriterionScores, LlmJudgeError>;
}
// ── helpers ───────────────────────────────────────────────────────────────────

fn jaccard_similarity(a: &str, b: &str) -> f32 {
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

// ── HeuristicJudge ────────────────────────────────────────────────────────────
/// Lexical heuristic implementation of [`JudgeModel`].
#[derive(Debug, Clone, Default)]
pub struct HeuristicJudge;
impl JudgeModel for HeuristicJudge {
    fn grade(
        &self,
        query: &str,
        answer: &str,
        context: &JudgeContext,
    ) -> Result<CriterionScores, LlmJudgeError> {
        if answer.trim().is_empty() {
            return Err(LlmJudgeError::EmptyAnswer);
        }
        let relevance = jaccard_similarity(query, answer);
        let groundedness = context
            .sources
            .iter()
            .map(|r| jaccard_similarity(answer, &r.document.content))
            .fold(0.0_f32, f32::max);
        let coherence = {
            let sents: Vec<&str> = answer.split(". ").collect();
            if sents.len() <= 1 { 0.8 } else { 0.7 }
        };
        let conciseness = {
            let words = answer.split_whitespace().count();
            if words > 300 {
                0.4
            } else if words > 100 {
                0.7
            } else {
                0.9
            }
        };
        let overall =
            (relevance * 0.35 + groundedness * 0.30 + coherence * 0.20 + conciseness * 0.15)
                .min(1.0);
        let mut scores_map = HashMap::new();
        scores_map.insert("relevance".to_string(), relevance);
        scores_map.insert("groundedness".to_string(), groundedness);
        scores_map.insert("coherence".to_string(), coherence);
        scores_map.insert("conciseness".to_string(), conciseness);
        Ok(CriterionScores {
            scores: scores_map,
            overall,
        })
    }
}
// ── LlmJudgeError ─────────────────────────────────────────────────────────────
/// Errors from the `llm_judge` module.
#[derive(Debug, Error)]
pub enum LlmJudgeError {
    /// The answer string was empty.
    #[error("Answer must not be empty")]
    EmptyAnswer,
    /// The rubric had no criteria.
    #[error("Rubric must contain at least one criterion")]
    EmptyRubric,
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
}
