//! Types for the `chain_of_verification` module.
use thiserror::Error;
// ── ClaimVerdict ──────────────────────────────────────────────────────────────
/// Verdict on a single verification question.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ClaimVerdict {
    /// The claim is supported by retrieved evidence.
    Supported,
    /// The claim is contradicted by retrieved evidence.
    Contradicted,
    /// Insufficient evidence to decide.
    #[default]
    Unverified,
}
impl ClaimVerdict {
    /// Return a short string label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Supported => "supported",
            Self::Contradicted => "contradicted",
            Self::Unverified => "unverified",
        }
    }
}
// ── VerificationQuestion ──────────────────────────────────────────────────────
/// A factual probe generated from a draft answer claim.
#[derive(Debug, Clone)]
pub struct VerificationQuestion {
    /// The question text.
    pub text: String,
    /// The claim this question targets.
    pub target_claim: String,
    /// Zero-based index among all verification questions.
    pub index: usize,
}
// ── VerificationAnswer ────────────────────────────────────────────────────────
/// The answer to a single verification question.
#[derive(Debug, Clone)]
pub struct VerificationAnswer {
    /// The question that was answered.
    pub question: VerificationQuestion,
    /// The short answer produced.
    pub answer: String,
    /// Jaccard-based support score in [0.0, 1.0].
    pub support_score: f32,
    /// Documents retrieved to answer this question.
    pub results: Vec<crate::types::SearchResult>,
    /// The verdict for this claim.
    pub verdict: ClaimVerdict,
}
// ── QuestionPlanner ───────────────────────────────────────────────────────────
/// Synchronous trait for planning verification questions from a draft answer.
pub trait QuestionPlanner {
    /// Generate a list of factual verification questions from `answer`.
    ///
    /// Always returns at least one question (falls back to a whole-answer probe).
    fn plan(&self, answer: &str) -> Vec<VerificationQuestion>;
}
// ── HeuristicQuestionPlanner ──────────────────────────────────────────────────
/// Heuristic question planner: splits on claim-boundary signals.
#[derive(Debug, Clone, Default)]
pub struct HeuristicQuestionPlanner {
    /// Maximum questions to generate. Defaults to `5`.
    pub max_questions: usize,
}
impl HeuristicQuestionPlanner {
    /// Create a new planner.
    #[must_use]
    pub fn new(max_questions: usize) -> Self {
        Self { max_questions }
    }
}
impl QuestionPlanner for HeuristicQuestionPlanner {
    fn plan(&self, answer: &str) -> Vec<VerificationQuestion> {
        if answer.trim().is_empty() {
            return vec![VerificationQuestion {
                text: "Is this correct?".into(),
                target_claim: String::new(),
                index: 0,
            }];
        }
        let max = if self.max_questions == 0 {
            5
        } else {
            self.max_questions
        };
        let sentences: Vec<&str> = answer
            .split(". ")
            .filter(|s| !s.trim().is_empty())
            .take(max)
            .collect();
        sentences
            .iter()
            .enumerate()
            .map(|(i, s)| VerificationQuestion {
                text: format!("Is it true that {s}?"),
                target_claim: s.to_string(),
                index: i,
            })
            .collect()
    }
}
// ── CoVeConfig ────────────────────────────────────────────────────────────────
/// Configuration for `ChainOfVerificationEngine`.
#[derive(Debug, Clone)]
pub struct CoVeConfig {
    /// Maximum verification questions to generate. Defaults to `5`.
    pub max_questions: usize,
    /// Jaccard support threshold below which a claim is `Contradicted`. Defaults to `0.3`.
    pub support_threshold: f32,
    /// Documents to retrieve per verification question. Defaults to `5`.
    pub top_k: usize,
    /// Whether to revise the answer by removing contradicted claims. Defaults to `true`.
    pub revise: bool,
}
impl Default for CoVeConfig {
    fn default() -> Self {
        Self {
            max_questions: 5,
            support_threshold: 0.3,
            top_k: 5,
            revise: true,
        }
    }
}
impl CoVeConfig {
    /// Set the max questions.
    #[must_use]
    pub fn with_max_questions(mut self, v: usize) -> Self {
        self.max_questions = v;
        self
    }
    /// Set the support threshold.
    #[must_use]
    pub fn with_support_threshold(mut self, v: f32) -> Self {
        self.support_threshold = v;
        self
    }
    /// Set top-k retrieval count.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }
    /// Set whether to revise.
    #[must_use]
    pub fn with_revise(mut self, v: bool) -> Self {
        self.revise = v;
        self
    }
}
// ── CoVeOutput ────────────────────────────────────────────────────────────────
/// Output of Chain-of-Verification.
#[derive(Debug, Clone)]
pub struct CoVeOutput {
    /// The initial draft answer before verification.
    pub baseline_answer: String,
    /// All verification questions that were generated.
    pub questions: Vec<VerificationQuestion>,
    /// Answers and verdicts for each verification question.
    pub verifications: Vec<VerificationAnswer>,
    /// Revised answer with contradicted claims removed or flagged.
    pub revised_answer: String,
    /// Number of claims judged as [`ClaimVerdict::Contradicted`].
    pub contradicted_claims: usize,
    /// Faithfulness improvement estimate (revised coverage − baseline coverage).
    pub faithfulness_delta: f32,
}
// ── ChainOfVerificationError ──────────────────────────────────────────────────
/// Errors from the `chain_of_verification` module.
#[derive(Debug, Error)]
pub enum ChainOfVerificationError {
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
    /// The underlying retrieval step failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
    /// No claims could be extracted from the baseline answer.
    #[error("No verifiable claims found in the answer")]
    NoClaimsFound,
}
