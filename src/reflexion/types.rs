//! Types for the `reflexion` module.
use thiserror::Error;
// ── AttemptScore ──────────────────────────────────────────────────────────────
/// Quality score produced by [`AttemptEvaluator`].
#[derive(Debug, Clone, Default)]
pub struct AttemptScore {
    /// Overall quality score in [0.0, 1.0].
    pub score: f32,
    /// Fraction of the answer grounded in retrieved documents.
    pub grounding: f32,
    /// Fraction of retrieved content covered in the answer.
    pub coverage: f32,
}
// ── Reflection ────────────────────────────────────────────────────────────────
/// A verbal self-reflection stored in episodic memory.
#[derive(Debug, Clone)]
pub struct Reflection {
    /// Index of the attempt that generated this reflection (0-indexed).
    pub attempt: usize,
    /// Critique of the previous answer.
    pub critique: String,
    /// Lesson learned for the next attempt.
    pub lesson: String,
    /// Score of the attempt that was reflected on.
    pub score: f32,
}
// ── EpisodicMemory ────────────────────────────────────────────────────────────
/// Bounded episodic memory storing past reflections.
#[derive(Debug, Clone, Default)]
pub struct EpisodicMemory {
    /// Stored reflections, oldest first.
    pub reflections: Vec<Reflection>,
    /// Maximum number of reflections to retain.
    pub max_size: usize,
}
impl EpisodicMemory {
    /// Create a new memory with the given capacity.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            reflections: Vec::new(),
            max_size,
        }
    }
    /// Add a reflection, evicting the oldest if at capacity.
    pub fn add(&mut self, r: Reflection) {
        if self.max_size > 0 && self.reflections.len() >= self.max_size {
            self.reflections.remove(0);
        }
        self.reflections.push(r);
    }
    /// Return the most recent `n` reflections.
    #[must_use]
    pub fn recent(&self, n: usize) -> &[Reflection] {
        let start = self.reflections.len().saturating_sub(n);
        &self.reflections[start..]
    }
    /// Produce a combined lessons summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        self.reflections
            .iter()
            .map(|r| r.lesson.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}
// ── AttemptEvaluator ──────────────────────────────────────────────────────────
/// Synchronous evaluator that scores a retrieval attempt.
pub trait AttemptEvaluator {
    /// Score `answer` relative to `query` and the supporting `results`.
    ///
    /// Returns an [`AttemptScore`] in [0.0, 1.0].
    ///
    /// # Errors
    ///
    /// Returns [`ReflexionError::EvaluationFailed`] on internal scorer errors.
    fn evaluate(
        &self,
        query: &str,
        answer: &str,
        results: &[crate::types::SearchResult],
    ) -> Result<AttemptScore, ReflexionError>;
}
// ── SelfReflector ─────────────────────────────────────────────────────────────
/// Synchronous reflector that produces a verbal critique and lesson.
pub trait SelfReflector {
    /// Reflect on `answer` given `query` and `score`.
    ///
    /// Returns a [`Reflection`].
    ///
    /// # Errors
    ///
    /// Returns [`ReflexionError::EvaluationFailed`] on internal errors.
    fn reflect(
        &self,
        query: &str,
        answer: &str,
        score: &AttemptScore,
    ) -> Result<Reflection, ReflexionError>;
}
// ── HeuristicEvaluator ────────────────────────────────────────────────────────
/// Lexical heuristic implementation of [`AttemptEvaluator`].
///
/// Scores an answer using Jaccard-based grounding and coverage:
/// - `grounding` = max Jaccard similarity between answer and any result content
/// - `coverage` = fraction of results with Jaccard overlap > 0.1
/// - `score` = 0.6 × grounding + 0.4 × coverage
#[derive(Debug, Clone, Default)]
pub struct HeuristicEvaluator;

fn jaccard_sim(a: &str, b: &str) -> f32 {
    let ta: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let tb: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let union_size = ta.union(&tb).count();
    if union_size == 0 {
        return 0.0;
    }
    let inter_size = ta.intersection(&tb).count();
    #[allow(clippy::cast_precision_loss)]
    {
        inter_size as f32 / union_size as f32
    }
}

impl AttemptEvaluator for HeuristicEvaluator {
    fn evaluate(
        &self,
        _query: &str,
        answer: &str,
        results: &[crate::types::SearchResult],
    ) -> Result<AttemptScore, ReflexionError> {
        if results.is_empty() {
            return Ok(AttemptScore::default());
        }
        let grounding = results
            .iter()
            .map(|r| jaccard_sim(answer, &r.document.content))
            .fold(0.0_f32, f32::max);
        #[allow(clippy::cast_precision_loss)]
        let coverage = results
            .iter()
            .filter(|r| jaccard_sim(answer, &r.document.content) > 0.1)
            .count() as f32
            / results.len() as f32;
        let score = 0.6 * grounding + 0.4 * coverage;
        Ok(AttemptScore {
            score,
            grounding,
            coverage,
        })
    }
}
// ── HeuristicSelfReflector ────────────────────────────────────────────────────
/// Lexical heuristic implementation of [`SelfReflector`].
#[derive(Debug, Clone, Default)]
pub struct HeuristicSelfReflector;
impl SelfReflector for HeuristicSelfReflector {
    fn reflect(
        &self,
        _query: &str,
        _answer: &str,
        score: &AttemptScore,
    ) -> Result<Reflection, ReflexionError> {
        let lesson = if score.score < 0.3 {
            "Answer lacked key document terms. Try expanding the query with synonyms.".to_string()
        } else if score.score < 0.6 {
            "Answer partially covered the sources. Focus on key entities.".to_string()
        } else {
            "Answer was reasonable. Broaden retrieval scope.".to_string()
        };
        let critique = format!(
            "Attempt scored {:.2}: grounding={:.2}, coverage={:.2}",
            score.score, score.grounding, score.coverage
        );
        Ok(Reflection {
            attempt: 0,
            critique,
            lesson,
            score: score.score,
        })
    }
}
// ── Attempt ───────────────────────────────────────────────────────────────────
/// One iteration of the Reflexion loop.
#[derive(Debug, Clone)]
pub struct Attempt {
    /// Zero-based attempt index.
    pub index: usize,
    /// Answer produced this attempt.
    pub answer: String,
    /// Quality score for this attempt.
    pub score: AttemptScore,
    /// Reflection generated after this attempt (None for the final attempt).
    pub reflection: Option<Reflection>,
    /// Documents retrieved this attempt.
    pub results: Vec<crate::types::SearchResult>,
}
// ── ReflexionConfig ───────────────────────────────────────────────────────────
/// Configuration for `ReflexionEngine`.
#[derive(Debug, Clone)]
pub struct ReflexionConfig {
    /// Maximum number of retrieval+reflection attempts. Defaults to `3`.
    pub max_attempts: usize,
    /// Score threshold above which the engine considers the attempt successful.
    ///
    /// Defaults to `0.7`.
    pub success_threshold: f32,
    /// Number of documents to retrieve per attempt. Defaults to `5`.
    pub top_k: usize,
    /// Maximum reflections kept in episodic memory. Defaults to `8`.
    pub memory_size: usize,
}
impl Default for ReflexionConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            success_threshold: 0.7,
            top_k: 5,
            memory_size: 8,
        }
    }
}
impl ReflexionConfig {
    /// Set the maximum number of attempts.
    #[must_use]
    pub fn with_max_attempts(mut self, v: usize) -> Self {
        self.max_attempts = v;
        self
    }
    /// Set the success threshold.
    #[must_use]
    pub fn with_success_threshold(mut self, v: f32) -> Self {
        self.success_threshold = v;
        self
    }
    /// Set the top-k retrieval count.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }
    /// Set the episodic memory size.
    #[must_use]
    pub fn with_memory_size(mut self, v: usize) -> Self {
        self.memory_size = v;
        self
    }
}
// ── ReflexionOutcome ──────────────────────────────────────────────────────────
/// Final outcome of a `ReflexionEngine` run.
#[derive(Debug, Clone)]
pub struct ReflexionOutcome {
    /// Best answer found across all attempts.
    pub best_answer: String,
    /// Score of the best answer.
    pub best_score: AttemptScore,
    /// All attempts made.
    pub attempts: Vec<Attempt>,
    /// Final state of episodic memory.
    pub memory: EpisodicMemory,
    /// Whether the success threshold was reached.
    pub succeeded: bool,
}
// ── ReflexionError ────────────────────────────────────────────────────────────
/// Errors from the `reflexion` module.
#[derive(Debug, Error)]
pub enum ReflexionError {
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
    /// The underlying retrieval step failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
    /// An evaluator or reflector returned an error.
    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),
}
