//! `ReflexionEngine` implementation.
use crate::reflexion::types::{
    Attempt, AttemptEvaluator, AttemptScore, EpisodicMemory, HeuristicEvaluator,
    HeuristicSelfReflector, Reflection, ReflexionConfig, ReflexionError, ReflexionOutcome,
    SelfReflector,
};
use crate::types::SearchResult;

// ── ReflexionEngine ───────────────────────────────────────────────────────────

/// Reflexion engine — retrieval + verbal self-reflection loop with episodic memory.
///
/// The [`Echo`] layer is provided *per call* through the generic `run<E>` method.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
#[derive(Debug, Clone)]
pub struct ReflexionEngine {
    /// Configuration for this engine.
    pub config: ReflexionConfig,
}

impl ReflexionEngine {
    /// Create a new engine with the given config.
    #[must_use]
    pub fn new(config: ReflexionConfig) -> Self {
        Self { config }
    }

    /// Run the Reflexion loop for `query`.
    ///
    /// # Errors
    ///
    /// Returns [`ReflexionError::EmptyQuery`] if `query` is empty.
    /// Returns [`ReflexionError::RetrievalFailed`] if the echo layer fails.
    /// Returns [`ReflexionError::EvaluationFailed`] if evaluation fails.
    pub async fn run<E>(&self, query: &str, echo: &E) -> Result<ReflexionOutcome, ReflexionError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(ReflexionError::EmptyQuery);
        }

        let evaluator = HeuristicEvaluator;
        let reflector = HeuristicSelfReflector;
        let mut memory = EpisodicMemory::new(self.config.memory_size);
        let mut attempts: Vec<Attempt> = Vec::new();
        let mut best_answer = String::new();
        let mut best_score = AttemptScore::default();
        let mut succeeded = false;

        for attempt_idx in 0..self.config.max_attempts.max(1) {
            let augmented = if memory.reflections.is_empty() {
                query.to_string()
            } else {
                format!("{query} (context: {})", memory.summary())
            };
            let results: Vec<SearchResult> = echo
                .search(&augmented, self.config.top_k, None)
                .await
                .map_err(|e| ReflexionError::RetrievalFailed(e.to_string()))?;

            let answer = build_draft(query, &results);
            let score = evaluator
                .evaluate(query, &answer, &results)
                .map_err(|e| ReflexionError::EvaluationFailed(e.to_string()))?;

            if score.score > best_score.score {
                best_score = score.clone();
                best_answer.clone_from(&answer);
            }

            let reflection = if score.score >= self.config.success_threshold {
                succeeded = true;
                attempts.push(Attempt {
                    index: attempt_idx,
                    answer,
                    score,
                    reflection: None,
                    results,
                });
                break;
            } else {
                let r = reflector
                    .reflect(query, &answer, &score)
                    .map_err(|e| ReflexionError::EvaluationFailed(e.to_string()))?;
                let reflection = Reflection {
                    attempt: attempt_idx,
                    ..r
                };
                memory.add(reflection.clone());
                Some(reflection)
            };
            attempts.push(Attempt {
                index: attempt_idx,
                answer,
                score,
                reflection,
                results,
            });
        }

        if best_answer.is_empty() && !attempts.is_empty() {
            best_answer = attempts[0].answer.clone();
        }

        Ok(ReflexionOutcome {
            best_answer,
            best_score,
            attempts,
            memory,
            succeeded,
        })
    }
}

impl Default for ReflexionEngine {
    fn default() -> Self {
        Self::new(ReflexionConfig::default())
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn build_draft(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No information found for: {query}");
    }
    results
        .iter()
        .take(3)
        .map(|r| r.document.content.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}
