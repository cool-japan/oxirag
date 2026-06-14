//! [`SelfAskEngine`] — the Self-Ask compositional self-questioning loop.

use crate::self_ask::types::{
    FollowUp, SelfAskConfig, SelfAskError, SelfAskModel, SelfAskTrace, SubAnswerer,
};

// ── SelfAskEngine ─────────────────────────────────────────────────────────────

/// Drives the Self-Ask loop: the model decides follow-up questions, a
/// [`SubAnswerer`] answers each, and the model composes the final answer.
///
/// The model and sub-answerer are provided *per call* through the generic
/// [`SelfAskEngine::run`] method, mirroring the caller-supplies-executor pattern
/// used across the crate.
#[derive(Debug, Clone, Default)]
pub struct SelfAskEngine {
    /// Configuration for this engine.
    pub config: SelfAskConfig,
}

impl SelfAskEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: SelfAskConfig) -> Self {
        Self { config }
    }

    /// Run the Self-Ask loop for `question`.
    ///
    /// The loop repeatedly asks `model` for the next follow-up question. While a
    /// follow-up is returned and the configured `max_follow_ups` cap has not been
    /// reached, the follow-up is answered by `answerer` and recorded. Once the
    /// model returns `None` (ready to answer) or the cap is hit, `model` composes
    /// the final answer from the accumulated history.
    ///
    /// # Errors
    ///
    /// Returns [`SelfAskError::EmptyQuestion`] if `question` is empty after
    /// trimming.
    pub fn run<M, A>(
        &self,
        question: &str,
        model: &M,
        answerer: &A,
    ) -> Result<SelfAskTrace, SelfAskError>
    where
        M: SelfAskModel + ?Sized,
        A: SubAnswerer + ?Sized,
    {
        if question.trim().is_empty() {
            return Err(SelfAskError::EmptyQuestion);
        }

        let mut history: Vec<FollowUp> = Vec::new();

        while history.len() < self.config.max_follow_ups {
            match model.next_follow_up(question, &history) {
                Some(follow_up) => {
                    let answer = answerer.answer(&follow_up);
                    history.push(FollowUp {
                        question: follow_up,
                        answer,
                    });
                }
                None => break,
            }
        }

        let final_answer = model.compose_final(question, &history);
        let num_hops = history.len();

        Ok(SelfAskTrace {
            follow_ups: history,
            final_answer,
            num_hops,
        })
    }
}
