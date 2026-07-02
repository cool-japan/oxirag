//! [`RatEngine`] — the Retrieval-Augmented Thoughts (RAT) draft → per-step
//! revision loop.

use crate::retrieval_augmented_thoughts::types::{
    RatConfig, RatError, RatGenerator, RatRetriever, RatThought, RatTrace,
};

// ── retrieval query construction ─────────────────────────────────────────────

/// Build the retrieval query for thought step `draft`, conditioned on the
/// original `question` and the thoughts already revised so far (`prior`).
///
/// Per the RAT paper, retrieval for step `i` is targeted using the
/// cumulative reasoning so far rather than the bare question alone: this
/// keeps later steps' retrieval grounded in what earlier steps concluded.
fn build_retrieval_query(question: &str, prior: &[String], draft: &str) -> String {
    if prior.is_empty() {
        format!("{question} {draft}")
    } else {
        format!("{question} {} {draft}", prior.join(" "))
    }
}

// ── RatEngine ─────────────────────────────────────────────────────────────────

/// Drives the RAT loop: draft a chain-of-thought once, then revise each
/// thought step left-to-right using retrieval targeted at that step.
///
/// Unlike `iterative_rag::IterativeRagEngine` (which regenerates the whole
/// answer every round from an expanded query) RAT never re-drafts: each
/// step's *own* text is corrected exactly once, using evidence retrieved
/// for a query built from the question, the already-revised prior steps,
/// and that step's draft. This also differs from `self_ask::SelfAskEngine`,
/// which asks explicit follow-up sub-questions rather than revising a
/// pre-drafted reasoning chain.
///
/// The generator and retriever are owned by the engine (supplied at
/// construction), rather than passed per-call — this lets a single engine
/// value be reused across many [`RatEngine::run`] calls without threading
/// the backends through every call site.
pub struct RatEngine<G, R>
where
    G: RatGenerator,
    R: RatRetriever,
{
    /// Configuration for this engine.
    pub config: RatConfig,
    generator: G,
    retriever: R,
}

impl<G, R> RatEngine<G, R>
where
    G: RatGenerator,
    R: RatRetriever,
{
    /// Create a new [`RatEngine`] from the given configuration, generator,
    /// and retriever.
    ///
    /// Construction never fails; an invalid `config` (e.g. `max_thoughts ==
    /// 0`) is instead reported by [`RatEngine::run`], mirroring how
    /// `self_ask::SelfAskEngine::new` defers all validation to `run`.
    #[must_use]
    pub fn new(config: RatConfig, generator: G, retriever: R) -> Self {
        Self {
            config,
            generator,
            retriever,
        }
    }

    /// Run the RAT loop for `question`.
    ///
    /// 1. Validates `question` and `self.config`.
    /// 2. Drafts an initial chain-of-thought via
    ///    [`RatGenerator::draft_cot`], truncated to `config.max_thoughts`
    ///    steps.
    /// 3. Walks the draft left-to-right. For each step whose trimmed draft
    ///    is at least `config.min_thought_len` bytes long: builds a
    ///    retrieval query from the question, the already-revised prior
    ///    steps, and the current draft; retrieves up to `config.top_k`
    ///    passages via [`RatRetriever::retrieve`]; and revises the draft via
    ///    [`RatGenerator::revise`]. Steps shorter than `min_thought_len`
    ///    skip retrieval and revision entirely (the draft is kept as-is,
    ///    with an empty retrieval query and no passages).
    /// 4. Synthesises the final answer via [`RatGenerator::finalize`] from
    ///    the fully revised chain.
    ///
    /// # Errors
    ///
    /// Returns [`RatError::EmptyQuestion`] when `question` is blank after
    /// trimming, [`RatError::InvalidConfig`] when `self.config` fails
    /// [`RatConfig::validate`], or [`RatError::NoThoughts`] when
    /// [`RatGenerator::draft_cot`] drafts zero steps.
    pub fn run(&self, question: &str) -> Result<RatTrace, RatError> {
        let question = question.trim();
        if question.is_empty() {
            return Err(RatError::EmptyQuestion);
        }

        self.config.validate()?;

        let mut drafts = self.generator.draft_cot(question);
        if drafts.len() > self.config.max_thoughts {
            drafts.truncate(self.config.max_thoughts);
        }

        if drafts.is_empty() {
            return Err(RatError::NoThoughts);
        }

        let mut revised_so_far: Vec<String> = Vec::with_capacity(drafts.len());
        let mut thoughts: Vec<RatThought> = Vec::with_capacity(drafts.len());

        for (index, draft) in drafts.into_iter().enumerate() {
            let too_short = draft.trim().len() < self.config.min_thought_len;

            if too_short {
                let revised = draft.clone();
                revised_so_far.push(revised.clone());
                thoughts.push(RatThought {
                    index,
                    draft,
                    revised,
                    retrieval_query: String::new(),
                    passages: Vec::new(),
                });
                continue;
            }

            let retrieval_query = build_retrieval_query(question, &revised_so_far, &draft);
            let passages = self.retriever.retrieve(&retrieval_query, self.config.top_k);
            let revised = self
                .generator
                .revise(question, &revised_so_far, &draft, &passages);

            revised_so_far.push(revised.clone());
            thoughts.push(RatThought {
                index,
                draft,
                revised,
                retrieval_query,
                passages,
            });
        }

        let final_answer = self.generator.finalize(question, &revised_so_far);

        Ok(RatTrace {
            question: question.to_string(),
            thoughts,
            final_answer,
        })
    }
}
