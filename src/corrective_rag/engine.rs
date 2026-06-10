//! CRAG orchestration engine.
//!
//! [`CorrectiveRagEngine`] implements the Corrective RAG loop described in Yan
//! et al. 2024:
//!
//! 1. Retrieve documents from the Echo layer.
//! 2. Grade each document's relevance via a [`RetrievalGrader`].
//! 3. Decide a corrective action based on the grade distribution.
//! 4. Apply knowledge-strip refinement and/or query rewriting.
//! 5. Repeat up to `max_corrections` times.
//! 6. Recompose the final result set with MMR for relevance–diversity balance.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::advanced_retrieval::mmr::{MmrConfig, MmrReranker};
use crate::types::SearchResult;

use super::grader::RetrievalGrader;
use super::strip::{KnowledgeRefiner, QueryRefiner};
use super::types::{
    CorrectiveAction, CorrectiveRagError, CragConfig, CragOutput, GradedDocument, KnowledgeStrip,
    RetrievalGrade,
};

// ── lexical pseudo-embedding ──────────────────────────────────────────────────

/// Tokenise `text` into a stream of string tokens for embedding.
fn tokenize_for_embedding(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() >= 2)
        .collect()
}

/// Build a lexical pseudo-embedding for `text` of length `dim`.
///
/// Each token is hashed to a bucket in `[0, dim)` and the bucket counter is
/// incremented.  The resulting histogram is then L2-normalised so that cosine
/// similarity is well-defined.
fn lexical_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];

    for token in tokenize_for_embedding(text) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        #[allow(clippy::cast_possible_truncation)]
        let idx = (hasher.finish() as usize) % dim;
        v[idx] += 1.0;
    }

    // L2 normalise.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }

    v
}

// ── action decision ───────────────────────────────────────────────────────────

/// Determine the [`CorrectiveAction`] for a set of graded documents.
///
/// Decision rules:
/// - All `Correct` (no `Incorrect`, no `Ambiguous`) → `UseAsIs`
/// - Mix of `Correct` and `Incorrect` → `Augment`
/// - All `Incorrect` → `Rewrite`
/// - No `Correct`, some `Ambiguous` → `Refine`
/// - `Correct` + `Ambiguous`, no `Incorrect` → `UseAsIs`
fn decide_action(grades: &[GradedDocument]) -> CorrectiveAction {
    if grades.is_empty() {
        return CorrectiveAction::Rewrite;
    }

    let correct_count = grades
        .iter()
        .filter(|g| g.grade == RetrievalGrade::Correct)
        .count();
    let incorrect_count = grades
        .iter()
        .filter(|g| g.grade == RetrievalGrade::Incorrect)
        .count();
    let ambiguous_count = grades
        .iter()
        .filter(|g| g.grade == RetrievalGrade::Ambiguous)
        .count();

    if incorrect_count == 0 {
        if correct_count > 0 || ambiguous_count == 0 {
            // All correct, or all ambiguous-no-incorrect but with correct present.
            CorrectiveAction::UseAsIs
        } else {
            // No correct, some ambiguous, no incorrect.
            CorrectiveAction::Refine
        }
    } else if correct_count > 0 {
        // Mix: at least one correct AND at least one incorrect.
        CorrectiveAction::Augment
    } else if ambiguous_count > 0 {
        // No correct, some ambiguous, some incorrect.
        CorrectiveAction::Refine
    } else {
        // All incorrect.
        CorrectiveAction::Rewrite
    }
}

// ── CorrectiveRagEngine ───────────────────────────────────────────────────────

/// The CRAG engine that orchestrates the retrieve → grade → correct → recompose
/// loop.
///
/// `G` is the [`RetrievalGrader`] implementation.  The [`Echo`] layer is
/// provided *per call* through the generic `run<E>` method so that the struct
/// remains `Send + Sync` without holding a reference to an unsized trait object.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
pub struct CorrectiveRagEngine<G: RetrievalGrader> {
    grader: G,
    refiner: KnowledgeRefiner,
    query_refiner: QueryRefiner,
    config: CragConfig,
}

impl<G: RetrievalGrader> CorrectiveRagEngine<G> {
    /// Create a new engine with the given `grader` and `config`.
    #[must_use]
    pub fn new(grader: G, config: CragConfig) -> Self {
        Self {
            grader,
            refiner: KnowledgeRefiner::new(),
            query_refiner: QueryRefiner::new(),
            config,
        }
    }

    /// Execute the CRAG loop for `query` using `echo` as the retrieval backend.
    ///
    /// # Errors
    ///
    /// * [`CorrectiveRagError::EmptyQuery`] — `query` is blank after trimming.
    /// * [`CorrectiveRagError::InvalidThresholds`] — `lower > upper` in config.
    /// * [`CorrectiveRagError::RetrievalFailed`] — the Echo search fails.
    /// * [`CorrectiveRagError::GradingFailed`] — the grader returns an error.
    pub async fn run<E>(&self, query: &str, echo: &E) -> Result<CragOutput, CorrectiveRagError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        // ── 1. Validate input ────────────────────────────────────────────────
        let trimmed_query = query.trim();
        if trimmed_query.is_empty() {
            return Err(CorrectiveRagError::EmptyQuery);
        }
        if self.config.lower_threshold > self.config.upper_threshold {
            return Err(CorrectiveRagError::InvalidThresholds {
                lower: self.config.lower_threshold,
                upper: self.config.upper_threshold,
            });
        }

        // ── 2. Loop state ────────────────────────────────────────────────────
        let mut state = LoopState::new(trimmed_query);

        loop {
            let done = self.loop_iteration(echo, &mut state).await?;
            if done {
                break;
            }
        }

        Ok(CragOutput {
            refined_results: state.final_results,
            grades: state.all_grades,
            actions_taken: state.actions,
            correction_rounds: state.rounds,
            knowledge_strips: state.all_strips,
            final_query: state.current_query,
        })
    }

    /// Execute one iteration of the CRAG loop.  Returns `true` when the loop
    /// should stop.
    async fn loop_iteration<E>(
        &self,
        echo: &E,
        state: &mut LoopState,
    ) -> Result<bool, CorrectiveRagError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        // a. Retrieve
        let raw_results = echo
            .search(&state.current_query, self.config.top_k, None)
            .await
            .map_err(|e| CorrectiveRagError::RetrievalFailed(e.to_string()))?;

        if raw_results.is_empty() {
            if state.rounds < self.config.max_corrections {
                state.rounds += 1;
                state.actions.push(CorrectiveAction::Rewrite);
                state.current_query = format!("{} rephrased", state.current_query);
                return Ok(false);
            }
            return Ok(true); // give up
        }

        // b. Grade
        let grades = self
            .grader
            .grade_all(&state.current_query, &raw_results, &self.config)
            .await?;
        state.all_grades.clone_from(&grades);

        // c. Decide and apply
        let action = decide_action(&grades);
        state.actions.push(action);
        let should_continue = self.apply_action(action, &grades, &raw_results, state);

        // loop_iteration returns true = "stop the loop".
        Ok(!should_continue)
    }

    /// Apply the chosen `action` and update `state`.
    ///
    /// Returns `true` when the loop should continue (Rewrite within budget),
    /// `false` when a terminal action was taken and the loop should stop.
    fn apply_action(
        &self,
        action: CorrectiveAction,
        grades: &[GradedDocument],
        raw_results: &[SearchResult],
        state: &mut LoopState,
    ) -> bool {
        match action {
            CorrectiveAction::Rewrite => {
                if state.rounds >= self.config.max_corrections {
                    // Budget exhausted — use what we have and stop.
                    state.final_results = self.apply_mmr(raw_results, &state.current_query);
                    return false;
                }
                state.rounds += 1;
                let rewritten = self.build_rewritten_query(raw_results, state);
                state.current_query = rewritten;
                true // continue the loop
            }

            CorrectiveAction::UseAsIs | CorrectiveAction::Refine | CorrectiveAction::Discard => {
                let candidates = self.candidates_with_strips(raw_results, state);
                state.final_results = self.apply_mmr(&candidates, &state.current_query);
                false // terminal — stop the loop
            }

            CorrectiveAction::Augment => {
                let candidates = self.augment_candidates(grades, raw_results, state);
                state.final_results = self.apply_mmr(&candidates, &state.current_query);
                false // terminal — stop the loop
            }
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a rewritten query using knowledge strips from the top document.
    fn build_rewritten_query(&self, raw_results: &[SearchResult], state: &mut LoopState) -> String {
        let strips_for_rewrite = if self.config.enable_knowledge_strip {
            if let Some(first) = raw_results.first() {
                self.refiner
                    .decompose(&state.current_query, first, &self.config)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let kept: Vec<KnowledgeStrip> = strips_for_rewrite
            .iter()
            .filter(|s| s.kept)
            .cloned()
            .collect();
        let rewritten = self.query_refiner.rewrite(&state.current_query, &kept);

        let final_query = if rewritten == state.current_query {
            format!("{} alternative", state.current_query)
        } else {
            rewritten
        };

        state.all_strips.extend(strips_for_rewrite);
        final_query
    }

    /// Return refined candidates (with knowledge strips if enabled), falling
    /// back to the raw results when strips produce nothing.
    fn candidates_with_strips(
        &self,
        raw_results: &[SearchResult],
        state: &mut LoopState,
    ) -> Vec<SearchResult> {
        if self.config.enable_knowledge_strip {
            let (kept_docs, new_strips) =
                self.run_knowledge_strip(&state.current_query, raw_results, &state.all_strips);
            state.all_strips.extend(new_strips);
            if kept_docs.is_empty() {
                raw_results.to_vec()
            } else {
                kept_docs
            }
        } else {
            raw_results.to_vec()
        }
    }

    /// Build the candidate set for the `Augment` action.
    fn augment_candidates(
        &self,
        grades: &[GradedDocument],
        raw_results: &[SearchResult],
        state: &mut LoopState,
    ) -> Vec<SearchResult> {
        let correct_docs: Vec<SearchResult> = grades
            .iter()
            .filter(|g| g.grade == RetrievalGrade::Correct)
            .map(|g| g.result.clone())
            .collect();

        let ambiguous_docs: Vec<SearchResult> = grades
            .iter()
            .filter(|g| g.grade == RetrievalGrade::Ambiguous)
            .map(|g| g.result.clone())
            .collect();

        let mut candidates = correct_docs;

        if self.config.enable_knowledge_strip && !ambiguous_docs.is_empty() {
            let (refined, new_strips) =
                self.run_knowledge_strip(&state.current_query, &ambiguous_docs, &state.all_strips);
            state.all_strips.extend(new_strips);
            candidates.extend(refined);
        } else {
            candidates.extend(ambiguous_docs);
        }

        if candidates.is_empty() {
            candidates = raw_results.to_vec();
        }
        candidates
    }

    /// Run knowledge-strip decomposition + recomposition on `docs`.
    ///
    /// Returns `(recomposed_results, new_strips)`.  When all strips are
    /// filtered out, falls back to the top-1 document unfiltered.
    fn run_knowledge_strip(
        &self,
        query: &str,
        docs: &[SearchResult],
        _existing_strips: &[KnowledgeStrip],
    ) -> (Vec<SearchResult>, Vec<KnowledgeStrip>) {
        let all_strips: Vec<KnowledgeStrip> = docs
            .iter()
            .flat_map(|doc| self.refiner.decompose(query, doc, &self.config))
            .collect();

        let recomposed = self.refiner.recompose(&all_strips);

        let final_docs = if recomposed.is_empty() {
            docs.first().map_or_else(Vec::new, |top| vec![top.clone()])
        } else {
            recomposed
        };

        (final_docs, all_strips)
    }

    /// Rerank `candidates` with MMR using lexical pseudo-embeddings.
    fn apply_mmr(&self, candidates: &[SearchResult], query: &str) -> Vec<SearchResult> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let dim = self.config.recomposition_dim;
        let q_emb = lexical_embedding(query, dim);

        let doc_embeddings: HashMap<String, Vec<f32>> = candidates
            .iter()
            .map(|r| {
                let emb = lexical_embedding(&r.document.content, dim);
                (r.document.id.as_str().to_string(), emb)
            })
            .collect();

        let reranker = MmrReranker::new(MmrConfig {
            lambda: self.config.mmr_lambda,
            top_k: self.config.top_k,
        });

        reranker.rerank(candidates, &q_emb, &doc_embeddings)
    }
}

// ── LoopState ─────────────────────────────────────────────────────────────────

/// Internal mutable state threaded through the CRAG loop iterations.
struct LoopState {
    current_query: String,
    rounds: usize,
    all_grades: Vec<GradedDocument>,
    all_strips: Vec<KnowledgeStrip>,
    actions: Vec<CorrectiveAction>,
    final_results: Vec<SearchResult>,
}

impl LoopState {
    fn new(initial_query: &str) -> Self {
        Self {
            current_query: initial_query.to_string(),
            rounds: 0,
            all_grades: Vec::new(),
            all_strips: Vec::new(),
            actions: Vec::new(),
            final_results: Vec::new(),
        }
    }
}
