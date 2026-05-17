//! FLARE engine — the core iterative generate-retrieve loop.
//!
//! [`FlareEngine`] orchestrates the full FLARE algorithm described by Jiang
//! et al. (2023):
//!
//! 1. **Draft** — generate a short partial response to estimate confidence.
//! 2. **Evaluate** — score each sentence against the current context window.
//! 3. **Retrieve** — if any sentence is uncertain, use its text (optionally
//!    augmented with the original query) to issue a retrieval call.
//! 4. **Regenerate** — with the enriched context, generate the full answer.
//! 5. **Converge** — repeat until all sentences are confident or the maximum
//!    iteration count is reached.

use tracing::{debug, instrument, warn};

use super::confidence::ConfidenceEstimator;
use super::generator::FlareGenerator;
// ConfidenceEstimator is used via its associated functions (e.g.
// `ConfidenceEstimator::identify_uncertain_sentences`); the PhantomData
// field keeps the import visible to the type system.
use super::retriever::FlareRetriever;
use super::types::{
    ContextDoc, ContextWindow, FlareConfig, FlareError, FlareOutput, IterationRecord,
};

// ── FlareEngine ───────────────────────────────────────────────────────────────

/// The FLARE iterative retrieval-generation engine.
///
/// # Type parameters
///
/// - `G`: generator backend implementing [`FlareGenerator`].
/// - `R`: retriever backend implementing [`FlareRetriever`].
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "flare")]
/// # {
/// # use oxirag::retrieval_loop::{
/// #     engine::FlareEngine,
/// #     generator::MockFlareGenerator,
/// #     retriever::MockFlareRetriever,
/// #     types::{FlareConfig, ContextDoc},
/// # };
/// # #[tokio::main]
/// # async fn main() {
/// let generator = MockFlareGenerator::new_single(
///     "Rust is a systems language focused on safety and performance.".to_string()
/// );
/// let retriever = MockFlareRetriever::new(vec![
///     ContextDoc::new("Rust prioritises memory safety.", 0.9, "doc-1"),
/// ]);
/// let engine = FlareEngine::new(generator, retriever, FlareConfig::default());
/// let output = engine.run_simple("What is Rust?").await.unwrap();
/// assert!(!output.final_answer.is_empty());
/// # }
/// # }
/// ```
pub struct FlareEngine<G: FlareGenerator, R: FlareRetriever> {
    /// FLARE algorithm configuration.
    pub config: FlareConfig,
    /// Text generator.
    generator: G,
    /// Document retriever.
    retriever: R,
    // ConfidenceEstimator is a zero-sized struct; kept for future extensibility
    // and to allow passing a configured estimator in later versions.
    _confidence_estimator: std::marker::PhantomData<ConfidenceEstimator>,
}

impl<G: FlareGenerator, R: FlareRetriever> FlareEngine<G, R> {
    /// Construct a new `FlareEngine`.
    #[must_use]
    pub fn new(generator: G, retriever: R, config: FlareConfig) -> Self {
        Self {
            config,
            generator,
            retriever,
            _confidence_estimator: std::marker::PhantomData,
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Build a retrieval query from an uncertain sentence.
    ///
    /// If `config.query_augment` is enabled, the original query is prepended to
    /// the uncertain span.  The resulting string is trimmed.
    fn build_retrieval_query(&self, original_query: &str, uncertain_span: &str) -> String {
        if self.config.query_augment {
            format!("{} {}", original_query.trim(), uncertain_span.trim())
        } else {
            uncertain_span.trim().to_string()
        }
    }

    /// Parse `initial_context` as a single synthetic [`ContextDoc`] and seed
    /// the [`ContextWindow`] with it.
    fn seed_context_window(&self, initial_context: &str) -> ContextWindow {
        let mut window = ContextWindow::new(self.config.context_window_chars);
        if !initial_context.trim().is_empty() {
            window.add_doc(ContextDoc::new(
                initial_context.trim(),
                1.0,
                "initial-context",
            ));
        }
        window
    }

    /// Evaluate average confidence for `text` against `context`.
    fn avg_text_confidence(text: &str, context: &str) -> f32 {
        let sentences = ConfidenceEstimator::split_into_sentences(text);
        if sentences.is_empty() {
            return 0.0;
        }
        let total: f32 = sentences
            .iter()
            .map(|s| ConfidenceEstimator::estimate_sentence_confidence(s, context).avg_confidence)
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let denom = sentences.len() as f32;
        total / denom
    }

    // ── Core loop ─────────────────────────────────────────────────────────────

    /// Run one FLARE iteration — returns `(record, converged, docs_added)`.
    ///
    /// Extracted to keep `run` under the 100-line function limit.
    async fn run_iteration(
        &self,
        iteration: usize,
        query: &str,
        context_window: &mut ContextWindow,
    ) -> Result<(IterationRecord, bool, usize), FlareError> {
        let context_str = context_window.as_context_string();
        let draft = self
            .generator
            .generate_partial(query, &context_str, 2)
            .await
            .map_err(|e| FlareError::GenerationFailed(e.to_string()))?;
        let uncertain = ConfidenceEstimator::identify_uncertain_sentences(
            &draft,
            &context_str,
            self.config.confidence_threshold,
        );
        let avg_conf = Self::avg_text_confidence(&draft, &context_str);

        // No uncertain sentence → generate full confident answer and converge.
        if uncertain.is_empty() {
            debug!(iteration, "no uncertain sentences — generating full answer");
            let answer = self
                .generator
                .generate(query, &context_str)
                .await
                .map_err(|e| FlareError::GenerationFailed(e.to_string()))?;
            let rec = IterationRecord {
                iteration,
                generated_text: answer,
                triggered_retrieval: false,
                retrieval_query: None,
                docs_retrieved: 0,
                avg_confidence: avg_conf,
            };
            return Ok((rec, true, 0));
        }

        let span = uncertain[0].uncertain_span();
        if span.len() < self.config.min_query_length {
            warn!(iteration, span_len = span.len(), "uncertain span too short");
            let answer = self
                .generator
                .generate(query, &context_str)
                .await
                .map_err(|e| FlareError::GenerationFailed(e.to_string()))?;
            let rec = IterationRecord {
                iteration,
                generated_text: answer,
                triggered_retrieval: false,
                retrieval_query: None,
                docs_retrieved: 0,
                avg_confidence: avg_conf,
            };
            return Ok((rec, false, 0));
        }

        // Retrieve for uncertain span.
        let rq = self.build_retrieval_query(query, &span);
        debug!(iteration, retrieval_query = %rq, "triggering retrieval");
        let docs = self
            .retriever
            .retrieve(&rq, self.config.max_context_docs)
            .await
            .map_err(|e| FlareError::RetrievalFailed(e.to_string()))?;
        let docs_added = docs.len();
        for doc in docs {
            context_window.add_doc(doc);
        }

        let enriched = context_window.as_context_string();
        let answer = self
            .generator
            .generate(query, &enriched)
            .await
            .map_err(|e| FlareError::GenerationFailed(e.to_string()))?;

        let converged = ConfidenceEstimator::identify_uncertain_sentences(
            &answer,
            &enriched,
            self.config.confidence_threshold,
        )
        .is_empty();

        let rec = IterationRecord {
            iteration,
            generated_text: answer,
            triggered_retrieval: true,
            retrieval_query: Some(rq),
            docs_retrieved: docs_added,
            avg_confidence: avg_conf,
        };
        Ok((rec, converged, docs_added))
    }

    /// Run the full FLARE loop, using `initial_context` as the seed context.
    ///
    /// # Errors
    ///
    /// - [`FlareError::GenerationFailed`] if the generator fails.
    /// - [`FlareError::RetrievalFailed`] if the retriever fails.
    #[instrument(skip(self, query, initial_context), fields(max_iter = self.config.max_iterations))]
    pub async fn run(&self, query: &str, initial_context: &str) -> Result<FlareOutput, FlareError> {
        let mut context_window = self.seed_context_window(initial_context);
        let mut iteration_records: Vec<IterationRecord> = Vec::new();
        let mut total_retrievals: usize = 0;
        let mut accumulated_context_docs: usize = usize::from(!initial_context.trim().is_empty());

        for iteration in 0..self.config.max_iterations {
            debug!(iteration, "FLARE loop iteration start");
            let (record, converged, docs_added) = self
                .run_iteration(iteration, query, &mut context_window)
                .await?;
            if record.triggered_retrieval {
                total_retrievals += 1;
            }
            accumulated_context_docs += docs_added;
            iteration_records.push(record);
            if converged {
                break;
            }
        }

        // If no iteration ran (max_iterations=0) generate a fallback answer.
        let final_answer = if let Some(last) = iteration_records.last() {
            last.generated_text.clone()
        } else {
            let ctx = context_window.as_context_string();
            self.generator
                .generate(query, &ctx)
                .await
                .map_err(|e| FlareError::GenerationFailed(e.to_string()))?
        };

        Ok(FlareOutput {
            final_answer,
            iterations: iteration_records,
            total_retrievals,
            context_doc_count: accumulated_context_docs,
        })
    }

    /// Run the FLARE loop with an empty initial context.
    ///
    /// Convenience wrapper around [`run`](FlareEngine::run).
    ///
    /// # Errors
    ///
    /// Propagates any error returned by [`run`](FlareEngine::run).
    pub async fn run_simple(&self, query: &str) -> Result<FlareOutput, FlareError> {
        self.run(query, "").await
    }
}
