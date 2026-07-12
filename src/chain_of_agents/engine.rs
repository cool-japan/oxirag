//! [`CoaEngine`] — sequential chain-of-agents orchestration: split a
//! document into chunks, run one [`CoaWorker`] per chunk in strict order
//! (threading the communication unit forward), then hand the final unit to
//! a [`CoaManager`].
//!
//! ## The local chunk splitter
//!
//! This crate already has a full-featured chunking pipeline at
//! [`crate::chunking`], but it lives behind the separate `chunking` Cargo
//! feature, which `chain-of-agents` does not enable and must not silently
//! start depending on (per this crate's workspace/feature-flag policy,
//! every optional module owns its own dependency footprint). `split_into_chunks`
//! is therefore a small, local, sentence-aware splitter: it never depends on
//! `crate::chunking`, has no dependencies of its own beyond `std`, and is
//! deliberately simple — good enough to demonstrate and test the sequential
//! chain-of-agents mechanic, not a replacement for the full chunking
//! pipeline. Callers who already have `chunking` enabled and want its
//! richer strategies (recursive, markdown-aware, ...) can chunk with it
//! directly and hand the resulting chunk texts to [`CoaEngine::run_chunks`]
//! instead of [`CoaEngine::run`].

use super::types::{CoaConfig, CoaError, CoaTrace, CoaWorkerStep};
use super::unit::CoaCommunicationUnit;
use super::worker::{CoaManager, CoaWorker};

// ── local text splitting ─────────────────────────────────────────────────────

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, `?`, or
/// newline boundaries. Operates on `char_indices`, so it never slices on a
/// non-UTF-8-boundary regardless of the input's script.
pub(crate) fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0usize;

    for (byte_idx, ch) in text.char_indices() {
        match ch {
            '.' | '!' | '?' => {
                let end = byte_idx + ch.len_utf8();
                let candidate = text[start..end].trim();
                if !candidate.is_empty() {
                    sentences.push(candidate);
                }
                start = end;
            }
            '\n' => {
                let candidate = text[start..byte_idx].trim();
                if !candidate.is_empty() {
                    sentences.push(candidate);
                }
                start = byte_idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

/// Split `document` into chunks of at most `chunk_size` characters, packing
/// whole sentences (via [`split_sentences`]) greedily so no sentence is ever
/// cut mid-word. Each chunk after the first is seeded with up to
/// `chunk_overlap` characters' worth of trailing sentences from the
/// previous chunk, so content right at a boundary remains visible on both
/// sides of the cut.
///
/// A single sentence longer than `chunk_size` on its own becomes its own
/// oversized chunk rather than being truncated or split mid-word — this
/// splitter always produces whole-sentence chunks, at the cost of not
/// strictly bounding every chunk's length in that one edge case.
///
/// Returns an empty `Vec` for an empty (or whitespace-only) `document`.
pub(crate) fn split_into_chunks(
    document: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    let sentences = split_sentences(document);
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut current_len = 0usize;

    for sentence in sentences {
        let sentence_len = sentence.chars().count();
        if !current.is_empty() && current_len + 1 + sentence_len > chunk_size {
            chunks.push(current.join(" "));

            // Seed the next chunk with as many trailing sentences of the
            // chunk just finished as fit within `chunk_overlap`.
            let mut seed: Vec<&str> = Vec::new();
            let mut seed_len = 0usize;
            for s in current.iter().rev() {
                let l = s.chars().count();
                if seed_len + l > chunk_overlap {
                    break;
                }
                seed.push(s);
                seed_len += l + 1;
            }
            seed.reverse();
            current_len = seed.iter().map(|s| s.chars().count() + 1).sum();
            current = seed;
        }
        current.push(sentence);
        current_len += sentence_len + 1;
    }
    if !current.is_empty() {
        chunks.push(current.join(" "));
    }
    chunks
}

// ── CoaEngine ────────────────────────────────────────────────────────────────

/// Orchestrates a complete chain-of-agents run: split, then run
/// [`CoaWorker`]s strictly in sequence (each threading the communication
/// unit forward to the next), then hand the final unit to a [`CoaManager`].
///
/// See the [module-level documentation](crate::chain_of_agents) for the
/// design rationale and a runnable example.
#[derive(Debug, Clone, Default)]
pub struct CoaEngine {
    /// Configuration for this engine.
    pub config: CoaConfig,
}

impl CoaEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: CoaConfig) -> Self {
        Self { config }
    }

    /// Split `document` with the local sentence-packing splitter (see the
    /// [module-level documentation](self)), then run the chain — equivalent
    /// to calling `split_into_chunks` followed by
    /// [`CoaEngine::run_chunks`].
    ///
    /// # Errors
    ///
    /// - [`CoaError::EmptyQuery`] if `query` is blank.
    /// - [`CoaError::EmptyDocument`] if `document` is blank or splits into
    ///   zero chunks.
    /// - [`CoaError::InvalidConfig`] if `self.config` fails
    ///   [`CoaConfig::validate`].
    /// - Whatever `worker`'s [`CoaWorker::process`], or `manager`'s
    ///   [`CoaManager::synthesize`], returns — propagated unchanged.
    pub fn run<W, M>(
        &self,
        query: &str,
        document: &str,
        worker: &W,
        manager: &M,
    ) -> Result<CoaTrace, CoaError>
    where
        W: CoaWorker + ?Sized,
        M: CoaManager + ?Sized,
    {
        self.config.validate()?;
        if document.trim().is_empty() {
            return Err(CoaError::EmptyDocument);
        }
        let chunks = split_into_chunks(document, self.config.chunk_size, self.config.chunk_overlap);
        self.run_chunks(query, &chunks, worker, manager)
    }

    /// Run the chain over pre-split `chunks`, in the order given.
    ///
    /// This is the sequential core of [`CoaEngine`]: a fresh, empty
    /// [`CoaCommunicationUnit`] (per [`CoaCommunicationUnit::new`]) is
    /// handed to `worker` together with `chunks[0]`; the unit it returns is
    /// handed to `worker` together with `chunks[1]`; and so on. Because
    /// this is a strict fold rather than an independent map over `chunks`,
    /// reversing (or otherwise reordering) `chunks` generally changes the
    /// resulting [`CoaTrace`] — unlike a map-reduce style aggregation, where
    /// per-item order is irrelevant. After every worker call the engine
    /// re-applies `worker_index`/`chunk_index` bookkeeping and both budgets
    /// (via [`CoaCommunicationUnit::enforce_budgets`]) to the returned unit,
    /// regardless of what the [`CoaWorker`] implementation did, so the
    /// bounded-size guarantee holds even against a misbehaving custom
    /// implementation.
    ///
    /// When `chunks.len()` exceeds [`CoaConfig::max_workers`], only the
    /// *first* `max_workers` chunks are processed (in order); the rest are
    /// dropped. This is always visible on the returned [`CoaTrace`] via
    /// [`CoaTrace::truncated`] and [`CoaTrace::chunk_count`] versus
    /// [`CoaTrace::workers_run`] — never a silent truncation.
    ///
    /// # Errors
    ///
    /// - [`CoaError::EmptyQuery`] if `query` is blank.
    /// - [`CoaError::EmptyDocument`] if `chunks` is empty.
    /// - [`CoaError::InvalidConfig`] if `self.config` fails
    ///   [`CoaConfig::validate`].
    /// - Whatever `worker`'s [`CoaWorker::process`], or `manager`'s
    ///   [`CoaManager::synthesize`], returns — propagated unchanged.
    pub fn run_chunks<W, M>(
        &self,
        query: &str,
        chunks: &[String],
        worker: &W,
        manager: &M,
    ) -> Result<CoaTrace, CoaError>
    where
        W: CoaWorker + ?Sized,
        M: CoaManager + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(CoaError::EmptyQuery);
        }
        self.config.validate()?;
        if chunks.is_empty() {
            return Err(CoaError::EmptyDocument);
        }

        let workers_run = chunks.len().min(self.config.max_workers);
        let truncated = chunks.len() > self.config.max_workers;

        let mut unit = CoaCommunicationUnit::new(
            self.config.evidence_budget,
            self.config.open_question_budget,
        );
        let mut steps = Vec::with_capacity(workers_run);

        for (worker_index, chunk) in chunks.iter().take(workers_run).enumerate() {
            let incoming = unit.clone();
            let mut outgoing = worker.process(query, chunk, &incoming)?;

            // The engine — not the worker — owns bookkeeping and budget
            // enforcement, mirroring `DebateEngine`'s treatment of
            // `DebateArgument`'s bookkeeping fields.
            outgoing.chunks_seen = incoming.chunks_seen + 1;
            outgoing.evidence_budget = self.config.evidence_budget;
            outgoing.open_question_budget = self.config.open_question_budget;
            outgoing.completeness = outgoing.completeness.clamp(0.0, 1.0);
            outgoing.enforce_budgets();

            steps.push(CoaWorkerStep {
                worker_index,
                chunk_index: worker_index,
                chunk: chunk.clone(),
                incoming_unit: incoming,
                outgoing_unit: outgoing.clone(),
            });
            unit = outgoing;
        }

        let answer = manager.synthesize(query, &unit)?;

        Ok(CoaTrace {
            query: query.to_string(),
            steps,
            final_unit: unit,
            answer,
            chunk_count: chunks.len(),
            workers_run,
            truncated,
        })
    }
}
