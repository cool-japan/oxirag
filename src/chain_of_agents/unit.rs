//! [`CoaCommunicationUnit`] — the evolving, sequentially-threaded state that
//! carries what the chain has learned so far from worker to worker.
//!
//! This is the load-bearing part of the whole `chain_of_agents` design: it
//! is a *real*, structured accumulator (query-relevant evidence snippets,
//! a running partial answer, unresolved open questions, and a completeness
//! signal) rather than a plain string that workers merely append to. A
//! plain-string design would either grow without bound as the chain
//! lengthens (defeating the entire purpose of chunking a long input in the
//! first place) or would need to silently truncate old content with no
//! principled rule for *what* to keep. Instead, every mutation on this type
//! goes through a bounded merge that always keeps the `evidence_budget`
//! (respectively `open_question_budget`) most query-relevant entries and
//! deterministically evicts the rest — see [`CoaCommunicationUnit::merge_evidence`]
//! for the exact eviction rule.

use std::collections::BTreeMap;

// ── CoaCommunicationUnit ─────────────────────────────────────────────────────

/// The evolving communication unit threaded through a chain of
/// [`CoaWorker`](crate::chain_of_agents::CoaWorker)s.
///
/// Worker `i` receives the communication unit worker `i - 1` produced (worker
/// `0` receives [`CoaCommunicationUnit::new`]'s empty value) and returns an
/// updated one; the
/// [`CoaManager`](crate::chain_of_agents::CoaManager) then reads only the
/// *final* unit — never a raw chunk — to produce the answer. This is the
/// mechanism by which evidence discovered in an early chunk can still inform
/// (and be combined with evidence found in) a much later chunk, without ever
/// holding the full document in context at once.
#[derive(Debug, Clone, PartialEq)]
pub struct CoaCommunicationUnit {
    /// Accumulated, query-relevant evidence snippets, each tagged with the
    /// chunk it came from. Always sorted by descending
    /// [`CoaEvidence::relevance_score`](crate::chain_of_agents::CoaEvidence::relevance_score)
    /// (ties broken by descending recency, then by text) and always has at
    /// most `evidence_budget` entries — see [`CoaCommunicationUnit::merge_evidence`].
    pub evidence: Vec<super::types::CoaEvidence>,
    /// A running best-guess answer, refined as evidence accumulates. Default
    /// worker implementations keep this in sync with `evidence`'s current
    /// top entry, but the field is independent so a custom
    /// [`CoaWorker`](crate::chain_of_agents::CoaWorker) may compute it
    /// differently (e.g. genuine free-text synthesis rather than pure
    /// extraction).
    pub partial_answer: String,
    /// Aspects of the query that remain unresolved by the evidence
    /// accumulated so far, in the order they were raised. Bounded to
    /// `open_question_budget` entries — see
    /// [`CoaCommunicationUnit::merge_open_questions`].
    pub open_questions: Vec<String>,
    /// A self-reported completeness / confidence signal in `[0.0, 1.0]`:
    /// how much of the query the accumulated evidence appears to cover.
    /// `0.0` for a fresh unit; `1.0` once every content term of the query
    /// is covered by some piece of evidence (a heuristic, not a guarantee
    /// of correctness).
    pub completeness: f32,
    /// Number of chunks that have been folded into this unit so far. Owned
    /// by the engine, not by workers: [`CoaEngine::run_chunks`](crate::chain_of_agents::CoaEngine::run_chunks)
    /// always overwrites it to `incoming.chunks_seen + 1` after each worker
    /// call, regardless of what the worker returned, so it is always a
    /// trustworthy count of "how many chunks are behind this unit" —
    /// mirroring how [`crate::multi_agent_debate::DebateEngine`] owns
    /// `DebateArgument`'s bookkeeping fields rather than trusting a
    /// `DebatePersona` to get them right.
    pub chunks_seen: usize,
    /// The maximum number of `evidence` entries this unit retains — copied
    /// from [`CoaConfig::evidence_budget`](crate::chain_of_agents::CoaConfig::evidence_budget)
    /// when the unit is created and re-applied by the engine after every
    /// worker call, so a misbehaving custom worker cannot inflate it.
    pub evidence_budget: usize,
    /// The maximum number of `open_questions` entries this unit retains —
    /// see `evidence_budget` above; same provenance and re-application
    /// rule, sourced from
    /// [`CoaConfig::open_question_budget`](crate::chain_of_agents::CoaConfig::open_question_budget).
    pub open_question_budget: usize,
}

impl CoaCommunicationUnit {
    /// Create a new, empty communication unit with the given budgets.
    ///
    /// This is what worker `0` in a chain always receives as its `incoming`
    /// unit.
    #[must_use]
    pub fn new(evidence_budget: usize, open_question_budget: usize) -> Self {
        Self {
            evidence: Vec::new(),
            partial_answer: String::new(),
            open_questions: Vec::new(),
            completeness: 0.0,
            chunks_seen: 0,
            evidence_budget,
            open_question_budget,
        }
    }

    /// Merge newly-found evidence into this unit, then enforce the bounded
    /// eviction policy.
    ///
    /// # Eviction rule
    ///
    /// After combining `self.evidence` with `new_items`:
    ///
    /// 1. **Exact-text duplicates are collapsed**, keeping whichever copy
    ///    has the higher `relevance_score` (ties broken by the more recent
    ///    `source_chunk_index`) — the same fact re-confirmed by a later
    ///    chunk supersedes, rather than duplicates, an earlier sighting.
    /// 2. The result is sorted by **descending `relevance_score`**, ties
    ///    broken by **descending `source_chunk_index`** (more recently
    ///    discovered evidence wins a tie — it is more likely to reflect the
    ///    chain's current understanding), final ties broken by ascending
    ///    text for full determinism.
    /// 3. Only the first `evidence_budget` entries survive; the rest are
    ///    evicted.
    ///
    /// The net effect: the unit always keeps the most query-relevant
    /// evidence found *anywhere* in the chain so far, regardless of how
    /// many chunks have been processed — the size of `self.evidence` is a
    /// function of `evidence_budget` alone, never of chunk count.
    pub fn merge_evidence(&mut self, new_items: Vec<super::types::CoaEvidence>) {
        let mut existing = std::mem::take(&mut self.evidence);
        existing.extend(new_items);

        let mut best: BTreeMap<String, super::types::CoaEvidence> = BTreeMap::new();
        for item in existing {
            best.entry(item.text.clone())
                .and_modify(|current| {
                    let better = match item.relevance_score.total_cmp(&current.relevance_score) {
                        std::cmp::Ordering::Greater => true,
                        std::cmp::Ordering::Equal => {
                            item.source_chunk_index > current.source_chunk_index
                        }
                        std::cmp::Ordering::Less => false,
                    };
                    if better {
                        *current = item.clone();
                    }
                })
                .or_insert(item);
        }

        let mut deduped: Vec<super::types::CoaEvidence> = best.into_values().collect();
        deduped.sort_by(|a, b| {
            b.relevance_score
                .total_cmp(&a.relevance_score)
                .then_with(|| b.source_chunk_index.cmp(&a.source_chunk_index))
                .then_with(|| a.text.cmp(&b.text))
        });
        deduped.truncate(self.evidence_budget);

        self.evidence = deduped;
    }

    /// Merge newly-raised open questions into this unit, then enforce the
    /// bounded eviction policy.
    ///
    /// # Eviction rule
    ///
    /// Questions already present (exact text match) are not duplicated.
    /// When the combined list exceeds `open_question_budget`, the
    /// **oldest** entries (by insertion order, i.e. the front of the list)
    /// are evicted first — a first-in-first-out policy, on the reasoning
    /// that a question raised many chunks ago and never resolved by
    /// [`CoaCommunicationUnit::resolve_open_questions`] is less actionable
    /// than one the chain is *currently* grappling with.
    pub fn merge_open_questions(&mut self, new_questions: Vec<String>) {
        for question in new_questions {
            if !self.open_questions.contains(&question) {
                self.open_questions.push(question);
            }
        }
        if self.open_questions.len() > self.open_question_budget {
            let excess = self.open_questions.len() - self.open_question_budget;
            self.open_questions.drain(0..excess);
        }
    }

    /// Drop every open question for which `is_resolved` returns `true`.
    ///
    /// Called by default worker implementations once newly-merged evidence
    /// covers a previously-open aspect of the query, so `open_questions`
    /// reflects what is *still* unresolved rather than growing forever
    /// alongside `evidence`.
    pub fn resolve_open_questions<F>(&mut self, mut is_resolved: F)
    where
        F: FnMut(&str) -> bool,
    {
        self.open_questions
            .retain(|question| !is_resolved(question));
    }

    /// Re-apply both budgets to this unit's current contents, without
    /// merging anything new.
    ///
    /// Idempotent: calling it twice in a row leaves an already-compliant
    /// unit byte-for-byte unchanged. Used defensively by
    /// [`CoaEngine::run_chunks`](crate::chain_of_agents::CoaEngine::run_chunks)
    /// after every worker call, so the bounded-size guarantee holds even
    /// against a custom [`CoaWorker`](crate::chain_of_agents::CoaWorker)
    /// implementation that forgets to call
    /// [`CoaCommunicationUnit::merge_evidence`] itself and just pushes onto
    /// `evidence` directly.
    pub fn enforce_budgets(&mut self) {
        self.merge_evidence(Vec::new());
        if self.open_questions.len() > self.open_question_budget {
            let excess = self.open_questions.len() - self.open_question_budget;
            self.open_questions.drain(0..excess);
        }
    }
}
