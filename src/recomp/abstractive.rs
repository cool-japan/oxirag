//! Template-based "abstractive-lite" compressor.
//!
//! # Honesty note
//!
//! This is **not** neural abstractive summarization. `OxiRAG` is a pure-Rust,
//! model-free heuristic crate, so "abstractive" here means: select the same
//! TF-IDF-lite top-scoring sentences used by
//! [`super::extractive::ExtractiveSummaryCompressor`], then lightly rewrite
//! and stitch them together with deterministic connective-phrase templates
//! (`"Additionally, "`, `"Furthermore, "`, …) into a single synthetic
//! paragraph, ranked by relevance rather than by original document order.
//! There is no paraphrasing, entailment, or generation model involved — the
//! output vocabulary is a strict subset of the input vocabulary plus a fixed
//! set of connective words.

use super::extractive::{
    ScoredSentence, approx_token_count, dedup_scored_sentences, score_sentences,
};

/// Default near-duplicate Jaccard threshold used when
/// [`AbstractiveSummaryCompressor::compress`] is called standalone (outside a
/// [`super::pipeline::RecompPipeline`]).
const DEFAULT_DEDUP_SIMILARITY_THRESHOLD: f32 = 0.85;

/// Connective phrases used to lightly fuse extracted sentences into a single
/// template-based paragraph. The first sentence gets no connective; later
/// ones cycle through this list before falling back to `"Also, "`.
const FUSION_CONNECTIVES: &[&str] = &[
    "",
    "Additionally, ",
    "Furthermore, ",
    "Moreover, ",
    "In addition, ",
];

/// Fallback connective used once [`FUSION_CONNECTIVES`] is exhausted.
const FALLBACK_CONNECTIVE: &str = "Also, ";

/// Lightly rewrite `sentence` as a fused clause introduced by `connective`:
/// the sentence's trailing terminal punctuation is normalized to a single
/// period, and — when a non-empty connective is prepended — the sentence's
/// first letter is lowercased so it reads as a subordinate clause rather than
/// a new standalone sentence.
fn fuse_clause(connective: &str, sentence: &str) -> String {
    let trimmed = sentence.trim().trim_end_matches(['.', '!', '?']);
    if connective.is_empty() {
        format!("{trimmed}.")
    } else {
        let mut chars = trimmed.chars();
        let lowered_body = match chars.next() {
            Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
            None => String::new(),
        };
        format!("{connective}{lowered_body}.")
    }
}

/// Truncate `sentence` to at most `token_budget` whitespace-delimited words,
/// used only as a last resort when even the single highest-scoring sentence
/// exceeds the entire budget.
fn truncate_to_budget(sentence: &str, token_budget: usize) -> String {
    let words: Vec<&str> = sentence.split_whitespace().take(token_budget).collect();
    if words.is_empty() {
        String::new()
    } else {
        format!("{}.", words.join(" ").trim_end_matches(['.', '!', '?']))
    }
}

// ── AbstractiveSummaryCompressor ───────────────────────────────────────────────

/// Template-based "abstractive-lite" compressor.
///
/// Selects the top-scoring sentences (same TF-IDF-lite lexical-overlap
/// scoring as [`super::extractive::ExtractiveSummaryCompressor`]), most
/// relevant first, and fuses them into a single paragraph using connective
/// phrases — a deterministic extractive-fusion template appropriate for a
/// pure-Rust heuristic codebase, not neural abstraction. See the module-level
/// docs for the full honesty note.
#[derive(Debug, Clone, Copy, Default)]
pub struct AbstractiveSummaryCompressor;

impl AbstractiveSummaryCompressor {
    /// Creates a new [`AbstractiveSummaryCompressor`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compresses `passages` relevant to `query` into a single synthetic
    /// paragraph capped at `token_budget` words.
    ///
    /// Uses the default near-duplicate threshold
    /// (`DEFAULT_DEDUP_SIMILARITY_THRESHOLD = 0.85`); to control
    /// deduplication strength use [`super::pipeline::RecompPipeline`] with a
    /// custom [`super::types::RecompConfig::dedup_similarity_threshold`]
    /// instead.
    ///
    /// Returns an empty string when `passages` is empty, `token_budget` is
    /// `0`, or no sentence can be extracted from `passages`.
    #[must_use]
    pub fn compress(&self, query: &str, passages: &[String], token_budget: usize) -> String {
        self.compress_with_dedup_threshold(
            query,
            passages,
            token_budget,
            DEFAULT_DEDUP_SIMILARITY_THRESHOLD,
        )
    }

    /// As [`Self::compress`], but with an explicit dedup Jaccard threshold.
    /// Used internally by [`super::pipeline::RecompPipeline`] to thread
    /// through the configured
    /// [`super::types::RecompConfig::dedup_similarity_threshold`].
    ///
    /// Takes `&self` (rather than being an associated function) to match the
    /// instance-method shape of [`Self::compress`] and to stay extensible if
    /// this compressor later grows configurable fields.
    #[must_use]
    #[allow(clippy::unused_self, clippy::trivially_copy_pass_by_ref)]
    pub(crate) fn compress_with_dedup_threshold(
        &self,
        query: &str,
        passages: &[String],
        token_budget: usize,
        dedup_threshold: f32,
    ) -> String {
        if passages.is_empty() || token_budget == 0 {
            return String::new();
        }

        let scored = score_sentences(query, passages);
        if scored.is_empty() {
            return String::new();
        }

        let deduped = dedup_scored_sentences(scored, dedup_threshold);
        let mut ranked: Vec<ScoredSentence> = deduped;
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.order.cmp(&b.order))
        });

        let mut fused_parts: Vec<String> = Vec::new();
        let mut token_sum = 0usize;

        for (rank, sentence) in ranked.iter().enumerate() {
            let connective = FUSION_CONNECTIVES
                .get(rank)
                .copied()
                .unwrap_or(FALLBACK_CONNECTIVE);
            let clause = fuse_clause(connective, &sentence.text);
            let clause_tokens = approx_token_count(&clause);

            if token_sum + clause_tokens > token_budget {
                if fused_parts.is_empty() {
                    // Even the single highest-scoring sentence overflows the
                    // budget: truncate it so the summary still honors the
                    // token cap, rather than returning nothing.
                    let truncated = truncate_to_budget(&sentence.text, token_budget);
                    if !truncated.is_empty() {
                        fused_parts.push(truncated);
                    }
                }
                break;
            }

            token_sum += clause_tokens;
            fused_parts.push(clause);
        }

        fused_parts.join(" ").trim().to_string()
    }
}
