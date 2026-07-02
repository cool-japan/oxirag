//! Concrete [`DownstreamTask`] and [`UtilityMetric`] implementations.
//!
//! [`MockDownstreamTask`] is a small, fully deterministic stand-in for a real
//! LLM-backed downstream task (e.g. answer generation): this module has no
//! access to an LLM, so it needs a reproducible task to drive per-document and
//! end-to-end eRAG runs in examples and tests. [`RougeLiteUtility`] is the
//! default label-free [`UtilityMetric`]: token-level F1 between the task
//! output and the gold reference answer (a ROUGE-L-lite proxy).

use super::types::{DownstreamTask, UtilityMetric};
use std::collections::HashMap;

// ── Tokenisation helpers ──────────────────────────────────────────────────────

/// Tokenise `text`: lowercase, split on non-alphanumeric boundaries, discard
/// empty tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, and `?`.
///
/// If no sentence-terminating punctuation is present (or `text` is a single
/// clause), the whole trimmed text is returned as one sentence.
fn split_sentences(text: &str) -> Vec<&str> {
    let parts: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed]
        }
    } else {
        parts
    }
}

/// The bag-of-words overlap count between two token multisets, i.e.
/// `sum(min(count_a[t], count_b[t]))` over every distinct token `t`.
fn overlap_count(a: &[String], b: &[String]) -> usize {
    let mut available: HashMap<&str, usize> = HashMap::new();
    for t in a {
        *available.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut overlap = 0usize;
    for t in b {
        if let Some(remaining) = available.get_mut(t.as_str())
            && *remaining > 0
        {
            *remaining -= 1;
            overlap += 1;
        }
    }
    overlap
}

// ── MockDownstreamTask ────────────────────────────────────────────────────────

/// A deterministic, LLM-free stand-in for a downstream generation task.
///
/// For every document in `context`, [`run`](DownstreamTask::run) splits the
/// document into sentences and selects the sentence with the highest
/// query-token overlap (ties are broken by keeping the earliest sentence).
/// The selected sentences — one per context document, in the given order —
/// are joined with a single space to form the task output.
///
/// This makes the output sensitive to *which* document(s) are supplied:
/// running the task on a single document (the per-document eRAG step) yields
/// just that document's best-matching sentence, while running it on the full
/// retrieved set (the end-to-end reference run) yields the concatenation of
/// every document's best-matching sentence — a meaningfully different, and
/// typically more complete, output. A document that contributes no
/// query-overlapping sentence still contributes its single best (highest
/// tied-for-zero) sentence, so every context document is represented.
#[derive(Debug, Clone, Copy, Default)]
pub struct MockDownstreamTask;

impl MockDownstreamTask {
    /// Create a new [`MockDownstreamTask`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// The best-matching sentence of `doc` given `query_tokens`: the sentence
    /// with the highest token overlap, with ties broken toward the earliest
    /// sentence. Returns `None` if `doc` has no sentences.
    fn best_sentence<'a>(doc: &'a str, query_tokens: &[String]) -> Option<&'a str> {
        let mut best: Option<(&'a str, usize)> = None;
        for sentence in split_sentences(doc) {
            let overlap = overlap_count(query_tokens, &tokenize(sentence));
            let is_better = match best {
                None => true,
                Some((_, best_overlap)) => overlap > best_overlap,
            };
            if is_better {
                best = Some((sentence, overlap));
            }
        }
        best.map(|(sentence, _)| sentence)
    }
}

impl DownstreamTask for MockDownstreamTask {
    fn run(&self, query: &str, context: &[String]) -> String {
        let query_tokens = tokenize(query);
        context
            .iter()
            .filter_map(|doc| Self::best_sentence(doc, &query_tokens))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ── RougeLiteUtility ──────────────────────────────────────────────────────────

/// The default label-free [`UtilityMetric`]: token-level F1 between the task
/// output and the gold reference answer.
///
/// This is a deterministic ROUGE-L-lite proxy: unlike true ROUGE-L it uses a
/// bag-of-words intersection rather than the longest common subsequence,
/// trading a little order-sensitivity for `O(n)` simplicity and zero
/// dependencies. `score` tokenises both strings (lowercase, split on
/// non-alphanumeric boundaries), computes the bag-of-words intersection size,
/// and returns the harmonic mean of precision (`overlap / |output_tokens|`)
/// and recall (`overlap / |gold_tokens|`). Returns `0.0` when either token set
/// is empty, or when precision and recall are both zero, rather than dividing
/// by zero.
#[derive(Debug, Clone, Copy, Default)]
pub struct RougeLiteUtility;

impl RougeLiteUtility {
    /// Create a new [`RougeLiteUtility`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl UtilityMetric for RougeLiteUtility {
    #[allow(clippy::cast_precision_loss)]
    fn score(&self, output: &str, gold: &str) -> f32 {
        let output_tokens = tokenize(output);
        let gold_tokens = tokenize(gold);
        if output_tokens.is_empty() || gold_tokens.is_empty() {
            return 0.0;
        }
        let overlap = overlap_count(&output_tokens, &gold_tokens) as f32;
        let precision = overlap / output_tokens.len() as f32;
        let recall = overlap / gold_tokens.len() as f32;
        if precision + recall <= 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
}
