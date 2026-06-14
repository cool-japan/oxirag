//! Extractive summarization and lexical embedding for the summary index.

use std::collections::HashMap;

use crate::types::Document;

// ── Lexical pseudo-embedding ──────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise → hash each token to a bucket with FNV-1a → accumulate
/// per-bucket counts → L2-normalise to the requested `dim`. Identical to the
/// scheme used elsewhere in `OxiRAG`, so embeddings are comparable and fully
/// deterministic.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        // Deterministic hash: FNV-1a over the lowercased token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.to_lowercase().as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

// ── Tokeniser ─────────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens of length >= 2.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── Sentence splitter ─────────────────────────────────────────────────────────

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, and `?`.
#[must_use]
pub fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

// ── Summarizer trait ──────────────────────────────────────────────────────────

/// Produces an extractive summary of a document.
pub trait Summarizer {
    /// Produce an extractive summary of the document.
    fn summarize(&self, doc: &Document, max_sentences: usize) -> String;
}

// ── ExtractiveSummarizer ──────────────────────────────────────────────────────

/// Term-frequency centrality extractive summarizer (no model, deterministic).
///
/// Each sentence is scored by the summed corpus-local term frequency of the
/// tokens it contains (its *salience*); the top `max_sentences` sentences are
/// selected and then re-emitted in their **original document order**, joined by
/// a single space.
#[derive(Debug, Clone, Default)]
pub struct ExtractiveSummarizer;

impl ExtractiveSummarizer {
    /// Create a new [`ExtractiveSummarizer`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Summarizer for ExtractiveSummarizer {
    fn summarize(&self, doc: &Document, max_sentences: usize) -> String {
        let sentences = split_sentences(&doc.content);
        if max_sentences == 0 || sentences.is_empty() {
            return String::new();
        }
        if sentences.len() <= max_sentences {
            // Whole document already fits within the budget.
            return sentences.join(" ");
        }

        // Document-level term frequencies drive salience.
        let mut term_freq: HashMap<String, usize> = HashMap::new();
        for sentence in &sentences {
            for token in tokenize(sentence) {
                *term_freq.entry(token).or_insert(0) += 1;
            }
        }

        // Score each sentence by the summed TF of its tokens, keeping the
        // original position so ties resolve deterministically and ordering can
        // be restored afterwards.
        #[allow(clippy::cast_precision_loss)]
        let mut scored: Vec<(usize, f32)> = sentences
            .iter()
            .enumerate()
            .map(|(idx, sentence)| {
                let salience: usize = tokenize(sentence)
                    .iter()
                    .map(|t| term_freq.get(t).copied().unwrap_or(0))
                    .sum();
                (idx, salience as f32)
            })
            .collect();

        // Select the top `max_sentences` by salience; ties broken by earlier
        // position (smaller index first) to remain deterministic.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(max_sentences);

        // Restore original document order for the chosen sentences.
        scored.sort_by_key(|(idx, _)| *idx);
        scored
            .into_iter()
            .map(|(idx, _)| sentences[idx].clone())
            .collect::<Vec<_>>()
            .join(" ")
    }
}
