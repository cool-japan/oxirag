//! Corpus-wide global memory formation: lexical primitives and gist building.
//!
//! These helpers are deterministic and model-free. The FNV-1a pseudo-embedding,
//! tokenizer, and sentence splitter mirror the conventions used elsewhere in
//! `OxiRAG`, so embeddings produced here are directly comparable to those of
//! sibling modules.

use std::collections::HashMap;

use crate::memorag::types::MemoryGist;
use crate::types::Document;

// ── Lexical pseudo-embedding ──────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenize → hash each token to a bucket with FNV-1a → accumulate
/// per-bucket counts → L2-normalize to the requested `dim`.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in tokenize(text) {
        // Deterministic hash: FNV-1a over the lowercased token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.as_bytes() {
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

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Tokenize `text` into lowercase alphanumeric tokens of length >= 2.
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

// ── Gist construction ─────────────────────────────────────────────────────────

/// Build a corpus-wide [`MemoryGist`] from `corpus`.
///
/// The whole corpus is treated as one stream of evidence:
///
/// 1. Corpus-wide term frequencies are accumulated across every document.
/// 2. Each sentence (tagged with its global position) is scored by the summed
///    corpus frequency of its tokens — its *salience*.
/// 3. The top `gist_sentences` sentences are selected (ties broken by earlier
///    global position) and re-emitted in their original corpus order to form the
///    summary.
/// 4. The top `key_terms` corpus terms by frequency become the memory key terms
///    (ties broken lexicographically for determinism).
///
/// The result is fully deterministic for a given corpus and configuration.
#[must_use]
pub fn build_gist(corpus: &[Document], gist_sentences: usize, key_terms: usize) -> MemoryGist {
    // (1) Corpus-wide term frequencies.
    let mut term_freq: HashMap<String, usize> = HashMap::new();
    // Flatten the corpus into globally-ordered sentences.
    let mut sentences: Vec<String> = Vec::new();
    for doc in corpus {
        for sentence in split_sentences(&doc.content) {
            for token in tokenize(&sentence) {
                *term_freq.entry(token).or_insert(0) += 1;
            }
            sentences.push(sentence);
        }
    }

    let summary = select_summary(&sentences, &term_freq, gist_sentences);
    let key_terms = select_key_terms(&term_freq, key_terms);

    MemoryGist { summary, key_terms }
}

/// Select the top `max_sentences` salient sentences and join them in corpus order.
fn select_summary(
    sentences: &[String],
    term_freq: &HashMap<String, usize>,
    max_sentences: usize,
) -> String {
    if max_sentences == 0 || sentences.is_empty() {
        return String::new();
    }
    if sentences.len() <= max_sentences {
        return sentences.join(" ");
    }

    // Score each sentence by the summed corpus TF of its tokens.
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

    // Top-k by salience; ties broken by earlier global position.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(max_sentences);

    // Restore original corpus order for the chosen sentences.
    scored.sort_by_key(|(idx, _)| *idx);
    scored
        .into_iter()
        .map(|(idx, _)| sentences[idx].clone())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Select the top `max_terms` corpus terms by descending frequency.
fn select_key_terms(term_freq: &HashMap<String, usize>, max_terms: usize) -> Vec<String> {
    if max_terms == 0 {
        return Vec::new();
    }
    let mut terms: Vec<(&String, usize)> = term_freq.iter().map(|(t, c)| (t, *c)).collect();
    // Descending frequency; ties broken lexicographically for determinism.
    terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    terms
        .into_iter()
        .take(max_terms)
        .map(|(t, _)| t.clone())
        .collect()
}
