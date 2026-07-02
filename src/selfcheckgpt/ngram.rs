//! [`SelfCheckVariant::NGram`](super::types::SelfCheckVariant::NGram): a
//! purely count-based inconsistency score.
//!
//! ## Algorithm
//!
//! 1. Tokenise every one of the `K` samples and build an n-gram (default:
//!    trigram) **document-frequency** table: for each distinct n-gram, how
//!    many of the `K` samples contain it at least once. Using document
//!    frequency (rather than raw occurrence counts) prevents a single
//!    sample that happens to repeat a phrase from dominating the estimate —
//!    it mirrors "how many independent samples corroborate this phrasing?"
//! 2. For each main-response sentence, estimate a Laplace-smoothed
//!    probability for each of its n-grams under that table:
//!    `p(gram) = (doc_freq(gram) + 1) / (K + 1)`.
//! 3. Average `-ln(p(gram))` over the sentence's n-grams — a
//!    "log-likelihood-lite" score. This is small (near zero) when the
//!    sentence's phrasing recurs across most/all of the `K` samples (well
//!    corroborated) and grows when its n-grams appear in few or none of the
//!    samples (poorly corroborated).
//! 4. Squash the non-negative average into `[0.0, 1.0)` via the monotonic
//!    transform `x / (x + 1)`, which preserves ordering (a higher raw score
//!    still maps to a higher inconsistency score) while keeping the result
//!    comparable to the other two variants' `[0.0, 1.0]` scores.
//!
//! No external language model is used anywhere in this computation; the
//! "model" is nothing more than n-gram document-frequency counting over the
//! `K` samples.

use std::collections::{HashMap, HashSet};

use super::scorer::tokenize;

/// A sentence with zero n-grams (e.g. shorter than `ngram_size` tokens)
/// cannot be scored against the pooled model; it is assigned this neutral
/// "unknown" inconsistency score rather than a fabricated 0 or 1.
const NO_NGRAM_NEUTRAL_SCORE: f32 = 0.5;

/// Builds the ordered list of `n`-length token windows (joined by a single
/// space) for `tokens`, preserving repeats. Returns an empty vector when
/// `n == 0` or `tokens.len() < n`.
fn ngrams(tokens: &[String], n: usize) -> Vec<String> {
    if n == 0 || tokens.len() < n {
        return Vec::new();
    }
    tokens.windows(n).map(|w| w.join(" ")).collect()
}

/// Builds the *distinct set* of `n`-length token windows for `tokens`, used
/// for document-frequency counting (a sample that repeats a phrase should
/// only count once towards that phrase's document frequency).
fn ngram_set(tokens: &[String], n: usize) -> HashSet<String> {
    ngrams(tokens, n).into_iter().collect()
}

/// Computes the pooled n-gram document-frequency inconsistency score for
/// each sentence in `main_sentences`, using `samples` as the pooling corpus.
///
/// Returns one score per entry of `main_sentences`, in the same order.
pub(crate) fn ngram_inconsistency_scores(
    main_sentences: &[String],
    samples: &[String],
    ngram_size: usize,
) -> Vec<f32> {
    if samples.is_empty() {
        return vec![NO_NGRAM_NEUTRAL_SCORE; main_sentences.len()];
    }

    // ── Build the pooled n-gram document-frequency table ────────────────────
    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    for sample in samples {
        let tokens = tokenize(sample);
        for gram in ngram_set(&tokens, ngram_size) {
            *doc_freq.entry(gram).or_insert(0) += 1;
        }
    }

    let num_docs = samples.len();
    #[allow(clippy::cast_precision_loss)]
    let num_docs_f32 = num_docs as f32;

    // ── Score each main-response sentence against the pooled table ─────────
    main_sentences
        .iter()
        .map(|sentence| {
            let tokens = tokenize(sentence);
            let grams = ngrams(&tokens, ngram_size);
            if grams.is_empty() {
                return NO_NGRAM_NEUTRAL_SCORE;
            }

            #[allow(clippy::cast_precision_loss)]
            let sum_neg_log_likelihood: f32 = grams
                .iter()
                .map(|gram| {
                    let df = doc_freq.get(gram).copied().unwrap_or(0) as f32;
                    let probability = (df + 1.0_f32) / (num_docs_f32 + 1.0_f32);
                    -probability.ln()
                })
                .sum();

            #[allow(clippy::cast_precision_loss)]
            let avg_neg_log_likelihood = sum_neg_log_likelihood / grams.len() as f32;

            // Monotonic squash into [0.0, 1.0): 0 -> 0.0, +inf -> 1.0.
            avg_neg_log_likelihood / (avg_neg_log_likelihood + 1.0_f32)
        })
        .collect()
}
