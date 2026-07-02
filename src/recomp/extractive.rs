//! Extractive sentence compressor and the shared lexical-scoring machinery
//! reused by [`super::abstractive`].
//!
//! # Algorithm
//!
//! 1. Split every passage into sentences and flatten them into a single list,
//!    remembering each sentence's global position (`order`) so original
//!    ordering can be restored later.
//! 2. Score each sentence against the query using a TF-IDF-lite lexical
//!    overlap: for every query token present in the sentence, add an inverse
//!    document-frequency weight computed over the sentence corpus (rarer
//!    shared tokens contribute more).
//! 3. Drop near-identical sentences (Jaccard similarity at or above a
//!    threshold), keeping the earliest occurrence.
//! 4. Greedily select the highest-scoring sentences that still fit within the
//!    token budget — scanning past sentences that don't fit so a later,
//!    shorter sentence can still be packed in (best-fit greedy, not "stop at
//!    first overflow").
//! 5. Emit the selected sentences in their **original** relative order.

use std::collections::{HashMap, HashSet};

/// Default near-duplicate Jaccard threshold used when
/// [`ExtractiveSummaryCompressor::compress`] is called standalone (outside a
/// [`super::pipeline::RecompPipeline`], which instead threads through the
/// configured [`super::types::RecompConfig::dedup_similarity_threshold`]).
pub(crate) const DEFAULT_DEDUP_SIMILARITY_THRESHOLD: f32 = 0.85;

// ── tokenisation helpers ──────────────────────────────────────────────────────

/// Tokenise `text`: split on non-alphanumeric characters, lowercase, and drop
/// single-character fragments. Mirrors the tokenizer idiom used throughout
/// `OxiRAG` (see `query_difficulty::predictor::tokenise` and
/// `context_pruning::pruner::query_tokens`).
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Tokenise `text` into a lowercase token set.
pub(crate) fn token_set(text: &str) -> HashSet<String> {
    tokenize(text).into_iter().collect()
}

/// Approximate token (word) count via whitespace splitting.
pub(crate) fn approx_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Jaccard similarity between two token sets, in `[0.0, 1.0]`.
pub(crate) fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let ratio = intersection as f32 / union as f32;
        ratio
    }
}

// ── sentence splitting ────────────────────────────────────────────────────────

/// Split `text` into sentences on `.`/`?`/`!` followed by whitespace, or on
/// blank lines, discarding empty fragments.
pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        current.push(chars[i]);
        if i + 1 < n {
            let end_punct = chars[i] == '.' || chars[i] == '?' || chars[i] == '!';
            if end_punct && chars[i + 1].is_whitespace() {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current = String::new();
                i += 2;
                continue;
            }
            if chars[i] == '\n' && chars[i + 1] == '\n' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current = String::new();
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

// ── shared scoring ─────────────────────────────────────────────────────────────

/// A sentence extracted from the passage set, carrying its relevance `score`
/// and its `order` (position across all passages, first passage first) so
/// callers can restore the original ordering after ranking by relevance.
#[derive(Debug, Clone)]
pub(crate) struct ScoredSentence {
    /// The sentence's original surface text.
    pub text: String,
    /// The sentence's position among all sentences flattened across `passages`,
    /// in original document order.
    pub order: usize,
    /// TF-IDF-lite lexical-overlap relevance score against the query.
    pub score: f32,
}

/// Flatten `passages` into sentences and score each one's relevance to
/// `query` using a TF-IDF-lite lexical overlap: for every query token also
/// present in the sentence, add an inverse-document-frequency weight computed
/// over the corpus of sentences.
///
/// Returns the sentences in their original document order (`order` is
/// `0..len()`, strictly increasing).
pub(crate) fn score_sentences(query: &str, passages: &[String]) -> Vec<ScoredSentence> {
    let mut raw_sentences: Vec<String> = Vec::new();
    for passage in passages {
        for sentence in split_sentences(passage) {
            if approx_token_count(&sentence) == 0 {
                continue;
            }
            raw_sentences.push(sentence);
        }
    }

    let sentence_count = raw_sentences.len();
    if sentence_count == 0 {
        return Vec::new();
    }

    let sentence_token_sets: Vec<HashSet<String>> =
        raw_sentences.iter().map(|s| token_set(s)).collect();

    // Document frequency of each token across the sentence corpus, used as
    // the "IDF" half of the TF-IDF-lite score.
    let mut document_frequency: HashMap<String, usize> = HashMap::new();
    for tokens_in_sentence in &sentence_token_sets {
        for token in tokens_in_sentence {
            *document_frequency.entry(token.clone()).or_insert(0) += 1;
        }
    }

    let query_tokens = token_set(query);

    raw_sentences
        .into_iter()
        .zip(sentence_token_sets.iter())
        .enumerate()
        .map(|(order, (text, sentence_tokens))| {
            let mut score = 0.0_f32;
            for query_token in &query_tokens {
                if sentence_tokens.contains(query_token) {
                    let count = document_frequency.get(query_token).copied().unwrap_or(1);
                    #[allow(clippy::cast_precision_loss)]
                    let inverse_document_frequency =
                        ((sentence_count as f32 + 1.0) / (count as f32 + 1.0)).ln() + 1.0;
                    score += inverse_document_frequency;
                }
            }
            ScoredSentence { text, order, score }
        })
        .collect()
}

/// Remove near-identical sentences from `sentences`, keeping the earliest
/// occurrence of every cluster whose pairwise Jaccard similarity is at or
/// above `threshold`. Relative order is preserved.
pub(crate) fn dedup_scored_sentences(
    sentences: Vec<ScoredSentence>,
    threshold: f32,
) -> Vec<ScoredSentence> {
    let mut kept: Vec<ScoredSentence> = Vec::new();
    let mut kept_tokens: Vec<HashSet<String>> = Vec::new();

    for sentence in sentences {
        let tokens = token_set(&sentence.text);
        let is_duplicate = kept_tokens.iter().any(|k| jaccard(k, &tokens) >= threshold);
        if !is_duplicate {
            kept_tokens.push(tokens);
            kept.push(sentence);
        }
    }
    kept
}

/// Rank `sentences` by descending score, breaking ties by ascending original
/// `order` so the ranking is fully deterministic, then greedily accept
/// sentences into the token budget — scanning past ones that don't fit so a
/// later, shorter sentence can still be packed in.
///
/// Returns the set of `order` indices selected.
fn select_within_budget(sentences: &[ScoredSentence], token_budget: usize) -> HashSet<usize> {
    let mut ranked: Vec<&ScoredSentence> = sentences.iter().collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.order.cmp(&b.order))
    });

    let mut selected: HashSet<usize> = HashSet::new();
    let mut token_sum = 0usize;
    for sentence in ranked {
        let sentence_tokens = approx_token_count(&sentence.text);
        if token_sum + sentence_tokens > token_budget {
            continue;
        }
        token_sum += sentence_tokens;
        selected.insert(sentence.order);
        if token_sum >= token_budget {
            break;
        }
    }
    selected
}

// ── ExtractiveSummaryCompressor ────────────────────────────────────────────────

/// Sentence-extractive compressor: scores every sentence in the passage set
/// against the query, deduplicates near-identical sentences, and packs the
/// highest-scoring ones within a token budget — while preserving their
/// **original relative order** across passages, so the compressed output
/// reads coherently rather than as a relevance-ranked jumble.
///
/// Unlike `context_compression::ExtractiveCompressor` (which emits sentences
/// in *score* order and operates over `SearchResult` sources), this
/// compressor restores original document order and operates on raw passage
/// strings, matching RECOMP's formulation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractiveSummaryCompressor;

impl ExtractiveSummaryCompressor {
    /// Creates a new [`ExtractiveSummaryCompressor`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compresses `passages` relevant to `query` into at most `token_budget`
    /// words, selecting whole sentences only (never truncating mid-sentence).
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
        let selected_orders = select_within_budget(&deduped, token_budget);

        deduped
            .iter()
            .filter(|s| selected_orders.contains(&s.order))
            .map(|s| s.text.as_str())
            .collect::<Vec<&str>>()
            .join(" ")
    }
}
