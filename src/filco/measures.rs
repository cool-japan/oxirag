//! Scoring functions for FILCO's three sentence-level filtering measures, plus
//! the shared tokenisation and sentence-splitting helpers they build on.
//!
//! # STRINC (String Inclusion)
//!
//! Wang et al. (2023) define STRINC, at *training* time, as a simple
//! substring test: a sentence is a positive filtering example if it contains
//! the gold answer string. [`strinc_strict_score`] implements exactly that
//! paper-faithful check and is used by
//! [`super::filter::FilcoFilter::filter_with_answer`] when the caller already
//! knows the expected answer (e.g. building training data, or re-scoring a
//! generated answer against its supporting context).
//!
//! At *inference* time there is no gold answer to test against, so
//! [`strinc_heuristic_score`] approximates the same intuition — "does this
//! sentence contain a plausible answer span?" — with two deterministic,
//! model-free signals, and keeps a sentence if *either* fires:
//!
//! 1. **Shared n-gram**: the sentence and the query share a contiguous run of
//!    at least [`super::types::FilcoConfig::strinc_min_ngram`] tokens (e.g. a
//!    multi-word phrase from the question reappears verbatim in the
//!    sentence — a strong signal that the sentence is *about* the thing the
//!    query asks about, and thus a likely carrier of its answer).
//! 2. **Answer-like entity overlap**: the sentence contains a capitalized
//!    proper-noun token or a numeric token that is also implied by the query
//!    (i.e. that same token, case-insensitively, also appears in the query).
//!    Proper nouns and numbers are the most common surface form of factoid
//!    answers, so a sentence introducing one that the query already
//!    references is treated as answer-bearing.
//!
//! This is an explicit **heuristic stand-in**, not the paper's STRINC (which
//! is only well-defined when the answer is known); it is documented here so
//! callers can judge whether it is appropriate for their use case.
//!
//! # Lexical overlap
//!
//! [`lexical_overlap_score`] is the Jaccard similarity between the sentence's
//! token set and the query's token set — a direct, symmetric, corpus-free
//! measure of surface overlap.
//!
//! # CXMI-lite
//!
//! The paper's CXMI measures how much a sentence changes a language model's
//! estimated likelihood of producing the correct answer, conditioned on the
//! query: `CXMI(sentence) = log P(answer | query, sentence) - log P(answer | query)`.
//! Computing that requires a language model. [`cxmi_lite_score`] instead
//! substitutes a **deterministic pseudo-likelihood proxy** —
//! `relevance(query, context)`, an IDF-weighted recall of the query's tokens
//! within `context` computed over the passage set's own sentence corpus (the
//! same TF-IDF-lite idiom used by `recomp::extractive::score_sentences`) —
//! and reports `relevance(query, [sentence]) - relevance(query, [])`. Because
//! an empty context shares no tokens with the query, `relevance(query, [])`
//! is always `0.0` under this proxy, so the score reduces in practice to a
//! single-sentence relevance estimate; the literal difference form is kept so
//! the implementation mirrors the paper's framing and leaves room for a
//! richer baseline later. This is a **heuristic proxy**, consistent with this
//! codebase's other `-lite` modules (e.g. `context_pruning`'s
//! LLMLingua-perplexity-lite, `semantic_entropy`'s entropy-lite) — it is not
//! the paper's LM-based CXMI.

use std::collections::{HashMap, HashSet};

// ── tokenisation ───────────────────────────────────────────────────────────────

/// Tokenise `text`: split on non-alphanumeric characters, lowercase, and drop
/// single-character fragments. Mirrors the tokenizer idiom used throughout
/// `OxiRAG` (see `query_difficulty::predictor::tokenise` and
/// `recomp::extractive::tokenize`).
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

/// Case-preserving alphanumeric tokens of `text` (length `>= 2`), used by the
/// heuristic STRINC entity check so capitalization survives.
fn raw_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_string)
        .collect()
}

/// Jaccard similarity between two token sets, in `[0.0, 1.0]`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

// ── sentence splitting ────────────────────────────────────────────────────────

/// Split `text` into sentences on `.`/`?`/`!` followed by whitespace, or on
/// blank lines, discarding empty fragments. Mirrors
/// `recomp::extractive::split_sentences`.
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

// ── STRINC ─────────────────────────────────────────────────────────────────────

/// Strict, paper-faithful STRINC score used when the answer is known: `1.0`
/// if `sentence` contains `expected_answer` as a case-insensitive substring
/// (and `expected_answer` is non-blank), else `0.0`.
pub(crate) fn strinc_strict_score(sentence: &str, expected_answer: &str) -> f32 {
    let trimmed_answer = expected_answer.trim();
    if trimmed_answer.is_empty() {
        return 0.0;
    }
    if sentence
        .to_lowercase()
        .contains(&trimmed_answer.to_lowercase())
    {
        1.0
    } else {
        0.0
    }
}

/// Length of the longest contiguous token n-gram (capped at `cap`) shared
/// between `sentence_tokens` and `query_tokens`, searching from `cap` down to
/// `1` so the first (longest) match found is returned. Returns `0` if no
/// shared n-gram of any length exists.
fn longest_shared_ngram_len(
    sentence_tokens: &[String],
    query_tokens: &[String],
    cap: usize,
) -> usize {
    if sentence_tokens.is_empty() || query_tokens.is_empty() || cap == 0 {
        return 0;
    }
    let max_possible = sentence_tokens.len().min(query_tokens.len()).min(cap);
    for n in (1..=max_possible).rev() {
        let query_ngrams: HashSet<&[String]> = query_tokens.windows(n).collect();
        if sentence_tokens.windows(n).any(|w| query_ngrams.contains(w)) {
            return n;
        }
    }
    0
}

/// Returns `true` if `sentence` contains a case-preserved token that is both
/// (a) also present, case-insensitively, in `query_token_set`, and (b) either
/// a capitalized proper-noun-like token or a purely numeric token.
fn has_answer_like_entity_overlap(sentence: &str, query_token_set: &HashSet<String>) -> bool {
    raw_tokens(sentence).iter().any(|token| {
        let lower = token.to_lowercase();
        if !query_token_set.contains(&lower) {
            return false;
        }
        let is_capitalized = token.chars().next().is_some_and(char::is_uppercase);
        let is_numeric = token.chars().all(|c| c.is_ascii_digit());
        is_capitalized || is_numeric
    })
}

/// Heuristic STRINC score used when no answer is known (see the module-level
/// documentation for the two signals combined here): `1.0` if `sentence`
/// shares a contiguous token n-gram of length `>= min_ngram` with the query,
/// or contains an answer-like entity/number token also implied by the query;
/// otherwise a fractional value in `[0.0, 1.0)` reflecting the longest
/// sub-threshold n-gram overlap found.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn strinc_heuristic_score(
    sentence: &str,
    query_tokens_ordered: &[String],
    query_token_set: &HashSet<String>,
    min_ngram: usize,
) -> f32 {
    let sentence_tokens = tokenize(sentence);
    let cap = min_ngram.max(1);
    let ngram_len = longest_shared_ngram_len(&sentence_tokens, query_tokens_ordered, cap);
    let ngram_component = ngram_len as f32 / cap as f32;

    let entity_component = if has_answer_like_entity_overlap(sentence, query_token_set) {
        1.0
    } else {
        0.0
    };

    ngram_component.max(entity_component)
}

// ── Lexical overlap ────────────────────────────────────────────────────────────

/// Lexical-overlap score: Jaccard similarity between `sentence_tokens` and
/// `query_tokens`.
pub(crate) fn lexical_overlap_score(
    sentence_tokens: &HashSet<String>,
    query_tokens: &HashSet<String>,
) -> f32 {
    jaccard(sentence_tokens, query_tokens)
}

// ── CXMI-lite ──────────────────────────────────────────────────────────────────

/// Document frequency (number of sentences containing each token) across a
/// sentence corpus, the "IDF" half of the TF-IDF-lite relevance proxy used by
/// [`cxmi_lite_score`]. Mirrors `recomp::extractive::score_sentences`'s
/// document-frequency accumulation.
pub(crate) fn document_frequency(
    sentence_token_sets: &[HashSet<String>],
) -> HashMap<String, usize> {
    let mut document_frequency: HashMap<String, usize> = HashMap::new();
    for tokens in sentence_token_sets {
        for token in tokens {
            *document_frequency.entry(token.clone()).or_insert(0) += 1;
        }
    }
    document_frequency
}

/// Inverse-document-frequency weight of `token` given its `document_frequency`
/// count (defaulting to `1` for tokens absent from the corpus, matching
/// `recomp::extractive::score_sentences`) and the total `sentence_count`.
#[allow(clippy::cast_precision_loss)]
fn idf_weight(
    token: &str,
    document_frequency: &HashMap<String, usize>,
    sentence_count: usize,
) -> f32 {
    let count = document_frequency.get(token).copied().unwrap_or(1);
    ((sentence_count as f32 + 1.0) / (count as f32 + 1.0)).ln() + 1.0
}

/// The CXMI-lite relevance proxy: an IDF-weighted recall of `query_tokens`
/// covered by `context_tokens`, in `[0.0, 1.0]`. Returns `0.0` when
/// `query_tokens` is empty (nothing to cover) or every weight is
/// non-positive.
pub(crate) fn idf_weighted_relevance(
    query_tokens: &HashSet<String>,
    context_tokens: &HashSet<String>,
    document_frequency: &HashMap<String, usize>,
    sentence_count: usize,
) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let mut covered = 0.0_f32;
    let mut total = 0.0_f32;
    for token in query_tokens {
        let weight = idf_weight(token, document_frequency, sentence_count);
        total += weight;
        if context_tokens.contains(token) {
            covered += weight;
        }
    }
    if total <= 0.0 { 0.0 } else { covered / total }
}

/// CXMI-lite score for a single sentence: the relevance gain the sentence
/// provides over an empty context, `relevance(query, [sentence]) -
/// relevance(query, [])`. See the module-level documentation for the
/// heuristic proxy this stands in for.
pub(crate) fn cxmi_lite_score(
    sentence_tokens: &HashSet<String>,
    query_tokens: &HashSet<String>,
    document_frequency: &HashMap<String, usize>,
    sentence_count: usize,
) -> f32 {
    let empty_context: HashSet<String> = HashSet::new();
    let relevance_with_sentence = idf_weighted_relevance(
        query_tokens,
        sentence_tokens,
        document_frequency,
        sentence_count,
    );
    let relevance_without_sentence = idf_weighted_relevance(
        query_tokens,
        &empty_context,
        document_frequency,
        sentence_count,
    );
    relevance_with_sentence - relevance_without_sentence
}
