//! The [`QuoteGrounder`] and its supporting lexical helpers.
//!
//! # Pipeline
//!
//! For each claim sentence in a generated answer the grounder:
//!
//! 1. Splits every source document into sentences (`.`/`!`/`?` boundaries).
//! 2. Tokenises the claim and each source sentence (non-alphanumeric split,
//!    length ≥ 2, lowercase).
//! 3. Scores every source sentence by token overlap against the claim and keeps
//!    the maximum.
//! 4. If the best score reaches `QuoteConfig::min_support`, trims the winning
//!    sentence to a window of at most `QuoteConfig::max_quote_tokens` whitespace
//!    tokens centred on the span of tokens that overlap the claim, and returns
//!    the resulting verbatim quote.
//! 5. Otherwise the claim is reported as ungrounded.

use std::collections::HashSet;

use super::types::{GroundedQuote, QuoteConfig, QuoteError};
use crate::types::Document;

// ── Lexical helpers ───────────────────────────────────────────────────────────

/// Tokenise text for overlap scoring.
///
/// Splits on every non-alphanumeric character, discards tokens shorter than two
/// characters, and lowercases the remainder.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Token overlap of `claim` against `source`, in `[0, 1]`.
///
/// Defined as the fraction of the claim's distinct tokens that also appear in
/// the source sentence (claim coverage). Returns `0.0` when the claim has no
/// usable tokens, so an empty claim never grounds.
#[allow(clippy::cast_precision_loss)]
fn overlap_score(claim_tokens: &HashSet<String>, source: &str) -> f32 {
    if claim_tokens.is_empty() {
        return 0.0;
    }
    let source_tokens = tokenize(source);
    let shared = claim_tokens.intersection(&source_tokens).count();
    shared as f32 / claim_tokens.len() as f32
}

/// Split text into sentences on `.`, `!`, and `?` boundaries.
///
/// The terminating punctuation is dropped; each returned sentence is trimmed and
/// non-empty.
fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

/// Trim `sentence` to a window of at most `max_tokens` whitespace tokens centred
/// on the tokens that overlap the claim.
///
/// The returned string is always a contiguous, verbatim slice of `sentence`
/// (joined on single spaces only when the original separators are collapsed by
/// whitespace splitting). When the sentence already fits within `max_tokens` it
/// is returned unchanged.
fn trim_to_window(sentence: &str, claim_tokens: &HashSet<String>, max_tokens: usize) -> String {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if max_tokens == 0 {
        return String::new();
    }
    if words.len() <= max_tokens {
        return words.join(" ");
    }

    // Indices of words whose normalised token overlaps the claim.
    let matched: Vec<usize> = words
        .iter()
        .enumerate()
        .filter_map(|(i, w)| {
            let normalised: String = w
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            if normalised.len() >= 2 && claim_tokens.contains(&normalised) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // Centre of the overlapping span; fall back to the sentence start when no
    // word matches (overlap came only from sub-token fragments).
    let (first, last) = match (matched.first(), matched.last()) {
        (Some(&f), Some(&l)) => (f, l),
        _ => (0, 0),
    };
    let centre = usize::midpoint(first, last);

    // Build a window of `max_tokens` words centred on `centre`, clamped to the
    // sentence bounds while preserving the requested width where possible.
    let half = max_tokens / 2;
    let mut start = centre.saturating_sub(half);
    let mut end = start + max_tokens; // exclusive
    if end > words.len() {
        end = words.len();
        start = end.saturating_sub(max_tokens);
    }
    words[start..end].join(" ")
}

// ── QuoteGrounder ─────────────────────────────────────────────────────────────

/// Finds the minimal verbatim supporting quote for each claim in an answer.
///
/// Distinct from the `attribution` module: attribution aligns whole answer
/// sentences to whole source passages to emit citation markers, whereas the
/// grounder extracts the single best verbatim source span per claim.
#[derive(Debug, Clone)]
pub struct QuoteGrounder {
    /// Configuration controlling the support threshold and quote length.
    pub config: QuoteConfig,
}

impl QuoteGrounder {
    /// Build a new grounder with the supplied configuration.
    #[must_use]
    pub fn new(config: QuoteConfig) -> Self {
        Self { config }
    }

    /// Find the best supporting quote for a single `claim` across all `docs`.
    ///
    /// Scores every sentence of every document by token overlap against the
    /// claim and keeps the maximum. If that score reaches
    /// `QuoteConfig::min_support`, the winning sentence is trimmed to a window of
    /// at most `QuoteConfig::max_quote_tokens` tokens centred on the overlapping
    /// span and returned as a [`GroundedQuote`]. Returns `None` when no sentence
    /// meets the threshold (including when the claim has no usable tokens).
    #[must_use]
    pub fn best_quote(&self, claim: &str, docs: &[Document]) -> Option<GroundedQuote> {
        let claim_tokens = tokenize(claim);
        if claim_tokens.is_empty() {
            return None;
        }

        let mut best: Option<(f32, &Document, String)> = None;
        for doc in docs {
            for sentence in split_sentences(&doc.content) {
                let score = overlap_score(&claim_tokens, &sentence);
                let better = match &best {
                    Some((best_score, _, _)) => score > *best_score,
                    None => true,
                };
                if better {
                    best = Some((score, doc, sentence));
                }
            }
        }

        let (score, doc, sentence) = best?;
        if score < self.config.min_support {
            return None;
        }
        let quote = trim_to_window(&sentence, &claim_tokens, self.config.max_quote_tokens);
        Some(GroundedQuote::new(claim, quote, doc.id.clone(), score))
    }

    /// Ground every claim in `answer`, returning one quote per supported claim.
    ///
    /// The answer is split into claim sentences; each claim that has a supporting
    /// quote contributes one [`GroundedQuote`] to the output, in claim order.
    /// Unsupported claims are omitted (see [`QuoteGrounder::ungrounded`]).
    ///
    /// # Errors
    ///
    /// Returns [`QuoteError::EmptyAnswer`] when `answer` is blank after trimming.
    pub fn ground(
        &self,
        answer: &str,
        docs: &[Document],
    ) -> Result<Vec<GroundedQuote>, QuoteError> {
        if answer.trim().is_empty() {
            return Err(QuoteError::EmptyAnswer);
        }
        Ok(split_sentences(answer)
            .iter()
            .filter_map(|claim| self.best_quote(claim, docs))
            .collect())
    }

    /// List the claims in `answer` that have no supporting quote.
    ///
    /// # Errors
    ///
    /// Returns [`QuoteError::EmptyAnswer`] when `answer` is blank after trimming.
    pub fn ungrounded(&self, answer: &str, docs: &[Document]) -> Result<Vec<String>, QuoteError> {
        if answer.trim().is_empty() {
            return Err(QuoteError::EmptyAnswer);
        }
        Ok(split_sentences(answer)
            .into_iter()
            .filter(|claim| self.best_quote(claim, docs).is_none())
            .collect())
    }
}
