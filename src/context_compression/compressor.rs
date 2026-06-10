//! Extractive compressor and redundancy filter implementations.

use std::collections::HashSet;

use crate::types::SearchResult;

use super::types::{CompressedContext, CompressionConfig, CompressionError, ContextCompressor};

// ── token helper ──────────────────────────────────────────────────────────────

/// Approximate token count (word-split).
fn approx_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Lowercase token set.
fn token_set(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() >= 2)
        .collect()
}

/// Jaccard similarity between two token sets.
#[allow(clippy::cast_precision_loss)]
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

// ── sentence splitter ─────────────────────────────────────────────────────────

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        current.push(chars[i]);
        if i + 1 < n {
            let end_punct = chars[i] == '.' || chars[i] == '?' || chars[i] == '!';
            if end_punct && chars[i + 1] == ' ' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current = String::new();
                i += 2;
                continue;
            }
        }
        if i + 1 < n && chars[i] == '\n' && chars[i + 1] == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
            i += 2;
            continue;
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

// ── RedundancyFilter ──────────────────────────────────────────────────────────

/// Filters near-duplicate sentences by Jaccard similarity.
#[derive(Debug, Clone, Default)]
pub struct RedundancyFilter;

impl RedundancyFilter {
    /// Create a new [`RedundancyFilter`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Return the subset of `sentences` that are not near-duplicates of each other.
    ///
    /// The first occurrence of each near-duplicate cluster is kept.
    #[must_use]
    pub fn filter(&self, sentences: &[String], threshold: f32) -> Vec<String> {
        let mut kept: Vec<String> = Vec::new();
        let mut kept_tokens: Vec<HashSet<String>> = Vec::new();

        for sentence in sentences {
            let tokens = token_set(sentence);
            let is_dup = kept_tokens.iter().any(|k| jaccard(k, &tokens) >= threshold);
            if !is_dup {
                kept.push(sentence.clone());
                kept_tokens.push(tokens);
            }
        }
        kept
    }
}

// ── ExtractiveCompressor ──────────────────────────────────────────────────────

/// Extractive compressor: splits, scores, deduplicates, packs under budget.
///
/// Steps:
/// 1. Collect all sentences from all source documents.
/// 2. Score each by Jaccard relevance to the query.
/// 3. Sort descending by score.
/// 4. Apply redundancy filtering.
/// 5. Greedily pack sentences within the token budget.
#[derive(Debug, Clone, Default)]
pub struct ExtractiveCompressor {
    filter: RedundancyFilter,
}

impl ExtractiveCompressor {
    /// Create a new [`ExtractiveCompressor`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            filter: RedundancyFilter::new(),
        }
    }
}

impl ContextCompressor for ExtractiveCompressor {
    fn compress(
        &self,
        query: &str,
        sources: &[SearchResult],
        config: &CompressionConfig,
    ) -> Result<CompressedContext, CompressionError> {
        if sources.is_empty() {
            return Err(CompressionError::EmptyInput);
        }

        let query_tokens = token_set(query);

        // Collect all sentences with their relevance scores
        let mut scored: Vec<(String, f32)> = Vec::new();
        let mut original_tokens = 0usize;

        for source in sources {
            let text = &source.document.content;
            original_tokens += approx_tokens(text);
            for sentence in split_sentences(text) {
                if sentence.len() < 4 {
                    continue;
                }
                let st = token_set(&sentence);
                let score = jaccard(&query_tokens, &st);
                if score >= config.relevance_threshold {
                    scored.push((sentence, score));
                }
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let sentences: Vec<String> = scored.into_iter().map(|(s, _)| s).collect();

        // Deduplicate
        let deduped = self.filter.filter(&sentences, config.redundancy_threshold);

        // Pack within token budget
        let mut kept_sentences: Vec<String> = Vec::new();
        let mut compressed_tokens = 0usize;

        for sentence in deduped {
            let t = approx_tokens(&sentence);
            if config.token_budget == 0 {
                break;
            }
            if compressed_tokens + t > config.token_budget {
                break;
            }
            compressed_tokens += t;
            kept_sentences.push(sentence);
        }

        let text = kept_sentences.join(" ");
        let ratio = if original_tokens == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let r = compressed_tokens as f32 / original_tokens as f32;
            r
        };

        Ok(CompressedContext {
            text,
            kept_sentences,
            original_tokens,
            compressed_tokens,
            ratio,
        })
    }
}

// ── MockCompressor ────────────────────────────────────────────────────────────

/// Scripted compressor for unit tests.
///
/// Returns a fixed text for every call.
pub struct MockCompressor {
    /// The text returned by every [`ContextCompressor::compress`] call.
    pub output: String,
}

impl MockCompressor {
    /// Create a mock with the given fixed `output`.
    #[must_use]
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
        }
    }
}

impl ContextCompressor for MockCompressor {
    fn compress(
        &self,
        _query: &str,
        sources: &[SearchResult],
        _config: &CompressionConfig,
    ) -> Result<CompressedContext, CompressionError> {
        if sources.is_empty() {
            return Err(CompressionError::EmptyInput);
        }
        let tokens = approx_tokens(&self.output);
        Ok(CompressedContext {
            text: self.output.clone(),
            kept_sentences: vec![self.output.clone()],
            original_tokens: tokens,
            compressed_tokens: tokens,
            ratio: 1.0,
        })
    }
}
