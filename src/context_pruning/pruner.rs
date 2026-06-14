//! Token-level pruner implementation (LLMLingua-style).

use std::collections::{HashMap, HashSet};

use crate::types::Document;

use super::types::{ContextPruningError, PruneConfig, PrunedContext};

// ── stopwords ─────────────────────────────────────────────────────────────────

/// A small English stopword set used to penalize low-content tokens.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "me", "might", "must", "my", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "our", "out", "over", "own", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "too", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your",
];

// ── tokenization helpers ──────────────────────────────────────────────────────

/// Reduce a surface word to its lowercase alphanumeric scoring token.
///
/// Non-alphanumeric characters are stripped; the result may be empty for tokens
/// that consist purely of punctuation (e.g. `"---"`).
fn scoring_token(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Tokenize text into lowercase alphanumeric tokens for query analysis.
fn query_tokens(query: &str) -> HashSet<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Return `true` if the surface word denotes a capitalized entity.
///
/// A word qualifies when its first alphabetic character is uppercase and it is
/// not an all-uppercase single letter that merely starts a sentence — any word
/// beginning with a capital letter is treated as a candidate entity, which is the
/// conservative LLMLingua-style behavior of never discarding proper nouns.
fn is_capitalized_entity(word: &str) -> bool {
    word.chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(char::is_uppercase)
}

// ── TokenPruner ───────────────────────────────────────────────────────────────

/// Importance assigned to preserved entities so they always survive pruning.
const PRESERVE_IMPORTANCE: f32 = f32::MAX;

/// Additive bonus for tokens that overlap the query.
const QUERY_BONUS: f32 = 4.0;

/// Multiplicative penalty applied to stopword content-ness.
const STOPWORD_PENALTY: f32 = 0.1;

/// Token-level prompt pruner.
///
/// Assigns each word an importance combining inverse-frequency rarity within the
/// context, a content-ness factor that penalizes stopwords, and an optional
/// query-overlap bonus, then keeps the highest-importance words to hit a target
/// compression ratio while emitting them in their original order.
#[derive(Debug, Clone, Default)]
pub struct TokenPruner {
    /// The pruning configuration.
    pub config: PruneConfig,
}

impl TokenPruner {
    /// Create a new [`TokenPruner`] with the given configuration.
    #[must_use]
    pub fn new(config: PruneConfig) -> Self {
        Self { config }
    }

    /// Compute a per-word importance score.
    ///
    /// The score for a word is
    /// `inverse_frequency * content_ness + query_bonus`, where
    /// `inverse_frequency = ln((n + 1) / (count + 1)) + 1` measures rarity within
    /// `words`, `content_ness` is `1.0` for normal words and a small penalty for
    /// stopwords, and `query_bonus` is added when the word's scoring token appears
    /// in `query`. When [`PruneConfig::preserve_entities`] is enabled, capitalized
    /// entities receive the maximum score so they always survive.
    ///
    /// The returned vector has the same length and order as `words`.
    #[must_use]
    pub fn token_importance(&self, words: &[&str], query: Option<&str>) -> Vec<f32> {
        let n = words.len();
        if n == 0 {
            return Vec::new();
        }

        // Frequency of each scoring token across the context.
        let mut freq: HashMap<String, usize> = HashMap::new();
        let tokens: Vec<String> = words
            .iter()
            .map(|w| {
                let tok = scoring_token(w);
                if !tok.is_empty() {
                    *freq.entry(tok.clone()).or_insert(0) += 1;
                }
                tok
            })
            .collect();

        let q_tokens = query.map(query_tokens).unwrap_or_default();

        words
            .iter()
            .zip(tokens.iter())
            .map(|(word, tok)| {
                if self.config.preserve_entities && is_capitalized_entity(word) {
                    return PRESERVE_IMPORTANCE;
                }
                if tok.is_empty() {
                    // Pure-punctuation words carry no information.
                    return 0.0;
                }
                let count = freq.get(tok).copied().unwrap_or(1);
                #[allow(clippy::cast_precision_loss)]
                let inverse_frequency = ((n as f32 + 1.0) / (count as f32 + 1.0)).ln() + 1.0;
                let content_ness = if STOPWORDS.contains(&tok.as_str()) {
                    STOPWORD_PENALTY
                } else {
                    1.0
                };
                let query_bonus = if q_tokens.contains(tok) {
                    QUERY_BONUS
                } else {
                    0.0
                };
                inverse_frequency * content_ness + query_bonus
            })
            .collect()
    }

    /// Number of tokens to keep for a context of `n` words.
    fn keep_count(&self, n: usize) -> usize {
        let ratio = self.config.target_ratio.clamp(0.0, 1.0);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        let target = (ratio * n as f32).round() as usize;
        target.max(self.config.min_tokens).min(n)
    }

    /// Prune a single context to the configured target ratio.
    ///
    /// The context is split on whitespace into surface words, each word is scored
    /// via [`TokenPruner::token_importance`], the top
    /// `max(min_tokens, round(target_ratio * n))` words by importance are
    /// selected (capped at `n`), and they are emitted in their original relative
    /// order joined by single spaces.
    ///
    /// # Errors
    ///
    /// Returns [`ContextPruningError::EmptyContext`] when `context` contains no
    /// words.
    pub fn prune(
        &self,
        context: &str,
        query: Option<&str>,
    ) -> Result<PrunedContext, ContextPruningError> {
        let words: Vec<&str> = context.split_whitespace().collect();
        let n = words.len();
        if n == 0 {
            return Err(ContextPruningError::EmptyContext);
        }

        let importance = self.token_importance(&words, query);
        let keep = self.keep_count(n);

        // Rank indices by importance descending, breaking ties by original order
        // ascending so the selection is fully deterministic.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            importance[b]
                .partial_cmp(&importance[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        // Mark the top `keep` indices as kept.
        let mut keep_mask = vec![false; n];
        for &idx in order.iter().take(keep) {
            keep_mask[idx] = true;
        }

        // Emit kept words in original order.
        let kept_words: Vec<&str> = words
            .iter()
            .zip(keep_mask.iter())
            .filter_map(|(w, &k)| if k { Some(*w) } else { None })
            .collect();
        let kept_tokens = kept_words.len();
        let text = kept_words.join(" ");

        #[allow(clippy::cast_precision_loss)]
        let compression_ratio = if n == 0 {
            0.0
        } else {
            kept_tokens as f32 / n as f32
        };

        Ok(PrunedContext {
            text,
            kept_tokens,
            original_tokens: n,
            compression_ratio,
        })
    }

    /// Prune every document's content independently.
    ///
    /// # Errors
    ///
    /// Returns [`ContextPruningError::EmptyContext`] if any document's content
    /// contains no words.
    pub fn prune_documents(
        &self,
        docs: &[Document],
        query: Option<&str>,
    ) -> Result<Vec<PrunedContext>, ContextPruningError> {
        docs.iter()
            .map(|doc| self.prune(&doc.content, query))
            .collect()
    }
}
