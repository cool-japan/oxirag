//! RAGAS-style lexical evaluation metrics.
//!
//! All metrics are heuristic / lexical — no external LLM call is required.
//! They operate on token sets computed by an internal `tokenize` helper, which lowercases text,
//! splits on whitespace and ASCII punctuation, filters English stop-words, and
//! removes tokens shorter than two characters.

use std::collections::HashSet;

use async_trait::async_trait;

use super::types::{EvalError, EvaluationSample};

// ---------------------------------------------------------------------------
// Private token-level helpers
// ---------------------------------------------------------------------------

/// English stop-words that are excluded from token sets.
static STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "about", "into", "through", "during", "before", "after", "above", "below", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "must", "can", "could", "not", "no", "nor", "so",
    "yet", "both", "either", "each", "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "just", "that", "this", "these", "those", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "how", "when", "where", "why", "if", "as", "any", "all", "only", "also",
];

/// Tokenise `text` into a set of meaningful lowercase tokens.
///
/// Steps:
/// 1. Lowercase the entire string.
/// 2. Replace every ASCII punctuation character with a space.
/// 3. Split on whitespace.
/// 4. Discard tokens that are shorter than two characters.
/// 5. Discard tokens that are in the stop-word list.
fn tokenize(text: &str) -> HashSet<String> {
    let lower = text.to_lowercase();
    let cleaned: String = lower
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect();

    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    cleaned
        .split_whitespace()
        .filter(|t| t.len() >= 2 && !stop.contains(t))
        .map(str::to_string)
        .collect()
}

/// Compute the Jaccard similarity between two token sets.
///
/// Returns `0.0` when both sets are empty.
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

/// Split `text` into sentences using common sentence-ending patterns.
///
/// Delimiters: `. `, `? `, `! `, and newline characters.
/// Empty or whitespace-only fragments are discarded.
fn split_sentences(text: &str) -> Vec<String> {
    // Replace sentence-ending punctuation followed by whitespace / newlines
    // with a sentinel, then split.
    let sentinel = "\x00";
    let normalised = text
        .replace(". ", &format!(". {sentinel}"))
        .replace("? ", &format!("? {sentinel}"))
        .replace("! ", &format!("! {sentinel}"))
        .replace('\n', sentinel);

    normalised
        .split(sentinel)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

// ---------------------------------------------------------------------------
// Trait definition
// ---------------------------------------------------------------------------

/// Trait implemented by all RAGAS-style evaluation metrics.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait EvaluationMetric: Send + Sync {
    /// Human-readable name for this metric.
    fn name(&self) -> &'static str;

    /// Score a single [`EvaluationSample`], returning a value in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::EmptyContext`] when the sample has no context
    /// passages.  Returns [`EvalError::MissingGroundTruth`] for metrics that
    /// require a reference answer.
    async fn score(&self, sample: &EvaluationSample) -> Result<f32, EvalError>;
}

// ---------------------------------------------------------------------------
// Answer Relevance
// ---------------------------------------------------------------------------

/// Measures lexical overlap between the answer and the query.
///
/// Score = Jaccard similarity of token sets.  When `boost_exact_match` is
/// `true` and the answer contains the verbatim query string, the raw Jaccard
/// score is boosted towards 1.0 by blending with 1.0 at 20 % weight.
pub struct AnswerRelevanceScorer {
    /// If `true`, apply a small boost when the answer contains the verbatim query phrase.
    pub boost_exact_match: bool,
}

impl Default for AnswerRelevanceScorer {
    fn default() -> Self {
        Self {
            boost_exact_match: true,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EvaluationMetric for AnswerRelevanceScorer {
    fn name(&self) -> &'static str {
        "answer_relevance"
    }

    async fn score(&self, sample: &EvaluationSample) -> Result<f32, EvalError> {
        if sample.answer.trim().is_empty() {
            return Ok(0.0);
        }

        let query_tokens = tokenize(&sample.query);
        let answer_tokens = tokenize(&sample.answer);
        let mut score = jaccard(&query_tokens, &answer_tokens);

        if self.boost_exact_match {
            let query_lower = sample.query.to_lowercase();
            let answer_lower = sample.answer.to_lowercase();
            if answer_lower.contains(&query_lower) {
                // Blend 80 % raw Jaccard + 20 % perfect match bonus.
                score = 0.8 * score + 0.2;
            }
        }

        Ok(score.min(1.0))
    }
}

// ---------------------------------------------------------------------------
// Faithfulness
// ---------------------------------------------------------------------------

/// Measures what fraction of answer sentences are supported by the context.
///
/// For each sentence in the answer, at least `min_word_overlap` fraction of
/// its content words must appear in at least one context chunk for that
/// sentence to be considered supported.
///
/// `score = supported_sentences / total_sentences`
pub struct FaithfulnessScorer {
    /// Minimum fraction of sentence content words that must appear in a
    /// context chunk for the sentence to be considered supported.
    pub min_word_overlap: f32,
}

impl Default for FaithfulnessScorer {
    fn default() -> Self {
        Self {
            min_word_overlap: 0.5,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EvaluationMetric for FaithfulnessScorer {
    fn name(&self) -> &'static str {
        "faithfulness"
    }

    #[allow(clippy::cast_precision_loss)]
    async fn score(&self, sample: &EvaluationSample) -> Result<f32, EvalError> {
        if sample.context.is_empty() {
            return Err(EvalError::EmptyContext);
        }
        if sample.answer.trim().is_empty() {
            return Ok(0.0);
        }

        // Tokenise all context chunks and union them together.
        let context_tokens: HashSet<String> =
            sample.context.iter().flat_map(|c| tokenize(c)).collect();

        let sentences = split_sentences(&sample.answer);
        if sentences.is_empty() {
            return Ok(0.0);
        }

        let supported = sentences
            .iter()
            .filter(|sentence| {
                let sent_tokens = tokenize(sentence);
                if sent_tokens.is_empty() {
                    return false;
                }
                let overlap = sent_tokens.intersection(&context_tokens).count() as f32;
                overlap / sent_tokens.len() as f32 >= self.min_word_overlap
            })
            .count();

        Ok(supported as f32 / sentences.len() as f32)
    }
}

// ---------------------------------------------------------------------------
// Context Precision
// ---------------------------------------------------------------------------

/// Measures what fraction of retrieved context chunks are relevant to the query.
///
/// A chunk is considered relevant when `jaccard(chunk_tokens, query_tokens) >= relevance_threshold`.
pub struct ContextPrecisionScorer {
    /// Minimum Jaccard score for a context chunk to be considered relevant.
    pub relevance_threshold: f32,
}

impl Default for ContextPrecisionScorer {
    fn default() -> Self {
        Self {
            relevance_threshold: 0.1,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EvaluationMetric for ContextPrecisionScorer {
    fn name(&self) -> &'static str {
        "context_precision"
    }

    #[allow(clippy::cast_precision_loss)]
    async fn score(&self, sample: &EvaluationSample) -> Result<f32, EvalError> {
        if sample.context.is_empty() {
            return Err(EvalError::EmptyContext);
        }

        let query_tokens = tokenize(&sample.query);
        let relevant = sample
            .context
            .iter()
            .filter(|chunk| {
                let chunk_tokens = tokenize(chunk);
                jaccard(&chunk_tokens, &query_tokens) >= self.relevance_threshold
            })
            .count();

        Ok(relevant as f32 / sample.context.len() as f32)
    }
}

// ---------------------------------------------------------------------------
// Context Recall
// ---------------------------------------------------------------------------

/// Measures what fraction of ground-truth answer tokens appear in the context.
///
/// Requires [`EvaluationSample::ground_truth`] to be `Some`.
///
/// `score = |gt_tokens ∩ context_tokens| / |gt_tokens|`
pub struct ContextRecallScorer;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EvaluationMetric for ContextRecallScorer {
    fn name(&self) -> &'static str {
        "context_recall"
    }

    #[allow(clippy::cast_precision_loss)]
    async fn score(&self, sample: &EvaluationSample) -> Result<f32, EvalError> {
        let ground_truth = sample
            .ground_truth
            .as_deref()
            .ok_or(EvalError::MissingGroundTruth)?;

        if sample.context.is_empty() {
            return Err(EvalError::EmptyContext);
        }

        let gt_tokens = tokenize(ground_truth);
        if gt_tokens.is_empty() {
            return Ok(0.0);
        }

        let context_tokens: HashSet<String> =
            sample.context.iter().flat_map(|c| tokenize(c)).collect();

        let recalled = gt_tokens.intersection(&context_tokens).count() as f32;
        Ok(recalled / gt_tokens.len() as f32)
    }
}

// ---------------------------------------------------------------------------
// Overall Scorer
// ---------------------------------------------------------------------------

/// Combines all four metrics into a single weighted overall score.
///
/// Weights must not be zero; their actual normalization is handled in
/// [`super::types::EvaluationResult::weighted_average`].
pub struct OverallScorer {
    /// Weight for answer relevance (default 0.25).
    pub ar_weight: f32,
    /// Weight for faithfulness (default 0.35).
    pub faith_weight: f32,
    /// Weight for context precision (default 0.25).
    pub cp_weight: f32,
    /// Weight for context recall (default 0.15).
    pub cr_weight: f32,
    /// Inner scorer instances.
    answer_relevance: AnswerRelevanceScorer,
    faithfulness: FaithfulnessScorer,
    context_precision: ContextPrecisionScorer,
    context_recall: ContextRecallScorer,
}

impl Default for OverallScorer {
    fn default() -> Self {
        Self {
            ar_weight: 0.25,
            faith_weight: 0.35,
            cp_weight: 0.25,
            cr_weight: 0.15,
            answer_relevance: AnswerRelevanceScorer::default(),
            faithfulness: FaithfulnessScorer::default(),
            context_precision: ContextPrecisionScorer::default(),
            context_recall: ContextRecallScorer,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EvaluationMetric for OverallScorer {
    fn name(&self) -> &'static str {
        "overall"
    }

    async fn score(&self, sample: &EvaluationSample) -> Result<f32, EvalError> {
        let ar = self.answer_relevance.score(sample).await?;
        let faith = self.faithfulness.score(sample).await?;
        let cp = self.context_precision.score(sample).await?;
        let cr = match self.context_recall.score(sample).await {
            Ok(v) => Some(v),
            Err(EvalError::MissingGroundTruth) => None,
            Err(e) => return Err(e),
        };

        // Use the weights stored on this scorer rather than the defaults from
        // EvaluationResult::weighted_average, so callers can customise them.
        let total_w = self.ar_weight + self.faith_weight + self.cp_weight;
        let score = if let Some(cr_val) = cr {
            let full_w = total_w + self.cr_weight;
            (ar * self.ar_weight
                + faith * self.faith_weight
                + cp * self.cp_weight
                + cr_val * self.cr_weight)
                / full_w
        } else {
            (ar * self.ar_weight + faith * self.faith_weight + cp * self.cp_weight) / total_w
        };

        Ok(score.min(1.0))
    }
}

// ---------------------------------------------------------------------------
// Unit tests for helpers
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod helper_tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("The quick brown fox");
        assert!(tokens.contains("quick"));
        assert!(tokens.contains("brown"));
        assert!(tokens.contains("fox"));
        // stop-word "the" must be removed
        assert!(!tokens.contains("the"));
    }

    #[test]
    fn test_tokenize_punctuation() {
        let tokens = tokenize("Hello, world! This is a test.");
        assert!(tokens.contains("hello"));
        assert!(tokens.contains("world"));
        assert!(tokens.contains("test"));
    }

    #[test]
    fn test_tokenize_min_len() {
        let tokens = tokenize("I am a big dog");
        // "i", "a" are single-char or stop-words; "big", "dog" survive
        assert!(tokens.contains("big"));
        assert!(tokens.contains("dog"));
    }

    #[test]
    fn test_jaccard_identical() {
        let a = tokenize("rust programming language");
        let b = tokenize("rust programming language");
        let score = jaccard(&a, &b);
        assert!((score - 1.0_f32).abs() < 1e-6_f32);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = tokenize("apple orange banana");
        let b = tokenize("computer network protocol");
        let score = jaccard(&a, &b);
        assert!(score < 0.01_f32);
    }

    #[test]
    fn test_jaccard_empty_sets() {
        let a: HashSet<String> = HashSet::new();
        let b: HashSet<String> = HashSet::new();
        let score = jaccard(&a, &b);
        assert!((score).abs() < 1e-6_f32);
    }

    #[test]
    fn test_jaccard_one_empty() {
        let a = tokenize("hello world");
        let b: HashSet<String> = HashSet::new();
        let score = jaccard(&a, &b);
        assert!((score).abs() < 1e-6_f32);
    }

    #[test]
    fn test_split_sentences_period() {
        let sentences = split_sentences("First sentence. Second sentence. Third one.");
        assert!(sentences.len() >= 2);
        assert!(sentences[0].contains("First"));
    }

    #[test]
    fn test_split_sentences_question() {
        let sentences = split_sentences("What is Rust? It is a systems language.");
        assert!(sentences.len() >= 2);
    }

    #[test]
    fn test_split_sentences_newline() {
        let sentences = split_sentences("Line one\nLine two\nLine three");
        assert!(sentences.len() >= 2);
    }
}
