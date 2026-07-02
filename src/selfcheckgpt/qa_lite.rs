//! [`SelfCheckVariant::QaLite`](super::types::SelfCheckVariant::QaLite): a
//! cloze/keyword-masking stand-in for the paper's QA-based variant.
//!
//! The real SelfCheckGPT-QA variant generates a genuine question from each
//! sentence (e.g. with an LLM or a trained question-generation model), poses
//! it against every sample using a question-answering model, and scores
//! agreement between the recovered answers. We have neither a generator nor
//! an answering model available, so this module uses a simple **cloze /
//! keyword-masking heuristic** as an honest proxy:
//!
//! 1. Tokenise the sentence and pick its most *salient* content token — the
//!    longest token that is not a common function word (a simple, cheap
//!    stand-in for "the answer span an LLM-generated question would probe").
//!    This is equivalent to masking that token and forming an implicit cloze
//!    question ("`... ___ ...`?").
//! 2. For each sample, check whether the same keyword token appears anywhere
//!    in it (case-insensitive, exact token match).
//! 3. The inconsistency score is the fraction of samples in which the
//!    keyword is **absent** — i.e. the samples that "fail to answer" the
//!    implicit cloze question the same way the main response did.
//!
//! This is documented honestly as a heuristic proxy: no real question
//! generation or question-answering model is used anywhere in this
//! computation.

use super::scorer::tokenize;

/// A sentence with no content word to mask (e.g. only function words) cannot
/// form a cloze question; it is assigned this neutral "unknown"
/// inconsistency score rather than a fabricated 0 or 1.
const NO_KEYWORD_NEUTRAL_SCORE: f32 = 0.5;

/// Minimum token length to be eligible as the masked keyword. Filters out
/// short function words that slip past [`QA_LITE_STOP_WORDS`].
const MIN_KEYWORD_LEN: usize = 3;

/// A small, self-contained function-word list used only to steer salient-token
/// selection away from common connective/auxiliary words. Intentionally
/// independent from other modules' stop-word lists to keep this module
/// self-contained.
static QA_LITE_STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "and", "or", "but", "if",
    "of", "in", "on", "at", "by", "for", "to", "with", "as", "from", "into", "this", "that",
    "these", "those", "it", "its", "they", "them", "he", "she", "we", "you", "his", "her", "our",
    "your", "their", "not", "no", "will", "would", "can", "could", "should", "may", "might", "has",
    "have", "had", "does", "did", "do",
];

/// Selects the most salient content token from `tokens`: the longest token
/// that is at least [`MIN_KEYWORD_LEN`] characters and not in
/// [`QA_LITE_STOP_WORDS`]. Ties are broken by first occurrence.
///
/// Returns `None` when no token qualifies (e.g. a sentence made entirely of
/// short function words).
fn select_salient_token(tokens: &[String]) -> Option<String> {
    tokens
        .iter()
        .filter(|t| t.len() >= MIN_KEYWORD_LEN && !QA_LITE_STOP_WORDS.contains(&t.as_str()))
        .max_by_key(|t| t.len())
        .cloned()
}

/// Computes the cloze/keyword-masking QA-lite inconsistency score for each
/// sentence in `main_sentences` against `samples`.
///
/// Returns one score per entry of `main_sentences`, in the same order.
/// Assumes `samples` is non-empty (callers must enforce
/// [`SelfCheckConfig::min_samples`](super::types::SelfCheckConfig::min_samples)
/// before calling this).
pub(crate) fn qa_lite_inconsistency_scores(
    main_sentences: &[String],
    samples: &[String],
) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; main_sentences.len()];
    }

    let sample_token_sets: Vec<Vec<String>> = samples.iter().map(|s| tokenize(s)).collect();
    #[allow(clippy::cast_precision_loss)]
    let sample_count_f32 = samples.len() as f32;

    main_sentences
        .iter()
        .map(|sentence| {
            let tokens = tokenize(sentence);
            match select_salient_token(&tokens) {
                None => NO_KEYWORD_NEUTRAL_SCORE,
                Some(keyword) => {
                    let absent_count = sample_token_sets
                        .iter()
                        .filter(|sample_tokens| !sample_tokens.contains(&keyword))
                        .count();
                    #[allow(clippy::cast_precision_loss)]
                    let absent_count_f32 = absent_count as f32;
                    absent_count_f32 / sample_count_f32
                }
            }
        })
        .collect()
}
