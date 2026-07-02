//! [`SelfCheckVariant::NliLite`](super::types::SelfCheckVariant::NliLite): a
//! lexical-overlap stand-in for the paper's NLI-based variant.
//!
//! The real SelfCheckGPT-NLI variant runs a trained natural-language
//! inference classifier over `(sample, sentence)` pairs and averages the
//! probability of the "contradiction" class. We have no such classifier
//! available, so this module uses **Jaccard token overlap** as an honest,
//! purely lexical proxy:
//!
//! 1. Split every sample into its own sentences.
//! 2. For each main-response sentence, find the *maximum* Jaccard overlap
//!    against any sentence of each sample (its single best-matching
//!    "evidence" sentence).
//! 3. If that maximum overlap is `>=` [`ENTAILMENT_OVERLAP_THRESHOLD`], the
//!    sample is treated as "entailing" the main-response sentence; otherwise
//!    it is treated as "contradiction/neutral" (SelfCheckGPT-NLI collapses
//!    the neutral and contradiction classes together in the same way).
//! 4. The inconsistency score is the fraction of samples that do **not**
//!    entail the sentence.
//!
//! This is documented honestly as a heuristic proxy: no real entailment
//! model is used anywhere in this computation.

use super::scorer::{jaccard, split_sentences, tokenize};

/// Minimum Jaccard token overlap between a main-response sentence and a
/// sample's best-matching sentence for that sample to be counted as
/// "entailing" the main-response sentence.
///
/// Chosen to be low enough that paraphrases with substantial lexical drift
/// still count as support, while unrelated sentences do not.
const ENTAILMENT_OVERLAP_THRESHOLD: f32 = 0.3;

/// Computes the lexical-overlap NLI-lite inconsistency score for each
/// sentence in `main_sentences` against `samples`.
///
/// Returns one score per entry of `main_sentences`, in the same order.
/// Assumes `samples` is non-empty (callers must enforce
/// [`SelfCheckConfig::min_samples`](super::types::SelfCheckConfig::min_samples)
/// before calling this).
pub(crate) fn nli_lite_inconsistency_scores(
    main_sentences: &[String],
    samples: &[String],
) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; main_sentences.len()];
    }

    // Pre-split and pre-tokenise every sample's sentences once, up front.
    let sample_sentence_tokens: Vec<Vec<Vec<String>>> = samples
        .iter()
        .map(|sample| {
            let sentences = split_sentences(sample);
            if sentences.is_empty() {
                vec![tokenize(sample)]
            } else {
                sentences.iter().map(|s| tokenize(s)).collect()
            }
        })
        .collect();

    #[allow(clippy::cast_precision_loss)]
    let sample_count_f32 = samples.len() as f32;

    main_sentences
        .iter()
        .map(|sentence| {
            let sentence_tokens = tokenize(sentence);
            let non_entailing = sample_sentence_tokens
                .iter()
                .filter(|candidate_sentences| {
                    let best_overlap = candidate_sentences
                        .iter()
                        .map(|cand_tokens| jaccard(&sentence_tokens, cand_tokens))
                        .fold(0.0_f32, f32::max);
                    best_overlap < ENTAILMENT_OVERLAP_THRESHOLD
                })
                .count();
            #[allow(clippy::cast_precision_loss)]
            let non_entailing_f32 = non_entailing as f32;
            non_entailing_f32 / sample_count_f32
        })
        .collect()
}
