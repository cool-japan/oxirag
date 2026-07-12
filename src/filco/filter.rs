//! [`FilcoFilter`]: sentence-level context filtering across FILCO's three
//! measures.

use std::collections::HashSet;

use super::measures::{
    cxmi_lite_score, document_frequency, lexical_overlap_score, split_sentences,
    strinc_heuristic_score, strinc_strict_score, token_set, tokenize,
};
use super::types::{FilcoConfig, FilcoError, FilcoReport, FilterMeasure, ScoredSentence};

/// FILCO (Wang et al., 2023): sentence-level context filtering for
/// retrieval-augmented generation.
///
/// Given a query and a set of retrieved passages, [`FilcoFilter::filter`]
/// splits every passage into sentences, scores each sentence with the
/// configured [`FilterMeasure`], drops the sentences that fall at or below
/// the measure's threshold, and reassembles the surviving sentences per
/// passage — preserving their original relative order — into
/// [`FilcoReport::filtered_passages`].
///
/// `FilcoFilter` holds no mutable state; it is safe to use from multiple
/// threads simultaneously.
#[derive(Debug, Clone, Default)]
pub struct FilcoFilter {
    /// The filtering configuration.
    config: FilcoConfig,
}

impl FilcoFilter {
    /// Creates a new [`FilcoFilter`] with the given configuration.
    #[must_use]
    pub fn new(config: FilcoConfig) -> Self {
        Self { config }
    }

    /// Returns the filter's configuration.
    #[must_use]
    pub fn config(&self) -> &FilcoConfig {
        &self.config
    }

    /// Filters `passages` relevant to `query` using the configured
    /// [`FilterMeasure`].
    ///
    /// When [`FilcoConfig::measure`] is [`FilterMeasure::StrInc`], this uses
    /// the heuristic (no known answer) variant — see
    /// `measures::strinc_heuristic_score` for the exact signals. To
    /// use the strict, paper-faithful STRINC check against a known answer
    /// span, call [`Self::filter_with_answer`] instead.
    ///
    /// # Errors
    ///
    /// - [`FilcoError::EmptyQuery`] — `query` is empty or whitespace-only.
    /// - [`FilcoError::EmptyPassages`] — `passages` is empty, or every
    ///   passage is empty or whitespace-only.
    /// - [`FilcoError::InvalidConfig`] — the filter's [`FilcoConfig`] fails
    ///   [`FilcoConfig::validate`].
    pub fn filter(&self, query: &str, passages: &[String]) -> Result<FilcoReport, FilcoError> {
        self.run(query, passages, None)
    }

    /// Filters `passages` relevant to `query`, using the strict,
    /// paper-faithful STRINC check against `expected_answer` regardless of
    /// [`FilcoConfig::measure`].
    ///
    /// This is FILCO's *training-time* formulation: a sentence is kept when
    /// it contains `expected_answer` as a case-insensitive substring. Useful
    /// when the caller already knows the correct answer (e.g. constructing
    /// supervised filtering examples, or re-scoring a generated answer
    /// against the context that produced it).
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::filter`].
    pub fn filter_with_answer(
        &self,
        query: &str,
        passages: &[String],
        expected_answer: &str,
    ) -> Result<FilcoReport, FilcoError> {
        self.run(query, passages, Some(expected_answer))
    }

    /// Shared implementation for [`Self::filter`] and
    /// [`Self::filter_with_answer`]. `expected_answer` forces the strict
    /// STRINC path (regardless of the configured measure) when `Some`.
    fn run(
        &self,
        query: &str,
        passages: &[String],
        expected_answer: Option<&str>,
    ) -> Result<FilcoReport, FilcoError> {
        let trimmed_query = query.trim();
        if trimmed_query.is_empty() {
            return Err(FilcoError::EmptyQuery);
        }
        if passages.is_empty() || passages.iter().all(|p| p.trim().is_empty()) {
            return Err(FilcoError::EmptyPassages);
        }
        self.config.validate()?;

        // Split every passage into sentences, keeping the per-passage
        // grouping so kept sentences can be reassembled in place afterward.
        let per_passage_sentences: Vec<Vec<String>> =
            passages.iter().map(|p| split_sentences(p)).collect();
        let original_sentence_count: usize = per_passage_sentences.iter().map(Vec::len).sum();

        if original_sentence_count == 0 {
            return Ok(FilcoReport {
                filtered_passages: vec![String::new(); passages.len()],
                scored_sentences: Vec::new(),
                original_sentence_count: 0,
                kept_sentence_count: 0,
                compression_ratio: 0.0,
            });
        }

        let query_tokens_ordered = tokenize(trimmed_query);
        let query_token_set: HashSet<String> = query_tokens_ordered.iter().cloned().collect();

        // Sentence-level token sets and corpus-wide document frequencies,
        // needed by the CXMI-lite measure (built once per call over every
        // sentence across every passage).
        let sentence_token_sets: Vec<HashSet<String>> = per_passage_sentences
            .iter()
            .flatten()
            .map(|s| token_set(s))
            .collect();
        let doc_freq = document_frequency(&sentence_token_sets);
        let sentence_count = sentence_token_sets.len();

        let mut scores: Vec<f32> = Vec::with_capacity(original_sentence_count);
        for (sentence, sentence_tokens) in per_passage_sentences
            .iter()
            .flatten()
            .zip(sentence_token_sets.iter())
        {
            let score = match (self.config.measure, expected_answer) {
                (_, Some(answer)) => strinc_strict_score(sentence, answer),
                (FilterMeasure::StrInc, None) => strinc_heuristic_score(
                    sentence,
                    &query_tokens_ordered,
                    &query_token_set,
                    self.config.strinc_min_ngram,
                ),
                (FilterMeasure::LexicalOverlap, None) => {
                    lexical_overlap_score(sentence_tokens, &query_token_set)
                }
                (FilterMeasure::CxmiLite, None) => {
                    cxmi_lite_score(sentence_tokens, &query_token_set, &doc_freq, sentence_count)
                }
            };
            scores.push(score);
        }

        // A known expected_answer always forces the strict STRINC keep rule,
        // regardless of the configured measure (mirrors the scoring match
        // above).
        let effective_measure = if expected_answer.is_some() {
            FilterMeasure::StrInc
        } else {
            self.config.measure
        };

        let mut filtered_passages = Vec::with_capacity(passages.len());
        let mut scored_sentences = Vec::with_capacity(original_sentence_count);
        let mut kept_sentence_count = 0usize;
        let mut cursor = 0usize;

        for sentences_in_passage in &per_passage_sentences {
            let mut kept_texts: Vec<&str> = Vec::with_capacity(sentences_in_passage.len());
            for sentence in sentences_in_passage {
                let score = scores[cursor];
                let kept = self.is_kept(effective_measure, score);
                if kept {
                    kept_texts.push(sentence.as_str());
                    kept_sentence_count += 1;
                }
                scored_sentences.push(ScoredSentence {
                    text: sentence.clone(),
                    score,
                    kept,
                });
                cursor += 1;
            }
            filtered_passages.push(kept_texts.join(" "));
        }

        #[allow(clippy::cast_precision_loss)]
        let compression_ratio = kept_sentence_count as f32 / original_sentence_count as f32;

        Ok(FilcoReport {
            filtered_passages,
            scored_sentences,
            original_sentence_count,
            kept_sentence_count,
            compression_ratio,
        })
    }

    /// Decides whether `score` clears the keep threshold for `measure`.
    ///
    /// STRINC (both heuristic and strict) is a binary match: its score is
    /// exactly `1.0` on a match and strictly less otherwise, so the keep rule
    /// is `score >= 1.0`. Lexical overlap and CXMI-lite compare against their
    /// respective configured thresholds.
    fn is_kept(&self, measure: FilterMeasure, score: f32) -> bool {
        match measure {
            FilterMeasure::StrInc => score >= 1.0,
            FilterMeasure::LexicalOverlap => score > self.config.lexical_threshold,
            FilterMeasure::CxmiLite => score > self.config.cxmi_threshold,
        }
    }
}
