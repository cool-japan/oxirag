//! Sentence-to-source alignment and the top-level [`Attributor`] orchestrator.
//!
//! # Alignment pipeline
//!
//! 1. [`AlignmentScorer`] — trait: score a sentence against a source passage.
//! 2. [`LexicalAligner`] — token-Jaccard implementation of [`AlignmentScorer`].
//! 3. [`SentenceAligner`] — splits the answer into sentences and attaches the
//!    top-N citations whose score is above the configured threshold.
//! 4. [`Attributor`] — full pipeline: align → deduplicate → annotate →
//!    faithfulness.

use std::collections::{HashMap, HashSet};

use super::citation::{CitationFormatter, CitationStyle, split_sentences};
use super::faithfulness::FaithfulnessChecker;
use super::types::{
    AttributedAnswer, AttributionConfig, AttributionError, Citation, CitationId, CitedSpan,
};
use crate::types::SearchResult;

// ── Private helpers ───────────────────────────────────────────────────────────

/// Tokenise text for Jaccard similarity.
///
/// Steps:
/// 1. Lowercase the entire string.
/// 2. Split on whitespace.
/// 3. Discard tokens that are entirely ASCII punctuation.
/// 4. Discard tokens shorter than 2 characters.
fn tokenize_for_jaccard(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split_whitespace()
        .filter(|t| {
            // Keep token only if it has at least one non-punctuation character.
            t.len() >= 2 && !t.chars().all(|c| c.is_ascii_punctuation())
        })
        .map(|t| {
            // Strip surrounding punctuation from each token so "rust," == "rust".
            t.trim_matches(|c: char| c.is_ascii_punctuation())
                .to_string()
        })
        .filter(|t| t.len() >= 2)
        .collect()
}

/// Token-Jaccard coefficient between two token sets derived from raw text.
///
/// Returns `0.0` when the union is empty.
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

// ── AlignmentScorer ───────────────────────────────────────────────────────────

/// Trait for synchronous sentence-to-source relevance scoring.
///
/// Implementations must be pure compute — no I/O, no async.
pub trait AlignmentScorer {
    /// Score the relevance of `sentence` to `source_text`.
    ///
    /// Returns a value in `[0.0, 1.0]` where `1.0` means identical.
    fn score(&self, sentence: &str, source_text: &str) -> f32;
}

// ── LexicalAligner ────────────────────────────────────────────────────────────

/// Token-Jaccard alignment scorer.
///
/// Tokenises both inputs (lowercase, remove punct-only tokens, len ≥ 2) and
/// returns the Jaccard coefficient over their token sets.
pub struct LexicalAligner;

impl LexicalAligner {
    /// Construct a new [`LexicalAligner`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for LexicalAligner {
    fn default() -> Self {
        Self::new()
    }
}

impl AlignmentScorer for LexicalAligner {
    fn score(&self, sentence: &str, source_text: &str) -> f32 {
        let a = tokenize_for_jaccard(sentence);
        let b = tokenize_for_jaccard(source_text);
        jaccard(&a, &b)
    }
}

// ── SentenceAligner ───────────────────────────────────────────────────────────

/// Splits an answer into sentences and attaches citations above the alignment
/// threshold.
pub struct SentenceAligner<A: AlignmentScorer> {
    scorer: A,
    config: AttributionConfig,
}

impl<A: AlignmentScorer> SentenceAligner<A> {
    /// Build a [`SentenceAligner`] with the provided scorer and config.
    #[must_use]
    pub fn new(scorer: A, config: AttributionConfig) -> Self {
        Self { scorer, config }
    }

    /// Split `answer` into [`CitedSpan`]s, attaching the top-N citations per
    /// sentence.
    ///
    /// # Algorithm
    ///
    /// 1. Split the answer into sentences (`.`/`?`/`!`/`\n\n` boundaries).
    /// 2. For each `(index, sentence)`, score against every source's content.
    /// 3. Retain `(score, source)` pairs where `score >= alignment_threshold`.
    /// 4. Sort descending by score; take up to `max_citations_per_sentence`.
    /// 5. Build [`Citation`]s (`id = "c{global_counter}"`).
    /// 6. `grounding_score` = best score among attached citations (`0.0` if
    ///    none).
    #[must_use]
    pub fn align(&self, answer: &str, sources: &[SearchResult]) -> Vec<CitedSpan> {
        let sentences = split_sentences(answer);
        let mut counter: usize = 0;
        let mut spans = Vec::with_capacity(sentences.len());

        for (sentence_index, sentence) in sentences.iter().enumerate() {
            // Score this sentence against every source.
            let mut scored: Vec<(f32, &SearchResult)> = sources
                .iter()
                .map(|src| {
                    let s = self.scorer.score(sentence, &src.document.content);
                    (s, src)
                })
                .filter(|(s, _)| *s >= self.config.alignment_threshold)
                .collect();

            // Sort descending by score.
            scored.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            // Cap to max citations per sentence.
            scored.truncate(self.config.max_citations_per_sentence);

            // Determine best score.
            let grounding_score = scored.first().map_or(0.0_f32, |(s, _)| *s);

            // Build Citation objects.
            let citations: Vec<Citation> = scored
                .iter()
                .map(|(_, src)| {
                    let id = CitationId::new(format!("c{counter}"));
                    counter += 1;
                    let mut c =
                        Citation::new(id, src.document.id.clone(), src.document.content.clone());
                    if let Some(title) = &src.document.title {
                        c = c.with_title(title.clone());
                    }
                    c
                })
                .collect();

            spans.push(CitedSpan {
                sentence: sentence.clone(),
                sentence_index,
                citations,
                grounding_score,
            });
        }

        spans
    }
}

// ── Attributor ────────────────────────────────────────────────────────────────

/// Top-level attribution orchestrator.
///
/// Runs the complete pipeline: split → align → deduplicate → annotate →
/// faithfulness.
pub struct Attributor<A: AlignmentScorer> {
    aligner: SentenceAligner<A>,
    formatter: CitationFormatter,
    checker: FaithfulnessChecker,
    config: AttributionConfig,
}

impl Attributor<LexicalAligner> {
    /// Build an [`Attributor`] with the default [`LexicalAligner`] pipeline.
    #[must_use]
    pub fn new(config: AttributionConfig) -> Self {
        Self {
            aligner: SentenceAligner::new(LexicalAligner::new(), config.clone()),
            formatter: CitationFormatter::new(),
            checker: FaithfulnessChecker::new(),
            config,
        }
    }
}

impl<A: AlignmentScorer> Attributor<A> {
    /// Build an [`Attributor`] with a custom [`AlignmentScorer`].
    #[must_use]
    pub fn with_scorer(scorer: A, config: AttributionConfig) -> Self {
        Self {
            aligner: SentenceAligner::new(scorer, config.clone()),
            formatter: CitationFormatter::new(),
            checker: FaithfulnessChecker::new(),
            config,
        }
    }

    /// Run the full attribution pipeline.
    ///
    /// # Errors
    ///
    /// - [`AttributionError::EmptyAnswer`] — answer is blank after trimming.
    /// - [`AttributionError::NoSources`] — no source documents provided.
    pub fn attribute(
        &self,
        answer: &str,
        sources: &[SearchResult],
    ) -> Result<AttributedAnswer, AttributionError> {
        // Step 1: guard empty answer.
        if answer.trim().is_empty() {
            return Err(AttributionError::EmptyAnswer);
        }

        // Step 2: guard no sources.
        if sources.is_empty() {
            return Err(AttributionError::NoSources);
        }

        // Step 3: align sentences to sources.
        let mut spans = self.aligner.align(answer, sources);

        // Step 4 & 5: deduplicate citations by source_id, assign stable numeric
        // ids in first-seen order.
        let mut source_to_dedup_index: HashMap<crate::types::DocumentId, usize> = HashMap::new();
        let mut deduped_citations: Vec<Citation> = Vec::new();

        for span in &mut spans {
            for citation in &mut span.citations {
                let next_index = source_to_dedup_index.len() + 1;
                let idx = *source_to_dedup_index
                    .entry(citation.source_id.clone())
                    .or_insert(next_index);

                // If this is a new source, push a deduped citation entry.
                if idx == source_to_dedup_index.len() {
                    // Just inserted — need the citation added to deduped list.
                    let deduped = Citation {
                        id: CitationId::new(idx.to_string()),
                        source_id: citation.source_id.clone(),
                        source_title: citation.source_title.clone(),
                        supporting_text: citation.supporting_text.clone(),
                    };
                    deduped_citations.push(deduped);
                }

                // Update this span's citation id to the dedup index.
                citation.id = CitationId::new(idx.to_string());
            }
        }

        // Rebuild deduped list in correct insertion order.
        // The loop above may have a subtle issue: we need to re-derive deduped_citations
        // from the final source_to_dedup_index map in insertion order.
        // Reset and rebuild cleanly.
        let mut ordered_sources: Vec<(crate::types::DocumentId, usize)> = source_to_dedup_index
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        ordered_sources.sort_by_key(|(_, idx)| *idx);

        // Rebuild deduped_citations from spans (first occurrence wins for text/title).
        deduped_citations = {
            let mut out: Vec<Option<Citation>> = vec![None; ordered_sources.len()];
            'outer: for span in &spans {
                for citation in &span.citations {
                    if let Some(idx) = source_to_dedup_index.get(&citation.source_id) {
                        let pos = idx - 1; // 0-based position
                        if out[pos].is_none() {
                            out[pos] = Some(Citation {
                                id: CitationId::new(idx.to_string()),
                                source_id: citation.source_id.clone(),
                                source_title: citation.source_title.clone(),
                                supporting_text: citation.supporting_text.clone(),
                            });
                        }
                        // Check if all slots are filled to short-circuit.
                        if out.iter().all(Option::is_some) {
                            break 'outer;
                        }
                    }
                }
            }
            out.into_iter().flatten().collect()
        };

        // Step 6: annotate the answer.
        let annotated_answer = self.formatter.annotate(
            answer,
            &spans,
            self.config.citation_style,
            &deduped_citations,
        );

        // Step 7: compute overall faithfulness.
        let overall_faithfulness = self
            .checker
            .faithfulness(&spans, self.config.grounding_threshold);

        // Step 8: return.
        Ok(AttributedAnswer {
            original_answer: answer.to_string(),
            annotated_answer,
            citations: deduped_citations,
            spans,
            overall_faithfulness,
        })
    }
}

// Keep CitationStyle in scope to satisfy the const type annotation used
// for dead-code suppression in the style constant below.
const _DEFAULT_STYLE: CitationStyle = CitationStyle::Numeric;
