//! Multi-candidate answer fusion via voting and weighted aggregation.

use std::collections::HashSet;

use super::types::{
    AggregatedAnswer, AggregationConfig, AggregationError, AggregationStrategy, CandidateAnswer,
};

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens.
fn tokenise(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Compute Jaccard similarity between two sentences (token-level).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn jaccard_sentences(a: &str, b: &str) -> f32 {
    let set_a = tokenise(a);
    let set_b = tokenise(b);

    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Greedy deduplication: iterate `sentences` and skip any sentence whose
/// Jaccard similarity to an already-kept sentence is ≥ `threshold`.
pub(crate) fn deduplicate(sentences: &[String], threshold: f32) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();

    'outer: for candidate in sentences {
        for kept_sentence in &kept {
            if jaccard_sentences(candidate, kept_sentence) >= threshold {
                continue 'outer;
            }
        }
        kept.push(candidate.clone());
    }

    kept
}

/// Split `text` into sentences at `. `, `! `, `? ` boundaries.
///
/// Each element is trimmed and non-empty.
pub(crate) fn extract_sentences(text: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        let ch = bytes[i];
        let is_end = ch == b'.' || ch == b'!' || ch == b'?';

        if is_end && i + 1 < len {
            let next = bytes[i + 1];
            if next == b' ' || next == b'\n' {
                let slice = text[start..=i].trim().to_string();
                if !slice.is_empty() {
                    parts.push(slice);
                }
                start = i + 2;
                i += 2;
                continue;
            }
        }

        if ch == b'\n' {
            let slice = text[start..i].trim().to_string();
            if !slice.is_empty() {
                parts.push(slice);
            }
            start = i + 1;
        }

        i += 1;
    }

    // Remainder
    if start < len {
        let slice = text[start..].trim().to_string();
        if !slice.is_empty() {
            parts.push(slice);
        }
    }

    parts
}

// ── AnswerAggregator ──────────────────────────────────────────────────────────

/// Aggregates multiple candidate answers into a single authoritative answer
/// using configurable voting or fusion strategies.
pub struct AnswerAggregator {
    config: AggregationConfig,
}

impl AnswerAggregator {
    /// Create a new [`AnswerAggregator`] with the given configuration.
    #[must_use]
    pub fn new(config: AggregationConfig) -> Self {
        Self { config }
    }

    /// Aggregate `candidates` using the specified `strategy`.
    ///
    /// # Errors
    ///
    /// - [`AggregationError::InsufficientCandidates`] when fewer than
    ///   `config.min_candidates` valid, non-empty candidates are provided.
    /// - [`AggregationError::AllCandidatesEmpty`] when every candidate's text
    ///   is empty after trimming.
    pub fn aggregate(
        &self,
        candidates: &[CandidateAnswer],
        strategy: &AggregationStrategy,
    ) -> Result<AggregatedAnswer, AggregationError> {
        // Filter to valid, non-empty candidates
        let valid: Vec<&CandidateAnswer> = candidates
            .iter()
            .filter(|c| c.is_valid() && !c.text.trim().is_empty())
            .collect();

        if valid.is_empty() {
            return Err(AggregationError::AllCandidatesEmpty);
        }

        if valid.len() < self.config.min_candidates {
            return Err(AggregationError::InsufficientCandidates(
                self.config.min_candidates,
            ));
        }

        let aggregated_text = match strategy {
            AggregationStrategy::MajorityVote => self.majority_vote(&valid),
            AggregationStrategy::WeightedFusion => self.weighted_fusion(&valid),
            AggregationStrategy::Extractive => self.extractive(&valid),
        };

        // Consensus score = mean confidence of all valid candidates
        #[allow(clippy::cast_precision_loss)]
        let consensus_score = {
            let sum: f32 = valid.iter().map(|c| c.confidence).sum();
            sum / valid.len() as f32
        };

        let contributing_candidates: Vec<String> = valid.iter().map(|c| c.text.clone()).collect();

        Ok(AggregatedAnswer {
            text: aggregated_text,
            consensus_score,
            contributing_candidates,
        })
    }

    // ── Strategy implementations ──────────────────────────────────────────────

    /// Majority vote: extract sentences from all candidates, keep those that
    /// appear (by Jaccard ≥ threshold) in the most candidates.
    fn majority_vote(&self, candidates: &[&CandidateAnswer]) -> String {
        // Collect all sentences from all candidates
        let mut all_sentences: Vec<String> = Vec::new();
        for c in candidates {
            all_sentences.extend(extract_sentences(&c.text));
        }

        if all_sentences.is_empty() {
            return candidates
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
        }

        // Unique representative sentences (deduplicated)
        let unique = deduplicate(&all_sentences, self.config.dedup_threshold);

        // For each unique sentence, count how many candidates contain it
        #[allow(clippy::cast_precision_loss)]
        let mut scored: Vec<(usize, f32, &String)> = unique
            .iter()
            .map(|sentence| {
                let vote_count = candidates
                    .iter()
                    .filter(|c| {
                        let c_sentences = extract_sentences(&c.text);
                        c_sentences.iter().any(|cs| {
                            jaccard_sentences(sentence, cs) >= self.config.dedup_threshold
                        }) || c.text.to_lowercase().contains(&sentence.to_lowercase())
                    })
                    .count();
                (vote_count, 0.0f32, sentence)
            })
            .collect();

        // Sort by vote count descending
        scored.sort_by_key(|item| std::cmp::Reverse(item.0));

        // Take top-K unique (up to candidate count)
        let top_k = candidates.len().max(3);
        scored
            .iter()
            .take(top_k)
            .map(|(_, _, s)| s.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Weighted fusion: weight each candidate's sentences by confidence, then
    /// pick the highest-weighted unique sentences.
    fn weighted_fusion(&self, candidates: &[&CandidateAnswer]) -> String {
        // Collect (sentence, weight) pairs
        let mut sentence_weights: Vec<(String, f32)> = Vec::new();
        for c in candidates {
            let sentences = extract_sentences(&c.text);
            if sentences.is_empty() {
                // Treat whole text as single "sentence"
                sentence_weights.push((c.text.clone(), c.confidence));
            } else {
                for s in sentences {
                    sentence_weights.push((s, c.confidence));
                }
            }
        }

        if sentence_weights.is_empty() {
            return String::new();
        }

        // Sort by weight descending
        sentence_weights
            .sort_by(|(_, wa), (_, wb)| wb.partial_cmp(wa).unwrap_or(std::cmp::Ordering::Equal));

        // Extract unique sentences (by dedup_threshold)
        let ordered_sentences: Vec<String> =
            sentence_weights.iter().map(|(s, _)| s.clone()).collect();
        let unique = deduplicate(&ordered_sentences, self.config.dedup_threshold);

        // Take top candidates.len() unique sentences
        let top_k = candidates.len().max(3);
        unique
            .iter()
            .take(top_k)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extractive: take the union of all sentences, deduplicate, join.
    fn extractive(&self, candidates: &[&CandidateAnswer]) -> String {
        let mut all_sentences: Vec<String> = Vec::new();
        for c in candidates {
            let sentences = extract_sentences(&c.text);
            if sentences.is_empty() {
                all_sentences.push(c.text.clone());
            } else {
                all_sentences.extend(sentences);
            }
        }

        if all_sentences.is_empty() {
            return String::new();
        }

        let unique = deduplicate(&all_sentences, self.config.dedup_threshold);
        unique.join(" ")
    }
}

impl Default for AnswerAggregator {
    fn default() -> Self {
        Self::new(AggregationConfig::default())
    }
}
