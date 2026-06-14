//! Fusion-in-Decoder engine: independent per-passage evidence extraction
//! followed by relevance-weighted fusion with multi-passage attribution.

use std::cmp::Ordering;
use std::collections::HashSet;

use crate::types::{Document, DocumentId};

use super::types::{FidConfig, FidError, FusedAnswer, PassageEvidence};

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens of length `>= 2`.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Split `text` into sentence slices on `.`, `!`, and `?` boundaries.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();

    for (i, &ch) in bytes.iter().enumerate() {
        if ch == b'.' || ch == b'!' || ch == b'?' {
            let slice = text[start..=i].trim();
            if !slice.is_empty() {
                parts.push(slice);
            }
            start = i + 1;
        }
    }

    if start < text.len() {
        let slice = text[start..].trim();
        if !slice.is_empty() {
            parts.push(slice);
        }
    }

    parts
}

/// Compute the token-Jaccard overlap between `a` and `b`.
///
/// Returns `0.0` when both token sets are empty.
#[allow(clippy::cast_precision_loss)]
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
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

/// Total ordering comparator for descending relevance with a stable tie-break.
///
/// Higher relevance comes first; ties are broken lexicographically by passage
/// id so that the fusion output is fully deterministic.
fn cmp_evidence_desc(a: &PassageEvidence, b: &PassageEvidence) -> Ordering {
    b.relevance
        .partial_cmp(&a.relevance)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.passage_id.as_str().cmp(b.passage_id.as_str()))
}

// ── FusionInDecoder ───────────────────────────────────────────────────────────

/// Heuristic Fusion-in-Decoder (Izacard & Grave, 2021).
///
/// Each retrieved passage is processed *independently* to extract its single
/// most query-relevant sentence together with a relevance weight.  The
/// per-passage evidence is then *fused* across passages — deduplicated and
/// concatenated in descending relevance order — into one answer that carries
/// multi-passage attribution.
///
/// This differs from `chain_of_note`, which builds per-document extractive
/// notes and then synthesises them: here extraction is strictly per-passage and
/// independent, and fusion is relevance-weighted with explicit deduplication.
#[derive(Debug, Clone, Default)]
pub struct FusionInDecoder {
    config: FidConfig,
}

impl FusionInDecoder {
    /// Create a new [`FusionInDecoder`] with the given configuration.
    #[must_use]
    pub fn new(config: FidConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration backing this engine.
    #[must_use]
    pub fn config(&self) -> &FidConfig {
        &self.config
    }

    /// Extract the best evidence from each passage, independently.
    ///
    /// For every document the most query-relevant sentence is selected, scored
    /// by the query↔sentence token-overlap (Jaccard).  The resulting evidence is
    /// sorted by descending relevance and capped at `config.top_passages`.
    #[must_use]
    pub fn extract_evidence(&self, query: &str, docs: &[Document]) -> Vec<PassageEvidence> {
        let query_tokens = tokenize(query);

        let mut evidence: Vec<PassageEvidence> = docs
            .iter()
            .map(|doc| {
                let (sentence, relevance) = best_sentence(&doc.content, &query_tokens);
                PassageEvidence::new(doc.id.clone(), sentence, relevance)
            })
            .collect();

        evidence.sort_by(cmp_evidence_desc);
        evidence.truncate(self.config.top_passages);
        evidence
    }

    /// Fuse per-passage evidence into a single attributed answer.
    ///
    /// Evidence is first extracted via [`Self::extract_evidence`], then
    /// near-identical evidence (token-Jaccard `>= config.dedup_threshold`) is
    /// removed — keeping the higher-relevance copy.  The surviving evidence is
    /// concatenated in descending relevance order into `answer`, and the
    /// contributing passage ids are collected into `attributions`.
    ///
    /// # Errors
    ///
    /// Returns [`FidError::EmptyQuery`] when `query` is blank, or
    /// [`FidError::EmptyCorpus`] when `docs` is empty.
    pub fn fuse(&self, query: &str, docs: &[Document]) -> Result<FusedAnswer, FidError> {
        if query.trim().is_empty() {
            return Err(FidError::EmptyQuery);
        }
        if docs.is_empty() {
            return Err(FidError::EmptyCorpus);
        }

        let extracted = self.extract_evidence(query, docs);
        let surviving = self.deduplicate(extracted);

        let answer = surviving
            .iter()
            .filter(|e| !e.evidence.is_empty())
            .map(|e| e.evidence.as_str())
            .collect::<Vec<&str>>()
            .join(" ");

        let mut attributions: Vec<DocumentId> = Vec::new();
        for e in &surviving {
            if !attributions.contains(&e.passage_id) {
                attributions.push(e.passage_id.clone());
            }
        }

        Ok(FusedAnswer {
            answer,
            evidence: surviving,
            attributions,
        })
    }

    /// Remove near-identical evidence, keeping the higher-relevance copy.
    ///
    /// Input is assumed to be sorted by descending relevance, so the first time
    /// a given piece of evidence is seen it is the highest-relevance instance;
    /// later near-duplicates (token-Jaccard `>= config.dedup_threshold`) are
    /// dropped.  Order is preserved.
    fn deduplicate(&self, evidence: Vec<PassageEvidence>) -> Vec<PassageEvidence> {
        let mut kept: Vec<PassageEvidence> = Vec::with_capacity(evidence.len());
        let mut kept_tokens: Vec<HashSet<String>> = Vec::with_capacity(evidence.len());

        for candidate in evidence {
            let candidate_tokens = tokenize(&candidate.evidence);
            let is_duplicate = kept_tokens
                .iter()
                .any(|prior| jaccard(prior, &candidate_tokens) >= self.config.dedup_threshold);
            if !is_duplicate {
                kept_tokens.push(candidate_tokens);
                kept.push(candidate);
            }
        }

        kept
    }
}

/// Select the single most query-relevant sentence from `content`.
///
/// Returns the best sentence and its query↔sentence token-overlap (Jaccard).
/// When `content` contains no sentence boundaries the whole content is scored as
/// one sentence; an empty `content` yields an empty string with relevance `0.0`.
fn best_sentence(content: &str, query_tokens: &HashSet<String>) -> (String, f32) {
    let sentences = split_sentences(content);
    if sentences.is_empty() {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return (String::new(), 0.0);
        }
        let relevance = jaccard(&tokenize(trimmed), query_tokens);
        return (trimmed.to_string(), relevance);
    }

    let mut best_text = sentences[0];
    let mut best_score = jaccard(&tokenize(best_text), query_tokens);

    for sentence in &sentences[1..] {
        let score = jaccard(&tokenize(sentence), query_tokens);
        if score > best_score {
            best_score = score;
            best_text = sentence;
        }
    }

    (best_text.to_string(), best_score)
}
