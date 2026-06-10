//! Chain-of-Note engine: per-document extractive notes synthesized into a final answer.

use std::collections::HashSet;

use crate::types::{Document, SearchResult};

use super::types::{ChainOfNoteError, DocumentNote, NoteChain, NoteConfig};

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Split `text` into sentence slices using `. `, `! `, `? `, and `.\n` as boundaries.
pub(crate) fn sentence_split(text: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        let ch = bytes[i];
        let is_end_punct = ch == b'.' || ch == b'!' || ch == b'?';

        if is_end_punct && i + 1 < len {
            let next = bytes[i + 1];
            if next == b' ' || next == b'\n' {
                let slice = text[start..=i].trim();
                if !slice.is_empty() {
                    parts.push(slice);
                }
                start = i + 2;
                i += 2;
                continue;
            }
        }

        // Also split on standalone newlines
        if ch == b'\n' {
            let slice = text[start..i].trim();
            if !slice.is_empty() {
                parts.push(slice);
            }
            start = i + 1;
        }

        i += 1;
    }

    // Remainder
    if start < len {
        let slice = text[start..].trim();
        if !slice.is_empty() {
            parts.push(slice);
        }
    }

    parts
}

/// Tokenise `text` into lowercase alphanumeric tokens.
fn tokenise(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Compute the Jaccard similarity between two strings.
///
/// Both strings are tokenised (split on non-alphanumeric characters, lowercased).
/// Returns `0.0` when both token sets are empty.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn jaccard(a: &str, b: &str) -> f32 {
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

/// Score a single `sentence` against `query` using Jaccard similarity.
fn score_sentence(sentence: &str, query: &str) -> f32 {
    jaccard(sentence, query)
}

/// Extract a note from `content` by selecting the top-`max_notes` sentences
/// by Jaccard score, joining them, and truncating to `max_len` characters.
///
/// Returns `(note_text, avg_score)`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn extract_note(
    content: &str,
    query: &str,
    max_notes: usize,
    max_len: usize,
) -> (String, f32) {
    if content.is_empty() || max_notes == 0 {
        return (String::new(), 0.0);
    }

    let sentences: Vec<&str> = sentence_split(content);
    if sentences.is_empty() {
        // Treat the whole content as one sentence
        let score = score_sentence(content, query);
        let note = if content.len() > max_len {
            content.chars().take(max_len).collect()
        } else {
            content.to_string()
        };
        return (note, score);
    }

    // Score each sentence
    let mut scored: Vec<(f32, &str)> = sentences
        .iter()
        .map(|s| (score_sentence(s, query), *s))
        .collect();

    // Sort by descending score, take top-max_notes
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(max_notes);

    let avg_score = if scored.is_empty() {
        0.0
    } else {
        let sum: f32 = scored.iter().map(|(s, _)| s).sum();
        sum / scored.len() as f32
    };

    // Join sentences in original order (sort by position back)
    let top_texts: HashSet<&str> = scored.iter().map(|(_, t)| *t).collect();
    let ordered: Vec<&str> = sentences
        .iter()
        .filter(|s| top_texts.contains(*s))
        .copied()
        .collect();

    let joined = ordered.join(". ");
    let note = if joined.len() > max_len {
        joined.chars().take(max_len).collect()
    } else {
        joined
    };

    (note, avg_score)
}

// ── ChainOfNoteEngine ─────────────────────────────────────────────────────────

/// Engine that builds a chain of per-document notes and synthesises them into
/// a final answer using extractive heuristics.
pub struct ChainOfNoteEngine {
    config: NoteConfig,
}

impl ChainOfNoteEngine {
    /// Create a new [`ChainOfNoteEngine`] with the given configuration.
    #[must_use]
    pub fn new(config: NoteConfig) -> Self {
        Self { config }
    }

    /// Process a slice of [`SearchResult`]s and produce a [`NoteChain`].
    ///
    /// Each result (up to `config.top_k`) is summarised into a [`DocumentNote`]
    /// via extractive sentence selection.  Notes are then filtered by the
    /// configured relevance threshold, sorted by descending score, and
    /// synthesised into a final answer string.
    ///
    /// # Errors
    ///
    /// Returns [`ChainOfNoteError::EmptyQuery`] when `query` is blank, or
    /// [`ChainOfNoteError::EmptyDocuments`] when `results` is empty.
    pub fn process(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> Result<NoteChain, ChainOfNoteError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(ChainOfNoteError::EmptyQuery);
        }
        if results.is_empty() {
            return Err(ChainOfNoteError::EmptyDocuments);
        }

        let mut notes: Vec<DocumentNote> = Vec::new();

        for result in results.iter().take(self.config.top_k) {
            let doc_id = result.document.id.as_str().to_string();
            let (note_text, avg_score) = extract_note(
                &result.document.content,
                query,
                self.config.max_notes_per_doc,
                self.config.max_note_length,
            );

            notes.push(DocumentNote {
                doc_id,
                note: note_text,
                relevance_score: avg_score,
            });
        }

        // Filter notes above relevance threshold; sort descending by score
        let mut relevant_notes: Vec<DocumentNote> = notes
            .into_iter()
            .filter(|n| n.is_relevant(self.config.relevance_threshold))
            .collect();
        relevant_notes.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Synthesise answer
        let synthesized = if relevant_notes.is_empty() {
            "Based on retrieved documents: No highly relevant information found.".to_string()
        } else {
            let parts: Vec<&str> = relevant_notes.iter().map(|n| n.note.as_str()).collect();
            match parts.len() {
                1 => format!("Based on retrieved documents: {}", parts[0]),
                2 => format!(
                    "Based on retrieved documents: {} Furthermore, {}",
                    parts[0], parts[1]
                ),
                _ => {
                    let head = parts[0];
                    let tail_joined = parts[1..].join(" Furthermore, ");
                    format!("Based on retrieved documents: {head} Furthermore, {tail_joined}")
                }
            }
        };

        Ok(NoteChain {
            notes: relevant_notes,
            synthesized,
            query: query.to_string(),
        })
    }

    /// Process a slice of [`Document`]s directly (assigns a uniform score of 1.0
    /// to each and delegates to [`Self::process`]).
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::process`].
    pub fn process_documents(
        &self,
        query: &str,
        docs: &[Document],
    ) -> Result<NoteChain, ChainOfNoteError> {
        let query_trimmed = query.trim();
        if query_trimmed.is_empty() {
            return Err(ChainOfNoteError::EmptyQuery);
        }
        if docs.is_empty() {
            return Err(ChainOfNoteError::EmptyDocuments);
        }

        let results: Vec<SearchResult> = docs
            .iter()
            .map(|doc| SearchResult {
                document: doc.clone(),
                score: 1.0,
                rank: 0,
            })
            .collect();

        self.process(query, &results)
    }
}

impl Default for ChainOfNoteEngine {
    fn default() -> Self {
        Self::new(NoteConfig::default())
    }
}
