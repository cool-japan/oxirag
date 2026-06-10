//! ITER-RETGEN engine: iterative retrieval, expansion, and draft assembly.

use std::cmp::Reverse;
use std::collections::HashSet;

use crate::layer1_echo::traits::Echo;
use crate::types::SearchResult;

use super::types::{IterationStep, IterativeConfig, IterativeOutput, IterativeRagError};

// ── Stop-word set ─────────────────────────────────────────────────────────────

const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "in", "of", "to", "for", "and", "or", "but", "it", "as", "at", "by",
    "do", "be", "on",
];

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Normalise a token: lowercase and strip leading/trailing punctuation.
fn normalise_token(token: &str) -> String {
    token
        .to_lowercase()
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

/// Extract up to `n` informative expansion terms from `text` that are not
/// already present in `query`.
///
/// Algorithm:
/// 1. Tokenise `text` on whitespace.
/// 2. Normalise each token (strip punctuation, lowercase).
/// 3. Remove stop words (short words ≤ 3 chars plus common English function words).
/// 4. Remove tokens already present in `query` (case-insensitive).
/// 5. Sort by token length descending (length proxy for informativeness).
/// 6. Return the top-`n` unique tokens.
pub(crate) fn extract_expansion_terms(text: &str, query: &str, n: usize) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }

    let query_tokens: HashSet<String> = query
        .split_whitespace()
        .map(normalise_token)
        .filter(|t| !t.is_empty())
        .collect();

    let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    let mut seen: HashSet<String> = HashSet::new();
    let mut candidates: Vec<String> = Vec::new();

    for raw in text.split_whitespace() {
        let norm = normalise_token(raw);
        if norm.is_empty() {
            continue;
        }
        // Drop stop words (by exact match and by length ≤ 3)
        if norm.len() <= 3 || stop_set.contains(norm.as_str()) {
            continue;
        }
        // Drop tokens already in the query
        if query_tokens.contains(&norm) {
            continue;
        }
        // Dedup
        if seen.contains(&norm) {
            continue;
        }
        seen.insert(norm.clone());
        candidates.push(norm);
    }

    // Sort by descending length (longer = more informative heuristic)
    candidates.sort_by_key(|b: &String| Reverse(b.len()));
    candidates.truncate(n);
    candidates
}

/// Build a draft answer from `results` by selecting sentences that contain
/// at least one token from `query`.
///
/// Takes the top-3 results and extracts relevant sentences, joining them
/// up to 300 characters total.  Falls back to the first 100 characters of
/// the best result's content when no sentences match.
pub(crate) fn build_draft(results: &[SearchResult], query: &str) -> String {
    if results.is_empty() {
        return String::new();
    }

    let query_tokens: HashSet<String> = query
        .split_whitespace()
        .map(normalise_token)
        .filter(|t| !t.is_empty())
        .collect();

    let top_results = results.iter().take(3);
    let mut sentences: Vec<String> = Vec::new();
    let mut total_len = 0usize;

    'outer: for result in top_results {
        for sentence in split_on_sentence_boundary(&result.document.content) {
            let sentence_norm = sentence.to_lowercase();
            let hit = query_tokens
                .iter()
                .any(|tok| sentence_norm.contains(tok.as_str()));
            if hit {
                let s = sentence.trim().to_string();
                if !s.is_empty() {
                    total_len += s.len();
                    sentences.push(s);
                    if total_len >= 300 {
                        break 'outer;
                    }
                }
            }
        }
    }

    if sentences.is_empty() {
        // Fallback: first 100 chars of the best result
        let best = &results[0].document.content;
        return best.chars().take(100).collect();
    }

    let joined = sentences.join(" ");
    if joined.len() > 300 {
        joined.chars().take(300).collect()
    } else {
        joined
    }
}

/// Split `text` into sentences using common sentence-ending patterns.
fn split_on_sentence_boundary(text: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let len = bytes.len();

    let mut i = 0usize;
    while i < len {
        let ch = bytes[i];
        let is_end = ch == b'.' || ch == b'!' || ch == b'?';
        if is_end {
            // Check next char
            let next_is_space = i + 1 < len && (bytes[i + 1] == b' ' || bytes[i + 1] == b'\n');
            let at_end = i + 1 == len;
            if next_is_space || at_end {
                let slice = text[start..=i].trim();
                if !slice.is_empty() {
                    parts.push(slice);
                }
                start = i + 2;
                i += 2;
                continue;
            }
        }
        if ch == b'\n' {
            let slice = text[start..i].trim();
            if !slice.is_empty() {
                parts.push(slice);
            }
            start = i + 1;
        }
        i += 1;
    }
    // Remaining
    if start < len {
        let slice = text[start..].trim();
        if !slice.is_empty() {
            parts.push(slice);
        }
    }
    parts
}

// ── IterativeRagEngine ────────────────────────────────────────────────────────

/// Engine that implements ITER-RETGEN: iterative retrieval, draft expansion,
/// and final answer synthesis.
///
/// Each iteration:
/// 1. Expands the original query with terms extracted from the previous draft.
/// 2. Retrieves documents from the [`Echo`] layer.
/// 3. Builds a draft answer from the retrieved results.
/// 4. Extracts expansion terms for the next iteration.
///
/// Iteration stops early when no new expansion terms can be found, or when
/// `config.max_iterations` is reached.
pub struct IterativeRagEngine {
    config: IterativeConfig,
}

impl IterativeRagEngine {
    /// Create a new [`IterativeRagEngine`] with the supplied configuration.
    #[must_use]
    pub fn new(config: IterativeConfig) -> Self {
        Self { config }
    }

    /// Run the iterative RAG loop for `query` using `echo` as the retrieval
    /// backend.
    ///
    /// # Errors
    ///
    /// Returns [`IterativeRagError::EmptyQuery`] when `query` is blank, or
    /// [`IterativeRagError::RetrievalFailed`] when the underlying retrieval
    /// call fails.
    pub async fn run<E>(&self, query: &str, echo: &E) -> Result<IterativeOutput, IterativeRagError>
    where
        E: Echo + ?Sized,
    {
        let query = query.trim();
        if query.is_empty() {
            return Err(IterativeRagError::EmptyQuery);
        }

        let mut all_results: Vec<SearchResult> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();
        let mut steps: Vec<IterationStep> = Vec::new();

        // Initial retrieval with the raw query
        let initial = echo
            .search(query, self.config.top_k, None)
            .await
            .map_err(|e| IterativeRagError::RetrievalFailed(e.to_string()))?;

        for r in &initial {
            let id = r.document.id.as_str().to_string();
            if seen_ids.insert(id) {
                all_results.push(r.clone());
            }
        }

        // Bootstrap draft and expansion from the initial retrieval
        let mut draft = build_draft(&initial, query);
        let mut expansion_terms =
            extract_expansion_terms(&draft, query, self.config.expansion_terms);

        steps.push(IterationStep {
            iteration: 0,
            expanded_query: query.to_string(),
            retrieved_count: initial.len(),
            draft: draft.clone(),
        });

        for iter_idx in 1..self.config.max_iterations {
            if expansion_terms.is_empty() {
                break;
            }

            // Build expanded query: original query + expansion terms
            let joined_terms = expansion_terms.join(" ");
            let expanded_query = format!("{query} {joined_terms}");

            let iter_results = echo
                .search(&expanded_query, self.config.top_k, None)
                .await
                .map_err(|e| IterativeRagError::RetrievalFailed(e.to_string()))?;

            let retrieved_count = iter_results.len();

            // Dedup-extend all_results
            for r in &iter_results {
                let id = r.document.id.as_str().to_string();
                if seen_ids.insert(id) {
                    all_results.push(r.clone());
                }
            }

            draft = build_draft(&iter_results, query);
            expansion_terms = extract_expansion_terms(&draft, query, self.config.expansion_terms);

            steps.push(IterationStep {
                iteration: iter_idx,
                expanded_query,
                retrieved_count,
                draft: draft.clone(),
            });
        }

        let final_answer = if all_results.is_empty() {
            "No relevant information found.".to_string()
        } else {
            let fa = build_draft(&all_results, query);
            if fa.is_empty() {
                "No relevant information found.".to_string()
            } else {
                fa
            }
        };

        let total_docs_retrieved = all_results.len();

        Ok(IterativeOutput {
            steps,
            final_answer,
            total_docs_retrieved,
        })
    }
}

impl Default for IterativeRagEngine {
    fn default() -> Self {
        Self::new(IterativeConfig::default())
    }
}
