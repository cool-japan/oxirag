//! Knowledge-strip decomposition, filtering, and recomposition.
//!
//! A *knowledge strip* is a single sentence extracted from a retrieved
//! document.  The [`KnowledgeRefiner`] decomposes documents into strips,
//! scores each strip via lexical Jaccard against the query, and flags strips
//! that meet the relevance threshold.  Kept strips are then recomposed into
//! new [`SearchResult`] objects that can be re-ranked by the CRAG engine.
//!
//! The [`QueryRefiner`] uses the vocabulary of kept strips to produce an
//! enriched rewrite of the original query.

use std::collections::{HashMap, HashSet};

use crate::types::{Document, DocumentId, SearchResult};

use super::types::{CragConfig, KnowledgeStrip};

// ── sentence splitter ─────────────────────────────────────────────────────────

/// Split `text` into individual sentences.
///
/// Sentence boundaries are detected at `. `, `? `, `! `, and `\n\n`.
/// Each fragment is trimmed; empty fragments are discarded.
///
/// The terminal punctuation mark (`.`, `?`, `!`) is retained with the preceding
/// sentence.  When `text` has no boundary sequences it is returned as a single
/// sentence (if non-empty after trimming).
fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // Paragraph break → flush current buffer.
        if c == '\n' && i + 1 < len && chars[i + 1] == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
            i += 2;
            continue;
        }

        // `. `, `? `, `! ` → keep the punctuation, flush on the space.
        if matches!(c, '.' | '?' | '!') && i + 1 < len && chars[i + 1] == ' ' {
            current.push(c);
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
            i += 2; // skip the space
            continue;
        }

        current.push(c);
        i += 1;
    }

    // Flush any remaining content.
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

// ── tokenise helper ───────────────────────────────────────────────────────────

/// Tokenise `text` for Jaccard relevance scoring.
///
/// Splits on every non-alphanumeric character, lowercases each token, and
/// filters tokens shorter than 2 characters.
fn tokenize_text(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() >= 2)
        .collect()
}

/// Compute the Jaccard coefficient between the token sets of `a` and `b`.
fn jaccard_relevance(a: &str, b: &str) -> f32 {
    let set_a: HashSet<String> = tokenize_text(a).into_iter().collect();
    let set_b: HashSet<String> = tokenize_text(b).into_iter().collect();

    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)]
    let score = intersection as f32 / union as f32;
    score
}

// ── KnowledgeRefiner ──────────────────────────────────────────────────────────

/// Decomposes retrieved documents into sentence-level strips, filters by
/// relevance, and recomposes surviving strips back into search results.
#[derive(Debug, Clone)]
pub struct KnowledgeRefiner;

impl KnowledgeRefiner {
    /// Create a new [`KnowledgeRefiner`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Decompose a single `doc` into sentence-level [`KnowledgeStrip`] values.
    ///
    /// Each strip is scored via lexical Jaccard against `query`.  Strips whose
    /// score meets `cfg.strip_relevance_threshold` have `kept = true`.
    #[must_use]
    pub fn decompose(
        &self,
        query: &str,
        doc: &SearchResult,
        cfg: &CragConfig,
    ) -> Vec<KnowledgeStrip> {
        let sentences = split_into_sentences(&doc.document.content);

        sentences
            .into_iter()
            .map(|text| {
                let relevance = jaccard_relevance(query, &text);
                let kept = relevance >= cfg.strip_relevance_threshold;
                KnowledgeStrip {
                    source_id: doc.document.id.clone(),
                    text,
                    relevance,
                    kept,
                }
            })
            .collect()
    }

    /// Refine a collection of `docs` into knowledge strips.
    ///
    /// Calls [`Self::decompose`] on each document and returns only the strips
    /// with `kept = true`.
    #[must_use]
    pub fn refine(
        &self,
        query: &str,
        docs: &[SearchResult],
        cfg: &CragConfig,
    ) -> Vec<KnowledgeStrip> {
        docs.iter()
            .flat_map(|doc| self.decompose(query, doc, cfg))
            .filter(|s| s.kept)
            .collect()
    }

    /// Recompose kept strips back into [`SearchResult`] objects.
    ///
    /// Groups strips by `source_id`, joins the text of kept strips with a
    /// space, and uses the *average* of the kept-strip relevance scores as
    /// the result score.  The original [`DocumentId`] is preserved.
    ///
    /// Documents for which all strips were filtered out are excluded from the
    /// output.
    #[must_use]
    pub fn recompose(&self, strips: &[KnowledgeStrip]) -> Vec<SearchResult> {
        // Only consider kept strips.
        let kept: Vec<&KnowledgeStrip> = strips.iter().filter(|s| s.kept).collect();

        if kept.is_empty() {
            return Vec::new();
        }

        // Group by source_id preserving insertion order.
        let mut order: Vec<String> = Vec::new();
        let mut groups: HashMap<String, Vec<&KnowledgeStrip>> = HashMap::new();
        for strip in &kept {
            let key = strip.source_id.as_str().to_string();
            groups.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                Vec::new()
            });
            if let Some(v) = groups.get_mut(&key) {
                v.push(strip);
            }
        }

        order
            .into_iter()
            .enumerate()
            .map(|(rank, key)| {
                let group = &groups[&key];
                let joined = group
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                #[allow(clippy::cast_precision_loss)]
                let avg_score = group.iter().map(|s| s.relevance).sum::<f32>() / group.len() as f32;

                let source_id = DocumentId::from(key.as_str());
                let doc = Document::new(joined).with_id(source_id);
                SearchResult::new(doc, avg_score, rank)
            })
            .collect()
    }
}

impl Default for KnowledgeRefiner {
    fn default() -> Self {
        Self::new()
    }
}

// ── QueryRefiner ──────────────────────────────────────────────────────────────

/// Rewrites the current query by appending salient terms from kept strips.
///
/// Salient terms are tokens that:
///
/// 1. Appear in more than one kept strip (cross-strip frequency > 1).
/// 2. Are **not** already present in the original query.
///
/// If no salient terms are found, or the resulting rewrite would be empty,
/// the original query is returned unchanged.
#[derive(Debug, Clone)]
pub struct QueryRefiner;

impl QueryRefiner {
    /// Create a new [`QueryRefiner`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Rewrite `query` by appending salient vocabulary from `kept_strips`.
    ///
    /// Returns the original `query` unchanged when no enrichment is possible.
    #[must_use]
    pub fn rewrite(&self, query: &str, kept_strips: &[KnowledgeStrip]) -> String {
        if kept_strips.is_empty() {
            return query.to_string();
        }

        let query_tokens: HashSet<String> = tokenize_text(query).into_iter().collect();

        // Count how many strips each token appears in.
        let mut strip_freq: HashMap<String, usize> = HashMap::new();
        for strip in kept_strips {
            let tokens: HashSet<String> = tokenize_text(&strip.text).into_iter().collect();
            for token in tokens {
                *strip_freq.entry(token).or_insert(0) += 1;
            }
        }

        // Collect salient terms: multi-strip AND not already in query.
        let mut salient: Vec<String> = strip_freq
            .into_iter()
            .filter(|(token, freq)| *freq > 1 && !query_tokens.contains(token))
            .map(|(token, _)| token)
            .collect();

        if salient.is_empty() {
            return query.to_string();
        }

        // Sort for deterministic output.
        salient.sort();

        let rewrite = format!("{} {}", query.trim(), salient.join(" "));

        // Paranoia guard: never return an empty string.
        if rewrite.trim().is_empty() {
            query.to_string()
        } else {
            rewrite
        }
    }
}

impl Default for QueryRefiner {
    fn default() -> Self {
        Self::new()
    }
}
