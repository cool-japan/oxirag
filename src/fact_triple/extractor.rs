//! Heuristic SVO (subject–verb–object) extraction from plain text.
//!
//! The pipeline is:
//!
//! 1. [`split_sentences`] — split text on sentence-ending punctuation.
//! 2. [`detect_verb`] — locate the first verb-like token in a token list.
//! 3. [`extract_from_sentence`] — produce raw `(subject, predicate, object,
//!    confidence)` tuples from a single sentence.
//! 4. [`TripleExtractor::extract`] / [`TripleExtractor::extract_store`] — drive
//!    the full pipeline with confidence filtering, per-sentence caps, and
//!    deduplication.

use super::types::{Triple, TripleExtractor, TripleStore};

// ── sentence splitter ─────────────────────────────────────────────────────────

/// Split `text` into sentence-like spans on `. `, `! `, and `? ` boundaries,
/// as well as trailing terminal punctuation at the end of the string.
///
/// The returned slices point into `text`; no allocation is needed.
#[must_use]
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences: Vec<&str> = Vec::new();
    let mut start = 0_usize;

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0_usize;

    while i < len {
        let ch = bytes[i] as char;
        if (ch == '.' || ch == '!' || ch == '?') && i + 1 < len && bytes[i + 1] == b' ' {
            let span = text[start..=i].trim();
            if !span.is_empty() {
                sentences.push(span);
            }
            start = i + 2;
            i += 2;
        } else {
            i += 1;
        }
    }

    // Remainder after last boundary.
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    sentences
}

// ── verb vocabulary ───────────────────────────────────────────────────────────

/// Common English verbs recognised as high-confidence predicate markers.
pub const COMMON_VERBS: &[&str] = &[
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "uses",
    "used",
    "creates",
    "created",
    "supports",
    "contains",
    "includes",
    "requires",
    "produces",
    "manages",
    "provides",
    "implements",
    "belongs",
    "relates",
    "connects",
    "depends",
];

// ── verb detection ────────────────────────────────────────────────────────────

/// Scan `tokens` for the first verb-like token.
///
/// A token is considered verb-like when it appears in [`COMMON_VERBS`] *or*
/// when it has at least 4 characters **and** ends in `"s"`, `"ed"`, or
/// `"ing"` (heuristic morphological endings).
///
/// Returns `(index, verb_token)` on success, or `None` when no verb is found.
#[must_use]
pub fn detect_verb<'a>(tokens: &'a [&str]) -> Option<(usize, &'a str)> {
    for (idx, &tok) in tokens.iter().enumerate() {
        let lower = tok.to_lowercase();
        if COMMON_VERBS.contains(&lower.as_str()) {
            return Some((idx, tok));
        }
        // Heuristic: token ≥ 4 chars ending in verb-like suffix.
        if lower.len() >= 4
            && (lower.ends_with('s') || lower.ends_with("ed") || lower.ends_with("ing"))
        {
            return Some((idx, tok));
        }
    }
    None
}

// ── single-sentence extraction ────────────────────────────────────────────────

/// Strip leading/trailing punctuation from a token.
fn strip_punct(s: &str) -> &str {
    let start = s.find(|c: char| c.is_alphanumeric()).unwrap_or(0);
    let end = s
        .rfind(|c: char| c.is_alphanumeric())
        .map_or(s.len(), |p| p + 1);
    if start < end { &s[start..end] } else { s }
}

/// Extract up to `max_triples` raw `(subject, predicate, object, confidence)`
/// tuples from a single `sentence`.
///
/// - Tokens are whitespace-split and punctuation-stripped.
/// - [`detect_verb`] locates the pivot.
/// - Subject = last 1–3 tokens before the verb; Object = first 1–3 tokens after.
/// - Confidence: `0.7` for known verbs, `0.5` for heuristic endings.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn extract_from_sentence(
    sentence: &str,
    max_triples: usize,
) -> Vec<(String, String, String, f32)> {
    let raw_tokens: Vec<&str> = sentence.split_whitespace().collect();
    let tokens: Vec<&str> = raw_tokens.iter().map(|t| strip_punct(t)).collect();

    let Some((verb_idx, raw_verb)) = detect_verb(&tokens) else {
        return Vec::new();
    };

    // Determine confidence.
    let verb_lower = raw_verb.to_lowercase();
    let confidence = if COMMON_VERBS.contains(&verb_lower.as_str()) {
        0.7_f32
    } else {
        0.5_f32
    };

    // Subject: up to 3 tokens before verb.
    let subj_start = verb_idx.saturating_sub(3);
    let subject_tokens: Vec<&str> = tokens[subj_start..verb_idx]
        .iter()
        .filter(|t| !t.is_empty())
        .copied()
        .collect();

    // Object: up to 3 tokens after verb.
    let obj_end = (verb_idx + 1 + 3).min(tokens.len());
    let object_tokens: Vec<&str> = tokens[verb_idx + 1..obj_end]
        .iter()
        .filter(|t| !t.is_empty())
        .copied()
        .collect();

    if subject_tokens.is_empty() || object_tokens.is_empty() {
        return Vec::new();
    }

    let subject = subject_tokens.join(" ");
    let predicate = verb_lower;
    let object = object_tokens.join(" ");

    // Only yield if both subject and object are non-empty strings.
    if subject.trim().is_empty() || object.trim().is_empty() {
        return Vec::new();
    }

    let result = vec![(subject, predicate, object, confidence)];
    result.into_iter().take(max_triples).collect()
}

// ── TripleExtractor methods ───────────────────────────────────────────────────

impl TripleExtractor {
    /// Extract triples from `text`, optionally tagging each with `source_id`.
    ///
    /// The process:
    /// 1. Split text into sentences.
    /// 2. Extract raw tuples per sentence, capped by `config.max_per_sentence`.
    /// 3. Filter by `config.min_confidence`.
    /// 4. Deduplicate on `(subject, predicate, object)`.
    /// 5. Return the surviving [`Triple`]s.
    #[must_use]
    pub fn extract(&self, text: &str, source_id: Option<&str>) -> Vec<Triple> {
        if text.trim().is_empty() {
            return Vec::new();
        }

        let sentences = split_sentences(text);
        let mut seen: std::collections::HashSet<(String, String, String)> =
            std::collections::HashSet::new();
        let mut out: Vec<Triple> = Vec::new();

        for sentence in sentences {
            let raw = extract_from_sentence(sentence, self.config.max_per_sentence);
            for (s, p, o, conf) in raw {
                if conf < self.config.min_confidence {
                    continue;
                }
                let key = (s.clone(), p.clone(), o.clone());
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key);
                let mut triple = Triple::new(s, p, o, conf);
                if let Some(sid) = source_id {
                    triple = triple.with_source(sid);
                }
                out.push(triple);
            }
        }

        out
    }

    /// Extract triples and return them wrapped in a [`TripleStore`].
    #[must_use]
    pub fn extract_store(&self, text: &str, source_id: Option<&str>) -> TripleStore {
        let triples = self.extract(text, source_id);
        let mut store = TripleStore::new();
        for t in triples {
            store.add(t);
        }
        store
    }
}
