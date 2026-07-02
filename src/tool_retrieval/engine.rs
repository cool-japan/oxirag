//! [`ToolRetrievalIndex`], [`ArgumentGrounder`], and [`ToolRetrievalEngine`]:
//! deterministic FNV-1a spec indexing, semantic + lexical matching, and
//! independent per-parameter argument grounding.
//!
//! See the [module documentation](crate::tool_retrieval) for how this
//! differs from `agentic`'s name-substring tool selection.

use std::cmp::Ordering;
use std::collections::HashSet;

use super::types::{
    ArgumentGroundingStatus, GroundedArgument, ToolMatch, ToolParameter, ToolParameterType,
    ToolRetrievalConfig, ToolRetrievalError, ToolRetrievalResult, ToolSpecEntry,
};

// ── FNV-1a pseudo-embeddings & lexical overlap ───────────────────────────────
//
// A private, self-contained lexical toolkit (not shared with any other
// module's private helpers, following this crate's convention — see e.g.
// `searchain::engine`'s equivalent header comment).

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 1_099_511_628_211;

/// FNV-1a 64-bit hash of `bytes`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Split `text` into lowercase alphanumeric tokens of length `>= 2`.
fn tokenize_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// L2-normalise `vector` in place. A zero (or near-zero) vector is left
/// unchanged rather than divided by (approximately) zero.
fn l2_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}

/// Deterministic FNV-1a token-bucket-histogram pseudo-embedding of `text`
/// into `dim` dimensions, L2-normalised. A `dim` of `0` returns an empty
/// vector.
fn embed_text(text: &str, dim: usize) -> Vec<f32> {
    let mut buckets = vec![0.0f32; dim];
    if dim == 0 {
        return buckets;
    }
    for token in tokenize_words(text) {
        let hash = fnv1a(token.as_bytes());
        #[allow(clippy::cast_possible_truncation)]
        let idx = (hash as usize) % dim;
        buckets[idx] += 1.0;
    }
    l2_normalize(&mut buckets);
    buckets
}

/// Cosine similarity between two equal-length embeddings. Since every
/// [`embed_text`] output is a non-negative, L2-normalised histogram, the
/// result is always within `[0.0, 1.0]` (clamped to absorb float drift).
/// Mismatched-length or empty inputs score `0.0`.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(0.0, 1.0)
}

/// Stopwords excluded from the *content* vocabulary used for lexical overlap.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
    "that", "this", "with", "from", "they", "have", "were", "what", "your", "when", "them", "then",
    "than", "into", "some", "such", "only", "also", "been", "more", "very", "will", "would",
    "there", "their", "which", "about", "could", "these", "those", "does", "please",
];

/// Return `true` when `word` is a content word: at least three characters and
/// not a stopword.
fn is_content_word(word: &str) -> bool {
    word.chars().count() >= 3 && !STOPWORDS.contains(&word)
}

/// The distinct content-word vocabulary of `text`.
fn content_terms(text: &str) -> HashSet<String> {
    tokenize_words(text)
        .into_iter()
        .filter(|w| is_content_word(w))
        .collect()
}

/// Jaccard overlap (`|intersection| / |union|`) between two content-word
/// vocabularies. Either side empty scores `0.0`.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    if intersection == 0 {
        return 0.0;
    }
    let union = a.union(b).count();
    #[allow(clippy::cast_precision_loss)]
    let ratio = intersection as f32 / union as f32;
    ratio
}

/// Build the text a [`ToolSpecEntry`] is embedded from: its name, its
/// description, and every parameter's name and description — **not just the
/// name** (the key difference from `agentic`'s name-substring selection; see
/// the [module documentation](crate::tool_retrieval)).
fn spec_embedding_text(spec: &ToolSpecEntry) -> String {
    let mut parts = Vec::with_capacity(2 + spec.parameters.len() * 2);
    parts.push(spec.name.as_str());
    parts.push(spec.description.as_str());
    for param in &spec.parameters {
        parts.push(param.name.as_str());
        parts.push(param.description.as_str());
    }
    parts.join(" ")
}

// ── ToolRetrievalIndex ────────────────────────────────────────────────────────

/// One indexed [`ToolSpecEntry`] together with its precomputed pseudo-
/// embedding and content-word vocabulary.
#[derive(Debug, Clone)]
struct IndexedToolEntry {
    /// The original spec.
    spec: ToolSpecEntry,
    /// FNV-1a pseudo-embedding of [`spec_embedding_text`].
    embedding: Vec<f32>,
    /// Content-word vocabulary of [`spec_embedding_text`].
    lexical_terms: HashSet<String>,
}

/// An index of [`ToolSpecEntry`] specs, each embedded via a deterministic
/// FNV-1a pseudo-embedding of its combined name, description, and parameter
/// names/descriptions.
///
/// Built once via [`ToolRetrievalIndex::build`]; [`ToolRetrievalEngine`]
/// wraps an index to provide [`ToolRetrievalEngine::retrieve`] and
/// [`ToolRetrievalEngine::ground`].
#[derive(Debug, Clone)]
pub struct ToolRetrievalIndex {
    /// The configuration this index was built with.
    config: ToolRetrievalConfig,
    /// Indexed tools, in the order they were supplied to
    /// [`ToolRetrievalIndex::build`].
    entries: Vec<IndexedToolEntry>,
}

impl ToolRetrievalIndex {
    /// Embed and index every spec in `tools`.
    ///
    /// # Errors
    ///
    /// - [`ToolRetrievalError::InvalidEmbeddingDim`],
    ///   [`ToolRetrievalError::InvalidDescriptionWeight`], or
    ///   [`ToolRetrievalError::InvalidTopK`] when `config` fails
    ///   [`ToolRetrievalConfig::validate`].
    /// - [`ToolRetrievalError::EmptyRegistry`] when `tools` is empty.
    /// - [`ToolRetrievalError::EmptyToolName`] when a spec's name is empty
    ///   (after trimming).
    /// - [`ToolRetrievalError::DuplicateToolName`] when two specs share the
    ///   same name (case-insensitive).
    pub fn build(
        tools: Vec<ToolSpecEntry>,
        config: ToolRetrievalConfig,
    ) -> ToolRetrievalResult<Self> {
        config.validate()?;
        if tools.is_empty() {
            return Err(ToolRetrievalError::EmptyRegistry);
        }

        let mut seen_names: HashSet<String> = HashSet::with_capacity(tools.len());
        let mut entries = Vec::with_capacity(tools.len());
        for spec in tools {
            let trimmed = spec.name.trim();
            if trimmed.is_empty() {
                return Err(ToolRetrievalError::EmptyToolName);
            }
            if !seen_names.insert(trimmed.to_lowercase()) {
                return Err(ToolRetrievalError::DuplicateToolName(spec.name.clone()));
            }

            let text = spec_embedding_text(&spec);
            let embedding = embed_text(&text, config.embedding_dim);
            let lexical_terms = content_terms(&text);
            entries.push(IndexedToolEntry {
                spec,
                embedding,
                lexical_terms,
            });
        }

        Ok(Self { config, entries })
    }

    /// Number of indexed tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the index holds no tools.
    ///
    /// Always `false` for a successfully [`ToolRetrievalIndex::build`]-ed
    /// index, since an empty registry is rejected at build time; provided for
    /// API symmetry with [`ToolRetrievalIndex::len`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Names of every indexed tool, in index order.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.spec.name.as_str()).collect()
    }

    /// Look up an indexed tool's spec by exact (case-sensitive) name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ToolSpecEntry> {
        self.entries
            .iter()
            .find(|e| e.spec.name == name)
            .map(|e| &e.spec)
    }

    /// Borrow this index's configuration.
    #[must_use]
    pub fn config(&self) -> &ToolRetrievalConfig {
        &self.config
    }
}

// ── Argument grounding: span-preserving tokeniser ────────────────────────────

/// A word token together with its byte span in the original query.
#[derive(Debug, Clone)]
struct SpanToken {
    /// Lowercased token text.
    lower: String,
    /// Original-cased token text.
    raw: String,
    /// Byte offset of the token's start.
    start: usize,
    /// Byte offset (exclusive) of the token's end.
    end: usize,
}

/// Split `text` into alphanumeric-run tokens, discarding everything else,
/// preserving each token's byte span in `text`. A private, self-contained
/// copy of the tokenisation strategy `self_query::parser` also uses
/// internally — not shared, per this crate's per-module lexical-toolkit
/// convention.
fn tokenize_spans(text: &str) -> Vec<SpanToken> {
    let mut tokens = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            tokens.push(SpanToken {
                lower: text[s..idx].to_lowercase(),
                raw: text[s..idx].to_string(),
                start: s,
                end: idx,
            });
        }
    }
    if let Some(s) = start.take() {
        tokens.push(SpanToken {
            lower: text[s..].to_lowercase(),
            raw: text[s..].to_string(),
            start: s,
            end: text.len(),
        });
    }
    tokens
}

/// A byte span (`start..end`) within a query, covering one or more tokens.
#[derive(Debug, Clone, Copy)]
struct TokenSpan {
    /// Byte offset of the span's start.
    start: usize,
    /// Byte offset (exclusive) of the span's end.
    end: usize,
}

/// If `tokens[start_idx..start_idx + words.len()]` lowercase-matches `words`
/// exactly, in order, return the byte span it covers.
fn match_word_run(tokens: &[SpanToken], start_idx: usize, words: &[String]) -> Option<TokenSpan> {
    if words.is_empty() || start_idx.saturating_add(words.len()) > tokens.len() {
        return None;
    }
    for (offset, word) in words.iter().enumerate() {
        if tokens[start_idx + offset].lower != *word {
            return None;
        }
    }
    Some(TokenSpan {
        start: tokens[start_idx].start,
        end: tokens[start_idx + words.len() - 1].end,
    })
}

// ── Argument grounding: parameter-name anchor ────────────────────────────────

/// The location of a parameter-name mention inside a tokenised query.
#[derive(Debug, Clone, Copy)]
struct AnchorMatch {
    /// Index of the first token of the matched name phrase.
    start_token_idx: usize,
    /// Index one past the last token of the matched name phrase.
    end_token_idx: usize,
    /// Byte offset one past the last character of the matched name phrase.
    end_byte: usize,
}

/// Search `tokens` for the first occurrence of `param_name`'s own word
/// sequence. Since both the parameter name and the query are tokenised on
/// non-alphanumeric boundaries, a parameter named `"max_results"` matches a
/// query mention of `"max results"` or `"max-results"` just as well as a
/// literal `"max_results"`.
fn find_anchor(tokens: &[SpanToken], param_name: &str) -> Option<AnchorMatch> {
    let words: Vec<String> = tokenize_spans(param_name)
        .into_iter()
        .map(|t| t.lower)
        .collect();
    if words.is_empty() {
        return None;
    }
    (0..tokens.len()).find_map(|i| {
        match_word_run(tokens, i, &words).map(|span| AnchorMatch {
            start_token_idx: i,
            end_token_idx: i + words.len(),
            end_byte: span.end,
        })
    })
}

/// Copulas/prepositions skipped when looking for the value immediately
/// following a parameter-name anchor — they carry no value information of
/// their own (e.g. `"city is Paris"`, `"city: Paris"` both resolve to the
/// same anchored value `"Paris"`).
const CONNECTOR_WORDS: &[&str] = &[
    "is", "was", "of", "as", "to", "the", "a", "an", "be", "equal", "equals",
];

/// Advance `idx` past up to two leading connector words.
fn skip_connectors(tokens: &[SpanToken], mut idx: usize) -> usize {
    let mut hops = 0;
    while hops < 2 && idx < tokens.len() && CONNECTOR_WORDS.contains(&tokens[idx].lower.as_str()) {
        idx += 1;
        hops += 1;
    }
    idx
}

// ── Argument grounding: String-type helpers ──────────────────────────────────

/// Maximum number of consecutive capitalised tokens folded into a single
/// proper-noun-like span.
const MAX_CAPITALIZED_SPAN_TOKENS: usize = 4;

/// Return `true` when `raw`'s first character is uppercase.
fn is_capitalized(raw: &str) -> bool {
    raw.chars().next().is_some_and(char::is_uppercase)
}

/// If `tokens[start_idx]` is capitalised, extend forward through any
/// immediately-following capitalised tokens (bounded by
/// [`MAX_CAPITALIZED_SPAN_TOKENS`]) and return the covering span.
fn capitalized_run(tokens: &[SpanToken], start_idx: usize) -> Option<TokenSpan> {
    let first = tokens.get(start_idx)?;
    if !is_capitalized(&first.raw) {
        return None;
    }
    let mut end_idx = start_idx;
    while end_idx + 1 < tokens.len()
        && end_idx + 1 - start_idx < MAX_CAPITALIZED_SPAN_TOKENS
        && is_capitalized(&tokens[end_idx + 1].raw)
    {
        end_idx += 1;
    }
    Some(TokenSpan {
        start: first.start,
        end: tokens[end_idx].end,
    })
}

/// Find the first capitalised run anywhere in `tokens`.
fn capitalized_run_anywhere(tokens: &[SpanToken]) -> Option<TokenSpan> {
    (0..tokens.len()).find_map(|i| capitalized_run(tokens, i))
}

/// The span of a single token at `idx`, if any.
fn single_token_span(tokens: &[SpanToken], idx: usize) -> Option<TokenSpan> {
    tokens.get(idx).map(|t| TokenSpan {
        start: t.start,
        end: t.end,
    })
}

/// Scan `text` for `quote`-delimited spans, returning the *inner* byte range
/// (excluding the quote characters) of each non-empty pair found, in order.
fn scan_quoted(text: &str, quote: char) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].1 == quote {
            if let Some(j) = ((i + 1)..chars.len()).find(|&k| chars[k].1 == quote) {
                let inner_start = chars[i].0 + chars[i].1.len_utf8();
                let inner_end = chars[j].0;
                if inner_end > inner_start {
                    spans.push((inner_start, inner_end));
                }
                i = j + 1;
                continue;
            }
            break;
        }
        i += 1;
    }
    spans
}

/// Find every quoted span in `text`. Double-quoted spans take priority;
/// single-quoted spans are only considered when no double-quoted span exists
/// anywhere in `text`, so a contraction like `"don't"` is not mistaken for an
/// open quote when the query genuinely uses double quotes elsewhere.
fn find_quoted_spans(text: &str) -> Vec<(usize, usize)> {
    let double = scan_quoted(text, '"');
    if !double.is_empty() {
        return double;
    }
    scan_quoted(text, '\'')
}

// ── Argument grounding: Number-type helpers ──────────────────────────────────

/// A numeric token with its byte span in the original query.
struct NumericToken {
    /// The verbatim numeric text (e.g. `"3.5"`, `"-7"`).
    text: String,
    /// Byte offset of the token's start.
    start: usize,
}

/// Scan `text` for numeric runs (digits, an optional single leading `-`, and
/// internal `.`), keeping only the runs that parse as `f64`.
fn extract_numeric_tokens(text: &str) -> Vec<NumericToken> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (start, ch) = chars[i];
        let starts_number = ch.is_ascii_digit()
            || (ch == '-' && chars.get(i + 1).is_some_and(|&(_, c)| c.is_ascii_digit()));
        if starts_number {
            let mut j = i + 1;
            while j < chars.len() && (chars[j].1.is_ascii_digit() || chars[j].1 == '.') {
                j += 1;
            }
            let end = chars.get(j).map_or(text.len(), |&(pos, _)| pos);
            let text_slice = &text[start..end];
            if text_slice.parse::<f64>().is_ok() {
                tokens.push(NumericToken {
                    text: text_slice.to_string(),
                    start,
                });
            }
            i = j;
        } else {
            i += 1;
        }
    }
    tokens
}

// ── Argument grounding: Boolean-type helpers ─────────────────────────────────

/// Exact boolean cues.
const BOOLEAN_STRONG_POSITIVE: &[&str] = &["true", "yes"];
/// Exact boolean cues.
const BOOLEAN_STRONG_NEGATIVE: &[&str] = &["false", "no"];
/// Softer boolean synonyms.
const BOOLEAN_WEAK_POSITIVE: &[&str] = &["enable", "enabled", "on", "affirmative"];
/// Softer boolean synonyms.
const BOOLEAN_WEAK_NEGATIVE: &[&str] = &["disable", "disabled", "off", "negative"];

/// Window radius (in tokens) searched on either side of a parameter-name
/// anchor for a boolean cue.
const BOOLEAN_WINDOW: usize = 4;

/// Classify `word` as a boolean cue: `Some((polarity, is_strong))`, where
/// `is_strong` distinguishes an exact cue (`true`/`false`/`yes`/`no`) from a
/// softer synonym (`enable`/`disable`/`on`/`off`/...).
fn boolean_cue(word: &str) -> Option<(bool, bool)> {
    if BOOLEAN_STRONG_POSITIVE.contains(&word) {
        Some((true, true))
    } else if BOOLEAN_STRONG_NEGATIVE.contains(&word) {
        Some((false, true))
    } else if BOOLEAN_WEAK_POSITIVE.contains(&word) {
        Some((true, false))
    } else if BOOLEAN_WEAK_NEGATIVE.contains(&word) {
        Some((false, false))
    } else {
        None
    }
}

// ── Argument grounding: confidence tiers ─────────────────────────────────────
//
// Confidence is always higher for an anchored match (the parameter's own
// name was mentioned in the query) than an unanchored one, and always higher
// for an exact/quoted match than a heuristic fallback inference.

/// A quoted value near an explicit parameter-name mention.
const CONFIDENCE_QUOTED_ANCHORED: f32 = 0.97;
/// A quoted value with no parameter-name mention anywhere in the query.
const CONFIDENCE_QUOTED_UNANCHORED: f32 = 0.85;
/// A `String` value read as the phrase immediately following a
/// parameter-name mention.
const CONFIDENCE_STRING_ANCHORED: f32 = 0.75;
/// A `String` value inferred from a capitalised span with no parameter-name
/// mention to anchor it.
const CONFIDENCE_STRING_FALLBACK: f32 = 0.45;
/// A numeric value nearest to an explicit parameter-name mention.
const CONFIDENCE_NUMBER_ANCHORED: f32 = 0.8;
/// A numeric value taken as the first number in the query, with no
/// parameter-name mention to anchor it.
const CONFIDENCE_NUMBER_UNANCHORED: f32 = 0.4;
/// An exact boolean cue near an explicit parameter-name mention.
const CONFIDENCE_BOOLEAN_STRONG_ANCHORED: f32 = 0.9;
/// An exact boolean cue found anywhere in the query, with no parameter-name
/// mention to anchor it.
const CONFIDENCE_BOOLEAN_STRONG_UNANCHORED: f32 = 0.6;
/// A softer boolean synonym near an explicit parameter-name mention.
const CONFIDENCE_BOOLEAN_WEAK_ANCHORED: f32 = 0.65;
/// A softer boolean synonym found anywhere in the query, with no
/// parameter-name mention to anchor it.
const CONFIDENCE_BOOLEAN_WEAK_UNANCHORED: f32 = 0.35;
/// An allowed enum value nearest to an explicit parameter-name mention.
const CONFIDENCE_ENUM_ANCHORED: f32 = 0.88;
/// An allowed enum value found anywhere in the query, with no parameter-name
/// mention to anchor it.
const CONFIDENCE_ENUM_UNANCHORED: f32 = 0.6;

/// `(value, confidence, source_span_text)` — the shared outcome shape every
/// type-specific grounding helper returns, before being wrapped into a
/// [`GroundedArgument`] by [`ground_parameter`].
type GroundOutcome = Option<(String, f32, String)>;

/// Ground a `String`-typed parameter: prefer a quoted span (nearest the
/// anchor, if any), then the phrase right after an anchor, then a
/// capitalised span found anywhere as a last resort.
fn ground_string(query: &str, tokens: &[SpanToken], anchor: Option<&AnchorMatch>) -> GroundOutcome {
    let quoted = find_quoted_spans(query);
    if !quoted.is_empty() {
        let (span, anchored) = match anchor {
            Some(a) => (
                quoted
                    .iter()
                    .copied()
                    .min_by_key(|&(s, _)| s.abs_diff(a.end_byte))?,
                true,
            ),
            None => (quoted[0], false),
        };
        let text = query[span.0..span.1].to_string();
        let confidence = if anchored {
            CONFIDENCE_QUOTED_ANCHORED
        } else {
            CONFIDENCE_QUOTED_UNANCHORED
        };
        return Some((text.clone(), confidence, text));
    }

    if let Some(a) = anchor {
        let value_start = skip_connectors(tokens, a.end_token_idx);
        let span =
            capitalized_run(tokens, value_start).or_else(|| single_token_span(tokens, value_start));
        if let Some(span) = span {
            let text = query[span.start..span.end].to_string();
            return Some((text.clone(), CONFIDENCE_STRING_ANCHORED, text));
        }
    }

    capitalized_run_anywhere(tokens).map(|span| {
        let text = query[span.start..span.end].to_string();
        (text.clone(), CONFIDENCE_STRING_FALLBACK, text)
    })
}

/// Ground a `Number`-typed parameter: the numeric token nearest the anchor
/// (searched in both directions), or the first numeric token in the query
/// when the parameter is never named.
fn ground_number(query: &str, anchor: Option<&AnchorMatch>) -> GroundOutcome {
    let numbers = extract_numeric_tokens(query);
    if let Some(a) = anchor {
        let nearest = numbers
            .iter()
            .min_by_key(|n| n.start.abs_diff(a.end_byte))?;
        Some((
            nearest.text.clone(),
            CONFIDENCE_NUMBER_ANCHORED,
            nearest.text.clone(),
        ))
    } else {
        let first = numbers.first()?;
        Some((
            first.text.clone(),
            CONFIDENCE_NUMBER_UNANCHORED,
            first.text.clone(),
        ))
    }
}

/// Ground a `Boolean`-typed parameter: a cue within [`BOOLEAN_WINDOW`] tokens
/// of the anchor (nearest wins), else the first cue found anywhere in the
/// query. The grounded value is normalised to `"true"`/`"false"`; the
/// original cue word is preserved as the source span.
fn ground_boolean(tokens: &[SpanToken], anchor: Option<&AnchorMatch>) -> GroundOutcome {
    if let Some(a) = anchor {
        let window_start = a.start_token_idx.saturating_sub(BOOLEAN_WINDOW);
        let window_end = a
            .end_token_idx
            .saturating_add(BOOLEAN_WINDOW)
            .min(tokens.len());
        let mut best: Option<(usize, bool, bool)> = None;
        for (idx, tok) in tokens
            .iter()
            .enumerate()
            .skip(window_start)
            .take(window_end.saturating_sub(window_start))
        {
            if idx >= a.start_token_idx && idx < a.end_token_idx {
                continue;
            }
            if let Some((polarity, strong)) = boolean_cue(&tok.lower) {
                let distance = if idx < a.start_token_idx {
                    a.start_token_idx - idx
                } else {
                    idx - (a.end_token_idx.saturating_sub(1))
                };
                if best.is_none_or(|(best_distance, _, _)| distance < best_distance) {
                    best = Some((distance, polarity, strong));
                }
            }
        }
        if let Some((_, polarity, strong)) = best {
            let confidence = if strong {
                CONFIDENCE_BOOLEAN_STRONG_ANCHORED
            } else {
                CONFIDENCE_BOOLEAN_WEAK_ANCHORED
            };
            let text = polarity.to_string();
            return Some((text.clone(), confidence, text));
        }
    }

    for tok in tokens {
        if let Some((polarity, strong)) = boolean_cue(&tok.lower) {
            let confidence = if strong {
                CONFIDENCE_BOOLEAN_STRONG_UNANCHORED
            } else {
                CONFIDENCE_BOOLEAN_WEAK_UNANCHORED
            };
            return Some((polarity.to_string(), confidence, tok.raw.clone()));
        }
    }

    None
}

/// Ground an `Enum`-typed parameter: a case-insensitive match against
/// `allowed`, nearest the anchor when one exists, else the first match found
/// in the query. A query span that does not equal one of `allowed`
/// (case-insensitively, whole-word) never grounds this parameter.
fn ground_enum(
    query: &str,
    tokens: &[SpanToken],
    allowed: &[String],
    anchor: Option<&AnchorMatch>,
) -> GroundOutcome {
    let mut candidates: Vec<(&String, TokenSpan)> = Vec::new();
    for value in allowed {
        let words: Vec<String> = tokenize_spans(value).into_iter().map(|t| t.lower).collect();
        if words.is_empty() {
            continue;
        }
        for start_idx in 0..tokens.len() {
            if let Some(span) = match_word_run(tokens, start_idx, &words) {
                candidates.push((value, span));
            }
        }
    }
    if candidates.is_empty() {
        return None;
    }

    let chosen = match anchor {
        Some(a) => candidates
            .iter()
            .min_by_key(|(_, span)| span.start.abs_diff(a.end_byte))?,
        None => &candidates[0],
    };
    let confidence = if anchor.is_some() {
        CONFIDENCE_ENUM_ANCHORED
    } else {
        CONFIDENCE_ENUM_UNANCHORED
    };
    let source_text = query[chosen.1.start..chosen.1.end].to_string();
    Some((chosen.0.clone(), confidence, source_text))
}

/// Ground a single parameter, dispatching on its [`ToolParameterType`], and
/// wrap the outcome into a [`GroundedArgument`] — flagging (never silently
/// dropping) a required parameter with no plausible value.
fn ground_parameter(query: &str, tokens: &[SpanToken], param: &ToolParameter) -> GroundedArgument {
    let anchor = find_anchor(tokens, &param.name);

    let outcome = match &param.param_type {
        ToolParameterType::String => ground_string(query, tokens, anchor.as_ref()),
        ToolParameterType::Number => ground_number(query, anchor.as_ref()),
        ToolParameterType::Boolean => ground_boolean(tokens, anchor.as_ref()),
        ToolParameterType::Enum(allowed) => ground_enum(query, tokens, allowed, anchor.as_ref()),
    };

    match outcome {
        Some((value, confidence, span)) => GroundedArgument {
            parameter_name: param.name.clone(),
            value: Some(value),
            confidence,
            source_span: Some(span),
            status: ArgumentGroundingStatus::Grounded,
        },
        None => GroundedArgument {
            parameter_name: param.name.clone(),
            value: None,
            confidence: 0.0,
            source_span: None,
            status: if param.required {
                ArgumentGroundingStatus::UngroundedRequired
            } else {
                ArgumentGroundingStatus::UngroundedOptional
            },
        },
    }
}

// ── ArgumentGrounder ──────────────────────────────────────────────────────────

/// Extracts [`GroundedArgument`]s for a tool's parameters from a query.
///
/// Every parameter is grounded independently: the presence or absence of a
/// value for one parameter never affects another. See the
/// [module documentation](crate::tool_retrieval) for the extraction
/// heuristics per [`ToolParameterType`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ArgumentGrounder;

impl ArgumentGrounder {
    /// Create a new grounder. Stateless — grounding is a pure function of the
    /// query and the tool spec passed to [`ArgumentGrounder::ground`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Ground every parameter of `spec` against `query`.
    ///
    /// Returns exactly one [`GroundedArgument`] per parameter in
    /// `spec.parameters`, in the same order — including parameters no
    /// plausible value was found for (see [`ArgumentGroundingStatus`]).
    ///
    /// # Errors
    ///
    /// Returns [`ToolRetrievalError::EmptyQuery`] when `query` is empty or
    /// whitespace-only.
    pub fn ground(
        &self,
        query: &str,
        spec: &ToolSpecEntry,
    ) -> ToolRetrievalResult<Vec<GroundedArgument>> {
        if query.trim().is_empty() {
            return Err(ToolRetrievalError::EmptyQuery);
        }
        let tokens = tokenize_spans(query);
        Ok(spec
            .parameters
            .iter()
            .map(|param| ground_parameter(query, &tokens, param))
            .collect())
    }
}

// ── ToolRetrievalEngine ───────────────────────────────────────────────────────

/// Ties spec indexing ([`ToolRetrievalIndex`]), semantic + lexical retrieval,
/// and argument grounding ([`ArgumentGrounder`]) together behind one API.
#[derive(Debug, Clone)]
pub struct ToolRetrievalEngine {
    /// The wrapped, already-embedded index.
    index: ToolRetrievalIndex,
    /// The (stateless) argument grounder.
    grounder: ArgumentGrounder,
}

impl ToolRetrievalEngine {
    /// Wrap an already-built [`ToolRetrievalIndex`].
    #[must_use]
    pub fn new(index: ToolRetrievalIndex) -> Self {
        Self {
            index,
            grounder: ArgumentGrounder::new(),
        }
    }

    /// Build an index from `tools` under `config` and wrap it.
    ///
    /// # Errors
    ///
    /// See [`ToolRetrievalIndex::build`].
    pub fn build(
        tools: Vec<ToolSpecEntry>,
        config: ToolRetrievalConfig,
    ) -> ToolRetrievalResult<Self> {
        Ok(Self::new(ToolRetrievalIndex::build(tools, config)?))
    }

    /// Borrow the wrapped index.
    #[must_use]
    pub fn index(&self) -> &ToolRetrievalIndex {
        &self.index
    }

    /// Rank every indexed tool against `query`, blending description-
    /// embedding cosine similarity with lexical (Jaccard) overlap per
    /// [`ToolRetrievalConfig::description_weight`], keeping only tools at or
    /// above [`ToolRetrievalConfig::min_match_score`], and returning at most
    /// `top_k` results (best score first, ties broken by ascending tool
    /// name).
    ///
    /// A `top_k` of `0` returns an empty (but `Ok`) result. When every tool
    /// scores below the configured minimum, the result is an empty `Vec`,
    /// not an error — only a genuinely invalid `query` or an empty registry
    /// are treated as errors.
    ///
    /// # Errors
    ///
    /// - [`ToolRetrievalError::EmptyQuery`] when `query` is empty or
    ///   whitespace-only.
    /// - [`ToolRetrievalError::EmptyRegistry`] when the wrapped index holds
    ///   no tools.
    pub fn retrieve(&self, query: &str, top_k: usize) -> ToolRetrievalResult<Vec<ToolMatch>> {
        if query.trim().is_empty() {
            return Err(ToolRetrievalError::EmptyQuery);
        }
        if self.index.entries.is_empty() {
            return Err(ToolRetrievalError::EmptyRegistry);
        }

        let config = &self.index.config;
        let query_embedding = embed_text(query, config.embedding_dim);
        let query_terms = content_terms(query);

        let mut matches: Vec<ToolMatch> = self
            .index
            .entries
            .iter()
            .map(|entry| {
                let description_score = cosine(&query_embedding, &entry.embedding);
                let lexical_score = jaccard(&query_terms, &entry.lexical_terms);
                let score = config.description_weight.mul_add(
                    description_score,
                    (1.0 - config.description_weight) * lexical_score,
                );
                ToolMatch {
                    tool_name: entry.spec.name.clone(),
                    score,
                    description_score,
                    lexical_score,
                }
            })
            .filter(|m| m.score >= config.min_match_score)
            .collect();

        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.tool_name.cmp(&b.tool_name))
        });
        matches.truncate(top_k);
        Ok(matches)
    }

    /// [`ToolRetrievalEngine::retrieve`] using
    /// [`ToolRetrievalConfig::default_top_k`] as `top_k`.
    ///
    /// # Errors
    ///
    /// See [`ToolRetrievalEngine::retrieve`].
    pub fn retrieve_default(&self, query: &str) -> ToolRetrievalResult<Vec<ToolMatch>> {
        self.retrieve(query, self.index.config.default_top_k)
    }

    /// Ground every parameter of `spec` against `query`. `spec` need not
    /// belong to this engine's index — grounding is a pure function of the
    /// query and the spec passed in.
    ///
    /// # Errors
    ///
    /// See [`ArgumentGrounder::ground`].
    pub fn ground(
        &self,
        query: &str,
        spec: &ToolSpecEntry,
    ) -> ToolRetrievalResult<Vec<GroundedArgument>> {
        self.grounder.ground(query, spec)
    }
}
