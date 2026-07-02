//! Shared lexical toolkit for the `structrag` module.
//!
//! Every helper here is a small, deterministic, regex-free piece of text
//! processing (tokenization, sentence splitting, entity-span detection,
//! key/value extraction, list/step-marker stripping). They are used by both
//! [`crate::structrag::restructure`] (to build knowledge structures from raw
//! passages) and [`crate::structrag::reason`] (to match a query against a
//! built structure). Kept `pub(super)` — visible throughout `structrag`, not
//! part of the module's public API.

// ── tokenization ─────────────────────────────────────────────────────────────

/// Split `text` into lowercase alphanumeric tokens.
pub(super) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Common short words excluded when chunking capitalized entity spans, even
/// though English capitalizes the first word of every sentence. Includes
/// quantifiers/determiners (`"many"`, `"every"`, ...) since those are the
/// most common source of spurious sentence-initial "entities".
const LEADING_STOPWORDS: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "it", "they", "he", "she", "we", "i",
    "there", "here", "when", "what", "why", "how", "who", "which", "and", "or", "but", "so", "if",
    "then", "for", "as", "of", "in", "on", "at", "to", "with", "from", "its", "their", "many",
    "some", "several", "most", "all", "every", "each", "any", "few", "much", "more", "less",
    "other", "another", "such", "same", "own", "no", "not", "only", "just", "also",
];

/// `true` when `word` (compared case-insensitively) is a stopword excluded
/// from entity spans and relation text.
pub(super) fn is_stopword(word: &str) -> bool {
    LEADING_STOPWORDS.contains(&word.to_lowercase().as_str())
}

// ── case-insensitive search safe for subsequent slicing ────────────────────

/// Case-insensitive (ASCII-fold) search for `pattern` in `haystack`,
/// returning the byte offset of the first match.
///
/// Unlike `haystack.to_lowercase().find(pattern)`, the returned offset is
/// always valid for slicing the *original* `haystack`: it is found by
/// comparing ASCII-folded bytes directly against `haystack`'s own indices,
/// never against a separately-lowercased copy whose byte length can drift
/// from the original on non-ASCII input.
pub(super) fn find_ci(haystack: &str, pattern: &str) -> Option<usize> {
    let h = haystack.as_bytes();
    let p = pattern.as_bytes();
    if p.is_empty() || p.len() > h.len() {
        return None;
    }
    'outer: for start in 0..=(h.len() - p.len()) {
        if !haystack.is_char_boundary(start) {
            continue;
        }
        for (offset, &pb) in p.iter().enumerate() {
            if !h[start + offset].eq_ignore_ascii_case(&pb) {
                continue 'outer;
            }
        }
        return Some(start);
    }
    None
}

// ── sentence splitting ───────────────────────────────────────────────────────

/// Split `text` into trimmed sentences on `.`, `!`, or `?`. Falls back to the
/// whole (trimmed) text as a single "sentence" when no terminator is found.
///
/// Splitting on single-byte ASCII terminators and slicing at the resulting
/// byte offsets is always UTF-8-safe: an ASCII byte can never be a
/// continuation byte of a multi-byte sequence, so any position immediately
/// before or after one is guaranteed to be a char boundary.
pub(super) fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (i, b) in text.bytes().enumerate() {
        if b == b'.' || b == b'!' || b == b'?' {
            let end = i + 1;
            let candidate = text[start..end].trim();
            if !candidate.is_empty() {
                sentences.push(candidate);
            }
            start = end;
        }
    }
    if start < text.len() {
        let candidate = text[start..].trim();
        if !candidate.is_empty() {
            sentences.push(candidate);
        }
    }
    if sentences.is_empty() {
        let candidate = text.trim();
        if !candidate.is_empty() {
            sentences.push(candidate);
        }
    }
    sentences
}

// ── entity-span extraction ───────────────────────────────────────────────────

/// Extract maximal runs of consecutive capitalized, non-stopword tokens from
/// `sentence` — a lightweight proper-noun chunker. Runs are returned in
/// order of appearance; duplicates are preserved (callers dedupe as needed).
pub(super) fn extract_entity_spans(sentence: &str) -> Vec<String> {
    let mut spans = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for raw in sentence.split_whitespace() {
        let cleaned = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-');
        let qualifies = cleaned.chars().next().is_some_and(char::is_uppercase)
            && cleaned.chars().any(char::is_alphabetic)
            && !is_stopword(cleaned);
        if qualifies {
            current.push(cleaned.to_string());
        } else if !current.is_empty() {
            spans.push(current.join(" "));
            current.clear();
        }
    }
    if !current.is_empty() {
        spans.push(current.join(" "));
    }
    spans
}

/// The first entity span found anywhere in `text` (scanned sentence by
/// sentence), or the text's first alphanumeric token, or `"item"` if `text`
/// has no alphanumeric content at all.
pub(super) fn first_entity_or_token(text: &str) -> String {
    for sentence in split_sentences(text) {
        if let Some(first) = extract_entity_spans(sentence).into_iter().next() {
            return first;
        }
    }
    text.split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .find(|s| !s.is_empty())
        .unwrap_or_else(|| "item".to_string())
}

// ── numeric extraction ───────────────────────────────────────────────────────

/// The first numeric token in `text`, formatted with a leading currency
/// symbol or trailing `%` preserved when present in the raw token. `None`
/// when no number is found.
pub(super) fn first_number_token(text: &str) -> Option<String> {
    for token in text.split_whitespace() {
        let cleaned: String = token
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if cleaned.is_empty()
            || cleaned.chars().all(|c| c == '.')
            || cleaned.parse::<f64>().is_err()
        {
            continue;
        }
        let has_currency =
            token.starts_with('$') || token.starts_with('€') || token.starts_with('£');
        let prefix = if has_currency {
            token.chars().next().map(String::from).unwrap_or_default()
        } else {
            String::new()
        };
        let suffix = if token.ends_with('%') { "%" } else { "" };
        return Some(format!("{prefix}{cleaned}{suffix}"));
    }
    None
}

// ── key/value + list segmentation ────────────────────────────────────────────

/// Split `text` into candidate key/value or list segments on `,`, `;`, and
/// newlines, trimming and dropping empty segments.
pub(super) fn split_segments(text: &str) -> Vec<&str> {
    text.split([',', ';', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extract explicit `key: value` / `key = value` pairs from `text`. A
/// segment only counts as a pair when its key is short (at most four words)
/// and both key and value are non-empty, which keeps ordinary prose
/// sentences (no real key/value structure) from being mis-parsed as one.
///
/// Requires at least half of `text`'s comma/semicolon/newline-delimited
/// segments to themselves look like `key: value` (so pairs are only
/// reported for genuine multi-field records like `"name: X, price: Y"`).
/// Without this majority check, a single-list intro like `"Ingredients:
/// flour, sugar, and eggs"` would mis-parse as the one pair
/// `("Ingredients", "flour")`, silently dropping `"sugar"` and `"eggs"` —
/// that shape is instead left for the caller's own list-handling fallback.
pub(super) fn extract_key_value_pairs(text: &str) -> Vec<(String, String)> {
    let segments = split_segments(text);
    if segments.is_empty() {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    for segment in &segments {
        let sep = segment.find(':').or_else(|| segment.find('='));
        if let Some(pos) = sep {
            let key = segment[..pos].trim();
            let value = segment[pos + 1..].trim().trim_end_matches('.').trim();
            if !key.is_empty() && !value.is_empty() && key.split_whitespace().count() <= 4 {
                pairs.push((key.to_string(), value.to_string()));
            }
        }
    }
    if pairs.len() * 2 >= segments.len() {
        pairs
    } else {
        Vec::new()
    }
}

/// Split a comma/`and`-separated list such as `"mammals, birds, and
/// reptiles"` into `["mammals", "birds", "reptiles"]`.
pub(super) fn split_list(text: &str) -> Vec<String> {
    text.split(',')
        .flat_map(|part| part.split(" and "))
        .map(|s| s.trim().trim_start_matches("and ").trim())
        .map(|s| s.trim_end_matches('.').trim())
        .map(strip_leading_article)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Strip a leading `"a "`, `"an "`, or `"the "` article (case-insensitive),
/// for cleaner extracted labels (e.g. `"A Sparrow"` -> `"Sparrow"`).
///
/// Safe to slice at `article.len()` after a match: [`find_ci`] only matches
/// literal ASCII bytes, and an ASCII byte can never be part of a multi-byte
/// UTF-8 sequence, so the position immediately after an ASCII match is
/// always a valid char boundary.
pub(super) fn strip_leading_article(text: &str) -> &str {
    let trimmed = text.trim();
    for article in ["a ", "an ", "the "] {
        if find_ci(trimmed, article) == Some(0) {
            return trimmed[article.len()..].trim();
        }
    }
    trimmed
}

/// Truncate `text` to at most `max_chars` Unicode scalar values, appending
/// `"..."` when truncated. Char-based (never byte-slices), so always
/// UTF-8-safe regardless of content.
pub(super) fn truncate_for_display(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        let truncated: String = trimmed.chars().take(max_chars).collect();
        format!("{truncated}...")
    }
}

// ── list-marker / indentation helpers (tree + catalogue) ────────────────────

/// Strip a leading bullet marker (`-`, `*`, `•`) plus following whitespace,
/// returning the remainder. `None` when `text` has no leading bullet.
pub(super) fn strip_list_marker(text: &str) -> Option<&str> {
    let after_ws = text.trim_start_matches([' ', '\t']);
    let first = after_ws.chars().next()?;
    if first == '-' || first == '*' || first == '\u{2022}' {
        Some(after_ws[first.len_utf8()..].trim_start())
    } else {
        None
    }
}

/// Estimate the outline nesting depth implied by `text`'s leading
/// whitespace/bullet markers: every two leading spaces (a tab counts as
/// two), or every leading bullet character, adds one level. Capped at `8` to
/// guard against runaway input.
pub(super) fn leading_indent_depth(text: &str) -> usize {
    let mut space_count = 0usize;
    let mut bullet_count = 0usize;
    for c in text.chars() {
        match c {
            ' ' => space_count += 1,
            '\t' => space_count += 2,
            '-' | '*' | '\u{2022}' => bullet_count += 1,
            _ => break,
        }
    }
    (space_count / 2 + bullet_count).min(8)
}

// ── step-marker stripping (algorithm) ────────────────────────────────────────

/// Connective words stripped from the front of a step's action text; they
/// carry no explicit numeric order.
const STEP_CONNECTIVES: &[&str] = &[
    "first,", "first", "second,", "second", "third,", "third", "fourth,", "fourth", "then,",
    "then", "next,", "next", "finally,", "finally", "lastly,", "lastly",
];

/// Recognize an explicit step marker at the start of `text`: `"Step N"`
/// (optionally followed by `:`/`.`), or a leading `"N."`/`"N)"`/`"N:"`
/// ordinal. Returns the marker's one-based order (if any) and the remaining
/// action text with the marker and any leading connective word stripped.
///
/// Operates entirely via whole-token comparisons (`eq_ignore_ascii_case`,
/// token removal + rejoin) rather than slicing at offsets derived from a
/// separately-cased copy of `text`, so it is UTF-8-safe on arbitrary input.
pub(super) fn strip_step_marker(text: &str) -> (Option<usize>, String) {
    let trimmed = text.trim();
    let mut tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.is_empty() {
        return (None, String::new());
    }

    if tokens.len() >= 2 && tokens[0].eq_ignore_ascii_case("step") {
        // An empty (or non-numeric) `digits` simply fails to parse below, so
        // no separate emptiness guard is needed.
        let digits: String = tokens[1].chars().filter(char::is_ascii_digit).collect();
        if let Ok(n) = digits.parse::<usize>() {
            tokens.remove(0);
            tokens.remove(0);
            return (Some(n), strip_leading_punct(&tokens.join(" ")));
        }
    }

    let first = tokens[0];
    let digits: String = first.chars().take_while(char::is_ascii_digit).collect();
    // Safe to slice at `digits.len()`: `digits` is built from a strict
    // ASCII-digit prefix of `first`'s chars, so its byte length never
    // exceeds `first`'s.
    let suffix = &first[digits.len()..];
    if (suffix == "." || suffix == ")" || suffix == ":")
        && let Ok(n) = digits.parse::<usize>()
    {
        tokens.remove(0);
        return (Some(n), strip_leading_punct(&tokens.join(" ")));
    }

    if STEP_CONNECTIVES
        .iter()
        .any(|c| first.eq_ignore_ascii_case(c))
    {
        tokens.remove(0);
        return (None, strip_leading_punct(&tokens.join(" ")));
    }

    (None, trimmed.to_string())
}

/// Trim leading `:`, `,`, `-`, and whitespace left over after marker
/// removal.
fn strip_leading_punct(text: &str) -> String {
    text.trim_start_matches([':', ',', '-', ' '])
        .trim()
        .to_string()
}
