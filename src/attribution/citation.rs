//! Citation formatting: inline markers and reference lists.
//!
//! [`CitationFormatter`] converts [`CitedSpan`] annotation results into
//! human-readable citation markers and inserts them into an answer string.

use serde::{Deserialize, Serialize};

use super::types::{Citation, CitedSpan};

// ── CitationStyle ─────────────────────────────────────────────────────────────

/// The visual style used for inline citation markers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CitationStyle {
    /// Square-bracket numeric markers: `[1]`, `[2]`, …
    #[default]
    Numeric,
    /// Markdown footnote-style markers: `[^1]`, `[^2]`, …
    Footnote,
    /// Author/title markers: `(Title)` or `(source_id)` when no title.
    Author,
}

// ── Private sentence splitter (own copy, no `flare` feature dependency) ──────

/// Split text into sentences.
///
/// Splits on `. `, `? `, `! ` (each requiring a trailing space) and `\n\n`.
/// A single sentence with no terminal space (e.g. `"Foo."`) is returned as-is
/// in a one-element `Vec`.
pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        current.push(chars[i]);
        // Check for sentence-ending punctuation followed by a space.
        if i + 1 < n {
            let end_punct = chars[i] == '.' || chars[i] == '?' || chars[i] == '!';
            let next_space = chars[i + 1] == ' ';
            if end_punct && next_space {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current = String::new();
                i += 2; // skip the space
                continue;
            }
        }
        // Check for paragraph break (\n\n).
        if i + 1 < n && chars[i] == '\n' && chars[i + 1] == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
            i += 2;
            continue;
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

// ── CitationFormatter ─────────────────────────────────────────────────────────

/// Formats citation markers and annotated answer strings.
pub struct CitationFormatter;

impl CitationFormatter {
    /// Construct a new [`CitationFormatter`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Format a single inline citation marker.
    ///
    /// | Style       | Example output                |
    /// |-------------|-------------------------------|
    /// | `Numeric`   | `[1]`                         |
    /// | `Footnote`  | `[^1]`                        |
    /// | `Author`    | `(Title)` or `(source_id)`    |
    #[must_use]
    pub fn format_marker(style: CitationStyle, index: usize, c: &Citation) -> String {
        match style {
            CitationStyle::Numeric => format!("[{index}]"),
            CitationStyle::Footnote => format!("[^{index}]"),
            CitationStyle::Author => {
                let label = c
                    .source_title
                    .as_deref()
                    .unwrap_or_else(|| c.source_id.as_str());
                format!("({label})")
            }
        }
    }

    /// Annotate the answer string by appending inline markers after each
    /// sentence that has at least one citation.
    ///
    /// The sentence list is reconstructed using the same sentence-splitting
    /// logic employed by the aligner, guaranteeing identical boundaries.
    /// All original sentences are preserved; only grounded ones receive
    /// appended markers.
    #[must_use]
    pub fn annotate(
        &self,
        answer: &str,
        spans: &[CitedSpan],
        style: CitationStyle,
        citations: &[Citation],
    ) -> String {
        let sentences = split_sentences(answer);
        let mut parts: Vec<String> = Vec::with_capacity(sentences.len());

        for (idx, sentence) in sentences.iter().enumerate() {
            // Find the corresponding span (by sentence_index).
            let span_opt = spans.iter().find(|s| s.sentence_index == idx);

            // Build markers and push annotated or plain sentence.
            let annotated = span_opt.and_then(|span| {
                if span.citations.is_empty() {
                    return None;
                }
                // Collect formatted markers for all citations that appear in
                // the deduped citations list.
                let markers: String = span
                    .citations
                    .iter()
                    .filter_map(|c| {
                        citations
                            .iter()
                            .position(|dc| dc.id == c.id)
                            .map(|pos| Self::format_marker(style, pos + 1, c))
                    })
                    .collect::<String>();

                if markers.is_empty() {
                    None
                } else {
                    Some(format!("{sentence}{markers}"))
                }
            });

            parts.push(annotated.unwrap_or_else(|| sentence.clone()));
        }

        // Rejoin with a single space between sentences.
        parts.join(" ")
    }

    /// Generate a bibliography / reference list for the provided citations.
    ///
    /// Format per style:
    /// - `Numeric`:  `"[1] Title\n[2] Another\n"`
    /// - `Footnote`: `"[^1] Title\n[^2] Another\n"`
    /// - `Author`:   `"(Title) Title\n(Another) Another\n"`
    #[must_use]
    pub fn reference_list(style: CitationStyle, citations: &[Citation]) -> String {
        use std::fmt::Write as _;

        citations
            .iter()
            .enumerate()
            .fold(String::new(), |mut out, (idx, c)| {
                let number = idx + 1;
                let label = c
                    .source_title
                    .as_deref()
                    .unwrap_or_else(|| c.source_id.as_str());
                let prefix = match style {
                    CitationStyle::Numeric => format!("[{number}]"),
                    CitationStyle::Footnote => format!("[^{number}]"),
                    CitationStyle::Author => {
                        let author = c
                            .source_title
                            .as_deref()
                            .unwrap_or_else(|| c.source_id.as_str());
                        format!("({author})")
                    }
                };
                // `write!` on a String is infallible; the error is `fmt::Error`
                // which cannot occur for `String`.
                let _ = writeln!(out, "{prefix} {label}");
                out
            })
    }
}

impl Default for CitationFormatter {
    fn default() -> Self {
        Self::new()
    }
}
