//! PII pseudonymization engine — hand-written char-class scanning (no regex).
//!
//! Detects PII spans deterministically and replaces each with a consistent
//! placeholder, keeping a reversible [`AnonMapping`] so the original text can
//! be restored via [`Anonymizer::deanonymize`].

use std::collections::HashMap;

use super::types::{AnonConfig, AnonError, AnonMapping, AnonymizedText, PiiKind};

// ── char-class helpers ──────────────────────────────────────────────────────

/// Return `true` if `c` is a decimal digit.
fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// Return `true` if `c` is a valid e-mail local-part or domain character.
fn is_email_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+')
}

/// Return `true` if `c` may separate digit groups within a phone number.
fn is_phone_separator(c: char) -> bool {
    matches!(c, '-' | '.' | ' ' | '(' | ')')
}

/// Return `true` if `c` starts a capitalized word (a person-name token).
fn is_name_start(c: char) -> bool {
    c.is_ascii_uppercase()
}

/// Return `true` if `c` may continue a capitalized word after its first letter.
fn is_name_continue(c: char) -> bool {
    c.is_ascii_alphabetic()
}

// ── detected span ────────────────────────────────────────────────────────────

/// A detected PII span over byte offsets into the original text.
#[derive(Debug, Clone, Copy)]
struct Span {
    kind: PiiKind,
    start: usize,
    end: usize,
}

// ── Anonymizer ───────────────────────────────────────────────────────────────

/// Pseudonymizes PII in text and restores it through a reversible mapping.
///
/// The same original value always maps to the same placeholder within a single
/// [`Anonymizer::anonymize`] call, and placeholders are assigned with an
/// incrementing per-kind counter in left-to-right order of first appearance.
#[derive(Debug, Clone, Default)]
pub struct Anonymizer {
    config: AnonConfig,
}

impl Anonymizer {
    /// Create a new anonymizer with the given `config`.
    #[must_use]
    pub fn new(config: AnonConfig) -> Self {
        Self { config }
    }

    /// Return a reference to the active configuration.
    #[must_use]
    pub fn config(&self) -> &AnonConfig {
        &self.config
    }

    /// Anonymize `text`, returning the rewritten text and its reversible mapping.
    ///
    /// Detects every enabled PII kind, assigns incrementing per-kind
    /// placeholders (`[EMAIL_1]`, `[EMAIL_2]`, `[PHONE_1]`, `[PERSON_1]`, …),
    /// and reuses the same placeholder whenever an identical original value
    /// reappears. This method never fails; for empty input it returns an empty
    /// result. Use [`Anonymizer::anonymize_checked`] to reject empty input.
    #[must_use]
    pub fn anonymize(&self, text: &str) -> AnonymizedText {
        let chars: Vec<char> = text.chars().collect();
        let byte_offsets = byte_offsets(&chars);

        let mut spans = Vec::new();
        if self.config.anonymize_emails {
            scan_emails(&chars, &byte_offsets, &mut spans);
        }
        if self.config.anonymize_phones {
            scan_phones(&chars, &byte_offsets, &mut spans);
        }
        if self.config.anonymize_persons {
            scan_persons(&chars, &byte_offsets, &mut spans);
        }

        let spans = resolve_overlaps(spans);
        self.rewrite(text, &spans)
    }

    /// Anonymize `text`, returning [`AnonError::Empty`] when `text` is empty.
    ///
    /// # Errors
    ///
    /// Returns [`AnonError::Empty`] if `text` is empty.
    pub fn anonymize_checked(&self, text: &str) -> Result<AnonymizedText, AnonError> {
        if text.is_empty() {
            return Err(AnonError::Empty);
        }
        Ok(self.anonymize(text))
    }

    /// Restore the original text by replacing placeholders with their originals.
    ///
    /// Placeholders absent from `mapping` are left unchanged. Longer
    /// placeholders are substituted before shorter ones so that, for example,
    /// `[EMAIL_10]` is never partially matched as `[EMAIL_1]`.
    #[must_use]
    pub fn deanonymize(&self, text: &str, mapping: &AnonMapping) -> String {
        let mut pairs = mapping.pairs();
        // Replace longer placeholders first to avoid prefix collisions.
        pairs.sort_by_key(|(placeholder, _)| std::cmp::Reverse(placeholder.len()));
        let mut result = text.to_string();
        for (placeholder, original) in pairs {
            if result.contains(&placeholder) {
                result = result.replace(&placeholder, &original);
            }
        }
        result
    }

    /// Rewrite `text`, substituting each resolved span with its placeholder and
    /// recording the association in a fresh [`AnonMapping`].
    fn rewrite(&self, text: &str, spans: &[Span]) -> AnonymizedText {
        let mut mapping = AnonMapping::new();
        let mut counters: HashMap<PiiKind, usize> = HashMap::new();
        let mut result = String::with_capacity(text.len());
        let bytes = text.as_bytes();
        let mut cursor = 0usize;

        for span in spans {
            if span.start < cursor {
                continue;
            }
            result.push_str(slice(bytes, cursor, span.start));
            let original = slice(bytes, span.start, span.end).to_string();
            let placeholder =
                self.placeholder_for(&original, span.kind, &mut counters, &mut mapping);
            result.push_str(&placeholder);
            cursor = span.end;
        }
        result.push_str(slice(bytes, cursor, bytes.len()));

        AnonymizedText {
            text: result,
            mapping,
        }
    }

    /// Return the placeholder for `original`, reusing the existing one when the
    /// value was already seen, otherwise minting the next per-kind placeholder.
    #[allow(clippy::unused_self)]
    fn placeholder_for(
        &self,
        original: &str,
        kind: PiiKind,
        counters: &mut HashMap<PiiKind, usize>,
        mapping: &mut AnonMapping,
    ) -> String {
        if let Some(existing) = mapping.placeholder_for(original) {
            return existing.to_string();
        }
        let next = counters.entry(kind).or_insert(0);
        *next += 1;
        let candidate = format!("[{}_{}]", kind.as_str(), *next);
        mapping.insert(candidate, original.to_string())
    }
}

// ── byte-offset index ────────────────────────────────────────────────────────

/// Build a char-index → byte-offset table with a trailing sentinel.
fn byte_offsets(chars: &[char]) -> Vec<usize> {
    let mut off = 0usize;
    let mut offs = Vec::with_capacity(chars.len() + 1);
    for c in chars {
        offs.push(off);
        off += c.len_utf8();
    }
    offs.push(off);
    offs
}

/// Borrow `bytes[start..end]` as `&str`, falling back to `""` on a bad slice.
fn slice(bytes: &[u8], start: usize, end: usize) -> &str {
    std::str::from_utf8(&bytes[start..end]).unwrap_or("")
}

// ── e-mail scanning ──────────────────────────────────────────────────────────

/// Detect `local@domain.tld` spans and push them onto `spans`.
fn scan_emails(chars: &[char], byte_offsets: &[usize], spans: &mut Vec<Span>) {
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && is_email_char(chars[j]) {
            j += 1;
        }
        if j == i || j >= n || chars[j] != '@' {
            i += 1;
            continue;
        }
        let at_pos = j;
        j += 1; // skip '@'
        let domain_start = j;
        while j < n && is_email_char(chars[j]) {
            j += 1;
        }
        let domain: String = chars[domain_start..j].iter().collect();
        if at_pos > i && domain.contains('.') && j > domain_start + 2 {
            spans.push(Span {
                kind: PiiKind::Email,
                start: byte_offsets[i],
                end: byte_offsets[j],
            });
            i = j;
            continue;
        }
        i += 1;
    }
}

// ── phone scanning ───────────────────────────────────────────────────────────

/// Detect runs of digits and separators with at least seven digits.
fn scan_phones(chars: &[char], byte_offsets: &[usize], spans: &mut Vec<Span>) {
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if !is_digit(chars[i]) && chars[i] != '+' {
            i += 1;
            continue;
        }
        let start = i;
        let mut digits = 0usize;
        let mut last_digit = start;
        let mut j = i;
        while j < n {
            let c = chars[j];
            if is_digit(c) {
                digits += 1;
                j += 1;
                last_digit = j;
            } else if (is_phone_separator(c) && digits > 0) || (c == '+' && j == start) {
                j += 1;
            } else {
                break;
            }
        }
        if digits >= 7 {
            // Trim trailing separators so the span ends on the last digit.
            spans.push(Span {
                kind: PiiKind::Phone,
                start: byte_offsets[start],
                end: byte_offsets[last_digit],
            });
            i = j;
        } else {
            i += 1;
        }
    }
}

// ── person-name scanning ─────────────────────────────────────────────────────

/// Detect runs of one or more capitalized words separated by single spaces.
fn scan_persons(chars: &[char], byte_offsets: &[usize], spans: &mut Vec<Span>) {
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if !is_name_start(chars[i]) {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = scan_capitalized_word(chars, i);
        let mut last_word_end = end;
        // Greedily extend across "<space><Capitalized>" sequences.
        loop {
            let mut k = end;
            if k < n && chars[k] == ' ' {
                k += 1;
                if k < n && is_name_start(chars[k]) {
                    end = scan_capitalized_word(chars, k);
                    last_word_end = end;
                    continue;
                }
            }
            break;
        }
        spans.push(Span {
            kind: PiiKind::Person,
            start: byte_offsets[start],
            end: byte_offsets[last_word_end],
        });
        i = last_word_end;
    }
}

/// Return the char index just past the capitalized word starting at `start`.
fn scan_capitalized_word(chars: &[char], start: usize) -> usize {
    let n = chars.len();
    let mut j = start + 1;
    while j < n && is_name_continue(chars[j]) {
        j += 1;
    }
    j
}

// ── overlap resolution ───────────────────────────────────────────────────────

/// Sort spans left-to-right and drop any that overlap an earlier kept span.
///
/// Ties on start position keep the longer span; this lets e-mail and phone
/// spans win over an incidental capitalized-word span at the same position.
fn resolve_overlaps(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then((b.end - b.start).cmp(&(a.end - a.start)))
            .then(kind_rank(a.kind).cmp(&kind_rank(b.kind)))
    });
    let mut kept: Vec<Span> = Vec::with_capacity(spans.len());
    let mut cursor = 0usize;
    for span in spans {
        if span.start >= cursor {
            cursor = span.end;
            kept.push(span);
        }
    }
    kept
}

/// Priority used to break start-position ties (lower wins).
fn kind_rank(kind: PiiKind) -> u8 {
    match kind {
        PiiKind::Email => 0,
        PiiKind::Phone => 1,
        PiiKind::Person => 2,
    }
}
