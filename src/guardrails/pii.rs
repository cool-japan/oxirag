//! PII detection and redaction — hand-written char-class scanning (no regex).

use std::fmt::Write as _;

use super::types::{PiiKind, PiiMatch};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Return `true` if `c` is a decimal digit.
fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// Return `true` if `c` is a valid email-local or domain character.
fn is_email_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+')
}

// ── PiiDetector ───────────────────────────────────────────────────────────────

/// Scans text for PII patterns using character-class state machines.
///
/// No external regex library is used; all matching is hand-written.
#[derive(Debug, Clone, Default)]
pub struct PiiDetector;

impl PiiDetector {
    /// Create a new [`PiiDetector`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Scan `text` and return all detected PII matches.
    ///
    /// Multiple PII types can match at the same span.
    #[must_use]
    pub fn scan(&self, text: &str) -> Vec<PiiMatch> {
        let chars: Vec<char> = text.chars().collect();
        let mut matches = Vec::new();

        // Build byte-offset index for char positions
        let byte_offsets: Vec<usize> = {
            let mut off = 0usize;
            let mut offs = Vec::with_capacity(chars.len() + 1);
            for c in &chars {
                offs.push(off);
                off += c.len_utf8();
            }
            offs.push(off); // sentinel
            offs
        };

        self.scan_emails(text, &chars, &byte_offsets, &mut matches);
        self.scan_phones(&chars, &byte_offsets, &mut matches);
        self.scan_ssn(&chars, &byte_offsets, &mut matches);
        self.scan_credit_cards(&chars, &byte_offsets, &mut matches);
        self.scan_ipv4(&chars, &byte_offsets, &mut matches);

        matches
    }

    /// Replace all PII matches in `text` with `[REDACTED-<KIND>]` labels.
    #[must_use]
    pub fn redact(&self, text: &str) -> String {
        let mut matches = self.scan(text);
        if matches.is_empty() {
            return text.to_string();
        }
        // Sort by start, longest first, then remove overlaps
        matches.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
        let mut result = String::new();
        let bytes = text.as_bytes();
        let mut cursor = 0usize;
        for m in &matches {
            if m.start >= cursor {
                result.push_str(std::str::from_utf8(&bytes[cursor..m.start]).unwrap_or(""));
                let _ = write!(result, "[REDACTED-{}]", m.kind.label().to_uppercase());
                cursor = m.end;
            }
        }
        result.push_str(std::str::from_utf8(&bytes[cursor..]).unwrap_or(""));
        result
    }

    // ── email ─────────────────────────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn scan_emails(
        &self,
        text: &str,
        chars: &[char],
        byte_offsets: &[usize],
        matches: &mut Vec<PiiMatch>,
    ) {
        let n = chars.len();
        let mut i = 0;
        while i < n {
            // Scan local part
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
            while j < n && (is_email_char(chars[j])) {
                j += 1;
            }
            // Must have a dot in the domain part
            let domain: String = chars[domain_start..j].iter().collect();
            if domain.contains('.') && j > domain_start + 2 {
                let start = byte_offsets[i];
                let end = byte_offsets[j];
                let pii_str = text[start..end].to_string();
                // Make sure local part is reasonable
                if at_pos > i {
                    matches.push(PiiMatch {
                        kind: PiiKind::Email,
                        start,
                        end,
                        matched: pii_str,
                    });
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
    }

    // ── phone ─────────────────────────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn scan_phones(&self, chars: &[char], byte_offsets: &[usize], matches: &mut Vec<PiiMatch>) {
        let n = chars.len();
        let mut i = 0;
        while i < n {
            // Phones: optional +1, then groups of digits separated by -, (, ), space
            // Must have 10-15 digits total
            if !is_digit(chars[i]) && chars[i] != '+' {
                i += 1;
                continue;
            }
            let start = i;
            let mut digits = 0usize;
            let mut j = i;
            while j < n {
                let c = chars[j];
                if is_digit(c) {
                    digits += 1;
                    j += 1;
                } else if (matches!(c, '-' | '.' | ' ' | '(' | ')') && digits > 0)
                    || (c == '+' && j == start)
                {
                    j += 1;
                } else {
                    break;
                }
            }
            if (10..=15).contains(&digits) {
                let start_byte = byte_offsets[start];
                let end_byte = byte_offsets[j];
                let pii_str: String = chars[start..j].iter().collect();
                matches.push(PiiMatch {
                    kind: PiiKind::Phone,
                    start: start_byte,
                    end: end_byte,
                    matched: pii_str,
                });
                i = j;
            } else {
                i += 1;
            }
        }
    }

    // ── SSN ───────────────────────────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn scan_ssn(&self, chars: &[char], byte_offsets: &[usize], matches: &mut Vec<PiiMatch>) {
        let n = chars.len();
        let mut i = 0;
        while i + 10 <= n {
            // XXX-XX-XXXX or XXX XX XXXX
            let sep = if chars[i + 3] == '-' { '-' } else { ' ' };
            if chars[i + 3] == sep
                && chars[i + 6] == sep
                && chars[i..i + 3].iter().all(|c| is_digit(*c))
                && chars[i + 4..i + 6].iter().all(|c| is_digit(*c))
                && chars[i + 7..i + 11].iter().all(|c| is_digit(*c))
            {
                let start = byte_offsets[i];
                let end = byte_offsets[i + 11];
                let pii_str: String = chars[i..i + 11].iter().collect();
                matches.push(PiiMatch {
                    kind: PiiKind::Ssn,
                    start,
                    end,
                    matched: pii_str,
                });
                i += 11;
            } else {
                i += 1;
            }
        }
    }

    // ── credit card ───────────────────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn scan_credit_cards(
        &self,
        chars: &[char],
        byte_offsets: &[usize],
        matches: &mut Vec<PiiMatch>,
    ) {
        let n = chars.len();
        let mut i = 0;
        while i < n {
            if !is_digit(chars[i]) {
                i += 1;
                continue;
            }
            // Collect digits and separators over up to 22 chars
            let start = i;
            let mut digits = 0usize;
            let mut j = i;
            while j < n && j - start < 22 {
                let c = chars[j];
                if is_digit(c) {
                    digits += 1;
                    j += 1;
                } else if (c == '-' || c == ' ') && digits > 0 {
                    j += 1;
                } else {
                    break;
                }
            }
            if digits == 16 {
                let start_byte = byte_offsets[start];
                let end_byte = byte_offsets[j];
                let pii_str: String = chars[start..j].iter().collect();
                matches.push(PiiMatch {
                    kind: PiiKind::CreditCard,
                    start: start_byte,
                    end: end_byte,
                    matched: pii_str,
                });
                i = j;
            } else {
                i += 1;
            }
        }
    }

    // ── IPv4 ──────────────────────────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn scan_ipv4(&self, chars: &[char], byte_offsets: &[usize], matches: &mut Vec<PiiMatch>) {
        let n = chars.len();
        let mut i = 0;
        while i < n {
            if !is_digit(chars[i]) {
                i += 1;
                continue;
            }
            let start = i;
            let mut octets = 0usize;
            let mut j = i;
            let mut valid = true;

            while octets < 4 {
                let oct_start = j;
                let mut oct_digits = 0usize;
                while j < n && is_digit(chars[j]) && oct_digits < 3 {
                    oct_digits += 1;
                    j += 1;
                }
                if oct_digits == 0 {
                    valid = false;
                    break;
                }
                // Check value 0-255
                let oct_str: String = chars[oct_start..j].iter().collect();
                let val: u32 = oct_str.parse().unwrap_or(999);
                if val > 255 {
                    valid = false;
                    break;
                }
                octets += 1;
                if octets < 4 {
                    if j < n && chars[j] == '.' {
                        j += 1;
                    } else {
                        valid = false;
                        break;
                    }
                }
            }

            if valid && octets == 4 {
                let start_byte = byte_offsets[start];
                let end_byte = byte_offsets[j];
                let pii_str: String = chars[start..j].iter().collect();
                matches.push(PiiMatch {
                    kind: PiiKind::IpAddress,
                    start: start_byte,
                    end: end_byte,
                    matched: pii_str,
                });
                i = j;
            } else {
                i += 1;
            }
        }
    }
}
