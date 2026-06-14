//! Types for the `anonymization` module.

use std::collections::HashMap;

use thiserror::Error;

// ── PiiKind ───────────────────────────────────────────────────────────────────

/// The category of personally-identifiable information that can be pseudonymized.
///
/// Each variant maps to a distinct placeholder prefix returned by [`PiiKind::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiKind {
    /// A person name (run of capitalized words).
    Person,
    /// An e-mail address (`local@domain.tld`).
    Email,
    /// A phone number (run of digits and separators).
    Phone,
}

impl PiiKind {
    /// Return the placeholder prefix used for this kind.
    ///
    /// The full placeholder for the `n`-th occurrence is `[<PREFIX>_<n>]`, e.g.
    /// `[EMAIL_1]` for the first detected e-mail.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Person => "PERSON",
            Self::Email => "EMAIL",
            Self::Phone => "PHONE",
        }
    }
}

// ── AnonMapping ───────────────────────────────────────────────────────────────

/// A reversible mapping between placeholders and the original PII values.
///
/// The mapping is bidirectional: a placeholder (e.g. `[EMAIL_1]`) maps to the
/// original value it replaced, and each original value maps back to the
/// placeholder that was assigned to it. Insertion order is preserved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnonMapping {
    /// Placeholder → original value, in assignment order.
    placeholder_to_original: Vec<(String, String)>,
    /// Original value → placeholder, for de-duplication during anonymization.
    original_to_placeholder: HashMap<String, String>,
}

impl AnonMapping {
    /// Create a new, empty mapping.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a `placeholder → original` association.
    ///
    /// If `original` is already mapped, the existing placeholder is kept and
    /// no new entry is created. Returns the placeholder now associated with
    /// `original` (either the existing one or the newly inserted one).
    pub(crate) fn insert(&mut self, placeholder: String, original: String) -> String {
        if let Some(existing) = self.original_to_placeholder.get(&original) {
            return existing.clone();
        }
        self.original_to_placeholder
            .insert(original.clone(), placeholder.clone());
        self.placeholder_to_original
            .push((placeholder.clone(), original));
        placeholder
    }

    /// Return the original value for a given `placeholder`, if present.
    #[must_use]
    pub fn original_for(&self, placeholder: &str) -> Option<&str> {
        self.placeholder_to_original
            .iter()
            .find(|(p, _)| p == placeholder)
            .map(|(_, original)| original.as_str())
    }

    /// Return the placeholder assigned to a given `original` value, if present.
    #[must_use]
    pub fn placeholder_for(&self, original: &str) -> Option<&str> {
        self.original_to_placeholder
            .get(original)
            .map(String::as_str)
    }

    /// Return the number of distinct placeholder/original pairs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.placeholder_to_original.len()
    }

    /// Return `true` when the mapping contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.placeholder_to_original.is_empty()
    }

    /// Iterate over `(placeholder, original)` pairs in assignment order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.placeholder_to_original
            .iter()
            .map(|(p, o)| (p.as_str(), o.as_str()))
    }

    /// Collect all `(placeholder, original)` pairs into an owned vector.
    #[must_use]
    pub fn pairs(&self) -> Vec<(String, String)> {
        self.placeholder_to_original.clone()
    }
}

// ── AnonConfig ────────────────────────────────────────────────────────────────

/// Configuration controlling which PII kinds are pseudonymized.
///
/// All kinds are enabled by default. Use the builder methods to disable a kind;
/// disabled kinds are left untouched in the output text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnonConfig {
    /// Whether to anonymize person names.
    pub anonymize_persons: bool,
    /// Whether to anonymize e-mail addresses.
    pub anonymize_emails: bool,
    /// Whether to anonymize phone numbers.
    pub anonymize_phones: bool,
}

impl Default for AnonConfig {
    fn default() -> Self {
        Self {
            anonymize_persons: true,
            anonymize_emails: true,
            anonymize_phones: true,
        }
    }
}

impl AnonConfig {
    /// Create a new configuration with all PII kinds enabled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether person names are anonymized.
    #[must_use]
    pub fn with_anonymize_persons(mut self, value: bool) -> Self {
        self.anonymize_persons = value;
        self
    }

    /// Set whether e-mail addresses are anonymized.
    #[must_use]
    pub fn with_anonymize_emails(mut self, value: bool) -> Self {
        self.anonymize_emails = value;
        self
    }

    /// Set whether phone numbers are anonymized.
    #[must_use]
    pub fn with_anonymize_phones(mut self, value: bool) -> Self {
        self.anonymize_phones = value;
        self
    }

    /// Return `true` when the given `kind` is enabled in this configuration.
    #[must_use]
    pub fn is_enabled(&self, kind: PiiKind) -> bool {
        match kind {
            PiiKind::Person => self.anonymize_persons,
            PiiKind::Email => self.anonymize_emails,
            PiiKind::Phone => self.anonymize_phones,
        }
    }
}

// ── AnonymizedText ────────────────────────────────────────────────────────────

/// The result of anonymizing a text: the rewritten text plus its mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnonymizedText {
    /// The text with PII values replaced by placeholders.
    pub text: String,
    /// The reversible mapping used to restore the original text.
    pub mapping: AnonMapping,
}

// ── AnonError ─────────────────────────────────────────────────────────────────

/// Errors from the `anonymization` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AnonError {
    /// The input text was empty.
    #[error("cannot anonymize empty input")]
    Empty,
}
