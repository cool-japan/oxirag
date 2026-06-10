//! Types for the `fact_triple` module.

use thiserror::Error;

// ── Triple ────────────────────────────────────────────────────────────────────

/// A subject–predicate–object fact triple extracted from natural language.
///
/// Each [`Triple`] has an associated `confidence` score in `[0.0, 1.0]` and an
/// optional `source_id` referring to the document it was extracted from.
#[derive(Debug, Clone)]
pub struct Triple {
    /// The subject of the fact (agent or topic).
    pub subject: String,
    /// The predicate (verb or relationship).
    pub predicate: String,
    /// The object of the fact (recipient or value).
    pub object: String,
    /// Extraction confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Optional source document identifier.
    pub source_id: Option<String>,
}

impl Triple {
    /// Create a new [`Triple`] with the given fields.
    ///
    /// `confidence` should be in `[0.0, 1.0]`.
    #[must_use]
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            confidence,
            source_id: None,
        }
    }

    /// Attach a source document ID to this triple.
    #[must_use]
    pub fn with_source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }
}

impl std::fmt::Display for Triple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {}",
            self.subject,
            self.predicate.to_uppercase(),
            self.object
        )
    }
}

// ── TripleStore ───────────────────────────────────────────────────────────────

/// An in-memory collection of [`Triple`]s with simple lookup helpers.
#[derive(Debug, Clone, Default)]
pub struct TripleStore {
    /// All triples held in this store.
    pub triples: Vec<Triple>,
}

impl TripleStore {
    /// Create an empty [`TripleStore`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a [`Triple`] to the store.
    pub fn add(&mut self, triple: Triple) {
        self.triples.push(triple);
    }

    /// Return all triples whose `subject` equals `s` (case-insensitive).
    #[must_use]
    pub fn find_by_subject<'a>(&'a self, s: &str) -> Vec<&'a Triple> {
        self.triples
            .iter()
            .filter(|t| t.subject.eq_ignore_ascii_case(s))
            .collect()
    }

    /// Return all triples whose `predicate` equals `p` (case-insensitive).
    #[must_use]
    pub fn find_by_predicate<'a>(&'a self, p: &str) -> Vec<&'a Triple> {
        self.triples
            .iter()
            .filter(|t| t.predicate.eq_ignore_ascii_case(p))
            .collect()
    }

    /// Return all triples whose `object` equals `o` (case-insensitive).
    #[must_use]
    pub fn find_by_object<'a>(&'a self, o: &str) -> Vec<&'a Triple> {
        self.triples
            .iter()
            .filter(|t| t.object.eq_ignore_ascii_case(o))
            .collect()
    }

    /// Return all triples where any field contains at least one token from `query`.
    ///
    /// Matching is case-insensitive substring matching on individual whitespace-
    /// separated tokens.
    #[must_use]
    pub fn match_query<'a>(&'a self, query: &str) -> Vec<&'a Triple> {
        let tokens: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();

        self.triples
            .iter()
            .filter(|t| {
                let fields = [
                    t.subject.to_lowercase(),
                    t.predicate.to_lowercase(),
                    t.object.to_lowercase(),
                ];
                tokens
                    .iter()
                    .any(|tok| fields.iter().any(|f| f.contains(tok.as_str())))
            })
            .collect()
    }

    /// Number of triples in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Returns `true` when the store contains no triples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}

// ── TripleConfig ──────────────────────────────────────────────────────────────

/// Configuration for [`TripleExtractor`].
#[derive(Debug, Clone)]
pub struct TripleConfig {
    /// Minimum confidence threshold; triples below this are discarded.
    ///
    /// Defaults to `0.5`.
    pub min_confidence: f32,
    /// Maximum number of triples extracted per sentence.
    ///
    /// Defaults to `3`.
    pub max_per_sentence: usize,
}

impl Default for TripleConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_per_sentence: 3,
        }
    }
}

impl TripleConfig {
    /// Create a new [`TripleConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set the maximum number of triples extracted per sentence.
    #[must_use]
    pub fn with_max_per_sentence(mut self, max_per_sentence: usize) -> Self {
        self.max_per_sentence = max_per_sentence;
        self
    }
}

// ── TripleExtractor ───────────────────────────────────────────────────────────

/// Heuristic subject–verb–object triple extractor.
///
/// Wraps a [`TripleConfig`] that controls confidence filtering and per-sentence
/// caps.  The actual extraction logic lives in `extractor.rs`.
#[derive(Debug, Clone)]
pub struct TripleExtractor {
    /// Configuration for this extractor instance.
    pub config: TripleConfig,
}

impl TripleExtractor {
    /// Create a new extractor with the given [`TripleConfig`].
    #[must_use]
    pub fn new(config: TripleConfig) -> Self {
        Self { config }
    }
}

impl Default for TripleExtractor {
    fn default() -> Self {
        Self::new(TripleConfig::default())
    }
}

// ── TripleError ───────────────────────────────────────────────────────────────

/// Errors that can be returned by triple extraction operations.
///
/// Note: an empty result (no triples found) is **not** an error — only
/// structurally invalid inputs produce errors.
#[derive(Debug, Error)]
pub enum TripleError {
    /// The input text was completely empty.
    #[error("Input must not be empty")]
    EmptyInput,

    /// The input contained text but no parseable triples were found.
    #[error("No parseable triples")]
    NoParseable,
}
