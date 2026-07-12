//! Core data structures for classical field-weighted BM25 (BM25F) retrieval.

use std::collections::BTreeMap;

use thiserror::Error;

// ── Bm25fError / Bm25fResult ────────────────────────────────────────────────

/// Errors raised while building or querying a
/// [`Bm25fIndex`](crate::bm25f_retrieval::Bm25fIndex).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Bm25fError {
    /// The supplied corpus contained no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The supplied query was blank.
    #[error("query must not be empty")]
    EmptyQuery,
    /// A search was attempted before the index had been built.
    #[error("index not built")]
    NotBuilt,
}

/// Convenience alias for fallible `bm25f_retrieval` results.
pub type Bm25fResult<T> = Result<T, Bm25fError>;

// ── Bm25fField ───────────────────────────────────────────────────────────────

/// A single named field on a [`Bm25fDocument`] (e.g. `"title"`, `"body"`,
/// `"tags"`), holding that field's own text.
///
/// BM25F treats every distinctly-named field as its own miniature document
/// for the purposes of term-frequency counting and length normalisation: two
/// fields with the same `name` across different documents are compared
/// against each other (and against a per-field average length), never
/// against a field of a different name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bm25fField {
    /// The field's name, used to look up its
    /// [`Bm25fFieldWeight`] in the
    /// index configuration and to group its length statistics with
    /// same-named fields across the corpus.
    pub name: String,
    /// The field's raw text content.
    pub text: String,
}

impl Bm25fField {
    /// Create a new field with the given name and text.
    #[must_use]
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            text: text.into(),
        }
    }
}

// ── Bm25fDocument ────────────────────────────────────────────────────────────

/// A document composed of zero or more named [`Bm25fField`]s.
///
/// Unlike single-field BM25 (see the
/// [module documentation](crate::bm25f_retrieval)), a `Bm25fDocument` is not
/// one flat string: it is a small collection of independently-weighted,
/// independently-length-normalised fields. Not every document needs to carry
/// every field — a document missing a field simply contributes nothing from
/// that field to any term's pseudo-frequency, without needing special-case
/// handling by callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bm25fDocument {
    /// The document's unique identifier.
    pub id: String,
    /// The document's fields, in the order they were added. A field name
    /// may appear more than once; each occurrence is treated as its own
    /// additive contribution to that field name's combined statistics.
    pub fields: Vec<Bm25fField>,
}

impl Bm25fDocument {
    /// Create a new, field-less document with the given id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: Vec::new(),
        }
    }

    /// Append a field with the given name and text, returning `self` for
    /// chaining.
    #[must_use]
    pub fn with_field(mut self, name: impl Into<String>, text: impl Into<String>) -> Self {
        self.fields.push(Bm25fField::new(name, text));
        self
    }

    /// Return the text of the first field named `name`, or `None` if the
    /// document has no such field.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|field| field.name == name)
            .map(|field| field.text.as_str())
    }
}

// ── Bm25fFieldWeight ─────────────────────────────────────────────────────────

/// Per-field tuning for BM25F: an importance weight `w_f` and a
/// length-normalisation parameter `b_f`, applied to one named field.
///
/// This is the crux of what distinguishes BM25F from single-field BM25: each
/// field gets *its own* `b`, so (for example) a long title need not be
/// penalised for length the same way a long body is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25fFieldWeight {
    /// The field's importance weight `w_f`. Raw per-field term frequency is
    /// multiplied by this weight before being folded into the combined
    /// pseudo-frequency. A weight of `0.0` (or negative) excludes the field
    /// from scoring entirely, as if it did not exist.
    pub weight: f32,
    /// The field's length-normalisation parameter `b_f`, conventionally in
    /// `[0.0, 1.0]`. `0.0` disables length normalisation for this field
    /// entirely (a long field is never penalised for its length); `1.0`
    /// applies full length normalisation.
    pub b: f32,
}

impl Bm25fFieldWeight {
    /// Create a new field weight from an importance weight and a
    /// length-normalisation parameter.
    #[must_use]
    pub fn new(weight: f32, b: f32) -> Self {
        Self { weight, b }
    }
}

impl Default for Bm25fFieldWeight {
    /// `weight = 1.0`, `b = 0.75` — the conventional single-field BM25
    /// default for `b`, applied per-field.
    fn default() -> Self {
        Self {
            weight: 1.0,
            b: 0.75,
        }
    }
}

// ── Bm25fConfig ──────────────────────────────────────────────────────────────

/// Configuration for a [`Bm25fIndex`](crate::bm25f_retrieval::Bm25fIndex).
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25fConfig {
    /// The BM25 term-frequency saturation constant `k1`. Larger values let
    /// pseudo-frequency keep contributing marginal score for longer before
    /// saturating; `0.0` collapses every present term to the same
    /// contribution regardless of frequency. Defaults to `1.2`.
    pub k1: f32,
    /// Per-field-name overrides of [`Bm25fFieldWeight`]. A field name with
    /// no entry here falls back to [`Bm25fFieldWeight::default`] — callers
    /// need only configure the fields they want to tune away from the
    /// default. Uses a [`BTreeMap`] (rather than a hash map) purely so
    /// configuration equality and debug-printing are stable and
    /// order-independent.
    pub field_weights: BTreeMap<String, Bm25fFieldWeight>,
    /// Smoothing constant added to a term's document frequency inside the
    /// IDF formula, keeping IDF finite and strictly positive even for a
    /// term present in every document. Defaults to `0.5`.
    pub idf_smoothing: f32,
}

impl Default for Bm25fConfig {
    fn default() -> Self {
        Self {
            k1: 1.2,
            field_weights: BTreeMap::new(),
            idf_smoothing: 0.5,
        }
    }
}

impl Bm25fConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the BM25 saturation constant `k1`.
    #[must_use]
    pub fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    /// Override the [`Bm25fFieldWeight`] used for the field named `name`.
    #[must_use]
    pub fn with_field_weight(mut self, name: impl Into<String>, weight: Bm25fFieldWeight) -> Self {
        self.field_weights.insert(name.into(), weight);
        self
    }

    /// Set the IDF smoothing constant.
    #[must_use]
    pub fn with_idf_smoothing(mut self, idf_smoothing: f32) -> Self {
        self.idf_smoothing = idf_smoothing;
        self
    }

    /// Resolve the effective [`Bm25fFieldWeight`] for the field named
    /// `name`: the configured override if present, otherwise
    /// [`Bm25fFieldWeight::default`].
    #[must_use]
    pub fn field_weight(&self, name: &str) -> Bm25fFieldWeight {
        self.field_weights.get(name).copied().unwrap_or_default()
    }
}

// ── Bm25fHit ─────────────────────────────────────────────────────────────────

/// A single scored search result from a
/// [`Bm25fIndex`](crate::bm25f_retrieval::Bm25fIndex).
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25fHit {
    /// Identifier of the matched document.
    pub doc_id: String,
    /// The BM25F score against the query (higher is better). Guaranteed
    /// strictly positive — documents with no matching signal are omitted
    /// rather than reported at score `0.0`.
    pub score: f32,
}
