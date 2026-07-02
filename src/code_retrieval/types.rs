//! Types, configuration, and errors for the `code_retrieval` module.
//!
//! `code_retrieval` performs **AST/structure-aware code search** over a corpus
//! of source-code items. A lightweight, deterministic structural tokenizer
//! (see [`parser`](crate::code_retrieval::parser)) breaks each item into
//! [`CodeRetrievalUnit`]s — one per detected function/method definition, plus a
//! whole-file fallback when none is found — recording each unit's identifier
//! set ([`CodeRetrievalSymbol`]s with occurrence counts), imported
//! module/paths, and heuristic call-sites. Search blends a **structural**
//! similarity (Jaccard overlap of identifiers, imports, and call-sites) with a
//! **textual** similarity (cosine of a deterministic FNV-1a pseudo-embedding)
//! into a single, config-weighted score.
//!
//! This file defines:
//!
//! * [`CodeRetrievalLanguageHint`] — brace-vs-indentation syntax family.
//! * [`CodeRetrievalUnitKind`] — whether a unit is a detected function or a
//!   whole-file fallback.
//! * [`CodeRetrievalSymbol`] — one distinct identifier plus its occurrence
//!   count.
//! * [`CodeRetrievalUnit`] — one structural unit of code.
//! * [`CodeRetrievalHit`] — one ranked search result with its score breakdown.
//! * [`CodeRetrievalResult`] — the full result of a search.
//! * [`CodeRetrievalConfig`] — search/parse configuration.
//! * [`CodeRetrievalError`] — the module's error type.

use std::collections::HashSet;

use thiserror::Error;

// ── CodeRetrievalLanguageHint ────────────────────────────────────────────────

/// The syntactic family a source item is treated as, which selects how the
/// structural tokenizer detects function bodies.
///
/// This is a *hint*, not a real grammar: the tokenizer is a heuristic scanner,
/// not a parser for any specific language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodeRetrievalLanguageHint {
    /// Detect the family from the item's content: if any `{` brace is present
    /// the item is treated as [`CodeRetrievalLanguageHint::BraceDelimited`],
    /// otherwise, if a `def ...:` header is present, as
    /// [`CodeRetrievalLanguageHint::IndentDelimited`]; failing both it falls
    /// back to brace-delimited scanning. This is the default.
    #[default]
    Auto,
    /// Brace-delimited bodies (C, C++, Rust, Java, Go, JavaScript, …): a
    /// function body is the region between a `{` and its matching `}`.
    BraceDelimited,
    /// Indentation-delimited bodies (Python-like): a function body is the run
    /// of lines indented more deeply than a `def NAME(...):` header.
    IndentDelimited,
}

impl CodeRetrievalLanguageHint {
    /// Return a stable lowercase string representation.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::BraceDelimited => "brace",
            Self::IndentDelimited => "indent",
        }
    }

    /// Return `true` for [`CodeRetrievalLanguageHint::Auto`].
    #[must_use]
    pub fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }
}

// ── CodeRetrievalUnitKind ────────────────────────────────────────────────────

/// Whether a [`CodeRetrievalUnit`] is a detected function/method definition or
/// a whole-file fallback produced when no function was detected in an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodeRetrievalUnitKind {
    /// A detected function or method definition, with a name and a body span.
    #[default]
    Function,
    /// A whole-file fallback: no function/method was detected in the source
    /// item, so the entire item is indexed as one unit.
    WholeFile,
}

impl CodeRetrievalUnitKind {
    /// Return a stable lowercase string representation.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::WholeFile => "whole_file",
        }
    }
}

// ── CodeRetrievalSymbol ──────────────────────────────────────────────────────

/// One distinct identifier appearing in a [`CodeRetrievalUnit`], together with
/// how many times it occurred.
///
/// Identifiers are lowercased so that structural matching is case-insensitive
/// (a query mentioning `readtoken` matches a `readToken` call). Language
/// keywords (per [`CodeRetrievalConfig::keywords`]), tokens shorter than
/// [`CodeRetrievalConfig::min_identifier_len`], and purely numeric tokens are
/// excluded before symbols are formed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeRetrievalSymbol {
    /// The lowercased identifier text.
    pub name: String,
    /// The number of times this identifier occurred in the unit's code
    /// (comments excluded).
    pub occurrences: usize,
}

impl CodeRetrievalSymbol {
    /// Construct a symbol from a `name` and an `occurrences` count.
    #[must_use]
    pub fn new(name: impl Into<String>, occurrences: usize) -> Self {
        Self {
            name: name.into(),
            occurrences,
        }
    }
}

// ── CodeRetrievalUnit ────────────────────────────────────────────────────────

/// One structural unit of code extracted from a corpus item: either a detected
/// function/method definition or a whole-file fallback.
///
/// A unit carries its purely *structural* features — identifier
/// [`symbols`](CodeRetrievalUnit::symbols), [`imports`](CodeRetrievalUnit::imports),
/// and [`call_sites`](CodeRetrievalUnit::call_sites) — and its raw source
/// [`raw_text`](CodeRetrievalUnit::raw_text). The FNV-1a pseudo-embedding used
/// for textual similarity is *not* stored here; it is derived from `raw_text`
/// by [`CodeRetrievalEngine`](crate::code_retrieval::CodeRetrievalEngine) at
/// index time and kept inside the [`CodeRetrievalIndex`](crate::code_retrieval::CodeRetrievalIndex).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeRetrievalUnit {
    /// Identifier of the corpus item this unit was extracted from.
    pub source_id: String,
    /// Zero-based position of this unit among the units extracted from the
    /// same corpus item (in source order).
    pub unit_index: usize,
    /// The function/method name, or the corpus item's `source_id` for a
    /// [`CodeRetrievalUnitKind::WholeFile`] fallback unit.
    pub name: String,
    /// Whether this unit is a detected function or a whole-file fallback.
    pub kind: CodeRetrievalUnitKind,
    /// The number of comma-separated parameters at the top nesting level of
    /// the signature's parameter list (`0` for `()` and for whole-file units).
    pub param_count: usize,
    /// Zero-based line (within the corpus item) where this unit's span begins
    /// (the signature/header line for a function).
    pub body_start_line: usize,
    /// Zero-based line (within the corpus item) where this unit's span ends,
    /// inclusive.
    pub body_end_line: usize,
    /// The resolved syntactic family this unit was scanned as (never
    /// [`CodeRetrievalLanguageHint::Auto`]).
    pub language: CodeRetrievalLanguageHint,
    /// Distinct identifiers occurring in the unit's code, with occurrence
    /// counts, sorted ascending by [`CodeRetrievalSymbol::name`].
    pub symbols: Vec<CodeRetrievalSymbol>,
    /// Imported module/paths referenced by the containing corpus item (shared
    /// by every unit from that item), sorted ascending and de-duplicated.
    pub imports: Vec<String>,
    /// Lowercased names that are *called* within this unit's body (an
    /// identifier immediately followed by `(`), sorted ascending and
    /// de-duplicated. Control- and definition-keyword heads such as `if`,
    /// `while`, and `fn` are excluded.
    pub call_sites: Vec<String>,
    /// The raw source text of this unit's span, comments included.
    pub raw_text: String,
}

impl CodeRetrievalUnit {
    /// Return `true` when this unit is a detected function/method definition.
    #[must_use]
    pub fn is_function(&self) -> bool {
        matches!(self.kind, CodeRetrievalUnitKind::Function)
    }

    /// Return `true` when this unit is a whole-file fallback.
    #[must_use]
    pub fn is_whole_file(&self) -> bool {
        matches!(self.kind, CodeRetrievalUnitKind::WholeFile)
    }

    /// The set of distinct identifier names in this unit (occurrence counts
    /// dropped), suitable for Jaccard overlap.
    #[must_use]
    pub fn identifier_set(&self) -> HashSet<String> {
        self.symbols.iter().map(|s| s.name.clone()).collect()
    }

    /// The occurrence count of `name` (already lowercased by the tokenizer),
    /// or `0` if the identifier does not appear in this unit.
    #[must_use]
    pub fn occurrences_of(&self, name: &str) -> usize {
        self.symbols
            .iter()
            .find(|s| s.name == name)
            .map_or(0, |s| s.occurrences)
    }

    /// The number of lines spanned by this unit (inclusive of both ends).
    #[must_use]
    pub fn line_span(&self) -> usize {
        self.body_end_line.saturating_sub(self.body_start_line) + 1
    }
}

// ── CodeRetrievalHit ─────────────────────────────────────────────────────────

/// One ranked search result: a matched [`CodeRetrievalUnit`] together with the
/// full breakdown of how its blended score was computed.
///
/// The breakdown is exposed for transparency and debuggability: `score` is the
/// blend of `structural_score` and `text_score` under
/// [`CodeRetrievalConfig::blend_weight`], and the three `*_overlap` fields are
/// the individual structural sub-signals that fed `structural_score`.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeRetrievalHit {
    /// The matched unit.
    pub unit: CodeRetrievalUnit,
    /// The final blended score: `blend_weight * structural_score + (1 -
    /// blend_weight) * text_score`.
    pub score: f32,
    /// The structural-similarity component in `[0.0, 1.0]` (the mean of the
    /// enabled structural sub-signals), before blending.
    pub structural_score: f32,
    /// The textual-similarity component in `[0.0, 1.0]` (pseudo-embedding
    /// cosine), before blending.
    pub text_score: f32,
    /// Jaccard overlap of the query's and unit's identifier sets.
    pub identifier_overlap: f32,
    /// Jaccard overlap of the query's and unit's import sets.
    pub import_overlap: f32,
    /// Jaccard overlap of the query's and unit's call-site sets.
    pub call_site_overlap: f32,
}

// ── CodeRetrievalResult ──────────────────────────────────────────────────────

/// The complete result of a [`CodeRetrievalEngine`](crate::code_retrieval::CodeRetrievalEngine)
/// search: the echoed query, the ranked hits (already truncated to
/// [`CodeRetrievalConfig::top_k`]), and how many units were scored.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeRetrievalResult {
    /// The original query text.
    pub query: String,
    /// The ranked hits, best-first.
    pub hits: Vec<CodeRetrievalHit>,
    /// The number of units that were scored (the full index size), before
    /// truncation to `top_k`.
    pub units_searched: usize,
}

impl CodeRetrievalResult {
    /// Return `true` when no hits were produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// The number of returned hits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hits.len()
    }

    /// The single best hit, or `None` when there were no hits.
    #[must_use]
    pub fn top(&self) -> Option<&CodeRetrievalHit> {
        self.hits.first()
    }

    /// The score of the best hit, or `0.0` when there were no hits.
    #[must_use]
    pub fn best_score(&self) -> f32 {
        self.hits.first().map_or(0.0, |h| h.score)
    }
}

// ── CodeRetrievalConfig ──────────────────────────────────────────────────────

/// Default language keywords excluded from identifier sets.
///
/// A broad, deliberately conservative union of common keywords and primitive
/// type names across C, C++, Rust, Java, Go, JavaScript, and Python. The list
/// is used *only* for identifier extraction; it does not affect call-site
/// detection (which excludes its own control/definition keyword heads).
const DEFAULT_KEYWORDS: &[&str] = &[
    "abstract",
    "and",
    "as",
    "async",
    "await",
    "bool",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "def",
    "default",
    "do",
    "double",
    "elif",
    "else",
    "enum",
    "export",
    "extends",
    "extern",
    "false",
    "final",
    "finally",
    "float",
    "fn",
    "for",
    "from",
    "func",
    "function",
    "global",
    "goto",
    "if",
    "impl",
    "implements",
    "import",
    "in",
    "int",
    "interface",
    "is",
    "let",
    "long",
    "loop",
    "match",
    "mod",
    "module",
    "move",
    "mut",
    "new",
    "none",
    "not",
    "null",
    "or",
    "package",
    "pass",
    "private",
    "protected",
    "pub",
    "public",
    "raise",
    "ref",
    "return",
    "self",
    "short",
    "signed",
    "sizeof",
    "static",
    "str",
    "string",
    "struct",
    "super",
    "switch",
    "this",
    "throw",
    "throws",
    "trait",
    "true",
    "try",
    "type",
    "typedef",
    "typeof",
    "union",
    "unsafe",
    "unsigned",
    "use",
    "var",
    "void",
    "volatile",
    "while",
    "with",
    "yield",
];

/// Configuration for parsing and searching a code corpus.
///
/// The parse-time fields ([`embedding_dim`](CodeRetrievalConfig::embedding_dim),
/// [`min_identifier_len`](CodeRetrievalConfig::min_identifier_len),
/// [`keywords`](CodeRetrievalConfig::keywords),
/// [`language_hint`](CodeRetrievalConfig::language_hint)) determine how corpus
/// items *and* queries are tokenized and embedded, and are stamped into the
/// [`CodeRetrievalIndex`](crate::code_retrieval::CodeRetrievalIndex) at build
/// time. The score-time fields ([`blend_weight`](CodeRetrievalConfig::blend_weight),
/// [`top_k`](CodeRetrievalConfig::top_k), and the `use_*` signal toggles) are
/// read at search time, so a single index can be searched under several blend
/// weights without re-indexing.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeRetrievalConfig {
    /// Blend weight in `[0.0, 1.0]` combining the two similarity components:
    /// `score = blend_weight * structural + (1 - blend_weight) * text`. `1.0`
    /// is structural-only; `0.0` is text-only. Defaults to `0.5`.
    pub blend_weight: f32,
    /// Minimum length (in characters) for a token to be admitted to a unit's
    /// identifier set. Defaults to `2`.
    pub min_identifier_len: usize,
    /// Dimensionality of the FNV-1a bag-of-tokens pseudo-embedding used for
    /// textual similarity. Must be greater than zero. Defaults to `256`.
    pub embedding_dim: usize,
    /// Maximum number of hits returned by a search. Defaults to `10`.
    pub top_k: usize,
    /// Whether the identifier-set Jaccard sub-signal contributes to the
    /// structural score. Defaults to `true`.
    pub use_identifiers: bool,
    /// Whether the shared-import Jaccard sub-signal contributes to the
    /// structural score. Defaults to `true`.
    pub use_imports: bool,
    /// Whether the call-site Jaccard sub-signal contributes to the structural
    /// score. Defaults to `true`.
    pub use_call_sites: bool,
    /// The syntactic family to scan corpus items and queries as. Defaults to
    /// [`CodeRetrievalLanguageHint::Auto`].
    pub language_hint: CodeRetrievalLanguageHint,
    /// Keywords (lowercased) excluded from identifier sets. Defaults to a
    /// broad multi-language union; see the module documentation.
    pub keywords: Vec<String>,
}

impl Default for CodeRetrievalConfig {
    fn default() -> Self {
        Self {
            blend_weight: 0.5,
            min_identifier_len: 2,
            embedding_dim: 256,
            top_k: 10,
            use_identifiers: true,
            use_imports: true,
            use_call_sites: true,
            language_hint: CodeRetrievalLanguageHint::Auto,
            keywords: DEFAULT_KEYWORDS.iter().map(|k| (*k).to_string()).collect(),
        }
    }
}

impl CodeRetrievalConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the structural-vs-text [`blend_weight`](CodeRetrievalConfig::blend_weight).
    #[must_use]
    pub fn with_blend_weight(mut self, blend_weight: f32) -> Self {
        self.blend_weight = blend_weight;
        self
    }

    /// Set the [`min_identifier_len`](CodeRetrievalConfig::min_identifier_len).
    #[must_use]
    pub fn with_min_identifier_len(mut self, min_identifier_len: usize) -> Self {
        self.min_identifier_len = min_identifier_len;
        self
    }

    /// Set the pseudo-embedding [`embedding_dim`](CodeRetrievalConfig::embedding_dim).
    #[must_use]
    pub fn with_embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the [`top_k`](CodeRetrievalConfig::top_k) hit cap.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Enable or disable the identifier-set structural sub-signal.
    #[must_use]
    pub fn with_use_identifiers(mut self, use_identifiers: bool) -> Self {
        self.use_identifiers = use_identifiers;
        self
    }

    /// Enable or disable the shared-import structural sub-signal.
    #[must_use]
    pub fn with_use_imports(mut self, use_imports: bool) -> Self {
        self.use_imports = use_imports;
        self
    }

    /// Enable or disable the call-site structural sub-signal.
    #[must_use]
    pub fn with_use_call_sites(mut self, use_call_sites: bool) -> Self {
        self.use_call_sites = use_call_sites;
        self
    }

    /// Set all three structural sub-signal toggles at once.
    #[must_use]
    pub fn with_signals(
        mut self,
        use_identifiers: bool,
        use_imports: bool,
        use_call_sites: bool,
    ) -> Self {
        self.use_identifiers = use_identifiers;
        self.use_imports = use_imports;
        self.use_call_sites = use_call_sites;
        self
    }

    /// Set the [`language_hint`](CodeRetrievalConfig::language_hint).
    #[must_use]
    pub fn with_language_hint(mut self, language_hint: CodeRetrievalLanguageHint) -> Self {
        self.language_hint = language_hint;
        self
    }

    /// Replace the identifier-exclusion [`keywords`](CodeRetrievalConfig::keywords)
    /// list. Entries are lowercased when compared against identifier tokens.
    #[must_use]
    pub fn with_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.keywords = keywords.into_iter().map(Into::into).collect();
        self
    }

    /// Whether at least one structural sub-signal is enabled.
    #[must_use]
    pub fn any_structural_signal(&self) -> bool {
        self.use_identifiers || self.use_imports || self.use_call_sites
    }

    /// Validate the parse/score-time invariants of this configuration.
    ///
    /// # Errors
    ///
    /// * [`CodeRetrievalError::ZeroEmbeddingDim`] when
    ///   [`embedding_dim`](CodeRetrievalConfig::embedding_dim) is zero.
    /// * [`CodeRetrievalError::InvalidBlendWeight`] when
    ///   [`blend_weight`](CodeRetrievalConfig::blend_weight) is not a finite
    ///   value in `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), CodeRetrievalError> {
        if self.embedding_dim == 0 {
            return Err(CodeRetrievalError::ZeroEmbeddingDim);
        }
        if !self.blend_weight.is_finite() || !(0.0..=1.0).contains(&self.blend_weight) {
            return Err(CodeRetrievalError::InvalidBlendWeight {
                weight: self.blend_weight,
            });
        }
        Ok(())
    }
}

// ── CodeRetrievalError ───────────────────────────────────────────────────────

/// Errors produced by the `code_retrieval` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CodeRetrievalError {
    /// A search was issued with an empty (or whitespace-only) query.
    #[error("query must not be empty")]
    EmptyQuery,
    /// An index build was attempted over a corpus with no items.
    #[error("cannot build an index from an empty corpus")]
    EmptyCorpus,
    /// The configured embedding dimension was zero.
    #[error("embedding dimension must be greater than zero")]
    ZeroEmbeddingDim,
    /// The blend weight was outside the valid `[0.0, 1.0]` range (or was not a
    /// finite value).
    #[error("blend weight must be a finite value in [0.0, 1.0], got {weight}")]
    InvalidBlendWeight {
        /// The offending blend weight.
        weight: f32,
    },
    /// A search config's embedding dimension did not match the index it was
    /// used against, so their pseudo-embeddings are not comparable.
    #[error(
        "index was built with embedding dimension {index_dim}, \
         but the search config uses {config_dim}"
    )]
    DimensionMismatch {
        /// The dimension the index was built with.
        index_dim: usize,
        /// The dimension the search configuration requested.
        config_dim: usize,
    },
}
