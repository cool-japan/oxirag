//! Core types for RQ-RAG query refinement.
//!
//! This module provides the building blocks for classifying a query into one
//! of four explicit *refinement actions* and describing the resulting
//! refinement outcome. It also carries the deterministic marker word tables
//! consulted by [`super::MockRefiner`]'s heuristic classifier.

use thiserror::Error;

// ── AMBIGUITY_SENSE_TABLE ─────────────────────────────────────────────────────

/// Curated table of polysemous (multi-sense) terms and their plausible
/// readings.
///
/// Each entry pairs a lowercase trigger term with the distinct senses in
/// which it commonly appears. When [`super::MockRefiner::disambiguate`]
/// detects one of these terms in a query, it emits one refined variant per
/// listed sense, faithfully modelling RQ-RAG's "ambiguous entity/term with
/// multiple plausible readings" refinement action.
///
/// The trigger terms in this table are also used as the default value of
/// [`RqRagConfig::ambiguity_markers`].
pub static AMBIGUITY_SENSE_TABLE: &[(&str, &[&str])] = &[
    ("bank", &["financial institution", "river bank"]),
    (
        "mercury",
        &["planet", "chemical element", "Roman messenger god"],
    ),
    ("python", &["programming language", "snake"]),
    (
        "java",
        &[
            "programming language",
            "Indonesian island",
            "type of coffee",
        ],
    ),
    ("amazon", &["e-commerce company", "river", "rainforest"]),
    ("apple", &["technology company", "fruit"]),
    ("jaguar", &["car manufacturer", "big cat"]),
    (
        "mars",
        &["planet", "chocolate bar brand", "Roman god of war"],
    ),
    ("saturn", &["planet", "Roman god", "car brand"]),
    (
        "bass",
        &["fish", "low-frequency sound", "musical instrument"],
    ),
    (
        "spring",
        &["season", "coiled mechanical device", "water source"],
    ),
    ("crane", &["bird", "construction machine"]),
    ("bat", &["flying mammal", "sports equipment"]),
    ("seal", &["marine mammal", "official stamp or emblem"]),
];

// ── DEFAULT_COLLOQUIAL_MARKERS ────────────────────────────────────────────────

/// Filler words and informal phrases that signal a query needs rephrasing
/// before it is suitable as a search query.
///
/// This is the default value of [`RqRagConfig::colloquial_markers`].
pub static DEFAULT_COLLOQUIAL_MARKERS: &[&str] = &[
    "kinda",
    "sorta",
    "gonna",
    "wanna",
    "gotta",
    "y'know",
    "you know",
    "like",
    "stuff",
    "basically",
    "literally",
    "um",
    "uh",
    "yeah",
    "dunno",
    "ain't",
    "guys",
    "whatnot",
    "pretty much",
    "sort of",
    "kind of",
    "i mean",
    "anyway",
    "honestly",
];

// ── RefinementAction ──────────────────────────────────────────────────────────

/// The refinement action assigned to a query by a [`super::QueryRefiner`].
///
/// RQ-RAG (Chan et al., 2024) explicitly classifies every incoming query into
/// exactly one of these four actions before any retrieval takes place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefinementAction {
    /// The query is a reasonable search target but its phrasing is ambiguous,
    /// colloquial, or otherwise ill-suited for retrieval. Produces exactly one
    /// rewritten query.
    Rewrite,
    /// The query bundles several independent questions (a compound / multi-part
    /// question). Produces one sub-query per part.
    Decompose,
    /// The query contains an entity or term with multiple plausible readings.
    /// Produces one disambiguated variant per reading.
    Disambiguate,
    /// The query is already clear and atomic; no refinement is necessary.
    Respond,
}

impl RefinementAction {
    /// Returns a stable lower-case identifier for the action.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rewrite => "rewrite",
            Self::Decompose => "decompose",
            Self::Disambiguate => "disambiguate",
            Self::Respond => "respond",
        }
    }
}

impl std::fmt::Display for RefinementAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── RefinementPlan ────────────────────────────────────────────────────────────

/// The complete output of [`super::QueryRefinementEngine::run`].
///
/// Carries the original query, the classified [`RefinementAction`], and the
/// concrete refined queries produced for that action. For
/// [`RefinementAction::Respond`], `refined_queries` is always a single-element
/// vector containing a clone of `original_query`.
#[derive(Debug, Clone, PartialEq)]
pub struct RefinementPlan {
    /// The original, trimmed query text.
    pub original_query: String,
    /// The refinement action classified for `original_query`.
    pub action: RefinementAction,
    /// The refined query text(s) to retrieve against.
    ///
    /// Never empty: [`Respond`](RefinementAction::Respond) yields a single
    /// unchanged entry, [`Rewrite`](RefinementAction::Rewrite) yields exactly
    /// one entry, and [`Decompose`](RefinementAction::Decompose) /
    /// [`Disambiguate`](RefinementAction::Disambiguate) yield two or more.
    pub refined_queries: Vec<String>,
}

impl RefinementPlan {
    /// Returns the number of refined queries produced.
    #[must_use]
    pub fn query_count(&self) -> usize {
        self.refined_queries.len()
    }

    /// Returns `true` if exactly one refined query was produced.
    #[must_use]
    pub fn is_single(&self) -> bool {
        self.refined_queries.len() == 1
    }
}

// ── RqRagConfig ───────────────────────────────────────────────────────────────

/// Configuration for the RQ-RAG [`super::QueryRefinementEngine`] and
/// [`super::MockRefiner`].
///
/// | Field | Default |
/// |-------|---------|
/// | `min_decompose_parts` | `2` |
/// | `ambiguity_markers` | keys of [`AMBIGUITY_SENSE_TABLE`] |
/// | `colloquial_markers` | [`DEFAULT_COLLOQUIAL_MARKERS`] |
#[derive(Debug, Clone, PartialEq)]
pub struct RqRagConfig {
    /// Minimum number of clauses a query must split into before it is
    /// classified as [`RefinementAction::Decompose`].
    ///
    /// Must be `>= 2` since decomposition by definition produces multiple
    /// sub-queries; see [`RqRagConfig::validate`].
    ///
    /// Default: `2`.
    pub min_decompose_parts: usize,
    /// Word-boundary-matched trigger terms that signal an ambiguous
    /// entity/term requiring disambiguation.
    ///
    /// Default: the trigger terms in [`AMBIGUITY_SENSE_TABLE`].
    pub ambiguity_markers: Vec<String>,
    /// Word-boundary-matched filler words/phrases that signal colloquial
    /// phrasing requiring a rewrite.
    ///
    /// Default: [`DEFAULT_COLLOQUIAL_MARKERS`].
    pub colloquial_markers: Vec<String>,
}

impl Default for RqRagConfig {
    fn default() -> Self {
        Self {
            min_decompose_parts: 2,
            ambiguity_markers: AMBIGUITY_SENSE_TABLE
                .iter()
                .map(|(term, _)| (*term).to_string())
                .collect(),
            colloquial_markers: DEFAULT_COLLOQUIAL_MARKERS
                .iter()
                .map(|marker| (*marker).to_string())
                .collect(),
        }
    }
}

impl RqRagConfig {
    /// Creates a new [`RqRagConfig`] with the default marker tables.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the minimum decompose-part threshold.
    #[must_use]
    pub fn with_min_decompose_parts(mut self, v: usize) -> Self {
        self.min_decompose_parts = v;
        self
    }

    /// Replaces the ambiguity marker table entirely.
    #[must_use]
    pub fn with_ambiguity_markers(mut self, v: Vec<String>) -> Self {
        self.ambiguity_markers = v;
        self
    }

    /// Replaces the colloquial marker table entirely.
    #[must_use]
    pub fn with_colloquial_markers(mut self, v: Vec<String>) -> Self {
        self.colloquial_markers = v;
        self
    }

    /// Appends a single custom ambiguity marker.
    #[must_use]
    pub fn with_extra_ambiguity_marker(mut self, marker: impl Into<String>) -> Self {
        self.ambiguity_markers.push(marker.into());
        self
    }

    /// Appends a single custom colloquial marker.
    #[must_use]
    pub fn with_extra_colloquial_marker(mut self, marker: impl Into<String>) -> Self {
        self.colloquial_markers.push(marker.into());
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RqRagError::InvalidConfig`] if `min_decompose_parts < 2`,
    /// since [`RefinementAction::Decompose`] is defined as producing
    /// *multiple* sub-queries.
    pub fn validate(&self) -> Result<(), RqRagError> {
        if self.min_decompose_parts < 2 {
            return Err(RqRagError::InvalidConfig(format!(
                "min_decompose_parts must be >= 2 (got {})",
                self.min_decompose_parts
            )));
        }
        Ok(())
    }
}

// ── RqRagError ────────────────────────────────────────────────────────────────

/// Errors produced by the `rq_rag` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RqRagError {
    /// The supplied query was empty or contained only whitespace.
    #[error("query is empty")]
    EmptyQuery,
    /// The supplied [`RqRagConfig`] failed validation.
    #[error("invalid rq-rag configuration: {0}")]
    InvalidConfig(String),
}
