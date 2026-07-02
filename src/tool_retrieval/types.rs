//! Types for the `tool_retrieval` module.
//!
//! Defines the tool/parameter spec schema ([`ToolSpecEntry`] /
//! [`ToolParameter`] / [`ToolParameterType`]), the engine configuration
//! ([`ToolRetrievalConfig`]), the retrieval and grounding output types
//! ([`ToolMatch`] / [`GroundedArgument`] / [`ArgumentGroundingStatus`]), and
//! the module's error type ([`ToolRetrievalError`]).

use thiserror::Error;

// ── ToolParameterType ────────────────────────────────────────────────────────

/// The value type of a [`ToolParameter`].
///
/// A simple type tag — deliberately not a full JSON-schema type system — that
/// drives which extraction heuristic [`crate::tool_retrieval::ArgumentGrounder`]
/// applies to a parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParameterType {
    /// Free text: grounded from a quoted span, a capitalised/proper-noun-like
    /// span, or the phrase immediately following the parameter's name
    /// mention in the query.
    String,
    /// A number: grounded from the numeric token nearest to the parameter's
    /// name mention (or the first numeric token in the query, when the
    /// parameter is never named).
    Number,
    /// A boolean flag: grounded from yes/no/true/false-like cues near the
    /// parameter's name mention.
    Boolean,
    /// A categorical value restricted to a fixed, ordered set of allowed
    /// strings. Grounding matches case-insensitively and never returns a
    /// value outside this set.
    Enum(Vec<String>),
}

impl ToolParameterType {
    /// Human-readable, stable name for this type tag.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Boolean => "boolean",
            Self::Enum(_) => "enum",
        }
    }
}

// ── ToolParameter ─────────────────────────────────────────────────────────────

/// A single parameter of a [`ToolSpecEntry`]: a name, a value type, whether
/// it is required, and a natural-language description.
///
/// The description contributes (alongside the tool's own name and
/// description) to the tool's retrieval embedding — see
/// [`crate::tool_retrieval::ToolRetrievalIndex`] — so a well-described
/// parameter can itself help a query match the right tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolParameter {
    /// The parameter's name, as it would appear in a call to the tool.
    pub name: String,
    /// The parameter's value type.
    pub param_type: ToolParameterType,
    /// Whether the tool cannot be invoked without this parameter.
    pub required: bool,
    /// A natural-language description of the parameter's purpose.
    pub description: String,
}

impl ToolParameter {
    /// Create a new parameter with the given name and type.
    ///
    /// `required` defaults to `false` and `description` defaults to empty;
    /// use [`ToolParameter::with_required`] and
    /// [`ToolParameter::with_description`] to set them.
    #[must_use]
    pub fn new(name: impl Into<String>, param_type: ToolParameterType) -> Self {
        Self {
            name: name.into(),
            param_type,
            required: false,
            description: String::new(),
        }
    }

    /// Set whether this parameter is required (builder).
    #[must_use]
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set the parameter's natural-language description (builder).
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

// ── ToolSpecEntry ─────────────────────────────────────────────────────────────

/// The specification of a single tool: a name, a natural-language
/// description, and its list of [`ToolParameter`]s.
///
/// This is the unit indexed by [`crate::tool_retrieval::ToolRetrievalIndex`]
/// and matched by [`crate::tool_retrieval::ToolRetrievalEngine::retrieve`].
/// Unlike `agentic`'s bare `(name, description)` [`Tool`](crate::agentic::Tool)
/// trait, a `ToolSpecEntry` carries a full parameter list, which
/// [`crate::tool_retrieval::ArgumentGrounder`] uses to extract call
/// arguments from a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpecEntry {
    /// The tool's unique name.
    pub name: String,
    /// A natural-language description of what the tool does.
    pub description: String,
    /// The tool's parameters, in declaration order.
    pub parameters: Vec<ToolParameter>,
}

impl ToolSpecEntry {
    /// Create a new tool spec with no parameters.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
        }
    }

    /// Append a parameter (builder).
    #[must_use]
    pub fn with_parameter(mut self, parameter: ToolParameter) -> Self {
        self.parameters.push(parameter);
        self
    }

    /// Append several parameters (builder).
    #[must_use]
    pub fn with_parameters<I>(mut self, parameters: I) -> Self
    where
        I: IntoIterator<Item = ToolParameter>,
    {
        self.parameters.extend(parameters);
        self
    }

    /// Return every parameter marked [`ToolParameter::required`].
    #[must_use]
    pub fn required_parameters(&self) -> Vec<&ToolParameter> {
        self.parameters.iter().filter(|p| p.required).collect()
    }

    /// Look up one of this tool's parameters by name (case-sensitive, exact
    /// match).
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<&ToolParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }
}

// ── ToolRetrievalConfig ───────────────────────────────────────────────────────

/// Configuration for [`crate::tool_retrieval::ToolRetrievalIndex`] and
/// [`crate::tool_retrieval::ToolRetrievalEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToolRetrievalConfig {
    /// Dimensionality of the FNV-1a pseudo-embedding bucket histogram used
    /// for description-based semantic matching. Must be greater than zero.
    /// Defaults to `128`.
    pub embedding_dim: usize,
    /// Blend weight (in `[0.0, 1.0]`) given to the description-embedding
    /// cosine similarity when computing a [`ToolMatch::score`]; the
    /// remainder (`1.0 - description_weight`) is given to the lexical
    /// (Jaccard) overlap score. Defaults to `0.65`.
    pub description_weight: f32,
    /// The default number of results returned by
    /// [`crate::tool_retrieval::ToolRetrievalEngine::retrieve_default`].
    /// Must be greater than zero. Defaults to `5`.
    pub default_top_k: usize,
    /// The minimum blended [`ToolMatch::score`] a tool must reach to be
    /// included in a retrieval result. Defaults to `0.05`.
    pub min_match_score: f32,
}

impl Default for ToolRetrievalConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            description_weight: 0.65,
            default_top_k: 5,
            min_match_score: 0.05,
        }
    }
}

impl ToolRetrievalConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding dimensionality (builder).
    #[must_use]
    pub fn with_embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the description-vs-lexical blend weight (builder).
    #[must_use]
    pub fn with_description_weight(mut self, description_weight: f32) -> Self {
        self.description_weight = description_weight;
        self
    }

    /// Set the default top-k (builder).
    #[must_use]
    pub fn with_default_top_k(mut self, default_top_k: usize) -> Self {
        self.default_top_k = default_top_k;
        self
    }

    /// Set the minimum match score (builder).
    #[must_use]
    pub fn with_min_match_score(mut self, min_match_score: f32) -> Self {
        self.min_match_score = min_match_score;
        self
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// - [`ToolRetrievalError::InvalidEmbeddingDim`] when
    ///   [`ToolRetrievalConfig::embedding_dim`] is `0`.
    /// - [`ToolRetrievalError::InvalidDescriptionWeight`] when
    ///   [`ToolRetrievalConfig::description_weight`] is outside `[0.0, 1.0]`.
    /// - [`ToolRetrievalError::InvalidTopK`] when
    ///   [`ToolRetrievalConfig::default_top_k`] is `0`.
    pub fn validate(&self) -> ToolRetrievalResult<()> {
        if self.embedding_dim == 0 {
            return Err(ToolRetrievalError::InvalidEmbeddingDim);
        }
        if !(0.0..=1.0).contains(&self.description_weight) {
            return Err(ToolRetrievalError::InvalidDescriptionWeight(
                self.description_weight,
            ));
        }
        if self.default_top_k == 0 {
            return Err(ToolRetrievalError::InvalidTopK);
        }
        Ok(())
    }
}

// ── ToolMatch ─────────────────────────────────────────────────────────────────

/// A single ranked retrieval hit produced by
/// [`crate::tool_retrieval::ToolRetrievalEngine::retrieve`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToolMatch {
    /// The name of the matched [`ToolSpecEntry`].
    pub tool_name: String,
    /// The blended score used for ranking: `description_weight *
    /// description_score + (1.0 - description_weight) * lexical_score`.
    pub score: f32,
    /// Cosine similarity between the query's and the tool's FNV-1a
    /// pseudo-embeddings (built from name + description + parameter
    /// names/descriptions).
    pub description_score: f32,
    /// Jaccard overlap between the query's and the tool's content-word
    /// vocabularies.
    pub lexical_score: f32,
}

// ── ArgumentGroundingStatus / GroundedArgument ────────────────────────────────

/// The outcome of attempting to ground one [`ToolParameter`] from a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgumentGroundingStatus {
    /// A plausible value was found; [`GroundedArgument::value`] is `Some`.
    Grounded,
    /// No plausible value was found for a **required** parameter.
    /// [`GroundedArgument::value`] is `None` — the parameter is flagged, not
    /// silently dropped or given a fabricated default.
    UngroundedRequired,
    /// No plausible value was found for an **optional** parameter.
    /// [`GroundedArgument::value`] is `None`.
    UngroundedOptional,
}

impl ArgumentGroundingStatus {
    /// Return `true` for [`ArgumentGroundingStatus::Grounded`].
    #[must_use]
    pub fn is_grounded(&self) -> bool {
        matches!(self, Self::Grounded)
    }

    /// Return `true` for [`ArgumentGroundingStatus::UngroundedRequired`] —
    /// the case a caller must not silently ignore before invoking a tool.
    #[must_use]
    pub fn is_missing_required(&self) -> bool {
        matches!(self, Self::UngroundedRequired)
    }
}

/// The result of grounding one [`ToolParameter`] against a query, produced by
/// [`crate::tool_retrieval::ArgumentGrounder::ground`].
///
/// Every parameter of a tool produces exactly one `GroundedArgument`, whether
/// or not a value was found — see [`ArgumentGroundingStatus`].
#[derive(Debug, Clone, PartialEq)]
pub struct GroundedArgument {
    /// The name of the [`ToolParameter`] this argument was grounded for.
    pub parameter_name: String,
    /// The extracted value, normalised for the parameter's
    /// [`ToolParameterType`] (e.g. `Boolean` cues normalise to `"true"` /
    /// `"false"`). `None` when nothing plausible was found.
    pub value: Option<String>,
    /// Confidence in `[0.0, 1.0]` that [`GroundedArgument::value`] is
    /// correct. `0.0` when [`GroundedArgument::value`] is `None`.
    pub confidence: f32,
    /// The verbatim substring of the query that produced
    /// [`GroundedArgument::value`], for traceability. `None` when
    /// [`GroundedArgument::value`] is `None`.
    pub source_span: Option<String>,
    /// How this parameter was resolved.
    pub status: ArgumentGroundingStatus,
}

impl GroundedArgument {
    /// Return `true` when this argument was successfully grounded.
    #[must_use]
    pub fn is_grounded(&self) -> bool {
        self.status.is_grounded()
    }
}

// ── ToolRetrievalError / ToolRetrievalResult ─────────────────────────────────

/// Errors produced by the `tool_retrieval` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ToolRetrievalError {
    /// The supplied query was empty (after trimming).
    #[error("query must not be empty")]
    EmptyQuery,
    /// [`crate::tool_retrieval::ToolRetrievalIndex::build`] was called with
    /// no tool specs, or [`crate::tool_retrieval::ToolRetrievalEngine::retrieve`]
    /// was called against an index that holds none.
    #[error("tool registry must not be empty")]
    EmptyRegistry,
    /// [`ToolRetrievalConfig::embedding_dim`] was `0`.
    #[error("embedding dimension must be greater than zero")]
    InvalidEmbeddingDim,
    /// [`ToolRetrievalConfig::description_weight`] was outside `[0.0, 1.0]`.
    #[error("description weight must be within [0.0, 1.0], got {0}")]
    InvalidDescriptionWeight(f32),
    /// [`ToolRetrievalConfig::default_top_k`] was `0`.
    #[error("default_top_k must be greater than zero")]
    InvalidTopK,
    /// A [`ToolSpecEntry::name`] was empty (after trimming).
    #[error("tool spec name must not be empty")]
    EmptyToolName,
    /// Two or more [`ToolSpecEntry`] values in the same registry shared the
    /// same name (case-insensitive).
    #[error("duplicate tool name in registry: {0}")]
    DuplicateToolName(String),
}

/// Convenience alias for `Result<T, ToolRetrievalError>`.
pub type ToolRetrievalResult<T> = Result<T, ToolRetrievalError>;
