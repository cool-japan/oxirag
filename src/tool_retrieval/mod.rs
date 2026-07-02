//! Tool/API spec retrieval and argument grounding.
//!
//! # How this differs from [`agentic`](crate::agentic)
//!
//! `crate::agentic` already ships a [`Tool`](crate::agentic::Tool) trait, a
//! [`ToolRegistry`](crate::agentic::ToolRegistry), and — inside
//! `ReActAgent`'s heuristic planner — tool *selection*. But that selection is
//! a crude, single-shot name-SUBSTRING test
//! (`query.to_lowercase().contains(&tool_name.to_lowercase())`, see
//! `agentic::agent::plan_action`): it has no notion of a tool's parameters at
//! all (no JSON-schema-like parameter list — `Tool` exposes only `name()` and
//! `description()`), no semantic matching against a tool's *description*
//! (only the bare name is ever compared against the query), and no attempt to
//! extract concrete argument values from the query — whatever text follows
//! the matched tool's name is handed to the tool verbatim as one opaque
//! `input: String`.
//!
//! `tool_retrieval` is a standalone layer that plugs both gaps. It is **not**
//! a reimplementation of tool invocation — it never calls a tool, only helps
//! a caller decide *which* tool to call and *what arguments* to call it with:
//!
//! 1. **Semantic spec retrieval** ([`ToolRetrievalIndex`] /
//!    [`ToolRetrievalEngine::retrieve`]). Every [`ToolSpecEntry`] — name,
//!    natural-language description, and a full [`ToolParameter`] list — is
//!    embedded via a deterministic FNV-1a pseudo-embedding of its *combined*
//!    name + description + parameter-names/descriptions text, not just its
//!    name. A query is ranked against every indexed tool by a blend of
//!    embedding cosine similarity and lexical (Jaccard) overlap, so a tool
//!    whose *description* — not its name — matches the query's intent is
//!    still found correctly; the module tests include a tool with a
//!    deliberately unhelpful name and a specific description to prove this.
//! 2. **Argument grounding** ([`ArgumentGrounder`] /
//!    [`ToolRetrievalEngine::ground`]). Given a query and a matched tool's
//!    parameter list, each parameter is independently scanned for a
//!    plausible value appropriate to its [`ToolParameterType`]: a quoted or
//!    capitalised/proper-noun-like span (or the phrase right after the
//!    parameter's name mention) for `String`; the nearest numeric token for
//!    `Number`; yes/no/true/false-like cues for `Boolean`; a case-insensitive
//!    match against the allowed set for `Enum`. Every parameter yields
//!    exactly one [`GroundedArgument`], carrying a confidence and a source
//!    span — including parameters the query gives no plausible value for,
//!    which are honestly flagged via
//!    [`ArgumentGroundingStatus::UngroundedRequired`] /
//!    [`ArgumentGroundingStatus::UngroundedOptional`] rather than silently
//!    dropped or given a fabricated default.
//!
//! See [`self_query`](crate::self_query) for a related but distinct concern:
//! `self_query` extracts a *metadata filter* (field/op/value triples) from a
//! query to narrow a document search. `tool_retrieval` extracts *tool call
//! arguments* (by parameter name and type) from a query to prepare a tool
//! invocation. Different target schema, different extraction goal, no code
//! sharing between the two.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ToolParameterType`] | `String` / `Number` / `Boolean` / `Enum(allowed values)` type tag |
//! | [`ToolParameter`] | One parameter's name, type, required flag, description |
//! | [`ToolSpecEntry`] | A tool's name, description, and parameter list |
//! | [`ToolRetrievalConfig`] | Embedding dim, description/lexical blend, top-k, min score |
//! | [`ToolRetrievalIndex`] | Embeds and stores a registry of [`ToolSpecEntry`] |
//! | [`ToolMatch`] | A ranked retrieval hit (tool name + blended/component scores) |
//! | [`ArgumentGrounder`] | Extracts [`GroundedArgument`]s from a query + parameter list |
//! | [`ArgumentGroundingStatus`] | Grounded / ungrounded-required / ungrounded-optional |
//! | [`GroundedArgument`] | One parameter's extracted value, confidence, and source span |
//! | [`ToolRetrievalEngine`] | Ties indexing, retrieval, and grounding together |
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "tool-retrieval")]
//! # {
//! use oxirag::tool_retrieval::{
//!     ToolParameter, ToolParameterType, ToolRetrievalConfig, ToolRetrievalEngine, ToolSpecEntry,
//! };
//!
//! // A deliberately unhelpful tool *name*; only the *description* hints at
//! // what it does.
//! let weather = ToolSpecEntry::new("util_seven", "Looks up the current weather for a named city")
//!     .with_parameter(
//!         ToolParameter::new("city", ToolParameterType::String)
//!             .with_required(true)
//!             .with_description("the city to look up"),
//!     )
//!     .with_parameter(ToolParameter::new(
//!         "units",
//!         ToolParameterType::Enum(vec!["celsius".to_string(), "fahrenheit".to_string()]),
//!     ));
//! let calculator = ToolSpecEntry::new("tool_two", "Evaluates an arithmetic expression");
//!
//! let engine = ToolRetrievalEngine::build(vec![weather, calculator], ToolRetrievalConfig::default())
//!     .expect("registry is non-empty");
//!
//! // The query matches `util_seven`'s *description*, not its name.
//! let matches = engine
//!     .retrieve("what is the current weather for Paris", 2)
//!     .expect("retrieval succeeds");
//! assert_eq!(matches[0].tool_name, "util_seven");
//!
//! let spec = engine.index().get("util_seven").expect("tool is indexed");
//! let grounded = engine
//!     .ground("what is the current weather for Paris in fahrenheit", spec)
//!     .expect("grounding succeeds");
//! assert!(
//!     grounded
//!         .iter()
//!         .any(|g| g.parameter_name == "city" && g.value.as_deref() == Some("Paris"))
//! );
//! assert!(
//!     grounded
//!         .iter()
//!         .any(|g| g.parameter_name == "units" && g.value.as_deref() == Some("fahrenheit"))
//! );
//! # }
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::{ArgumentGrounder, ToolRetrievalEngine, ToolRetrievalIndex};
pub use types::{
    ArgumentGroundingStatus, GroundedArgument, ToolMatch, ToolParameter, ToolParameterType,
    ToolRetrievalConfig, ToolRetrievalError, ToolRetrievalResult, ToolSpecEntry,
};
