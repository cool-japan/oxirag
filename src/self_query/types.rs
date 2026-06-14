//! Types for the `self_query` module.
//!
//! Defines the configurable metadata-field schema, the self-contained filter
//! representation, and the structured-query result produced by parsing a
//! natural-language query.

use std::collections::HashMap;
use thiserror::Error;

use crate::types::Query;

// ── FieldType ─────────────────────────────────────────────────────────────────

/// The semantic type of a metadata field in a [`FilterSchema`].
///
/// The type drives which natural-language patterns the parser associates with
/// the field and how comparisons are performed at filter time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// A numeric field compared with `<`, `>`, `=`, etc. (parsed as `f64`).
    Number,
    /// A date/year field. Year-style comparisons (`after`, `before`, `in`) map here.
    Date,
    /// A free-text field compared by string equality or containment.
    Text,
    /// A categorical/tag field, typically matched with containment semantics.
    Tag,
}

impl FieldType {
    /// Human-readable name for the field type.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Number => "number",
            Self::Date => "date",
            Self::Text => "text",
            Self::Tag => "tag",
        }
    }
}

// ── FieldSpec ─────────────────────────────────────────────────────────────────

/// Specification of a single known metadata field.
///
/// A field is identified by its canonical `name` and zero or more `aliases`.
/// Both the name and the aliases are matched case-insensitively when resolving
/// a token to a field (see [`FilterSchema::field_for`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSpec {
    /// Canonical field name (the key written into [`FilterCondition::field`]).
    pub name: String,
    /// Semantic type of the field.
    pub field_type: FieldType,
    /// Alternative spellings/synonyms that also resolve to this field.
    pub aliases: Vec<String>,
}

impl FieldSpec {
    /// Create a new field specification with no aliases.
    #[must_use]
    pub fn new(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            aliases: Vec::new(),
        }
    }

    /// Add an alias for this field (builder).
    #[must_use]
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Add several aliases for this field (builder).
    #[must_use]
    pub fn with_aliases<I, S>(mut self, aliases: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.aliases.extend(aliases.into_iter().map(Into::into));
        self
    }

    /// Return `true` if `token` matches this field's name or any alias
    /// (case-insensitive).
    #[must_use]
    pub fn matches_token(&self, token: &str) -> bool {
        if self.name.eq_ignore_ascii_case(token) {
            return true;
        }
        self.aliases
            .iter()
            .any(|alias| alias.eq_ignore_ascii_case(token))
    }
}

// ── FilterSchema ──────────────────────────────────────────────────────────────

/// A collection of [`FieldSpec`]s describing the metadata fields the parser
/// knows about.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterSchema {
    /// The known fields, in declaration order.
    pub fields: Vec<FieldSpec>,
}

impl FilterSchema {
    /// Create a schema from a list of field specifications.
    #[must_use]
    pub fn new(fields: Vec<FieldSpec>) -> Self {
        Self { fields }
    }

    /// Add a field to the schema (builder).
    #[must_use]
    pub fn with_field(mut self, field: FieldSpec) -> Self {
        self.fields.push(field);
        self
    }

    /// Resolve `token` to a field by matching its name or any alias,
    /// case-insensitively.
    ///
    /// Returns the first matching [`FieldSpec`], or `None` when no field
    /// matches.
    #[must_use]
    pub fn field_for(&self, token: &str) -> Option<&FieldSpec> {
        self.fields.iter().find(|spec| spec.matches_token(token))
    }

    /// Return the first field of the given [`FieldType`], if any.
    ///
    /// This is used by the parser to pick a default target field for patterns
    /// that imply a type but not a specific field (for example, a bare year
    /// after `since` targets the first [`FieldType::Date`] field).
    #[must_use]
    pub fn first_of_type(&self, field_type: FieldType) -> Option<&FieldSpec> {
        self.fields
            .iter()
            .find(|spec| spec.field_type == field_type)
    }

    /// Return `true` when the schema has no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

// ── FilterOp ──────────────────────────────────────────────────────────────────

/// A comparison operator used in a [`FilterCondition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    /// Equal (`==`).
    Eq,
    /// Not equal (`!=`).
    Ne,
    /// Greater than (`>`).
    Gt,
    /// Less than (`<`).
    Lt,
    /// Greater than or equal (`>=`).
    Gte,
    /// Less than or equal (`<=`).
    Lte,
    /// Substring/containment match.
    Contains,
}

impl FilterOp {
    /// Compact symbolic representation of the operator.
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Gt => ">",
            Self::Lt => "<",
            Self::Gte => ">=",
            Self::Lte => "<=",
            Self::Contains => "contains",
        }
    }

    /// Evaluate this operator over two `f64` operands (`left op right`).
    #[must_use]
    pub fn apply_number(&self, left: f64, right: f64) -> bool {
        match self {
            // Containment over numbers falls back to equality.
            Self::Eq | Self::Contains => (left - right).abs() < f64::EPSILON,
            Self::Ne => (left - right).abs() >= f64::EPSILON,
            Self::Gt => left > right,
            Self::Lt => left < right,
            Self::Gte => left >= right,
            Self::Lte => left <= right,
        }
    }

    /// Evaluate this operator over two string operands (`left op right`),
    /// case-insensitively.
    #[must_use]
    pub fn apply_str(&self, left: &str, right: &str) -> bool {
        let l = left.to_lowercase();
        let r = right.to_lowercase();
        match self {
            Self::Eq => l == r,
            Self::Ne => l != r,
            Self::Contains => l.contains(&r),
            // Ordering operators are not meaningful for non-numeric strings;
            // treat them as a lexicographic comparison for determinism.
            Self::Gt => l > r,
            Self::Lt => l < r,
            Self::Gte => l >= r,
            Self::Lte => l <= r,
        }
    }
}

// ── FilterCondition ───────────────────────────────────────────────────────────

/// A single `field op value` predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterCondition {
    /// Canonical field name this condition applies to.
    pub field: String,
    /// Comparison operator.
    pub op: FilterOp,
    /// Right-hand-side value, stored as a string and parsed on demand.
    pub value: String,
}

impl FilterCondition {
    /// Create a new filter condition.
    #[must_use]
    pub fn new(field: impl Into<String>, op: FilterOp, value: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            op,
            value: value.into(),
        }
    }

    /// Evaluate this condition against a single metadata value.
    ///
    /// When both this condition's value and `actual` parse as `f64`, a numeric
    /// comparison is used; otherwise a string comparison is used.
    #[must_use]
    pub fn matches_value(&self, actual: &str) -> bool {
        match (
            actual.trim().parse::<f64>(),
            self.value.trim().parse::<f64>(),
        ) {
            (Ok(lhs), Ok(rhs)) => self.op.apply_number(lhs, rhs),
            _ => self.op.apply_str(actual, &self.value),
        }
    }

    /// Evaluate this condition against a metadata map.
    ///
    /// Returns `false` when the field is absent from `metadata`.
    #[must_use]
    pub fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        match metadata.get(&self.field) {
            Some(actual) => self.matches_value(actual),
            None => false,
        }
    }
}

// ── ParsedFilter ──────────────────────────────────────────────────────────────

/// A conjunction (AND) of [`FilterCondition`]s.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedFilter {
    /// The conditions, all of which must hold for a match.
    pub conditions: Vec<FilterCondition>,
}

impl ParsedFilter {
    /// Create an empty filter (matches everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a condition (builder).
    #[must_use]
    pub fn with_condition(mut self, condition: FilterCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Return `true` when no conditions are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Number of conditions in the filter.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// Evaluate all conditions against `metadata` (logical AND).
    ///
    /// An empty filter matches any metadata map.
    #[must_use]
    pub fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        self.conditions.iter().all(|cond| cond.matches(metadata))
    }
}

// ── StructuredQuery ───────────────────────────────────────────────────────────

/// The result of parsing a natural-language query.
///
/// Pairs the residual free-text [`semantic_query`](Self::semantic_query) (the
/// part to send to a semantic retriever) with the extracted structured
/// [`filter`](Self::filter).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredQuery {
    /// The free-text portion of the query, with filter phrases removed when
    /// stripping is enabled.
    pub semantic_query: String,
    /// The structured metadata filter extracted from the query.
    pub filter: ParsedFilter,
}

impl StructuredQuery {
    /// Create a new structured query.
    #[must_use]
    pub fn new(semantic_query: impl Into<String>, filter: ParsedFilter) -> Self {
        Self {
            semantic_query: semantic_query.into(),
            filter,
        }
    }

    /// Build a [`Query`] whose text is the semantic portion of this structured
    /// query.
    ///
    /// Only [`Query::text`] is populated; the structured [`filter`](Self::filter)
    /// is intentionally *not* translated into [`Query::metadata_filter`] — this
    /// module keeps a self-contained filter representation. Apply the filter
    /// separately (e.g. via [`ParsedFilter::matches`]).
    #[must_use]
    pub fn to_query(&self) -> Query {
        Query::new(self.semantic_query.clone())
    }
}

// ── SelfQueryConfig ───────────────────────────────────────────────────────────

/// Configuration for the [`SelfQueryParser`](super::parser::SelfQueryParser).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfQueryConfig {
    /// When `true`, spans matched as filter phrases are removed from the
    /// resulting [`StructuredQuery::semantic_query`].
    ///
    /// Defaults to `true`.
    pub strip_filter_phrases: bool,
}

impl Default for SelfQueryConfig {
    fn default() -> Self {
        Self {
            strip_filter_phrases: true,
        }
    }
}

impl SelfQueryConfig {
    /// Set whether matched filter phrases are stripped from the semantic query
    /// (builder).
    #[must_use]
    pub fn with_strip_filter_phrases(mut self, strip: bool) -> Self {
        self.strip_filter_phrases = strip;
        self
    }
}

// ── SelfQueryError ────────────────────────────────────────────────────────────

/// Errors produced by the `self_query` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SelfQueryError {
    /// The supplied natural-language query was empty (after trimming).
    #[error("query must not be empty")]
    EmptyQuery,
}
