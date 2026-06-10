//! Types for the `structured_extraction` module.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ── FieldType ─────────────────────────────────────────────────────────────────

/// The type of a field in the extraction schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    /// Plain text value.
    Text,
    /// Floating-point number.
    Number,
    /// ISO-8601 date string (heuristically detected).
    Date,
    /// Boolean (`true`/`false`, `yes`/`no`, `1`/`0`).
    Boolean,
    /// One of a set of allowed string values.
    Enum(Vec<String>),
}

impl FieldType {
    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Date => "date",
            Self::Boolean => "boolean",
            Self::Enum(_) => "enum",
        }
    }
}

// ── FieldSchema ───────────────────────────────────────────────────────────────

/// Schema definition for a single extracted field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    /// Field name (key in the output record).
    pub name: String,
    /// Expected value type.
    pub field_type: FieldType,
    /// Whether this field must be present.
    pub required: bool,
    /// Proximity keywords that mark the field's location in text.
    pub keywords: Vec<String>,
}

impl FieldSchema {
    /// Create a new required text field.
    #[must_use]
    pub fn new(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            required: false,
            keywords: Vec::new(),
        }
    }

    /// Mark the field as required.
    #[must_use]
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Add a keyword hint.
    #[must_use]
    pub fn with_keyword(mut self, kw: impl Into<String>) -> Self {
        self.keywords.push(kw.into());
        self
    }
}

// ── ExtractionSchema ──────────────────────────────────────────────────────────

/// A collection of field schemas describing the extraction target.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionSchema {
    /// Field definitions.
    pub fields: Vec<FieldSchema>,
}

impl ExtractionSchema {
    /// Create an empty schema.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a field schema.
    #[must_use]
    pub fn with_field(mut self, field: FieldSchema) -> Self {
        self.fields.push(field);
        self
    }

    /// Return `true` if no fields are defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

// ── ExtractedValue ────────────────────────────────────────────────────────────

/// A parsed value for a single field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExtractedValue {
    /// Extracted text.
    Text(String),
    /// Parsed floating-point number.
    Number(f64),
    /// Date string in ISO-8601-like format.
    Date(String),
    /// Boolean value.
    Boolean(bool),
    /// Matched enum variant.
    Enum(String),
}

impl ExtractedValue {
    /// Return the value as a JSON-compatible string representation.
    #[must_use]
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::Text(s) => serde_json::Value::String(s.clone()),
            Self::Number(n) => serde_json::Value::Number(
                serde_json::Number::from_f64(*n).unwrap_or(serde_json::Number::from(0)),
            ),
            Self::Date(d) => serde_json::Value::String(d.clone()),
            Self::Boolean(b) => serde_json::Value::Bool(*b),
            Self::Enum(e) => serde_json::Value::String(e.clone()),
        }
    }
}

// ── ExtractedRecord ───────────────────────────────────────────────────────────

/// A single extracted record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRecord {
    /// Map of field name → extracted value.
    pub fields: HashMap<String, ExtractedValue>,
    /// Names of required fields that could not be extracted.
    pub missing_required: Vec<String>,
}

impl ExtractedRecord {
    /// Serialize all fields to a `serde_json::Value::Object`.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let map: serde_json::Map<String, serde_json::Value> = self
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), v.to_json_value()))
            .collect();
        serde_json::Value::Object(map)
    }

    /// Return `true` when all required fields were extracted.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.missing_required.is_empty()
    }
}

// ── ExtractionConfig ──────────────────────────────────────────────────────────

/// Configuration for the structured extractor.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Match keywords case-insensitively.
    ///
    /// Defaults to `true`.
    pub case_insensitive: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            case_insensitive: true,
        }
    }
}

impl ExtractionConfig {
    /// Set case sensitivity.
    #[must_use]
    pub fn with_case_insensitive(mut self, v: bool) -> Self {
        self.case_insensitive = v;
        self
    }
}

// ── StructuredExtractionError ─────────────────────────────────────────────────

/// Errors from the `structured_extraction` module.
#[derive(Debug, Error)]
pub enum StructuredExtractionError {
    /// Schema has no fields defined.
    #[error("Schema must define at least one field")]
    EmptySchema,

    /// Value could not be parsed to the declared type.
    #[error("Parse failed for field '{field}': {reason}")]
    ParseFailed {
        /// Field name that failed.
        field: String,
        /// Failure reason.
        reason: String,
    },
}
