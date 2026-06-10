//! Schema-constrained text extraction logic.

use super::types::{
    ExtractedRecord, ExtractedValue, ExtractionConfig, ExtractionSchema, FieldType,
    StructuredExtractionError,
};
use std::collections::HashMap;

// ── parse helpers ─────────────────────────────────────────────────────────────

/// Extract the first number-like token after a keyword.
fn extract_number_near(text: &str, keyword: &str, case_insensitive: bool) -> Option<f64> {
    let haystack = if case_insensitive {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    let needle = if case_insensitive {
        keyword.to_lowercase()
    } else {
        keyword.to_string()
    };

    let pos = haystack.find(needle.as_str())?;
    let after = &haystack[pos + needle.len()..];

    // Scan for first number token
    for token in after.split_whitespace().take(5) {
        let cleaned: String = token
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        if let Ok(n) = cleaned.parse::<f64>() {
            return Some(n);
        }
    }
    None
}

/// Extract the first boolean-like token near a keyword.
fn extract_bool_near(text: &str, keyword: &str, case_insensitive: bool) -> Option<bool> {
    let haystack = if case_insensitive {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    let needle = if case_insensitive {
        keyword.to_lowercase()
    } else {
        keyword.to_string()
    };

    let pos = haystack.find(needle.as_str())?;
    let after = &haystack[pos + needle.len()..];

    for token in after.split_whitespace().take(3) {
        match token.trim_matches(|c: char| !c.is_alphanumeric()) {
            "true" | "yes" | "1" => return Some(true),
            "false" | "no" | "0" => return Some(false),
            _ => {}
        }
    }
    None
}

/// Extract a date-like token (YYYY-MM-DD or MM/DD/YYYY) near a keyword.
fn extract_date_near(text: &str, keyword: &str, case_insensitive: bool) -> Option<String> {
    let haystack = if case_insensitive {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    let needle = if case_insensitive {
        keyword.to_lowercase()
    } else {
        keyword.to_string()
    };

    let pos = haystack.find(needle.as_str())?;
    // Search in a window after the keyword
    let window = &text[pos.min(text.len())..];

    for token in window.split_whitespace().take(6) {
        let t = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '/');
        // YYYY-MM-DD
        if t.len() == 10 && t.chars().nth(4) == Some('-') && t.chars().nth(7) == Some('-') {
            return Some(t.to_string());
        }
        // MM/DD/YYYY
        if t.len() == 10 && t.chars().nth(2) == Some('/') && t.chars().nth(5) == Some('/') {
            return Some(t.to_string());
        }
        // YYYY/MM/DD
        if t.len() == 10 && t.chars().nth(4) == Some('/') && t.chars().nth(7) == Some('/') {
            return Some(t.to_string());
        }
    }
    None
}

/// Extract text snippet near a keyword.
fn extract_text_near(text: &str, keyword: &str, case_insensitive: bool) -> Option<String> {
    let haystack = if case_insensitive {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    let needle = if case_insensitive {
        keyword.to_lowercase()
    } else {
        keyword.to_string()
    };

    let pos = haystack.find(needle.as_str())?;
    let after = &text[pos + needle.len()..];

    // Grab the next 1-5 word tokens after optional colon/whitespace
    let tokens: Vec<&str> = after
        .trim_start_matches([':', ' ', '\t'])
        .split_whitespace()
        .take(5)
        .collect();

    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

/// Match one of the enum variants near a keyword.
fn extract_enum_near(
    text: &str,
    keyword: &str,
    variants: &[String],
    case_insensitive: bool,
) -> Option<String> {
    let haystack = if case_insensitive {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    let needle = if case_insensitive {
        keyword.to_lowercase()
    } else {
        keyword.to_string()
    };

    let pos = haystack.find(needle.as_str())?;
    let window = &haystack[pos..];

    for variant in variants {
        let v = if case_insensitive {
            variant.to_lowercase()
        } else {
            variant.clone()
        };
        if window.contains(v.as_str()) {
            return Some(variant.clone());
        }
    }
    None
}

// ── SchemaExtractor ───────────────────────────────────────────────────────────

/// Extracts structured values from unstructured text based on an [`ExtractionSchema`].
///
/// Uses keyword-proximity heuristics — no regex, no external NLP.
#[derive(Debug, Clone, Default)]
pub struct SchemaExtractor;

impl SchemaExtractor {
    /// Create a new [`SchemaExtractor`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract a record from `text` using the given `schema`.
    ///
    /// # Errors
    ///
    /// Returns [`StructuredExtractionError::EmptySchema`] when `schema.is_empty()`.
    pub fn extract(
        &self,
        text: &str,
        schema: &ExtractionSchema,
        config: &ExtractionConfig,
    ) -> Result<ExtractedRecord, StructuredExtractionError> {
        if schema.is_empty() {
            return Err(StructuredExtractionError::EmptySchema);
        }

        let mut fields: HashMap<String, ExtractedValue> = HashMap::new();
        let mut missing_required: Vec<String> = Vec::new();
        let ci = config.case_insensitive;

        for field_schema in &schema.fields {
            // Try each keyword as an anchor; take the first successful extraction
            let mut value: Option<ExtractedValue> = None;

            // Prefer schema keywords; fall back to using the field name itself
            let keywords_to_try: Vec<&str> = if field_schema.keywords.is_empty() {
                vec![field_schema.name.as_str()]
            } else {
                field_schema.keywords.iter().map(String::as_str).collect()
            };

            'kw: for kw in &keywords_to_try {
                match &field_schema.field_type {
                    FieldType::Number => {
                        if let Some(n) = extract_number_near(text, kw, ci) {
                            value = Some(ExtractedValue::Number(n));
                            break 'kw;
                        }
                    }
                    FieldType::Boolean => {
                        if let Some(b) = extract_bool_near(text, kw, ci) {
                            value = Some(ExtractedValue::Boolean(b));
                            break 'kw;
                        }
                    }
                    FieldType::Date => {
                        if let Some(d) = extract_date_near(text, kw, ci) {
                            value = Some(ExtractedValue::Date(d));
                            break 'kw;
                        }
                    }
                    FieldType::Enum(variants) => {
                        if let Some(e) = extract_enum_near(text, kw, variants, ci) {
                            value = Some(ExtractedValue::Enum(e));
                            break 'kw;
                        }
                    }
                    FieldType::Text => {
                        if let Some(t) = extract_text_near(text, kw, ci) {
                            value = Some(ExtractedValue::Text(t));
                            break 'kw;
                        }
                    }
                }
            }

            match value {
                Some(v) => {
                    fields.insert(field_schema.name.clone(), v);
                }
                None => {
                    if field_schema.required {
                        missing_required.push(field_schema.name.clone());
                    }
                }
            }
        }

        Ok(ExtractedRecord {
            fields,
            missing_required,
        })
    }
}
