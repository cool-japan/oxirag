//! Tests for the `structured_extraction` module.

use super::extractor::SchemaExtractor;
use super::types::{
    ExtractedValue, ExtractionConfig, ExtractionSchema, FieldSchema, FieldType,
    StructuredExtractionError,
};

// ── FieldType tests ───────────────────────────────────────────────────────────

#[test]
fn test_field_type_names() {
    assert_eq!(FieldType::Text.name(), "text");
    assert_eq!(FieldType::Number.name(), "number");
    assert_eq!(FieldType::Date.name(), "date");
    assert_eq!(FieldType::Boolean.name(), "boolean");
    assert_eq!(FieldType::Enum(vec![]).name(), "enum");
}

// ── ExtractionSchema tests ────────────────────────────────────────────────────

#[test]
fn test_schema_empty() {
    let schema = ExtractionSchema::new();
    assert!(schema.is_empty());
}

#[test]
fn test_schema_with_fields() {
    let schema =
        ExtractionSchema::new().with_field(FieldSchema::new("name", FieldType::Text).required());
    assert!(!schema.is_empty());
    assert_eq!(schema.fields.len(), 1);
    assert!(schema.fields[0].required);
}

// ── SchemaExtractor tests ─────────────────────────────────────────────────────

#[test]
fn test_extractor_empty_schema_error() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new();
    let cfg = ExtractionConfig::default();
    let err = extractor
        .extract("some text", &schema, &cfg)
        .expect_err("should fail");
    assert!(matches!(err, StructuredExtractionError::EmptySchema));
}

#[test]
fn test_extractor_text_field() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("name", FieldType::Text).with_keyword("name"));
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("name: Alice Smith", &schema, &cfg)
        .expect("ok");
    let val = record.fields.get("name");
    assert!(val.is_some(), "name field should be extracted");
}

#[test]
fn test_extractor_number_field() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("price", FieldType::Number).with_keyword("price"));
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("price: 42.50 USD", &schema, &cfg)
        .expect("ok");
    if let Some(ExtractedValue::Number(n)) = record.fields.get("price") {
        assert!((*n - 42.5).abs() < 1e-5, "expected 42.5, got {n}");
    } else {
        panic!("price not extracted: {:?}", record.fields);
    }
}

#[test]
fn test_extractor_boolean_true() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("active", FieldType::Boolean).with_keyword("active"));
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("active: true", &schema, &cfg)
        .expect("ok");
    assert_eq!(
        record.fields.get("active"),
        Some(&ExtractedValue::Boolean(true))
    );
}

#[test]
fn test_extractor_boolean_yes() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("enabled", FieldType::Boolean).with_keyword("enabled"));
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("enabled: yes", &schema, &cfg)
        .expect("ok");
    assert_eq!(
        record.fields.get("enabled"),
        Some(&ExtractedValue::Boolean(true))
    );
}

#[test]
fn test_extractor_boolean_false() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("flag", FieldType::Boolean).with_keyword("flag"));
    let cfg = ExtractionConfig::default();
    let record = extractor.extract("flag: false", &schema, &cfg).expect("ok");
    assert_eq!(
        record.fields.get("flag"),
        Some(&ExtractedValue::Boolean(false))
    );
}

#[test]
fn test_extractor_date_field() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("date", FieldType::Date).with_keyword("date"));
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("date: 2024-06-15", &schema, &cfg)
        .expect("ok");
    if let Some(ExtractedValue::Date(d)) = record.fields.get("date") {
        assert_eq!(d, "2024-06-15");
    } else {
        panic!("date not extracted: {:?}", record.fields);
    }
}

#[test]
fn test_extractor_enum_field() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new().with_field(
        FieldSchema::new(
            "status",
            FieldType::Enum(vec!["active".into(), "inactive".into(), "pending".into()]),
        )
        .with_keyword("status"),
    );
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("status: pending review", &schema, &cfg)
        .expect("ok");
    if let Some(ExtractedValue::Enum(e)) = record.fields.get("status") {
        assert_eq!(e, "pending");
    } else {
        panic!("enum not extracted: {:?}", record.fields);
    }
}

#[test]
fn test_extractor_missing_required() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new().with_field(
        FieldSchema::new("title", FieldType::Text)
            .required()
            .with_keyword("title"),
    );
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("no matching keyword here", &schema, &cfg)
        .expect("ok");
    assert!(
        record.missing_required.contains(&"title".to_string()),
        "required field should be listed as missing"
    );
    assert!(!record.is_complete());
}

#[test]
fn test_extractor_optional_missing_ok() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("optional_field", FieldType::Text).with_keyword("xyz_token"));
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("unrelated text", &schema, &cfg)
        .expect("ok");
    assert!(
        record.missing_required.is_empty(),
        "optional fields should not be in missing_required"
    );
    assert!(record.is_complete());
}

#[test]
fn test_extractor_case_insensitive() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("count", FieldType::Number).with_keyword("COUNT"));
    let cfg = ExtractionConfig::default().with_case_insensitive(true);
    let record = extractor
        .extract("count: 7 items", &schema, &cfg)
        .expect("ok");
    assert!(
        record.fields.contains_key("count"),
        "case-insensitive match"
    );
}

#[test]
fn test_extracted_record_to_json() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("score", FieldType::Number).with_keyword("score"));
    let cfg = ExtractionConfig::default();
    let record = extractor.extract("score: 99", &schema, &cfg).expect("ok");
    let json = record.to_json();
    assert!(json.is_object());
}

#[test]
fn test_extracted_value_to_json_variants() {
    use super::types::ExtractedValue;
    assert!(ExtractedValue::Text("x".into()).to_json_value().is_string());
    assert!(ExtractedValue::Number(3.0).to_json_value().is_number());
    assert!(ExtractedValue::Boolean(true).to_json_value().is_boolean());
    assert!(
        ExtractedValue::Date("2024-01-01".into())
            .to_json_value()
            .is_string()
    );
    assert!(ExtractedValue::Enum("a".into()).to_json_value().is_string());
}

#[test]
fn test_extractor_multiple_fields() {
    let extractor = SchemaExtractor::new();
    let schema = ExtractionSchema::new()
        .with_field(FieldSchema::new("price", FieldType::Number).with_keyword("price"))
        .with_field(FieldSchema::new("available", FieldType::Boolean).with_keyword("available"));
    let cfg = ExtractionConfig::default();
    let record = extractor
        .extract("price: 12.99 available: true", &schema, &cfg)
        .expect("ok");
    assert!(record.fields.contains_key("price"));
    assert!(record.fields.contains_key("available"));
}
