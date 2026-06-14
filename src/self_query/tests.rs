//! Tests for the `self_query` module.
#![allow(clippy::float_cmp, clippy::similar_names)]

use std::collections::HashMap;

use super::parser::{SelfQueryParser, SelfQueryRetriever};
use super::types::{
    FieldSpec, FieldType, FilterCondition, FilterOp, FilterSchema, ParsedFilter, SelfQueryConfig,
    SelfQueryError, StructuredQuery,
};
use crate::types::Document;

// ── helpers ─────────────────────────────────────────────────────────────────

/// A canonical paper schema: date `year`, text `author`, number `rating`,
/// tag `topic`.
fn paper_schema() -> FilterSchema {
    FilterSchema::new(vec![
        FieldSpec::new("year", FieldType::Date).with_alias("published"),
        FieldSpec::new("author", FieldType::Text).with_aliases(["writer", "creator"]),
        FieldSpec::new("rating", FieldType::Number).with_alias("score"),
        FieldSpec::new("topic", FieldType::Tag).with_aliases(["category", "tag"]),
    ])
}

fn paper_parser() -> SelfQueryParser {
    SelfQueryParser::new(paper_schema())
}

fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

fn doc_with(pairs: &[(&str, &str)], content: &str) -> Document {
    let mut doc = Document::new(content);
    doc.metadata = meta(pairs);
    doc
}

// ── FieldType ─────────────────────────────────────────────────────────────────

#[test]
fn test_field_type_names() {
    assert_eq!(FieldType::Number.name(), "number");
    assert_eq!(FieldType::Date.name(), "date");
    assert_eq!(FieldType::Text.name(), "text");
    assert_eq!(FieldType::Tag.name(), "tag");
}

// ── FieldSpec / FilterSchema lookup ───────────────────────────────────────────

#[test]
fn test_field_spec_new_no_aliases() {
    let spec = FieldSpec::new("year", FieldType::Date);
    assert_eq!(spec.name, "year");
    assert_eq!(spec.field_type, FieldType::Date);
    assert!(spec.aliases.is_empty());
}

#[test]
fn test_field_spec_with_alias_builder() {
    let spec = FieldSpec::new("author", FieldType::Text)
        .with_alias("writer")
        .with_alias("creator");
    assert_eq!(
        spec.aliases,
        vec!["writer".to_string(), "creator".to_string()]
    );
}

#[test]
fn test_field_spec_with_aliases_bulk() {
    let spec = FieldSpec::new("topic", FieldType::Tag).with_aliases(["tag", "category"]);
    assert_eq!(spec.aliases.len(), 2);
}

#[test]
fn test_field_spec_matches_token_name_case_insensitive() {
    let spec = FieldSpec::new("Year", FieldType::Date);
    assert!(spec.matches_token("year"));
    assert!(spec.matches_token("YEAR"));
    assert!(!spec.matches_token("author"));
}

#[test]
fn test_field_spec_matches_token_alias() {
    let spec = FieldSpec::new("author", FieldType::Text).with_alias("Writer");
    assert!(spec.matches_token("writer"));
    assert!(spec.matches_token("WRITER"));
}

#[test]
fn test_schema_field_for_by_name() {
    let schema = paper_schema();
    let spec = schema.field_for("rating").expect("rating field");
    assert_eq!(spec.name, "rating");
    assert_eq!(spec.field_type, FieldType::Number);
}

#[test]
fn test_schema_field_for_by_alias() {
    let schema = paper_schema();
    let spec = schema.field_for("published").expect("alias of year");
    assert_eq!(spec.name, "year");
}

#[test]
fn test_schema_field_for_case_insensitive() {
    let schema = paper_schema();
    assert_eq!(
        schema.field_for("AUTHOR").map(|s| s.name.as_str()),
        Some("author")
    );
    assert_eq!(
        schema.field_for("Score").map(|s| s.name.as_str()),
        Some("rating")
    );
}

#[test]
fn test_schema_field_for_unknown() {
    let schema = paper_schema();
    assert!(schema.field_for("nonexistent").is_none());
}

#[test]
fn test_schema_first_of_type() {
    let schema = paper_schema();
    assert_eq!(
        schema
            .first_of_type(FieldType::Date)
            .map(|s| s.name.as_str()),
        Some("year")
    );
    assert_eq!(
        schema
            .first_of_type(FieldType::Number)
            .map(|s| s.name.as_str()),
        Some("rating")
    );
    assert_eq!(
        schema
            .first_of_type(FieldType::Tag)
            .map(|s| s.name.as_str()),
        Some("topic")
    );
}

#[test]
fn test_schema_is_empty() {
    assert!(FilterSchema::default().is_empty());
    assert!(!paper_schema().is_empty());
}

#[test]
fn test_schema_with_field_builder() {
    let schema = FilterSchema::default()
        .with_field(FieldSpec::new("a", FieldType::Text))
        .with_field(FieldSpec::new("b", FieldType::Number));
    assert_eq!(schema.fields.len(), 2);
}

// ── FilterOp ──────────────────────────────────────────────────────────────────

#[test]
fn test_filter_op_symbol() {
    assert_eq!(FilterOp::Eq.symbol(), "=");
    assert_eq!(FilterOp::Ne.symbol(), "!=");
    assert_eq!(FilterOp::Gt.symbol(), ">");
    assert_eq!(FilterOp::Lt.symbol(), "<");
    assert_eq!(FilterOp::Gte.symbol(), ">=");
    assert_eq!(FilterOp::Lte.symbol(), "<=");
    assert_eq!(FilterOp::Contains.symbol(), "contains");
}

#[test]
fn test_filter_op_apply_number() {
    assert!(FilterOp::Gt.apply_number(5.0, 4.0));
    assert!(!FilterOp::Gt.apply_number(4.0, 5.0));
    assert!(FilterOp::Lt.apply_number(3.0, 4.0));
    assert!(FilterOp::Gte.apply_number(4.0, 4.0));
    assert!(FilterOp::Lte.apply_number(4.0, 4.0));
    assert!(FilterOp::Eq.apply_number(4.0, 4.0));
    assert!(FilterOp::Ne.apply_number(4.0, 5.0));
}

#[test]
fn test_filter_op_apply_str() {
    assert!(FilterOp::Eq.apply_str("Vaswani", "vaswani"));
    assert!(FilterOp::Ne.apply_str("a", "b"));
    assert!(FilterOp::Contains.apply_str("deep learning", "learning"));
    assert!(!FilterOp::Contains.apply_str("deep", "learning"));
}

// ── parsing: date patterns ────────────────────────────────────────────────────

#[test]
fn test_parse_after_year_gt() {
    let parsed = paper_parser().parse("papers after 2020").expect("ok");
    assert_eq!(parsed.filter.conditions.len(), 1);
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "year");
    assert_eq!(cond.op, FilterOp::Gt);
    assert_eq!(cond.value, "2020");
}

#[test]
fn test_parse_since_year_gt() {
    let parsed = paper_parser().parse("work since 2015").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.op, FilterOp::Gt);
    assert_eq!(cond.value, "2015");
}

#[test]
fn test_parse_before_year_lt() {
    let parsed = paper_parser().parse("articles before 2019").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "year");
    assert_eq!(cond.op, FilterOp::Lt);
    assert_eq!(cond.value, "2019");
}

#[test]
fn test_parse_until_year_lt() {
    let parsed = paper_parser().parse("stuff until 2022").expect("ok");
    assert_eq!(parsed.filter.conditions[0].op, FilterOp::Lt);
}

#[test]
fn test_parse_in_year_eq() {
    let parsed = paper_parser().parse("papers in 2021").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "year");
    assert_eq!(cond.op, FilterOp::Eq);
    assert_eq!(cond.value, "2021");
}

#[test]
fn test_parse_non_year_after_is_not_date() {
    // "after lunch" — lunch is not a year, so no condition is produced.
    let parsed = paper_parser().parse("notes after lunch").expect("ok");
    assert!(parsed.filter.is_empty());
}

// ── parsing: author pattern ───────────────────────────────────────────────────

#[test]
fn test_parse_by_author_eq() {
    let parsed = paper_parser().parse("papers by Vaswani").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "author");
    assert_eq!(cond.op, FilterOp::Eq);
    assert_eq!(cond.value, "Vaswani");
}

#[test]
fn test_parse_by_multiword_author() {
    let parsed = paper_parser()
        .parse("research by Ashish Vaswani")
        .expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "author");
    assert_eq!(cond.value, "Ashish Vaswani");
}

#[test]
fn test_parse_by_lowercase_author_single_token() {
    let parsed = paper_parser().parse("texts by smith").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.value, "smith");
}

// ── parsing: number patterns ──────────────────────────────────────────────────

#[test]
fn test_parse_above_number_gt() {
    let parsed = paper_parser().parse("papers above 4").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "rating");
    assert_eq!(cond.op, FilterOp::Gt);
    assert_eq!(cond.value, "4");
}

#[test]
fn test_parse_rated_number_gt() {
    let parsed = paper_parser().parse("movies rated 4").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "rating");
    assert_eq!(cond.op, FilterOp::Gt);
    assert_eq!(cond.value, "4");
}

#[test]
fn test_parse_over_number_gt() {
    let parsed = paper_parser().parse("scored over 3").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.op, FilterOp::Gt);
    assert_eq!(cond.value, "3");
}

#[test]
fn test_parse_below_number_lt() {
    let parsed = paper_parser().parse("items below 2").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "rating");
    assert_eq!(cond.op, FilterOp::Lt);
    assert_eq!(cond.value, "2");
}

#[test]
fn test_parse_under_number_lt() {
    let parsed = paper_parser().parse("things under 5").expect("ok");
    assert_eq!(parsed.filter.conditions[0].op, FilterOp::Lt);
}

// ── parsing: tag pattern ──────────────────────────────────────────────────────

#[test]
fn test_parse_tagged_contains() {
    let parsed = paper_parser().parse("docs tagged nlp").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "topic");
    assert_eq!(cond.op, FilterOp::Contains);
    assert_eq!(cond.value, "nlp");
}

#[test]
fn test_parse_category_contains() {
    let parsed = paper_parser().parse("posts category science").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.field, "topic");
    assert_eq!(cond.op, FilterOp::Contains);
    assert_eq!(cond.value, "science");
}

#[test]
fn test_parse_tagged_as_connector() {
    let parsed = paper_parser().parse("docs tagged as vision").expect("ok");
    let cond = &parsed.filter.conditions[0];
    assert_eq!(cond.value, "vision");
}

// ── parsing: multiple conditions / semantic stripping ─────────────────────────

#[test]
fn test_parse_multiple_conditions() {
    let parsed = paper_parser()
        .parse("papers about transformers after 2020 by Vaswani rated above 4")
        .expect("ok");
    assert_eq!(parsed.filter.conditions.len(), 3);

    let by_field = |name: &str| {
        parsed
            .filter
            .conditions
            .iter()
            .find(|c| c.field == name)
            .cloned()
    };

    let year = by_field("year").expect("year cond");
    assert_eq!(year.op, FilterOp::Gt);
    assert_eq!(year.value, "2020");

    let author = by_field("author").expect("author cond");
    assert_eq!(author.op, FilterOp::Eq);
    assert_eq!(author.value, "Vaswani");

    let rating = by_field("rating").expect("rating cond");
    assert_eq!(rating.op, FilterOp::Gt);
    assert_eq!(rating.value, "4");
}

#[test]
fn test_parse_strips_structured_phrases() {
    let parsed = paper_parser()
        .parse("papers about transformers after 2020 by Vaswani rated above 4")
        .expect("ok");
    assert_eq!(parsed.semantic_query, "papers about transformers");
}

#[test]
fn test_parse_strips_single_phrase() {
    let parsed = paper_parser()
        .parse("deep learning after 2018")
        .expect("ok");
    assert_eq!(parsed.semantic_query, "deep learning");
}

#[test]
fn test_parse_strip_leaves_clean_when_filter_in_middle() {
    let parsed = paper_parser()
        .parse("survey by Bengio on attention")
        .expect("ok");
    assert_eq!(parsed.semantic_query, "survey on attention");
    assert_eq!(parsed.filter.conditions[0].value, "Bengio");
}

#[test]
fn test_parse_no_strip_when_disabled() {
    let parser =
        paper_parser().with_config(SelfQueryConfig::default().with_strip_filter_phrases(false));
    let parsed = parser.parse("papers after 2020 by Vaswani").expect("ok");
    assert_eq!(parsed.semantic_query, "papers after 2020 by Vaswani");
    assert_eq!(parsed.filter.conditions.len(), 2);
}

// ── parsing: no-filter / empty ────────────────────────────────────────────────

#[test]
fn test_parse_no_filter_keeps_full_text() {
    let parsed = paper_parser()
        .parse("what is a transformer model")
        .expect("ok");
    assert!(parsed.filter.is_empty());
    assert_eq!(parsed.semantic_query, "what is a transformer model");
}

#[test]
fn test_parse_empty_query_errors() {
    let err = paper_parser().parse("").expect_err("should error");
    assert_eq!(err, SelfQueryError::EmptyQuery);
}

#[test]
fn test_parse_whitespace_query_errors() {
    let err = paper_parser().parse("   \t  ").expect_err("should error");
    assert!(matches!(err, SelfQueryError::EmptyQuery));
}

#[test]
fn test_parse_trigger_without_value_no_condition() {
    // Trailing "after" with no following token must not panic or match.
    let parsed = paper_parser().parse("papers after").expect("ok");
    assert!(parsed.filter.is_empty());
}

#[test]
fn test_parse_missing_field_type_in_schema() {
    // Schema with only a text field: a year pattern has no date field to bind.
    let schema = FilterSchema::new(vec![FieldSpec::new("author", FieldType::Text)]);
    let parser = SelfQueryParser::new(schema);
    let parsed = parser.parse("papers after 2020").expect("ok");
    assert!(parsed.filter.is_empty());
}

// ── ParsedFilter ──────────────────────────────────────────────────────────────

#[test]
fn test_parsed_filter_empty_default() {
    let f = ParsedFilter::new();
    assert!(f.is_empty());
    assert_eq!(f.len(), 0);
}

#[test]
fn test_parsed_filter_builder() {
    let f = ParsedFilter::new()
        .with_condition(FilterCondition::new("year", FilterOp::Gt, "2020"))
        .with_condition(FilterCondition::new("author", FilterOp::Eq, "Vaswani"));
    assert_eq!(f.len(), 2);
    assert!(!f.is_empty());
}

#[test]
fn test_parsed_filter_empty_matches_anything() {
    let f = ParsedFilter::new();
    assert!(f.matches(&meta(&[("anything", "goes")])));
    assert!(f.matches(&HashMap::new()));
}

#[test]
fn test_filter_condition_matches_numeric() {
    // 5 > 4 numerically.
    let cond = FilterCondition::new("rating", FilterOp::Gt, "4");
    assert!(cond.matches(&meta(&[("rating", "5")])));
    assert!(!cond.matches(&meta(&[("rating", "3")])));
}

#[test]
fn test_filter_condition_numeric_boundary() {
    let cond = FilterCondition::new("rating", FilterOp::Gte, "4");
    assert!(cond.matches(&meta(&[("rating", "4")])));
    assert!(!cond.matches(&meta(&[("rating", "3.9")])));
}

#[test]
fn test_filter_condition_matches_string_eq() {
    let cond = FilterCondition::new("author", FilterOp::Eq, "Vaswani");
    assert!(cond.matches(&meta(&[("author", "vaswani")])));
    assert!(!cond.matches(&meta(&[("author", "Bengio")])));
}

#[test]
fn test_filter_condition_matches_contains() {
    let cond = FilterCondition::new("topic", FilterOp::Contains, "nlp");
    assert!(cond.matches(&meta(&[("topic", "applied-nlp-research")])));
    assert!(!cond.matches(&meta(&[("topic", "vision")])));
}

#[test]
fn test_filter_condition_absent_field_no_match() {
    let cond = FilterCondition::new("rating", FilterOp::Gt, "4");
    assert!(!cond.matches(&meta(&[("year", "2020")])));
}

#[test]
fn test_filter_condition_string_vs_number_fallback() {
    // actual value is non-numeric, so string comparison is used.
    let cond = FilterCondition::new("rating", FilterOp::Eq, "high");
    assert!(cond.matches(&meta(&[("rating", "HIGH")])));
}

#[test]
fn test_parsed_filter_and_semantics() {
    let f = ParsedFilter::new()
        .with_condition(FilterCondition::new("year", FilterOp::Gt, "2019"))
        .with_condition(FilterCondition::new("author", FilterOp::Eq, "Vaswani"));
    assert!(f.matches(&meta(&[("year", "2021"), ("author", "Vaswani")])));
    // Fails the author condition.
    assert!(!f.matches(&meta(&[("year", "2021"), ("author", "Bengio")])));
    // Fails the year condition.
    assert!(!f.matches(&meta(&[("year", "2018"), ("author", "Vaswani")])));
}

// ── StructuredQuery::to_query ─────────────────────────────────────────────────

#[test]
fn test_structured_query_to_query_sets_text_only() {
    let sq = StructuredQuery::new(
        "transformers",
        ParsedFilter::new().with_condition(FilterCondition::new("year", FilterOp::Gt, "2020")),
    );
    let query = sq.to_query();
    assert_eq!(query.text, "transformers");
    assert!(query.metadata_filter.is_none());
    assert!(query.filters.is_empty());
}

// ── SelfQueryRetriever ────────────────────────────────────────────────────────

#[test]
fn test_retriever_filters_docs() {
    let retriever = SelfQueryRetriever::new(paper_schema());
    let docs = vec![
        doc_with(
            &[("year", "2021"), ("author", "Vaswani")],
            "attention paper",
        ),
        doc_with(&[("year", "2018"), ("author", "Vaswani")], "older paper"),
        doc_with(&[("year", "2022"), ("author", "Bengio")], "other author"),
    ];
    let (sq, matched) = retriever
        .retrieve("papers after 2020 by Vaswani", &docs)
        .expect("ok");
    assert_eq!(sq.filter.conditions.len(), 2);
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].content, "attention paper");
}

#[test]
fn test_retriever_no_matches_returns_empty() {
    let retriever = SelfQueryRetriever::new(paper_schema());
    let docs = vec![
        doc_with(&[("year", "2010")], "ancient"),
        doc_with(&[("year", "2015")], "old"),
    ];
    let (_, matched) = retriever.retrieve("papers after 2020", &docs).expect("ok");
    assert!(matched.is_empty());
}

#[test]
fn test_retriever_empty_filter_returns_all() {
    let retriever = SelfQueryRetriever::new(paper_schema());
    let docs = vec![
        doc_with(&[("year", "2021")], "a"),
        doc_with(&[("year", "2010")], "b"),
    ];
    let (sq, matched) = retriever.retrieve("what is attention", &docs).expect("ok");
    assert!(sq.filter.is_empty());
    assert_eq!(matched.len(), 2);
}

#[test]
fn test_retriever_empty_query_errors() {
    let retriever = SelfQueryRetriever::new(paper_schema());
    let docs: Vec<Document> = vec![];
    let err = retriever.retrieve("  ", &docs).expect_err("should error");
    assert_eq!(err, SelfQueryError::EmptyQuery);
}

#[test]
fn test_retriever_from_parser() {
    let parser = paper_parser();
    let retriever = SelfQueryRetriever::from_parser(parser);
    let docs = vec![doc_with(&[("rating", "5")], "great")];
    let (_, matched) = retriever.retrieve("things above 4", &docs).expect("ok");
    assert_eq!(matched.len(), 1);
}

#[test]
fn test_retriever_multiple_conditions_filter() {
    let retriever = SelfQueryRetriever::new(paper_schema());
    let docs = vec![
        doc_with(
            &[("year", "2021"), ("rating", "5"), ("author", "Vaswani")],
            "match",
        ),
        doc_with(
            &[("year", "2021"), ("rating", "2"), ("author", "Vaswani")],
            "low rating",
        ),
        doc_with(
            &[("year", "2019"), ("rating", "5"), ("author", "Vaswani")],
            "too old",
        ),
    ];
    let (_, matched) = retriever
        .retrieve("papers after 2020 by Vaswani rated above 4", &docs)
        .expect("ok");
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].content, "match");
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_parse_is_deterministic() {
    let parser = paper_parser();
    let q = "papers about transformers after 2020 by Vaswani rated above 4 tagged nlp";
    let first = parser.parse(q).expect("ok");
    for _ in 0..16 {
        let again = parser.parse(q).expect("ok");
        assert_eq!(first, again);
    }
}

#[test]
fn test_retrieve_is_deterministic() {
    let retriever = SelfQueryRetriever::new(paper_schema());
    let docs = vec![
        doc_with(&[("year", "2021"), ("author", "Vaswani")], "a"),
        doc_with(&[("year", "2022"), ("author", "Vaswani")], "b"),
        doc_with(&[("year", "2010"), ("author", "Vaswani")], "c"),
    ];
    let (sq0, m0) = retriever
        .retrieve("papers after 2020 by Vaswani", &docs)
        .expect("ok");
    let order0: Vec<&str> = m0.iter().map(|d| d.content.as_str()).collect();
    for _ in 0..16 {
        let (sq, m) = retriever
            .retrieve("papers after 2020 by Vaswani", &docs)
            .expect("ok");
        let order: Vec<&str> = m.iter().map(|d| d.content.as_str()).collect();
        assert_eq!(order0, order);
        assert_eq!(sq0, sq);
    }
}

// ── config ────────────────────────────────────────────────────────────────────

#[test]
fn test_config_default_strips() {
    assert!(SelfQueryConfig::default().strip_filter_phrases);
}

#[test]
fn test_config_builder_toggle() {
    let cfg = SelfQueryConfig::default().with_strip_filter_phrases(false);
    assert!(!cfg.strip_filter_phrases);
}

// ── error trait ───────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    let err = SelfQueryError::EmptyQuery;
    assert_eq!(err.to_string(), "query must not be empty");
}
