#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]
//! Tests for the attribute model, the bound model, and the predicate AST.

use super::super::predicate::FilterPredicate;
use super::super::types::{AttrValue, FilterBound, FilteredMetadata, FilteredSearchError};

// ═══════════════════════════════════════════════════════════════════════════
// AttrValue
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn attr_value_equality_is_type_tagged() {
    assert_ne!(AttrValue::Int(3), AttrValue::Float(3.0));
    assert_ne!(AttrValue::Int(1), AttrValue::Bool(true));
    assert_ne!(AttrValue::Str("1".into()), AttrValue::Int(1));
    assert_eq!(AttrValue::Int(3), AttrValue::Int(3));
    assert_eq!(AttrValue::Float(3.0), AttrValue::Float(3.0));
}

#[test]
fn attr_value_float_equality_is_canonicalized() {
    // `Eq` demands reflexivity, which raw IEEE-754 NaN does not provide; the
    // canonicalized bit key restores it.
    assert_eq!(AttrValue::Float(f64::NAN), AttrValue::Float(f64::NAN));
    // ...and `-0.0 == 0.0`, as arithmetic (though not raw bit comparison)
    // demands.
    assert_eq!(AttrValue::Float(-0.0), AttrValue::Float(0.0));
    assert_ne!(AttrValue::Float(0.0), AttrValue::Float(1e-300));
}

#[test]
fn attr_value_hash_agrees_with_equality() {
    use std::collections::HashMap;
    let mut map: HashMap<AttrValue, u32> = HashMap::new();
    map.insert(AttrValue::Float(-0.0), 1);
    // Equal keys must collide in the map, or the inverted index would split one
    // logical value across two posting lists.
    assert_eq!(map.get(&AttrValue::Float(0.0)), Some(&1));

    map.insert(AttrValue::Float(f64::NAN), 2);
    assert_eq!(map.get(&AttrValue::Float(f64::NAN)), Some(&2));

    map.insert(AttrValue::Int(3), 3);
    // Type-tagged: `Float(3.0)` must not find `Int(3)`.
    assert_eq!(map.get(&AttrValue::Float(3.0)), None);
    assert_eq!(map.get(&AttrValue::Int(3)), Some(&3));
}

#[test]
fn attr_value_numeric_projection() {
    assert_eq!(AttrValue::Int(7).as_f64(), Some(7.0));
    assert_eq!(AttrValue::Float(0.5).as_f64(), Some(0.5));
    assert_eq!(AttrValue::Str("7".into()).as_f64(), None);
    assert_eq!(AttrValue::Bool(true).as_f64(), None);
    assert!(AttrValue::Int(1).is_numeric());
    assert!(!AttrValue::Bool(true).is_numeric());
    assert_eq!(AttrValue::Str("x".into()).as_str(), Some("x"));
    assert_eq!(AttrValue::Bool(false).as_bool(), Some(false));
    assert_eq!(AttrValue::Int(1).kind_name(), "int");
}

#[test]
fn attr_value_from_conversions() {
    assert_eq!(AttrValue::from("x"), AttrValue::Str("x".into()));
    assert_eq!(
        AttrValue::from(String::from("y")),
        AttrValue::Str("y".into())
    );
    assert_eq!(AttrValue::from(3_i64), AttrValue::Int(3));
    assert_eq!(AttrValue::from(1.5_f64), AttrValue::Float(1.5));
    assert_eq!(AttrValue::from(true), AttrValue::Bool(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// FilteredMetadata
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn metadata_builder_and_accessors() {
    let mut metadata = FilteredMetadata::new()
        .with("lang", "ja")
        .with("year", 2024_i64);
    assert_eq!(metadata.len(), 2);
    assert!(!metadata.is_empty());
    assert!(metadata.contains("lang"));
    assert!(!metadata.contains("missing"));
    assert_eq!(metadata.get("year"), Some(&AttrValue::Int(2024)));

    let previous = metadata.insert("lang", "en");
    assert_eq!(previous, Some(AttrValue::Str("ja".into())));
    assert_eq!(metadata.get("lang"), Some(&AttrValue::Str("en".into())));

    assert!(FilteredMetadata::new().is_empty());
    assert_eq!(metadata.iter().count(), 2);
    assert_eq!((&metadata).into_iter().count(), 2);
}

#[test]
fn metadata_from_iterator() {
    let metadata: FilteredMetadata = vec![
        ("a".to_string(), AttrValue::Int(1)),
        ("b".to_string(), AttrValue::Bool(true)),
    ]
    .into_iter()
    .collect();
    assert_eq!(metadata.len(), 2);
    assert_eq!(metadata.get("a"), Some(&AttrValue::Int(1)));
}

// ═══════════════════════════════════════════════════════════════════════════
// FilterBound
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn filter_bound_acceptance() {
    assert!(FilterBound::Inclusive(5.0).accepts_as_lower(5.0));
    assert!(!FilterBound::Exclusive(5.0).accepts_as_lower(5.0));
    assert!(FilterBound::Exclusive(5.0).accepts_as_lower(5.1));
    assert!(FilterBound::Unbounded.accepts_as_lower(f64::NEG_INFINITY));

    assert!(FilterBound::Inclusive(5.0).accepts_as_upper(5.0));
    assert!(!FilterBound::Exclusive(5.0).accepts_as_upper(5.0));
    assert!(FilterBound::Exclusive(5.0).accepts_as_upper(4.9));
    assert!(FilterBound::Unbounded.accepts_as_upper(f64::INFINITY));

    assert_eq!(FilterBound::Inclusive(2.0).value(), Some(2.0));
    assert_eq!(FilterBound::Unbounded.value(), None);
    assert_eq!(FilterBound::Unbounded.as_lower_f64(), f64::NEG_INFINITY);
    assert_eq!(FilterBound::Unbounded.as_upper_f64(), f64::INFINITY);
}

// ═══════════════════════════════════════════════════════════════════════════
// Predicate AST: one section per operator, then nesting
// ═══════════════════════════════════════════════════════════════════════════

fn sample_metadata() -> FilteredMetadata {
    FilteredMetadata::new()
        .with("lang", "ja")
        .with("year", 2024_i64)
        .with("score", 0.75_f64)
        .with("public", true)
}

#[test]
fn predicate_eq() {
    let metadata = sample_metadata();
    assert!(FilterPredicate::eq("lang", "ja").matches(&metadata));
    assert!(!FilterPredicate::eq("lang", "en").matches(&metadata));
    assert!(FilterPredicate::eq("year", 2024_i64).matches(&metadata));
    assert!(FilterPredicate::eq("public", true).matches(&metadata));
    // Type-tagged: the year is an `Int`, so a `Float` never equals it.
    assert!(!FilterPredicate::eq("year", 2024.0_f64).matches(&metadata));
    // A missing attribute fails every leaf.
    assert!(!FilterPredicate::eq("missing", "x").matches(&metadata));
}

#[test]
fn predicate_ne_requires_presence_but_not_not_eq() {
    let metadata = sample_metadata();
    assert!(FilterPredicate::ne("lang", "en").matches(&metadata));
    assert!(!FilterPredicate::ne("lang", "ja").matches(&metadata));

    // The load-bearing distinction: `Ne` demands the attribute be present,
    // `Not(Eq)` does not. Conflating them is a classic silent-filter bug.
    let missing = FilterPredicate::ne("absent", "x");
    let negated = FilterPredicate::not(FilterPredicate::eq("absent", "x"));
    assert!(!missing.matches(&metadata));
    assert!(negated.matches(&metadata));
}

#[test]
fn predicate_range_inclusive_and_exclusive() {
    let metadata = sample_metadata();

    assert!(FilterPredicate::range_inclusive("year", 2020.0, 2024.0).matches(&metadata));
    assert!(FilterPredicate::range_inclusive("year", 2024.0, 2024.0).matches(&metadata));
    assert!(!FilterPredicate::range_inclusive("year", 2020.0, 2023.0).matches(&metadata));

    assert!(
        FilterPredicate::range(
            "year",
            FilterBound::Exclusive(2023.0),
            FilterBound::Exclusive(2025.0)
        )
        .matches(&metadata)
    );
    assert!(
        !FilterPredicate::range(
            "year",
            FilterBound::Exclusive(2024.0),
            FilterBound::Unbounded
        )
        .matches(&metadata)
    );

    assert!(FilterPredicate::at_least("year", 2024.0).matches(&metadata));
    assert!(FilterPredicate::at_most("year", 2024.0).matches(&metadata));
    assert!(!FilterPredicate::at_least("year", 2025.0).matches(&metadata));

    // Ranges coerce numerically across `Int` and `Float`.
    assert!(FilterPredicate::range_inclusive("score", 0.5, 1.0).matches(&metadata));
    // ...but a non-numeric attribute satisfies no range at all.
    assert!(!FilterPredicate::range_inclusive("lang", 0.0, 1e9).matches(&metadata));
    assert!(!FilterPredicate::range_inclusive("public", 0.0, 1e9).matches(&metadata));
    // Absent attribute.
    assert!(!FilterPredicate::range_inclusive("absent", 0.0, 1.0).matches(&metadata));
}

#[test]
fn predicate_range_rejects_nan_attribute_values() {
    let metadata = FilteredMetadata::new().with("value", f64::NAN);
    // Every comparison against NaN is false, so a NaN value satisfies no bound.
    assert!(!FilterPredicate::range_inclusive("value", -1e9, 1e9).matches(&metadata));
    assert!(!FilterPredicate::at_least("value", f64::NEG_INFINITY).matches(&metadata));
    // ...but it is still *present*.
    assert!(FilterPredicate::exists("value").matches(&metadata));
}

#[test]
fn predicate_in_set() {
    let metadata = sample_metadata();
    assert!(FilterPredicate::in_set("lang", ["en", "ja"]).matches(&metadata));
    assert!(!FilterPredicate::in_set("lang", ["en", "fr"]).matches(&metadata));
    assert!(FilterPredicate::in_set("year", [2023_i64, 2024_i64]).matches(&metadata));
    // An empty set is `Or` over nothing: it matches nothing.
    assert!(!FilterPredicate::in_set("lang", Vec::<&str>::new()).matches(&metadata));
    assert!(!FilterPredicate::in_set("absent", ["x"]).matches(&metadata));
}

#[test]
fn predicate_exists() {
    let metadata = sample_metadata();
    assert!(FilterPredicate::exists("lang").matches(&metadata));
    assert!(!FilterPredicate::exists("absent").matches(&metadata));
    // Presence, not truthiness: a `false` boolean still exists.
    let falsy = FilteredMetadata::new().with("flag", false);
    assert!(FilterPredicate::exists("flag").matches(&falsy));
}

#[test]
fn predicate_and_or_not_and_empty_connectives() {
    let metadata = sample_metadata();

    assert!(
        FilterPredicate::and([
            FilterPredicate::eq("lang", "ja"),
            FilterPredicate::at_least("year", 2020.0),
        ])
        .matches(&metadata)
    );
    assert!(
        !FilterPredicate::and([
            FilterPredicate::eq("lang", "ja"),
            FilterPredicate::at_least("year", 2025.0),
        ])
        .matches(&metadata)
    );

    assert!(
        FilterPredicate::or([
            FilterPredicate::eq("lang", "en"),
            FilterPredicate::eq("lang", "ja"),
        ])
        .matches(&metadata)
    );
    assert!(
        !FilterPredicate::or([
            FilterPredicate::eq("lang", "en"),
            FilterPredicate::eq("lang", "fr"),
        ])
        .matches(&metadata)
    );

    assert!(FilterPredicate::not(FilterPredicate::eq("lang", "en")).matches(&metadata));
    assert!(!FilterPredicate::not(FilterPredicate::eq("lang", "ja")).matches(&metadata));

    // Empty conjunction is vacuously true; empty disjunction vacuously false.
    assert!(FilterPredicate::all().matches(&metadata));
    assert!(!FilterPredicate::none().matches(&metadata));
    assert!(FilterPredicate::all().is_unconstrained());
    assert!(!FilterPredicate::none().is_unconstrained());
}

#[test]
fn predicate_deeply_nested_composition() {
    let metadata = sample_metadata();

    // (lang in {ja, ko} AND NOT(year < 2020)) OR (public == false AND score > 0.9)
    let predicate = FilterPredicate::or([
        FilterPredicate::and([
            FilterPredicate::in_set("lang", ["ja", "ko"]),
            FilterPredicate::not(FilterPredicate::range(
                "year",
                FilterBound::Unbounded,
                FilterBound::Exclusive(2020.0),
            )),
        ]),
        FilterPredicate::and([
            FilterPredicate::eq("public", false),
            FilterPredicate::range("score", FilterBound::Exclusive(0.9), FilterBound::Unbounded),
        ]),
    ]);
    assert!(predicate.matches(&metadata));

    // Flip the year out of range and the left disjunct fails; the right one
    // already failed, so the whole thing must fail.
    let older = FilteredMetadata::new()
        .with("lang", "ja")
        .with("year", 2019_i64)
        .with("score", 0.75_f64)
        .with("public", true);
    assert!(!predicate.matches(&older));

    // De Morgan, checked against the evaluator rather than assumed.
    let left = FilterPredicate::not(FilterPredicate::and([
        FilterPredicate::eq("lang", "ja"),
        FilterPredicate::eq("public", true),
    ]));
    let right = FilterPredicate::or([
        FilterPredicate::not(FilterPredicate::eq("lang", "ja")),
        FilterPredicate::not(FilterPredicate::eq("public", true)),
    ]);
    for candidate in [&metadata, &older] {
        assert_eq!(left.matches(candidate), right.matches(candidate));
    }
}

#[test]
fn predicate_introspection() {
    let predicate = FilterPredicate::and([
        FilterPredicate::eq("lang", "ja"),
        FilterPredicate::not(FilterPredicate::exists("tag")),
        FilterPredicate::or([
            FilterPredicate::at_least("year", 2020.0),
            FilterPredicate::eq("lang", "en"),
        ]),
    ]);
    assert_eq!(
        predicate.referenced_attributes(),
        vec!["lang".to_string(), "tag".to_string(), "year".to_string()]
    );
    // 1 And + 1 Eq + 1 Not + 1 Exists + 1 Or + 2 leaves = 7.
    assert_eq!(predicate.node_count(), 7);
    assert!(predicate.requires_independence_assumption());

    let simple = FilterPredicate::eq("lang", "ja");
    assert!(!simple.requires_independence_assumption());
    assert_eq!(simple.node_count(), 1);
    // A single-child `And` folds no independent estimates together.
    assert!(!FilterPredicate::and([simple]).requires_independence_assumption());
}

#[test]
fn predicate_validation_rejects_malformed_asts() {
    assert!(FilterPredicate::eq("lang", "ja").validate().is_ok());
    assert!(FilterPredicate::all().validate().is_ok());

    let empty_name = FilterPredicate::eq("", "ja");
    assert!(matches!(
        empty_name.validate(),
        Err(FilteredSearchError::InvalidPredicate { .. })
    ));

    let inverted = FilterPredicate::range_inclusive("year", 2025.0, 2020.0);
    assert!(matches!(
        inverted.validate(),
        Err(FilteredSearchError::InvalidPredicate { .. })
    ));

    // A zero-width *exclusive* interval admits nothing; that is a caller bug,
    // not a filter that "just returns nothing".
    let degenerate = FilterPredicate::range(
        "year",
        FilterBound::Exclusive(2020.0),
        FilterBound::Inclusive(2020.0),
    );
    assert!(matches!(
        degenerate.validate(),
        Err(FilteredSearchError::InvalidPredicate { .. })
    ));
    // ...but a zero-width *inclusive* interval is a legitimate point query.
    assert!(
        FilterPredicate::range_inclusive("year", 2020.0, 2020.0)
            .validate()
            .is_ok()
    );

    let nan_bound = FilterPredicate::at_least("year", f64::NAN);
    assert!(matches!(
        nan_bound.validate(),
        Err(FilteredSearchError::InvalidPredicate { .. })
    ));

    let duplicated = FilterPredicate::in_set("lang", ["ja", "en", "ja"]);
    assert!(matches!(
        duplicated.validate(),
        Err(FilteredSearchError::InvalidPredicate { .. })
    ));

    // Validation recurses through every connective.
    let nested = FilterPredicate::not(FilterPredicate::or([
        FilterPredicate::eq("lang", "ja"),
        FilterPredicate::and([FilterPredicate::eq("", "x")]),
    ]));
    assert!(nested.validate().is_err());
}
