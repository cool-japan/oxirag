#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::format_push_string
)]
//! Tests for the `anonymization` module.

use super::anonymizer::Anonymizer;
use super::types::{AnonConfig, AnonError, AnonMapping, PiiKind};

// ── helpers ─────────────────────────────────────────────────────────────────

fn anon() -> Anonymizer {
    Anonymizer::new(AnonConfig::default())
}

// ── PiiKind ─────────────────────────────────────────────────────────────────

#[test]
fn test_pii_kind_as_str_person() {
    assert_eq!(PiiKind::Person.as_str(), "PERSON");
}

#[test]
fn test_pii_kind_as_str_email() {
    assert_eq!(PiiKind::Email.as_str(), "EMAIL");
}

#[test]
fn test_pii_kind_as_str_phone() {
    assert_eq!(PiiKind::Phone.as_str(), "PHONE");
}

#[test]
fn test_pii_kind_equality() {
    assert_eq!(PiiKind::Email, PiiKind::Email);
    assert_ne!(PiiKind::Email, PiiKind::Phone);
}

// ── AnonConfig defaults + builders ──────────────────────────────────────────

#[test]
fn test_config_default_all_enabled() {
    let config = AnonConfig::default();
    assert!(config.anonymize_persons);
    assert!(config.anonymize_emails);
    assert!(config.anonymize_phones);
}

#[test]
fn test_config_new_matches_default() {
    assert_eq!(AnonConfig::new(), AnonConfig::default());
}

#[test]
fn test_config_builder_persons_off() {
    let config = AnonConfig::new().with_anonymize_persons(false);
    assert!(!config.anonymize_persons);
    assert!(config.anonymize_emails);
    assert!(config.anonymize_phones);
}

#[test]
fn test_config_builder_emails_off() {
    let config = AnonConfig::new().with_anonymize_emails(false);
    assert!(!config.anonymize_emails);
    assert!(config.anonymize_persons);
}

#[test]
fn test_config_builder_phones_off() {
    let config = AnonConfig::new().with_anonymize_phones(false);
    assert!(!config.anonymize_phones);
    assert!(config.anonymize_emails);
}

#[test]
fn test_config_builder_chained() {
    let config = AnonConfig::new()
        .with_anonymize_persons(false)
        .with_anonymize_emails(false)
        .with_anonymize_phones(false);
    assert!(!config.anonymize_persons);
    assert!(!config.anonymize_emails);
    assert!(!config.anonymize_phones);
}

#[test]
fn test_config_is_enabled() {
    let config = AnonConfig::new().with_anonymize_emails(false);
    assert!(!config.is_enabled(PiiKind::Email));
    assert!(config.is_enabled(PiiKind::Phone));
    assert!(config.is_enabled(PiiKind::Person));
}

// ── email anonymization ─────────────────────────────────────────────────────

#[test]
fn test_email_replaced_with_placeholder() {
    let result = anon().anonymize("Contact us at test@example.com today");
    assert!(result.text.contains("[EMAIL_1]"), "{}", result.text);
    assert!(!result.text.contains("test@example.com"), "{}", result.text);
}

#[test]
fn test_email_mapping_records_original() {
    let result = anon().anonymize("test@example.com");
    assert_eq!(
        result.mapping.original_for("[EMAIL_1]"),
        Some("test@example.com")
    );
}

#[test]
fn test_same_email_twice_same_placeholder() {
    let result = anon().anonymize("a@b.com and again a@b.com here");
    let count = result.text.matches("[EMAIL_1]").count();
    assert_eq!(count, 2, "{}", result.text);
    assert!(!result.text.contains("[EMAIL_2]"), "{}", result.text);
    assert_eq!(result.mapping.len(), 1);
}

#[test]
fn test_two_different_emails_distinct_placeholders() {
    let result = anon().anonymize("first a@b.com then c@d.org");
    assert!(result.text.contains("[EMAIL_1]"), "{}", result.text);
    assert!(result.text.contains("[EMAIL_2]"), "{}", result.text);
    assert_eq!(result.mapping.original_for("[EMAIL_1]"), Some("a@b.com"));
    assert_eq!(result.mapping.original_for("[EMAIL_2]"), Some("c@d.org"));
}

#[test]
fn test_email_with_plus_and_dots() {
    let result = anon().anonymize("ping user.name+tag@sub.example.co");
    assert!(result.text.contains("[EMAIL_1]"), "{}", result.text);
    assert_eq!(
        result.mapping.original_for("[EMAIL_1]"),
        Some("user.name+tag@sub.example.co")
    );
}

#[test]
fn test_email_disabled_left_untouched() {
    let config = AnonConfig::new().with_anonymize_emails(false);
    let result = Anonymizer::new(config).anonymize("write to keep@me.com please");
    assert!(result.text.contains("keep@me.com"), "{}", result.text);
    assert!(!result.text.contains("[EMAIL_1]"), "{}", result.text);
}

// ── phone anonymization ─────────────────────────────────────────────────────

#[test]
fn test_phone_replaced_with_placeholder() {
    let result = anon().anonymize("call 555-123-4567 now");
    assert!(result.text.contains("[PHONE_1]"), "{}", result.text);
    assert!(!result.text.contains("555-123-4567"), "{}", result.text);
}

#[test]
fn test_phone_plain_digits() {
    let result = anon().anonymize("number 5551234567 here");
    assert!(result.text.contains("[PHONE_1]"), "{}", result.text);
}

#[test]
fn test_phone_seven_digit_run_detected() {
    let result = anon().anonymize("dial 123-4567 ok");
    assert!(result.text.contains("[PHONE_1]"), "{}", result.text);
}

#[test]
fn test_short_digit_run_not_phone() {
    let result = anon().anonymize("order 12345 placed");
    assert!(!result.text.contains("[PHONE_1]"), "{}", result.text);
    assert!(result.text.contains("12345"), "{}", result.text);
}

#[test]
fn test_same_phone_twice_same_placeholder() {
    let result = anon().anonymize("555-123-4567 / 555-123-4567");
    let count = result.text.matches("[PHONE_1]").count();
    assert_eq!(count, 2, "{}", result.text);
    assert!(!result.text.contains("[PHONE_2]"), "{}", result.text);
}

#[test]
fn test_two_different_phones_distinct() {
    let result = anon().anonymize("a 555-123-4567 b 555-987-6543");
    assert!(result.text.contains("[PHONE_1]"), "{}", result.text);
    assert!(result.text.contains("[PHONE_2]"), "{}", result.text);
}

#[test]
fn test_phone_disabled_left_untouched() {
    let config = AnonConfig::new().with_anonymize_phones(false);
    let result = Anonymizer::new(config).anonymize("keep 555-123-4567 raw");
    assert!(result.text.contains("555-123-4567"), "{}", result.text);
    assert!(!result.text.contains("[PHONE_1]"), "{}", result.text);
}

// ── person anonymization ────────────────────────────────────────────────────

#[test]
fn test_person_replaced_with_placeholder() {
    let config = AnonConfig::new()
        .with_anonymize_emails(false)
        .with_anonymize_phones(false);
    let result = Anonymizer::new(config).anonymize("hello Alice there");
    assert!(result.text.contains("[PERSON_1]"), "{}", result.text);
    assert!(!result.text.contains("Alice"), "{}", result.text);
}

#[test]
fn test_person_full_name_single_placeholder() {
    let config = AnonConfig::new()
        .with_anonymize_emails(false)
        .with_anonymize_phones(false);
    let result = Anonymizer::new(config).anonymize("from Alice Smith regards");
    assert!(result.text.contains("[PERSON_1]"), "{}", result.text);
    assert_eq!(
        result.mapping.original_for("[PERSON_1]"),
        Some("Alice Smith")
    );
}

#[test]
fn test_same_person_twice_same_placeholder() {
    let config = AnonConfig::new()
        .with_anonymize_emails(false)
        .with_anonymize_phones(false);
    let result = Anonymizer::new(config).anonymize("Bob met Bob");
    let count = result.text.matches("[PERSON_1]").count();
    assert_eq!(count, 2, "{}", result.text);
}

#[test]
fn test_person_disabled_left_untouched() {
    let config = AnonConfig::new().with_anonymize_persons(false);
    let result = Anonymizer::new(config).anonymize("hello Alice there");
    assert!(result.text.contains("Alice"), "{}", result.text);
    assert!(!result.text.contains("[PERSON_1]"), "{}", result.text);
}

#[test]
fn test_lowercase_word_not_person() {
    let config = AnonConfig::new()
        .with_anonymize_emails(false)
        .with_anonymize_phones(false);
    let result = Anonymizer::new(config).anonymize("just a word");
    assert!(!result.text.contains("[PERSON_1]"), "{}", result.text);
}

// ── round-trip de-anonymization ─────────────────────────────────────────────

#[test]
fn test_roundtrip_email() {
    let original = "Reach me at jane@example.com anytime";
    let result = anon().anonymize(original);
    let restored = anon().deanonymize(&result.text, &result.mapping);
    assert_eq!(restored, original);
}

#[test]
fn test_roundtrip_phone() {
    let original = "phone: 555-123-4567 end";
    let result = anon().anonymize(original);
    let restored = anon().deanonymize(&result.text, &result.mapping);
    assert_eq!(restored, original);
}

#[test]
fn test_roundtrip_repeated_email() {
    let original = "a@b.com talks to a@b.com twice";
    let result = anon().anonymize(original);
    let restored = anon().deanonymize(&result.text, &result.mapping);
    assert_eq!(restored, original);
}

#[test]
fn test_roundtrip_two_emails() {
    let original = "one a@b.com two c@d.org done";
    let result = anon().anonymize(original);
    let restored = anon().deanonymize(&result.text, &result.mapping);
    assert_eq!(restored, original);
}

#[test]
fn test_roundtrip_mixed_kinds() {
    let config = AnonConfig::new();
    let a = Anonymizer::new(config);
    let original = "Email jane@example.com or call 555-123-4567";
    let result = a.anonymize(original);
    let restored = a.deanonymize(&result.text, &result.mapping);
    assert_eq!(restored, original);
}

#[test]
fn test_deanonymize_unknown_placeholder_unchanged() {
    let mapping = AnonMapping::new();
    let restored = anon().deanonymize("text [EMAIL_9] stays", &mapping);
    assert_eq!(restored, "text [EMAIL_9] stays");
}

#[test]
fn test_deanonymize_handles_ten_plus_placeholders() {
    // Build an input with 12 distinct e-mails so [EMAIL_10]..[EMAIL_12] exist,
    // then confirm a full round-trip is not corrupted by [EMAIL_1] prefixes.
    let mut input = String::new();
    for i in 1..=12 {
        input.push_str(&format!("u{i}@x.com "));
    }
    let input = input.trim_end();
    let result = anon().anonymize(input);
    assert!(result.text.contains("[EMAIL_10]"), "{}", result.text);
    assert!(result.text.contains("[EMAIL_12]"), "{}", result.text);
    let restored = anon().deanonymize(&result.text, &result.mapping);
    assert_eq!(restored, input);
}

// ── AnonMapping API ─────────────────────────────────────────────────────────

#[test]
fn test_mapping_new_is_empty() {
    let mapping = AnonMapping::new();
    assert!(mapping.is_empty());
    assert_eq!(mapping.len(), 0);
}

#[test]
fn test_mapping_len_after_anonymize() {
    let result = anon().anonymize("a@b.com and c@d.org");
    assert_eq!(result.mapping.len(), 2);
    assert!(!result.mapping.is_empty());
}

#[test]
fn test_mapping_placeholder_for() {
    let result = anon().anonymize("a@b.com here");
    assert_eq!(result.mapping.placeholder_for("a@b.com"), Some("[EMAIL_1]"));
    assert_eq!(result.mapping.placeholder_for("missing@x.com"), None);
}

#[test]
fn test_mapping_original_for_missing() {
    let result = anon().anonymize("a@b.com here");
    assert_eq!(result.mapping.original_for("[EMAIL_99]"), None);
}

#[test]
fn test_mapping_entries_ordered() {
    let result = anon().anonymize("a@b.com then c@d.org");
    let entries: Vec<(&str, &str)> = result.mapping.entries().collect();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0], ("[EMAIL_1]", "a@b.com"));
    assert_eq!(entries[1], ("[EMAIL_2]", "c@d.org"));
}

#[test]
fn test_mapping_pairs() {
    let result = anon().anonymize("a@b.com here");
    let pairs = result.mapping.pairs();
    assert_eq!(
        pairs,
        vec![("[EMAIL_1]".to_string(), "a@b.com".to_string())]
    );
}

#[test]
fn test_mapping_roundtrip_bidirectional() {
    let result = anon().anonymize("a@b.com here");
    let placeholder = result.mapping.placeholder_for("a@b.com").unwrap();
    assert_eq!(result.mapping.original_for(placeholder), Some("a@b.com"));
}

// ── empty + checked + determinism ───────────────────────────────────────────

#[test]
fn test_empty_input_unchanged() {
    let result = anon().anonymize("");
    assert_eq!(result.text, "");
    assert!(result.mapping.is_empty());
}

#[test]
fn test_anonymize_checked_errors_on_empty() {
    let err = anon().anonymize_checked("").unwrap_err();
    assert_eq!(err, AnonError::Empty);
}

#[test]
fn test_anonymize_checked_ok_on_nonempty() {
    let result = anon().anonymize_checked("a@b.com").unwrap();
    assert!(result.text.contains("[EMAIL_1]"));
}

#[test]
fn test_anon_error_display() {
    let msg = AnonError::Empty.to_string();
    assert!(msg.contains("empty"), "{msg}");
}

#[test]
fn test_no_pii_text_unchanged() {
    let result = anon().anonymize("the quick brown fox");
    // Capitalized-only at sentence start would not trigger here (all lowercase).
    assert_eq!(result.text, "the quick brown fox");
    assert!(result.mapping.is_empty());
}

#[test]
fn test_determinism_same_input_same_output() {
    let input = "Email jane@example.com or call 555-123-4567 for Alice Smith";
    let first = anon().anonymize(input);
    let second = anon().anonymize(input);
    assert_eq!(first.text, second.text);
    assert_eq!(first.mapping, second.mapping);
}

#[test]
fn test_determinism_mapping_pairs_stable() {
    let input = "a@b.com c@d.org e@f.net";
    let first = anon().anonymize(input).mapping.pairs();
    let second = anon().anonymize(input).mapping.pairs();
    assert_eq!(first, second);
}

#[test]
fn test_anonymizer_default_constructs() {
    let a = Anonymizer::default();
    assert!(a.config().anonymize_emails);
}

#[test]
fn test_config_accessor() {
    let a = anon();
    assert_eq!(*a.config(), AnonConfig::default());
}

#[test]
fn test_unicode_text_preserved() {
    let original = "café owner jane@example.com 日本語";
    let result = anon().anonymize(original);
    assert!(result.text.contains("café"), "{}", result.text);
    assert!(result.text.contains("日本語"), "{}", result.text);
    let restored = anon().deanonymize(&result.text, &result.mapping);
    assert_eq!(restored, original);
}

#[test]
fn test_multiple_kinds_independent_counters() {
    let input = "a@b.com 555-123-4567 c@d.org 555-987-6543";
    let result = anon().anonymize(input);
    assert!(result.text.contains("[EMAIL_1]"), "{}", result.text);
    assert!(result.text.contains("[EMAIL_2]"), "{}", result.text);
    assert!(result.text.contains("[PHONE_1]"), "{}", result.text);
    assert!(result.text.contains("[PHONE_2]"), "{}", result.text);
}
