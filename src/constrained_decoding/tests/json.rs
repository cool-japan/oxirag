//! JSON Schema compilation, checked against `serde_json` — an oracle this crate did
//! not write.
//!
//! # Headline (e)
//!
//! For a compiled schema, *every* string the masked decoder can emit must
//!
//! 1. parse: `serde_json::from_str::<Value>()` succeeds, and
//! 2. conform: an independent, hand-written check of *that* schema's constraints
//!    holds on the parsed value.
//!
//! Both halves are enumerated exhaustively over a bounded token count. `serde_json`
//! decides half of it; the other half is a validator written per-schema in this file,
//! deliberately not sharing a line with the compiler under test.
//!
//! # A note on token budgets
//!
//! The vocabulary is byte-per-token, so a document costs one token per byte, and the
//! exhaustive walk's depth budget is set per test to reach the documents that matter.
//! Where a schema has an *unbounded* free repetition (a bare integer's digits, a
//! string's characters) the budget is kept small, because the walk explores every
//! prefix — including the long digit-runs that can never terminate in time — and that
//! is exponential in the budget. Structural schemas (objects, arrays, nesting) instead
//! use rigid boolean or enum leaves, which have no free repetition, so a deep budget
//! there stays cheap. Integer-inside-structure is covered separately, by membership
//! rather than enumeration, in [`object_with_integer_field_membership`].

use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::constrained_decoding::engine::ConstrainedDecoder;
use crate::constrained_decoding::json_schema::compile_json_schema;
use crate::constrained_decoding::tests::dfa_from_ast;
use crate::constrained_decoding::types::{
    ConstrainedDecoderConfig, ConstrainedDecodingError, Constraint, JsonSchemaOptions,
    PropertyOrder, StateId, StaticVocabulary,
};

/// A byte-per-token JSON vocabulary: every lowercase letter, every digit, and the JSON
/// punctuation, each as its own single-byte token, plus an end-of-sequence token last.
fn json_vocab() -> (StaticVocabulary, u32) {
    let mut tokens: Vec<String> = Vec::new();
    for byte in b'a'..=b'z' {
        tokens.push((byte as char).to_string());
    }
    for byte in b'0'..=b'9' {
        tokens.push((byte as char).to_string());
    }
    for symbol in ['"', '{', '}', '[', ']', ':', ',', '-', '.', 'E', '+'] {
        tokens.push(symbol.to_string());
    }
    tokens.push("<eos>".to_string());
    let eos = (tokens.len() - 1) as u32;
    let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
    (StaticVocabulary::from_strs(&refs).expect("non-empty"), eos)
}

/// Walk the legal-token tree and collect every string the decoder can terminate on
/// within `max_tokens` tokens.
fn terminable_outputs(decoder: &ConstrainedDecoder, max_tokens: usize) -> BTreeSet<String> {
    fn walk(
        decoder: &ConstrainedDecoder,
        state: StateId,
        prefix: &mut Vec<u8>,
        depth: usize,
        max_tokens: usize,
        out: &mut BTreeSet<String>,
    ) {
        if decoder.is_accepting(state) {
            out.insert(String::from_utf8(prefix.clone()).expect("automaton emits valid utf-8"));
        }
        if depth >= max_tokens {
            return;
        }
        for token in decoder.allowed_tokens(state).unwrap().allowed() {
            if decoder.is_eos(token) {
                continue;
            }
            let target = decoder.step(state, token).unwrap();
            let restore = prefix.len();
            prefix.extend_from_slice(decoder.token_bytes(token));
            walk(decoder, target, prefix, depth + 1, max_tokens, out);
            prefix.truncate(restore);
        }
    }
    let mut out = BTreeSet::new();
    let mut prefix = Vec::new();
    walk(
        decoder,
        decoder.start_state(),
        &mut prefix,
        0,
        max_tokens,
        &mut out,
    );
    out
}

/// The core of headline (e): compile `schema`, enumerate the decoder's outputs, and
/// assert each parses and conforms. Returns the outputs for further assertions.
fn assert_outputs_parse_and_conform(
    schema: &Value,
    max_tokens: usize,
    conforms: impl Fn(&Value) -> bool,
) -> BTreeSet<String> {
    let (vocab, eos) = json_vocab();
    let config = ConstrainedDecoderConfig::default().with_eos(eos);
    let decoder =
        ConstrainedDecoder::compile(&Constraint::json_schema(schema.clone()), &vocab, &config)
            .expect("schema compiles");

    let outputs = terminable_outputs(&decoder, max_tokens);
    assert!(!outputs.is_empty(), "schema emitted nothing: {schema}");
    for output in &outputs {
        let value: Value = serde_json::from_str(output).unwrap_or_else(|error| {
            panic!("serde_json rejects decoder output {output:?}: {error}")
        });
        assert!(
            conforms(&value),
            "decoder output {output:?} parsed but violates the schema {schema}",
        );
    }
    outputs
}

/// **Boolean.** Only `true` and `false`, nothing else.
#[test]
fn schema_boolean() {
    let outputs =
        assert_outputs_parse_and_conform(&json!({"type": "boolean"}), 5, Value::is_boolean);
    assert_eq!(
        outputs,
        BTreeSet::from(["true".to_string(), "false".to_string()])
    );
}

/// **Null.** Only `null`.
#[test]
fn schema_null() {
    let outputs = assert_outputs_parse_and_conform(&json!({"type": "null"}), 5, Value::is_null);
    assert_eq!(outputs, BTreeSet::from(["null".to_string()]));
}

/// **Integer.** Every output parses as a JSON integer, and none has a leading zero or
/// a lone/`-0` sign — the failure modes of a hand-rolled number grammar. Enumerated to
/// three tokens, which reaches two-digit magnitudes and the negative sign.
///
/// The `-0` case is not hypothetical: an earlier `-?(0|[1-9][0-9]*)` integer grammar
/// admitted it, `serde_json` read it back as the float `-0.0`, and this enumeration is
/// exactly what caught it.
#[test]
fn schema_integer() {
    let outputs = assert_outputs_parse_and_conform(&json!({"type": "integer"}), 3, |value| {
        value.is_i64() || value.is_u64()
    });
    assert!(outputs.contains("0"));
    assert!(outputs.contains("9"));
    assert!(outputs.contains("-1"));
    assert!(outputs.contains("10"));
    // Non-canonical integers must never appear.
    assert!(!outputs.contains("01"));
    assert!(!outputs.contains("00"));
    assert!(!outputs.contains("-0"));
    assert!(!outputs.contains("-"));
}

/// **Number.** Fractions and exponents, all serde-parseable as numbers.
#[test]
fn schema_number() {
    let outputs = assert_outputs_parse_and_conform(&json!({"type": "number"}), 4, Value::is_number);
    assert!(outputs.contains("0"));
    // No number starts with a dot nor ends with one: `.5` and `5.` are not JSON.
    for output in &outputs {
        assert!(!output.starts_with('.'), "{output} starts with a dot");
        assert!(!output.ends_with('.'), "{output} ends with a dot");
    }
}

/// **Enum.** Exactly the enumerated values, in their canonical `serde_json`
/// serialisations, across mixed types.
#[test]
fn schema_enum() {
    let schema = json!({"enum": ["yes", "no", 7, true, null]});
    let allowed: Vec<Value> = vec![
        json!("yes"),
        json!("no"),
        json!(7),
        json!(true),
        json!(null),
    ];
    let outputs = assert_outputs_parse_and_conform(&schema, 5, |value| allowed.contains(value));
    assert_eq!(
        outputs,
        BTreeSet::from([
            "\"yes\"".to_string(),
            "\"no\"".to_string(),
            "7".to_string(),
            "true".to_string(),
            "null".to_string(),
        ])
    );
}

/// **`const`.** Exactly one string.
#[test]
fn schema_const() {
    let outputs = assert_outputs_parse_and_conform(&json!({"const": "fixed"}), 7, |value| {
        value == &json!("fixed")
    });
    assert_eq!(outputs, BTreeSet::from(["\"fixed\"".to_string()]));
}

/// **String with a length bound.** Every output is a quoted JSON string whose *content*
/// has between one and two characters.
#[test]
fn schema_string_length() {
    let schema = json!({"type": "string", "minLength": 1, "maxLength": 2});
    let outputs = assert_outputs_parse_and_conform(&schema, 4, |value| {
        value
            .as_str()
            .is_some_and(|text| (1..=2).contains(&text.chars().count()))
    });
    assert!(
        outputs
            .iter()
            .all(|output| output.starts_with('"') && output.ends_with('"'))
    );
    // The empty string violates minLength and must not appear.
    assert!(!outputs.contains("\"\""));
}

/// **Array of booleans, bounded.** Every output is a JSON array of length at most two
/// whose every element is a boolean — and the commas are right, which is where a
/// hand-written array grammar trips.
#[test]
fn schema_array() {
    let schema = json!({"type": "array", "items": {"type": "boolean"}, "maxItems": 2});
    let outputs = assert_outputs_parse_and_conform(&schema, 12, |value| {
        value
            .as_array()
            .is_some_and(|items| items.len() <= 2 && items.iter().all(Value::is_boolean))
    });
    assert!(outputs.contains("[]"));
    assert!(outputs.contains("[true]"));
    assert!(outputs.contains("[true,false]"));
    // No trailing comma, no leading comma, no doubled comma.
    for output in &outputs {
        assert!(!output.contains(",]"));
        assert!(!output.contains("[,"));
        assert!(!output.contains(",,"));
    }
}

/// **Object with a required and an optional property.** Every output is an object that
/// has the required key, whose values have the declared types, that carries no other
/// key, and whose keys are comma-separated with no trailing comma — the classic
/// structured-output failure the whole module exists to prevent. Boolean leaves keep
/// the enumeration rigid; see [`object_with_integer_field_membership`] for an integer
/// value.
#[test]
fn schema_object() {
    let schema = json!({
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "n": {"type": "boolean"}
        },
        "required": ["ok"]
    });
    let outputs = assert_outputs_parse_and_conform(&schema, 20, |value| {
        let Some(object) = value.as_object() else {
            return false;
        };
        let ok_valid = object.get("ok").is_some_and(Value::is_boolean);
        let n_valid = object.get("n").is_none_or(Value::is_boolean);
        let no_extras = object.keys().all(|key| key == "ok" || key == "n");
        ok_valid && n_valid && no_extras
    });

    // Both the required-only and the both-keys forms must be reachable.
    assert!(outputs.contains("{\"ok\":true}"));
    assert!(outputs.iter().any(|output| output.starts_with("{\"n\":")));
    // The bare `{}` omits the required key and must never appear.
    assert!(!outputs.contains("{}"));
    for output in &outputs {
        assert!(!output.contains(",}"), "{output} has a trailing comma");
    }
    // Keys always come in declaration (== lexicographic) order: `n` before `ok`.
    assert!(
        !outputs
            .iter()
            .any(|output| output.starts_with("{\"ok\":") && output.contains("\"n\""))
    );
}

/// **Nested object inside an array inside an object**, to prove the recursion composes
/// and the whole thing still round-trips through `serde_json`. Boolean array items keep
/// it rigid.
#[test]
fn schema_nested() {
    let schema = json!({
        "type": "object",
        "properties": {
            "flags": {"type": "array", "items": {"type": "boolean"}, "maxItems": 2}
        },
        "required": ["flags"]
    });
    let outputs = assert_outputs_parse_and_conform(&schema, 22, |value| {
        value
            .get("flags")
            .and_then(Value::as_array)
            .is_some_and(|items| items.len() <= 2 && items.iter().all(Value::is_boolean))
    });
    assert!(outputs.contains("{\"flags\":[]}"));
    assert!(outputs.contains("{\"flags\":[true]}"));
    assert!(outputs.contains("{\"flags\":[true,false]}"));
}

/// **`anyOf` is a union**: an integer or a boolean, and every output is one or the
/// other.
#[test]
fn schema_any_of() {
    let schema = json!({"anyOf": [{"type": "integer"}, {"type": "boolean"}]});
    let outputs = assert_outputs_parse_and_conform(&schema, 5, |value| {
        value.is_boolean() || value.is_i64() || value.is_u64()
    });
    assert!(outputs.contains("true"));
    assert!(outputs.contains("false"));
    assert!(outputs.contains("0"));
    assert!(outputs.contains("-1"));
}

/// An object with an integer field, checked by **membership** rather than enumeration
/// (a free integer nested in structure makes the exhaustive walk explode). Hand-written
/// canonical documents must match; malformed ones must not; and every canonical form
/// round-trips through `serde_json`.
#[test]
fn object_with_integer_field_membership() {
    let schema = json!({
        "type": "object",
        "properties": {"id": {"type": "integer"}, "ok": {"type": "boolean"}},
        "required": ["id", "ok"]
    });
    // Keys sort to `id` before `ok`.
    let ast = compile_json_schema(&schema, JsonSchemaOptions::default()).unwrap();
    let dfa = dfa_from_ast(&ast);

    for good in [
        "{\"id\":0,\"ok\":true}",
        "{\"id\":-42,\"ok\":false}",
        "{\"id\":100,\"ok\":true}",
    ] {
        assert!(dfa.matches(good), "{good} should match");
        let value: Value = serde_json::from_str(good).unwrap();
        assert!(value["id"].is_i64());
        assert!(value["ok"].is_boolean());
    }
    for bad in [
        "{\"ok\":true,\"id\":0}",  // wrong key order
        "{\"id\":0}",              // missing required `ok`
        "{\"id\":-0,\"ok\":true}", // non-canonical -0
        "{\"id\":01,\"ok\":true}", // leading zero
        "{\"id\":0,\"ok\":true,}", // trailing comma
    ] {
        assert!(!dfa.matches(bad), "{bad} should not match");
    }
}

// ── The compiler's own error surface ─────────────────────────────────────────

/// An unsupported keyword is rejected loudly, never ignored — because ignoring it would
/// widen the language past the schema.
#[test]
fn unsupported_keywords_are_rejected() {
    for schema in [
        json!({"type": "number", "multipleOf": 2}),
        json!({"oneOf": [{"type": "integer"}]}),
        json!({"allOf": [{"type": "integer"}]}),
        json!({"type": "object", "additionalProperties": true}),
    ] {
        let result = compile_json_schema(&schema, JsonSchemaOptions::default());
        assert!(
            matches!(
                result,
                Err(ConstrainedDecodingError::UnsupportedKeyword { .. })
            ),
            "schema {schema} should be rejected as unsupported",
        );
    }
}

/// A schema with no `type` (and no `enum`/`const`/`anyOf`) is refused: a masked decoder
/// must know what shape of value it may start emitting.
#[test]
fn missing_type_is_rejected() {
    let result = compile_json_schema(&json!({"minLength": 1}), JsonSchemaOptions::default());
    assert!(matches!(
        result,
        Err(ConstrainedDecodingError::InvalidSchema { .. })
    ));
}

/// An array schema without `items` is refused for the same reason.
#[test]
fn array_without_items_is_rejected() {
    let result = compile_json_schema(&json!({"type": "array"}), JsonSchemaOptions::default());
    assert!(matches!(
        result,
        Err(ConstrainedDecodingError::InvalidSchema { .. })
    ));
}

/// A `pattern` that could match characters a JSON string cannot carry literally (here a
/// literal quote) is refused rather than silently producing malformed JSON.
#[test]
fn json_unsafe_pattern_is_rejected() {
    let schema = json!({"type": "string", "pattern": "a\"b"});
    let result = compile_json_schema(&schema, JsonSchemaOptions::default());
    assert!(matches!(
        result,
        Err(ConstrainedDecodingError::InvalidSchema { .. })
    ));
}

/// A `pattern` that stays inside JSON-safe ASCII compiles, and constrains the string's
/// content.
#[test]
fn json_safe_pattern_compiles() {
    let schema = json!({"type": "string", "pattern": "[a-z]{2}"});
    let ast = compile_json_schema(&schema, JsonSchemaOptions::default()).unwrap();
    let dfa = dfa_from_ast(&ast);
    assert!(dfa.matches("\"ab\""));
    assert!(!dfa.matches("\"a\""));
    assert!(!dfa.matches("\"abc\""));
}

/// **`AnyPermutation` accepts every key ordering**, and each ordering still round-trips
/// through `serde_json`. Two properties, both required, gives exactly two orderings.
#[test]
fn object_any_permutation_orders() {
    let schema = json!({
        "type": "object",
        "properties": {"a": {"type": "boolean"}, "b": {"type": "boolean"}},
        "required": ["a", "b"]
    });
    let options = JsonSchemaOptions {
        property_order: PropertyOrder::AnyPermutation,
        ..JsonSchemaOptions::default()
    };
    let ast = compile_json_schema(&schema, options).unwrap();
    let dfa = dfa_from_ast(&ast);
    // Both orderings are in the language...
    assert!(dfa.matches("{\"a\":true,\"b\":false}"));
    assert!(dfa.matches("{\"b\":false,\"a\":true}"));
    // ...and both parse back to the same object.
    for text in ["{\"a\":true,\"b\":false}", "{\"b\":false,\"a\":true}"] {
        let value: Value = serde_json::from_str(text).unwrap();
        assert_eq!(value, json!({"a": true, "b": false}));
    }
}
