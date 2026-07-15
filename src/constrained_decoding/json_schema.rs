//! Compiling a JSON Schema into a [`RegexAst`] — the same syntax tree the regex
//! parser produces, and therefore the same automaton machinery downstream.
//!
//! # What the compiled language is
//!
//! **Canonical JSON**: no insignificant whitespace anywhere. This is not a
//! shortcut, it is the point. The decoder is *masked*, so it cannot emit a space
//! unless the language contains one; a language that tolerates optional whitespace
//! merely hands the model a way to waste tokens, and an unbounded run of them is a
//! way to never terminate. Every string this compiler's language contains is
//! schema-valid JSON, which is the property that matters.
//!
//! Strings are emitted as literal UTF-8, using the two-character escapes `\"`,
//! `\\`, `\b`, `\f`, `\n`, `\r` and `\t` where JSON requires them. `\uXXXX` escapes
//! are never generated: they are never *necessary* — any scalar value can be written
//! out directly — and admitting them would break the code-point arithmetic that
//! `minLength` and `maxLength` depend on, because a surrogate pair is two escapes
//! for one character. The control characters that JSON can *only* spell with
//! `\uXXXX` (everything under `0x20` except tab, newline, form feed, carriage return
//! and backspace) are consequently not in the language.
//!
//! # Unsupported keywords are rejected, never ignored
//!
//! Quietly dropping a keyword *widens* the compiled language, and a decoder masked
//! against a language wider than its schema can emit documents the schema rejects —
//! which is the one thing this module exists to make impossible. So anything not on
//! the supported list below is a
//! [`ConstrainedDecodingError::UnsupportedKeyword`], loudly.
//!
//! | Supported | Notes |
//! |---|---|
//! | `type` | including an array of types, compiled as an alternation |
//! | `enum`, `const` | serialised by `serde_json` and emitted as literals |
//! | `properties`, `required` | see [`PropertyOrder`] |
//! | `additionalProperties` | only `false`; absent means `false` |
//! | `items`, `minItems`, `maxItems` | `items` is mandatory for an array |
//! | `minLength`, `maxLength` | counted in *characters*, not bytes |
//! | `pattern` | this module's regex dialect; must stay inside JSON-safe ASCII |
//! | `anyOf` | an alternation |
//!
//! `oneOf` is rejected rather than aliased to `anyOf`: `oneOf` demands that
//! *exactly* one branch match, and a union of overlapping branches would accept
//! documents the schema does not. `allOf` is rejected here too, but is available a
//! level up as [`crate::constrained_decoding::Constraint::all_of`], where the
//! intersection can be taken on the automata instead of the syntax trees.
//!
//! # The UTF-8 automaton
//!
//! [`utf8_scalar_ast`] is the well-formedness automaton of Table 3-7 of the Unicode
//! standard, written out as byte ranges. It is not "any byte with the high bit
//! set": it excludes the overlong encodings (`C0`, `C1`, and the `E0 80`/`F0 80`
//! prefixes), the surrogate halves (`ED A0`..`ED BF`), and everything above
//! `U+10FFFF` (`F4 90` and up, and `F5`..`FF` entirely). The tests check it against
//! `std::str::from_utf8`, byte sequence by byte sequence.
//!
//! This is the reason [`crate::constrained_decoding::ConstrainedVocabulary`] is
//! typed on bytes. A byte-pair token can end in the middle of a codepoint, so the
//! mask has to be able to say "you may emit `0xC3` now, and afterwards you will owe
//! me a continuation byte" — a statement no `&str`-shaped API can make.

use serde_json::{Map, Value};

use crate::constrained_decoding::regex_parser::parse_regex;
use crate::constrained_decoding::types::{
    ByteRange, CharClass, ConstrainedDecodingError, ConstrainedDecodingResult, JsonSchemaOptions,
    MAX_PERMUTED_PROPERTIES, PropertyOrder, RegexAst, RegexRepeat,
};

/// Keywords that carry no constraint and are simply ignored.
const ANNOTATION_KEYWORDS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$schema",
    "$id",
    "$comment",
];

/// Keywords this compiler implements.
const SUPPORTED_KEYWORDS: &[&str] = &[
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "anyOf",
];

/// Compile `schema` into a syntax tree accepting exactly the canonical JSON
/// documents it describes.
///
/// # Errors
///
/// Returns [`ConstrainedDecodingError::UnsupportedKeyword`] for any keyword outside
/// the supported set, [`ConstrainedDecodingError::InvalidSchema`] for a structurally
/// broken schema, [`ConstrainedDecodingError::NestingTooDeep`] beyond
/// `options.max_depth`, and whatever [`parse_regex`] returns for a malformed
/// `pattern`.
pub fn compile_json_schema(
    schema: &Value,
    options: JsonSchemaOptions,
) -> ConstrainedDecodingResult<RegexAst> {
    lower(schema, options, 0, "#")
}

// ── UTF-8 ────────────────────────────────────────────────────────────────────

/// The automaton of one well-formed UTF-8 scalar value, restricted in its ASCII
/// range to `allowed_ascii`.
///
/// `allowed_ascii` is intersected with `0x00..=0x7f`; the multi-byte branches are
/// unaffected by it, since they cannot encode an ASCII codepoint anyway.
///
/// # Errors
///
/// Returns [`ConstrainedDecodingError::RegexSyntax`] only if a byte range in the
/// table is malformed, which cannot happen — the ranges are literals.
pub fn utf8_scalar_ast(allowed_ascii: &CharClass) -> ConstrainedDecodingResult<RegexAst> {
    let ascii = allowed_ascii.intersect(&CharClass::range(0x00, 0x7f)?);
    let tail = CharClass::range(0x80, 0xbf)?;

    let mut branches = Vec::with_capacity(8);
    if !ascii.is_empty() {
        branches.push(RegexAst::Class(ascii));
    }
    // U+0080..U+07FF. C0 and C1 are absent: they could only encode an ASCII
    // codepoint, and an overlong encoding is ill-formed.
    branches.push(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::range(0xc2, 0xdf)?),
        RegexAst::Class(tail.clone()),
    ]));
    // U+0800..U+0FFF. `E0 80`..`E0 9F` would be overlong.
    branches.push(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::single(0xe0)),
        RegexAst::Class(CharClass::range(0xa0, 0xbf)?),
        RegexAst::Class(tail.clone()),
    ]));
    // U+1000..U+CFFF.
    branches.push(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::range(0xe1, 0xec)?),
        RegexAst::Class(tail.clone()),
        RegexAst::Class(tail.clone()),
    ]));
    // U+D000..U+D7FF. `ED A0`..`ED BF` are the surrogate halves, which are not
    // scalar values and are ill-formed in UTF-8.
    branches.push(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::single(0xed)),
        RegexAst::Class(CharClass::range(0x80, 0x9f)?),
        RegexAst::Class(tail.clone()),
    ]));
    // U+E000..U+FFFF.
    branches.push(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::range(0xee, 0xef)?),
        RegexAst::Class(tail.clone()),
        RegexAst::Class(tail.clone()),
    ]));
    // U+10000..U+3FFFF. `F0 80`..`F0 8F` would be overlong.
    branches.push(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::single(0xf0)),
        RegexAst::Class(CharClass::range(0x90, 0xbf)?),
        RegexAst::Class(tail.clone()),
        RegexAst::Class(tail.clone()),
    ]));
    // U+40000..U+FFFFF.
    branches.push(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::range(0xf1, 0xf3)?),
        RegexAst::Class(tail.clone()),
        RegexAst::Class(tail.clone()),
        RegexAst::Class(tail.clone()),
    ]));
    // U+100000..U+10FFFF. `F4 90` and beyond is past the last codepoint; `F5`..`FF`
    // never begin a well-formed sequence at all.
    branches.push(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::single(0xf4)),
        RegexAst::Class(CharClass::range(0x80, 0x8f)?),
        RegexAst::Class(tail.clone()),
        RegexAst::Class(tail),
    ]));

    Ok(RegexAst::Alternate(branches))
}

/// The bytes a JSON string may carry without escaping: printable ASCII other than
/// `"` and `\`.
fn json_safe_ascii() -> ConstrainedDecodingResult<CharClass> {
    Ok(CharClass::range(0x20, 0x7f)?
        .difference(&CharClass::single(b'"'))
        .difference(&CharClass::single(b'\\')))
}

/// One character of a JSON string: an unescaped scalar, or one of the seven
/// two-character escapes.
///
/// # Errors
///
/// As [`utf8_scalar_ast`].
pub fn json_string_char_ast() -> ConstrainedDecodingResult<RegexAst> {
    let unescaped = utf8_scalar_ast(&json_safe_ascii()?)?;
    let escape = RegexAst::Concat(vec![
        RegexAst::literal_byte(b'\\'),
        RegexAst::Class(CharClass::from_ranges(vec![
            ByteRange {
                start: b'"',
                end: b'"',
            },
            ByteRange {
                start: b'\\',
                end: b'\\',
            },
            ByteRange {
                start: b'b',
                end: b'b',
            },
            ByteRange {
                start: b'f',
                end: b'f',
            },
            ByteRange {
                start: b'n',
                end: b'n',
            },
            ByteRange {
                start: b'r',
                end: b'r',
            },
            ByteRange {
                start: b't',
                end: b't',
            },
        ])),
    ]);
    Ok(RegexAst::Alternate(vec![unescaped, escape]))
}

// ── Scalars ──────────────────────────────────────────────────────────────────

/// `(0|[1-9][0-9]*)`
fn int_magnitude() -> ConstrainedDecodingResult<RegexAst> {
    Ok(RegexAst::Alternate(vec![
        RegexAst::literal_byte(b'0'),
        RegexAst::Concat(vec![
            RegexAst::Class(CharClass::range(b'1', b'9')?),
            RegexAst::repeat(
                RegexAst::Class(CharClass::range(b'0', b'9')?),
                RegexRepeat::star(),
            ),
        ]),
    ]))
}

/// `0 | -?[1-9][0-9]*` — a JSON integer whose sign attaches only to a *nonzero*
/// magnitude.
///
/// This is deliberately narrower than the `-?(0|[1-9][0-9]*)` that the RFC 8259
/// number grammar would allow, because that wider form admits `"-0"` — and
/// `serde_json` reads `"-0"` back as the floating-point value `-0.0`, not an integer,
/// so a decoder that could emit it would be producing a "valid integer" the JSON
/// parser disagrees is one. (This exact bug was caught by the `serde_json`-oracle
/// enumeration in the tests.)
fn json_integer_ast() -> ConstrainedDecodingResult<RegexAst> {
    Ok(RegexAst::Alternate(vec![
        RegexAst::literal_byte(b'0'),
        RegexAst::Concat(vec![
            RegexAst::optional(RegexAst::literal_byte(b'-')),
            RegexAst::Class(CharClass::range(b'1', b'9')?),
            RegexAst::repeat(
                RegexAst::Class(CharClass::range(b'0', b'9')?),
                RegexRepeat::star(),
            ),
        ]),
    ]))
}

/// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?` — the JSON grammar's number.
fn json_number_ast() -> ConstrainedDecodingResult<RegexAst> {
    let digits = RegexAst::repeat(
        RegexAst::Class(CharClass::range(b'0', b'9')?),
        RegexRepeat::plus(),
    );
    let fraction = RegexAst::optional(RegexAst::Concat(vec![
        RegexAst::literal_byte(b'.'),
        digits.clone(),
    ]));
    let exponent = RegexAst::optional(RegexAst::Concat(vec![
        RegexAst::Class(CharClass::from_ranges(vec![
            ByteRange {
                start: b'E',
                end: b'E',
            },
            ByteRange {
                start: b'e',
                end: b'e',
            },
        ])),
        RegexAst::optional(RegexAst::Class(CharClass::from_ranges(vec![
            ByteRange {
                start: b'+',
                end: b'+',
            },
            ByteRange {
                start: b'-',
                end: b'-',
            },
        ]))),
        digits,
    ]));
    Ok(RegexAst::Concat(vec![
        RegexAst::optional(RegexAst::literal_byte(b'-')),
        int_magnitude()?,
        fraction,
        exponent,
    ]))
}

fn json_boolean_ast() -> RegexAst {
    RegexAst::Alternate(vec![
        RegexAst::literal_str("true"),
        RegexAst::literal_str("false"),
    ])
}

// ── Strings ──────────────────────────────────────────────────────────────────

fn json_string_ast(node: &Map<String, Value>, path: &str) -> ConstrainedDecodingResult<RegexAst> {
    let min_length = read_u32(node, "minLength", path)?.unwrap_or(0);
    let max_length = read_u32(node, "maxLength", path)?;

    let content = if let Some(pattern) = node.get("pattern") {
        let Some(pattern) = pattern.as_str() else {
            return Err(invalid(path, "`pattern` must be a string"));
        };
        if node.contains_key("minLength") || node.contains_key("maxLength") {
            return Err(ConstrainedDecodingError::UnsupportedKeyword {
                keyword: "pattern".to_string(),
                detail: Some(
                    "`pattern` cannot be combined with `minLength` or `maxLength`: the two \
                     constraints would have to be intersected, and a syntax tree has no \
                     intersection. Fold the length bound into the pattern itself."
                        .to_string(),
                ),
            });
        }
        let ast = parse_regex(pattern)?;
        assert_pattern_is_json_safe(&ast, path)?;
        ast
    } else {
        let repeat = RegexRepeat {
            min: min_length,
            max: max_length,
        };
        repeat.validate()?;
        RegexAst::repeat(json_string_char_ast()?, repeat)
    };

    Ok(RegexAst::Concat(vec![
        RegexAst::literal_byte(b'"'),
        content,
        RegexAst::literal_byte(b'"'),
    ]))
}

/// Refuse a `pattern` whose alphabet strays outside what a JSON string can hold
/// literally.
///
/// A pattern describes the string's *content*, but the automaton emits its *encoded
/// form*. For content that needs no escaping the two coincide and the pattern can be
/// spliced straight in. For content that does — a `"` , a backslash, a control
/// character, or a raw byte that is not valid UTF-8 on its own — they do not, and
/// splicing would produce a language full of malformed JSON. Composing the pattern
/// with the escaping transducer would be the general answer; refusing is the honest
/// one.
fn assert_pattern_is_json_safe(ast: &RegexAst, path: &str) -> ConstrainedDecodingResult<()> {
    let safe = json_safe_ascii()?;
    let mut offenders = CharClass::empty();
    ast.for_each_class(&mut |class| {
        offenders = offenders.union(&class.difference(&safe));
    });
    if offenders.is_empty() {
        return Ok(());
    }
    Err(invalid(
        path,
        format!(
            "`pattern` may only match characters a JSON string can carry literally \
             (printable ASCII other than `\"` and `\\`), but it can match {:?}. Note that \
             `.` matches raw bytes here, including bytes that are not valid UTF-8; use an \
             explicit character class.",
            offenders.ranges()
        ),
    ))
}

// ── Arrays ───────────────────────────────────────────────────────────────────

fn json_array_ast(
    node: &Map<String, Value>,
    options: JsonSchemaOptions,
    depth: u32,
    path: &str,
) -> ConstrainedDecodingResult<RegexAst> {
    let Some(items) = node.get("items") else {
        return Err(invalid(
            path,
            "an array schema needs `items`: without it there is nothing to say about what \
             the elements may be, and a masked decoder has to know before it emits one",
        ));
    };
    let item = lower(items, options, depth + 1, &format!("{path}/items"))?;
    let min_items = read_u32(node, "minItems", path)?.unwrap_or(0);
    let max_items = read_u32(node, "maxItems", path)?;

    if max_items == Some(0) {
        return Ok(RegexAst::literal_str("[]"));
    }

    let tail_repeat = RegexRepeat {
        min: min_items.saturating_sub(1),
        max: max_items.map(|count| count.saturating_sub(1)),
    };
    tail_repeat.validate()?;
    let comma_item = RegexAst::Concat(vec![RegexAst::literal_byte(b','), item.clone()]);
    let non_empty = RegexAst::Concat(vec![item, RegexAst::repeat(comma_item, tail_repeat)]);

    let body = if min_items == 0 {
        RegexAst::optional(non_empty)
    } else {
        non_empty
    };

    Ok(RegexAst::Concat(vec![
        RegexAst::literal_byte(b'['),
        body,
        RegexAst::literal_byte(b']'),
    ]))
}

// ── Objects ──────────────────────────────────────────────────────────────────

/// `"key":<value>` — the key serialised by `serde_json`, so its escaping is not this
/// module's problem.
fn property_ast(key: &str, value: &RegexAst) -> ConstrainedDecodingResult<RegexAst> {
    let quoted =
        serde_json::to_string(key).map_err(|error| ConstrainedDecodingError::InvalidSchema {
            path: "#".to_string(),
            reason: format!("property name {key:?} could not be serialised: {error}"),
        })?;
    Ok(RegexAst::Concat(vec![
        RegexAst::literal_str(&quoted),
        RegexAst::literal_byte(b':'),
        value.clone(),
    ]))
}

struct ObjectProperties {
    entries: Vec<(String, RegexAst)>,
    required: Vec<bool>,
}

fn json_object_ast(
    node: &Map<String, Value>,
    options: JsonSchemaOptions,
    depth: u32,
    path: &str,
) -> ConstrainedDecodingResult<RegexAst> {
    match node.get("additionalProperties") {
        None | Some(Value::Bool(false)) => {}
        Some(_) => {
            return Err(ConstrainedDecodingError::UnsupportedKeyword {
                keyword: "additionalProperties".to_string(),
                detail: Some(
                    "only `false` is supported (and is the default here): an open-ended object \
                     has no finite set of keys for the mask to choose between"
                        .to_string(),
                ),
            });
        }
    }

    let empty = Map::new();
    let properties = match node.get("properties") {
        None => &empty,
        Some(Value::Object(map)) => map,
        Some(_) => return Err(invalid(path, "`properties` must be an object")),
    };

    let required_names: Vec<String> = match node.get("required") {
        None => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid(path, "`required` must be an array of strings"))
            })
            .collect::<ConstrainedDecodingResult<Vec<String>>>()?,
        Some(_) => return Err(invalid(path, "`required` must be an array of strings")),
    };

    let mut entries: Vec<(String, RegexAst)> = Vec::with_capacity(properties.len());
    for (key, sub_schema) in properties {
        let sub_path = format!("{path}/properties/{key}");
        let value = lower(sub_schema, options, depth + 1, &sub_path)?;
        entries.push((key.clone(), value));
    }

    let required: Vec<bool> = entries
        .iter()
        .map(|(key, _)| required_names.contains(key))
        .collect();
    for name in &required_names {
        if !entries.iter().any(|(key, _)| key == name) {
            return Err(invalid(
                path,
                format!("`required` names {name:?}, which is not among `properties`"),
            ));
        }
    }

    let props = ObjectProperties { entries, required };
    let body = match options.property_order {
        PropertyOrder::Declaration => object_body_declaration_order(&props)?,
        PropertyOrder::AnyPermutation => object_body_any_permutation(&props)?,
    };

    Ok(RegexAst::Concat(vec![
        RegexAst::literal_byte(b'{'),
        body,
        RegexAst::literal_byte(b'}'),
    ]))
}

/// Keys in `properties` order; optional keys present or absent; commas only between
/// keys that are actually there.
///
/// The trailing-comma bug lives here, and the shape of the construction is what
/// kills it. The first key emitted carries no comma; every later key carries one.
/// So the body is an alternation over *which key comes first*, and that key can be
/// any of them up to and including the first required one — going further would skip
/// a required key. Everything after the first is a comma-prefixed piece, mandatory
/// if the key is required and optional if it is not. That is `O(n^2)` nodes, where
/// the obvious recursive formulation ("include it or don't, and pass down whether
/// anything has been emitted yet") is `O(2^n)`.
fn object_body_declaration_order(props: &ObjectProperties) -> ConstrainedDecodingResult<RegexAst> {
    let count = props.entries.len();
    let first_required = props
        .required
        .iter()
        .position(|flag| *flag)
        .unwrap_or(count);

    let mut branches: Vec<RegexAst> = Vec::new();
    // `{}` — legal exactly when nothing is required.
    if first_required == count {
        branches.push(RegexAst::Empty);
    }

    for first in 0..=first_required.min(count.saturating_sub(1)) {
        if count == 0 {
            break;
        }
        let (key, value) = &props.entries[first];
        let mut pieces = vec![property_ast(key, value)?];
        for (index, (later_key, later_value)) in props.entries.iter().enumerate().skip(first + 1) {
            let piece = RegexAst::Concat(vec![
                RegexAst::literal_byte(b','),
                property_ast(later_key, later_value)?,
            ]);
            pieces.push(if props.required[index] {
                piece
            } else {
                RegexAst::optional(piece)
            });
        }
        branches.push(RegexAst::Concat(pieces));
    }

    Ok(match branches.len() {
        0 => RegexAst::Empty,
        1 => branches.swap_remove(0),
        _ => RegexAst::Alternate(branches),
    })
}

/// Every ordering of every legal key set.
fn object_body_any_permutation(props: &ObjectProperties) -> ConstrainedDecodingResult<RegexAst> {
    let count = props.entries.len();
    if count > MAX_PERMUTED_PROPERTIES {
        return Err(ConstrainedDecodingError::UnsupportedKeyword {
            keyword: "properties".to_string(),
            detail: Some(format!(
                "`PropertyOrder::AnyPermutation` enumerates every key ordering, so it is \
                 capped at {MAX_PERMUTED_PROPERTIES} properties; this object has {count}"
            )),
        });
    }

    let mut branches: Vec<RegexAst> = Vec::new();
    for mask in 0u32..(1u32 << count) {
        let chosen: Vec<usize> = (0..count)
            .filter(|index| mask & (1 << index) != 0)
            .collect();
        // Every required key must be in the chosen set.
        if (0..count).any(|index| props.required[index] && !chosen.contains(&index)) {
            continue;
        }
        if chosen.is_empty() {
            branches.push(RegexAst::Empty);
            continue;
        }
        for permutation in permutations(&chosen) {
            let mut pieces: Vec<RegexAst> = Vec::with_capacity(permutation.len() * 2);
            for (position, index) in permutation.iter().enumerate() {
                if position > 0 {
                    pieces.push(RegexAst::literal_byte(b','));
                }
                let (key, value) = &props.entries[*index];
                pieces.push(property_ast(key, value)?);
            }
            branches.push(RegexAst::Concat(pieces));
        }
    }

    Ok(match branches.len() {
        0 => RegexAst::Empty,
        1 => branches.swap_remove(0),
        _ => RegexAst::Alternate(branches),
    })
}

/// Every ordering of `items`, in a deterministic order (plain insertion recursion —
/// no shuffling, and therefore no randomness).
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    let Some((&head, tail)) = items.split_first() else {
        return vec![Vec::new()];
    };
    let mut result = Vec::new();
    for sub in permutations(tail) {
        for position in 0..=sub.len() {
            let mut candidate = sub.clone();
            candidate.insert(position, head);
            result.push(candidate);
        }
    }
    result
}

// ── The recursion ────────────────────────────────────────────────────────────

fn lower(
    schema: &Value,
    options: JsonSchemaOptions,
    depth: u32,
    path: &str,
) -> ConstrainedDecodingResult<RegexAst> {
    if depth > options.max_depth {
        return Err(ConstrainedDecodingError::NestingTooDeep {
            limit: options.max_depth,
        });
    }
    let Some(node) = schema.as_object() else {
        return Err(invalid(
            path,
            "a schema must be an object; boolean schemas (`true` / `false`) are not supported",
        ));
    };
    check_keywords(node)?;

    if let Some(values) = enumerated_values(node, path)? {
        return literal_alternation(&values, node, path);
    }

    if let Some(Value::Array(branches)) = node.get("anyOf") {
        let mut compiled = Vec::with_capacity(branches.len());
        for (index, branch) in branches.iter().enumerate() {
            compiled.push(lower(
                branch,
                options,
                depth + 1,
                &format!("{path}/anyOf/{index}"),
            )?);
        }
        if compiled.is_empty() {
            return Err(invalid(path, "`anyOf` must have at least one branch"));
        }
        return Ok(RegexAst::Alternate(compiled));
    }
    if node.contains_key("anyOf") {
        return Err(invalid(path, "`anyOf` must be an array of schemas"));
    }

    let types = declared_types(node, path)?;
    let mut branches = Vec::with_capacity(types.len());
    for type_name in &types {
        branches.push(lower_type(type_name, node, options, depth, path)?);
    }
    Ok(match branches.len() {
        1 => branches.swap_remove(0),
        _ => RegexAst::Alternate(branches),
    })
}

fn lower_type(
    type_name: &str,
    node: &Map<String, Value>,
    options: JsonSchemaOptions,
    depth: u32,
    path: &str,
) -> ConstrainedDecodingResult<RegexAst> {
    match type_name {
        "string" => json_string_ast(node, path),
        "integer" => json_integer_ast(),
        "number" => json_number_ast(),
        "boolean" => Ok(json_boolean_ast()),
        "null" => Ok(RegexAst::literal_str("null")),
        "object" => json_object_ast(node, options, depth, path),
        "array" => json_array_ast(node, options, depth, path),
        other => Err(invalid(path, format!("unknown `type` {other:?}"))),
    }
}

/// `enum`, or the single value of `const`.
fn enumerated_values(
    node: &Map<String, Value>,
    path: &str,
) -> ConstrainedDecodingResult<Option<Vec<Value>>> {
    if let Some(constant) = node.get("const") {
        return Ok(Some(vec![constant.clone()]));
    }
    match node.get("enum") {
        None => Ok(None),
        Some(Value::Array(values)) if !values.is_empty() => Ok(Some(values.clone())),
        Some(Value::Array(_)) => Err(invalid(path, "`enum` must not be empty")),
        Some(_) => Err(invalid(path, "`enum` must be an array")),
    }
}

/// Serialise each value with `serde_json` and alternate over the results.
///
/// When `type` is also present every value must satisfy it, or the compiled language
/// would contain a document the schema rejects.
fn literal_alternation(
    values: &[Value],
    node: &Map<String, Value>,
    path: &str,
) -> ConstrainedDecodingResult<RegexAst> {
    let declared = if node.contains_key("type") {
        Some(declared_types(node, path)?)
    } else {
        None
    };

    let mut branches = Vec::with_capacity(values.len());
    for value in values {
        if let Some(types) = &declared
            && !types.iter().any(|name| json_type_matches(value, name))
        {
            return Err(invalid(
                path,
                format!(
                    "the enumerated value {value} does not satisfy the declared `type` {types:?}"
                ),
            ));
        }
        let text = serde_json::to_string(value).map_err(|error| {
            ConstrainedDecodingError::InvalidSchema {
                path: path.to_string(),
                reason: format!("enumerated value could not be serialised: {error}"),
            }
        })?;
        branches.push(RegexAst::literal_str(&text));
    }
    Ok(match branches.len() {
        1 => branches.swap_remove(0),
        _ => RegexAst::Alternate(branches),
    })
}

fn json_type_matches(value: &Value, type_name: &str) -> bool {
    match type_name {
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        _ => false,
    }
}

fn declared_types(node: &Map<String, Value>, path: &str) -> ConstrainedDecodingResult<Vec<String>> {
    match node.get("type") {
        Some(Value::String(name)) => Ok(vec![name.clone()]),
        Some(Value::Array(names)) if !names.is_empty() => names
            .iter()
            .map(|name| {
                name.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid(path, "`type` array entries must be strings"))
            })
            .collect(),
        Some(_) => Err(invalid(
            path,
            "`type` must be a string or a non-empty array of strings",
        )),
        None => Err(invalid(
            path,
            "`type` is required (unless `enum`, `const` or `anyOf` is present): a masked \
             decoder must know what shape of value it is allowed to start emitting",
        )),
    }
}

fn check_keywords(node: &Map<String, Value>) -> ConstrainedDecodingResult<()> {
    for keyword in node.keys() {
        let key = keyword.as_str();
        if SUPPORTED_KEYWORDS.contains(&key) || ANNOTATION_KEYWORDS.contains(&key) {
            continue;
        }
        let detail = match key {
            "oneOf" => Some(
                "`oneOf` requires that exactly one branch match. Compiling it as a union would \
                 accept documents matching two branches, which the schema rejects; use `anyOf` \
                 if the branches are genuinely disjoint."
                    .to_string(),
            ),
            "allOf" => Some(
                "an intersection cannot be taken on syntax trees. Use \
                 `Constraint::all_of`, which intersects the compiled automata instead."
                    .to_string(),
            ),
            _ => None,
        };
        return Err(ConstrainedDecodingError::UnsupportedKeyword {
            keyword: key.to_string(),
            detail,
        });
    }
    Ok(())
}

fn read_u32(
    node: &Map<String, Value>,
    keyword: &str,
    path: &str,
) -> ConstrainedDecodingResult<Option<u32>> {
    match node.get(keyword) {
        None => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|number| u32::try_from(number).ok())
            .map(Some)
            .ok_or_else(|| {
                invalid(
                    path,
                    format!("`{keyword}` must be a non-negative integer that fits in 32 bits"),
                )
            }),
    }
}

fn invalid(path: &str, reason: impl Into<String>) -> ConstrainedDecodingError {
    ConstrainedDecodingError::InvalidSchema {
        path: path.to_string(),
        reason: reason.into(),
    }
}
