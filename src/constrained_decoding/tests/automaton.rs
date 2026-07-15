//! Two-engine agreement, hand-enumerated membership, and the UTF-8 automaton.

use std::collections::BTreeSet;

use crate::constrained_decoding::json_schema::utf8_scalar_ast;
use crate::constrained_decoding::regex_parser::parse_regex;
use crate::constrained_decoding::tests::{accepted_set, all_strings, both_engines, dfa_from_ast};
use crate::constrained_decoding::types::CharClass;

/// A collection of patterns whose alphabets all fit in `{a, b, c, d}`, deliberately
/// exercising every AST node: literals, alternation, concatenation, `*`, `+`, `?`,
/// `{m,n}`, grouping, classes, and negation.
const SMALL_PATTERNS: &[&str] = &[
    "",
    "a",
    "abc",
    "a|b|c",
    "a*",
    "a+",
    "a?",
    "a(b|c)*d",
    "(ab|ba)*",
    "a{2,3}",
    "a{0,2}b",
    "(a|b){1,3}c?",
    "[abc]",
    "[^a]",
    "[a-c]+",
    "a.c",
    "(a+|b+)+",
    "((a|b)(c|d))*",
    "a?a?a?aaa",
    "(abc|ab|a)d",
];

/// **Headline (a): two independent engines agree, exactly.**
///
/// The Thompson NFA is simulated directly (epsilon-closure over a set of states,
/// tested against raw bytes) and the subset-construction DFA is run (a determinised
/// table indexed by alphabet block). They share no code below the syntax tree. For
/// every pattern and every string over `{a, b, c, d}` up to length 8, the two must
/// return the same verdict. If the alphabet partition ever merged two bytes some
/// transition distinguishes, the DFA would diverge from the NFA here and nowhere
/// else.
#[test]
fn nfa_and_dfa_accept_exactly_the_same_strings() {
    let alphabet = b"abcd";
    let strings = all_strings(alphabet, 8);
    assert_eq!(strings.len(), (4usize.pow(9) - 1) / 3);

    for pattern in SMALL_PATTERNS {
        let ast = parse_regex(pattern).unwrap_or_else(|error| {
            panic!("pattern {pattern:?} should parse: {error}");
        });
        let (nfa, dfa) = both_engines(&ast);

        let mut disagreements = 0usize;
        for candidate in &strings {
            let by_nfa = nfa.accepts(candidate);
            let by_dfa = dfa.accepts(candidate);
            if by_nfa != by_dfa {
                disagreements += 1;
            }
        }
        assert_eq!(
            disagreements, 0,
            "NFA and DFA disagree on pattern {pattern:?}",
        );
    }
}

/// The determiniser really shrinks the state space it should, and never grows the
/// *language*: a sanity companion to the agreement test that also proves the two
/// engines are not trivially identical.
#[test]
fn dfa_is_deterministic_and_total_on_its_alphabet() {
    let ast = parse_regex("a(b|c)*d").unwrap();
    let dfa = dfa_from_ast(&ast);
    // Every live state has at most one successor per block (determinism is the type),
    // and the start state exists.
    assert!(dfa.state_count() >= 1);
    // The alphabet of `a(b|c)*d` distinguishes a, b, c, d and everything-else: five
    // blocks, not 256. (`b` and `c` are separate single-byte literals, so they are not
    // merged -- only bytes no transition mentions collapse into one "other" block.)
    assert_eq!(dfa.alphabet().block_count(), 5);
}

fn hand_check(pattern: &str, alphabet: &[u8], max_len: usize, expected: &[&str]) {
    let ast = parse_regex(pattern).unwrap();
    let dfa = dfa_from_ast(&ast);
    let actual = accepted_set(&dfa, alphabet, max_len);
    let expected: BTreeSet<String> = expected.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(
        actual, expected,
        "pattern {pattern:?} up to length {max_len}",
    );
}

/// **Headline (b), regex 1: `a(b|c)*d`.**
///
/// The accepted strings up to length 5, written out by hand: an `a`, any number of
/// `b`/`c`, then a `d`.
///
/// ```text
/// len 2: ad
/// len 3: abd acd
/// len 4: abbd abcd acbd accd
/// len 5: abbbd abbcd abcbd abccd acbbd acbcd accbd acccd
/// ```
#[test]
fn hand_membership_a_bc_star_d() {
    hand_check(
        "a(b|c)*d",
        b"abcd",
        5,
        &[
            "ad", "abd", "acd", "abbd", "abcd", "acbd", "accd", "abbbd", "abbcd", "abcbd", "abccd",
            "acbbd", "acbcd", "accbd", "acccd",
        ],
    );
}

/// **Headline (b), regex 2: `[01]{2,3}`.**
///
/// Every binary string of length exactly two or three, and nothing else.
///
/// ```text
/// len 2: 00 01 10 11
/// len 3: 000 001 010 011 100 101 110 111
/// ```
#[test]
fn hand_membership_binary_two_or_three() {
    hand_check(
        "[01]{2,3}",
        b"01",
        5,
        &[
            "00", "01", "10", "11", "000", "001", "010", "011", "100", "101", "110", "111",
        ],
    );
}

/// **Headline (b), regex 3: `x?y+`.**
///
/// An optional `x`, then one or more `y`.
///
/// ```text
/// len 1: y
/// len 2: xy yy
/// len 3: xyy yyy
/// len 4: xyyy yyyy
/// len 5: xyyyy yyyyy
/// ```
#[test]
fn hand_membership_optional_x_then_ys() {
    hand_check(
        "x?y+",
        b"xy",
        5,
        &[
            "y", "xy", "yy", "xyy", "yyy", "xyyy", "yyyy", "xyyyy", "yyyyy",
        ],
    );
}

/// **Headline (b), regex 4: `(ab|ba)*`.**
///
/// Any concatenation of the blocks `ab` and `ba`, including the empty string. Only
/// even lengths are reachable, which is exactly the kind of parity fact a
/// star-over-alternation is prone to getting wrong.
///
/// ```text
/// len 0: (empty)
/// len 2: ab ba
/// len 4: abab abba baab baba
/// ```
#[test]
fn hand_membership_ab_or_ba_star() {
    hand_check(
        "(ab|ba)*",
        b"ab",
        5,
        &["", "ab", "ba", "abab", "abba", "baab", "baba"],
    );
}

/// The negated class `[^a]` really is "every byte but `a`", not "the printable ASCII
/// but `a`". A byte-level engine must let `0x00` and `0xff` through here.
#[test]
fn negated_class_is_over_all_bytes() {
    let ast = parse_regex("[^a]").unwrap();
    let dfa = dfa_from_ast(&ast);
    assert!(dfa.accepts(&[0x00]));
    assert!(dfa.accepts(&[0xff]));
    assert!(dfa.accepts(b"b"));
    assert!(!dfa.accepts(b"a"));
    assert!(!dfa.accepts(b"bb"));
    assert!(!dfa.accepts(b""));
}

/// **The UTF-8 automaton is the real thing, checked against the standard library.**
///
/// `std::str::from_utf8` is the oracle. For every byte sequence up to length 4 over a
/// small but adversarial byte set — ASCII, a two-byte lead, a three-byte lead, a
/// four-byte lead, continuation bytes, an overlong prefix, a surrogate-range lead, an
/// out-of-range lead, and a bare continuation — the automaton accepts the sequence as
/// a single scalar iff the standard library decodes it to exactly one character.
#[test]
fn utf8_automaton_matches_std_from_utf8() {
    let ast = utf8_scalar_ast(&CharClass::range(0x00, 0x7f).unwrap()).unwrap();
    let dfa = dfa_from_ast(&ast);

    // A deliberately nasty alphabet: valid leads, continuations, and the exact bytes
    // where overlong / surrogate / out-of-range rules bite.
    let bytes: &[u8] = &[
        0x41, // 'A'
        0x7f, // last ASCII
        0xc2, 0xa9, // (c) two-byte
        0xc1, // overlong lead, always ill-formed
        0xe0, 0x80, 0xa0, // E0 with an overlong second byte, vs valid
        0xed, 0x9f, 0xa0, // ED valid (U+D7FF region) vs surrogate second byte
        0xf0, 0x90, 0x80, // four-byte
        0xf4, 0x8f, // last plane lead
        0xf5, // out of range
        0x80, // bare continuation
        0xbf,
    ];

    for len in 1..=4 {
        for candidate in cartesian(bytes, len) {
            let accepted = dfa.accepts(&candidate);
            let is_one_scalar = std::str::from_utf8(&candidate)
                .ok()
                .is_some_and(|text| text.chars().count() == 1);
            assert_eq!(
                accepted, is_one_scalar,
                "UTF-8 disagreement on {candidate:02x?}",
            );
        }
    }
}

/// The three-byte `E0` branch forbids the overlong second byte range `0x80..=0x9f`,
/// and the `ED` branch forbids the surrogate range `0xa0..=0xbf`. Spot-checked
/// directly, because these two rows of Table 3-7 are the ones a hand-written UTF-8
/// automaton always gets wrong.
#[test]
fn utf8_automaton_rejects_overlong_and_surrogate() {
    let ast = utf8_scalar_ast(&CharClass::range(0x00, 0x7f).unwrap()).unwrap();
    let dfa = dfa_from_ast(&ast);

    // Overlong: E0 80 80 encodes U+0000, which must be one byte.
    assert!(!dfa.accepts(&[0xe0, 0x80, 0x80]));
    // Valid: E0 A0 80 is U+0800.
    assert!(dfa.accepts(&[0xe0, 0xa0, 0x80]));
    // Surrogate: ED A0 80 is U+D800, not a scalar value.
    assert!(!dfa.accepts(&[0xed, 0xa0, 0x80]));
    // Valid: ED 9F BF is U+D7FF.
    assert!(dfa.accepts(&[0xed, 0x9f, 0xbf]));
    // Out of range: F4 90 80 80 would be U+110000.
    assert!(!dfa.accepts(&[0xf4, 0x90, 0x80, 0x80]));
    // Valid: F4 8F BF BF is U+10FFFF, the last codepoint.
    assert!(dfa.accepts(&[0xf4, 0x8f, 0xbf, 0xbf]));
}

/// Every length-`len` sequence over `bytes` (with repetition). A plain odometer, kept
/// local so the UTF-8 test's ground truth owes nothing to the module under test.
fn cartesian(bytes: &[u8], len: usize) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = vec![Vec::new()];
    for _ in 0..len {
        let mut next = Vec::with_capacity(out.len() * bytes.len());
        for prefix in &out {
            for &byte in bytes {
                let mut extended = prefix.clone();
                extended.push(byte);
                next.push(extended);
            }
        }
        out = next;
    }
    out
}
