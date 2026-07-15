//! The regex dialect: structure, precedence, classes, escapes, and the errors.
//!
//! The parser's *output* is checked mostly through the automaton (membership is the
//! real specification), but a handful of AST-shape and error-surface facts are worth
//! pinning directly.

use crate::constrained_decoding::regex_parser::{
    MAX_REGEX_DEPTH, digit_class, parse_regex, space_class, word_class,
};
use crate::constrained_decoding::tests::dfa_from_ast;
use crate::constrained_decoding::types::{CharClass, ConstrainedDecodingError, RegexAst};

/// Alternation binds looser than concatenation: `ab|cd` is `(ab)|(cd)`, not
/// `a(b|c)d`. Checked by membership, since that is what the reader actually cares
/// about.
#[test]
fn alternation_is_looser_than_concatenation() {
    let dfa = dfa_from_ast(&parse_regex("ab|cd").unwrap());
    assert!(dfa.matches("ab"));
    assert!(dfa.matches("cd"));
    assert!(!dfa.matches("abd"));
    assert!(!dfa.matches("acd"));
    assert!(!dfa.matches("ad"));
}

/// A quantifier binds only the atom immediately to its left: `ab*` is `a(b*)`, so it
/// matches `a`, `ab`, `abb`, but never `ab` repeated.
#[test]
fn quantifier_binds_the_nearest_atom() {
    let dfa = dfa_from_ast(&parse_regex("ab*").unwrap());
    assert!(dfa.matches("a"));
    assert!(dfa.matches("ab"));
    assert!(dfa.matches("abbb"));
    assert!(!dfa.matches("abab"));
    assert!(!dfa.matches("aab"));
}

/// `{m,n}` in all four spellings.
#[test]
fn repetition_bounds_in_every_spelling() {
    let exact = dfa_from_ast(&parse_regex("a{3}").unwrap());
    assert!(!exact.matches("aa"));
    assert!(exact.matches("aaa"));
    assert!(!exact.matches("aaaa"));

    let between = dfa_from_ast(&parse_regex("a{2,4}").unwrap());
    assert!(!between.matches("a"));
    assert!(between.matches("aa"));
    assert!(between.matches("aaaa"));
    assert!(!between.matches("aaaaa"));

    let at_least = dfa_from_ast(&parse_regex("a{2,}").unwrap());
    assert!(!at_least.matches("a"));
    assert!(at_least.matches("aa"));
    assert!(at_least.matches("aaaaaaaa"));

    let zero = dfa_from_ast(&parse_regex("a{0,2}").unwrap());
    assert!(zero.matches(""));
    assert!(zero.matches("aa"));
    assert!(!zero.matches("aaa"));
}

/// The predefined classes are the ASCII sets they claim to be, not something wider.
#[test]
fn predefined_classes_have_the_right_bytes() {
    let digits = digit_class();
    for byte in b'0'..=b'9' {
        assert!(digits.contains(byte));
    }
    assert!(!digits.contains(b'a'));
    assert_eq!(digits.cardinality(), 10);

    let word = word_class();
    assert!(word.contains(b'_'));
    assert!(word.contains(b'A'));
    assert!(word.contains(b'z'));
    assert!(word.contains(b'5'));
    assert!(!word.contains(b'-'));
    assert_eq!(word.cardinality(), 26 + 26 + 10 + 1);

    let space = space_class();
    assert!(space.contains(b' '));
    assert!(space.contains(b'\t'));
    assert!(space.contains(b'\n'));
    assert!(!space.contains(b'x'));
}

/// `\d` inside a negated class is a set, and negating a set that already spans a range
/// must land on the complement — a spot where off-by-one range arithmetic hides.
#[test]
fn class_negation_and_ranges() {
    let not_digit = dfa_from_ast(&parse_regex("[^0-9]").unwrap());
    assert!(!not_digit.accepts(b"5"));
    assert!(not_digit.accepts(b"a"));
    assert!(not_digit.accepts(&[0x00]));
    assert!(not_digit.accepts(b"/")); // 0x2f, just below '0'
    assert!(not_digit.accepts(b":")); // 0x3a, just above '9'
}

/// The literal-`]` -as-first-member POSIX rule, and `-` as a literal at the class
/// edge.
#[test]
fn class_edge_cases() {
    let bracket = dfa_from_ast(&parse_regex(r"[]a]").unwrap());
    assert!(bracket.accepts(b"]"));
    assert!(bracket.accepts(b"a"));
    assert!(!bracket.accepts(b"b"));

    let dash = dfa_from_ast(&parse_regex("[a-]").unwrap());
    assert!(dash.accepts(b"a"));
    assert!(dash.accepts(b"-"));
    assert!(!dash.accepts(b"b"));
}

/// `\xNN` names a byte directly, including bytes that are not printable and not valid
/// UTF-8 on their own.
#[test]
fn hex_escape_names_a_byte() {
    let ast = parse_regex(r"\xff\x00").unwrap();
    let dfa = dfa_from_ast(&ast);
    assert!(dfa.accepts(&[0xff, 0x00]));
    assert!(!dfa.accepts(&[0x00, 0xff]));
}

/// `(?:...)` is grouping without capture, and means exactly `(...)`.
#[test]
fn non_capturing_group_is_plain_grouping() {
    let plain = dfa_from_ast(&parse_regex("(ab)+").unwrap());
    let non_capturing = dfa_from_ast(&parse_regex("(?:ab)+").unwrap());
    for candidate in ["ab", "abab", "ababab"] {
        assert!(plain.matches(candidate));
        assert!(non_capturing.matches(candidate));
    }
    assert!(!non_capturing.matches("aba"));
}

/// `.` is any byte but newline — a byte, not a character.
#[test]
fn dot_is_any_byte_but_newline() {
    let dfa = dfa_from_ast(&parse_regex("a.c").unwrap());
    assert!(dfa.accepts(b"abc"));
    assert!(dfa.accepts(&[b'a', 0xff, b'c']));
    assert!(!dfa.accepts(b"a\nc"));
    // A two-byte codepoint in the middle needs two dots, not one.
    assert!(!dfa.accepts(&[b'a', 0xc3, 0xa9, b'c']));
    let two = dfa_from_ast(&parse_regex("a..c").unwrap());
    assert!(two.accepts(&[b'a', 0xc3, 0xa9, b'c']));
}

// ── The error surface ────────────────────────────────────────────────────────

fn parse_err(pattern: &str) -> ConstrainedDecodingError {
    parse_regex(pattern).expect_err(&format!("{pattern:?} should not parse"))
}

/// Anchors are rejected, not silently ignored, because ignoring one would change the
/// language of every pattern it appears in.
#[test]
fn anchors_are_rejected() {
    assert!(matches!(
        parse_err("^a"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
    assert!(matches!(
        parse_err("a$"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
    // The escaped forms are the literal characters and parse fine.
    assert!(dfa_from_ast(&parse_regex(r"\^a\$").unwrap()).accepts(b"^a$"));
}

/// A quantifier with no preceding atom, and a stacked quantifier, are both errors.
#[test]
fn dangling_and_stacked_quantifiers_are_rejected() {
    assert!(matches!(
        parse_err("*a"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
    assert!(matches!(
        parse_err("a**"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
    assert!(matches!(
        parse_err("+"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
}

/// Unbalanced groups and classes are errors with a position.
#[test]
fn unbalanced_delimiters_are_rejected() {
    assert!(matches!(
        parse_err("(ab"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
    assert!(matches!(
        parse_err("ab)"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
    assert!(matches!(
        parse_err("[abc"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
}

/// A reversed range and a reversed `{m,n}` are rejected.
#[test]
fn reversed_bounds_are_rejected() {
    assert!(matches!(
        parse_err("[z-a]"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
    assert!(matches!(
        parse_regex("a{5,2}"),
        Err(ConstrainedDecodingError::InvalidRepeat { .. })
    ));
}

/// A repetition count past the limit is refused rather than silently truncated or
/// allowed to blow up the automaton.
#[test]
fn oversized_repetition_is_rejected() {
    assert!(matches!(
        parse_regex("a{100000}"),
        Err(ConstrainedDecodingError::InvalidRepeat { .. })
    ));
}

/// Grouping deeper than the limit is an error, not a stack overflow.
#[test]
fn excessive_nesting_is_rejected() {
    let deep = format!(
        "{}a{}",
        "(".repeat(MAX_REGEX_DEPTH as usize + 1),
        ")".repeat(MAX_REGEX_DEPTH as usize + 1)
    );
    assert!(matches!(
        parse_regex(&deep),
        Err(ConstrainedDecodingError::NestingTooDeep { .. })
    ));
}

/// An unknown escape is an error; every ASCII-punctuation escape is the literal
/// character.
#[test]
fn escapes_are_validated() {
    assert!(matches!(
        parse_err(r"\q"),
        ConstrainedDecodingError::RegexSyntax { .. }
    ));
    let dfa = dfa_from_ast(&parse_regex(r"\.\*\+\(\)").unwrap());
    assert!(dfa.accepts(b".*+()"));
}

/// The empty pattern is the language `{""}`: it matches the empty string and nothing
/// else. This is a real regex, not an error.
#[test]
fn empty_pattern_matches_only_empty() {
    let ast = parse_regex("").unwrap();
    assert_eq!(ast, RegexAst::Empty);
    let dfa = dfa_from_ast(&ast);
    assert!(dfa.accepts(b""));
    assert!(!dfa.accepts(b"a"));
}

/// An empty alternation branch contributes the empty string: `(a|)` matches `a` and
/// `""`.
#[test]
fn empty_alternation_branch_is_the_empty_string() {
    let dfa = dfa_from_ast(&parse_regex("(a|)b").unwrap());
    assert!(dfa.matches("ab"));
    assert!(dfa.matches("b"));
    assert!(!dfa.matches("aab"));
}

/// An empty character class matches nothing at all — not even the empty string — and
/// its intersection with anything is empty. Built directly, since the parser has no
/// surface syntax for it.
#[test]
fn empty_class_matches_nothing() {
    let dfa = dfa_from_ast(&RegexAst::Class(CharClass::empty()));
    assert!(!dfa.accepts(b""));
    assert!(!dfa.accepts(b"a"));
    assert!(dfa.is_empty_language());
}
