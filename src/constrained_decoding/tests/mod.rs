//! Tests for constrained decoding.
//!
//! The suite is organised around one refusal: **no test may treat the
//! implementation as its own oracle.** Every headline claim is checked against a
//! ground truth built by a *different* algorithm, or written out by hand in the
//! test's comments, or supplied by `serde_json` — code this module did not write.
//!
//! * [`automaton`] — the two-engine agreement. The Thompson NFA simulator and the
//!   subset-construction DFA are enumerated over a small alphabet up to length 8 and
//!   asserted to accept *exactly* the same strings. Neither is the other's
//!   reference; they share no code below the syntax tree. Also the hand-enumerated
//!   membership sets, and the UTF-8 automaton checked against `std::str::from_utf8`.
//! * [`parser`] — the regex dialect: precedence, quantifiers, classes, escapes,
//!   anchoring rejection, and the error surface.
//! * [`liveness`] — for every reachable DFA state and every byte, the mask's verdict
//!   is checked against an *independent* forward search for a completion, never
//!   against the co-accessibility set the mask was built from. Plus the token-level
//!   liveness that byte-level co-accessibility misses, and the no-dead-end trap.
//! * [`decoder`] — the masked decoder driven over *every* sampling choice, bounded,
//!   with every terminated output asserted to be in the language; and the mask as a
//!   pure logit transform.
//! * [`json`] — `serde_json` as the oracle: every string the masked decoder can emit
//!   under a compiled schema must parse and must satisfy the schema's constraints,
//!   enumerated exhaustively.

#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::needless_range_loop,
    clippy::bool_assert_comparison
)]

mod automaton;
mod decoder;
mod json;
mod liveness;
mod parser;

use std::collections::BTreeSet;

use crate::constrained_decoding::automaton::{Dfa, Nfa};
use crate::constrained_decoding::types::{ByteAlphabet, RegexAst};

/// Build a DFA straight from a syntax tree, the way the whole pipeline does:
/// gather its classes, form the alphabet, Thompson-compile, determinise.
pub(crate) fn dfa_from_ast(ast: &RegexAst) -> Dfa {
    let mut classes = Vec::new();
    ast.for_each_class(&mut |class| classes.push(class.clone()));
    let alphabet = ByteAlphabet::from_classes(&classes);
    let nfa = Nfa::compile(ast, 100_000).expect("nfa compiles");
    Dfa::from_nfa(&nfa, &alphabet, 100_000).expect("dfa determinises")
}

/// Build both engines from one syntax tree, sharing the alphabet.
pub(crate) fn both_engines(ast: &RegexAst) -> (Nfa, Dfa) {
    let mut classes = Vec::new();
    ast.for_each_class(&mut |class| classes.push(class.clone()));
    let alphabet = ByteAlphabet::from_classes(&classes);
    let nfa = Nfa::compile(ast, 100_000).expect("nfa compiles");
    let dfa = Dfa::from_nfa(&nfa, &alphabet, 100_000).expect("dfa determinises");
    (nfa, dfa)
}

/// Every byte string over `alphabet` of length `0..=max_len`, shortest first.
///
/// This is the exhaustive-enumeration primitive the two-engine agreement and the
/// hand-membership tests both stand on. It has nothing to do with the module under
/// test — it is a plain odometer over a small alphabet.
pub(crate) fn all_strings(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = vec![Vec::new()];
    let mut frontier: Vec<Vec<u8>> = vec![Vec::new()];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for prefix in &frontier {
            for &symbol in alphabet {
                let mut extended = prefix.clone();
                extended.push(symbol);
                next.push(extended);
            }
        }
        out.extend(next.iter().cloned());
        frontier = next;
    }
    out
}

/// The set of strings (as `String`, ASCII assumed) in `alphabet^{0..=max_len}` that
/// `dfa` accepts.
pub(crate) fn accepted_set(dfa: &Dfa, alphabet: &[u8], max_len: usize) -> BTreeSet<String> {
    all_strings(alphabet, max_len)
        .into_iter()
        .filter(|candidate| dfa.accepts(candidate))
        .map(|bytes| String::from_utf8(bytes).expect("ascii test alphabet"))
        .collect()
}
