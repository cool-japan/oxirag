//! Constrained decoding: compile a regex or a JSON Schema to an automaton, and mask
//! the logits so that only tokens which keep the automaton *able to finish* can be
//! sampled.
//!
//! A language model does not know what a closing brace is for. It emits a
//! distribution over tokens, and if the syntactically necessary token happens to rank
//! second, the syntactically necessary token is not what comes out. Every strategy for
//! fixing that after the fact — retry, repair, validate-and-reject — shares one flaw:
//! the invalid string had to exist first.
//!
//! This module makes it not exist. The constraint is compiled to a deterministic
//! automaton; the automaton's current state determines a set of legal tokens; every
//! other token's logit is set to `f64::NEG_INFINITY` before the sampler ever sees it.
//! Whatever the model prefers, whatever the temperature, whatever the sampler, the
//! output is in the language. Not usually. Always.
//!
//! # The pipeline
//!
//! ```text
//! regex source ──parse──┐
//!                       ├──> RegexAst ──Thompson──> Nfa ──subset──> Dfa ──┐
//! JSON Schema ──lower───┘                                                 ├──> product ──> Dfa
//!                                                                         │   (AllOf/AnyOf)
//!                                        vocabulary (bytes!) ─────────────┴──> VocabIndex ──> TokenMask
//! ```
//!
//! The regex dialect's grammar is written out as EBNF in [`regex_parser`]; the
//! JSON-Schema subset, and the exact shape of the JSON it emits, in [`json_schema`].
//!
//! # A token is legal when accepting it leaves the automaton able to finish
//!
//! That is the entire specification, and both halves of it do work.
//!
//! *Able to be consumed* is the easy half: walk the token's bytes through the
//! transition table and see whether you fall off.
//!
//! *Able to finish* is the half a plausible implementation gets wrong. The textbook
//! answer is **co-accessibility**: a state is live when some byte string leads from it
//! to an accepting state, computed by a backward search from the accepting states.
//! [`Dfa::is_live`] does exactly that, and it is necessary.
//!
//! It is also **not sufficient**, because a decoder does not emit bytes — it emits
//! *tokens*, which are indivisible multi-byte lumps. Consider the language `{"a"}` and
//! a vocabulary holding `"ab"` and `"ba"` but not `"a"`. The start state is byte-live;
//! the string `"a"` is right there in the language. And there is no legal first token:
//! `"ab"` walks off the automaton at its second byte and `"ba"` at its first. Byte-level
//! co-accessibility reports a healthy automaton, and the decoder is stuck before it
//! starts.
//!
//! So [`VocabIndex`] computes a second fixed point, over the graph whose edges are
//! *whole tokens*; a state is **token-live** when an accepting state is reachable in
//! *that* graph. Only then does the guarantee hold that
//!
//! > from every state the masked decoder can reach, at least one legal token exists.
//!
//! When even the start state fails that test, the constraint and the vocabulary are
//! each individually fine and simply do not fit together — and this module says so,
//! with [`ConstrainedDecodingError::VocabularyCannotExpressLanguage`], instead of
//! discovering it half-way through a generation.
//!
//! # When dead states actually exist
//!
//! Worth stating plainly, because the obvious claim is false in a way that would leave
//! the liveness machinery untestable. For an automaton built by Thompson construction
//! from a regex whose character classes are all non-empty, **every state is already
//! live**: each one lies on a path to its fragment's accept state by construction. So
//! pruning such a machine removes nothing, and a test of it passes vacuously.
//!
//! Dead states appear when an automaton is built by some *other* means — which here
//! means [`Dfa::product`], reached through [`Constraint::all_of`]:
//!
//! ```text
//! (ab|abcd)  ∩  "at most 3 bytes"   =   ab
//! ```
//!
//! After `ab`, the automaton still has an outgoing `c`, inherited from the `abcd`
//! branch. The state that `c` leads to needs two more bytes and has one left: it is
//! dead, and nothing but co-accessibility forbids that `c`. Two disjoint languages
//! intersect to nothing at all, and then the *start* state is dead. An empty character
//! class (`[^\x00-\xff]`) manages it with no product at all.
//!
//! # Distinct from the crate's other logit and schema machinery
//!
//! Distinct from `watermarking`, which also drives logits to `-inf` (see
//! `WatermarkMode::Hard`): that mask is a keyed hash of the preceding tokens, covers a
//! fixed `gamma * |V|` fraction of the vocabulary, and is blind to anything the decoder
//! has structurally emitted — it constrains *provenance*, not *syntax*. This module's
//! mask is a function of a compiled automaton's **current state**, so the legal-token
//! set changes with the parse, and a token is forbidden precisely when accepting it
//! would leave the automaton unable to complete any valid string in the language.
//! Distinct also from `structured_extraction` and `output_validation`, which *reject an
//! invalid string after it exists*; this module makes an invalid string **unreachable**.
//!
//! # Bytes, not `&str`
//!
//! [`ConstrainedVocabulary`] is typed on `&[u8]`, and every automaton here transitions
//! on bytes. This is not a stylistic preference. A byte-pair-encoding merge may end in
//! the middle of a UTF-8 codepoint — `é` is `0xC3 0xA9`, and a tokenizer holding those
//! as two separate tokens is entirely ordinary — so a `&str`-shaped vocabulary cannot
//! even *name* half the tokens it is supposed to be masking. The mask has to be able to
//! say "you may emit `0xC3` now, and afterwards you will owe me a continuation byte",
//! and it can.
//!
//! The JSON-string automaton is correspondingly the real UTF-8 well-formedness table
//! (Unicode Table 3-7, in [`utf8_scalar_ast`]) — overlong encodings, surrogate halves,
//! and everything past `U+10FFFF` excluded — so a decode that reaches an accepting state
//! has produced valid UTF-8, byte by byte, without ever having been able to check.
//!
//! # Example
//!
//! ```
//! use oxirag::constrained_decoding::{
//!     ConstrainedDecoder, ConstrainedDecoderConfig, Constraint, StaticVocabulary,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // A four-token vocabulary, plus an end-of-sequence token at id 4.
//! let vocab = StaticVocabulary::from_strs(&["a", "b", "c", "d", "<eos>"])?;
//! let config = ConstrainedDecoderConfig::default().with_eos(4);
//!
//! let decoder = ConstrainedDecoder::compile(&Constraint::regex("a(b|c)*d"), &vocab, &config)?;
//!
//! // At the start only `a` is legal -- not even stopping.
//! let start = decoder.start_state();
//! assert_eq!(decoder.allowed_tokens(start)?.allowed(), vec![0]);
//!
//! // The mask is the product: everything illegal goes to negative infinity, and
//! // everything legal is left exactly as the model scored it.
//! let mut logits = vec![0.5, 9.9, 9.9, 9.9, 9.9];
//! decoder.mask_logits(&mut logits, start)?;
//! assert_eq!(
//!     logits,
//!     vec![
//!         0.5,
//!         f64::NEG_INFINITY,
//!         f64::NEG_INFINITY,
//!         f64::NEG_INFINITY,
//!         f64::NEG_INFINITY
//!     ]
//! );
//!
//! // After `a`, `b`/`c`/`d` are legal and stopping is not: `a` is not in the language.
//! let after_a = decoder.step(start, 0)?;
//! assert_eq!(decoder.allowed_tokens(after_a)?.allowed(), vec![1, 2, 3]);
//! assert!(!decoder.is_accepting(after_a));
//!
//! // After `abd` the automaton accepts, so end-of-sequence becomes legal.
//! let after_b = decoder.step(after_a, 1)?;
//! let after_d = decoder.step(after_b, 3)?;
//! assert!(decoder.is_accepting(after_d));
//! assert!(decoder.allowed_tokens(after_d)?.allows(4));
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! * Thompson (1968), *Programming Techniques: Regular Expression Search Algorithm*.
//! * Rabin & Scott (1959), *Finite Automata and Their Decision Problems* — the subset
//!   construction.
//! * Willard & Louf (2023), *Efficient Guided Generation for Large Language Models* —
//!   the vocabulary-indexed finite-state machine.
//! * The Unicode Consortium, *The Unicode Standard*, Table 3-7, *Well-Formed UTF-8 Byte
//!   Sequences*.

pub mod automaton;
pub mod engine;
pub mod json_schema;
pub mod regex_parser;
pub mod types;

#[cfg(test)]
mod tests;

pub use automaton::{Dfa, DfaState, Nfa, NfaState, ProductMode};
pub use engine::{ConstrainedDecoder, ConstrainedGeneration, VocabIndex, compile_constraint};
pub use json_schema::{compile_json_schema, json_string_char_ast, utf8_scalar_ast};
pub use regex_parser::{digit_class, parse_regex, space_class, word_class};
pub use types::{
    ByteAlphabet, ByteRange, CharClass, ConstrainedDecoderConfig, ConstrainedDecodingError,
    ConstrainedDecodingResult, ConstrainedTokenId, ConstrainedVocabulary, Constraint,
    JsonSchemaOptions, PropertyOrder, RegexAst, RegexRepeat, StateId, StaticVocabulary, TokenMask,
};
