//! The masked decoder: everything it can emit is in the language, and the mask is a
//! faithful logit transform.

use std::collections::BTreeSet;

use crate::constrained_decoding::automaton::Nfa;
use crate::constrained_decoding::engine::ConstrainedDecoder;
use crate::constrained_decoding::regex_parser::parse_regex;
use crate::constrained_decoding::types::{
    ConstrainedDecoderConfig, ConstrainedDecodingError, ConstrainedTokenId, Constraint, StateId,
    StaticVocabulary,
};

fn nfa_for(pattern: &str) -> Nfa {
    Nfa::compile(&parse_regex(pattern).unwrap(), 100_000).unwrap()
}

fn vocab(tokens: &[&str]) -> StaticVocabulary {
    StaticVocabulary::from_strs(tokens).expect("non-empty vocab")
}

/// Exhaustively collect every byte string the masked decoder could terminate on,
/// within `max_tokens` tokens, by walking the legal-token tree with the decoder's own
/// mask and `step` — but recording acceptance independently, at every accepting state.
///
/// This is the drive-every-sampling-choice enumeration of headline (d). It uses only
/// [`ConstrainedDecoder::allowed_tokens`] and [`ConstrainedDecoder::step`]; the
/// *judgement* of whether each output is valid is made afterwards, by a different
/// engine.
fn all_terminable_outputs(decoder: &ConstrainedDecoder, max_tokens: usize) -> BTreeSet<Vec<u8>> {
    fn walk(
        decoder: &ConstrainedDecoder,
        state: StateId,
        prefix: &mut Vec<u8>,
        depth: usize,
        max_tokens: usize,
        out: &mut BTreeSet<Vec<u8>>,
    ) {
        // Terminable here iff the automaton accepts -- exactly the states at which the
        // decoder is allowed to stop (and at which the end-of-sequence token, if any,
        // is unmasked).
        if decoder.is_accepting(state) {
            out.insert(prefix.clone());
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

/// **Headline (d): every string the masked decoder can emit is in the language.**
///
/// Drive the decoder over *every* sampling choice up to a bounded token count, and
/// assert every terminable output is accepted by the **NFA** — a different engine from
/// the DFA the mask was compiled from. Nothing outside the language is reachable, no
/// matter what a sampler prefers.
fn assert_every_output_in_language(
    pattern: &str,
    vocab: &StaticVocabulary,
    config: &ConstrainedDecoderConfig,
    max_tokens: usize,
) -> usize {
    let decoder =
        ConstrainedDecoder::compile(&Constraint::regex(pattern), vocab, config).expect("compiles");
    let nfa = nfa_for(pattern);

    let outputs = all_terminable_outputs(&decoder, max_tokens);
    assert!(!outputs.is_empty(), "pattern {pattern:?} emitted nothing");
    for output in &outputs {
        assert!(
            nfa.accepts(output),
            "pattern {pattern:?}: decoder emitted {output:02x?}, which the NFA rejects",
        );
    }
    outputs.len()
}

#[test]
fn decoder_only_emits_language_members() {
    let vocab = vocab(&["a", "b", "c", "d", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(4);
    assert_every_output_in_language("a(b|c)*d", &vocab, &config, 8);
    assert_every_output_in_language("a{2,3}", &vocab, &config, 6);
    assert_every_output_in_language("(ab|ba)*", &vocab, &config, 6);
    assert_every_output_in_language("[abc]+d", &vocab, &config, 6);
}

#[test]
fn decoder_only_emits_language_members_multi_byte() {
    let vocab = vocab(&["a", "b", "c", "d", "ab", "bc", "cd", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(7);
    assert_every_output_in_language("a(b|c)*d", &vocab, &config, 6);
    assert_every_output_in_language("(abc)+", &vocab, &config, 6);
}

/// The decoder's own [`ConstrainedDecoder::enumerate`] must return exactly the
/// terminable-output set the independent walk finds — the two enumerations agree.
#[test]
fn enumerate_matches_independent_walk() {
    let vocab = vocab(&["a", "b", "c", "d", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(4);
    let decoder =
        ConstrainedDecoder::compile(&Constraint::regex("a(b|c)*d"), &vocab, &config).unwrap();

    let independent = all_terminable_outputs(&decoder, 6);
    let built_in: BTreeSet<Vec<u8>> = decoder.enumerate(6, 100_000).unwrap().into_iter().collect();
    assert_eq!(independent, built_in);
}

/// **The mask is a pure logit transform.** Legal logits are left bit-for-bit
/// untouched; illegal ones become `f64::NEG_INFINITY`. It is a mask, not a bias: the
/// model's preferences among legal tokens are preserved exactly.
#[test]
fn mask_logits_preserves_legal_and_kills_illegal() {
    let vocab = vocab(&["a", "b", "c", "d", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(4);
    let decoder =
        ConstrainedDecoder::compile(&Constraint::regex("a(b|c)*d"), &vocab, &config).unwrap();

    let start = decoder.start_state();
    // At the start only `a` (id 0) is legal.
    let original = vec![1.5, -2.0, 3.0, 0.25, 42.0];
    let mut logits = original.clone();
    decoder.mask_logits(&mut logits, start).unwrap();
    assert_eq!(logits[0], original[0]); // legal, untouched
    for &illegal in &logits[1..] {
        assert_eq!(illegal, f64::NEG_INFINITY);
    }
}

/// The length guard mirrors `watermarking::WatermarkGenerator::bias_logits`.
#[test]
fn mask_logits_rejects_wrong_length() {
    let vocab = vocab(&["a", "b", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(2);
    let decoder = ConstrainedDecoder::compile(&Constraint::regex("a+"), &vocab, &config).unwrap();
    let mut wrong = vec![0.0; 2];
    assert_eq!(
        decoder.mask_logits(&mut wrong, decoder.start_state()),
        Err(ConstrainedDecodingError::LogitsVocabMismatch {
            expected: 3,
            actual: 2,
        })
    );
}

/// **Greedy decoding under an adversarial model still stays in the language.**
///
/// The model here always ranks an *illegal* token highest at every step. Because the
/// mask has already driven that token to `-inf`, `argmax` (via `f64::total_cmp`, so
/// `-inf` orders below every finite logit) picks the best *legal* token instead. The
/// result is guaranteed to be in the language whatever the model wanted.
#[test]
fn greedy_decode_ignores_illegal_preferences() {
    let vocab = vocab(&["a", "b", "c", "d", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(4);
    let decoder =
        ConstrainedDecoder::compile(&Constraint::regex("a(b|c)*d"), &vocab, &config).unwrap();
    let nfa = nfa_for("a(b|c)*d");

    // Six steps, each preferring the (illegal-at-the-start) end-of-sequence token, then
    // `d`, then `a`... a schedule engineered to derail an unmasked decoder.
    let logits_per_step: Vec<Vec<f64>> = vec![
        vec![0.1, 0.2, 0.3, 5.0, 9.0], // wants eos, then d
        vec![0.1, 5.0, 0.3, 9.0, 8.0], // wants eos, then d
        vec![0.1, 0.2, 5.0, 9.0, 8.0],
        vec![0.1, 0.2, 0.3, 9.0, 8.0],
        vec![0.1, 0.2, 0.3, 0.4, 9.0],
        vec![0.1, 0.2, 0.3, 0.4, 9.0],
    ];
    let generation = decoder.decode_greedy(&logits_per_step).unwrap();

    assert!(generation.terminated);
    assert!(generation.accepted);
    assert!(nfa.accepts(&generation.bytes));
    // Its first token could only have been `a`; the last real token, `d`.
    assert_eq!(generation.tokens.first(), Some(&0));
}

/// Greedy decoding stops the instant the end-of-sequence token is the best legal
/// choice at an accepting state.
#[test]
fn greedy_decode_terminates_on_eos() {
    let vocab = vocab(&["a", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(1);
    let decoder = ConstrainedDecoder::compile(&Constraint::regex("a+"), &vocab, &config).unwrap();

    // Step 1 must emit `a` (eos illegal before any `a`); step 2 prefers eos.
    let logits_per_step = vec![vec![9.0, 100.0], vec![0.0, 100.0]];
    let generation = decoder.decode_greedy(&logits_per_step).unwrap();
    assert_eq!(generation.tokens, vec![0]);
    assert_eq!(generation.bytes, b"a");
    assert!(generation.terminated);
    assert!(generation.accepted);
}

/// Stepping the end-of-sequence token is a usage error: it terminates, it does not
/// advance.
#[test]
fn stepping_eos_is_an_error() {
    let vocab = vocab(&["a", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(1);
    let decoder = ConstrainedDecoder::compile(&Constraint::regex("a+"), &vocab, &config).unwrap();
    let after_a = decoder.step(decoder.start_state(), 0).unwrap();
    assert_eq!(
        decoder.step(after_a, 1),
        Err(ConstrainedDecodingError::CannotStepEos { token_id: 1 })
    );
}

/// Stepping a masked token is refused rather than silently walking off the automaton.
#[test]
fn stepping_illegal_token_is_an_error() {
    let vocab = vocab(&["a", "b", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(2);
    let decoder = ConstrainedDecoder::compile(&Constraint::regex("a+"), &vocab, &config).unwrap();
    // `b` (id 1) is never legal.
    assert!(matches!(
        decoder.step(decoder.start_state(), 1),
        Err(ConstrainedDecodingError::IllegalToken { token_id: 1, .. })
    ));
}

/// A zero-byte non-EOS token is rejected at build time: it could be emitted forever
/// without advancing the automaton, so it can never be legal.
#[test]
fn empty_non_eos_token_is_rejected() {
    let vocab = vocab(&["a", "", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(2);
    assert_eq!(
        ConstrainedDecoder::compile(&Constraint::regex("a+"), &vocab, &config),
        Err(ConstrainedDecodingError::EmptyToken { token_id: 1 })
    );
}

/// Without an end-of-sequence token the decoder never terminates itself, but the
/// caller can still read acceptance off the state, and every prefix it produces is a
/// prefix of some language member.
#[test]
fn decode_without_eos_reports_acceptance() {
    let vocab = vocab(&["a", "b", "c", "d"]);
    let config = ConstrainedDecoderConfig::default();
    let decoder =
        ConstrainedDecoder::compile(&Constraint::regex("a(b|c)*d"), &vocab, &config).unwrap();
    let logits_per_step = vec![vec![9.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 9.0]];
    let generation = decoder.decode_greedy(&logits_per_step).unwrap();
    assert!(!generation.terminated); // ran out of steps, not an eos
    assert_eq!(generation.bytes, b"ad");
    assert!(generation.accepted); // "ad" is in the language
}

/// The token ids reported legal are consistent with what `step` will accept: nothing
/// the mask permits can fail to step.
#[test]
fn allowed_tokens_are_always_steppable() {
    let vocab = vocab(&["a", "b", "c", "d", "ab", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(5);
    let decoder =
        ConstrainedDecoder::compile(&Constraint::regex("a(b|c)*d"), &vocab, &config).unwrap();

    let mut queue = vec![decoder.start_state()];
    let mut seen: BTreeSet<StateId> = BTreeSet::new();
    while let Some(state) = queue.pop() {
        if !seen.insert(state) {
            continue;
        }
        for token in decoder.allowed_tokens(state).unwrap().allowed() {
            if decoder.is_eos(token) {
                assert!(decoder.is_accepting(state));
                continue;
            }
            let target = decoder
                .step(state, token as ConstrainedTokenId)
                .expect("a permitted token must step");
            queue.push(target);
        }
    }
}
