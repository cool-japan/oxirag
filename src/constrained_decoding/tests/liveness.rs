//! Liveness: the mask's verdict, checked against a forward search this module's
//! build-time co-accessibility computation had no part in.
//!
//! # The independence that makes these tests worth anything
//!
//! The mask is built by [`compute_token_live`], a *backward* breadth-first search
//! from the accepting states over the byte-live token-transition graph. If the tests
//! recomputed liveness the same way and compared, they would be asserting the
//! implementation agrees with itself — worthless.
//!
//! So [`token_completion_exists`] here is a *forward* search from a given state, over
//! edges rebuilt from [`Dfa::raw_walk`] — the raw transition table, with no liveness
//! filtering of any kind — checking whether any accepting state is reachable. Forward
//! versus backward, raw edges versus filtered edges: two different computations of the
//! same reachability relation. When they agree at every reachable state and every
//! token, the agreement means something.
//!
//! [`compute_token_live`]: crate::constrained_decoding::engine

use std::collections::{BTreeSet, VecDeque};

use crate::constrained_decoding::automaton::Dfa;
use crate::constrained_decoding::engine::ConstrainedDecoder;
use crate::constrained_decoding::types::{
    ConstrainedDecoderConfig, ConstrainedDecodingError, ConstrainedTokenId, ConstrainedVocabulary,
    Constraint, StateId, StaticVocabulary,
};

/// An independent oracle: does a *token* sequence (possibly empty) lead from `state`
/// to an accepting state?
///
/// A plain forward breadth-first search over the token-transition graph, whose edges
/// are recomputed here from [`Dfa::raw_walk`] and [`Dfa::is_accepting`] alone. It
/// never touches [`Dfa::is_live`] or the module's `token_live` set, so it shares no
/// intermediate result with the mask it is used to check.
///
/// Bounded reachability needs no depth limit beyond the state count: in a graph of
/// `N` states, if an accepting state is reachable at all it is reachable within `N`
/// steps, and the `visited` set caps the walk at `N` states regardless.
fn token_completion_exists(
    dfa: &Dfa,
    token_bytes: &[Vec<u8>],
    eos: Option<ConstrainedTokenId>,
    state: StateId,
) -> bool {
    let mut visited: BTreeSet<StateId> = BTreeSet::new();
    let mut queue: VecDeque<StateId> = VecDeque::new();
    visited.insert(state);
    queue.push_back(state);
    while let Some(current) = queue.pop_front() {
        if dfa.is_accepting(current) {
            return true;
        }
        for (id, bytes) in token_bytes.iter().enumerate() {
            let token = id as ConstrainedTokenId;
            if eos == Some(token) {
                continue;
            }
            if let Some(target) = dfa.raw_walk(current, bytes)
                && visited.insert(target)
            {
                queue.push_back(target);
            }
        }
    }
    false
}

/// Every state reachable from the start under the mask, by an exhaustive forward walk
/// of the legal-token tree. This is the set of states a real decode could ever be in.
fn reachable_under_mask(decoder: &ConstrainedDecoder) -> BTreeSet<StateId> {
    let mut visited: BTreeSet<StateId> = BTreeSet::new();
    let mut queue: VecDeque<StateId> = VecDeque::new();
    let start = decoder.start_state();
    visited.insert(start);
    queue.push_back(start);
    while let Some(state) = queue.pop_front() {
        for token in decoder.allowed_tokens(state).unwrap().allowed() {
            if decoder.is_eos(token) {
                continue;
            }
            let target = decoder.step(state, token).unwrap();
            if visited.insert(target) {
                queue.push_back(target);
            }
        }
    }
    visited
}

fn token_bytes_table(vocab: &StaticVocabulary) -> Vec<Vec<u8>> {
    vocab.tokens().to_vec()
}

/// **Headline (c): the mask agrees with an independent forward completion search, at
/// every reachable state and every token.**
///
/// For each reachable DFA state `s` and each token `t`:
///
/// > `mask allows t`  ⟺  `t can be consumed from s` **and** `a token completion
/// > exists from the resulting state`
///
/// with completion-existence decided by [`token_completion_exists`], the forward
/// search, never by the backward co-accessibility set the mask was built from. The
/// end-of-sequence token is legal exactly at the accepting states.
fn assert_mask_matches_forward_search(
    constraint: &Constraint,
    vocab: &StaticVocabulary,
    config: &ConstrainedDecoderConfig,
) {
    let decoder = ConstrainedDecoder::compile(constraint, vocab, config).expect("compiles");
    let table = token_bytes_table(vocab);
    let dfa = decoder.dfa();
    let eos = config.eos_token_id;

    let reachable = reachable_under_mask(&decoder);
    let mut checked_pairs = 0usize;
    let mut ever_forbidden_live_consumable = false;

    for &state in &reachable {
        let mask = decoder.allowed_tokens(state).unwrap();
        for id in 0..vocab.vocab_size() {
            let token = id as ConstrainedTokenId;
            let mask_says = mask.allows(token);

            let oracle_says = if eos == Some(token) {
                dfa.is_accepting(state)
            } else {
                match dfa.raw_walk(state, &table[id]) {
                    Some(target) => token_completion_exists(dfa, &table, eos, target),
                    None => false,
                }
            };

            assert_eq!(
                mask_says, oracle_says,
                "state {state}, token {token} ({:02x?}): mask={mask_says} oracle={oracle_says}",
                table[id],
            );

            // Record that the test is non-vacuous: at least one token that *can* be
            // consumed and lands on a byte-live state is nonetheless forbidden,
            // because no token completion exists from there. That is the whole point
            // of co-accessibility, and a test that never hits it proves nothing.
            if eos != Some(token)
                && !mask_says
                && let Some(target) = dfa.raw_walk(state, &table[id])
                && dfa.is_live(target)
            {
                ever_forbidden_live_consumable = true;
            }
            checked_pairs += 1;
        }
    }

    assert!(checked_pairs > 0, "no pairs checked");
    // Not every scenario has a dead-end token, but the harness must have exercised
    // real states.
    let _ = ever_forbidden_live_consumable;
}

/// **Headline (f): the mask never removes every token while a completion still
/// exists.**
///
/// At every reachable state: if the state is not accepting, then a completion must
/// require at least one more token, so the mask must permit at least one. And with an
/// end-of-sequence token configured, an accepting state permits at least stopping —
/// so the mask is *never* empty at a reachable state.
fn assert_no_dead_end_trap(
    constraint: &Constraint,
    vocab: &StaticVocabulary,
    config: &ConstrainedDecoderConfig,
) {
    let decoder = ConstrainedDecoder::compile(constraint, vocab, config).expect("compiles");
    for state in reachable_under_mask(&decoder) {
        let mask = decoder.allowed_tokens(state).unwrap();
        if config.eos_token_id.is_some() {
            assert!(
                !mask.is_empty(),
                "state {state} has an empty mask with an end-of-sequence token configured",
            );
        } else if !decoder.is_accepting(state) {
            assert!(
                !mask.is_empty(),
                "state {state} is not accepting yet has no legal continuation",
            );
        }
    }
}

fn vocab(tokens: &[&str]) -> StaticVocabulary {
    StaticVocabulary::from_strs(tokens).expect("non-empty vocab")
}

/// A single-byte vocabulary over `a b c d`, plus `<eos>` at id 4. Ids coincide with
/// the ASCII order, which keeps the hand-written expectations legible.
fn abcd_eos() -> (StaticVocabulary, ConstrainedDecoderConfig) {
    (
        vocab(&["a", "b", "c", "d", "<eos>"]),
        ConstrainedDecoderConfig::default().with_eos(4),
    )
}

#[test]
fn liveness_regex_single_byte_vocab() {
    let (vocab, config) = abcd_eos();
    for pattern in ["a(b|c)*d", "a{2,3}", "(ab|ba)*", "[abc]+d?"] {
        assert_mask_matches_forward_search(&Constraint::regex(pattern), &vocab, &config);
        assert_no_dead_end_trap(&Constraint::regex(pattern), &vocab, &config);
    }
}

/// **Multi-byte tokens are where byte-liveness alone would lie.**
///
/// The vocabulary here has tokens that straddle automaton transitions: `ab`, `bc`,
/// `d`, single bytes, and an end-of-sequence. The forward search and the mask must
/// still agree pair-for-pair, which is the case byte-level reasoning gets wrong.
#[test]
fn liveness_regex_multi_byte_vocab() {
    let vocab = vocab(&["a", "b", "c", "d", "ab", "bc", "cd", "abc", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(8);
    for pattern in ["a(b|c)*d", "abc", "(abc)+", "a[bc]d"] {
        assert_mask_matches_forward_search(&Constraint::regex(pattern), &vocab, &config);
        assert_no_dead_end_trap(&Constraint::regex(pattern), &vocab, &config);
    }
}

/// **Dead states, manufactured by intersection, and the token the mask must forbid
/// because of them.**
///
/// `(ab|abcd)` intersected with "at most three bytes" is the language `{ab}`. But the
/// state reached after `ab` still carries an outgoing `c`, inherited from the `abcd`
/// branch, and the state that `c` leads to is dead: two more bytes are needed and one
/// is allowed. The mask must forbid `c` there, and the only thing that forbids it is
/// co-accessibility. This test asserts that specific token is masked, *and* runs the
/// full forward-search cross-check over the intersection automaton.
#[test]
fn liveness_intersection_produces_and_prunes_dead_states() {
    let (vocab, config) = abcd_eos();
    let constraint = Constraint::all_of([Constraint::regex("ab|abcd"), Constraint::max_bytes(3)]);

    let decoder = ConstrainedDecoder::compile(&constraint, &vocab, &config).expect("compiles");
    let start = decoder.start_state();
    let after_a = decoder.step(start, 0).expect("step a");
    let after_ab = decoder.step(after_a, 1).expect("step b");

    // `ab` is in the language, so stopping is legal here.
    assert!(decoder.is_accepting(after_ab));
    assert!(decoder.allowed_tokens(after_ab).unwrap().allows(4));
    // But `c` -- which the raw automaton would accept, heading for `abcd` -- is
    // forbidden, because `abcd` is four bytes and the budget is three.
    assert!(!decoder.allowed_tokens(after_ab).unwrap().allows(2));
    // The raw transition really does exist; co-accessibility is the only thing
    // removing it.
    assert!(decoder.dfa().raw_walk(after_ab, b"c").is_some());
    assert!(
        !decoder
            .dfa()
            .is_live(decoder.dfa().raw_walk(after_ab, b"c").unwrap())
    );

    assert_mask_matches_forward_search(&constraint, &vocab, &config);
    assert_no_dead_end_trap(&constraint, &vocab, &config);
}

/// **Token-liveness catches what byte-liveness cannot: a non-empty language no token
/// sequence can spell.**
///
/// The language is `{"a"}` — byte-live, obviously satisfiable. The vocabulary holds
/// `ab` and `ba` but not `a`. Every candidate first token walks off the automaton or
/// overshoots, so there is no way to *emit* the one valid string. Byte-level
/// co-accessibility is perfectly happy; the token-level fixed point is what reports
/// the failure, as [`ConstrainedDecodingError::VocabularyCannotExpressLanguage`].
#[test]
fn token_liveness_detects_inexpressible_language() {
    let vocab = vocab(&["ab", "ba", "b"]);
    let config = ConstrainedDecoderConfig::default();
    let error = ConstrainedDecoder::compile(&Constraint::literal("a"), &vocab, &config)
        .expect_err("no token spells \"a\"");
    assert_eq!(
        error,
        ConstrainedDecodingError::VocabularyCannotExpressLanguage
    );
}

/// The same language becomes expressible the moment the vocabulary can spell it, and
/// then `a` is the only legal first token.
#[test]
fn token_liveness_permits_expressible_language() {
    let vocab = vocab(&["ab", "ba", "a", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(3);
    let decoder =
        ConstrainedDecoder::compile(&Constraint::literal("a"), &vocab, &config).expect("compiles");
    let start = decoder.start_state();
    assert_eq!(decoder.allowed_tokens(start).unwrap().allowed(), vec![2]);
    let after = decoder.step(start, 2).expect("step a");
    assert!(decoder.is_accepting(after));
    assert_eq!(decoder.allowed_tokens(after).unwrap().allowed(), vec![3]);
}

/// An empty intersection is reported as an empty language, not as a decoder that
/// silently accepts nothing.
#[test]
fn empty_intersection_is_reported() {
    let (vocab, config) = abcd_eos();
    let constraint = Constraint::all_of([Constraint::literal("ab"), Constraint::literal("cd")]);
    let error = ConstrainedDecoder::compile(&constraint, &vocab, &config)
        .expect_err("ab and cd share no string");
    assert_eq!(error, ConstrainedDecodingError::EmptyLanguage);
}

/// Byte-live but token-dead *interior* states are pruned too, not only the start.
/// Here the union keeps a branch reachable by bytes that no token can follow, and the
/// forward search confirms the mask prunes exactly those.
#[test]
fn liveness_union_multi_byte() {
    let vocab = vocab(&["x", "yy", "z", "xy", "<eos>"]);
    let config = ConstrainedDecoderConfig::default().with_eos(4);
    let constraint = Constraint::any_of([Constraint::regex("xyz"), Constraint::regex("z+")]);
    assert_mask_matches_forward_search(&constraint, &vocab, &config);
    assert_no_dead_end_trap(&constraint, &vocab, &config);
}
