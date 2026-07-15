//! Compiling a [`Constraint`] to an automaton, indexing a vocabulary against it,
//! and decoding under the resulting mask.
//!
//! # The two kinds of liveness, and why one is not enough
//!
//! [`Dfa::is_live`](crate::constrained_decoding::Dfa::is_live) answers a question
//! about *byte strings*: from this state, is there any sequence of bytes that
//! reaches an accepting state? That is the classical co-accessibility computation,
//! and it is necessary. It is also **not sufficient**, and the gap is the whole
//! difficulty of constrained decoding.
//!
//! A decoder does not emit bytes. It emits *tokens*, and a token is an indivisible
//! multi-byte lump. Take the language `{"a"}` and the vocabulary `{"ab", "ba"}`.
//! The start state is byte-live — `"a"` is right there in the language — and yet
//! there is no first token: `"ab"` walks off the automaton at its second byte, and
//! so does `"ba"` at its first. The mask is empty, the decoder is stuck, and byte-level
//! co-accessibility said everything was fine. This module reports that case as
//! [`ConstrainedDecodingError::VocabularyCannotExpressLanguage`] rather than
//! discovering it half-way through a generation.
//!
//! So [`VocabIndex`] computes a second, strictly stronger fixed point. Build the
//! graph whose nodes are automaton states and whose edges are `q --t--> δ*(q, bytes(t))`
//! for every token `t` whose bytes can be consumed from `q`; a state is
//! **token-live** when an accepting state is reachable *in that graph*. Only then is
//! it true that
//!
//! > from every state the masked decoder can reach, at least one legal token exists.
//!
//! That statement is the module's no-dead-end guarantee, it is what the mask is
//! actually built from, and it is false if you stop at bytes.
//!
//! Token-liveness implies byte-liveness (a token path is a byte path); the converse
//! is the counterexample above. Byte-liveness is still computed first, because it is
//! cheap, it prunes most of the work before the expensive `states x vocabulary` scan,
//! and it is the thing the tests can check against an independent forward search.

use std::cmp::Ordering;
use std::collections::{BTreeSet, VecDeque};

use crate::constrained_decoding::automaton::{Dfa, Nfa, ProductMode};
use crate::constrained_decoding::json_schema::compile_json_schema;
use crate::constrained_decoding::regex_parser::parse_regex;
use crate::constrained_decoding::types::{
    ByteAlphabet, CharClass, ConstrainedDecoderConfig, ConstrainedDecodingError,
    ConstrainedDecodingResult, ConstrainedTokenId, ConstrainedVocabulary, Constraint, RegexAst,
    RegexRepeat, StateId, TokenMask,
};

/// Token ids are `u32`. [`VocabIndex::build`] rejects a vocabulary larger than
/// `u32::MAX`, so this cannot truncate.
#[allow(clippy::cast_possible_truncation)]
const fn as_token_id(index: usize) -> ConstrainedTokenId {
    index as ConstrainedTokenId
}

/// State ids are `u32`. The automaton builders refuse to allocate past
/// `StateId::MAX`, so this cannot truncate.
#[allow(clippy::cast_possible_truncation)]
const fn as_state_id(index: usize) -> StateId {
    index as StateId
}

// ── Compiling a constraint ───────────────────────────────────────────────────

/// A [`Constraint`] with its leaves already lowered to syntax trees.
///
/// Lowering happens in one pass so that every character class in the whole
/// constraint is known *before* any automaton is built: all the leaves must share
/// one [`ByteAlphabet`], or the products could not pair their transitions up.
enum Lowered {
    Leaf(RegexAst),
    Product(ProductMode, Vec<Lowered>),
}

fn lower_constraint(constraint: &Constraint) -> ConstrainedDecodingResult<Lowered> {
    let lowered = match constraint {
        Constraint::Regex(pattern) => Lowered::Leaf(parse_regex(pattern)?),
        Constraint::Ast(ast) => Lowered::Leaf(ast.clone()),
        Constraint::Literal(bytes) => Lowered::Leaf(RegexAst::literal_bytes(bytes)),
        Constraint::JsonSchema { schema, options } => {
            Lowered::Leaf(compile_json_schema(schema, *options)?)
        }
        Constraint::MaxBytes(limit) => {
            let repeat = RegexRepeat {
                min: 0,
                max: Some(*limit),
            };
            repeat.validate()?;
            Lowered::Leaf(RegexAst::repeat(
                RegexAst::Class(CharClass::any_byte()),
                repeat,
            ))
        }
        Constraint::AllOf(children) => Lowered::Product(
            ProductMode::Intersection,
            lower_children(children, "all_of")?,
        ),
        Constraint::AnyOf(children) => {
            Lowered::Product(ProductMode::Union, lower_children(children, "any_of")?)
        }
    };
    Ok(lowered)
}

fn lower_children(
    children: &[Constraint],
    name: &'static str,
) -> ConstrainedDecodingResult<Vec<Lowered>> {
    if children.is_empty() {
        return Err(ConstrainedDecodingError::InvalidSchema {
            path: name.to_string(),
            reason: format!("`{name}` needs at least one child constraint"),
        });
    }
    children.iter().map(lower_constraint).collect()
}

fn collect_classes(lowered: &Lowered, out: &mut Vec<CharClass>) {
    match lowered {
        Lowered::Leaf(ast) => ast.for_each_class(&mut |class| out.push(class.clone())),
        Lowered::Product(_, children) => {
            for child in children {
                collect_classes(child, out);
            }
        }
    }
}

fn build_automaton(
    lowered: &Lowered,
    alphabet: &ByteAlphabet,
    max_states: usize,
) -> ConstrainedDecodingResult<Dfa> {
    match lowered {
        Lowered::Leaf(ast) => {
            let nfa = Nfa::compile(ast, max_states)?;
            Dfa::from_nfa(&nfa, alphabet, max_states)
        }
        Lowered::Product(mode, children) => {
            let mut folded: Option<Dfa> = None;
            for child in children {
                let next = build_automaton(child, alphabet, max_states)?;
                folded = Some(match folded {
                    None => next,
                    Some(accumulated) => Dfa::product(&accumulated, &next, *mode, max_states)?,
                });
            }
            folded.ok_or(ConstrainedDecodingError::EmptyLanguage)
        }
    }
}

/// Compile a [`Constraint`] into a deterministic automaton with its dead states
/// marked.
///
/// # Errors
///
/// Returns [`ConstrainedDecodingError::EmptyLanguage`] when the constraint accepts
/// no string at all — which an intersection very easily produces — plus whatever the
/// regex parser, the schema compiler and the automaton builders return.
pub fn compile_constraint(
    constraint: &Constraint,
    max_states: usize,
) -> ConstrainedDecodingResult<Dfa> {
    let lowered = lower_constraint(constraint)?;
    let mut classes = Vec::new();
    collect_classes(&lowered, &mut classes);
    let alphabet = ByteAlphabet::from_classes(&classes);
    let dfa = build_automaton(&lowered, &alphabet, max_states)?;
    if dfa.is_empty_language() {
        return Err(ConstrainedDecodingError::EmptyLanguage);
    }
    Ok(dfa)
}

// ── VocabIndex ───────────────────────────────────────────────────────────────

/// Sentinel for "this token cannot be consumed from this state" in the dense
/// build-time edge table.
const NO_EDGE: StateId = StateId::MAX;

/// For every automaton state, the set of token ids that may legally be emitted from
/// it.
///
/// See the [module documentation](crate::constrained_decoding::engine) for the
/// difference between byte-liveness and token-liveness, which is what this index
/// exists to compute.
///
/// The masks are dense (`states x vocabulary` bits) and are built eagerly. The
/// `states x vocabulary` table of *successor states* used to compute them is not
/// retained: token byte strings are short, so re-walking one during
/// [`ConstrainedDecoder::step`] is cheaper than storing four bytes per cell forever.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabIndex {
    vocab_size: usize,
    token_bytes: Vec<Vec<u8>>,
    eos_token_id: Option<ConstrainedTokenId>,
    masks: Vec<TokenMask>,
    token_live: Vec<bool>,
}

impl VocabIndex {
    /// Index `vocab` against `dfa`.
    ///
    /// # Errors
    ///
    /// * [`ConstrainedDecodingError::EmptyVocabulary`] — no tokens.
    /// * [`ConstrainedDecodingError::EmptyToken`] — a token with no bytes that is not
    ///   the end-of-sequence token. Such a token could be emitted forever without
    ///   advancing the automaton.
    /// * [`ConstrainedDecodingError::TokenOutOfRange`] — the configured
    ///   end-of-sequence id is not in the vocabulary.
    /// * [`ConstrainedDecodingError::VocabIndexTooLarge`] — `states x vocab_size`
    ///   exceeds `config.max_index_cells`.
    /// * [`ConstrainedDecodingError::EmptyLanguage`] — the automaton accepts nothing.
    /// * [`ConstrainedDecodingError::VocabularyCannotExpressLanguage`] — it accepts
    ///   something, but no token sequence spells any of it.
    pub fn build<V: ConstrainedVocabulary + ?Sized>(
        dfa: &Dfa,
        vocab: &V,
        config: &ConstrainedDecoderConfig,
    ) -> ConstrainedDecodingResult<Self> {
        let vocab_size = vocab.vocab_size();
        if vocab_size == 0 {
            return Err(ConstrainedDecodingError::EmptyVocabulary);
        }
        if vocab_size > ConstrainedTokenId::MAX as usize {
            return Err(ConstrainedDecodingError::TokenOutOfRange {
                token_id: ConstrainedTokenId::MAX,
                vocab_size,
            });
        }
        let states = dfa.state_count();
        let cells =
            states
                .checked_mul(vocab_size)
                .ok_or(ConstrainedDecodingError::VocabIndexTooLarge {
                    states,
                    vocab_size,
                    limit: config.max_index_cells,
                })?;
        if cells > config.max_index_cells {
            return Err(ConstrainedDecodingError::VocabIndexTooLarge {
                states,
                vocab_size,
                limit: config.max_index_cells,
            });
        }

        let eos_token_id = config.eos_token_id;
        if let Some(eos) = eos_token_id
            && eos as usize >= vocab_size
        {
            return Err(ConstrainedDecodingError::TokenOutOfRange {
                token_id: eos,
                vocab_size,
            });
        }

        let token_bytes: Vec<Vec<u8>> = (0..vocab_size)
            .map(|index| vocab.token_bytes(as_token_id(index)).to_vec())
            .collect();
        for (index, bytes) in token_bytes.iter().enumerate() {
            if bytes.is_empty() && eos_token_id != Some(as_token_id(index)) {
                return Err(ConstrainedDecodingError::EmptyToken {
                    token_id: as_token_id(index),
                });
            }
        }

        // Edges of the token-transition graph: `q --t--> δ*(q, bytes(t))`, over the
        // byte-live automaton. Dead byte-states have no edges, and no way in.
        let mut next: Vec<StateId> = vec![NO_EDGE; cells];
        let mut predecessors: Vec<BTreeSet<StateId>> = vec![BTreeSet::new(); states];
        for state in 0..states {
            let id = as_state_id(state);
            if !dfa.is_live(id) {
                continue;
            }
            for token in 0..vocab_size {
                if eos_token_id == Some(as_token_id(token)) {
                    continue;
                }
                if let Some(target) = dfa.walk(id, &token_bytes[token]) {
                    next[state * vocab_size + token] = target;
                    predecessors[target as usize].insert(id);
                }
            }
        }

        let token_live = compute_token_live(dfa, &predecessors, states);

        let start = dfa.start();
        if !dfa.is_live(start) {
            return Err(ConstrainedDecodingError::EmptyLanguage);
        }
        if !token_live[start as usize] {
            return Err(ConstrainedDecodingError::VocabularyCannotExpressLanguage);
        }

        let mut masks: Vec<TokenMask> = Vec::with_capacity(states);
        for state in 0..states {
            let mut mask = TokenMask::forbid_all(vocab_size);
            // Nothing is legal at a state no token sequence can rescue. Such a state
            // is unreachable under the mask anyway; an empty mask makes that a fact
            // rather than a hope.
            if token_live[state] {
                for token in 0..vocab_size {
                    if eos_token_id == Some(as_token_id(token)) {
                        continue;
                    }
                    let target = next[state * vocab_size + token];
                    if target != NO_EDGE && token_live[target as usize] {
                        mask.allow(as_token_id(token));
                    }
                }
                // End-of-sequence is legal exactly at the accepting states: that is
                // what makes "the automaton accepts" and "the decoder may stop" the
                // same statement.
                if let Some(eos) = eos_token_id
                    && dfa.is_accepting(as_state_id(state))
                {
                    mask.allow(eos);
                }
            }
            masks.push(mask);
        }

        Ok(Self {
            vocab_size,
            token_bytes,
            eos_token_id,
            masks,
            token_live,
        })
    }

    /// The vocabulary size the index was built against.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The end-of-sequence token, if one was configured.
    #[must_use]
    pub fn eos_token_id(&self) -> Option<ConstrainedTokenId> {
        self.eos_token_id
    }

    /// The bytes of token `id`; empty for an out-of-range id.
    #[must_use]
    pub fn token_bytes(&self, id: ConstrainedTokenId) -> &[u8] {
        self.token_bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Whether some *token sequence* leads from `state` to an accepting state.
    ///
    /// Strictly stronger than
    /// [`Dfa::is_live`](crate::constrained_decoding::Dfa::is_live).
    #[must_use]
    pub fn is_token_live(&self, state: StateId) -> bool {
        self.token_live
            .get(state as usize)
            .copied()
            .unwrap_or(false)
    }

    /// The legal tokens at `state`.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::StateOutOfRange`] for an unknown state.
    pub fn mask(&self, state: StateId) -> ConstrainedDecodingResult<&TokenMask> {
        self.masks
            .get(state as usize)
            .ok_or(ConstrainedDecodingError::StateOutOfRange {
                state,
                state_count: self.masks.len(),
            })
    }

    /// Drive every illegal token's logit to `f64::NEG_INFINITY`, in place.
    ///
    /// Legal logits are left exactly as they were: this is a *mask*, not a bias. The
    /// model's preferences among the tokens it is allowed to emit are none of this
    /// module's business.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::LogitsVocabMismatch`] when `logits` is not
    /// `vocab_size` long, and [`ConstrainedDecodingError::StateOutOfRange`] for an
    /// unknown state.
    pub fn mask_logits(&self, logits: &mut [f64], state: StateId) -> ConstrainedDecodingResult<()> {
        if logits.len() != self.vocab_size {
            return Err(ConstrainedDecodingError::LogitsVocabMismatch {
                expected: self.vocab_size,
                actual: logits.len(),
            });
        }
        let mask = self.mask(state)?;
        for (index, logit) in logits.iter_mut().enumerate() {
            if !mask.allows(as_token_id(index)) {
                *logit = f64::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

/// Backward breadth-first search from the accepting states over the token-transition
/// graph.
fn compute_token_live(dfa: &Dfa, predecessors: &[BTreeSet<StateId>], states: usize) -> Vec<bool> {
    let mut live = vec![false; states];
    let mut queue: VecDeque<StateId> = VecDeque::new();
    for (state, live_flag) in live.iter_mut().enumerate() {
        // Zero tokens is a legal token sequence, so an accepting state is trivially
        // token-live.
        if dfa.is_accepting(as_state_id(state)) {
            *live_flag = true;
            queue.push_back(as_state_id(state));
        }
    }
    while let Some(state) = queue.pop_front() {
        for &source in &predecessors[state as usize] {
            if !live[source as usize] {
                live[source as usize] = true;
                queue.push_back(source);
            }
        }
    }
    live
}

// ── ConstrainedGeneration ────────────────────────────────────────────────────

/// The outcome of a masked decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstrainedGeneration {
    /// The tokens chosen, in order. The end-of-sequence token is not among them.
    pub tokens: Vec<ConstrainedTokenId>,
    /// Their concatenated bytes: the generated string.
    pub bytes: Vec<u8>,
    /// The automaton state the decode ended in.
    pub final_state: StateId,
    /// Whether the decode stopped because the end-of-sequence token was chosen,
    /// rather than because it ran out of steps.
    pub terminated: bool,
    /// Whether [`ConstrainedGeneration::final_state`] is accepting — i.e. whether
    /// [`ConstrainedGeneration::bytes`] is a *complete* member of the language rather
    /// than a prefix of one.
    ///
    /// A masked decode can only ever produce a prefix of some member of the language;
    /// this is what says the prefix is the whole thing.
    pub accepted: bool,
}

impl ConstrainedGeneration {
    /// The generated bytes as text, if they happen to be valid UTF-8.
    ///
    /// They need not be: a decode cut short mid-codepoint is a perfectly ordinary
    /// prefix, and the point of a byte-level automaton is that it can represent one.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        std::str::from_utf8(&self.bytes).ok()
    }
}

// ── ConstrainedDecoder ───────────────────────────────────────────────────────

/// A compiled constraint plus a vocabulary indexed against it: everything needed to
/// mask a decode step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstrainedDecoder {
    dfa: Dfa,
    index: VocabIndex,
}

impl ConstrainedDecoder {
    /// Compile `constraint` and index `vocab` against it.
    ///
    /// # Errors
    ///
    /// Everything [`compile_constraint`] and [`VocabIndex::build`] can return.
    pub fn compile<V: ConstrainedVocabulary + ?Sized>(
        constraint: &Constraint,
        vocab: &V,
        config: &ConstrainedDecoderConfig,
    ) -> ConstrainedDecodingResult<Self> {
        let dfa = compile_constraint(constraint, config.max_dfa_states)?;
        Self::from_dfa(dfa, vocab, config)
    }

    /// Index `vocab` against an automaton you compiled yourself.
    ///
    /// # Errors
    ///
    /// Everything [`VocabIndex::build`] can return.
    pub fn from_dfa<V: ConstrainedVocabulary + ?Sized>(
        dfa: Dfa,
        vocab: &V,
        config: &ConstrainedDecoderConfig,
    ) -> ConstrainedDecodingResult<Self> {
        let index = VocabIndex::build(&dfa, vocab, config)?;
        Ok(Self { dfa, index })
    }

    /// The compiled automaton.
    #[must_use]
    pub fn dfa(&self) -> &Dfa {
        &self.dfa
    }

    /// The vocabulary index.
    #[must_use]
    pub fn index(&self) -> &VocabIndex {
        &self.index
    }

    /// The vocabulary size, which is also the length every logit slice must have.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.index.vocab_size()
    }

    /// The state a decode starts in.
    #[must_use]
    pub fn start_state(&self) -> StateId {
        self.dfa.start()
    }

    /// The end-of-sequence token, if one was configured.
    #[must_use]
    pub fn eos_token_id(&self) -> Option<ConstrainedTokenId> {
        self.index.eos_token_id()
    }

    /// Whether `token` is the end-of-sequence token.
    #[must_use]
    pub fn is_eos(&self, token: ConstrainedTokenId) -> bool {
        self.index.eos_token_id() == Some(token)
    }

    /// Whether the decode may legally stop in `state`.
    #[must_use]
    pub fn is_accepting(&self, state: StateId) -> bool {
        self.dfa.is_accepting(state)
    }

    /// The bytes of `token`.
    #[must_use]
    pub fn token_bytes(&self, token: ConstrainedTokenId) -> &[u8] {
        self.index.token_bytes(token)
    }

    /// The legal tokens at `state`.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::StateOutOfRange`] for an unknown state.
    pub fn allowed_tokens(&self, state: StateId) -> ConstrainedDecodingResult<&TokenMask> {
        self.index.mask(state)
    }

    /// Drive every illegal token's logit to `f64::NEG_INFINITY`.
    ///
    /// This is the whole product. Everything above exists to make the set of "illegal"
    /// tokens correct.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::LogitsVocabMismatch`] when `logits` is not
    /// `vocab_size` long, and [`ConstrainedDecodingError::StateOutOfRange`] for an
    /// unknown state.
    pub fn mask_logits(&self, logits: &mut [f64], state: StateId) -> ConstrainedDecodingResult<()> {
        self.index.mask_logits(logits, state)
    }

    /// Advance the automaton by accepting `token`.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::CannotStepEos`] for the end-of-sequence
    /// token, which terminates rather than advances;
    /// [`ConstrainedDecodingError::IllegalToken`] for a token the mask forbids; and
    /// [`ConstrainedDecodingError::StateOutOfRange`] for an unknown state.
    pub fn step(
        &self,
        state: StateId,
        token: ConstrainedTokenId,
    ) -> ConstrainedDecodingResult<StateId> {
        if self.is_eos(token) {
            return Err(ConstrainedDecodingError::CannotStepEos { token_id: token });
        }
        if !self.index.mask(state)?.allows(token) {
            return Err(ConstrainedDecodingError::IllegalToken {
                token_id: token,
                state,
            });
        }
        self.dfa.walk(state, self.index.token_bytes(token)).ok_or(
            ConstrainedDecodingError::IllegalToken {
                token_id: token,
                state,
            },
        )
    }

    /// Greedily decode a whole sequence from a table of per-step logit vectors,
    /// masking each step first.
    ///
    /// The choice is the largest *masked* logit, compared with `f64::total_cmp`
    /// specifically so that `f64::NEG_INFINITY` orders below everything and a `NaN`
    /// never silently mis-orders — the same reason
    /// `watermarking::WatermarkGenerator` uses it.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::LogitsVocabMismatch`] when a step's vector
    /// is the wrong length, and [`ConstrainedDecodingError::NoLegalToken`] when the
    /// largest masked logit is one the mask forbids — which happens only if the caller
    /// supplied `f64::NEG_INFINITY` as the logit of every legal token, since every
    /// state a masked decode can reach has at least one.
    pub fn decode_greedy(
        &self,
        logits_per_step: &[Vec<f64>],
    ) -> ConstrainedDecodingResult<ConstrainedGeneration> {
        let mut state = self.start_state();
        let mut tokens: Vec<ConstrainedTokenId> = Vec::new();
        let mut bytes: Vec<u8> = Vec::new();
        let mut terminated = false;

        for logits in logits_per_step {
            let mut masked = logits.clone();
            self.mask_logits(&mut masked, state)?;
            let choice = argmax(&masked).ok_or(ConstrainedDecodingError::NoLegalToken { state })?;
            if !self.index.mask(state)?.allows(choice) {
                return Err(ConstrainedDecodingError::NoLegalToken { state });
            }
            if self.is_eos(choice) {
                terminated = true;
                break;
            }
            bytes.extend_from_slice(self.index.token_bytes(choice));
            tokens.push(choice);
            state = self.step(state, choice)?;
        }

        Ok(ConstrainedGeneration {
            tokens,
            bytes,
            final_state: state,
            terminated,
            accepted: self.is_accepting(state),
        })
    }

    /// Every string the masked decoder could emit in at most `max_tokens` tokens.
    ///
    /// A depth-first walk of the token tree the mask permits, recording the byte
    /// string at every accepting state. Intended for small languages: exact for a
    /// bounded one, a truncation for anything else.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::EnumerationLimitExceeded`] when more than
    /// `max_outputs` distinct strings are found, and
    /// [`ConstrainedDecodingError::StateOutOfRange`] if the automaton is inconsistent.
    pub fn enumerate(
        &self,
        max_tokens: usize,
        max_outputs: usize,
    ) -> ConstrainedDecodingResult<Vec<Vec<u8>>> {
        let mut outputs: Vec<Vec<u8>> = Vec::new();
        let mut prefix: Vec<u8> = Vec::new();
        self.enumerate_from(
            self.start_state(),
            0,
            max_tokens,
            max_outputs,
            &mut prefix,
            &mut outputs,
        )?;
        Ok(outputs)
    }

    fn enumerate_from(
        &self,
        state: StateId,
        depth: usize,
        max_tokens: usize,
        max_outputs: usize,
        prefix: &mut Vec<u8>,
        outputs: &mut Vec<Vec<u8>>,
    ) -> ConstrainedDecodingResult<()> {
        if self.is_accepting(state) {
            if outputs.len() >= max_outputs {
                return Err(ConstrainedDecodingError::EnumerationLimitExceeded {
                    limit: max_outputs,
                });
            }
            outputs.push(prefix.clone());
        }
        if depth >= max_tokens {
            return Ok(());
        }
        for token in self.index.mask(state)?.allowed() {
            if self.is_eos(token) {
                continue;
            }
            let target = self.step(state, token)?;
            let token_bytes = self.index.token_bytes(token);
            let restore = prefix.len();
            prefix.extend_from_slice(token_bytes);
            self.enumerate_from(target, depth + 1, max_tokens, max_outputs, prefix, outputs)?;
            prefix.truncate(restore);
        }
        Ok(())
    }
}

/// The index of the largest value, ties broken toward the lowest index, via
/// `f64::total_cmp` so that `f64::NEG_INFINITY` orders correctly and a `NaN` never
/// mis-orders or panics.
fn argmax(values: &[f64]) -> Option<ConstrainedTokenId> {
    let mut best: Option<(usize, f64)> = None;
    for (index, &value) in values.iter().enumerate() {
        let better = match best {
            None => true,
            Some((_, incumbent)) => value.total_cmp(&incumbent) == Ordering::Greater,
        };
        if better {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| as_token_id(index))
}
