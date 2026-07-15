//! Thompson construction, subset construction, co-accessibility, and the product
//! of two automata.
//!
//! The pipeline is the textbook one, over bytes rather than characters:
//!
//! ```text
//! RegexAst --Thompson--> Nfa --subset construction--> Dfa --backward BFS--> live
//! ```
//!
//! # Two engines, on purpose
//!
//! [`Nfa::accepts`] simulates the non-deterministic automaton directly — an
//! epsilon-closure over a set of states, tested against raw bytes via
//! `CharClass::contains` — and never so much as looks at a [`ByteAlphabet`].
//! [`Dfa::accepts`] runs the determinised table, whose every transition was
//! decided by probing *one representative byte per alphabet block*. The two share
//! no code below the syntax tree.
//!
//! That redundancy is deliberate: the whole point of the alphabet partition is
//! the claim that bytes inside a block are interchangeable, and the only way to
//! believe it is to have something that does not assume it disagree with the thing
//! that does. The module's headline test enumerates every string over a small
//! alphabet and asserts the two engines accept exactly the same set.
//!
//! # Liveness is not decoration
//!
//! A state is **live** (co-accessible, productive) when some byte string — possibly
//! the empty one — leads from it to an accepting state. Everything else is a dead
//! end that the automaton can enter and never leave, and a decoder that walks into
//! one has already emitted a prefix that no continuation can rescue.
//!
//! It is worth being precise about when this matters, because a plausible-sounding
//! claim here is false. For an automaton built by Thompson construction from a
//! regex whose character classes are all non-empty, **every** state is live, and
//! therefore so is every non-empty subset the determiniser can produce: each
//! Thompson state lies on a path to that fragment's accept state by construction.
//! Pruning such a machine removes nothing at all.
//!
//! Dead states appear as soon as an automaton is built by any means *other* than
//! Thompson construction — which is to say, as soon as
//! [`Dfa::product`] is involved:
//!
//! * `(ab|abcd)` intersected with "at most three bytes" is `ab`. But the state
//!   reached after `ab` still has an outgoing `c`, inherited from the `abcd` branch,
//!   and the state that `c` leads to is dead: two more bytes are needed and only one
//!   is left. Nothing but co-accessibility forbids that `c`.
//! * Two disjoint languages intersect to nothing, and the *start* state is dead.
//!
//! An empty character class (`[^\x00-\xff]`) does it too, at the Thompson level,
//! by giving a fragment a transition no byte can take.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::constrained_decoding::types::{
    ByteAlphabet, CharClass, ConstrainedDecodingError, ConstrainedDecodingResult, RegexAst,
    RegexRepeat, StateId,
};

/// The default limit on syntax-tree nesting during Thompson construction.
pub const MAX_AST_DEPTH: u32 = 128;

/// Allocate the next state id, refusing to exceed `limit`.
fn next_state_id(count: usize, limit: usize) -> ConstrainedDecodingResult<StateId> {
    if count >= limit || count >= StateId::MAX as usize {
        return Err(ConstrainedDecodingError::AutomatonTooLarge { limit });
    }
    StateId::try_from(count).map_err(|_| ConstrainedDecodingError::AutomatonTooLarge { limit })
}

// ── NFA ──────────────────────────────────────────────────────────────────────

/// One state of a [`Nfa`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NfaState {
    transitions: Vec<(CharClass, StateId)>,
    epsilons: Vec<StateId>,
}

impl NfaState {
    /// The byte-consuming transitions out of this state.
    #[must_use]
    pub fn transitions(&self) -> &[(CharClass, StateId)] {
        &self.transitions
    }

    /// The epsilon transitions out of this state.
    #[must_use]
    pub fn epsilons(&self) -> &[StateId] {
        &self.epsilons
    }
}

/// A Thompson non-deterministic finite automaton over bytes.
///
/// Thompson's construction gives every fragment exactly one start and one accept
/// state, which is what makes the recursion in [`Nfa::compile`] a two-line case per
/// syntax-tree node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nfa {
    states: Vec<NfaState>,
    start: StateId,
    accept: StateId,
}

impl Nfa {
    /// Compile a syntax tree by Thompson's construction.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::AutomatonTooLarge`] when the tree needs
    /// more than `max_states` states — bounded repetition is compiled by
    /// unrolling, so `a{1000}` really does cost a thousand fragments —
    /// [`ConstrainedDecodingError::NestingTooDeep`] beyond [`MAX_AST_DEPTH`], and
    /// [`ConstrainedDecodingError::InvalidRepeat`] for malformed `{m,n}` bounds.
    pub fn compile(ast: &RegexAst, max_states: usize) -> ConstrainedDecodingResult<Self> {
        let mut builder = NfaBuilder {
            states: Vec::new(),
            max_states,
            depth: 0,
        };
        let fragment = builder.fragment(ast)?;
        Ok(Self {
            states: builder.states,
            start: fragment.start,
            accept: fragment.accept,
        })
    }

    /// The state table.
    #[must_use]
    pub fn states(&self) -> &[NfaState] {
        &self.states
    }

    /// How many states the automaton has.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// The single start state.
    #[must_use]
    pub fn start(&self) -> StateId {
        self.start
    }

    /// The single accept state.
    #[must_use]
    pub fn accept(&self) -> StateId {
        self.accept
    }

    /// Every character class appearing on a transition, de-duplicated.
    ///
    /// This is what a [`ByteAlphabet`] is built from.
    #[must_use]
    pub fn classes(&self) -> Vec<CharClass> {
        let mut unique: BTreeSet<Vec<u8>> = BTreeSet::new();
        let mut classes = Vec::new();
        for state in &self.states {
            for (class, _) in &state.transitions {
                // Key on the normalised range list, which is canonical.
                let key: Vec<u8> = class
                    .ranges()
                    .iter()
                    .flat_map(|range| [range.start, range.end])
                    .collect();
                if unique.insert(key) {
                    classes.push(class.clone());
                }
            }
        }
        classes
    }

    /// The epsilon closure of `seeds`: every state reachable from one of them by
    /// epsilon transitions alone, including the seeds themselves.
    ///
    /// Cycle-safe. Epsilon cycles are not exotic — `(a?)*` produces one — so the
    /// visited set is load-bearing rather than defensive.
    #[must_use]
    pub fn epsilon_closure(&self, seeds: impl IntoIterator<Item = StateId>) -> BTreeSet<StateId> {
        let mut closure: BTreeSet<StateId> = BTreeSet::new();
        let mut stack: Vec<StateId> = Vec::new();
        for seed in seeds {
            if closure.insert(seed) {
                stack.push(seed);
            }
        }
        while let Some(state) = stack.pop() {
            let Some(node) = self.states.get(state as usize) else {
                continue;
            };
            for &target in &node.epsilons {
                if closure.insert(target) {
                    stack.push(target);
                }
            }
        }
        closure
    }

    /// The set of states reachable from `current` by consuming exactly `byte`,
    /// epsilon-closed.
    #[must_use]
    pub fn step(&self, current: &BTreeSet<StateId>, byte: u8) -> BTreeSet<StateId> {
        let mut moved: Vec<StateId> = Vec::new();
        for &state in current {
            let Some(node) = self.states.get(state as usize) else {
                continue;
            };
            for (class, target) in &node.transitions {
                if class.contains(byte) {
                    moved.push(*target);
                }
            }
        }
        self.epsilon_closure(moved)
    }

    /// The set of states the automaton could be in having read `input` from the
    /// start.
    #[must_use]
    pub fn run(&self, input: &[u8]) -> BTreeSet<StateId> {
        let mut current = self.epsilon_closure([self.start]);
        for &byte in input {
            if current.is_empty() {
                return current;
            }
            current = self.step(&current, byte);
        }
        current
    }

    /// Whether the automaton accepts `input` in full.
    ///
    /// This is the reference implementation: a direct set-of-states simulation
    /// that consults `CharClass::contains` on the raw byte and knows nothing
    /// whatever about [`ByteAlphabet`] or the subset construction.
    #[must_use]
    pub fn accepts(&self, input: &[u8]) -> bool {
        self.run(input).contains(&self.accept)
    }
}

/// A Thompson fragment: one entry, one exit.
#[derive(Debug, Clone, Copy)]
struct Fragment {
    start: StateId,
    accept: StateId,
}

struct NfaBuilder {
    states: Vec<NfaState>,
    max_states: usize,
    depth: u32,
}

impl NfaBuilder {
    fn push_state(&mut self) -> ConstrainedDecodingResult<StateId> {
        let id = next_state_id(self.states.len(), self.max_states)?;
        self.states.push(NfaState::default());
        Ok(id)
    }

    fn epsilon(&mut self, from: StateId, to: StateId) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.epsilons.push(to);
        }
    }

    fn transition(&mut self, from: StateId, class: CharClass, to: StateId) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.push((class, to));
        }
    }

    /// `node?`
    fn optional_fragment(&mut self, node: &RegexAst) -> ConstrainedDecodingResult<Fragment> {
        let start = self.push_state()?;
        let accept = self.push_state()?;
        let inner = self.fragment(node)?;
        self.epsilon(start, inner.start);
        self.epsilon(start, accept);
        self.epsilon(inner.accept, accept);
        Ok(Fragment { start, accept })
    }

    /// `node*`
    fn star_fragment(&mut self, node: &RegexAst) -> ConstrainedDecodingResult<Fragment> {
        let start = self.push_state()?;
        let accept = self.push_state()?;
        let inner = self.fragment(node)?;
        self.epsilon(start, inner.start);
        self.epsilon(start, accept);
        // The back-edge. When `node` is nullable this closes an epsilon cycle,
        // which `epsilon_closure` is written to survive.
        self.epsilon(inner.accept, inner.start);
        self.epsilon(inner.accept, accept);
        Ok(Fragment { start, accept })
    }

    /// Chain fragments end to end.
    fn chain(&mut self, pieces: &[Fragment]) -> ConstrainedDecodingResult<Fragment> {
        let (Some(first), Some(last)) = (pieces.first(), pieces.last()) else {
            // An empty chain matches the empty string.
            let start = self.push_state()?;
            let accept = self.push_state()?;
            self.epsilon(start, accept);
            return Ok(Fragment { start, accept });
        };
        for window in pieces.windows(2) {
            self.epsilon(window[0].accept, window[1].start);
        }
        Ok(Fragment {
            start: first.start,
            accept: last.accept,
        })
    }

    fn repeat_fragment(
        &mut self,
        node: &RegexAst,
        repeat: RegexRepeat,
    ) -> ConstrainedDecodingResult<Fragment> {
        repeat.validate()?;
        if repeat.max == Some(0) {
            // `x{0,0}` is the empty string, whatever `x` is.
            return self.chain(&[]);
        }
        let mut pieces: Vec<Fragment> = Vec::new();
        for _ in 0..repeat.min {
            pieces.push(self.fragment(node)?);
        }
        match repeat.max {
            None => {
                let star = self.star_fragment(node)?;
                pieces.push(star);
            }
            Some(max) => {
                // `x{2,4}` is `xx x? x?`: the optional tail contributes 0, 1 or 2
                // further copies, so the whole thing matches 2, 3 or 4 of them.
                for _ in repeat.min..max {
                    pieces.push(self.optional_fragment(node)?);
                }
            }
        }
        self.chain(&pieces)
    }

    fn fragment(&mut self, ast: &RegexAst) -> ConstrainedDecodingResult<Fragment> {
        self.depth += 1;
        if self.depth > MAX_AST_DEPTH {
            return Err(ConstrainedDecodingError::NestingTooDeep {
                limit: MAX_AST_DEPTH,
            });
        }
        let fragment = self.fragment_inner(ast)?;
        self.depth -= 1;
        Ok(fragment)
    }

    fn fragment_inner(&mut self, ast: &RegexAst) -> ConstrainedDecodingResult<Fragment> {
        match ast {
            RegexAst::Empty => self.chain(&[]),
            RegexAst::Class(class) => {
                let start = self.push_state()?;
                let accept = self.push_state()?;
                // An *empty* class is not an error: it is a transition no byte can
                // take, so `accept` becomes unreachable and the fragment's language
                // is empty. That is a legitimate thing to say, and it is one of the
                // few ways a Thompson automaton acquires a state that is not live.
                self.transition(start, class.clone(), accept);
                Ok(Fragment { start, accept })
            }
            RegexAst::Concat(items) => {
                let mut pieces = Vec::with_capacity(items.len());
                for item in items {
                    pieces.push(self.fragment(item)?);
                }
                self.chain(&pieces)
            }
            RegexAst::Alternate(branches) => {
                let start = self.push_state()?;
                let accept = self.push_state()?;
                // An empty alternation accepts *nothing* — not even the empty
                // string. `accept` stays unreachable, and the language is empty.
                for branch in branches {
                    let inner = self.fragment(branch)?;
                    self.epsilon(start, inner.start);
                    self.epsilon(inner.accept, accept);
                }
                Ok(Fragment { start, accept })
            }
            RegexAst::Repeat { node, repeat } => self.repeat_fragment(node, *repeat),
        }
    }
}

// ── DFA ──────────────────────────────────────────────────────────────────────

/// One state of a [`Dfa`]. Transitions are indexed by [`ByteAlphabet`] block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DfaState {
    transitions: Vec<Option<StateId>>,
    accepting: bool,
}

impl DfaState {
    /// Whether the automaton accepts when it halts here.
    #[must_use]
    pub fn accepting(&self) -> bool {
        self.accepting
    }

    /// The transition table, indexed by alphabet block. `None` is "no transition":
    /// the determiniser reached the empty subset.
    #[must_use]
    pub fn transitions(&self) -> &[Option<StateId>] {
        &self.transitions
    }
}

/// How [`Dfa::product`] decides which product states accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProductMode {
    /// Accept when *both* operands accept: the intersection of the two languages.
    Intersection,
    /// Accept when *either* operand accepts: the union of the two languages.
    Union,
}

/// A deterministic finite automaton over bytes, with its co-accessible states
/// marked.
///
/// # Two transition functions, and why
///
/// * [`Dfa::raw_step`] is the determiniser's `δ`, exactly as the subset
///   construction left it. `None` means the empty subset: no path at all.
/// * [`Dfa::step`] additionally refuses to enter a state that is not **live**.
///
/// The mask is built from `step`; the *tests* recompute liveness from `raw_step`
/// and compare. Physically deleting the dead states would collapse the two
/// functions into one and leave nothing to check the pruning against, so they
/// stay.
///
/// Note that `step` need only test liveness at the *end* of a byte string, not at
/// every byte of it: if the final state is live then so is every state on the way
/// there, since each of them reaches the final state and the final state reaches an
/// accepting one. Testing at every byte, as `step` does, is the same answer arrived
/// at sooner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dfa {
    states: Vec<DfaState>,
    start: StateId,
    alphabet: ByteAlphabet,
    live: Vec<bool>,
}

impl Dfa {
    /// Determinise `nfa` by subset construction, over the blocks of `alphabet`.
    ///
    /// `alphabet` must have been built from a class set that includes every class
    /// appearing in `nfa` (see [`Nfa::classes`]); otherwise a block could contain
    /// two bytes that some transition distinguishes, and probing one representative
    /// would answer for the other incorrectly.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::AutomatonTooLarge`] when the determinised
    /// machine would exceed `max_states`. Determinisation is exponential in the
    /// worst case, so this is a real limit and not a formality.
    pub fn from_nfa(
        nfa: &Nfa,
        alphabet: &ByteAlphabet,
        max_states: usize,
    ) -> ConstrainedDecodingResult<Self> {
        let blocks = alphabet.block_count();
        let mut ids: BTreeMap<BTreeSet<StateId>, StateId> = BTreeMap::new();
        let mut subsets: Vec<BTreeSet<StateId>> = Vec::new();
        let mut states: Vec<DfaState> = Vec::new();

        let start_subset = nfa.epsilon_closure([nfa.start()]);
        let start = next_state_id(0, max_states)?;
        ids.insert(start_subset.clone(), start);
        subsets.push(start_subset);
        states.push(DfaState {
            transitions: vec![None; blocks],
            accepting: false,
        });

        let mut cursor = 0usize;
        while cursor < subsets.len() {
            let subset = subsets[cursor].clone();
            states[cursor].accepting = subset.contains(&nfa.accept());

            for block in 0..blocks {
                let representative = alphabet.representative(block)?;
                let target = nfa.step(&subset, representative);
                if target.is_empty() {
                    continue;
                }
                let id = if let Some(existing) = ids.get(&target) {
                    *existing
                } else {
                    let id = next_state_id(subsets.len(), max_states)?;
                    ids.insert(target.clone(), id);
                    subsets.push(target);
                    states.push(DfaState {
                        transitions: vec![None; blocks],
                        accepting: false,
                    });
                    id
                };
                states[cursor].transitions[block] = Some(id);
            }
            cursor += 1;
        }

        let live = compute_live(&states);
        Ok(Self {
            states,
            start,
            alphabet: alphabet.clone(),
            live,
        })
    }

    /// The product of two automata over a shared alphabet.
    ///
    /// A regular language is closed under intersection and union, but a *syntax
    /// tree* is not, which is why these two combinators live here and not in
    /// [`RegexAst`].
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::AlphabetMismatch`] when the operands were
    /// compiled against different alphabets — a product pairs transitions up
    /// block-by-block, so the blocks have to mean the same thing on both sides —
    /// and [`ConstrainedDecodingError::AutomatonTooLarge`] when the product exceeds
    /// `max_states`.
    pub fn product(
        left: &Self,
        right: &Self,
        mode: ProductMode,
        max_states: usize,
    ) -> ConstrainedDecodingResult<Self> {
        type Pair = (Option<StateId>, Option<StateId>);
        if left.alphabet != right.alphabet {
            return Err(ConstrainedDecodingError::AlphabetMismatch);
        }
        let blocks = left.alphabet.block_count();

        let mut ids: BTreeMap<Pair, StateId> = BTreeMap::new();
        let mut pairs: Vec<Pair> = Vec::new();
        let mut states: Vec<DfaState> = Vec::new();

        let start_pair: Pair = (Some(left.start), Some(right.start));
        let start = next_state_id(0, max_states)?;
        ids.insert(start_pair, start);
        pairs.push(start_pair);
        states.push(DfaState {
            transitions: vec![None; blocks],
            accepting: false,
        });

        let mut cursor = 0usize;
        while cursor < pairs.len() {
            let (left_state, right_state) = pairs[cursor];
            let left_accepts = left_state.is_some_and(|state| left.is_accepting(state));
            let right_accepts = right_state.is_some_and(|state| right.is_accepting(state));
            states[cursor].accepting = match mode {
                ProductMode::Intersection => left_accepts && right_accepts,
                ProductMode::Union => left_accepts || right_accepts,
            };

            for block in 0..blocks {
                let next_left = left_state.and_then(|state| left.block_step(state, block));
                let next_right = right_state.and_then(|state| right.block_step(state, block));
                // Under intersection, a side that has fallen off its automaton can
                // never accept again, so the pair is worthless and gets no state at
                // all. Under union it is still worth tracking, because the *other*
                // side may yet accept.
                let target: Pair = match mode {
                    ProductMode::Intersection => match (next_left, next_right) {
                        (Some(_), Some(_)) => (next_left, next_right),
                        _ => continue,
                    },
                    ProductMode::Union => match (next_left, next_right) {
                        (None, None) => continue,
                        pair => pair,
                    },
                };
                let id = if let Some(existing) = ids.get(&target) {
                    *existing
                } else {
                    let id = next_state_id(pairs.len(), max_states)?;
                    ids.insert(target, id);
                    pairs.push(target);
                    states.push(DfaState {
                        transitions: vec![None; blocks],
                        accepting: false,
                    });
                    id
                };
                states[cursor].transitions[block] = Some(id);
            }
            cursor += 1;
        }

        let live = compute_live(&states);
        Ok(Self {
            states,
            start,
            alphabet: left.alphabet.clone(),
            live,
        })
    }

    /// The state table.
    #[must_use]
    pub fn states(&self) -> &[DfaState] {
        &self.states
    }

    /// How many states the automaton has, live and dead alike.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// How many of them are live.
    #[must_use]
    pub fn live_state_count(&self) -> usize {
        self.live.iter().filter(|flag| **flag).count()
    }

    /// The start state.
    #[must_use]
    pub fn start(&self) -> StateId {
        self.start
    }

    /// The byte alphabet the transition table is indexed by.
    #[must_use]
    pub fn alphabet(&self) -> &ByteAlphabet {
        &self.alphabet
    }

    /// Whether halting in `state` accepts. Out-of-range states never accept.
    #[must_use]
    pub fn is_accepting(&self, state: StateId) -> bool {
        self.states
            .get(state as usize)
            .is_some_and(DfaState::accepting)
    }

    /// Whether some byte string leads from `state` to an accepting state.
    ///
    /// Out-of-range states are not live.
    #[must_use]
    pub fn is_live(&self, state: StateId) -> bool {
        self.live.get(state as usize).copied().unwrap_or(false)
    }

    /// Whether the automaton accepts nothing whatsoever.
    #[must_use]
    pub fn is_empty_language(&self) -> bool {
        !self.is_live(self.start)
    }

    /// The determiniser's transition function, before dead states are taken into
    /// account. `None` means no transition exists.
    #[must_use]
    pub fn raw_step(&self, state: StateId, byte: u8) -> Option<StateId> {
        self.block_step(state, self.alphabet.block_of(byte))
    }

    /// The transition function the mask is built from: [`Dfa::raw_step`], refusing
    /// to enter a state from which no accepting state is reachable.
    #[must_use]
    pub fn step(&self, state: StateId, byte: u8) -> Option<StateId> {
        self.raw_step(state, byte)
            .filter(|target| self.is_live(*target))
    }

    /// Follow `bytes` from `state` using [`Dfa::raw_step`].
    #[must_use]
    pub fn raw_walk(&self, state: StateId, bytes: &[u8]) -> Option<StateId> {
        let mut current = state;
        for &byte in bytes {
            current = self.raw_step(current, byte)?;
        }
        Some(current)
    }

    /// Follow `bytes` from `state` using [`Dfa::step`].
    #[must_use]
    pub fn walk(&self, state: StateId, bytes: &[u8]) -> Option<StateId> {
        let mut current = state;
        for &byte in bytes {
            current = self.step(current, byte)?;
        }
        Some(current)
    }

    /// Whether the automaton accepts `input` in full.
    #[must_use]
    pub fn accepts(&self, input: &[u8]) -> bool {
        self.raw_walk(self.start, input)
            .is_some_and(|state| self.is_accepting(state))
    }

    /// Whether the automaton accepts the UTF-8 encoding of `text`.
    #[must_use]
    pub fn matches(&self, text: &str) -> bool {
        self.accepts(text.as_bytes())
    }

    fn block_step(&self, state: StateId, block: usize) -> Option<StateId> {
        self.states
            .get(state as usize)
            .and_then(|node| node.transitions.get(block))
            .copied()
            .flatten()
    }
}

/// Mark every state from which an accepting state is reachable.
///
/// A backward breadth-first search from the accepting states over the reversed
/// transition relation. The test suite checks this against a *forward* search run
/// separately from each state, which is a different algorithm reaching the same
/// answer rather than the same algorithm run twice.
fn compute_live(states: &[DfaState]) -> Vec<bool> {
    let mut predecessors: Vec<Vec<StateId>> = vec![Vec::new(); states.len()];
    for (index, state) in states.iter().enumerate() {
        let Ok(source) = StateId::try_from(index) else {
            continue;
        };
        for target in state.transitions.iter().flatten() {
            predecessors[*target as usize].push(source);
        }
    }

    let mut live = vec![false; states.len()];
    let mut queue: VecDeque<StateId> = VecDeque::new();
    for (index, state) in states.iter().enumerate() {
        if state.accepting {
            live[index] = true;
            if let Ok(id) = StateId::try_from(index) {
                queue.push_back(id);
            }
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
