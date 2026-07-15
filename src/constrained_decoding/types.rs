//! Vocabulary, alphabet, syntax-tree and error types for constrained decoding.
//!
//! Nothing in this file runs an automaton. It defines the byte-level primitives
//! everything else is built out of:
//!
//! * [`CharClass`] / [`ByteRange`] — a set of bytes, normalised to sorted,
//!   disjoint, non-adjacent inclusive ranges.
//! * [`ByteAlphabet`] — the coarsest partition of `0..=255` that every
//!   [`CharClass`] in a compilation unit respects, so the subset construction can
//!   iterate over a handful of *blocks* instead of 256 bytes.
//! * [`RegexAst`] / [`RegexRepeat`] — the abstract syntax tree the regex parser
//!   and the JSON-Schema compiler both produce.
//! * [`Constraint`] — what the caller actually asks for, including the two
//!   combinators ([`Constraint::all_of`], [`Constraint::any_of`]) that are
//!   realised as automaton products rather than syntax-tree nodes.
//! * [`ConstrainedVocabulary`] — the **byte-string** view of a tokenizer, and
//!   [`StaticVocabulary`], a deterministic table-driven implementation of it.
//! * [`TokenMask`] — a bitset over token ids: the thing a decode step is
//!   ultimately gated by.
//! * [`ConstrainedDecodingError`] — the error surface.
//!
//! # Why byte strings and not `&str`
//!
//! A byte-pair-encoding vocabulary is built over *bytes*, and a merge is free to
//! stop in the middle of a multi-byte UTF-8 sequence: `é` is `0xC3 0xA9`, and it
//! is entirely ordinary for a tokenizer to hold `0xC3` and `0xA9` as two separate
//! tokens. Such a token is not a `&str` and never can be. Any constrained-decoding
//! layer typed on `&str` therefore cannot represent — let alone mask — half the
//! vocabulary it is supposed to be masking, so [`ConstrainedVocabulary`] is typed
//! on `&[u8]` and every automaton in this module transitions on bytes.

use std::collections::BTreeMap;

use serde_json::Value;
use thiserror::Error;

/// A token identifier: an index into a [`ConstrainedVocabulary`].
pub type ConstrainedTokenId = u32;

/// A state identifier: an index into an automaton's state table.
pub type StateId = u32;

/// Convenience alias for this module's fallible return type.
pub type ConstrainedDecodingResult<T> = Result<T, ConstrainedDecodingError>;

/// The most properties an object schema may have when
/// [`PropertyOrder::AnyPermutation`] is requested.
///
/// The number of distinct key orderings of an object with `n` properties, `k` of
/// them required, is `sum over S ⊇ required of |S|!`, which for `n = 6` and no
/// required properties is already 1957 alternation branches. The limit exists so
/// that an innocent-looking schema cannot silently detonate the compiler.
pub const MAX_PERMUTED_PROPERTIES: usize = 6;

/// The largest repetition count a single `{m,n}` may expand to.
///
/// Bounded repetition is compiled by *unrolling* — `a{3,5}` becomes `aaa(a)?(a)?`
/// — so the count is a direct multiplier on automaton size. It also caps
/// [`Constraint::max_bytes`], which is `.{0,n}` over every byte.
pub const MAX_REPEAT: u32 = 4096;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced by the `constrained_decoding` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConstrainedDecodingError {
    /// The regex source was malformed.
    #[error("regex syntax error at byte {position}: {reason}")]
    RegexSyntax {
        /// Byte offset into the pattern at which the error was detected.
        position: usize,
        /// What went wrong.
        reason: String,
    },
    /// A `{m,n}` repetition had `m > n`, or a bound larger than [`MAX_REPEAT`].
    #[error("invalid repetition {{{min},{max:?}}}: {reason}")]
    InvalidRepeat {
        /// The lower bound.
        min: u32,
        /// The upper bound, `None` for unbounded.
        max: Option<u32>,
        /// Why it was rejected.
        reason: &'static str,
    },
    /// A regex or a schema nested deeper than the configured limit.
    ///
    /// The limit exists because compilation is recursive: without it a
    /// pathological input would overflow the stack rather than return an error.
    #[error("nesting deeper than the limit of {limit}")]
    NestingTooDeep {
        /// The depth limit that was exceeded.
        limit: u32,
    },
    /// The compiled automaton exceeded the configured state budget.
    #[error("automaton exceeded the budget of {limit} states")]
    AutomatonTooLarge {
        /// The state budget.
        limit: usize,
    },
    /// The constraint's language is empty: no byte string at all satisfies it.
    ///
    /// This is a *property of the constraint*, independent of any vocabulary.
    /// It is produced, for example, by intersecting two disjoint languages, or
    /// by a character class that contains no bytes.
    #[error("the constraint accepts no string at all")]
    EmptyLanguage,
    /// The constraint's language is non-empty, but no *sequence of tokens from
    /// this vocabulary* spells any string in it.
    ///
    /// The language and the vocabulary are each individually fine; they simply
    /// do not fit together. The classic instance is a language whose only member
    /// is `"a"` against a vocabulary that holds `"ab"` and `"ba"` but no `"a"`:
    /// there is a valid string, and there is no way to *emit* it. This is exactly
    /// the failure mode that byte-level co-accessibility alone does not detect —
    /// see [`crate::constrained_decoding::VocabIndex`].
    #[error(
        "the constraint's language is non-empty but no token sequence from this vocabulary spells any member of it"
    )]
    VocabularyCannotExpressLanguage,
    /// The vocabulary was empty.
    #[error("the vocabulary is empty")]
    EmptyVocabulary,
    /// A non-terminal token had an empty byte string.
    ///
    /// A zero-byte token could be appended from any state without advancing the
    /// automaton, so a decoder could emit it forever without ever terminating.
    /// Only the end-of-sequence token is allowed to consume no bytes.
    #[error("token {token_id} has no bytes; only the end-of-sequence token may be empty")]
    EmptyToken {
        /// The offending token id.
        token_id: ConstrainedTokenId,
    },
    /// A token id was outside `0..vocab_size`.
    #[error("token id {token_id} is out of range for a vocabulary of size {vocab_size}")]
    TokenOutOfRange {
        /// The offending token id.
        token_id: ConstrainedTokenId,
        /// The vocabulary size.
        vocab_size: usize,
    },
    /// A state id was outside the automaton's state table.
    #[error("state {state} is out of range for an automaton with {state_count} states")]
    StateOutOfRange {
        /// The offending state id.
        state: StateId,
        /// The number of states the automaton has.
        state_count: usize,
    },
    /// The logit slice handed to a masking call had the wrong length.
    #[error("expected {expected} logits (the vocabulary size), got {actual}")]
    LogitsVocabMismatch {
        /// The vocabulary size, which is how many logits there must be.
        expected: usize,
        /// How many were supplied.
        actual: usize,
    },
    /// A token was stepped that the mask forbids in the current state.
    #[error("token {token_id} is not legal in state {state}")]
    IllegalToken {
        /// The token that was rejected.
        token_id: ConstrainedTokenId,
        /// The state it was rejected in.
        state: StateId,
    },
    /// The end-of-sequence token was passed to a stepping call.
    ///
    /// End-of-sequence consumes no bytes and moves to no state; it *terminates*.
    /// Callers must test for it rather than step it.
    #[error("token {token_id} is the end-of-sequence token and cannot be stepped")]
    CannotStepEos {
        /// The end-of-sequence token id.
        token_id: ConstrainedTokenId,
    },
    /// Greedy decoding found no legal token to emit.
    ///
    /// This cannot happen at a state the masked decoder itself reached — every
    /// such state is token-live, so it has at least one legal continuation or is
    /// accepting. It *can* happen if the caller supplies `-inf` (or nothing but
    /// `-inf`) as the logit of every legal token, or steers the decoder into a
    /// state by hand.
    #[error("no legal token could be selected in state {state}")]
    NoLegalToken {
        /// The state at which decoding got stuck.
        state: StateId,
    },
    /// The vocabulary index would need more cells than the configured budget.
    ///
    /// The index is dense in `states x vocabulary`, so this guards the product.
    #[error("the vocabulary index needs {states} x {vocab_size} cells, over the budget of {limit}")]
    VocabIndexTooLarge {
        /// The automaton's state count.
        states: usize,
        /// The vocabulary size.
        vocab_size: usize,
        /// The cell budget.
        limit: usize,
    },
    /// Two automata that had to share an alphabet did not.
    #[error("cannot combine automata compiled against different byte alphabets")]
    AlphabetMismatch,
    /// A JSON Schema used a keyword this compiler does not implement.
    ///
    /// Unsupported keywords are rejected rather than ignored: silently dropping
    /// a constraint would widen the compiled language, and a decoder masked
    /// against a language *wider* than the schema can emit output the schema
    /// rejects — which is precisely the bug this module exists to make
    /// impossible.
    #[error("unsupported JSON Schema keyword `{keyword}`{}", .detail.as_deref().map(|d| format!(": {d}")).unwrap_or_default())]
    UnsupportedKeyword {
        /// The keyword that was rejected.
        keyword: String,
        /// Optional extra explanation.
        detail: Option<String>,
    },
    /// A JSON Schema was structurally invalid.
    #[error("invalid JSON Schema at {path}: {reason}")]
    InvalidSchema {
        /// A slash-separated path to the offending sub-schema.
        path: String,
        /// What went wrong.
        reason: String,
    },
    /// Enumerating a language produced more outputs than the caller allowed.
    #[error("enumeration produced more than {limit} outputs")]
    EnumerationLimitExceeded {
        /// The output budget.
        limit: usize,
    },
}

// ── ByteRange ────────────────────────────────────────────────────────────────

/// An inclusive range of byte values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ByteRange {
    /// The lowest byte in the range.
    pub start: u8,
    /// The highest byte in the range, inclusive.
    pub end: u8,
}

impl ByteRange {
    /// Construct the inclusive range `start..=end`.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::RegexSyntax`] when `start > end`.
    pub fn new(start: u8, end: u8) -> ConstrainedDecodingResult<Self> {
        if start > end {
            return Err(ConstrainedDecodingError::RegexSyntax {
                position: 0,
                reason: format!("character range is reversed: {start:#04x}-{end:#04x}"),
            });
        }
        Ok(Self { start, end })
    }

    /// Whether `byte` lies inside the range.
    #[must_use]
    pub const fn contains(self, byte: u8) -> bool {
        self.start <= byte && byte <= self.end
    }

    /// How many bytes the range covers (never zero, so there is no `is_empty`).
    #[must_use]
    pub const fn size(self) -> u16 {
        (self.end as u16) - (self.start as u16) + 1
    }
}

// ── CharClass ────────────────────────────────────────────────────────────────

/// A set of byte values, held as sorted, disjoint, non-adjacent inclusive ranges.
///
/// The normal form is canonical: two `CharClass`es denote the same set of bytes
/// if and only if they are `==`. Every constructor and combinator restores it, so
/// [`ByteAlphabet`] can key its partition on class membership without worrying
/// about `[a-b]` and `[ab]` being spelled differently.
///
/// A class may legitimately be *empty* (`[^\x00-\xff]`), in which case no byte can
/// be consumed through it — which is one of the ways an automaton acquires states
/// from which no accepting state is reachable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct CharClass {
    ranges: Vec<ByteRange>,
}

impl CharClass {
    /// The empty class: matches no byte at all.
    #[must_use]
    pub const fn empty() -> Self {
        Self { ranges: Vec::new() }
    }

    /// The class containing exactly `byte`.
    #[must_use]
    pub fn single(byte: u8) -> Self {
        Self {
            ranges: vec![ByteRange {
                start: byte,
                end: byte,
            }],
        }
    }

    /// The class containing every byte from `start` to `end` inclusive.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::RegexSyntax`] when `start > end`.
    pub fn range(start: u8, end: u8) -> ConstrainedDecodingResult<Self> {
        Ok(Self {
            ranges: vec![ByteRange::new(start, end)?],
        })
    }

    /// Every byte, `0x00..=0xff`.
    #[must_use]
    pub fn any_byte() -> Self {
        Self {
            ranges: vec![ByteRange {
                start: 0x00,
                end: 0xff,
            }],
        }
    }

    /// Build a class from arbitrary, possibly overlapping, possibly unsorted
    /// ranges, normalising them.
    #[must_use]
    pub fn from_ranges(ranges: Vec<ByteRange>) -> Self {
        let mut class = Self { ranges };
        class.normalise();
        class
    }

    /// Sort, merge and de-duplicate the ranges into the canonical normal form.
    fn normalise(&mut self) {
        if self.ranges.len() < 2 {
            return;
        }
        self.ranges.sort_unstable();
        let mut merged: Vec<ByteRange> = Vec::with_capacity(self.ranges.len());
        for range in self.ranges.drain(..) {
            match merged.last_mut() {
                // Merge when overlapping *or* adjacent: `[a-b][c-d]` is `[a-d]`.
                // The comparison is done in `u16` because `end + 1` overflows a
                // `u8` at `0xff`, which is exactly where a naive version silently
                // wraps and drops the tail of the class.
                Some(last) if u16::from(range.start) <= u16::from(last.end) + 1 => {
                    if range.end > last.end {
                        last.end = range.end;
                    }
                }
                _ => merged.push(range),
            }
        }
        self.ranges = merged;
    }

    /// The complement of this class within `0x00..=0xff`.
    #[must_use]
    pub fn negate(&self) -> Self {
        let mut ranges = Vec::with_capacity(self.ranges.len() + 1);
        let mut next: u16 = 0x00;
        for range in &self.ranges {
            if u16::from(range.start) > next {
                #[allow(clippy::cast_possible_truncation)] // `next < range.start <= 0xff`.
                let start = next as u8;
                ranges.push(ByteRange {
                    start,
                    end: range.start - 1,
                });
            }
            next = u16::from(range.end) + 1;
        }
        if next <= 0xff {
            #[allow(clippy::cast_possible_truncation)] // Guarded by `next <= 0xff`.
            let start = next as u8;
            ranges.push(ByteRange { start, end: 0xff });
        }
        Self { ranges }
    }

    /// The union of two classes.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut ranges = self.ranges.clone();
        ranges.extend_from_slice(&other.ranges);
        Self::from_ranges(ranges)
    }

    /// The intersection of two classes.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        let mut ranges = Vec::new();
        for left in &self.ranges {
            for right in &other.ranges {
                let start = left.start.max(right.start);
                let end = left.end.min(right.end);
                if start <= end {
                    ranges.push(ByteRange { start, end });
                }
            }
        }
        Self::from_ranges(ranges)
    }

    /// The set difference `self \ other`.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        self.intersect(&other.negate())
    }

    /// Whether `byte` is a member of the class.
    #[must_use]
    pub fn contains(&self, byte: u8) -> bool {
        self.ranges.iter().any(|range| range.contains(byte))
    }

    /// Whether the class contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// How many distinct bytes the class contains.
    #[must_use]
    pub fn cardinality(&self) -> u16 {
        self.ranges.iter().map(|range| range.size()).sum()
    }

    /// The class's normalised ranges.
    #[must_use]
    pub fn ranges(&self) -> &[ByteRange] {
        &self.ranges
    }
}

// ── RegexRepeat ──────────────────────────────────────────────────────────────

/// The bounds of a repetition: `{min,max}`, with `max == None` meaning unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegexRepeat {
    /// The minimum number of repetitions (may be zero).
    pub min: u32,
    /// The maximum number of repetitions, `None` for unbounded.
    pub max: Option<u32>,
}

impl RegexRepeat {
    /// `*` — zero or more.
    #[must_use]
    pub const fn star() -> Self {
        Self { min: 0, max: None }
    }

    /// `+` — one or more.
    #[must_use]
    pub const fn plus() -> Self {
        Self { min: 1, max: None }
    }

    /// `?` — zero or one.
    #[must_use]
    pub const fn optional() -> Self {
        Self {
            min: 0,
            max: Some(1),
        }
    }

    /// `{n}` — exactly `n`.
    #[must_use]
    pub const fn exactly(count: u32) -> Self {
        Self {
            min: count,
            max: Some(count),
        }
    }

    /// `{m,n}` — between `min` and `max` inclusive.
    #[must_use]
    pub const fn between(min: u32, max: u32) -> Self {
        Self {
            min,
            max: Some(max),
        }
    }

    /// `{m,}` — `min` or more.
    #[must_use]
    pub const fn at_least(min: u32) -> Self {
        Self { min, max: None }
    }

    /// Check the bounds are well-formed and within [`MAX_REPEAT`].
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::InvalidRepeat`] when `min > max`, or
    /// when either bound exceeds [`MAX_REPEAT`].
    pub fn validate(self) -> ConstrainedDecodingResult<()> {
        if self.min > MAX_REPEAT {
            return Err(ConstrainedDecodingError::InvalidRepeat {
                min: self.min,
                max: self.max,
                reason: "lower bound exceeds the repetition limit",
            });
        }
        match self.max {
            Some(max) if max > MAX_REPEAT => Err(ConstrainedDecodingError::InvalidRepeat {
                min: self.min,
                max: self.max,
                reason: "upper bound exceeds the repetition limit",
            }),
            Some(max) if self.min > max => Err(ConstrainedDecodingError::InvalidRepeat {
                min: self.min,
                max: self.max,
                reason: "lower bound exceeds upper bound",
            }),
            _ => Ok(()),
        }
    }
}

// ── RegexAst ─────────────────────────────────────────────────────────────────

/// The abstract syntax tree of a compiled pattern.
///
/// Both the regex parser and the JSON-Schema compiler produce this type; from
/// here on there is exactly one code path. Grouping and capture leave no trace —
/// a group is simply the sub-tree it contained, because a matcher that only ever
/// asks "is this whole string in the language?" has no use for capture spans.
///
/// The name is prefixed because the crate root already owns `Expr`, `Op`,
/// `Statement` and `Program` (they belong to `program_of_thought`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexAst {
    /// Matches the empty string, and nothing else.
    Empty,
    /// Matches exactly one byte, drawn from the class.
    ///
    /// An empty class here matches *nothing* — not even the empty string — which
    /// makes the whole sub-tree's language empty.
    Class(CharClass),
    /// Matches the concatenation of its children, in order. An empty `Concat`
    /// matches the empty string.
    Concat(Vec<RegexAst>),
    /// Matches any one of its children. An empty `Alternate` matches *nothing*
    /// (the empty language), which is not the same as matching the empty string.
    Alternate(Vec<RegexAst>),
    /// Matches `node` repeated between `repeat.min` and `repeat.max` times.
    Repeat {
        /// The sub-tree being repeated.
        node: Box<RegexAst>,
        /// The repetition bounds.
        repeat: RegexRepeat,
    },
}

impl RegexAst {
    /// A tree matching the single byte `byte`.
    #[must_use]
    pub fn literal_byte(byte: u8) -> Self {
        Self::Class(CharClass::single(byte))
    }

    /// A tree matching exactly the byte string `bytes`.
    #[must_use]
    pub fn literal_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::Empty;
        }
        Self::Concat(bytes.iter().copied().map(Self::literal_byte).collect())
    }

    /// A tree matching exactly the UTF-8 encoding of `text`.
    #[must_use]
    pub fn literal_str(text: &str) -> Self {
        Self::literal_bytes(text.as_bytes())
    }

    /// A tree matching `node` repeated per `repeat`.
    #[must_use]
    pub fn repeat(node: Self, repeat: RegexRepeat) -> Self {
        Self::Repeat {
            node: Box::new(node),
            repeat,
        }
    }

    /// A tree matching `node` zero or one times.
    #[must_use]
    pub fn optional(node: Self) -> Self {
        Self::repeat(node, RegexRepeat::optional())
    }

    /// Visit every [`CharClass`] in the tree, in a depth-first left-to-right order.
    ///
    /// [`ByteAlphabet::from_classes`] uses this to learn which byte distinctions a
    /// compilation unit is actually capable of making.
    pub fn for_each_class(&self, visit: &mut impl FnMut(&CharClass)) {
        match self {
            Self::Empty => {}
            Self::Class(class) => visit(class),
            Self::Concat(children) | Self::Alternate(children) => {
                for child in children {
                    child.for_each_class(visit);
                }
            }
            Self::Repeat { node, .. } => node.for_each_class(visit),
        }
    }
}

// ── ByteAlphabet ─────────────────────────────────────────────────────────────

/// The coarsest partition of `0x00..=0xff` that every [`CharClass`] in a
/// compilation unit respects.
///
/// Two bytes land in the same *block* exactly when no class in the unit
/// distinguishes them. Because every transition of the automaton is labelled with
/// one of those classes, bytes in the same block are interchangeable everywhere,
/// so the subset construction can enumerate blocks (typically five or ten) rather
/// than all 256 bytes, and the product construction can pair transitions up
/// block-by-block instead of byte-by-byte.
///
/// This is *not* an optimisation the tests take on trust: the NFA simulator
/// ignores the alphabet entirely and tests `CharClass::contains` against raw
/// bytes, so any block that wrongly merges two distinguishable bytes shows up
/// immediately as a disagreement between the two engines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteAlphabet {
    block_of: Vec<u16>,
    representatives: Vec<u8>,
}

impl ByteAlphabet {
    /// Build the partition induced by `classes`.
    ///
    /// With no classes at all every byte is interchangeable and the result has a
    /// single block.
    #[must_use]
    pub fn from_classes(classes: &[CharClass]) -> Self {
        // The signature of a byte is the set of classes that contain it; bytes
        // with equal signatures are indistinguishable and share a block.
        let mut blocks: BTreeMap<Vec<bool>, u16> = BTreeMap::new();
        let mut block_of = vec![0u16; 256];
        let mut representatives: Vec<u8> = Vec::new();

        for byte in 0u16..=0xff {
            #[allow(clippy::cast_possible_truncation)] // Loop bound is `0..=0xff`.
            let byte = byte as u8;
            let signature: Vec<bool> = classes.iter().map(|class| class.contains(byte)).collect();
            let next_id = representatives.len();
            let id = *blocks.entry(signature).or_insert_with(|| {
                representatives.push(byte);
                // `next_id < 256` because there are at most 256 distinct bytes.
                #[allow(clippy::cast_possible_truncation)]
                let id = next_id as u16;
                id
            });
            block_of[byte as usize] = id;
        }

        Self {
            block_of,
            representatives,
        }
    }

    /// How many blocks the partition has (at least one, at most 256).
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.representatives.len()
    }

    /// The block `byte` belongs to.
    #[must_use]
    pub fn block_of(&self, byte: u8) -> usize {
        self.block_of[byte as usize] as usize
    }

    /// A representative byte of `block` — the smallest one it contains.
    ///
    /// Because every class in the unit either contains all of a block or none of
    /// it, transitioning on the representative is the same as transitioning on any
    /// other member.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::StateOutOfRange`] when `block` is not a
    /// block of this partition.
    pub fn representative(&self, block: usize) -> ConstrainedDecodingResult<u8> {
        self.representatives
            .get(block)
            .copied()
            .ok_or(ConstrainedDecodingError::StateOutOfRange {
                #[allow(clippy::cast_possible_truncation)] // Diagnostic only.
                state: block as StateId,
                state_count: self.representatives.len(),
            })
    }

    /// Every byte belonging to `block`, ascending.
    #[must_use]
    pub fn bytes_in_block(&self, block: usize) -> Vec<u8> {
        (0u16..=0xff)
            .filter_map(|byte| {
                #[allow(clippy::cast_possible_truncation)] // Loop bound is `0..=0xff`.
                let byte = byte as u8;
                (self.block_of(byte) == block).then_some(byte)
            })
            .collect()
    }
}

// ── TokenMask ────────────────────────────────────────────────────────────────

/// A bitset over token ids: which tokens a decode step may choose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenMask {
    words: Vec<u64>,
    vocab_size: usize,
}

impl TokenMask {
    /// A mask that forbids every token.
    #[must_use]
    pub fn forbid_all(vocab_size: usize) -> Self {
        Self {
            words: vec![0u64; vocab_size.div_ceil(64)],
            vocab_size,
        }
    }

    /// The vocabulary this mask is over.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Whether `token` may be chosen. Out-of-range ids are never allowed.
    #[must_use]
    pub fn allows(&self, token: ConstrainedTokenId) -> bool {
        let index = token as usize;
        if index >= self.vocab_size {
            return false;
        }
        self.words[index / 64] & (1u64 << (index % 64)) != 0
    }

    /// Permit `token`. Out-of-range ids are ignored.
    pub fn allow(&mut self, token: ConstrainedTokenId) {
        let index = token as usize;
        if index < self.vocab_size {
            self.words[index / 64] |= 1u64 << (index % 64);
        }
    }

    /// Forbid `token`. Out-of-range ids are ignored.
    pub fn forbid(&mut self, token: ConstrainedTokenId) {
        let index = token as usize;
        if index < self.vocab_size {
            self.words[index / 64] &= !(1u64 << (index % 64));
        }
    }

    /// How many tokens are permitted.
    #[must_use]
    pub fn count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Whether *no* token is permitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// Every permitted token id, ascending.
    #[must_use]
    pub fn allowed(&self) -> Vec<ConstrainedTokenId> {
        (0..self.vocab_size)
            .filter_map(|index| {
                #[allow(clippy::cast_possible_truncation)]
                // `vocab_size <= u32::MAX`, enforced on construction.
                let token = index as ConstrainedTokenId;
                self.allows(token).then_some(token)
            })
            .collect()
    }
}

// ── ConstrainedVocabulary ────────────────────────────────────────────────────

/// A tokenizer, viewed the only way a byte-level automaton can use one: as a
/// table from token id to **byte string**.
///
/// See the [module documentation](crate::constrained_decoding) for why this is
/// `&[u8]` and not `&str`.
///
/// Implementations must be **deterministic and total**: `token_bytes` is called
/// once per `(state, token)` pair while the index is built and must return the
/// same slice every time. For an id outside `0..vocab_size` it must return an
/// empty slice rather than panic.
pub trait ConstrainedVocabulary {
    /// How many token ids there are. Ids are `0..vocab_size`.
    fn vocab_size(&self) -> usize;

    /// The byte string token `id` decodes to.
    ///
    /// Returns an empty slice for an out-of-range `id`.
    fn token_bytes(&self, id: u32) -> &[u8];
}

/// A deterministic, table-driven [`ConstrainedVocabulary`] built from an explicit
/// list of byte strings.
///
/// This is the fixture the module's own tests are written against, and it is
/// public for the same reason `replug::ReplugStaticLanguageModel` is:
/// every claim made here about masking, liveness and reachability is checked
/// against a vocabulary whose contents are *written down*, with no tokenizer, no
/// model weights and no randomness anywhere in the loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticVocabulary {
    tokens: Vec<Vec<u8>>,
}

impl StaticVocabulary {
    /// Build a vocabulary from raw byte strings, in id order.
    ///
    /// # Errors
    ///
    /// Returns [`ConstrainedDecodingError::EmptyVocabulary`] when `tokens` is
    /// empty. Empty *tokens* are permitted here — the decoder rejects them unless
    /// they are nominated as end-of-sequence — see
    /// [`ConstrainedDecodingError::EmptyToken`].
    pub fn new(tokens: Vec<Vec<u8>>) -> ConstrainedDecodingResult<Self> {
        if tokens.is_empty() {
            return Err(ConstrainedDecodingError::EmptyVocabulary);
        }
        Ok(Self { tokens })
    }

    /// Build a vocabulary from string surfaces, in id order.
    ///
    /// # Errors
    ///
    /// As [`StaticVocabulary::new`].
    pub fn from_strs(tokens: &[&str]) -> ConstrainedDecodingResult<Self> {
        Self::new(tokens.iter().map(|text| text.as_bytes().to_vec()).collect())
    }

    /// The id of the token whose bytes are exactly `bytes`, if any. The *lowest*
    /// such id, if the table holds duplicates.
    #[must_use]
    pub fn token_id(&self, bytes: &[u8]) -> Option<ConstrainedTokenId> {
        self.tokens.iter().position(|token| token == bytes).map(
            #[allow(clippy::cast_possible_truncation)] // Bounded by `vocab_size <= u32::MAX`.
            |index| index as ConstrainedTokenId,
        )
    }

    /// The token table.
    #[must_use]
    pub fn tokens(&self) -> &[Vec<u8>] {
        &self.tokens
    }
}

impl ConstrainedVocabulary for StaticVocabulary {
    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn token_bytes(&self, id: u32) -> &[u8] {
        self.tokens.get(id as usize).map_or(&[], Vec::as_slice)
    }
}

// ── JSON Schema options ──────────────────────────────────────────────────────

/// How the compiled object automaton orders an object's keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PropertyOrder {
    /// Keys appear in the order `serde_json` iterates the schema's `properties`
    /// map, which (without its `preserve_order` feature, which this crate does not
    /// enable) is lexicographic by key name.
    ///
    /// The compiled language is a *subset* of the schema's — every string it
    /// accepts satisfies the schema, but a schema-valid document whose keys come
    /// in a different order is not accepted. That is exactly what is wanted at
    /// decode time: the mask is what chooses the order, and there is no reason to
    /// let the model pick one.
    #[default]
    Declaration,
    /// Every ordering of the keys is accepted.
    ///
    /// The compiled language is then exactly the schema's (restricted to canonical,
    /// whitespace-free JSON). Rejected with
    /// [`ConstrainedDecodingError::UnsupportedKeyword`] beyond
    /// [`MAX_PERMUTED_PROPERTIES`] properties, because the branch count grows
    /// factorially.
    AnyPermutation,
}

/// Knobs of the JSON-Schema compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonSchemaOptions {
    /// How object keys are ordered. See [`PropertyOrder`].
    pub property_order: PropertyOrder,
    /// How deeply sub-schemas may nest before compilation is refused.
    pub max_depth: u32,
}

impl Default for JsonSchemaOptions {
    fn default() -> Self {
        Self {
            property_order: PropertyOrder::Declaration,
            max_depth: 32,
        }
    }
}

// ── Constraint ───────────────────────────────────────────────────────────────

/// What the caller wants the decoder to be unable to violate.
///
/// The leaves ([`Constraint::Regex`], [`Constraint::Ast`], [`Constraint::Literal`],
/// [`Constraint::JsonSchema`], [`Constraint::MaxBytes`]) all lower to a
/// [`RegexAst`] and then to one automaton apiece. The two combinators do not: a
/// regular language is closed under intersection and union, but a *syntax tree*
/// is not, so [`Constraint::AllOf`] and [`Constraint::AnyOf`] are realised as
/// products of the compiled automata.
///
/// [`Constraint::AllOf`] is what makes dead-state elimination load-bearing rather
/// than decorative — see the [module documentation](crate::constrained_decoding).
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// A pattern in this module's regex dialect.
    Regex(String),
    /// An already-parsed syntax tree.
    Ast(RegexAst),
    /// Exactly this byte string, and nothing else.
    Literal(Vec<u8>),
    /// Canonical JSON conforming to a schema.
    JsonSchema {
        /// The schema, as a `serde_json` value.
        schema: Value,
        /// Compiler options.
        options: JsonSchemaOptions,
    },
    /// At most this many bytes. Intended to be intersected with something else.
    MaxBytes(u32),
    /// Every child must accept — the intersection of their languages.
    AllOf(Vec<Constraint>),
    /// At least one child must accept — the union of their languages.
    AnyOf(Vec<Constraint>),
}

impl Constraint {
    /// A pattern in this module's regex dialect.
    #[must_use]
    pub fn regex(pattern: &str) -> Self {
        Self::Regex(pattern.to_string())
    }

    /// Exactly the UTF-8 bytes of `text`.
    #[must_use]
    pub fn literal(text: &str) -> Self {
        Self::Literal(text.as_bytes().to_vec())
    }

    /// Canonical JSON conforming to `schema`, with default options.
    #[must_use]
    pub fn json_schema(schema: Value) -> Self {
        Self::JsonSchema {
            schema,
            options: JsonSchemaOptions::default(),
        }
    }

    /// Canonical JSON conforming to `schema`, with explicit options.
    #[must_use]
    pub fn json_schema_with(schema: Value, options: JsonSchemaOptions) -> Self {
        Self::JsonSchema { schema, options }
    }

    /// At most `limit` bytes of output.
    #[must_use]
    pub fn max_bytes(limit: u32) -> Self {
        Self::MaxBytes(limit)
    }

    /// The intersection of every child's language.
    #[must_use]
    pub fn all_of(children: impl IntoIterator<Item = Self>) -> Self {
        Self::AllOf(children.into_iter().collect())
    }

    /// The union of every child's language.
    #[must_use]
    pub fn any_of(children: impl IntoIterator<Item = Self>) -> Self {
        Self::AnyOf(children.into_iter().collect())
    }
}

// ── ConstrainedDecoderConfig ─────────────────────────────────────────────────

/// Tunables of a [`crate::constrained_decoding::ConstrainedDecoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstrainedDecoderConfig {
    /// The token that terminates generation.
    ///
    /// It is permitted **exactly at the accepting states** and nowhere else,
    /// which is what turns "the automaton is in an accepting state" into "the
    /// decoder is allowed to stop". Its byte string is ignored: end-of-sequence
    /// consumes no bytes.
    ///
    /// With `None`, the decoder never terminates itself and the caller must
    /// consult `is_accepting` to know when stopping is legal.
    pub eos_token_id: Option<ConstrainedTokenId>,
    /// The most states a compiled automaton may have.
    pub max_dfa_states: usize,
    /// The most `states x vocab_size` cells the vocabulary index may need.
    pub max_index_cells: usize,
}

impl Default for ConstrainedDecoderConfig {
    fn default() -> Self {
        Self {
            eos_token_id: None,
            max_dfa_states: 100_000,
            max_index_cells: 64 << 20,
        }
    }
}

impl ConstrainedDecoderConfig {
    /// This configuration with `eos_token_id` set.
    #[must_use]
    pub const fn with_eos(mut self, eos_token_id: ConstrainedTokenId) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }
}
