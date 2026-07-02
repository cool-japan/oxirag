//! Instruction-conditioned embedding, instruction registry, and
//! instruction-aware document index.
//!
//! # Encoding
//!
//! [`InstructionEmbedder::embed`] combines two deterministic, hash-based
//! signals into one vector:
//!
//! 1. A **base embedding**: an FNV-1a token-bucket histogram of the input
//!    text — the same technique [`matryoshka`](crate::matryoshka) uses for
//!    its (task-agnostic) embedding.
//! 2. An **instruction-derived modulation** (an internal, private
//!    `instruction_modulation` helper): a per-dimension multiplicative *gate*
//!    (a splitmix64 pseudo-random stream seeded from the instruction's own
//!    hash) that re-weights the base embedding, plus an additive *task-shift*
//!    vector (the instruction text's own FNV-1a token histogram) that
//!    translates it.
//!
//! The two are combined as `combined[i] = base[i] * gate[i] + shift[i]`, then
//! L2-normalised. Because the shift vector is built with the *same* hashing
//! routine as ordinary text, an instruction whose vocabulary overlaps a
//! document's or query's own vocabulary genuinely pulls the combined
//! embedding toward it — the mechanism that lets a single corpus rank
//! differently under different task instructions.
//!
//! Everything is deterministic: identical `(text, instruction, config)`
//! always encodes to an identical vector, with no randomness and no
//! machine-learning dependencies.

use std::collections::BTreeMap;

use super::types::{
    InstructionEmbedConfig, InstructionEmbedError, InstructionEmbedResult, InstructionEmbedding,
    TaskInstruction,
};
use crate::types::{Document, DocumentId};

// ── Domain-separation seeds ──────────────────────────────────────────────────
//
// Both constants are the well-known 64-bit mixing constants used inside
// splitmix64 itself; reused here purely as fixed, arbitrary seeds (nothing
// cryptographic is implied). `TEXT_SEED` hashes ordinary text — both corpus
// text and, deliberately, the instruction text used to build the task-shift
// vector, so shared vocabulary between an instruction and a text genuinely
// aligns their embeddings. `GATE_SEED` is a different seed used only to seed
// the per-dimension gate stream, so the gate never accidentally coincides
// with the base embedding's hash space.

/// FNV-1a domain-separation seed for the base text/document/query embedding
/// space (also reused, deliberately, for the instruction task-shift vector).
const TEXT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// FNV-1a domain-separation seed for the per-dimension instruction gate
/// stream.
const GATE_SEED: u64 = 0xBF58_476D_1CE4_E5B9;

// ── Tokenisation ──────────────────────────────────────────────────────────────

/// Split `text` into lowercase alphanumeric tokens of length `>= 2`.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── Hashing ───────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of `bytes`, seeded with `seed`.
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

/// One splitmix64 mixing step (Vigna, public domain): advances `state` and
/// returns a pseudo-random `u64`.
///
/// Used to derive the instruction's per-dimension gate stream deterministically
/// — the same `state` always produces the same output, and successive calls
/// walk a long, well-mixed pseudo-random sequence.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ── Base embedding ────────────────────────────────────────────────────────────

/// Deterministic instruction-agnostic bag-of-tokens embedding of `text`: an
/// FNV-1a token-bucket histogram of `dim` buckets. Unnormalised; callers
/// normalise (or not) as needed.
///
/// A `dim` of `0` returns an empty vector.
fn base_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut buckets = vec![0.0f32; dim];
    if dim == 0 {
        return buckets;
    }
    for token in tokenize(text) {
        let hash = fnv1a(token.as_bytes(), TEXT_SEED);
        #[allow(clippy::cast_possible_truncation)]
        let idx = (hash as usize) % dim;
        buckets[idx] += 1.0;
    }
    buckets
}

/// L2-normalise `vector` in place. A zero (or near-zero) vector is left
/// unchanged rather than divided by (approximately) zero.
fn l2_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

// ── Instruction-derived modulation ───────────────────────────────────────────

/// Derive the instruction-conditioned modulation applied on top of a base
/// text embedding: a per-dimension multiplicative *gate* plus an additive
/// *task-shift* vector, both deterministic functions of `instruction_text`
/// and scaled by `influence`. Both outputs have length `dim`.
///
/// - The **gate** is a splitmix64 pseudo-random stream seeded from the
///   instruction text's own FNV-1a hash (domain-separated from the base text
///   embedding via [`GATE_SEED`]), walked one step per dimension so every
///   dimension gets an independent, reproducible deviation from `1.0` within
///   `[-gate_strength, gate_strength)`, where `gate_strength =
///   influence.clamp(0.0, 1.0)`. This is the *re-weighting* half of the
///   conditioning: it can amplify, dampen, or flip the sign of any given
///   dimension of the base embedding, differently per instruction. Its
///   deviation *saturates* at `influence == 1.0` — deliberately, so that
///   arbitrarily large `influence` values cannot let this per-dimension
///   noise term outgrow and swamp the semantically-grounded shift term
///   below.
/// - The **task-shift** vector is [`base_embed`] applied to the instruction
///   text itself — the *same* hash space used for ordinary text — then
///   L2-normalised and scaled by the *full, unclamped* `influence`. Because
///   it shares that hash space, an instruction whose vocabulary overlaps a
///   document's or query's own vocabulary genuinely pulls the combined
///   embedding toward it, with a magnitude that grows without bound as
///   `influence` grows. This is the *translation* half of the conditioning,
///   and — thanks to the gate's saturation above — the half that dominates
///   whenever a caller asks for strong conditioning.
///
/// At `influence == 0.0` the gate is uniformly `1.0` and the shift is the
/// zero vector for *any* `instruction_text` — the instruction-agnostic
/// boundary that [`InstructionEmbedder::embed`] collapses to.
fn instruction_modulation(
    instruction_text: &str,
    dim: usize,
    influence: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut gate = vec![1.0f32; dim];
    if dim == 0 {
        return (gate, Vec::new());
    }

    let normalized_instruction = instruction_text.trim().to_lowercase();
    let instruction_hash = fnv1a(normalized_instruction.as_bytes(), GATE_SEED);
    let gate_strength = influence.clamp(0.0, 1.0);

    for (i, slot) in gate.iter_mut().enumerate() {
        let dim_seed = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut state = instruction_hash ^ dim_seed;
        let bits = splitmix64_next(&mut state);
        #[allow(clippy::cast_possible_truncation)]
        let top24 = (bits >> 40) as u32;
        #[allow(clippy::cast_precision_loss)]
        let unit = top24 as f32 / 16_777_216.0; // 2^24; unit in [0.0, 1.0)
        let signed = unit * 2.0 - 1.0; // [-1.0, 1.0)
        *slot = 1.0 + gate_strength * signed;
    }

    let mut shift = base_embed(instruction_text, dim);
    l2_normalize(&mut shift);
    for value in &mut shift {
        *value *= influence;
    }

    (gate, shift)
}

// ── InstructionEmbedder ───────────────────────────────────────────────────────

/// Produces instruction-conditioned embeddings: the same input text yields a
/// different vector under a different [`TaskInstruction`].
///
/// See the [`engine`](crate::instruction_embed::engine) module documentation
/// for the encoding algorithm.
#[derive(Debug, Clone)]
pub struct InstructionEmbedder {
    /// Embedder configuration.
    config: InstructionEmbedConfig,
}

impl InstructionEmbedder {
    /// Create a new embedder from the given configuration.
    ///
    /// Construction never fails; an invalid (zero) configured
    /// [`InstructionEmbedConfig::dim`] is reported lazily, as
    /// [`InstructionEmbedError::InvalidDim`], the first time
    /// [`InstructionEmbedder::embed`] is called.
    #[must_use]
    pub fn new(config: InstructionEmbedConfig) -> Self {
        Self { config }
    }

    /// Borrow the embedder's configuration.
    #[must_use]
    pub fn config(&self) -> &InstructionEmbedConfig {
        &self.config
    }

    /// Produce an instruction-conditioned embedding of `text` under
    /// `instruction`.
    ///
    /// The result is a deterministic function of `(text, instruction.text,
    /// self.config)`: calling this again with the same arguments always
    /// returns bit-for-bit the same vector. Calling it with the *same* `text`
    /// but a *different* `instruction` returns a genuinely different vector
    /// whenever [`InstructionEmbedConfig::instruction_influence`] is
    /// non-zero, since the instruction re-weights (gate) and translates
    /// (shift) the underlying base embedding rather than being appended
    /// alongside it.
    ///
    /// # Errors
    ///
    /// - [`InstructionEmbedError::EmptyText`] when `text` is empty or
    ///   whitespace-only.
    /// - [`InstructionEmbedError::InvalidDim`] when
    ///   [`InstructionEmbedConfig::dim`] is `0`.
    pub fn embed(
        &self,
        text: &str,
        instruction: &TaskInstruction,
    ) -> InstructionEmbedResult<InstructionEmbedding> {
        if text.trim().is_empty() {
            return Err(InstructionEmbedError::EmptyText);
        }
        self.config.validate()?;
        let dim = self.config.dim;

        let base = base_embed(text, dim);
        let (gate, shift) =
            instruction_modulation(&instruction.text, dim, self.config.instruction_influence);

        let mut combined: Vec<f32> = base
            .iter()
            .zip(gate.iter())
            .zip(shift.iter())
            .map(|((base_i, gate_i), shift_i)| base_i * gate_i + shift_i)
            .collect();

        if self.config.normalize {
            l2_normalize(&mut combined);
        }

        Ok(InstructionEmbedding {
            vector: combined,
            instruction_name: instruction.name.clone(),
        })
    }

    /// Cosine similarity between two equal-length slices.
    ///
    /// Mismatched-length or empty inputs score `0.0`. When both inputs are
    /// L2-normalised (as every [`InstructionEmbedding::vector`] is when
    /// [`InstructionEmbedConfig::normalize`] is enabled) this is simply their
    /// dot product, clamped to `[-1.0, 1.0]` to absorb floating-point drift.
    #[must_use]
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        dot.clamp(-1.0, 1.0)
    }
}

// ── InstructionRegistry ───────────────────────────────────────────────────────

/// A registry of named [`TaskInstruction`]s.
///
/// Lets callers register instructions once (e.g. at start-up) and look them
/// up by name at embedding time, rather than re-constructing instruction text
/// ad hoc at every call site.
#[derive(Debug, Clone, Default)]
pub struct InstructionRegistry {
    entries: BTreeMap<String, TaskInstruction>,
}

impl InstructionRegistry {
    /// Create a new, empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `instruction` under its own [`TaskInstruction::name`].
    ///
    /// Returns the previously registered instruction of the same name, if
    /// any — re-registering a name replaces it.
    pub fn register(&mut self, instruction: TaskInstruction) -> Option<TaskInstruction> {
        self.entries.insert(instruction.name.clone(), instruction)
    }

    /// Look up a registered instruction by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TaskInstruction> {
        self.entries.get(name)
    }

    /// Look up a registered instruction by name.
    ///
    /// # Errors
    ///
    /// Returns [`InstructionEmbedError::UnknownInstruction`] when no
    /// instruction was ever registered under `name`.
    pub fn try_get(&self, name: &str) -> InstructionEmbedResult<&TaskInstruction> {
        self.entries
            .get(name)
            .ok_or_else(|| InstructionEmbedError::UnknownInstruction(name.to_string()))
    }

    /// Return `true` when an instruction is registered under `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Remove and return the instruction registered under `name`, if any.
    pub fn remove(&mut self, name: &str) -> Option<TaskInstruction> {
        self.entries.remove(name)
    }

    /// Number of registered instructions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no instructions are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Registered instruction names, in ascending lexical order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }
}

// ── InstructionIndex ──────────────────────────────────────────────────────────

/// An instruction-conditioned document index.
///
/// [`InstructionIndex::index`] embeds a corpus under a single *document*
/// instruction; [`InstructionIndex::search`] embeds a query under a
/// (possibly different) *query* instruction and ranks the corpus by cosine
/// similarity. Re-running [`InstructionIndex::index`] with a different
/// document instruction, or calling [`InstructionIndex::search`] with a
/// different query instruction, changes which documents rank highest for the
/// same underlying text — the defining property of instruction-conditioned
/// embeddings (INSTRUCTOR, Su et al. 2023; TART, Asai et al. 2023).
#[derive(Debug, Clone)]
pub struct InstructionIndex {
    /// Index configuration.
    config: InstructionEmbedConfig,
    /// Embedder used for both documents (at index time) and queries (at
    /// search time).
    embedder: InstructionEmbedder,
    /// Indexed `(id, embedding)` pairs, in insertion order.
    entries: Vec<(DocumentId, InstructionEmbedding)>,
    /// Name of the document instruction the current index was built with.
    doc_instruction: Option<String>,
    /// Whether [`InstructionIndex::index`] has ever populated this index.
    indexed: bool,
}

impl InstructionIndex {
    /// Create a new, empty index from the given configuration.
    #[must_use]
    pub fn new(config: InstructionEmbedConfig) -> Self {
        let embedder = InstructionEmbedder::new(config.clone());
        Self {
            config,
            embedder,
            entries: Vec::new(),
            doc_instruction: None,
            indexed: false,
        }
    }

    /// Borrow the index's configuration.
    #[must_use]
    pub fn config(&self) -> &InstructionEmbedConfig {
        &self.config
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the index holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return `true` once [`InstructionIndex::index`] has populated the
    /// index at least once.
    #[must_use]
    pub fn is_indexed(&self) -> bool {
        self.indexed
    }

    /// Name of the document instruction used to build the current index, or
    /// `None` before [`InstructionIndex::index`] has ever run.
    #[must_use]
    pub fn doc_instruction_name(&self) -> Option<&str> {
        self.doc_instruction.as_deref()
    }

    /// Encode `documents` under `doc_instruction` and (re)build the index.
    ///
    /// Any previously indexed content is discarded first, so repeated calls
    /// are idempotent for a fixed corpus and instruction, and calling this
    /// again with a *different* `doc_instruction` re-embeds the same corpus
    /// under a different task intent (changing subsequent
    /// [`InstructionIndex::search`] rankings).
    ///
    /// # Errors
    ///
    /// - [`InstructionEmbedError::EmptyCorpus`] when `documents` is empty.
    /// - [`InstructionEmbedError::InvalidDim`] when
    ///   [`InstructionEmbedConfig::dim`] is `0`.
    /// - [`InstructionEmbedError::EmptyText`] when a document's content is
    ///   empty or whitespace-only.
    pub fn index(
        &mut self,
        documents: &[Document],
        doc_instruction: &TaskInstruction,
    ) -> InstructionEmbedResult<()> {
        if documents.is_empty() {
            return Err(InstructionEmbedError::EmptyCorpus);
        }

        let mut entries = Vec::with_capacity(documents.len());
        for document in documents {
            let embedding = self.embedder.embed(&document.content, doc_instruction)?;
            entries.push((document.id.clone(), embedding));
        }

        self.entries = entries;
        self.doc_instruction = Some(doc_instruction.name.clone());
        self.indexed = true;
        Ok(())
    }

    /// Search for the `k` best-scoring documents for `query`, embedded under
    /// `query_instruction`.
    ///
    /// Documents are ranked by descending cosine similarity between the
    /// query's instruction-conditioned embedding and each document's
    /// (separately instruction-conditioned, at index time) embedding, ties
    /// broken by ascending document id for determinism.
    ///
    /// # Errors
    ///
    /// - [`InstructionEmbedError::NotIndexed`] when
    ///   [`InstructionIndex::index`] has not run.
    /// - [`InstructionEmbedError::EmptyCorpus`] when the index holds no
    ///   documents.
    /// - [`InstructionEmbedError::EmptyQuery`] when `query` is empty or
    ///   whitespace-only.
    pub fn search(
        &self,
        query: &str,
        query_instruction: &TaskInstruction,
        k: usize,
    ) -> InstructionEmbedResult<Vec<(DocumentId, f32)>> {
        if !self.indexed {
            return Err(InstructionEmbedError::NotIndexed);
        }
        if self.entries.is_empty() {
            return Err(InstructionEmbedError::EmptyCorpus);
        }
        if query.trim().is_empty() {
            return Err(InstructionEmbedError::EmptyQuery);
        }

        let query_embedding = self.embedder.embed(query, query_instruction)?;
        let mut hits: Vec<(DocumentId, f32)> = self
            .entries
            .iter()
            .map(|(id, embedding)| {
                (
                    id.clone(),
                    InstructionEmbedder::cosine(&query_embedding.vector, &embedding.vector),
                )
            })
            .collect();
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.as_str().cmp(b.0.as_str()))
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Search using [`InstructionEmbedConfig::default_top_k`] as `k`.
    ///
    /// # Errors
    ///
    /// See [`InstructionIndex::search`].
    pub fn search_default(
        &self,
        query: &str,
        query_instruction: &TaskInstruction,
    ) -> InstructionEmbedResult<Vec<(DocumentId, f32)>> {
        self.search(query, query_instruction, self.config.default_top_k)
    }

    /// Remove every indexed document and reset
    /// [`InstructionIndex::is_indexed`] to `false`.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.doc_instruction = None;
        self.indexed = false;
    }
}
