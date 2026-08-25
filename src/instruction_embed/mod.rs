//! Instruction-conditioned embeddings (INSTRUCTOR, Su et al. 2023, "One
//! Embedder, Any Task: Instruction-Finetuned Text Embeddings"; TART, Asai et
//! al. 2023, "Task-Aware Retrieval with Instructions").
//!
//! A single encoder produces *different* embeddings for the *same* piece of
//! text depending on a natural-language task instruction it is paired with —
//! e.g. `"Represent the science document for retrieval:"` versus
//! `"Represent the question for retrieving supporting documents:"`. Because
//! the instruction genuinely re-weights and translates the underlying text
//! representation (not a label bolted on afterwards, and not a second vector
//! concatenated alongside it), indexing or querying the *same* corpus under a
//! *different* instruction changes which documents rank highest — one
//! instruction-conditioned encoder serves every task, instead of one
//! fine-tuned embedding model per task.
//!
//! # How this differs from its neighbours
//!
//! - [`self_query`](crate::self_query) *parses* a query's surface text into a
//!   structured metadata filter plus a residual semantic string — the
//!   embedding of that residual string is produced exactly as it would be
//!   without `self_query` involved at all. Nothing about the embedding
//!   *vector itself* is conditioned on anything; only which documents are
//!   *eligible* to be scored changes (via the extracted filter).
//! - [`matryoshka`](crate::matryoshka) truncates an embedding's *dimensions*
//!   to trade fidelity for speed: a coarse-to-fine prefix hierarchy of the
//!   *same* underlying vector, for the *same* task. The ranking a
//!   full-length Matryoshka embedding produces does not depend on any notion
//!   of task intent — only on how many of its (task-agnostic) dimensions are
//!   kept.
//! - `instruction_embed`, by contrast, conditions the embedding *itself* on a
//!   task instruction: the same text, embedded under two different
//!   instructions, is genuinely two different vectors. The *same* corpus,
//!   indexed once, can then be queried for retrieval, classification-flavoured
//!   similarity, or clustering — each intent supplied purely as a different
//!   instruction string, with no re-training and no separate model per task.
//!
//! # Algorithm
//!
//! [`InstructionEmbedder::embed`] combines two deterministic, hash-based
//! signals (full detail in the [`engine`] module documentation):
//!
//! 1. A **base embedding**: an FNV-1a token-bucket histogram of the input
//!    text (the same technique [`matryoshka`](crate::matryoshka) uses for its
//!    task-agnostic embedding).
//! 2. An **instruction-derived modulation**: a per-dimension *gate* (a
//!    splitmix64 pseudo-random stream seeded by the instruction's own hash,
//!    deviating from `1.0` by up to
//!    [`InstructionEmbedConfig::instruction_influence`]) that re-weights the
//!    base embedding, plus an additive *task-shift* vector (the instruction
//!    text's own FNV-1a token histogram, L2-normalised and scaled by the
//!    same influence) that translates it.
//!
//! The two signals combine as `combined[i] = base[i] * gate[i] + shift[i]`,
//! then L2-normalise. At `instruction_influence == 0.0` the gate is uniformly
//! `1.0` and the shift is the zero vector, so the result collapses exactly to
//! the instruction-agnostic base embedding — a sanity boundary confirming the
//! instruction is the *only* source of the conditioning effect.
//!
//! [`InstructionRegistry`] lets named instructions (e.g. `"science_doc"`,
//! `"science_query"`) be registered once and looked up by name.
//! [`InstructionIndex`] indexes a corpus under one *document* instruction and
//! answers [`InstructionIndex::search`] queries embedded under a *query*
//! instruction, by cosine similarity.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`InstructionEmbedConfig`] | Dimension, instruction influence, normalisation, default top-k |
//! | [`TaskInstruction`] | A named natural-language task instruction |
//! | [`InstructionEmbedder`] | Text + instruction → conditioned [`InstructionEmbedding`] |
//! | [`InstructionRegistry`] | Named instruction registration and lookup |
//! | [`InstructionIndex`] | Corpus indexing + instruction-conditioned retrieval |
//! | [`InstructionEmbedError`] / [`InstructionEmbedResult`] | Error type and result alias |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "instruction-embed")] {
//! use oxirag::instruction_embed::{InstructionEmbedConfig, InstructionIndex, TaskInstruction};
//! use oxirag::types::Document;
//!
//! let doc_instruction =
//!     TaskInstruction::new("science_doc", "Represent the science document for retrieval:");
//! let query_instruction = TaskInstruction::new(
//!     "science_query",
//!     "Represent the question for retrieving supporting science documents:",
//! );
//!
//! let corpus = vec![
//!     Document::new("Photosynthesis converts light into chemical energy.").with_id("bio"),
//!     Document::new("The stock market closed higher on Friday.").with_id("finance"),
//! ];
//!
//! let mut index = InstructionIndex::new(InstructionEmbedConfig::default());
//! index.index(&corpus, &doc_instruction).unwrap();
//! let hits = index
//!     .search("energy conversion in plants", &query_instruction, 1)
//!     .unwrap();
//! assert_eq!(hits[0].0.as_str(), "bio");
//! # }
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::{InstructionEmbedder, InstructionIndex, InstructionRegistry};
pub use types::{
    InstructionEmbedConfig, InstructionEmbedError, InstructionEmbedResult, InstructionEmbedding,
    TaskInstruction,
};
