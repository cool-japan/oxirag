//! `MemoRAG` — memory-augmented retrieval via a corpus-wide gist (Qian et al. 2024).
//!
//! `MemoRAG` forms a single, compressed **global memory** of the whole corpus — a
//! short gist of the most salient sentences plus a ranked list of key terms — and
//! then uses that memory to *generate retrieval clues*. A clue is a surrogate
//! sub-query that splices the user's question together with gist key terms,
//! bridging the lexical gap between an under-specified query and the evidence
//! that answers it. The clues retrieve independently and their results are fused
//! (maximum score per document), so a document can be surfaced by whichever clue
//! best matches it.
//!
//! This is deliberately **distinct** from the agent/conversation memory modules
//! (`long_term_memory`, `memory_compression`): those recall past dialogue turns,
//! whereas the memory here is over the *corpus* and exists solely to drive
//! retrieval. It is also distinct from `summary_index`, which builds one summary
//! *per document*; `MemoRAG` builds one gist for the *entire corpus*.
//!
//! Everything is deterministic and model-free: a corpus-wide term-frequency
//! salience drives the gist, an FNV-1a lexical pseudo-embedding drives scoring,
//! and no randomness or machine-learning model is involved.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`MemoryGist`] | Corpus-wide summary + ranked key terms |
//! | [`MemoRagEngine`] | Builds memory, generates clues, fuses clue retrieval |
//! | [`MemoHit`] | A scored evidence document + its winning clue |
//! | [`MemoRagConfig`] | Gist size, clue count, key-term count, embedding dim |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "memorag")] {
//! use oxirag::memorag::{MemoRagConfig, MemoRagEngine};
//! use oxirag::types::Document;
//!
//! let corpus = vec![
//!     Document::new("Rust prevents data races. The borrow checker enforces ownership."),
//!     Document::new("Ownership in Rust governs memory. Lifetimes track references."),
//! ];
//!
//! let mut engine = MemoRagEngine::new(MemoRagConfig::default());
//! engine.build_memory(&corpus).unwrap();
//!
//! // The gist is a corpus-wide memory, not a per-document summary.
//! let gist = engine.gist().unwrap();
//! assert!(!gist.key_terms.is_empty());
//!
//! // Clues splice the query with gist key terms to bridge to evidence.
//! let clues = engine.generate_clues("how is memory managed?").unwrap();
//! assert!(clues.iter().all(|c| c.contains("how is memory managed?")));
//!
//! let hits = engine.retrieve("how is memory managed?", 2).unwrap();
//! // Each hit records the clue that won the document.
//! # }
//! ```

pub mod engine;
pub mod memory;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::MemoRagEngine;
pub use memory::{build_gist, embed, split_sentences, tokenize};
pub use types::{MemoHit, MemoRagConfig, MemoRagError, MemoryGist};
