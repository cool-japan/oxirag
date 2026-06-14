//! Cross-lingual retrieval: query in one language, documents in another.
//!
//! Bridges a language gap with a *caller-supplied* bilingual lexicon plus
//! diacritic normalization — no machine-translation model and no learned
//! embeddings. The pipeline is a pure-Rust, deterministic heuristic:
//!
//! 1. **Normalize** — lowercase and strip common Latin diacritics so that
//!    accented and unaccented spellings collapse to one shared surface form
//!    (`café` → `cafe`, `München` → `munchen`).
//! 2. **Expand** — translate each query token through the [`BilingualLexicon`]
//!    and append the (normalized) translations, carrying the query across the
//!    language boundary as a cross-lingual form of query expansion.
//! 3. **Match** — embed the expanded token set with a deterministic FNV-1a
//!    pseudo-embedding and rank documents by cosine similarity.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`BilingualLexicon`] | Caller-supplied source → target token translations |
//! | [`CrossLingualConfig`] | Embedding dimension and expansion toggle |
//! | [`CrossLingualRetriever`] | Normalizes, embeds, expands, and retrieves |
//! | [`CrossLingualHit`] | A matched document with its similarity score |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "cross-lingual")] {
//! use oxirag::cross_lingual::{BilingualLexicon, CrossLingualConfig, CrossLingualRetriever};
//! use oxirag::types::Document;
//!
//! let mut lexicon = BilingualLexicon::new();
//! lexicon.add("chien", "dog"); // French query token → English document token
//!
//! let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), lexicon);
//! retriever.build(&[Document::new("The dog runs fast.")]);
//!
//! // A French query retrieves the English document.
//! let hits = retriever.search("chien", 5).unwrap();
//! assert!(!hits.is_empty());
//! # }
//! ```

pub mod lexicon;
pub mod retriever;
#[cfg(test)]
mod tests;
pub mod types;

pub use lexicon::BilingualLexicon;
pub use retriever::CrossLingualRetriever;
pub use types::{CrossLingualConfig, CrossLingualError, CrossLingualHit};
