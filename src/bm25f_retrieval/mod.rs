//! BM25F: the classical field-weighted BM25 ranking function (Robertson &
//! Zaragoza, 2004, "Simple BM25 Extension to Multiple Weighted Fields") for
//! documents composed of multiple named fields.
//!
//! Real documents are rarely one flat blob of text: a title, a body, and a
//! set of tags typically carry very different amounts of relevance signal
//! per occurrence, and behave differently as their length grows. BM25F
//! models this directly — every [`Bm25fField`] on a [`Bm25fDocument`] gets
//! its own importance weight `w_f` *and* its own length-normalisation
//! parameter `b_f` (via [`Bm25fFieldWeight`]), instead of forcing every
//! field through one global `b`.
//!
//! # The algorithm
//!
//! Indexing ([`Bm25fIndex::build`]) computes, for every term `t` and
//! document `d`, a single **combined pseudo-frequency** across all of `d`'s
//! fields:
//!
//! ```text
//! tf~(t, d) = Σ_f  w_f * tf(t, field_f) / (1 - b_f + b_f * len(field_f) / avg_len_f)
//! ```
//!
//! where the sum ranges over `d`'s fields, `tf(t, field_f)` is the raw term
//! frequency of `t` within that one field, `len(field_f)` is that field's
//! token count, and `avg_len_f` is the corpus-wide average length of that
//! *named* field (e.g. the average title length, tracked completely
//! separately from the average body length). Each field is length-normalised
//! against its own average, weighted by its own `w_f`, and only *then* added
//! into the combined total.
//!
//! Scoring ([`Bm25fIndex::search`]) applies the ordinary BM25 saturation
//! nonlinearity to that already-combined pseudo-frequency, once per unique
//! query term:
//!
//! ```text
//! score(q, d) = Σ_{t in q}  IDF(t) * tf~(t, d) * (k1 + 1) / (k1 + tf~(t, d))
//! ```
//!
//! with `IDF(t)` computed the usual BM25 way from document frequency over
//! the whole corpus (a document "contains" `t` if *any* of its fields does,
//! irrespective of that field's weight). Combining fields **before** the
//! saturation nonlinearity — rather than scoring each field with ordinary
//! BM25 and summing the per-field *scores* afterwards — is the defining
//! trick of BM25F: it lets a term that is moderately frequent across two
//! fields outrank a term that is only very frequent in one, without letting
//! any single field's saturation cap the others' contribution in isolation.
//!
//! # Not to be confused with...
//!
//! - `hybrid_search`'s `BM25Encoder` is **single-field** classical BM25: it
//!   encodes one flat `&str` per document into a sparse vector using one
//!   global `k1`/`b` pair (`hybrid_search::BM25Params`). There is no concept
//!   of named fields, per-field weights, or per-field length normalisation
//!   — every document is one undifferentiated bag of words.
//! - `sparse_retrieval` is SPLADE-style **learned** sparse retrieval: term
//!   weights are a log-saturated `tf * idf` transform, and the encoder
//!   *expands* the term set with discounted co-occurring terms that may not
//!   literally appear in the text. It is not classical BM25 at all — there
//!   is no `k1`/`b` saturation formula, and no notion of document fields.
//!
//! `bm25f_retrieval`, by contrast, is **classical, unlearned, exact BM25F**:
//! deterministic term statistics, explicit per-field weights and `b`
//! parameters, and the literal Robertson–Zaragoza combine-then-saturate
//! formula — nothing here is learned, expanded, or approximated.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|-----------------|
//! | [`Bm25fField`] / [`Bm25fDocument`] | A named field of text, and a document made of several |
//! | [`Bm25fFieldWeight`] | Per-field `w_f` / `b_f` tuning |
//! | [`Bm25fConfig`] | `k1`, IDF smoothing, and the per-field-name weight table |
//! | [`Bm25fIndex`] | Builds per-field statistics over a corpus and scores queries |
//! | [`Bm25fHit`] | One scored, ranked search result |
//!
//! # Quick start
//!
//! ```
//! use oxirag::bm25f_retrieval::{Bm25fConfig, Bm25fDocument, Bm25fFieldWeight, Bm25fIndex};
//!
//! // Titles count for more per occurrence and are never length-penalised;
//! // bodies count less per occurrence and are penalised for length.
//! let config = Bm25fConfig::new()
//!     .with_field_weight("title", Bm25fFieldWeight::new(2.0, 0.0))
//!     .with_field_weight("body", Bm25fFieldWeight::new(1.0, 0.75));
//!
//! let corpus = vec![
//!     Bm25fDocument::new("d1")
//!         .with_field("title", "Rust programming")
//!         .with_field("body", "Rust is a systems programming language"),
//!     Bm25fDocument::new("d2")
//!         .with_field("title", "Cooking recipes")
//!         .with_field("body", "A collection of recipes for cooking"),
//! ];
//!
//! let mut index = Bm25fIndex::new(config);
//! index.build(&corpus).expect("build should succeed");
//!
//! let hits = index.search("programming", 5).expect("search should succeed");
//! assert_eq!(hits[0].doc_id, "d1");
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::Bm25fIndex;
pub use types::{
    Bm25fConfig, Bm25fDocument, Bm25fError, Bm25fField, Bm25fFieldWeight, Bm25fHit, Bm25fResult,
};
