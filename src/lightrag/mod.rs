//! `LightRAG` (Guo et al. 2024, "`LightRAG`: Simple and Fast Retrieval-Augmented
//! Generation") — dual-level graph+vector retrieval over an incrementally
//! deduplicated knowledge graph.
//!
//! `LightRAG` is deliberately **not community-based** — this is what sets it
//! apart from its graph-RAG neighbour in this crate:
//!
//! * [`crate::drift_search`] (`GraphRAG` DRIFT) builds a **community
//!   hierarchy** ahead of time (via external clustering) and searches it
//!   **coarse-to-fine**: a global pass over community summaries picks the
//!   most relevant communities, follow-up sub-queries are spun out from their
//!   salient terms, and a local pass then drills into the passages those
//!   communities own. There is exactly one traversal direction (coarse →
//!   fine) and every result is mediated by community membership.
//!
//! `LightRAG`, by contrast, never clusters the graph at all. It extracts a
//! flat entity/relation graph and answers every query along **two
//! independent, parallel paths**, chosen by splitting the query itself into
//! [`LightRagDualKeywords`]:
//!
//! * **Low-level keys** — specific entities and concrete nouns — drive
//!   [`LightRagMode::Local`] retrieval: match keys against entity
//!   descriptions (vector + lexical), then expand each match to its 1-hop
//!   neighborhood (incident relations and neighbor entities).
//! * **High-level keys** — abstract themes and concepts — drive
//!   [`LightRagMode::Global`] retrieval: match keys against *relation*-level
//!   keywords, then gather those relations' connected entities.
//! * [`LightRagMode::Hybrid`] runs both paths and **fuses** them —
//!   deduplicating by key and summing per-path scores, so an entity or
//!   relation surfaced by both the low-level and high-level path outranks
//!   one found by only one.
//!
//! There is no community layer, no hierarchy, and no coarse-to-fine
//! direction: Local and Global are two independent lenses over the *same*
//! flat graph, fused on demand.
//!
//! # Incremental indexing
//!
//! [`LightRagIndex`] builds this graph incrementally, one [`LightRagChunk`]
//! at a time. Entities and relations are dedup-keyed by canonicalized name
//! (relations by their *unordered* endpoint pair, so a relationship
//! re-extracted the other way round still merges): re-inserting a chunk that
//! mentions an already-known entity or relation **merges** into the existing
//! record — accumulating its description and the new source chunk id, and
//! incrementing its occurrence count — rather than duplicating it. Every
//! entity and relation description is also embedded with a deterministic
//! FNV-1a lexical pseudo-embedding, recomputed on every merge.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "lightrag")] {
//! use oxirag::lightrag::{LightRagChunk, LightRagConfig, LightRagEngine, LightRagMode};
//!
//! let mut engine = LightRagEngine::new(LightRagConfig::default());
//! engine
//!     .insert_chunk(LightRagChunk::new(
//!         "c1",
//!         "Marie Curie discovered radium. Marie Curie worked at the University of Paris.",
//!     ))
//!     .unwrap();
//! engine
//!     .insert_chunk(LightRagChunk::new(
//!         "c2",
//!         "The University of Paris is a research institution in France.",
//!     ))
//!     .unwrap();
//!
//! // Local: the low-level key "Marie Curie" pulls in her 1-hop neighborhood.
//! let local = engine
//!     .query("What did Marie Curie discover?", LightRagMode::Local)
//!     .unwrap();
//! assert!(local.entities.iter().any(|e| e.name == "Marie Curie"));
//!
//! // Hybrid: fuses Local and Global, deduplicated by entity/relation key.
//! let hybrid = engine
//!     .query("Marie Curie research institution", LightRagMode::Hybrid)
//!     .unwrap();
//! assert!(!hybrid.is_empty());
//! # }
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::{LightRagEngine, LightRagIndex};
pub use types::{
    LightRagChunk, LightRagConfig, LightRagDualKeywords, LightRagEntity, LightRagEntityKind,
    LightRagError, LightRagIndexStats, LightRagMode, LightRagRelation, LightRagResult,
};
