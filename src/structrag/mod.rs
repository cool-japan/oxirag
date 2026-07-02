//! `StructRAG` (Li et al. 2024, "`StructRAG`: Boosting Knowledge Intensive
//! Reasoning of LLMs via Inference-time Hybrid Information Structurization").
//!
//! # The idea
//!
//! Different tasks are best reasoned over in different *shapes* of
//! knowledge. A question comparing several items by a numeric attribute is
//! easiest to answer from a table; a question about how entities relate is
//! easiest to answer from a graph; a question about a taxonomy is easiest to
//! answer from a tree; a request to enumerate options is easiest to answer
//! from a keyed list; a "how do I..." question is easiest to answer from an
//! ordered procedure. Plain retrieved passages are none of these — they are
//! unstructured prose. `StructRAG` closes that gap at inference time, in two
//! steps before any answer is produced:
//!
//! 1. **Route.** [`StructRagRouter`] inspects the *task*, not the passages,
//!    and infers which [`StructRagStructureKind`] — [`StructRagTable`],
//!    [`StructRagGraph`], [`StructRagTree`], [`StructRagCatalogue`], or
//!    [`StructRagAlgorithm`] — the task is best reasoned over in, returning a
//!    [`StructRagRoutingDecision`] with a confidence and a rationale.
//! 2. **Restructure.** [`StructRagRestructurer`] then re-shapes the
//!    *retrieved* [`StructRagPassage`]s into a concrete, populated
//!    [`StructRagKnowledgeStructure`] of exactly that kind: rows/columns
//!    extracted for a table, entities/edges extracted for a graph,
//!    parent/child links built for a tree, keyed items assembled for a
//!    catalogue, or an ordered step list assembled for a procedure.
//!
//! Only then does [`StructRagReasoner`] read an answer *from the structure*
//! — scanning a table's numeric columns, traversing a graph, walking a
//! tree's ancestor/descendant links, looking an entry up in a catalogue, or
//! walking a procedure's steps in order — rather than re-reading raw prose.
//! [`StructRagEngine::run`] drives all three steps and returns a
//! [`StructRagResult`] carrying the answer *and* the intermediate structure,
//! so the reasoning remains inspectable rather than opaque.
//!
//! # Differentiator: vs `structured_extraction`
//!
//! [`crate::structured_extraction`] and `structrag` can look superficially
//! similar — both turn prose into something structured — but they solve
//! different problems:
//!
//! * [`crate::structured_extraction::SchemaExtractor`] requires the caller
//!   to supply a **fixed** [`crate::structured_extraction::ExtractionSchema`]
//!   up front (field names, types, keywords) and extracts one record
//!   conforming to *that exact schema* — the shape of the output is decided
//!   before the extractor ever sees a query. It answers "pull these named
//!   fields out of this text."
//! * `structrag` is not given a schema at all. Its first move is to *infer
//!   the shape itself* from the task (the router step above) — table vs.
//!   graph vs. tree vs. catalogue vs. algorithm — and only then builds a
//!   structure of that inferred shape from whatever passages were
//!   retrieved. The schema is an **output** of routing, not an input to
//!   extraction. It answers "what shape of knowledge does this task need,
//!   and what does that shape look like for these passages?"
//!
//! The router-then-restructure meta-step is exactly what makes `structrag`
//! a *retrieval-augmented reasoning* technique rather than an extraction
//! utility: the same five passages restructure differently depending on
//! what is asked of them.
//!
//! # Determinism
//!
//! Every step — cue-phrase routing, passage restructuring, and structure
//! reading — is pure, deterministic, keyword/pattern-heuristic logic. There
//! is no LLM call, no embedding model, and no randomness (no `rand`, no
//! `ndarray`): identical inputs always produce identical
//! [`StructRagResult`]s.
//!
//! # Example
//!
//! ```rust
//! use oxirag::structrag::{StructRagConfig, StructRagEngine, StructRagPassage, StructRagStructureKind};
//!
//! let passages = vec![
//!     StructRagPassage::new(
//!         "p1",
//!         "Zenith costs $19.99 and has a rating of 4.5 stars.",
//!     ),
//!     StructRagPassage::new(
//!         "p2",
//!         "Nimbus costs $9.99 and has a rating of 4.0 stars.",
//!     ),
//! ];
//!
//! let engine = StructRagEngine::new(StructRagConfig::default());
//! let result = engine
//!     .run("Which product is cheapest?", &passages)
//!     .expect("run should succeed");
//!
//! // The comparison cue ("cheapest") routes this to a table...
//! assert_eq!(result.routing.kind, StructRagStructureKind::Table);
//! assert!(result.routing.confidence > 0.0);
//! // ...restructured into a real, populated table...
//! assert_eq!(result.structure.kind(), StructRagStructureKind::Table);
//! assert!(!result.structure.is_empty());
//! // ...that the reasoner then reads to find the cheapest product.
//! assert!(result.answer.contains("Nimbus"));
//! ```

pub mod engine;
mod lexical;
pub mod reason;
pub mod restructure;
pub mod router;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::StructRagEngine;
pub use reason::StructRagReasoner;
pub use restructure::StructRagRestructurer;
pub use router::StructRagRouter;
pub use types::{
    StructRagAlgorithm, StructRagCatalogue, StructRagCatalogueItem, StructRagConfig,
    StructRagError, StructRagGraph, StructRagGraphEdge, StructRagGraphNode,
    StructRagKnowledgeStructure, StructRagPassage, StructRagResult, StructRagRoutingDecision,
    StructRagStep, StructRagStructureKind, StructRagTable, StructRagTableRow, StructRagTree,
    StructRagTreeNode,
};
