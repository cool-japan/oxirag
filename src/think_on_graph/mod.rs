//! Think-on-Graph (`ToG`, Sun et al. 2023, "Think-on-Graph: Deep and
//! Responsible Reasoning of Large Language Model on Knowledge Graph").
//!
//! Think-on-Graph answers a query by *reasoning over a knowledge graph*: it
//! matches the query to topic entities, then runs a **bounded-width beam
//! search** over knowledge-graph paths, at each depth exploring and *pruning*
//! candidate relations and neighbour entities by their relevance to the query,
//! and checking whether the surviving paths already provide enough evidence to
//! answer — stopping early the moment they do.
//!
//! # How this differs from [`multi_hop`](crate::multi_hop)
//!
//! Both modules traverse a knowledge graph, but the traversal disciplines are
//! opposites:
//!
//! * [`multi_hop`](crate::multi_hop) does **exhaustive entity-chain
//!   edge-following**: at every hop it expands *all* reachable neighbours of the
//!   active entity set (bounded only by a coarse `entities_per_hop` cap), with
//!   **no beam, no per-relation scoring, and no relevance pruning**. Every edge
//!   is followed blindly; there is no notion of "keep the best paths".
//!
//! * `think_on_graph`, by contrast, does **LLM-guided bounded beam search**. It
//!   maintains at most `beam_width` (`W`) *paths* — not a flat entity frontier —
//!   and at each depth: (1) *scores* each candidate relation's relevance to the
//!   query and keeps only the top few (**relation pruning**), (2) *scores* the
//!   neighbour entities reached through the kept relations, (3) ranks *all*
//!   extended paths by cumulative relevance and keeps only the top-`W`
//!   (**beam pruning**), and (4) runs a **sufficiency check** — if the beam's
//!   triples already cover enough of the query, it stops early and synthesizes,
//!   instead of always running to the depth bound.
//!
//! In short: `multi_hop` is breadth-first and unpruned; `think_on_graph` is a
//! best-first beam that prunes by relevance and knows when to stop.
//!
//! # Determinism
//!
//! The engine invokes no model. All relevance scoring is a deterministic FNV-1a
//! lexical pseudo-embedding cosine blended with token overlap, and every score
//! tie is broken on a stable key, so a given `(query, graph, config)` always
//! yields byte-identical results.
//!
//! # Example
//!
//! ```
//! use oxirag::think_on_graph::{
//!     TogConfig, TogEngine, TogEntity, TogKnowledgeGraph, TogTriple,
//! };
//!
//! // A tiny two-hop knowledge graph: Alice -[mentor]-> Bob -[wrote]-> Book.
//! let graph = TogKnowledgeGraph::from_parts(
//!     vec![
//!         TogEntity::new("alice", "Alice"),
//!         TogEntity::new("bob", "Bob"),
//!         TogEntity::new("book", "Compilers"),
//!     ],
//!     vec![
//!         TogTriple::new("alice", "mentor", "bob"),
//!         TogTriple::new("bob", "wrote", "book"),
//!     ],
//! )
//! .expect("graph is well-formed");
//!
//! let engine = TogEngine::new(TogConfig::default());
//! let result = engine
//!     .run("What did the mentor of Alice write?", &graph)
//!     .expect("run should succeed");
//!
//! assert!(result.has_answer());
//! assert!(result.topic_entity_ids.contains(&"alice".to_string()));
//! assert!(!result.paths.is_empty());
//! ```

pub mod engine;
pub mod types;

#[cfg(test)]
mod tests;

pub use engine::TogEngine;
pub use types::{
    TogBeamPath, TogConfig, TogDecision, TogEntity, TogError, TogExploration, TogKnowledgeGraph,
    TogRelation, TogResult, TogTriple,
};
