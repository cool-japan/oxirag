//! `G-Retriever` — subgraph retrieval for textual-graph RAG, framed as a
//! Prize-Collecting Steiner Tree (He et al. 2024, "G-Retriever:
//! Retrieval-Augmented Generation for Textual Graph Understanding and Question
//! Answering").
//!
//! Given a natural-language query and a textual knowledge graph — entities
//! ([`GRetrieverEntity`]) as nodes, labelled relations
//! ([`GRetrieverRelation`]) as edges — G-Retriever returns the *subgraph* most
//! useful for answering the query, then textualises it for a downstream
//! language model. The retrieval is posed as an **optimisation**: assign each
//! node a **prize** (its query relevance) and each edge a **cost**, and find
//! the connected subgraph that **maximises collected prize minus paid cost**.
//! A very relevant node is worth "buying" edges to reach; an off-topic node is
//! left out — unless it is a cheap *Steiner* waypoint that two relevant nodes
//! must pass through to be connected.
//!
//! # How this differs from [`knowledge_graph_qa`](crate::knowledge_graph_qa)
//!
//! Both modules extract a query-relevant subgraph from a knowledge graph, but
//! they do so by fundamentally different means:
//!
//! * [`knowledge_graph_qa`](crate::knowledge_graph_qa) performs **plain
//!   breadth-first subgraph expansion**: it seeds on entities whose names match
//!   the query and grows outward a fixed number of hops, keeping every node it
//!   reaches within the hop/`top_k` budget. It never weighs a node's relevance
//!   against the cost of the edges spent reaching it; connectivity is decided
//!   purely by hop distance.
//! * **G-Retriever**, by contrast, frames subgraph retrieval as a
//!   **Prize-Collecting Steiner Tree (PCST) optimisation**. There is no hop
//!   budget: which nodes and edges are included falls out of a global
//!   prize-versus-cost trade-off. An expensive edge to a marginally relevant
//!   node is *excluded* even if it is one hop away; a chain of cheap edges
//!   through irrelevant Steiner nodes is *included* when it links two highly
//!   relevant nodes. The trade-off is solved with the **Goemans–Williamson
//!   primal-dual `2`-approximation** followed by **strong pruning** — the same
//!   algorithm family that the reference `pcst_fast` library, and the original
//!   G-Retriever paper, rely on.
//!
//! # Layout
//!
//! * [`types`] — configuration, knowledge-graph and result data structures, the
//!   [`GRetrieverError`] enum, and the abstract PCST view ([`PcstNode`],
//!   [`PcstEdge`], [`PcstForest`]).
//! * [`pcst`] — the [`PcstSolver`]: Goemans–Williamson moat growing plus strong
//!   pruning. This is the algorithmic core.
//! * [`engine`] — the [`GRetrieverEngine`]: builds prizes and costs from the
//!   query, runs the solver, and maps the result back into a
//!   [`GRetrieverSubgraph`].
//!
//! # Examples
//!
//! Solving a Prize-Collecting Steiner Tree directly: two high-prize terminals
//! are connected through a zero-prize Steiner waypoint because the connecting
//! edges are cheap.
//!
//! ```
//! use oxirag::g_retriever::{PcstEdge, PcstNode, PcstSolver};
//!
//! let nodes = vec![
//!     PcstNode::new(0, 10.0), // terminal
//!     PcstNode::new(1, 0.0),  // Steiner waypoint (no prize of its own)
//!     PcstNode::new(2, 10.0), // terminal
//! ];
//! let edges = vec![PcstEdge::new(0, 1, 1.0), PcstEdge::new(1, 2, 1.0)];
//!
//! let solver = PcstSolver::new(/* prune */ true, /* single_component */ false);
//! let forest = solver.solve(&nodes, &edges).expect("valid PCST instance");
//!
//! // The waypoint is pulled in so the two terminals end up connected.
//! assert!(forest.selected_nodes().contains(&1));
//! assert_eq!(forest.selected_edges().len(), 2);
//! ```
//!
//! Retrieving a subgraph from a textual knowledge graph:
//!
//! ```
//! use oxirag::g_retriever::{
//!     GRetrieverConfig, GRetrieverEngine, GRetrieverEntity, GRetrieverRelation,
//! };
//!
//! let query = "graph neural network question answering";
//! let entities = vec![
//!     GRetrieverEntity::new("gnn", query),
//!     GRetrieverEntity::new("task", "graph question answering task"),
//!     GRetrieverEntity::new("fruit", "banana smoothie recipe"),
//! ];
//! let relations = vec![
//!     GRetrieverRelation::new("gnn", "task", "used for"),
//!     GRetrieverRelation::new("task", "fruit", "unrelated to"),
//! ];
//!
//! let engine = GRetrieverEngine::new(GRetrieverConfig::default());
//! let subgraph = engine
//!     .retrieve(query, &entities, &relations)
//!     .expect("retrieval succeeds");
//!
//! // The on-topic entity is retrieved; the off-topic one is not.
//! assert!(subgraph.contains_entity("gnn"));
//! assert!(!subgraph.contains_entity("fruit"));
//! let _prompt = subgraph.textualize();
//! ```

pub mod engine;
pub mod pcst;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::GRetrieverEngine;
pub use pcst::PcstSolver;
pub use types::{
    GRetrieverConfig, GRetrieverEntity, GRetrieverError, GRetrieverRelation, GRetrieverResult,
    GRetrieverRootMode, GRetrieverSubgraph, PcstEdge, PcstForest, PcstNode,
};
