//! Astute RAG: reconciling internal (parametric) and external (retrieved) knowledge.
//!
//! Based on Wang et al. 2024, *"Astute RAG: Overcoming Imperfect Retrieval
//! Augmentation and Knowledge Conflicts for Large Language Models"*.
//!
//! Retrieval is rarely perfect: retrieved passages may be irrelevant, outdated,
//! or contradict what the model already knows. Naively trusting retrieved text
//! can inject errors, while ignoring it discards genuinely useful evidence.
//! Astute RAG addresses this by treating the model's **internal** parametric
//! knowledge and the **external** retrieved knowledge as two independent
//! sources, then *consolidating* them: shared facts reinforce one another,
//! conflicts are detected and resolved by reliability, and a single attributed
//! answer is synthesized.
//!
//! This contrasts with corrective RAG (`corrective_rag`), which grades and
//! re-retrieves documents but never reconciles them against the model's own
//! parametric knowledge.
//!
//! # Algorithm
//!
//! 1. **Recall** internal statements for the query (via [`InternalKnowledge`]).
//! 2. **Extract** salient external statements from retrieved documents, scoring
//!    each one's reliability by cross-document corroboration
//!    ([`AstuteConsolidator::external_statements`]).
//! 3. **Detect** conflicts: an internal claim and an external claim that share a
//!    subject but disagree via negation or numeric mismatch
//!    ([`AstuteConsolidator::detect_conflict`]).
//! 4. **Resolve** each conflict by reliability and the `prefer_external` policy
//!    (external generally wins when corroborated; internal wins when external is
//!    uncorroborated).
//! 5. **Synthesize** an attributed answer from the winning statements
//!    ([`AstuteConsolidator::consolidate`]).
//!
//! The whole pipeline is pure Rust, deterministic, and free of ML or RNG
//! dependencies.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`KnowledgeSource`] | Tag a claim as `Internal` or `External(DocumentId)` |
//! | [`KnowledgeStatement`] | A claim with provenance and reliability |
//! | [`KnowledgeConflict`] | A resolved internal-vs-external disagreement |
//! | [`ConsolidatedKnowledge`] | Final statements, conflicts, and answer |
//! | [`InternalKnowledge`] | Recall the model's parametric claims |
//! | [`MockInternalKnowledge`] | Scripted parametric source for tests |
//! | [`AstuteConfig`] | Reliability threshold and resolution policy |
//! | [`AstuteConsolidator`] | Orchestrates extraction → conflict → resolution |
//!
//! # Quick start
//!
//! ```rust
//! use oxirag::astute_rag::{
//!     AstuteConfig, AstuteConsolidator, MockInternalKnowledge,
//! };
//! use oxirag::types::{Document, DocumentId};
//!
//! let docs = vec![
//!     Document::new("The tower was completed in 1889.")
//!         .with_id(DocumentId::from_string("d1")),
//! ];
//! let model = MockInternalKnowledge::new(vec![
//!     "The tower was completed in 1989.".to_string(),
//! ]);
//!
//! let consolidator = AstuteConsolidator::new(AstuteConfig::default());
//! let result = consolidator.run("when was the tower completed", &model, &docs).unwrap();
//!
//! assert!(!result.answer.is_empty());
//! assert!(result.has_conflicts());
//! ```

pub mod consolidator;
#[cfg(test)]
mod tests;
pub mod types;

pub use consolidator::AstuteConsolidator;
pub use types::{
    AstuteConfig, AstuteError, ConsolidatedKnowledge, InternalKnowledge, KnowledgeConflict,
    KnowledgeSource, KnowledgeStatement, MockInternalKnowledge,
};
