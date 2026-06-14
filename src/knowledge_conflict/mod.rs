//! Knowledge-conflict detection and resolution across retrieved passages.
//!
//! Retrieval-augmented generation frequently surfaces passages that *disagree*:
//! one source says a product was approved, another says it was not; one reports a
//! population of five million, another eight; one dates a founding to 1990,
//! another to 2005. This module both **detects** those contradictions *between*
//! passages and **resolves** them under a configurable policy.
//!
//! It is distinct from two neighbouring modules:
//!
//! * `consistency_checker` scans a *single generated answer* for internal,
//!   intra-answer contradictions.
//! * `fact_check` scores a *claim* against a corpus and emits a `SUPPORTS` /
//!   `REFUTES` / `NOT_ENOUGH_INFO` verdict.
//!
//! Here the unit of comparison is a *pair of retrieved passages*, and the output
//! is a list of [`PassageConflict`]s plus, optionally, a [`ConflictResolution`]
//! per conflict.
//!
//! # Detection
//!
//! [`ConflictDetector`] walks every pair of passages, splits each into sentences,
//! and flags a sentence pair when it shares at least `min_shared_terms` content
//! tokens (the *same subject*) yet disagrees via:
//!
//! * [`ConflictKind::Negation`] — exactly one side carries a negation marker.
//! * [`ConflictKind::Numeric`] — the sides cite different numbers.
//! * [`ConflictKind::Temporal`] — the sides cite different years.
//!
//! # Resolution
//!
//! [`ConflictResolver`] picks a winning side under a [`ConflictPolicy`]:
//!
//! * [`ConflictPolicy::Recency`] — the newer source wins (lower age in days,
//!   supplied as an explicit ages map — the resolver never reads the wall clock).
//! * [`ConflictPolicy::Authority`] — the higher-authority source wins.
//! * [`ConflictPolicy::Majority`] — the claim corroborated by more other passages
//!   wins.
//!
//! All detection and resolution is pure, deterministic, and free of randomness or
//! machine learning.
//!
//! # Example
//!
//! ```
//! use oxirag::knowledge_conflict::{
//!     ConflictDetector, ConflictKind, ConflictPolicy, ConflictResolver,
//!     KnowledgeConflictConfig,
//! };
//! use oxirag::types::{Document, DocumentId};
//! use std::collections::HashMap;
//!
//! let docs = vec![
//!     Document::new("The treaty was ratified in 1990.").with_id("a"),
//!     Document::new("The treaty was ratified in 2005.").with_id("b"),
//! ];
//!
//! let config = KnowledgeConflictConfig::new().with_policy(ConflictPolicy::Recency);
//! let detector = ConflictDetector::new(config.clone());
//! let conflicts = detector.detect(&docs);
//! assert_eq!(conflicts.len(), 1);
//! assert_eq!(conflicts[0].kind, ConflictKind::Temporal);
//!
//! // Resolve: the newer (lower-age) source wins.
//! let mut ages = HashMap::new();
//! ages.insert(DocumentId::from_string("a"), 3650.0_f64);
//! ages.insert(DocumentId::from_string("b"), 10.0_f64);
//! let authority: HashMap<DocumentId, f32> = HashMap::new();
//!
//! let resolver = ConflictResolver::new(config);
//! let resolution = resolver.resolve(&conflicts[0], &docs, &ages, &authority);
//! assert_eq!(resolution.winner, 1);
//! assert!(!resolution.rationale.is_empty());
//! ```

pub mod detector;
pub mod resolver;
#[cfg(test)]
mod tests;
pub mod types;

pub use detector::ConflictDetector;
pub use resolver::ConflictResolver;
pub use types::{
    ConflictKind, ConflictPolicy, ConflictResolution, KnowledgeConflictConfig,
    KnowledgeConflictError, PassageConflict,
};
