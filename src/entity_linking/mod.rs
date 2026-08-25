//! Entity linking / disambiguation against a canonical catalog.
//!
//! Resolves a surface-form **mention** to a **canonical entity** drawn from an
//! [`EntityCatalog`]. Each canonical entity carries a primary name, optional
//! aliases, and a free-text description. Candidate generation is an exact,
//! case-insensitive match on names and aliases; when a single surface form
//! matches several entities, the ambiguity is resolved by comparing the
//! surrounding **context** to each candidate's **description** (cosine over
//! deterministic lexical pseudo-embeddings). Mentions with no candidate, or
//! whose best candidate scores below a confidence floor, are left unlinked
//! (NIL).
//!
//! This is *distinct* from mention extraction (see the `entity_memory` module):
//! here the surface forms are mapped onto a curated catalog and disambiguated,
//! rather than merely detected and accumulated.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`CanonicalEntity`] | A catalog entry: id, name, aliases, description |
//! | [`EntityCatalog`] | Stores entities and a case-insensitive alias index |
//! | [`EntityLinker`] | Detects mentions and links them to the catalog |
//! | [`LinkedEntity`] | The per-mention outcome (a canonical id or NIL) |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "entity-linking")] {
//! use oxirag::entity_linking::{
//!     CanonicalEntity, EntityCatalog, EntityLinkConfig, EntityLinker,
//! };
//!
//! let mut catalog = EntityCatalog::new();
//! catalog.add(
//!     CanonicalEntity::new("Q1", "Mercury")
//!         .with_alias("Mercury")
//!         .with_description("the smallest planet orbiting the sun"),
//! );
//! catalog.add(
//!     CanonicalEntity::new("Q2", "Mercury")
//!         .with_alias("Mercury")
//!         .with_description("a silvery liquid metal chemical element"),
//! );
//!
//! let linker = EntityLinker::new(EntityLinkConfig::default(), catalog);
//! let linked = linker.link("Mercury", "the planet orbiting the sun is hot");
//! assert_eq!(linked.entity_id.as_deref(), Some("Q1"));
//! # }
//! ```

pub mod catalog;
pub mod linker;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use catalog::EntityCatalog;
pub use linker::{EntityLinker, cosine, embed};
pub use types::{CanonicalEntity, EntityLinkConfig, EntityLinkError, EntityMention, LinkedEntity};
