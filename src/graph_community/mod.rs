//! Louvain modularity community detection over knowledge graphs.
//!
//! Operates on plain `&[GraphEntity]` + `&[GraphRelationship]` slices —
//! fully independent of any store backend.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`CommunityDetector`] | Trait for sync detection algorithms |
//! | [`LouvainDetector`] | Greedy modularity optimiser |
//! | [`Community`] | Set of co-community entity ids |
//! | [`CommunityGraph`] | Full detection result + modularity score |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(all(feature = "graph-community", feature = "graphrag"))] {
//! use oxirag::prelude::*;
//!
//! let detector = LouvainDetector::new();
//! # }
//! ```

pub mod louvain;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use louvain::LouvainDetector;
pub use types::{CommunityId, GraphCommunityConfig, GraphCommunityError};

#[cfg(feature = "graphrag")]
pub use types::{Community, CommunityDetector, CommunityGraph};
