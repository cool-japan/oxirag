//! Types for the `graph_community` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── CommunityId ───────────────────────────────────────────────────────────────

/// Opaque community identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommunityId(pub usize);

impl CommunityId {
    /// Create a new community id.
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the inner integer value.
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for CommunityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "community:{}", self.0)
    }
}

// ── Community ─────────────────────────────────────────────────────────────────

/// A detected graph community.
#[cfg(feature = "graphrag")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    /// Unique community identifier.
    pub id: CommunityId,
    /// Entity ids belonging to this community.
    pub members: Vec<crate::layer4_graph::types::EntityId>,
    /// Hierarchy level (0 = leaf-level).
    pub level: usize,
}

#[cfg(feature = "graphrag")]
impl Community {
    /// Return the number of members.
    #[must_use]
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Return `true` when the community contains only one entity.
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.members.len() == 1
    }
}

// ── CommunityGraph ────────────────────────────────────────────────────────────

/// The full set of detected communities together with the graph modularity score.
#[cfg(feature = "graphrag")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityGraph {
    /// Detected communities.
    pub communities: Vec<Community>,
    /// Graph modularity Q in `[-0.5, 1.0]`.
    pub modularity: f64,
}

#[cfg(feature = "graphrag")]
impl CommunityGraph {
    /// Return the number of communities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.communities.len()
    }

    /// Return `true` if no communities are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.communities.is_empty()
    }
}

// ── GraphCommunityConfig ──────────────────────────────────────────────────────

/// Configuration for community detection.
#[derive(Debug, Clone)]
pub struct GraphCommunityConfig {
    /// Resolution parameter for Louvain (higher → smaller communities).
    ///
    /// Defaults to `1.0`.
    pub resolution: f64,
    /// Maximum number of hierarchy levels.
    ///
    /// Defaults to `5`.
    pub max_levels: usize,
}

impl Default for GraphCommunityConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_levels: 5,
        }
    }
}

impl GraphCommunityConfig {
    /// Set the resolution.
    #[must_use]
    pub fn with_resolution(mut self, v: f64) -> Self {
        self.resolution = v;
        self
    }

    /// Set the maximum number of hierarchy levels.
    #[must_use]
    pub fn with_max_levels(mut self, v: usize) -> Self {
        self.max_levels = v;
        self
    }
}

// ── CommunityDetector trait ───────────────────────────────────────────────────

/// Trait for synchronous community detection algorithms.
///
/// All inputs are **plain slices** — no store dependency, fully testable.
#[cfg(feature = "graphrag")]
pub trait CommunityDetector {
    /// Detect communities from entity and relationship slices.
    ///
    /// # Errors
    ///
    /// Returns [`GraphCommunityError::EmptyGraph`] when no entities are provided.
    fn detect(
        &self,
        entities: &[crate::layer4_graph::types::GraphEntity],
        relationships: &[crate::layer4_graph::types::GraphRelationship],
        config: &GraphCommunityConfig,
    ) -> Result<CommunityGraph, GraphCommunityError>;
}

// ── GraphCommunityError ───────────────────────────────────────────────────────

/// Errors from the `graph_community` module.
#[derive(Debug, Error)]
pub enum GraphCommunityError {
    /// Entity list was empty.
    #[error("Entity list must not be empty")]
    EmptyGraph,
}
