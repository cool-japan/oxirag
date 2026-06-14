//! Chunk hierarchy: a tree of parent / child text nodes.

// ── ChunkNode ─────────────────────────────────────────────────────────────────

/// A single node in a [`ChunkHierarchy`].
///
/// Leaves (`level == 0`) carry the finest-grained text and reference a parent;
/// interior nodes aggregate the text of their descendants and reference both a
/// parent (unless they are the root level) and their children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkNode {
    /// Identifier of this node, unique within its hierarchy.
    pub id: usize,
    /// The text covered by this node.
    pub text: String,
    /// Identifier of the parent node, or `None` for top-level nodes.
    pub parent: Option<usize>,
    /// Identifiers of the direct children, in document order (empty for leaves).
    pub children: Vec<usize>,
    /// Depth from the leaves: `0` for leaves, increasing toward the root.
    pub level: usize,
}

// ── ChunkHierarchy ────────────────────────────────────────────────────────────

/// A tree of [`ChunkNode`]s linking fine-grained leaves to coarser ancestors.
///
/// Nodes are stored in a flat vector indexed by their `id`; the structure is
/// built once (typically by the retriever) and queried thereafter. Auto-merging
/// walks this tree upward, collapsing children into a parent when enough of them
/// are retrieved.
#[derive(Debug, Clone, Default)]
pub struct ChunkHierarchy {
    /// All nodes, indexed by id (the vector position equals [`ChunkNode::id`]).
    pub nodes: Vec<ChunkNode>,
}

impl ChunkHierarchy {
    /// Create a new, empty hierarchy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a node by its identifier.
    #[must_use]
    pub fn node(&self, id: usize) -> Option<&ChunkNode> {
        self.nodes.get(id)
    }

    /// Return the identifiers of the direct children of `id`.
    ///
    /// Returns an empty slice when `id` is unknown or is a leaf.
    #[must_use]
    pub fn children_of(&self, id: usize) -> &[usize] {
        self.nodes.get(id).map_or(&[], |n| n.children.as_slice())
    }

    /// Return the identifier of the parent of `id`, if any.
    ///
    /// Returns `None` when `id` is unknown or is a top-level node.
    #[must_use]
    pub fn parent_of(&self, id: usize) -> Option<usize> {
        self.nodes.get(id).and_then(|n| n.parent)
    }

    /// Return the identifiers of every leaf node (those with no children).
    ///
    /// Leaves are returned in ascending id order.
    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.children.is_empty())
            .map(|n| n.id)
            .collect()
    }

    /// Total number of nodes in the hierarchy.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return `true` when the hierarchy holds no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Push a node, returning its assigned identifier.
    ///
    /// The caller must supply nodes such that ids are dense and sequential; the
    /// returned id is the node's position in [`ChunkHierarchy::nodes`].
    pub(crate) fn push(
        &mut self,
        text: String,
        parent: Option<usize>,
        children: Vec<usize>,
        level: usize,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(ChunkNode {
            id,
            text,
            parent,
            children,
            level,
        });
        id
    }
}
