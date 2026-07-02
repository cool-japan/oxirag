//! The [`ClarificationTree`] structure and its recursive pruning.
//!
//! A [`ClarificationTree`] is a *forest*: one [`ClarificationNode`] per
//! top-level disambiguated reading of the original question (or a single
//! root-equals-leaf node when the question was judged unambiguous). Each node
//! may itself have children when its own reading was still ambiguous and was
//! disambiguated further, up to the configured `max_depth`.
//!
//! Every node — leaf or not — carries its own `relevance_score` and `answer`,
//! computed once during tree construction from that node's own retrieval. This
//! is what makes the recursive prune-and-fallback rule in [`ClarificationTree::prune`]
//! possible: when a node loses every child to pruning, it simply reverts to
//! being a leaf using the answer it already had.

// ── ClarificationNode ─────────────────────────────────────────────────────────

/// A single node in a [`ClarificationTree`]: one disambiguated reading of a
/// question, together with its retrieval-backed answer and any further
/// disambiguations of that reading.
#[derive(Debug, Clone, PartialEq)]
pub struct ClarificationNode {
    /// The (disambiguated) question this node represents.
    pub question: String,
    /// Retrieval-support-based confidence in `[0.0, 1.0]` for this node's own
    /// question, independent of its children.
    pub relevance_score: f32,
    /// This node's own best-effort answer, computed from its own retrieval.
    /// Always `Some` once the tree has been built; it is what a node falls
    /// back to when every one of its children is pruned away.
    pub answer: Option<String>,
    /// Further disambiguations of [`ClarificationNode::question`], if any.
    /// Empty when this node is a leaf (either because the question was not
    /// judged ambiguous, or because `max_depth` was reached).
    pub children: Vec<ClarificationNode>,
}

impl ClarificationNode {
    /// Create a new node with no children.
    #[must_use]
    pub fn new(
        question: impl Into<String>,
        relevance_score: f32,
        answer: impl Into<String>,
    ) -> Self {
        Self {
            question: question.into(),
            relevance_score,
            answer: Some(answer.into()),
            children: Vec::new(),
        }
    }

    /// `true` when this node has no children (it is a leaf of the tree).
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collect references to every leaf reachable from this node (including
    /// itself, if it is already a leaf) into `out`.
    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a ClarificationNode>) {
        if self.children.is_empty() {
            out.push(self);
        } else {
            for child in &self.children {
                child.collect_leaves(out);
            }
        }
    }

    /// Recursively drop children (and their descendants) whose own
    /// `relevance_score` falls below `threshold`. See
    /// [`ClarificationTree::prune`] for the full semantics.
    fn prune_children(&mut self, threshold: f32) {
        self.children
            .retain(|child| child.relevance_score >= threshold);
        for child in &mut self.children {
            child.prune_children(threshold);
        }
    }

    /// Total number of nodes in the subtree rooted at this node, including
    /// itself.
    #[must_use]
    pub fn node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(ClarificationNode::node_count)
            .sum::<usize>()
    }
}

// ── ClarificationTree ─────────────────────────────────────────────────────────

/// A forest of [`ClarificationNode`] trees, one per top-level disambiguated
/// reading of the original question.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClarificationTree {
    /// The top-level disambiguated readings (or the single root-equals-leaf
    /// node when the original question was judged unambiguous).
    pub root_questions: Vec<ClarificationNode>,
}

impl ClarificationTree {
    /// Create an empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` when the tree holds no root questions at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.root_questions.is_empty()
    }

    /// Total number of nodes across the whole forest.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.root_questions
            .iter()
            .map(ClarificationNode::node_count)
            .sum()
    }

    /// Collect references to every leaf node in the forest, in depth-first,
    /// left-to-right order.
    #[must_use]
    pub fn leaves(&self) -> Vec<&ClarificationNode> {
        let mut out = Vec::new();
        for root in &self.root_questions {
            root.collect_leaves(&mut out);
        }
        out
    }

    /// Recursively prune the tree, dropping every node whose own
    /// `relevance_score` is below `threshold`.
    ///
    /// The rule is applied uniformly at every level, treating `root_questions`
    /// itself as the top-level "children" list of an implicit root: a node is
    /// removed from whichever list currently holds it (`root_questions`, or a
    /// parent's `children`) precisely when its own `relevance_score` is below
    /// `threshold`.
    ///
    /// Crucially, a *surviving* node is never removed merely because all of
    /// its children were pruned — since every node already carries its own
    /// precomputed `answer`, it simply reverts to being a leaf and that answer
    /// becomes the effective (best-effort) answer for the reading it
    /// represents. This mirrors the paper's recursive prune-and-fallback rule.
    pub fn prune(&mut self, threshold: f32) {
        self.root_questions
            .retain(|node| node.relevance_score >= threshold);
        for root in &mut self.root_questions {
            root.prune_children(threshold);
        }
    }
}
