//! Generative retriever: builds semantic doc-ids and generates them by
//! constrained beam traversal of the centroid tree.

use std::cmp::Ordering;

use super::tree::{TreeNode, build_tree};
use super::types::{GenHit, GenRetrievalConfig, GenRetrievalError, SemanticDocId, cosine, embed};
use crate::types::Document;

// ── GenerativeRetriever ───────────────────────────────────────────────────────

/// DSI-style generative retriever (Tay et al., 2022).
///
/// Each document is assigned a hierarchical semantic id by recursively
/// clustering the corpus; the path of cluster indices from the root to a
/// document's leaf becomes its [`SemanticDocId`]. At query time the retriever
/// "generates" ids by constrained traversal: starting at the root it
/// beam-searches downward, at every level keeping the `beam` clusters whose
/// centroid best matches the query, until it reaches leaf documents, which are
/// then ranked by full lexical cosine similarity.
#[derive(Debug, Clone)]
pub struct GenerativeRetriever {
    /// Construction and traversal configuration.
    config: GenRetrievalConfig,
    /// The indexed documents (parallel to `doc_ids` and `embeddings`).
    docs: Vec<Document>,
    /// Per-document semantic ids (parallel to `docs`).
    doc_ids: Vec<SemanticDocId>,
    /// Per-document lexical embeddings (parallel to `docs`).
    embeddings: Vec<Vec<f32>>,
    /// Centroid tree nodes; node `0` is the root when the tree is non-empty.
    nodes: Vec<TreeNode>,
    /// Whether [`GenerativeRetriever::build`] has populated the tree.
    built: bool,
}

impl GenerativeRetriever {
    /// Create a new, empty retriever with the given configuration.
    #[must_use]
    pub fn new(config: GenRetrievalConfig) -> Self {
        Self {
            config,
            docs: Vec::new(),
            doc_ids: Vec::new(),
            embeddings: Vec::new(),
            nodes: Vec::new(),
            built: false,
        }
    }

    /// Build the semantic id tree from `docs`.
    ///
    /// Embeds every document, recursively KMeans-lite-clusters the corpus into
    /// at most `branching` groups per level up to `max_depth`, and assigns each
    /// document its semantic id (the cluster-index path to its leaf). Calling
    /// `build` again replaces any previously indexed corpus.
    ///
    /// # Errors
    ///
    /// Returns [`GenRetrievalError::EmptyCorpus`] when `docs` is empty.
    pub fn build(&mut self, docs: &[Document]) -> Result<(), GenRetrievalError> {
        if docs.is_empty() {
            return Err(GenRetrievalError::EmptyCorpus);
        }
        let dim = self.config.dim;
        let embeddings: Vec<Vec<f32>> = docs.iter().map(|d| embed(&d.content, dim)).collect();
        let output = build_tree(
            &embeddings,
            self.config.branching,
            self.config.max_depth,
            dim,
        );
        self.doc_ids = output.paths.into_iter().map(SemanticDocId::new).collect();
        self.nodes = output.nodes;
        self.docs = docs.to_vec();
        self.embeddings = embeddings;
        self.built = true;
        Ok(())
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Return `true` when no documents are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &GenRetrievalConfig {
        &self.config
    }

    /// Return the semantic id assigned to the document at `index`, if any.
    #[must_use]
    pub fn doc_id(&self, index: usize) -> Option<&SemanticDocId> {
        self.doc_ids.get(index)
    }

    /// Generate the best `top_k` documents for `query` by constrained traversal.
    ///
    /// Beam-searches the centroid tree from the root, keeping the `beam`
    /// best-matching clusters at every level, gathers the candidate leaf
    /// documents reached, and ranks them by full lexical cosine similarity.
    ///
    /// # Errors
    ///
    /// Returns [`GenRetrievalError::NotBuilt`] when called before
    /// [`GenerativeRetriever::build`], and [`GenRetrievalError::EmptyQuery`]
    /// when `query` is blank.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<GenHit>, GenRetrievalError> {
        if !self.built {
            return Err(GenRetrievalError::NotBuilt);
        }
        if query.trim().is_empty() {
            return Err(GenRetrievalError::EmptyQuery);
        }
        if top_k == 0 || self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let q_emb = embed(query, self.config.dim);
        let candidate_docs = self.traverse(&q_emb);

        let mut hits: Vec<GenHit> = candidate_docs
            .into_iter()
            .map(|doc_index| GenHit {
                document: self.docs[doc_index].clone(),
                doc_id: self.doc_ids[doc_index].clone(),
                score: cosine(&q_emb, &self.embeddings[doc_index]),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.doc_id.as_string().cmp(&b.doc_id.as_string()))
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Beam-traverse the centroid tree, returning the document indices held by
    /// every leaf the beam reaches.
    fn traverse(&self, q_emb: &[f32]) -> Vec<usize> {
        let beam = self.config.beam.max(1);
        let mut frontier = vec![0usize];
        let mut reached_leaves: Vec<usize> = Vec::new();

        while !frontier.is_empty() {
            let mut candidates: Vec<usize> = Vec::new();
            for &node_id in &frontier {
                let node = &self.nodes[node_id];
                if node.is_leaf() {
                    reached_leaves.push(node_id);
                } else {
                    candidates.extend(node.children.iter().copied());
                }
            }
            if candidates.is_empty() {
                break;
            }
            // Score child clusters by centroid match and keep the top `beam`.
            candidates.sort_by(|&a, &b| {
                let sa = cosine(q_emb, &self.nodes[a].centroid);
                let sb = cosine(q_emb, &self.nodes[b].centroid);
                sb.partial_cmp(&sa)
                    .unwrap_or(Ordering::Equal)
                    .then(a.cmp(&b))
            });
            candidates.truncate(beam);
            frontier = candidates;
        }

        let mut docs: Vec<usize> = Vec::new();
        for leaf_id in reached_leaves {
            docs.extend(self.nodes[leaf_id].doc_indices.iter().copied());
        }
        docs
    }
}
