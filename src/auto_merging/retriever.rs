//! Auto-merging retriever: index leaves, collapse children into parents.

use std::collections::HashMap;

use crate::types::Document;

use super::hierarchy::ChunkHierarchy;
use super::types::{AutoMergeConfig, AutoMergeError, MergedHit};

// ── Lexical pseudo-embedding ──────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Tokens are hashed into `dim` buckets with FNV-1a; per-bucket counts are then
/// L2-normalised. The mapping is fully deterministic and model-free.
#[must_use]
fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.to_lowercase().as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

/// Cosine similarity between two L2-normalised, equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Split `text` into consecutive windows of at most `window_words` words.
///
/// Whitespace runs collapse to single spaces; a trailing partial window is
/// retained. Returns an empty vector when `text` holds no words.
fn word_windows(text: &str, window_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let step = window_words.max(1);
    words.chunks(step).map(|chunk| chunk.join(" ")).collect()
}

// ── AutoMergingRetriever ──────────────────────────────────────────────────────

/// Embedding paired with the leaf node it belongs to.
#[derive(Debug, Clone)]
struct LeafEmbedding {
    /// Identifier of the leaf node in the hierarchy.
    node_id: usize,
    /// L2-normalised lexical embedding of the leaf text.
    embedding: Vec<f32>,
}

/// Auto-merging retriever (LlamaIndex-style).
///
/// Indexes leaf chunks that reference their parents. At query time the most
/// relevant leaves are retrieved; when at least [`AutoMergeConfig::merge_threshold`]
/// of a parent's children appear in the retrieved set, those children are
/// replaced by the single parent chunk — recursively up the hierarchy. This
/// differs from parent-document retrieval, which expands one matched chunk to
/// its parent regardless of sibling coverage.
#[derive(Debug, Clone)]
pub struct AutoMergingRetriever {
    /// Hierarchy granularity and merge configuration.
    config: AutoMergeConfig,
    /// The parent / child chunk tree built from the indexed document.
    hierarchy: ChunkHierarchy,
    /// Embeddings for every leaf node, in leaf-id order.
    leaf_embeddings: Vec<LeafEmbedding>,
    /// Whether `build` has populated the hierarchy.
    built: bool,
}

impl AutoMergingRetriever {
    /// Create a new, unbuilt retriever with the given configuration.
    #[must_use]
    pub fn new(config: AutoMergeConfig) -> Self {
        Self {
            config,
            hierarchy: ChunkHierarchy::new(),
            leaf_embeddings: Vec::new(),
            built: false,
        }
    }

    /// Build the chunk hierarchy and leaf embeddings from `doc`.
    ///
    /// The document content is split into parent windows of
    /// [`AutoMergeConfig::parent_words`] words; each parent window is split into
    /// child windows of [`AutoMergeConfig::child_words`] words. A two-level
    /// hierarchy is produced (parents at level `1`, leaves at level `0`) and
    /// every leaf is embedded. Re-building replaces any prior state.
    ///
    /// # Errors
    ///
    /// Returns [`AutoMergeError::EmptyDocument`] when the content holds no words.
    pub fn build(&mut self, doc: &Document) -> Result<(), AutoMergeError> {
        let parent_windows = word_windows(&doc.content, self.config.parent_words);
        if parent_windows.is_empty() {
            return Err(AutoMergeError::EmptyDocument);
        }

        let mut hierarchy = ChunkHierarchy::new();
        let mut leaf_embeddings: Vec<LeafEmbedding> = Vec::new();

        for parent_text in &parent_windows {
            // Reserve the parent id before its children so children can point at it.
            let parent_id = hierarchy.push(String::new(), None, Vec::new(), 1);

            let child_windows = word_windows(parent_text, self.config.child_words);
            let mut child_ids: Vec<usize> = Vec::with_capacity(child_windows.len());
            for child_text in child_windows {
                let embedding = embed(&child_text, self.config.dim);
                let leaf_id = hierarchy.push(child_text, Some(parent_id), Vec::new(), 0);
                leaf_embeddings.push(LeafEmbedding {
                    node_id: leaf_id,
                    embedding,
                });
                child_ids.push(leaf_id);
            }

            // Fill in the parent's text (full window) and child links.
            if let Some(parent) = hierarchy.nodes.get_mut(parent_id) {
                parent.text.clone_from(parent_text);
                parent.children = child_ids;
            }
        }

        self.hierarchy = hierarchy;
        self.leaf_embeddings = leaf_embeddings;
        self.built = true;
        Ok(())
    }

    /// Borrow the chunk hierarchy built by [`AutoMergingRetriever::build`].
    #[must_use]
    pub fn hierarchy(&self) -> &ChunkHierarchy {
        &self.hierarchy
    }

    /// Retrieve the most relevant chunks for `query`, auto-merging as needed.
    ///
    /// The top `top_k` leaves are scored by cosine similarity; the retrieved
    /// leaves are then grouped by parent. Whenever the fraction of a parent's
    /// children present in the retrieved set reaches
    /// [`AutoMergeConfig::merge_threshold`], those children are replaced by the
    /// parent (recursively up the tree). Surviving hits are re-ranked by score
    /// and truncated to `top_k`.
    ///
    /// # Errors
    ///
    /// - [`AutoMergeError::NotBuilt`] when called before `build`.
    /// - [`AutoMergeError::EmptyQuery`] when `query` is blank.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<MergedHit>, AutoMergeError> {
        if !self.built {
            return Err(AutoMergeError::NotBuilt);
        }
        if query.trim().is_empty() {
            return Err(AutoMergeError::EmptyQuery);
        }
        if top_k == 0 || self.leaf_embeddings.is_empty() {
            return Ok(Vec::new());
        }

        let q_emb = embed(query, self.config.dim);

        // Score every leaf, then keep the best `top_k` as the retrieved set.
        let mut scored: Vec<(usize, f32)> = self
            .leaf_embeddings
            .iter()
            .map(|leaf| (leaf.node_id, cosine(&q_emb, &leaf.embedding)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(top_k);

        let merged = self.merge(scored);

        let mut hits: Vec<MergedHit> = merged
            .into_iter()
            .filter_map(|(node_id, score, merged_from)| {
                let node = self.hierarchy.node(node_id)?;
                Some(MergedHit {
                    text: node.text.clone(),
                    node_id,
                    level: node.level,
                    score,
                    merged_from,
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node_id.cmp(&b.node_id))
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Collapse retrieved leaves into ancestors, bottom-up.
    ///
    /// Each entry of `retrieved` is `(node_id, score)`. The set of active hits is
    /// repeatedly scanned: for any parent where the number of its direct children
    /// present among the active hits reaches the merge threshold, those children
    /// are removed and the parent is inserted with the best child score and the
    /// summed `merged_from` count. The loop continues up the tree until no merge
    /// applies. Returns `(node_id, score, merged_from)` for the surviving hits.
    fn merge(&self, retrieved: Vec<(usize, f32)>) -> Vec<(usize, f32, usize)> {
        // Active hits keyed by node id → (best score, leaf children collapsed).
        let mut active: HashMap<usize, (f32, usize)> = HashMap::new();
        for (node_id, score) in retrieved {
            let entry = active.entry(node_id).or_insert((f32::NEG_INFINITY, 1));
            if score > entry.0 {
                entry.0 = score;
            }
        }

        loop {
            // Group active hits by their parent.
            let mut by_parent: HashMap<usize, Vec<usize>> = HashMap::new();
            for &node_id in active.keys() {
                if let Some(parent) = self.hierarchy.parent_of(node_id) {
                    by_parent.entry(parent).or_default().push(node_id);
                }
            }

            // Find every parent that meets the merge threshold, deterministically.
            let mut to_merge: Vec<(usize, Vec<usize>)> = by_parent
                .into_iter()
                .filter(|(parent, present)| {
                    let total = self.hierarchy.children_of(*parent).len();
                    if total == 0 {
                        return false;
                    }
                    #[allow(clippy::cast_precision_loss)]
                    let fraction = present.len() as f32 / total as f32;
                    fraction >= self.config.merge_threshold
                })
                .collect();

            if to_merge.is_empty() {
                break;
            }
            to_merge.sort_by_key(|(parent, _)| *parent);

            for (parent, mut present) in to_merge {
                present.sort_unstable();
                let mut best_score = f32::NEG_INFINITY;
                let mut collapsed = 0usize;
                for child in &present {
                    if let Some((score, count)) = active.remove(child) {
                        if score > best_score {
                            best_score = score;
                        }
                        collapsed += count;
                    }
                }
                // Insert (or reinforce) the parent hit.
                let entry = active.entry(parent).or_insert((f32::NEG_INFINITY, 0));
                if best_score > entry.0 {
                    entry.0 = best_score;
                }
                entry.1 += collapsed;
            }
        }

        let mut out: Vec<(usize, f32, usize)> = active
            .into_iter()
            .map(|(node_id, (score, count))| (node_id, score, count))
            .collect();
        out.sort_by_key(|(node_id, _, _)| *node_id);
        out
    }
}
