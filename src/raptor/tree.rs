//! RAPTOR tree builder.

use super::cluster::{cluster, embed};
use super::types::{RaptorConfig, RaptorError, RaptorNode, RaptorTree};

// ── sentence splitter ─────────────────────────────────────────────────────────

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        cur.push(chars[i]);
        let end = chars[i] == '.' || chars[i] == '?' || chars[i] == '!';
        let next_space = i + 1 < n && chars[i + 1] == ' ';
        if end && next_space {
            let t = cur.trim().to_string();
            if !t.is_empty() {
                out.push(t);
            }
            cur = String::new();
            i += 2;
            continue;
        }
        i += 1;
    }
    let t = cur.trim().to_string();
    if !t.is_empty() {
        out.push(t);
    }
    out
}

/// Extractive summary: take the first `n` sentences.
fn extractive_summary(texts: &[&str], max_sentences: usize) -> String {
    let mut sentences: Vec<String> = Vec::new();
    for text in texts {
        sentences.extend(split_sentences(text));
        if sentences.len() >= max_sentences {
            break;
        }
    }
    sentences.truncate(max_sentences);
    sentences.join(" ")
}

// ── RaptorBuilder ─────────────────────────────────────────────────────────────

/// Builds a RAPTOR recursive summarization tree.
///
/// Leaf nodes hold the original text chunks; internal nodes hold extractive
/// summaries of their children.
#[derive(Debug, Clone, Default)]
pub struct RaptorBuilder;

impl RaptorBuilder {
    /// Create a new [`RaptorBuilder`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Build a [`RaptorTree`] from the given text chunks.
    ///
    /// # Errors
    ///
    /// Returns [`RaptorError::EmptyInput`] when `texts` is empty.
    pub fn build(
        &self,
        texts: &[String],
        config: &RaptorConfig,
    ) -> Result<RaptorTree, RaptorError> {
        if texts.is_empty() {
            return Err(RaptorError::EmptyInput);
        }

        let mut all_nodes: Vec<RaptorNode> = Vec::new();
        let mut next_id = 0usize;

        // Compute leaf embeddings
        let leaf_embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embed(t, config.dim)).collect();

        // Create leaf nodes (level 0)
        let leaf_ids: Vec<usize> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let node = RaptorNode {
                    id: next_id,
                    text: text.clone(),
                    level: 0,
                    children: Vec::new(),
                    embedding: leaf_embeddings[i].clone(),
                };
                let id = next_id;
                next_id += 1;
                all_nodes.push(node);
                id
            })
            .collect();

        // Recursively cluster and summarize
        let mut current_level_ids = leaf_ids;
        let mut current_level = 0usize;

        while current_level_ids.len() > 1 && current_level + 1 < config.max_levels {
            let current_embeddings: Vec<Vec<f32>> = current_level_ids
                .iter()
                .map(|&id| all_nodes[id].embedding.clone())
                .collect();

            let groups = cluster(
                &current_embeddings,
                config.cluster_size,
                config.cluster_strategy,
            );

            if groups.len() >= current_level_ids.len() {
                // No merging occurred — stop
                break;
            }

            current_level += 1;
            let mut next_level_ids = Vec::new();

            for group in &groups {
                let child_ids: Vec<usize> = group.iter().map(|&gi| current_level_ids[gi]).collect();
                let child_texts: Vec<&str> = child_ids
                    .iter()
                    .map(|&cid| all_nodes[cid].text.as_str())
                    .collect();

                let summary_text = extractive_summary(&child_texts, config.summary_sentences);
                let summary_emb = embed(&summary_text, config.dim);

                let node = RaptorNode {
                    id: next_id,
                    text: summary_text,
                    level: current_level,
                    children: child_ids,
                    embedding: summary_emb,
                };
                let id = next_id;
                next_id += 1;
                all_nodes.push(node);
                next_level_ids.push(id);
            }

            current_level_ids = next_level_ids;
        }

        let root_ids = current_level_ids;

        // Compute sorted unique levels
        let mut levels: Vec<usize> = all_nodes.iter().map(|n| n.level).collect();
        levels.sort_unstable();
        levels.dedup();

        Ok(RaptorTree {
            nodes: all_nodes,
            levels,
            root_ids,
        })
    }
}
