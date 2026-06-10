//! Parent-document retriever: search children, expand to parents.

use std::collections::HashMap;

use crate::types::DocumentId;

use super::types::{ExpandedResult, ParentChildIndex, ParentDocumentConfig, ParentDocumentError};

// ── ParentDocumentRetriever ───────────────────────────────────────────────────

/// Retrieves child chunks, then expands each match to its parent document.
///
/// Deduplicates parent returns — multiple child hits for the same parent are
/// combined into one [`ExpandedResult`].
pub struct ParentDocumentRetriever {
    index: ParentChildIndex,
    config: ParentDocumentConfig,
}

impl ParentDocumentRetriever {
    /// Create a new retriever with the given index and config.
    #[must_use]
    pub fn new(index: ParentChildIndex, config: ParentDocumentConfig) -> Self {
        Self { index, config }
    }

    /// Search `query` against the Echo, then expand child hits to parents.
    ///
    /// # Errors
    ///
    /// - [`ParentDocumentError::EmptyQuery`] when `query` is blank.
    /// - [`ParentDocumentError::RetrievalFailed`] when Echo search fails.
    ///
    /// Child hits whose parent is absent from the index are **skipped** (not
    /// returned as an error).
    pub async fn run<E>(
        &self,
        query: &str,
        echo: &E,
    ) -> Result<Vec<ExpandedResult>, ParentDocumentError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        let query_trimmed = query.trim();
        if query_trimmed.is_empty() {
            return Err(ParentDocumentError::EmptyQuery);
        }

        let child_results = echo
            .search(query_trimmed, self.config.top_k, None)
            .await
            .map_err(|e| ParentDocumentError::RetrievalFailed(e.to_string()))?;

        if !self.config.return_parent {
            // Return child results wrapped as their own "parents"
            return Ok(child_results
                .into_iter()
                .map(|r| ExpandedResult {
                    parent: r.document.clone(),
                    matched_children: vec![r.document.id.clone()],
                    score: r.score,
                })
                .collect());
        }

        // Group by parent: parent_id → (best_score, Vec<child_id>)
        let mut parent_groups: HashMap<String, (f32, Vec<DocumentId>)> = HashMap::new();

        for result in &child_results {
            let child_id = result.document.id.as_str();
            let parent_id = if let Some(pid) = self.index.child_to_parent.get(child_id) {
                pid.as_str().to_string()
            } else {
                // Try metadata "parent_id" key
                if let Some(pid) = result.document.metadata.get("parent_id") {
                    pid.clone()
                } else {
                    // No parent found — skip this child
                    continue;
                }
            };

            let entry = parent_groups
                .entry(parent_id)
                .or_insert((0.0f32, Vec::new()));
            if result.score > entry.0 {
                entry.0 = result.score;
            }
            entry.1.push(result.document.id.clone());
        }

        // Build ExpandedResult from parent groups
        let mut expanded: Vec<ExpandedResult> = Vec::new();
        for (parent_id, (best_score, children)) in parent_groups {
            if let Some(parent_doc) = self.index.parents.get(&parent_id) {
                expanded.push(ExpandedResult {
                    parent: parent_doc.clone(),
                    matched_children: children,
                    score: best_score,
                });
            }
            // If parent not in index, silently skip
        }

        // Sort by score descending
        expanded.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(expanded)
    }
}
