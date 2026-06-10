//! Global and local search engines for the `graph_summarization` module.

use std::collections::HashSet;

#[cfg(feature = "graphrag")]
use crate::layer4_graph::types::{GraphEntity, GraphRelationship};

use super::types::{
    CommunitySummary, GraphSummarizationConfig, GraphSummarizationError, SummaryReport,
};

// ── Jaccard helper ────────────────────────────────────────────────────────────

fn token_set(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() >= 2)
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

// ── GlobalSearchEngine ────────────────────────────────────────────────────────

/// Global search: scores all community summaries against the query,
/// takes the top-K, then reduces them into a single answer.
///
/// This mirrors the Microsoft-GraphRAG map-reduce pipeline.
#[derive(Debug, Clone, Default)]
pub struct GlobalSearchEngine;

impl GlobalSearchEngine {
    /// Create a new [`GlobalSearchEngine`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Search across all community summaries.
    ///
    /// # Errors
    ///
    /// Returns [`GraphSummarizationError::NoCommunities`] when `summaries` is empty.
    pub fn search(
        &self,
        query: &str,
        summaries: &[CommunitySummary],
        config: &GraphSummarizationConfig,
    ) -> Result<SummaryReport, GraphSummarizationError> {
        if summaries.is_empty() {
            return Err(GraphSummarizationError::NoCommunities);
        }

        let q_tokens = token_set(query);

        // Map: score each summary
        let mut scored: Vec<(f32, &CommunitySummary)> = summaries
            .iter()
            .map(|s| {
                let s_tokens = token_set(&s.summary);
                let score = jaccard(&q_tokens, &s_tokens);
                (score, s)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Reduce: pick top_communities, fuse answers
        let top: Vec<_> = scored
            .into_iter()
            .take(config.top_communities)
            .filter(|(score, _)| *score > 0.0)
            .collect();

        let used_communities: Vec<_> = top.iter().map(|(_, s)| s.community_id).collect();
        let answer = top
            .iter()
            .map(|(_, s)| s.summary.clone())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(SummaryReport {
            answer,
            used_communities,
        })
    }
}

// ── LocalSearchEngine ─────────────────────────────────────────────────────────

/// Local search: finds entities matching the query by name, expands to
/// neighbour entities, and gathers the context from their summaries.
#[cfg(feature = "graphrag")]
#[derive(Debug, Clone, Default)]
pub struct LocalSearchEngine;

#[cfg(feature = "graphrag")]
impl LocalSearchEngine {
    /// Create a new [`LocalSearchEngine`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Search by entity neighbourhood.
    ///
    /// # Errors
    ///
    /// Returns [`GraphSummarizationError::NoCommunities`] when `summaries` is empty.
    pub fn search(
        &self,
        query: &str,
        entities: &[GraphEntity],
        relationships: &[GraphRelationship],
        summaries: &[CommunitySummary],
        _config: &GraphSummarizationConfig,
    ) -> Result<SummaryReport, GraphSummarizationError> {
        if summaries.is_empty() {
            return Err(GraphSummarizationError::NoCommunities);
        }

        let q_lower = query.to_lowercase();

        // Seed: entities whose name contains a query token
        let seed_ids: HashSet<String> = entities
            .iter()
            .filter(|e| {
                let name_lower = e.name.to_lowercase();
                q_lower.split_whitespace().any(|t| name_lower.contains(t))
            })
            .map(|e| e.id.as_str().to_string())
            .collect();

        // Expand: collect neighbor entity ids
        let mut neighbor_ids = seed_ids.clone();
        for rel in relationships {
            let src = rel.source_id.as_str().to_string();
            let tgt = rel.target_id.as_str().to_string();
            if seed_ids.contains(&src) {
                neighbor_ids.insert(tgt);
            } else if seed_ids.contains(&tgt) {
                neighbor_ids.insert(src);
            }
        }

        // Gather summaries whose key_entities overlap with the expanded set
        let entity_names: HashSet<String> = entities
            .iter()
            .filter(|e| neighbor_ids.contains(e.id.as_str()))
            .map(|e| e.name.clone())
            .collect();

        let matching_summaries: Vec<&CommunitySummary> = summaries
            .iter()
            .filter(|s| s.key_entities.iter().any(|ke| entity_names.contains(ke)))
            .collect();

        let used_communities: Vec<_> = matching_summaries.iter().map(|s| s.community_id).collect();
        let answer = matching_summaries
            .iter()
            .map(|s| s.summary.clone())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(SummaryReport {
            answer,
            used_communities,
        })
    }
}
