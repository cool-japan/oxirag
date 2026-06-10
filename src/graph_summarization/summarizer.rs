//! Community summarizer (extractive, GraphRAG-style).

use std::collections::HashMap;

#[cfg(feature = "graphrag")]
use crate::layer4_graph::types::{GraphEntity, GraphRelationship};

#[cfg(feature = "graphrag")]
use crate::graph_community::types::{Community, CommunityGraph};

use super::types::{CommunitySummary, GraphSummarizationConfig};

// ── degree helper ─────────────────────────────────────────────────────────────

/// Build a degree map (`entity_id` → number of incident relationships).
#[cfg(feature = "graphrag")]
fn degree_map(relationships: &[GraphRelationship]) -> HashMap<String, usize> {
    let mut deg: HashMap<String, usize> = HashMap::new();
    for rel in relationships {
        *deg.entry(rel.source_id.clone()).or_insert(0) += 1;
        *deg.entry(rel.target_id.clone()).or_insert(0) += 1;
    }
    deg
}

/// Return the name of the entity with `id`, or `id` as fallback.
#[cfg(feature = "graphrag")]
fn entity_name_for_id(id: &str, entity_map: &HashMap<String, &GraphEntity>) -> String {
    entity_map
        .get(id)
        .map_or_else(|| id.to_string(), |e| e.name.clone())
}

// ── CommunitySummarizer ───────────────────────────────────────────────────────

/// Builds extractive summaries for each community in a [`CommunityGraph`].
///
/// Each summary is derived from relationship phrases involving the most-connected
/// (highest-degree) entities in the community.
#[cfg(feature = "graphrag")]
#[derive(Debug, Clone, Default)]
pub struct CommunitySummarizer;

#[cfg(feature = "graphrag")]
impl CommunitySummarizer {
    /// Create a new [`CommunitySummarizer`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Summarize all communities in `graph`.
    #[must_use]
    pub fn summarize(
        &self,
        graph: &CommunityGraph,
        entities: &[GraphEntity],
        relationships: &[GraphRelationship],
        config: &GraphSummarizationConfig,
    ) -> Vec<CommunitySummary> {
        let entity_map: HashMap<String, &GraphEntity> =
            entities.iter().map(|e| (e.id.clone(), e)).collect();

        let deg = degree_map(relationships);

        graph
            .communities
            .iter()
            .map(|community| {
                self.summarize_community(community, &entity_map, relationships, &deg, config)
            })
            .collect()
    }

    #[allow(clippy::unused_self)]
    fn summarize_community(
        &self,
        community: &Community,
        entity_map: &HashMap<String, &GraphEntity>,
        relationships: &[GraphRelationship],
        deg: &HashMap<String, usize>,
        config: &GraphSummarizationConfig,
    ) -> CommunitySummary {
        use std::cmp::Reverse;

        // Sort members by degree descending → key entities
        let mut member_deg: Vec<(String, usize)> = community
            .members
            .iter()
            .map(|id| {
                let d = deg.get(id.as_str()).copied().unwrap_or(0);
                (id.clone(), d)
            })
            .collect();
        member_deg.sort_by_key(|x| Reverse(x.1));

        let key_entities: Vec<String> = member_deg
            .iter()
            .take(3)
            .map(|(id, _)| {
                entity_map
                    .get(id)
                    .map_or_else(|| id.clone(), |e| e.name.clone())
            })
            .collect();

        let title = key_entities.join(", ");

        // Collect member id set
        let member_ids: std::collections::HashSet<&str> =
            community.members.iter().map(String::as_str).collect();

        // Build sentences from relationships that are internal to this community
        let mut sentences: Vec<String> = Vec::new();
        for rel in relationships {
            let src: &str = &rel.source_id;
            let tgt: &str = &rel.target_id;
            if member_ids.contains(src) && member_ids.contains(tgt) {
                let src_name = entity_name_for_id(src, entity_map);
                let tgt_name = entity_name_for_id(tgt, entity_map);
                sentences.push(format!("{src_name} {} {tgt_name}.", rel.relationship_type));
            }
        }

        sentences.truncate(config.max_summary_sentences);
        let summary = if sentences.is_empty() {
            format!("Community contains: {title}.")
        } else {
            sentences.join(" ")
        };

        CommunitySummary {
            community_id: community.id,
            title,
            summary,
            key_entities,
        }
    }
}
