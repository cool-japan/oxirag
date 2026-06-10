//! `KgqaEngine` — knowledge-graph QA via subgraph extraction and fact synthesis.
//!
//! Given a natural-language `query` and a set of [`GraphEntity`] /
//! [`GraphRelationship`] records, the engine:
//!
//! 1. Matches query tokens against entity names.
//! 2. Falls back to "best-guess" entities when no match is found.
//! 3. Expands a subgraph via BFS up to `config.max_hops`.
//! 4. Extracts [`Triple`]s from entity name pairs and relationship types.
//! 5. Builds a template answer and returns a [`KgqaAnswer`].

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::fact_triple::{Triple, TripleConfig, TripleExtractor};
use crate::layer4_graph::types::{GraphEntity, GraphRelationship};

use super::types::{KgqaAnswer, KgqaConfig, KgqaError};

// ── KgqaEngine ────────────────────────────────────────────────────────────────

/// Synchronous knowledge-graph question-answering engine.
///
/// The engine is **pure compute** — no I/O, no async.  It operates entirely
/// over the in-memory entity and relationship slices supplied by the caller.
#[derive(Debug, Clone)]
pub struct KgqaEngine {
    /// Configuration for this engine instance.
    pub config: KgqaConfig,
}

impl KgqaEngine {
    /// Create a new engine with the supplied [`KgqaConfig`].
    #[must_use]
    pub fn new(config: KgqaConfig) -> Self {
        Self { config }
    }

    /// Answer `query` using the provided knowledge graph.
    ///
    /// # Algorithm
    ///
    /// 1. Guard: empty query → [`KgqaError::EmptyQuery`].
    /// 2. Guard: no entities → [`KgqaError::NoEntities`].
    /// 3. Match query tokens against entity names (case-insensitive substring).
    /// 4. Fallback: if no match, pick the top-3 entities by name length as a
    ///    "best guess".
    /// 5. BFS over `relationships` up to `config.max_hops` to build a subgraph.
    /// 6. Extract triples from entity-pair sentences built from the subgraph.
    /// 7. Compute confidence and build a template answer.
    ///
    /// # Errors
    ///
    /// Returns [`KgqaError::EmptyQuery`] or [`KgqaError::NoEntities`] for
    /// structurally invalid inputs.
    pub fn answer(
        &self,
        query: &str,
        entities: &[GraphEntity],
        relationships: &[GraphRelationship],
    ) -> Result<KgqaAnswer, KgqaError> {
        // ── 1–2. guards ───────────────────────────────────────────────────────
        if query.trim().is_empty() {
            return Err(KgqaError::EmptyQuery);
        }
        if entities.is_empty() {
            return Err(KgqaError::NoEntities);
        }

        // ── 3. match query against entity names ───────────────────────────────
        let query_lower = query.to_lowercase();
        let mut matched_ids: Vec<String> = entities
            .iter()
            .filter(|e| {
                let name_lower = e.name.to_lowercase();
                !name_lower.is_empty() && query_lower.contains(name_lower.as_str())
            })
            .map(|e| e.id.clone())
            .collect();

        // ── 4. fallback: best-guess by name length ────────────────────────────
        if matched_ids.is_empty() {
            let mut sorted: Vec<&GraphEntity> = entities.iter().collect();
            sorted.sort_by_key(|b| Reverse(b.name.len()));
            matched_ids = sorted.iter().take(3).map(|e| e.id.clone()).collect();
        }

        // Build an ID → entity lookup.
        let entity_map: HashMap<&str, &GraphEntity> =
            entities.iter().map(|e| (e.id.as_str(), e)).collect();

        // ── 5. BFS subgraph expansion ─────────────────────────────────────────
        let subgraph_entity_ids = bfs_expand(
            &matched_ids,
            relationships,
            self.config.max_hops,
            self.config.top_k,
        );

        // Collect subgraph entities.
        let subgraph_entities: Vec<&GraphEntity> = subgraph_entity_ids
            .iter()
            .filter_map(|id| entity_map.get(id.as_str()))
            .copied()
            .collect();

        let evidence_entity_names: Vec<String> =
            subgraph_entities.iter().map(|e| e.name.clone()).collect();

        // Collect subgraph relationships.
        let subgraph_id_set: HashSet<&str> =
            subgraph_entity_ids.iter().map(String::as_str).collect();

        let subgraph_rels: Vec<&GraphRelationship> = relationships
            .iter()
            .filter(|r| {
                subgraph_id_set.contains(r.source_id.as_str())
                    && subgraph_id_set.contains(r.target_id.as_str())
            })
            .collect();

        // ── 6. extract triples from relationship sentences ────────────────────
        let extractor = TripleExtractor::new(
            TripleConfig::new().with_min_confidence(self.config.min_confidence),
        );

        let mut evidence_triples: Vec<Triple> = Vec::new();
        for rel in &subgraph_rels {
            let src_name = entity_map
                .get(rel.source_id.as_str())
                .map_or(rel.source_id.as_str(), |e| e.name.as_str());
            let tgt_name = entity_map
                .get(rel.target_id.as_str())
                .map_or(rel.target_id.as_str(), |e| e.name.as_str());
            let rel_label = rel.relationship_type.to_string().to_lowercase();
            // Synthesise a pseudo-sentence for the extractor.
            let sentence = format!("{src_name} {rel_label} {tgt_name}.");
            let mut triples = extractor.extract(&sentence, None);
            evidence_triples.append(&mut triples);
        }

        // If no relationship-derived triples, try extracting from entity names directly.
        if evidence_triples.is_empty() && !subgraph_entities.is_empty() {
            for entity in &subgraph_entities {
                let sentence = format!("{} is {}", entity.name, entity.entity_type);
                let mut triples = extractor.extract(&sentence, None);
                evidence_triples.append(&mut triples);
            }
        }

        // ── 7. build answer ───────────────────────────────────────────────────
        let answer = build_answer(
            &evidence_entity_names,
            &evidence_triples,
            &subgraph_rels,
            &entity_map,
        );

        // Confidence: fraction of matched-to-total entities, scaled by triple count.
        #[allow(clippy::cast_precision_loss)]
        let match_ratio = (matched_ids.len() as f32 / entities.len() as f32).min(1.0_f32);
        #[allow(clippy::cast_precision_loss)]
        let triple_factor = if evidence_triples.is_empty() {
            0.1_f32
        } else {
            (evidence_triples.len() as f32 / 10.0_f32).min(0.5_f32) + 0.5_f32
        };
        let confidence = (match_ratio * triple_factor).clamp(0.0, 1.0);

        Ok(KgqaAnswer::new(
            answer,
            evidence_entity_names,
            evidence_triples,
            confidence,
        ))
    }
}

impl Default for KgqaEngine {
    fn default() -> Self {
        Self::new(KgqaConfig::default())
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// BFS from `start_ids` over `relationships` up to `max_hops`, collecting at
/// most `max_entities` unique entity IDs (including the start set).
fn bfs_expand(
    start_ids: &[String],
    relationships: &[GraphRelationship],
    max_hops: usize,
    max_entities: usize,
) -> Vec<String> {
    let mut visited: HashSet<String> = start_ids.iter().cloned().collect();
    let mut queue: VecDeque<(String, usize)> = start_ids.iter().map(|id| (id.clone(), 0)).collect();
    let mut result: Vec<String> = start_ids.to_vec();

    while let Some((current_id, hop)) = queue.pop_front() {
        if hop >= max_hops {
            continue;
        }
        if result.len() >= max_entities {
            break;
        }

        for rel in relationships {
            let neighbour = if rel.source_id == current_id {
                &rel.target_id
            } else if rel.target_id == current_id {
                &rel.source_id
            } else {
                continue;
            };

            if !visited.contains(neighbour) {
                visited.insert(neighbour.clone());
                result.push(neighbour.clone());
                queue.push_back((neighbour.clone(), hop + 1));
                if result.len() >= max_entities {
                    break;
                }
            }
        }
    }

    result
}

/// Build a template answer string from evidence entities, triples, and
/// relationships.
fn build_answer(
    entity_names: &[String],
    triples: &[Triple],
    rels: &[&GraphRelationship],
    entity_map: &HashMap<&str, &GraphEntity>,
) -> String {
    if entity_names.is_empty() {
        return "No relevant entities found in the knowledge graph.".to_string();
    }

    let mut parts: Vec<String> = Vec::new();
    parts.push("Based on the knowledge graph:".to_string());

    // Emit relationship facts first.
    for rel in rels.iter().take(3) {
        let src = entity_map
            .get(rel.source_id.as_str())
            .map_or(rel.source_id.as_str(), |e| e.name.as_str());
        let tgt = entity_map
            .get(rel.target_id.as_str())
            .map_or(rel.target_id.as_str(), |e| e.name.as_str());
        parts.push(format!("{} {} {}.", src, rel.relationship_type, tgt));
    }

    // If no relationship facts, emit triple-derived facts.
    if rels.is_empty() {
        for t in triples.iter().take(3) {
            parts.push(format!("{t}"));
        }
    }

    // Mention all evidence entities.
    if !entity_names.is_empty() {
        parts.push(format!("Relevant entities: {}.", entity_names.join(", ")));
    }

    parts.join(" ")
}
