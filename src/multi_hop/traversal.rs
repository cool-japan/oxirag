//! Multi-hop traversal engine: entity detection, hop expansion, and retrieval.
//!
//! The core logic is split into two pure functions — [`detect_entity_mentions`]
//! and [`expand_hop`] — and an async orchestrator [`MultiHopRetriever`] that
//! calls them in sequence while querying an `Echo` store at each hop.

use std::collections::HashSet;

use crate::layer4_graph::types::{GraphEntity, GraphRelationship};

use super::types::{EntityMention, HopConfig, HopState, MultiHopError, MultiHopResult};

// ── entity-mention detection ──────────────────────────────────────────────────

/// Tokenise `query` and find any [`GraphEntity`] whose `name` appears as a
/// case-insensitive substring of the query.
///
/// Returns one [`EntityMention`] per matched entity; unmatched tokens are
/// ignored.
#[must_use]
pub fn detect_entity_mentions(query: &str, entities: &[GraphEntity]) -> Vec<EntityMention> {
    let query_lower = query.to_lowercase();
    let mut mentions = Vec::new();

    for entity in entities {
        let name_lower = entity.name.to_lowercase();
        if name_lower.is_empty() {
            continue;
        }
        if query_lower.contains(name_lower.as_str()) {
            mentions.push(EntityMention::new(
                entity.name.clone(),
                Some(entity.id.clone()),
            ));
        }
    }

    mentions
}

// ── hop expansion ─────────────────────────────────────────────────────────────

/// Given the current set of entity IDs, traverse `relationships` bidirectionally
/// and return up to `entities_per_hop` new entity IDs that have not yet been
/// visited.
///
/// Both the `source_id → target_id` and `target_id → source_id` directions are
/// explored so the traversal is undirected with respect to the knowledge graph.
#[must_use]
pub fn expand_hop(
    current_entity_ids: &[String],
    relationships: &[GraphRelationship],
    entities_per_hop: usize,
) -> Vec<String> {
    let current_set: HashSet<&str> = current_entity_ids.iter().map(String::as_str).collect();
    let mut new_ids: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for rel in relationships {
        if current_set.contains(rel.source_id.as_str())
            && !current_set.contains(rel.target_id.as_str())
            && !seen.contains(&rel.target_id)
        {
            seen.insert(rel.target_id.clone());
            new_ids.push(rel.target_id.clone());
        }

        if current_set.contains(rel.target_id.as_str())
            && !current_set.contains(rel.source_id.as_str())
            && !seen.contains(&rel.source_id)
        {
            seen.insert(rel.source_id.clone());
            new_ids.push(rel.source_id.clone());
        }

        if new_ids.len() >= entities_per_hop {
            break;
        }
    }

    new_ids.truncate(entities_per_hop);
    new_ids
}

// ── MultiHopRetriever ─────────────────────────────────────────────────────────

/// Orchestrates multi-hop entity-chain traversal over a knowledge graph,
/// issuing semantic searches via an [`Echo`] store at every hop.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
#[derive(Debug, Clone)]
pub struct MultiHopRetriever {
    /// Configuration governing traversal behaviour.
    pub config: HopConfig,
}

impl MultiHopRetriever {
    /// Create a new retriever with the supplied [`HopConfig`].
    #[must_use]
    pub fn new(config: HopConfig) -> Self {
        Self { config }
    }

    /// Execute multi-hop traversal for `query`.
    ///
    /// # Steps
    ///
    /// 1. Guard against empty queries.
    /// 2. Detect entity mentions in the query via [`detect_entity_mentions`].
    /// 3. For each hop up to `config.max_hops`:
    ///    - Search the `echo` store with the names of active entities.
    ///    - Expand to neighbour entities via [`expand_hop`].
    ///    - Record a [`HopState`].
    ///    - Stop early when no new entities are reachable.
    /// 4. Synthesise an answer from the collected document content.
    ///
    /// # Errors
    ///
    /// - [`MultiHopError::EmptyQuery`] — `query` is blank.
    /// - [`MultiHopError::NoEntitiesFound`] — no entity names match the query.
    /// - [`MultiHopError::RetrievalFailed`] — the echo store returned an error.
    pub async fn run<E>(
        &self,
        query: &str,
        entities: &[GraphEntity],
        relationships: &[GraphRelationship],
        echo: &E,
    ) -> Result<MultiHopResult, MultiHopError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        // ── 1. guard ──────────────────────────────────────────────────────────
        if query.trim().is_empty() {
            return Err(MultiHopError::EmptyQuery);
        }

        // ── 2. initial entity detection ───────────────────────────────────────
        let mentions = detect_entity_mentions(query, entities);
        if mentions.is_empty() {
            return Err(MultiHopError::NoEntitiesFound);
        }

        // Collect initial entity IDs and build a lookup from ID → name.
        let id_to_name: std::collections::HashMap<&str, &str> = entities
            .iter()
            .map(|e| (e.id.as_str(), e.name.as_str()))
            .collect();

        let mut current_ids: Vec<String> = mentions
            .iter()
            .filter_map(|m| m.matched_id.clone())
            .collect();
        // Deduplicate
        current_ids.sort_unstable();
        current_ids.dedup();

        let mut visited: HashSet<String> = current_ids.iter().cloned().collect();
        let mut trace: Vec<HopState> = Vec::new();
        let mut all_doc_contents: Vec<String> = Vec::new();

        // ── 3. hop loop ───────────────────────────────────────────────────────
        for hop_idx in 0..self.config.max_hops {
            // Build search query from active entity names
            let entity_names: Vec<&str> = current_ids
                .iter()
                .filter_map(|id| id_to_name.get(id.as_str()).copied())
                .collect();

            let search_query = if entity_names.is_empty() {
                query.to_string()
            } else {
                entity_names.join(" ")
            };

            // Retrieve documents
            let results = echo
                .search(&search_query, self.config.top_k, None)
                .await
                .map_err(|e| MultiHopError::RetrievalFailed(e.to_string()))?;

            let doc_ids: Vec<String> = results
                .iter()
                .map(|r| r.document.id.as_str().to_string())
                .collect();

            for r in &results {
                all_doc_contents.push(r.document.content.clone());
            }

            // Expand to neighbours
            let new_ids = expand_hop(&current_ids, relationships, self.config.entities_per_hop);

            trace.push(HopState::new(hop_idx, current_ids.clone(), doc_ids));

            if new_ids.is_empty() {
                // No further expansion possible — terminate early.
                break;
            }

            // Merge new IDs into visited + current set
            for id in &new_ids {
                visited.insert(id.clone());
            }
            // Next hop starts from the newly discovered entities
            current_ids = new_ids;
        }

        // ── 4. synthesise answer ──────────────────────────────────────────────
        let hops_used = trace.len();
        let answer = synthesise_answer(query, &all_doc_contents);

        Ok(MultiHopResult::new(answer, trace, hops_used))
    }
}

impl Default for MultiHopRetriever {
    fn default() -> Self {
        Self::new(HopConfig::default())
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a simple extractive answer by joining the first sentence of each
/// collected document, capped to keep the output concise.
fn synthesise_answer(query: &str, doc_contents: &[String]) -> String {
    if doc_contents.is_empty() {
        return String::new();
    }

    let snippets: Vec<&str> = doc_contents
        .iter()
        .take(5)
        .map(|c| {
            // Take up to the first sentence boundary or first 200 chars.
            let boundary = c.find(['.', '!', '?']).map_or(c.len().min(200), |p| p + 1);
            &c[..boundary]
        })
        .collect();

    format!("[Query: {}] {}", query, snippets.join(" "))
}
