//! The DRIFT search engine: global → follow-ups → local → aggregate.

use std::collections::HashMap;

use crate::types::{Document, DocumentId};

use super::types::{
    CommunityReport, DriftAnswer, DriftConfig, DriftError, DriftStep, DriftStepKind,
};

// ── Lexical pseudo-embedding ────────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise → hash each token to a bucket with FNV-1a → accumulate
/// per-bucket counts → L2-normalise to the requested `dim`. This mirrors the
/// embedding used elsewhere in the crate so that scores are comparable.
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

/// Tokenise `text` into lowercase alphanumeric tokens of length >= 2.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── DriftSearchEngine ───────────────────────────────────────────────────────────

/// `GraphRAG` DRIFT search engine (Microsoft, 2024).
///
/// DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) iterates from
/// a coarse, community-level view down to a fine, passage-level view. A single
/// search performs four phases:
///
/// 1. **Global** — the query is matched against every [`CommunityReport`]
///    summary; the best `top_communities` are selected.
/// 2. **Follow-ups** — sub-queries are synthesized from the salient terms and
///    entities of the selected communities (at most `follow_ups`).
/// 3. **Local** — each follow-up is matched only against the passages belonging
///    to the selected communities.
/// 4. **Aggregate** — the retrieved passages are de-duplicated and ordered, and
///    a summary is synthesized from the selected community reports.
///
/// The engine is fully self-contained and deterministic: it embeds the supplied
/// community summaries and passages with the same FNV-1a lexical embedding and
/// never calls out to a model.
#[derive(Debug, Clone)]
pub struct DriftSearchEngine {
    /// Search configuration.
    config: DriftConfig,
    /// Indexed community reports, in build order.
    communities: Vec<IndexedCommunity>,
    /// All passages keyed by identifier.
    passages: HashMap<DocumentId, Document>,
    /// Per-passage embeddings keyed by identifier.
    passage_embeddings: HashMap<DocumentId, Vec<f32>>,
    /// Whether [`DriftSearchEngine::build`] has run successfully.
    built: bool,
}

/// A community report augmented with its pre-computed summary embedding.
#[derive(Debug, Clone)]
struct IndexedCommunity {
    /// The original report.
    report: CommunityReport,
    /// Pre-computed embedding of the report summary.
    summary_embedding: Vec<f32>,
}

impl DriftSearchEngine {
    /// Create a new, empty engine with the given configuration.
    #[must_use]
    pub fn new(config: DriftConfig) -> Self {
        Self {
            config,
            communities: Vec::new(),
            passages: HashMap::new(),
            passage_embeddings: HashMap::new(),
            built: false,
        }
    }

    /// Build the engine from `communities` and their `passages`.
    ///
    /// Every community summary and every passage is embedded eagerly so that
    /// subsequent searches are pure lookups. Re-building replaces any previous
    /// state.
    ///
    /// # Errors
    ///
    /// Returns [`DriftError::EmptyCommunities`] when `communities` is empty.
    pub fn build(
        &mut self,
        communities: Vec<CommunityReport>,
        passages: &[Document],
    ) -> Result<(), DriftError> {
        if communities.is_empty() {
            return Err(DriftError::EmptyCommunities);
        }

        let dim = self.config.dim;

        self.passages.clear();
        self.passage_embeddings.clear();
        for doc in passages {
            let emb = embed(&doc.content, dim);
            self.passage_embeddings.insert(doc.id.clone(), emb);
            self.passages.insert(doc.id.clone(), doc.clone());
        }

        self.communities = communities
            .into_iter()
            .map(|report| {
                let summary_embedding = embed(&report.summary, dim);
                IndexedCommunity {
                    report,
                    summary_embedding,
                }
            })
            .collect();

        self.built = true;
        Ok(())
    }

    /// Return `true` once the engine has been built.
    #[must_use]
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Number of indexed communities.
    #[must_use]
    pub fn community_count(&self) -> usize {
        self.communities.len()
    }

    /// Number of indexed passages.
    #[must_use]
    pub fn passage_count(&self) -> usize {
        self.passages.len()
    }

    /// Run a DRIFT search for `query`.
    ///
    /// Executes the global phase, generates follow-up sub-queries, runs a local
    /// phase per follow-up, and aggregates the retrieved passages.
    ///
    /// # Errors
    ///
    /// Returns [`DriftError::NotBuilt`] when the engine has not been built and
    /// [`DriftError::EmptyQuery`] when `query` is blank.
    pub fn search(&self, query: &str) -> Result<DriftAnswer, DriftError> {
        if !self.built {
            return Err(DriftError::NotBuilt);
        }
        if query.trim().is_empty() {
            return Err(DriftError::EmptyQuery);
        }

        let mut steps: Vec<DriftStep> = Vec::new();

        // ── Phase 1: global community selection ──────────────────────────────
        let selected = self.global_select(query);
        let global_results: Vec<(DocumentId, f32)> = selected
            .iter()
            .map(|(idx, score)| {
                let id = community_doc_id(self.communities[*idx].report.id);
                (id, *score)
            })
            .collect();
        let communities: Vec<usize> = selected
            .iter()
            .map(|(idx, _)| self.communities[*idx].report.id)
            .collect();
        steps.push(DriftStep::new(query, DriftStepKind::Global, global_results));

        // ── Phase 2: follow-up sub-queries ───────────────────────────────────
        let follow_ups = self.generate_follow_ups(query, &selected);

        // Passages reachable from the selected communities only.
        let local_scope = self.local_scope(&selected);

        // ── Phase 3: local retrieval per follow-up ───────────────────────────
        let mut aggregate: Vec<(DocumentId, f32)> = Vec::new();
        for follow_up in &follow_ups {
            let hits = self.local_search(follow_up, &local_scope);
            for (id, score) in &hits {
                aggregate.push((id.clone(), *score));
            }
            steps.push(DriftStep::new(
                follow_up.clone(),
                DriftStepKind::Local,
                hits,
            ));
        }

        // ── Phase 4: aggregate context + synthesize summary ──────────────────
        let final_context = aggregate_context(aggregate);
        let summary = self.synthesize_summary(&selected);

        Ok(DriftAnswer {
            steps,
            communities,
            final_context,
            summary,
        })
    }

    /// Score every community summary against `query` and return the selected
    /// `(community-index, score)` pairs, best-first.
    fn global_select(&self, query: &str) -> Vec<(usize, f32)> {
        let q_emb = embed(query, self.config.dim);
        let mut scored: Vec<(usize, f32)> = self
            .communities
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, cosine(&q_emb, &c.summary_embedding)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    self.communities[a.0]
                        .report
                        .id
                        .cmp(&self.communities[b.0].report.id)
                })
        });
        scored.truncate(self.config.top_communities);
        scored
    }

    /// Synthesize follow-up sub-queries from the salient terms and entities of
    /// the selected communities.
    ///
    /// Each follow-up combines the original query with one distinct salient term
    /// drawn from the selected communities, so that the local phase explores a
    /// genuinely narrower facet of the question. The number is capped at
    /// `follow_ups`.
    fn generate_follow_ups(&self, query: &str, selected: &[(usize, f32)]) -> Vec<String> {
        if self.config.follow_ups == 0 {
            return Vec::new();
        }

        let query_terms: Vec<String> = tokenize(query);

        // Collect candidate salient terms in deterministic order: entities first
        // (they are the most discriminating), then summary tokens. Skip any term
        // already present in the query so follow-ups add new signal.
        let mut seen: Vec<String> = Vec::new();
        let mut candidates: Vec<String> = Vec::new();
        let push_candidate = |term: &str, seen: &mut Vec<String>, out: &mut Vec<String>| {
            let lower = term.to_lowercase();
            if lower.len() < 2 {
                return;
            }
            if query_terms.contains(&lower) {
                return;
            }
            if seen.contains(&lower) {
                return;
            }
            seen.push(lower.clone());
            out.push(lower);
        };

        for (idx, _) in selected {
            for entity in &self.communities[*idx].report.entities {
                for token in tokenize(entity) {
                    push_candidate(&token, &mut seen, &mut candidates);
                }
            }
        }
        for (idx, _) in selected {
            for token in tokenize(&self.communities[*idx].report.summary) {
                push_candidate(&token, &mut seen, &mut candidates);
            }
        }

        candidates.truncate(self.config.follow_ups);
        candidates
            .into_iter()
            .map(|term| format!("{query} {term}"))
            .collect()
    }

    /// Build the set of passages reachable from the selected communities.
    ///
    /// Returns the passage ids in deterministic (community then listed) order,
    /// de-duplicated, restricted to passages actually held by the engine.
    fn local_scope(&self, selected: &[(usize, f32)]) -> Vec<DocumentId> {
        let mut scope: Vec<DocumentId> = Vec::new();
        for (idx, _) in selected {
            for pid in &self.communities[*idx].report.passage_ids {
                if !self.passages.contains_key(pid) {
                    continue;
                }
                if !scope.contains(pid) {
                    scope.push(pid.clone());
                }
            }
        }
        scope
    }

    /// Search `scope` passages for the ones most similar to `follow_up`.
    fn local_search(&self, follow_up: &str, scope: &[DocumentId]) -> Vec<(DocumentId, f32)> {
        let q_emb = embed(follow_up, self.config.dim);
        let mut hits: Vec<(DocumentId, f32)> = scope
            .iter()
            .filter_map(|pid| {
                self.passage_embeddings
                    .get(pid)
                    .map(|emb| (pid.clone(), cosine(&q_emb, emb)))
            })
            .collect();
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.as_str().cmp(b.0.as_str()))
        });
        hits.truncate(self.config.top_local);
        hits
    }

    /// Concatenate the summaries of the selected communities into a single
    /// synthesized answer summary.
    fn synthesize_summary(&self, selected: &[(usize, f32)]) -> String {
        selected
            .iter()
            .map(|(idx, _)| self.communities[*idx].report.summary.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &DriftConfig {
        &self.config
    }
}

// ── Free helpers ────────────────────────────────────────────────────────────────

/// Synthetic document identifier representing a community in a global step.
fn community_doc_id(community_id: usize) -> DocumentId {
    DocumentId::from_string(format!("community:{community_id}"))
}

/// Aggregate `(id, score)` pairs from every local step into a de-duplicated,
/// relevance-ordered context list.
///
/// Each passage keeps its single best score across all follow-ups; ties are
/// broken by passage id for determinism.
fn aggregate_context(scored: Vec<(DocumentId, f32)>) -> Vec<DocumentId> {
    let mut best: HashMap<String, f32> = HashMap::new();
    for (id, score) in scored {
        best.entry(id.as_str().to_string())
            .and_modify(|s| {
                if score > *s {
                    *s = score;
                }
            })
            .or_insert(score);
    }
    let mut ordered: Vec<(String, f32)> = best.into_iter().collect();
    ordered.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ordered
        .into_iter()
        .map(|(id, _)| DocumentId::from_string(id))
        .collect()
}
