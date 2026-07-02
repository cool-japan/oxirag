//! [`TogEngine`] — the Think-on-Graph (`ToG`) bounded beam-search engine.
//!
//! The engine seeds a beam from the query's topic entities and then, for each
//! depth up to `D`, performs the four `ToG` stages: relation exploration
//! (score and prune a node's incident relations), entity exploration (follow
//! kept relations to scored neighbours), pruning (keep the top-`W` extended
//! paths), and a reasoning / sufficiency check (stop early once the beam's
//! triples cover enough of the query). All scoring is deterministic: a
//! self-contained FNV-1a lexical pseudo-embedding blended with token overlap.

use std::collections::{HashMap, HashSet};

use super::types::{
    TogBeamPath, TogConfig, TogDecision, TogError, TogExploration, TogKnowledgeGraph, TogRelation,
    TogResult, TogTriple,
};

// ── Lexical primitives (deterministic, self-contained) ──────────────────────

/// Stopwords excluded from the *content* vocabulary used for lexical overlap
/// and query-coverage decisions.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
    "that", "this", "with", "from", "they", "have", "were", "what", "your", "when", "them", "then",
    "than", "into", "some", "such", "only", "also", "been", "more", "very", "will", "would",
    "there", "their", "which", "about", "could", "these", "those", "does", "where", "whom",
];

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: split on non-alphanumeric boundaries, keep lowercased tokens of
/// length `>= 2`, hash each with FNV-1a into a bucket, accumulate per-bucket
/// counts, and L2-normalise. This mirrors the embedding used elsewhere in the
/// crate so that scores are comparable, and is fully deterministic — identical
/// input always yields an identical vector.
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

/// Cosine similarity between two equal-length, L2-normalised vectors. Returns
/// `0.0` when the lengths differ or either vector is empty.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// The distinct *content* token set of `text`: lowercased alphanumeric tokens
/// of length `>= 3` that are not stopwords and not purely digits.
fn content_tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .filter(|t| {
            t.chars().count() >= 3
                && !STOPWORDS.contains(&t.as_str())
                && !t.chars().all(|c| c.is_ascii_digit())
        })
        .collect()
}

/// Pre-computed query context reused across every scoring call in a run.
struct QueryContext {
    /// The query's lexical pseudo-embedding.
    embedding: Vec<f32>,
    /// The query's content token set.
    tokens: HashSet<String>,
}

impl QueryContext {
    /// Build a query context: embed the query and extract its content tokens.
    fn build(query: &str, dim: usize) -> Self {
        Self {
            embedding: embed(query, dim),
            tokens: content_tokens(query),
        }
    }
}

/// Score `text`'s relevance to the query in `[0.0, 1.0]`.
///
/// Blends a Jaccard lexical overlap of content tokens with the cosine of the
/// pseudo-embeddings: `weight * lexical + (1 - weight) * cosine`. Both
/// components are non-negative, so the result is non-negative — which keeps
/// cumulative path scores monotonically non-decreasing as hops are appended.
#[allow(clippy::cast_precision_loss)]
fn relevance(text: &str, ctx: &QueryContext, dim: usize, lexical_weight: f32) -> f32 {
    let emb = embed(text, dim);
    let cos = cosine(&emb, &ctx.embedding);
    let text_tokens = content_tokens(text);
    let lexical = if ctx.tokens.is_empty() || text_tokens.is_empty() {
        0.0
    } else {
        let intersection = text_tokens
            .iter()
            .filter(|t| ctx.tokens.contains(*t))
            .count();
        let union = text_tokens.union(&ctx.tokens).count();
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    };
    lexical_weight * lexical + (1.0 - lexical_weight) * cos
}

/// A relation label paired with its query-relevance score and the
/// `(neighbour_id, triple)` edges reachable through it. Used internally during
/// relation exploration before pruning.
type ScoredRelationEdges = (String, f32, Vec<(String, TogTriple)>);

/// Blend a relation score and a neighbour-entity score into a single hop score.
/// Both inputs are non-negative, so the output is non-negative.
fn hop_score(relation_score: f32, entity_score: f32) -> f32 {
    0.5 * relation_score + 0.5 * entity_score
}

/// Return `true` when `needle` occurs in `haystack` delimited by word
/// boundaries (the string ends, or non-alphanumeric characters on either
/// side).
///
/// This is stricter than a bare substring test: a short entity name such as
/// `"a"` must not spuriously match *inside* an unrelated word like `"island"`.
/// Multi-word entity names are matched as whole phrases the same way.
fn contains_word_boundary(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let mut start = 0;
    while start <= haystack.len() {
        let Some(pos) = haystack[start..].find(needle) else {
            break;
        };
        let abs = start + pos;
        let after = abs + needle.len();
        let before_ok = haystack[..abs]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());
        let after_ok = haystack[after..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        start = after;
    }
    false
}

/// A deterministic ordering key for a beam path, used to break score ties so
/// that pruning (and hence the whole run) is reproducible.
fn path_key(path: &TogBeamPath) -> String {
    let mut key = path.visited_ids.join(">");
    for hop in &path.hops {
        key.push('#');
        key.push_str(&hop.relation);
    }
    key
}

// ── TogEngine ───────────────────────────────────────────────────────────────

/// The Think-on-Graph (`ToG`, Sun et al. 2023) bounded beam-search engine.
///
/// Given a [`TogKnowledgeGraph`] and a query, [`TogEngine::run`] seeds a beam
/// from the query's topic entities and iteratively deepens it, at each step
/// exploring and pruning relations, exploring and pruning neighbour entities,
/// pruning the extended paths to the beam width, and testing whether the beam's
/// triples already cover enough of the query to answer it.
#[derive(Debug, Clone, Default)]
pub struct TogEngine {
    /// The beam-search configuration.
    pub config: TogConfig,
}

impl TogEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: TogConfig) -> Self {
        Self { config }
    }

    /// Match query terms to graph entities, returning the matched entity ids
    /// ranked by descending relevance (ties broken by ascending id).
    ///
    /// An entity matches when its (lowercased) name occurs as a substring of
    /// the query, or when a content token of its name is a content token of the
    /// query.
    #[must_use]
    pub fn match_topic_entities(&self, query: &str, graph: &TogKnowledgeGraph) -> Vec<String> {
        let ctx = QueryContext::build(query, self.config.embedding_dim);
        self.scored_topic_entities(query, graph, &ctx)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// The topic-entity match with relevance scores attached.
    fn scored_topic_entities(
        &self,
        query: &str,
        graph: &TogKnowledgeGraph,
        ctx: &QueryContext,
    ) -> Vec<(String, f32)> {
        let query_lower = query.to_lowercase();
        let mut matched: Vec<(String, f32)> = Vec::new();
        for entity in graph.entities() {
            let name_lower = entity.name.to_lowercase();
            if name_lower.is_empty() {
                continue;
            }
            let substring_hit = contains_word_boundary(&query_lower, &name_lower);
            let token_hit = {
                let name_tokens = content_tokens(&entity.name);
                !name_tokens.is_empty() && name_tokens.iter().any(|t| ctx.tokens.contains(t))
            };
            if substring_hit || token_hit {
                let score = relevance(
                    &entity.name,
                    ctx,
                    self.config.embedding_dim,
                    self.config.lexical_weight,
                );
                matched.push((entity.id.clone(), score));
            }
        }
        matched.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        matched
    }

    /// Run the full Think-on-Graph pipeline for `query` over `graph`.
    ///
    /// 1. Validate the config, query, and graph.
    /// 2. Match topic entities and seed the beam (pruned to width `W`).
    /// 3. For each depth `1..=D`: relation exploration + pruning, entity
    ///    exploration + pruning, beam pruning, then a sufficiency check that
    ///    stops early once the beam's triples cover
    ///    `sufficiency_threshold` of the query's content terms.
    /// 4. Synthesize the answer, evidence triples, and trace.
    ///
    /// # Errors
    ///
    /// - [`TogError::InvalidConfig`] when the configuration is invalid.
    /// - [`TogError::EmptyQuery`] when `query` is blank.
    /// - [`TogError::EmptyGraph`] when the graph has no entities.
    /// - [`TogError::NoTopicEntity`] when no entity matches the query.
    pub fn run(&self, query: &str, graph: &TogKnowledgeGraph) -> Result<TogResult, TogError> {
        self.config.validate()?;
        if query.trim().is_empty() {
            return Err(TogError::EmptyQuery);
        }
        if graph.is_empty() {
            return Err(TogError::EmptyGraph);
        }

        let ctx = QueryContext::build(query, self.config.embedding_dim);

        // ── Topic-entity initialisation ─────────────────────────────────────
        let scored_topics = self.scored_topic_entities(query, graph, &ctx);
        if scored_topics.is_empty() {
            return Err(TogError::NoTopicEntity);
        }
        let topic_entity_ids: Vec<String> =
            scored_topics.iter().map(|(id, _)| id.clone()).collect();

        // Seed the beam (already ranked by score) and prune to width `W`.
        let mut beam: Vec<TogBeamPath> = scored_topics
            .into_iter()
            .map(|(id, score)| TogBeamPath::seed(id, score))
            .collect();
        beam.truncate(self.config.beam_width);

        let mut explorations: Vec<TogExploration> = Vec::new();
        let mut stopped_early = false;
        let mut depth_reached: usize = 0;
        let mut final_coverage = coverage(&ctx.tokens, &beam, graph);

        // ── Iterative depth-bounded beam search ─────────────────────────────
        for depth in 1..=self.config.max_depth {
            let (candidates, scored_relations) = self.expand_beam(&beam, graph, &ctx);
            let candidates_generated = candidates.len();

            if candidates.is_empty() {
                let decision = TogDecision::Exhausted {
                    depth,
                    coverage: final_coverage,
                };
                explorations.push(TogExploration::new(
                    depth,
                    0,
                    beam.clone(),
                    scored_relations,
                    decision,
                ));
                break;
            }

            beam = prune_beam(candidates, self.config.beam_width);
            depth_reached = depth;
            final_coverage = coverage(&ctx.tokens, &beam, graph);

            let decision = if final_coverage >= self.config.sufficiency_threshold {
                stopped_early = true;
                TogDecision::Sufficient {
                    depth,
                    coverage: final_coverage,
                }
            } else if depth == self.config.max_depth {
                TogDecision::MaxDepthReached {
                    depth,
                    coverage: final_coverage,
                }
            } else {
                TogDecision::Continue {
                    depth,
                    coverage: final_coverage,
                }
            };
            let terminal = decision.is_terminal();
            explorations.push(TogExploration::new(
                depth,
                candidates_generated,
                beam.clone(),
                scored_relations,
                decision,
            ));
            if terminal {
                break;
            }
        }

        // ── Answer synthesis ────────────────────────────────────────────────
        let evidence = collect_evidence(&beam);
        let answer = synthesize_answer(query, &beam, graph);

        Ok(TogResult {
            query: query.to_string(),
            topic_entity_ids,
            paths: beam,
            evidence,
            answer,
            stopped_early,
            depth_reached,
            coverage: final_coverage,
            explorations,
        })
    }

    /// Expand every path in `beam` by one hop, returning all candidate extended
    /// paths together with the relations scored during this depth (deduplicated
    /// by label, keeping the highest score seen).
    fn expand_beam(
        &self,
        beam: &[TogBeamPath],
        graph: &TogKnowledgeGraph,
        ctx: &QueryContext,
    ) -> (Vec<TogBeamPath>, Vec<(TogRelation, f32)>) {
        let mut candidates: Vec<TogBeamPath> = Vec::new();
        let mut relation_scores: HashMap<String, f32> = HashMap::new();

        for path in beam {
            let (expansions, scored) = self.expand_path(path, graph, ctx);
            candidates.extend(expansions);
            for (label, score) in scored {
                relation_scores
                    .entry(label)
                    .and_modify(|best| {
                        if score > *best {
                            *best = score;
                        }
                    })
                    .or_insert(score);
            }
        }

        let mut scored_relations: Vec<(TogRelation, f32)> = relation_scores
            .into_iter()
            .map(|(label, score)| (TogRelation::new(label), score))
            .collect();
        scored_relations.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.label().cmp(b.0.label()))
        });

        (candidates, scored_relations)
    }

    /// Expand a single path: relation exploration + pruning, then entity
    /// exploration + pruning. Returns the extended paths and every relation
    /// considered (with its score), so the caller can record the full set.
    fn expand_path(
        &self,
        path: &TogBeamPath,
        graph: &TogKnowledgeGraph,
        ctx: &QueryContext,
    ) -> (Vec<TogBeamPath>, Vec<(String, f32)>) {
        let current = path.current_entity_id();

        // Group incident edges by relation label; each entry lists the
        // (neighbour_id, triple) reachable via that label. Both out- and
        // in-edges are explored, so traversal is bidirectional.
        let mut by_relation: HashMap<String, Vec<(String, TogTriple)>> = HashMap::new();
        for triple in graph.outgoing_triples(current) {
            by_relation
                .entry(triple.relation.clone())
                .or_default()
                .push((triple.tail.clone(), triple.clone()));
        }
        for triple in graph.incoming_triples(current) {
            by_relation
                .entry(triple.relation.clone())
                .or_default()
                .push((triple.head.clone(), triple.clone()));
        }

        // ── Relation exploration: score every label, sort, prune to top-k. ──
        let mut scored_relations: Vec<ScoredRelationEdges> = by_relation
            .into_iter()
            .map(|(label, neighbours)| {
                let score = relevance(
                    &label,
                    ctx,
                    self.config.embedding_dim,
                    self.config.lexical_weight,
                );
                (label, score, neighbours)
            })
            .collect();
        scored_relations.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let considered: Vec<(String, f32)> = scored_relations
            .iter()
            .map(|(label, score, _)| (label.clone(), *score))
            .collect();

        // ── Entity exploration: follow kept relations to scored neighbours. ─
        let mut expansions: Vec<TogBeamPath> = Vec::new();
        for (_label, relation_score, neighbours) in scored_relations
            .into_iter()
            .filter(|(_, score, _)| *score >= self.config.min_relation_score)
            .take(self.config.relations_per_expansion)
        {
            let mut relation_candidates: Vec<TogBeamPath> = Vec::new();
            for (neighbour_id, triple) in neighbours {
                if path.contains_entity(&neighbour_id) {
                    continue; // cycle avoidance
                }
                let entity_score = graph.entity_name(&neighbour_id).map_or(0.0, |name| {
                    relevance(
                        name,
                        ctx,
                        self.config.embedding_dim,
                        self.config.lexical_weight,
                    )
                });
                let added = hop_score(relation_score, entity_score);
                relation_candidates.push(path.extended(triple, neighbour_id, added));
            }
            relation_candidates.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| path_key(a).cmp(&path_key(b)))
            });
            relation_candidates.truncate(self.config.entities_per_relation);
            expansions.extend(relation_candidates);
        }

        (expansions, considered)
    }
}

/// Prune `candidates` to the top-`beam_width` by descending score, breaking
/// ties deterministically on the path key.
fn prune_beam(mut candidates: Vec<TogBeamPath>, beam_width: usize) -> Vec<TogBeamPath> {
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| path_key(a).cmp(&path_key(b)))
    });
    candidates.truncate(beam_width);
    candidates
}

/// Coverage of the query's content `tokens` by the beam paths' evidence: the
/// fraction of query tokens that appear among the entity names and relation
/// labels along the beam. Returns `0.0` when the query has no content tokens.
#[allow(clippy::cast_precision_loss)]
fn coverage(tokens: &HashSet<String>, beam: &[TogBeamPath], graph: &TogKnowledgeGraph) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut evidence_tokens: HashSet<String> = HashSet::new();
    for path in beam {
        for id in path.entities() {
            if let Some(name) = graph.entity_name(id) {
                evidence_tokens.extend(content_tokens(name));
            }
        }
        for hop in path.triples() {
            evidence_tokens.extend(content_tokens(&hop.relation));
        }
    }
    let covered = tokens
        .iter()
        .filter(|t| evidence_tokens.contains(*t))
        .count();
    covered as f32 / tokens.len() as f32
}

/// The de-duplicated union of every beam path's hops, in beam order.
fn collect_evidence(beam: &[TogBeamPath]) -> Vec<TogTriple> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut evidence: Vec<TogTriple> = Vec::new();
    for path in beam {
        for hop in path.triples() {
            let key = (hop.head.clone(), hop.relation.clone(), hop.tail.clone());
            if seen.insert(key) {
                evidence.push(hop.clone());
            }
        }
    }
    evidence
}

/// Render a path as `topic -[rel]-> n1 -[rel2]-> n2`, resolving ids to names.
fn describe_path(path: &TogBeamPath, graph: &TogKnowledgeGraph) -> String {
    let names: Vec<&str> = path
        .entities()
        .iter()
        .map(|id| graph.entity_name(id).unwrap_or(id.as_str()))
        .collect();
    let mut rendered = names.first().copied().unwrap_or("").to_string();
    for (idx, hop) in path.triples().iter().enumerate() {
        let next = names.get(idx + 1).copied().unwrap_or("?");
        rendered.push_str(" -[");
        rendered.push_str(&hop.relation);
        rendered.push_str("]-> ");
        rendered.push_str(next);
    }
    rendered
}

/// Synthesize a natural-language answer from the best beam path.
fn synthesize_answer(query: &str, beam: &[TogBeamPath], graph: &TogKnowledgeGraph) -> String {
    match beam.first() {
        None => format!("No reasoning path was found for query: {query}"),
        Some(best) => {
            let target_id = best.current_entity_id();
            let target = graph.entity_name(target_id).unwrap_or(target_id);
            if best.is_seed() {
                format!(
                    "Answer: {target}. No supporting relations were found in the knowledge graph (query: {query})"
                )
            } else {
                let chain = describe_path(best, graph);
                format!("Answer: {target}. Reasoning path: {chain} (query: {query})")
            }
        }
    }
}
