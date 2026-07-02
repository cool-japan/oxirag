//! The `LightRAG` engine: deterministic entity/relation extraction,
//! incremental graph+vector indexing ([`LightRagIndex`]), and dual-level
//! local/global/hybrid retrieval ([`LightRagEngine`]).

use std::collections::{HashMap, HashSet};

use super::types::{
    LightRagChunk, LightRagConfig, LightRagDualKeywords, LightRagEntity, LightRagEntityKind,
    LightRagError, LightRagIndexStats, LightRagMode, LightRagRelation, LightRagResult,
    relation_dedup_key,
};

// ── Lexical toolkit (deliberately self-contained) ───────────────────────────
//
// Mirrors the pattern used across this crate (see e.g. `searchain::engine`):
// each module owns its own small lexical toolkit rather than importing
// private helpers from a neighbour module.

/// Stopwords excluded from the *content* vocabulary used for high-level
/// keyword extraction and relation-keyword extraction.
const STOPWORDS: &[&str] = &[
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "any",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "get",
    "has",
    "him",
    "his",
    "how",
    "man",
    "new",
    "now",
    "old",
    "see",
    "two",
    "way",
    "who",
    "boy",
    "did",
    "its",
    "let",
    "put",
    "say",
    "she",
    "too",
    "use",
    "that",
    "this",
    "with",
    "from",
    "they",
    "have",
    "were",
    "what",
    "your",
    "when",
    "them",
    "then",
    "than",
    "into",
    "some",
    "such",
    "only",
    "also",
    "been",
    "more",
    "very",
    "will",
    "would",
    "there",
    "their",
    "which",
    "about",
    "could",
    "these",
    "those",
    "does",
    "in",
    "on",
    "at",
    "as",
    "if",
    "so",
    "or",
    "because",
    "since",
    "before",
    "after",
    "during",
    "while",
    "although",
    "however",
    "thus",
    "therefore",
    "a",
    "an",
    "i",
    "my",
    "we",
    "he",
    "between",
    "over",
    "under",
    "through",
    "against",
    "among",
    "within",
    "without",
    "upon",
    "toward",
    "towards",
    "each",
    "every",
    "both",
    "few",
    "most",
    "other",
    "same",
    "again",
];

/// Common sentence-initial function words: a single capitalized token at the
/// very start of a sentence is *not* treated as an entity candidate when its
/// lowercase form appears here, since ordinary sentence-initial
/// capitalization is grammar, not a proper-noun signal.
const SENTENCE_STARTER_STOPWORDS: &[&str] = &[
    "the",
    "this",
    "that",
    "these",
    "those",
    "there",
    "here",
    "it",
    "we",
    "you",
    "they",
    "he",
    "she",
    "his",
    "her",
    "its",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "if",
    "as",
    "so",
    "but",
    "and",
    "or",
    "because",
    "since",
    "before",
    "after",
    "during",
    "while",
    "although",
    "however",
    "thus",
    "therefore",
    "then",
    "also",
    "in",
    "on",
    "at",
    "a",
    "an",
    "i",
    "my",
    "our",
    "your",
    "their",
];

/// Lowercase connector words that may bridge two qualifying capitalized
/// tokens into a single multi-word entity span (e.g. "Bank __of__ America",
/// "University __of__ Tokyo").
const CONNECTOR_WORDS: &[&str] = &["of", "and", "the", "for", "de", "van", "der"];

/// Gazetteer of organizational suffix/marker words.
const ORG_GAZETTEER: &[&str] = &[
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "university",
    "institute",
    "association",
    "organization",
    "agency",
    "council",
    "committee",
    "foundation",
    "ltd",
    "llc",
    "group",
    "society",
    "academy",
    "bank",
    "union",
    "federation",
];

/// Gazetteer of geographic marker words.
const LOCATION_GAZETTEER: &[&str] = &[
    "city",
    "river",
    "mountain",
    "mountains",
    "lake",
    "island",
    "islands",
    "county",
    "state",
    "republic",
    "kingdom",
    "province",
    "ocean",
    "sea",
    "valley",
    "desert",
    "bay",
    "strait",
];

/// Month names, used as a date signal.
const MONTH_NAMES: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Maximum relation-level keywords retained per extracted relation.
const MAX_RELATION_KEYWORDS: usize = 6;

/// Maximum keywords retained per level (low/high) by dual-keyword extraction.
const MAX_KEYWORDS_PER_LEVEL: usize = 8;

/// Tokenize `text` into lowercase alphanumeric tokens of length >= 2.
fn tokenize_lower(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Return `true` when `token` belongs to the content vocabulary: at least
/// three characters and not a stopword.
fn is_content_token(token: &str) -> bool {
    token.chars().count() >= 3 && !STOPWORDS.contains(&token)
}

/// Trim leading/trailing non-alphanumeric characters from `token`.
fn strip_punct(token: &str) -> &str {
    token.trim_matches(|c: char| !c.is_alphanumeric())
}

/// Return `true` when `word`'s first character is uppercase.
fn is_capitalized(word: &str) -> bool {
    word.chars().next().is_some_and(char::is_uppercase)
}

/// Return `true` when `word` is an all-uppercase alphabetic run of at least
/// two letters (an acronym, e.g. "NASA").
fn is_all_caps_acronym(word: &str) -> bool {
    let letters: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
    letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase())
}

/// Return `true` when `token` is a four-digit year in `1000..=2100`.
fn is_year_like(token: &str) -> bool {
    token.len() == 4
        && token.chars().all(|c| c.is_ascii_digit())
        && token
            .parse::<u32>()
            .is_ok_and(|y| (1000..=2100).contains(&y))
}

/// Normalize `name` into a dedup-stable key: lowercased, punctuation
/// stripped to spaces, internal whitespace collapsed, trimmed.
fn canonicalize(name: &str) -> String {
    let cleaned: String = name
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── Entity span extraction ──────────────────────────────────────────────────

/// Return `true` when `word` at `position` (0-based, within its sentence)
/// qualifies as an entity-span token: a year, an all-caps acronym, or a
/// capitalized word that (if sentence-initial) is not a common sentence
/// starter.
fn qualifies_as_entity_token(word: &str, position: usize) -> bool {
    if word.is_empty() {
        return false;
    }
    if is_year_like(word) {
        return true;
    }
    if is_all_caps_acronym(word) {
        return true;
    }
    if !is_capitalized(word) {
        return false;
    }
    if position == 0 {
        !SENTENCE_STARTER_STOPWORDS.contains(&word.to_lowercase().as_str())
    } else {
        true
    }
}

/// Extract candidate entity spans from `sentence`, in order of first
/// appearance, preserving original casing.
///
/// Greedily merges consecutive qualifying tokens into a single span, and
/// allows exactly one [`CONNECTOR_WORDS`] token to bridge two qualifying
/// tokens (so "Bank of America" and "University of Tokyo" extract as single
/// spans rather than two).
fn extract_entity_spans(sentence: &str) -> Vec<String> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let stripped: Vec<&str> = words.iter().map(|w| strip_punct(w)).collect();
    let qualifies: Vec<bool> = stripped
        .iter()
        .enumerate()
        .map(|(i, w)| qualifies_as_entity_token(w, i))
        .collect();

    let mut spans: Vec<String> = Vec::new();
    let mut i = 0;
    while i < stripped.len() {
        if !qualifies[i] {
            i += 1;
            continue;
        }
        let mut span_words: Vec<&str> = vec![stripped[i]];
        let mut j = i + 1;
        loop {
            if j < stripped.len() && qualifies[j] {
                span_words.push(stripped[j]);
                j += 1;
            } else if j + 1 < stripped.len()
                && !stripped[j].is_empty()
                && CONNECTOR_WORDS.contains(&stripped[j].to_lowercase().as_str())
                && qualifies[j + 1]
            {
                span_words.push(stripped[j]);
                span_words.push(stripped[j + 1]);
                j += 2;
            } else {
                break;
            }
        }
        spans.push(span_words.join(" "));
        i = j;
    }
    spans
}

/// Heuristically classify an extracted entity `span` (its full text) using
/// only surface signals: a year/month signal, an organizational or
/// geographic gazetteer hit, a locative preposition immediately preceding
/// the span in `sentence`, or (as a fallback) its word count.
fn infer_entity_kind(span: &str, sentence: &str) -> LightRagEntityKind {
    let lower_words: Vec<String> = span
        .to_lowercase()
        .split_whitespace()
        .map(str::to_string)
        .collect();

    if lower_words
        .iter()
        .any(|w| is_year_like(w) || MONTH_NAMES.contains(&w.as_str()))
    {
        return LightRagEntityKind::Date;
    }
    if lower_words
        .iter()
        .any(|w| ORG_GAZETTEER.contains(&w.as_str()))
    {
        return LightRagEntityKind::Organization;
    }
    if lower_words
        .iter()
        .any(|w| LOCATION_GAZETTEER.contains(&w.as_str()))
    {
        return LightRagEntityKind::Location;
    }

    let sentence_lower = sentence.to_lowercase();
    let span_lower = span.to_lowercase();
    if let Some(pos) = sentence_lower.find(span_lower.as_str()) {
        let before = sentence_lower[..pos].trim_end();
        if before.ends_with(" in") || before.ends_with(" at") || before.ends_with(" from") {
            return LightRagEntityKind::Location;
        }
    }

    if lower_words.len() == 2 {
        return LightRagEntityKind::Person;
    }

    if lower_words.len() == 1 {
        LightRagEntityKind::Concept
    } else {
        LightRagEntityKind::Other
    }
}

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, `?`, and
/// newlines.
fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Extract relation-level keywords from `sentence`: content tokens that are
/// not part of any entity span in `spans`, capped at
/// [`MAX_RELATION_KEYWORDS`], in order of first appearance.
fn extract_relation_keywords(sentence: &str, spans: &[String]) -> Vec<String> {
    let span_tokens: Vec<String> = spans
        .iter()
        .flat_map(|s| {
            s.to_lowercase()
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();

    let mut keywords: Vec<String> = Vec::new();
    for token in tokenize_lower(sentence) {
        if keywords.len() >= MAX_RELATION_KEYWORDS {
            break;
        }
        if !is_content_token(&token) {
            continue;
        }
        if token.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if span_tokens.contains(&token) {
            continue;
        }
        if !keywords.contains(&token) {
            keywords.push(token);
        }
    }
    keywords
}

// ── Combined entity + relation extraction ───────────────────────────────────

/// A single entity occurrence extracted from one chunk of text, before it is
/// merged into a [`LightRagIndex`].
struct ExtractedEntity {
    base_key: String,
    name: String,
    kind: LightRagEntityKind,
    description: String,
}

/// A single relation occurrence extracted from one chunk of text, before it
/// is merged into a [`LightRagIndex`].
struct ExtractedRelation {
    src_base: String,
    dst_base: String,
    description: String,
    keywords: Vec<String>,
}

/// Extract every entity and relation occurrence from `text`.
///
/// Entities are deduplicated *within this call* (one [`ExtractedEntity`] per
/// distinct canonical key, in first-appearance order, its description
/// gathering every distinct sentence it was mentioned in). Relations connect
/// adjacent pairs of distinct entities within the same sentence.
fn extract_entities_and_relations(text: &str) -> (Vec<ExtractedEntity>, Vec<ExtractedRelation>) {
    let sentences = split_sentences(text);

    let mut order: Vec<String> = Vec::new();
    let mut names: HashMap<String, String> = HashMap::new();
    let mut kinds: HashMap<String, LightRagEntityKind> = HashMap::new();
    let mut descriptions: HashMap<String, Vec<String>> = HashMap::new();
    let mut relations: Vec<ExtractedRelation> = Vec::new();

    for sentence in &sentences {
        let spans = extract_entity_spans(sentence);
        let mut sentence_keys: Vec<String> = Vec::new();

        for span in &spans {
            let base_key = canonicalize(span);
            if base_key.is_empty() {
                continue;
            }
            if !sentence_keys.contains(&base_key) {
                sentence_keys.push(base_key.clone());
            }
            if !names.contains_key(&base_key) {
                names.insert(base_key.clone(), span.clone());
                kinds.insert(base_key.clone(), infer_entity_kind(span, sentence));
                order.push(base_key.clone());
            }
            let bucket = descriptions.entry(base_key).or_default();
            let trimmed = sentence.trim().to_string();
            if !bucket.contains(&trimmed) {
                bucket.push(trimmed);
            }
        }

        if sentence_keys.len() >= 2 {
            let keywords = extract_relation_keywords(sentence, &spans);
            for pair in sentence_keys.windows(2) {
                let (a, b) = (pair[0].clone(), pair[1].clone());
                if a == b {
                    continue;
                }
                relations.push(ExtractedRelation {
                    src_base: a,
                    dst_base: b,
                    description: sentence.trim().to_string(),
                    keywords: keywords.clone(),
                });
            }
        }
    }

    let entities: Vec<ExtractedEntity> = order
        .into_iter()
        .map(|base_key| {
            let name = names.remove(&base_key).unwrap_or_else(|| base_key.clone());
            let kind = kinds.remove(&base_key).unwrap_or(LightRagEntityKind::Other);
            let description = descriptions
                .remove(&base_key)
                .map(|v| v.join(" "))
                .unwrap_or_default();
            ExtractedEntity {
                base_key,
                name,
                kind,
                description,
            }
        })
        .collect();

    (entities, relations)
}

// ── Deterministic lexical pseudo-embedding ──────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenize, hash each token to a bucket with FNV-1a, accumulate
/// per-bucket counts, then L2-normalize to `dim`.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        let mut hash: u64 = 14_695_981_039_346_656_037;
        for byte in token.to_lowercase().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (hash as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for value in &mut buckets {
            *value /= norm;
        }
    }
    buckets
}

/// Cosine similarity between two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Dual-keyword extraction ──────────────────────────────────────────────────

/// Extract low-level and high-level keywords from `query`.
///
/// Low-level keys are entity-span text (the same capitalized-run detector
/// used during indexing) plus purely numeric tokens, lowercased. High-level
/// keys are the remaining content tokens (length >= 3, not a stopword, not
/// numeric, not already part of a low-level key). If no low-level key is
/// found at all, the single longest high-level token is promoted to
/// low-level, so a fully lowercase query still yields a specific-ish anchor
/// for local retrieval.
fn compute_dual_keywords(query: &str) -> LightRagDualKeywords {
    let spans = extract_entity_spans(query);

    let mut low_level: Vec<String> = Vec::new();
    let mut low_level_tokens: Vec<String> = Vec::new();
    for span in &spans {
        let lower = span.to_lowercase();
        if !low_level.contains(&lower) {
            low_level.push(lower.clone());
        }
        for token in lower.split_whitespace() {
            if !low_level_tokens.iter().any(|t| t == token) {
                low_level_tokens.push(token.to_string());
            }
        }
    }

    let tokens = tokenize_lower(query);
    for token in &tokens {
        if token.chars().all(|c| c.is_ascii_digit()) && !low_level.contains(token) {
            low_level.push(token.clone());
            low_level_tokens.push(token.clone());
        }
    }

    let mut high_level: Vec<String> = Vec::new();
    for token in &tokens {
        if !is_content_token(token) {
            continue;
        }
        if token.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if low_level_tokens.contains(token) {
            continue;
        }
        if !high_level.contains(token) {
            high_level.push(token.clone());
        }
    }

    if low_level.is_empty()
        && let Some(longest) = high_level.iter().max_by_key(|t| t.chars().count()).cloned()
    {
        high_level.retain(|t| t != &longest);
        low_level.push(longest);
    }

    low_level.truncate(MAX_KEYWORDS_PER_LEVEL);
    high_level.truncate(MAX_KEYWORDS_PER_LEVEL);
    LightRagDualKeywords::new(low_level, high_level)
}

// ── LightRagIndex ─────────────────────────────────────────────────────────────

/// The incrementally-updated `LightRAG` graph + vector hybrid index.
///
/// Holds every indexed [`LightRagEntity`] and [`LightRagRelation`] (keyed for
/// dedup as described on those types), their deterministic pseudo-
/// embeddings, every source [`LightRagChunk`], and an adjacency list mapping
/// each entity key to the relation dedup-keys incident to it (maintained
/// incrementally as relations are first created, so neighborhood lookups are
/// O(degree) rather than O(relation count)).
#[derive(Debug, Clone)]
pub struct LightRagIndex {
    embedding_dim: usize,
    dedup_enabled: bool,
    entities: HashMap<String, LightRagEntity>,
    relations: HashMap<(String, String), LightRagRelation>,
    entity_embeddings: HashMap<String, Vec<f32>>,
    relation_embeddings: HashMap<(String, String), Vec<f32>>,
    chunks: HashMap<String, LightRagChunk>,
    adjacency: HashMap<String, Vec<(String, String)>>,
    uniq_seq: usize,
}

impl LightRagIndex {
    /// Create a new, empty index.
    ///
    /// `embedding_dim` sets the dimension of the deterministic pseudo-
    /// embedding computed for every entity/relation. `dedup_enabled`
    /// controls whether re-extracting an already-known entity/relation
    /// merges into its existing record (`true`) or always creates a
    /// distinct record (`false`, by suffixing storage keys with a
    /// uniquifying sequence number so no lookup can ever collide).
    #[must_use]
    pub fn new(embedding_dim: usize, dedup_enabled: bool) -> Self {
        Self {
            embedding_dim,
            dedup_enabled,
            entities: HashMap::new(),
            relations: HashMap::new(),
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            chunks: HashMap::new(),
            adjacency: HashMap::new(),
            uniq_seq: 0,
        }
    }

    /// Number of distinct indexed entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of distinct indexed relations.
    #[must_use]
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    /// Number of indexed chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// The configured pseudo-embedding dimension.
    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Whether incremental indexing merges re-extracted entities/relations.
    #[must_use]
    pub fn dedup_enabled(&self) -> bool {
        self.dedup_enabled
    }

    /// Look up an entity by its dedup key.
    #[must_use]
    pub fn get_entity(&self, key: &str) -> Option<&LightRagEntity> {
        self.entities.get(key)
    }

    /// Look up a relation by its endpoint keys, in either orientation.
    #[must_use]
    pub fn get_relation(&self, a: &str, b: &str) -> Option<&LightRagRelation> {
        self.relations.get(&relation_dedup_key(a, b))
    }

    /// Look up a chunk by id.
    #[must_use]
    pub fn get_chunk(&self, id: &str) -> Option<&LightRagChunk> {
        self.chunks.get(id)
    }

    /// Iterate over every indexed entity, in unspecified order.
    pub fn entities(&self) -> impl Iterator<Item = &LightRagEntity> {
        self.entities.values()
    }

    /// Iterate over every indexed relation, in unspecified order.
    pub fn relations(&self) -> impl Iterator<Item = &LightRagRelation> {
        self.relations.values()
    }

    /// The pseudo-embedding of an entity's accumulated name + description.
    #[must_use]
    pub fn entity_embedding(&self, key: &str) -> Option<&[f32]> {
        self.entity_embeddings.get(key).map(Vec::as_slice)
    }

    /// The pseudo-embedding of a relation's accumulated description +
    /// keywords.
    #[must_use]
    pub fn relation_embedding(&self, key: &(String, String)) -> Option<&[f32]> {
        self.relation_embeddings.get(key).map(Vec::as_slice)
    }

    /// Relation dedup-keys incident to entity `key`, in insertion order.
    #[must_use]
    pub fn neighbors_of(&self, key: &str) -> &[(String, String)] {
        self.adjacency.get(key).map_or(&[], Vec::as_slice)
    }

    /// Insert a [`LightRagChunk`], extracting and incrementally merging its
    /// entities and relations into the graph.
    ///
    /// # Errors
    ///
    /// - [`LightRagError::EmptyChunkId`] if `chunk.id` is blank.
    /// - [`LightRagError::EmptyChunkText`] if `chunk.text` is blank.
    /// - [`LightRagError::DuplicateChunkId`] if `chunk.id` was already
    ///   inserted.
    pub fn insert_chunk(
        &mut self,
        chunk: LightRagChunk,
    ) -> Result<LightRagIndexStats, LightRagError> {
        if chunk.id.trim().is_empty() {
            return Err(LightRagError::EmptyChunkId);
        }
        if chunk.text.trim().is_empty() {
            return Err(LightRagError::EmptyChunkText);
        }
        if self.chunks.contains_key(&chunk.id) {
            return Err(LightRagError::DuplicateChunkId(chunk.id));
        }

        let (extracted_entities, extracted_relations) = extract_entities_and_relations(&chunk.text);
        let mut stats = LightRagIndexStats::default();
        let mut key_map: HashMap<String, String> = HashMap::new();

        for extracted in &extracted_entities {
            let storage_key = self.next_entity_storage_key(&extracted.base_key);
            key_map.insert(extracted.base_key.clone(), storage_key.clone());
            self.merge_entity(&storage_key, extracted, &chunk.id, &mut stats);
        }

        for extracted in &extracted_relations {
            let src_key = key_map
                .get(&extracted.src_base)
                .cloned()
                .unwrap_or_else(|| extracted.src_base.clone());
            let dst_key = key_map
                .get(&extracted.dst_base)
                .cloned()
                .unwrap_or_else(|| extracted.dst_base.clone());
            self.merge_relation(&src_key, &dst_key, extracted, &chunk.id, &mut stats);
        }

        self.chunks.insert(chunk.id.clone(), chunk);
        Ok(stats)
    }

    /// Compute the unscored structural neighborhood of `seeds`: every entity
    /// reachable within `hop_depth` hops — inclusive of the seeds
    /// themselves — and every relation traversed along the way.
    #[must_use]
    pub fn neighborhood(
        &self,
        seeds: &[String],
        hop_depth: usize,
    ) -> (Vec<String>, Vec<(String, String)>) {
        let mut visited_entities: HashSet<String> = seeds.iter().cloned().collect();
        let mut visited_relations: HashSet<(String, String)> = HashSet::new();
        let mut frontier: Vec<String> = seeds.to_vec();

        for _ in 0..hop_depth {
            let mut next_frontier: Vec<String> = Vec::new();
            for key in &frontier {
                for rel_key in self.neighbors_of(key) {
                    visited_relations.insert(rel_key.clone());
                    if let Some(relation) = self.get_relation(&rel_key.0, &rel_key.1)
                        && let Some(other) = relation.other_end(key)
                        && visited_entities.insert(other.to_string())
                    {
                        next_frontier.push(other.to_string());
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        let mut entity_keys: Vec<String> = visited_entities.into_iter().collect();
        entity_keys.sort_unstable();
        let mut relation_keys: Vec<(String, String)> = visited_relations.into_iter().collect();
        relation_keys.sort_unstable();
        (entity_keys, relation_keys)
    }

    /// Compute the storage key a freshly extracted entity occurrence should
    /// be merged into: the base canonical key when dedup is enabled, or a
    /// freshly uniquified key (guaranteed to be absent from `self.entities`)
    /// when it is not.
    fn next_entity_storage_key(&mut self, base_key: &str) -> String {
        if self.dedup_enabled {
            base_key.to_string()
        } else {
            self.uniq_seq += 1;
            format!("{base_key}#{}", self.uniq_seq)
        }
    }

    /// Insert-or-merge one extracted entity occurrence at `storage_key`, then
    /// recompute its pseudo-embedding from the resulting record.
    fn merge_entity(
        &mut self,
        storage_key: &str,
        extracted: &ExtractedEntity,
        chunk_id: &str,
        stats: &mut LightRagIndexStats,
    ) {
        let is_new = !self.entities.contains_key(storage_key);
        if is_new {
            self.entities.insert(
                storage_key.to_string(),
                LightRagEntity::new(
                    storage_key,
                    extracted.name.clone(),
                    extracted.kind,
                    extracted.description.clone(),
                    chunk_id,
                ),
            );
            stats.entities_added += 1;
        } else if let Some(existing) = self.entities.get_mut(storage_key) {
            if !existing
                .description
                .contains(extracted.description.as_str())
            {
                if existing.description.is_empty() {
                    existing.description.clone_from(&extracted.description);
                } else {
                    existing.description.push(' ');
                    existing.description.push_str(&extracted.description);
                }
            }
            if !existing.source_chunks.iter().any(|c| c == chunk_id) {
                existing.source_chunks.push(chunk_id.to_string());
            }
            existing.occurrence_count += 1;
            stats.entities_merged += 1;
        }

        if let Some(entity) = self.entities.get(storage_key) {
            let text = format!("{} {}", entity.name, entity.description);
            let embedding = embed(&text, self.embedding_dim);
            self.entity_embeddings
                .insert(storage_key.to_string(), embedding);
        }
    }

    /// Insert-or-merge one extracted relation occurrence between `src_key`
    /// and `dst_key`, updating adjacency on first creation, then recompute
    /// its pseudo-embedding from the resulting record.
    fn merge_relation(
        &mut self,
        src_key: &str,
        dst_key: &str,
        extracted: &ExtractedRelation,
        chunk_id: &str,
        stats: &mut LightRagIndexStats,
    ) {
        let dedup_key = relation_dedup_key(src_key, dst_key);
        let is_new = !self.relations.contains_key(&dedup_key);
        if is_new {
            let relation = LightRagRelation::new(
                src_key,
                dst_key,
                extracted.description.clone(),
                extracted.keywords.clone(),
                chunk_id,
            );
            self.relations.insert(dedup_key.clone(), relation);
            self.adjacency
                .entry(src_key.to_string())
                .or_default()
                .push(dedup_key.clone());
            if src_key != dst_key {
                self.adjacency
                    .entry(dst_key.to_string())
                    .or_default()
                    .push(dedup_key.clone());
            }
            stats.relations_added += 1;
        } else if let Some(existing) = self.relations.get_mut(&dedup_key) {
            if !existing
                .description
                .contains(extracted.description.as_str())
            {
                if existing.description.is_empty() {
                    existing.description.clone_from(&extracted.description);
                } else {
                    existing.description.push(' ');
                    existing.description.push_str(&extracted.description);
                }
            }
            for kw in &extracted.keywords {
                if !existing.keywords.contains(kw) {
                    existing.keywords.push(kw.clone());
                }
            }
            if !existing.source_chunks.iter().any(|c| c == chunk_id) {
                existing.source_chunks.push(chunk_id.to_string());
            }
            existing.occurrence_count += 1;
            stats.relations_merged += 1;
        }

        if let Some(relation) = self.relations.get(&dedup_key) {
            let text = format!("{} {}", relation.description, relation.keywords.join(" "));
            let embedding = embed(&text, self.embedding_dim);
            self.relation_embeddings.insert(dedup_key, embedding);
        }
    }
}

// ── Scoring ──────────────────────────────────────────────────────────────────

/// Weight given to vector (embedding) similarity in local (low-level)
/// scoring; the remainder goes to lexical overlap.
const LOCAL_VECTOR_WEIGHT: f32 = 0.6;
/// Weight given to lexical overlap in local (low-level) scoring.
const LOCAL_LEXICAL_WEIGHT: f32 = 0.4;
/// Weight given to vector (embedding) similarity in global (high-level)
/// scoring; the remainder goes to lexical overlap.
const GLOBAL_VECTOR_WEIGHT: f32 = 0.6;
/// Weight given to lexical overlap in global (high-level) scoring.
const GLOBAL_LEXICAL_WEIGHT: f32 = 0.4;
/// Per-hop score decay applied when propagating a seed's score onto its
/// neighborhood during local retrieval.
const NEIGHBOR_SCORE_DECAY: f32 = 0.5;

/// A scored entity key, used internally while ranking retrieval candidates.
#[derive(Debug, Clone)]
struct ScoredKey {
    key: String,
    score: f32,
}

/// A scored relation dedup-key, used internally while ranking retrieval
/// candidates.
#[derive(Debug, Clone)]
struct ScoredPair {
    key: (String, String),
    score: f32,
}

fn sort_scored_keys(items: &mut [ScoredKey]) {
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.key.cmp(&b.key))
    });
}

fn sort_scored_pairs(items: &mut [ScoredPair]) {
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.key.cmp(&b.key))
    });
}

fn to_scored_keys(map: HashMap<String, f32>) -> Vec<ScoredKey> {
    let mut items: Vec<ScoredKey> = map
        .into_iter()
        .map(|(key, score)| ScoredKey { key, score })
        .collect();
    sort_scored_keys(&mut items);
    items
}

fn to_scored_pairs(map: HashMap<(String, String), f32>) -> Vec<ScoredPair> {
    let mut items: Vec<ScoredPair> = map
        .into_iter()
        .map(|(key, score)| ScoredPair { key, score })
        .collect();
    sort_scored_pairs(&mut items);
    items
}

/// Insert `score` for `key` into `map` if `key` is new or `score` improves on
/// the existing value. Returns `true` when the map changed.
fn upsert_max(map: &mut HashMap<String, f32>, key: &str, score: f32) -> bool {
    match map.get(key) {
        Some(&existing) if existing >= score => false,
        _ => {
            map.insert(key.to_string(), score);
            true
        }
    }
}

/// The [`ScoredPair`] analogue of [`upsert_max`].
fn upsert_max_pair(
    map: &mut HashMap<(String, String), f32>,
    key: &(String, String),
    score: f32,
) -> bool {
    match map.get(key) {
        Some(&existing) if existing >= score => false,
        _ => {
            map.insert(key.clone(), score);
            true
        }
    }
}

/// Fuse two scored-entity lists by key, summing scores where a key appears
/// in both — so an entity found by both retrieval paths outranks one found
/// by only one.
fn fuse_scored_keys(a: Vec<ScoredKey>, b: Vec<ScoredKey>) -> Vec<ScoredKey> {
    let mut combined: HashMap<String, f32> = HashMap::new();
    for item in a.into_iter().chain(b) {
        *combined.entry(item.key).or_insert(0.0) += item.score;
    }
    to_scored_keys(combined)
}

/// The [`ScoredPair`] analogue of [`fuse_scored_keys`].
fn fuse_scored_pairs(a: Vec<ScoredPair>, b: Vec<ScoredPair>) -> Vec<ScoredPair> {
    let mut combined: HashMap<(String, String), f32> = HashMap::new();
    for item in a.into_iter().chain(b) {
        *combined.entry(item.key).or_insert(0.0) += item.score;
    }
    to_scored_pairs(combined)
}

/// Lexical tokens used to match an entity against low-level query keywords:
/// its canonical key's whitespace-split tokens.
fn entity_match_tokens(entity: &LightRagEntity) -> Vec<String> {
    entity.key.split_whitespace().map(str::to_string).collect()
}

/// Lexical tokens used to match a relation against high-level query
/// keywords: its own keywords plus its endpoints' key tokens.
fn relation_match_tokens(relation: &LightRagRelation) -> Vec<String> {
    let mut tokens: Vec<String> = relation.keywords.clone();
    tokens.extend(relation.src.split_whitespace().map(str::to_string));
    tokens.extend(relation.dst.split_whitespace().map(str::to_string));
    tokens
}

/// Fraction of `query_keywords` that lexically match (equal to, contain, or
/// are contained by) at least one of `target_tokens`.
fn lexical_match_score(query_keywords: &[String], target_tokens: &[String]) -> f32 {
    if query_keywords.is_empty() {
        return 0.0;
    }
    let hits = query_keywords
        .iter()
        .filter(|kw| {
            target_tokens
                .iter()
                .any(|t| t == *kw || t.contains(kw.as_str()) || kw.contains(t.as_str()))
        })
        .count();
    #[allow(clippy::cast_precision_loss)]
    let score = hits as f32 / query_keywords.len() as f32;
    score
}

fn finalize_entities(scored: Vec<ScoredKey>, index: &LightRagIndex) -> Vec<LightRagEntity> {
    scored
        .into_iter()
        .filter_map(|s| index.get_entity(&s.key).cloned())
        .collect()
}

fn finalize_relations(scored: Vec<ScoredPair>, index: &LightRagIndex) -> Vec<LightRagRelation> {
    scored
        .into_iter()
        .filter_map(|s| index.get_relation(&s.key.0, &s.key.1).cloned())
        .collect()
}

fn collect_source_chunks(
    entities: &[LightRagEntity],
    relations: &[LightRagRelation],
    index: &LightRagIndex,
) -> Vec<LightRagChunk> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<LightRagChunk> = Vec::new();
    let ids = entities
        .iter()
        .flat_map(|e| e.source_chunks.iter())
        .chain(relations.iter().flat_map(|r| r.source_chunks.iter()));
    for id in ids {
        if seen.insert(id.clone())
            && let Some(chunk) = index.get_chunk(id)
        {
            out.push(chunk.clone());
        }
    }
    out
}

/// Look up an entity's display name by key, falling back to the key itself.
fn display_name<'a>(index: &'a LightRagIndex, key: &'a str) -> &'a str {
    index.get_entity(key).map_or(key, |e| e.name.as_str())
}

fn synthesize_context(
    entities: &[LightRagEntity],
    relations: &[LightRagRelation],
    index: &LightRagIndex,
) -> String {
    let mut out = String::new();
    if !entities.is_empty() {
        out.push_str("Entities:\n");
        for entity in entities {
            out.push_str("- ");
            out.push_str(&entity.name);
            out.push_str(" (");
            out.push_str(entity.kind.label());
            out.push_str("): ");
            out.push_str(&entity.description);
            out.push('\n');
        }
    }
    if !relations.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("Relations:\n");
        for relation in relations {
            out.push_str("- ");
            out.push_str(display_name(index, &relation.src));
            out.push_str(" -> ");
            out.push_str(display_name(index, &relation.dst));
            out.push_str(": ");
            out.push_str(&relation.description);
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

// ── LightRagEngine ───────────────────────────────────────────────────────────

/// Drives `LightRAG`'s incremental indexing and dual-level (local / global /
/// hybrid) retrieval over a [`LightRagIndex`] it owns.
#[derive(Debug, Clone)]
pub struct LightRagEngine {
    config: LightRagConfig,
    index: LightRagIndex,
}

impl LightRagEngine {
    /// Create a new engine with an empty index, configured by `config`.
    #[must_use]
    pub fn new(config: LightRagConfig) -> Self {
        let index = LightRagIndex::new(config.embedding_dim, config.dedup_enabled);
        Self { config, index }
    }

    /// The engine's configuration.
    #[must_use]
    pub fn config(&self) -> &LightRagConfig {
        &self.config
    }

    /// The engine's underlying index.
    #[must_use]
    pub fn index(&self) -> &LightRagIndex {
        &self.index
    }

    /// Insert a chunk into the underlying index.
    ///
    /// # Errors
    ///
    /// See [`LightRagIndex::insert_chunk`].
    pub fn insert_chunk(
        &mut self,
        chunk: LightRagChunk,
    ) -> Result<LightRagIndexStats, LightRagError> {
        self.index.insert_chunk(chunk)
    }

    /// Extract [`LightRagDualKeywords`] from `query`: low-level keys
    /// (specific entities / concrete nouns, detected with the same
    /// capitalized-run heuristic used during indexing, plus purely numeric
    /// tokens) and high-level keys (broad thematic content words not already
    /// captured as low-level).
    ///
    /// # Errors
    ///
    /// Returns [`LightRagError::EmptyQuery`] when `query` is blank.
    pub fn extract_dual_keywords(
        &self,
        query: &str,
    ) -> Result<LightRagDualKeywords, LightRagError> {
        if query.trim().is_empty() {
            return Err(LightRagError::EmptyQuery);
        }
        Ok(compute_dual_keywords(query))
    }

    /// Query the index using an explicit retrieval `mode`.
    ///
    /// # Errors
    ///
    /// - [`LightRagError::EmptyQuery`] if `query_text` is blank.
    /// - [`LightRagError::EmptyIndex`] if no entities have been indexed yet.
    pub fn query(
        &self,
        query_text: &str,
        mode: LightRagMode,
    ) -> Result<LightRagResult, LightRagError> {
        if query_text.trim().is_empty() {
            return Err(LightRagError::EmptyQuery);
        }
        if self.index.entity_count() == 0 {
            return Err(LightRagError::EmptyIndex);
        }

        let keywords = compute_dual_keywords(query_text);

        let (scored_entities, scored_relations) = match mode {
            LightRagMode::Local => self.local_pass(&keywords),
            LightRagMode::Global => self.global_pass(&keywords),
            LightRagMode::Hybrid => {
                let (local_entities, local_relations) = self.local_pass(&keywords);
                let (global_entities, global_relations) = self.global_pass(&keywords);
                (
                    fuse_scored_keys(local_entities, global_entities),
                    fuse_scored_pairs(local_relations, global_relations),
                )
            }
        };

        let entities = finalize_entities(scored_entities, &self.index);
        let relations = finalize_relations(scored_relations, &self.index);
        let source_chunks = collect_source_chunks(&entities, &relations, &self.index);
        let context = synthesize_context(&entities, &relations, &self.index);

        Ok(LightRagResult {
            query: query_text.to_string(),
            mode,
            keywords,
            entities,
            relations,
            source_chunks,
            context,
        })
    }

    /// Query using [`LightRagConfig::default_mode`].
    ///
    /// # Errors
    ///
    /// See [`LightRagEngine::query`].
    pub fn query_default(&self, query_text: &str) -> Result<LightRagResult, LightRagError> {
        self.query(query_text, self.config.default_mode)
    }

    /// Local (low-level) retrieval pass: score every entity against the
    /// query's low-level keywords, keep the top
    /// [`LightRagConfig::top_k_entities`], then expand to their
    /// [`LightRagConfig::hop_depth`] neighborhood.
    fn local_pass(&self, keywords: &LightRagDualKeywords) -> (Vec<ScoredKey>, Vec<ScoredPair>) {
        if keywords.low_level.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let query_text = keywords.low_level.join(" ");
        let query_embedding = embed(&query_text, self.index.embedding_dim());

        let mut seeds: Vec<ScoredKey> = self
            .index
            .entities()
            .filter_map(|entity| {
                let vector_score = self
                    .index
                    .entity_embedding(&entity.key)
                    .map_or(0.0, |emb| cosine(&query_embedding, emb));
                let lexical_score =
                    lexical_match_score(&keywords.low_level, &entity_match_tokens(entity));
                let score =
                    LOCAL_VECTOR_WEIGHT * vector_score + LOCAL_LEXICAL_WEIGHT * lexical_score;
                if score > 0.0 {
                    Some(ScoredKey {
                        key: entity.key.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();
        sort_scored_keys(&mut seeds);
        seeds.truncate(self.config.top_k_entities);

        self.expand_scored(&seeds, self.config.hop_depth)
    }

    /// Global (high-level) retrieval pass: score every relation against the
    /// query's high-level keywords, keep the top
    /// [`LightRagConfig::top_k_relations`], then gather their connected
    /// entities.
    fn global_pass(&self, keywords: &LightRagDualKeywords) -> (Vec<ScoredKey>, Vec<ScoredPair>) {
        if keywords.high_level.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let query_text = keywords.high_level.join(" ");
        let query_embedding = embed(&query_text, self.index.embedding_dim());

        let mut top_relations: Vec<ScoredPair> = self
            .index
            .relations()
            .filter_map(|relation| {
                let dedup_key = relation.dedup_key();
                let vector_score = self
                    .index
                    .relation_embedding(&dedup_key)
                    .map_or(0.0, |emb| cosine(&query_embedding, emb));
                let lexical_score =
                    lexical_match_score(&keywords.high_level, &relation_match_tokens(relation));
                let score =
                    GLOBAL_VECTOR_WEIGHT * vector_score + GLOBAL_LEXICAL_WEIGHT * lexical_score;
                if score > 0.0 {
                    Some(ScoredPair {
                        key: dedup_key,
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();
        sort_scored_pairs(&mut top_relations);
        top_relations.truncate(self.config.top_k_relations);

        let mut entity_scores: HashMap<String, f32> = HashMap::new();
        for rel in &top_relations {
            if let Some(relation) = self.index.get_relation(&rel.key.0, &rel.key.1) {
                upsert_max(&mut entity_scores, &relation.src, rel.score);
                upsert_max(&mut entity_scores, &relation.dst, rel.score);
            }
        }

        (to_scored_keys(entity_scores), top_relations)
    }

    /// Expand `seeds` outward through the graph up to `hop_depth` hops,
    /// propagating each seed's score to the entities/relations it reaches
    /// with a per-hop [`NEIGHBOR_SCORE_DECAY`], keeping the best
    /// (provenance-aware) score whenever a node is reachable via multiple
    /// paths.
    fn expand_scored(
        &self,
        seeds: &[ScoredKey],
        hop_depth: usize,
    ) -> (Vec<ScoredKey>, Vec<ScoredPair>) {
        let mut entity_scores: HashMap<String, f32> = HashMap::new();
        for seed in seeds {
            upsert_max(&mut entity_scores, &seed.key, seed.score);
        }
        let mut relation_scores: HashMap<(String, String), f32> = HashMap::new();

        let mut frontier: Vec<(String, f32)> =
            seeds.iter().map(|s| (s.key.clone(), s.score)).collect();

        for _ in 0..hop_depth {
            let mut next_frontier: Vec<(String, f32)> = Vec::new();
            for (key, score) in &frontier {
                for rel_key in self.index.neighbors_of(key) {
                    let propagated = score * NEIGHBOR_SCORE_DECAY;
                    upsert_max_pair(&mut relation_scores, rel_key, propagated);
                    if let Some(relation) = self.index.get_relation(&rel_key.0, &rel_key.1)
                        && let Some(other) = relation.other_end(key)
                        && upsert_max(&mut entity_scores, other, propagated)
                    {
                        next_frontier.push((other.to_string(), propagated));
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        (
            to_scored_keys(entity_scores),
            to_scored_pairs(relation_scores),
        )
    }
}
