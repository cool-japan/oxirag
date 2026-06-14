//! Grouping of source documents into long retrieval units.

use crate::long_rag::types::{GroupingStrategy, LongRagConfig, LongUnit};
use crate::types::{Document, DocumentId};

// ── Lexical pseudo-embedding ──────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise → hash each token to a bucket with FNV-1a → accumulate
/// per-bucket counts → L2-normalise to the requested `dim`.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        // Deterministic hash: FNV-1a over the lowercased token bytes.
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

// ── Tokeniser ─────────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens of length >= 2.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── Token counting ────────────────────────────────────────────────────────────

/// Count whitespace-delimited words in `text`.
#[must_use]
pub fn token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity between two L2-normalised, equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Truncate `text` to at most `max_tokens` whitespace-delimited words.
///
/// Returns the original string untouched when it already fits.
fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    if max_tokens == 0 {
        return String::new();
    }
    let mut words = text.split_whitespace();
    let mut kept: Vec<&str> = Vec::new();
    for _ in 0..max_tokens {
        match words.next() {
            Some(w) => kept.push(w),
            None => break,
        }
    }
    kept.join(" ")
}

// ── LongUnitGrouper ───────────────────────────────────────────────────────────

/// Assembles source documents into long retrieval units per the configured
/// [`GroupingStrategy`].
///
/// The grouper is the heart of `LongRAG`: it is what turns a fragmented,
/// many-chunk corpus into a small number of long, self-contained units, reducing
/// fragmentation and improving recall during retrieval.
#[derive(Debug, Clone)]
pub struct LongUnitGrouper {
    /// Grouping and embedding configuration.
    config: LongRagConfig,
}

impl LongUnitGrouper {
    /// Create a new grouper with the given configuration.
    #[must_use]
    pub fn new(config: LongRagConfig) -> Self {
        Self { config }
    }

    /// Access the grouper's configuration.
    #[must_use]
    pub fn config(&self) -> &LongRagConfig {
        &self.config
    }

    /// Group `docs` into long units according to the configured strategy.
    ///
    /// - [`GroupingStrategy::ByDocument`] emits one unit per document, truncated
    ///   to the token budget.
    /// - [`GroupingStrategy::FixedTokenWindow`] concatenates documents in order
    ///   and slices the stream into units of at most the token budget.
    /// - [`GroupingStrategy::BySemanticAdjacency`] merges consecutive documents
    ///   whose embeddings are similar until the budget is reached.
    #[must_use]
    pub fn group(&self, docs: &[Document]) -> Vec<LongUnit> {
        match self.config.group_by {
            GroupingStrategy::ByDocument => self.group_by_document(docs),
            GroupingStrategy::FixedTokenWindow => self.group_fixed_window(docs),
            GroupingStrategy::BySemanticAdjacency => self.group_semantic_adjacency(docs),
        }
    }

    /// Build one long unit per document, capping each at the token budget.
    fn group_by_document(&self, docs: &[Document]) -> Vec<LongUnit> {
        let mut units: Vec<LongUnit> = Vec::with_capacity(docs.len());
        for (id, doc) in docs.iter().enumerate() {
            let text = truncate_to_tokens(&doc.content, self.config.max_unit_tokens);
            units.push(self.make_unit(id, text, vec![doc.id.clone()]));
        }
        units
    }

    /// Concatenate documents in order and cut into fixed token windows.
    fn group_fixed_window(&self, docs: &[Document]) -> Vec<LongUnit> {
        // Flatten the corpus into (word, owning-document-id) pairs so each window
        // can faithfully report which documents it spans.
        let mut words: Vec<(&str, &DocumentId)> = Vec::new();
        for doc in docs {
            for word in doc.content.split_whitespace() {
                words.push((word, &doc.id));
            }
        }
        if words.is_empty() {
            return Vec::new();
        }
        let budget = self.config.max_unit_tokens.max(1);
        let mut units: Vec<LongUnit> = Vec::new();
        for (id, window) in words.chunks(budget).enumerate() {
            let text = window.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(" ");
            let mut source_ids: Vec<DocumentId> = Vec::new();
            for (_, doc_id) in window {
                if source_ids.last() != Some(*doc_id) {
                    source_ids.push((*doc_id).clone());
                }
            }
            units.push(self.make_unit(id, text, source_ids));
        }
        units
    }

    /// Merge consecutive documents whose embeddings are similar.
    ///
    /// A running unit grows while (a) the next document's embedding is at least
    /// [`LongRagConfig::similarity_threshold`] similar to the *previous*
    /// document and (b) adding it keeps the unit within the token budget. When
    /// either condition fails, the current unit is sealed and a new one starts.
    fn group_semantic_adjacency(&self, docs: &[Document]) -> Vec<LongUnit> {
        let mut units: Vec<LongUnit> = Vec::new();
        let mut next_id = 0;
        let mut cur_text = String::new();
        let mut cur_sources: Vec<DocumentId> = Vec::new();
        let mut cur_tokens = 0usize;
        let mut prev_emb: Option<Vec<f32>> = None;
        let budget = self.config.max_unit_tokens;

        for doc in docs {
            let doc_emb = embed(&doc.content, self.config.dim);
            let doc_tokens = token_count(&doc.content);
            let similar = prev_emb
                .as_ref()
                .is_some_and(|p| cosine(p, &doc_emb) >= self.config.similarity_threshold);
            let fits = cur_tokens + doc_tokens <= budget;

            if cur_sources.is_empty() {
                // Seed a fresh unit with the first document.
                cur_text.push_str(&doc.content);
                cur_sources.push(doc.id.clone());
                cur_tokens = doc_tokens;
            } else if similar && fits {
                // Extend the current unit with a similar, in-budget neighbour.
                cur_text.push(' ');
                cur_text.push_str(&doc.content);
                cur_sources.push(doc.id.clone());
                cur_tokens += doc_tokens;
            } else {
                // Seal the current unit and start a new one from this document.
                let text = truncate_to_tokens(&cur_text, budget);
                let sources = std::mem::take(&mut cur_sources);
                units.push(self.make_unit(next_id, text, sources));
                next_id += 1;
                cur_text.clone_from(&doc.content);
                cur_sources.push(doc.id.clone());
                cur_tokens = doc_tokens;
            }
            prev_emb = Some(doc_emb);
        }

        if !cur_sources.is_empty() {
            let text = truncate_to_tokens(&cur_text, budget);
            units.push(self.make_unit(next_id, text, cur_sources));
        }
        units
    }

    /// Build a [`LongUnit`] from finished text and its contributing sources.
    fn make_unit(&self, id: usize, text: String, source_ids: Vec<DocumentId>) -> LongUnit {
        let embedding = embed(&text, self.config.dim);
        let token_count = token_count(&text);
        LongUnit {
            id,
            text,
            source_ids,
            token_count,
            embedding,
        }
    }
}
