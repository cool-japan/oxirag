//! The `MemoRAG` engine: corpus memory, clue generation, and clue-fused retrieval.

use std::collections::HashMap;

use crate::memorag::memory::{build_gist, embed, tokenize};
use crate::memorag::types::{MemoHit, MemoRagConfig, MemoRagError, MemoryGist};
use crate::types::{Document, DocumentId};

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity between two L2-normalised, equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── MemoRagEngine ─────────────────────────────────────────────────────────────

/// A `MemoRAG` engine (Qian et al. 2024).
///
/// The engine forms a single, corpus-wide [`MemoryGist`] — a compressed *global
/// memory* of the whole corpus. At query time it uses that memory to generate
/// retrieval **clues**: surrogate sub-queries that splice the user's query
/// together with the gist's key terms, bridging the lexical gap between the
/// query and the evidence. Each clue retrieves independently and the results are
/// fused (max score per document), so a document can be surfaced by whichever
/// clue best matches it.
///
/// This is deliberately distinct from agent/conversation memory: the memory here
/// is over the *corpus*, and it is consumed to drive retrieval rather than to
/// recall past dialogue turns.
#[derive(Debug, Clone)]
pub struct MemoRagEngine {
    /// Memory, clue, and embedding configuration.
    config: MemoRagConfig,
    /// The indexed corpus of evidence documents.
    corpus: Vec<Document>,
    /// The corpus-wide global memory, populated by [`build_memory`].
    ///
    /// [`build_memory`]: MemoRagEngine::build_memory
    gist: Option<MemoryGist>,
}

impl MemoRagEngine {
    /// Create a new, empty engine with the given configuration.
    #[must_use]
    pub fn new(config: MemoRagConfig) -> Self {
        Self {
            config,
            corpus: Vec::new(),
            gist: None,
        }
    }

    /// Form the corpus-wide global memory from `corpus`.
    ///
    /// Computes corpus-wide term-frequency salience, selecting the top
    /// `gist_sentences` sentences as the summary and the top `key_terms` terms as
    /// the memory key terms. The corpus is retained for subsequent retrieval.
    ///
    /// # Errors
    ///
    /// Returns [`MemoRagError::EmptyCorpus`] when `corpus` is empty.
    pub fn build_memory(&mut self, corpus: &[Document]) -> Result<(), MemoRagError> {
        if corpus.is_empty() {
            return Err(MemoRagError::EmptyCorpus);
        }
        self.corpus = corpus.to_vec();
        self.gist = Some(build_gist(
            corpus,
            self.config.gist_sentences,
            self.config.key_terms,
        ));
        Ok(())
    }

    /// Access the corpus-wide global memory, if it has been built.
    #[must_use]
    pub fn gist(&self) -> Option<&MemoryGist> {
        self.gist.as_ref()
    }

    /// Generate up to `num_clues` retrieval clues for `query`.
    ///
    /// Each clue is the original `query` augmented with one *distinct* gist key
    /// term, prioritising terms most relevant to the query. Relevance is scored
    /// by how often a key term co-occurs (lexically) with the query tokens across
    /// the corpus memory; remaining slots are filled by overall key-term salience.
    /// Every returned clue therefore contains the query verbatim, and no two
    /// clues share the same augmenting term.
    ///
    /// The first clue is always the bare query so that direct matches are never
    /// lost when the gist key terms are off-topic.
    ///
    /// # Errors
    ///
    /// Returns [`MemoRagError::EmptyQuery`] when `query` is blank and
    /// [`MemoRagError::MemoryNotBuilt`] when the memory has not been built.
    pub fn generate_clues(&self, query: &str) -> Result<Vec<String>, MemoRagError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(MemoRagError::EmptyQuery);
        }
        let gist = self.gist.as_ref().ok_or(MemoRagError::MemoryNotBuilt)?;

        if self.config.num_clues == 0 {
            return Ok(Vec::new());
        }

        // The bare query is always the first clue.
        let mut clues: Vec<String> = vec![trimmed.to_string()];

        // Rank the gist key terms by relevance to the query, then splice the top
        // distinct ones onto the query to form the remaining clues.
        let query_tokens: Vec<String> = tokenize(trimmed);
        for term in self.rank_key_terms(gist, &query_tokens) {
            if clues.len() >= self.config.num_clues {
                break;
            }
            // Skip terms already literally present in the query to keep clues
            // distinct and informative.
            if query_tokens.iter().any(|t| t == &term) {
                continue;
            }
            clues.push(format!("{trimmed} {term}"));
        }

        clues.truncate(self.config.num_clues);
        Ok(clues)
    }

    /// Rank the gist key terms by relevance to the query tokens.
    ///
    /// A key term's relevance is its co-occurrence count with the query tokens
    /// inside sentences of the corpus memory; ties (including zero co-occurrence)
    /// fall back to the term's original salience rank in the gist.
    fn rank_key_terms(&self, gist: &MemoryGist, query_tokens: &[String]) -> Vec<String> {
        // Co-occurrence: count corpus sentences containing both the key term and
        // at least one query token.
        let mut cooccurrence: HashMap<&str, usize> = HashMap::new();
        for term in &gist.key_terms {
            cooccurrence.insert(term.as_str(), 0);
        }
        for doc in &self.corpus {
            for sentence in crate::memorag::memory::split_sentences(&doc.content) {
                let toks = tokenize(&sentence);
                let has_query_token = toks.iter().any(|t| query_tokens.contains(t));
                if !has_query_token {
                    continue;
                }
                for term in &gist.key_terms {
                    if toks.iter().any(|t| t == term)
                        && let Some(c) = cooccurrence.get_mut(term.as_str())
                    {
                        *c += 1;
                    }
                }
            }
        }

        // Stable order by descending co-occurrence, then by original gist rank.
        let mut ranked: Vec<(usize, &String)> = gist.key_terms.iter().enumerate().collect();
        ranked.sort_by(|a, b| {
            let ca = cooccurrence.get(a.1.as_str()).copied().unwrap_or(0);
            let cb = cooccurrence.get(b.1.as_str()).copied().unwrap_or(0);
            cb.cmp(&ca).then_with(|| a.0.cmp(&b.0))
        });
        ranked.into_iter().map(|(_, t)| t.clone()).collect()
    }

    /// Retrieve the `top_k` evidence documents for `query` using memory clues.
    ///
    /// Each generated clue is embedded with the lexical scheme and scored by
    /// cosine similarity against every corpus document. Scores are fused across
    /// clues by taking the **maximum** score per document, and the clue that
    /// achieved that maximum is recorded on the resulting [`MemoHit`]. Hits are
    /// sorted by descending fused score and truncated to `top_k`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoRagError::EmptyQuery`] when `query` is blank,
    /// [`MemoRagError::MemoryNotBuilt`] when the memory has not been built, and
    /// [`MemoRagError::EmptyCorpus`] when no documents are indexed.
    pub fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<MemoHit>, MemoRagError> {
        if query.trim().is_empty() {
            return Err(MemoRagError::EmptyQuery);
        }
        if self.gist.is_none() {
            return Err(MemoRagError::MemoryNotBuilt);
        }
        if self.corpus.is_empty() {
            return Err(MemoRagError::EmptyCorpus);
        }

        let clues = self.generate_clues(query)?;
        // Pre-embed clues once.
        let clue_embeddings: Vec<(String, Vec<f32>)> = clues
            .into_iter()
            .map(|clue| {
                let emb = embed(&clue, self.config.dim);
                (clue, emb)
            })
            .collect();

        // Fuse across clues by max score per document.
        let mut best: HashMap<&DocumentId, (f32, &str)> = HashMap::new();
        for doc in &self.corpus {
            let doc_emb = embed(&doc.content, self.config.dim);
            for (clue, clue_emb) in &clue_embeddings {
                let score = cosine(clue_emb, &doc_emb);
                let entry = best.entry(&doc.id).or_insert((f32::NEG_INFINITY, ""));
                if score > entry.0 {
                    *entry = (score, clue.as_str());
                }
            }
        }

        let mut hits: Vec<MemoHit> = self
            .corpus
            .iter()
            .filter_map(|doc| {
                best.get(&doc.id).map(|(score, clue)| MemoHit {
                    document: doc.clone(),
                    score: *score,
                    clue: (*clue).to_string(),
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.document.id.as_str().cmp(b.document.id.as_str()))
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Number of documents indexed in the corpus.
    #[must_use]
    pub fn len(&self) -> usize {
        self.corpus.len()
    }

    /// Return `true` when no documents have been indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.corpus.is_empty()
    }
}
