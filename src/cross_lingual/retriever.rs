//! Cross-lingual retriever: query in one language, documents in another.

use crate::cross_lingual::lexicon::{BilingualLexicon, normalize};
use crate::cross_lingual::types::{CrossLingualConfig, CrossLingualError, CrossLingualHit};
use crate::types::Document;

// ── Lexical pseudo-embedding ────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `tokens`.
///
/// Algorithm: hash each token to a bucket with FNV-1a → accumulate per-bucket
/// counts → L2-normalise to the requested `dim`. Tokens shorter than two bytes
/// are skipped. This mirrors the embedding used elsewhere in the crate so that
/// cosine similarity is comparable across modules.
fn embed_tokens(tokens: &[String], dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in tokens {
        if token.len() < 2 {
            continue;
        }
        // Deterministic hash: FNV-1a over the token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.as_bytes() {
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

/// Tokenise already-normalised `text` into alphanumeric tokens of length >= 2.
fn tokenize_normalized(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_string)
        .collect()
}

// ── Indexed document ────────────────────────────────────────────────────────

/// A document together with its precomputed normalised embedding.
#[derive(Debug, Clone)]
struct IndexedDoc {
    /// The original document, returned verbatim in search hits.
    document: Document,
    /// Pseudo-embedding of the normalised document content.
    embedding: Vec<f32>,
}

// ── CrossLingualRetriever ───────────────────────────────────────────────────

/// An in-memory retriever that matches queries across a language boundary.
///
/// Documents are normalised to a shared diacritic-free form at build time and
/// embedded. At query time the input is normalised, its tokens are translated
/// through the [`BilingualLexicon`] (when enabled), and the expanded token set
/// is embedded and scored by cosine similarity against every document.
#[derive(Debug, Clone)]
pub struct CrossLingualRetriever {
    /// Embedding and expansion configuration.
    config: CrossLingualConfig,
    /// The bilingual lexicon used to translate query tokens.
    lexicon: BilingualLexicon,
    /// Indexed documents with their precomputed embeddings.
    docs: Vec<IndexedDoc>,
}

impl CrossLingualRetriever {
    /// Create a new, empty retriever with the given configuration and lexicon.
    #[must_use]
    pub fn new(config: CrossLingualConfig, lexicon: BilingualLexicon) -> Self {
        Self {
            config,
            lexicon,
            docs: Vec::new(),
        }
    }

    /// Lowercase `text` and strip common Latin diacritics to a shared form.
    ///
    /// For example `é`→`e`, `ü`→`u`, `ñ`→`n`, `ç`→`c`, `ö`→`o`, and `ä`→`a`.
    /// This is the same normalisation applied to documents at build time.
    #[must_use]
    pub fn normalize(text: &str) -> String {
        normalize(text)
    }

    /// Index a batch of documents, replacing any previously built corpus.
    ///
    /// Each document's content is normalised and embedded so that subsequent
    /// searches compare against the shared diacritic-free form.
    pub fn build(&mut self, docs: &[Document]) {
        self.docs = docs
            .iter()
            .map(|doc| {
                let normalized = normalize(&doc.content);
                let tokens = tokenize_normalized(&normalized);
                let embedding = embed_tokens(&tokens, self.config.dim);
                IndexedDoc {
                    document: doc.clone(),
                    embedding,
                }
            })
            .collect();
    }

    /// Expand `query` into normalised tokens plus their lexicon translations.
    ///
    /// The original normalised query tokens always appear first; when
    /// [`CrossLingualConfig::expand_with_lexicon`] is `true`, each token's
    /// (normalised) translations are appended. Duplicates are removed while
    /// preserving first-seen order, yielding a deterministic result.
    #[must_use]
    pub fn expand_query(&self, query: &str) -> Vec<String> {
        let normalized = normalize(query);
        let base = tokenize_normalized(&normalized);
        let mut expanded: Vec<String> = Vec::new();
        for token in &base {
            if !expanded.contains(token) {
                expanded.push(token.clone());
            }
        }
        if self.config.expand_with_lexicon {
            for token in &base {
                for translation in self.lexicon.translate(token) {
                    let normalized_translation = normalize(&translation);
                    if !normalized_translation.is_empty()
                        && !expanded.contains(&normalized_translation)
                    {
                        expanded.push(normalized_translation);
                    }
                }
            }
        }
        expanded
    }

    /// Search for the `top_k` documents most relevant to `query`.
    ///
    /// The query is expanded across languages via [`Self::expand_query`],
    /// embedded, and scored by cosine similarity against the normalised corpus.
    ///
    /// # Errors
    ///
    /// Returns [`CrossLingualError::EmptyQuery`] when `query` is blank and
    /// [`CrossLingualError::EmptyCorpus`] when no documents have been built.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<CrossLingualHit>, CrossLingualError> {
        if query.trim().is_empty() {
            return Err(CrossLingualError::EmptyQuery);
        }
        if self.docs.is_empty() {
            return Err(CrossLingualError::EmptyCorpus);
        }
        let tokens = self.expand_query(query);
        let q_emb = embed_tokens(&tokens, self.config.dim);
        let mut hits: Vec<CrossLingualHit> = self
            .docs
            .iter()
            .map(|d| CrossLingualHit {
                document: d.document.clone(),
                score: cosine(&q_emb, &d.embedding),
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

    /// Number of documents held by the retriever.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Return `true` when no documents have been built.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
}
