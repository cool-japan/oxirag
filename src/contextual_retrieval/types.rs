//! Types for the `contextual_retrieval` module.
use thiserror::Error;
// ── ChunkContext ──────────────────────────────────────────────────────────────
/// Context information prepended to a chunk for contextual retrieval.
#[derive(Debug, Clone, Default)]
pub struct ChunkContext {
    /// Document title, if available.
    pub doc_title: Option<String>,
    /// Extractive summary of the parent document.
    pub doc_summary: String,
    /// Relative position of this chunk in the document (0.0–1.0).
    pub position_ratio: f32,
    /// Gist of the immediately preceding chunk.
    pub preceding_gist: Option<String>,
}
// ── ContextualChunk ───────────────────────────────────────────────────────────
/// A chunk augmented with situating document-level context.
#[derive(Debug, Clone)]
pub struct ContextualChunk {
    /// The original chunk document.
    pub original: crate::types::Document,
    /// Chunk content prepended with situating context.
    pub contextualized_content: String,
    /// The context that was added.
    pub context: ChunkContext,
}
// ── ContextualConfig ──────────────────────────────────────────────────────────
/// Configuration for `ContextualIndexBuilder`.
#[derive(Debug, Clone)]
pub struct ContextualConfig {
    /// Number of sentences in the extractive doc summary. Defaults to `2`.
    pub summary_sentences: usize,
    /// Whether to include the document title. Defaults to `true`.
    pub include_title: bool,
    /// Whether to include positional hints. Defaults to `true`.
    pub include_position: bool,
}
impl Default for ContextualConfig {
    fn default() -> Self {
        Self {
            summary_sentences: 2,
            include_title: true,
            include_position: true,
        }
    }
}
impl ContextualConfig {
    /// Set the number of summary sentences.
    #[must_use]
    pub fn with_summary_sentences(mut self, v: usize) -> Self {
        self.summary_sentences = v;
        self
    }
    /// Set whether to include the title.
    #[must_use]
    pub fn with_include_title(mut self, v: bool) -> Self {
        self.include_title = v;
        self
    }
    /// Set whether to include positional hints.
    #[must_use]
    pub fn with_include_position(mut self, v: bool) -> Self {
        self.include_position = v;
        self
    }
}
// ── ChunkNeighborhood ─────────────────────────────────────────────────────────
/// Neighboring chunks for a given chunk during contextualization.
#[derive(Debug, Clone, Default)]
pub struct ChunkNeighborhood {
    /// The chunk immediately before this one, if any.
    pub preceding: Option<crate::types::Document>,
    /// The chunk immediately after this one, if any.
    pub following: Option<crate::types::Document>,
}
// ── Contextualizer ────────────────────────────────────────────────────────────
/// Sync trait for generating a contextualized string from a chunk.
///
/// # Errors
///
/// Implementations may return [`ContextualRetrievalError`] on configuration errors.
pub trait Contextualizer {
    /// Generate a contextualized content string for `chunk` within `parent`.
    fn contextualize(
        &self,
        chunk: &crate::types::Document,
        parent: &crate::types::Document,
        neighborhood: &ChunkNeighborhood,
    ) -> String;
}
// ── ExtractiveContextualizer ──────────────────────────────────────────────────
/// Extractive contextualizer: prepends title + doc summary + positional hint + preceding gist.
#[derive(Debug, Clone)]
pub struct ExtractiveContextualizer {
    /// Configuration controlling how context is built.
    pub config: ContextualConfig,
}
impl ExtractiveContextualizer {
    /// Create with the given config.
    #[must_use]
    pub fn new(config: ContextualConfig) -> Self {
        Self { config }
    }
}
impl Contextualizer for ExtractiveContextualizer {
    fn contextualize(
        &self,
        chunk: &crate::types::Document,
        parent: &crate::types::Document,
        neighborhood: &ChunkNeighborhood,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        // ── title ──────────────────────────────────────────────────────────
        if self.config.include_title {
            let title = parent
                .metadata
                .get("title")
                .cloned()
                .or_else(|| parent.title.clone())
                .unwrap_or_else(|| "Document".to_string());
            parts.push(format!("Document: {title}."));
        }

        // ── extractive TF-IDF summary ──────────────────────────────────────
        let summary = {
            let n = self.config.summary_sentences.max(1);
            // Split parent content into sentences on '.', '!', '?'
            let sentences: Vec<&str> = parent
                .content
                .split(['.', '!', '?'])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();

            if sentences.len() <= n {
                sentences.join(". ")
            } else {
                // Tokenize all sentences to compute document term frequencies
                let total = sentences.len();
                // Count how many sentences contain each token (sentence-level TF)
                let mut sent_count: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                let tokenized: Vec<Vec<String>> = sentences
                    .iter()
                    .map(|s| {
                        s.split(|c: char| !c.is_alphanumeric())
                            .filter(|t| t.len() >= 2)
                            .map(str::to_lowercase)
                            .collect::<Vec<_>>()
                    })
                    .collect();
                for toks in &tokenized {
                    let unique: std::collections::HashSet<&String> = toks.iter().collect();
                    for tok in unique {
                        *sent_count.entry(tok.clone()).or_insert(0) += 1;
                    }
                }
                // Score each sentence: sum of log(1 + doc_tf / total_sentences)
                #[allow(clippy::cast_precision_loss)]
                let mut scored: Vec<(usize, f64)> = tokenized
                    .iter()
                    .enumerate()
                    .map(|(i, toks)| {
                        let score: f64 = toks
                            .iter()
                            .map(|tok| {
                                let tf = *sent_count.get(tok).unwrap_or(&0);
                                (1.0 + tf as f64 / total as f64).ln()
                            })
                            .sum();
                        (i, score)
                    })
                    .collect();
                // Sort descending by score, then pick first N by original position
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut chosen: Vec<usize> = scored.iter().take(n).map(|(i, _)| *i).collect();
                chosen.sort_unstable();
                chosen
                    .iter()
                    .map(|&i| sentences[i])
                    .collect::<Vec<_>>()
                    .join(". ")
            }
        };
        if !summary.is_empty() {
            parts.push(format!("Context: {summary}."));
        }

        // ── preceding chunk gist ───────────────────────────────────────────
        if let Some(preceding) = &neighborhood.preceding {
            let gist = preceding
                .content
                .split(['.', '!', '?'])
                .map(str::trim)
                .find(|s| !s.is_empty())
                .unwrap_or("")
                .to_string();
            if !gist.is_empty() {
                parts.push(format!("Preceding: {gist}."));
            }
        }

        // ── positional hint ────────────────────────────────────────────────
        if self.config.include_position {
            // position_ratio is carried in via ChunkContext; here we don't have it
            // directly, so we derive a placeholder using the parent content offset.
            // Callers (ContextualIndexBuilder) embed the ratio via build(), which
            // calls contextualize before filling ChunkContext.  We therefore embed
            // a sentinel that build() can overwrite, but for a standalone call we
            // can still provide a meaningful default.
            parts.push("This chunk appears within the document.".to_string());
        }

        // ── final content ──────────────────────────────────────────────────
        parts.push(chunk.content.clone());
        parts.join(" ")
    }
}
// ── ContextualIndexBuilder ────────────────────────────────────────────────────
/// Builds a list of [`ContextualChunk`]s from a parent document and its chunks.
#[derive(Debug, Clone)]
pub struct ContextualIndexBuilder {
    /// Contextualizer implementation to use.
    pub config: ContextualConfig,
}
impl ContextualIndexBuilder {
    /// Create a new builder with the given config.
    #[must_use]
    pub fn new(config: ContextualConfig) -> Self {
        Self { config }
    }
    /// Situate all `chunks` within `parent`.
    ///
    /// # Errors
    ///
    /// Returns [`ContextualRetrievalError::EmptyDocument`] if `parent` content is empty.
    pub fn build(
        &self,
        parent: &crate::types::Document,
        chunks: &[crate::types::Document],
    ) -> Result<Vec<ContextualChunk>, ContextualRetrievalError> {
        if parent.content.trim().is_empty() {
            return Err(ContextualRetrievalError::EmptyDocument);
        }
        if chunks.is_empty() {
            return Err(ContextualRetrievalError::NoChunks);
        }

        let contextualizer = ExtractiveContextualizer::new(self.config.clone());

        // Build extractive doc summary once for all chunks
        let doc_summary = {
            let n = self.config.summary_sentences.max(1);
            let sentences: Vec<&str> = parent
                .content
                .split(['.', '!', '?'])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            if sentences.is_empty() {
                parent.content.clone()
            } else {
                let take = n.min(sentences.len());
                sentences[..take].join(". ")
            }
        };

        let doc_title = parent
            .metadata
            .get("title")
            .cloned()
            .or_else(|| parent.title.clone());

        let n_chunks = chunks.len();
        let mut result = Vec::with_capacity(n_chunks);

        for (idx, chunk) in chunks.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let position_ratio = if n_chunks == 1 {
                0.0_f32
            } else {
                idx as f32 / (n_chunks - 1) as f32
            };

            let preceding = if idx > 0 {
                Some(chunks[idx - 1].clone())
            } else {
                None
            };
            let following = if idx + 1 < n_chunks {
                Some(chunks[idx + 1].clone())
            } else {
                None
            };
            let neighborhood = ChunkNeighborhood {
                preceding,
                following,
            };

            // Generate base contextualised content via the contextualizer.
            // Then replace the generic positional sentinel with a precise ratio.
            let mut base = contextualizer.contextualize(chunk, parent, &neighborhood);
            if self.config.include_position {
                let sentinel = "This chunk appears within the document.";
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let pct = (position_ratio * 100.0).round() as u32;
                let precise = format!("This chunk is at position {pct}% of the document.");
                base = base.replace(sentinel, &precise);
            }

            let preceding_gist = neighborhood.preceding.as_ref().and_then(|p| {
                p.content
                    .split(['.', '!', '?'])
                    .map(str::trim)
                    .find(|s| !s.is_empty())
                    .map(str::to_string)
            });

            let context = ChunkContext {
                doc_title: doc_title.clone(),
                doc_summary: doc_summary.clone(),
                position_ratio,
                preceding_gist,
            };

            result.push(ContextualChunk {
                original: chunk.clone(),
                contextualized_content: base,
                context,
            });
        }

        Ok(result)
    }
}
// ── ContextualRetrievalError ──────────────────────────────────────────────────
/// Errors from the `contextual_retrieval` module.
#[derive(Debug, Error)]
pub enum ContextualRetrievalError {
    /// The parent document content was empty.
    #[error("Parent document content must not be empty")]
    EmptyDocument,
    /// No chunks were provided.
    #[error("At least one chunk is required")]
    NoChunks,
}
