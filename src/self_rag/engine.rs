//! `SelfRagEngine` — the Self-RAG orchestration engine.
//!
//! Implements the Self-RAG loop:
//! 1. Segment the generation into sentences.
//! 2. For each segment decide whether to retrieve (based on heuristic signals).
//! 3. Retrieve top-k documents and grade them for relevance.
//! 4. Compose the answer from supported segments.
//! 5. Assign overall utility token.

use super::reflect::Reflector;
use super::types::{ReflectionToken, SegmentCritique, SelfRagConfig, SelfRagError, SelfRagOutput};

// ── sentence splitter (local copy) ────────────────────────────────────────────

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        current.push(chars[i]);
        if i + 1 < n {
            let end_punct = chars[i] == '.' || chars[i] == '?' || chars[i] == '!';
            let next_space = chars[i + 1] == ' ';
            if end_punct && next_space {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current = String::new();
                i += 2;
                continue;
            }
        }
        if i + 1 < n && chars[i] == '\n' && chars[i + 1] == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
            i += 2;
            continue;
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

// ── retrieve trigger ──────────────────────────────────────────────────────────

/// Heuristic: should we retrieve for this segment?
///
/// Returns `true` when the segment contains uncertainty markers or references
/// that suggest external knowledge is needed.
fn should_retrieve(segment: &str) -> bool {
    let lower = segment.to_lowercase();
    let triggers = [
        "according to",
        "research shows",
        "studies show",
        "evidence suggests",
        "it is known",
        "experts say",
        "data shows",
        "statistics",
        "published",
        "reported",
        "discovered",
        "founded",
        "invented",
        "created",
        "located",
        "born",
        "died",
        "when",
        "where",
        "who",
        "how many",
        "how much",
        "what year",
        "which",
    ];
    triggers.iter().any(|t| lower.contains(t))
}

// ── SelfRagEngine ─────────────────────────────────────────────────────────────

/// Self-Reflective RAG engine.
///
/// `R` is the [`Reflector`] implementation. The [`Echo`] layer is provided
/// *per call* through the generic `run<E>` method so that the struct remains
/// `Send + Sync` without holding a reference to an unsized trait object.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
pub struct SelfRagEngine<R: Reflector> {
    /// The reflector used for relevance, support, and utility scoring.
    reflector: R,
    /// Engine configuration.
    config: SelfRagConfig,
}

impl<R: Reflector> SelfRagEngine<R> {
    /// Create a new [`SelfRagEngine`].
    #[must_use]
    pub fn new(reflector: R, config: SelfRagConfig) -> Self {
        Self { reflector, config }
    }

    /// Grade a single segment against retrieved docs, appending to critiques and tokens.
    ///
    /// Returns the graded segment text to include in the answer.
    ///
    /// # Errors
    ///
    /// Returns [`SelfRagError::ReflectionFailed`] when a scorer returns an error.
    fn grade_segment(
        &self,
        segment: &str,
        query_trimmed: &str,
        docs: Vec<crate::types::SearchResult>,
        reflection_tokens: &mut Vec<ReflectionToken>,
        critiques: &mut Vec<SegmentCritique>,
    ) -> Result<(), SelfRagError> {
        if docs.is_empty() {
            critiques.push(SegmentCritique {
                segment: segment.to_string(),
                relevance_token: ReflectionToken::Irrelevant,
                support_token: ReflectionToken::Unsupported,
                relevance_score: 0.0,
                support_score: 0.0,
                retrieved_docs: Vec::new(),
            });
            reflection_tokens.push(ReflectionToken::Irrelevant);
            reflection_tokens.push(ReflectionToken::Unsupported);
            return Ok(());
        }

        let mut best_relevance = 0.0f32;
        let mut best_support = 0.0f32;
        for doc in &docs {
            let rel = self
                .reflector
                .relevance(query_trimmed, doc)
                .map_err(|e| SelfRagError::ReflectionFailed(e.to_string()))?;
            let sup = self
                .reflector
                .support(segment, doc)
                .map_err(|e| SelfRagError::ReflectionFailed(e.to_string()))?;
            if rel > best_relevance {
                best_relevance = rel;
            }
            if sup > best_support {
                best_support = sup;
            }
        }

        let relevance_token = if best_relevance >= self.config.relevance_threshold {
            ReflectionToken::Relevant
        } else {
            ReflectionToken::Irrelevant
        };
        let support_token = if best_support >= self.config.support_threshold {
            ReflectionToken::Supported
        } else if best_support >= self.config.support_threshold * 0.5 {
            ReflectionToken::PartiallySupported
        } else {
            ReflectionToken::Unsupported
        };

        reflection_tokens.push(relevance_token.clone());
        reflection_tokens.push(support_token.clone());

        critiques.push(SegmentCritique {
            segment: segment.to_string(),
            relevance_token,
            support_token,
            relevance_score: best_relevance,
            support_score: best_support,
            retrieved_docs: docs,
        });

        Ok(())
    }

    /// Run the Self-RAG loop for `query` and a candidate `generation`.
    ///
    /// The `generation` is the candidate answer text (produced by an LLM or
    /// heuristic generator). The engine segments it, optionally retrieves
    /// supporting documents via `echo`, grades each segment, and returns a
    /// [`SelfRagOutput`] with reflection tokens and critiques.
    ///
    /// # Errors
    ///
    /// Returns [`SelfRagError::EmptyQuery`] when `query` is blank.
    /// Returns [`SelfRagError::RetrievalFailed`] when the Echo search fails.
    /// Returns [`SelfRagError::ReflectionFailed`] when a scorer returns an error.
    pub async fn run<E>(
        &self,
        query: &str,
        generation: &str,
        echo: &E,
    ) -> Result<SelfRagOutput, SelfRagError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        let query_trimmed = query.trim();
        if query_trimmed.is_empty() {
            return Err(SelfRagError::EmptyQuery);
        }

        let raw_segments = split_sentences(generation);
        let generation_is_empty = generation.trim().is_empty();

        let segments: Vec<String> = if generation_is_empty {
            vec![query_trimmed.to_string()]
        } else {
            raw_segments
                .into_iter()
                .take(self.config.max_segments)
                .collect()
        };

        let mut reflection_tokens: Vec<ReflectionToken> = Vec::new();
        let mut critiques: Vec<SegmentCritique> = Vec::new();
        let mut retrieved = false;
        let mut answer_parts: Vec<String> = Vec::new();

        for segment in &segments {
            let needs_retrieve = generation_is_empty || should_retrieve(segment);
            reflection_tokens.push(if needs_retrieve {
                ReflectionToken::Retrieve
            } else {
                ReflectionToken::NoRetrieve
            });

            if needs_retrieve {
                retrieved = true;
                let docs = echo
                    .search(query_trimmed, self.config.top_k, None)
                    .await
                    .map_err(|e| SelfRagError::RetrievalFailed(e.to_string()))?;

                self.grade_segment(
                    segment,
                    query_trimmed,
                    docs,
                    &mut reflection_tokens,
                    &mut critiques,
                )?;
            }
            answer_parts.push(segment.clone());
        }

        let answer = if answer_parts.is_empty() {
            generation.trim().to_string()
        } else {
            answer_parts.join(" ")
        };

        let utility_score = self
            .reflector
            .utility(query_trimmed, &answer)
            .map_err(|e| SelfRagError::ReflectionFailed(e.to_string()))?;
        let utility_token = if utility_score >= self.config.utility_threshold {
            ReflectionToken::Useful
        } else {
            ReflectionToken::NotUseful
        };
        reflection_tokens.push(utility_token.clone());

        Ok(SelfRagOutput {
            query: query_trimmed.to_string(),
            answer,
            retrieved,
            reflection_tokens,
            critiques,
            utility_token,
        })
    }
}
