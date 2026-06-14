//! Answerability-based router for the RAG vs. long-context decision.
//!
//! The [`SelfRouter`] inspects a query alongside its retrieved documents and
//! blends two cheap, deterministic signals into an *answerability* score:
//!
//! - **query coverage** — the fraction of distinct query content tokens that
//!   appear anywhere in the retrieved documents (see
//!   [`SelfRouter::query_coverage`]), and
//! - **top relevance** — the best query↔document token-overlap (Jaccard) over
//!   the retrieved documents.
//!
//! When the blended score meets [`SelfRouteConfig::answerability_threshold`] the
//! query routes to [`RouteDecision::Rag`]; otherwise it falls back to
//! [`RouteDecision::LongContext`]. An empty retrieved set always routes to
//! [`RouteDecision::LongContext`] with answerability `0.0`.

use std::collections::HashSet;

use crate::types::Document;

use super::types::{RouteAssessment, RouteDecision, SelfRouteConfig, SelfRouteError};

// ── tokenization ──────────────────────────────────────────────────────────────

/// Splits `text` into lowercase content tokens of length `>= 2`.
///
/// Splitting is on any non-alphanumeric character, matching the tokenizer used
/// throughout `OxiRAG`'s lexical components.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Computes the Jaccard token-overlap of two pre-tokenized sets.
///
/// Returns `0.0` when either set is empty.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let value = intersection as f32 / union as f32;
        value
    }
}

// ── SelfRouter ────────────────────────────────────────────────────────────────

/// Routes a query to RAG or long context based on retrieved-context
/// answerability.
#[derive(Debug, Clone)]
pub struct SelfRouter {
    /// The configuration governing the answerability blend and threshold.
    pub config: SelfRouteConfig,
}

impl SelfRouter {
    /// Creates a new router from a configuration.
    #[must_use]
    pub fn new(config: SelfRouteConfig) -> Self {
        Self { config }
    }

    /// Computes the fraction of distinct query content tokens that appear
    /// anywhere in the retrieved documents.
    ///
    /// Both query and documents are tokenized with the module tokenizer
    /// (non-alphanumeric split, length `>= 2`, lowercase). The result is in
    /// `[0.0, 1.0]`; it is `0.0` when the query has no content tokens or when
    /// `docs` is empty.
    #[must_use]
    pub fn query_coverage(&self, query: &str, docs: &[Document]) -> f32 {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return 0.0;
        }
        let mut doc_tokens: HashSet<String> = HashSet::new();
        for doc in docs {
            doc_tokens.extend(tokenize(&doc.content));
        }
        if doc_tokens.is_empty() {
            return 0.0;
        }
        let covered = query_tokens
            .iter()
            .filter(|t| doc_tokens.contains(*t))
            .count();
        #[allow(clippy::cast_precision_loss)]
        let value = covered as f32 / query_tokens.len() as f32;
        value
    }

    /// Computes the best query↔document token-overlap over the retrieved
    /// documents.
    ///
    /// Each document's overlap is the Jaccard coefficient of its token set with
    /// the query token set; the maximum is returned. The result is `0.0` when
    /// `docs` is empty or the query has no content tokens.
    fn top_relevance(query_tokens: &HashSet<String>, docs: &[Document]) -> f32 {
        let mut best = 0.0_f32;
        for doc in docs {
            let doc_tokens = tokenize(&doc.content);
            let score = jaccard(query_tokens, &doc_tokens);
            if score > best {
                best = score;
            }
        }
        best
    }

    /// Assesses whether the retrieved context can answer `query`.
    ///
    /// The answerability score blends query coverage and top relevance per
    /// [`SelfRouteConfig`]; the decision is [`RouteDecision::Rag`] when the score
    /// meets the threshold and [`RouteDecision::LongContext`] otherwise. An empty
    /// `retrieved` slice yields a [`RouteDecision::LongContext`] assessment with
    /// answerability `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`SelfRouteError::EmptyQuery`] if `query` is empty or contains
    /// only whitespace.
    pub fn assess(
        &self,
        query: &str,
        retrieved: &[Document],
    ) -> Result<RouteAssessment, SelfRouteError> {
        if query.trim().is_empty() {
            return Err(SelfRouteError::EmptyQuery);
        }

        if retrieved.is_empty() {
            return Ok(RouteAssessment {
                decision: RouteDecision::LongContext,
                answerability: 0.0,
                query_coverage: 0.0,
                top_relevance: 0.0,
                reason: "no documents retrieved; falling back to long context".to_string(),
            });
        }

        let query_tokens = tokenize(query);
        let query_coverage = self.query_coverage(query, retrieved);
        let top_relevance = Self::top_relevance(&query_tokens, retrieved);
        let weight = self.config.coverage_weight;
        let answerability = weight * query_coverage + (1.0 - weight) * top_relevance;

        let decision = if answerability >= self.config.answerability_threshold {
            RouteDecision::Rag
        } else {
            RouteDecision::LongContext
        };

        let reason = format!(
            "answerability {answerability:.3} ({} threshold {:.3}); coverage {query_coverage:.3}, top relevance {top_relevance:.3} ⇒ {}",
            if answerability >= self.config.answerability_threshold {
                "≥"
            } else {
                "<"
            },
            self.config.answerability_threshold,
            decision.as_str(),
        );

        Ok(RouteAssessment {
            decision,
            answerability,
            query_coverage,
            top_relevance,
            reason,
        })
    }

    /// Assesses `query` against `retrieved` and returns only the
    /// [`RouteDecision`].
    ///
    /// # Errors
    ///
    /// Returns [`SelfRouteError::EmptyQuery`] if `query` is empty or contains
    /// only whitespace.
    pub fn route(
        &self,
        query: &str,
        retrieved: &[Document],
    ) -> Result<RouteDecision, SelfRouteError> {
        Ok(self.assess(query, retrieved)?.decision)
    }
}
