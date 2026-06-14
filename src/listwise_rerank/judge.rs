//! Lexical listwise judge implementation.
use crate::listwise_rerank::types::ListwiseJudge;
use crate::types::Document;
use std::collections::{HashMap, HashSet};

// ── helpers ───────────────────────────────────────────────────────────────────

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── LexicalListwiseJudge ──────────────────────────────────────────────────────

/// A pure-Rust stand-in for an LLM listwise judge.
///
/// It emulates the ordering an instruction-tuned model would produce by scoring
/// every document in the window with a lexical relevance heuristic (token
/// overlap, IDF-weighted overlap, and query coverage) and returning the
/// window-local indices sorted by score descending. The IDF statistics are
/// computed *from the supplied window*, so the judge is fully self-contained and
/// deterministic.
#[derive(Debug, Clone, Default)]
pub struct LexicalListwiseJudge;

impl LexicalListwiseJudge {
    /// Create a new lexical listwise judge.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute inverse-document-frequency weights over the window's documents.
    fn compute_idf(docs: &[Document]) -> HashMap<String, f32> {
        let n = docs.len();
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in docs {
            for tok in tokenize(&doc.content) {
                *df.entry(tok).or_insert(0) += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        df.into_iter()
            .map(|(t, df_val)| (t, ((n as f32 + 1.0) / (df_val as f32 + 1.0)).ln() + 1.0))
            .collect()
    }

    /// Relevance score for a single document against `query` using `idf`.
    ///
    /// Combines plain token overlap, IDF-weighted overlap, and query coverage,
    /// mirroring the feature blend used by the cross-encoder scorer.
    #[must_use]
    pub fn score(&self, query: &str, doc: &Document, idf: &HashMap<String, f32>) -> f32 {
        let q_tokens = tokenize(query);
        let d_tokens = tokenize(&doc.content);
        if q_tokens.is_empty() || d_tokens.is_empty() {
            return 0.0;
        }

        let intersection: HashSet<&String> = q_tokens.intersection(&d_tokens).collect();
        let union_count = q_tokens.union(&d_tokens).count();

        #[allow(clippy::cast_precision_loss)]
        let term_overlap = if union_count == 0 {
            0.0
        } else {
            intersection.len() as f32 / union_count as f32
        };

        let idf_sum: f32 = q_tokens
            .iter()
            .map(|t| idf.get(t).copied().unwrap_or(1.0))
            .sum::<f32>();
        let idf_match: f32 = intersection
            .iter()
            .map(|t| idf.get(*t).copied().unwrap_or(1.0))
            .sum::<f32>();
        let idf_weighted_overlap = if idf_sum == 0.0 {
            0.0
        } else {
            idf_match / idf_sum
        };

        #[allow(clippy::cast_precision_loss)]
        let query_coverage = intersection.len() as f32 / q_tokens.len() as f32;

        0.35 * term_overlap + 0.45 * idf_weighted_overlap + 0.20 * query_coverage
    }
}

impl ListwiseJudge for LexicalListwiseJudge {
    fn permute(&self, query: &str, docs: &[Document]) -> Vec<usize> {
        let idf = Self::compute_idf(docs);
        let mut indexed: Vec<(usize, f32)> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i, self.score(query, d, &idf)))
            .collect();
        // Sort by score descending; ties keep the lower (earlier) index first so
        // the permutation is stable and deterministic.
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        indexed.into_iter().map(|(i, _)| i).collect()
    }
}
