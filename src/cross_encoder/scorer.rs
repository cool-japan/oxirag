//! Cross-encoder scorer implementations.
use crate::cross_encoder::types::{
    CrossEncoderError, CrossEncoderScorer, FeatureWeights, InteractionFeatures,
};
use crate::types::{Document, SearchResult};
use std::collections::{HashMap, HashSet};

// ── helpers ───────────────────────────────────────────────────────────────────

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

fn bigrams(text: &str) -> Vec<(String, String)> {
    let tokens: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
    tokens
        .windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect()
}

// ── LexicalCrossEncoder ───────────────────────────────────────────────────────

/// Lexical cross-encoder scorer using interaction features and IDF weighting.
#[derive(Debug, Clone, Default)]
pub struct LexicalCrossEncoder {
    /// Feature weights.
    pub weights: FeatureWeights,
}

impl LexicalCrossEncoder {
    /// Create a new scorer with the given feature weights.
    #[must_use]
    pub fn new(weights: FeatureWeights) -> Self {
        Self { weights }
    }

    /// Extract interaction features for a (query, doc) pair given corpus-level IDF.
    #[must_use]
    pub fn extract_features(
        &self,
        query: &str,
        doc: &str,
        idf: &HashMap<String, f32>,
    ) -> InteractionFeatures {
        let q_tokens = tokenize(query);
        let d_tokens = tokenize(doc);
        if q_tokens.is_empty() || d_tokens.is_empty() {
            return InteractionFeatures::default();
        }

        let intersection: HashSet<&String> = q_tokens.intersection(&d_tokens).collect();
        let union_count = q_tokens.union(&d_tokens).count();

        #[allow(clippy::cast_precision_loss)]
        let exact_match_ratio = intersection.len() as f32 / q_tokens.len() as f32;
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
        #[allow(clippy::cast_precision_loss)]
        let doc_coverage = intersection.len() as f32 / d_tokens.len() as f32;

        let q_bigrams: HashSet<(String, String)> = bigrams(query).into_iter().collect();
        let d_bigrams: HashSet<(String, String)> = bigrams(doc).into_iter().collect();
        #[allow(clippy::cast_precision_loss)]
        let ordered_bigram_match = if q_bigrams.is_empty() {
            0.0
        } else {
            q_bigrams.intersection(&d_bigrams).count() as f32 / q_bigrams.len() as f32
        };

        let q_len = q_tokens.len();
        let d_len = d_tokens.len();
        #[allow(clippy::cast_precision_loss)]
        let length_ratio = if q_len == 0 || d_len == 0 {
            0.0
        } else {
            q_len.min(d_len) as f32 / q_len.max(d_len) as f32
        };

        InteractionFeatures {
            exact_match_ratio,
            term_overlap,
            idf_weighted_overlap,
            query_coverage,
            doc_coverage,
            ordered_bigram_match,
            length_ratio,
        }
    }

    #[allow(clippy::unused_self)]
    fn dot(&self, f: &InteractionFeatures, w: &FeatureWeights) -> f32 {
        f.exact_match_ratio * w.exact_match
            + f.term_overlap * w.term_overlap
            + f.idf_weighted_overlap * w.idf_weighted
            + f.query_coverage * w.query_coverage
            + f.doc_coverage * w.doc_coverage
            + f.ordered_bigram_match * w.bigram_match
            + f.length_ratio * w.length_ratio
    }

    fn compute_idf(results: &[SearchResult]) -> HashMap<String, f32> {
        let n = results.len();
        let mut df: HashMap<String, usize> = HashMap::new();
        for r in results {
            for tok in tokenize(&r.document.content) {
                *df.entry(tok).or_insert(0) += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        df.into_iter()
            .map(|(t, df_val)| (t, ((n as f32 + 1.0) / (df_val as f32 + 1.0)).ln() + 1.0))
            .collect()
    }

    /// Score a single (query, document) pair using the pre-computed IDF map.
    #[must_use]
    pub fn score_with_idf(&self, query: &str, doc: &Document, idf: &HashMap<String, f32>) -> f32 {
        let features = self.extract_features(query, &doc.content, idf);
        let raw = self.dot(&features, &self.weights);
        1.0 / (1.0 + (-10.0 * (raw - 0.5)).exp())
    }

    /// Score and rerank `results` for `query`, returning [`InteractionFeatures`]-scored pairs.
    ///
    /// # Errors
    ///
    /// Returns [`CrossEncoderError`] if query is empty or results is empty.
    pub fn score_all(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> Result<Vec<(usize, f32, f32)>, CrossEncoderError> {
        if query.trim().is_empty() {
            return Err(CrossEncoderError::EmptyQuery);
        }
        if results.is_empty() {
            return Err(CrossEncoderError::EmptyCandidates);
        }
        let idf = Self::compute_idf(results);
        Ok(results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let cs = self.score_with_idf(query, &r.document, &idf);
                (i, r.score, cs)
            })
            .collect())
    }
}

impl CrossEncoderScorer for LexicalCrossEncoder {
    fn score(&self, query: &str, doc: &Document) -> f32 {
        let idf = HashMap::new();
        self.score_with_idf(query, doc, &idf)
    }
}

// ── CrossEncoderScorer trait ──────────────────────────────────────────────────

impl CrossEncoderScorer for &dyn CrossEncoderScorer {
    fn score(&self, query: &str, doc: &Document) -> f32 {
        (*self).score(query, doc)
    }
}
