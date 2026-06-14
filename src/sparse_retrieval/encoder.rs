//! SPLADE-style learned sparse encoder with term expansion.

use std::collections::HashMap;

use crate::sparse_retrieval::types::{SparseConfig, SparseVector};
use crate::types::Document;

// ── Tokeniser ─────────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens of length >= 2.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── SparseEncoder ─────────────────────────────────────────────────────────────

/// A learned sparse encoder.
///
/// After being [`fit`](SparseEncoder::fit) on a corpus it knows the inverse
/// document frequency (IDF) of every term and a co-occurrence table describing
/// which terms tend to appear together. [`encode`](SparseEncoder::encode) turns
/// arbitrary text into a [`SparseVector`] whose weights are log-saturated
/// `tf * idf` values, and which is *expanded* with discounted co-occurring terms
/// — a deterministic stand-in for the neural expansion performed by SPLADE.
#[derive(Debug, Clone)]
pub struct SparseEncoder {
    /// Encoding and expansion configuration.
    config: SparseConfig,
    /// Inverse document frequency per term, learned during [`fit`](Self::fit).
    idf: HashMap<String, f32>,
    /// Per-term ranked list of co-occurring terms with normalised strengths.
    cooccur: HashMap<String, Vec<(String, f32)>>,
    /// Whether [`fit`](Self::fit) has populated the statistics.
    fitted: bool,
}

impl SparseEncoder {
    /// Create a new, unfitted encoder with the given configuration.
    #[must_use]
    pub fn new(config: SparseConfig) -> Self {
        Self {
            config,
            idf: HashMap::new(),
            cooccur: HashMap::new(),
            fitted: false,
        }
    }

    /// Whether the encoder has been fitted on a corpus.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Access the configuration backing this encoder.
    #[must_use]
    pub fn config(&self) -> &SparseConfig {
        &self.config
    }

    /// Look up the learned IDF of `term`, if any.
    #[must_use]
    pub fn idf_of(&self, term: &str) -> Option<f32> {
        self.idf.get(term).copied()
    }

    /// Fit the encoder, learning IDF weights and a term co-occurrence table.
    ///
    /// IDF uses the smoothed form `ln((N + 1) / (df + 1)) + 1`. The co-occurrence
    /// table records, for every term, how often each other term shares a document
    /// with it; the counts are normalised by the term's own document frequency so
    /// the resulting strengths lie in `[0, 1]`.
    pub fn fit(&mut self, corpus: &[Document]) {
        #[allow(clippy::cast_precision_loss)]
        let n = corpus.len() as f32;

        let mut df: HashMap<String, usize> = HashMap::new();
        let mut cooccur_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();

        for doc in corpus {
            // Distinct terms in this document.
            let mut unique: Vec<String> = tokenize(&doc.content);
            unique.sort_unstable();
            unique.dedup();

            for term in &unique {
                *df.entry(term.clone()).or_insert(0) += 1;
            }

            // Count unordered co-occurrences between every distinct pair.
            for (i, term_a) in unique.iter().enumerate() {
                let entry = cooccur_counts.entry(term_a.clone()).or_default();
                for term_b in unique.iter().skip(i + 1) {
                    *entry.entry(term_b.clone()).or_insert(0) += 1;
                }
                // Mirror the pair for the symmetric partner.
                for term_b in unique.iter().take(i) {
                    *cooccur_counts
                        .entry(term_a.clone())
                        .or_default()
                        .entry(term_b.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let idf: HashMap<String, f32> = df
            .iter()
            .map(|(term, &df_val)| (term.clone(), ((n + 1.0) / (df_val as f32 + 1.0)).ln() + 1.0))
            .collect();

        // Normalise co-occurrence counts by the host term's document frequency
        // and rank partners deterministically (strength desc, then term asc).
        let mut cooccur: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        for (term, partners) in cooccur_counts {
            let host_df = *df.get(&term).unwrap_or(&1);
            #[allow(clippy::cast_precision_loss)]
            let host_df_f = host_df.max(1) as f32;
            let mut ranked: Vec<(String, f32)> = partners
                .into_iter()
                .map(|(partner, count)| {
                    #[allow(clippy::cast_precision_loss)]
                    let strength = (count as f32 / host_df_f).min(1.0);
                    (partner, strength)
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            cooccur.insert(term, ranked);
        }

        self.idf = idf;
        self.cooccur = cooccur;
        self.fitted = true;
    }

    /// Encode `text` into a learned [`SparseVector`].
    ///
    /// Each present term receives weight `log(1 + max(0, tf * idf * saturation))`.
    /// For every present term, up to `expansion_terms` of its strongest
    /// co-occurring terms are added at a discounted weight (the present term's
    /// weight scaled by the co-occurrence strength), simulating SPLADE expansion.
    /// Expansion contributions accumulate, and any final weight strictly below
    /// `min_weight` is dropped. When the encoder is unfitted, IDF defaults to
    /// `1.0` and no expansion is performed.
    #[must_use]
    pub fn encode(&self, text: &str) -> SparseVector {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return SparseVector::default();
        }

        // Term frequencies.
        let mut tf: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0) += 1;
        }

        // Base learned weights: log-saturated tf * idf.
        let mut weights: HashMap<String, f32> = HashMap::new();
        let mut base: Vec<(String, f32)> = Vec::with_capacity(tf.len());
        for (term, &count) in &tf {
            let idf = self.idf.get(term).copied().unwrap_or(1.0);
            #[allow(clippy::cast_precision_loss)]
            let raw = count as f32 * idf * self.config.saturation;
            let weight = (1.0 + raw.max(0.0)).ln();
            if weight > 0.0 {
                weights.insert(term.clone(), weight);
                base.push((term.clone(), weight));
            }
        }

        // SPLADE-style expansion: add discounted co-occurring terms.
        if self.fitted && self.config.expansion_terms > 0 {
            for (term, term_weight) in &base {
                if let Some(partners) = self.cooccur.get(term) {
                    for (partner, strength) in partners.iter().take(self.config.expansion_terms) {
                        let contribution = term_weight * strength;
                        if contribution <= 0.0 {
                            continue;
                        }
                        weights
                            .entry(partner.clone())
                            .and_modify(|w| {
                                if contribution > *w {
                                    *w = contribution;
                                }
                            })
                            .or_insert(contribution);
                    }
                }
            }
        }

        // Prune sub-threshold weights.
        weights.retain(|_, w| *w >= self.config.min_weight);

        SparseVector::new(weights)
    }
}
