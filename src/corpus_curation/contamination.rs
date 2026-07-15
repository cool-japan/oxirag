//! Pillar 3 — eval-set contamination detection.
//!
//! Given a held-out benchmark (a set of [`BenchmarkItem`]s that must never
//! leak into a training/index corpus) and the corpus itself,
//! [`ContaminationDetector::detect`] reports, per benchmark item, whether it
//! appears **verbatim** or **near-verbatim** (by word n-gram overlap) in any
//! corpus document, plus a corpus-level [`ContaminationReport::contamination_rate`]
//! (the fraction of corpus documents implicated in at least one match) and a
//! configurable [`DecontaminationAction`].
//!
//! # Distinct from `knowledge_unlearning`'s leakage audit
//!
//! `crate::knowledge_unlearning`'s `UnlearningAuditor`
//! asks "did *this one deleted text* survive, verbatim or near-verbatim,
//! anywhere in the store?" — one probe against the whole store. This pillar
//! runs the *opposite* shape of query: **many** benchmark items against the
//! whole corpus, yielding a per-item clean/dirty verdict and an aggregate
//! corpus-level rate, answering "how much of my held-out eval set has already
//! leaked into what I'm about to index?"
//!
//! # Why not `lsh_index::MinHashIndex` here
//!
//! Near-duplicate clustering ([`super::near_dup`]) asks whether two documents
//! are *holistically* similar as unordered bags of shingles — exactly what
//! `MinHash`-estimated `Jaccard` similarity measures. Contamination asks
//! whether a specific, *ordered* phrase from a benchmark item is reproduced
//! literally inside a document; that needs substring/n-gram containment, not
//! a set-similarity estimate, so this module hand-rolls a small, dependency-free
//! n-gram overlap primitive (in the spirit of
//! `crate::knowledge_unlearning::audit`'s stop-word-filtered probe terms and
//! case-insensitive verbatim scan) rather than routing through the near-dup
//! index.

use std::collections::BTreeSet;

use crate::types::Document;

use super::heuristics::is_stopword;
use super::types::CurationError;

// ── text normalisation & n-grams ─────────────────────────────────────────────

/// Lowercase `text` and collapse all whitespace runs (including newlines) to
/// single spaces, so verbatim matching is robust to incidental formatting
/// differences between a benchmark item and its copy in a corpus document.
fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Word n-grams of already-[`normalize`]d `text`. When `text` has fewer than
/// `n` words the whole text is treated as one gram (so short items are still
/// comparable). An empty `text` yields no grams.
fn word_ngrams(normalized_text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = normalized_text
        .split(' ')
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return Vec::new();
    }
    let size = n.max(1);
    if words.len() <= size {
        return vec![words.join(" ")];
    }
    words.windows(size).map(|w| w.join(" ")).collect()
}

/// The n-grams of `normalized_text` that carry distinctive content: windows
/// composed entirely of stop words are dropped, since those are exactly the
/// windows likely to appear *by chance* in an unrelated document and would
/// otherwise inflate the overlap score into false positives. If filtering
/// would remove every window (a benchmark item that is nothing but function
/// words), the unfiltered windows are used instead so detection is never
/// vacuously blind for such an item.
fn informative_ngrams(normalized_text: &str, n: usize) -> Vec<String> {
    let all = word_ngrams(normalized_text, n);
    let filtered: Vec<String> = all
        .iter()
        .filter(|gram| !gram.split(' ').all(is_stopword))
        .cloned()
        .collect();
    if filtered.is_empty() { all } else { filtered }
}

// ── BenchmarkItem ────────────────────────────────────────────────────────────

/// One held-out benchmark/eval item that must not leak into an ingest corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkItem {
    /// A stable identifier for this benchmark item.
    pub id: String,
    /// The item's text (e.g. a QA pair's question and reference answer,
    /// concatenated, or a held-out passage).
    pub text: String,
}

impl BenchmarkItem {
    /// Construct a benchmark item.
    #[must_use]
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

// ── DecontaminationAction ────────────────────────────────────────────────────

/// What a [`ContaminationDetector`] instructs the caller to do with a corpus
/// document implicated in a contamination match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecontaminationAction {
    /// Keep the document but surface it as flagged for human review. The
    /// conservative default: contamination detection never silently destroys
    /// data unless explicitly told to.
    #[default]
    Flag,
    /// Remove the document from the admitted set.
    Drop,
}

// ── ContaminationKind / ContaminationMatch ──────────────────────────────────

/// How a corpus document matched a benchmark item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContaminationKind {
    /// The benchmark item's normalised text appears as an exact substring of
    /// the document.
    Verbatim,
    /// The benchmark item did not appear verbatim, but its informative n-gram
    /// overlap with the document met or exceeded the configured threshold.
    NearVerbatim,
}

/// One corpus document's match against one benchmark item.
#[derive(Debug, Clone, PartialEq)]
pub struct ContaminationMatch {
    /// The id of the matched corpus document.
    pub document_id: String,
    /// The kind of match found.
    pub kind: ContaminationKind,
    /// Fraction of the benchmark item's informative n-grams found in the
    /// document (`1.0` for a [`ContaminationKind::Verbatim`] match).
    pub ngram_overlap: f64,
}

// ── BenchmarkVerdict / ContaminationReport ──────────────────────────────────

/// The verdict for one benchmark item against the whole corpus.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkVerdict {
    /// The benchmark item's id.
    pub benchmark_id: String,
    /// `true` when at least one corpus document matched this item.
    pub dirty: bool,
    /// Every matching document, sorted by descending `ngram_overlap` (ties
    /// broken by ascending document id).
    pub matches: Vec<ContaminationMatch>,
}

/// The result of running [`ContaminationDetector::detect`] over a whole
/// corpus.
#[derive(Debug, Clone, PartialEq)]
pub struct ContaminationReport {
    /// One [`BenchmarkVerdict`] per benchmark item, in input order.
    pub verdicts: Vec<BenchmarkVerdict>,
    /// The fraction of **corpus documents** (not benchmark items) implicated
    /// in at least one match; `0.0` for an empty corpus.
    pub contamination_rate: f64,
    /// Distinct implicated corpus document ids, sorted ascending.
    pub dirty_document_ids: Vec<String>,
    /// The configured action a caller should take on the implicated
    /// documents.
    pub action: DecontaminationAction,
}

impl ContaminationReport {
    /// The number of benchmark items with at least one match.
    #[must_use]
    pub fn dirty_item_count(&self) -> usize {
        self.verdicts.iter().filter(|v| v.dirty).count()
    }
}

// ── ContaminationConfig ──────────────────────────────────────────────────────

/// Configuration for [`ContaminationDetector`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContaminationConfig {
    /// Word n-gram window size for near-verbatim overlap. Defaults to `6`.
    pub ngram_size: usize,
    /// Minimum informative-n-gram overlap fraction, in `[0, 1]`, for a
    /// non-verbatim match to count as [`ContaminationKind::NearVerbatim`].
    /// Defaults to `0.6`.
    pub near_verbatim_threshold: f64,
    /// The action to report for implicated documents. Defaults to
    /// [`DecontaminationAction::Flag`].
    pub action: DecontaminationAction,
}

impl Default for ContaminationConfig {
    fn default() -> Self {
        Self {
            ngram_size: 6,
            near_verbatim_threshold: 0.6,
            action: DecontaminationAction::Flag,
        }
    }
}

impl ContaminationConfig {
    /// Construct a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the n-gram window size.
    #[must_use]
    pub fn with_ngram_size(mut self, ngram_size: usize) -> Self {
        self.ngram_size = ngram_size;
        self
    }

    /// Set the near-verbatim overlap threshold.
    #[must_use]
    pub fn with_near_verbatim_threshold(mut self, threshold: f64) -> Self {
        self.near_verbatim_threshold = threshold;
        self
    }

    /// Set the decontamination action.
    #[must_use]
    pub fn with_action(mut self, action: DecontaminationAction) -> Self {
        self.action = action;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::InvalidConfig`] when `ngram_size` is zero, or
    /// `near_verbatim_threshold` is non-finite or outside `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), CurationError> {
        if self.ngram_size == 0 {
            return Err(CurationError::InvalidConfig(
                "ngram_size must be > 0".into(),
            ));
        }
        if !self.near_verbatim_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.near_verbatim_threshold)
        {
            return Err(CurationError::InvalidConfig(format!(
                "near_verbatim_threshold must be finite and within [0, 1], got {}",
                self.near_verbatim_threshold
            )));
        }
        Ok(())
    }
}

// ── ContaminationDetector ────────────────────────────────────────────────────

/// Detects benchmark leakage in a corpus (see the [module documentation](self)).
#[derive(Debug, Clone, PartialEq)]
pub struct ContaminationDetector {
    /// The active configuration.
    pub config: ContaminationConfig,
}

impl ContaminationDetector {
    /// Create a new detector.
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::InvalidConfig`] when `config` fails
    /// [`ContaminationConfig::validate`].
    pub fn new(config: ContaminationConfig) -> Result<Self, CurationError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Check every item in `benchmark` against every document in `corpus`.
    ///
    /// A benchmark item is **dirty** when its normalised text is found as an
    /// exact substring of a document (verbatim), or when its informative,
    /// stop-word-filtered n-grams overlap a document at or above
    /// [`ContaminationConfig::near_verbatim_threshold`] (near-verbatim).
    /// An item with empty (or whitespace-only) text is reported clean with no
    /// matches, since there is nothing distinctive to search for.
    #[must_use]
    pub fn detect(&self, benchmark: &[BenchmarkItem], corpus: &[Document]) -> ContaminationReport {
        let normalized_docs: Vec<(String, String)> = corpus
            .iter()
            .map(|d| (d.id.as_str().to_string(), normalize(&d.content)))
            .collect();

        let mut verdicts: Vec<BenchmarkVerdict> = Vec::with_capacity(benchmark.len());
        let mut dirty_ids: BTreeSet<String> = BTreeSet::new();

        for item in benchmark {
            let normalized_item = normalize(&item.text);
            if normalized_item.is_empty() {
                verdicts.push(BenchmarkVerdict {
                    benchmark_id: item.id.clone(),
                    dirty: false,
                    matches: Vec::new(),
                });
                continue;
            }

            let item_ngrams = informative_ngrams(&normalized_item, self.config.ngram_size);
            let mut matches: Vec<ContaminationMatch> = Vec::new();

            for (document_id, doc_normalized) in &normalized_docs {
                if doc_normalized.contains(normalized_item.as_str()) {
                    matches.push(ContaminationMatch {
                        document_id: document_id.clone(),
                        kind: ContaminationKind::Verbatim,
                        ngram_overlap: 1.0,
                    });
                    continue;
                }
                if item_ngrams.is_empty() {
                    continue;
                }
                let hits = item_ngrams
                    .iter()
                    .filter(|gram| doc_normalized.contains(gram.as_str()))
                    .count();
                #[allow(clippy::cast_precision_loss)]
                let overlap = hits as f64 / item_ngrams.len() as f64;
                if overlap >= self.config.near_verbatim_threshold {
                    matches.push(ContaminationMatch {
                        document_id: document_id.clone(),
                        kind: ContaminationKind::NearVerbatim,
                        ngram_overlap: overlap,
                    });
                }
            }

            for matched in &matches {
                dirty_ids.insert(matched.document_id.clone());
            }
            matches.sort_by(|a, b| {
                b.ngram_overlap
                    .total_cmp(&a.ngram_overlap)
                    .then_with(|| a.document_id.cmp(&b.document_id))
            });

            verdicts.push(BenchmarkVerdict {
                benchmark_id: item.id.clone(),
                dirty: !matches.is_empty(),
                matches,
            });
        }

        let dirty_document_ids: Vec<String> = dirty_ids.into_iter().collect();
        #[allow(clippy::cast_precision_loss)]
        let contamination_rate = if corpus.is_empty() {
            0.0
        } else {
            dirty_document_ids.len() as f64 / corpus.len() as f64
        };

        ContaminationReport {
            verdicts,
            contamination_rate,
            dirty_document_ids,
            action: self.config.action,
        }
    }
}
