//! Corpus curation — ingest-time corpus quality: heuristic quality signals, a
//! trained quality classifier, and eval-set contamination detection.
//!
//! Every other "filter" module in this crate (`noise_filter`,
//! `source_credibility`, `poisoning_defense`, `filco`, `recomp`,
//! `context_pruning`) takes a **query** and filters what is *returned* by a
//! search. The crate's only *ingest-time* filter is the exact-content-hash
//! dedup in `crate::document_pipeline::indexing`. `corpus_curation` fills
//! that gap: it is a standalone `CorpusCurator` library that scores and
//! filters documents **before** they enter an index, and deliberately does
//! not wire itself into `crate::document_pipeline` — a pipeline composes it
//! as a pre-indexing gate, not the other way around.
//!
//! # The four pillars
//!
//! 1. **Heuristic quality signals** ([`heuristics`]) — `Gopher`
//!    (Rae et al., 2021) / `C4` (Raffel et al., 2019) style rules: word-count
//!    bounds, mean word length, symbol density, stop-word density, line-level
//!    repetition, bullet/ellipsis density, and alphabetic-character density.
//!    A configurable [`QualityRule`] set produces a per-document pass/fail
//!    *with reasons* ([`QualityVerdict`]).
//! 2. **A trained quality classifier** ([`classifier`]) — a binary
//!    logistic-regression [`QualityClassifier`] over hashed bag-of-words/
//!    bigram text features, fit by full-batch gradient descent. No classifier
//!    of any kind exists elsewhere in this crate.
//! 3. **Eval-set contamination detection** ([`contamination`]) — given a held-
//!    out benchmark, [`ContaminationDetector`] finds corpus documents that
//!    reproduce a benchmark item verbatim or near-verbatim (by word n-gram
//!    overlap), producing a per-item verdict, a corpus-level contamination
//!    rate, and a configurable [`DecontaminationAction`].
//! 4. **Near-duplicate clustering** ([`near_dup`]) — composed on top of
//!    [`crate::lsh_index::MinHashIndex`] (this is why the `corpus-curation`
//!    feature depends on `lsh`) rather than reimplemented; `MinHash`
//!    near-duplicate detection already exists in [`crate::lsh_index`],
//!    `crate::semantic_dedup`, and `crate::knowledge_unlearning::dedup`,
//!    so this pillar reuses the first of those instead of writing a fourth.
//!
//! [`engine::CorpusCurator`] composes all four into a single staged
//! admission funnel; see its docs for the exact stage order and for the
//! "distinct from" comparisons against `semantic_dedup`, `lsh_index`, and
//! `knowledge_unlearning`.
//!
//! # Quick start
//!
//! ```
//! # #[cfg(feature = "corpus-curation")] {
//! use oxirag::corpus_curation::{
//!     BenchmarkItem, ClassifierExample, CorpusCurator, CurationConfig,
//! };
//! use oxirag::types::Document;
//!
//! // Pillar 1: heuristic rules alone, no classifier or benchmark needed.
//! let curator = CorpusCurator::new(CurationConfig::new());
//! let verdict = curator.assess_heuristics(&Document::new("too short"));
//! assert!(!verdict.passed);
//!
//! // Pillar 3: does a corpus leak a held-out benchmark item?
//! let benchmark = vec![BenchmarkItem::new("q1", "the exact benchmark sentence")];
//! let corpus = vec![
//!     Document::new("an unrelated document about something else entirely").with_id("clean"),
//!     Document::new("this document contains the exact benchmark sentence verbatim")
//!         .with_id("leaked"),
//! ];
//! let report = curator.detect_contamination(&corpus, &benchmark).expect("valid config");
//! assert_eq!(report.dirty_document_ids, vec!["leaked".to_string()]);
//! # let _ = ClassifierExample::good("unused in this example");
//! # }
//! ```

pub mod classifier;
pub mod contamination;
pub mod engine;
pub mod heuristics;
pub mod near_dup;
pub mod rng;
#[cfg(test)]
mod tests;
pub mod types;

pub use classifier::{
    ClassifierConfig, ClassifierEvaluation, ClassifierExample, QualityClassifier, train_test_split,
};
pub use contamination::{
    BenchmarkItem, BenchmarkVerdict, ContaminationConfig, ContaminationDetector, ContaminationKind,
    ContaminationMatch, ContaminationReport, DecontaminationAction,
};
pub use engine::CorpusCurator;
pub use heuristics::{QualityRule, QualitySignal, QualityVerdict, evaluate_quality_rules};
pub use near_dup::{NearDupConfig, NearDuplicateCluster, find_near_duplicate_clusters};
pub use rng::CurationRng;
pub use types::{
    CorpusReport, CurationConfig, CurationError, CurationReport, DocumentQuality, QualityReport,
    QualityScore, RejectedDocument, RejectionStage,
};
