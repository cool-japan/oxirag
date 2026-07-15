//! [`CorpusCurator`] — the orchestrator that composes all four pillars into a
//! single ingest-time admission decision.
//!
//! `corpus_curation` does **not** wire itself into
//! `crate::document_pipeline`; it is a standalone library that a pipeline
//! can call *before* indexing. [`CorpusCurator::curate`] runs a staged funnel
//! over the input documents — cheapest filters first — so every rejected
//! document is attributed to exactly one stage:
//!
//! 1. **Quality** (pillars 1+2): every document must pass every configured
//!    [`super::heuristics::QualityRule`]; if the classifier stage is enabled
//!    it must additionally score at or above the classifier's decision
//!    threshold.
//! 2. **Near-duplicate** (pillar 4): among the quality-stage survivors, every
//!    non-representative member of a near-duplicate cluster is removed.
//! 3. **Contamination** (pillar 3): among the near-duplicate-stage survivors,
//!    documents matching a supplied benchmark are flagged, and dropped too
//!    when [`super::contamination::DecontaminationAction::Drop`] is
//!    configured.
//!
//! Each pillar also has a standalone entry point
//! ([`CorpusCurator::assess_heuristics`], [`CorpusCurator::find_near_duplicates`],
//! [`CorpusCurator::detect_contamination`]) for callers who want just one
//! signal rather than the full funnel.

use std::collections::{HashMap, HashSet};

use crate::types::Document;

use super::classifier::{ClassifierExample, QualityClassifier};
use super::contamination::{
    BenchmarkItem, ContaminationDetector, ContaminationReport, DecontaminationAction,
};
use super::heuristics::{QualityVerdict, evaluate_quality_rules};
use super::near_dup::{NearDuplicateCluster, find_near_duplicate_clusters};
use super::types::{
    CorpusReport, CurationConfig, CurationError, CurationReport, DocumentQuality, QualityReport,
    QualityScore, RejectedDocument, RejectionStage,
};

/// Return type of [`CorpusCurator::apply_contamination_stage`]: the documents
/// that survive stage 3, the flagged-contaminated ids, and the contamination
/// report (`None` when no benchmark was supplied).
type ContaminationStageOutcome = (Vec<Document>, Vec<String>, Option<ContaminationReport>);

/// Scores and filters a corpus **at ingest time**: heuristic quality rules, a
/// trained quality classifier, and eval-set contamination decide what is
/// *admitted to the index*, never what is *returned from it*.
///
/// # Distinct from `semantic_dedup` and `lsh_index::MinHashIndex`
///
/// `crate::semantic_dedup` near-duplicate-clusters a *retrieved result set*
/// with an O(n²) all-pairs comparison, and
/// [`crate::lsh_index::MinHashIndex`] serves banded-`MinHash` k-NN queries.
/// `corpus_curation` scores and filters the corpus itself before it ever
/// reaches an index — heuristic quality rules, a trained quality classifier,
/// and eval-set contamination — deciding what is admitted, not what is later
/// retrieved.
///
/// # Distinct from `knowledge_unlearning`'s leakage audit
///
/// `crate::knowledge_unlearning`'s post-deletion audit asks "did *this one
/// deleted text* survive, anywhere in the store?" — one probe against the
/// whole store. Contamination detection here runs *many* benchmark items
/// against the whole corpus, yielding a per-item verdict and a corpus-level
/// contamination rate.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "corpus-curation")] {
/// use oxirag::corpus_curation::{CorpusCurator, CurationConfig};
/// use oxirag::types::Document;
///
/// let curator = CorpusCurator::new(CurationConfig::new());
/// let docs = vec![
///     Document::new(
///         "This is a perfectly ordinary paragraph of English prose about gardening, \
///          written with enough words and enough common function words to read as \
///          natural text. It describes the slow, quiet satisfaction of tending a \
///          small vegetable patch through every season, from the first spring \
///          seedlings to the last autumn harvest of the year.",
///     )
///     .with_id("doc-1"),
///     Document::new("buy buy buy buy buy buy buy buy buy buy").with_id("doc-2"),
/// ];
/// let report = curator.curate(&docs, None).expect("curation succeeds");
/// assert!(report.is_admitted("doc-1"));
/// assert!(!report.is_admitted("doc-2"));
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CorpusCurator {
    /// The active configuration.
    pub config: CurationConfig,
    /// The trained classifier, if any (set via [`Self::with_classifier`] or
    /// [`Self::train_classifier`]).
    classifier: Option<QualityClassifier>,
}

impl CorpusCurator {
    /// Construct a curator with no classifier attached.
    #[must_use]
    pub fn new(config: CurationConfig) -> Self {
        Self {
            config,
            classifier: None,
        }
    }

    /// Attach an already-trained classifier, returning `self` for chaining.
    #[must_use]
    pub fn with_classifier(mut self, classifier: QualityClassifier) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Train a classifier on `examples` under [`CurationConfig::classifier`]
    /// and attach it, replacing any previously attached classifier.
    ///
    /// # Errors
    ///
    /// Propagates [`QualityClassifier::train`]'s errors: an invalid
    /// classifier configuration, or an empty `examples` slice.
    pub fn train_classifier(
        &mut self,
        examples: &[ClassifierExample],
    ) -> Result<(), CurationError> {
        let classifier = QualityClassifier::train(examples, &self.config.classifier)?;
        self.classifier = Some(classifier);
        Ok(())
    }

    /// Borrow the attached classifier, if any.
    #[must_use]
    pub fn classifier(&self) -> Option<&QualityClassifier> {
        self.classifier.as_ref()
    }

    /// Run only the heuristic rule set (pillar 1) against `doc`.
    #[must_use]
    pub fn assess_heuristics(&self, doc: &Document) -> QualityVerdict {
        evaluate_quality_rules(&doc.content, &self.config.quality_rules)
    }

    /// Score `doc`'s quality (pillars 1+2 blended; see [`QualityScore`]).
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::NoClassifierTrained`] when
    /// [`CurationConfig::use_classifier`] is `true` but no classifier is
    /// attached.
    pub fn score_quality(&self, doc: &Document) -> Result<QualityScore, CurationError> {
        self.assess_document(doc).map(|dq| dq.score)
    }

    /// Run the pillar 1 + 2 quality pass over every document in `docs`.
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::NoClassifierTrained`] when
    /// [`CurationConfig::use_classifier`] is `true` but no classifier is
    /// attached.
    pub fn quality_report(&self, docs: &[Document]) -> Result<QualityReport, CurationError> {
        let mut documents = Vec::with_capacity(docs.len());
        for doc in docs {
            documents.push(self.assess_document(doc)?);
        }
        let total = documents.len();
        let admitted = documents.iter().filter(|d| d.admitted).count();
        #[allow(clippy::cast_precision_loss)]
        let mean_combined_score = if total == 0 {
            0.0
        } else {
            documents.iter().map(|d| d.score.combined).sum::<f64>() / total as f64
        };
        Ok(QualityReport {
            total,
            admitted,
            mean_combined_score,
            documents,
        })
    }

    /// Run only the near-duplicate pillar (pillar 4) over `docs`, composed on
    /// [`crate::lsh_index::MinHashIndex`] (see [`super::near_dup`]).
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::NearDupIndexFailed`] when the underlying
    /// index composition fails.
    pub fn find_near_duplicates(
        &self,
        docs: &[Document],
    ) -> Result<Vec<NearDuplicateCluster>, CurationError> {
        find_near_duplicate_clusters(docs, &self.config.near_dup)
    }

    /// Run only the contamination pillar (pillar 3): check `benchmark`
    /// against `docs`.
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::InvalidConfig`] when
    /// [`CurationConfig::contamination`] is invalid.
    pub fn detect_contamination(
        &self,
        docs: &[Document],
        benchmark: &[BenchmarkItem],
    ) -> Result<ContaminationReport, CurationError> {
        let detector = ContaminationDetector::new(self.config.contamination)?;
        Ok(detector.detect(benchmark, docs))
    }

    /// Run the full staged funnel (see the [module documentation](self)) over
    /// `docs`, optionally checking `benchmark` for contamination as the final
    /// stage.
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::NoClassifierTrained`],
    /// [`CurationError::NearDupIndexFailed`], or
    /// [`CurationError::InvalidConfig`] under the same conditions as
    /// [`Self::quality_report`], [`Self::find_near_duplicates`], and
    /// [`Self::detect_contamination`] respectively.
    pub fn curate(
        &self,
        docs: &[Document],
        benchmark: Option<&[BenchmarkItem]>,
    ) -> Result<CurationReport, CurationError> {
        // Stage 1: quality (heuristics, plus the classifier when enabled).
        let quality = self.quality_report(docs)?;

        let mut rejected: Vec<RejectedDocument> = Vec::new();
        let mut survivors: Vec<Document> = Vec::new();

        for (doc, doc_quality) in docs.iter().zip(quality.documents.iter()) {
            if doc_quality.admitted {
                survivors.push(doc.clone());
            } else if doc_quality.verdict.passed {
                // Heuristics passed; the classifier gate is what failed.
                let reason = format!(
                    "classifier probability {:.4} below decision threshold",
                    doc_quality.score.classifier_probability.unwrap_or(0.0)
                );
                rejected.push(RejectedDocument {
                    document_id: doc_quality.document_id.clone(),
                    stage: RejectionStage::Classifier,
                    reasons: vec![reason],
                });
            } else {
                let reasons: Vec<String> = doc_quality
                    .verdict
                    .failed_signals()
                    .map(|s| s.reason.clone())
                    .collect();
                rejected.push(RejectedDocument {
                    document_id: doc_quality.document_id.clone(),
                    stage: RejectionStage::Heuristic,
                    reasons,
                });
            }
        }

        // Stage 2: near-duplicate clustering among the quality survivors.
        let (after_dedup, near_duplicate_clusters) =
            self.apply_near_duplicate_stage(survivors, &mut rejected)?;

        // Stage 3: contamination among the near-duplicate survivors.
        let (final_survivors, flagged_contaminated_ids, contamination_report) =
            self.apply_contamination_stage(after_dedup, benchmark, &mut rejected)?;

        let admitted_ids: Vec<String> = final_survivors
            .iter()
            .map(|d| d.id.as_str().to_string())
            .collect();

        let rejected_heuristic = rejected
            .iter()
            .filter(|r| r.stage == RejectionStage::Heuristic)
            .count();
        let rejected_classifier = rejected
            .iter()
            .filter(|r| r.stage == RejectionStage::Classifier)
            .count();
        let rejected_near_duplicate = rejected
            .iter()
            .filter(|r| r.stage == RejectionStage::NearDuplicate)
            .count();
        let rejected_contamination = rejected
            .iter()
            .filter(|r| r.stage == RejectionStage::Contamination)
            .count();

        let summary = CorpusReport {
            total_documents: docs.len(),
            admitted: admitted_ids.len(),
            rejected_heuristic,
            rejected_classifier,
            rejected_near_duplicate,
            rejected_contamination,
            duplicate_cluster_count: near_duplicate_clusters.len(),
            contamination_rate: contamination_report
                .as_ref()
                .map_or(0.0, |c| c.contamination_rate),
            mean_quality_score: quality.mean_combined_score,
        };

        Ok(CurationReport {
            quality,
            near_duplicate_clusters,
            contamination: contamination_report,
            admitted_ids,
            rejected,
            flagged_contaminated_ids,
            summary,
        })
    }

    /// Stage 2 of [`Self::curate`]: cluster `survivors` for near-duplicates
    /// and split them into the documents that proceed to stage 3 and the
    /// clusters found, pushing a [`RejectedDocument`] for every
    /// non-representative member onto `rejected`.
    fn apply_near_duplicate_stage(
        &self,
        survivors: Vec<Document>,
        rejected: &mut Vec<RejectedDocument>,
    ) -> Result<(Vec<Document>, Vec<NearDuplicateCluster>), CurationError> {
        let near_duplicate_clusters = if self.config.near_dup.enabled {
            find_near_duplicate_clusters(&survivors, &self.config.near_dup)?
        } else {
            Vec::new()
        };

        let mut duplicate_of: HashMap<String, String> = HashMap::new();
        for cluster in &near_duplicate_clusters {
            for member in cluster.duplicates() {
                duplicate_of.insert(member.to_string(), cluster.representative_id.clone());
            }
        }

        let mut after_dedup: Vec<Document> = Vec::new();
        for doc in survivors {
            let id = doc.id.as_str().to_string();
            if let Some(representative) = duplicate_of.get(&id) {
                rejected.push(RejectedDocument {
                    document_id: id,
                    stage: RejectionStage::NearDuplicate,
                    reasons: vec![format!(
                        "near-duplicate of admitted document {representative}"
                    )],
                });
            } else {
                after_dedup.push(doc);
            }
        }

        Ok((after_dedup, near_duplicate_clusters))
    }

    /// Stage 3 of [`Self::curate`]: when `benchmark` is supplied, check
    /// `survivors` for contamination and (when
    /// [`super::contamination::DecontaminationAction::Drop`] is configured)
    /// drop implicated documents, pushing a [`RejectedDocument`] for each
    /// onto `rejected`. Returns the documents that proceed to admission, the
    /// flagged-contaminated ids, and the contamination report (`None` when no
    /// benchmark was supplied).
    fn apply_contamination_stage(
        &self,
        survivors: Vec<Document>,
        benchmark: Option<&[BenchmarkItem]>,
        rejected: &mut Vec<RejectedDocument>,
    ) -> Result<ContaminationStageOutcome, CurationError> {
        let Some(benchmark_items) = benchmark else {
            return Ok((survivors, Vec::new(), None));
        };

        let detector = ContaminationDetector::new(self.config.contamination)?;
        let report = detector.detect(benchmark_items, &survivors);
        let flagged_contaminated_ids = report.dirty_document_ids.clone();

        let final_survivors = if report.action == DecontaminationAction::Drop {
            let dirty: HashSet<&str> = report
                .dirty_document_ids
                .iter()
                .map(String::as_str)
                .collect();
            let mut kept = Vec::with_capacity(survivors.len());
            for doc in survivors {
                let id = doc.id.as_str().to_string();
                if dirty.contains(id.as_str()) {
                    let reasons: Vec<String> = report
                        .verdicts
                        .iter()
                        .filter(|v| v.matches.iter().any(|m| m.document_id == id))
                        .map(|v| format!("matches benchmark item {}", v.benchmark_id))
                        .collect();
                    rejected.push(RejectedDocument {
                        document_id: id,
                        stage: RejectionStage::Contamination,
                        reasons,
                    });
                } else {
                    kept.push(doc);
                }
            }
            kept
        } else {
            survivors
        };

        Ok((final_survivors, flagged_contaminated_ids, Some(report)))
    }

    /// The pillar 1 + 2 assessment for one document, shared by
    /// [`Self::score_quality`] and [`Self::quality_report`].
    fn assess_document(&self, doc: &Document) -> Result<DocumentQuality, CurationError> {
        let verdict = self.assess_heuristics(doc);
        let heuristic_pass_rate = verdict.pass_rate();

        let mut classifier_probability = None;
        let mut combined = heuristic_pass_rate;
        let mut classifier_admits = true;

        if self.config.use_classifier {
            let classifier = self
                .classifier
                .as_ref()
                .ok_or(CurationError::NoClassifierTrained)?;
            let probability = classifier.predict_probability(&doc.content);
            let weight = self.config.classifier_weight.clamp(0.0, 1.0);
            combined = (1.0 - weight).mul_add(heuristic_pass_rate, weight * probability);
            classifier_admits = probability >= classifier.decision_threshold;
            classifier_probability = Some(probability);
        }

        let admitted = verdict.passed && classifier_admits;

        Ok(DocumentQuality {
            document_id: doc.id.as_str().to_string(),
            verdict,
            score: QualityScore {
                heuristic_pass_rate,
                classifier_probability,
                combined,
            },
            admitted,
        })
    }
}

impl Default for CorpusCurator {
    /// A curator with default configuration and no classifier attached.
    fn default() -> Self {
        Self::new(CurationConfig::default())
    }
}
