//! Shared, top-level types for `corpus_curation`: the aggregate configuration,
//! the error type, and the report types that tie the four pillars together.
//!
//! Each pillar owns its own configuration and result types in its own file
//! ([`super::heuristics::QualityRule`], [`super::classifier::ClassifierConfig`],
//! [`super::contamination::ContaminationConfig`],
//! [`super::near_dup::NearDupConfig`]); this file holds the vocabulary that
//! spans them: [`CurationConfig`] aggregates the four sub-configurations,
//! [`CurationError`] is the one error type every fallible operation in the
//! module returns, and [`QualityReport`]/[`CorpusReport`]/[`CurationReport`]
//! are the outputs [`super::engine::CorpusCurator`] produces by combining the
//! pillars.

use thiserror::Error;

use super::classifier::ClassifierConfig;
use super::contamination::{ContaminationConfig, ContaminationReport};
use super::heuristics::{QualityRule, QualityVerdict};
use super::near_dup::{NearDupConfig, NearDuplicateCluster};

// ── CurationError ────────────────────────────────────────────────────────────

/// Errors produced by the `corpus_curation` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CurationError {
    /// A configuration value was out of its valid range.
    #[error("configuration is invalid: {0}")]
    InvalidConfig(String),
    /// [`super::classifier::QualityClassifier::train`] was called with no
    /// training examples.
    #[error("training set must contain at least one example")]
    EmptyTrainingSet,
    /// [`CurationConfig::use_classifier`] is `true` but
    /// [`super::engine::CorpusCurator`] has no trained or attached
    /// classifier.
    #[error("the classifier stage is enabled but no classifier has been trained or attached")]
    NoClassifierTrained,
    /// The underlying [`crate::lsh_index::MinHashIndex`] composition used by
    /// [`super::near_dup`] failed (construction, insertion, or search).
    #[error("near-duplicate index operation failed: {0}")]
    NearDupIndexFailed(String),
}

// ── CurationConfig ───────────────────────────────────────────────────────────

/// Aggregate configuration for [`super::engine::CorpusCurator`], combining all
/// four pillars.
#[derive(Debug, Clone, PartialEq)]
pub struct CurationConfig {
    /// The heuristic rule set (pillar 1). Defaults to
    /// [`QualityRule::default_rule_set`].
    pub quality_rules: Vec<QualityRule>,
    /// Whether the trained classifier stage (pillar 2) participates in the
    /// admission decision. When `true`, a classifier must be trained or
    /// attached before curation runs. Defaults to `false`.
    pub use_classifier: bool,
    /// Weight of the classifier probability in [`QualityScore::combined`],
    /// in `[0, 1]`. The heuristic pass rate carries the remaining weight.
    /// Defaults to `0.5`. Ignored when `use_classifier` is `false`.
    pub classifier_weight: f64,
    /// Hyperparameters for training the pillar-2 classifier.
    pub classifier: ClassifierConfig,
    /// Configuration for the near-duplicate pillar (pillar 4).
    pub near_dup: NearDupConfig,
    /// Configuration for the contamination pillar (pillar 3), used only when
    /// a benchmark is supplied to [`super::engine::CorpusCurator::curate`].
    pub contamination: ContaminationConfig,
}

impl Default for CurationConfig {
    fn default() -> Self {
        Self {
            quality_rules: QualityRule::default_rule_set(),
            use_classifier: false,
            classifier_weight: 0.5,
            classifier: ClassifierConfig::default(),
            near_dup: NearDupConfig::default(),
            contamination: ContaminationConfig::default(),
        }
    }
}

impl CurationConfig {
    /// Construct a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the heuristic rule set.
    #[must_use]
    pub fn with_quality_rules(mut self, quality_rules: Vec<QualityRule>) -> Self {
        self.quality_rules = quality_rules;
        self
    }

    /// Enable or disable the classifier stage.
    #[must_use]
    pub fn with_use_classifier(mut self, use_classifier: bool) -> Self {
        self.use_classifier = use_classifier;
        self
    }

    /// Set the classifier's weight in the blended [`QualityScore::combined`].
    #[must_use]
    pub fn with_classifier_weight(mut self, classifier_weight: f64) -> Self {
        self.classifier_weight = classifier_weight;
        self
    }

    /// Set the classifier training hyperparameters.
    #[must_use]
    pub fn with_classifier_config(mut self, classifier: ClassifierConfig) -> Self {
        self.classifier = classifier;
        self
    }

    /// Set the near-duplicate pillar configuration.
    #[must_use]
    pub fn with_near_dup(mut self, near_dup: NearDupConfig) -> Self {
        self.near_dup = near_dup;
        self
    }

    /// Set the contamination pillar configuration.
    #[must_use]
    pub fn with_contamination(mut self, contamination: ContaminationConfig) -> Self {
        self.contamination = contamination;
        self
    }
}

// ── QualityScore / DocumentQuality / QualityReport ──────────────────────────

/// A document's blended quality score: the heuristic pass rate (pillar 1)
/// optionally blended with the classifier probability (pillar 2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityScore {
    /// Fraction of heuristic rules the document passed (see
    /// [`QualityVerdict::pass_rate`]).
    pub heuristic_pass_rate: f64,
    /// The classifier's predicted "good" probability, or `None` when the
    /// classifier stage was not used.
    pub classifier_probability: Option<f64>,
    /// The blended score used for display/ranking: equal to
    /// `heuristic_pass_rate` when no classifier is used, otherwise
    /// `(1 - classifier_weight) * heuristic_pass_rate + classifier_weight *
    /// classifier_probability`.
    ///
    /// Admission itself is **not** decided from this blended number — it is a
    /// hard AND of "every heuristic rule passed" and (when enabled) "the
    /// classifier probability meets its decision threshold" (see
    /// [`DocumentQuality::admitted`]). `combined` exists for ranking and
    /// human inspection of borderline documents.
    pub combined: f64,
}

/// The pillar 1 + pillar 2 outcome for one document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentQuality {
    /// The document's id.
    pub document_id: String,
    /// The full heuristic rule verdict.
    pub verdict: QualityVerdict,
    /// The blended quality score.
    pub score: QualityScore,
    /// `true` when the document passes both the heuristic gate and (if
    /// enabled) the classifier gate.
    pub admitted: bool,
}

/// Corpus-level aggregate of the pillar 1 + pillar 2 quality pass, produced by
/// [`super::engine::CorpusCurator::quality_report`].
#[derive(Debug, Clone, PartialEq)]
pub struct QualityReport {
    /// Number of documents assessed.
    pub total: usize,
    /// Number of documents admitted by the quality gate.
    pub admitted: usize,
    /// Mean [`QualityScore::combined`] across all assessed documents.
    pub mean_combined_score: f64,
    /// Per-document detail, in input order.
    pub documents: Vec<DocumentQuality>,
}

// ── RejectionStage / RejectedDocument ────────────────────────────────────────

/// Which curation stage rejected a document. Stages run in this order, and a
/// document is attributed to the *first* stage that rejects it (later stages
/// never see a document a prior stage has already removed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionStage {
    /// Failed at least one heuristic rule (pillar 1).
    Heuristic,
    /// Passed every heuristic rule but scored below the classifier's decision
    /// threshold (pillar 2).
    Classifier,
    /// Was a non-representative member of a near-duplicate cluster
    /// (pillar 4).
    NearDuplicate,
    /// Matched a benchmark item and [`super::contamination::DecontaminationAction::Drop`]
    /// was configured (pillar 3).
    Contamination,
}

/// A document excluded from the admitted set, with the stage and reasons.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectedDocument {
    /// The rejected document's id.
    pub document_id: String,
    /// The stage that rejected it.
    pub stage: RejectionStage,
    /// Human-readable reasons (e.g. failed rule descriptions, the id of the
    /// cluster representative it duplicates, or the benchmark item it
    /// matches).
    pub reasons: Vec<String>,
}

// ── CorpusReport / CurationReport ────────────────────────────────────────────

/// A flattened, numeric summary of a [`CurationReport`] — counts and rates
/// only, no per-document detail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorpusReport {
    /// Total documents considered.
    pub total_documents: usize,
    /// Documents admitted after every configured stage.
    pub admitted: usize,
    /// Documents rejected for failing a heuristic rule.
    pub rejected_heuristic: usize,
    /// Documents rejected for scoring below the classifier threshold.
    pub rejected_classifier: usize,
    /// Documents rejected as non-representative near-duplicates.
    pub rejected_near_duplicate: usize,
    /// Documents rejected (dropped) for benchmark contamination.
    pub rejected_contamination: usize,
    /// Number of near-duplicate clusters found (each with two or more
    /// members).
    pub duplicate_cluster_count: usize,
    /// The contamination pillar's corpus-level contamination rate; `0.0` when
    /// no benchmark was supplied.
    pub contamination_rate: f64,
    /// Mean [`QualityScore::combined`] across all considered documents.
    pub mean_quality_score: f64,
}

/// The full result of [`super::engine::CorpusCurator::curate`]: every
/// pillar's detailed output plus the final admit/reject decision.
#[derive(Debug, Clone, PartialEq)]
pub struct CurationReport {
    /// The pillar 1 + 2 quality pass over every input document.
    pub quality: QualityReport,
    /// Near-duplicate clusters found among the quality-stage survivors.
    pub near_duplicate_clusters: Vec<NearDuplicateCluster>,
    /// The contamination pass over the near-duplicate-stage survivors, or
    /// `None` when no benchmark was supplied to `curate`.
    pub contamination: Option<ContaminationReport>,
    /// Ids of documents admitted through every configured stage, in input
    /// order.
    pub admitted_ids: Vec<String>,
    /// Every rejected document, with its stage and reasons.
    pub rejected: Vec<RejectedDocument>,
    /// Ids implicated by the contamination pass regardless of the configured
    /// [`super::contamination::DecontaminationAction`] (populated even when
    /// the action is `Flag` and the documents therefore remain admitted).
    pub flagged_contaminated_ids: Vec<String>,
    /// A flattened numeric summary of this report.
    pub summary: CorpusReport,
}

impl CurationReport {
    /// `true` when `document_id` is in [`Self::admitted_ids`].
    #[must_use]
    pub fn is_admitted(&self, document_id: &str) -> bool {
        self.admitted_ids.iter().any(|id| id == document_id)
    }

    /// The [`RejectedDocument`] record for `document_id`, if it was rejected.
    #[must_use]
    pub fn rejection(&self, document_id: &str) -> Option<&RejectedDocument> {
        self.rejected.iter().find(|r| r.document_id == document_id)
    }
}
