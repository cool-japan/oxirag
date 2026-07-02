//! Types, configuration, and errors for the `crud_rag` module.
//!
//! - [`CrudOperation`] — the four CRUD operation types a case is tagged with.
//! - [`CrudCase`] — a single evaluation case: the system output, its
//!   reference(s), and any operation-specific fields (key points for
//!   [`CrudOperation::Delete`], error/correct spans for
//!   [`CrudOperation::Update`]).
//! - [`CrudMetric`] — the named metrics an operation's scorer can produce.
//! - [`CrudScore`] — one fully-identified metric value: which case, which
//!   operation, which metric, and its value.
//! - [`CrudRagConfig`] — n-gram order, ROUGE beta, text normalization,
//!   redundancy penalty weight, and which operations are enabled.
//! - [`CrudRagReport`] — the aggregated output of a
//!   [`crate::crud_rag::CrudRagHarness`] run: every [`CrudScore`] produced,
//!   plus per-operation means and an overall score.
//! - [`CrudRagError`] / [`CrudRagResult`] — the module's error type and
//!   result alias.

use thiserror::Error;

// ── CrudOperation ────────────────────────────────────────────────────────────

/// One of the four CRUD operation types a [`CrudCase`] is evaluated under.
///
/// Each operation models a distinct RAG task and is scored by its own
/// metric(s) — see the [module docs](crate::crud_rag) for the full mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrudOperation {
    /// Continuation / creative generation: extend or generate text that
    /// should read like the reference continuation. Scored by
    /// [`CrudMetric::RougeL`] and [`CrudMetric::Bleu`].
    Create,
    /// Single-document question answering: answer a question from one
    /// document. Scored by [`CrudMetric::ExactMatch`] and
    /// [`CrudMetric::TokenF1`].
    Read,
    /// Hallucination / error correction: fix an injected error in a text.
    /// Scored by [`CrudMetric::CorrectionSimilarity`],
    /// [`CrudMetric::ErrorRemoval`], and [`CrudMetric::CorrectSpanIntroduced`].
    Update,
    /// Multi-document summarization / redundancy removal: summarize several
    /// documents into one, non-redundant digest. Scored by
    /// [`CrudMetric::Coverage`] and [`CrudMetric::Redundancy`].
    Delete,
}

impl CrudOperation {
    /// Return a stable, lowercase string label for this operation.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Read => "read",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }

    /// Return all four operations, in `Create, Read, Update, Delete` order.
    #[must_use]
    pub fn all() -> [CrudOperation; 4] {
        [Self::Create, Self::Read, Self::Update, Self::Delete]
    }
}

impl std::fmt::Display for CrudOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ── CrudMetric ───────────────────────────────────────────────────────────────

/// A named metric produced while scoring a [`CrudCase`].
///
/// Every operation additionally reports [`CrudMetric::Combined`] — the
/// single operation-specific primary score for the case, which
/// [`CrudRagReport`] averages into its per-operation means and overall
/// score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrudMetric {
    /// [`CrudOperation::Create`]: ROUGE-L longest-common-subsequence F-beta
    /// score against the best-matching reference.
    RougeL,
    /// [`CrudOperation::Create`]: BLEU-style clipped n-gram precision (up to
    /// the configured order), scaled by a brevity penalty, against the
    /// best-matching reference.
    Bleu,
    /// [`CrudOperation::Read`]: `1.0` when the normalized output exactly
    /// equals the best-matching normalized reference, else `0.0`.
    ExactMatch,
    /// [`CrudOperation::Read`]: token-level F1 (precision/recall over
    /// normalized tokens) against the best-matching reference.
    TokenF1,
    /// [`CrudOperation::Update`]: token-level F1 similarity of the corrected
    /// output against the best-matching corrected reference.
    CorrectionSimilarity,
    /// [`CrudOperation::Update`]: `1.0` when the erroneous span no longer
    /// appears (after normalization) in the output, else `0.0`.
    ErrorRemoval,
    /// [`CrudOperation::Update`]: `1.0` when the correct replacement span
    /// appears (after normalization) in the output, else `0.0`.
    CorrectSpanIntroduced,
    /// [`CrudOperation::Delete`]: fraction of reference key points whose
    /// content tokens are fully covered by the summary.
    Coverage,
    /// [`CrudOperation::Delete`]: the summary's repeated-content ratio (mean
    /// pairwise content-token Jaccard similarity across its sentences).
    Redundancy,
    /// Every operation: the operation-specific combined/primary score for
    /// the case.
    Combined,
}

impl CrudMetric {
    /// Return a stable, lowercase `snake_case` string label for this metric.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::RougeL => "rouge_l",
            Self::Bleu => "bleu",
            Self::ExactMatch => "exact_match",
            Self::TokenF1 => "token_f1",
            Self::CorrectionSimilarity => "correction_similarity",
            Self::ErrorRemoval => "error_removal",
            Self::CorrectSpanIntroduced => "correct_span_introduced",
            Self::Coverage => "coverage",
            Self::Redundancy => "redundancy",
            Self::Combined => "combined",
        }
    }
}

// ── CrudCase ─────────────────────────────────────────────────────────────────

/// A single evaluation case tagged with its [`CrudOperation`].
///
/// `output` is the system-produced text being scored; `references` holds one
/// or more acceptable gold texts (when more than one is given, scoring uses
/// whichever reference yields the best [`CrudMetric::Combined`] score).
/// `key_points` is used only by [`CrudOperation::Delete`]; `error_span` and
/// `correct_span` are used only by [`CrudOperation::Update`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrudCase {
    /// A stable case identifier, used only for reporting and error messages.
    pub id: String,
    /// Which CRUD operation this case is evaluated under.
    pub operation: CrudOperation,
    /// The system-produced output being scored.
    pub output: String,
    /// One or more acceptable reference texts. Required (non-empty) for
    /// [`CrudOperation::Create`], [`CrudOperation::Read`], and
    /// [`CrudOperation::Update`]; unused by [`CrudOperation::Delete`].
    pub references: Vec<String>,
    /// [`CrudOperation::Delete`]-only: the key points a summary is expected
    /// to cover. Required (non-empty) for `Delete`.
    pub key_points: Vec<String>,
    /// [`CrudOperation::Update`]-only: the erroneous text injected into the
    /// original (pre-correction) text. Required for `Update`.
    pub error_span: Option<String>,
    /// [`CrudOperation::Update`]-only: the correct text that should replace
    /// `error_span`. Required for `Update`.
    pub correct_span: Option<String>,
}

impl CrudCase {
    /// Create a new case with no references, key points, or spans.
    #[must_use]
    pub fn new(id: impl Into<String>, operation: CrudOperation, output: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            operation,
            output: output.into(),
            references: Vec::new(),
            key_points: Vec::new(),
            error_span: None,
            correct_span: None,
        }
    }

    /// Append a single reference text (builder).
    #[must_use]
    pub fn with_reference(mut self, reference: impl Into<String>) -> Self {
        self.references.push(reference.into());
        self
    }

    /// Replace the full set of reference texts (builder).
    #[must_use]
    pub fn with_references(mut self, references: Vec<String>) -> Self {
        self.references = references;
        self
    }

    /// Set the [`CrudOperation::Delete`] key points (builder).
    #[must_use]
    pub fn with_key_points(mut self, key_points: Vec<String>) -> Self {
        self.key_points = key_points;
        self
    }

    /// Set the [`CrudOperation::Update`] erroneous span (builder).
    #[must_use]
    pub fn with_error_span(mut self, error_span: impl Into<String>) -> Self {
        self.error_span = Some(error_span.into());
        self
    }

    /// Set the [`CrudOperation::Update`] correct replacement span (builder).
    #[must_use]
    pub fn with_correct_span(mut self, correct_span: impl Into<String>) -> Self {
        self.correct_span = Some(correct_span.into());
        self
    }
}

// ── CrudScore ────────────────────────────────────────────────────────────────

/// A single fully-identified metric value produced while scoring a
/// [`CrudCase`]: which case, which operation, which named [`CrudMetric`],
/// and its value.
///
/// Every value lies in `[0.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CrudScore {
    /// The id of the [`CrudCase`] this score was computed for.
    pub case_id: String,
    /// The operation the case was tagged with.
    pub operation: CrudOperation,
    /// Which named metric this value represents.
    pub metric: CrudMetric,
    /// The metric's value, in `[0.0, 1.0]`.
    pub value: f32,
}

// ── CrudRagConfig ────────────────────────────────────────────────────────────

/// Configuration for [`crate::crud_rag::CrudRagHarness`].
#[derive(Debug, Clone, PartialEq)]
pub struct CrudRagConfig {
    /// Maximum n-gram order considered by [`CrudMetric::Bleu`] (e.g. `4` →
    /// unigrams through 4-grams). Must be at least `1`; the effective order
    /// used for a given case is additionally capped at the candidate's own
    /// token count, so short outputs are scored over the orders they can
    /// actually produce rather than collapsing to `0.0`. Defaults to `4`.
    pub bleu_ngram_order: usize,
    /// The `beta` weight of the [`CrudMetric::RougeL`] F-beta combination:
    /// `beta > 1.0` favors recall, `beta < 1.0` favors precision, and
    /// `beta == 1.0` is the standard harmonic mean (F1). Defaults to `1.0`.
    pub rouge_beta: f32,
    /// Lowercase text before tokenizing / comparing. Defaults to `true`.
    pub lowercase: bool,
    /// Replace punctuation with whitespace before tokenizing (so adjoining
    /// words never fuse together). When `false`, punctuation stays attached
    /// to its token, which makes e.g. `"Paris."` and `"Paris"` compare as
    /// different tokens. Defaults to `true`.
    pub strip_punctuation: bool,
    /// Weight applied to [`CrudMetric::Redundancy`] when computing
    /// [`CrudOperation::Delete`]'s [`CrudMetric::Combined`] score:
    /// `combined = clamp(coverage - weight * redundancy, 0.0, 1.0)`.
    /// Defaults to `0.5`.
    pub redundancy_penalty_weight: f32,
    /// Which operations [`crate::crud_rag::CrudRagHarness`] will score; a
    /// case tagged with an operation not in this list makes
    /// [`crate::crud_rag::CrudRagHarness::evaluate_case`] return
    /// [`CrudRagError::OperationDisabled`]. Defaults to all four operations
    /// ([`CrudOperation::all`]).
    pub enabled_operations: Vec<CrudOperation>,
}

impl Default for CrudRagConfig {
    fn default() -> Self {
        Self {
            bleu_ngram_order: 4,
            rouge_beta: 1.0,
            lowercase: true,
            strip_punctuation: true,
            redundancy_penalty_weight: 0.5,
            enabled_operations: CrudOperation::all().to_vec(),
        }
    }
}

impl CrudRagConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum BLEU n-gram order (clamped to at least `1`).
    #[must_use]
    pub fn with_bleu_ngram_order(mut self, bleu_ngram_order: usize) -> Self {
        self.bleu_ngram_order = bleu_ngram_order.max(1);
        self
    }

    /// Set the ROUGE-L F-beta weight.
    #[must_use]
    pub fn with_rouge_beta(mut self, rouge_beta: f32) -> Self {
        self.rouge_beta = rouge_beta;
        self
    }

    /// Set whether text is lowercased before comparison.
    #[must_use]
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set whether punctuation is stripped before tokenizing.
    #[must_use]
    pub fn with_strip_punctuation(mut self, strip_punctuation: bool) -> Self {
        self.strip_punctuation = strip_punctuation;
        self
    }

    /// Set the [`CrudOperation::Delete`] redundancy penalty weight.
    #[must_use]
    pub fn with_redundancy_penalty_weight(mut self, redundancy_penalty_weight: f32) -> Self {
        self.redundancy_penalty_weight = redundancy_penalty_weight;
        self
    }

    /// Replace the set of enabled operations.
    #[must_use]
    pub fn with_enabled_operations(mut self, enabled_operations: Vec<CrudOperation>) -> Self {
        self.enabled_operations = enabled_operations;
        self
    }

    /// Return `true` when `operation` is in [`CrudRagConfig::enabled_operations`].
    #[must_use]
    pub fn is_enabled(&self, operation: CrudOperation) -> bool {
        self.enabled_operations.contains(&operation)
    }
}

// ── CrudRagReport ────────────────────────────────────────────────────────────

/// The aggregated output of a [`crate::crud_rag::CrudRagHarness::evaluate`]
/// run: every [`CrudScore`] produced, per-operation mean
/// [`CrudMetric::Combined`] scores, and an overall score.
///
/// A per-operation mean is `0.0` when no case of that operation was present;
/// [`CrudRagReport::overall`] is the mean over only the operations that were
/// actually present among the evaluated cases.
#[derive(Debug, Clone, PartialEq)]
pub struct CrudRagReport {
    /// Every named metric value produced across all evaluated cases
    /// (including each case's per-operation sub-metrics and its
    /// [`CrudMetric::Combined`] score).
    pub scores: Vec<CrudScore>,
    /// Mean [`CrudMetric::Combined`] score over [`CrudOperation::Create`]
    /// cases (`0.0` if none were present).
    pub create_mean: f32,
    /// Mean [`CrudMetric::Combined`] score over [`CrudOperation::Read`]
    /// cases (`0.0` if none were present).
    pub read_mean: f32,
    /// Mean [`CrudMetric::Combined`] score over [`CrudOperation::Update`]
    /// cases (`0.0` if none were present).
    pub update_mean: f32,
    /// Mean [`CrudMetric::Combined`] score over [`CrudOperation::Delete`]
    /// cases (`0.0` if none were present).
    pub delete_mean: f32,
    /// Mean of the per-operation means, restricted to operations that had
    /// at least one case (`0.0` when no cases were evaluated at all).
    pub overall: f32,
}

impl CrudRagReport {
    /// Return the mean [`CrudMetric::Combined`] score for `operation`.
    #[must_use]
    pub fn mean_for(&self, operation: CrudOperation) -> f32 {
        match operation {
            CrudOperation::Create => self.create_mean,
            CrudOperation::Read => self.read_mean,
            CrudOperation::Update => self.update_mean,
            CrudOperation::Delete => self.delete_mean,
        }
    }

    /// Return the number of cases of `operation` that contributed a
    /// [`CrudMetric::Combined`] score to this report.
    #[must_use]
    pub fn count_for(&self, operation: CrudOperation) -> usize {
        self.scores
            .iter()
            .filter(|s| s.operation == operation && s.metric == CrudMetric::Combined)
            .count()
    }

    /// Return the total number of cases (across every operation) that
    /// contributed a [`CrudMetric::Combined`] score to this report.
    #[must_use]
    pub fn total_count(&self) -> usize {
        CrudOperation::all()
            .into_iter()
            .map(|op| self.count_for(op))
            .sum()
    }
}

// ── CrudRagError / CrudRagResult ─────────────────────────────────────────────

/// Errors from the `crud_rag` module.
#[derive(Debug, Error)]
pub enum CrudRagError {
    /// No cases were supplied to
    /// [`crate::crud_rag::CrudRagHarness::evaluate`].
    #[error("no cases supplied")]
    EmptyCases,
    /// A case's `output` was empty or contained only whitespace.
    #[error("case {case_id}: output must not be empty")]
    EmptyOutput {
        /// The offending case's id.
        case_id: String,
    },
    /// A case needed at least one reference but `references` was empty.
    #[error("case {case_id}: operation {operation} requires at least one reference")]
    MissingReference {
        /// The offending case's id.
        case_id: String,
        /// The operation that required a reference.
        operation: CrudOperation,
    },
    /// A [`CrudOperation::Delete`] case's `key_points` was empty.
    #[error("case {case_id}: delete requires at least one key point")]
    MissingKeyPoints {
        /// The offending case's id.
        case_id: String,
    },
    /// A [`CrudOperation::Update`] case had no `error_span`.
    #[error("case {case_id}: update requires an error_span")]
    MissingErrorSpan {
        /// The offending case's id.
        case_id: String,
    },
    /// A [`CrudOperation::Update`] case had no `correct_span`.
    #[error("case {case_id}: update requires a correct_span")]
    MissingCorrectSpan {
        /// The offending case's id.
        case_id: String,
    },
    /// The case's operation is not in
    /// [`CrudRagConfig::enabled_operations`].
    #[error("case {case_id}: operation {operation} is disabled by config")]
    OperationDisabled {
        /// The offending case's id.
        case_id: String,
        /// The disabled operation.
        operation: CrudOperation,
    },
}

/// The result type used throughout the `crud_rag` module.
pub type CrudRagResult<T> = Result<T, CrudRagError>;
