//! Core data structures for `OxiRAG`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::layer1_echo::filter::MetadataFilter;

/// A unique identifier for a document.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(pub String);

impl DocumentId {
    /// Create a new random document ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a document ID from an existing string.
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the inner string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for DocumentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for DocumentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DocumentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// A document that can be indexed and searched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for the document.
    pub id: DocumentId,
    /// The main content of the document.
    pub content: String,
    /// Optional title.
    pub title: Option<String>,
    /// Optional source URL or path.
    pub source: Option<String>,
    /// Metadata key-value pairs.
    pub metadata: HashMap<String, String>,
    /// When the document was created.
    pub created_at: DateTime<Utc>,
    /// When the document was last updated.
    pub updated_at: DateTime<Utc>,
}

impl Document {
    /// Create a new document with the given content.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: DocumentId::new(),
            content: content.into(),
            title: None,
            source: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Set the document ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<DocumentId>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the document title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the document source.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add a metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// The query text.
    pub text: String,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Minimum similarity score threshold (0.0 to 1.0).
    pub min_score: Option<f32>,
    /// Simple key-value metadata filters (legacy, for basic equality matching).
    pub filters: HashMap<String, String>,
    /// Advanced metadata filter for complex filtering expressions.
    pub metadata_filter: Option<MetadataFilter>,
}

impl Query {
    /// Create a new query with the given text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            top_k: 10,
            min_score: None,
            filters: HashMap::new(),
            metadata_filter: None,
        }
    }

    /// Set the maximum number of results.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the minimum similarity score.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Add a simple key-value metadata filter (equality).
    #[must_use]
    pub fn with_filter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.filters.insert(key.into(), value.into());
        self
    }

    /// Set an advanced metadata filter.
    ///
    /// This allows for complex filtering expressions using AND, OR, and various
    /// comparison operators like Eq, Ne, Contains, Exists, etc.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::types::Query;
    /// use oxirag::layer1_echo::MetadataFilter;
    ///
    /// let query = Query::new("search text")
    ///     .with_metadata_filter(MetadataFilter::and(vec![
    ///         MetadataFilter::eq("status", "published"),
    ///         MetadataFilter::or(vec![
    ///             MetadataFilter::eq("category", "science"),
    ///             MetadataFilter::eq("category", "technology"),
    ///         ]),
    ///     ]));
    /// ```
    #[must_use]
    pub fn with_metadata_filter(mut self, filter: MetadataFilter) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    /// Get a reference to the metadata filter if set.
    #[must_use]
    pub fn metadata_filter(&self) -> Option<&MetadataFilter> {
        self.metadata_filter.as_ref()
    }
}

/// A search result from the Echo layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matched document.
    pub document: Document,
    /// Similarity score (0.0 to 1.0, higher is better).
    pub score: f32,
    /// Rank in the result set (0-indexed).
    pub rank: usize,
}

impl SearchResult {
    /// Create a new search result.
    #[must_use]
    pub fn new(document: Document, score: f32, rank: usize) -> Self {
        Self {
            document,
            score,
            rank,
        }
    }
}

/// A draft answer generated from retrieved documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Draft {
    /// The draft answer text.
    pub content: String,
    /// The query that produced this draft.
    pub query: String,
    /// Source documents used to generate the draft.
    pub sources: Vec<DocumentId>,
    /// Confidence score from the generator (0.0 to 1.0).
    pub confidence: f32,
    /// When the draft was generated.
    pub generated_at: DateTime<Utc>,
}

impl Draft {
    /// Create a new draft.
    #[must_use]
    pub fn new(content: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            query: query.into(),
            sources: Vec::new(),
            confidence: 0.0,
            generated_at: Utc::now(),
        }
    }

    /// Add a source document ID.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<DocumentId>) -> Self {
        self.sources.push(source.into());
        self
    }

    /// Set the confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Result of speculative verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeculationDecision {
    /// The draft is accepted as-is.
    Accept,
    /// The draft needs revision.
    Revise,
    /// The draft should be rejected.
    Reject,
}

/// Result from the Speculator layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationResult {
    /// The decision made by the speculator.
    pub decision: SpeculationDecision,
    /// Confidence in the decision (0.0 to 1.0).
    pub confidence: f32,
    /// Explanation for the decision.
    pub explanation: String,
    /// Suggested revisions if decision is Revise.
    pub suggested_revisions: Option<String>,
    /// Issues found in the draft.
    pub issues: Vec<String>,
}

impl SpeculationResult {
    /// Create a new speculation result.
    #[must_use]
    pub fn new(decision: SpeculationDecision, confidence: f32) -> Self {
        Self {
            decision,
            confidence,
            explanation: String::new(),
            suggested_revisions: None,
            issues: Vec::new(),
        }
    }

    /// Set the explanation.
    #[must_use]
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }

    /// Set suggested revisions.
    #[must_use]
    pub fn with_revisions(mut self, revisions: impl Into<String>) -> Self {
        self.suggested_revisions = Some(revisions.into());
        self
    }

    /// Add an issue.
    #[must_use]
    pub fn with_issue(mut self, issue: impl Into<String>) -> Self {
        self.issues.push(issue.into());
        self
    }
}

/// A logical claim extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalClaim {
    /// Unique identifier for this claim.
    pub id: String,
    /// The original text of the claim.
    pub text: String,
    /// Structured representation suitable for SMT encoding.
    pub structure: ClaimStructure,
    /// Confidence in the extraction (0.0 to 1.0).
    pub confidence: f32,
    /// Source span in the original text.
    pub source_span: Option<(usize, usize)>,
}

impl LogicalClaim {
    /// Create a new logical claim.
    #[must_use]
    pub fn new(text: impl Into<String>, structure: ClaimStructure) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            text: text.into(),
            structure,
            confidence: 1.0,
            source_span: None,
        }
    }

    /// Set the confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the source span.
    #[must_use]
    pub fn with_span(mut self, start: usize, end: usize) -> Self {
        self.source_span = Some((start, end));
        self
    }
}

/// Structured representation of a claim for SMT encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClaimStructure {
    /// A simple predicate: subject has property.
    Predicate {
        /// The subject of the predicate.
        subject: String,
        /// The predicate (verb or property).
        predicate: String,
        /// Optional object of the predicate.
        object: Option<String>,
    },
    /// A comparison: left op right.
    Comparison {
        /// Left operand.
        left: String,
        /// Comparison operator.
        operator: ComparisonOp,
        /// Right operand.
        right: String,
    },
    /// A quantified statement.
    Quantified {
        /// The quantifier (forall or exists).
        quantifier: Quantifier,
        /// The bound variable name.
        variable: String,
        /// The domain/sort of the variable.
        domain: String,
        /// The body of the quantified statement.
        body: Box<ClaimStructure>,
    },
    /// Logical conjunction (AND).
    And(Vec<ClaimStructure>),
    /// Logical disjunction (OR).
    Or(Vec<ClaimStructure>),
    /// Logical negation (NOT).
    Not(Box<ClaimStructure>),
    /// Implication: if premise then conclusion.
    Implies {
        /// The premise/antecedent.
        premise: Box<ClaimStructure>,
        /// The conclusion/consequent.
        conclusion: Box<ClaimStructure>,
    },
    /// Temporal claim about time relationships.
    Temporal {
        /// The event being described.
        event: String,
        /// The time relation (before, after, during, simultaneous).
        time_relation: TimeRelation,
        /// The reference point for the temporal relation.
        reference: String,
    },
    /// Causal claim about cause-effect relationships.
    Causal {
        /// The cause.
        cause: Box<ClaimStructure>,
        /// The effect.
        effect: Box<ClaimStructure>,
        /// The strength of the causal relationship.
        strength: CausalStrength,
    },
    /// Modal claim with possibility/necessity modifiers.
    Modal {
        /// The underlying claim.
        claim: Box<ClaimStructure>,
        /// The modality (possible, necessary, likely, unlikely).
        modality: Modality,
    },
    /// A raw SMT-LIB expression.
    Raw(String),
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    /// Equal (=).
    Equal,
    /// Not equal (distinct).
    NotEqual,
    /// Less than (<).
    LessThan,
    /// Less than or equal (<=).
    LessOrEqual,
    /// Greater than (>).
    GreaterThan,
    /// Greater than or equal (>=).
    GreaterOrEqual,
}

impl ComparisonOp {
    /// Convert to SMT-LIB operator string.
    #[must_use]
    pub fn to_smtlib(&self) -> &'static str {
        match self {
            Self::Equal => "=",
            Self::NotEqual => "distinct",
            Self::LessThan => "<",
            Self::LessOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterOrEqual => ">=",
        }
    }
}

/// Quantifiers for logical statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantifier {
    /// Universal quantifier (forall).
    ForAll,
    /// Existential quantifier (exists).
    Exists,
}

impl Quantifier {
    /// Convert to SMT-LIB quantifier string.
    #[must_use]
    pub fn to_smtlib(&self) -> &'static str {
        match self {
            Self::ForAll => "forall",
            Self::Exists => "exists",
        }
    }
}

/// Time relationship for temporal claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeRelation {
    /// Event occurs before the reference.
    Before,
    /// Event occurs after the reference.
    After,
    /// Event occurs during the reference.
    During,
    /// Event occurs at the same time as the reference.
    Simultaneous,
}

impl TimeRelation {
    /// Convert to SMT-LIB relation string.
    #[must_use]
    pub fn to_smtlib(&self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::After => "after",
            Self::During => "during",
            Self::Simultaneous => "simultaneous",
        }
    }
}

/// Strength of causal relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalStrength {
    /// A directly causes B.
    Direct,
    /// A contributes to B (indirect causation).
    Indirect,
    /// A and B are correlated (not necessarily causal).
    Correlated,
}

impl CausalStrength {
    /// Convert to SMT-LIB strength string.
    #[must_use]
    pub fn to_smtlib(&self) -> &'static str {
        match self {
            Self::Direct => "causes",
            Self::Indirect => "contributes_to",
            Self::Correlated => "correlates_with",
        }
    }
}

/// Modal operators for possibility/necessity claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    /// Might be true (possibility).
    Possible,
    /// Must be true (necessity).
    Necessary,
    /// Probably true (likelihood).
    Likely,
    /// Probably false (unlikelihood).
    Unlikely,
}

impl Modality {
    /// Convert to SMT-LIB modality string.
    #[must_use]
    pub fn to_smtlib(&self) -> &'static str {
        match self {
            Self::Possible => "possibly",
            Self::Necessary => "necessarily",
            Self::Likely => "likely",
            Self::Unlikely => "unlikely",
        }
    }
}

/// Status of a verification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// The claim is verified as true.
    Verified,
    /// The claim is verified as false.
    Falsified,
    /// The verification was inconclusive.
    Unknown,
    /// Verification timed out.
    Timeout,
    /// An error occurred during verification.
    Error,
}

/// Result of verifying a single claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimVerificationResult {
    /// The claim that was verified.
    pub claim: LogicalClaim,
    /// The verification status.
    pub status: VerificationStatus,
    /// Explanation or counterexample.
    pub explanation: Option<String>,
    /// Time taken for verification in milliseconds.
    pub duration_ms: u64,
}

impl ClaimVerificationResult {
    /// Create a new claim verification result.
    #[must_use]
    pub fn new(claim: LogicalClaim, status: VerificationStatus) -> Self {
        Self {
            claim,
            status,
            explanation: None,
            duration_ms: 0,
        }
    }

    /// Set the explanation.
    #[must_use]
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }

    /// Set the duration.
    #[must_use]
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }
}

/// Overall result from the Judge layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Overall status of the verification.
    pub status: VerificationStatus,
    /// Results for individual claims.
    pub claim_results: Vec<ClaimVerificationResult>,
    /// Overall confidence (0.0 to 1.0).
    pub confidence: f32,
    /// Summary of the verification.
    pub summary: String,
    /// Total time taken in milliseconds.
    pub total_duration_ms: u64,
}

impl VerificationResult {
    /// Create a new verification result.
    #[must_use]
    pub fn new(status: VerificationStatus) -> Self {
        Self {
            status,
            claim_results: Vec::new(),
            confidence: 0.0,
            summary: String::new(),
            total_duration_ms: 0,
        }
    }

    /// Add a claim result.
    #[must_use]
    pub fn with_claim_result(mut self, result: ClaimVerificationResult) -> Self {
        self.claim_results.push(result);
        self
    }

    /// Set the confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the summary.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }

    /// Set the total duration.
    #[must_use]
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.total_duration_ms = duration_ms;
        self
    }
}

/// The final output from the RAG pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOutput {
    /// The original query.
    pub query: Query,
    /// Search results from Echo layer.
    pub search_results: Vec<SearchResult>,
    /// The draft answer.
    pub draft: Draft,
    /// Speculation result from Speculator layer.
    pub speculation: Option<SpeculationResult>,
    /// Verification result from Judge layer.
    pub verification: Option<VerificationResult>,
    /// The final answer after all processing.
    pub final_answer: String,
    /// Overall pipeline confidence.
    pub confidence: f32,
    /// Which layers were used.
    pub layers_used: Vec<String>,
    /// Total pipeline execution time in milliseconds.
    pub total_duration_ms: u64,
}

impl PipelineOutput {
    /// Create a new pipeline output.
    #[must_use]
    pub fn new(query: Query, draft: Draft) -> Self {
        Self {
            query,
            search_results: Vec::new(),
            draft: draft.clone(),
            speculation: None,
            verification: None,
            final_answer: draft.content,
            confidence: 0.0,
            layers_used: Vec::new(),
            total_duration_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_id_generation() {
        let id1 = DocumentId::new();
        let id2 = DocumentId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_document_builder() {
        let doc = Document::new("Test content")
            .with_title("Test Title")
            .with_source("test.txt")
            .with_metadata("key", "value");

        assert_eq!(doc.content, "Test content");
        assert_eq!(doc.title, Some("Test Title".to_string()));
        assert_eq!(doc.source, Some("test.txt".to_string()));
        assert_eq!(doc.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_query_builder() {
        let query = Query::new("test query")
            .with_top_k(5)
            .with_min_score(0.5)
            .with_filter("category", "test");

        assert_eq!(query.text, "test query");
        assert_eq!(query.top_k, 5);
        assert_eq!(query.min_score, Some(0.5));
        assert_eq!(query.filters.get("category"), Some(&"test".to_string()));
    }

    #[test]
    fn test_query_with_metadata_filter() {
        let filter = MetadataFilter::and(vec![
            MetadataFilter::eq("status", "published"),
            MetadataFilter::or(vec![
                MetadataFilter::eq("category", "science"),
                MetadataFilter::eq("category", "technology"),
            ]),
        ]);

        let query = Query::new("test query").with_metadata_filter(filter.clone());

        assert_eq!(query.text, "test query");
        assert!(query.metadata_filter.is_some());
        assert_eq!(query.metadata_filter(), Some(&filter));
    }

    #[test]
    fn test_query_metadata_filter_none_by_default() {
        let query = Query::new("test query");
        assert!(query.metadata_filter.is_none());
        assert!(query.metadata_filter().is_none());
    }

    #[test]
    fn test_comparison_op_smtlib() {
        assert_eq!(ComparisonOp::Equal.to_smtlib(), "=");
        assert_eq!(ComparisonOp::LessThan.to_smtlib(), "<");
        assert_eq!(ComparisonOp::GreaterOrEqual.to_smtlib(), ">=");
    }

    #[test]
    fn test_quantifier_smtlib() {
        assert_eq!(Quantifier::ForAll.to_smtlib(), "forall");
        assert_eq!(Quantifier::Exists.to_smtlib(), "exists");
    }
}
