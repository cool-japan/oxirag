//! Enums and structs representing incremental consistency state.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ConsistencyResult
// ---------------------------------------------------------------------------

/// Result of a consistency check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsistencyResult {
    /// All claims are consistent with each other.
    Consistent,
    /// Claims are inconsistent; contains the list of conflicts.
    Inconsistent(Vec<ClaimConflict>),
    /// Consistency cannot be determined.
    Unknown,
}

impl ConsistencyResult {
    /// Returns true if the result is consistent.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        matches!(self, Self::Consistent)
    }

    /// Returns true if the result is inconsistent.
    #[must_use]
    pub fn is_inconsistent(&self) -> bool {
        matches!(self, Self::Inconsistent(_))
    }

    /// Returns the conflicts if inconsistent, empty vec otherwise.
    #[must_use]
    pub fn conflicts(&self) -> Vec<ClaimConflict> {
        match self {
            Self::Inconsistent(conflicts) => conflicts.clone(),
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ConflictType
// ---------------------------------------------------------------------------

/// Types of conflicts that can be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    /// Direct contradiction (A and NOT A).
    DirectContradiction,
    /// Semantic contradiction (claims mean opposite things).
    SemanticContradiction,
    /// Comparison conflict (e.g., A > B and B > A).
    ComparisonConflict,
    /// Temporal conflict (e.g., A before B and B before A).
    TemporalConflict,
    /// Predicate conflict (same subject, contradictory predicates).
    PredicateConflict,
    /// Quantifier conflict (forall vs exists contradiction).
    QuantifierConflict,
    /// Causal conflict (conflicting cause-effect relationships).
    CausalConflict,
    /// Modal conflict (necessary and impossible).
    ModalConflict,
}

impl ConflictType {
    /// Get the default severity for this conflict type.
    #[must_use]
    pub fn default_severity(self) -> f32 {
        match self {
            Self::DirectContradiction => 1.0,
            Self::SemanticContradiction => 0.9,
            Self::ComparisonConflict => 0.85,
            Self::TemporalConflict => 0.8,
            Self::PredicateConflict => 0.75,
            Self::QuantifierConflict | Self::CausalConflict => 0.7,
            Self::ModalConflict => 0.65,
        }
    }
}

// ---------------------------------------------------------------------------
// ClaimConflict
// ---------------------------------------------------------------------------

/// A conflict between two claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimConflict {
    /// ID of the first conflicting claim.
    pub claim1_id: String,
    /// ID of the second conflicting claim.
    pub claim2_id: String,
    /// The type of conflict detected.
    pub conflict_type: ConflictType,
    /// Human-readable explanation of the conflict.
    pub explanation: String,
    /// Severity of the conflict (0.0 to 1.0, higher is more severe).
    pub severity: f32,
}

impl ClaimConflict {
    /// Create a new claim conflict.
    #[must_use]
    pub fn new(
        claim1_id: impl Into<String>,
        claim2_id: impl Into<String>,
        conflict_type: ConflictType,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            claim1_id: claim1_id.into(),
            claim2_id: claim2_id.into(),
            conflict_type,
            explanation: explanation.into(),
            severity: conflict_type.default_severity(),
        }
    }

    /// Set the severity level.
    #[must_use]
    pub fn with_severity(mut self, severity: f32) -> Self {
        self.severity = severity.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Suggested resolution for a conflict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Resolution {
    /// Remove one of the conflicting claims.
    RemoveClaim {
        /// ID of the claim to remove.
        claim_id: String,
        /// Reason for removing this claim.
        reason: String,
    },
    /// Modify a claim to resolve the conflict.
    ModifyClaim {
        /// ID of the claim to modify.
        claim_id: String,
        /// Suggested modification.
        suggestion: String,
    },
    /// Add an exception to make claims compatible.
    AddException {
        /// The exception claim to add.
        exception: String,
        /// Explanation of the exception.
        explanation: String,
    },
    /// Prioritize one claim over another based on confidence.
    PrioritizeByCofidence {
        /// ID of the higher-confidence claim to keep.
        keep_claim_id: String,
        /// ID of the lower-confidence claim to remove.
        remove_claim_id: String,
    },
}
