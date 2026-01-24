//! Incremental consistency checking for logical claims.
//!
//! This module provides efficient incremental consistency checking that avoids
//! re-checking all claims on each addition. It maintains a knowledge base of
//! claims and can detect conflicts between them.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::types::{ClaimStructure, LogicalClaim};

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

/// Incremental consistency checker for logical claims.
///
/// This checker maintains a knowledge base of claims and efficiently checks
/// consistency when new claims are added. It uses incremental checking to
/// avoid re-checking all pairs of claims on each addition.
pub struct IncrementalConsistencyChecker {
    /// The knowledge base of claims, keyed by ID.
    claims: HashMap<String, LogicalClaim>,
    /// Index of normalized claim hashes for quick lookup.
    normalized_hashes: HashMap<String, u64>,
    /// Index mapping subjects to claim IDs for predicate claims.
    subject_index: HashMap<String, HashSet<String>>,
    /// Known conflicts between claims.
    known_conflicts: Vec<ClaimConflict>,
    /// Set of claim ID pairs that have been checked.
    checked_pairs: HashSet<(String, String)>,
}

impl Default for IncrementalConsistencyChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalConsistencyChecker {
    /// Create a new incremental consistency checker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
            normalized_hashes: HashMap::new(),
            subject_index: HashMap::new(),
            known_conflicts: Vec::new(),
            checked_pairs: HashSet::new(),
        }
    }

    /// Add a new claim to the knowledge base.
    ///
    /// This adds the claim and incrementally checks it against existing claims.
    /// Returns the result of checking the new claim against existing claims.
    pub fn add_claim(&mut self, claim: LogicalClaim) -> ConsistencyResult {
        let claim_id = claim.id.clone();

        // Check if claim already exists
        if self.claims.contains_key(&claim_id) {
            return self.check_consistency();
        }

        // First check the new claim against existing claims
        let new_conflicts = self.find_conflicts_with_new_claim(&claim);

        // Add the claim to our knowledge base
        self.index_claim(&claim);
        self.claims.insert(claim_id, claim);

        // Add any new conflicts
        self.known_conflicts.extend(new_conflicts.clone());

        if new_conflicts.is_empty() {
            ConsistencyResult::Consistent
        } else {
            ConsistencyResult::Inconsistent(new_conflicts)
        }
    }

    /// Check if the current knowledge base is consistent.
    ///
    /// Returns the cached consistency result based on known conflicts.
    #[must_use]
    pub fn check_consistency(&self) -> ConsistencyResult {
        if self.known_conflicts.is_empty() {
            ConsistencyResult::Consistent
        } else {
            ConsistencyResult::Inconsistent(self.known_conflicts.clone())
        }
    }

    /// Check if a new claim would be consistent with the existing knowledge base.
    ///
    /// This does not add the claim; it only checks for potential conflicts.
    #[must_use]
    pub fn check_new_claim(&self, claim: &LogicalClaim) -> ConsistencyResult {
        let conflicts = self.find_conflicts_with_new_claim(claim);

        if conflicts.is_empty() {
            ConsistencyResult::Consistent
        } else {
            ConsistencyResult::Inconsistent(conflicts)
        }
    }

    /// Get all known conflicts.
    #[must_use]
    pub fn get_conflicts(&self) -> Vec<ClaimConflict> {
        self.known_conflicts.clone()
    }

    /// Remove a claim from the knowledge base.
    ///
    /// This also removes any conflicts involving the removed claim.
    pub fn remove_claim(&mut self, claim_id: &str) {
        if let Some(claim) = self.claims.remove(claim_id) {
            // Remove from normalized hashes index
            self.normalized_hashes.remove(claim_id);

            // Remove from subject index
            if let Some(subject) = Self::extract_subject(&claim.structure)
                && let Some(ids) = self.subject_index.get_mut(&subject)
            {
                ids.remove(claim_id);
                if ids.is_empty() {
                    self.subject_index.remove(&subject);
                }
            }

            // Remove conflicts involving this claim
            self.known_conflicts
                .retain(|c| c.claim1_id != claim_id && c.claim2_id != claim_id);

            // Remove checked pairs involving this claim
            self.checked_pairs
                .retain(|(id1, id2)| id1 != claim_id && id2 != claim_id);
        }
    }

    /// Suggest resolutions for a conflict.
    #[must_use]
    pub fn suggest_resolution(&self, conflict: &ClaimConflict) -> Vec<Resolution> {
        let mut resolutions = Vec::new();

        // Get the claims involved
        let claim1 = self.claims.get(&conflict.claim1_id);
        let claim2 = self.claims.get(&conflict.claim2_id);

        match (claim1, claim2) {
            (Some(c1), Some(c2)) => {
                // Resolution 1: Prioritize by confidence
                if (c1.confidence - c2.confidence).abs() > 0.1 {
                    if c1.confidence > c2.confidence {
                        resolutions.push(Resolution::PrioritizeByCofidence {
                            keep_claim_id: c1.id.clone(),
                            remove_claim_id: c2.id.clone(),
                        });
                    } else {
                        resolutions.push(Resolution::PrioritizeByCofidence {
                            keep_claim_id: c2.id.clone(),
                            remove_claim_id: c1.id.clone(),
                        });
                    }
                }

                // Resolution 2: Remove the less specific claim
                let c1_specificity = Self::estimate_specificity(&c1.structure);
                let c2_specificity = Self::estimate_specificity(&c2.structure);

                if c1_specificity != c2_specificity {
                    let (remove_id, reason) = if c1_specificity < c2_specificity {
                        (&c1.id, format!("Less specific than claim '{}'", c2.text))
                    } else {
                        (&c2.id, format!("Less specific than claim '{}'", c1.text))
                    };

                    resolutions.push(Resolution::RemoveClaim {
                        claim_id: remove_id.clone(),
                        reason,
                    });
                }

                // Resolution 3: Add exception for temporal/conditional conflicts
                if matches!(
                    conflict.conflict_type,
                    ConflictType::TemporalConflict | ConflictType::PredicateConflict
                ) {
                    resolutions.push(Resolution::AddException {
                        exception: format!(
                            "Under different circumstances: {} AND {}",
                            c1.text, c2.text
                        ),
                        explanation: "Both claims may be true in different contexts".to_string(),
                    });
                }

                // Resolution 4: Suggest modification for predicate conflicts
                if conflict.conflict_type == ConflictType::PredicateConflict {
                    resolutions.push(Resolution::ModifyClaim {
                        claim_id: c2.id.clone(),
                        suggestion: format!(
                            "Consider qualifying the claim: 'Sometimes {}' or 'Under certain conditions {}'",
                            c2.text, c2.text
                        ),
                    });
                }
            }
            _ => {
                // One or both claims not found - suggest removal
                resolutions.push(Resolution::RemoveClaim {
                    claim_id: conflict.claim1_id.clone(),
                    reason: "Claim involved in unresolvable conflict".to_string(),
                });
            }
        }

        resolutions
    }

    /// Get the number of claims in the knowledge base.
    #[must_use]
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    /// Get a claim by ID.
    #[must_use]
    pub fn get_claim(&self, claim_id: &str) -> Option<&LogicalClaim> {
        self.claims.get(claim_id)
    }

    /// Clear all claims and reset the checker.
    pub fn clear(&mut self) {
        self.claims.clear();
        self.normalized_hashes.clear();
        self.subject_index.clear();
        self.known_conflicts.clear();
        self.checked_pairs.clear();
    }

    /// Index a claim for efficient lookup.
    fn index_claim(&mut self, claim: &LogicalClaim) {
        // Store normalized hash
        let hash = Self::hash_structure(&claim.structure);
        self.normalized_hashes.insert(claim.id.clone(), hash);

        // Index by subject if applicable
        if let Some(subject) = Self::extract_subject(&claim.structure) {
            self.subject_index
                .entry(subject)
                .or_default()
                .insert(claim.id.clone());
        }
    }

    /// Find conflicts between a new claim and existing claims.
    fn find_conflicts_with_new_claim(&self, new_claim: &LogicalClaim) -> Vec<ClaimConflict> {
        let mut conflicts = Vec::new();
        let new_hash = Self::hash_structure(&new_claim.structure);

        // Check against all existing claims
        for (existing_id, existing_claim) in &self.claims {
            // Skip if already checked
            let pair = Self::make_pair(&new_claim.id, existing_id);
            if self.checked_pairs.contains(&pair) {
                continue;
            }

            // Quick check: if hashes are identical, claims are equivalent (not conflicting)
            if let Some(&existing_hash) = self.normalized_hashes.get(existing_id)
                && new_hash == existing_hash
            {
                continue;
            }

            // Detailed conflict check
            if let Some(conflict) = self.check_claim_pair(new_claim, existing_claim) {
                conflicts.push(conflict);
            }
        }

        conflicts
    }

    /// Check if two claims conflict.
    fn check_claim_pair(
        &self,
        claim1: &LogicalClaim,
        claim2: &LogicalClaim,
    ) -> Option<ClaimConflict> {
        // Check for direct contradiction (A and NOT A)
        if let Some(conflict) = self.check_direct_contradiction(claim1, claim2) {
            return Some(conflict);
        }

        // Check for predicate conflicts
        if let Some(conflict) = self.check_predicate_conflict(claim1, claim2) {
            return Some(conflict);
        }

        // Check for comparison conflicts
        if let Some(conflict) = self.check_comparison_conflict(claim1, claim2) {
            return Some(conflict);
        }

        // Check for temporal conflicts
        if let Some(conflict) = self.check_temporal_conflict(claim1, claim2) {
            return Some(conflict);
        }

        // Check for modal conflicts
        if let Some(conflict) = self.check_modal_conflict(claim1, claim2) {
            return Some(conflict);
        }

        // Check for causal conflicts
        if let Some(conflict) = self.check_causal_conflict(claim1, claim2) {
            return Some(conflict);
        }

        None
    }

    /// Check for direct contradiction (A and NOT A).
    #[allow(clippy::unused_self)]
    fn check_direct_contradiction(
        &self,
        claim1: &LogicalClaim,
        claim2: &LogicalClaim,
    ) -> Option<ClaimConflict> {
        // Check if one is the negation of the other
        if let ClaimStructure::Not(inner) = &claim1.structure {
            let inner_hash = Self::hash_structure(inner);
            let claim2_hash = Self::hash_structure(&claim2.structure);
            if inner_hash == claim2_hash {
                return Some(ClaimConflict::new(
                    &claim1.id,
                    &claim2.id,
                    ConflictType::DirectContradiction,
                    format!("'{}' directly contradicts '{}'", claim1.text, claim2.text),
                ));
            }
        }

        if let ClaimStructure::Not(inner) = &claim2.structure {
            let inner_hash = Self::hash_structure(inner);
            let claim1_hash = Self::hash_structure(&claim1.structure);
            if inner_hash == claim1_hash {
                return Some(ClaimConflict::new(
                    &claim1.id,
                    &claim2.id,
                    ConflictType::DirectContradiction,
                    format!("'{}' directly contradicts '{}'", claim2.text, claim1.text),
                ));
            }
        }

        None
    }

    /// Check for predicate conflicts (same subject, contradictory predicates).
    #[allow(clippy::unused_self)]
    fn check_predicate_conflict(
        &self,
        claim1: &LogicalClaim,
        claim2: &LogicalClaim,
    ) -> Option<ClaimConflict> {
        match (&claim1.structure, &claim2.structure) {
            (
                ClaimStructure::Predicate {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                ClaimStructure::Predicate {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                // Same subject with contradictory predicates
                if Self::normalize_string(s1) == Self::normalize_string(s2) {
                    // Check for opposite predicates
                    if Self::are_opposite_predicates(p1, p2)
                        || (Self::normalize_string(p1) == Self::normalize_string(p2)
                            && Self::are_contradictory_objects(o1.as_deref(), o2.as_deref()))
                    {
                        return Some(ClaimConflict::new(
                            &claim1.id,
                            &claim2.id,
                            ConflictType::PredicateConflict,
                            format!(
                                "Subject '{s1}' has contradictory properties: '{p1}' vs '{p2}'"
                            ),
                        ));
                    }
                }
            }
            (ClaimStructure::Predicate { .. }, ClaimStructure::Not(inner))
            | (ClaimStructure::Not(inner), ClaimStructure::Predicate { .. }) => {
                // Check if the Not wraps a contradictory predicate
                if let ClaimStructure::Predicate {
                    subject: ns,
                    predicate: np,
                    ..
                } = inner.as_ref()
                {
                    let (pred_claim, other_claim) =
                        if matches!(&claim1.structure, ClaimStructure::Not(_)) {
                            (claim2, claim1)
                        } else {
                            (claim1, claim2)
                        };

                    if let ClaimStructure::Predicate {
                        subject: ps,
                        predicate: pp,
                        ..
                    } = &pred_claim.structure
                        && Self::normalize_string(ns) == Self::normalize_string(ps)
                        && Self::normalize_string(np) == Self::normalize_string(pp)
                    {
                        return Some(ClaimConflict::new(
                            &claim1.id,
                            &claim2.id,
                            ConflictType::DirectContradiction,
                            format!("'{}' is negated by '{}'", pred_claim.text, other_claim.text),
                        ));
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Check for comparison conflicts.
    #[allow(clippy::unused_self)]
    fn check_comparison_conflict(
        &self,
        claim1: &LogicalClaim,
        claim2: &LogicalClaim,
    ) -> Option<ClaimConflict> {
        if let (
            ClaimStructure::Comparison {
                left: l1,
                operator: op1,
                right: r1,
            },
            ClaimStructure::Comparison {
                left: l2,
                operator: op2,
                right: r2,
            },
        ) = (&claim1.structure, &claim2.structure)
        {
            // Same operands but contradictory operators
            let l1_norm = Self::normalize_string(l1);
            let r1_norm = Self::normalize_string(r1);
            let l2_norm = Self::normalize_string(l2);
            let r2_norm = Self::normalize_string(r2);

            // A > B and A < B
            if l1_norm == l2_norm
                && r1_norm == r2_norm
                && Self::are_opposite_comparisons(*op1, *op2)
            {
                return Some(ClaimConflict::new(
                    &claim1.id,
                    &claim2.id,
                    ConflictType::ComparisonConflict,
                    format!(
                        "Contradictory comparisons: '{}' vs '{}'",
                        claim1.text, claim2.text
                    ),
                ));
            }

            // A > B and B > A (transitive violation)
            if l1_norm == r2_norm && r1_norm == l2_norm {
                use crate::types::ComparisonOp::{GreaterThan, LessThan};
                if matches!(
                    (op1, op2),
                    (GreaterThan, GreaterThan) | (LessThan, LessThan)
                ) {
                    return Some(ClaimConflict::new(
                        &claim1.id,
                        &claim2.id,
                        ConflictType::ComparisonConflict,
                        format!(
                            "Circular comparison: '{}' and '{}'",
                            claim1.text, claim2.text
                        ),
                    ));
                }
            }
        }

        None
    }

    /// Check for temporal conflicts.
    #[allow(clippy::unused_self)]
    fn check_temporal_conflict(
        &self,
        claim1: &LogicalClaim,
        claim2: &LogicalClaim,
    ) -> Option<ClaimConflict> {
        use crate::types::TimeRelation;

        if let (
            ClaimStructure::Temporal {
                event: e1,
                time_relation: tr1,
                reference: r1,
            },
            ClaimStructure::Temporal {
                event: e2,
                time_relation: tr2,
                reference: r2,
            },
        ) = (&claim1.structure, &claim2.structure)
        {
            let e1_norm = Self::normalize_string(e1);
            let r1_norm = Self::normalize_string(r1);
            let e2_norm = Self::normalize_string(e2);
            let r2_norm = Self::normalize_string(r2);

            // A before B and B before A
            if e1_norm == r2_norm
                && r1_norm == e2_norm
                && matches!(
                    (tr1, tr2),
                    (TimeRelation::Before, TimeRelation::Before)
                        | (TimeRelation::After, TimeRelation::After)
                )
            {
                return Some(ClaimConflict::new(
                    &claim1.id,
                    &claim2.id,
                    ConflictType::TemporalConflict,
                    format!(
                        "Circular temporal relationship: '{}' and '{}'",
                        claim1.text, claim2.text
                    ),
                ));
            }

            // Same events but contradictory relations
            if e1_norm == e2_norm
                && r1_norm == r2_norm
                && Self::are_opposite_time_relations(tr1, tr2)
            {
                return Some(ClaimConflict::new(
                    &claim1.id,
                    &claim2.id,
                    ConflictType::TemporalConflict,
                    format!(
                        "Contradictory temporal relations: '{}' vs '{}'",
                        claim1.text, claim2.text
                    ),
                ));
            }
        }

        None
    }

    /// Check for modal conflicts.
    #[allow(clippy::unused_self)]
    fn check_modal_conflict(
        &self,
        claim1: &LogicalClaim,
        claim2: &LogicalClaim,
    ) -> Option<ClaimConflict> {
        use crate::types::Modality;

        if let (
            ClaimStructure::Modal {
                claim: c1,
                modality: m1,
            },
            ClaimStructure::Modal {
                claim: c2,
                modality: m2,
            },
        ) = (&claim1.structure, &claim2.structure)
        {
            let c1_hash = Self::hash_structure(c1);
            let c2_hash = Self::hash_structure(c2);

            // Same claim with contradictory modalities
            if c1_hash == c2_hash
                && matches!(
                    (m1, m2),
                    (Modality::Necessary, Modality::Unlikely)
                        | (Modality::Unlikely, Modality::Necessary)
                )
            {
                return Some(ClaimConflict::new(
                    &claim1.id,
                    &claim2.id,
                    ConflictType::ModalConflict,
                    format!(
                        "Modal conflict: '{}' cannot be both {} and {}",
                        claim1.text,
                        m1.to_smtlib(),
                        m2.to_smtlib()
                    ),
                ));
            }
        }

        None
    }

    /// Check for causal conflicts.
    #[allow(clippy::unused_self)]
    fn check_causal_conflict(
        &self,
        claim1: &LogicalClaim,
        claim2: &LogicalClaim,
    ) -> Option<ClaimConflict> {
        if let (
            ClaimStructure::Causal {
                cause: c1,
                effect: e1,
                ..
            },
            ClaimStructure::Causal {
                cause: c2,
                effect: e2,
                ..
            },
        ) = (&claim1.structure, &claim2.structure)
        {
            let c1_hash = Self::hash_structure(c1);
            let e1_hash = Self::hash_structure(e1);
            let c2_hash = Self::hash_structure(c2);
            let e2_hash = Self::hash_structure(e2);

            // Circular causation: A causes B and B causes A
            if c1_hash == e2_hash && e1_hash == c2_hash {
                return Some(ClaimConflict::new(
                    &claim1.id,
                    &claim2.id,
                    ConflictType::CausalConflict,
                    format!(
                        "Circular causation: '{}' and '{}'",
                        claim1.text, claim2.text
                    ),
                ));
            }
        }

        None
    }

    /// Hash a claim structure for comparison.
    fn hash_structure(structure: &ClaimStructure) -> u64 {
        let mut hasher = DefaultHasher::new();
        Self::hash_structure_recursive(structure, &mut hasher);
        hasher.finish()
    }

    /// Recursively hash a claim structure.
    fn hash_structure_recursive<H: Hasher>(structure: &ClaimStructure, hasher: &mut H) {
        match structure {
            ClaimStructure::Predicate {
                subject,
                predicate,
                object,
            } => {
                "predicate".hash(hasher);
                Self::normalize_string(subject).hash(hasher);
                Self::normalize_string(predicate).hash(hasher);
                object
                    .as_ref()
                    .map(|o| Self::normalize_string(o))
                    .hash(hasher);
            }
            ClaimStructure::Comparison {
                left,
                operator,
                right,
            } => {
                "comparison".hash(hasher);
                Self::normalize_string(left).hash(hasher);
                format!("{operator:?}").hash(hasher);
                Self::normalize_string(right).hash(hasher);
            }
            ClaimStructure::And(claims) => {
                "and".hash(hasher);
                for claim in claims {
                    Self::hash_structure_recursive(claim, hasher);
                }
            }
            ClaimStructure::Or(claims) => {
                "or".hash(hasher);
                for claim in claims {
                    Self::hash_structure_recursive(claim, hasher);
                }
            }
            ClaimStructure::Not(inner) => {
                "not".hash(hasher);
                Self::hash_structure_recursive(inner, hasher);
            }
            ClaimStructure::Implies {
                premise,
                conclusion,
            } => {
                "implies".hash(hasher);
                Self::hash_structure_recursive(premise, hasher);
                Self::hash_structure_recursive(conclusion, hasher);
            }
            ClaimStructure::Quantified {
                quantifier,
                variable,
                domain,
                body,
            } => {
                "quantified".hash(hasher);
                format!("{quantifier:?}").hash(hasher);
                Self::normalize_string(variable).hash(hasher);
                Self::normalize_string(domain).hash(hasher);
                Self::hash_structure_recursive(body, hasher);
            }
            ClaimStructure::Temporal {
                event,
                time_relation,
                reference,
            } => {
                "temporal".hash(hasher);
                Self::normalize_string(event).hash(hasher);
                format!("{time_relation:?}").hash(hasher);
                Self::normalize_string(reference).hash(hasher);
            }
            ClaimStructure::Causal {
                cause,
                effect,
                strength,
            } => {
                "causal".hash(hasher);
                Self::hash_structure_recursive(cause, hasher);
                Self::hash_structure_recursive(effect, hasher);
                format!("{strength:?}").hash(hasher);
            }
            ClaimStructure::Modal { claim, modality } => {
                "modal".hash(hasher);
                Self::hash_structure_recursive(claim, hasher);
                format!("{modality:?}").hash(hasher);
            }
            ClaimStructure::Raw(text) => {
                "raw".hash(hasher);
                Self::normalize_string(text).hash(hasher);
            }
        }
    }

    /// Extract the subject from a claim structure if applicable.
    fn extract_subject(structure: &ClaimStructure) -> Option<String> {
        match structure {
            ClaimStructure::Predicate { subject, .. } => Some(Self::normalize_string(subject)),
            ClaimStructure::Comparison { left, .. } => Some(Self::normalize_string(left)),
            ClaimStructure::Temporal { event, .. } => Some(Self::normalize_string(event)),
            ClaimStructure::Not(inner) => Self::extract_subject(inner),
            ClaimStructure::Modal { claim, .. } => Self::extract_subject(claim),
            _ => None,
        }
    }

    /// Normalize a string for comparison.
    fn normalize_string(s: &str) -> String {
        s.trim().to_lowercase()
    }

    /// Create a canonical pair from two IDs.
    fn make_pair(id1: &str, id2: &str) -> (String, String) {
        if id1 < id2 {
            (id1.to_string(), id2.to_string())
        } else {
            (id2.to_string(), id1.to_string())
        }
    }

    /// Check if two predicates are opposites.
    fn are_opposite_predicates(p1: &str, p2: &str) -> bool {
        let p1_norm = Self::normalize_string(p1);
        let p2_norm = Self::normalize_string(p2);

        let opposites = [
            ("is", "is not"),
            ("are", "are not"),
            ("has", "has not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("true", "false"),
            ("yes", "no"),
            ("alive", "dead"),
            ("open", "closed"),
            ("on", "off"),
            ("hot", "cold"),
            ("big", "small"),
            ("fast", "slow"),
        ];

        for (a, b) in opposites {
            if (p1_norm == a && p2_norm == b) || (p1_norm == b && p2_norm == a) {
                return true;
            }
        }

        // Check for "not" prefix
        if p1_norm.starts_with("not ") && p1_norm[4..] == p2_norm {
            return true;
        }
        if p2_norm.starts_with("not ") && p2_norm[4..] == p1_norm {
            return true;
        }

        false
    }

    /// Check if two objects are contradictory (for same subject and predicate).
    fn are_contradictory_objects(o1: Option<&str>, o2: Option<&str>) -> bool {
        match (o1, o2) {
            (Some(obj1), Some(obj2)) => {
                let o1_norm = Self::normalize_string(obj1);
                let o2_norm = Self::normalize_string(obj2);
                // Different non-empty objects for the same predicate are contradictory
                !o1_norm.is_empty() && !o2_norm.is_empty() && o1_norm != o2_norm
            }
            _ => false,
        }
    }

    /// Check if two comparison operators are opposites.
    fn are_opposite_comparisons(
        op1: crate::types::ComparisonOp,
        op2: crate::types::ComparisonOp,
    ) -> bool {
        use crate::types::ComparisonOp::{
            Equal, GreaterOrEqual, GreaterThan, LessOrEqual, LessThan, NotEqual,
        };
        matches!(
            (op1, op2),
            (Equal, NotEqual)
                | (NotEqual, Equal)
                | (LessThan, GreaterOrEqual | GreaterThan)
                | (GreaterOrEqual | GreaterThan, LessThan)
                | (GreaterThan, LessOrEqual)
                | (LessOrEqual, GreaterThan)
        )
    }

    /// Check if two time relations are opposite.
    fn are_opposite_time_relations(
        tr1: &crate::types::TimeRelation,
        tr2: &crate::types::TimeRelation,
    ) -> bool {
        use crate::types::TimeRelation::{After, Before};
        matches!((tr1, tr2), (Before, After) | (After, Before))
    }

    /// Estimate the specificity of a claim structure (higher = more specific).
    fn estimate_specificity(structure: &ClaimStructure) -> u32 {
        match structure {
            ClaimStructure::Raw(_) => 1,
            ClaimStructure::Predicate { object: None, .. } => 2,
            ClaimStructure::Predicate {
                object: Some(_), ..
            }
            | ClaimStructure::Comparison { .. }
            | ClaimStructure::Temporal { .. } => 3,
            ClaimStructure::Not(inner) => Self::estimate_specificity(inner) + 1,
            ClaimStructure::Modal { claim, .. } => Self::estimate_specificity(claim) + 1,
            ClaimStructure::Implies {
                premise,
                conclusion,
            } => Self::estimate_specificity(premise) + Self::estimate_specificity(conclusion),
            ClaimStructure::And(claims) | ClaimStructure::Or(claims) => {
                claims.iter().map(Self::estimate_specificity).sum::<u32>() + 1
            }
            ClaimStructure::Quantified { body, .. } => Self::estimate_specificity(body) + 2,
            ClaimStructure::Causal { cause, effect, .. } => {
                Self::estimate_specificity(cause) + Self::estimate_specificity(effect) + 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CausalStrength, ComparisonOp, Modality, TimeRelation};

    fn make_predicate_claim(
        id: &str,
        subject: &str,
        predicate: &str,
        object: Option<&str>,
    ) -> LogicalClaim {
        let mut claim = LogicalClaim::new(
            format!("{subject} {predicate} {}", object.unwrap_or("")),
            ClaimStructure::Predicate {
                subject: subject.to_string(),
                predicate: predicate.to_string(),
                object: object.map(ToString::to_string),
            },
        );
        claim.id = id.to_string();
        claim
    }

    fn make_comparison_claim(id: &str, left: &str, op: ComparisonOp, right: &str) -> LogicalClaim {
        let mut claim = LogicalClaim::new(
            format!("{left} {} {right}", op.to_smtlib()),
            ClaimStructure::Comparison {
                left: left.to_string(),
                operator: op,
                right: right.to_string(),
            },
        );
        claim.id = id.to_string();
        claim
    }

    fn make_temporal_claim(
        id: &str,
        event: &str,
        relation: TimeRelation,
        reference: &str,
    ) -> LogicalClaim {
        let mut claim = LogicalClaim::new(
            format!("{event} {} {reference}", relation.to_smtlib()),
            ClaimStructure::Temporal {
                event: event.to_string(),
                time_relation: relation,
                reference: reference.to_string(),
            },
        );
        claim.id = id.to_string();
        claim
    }

    #[test]
    fn test_add_consistent_claims() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));
        let claim2 = make_predicate_claim("c2", "grass", "is", Some("green"));

        let result1 = checker.add_claim(claim1);
        assert!(result1.is_consistent());

        let result2 = checker.add_claim(claim2);
        assert!(result2.is_consistent());

        assert_eq!(checker.claim_count(), 2);
        assert!(checker.get_conflicts().is_empty());
    }

    #[test]
    fn test_detect_direct_contradiction() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));

        let negated_structure = ClaimStructure::Not(Box::new(ClaimStructure::Predicate {
            subject: "sky".to_string(),
            predicate: "is".to_string(),
            object: Some("blue".to_string()),
        }));
        let mut claim2 = LogicalClaim::new("sky is not blue", negated_structure);
        claim2.id = "c2".to_string();

        checker.add_claim(claim1);
        let result = checker.add_claim(claim2);

        assert!(result.is_inconsistent());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(
            conflicts[0].conflict_type,
            ConflictType::DirectContradiction
        );
    }

    #[test]
    fn test_detect_predicate_conflict() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_predicate_claim("c1", "cat", "is", Some("alive"));
        let claim2 = make_predicate_claim("c2", "cat", "is", Some("dead"));

        checker.add_claim(claim1);
        let result = checker.add_claim(claim2);

        assert!(result.is_inconsistent());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::PredicateConflict);
    }

    #[test]
    fn test_detect_comparison_conflict() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_comparison_claim("c1", "A", ComparisonOp::GreaterThan, "B");
        let claim2 = make_comparison_claim("c2", "A", ComparisonOp::LessThan, "B");

        checker.add_claim(claim1);
        let result = checker.add_claim(claim2);

        assert!(result.is_inconsistent());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::ComparisonConflict);
    }

    #[test]
    fn test_detect_circular_comparison() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_comparison_claim("c1", "A", ComparisonOp::GreaterThan, "B");
        let claim2 = make_comparison_claim("c2", "B", ComparisonOp::GreaterThan, "A");

        checker.add_claim(claim1);
        let result = checker.add_claim(claim2);

        assert!(result.is_inconsistent());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::ComparisonConflict);
    }

    #[test]
    fn test_detect_temporal_conflict() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_temporal_claim("c1", "event_a", TimeRelation::Before, "event_b");
        let claim2 = make_temporal_claim("c2", "event_b", TimeRelation::Before, "event_a");

        checker.add_claim(claim1);
        let result = checker.add_claim(claim2);

        assert!(result.is_inconsistent());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::TemporalConflict);
    }

    #[test]
    fn test_detect_opposite_temporal_relations() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_temporal_claim("c1", "meeting", TimeRelation::Before, "lunch");
        let claim2 = make_temporal_claim("c2", "meeting", TimeRelation::After, "lunch");

        checker.add_claim(claim1);
        let result = checker.add_claim(claim2);

        assert!(result.is_inconsistent());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::TemporalConflict);
    }

    #[test]
    fn test_detect_modal_conflict() {
        let mut checker = IncrementalConsistencyChecker::new();

        let inner = ClaimStructure::Raw("it will rain".to_string());

        let mut claim1 = LogicalClaim::new(
            "it must rain",
            ClaimStructure::Modal {
                claim: Box::new(inner.clone()),
                modality: Modality::Necessary,
            },
        );
        claim1.id = "c1".to_string();

        let mut claim2 = LogicalClaim::new(
            "it is unlikely to rain",
            ClaimStructure::Modal {
                claim: Box::new(inner),
                modality: Modality::Unlikely,
            },
        );
        claim2.id = "c2".to_string();

        checker.add_claim(claim1);
        let result = checker.add_claim(claim2);

        assert!(result.is_inconsistent());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::ModalConflict);
    }

    #[test]
    fn test_detect_causal_conflict() {
        let mut checker = IncrementalConsistencyChecker::new();

        let cause_a = ClaimStructure::Raw("event_a".to_string());
        let effect_b = ClaimStructure::Raw("event_b".to_string());

        let mut claim1 = LogicalClaim::new(
            "A causes B",
            ClaimStructure::Causal {
                cause: Box::new(cause_a.clone()),
                effect: Box::new(effect_b.clone()),
                strength: CausalStrength::Direct,
            },
        );
        claim1.id = "c1".to_string();

        let mut claim2 = LogicalClaim::new(
            "B causes A",
            ClaimStructure::Causal {
                cause: Box::new(effect_b),
                effect: Box::new(cause_a),
                strength: CausalStrength::Direct,
            },
        );
        claim2.id = "c2".to_string();

        checker.add_claim(claim1);
        let result = checker.add_claim(claim2);

        assert!(result.is_inconsistent());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::CausalConflict);
    }

    #[test]
    fn test_check_new_claim_without_adding() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));
        checker.add_claim(claim1);

        // Check a contradicting claim without adding it
        let claim2 = make_predicate_claim("c2", "sky", "is", Some("green"));
        let result = checker.check_new_claim(&claim2);

        assert!(result.is_inconsistent());

        // Verify it wasn't added
        assert_eq!(checker.claim_count(), 1);
        assert!(checker.get_claim("c2").is_none());
    }

    #[test]
    fn test_remove_claim() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));
        let claim2 = make_predicate_claim("c2", "sky", "is", Some("green"));

        checker.add_claim(claim1);
        checker.add_claim(claim2);

        assert!(!checker.get_conflicts().is_empty());

        // Remove one of the conflicting claims
        checker.remove_claim("c2");

        assert_eq!(checker.claim_count(), 1);
        assert!(checker.get_conflicts().is_empty());
        assert!(checker.check_consistency().is_consistent());
    }

    #[test]
    fn test_suggest_resolution() {
        let mut checker = IncrementalConsistencyChecker::new();

        let mut claim1 = make_predicate_claim("c1", "sky", "is", Some("blue"));
        claim1.confidence = 0.9;

        let mut claim2 = make_predicate_claim("c2", "sky", "is", Some("green"));
        claim2.confidence = 0.5;

        checker.add_claim(claim1);
        checker.add_claim(claim2);

        let conflicts = checker.get_conflicts();
        assert_eq!(conflicts.len(), 1);

        let resolutions = checker.suggest_resolution(&conflicts[0]);
        assert!(!resolutions.is_empty());

        // Should suggest prioritizing by confidence
        let has_priority_resolution = resolutions.iter().any(|r| {
            matches!(r, Resolution::PrioritizeByCofidence { keep_claim_id, remove_claim_id }
                if keep_claim_id == "c1" && remove_claim_id == "c2")
        });
        assert!(has_priority_resolution);
    }

    #[test]
    fn test_clear() {
        let mut checker = IncrementalConsistencyChecker::new();

        checker.add_claim(make_predicate_claim("c1", "sky", "is", Some("blue")));
        checker.add_claim(make_predicate_claim("c2", "grass", "is", Some("green")));

        assert_eq!(checker.claim_count(), 2);

        checker.clear();

        assert_eq!(checker.claim_count(), 0);
        assert!(checker.get_conflicts().is_empty());
    }

    #[test]
    fn test_consistency_result_methods() {
        let consistent = ConsistencyResult::Consistent;
        assert!(consistent.is_consistent());
        assert!(!consistent.is_inconsistent());
        assert!(consistent.conflicts().is_empty());

        let conflict = ClaimConflict::new("c1", "c2", ConflictType::DirectContradiction, "test");
        let inconsistent = ConsistencyResult::Inconsistent(vec![conflict.clone()]);
        assert!(!inconsistent.is_consistent());
        assert!(inconsistent.is_inconsistent());
        assert_eq!(inconsistent.conflicts().len(), 1);

        let unknown = ConsistencyResult::Unknown;
        assert!(!unknown.is_consistent());
        assert!(!unknown.is_inconsistent());
        assert!(unknown.conflicts().is_empty());
    }

    #[test]
    fn test_conflict_severity() {
        let conflict = ClaimConflict::new("c1", "c2", ConflictType::DirectContradiction, "test");
        assert!((conflict.severity - 1.0).abs() < f32::EPSILON);

        let conflict_with_severity = conflict.with_severity(0.5);
        assert!((conflict_with_severity.severity - 0.5).abs() < f32::EPSILON);

        // Test clamping
        let clamped = ClaimConflict::new("c1", "c2", ConflictType::SemanticContradiction, "test")
            .with_severity(1.5);
        assert!((clamped.severity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_conflict_type_severity() {
        assert!((ConflictType::DirectContradiction.default_severity() - 1.0).abs() < f32::EPSILON);
        assert!(
            ConflictType::SemanticContradiction.default_severity()
                > ConflictType::ModalConflict.default_severity()
        );
    }

    #[test]
    fn test_duplicate_claim_handling() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim = make_predicate_claim("c1", "sky", "is", Some("blue"));
        checker.add_claim(claim.clone());

        // Adding same claim again should not create duplicates
        let result = checker.add_claim(claim);
        assert!(result.is_consistent());
        assert_eq!(checker.claim_count(), 1);
    }

    #[test]
    fn test_opposite_predicates() {
        assert!(IncrementalConsistencyChecker::are_opposite_predicates(
            "is", "is not"
        ));
        assert!(IncrementalConsistencyChecker::are_opposite_predicates(
            "true", "false"
        ));
        assert!(IncrementalConsistencyChecker::are_opposite_predicates(
            "alive", "dead"
        ));
        assert!(!IncrementalConsistencyChecker::are_opposite_predicates(
            "is", "has"
        ));
    }

    #[test]
    fn test_get_claim() {
        let mut checker = IncrementalConsistencyChecker::new();

        let claim = make_predicate_claim("c1", "sky", "is", Some("blue"));
        checker.add_claim(claim);

        assert!(checker.get_claim("c1").is_some());
        assert!(checker.get_claim("nonexistent").is_none());
    }
}
