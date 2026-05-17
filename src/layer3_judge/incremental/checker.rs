//! `IncrementalConsistencyChecker` struct and its full implementation.

use std::collections::{HashMap, HashSet};

use crate::types::LogicalClaim;

use super::conflict::{
    are_opposite_predicates, check_causal_conflict, check_comparison_conflict,
    check_direct_contradiction, check_modal_conflict, check_predicate_conflict,
    check_temporal_conflict, estimate_specificity, extract_subject, hash_structure,
    normalize_string,
};
use super::types::{ClaimConflict, ConsistencyResult, Resolution};

/// Incremental consistency checker for logical claims.
///
/// This checker maintains a knowledge base of claims and efficiently checks
/// consistency when new claims are added. It uses incremental checking to
/// avoid re-checking all pairs of claims on each addition.
pub struct IncrementalConsistencyChecker {
    /// The knowledge base of claims, keyed by ID.
    pub(super) claims: HashMap<String, LogicalClaim>,
    /// Index of normalized claim hashes for quick lookup.
    pub(super) normalized_hashes: HashMap<String, u64>,
    /// Index mapping subjects to claim IDs for predicate claims.
    pub(super) subject_index: HashMap<String, HashSet<String>>,
    /// Known conflicts between claims.
    pub(super) known_conflicts: Vec<ClaimConflict>,
    /// Set of claim ID pairs that have been checked.
    pub(super) checked_pairs: HashSet<(String, String)>,
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
            if let Some(subject) = extract_subject(&claim.structure)
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
        use super::types::ConflictType;

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
                let c1_specificity = estimate_specificity(&c1.structure);
                let c2_specificity = estimate_specificity(&c2.structure);

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

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Index a claim for efficient lookup.
    fn index_claim(&mut self, claim: &LogicalClaim) {
        // Store normalized hash
        let hash = hash_structure(&claim.structure);
        self.normalized_hashes.insert(claim.id.clone(), hash);

        // Index by subject if applicable
        if let Some(subject) = extract_subject(&claim.structure) {
            self.subject_index
                .entry(subject)
                .or_default()
                .insert(claim.id.clone());
        }
    }

    /// Find conflicts between a new claim and existing claims.
    pub(super) fn find_conflicts_with_new_claim(
        &self,
        new_claim: &LogicalClaim,
    ) -> Vec<ClaimConflict> {
        let mut conflicts = Vec::new();
        let new_hash = hash_structure(&new_claim.structure);

        // Check against all existing claims
        for (existing_id, existing_claim) in &self.claims {
            // Skip if already checked
            let pair = make_pair(&new_claim.id, existing_id);
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
            if let Some(conflict) = check_claim_pair(new_claim, existing_claim) {
                conflicts.push(conflict);
            }
        }

        conflicts
    }

    // -----------------------------------------------------------------------
    // Exposed for testing
    // -----------------------------------------------------------------------

    /// Check if two predicates are opposites (delegates to conflict module).
    #[must_use]
    pub fn are_opposite_predicates(p1: &str, p2: &str) -> bool {
        are_opposite_predicates(p1, p2)
    }

    /// Normalize a string for comparison (delegates to conflict module).
    #[must_use]
    pub fn normalize_string(s: &str) -> String {
        normalize_string(s)
    }
}

// ---------------------------------------------------------------------------
// Pair dispatch helper
// ---------------------------------------------------------------------------

/// Create a canonical ordered pair from two IDs.
#[must_use]
pub fn make_pair(id1: &str, id2: &str) -> (String, String) {
    if id1 < id2 {
        (id1.to_string(), id2.to_string())
    } else {
        (id2.to_string(), id1.to_string())
    }
}

/// Dispatch all conflict checks for a pair of claims.
fn check_claim_pair(claim1: &LogicalClaim, claim2: &LogicalClaim) -> Option<ClaimConflict> {
    // Check for direct contradiction (A and NOT A)
    if let Some(conflict) = check_direct_contradiction(claim1, claim2) {
        return Some(conflict);
    }

    // Check for predicate conflicts
    if let Some(conflict) = check_predicate_conflict(claim1, claim2) {
        return Some(conflict);
    }

    // Check for comparison conflicts
    if let Some(conflict) = check_comparison_conflict(claim1, claim2) {
        return Some(conflict);
    }

    // Check for temporal conflicts
    if let Some(conflict) = check_temporal_conflict(claim1, claim2) {
        return Some(conflict);
    }

    // Check for modal conflicts
    if let Some(conflict) = check_modal_conflict(claim1, claim2) {
        return Some(conflict);
    }

    // Check for causal conflicts
    if let Some(conflict) = check_causal_conflict(claim1, claim2) {
        return Some(conflict);
    }

    None
}
