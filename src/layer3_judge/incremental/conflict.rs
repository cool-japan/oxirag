//! Conflict detection algorithms for incremental consistency checking.
//!
//! Provides a family of `check_*` functions that examine pairs of
//! [`LogicalClaim`] values for specific categories of logical conflict.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::types::{ClaimStructure, LogicalClaim};

use super::types::{ClaimConflict, ConflictType};

// ---------------------------------------------------------------------------
// Hash helpers (also used by checker.rs)
// ---------------------------------------------------------------------------

/// Hash a claim structure for comparison.
#[must_use]
pub fn hash_structure(structure: &ClaimStructure) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_structure_recursive(structure, &mut hasher);
    hasher.finish()
}

/// Recursively hash a claim structure.
pub fn hash_structure_recursive<H: Hasher>(structure: &ClaimStructure, hasher: &mut H) {
    match structure {
        ClaimStructure::Predicate {
            subject,
            predicate,
            object,
        } => {
            "predicate".hash(hasher);
            normalize_string(subject).hash(hasher);
            normalize_string(predicate).hash(hasher);
            object.as_ref().map(|o| normalize_string(o)).hash(hasher);
        }
        ClaimStructure::Comparison {
            left,
            operator,
            right,
        } => {
            "comparison".hash(hasher);
            normalize_string(left).hash(hasher);
            format!("{operator:?}").hash(hasher);
            normalize_string(right).hash(hasher);
        }
        ClaimStructure::And(claims) => {
            "and".hash(hasher);
            for claim in claims {
                hash_structure_recursive(claim, hasher);
            }
        }
        ClaimStructure::Or(claims) => {
            "or".hash(hasher);
            for claim in claims {
                hash_structure_recursive(claim, hasher);
            }
        }
        ClaimStructure::Not(inner) => {
            "not".hash(hasher);
            hash_structure_recursive(inner, hasher);
        }
        ClaimStructure::Implies {
            premise,
            conclusion,
        } => {
            "implies".hash(hasher);
            hash_structure_recursive(premise, hasher);
            hash_structure_recursive(conclusion, hasher);
        }
        ClaimStructure::Quantified {
            quantifier,
            variable,
            domain,
            body,
        } => {
            "quantified".hash(hasher);
            format!("{quantifier:?}").hash(hasher);
            normalize_string(variable).hash(hasher);
            normalize_string(domain).hash(hasher);
            hash_structure_recursive(body, hasher);
        }
        ClaimStructure::Temporal {
            event,
            time_relation,
            reference,
        } => {
            "temporal".hash(hasher);
            normalize_string(event).hash(hasher);
            format!("{time_relation:?}").hash(hasher);
            normalize_string(reference).hash(hasher);
        }
        ClaimStructure::Causal {
            cause,
            effect,
            strength,
        } => {
            "causal".hash(hasher);
            hash_structure_recursive(cause, hasher);
            hash_structure_recursive(effect, hasher);
            format!("{strength:?}").hash(hasher);
        }
        ClaimStructure::Modal { claim, modality } => {
            "modal".hash(hasher);
            hash_structure_recursive(claim, hasher);
            format!("{modality:?}").hash(hasher);
        }
        ClaimStructure::Raw(text) => {
            "raw".hash(hasher);
            normalize_string(text).hash(hasher);
        }
    }
}

// ---------------------------------------------------------------------------
// String utilities
// ---------------------------------------------------------------------------

/// Normalize a string for comparison.
#[must_use]
pub fn normalize_string(s: &str) -> String {
    s.trim().to_lowercase()
}

// ---------------------------------------------------------------------------
// Subject extraction
// ---------------------------------------------------------------------------

/// Extract the subject from a claim structure if applicable.
#[must_use]
pub fn extract_subject(structure: &ClaimStructure) -> Option<String> {
    match structure {
        ClaimStructure::Predicate { subject, .. } => Some(normalize_string(subject)),
        ClaimStructure::Comparison { left, .. } => Some(normalize_string(left)),
        ClaimStructure::Temporal { event, .. } => Some(normalize_string(event)),
        ClaimStructure::Not(inner) => extract_subject(inner),
        ClaimStructure::Modal { claim, .. } => extract_subject(claim),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Specificity estimation
// ---------------------------------------------------------------------------

/// Estimate the specificity of a claim structure (higher = more specific).
#[must_use]
pub fn estimate_specificity(structure: &ClaimStructure) -> u32 {
    match structure {
        ClaimStructure::Raw(_) => 1,
        ClaimStructure::Predicate { object: None, .. } => 2,
        ClaimStructure::Predicate {
            object: Some(_), ..
        }
        | ClaimStructure::Comparison { .. }
        | ClaimStructure::Temporal { .. } => 3,
        ClaimStructure::Not(inner) => estimate_specificity(inner) + 1,
        ClaimStructure::Modal { claim, .. } => estimate_specificity(claim) + 1,
        ClaimStructure::Implies {
            premise,
            conclusion,
        } => estimate_specificity(premise) + estimate_specificity(conclusion),
        ClaimStructure::And(claims) | ClaimStructure::Or(claims) => {
            claims.iter().map(estimate_specificity).sum::<u32>() + 1
        }
        ClaimStructure::Quantified { body, .. } => estimate_specificity(body) + 2,
        ClaimStructure::Causal { cause, effect, .. } => {
            estimate_specificity(cause) + estimate_specificity(effect) + 1
        }
    }
}

// ---------------------------------------------------------------------------
// Conflict detection functions
// ---------------------------------------------------------------------------

/// Check for direct contradiction (A and NOT A).
#[must_use]
pub fn check_direct_contradiction(
    claim1: &LogicalClaim,
    claim2: &LogicalClaim,
) -> Option<ClaimConflict> {
    // Check if one is the negation of the other
    if let ClaimStructure::Not(inner) = &claim1.structure {
        let inner_hash = hash_structure(inner);
        let claim2_hash = hash_structure(&claim2.structure);
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
        let inner_hash = hash_structure(inner);
        let claim1_hash = hash_structure(&claim1.structure);
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
#[must_use]
pub fn check_predicate_conflict(
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
        )
            // Same subject with contradictory predicates
            if normalize_string(s1) == normalize_string(s2)
                && (are_opposite_predicates(p1, p2)
                    || (normalize_string(p1) == normalize_string(p2)
                        && are_contradictory_objects(o1.as_deref(), o2.as_deref())))
            => {
                return Some(ClaimConflict::new(
                    &claim1.id,
                    &claim2.id,
                    ConflictType::PredicateConflict,
                    format!(
                        "Subject '{s1}' has contradictory properties: '{p1}' vs '{p2}'"
                    ),
                ));
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
                    && normalize_string(ns) == normalize_string(ps)
                    && normalize_string(np) == normalize_string(pp)
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
#[must_use]
pub fn check_comparison_conflict(
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
        let l1_norm = normalize_string(l1);
        let r1_norm = normalize_string(r1);
        let l2_norm = normalize_string(l2);
        let r2_norm = normalize_string(r2);

        // A > B and A < B
        if l1_norm == l2_norm && r1_norm == r2_norm && are_opposite_comparisons(*op1, *op2) {
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
#[must_use]
pub fn check_temporal_conflict(
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
        let e1_norm = normalize_string(e1);
        let r1_norm = normalize_string(r1);
        let e2_norm = normalize_string(e2);
        let r2_norm = normalize_string(r2);

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
        if e1_norm == e2_norm && r1_norm == r2_norm && are_opposite_time_relations(tr1, tr2) {
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
#[must_use]
pub fn check_modal_conflict(claim1: &LogicalClaim, claim2: &LogicalClaim) -> Option<ClaimConflict> {
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
        let c1_hash = hash_structure(c1);
        let c2_hash = hash_structure(c2);

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
#[must_use]
pub fn check_causal_conflict(
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
        let c1_hash = hash_structure(c1);
        let e1_hash = hash_structure(e1);
        let c2_hash = hash_structure(c2);
        let e2_hash = hash_structure(e2);

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

// ---------------------------------------------------------------------------
// Predicate/operator comparison helpers
// ---------------------------------------------------------------------------

/// Check if two predicates are opposites.
#[must_use]
pub fn are_opposite_predicates(p1: &str, p2: &str) -> bool {
    let p1_norm = normalize_string(p1);
    let p2_norm = normalize_string(p2);

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
#[must_use]
pub fn are_contradictory_objects(o1: Option<&str>, o2: Option<&str>) -> bool {
    match (o1, o2) {
        (Some(obj1), Some(obj2)) => {
            let o1_norm = normalize_string(obj1);
            let o2_norm = normalize_string(obj2);
            // Different non-empty objects for the same predicate are contradictory
            !o1_norm.is_empty() && !o2_norm.is_empty() && o1_norm != o2_norm
        }
        _ => false,
    }
}

/// Check if two comparison operators are opposites.
#[must_use]
pub fn are_opposite_comparisons(
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
#[must_use]
pub fn are_opposite_time_relations(
    tr1: &crate::types::TimeRelation,
    tr2: &crate::types::TimeRelation,
) -> bool {
    use crate::types::TimeRelation::{After, Before};
    matches!((tr1, tr2), (Before, After) | (After, Before))
}
