//! Conflict resolution under a configurable policy.
//!
//! Given a detected [`PassageConflict`], the resolver decides which side wins
//! using one of three deterministic policies — recency, authority, or majority —
//! and returns a [`ConflictResolution`] with a human-readable rationale.

use std::collections::HashMap;
use std::collections::HashSet;

use super::detector::{shared_terms, tokenize};
use super::types::{ConflictPolicy, ConflictResolution, KnowledgeConflictConfig, PassageConflict};
use crate::types::{Document, DocumentId};

// ── support counting (Majority) ─────────────────────────────────────────────────

/// Count how many passages in `docs` echo the subject vocabulary of `claim`.
///
/// A passage counts as supporting `claim` when its content shares at least
/// `min_shared` content tokens with the claim. The two passages directly party to
/// the conflict (`exclude_a`, `exclude_b`) are skipped so that support is measured
/// across the *rest* of the corpus.
fn support_count(
    claim: &str,
    docs: &[Document],
    min_shared: usize,
    exclude_a: usize,
    exclude_b: usize,
) -> usize {
    let claim_terms: HashSet<String> = tokenize(claim)
        .into_iter()
        .filter(|t| !t.chars().all(|c| c.is_ascii_digit()))
        .collect();
    if claim_terms.is_empty() {
        return 0;
    }

    docs.iter()
        .enumerate()
        .filter(|(idx, _)| *idx != exclude_a && *idx != exclude_b)
        .filter(|(_, doc)| {
            let shared = shared_terms(claim, &doc.content);
            shared.len() >= min_shared
        })
        .count()
}

// ── ConflictResolver ────────────────────────────────────────────────────────────

/// Resolves [`PassageConflict`]s by selecting a winning side under the configured
/// [`ConflictPolicy`].
///
/// The three policies are:
///
/// * [`ConflictPolicy::Recency`] — the passage with the lower age (in days,
///   supplied via the `ages_days` map) wins.
/// * [`ConflictPolicy::Authority`] — the passage with the higher authority score
///   (supplied via the `authority` map) wins.
/// * [`ConflictPolicy::Majority`] — the claim whose subject terms are echoed by
///   more *other* passages in the corpus wins.
///
/// All policies break ties deterministically in favour of `passage_a`.
#[derive(Debug, Clone)]
pub struct ConflictResolver {
    /// Configuration carrying the active policy and shared-term gate.
    pub config: KnowledgeConflictConfig,
}

impl ConflictResolver {
    /// Construct a resolver with the given configuration.
    #[must_use]
    pub fn new(config: KnowledgeConflictConfig) -> Self {
        Self { config }
    }

    /// Resolve a single `conflict` under the configured policy.
    ///
    /// The `ages_days` map provides the age (in days) of each document by
    /// [`DocumentId`]; the `authority` map provides each document's authority
    /// score. Missing entries are treated as the *least* favourable value
    /// (maximum age, minimum authority) so that documents lacking metadata never
    /// out-rank documents that carry it.
    ///
    /// The returned [`ConflictResolution`] always lists the single losing passage
    /// in `losers` and carries a non-empty `rationale`.
    #[must_use]
    pub fn resolve(
        &self,
        conflict: &PassageConflict,
        docs: &[Document],
        ages_days: &HashMap<DocumentId, f64>,
        authority: &HashMap<DocumentId, f32>,
    ) -> ConflictResolution {
        let policy = self.config.policy;
        match policy {
            ConflictPolicy::Recency => Self::resolve_recency(conflict, docs, ages_days),
            ConflictPolicy::Authority => Self::resolve_authority(conflict, docs, authority),
            ConflictPolicy::Majority => self.resolve_majority(conflict, docs),
        }
    }

    /// Resolve every conflict in `conflicts` under the configured policy.
    ///
    /// Equivalent to mapping [`ConflictResolver::resolve`] over the slice; the
    /// output preserves input order.
    #[must_use]
    pub fn resolve_all(
        &self,
        conflicts: &[PassageConflict],
        docs: &[Document],
        ages_days: &HashMap<DocumentId, f64>,
        authority: &HashMap<DocumentId, f32>,
    ) -> Vec<ConflictResolution> {
        conflicts
            .iter()
            .map(|c| self.resolve(c, docs, ages_days, authority))
            .collect()
    }

    // ── policy implementations ──────────────────────────────────────────────

    /// Recency policy: the passage with the lower age (days) wins.
    fn resolve_recency(
        conflict: &PassageConflict,
        docs: &[Document],
        ages_days: &HashMap<DocumentId, f64>,
    ) -> ConflictResolution {
        let age = |idx: usize| -> f64 {
            docs.get(idx)
                .and_then(|d| ages_days.get(&d.id).copied())
                .unwrap_or(f64::INFINITY)
        };
        let age_a = age(conflict.passage_a);
        let age_b = age(conflict.passage_b);

        // Lower age wins; ties favour passage_a.
        let a_wins = age_a <= age_b;
        let (winner, loser, winner_age, loser_age) = if a_wins {
            (conflict.passage_a, conflict.passage_b, age_a, age_b)
        } else {
            (conflict.passage_b, conflict.passage_a, age_b, age_a)
        };

        let rationale = format!(
            "Recency policy: passage {winner} (age {winner_age:.1} days) is newer than passage {loser} (age {loser_age:.1} days), so its claim is retained."
        );
        ConflictResolution::new(winner, vec![loser], ConflictPolicy::Recency, rationale)
    }

    /// Authority policy: the passage with the higher authority score wins.
    fn resolve_authority(
        conflict: &PassageConflict,
        docs: &[Document],
        authority: &HashMap<DocumentId, f32>,
    ) -> ConflictResolution {
        let auth = |idx: usize| -> f32 {
            docs.get(idx)
                .and_then(|d| authority.get(&d.id).copied())
                .unwrap_or(f32::NEG_INFINITY)
        };
        let auth_a = auth(conflict.passage_a);
        let auth_b = auth(conflict.passage_b);

        // Higher authority wins; ties favour passage_a.
        let a_wins = auth_a >= auth_b;
        let (winner, loser, winner_auth, loser_auth) = if a_wins {
            (conflict.passage_a, conflict.passage_b, auth_a, auth_b)
        } else {
            (conflict.passage_b, conflict.passage_a, auth_b, auth_a)
        };

        let rationale = format!(
            "Authority policy: passage {winner} (authority {winner_auth:.2}) outranks passage {loser} (authority {loser_auth:.2}), so its claim is retained."
        );
        ConflictResolution::new(winner, vec![loser], ConflictPolicy::Authority, rationale)
    }

    /// Majority policy: the claim echoed by more *other* passages wins.
    fn resolve_majority(
        &self,
        conflict: &PassageConflict,
        docs: &[Document],
    ) -> ConflictResolution {
        let min_shared = self.config.min_shared_terms;
        let support_a = support_count(
            &conflict.claim_a,
            docs,
            min_shared,
            conflict.passage_a,
            conflict.passage_b,
        );
        let support_b = support_count(
            &conflict.claim_b,
            docs,
            min_shared,
            conflict.passage_a,
            conflict.passage_b,
        );

        // More supporters wins; ties favour passage_a.
        let a_wins = support_a >= support_b;
        let (winner, loser, winner_support, loser_support) = if a_wins {
            (conflict.passage_a, conflict.passage_b, support_a, support_b)
        } else {
            (conflict.passage_b, conflict.passage_a, support_b, support_a)
        };

        let rationale = format!(
            "Majority policy: passage {winner}'s claim is corroborated by {winner_support} other passage(s) versus {loser_support} for passage {loser}, so it is retained."
        );
        ConflictResolution::new(winner, vec![loser], ConflictPolicy::Majority, rationale)
    }
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new(KnowledgeConflictConfig::default())
    }
}
