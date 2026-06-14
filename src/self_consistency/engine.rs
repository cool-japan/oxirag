//! [`SelfConsistencyEngine`] — clustering and marginalization over reasoning paths.

use crate::self_consistency::types::{
    AnswerCluster, ReasoningPath, ReasoningSampler, SelfConsistencyConfig, SelfConsistencyError,
    SelfConsistencyOutput, answers_equivalent,
};

// ── SelfConsistencyEngine ────────────────────────────────────────────────────────

/// Engine that marginalizes a winning answer over diverse reasoning paths.
///
/// Given a set of sampled [`ReasoningPath`]s, the engine clusters their final
/// answers by semantic equivalence (greedy single-linkage), tallies each
/// cluster's vote mass under the configured weighting, and selects the cluster
/// with the most votes. Confidence is the winning cluster's share of the total
/// vote mass. All steps are fully deterministic, including tie-breaking.
#[derive(Debug, Clone)]
pub struct SelfConsistencyEngine {
    /// Configuration for this engine.
    pub config: SelfConsistencyConfig,
}

impl SelfConsistencyEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: SelfConsistencyConfig) -> Self {
        Self { config }
    }

    /// Sample `config.num_paths` reasoning paths and marginalize over them.
    ///
    /// Paths are drawn with indices `0..num_paths` so that a deterministic
    /// sampler yields a deterministic result.
    ///
    /// # Errors
    ///
    /// - [`SelfConsistencyError::EmptyQuestion`] when `question` is empty after
    ///   trimming.
    /// - [`SelfConsistencyError::NoPaths`] when `config.num_paths` is zero.
    pub fn run<S: ReasoningSampler + ?Sized>(
        &self,
        question: &str,
        sampler: &S,
    ) -> Result<SelfConsistencyOutput, SelfConsistencyError> {
        if question.trim().is_empty() {
            return Err(SelfConsistencyError::EmptyQuestion);
        }
        if self.config.num_paths == 0 {
            return Err(SelfConsistencyError::NoPaths);
        }
        let paths: Vec<ReasoningPath> = (0..self.config.num_paths)
            .map(|index| sampler.sample(question, index))
            .collect();
        self.marginalize(paths)
    }

    /// Cluster the answers in `paths` and select the modal answer by vote.
    ///
    /// Clustering uses greedy single-linkage: each path joins the first
    /// existing cluster it is equivalent to (per
    /// [`SelfConsistencyConfig::equivalence_threshold`]), otherwise it seeds a
    /// new cluster. The winning cluster has the highest vote total; ties break
    /// by most members, then by lexicographically smallest canonical answer.
    /// Confidence is `winning_votes / total_votes`.
    ///
    /// # Errors
    ///
    /// Returns [`SelfConsistencyError::NoPaths`] when `paths` is empty.
    pub fn marginalize(
        &self,
        paths: Vec<ReasoningPath>,
    ) -> Result<SelfConsistencyOutput, SelfConsistencyError> {
        if paths.is_empty() {
            return Err(SelfConsistencyError::NoPaths);
        }

        let threshold = self.config.equivalence_threshold;
        let mut clusters: Vec<AnswerCluster> = Vec::new();

        for (idx, path) in paths.iter().enumerate() {
            let weight = self.config.weighting.weight_of(path);
            let mut joined = false;
            for cluster in &mut clusters {
                // Single-linkage: equivalent to the cluster's representative.
                if answers_equivalent(&cluster.canonical, &path.answer, threshold) {
                    cluster.members.push(idx);
                    cluster.votes += weight;
                    joined = true;
                    break;
                }
            }
            if !joined {
                clusters.push(AnswerCluster {
                    canonical: normalize_for_display(&path.answer),
                    members: vec![idx],
                    votes: weight,
                });
            }
        }

        let total_votes: f32 = clusters.iter().map(|c| c.votes).sum();

        // Deterministic ordering: votes desc, then members desc, then canonical asc.
        clusters.sort_by(cluster_order);

        // The winner is the first cluster after sorting (best by the same key).
        // `clusters` is guaranteed non-empty because `paths` is non-empty, so we
        // derive the winner's stats without an unwrap/expect.
        let (answer, winning_votes) = match clusters.first() {
            Some(winner) => (winner.canonical.clone(), winner.votes),
            None => (String::new(), 0.0),
        };
        let confidence = if total_votes > 0.0 {
            winning_votes / total_votes
        } else {
            0.0
        };

        Ok(SelfConsistencyOutput {
            answer,
            confidence,
            clusters,
            paths,
        })
    }
}

impl Default for SelfConsistencyEngine {
    fn default() -> Self {
        Self::new(SelfConsistencyConfig::default())
    }
}

// ── Ordering & display helpers ───────────────────────────────────────────────────

/// Total order over clusters used both for ranking and selecting the winner:
/// votes descending, then member count descending, then canonical ascending.
fn cluster_order(a: &AnswerCluster, b: &AnswerCluster) -> std::cmp::Ordering {
    b.votes
        .partial_cmp(&a.votes)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.members.len().cmp(&a.members.len()))
        .then_with(|| a.canonical.cmp(&b.canonical))
}

/// Normalize an answer for use as a cluster's canonical display string.
///
/// Falls back to the trimmed original when normalization yields an empty
/// string (e.g. punctuation-only answers), so clusters remain identifiable.
fn normalize_for_display(answer: &str) -> String {
    let canon = crate::self_consistency::types::normalize_answer(answer);
    if canon.is_empty() {
        answer.trim().to_string()
    } else {
        canon
    }
}
