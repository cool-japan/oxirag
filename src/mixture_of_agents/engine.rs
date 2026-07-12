//! [`MoaEngine`] — layered mixture-of-agents orchestration — together with
//! the deterministic [`MockMoaProposer`] implementation of
//! [`super::types::MoaProposer`].

use std::collections::BTreeSet;

use super::aggregator::{
    DEFAULT_CLAIM_SIMILARITY_THRESHOLD, cluster_sentences, content_terms, jaccard, split_sentences,
};
use super::types::{
    MoaAggregator, MoaConfig, MoaContextMode, MoaError, MoaLayer, MoaLayerStats, MoaProposer,
    MoaResponse, MoaTrace,
};

// ── MockMoaProposer ──────────────────────────────────────────────────────────

/// Deterministic [`MoaProposer`] for tests and examples.
///
/// No live model is involved: [`MockMoaProposer::propose`] always opens with
/// a fixed sentence naming its `label` and the query, followed by every
/// fact in `self.facts` verbatim. When `prior_responses` is non-empty, it
/// then scans every prior response for sentences carrying content terms it
/// does not already know (from its own opening/facts, or from an
/// already-incorporated prior sentence) and appends exactly those — modeling
/// a proposer that genuinely reads and builds on what came before, without
/// pointlessly re-stating content it already produced. Once a proposer's
/// own knowledge already covers everything visible in `prior_responses`
/// (e.g. because a later layer's context is a strict subset of what it
/// already incorporated one layer earlier), its output stops changing —
/// which is exactly the convergence signal
/// [`MoaConfig::early_stop_similarity`] is designed to detect.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MockMoaProposer {
    /// A short human-readable tag folded into the opening sentence.
    pub label: String,
    /// Facts this proposer states unconditionally, regardless of context.
    pub facts: Vec<String>,
}

impl MockMoaProposer {
    /// Create a new mock proposer with the given human-readable `label` and
    /// no facts.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            facts: Vec::new(),
        }
    }

    /// Append one fact this proposer states unconditionally.
    #[must_use]
    pub fn with_fact(mut self, fact: impl Into<String>) -> Self {
        self.facts.push(fact.into());
        self
    }
}

impl MoaProposer for MockMoaProposer {
    fn propose(&self, query: &str, prior_responses: &[MoaResponse]) -> Result<String, MoaError> {
        let label = if self.label.trim().is_empty() {
            "Proposer".to_string()
        } else {
            self.label.clone()
        };

        let mut sentences: Vec<String> = vec![format!("{label} answers \"{query}\".")];
        sentences.extend(self.facts.iter().cloned());

        if !prior_responses.is_empty() {
            let mut known_terms = content_terms(&sentences.join(" "));
            for prior in prior_responses {
                for sentence in split_sentences(&prior.text) {
                    let terms = content_terms(&sentence);
                    if terms.is_empty() {
                        continue;
                    }
                    if terms.iter().any(|term| !known_terms.contains(term)) {
                        known_terms.extend(terms);
                        sentences.push(format!("Incorporating prior input: {sentence}"));
                    }
                }
            }
        }

        Ok(sentences.join(" "))
    }
}

// ── MoaLayerStats computation ────────────────────────────────────────────────

/// Compute [`MoaLayerStats`] for a completed layer.
fn compute_layer_stats(
    layer_index: usize,
    proposals: &[MoaResponse],
    aggregate_text: &str,
    previous_aggregate_text: Option<&str>,
) -> MoaLayerStats {
    let term_sets: Vec<BTreeSet<String>> =
        proposals.iter().map(|p| content_terms(&p.text)).collect();
    let mut agreement_sum = 0.0_f32;
    let mut pair_count = 0_usize;
    for i in 0..term_sets.len() {
        for j in (i + 1)..term_sets.len() {
            agreement_sum += jaccard(&term_sets[i], &term_sets[j]);
            pair_count += 1;
        }
    }
    let agreement = if pair_count == 0 {
        1.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        {
            agreement_sum / pair_count as f32
        }
    };

    let clusters = cluster_sentences(proposals, DEFAULT_CLAIM_SIMILARITY_THRESHOLD);
    let distinct_claims = clusters.len();
    let aggregate_term_sets: Vec<BTreeSet<String>> = split_sentences(aggregate_text)
        .iter()
        .map(|sentence| content_terms(sentence))
        .collect();
    let covered_claims = clusters
        .iter()
        .filter(|cluster| {
            aggregate_term_sets
                .iter()
                .any(|terms| jaccard(&cluster.terms, terms) >= DEFAULT_CLAIM_SIMILARITY_THRESHOLD)
        })
        .count();
    let coverage_ratio = if distinct_claims == 0 {
        1.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        {
            covered_claims as f32 / distinct_claims as f32
        }
    };

    let change_from_previous = previous_aggregate_text.map(|previous| {
        let similarity = jaccard(&content_terms(previous), &content_terms(aggregate_text));
        1.0 - similarity
    });

    MoaLayerStats {
        layer_index,
        agreement,
        distinct_claims,
        covered_claims,
        coverage_ratio,
        change_from_previous,
    }
}

// ── input validation ─────────────────────────────────────────────────────────

/// Validate a run's inputs before any layer is run.
fn validate_inputs(
    query: &str,
    proposers: &[&dyn MoaProposer],
    config: &MoaConfig,
) -> Result<(), MoaError> {
    if query.trim().is_empty() {
        return Err(MoaError::EmptyQuery);
    }
    if config.num_layers == 0 {
        return Err(MoaError::ZeroLayers);
    }
    if proposers.is_empty() {
        return Err(MoaError::ZeroProposers);
    }
    if proposers.len() != config.proposers_per_layer {
        return Err(MoaError::ProposerCountMismatch {
            expected: config.proposers_per_layer,
            actual: proposers.len(),
        });
    }
    if let Some(threshold) = config.early_stop_similarity
        && !(0.0..=1.0).contains(&threshold)
    {
        return Err(MoaError::InvalidSimilarityThreshold { value: threshold });
    }
    Ok(())
}

// ── MoaEngine ─────────────────────────────────────────────────────────────────

/// Orchestrates a layered mixture-of-agents run: [`MoaConfig::num_layers`]
/// layers of [`MoaConfig::proposers_per_layer`] proposers each, synthesized
/// by a caller-supplied [`MoaAggregator`] after every layer.
///
/// Layer `0`'s proposers always receive an empty `prior_responses` slice —
/// they answer completely independently, exactly like
/// `crate::self_consistency`'s sampled chains. From layer `1` onward, every
/// proposer in the layer receives the **same** `prior_responses` snapshot,
/// built from the *immediately preceding* layer's completed aggregate and/or
/// individual proposals per [`MoaConfig::context_mode`] — never that layer's
/// own not-yet-complete proposals from other proposers, and never anything
/// from two or more layers back. This one-layer-back, synthesis-then-next-layer
/// structure is the defining mechanic that distinguishes mixture-of-agents
/// from both `crate::self_consistency` (no cross-agent communication at all)
/// and `crate::multi_agent_debate` (participants see the *raw* multi-round
/// transcript of *every* earlier round, not one layer's *synthesized*
/// output) — see the [module-level documentation](crate::mixture_of_agents).
///
/// See the [module-level documentation](crate::mixture_of_agents) for a
/// complete runnable example.
#[derive(Debug, Clone, Default)]
pub struct MoaEngine {
    /// Configuration for this engine.
    pub config: MoaConfig,
}

impl MoaEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: MoaConfig) -> Self {
        Self { config }
    }

    /// Run a complete mixture-of-agents pass: `num_layers` layers of
    /// propose-then-aggregate (or fewer, if early stopping fires).
    ///
    /// # Errors
    ///
    /// - [`MoaError::EmptyQuery`] if `query` is blank.
    /// - [`MoaError::ZeroLayers`] if `self.config.num_layers` is `0`.
    /// - [`MoaError::ZeroProposers`] if `proposers` is empty.
    /// - [`MoaError::ProposerCountMismatch`] if `proposers.len()` does not
    ///   equal `self.config.proposers_per_layer`.
    /// - [`MoaError::InvalidSimilarityThreshold`] if
    ///   `self.config.early_stop_similarity` is `Some` and out of
    ///   `[0.0, 1.0]`.
    /// - Whatever a proposer's [`MoaProposer::propose`], or the supplied
    ///   [`MoaAggregator::aggregate`], returns — propagated unchanged.
    pub fn run<A>(
        &self,
        query: &str,
        proposers: &[&dyn MoaProposer],
        aggregator: &A,
    ) -> Result<MoaTrace, MoaError>
    where
        A: MoaAggregator + ?Sized,
    {
        validate_inputs(query, proposers, &self.config)?;

        let mut layers: Vec<MoaLayer> = Vec::with_capacity(self.config.num_layers);
        let mut prior_context: Vec<MoaResponse> = Vec::new();
        let mut previous_aggregate_text: Option<String> = None;
        let mut stopped_early = false;
        let mut early_stop_reason: Option<String> = None;

        for layer_index in 0..self.config.num_layers {
            let mut proposals: Vec<MoaResponse> = Vec::with_capacity(proposers.len());
            for (proposer_id, proposer) in proposers.iter().enumerate() {
                let text = proposer.propose(query, &prior_context)?;
                proposals.push(MoaResponse::proposal(proposer_id, layer_index, text));
            }

            let aggregate_text = aggregator.aggregate(query, &proposals)?;
            let aggregate = MoaResponse::aggregate(layer_index, aggregate_text.clone());

            let stats = compute_layer_stats(
                layer_index,
                &proposals,
                &aggregate_text,
                previous_aggregate_text.as_deref(),
            );

            prior_context = match self.config.context_mode {
                MoaContextMode::AggregateOnly => vec![aggregate.clone()],
                MoaContextMode::ProposalsOnly => proposals.clone(),
                MoaContextMode::AggregateAndProposals => {
                    let mut context = proposals.clone();
                    context.push(aggregate.clone());
                    context
                }
            };

            let should_stop = match (
                self.config.early_stop_similarity,
                stats.change_from_previous,
            ) {
                (Some(threshold), Some(change)) if 1.0 - change >= threshold => {
                    early_stop_reason = Some(format!(
                        "layer {layer_index} aggregate changed only {change:.4} from the previous layer's aggregate (similarity {:.4} >= threshold {threshold:.4})",
                        1.0 - change
                    ));
                    true
                }
                _ => false,
            };

            layers.push(MoaLayer {
                layer_index,
                proposals,
                aggregate,
                stats,
            });

            previous_aggregate_text = Some(aggregate_text);

            if should_stop {
                stopped_early = true;
                break;
            }
        }

        let final_response = layers
            .last()
            .map(|layer| layer.aggregate.text.clone())
            .unwrap_or_default();
        let layers_run = layers.len();

        Ok(MoaTrace {
            query: query.to_string(),
            layers,
            final_response,
            layers_run,
            stopped_early,
            early_stop_reason,
        })
    }
}
