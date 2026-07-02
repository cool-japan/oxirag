//! [`StructRagEngine`] — ties the router, restructurer, and reasoner
//! together: route the query to its optimal [`StructRagStructureKind`],
//! restructure the retrieved passages into it, and reason over the result.

use super::reason::StructRagReasoner;
use super::restructure::StructRagRestructurer;
use super::router::StructRagRouter;
use super::types::{
    StructRagConfig, StructRagError, StructRagPassage, StructRagResult, StructRagRoutingDecision,
    StructRagStructureKind,
};

// ── StructRagEngine ──────────────────────────────────────────────────────────

/// Runs the full `StructRAG` pipeline: **route** (infer the optimal
/// [`StructRagStructureKind`] for the query via [`StructRagRouter`]),
/// **restructure** (transform the retrieved passages into that kind's
/// concrete [`crate::structrag::StructRagKnowledgeStructure`] via
/// [`StructRagRestructurer`]), and **reason** (read an answer from the
/// structure via [`StructRagReasoner`]).
///
/// The router, restructurer, and reasoner are all stateless (zero-sized)
/// helpers; only [`StructRagEngine::config`] carries state, so the engine is
/// cheap to construct and clone.
#[derive(Debug, Clone, Default)]
pub struct StructRagEngine {
    /// Configuration governing routing thresholds, enabled kinds, structure
    /// size, and tie-break order.
    pub config: StructRagConfig,
    router: StructRagRouter,
    restructurer: StructRagRestructurer,
    reasoner: StructRagReasoner,
}

impl StructRagEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: StructRagConfig) -> Self {
        Self {
            config,
            router: StructRagRouter::new(),
            restructurer: StructRagRestructurer::new(),
            reasoner: StructRagReasoner::new(),
        }
    }

    /// Infer the optimal structure kind for `query` without restructuring or
    /// reasoning. Exposed standalone so callers can inspect or log the
    /// routing decision before committing to the (potentially larger)
    /// restructuring step.
    ///
    /// # Errors
    ///
    /// See [`StructRagRouter::route`].
    pub fn route(&self, query: &str) -> Result<StructRagRoutingDecision, StructRagError> {
        self.router.route(query, &self.config)
    }

    /// Run the complete pipeline: route `query` to its optimal structure
    /// kind, restructure `passages` into it, and reason over the result.
    ///
    /// # Errors
    ///
    /// - [`StructRagError::EmptyQuery`] if `query` is blank.
    /// - [`StructRagError::EmptyPassages`] if `passages` is empty.
    /// - [`StructRagError::NoEnabledStructureKinds`] if
    ///   [`StructRagConfig::enabled_kinds`] is empty.
    pub fn run(
        &self,
        query: &str,
        passages: &[StructRagPassage],
    ) -> Result<StructRagResult, StructRagError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(StructRagError::EmptyQuery);
        }
        if passages.is_empty() {
            return Err(StructRagError::EmptyPassages);
        }

        let routing = self.router.route(trimmed, &self.config)?;
        let structure = self
            .restructurer
            .restructure(routing.kind, passages, &self.config);
        let answer = self.reasoner.reason(trimmed, &structure);

        Ok(StructRagResult {
            query: trimmed.to_string(),
            routing,
            structure,
            answer,
        })
    }

    /// Run the pipeline forcing a specific `kind`, bypassing the router
    /// entirely. Useful when the caller already knows which knowledge
    /// representation fits the task (e.g. a UI toggle), or wants to compare
    /// answers across kinds for the same passages.
    ///
    /// # Errors
    ///
    /// - [`StructRagError::EmptyQuery`] if `query` is blank.
    /// - [`StructRagError::EmptyPassages`] if `passages` is empty.
    /// - [`StructRagError::DisabledStructureKind`] if `kind` is not present
    ///   in [`StructRagConfig::enabled_kinds`].
    pub fn run_with_kind(
        &self,
        query: &str,
        passages: &[StructRagPassage],
        kind: StructRagStructureKind,
    ) -> Result<StructRagResult, StructRagError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(StructRagError::EmptyQuery);
        }
        if passages.is_empty() {
            return Err(StructRagError::EmptyPassages);
        }
        if !self.config.is_enabled(kind) {
            return Err(StructRagError::DisabledStructureKind(kind));
        }

        let structure = self.restructurer.restructure(kind, passages, &self.config);
        let answer = self.reasoner.reason(trimmed, &structure);
        let routing = StructRagRoutingDecision {
            kind,
            confidence: 1.0,
            rationale: format!("kind forced by caller via run_with_kind({kind}), router bypassed"),
            scores: StructRagStructureKind::all()
                .into_iter()
                .map(|k| (k, if k == kind { 1.0 } else { 0.0 }))
                .collect(),
        };

        Ok(StructRagResult {
            query: trimmed.to_string(),
            routing,
            structure,
            answer,
        })
    }
}
