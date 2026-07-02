//! [`ConstitutionalEngine`] — the per-principle critique-then-revise loop.

use crate::constitutional_critique::types::{
    ConstitutionalConfig, ConstitutionalError, ConstitutionalPrinciple, ConstitutionalResult,
    ConstitutionalReviser, ConstitutionalTraceEntry,
};

// ── ConstitutionalEngine ─────────────────────────────────────────────────

/// Drives Constitutional-AI-style critique-and-revise: applies a fixed,
/// ordered list of [`ConstitutionalPrinciple`]s to a draft, one at a time.
///
/// For each principle, the *current* draft — which may already carry
/// revisions from earlier principles — is critiqued; on violation it is
/// revised, and the revision becomes the current draft for every later
/// principle. The reviser is provided *per call* through the generic
/// [`ConstitutionalEngine::run`] method, mirroring the
/// caller-supplies-executor pattern used across this crate (see e.g.
/// `self_refine::SelfRefineEngine`).
#[derive(Debug, Clone, Default)]
pub struct ConstitutionalEngine {
    /// Configuration for this engine.
    pub config: ConstitutionalConfig,
}

impl ConstitutionalEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: ConstitutionalConfig) -> Self {
        Self { config }
    }

    /// Run the full constitution — `principles`, in order — against
    /// `draft`.
    ///
    /// For each principle (skipping any not named in
    /// [`ConstitutionalConfig::enabled_principle_ids`], when set):
    ///
    /// 1. `reviser` critiques the *current* draft (the original draft, or
    ///    the most recent revision from an earlier principle) against
    ///    this single principle.
    /// 2. If the critique flags a violation **and**
    ///    [`ConstitutionalConfig::max_revisions`] has not yet been
    ///    reached, `reviser` revises the current draft and the revision
    ///    becomes the current draft for every later principle.
    /// 3. If the critique found no violation, or a violation was found
    ///    but the revision cap had already been reached, the draft is
    ///    left unchanged for this step.
    /// 4. The `(principle, critique, revision)` triple is appended to the
    ///    trace regardless of outcome, so the trace always covers every
    ///    principle that was not filtered out by
    ///    `enabled_principle_ids` — unless the run stops early (see
    ///    below).
    ///
    /// A running count of *consecutive* principles that found no
    /// violation is maintained (a violation — revised or not, because the
    /// cap was reached — resets it to zero). If
    /// [`ConstitutionalConfig::stop_after_consecutive_clean`] is
    /// `Some(n)` with `n > 0` and that count reaches `n`, the run stops
    /// immediately and [`ConstitutionalResult::stopped_early`] is `true`;
    /// principles after that point are never critiqued and never appear
    /// in the trace.
    ///
    /// # Errors
    ///
    /// - [`ConstitutionalError::EmptyDraft`] if `draft` is empty or
    ///   contains only whitespace.
    /// - Any error returned by `reviser`'s
    ///   [`critique`](ConstitutionalReviser::critique) or
    ///   [`revise`](ConstitutionalReviser::revise) methods is propagated
    ///   unchanged, immediately stopping the run.
    pub fn run<Rv>(
        &self,
        draft: &str,
        principles: &[ConstitutionalPrinciple],
        reviser: &Rv,
    ) -> Result<ConstitutionalResult, ConstitutionalError>
    where
        Rv: ConstitutionalReviser + ?Sized,
    {
        if draft.trim().is_empty() {
            return Err(ConstitutionalError::EmptyDraft);
        }

        let original_draft = draft.to_string();
        let mut current = original_draft.clone();
        let mut trace: Vec<ConstitutionalTraceEntry> = Vec::with_capacity(principles.len());
        let mut revisions_applied = 0usize;
        let mut consecutive_clean = 0usize;
        let mut stopped_early = false;

        for principle in principles {
            if !self.is_enabled(principle) {
                continue;
            }

            let critique = reviser.critique(&current, principle)?;
            let can_revise = critique.violated && revisions_applied < self.config.max_revisions;

            if can_revise {
                let revision = reviser.revise(&current, &critique)?;
                current.clone_from(&revision.after);
                revisions_applied += 1;
                consecutive_clean = 0;
                trace.push((principle.clone(), critique, Some(revision)));
            } else {
                consecutive_clean = if critique.violated {
                    0
                } else {
                    consecutive_clean + 1
                };
                trace.push((principle.clone(), critique, None));
            }

            if let Some(threshold) = self.config.stop_after_consecutive_clean
                && threshold > 0
                && consecutive_clean >= threshold
            {
                stopped_early = true;
                break;
            }
        }

        Ok(ConstitutionalResult {
            original_draft,
            final_draft: current,
            trace,
            revisions_applied,
            stopped_early,
        })
    }

    /// Return `true` when `principle` should be applied under this
    /// engine's configuration.
    fn is_enabled(&self, principle: &ConstitutionalPrinciple) -> bool {
        self.config
            .enabled_principle_ids
            .as_ref()
            .is_none_or(|ids| ids.iter().any(|id| id == &principle.id))
    }
}
