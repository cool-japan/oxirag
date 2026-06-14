//! [`SkeletonOfThoughtEngine`] — two-stage skeleton-then-expand generation.

use crate::skeleton_of_thought::types::{
    PointExpander, SkeletonConfig, SkeletonGenerator, SkeletonOutput, SkeletonPoint, SotError,
};

// ── SkeletonOfThoughtEngine ─────────────────────────────────────────────────────

/// Drives the Skeleton-of-Thought pipeline.
///
/// Stage 1 asks a [`SkeletonGenerator`] for concise point headers; stage 2 has a
/// [`PointExpander`] expand each header independently into full content. The
/// per-point contents are then joined into the final answer. The generator and
/// expander are supplied *per call* through [`SkeletonOfThoughtEngine::run`],
/// mirroring the caller-supplies-executor pattern used across the crate.
///
/// Every step is deterministic given deterministic collaborators: the skeleton
/// is processed in order, capped at [`SkeletonConfig::max_points`], with empty
/// headers dropped before expansion.
#[derive(Debug, Clone, Default)]
pub struct SkeletonOfThoughtEngine {
    /// Configuration for this engine.
    pub config: SkeletonConfig,
}

impl SkeletonOfThoughtEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: SkeletonConfig) -> Self {
        Self { config }
    }

    /// Run the two-stage Skeleton-of-Thought pipeline for `query`.
    ///
    /// 1. Ask `generator` for the skeleton (point headers).
    /// 2. Drop headers that are empty after trimming, then truncate to at most
    ///    [`SkeletonConfig::max_points`] points.
    /// 3. Expand each remaining header in order via `expander`, building a
    ///    [`SkeletonPoint`] whose `index` is its 0-based position.
    /// 4. Join the expanded contents with
    ///    [`SkeletonConfig::join_separator`] to form the answer.
    ///
    /// # Errors
    ///
    /// - [`SotError::EmptyQuery`] when `query` is empty after trimming.
    /// - [`SotError::EmptySkeleton`] when no usable point headers remain after
    ///   dropping empties (including when the generator returns nothing).
    pub fn run<G, E>(
        &self,
        query: &str,
        generator: &G,
        expander: &E,
    ) -> Result<SkeletonOutput, SotError>
    where
        G: SkeletonGenerator + ?Sized,
        E: PointExpander + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(SotError::EmptyQuery);
        }

        // Stage 1: skeleton. Drop empty headers, then cap at `max_points`.
        let headers: Vec<String> = generator
            .skeleton(query)
            .into_iter()
            .filter(|header| !header.trim().is_empty())
            .take(self.config.max_points)
            .collect();

        if headers.is_empty() {
            return Err(SotError::EmptySkeleton);
        }

        // Stage 2: expand each point independently, preserving order.
        let points: Vec<SkeletonPoint> = headers
            .into_iter()
            .enumerate()
            .map(|(index, header)| {
                let content = expander.expand(query, &header);
                SkeletonPoint {
                    index,
                    header,
                    content,
                }
            })
            .collect();

        // Join the expansions in skeleton order to form the answer.
        let answer = points
            .iter()
            .map(|point| point.content.as_str())
            .collect::<Vec<_>>()
            .join(&self.config.join_separator);

        Ok(SkeletonOutput { points, answer })
    }
}
