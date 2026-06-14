//! Skeleton-of-Thought (Ning et al. 2023) — two-stage skeleton-then-expand
//! generation.
//!
//! Skeleton-of-Thought decouples answer generation into two stages to reduce
//! latency and impose structure:
//!
//! 1. **Skeleton** — the model emits a short list of concise *point headers*
//!    that outline the answer (the "skeleton").
//! 2. **Expansion** — each point is then expanded *independently* (conceptually
//!    in parallel) into a full sentence or short paragraph.
//!
//! The per-point expansions are finally joined into the answer. Because the
//! expansions are independent, they can be decoded concurrently; the resulting
//! answer is also cleanly list-structured.
//!
//! The language model is modeled by two **pure sync** caller-supplied traits:
//! [`SkeletonGenerator`] (stage 1) and [`PointExpander`] (stage 2). Deterministic
//! [`MockSkeletonGenerator`] and [`MockPointExpander`] implementations are
//! provided for testing.
//!
//! # Example
//!
//! ```
//! use oxirag::skeleton_of_thought::{
//!     MockPointExpander, MockSkeletonGenerator, SkeletonConfig, SkeletonOfThoughtEngine,
//! };
//!
//! let generator = MockSkeletonGenerator::new(vec![
//!     "Define photosynthesis".to_string(),
//!     "List the inputs".to_string(),
//!     "List the outputs".to_string(),
//! ]);
//! let expander = MockPointExpander::echo();
//!
//! let engine = SkeletonOfThoughtEngine::new(SkeletonConfig::default());
//! let out = engine
//!     .run("Explain photosynthesis", &generator, &expander)
//!     .unwrap();
//!
//! assert_eq!(out.points.len(), 3);
//! assert_eq!(out.points[0].index, 0);
//! assert_eq!(out.points[0].content, "Define photosynthesis: details");
//! assert_eq!(
//!     out.answer,
//!     "Define photosynthesis: details\nList the inputs: details\nList the outputs: details",
//! );
//! ```

pub mod engine;
pub mod types;

#[cfg(test)]
mod tests;

pub use engine::SkeletonOfThoughtEngine;
pub use types::{
    MockPointExpander, MockSkeletonGenerator, PointExpander, SkeletonConfig, SkeletonGenerator,
    SkeletonOutput, SkeletonPoint, SotError,
};
