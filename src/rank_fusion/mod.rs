//! Multi-list rank/score fusion beyond Reciprocal Rank Fusion.
//!
//! This module fuses several independently ranked result lists — for example
//! the outputs of a dense vector retriever, a sparse lexical retriever, and a
//! metadata filter — into a single ranked list. It is a deliberately broader,
//! standalone toolbox than the single-method `reciprocal_rank_fusion` helper in
//! [`crate::advanced_retrieval`]: here you can choose among seven fusion methods
//! and four per-list score normalization strategies.
//!
//! # Fusion methods
//!
//! | Method | Contribution per list entry | Notes |
//! |---|---|---|
//! | [`FusionMethod::CombSum`] | normalized score | plain sum across lists |
//! | [`FusionMethod::CombMnz`] | normalized score | `CombSum × hit-count` (rewards agreement) |
//! | [`FusionMethod::CombAnz`] | normalized score | `CombSum ÷ hit-count` (mean score) |
//! | [`FusionMethod::Borda`] | `L − r` | rank-based points, list length `L`, 0-based rank `r` |
//! | [`FusionMethod::Isr`] | `1 / (r + 1)^2` | inverse square rank |
//! | [`FusionMethod::WeightedSum`] | `weight × score` | per-list weights required |
//! | [`FusionMethod::Rrf`] | `1 / (k + r + 1)` | reciprocal rank fusion |
//!
//! # Score normalization
//!
//! Score-based methods consume per-list scores, which are normalized first via
//! [`ScoreNormalization`] so retrievers with different score scales combine
//! fairly. Rank-based methods ([`FusionMethod::Borda`], [`FusionMethod::Isr`],
//! [`FusionMethod::Rrf`]) ignore raw scores and are unaffected by normalization.
//!
//! # Determinism
//!
//! All operations are deterministic. Documents are deduplicated by their
//! [`DocumentId`] string and ties in the final ranking are broken by ascending
//! [`DocumentId`] string order, so identical inputs always yield identical
//! outputs.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "rank-fusion")]
//! # {
//! use oxirag::rank_fusion::{RankFusion, RankFusionConfig, FusionMethod};
//!
//! let fusion = RankFusion::new(
//!     RankFusionConfig::new().with_method(FusionMethod::CombMnz),
//! );
//!
//! let dense = vec![("doc-a".into(), 0.9_f32), ("doc-b".into(), 0.5)];
//! let sparse = vec![("doc-b".into(), 0.8_f32), ("doc-c".into(), 0.2)];
//!
//! let fused = fusion.fuse_ids(&[dense, sparse]).unwrap();
//! // "doc-b" appears in both lists, so CombMNZ boosts it.
//! assert_eq!(fused[0].0.as_str(), "doc-b");
//! # }
//! ```
//!
//! [`DocumentId`]: crate::types::DocumentId

mod fusion;
mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use fusion::RankFusion;
pub use types::{FusionMethod, RankFusionConfig, RankFusionError, ScoreNormalization};
