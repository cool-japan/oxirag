//! Corpus-poisoning / adversarial-passage defense for RAG security.
//!
//! Retrieval-augmented generation trusts whatever the retriever surfaces. That
//! trust is exploitable: an attacker who can write into the corpus may craft a
//! passage *engineered* to be retrieved for a target query and to steer the
//! generator toward a chosen (often false) answer — a **corpus-poisoning** or
//! **adversarial-passage** attack. This module screens a retrieved result set
//! for such passages *before* they reach generation.
//!
//! This is deliberately **distinct** from a noise / distractor filter. A noise
//! filter drops passages that are merely *irrelevant* — they share surface
//! keywords with the query but carry little real relevance. The
//! [`PoisoningDetector`] instead hunts for passages that are *too* engineered:
//! suspiciously query-stuffed, abnormally repetitive, or anomalous against the
//! corpus consensus.
//!
//! # Signals
//!
//! Each passage is scored on three orthogonal signals, blended into a single
//! poison `risk`:
//!
//! | Signal | Meaning | Suspicious when |
//! |--------|---------|-----------------|
//! | **stuffing** | query-term token density in the passage | high |
//! | **diversity** | type/token ratio of the passage | low |
//! | **anomaly** | `1 - mean similarity` to the other passages | high |
//!
//! `risk = 0.4 * stuffing + 0.3 * (1 - diversity) + 0.3 * anomaly`. A passage is
//! flagged when `risk >= risk_threshold`, **or** when either hard signal trips
//! on its own (`stuffing >= stuffing_threshold` or
//! `diversity <= diversity_threshold`).
//!
//! All scoring is deterministic token / set arithmetic — no model, no I/O, no
//! randomness.
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "poisoning-defense")] {
//! use oxirag::poisoning_defense::{PoisonConfig, PoisoningDetector};
//!
//! let detector = PoisoningDetector::new(PoisonConfig::new());
//! let assessments = detector.scan("query text", &results);
//! let clean = detector.filter("query text", &results)?;
//! # }
//! ```

pub mod detector;
#[cfg(test)]
mod tests;
pub mod types;

pub use detector::PoisoningDetector;
pub use types::{PoisonAssessment, PoisonConfig, PoisonError};
