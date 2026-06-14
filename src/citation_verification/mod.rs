//! Citation Verification: per-citation grounding checks with a pure-Rust NLI-lite heuristic.
//!
//! This module verifies that each claim or quoted span in a generated answer is
//! *grounded* in (entailed by) its cited source passage. No ML model, network
//! access, or external dependency is involved — grounding is determined by a
//! two-signal lexical heuristic:
//!
//! | Signal | Description |
//! |--------|-------------|
//! | Token overlap | Fraction of claim tokens that appear in the source |
//! | N-gram substring | 1.0 if any 3-token span from the claim occurs verbatim in the token-normalised source |
//!
//! The signals are combined with [`CitationConfig`]-supplied weights and the
//! result is compared against a configurable threshold to yield a boolean
//! `is_grounded` verdict alongside the raw `grounding_score`.
//!
//! # Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`CitationConfig`] | Weights and the grounding threshold |
//! | [`CitationCheck`] | Input: one (claim, source-passage) pair |
//! | [`VerifiedCitation`] | Output: score, verdict, matched spans |
//! | [`CitationReport`] | Aggregate batch result |
//! | [`CitationVerifier`] | The verifier struct |
//! | [`CitationError`] | Error variants |
//!
//! # Example
//!
//! ```
//! use oxirag::citation_verification::{
//!     CitationCheck, CitationConfig, CitationVerifier,
//! };
//!
//! let verifier = CitationVerifier::new(CitationConfig::default());
//! let check = CitationCheck::new(
//!     "The Eiffel Tower is in Paris.",
//!     "The Eiffel Tower is located in Paris, France.",
//! );
//! let result = verifier.verify_one(check).expect("non-empty inputs");
//! assert!(result.is_grounded);
//! assert!(result.grounding_score > 0.5);
//! assert!(!result.matched_spans.is_empty());
//! ```

#[cfg(test)]
mod tests;
pub mod types;
pub mod verifier;

pub use types::{CitationCheck, CitationConfig, CitationError, CitationReport, VerifiedCitation};
pub use verifier::CitationVerifier;
