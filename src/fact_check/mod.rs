//! FEVER-style fact verification for RAG claims.
//!
//! Given a claim and a corpus of evidence [`crate::types::Document`]s, the
//! [`FactChecker`] retrieves the most relevant evidence sentences and assigns a
//! three-way [`Verdict`] — `SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFO` — together
//! with a confidence and a human-readable rationale.
//!
//! The verdict is purely heuristic and deterministic:
//!
//! * Strong lexical overlap with **no** contradiction signal → [`Verdict::Supports`].
//! * Strong overlap **with** a contradiction signal (negation mismatch or
//!   number/quantity mismatch) → [`Verdict::Refutes`].
//! * Insufficient overlap → [`Verdict::NotEnoughInfo`].
//!
//! # Example
//!
//! ```
//! use oxirag::fact_check::{FactChecker, FactCheckConfig, Verdict};
//! use oxirag::types::Document;
//!
//! let checker = FactChecker::new(FactCheckConfig::default());
//! let docs = vec![Document::new("Marie Curie was born in 1867 in Warsaw.").with_id("d1")];
//! let result = checker.verify("Marie Curie was born in 1867.", &docs).unwrap();
//! assert_eq!(result.verdict, Verdict::Supports);
//! ```
pub mod checker;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use checker::FactChecker;
pub use types::{Evidence, FactCheckConfig, FactCheckError, FactCheckResult, Verdict};
