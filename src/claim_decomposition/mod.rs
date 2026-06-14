//! Claim Decomposition — FActScore-style atomic fact extraction.
//!
//! Decomposes a generated answer into **atomic claims**: single, self-contained,
//! verifiable facts. Following the `FActScore` approach of Min et al. (2023), an
//! answer is split into sentences, each sentence into clauses, and every clause
//! is *decontextualized* — a leading pronoun is replaced with the answer's main
//! subject so that each atom can be verified independently of its neighbours.
//!
//! This is distinct from the `hallucination_detector` module, which *scores*
//! claims for source support; here the focus is purely on **breaking an answer
//! apart** into the atomic units that such a verifier would consume.
//!
//! # Example
//!
//! ```
//! use oxirag::claim_decomposition::{AtomicClaimExtractor, ClaimDecompConfig};
//!
//! let extractor = AtomicClaimExtractor::new(ClaimDecompConfig::default());
//! let claims = extractor
//!     .decompose("Einstein was born in Germany and developed the theory of relativity.")
//!     .expect("non-empty answer");
//! assert!(claims.len() >= 2);
//! ```

pub mod extractor;
#[cfg(test)]
mod tests;
pub mod types;

pub use extractor::{AtomicClaimExtractor, ClaimExtractor, HeuristicAtomicExtractor};
pub use types::{AtomicClaim, ClaimDecompConfig, ClaimDecompError};
