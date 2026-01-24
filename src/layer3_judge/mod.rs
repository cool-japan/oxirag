//! Layer 3: Judge - SMT-based logic verification.
//!
//! The Judge layer provides:
//! - Claim extraction from text
//! - SMT-LIB encoding of logical claims
//! - SMT-based verification using `OxiZ`
//! - Claim normalization and deduplication
//! - Explanation generation for verification results
//! - Incremental consistency checking
//! - Dependency parsing for improved claim extraction

pub mod claim_extractor;
pub mod dependency_parser;
pub mod explanation;
pub mod incremental;
pub mod normalizer;
pub mod oxiz_verifier;
pub mod traits;

pub use claim_extractor::AdvancedClaimExtractor;
pub use dependency_parser::{
    DependencyClaimExtractor, DependencyNode, DependencyParser, DependencyRelation, DependencyTree,
    PosTag, SimpleDependencyParser, SvoTriple,
};
pub use explanation::{ExplanationBuilder, generate_counterexample, generate_summary};
pub use incremental::{
    ClaimConflict, ConflictType, ConsistencyResult, IncrementalConsistencyChecker, Resolution,
};
pub use normalizer::{ClaimDeduplicator, ClaimNormalizer, DefaultClaimNormalizer};
pub use oxiz_verifier::{JudgeImpl, MockSmtVerifier};
pub use traits::{
    ClaimExtractor, ClaimPattern, Judge, JudgeConfig, PatternClaimExtractor, SmtVerifier,
};

#[cfg(feature = "judge")]
pub use oxiz_verifier::OxizVerifier;
