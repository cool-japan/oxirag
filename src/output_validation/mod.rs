//! Rule-based validation of generated answers.
//!
//! Checks answers against configurable rules (length, citation presence,
//! banned phrases, JSON parsability, repetition ratio) and produces a
//! [`ValidationReport`].
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`OutputValidator`] | Applies rules and builds the report |
//! | [`ValidationRule`] | Rule definition with severity |
//! | [`ValidationReport`] | Pass/fail + violation list |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "output-validation")] {
//! use oxirag::prelude::*;
//!
//! let validator = OutputValidator::new();
//! let rules = vec![ValidationRule::new(RuleKind::MinLength(50))];
//! # }
//! ```

#[cfg(test)]
mod tests;
pub mod types;
pub mod validator;

pub use types::{
    OutputValidationError, RuleKind, RuleSeverity, RuleViolation, ValidationConfig,
    ValidationReport, ValidationRule,
};
pub use validator::OutputValidator;
