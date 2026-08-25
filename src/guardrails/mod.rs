//! PII detection, prompt injection blocking, content moderation, and topical rails.
//!
//! Implements a NeMo-Guardrails-style safety layer that runs fully synchronously
//! (no I/O) over arbitrary text.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`PiiDetector`] | Char-class state-machine email/phone/SSN/CC/IP scanner |
//! | [`InjectionDetector`] | Phrase-signal injection/jailbreak detector |
//! | [`ContentModerator`] | Toxic-word list scoring |
//! | [`GuardrailEngine`] | Orchestrates all checks in one pass |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "guardrails")] {
//! use oxirag::prelude::*;
//!
//! let engine = GuardrailEngine::new();
//! let report = engine.check("my email is test@example.com", &GuardrailConfig::default()).unwrap();
//! assert!(report.blocked);
//! # }
//! ```

pub mod engine;
pub mod injection;
pub mod moderation;
pub mod pii;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::GuardrailEngine;
pub use injection::InjectionDetector;
pub use moderation::ContentModerator;
pub use pii::PiiDetector;
pub use types::{
    GuardrailConfig, GuardrailError, GuardrailReport, PiiKind, PiiMatch, Severity, TopicalRail,
    Violation,
};
