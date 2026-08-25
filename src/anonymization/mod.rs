//! PII anonymization / pseudonymization with a reversible mapping.
//!
//! Unlike the `guardrails` module — which *detects* and irreversibly *redacts*
//! PII — this module *pseudonymizes* it: each detected value is replaced with a
//! consistent placeholder (`[EMAIL_1]`, `[PHONE_1]`, `[PERSON_1]`, …) and the
//! association is recorded in a reversible [`AnonMapping`]. The original text
//! can be reconstructed exactly with [`Anonymizer::deanonymize`].
//!
//! Detection is fully deterministic and synchronous (no I/O, no regex, no
//! randomness): e-mails are found via `local@domain.tld` char-class scanning,
//! phone numbers via runs of digits and separators containing at least seven
//! digits, and person names via runs of capitalized words.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "anonymization")] {
//! use oxirag::anonymization::{Anonymizer, AnonConfig};
//!
//! let anon = Anonymizer::new(AnonConfig::default());
//! let result = anon.anonymize("Email Alice at alice@example.com or alice@example.com");
//! // The repeated e-mail reuses the same placeholder.
//! assert!(result.text.contains("[EMAIL_1]"));
//! assert!(!result.text.contains("[EMAIL_2]"));
//!
//! // The mapping is reversible.
//! let restored = anon.deanonymize(&result.text, &result.mapping);
//! assert_eq!(restored, "Email Alice at alice@example.com or alice@example.com");
//! # }
//! ```

pub mod anonymizer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use anonymizer::Anonymizer;
pub use types::{AnonConfig, AnonError, AnonMapping, AnonymizedText, PiiKind};
