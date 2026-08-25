//! Verbatim quote grounding for generated answers.
//!
//! For each *claim* (a sentence of a generated answer) the grounder finds the
//! minimal **verbatim** supporting quote: the source sentence with maximal token
//! overlap against the claim, optionally trimmed to a window of at most
//! `QuoteConfig::max_quote_tokens` tokens centred on the overlapping span. A
//! claim whose best source sentence falls below `QuoteConfig::min_support` is
//! reported as *ungrounded*.
//!
//! # Distinction from `attribution`
//!
//! The `attribution` module aligns whole answer sentences to whole source
//! passages and emits inline citation markers. Quote grounding instead extracts
//! the single smallest verbatim span of source text that supports each claim, so
//! the returned `quote` is always a contiguous slice of some source document.
//!
//! # Components
//!
//! | Item | Responsibility |
//! |------|----------------|
//! | [`QuoteGrounder`] | Score claims against source sentences and extract quotes |
//! | [`QuoteConfig`] | Support threshold and maximum quote length |
//! | [`GroundedQuote`] | A claim paired with its verbatim supporting quote |
//! | [`QuoteError`] | Error variants for the grounding pipeline |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "quote-grounding")] {
//! use oxirag::prelude::*;
//!
//! let docs = vec![/* Document values */];
//! let grounder = QuoteGrounder::new(QuoteConfig::default());
//! let quotes = grounder.ground("Rust is memory safe.", &docs).unwrap();
//! for q in &quotes {
//!     println!("{} -> {}", q.claim, q.quote);
//! }
//! # }
//! ```

pub mod grounder;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use grounder::QuoteGrounder;
pub use types::{GroundedQuote, QuoteConfig, QuoteError};
