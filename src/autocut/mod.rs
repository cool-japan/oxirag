//! Autocut — relevance-gap result truncation (Weaviate-style autocut).
//!
//! Dynamically decides how many leading results to keep by detecting
//! discontinuities ("jumps") in the descending score sequence, instead of
//! returning a fixed `top_k`. Several strategies are available via
//! [`AutoCutStrategy`]; the keep count is always clamped into a configurable
//! `[min_keep, max_keep]` window.
pub mod cutter;
#[cfg(test)]
mod tests;
pub mod types;
pub use cutter::AutoCutter;
pub use types::{AutoCutConfig, AutoCutError, AutoCutReport, AutoCutStrategy};
