//! `MemGPT`-style virtual context management (Packer et al. 2023, *"`MemGPT`:
//! Towards LLMs as Operating Systems"*).
//!
//! This module models an LLM's context window the way an operating system
//! models RAM: a small, bounded set of pages the CPU (the LLM) can see right
//! now, backed by effectively unbounded disk it cannot see but can explicitly
//! swap pages into and out of.
//!
//! # Architecture
//!
//! - [`pager::MainContext`] — the bounded "physical RAM" tier: a small
//!   number of active [`types::MemoryPage`]s directly visible to the LLM,
//!   tracked against a configurable [`types::MemoryPagingConfig::capacity`].
//! - [`pager::ArchivalStore`] — the "disk" tier: effectively unbounded
//!   storage for pages evicted from main context. Nothing is ever discarded
//!   here, only made invisible.
//! - [`types::PagingAction`] — the function-call-style actions `MemGPT`
//!   describes an LLM issuing to manage its own memory: `Write` (append new
//!   content), `PageIn` (recall a specific archived page), `PageOut`
//!   (explicitly evict a resident page), and `Search` (semantic search over
//!   archival storage, paging in the best match).
//! - [`pager::ContextPager`] — the engine that executes
//!   [`types::PagingAction`]s against a `MainContext` / `ArchivalStore` pair.
//!
//! # Self-directed eviction under pressure
//!
//! The central mechanic: whenever a page-in ([`types::PagingAction::Write`],
//! [`types::PagingAction::PageIn`], or a matching
//! [`types::PagingAction::Search`]) would push main context's occupied size
//! past its capacity, [`pager::ContextPager`] automatically evicts resident
//! pages to archival storage — selected by the configured
//! [`types::PagingPolicy`] — **before** completing the page-in. Eviction only
//! ever moves a page (never copies or drops it), so every evicted page
//! remains fully, losslessly recoverable via a later `PageIn` or `Search`.
//!
//! # Example
//!
//! ```
//! use oxirag::memory_paging::{ContextPager, MemoryPagingConfig, PagingAction, PagingPolicy};
//!
//! // A tiny main context that can hold at most 8 whitespace-token "units".
//! let mut pager = ContextPager::new(
//!     MemoryPagingConfig::default()
//!         .with_capacity(8)
//!         .with_policy(PagingPolicy::LeastRecentlyUsed),
//! );
//!
//! // First page: "alpha beta" (2 tokens) fits with room to spare.
//! let first = pager
//!     .execute(PagingAction::Write { content: "alpha beta".into(), importance: 0.5 })
//!     .expect("write should succeed")
//!     .paged_in
//!     .expect("write always pages in");
//!
//! // Second page: "gamma delta epsilon" (3 tokens) still fits (2 + 3 = 5 <= 8).
//! pager
//!     .execute(PagingAction::Write { content: "gamma delta epsilon".into(), importance: 0.5 })
//!     .expect("write should succeed");
//!
//! // Third page: "zeta eta theta iota" (4 tokens) does NOT fit alongside the
//! // other two (2 + 3 + 4 = 9 > 8): the pager must evict under pressure
//! // BEFORE completing this page-in. LRU means the oldest page ("alpha
//! // beta", never touched again) is evicted first — and evicting just that
//! // one page is enough to make room (5 - 2 + 4 = 7 <= 8), so the second
//! // page stays resident.
//! let outcome = pager
//!     .execute(PagingAction::Write { content: "zeta eta theta iota".into(), importance: 0.5 })
//!     .expect("write should succeed");
//! assert_eq!(outcome.paged_out, vec![first.clone()]);
//! assert!(!pager.main.contains(&first));
//! assert!(pager.archival.contains(&first));
//!
//! // The evicted page is not lost — it round-trips losslessly back into
//! // main context on demand.
//! let paged_back_in = pager
//!     .execute(PagingAction::PageIn { page_id: first.clone() })
//!     .expect("page-in should succeed");
//! assert_eq!(paged_back_in.paged_in, Some(first));
//! ```

pub mod pager;
#[cfg(test)]
mod tests;
pub mod types;

pub use pager::{ArchivalStore, ContextPager, MainContext};
pub use types::{
    MemoryPage, MemoryPagingConfig, MemoryPagingError, PagingAction, PagingOutcome, PagingPolicy,
};
