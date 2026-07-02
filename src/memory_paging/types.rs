//! Types for the `memory_paging` module.
use thiserror::Error;

// ── MemoryPage ───────────────────────────────────────────────────────────────

/// A single unit of virtual memory: either resident in [`crate::memory_paging::pager::MainContext`]
/// (visible to the LLM right now) or swapped out to
/// [`crate::memory_paging::pager::ArchivalStore`] (recoverable, but not visible).
///
/// Eviction only ever *moves* a page between the two tiers — the content,
/// embedding, and bookkeeping fields are preserved bit-for-bit, so a page is
/// always losslessly recoverable via [`PagingAction::PageIn`] or
/// [`PagingAction::Search`].
///
/// Recency is tracked via a logical *tick* counter (owned by
/// [`crate::memory_paging::pager::ContextPager`]) rather than a wall-clock
/// timestamp. This keeps least-recently-used eviction ordering exactly
/// reproducible in tests (no `sleep`s, no clock-resolution ties) while still
/// giving a strict, monotonically increasing notion of "how recently was this
/// page touched".
#[derive(Debug, Clone)]
pub struct MemoryPage {
    /// Unique identifier for this page.
    pub id: String,
    /// The page's textual content (verbatim; never summarised or truncated by
    /// this module).
    pub content: String,
    /// Occupied size of this page in the same units as
    /// [`MemoryPagingConfig::capacity`] (whitespace-token count of `content`).
    pub size: usize,
    /// Importance score in `[0.0, 1.0]`, used by
    /// [`PagingPolicy::LeastImportant`] to rank eviction candidates.
    pub importance: f32,
    /// Logical tick at which this page was first created.
    pub created_tick: u64,
    /// Logical tick at which this page was last made resident/touched in
    /// main context. Used by [`PagingPolicy::LeastRecentlyUsed`].
    pub last_accessed_tick: u64,
    /// Number of times this page has been paged in (or touched while
    /// already resident).
    pub access_count: u32,
    /// Deterministic FNV-1a bag-of-tokens pseudo-embedding of `content`,
    /// used by [`PagingAction::Search`] to rank archival candidates via
    /// cosine similarity.
    pub embedding: Vec<f32>,
}

// ── PagingPolicy ─────────────────────────────────────────────────────────────

/// Selects which resident page(s) [`crate::memory_paging::pager::ContextPager`]
/// evicts to [`crate::memory_paging::pager::ArchivalStore`] when a page-in
/// would exceed [`MemoryPagingConfig::capacity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PagingPolicy {
    /// Evict the page with the oldest [`MemoryPage::last_accessed_tick`]
    /// (classic OS-style LRU). Ties are broken by page id for determinism.
    #[default]
    LeastRecentlyUsed,
    /// Evict the page with the lowest [`MemoryPage::importance`] (a nod to
    /// `long_term_memory`'s importance-weighted retention, applied here to
    /// *eviction* rather than retrieval ranking). Ties are broken by oldest
    /// [`MemoryPage::last_accessed_tick`], then by page id.
    LeastImportant,
}

// ── PagingAction ─────────────────────────────────────────────────────────────

/// A self-directed, function-call-style memory-management action that an
/// agent (the LLM itself, per the `MemGPT` paper) can issue against a
/// [`crate::memory_paging::pager::ContextPager`].
#[derive(Debug, Clone)]
pub enum PagingAction {
    /// Write new content directly into main context as a fresh, resident
    /// page (`MemGPT`'s `core_memory_append`-style self-edit). May trigger
    /// eviction under pressure if main context is at or near capacity.
    Write {
        /// The page's textual content. Must not be empty (after trimming).
        content: String,
        /// Importance score in `[0.0, 1.0]`, used by
        /// [`PagingPolicy::LeastImportant`]. Out-of-range values are clamped.
        importance: f32,
    },
    /// Page a specific, already-archived page back into main context by id
    /// (`MemGPT`'s `archival_memory_load`-style recall). May trigger
    /// eviction under pressure. A no-op "touch" (refreshes recency) if the
    /// page is already resident.
    PageIn {
        /// Id of the page to bring into main context.
        page_id: String,
    },
    /// Explicitly evict a specific resident page from main context to
    /// archival storage. Never fails due to capacity — archival storage is
    /// effectively unbounded.
    PageOut {
        /// Id of the page to evict from main context.
        page_id: String,
    },
    /// Search archival storage for the best semantic match to `query`
    /// (`MemGPT`'s `archival_memory_search`) and page the best match into
    /// main context, provided it clears
    /// [`MemoryPagingConfig::min_search_similarity`].
    Search {
        /// Free-text search query.
        query: String,
    },
}

// ── PagingOutcome ────────────────────────────────────────────────────────────

/// The effect a single [`PagingAction`] had on
/// [`crate::memory_paging::pager::MainContext`] /
/// [`crate::memory_paging::pager::ArchivalStore`].
#[derive(Debug, Clone, Default)]
pub struct PagingOutcome {
    /// Id of the page that is now resident in main context as a direct
    /// result of this action. `None` for [`PagingAction::PageOut`], and for
    /// a [`PagingAction::Search`] whose best match did not clear the
    /// similarity threshold (or archival storage was empty).
    pub paged_in: Option<String>,
    /// Ids of every page moved from main context to archival storage during
    /// this action — the explicit [`PagingAction::PageOut`] target, plus any
    /// pages evicted under capacity pressure to make room for a page-in.
    pub paged_out: Vec<String>,
    /// For [`PagingAction::Search`]: up to
    /// [`MemoryPagingConfig::search_top_k`] archival candidates considered,
    /// most relevant first, as `(page_id, cosine_similarity)` pairs. Empty
    /// for all other actions.
    pub search_matches: Vec<(String, f32)>,
}

// ── MemoryPagingConfig ───────────────────────────────────────────────────────

/// Configuration for [`crate::memory_paging::pager::ContextPager`].
#[derive(Debug, Clone)]
pub struct MemoryPagingConfig {
    /// Maximum total occupied size (whitespace-token units, summed across
    /// resident pages) of main context. Defaults to `128`.
    pub capacity: usize,
    /// Eviction policy applied when a page-in would exceed `capacity`.
    /// Defaults to [`PagingPolicy::LeastRecentlyUsed`].
    pub policy: PagingPolicy,
    /// Embedding dimension for FNV-1a pseudo-embeddings used by
    /// [`PagingAction::Search`]. Defaults to `64`.
    pub dim: usize,
    /// Maximum number of archival candidates reported in
    /// [`PagingOutcome::search_matches`]. Defaults to `5`.
    pub search_top_k: usize,
    /// Minimum cosine similarity a [`PagingAction::Search`] best match must
    /// clear before it is paged in. Defaults to `0.0` (always page in the
    /// best available match, if any).
    pub min_search_similarity: f32,
}

impl Default for MemoryPagingConfig {
    fn default() -> Self {
        Self {
            capacity: 128,
            policy: PagingPolicy::LeastRecentlyUsed,
            dim: 64,
            search_top_k: 5,
            min_search_similarity: 0.0,
        }
    }
}

impl MemoryPagingConfig {
    /// Set the main context capacity.
    #[must_use]
    pub fn with_capacity(mut self, v: usize) -> Self {
        self.capacity = v;
        self
    }
    /// Set the eviction policy.
    #[must_use]
    pub fn with_policy(mut self, v: PagingPolicy) -> Self {
        self.policy = v;
        self
    }
    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }
    /// Set the maximum number of reported search candidates.
    #[must_use]
    pub fn with_search_top_k(mut self, v: usize) -> Self {
        self.search_top_k = v;
        self
    }
    /// Set the minimum search similarity threshold.
    #[must_use]
    pub fn with_min_search_similarity(mut self, v: f32) -> Self {
        self.min_search_similarity = v;
        self
    }
}

// ── MemoryPagingError ────────────────────────────────────────────────────────

/// Errors from the `memory_paging` module.
#[derive(Debug, Error)]
pub enum MemoryPagingError {
    /// The content string supplied to [`PagingAction::Write`] was empty.
    #[error("Page content must not be empty")]
    EmptyContent,
    /// A page id supplied to [`PagingAction::PageOut`] was not resident in
    /// main context.
    #[error("Page {0} is not resident in main context")]
    PageNotInMainContext(String),
    /// A page id supplied to [`PagingAction::PageIn`] was not found in
    /// archival storage (and was not already resident in main context).
    #[error("Page {0} was not found in archival storage")]
    PageNotInArchivalStore(String),
    /// A page's size exceeds main context capacity, so it can never be
    /// paged in even after evicting every other resident page.
    #[error("page {page_id} (size {size}) exceeds main context capacity {capacity}")]
    PageTooLargeForCapacity {
        /// Id of the offending page.
        page_id: String,
        /// The page's occupied size.
        size: usize,
        /// Main context's total capacity.
        capacity: usize,
    },
    /// The search query text supplied to [`PagingAction::Search`] was empty.
    #[error("Search query must not be empty")]
    EmptyQuery,
    /// Internal invariant violation: eviction was required to page in
    /// `page_id`, but main context had no evictable page. This should be
    /// unreachable in practice (every stored page has size `>= 1`, so a
    /// capacity shortfall always implies a non-empty main context) and
    /// exists only as a defensive guard against future refactors.
    #[error("eviction required to page in {0} but main context has no evictable pages")]
    NoEvictionCandidate(String),
}
