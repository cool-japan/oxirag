//! The paging engine: [`MainContext`], [`ArchivalStore`], and the
//! [`ContextPager`] that moves [`MemoryPage`]s between them in response to
//! [`PagingAction`]s.

use std::collections::HashMap;

use crate::memory_paging::types::{
    MemoryPage, MemoryPagingConfig, MemoryPagingError, PagingAction, PagingOutcome, PagingPolicy,
};
use crate::types::DocumentId;

// ── embed / cosine helpers ───────────────────────────────────────────────────
//
// Deterministic FNV-1a bag-of-tokens pseudo-embedding, mirroring the pattern
// established by `long_term_memory` and `eigenscore`: a real embedding
// provider is out of scope for this module, but `PagingAction::Search` still
// needs *some* notion of semantic similarity to rank archival candidates.

const EMBED_OFFSET: u64 = 14_695_981_039_346_656_037;
const EMBED_PRIME: u64 = 1_099_511_628_211;

fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let tokens: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .collect();
    let mut buckets = vec![0.0_f32; dim];
    for tok in &tokens {
        #[allow(clippy::cast_possible_truncation)]
        let idx = {
            let mut h = EMBED_OFFSET;
            for b in tok.as_bytes() {
                h = h.wrapping_mul(EMBED_PRIME) ^ u64::from(*b);
            }
            (h % dim as u64) as usize
        };
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for b in &mut buckets {
            *b /= norm;
        }
    }
    buckets
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

// ── MainContext ──────────────────────────────────────────────────────────────

/// The bounded "physical RAM" tier: a small set of [`MemoryPage`]s directly
/// visible to the LLM right now.
///
/// Mutations that could violate the capacity invariant (inserting without
/// first making room) are restricted to `pub(crate)` so they can only be
/// reached through [`ContextPager`], which is responsible for evicting under
/// pressure *before* admitting a new page. Read access is fully public.
#[derive(Debug, Clone)]
pub struct MainContext {
    pages: HashMap<String, MemoryPage>,
    capacity: usize,
}

impl MainContext {
    /// Create a new, empty main context with the given capacity (in the same
    /// units as [`MemoryPage::size`]).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            pages: HashMap::new(),
            capacity,
        }
    }

    /// Total capacity of main context.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of pages currently resident.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Return true if no pages are resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Sum of [`MemoryPage::size`] across all resident pages.
    #[must_use]
    pub fn occupied_size(&self) -> usize {
        self.pages.values().map(|p| p.size).sum()
    }

    /// Remaining capacity (`capacity - occupied_size`, saturating at zero).
    #[must_use]
    pub fn available_size(&self) -> usize {
        self.capacity.saturating_sub(self.occupied_size())
    }

    /// Return true if `page_id` is currently resident.
    #[must_use]
    pub fn contains(&self, page_id: &str) -> bool {
        self.pages.contains_key(page_id)
    }

    /// Get a resident page by id.
    #[must_use]
    pub fn get(&self, page_id: &str) -> Option<&MemoryPage> {
        self.pages.get(page_id)
    }

    /// Iterate over all resident pages (arbitrary order).
    pub fn iter(&self) -> impl Iterator<Item = &MemoryPage> {
        self.pages.values()
    }

    /// Insert `page`, replacing any existing page with the same id, without
    /// checking or enforcing the capacity invariant. Only reachable from
    /// within this module (via [`ContextPager::admit`], which evicts first).
    pub(crate) fn insert_unchecked(&mut self, page: MemoryPage) -> Option<MemoryPage> {
        self.pages.insert(page.id.clone(), page)
    }

    /// Remove and return a resident page by id, if present. Removal can
    /// never violate the capacity invariant, but is kept `pub(crate)` so all
    /// main-context mutation flows through [`ContextPager`].
    pub(crate) fn remove(&mut self, page_id: &str) -> Option<MemoryPage> {
        self.pages.remove(page_id)
    }

    /// Refresh a resident page's recency bookkeeping. Returns `false` if
    /// `page_id` is not resident.
    pub(crate) fn touch(&mut self, page_id: &str, tick: u64) -> bool {
        if let Some(page) = self.pages.get_mut(page_id) {
            page.last_accessed_tick = tick;
            page.access_count = page.access_count.saturating_add(1);
            true
        } else {
            false
        }
    }
}

// ── ArchivalStore ────────────────────────────────────────────────────────────

/// The effectively unbounded "disk" tier: pages evicted from
/// [`MainContext`] (or pre-seeded directly, e.g. bulk-loading a prior
/// session's memory) that remain fully, losslessly recoverable.
#[derive(Debug, Clone, Default)]
pub struct ArchivalStore {
    pages: HashMap<String, MemoryPage>,
}

impl ArchivalStore {
    /// Number of pages currently archived.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Return true if no pages are archived.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Return true if `page_id` is currently archived.
    #[must_use]
    pub fn contains(&self, page_id: &str) -> bool {
        self.pages.contains_key(page_id)
    }

    /// Get an archived page by id.
    #[must_use]
    pub fn get(&self, page_id: &str) -> Option<&MemoryPage> {
        self.pages.get(page_id)
    }

    /// Iterate over all archived pages (arbitrary order).
    pub fn iter(&self) -> impl Iterator<Item = &MemoryPage> {
        self.pages.values()
    }

    /// Insert `page` into archival storage, replacing any existing page with
    /// the same id and returning it. Archival storage is unbounded, so this
    /// never fails or evicts anything.
    pub fn insert(&mut self, page: MemoryPage) -> Option<MemoryPage> {
        self.pages.insert(page.id.clone(), page)
    }

    /// Remove and return an archived page by id, if present.
    pub fn remove(&mut self, page_id: &str) -> Option<MemoryPage> {
        self.pages.remove(page_id)
    }
}

// ── ContextPager ─────────────────────────────────────────────────────────────

/// The paging engine. Executes [`PagingAction`]s against a [`MainContext`]
/// and [`ArchivalStore`] pair, evicting under pressure per
/// [`MemoryPagingConfig::policy`] whenever a page-in would otherwise exceed
/// [`MemoryPagingConfig::capacity`].
///
/// # Invariants
///
/// - A page id is resident in at most one of `main` / `archival` at a time;
///   every transition is a *move*, never a copy, so no page is ever
///   duplicated or silently dropped.
/// - `main.occupied_size() <= main.capacity()` holds before and after every
///   successful [`ContextPager::execute`] call.
#[derive(Debug, Clone)]
pub struct ContextPager {
    /// The bounded, LLM-visible tier.
    pub main: MainContext,
    /// The unbounded, evicted-but-recoverable tier.
    pub archival: ArchivalStore,
    /// Configuration (capacity, eviction policy, search parameters).
    pub config: MemoryPagingConfig,
    tick: u64,
}

impl ContextPager {
    /// Create a new pager with empty main context and archival storage.
    #[must_use]
    pub fn new(config: MemoryPagingConfig) -> Self {
        let capacity = config.capacity;
        Self {
            main: MainContext::new(capacity),
            archival: ArchivalStore::default(),
            config,
            tick: 0,
        }
    }

    /// Current value of the internal logical clock. Exposed read-only for
    /// diagnostics and tests; every page-in / touch stamps
    /// [`MemoryPage::last_accessed_tick`] with a fresh, strictly increasing
    /// value from this clock, which is what [`PagingPolicy::LeastRecentlyUsed`]
    /// orders by.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Build a [`MemoryPage`] from `content` using this pager's configured
    /// embedding dimension, without inserting it anywhere. Useful for
    /// pre-seeding [`ArchivalStore`] directly (e.g. bulk-loading a prior
    /// session's archived memory) with pages that carry a real, searchable
    /// embedding.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryPagingError::EmptyContent`] if `content` is empty
    /// (after trimming).
    pub fn make_page(
        &self,
        content: impl Into<String>,
        importance: f32,
    ) -> Result<MemoryPage, MemoryPagingError> {
        let content = content.into();
        if content.trim().is_empty() {
            return Err(MemoryPagingError::EmptyContent);
        }
        let id = DocumentId::new().as_str().to_string();
        let size = content.split_whitespace().count().max(1);
        let embedding = embed(&content, self.config.dim);
        Ok(MemoryPage {
            id,
            content,
            size,
            importance: importance.clamp(0.0, 1.0),
            created_tick: self.tick,
            last_accessed_tick: self.tick,
            access_count: 0,
            embedding,
        })
    }

    /// Execute a single self-directed [`PagingAction`] against this pager.
    ///
    /// # Errors
    ///
    /// - [`MemoryPagingError::EmptyContent`] — [`PagingAction::Write`] with
    ///   empty (or whitespace-only) content.
    /// - [`MemoryPagingError::EmptyQuery`] — [`PagingAction::Search`] with an
    ///   empty (or whitespace-only) query.
    /// - [`MemoryPagingError::PageNotInArchivalStore`] — [`PagingAction::PageIn`]
    ///   for a page id that is neither resident nor archived.
    /// - [`MemoryPagingError::PageNotInMainContext`] — [`PagingAction::PageOut`]
    ///   for a page id that is not resident.
    /// - [`MemoryPagingError::PageTooLargeForCapacity`] — the page being
    ///   paged in is larger than [`MemoryPagingConfig::capacity`], so it can
    ///   never fit even after evicting every other resident page.
    pub fn execute(&mut self, action: PagingAction) -> Result<PagingOutcome, MemoryPagingError> {
        match action {
            PagingAction::Write {
                content,
                importance,
            } => self.write(content, importance),
            PagingAction::PageIn { page_id } => self.page_in(&page_id),
            PagingAction::PageOut { page_id } => self.page_out(&page_id),
            PagingAction::Search { query } => self.search(&query),
        }
    }

    // ── PagingAction::Write ─────────────────────────────────────────────────

    fn write(
        &mut self,
        content: String,
        importance: f32,
    ) -> Result<PagingOutcome, MemoryPagingError> {
        let mut page = self.make_page(content, importance)?;
        self.tick += 1;
        page.created_tick = self.tick;
        page.last_accessed_tick = self.tick;
        self.admit(page)
    }

    // ── PagingAction::PageIn ────────────────────────────────────────────────

    fn page_in(&mut self, page_id: &str) -> Result<PagingOutcome, MemoryPagingError> {
        if self.main.contains(page_id) {
            self.tick += 1;
            self.main.touch(page_id, self.tick);
            return Ok(PagingOutcome {
                paged_in: Some(page_id.to_string()),
                paged_out: Vec::new(),
                search_matches: Vec::new(),
            });
        }

        // Peek the archived page's size *before* removing it, so a
        // too-large-for-capacity rejection never removes a page from
        // archival storage without a place to put it back — no data loss
        // on a failed page-in.
        let archived_size = self
            .archival
            .get(page_id)
            .map(|p| p.size)
            .ok_or_else(|| MemoryPagingError::PageNotInArchivalStore(page_id.to_string()))?;
        if archived_size > self.main.capacity() {
            return Err(MemoryPagingError::PageTooLargeForCapacity {
                page_id: page_id.to_string(),
                size: archived_size,
                capacity: self.main.capacity(),
            });
        }

        let mut page = self
            .archival
            .remove(page_id)
            .ok_or_else(|| MemoryPagingError::PageNotInArchivalStore(page_id.to_string()))?;
        self.tick += 1;
        page.last_accessed_tick = self.tick;
        page.access_count = page.access_count.saturating_add(1);
        self.admit(page)
    }

    // ── PagingAction::PageOut ───────────────────────────────────────────────

    fn page_out(&mut self, page_id: &str) -> Result<PagingOutcome, MemoryPagingError> {
        let page = self
            .main
            .remove(page_id)
            .ok_or_else(|| MemoryPagingError::PageNotInMainContext(page_id.to_string()))?;
        self.archival.insert(page);
        Ok(PagingOutcome {
            paged_in: None,
            paged_out: vec![page_id.to_string()],
            search_matches: Vec::new(),
        })
    }

    // ── PagingAction::Search ────────────────────────────────────────────────

    fn search(&mut self, query: &str) -> Result<PagingOutcome, MemoryPagingError> {
        if query.trim().is_empty() {
            return Err(MemoryPagingError::EmptyQuery);
        }
        let q_embedding = embed(query, self.config.dim);
        let mut scored: Vec<(String, f32)> = self
            .archival
            .iter()
            .map(|p| (p.id.clone(), cosine(&q_embedding, &p.embedding)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(self.config.search_top_k.max(1));

        if let Some((best_id, best_score)) = scored.first().cloned()
            && best_score >= self.config.min_search_similarity
        {
            let PagingOutcome {
                paged_in,
                paged_out,
                ..
            } = self.page_in(&best_id)?;
            return Ok(PagingOutcome {
                paged_in,
                paged_out,
                search_matches: scored,
            });
        }

        Ok(PagingOutcome {
            paged_in: None,
            paged_out: Vec::new(),
            search_matches: scored,
        })
    }

    // ── admission + eviction-under-pressure ─────────────────────────────────

    /// Admit `page` into main context, evicting resident pages to archival
    /// storage per [`MemoryPagingConfig::policy`] until there is room —
    /// **before** the page is inserted, never after.
    fn admit(&mut self, page: MemoryPage) -> Result<PagingOutcome, MemoryPagingError> {
        if page.size > self.main.capacity() {
            return Err(MemoryPagingError::PageTooLargeForCapacity {
                page_id: page.id.clone(),
                size: page.size,
                capacity: self.main.capacity(),
            });
        }

        let mut paged_out = Vec::new();
        while self.main.occupied_size() + page.size > self.main.capacity() {
            let victim_id = self
                .select_victim()
                .ok_or_else(|| MemoryPagingError::NoEvictionCandidate(page.id.clone()))?;
            let victim = self
                .main
                .remove(&victim_id)
                .ok_or_else(|| MemoryPagingError::NoEvictionCandidate(victim_id.clone()))?;
            self.archival.insert(victim);
            paged_out.push(victim_id);
        }

        let id = page.id.clone();
        self.main.insert_unchecked(page);
        Ok(PagingOutcome {
            paged_in: Some(id),
            paged_out,
            search_matches: Vec::new(),
        })
    }

    /// Select the id of the resident page [`MemoryPagingConfig::policy`]
    /// would evict next, or `None` if main context is empty.
    fn select_victim(&self) -> Option<String> {
        match self.config.policy {
            PagingPolicy::LeastRecentlyUsed => self
                .main
                .iter()
                .min_by(|a, b| {
                    a.last_accessed_tick
                        .cmp(&b.last_accessed_tick)
                        .then_with(|| a.id.cmp(&b.id))
                })
                .map(|p| p.id.clone()),
            PagingPolicy::LeastImportant => self
                .main
                .iter()
                .min_by(|a, b| {
                    a.importance
                        .partial_cmp(&b.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.last_accessed_tick.cmp(&b.last_accessed_tick))
                        .then_with(|| a.id.cmp(&b.id))
                })
                .map(|p| p.id.clone()),
        }
    }
}

impl Default for ContextPager {
    fn default() -> Self {
        Self::new(MemoryPagingConfig::default())
    }
}
