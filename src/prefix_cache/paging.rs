//! Paged cache system inspired by `PagedAttention`.
//!
//! This module implements a paging system for KV cache that enables more
//! efficient memory utilization by dividing cache entries into fixed-size
//! pages that can be allocated and deallocated independently.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::types::ContextFingerprint;

/// A page of KV cache data.
///
/// Each page represents a fixed-size block of cache memory that can be
/// independently allocated and freed. Pages are the basic unit of memory
/// management in the paged cache system.
#[derive(Debug, Clone)]
pub struct CachePage {
    /// Unique identifier for this page.
    pub page_id: u64,
    /// The actual cache data stored in this page.
    pub data: Vec<f32>,
    /// Size of this page in number of elements.
    pub page_size: usize,
    /// Whether this page is currently allocated.
    pub is_allocated: bool,
    /// Reference count for copy-on-write semantics.
    pub ref_count: u32,
    /// When this page was last accessed.
    pub last_accessed: Instant,
    /// Whether this page has been modified since allocation.
    pub is_dirty: bool,
}

impl CachePage {
    /// Create a new cache page with the given ID and size.
    #[must_use]
    pub fn new(page_id: u64, page_size: usize) -> Self {
        Self {
            page_id,
            data: vec![0.0; page_size],
            page_size,
            is_allocated: false,
            ref_count: 0,
            last_accessed: Instant::now(),
            is_dirty: false,
        }
    }

    /// Create a new cache page with pre-filled data.
    #[must_use]
    pub fn with_data(page_id: u64, data: Vec<f32>) -> Self {
        let page_size = data.len();
        Self {
            page_id,
            data,
            page_size,
            is_allocated: true,
            ref_count: 1,
            last_accessed: Instant::now(),
            is_dirty: false,
        }
    }

    /// Mark this page as accessed.
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }

    /// Increment the reference count.
    pub fn add_ref(&mut self) {
        self.ref_count = self.ref_count.saturating_add(1);
    }

    /// Decrement the reference count. Returns true if the page can be freed.
    pub fn release(&mut self) -> bool {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count == 0
    }

    /// Get the memory size of this page in bytes.
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Clear the page data and reset state.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
        self.is_allocated = false;
        self.ref_count = 0;
        self.is_dirty = false;
    }

    /// Copy data into this page starting at the given offset.
    ///
    /// # Panics
    ///
    /// Panics if the data would overflow the page.
    pub fn write(&mut self, offset: usize, data: &[f32]) {
        assert!(
            offset + data.len() <= self.page_size,
            "Data would overflow page: offset={offset}, data_len={}, page_size={}",
            data.len(),
            self.page_size
        );
        self.data[offset..offset + data.len()].copy_from_slice(data);
        self.is_dirty = true;
        self.touch();
    }

    /// Read data from this page starting at the given offset.
    #[must_use]
    pub fn read(&self, offset: usize, len: usize) -> &[f32] {
        let end = (offset + len).min(self.data.len());
        &self.data[offset..end]
    }
}

/// Page table for managing cache pages.
///
/// The page table is responsible for allocating, freeing, and tracking
/// all cache pages. It maintains a pool of pages and implements memory
/// management policies like defragmentation.
#[derive(Debug)]
pub struct PageTable {
    /// All pages managed by this table.
    pages: HashMap<u64, CachePage>,
    /// List of free page IDs available for allocation.
    free_pages: Vec<u64>,
    /// Size of each page in elements.
    page_size: usize,
    /// Maximum number of pages.
    max_pages: usize,
    /// Next page ID to assign.
    next_page_id: u64,
    /// Total pages currently allocated.
    allocated_count: usize,
}

impl PageTable {
    /// Create a new page table with the given page size and maximum pages.
    #[must_use]
    pub fn new(page_size: usize, max_pages: usize) -> Self {
        Self {
            pages: HashMap::with_capacity(max_pages),
            free_pages: Vec::with_capacity(max_pages),
            page_size,
            max_pages,
            next_page_id: 0,
            allocated_count: 0,
        }
    }

    /// Get the page size.
    #[must_use]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Get the maximum number of pages.
    #[must_use]
    pub fn max_pages(&self) -> usize {
        self.max_pages
    }

    /// Get the number of currently allocated pages.
    #[must_use]
    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    /// Get the number of free pages available.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free_pages.len() + (self.max_pages.saturating_sub(self.pages.len()))
    }

    /// Calculate total memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.pages
            .values()
            .filter(|p| p.is_allocated)
            .map(CachePage::memory_size)
            .sum()
    }

    /// Allocate a new page. Returns the page ID if successful.
    pub fn allocate_page(&mut self) -> Option<u64> {
        // First try to reuse a freed page
        if let Some(page_id) = self.free_pages.pop()
            && let Some(page) = self.pages.get_mut(&page_id)
        {
            page.is_allocated = true;
            page.ref_count = 1;
            page.touch();
            self.allocated_count += 1;
            return Some(page_id);
        }

        // Check if we can create a new page
        if self.pages.len() >= self.max_pages {
            return None;
        }

        // Create a new page
        let page_id = self.next_page_id;
        self.next_page_id += 1;

        let mut page = CachePage::new(page_id, self.page_size);
        page.is_allocated = true;
        page.ref_count = 1;

        self.pages.insert(page_id, page);
        self.allocated_count += 1;

        Some(page_id)
    }

    /// Allocate a page with initial data. Returns the page ID if successful.
    pub fn allocate_page_with_data(&mut self, data: Vec<f32>) -> Option<u64> {
        // Check size matches page size
        if data.len() != self.page_size {
            return None;
        }

        // Check capacity
        if self.pages.len() >= self.max_pages && self.free_pages.is_empty() {
            return None;
        }

        // Try to reuse a freed page first
        if let Some(page_id) = self.free_pages.pop()
            && let Some(page) = self.pages.get_mut(&page_id)
        {
            page.data = data;
            page.is_allocated = true;
            page.ref_count = 1;
            page.is_dirty = true;
            page.touch();
            self.allocated_count += 1;
            return Some(page_id);
        }

        // Create a new page with the data
        let page_id = self.next_page_id;
        self.next_page_id += 1;

        let page = CachePage::with_data(page_id, data);
        self.pages.insert(page_id, page);
        self.allocated_count += 1;

        Some(page_id)
    }

    /// Free a page by its ID.
    pub fn free_page(&mut self, page_id: u64) {
        if let Some(page) = self.pages.get_mut(&page_id)
            && page.is_allocated
        {
            let can_free = page.release();
            if can_free {
                page.clear();
                self.free_pages.push(page_id);
                self.allocated_count = self.allocated_count.saturating_sub(1);
            }
        }
    }

    /// Force free a page regardless of reference count.
    pub fn force_free_page(&mut self, page_id: u64) {
        if let Some(page) = self.pages.get_mut(&page_id)
            && page.is_allocated
        {
            page.clear();
            self.free_pages.push(page_id);
            self.allocated_count = self.allocated_count.saturating_sub(1);
        }
    }

    /// Get an immutable reference to a page.
    #[must_use]
    pub fn get_page(&self, page_id: u64) -> Option<&CachePage> {
        self.pages.get(&page_id).filter(|p| p.is_allocated)
    }

    /// Get a mutable reference to a page.
    pub fn get_page_mut(&mut self, page_id: u64) -> Option<&mut CachePage> {
        self.pages.get_mut(&page_id).filter(|p| p.is_allocated)
    }

    /// Defragment the page table by removing unallocated pages and compacting.
    ///
    /// Returns the number of pages removed.
    pub fn defragment(&mut self) -> usize {
        // Remove pages that have been freed and have no references
        let pages_to_remove: Vec<u64> = self
            .pages
            .iter()
            .filter(|(_, page)| !page.is_allocated && page.ref_count == 0)
            .map(|(id, _)| *id)
            .collect();

        let removed_count = pages_to_remove.len();

        for page_id in pages_to_remove {
            self.pages.remove(&page_id);
            self.free_pages.retain(|&id| id != page_id);
        }

        removed_count
    }

    /// Evict the least recently used page. Returns the evicted page ID if successful.
    pub fn evict_lru(&mut self) -> Option<u64> {
        let lru_page_id = self
            .pages
            .iter()
            .filter(|(_, page)| page.is_allocated && page.ref_count <= 1)
            .min_by_key(|(_, page)| page.last_accessed)
            .map(|(id, _)| *id);

        if let Some(page_id) = lru_page_id {
            self.force_free_page(page_id);
            Some(page_id)
        } else {
            None
        }
    }

    /// Get all allocated page IDs.
    #[must_use]
    pub fn allocated_page_ids(&self) -> Vec<u64> {
        self.pages
            .iter()
            .filter(|(_, page)| page.is_allocated)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Increment reference count for a page.
    pub fn add_page_ref(&mut self, page_id: u64) {
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.add_ref();
        }
    }

    /// Clear all pages.
    pub fn clear(&mut self) {
        for page in self.pages.values_mut() {
            page.clear();
        }
        self.free_pages.clear();
        self.free_pages.extend(self.pages.keys().copied());
        self.allocated_count = 0;
    }
}

/// A paged KV cache entry that spans multiple pages.
///
/// This structure represents a single cache entry that may be stored
/// across multiple pages in the page table. It tracks which pages
/// contain its data and provides methods for reading and writing.
#[derive(Debug, Clone)]
pub struct PagedKVEntry {
    /// Fingerprint identifying the cached context.
    pub fingerprint: ContextFingerprint,
    /// IDs of pages containing this entry's data, in order.
    pub page_ids: Vec<u64>,
    /// Total number of tokens in this entry.
    pub total_tokens: usize,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
}

impl PagedKVEntry {
    /// Create a new paged KV entry.
    #[must_use]
    pub fn new(fingerprint: ContextFingerprint, page_ids: Vec<u64>, total_tokens: usize) -> Self {
        let now = Instant::now();
        Self {
            fingerprint,
            page_ids,
            total_tokens,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    /// Record an access to this entry.
    pub fn record_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Get the number of pages this entry spans.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.page_ids.len()
    }

    /// Get the age of this entry.
    #[must_use]
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Get time since last access.
    #[must_use]
    pub fn time_since_access(&self) -> std::time::Duration {
        self.last_accessed.elapsed()
    }
}

/// A complete paged cache system combining page table and entry management.
///
/// This provides a high-level interface for storing and retrieving KV cache
/// entries using the paged memory system.
#[derive(Debug)]
pub struct PagedCache {
    /// The underlying page table.
    page_table: Arc<RwLock<PageTable>>,
    /// Mapping from fingerprint hash to paged entry.
    entries: Arc<RwLock<HashMap<u64, PagedKVEntry>>>,
    /// Page size in elements.
    page_size: usize,
}

impl PagedCache {
    /// Create a new paged cache.
    #[must_use]
    pub fn new(page_size: usize, max_pages: usize) -> Self {
        Self {
            page_table: Arc::new(RwLock::new(PageTable::new(page_size, max_pages))),
            entries: Arc::new(RwLock::new(HashMap::new())),
            page_size,
        }
    }

    /// Get the page size.
    #[must_use]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Store data in the paged cache.
    ///
    /// # Panics
    ///
    /// Panics if any internal lock is poisoned.
    ///
    /// # Returns
    ///
    /// Returns `None` if there isn't enough space to store the data.
    #[must_use]
    pub fn put(&self, fingerprint: &ContextFingerprint, data: &[f32]) -> Option<()> {
        let num_pages = data.len().div_ceil(self.page_size);

        let mut page_table = self.page_table.write().expect("lock poisoned");
        let mut page_ids = Vec::with_capacity(num_pages);

        // Allocate pages
        for i in 0..num_pages {
            let start = i * self.page_size;
            let end = ((i + 1) * self.page_size).min(data.len());
            let chunk = &data[start..end];

            // Pad to page size if needed
            let mut page_data = vec![0.0; self.page_size];
            page_data[..chunk.len()].copy_from_slice(chunk);

            let page_id = page_table.allocate_page_with_data(page_data)?;
            page_ids.push(page_id);
        }

        drop(page_table);

        // Create entry
        let entry = PagedKVEntry::new(fingerprint.clone(), page_ids, data.len());

        let mut entries = self.entries.write().expect("lock poisoned");

        // Remove old entry with same fingerprint if exists
        if let Some(old_entry) = entries.remove(&fingerprint.hash) {
            let mut page_table = self.page_table.write().expect("lock poisoned");
            for page_id in old_entry.page_ids {
                page_table.free_page(page_id);
            }
        }

        entries.insert(fingerprint.hash, entry);
        Some(())
    }

    /// Retrieve data from the paged cache.
    ///
    /// # Panics
    ///
    /// Panics if any internal lock is poisoned.
    #[must_use]
    pub fn get(&self, fingerprint: &ContextFingerprint) -> Option<Vec<f32>> {
        let mut entries = self.entries.write().expect("lock poisoned");
        let entry = entries.get_mut(&fingerprint.hash)?;
        entry.record_access();

        let page_table = self.page_table.read().expect("lock poisoned");
        let mut result = Vec::with_capacity(entry.total_tokens);

        for &page_id in &entry.page_ids {
            if let Some(page) = page_table.get_page(page_id) {
                result.extend_from_slice(&page.data);
            } else {
                return None;
            }
        }

        result.truncate(entry.total_tokens);
        Some(result)
    }

    /// Remove an entry from the cache.
    ///
    /// # Panics
    ///
    /// Panics if any internal lock is poisoned.
    #[must_use]
    pub fn remove(&self, fingerprint: &ContextFingerprint) -> Option<PagedKVEntry> {
        let mut entries = self.entries.write().expect("lock poisoned");
        let entry = entries.remove(&fingerprint.hash)?;

        let mut page_table = self.page_table.write().expect("lock poisoned");
        for page_id in &entry.page_ids {
            page_table.free_page(*page_id);
        }

        Some(entry)
    }

    /// Check if an entry exists.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn contains(&self, fingerprint: &ContextFingerprint) -> bool {
        let entries = self.entries.read().expect("lock poisoned");
        entries.contains_key(&fingerprint.hash)
    }

    /// Get the number of entries in the cache.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        let entries = self.entries.read().expect("lock poisoned");
        entries.len()
    }

    /// Get total memory usage in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let page_table = self.page_table.read().expect("lock poisoned");
        page_table.memory_usage()
    }

    /// Clear all entries and pages.
    ///
    /// # Panics
    ///
    /// Panics if any internal lock is poisoned.
    pub fn clear(&self) {
        let mut entries = self.entries.write().expect("lock poisoned");
        let mut page_table = self.page_table.write().expect("lock poisoned");

        entries.clear();
        page_table.clear();
    }

    /// Defragment the underlying page table.
    ///
    /// Returns the number of pages removed.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn defragment(&self) -> usize {
        let mut page_table = self.page_table.write().expect("lock poisoned");
        page_table.defragment()
    }

    /// Evict entries to free up space.
    ///
    /// Returns the number of entries evicted.
    ///
    /// # Panics
    ///
    /// Panics if any internal lock is poisoned.
    #[must_use]
    pub fn evict_lru_entries(&self, count: usize) -> usize {
        let mut entries = self.entries.write().expect("lock poisoned");

        // Find LRU entries
        let mut entry_times: Vec<_> = entries
            .iter()
            .map(|(hash, entry)| (*hash, entry.last_accessed))
            .collect();
        entry_times.sort_by_key(|(_, time)| *time);

        let mut evicted = 0;
        let mut page_table = self.page_table.write().expect("lock poisoned");

        for (hash, _) in entry_times.into_iter().take(count) {
            if let Some(entry) = entries.remove(&hash) {
                for page_id in entry.page_ids {
                    page_table.free_page(page_id);
                }
                evicted += 1;
            }
        }

        evicted
    }
}

impl Clone for PagedCache {
    fn clone(&self) -> Self {
        Self {
            page_table: Arc::clone(&self.page_table),
            entries: Arc::clone(&self.entries),
            page_size: self.page_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_page_new() {
        let page = CachePage::new(1, 256);
        assert_eq!(page.page_id, 1);
        assert_eq!(page.page_size, 256);
        assert!(!page.is_allocated);
        assert_eq!(page.ref_count, 0);
    }

    #[test]
    fn test_cache_page_with_data() {
        let data = vec![1.0, 2.0, 3.0];
        let page = CachePage::with_data(1, data.clone());
        assert_eq!(page.page_id, 1);
        assert_eq!(page.data, data);
        assert!(page.is_allocated);
        assert_eq!(page.ref_count, 1);
    }

    #[test]
    fn test_cache_page_write_read() {
        let mut page = CachePage::new(1, 10);
        page.write(2, &[1.0, 2.0, 3.0]);
        let read = page.read(2, 3);
        assert_eq!(read, &[1.0, 2.0, 3.0]);
        assert!(page.is_dirty);
    }

    #[test]
    fn test_cache_page_ref_counting() {
        let mut page = CachePage::new(1, 256);
        page.add_ref();
        page.add_ref();
        assert_eq!(page.ref_count, 2);

        assert!(!page.release());
        assert_eq!(page.ref_count, 1);

        assert!(page.release());
        assert_eq!(page.ref_count, 0);
    }

    #[test]
    fn test_cache_page_clear() {
        let mut page = CachePage::with_data(1, vec![1.0, 2.0, 3.0]);
        page.is_dirty = true;
        page.clear();

        assert!(!page.is_allocated);
        assert_eq!(page.ref_count, 0);
        assert!(!page.is_dirty);
        assert_eq!(page.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_page_table_new() {
        let table = PageTable::new(256, 100);
        assert_eq!(table.page_size(), 256);
        assert_eq!(table.max_pages(), 100);
        assert_eq!(table.allocated_count(), 0);
    }

    #[test]
    fn test_page_table_allocate() {
        let mut table = PageTable::new(256, 10);

        let page_id = table.allocate_page();
        assert!(page_id.is_some());
        assert_eq!(table.allocated_count(), 1);

        let page = table.get_page(page_id.unwrap());
        assert!(page.is_some());
        assert!(page.unwrap().is_allocated);
    }

    #[test]
    fn test_page_table_allocate_with_data() {
        let mut table = PageTable::new(4, 10);
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let page_id = table.allocate_page_with_data(data.clone());
        assert!(page_id.is_some());

        let page = table.get_page(page_id.unwrap()).unwrap();
        assert_eq!(page.data, data);
    }

    #[test]
    fn test_page_table_free() {
        let mut table = PageTable::new(256, 10);

        let page_id = table.allocate_page().unwrap();
        assert_eq!(table.allocated_count(), 1);

        table.free_page(page_id);
        assert_eq!(table.allocated_count(), 0);
        assert_eq!(table.free_count(), 10);
    }

    #[test]
    fn test_page_table_reuse_freed_pages() {
        let mut table = PageTable::new(256, 10);

        let page_id1 = table.allocate_page().unwrap();
        table.free_page(page_id1);

        let page_id2 = table.allocate_page().unwrap();
        assert_eq!(page_id1, page_id2);
    }

    #[test]
    fn test_page_table_max_pages_limit() {
        let mut table = PageTable::new(256, 2);

        let _id1 = table.allocate_page().unwrap();
        let _id2 = table.allocate_page().unwrap();
        let id3 = table.allocate_page();

        assert!(id3.is_none());
    }

    #[test]
    fn test_page_table_defragment() {
        let mut table = PageTable::new(256, 10);

        let id1 = table.allocate_page().unwrap();
        let _id2 = table.allocate_page().unwrap();
        let id3 = table.allocate_page().unwrap();

        table.free_page(id1);
        table.free_page(id3);

        let removed = table.defragment();
        assert_eq!(removed, 2);
    }

    #[test]
    fn test_page_table_evict_lru() {
        let mut table = PageTable::new(256, 10);

        let _id1 = table.allocate_page().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _id2 = table.allocate_page().unwrap();

        let evicted = table.evict_lru();
        assert!(evicted.is_some());
        assert_eq!(table.allocated_count(), 1);
    }

    #[test]
    fn test_page_table_clear() {
        let mut table = PageTable::new(256, 10);

        table.allocate_page().unwrap();
        table.allocate_page().unwrap();

        table.clear();
        assert_eq!(table.allocated_count(), 0);
    }

    #[test]
    fn test_paged_kv_entry_new() {
        let fp = ContextFingerprint::new(123, 100, "test");
        let entry = PagedKVEntry::new(fp.clone(), vec![1, 2, 3], 300);

        assert_eq!(entry.fingerprint, fp);
        assert_eq!(entry.page_ids, vec![1, 2, 3]);
        assert_eq!(entry.total_tokens, 300);
        assert_eq!(entry.access_count, 0);
    }

    #[test]
    fn test_paged_kv_entry_record_access() {
        let fp = ContextFingerprint::new(123, 100, "test");
        let mut entry = PagedKVEntry::new(fp, vec![1], 100);

        let initial = entry.last_accessed;
        std::thread::sleep(std::time::Duration::from_millis(1));
        entry.record_access();

        assert_eq!(entry.access_count, 1);
        assert!(entry.last_accessed > initial);
    }

    #[test]
    fn test_paged_cache_new() {
        let cache = PagedCache::new(256, 100);
        assert_eq!(cache.page_size(), 256);
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn test_paged_cache_put_get() {
        let cache = PagedCache::new(4, 100);
        let fp = ContextFingerprint::new(123, 10, "test");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = cache.put(&fp, &data);
        assert!(result.is_some());

        let retrieved = cache.get(&fp);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
    }

    #[test]
    fn test_paged_cache_contains() {
        let cache = PagedCache::new(256, 100);
        let fp = ContextFingerprint::new(123, 10, "test");
        let data = vec![1.0, 2.0];

        assert!(!cache.contains(&fp));
        let _ = cache.put(&fp, &data);
        assert!(cache.contains(&fp));
    }

    #[test]
    fn test_paged_cache_remove() {
        let cache = PagedCache::new(256, 100);
        let fp = ContextFingerprint::new(123, 10, "test");
        let data = vec![1.0, 2.0];

        let _ = cache.put(&fp, &data);
        assert!(cache.contains(&fp));

        let removed = cache.remove(&fp);
        assert!(removed.is_some());
        assert!(!cache.contains(&fp));
    }

    #[test]
    #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn test_paged_cache_clear() {
        let cache = PagedCache::new(256, 100);

        for i in 0..5_i32 {
            let fp = ContextFingerprint::new(i as u64, 10, format!("test{i}"));
            let data = [i as f32; 10];
            let _ = cache.put(&fp, &data);
        }

        assert_eq!(cache.entry_count(), 5);

        cache.clear();
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn test_paged_cache_evict_lru() {
        let cache = PagedCache::new(4, 100);

        for i in 0..5_i32 {
            let fp = ContextFingerprint::new(i as u64, 10, format!("test{i}"));
            let data = [i as f32; 4];
            let _ = cache.put(&fp, &data);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let evicted = cache.evict_lru_entries(2);
        assert_eq!(evicted, 2);
        assert_eq!(cache.entry_count(), 3);
    }

    #[test]
    fn test_paged_cache_replace_existing() {
        let cache = PagedCache::new(4, 100);
        let fp = ContextFingerprint::new(123, 10, "test");

        let _ = cache.put(&fp, &[1.0, 2.0, 3.0, 4.0]);
        let _ = cache.put(&fp, &[5.0, 6.0, 7.0, 8.0]);

        let retrieved = cache.get(&fp).expect("should retrieve");
        assert_eq!(retrieved, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn test_paged_cache_clone_shares_state() {
        let cache1 = PagedCache::new(256, 100);
        let cache2 = cache1.clone();

        let fp = ContextFingerprint::new(123, 10, "test");
        let _ = cache1.put(&fp, &[1.0, 2.0]);

        assert!(cache2.contains(&fp));
        assert_eq!(cache1.entry_count(), cache2.entry_count());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_paged_cache_multi_page_entry() {
        let cache = PagedCache::new(4, 100);
        let fp = ContextFingerprint::new(123, 10, "test");

        // Data that spans 3 pages
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let _ = cache.put(&fp, &data);

        let retrieved = cache.get(&fp).expect("get failed");
        assert_eq!(retrieved, data);
    }
}
