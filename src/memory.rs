//! Memory monitoring and resource management for `OxiRAG`.
//!
//! This module provides memory tracking, limits, and budget management
//! to prevent out-of-memory conditions during RAG pipeline operations.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Errors related to memory operations.
#[derive(Debug, Clone, Error)]
pub enum MemoryError {
    /// Memory limit exceeded.
    #[error(
        "Memory limit exceeded: requested {requested} bytes, available {available} bytes, limit {limit} bytes"
    )]
    LimitExceeded {
        /// Bytes requested for allocation.
        requested: usize,
        /// Bytes currently available.
        available: usize,
        /// Total memory limit.
        limit: usize,
    },

    /// Allocation failed.
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),

    /// Deallocation of more bytes than tracked.
    #[error(
        "Deallocation underflow: tried to free {requested} bytes but only {tracked} bytes tracked"
    )]
    DeallocationUnderflow {
        /// Bytes requested for deallocation.
        requested: usize,
        /// Bytes currently tracked.
        tracked: usize,
    },

    /// Invalid memory limit.
    #[error("Invalid memory limit: {0}")]
    InvalidLimit(String),
}

/// Memory usage breakdown by component.
#[derive(Debug, Clone, Default)]
pub struct MemoryBreakdown {
    /// Memory used by embedding cache.
    pub cache_memory: usize,
    /// Memory used by vector index.
    pub index_memory: usize,
    /// Memory used by loaded models.
    pub model_memory: usize,
}

/// Statistics about memory usage.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current memory usage in bytes.
    pub current_bytes: usize,
    /// Peak memory usage in bytes.
    pub peak_bytes: usize,
    /// Memory limit in bytes (0 means unlimited).
    pub limit_bytes: usize,
    /// Total number of allocations tracked.
    pub allocation_count: usize,
    /// Total number of deallocations tracked.
    pub deallocation_count: usize,
    /// Memory breakdown by component.
    pub breakdown: MemoryBreakdown,
}

impl MemoryStats {
    /// Returns the percentage of memory limit used.
    #[must_use]
    pub fn usage_percentage(&self) -> f64 {
        if self.limit_bytes == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                (self.current_bytes as f64 / self.limit_bytes as f64) * 100.0
            }
        }
    }

    /// Returns the available memory in bytes.
    #[must_use]
    pub fn available_bytes(&self) -> usize {
        if self.limit_bytes == 0 {
            usize::MAX
        } else {
            self.limit_bytes.saturating_sub(self.current_bytes)
        }
    }

    /// Check if memory usage is within limits.
    #[must_use]
    pub fn is_within_limit(&self) -> bool {
        self.limit_bytes == 0 || self.current_bytes <= self.limit_bytes
    }
}

/// Thread-safe memory monitor for tracking allocations and enforcing limits.
pub struct MemoryMonitor {
    current_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
    limit_bytes: AtomicUsize,
    allocation_count: AtomicUsize,
    deallocation_count: AtomicUsize,
    breakdown: RwLock<MemoryBreakdown>,
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryMonitor {
    /// Create a new memory monitor with no limit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            limit_bytes: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
            breakdown: RwLock::new(MemoryBreakdown::default()),
        }
    }

    /// Create a new memory monitor with a specified limit.
    #[must_use]
    pub fn with_limit(limit_bytes: usize) -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            limit_bytes: AtomicUsize::new(limit_bytes),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
            breakdown: RwLock::new(MemoryBreakdown::default()),
        }
    }

    /// Get current memory usage estimate.
    #[must_use]
    pub fn current_usage(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Get peak memory usage.
    #[must_use]
    pub fn peak_usage(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Get the current memory limit.
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit_bytes.load(Ordering::Relaxed)
    }

    /// Set memory limit in bytes. Use 0 for unlimited.
    pub fn set_limit(&self, bytes: usize) {
        self.limit_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Check if current usage is within limits.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::LimitExceeded` if current usage exceeds the limit.
    pub fn check_limit(&self) -> Result<(), MemoryError> {
        let current = self.current_bytes.load(Ordering::Relaxed);
        let limit = self.limit_bytes.load(Ordering::Relaxed);

        if limit > 0 && current > limit {
            Err(MemoryError::LimitExceeded {
                requested: 0,
                available: 0,
                limit,
            })
        } else {
            Ok(())
        }
    }

    /// Check if an allocation of the given size would exceed the limit.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::LimitExceeded` if the allocation would exceed the limit.
    pub fn check_allocation(&self, bytes: usize) -> Result<(), MemoryError> {
        let current = self.current_bytes.load(Ordering::Relaxed);
        let limit = self.limit_bytes.load(Ordering::Relaxed);

        if limit > 0 && current.saturating_add(bytes) > limit {
            Err(MemoryError::LimitExceeded {
                requested: bytes,
                available: limit.saturating_sub(current),
                limit,
            })
        } else {
            Ok(())
        }
    }

    /// Register an allocation of the given size.
    pub fn register_allocation(&self, bytes: usize) {
        let new_current = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        // Update peak using compare-and-swap loop
        let mut current_peak = self.peak_bytes.load(Ordering::Relaxed);
        while new_current > current_peak {
            match self.peak_bytes.compare_exchange_weak(
                current_peak,
                new_current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_peak = x,
            }
        }
    }

    /// Unregister an allocation (deallocation).
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::DeallocationUnderflow` if attempting to free more
    /// bytes than currently tracked.
    pub fn unregister_allocation(&self, bytes: usize) -> Result<(), MemoryError> {
        let current = self.current_bytes.load(Ordering::Relaxed);
        if bytes > current {
            return Err(MemoryError::DeallocationUnderflow {
                requested: bytes,
                tracked: current,
            });
        }

        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Update memory breakdown for a specific component.
    pub fn update_breakdown(&self, component: MemoryComponent, bytes: usize) {
        if let Ok(mut breakdown) = self.breakdown.write() {
            match component {
                MemoryComponent::Cache => breakdown.cache_memory = bytes,
                MemoryComponent::Index => breakdown.index_memory = bytes,
                MemoryComponent::Model => breakdown.model_memory = bytes,
            }
        }
    }

    /// Get a snapshot of current memory statistics.
    #[must_use]
    pub fn stats(&self) -> MemoryStats {
        let breakdown = self.breakdown.read().map(|b| b.clone()).unwrap_or_default();

        MemoryStats {
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            limit_bytes: self.limit_bytes.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.deallocation_count.load(Ordering::Relaxed),
            breakdown,
        }
    }

    /// Reset all statistics to zero.
    pub fn reset(&self) {
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);

        if let Ok(mut breakdown) = self.breakdown.write() {
            *breakdown = MemoryBreakdown::default();
        }
    }
}

/// Memory component identifiers for breakdown tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryComponent {
    /// Embedding cache memory.
    Cache,
    /// Vector index memory.
    Index,
    /// Model memory.
    Model,
}

/// Memory budget manager for controlled resource allocation.
#[derive(Clone)]
pub struct MemoryBudget {
    monitor: Arc<MemoryMonitor>,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBudget {
    /// Create a new memory budget with no limit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            monitor: Arc::new(MemoryMonitor::new()),
        }
    }

    /// Create a new memory budget with a specified limit.
    #[must_use]
    pub fn with_limit(limit_bytes: usize) -> Self {
        Self {
            monitor: Arc::new(MemoryMonitor::with_limit(limit_bytes)),
        }
    }

    /// Create a memory budget from an existing monitor.
    #[must_use]
    pub fn from_monitor(monitor: Arc<MemoryMonitor>) -> Self {
        Self { monitor }
    }

    /// Get the underlying memory monitor.
    #[must_use]
    pub fn monitor(&self) -> &MemoryMonitor {
        &self.monitor
    }

    /// Allocate memory from the budget, returning a guard that frees on drop.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::LimitExceeded` if the allocation would exceed the limit.
    pub fn allocate(&self, size: usize) -> Result<MemoryGuard, MemoryError> {
        self.monitor.check_allocation(size)?;
        self.monitor.register_allocation(size);
        Ok(MemoryGuard {
            monitor: Arc::clone(&self.monitor),
            size,
        })
    }

    /// Try to allocate memory, returning None if it would exceed the limit.
    #[must_use]
    pub fn try_allocate(&self, size: usize) -> Option<MemoryGuard> {
        self.allocate(size).ok()
    }

    /// Get current budget statistics.
    #[must_use]
    pub fn stats(&self) -> MemoryStats {
        self.monitor.stats()
    }

    /// Get the available bytes in the budget.
    #[must_use]
    pub fn available(&self) -> usize {
        let current = self.monitor.current_usage();
        let limit = self.monitor.limit();
        if limit == 0 {
            usize::MAX
        } else {
            limit.saturating_sub(current)
        }
    }

    /// Set the memory limit.
    pub fn set_limit(&self, bytes: usize) {
        self.monitor.set_limit(bytes);
    }
}

/// A guard that automatically frees allocated memory when dropped.
pub struct MemoryGuard {
    monitor: Arc<MemoryMonitor>,
    size: usize,
}

impl MemoryGuard {
    /// Get the size of this allocation.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Resize this allocation.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::LimitExceeded` if increasing the size would exceed the limit.
    pub fn resize(&mut self, new_size: usize) -> Result<(), MemoryError> {
        if new_size > self.size {
            let additional = new_size - self.size;
            self.monitor.check_allocation(additional)?;
            self.monitor.register_allocation(additional);
        } else if new_size < self.size {
            let freed = self.size - new_size;
            // Ignore underflow errors during resize - shouldn't happen
            let _ = self.monitor.unregister_allocation(freed);
        }
        self.size = new_size;
        Ok(())
    }

    /// Consume this guard without freeing the memory.
    ///
    /// This is useful when transferring ownership of the allocation.
    #[must_use]
    pub fn leak(self) -> usize {
        let size = self.size;
        std::mem::forget(self);
        size
    }
}

impl Drop for MemoryGuard {
    fn drop(&mut self) {
        // Ignore underflow errors on drop - shouldn't happen in normal use
        let _ = self.monitor.unregister_allocation(self.size);
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_memory_monitor_basic() {
        let monitor = MemoryMonitor::new();

        assert_eq!(monitor.current_usage(), 0);
        assert_eq!(monitor.peak_usage(), 0);

        monitor.register_allocation(1000);
        assert_eq!(monitor.current_usage(), 1000);
        assert_eq!(monitor.peak_usage(), 1000);

        monitor.register_allocation(500);
        assert_eq!(monitor.current_usage(), 1500);
        assert_eq!(monitor.peak_usage(), 1500);

        monitor.unregister_allocation(800).unwrap();
        assert_eq!(monitor.current_usage(), 700);
        assert_eq!(monitor.peak_usage(), 1500);
    }

    #[test]
    fn test_memory_monitor_with_limit() {
        let monitor = MemoryMonitor::with_limit(1000);

        assert!(monitor.check_allocation(500).is_ok());
        assert!(monitor.check_allocation(1000).is_ok());
        assert!(monitor.check_allocation(1001).is_err());

        monitor.register_allocation(600);
        assert!(monitor.check_allocation(400).is_ok());
        assert!(monitor.check_allocation(401).is_err());
    }

    #[test]
    fn test_memory_monitor_set_limit() {
        let monitor = MemoryMonitor::new();
        assert_eq!(monitor.limit(), 0);

        monitor.set_limit(2000);
        assert_eq!(monitor.limit(), 2000);

        monitor.register_allocation(1500);
        assert!(monitor.check_limit().is_ok());

        monitor.register_allocation(600);
        assert!(monitor.check_limit().is_err());
    }

    #[test]
    fn test_deallocation_underflow() {
        let monitor = MemoryMonitor::new();
        monitor.register_allocation(100);

        let result = monitor.unregister_allocation(200);
        assert!(matches!(
            result,
            Err(MemoryError::DeallocationUnderflow { .. })
        ));
    }

    #[test]
    fn test_memory_stats() {
        let monitor = MemoryMonitor::with_limit(10000);

        monitor.register_allocation(1000);
        monitor.register_allocation(2000);
        monitor.unregister_allocation(500).unwrap();

        let stats = monitor.stats();
        assert_eq!(stats.current_bytes, 2500);
        assert_eq!(stats.peak_bytes, 3000);
        assert_eq!(stats.limit_bytes, 10000);
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.deallocation_count, 1);
        assert_eq!(stats.available_bytes(), 7500);
        assert!((stats.usage_percentage() - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_memory_breakdown() {
        let monitor = MemoryMonitor::new();

        monitor.update_breakdown(MemoryComponent::Cache, 1000);
        monitor.update_breakdown(MemoryComponent::Index, 2000);
        monitor.update_breakdown(MemoryComponent::Model, 5000);

        let stats = monitor.stats();
        assert_eq!(stats.breakdown.cache_memory, 1000);
        assert_eq!(stats.breakdown.index_memory, 2000);
        assert_eq!(stats.breakdown.model_memory, 5000);
    }

    #[test]
    fn test_memory_budget_allocation() {
        let budget = MemoryBudget::with_limit(1000);

        let guard1 = budget.allocate(400).unwrap();
        assert_eq!(guard1.size(), 400);
        assert_eq!(budget.monitor().current_usage(), 400);

        let guard2 = budget.allocate(400).unwrap();
        assert_eq!(budget.monitor().current_usage(), 800);

        // This should fail - would exceed limit
        let result = budget.allocate(300);
        assert!(result.is_err());

        // Drop guard1, should free memory
        drop(guard1);
        assert_eq!(budget.monitor().current_usage(), 400);

        // Now we can allocate more
        let _guard3 = budget.allocate(500).unwrap();
        assert_eq!(budget.monitor().current_usage(), 900);

        drop(guard2);
    }

    #[test]
    fn test_memory_guard_drop() {
        let budget = MemoryBudget::with_limit(1000);

        {
            let _guard = budget.allocate(500).unwrap();
            assert_eq!(budget.monitor().current_usage(), 500);
        }

        assert_eq!(budget.monitor().current_usage(), 0);
    }

    #[test]
    fn test_memory_guard_resize() {
        let budget = MemoryBudget::with_limit(1000);

        let mut guard = budget.allocate(300).unwrap();
        assert_eq!(guard.size(), 300);
        assert_eq!(budget.monitor().current_usage(), 300);

        // Increase size
        guard.resize(500).unwrap();
        assert_eq!(guard.size(), 500);
        assert_eq!(budget.monitor().current_usage(), 500);

        // Decrease size
        guard.resize(200).unwrap();
        assert_eq!(guard.size(), 200);
        assert_eq!(budget.monitor().current_usage(), 200);

        // Try to resize beyond limit (current usage is 200, limit is 1000)
        // Trying to resize to 1100 would require 900 additional bytes
        // but only 800 are available (1000 - 200)
        let result = guard.resize(1100);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_guard_leak() {
        let budget = MemoryBudget::with_limit(1000);

        let guard = budget.allocate(500).unwrap();
        let size = guard.leak();

        assert_eq!(size, 500);
        // Memory should still be allocated since we leaked the guard
        assert_eq!(budget.monitor().current_usage(), 500);
    }

    #[test]
    fn test_try_allocate() {
        let budget = MemoryBudget::with_limit(1000);

        let guard1 = budget.try_allocate(800);
        assert!(guard1.is_some());

        let guard2 = budget.try_allocate(300);
        assert!(guard2.is_none());

        drop(guard1);
    }

    #[test]
    fn test_unlimited_budget() {
        let budget = MemoryBudget::new();

        assert_eq!(budget.available(), usize::MAX);

        let _guard1 = budget.allocate(1_000_000).unwrap();
        let _guard2 = budget.allocate(1_000_000_000).unwrap();

        // Should still be able to allocate more
        let _guard3 = budget.allocate(1_000_000_000).unwrap();
    }

    #[test]
    fn test_memory_monitor_reset() {
        let monitor = MemoryMonitor::with_limit(10000);

        monitor.register_allocation(5000);
        monitor.update_breakdown(MemoryComponent::Cache, 1000);

        let stats = monitor.stats();
        assert_eq!(stats.current_bytes, 5000);
        assert_eq!(stats.breakdown.cache_memory, 1000);

        monitor.reset();

        let stats = monitor.stats();
        assert_eq!(stats.current_bytes, 0);
        assert_eq!(stats.peak_bytes, 0);
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.breakdown.cache_memory, 0);
    }

    #[test]
    fn test_concurrent_allocations() {
        let budget = Arc::new(MemoryBudget::with_limit(100_000));
        let mut handles = vec![];

        for _ in 0..10 {
            let b = Arc::clone(&budget);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    if let Ok(guard) = b.allocate(10) {
                        thread::yield_now();
                        drop(guard);
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // After all threads complete, memory should be back to 0
        assert_eq!(budget.monitor().current_usage(), 0);
    }

    #[test]
    fn test_memory_stats_within_limit() {
        let stats = MemoryStats {
            current_bytes: 500,
            peak_bytes: 800,
            limit_bytes: 1000,
            allocation_count: 5,
            deallocation_count: 2,
            breakdown: MemoryBreakdown::default(),
        };

        assert!(stats.is_within_limit());
        assert_eq!(stats.available_bytes(), 500);

        let stats_exceeded = MemoryStats {
            current_bytes: 1500,
            limit_bytes: 1000,
            ..stats.clone()
        };

        assert!(!stats_exceeded.is_within_limit());
    }

    #[test]
    fn test_memory_stats_unlimited() {
        let stats = MemoryStats {
            current_bytes: 1_000_000,
            peak_bytes: 2_000_000,
            limit_bytes: 0, // unlimited
            allocation_count: 100,
            deallocation_count: 50,
            breakdown: MemoryBreakdown::default(),
        };

        assert!(stats.is_within_limit());
        assert_eq!(stats.available_bytes(), usize::MAX);
        assert_eq!(stats.usage_percentage(), 0.0);
    }

    #[test]
    fn test_memory_error_display() {
        let err = MemoryError::LimitExceeded {
            requested: 1000,
            available: 500,
            limit: 2000,
        };
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
        assert!(msg.contains("2000"));

        let err = MemoryError::DeallocationUnderflow {
            requested: 1000,
            tracked: 500,
        };
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));

        let err = MemoryError::AllocationFailed("test error".to_string());
        assert!(err.to_string().contains("test error"));

        let err = MemoryError::InvalidLimit("negative value".to_string());
        assert!(err.to_string().contains("negative value"));
    }

    #[test]
    fn test_memory_budget_from_monitor() {
        let monitor = Arc::new(MemoryMonitor::with_limit(5000));
        monitor.register_allocation(1000);

        let budget = MemoryBudget::from_monitor(monitor);
        assert_eq!(budget.monitor().current_usage(), 1000);
        assert_eq!(budget.monitor().limit(), 5000);
    }

    #[test]
    fn test_peak_tracking_with_fluctuations() {
        let monitor = MemoryMonitor::new();

        monitor.register_allocation(1000);
        assert_eq!(monitor.peak_usage(), 1000);

        monitor.register_allocation(2000);
        assert_eq!(monitor.peak_usage(), 3000);

        monitor.unregister_allocation(2500).unwrap();
        assert_eq!(monitor.current_usage(), 500);
        assert_eq!(monitor.peak_usage(), 3000); // Peak should not decrease

        monitor.register_allocation(1000);
        assert_eq!(monitor.peak_usage(), 3000); // Still below previous peak

        monitor.register_allocation(2000);
        assert_eq!(monitor.peak_usage(), 3500); // New peak
    }
}
