//! Target-portable async synchronisation primitives.
//!
//! Builds carrying the `native` feature use `tokio::sync`, which is what the
//! multi-thread runtime schedules against. Every other build — `wasm32`, which
//! has no runtime and no threads, and any `--no-default-features` build on any
//! target — takes the same types from `async-lock`, a runtime-agnostic pure-Rust
//! implementation with a matching `.read().await` / `.write().await` /
//! `.lock().await` surface.
//!
//! Modules that hold state behind an async lock import from here rather than
//! naming either crate directly, so a module stays buildable on both targets
//! without a `cfg` of its own. See ADR-0005 for the sibling decision on
//! `async_trait`'s `?Send` bound.

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub use tokio::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(any(target_arch = "wasm32", not(feature = "native")))]
pub use async_lock::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// `Send + Sync` on every target that has threads, and no bound at all on `wasm32`.
///
/// Traits whose supertrait list reads `Send + Sync` are unimplementable in the
/// browser by anything holding a `JsValue` or a `RefCell`, which is what the
/// single-threaded wasm implementations of these layers use. Naming this alias
/// instead keeps one trait definition for both targets, the same way ADR-0005's
/// `cfg_attr` pair does for `async_trait`.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> MaybeSendSync for T {}

/// The `wasm32` half of [`MaybeSendSync`] — no bound, because there are no threads.
#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}

#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSendSync for T {}

/// `RwLock::try_read()` as an [`Option`], whichever backend is in play.
///
/// `tokio::sync::RwLock::try_read` returns `Result<_, TryLockError>` and
/// `async_lock::RwLock::try_read` returns `Option<_>`. Callers that only want
/// "a guard, or nothing" go through here instead of carrying a `cfg` for the
/// difference.
#[must_use]
pub fn try_read<T: ?Sized>(lock: &RwLock<T>) -> Option<RwLockReadGuard<'_, T>> {
    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    {
        lock.try_read().ok()
    }
    #[cfg(any(target_arch = "wasm32", not(feature = "native")))]
    {
        lock.try_read()
    }
}
