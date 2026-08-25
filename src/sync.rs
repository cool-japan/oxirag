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
/// For a trait whose supertrait list must read `Send + Sync` on native and
/// nothing on `wasm32`, because a `wasm32` implementation holds a `JsValue` or a
/// `RefCell`. Naming this alias keeps one trait definition for both targets, the
/// same way ADR-0005's `cfg_attr` pair does for `async_trait`.
///
/// **Nothing in the crate uses it today, deliberately.** It was written for
/// `Speculator`, whose `Send + Sync` supertrait made it unimplementable on
/// `wasm32` — and then the real cause turned out to be `HiddenStateCache`
/// forking to `Arc<RefCell<…>>` under `cfg(not(feature = "native"))` for no
/// reason (`std::sync::RwLock` works on `wasm32`). Deleting the fork fixed it
/// without weakening a public bound, which is the better repair: relaxing a
/// supertrait is a semantic change every downstream caller sees.
///
/// Kept because the next trait to hit this will hit it for a reason that cannot
/// be deleted — an implementation that genuinely holds a `JsValue`. Reach for
/// the deletion first; reach for this only when there is nothing to delete.
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
