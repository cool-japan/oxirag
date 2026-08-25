//! Target-portable monotonic and wall clocks.
//!
//! `std::time::Instant::now()` and `std::time::SystemTime::now()` **panic** on
//! `wasm32-unknown-unknown` — the target has no clock behind either of them.
//! Verified rather than assumed; the probe that established it is recorded in
//! `tests/wasm_clock.rs`:
//!
//! ```text
//! RuntimeError: unreachable
//!   at <std::time::Instant>::now (wasm-function[3155])
//! ```
//!
//! That is not a recoverable error. A panic on `wasm32` aborts the instance, and
//! under a `panic = "abort"` release profile — which is what browser consumers
//! build with, including the COOLJAPAN Playground — it is an uncatchable trap
//! that poisons every later call. `Pipeline::process` timed itself with
//! `Instant::now()`, so the whole engine was one query away from killing the
//! page it ran in.
//!
//! # The gate
//!
//! `clippy.toml` disallows `std::time::Instant::now`, `std::time::SystemTime::now`
//! and `web_sys::window` crate-wide. That is why [`Instant`] below is a newtype
//! on **both** targets rather than a re-export on native: clippy resolves a
//! re-export back to the original path, so a native `pub use std::time::Instant`
//! would make every call site light up and force a blanket `allow` — which is
//! the same as having no gate at all.
//!
//! The cost is a wrapper that native builds pay nothing for at runtime
//! (`#[repr(transparent)]`, every method `#[inline]`) and an [`Instant::into_std`]
//! escape hatch for callers that need the real thing.
//!
//! This gate exists because ADR-0005's `?Send` decision was applied to six traits
//! and then not applied to the 122 written after it. A rule nobody can forget is
//! worth more than a rule nobody remembers.
//!
//! # What the `wasm32` clock is
//!
//! `js_sys::Date::now()` — milliseconds since the Unix epoch, as an `f64`.
//! Deliberately not `performance.now()`: `Date` needs no `web-sys` feature and
//! resolves in a `Window` and a `WorkerGlobalScope` alike, and this engine's
//! browser consumers run inside a Web Worker.
//!
//! The cost is millisecond resolution and a clock that is not guaranteed
//! monotonic across a system time change. Both are acceptable and neither is
//! hidden: every duration this crate reports on `wasm32` is a whole number of
//! milliseconds, and callers needing better measure elapsed time in JavaScript
//! around the call — which is what every page in the Playground already does.
//!
//! Because that clock can move backwards, every subtraction here **saturates at
//! zero**. `std::time::Instant::duration_since` panics when the argument is
//! later; this one cannot, deliberately. A trap is not an acceptable outcome for
//! a metrics call, and a duration reported as `0ms` is.
//!
//! `chrono`'s `Utc::now()` is NOT affected and is not wrapped: OxiRAG enables
//! `chrono`'s `wasmbind` feature, which routes it through `Date.now()` already.
//! `tests/wasm_clock.rs` asserts that, so a dependency bump that dropped the
//! feature fails there rather than in a browser.

use core::ops::{Add, Sub};
use std::time::Duration;

/// A measurement point on a monotonic clock, on every target.
///
/// Backed by `std::time::Instant` off `wasm32`, and by `Date.now()` on it. See
/// the module documentation for why this is a newtype rather than a re-export,
/// and for what the `wasm32` clock does and does not guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Instant(Inner);

#[cfg(not(target_arch = "wasm32"))]
type Inner = std::time::Instant;

/// On `wasm32`, milliseconds since the Unix epoch. `Duration` is totally
/// ordered, which is what makes the derived `Ord` sound.
#[cfg(target_arch = "wasm32")]
type Inner = Duration;

impl Instant {
    /// Read the clock.
    #[must_use]
    #[inline]
    pub fn now() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            #[allow(clippy::disallowed_methods)]
            Self(std::time::Instant::now())
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self(epoch_millis())
        }
    }

    /// How long since this instant, saturating at zero.
    #[must_use]
    #[inline]
    pub fn elapsed(&self) -> Duration {
        Self::now().saturating_duration_since(*self)
    }

    /// How long between `earlier` and this instant, saturating at zero.
    ///
    /// Unlike `std::time::Instant::duration_since` this cannot panic; see the
    /// module documentation.
    #[must_use]
    #[inline]
    pub fn duration_since(&self, earlier: Self) -> Duration {
        self.saturating_duration_since(earlier)
    }

    /// How long between `earlier` and this instant, saturating at zero.
    #[must_use]
    #[inline]
    pub fn saturating_duration_since(&self, earlier: Self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.0.saturating_duration_since(earlier.0)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.0.saturating_sub(earlier.0)
        }
    }

    /// This instant plus `duration`, or `None` on overflow.
    #[must_use]
    #[inline]
    pub fn checked_add(&self, duration: Duration) -> Option<Self> {
        self.0.checked_add(duration).map(Self)
    }

    /// This instant minus `duration`, or `None` if that is not representable.
    #[must_use]
    #[inline]
    pub fn checked_sub(&self, duration: Duration) -> Option<Self> {
        self.0.checked_sub(duration).map(Self)
    }

    /// The underlying `std::time::Instant`, for callers that need to hand one to
    /// an API this crate does not control. Native only, because on `wasm32`
    /// there is no such value to hand over.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    #[inline]
    pub fn into_std(self) -> std::time::Instant {
        self.0
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<std::time::Instant> for Instant {
    #[inline]
    fn from(instant: std::time::Instant) -> Self {
        Self(instant)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<Instant> for std::time::Instant {
    #[inline]
    fn from(instant: Instant) -> Self {
        instant.0
    }
}

impl Sub for Instant {
    type Output = Duration;

    #[inline]
    fn sub(self, earlier: Self) -> Duration {
        self.saturating_duration_since(earlier)
    }
}

impl core::ops::AddAssign<Duration> for Instant {
    #[inline]
    fn add_assign(&mut self, duration: Duration) {
        *self = *self + duration;
    }
}

impl Add<Duration> for Instant {
    type Output = Self;

    /// Saturating rather than panicking, for the reason given in the module
    /// documentation.
    #[inline]
    fn add(self, duration: Duration) -> Self {
        self.checked_add(duration).unwrap_or(self)
    }
}

/// The wall clock, as a `SystemTime`.
///
/// On `wasm32` a `Date.now()` reporting a non-finite or pre-epoch value yields
/// [`std::time::UNIX_EPOCH`] rather than panicking — `Duration::from_secs_f64`
/// panics on both, and this function is called from paths that must not trap.
#[must_use]
#[inline]
pub fn system_now() -> std::time::SystemTime {
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[allow(clippy::disallowed_methods)]
        std::time::SystemTime::now()
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::time::UNIX_EPOCH + epoch_millis()
    }
}

/// Milliseconds since the Unix epoch, as a `Duration`.
#[cfg(target_arch = "wasm32")]
#[inline]
fn epoch_millis() -> Duration {
    let millis = js_sys::Date::now();
    if millis.is_finite() && millis > 0.0 {
        // `Date.now()` is a whole number of milliseconds well inside u64 range;
        // the guard above rules out the cases where this cast would be wrong.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Duration::from_millis(millis as u64)
    } else {
        Duration::ZERO
    }
}
