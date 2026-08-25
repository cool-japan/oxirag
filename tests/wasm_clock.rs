//! Proves, in a real wasm runtime, why `src/time.rs` and
//! `src/global_scope.rs` exist.
//!
//! Run with:
//!
//! ```text
//! wasm-pack test --node --no-default-features --features echo
//! ```
//!
//! The first two tests assert the *replacement* works. The behaviour they
//! replace is recorded here rather than tested, because a test that asserts a
//! panic on `wasm32` cannot be written: the panic aborts the instance and takes
//! the harness with it. Measured directly, `std::time::Instant::now()` and
//! `std::time::SystemTime::now()` both produce:
//!
//! ```text
//! RuntimeError: unreachable
//!   at <std::time::Instant>::now (wasm-function[3155])
//! ```
//!
//! `clippy.toml` disallows both so the crate cannot reacquire them.
#![cfg(target_arch = "wasm32")]

use std::time::{Duration, UNIX_EPOCH};

use oxirag::time::{Instant, system_now};
use wasm_bindgen_test::*;

/// `crate::time::Instant` reads a clock instead of trapping, and its `elapsed`
/// is a duration rather than a panic.
#[wasm_bindgen_test]
fn the_monotonic_clock_reads_without_trapping() {
    let start = Instant::now();
    let elapsed = start.elapsed();
    // A same-tick read is legitimately zero; what matters is that it returned.
    assert!(elapsed < Duration::from_secs(60));
}

/// `crate::time::system_now()` is a real wall clock, not the epoch. Anything
/// after 2020 proves `Date.now()` was actually consulted.
#[wasm_bindgen_test]
fn the_wall_clock_is_a_real_date() {
    let since_epoch = system_now()
        .duration_since(UNIX_EPOCH)
        .expect("system_now is never before the epoch by construction");
    assert!(
        since_epoch.as_secs() > 1_577_836_800,
        "system_now() returned {since_epoch:?}, which is before 2020"
    );
}

/// Instants order the way callers assume when they sort or compare them.
#[wasm_bindgen_test]
fn instants_order_forwards() {
    let first = Instant::now();
    let second = first + Duration::from_millis(5);
    assert!(second > first);
    assert_eq!(second.duration_since(first), Duration::from_millis(5));
}

/// Subtracting a later instant from an earlier one saturates instead of
/// panicking. `std::time::Instant::duration_since` panics here; under
/// `panic = "abort"` that would abort the page from inside a metrics call.
#[wasm_bindgen_test]
fn a_backwards_measurement_saturates_rather_than_trapping() {
    let later = Instant::now() + Duration::from_secs(10);
    assert_eq!(Instant::now().duration_since(later), Duration::ZERO);
}

/// `chrono::Utc::now()` is unaffected — OxiRAG enables chrono's `wasmbind`
/// feature, which routes it through `Date.now()`. Asserted so that a future
/// dependency bump dropping that feature fails here rather than in a browser.
#[wasm_bindgen_test]
fn chrono_utc_now_is_already_wasm_safe() {
    assert!(chrono::Utc::now().timestamp() > 1_577_836_800);
}
