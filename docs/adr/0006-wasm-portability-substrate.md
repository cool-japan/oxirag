# ADR-0006: The wasm32 Portability Substrate — `crate::sync`, `crate::time`, `crate::global_scope`

**Status:** Accepted
**Date:** 2026-08-25
**Deciders:** KitaSan

## Context

OxiRAG 0.24.0 listed `"wasm"` in its `categories`, shipped `src/wasm.rs`,
`src/wasm_worker.rs`, two IndexedDB backends and an npm package — and did not
compile for `wasm32-unknown-unknown`. `cargo check --target
wasm32-unknown-unknown --no-default-features --features wasm` reported 34 errors;
adding `echo` took it to 39.

ADR-0005 had already established the shape of the answer for one of the three
causes. What it had not established was a place to put the other two, or a way to
keep any of them applied.

Three distinct problems were tangled together:

1. **Compile-time absence.** `tokio::sync::RwLock` was imported unconditionally
   by twelve modules that contain no runtime and no threads — `hybrid_search/bm25`,
   `circuit_breaker`, `layer1_echo/storage/memory`, `collections`, `conversation`,
   `document_pipeline`, `distillation/hotswap`, `index_management`. None of them
   needed tokio; they needed *an async lock*.

2. **Run-time traps.** `std::time::Instant::now()` and
   `std::time::SystemTime::now()` compile for `wasm32-unknown-unknown` and
   **panic** when called. There were 56 and 9 call sites respectively, including
   `Pipeline::process`, which timed every query it ran. Browser consumers build
   with `panic = "abort"`, where a panic is an uncatchable trap that poisons the
   instance — so the engine was one query away from killing the page hosting it.

3. **The wrong global.** Four sites reached for `web_sys::window()`. A RAG
   pipeline blocks its thread on embedding and similarity search, so its correct
   host is a Web Worker, where `window()` is `None`. Two sites wrote
   `.expect("no window")` — a trap on the retry path. The other two returned
   `"IndexedDB not available in this browser"`, which is false: IndexedDB is
   fully available to workers via `WorkerGlobalScope::indexed_db()`.

Only (1) is visible to the compiler. (2) and (3) build clean and fail in a
browser, which is why they had survived a release that advertised WASM support.

## Decision

### Three substrate modules, each aliasing rather than duplicating

**`src/sync.rs`** re-exports `RwLock` / `Mutex` and their guards from
`tokio::sync` when the `native` feature is on, and from `async-lock` otherwise —
which covers every `wasm32` build and every `--no-default-features` build on any
target. `async-lock` is an unconditional dependency rather than an optional one,
so "OxiRAG without tokio" is a configuration that compiles everywhere rather than
one that compiles on `wasm32` and nowhere else.

It also carries `try_read`, because `tokio`'s `try_read()` returns `Result` and
`async-lock`'s returns `Option`, and one call site should not need a `cfg` for
that.

**`src/time.rs`** provides `Instant` and `system_now()`. On `wasm32` both are
backed by `js_sys::Date::now()`; off it, by `std::time`.

`Instant` is a `#[repr(transparent)]` newtype on **both** targets, not a
re-export on native. That is a deliberate cost, paid for the gate described
below: clippy resolves a re-export back to the original path, so a native
`pub use std::time::Instant` would make every call site trip the lint and force a
blanket `allow`, which is the same as having no gate. `into_std()` and `From`
impls give native callers the real type back.

Every subtraction saturates at zero. `std::time::Instant::duration_since` panics
when its argument is later, and `Date.now()` is a wall clock that a system time
adjustment can move backwards — so the honest choices are "trap" or "report
`0ms`", and a metrics call must not trap.

**`src/global_scope.rs`** is an enum over `web_sys::Window` and
`web_sys::WorkerGlobalScope` exposing `indexed_db()` and `set_timeout()`. The two
types share those APIs by name but not by trait, so the bridge is written once
here rather than guessed at four call sites.

### A lint, because a convention was already proven insufficient

`clippy.toml` disallows `std::time::Instant::now`, `std::time::SystemTime::now`
and `web_sys::window` crate-wide, each with a reason naming the replacement.

This is the load-bearing half of the decision. ADR-0005 made the correct call
about `async_trait`'s `Send` bound, applied it to six traits, and was then not
applied to the 122 `#[async_trait]` sites written afterwards — which is most of
why the wasm32 build was broken at all. A decision that depends on everyone
remembering it decays at exactly the rate the codebase grows. The lint does not.

### Tests that run in a real wasm runtime

`tests/wasm_clock.rs` executes under `wasm-pack test --node`. Making that
possible required three manifest changes, because `cargo build --tests` resolves
every dev-dependency and builds every example target regardless of which test is
being run:

- native-only dev-dependencies (`proptest` → `rusty-fork` → `wait-timeout`, which
  does not compile for `wasm32` at all) moved under
  `[target.'cfg(not(target_arch = "wasm32"))'.dev-dependencies]`;
- every `#[cfg(test)]` module became
  `#[cfg(all(test, not(target_arch = "wasm32")))]` — 298 files — because those
  tests use `tokio::test` and `tokio::spawn`;
- examples and benches gained `native` in their `required-features`.

The repository already contained `tests/wasm_indexeddb.rs` and
`tests/wasm_worker.rs`. Neither had ever executed.

## Rationale

- **Alias, do not fork.** Every alternative considered — a parallel `LocalX`
  trait, a `wasm` module with its own copies, per-call-site `cfg` — duplicates an
  API surface that then drifts. ADR-0005 rejected the same shape for the same
  reason.
- **The newtype cost is small and the gate value is large.**
  `#[repr(transparent)]` with `#[inline]` methods compiles to the same code as
  `std::time::Instant` on native. What it buys is that the next module wanting a
  clock cannot silently reintroduce a trap.
- **Saturating arithmetic is the correct semantics, not a workaround.** A
  duration reported as `0ms` is wrong by milliseconds. A trap is wrong by the
  whole page.
- **`Date.now()` over `performance.now()`.** `Date` needs no `web-sys` feature
  and resolves in a `Window` and a `WorkerGlobalScope` alike. `performance.now()`
  would buy sub-millisecond resolution that nothing here reports: browser
  consumers measure elapsed time in JavaScript around the call.

## Consequences

- `Instant` in public struct fields (`prefix_cache::paging::CachePage`,
  `semantic_cache::entry`, `hidden_states::cache`, `prefix_cache::types`) is now
  `oxirag::time::Instant` rather than `std::time::Instant`. **This is a breaking
  change** for code that names the type; `From`/`Into` and `into_std()` convert.
- `async-lock` is in the dependency graph of every build, including native ones
  that never touch it. Three small pure-Rust crates
  (`async-lock`, `event-listener`, `event-listener-strategy`, `concurrent-queue`),
  no C, no build script.
- In-crate unit tests no longer compile for `wasm32`. They test native behaviour
  through `tokio::test`; wasm behaviour belongs in `tests/wasm_*.rs`, which now
  actually runs.
- A new module needing a clock must go through `crate::time` or add an explicit
  `#[allow(clippy::disallowed_methods)]`, which is reviewable.
- `cargo clippy --target wasm32-unknown-unknown` had never been run before this
  change and reported 39 warnings on code no native lint had ever seen. They are
  fixed; the target belongs in CI now, because a lint that is not run is a lint
  that does not exist.

## Alternatives Considered

### Keep `Instant` a re-export on native and skip the lint

Cheaper, and it was the first implementation. Rejected once the lint was written:
clippy flagged all 56 production call sites through the re-export, so keeping it
meant either a crate-level `allow` or no gate. The whole reason this ADR exists
is that the previous decision had no gate.

### `web-time` instead of `crate::time`

`web-time` is the established crate for exactly this. Rejected on the COOLJAPAN
dependency policy — the substrate here is 60 lines with no dependency beyond
`js-sys`, which the crate already carries — and because `web-time` re-exports
`std::time::Instant` on native, which reintroduces the lint problem above.

### Per-call-site `#[cfg(target_arch = "wasm32")]`

What the codebase was already doing for sleeps and hidden-state caches, in two
different styles. 65 clock sites and 4 global sites would have become 69 places
to get it wrong, with no gate. It also produced the `HiddenStateCache`
`Arc<RefCell<…>>` variant that made `Speculator` unimplementable on `wasm32` —
resolved here by deleting the variant, since `std::sync::RwLock` works on
`wasm32` and the fork bought nothing.
