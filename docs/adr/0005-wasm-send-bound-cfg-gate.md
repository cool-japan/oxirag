# ADR-0005: cfg-Gated Send Bounds on VectorStore, Echo, and PrefixCacheStore

**Status:** Accepted
**Date:** 2026-05-17
**Deciders:** KitaSan

## Context

OxiRAG's WASM backend for persistent storage uses the browser's IndexedDB API
via `web-sys`. The key type `web_sys::IdbDatabase` (and the futures produced by
IndexedDB callbacks) holds `JsValue` internally. `JsValue` is deliberately
`!Send`: JavaScript values cannot be transferred across threads, and the
single-threaded WASM executor model does not support cross-thread scheduling.

The original trait definitions in `src/layer1_echo/traits.rs` and
`src/prefix_cache/traits.rs` unconditionally required `Send + Sync` on their
generic bounds, which is the correct constraint for native multi-thread Tokio:

```rust
// Before — breaks WASM IndexedDB implementations
#[async_trait]
pub trait VectorStore {
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError>;
    // ...
}
```

`async_trait` expands the `async fn` bodies into `Pin<Box<dyn Future + Send>>`,
which requires the captured types (including any `JsValue` stored in the
implementation struct) to be `Send`. The `IndexedDbVectorStore` implementation
therefore failed to compile for the `wasm32-unknown-unknown` target.

## Decision

Apply cfg-attribute pairs on all affected traits to select the appropriate
`async_trait` variant based on compilation target:

```rust
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait VectorStore {
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError>;
    // ... all other methods unchanged
}
```

The same pattern is applied to:

- `VectorStore` (`src/layer1_echo/traits.rs`)
- `Echo` (`src/layer1_echo/traits.rs`)
- `EmbeddingProvider` (`src/layer1_echo/traits.rs`)
- `MultiModalEmbeddingProvider` (`src/layer1_echo/traits.rs`)
- `PrefixCacheStore` (`src/prefix_cache/traits.rs`)
- `PrefixCacheExt` (`src/prefix_cache/traits.rs`)

All concrete implementations receive the same cfg-attribute pair so they compile
cleanly on both targets.

The `wasm::init()` function (called from JavaScript on module load) continues to
install `console_error_panic_hook` for human-readable panic messages in the
browser console.

## Rationale

- Single trait definition: there is no duplication of trait methods. The only
  change is which `async_trait` variant is selected at compile time.
- Existing native callers are unaffected: on x86_64 and aarch64 Linux/macOS
  targets, `not(target_arch = "wasm32")` is true and the `Send`-requiring
  expansion is selected, which is identical to the original behavior.
- The WASM IndexedDB backend runs in the browser's single-threaded WASM executor
  where no parallelism is possible anyway, so the relaxed `?Send` bound is
  semantically correct — not just a compilation workaround.
- The approach is consistent with how `tokio` and other async crates handle WASM
  targets: `tokio::task::spawn_local` accepts `!Send` futures on WASM.

## Consequences

- `dyn VectorStore` on `wasm32` does not implement `Send`. Code that stores a
  `Box<dyn VectorStore>` and passes it across a `.await` point on a multi-thread
  runtime will fail to compile on native targets if the concrete type is `!Send`.
  This is the correct behavior: native multi-thread callers should only use
  `Send`-capable stores. The cfg guards make this error surface at compile time
  rather than at runtime.
- Callers that write generic code over `VectorStore` in contexts that require
  `Send` (e.g., `tokio::spawn(async move { store.insert(...).await })`) may need
  to add `where V: VectorStore + Send` explicitly on native targets. The compiler
  error message is clear when this is needed.
- The `LocalVectorStore: 'static` trait alternative (described below) would
  have required a complete duplicate API surface, causing significant maintenance
  overhead for every future addition to `VectorStore`. Avoided.

## Alternatives Considered

### Parallel `LocalVectorStore: 'static` trait

Define a second trait `LocalVectorStore` with `async_trait(?Send)` and implement
it separately for IndexedDB backends. Native backends implement `VectorStore`;
WASM backends implement `LocalVectorStore`. Rejected: doubles the API surface,
breaks the ability to write generic functions that work on both targets with a
single bound, and creates confusion for users reading documentation.

### No WASM persistence

Keep the existing `Send`-only `VectorStore` trait and ship WASM with only the
`InMemoryVectorStore`. Rejected: IndexedDB persistence is a key differentiator
for browser-based RAG applications. Without it, reloading the browser tab
destroys the entire vector index, making the WASM target unsuitable for any
non-trivial deployment.

## Addendum (2026-08-25) — the decision was correct and was not kept

This ADR named six traits. By OxiRAG 0.24.0 the crate carried **122 further**
`#[async_trait]` sites written after it, none of which had received the
`cfg_attr` pair — which is most of why `wasm32-unknown-unknown` had stopped
compiling entirely (39 errors). The `?Send` relaxation is viral: an implementation
missing the pair does not fail locally, it fails at every caller that awaits it,
which is why the symptom presented as "future is not `Send`" in `reranker.rs` and
`pipeline.rs` rather than at the impls actually at fault.

All 122 have been brought in line, and the pair is now applied uniformly rather
than selectively: on `wasm32` the `?Send` variant is chosen for every
`#[async_trait]` in the crate, and off it nothing changes.

The lesson is recorded in [ADR-0006](0006-wasm-portability-substrate.md), which
adds a `clippy.toml` gate for its own decisions rather than relying on the next
author having read this file. A convention that only holds while everyone
remembers it decays at the rate the codebase grows.

One consequence discovered while repairing this: `Speculator` could not be
implemented on `wasm32` at all, because `HiddenStateCache` used
`Arc<RefCell<…>>` under `cfg(not(feature = "native"))` while `Speculator`
requires `Send + Sync`. The fix was not to relax the supertrait but to delete the
`RefCell` fork — `std::sync::RwLock` works on `wasm32` and the fork bought
nothing. `Speculator: Send + Sync` is unchanged.
