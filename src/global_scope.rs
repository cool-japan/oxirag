//! `Window`-or-`WorkerGlobalScope` — the browser APIs OxiRAG needs, from either.
//!
//! Every wasm entry point in this crate used to reach for `web_sys::window()`,
//! and that is wrong in the place this engine is most likely to run. A RAG
//! pipeline blocks its thread on embedding and similarity search, so the correct
//! host for it is a **Web Worker** — which is exactly how the COOLJAPAN
//! Playground runs every one of its wasm modules. In a worker `window()` returns
//! `None`, and the two ways this crate handled that were both wrong:
//!
//! * `src/retry.rs` and `src/pipeline.rs` wrote `.expect("no window")`. Under a
//!   `panic = "abort"` release profile that is an uncatchable trap on the retry
//!   path — the engine killing the page rather than retrying.
//! * `src/layer1_echo/storage/indexeddb.rs` and
//!   `src/prefix_cache/indexeddb_backend.rs` handled `None` politely and
//!   returned `"IndexedDB not available in this browser"`. No trap, but the
//!   sentence is false: IndexedDB is fully available to workers through
//!   `WorkerGlobalScope::indexed_db()`. Persistence was reported as unsupported
//!   on the one host where the engine belongs.
//!
//! Both APIs — `indexedDB` and `setTimeout` — exist on `Window` and on
//! `WorkerGlobalScope` under the same names. They are unrelated types in
//! `web-sys` (no shared trait), so the two are bridged here once by an enum
//! rather than at four call sites by four `cfg`-shaped guesses.

use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;

/// Whichever global this module was instantiated in.
pub enum GlobalScope {
    /// A document context — the main thread.
    Window(web_sys::Window),
    /// A worker context — `DedicatedWorkerGlobalScope` and friends.
    Worker(web_sys::WorkerGlobalScope),
}

impl GlobalScope {
    /// Resolve the current global, or `None` in a context that is neither
    /// (a bare `wasm32-unknown-unknown` host such as `wasm-bindgen-test`'s Node
    /// runner, which is a legitimate place to find oneself and not an error to
    /// panic over).
    #[must_use]
    pub fn current() -> Option<Self> {
        // The one place in the crate allowed to ask: this is the wrapper the
        // `clippy.toml` gate points every other call site at.
        #[allow(clippy::disallowed_methods)]
        let window = web_sys::window();
        if let Some(window) = window {
            return Some(Self::Window(window));
        }
        js_sys::global()
            .dyn_into::<web_sys::WorkerGlobalScope>()
            .ok()
            .map(Self::Worker)
    }

    /// The `indexedDB` factory for this global.
    ///
    /// # Errors
    ///
    /// The `JsValue` the browser threw — typically a `SecurityError` when
    /// storage is blocked by the user's settings.
    pub fn indexed_db(&self) -> Result<Option<web_sys::IdbFactory>, JsValue> {
        match self {
            Self::Window(window) => window.indexed_db(),
            Self::Worker(worker) => worker.indexed_db(),
        }
    }

    /// `setTimeout(handler, millis)`.
    ///
    /// # Errors
    ///
    /// The `JsValue` the browser threw.
    pub fn set_timeout(&self, handler: &js_sys::Function, millis: i32) -> Result<i32, JsValue> {
        match self {
            Self::Window(window) => {
                window.set_timeout_with_callback_and_timeout_and_arguments_0(handler, millis)
            }
            Self::Worker(worker) => {
                worker.set_timeout_with_callback_and_timeout_and_arguments_0(handler, millis)
            }
        }
    }
}
