# OxiRAG Troubleshooting Guide

This guide covers common build errors, runtime issues, and performance problems.
Questions are grouped by category. For performance tuning beyond what is covered
here, see `docs/perf.md`.

---

## Build Issues

### `error[E0277]: the trait 'Send' is not satisfied` when implementing VectorStore for WASM types

**Symptom:** Compilation of a `VectorStore` or `PrefixCacheStore` implementation
for a WASM IndexedDB-backed type fails with:

```
error[E0277]: `JsValue` cannot be sent between threads safely
   --> src/my_store.rs:12:1
    |
12  | #[async_trait]
    | ^^^^^^^^^^^^^^ `JsValue` cannot be sent between threads safely
```

**Cause:** The non-WASM variant of `#[async_trait]` expands `async fn` bodies into
`Pin<Box<dyn Future + Send>>`, which requires all captured types to be `Send`.
`JsValue` is `!Send` by design in the JavaScript/WASM execution model.

**Solution:** Use the cfg-gated pair of attributes on your implementation:

```rust
use async_trait::async_trait;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl VectorStore for MyIndexedDbStore {
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError> {
        // ...
    }
    // ... other methods
}
```

Apply the same pair to the trait _definition_ if you are defining your own trait
that extends `VectorStore`. See ADR-0005 (`docs/adr/0005-wasm-send-bound-cfg-gate.md`)
for the full rationale.

---

### `error: unresolved import 'oxirag::layer1_echo::RedbVectorStore'`

**Symptom:**

```
error[E0432]: unresolved import `oxirag::layer1_echo::RedbVectorStore`
 --> src/main.rs:3:5
  |
3 | use oxirag::layer1_echo::RedbVectorStore;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ no `RedbVectorStore` in `layer1_echo`
```

**Cause:** `RedbVectorStore` is gated behind the `echo-redb` feature. Similarly:

| Type | Required feature |
|---|---|
| `RedbVectorStore` | `echo-redb` |
| `RedbGraphStore` | `graphrag-redb` |
| `RedbPrefixCache` | `prefix-cache-redb` |
| `CandleEmbeddingProvider` | `speculator` |
| `CandleSlmSpeculator` | `speculator` |
| `CandleClipProvider` | `multimodal` |
| `OxizVerifier` | `judge` |
| `OtelSpanObserver` | `otel` |
| `build_router` / `build_and_serve` | `rest-server` |

**Solution:** Add the required feature to your `Cargo.toml`:

```toml
# For persistent vector store:
oxirag = { version = "0.6", features = ["echo-redb"] }

# For multiple features:
oxirag = { version = "0.6", features = ["echo-redb", "graphrag-redb", "speculator"] }
```

---

### `redb: error: method 'begin_read' not found` or similar redb 4.x migration errors

**Symptom:** Compiling with `redb = "4"` produces errors like:

```
error[E0599]: no method named `begin_read` found for type `redb::Database`
```

or

```
error[E0277]: the trait bound `redb::Table<'_, ...>: redb::ReadableTable<...>` is not satisfied
```

**Cause:** redb 4.x moved several methods from inherent impls to traits. The
most common fixes:

**Solution:** Add the required trait imports:

```rust
// For reading from a redb table:
use redb::ReadableTable;        // .get(), .iter(), .range()
use redb::ReadableDatabase;     // .begin_read()

// For writing:
use redb::ReadableTableMetadata; // .len()

// Typical import block for redb 4.x:
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
```

If you are implementing a custom redb backend on top of OxiRAG, open a read
transaction with `db.begin_read()?` (requires `use redb::ReadableDatabase`).

---

### napi-rs / maturin build failures

**Symptom (napi-rs):**

```
error: could not find `napi-build` or `build.rs` for crate `oxirag`
```

or

```
error: could not compile `oxirag` due to CDYLIB / RLIB conflict
```

**Cause:** Node.js native modules require `crate-type = ["cdylib", "rlib"]` in
`Cargo.toml`. The `build.rs` must call `napi_build::setup()`.

**Solution:** Verify your `Cargo.toml` has the correct crate type:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

And `build.rs` contains:

```rust
fn main() {
    napi_build::setup();
}
```

For maturin (Python wheel) builds, ensure you are using maturin 1.x and the
`python` feature is enabled:

```sh
# Check the feature is present:
cargo check --features python

# Build the wheel:
maturin build --release --features python
```

**Symptom (maturin, WASM target conflict):** `maturin develop` fails because
the default Rust toolchain targets WASM but the Python extension requires a
native target.

**Solution:** Specify the target explicitly:

```sh
maturin develop --features python --target x86_64-unknown-linux-gnu
```

---

## Runtime Issues

### Pipeline returns empty search results

**Checklist:**

1. **Were documents indexed before the query?** `EchoLayer` starts empty. Call
   `echo.index(doc).await?` or `echo.index_batch(docs).await?` before calling
   `echo.search(...)`.

   ```rust
   assert!(echo.count().await > 0, "index documents before searching");
   ```

2. **Do the provider and store have the same dimension?**
   `InMemoryVectorStore::new(dim)` must receive the same `dim` as
   `MockEmbeddingProvider::new(dim)` or `CandleEmbeddingProvider` reports.

   ```rust
   // Correct:
   let dim = provider.dimension();
   let store = InMemoryVectorStore::new(dim);

   // Wrong — dimension mismatch will cause a VectorStoreError::DimensionMismatch:
   let provider = MockEmbeddingProvider::new(384);
   let store = InMemoryVectorStore::new(128); // different dim!
   ```

3. **Is `min_score` too high?** If you pass `Some(0.99)` as the threshold, only
   near-perfect matches will be returned. Try `None` first, then lower the
   threshold incrementally.

4. **Is the metadata filter excluding all results?** If using
   `search_with_filter`, verify that at least one document has metadata matching
   your filter:

   ```rust
   // Debug: search without filter first.
   let unfiltered = echo.search("query", 10, None).await?;
   println!("Unfiltered results: {}", unfiltered.len());
   ```

---

### `CandleSlmSpeculator: model not found` or HuggingFace Hub errors

**Symptom:**

```
EmbeddingError::ModelLoad("Failed to load tokenizer: 404 Not Found")
```

or

```
thread 'main' panicked at 'called Result::unwrap() on Err: hf_hub API error'
```

**Cause:** Models are downloaded from HuggingFace Hub (`huggingface.co`) on the
first use. This requires internet access and approximately 400 MB – 5.5 GB of
storage depending on the model.

**Solutions:**

- **Set a custom cache directory** via `HF_HOME`:

  ```sh
  export HF_HOME=/data/models
  ```

- **Check available disk space:** `CandleSlmSpeculator::new_phi2()` downloads
  approximately 5.5 GB. BGE-base downloads approximately 400 MB.

- **Offline / air-gapped environments:** Download the model files manually and
  place them in the HuggingFace Hub cache directory structure:

  ```
  $HF_HOME/hub/models--sentence-transformers--all-MiniLM-L6-v2/
      snapshots/
          main/
              tokenizer.json
              config.json
              model.safetensors
  ```

  The `hf-hub` crate will find the files without downloading if the snapshot
  directory is present.

- **Using `from_local_path`** (for custom or fine-tuned models):

  ```rust
  use oxirag::layer1_echo::{CandleEmbeddingConfig, CandleEmbeddingProvider};

  let config = CandleEmbeddingConfig {
      model_id: "my-local-bert".to_string(),
      revision: "main".to_string(),
      ..CandleEmbeddingConfig::default()
  };
  // Then set HF_HOME to the directory containing
  // `hub/models--my-local-bert/snapshots/main/`
  let provider = CandleEmbeddingProvider::new(config)?;
  ```

---

### WASM: IndexedDB permission denied

**Symptom:** In the browser console:

```
SecurityError: Failed to execute 'indexedDB' on 'Window': access is denied for this document
```

**Cause:** IndexedDB is restricted in certain browser contexts:

- `file://` origins — IndexedDB is blocked in many browsers for local files.
- Third-party iframes with restricted permissions.
- Private / incognito mode in some browsers (write quota may be 0).
- Browsers with strict storage isolation enabled.

**Solutions:**

- Serve your application over `http://localhost` or `https://` instead of
  opening the HTML file directly.
- For development: `python3 -m http.server 8000` or `npx serve` in the
  directory containing your built WASM.
- If you must support private mode, catch the error and fall back to in-memory
  storage:

  ```typescript
  import init, { WasmPipeline } from './oxirag_wasm';

  async function createPipeline() {
    try {
      return await WasmPipeline.with_indexeddb('my-db');
    } catch (e) {
      console.warn('IndexedDB unavailable, using in-memory store:', e);
      return await WasmPipeline.new();
    }
  }
  ```

---

### `panic: called 'unwrap()' on an 'Err' value` in WASM

**Symptom:** The browser tab freezes or shows an `unreachable` error. The browser
console shows only a numeric WebAssembly backtrace without source information.

**Cause:** OxiRAG's WASM module calls `wasm::init()` at startup which installs
`console_error_panic_hook`. If `init()` was not called, panics produce unhelpful
stack traces.

**Solution:** Ensure `oxirag.init()` (or the equivalent `wasm_bindgen` export)
is called before using the pipeline:

```typescript
import init, { init as oxiragInit, WasmPipeline } from './oxirag_wasm';

async function main() {
  await init(); // wasm-bindgen init
  oxiragInit(); // install console_error_panic_hook
  const pipeline = WasmPipeline.new();
  // ...
}
```

After calling `oxiragInit()`, panics will print a full Rust backtrace to the
browser console including the file name and line number.

To also capture panics in JavaScript error tracking (e.g. Sentry):

```typescript
import init, { set_panic_hook } from './oxirag_wasm';
await init();
set_panic_hook(); // equivalent to oxiragInit() — name may vary by binding version
```

---

## Performance Issues

### Search is slow for large document sets (> 10K documents)

**Symptom:** `echo.search(...)` takes hundreds of milliseconds with a corpus of
50K+ documents.

**Cause:** `InMemoryVectorStore` performs an O(n) brute-force linear scan. For
large n this is the dominant cost.

**Solution:** Switch to `AnnVectorStore` which uses the HNSW graph for approximate
nearest neighbor search in O(log n):

```rust
use oxirag::layer1_echo::ann::{AnnVectorStore, AnnConfig};

// Tune for your recall / latency target:
let config = AnnConfig {
    ef_search: 64,   // higher = better recall, slower search
    ef_construction: 200,
    m: 16,
    ..AnnConfig::default()
};

let store = AnnVectorStore::new(384, config);
```

Expected speedup for 50K documents: 50–200x vs. linear scan. See `docs/perf.md`
for detailed benchmark data.

Additionally, ensure your build uses the `release` profile and target-native
SIMD:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

---

### High memory usage with large knowledge graphs

**Symptom:** The `InMemoryGraphStore` consumes gigabytes of RAM with millions of
entities.

**Solutions:**

1. **Switch to `RedbGraphStore`** (feature `graphrag-redb`). Entity and relationship
   data are paged to disk by redb's storage engine. Only the BFS working set and
   the redb page cache are held in RAM.

   ```rust
   use oxirag::layer4_graph::RedbGraphStore;
   let store = RedbGraphStore::open("./data/graph.redb")?;
   ```

2. **Enable prefix cache** to avoid re-embedding identical text segments. Repeated
   document intros (e.g., legal boilerplate) will hit the cache instead of calling
   the embedding provider:

   ```rust
   use oxirag::layer1_echo::{CachedEmbeddingProvider, EmbeddingCacheConfig};

   let cache_config = EmbeddingCacheConfig {
       max_entries: 50_000,
       ..EmbeddingCacheConfig::default()
   };
   let provider = CachedEmbeddingProvider::new(inner_provider, cache_config);
   ```

3. **Limit BFS hop depth.** Each additional hop multiplies the in-memory working
   set. Use `max_hops = 2` for most workloads. See `docs/layers/graph.md` for
   hop-count scaling guidance.

---

### OTel spans showing 100ms overhead

**Symptom:** `SpanReport::total_ms` is approximately 100ms higher than expected,
even for simple queries.

**Cause:** The `OtelSpanObserver` batches spans and exports them via OTLP/gRPC on
a background Tokio task. If the collector is unavailable or slow, the batch queue
fills and the export future takes time to complete. The `on_layer_complete`
callback itself is non-blocking (it only enqueues spans), but the background
export loop may increase observable latency when the collector is congested.

**Solutions:**

- **For zero-overhead in-process metrics:** use `MemoryObserver` instead. It
  has no network I/O and collects records in a `Mutex<Vec>`:

  ```rust
  use std::sync::Arc;
  use oxirag::observability::MemoryObserver;

  let observer = Arc::new(MemoryObserver::new());
  // Register via PipelineBuilder::with_observer(observer.clone())

  // After queries, read the records:
  let records = observer.records();
  let snapshots = observer.pipeline_snapshots();
  ```

- **For OTel export:** verify your OTLP collector endpoint is reachable:

  ```sh
  # Quick check: does the collector accept connections?
  grpc_cli ls localhost:4317
  ```

  Increase the batch timeout or switch to stdout export for debugging:

  ```rust
  // Use OtelSpanObserver::new_stdout() to export to stdout without a collector.
  use oxirag::observability::OtelSpanObserver;
  let observer = OtelSpanObserver::new_stdout()?;
  ```

- **Check export batching configuration.** The `opentelemetry_sdk` BatchSpanProcessor
  defaults to 5-second flush intervals. For latency-sensitive applications, configure
  a shorter export interval or use the `SimpleSpanProcessor` instead.

---

## Feature Flag Reference

Complete list of OxiRAG optional features and what they add:

| Feature | Adds | Dependencies pulled in |
|---|---|---|
| `native` (default) | `tokio`, async streams, `futures` | tokio, tokio-stream, futures |
| `echo` (default) | Layer 1 Echo module | (none beyond core) |
| `speculator` | `CandleEmbeddingProvider`, `CandleSlmSpeculator`, SLM inference | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers |
| `judge` | `OxizVerifier` (real SMT solver) | oxiz |
| `multimodal` | `CandleClipProvider`, image+text embeddings | speculator + image |
| `graphrag` | Layer 4 Graph module (in-memory only) | (none beyond core) |
| `graphrag-redb` | `RedbGraphStore` | graphrag + redb |
| `echo-redb` | `RedbVectorStore` | echo + redb |
| `prefix-cache` | Prefix caching types | (none beyond core) |
| `prefix-cache-redb` | `RedbPrefixCache` | prefix-cache + redb |
| `hidden-states` | Hidden state speculation types | (none beyond core) |
| `distillation` | On-the-fly distillation types | (none beyond core) |
| `quantization` | INT8/INT4/Binary quantization | (none beyond core) |
| `otel` | `OtelSpanObserver`, OTLP+stdout exporters | opentelemetry, opentelemetry_sdk, opentelemetry-otlp, opentelemetry-stdout, tonic |
| `rest-server` | `build_router`, `build_and_serve`, HTTP routes | axum, tower, tower-http, tracing-subscriber |
| `python` | `DefaultPipeline` PyO3 bindings | pyo3, pyo3-async-runtimes |
| `nodejs` | napi-rs Node.js bindings | napi, napi-derive |
| `wasm` | WASM bindings and browser glue | wasm-bindgen, wasm-bindgen-futures, js-sys, web-sys, getrandom, console_error_panic_hook, serde-wasm-bindgen |
| `wasm-indexeddb` | `IndexedDbVectorStore` (persistent WASM) | wasm + echo |
| `wasm-prefix-indexeddb` | IndexedDB PrefixCache (persistent WASM) | wasm + prefix-cache |
| `cuda` | CUDA device support for Candle | speculator + cudarc (x86_64 Linux/Windows only) |
| `metal` | Metal device support (macOS) | speculator |
| `full` | All native features (no WASM, no Python/Node) | Everything above except wasm/python/nodejs |

### Common Feature Combinations

```toml
# Minimal production setup (semantic search + rule-based verification):
oxirag = { version = "0.6", features = [] }  # uses default = ["native", "echo"]

# High-quality production (real embeddings + real SMT verification):
oxirag = { version = "0.6", features = ["speculator", "judge", "echo-redb"] }

# Full GraphRAG deployment:
oxirag = { version = "0.6", features = ["graphrag-redb", "speculator", "judge"] }

# REST API server:
oxirag = { version = "0.6", features = ["rest-server"] }

# Python wheel (maturin):
oxirag = { version = "0.6", features = ["python"] }

# Browser WASM with persistence:
oxirag = { version = "0.6", features = ["wasm", "wasm-indexeddb"] }

# Observability with OTel:
oxirag = { version = "0.6", features = ["otel"] }
```

---

## Checking Your OxiRAG Version

```sh
# Show the version in Cargo.lock:
cargo tree -p oxirag | head -5

# Or in Rust code:
println!("{}", oxirag::VERSION);
```

## Getting Help

- GitHub Issues: https://github.com/cool-japan/oxirag/issues
- Crate documentation: https://docs.rs/oxirag
- Performance tuning: `docs/perf.md`
- Docker deployment: `docs/docker.md`
- ADRs (design decisions): `docs/adr/`
