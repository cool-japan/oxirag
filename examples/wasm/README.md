# OxiRAG WASM Browser Example

A minimal browser demo showing OxiRAG running entirely in the browser with
IndexedDB-backed persistent vector storage and an optional Web Worker for
off-main-thread embedding.

## Quick Start

1. Install wasm-pack: https://rustwasm.github.io/wasm-pack/installer/
2. Build: wasm-pack build --target web --release --features wasm,wasm-indexeddb
3. Serve: python3 -m http.server 8080 (from examples/wasm/)
4. Open: http://localhost:8080

## Features

- Persistent vector storage via IndexedDB
- Full-pipeline query (Echo -> Speculator -> Judge)
- Streaming search results via `query_search_array`
- Web Worker offloading (optional, see worker.js)

## File overview

| File          | Purpose                                              |
|---------------|------------------------------------------------------|
| `index.html`  | Main page — loads the WASM module and drives the UI  |
| `main.js`     | Application logic using `WasmRagEngine`              |
| `worker.js`   | Optional Web Worker that runs the engine off-thread  |
| `README.md`   | This file                                            |

## Building with different feature sets

```bash
# Basic WASM (in-memory store, no IndexedDB persistence)
wasm-pack build --target web --release --features wasm

# With IndexedDB persistence for the vector store
wasm-pack build --target web --release --features wasm,wasm-indexeddb

# With IndexedDB persistence for both vector store and prefix cache
wasm-pack build --target web --release --features wasm,wasm-indexeddb,wasm-prefix-indexeddb
```

## Browser compatibility

Requires a modern browser with:
- WebAssembly support (all major browsers since 2017)
- IndexedDB (all major browsers since 2012)
- ES modules (`type="module"` script support, Chrome 61+, Firefox 60+, Safari 10.1+)
