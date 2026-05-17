# @cool-japan/oxirag-wasm

Typed TypeScript/WASM wrapper around the [OxiRAG](https://github.com/cool-japan/oxirag) four-layer Retrieval-Augmented Generation engine.  Runs entirely in the browser or at the edge — no server required.

## Features

- Full four-layer pipeline: Echo retrieval → Rule-based speculation → SMT-backed verification → final answer
- TypeScript-first API with complete type definitions
- Streaming search results via `AsyncGenerator`
- Off-main-thread operation via `WorkerEngine` (Web Worker)
- Composable streaming combinators: `take`, `filterScore`, `map`, `deduplicate`, …
- Zero runtime dependencies — pure WASM + TypeScript
- Tree-shakeable exports

---

## Installation

```bash
npm install @cool-japan/oxirag-wasm
```

The package ships pre-built TypeScript declarations and a compiled WASM binary in `pkg/`.

---

## Building from source

Prerequisites: [Rust toolchain](https://rustup.rs/) and [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/).

```bash
# Clone the repo
git clone https://github.com/cool-japan/oxirag
cd oxirag/npm

# Install TypeScript dev tools
npm install

# Build the WASM module + TypeScript declarations in one step
npm run build
```

Alternatively, run the steps individually:

```bash
# Step 1 — compile Rust → WASM (output: ../pkg/)
npm run build:wasm

# Step 2 — compile TypeScript (output: dist/)
npm run build:ts

# Type-check only (no output)
npm run typecheck
```

### Feature flags

```bash
# Basic WASM (in-memory vector store)
wasm-pack build --target web --release --features wasm ../../

# With IndexedDB persistence
wasm-pack build --target web --release --features wasm,wasm-indexeddb ../../
```

---

## Usage

### OxiRagEngine (main-thread)

```typescript
import { OxiRagEngine } from '@cool-japan/oxirag-wasm';

// Create an engine. The WASM binary is loaded and compiled once.
const engine = await OxiRagEngine.create({ dimension: 128 });

// Index documents
const id1 = await engine.index({ content: 'Rust is a systems programming language focused on safety.' });
const id2 = await engine.index({
  content: 'WebAssembly enables near-native performance in the browser.',
  title: 'What is WASM?',
});

// Full four-layer pipeline query
const output = await engine.query('What is Rust?', { topK: 5 });
console.log(output.final_answer);   // string
console.log(output.confidence);     // 0.0–1.0
console.log(output.search_results); // SearchResult[]
console.log(output.speculation);    // SpeculationResult | undefined
console.log(output.verification);   // VerificationResult | undefined

// Echo-only search (no speculation or verification)
const results = await engine.search('memory safety', 5);
for (const r of results) {
  console.log(r.rank, r.score.toFixed(3), r.document.content);
}

// Document count
const n = await engine.count();
console.log(`${n} documents indexed`);

// Clear the index
await engine.clear();
```

### Streaming results

```typescript
import { OxiRagEngine } from '@cool-japan/oxirag-wasm';

const engine = await OxiRagEngine.create();
await engine.index({ content: 'Ownership in Rust prevents use-after-free bugs.' });

// Lazy async generator — process results one by one
for await (const result of engine.searchStream('ownership', { topK: 10, minScore: 0.3 })) {
  console.log(result.score.toFixed(3), result.document.content);
}
```

### Streaming combinators

```typescript
import { OxiRagEngine, take, filterScore, map, collectStream } from '@cool-japan/oxirag-wasm';

const engine = await OxiRagEngine.create();

const stream = engine.searchStream('Rust safety', { topK: 50 });

// Compose: keep scores ≥ 0.4, take first 5, project to content strings
const topContents = await collectStream(
  map(take(filterScore(stream, 0.4), 5), (r) => r.document.content)
);
console.log(topContents);
```

Available streaming operators:

| Combinator       | Description                                        |
|------------------|----------------------------------------------------|
| `take(n)`        | Yield at most `n` items                            |
| `skip(n)`        | Skip the first `n` items                           |
| `filterScore(t)` | Keep results with `score >= t`                     |
| `filter(fn)`     | Keep results matching a predicate                  |
| `map(fn)`        | Project results to another type                    |
| `deduplicate()`  | Emit each document id only once                    |

Terminal operators: `collectStream`, `forEach`, `reduce`, `first`, `best`.

### Static helpers

```typescript
import { OxiRagEngine } from '@cool-japan/oxirag-wasm';

// Must have called create() at least once before using static helpers.
await OxiRagEngine.create();

const a = new Float32Array([1, 0, 0]);
const b = new Float32Array([0.9, 0.1, 0]);

const sim = OxiRagEngine.cosineSimilarity(a, b);
const norm = OxiRagEngine.normalizeVector(a);

const ver = await OxiRagEngine.version();  // e.g. "0.6.0"
```

---

### WorkerEngine (off-main-thread)

Offload the engine to a dedicated Web Worker to prevent main-thread jank:

```typescript
import { WorkerEngine } from '@cool-japan/oxirag-wasm';

// Synchronous constructor — the worker initialises lazily on the first message
const engine = WorkerEngine.create({ dimension: 128 });

const docId = await engine.index({ content: 'Hello, world!' });

const output = await engine.query('hello', 5);
console.log(output.final_answer);

const n = await engine.count();
await engine.clear();

// Shut down the worker when you are done
engine.terminate();
```

The `WorkerEngine` uses a correlation-id scheme so multiple in-flight requests are correctly matched to their responses.  No `SharedArrayBuffer` or special HTTP headers (`COOP`/`COEP`) are required.

---

## TypeScript types

All public types are exported from the package root:

```typescript
import type {
  Document,
  Draft,
  SearchResult,
  QueryOptions,
  StreamOptions,
  PipelineOutput,
  SpeculationDecision,
  SpeculationResult,
  VerificationStatus,
  VerificationResult,
  ClaimVerificationResult,
  LogicalClaim,
  EngineConfig,
  WorkerMessageType,
  WorkerRequest,
  WorkerResponse,
  SearchStream,
} from '@cool-japan/oxirag-wasm';
```

### `PipelineOutput` shape

The object returned by `engine.query()` mirrors the Rust `PipelineOutput` struct and uses **snake_case** field names (Serde defaults):

```typescript
interface PipelineOutput {
  query: { text: string; top_k: number; min_score?: number; filters: Record<string, string> };
  search_results: SearchResult[];
  draft: Draft;
  speculation?: SpeculationResult;   // Layer 2 — absent if skipped
  verification?: VerificationResult; // Layer 3 — absent if skipped
  final_answer: string;
  confidence: number;
  layers_used: string[];
  total_duration_ms: number;
}
```

### `VerificationStatus` values

Mirrors the Rust `VerificationStatus` enum:

```typescript
type VerificationStatus = 'Verified' | 'Falsified' | 'Unknown' | 'Timeout' | 'Error';
```

---

## Bundle size tips

1. **Target `web` (not `bundler`)** when calling `wasm-pack build` — this generates an async `init()` function that fetches only the bytes actually needed.
2. **Lazy-load the engine** — `OxiRagEngine.create()` defers the WASM fetch until it is actually called, so it does not block initial page load.
3. **Use `WorkerEngine`** in latency-sensitive UIs — moving WASM off the main thread prevents blocking the render pipeline.
4. **Strip unused streaming operators** — all exports in `streaming.ts` are individually named and tree-shakeable by bundlers that respect `"sideEffects": false`.

---

## React integration

```tsx
import { useEffect, useRef, useState } from 'react';
import { OxiRagEngine, type SearchResult } from '@cool-japan/oxirag-wasm';

export function OxiRagSearch() {
  const engineRef = useRef<OxiRagEngine | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    OxiRagEngine.create({ dimension: 128 }).then((engine) => {
      engineRef.current = engine;
    });
  }, []);

  const handleSearch = async () => {
    if (!engineRef.current || !query.trim()) return;
    setLoading(true);
    try {
      const found = await engineRef.current.search(query, 5);
      setResults(found);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search…" />
      <button onClick={handleSearch} disabled={loading}>
        {loading ? 'Searching…' : 'Search'}
      </button>
      <ul>
        {results.map((r) => (
          <li key={r.document.id}>
            <strong>{r.score.toFixed(3)}</strong> — {r.document.content}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

## Worker dimension override

The `WorkerEngine` worker script reads an optional `?dimension=NNN` query parameter from its URL so you can customise the embedding dimension without modifying the script:

```typescript
const worker = new Worker(
  new URL('../pkg/worker.js', import.meta.url).href + '?dimension=384',
  { type: 'module' }
);
```

When using `WorkerEngine.create()`, the dimension defaults to 128.  To support other dimensions, subclass `WorkerEngine` or construct the `Worker` manually and pass it through a modified `worker.js`.

---

## License

Apache-2.0 — see [LICENSE](../LICENSE).
