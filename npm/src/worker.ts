/**
 * `WorkerEngine` — runs the OxiRAG WASM engine in a dedicated Web Worker.
 *
 * Offloading the engine to a worker thread keeps the main thread responsive
 * even during expensive embedding or search operations.  The worker uses the
 * `worker_init` / `worker_handle_message` WASM bindings defined in
 * `src/wasm_worker.rs` and bundled into `pkg/oxirag.js`.
 *
 * No `SharedArrayBuffer` is required — all communication uses plain
 * structured-clone messages (JSON-safe objects and strings).
 *
 * @example
 * ```typescript
 * import { WorkerEngine } from '@cool-japan/oxirag-wasm';
 *
 * const engine = WorkerEngine.create({ dimension: 128 });
 *
 * const docId = await engine.index({ content: 'Hello, world!' });
 * const output = await engine.query('hello', 5);
 * console.log(output.final_answer);
 *
 * engine.terminate();
 * ```
 *
 * @remarks
 * `WorkerEngine` is only usable in environments that support the Web Worker
 * API (browsers and Deno).  In Node.js, use `OxiRagEngine` directly (from
 * `engine.ts`) with the `@node-rs/worker-threads` polyfill if needed.
 */

import type {
  Document,
  EngineConfig,
  PipelineOutput,
  WorkerMessageType,
  WorkerResponse,
} from './types.js';

// ─── Pending-request registry ─────────────────────────────────────────────────

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
}

// ─── WorkerEngine ─────────────────────────────────────────────────────────────

/**
 * Promise-based proxy that forwards calls to a `WasmRagEngine` running inside
 * a Web Worker.
 *
 * Each call assigns a unique correlation id to the outbound message and stores
 * a `{resolve, reject}` pair in a `Map`.  When the worker responds with the
 * same id, the pair is removed and the promise is settled.
 */
export class WorkerEngine {
  private readonly worker: Worker;
  private readonly pending = new Map<string, PendingRequest>();
  private requestCounter = 0;

  /** Internal — use `WorkerEngine.create()` instead. */
  private constructor(worker: Worker) {
    this.worker = worker;

    this.worker.onmessage = (event: MessageEvent<unknown>) => {
      this.handleWorkerMessage(event.data);
    };

    this.worker.onerror = (event: ErrorEvent) => {
      // Reject all outstanding requests with the worker-level error.
      const err = new Error(event.message ?? 'Worker error');
      for (const pending of this.pending.values()) {
        pending.reject(err);
      }
      this.pending.clear();
    };
  }

  // ── Factory ───────────────────────────────────────────────────────────────

  /**
   * Create a `WorkerEngine` that spawns a module-type Web Worker from
   * `worker.js` (located in the package root next to `pkg/`).
   *
   * The worker initialises itself lazily on the first message it receives,
   * using the `dimension` passed here.
   *
   * @param config Engine configuration (`dimension` defaults to 128).
   */
  static create(config: EngineConfig = {}): WorkerEngine {
    const dimension = config.dimension ?? 128;

    // `new URL('../worker.js', import.meta.url)` is the bundler-friendly way to
    // reference a worker script relative to this module.  Both Vite and webpack
    // understand this pattern and emit the worker as a separate chunk.
    const worker = new Worker(new URL('../worker.js', import.meta.url), {
      type: 'module',
      name: `oxirag-worker-${dimension}`,
    });

    return new WorkerEngine(worker);
  }

  // ── Message handling ──────────────────────────────────────────────────────

  /**
   * Dispatch an inbound message from the worker to the appropriate pending
   * promise.  Accepts both:
   * - Plain `WorkerResponse` objects (structured-clone)
   * - JSON strings (the worker's current implementation serialises to JSON)
   */
  private handleWorkerMessage(data: unknown): void {
    let response: WorkerResponse;

    if (typeof data === 'string') {
      try {
        response = JSON.parse(data) as WorkerResponse;
      } catch {
        console.error('[WorkerEngine] received unparseable string from worker:', data);
        return;
      }
    } else if (data !== null && typeof data === 'object' && 'id' in data) {
      response = data as WorkerResponse;
    } else {
      console.error('[WorkerEngine] received unexpected message shape:', data);
      return;
    }

    const pending = this.pending.get(response.id);
    if (pending === undefined) {
      // This can happen if the same response arrives twice (defensive).
      return;
    }
    this.pending.delete(response.id);

    if (response.ok) {
      pending.resolve(response.result);
    } else {
      pending.reject(new Error(response.error ?? 'Worker returned an error without a message'));
    }
  }

  // ── Low-level RPC ─────────────────────────────────────────────────────────

  /**
   * Post a typed message to the worker and return a promise that resolves
   * when the corresponding response arrives.
   */
  private send(
    type: WorkerMessageType,
    payload: Record<string, unknown> = {},
  ): Promise<unknown> {
    const id = `req-${(++this.requestCounter).toString(36)}`;

    return new Promise<unknown>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, payload });
    });
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Index a document in the worker-side engine and return its UUID.
   *
   * @param doc Object with `content` and optional `title`.
   */
  async index(doc: Pick<Document, 'content' | 'title'>): Promise<string> {
    const payload: Record<string, unknown> = { content: doc.content };
    if (doc.title !== undefined) {
      payload['title'] = doc.title;
    }
    return this.send('index', payload) as Promise<string>;
  }

  /**
   * Run the full pipeline query in the worker and return the parsed output.
   *
   * @param text  Natural-language query string.
   * @param topK  Maximum results (default 10).
   */
  async query(text: string, topK = 10): Promise<PipelineOutput> {
    return this.send('query', { query: text, top_k: topK }) as Promise<PipelineOutput>;
  }

  /**
   * Return the number of indexed documents from the worker-side engine.
   */
  async count(): Promise<number> {
    return this.send('count') as Promise<number>;
  }

  /**
   * Clear all documents from the worker-side engine.
   */
  async clear(): Promise<void> {
    await this.send('clear');
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /**
   * Terminate the underlying worker.
   *
   * After calling this, all pending promises are rejected and further calls
   * to `index`, `query`, `count`, or `clear` will throw.
   */
  terminate(): void {
    const err = new Error('WorkerEngine terminated');
    for (const pending of this.pending.values()) {
      pending.reject(err);
    }
    this.pending.clear();
    this.worker.terminate();
  }
}
