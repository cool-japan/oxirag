/**
 * OxiRAG Web Worker entry point.
 *
 * This script is loaded by `WorkerEngine.create()` and runs inside a
 * dedicated worker thread.  It bridges the message-passing protocol defined
 * in `wasm_worker.rs` with the standard `self.onmessage` / `self.postMessage`
 * Web Worker API.
 *
 * Message protocol — request shape (structured-clone or JSON string):
 *   { id: string, type: 'index'|'query'|'count'|'clear', payload: object }
 *
 * Response shape (JSON string):
 *   { id: string, ok: true,  result: any }
 *   { id: string, ok: false, error:  string }
 *
 * See `src/wasm_worker.rs` for the Rust-side implementation details and the
 * full payload field specification.
 */

// Import path is relative to the package root where this file lives, not to
// the compiled pkg/ directory — bundlers (Vite, Rollup) will rewrite this.
import init, { worker_init, worker_handle_message } from '../pkg/oxirag.js';

/** Whether the WASM module and engine have been initialised for this worker. */
let initialized = false;

/**
 * Default embedding dimension used when the worker starts up.
 *
 * Callers that need a different dimension should post an `init` message
 * (not yet a formal protocol verb, but can be added) or pass it via a
 * URL search parameter on the Worker constructor.
 *
 * The dimension must match the embedding model used by the main thread:
 * - 128  — MockEmbeddingProvider (fast, for testing)
 * - 384  — MiniLM-L6-v2
 * - 768  — BERT-base / DistilBERT
 * - 1536 — OpenAI text-embedding-3-small
 */
const DEFAULT_DIMENSION = 128;

/**
 * Parse the optional `?dimension=NNN` search parameter from the worker URL so
 * the caller can override the dimension without changing the worker script.
 */
function parseDimension() {
    try {
        const url = new URL(self.location.href);
        const raw = url.searchParams.get('dimension');
        if (raw !== null) {
            const parsed = parseInt(raw, 10);
            if (Number.isFinite(parsed) && parsed > 0) {
                return parsed;
            }
        }
    } catch {
        // `self.location` may not be available in all environments.
    }
    return DEFAULT_DIMENSION;
}

self.onmessage = async (event) => {
    // Lazily initialise on the first inbound message.
    if (!initialized) {
        await init();
        worker_init(parseDimension());
        initialized = true;
    }

    try {
        // `worker_handle_message` accepts both a JS object (structured-clone)
        // and a JSON string.  Pass the raw event.data through; the Rust side
        // handles both formats.
        const response = await worker_handle_message(event.data);
        // The Rust binding returns a JSON string — forward it as-is so that
        // `WorkerEngine.handleWorkerMessage()` can parse it on the main thread.
        self.postMessage(response);
    } catch (err) {
        const id = (event.data && typeof event.data === 'object' && event.data.id)
            ? event.data.id
            : 'unknown';
        self.postMessage(JSON.stringify({
            id,
            ok: false,
            error: String(err),
        }));
    }
};
