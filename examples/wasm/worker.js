/**
 * OxiRAG Web Worker — worker.js
 *
 * Runs WasmRagEngine in a dedicated thread, keeping the main thread free.
 *
 * Usage from the main thread:
 *
 *   const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
 *
 *   // Index a document
 *   worker.postMessage({ id: 'r1', type: 'index', payload: { content: 'hello world' } });
 *
 *   // Query
 *   worker.postMessage({ id: 'r2', type: 'query', payload: { query: 'hello', top_k: 5 } });
 *
 *   // Count
 *   worker.postMessage({ id: 'r3', type: 'count', payload: {} });
 *
 *   // Clear
 *   worker.postMessage({ id: 'r4', type: 'clear', payload: {} });
 *
 *   worker.onmessage = (e) => console.log(e.data);
 *
 * All responses have the shape:
 *   { id, ok: true,  result: <value> }
 *   { id, ok: false, error:  '<message>' }
 */
import init, { worker_init, worker_handle_message } from '../../pkg/oxirag.js';

let initialized = false;

self.onmessage = async (event) => {
    // Initialise the WASM module and engine on the first message.
    if (!initialized) {
        await init();
        // Dimension 128 — change to match the embedding model used in production.
        worker_init(128);
        initialized = true;
    }

    try {
        const response = await worker_handle_message(event.data);
        self.postMessage(response);
    } catch (err) {
        // Construct a synthetic error response so the main thread always gets
        // a well-formed object back.
        const id = (event.data && event.data.id) ? event.data.id : 'unknown';
        self.postMessage(JSON.stringify({
            id,
            ok: false,
            error: String(err),
        }));
    }
};
