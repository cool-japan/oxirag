/**
 * OxiRAG WASM browser demo — main.js
 *
 * Uses WasmRagEngine directly in the main thread.
 * For off-main-thread usage, see worker.js.
 *
 * Expects the WASM package to have been built with:
 *   wasm-pack build --target web --release --features wasm,wasm-indexeddb
 */
import init, { WasmRagEngine } from '../../pkg/oxirag.js';

// ── DOM references ───────────────────────────────────────────────────────────

const statusEl   = document.getElementById('status');
const indexBtn   = document.getElementById('index-btn');
const searchBtn  = document.getElementById('search-btn');
const clearBtn   = document.getElementById('clear-btn');
const contentEl  = document.getElementById('content');
const titleEl    = document.getElementById('doc-title');
const queryEl    = document.getElementById('query');
const resultsEl  = document.getElementById('results');
const countEl    = document.getElementById('doc-count');

// ── Helpers ──────────────────────────────────────────────────────────────────

function setStatus(msg, kind = 'info') {
    statusEl.textContent = msg;
    statusEl.className = kind;
}

async function refreshCount(engine) {
    const n = await engine.count();
    countEl.textContent = `Indexed documents: ${n}`;
}

// ── Initialise ───────────────────────────────────────────────────────────────

async function main() {
    await init();
    setStatus('WASM module loaded. Ready.', 'success');

    // Dimension 128 — matches the MockEmbeddingProvider default.
    // Change to 384 / 768 / 1536 when using a real embedding model.
    const engine = new WasmRagEngine(128);

    await refreshCount(engine);

    // Enable buttons now that the engine is ready.
    indexBtn.disabled  = false;
    searchBtn.disabled = false;
    clearBtn.disabled  = false;

    // ── Index handler ────────────────────────────────────────────────────────
    indexBtn.addEventListener('click', async () => {
        const content = contentEl.value.trim();
        if (!content) {
            setStatus('Please enter some content before indexing.', 'error');
            return;
        }

        indexBtn.disabled = true;
        setStatus('Indexing...', 'info');

        try {
            const title = titleEl.value.trim() || null;
            const docId = await engine.index(content, title);
            contentEl.value = '';
            titleEl.value   = '';
            await refreshCount(engine);
            setStatus(`Indexed document ${docId}`, 'success');
        } catch (err) {
            setStatus(`Index failed: ${err}`, 'error');
        } finally {
            indexBtn.disabled = false;
        }
    });

    // ── Search handler ───────────────────────────────────────────────────────
    searchBtn.addEventListener('click', async () => {
        const query = queryEl.value.trim();
        if (!query) {
            setStatus('Please enter a search query.', 'error');
            return;
        }

        searchBtn.disabled = true;
        setStatus('Searching...', 'info');

        try {
            // query_search_array returns an Array of JSON strings, one per result.
            const items = await engine.query_search_array(query, 5);

            if (items.length === 0) {
                resultsEl.textContent = '(no results)';
            } else {
                const parsed = [];
                for (const item of items) {
                    parsed.push(JSON.parse(item));
                }
                resultsEl.textContent = JSON.stringify(parsed, null, 2);
            }

            resultsEl.style.display = 'block';
            setStatus(`Found ${items.length} result(s).`, 'success');
        } catch (err) {
            setStatus(`Search failed: ${err}`, 'error');
        } finally {
            searchBtn.disabled = false;
        }
    });

    // ── Clear handler ────────────────────────────────────────────────────────
    clearBtn.addEventListener('click', async () => {
        if (!confirm('Delete all indexed documents?')) return;

        clearBtn.disabled = true;
        setStatus('Clearing...', 'info');

        try {
            await engine.clear();
            await refreshCount(engine);
            resultsEl.style.display = 'none';
            setStatus('All documents cleared.', 'success');
        } catch (err) {
            setStatus(`Clear failed: ${err}`, 'error');
        } finally {
            clearBtn.disabled = false;
        }
    });
}

main().catch((err) => {
    setStatus(`Fatal error: ${err}`, 'error');
    console.error(err);
});
