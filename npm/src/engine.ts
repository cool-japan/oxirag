/**
 * `OxiRagEngine` — typed wrapper around the OxiRAG WASM module.
 *
 * This class provides a fully typed, Promise-based interface that sits on top
 * of the raw `WasmRagEngine` class exported by `wasm-pack`.  It handles
 * module initialisation, argument normalisation, and JSON parsing so callers
 * never have to deal with raw `JsValue` strings.
 *
 * @example
 * ```typescript
 * import { OxiRagEngine } from '@cool-japan/oxirag-wasm';
 *
 * const engine = await OxiRagEngine.create({ dimension: 384 });
 *
 * const docId = await engine.index({ content: 'Rust is memory-safe.' });
 * console.log('indexed', docId);
 *
 * const output = await engine.query('What is Rust?', { topK: 5 });
 * console.log(output.final_answer, output.confidence);
 *
 * for await (const result of engine.searchStream('memory safety')) {
 *   console.log(result.score, result.document.content);
 * }
 * ```
 */

import type {
  Document,
  EngineConfig,
  PipelineOutput,
  QueryOptions,
  SearchResult,
  StreamOptions,
} from './types.js';

// ─── Internal WASM module type ───────────────────────────────────────────────

/**
 * Minimal type declaration for the wasm-pack generated module.
 *
 * The full generated `d.ts` lives at `../pkg/oxirag.d.ts` after building.
 * We only declare what we actually use here so the TypeScript compiler stays
 * happy even before `wasm-pack build` has been run.
 */
interface WasmModule {
  default?: () => Promise<void>;
  WasmRagEngine: {
    new (dimension: number): RawWasmEngine;
  };
  cosine_similarity(a: Float32Array, b: Float32Array): number;
  normalize_vector(v: Float32Array): Float32Array;
  version(): string;
}

interface RawWasmEngine {
  index(content: string, title: string | null): Promise<string>;
  query(queryText: string, topK: number): Promise<string>;
  query_search_array(queryText: string, topK: number): Promise<string[]>;
  count(): Promise<number>;
  clear(): Promise<void>;
}

// ─── Module loader ───────────────────────────────────────────────────────────

/** Cache the loaded module so we only call the init function once. */
let _wasmModule: WasmModule | null = null;

/**
 * Load and initialise the WASM module exactly once.
 * Subsequent calls return the cached module.
 */
async function loadModule(): Promise<WasmModule> {
  if (_wasmModule !== null) {
    return _wasmModule;
  }

  // Dynamic import — bundlers (Vite, Rollup, webpack) resolve this at build
  // time; Node.js ESM resolves it at runtime.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mod = (await import('../pkg/oxirag.js' as string)) as unknown as WasmModule;

  // wasm-pack generates an async default export that fetches and compiles the
  // `.wasm` binary.  Call it if present (target=web); target=bundler modules
  // initialise synchronously.
  if (typeof mod.default === 'function') {
    await mod.default();
  }

  _wasmModule = mod;
  return mod;
}

// ─── OxiRagEngine ────────────────────────────────────────────────────────────

/**
 * High-level typed wrapper over `WasmRagEngine`.
 *
 * Instances are created via the async factory `OxiRagEngine.create()`.
 * Direct construction (`new OxiRagEngine(...)`) is intentionally not exposed.
 */
export class OxiRagEngine {
  private readonly raw: RawWasmEngine;

  /** Use `OxiRagEngine.create()` instead. */
  private constructor(raw: RawWasmEngine) {
    this.raw = raw;
  }

  // ── Factory ──────────────────────────────────────────────────────────────

  /**
   * Create and initialise a new engine instance.
   *
   * Loads the WASM binary on the first call (subsequent calls reuse the
   * cached module).  Safe to call concurrently from multiple call sites.
   *
   * @param config Engine configuration (dimension defaults to 128).
   */
  static async create(config: EngineConfig = {}): Promise<OxiRagEngine> {
    const dimension = config.dimension ?? 128;
    const mod = await loadModule();
    const raw = new mod.WasmRagEngine(dimension);
    return new OxiRagEngine(raw);
  }

  // ── Indexing ─────────────────────────────────────────────────────────────

  /**
   * Embed and store a document in the in-memory vector index.
   *
   * @param doc  Object with at minimum a `content` string.
   *             `title` is optional and aids draft generation.
   * @returns    The UUID string assigned by the engine.
   */
  async index(doc: Pick<Document, 'content' | 'title'>): Promise<string> {
    return this.raw.index(doc.content, doc.title ?? null);
  }

  /**
   * Index multiple documents in sequence and return their assigned IDs.
   *
   * @param docs Array of documents to index.
   */
  async indexMany(docs: Pick<Document, 'content' | 'title'>[]): Promise<string[]> {
    const ids: string[] = [];
    for (const doc of docs) {
      ids.push(await this.index(doc));
    }
    return ids;
  }

  // ── Querying ─────────────────────────────────────────────────────────────

  /**
   * Run the full four-layer pipeline and return the parsed output.
   *
   * The pipeline executes:
   *   1. Echo (Layer 1): dense vector retrieval
   *   2. RuleBasedSpeculator (Layer 2): draft speculation
   *   3. JudgeImpl (Layer 3): SMT-backed claim verification
   *
   * @param text    Natural-language query string.
   * @param options `topK` (default 10) and `minScore` (default none).
   * @returns       Full `PipelineOutput` with all layer results.
   */
  async query(text: string, options: QueryOptions = {}): Promise<PipelineOutput> {
    const topK = options.topK ?? 10;
    const json = await this.raw.query(text, topK);
    return JSON.parse(json) as PipelineOutput;
  }

  /**
   * Query the Echo layer only and return raw search results.
   *
   * Bypasses the Speculator and Judge layers — useful when you only need
   * similarity-ranked documents without the full pipeline overhead.
   *
   * @param text  Query string.
   * @param topK  Maximum results to return (default 10).
   * @returns     Array of `SearchResult` objects sorted by descending score.
   */
  async search(text: string, topK = 10): Promise<SearchResult[]> {
    const arr = await this.raw.query_search_array(text, topK);
    return arr.map((item) => JSON.parse(item) as SearchResult);
  }

  /**
   * Yield search results one at a time as an `AsyncGenerator`.
   *
   * This is the streaming counterpart of `search()`: it iterates the same
   * `query_search_array` result but yields each entry lazily, letting callers
   * process results incrementally without waiting for the whole array.
   *
   * Optional `minScore` filtering is applied at the TypeScript layer so that
   * low-relevance results are never surfaced to the consumer.
   *
   * @example
   * ```typescript
   * for await (const result of engine.searchStream('ownership in Rust', { topK: 20, minScore: 0.4 })) {
   *   console.log(result.rank, result.score.toFixed(3), result.document.content);
   * }
   * ```
   *
   * @param text    Query string.
   * @param options `topK` and `minScore`.
   */
  async *searchStream(
    text: string,
    options: StreamOptions = {},
  ): AsyncGenerator<SearchResult, void, undefined> {
    const topK = options.topK ?? 10;
    const minScore = options.minScore ?? 0;

    const arr = await this.raw.query_search_array(text, topK);
    for (const item of arr) {
      const result = JSON.parse(item) as SearchResult;
      if (result.score >= minScore) {
        yield result;
      }
    }
  }

  // ── Metadata ─────────────────────────────────────────────────────────────

  /**
   * Return the number of documents currently in the index.
   */
  async count(): Promise<number> {
    return this.raw.count();
  }

  /**
   * Delete all indexed documents and reset the vector store.
   *
   * This operation is irreversible — there is no soft-delete or recycle bin.
   */
  async clear(): Promise<void> {
    return this.raw.clear();
  }

  // ── Static helpers ───────────────────────────────────────────────────────

  /**
   * Compute the cosine similarity between two equal-length vectors.
   *
   * The WASM module must already be loaded (i.e. `OxiRagEngine.create()` must
   * have been called at least once) before invoking this helper.
   *
   * @param a First vector.
   * @param b Second vector (must be the same length as `a`).
   * @returns  Similarity score in [-1.0, 1.0] (1.0 = identical direction).
   * @throws   If the WASM module has not been loaded yet.
   */
  static cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (_wasmModule === null) {
      throw new Error(
        'OxiRagEngine.cosineSimilarity() called before the WASM module was loaded. ' +
          'Await OxiRagEngine.create() first.',
      );
    }
    return _wasmModule.cosine_similarity(a, b);
  }

  /**
   * Normalise a vector to unit length.
   *
   * Requires the WASM module to be loaded (see `cosineSimilarity`).
   *
   * @param v  Input vector.
   * @returns  New `Float32Array` with unit-length norm.
   * @throws   If the WASM module has not been loaded yet.
   */
  static normalizeVector(v: Float32Array): Float32Array {
    if (_wasmModule === null) {
      throw new Error(
        'OxiRagEngine.normalizeVector() called before the WASM module was loaded. ' +
          'Await OxiRagEngine.create() first.',
      );
    }
    return _wasmModule.normalize_vector(v);
  }

  /**
   * Return the OxiRAG library version string (e.g. `"0.6.0"`).
   *
   * Loads the WASM module if it has not been loaded already.
   */
  static async version(): Promise<string> {
    const mod = await loadModule();
    return mod.version();
  }
}
