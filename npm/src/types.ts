/**
 * TypeScript interfaces matching the OxiRAG Rust type system.
 *
 * Field names use camelCase on the TypeScript side; the JSON produced by
 * the WASM module uses snake_case (Serde defaults), so callers must be aware
 * that raw JSON objects from `WasmRagEngine.query()` use snake_case keys.
 * The typed wrappers in `engine.ts` normalise these into camelCase.
 */

// ─── Document ────────────────────────────────────────────────────────────────

/** A document that has been indexed or is about to be indexed. */
export interface Document {
  /** UUID string assigned by the engine. */
  id: string;
  /** Main textual content. */
  content: string;
  /** Optional human-readable title. */
  title?: string;
  /** Optional source URL or file path. */
  source?: string;
  /** Arbitrary key-value metadata pairs. */
  metadata?: Record<string, string>;
  /** ISO-8601 creation timestamp (present on retrieved documents). */
  created_at?: string;
  /** ISO-8601 last-updated timestamp (present on retrieved documents). */
  updated_at?: string;
}

// ─── Query ───────────────────────────────────────────────────────────────────

/** Options forwarded to the pipeline when calling `OxiRagEngine.query()`. */
export interface QueryOptions {
  /**
   * Maximum number of documents to retrieve from the Echo layer.
   * @default 10
   */
  topK?: number;
  /**
   * Minimum cosine-similarity threshold (0.0–1.0).
   * Results below this score are discarded by the Echo layer.
   */
  minScore?: number;
}

// ─── Search result ───────────────────────────────────────────────────────────

/**
 * A single ranked document returned by the Echo layer.
 *
 * This is the element type of the array returned by
 * `WasmRagEngine.query_search_array()`.
 */
export interface SearchResult {
  /** The matched document. */
  document: Document;
  /** Cosine-similarity score in the range [0.0, 1.0]. */
  score: number;
  /** Zero-based rank in the result set (0 = most similar). */
  rank: number;
}

// ─── Speculation ─────────────────────────────────────────────────────────────

/**
 * Decision emitted by the Layer-2 Speculator.
 *
 * Mirrors `SpeculationDecision` in `src/types.rs`.
 */
export type SpeculationDecision = 'Accept' | 'Revise' | 'Reject';

/** Full result from the Layer-2 Speculator. */
export interface SpeculationResult {
  decision: SpeculationDecision;
  /** Confidence in the decision (0.0–1.0). */
  confidence: number;
  /** Human-readable explanation. */
  explanation: string;
  /** Suggested rewrite when `decision === 'Revise'`. */
  suggested_revisions?: string;
  /** List of issues found in the draft. */
  issues: string[];
}

// ─── Verification ────────────────────────────────────────────────────────────

/**
 * Status of a single-claim or overall verification result.
 *
 * Mirrors `VerificationStatus` in `src/types.rs`.
 * Note: the Rust enum includes `Falsified`, `Timeout`, and `Error` variants
 * that the pipeline may emit; the JS-visible values are the serialised strings.
 */
export type VerificationStatus =
  | 'Verified'
  | 'Falsified'
  | 'Unknown'
  | 'Timeout'
  | 'Error';

/** Overall result from the Layer-3 Judge. */
export interface VerificationResult {
  status: VerificationStatus;
  /** Per-claim verification details. */
  claim_results: ClaimVerificationResult[];
  /** Overall confidence (0.0–1.0). */
  confidence: number;
  /** Human-readable summary. */
  summary: string;
  /** Wall-clock time for the full verification pass (ms). */
  total_duration_ms: number;
}

/** Verification outcome for a single extracted claim. */
export interface ClaimVerificationResult {
  claim: LogicalClaim;
  status: VerificationStatus;
  explanation?: string;
  duration_ms: number;
}

/** A logical claim extracted from the draft answer. */
export interface LogicalClaim {
  id: string;
  text: string;
  /** Structured JSON representation (varies by claim type). */
  structure: unknown;
  confidence: number;
  source_span?: [number, number];
}

// ─── Draft ───────────────────────────────────────────────────────────────────

/** The intermediate draft answer produced by the retrieval stage. */
export interface Draft {
  content: string;
  query: string;
  /** IDs of source documents used to synthesise the draft. */
  sources: string[];
  confidence: number;
  generated_at: string;
}

// ─── Pipeline output ─────────────────────────────────────────────────────────

/**
 * Complete output from the four-layer RAG pipeline.
 *
 * Returned as a JSON string by `WasmRagEngine.query()` and parsed into this
 * shape by `OxiRagEngine.query()`.
 *
 * Field names match the Serde-serialised snake_case output from Rust.
 */
export interface PipelineOutput {
  /** The original query object. */
  query: {
    text: string;
    top_k: number;
    min_score?: number;
    filters: Record<string, string>;
  };
  /** Ranked search results from the Echo layer. */
  search_results: SearchResult[];
  /** Intermediate draft answer. */
  draft: Draft;
  /** Layer-2 speculator result (absent if layer was skipped). */
  speculation?: SpeculationResult;
  /** Layer-3 judge result (absent if layer was skipped). */
  verification?: VerificationResult;
  /** Final answer string after all layers have processed the draft. */
  final_answer: string;
  /** Overall pipeline confidence (0.0–1.0). */
  confidence: number;
  /** Names of pipeline stages that ran (e.g. `["echo", "speculator", "judge"]`). */
  layers_used: string[];
  /** Total wall-clock time for the full pipeline (ms). */
  total_duration_ms: number;
}

// ─── Engine config ───────────────────────────────────────────────────────────

/** Options passed to `OxiRagEngine.create()`. */
export interface EngineConfig {
  /**
   * Embedding vector dimension.
   *
   * Must match the dimension used by the embedding model:
   * - 128 — default mock provider (fast, for testing)
   * - 384 — MiniLM-L6
   * - 768 — BERT-base / DistilBERT
   * - 1536 — OpenAI text-embedding-3-small
   *
   * @default 128
   */
  dimension?: number;
}

// ─── Worker protocol ─────────────────────────────────────────────────────────

/** Verb types accepted by the Web Worker. */
export type WorkerMessageType = 'index' | 'query' | 'count' | 'clear';

/**
 * Message sent from the main thread to the `WorkerEngine`.
 *
 * Payload fields by type:
 * - `index`: `{ content: string; title?: string }`
 * - `query`: `{ query: string; top_k?: number }`
 * - `count`: `{}`
 * - `clear`: `{}`
 */
export interface WorkerRequest {
  /** Caller-generated correlation id, echoed back in the response. */
  id: string;
  type: WorkerMessageType;
  payload: Record<string, unknown>;
}

/** Message sent from the worker back to the main thread. */
export interface WorkerResponse {
  /** Echoed correlation id from the request. */
  id: string;
  /** `true` on success, `false` on error. */
  ok: boolean;
  /** Present when `ok === true`. */
  result?: unknown;
  /** Present when `ok === false`. */
  error?: string;
}

// ─── Streaming ───────────────────────────────────────────────────────────────

/** Options for the `OxiRagEngine.searchStream()` generator. */
export interface StreamOptions {
  /**
   * Maximum number of results to yield.
   * @default 10
   */
  topK?: number;
  /**
   * Minimum score threshold; results below this are silently skipped.
   * @default 0
   */
  minScore?: number;
}
