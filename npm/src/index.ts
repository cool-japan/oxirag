/**
 * @cool-japan/oxirag-wasm — public API
 *
 * Re-exports all public symbols so consumers only need one import path:
 *
 * ```typescript
 * import {
 *   OxiRagEngine,
 *   WorkerEngine,
 *   take, filterScore, collectStream,
 *   type PipelineOutput, type SearchResult,
 * } from '@cool-japan/oxirag-wasm';
 * ```
 */

// ─── Engine classes ───────────────────────────────────────────────────────────

export { OxiRagEngine } from './engine.js';
export { WorkerEngine } from './worker.js';

// ─── Streaming combinators and terminal operators ─────────────────────────────

export {
  // Combinators
  take,
  skip,
  filterScore,
  filter,
  map,
  deduplicate,
  // Terminal operators
  collectStream,
  forEach,
  reduce,
  first,
  best,
  // Utility
  fromArray,
} from './streaming.js';
export type { SearchStream } from './streaming.js';

// ─── TypeScript types ─────────────────────────────────────────────────────────

export type {
  // Core domain types
  Document,
  Draft,
  SearchResult,
  // Query API
  QueryOptions,
  StreamOptions,
  // Speculation (Layer 2)
  SpeculationDecision,
  SpeculationResult,
  // Verification (Layer 3)
  VerificationStatus,
  VerificationResult,
  ClaimVerificationResult,
  LogicalClaim,
  // Pipeline output
  PipelineOutput,
  // Configuration
  EngineConfig,
  // Worker protocol
  WorkerMessageType,
  WorkerRequest,
  WorkerResponse,
} from './types.js';
