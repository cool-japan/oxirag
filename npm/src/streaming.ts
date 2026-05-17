/**
 * Streaming utilities for OxiRAG search results.
 *
 * These helpers wrap the `AsyncGenerator` produced by
 * `OxiRagEngine.searchStream()` with composable operators modelled after the
 * Async Iterator Helpers proposal (TC39 Stage 3) and familiar to RxJS users.
 *
 * All functions are pure (no side-effects) and tree-shakeable — import only
 * what you use.
 *
 * @example
 * ```typescript
 * import { OxiRagEngine } from './engine.js';
 * import { take, filterScore, collectStream } from './streaming.js';
 *
 * const engine = await OxiRagEngine.create();
 * const stream = engine.searchStream('memory safety', { topK: 50 });
 *
 * const top5 = await collectStream(take(filterScore(stream, 0.5), 5));
 * ```
 */

import type { SearchResult } from './types.js';

// ─── Core generator type alias ────────────────────────────────────────────────

export type SearchStream = AsyncGenerator<SearchResult, void, undefined>;

// ─── Combinators ─────────────────────────────────────────────────────────────

/**
 * Yield at most `n` items from the input stream, then stop.
 *
 * @param stream  Source `SearchResult` async generator.
 * @param n       Maximum number of items to forward.
 */
export async function* take(stream: SearchStream, n: number): SearchStream {
  if (n <= 0) return;
  let count = 0;
  for await (const result of stream) {
    yield result;
    if (++count >= n) return;
  }
}

/**
 * Skip the first `n` items from the input stream.
 *
 * @param stream  Source `SearchResult` async generator.
 * @param n       Number of leading items to discard.
 */
export async function* skip(stream: SearchStream, n: number): SearchStream {
  let skipped = 0;
  for await (const result of stream) {
    if (skipped < n) {
      skipped++;
    } else {
      yield result;
    }
  }
}

/**
 * Keep only results whose `score` is at or above the given threshold.
 *
 * @param stream    Source stream.
 * @param minScore  Minimum inclusive score (0.0–1.0).
 */
export async function* filterScore(stream: SearchStream, minScore: number): SearchStream {
  for await (const result of stream) {
    if (result.score >= minScore) {
      yield result;
    }
  }
}

/**
 * Apply a predicate function to each result and only yield those for which
 * the predicate returns `true`.
 *
 * @param stream     Source stream.
 * @param predicate  Synchronous or asynchronous filter function.
 */
export async function* filter(
  stream: SearchStream,
  predicate: (result: SearchResult) => boolean | Promise<boolean>,
): SearchStream {
  for await (const result of stream) {
    if (await predicate(result)) {
      yield result;
    }
  }
}

/**
 * Apply a mapping function to each result.
 *
 * Returns an `AsyncGenerator<T>` rather than a `SearchStream` so that
 * you can project into a different shape (e.g. extracting only `content`).
 *
 * @param stream   Source stream.
 * @param mapFn    Synchronous or asynchronous projection function.
 */
export async function* map<T>(
  stream: SearchStream,
  mapFn: (result: SearchResult) => T | Promise<T>,
): AsyncGenerator<T, void, undefined> {
  for await (const result of stream) {
    yield await mapFn(result);
  }
}

/**
 * Deduplicate results by document id, keeping only the first occurrence of
 * each document across the stream.
 *
 * Useful when the same document is indexed multiple times under different
 * titles and you want a unique set of matching documents.
 *
 * @param stream  Source stream.
 */
export async function* deduplicate(stream: SearchStream): SearchStream {
  const seen = new Set<string>();
  for await (const result of stream) {
    if (!seen.has(result.document.id)) {
      seen.add(result.document.id);
      yield result;
    }
  }
}

// ─── Terminal operators ───────────────────────────────────────────────────────

/**
 * Drain an async generator into an array.
 *
 * @param stream  Any `SearchResult` async generator (including composed ones).
 * @returns       Resolved array of all yielded results.
 */
export async function collectStream(stream: SearchStream): Promise<SearchResult[]> {
  const results: SearchResult[] = [];
  for await (const result of stream) {
    results.push(result);
  }
  return results;
}

/**
 * Run a side-effecting callback for each result and drain the stream.
 *
 * Equivalent to `for await (const r of stream) { fn(r); }` but composable
 * in a pipeline.
 *
 * @param stream  Source stream.
 * @param fn      Callback; may be async.
 */
export async function forEach(
  stream: SearchStream,
  fn: (result: SearchResult) => void | Promise<void>,
): Promise<void> {
  for await (const result of stream) {
    await fn(result);
  }
}

/**
 * Reduce a stream to a single accumulated value.
 *
 * @param stream       Source stream.
 * @param reducer      Accumulator function.
 * @param initialValue Starting accumulator value.
 */
export async function reduce<T>(
  stream: SearchStream,
  reducer: (acc: T, result: SearchResult) => T | Promise<T>,
  initialValue: T,
): Promise<T> {
  let acc = initialValue;
  for await (const result of stream) {
    acc = await reducer(acc, result);
  }
  return acc;
}

/**
 * Return the first result from a stream (or `undefined` if the stream is
 * empty).
 *
 * @param stream  Source stream.
 */
export async function first(stream: SearchStream): Promise<SearchResult | undefined> {
  for await (const result of stream) {
    return result;
  }
  return undefined;
}

/**
 * Return the highest-scoring result from a stream (or `undefined` if empty).
 *
 * Note: this drains the entire stream, so it should only be used after
 * applying a `take()` combinator when working with large indexes.
 *
 * @param stream  Source stream.
 */
export async function best(stream: SearchStream): Promise<SearchResult | undefined> {
  return reduce(
    stream,
    (acc, result) => (acc === undefined || result.score > acc.score ? result : acc),
    undefined as SearchResult | undefined,
  );
}

// ─── Utility ──────────────────────────────────────────────────────────────────

/**
 * Convert an array of `SearchResult` objects into an async generator.
 *
 * Useful for testing composed pipelines with a fixed dataset.
 *
 * @param results  Array of results to stream.
 */
export async function* fromArray(results: SearchResult[]): SearchStream {
  for (const result of results) {
    yield result;
  }
}
