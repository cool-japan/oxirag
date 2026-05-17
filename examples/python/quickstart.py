"""OxiRAG Python quickstart — indexes Rust-related documents and queries them.

Run after building the extension:
    maturin develop --features python
    python examples/python/quickstart.py
"""

import asyncio

try:
    import oxirag
except ImportError as exc:
    raise SystemExit(
        "oxirag is not installed. Run: maturin develop --features python"
    ) from exc


DOCUMENTS = [
    "Rust is a systems programming language focused on safety, speed, and concurrency.",
    "Rust achieves memory safety without a garbage collector through its ownership model.",
    "The Rust borrow checker enforces ownership rules at compile time.",
    "Rust prevents data races by enforcing that shared data is either immutable or exclusively owned.",
    "Cargo is the official Rust package manager and build tool.",
]


async def main() -> None:
    print("Building pipeline (dimension=128)…")
    pipeline = oxirag.PipelineBuilder().with_dimension(128).build()
    print(pipeline)

    print("\nIndexing documents…")
    for i, text in enumerate(DOCUMENTS, start=1):
        doc = oxirag.Document(text, title=f"Rust Fact #{i}")
        doc_id = await pipeline.index(doc)
        print(f"  [{i}] indexed id={doc_id[:8]}…")

    total = await pipeline.count()
    print(f"\nTotal indexed: {total}")

    print("\nQuerying: 'What is Rust?'")
    query = oxirag.Query("What is Rust?").with_top_k(5)
    output = await pipeline.query(query)

    print(f"\nFinal answer  : {output.final_answer[:120]}")
    print(f"Confidence    : {output.confidence:.3f}")
    print(f"Layers used   : {output.layers_used}")
    print(f"Duration (ms) : {output.total_duration_ms}")

    print("\nTop search results:")
    for result in output.search_results:
        snippet = result.document.content[:80]
        print(f"  rank={result.rank}  score={result.score:.4f}  {snippet!r}")


if __name__ == "__main__":
    asyncio.run(main())
