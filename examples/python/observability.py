"""Demonstrate OxiRAG Python observability via span reports.

Run after building the extension:
    maturin develop --features python
    python examples/python/observability.py
"""

import asyncio

try:
    import oxirag
except ImportError as exc:
    raise SystemExit(
        "oxirag is not installed. Run: maturin develop --features python"
    ) from exc


DOCUMENTS = [
    "Rust uses ownership to guarantee memory safety.",
    "The Rust compiler enforces lifetime rules at compile time.",
    "Zero-cost abstractions are a core design principle of Rust.",
]


async def main() -> None:
    print("Building pipeline…")
    pipeline = oxirag.PipelineBuilder().with_dimension(64).with_fast_path(False).build()

    print("\nIndexing 3 documents…")
    ids = await pipeline.index_batch(
        [oxirag.Document(text) for text in DOCUMENTS]
    )
    print(f"  Indexed IDs: {[i[:8] + '…' for i in ids]}")

    print("\nRunning query: 'What does Rust guarantee?'")
    query = oxirag.Query("What does Rust guarantee?").with_top_k(3)
    output = await pipeline.query(query)

    print(f"  Answer    : {output.final_answer[:100]}")
    print(f"  Confidence: {output.confidence:.3f}")

    print("\nSpan report:")
    report = pipeline.span_report()
    print(f"  Total records: {len(report)}")

    col_widths = (18, 14, 12, 10)
    header = (
        f"  {'Layer':<{col_widths[0]}}"
        f"{'Duration (ms)':<{col_widths[1]}}"
        f"{'Status':<{col_widths[2]}}"
        f"{'Items':<{col_widths[3]}}"
    )
    print(header)
    print("  " + "-" * (sum(col_widths)))

    for record in report.records:
        item_count = str(record.item_count) if record.item_count is not None else "—"
        row = (
            f"  {record.layer_name:<{col_widths[0]}}"
            f"{record.duration_ms:<{col_widths[1]}}"
            f"{record.status:<{col_widths[2]}}"
            f"{item_count:<{col_widths[3]}}"
        )
        print(row)


if __name__ == "__main__":
    asyncio.run(main())
