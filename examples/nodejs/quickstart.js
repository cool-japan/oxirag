// OxiRAG quickstart for Node.js
// Requires: npm run build (first-time setup)

let oxirag;
try {
    oxirag = require('../../index.js');
} catch {
    console.error('Build the native module first: npm run build');
    process.exit(1);
}

const { Document, Query, PipelineBuilder } = oxirag;

async function main() {
    const pipeline = new PipelineBuilder()
        .withDimension(128)
        .withMaxResults(5)
        .build();

    // Index documents
    const docs = [
        new Document("Rust is a systems programming language focused on safety."),
        new Document("The Rust compiler prevents data races at compile time."),
        new Document("Cargo is Rust's package manager and build system."),
    ];

    for (const doc of docs) {
        const id = await pipeline.index(doc);
        console.log(`Indexed: ${id}`);
    }

    const count = await pipeline.count();
    console.log(`\nTotal indexed: ${count}`);

    // Query
    const query = new Query("What is Rust?").withTopK(5);
    const result = await pipeline.query(query);

    console.log(`\nAnswer: ${result.finalAnswer}`);
    console.log(`Confidence: ${result.confidence.toFixed(2)}`);
    console.log(`Layers used: ${result.layersUsed.join(', ')}`);
    console.log(`Duration: ${result.totalDurationMs}ms`);
    console.log(`\nTop results:`);
    for (const r of result.searchResults) {
        const snippet = r.content.substring(0, 60);
        console.log(`  [${r.score.toFixed(3)}] rank=${r.rank} ${snippet}...`);
    }
}

main().catch(console.error);
