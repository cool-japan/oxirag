# Layer 4: Graph — Knowledge Graph RAG

Layer 4 (GraphRAG) enriches retrieval by building and querying a knowledge graph
over the indexed documents. While Layer 1 (Echo) finds semantically similar
documents, the Graph layer finds *structurally related* entities — following
relationships across the graph rather than just computing vector similarity.

Combining vector retrieval with graph traversal (hybrid search) often dramatically
improves recall for queries that involve multi-hop reasoning: "Which companies
acquired startups founded by Stanford graduates?" requires following
`founded_by` and `acquired_by` edges that a pure vector search cannot traverse.

## GraphRAG Concept

```
Documents
    │  index_document
    ▼
┌──────────────────┐     ┌───────────────────────┐
│ EntityExtractor  │────▶│ PatternEntityExtractor │
│ (identifies who/ │     │ MockEntityExtractor    │
│  what entities)  │     └───────────────────────┘
└──────────────────┘
    │  Vec<GraphEntity>
    ▼
┌──────────────────────┐     ┌────────────────────────────┐
│ RelationshipExtractor│────▶│ PatternRelationshipExtractor│
│ (identifies edges    │     │ MockRelationshipExtractor   │
│  between entities)   │     └────────────────────────────┘
└──────────────────────┘
    │  Vec<GraphRelationship>
    ▼
┌──────────────────┐
│   GraphStore     │  InMemoryGraphStore | RedbGraphStore
│   (persists      │
│    the graph)    │
└──────────────────┘
    │
    ▼  BFS traversal, shortest path, N-hop neighbors
  Vec<GraphPath> → merged with Echo results for hybrid retrieval
```

Relevant source files:

- `src/layer4_graph/mod.rs` — `GraphLayer` struct and `Graph` trait impl
- `src/layer4_graph/traits.rs` — `EntityExtractor`, `RelationshipExtractor`, `GraphStore` traits
- `src/layer4_graph/extractor.rs` — concrete extractors
- `src/layer4_graph/memory.rs` — `InMemoryGraphStore`
- `src/layer4_graph/redb_store.rs` — `RedbGraphStore` (feature `graphrag-redb`)
- `src/layer4_graph/traversal.rs` — `bfs_traverse`, `find_entities_within_hops`, `find_shortest_path`
- `src/layer4_graph/types.rs` — `GraphEntity`, `GraphRelationship`, `GraphPath`, etc.

## Entity and Relationship Types

```rust
use oxirag::layer4_graph::types::{
    GraphEntity, EntityType, GraphRelationship, RelationshipType, EntityId,
};

// Entity types shipped with OxiRAG:
// EntityType::Person, Technology, Organization, Location, Concept, Event, Other

let entity = GraphEntity::new("Rust", EntityType::Technology)
    .with_id("rust-lang")              // Optional stable ID (auto-generated if omitted)
    .with_attribute("year", "2015")    // Arbitrary metadata
    .with_attribute("paradigm", "systems");

// Relationship types:
// RelationshipType::Uses, Creates, IsA, PartOf, RelatedTo, Causes, Precedes,
// Located, Manages, OwnedBy, FoundedBy, DevelopedBy, Custom(String)

let rel = GraphRelationship::new("rust-lang", "llvm", RelationshipType::Uses)
    .with_weight(0.9)
    .with_attribute("via", "codegen");
```

`EntityId` is a `String` alias. If `.with_id()` is not called, a UUID is
generated automatically when the entity is inserted into a store.

## Building a Graph Layer

### Using the Builder Pattern

```rust
use oxirag::layer4_graph::{
    GraphLayer, GraphLayerBuilder,
    PatternEntityExtractor, PatternRelationshipExtractor,
    InMemoryGraphStore,
};
use oxirag::layer4_graph::traits::Graph;
use oxirag::types::Document;

let graph = GraphLayerBuilder::new()
    .with_entity_extractor(PatternEntityExtractor::new())
    .with_relationship_extractor(PatternRelationshipExtractor::new())
    .with_store(InMemoryGraphStore::new())
    .build()?;
```

`GraphLayerBuilder::build` returns `Result<GraphLayer<E, R, S>, GraphError>` and
will return an error if any required component is not configured.

### Using Constructor Directly

```rust
let mut graph = GraphLayer::new(
    PatternEntityExtractor::new(),
    PatternRelationshipExtractor::new(),
    InMemoryGraphStore::new(),
);
```

### Indexing Documents

```rust
// Single document.
let doc = Document::new(
    "Rust is a systems programming language developed by Mozilla Research. \
     It uses LLVM for code generation.",
);
graph.index_document(&doc).await?;

// Batch of documents.
let docs = vec![
    Document::new("Python was created by Guido van Rossum."),
    Document::new("TensorFlow is maintained by Google."),
    Document::new("PyTorch was developed by Facebook AI Research."),
];
graph.index_documents(&docs).await?;

println!("Entities: {}", graph.entity_count().await);
println!("Relationships: {}", graph.relationship_count().await);
```

`PatternEntityExtractor` recognises capitalized nouns, known technology names,
and organisation patterns. `PatternRelationshipExtractor` matches verb patterns
like "X uses Y", "X was created by Y", "X is part of Y".

## Pattern-Based Extractors

### PatternEntityExtractor

Identifies entities in text by matching against configurable patterns:

```rust
use oxirag::layer4_graph::PatternEntityExtractor;

let mut extractor = PatternEntityExtractor::new();
// The default configuration detects technologies, organisations, and people.
// Inspect extracted entities:
let entities = extractor.extract_entities(
    "Linus Torvalds created Linux. Rust is maintained by the Rust Foundation."
).await?;
for entity in &entities {
    println!("{} ({:?})", entity.name, entity.entity_type);
}
```

### PatternRelationshipExtractor

```rust
use oxirag::layer4_graph::{PatternRelationshipExtractor, GraphEntity};
use oxirag::layer4_graph::traits::RelationshipExtractor;

let extractor = PatternRelationshipExtractor::new();
let relationships = extractor.extract_relationships(
    "Rust uses LLVM for code generation.",
    &entities,
).await?;
for rel in &relationships {
    println!("{} --[{:?}]--> {}", rel.from_id, rel.relationship_type, rel.to_id);
}
```

### Mock Extractors (Testing)

```rust
use oxirag::layer4_graph::{
    MockEntityExtractor, MockRelationshipExtractor,
    GraphEntity, GraphRelationship, EntityType, RelationshipType,
};

let entities = vec![
    GraphEntity::new("Rust", EntityType::Technology).with_id("rust"),
    GraphEntity::new("LLVM", EntityType::Technology).with_id("llvm"),
];
let relationships = vec![
    GraphRelationship::new("rust", "llvm", RelationshipType::Uses),
];

let mut graph = GraphLayer::new(
    MockEntityExtractor::with_entities(entities),
    MockRelationshipExtractor::with_relationships(relationships),
    InMemoryGraphStore::new(),
);

graph.index_document(&Document::new("Rust uses LLVM.")).await?;
```

`MockEntityExtractor::new()` returns an empty extractor. Use `.add_entity(e)` to
populate it incrementally, or `MockEntityExtractor::with_entities(vec)` to
initialise from a list.

## Traversal APIs

All traversal functions operate on a `GraphStore` reference and return
`Vec<GraphPath>`. A `GraphPath` records the sequence of entities and relationships
traversed from the start entity to the terminal entity.

### BFS Traversal

```rust
use oxirag::layer4_graph::traversal::bfs_traverse;
use oxirag::layer4_graph::types::GraphQuery;

let store = graph.store();

// Build a query starting from "rust-lang", max 3 hops.
let query = GraphQuery::new(vec!["rust-lang".to_string()])
    .with_max_hops(3);

let paths = bfs_traverse(store, &query).await?;
for path in &paths {
    let names: Vec<&str> = path.entities.iter().map(|e| e.name.as_str()).collect();
    println!("Path: {}", names.join(" → "));
}
```

### Find Entities Within N Hops

```rust
use oxirag::layer4_graph::traversal::find_entities_within_hops;

// All entities reachable from "python" within 2 hops.
let reachable = find_entities_within_hops(store, "python", 2).await?;
for entity in &reachable {
    println!("Reachable: {} ({:?})", entity.name, entity.entity_type);
}
```

### Shortest Path

```rust
use oxirag::layer4_graph::traversal::find_shortest_path;

// Shortest path from "rust-lang" to "webassembly".
let path = find_shortest_path(store, "rust-lang", "webassembly").await?;
match path {
    Some(p) => {
        println!("Path length: {} hops", p.hops());
        for entity in &p.entities {
            println!("  {}", entity.name);
        }
    }
    None => println!("No path found"),
}
```

### Graph::find_related (Higher-Level API)

The `Graph` trait provides `find_related` for entity-name–based lookups:

```rust
use oxirag::layer4_graph::traits::Graph;

// Look up entity by name and traverse up to 2 hops.
let related_paths = graph.find_related("Rust", 2).await?;
```

This internally calls `store.find_entities_by_name("Rust")` then runs `bfs_traverse`.

### Direct Query

```rust
use oxirag::layer4_graph::traits::Graph;
use oxirag::layer4_graph::types::{GraphQuery, EntityId};

let query = GraphQuery::new(vec!["rust-lang".to_string()])
    .with_max_hops(2)
    .with_relationship_filter(vec![RelationshipType::Uses, RelationshipType::DevelopedBy]);

let paths = graph.query(&query).await?;
```

## Persistent Graph Store (Requires `graphrag-redb` Feature)

For production deployments, persist the knowledge graph across restarts using
`RedbGraphStore`.

```toml
# Cargo.toml
oxirag = { version = "0.6", features = ["graphrag-redb"] }
```

```rust
use oxirag::layer4_graph::RedbGraphStore;
use oxirag::layer4_graph::{GraphLayer, PatternEntityExtractor, PatternRelationshipExtractor};

// Open or create the graph database file.
let store = RedbGraphStore::open("./data/graph.redb")?;

let mut graph = GraphLayer::new(
    PatternEntityExtractor::new(),
    PatternRelationshipExtractor::new(),
    store,
);

// Index documents — entities and relationships are committed atomically.
graph.index_document(&doc).await?;

// The graph survives process restarts.
// Reopen the same file to continue where you left off.
let persistent_store = RedbGraphStore::open("./data/graph.redb")?;
println!("Restored {} entities", persistent_store.entity_count().await);
```

`RedbGraphStore` uses 6 redb tables:

- `ENTITIES` — entity ID → serialised `GraphEntity`
- `RELATIONSHIPS` — relationship ID → serialised `GraphRelationship`
- `OUTGOING` — entity ID → list of outgoing relationship IDs
- `INCOMING` — entity ID → list of incoming relationship IDs
- `NAME_IDX` — entity name → list of entity IDs (secondary index for name lookup)
- `TYPE_IDX` — `EntityType` → list of entity IDs (secondary index for type filter)

Each write (`add_entities`, `add_relationships`) is wrapped in a single redb
write transaction for atomicity. Reads use snapshot-consistent read transactions.

## Hybrid Search

Combine vector similarity results (Layer 1) with graph traversal results for
richer retrieval. `HybridSearchResult` fuses the two result sets:

```rust
use oxirag::layer4_graph::types::HybridSearchResult;

// Perform vector search.
let vector_results = echo.search("Rust memory safety", 10, None).await?;

// Perform graph traversal from entities mentioned in the top vector result.
let entity_name = extract_primary_entity(&vector_results[0]); // your own logic
let graph_paths = graph.find_related(&entity_name, 2).await?;

// Merge: documents appearing in both get a score boost.
let hybrid = HybridSearchResult::merge(vector_results, graph_paths);
for r in hybrid.results.iter().take(5) {
    println!("[vector={:.3} graph={:.3}] {}", r.vector_score, r.graph_score, r.content);
}
```

## Custom Entity and Relationship Extractors

Implement the `EntityExtractor` and `RelationshipExtractor` traits to plug in
your own extraction logic (NLP pipeline, LLM-based extraction, etc.):

```rust
use async_trait::async_trait;
use oxirag::layer4_graph::traits::{EntityExtractor, RelationshipExtractor};
use oxirag::layer4_graph::types::{GraphEntity, GraphRelationship, EntityType, EntityId};
use oxirag::error::GraphError;

struct MyNlpEntityExtractor {
    // ... your NLP client
}

#[async_trait]
impl EntityExtractor for MyNlpEntityExtractor {
    async fn extract_entities(&self, text: &str) -> Result<Vec<GraphEntity>, GraphError> {
        // Call your NLP service or run a local model.
        let ner_results = self.ner_client.tag(text).await
            .map_err(|e| GraphError::ExtractionError(e.to_string()))?;

        Ok(ner_results.into_iter().map(|r| {
            GraphEntity::new(r.text, map_ner_type(r.label))
        }).collect())
    }
}
```

## Performance Tips

### Memory vs. Disk

- `InMemoryGraphStore` is fastest for queries but loses data on process exit.
  Suitable for read-heavy workloads where you can re-index on startup.
- `RedbGraphStore` adds 2–5 ms overhead per `index_document` call due to ACID
  transaction commit. Query latency is comparable to in-memory for graphs up to
  ~500K entities.

### Graph Size Guidelines

| Entities | Relationships | Recommended store | BFS latency (3 hops) |
|---|---|---|---|
| < 10K | < 100K | `InMemoryGraphStore` | < 5 ms |
| 10K – 500K | < 5M | `RedbGraphStore` | 5–50 ms |
| > 500K | > 5M | External graph DB (planned) | — |

### Selective Indexing

Only index documents that are expected to contain named entities. Running the
full pipeline over boilerplate or template text wastes extractor CPU cycles
without adding useful graph structure.

```rust
// Filter documents before GraphRAG indexing.
let substantive_docs: Vec<Document> = all_docs
    .into_iter()
    .filter(|d| d.content.split_whitespace().count() > 50) // at least 50 words
    .collect();

graph.index_documents(&substantive_docs).await?;
```

### BFS Hop Limit

Prefer `max_hops = 2` or `max_hops = 3` for most queries. Setting a high hop
count (> 5) on dense graphs can cause exponential path explosion. The `GraphQuery`
hop limit is enforced strictly inside `bfs_traverse`.
