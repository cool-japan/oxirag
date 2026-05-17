//! REST API server for OxiRAG.
//!
//! Run:
//! ```
//! cargo run --example rest_server --features rest-server
//! ```
//!
//! Then in another terminal:
//! ```sh
//! # Index a document
//! curl -X POST http://localhost:3000/documents \
//!   -H 'Content-Type: application/json' \
//!   -d '{"content":"Paris is the capital of France","title":"France"}'
//!
//! # Search
//! curl -X POST http://localhost:3000/search \
//!   -H 'Content-Type: application/json' \
//!   -d '{"query":"capital of France","top_k":3}'
//!
//! # Health check
//! curl http://localhost:3000/health
//!
//! # Metrics
//! curl http://localhost:3000/metrics
//!
//! # Full pipeline query
//! curl -X POST http://localhost:3000/pipeline/query \
//!   -H 'Content-Type: application/json' \
//!   -d '{"query":"European capitals","top_k":3}'
//! ```

#[cfg(feature = "rest-server")]
use oxirag::rest_server::{ServerConfig, build_and_serve};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "rest-server"))]
    {
        eprintln!("This example requires the `rest-server` feature.");
        eprintln!("Run: cargo run --example rest_server --features rest-server");
        std::process::exit(1);
    }

    #[cfg(feature = "rest-server")]
    {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "oxirag=info,tower_http=debug".into()),
            )
            .init();

        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 3000,
            vector_dim: 128,
        };

        println!(
            "OxiRAG REST server starting on http://{}:{}",
            config.host, config.port
        );
        println!("Endpoints:");
        println!("  GET  /health          — liveness probe");
        println!("  POST /documents       — index a new document");
        println!("  POST /search          — semantic similarity search");
        println!("  GET  /metrics         — observability span metrics");
        println!("  POST /pipeline/query  — end-to-end query with draft synthesis");

        build_and_serve(config).await
    }
}
