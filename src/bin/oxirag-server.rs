//! Entry point for the OxiRAG REST server binary.
//!
//! Configuration is read from environment variables:
//!
//! - `OXIRAG_HOST` — bind address (default: `0.0.0.0`)
//! - `OXIRAG_PORT` — TCP port (default: `3000`)
//! - `OXIRAG_DIMENSION` — embedding vector dimension (default: `128`)

use oxirag::rest_server::{ServerConfig, build_and_serve};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let host = std::env::var("OXIRAG_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("OXIRAG_PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(3000);
    let vector_dim = std::env::var("OXIRAG_DIMENSION")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(128);

    let config = ServerConfig {
        host,
        port,
        vector_dim,
    };

    tracing::info!(host = %config.host, port = config.port, vector_dim = config.vector_dim, "Starting OxiRAG server");

    build_and_serve(config).await
}
