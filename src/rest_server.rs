//! HTTP REST server for the OxiRAG pipeline.
//!
//! This module provides a standalone HTTP API over an in-memory OxiRAG Echo
//! layer backed by [`MockEmbeddingProvider`] and [`InMemoryVectorStore`].
//! It is intentionally self-contained: no external databases or model weights
//! are required to start the server.
//!
//! # Feature
//!
//! Enable with `features = ["rest-server"]`.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use oxirag::rest_server::{ServerConfig, build_and_serve};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     build_and_serve(ServerConfig::default()).await
//! }
//! ```
//!
//! # Endpoints
//!
//! | Method | Path               | Description                       |
//! |--------|--------------------|-----------------------------------|
//! | GET    | `/health`          | Health-check                      |
//! | POST   | `/documents`       | Index a new document              |
//! | POST   | `/search`          | Semantic similarity search        |
//! | GET    | `/metrics`         | Observability span metrics        |
//! | POST   | `/pipeline/query`  | End-to-end query (echo + draft)   |

#![cfg(feature = "rest-server")]
#![allow(clippy::doc_markdown)]

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

use crate::layer1_echo::embedding::MockEmbeddingProvider;
use crate::layer1_echo::storage::InMemoryVectorStore;
use crate::layer1_echo::traits::{EmbeddingProvider, IndexedDocument, VectorStore};
use crate::observability::{LayerSpanRecord, MemoryObserver, PipelineSpanContext, SpanObserver};
use crate::types::{Document, DocumentId, SearchResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the OxiRAG REST server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind address (e.g. `"127.0.0.1"` or `"0.0.0.0"`).
    pub host: String,
    /// TCP port to listen on.
    pub port: u16,
    /// Embedding dimension used by the in-memory vector store.
    pub vector_dim: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            vector_dim: 128,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared state
// ─────────────────────────────────────────────────────────────────────────────

/// Inner mutable state wrapped behind an `Arc<RwLock<…>>`.
struct InnerState {
    embedding_provider: MockEmbeddingProvider,
    vector_store: InMemoryVectorStore,
}

/// Shared application state passed to every Axum handler via
/// [`axum::extract::State`].
///
/// `Clone` is cheap — all expensive data lives behind `Arc`.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<RwLock<InnerState>>,
    observer: Arc<MemoryObserver>,
}

impl AppState {
    /// Create a new [`AppState`] with the given embedding dimension.
    #[must_use]
    pub fn new(vector_dim: usize) -> Self {
        let embedding_provider = MockEmbeddingProvider::new(vector_dim);
        let vector_store = InMemoryVectorStore::new(vector_dim);
        Self {
            inner: Arc::new(RwLock::new(InnerState {
                embedding_provider,
                vector_store,
            })),
            observer: Arc::new(MemoryObserver::new()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Request / response types
// ─────────────────────────────────────────────────────────────────────────────

/// Request body for `POST /documents`.
#[derive(Debug, Deserialize)]
pub struct AddDocumentRequest {
    /// The document body text.
    pub content: String,
    /// Optional document title.
    pub title: Option<String>,
}

/// Response body for `POST /documents`.
#[derive(Debug, Serialize)]
pub struct AddDocumentResponse {
    /// The assigned document ID.
    pub id: String,
    /// Always `"indexed"` on success.
    pub status: &'static str,
}

/// Request body for `POST /search`.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// The search query text.
    pub query: String,
    /// Maximum number of results to return (defaults to `5`).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Minimum similarity score threshold (defaults to `0.0`).
    #[serde(default)]
    pub min_score: f32,
}

fn default_top_k() -> usize {
    5
}

/// A single entry inside a [`SearchResponse`].
#[derive(Debug, Serialize)]
pub struct SearchHit {
    /// Document ID.
    pub id: String,
    /// Document title, or `""` if not set.
    pub title: String,
    /// Similarity score in `[0, 1]`.
    pub score: f32,
}

/// Response body for `POST /search`.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    /// Ordered list of search hits (best match first).
    pub results: Vec<SearchHit>,
}

/// Response body for `GET /metrics`.
#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    /// Completed layer-span records accumulated by the observer.
    pub layer_records: Vec<LayerSpanRecord>,
    /// Number of pipeline executions completed so far.
    pub pipeline_count: usize,
}

/// Request body for `POST /pipeline/query`.
#[derive(Debug, Deserialize)]
pub struct PipelineQueryRequest {
    /// The query text.
    pub query: String,
    /// Maximum number of context documents to retrieve (defaults to `3`).
    #[serde(default = "default_pipeline_top_k")]
    pub top_k: usize,
}

fn default_pipeline_top_k() -> usize {
    3
}

/// Response body for `POST /pipeline/query`.
#[derive(Debug, Serialize)]
pub struct PipelineQueryResponse {
    /// Retrieved context documents.
    pub results: Vec<SearchHit>,
    /// Synthesised draft answer built from the retrieved context.
    pub draft: String,
}

/// Uniform error body returned on server-side failures.
#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Encode an error message as a JSON `500 Internal Server Error`.
fn internal_error(msg: impl std::fmt::Display) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorBody {
            error: msg.to_string(),
        }),
    )
}

/// Convert a list of [`SearchResult`]s to [`SearchHit`]s.
fn to_search_hits(results: Vec<SearchResult>) -> Vec<SearchHit> {
    results
        .into_iter()
        .map(|r| SearchHit {
            id: r.document.id.to_string(),
            title: r.document.title.unwrap_or_default(),
            score: r.score,
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

/// `GET /health` — liveness probe.
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": "0.4.0"
    }))
}

/// `POST /documents` — embed and index a new document.
async fn add_document(
    State(state): State<AppState>,
    Json(req): Json<AddDocumentRequest>,
) -> Result<Json<AddDocumentResponse>, (StatusCode, Json<ErrorBody>)> {
    let mut ctx = PipelineSpanContext::new();
    ctx.add_observer(Arc::clone(&state.observer) as Arc<dyn SpanObserver>);
    ctx.set_attribute("endpoint", "add_document");

    let doc_id = DocumentId::new();
    let doc = {
        let mut d = Document::new(req.content.clone());
        d.id = doc_id.clone();
        if let Some(title) = req.title {
            d.title = Some(title);
        }
        d
    };

    // ── embed ──────────────────────────────────────────────────────────────
    let embedding = {
        let mut span = ctx.begin_layer("echo_embed");
        span.set_attribute("doc_id", doc_id.as_str());

        let result = {
            let guard = state.inner.read().await;
            guard.embedding_provider.embed(&req.content).await
        };

        match result {
            Ok(emb) => {
                span.success();
                emb
            }
            Err(e) => {
                let msg = e.to_string();
                span.error(msg.clone());
                ctx.finalize();
                return Err(internal_error(msg));
            }
        }
    };

    // ── index ──────────────────────────────────────────────────────────────
    {
        let span = ctx.begin_layer("echo_index");

        let result = {
            let mut guard = state.inner.write().await;
            guard
                .vector_store
                .insert(IndexedDocument::new(doc, embedding))
                .await
        };

        match result {
            Ok(()) => span.success(),
            Err(e) => {
                let msg = e.to_string();
                span.error(msg.clone());
                ctx.finalize();
                return Err(internal_error(msg));
            }
        }
    }

    ctx.finalize();

    Ok(Json(AddDocumentResponse {
        id: doc_id.to_string(),
        status: "indexed",
    }))
}

/// `POST /search` — semantic similarity search over indexed documents.
async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorBody>)> {
    let mut ctx = PipelineSpanContext::new();
    ctx.add_observer(Arc::clone(&state.observer) as Arc<dyn SpanObserver>);
    ctx.set_attribute("endpoint", "search");
    ctx.set_attribute("query", req.query.clone());

    let min_score = if req.min_score > 0.0 {
        Some(req.min_score)
    } else {
        None
    };

    // ── embed query ────────────────────────────────────────────────────────
    let query_embedding = {
        let span = ctx.begin_layer("echo_embed_query");

        let result = {
            let guard = state.inner.read().await;
            guard.embedding_provider.embed(&req.query).await
        };

        match result {
            Ok(emb) => {
                span.success();
                emb
            }
            Err(e) => {
                let msg = e.to_string();
                span.error(msg.clone());
                ctx.finalize();
                return Err(internal_error(msg));
            }
        }
    };

    // ── vector search ──────────────────────────────────────────────────────
    let hits = {
        let mut span = ctx.begin_layer("echo_search");
        span.set_attribute("top_k", req.top_k.to_string());

        let result = {
            let guard = state.inner.read().await;
            guard
                .vector_store
                .search(&query_embedding, req.top_k, min_score)
                .await
        };

        match result {
            Ok(results) => {
                span.set_item_count(results.len());
                span.success();
                to_search_hits(results)
            }
            Err(e) => {
                let msg = e.to_string();
                span.error(msg.clone());
                ctx.finalize();
                return Err(internal_error(msg));
            }
        }
    };

    ctx.finalize();

    Ok(Json(SearchResponse { results: hits }))
}

/// `GET /metrics` — return accumulated span metrics from the [`MemoryObserver`].
async fn metrics(State(state): State<AppState>) -> Json<MetricsResponse> {
    let records = state.observer.records();
    let pipeline_count = state.observer.pipeline_snapshots().len();
    Json(MetricsResponse {
        layer_records: records,
        pipeline_count,
    })
}

/// `POST /pipeline/query` — retrieve context then synthesise a draft answer.
///
/// Performs a semantic search (Echo layer) and concatenates the retrieved
/// passages into a deterministic draft.  In a full deployment you would route
/// the context through the Speculator / Judge layers and a language model.
async fn query_pipeline(
    State(state): State<AppState>,
    Json(req): Json<PipelineQueryRequest>,
) -> Result<Json<PipelineQueryResponse>, (StatusCode, Json<ErrorBody>)> {
    let mut ctx = PipelineSpanContext::new();
    ctx.add_observer(Arc::clone(&state.observer) as Arc<dyn SpanObserver>);
    ctx.set_attribute("endpoint", "pipeline_query");
    ctx.set_attribute("query", req.query.clone());

    // ── Layer 1: embed query ───────────────────────────────────────────────
    let query_embedding = {
        let span = ctx.begin_layer("echo_embed_query");

        let result = {
            let guard = state.inner.read().await;
            guard.embedding_provider.embed(&req.query).await
        };

        match result {
            Ok(emb) => {
                span.success();
                emb
            }
            Err(e) => {
                let msg = e.to_string();
                span.error(msg.clone());
                ctx.finalize();
                return Err(internal_error(msg));
            }
        }
    };

    // ── Layer 1: vector search ─────────────────────────────────────────────
    let search_results = {
        let mut span = ctx.begin_layer("echo_search");
        span.set_attribute("top_k", req.top_k.to_string());

        let result = {
            let guard = state.inner.read().await;
            guard
                .vector_store
                .search(&query_embedding, req.top_k, None)
                .await
        };

        match result {
            Ok(results) => {
                span.set_item_count(results.len());
                span.success();
                results
            }
            Err(e) => {
                let msg = e.to_string();
                span.error(msg.clone());
                ctx.finalize();
                return Err(internal_error(msg));
            }
        }
    };

    // ── Draft synthesis ────────────────────────────────────────────────────
    let draft = {
        let mut span = ctx.begin_layer("draft_synthesis");

        let draft_text = if search_results.is_empty() {
            format!("No relevant context found for query: \"{}\"", req.query)
        } else {
            let passages: Vec<String> = search_results
                .iter()
                .enumerate()
                .map(|(i, r)| format!("[{}] (score={:.3}) {}", i + 1, r.score, r.document.content))
                .collect();
            format!(
                "Synthesized answer based on context:\n\nQuery: {}\n\nContext passages:\n{}",
                req.query,
                passages.join("\n")
            )
        };

        span.set_item_count(search_results.len());
        span.success();
        draft_text
    };

    ctx.finalize();

    let hits = to_search_hits(search_results);

    Ok(Json(PipelineQueryResponse {
        results: hits,
        draft,
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Router
// ─────────────────────────────────────────────────────────────────────────────

/// Build the Axum [`Router`] for the OxiRAG REST API.
///
/// The returned router is **not** bound to a port.  Callers may `.merge()` it
/// into a larger application or pass it directly to `axum::serve`.
///
/// # Example
///
/// ```rust,ignore
/// let state = AppState::new(128);
/// let app = build_router(state);
/// let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
/// axum::serve(listener, app).await?;
/// ```
pub fn build_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health))
        .route("/documents", post(add_document))
        .route("/search", post(search))
        .route("/metrics", get(metrics))
        .route("/pipeline/query", post(query_pipeline))
        .with_state(state)
        .layer(cors)
}

/// Start the OxiRAG REST server, blocking until the process is terminated.
///
/// # Errors
///
/// Returns an error if the TCP listener cannot be bound or if the server
/// encounters a fatal I/O error.
pub async fn build_and_serve(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState::new(config.vector_dim);
    let app = build_router(state);
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::json;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        AppState::new(128)
    }

    // ── helpers ──────────────────────────────────────────────────────────

    async fn post_json(
        app: Router,
        uri: &str,
        body: serde_json::Value,
    ) -> axum::response::Response {
        let req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builder");
        app.oneshot(req).await.expect("oneshot failed")
    }

    async fn get_req(app: Router, uri: &str) -> axum::response::Response {
        let req = Request::builder()
            .uri(uri)
            .body(Body::empty())
            .expect("request builder");
        app.oneshot(req).await.expect("oneshot failed")
    }

    async fn resp_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        serde_json::from_slice(&bytes).expect("json parse")
    }

    // ── individual endpoint tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = build_router(test_state());
        let resp = get_req(app, "/health").await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_returns_version() {
        let app = build_router(test_state());
        let resp = get_req(app, "/health").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp_json(resp).await;
        assert_eq!(val["status"], "ok");
        assert_eq!(val["version"], "0.4.0");
    }

    #[tokio::test]
    async fn test_add_document_returns_ok() {
        let app = build_router(test_state());
        let body = json!({"content": "The Eiffel Tower is in Paris", "title": "Eiffel"});
        let resp = post_json(app, "/documents", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_add_document_returns_id_and_status() {
        let app = build_router(test_state());
        let body = json!({"content": "Rome is the capital of Italy"});
        let resp = post_json(app, "/documents", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp_json(resp).await;
        assert!(val["id"].is_string());
        assert!(!val["id"].as_str().unwrap().is_empty());
        assert_eq!(val["status"], "indexed");
    }

    #[tokio::test]
    async fn test_search_on_empty_store() {
        let app = build_router(test_state());
        let body = json!({"query": "capital of France", "top_k": 3});
        let resp = post_json(app, "/search", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp_json(resp).await;
        assert!(val["results"].is_array());
    }

    #[tokio::test]
    async fn test_add_document_and_search() {
        let state = test_state();

        // Index a document via a clone of the state.
        let add_resp = post_json(
            build_router(state.clone()),
            "/documents",
            json!({"content": "The Eiffel Tower is in Paris", "title": "Eiffel"}),
        )
        .await;
        assert_eq!(add_resp.status(), StatusCode::OK);

        // Search using the same shared state.
        let search_resp = post_json(
            build_router(state),
            "/search",
            json!({"query": "Eiffel", "top_k": 1}),
        )
        .await;
        assert_eq!(search_resp.status(), StatusCode::OK);

        let val = resp_json(search_resp).await;
        assert!(val["results"].is_array());
    }

    #[tokio::test]
    async fn test_search_with_min_score() {
        let app = build_router(test_state());
        let body = json!({"query": "Rust programming", "top_k": 5, "min_score": 0.5});
        let resp = post_json(app, "/search", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_empty() {
        let app = build_router(test_state());
        let resp = get_req(app, "/metrics").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp_json(resp).await;
        assert!(val["layer_records"].is_array());
        assert_eq!(val["pipeline_count"], 0);
    }

    #[tokio::test]
    async fn test_metrics_accumulates_records() {
        let state = test_state();

        // Index something so spans are recorded.
        post_json(
            build_router(state.clone()),
            "/documents",
            json!({"content": "Berlin is the capital of Germany"}),
        )
        .await;

        // Metrics should now have at least one record.
        let metrics_resp = get_req(build_router(state), "/metrics").await;
        assert_eq!(metrics_resp.status(), StatusCode::OK);
        let val = resp_json(metrics_resp).await;
        assert!(val["layer_records"].is_array());
        let records = val["layer_records"].as_array().expect("array");
        assert!(!records.is_empty());
    }

    #[tokio::test]
    async fn test_pipeline_query_endpoint() {
        let app = build_router(test_state());
        let body = json!({"query": "What is the capital of France?", "top_k": 3});
        let resp = post_json(app, "/pipeline/query", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp_json(resp).await;
        assert!(val["results"].is_array());
        assert!(val["draft"].is_string());
        assert!(!val["draft"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_pipeline_query_draft_contains_query() {
        let app = build_router(test_state());
        let query_text = "What is Rust?";
        let body = json!({"query": query_text, "top_k": 3});
        let resp = post_json(app, "/pipeline/query", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp_json(resp).await;
        let draft = val["draft"].as_str().expect("draft string");
        assert!(draft.contains(query_text));
    }

    #[tokio::test]
    async fn test_pipeline_query_with_indexed_docs() {
        let state = test_state();

        // Index context documents.
        for (content, title) in [
            ("Paris is the capital of France", "France"),
            ("Berlin is the capital of Germany", "Germany"),
        ] {
            post_json(
                build_router(state.clone()),
                "/documents",
                json!({"content": content, "title": title}),
            )
            .await;
        }

        let resp = post_json(
            build_router(state),
            "/pipeline/query",
            json!({"query": "European capitals", "top_k": 2}),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp_json(resp).await;
        assert!(val["results"].is_array());
        assert!(val["draft"].is_string());
    }

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let app = build_router(test_state());
        let req = Request::builder()
            .uri("/nonexistent")
            .body(Body::empty())
            .expect("request builder");
        let resp = app.oneshot(req).await.expect("oneshot failed");
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_default_top_k_is_applied() {
        let app = build_router(test_state());
        // Omit top_k — should fall back to default (5).
        let body = json!({"query": "test query"});
        let resp = post_json(app, "/search", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_add_document_without_title() {
        let app = build_router(test_state());
        let body = json!({"content": "Content without a title"});
        let resp = post_json(app, "/documents", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let val = resp_json(resp).await;
        assert_eq!(val["status"], "indexed");
    }

    #[tokio::test]
    async fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 3000);
        assert_eq!(config.vector_dim, 128);
    }

    #[tokio::test]
    async fn test_search_hit_fields() {
        let state = test_state();

        let add_resp = post_json(
            build_router(state.clone()),
            "/documents",
            json!({"content": "Tokyo is the capital of Japan", "title": "Japan"}),
        )
        .await;
        assert_eq!(add_resp.status(), StatusCode::OK);

        let search_resp = post_json(
            build_router(state),
            "/search",
            json!({"query": "Tokyo", "top_k": 1}),
        )
        .await;
        assert_eq!(search_resp.status(), StatusCode::OK);
        let val = resp_json(search_resp).await;
        let results = val["results"].as_array().expect("array");
        // If indexed doc is returned, check structure.
        if !results.is_empty() {
            let hit = &results[0];
            assert!(hit["id"].is_string());
            assert!(hit["title"].is_string());
            assert!(hit["score"].is_number());
        }
    }

    #[tokio::test]
    async fn test_pipeline_query_default_top_k() {
        let app = build_router(test_state());
        // Omit top_k — should default to 3.
        let body = json!({"query": "test"});
        let resp = post_json(app, "/pipeline/query", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_counts_pipeline_after_pipeline_query() {
        let state = test_state();

        post_json(
            build_router(state.clone()),
            "/pipeline/query",
            json!({"query": "anything", "top_k": 1}),
        )
        .await;

        let metrics_resp = get_req(build_router(state), "/metrics").await;
        assert_eq!(metrics_resp.status(), StatusCode::OK);
        let val = resp_json(metrics_resp).await;
        // At least some records from the pipeline query.
        let records = val["layer_records"].as_array().expect("array");
        assert!(!records.is_empty());
    }
}
