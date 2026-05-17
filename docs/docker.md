# OxiRAG Docker Deployment Guide

This guide covers building, running, and operating OxiRAG in Docker containers.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 24.0 or later
- [Docker Compose](https://docs.docker.com/compose/install/) v2.20 or later (for the Compose stack)
- `curl` for health-check examples

---

## Building the Image

```bash
docker build -t oxirag:0.5.0 .
```

The multi-stage `Dockerfile` uses:

- **Build stage**: `rust:1.85-slim` — compiles `oxirag-server` with `--features rest-server`
- **Runtime stage**: `debian:12-slim` — only `ca-certificates` added, no Rust toolchain

Typical image sizes:
- Build stage: ~1.5 GB (discarded)
- Runtime image: < 50 MB

To verify the final image size:

```bash
docker image inspect oxirag:0.5.0 --format '{{.Size}}' | numfmt --to=iec
```

---

## Running the Server (Standalone)

```bash
docker run -d \
  -p 3000:3000 \
  --name oxirag \
  oxirag:0.5.0
```

Check that it started:

```bash
docker logs oxirag
docker ps --filter name=oxirag
```

---

## Environment Variables

All configuration is via environment variables; no config file is needed.

| Variable            | Default       | Description                                      |
|---------------------|---------------|--------------------------------------------------|
| `OXIRAG_HOST`       | `0.0.0.0`     | Bind address for the HTTP listener               |
| `OXIRAG_PORT`       | `3000`        | TCP port                                         |
| `OXIRAG_DIMENSION`  | `128`         | Embedding vector dimension for the in-memory store |
| `RUST_LOG`          | *(not set)*   | Log filter (e.g. `info`, `oxirag=debug`)         |

Example with custom settings:

```bash
docker run -d \
  -p 8080:8080 \
  -e OXIRAG_HOST=0.0.0.0 \
  -e OXIRAG_PORT=8080 \
  -e OXIRAG_DIMENSION=384 \
  -e RUST_LOG=info \
  --name oxirag \
  oxirag:0.5.0
```

---

## Health Check

The container runs an automatic Docker health check. You can also verify manually:

```bash
curl -s http://localhost:3000/health | jq .
```

Expected response:

```json
{"status":"ok","version":"0.5.0"}
```

---

## With OpenTelemetry (Jaeger)

The `docker-compose.yml` in the project root starts OxiRAG plus a Jaeger all-in-one instance:

```bash
docker compose up -d
```

Services started:

| Service  | Port  | Description                          |
|----------|-------|--------------------------------------|
| `oxirag` | 3000  | REST API                             |
| `jaeger` | 16686 | Jaeger UI                            |
| `jaeger` | 4317  | OTLP gRPC collector (OxiRAG sends here) |

Open the Jaeger UI: <http://localhost:16686>

Select service `oxirag` in the search panel to see pipeline traces for each query.

To stop:

```bash
docker compose down
```

---

## REST API Examples

All examples assume the server is running on `localhost:3000`.

### Health Check

```bash
curl -s http://localhost:3000/health
```

### Index a Document

```bash
curl -s -X POST http://localhost:3000/documents \
  -H 'Content-Type: application/json' \
  -d '{"content":"Rust is a systems programming language focused on safety, speed, and concurrency."}'
```

Response:

```json
{"id":"<uuid>"}
```

### Index Multiple Documents

```bash
curl -s -X POST http://localhost:3000/documents \
  -H 'Content-Type: application/json' \
  -d '{"content":"The Rust compiler prevents data races at compile time."}'

curl -s -X POST http://localhost:3000/documents \
  -H 'Content-Type: application/json' \
  -d '{"content":"Cargo is the Rust package manager and build system."}'
```

### Semantic Search

```bash
curl -s -X POST http://localhost:3000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is Rust?","top_k":3}'
```

Response:

```json
{
  "results": [
    {"id":"<uuid>","score":0.95,"content":"Rust is a systems programming language..."},
    {"id":"<uuid>","score":0.82,"content":"The Rust compiler prevents data races..."}
  ],
  "took_ms": 1
}
```

### End-to-End Pipeline Query

```bash
curl -s -X POST http://localhost:3000/pipeline/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the Rust package manager?","top_k":5}'
```

Response includes the retrieved context, a generated answer, and confidence score:

```json
{
  "answer": "Cargo is Rust's package manager and build system.",
  "confidence": 0.91,
  "search_results": [...],
  "layers_used": ["echo","speculator"]
}
```

### Observability Metrics

```bash
curl -s http://localhost:3000/metrics | jq .
```

Returns accumulated `SpanReport` data for all pipeline runs since startup.

---

## Persistent Storage

The default server uses an **in-memory** vector store that resets on restart. For
production deployments requiring persistence, build with the `echo-redb` feature:

```dockerfile
RUN cargo build --release --features rest-server,echo-redb --bin oxirag-server
```

Then mount a volume:

```bash
docker run -d \
  -p 3000:3000 \
  -v oxirag-data:/data \
  -e OXIRAG_STORE_PATH=/data/vectors.redb \
  --name oxirag \
  oxirag:0.5.0
```

---

## Reducing Image Size

The Dockerfile already uses `debian:12-slim` and strips the binary via the
`release-wasm` profile. To further reduce size:

1. Use `--features rest-server` only (avoids pulling in optional candle/redb deps).
2. Add `strip = true` to `[profile.release]` in `Cargo.toml` (already set for `release-wasm`).
3. Use `UPX` to compress the binary (adds ~100ms startup cost):

```dockerfile
RUN upx --best /usr/local/bin/oxirag-server
```

---

## Troubleshooting

**Container exits immediately**

```bash
docker logs oxirag
```

Check `OXIRAG_PORT` is not already in use on the host:

```bash
ss -tlnp | grep 3000
```

**Health check failing**

Ensure `curl` is available in the runtime image. The `debian:12-slim` base includes
it via `ca-certificates`. If you switch to `scratch` or `alpine`, install `curl`
explicitly.

**Tracing not appearing in Jaeger**

Verify `OTEL_EXPORTER_OTLP_ENDPOINT` points to the correct Jaeger OTLP gRPC address
and that the `otel` feature was enabled at build time:

```bash
docker run --rm oxirag:0.5.0 --version 2>&1 | grep features
```
