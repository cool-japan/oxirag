# Build stage
FROM rust:1.85-slim AS builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for layer caching
COPY Cargo.toml ./
COPY src/bin/ src/bin/
COPY src/ src/

RUN cargo build --release --features rest-server --bin oxirag-server

# Runtime stage
FROM debian:12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/oxirag-server /usr/local/bin/oxirag-server

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${OXIRAG_PORT:-3000}/health || exit 1

ENTRYPOINT ["/usr/local/bin/oxirag-server"]
