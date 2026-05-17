# ADR-0003: REST Server as an In-Crate Feature

**Status:** Accepted
**Date:** 2026-05-17
**Deciders:** KitaSan

## Context

OxiRAG needs an HTTP API for operators who want to deploy it as a standalone
service rather than embedding the library. The REST layer needs routes for
document indexing, querying, health checks, and metrics. There are two broad
approaches:

1. A separate `oxirag-server` crate that depends on `oxirag` as a library.
2. An optional feature inside the existing `oxirag` crate.

The v0.4.0 feature set already ships a `src/rest_server.rs` with five routes and
19 tests. A server binary (`src/bin/oxirag-server.rs`) is included in the same
crate and guarded by `required-features = ["rest-server"]` in `Cargo.toml`.

## Decision

Ship the REST API as the `rest-server` optional feature within the `oxirag` crate:

```toml
# Cargo.toml
rest-server = [
    "dep:axum",
    "dep:tower",
    "dep:tower-http",
    "dep:tracing-subscriber",
    "native",
]
```

Key surface:

- `build_router(state: AppState) -> Router` — assembles all axum routes.
- `build_and_serve(addr: SocketAddr, state: AppState) -> Result<(), ServerError>` —
  binds and serves indefinitely.
- `AppState` — shared Arc-wrapped pipeline state passed to every handler.
- Routes: `GET /health`, `POST /index`, `POST /query`, `DELETE /documents/:id`,
  `GET /metrics`.

The `oxirag-server` binary is built with:

```sh
cargo build --bin oxirag-server --features rest-server --release
```

## Rationale

- Avoids workspace fragmentation: a separate crate would require its own
  `Cargo.toml`, version management, and CI matrix entry. For a single user of the
  REST feature, this overhead is not justified.
- Reuses all existing types directly: `AppState` holds a `Pipeline<...>` concrete
  type. A separate crate would need to depend on `oxirag` and construct the pipeline
  externally, adding indirection and potential API surface duplication.
- The same CI run that tests the library tests the server: `cargo nextest run
  --features rest-server` exercises both. A separate crate would require a separate
  CI job that pulls in the published `oxirag` crate (or a path dependency), which
  complicates local development.
- Consumers that embed OxiRAG as a library without the REST feature pay zero cost:
  axum, tower, and tower-http are not compiled when `rest-server` is absent.

## Consequences

- Downstream crates that enable `rest-server` pay the compilation cost of
  axum + tower + tower-http even if they only use the REST feature in a thin
  binary entrypoint. For most deployments this is acceptable: the server binary
  is the deployment artifact.
- The feature name `rest-server` intentionally uses a hyphen rather than an
  underscore to match the Cargo convention for multi-word feature names.
- Adding GRPC or GraphQL endpoints in the future would require either extending
  `rest_server.rs` (growing the file) or adding new feature-gated modules. The
  refactoring policy (max 2000 lines per file) should be applied if `rest_server.rs`
  grows beyond that threshold.
- The `native` implied feature ensures that `tokio` is available: the axum handler
  functions are `async fn` and require a multi-thread Tokio runtime.

## Alternatives Considered

### Separate `oxirag-server` crate

A dedicated workspace member `crates/oxirag-server` that depends on `oxirag`.
Rejected: adds workspace complexity (separate version, separate publish step, path
dependency in development) for a single operational user. The primary motivation
for a separate crate would be to publish the server independently or allow it to
depend on a stable `oxirag` API — neither of which applies during active v0.x
development.
