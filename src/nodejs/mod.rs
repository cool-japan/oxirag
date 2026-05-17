#![cfg(feature = "nodejs")]
#![allow(missing_docs)]
//! Node.js bindings for `OxiRAG` via napi-rs 2.x.
//!
//! Build with: `napi build --platform --release --features nodejs`

pub mod builder;
pub mod pipeline;
pub mod types;

pub use builder::NapiPipelineBuilder;
pub use pipeline::NapiPipeline;
pub use types::{NapiDocument, NapiQuery, NapiSearchResult};
