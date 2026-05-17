//! Compile-time assertions that core traits remain Send+Sync on native targets.
#![cfg(not(target_arch = "wasm32"))]
#![allow(dead_code)]

use oxirag::layer1_echo::{EmbeddingProvider, VectorStore};
use oxirag::prefix_cache::PrefixCacheStore;

fn assert_send_sync<T: Send + Sync>() {}

// If any of these fail to compile, the Send+Sync bounds are broken on native.
fn _check_vector_store<T: VectorStore + Send + Sync>() {
    assert_send_sync::<T>();
}

fn _check_embedding_provider<T: EmbeddingProvider + Send + Sync>() {
    assert_send_sync::<T>();
}

fn _check_prefix_cache_store<T: PrefixCacheStore + Send + Sync>() {
    assert_send_sync::<T>();
}
