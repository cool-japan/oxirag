//! IndexedDB-backed prefix cache for WASM targets.
//!
//! Implements [`PrefixCacheStore`] using the browser's native IndexedDB API so
//! that KV-cache entries survive page reloads in browser-based RAG deployments.
//!
//! # Design notes
//!
//! - The local `entry_count` field tracks the number of entries synchronously to
//!   avoid an async `IdbObjectStore::count()` call on every `len()` / `is_empty()`
//!   invocation.
//! - TTL expiry is checked eagerly on `get` and lazily batch-pruned by
//!   `evict_expired`.
//! - Prefix matching (`find_prefix_match`) delegates to the default trait
//!   implementation (returns `None`), keeping the implementation focused on
//!   correct persistence semantics rather than partial-hit heuristics.
//!
//! # Feature gate
//!
//! This file is compiled only when **both** `target_arch = "wasm32"` and the
//! `wasm-prefix-indexeddb` Cargo feature are active.

#![cfg(all(target_arch = "wasm32", feature = "wasm-prefix-indexeddb"))]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::error::OxiRagError;
use crate::prefix_cache::traits::PrefixCacheStore;
use crate::prefix_cache::types::{CacheKey, CacheStats, ContextFingerprint, KVCacheEntry};

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Default IndexedDB database name for the prefix cache.
const DEFAULT_DB_NAME: &str = "oxirag_prefix_cache";
/// Object-store name within the database.
const STORE_NAME: &str = "prefix_cache";
/// IDB schema version.
const DB_VERSION: u32 = 1;

// ────────────────────────────────────────────────────────────────────────────
// On-disk representation
// ────────────────────────────────────────────────────────────────────────────

/// Serialisable projection of [`KVCacheEntry`] that can survive IDB round-trips.
///
/// [`std::time::Instant`] is not serialisable, so we store the creation timestamp
/// as milliseconds-since-Unix-epoch using `js_sys::Date::now()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedKVEntry {
    /// Cache key / IDB record key.
    key: CacheKey,
    /// Fingerprint identifying the cached context.
    fingerprint_hash: u64,
    fingerprint_prefix_length: usize,
    fingerprint_content_summary: String,
    /// Cached KV data.
    kv_data: Vec<f32>,
    /// Sequence length in tokens.
    sequence_length: usize,
    /// Creation timestamp in ms since Unix epoch.
    created_at_ms: f64,
    /// Optional TTL in seconds.
    ttl_secs: Option<u64>,
}

impl PersistedKVEntry {
    fn from_entry(entry: &KVCacheEntry) -> Self {
        Self {
            key: entry.key.clone(),
            fingerprint_hash: entry.fingerprint.hash,
            fingerprint_prefix_length: entry.fingerprint.prefix_length,
            fingerprint_content_summary: entry.fingerprint.content_summary.clone(),
            kv_data: entry.kv_data.clone(),
            sequence_length: entry.sequence_length,
            created_at_ms: js_sys::Date::now(),
            ttl_secs: entry.ttl.map(|d| d.as_secs()),
        }
    }

    fn into_entry(self) -> KVCacheEntry {
        let fp = ContextFingerprint::new(
            self.fingerprint_hash,
            self.fingerprint_prefix_length,
            self.fingerprint_content_summary,
        );
        let mut e = KVCacheEntry::new(self.key, fp, self.kv_data, self.sequence_length);
        if let Some(secs) = self.ttl_secs {
            e = e.with_ttl_secs(secs);
        }
        e
    }

    /// Returns `true` when the entry has exceeded its TTL based on wall-clock time.
    fn is_expired_wall(&self) -> bool {
        let Some(ttl_secs) = self.ttl_secs else {
            return false;
        };
        let age_ms = js_sys::Date::now() - self.created_at_ms;
        let age_secs = (age_ms / 1000.0) as u64;
        age_secs >= ttl_secs
    }
}

// ────────────────────────────────────────────────────────────────────────────
// IDB helpers
// ────────────────────────────────────────────────────────────────────────────

fn js_err_to_oxirag(js: JsValue) -> OxiRagError {
    let msg = js.as_string().unwrap_or_else(|| format!("{js:?}"));
    OxiRagError::Config(format!("IndexedDB error: {msg}"))
}

/// Open (or upgrade) the prefix cache database.
async fn open_db(db_name: &str) -> Result<web_sys::IdbDatabase, OxiRagError> {
    let window = web_sys::window().ok_or_else(|| OxiRagError::Config("no Window object".into()))?;

    let idb_factory = window
        .indexed_db()
        .map_err(js_err_to_oxirag)?
        .ok_or_else(|| OxiRagError::Config("IndexedDB not available".into()))?;

    let open_request: web_sys::IdbOpenDbRequest = idb_factory
        .open_with_u32(db_name, DB_VERSION)
        .map_err(js_err_to_oxirag)?;

    // Use out-of-line key (no keyPath) — pass key explicitly in every IDB call.
    let on_upgrade: Closure<dyn FnMut(web_sys::IdbVersionChangeEvent)> =
        Closure::once(|event: web_sys::IdbVersionChangeEvent| {
            if let Some(target) = event.target() {
                let req: web_sys::IdbOpenDbRequest = target.unchecked_into();
                if let Ok(Some(result)) = req.result() {
                    let db: web_sys::IdbDatabase = result.unchecked_into();
                    if !db.object_store_names().contains(STORE_NAME) {
                        let _ = db.create_object_store(STORE_NAME);
                    }
                }
            }
        });
    open_request.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
    on_upgrade.forget();

    // Clone so we can retrieve the result after the Promise resolves.
    let open_req_clone = open_request.clone();

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        // IdbOpenDbRequest extends IdbRequest — use AsRef to call inherited callbacks.
        let req_ref: &web_sys::IdbRequest = AsRef::<web_sys::IdbRequest>::as_ref(&open_req_clone);

        let onsuccess: Closure<dyn FnMut(web_sys::Event)> =
            Closure::once(move |_: web_sys::Event| {
                let _ = resolve.call0(&JsValue::undefined());
            });
        req_ref.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
        onsuccess.forget();

        let req_ref2: &web_sys::IdbRequest = AsRef::<web_sys::IdbRequest>::as_ref(&open_req_clone);
        let onerror: Closure<dyn FnMut(web_sys::Event)> =
            Closure::once(move |_: web_sys::Event| {
                let _ = reject.call1(
                    &JsValue::undefined(),
                    &JsValue::from_str("IDB prefix cache open failed"),
                );
            });
        req_ref2.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    });

    JsFuture::from(promise).await.map_err(js_err_to_oxirag)?;

    let db: web_sys::IdbDatabase = open_request
        .result()
        .map_err(js_err_to_oxirag)?
        .ok_or_else(|| OxiRagError::Config("IDB open succeeded but result is null".into()))?
        .unchecked_into();

    Ok(db)
}

/// Await an [`web_sys::IdbRequest`] and return its result as a [`JsValue`].
async fn await_request(request: web_sys::IdbRequest) -> Result<JsValue, OxiRagError> {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let req_clone = request.clone();
        let onsuccess: Closure<dyn FnMut(web_sys::Event)> =
            Closure::once(move |_: web_sys::Event| {
                let val = req_clone.result().unwrap_or(JsValue::undefined());
                let _ = resolve.call1(&JsValue::undefined(), &val);
            });
        request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
        onsuccess.forget();

        let onerror: Closure<dyn FnMut(web_sys::Event)> =
            Closure::once(move |e: web_sys::Event| {
                let _ = reject.call1(&JsValue::undefined(), &JsValue::from(e));
            });
        request.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    });

    JsFuture::from(promise).await.map_err(js_err_to_oxirag)
}

// ────────────────────────────────────────────────────────────────────────────
// Public struct
// ────────────────────────────────────────────────────────────────────────────

/// An IndexedDB-backed [`PrefixCacheStore`] for WASM browser environments.
///
/// KV-cache entries are serialised to JSON and stored in the browser's native
/// IndexedDB, providing persistence across page reloads.
///
/// # Example
///
/// ```rust,ignore
/// use oxirag::prefix_cache::{IndexedDbPrefixCache, PrefixCacheStore, ContextFingerprint, KVCacheEntry};
///
/// let mut cache = IndexedDbPrefixCache::new(100);
/// let fp = ContextFingerprint::new(42, 10, "hello");
/// let entry = KVCacheEntry::new("k1", fp.clone(), vec![1.0; 32], 10);
/// cache.put(entry).await.expect("put ok");
/// assert!(cache.contains(&fp).await);
/// ```
pub struct IndexedDbPrefixCache {
    /// IndexedDB database name.
    db_name: String,
    /// Maximum number of entries before evictions begin.
    max_capacity: usize,
    /// Locally tracked entry count (avoids an async IDB `count()` on every call).
    entry_count: usize,
}

impl IndexedDbPrefixCache {
    /// Create a new cache backed by the default database name.
    #[must_use]
    pub fn new(max_capacity: usize) -> Self {
        Self {
            db_name: DEFAULT_DB_NAME.to_string(),
            max_capacity,
            entry_count: 0,
        }
    }

    /// Create a new cache with a custom database name (useful in tests to isolate state).
    #[must_use]
    pub fn with_db_name(db_name: impl Into<String>, max_capacity: usize) -> Self {
        Self {
            db_name: db_name.into(),
            max_capacity,
            entry_count: 0,
        }
    }

    /// Derive a cache key from a [`ContextFingerprint`].
    fn fingerprint_to_key(fp: &ContextFingerprint) -> String {
        format!("{}:{}", fp.hash, fp.prefix_length)
    }

    /// Load a [`PersistedKVEntry`] from IDB by its cache key, returning `None` if
    /// the key does not exist.
    async fn load_by_key(&self, key: &str) -> Result<Option<PersistedKVEntry>, OxiRagError> {
        let db = open_db(&self.db_name).await?;
        let tx = db
            .transaction_with_str(STORE_NAME)
            .map_err(js_err_to_oxirag)?;
        let store = tx.object_store(STORE_NAME).map_err(js_err_to_oxirag)?;
        let request = store
            .get(&JsValue::from_str(key))
            .map_err(js_err_to_oxirag)?;

        let result = await_request(request).await?;

        if result.is_undefined() || result.is_null() {
            return Ok(None);
        }

        let entry: PersistedKVEntry = serde_wasm_bindgen::from_value(result)
            .map_err(|e| OxiRagError::Config(e.to_string()))?;

        Ok(Some(entry))
    }

    /// Remove a record by its cache key (does not update `entry_count`).
    async fn remove_by_key_raw(&self, key: &str) -> Result<(), OxiRagError> {
        let db = open_db(&self.db_name).await?;
        let tx = db
            .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
            .map_err(js_err_to_oxirag)?;
        let store = tx.object_store(STORE_NAME).map_err(js_err_to_oxirag)?;
        let request = store
            .delete(&JsValue::from_str(key))
            .map_err(js_err_to_oxirag)?;
        await_request(request).await?;
        Ok(())
    }

    /// Return all persisted entries from IDB.
    async fn load_all(&self) -> Result<Vec<PersistedKVEntry>, OxiRagError> {
        let db = open_db(&self.db_name).await?;
        let tx = db
            .transaction_with_str(STORE_NAME)
            .map_err(js_err_to_oxirag)?;
        let store = tx.object_store(STORE_NAME).map_err(js_err_to_oxirag)?;
        let request = store.get_all().map_err(js_err_to_oxirag)?;
        let result = await_request(request).await?;

        if result.is_undefined() || result.is_null() {
            return Ok(Vec::new());
        }

        let array: js_sys::Array = result.unchecked_into();
        let mut entries = Vec::with_capacity(array.length() as usize);
        for i in 0..array.length() {
            let item = array.get(i);
            let entry: PersistedKVEntry = serde_wasm_bindgen::from_value(item)
                .map_err(|e| OxiRagError::Config(e.to_string()))?;
            entries.push(entry);
        }
        Ok(entries)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PrefixCacheStore impl
// ────────────────────────────────────────────────────────────────────────────

#[async_trait(?Send)]
impl PrefixCacheStore for IndexedDbPrefixCache {
    async fn get(&self, fingerprint: &ContextFingerprint) -> Option<KVCacheEntry> {
        let key = Self::fingerprint_to_key(fingerprint);
        let Ok(Some(persisted)) = self.load_by_key(&key).await else {
            return None;
        };

        // Honour wall-clock TTL.
        if persisted.is_expired_wall() {
            return None;
        }

        Some(persisted.into_entry())
    }

    async fn put(&mut self, entry: KVCacheEntry) -> Result<CacheKey, OxiRagError> {
        // Respect capacity limit — evict oldest entry if at max.
        if self.max_capacity > 0 && self.entry_count >= self.max_capacity {
            // Remove the first record we can find.
            if let Ok(all) = self.load_all().await {
                if let Some(oldest) = all.into_iter().next() {
                    self.remove_by_key_raw(&oldest.key).await?;
                    self.entry_count = self.entry_count.saturating_sub(1);
                }
            }
        }

        let key = entry.key.clone();
        let persisted = PersistedKVEntry::from_entry(&entry);

        let js_key = JsValue::from_str(&key);
        let js_val = serde_wasm_bindgen::to_value(&persisted)
            .map_err(|e| OxiRagError::Config(e.to_string()))?;

        let db = open_db(&self.db_name).await?;
        let tx = db
            .transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
            .map_err(js_err_to_oxirag)?;
        let store = tx.object_store(STORE_NAME).map_err(js_err_to_oxirag)?;
        // Use `put_with_key` because we use out-of-line keys (no keyPath in the store).
        let request = store
            .put_with_key(&js_val, &js_key)
            .map_err(js_err_to_oxirag)?;
        await_request(request).await?;

        self.entry_count += 1;

        Ok(key)
    }

    async fn remove(&mut self, key: &CacheKey) -> Option<KVCacheEntry> {
        let Ok(Some(persisted)) = self.load_by_key(key).await else {
            return None;
        };

        let entry = persisted.into_entry();

        // Best-effort delete.
        let _ = self.remove_by_key_raw(key).await;
        self.entry_count = self.entry_count.saturating_sub(1);

        Some(entry)
    }

    async fn contains(&self, fingerprint: &ContextFingerprint) -> bool {
        let key = Self::fingerprint_to_key(fingerprint);
        matches!(self.load_by_key(&key).await, Ok(Some(e)) if !e.is_expired_wall())
    }

    async fn clear(&mut self) {
        let Ok(db) = open_db(&self.db_name).await else {
            return;
        };
        let Ok(tx) =
            db.transaction_with_str_and_mode(STORE_NAME, web_sys::IdbTransactionMode::Readwrite)
        else {
            return;
        };
        let Ok(store) = tx.object_store(STORE_NAME) else {
            return;
        };
        if let Ok(request) = store.clear() {
            let _ = await_request(request).await;
        }
        self.entry_count = 0;
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.entry_count,
            ..Default::default()
        }
    }

    fn len(&self) -> usize {
        self.entry_count
    }

    fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    async fn evict_expired(&mut self) -> usize {
        let Ok(all) = self.load_all().await else {
            return 0;
        };

        let mut removed = 0;
        for entry in all {
            if entry.is_expired_wall() {
                if self.remove_by_key_raw(&entry.key).await.is_ok() {
                    removed += 1;
                }
            }
        }

        self.entry_count = self.entry_count.saturating_sub(removed);
        removed
    }

    fn memory_usage(&self) -> usize {
        // Cannot determine in-browser storage usage without async Quota API.
        // Return a per-entry estimate of 1 KiB as a conservative proxy.
        self.entry_count * 1024
    }
}
