//! Helper operations for the redb prefix-cache backend.
//!
//! Contains TTL checks, LRU (oldest-entry) selection, capacity enforcement
//! helpers, and encode/decode utilities that are used by `store.rs`.

use redb::{Database, ReadableDatabase, ReadableTable};

use super::super::types::CacheStats;
use super::types::{CACHE_TABLE, PersistedKVEntry};
use crate::error::OxiRagError;

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

/// Serialise a [`PersistedKVEntry`] to bytes for storage.
///
/// # Errors
///
/// Returns [`OxiRagError::Config`] if JSON serialisation fails.
pub fn encode_entry(entry: &PersistedKVEntry) -> Result<Vec<u8>, OxiRagError> {
    serde_json::to_vec(entry)
        .map_err(|e| OxiRagError::Config(format!("redb serialise failed: {e}")))
}

/// Deserialise a byte slice into a [`PersistedKVEntry`].
///
/// # Errors
///
/// Returns [`OxiRagError::Config`] if the byte slice is not valid JSON.
pub fn decode_entry_static(bytes: &[u8]) -> Result<PersistedKVEntry, OxiRagError> {
    serde_json::from_slice(bytes)
        .map_err(|e| OxiRagError::Config(format!("redb deserialise failed: {e}")))
}

// ---------------------------------------------------------------------------
// Scan helpers
// ---------------------------------------------------------------------------

/// Count the live (non-expired) entries and accumulate their estimated sizes.
///
/// # Errors
///
/// Returns [`OxiRagError::Config`] if a redb transaction or iteration fails.
pub fn scan_counts(db: &Database) -> Result<(usize, usize), OxiRagError> {
    let read_txn = db
        .begin_read()
        .map_err(|e| OxiRagError::Config(format!("redb begin_read failed: {e}")))?;
    let table = read_txn
        .open_table(CACHE_TABLE)
        .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;

    let mut count = 0usize;
    let mut bytes = 0usize;

    let iter = table
        .iter()
        .map_err(|e| OxiRagError::Config(format!("redb iter failed: {e}")))?;

    for result in iter {
        let (_key, value) =
            result.map_err(|e| OxiRagError::Config(format!("redb iter item failed: {e}")))?;
        if let Ok(entry) = decode_entry_static(value.value())
            && !entry.is_expired()
        {
            count += 1;
            bytes += entry.estimated_size();
        }
    }

    Ok((count, bytes))
}

/// Return the key of the oldest entry (by `created_at_secs`) found in a
/// full table scan. Returns `None` if the table is empty.
///
/// # Errors
///
/// Returns [`OxiRagError::Config`] if a redb transaction or iteration fails.
pub fn find_oldest_key(db: &Database) -> Result<Option<String>, OxiRagError> {
    let read_txn = db
        .begin_read()
        .map_err(|e| OxiRagError::Config(format!("redb begin_read failed: {e}")))?;
    let table = read_txn
        .open_table(CACHE_TABLE)
        .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;

    let iter = table
        .iter()
        .map_err(|e| OxiRagError::Config(format!("redb iter failed: {e}")))?;

    let mut oldest_key: Option<String> = None;
    let mut oldest_ts = u64::MAX;

    for result in iter {
        let (key, value) =
            result.map_err(|e| OxiRagError::Config(format!("redb iter item failed: {e}")))?;
        if let Ok(entry) = decode_entry_static(value.value())
            && entry.created_at_secs < oldest_ts
        {
            oldest_ts = entry.created_at_secs;
            oldest_key = Some(key.value().to_owned());
        }
    }

    Ok(oldest_key)
}

/// Delete a single entry by its composite key, returning the removed entry if
/// it was present and decodable.
///
/// # Errors
///
/// Returns [`OxiRagError::Config`] if a redb transaction, table open, or
/// removal operation fails.
pub fn delete_by_raw_key(
    db: &Database,
    raw_key: &str,
) -> Result<Option<PersistedKVEntry>, OxiRagError> {
    let write_txn = db
        .begin_write()
        .map_err(|e| OxiRagError::Config(format!("redb begin_write failed: {e}")))?;
    let removed = {
        let mut table = write_txn
            .open_table(CACHE_TABLE)
            .map_err(|e| OxiRagError::Config(format!("redb open_table failed: {e}")))?;

        let existing = table
            .remove(raw_key)
            .map_err(|e| OxiRagError::Config(format!("redb remove failed: {e}")))?;

        existing.map(|guard| decode_entry_static(guard.value()).ok())
    };
    write_txn
        .commit()
        .map_err(|e| OxiRagError::Config(format!("redb commit failed: {e}")))?;

    Ok(removed.flatten())
}

/// Rebuild `stats.total_bytes` and `stats.entry_count` from a full scan.
///
/// Called after bulk operations (`clear`, `evict_expired`) that make
/// incremental tracking impractical.
///
/// # Errors
///
/// Returns [`OxiRagError::Config`] if the underlying `scan_counts` fails.
pub fn rebuild_stats(db: &Database, stats: &mut CacheStats) -> Result<(), OxiRagError> {
    let (count, bytes) = scan_counts(db)?;
    stats.update_memory(bytes, count);
    Ok(())
}
