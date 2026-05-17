//! File I/O helpers for the persistent prefix-cache backend.
//!
//! Provides low-level read/write operations on the binary data file and
//! JSON index file.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{CacheIndex, IndexEntry, PersistedEntry};
use crate::error::OxiRagError;

// ---------------------------------------------------------------------------
// Data file operations
// ---------------------------------------------------------------------------

/// Append `entry` to the data file at `data_file_path`, returning the
/// `(file_offset, total_bytes_written)` pair where `total_bytes_written`
/// includes the 8-byte length prefix.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or the write fails.
///
/// # Panics
///
/// Panics if the `dirty_flag` lock is poisoned.
pub fn append_entry(
    data_file_path: &Path,
    entry: &PersistedEntry,
    dirty_flag: &std::sync::RwLock<bool>,
) -> Result<(u64, usize), OxiRagError> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(data_file_path)?;

    let offset = file.seek(SeekFrom::End(0))?;

    // Serialize entry to JSON with length prefix for easier reading.
    let data = serde_json::to_vec(entry)?;
    let size = data.len();

    // Write length prefix for easier reading
    let len_bytes = (size as u64).to_le_bytes();
    file.write_all(&len_bytes)?;
    file.write_all(&data)?;

    *dirty_flag.write().expect("lock poisoned") = true;

    Ok((offset, size + 8)) // +8 for length prefix
}

/// Read and deserialise the entry at the given byte `offset` in `data_file_path`.
///
/// # Errors
///
/// Returns an error if the file cannot be read or deserialisation fails.
#[allow(clippy::cast_possible_truncation)]
pub fn read_entry(data_file_path: &Path, offset: u64) -> Result<PersistedEntry, OxiRagError> {
    let mut file = File::open(data_file_path)?;
    file.seek(SeekFrom::Start(offset))?;

    // Read length prefix
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    // Note: On 32-bit systems this could truncate, but we're targeting 64-bit
    let size = u64::from_le_bytes(len_bytes) as usize;

    // Read entry data
    let mut data = vec![0u8; size];
    file.read_exact(&mut data)?;

    let entry: PersistedEntry = serde_json::from_slice(&data)?;
    Ok(entry)
}

// ---------------------------------------------------------------------------
// Index file operations
// ---------------------------------------------------------------------------

/// Deserialise the index from `index_file_path`.
///
/// Returns `None` if the file does not exist yet (first open).
///
/// # Errors
///
/// Returns an error if the file exists but cannot be parsed.
pub fn load_index(index_file_path: &Path) -> Result<Option<CacheIndex>, OxiRagError> {
    if !index_file_path.exists() {
        return Ok(None);
    }
    let file = File::open(index_file_path)?;
    let reader = BufReader::new(file);
    let index: CacheIndex = serde_json::from_reader(reader)?;
    Ok(Some(index))
}

/// Serialise `index` to `index_file_path` and clear the dirty flag.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
///
/// # Panics
///
/// Panics if the `dirty_flag` lock is poisoned.
pub fn save_index(
    index_file_path: &Path,
    index: &CacheIndex,
    dirty_flag: &std::sync::RwLock<bool>,
) -> Result<(), OxiRagError> {
    let file = File::create(index_file_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, index)?;
    *dirty_flag.write().expect("lock poisoned") = false;
    Ok(())
}

// ---------------------------------------------------------------------------
// Compaction helper
// ---------------------------------------------------------------------------

/// Rewrite the data file keeping only the entries described by `valid_pairs`,
/// and return a fresh [`CacheIndex`] reflecting the new layout.
///
/// `valid_pairs` is a list of `(cache_key, PersistedEntry)` to keep.
///
/// # Errors
///
/// Returns an error if I/O fails at any stage.
#[allow(clippy::cast_possible_truncation)]
pub fn rewrite_data_file(
    data_file_path: &Path,
    valid_pairs: Vec<(String, PersistedEntry)>,
) -> Result<CacheIndex, OxiRagError> {
    let temp_data = data_file_path.with_extension("tmp");
    let mut new_index = CacheIndex::new();

    {
        let mut file = File::create(&temp_data)?;

        for (key, entry) in valid_pairs {
            let offset = file.stream_position()?;
            let data = serde_json::to_vec(&entry)?;
            let size = data.len();

            let len_bytes = (size as u64).to_le_bytes();
            file.write_all(&len_bytes)?;
            file.write_all(&data)?;

            let idx_entry = IndexEntry {
                fingerprint_hash: entry.fingerprint_hash,
                file_offset: offset,
                entry_size: size + 8,
                created_at_unix: entry.created_at_unix,
                ttl_secs: entry.ttl_secs,
            };
            new_index.add(key, idx_entry);
        }

        new_index.last_compaction = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        );
    }

    // Atomic rename
    fs::rename(&temp_data, data_file_path)?;
    Ok(new_index)
}
