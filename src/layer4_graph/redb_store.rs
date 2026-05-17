//! Persistent knowledge-graph store backed by `redb` — an embedded, ACID-compliant
//! key-value store written in pure Rust.
//!
//! Six tables are maintained in a single database file:
//!
//! | Table        | Key              | Value                            |
//! |--------------|------------------|----------------------------------|
//! | `entities`   | entity id        | JSON `GraphEntity`               |
//! | `relationships` | rel id        | JSON `GraphRelationship`         |
//! | `outgoing`   | entity id        | JSON `Vec<String>` (rel ids)     |
//! | `incoming`   | entity id        | JSON `Vec<String>` (rel ids)     |
//! | `name_idx`   | lowercase name   | JSON `Vec<String>` (entity ids)  |
//! | `type_idx`   | entity type str  | JSON `Vec<String>` (entity ids)  |
//!
//! All writes are committed atomically before returning so the store is durable
//! even if the process terminates immediately after a write.

#![cfg(feature = "graphrag-redb")]

use std::collections::{HashSet, VecDeque};
use std::path::Path;

use async_trait::async_trait;
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};

use crate::error::GraphError;
use crate::layer4_graph::traits::GraphStore;
use crate::layer4_graph::types::{
    Direction, EntityId, EntityType, GraphEntity, GraphPath, GraphQuery, GraphRelationship,
};

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

/// Stores serialised [`GraphEntity`] values keyed by entity id.
const ENTITIES: TableDefinition<&str, &[u8]> = TableDefinition::new("entities");
/// Stores serialised [`GraphRelationship`] values keyed by relationship id.
const RELATIONSHIPS: TableDefinition<&str, &[u8]> = TableDefinition::new("relationships");
/// Maps entity id → JSON `Vec<String>` of outgoing relationship ids.
const OUTGOING: TableDefinition<&str, &[u8]> = TableDefinition::new("outgoing");
/// Maps entity id → JSON `Vec<String>` of incoming relationship ids.
const INCOMING: TableDefinition<&str, &[u8]> = TableDefinition::new("incoming");
/// Maps `entity.name.to_lowercase()` → JSON `Vec<String>` of entity ids.
const NAME_IDX: TableDefinition<&str, &[u8]> = TableDefinition::new("name_idx");
/// Maps `entity_type.to_string()` → JSON `Vec<String>` of entity ids.
const TYPE_IDX: TableDefinition<&str, &[u8]> = TableDefinition::new("type_idx");

/// Maximum number of graph paths returned by [`RedbGraphStore::traverse`].
const MAX_TRAVERSE_PATHS: usize = 100;

// ---------------------------------------------------------------------------
// Helper macro for mapping redb errors
// ---------------------------------------------------------------------------

macro_rules! storage_err {
    ($e:expr) => {
        GraphError::StorageError($e.to_string())
    };
}

// ---------------------------------------------------------------------------
// Store struct
// ---------------------------------------------------------------------------

/// A knowledge-graph store that persists all entities, relationships, and
/// index structures to disk using [`redb`](https://docs.rs/redb).
///
/// # Persistence guarantee
///
/// Every successful mutation ([`add_entity`](RedbGraphStore::add_entity),
/// [`add_relationship`](RedbGraphStore::add_relationship),
/// [`clear`](RedbGraphStore::clear)) commits a `redb` write transaction before
/// returning, so data survives process restarts.
///
/// # Thread safety
///
/// `redb::Database` is `Send + Sync`.  The store can therefore be used behind
/// an `Arc<Mutex<…>>` when shared across async tasks.
pub struct RedbGraphStore {
    /// The underlying redb database handle.
    db: Database,
    /// In-memory cache of the entity count; kept in sync with the database.
    entity_count: usize,
    /// In-memory cache of the relationship count; kept in sync with the database.
    rel_count: usize,
}

impl RedbGraphStore {
    /// Open (or create) a [`RedbGraphStore`] at the given filesystem path.
    ///
    /// When the file does not exist it is created and all six tables are
    /// initialised.  When the file already exists, pre-existing data is
    /// preserved and the cached counts are initialised by reading the table
    /// lengths.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] when the database cannot be opened
    /// or the initial table setup fails.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, GraphError> {
        let db = Database::create(path).map_err(|e| storage_err!(e))?;

        // Ensure all six tables exist (creates them on first open).
        {
            let write_txn = db.begin_write().map_err(|e| storage_err!(e))?;
            {
                let _e = write_txn
                    .open_table(ENTITIES)
                    .map_err(|e| storage_err!(e))?;
                let _r = write_txn
                    .open_table(RELATIONSHIPS)
                    .map_err(|e| storage_err!(e))?;
                let _o = write_txn
                    .open_table(OUTGOING)
                    .map_err(|e| storage_err!(e))?;
                let _i = write_txn
                    .open_table(INCOMING)
                    .map_err(|e| storage_err!(e))?;
                let _n = write_txn
                    .open_table(NAME_IDX)
                    .map_err(|e| storage_err!(e))?;
                let _t = write_txn
                    .open_table(TYPE_IDX)
                    .map_err(|e| storage_err!(e))?;
            }
            write_txn.commit().map_err(|e| storage_err!(e))?;
        }

        // Read the counts from the existing tables.
        let entity_count = {
            let read_txn = db.begin_read().map_err(|e| storage_err!(e))?;
            let table = read_txn.open_table(ENTITIES).map_err(|e| storage_err!(e))?;
            usize::try_from(table.len().map_err(|e| storage_err!(e))?).unwrap_or(0)
        };

        let rel_count = {
            let read_txn = db.begin_read().map_err(|e| storage_err!(e))?;
            let table = read_txn
                .open_table(RELATIONSHIPS)
                .map_err(|e| storage_err!(e))?;
            usize::try_from(table.len().map_err(|e| storage_err!(e))?).unwrap_or(0)
        };

        Ok(Self {
            db,
            entity_count,
            rel_count,
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Read a JSON-encoded `Vec<String>` from a table at `key`.
    ///
    /// Returns an empty [`HashSet`] when the key does not exist.
    fn read_string_set(
        &self,
        table_def: TableDefinition<&str, &[u8]>,
        key: &str,
    ) -> Result<HashSet<String>, GraphError> {
        let read_txn = self.db.begin_read().map_err(|e| storage_err!(e))?;
        let table = read_txn
            .open_table(table_def)
            .map_err(|e| storage_err!(e))?;
        match table.get(key).map_err(|e| storage_err!(e))? {
            Some(guard) => {
                let vec: Vec<String> =
                    serde_json::from_slice(guard.value()).map_err(|e| storage_err!(e))?;
                Ok(vec.into_iter().collect())
            }
            None => Ok(HashSet::new()),
        }
    }

    /// Load and deserialise a single [`GraphEntity`] from the database.
    ///
    /// Returns `None` when no entity with the given id exists.
    fn load_entity(&self, id: &str) -> Result<Option<GraphEntity>, GraphError> {
        let read_txn = self.db.begin_read().map_err(|e| storage_err!(e))?;
        let table = read_txn.open_table(ENTITIES).map_err(|e| storage_err!(e))?;
        match table.get(id).map_err(|e| storage_err!(e))? {
            Some(guard) => {
                let entity: GraphEntity =
                    serde_json::from_slice(guard.value()).map_err(|e| storage_err!(e))?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Load and deserialise a single [`GraphRelationship`] from the database.
    ///
    /// Returns `None` when no relationship with the given id exists.
    fn load_relationship(&self, id: &str) -> Result<Option<GraphRelationship>, GraphError> {
        let read_txn = self.db.begin_read().map_err(|e| storage_err!(e))?;
        let table = read_txn
            .open_table(RELATIONSHIPS)
            .map_err(|e| storage_err!(e))?;
        match table.get(id).map_err(|e| storage_err!(e))? {
            Some(guard) => {
                let rel: GraphRelationship =
                    serde_json::from_slice(guard.value()).map_err(|e| storage_err!(e))?;
                Ok(Some(rel))
            }
            None => Ok(None),
        }
    }

    /// Collect all relationship ids from the given adjacency-list table entry,
    /// then load the paired relationship and neighbour entity for each.
    ///
    /// When `direction` is [`Direction::Outgoing`] the `neighbour_id_fn` should
    /// extract `rel.target_id`; for [`Direction::Incoming`] it should extract
    /// `rel.source_id`.
    fn collect_neighbors_for_direction(
        &self,
        entity_id: &EntityId,
        adj_table: TableDefinition<&str, &[u8]>,
        neighbour_id_fn: impl Fn(&GraphRelationship) -> &EntityId,
    ) -> Result<Vec<(GraphRelationship, GraphEntity)>, GraphError> {
        let rel_ids = self.read_string_set(adj_table, entity_id.as_str())?;
        let mut result = Vec::new();
        for rel_id in &rel_ids {
            let Some(rel) = self.load_relationship(rel_id)? else {
                continue;
            };
            let neighbour_id = neighbour_id_fn(&rel).clone();
            let Some(entity) = self.load_entity(&neighbour_id)? else {
                continue;
            };
            result.push((rel, entity));
        }
        Ok(result)
    }

    /// Write all table-initialisation and entity/index rows within a single
    /// write transaction.  Used by [`add_entity`](Self::add_entity) so the
    /// ENTITIES row, `NAME_IDX` update, and `TYPE_IDX` update are atomic.
    fn persist_entity_atomic(&self, entity: &GraphEntity) -> Result<(), GraphError> {
        let entity_bytes = serde_json::to_vec(entity).map_err(|e| storage_err!(e))?;

        // --  name index: read current set, add new id, serialise  ----------
        let name_lower = entity.name.to_lowercase();
        let name_set = {
            let mut s = self.read_string_set(NAME_IDX, &name_lower)?;
            s.insert(entity.id.clone());
            s
        };
        let name_bytes = {
            let v: Vec<&String> = name_set.iter().collect();
            serde_json::to_vec(&v).map_err(|e| storage_err!(e))?
        };

        // --  type index  ----------------------------------------------------
        let type_key = entity.entity_type.to_string();
        let type_set = {
            let mut s = self.read_string_set(TYPE_IDX, &type_key)?;
            s.insert(entity.id.clone());
            s
        };
        let type_bytes = {
            let v: Vec<&String> = type_set.iter().collect();
            serde_json::to_vec(&v).map_err(|e| storage_err!(e))?
        };

        // --  empty adjacency lists (only written if not already present)  ---
        let empty: Vec<String> = Vec::new();
        let empty_bytes = serde_json::to_vec(&empty).map_err(|e| storage_err!(e))?;

        // --  single atomic write transaction  --------------------------------
        let write_txn = self.db.begin_write().map_err(|e| storage_err!(e))?;
        {
            // entities
            let mut tbl = write_txn
                .open_table(ENTITIES)
                .map_err(|e| storage_err!(e))?;
            tbl.insert(entity.id.as_str(), entity_bytes.as_slice())
                .map_err(|e| storage_err!(e))?;
        }
        {
            // name index
            let mut tbl = write_txn
                .open_table(NAME_IDX)
                .map_err(|e| storage_err!(e))?;
            tbl.insert(name_lower.as_str(), name_bytes.as_slice())
                .map_err(|e| storage_err!(e))?;
        }
        {
            // type index
            let mut tbl = write_txn
                .open_table(TYPE_IDX)
                .map_err(|e| storage_err!(e))?;
            tbl.insert(type_key.as_str(), type_bytes.as_slice())
                .map_err(|e| storage_err!(e))?;
        }
        {
            // outgoing adjacency (initialise only if absent)
            let mut tbl = write_txn
                .open_table(OUTGOING)
                .map_err(|e| storage_err!(e))?;
            if tbl
                .get(entity.id.as_str())
                .map_err(|e| storage_err!(e))?
                .is_none()
            {
                tbl.insert(entity.id.as_str(), empty_bytes.as_slice())
                    .map_err(|e| storage_err!(e))?;
            }
        }
        {
            // incoming adjacency (initialise only if absent)
            let mut tbl = write_txn
                .open_table(INCOMING)
                .map_err(|e| storage_err!(e))?;
            if tbl
                .get(entity.id.as_str())
                .map_err(|e| storage_err!(e))?
                .is_none()
            {
                tbl.insert(entity.id.as_str(), empty_bytes.as_slice())
                    .map_err(|e| storage_err!(e))?;
            }
        }
        write_txn.commit().map_err(|e| storage_err!(e))?;
        Ok(())
    }

    /// Persist a relationship and update both adjacency lists in a single
    /// write transaction.
    fn persist_relationship_atomic(
        &self,
        relationship: &GraphRelationship,
        updated_outgoing: &HashSet<String>,
        updated_incoming: &HashSet<String>,
    ) -> Result<(), GraphError> {
        let rel_bytes = serde_json::to_vec(relationship).map_err(|e| storage_err!(e))?;

        let out_v: Vec<&String> = updated_outgoing.iter().collect();
        let out_bytes = serde_json::to_vec(&out_v).map_err(|e| storage_err!(e))?;

        let in_v: Vec<&String> = updated_incoming.iter().collect();
        let in_bytes = serde_json::to_vec(&in_v).map_err(|e| storage_err!(e))?;

        let write_txn = self.db.begin_write().map_err(|e| storage_err!(e))?;
        {
            let mut tbl = write_txn
                .open_table(RELATIONSHIPS)
                .map_err(|e| storage_err!(e))?;
            tbl.insert(relationship.id.as_str(), rel_bytes.as_slice())
                .map_err(|e| storage_err!(e))?;
        }
        {
            let mut tbl = write_txn
                .open_table(OUTGOING)
                .map_err(|e| storage_err!(e))?;
            tbl.insert(relationship.source_id.as_str(), out_bytes.as_slice())
                .map_err(|e| storage_err!(e))?;
        }
        {
            let mut tbl = write_txn
                .open_table(INCOMING)
                .map_err(|e| storage_err!(e))?;
            tbl.insert(relationship.target_id.as_str(), in_bytes.as_slice())
                .map_err(|e| storage_err!(e))?;
        }
        write_txn.commit().map_err(|e| storage_err!(e))?;
        Ok(())
    }

    /// Collect all keys from a table as owned `String`s.
    fn all_keys(&self, table_def: TableDefinition<&str, &[u8]>) -> Result<Vec<String>, GraphError> {
        let read_txn = self.db.begin_read().map_err(|e| storage_err!(e))?;
        let table = read_txn
            .open_table(table_def)
            .map_err(|e| storage_err!(e))?;
        let mut keys = Vec::new();
        for entry in table.iter().map_err(|e| storage_err!(e))? {
            let (k, _v) = entry.map_err(|e| storage_err!(e))?;
            keys.push(k.value().to_string());
        }
        Ok(keys)
    }
}

// ---------------------------------------------------------------------------
// GraphStore trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl GraphStore for RedbGraphStore {
    /// Add an entity to the graph.
    ///
    /// If an entity with the same id already exists it is overwritten (upsert
    /// semantics mirror [`InMemoryGraphStore`](super::memory::InMemoryGraphStore)).
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] on any database I/O failure.
    async fn add_entity(&mut self, entity: GraphEntity) -> Result<EntityId, GraphError> {
        let id = entity.id.clone();
        let is_new = self.load_entity(&id)?.is_none();
        self.persist_entity_atomic(&entity)?;
        if is_new {
            self.entity_count += 1;
        }
        Ok(id)
    }

    /// Add multiple entities to the graph.
    ///
    /// Stops and propagates the error on the first failure.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] on any database I/O failure.
    async fn add_entities(
        &mut self,
        entities: Vec<GraphEntity>,
    ) -> Result<Vec<EntityId>, GraphError> {
        let mut ids = Vec::with_capacity(entities.len());
        for entity in entities {
            let id = self.add_entity(entity).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Add a relationship to the graph.
    ///
    /// Both the source and the target entity must exist before calling this
    /// method; otherwise a [`GraphError::EntityNotFound`] is returned.
    ///
    /// # Errors
    ///
    /// - [`GraphError::EntityNotFound`] when either endpoint entity is missing.
    /// - [`GraphError::StorageError`] on any database I/O failure.
    async fn add_relationship(
        &mut self,
        relationship: GraphRelationship,
    ) -> Result<String, GraphError> {
        let rel_id = relationship.id.clone();
        let source_id = relationship.source_id.clone();
        let target_id = relationship.target_id.clone();

        // Verify both endpoints exist.
        if self.load_entity(&source_id)?.is_none() {
            return Err(GraphError::EntityNotFound(source_id));
        }
        if self.load_entity(&target_id)?.is_none() {
            return Err(GraphError::EntityNotFound(target_id));
        }

        // Read-modify the adjacency lists before the write transaction.
        let mut outgoing_set = self.read_string_set(OUTGOING, &source_id)?;
        outgoing_set.insert(rel_id.clone());

        let mut incoming_set = self.read_string_set(INCOMING, &target_id)?;
        incoming_set.insert(rel_id.clone());

        self.persist_relationship_atomic(&relationship, &outgoing_set, &incoming_set)?;
        self.rel_count += 1;
        Ok(rel_id)
    }

    /// Add multiple relationships to the graph.
    ///
    /// Stops and propagates the error on the first failure.
    ///
    /// # Errors
    ///
    /// - [`GraphError::EntityNotFound`] when an endpoint entity is missing.
    /// - [`GraphError::StorageError`] on any database I/O failure.
    async fn add_relationships(
        &mut self,
        relationships: Vec<GraphRelationship>,
    ) -> Result<Vec<String>, GraphError> {
        let mut ids = Vec::with_capacity(relationships.len());
        for rel in relationships {
            let id = self.add_relationship(rel).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Retrieve an entity by its id.
    ///
    /// Returns `Ok(None)` when no entity with the given id exists.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] on any database I/O failure.
    async fn get_entity(&self, id: &EntityId) -> Result<Option<GraphEntity>, GraphError> {
        self.load_entity(id.as_str())
    }

    /// Retrieve neighboring entities and the connecting relationships.
    ///
    /// The `direction` parameter controls which edges are followed:
    ///
    /// - [`Direction::Outgoing`] — only edges where `entity` is the source.
    /// - [`Direction::Incoming`] — only edges where `entity` is the target.
    /// - [`Direction::Both`] — all edges in either direction.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] on any database I/O failure.
    async fn get_neighbors(
        &self,
        id: &EntityId,
        direction: Direction,
    ) -> Result<Vec<(GraphRelationship, GraphEntity)>, GraphError> {
        let mut neighbors: Vec<(GraphRelationship, GraphEntity)> = Vec::new();

        match direction {
            Direction::Outgoing | Direction::Both => {
                let mut out =
                    self.collect_neighbors_for_direction(id, OUTGOING, |rel| &rel.target_id)?;
                neighbors.append(&mut out);
            }
            Direction::Incoming => {}
        }

        match direction {
            Direction::Incoming | Direction::Both => {
                let mut inc =
                    self.collect_neighbors_for_direction(id, INCOMING, |rel| &rel.source_id)?;
                neighbors.append(&mut inc);
            }
            Direction::Outgoing => {}
        }

        Ok(neighbors)
    }

    /// Find all entities whose type matches `entity_type`.
    ///
    /// Uses the persisted `TYPE_IDX` table for an efficient lookup without a
    /// full table scan.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] on any database I/O failure.
    async fn find_entities_by_type(
        &self,
        entity_type: &EntityType,
    ) -> Result<Vec<GraphEntity>, GraphError> {
        let type_key = entity_type.to_string();
        let ids = self.read_string_set(TYPE_IDX, &type_key)?;
        let mut result = Vec::new();
        for id in &ids {
            if let Some(entity) = self.load_entity(id)? {
                result.push(entity);
            }
        }
        Ok(result)
    }

    /// Find all entities whose name matches `name`.
    ///
    /// First tries an exact (case-insensitive) match via the `NAME_IDX` index.
    /// If no results are found, performs a full table scan comparing
    /// `entity.name.to_lowercase().contains(&name.to_lowercase())` or
    /// `name.to_lowercase().contains(&indexed_name)` to catch superstrings.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] on any database I/O failure.
    async fn find_entities_by_name(&self, name: &str) -> Result<Vec<GraphEntity>, GraphError> {
        let name_lower = name.to_lowercase();
        let mut results: Vec<GraphEntity> = Vec::new();

        // -- exact index lookup  -------------------------------------------
        let ids = self.read_string_set(NAME_IDX, &name_lower)?;
        for id in &ids {
            if let Some(entity) = self.load_entity(id)? {
                results.push(entity);
            }
        }

        // -- partial / substring scan if exact match yielded nothing  ------
        if results.is_empty() {
            let read_txn = self.db.begin_read().map_err(|e| storage_err!(e))?;
            let name_table = read_txn.open_table(NAME_IDX).map_err(|e| storage_err!(e))?;
            for entry in name_table.iter().map_err(|e| storage_err!(e))? {
                let (k_guard, v_guard) = entry.map_err(|e| storage_err!(e))?;
                let indexed_name = k_guard.value();
                if indexed_name.contains(&name_lower as &str) || name_lower.contains(indexed_name) {
                    let entity_ids: Vec<String> =
                        serde_json::from_slice(v_guard.value()).map_err(|e| storage_err!(e))?;
                    for eid in entity_ids {
                        if let Some(entity) = self.load_entity(&eid)? {
                            results.push(entity);
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Traverse the graph using breadth-first search starting from the entity
    /// ids listed in `query.start_entities`.
    ///
    /// The traversal mirrors the algorithm in
    /// [`traversal::bfs_traverse`](crate::layer4_graph::traversal::bfs_traverse)
    /// and respects all filter fields on [`GraphQuery`]:
    ///
    /// - `max_hops` — maximum edge depth.
    /// - `min_confidence` — minimum path confidence threshold.
    /// - `entity_filter` — optional whitelist of allowed entity types.
    /// - `relationship_filter` — optional whitelist of allowed relationship types.
    /// - `direction` — traversal direction.
    ///
    /// At most `MAX_TRAVERSE_PATHS` paths are returned, sorted by descending
    /// total confidence.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] on any database I/O failure.
    async fn traverse(&self, query: &GraphQuery) -> Result<Vec<GraphPath>, GraphError> {
        let mut paths: Vec<GraphPath> = Vec::new();

        for start_id in &query.start_entities {
            let Some(start_entity) = self.load_entity(start_id)? else {
                continue;
            };

            // Apply entity filter to the starting entity.
            if let Some(ref filter) = query.entity_filter
                && !filter.contains(&start_entity.entity_type)
            {
                continue;
            }

            // BFS queue: (current path, set of already-visited entity ids)
            let mut queue: VecDeque<(GraphPath, HashSet<EntityId>)> = VecDeque::new();

            let initial_path = GraphPath::from_entity(start_entity);
            let mut initial_visited: HashSet<EntityId> = HashSet::new();
            initial_visited.insert(start_id.clone());
            queue.push_back((initial_path, initial_visited));

            'bfs: while let Some((current_path, visited)) = queue.pop_front() {
                // Emit this path if it meets the confidence threshold.
                if current_path.total_confidence >= query.min_confidence {
                    paths.push(current_path.clone());
                    if paths.len() >= MAX_TRAVERSE_PATHS {
                        break 'bfs;
                    }
                }

                // Do not expand beyond max_hops.
                if current_path.len() >= query.max_hops {
                    continue;
                }

                // The frontier entity is the last one in the current path.
                let current_id = current_path.end().map(|e| e.id.clone()).unwrap_or_default();

                // Get neighbors according to query direction.
                let neighbors = self.get_neighbors(&current_id, query.direction).await?;

                for (relationship, neighbor) in neighbors {
                    // Cycle prevention.
                    if visited.contains(&neighbor.id) {
                        continue;
                    }

                    // Apply relationship filter.
                    if let Some(ref filter) = query.relationship_filter
                        && !filter.contains(&relationship.relationship_type)
                    {
                        continue;
                    }

                    // Apply entity filter to the neighbor.
                    if let Some(ref filter) = query.entity_filter
                        && !filter.contains(&neighbor.entity_type)
                    {
                        continue;
                    }

                    // Extend the path and check the confidence threshold.
                    let mut new_path = current_path.clone();
                    new_path.extend(relationship, neighbor.clone());

                    if new_path.total_confidence >= query.min_confidence {
                        let mut new_visited = visited.clone();
                        new_visited.insert(neighbor.id.clone());
                        queue.push_back((new_path, new_visited));
                    }
                }
            }

            if paths.len() >= MAX_TRAVERSE_PATHS {
                break;
            }
        }

        // Sort descending by total confidence.
        paths.sort_by(|a, b| {
            b.total_confidence
                .partial_cmp(&a.total_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(paths)
    }

    /// Return the total number of entities currently stored.
    async fn entity_count(&self) -> usize {
        self.entity_count
    }

    /// Return the total number of relationships currently stored.
    async fn relationship_count(&self) -> usize {
        self.rel_count
    }

    /// Remove all entities, relationships, and index data from the store.
    ///
    /// All six tables are cleared in a single write transaction and the
    /// in-memory counts are reset to zero.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::StorageError`] on any database I/O failure.
    async fn clear(&mut self) -> Result<(), GraphError> {
        // Collect all keys from every table before we start mutating.
        let entity_keys = self.all_keys(ENTITIES)?;
        let rel_keys = self.all_keys(RELATIONSHIPS)?;
        let out_keys = self.all_keys(OUTGOING)?;
        let inc_keys = self.all_keys(INCOMING)?;
        let name_keys = self.all_keys(NAME_IDX)?;
        let type_keys = self.all_keys(TYPE_IDX)?;

        // Delete everything in one write transaction.
        let write_txn = self.db.begin_write().map_err(|e| storage_err!(e))?;
        {
            let mut tbl = write_txn
                .open_table(ENTITIES)
                .map_err(|e| storage_err!(e))?;
            for k in &entity_keys {
                tbl.remove(k.as_str()).map_err(|e| storage_err!(e))?;
            }
        }
        {
            let mut tbl = write_txn
                .open_table(RELATIONSHIPS)
                .map_err(|e| storage_err!(e))?;
            for k in &rel_keys {
                tbl.remove(k.as_str()).map_err(|e| storage_err!(e))?;
            }
        }
        {
            let mut tbl = write_txn
                .open_table(OUTGOING)
                .map_err(|e| storage_err!(e))?;
            for k in &out_keys {
                tbl.remove(k.as_str()).map_err(|e| storage_err!(e))?;
            }
        }
        {
            let mut tbl = write_txn
                .open_table(INCOMING)
                .map_err(|e| storage_err!(e))?;
            for k in &inc_keys {
                tbl.remove(k.as_str()).map_err(|e| storage_err!(e))?;
            }
        }
        {
            let mut tbl = write_txn
                .open_table(NAME_IDX)
                .map_err(|e| storage_err!(e))?;
            for k in &name_keys {
                tbl.remove(k.as_str()).map_err(|e| storage_err!(e))?;
            }
        }
        {
            let mut tbl = write_txn
                .open_table(TYPE_IDX)
                .map_err(|e| storage_err!(e))?;
            for k in &type_keys {
                tbl.remove(k.as_str()).map_err(|e| storage_err!(e))?;
            }
        }
        write_txn.commit().map_err(|e| storage_err!(e))?;

        self.entity_count = 0;
        self.rel_count = 0;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer4_graph::types::{
        EntityType, GraphEntity, GraphRelationship, RelationshipType,
    };
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_store(dir: &TempDir) -> RedbGraphStore {
        RedbGraphStore::new(dir.path().join("graph.redb")).expect("store creation should succeed")
    }

    fn tech_entity(name: &str, id: &str) -> GraphEntity {
        GraphEntity::new(name, EntityType::Technology).with_id(id)
    }

    fn concept_entity(name: &str, id: &str) -> GraphEntity {
        GraphEntity::new(name, EntityType::Concept).with_id(id)
    }

    fn uses_rel(src: &str, tgt: &str) -> GraphRelationship {
        GraphRelationship::new(src, tgt, RelationshipType::Uses)
    }

    fn related_rel(src: &str, tgt: &str) -> GraphRelationship {
        GraphRelationship::new(src, tgt, RelationshipType::RelatedTo)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_redb_graph_store_add_and_get_entity() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        let entity = tech_entity("Rust", "rust");
        let id = store
            .add_entity(entity)
            .await
            .expect("add_entity should succeed");
        assert_eq!(id, "rust");

        let retrieved = store
            .get_entity(&id)
            .await
            .expect("get_entity should succeed");
        assert!(retrieved.is_some(), "entity should be found");
        let e = retrieved.expect("checked above");
        assert_eq!(e.name, "Rust");
        assert_eq!(e.entity_type, EntityType::Technology);
    }

    #[tokio::test]
    async fn test_redb_graph_store_add_entity_duplicate() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        // First insert.
        store
            .add_entity(tech_entity("Rust", "rust"))
            .await
            .expect("first add should succeed");
        assert_eq!(store.entity_count().await, 1);

        // Second insert with same id — should overwrite, count stays at 1.
        store
            .add_entity(tech_entity("Rust (updated)", "rust"))
            .await
            .expect("second add should succeed");
        assert_eq!(
            store.entity_count().await,
            1,
            "duplicate id must not increment count"
        );

        let retrieved = store
            .get_entity(&"rust".to_string())
            .await
            .expect("get should succeed")
            .expect("entity must exist");
        assert_eq!(retrieved.name, "Rust (updated)");
    }

    #[tokio::test]
    async fn test_redb_graph_store_add_relationship() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        store
            .add_entity(tech_entity("Rust", "rust"))
            .await
            .expect("add rust");
        store
            .add_entity(tech_entity("LLVM", "llvm"))
            .await
            .expect("add llvm");

        let rel_id = store
            .add_relationship(uses_rel("rust", "llvm"))
            .await
            .expect("add_relationship should succeed");
        assert!(!rel_id.is_empty(), "relationship id must not be empty");
        assert_eq!(store.relationship_count().await, 1);

        // Missing target entity must return EntityNotFound.
        let bad = GraphRelationship::new("rust", "nonexistent", RelationshipType::Uses);
        let err = store.add_relationship(bad).await;
        assert!(
            err.is_err(),
            "adding relationship with missing endpoint must fail"
        );
    }

    #[tokio::test]
    async fn test_redb_graph_store_get_neighbors_outgoing() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        store
            .add_entity(tech_entity("Rust", "rust"))
            .await
            .expect("rust");
        store
            .add_entity(tech_entity("LLVM", "llvm"))
            .await
            .expect("llvm");
        store
            .add_entity(tech_entity("Cargo", "cargo"))
            .await
            .expect("cargo");
        store
            .add_relationship(uses_rel("rust", "llvm"))
            .await
            .expect("r1");
        store
            .add_relationship(uses_rel("rust", "cargo"))
            .await
            .expect("r2");

        let neighbors = store
            .get_neighbors(&"rust".to_string(), Direction::Outgoing)
            .await
            .expect("get_neighbors outgoing");
        assert_eq!(neighbors.len(), 2, "rust has two outgoing neighbours");

        // The start entity itself should have no incoming edges.
        let incoming = store
            .get_neighbors(&"rust".to_string(), Direction::Incoming)
            .await
            .expect("get_neighbors incoming");
        assert!(incoming.is_empty(), "rust has no incoming edges");
    }

    #[tokio::test]
    async fn test_redb_graph_store_get_neighbors_incoming() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        store
            .add_entity(tech_entity("Rust", "rust"))
            .await
            .expect("rust");
        store
            .add_entity(tech_entity("LLVM", "llvm"))
            .await
            .expect("llvm");
        store
            .add_relationship(uses_rel("rust", "llvm"))
            .await
            .expect("r1");

        let incoming = store
            .get_neighbors(&"llvm".to_string(), Direction::Incoming)
            .await
            .expect("get_neighbors incoming llvm");
        assert_eq!(incoming.len(), 1, "llvm has one incoming edge from rust");
        assert_eq!(incoming[0].1.id, "rust");
    }

    #[tokio::test]
    async fn test_redb_graph_store_get_neighbors_both() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        // A -> B -> C  (B has 1 incoming, 1 outgoing)
        store.add_entity(tech_entity("A", "a")).await.expect("a");
        store.add_entity(tech_entity("B", "b")).await.expect("b");
        store.add_entity(tech_entity("C", "c")).await.expect("c");
        store
            .add_relationship(uses_rel("a", "b"))
            .await
            .expect("ab");
        store
            .add_relationship(uses_rel("b", "c"))
            .await
            .expect("bc");

        let both = store
            .get_neighbors(&"b".to_string(), Direction::Both)
            .await
            .expect("get_neighbors both");
        assert_eq!(both.len(), 2, "B has 1 outgoing + 1 incoming = 2 neighbors");
    }

    #[tokio::test]
    async fn test_redb_graph_store_find_by_type() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        store
            .add_entity(GraphEntity::new("Rust", EntityType::Technology))
            .await
            .expect("rust");
        store
            .add_entity(GraphEntity::new("Python", EntityType::Technology))
            .await
            .expect("python");
        store
            .add_entity(GraphEntity::new("Mozilla", EntityType::Organization))
            .await
            .expect("mozilla");

        let tech = store
            .find_entities_by_type(&EntityType::Technology)
            .await
            .expect("find tech");
        assert_eq!(tech.len(), 2, "should find 2 Technology entities");

        let orgs = store
            .find_entities_by_type(&EntityType::Organization)
            .await
            .expect("find org");
        assert_eq!(orgs.len(), 1, "should find 1 Organization entity");

        let persons = store
            .find_entities_by_type(&EntityType::Person)
            .await
            .expect("find person");
        assert!(persons.is_empty(), "should find no Person entities");
    }

    #[tokio::test]
    async fn test_redb_graph_store_find_by_name() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        store
            .add_entity(GraphEntity::new("Rust Language", EntityType::Technology))
            .await
            .expect("rust language");
        store
            .add_entity(GraphEntity::new("Rusty", EntityType::Person))
            .await
            .expect("rusty");

        // Exact match (case-insensitive).
        let exact = store
            .find_entities_by_name("rust language")
            .await
            .expect("find exact");
        assert_eq!(exact.len(), 1, "exact match should return 1 result");
        assert_eq!(exact[0].name, "Rust Language");

        // Partial / substring match when exact yields nothing.
        let partial = store
            .find_entities_by_name("rust")
            .await
            .expect("find partial");
        assert_eq!(
            partial.len(),
            2,
            "substring 'rust' should match both entities"
        );
    }

    #[tokio::test]
    async fn test_redb_graph_store_traverse_bfs() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        // Graph: A -> B -> C -> D
        //         \-> E
        store.add_entity(concept_entity("A", "a")).await.expect("a");
        store.add_entity(concept_entity("B", "b")).await.expect("b");
        store.add_entity(concept_entity("C", "c")).await.expect("c");
        store.add_entity(concept_entity("D", "d")).await.expect("d");
        store.add_entity(tech_entity("E", "e")).await.expect("e");
        store
            .add_relationship(related_rel("a", "b"))
            .await
            .expect("ab");
        store
            .add_relationship(related_rel("b", "c"))
            .await
            .expect("bc");
        store
            .add_relationship(related_rel("c", "d"))
            .await
            .expect("cd");
        store
            .add_relationship(uses_rel("a", "e"))
            .await
            .expect("ae");

        // 2-hop BFS from A.
        let query = GraphQuery::new(vec!["a".to_string()]).with_max_hops(2);
        let paths = store.traverse(&query).await.expect("traverse");
        assert!(!paths.is_empty(), "BFS should find at least one path");

        // All paths should start from A.
        for p in &paths {
            let start = p.start().expect("path must have start entity");
            assert_eq!(start.id, "a", "all paths start at A");
        }

        // 3-hop BFS should be able to reach D.
        let query3 = GraphQuery::new(vec!["a".to_string()]).with_max_hops(3);
        let paths3 = store.traverse(&query3).await.expect("traverse 3");
        let reached: std::collections::HashSet<&str> = paths3
            .iter()
            .filter_map(|p| p.end().map(|e| e.id.as_str()))
            .collect();
        assert!(
            reached.contains("d"),
            "3-hop BFS from A must be able to reach D"
        );
    }

    #[tokio::test]
    async fn test_redb_graph_store_entity_and_rel_counts() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        assert_eq!(store.entity_count().await, 0, "initially 0 entities");
        assert_eq!(
            store.relationship_count().await,
            0,
            "initially 0 relationships"
        );

        store
            .add_entity(concept_entity("A", "a"))
            .await
            .expect("add a");
        store
            .add_entity(concept_entity("B", "b"))
            .await
            .expect("add b");
        store
            .add_relationship(related_rel("a", "b"))
            .await
            .expect("add rel");

        assert_eq!(store.entity_count().await, 2, "should be 2 entities");
        assert_eq!(
            store.relationship_count().await,
            1,
            "should be 1 relationship"
        );
    }

    #[tokio::test]
    async fn test_redb_graph_store_clear() {
        let tmp = TempDir::new().expect("tempdir");
        let mut store = make_store(&tmp);

        store
            .add_entity(tech_entity("Rust", "rust"))
            .await
            .expect("rust");
        store
            .add_entity(tech_entity("LLVM", "llvm"))
            .await
            .expect("llvm");
        store
            .add_relationship(uses_rel("rust", "llvm"))
            .await
            .expect("rel");

        assert_eq!(store.entity_count().await, 2);
        assert_eq!(store.relationship_count().await, 1);

        store.clear().await.expect("clear should succeed");

        assert_eq!(
            store.entity_count().await,
            0,
            "entity count should be 0 after clear"
        );
        assert_eq!(
            store.relationship_count().await,
            0,
            "rel count should be 0 after clear"
        );

        // Store must be usable after clearing.
        store
            .add_entity(tech_entity("Fresh", "fresh"))
            .await
            .expect("add after clear");
        assert_eq!(store.entity_count().await, 1);
    }

    #[tokio::test]
    async fn test_redb_graph_store_persistence() {
        let tmp = TempDir::new().expect("tempdir");
        let db_path = tmp.path().join("persist.redb");

        // Phase 1: populate and drop.
        {
            let mut store = RedbGraphStore::new(&db_path).expect("first open");
            store
                .add_entity(tech_entity("Rust", "rust"))
                .await
                .expect("add rust");
            store
                .add_entity(tech_entity("LLVM", "llvm"))
                .await
                .expect("add llvm");
            store
                .add_relationship(uses_rel("rust", "llvm"))
                .await
                .expect("add rel");
            assert_eq!(store.entity_count().await, 2);
            assert_eq!(store.relationship_count().await, 1);
        } // `store` is dropped — file remains on disk.

        // Phase 2: reopen and verify counts.
        let store = RedbGraphStore::new(&db_path).expect("second open");
        assert_eq!(
            store.entity_count().await,
            2,
            "entity count must survive reopen"
        );
        assert_eq!(
            store.relationship_count().await,
            1,
            "relationship count must survive reopen"
        );

        let rust = store
            .get_entity(&"rust".to_string())
            .await
            .expect("get rust")
            .expect("rust must exist after reopen");
        assert_eq!(rust.name, "Rust");
        assert_eq!(rust.entity_type, EntityType::Technology);

        let neighbors = store
            .get_neighbors(&"rust".to_string(), Direction::Outgoing)
            .await
            .expect("neighbors after reopen");
        assert_eq!(neighbors.len(), 1, "rust → llvm edge must survive reopen");
        assert_eq!(neighbors[0].1.id, "llvm");
    }
}
