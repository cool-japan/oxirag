//! Statistics for the ANN (HNSW) index.

/// Statistics for the ANN index.
#[derive(Debug, Clone, Default)]
pub struct AnnStats {
    /// Number of nodes in the index.
    pub num_nodes: usize,
    /// Maximum level in the hierarchy.
    pub max_level: usize,
    /// Average number of connections per node.
    pub avg_connections: f32,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
}
