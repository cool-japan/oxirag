//! Cache invalidation strategies.
//!
//! This module provides various invalidation policies and dependency tracking
//! for cache entries, enabling sophisticated cache management strategies.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use super::types::{CacheKey, ContextFingerprint, KVCacheEntry};

/// Invalidation policy types.
#[derive(Debug, Clone, PartialEq)]
pub enum InvalidationPolicy {
    /// Time-to-live based invalidation.
    /// Entry is invalid after the specified duration from creation.
    Ttl(Duration),
    /// Maximum age based invalidation.
    /// Entry is invalid if created more than the specified duration ago.
    MaxAge(Duration),
    /// Maximum stale based invalidation.
    /// Entry is invalid if not accessed within the specified duration.
    MaxStale(Duration),
    /// Dependency-based invalidation.
    /// Entry is invalid when any of its dependencies are invalidated.
    DependencyBased,
    /// Combined policy - entry is invalid if any sub-policy triggers.
    Combined(Vec<InvalidationPolicy>),
    /// Never invalidate automatically (manual only).
    Never,
}

impl InvalidationPolicy {
    /// Create a TTL policy with the given seconds.
    #[must_use]
    pub fn ttl_secs(secs: u64) -> Self {
        Self::Ttl(Duration::from_secs(secs))
    }

    /// Create a max age policy with the given seconds.
    #[must_use]
    pub fn max_age_secs(secs: u64) -> Self {
        Self::MaxAge(Duration::from_secs(secs))
    }

    /// Create a max stale policy with the given seconds.
    #[must_use]
    pub fn max_stale_secs(secs: u64) -> Self {
        Self::MaxStale(Duration::from_secs(secs))
    }

    /// Create a combined policy.
    #[must_use]
    pub fn combined(policies: Vec<InvalidationPolicy>) -> Self {
        Self::Combined(policies)
    }

    /// Check if this policy is TTL-based.
    #[must_use]
    pub fn is_ttl_based(&self) -> bool {
        matches!(self, Self::Ttl(_))
    }

    /// Check if this policy is dependency-based.
    #[must_use]
    pub fn is_dependency_based(&self) -> bool {
        matches!(self, Self::DependencyBased)
    }

    /// Get the TTL duration if this is a TTL policy.
    #[must_use]
    pub fn ttl_duration(&self) -> Option<Duration> {
        match self {
            Self::Ttl(d) => Some(*d),
            _ => None,
        }
    }
}

impl Default for InvalidationPolicy {
    fn default() -> Self {
        Self::Ttl(Duration::from_secs(3600)) // 1 hour default
    }
}

/// Reason for cache invalidation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidationReason {
    /// Entry expired based on TTL.
    TtlExpired,
    /// Entry exceeded maximum age.
    MaxAgeExceeded,
    /// Entry became stale (not accessed recently).
    Stale,
    /// Dependency was invalidated.
    DependencyInvalidated(CacheKey),
    /// Version mismatch.
    VersionMismatch,
    /// Manual invalidation requested.
    Manual,
    /// Cache capacity exceeded.
    Evicted,
}

/// Entry in the dependency graph.
#[derive(Debug, Clone)]
struct DependencyEntry {
    /// Keys that this entry depends on.
    depends_on: HashSet<CacheKey>,
    /// Keys that depend on this entry.
    dependents: HashSet<CacheKey>,
    /// When this dependency was registered (for future use in stale dependency detection).
    #[allow(dead_code)]
    registered_at: Instant,
}

impl DependencyEntry {
    fn new() -> Self {
        Self {
            depends_on: HashSet::new(),
            dependents: HashSet::new(),
            registered_at: Instant::now(),
        }
    }
}

/// Manager for cache invalidation.
///
/// Handles invalidation policies, dependency tracking, and version management
/// for cache entries.
#[derive(Debug)]
pub struct InvalidationManager {
    /// The active invalidation policy.
    policy: InvalidationPolicy,
    /// Dependency graph for entries.
    dependencies: HashMap<CacheKey, DependencyEntry>,
    /// Version map for context IDs (for versioned invalidation).
    version_map: HashMap<String, u64>,
    /// Invalidation log for debugging.
    invalidation_log: Vec<(Instant, CacheKey, InvalidationReason)>,
    /// Maximum log entries to keep.
    max_log_entries: usize,
}

impl InvalidationManager {
    /// Create a new invalidation manager with the given policy.
    #[must_use]
    pub fn new(policy: InvalidationPolicy) -> Self {
        Self {
            policy,
            dependencies: HashMap::new(),
            version_map: HashMap::new(),
            invalidation_log: Vec::new(),
            max_log_entries: 1000,
        }
    }

    /// Create a new invalidation manager with default policy.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(InvalidationPolicy::default())
    }

    /// Get the current policy.
    #[must_use]
    pub fn policy(&self) -> &InvalidationPolicy {
        &self.policy
    }

    /// Set a new policy.
    pub fn set_policy(&mut self, policy: InvalidationPolicy) {
        self.policy = policy;
    }

    /// Check if an entry should be invalidated based on the current policy.
    #[must_use]
    pub fn should_invalidate(&self, entry: &KVCacheEntry) -> bool {
        self.check_invalidation(entry, &self.policy)
    }

    /// Check if an entry should be invalidated with a specific policy.
    #[must_use]
    pub fn check_invalidation(&self, entry: &KVCacheEntry, policy: &InvalidationPolicy) -> bool {
        match policy {
            InvalidationPolicy::Ttl(ttl) => entry.created_at.elapsed() >= *ttl,
            InvalidationPolicy::MaxAge(max_age) => entry.age() >= *max_age,
            InvalidationPolicy::MaxStale(max_stale) => entry.time_since_access() >= *max_stale,
            InvalidationPolicy::DependencyBased => {
                // Check if any dependency has been invalidated
                if let Some(dep_entry) = self.dependencies.get(&entry.key) {
                    for dep_key in &dep_entry.depends_on {
                        // If the dependency doesn't exist in our tracking anymore,
                        // it may have been invalidated
                        if !self.dependencies.contains_key(dep_key) {
                            return true;
                        }
                    }
                }
                false
            }
            InvalidationPolicy::Combined(policies) => {
                policies.iter().any(|p| self.check_invalidation(entry, p))
            }
            InvalidationPolicy::Never => false,
        }
    }

    /// Get the reason for invalidation if the entry should be invalidated.
    #[must_use]
    pub fn get_invalidation_reason(&self, entry: &KVCacheEntry) -> Option<InvalidationReason> {
        self.get_invalidation_reason_for_policy(entry, &self.policy)
    }

    /// Get the reason for invalidation with a specific policy.
    #[must_use]
    pub fn get_invalidation_reason_for_policy(
        &self,
        entry: &KVCacheEntry,
        policy: &InvalidationPolicy,
    ) -> Option<InvalidationReason> {
        match policy {
            InvalidationPolicy::Ttl(ttl) => {
                if entry.created_at.elapsed() >= *ttl {
                    Some(InvalidationReason::TtlExpired)
                } else {
                    None
                }
            }
            InvalidationPolicy::MaxAge(max_age) => {
                if entry.age() >= *max_age {
                    Some(InvalidationReason::MaxAgeExceeded)
                } else {
                    None
                }
            }
            InvalidationPolicy::MaxStale(max_stale) => {
                if entry.time_since_access() >= *max_stale {
                    Some(InvalidationReason::Stale)
                } else {
                    None
                }
            }
            InvalidationPolicy::DependencyBased => {
                if let Some(dep_entry) = self.dependencies.get(&entry.key) {
                    for dep_key in &dep_entry.depends_on {
                        if !self.dependencies.contains_key(dep_key) {
                            return Some(InvalidationReason::DependencyInvalidated(
                                dep_key.clone(),
                            ));
                        }
                    }
                }
                None
            }
            InvalidationPolicy::Combined(policies) => {
                for p in policies {
                    if let Some(reason) = self.get_invalidation_reason_for_policy(entry, p) {
                        return Some(reason);
                    }
                }
                None
            }
            InvalidationPolicy::Never => None,
        }
    }

    /// Register a dependency between cache entries.
    ///
    /// When `depends_on` is invalidated, `key` will also be invalidated.
    pub fn register_dependency(&mut self, key: CacheKey, depends_on: CacheKey) {
        // Add forward dependency (key depends on depends_on)
        self.dependencies
            .entry(key.clone())
            .or_insert_with(DependencyEntry::new)
            .depends_on
            .insert(depends_on.clone());

        // Add reverse dependency (depends_on has key as dependent)
        self.dependencies
            .entry(depends_on)
            .or_insert_with(DependencyEntry::new)
            .dependents
            .insert(key);
    }

    /// Remove a dependency registration.
    pub fn remove_dependency(&mut self, key: &CacheKey, depends_on: &CacheKey) {
        if let Some(entry) = self.dependencies.get_mut(key) {
            entry.depends_on.remove(depends_on);
        }
        if let Some(entry) = self.dependencies.get_mut(depends_on) {
            entry.dependents.remove(key);
        }
    }

    /// Invalidate an entry and get all dependent entries that should also be invalidated.
    ///
    /// Returns a list of cache keys that should be invalidated as a result.
    pub fn invalidate_dependents(&mut self, key: &CacheKey) -> Vec<CacheKey> {
        let mut to_invalidate = Vec::new();
        let mut visited = HashSet::new();
        self.collect_dependents(key, &mut to_invalidate, &mut visited);

        // Log invalidations
        for invalidated_key in &to_invalidate {
            self.log_invalidation(
                invalidated_key.clone(),
                InvalidationReason::DependencyInvalidated(key.clone()),
            );
        }

        // Remove the invalidated entries from dependency tracking
        for invalidated_key in &to_invalidate {
            self.dependencies.remove(invalidated_key);
        }
        self.dependencies.remove(key);

        to_invalidate
    }

    /// Recursively collect all dependents of a key.
    fn collect_dependents(
        &self,
        key: &CacheKey,
        result: &mut Vec<CacheKey>,
        visited: &mut HashSet<CacheKey>,
    ) {
        if visited.contains(key) {
            return;
        }
        visited.insert(key.clone());

        if let Some(entry) = self.dependencies.get(key) {
            for dependent in &entry.dependents {
                result.push(dependent.clone());
                self.collect_dependents(dependent, result, visited);
            }
        }
    }

    /// Update the version for a context ID.
    ///
    /// This can be used to invalidate all entries associated with a particular
    /// context when the context changes.
    pub fn update_version(&mut self, context_id: &str) {
        let version = self.version_map.entry(context_id.to_string()).or_insert(0);
        *version += 1;
    }

    /// Get the current version for a context ID.
    #[must_use]
    pub fn get_version(&self, context_id: &str) -> u64 {
        self.version_map.get(context_id).copied().unwrap_or(0)
    }

    /// Check if an entry's version matches the current version.
    #[must_use]
    pub fn check_version(&self, context_id: &str, entry_version: u64) -> bool {
        self.get_version(context_id) == entry_version
    }

    /// Log an invalidation event.
    fn log_invalidation(&mut self, key: CacheKey, reason: InvalidationReason) {
        self.invalidation_log.push((Instant::now(), key, reason));

        // Trim log if too large
        if self.invalidation_log.len() > self.max_log_entries {
            let drain_count = self.invalidation_log.len() - self.max_log_entries;
            self.invalidation_log.drain(..drain_count);
        }
    }

    /// Manually invalidate an entry.
    pub fn invalidate(&mut self, key: &CacheKey) -> Vec<CacheKey> {
        self.log_invalidation(key.clone(), InvalidationReason::Manual);
        self.invalidate_dependents(key)
    }

    /// Get recent invalidation events.
    #[must_use]
    pub fn recent_invalidations(&self, count: usize) -> Vec<(&CacheKey, &InvalidationReason)> {
        self.invalidation_log
            .iter()
            .rev()
            .take(count)
            .map(|(_, key, reason)| (key, reason))
            .collect()
    }

    /// Get all keys that depend on the given key.
    #[must_use]
    pub fn get_dependents(&self, key: &CacheKey) -> Vec<CacheKey> {
        self.dependencies
            .get(key)
            .map(|entry| entry.dependents.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all keys that the given key depends on.
    #[must_use]
    pub fn get_dependencies(&self, key: &CacheKey) -> Vec<CacheKey> {
        self.dependencies
            .get(key)
            .map(|entry| entry.depends_on.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Clear all dependency tracking.
    pub fn clear_dependencies(&mut self) {
        self.dependencies.clear();
    }

    /// Clear the invalidation log.
    pub fn clear_log(&mut self) {
        self.invalidation_log.clear();
    }

    /// Get statistics about the dependency graph.
    #[must_use]
    pub fn dependency_stats(&self) -> DependencyStats {
        let total_entries = self.dependencies.len();
        let total_dependencies: usize =
            self.dependencies.values().map(|e| e.depends_on.len()).sum();
        let total_dependents: usize = self.dependencies.values().map(|e| e.dependents.len()).sum();
        let max_depth = self.calculate_max_depth();

        DependencyStats {
            total_entries,
            total_dependencies,
            total_dependents,
            max_depth,
            version_count: self.version_map.len(),
        }
    }

    /// Calculate the maximum depth of the dependency graph.
    fn calculate_max_depth(&self) -> usize {
        let mut max_depth = 0;
        let mut visited = HashSet::new();

        for key in self.dependencies.keys() {
            let depth = self.depth_from(key, &mut visited);
            max_depth = max_depth.max(depth);
        }

        max_depth
    }

    /// Calculate depth from a specific node.
    fn depth_from(&self, key: &CacheKey, visited: &mut HashSet<CacheKey>) -> usize {
        if visited.contains(key) {
            return 0;
        }
        visited.insert(key.clone());

        let mut max_child_depth = 0;
        if let Some(entry) = self.dependencies.get(key) {
            for dep in &entry.depends_on {
                let child_depth = self.depth_from(dep, visited);
                max_child_depth = max_child_depth.max(child_depth);
            }
        }

        visited.remove(key);
        max_child_depth + 1
    }
}

impl Default for InvalidationManager {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Statistics about the dependency graph.
#[derive(Debug, Clone, Default)]
pub struct DependencyStats {
    /// Total number of entries in the dependency graph.
    pub total_entries: usize,
    /// Total number of dependency relationships.
    pub total_dependencies: usize,
    /// Total number of dependent relationships.
    pub total_dependents: usize,
    /// Maximum depth of the dependency chain.
    pub max_depth: usize,
    /// Number of versioned contexts.
    pub version_count: usize,
}

/// Builder for invalidation rules.
#[derive(Debug, Default)]
pub struct InvalidationRuleBuilder {
    policies: Vec<InvalidationPolicy>,
}

impl InvalidationRuleBuilder {
    /// Create a new rule builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a TTL policy.
    #[must_use]
    pub fn with_ttl(mut self, duration: Duration) -> Self {
        self.policies.push(InvalidationPolicy::Ttl(duration));
        self
    }

    /// Add a TTL policy in seconds.
    #[must_use]
    pub fn with_ttl_secs(self, secs: u64) -> Self {
        self.with_ttl(Duration::from_secs(secs))
    }

    /// Add a max age policy.
    #[must_use]
    pub fn with_max_age(mut self, duration: Duration) -> Self {
        self.policies.push(InvalidationPolicy::MaxAge(duration));
        self
    }

    /// Add a max age policy in seconds.
    #[must_use]
    pub fn with_max_age_secs(self, secs: u64) -> Self {
        self.with_max_age(Duration::from_secs(secs))
    }

    /// Add a max stale policy.
    #[must_use]
    pub fn with_max_stale(mut self, duration: Duration) -> Self {
        self.policies.push(InvalidationPolicy::MaxStale(duration));
        self
    }

    /// Add a max stale policy in seconds.
    #[must_use]
    pub fn with_max_stale_secs(self, secs: u64) -> Self {
        self.with_max_stale(Duration::from_secs(secs))
    }

    /// Add dependency-based invalidation.
    #[must_use]
    pub fn with_dependency_tracking(mut self) -> Self {
        self.policies.push(InvalidationPolicy::DependencyBased);
        self
    }

    /// Build the final policy.
    ///
    /// # Panics
    ///
    /// This function should not panic as all cases are handled.
    #[must_use]
    pub fn build(self) -> InvalidationPolicy {
        match self.policies.len() {
            0 => InvalidationPolicy::Never,
            1 => self.policies.into_iter().next().expect("len is 1"),
            _ => InvalidationPolicy::Combined(self.policies),
        }
    }
}

/// Validator for checking cache entry validity.
pub struct CacheValidator {
    manager: InvalidationManager,
}

impl CacheValidator {
    /// Create a new validator with the given manager.
    #[must_use]
    pub fn new(manager: InvalidationManager) -> Self {
        Self { manager }
    }

    /// Check if an entry is valid.
    #[must_use]
    pub fn is_valid(&self, entry: &KVCacheEntry) -> bool {
        !self.manager.should_invalidate(entry)
    }

    /// Check if an entry is valid with a specific version.
    #[must_use]
    pub fn is_valid_with_version(
        &self,
        entry: &KVCacheEntry,
        context_id: &str,
        version: u64,
    ) -> bool {
        self.is_valid(entry) && self.manager.check_version(context_id, version)
    }

    /// Validate an entry and return the reason if invalid.
    ///
    /// # Errors
    ///
    /// Returns the invalidation reason if the entry is invalid.
    pub fn validate(&self, entry: &KVCacheEntry) -> Result<(), InvalidationReason> {
        match self.manager.get_invalidation_reason(entry) {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    /// Get a reference to the underlying manager.
    #[must_use]
    pub fn manager(&self) -> &InvalidationManager {
        &self.manager
    }

    /// Get a mutable reference to the underlying manager.
    pub fn manager_mut(&mut self) -> &mut InvalidationManager {
        &mut self.manager
    }
}

/// Context for fingerprint-based invalidation.
#[derive(Debug, Clone)]
pub struct FingerprintInvalidationContext {
    /// Fingerprint hash to key mapping.
    fingerprint_to_key: HashMap<u64, CacheKey>,
    /// Key to fingerprint mapping.
    key_to_fingerprint: HashMap<CacheKey, u64>,
}

impl FingerprintInvalidationContext {
    /// Create a new fingerprint invalidation context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fingerprint_to_key: HashMap::new(),
            key_to_fingerprint: HashMap::new(),
        }
    }

    /// Register a fingerprint-key association.
    pub fn register(&mut self, fingerprint: &ContextFingerprint, key: CacheKey) {
        self.fingerprint_to_key
            .insert(fingerprint.hash, key.clone());
        self.key_to_fingerprint.insert(key, fingerprint.hash);
    }

    /// Unregister by key.
    pub fn unregister_by_key(&mut self, key: &CacheKey) {
        if let Some(hash) = self.key_to_fingerprint.remove(key) {
            self.fingerprint_to_key.remove(&hash);
        }
    }

    /// Unregister by fingerprint.
    pub fn unregister_by_fingerprint(&mut self, fingerprint: &ContextFingerprint) {
        if let Some(key) = self.fingerprint_to_key.remove(&fingerprint.hash) {
            self.key_to_fingerprint.remove(&key);
        }
    }

    /// Get key by fingerprint.
    #[must_use]
    pub fn get_key(&self, fingerprint: &ContextFingerprint) -> Option<&CacheKey> {
        self.fingerprint_to_key.get(&fingerprint.hash)
    }

    /// Get fingerprint hash by key.
    #[must_use]
    pub fn get_fingerprint_hash(&self, key: &CacheKey) -> Option<u64> {
        self.key_to_fingerprint.get(key).copied()
    }

    /// Clear all associations.
    pub fn clear(&mut self) {
        self.fingerprint_to_key.clear();
        self.key_to_fingerprint.clear();
    }

    /// Get the number of registered associations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fingerprint_to_key.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fingerprint_to_key.is_empty()
    }
}

impl Default for FingerprintInvalidationContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::cast_sign_loss)]
mod tests {
    use super::*;

    fn create_test_entry(id: &str, hash: u64) -> KVCacheEntry {
        let fp = ContextFingerprint::new(hash, 100, format!("test {id}"));
        KVCacheEntry::new(id, fp, vec![0.0; 10], 100)
    }

    #[test]
    fn test_invalidation_policy_default() {
        let policy = InvalidationPolicy::default();
        assert!(matches!(policy, InvalidationPolicy::Ttl(_)));
    }

    #[test]
    fn test_invalidation_policy_ttl_secs() {
        let policy = InvalidationPolicy::ttl_secs(300);
        assert_eq!(policy.ttl_duration(), Some(Duration::from_secs(300)));
        assert!(policy.is_ttl_based());
    }

    #[test]
    fn test_invalidation_policy_is_dependency_based() {
        let policy = InvalidationPolicy::DependencyBased;
        assert!(policy.is_dependency_based());
        assert!(!policy.is_ttl_based());
    }

    #[test]
    fn test_invalidation_policy_combined() {
        let policy = InvalidationPolicy::combined(vec![
            InvalidationPolicy::ttl_secs(300),
            InvalidationPolicy::max_stale_secs(60),
        ]);
        assert!(matches!(policy, InvalidationPolicy::Combined(_)));
    }

    #[test]
    fn test_invalidation_manager_new() {
        let manager = InvalidationManager::new(InvalidationPolicy::ttl_secs(600));
        assert!(matches!(manager.policy(), InvalidationPolicy::Ttl(_)));
    }

    #[test]
    fn test_invalidation_manager_should_invalidate_ttl() {
        let manager = InvalidationManager::new(InvalidationPolicy::Ttl(Duration::from_secs(0)));
        let entry = create_test_entry("test1", 12345);

        std::thread::sleep(Duration::from_millis(1));
        assert!(manager.should_invalidate(&entry));
    }

    #[test]
    fn test_invalidation_manager_should_not_invalidate_ttl() {
        let manager = InvalidationManager::new(InvalidationPolicy::Ttl(Duration::from_secs(3600)));
        let entry = create_test_entry("test1", 12345);

        assert!(!manager.should_invalidate(&entry));
    }

    #[test]
    fn test_invalidation_manager_max_stale() {
        let manager =
            InvalidationManager::new(InvalidationPolicy::MaxStale(Duration::from_secs(0)));
        let entry = create_test_entry("test1", 12345);

        std::thread::sleep(Duration::from_millis(1));
        assert!(manager.should_invalidate(&entry));
    }

    #[test]
    fn test_invalidation_manager_never() {
        let manager = InvalidationManager::new(InvalidationPolicy::Never);
        let entry = create_test_entry("test1", 12345);

        assert!(!manager.should_invalidate(&entry));
    }

    #[test]
    fn test_invalidation_manager_get_reason_ttl() {
        let manager = InvalidationManager::new(InvalidationPolicy::Ttl(Duration::from_secs(0)));
        let entry = create_test_entry("test1", 12345);

        std::thread::sleep(Duration::from_millis(1));
        let reason = manager.get_invalidation_reason(&entry);
        assert_eq!(reason, Some(InvalidationReason::TtlExpired));
    }

    #[test]
    fn test_invalidation_manager_register_dependency() {
        let mut manager = InvalidationManager::with_defaults();

        manager.register_dependency("child".to_string(), "parent".to_string());

        let deps = manager.get_dependencies(&"child".to_string());
        assert!(deps.contains(&"parent".to_string()));

        let dependents = manager.get_dependents(&"parent".to_string());
        assert!(dependents.contains(&"child".to_string()));
    }

    #[test]
    fn test_invalidation_manager_remove_dependency() {
        let mut manager = InvalidationManager::with_defaults();

        manager.register_dependency("child".to_string(), "parent".to_string());
        manager.remove_dependency(&"child".to_string(), &"parent".to_string());

        let deps = manager.get_dependencies(&"child".to_string());
        assert!(!deps.contains(&"parent".to_string()));
    }

    #[test]
    fn test_invalidation_manager_invalidate_dependents() {
        let mut manager = InvalidationManager::with_defaults();

        manager.register_dependency("child1".to_string(), "parent".to_string());
        manager.register_dependency("child2".to_string(), "parent".to_string());
        manager.register_dependency("grandchild".to_string(), "child1".to_string());

        let invalidated = manager.invalidate_dependents(&"parent".to_string());

        assert!(invalidated.contains(&"child1".to_string()));
        assert!(invalidated.contains(&"child2".to_string()));
        assert!(invalidated.contains(&"grandchild".to_string()));
    }

    #[test]
    fn test_invalidation_manager_version() {
        let mut manager = InvalidationManager::with_defaults();

        assert_eq!(manager.get_version("ctx1"), 0);

        manager.update_version("ctx1");
        assert_eq!(manager.get_version("ctx1"), 1);

        manager.update_version("ctx1");
        assert_eq!(manager.get_version("ctx1"), 2);
    }

    #[test]
    fn test_invalidation_manager_check_version() {
        let mut manager = InvalidationManager::with_defaults();

        manager.update_version("ctx1");

        assert!(manager.check_version("ctx1", 1));
        assert!(!manager.check_version("ctx1", 0));
        assert!(!manager.check_version("ctx1", 2));
    }

    #[test]
    fn test_invalidation_manager_manual_invalidate() {
        let mut manager = InvalidationManager::with_defaults();

        manager.register_dependency("child".to_string(), "parent".to_string());
        let invalidated = manager.invalidate(&"parent".to_string());

        assert!(invalidated.contains(&"child".to_string()));
    }

    #[test]
    fn test_invalidation_manager_recent_invalidations() {
        let mut manager = InvalidationManager::with_defaults();

        manager.register_dependency("child".to_string(), "parent".to_string());
        manager.invalidate(&"parent".to_string());

        let recent = manager.recent_invalidations(10);
        assert!(!recent.is_empty());
    }

    #[test]
    fn test_invalidation_manager_dependency_stats() {
        let mut manager = InvalidationManager::with_defaults();

        manager.register_dependency("child1".to_string(), "parent".to_string());
        manager.register_dependency("child2".to_string(), "parent".to_string());
        manager.update_version("ctx1");

        let stats = manager.dependency_stats();
        assert_eq!(stats.total_entries, 3); // parent, child1, child2
        assert_eq!(stats.version_count, 1);
    }

    #[test]
    fn test_invalidation_manager_clear_dependencies() {
        let mut manager = InvalidationManager::with_defaults();

        manager.register_dependency("child".to_string(), "parent".to_string());
        manager.clear_dependencies();

        assert!(manager.get_dependencies(&"child".to_string()).is_empty());
    }

    #[test]
    fn test_invalidation_rule_builder() {
        let policy = InvalidationRuleBuilder::new()
            .with_ttl_secs(300)
            .with_max_stale_secs(60)
            .with_dependency_tracking()
            .build();

        assert!(matches!(policy, InvalidationPolicy::Combined(_)));
    }

    #[test]
    fn test_invalidation_rule_builder_single() {
        let policy = InvalidationRuleBuilder::new().with_ttl_secs(300).build();

        assert!(matches!(policy, InvalidationPolicy::Ttl(_)));
    }

    #[test]
    fn test_invalidation_rule_builder_empty() {
        let policy = InvalidationRuleBuilder::new().build();
        assert!(matches!(policy, InvalidationPolicy::Never));
    }

    #[test]
    fn test_cache_validator_is_valid() {
        let manager = InvalidationManager::new(InvalidationPolicy::ttl_secs(3600));
        let validator = CacheValidator::new(manager);
        let entry = create_test_entry("test1", 12345);

        assert!(validator.is_valid(&entry));
    }

    #[test]
    fn test_cache_validator_is_invalid() {
        let manager = InvalidationManager::new(InvalidationPolicy::Ttl(Duration::from_secs(0)));
        let validator = CacheValidator::new(manager);
        let entry = create_test_entry("test1", 12345);

        std::thread::sleep(Duration::from_millis(1));
        assert!(!validator.is_valid(&entry));
    }

    #[test]
    fn test_cache_validator_validate() {
        let manager = InvalidationManager::new(InvalidationPolicy::Ttl(Duration::from_secs(0)));
        let validator = CacheValidator::new(manager);
        let entry = create_test_entry("test1", 12345);

        std::thread::sleep(Duration::from_millis(1));
        let result = validator.validate(&entry);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InvalidationReason::TtlExpired);
    }

    #[test]
    fn test_cache_validator_with_version() {
        let mut manager = InvalidationManager::new(InvalidationPolicy::ttl_secs(3600));
        manager.update_version("ctx1");
        let validator = CacheValidator::new(manager);
        let entry = create_test_entry("test1", 12345);

        assert!(validator.is_valid_with_version(&entry, "ctx1", 1));
        assert!(!validator.is_valid_with_version(&entry, "ctx1", 0));
    }

    #[test]
    fn test_fingerprint_invalidation_context_new() {
        let ctx = FingerprintInvalidationContext::new();
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_fingerprint_invalidation_context_register() {
        let mut ctx = FingerprintInvalidationContext::new();
        let fp = ContextFingerprint::new(12345, 100, "test");

        ctx.register(&fp, "key1".to_string());

        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx.get_key(&fp), Some(&"key1".to_string()));
        assert_eq!(ctx.get_fingerprint_hash(&"key1".to_string()), Some(12345));
    }

    #[test]
    fn test_fingerprint_invalidation_context_unregister_by_key() {
        let mut ctx = FingerprintInvalidationContext::new();
        let fp = ContextFingerprint::new(12345, 100, "test");

        ctx.register(&fp, "key1".to_string());
        ctx.unregister_by_key(&"key1".to_string());

        assert!(ctx.is_empty());
        assert!(ctx.get_key(&fp).is_none());
    }

    #[test]
    fn test_fingerprint_invalidation_context_unregister_by_fingerprint() {
        let mut ctx = FingerprintInvalidationContext::new();
        let fp = ContextFingerprint::new(12345, 100, "test");

        ctx.register(&fp, "key1".to_string());
        ctx.unregister_by_fingerprint(&fp);

        assert!(ctx.is_empty());
    }

    #[test]
    fn test_fingerprint_invalidation_context_clear() {
        let mut ctx = FingerprintInvalidationContext::new();

        for i in 0..5 {
            let fp = ContextFingerprint::new(i as u64, 100, format!("test{i}"));
            ctx.register(&fp, format!("key{i}"));
        }

        assert_eq!(ctx.len(), 5);

        ctx.clear();
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_dependency_stats_default() {
        let stats = DependencyStats::default();
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.max_depth, 0);
    }

    #[test]
    fn test_invalidation_manager_combined_policy() {
        let policy = InvalidationPolicy::Combined(vec![
            InvalidationPolicy::Ttl(Duration::from_secs(3600)),
            InvalidationPolicy::MaxStale(Duration::from_secs(0)),
        ]);
        let manager = InvalidationManager::new(policy);
        let entry = create_test_entry("test1", 12345);

        std::thread::sleep(Duration::from_millis(1));
        // Should be invalid due to max stale
        assert!(manager.should_invalidate(&entry));
    }
}
