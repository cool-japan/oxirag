//! `CircuitBreakerRegistry` for managing multiple circuit breakers.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::breaker::CircuitBreaker;
use super::types::{CircuitBreakerConfig, CircuitBreakerStats};

/// Registry for managing multiple circuit breakers.
///
/// Useful when you have multiple external services that each need their own
/// circuit breaker with potentially different configurations.
pub struct CircuitBreakerRegistry {
    pub(super) breakers: RwLock<HashMap<String, Arc<CircuitBreaker>>>,
    pub(super) default_config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    /// Create a new registry with the given default configuration.
    #[must_use]
    pub fn new(default_config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: RwLock::new(HashMap::new()),
            default_config,
        }
    }

    /// Get or create a circuit breaker for a service.
    pub async fn get_or_create(&self, service_name: &str) -> Arc<CircuitBreaker> {
        // Try read lock first
        {
            let breakers = self.breakers.read().await;
            if let Some(breaker) = breakers.get(service_name) {
                return Arc::clone(breaker);
            }
        }

        // Need write lock to create
        let mut breakers = self.breakers.write().await;
        // Double-check in case another task created it
        if let Some(breaker) = breakers.get(service_name) {
            return Arc::clone(breaker);
        }

        let breaker = Arc::new(CircuitBreaker::new(self.default_config.clone()));
        breakers.insert(service_name.to_string(), Arc::clone(&breaker));
        breaker
    }

    /// Get a circuit breaker for a service if it exists.
    pub async fn get(&self, service_name: &str) -> Option<Arc<CircuitBreaker>> {
        let breakers = self.breakers.read().await;
        breakers.get(service_name).cloned()
    }

    /// Get all circuit breaker stats.
    pub async fn all_stats(&self) -> HashMap<String, CircuitBreakerStats> {
        let breakers = self.breakers.read().await;
        let mut stats = HashMap::new();
        for (name, breaker) in breakers.iter() {
            stats.insert(name.clone(), breaker.stats_async().await);
        }
        stats
    }

    /// Reset all circuit breakers.
    pub async fn reset_all(&self) {
        let breakers = self.breakers.read().await;
        for breaker in breakers.values() {
            breaker.reset().await;
        }
    }

    /// Remove a circuit breaker from the registry.
    pub async fn remove(&self, service_name: &str) -> Option<Arc<CircuitBreaker>> {
        let mut breakers = self.breakers.write().await;
        breakers.remove(service_name)
    }

    /// Get the number of registered circuit breakers.
    pub async fn len(&self) -> usize {
        let breakers = self.breakers.read().await;
        breakers.len()
    }

    /// Check if the registry is empty.
    pub async fn is_empty(&self) -> bool {
        let breakers = self.breakers.read().await;
        breakers.is_empty()
    }
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}
