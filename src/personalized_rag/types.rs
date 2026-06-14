//! Core data structures for personalized reranking.
//!
//! This module defines the [`UserProfile`] (explicit topic interests plus an
//! interaction history), the [`PersonalizedConfig`] tuning knobs, and the
//! [`PersonalizedError`] type surfaced by the checked reranking entry point.

use std::collections::HashMap;

use thiserror::Error;

/// A persistent model of an individual user's preferences.
///
/// A profile combines two complementary signals:
///
/// * **Explicit topic interests** — a map from a topic keyword to a non-negative
///   affinity weight. Higher weights express stronger interest.
/// * **Interaction history** — free text drawn from documents the user engaged
///   with (clicked, read, bookmarked). The history is later condensed into a
///   single lexical centroid used for content-based affinity.
///
/// The profile is intentionally storage-agnostic and deterministic: it holds no
/// timestamps and performs no I/O, so identical inputs always yield identical
/// reranking behaviour.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserProfile {
    /// Topic-interest weights keyed by a lowercase topic keyword.
    topics: HashMap<String, f32>,
    /// Raw interaction-history text snippets.
    history: Vec<String>,
}

impl UserProfile {
    /// Create a new, empty user profile.
    ///
    /// The profile starts with no topic interests and no interaction history;
    /// [`UserProfile::is_empty`] returns `true` until either is populated.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or overwrite) an explicit interest in `topic` with `weight`.
    ///
    /// The topic key is lowercased so lookups are case-insensitive. Negative
    /// weights are clamped to `0.0` because affinity is defined to be
    /// non-negative; supplying a negative weight therefore disables the topic.
    ///
    /// This is a chainable builder method, so several interests can be declared
    /// fluently when constructing a profile.
    #[must_use]
    pub fn with_interest(mut self, topic: &str, weight: f32) -> Self {
        self.topics.insert(topic.to_lowercase(), weight.max(0.0));
        self
    }

    /// Append a free-text snippet to the interaction history.
    ///
    /// Blank or whitespace-only snippets are ignored so they cannot dilute the
    /// history centroid. The text is stored verbatim; tokenization happens later
    /// when the centroid is computed.
    pub fn add_history(&mut self, text: &str) {
        if !text.trim().is_empty() {
            self.history.push(text.to_string());
        }
    }

    /// Return `true` when the profile carries no personalization signal.
    ///
    /// A profile is empty when it has neither topic interests nor any history
    /// entries; reranking against an empty profile is the identity transform.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.topics.is_empty() && self.history.is_empty()
    }

    /// Look up the interest weight for `topic`.
    ///
    /// The lookup is case-insensitive and returns `0.0` for any topic that was
    /// never declared.
    #[must_use]
    pub fn interest(&self, topic: &str) -> f32 {
        self.topics
            .get(&topic.to_lowercase())
            .copied()
            .unwrap_or(0.0)
    }

    /// Borrow the explicit topic-interest map.
    ///
    /// Keys are lowercase topic keywords; values are non-negative weights.
    #[must_use]
    pub fn topics(&self) -> &HashMap<String, f32> {
        &self.topics
    }

    /// Borrow the recorded interaction-history snippets.
    #[must_use]
    pub fn history(&self) -> &[String] {
        &self.history
    }
}

/// Tuning parameters for [`PersonalizedReranker`](crate::personalized_rag::PersonalizedReranker).
///
/// The reranker blends each result's original relevance with a personal
/// affinity score. `personalization_weight` controls how aggressively the
/// affinity overrides relevance, and `dim` sizes the lexical embedding used for
/// the content-based half of the affinity.
#[derive(Debug, Clone, PartialEq)]
pub struct PersonalizedConfig {
    /// Blend factor in `[0, 1]` applied to affinity versus relevance.
    ///
    /// The final score is
    /// `(1 - personalization_weight) * relevance + personalization_weight * affinity`.
    /// At `0.0` the original ranking is preserved exactly; at `1.0` the ranking
    /// is driven purely by personal affinity.
    pub personalization_weight: f32,
    /// Dimensionality of the deterministic lexical embedding.
    ///
    /// Larger values reduce hash collisions between distinct tokens at the cost
    /// of slightly more work per document.
    pub dim: usize,
}

impl Default for PersonalizedConfig {
    fn default() -> Self {
        Self {
            personalization_weight: 0.3,
            dim: 128,
        }
    }
}

impl PersonalizedConfig {
    /// Create a configuration with default values.
    ///
    /// Equivalent to [`PersonalizedConfig::default`]: a personalization weight
    /// of `0.3` and an embedding dimensionality of `128`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the personalization weight, clamped to `[0, 1]`.
    #[must_use]
    pub fn with_personalization_weight(mut self, weight: f32) -> Self {
        self.personalization_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the embedding dimensionality.
    ///
    /// A `dim` of `0` disables the content-based half of the affinity, leaving
    /// only the topic-interest signal.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }
}

/// Errors raised by personalized reranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PersonalizedError {
    /// The supplied result set was empty.
    #[error("results must not be empty")]
    EmptyResults,
}
